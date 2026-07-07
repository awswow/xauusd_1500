"""
target_r sweep for XAUUSD M15 pullback strategy.

Sweeps target_r values [1.0, 1.5, 2.0, 2.5, 3.0] on full 4-year period,
reporting trades / win% / total R / max DD / Calmar by year and in aggregate.

Current baseline: target_r=2.0, +126.90R over 4 years.

Usage:
  python scripts/target_sweep.py --config configs/live_demo_bt.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import yaml

from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec
from backtest_demo_executor_live_like import parse_dt, build_weekly_regime
from walk_forward import precompute, fast_backtest, _stats

TARGET_RS = [1.0, 1.5, 2.0, 2.5, 3.0]

YEARS = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-19"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/derived/runs/target_sweep")
    args = ap.parse_args()

    cfg      = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol   = cfg["symbol"]
    tf_name  = cfg["timeframe"]
    tf       = {"M15": mt5.TIMEFRAME_M15, "M1": mt5.TIMEFRAME_M1,
                "H1": mt5.TIMEFRAME_H1}[tf_name]
    ex_cfg   = cfg.get("execution", {}) or {}
    risk_cfg = cfg.get("risk",      {}) or {}
    rf_cfg   = cfg.get("risk_filters", {}) or {}
    regime_cfg = cfg.get("regime_filter", {}) or {}
    sp       = cfg["strategy"]["params"]

    fixed_exec = {
        "cooldown_bars":        int(risk_cfg.get("cooldown_bars", 16)),
        "daily_loss_limit_usd": float(ex_cfg.get("daily_loss_limit_usd", 45.0)),
        "max_trades_per_day":   int(ex_cfg.get("max_trades_per_day", 4)),
        "risk_usd_per_trade":   float(ex_cfg.get("risk_usd_per_trade", 15.0)),
        "pending_expiry_bars":  int(ex_cfg.get("pending_expiry_bars", 4)),
        "monthly_loss_limit_r": float(risk_cfg.get("monthly_loss_limit_r", 0.0) or 0.0),
    }
    max_sp = int(ex_cfg.get("max_spread_points", 45))

    base_params = dict(
        adx_min      = float(sp.get("adx_min",       24.3)),
        sma_len      = int(sp.get("sma_len",           60)),
        pullback_atr = float(sp.get("pullback_atr",  0.30)),
        stop_atr     = float(sp.get("stop_atr",      1.60)),
    )

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    spec  = get_symbol_spec(symbol)
    point = float(spec.point)

    print(f"\ntarget_r sweep — {symbol} {tf_name}")
    print(f"Base params: adx={base_params['adx_min']}  sma={base_params['sma_len']}  "
          f"pa={base_params['pullback_atr']}  sa={base_params['stop_atr']}")
    print(f"Testing target_r: {TARGET_RS}\n")

    # Fetch and precompute each year's bars once, then sweep all target_r values
    yearly_pcs: list[tuple[str, dict]] = []
    for label, yr_start, yr_end in YEARS:
        start_dt = parse_dt(yr_start)
        end_dt   = parse_dt(yr_end)
        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            print(f"  {label}: no data, skipping")
            continue

        regime_dict = None
        if regime_cfg.get("enabled", False):
            sma_w  = int(regime_cfg.get("sma_weeks", 40))
            slope  = bool(regime_cfg.get("require_slope", True))
            lb_wks = int(regime_cfg.get("slope_lookback_weeks", 4))
            regime_dict = build_weekly_regime(symbol, start_dt, end_dt, sma_w,
                                              require_slope=slope,
                                              slope_lookback_weeks=lb_wks)

        pc = precompute(rates, sma_lens=[base_params["sma_len"]],
                        regime_dict=regime_dict,
                        max_spread_points=max_sp,
                        gate_cfg=rf_cfg, point=point)
        yearly_pcs.append((label, pc))

    sep = "=" * 90
    print(sep)
    hdr = f"  {'target_r':>8}  {'Year':<6}  {'Trades':>6}  {'Win%':>5}  "
    hdr += f"{'Total R':>8}  {'Max DD':>7}  {'Calmar':>7}  {'$/trade':>7}"
    print(hdr)
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

    all_results: list[dict] = []

    for tr in TARGET_RS:
        yearly_stats = []
        for label, pc in yearly_pcs:
            trades = fast_backtest(pc, **base_params, target_r=tr, **fixed_exec)
            s = _stats(trades)
            yearly_stats.append((label, s))

            cal = f"{s['calmar']:>7.2f}" if s['calmar'] not in (-999, 999) else "   n/a "
            marker = " <-- baseline" if tr == 2.0 else ""
            print(f"  {tr:>8.1f}  {label:<6}  {s['trades']:>6}  "
                  f"{s['win_rate']*100:>4.1f}%  "
                  f"{s['total_r']:>+8.2f}  {s['max_dd']:>7.2f}  {cal}  "
                  f"{s['avg_r']:>+7.3f}{marker}")

        # 4-year aggregate
        tot_r    = sum(s["total_r"] for _, s in yearly_stats)
        tot_tr   = sum(s["trades"]  for _, s in yearly_stats)
        avg_wr   = sum(s["trades"] * s["win_rate"] for _, s in yearly_stats) / max(tot_tr, 1)
        worst_dd = max(s["max_dd"] for _, s in yearly_stats)
        calmar   = tot_r / worst_dd if worst_dd > 0 and tot_r > 0 else -999.0
        avg_r    = tot_r / tot_tr if tot_tr > 0 else 0.0
        cal_str  = f"{calmar:>7.2f}" if calmar != -999 else "   n/a "
        marker   = " <-- baseline" if tr == 2.0 else ""
        print(f"  {tr:>8.1f}  {'4-yr':<6}  {tot_tr:>6}  "
              f"{avg_wr*100:>4.1f}%  "
              f"{tot_r:>+8.2f}  {worst_dd:>7.2f}  {cal_str}  "
              f"{avg_r:>+7.3f}{marker}")
        print()

        all_results.append({
            "target_r": tr, "years": {k: s for k, s in yearly_stats},
            "total_r": tot_r, "trades": tot_tr, "win_rate": avg_wr,
            "max_dd": worst_dd, "calmar": calmar,
        })

    print(sep)
    print("\n  SUMMARY — 4-year total R by target_r:")
    print(f"  {'target_r':>8}  {'Total R':>8}  {'Trades':>6}  {'Win%':>5}  {'Calmar':>7}")
    for r in all_results:
        cal_str = f"{r['calmar']:>7.2f}" if r['calmar'] != -999 else "   n/a "
        marker  = " <-- current" if r["target_r"] == 2.0 else ""
        print(f"  {r['target_r']:>8.1f}  {r['total_r']:>+8.2f}  "
              f"{r['trades']:>6}  {r['win_rate']*100:>4.1f}%  {cal_str}{marker}")

    best = max(all_results, key=lambda x: x["total_r"])
    print(f"\n  Best total R: target_r={best['target_r']:.1f} → {best['total_r']:+.2f}R")
    best_cal = max((r for r in all_results if r["calmar"] > 0), key=lambda x: x["calmar"],
                   default=None)
    if best_cal:
        print(f"  Best Calmar:  target_r={best_cal['target_r']:.1f} → "
              f"Calmar={best_cal['calmar']:.2f}  ({best_cal['total_r']:+.2f}R)")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_sweep_results.json"
    out_path.write_text(json.dumps(
        {"base_params": base_params, "fixed_exec": fixed_exec, "results": all_results},
        indent=2, default=str), encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
