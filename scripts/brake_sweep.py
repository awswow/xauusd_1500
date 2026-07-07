"""
Monthly brake threshold sweep for XAUUSD M15 pullback strategy.

Tests monthly_loss_limit_r values [3.0, 4.0, 5.0, 6.0, 7.0, 10.0] on
the full 4-year period using fast_backtest with current validated params.

Current baseline: monthly_loss_limit_r=5.0, +225.85R (demo executor).
fast_backtest gives relative comparison — absolute R will differ from demo
executor due to pending order mechanics, but ranking should hold.

Usage:
  python scripts/brake_sweep.py --config configs/live_demo_bt.yaml
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

BRAKE_VALS = [3.0, 4.0, 5.0, 6.0, 7.0, 10.0]

YEARS = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-19"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/derived/runs/brake_sweep")
    args = ap.parse_args()

    cfg       = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol    = cfg["symbol"]
    tf_name   = cfg["timeframe"]
    tf        = {"M15": mt5.TIMEFRAME_M15, "M1": mt5.TIMEFRAME_M1,
                 "H1": mt5.TIMEFRAME_H1}[tf_name]
    ex_cfg    = cfg.get("execution", {}) or {}
    risk_cfg  = cfg.get("risk",      {}) or {}
    rf_cfg    = cfg.get("risk_filters", {}) or {}
    regime_cfg = cfg.get("regime_filter", {}) or {}
    sp        = cfg["strategy"]["params"]
    max_sp    = int(ex_cfg.get("max_spread_points", 45))

    base_params = dict(
        adx_min      = float(sp.get("adx_min",       18.0)),
        sma_len      = int(sp.get("sma_len",           80)),
        pullback_atr = float(sp.get("pullback_atr",  0.30)),
        stop_atr     = float(sp.get("stop_atr",      1.60)),
        target_r     = float(sp.get("target_r",      2.50)),
    )

    fixed_exec_base = dict(
        cooldown_bars        = int(risk_cfg.get("cooldown_bars", 16)),
        daily_loss_limit_usd = float(ex_cfg.get("daily_loss_limit_usd", 45.0)),
        max_trades_per_day   = int(ex_cfg.get("max_trades_per_day", 4)),
        risk_usd_per_trade   = float(ex_cfg.get("risk_usd_per_trade", 15.0)),
        pending_expiry_bars  = int(ex_cfg.get("pending_expiry_bars", 4)),
    )

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    spec  = get_symbol_spec(symbol)
    point = float(spec.point)

    print(f"\nMonthly brake sweep — {symbol} {tf_name}")
    print(f"Base params: {base_params}")
    print(f"Testing monthly_loss_limit_r: {BRAKE_VALS}\n")

    # Precompute bars once per year
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
            regime_dict = build_weekly_regime(
                symbol, start_dt, end_dt,
                int(regime_cfg.get("sma_weeks", 40)),
                require_slope=bool(regime_cfg.get("require_slope", True)),
                slope_lookback_weeks=int(regime_cfg.get("slope_lookback_weeks", 4)),
            )
        pc = precompute(rates, sma_lens=[base_params["sma_len"]],
                        regime_dict=regime_dict,
                        max_spread_points=max_sp,
                        gate_cfg=rf_cfg, point=point)
        yearly_pcs.append((label, pc))

    sep = "=" * 88
    print(sep)
    hdr = (f"  {'brake_R':>8}  {'Year':<6}  {'Trades':>6}  {'Win%':>5}  "
           f"{'Total R':>8}  {'Max DD':>7}  {'Calmar':>7}  {'$/trade':>7}")
    print(hdr)
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

    all_results: list[dict] = []

    for br in BRAKE_VALS:
        fixed_exec = {**fixed_exec_base, "monthly_loss_limit_r": br}
        yearly_stats = []

        for label, pc in yearly_pcs:
            trades = fast_backtest(pc, **base_params, **fixed_exec)
            s      = _stats(trades)
            yearly_stats.append((label, s))
            cal    = f"{s['calmar']:>7.2f}" if s['calmar'] not in (-999, 999) else "   n/a "
            marker = " <-- current" if br == 5.0 else ""
            print(f"  {br:>8.1f}  {label:<6}  {s['trades']:>6}  "
                  f"{s['win_rate']*100:>4.1f}%  "
                  f"{s['total_r']:>+8.2f}  {s['max_dd']:>7.2f}  {cal}  "
                  f"{s['avg_r']:>+7.3f}{marker}")

        tot_r  = sum(s["total_r"] for _, s in yearly_stats)
        tot_tr = sum(s["trades"]  for _, s in yearly_stats)
        avg_wr = sum(s["trades"] * s["win_rate"] for _, s in yearly_stats) / max(tot_tr, 1)
        wdd    = max(s["max_dd"] for _, s in yearly_stats)
        cal    = tot_r / wdd if wdd > 0 and tot_r > 0 else -999.0
        avg_r  = tot_r / tot_tr if tot_tr > 0 else 0.0
        cal_s  = f"{cal:>7.2f}" if cal != -999 else "   n/a "
        marker = " <-- current" if br == 5.0 else ""
        print(f"  {br:>8.1f}  {'4-yr':<6}  {tot_tr:>6}  "
              f"{avg_wr*100:>4.1f}%  "
              f"{tot_r:>+8.2f}  {wdd:>7.2f}  {cal_s}  "
              f"{avg_r:>+7.3f}{marker}")
        print()

        all_results.append({"brake_r": br, "total_r": tot_r, "trades": tot_tr,
                             "win_rate": avg_wr, "max_dd": wdd, "calmar": cal,
                             "years": {k: s for k, s in yearly_stats}})

    print(sep)
    print("\n  SUMMARY — 4-year total R by brake threshold:")
    print(f"  {'brake_R':>8}  {'Total R':>8}  {'Trades':>6}  {'Win%':>5}  {'Calmar':>7}")
    for r in all_results:
        cal_s  = f"{r['calmar']:>7.2f}" if r['calmar'] != -999 else "   n/a "
        marker = " <-- current" if r["brake_r"] == 5.0 else ""
        print(f"  {r['brake_r']:>8.1f}  {r['total_r']:>+8.2f}  "
              f"{r['trades']:>6}  {r['win_rate']*100:>4.1f}%  {cal_s}{marker}")

    best = max(all_results, key=lambda x: x["total_r"])
    print(f"\n  Best total R: brake={best['brake_r']:.1f} → {best['total_r']:+.2f}R")
    best_cal = max((r for r in all_results if r["calmar"] > 0), key=lambda x: x["calmar"],
                   default=None)
    if best_cal:
        print(f"  Best Calmar:  brake={best_cal['brake_r']:.1f} → "
              f"Calmar={best_cal['calmar']:.2f}  ({best_cal['total_r']:+.2f}R)")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brake_sweep_results.json"
    out_path.write_text(json.dumps({"base_params": base_params, "results": all_results},
                                   indent=2, default=str), encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
