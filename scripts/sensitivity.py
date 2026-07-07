"""
Parameter sensitivity heatmap for XAUUSD M15 pullback strategy.

Sweeps two 2-D grids and shows Calmar ratio on Train / Test / Stability:
  Grid-1: adx_min (rows)     x sma_len (cols)       -- fixed pullback_atr, stop_atr
  Grid-2: pullback_atr (rows) x stop_atr (cols)      -- fixed adx_min, sma_len

Reads fixed params from the config; [*] marks current config value in each heatmap.

Usage:
  python scripts/sensitivity.py --config configs/live_demo_bt.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import numpy as np
import yaml

from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec

from backtest_demo_executor_live_like import build_weekly_regime, parse_dt
from walk_forward import precompute, fast_backtest, _stats

TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-19"

# Grid-1: adx_min x sma_len
ADX_MINS = [15.0, 18.0, 21.0, 24.3, 27.0, 30.0]
SMA_LENS = [30, 40, 50, 60, 70, 80]

# Grid-2: pullback_atr x stop_atr
PULLBACK_ATRS = [0.10, 0.20, 0.30, 0.40, 0.50]
STOP_ATRS     = [0.80, 1.00, 1.20, 1.40, 1.60]


# ── ASCII cell formatting ─────────────────────────────────────────────────────

def _cell(calmar: float, trades: int, is_cur: bool, width: int = 5) -> str:
    """Format one heatmap cell. ! >= 3, + >= 2, blank >= 1, . >= 0, X neg."""
    if trades < 20:
        val = " n/a"
    elif calmar >= 3.0:
        val = f"{min(calmar, 9.99):>4.2f}!"
    elif calmar >= 2.0:
        val = f"{calmar:>4.2f}+"
    elif calmar >= 1.0:
        val = f"{calmar:>4.2f} "
    elif calmar >= 0.0:
        val = f"{calmar:>4.2f}."
    else:
        val = f"{max(calmar,-9.99):>4.2f}X"

    if is_cur:
        return f"[{val}]"
    return f" {val} "


def _print_heatmap(
    data: dict,          # (row_val, col_val) -> {"calmar": float, "trades": int}
    row_vals: list,
    col_vals: list,
    cur_row,
    cur_col,
    title: str,
    row_label: str,
    col_label: str,
    col_fmt: str = "6.2f",
) -> None:
    COL_W = 6
    print(f"\n  {title}")
    print(f"  Legend: ! >= 3.0  + >= 2.0  (blank) >= 1.0  . >= 0.0  X negative  n/a < 20  [ ] current")

    # header
    hdr = f"\n  {row_label:>10} |"
    for cv in col_vals:
        s = f"{col_label}={cv:{col_fmt}}"
        hdr += f" {s:^{COL_W}}"
    print(hdr)
    print(f"  {'-'*11}+" + "-" * (len(col_vals) * (COL_W + 1)))

    for rv in row_vals:
        row = f"  {rv:>10.2f} |"
        for cv in col_vals:
            d = data.get((rv, cv), {"calmar": -999, "trades": 0})
            is_cur = (abs(float(rv) - float(cur_row)) < 0.01 and
                      abs(float(cv) - float(cur_col)) < 0.01)
            row += _cell(d["calmar"], d["trades"], is_cur)
        print(row)


def _stability_map(
    train_map: dict, test_map: dict, rows: list, cols: list
) -> dict:
    """Return test_calmar / train_calmar for each cell. -999 if undefined."""
    out = {}
    for r in rows:
        for c in cols:
            tr = train_map.get((r, c), {"calmar": 0, "trades": 0})
            te = test_map.get( (r, c), {"calmar": 0, "trades": 0})
            if tr["calmar"] > 0 and tr["trades"] >= 20 and te["trades"] >= 20:
                ratio = te["calmar"] / tr["calmar"]
                out[(r, c)] = {"calmar": ratio, "trades": te["trades"]}
            else:
                out[(r, c)] = {"calmar": -999, "trades": 0}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/derived/runs/sensitivity")
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

    cur_adx = float(sp.get("adx_min",      24.3))
    cur_sma = int(sp.get("sma_len",         60))
    cur_pa  = float(sp.get("pullback_atr",  0.30))
    cur_sa  = float(sp.get("stop_atr",      1.60))

    print(f"\nParameter Sensitivity Analysis -- {symbol} {tf_name}")
    print(f"Current config params: adx={cur_adx}  sma={cur_sma}  pa={cur_pa}  sa={cur_sa}")

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    spec  = get_symbol_spec(symbol)
    point = float(spec.point)

    train_start_dt = parse_dt(TRAIN_START)
    train_end_dt   = parse_dt(TRAIN_END)
    test_start_dt  = parse_dt(TEST_START)
    test_end_dt    = parse_dt(TEST_END)

    def _fetch(label: str, dt_from, dt_to):
        print(f"  Fetching {label}...", end=" ", flush=True)
        rates = mt5.copy_rates_range(symbol, tf, dt_from, dt_to)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"\n  MT5 last_error: {err}")
            print("  Ensure MT5 terminal is open, logged in, and XAUUSD is in Market Watch.")
            raise SystemExit(f"Failed to fetch {label} bars from MT5.")
        print(f"{len(rates)} bars")
        return rates

    print()
    train_rates = _fetch("train 2022-2023", train_start_dt, train_end_dt)
    test_rates  = _fetch("test  2024-2025", test_start_dt,  test_end_dt)
    full_rates  = np.concatenate([train_rates, test_rates])
    print(f"  Combined: {len(full_rates)} bars")

    regime_dict = None
    if regime_cfg.get("enabled", False):
        sma_w  = int(regime_cfg.get("sma_weeks", 40))
        slope  = bool(regime_cfg.get("require_slope", True))
        lb_wks = int(regime_cfg.get("slope_lookback_weeks", 4))
        print("  Building weekly regime dict (2022-2025)...", end=" ", flush=True)
        regime_dict = build_weekly_regime(
            symbol, train_start_dt, test_end_dt, sma_w,
            require_slope=slope, slope_lookback_weeks=lb_wks
        )
        print("done")

    # Pre-compute indicators once per period (one pass per rates array)
    all_sma_lens = sorted(set(SMA_LENS + [cur_sma]))
    print(f"  Pre-computing indicators (train)...", end=" ", flush=True)
    t0 = _time.time()
    train_pc = precompute(
        train_rates, sma_lens=all_sma_lens, regime_dict=regime_dict,
        max_spread_points=max_sp, gate_cfg=rf_cfg, point=point
    )
    print(f"{_time.time()-t0:.1f}s")

    print(f"  Pre-computing indicators (test)... ", end=" ", flush=True)
    t0 = _time.time()
    test_pc = precompute(
        test_rates, sma_lens=all_sma_lens, regime_dict=regime_dict,
        max_spread_points=max_sp, gate_cfg=rf_cfg, point=point
    )
    print(f"{_time.time()-t0:.1f}s")

    def run(pc, adx, sma, pa, sa) -> dict:
        trades = fast_backtest(
            pc, adx_min=adx, pullback_atr=pa, sma_len=sma, stop_atr=sa, **fixed_exec
        )
        return _stats(trades)

    # ── Grid-1: adx_min x sma_len ────────────────────────────────────────────
    total1 = len(ADX_MINS) * len(SMA_LENS)
    print(f"\nGrid-1 (adx_min x sma_len): {len(ADX_MINS)} x {len(SMA_LENS)} = {total1} combos x 2 periods...",
          end=" ", flush=True)
    t0 = _time.time()
    g1_train: dict = {}
    g1_test:  dict = {}
    g1_results = []
    for adx in ADX_MINS:
        for sma in SMA_LENS:
            tr = run(train_pc, adx, sma, cur_pa, cur_sa)
            te = run(test_pc,  adx, sma, cur_pa, cur_sa)
            g1_train[(adx, sma)] = tr
            g1_test[ (adx, sma)] = te
            g1_results.append({"adx_min": adx, "sma_len": sma, "train": tr, "test": te})
    print(f"{_time.time()-t0:.1f}s")

    # ── Grid-2: pullback_atr x stop_atr ──────────────────────────────────────
    total2 = len(PULLBACK_ATRS) * len(STOP_ATRS)
    print(f"Grid-2 (pullback_atr x stop_atr): {len(PULLBACK_ATRS)} x {len(STOP_ATRS)} = {total2} combos x 2 periods...",
          end=" ", flush=True)
    t0 = _time.time()
    g2_train: dict = {}
    g2_test:  dict = {}
    g2_results = []
    for pa in PULLBACK_ATRS:
        for sa in STOP_ATRS:
            tr = run(train_pc, cur_adx, cur_sma, pa, sa)
            te = run(test_pc,  cur_adx, cur_sma, pa, sa)
            g2_train[(pa, sa)] = tr
            g2_test[ (pa, sa)] = te
            g2_results.append({"pullback_atr": pa, "stop_atr": sa, "train": tr, "test": te})
    print(f"{_time.time()-t0:.1f}s")

    # ── Print heatmaps ────────────────────────────────────────────────────────
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"SENSITIVITY HEATMAPS")
    print(f"  Fixed params for Grid-1: pullback_atr={cur_pa:.2f}  stop_atr={cur_sa:.2f}")
    print(f"  Fixed params for Grid-2: adx_min={cur_adx:.1f}  sma_len={cur_sma}")
    print(f"  [ ] marks current config param values")
    print(sep)

    # Grid-1 Train
    _print_heatmap(g1_train, ADX_MINS, SMA_LENS, cur_adx, float(cur_sma),
                   "Grid-1  TRAIN 2022-2023  (Calmar)",
                   "adx_min", "sma", col_fmt="d")

    # Grid-1 Test
    _print_heatmap(g1_test,  ADX_MINS, SMA_LENS, cur_adx, float(cur_sma),
                   "Grid-1  TEST  2024-2025  (Calmar)",
                   "adx_min", "sma", col_fmt="d")

    # Grid-1 Stability
    g1_stab = _stability_map(g1_train, g1_test, ADX_MINS, [float(s) for s in SMA_LENS])
    _print_heatmap(g1_stab, ADX_MINS, [float(s) for s in SMA_LENS],
                   cur_adx, float(cur_sma),
                   "Grid-1  STABILITY (test Calmar / train Calmar;  0.5-1.0 = healthy decay)",
                   "adx_min", "sma", col_fmt=".0f")

    print(f"\n{sep}")

    # Grid-2 Train
    _print_heatmap(g2_train, PULLBACK_ATRS, STOP_ATRS, cur_pa, cur_sa,
                   "Grid-2  TRAIN 2022-2023  (Calmar)",
                   "pa", "sa")

    # Grid-2 Test
    _print_heatmap(g2_test,  PULLBACK_ATRS, STOP_ATRS, cur_pa, cur_sa,
                   "Grid-2  TEST  2024-2025  (Calmar)",
                   "pa", "sa")

    # Grid-2 Stability
    g2_stab = _stability_map(g2_train, g2_test, PULLBACK_ATRS, STOP_ATRS)
    _print_heatmap(g2_stab, PULLBACK_ATRS, STOP_ATRS, cur_pa, cur_sa,
                   "Grid-2  STABILITY (test Calmar / train Calmar)",
                   "pa", "sa")

    # ── Plateau analysis ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PLATEAU ANALYSIS")
    print(sep)

    def plateau_score(matrix: dict, rows, cols, cur_r, cur_c, min_calmar=0.5) -> tuple[int, int]:
        cur_cal = matrix.get((cur_r, cur_c), {"calmar": -999})["calmar"]
        neighbors_above = 0
        neighbors_total = 0
        for rv in rows:
            for cv in cols:
                if (rv, cv) == (cur_r, cur_c):
                    continue
                d_r = abs(rows.index(rv) - rows.index(cur_r))
                d_c = abs(list(cols).index(cv) - list(cols).index(cur_c))
                if d_r <= 1 and d_c <= 1:
                    val = matrix.get((rv, cv), {"calmar": -999, "trades": 0})
                    if val["trades"] >= 20:
                        neighbors_total += 1
                        if val["calmar"] >= min_calmar and val["calmar"] >= 0.5 * cur_cal:
                            neighbors_above += 1
        return neighbors_above, neighbors_total

    for label, tm, rm, rows, cols in [
        ("Grid-1 train", g1_train, g1_train, ADX_MINS, [float(s) for s in SMA_LENS]),
        ("Grid-1 test",  g1_test,  g1_test,  ADX_MINS, [float(s) for s in SMA_LENS]),
        ("Grid-2 train", g2_train, g2_train, PULLBACK_ATRS, STOP_ATRS),
        ("Grid-2 test",  g2_test,  g2_test,  PULLBACK_ATRS, STOP_ATRS),
    ]:
        cr = cur_adx if "Grid-1" in label else cur_pa
        cc = float(cur_sma) if "Grid-1" in label else cur_sa
        cur_cal = tm.get((cr, cc), {"calmar": -999, "trades": 0})["calmar"]
        na, nt = plateau_score(tm, rows, cols, cr, cc)
        pct = 100 * na / nt if nt > 0 else 0
        verdict = "[OK] plateau" if pct >= 50 else "[!] peak"
        print(f"  {label:<16}  current Calmar={cur_cal:>5.2f}  "
              f"neighbors >= half of current: {na}/{nt} ({pct:.0f}%)  {verdict}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_results.json"
    out_path.write_text(
        json.dumps({
            "config_params": {
                "adx_min": cur_adx, "sma_len": cur_sma,
                "pullback_atr": cur_pa, "stop_atr": cur_sa,
            },
            "grid1_adx_sma": g1_results,
            "grid2_pa_sa":   g2_results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
