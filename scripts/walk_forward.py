"""
Walk-forward validation for XAUUSD M15 pullback strategy.

Two modes:

Single-split (default):
  Train : 2022-01-01 to 2023-12-31  (parameter selection)
  Test  : 2024-01-01 to 2025-12-19  (blind hold-out)

Nested rolling (--nested):
  Three rolling windows, each training on all prior years and evaluating on
  the next unseen year.  All five strategy params (adx_min, sma_len,
  stop_atr, target_r, pullback_atr) are optimised within each training
  window.  Only OOS results are aggregated — no in-sample leakage.
  432-combo grid; runs in seconds per window.

Usage:
  python scripts/walk_forward.py --config configs/live_demo_bt.yaml
  python scripts/walk_forward.py --config configs/live_demo_bt.yaml --nested
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import numpy as np
import yaml

from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec

from backtest_demo_executor_live_like import build_weekly_regime, parse_dt

# ── Single-split constants ────────────────────────────────────────────────────
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-19"

GRID = {
    "adx_min":      [18.0, 24.3, 30.0],
    "pullback_atr": [0.20, 0.30, 0.45],
    "sma_len":      [40,   60,   80  ],
    "stop_atr":     [1.00, 1.20, 1.60],
}

# ── Nested rolling WF constants ───────────────────────────────────────────────

# All five strategy params in the grid (adx_min × sma_len × stop_atr × target_r × pullback_atr)
NESTED_GRID = {
    "adx_min":      [15.0, 18.0, 21.0, 24.3],
    "sma_len":      [60,   80,   100],
    "stop_atr":     [1.20, 1.40, 1.60, 1.80],
    "target_r":     [2.0,  2.5,  3.0],
    "pullback_atr": [0.20, 0.30, 0.45],
}

# Fetched as separate yearly calls to stay under the MT5 two-year rate limit
YEARLY_SLICES = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-19"),
]

# (label, train_year_keys, test_year_key)
ROLLING_WINDOWS = [
    ("W1: train=2022       test=2023",   ["2022"],                      "2023"),
    ("W2: train=2022-23    test=2024",   ["2022", "2023"],               "2024"),
    ("W3: train=2022-24    test=2025",   ["2022", "2023", "2024"],       "2025"),
]

FIXED_ATR_LEN          = 14
FIXED_TARGET_R         = 2.0
FIXED_PULLBACK_LOOKBACK = 3
WARMUP = 200  # bars skipped at start of every backtest


# ── Indicator computation (numpy, causal, computed once) ─────────────────────

def _sma(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    kernel = np.ones(n) / n
    valid = np.convolve(close, kernel, mode="valid")
    out[n - 1:] = valid
    return out


def _wilder_sum(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = float(np.sum(x[:n]))
    for i in range(n, len(x)):
        out[i] = out[i - 1] - out[i - 1] / n + x[i]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    hl = high - low
    hpc = np.abs(np.concatenate([[0.0], high[1:] - close[:-1]]))
    lpc = np.abs(np.concatenate([[0.0], low[1:]  - close[:-1]]))
    tr = np.maximum(hl, np.maximum(hpc, lpc))
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    out[n - 1] = float(np.mean(tr[:n]))
    alpha = 1.0 / n
    for i in range(n, len(close)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(high)
    out = np.full(nb, np.nan)
    if nb < 2 * n:
        return out

    up   = np.diff(high, prepend=high[0])
    down = -np.diff(low,  prepend=low[0])
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)

    hl  = high - low
    hpc = np.abs(np.concatenate([[0.0], high[1:] - close[:-1]]))
    lpc = np.abs(np.concatenate([[0.0], low[1:]  - close[:-1]]))
    tr  = np.maximum(hl, np.maximum(hpc, lpc))

    tr_s   = _wilder_sum(tr, n)
    pdm_s  = _wilder_sum(pdm, n)
    mdm_s  = _wilder_sum(mdm, n)

    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = np.where(tr_s > 0, 100.0 * pdm_s / tr_s, np.nan)
        mdi = np.where(tr_s > 0, 100.0 * mdm_s / tr_s, np.nan)
        denom = pdi + mdi
        dx  = np.where(denom > 0, 100.0 * np.abs(pdi - mdi) / denom, np.nan)

    seed = 2 * n - 1
    out[seed] = float(np.nanmean(dx[n: 2 * n]))
    prev = out[seed]
    for i in range(2 * n, nb):
        dxi = dx[i]
        if not np.isnan(dxi):
            prev = (prev * (n - 1) + dxi) / n
        out[i] = prev
    return out


def _tq_array(
    atr14: np.ndarray,
    adx14: np.ndarray,
    adx_min: float,
    atr_window: int,
    atr_quantile: float,
) -> np.ndarray:
    """Pre-compute trend-quality gate result for every bar (O(n * window))."""
    n = len(atr14)
    out = np.zeros(n, dtype=bool)
    for i in range(WARMUP, n):
        adx_v = adx14[i]
        atr_v = atr14[i]
        if np.isnan(adx_v) or np.isnan(atr_v) or atr_v <= 0:
            continue
        if adx_v < adx_min:
            continue
        w_start = max(0, i - atr_window)
        window = atr14[w_start:i]
        window = window[~np.isnan(window)]
        if len(window) < max(10, atr_window // 2):
            continue
        q = float(np.quantile(window, atr_quantile))
        if atr_v >= q:
            out[i] = True
    return out


def precompute(
    rates,
    *,
    sma_lens: list[int],
    regime_dict: Optional[dict],
    max_spread_points: int,
    gate_cfg: dict,
    point: float,
) -> dict:
    """Compute all indicator series once for a rates array."""
    n = len(rates)

    high  = np.array([float(r["high"])  for r in rates])
    low   = np.array([float(r["low"])   for r in rates])
    close = np.array([float(r["close"]) for r in rates])
    open_ = np.array([float(r["open"])  for r in rates])
    times = [datetime.fromtimestamp(int(r["time"]), tz=timezone.utc) for r in rates]
    spreads = np.array([int(r["spread"]) for r in rates], dtype=int)

    atr14 = _atr(high, low, close, FIXED_ATR_LEN)
    adx14 = _adx(high, low, close, FIXED_ATR_LEN)

    sma_series: dict[int, np.ndarray] = {}
    for slen in sma_lens:
        sma_series[slen] = _sma(close, slen)

    tq_ok = _tq_array(
        atr14, adx14,
        adx_min=float(gate_cfg.get("tq_adx_min", 15.0)),
        atr_window=int(gate_cfg.get("tq_atr_ref_window", 120)),
        atr_quantile=float(gate_cfg.get("tq_atr_quantile", 0.20)),
    )

    spread_ok = spreads <= max_spread_points

    regime_ok = np.zeros(n, dtype=bool)
    if regime_dict is not None:
        for i, t in enumerate(times):
            monday = t.date() - timedelta(days=t.weekday())
            regime_ok[i] = regime_dict.get(monday, False)
    else:
        regime_ok[:] = True

    return {
        "n": n,
        "high": high, "low": low, "close": close, "open_": open_,
        "times": times, "spreads": spreads,
        "atr14": atr14, "adx14": adx14,
        "sma_series": sma_series,
        "tq_ok": tq_ok,
        "spread_ok": spread_ok,
        "regime_ok": regime_ok,
        "point": point,
    }


# ── Fast O(n) backtest loop ───────────────────────────────────────────────────

def fast_backtest(
    pc: dict,
    *,
    adx_min: float,
    pullback_atr: float,
    sma_len: int,
    stop_atr: float,
    target_r: float = FIXED_TARGET_R,
    # fixed controls
    cooldown_bars: int,
    daily_loss_limit_usd: float,
    max_trades_per_day: int,
    risk_usd_per_trade: float,
    pending_expiry_bars: int,
    monthly_loss_limit_r: float,
) -> list[dict]:
    n       = pc["n"]
    high    = pc["high"]
    low     = pc["low"]
    close_  = pc["close"]
    times   = pc["times"]
    atr14   = pc["atr14"]
    adx14   = pc["adx14"]
    sma_s   = pc["sma_series"][sma_len]
    tq_ok   = pc["tq_ok"]
    sp_ok   = pc["spread_ok"]
    reg_ok  = pc["regime_ok"]
    point   = pc["point"]
    risk_usd = risk_usd_per_trade

    trades: list[dict] = []

    in_pos = False
    entry_price = stop_price = target_price = 0.0
    entry_time = None

    pending_price = 0.0
    pending_sl    = 0.0
    pending_set   = False
    pending_idx   = 0

    last_fill_i   = -cooldown_bars - 1

    day_key:   Optional[str] = None
    day_pnl_usd = 0.0
    day_trades  = 0
    month_key:  Optional[str] = None
    month_r     = 0.0

    for i in range(WARMUP, n):
        t  = times[i]
        dk = t.date().isoformat()
        mk = t.strftime("%Y-%m")

        # Day / month rollover
        if day_key is None:   day_key   = dk
        if month_key is None: month_key = mk
        if dk != day_key:
            day_key = dk; day_pnl_usd = 0.0; day_trades = 0
        if mk != month_key:
            month_key = mk; month_r = 0.0

        # Spread gate (always applied)
        if not sp_ok[i]:
            continue

        hi = high[i]
        lo = low[i]

        # ── Manage open position ────────────────────────────────────────
        if in_pos:
            hit_stop   = lo <= stop_price
            hit_target = hi >= target_price
            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    reason = "stop"; ex_price = stop_price    # conservative
                elif hit_stop:
                    reason = "stop"; ex_price = stop_price
                else:
                    reason = "target"; ex_price = target_price

                pnl_pts  = (ex_price - entry_price) / point
                risk_pts = abs(entry_price - stop_price) / point
                r_mult   = (pnl_pts / risk_pts) if risk_pts > 0 else 0.0

                realized = r_mult * risk_usd
                day_pnl_usd += realized
                month_r     += r_mult

                trades.append({
                    "r_multiple":  r_mult,
                    "exit_reason": reason,
                    "entry_time":  entry_time,
                    "exit_time":   t,
                })
                in_pos = False
            continue

        # ── Manage pending order ────────────────────────────────────────
        if pending_set:
            if (i - pending_idx) >= pending_expiry_bars:
                pending_set = False
            elif hi >= pending_price:
                # Fill
                ent   = pending_price
                sl    = pending_sl
                tp    = ent + target_r * (ent - sl)
                if sl < ent < tp:
                    in_pos       = True
                    entry_price  = ent
                    stop_price   = sl
                    target_price = tp
                    entry_time   = t
                    last_fill_i  = i
                    day_trades  += 1
                pending_set = False
            continue

        # ── Risk controls ───────────────────────────────────────────────
        if day_pnl_usd <= -abs(daily_loss_limit_usd):
            continue
        if day_trades >= max_trades_per_day:
            continue
        if (i - last_fill_i) < cooldown_bars:
            continue
        if monthly_loss_limit_r > 0 and month_r <= -monthly_loss_limit_r:
            continue

        # ── Regime gate ─────────────────────────────────────────────────
        if not reg_ok[i]:
            continue

        # ── Trend-quality gate ──────────────────────────────────────────
        if not tq_ok[i]:
            continue

        # ── Strategy signal ─────────────────────────────────────────────
        sma_now  = sma_s[i]
        sma_prev = sma_s[i - 1]
        atr_now  = atr14[i]
        adx_now  = adx14[i]

        if np.isnan(sma_now) or np.isnan(sma_prev): continue
        if np.isnan(atr_now) or atr_now <= 0:       continue
        if np.isnan(adx_now):                        continue

        if close_[i] <= sma_now:  continue   # S1: close > SMA
        if sma_now <= sma_prev:   continue   # S2: SMA slope > 0
        if adx_now < adx_min:     continue   # S3: ADX gate

        # S4: any of last 3 lows within pullback_atr * ATR of SMA
        lb_start = max(0, i - (FIXED_PULLBACK_LOOKBACK - 1))
        recent_lows = low[lb_start: i + 1]
        if not np.any((sma_now - recent_lows) <= pullback_atr * atr_now):
            continue

        # Signal: place BUY_STOP at current bar's high
        entry_level = hi
        sl_level    = entry_level - stop_atr * atr_now
        if sl_level >= entry_level:
            continue

        pending_price = entry_level
        pending_sl    = sl_level
        pending_set   = True
        pending_idx   = i

    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "total_r": 0.0,
                "max_dd": 0.0, "calmar": -999.0, "avg_r": 0.0}
    rs    = [t["r_multiple"] for t in trades]
    wins  = sum(1 for r in rs if r > 0)
    n     = len(rs)
    total = sum(rs)
    # max drawdown in R
    eq = 0.0; pk = 0.0; md = 0.0
    for r in rs:
        eq += r
        if eq > pk: pk = eq
        d = pk - eq
        if d > md: md = d
    calmar = total / md if (md > 0 and total > 0 and n >= 20) else (
        -999.0 if total <= 0 else 999.0
    )
    return {
        "trades":   n,
        "win_rate": wins / n,
        "total_r":  round(total, 3),
        "max_dd":   round(md, 3),
        "calmar":   round(calmar, 3),
        "avg_r":    round(total / n, 4),
    }


# ── Nested rolling walk-forward ───────────────────────────────────────────────

def run_nested_wf(
    yearly_rates: dict[str, np.ndarray],
    full_regime: Optional[dict],
    *,
    max_spread_points: int,
    gate_cfg: dict,
    point: float,
    fixed_exec: dict,
    top_n: int = 3,
) -> dict:
    """
    For each rolling window: optimise the full NESTED_GRID on the train
    period, freeze the winner, evaluate on the unseen test year.
    Aggregate only OOS results — no in-sample leakage.
    """
    ng_keys  = list(NESTED_GRID.keys())
    ng_vals  = list(NESTED_GRID.values())
    combos   = list(itertools.product(*ng_vals))
    n_combos = len(combos)
    sma_lens = sorted(set(NESTED_GRID["sma_len"]))

    sep = "=" * 96
    print(f"\n{sep}")
    print(f"NESTED ROLLING WALK-FORWARD")
    grid_desc = " x ".join(f"{k}({len(v)})" for k, v in NESTED_GRID.items())
    print(f"Grid: {grid_desc} = {n_combos} combos per window")
    print(sep)

    window_results: list[dict] = []
    all_oos_trades: list[dict] = []

    for win_label, train_years, test_year in ROLLING_WINDOWS:
        print(f"\n  {win_label}")
        print(f"  {'─' * 70}")

        tr_parts = [yearly_rates[y] for y in train_years if y in yearly_rates]
        te_parts = [yearly_rates[test_year]] if test_year in yearly_rates else []
        if not tr_parts or not te_parts:
            print("  [skip] missing rate data for this window")
            continue

        train_arr = np.concatenate(tr_parts)
        test_arr  = te_parts[0]

        print(f"  Bars — train: {len(train_arr):,}  test: {len(test_arr):,}  ", end="", flush=True)
        t0 = _time.time()
        train_pc = precompute(train_arr, sma_lens=sma_lens,
                              regime_dict=full_regime,
                              max_spread_points=max_spread_points,
                              gate_cfg=gate_cfg, point=point)
        test_pc  = precompute(test_arr, sma_lens=sma_lens,
                              regime_dict=full_regime,
                              max_spread_points=max_spread_points,
                              gate_cfg=gate_cfg, point=point)
        print(f"precompute: {_time.time() - t0:.1f}s")

        print(f"  Sweeping {n_combos} combos on train...", end=" ", flush=True)
        t0 = _time.time()
        grid_res: list[dict] = []
        for combo in combos:
            p      = dict(zip(ng_keys, combo))
            trades = fast_backtest(train_pc, **p, **fixed_exec)
            s      = _stats(trades)
            grid_res.append({"params": p, "stats": s})
        print(f"{_time.time() - t0:.1f}s")

        valid = [r for r in grid_res
                 if r["stats"]["trades"] >= 20 and r["stats"]["total_r"] > 0]
        valid.sort(key=lambda x: x["stats"]["calmar"], reverse=True)

        if not valid:
            print("  [!] No valid combo on train — skipping window")
            continue

        winner = valid[0]
        win_p  = winner["params"]
        win_tr = winner["stats"]

        oos_trades = fast_backtest(test_pc, **win_p, **fixed_exec)
        oos_stats  = _stats(oos_trades)
        all_oos_trades.extend(oos_trades)

        pstr = (f"adx={win_p['adx_min']:.1f}  sma={win_p['sma_len']}  "
                f"sa={win_p['stop_atr']:.2f}  tr={win_p['target_r']:.1f}  "
                f"pa={win_p['pullback_atr']:.2f}")
        print(f"\n  Train winner: {pstr}")

        hdr = (f"  {'Segment':<12}  {'Trades':>6}  {'Win%':>5}  "
               f"{'Total R':>8}  {'MaxDD':>6}  {'Calmar':>7}")
        print(hdr)
        print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*7}")
        for seg_name, stats, note in [
            ("Train",    win_tr,    ""),
            ("OOS test", oos_stats, "  <- hold-out"),
        ]:
            cal_s = f"{stats['calmar']:>7.2f}" if stats["calmar"] not in (-999, 999) else "    n/a"
            print(f"  {seg_name:<12}  {stats['trades']:>6}  {stats['win_rate']*100:>4.1f}%  "
                  f"{stats['total_r']:>+8.2f}  {stats['max_dd']:>6.2f}  {cal_s}{note}")

        print(f"\n  Top-{top_n} train combos (param stability check):")
        for rank, r in enumerate(valid[:top_n], 1):
            rp = r["params"]; rs = r["stats"]
            print(f"    #{rank}: adx={rp['adx_min']:.1f} sma={rp['sma_len']:>3} "
                  f"sa={rp['stop_atr']:.2f} tr={rp['target_r']:.1f} pa={rp['pullback_atr']:.2f}"
                  f"  ->  {rs['total_r']:>+7.2f}R  Cal={rs['calmar']:.2f}")

        window_results.append({
            "label":        win_label,
            "winner_params": win_p,
            "train_stats":  win_tr,
            "oos_stats":    oos_stats,
            "top_n_train":  [{"params": r["params"], "stats": r["stats"]} for r in valid[:top_n]],
        })

    agg = _stats(all_oos_trades)

    print(f"\n{sep}")
    print("AGGREGATED OOS RESULTS  (concatenated across all rolling windows)")
    print(sep)
    cal_s = f"{agg['calmar']:.2f}" if agg["calmar"] not in (-999, 999) else "n/a"
    print(f"\n  Trades: {agg['trades']}  |  Win%: {agg['win_rate']*100:.1f}%  |  "
          f"Total R: {agg['total_r']:+.2f}R  |  Max DD: {agg['max_dd']:.2f}R  |  "
          f"Calmar: {cal_s}")

    print(f"\n  Per-window OOS breakdown:")
    for wr in window_results:
        s   = wr["oos_stats"]
        cal = f"{s['calmar']:.2f}" if s["calmar"] not in (-999, 999) else "n/a"
        print(f"    {wr['label']:<42}  "
              f"{s['trades']:>4} trades  {s['total_r']:>+7.2f}R  Calmar={cal}")

    if len(window_results) > 1:
        print(f"\n  Winner params per window (consistency check):")
        for wr in window_results:
            wp = wr["winner_params"]
            print(f"    {wr['label'][:34]:<34}  "
                  f"adx={wp['adx_min']:.1f}  sma={wp['sma_len']}  "
                  f"sa={wp['stop_atr']:.2f}  tr={wp['target_r']:.1f}  "
                  f"pa={wp['pullback_atr']:.2f}")

    # How close is the current live config to the nested WF winners?
    live_p = {"adx_min": 18.0, "sma_len": 80, "stop_atr": 1.60, "target_r": 2.5, "pullback_atr": 0.30}
    print(f"\n  Current live config for reference:")
    print(f"    adx={live_p['adx_min']:.1f}  sma={live_p['sma_len']}  "
          f"sa={live_p['stop_atr']:.2f}  tr={live_p['target_r']:.1f}  "
          f"pa={live_p['pullback_atr']:.2f}")

    print(f"\n{sep}\n")

    return {
        "windows":    window_results,
        "agg_oos":    agg,
        "oos_trades": all_oos_trades,
    }


# ── Entry points ──────────────────────────────────────────────────────────────

def _run_nested(
    args,
    cfg: dict,
    symbol: str,
    tf: int,
    rf_cfg: dict,
    regime_cfg: dict,
    fixed_exec: dict,
    max_spread_points: int,
    point: float,
) -> None:
    """Nested rolling walk-forward: fetch yearly bars, run rolling optimisation."""
    timeframe_name = cfg["timeframe"]
    print(f"\nLoading bars: {symbol} {timeframe_name}")

    yearly_rates: dict[str, np.ndarray] = {}
    for yr_label, yr_start, yr_end in YEARLY_SLICES:
        rates = mt5.copy_rates_range(symbol, tf, parse_dt(yr_start), parse_dt(yr_end))
        if rates is None or len(rates) == 0:
            print(f"  {yr_label}: no data — skipping")
            continue
        yearly_rates[yr_label] = rates
        print(f"  {yr_label}: {len(rates):,} bars")

    if len(yearly_rates) < 2:
        raise SystemExit("Not enough yearly data to run nested WF.")

    full_regime: Optional[dict] = None
    if regime_cfg.get("enabled", False):
        sma_w    = int(regime_cfg.get("sma_weeks", 40))
        slope    = bool(regime_cfg.get("require_slope", True))
        lb_weeks = int(regime_cfg.get("slope_lookback_weeks", 4))
        print("  Building full-range regime dict (2022-2025)...")
        # Two calls to stay within MT5 two-year limit for weekly bars
        r1 = build_weekly_regime(symbol, parse_dt("2022-01-01"), parse_dt("2023-12-31"),
                                 sma_w, require_slope=slope, slope_lookback_weeks=lb_weeks)
        r2 = build_weekly_regime(symbol, parse_dt("2024-01-01"), parse_dt("2025-12-19"),
                                 sma_w, require_slope=slope, slope_lookback_weeks=lb_weeks)
        full_regime = {}
        if r1: full_regime.update(r1)
        if r2: full_regime.update(r2)
        print(f"  Regime dict: {len(full_regime)} weeks")

    result = run_nested_wf(
        yearly_rates, full_regime,
        max_spread_points=max_spread_points,
        gate_cfg=rf_cfg, point=point,
        fixed_exec=fixed_exec,
        top_n=args.top_n,
    )

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nested_wf_results.json"
    out_path.write_text(
        json.dumps({"windows": result["windows"], "agg_oos": result["agg_oos"]},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Results saved: {out_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out",    default="data/derived/runs/walk_forward")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument(
        "--nested", action="store_true",
        help="Run nested rolling WF: all 5 params in grid, 3 rolling windows, pure OOS aggregation",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    tf        = {"M15": mt5.TIMEFRAME_M15, "M1": mt5.TIMEFRAME_M1,
                 "H1": mt5.TIMEFRAME_H1}[timeframe]
    ex_cfg    = cfg.get("execution", {}) or {}
    risk_cfg  = cfg.get("risk",      {}) or {}
    rf_cfg    = cfg.get("risk_filters", {}) or {}
    regime_cfg = cfg.get("regime_filter", {}) or {}

    fixed_exec = {
        "cooldown_bars":        int(risk_cfg.get("cooldown_bars", 16)),
        "daily_loss_limit_usd": float(ex_cfg.get("daily_loss_limit_usd", 45.0)),
        "max_trades_per_day":   int(ex_cfg.get("max_trades_per_day", 4)),
        "risk_usd_per_trade":   float(ex_cfg.get("risk_usd_per_trade", 15.0)),
        "pending_expiry_bars":  int(ex_cfg.get("pending_expiry_bars", 4)),
        "monthly_loss_limit_r": float(risk_cfg.get("monthly_loss_limit_r", 0.0) or 0.0),
    }
    max_spread_points = int(ex_cfg.get("max_spread_points", 45))

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    spec  = get_symbol_spec(symbol)
    point = float(spec.point)

    # ── Nested rolling mode ──────────────────────────────────────────────────
    if args.nested:
        _run_nested(args, cfg, symbol, tf, rf_cfg, regime_cfg,
                    fixed_exec, max_spread_points, point)
        return

    # ── Single-split mode (original) ─────────────────────────────────────────
    cur_sp = dict(cfg["strategy"]["params"])
    cur_sp.pop("min_bars_between_entries", None)

    train_start = parse_dt(TRAIN_START)
    train_end   = parse_dt(TRAIN_END)
    test_start  = parse_dt(TEST_START)
    test_end    = parse_dt(TEST_END)

    print(f"\nLoading bars: {symbol} {timeframe}")
    train_rates = mt5.copy_rates_range(symbol, tf, train_start, train_end)
    test_rates  = mt5.copy_rates_range(symbol, tf, test_start,  test_end)
    if train_rates is None or test_rates is None:
        raise SystemExit("Failed to fetch rates from MT5.")
    print(f"  Train bars: {len(train_rates)}   Test bars: {len(test_rates)}")

    sma_lens_in_grid = sorted(set(GRID["sma_len"] + [int(cur_sp.get("sma_len", 60))]))

    train_regime: Optional[dict] = None
    test_regime:  Optional[dict] = None
    if regime_cfg.get("enabled", False):
        sma_w    = int(regime_cfg.get("sma_weeks", 40))
        slope    = bool(regime_cfg.get("require_slope", True))
        lb_weeks = int(regime_cfg.get("slope_lookback_weeks", 4))
        print("  Building weekly regime dicts...")
        train_regime = build_weekly_regime(symbol, train_start, train_end, sma_w,
                                           require_slope=slope, slope_lookback_weeks=lb_weeks)
        test_regime  = build_weekly_regime(symbol, test_start,  test_end,  sma_w,
                                           require_slope=slope, slope_lookback_weeks=lb_weeks)

    print("  Pre-computing indicators (train)...", end=" ", flush=True)
    t0 = _time.time()
    train_pc = precompute(train_rates, sma_lens=sma_lens_in_grid,
                          regime_dict=train_regime,
                          max_spread_points=max_spread_points,
                          gate_cfg=rf_cfg, point=point)
    print(f"{_time.time()-t0:.1f}s")

    print("  Pre-computing indicators (test)...", end=" ", flush=True)
    t0 = _time.time()
    test_pc  = precompute(test_rates,  sma_lens=sma_lens_in_grid,
                          regime_dict=test_regime,
                          max_spread_points=max_spread_points,
                          gate_cfg=rf_cfg, point=point)
    print(f"{_time.time()-t0:.1f}s")

    def run(pc, params):
        trades = fast_backtest(pc, **params, **fixed_exec)
        return _stats(trades)

    def grid_params(combo_dict):
        return {
            "adx_min":      combo_dict["adx_min"],
            "pullback_atr": combo_dict["pullback_atr"],
            "sma_len":      combo_dict["sma_len"],
            "stop_atr":     combo_dict["stop_atr"],
        }

    # ── Step 1: Current params segmented ────────────────────────────────────
    print("\n" + "="*92)
    print("STEP 1 -- Segmented performance: current parameters (selected on full 4-year history)")
    print("="*92)

    cur_params = {
        "adx_min":      float(cur_sp.get("adx_min", 24.3)),
        "pullback_atr": float(cur_sp.get("pullback_atr", 0.30)),
        "sma_len":      int(cur_sp.get("sma_len", 60)),
        "stop_atr":     float(cur_sp.get("stop_atr", 1.60)),
    }
    cur_train = run(train_pc, cur_params)
    cur_test  = run(test_pc,  cur_params)

    print(f"\n  {'Segment':<22} {'Trades':>6}  {'Win%':>6}  {'Total R':>8}  {'Max DD':>7}  {'Calmar':>7}")
    print(f"  {'-'*22} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}")
    for name, r in [("Train 2022-2023", cur_train), ("Test  2024-2025", cur_test)]:
        print(f"  {name:<22} {r['trades']:>6}  {r['win_rate']*100:>5.1f}%  "
              f"{r['total_r']:>+8.2f}  {r['max_dd']:>7.2f}  {r['calmar']:>7.2f}")

    if cur_train["calmar"] > 0:
        decay = (cur_test["calmar"] - cur_train["calmar"]) / abs(cur_train["calmar"]) * 100
        print(f"\n  Calmar change train->test: {decay:+.1f}%")
    print(f"\n  [!] Parameters chosen with knowledge of both periods.")
    print(f"      This split shows stability, not true out-of-sample validation.")

    # ── Step 2: Grid sweep on train ──────────────────────────────────────────
    print("\n" + "="*92)
    print("STEP 2 -- Grid sweep on TRAIN period (2022-2023)")
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"  {' x '.join(f'{k}({len(v)})' for k, v in GRID.items())} = {total} combinations")
    print("="*92)

    grid_results = []
    t0 = _time.time()
    for i, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        r = run(train_pc, p)
        grid_results.append({"params": p, "train": r})
        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = _time.time() - t0
            print(f"  [{i+1:>3}/{total}]  {elapsed:.1f}s", end="\r")

    elapsed_total = _time.time() - t0
    print(f"  Grid complete: {total} combinations in {elapsed_total:.1f}s")

    valid = [r for r in grid_results if r["train"]["trades"] >= 20 and r["train"]["total_r"] > 0]
    valid.sort(key=lambda x: x["train"]["calmar"], reverse=True)
    print(f"  {len(valid)}/{total} combinations with >=20 trades and positive R on train")

    print(f"\n  Top {min(args.top_n, len(valid))} by Calmar (train):")
    hdr = f"  {'Rank':<5} {'Trades':>6}  {'Win%':>5}  {'Total R':>8}  {'MaxDD':>6}  {'Calmar':>6}  Params"
    print(hdr)
    print(f"  {'-'*5} {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*40}")
    for rank, row in enumerate(valid[:args.top_n], 1):
        r = row["train"]; p = row["params"]
        print(f"  #{rank:<4} {r['trades']:>6}  {r['win_rate']*100:>4.1f}%  "
              f"{r['total_r']:>+8.2f}  {r['max_dd']:>6.2f}  {r['calmar']:>6.2f}  "
              f"adx={p['adx_min']:.1f} pa={p['pullback_atr']:.2f} "
              f"sma={p['sma_len']:>2} sa={p['stop_atr']:.2f}")

    # ── Step 3: Blind test ───────────────────────────────────────────────────
    print("\n" + "="*92)
    print("STEP 3 -- Blind test: top-N train winners applied to TEST period (2024-2025)")
    print("="*92)

    top_n = valid[:args.top_n]
    for row in top_n:
        row["test"] = run(test_pc, row["params"])

    print(f"\n  {'Rnk':<4}  {'--- TRAIN 2022-2023 ---':^35}  {'--- TEST 2024-2025 ---':^35}  Params")
    print(f"  {'':4}  {'n':>4} {'WR%':>5} {'R':>8} {'DD':>6} {'Cal':>6}  "
          f"  {'n':>4} {'WR%':>5} {'R':>8} {'DD':>6} {'Cal':>6}")
    print(f"  {'-'*4}  {'-'*4} {'-'*5} {'-'*8} {'-'*6} {'-'*6}  "
          f"  {'-'*4} {'-'*5} {'-'*8} {'-'*6} {'-'*6}  {'-'*38}")
    for rank, row in enumerate(top_n, 1):
        tr = row["train"]; te = row["test"]; p = row["params"]
        print(f"  #{rank:<3}  {tr['trades']:>4} {tr['win_rate']*100:>4.1f}% "
              f"{tr['total_r']:>+8.2f} {tr['max_dd']:>6.2f} {tr['calmar']:>6.2f}  "
              f"  {te['trades']:>4} {te['win_rate']*100:>4.1f}% "
              f"{te['total_r']:>+8.2f} {te['max_dd']:>6.2f} {te['calmar']:>6.2f}  "
              f"adx={p['adx_min']:.1f} pa={p['pullback_atr']:.2f} "
              f"sma={p['sma_len']:>2} sa={p['stop_atr']:.2f}")

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    print("\n" + "="*92)
    print("STEP 4 -- Summary: current params vs. walk-forward selected params")
    print("="*92)

    if top_n:
        best       = top_n[0]
        best_train = best["train"]
        best_test  = best["test"]
        best_p     = best["params"]

        print(f"\n  Walk-forward best params (by train Calmar):")
        print(f"    adx_min={best_p['adx_min']:.1f}  pullback_atr={best_p['pullback_atr']:.2f}  "
              f"sma_len={best_p['sma_len']}  stop_atr={best_p['stop_atr']:.2f}")
        print(f"\n  Current params (4-year selected):")
        print(f"    adx_min={cur_params['adx_min']:.1f}  pullback_atr={cur_params['pullback_atr']:.2f}  "
              f"sma_len={cur_params['sma_len']}  stop_atr={cur_params['stop_atr']:.2f}")

        print(f"\n  {'':34}  {'Train R / Cal':>14}  {'Test R / Cal':>14}  {'Cal decay':>10}")
        print(f"  {'-'*34}  {'-'*14}  {'-'*14}  {'-'*10}")
        for label, tr_r, te_r in [
            ("Current params (4-year selection)", cur_train, cur_test),
            ("WF params (train-only selection)", best_train, best_test),
        ]:
            if tr_r["calmar"] != 0:
                decay_pct = (te_r["calmar"] - tr_r["calmar"]) / abs(tr_r["calmar"]) * 100
            else:
                decay_pct = 0.0
            print(f"  {label:<34}  "
                  f"{tr_r['total_r']:>+6.1f}R / {tr_r['calmar']:>5.2f}  "
                  f"  {te_r['total_r']:>+6.1f}R / {te_r['calmar']:>5.2f}  "
                  f"  {decay_pct:>+8.1f}%")

        # Win-rate shift
        wr_cur  = (cur_test["win_rate"]  - cur_train["win_rate"])  * 100
        wr_best = (best_test["win_rate"] - best_train["win_rate"]) * 100
        print(f"\n  Win-rate shift train->test:  current={wr_cur:+.1f}pp  WF={wr_best:+.1f}pp")

        # Verdict
        cal_cur_decay  = abs((cur_test["calmar"]  - cur_train["calmar"])  / (cur_train["calmar"]  or 1))
        cal_best_decay = abs((best_test["calmar"] - best_train["calmar"]) / (best_train["calmar"] or 1))
        print()
        if cal_best_decay < cal_cur_decay:
            print(f"  [OK] WF params show less Calmar decay ({cal_best_decay*100:.1f}% vs {cal_cur_decay*100:.1f}%)")
            print(f"       Current params likely have some 4-year in-sample bias.")
        else:
            print(f"  [OK] Current params hold up as well as WF-selected params.")
            print(f"       No meaningful overfitting detected at this grid resolution.")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "walk_forward_results.json"
    out_path.write_text(
        json.dumps({
            "train_period": f"{TRAIN_START} to {TRAIN_END}",
            "test_period":  f"{TEST_START} to {TEST_END}",
            "current_params": cur_params,
            "current_segmented": {"train": cur_train, "test": cur_test},
            "grid_top": [{"params": r["params"], "train": r["train"],
                          "test": r.get("test")} for r in top_n],
            "full_grid_train": sorted(
                [{"params": r["params"], "train": r["train"]} for r in grid_results],
                key=lambda x: x["train"]["calmar"], reverse=True,
            ),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
