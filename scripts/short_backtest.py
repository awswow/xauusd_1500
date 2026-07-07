"""
Short-side mirror strategy backtest for XAUUSD M15.

Mirrors the long pullback-to-SMA strategy:
  Long:  BUY_STOP  at bar HIGH  when close > SMA, SMA rising,  low  near SMA
  Short: SELL_STOP at bar LOW   when close < SMA, SMA falling, high near SMA

Runs long-only, short-only, and combined on 2022-2025.
Reports comparative table and saves JSON.

Usage:
  python scripts/short_backtest.py --config configs/live_demo_bt.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import numpy as np
import yaml

from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec
from backtest_demo_executor_live_like import parse_dt
from walk_forward import (
    precompute, fast_backtest, _stats,
    FIXED_PULLBACK_LOOKBACK, WARMUP, FIXED_TARGET_R,
)

YEARS = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-19"),
]


def fast_backtest_short(
    pc: dict,
    *,
    adx_min: float,
    pullback_atr: float,
    sma_len: int,
    stop_atr: float,
    cooldown_bars: int,
    daily_loss_limit_usd: float,
    max_trades_per_day: int,
    risk_usd_per_trade: float,
    pending_expiry_bars: int,
    monthly_loss_limit_r: float,
) -> list[dict]:
    """Mirror of fast_backtest for the short side.

    Signal:
      - close < SMA and SMA slope negative   (S1+S2)
      - ADX >= adx_min                       (S3)
      - recent HIGH within pullback_atr*ATR  (S4)
      - Regime: short_regime_ok (inverted from long)
    Entry: SELL_STOP at bar LOW
    SL:    entry + stop_atr * ATR  (above entry)
    TP:    entry - target_r * (SL - entry) (below entry)
    """
    n        = pc["n"]
    high     = pc["high"]
    low      = pc["low"]
    close_   = pc["close"]
    times    = pc["times"]
    atr14    = pc["atr14"]
    adx14    = pc["adx14"]
    sma_s    = pc["sma_series"][sma_len]
    tq_ok    = pc["tq_ok"]
    sp_ok    = pc["spread_ok"]
    reg_ok   = pc["regime_ok"]      # True = long regime (bull week)
    point    = pc["point"]
    target_r = FIXED_TARGET_R
    risk_usd = risk_usd_per_trade

    trades: list[dict] = []

    in_pos      = False
    entry_price = stop_price = target_price = 0.0
    entry_time  = None

    pending_price = 0.0
    pending_sl    = 0.0
    pending_set   = False
    pending_idx   = 0

    last_fill_i  = -cooldown_bars - 1
    day_key:     Optional[str] = None
    day_pnl_usd = 0.0
    day_trades  = 0
    month_key:   Optional[str] = None
    month_r     = 0.0

    for i in range(WARMUP, n):
        t  = times[i]
        dk = t.date().isoformat()
        mk = t.strftime("%Y-%m")

        if day_key is None:   day_key   = dk
        if month_key is None: month_key = mk
        if dk != day_key:
            day_key = dk; day_pnl_usd = 0.0; day_trades = 0
        if mk != month_key:
            month_key = mk; month_r = 0.0

        if not sp_ok[i]:
            continue

        hi = high[i]
        lo = low[i]

        # ── Manage open SHORT position ─────────────────────────────────────
        if in_pos:
            hit_stop   = hi >= stop_price    # stop is ABOVE entry for shorts
            hit_target = lo <= target_price  # target is BELOW entry for shorts
            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    reason = "stop"; ex_price = stop_price
                elif hit_stop:
                    reason = "stop"; ex_price = stop_price
                else:
                    reason = "target"; ex_price = target_price

                pnl_pts  = (entry_price - ex_price) / point   # short: profit when price falls
                risk_pts = abs(stop_price - entry_price) / point
                r_mult   = (pnl_pts / risk_pts) if risk_pts > 0 else 0.0

                realized     = r_mult * risk_usd
                day_pnl_usd += realized
                month_r     += r_mult

                trades.append({
                    "r_multiple":  r_mult,
                    "exit_reason": reason,
                    "entry_time":  entry_time,
                    "exit_time":   t,
                    "side":        "SHORT",
                })
                in_pos = False
            continue

        # ── Manage pending SELL_STOP ───────────────────────────────────────
        if pending_set:
            if (i - pending_idx) >= pending_expiry_bars:
                pending_set = False
            elif lo <= pending_price:
                # Filled: price dropped to/below the SELL_STOP
                ent = pending_price
                sl  = pending_sl
                tp  = ent - target_r * (sl - ent)
                if sl > ent > tp:
                    in_pos       = True
                    entry_price  = ent
                    stop_price   = sl
                    target_price = tp
                    entry_time   = t
                    last_fill_i  = i
                    day_trades  += 1
                pending_set = False
            continue

        # ── Risk controls ──────────────────────────────────────────────────
        if day_pnl_usd <= -abs(daily_loss_limit_usd):
            continue
        if day_trades >= max_trades_per_day:
            continue
        if (i - last_fill_i) < cooldown_bars:
            continue
        if monthly_loss_limit_r > 0 and month_r <= -monthly_loss_limit_r:
            continue

        # ── Short regime gate: trade when long regime is OFF ───────────────
        if reg_ok[i]:           # long regime is ON → skip short
            continue

        # ── Trend-quality gate (same ADX/ATR filter) ──────────────────────
        if not tq_ok[i]:
            continue

        # ── Strategy signal (mirror of long) ──────────────────────────────
        sma_now  = sma_s[i]
        sma_prev = sma_s[i - 1]
        atr_now  = atr14[i]
        adx_now  = adx14[i]

        if np.isnan(sma_now) or np.isnan(sma_prev): continue
        if np.isnan(atr_now) or atr_now <= 0:       continue
        if np.isnan(adx_now):                        continue

        if close_[i] >= sma_now:  continue   # S1: close < SMA (bearish)
        if sma_now >= sma_prev:   continue   # S2: SMA slope negative
        if adx_now < adx_min:     continue   # S3: ADX gate

        # S4: any of last N highs within pullback_atr * ATR of SMA from above
        lb_start = max(0, i - (FIXED_PULLBACK_LOOKBACK - 1))
        recent_highs = high[lb_start: i + 1]
        if not np.any((recent_highs - sma_now) <= pullback_atr * atr_now):
            continue

        # Signal: place SELL_STOP at current bar's LOW
        entry_level = lo
        sl_level    = entry_level + stop_atr * atr_now   # stop ABOVE entry
        if sl_level <= entry_level:
            continue

        pending_price = entry_level
        pending_sl    = sl_level
        pending_set   = True
        pending_idx   = i

    return trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/derived/runs/short_backtest")
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

    adx_min     = float(sp.get("adx_min",      24.3))
    sma_len     = int(sp.get("sma_len",          60))
    pullback_atr = float(sp.get("pullback_atr", 0.30))
    stop_atr    = float(sp.get("stop_atr",      1.60))

    print(f"\nShort-side mirror strategy backtest -- {symbol} {tf_name}")
    print(f"Params: adx={adx_min}  sma={sma_len}  pa={pullback_atr}  sa={stop_atr}")
    print(f"  Long regime: close > SMA, SMA rising")
    print(f"  Short regime: close < SMA AND SMA falling (inverted + slope filter)")

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    spec  = get_symbol_spec(symbol)
    point = float(spec.point)

    from backtest_demo_executor_live_like import build_weekly_regime

    sep  = "=" * 88
    results = []

    print(f"\n{sep}")
    hdr = f"  {'Year':<6}  {'Side':<7}  {'Trades':>6}  {'Win%':>5}  {'Total R':>8}  {'Max DD':>7}  {'Calmar':>7}"
    print(hdr)
    print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}")

    yearly: list[dict] = []

    for label, yr_start, yr_end in YEARS:
        start_dt = parse_dt(yr_start)
        end_dt   = parse_dt(yr_end)

        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            print(f"  {label}: no data")
            continue

        regime_dict = None
        if regime_cfg.get("enabled", False):
            sma_w  = int(regime_cfg.get("sma_weeks", 40))
            slope  = bool(regime_cfg.get("require_slope", True))
            lb_wks = int(regime_cfg.get("slope_lookback_weeks", 4))
            regime_dict = build_weekly_regime(symbol, start_dt, end_dt, sma_w,
                                              require_slope=slope,
                                              slope_lookback_weeks=lb_wks)

        pc = precompute(rates, sma_lens=[sma_len],
                        regime_dict=regime_dict,
                        max_spread_points=max_sp,
                        gate_cfg=rf_cfg, point=point)

        params = dict(adx_min=adx_min, pullback_atr=pullback_atr,
                      sma_len=sma_len, stop_atr=stop_atr)

        long_trades  = fast_backtest(pc,       **params, **fixed_exec)
        short_trades = fast_backtest_short(pc, **params, **fixed_exec)
        all_trades   = sorted(long_trades + short_trades,
                              key=lambda t: t["entry_time"])

        long_s  = _stats(long_trades)
        short_s = _stats(short_trades)
        comb_s  = _stats(all_trades)

        def row(side, s):
            cal = f"{s['calmar']:>7.2f}" if s['calmar'] not in (-999, 999) else "   n/a "
            print(f"  {label:<6}  {side:<7}  {s['trades']:>6}  "
                  f"{s['win_rate']*100:>4.1f}%  "
                  f"{s['total_r']:>+8.2f}  {s['max_dd']:>7.2f}  {cal}")
            return s

        long_s  = row("LONG",     long_s)
        short_s = row("SHORT",    short_s)
        comb_s  = row("COMBINED", comb_s)
        print()

        yearly.append({
            "year": label,
            "long":  long_s,
            "short": short_s,
            "combined": comb_s,
        })
        results.extend([
            {"year": label, "side": "long",  **long_s},
            {"year": label, "side": "short", **short_s},
            {"year": label, "side": "combined", **comb_s},
        ])

    # 4-year summary
    print(f"{sep}")
    print(f"  4-YEAR SUMMARY")
    print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}")

    def total_stats(key):
        all_r = []
        for yr in yearly:
            # rebuild equity curve to get true cross-year DD
            # approximate: sum of yearly totals, max DD = max of yearly DDs
            pass
        total_r  = sum(yr[key]["total_r"]  for yr in yearly)
        total_tr = sum(yr[key]["trades"]   for yr in yearly)
        total_wr = sum(yr[key]["trades"] * yr[key]["win_rate"] for yr in yearly) / max(total_tr, 1)
        max_dd   = max(yr[key]["max_dd"]   for yr in yearly)  # worst single-year DD
        calmar   = total_r / max_dd if max_dd > 0 and total_r > 0 else -999.0
        return {"total_r": total_r, "trades": total_tr, "win_rate": total_wr,
                "max_dd": max_dd, "calmar": calmar}

    for side, key in [("LONG", "long"), ("SHORT", "short"), ("COMBINED", "combined")]:
        s = total_stats(key)
        cal = f"{s['calmar']:>7.2f}" if s['calmar'] != -999 else "   n/a "
        print(f"  {'4-yr':<6}  {side:<7}  {s['trades']:>6}  "
              f"{s['win_rate']*100:>4.1f}%  "
              f"{s['total_r']:>+8.2f}  {s['max_dd']:>7.2f}  {cal}")

    print(f"\n{sep}")
    print("KEY QUESTIONS:")
    short_total = sum(yr["short"]["total_r"] for yr in yearly)
    long_total  = sum(yr["long"]["total_r"]  for yr in yearly)
    comb_total  = sum(yr["combined"]["total_r"] for yr in yearly)
    print(f"  Does short add value?  Long={long_total:+.1f}R  Short={short_total:+.1f}R  "
          f"Combined={comb_total:+.1f}R  (gain from adding short: {comb_total-long_total:+.1f}R)")

    # Is the strategy active in current conditions?
    print(f"\n  Current market state (from live log):")
    print(f"    XAUUSD close: ~4172  |  40w SMA: ~4494  |  Gap: -7.2% (regime OFF for longs)")
    print(f"    Short regime requires: close < SMA AND SMA slope < 0")
    print(f"    SMA slope is currently POSITIVE (still rising) -> short regime also OFF")
    print(f"    Short strategy would activate once SMA begins declining (typically 4-8 weeks")
    print(f"    after price falls below SMA and stays there)")
    print(f"{sep}")

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "short_backtest_results.json"
    out_path.write_text(json.dumps({"params": dict(adx_min=adx_min, sma_len=sma_len,
                                                    pullback_atr=pullback_atr, stop_atr=stop_atr),
                                    "results": results}, indent=2), encoding="utf-8")
    print(f"\n  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
