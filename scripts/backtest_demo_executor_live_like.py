# scripts/backtest_demo_executor_live_like.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import MetaTrader5 as mt5
import pandas as pd
import yaml

from xauusd100.engine.models import Bar, Side
from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec
from xauusd100.strategy.pullback_trend import PullbackTrendStrategy, PullbackTrendParams
from xauusd100.strategy.base import StrategyContext
from xauusd100.metrics import equity_curve_from_r, max_drawdown_stats

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


def parse_dt(s: str) -> datetime:
    if len(s) == 10:
        s = s + "T00:00:00"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def rates_to_bars(rates) -> list[Bar]:
    bars: list[Bar] = []
    for r in rates:
        t = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
        bars.append(
            Bar(
                time_utc=t,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
            )
        )
    return bars


@dataclass(frozen=True)
class BacktestSpreadCfg:
    spread_source: str = "fixed"  # fixed | rates
    spread_points: int = 60
    spread_mode: str = "half"  # half | full


def _spread_points_for_bar(*, rates, i: int, cfg: BacktestSpreadCfg) -> int:
    if (cfg.spread_source or "").lower() == "rates":
        names = set(rates.dtype.names or [])
        if "spread" in names:
            try:
                return int(rates[i]["spread"])
            except Exception:
                pass
    return int(cfg.spread_points)


def _apply_spread_fill(price: float, *, side: Side, spread_points: int, point: float, mode: str) -> float:
    s = float(spread_points) * float(point)
    k = 0.5 if (mode or "half").lower().strip() == "half" else 1.0
    return float(price + k * s) if side == Side.BUY else float(price - k * s)


def _apply_spread_fill_exit(price: float, *, side: Side, spread_points: int, point: float, mode: str) -> float:
    s = float(spread_points) * float(point)
    k = 0.5 if (mode or "half").lower().strip() == "half" else 1.0
    return float(price - k * s) if side == Side.BUY else float(price + k * s)


# =========================
# Trend Quality Gate helpers (copied from backtest_mt5_history.py)
# =========================
def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _wilder_rma(values: list[float], length: int) -> list[float]:
    """Wilder's RMA (EMA with alpha=1/length). Returns full-length list with leading NaNs."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if not values:
        return []
    if len(values) < length:
        return [float("nan")] * len(values)

    seed = sum(values[:length]) / length
    out = [float("nan")] * (length - 1) + [seed]

    alpha = 1.0 / length
    prev = seed
    for v in values[length:]:
        prev = prev + alpha * (v - prev)
        out.append(prev)
    return out


def compute_atr_series(bars: list[Bar], length: int) -> list[float]:
    """ATR series aligned to bars length. Leading values will be NaN until enough history."""
    n = len(bars)
    if n < 2:
        return [float("nan")] * n

    tr_vals: list[float] = []
    for i in range(1, n):
        tr_vals.append(_true_range(float(bars[i].high), float(bars[i].low), float(bars[i - 1].close)))

    atr = _wilder_rma(tr_vals, length=length)  # aligned to bars[1:]
    return [float("nan")] + atr


def compute_adx_series(bars: list[Bar], length: int) -> list[float]:
    """
    Wilder ADX implementation that is NaN-safe and does not propagate NaNs.
    Returns a list aligned to bars length, with NaNs until warmup is complete.
    """
    n = len(bars)
    if length <= 0:
        raise ValueError("length must be > 0")
    if n < 2:
        return [float("nan")] * n

    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    closes = [float(b.close) for b in bars]

    tr: list[float] = [0.0] * (n - 1)
    plus_dm: list[float] = [0.0] * (n - 1)
    minus_dm: list[float] = [0.0] * (n - 1)

    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm[i - 1] = up if (up > down and up > 0) else 0.0
        minus_dm[i - 1] = down if (down > up and down > 0) else 0.0
        tr[i - 1] = _true_range(highs[i], lows[i], closes[i - 1])

    def wilder_smooth(x: list[float], period: int) -> list[float]:
        m = len(x)
        out = [float("nan")] * m
        if m < period:
            return out
        seed = sum(x[:period])
        out[period - 1] = seed
        prev = seed
        for k in range(period, m):
            prev = prev - (prev / period) + x[k]
            out[k] = prev
        return out

    tr_s = wilder_smooth(tr, length)
    pdm_s = wilder_smooth(plus_dm, length)
    mdm_s = wilder_smooth(minus_dm, length)

    dx: list[float] = [float("nan")] * (n - 1)
    for i in range(n - 1):
        trv = tr_s[i]
        if trv != trv or trv <= 0:
            continue
        pdi = 100.0 * (pdm_s[i] / trv) if pdm_s[i] == pdm_s[i] else float("nan")
        mdi = 100.0 * (mdm_s[i] / trv) if mdm_s[i] == mdm_s[i] else float("nan")
        if pdi != pdi or mdi != mdi:
            continue
        denom = pdi + mdi
        if denom <= 0:
            continue
        dx[i] = 100.0 * abs(pdi - mdi) / denom

    adx_out = [float("nan")] * (n - 1)

    for start in range(0, (n - 1) - length + 1):
        window = dx[start : start + length]
        if all(v == v for v in window):
            seed = sum(window) / length
            idx_seed = start + length - 1
            adx_out[idx_seed] = seed
            prev = seed
            for k in range(idx_seed + 1, n - 1):
                if dx[k] != dx[k]:
                    adx_out[k] = prev
                    continue
                prev = (prev * (length - 1) + dx[k]) / length
                adx_out[k] = prev
            break

    return [float("nan")] + adx_out


def _last_finite(xs: list[float]) -> Optional[float]:
    for v in reversed(xs):
        if v == v:
            return float(v)
    return None


def _finite_at_offset_from_end(xs: list[float], offset_from_end: int) -> Optional[float]:
    if not xs:
        return None
    j = len(xs) - 1 - int(offset_from_end)
    j = min(max(j, 0), len(xs) - 1)
    for k in range(j, -1, -1):
        v = xs[k]
        if v == v:
            return float(v)
    return None


def trend_quality_gate_ok_debug(
    hist: list[Bar],
    *,
    adx_len: int,
    atr_len: int,
    adx_min: float,
    adx_rise_bars: int,
    atr_ref_window: int,
    atr_quantile: float,
) -> tuple[bool, str]:
    need = max(adx_len * 3 + 5, atr_len * 3 + 5, atr_ref_window + 5)
    if len(hist) < need:
        return False, f"warmup(len={len(hist)} need={need})"

    adx = compute_adx_series(hist, length=int(adx_len))
    atr = compute_atr_series(hist, length=int(atr_len))

    adx_now = _last_finite(adx)
    atr_now = _last_finite(atr)
    if adx_now is None:
        return False, "adx_now_none"
    if atr_now is None:
        return False, "atr_now_none"

    if adx_now < float(adx_min):
        return False, f"adx_min(adx_now={adx_now:.2f} < {float(adx_min):.2f})"

    rise_bars = int(adx_rise_bars)
    if rise_bars > 0:
        adx_prev = _finite_at_offset_from_end(adx, offset_from_end=rise_bars)
        if adx_prev is None:
            return False, "adx_prev_none"
        if not (adx_now > adx_prev):
            return False, f"adx_not_rising(adx_now={adx_now:.2f} adx_prev={adx_prev:.2f})"

    w = atr[-int(atr_ref_window) :]
    w = [float(x) for x in w if x == x]
    if len(w) == 0:
        return False, "atr_window_empty"
    q = float(pd.Series(w).quantile(float(atr_quantile)))
    if atr_now < q:
        return False, f"atr_below_q(atr_now={atr_now:.4f} q={q:.4f})"

    return True, "ok"


def simulate_live_like_executor(
    *,
    bars: list[Bar],
    rates,
    symbol: str,
    timeframe: str,
    strategy: PullbackTrendStrategy,
    point: float,
    spread_cfg: BacktestSpreadCfg,
    intrabar_policy: str = "conservative",
    entry_confirmation: str = "prev_high_break",  # matches your backtest + demo logic

    # A/B toggles
    use_spread_gate: bool = True,
    use_trend_gate: bool = False,
    gate_cfg: Optional[dict] = None,

    # live-exec behavior knobs:
    max_spread_points_live_gate: Optional[int] = 45,  # THIS uses rates[i]["spread"]
    pending_expiry_bars: int = 4,
    convert_to_market_on_breakout: bool = True,
    cooldown_bars: int = 16,
    daily_loss_limit_usd: float = 30.0,  # approximate using R below
    max_trades_per_day: int = 2,
    risk_usd_per_trade: float = 10.0,
    warmup_min_bars: int = 120,
) -> pd.DataFrame:
    """
    Live-like backtest of demo_executor_mt5.py behavior.

    - Spread gate: uses historical `rates[i]["spread"]` (requires MT5 'spread' column).
    - Trend gate: uses risk_filters.trend_quality_gate (same implementation as baseline backtest).
    - Daily loss limit in USD is approximated using R * risk_usd_per_trade (we don't model lot sizing here).
    """

    def to_points(price_delta: float) -> float:
        return float(price_delta / point)

    def resolve_intrabar_exit(b: Bar, side_: Side, stop_: float, target_: float) -> Tuple[Optional[str], Optional[float]]:
        lo, hi = float(b.low), float(b.high)
        optimistic = (intrabar_policy or "conservative").lower().strip() == "optimistic"

        if side_ == Side.BUY:
            hit_stop = lo <= stop_
            hit_target = hi >= target_
            if not hit_stop and not hit_target:
                return None, None
            if hit_stop and hit_target:
                return ("target", target_) if optimistic else ("stop", stop_)
            return ("stop", stop_) if hit_stop else ("target", target_)

        hit_stop = hi >= stop_
        hit_target = lo <= target_
        if not hit_stop and not hit_target:
            return None, None
        if hit_stop and hit_target:
            return ("target", target_) if optimistic else ("stop", stop_)
        return ("stop", stop_) if hit_stop else ("target", target_)

    names = set(rates.dtype.names or [])
    if "spread" not in names:
        raise ValueError("rates do not contain 'spread' column; cannot do live-like spread gate.")

    # Trend gate config (from YAML risk_filters)
    g = gate_cfg or {}

    in_pos = False
    side: Optional[Side] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    entry_time: Optional[datetime] = None
    entry_bar_index: Optional[int] = None
    entry_type: Optional[str] = None  # "stop" | "market"

    rows: list[dict] = []

    # Pending order simulation (BUY_STOP only)
    pending_active = False
    pending_price: Optional[float] = None
    pending_sl: Optional[float] = None
    pending_tp: Optional[float] = None
    pending_created_index: Optional[int] = None

    # Cooldown anchored to fills
    last_fill_bar_index: Optional[int] = None

    # Daily controls (approximate using R)
    day_key: Optional[str] = None
    day_realized_usd = 0.0
    day_trades = 0

    bar_index = 0

    for i in range(2, len(bars)):
        bar_index += 1
        b = bars[i]

        # Warmup (match live executor needing enough bars for indicators)
        if i < int(warmup_min_bars):
            continue

        # Day rollover (UTC date)
        dk = b.time_utc.date().isoformat()
        if day_key is None:
            day_key = dk
        if dk != day_key:
            day_key = dk
            day_realized_usd = 0.0
            day_trades = 0

        # Spread gate (LIVE-LIKE): use historical per-bar spread
        if use_spread_gate and max_spread_points_live_gate is not None:
            cur_sp = int(rates[i]["spread"])
            if cur_sp > int(max_spread_points_live_gate):
                continue

        # If in position: manage exits
        fill_sp_points = _spread_points_for_bar(rates=rates, i=i, cfg=spread_cfg)

        if in_pos:
            assert side and entry is not None and stop is not None and target is not None and entry_time is not None

            exit_reason, exit_level = resolve_intrabar_exit(b, side, stop, target)
            if exit_reason and exit_level is not None:
                exit_fill = _apply_spread_fill_exit(
                    exit_level, side=side, spread_points=fill_sp_points, point=point, mode=spread_cfg.spread_mode
                )

                pnl_price = (exit_fill - entry) if side == Side.BUY else (entry - exit_fill)
                pnl_points = to_points(pnl_price)

                risk_points = abs(to_points(entry - stop)) if side == Side.BUY else abs(to_points(stop - entry))
                r_mult = float(pnl_points / risk_points) if risk_points > 0 else 0.0

                # Approx realized USD for daily controls
                realized_usd = float(r_mult) * float(risk_usd_per_trade)
                day_realized_usd += realized_usd

                rows.append(
                    {
                        "entry_time_utc": entry_time.isoformat(),
                        "exit_time_utc": b.time_utc.isoformat(),
                        "side": side.value,
                        "entry_type": entry_type or "unknown",
                        "entry_price": float(entry),
                        "exit_price": float(exit_fill),
                        "stop_price": float(stop),
                        "target_price": float(target),
                        "spread_points": int(fill_sp_points),
                        "pnl_points": float(pnl_points),
                        "risk_points": float(risk_points),
                        "r_multiple": float(r_mult),
                        "exit_reason": exit_reason,
                        "holding_bars": int(bar_index - (entry_bar_index or bar_index)),
                    }
                )

                # reset position
                in_pos = False
                side = None
                entry = stop = target = None
                entry_time = None
                entry_bar_index = None
                entry_type = None
            continue

        # If pending exists: expire or fill
        if pending_active:
            assert pending_created_index is not None and pending_price is not None

            # Expiry
            if (bar_index - pending_created_index) >= int(pending_expiry_bars):
                pending_active = False
                pending_price = pending_sl = pending_tp = None
                pending_created_index = None
                continue

            # Fill condition for BUY_STOP: current bar trades through pending_price
            if float(b.high) >= float(pending_price):
                entry_level = float(pending_price)
                entry_fill = _apply_spread_fill(
                    entry_level, side=Side.BUY, spread_points=fill_sp_points, point=point, mode=spread_cfg.spread_mode
                )

                if pending_sl is None or pending_tp is None:
                    pending_active = False
                    pending_price = pending_sl = pending_tp = None
                    pending_created_index = None
                    continue
                if not (float(pending_sl) < entry_fill < float(pending_tp)):
                    pending_active = False
                    pending_price = pending_sl = pending_tp = None
                    pending_created_index = None
                    continue

                # Fill opens position
                in_pos = True
                side = Side.BUY
                entry = entry_fill
                stop = float(pending_sl)
                target = float(pending_tp)
                entry_time = b.time_utc
                entry_bar_index = bar_index
                entry_type = "stop"
                last_fill_bar_index = bar_index

                day_trades += 1

                # clear pending
                pending_active = False
                pending_price = pending_sl = pending_tp = None
                pending_created_index = None

            continue

        # Daily controls (like demo executor)
        if day_realized_usd <= -abs(float(daily_loss_limit_usd)):
            continue
        if day_trades >= int(max_trades_per_day):
            continue

        # Cooldown anchored to fills (bars)
        if cooldown_bars and last_fill_bar_index is not None:
            if (bar_index - last_fill_bar_index) < int(cooldown_bars):
                continue

        # Strategy decision on last CLOSED bar (same logic as demo executor)
        hist = bars[:i]  # bars up to previous closed
        ctx = StrategyContext(symbol=symbol, timeframe=timeframe, bars=hist, bar_index=len(hist), meta={"point": point})
        decision = strategy.on_bar(ctx)

        if decision is None:
            continue
        if decision.side != Side.BUY:
            continue

        # Trend Quality Gate (optional) — blocks signals pre-order, like a live filter
        if use_trend_gate and bool(g.get("trend_quality_gate", False)):
            ok, _reason = trend_quality_gate_ok_debug(
                hist,
                adx_len=int(g.get("tq_adx_len", 14)),
                atr_len=int(g.get("tq_atr_len", 14)),
                adx_min=float(g.get("tq_adx_min", 18.0)),
                adx_rise_bars=int(g.get("tq_adx_rise_bars", 2)),
                atr_ref_window=int(g.get("tq_atr_ref_window", 120)),
                atr_quantile=float(g.get("tq_atr_quantile", 0.35)),
            )
            if not ok:
                continue

        sl = float(decision.stop_price)
        tp = float(decision.target_price)

        prev = bars[i - 1]
        stop_entry = float(prev.high)

        # If breakout already happened, convert to market
        if convert_to_market_on_breakout and float(b.open) >= stop_entry:
            entry_fill = _apply_spread_fill(
                float(b.open), side=Side.BUY, spread_points=fill_sp_points, point=point, mode=spread_cfg.spread_mode
            )
            if not (sl < entry_fill < tp):
                continue

            in_pos = True
            side = Side.BUY
            entry = entry_fill
            stop = sl
            target = tp
            entry_time = b.time_utc
            entry_bar_index = bar_index
            entry_type = "market"
            last_fill_bar_index = bar_index

            day_trades += 1
            continue

        # Otherwise place pending BUY_STOP
        pending_active = True
        pending_price = stop_entry
        pending_sl = sl
        pending_tp = tp
        pending_created_index = bar_index

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_utc", required=True)
    ap.add_argument("--end_utc", required=True)
    ap.add_argument("--out", default="data/derived/runs/bt_demo_live_like")

    # A/B toggles
    ap.add_argument("--use_spread_gate", type=int, default=1)  # 1/0
    ap.add_argument("--use_trend_gate", type=int, default=0)  # 1/0

    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    tf = TF_MAP[timeframe]

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()

    spec = get_symbol_spec(symbol)
    point = float(spec.point)

    start = parse_dt(args.start_utc)
    end = parse_dt(args.end_utc)

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        raise SystemExit(f"No rates returned for {symbol} {timeframe} in {start}..{end}.")

    if "spread" not in (rates.dtype.names or ()):
        raise SystemExit("rates do not include 'spread' column; cannot run live-like gate.")

    bars = rates_to_bars(rates)

    sp = dict(cfg["strategy"]["params"])
    sp.pop("min_bars_between_entries", None)
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    bt_cfg = cfg.get("backtest", {}) or {}
    ex_cfg = cfg.get("execution", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    risk_filters = cfg.get("risk_filters", {}) or {}

    spread_cfg = BacktestSpreadCfg(
        spread_source=str(bt_cfg.get("spread_source", "fixed") or "fixed"),
        spread_points=int(bt_cfg.get("spread_points", 60) or 60),
        spread_mode=str(bt_cfg.get("spread_mode", "half") or "half"),
    )

    use_spread_gate = bool(int(args.use_spread_gate))
    use_trend_gate = bool(int(args.use_trend_gate))

    trades = simulate_live_like_executor(
        bars=bars,
        rates=rates,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        point=point,
        spread_cfg=spread_cfg,
        intrabar_policy=str(bt_cfg.get("intrabar_policy", "conservative") or "conservative"),
        entry_confirmation=str(bt_cfg.get("entry_confirmation", "prev_high_break") or "prev_high_break"),
        use_spread_gate=use_spread_gate,
        use_trend_gate=use_trend_gate,
        gate_cfg=risk_filters,
        max_spread_points_live_gate=int(ex_cfg.get("max_spread_points", 45)) if ex_cfg.get("max_spread_points", None) is not None else None,
        pending_expiry_bars=int(ex_cfg.get("pending_expiry_bars", 4) or 4),
        convert_to_market_on_breakout=bool(ex_cfg.get("convert_to_market_on_breakout", True)),
        cooldown_bars=int(risk_cfg.get("cooldown_bars", 0) or 0),
        daily_loss_limit_usd=float(ex_cfg.get("daily_loss_limit_usd", 30.0) or 30.0),
        max_trades_per_day=int(ex_cfg.get("max_trades_per_day", 2) or 2),
        risk_usd_per_trade=float(ex_cfg.get("risk_usd_per_trade", 10.0) or 10.0),
        warmup_min_bars=int(bt_cfg.get("warmup_min_bars", 120) or 120),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "ledger.csv"
    trades.to_csv(trades_path, index=False)

    r = trades["r_multiple"].astype(float) if len(trades) else pd.Series(dtype=float)
    equity_r = equity_curve_from_r(r)
    dd = max_drawdown_stats(equity_r)

    equity_path = out_dir / "equity_r.csv"
    pd.DataFrame({"equity_r": equity_r}).to_csv(equity_path, index=False)

    stats = {
        "mode": "demo_executor_live_like_backtest",
        "symbol": symbol,
        "timeframe": timeframe,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "point": point,
        "spread_cfg": {
            "spread_source": spread_cfg.spread_source,
            "spread_points": spread_cfg.spread_points,
            "spread_mode": spread_cfg.spread_mode,
        },
        "ab": {
            "use_spread_gate": use_spread_gate,
            "use_trend_gate": use_trend_gate,
        },
        "live_gate": {
            "max_spread_points": ex_cfg.get("max_spread_points", None),
            "pending_expiry_bars": ex_cfg.get("pending_expiry_bars", None),
            "convert_to_market_on_breakout": ex_cfg.get("convert_to_market_on_breakout", None),
        },
        "risk_filters": risk_filters,
        "trades": int(len(trades)),
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "median_r": float(r.median()) if len(r) else 0.0,
        "total_r": float(r.sum()) if len(r) else 0.0,
        "max_drawdown_r": float(dd.max_drawdown),
        "dd_peak_index": int(dd.peak_index),
        "dd_trough_index": int(dd.trough_index),
        "dd_recovery_index": None if dd.recovery_index is None else int(dd.recovery_index),
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict() if len(trades) and "exit_reason" in trades.columns else {},
        "entry_type_counts": trades["entry_type"].value_counts().to_dict() if len(trades) and "entry_type" in trades.columns else {},
    }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Wrote {trades_path}")
    print(f"Wrote {equity_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
