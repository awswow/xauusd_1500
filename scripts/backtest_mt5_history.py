# scripts/backtest_mt5_history.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Allow running without "pip install -e ."
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import MetaTrader5 as mt5
import pandas as pd
import yaml

from xauusd100.engine.models import Bar, Side
from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec
from xauusd100.strategy.pullback_trend import PullbackTrendStrategy, PullbackTrendParams
from xauusd100.strategy.base import StrategyContext

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
    spread_mode: str = "half"     # half | full


def _spread_points_for_bar(*, rates, i: int, cfg: BacktestSpreadCfg) -> int:
    if cfg.spread_source.lower() == "rates":
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


def _estimate_tick_value_and_size(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None, None
    tick_value = getattr(info, "trade_tick_value", None)
    tick_size = getattr(info, "trade_tick_size", None)
    if tick_size is None or tick_size == 0:
        tick_size = getattr(info, "point", None)
    try:
        tick_value = float(tick_value) if tick_value is not None else None
    except Exception:
        tick_value = None
    try:
        tick_size = float(tick_size) if tick_size is not None else None
    except Exception:
        tick_size = None
    return tick_value, tick_size


def simulate(
    *,
    bars: list[Bar],
    rates,
    symbol: str,
    timeframe: str,
    strategy: PullbackTrendStrategy,
    point: float,
    spread_cfg: BacktestSpreadCfg,
    intrabar_policy: str = "conservative",
    time_stop_bars: int = 0,
    cooldown_bars: int = 0,
    entry_confirmation: str = "prev_high_break",  # prev_high_break | open
) -> pd.DataFrame:
    if point <= 0:
        raise ValueError(f"Invalid point={point}. Check symbol spec.")

    tick_value, tick_size = _estimate_tick_value_and_size(symbol)

    def to_points(price_delta: float) -> float:
        return float(price_delta / point)

    def loss_usd_per_lot_for_stop(entry_price: float, stop_price: float) -> Optional[float]:
        if tick_value is None or tick_size is None or tick_size <= 0:
            return None
        dist = abs(entry_price - stop_price)
        ticks = dist / tick_size
        return float(ticks * tick_value)

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

    in_pos = False
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    entry_time: Optional[datetime] = None
    side: Optional[Side] = None

    entry_meta: Dict[str, Any] = {}
    entry_bar_index: Optional[int] = None
    last_entry_bar_index: Optional[int] = None

    rows: list[dict] = []
    bar_index = 0

    for i in range(2, len(bars)):
        bar_index += 1

        hist = bars[:i]
        ctx = StrategyContext(symbol=symbol, timeframe=timeframe, bars=hist, bar_index=bar_index, meta={"point": point})

        b = bars[i]
        sp_points = _spread_points_for_bar(rates=rates, i=i, cfg=spread_cfg)

        # -----------------
        # Manage open trade
        # -----------------
        if in_pos:
            assert side is not None and entry is not None and stop is not None and target is not None and entry_time is not None

            exit_reason, exit_level = resolve_intrabar_exit(b, side, stop, target)

            if exit_reason is None and time_stop_bars and entry_bar_index is not None:
                if (bar_index - entry_bar_index) >= time_stop_bars:
                    exit_reason = "time_stop"
                    exit_level = float(b.close)

            if exit_reason and exit_level is not None:
                exit_fill = _apply_spread_fill_exit(exit_level, side=side, spread_points=sp_points, point=point, mode=spread_cfg.spread_mode)

                pnl_price = (exit_fill - entry) if side == Side.BUY else (entry - exit_fill)
                pnl_points = to_points(pnl_price)

                risk_points = abs(to_points(entry - stop)) if side == Side.BUY else abs(to_points(stop - entry))
                r_mult = float(pnl_points / risk_points) if risk_points > 0 else 0.0

                loss_per_lot = loss_usd_per_lot_for_stop(entry, stop)
                pnl_usd_1lot = float(r_mult * loss_per_lot) if (loss_per_lot is not None and risk_points > 0) else None

                rows.append(
                    {
                        "entry_time_utc": entry_time.isoformat(),
                        "exit_time_utc": b.time_utc.isoformat(),
                        "side": side.value,
                        "entry_price": float(entry),
                        "exit_price": float(exit_fill),
                        "stop_price": float(stop),
                        "target_price": float(target),
                        "spread_points": int(sp_points),
                        "pnl_points": float(pnl_points),
                        "risk_points": float(risk_points),
                        "r_multiple": float(r_mult),
                        "pnl_usd_1lot": pnl_usd_1lot,
                        "exit_reason": exit_reason,
                        "holding_bars": int(bar_index - (entry_bar_index or bar_index)),
                        "adx": entry_meta.get("adx"),
                        "atr": entry_meta.get("atr"),
                        "sma_slope": entry_meta.get("sma_slope"),
                        "sma_slope_atr": entry_meta.get("sma_slope_atr"),
                    }
                )

                in_pos = False
                entry = stop = target = None
                entry_time = None
                side = None
                entry_meta = {}
                entry_bar_index = None

            continue

        # ------------
        # Cooldown gate
        # ------------
        if cooldown_bars and last_entry_bar_index is not None:
            if (bar_index - last_entry_bar_index) < cooldown_bars:
                continue

        # ------------
        # Entry decision
        # ------------
        decision = strategy.on_bar(ctx)
        if decision is None:
            continue
        if decision.side == Side.SELL:
            continue

        stop_level = float(decision.stop_price) if decision.stop_price is not None else None
        target_level = float(decision.target_price) if decision.target_price is not None else None
        if stop_level is None or target_level is None:
            continue

        # Confirmation entry: BUY STOP at prev bar high; fill only if current bar trades through it
        if (entry_confirmation or "prev_high_break").lower().strip() == "prev_high_break":
            prev = bars[i - 1]
            entry_level = float(prev.high)
            if float(b.high) < entry_level:
                continue  # not filled
        else:
            entry_level = float(b.open)

        entry_fill = _apply_spread_fill(entry_level, side=decision.side, spread_points=sp_points, point=point, mode=spread_cfg.spread_mode)

        if not (stop_level < entry_fill < target_level):
            continue

        entry = entry_fill
        stop = stop_level
        target = target_level
        entry_time = b.time_utc
        side = decision.side
        entry_meta = dict(decision.meta or {})

        in_pos = True
        entry_bar_index = bar_index
        last_entry_bar_index = bar_index

    return pd.DataFrame(rows)


def max_drawdown_r(equity_r: pd.Series) -> float:
    if equity_r.empty:
        return 0.0
    peak = equity_r.cummax()
    dd = equity_r - peak
    return float(dd.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_utc", required=True)
    ap.add_argument("--end_utc", required=True)
    ap.add_argument("--out", default="data/derived/runs/backtest_latest")
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

    bars = rates_to_bars(rates)

    sp = dict(cfg["strategy"]["params"])
    sp.pop("min_bars_between_entries", None)  # legacy safety
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    risk_cfg = cfg.get("risk", {}) or {}
    bt_cfg = cfg.get("backtest", {}) or {}

    spread_cfg = BacktestSpreadCfg(
        spread_source=str(bt_cfg.get("spread_source", "fixed") or "fixed"),
        spread_points=int(bt_cfg.get("spread_points", 60) or 60),
        spread_mode=str(bt_cfg.get("spread_mode", "half") or "half"),
    )

    trades = simulate(
        bars=bars,
        rates=rates,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        point=point,
        spread_cfg=spread_cfg,
        intrabar_policy=str(bt_cfg.get("intrabar_policy", "conservative") or "conservative"),
        time_stop_bars=int(risk_cfg.get("time_stop_bars", 0) or 0),
        cooldown_bars=int(risk_cfg.get("cooldown_bars", 0) or 0),
        entry_confirmation=str(bt_cfg.get("entry_confirmation", "prev_high_break") or "prev_high_break"),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "ledger.csv"
    trades.to_csv(trades_path, index=False)

    equity_r = trades["r_multiple"].fillna(0.0).cumsum() if len(trades) else pd.Series(dtype=float)
    equity_path = out_dir / "equity_r.csv"
    pd.DataFrame({"equity_r": equity_r}).to_csv(equity_path, index=False)

    stats = {
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
        "trades": int(len(trades)),
        "avg_r": float(trades["r_multiple"].mean()) if len(trades) else 0.0,
        "median_r": float(trades["r_multiple"].median()) if len(trades) else 0.0,
        "total_r": float(trades["r_multiple"].sum()) if len(trades) else 0.0,
        "max_drawdown_r": max_drawdown_r(equity_r) if len(trades) else 0.0,
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict() if len(trades) and "exit_reason" in trades.columns else {},
    }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Wrote {trades_path}")
    print(f"Wrote {equity_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
