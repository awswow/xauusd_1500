from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import MetaTrader5 as mt5
import yaml

# local imports (repo style)
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1] / "src"))

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

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=UTC)


def rates_to_bars(rates) -> list[Bar]:
    out: list[Bar] = []
    for r in rates:
        out.append(
            Bar(
                time_utc=to_dt(r["time"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
            )
        )
    return out


@dataclass(frozen=True)
class ExecCfg:
    risk_usd_per_trade: float = 10.0
    daily_loss_limit_usd: float = 30.0
    max_trades_per_day: int = 2
    pending_expiry_bars: int = 4
    magic: int = 101001
    comment: str = "xauusd100_demo"
    slippage_points: int = 50
    deviation_points: int = 50
    dry_run: bool = False


class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        if self.path.exists():
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "time_utc",
                    "event",
                    "symbol",
                    "timeframe",
                    "details_json",
                ],
            )
            w.writeheader()

    def log(self, event: str, symbol: str, timeframe: str, details: Dict[str, Any]):
        row = {
            "time_utc": iso(utc_now()),
            "event": event,
            "symbol": symbol,
            "timeframe": timeframe,
            "details_json": json.dumps(details, ensure_ascii=False),
        }
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)


def mt5_point(symbol: str) -> float:
    spec = get_symbol_spec(symbol)
    return float(spec.point)


def tick_value_and_size(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None, None
    tick_value = getattr(info, "trade_tick_value", None)
    tick_size = getattr(info, "trade_tick_size", None)
    if tick_size in (None, 0):
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


def calc_lot_for_risk(symbol: str, entry: float, stop: float, risk_usd: float) -> float:
    """
    Lot size so that 1R loss ~= risk_usd, using MT5 tick_value/tick_size.
    Falls back to a conservative minimum if info missing.
    """
    tick_value, tick_size = tick_value_and_size(symbol)
    if tick_value is None or tick_size is None or tick_size <= 0:
        return 0.01  # fallback

    dist = abs(entry - stop)
    ticks = dist / tick_size
    loss_per_lot = ticks * tick_value
    if loss_per_lot <= 0:
        return 0.01

    lot = risk_usd / loss_per_lot
    # clamp to sane demo minimums; round down to 0.01
    lot = max(0.01, lot)
    lot = (int(lot * 100) / 100.0)
    return lot


def today_key_utc() -> str:
    n = utc_now()
    return n.strftime("%Y-%m-%d")


def get_closed_deals_today_usd(magic: int) -> float:
    """
    Approx daily realized PnL using history deals.
    """
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0.0
    pnl = 0.0
    for d in deals:
        # magic check if present
        if hasattr(d, "magic") and int(getattr(d, "magic")) != int(magic):
            continue
        # profit field
        p = getattr(d, "profit", 0.0)
        try:
            pnl += float(p)
        except Exception:
            pass
    return float(pnl)


def count_closed_trades_today(magic: int) -> int:
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0
    # count exit deals (DEAL_ENTRY_OUT) if available, else count unique positions roughly
    cnt = 0
    for d in deals:
        if hasattr(d, "magic") and int(getattr(d, "magic")) != int(magic):
            continue
        entry = getattr(d, "entry", None)
        # 1 typically equals DEAL_ENTRY_OUT; if unknown, just count profit-bearing deals
        if entry == 1 or (entry is None and getattr(d, "profit", 0.0) != 0.0):
            cnt += 1
    return int(cnt)


def has_open_position(symbol: str, magic: int) -> bool:
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return False
    for p in pos:
        if int(getattr(p, "magic", -1)) == int(magic):
            return True
    return False


def existing_pending(symbol: str, magic: int) -> list:
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return []
    out = []
    for o in orders:
        if int(getattr(o, "magic", -1)) == int(magic):
            out.append(o)
    return out


def cancel_order(ticket: int, logger: CSVLogger, symbol: str, timeframe: str):
    req = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": int(ticket),
    }
    res = mt5.order_send(req)
    logger.log("order_cancel", symbol, timeframe, {"ticket": ticket, "result": str(res)})


def place_buy_stop(
    *,
    symbol: str,
    lot: float,
    price: float,
    sl: float,
    tp: float,
    magic: int,
    comment: str,
    deviation_points: int,
    logger: CSVLogger,
    timeframe: str,
    dry_run: bool,
):
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY_STOP,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(deviation_points),
        "magic": int(magic),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    logger.log("order_place_intent", symbol, timeframe, {"req": req, "dry_run": dry_run})
    if dry_run:
        return None

    res = mt5.order_send(req)
    logger.log("order_place_result", symbol, timeframe, {"req": req, "result": str(res)})

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--poll_sec", type=int, default=10)
    ap.add_argument("--bars", type=int, default=250)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    tf = TF_MAP[timeframe]

    exec_cfg = ExecCfg(**(cfg.get("execution", {}) or {}))
    risk_cfg = cfg.get("risk", {}) or {}
    cooldown_bars = int(risk_cfg.get("cooldown_bars", 0) or 0)

    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    _ = get_symbol_spec(symbol)
    point = mt5_point(symbol)

    sp = dict(cfg["strategy"]["params"])
    sp.pop("min_bars_between_entries", None)
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    out_dir = Path("data/derived/demo")
    logger = CSVLogger(out_dir / "demo_events.csv")

    logger.log("demo_start", symbol, timeframe, {"config": {"strategy": sp, "execution": exec_cfg.__dict__, "risk": risk_cfg}})

    last_signal_bar_time: Optional[datetime] = None
    last_entry_bar_time: Optional[datetime] = None

    while True:
        # Daily controls
        realized = get_closed_deals_today_usd(exec_cfg.magic)
        trades_today = count_closed_trades_today(exec_cfg.magic)
        if realized <= -abs(exec_cfg.daily_loss_limit_usd):
            logger.log("daily_stop_hit", symbol, timeframe, {"realized_usd": realized, "limit": exec_cfg.daily_loss_limit_usd})
            time.sleep(args.poll_sec)
            continue
        if trades_today >= int(exec_cfg.max_trades_per_day):
            logger.log("daily_trade_cap_hit", symbol, timeframe, {"trades_today": trades_today, "cap": exec_cfg.max_trades_per_day})
            time.sleep(args.poll_sec)
            continue

        # If we already have an open position, do nothing (strategy is 1-position model)
        if has_open_position(symbol, exec_cfg.magic):
            time.sleep(args.poll_sec)
            continue

        # Manage existing pending orders: expire after N bars
        pendings = existing_pending(symbol, exec_cfg.magic)
        if pendings:
            # expire logic based on order setup time vs latest bars count approximation
            # simplest: cancel if order older than pending_expiry_bars * timeframe_minutes
            # (good enough for demo)
            tf_minutes = int(timeframe[1:]) if timeframe.startswith("M") else 60
            max_age = timedelta(minutes=tf_minutes * int(exec_cfg.pending_expiry_bars))
            now = utc_now()
            for o in pendings:
                setup_time = to_dt(getattr(o, "time_setup", getattr(o, "time", 0)))
                if now - setup_time > max_age:
                    cancel_order(int(o.ticket), logger, symbol, timeframe)
            time.sleep(args.poll_sec)
            continue

        # Pull latest bars
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, args.bars)
        if rates is None or len(rates) < 120:
            logger.log("rates_insufficient", symbol, timeframe, {"n": 0 if rates is None else len(rates)})
            time.sleep(args.poll_sec)
            continue

        bars = rates_to_bars(rates)
        # Current (forming) bar is last element in MT5 rates; use the last CLOSED bar for signaling
        # We'll build context up to last_closed_index inclusive.
        last_closed_index = len(bars) - 2
        if last_closed_index < 2:
            time.sleep(args.poll_sec)
            continue

        last_closed_bar = bars[last_closed_index]
        if last_signal_bar_time is not None and last_closed_bar.time_utc <= last_signal_bar_time:
            time.sleep(args.poll_sec)
            continue

        # Strategy decision on last CLOSED bar
        hist = bars[: last_closed_index + 1]
        ctx = StrategyContext(symbol=symbol, timeframe=timeframe, bars=hist, bar_index=len(hist), meta={"point": point})
        decision = strategy.on_bar(ctx)

        last_signal_bar_time = last_closed_bar.time_utc

        if decision is None:
            logger.log("no_signal", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc)})
            time.sleep(args.poll_sec)
            continue

        if decision.side != Side.BUY:
            logger.log("signal_ignored_nonbuy", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc)})
            time.sleep(args.poll_sec)
            continue

        # Cooldown (based on bar time, not tick time)
        if cooldown_bars and last_entry_bar_time is not None:
            # approximate cooldown by minutes
            tf_minutes = int(timeframe[1:]) if timeframe.startswith("M") else 60
            min_gap = timedelta(minutes=tf_minutes * cooldown_bars)
            if last_closed_bar.time_utc - last_entry_bar_time < min_gap:
                logger.log("cooldown_block", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), "last_entry_bar_time": iso(last_entry_bar_time)})
                time.sleep(args.poll_sec)
                continue

        # Confirmation entry: BUY STOP at previous CLOSED bar high
        prev_bar = bars[last_closed_index - 1]
        entry_level = float(prev_bar.high)

        sl = float(decision.stop_price)
        tp = float(decision.target_price)

        if not (sl < entry_level < tp):
            logger.log("geometry_reject", symbol, timeframe, {"entry_level": entry_level, "sl": sl, "tp": tp})
            time.sleep(args.poll_sec)
            continue

        lot = calc_lot_for_risk(symbol, entry_level, sl, exec_cfg.risk_usd_per_trade)

        # Place pending order
        logger.log(
            "signal_place_order",
            symbol,
            timeframe,
            {
                "signal_bar_time": iso(last_closed_bar.time_utc),
                "entry_level": entry_level,
                "sl": sl,
                "tp": tp,
                "lot": lot,
                "meta": decision.meta or {},
                "realized_today_usd": realized,
                "trades_today": trades_today,
            },
        )

        place_buy_stop(
            symbol=symbol,
            lot=lot,
            price=entry_level,
            sl=sl,
            tp=tp,
            magic=exec_cfg.magic,
            comment=exec_cfg.comment,
            deviation_points=exec_cfg.deviation_points,
            logger=logger,
            timeframe=timeframe,
            dry_run=exec_cfg.dry_run,
        )

        last_entry_bar_time = last_closed_bar.time_utc

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
