# scripts/demo_executor_mt5.py
from __future__ import annotations

import argparse
import csv
import json
import time as _time
import traceback
from dataclasses import dataclass, fields
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import MetaTrader5 as mt5
import yaml
import numpy as np

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


def dataclass_from_dict(dc_type, raw: Dict[str, Any]):
    """Ignore unknown YAML keys safely."""
    allowed = {f.name for f in fields(dc_type)}
    cleaned = {k: v for k, v in (raw or {}).items() if k in allowed}
    return dc_type(**cleaned)


@dataclass(frozen=True)
class ExecCfg:
    risk_usd_per_trade: float = 10.0
    daily_loss_limit_usd: float = 30.0
    max_trades_per_day: int = 2
    pending_expiry_bars: int = 4
    magic: int = 101002
    comment: str = "xauusd100_demo"
    slippage_points: int = 50
    deviation_points: int = 50
    dry_run: bool = False
    heartbeat_sec: int = 60


@dataclass(frozen=True)
class RiskFilterCfg:
    enabled: bool = True
    tq_adx_len: int = 14
    tq_atr_len: int = 14
    tq_adx_min: float = 15.0
    tq_adx_rise_bars: int = 0
    tq_atr_ref_window: int = 120
    tq_atr_quantile: float = 0.20


class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["time_utc", "event", "symbol", "timeframe", "details_json"],
                ).writeheader()

    def log(self, event: str, symbol: str, timeframe: str, details: Dict[str, Any]):
        row = {
            "time_utc": iso(utc_now()),
            "event": event,
            "symbol": symbol,
            "timeframe": timeframe,
            "details_json": json.dumps(details, ensure_ascii=False),
        }
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(
                f,
                fieldnames=["time_utc", "event", "symbol", "timeframe", "details_json"],
            ).writerow(row)


# ----------------------------
# MT5 time normalization (portable: no mt5.time_current())
# ----------------------------
def compute_mt5_offset_sec(symbol: str) -> int:
    """
    Compute offset so that: real_utc_epoch ~= mt5_epoch + offset.
    Uses symbol_info_tick(symbol).time as MT5 server epoch seconds.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or getattr(tick, "time", None) in (None, 0):
        return 0
    server_epoch = int(tick.time)
    local_epoch = int(_time.time())
    return local_epoch - server_epoch  # often ~ -7200 for UTC+2 server


def to_dt_mt5(ts: int, offset_sec: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=UTC) + timedelta(seconds=int(offset_sec))


def rates_to_bars(rates, offset_sec: int) -> list[Bar]:
    out: list[Bar] = []
    for r in rates:
        out.append(
            Bar(
                time_utc=to_dt_mt5(r["time"], offset_sec),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
            )
        )
    return out


# ----------------------------
# Symbol helpers
# ----------------------------
def symbol_info_or_raise(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"mt5.symbol_info({symbol}) returned None")
    return info


def symbol_point(symbol: str) -> float:
    # prefer repo spec, but fall back to MT5 info if needed
    try:
        return float(get_symbol_spec(symbol).point)
    except Exception:
        info = symbol_info_or_raise(symbol)
        return float(getattr(info, "point", 0.01) or 0.01)


def symbol_digits(symbol: str) -> int:
    info = symbol_info_or_raise(symbol)
    return int(getattr(info, "digits", 2) or 2)


def round_to_symbol(symbol: str, price: float) -> float:
    return round(float(price), symbol_digits(symbol))


def stops_level_points(symbol: str) -> int:
    info = symbol_info_or_raise(symbol)
    return int(getattr(info, "trade_stops_level", 0) or 0)


def freeze_level_points(symbol: str) -> int:
    info = symbol_info_or_raise(symbol)
    return int(getattr(info, "trade_freeze_level", 0) or 0)


def get_tick(symbol: str):
    return mt5.symbol_info_tick(symbol)


def spread_points(symbol: str) -> Optional[float]:
    tick = get_tick(symbol)
    if tick is None:
        return None
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0:
        return None
    pt = symbol_point(symbol)
    return (ask - bid) / pt if pt > 0 else None


# ----------------------------
# Simple indicators for risk_filters gate
# ----------------------------
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(close) < n:
        return np.array([], dtype=float)
    tr = _true_range(high, low, close)
    out = np.full_like(tr, np.nan, dtype=float)
    out[n - 1] = float(np.mean(tr[:n]))
    alpha = 1.0 / float(n)
    for i in range(n, len(tr)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    return out


def adx_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(close) < (2 * n):
        return np.array([], dtype=float)

    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)

    def wilder_sum(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=float)
        out[n - 1] = float(np.sum(x[:n]))
        for i in range(n, len(x)):
            out[i] = out[i - 1] - (out[i - 1] / float(n)) + float(x[i])
        return out

    tr_s = wilder_sum(tr)
    p_dm_s = wilder_sum(plus_dm)
    m_dm_s = wilder_sum(minus_dm)

    plus_di = 100.0 * (p_dm_s / tr_s)
    minus_di = 100.0 * (m_dm_s / tr_s)

    denom = plus_di + minus_di
    dx = np.full_like(denom, np.nan, dtype=float)
    valid = denom > 0
    dx[valid] = 100.0 * (np.abs(plus_di[valid] - minus_di[valid]) / denom[valid])

    adx_out = np.full_like(dx, np.nan, dtype=float)
    seed_index = (2 * n) - 1
    adx_out[seed_index] = float(np.nanmean(dx[n : 2 * n]))
    for i in range(2 * n, len(dx)):
        adx_out[i] = ((adx_out[i - 1] * (n - 1)) + dx[i]) / float(n)

    return adx_out


def trend_quality_pass(bars: list[Bar], rf: RiskFilterCfg) -> Tuple[bool, Dict[str, Any]]:
    if not rf.enabled:
        return True, {"enabled": False}

    need = max(2 * rf.tq_adx_len + 5, rf.tq_atr_ref_window + rf.tq_atr_len + 5)
    if len(bars) < need:
        return False, {"enabled": True, "reason": "warmup", "need": need, "have": len(bars)}

    high = np.array([b.high for b in bars], dtype=float)
    low = np.array([b.low for b in bars], dtype=float)
    close = np.array([b.close for b in bars], dtype=float)

    atr_arr = atr_wilder(high, low, close, rf.tq_atr_len)
    adx_arr = adx_wilder(high, low, close, rf.tq_adx_len)

    if len(atr_arr) == 0 or len(adx_arr) == 0:
        return False, {"enabled": True, "reason": "indicator_empty"}

    atr_now = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else None
    adx_now = float(adx_arr[-1]) if np.isfinite(adx_arr[-1]) else None
    if atr_now is None or adx_now is None:
        return False, {"enabled": True, "reason": "indicator_nan"}

    # ADX minimum
    if adx_now < float(rf.tq_adx_min):
        return False, {"enabled": True, "reason": "adx_below_min", "adx_now": adx_now, "min": rf.tq_adx_min}

    # Optional ADX rising requirement over last N bars
    rise = int(rf.tq_adx_rise_bars or 0)
    if rise > 0:
        idx_prev = max(0, len(adx_arr) - 1 - rise)
        adx_prev = float(adx_arr[idx_prev]) if np.isfinite(adx_arr[idx_prev]) else None
        if adx_prev is None or adx_now <= adx_prev:
            return False, {"enabled": True, "reason": "adx_not_rising", "adx_now": adx_now, "adx_prev": adx_prev, "rise_bars": rise}

    # ATR quantile gate (avoid low-vol compression)
    w = int(rf.tq_atr_ref_window or 120)
    w = max(20, w)
    atr_window = atr_arr[-w:]
    atr_window = atr_window[np.isfinite(atr_window)]
    if len(atr_window) < max(20, int(0.5 * w)):
        return False, {"enabled": True, "reason": "atr_ref_insufficient", "have": int(len(atr_window)), "want": w}

    q = float(np.quantile(atr_window, float(rf.tq_atr_quantile)))
    if atr_now < q:
        return False, {"enabled": True, "reason": "atr_below_q", "atr_now": atr_now, "q": q, "quantile": rf.tq_atr_quantile, "ref_window": w}

    return True, {"enabled": True, "reason": "pass", "adx_now": adx_now, "atr_now": atr_now, "atr_q": q}


# ----------------------------
# Risk sizing
# ----------------------------
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
    tick_value, tick_size = tick_value_and_size(symbol)
    if tick_value is None or tick_size is None or tick_size <= 0:
        return 0.01

    dist = abs(float(entry) - float(stop))
    ticks = dist / tick_size
    loss_per_lot = ticks * tick_value
    if loss_per_lot <= 0:
        return 0.01

    lot = float(risk_usd) / loss_per_lot
    lot = max(0.01, lot)
    lot = int(lot * 100) / 100.0  # round down to 0.01
    return float(lot)


# ----------------------------
# Daily controls (FILTER by magic + symbol)
# ----------------------------
def get_closed_deals_today_usd(magic: int, symbol: str) -> float:
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0.0
    pnl = 0.0
    for d in deals:
        if int(getattr(d, "magic", -1)) != int(magic):
            continue
        if str(getattr(d, "symbol", "")) != str(symbol):
            continue
        try:
            pnl += float(getattr(d, "profit", 0.0) or 0.0)
        except Exception:
            pass
    return float(pnl)


def count_closed_trades_today(magic: int, symbol: str) -> int:
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0
    cnt = 0
    for d in deals:
        if int(getattr(d, "magic", -1)) != int(magic):
            continue
        if str(getattr(d, "symbol", "")) != str(symbol):
            continue
        entry = getattr(d, "entry", None)
        if entry == 1 or (entry is None and float(getattr(d, "profit", 0.0) or 0.0) != 0.0):
            cnt += 1
    return int(cnt)


# ----------------------------
# Orders / positions
# ----------------------------
def has_open_position(symbol: str, magic: int) -> bool:
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return False
    return any(int(getattr(p, "magic", -1)) == int(magic) for p in pos)


def get_open_position_ticket(symbol: str, magic: int) -> Optional[int]:
    pos = mt5.positions_get(symbol=symbol)
    if not pos:
        return None
    for p in pos:
        if int(getattr(p, "magic", -1)) == int(magic):
            try:
                return int(getattr(p, "ticket"))
            except Exception:
                return None
    return None


def existing_pending(symbol: str, magic: int) -> list:
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return []
    return [o for o in orders if int(getattr(o, "magic", -1)) == int(magic)]


def cancel_order(ticket: int, logger: CSVLogger, symbol: str, timeframe: str):
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
    res = mt5.order_send(req)
    logger.log("order_cancel", symbol, timeframe, {"ticket": ticket, "result": str(res)})


def send_market_buy(
    *,
    symbol: str,
    lot: float,
    sl: float,
    tp: float,
    magic: int,
    comment: str,
    deviation_points: int,
    logger: CSVLogger,
    timeframe: str,
    dry_run: bool,
) -> Any:
    tick = get_tick(symbol)
    if tick is None or float(getattr(tick, "ask", 0.0) or 0.0) <= 0:
        logger.log("market_buy_reject", symbol, timeframe, {"reason": "tick_missing_or_bad"})
        return None

    ask = round_to_symbol(symbol, float(tick.ask))
    sl = round_to_symbol(symbol, sl)
    tp = round_to_symbol(symbol, tp)

    fill_try = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
    res = None
    for fill in fill_try:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5.ORDER_TYPE_BUY,
            "price": float(ask),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(deviation_points),
            "magic": int(magic),
            "comment": comment,
            "type_filling": int(fill),
        }
        logger.log("market_buy_intent", symbol, timeframe, {"req": req, "dry_run": dry_run})
        if dry_run:
            return None
        res = mt5.order_send(req)
        logger.log("market_buy_result", symbol, timeframe, {"req": req, "result": str(res)})
        if res is not None and (
            getattr(res, "retcode", None) in (10009, 10008)
            or getattr(res, "deal", 0) != 0
            or getattr(res, "order", 0) != 0
        ):
            return res
    return res


def ensure_position_sltp(
    *,
    symbol: str,
    magic: int,
    sl: float,
    tp: float,
    logger: CSVLogger,
    timeframe: str,
) -> bool:
    pos = mt5.positions_get(symbol=symbol)
    if not pos:
        return False

    sl = round_to_symbol(symbol, sl)
    tp = round_to_symbol(symbol, tp)

    for p in pos:
        if int(getattr(p, "magic", -1)) != int(magic):
            continue
        ticket = int(getattr(p, "ticket", 0))
        current_sl = float(getattr(p, "sl", 0.0) or 0.0)
        current_tp = float(getattr(p, "tp", 0.0) or 0.0)

        if current_sl > 0 and current_tp > 0:
            return True

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(magic),
        }
        res = mt5.order_send(req)
        logger.log("position_sltp_set", symbol, timeframe, {"ticket": ticket, "req": req, "result": str(res)})
        return True

    return False


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


# ----------------------------
# Entry validation for BUY_STOP
# ----------------------------
def validate_buy_stop(
    symbol: str,
    entry: float,
    sl: float,
    tp: float,
    extra_buffer_points: int = 0,
) -> Tuple[bool, Dict[str, Any]]:
    tick = get_tick(symbol)
    if tick is None:
        return False, {"reason": "tick_missing"}

    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0:
        return False, {"reason": "tick_bad", "bid": bid, "ask": ask}

    pt = symbol_point(symbol)
    stops_pts = stops_level_points(symbol)
    freeze_pts = freeze_level_points(symbol)

    min_dist_pts = max(stops_pts, freeze_pts) + int(extra_buffer_points)
    min_dist = float(min_dist_pts) * float(pt)

    entry_r = round_to_symbol(symbol, entry)
    sl_r = round_to_symbol(symbol, sl)
    tp_r = round_to_symbol(symbol, tp)

    if not (sl_r < entry_r < tp_r):
        return False, {"reason": "geometry", "entry": entry_r, "sl": sl_r, "tp": tp_r}

    if entry_r <= ask + min_dist:
        return False, {
            "reason": "entry_too_close_or_below_ask",
            "entry": entry_r,
            "ask": ask,
            "bid": bid,
            "min_dist": min_dist,
            "min_dist_pts": min_dist_pts,
            "stops_pts": stops_pts,
            "freeze_pts": freeze_pts,
        }

    if (entry_r - sl_r) < min_dist:
        return False, {"reason": "sl_too_close", "entry": entry_r, "sl": sl_r, "min_dist": min_dist, "min_dist_pts": min_dist_pts}

    if (tp_r - entry_r) < min_dist:
        return False, {"reason": "tp_too_close", "entry": entry_r, "tp": tp_r, "min_dist": min_dist, "min_dist_pts": min_dist_pts}

    return True, {
        "entry": entry_r,
        "sl": sl_r,
        "tp": tp_r,
        "ask": ask,
        "bid": bid,
        "min_dist": min_dist,
        "min_dist_pts": min_dist_pts,
        "stops_pts": stops_pts,
        "freeze_pts": freeze_pts,
    }


# ----------------------------
# Session windows (UTC)
# ----------------------------
def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)


def _in_any_window(now_utc: datetime, windows: List[str]) -> bool:
    if not windows:
        return True
    m = now_utc.hour * 60 + now_utc.minute
    for w in windows:
        a, b = w.split("-")
        ah, am = _parse_hhmm(a)
        bh, bm = _parse_hhmm(b)
        start = ah * 60 + am
        end = bh * 60 + bm
        if start <= end:
            if start <= m < end:
                return True
        else:
            if m >= start or m < end:
                return True
    return False


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

    exec_cfg = dataclass_from_dict(ExecCfg, cfg.get("execution", {}) or {})
    risk_cfg = cfg.get("risk", {}) or {}
    cooldown_bars = int(risk_cfg.get("cooldown_bars", 0) or 0)

    # risk_filters (trend_quality_gate)
    rf_raw = cfg.get("risk_filters", {}) or {}
    rf_cfg = RiskFilterCfg(
        enabled=bool(rf_raw.get("trend_quality_gate", True)),
        tq_adx_len=int(rf_raw.get("tq_adx_len", 14) or 14),
        tq_atr_len=int(rf_raw.get("tq_atr_len", 14) or 14),
        tq_adx_min=float(rf_raw.get("tq_adx_min", 15.0) or 15.0),
        tq_adx_rise_bars=int(rf_raw.get("tq_adx_rise_bars", 0) or 0),
        tq_atr_ref_window=int(rf_raw.get("tq_atr_ref_window", 120) or 120),
        tq_atr_quantile=float(rf_raw.get("tq_atr_quantile", 0.20) or 0.20),
    )

    # Optional execution constraints
    max_spread_points_cfg = cfg.get("execution", {}).get("max_spread_points", None)
    try:
        max_spread_points_cfg = int(max_spread_points_cfg) if max_spread_points_cfg is not None else None
    except Exception:
        max_spread_points_cfg = None

    # Entry buffer for stop validation (points)
    entry_buffer_points = cfg.get("execution", {}).get("entry_buffer_points", 2)
    try:
        entry_buffer_points = int(entry_buffer_points)
    except Exception:
        entry_buffer_points = 2

    convert_to_market_on_breakout = bool(cfg.get("execution", {}).get("convert_to_market_on_breakout", True))

    trade_windows_utc = list(cfg.get("trade_windows_utc", []) or [])
    block_windows_utc = list(cfg.get("block_windows_utc", []) or [])

    out_dir = Path("data/derived/demo")
    logger = CSVLogger(out_dir / "demo_events.csv")

    def _reconnect(reason: str):
        logger.log("mt5_reconnect", symbol, timeframe, {"reason": reason, "last_error": str(mt5.last_error())})
        try:
            mt5.shutdown()
        except Exception:
            pass
        _time.sleep(1)
        MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
        mt5.symbol_select(symbol, True)

    # Connect MT5 once, then force symbol selection
    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()
    mt5.symbol_select(symbol, True)

    # Ensure symbol spec exists
    _ = get_symbol_spec(symbol)
    _ = symbol_info_or_raise(symbol)

    pt = symbol_point(symbol)
    mt5_offset_sec = compute_mt5_offset_sec(symbol)

    # Strategy init
    sp = dict(cfg["strategy"]["params"])
    sp.pop("min_bars_between_entries", None)
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    # Warmup: require tick + rates before starting the loop
    warmup_ok = False
    for _ in range(30):
        tick = mt5.symbol_info_tick(symbol)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, args.bars)
        ok_tick = tick is not None and float(getattr(tick, "bid", 0.0) or 0.0) > 0 and float(getattr(tick, "ask", 0.0) or 0.0) > 0
        ok_rates = rates is not None and len(rates) >= 120
        if ok_tick and ok_rates:
            warmup_ok = True
            break
        _time.sleep(1)

    logger.log(
        "demo_start",
        symbol,
        timeframe,
        {
            "mt5_offset_sec": mt5_offset_sec,
            "execution": exec_cfg.__dict__,
            "risk": risk_cfg,
            "strategy": sp,
            "risk_filters": rf_cfg.__dict__,
            "max_spread_points": max_spread_points_cfg,
            "entry_buffer_points": entry_buffer_points,
            "convert_to_market_on_breakout": convert_to_market_on_breakout,
            "trade_windows_utc": trade_windows_utc,
            "block_windows_utc": block_windows_utc,
            "warmup_ok": warmup_ok,
        },
    )

    if not warmup_ok:
        raise SystemExit("MT5 warmup failed: no tick and/or not enough bars. Open XAUUSD M15 chart and ensure quotes are live.")

    last_signal_bar_time: Optional[datetime] = None
    last_fill_time: Optional[datetime] = None
    last_pos_ticket: Optional[int] = None
    last_intended_sl: Optional[float] = None
    last_intended_tp: Optional[float] = None
    last_heartbeat = utc_now()

    while True:
        try:
            now = utc_now()

            if (now - last_heartbeat).total_seconds() >= int(exec_cfg.heartbeat_sec):
                logger.log("heartbeat", symbol, timeframe, {})
                last_heartbeat = now

            # Session gating (UTC)
            if trade_windows_utc and not _in_any_window(now, trade_windows_utc):
                logger.log("trade_window_block", symbol, timeframe, {"now_utc": iso(now)})
                _time.sleep(args.poll_sec)
                continue
            if block_windows_utc and _in_any_window(now, block_windows_utc):
                logger.log("block_window_active", symbol, timeframe, {"now_utc": iso(now)})
                _time.sleep(args.poll_sec)
                continue

            # Tick health (avoid spread_unavailable loops)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or float(getattr(tick, "bid", 0.0) or 0.0) <= 0 or float(getattr(tick, "ask", 0.0) or 0.0) <= 0:
                _reconnect("tick_missing_or_zero")
                _time.sleep(max(2, args.poll_sec))
                continue

            # Detect fills
            current_ticket = get_open_position_ticket(symbol, exec_cfg.magic)
            if current_ticket is not None and current_ticket != last_pos_ticket:
                last_pos_ticket = current_ticket
                last_fill_time = now
                logger.log("position_open_detected", symbol, timeframe, {"ticket": current_ticket, "fill_time_utc": iso(last_fill_time)})
                if last_intended_sl is not None and last_intended_tp is not None:
                    ensure_position_sltp(
                        symbol=symbol,
                        magic=exec_cfg.magic,
                        sl=last_intended_sl,
                        tp=last_intended_tp,
                        logger=logger,
                        timeframe=timeframe,
                    )

            if current_ticket is None and last_pos_ticket is not None:
                logger.log("position_closed_detected", symbol, timeframe, {"prev_ticket": last_pos_ticket, "time_utc": iso(now)})
                last_pos_ticket = None

            # Daily controls (magic + symbol filtered)
            realized = get_closed_deals_today_usd(exec_cfg.magic, symbol)
            trades_today = count_closed_trades_today(exec_cfg.magic, symbol)

            if realized <= -abs(exec_cfg.daily_loss_limit_usd):
                logger.log("daily_stop_hit", symbol, timeframe, {"realized_usd": realized, "limit": exec_cfg.daily_loss_limit_usd})
                _time.sleep(args.poll_sec)
                continue

            if trades_today >= int(exec_cfg.max_trades_per_day):
                logger.log("daily_trade_cap_hit", symbol, timeframe, {"trades_today": trades_today, "cap": exec_cfg.max_trades_per_day})
                _time.sleep(args.poll_sec)
                continue

            # Spread gate
            if max_spread_points_cfg is not None:
                sp_pts = spread_points(symbol)
                if sp_pts is None:
                    logger.log("spread_unavailable", symbol, timeframe, {})
                    _reconnect("spread_unavailable")
                    _time.sleep(max(2, args.poll_sec))
                    continue
                if sp_pts > float(max_spread_points_cfg):
                    logger.log("spread_block", symbol, timeframe, {"spread_points": sp_pts, "max_spread_points": max_spread_points_cfg})
                    _time.sleep(args.poll_sec)
                    continue

            # If open position, do nothing
            if has_open_position(symbol, exec_cfg.magic):
                _time.sleep(args.poll_sec)
                continue

            # Expire pending orders
            pendings = existing_pending(symbol, exec_cfg.magic)
            if pendings:
                tf_minutes = int(timeframe[1:]) if timeframe.startswith("M") else 60
                max_age = timedelta(minutes=tf_minutes * int(exec_cfg.pending_expiry_bars))
                for o in pendings:
                    setup_ts = getattr(o, "time_setup", getattr(o, "time", 0))
                    try:
                        setup_time = to_dt_mt5(int(setup_ts), mt5_offset_sec)
                    except Exception:
                        setup_time = now
                    age = now - setup_time
                    if age > max_age:
                        cancel_order(int(o.ticket), logger, symbol, timeframe)
                    else:
                        logger.log("pending_exists", symbol, timeframe, {"ticket": int(getattr(o, "ticket", 0)), "age_sec": int(age.total_seconds())})
                _time.sleep(args.poll_sec)
                continue

            # Pull latest bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, args.bars)
            if rates is None:
                logger.log("rates_none", symbol, timeframe, {"last_error": str(mt5.last_error())})
                _reconnect("rates_none")
                _time.sleep(max(2, args.poll_sec))
                continue
            if len(rates) < 120:
                logger.log("rates_insufficient", symbol, timeframe, {"n": int(len(rates))})
                _time.sleep(args.poll_sec)
                continue

            bars = rates_to_bars(rates, mt5_offset_sec)

            # last CLOSED bar for signaling
            last_closed_index = len(bars) - 2
            if last_closed_index < 2:
                _time.sleep(args.poll_sec)
                continue

            last_closed_bar = bars[last_closed_index]
            if last_signal_bar_time is not None and last_closed_bar.time_utc <= last_signal_bar_time:
                _time.sleep(args.poll_sec)
                continue

            # Trend-quality gate (risk_filters)
            ok_tq, tq_diag = trend_quality_pass(bars[: last_closed_index + 1], rf_cfg)
            if not ok_tq:
                logger.log("gate_block", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), **tq_diag})
                last_signal_bar_time = last_closed_bar.time_utc
                _time.sleep(args.poll_sec)
                continue

            hist = bars[: last_closed_index + 1]
            ctx = StrategyContext(symbol=symbol, timeframe=timeframe, bars=hist, bar_index=len(hist), meta={"point": pt})
            decision = strategy.on_bar(ctx)
            last_signal_bar_time = last_closed_bar.time_utc

            if decision is None:
                logger.log("no_signal", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc)})
                _time.sleep(args.poll_sec)
                continue

            if decision.side != Side.BUY:
                logger.log("signal_ignored_nonbuy", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc)})
                _time.sleep(args.poll_sec)
                continue

            # Cooldown based on last fill time
            if cooldown_bars and last_fill_time is not None:
                tf_minutes = int(timeframe[1:]) if timeframe.startswith("M") else 60
                min_gap = timedelta(minutes=tf_minutes * cooldown_bars)
                if now - last_fill_time < min_gap:
                    logger.log("cooldown_block", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), "last_fill_time": iso(last_fill_time)})
                    _time.sleep(args.poll_sec)
                    continue

            # Entry confirmation: BUY_STOP at previous CLOSED bar high
            prev_bar = bars[last_closed_index - 1]
            entry_level = float(prev_bar.high)
            sl = float(decision.stop_price)
            tp = float(decision.target_price)

            last_intended_sl = sl
            last_intended_tp = tp

            lot = calc_lot_for_risk(symbol, entry_level, sl, exec_cfg.risk_usd_per_trade)

            ok, diag = validate_buy_stop(symbol, entry_level, sl, tp, extra_buffer_points=int(entry_buffer_points))
            if not ok:
                logger.log("buy_stop_reject", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), **diag})

                if convert_to_market_on_breakout and diag.get("reason") == "entry_too_close_or_below_ask":
                    logger.log(
                        "convert_to_market",
                        symbol,
                        timeframe,
                        {
                            "bar_time": iso(last_closed_bar.time_utc),
                            "orig_entry": diag.get("entry"),
                            "ask": diag.get("ask"),
                            "sl": round_to_symbol(symbol, sl),
                            "tp": round_to_symbol(symbol, tp),
                            "lot": lot,
                        },
                    )
                    send_market_buy(
                        symbol=symbol,
                        lot=lot,
                        sl=sl,
                        tp=tp,
                        magic=exec_cfg.magic,
                        comment=exec_cfg.comment,
                        deviation_points=exec_cfg.deviation_points,
                        logger=logger,
                        timeframe=timeframe,
                        dry_run=exec_cfg.dry_run,
                    )
                _time.sleep(args.poll_sec)
                continue

            entry_level = float(diag["entry"])
            sl = float(diag["sl"])
            tp = float(diag["tp"])

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
                    "tq": tq_diag,
                    "realized_today_usd": realized,
                    "trades_today": trades_today,
                    "tick": {"bid": diag.get("bid"), "ask": diag.get("ask")},
                    "min_dist_pts": diag.get("min_dist_pts"),
                    "stops_pts": diag.get("stops_pts"),
                    "freeze_pts": diag.get("freeze_pts"),
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

            _time.sleep(args.poll_sec)

        except Exception as e:
            logger.log("executor_exception", symbol, timeframe, {"error": repr(e), "traceback": traceback.format_exc()})
            _time.sleep(max(5, args.poll_sec))


if __name__ == "__main__":
    main()
