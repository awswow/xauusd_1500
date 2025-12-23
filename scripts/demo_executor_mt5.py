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
    magic: int = 101001
    comment: str = "xauusd100_demo"
    slippage_points: int = 50
    deviation_points: int = 50
    dry_run: bool = False
    heartbeat_sec: int = 60
    # extra safety:
    max_spread_points: Optional[int] = None
    entry_buffer_points: int = 2
    convert_to_market_on_breakout: bool = True


@dataclass(frozen=True)
class TrendQualityCfg:
    enabled: bool = True
    tq_adx_len: int = 14
    tq_atr_len: int = 14
    tq_adx_min: float = 15.0
    tq_adx_rise_bars: int = 0
    tq_atr_ref_window: int = 96
    tq_atr_quantile: float = 0.10  # PnL-first: less likely to deadlock


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


def ensure_symbol_selected(symbol: str) -> bool:
    """
    Ensures symbol is visible in Market Watch. Many 'tick missing' / 'rates None' issues
    are simply symbol not selected.
    """
    try:
        if mt5.symbol_select(symbol, True):
            return True
        return False
    except Exception:
        return False


def symbol_point(symbol: str) -> float:
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
# Indicators for risk filter
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
    adx_out[seed_index] = float(np.nanmean(dx[n:2 * n]))
    for i in range(2 * n, len(dx)):
        adx_out[i] = ((adx_out[i - 1] * (n - 1)) + dx[i]) / float(n)

    return adx_out


def trend_quality_gate(
    bars: list[Bar],
    cfg: TrendQualityCfg,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (pass, diag). If blocked, diag includes reason.
    """
    if not cfg.enabled:
        return True, {"enabled": False}

    need = max(2 * cfg.tq_atr_len + 5, cfg.tq_atr_ref_window + cfg.tq_atr_len + 5)
    if len(bars) < need:
        return True, {"enabled": True, "reason": "warmup", "need": need, "have": len(bars)}

    high = np.array([b.high for b in bars], dtype=float)
    low = np.array([b.low for b in bars], dtype=float)
    close = np.array([b.close for b in bars], dtype=float)

    atr_arr = atr_wilder(high, low, close, int(cfg.tq_atr_len))
    adx_arr = adx_wilder(high, low, close, int(cfg.tq_adx_len))

    if len(atr_arr) == 0 or len(adx_arr) == 0:
        return True, {"enabled": True, "reason": "indicator_empty"}

    atr_now = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else None
    adx_now = float(adx_arr[-1]) if np.isfinite(adx_arr[-1]) else None
    if atr_now is None or adx_now is None:
        return True, {"enabled": True, "reason": "indicator_nan"}

    # ADX min gate
    if adx_now < float(cfg.tq_adx_min):
        return False, {
            "enabled": True,
            "reason": "adx_below_min",
            "adx_now": adx_now,
            "adx_min": float(cfg.tq_adx_min),
        }

    # Optional "ADX rising" gate
    rise = int(cfg.tq_adx_rise_bars or 0)
    if rise > 0 and len(adx_arr) > (rise + 1):
        prev = float(adx_arr[-(rise + 1)]) if np.isfinite(adx_arr[-(rise + 1)]) else None
        if prev is not None and adx_now < prev:
            return False, {
                "enabled": True,
                "reason": "adx_not_rising",
                "adx_now": adx_now,
                "adx_prev": prev,
                "rise_bars": rise,
            }

    # ATR quantile gate
    ref = int(cfg.tq_atr_ref_window)
    qv = float(cfg.tq_atr_quantile)
    atr_tail = atr_arr[-ref:] if len(atr_arr) >= ref else atr_arr
    atr_tail = atr_tail[np.isfinite(atr_tail)]
    if len(atr_tail) < max(20, ref // 2):
        # not enough stable samples; don't block
        return True, {"enabled": True, "reason": "atr_ref_insufficient", "have": int(len(atr_tail)), "ref_window": ref}

    q = float(np.quantile(atr_tail, qv))
    if atr_now < q:
        return False, {
            "enabled": True,
            "reason": "atr_below_q",
            "atr_now": atr_now,
            "q": q,
            "quantile": qv,
            "ref_window": ref,
        }

    return True, {
        "enabled": True,
        "reason": "pass",
        "atr_now": atr_now,
        "adx_now": adx_now,
        "q": float(np.quantile(atr_tail, qv)),
        "quantile": qv,
        "ref_window": ref,
    }


# ----------------------------
# Daily controls (FIXED baseline)
# ----------------------------
def get_closed_deals_today_usd(magic: int, symbol: str) -> float:
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0.0
    pnl = 0.0
    for d in deals:
        try:
            if int(getattr(d, "magic", -1)) != int(magic):
                continue
            if str(getattr(d, "symbol", "")) != str(symbol):
                continue
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
        try:
            if int(getattr(d, "magic", -1)) != int(magic):
                continue
            if str(getattr(d, "symbol", "")) != str(symbol):
                continue
            # entry==1 corresponds to DEAL_ENTRY_OUT in MT5 python wrapper for many brokers
            entry = getattr(d, "entry", None)
            if entry == 1:
                cnt += 1
        except Exception:
            pass
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
    res_last = None
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
        res_last = res
        logger.log("market_buy_result", symbol, timeframe, {"req": req, "result": str(res)})
        if res is not None and (getattr(res, "retcode", None) in (10009, 10008) or getattr(res, "deal", 0) != 0 or getattr(res, "order", 0) != 0):
            return res
    return res_last


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
# Risk sizing (unchanged)
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
# BUY_STOP validation (simple)
# ----------------------------
def validate_buy_stop_simple(symbol: str, entry: float, sl: float, tp: float, extra_buffer_points: int = 0) -> Tuple[bool, Dict[str, Any]]:
    tick = get_tick(symbol)
    if tick is None:
        return False, {"reason": "tick_missing"}

    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0:
        return False, {"reason": "tick_bad", "bid": bid, "ask": ask}

    entry_r = round_to_symbol(symbol, entry)
    sl_r = round_to_symbol(symbol, sl)
    tp_r = round_to_symbol(symbol, tp)

    if not (sl_r < entry_r < tp_r):
        return False, {"reason": "geometry", "entry": entry_r, "sl": sl_r, "tp": tp_r}

    # Minimal buffer above ask to avoid immediate reject
    pt = symbol_point(symbol)
    min_dist = float(max(1, int(extra_buffer_points))) * float(pt)
    if entry_r <= ask + min_dist:
        return False, {"reason": "entry_too_close_or_below_ask", "entry": entry_r, "ask": ask, "bid": bid, "min_dist": min_dist}

    return True, {"entry": entry_r, "sl": sl_r, "tp": tp_r, "ask": ask, "bid": bid, "min_dist": min_dist}


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
    ap.add_argument("--min_bars", type=int, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    tf = TF_MAP[timeframe]

    exec_cfg = dataclass_from_dict(ExecCfg, cfg.get("execution", {}) or {})
    risk_cfg = cfg.get("risk", {}) or {}
    cooldown_bars = int(risk_cfg.get("cooldown_bars", 0) or 0)

    tq_cfg = dataclass_from_dict(TrendQualityCfg, cfg.get("risk_filters", {}) or {})
    # Accept legacy key "trend_quality_gate: true"
    if "trend_quality_gate" in (cfg.get("risk_filters", {}) or {}):
        tq_cfg = TrendQualityCfg(
            enabled=bool((cfg.get("risk_filters", {}) or {}).get("trend_quality_gate", True)),
            tq_adx_len=int((cfg.get("risk_filters", {}) or {}).get("tq_adx_len", tq_cfg.tq_adx_len)),
            tq_atr_len=int((cfg.get("risk_filters", {}) or {}).get("tq_atr_len", tq_cfg.tq_atr_len)),
            tq_adx_min=float((cfg.get("risk_filters", {}) or {}).get("tq_adx_min", tq_cfg.tq_adx_min)),
            tq_adx_rise_bars=int((cfg.get("risk_filters", {}) or {}).get("tq_adx_rise_bars", tq_cfg.tq_adx_rise_bars)),
            tq_atr_ref_window=int((cfg.get("risk_filters", {}) or {}).get("tq_atr_ref_window", tq_cfg.tq_atr_ref_window)),
            tq_atr_quantile=float((cfg.get("risk_filters", {}) or {}).get("tq_atr_quantile", tq_cfg.tq_atr_quantile)),
        )

    trade_windows_utc = list(cfg.get("trade_windows_utc", []) or [])
    block_windows_utc = list(cfg.get("block_windows_utc", []) or [])

    # Connect MT5 once
    MT5Connector(MT5Config(**cfg.get("mt5", {}))).connect()

    # Ensure symbol selected/known
    _ = get_symbol_spec(symbol)
    if not ensure_symbol_selected(symbol):
        raise RuntimeError(f"Failed mt5.symbol_select({symbol}, True). Add symbol to Market Watch.")
    _ = symbol_info_or_raise(symbol)

    pt = symbol_point(symbol)
    mt5_offset_sec = compute_mt5_offset_sec(symbol)

    # Strategy init
    sp = dict(cfg["strategy"]["params"])
    sp.pop("min_bars_between_entries", None)
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    out_dir = Path("data/derived/demo")
    logger = CSVLogger(out_dir / "demo_events.csv")

    # Baselines: prevent inheriting earlier PnL / trades for the day (FIX)
    baseline_realized = get_closed_deals_today_usd(exec_cfg.magic, symbol)
    baseline_trades = count_closed_trades_today(exec_cfg.magic, symbol)

    logger.log(
        "demo_start",
        symbol,
        timeframe,
        {
            "mt5_offset_sec": mt5_offset_sec,
            "execution": exec_cfg.__dict__,
            "risk": risk_cfg,
            "strategy": sp,
            "risk_filters": tq_cfg.__dict__,
            "baseline_realized_usd": baseline_realized,
            "baseline_trades": baseline_trades,
            "point": pt,
            "trade_windows_utc": trade_windows_utc,
            "block_windows_utc": block_windows_utc,
        },
    )

    last_signal_bar_time: Optional[datetime] = None

    # Cooldown anchored to fills
    last_fill_time: Optional[datetime] = None
    last_pos_ticket: Optional[int] = None

    last_heartbeat = utc_now()
    rates_fail_streak = 0

    while True:
        try:
            now = utc_now()

            # Heartbeat
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

            # Detect fills (position open/close transitions)
            current_ticket = get_open_position_ticket(symbol, exec_cfg.magic)
            if current_ticket is not None and current_ticket != last_pos_ticket:
                last_pos_ticket = current_ticket
                last_fill_time = now
                logger.log("position_open_detected", symbol, timeframe, {"ticket": current_ticket, "fill_time_utc": iso(last_fill_time)})

            if current_ticket is None and last_pos_ticket is not None:
                logger.log("position_closed_detected", symbol, timeframe, {"prev_ticket": last_pos_ticket, "time_utc": iso(now)})
                last_pos_ticket = None

            # Daily controls (FIXED baseline)
            realized_total = get_closed_deals_today_usd(exec_cfg.magic, symbol)
            trades_total = count_closed_trades_today(exec_cfg.magic, symbol)
            realized = float(realized_total - baseline_realized)
            trades_today = int(trades_total - baseline_trades)

            if realized <= -abs(exec_cfg.daily_loss_limit_usd):
                logger.log("daily_stop_hit", symbol, timeframe, {"realized_usd": realized, "limit": exec_cfg.daily_loss_limit_usd})
                _time.sleep(args.poll_sec)
                continue

            if trades_today >= int(exec_cfg.max_trades_per_day):
                logger.log("daily_trade_cap_hit", symbol, timeframe, {"trades_today": trades_today, "cap": exec_cfg.max_trades_per_day})
                _time.sleep(args.poll_sec)
                continue

            # Spread gate
            if exec_cfg.max_spread_points is not None:
                sp_pts = spread_points(symbol)
                if sp_pts is None:
                    logger.log("spread_unavailable", symbol, timeframe, {})
                    _time.sleep(args.poll_sec)
                    continue
                if sp_pts > float(exec_cfg.max_spread_points):
                    logger.log("spread_block", symbol, timeframe, {"spread_points": sp_pts, "max_spread_points": exec_cfg.max_spread_points})
                    _time.sleep(args.poll_sec)
                    continue

            # If open position, do nothing
            if has_open_position(symbol, exec_cfg.magic):
                _time.sleep(args.poll_sec)
                continue

            # Pull latest bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(args.bars))
            if rates is None or len(rates) < int(args.min_bars):
                rates_fail_streak += 1
                logger.log("rates_insufficient", symbol, timeframe, {"n": 0 if rates is None else int(len(rates)), "fail_streak": rates_fail_streak})
                # Periodically re-select symbol (fixes many MT5 “0 bars” loops)
                if rates_fail_streak % 6 == 0:
                    ensure_symbol_selected(symbol)
                _time.sleep(args.poll_sec)
                continue
            rates_fail_streak = 0

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

            # Trend-quality gate BEFORE decision (this is what your logs show)
            warmup_ok = len(bars) >= int(args.min_bars)
            ok_gate, diag_gate = trend_quality_gate(bars[: last_closed_index + 1], tq_cfg)
            if not ok_gate:
                logger.log("gate_block", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), **diag_gate, "warmup_ok": warmup_ok})
                last_signal_bar_time = last_closed_bar.time_utc
                _time.sleep(args.poll_sec)
                continue

            # Strategy decision on CLOSED bar
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

            lot = calc_lot_for_risk(symbol, entry_level, sl, exec_cfg.risk_usd_per_trade)

            ok, diag = validate_buy_stop_simple(symbol, entry_level, sl, tp, extra_buffer_points=int(exec_cfg.entry_buffer_points))
            if not ok:
                logger.log("buy_stop_reject", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), **diag})
                # If breakout already happened, convert to market
                if exec_cfg.convert_to_market_on_breakout and diag.get("reason") == "entry_too_close_or_below_ask":
                    logger.log("convert_to_market", symbol, timeframe, {"bar_time": iso(last_closed_bar.time_utc), "lot": lot, "sl": sl, "tp": tp})
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
                    "realized_today_usd": realized,
                    "trades_today": trades_today,
                    "tick": {"bid": diag.get("bid"), "ask": diag.get("ask")},
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
