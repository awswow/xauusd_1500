from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Sequence

import MetaTrader5 as mt5


@dataclass(frozen=True)
class ExecutionGuardrails:
    max_spread_points: int


def current_spread_points(symbol: str) -> int | None:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return None
    spread = (tick.ask - tick.bid) / float(info.point)
    return int(round(spread))


def _in_window(t: time, window: str) -> bool:
    a, b = window.split("-")
    h1, m1 = map(int, a.split(":"))
    h2, m2 = map(int, b.split(":"))
    start = time(h1, m1)
    end = time(h2, m2)
    return start <= t <= end


def is_within_windows_utc(ts: datetime, windows: Sequence[str]) -> bool:
    if not windows:
        return True
    t = ts.time()
    return any(_in_window(t, w) for w in windows)


def can_trade_now(
    *,
    symbol: str,
    ts_utc: datetime,
    guard: ExecutionGuardrails,
    trade_windows_utc: Sequence[str],
    block_windows_utc: Sequence[str],
) -> tuple[bool, str]:
    if trade_windows_utc and not is_within_windows_utc(ts_utc, trade_windows_utc):
        return False, "blocked:outside_trade_window"
    if block_windows_utc and is_within_windows_utc(ts_utc, block_windows_utc):
        return False, "blocked:block_window"

    sp = current_spread_points(symbol)
    if sp is None:
        return False, "blocked:spread_unknown"
    if sp > guard.max_spread_points:
        return False, f"blocked:spread_too_high({sp})"
    return True, "ok"
