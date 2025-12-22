from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class RiskLimits:
    max_trades_per_day: int
    max_daily_loss_usd: float
    cooldown_bars: int
    time_stop_bars: int


@dataclass
class RiskState:
    day: date
    trades_today: int = 0
    pnl_today_usd: float = 0.0
    last_entry_bar_index: Optional[int] = None


def can_trade_today(state: RiskState, limits: RiskLimits) -> tuple[bool, str]:
    if state.trades_today >= limits.max_trades_per_day:
        return False, "blocked:max_trades_per_day"
    if state.pnl_today_usd <= -abs(limits.max_daily_loss_usd):
        return False, "blocked:max_daily_loss"
    return True, "ok"


def can_enter_cooldown(state: RiskState, limits: RiskLimits, bar_index: int) -> tuple[bool, str]:
    if state.last_entry_bar_index is None:
        return True, "ok"
    if (bar_index - state.last_entry_bar_index) < limits.cooldown_bars:
        return False, "blocked:cooldown"
    return True, "ok"
