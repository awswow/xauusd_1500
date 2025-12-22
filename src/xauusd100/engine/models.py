from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Bar:
    time_utc: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: Optional[int] = None


@dataclass
class Decision:
    time_utc: datetime
    symbol: str
    side: Side
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    volume: float
    magic: int
    deviation_points: int
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    comment: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    time_utc: datetime
    symbol: str
    side: Side
    volume: float
    price: float
    order_ticket: Optional[int]
    deal_ticket: Optional[int]
    retcode: int
    message: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionState:
    symbol: str
    magic: int
    net_volume: float
    avg_price: Optional[float]
    side: Optional[Side]
    tickets: list[int] = field(default_factory=list)


@dataclass
class RunEvent:
    kind: str  # decision|fill|info|error
    time_utc: datetime
    payload: dict[str, Any]
