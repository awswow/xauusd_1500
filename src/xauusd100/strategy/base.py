from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from ..engine.models import Bar, Decision


@dataclass(frozen=True)
class StrategyContext:
    symbol: str
    timeframe: str
    bars: Sequence[Bar]  # includes closed bars, last element is last closed
    bar_index: int
    meta: dict[str, Any]


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, ctx: StrategyContext) -> Decision | None:
        raise NotImplementedError
