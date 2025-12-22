from __future__ import annotations

import math
from dataclasses import dataclass

from .symbols import SymbolSpec


@dataclass(frozen=True)
class BrokerRules:
    spec: SymbolSpec

    def round_volume(self, vol: float) -> float:
        v = max(self.spec.volume_min, min(vol, self.spec.volume_max))
        step = self.spec.volume_step
        if step <= 0:
            return float(f"{v:.8f}")
        steps = math.floor((v - self.spec.volume_min) / step + 1e-12)
        rounded = self.spec.volume_min + steps * step
        return float(f"{rounded:.8f}")

    def min_stop_distance_price(self) -> float:
        return self.spec.stops_level_points * self.spec.point

    def enforce_stop_distance(self, entry_price: float, stop_price: float, side: str) -> float:
        dmin = self.min_stop_distance_price()
        if dmin <= 0:
            return stop_price
        if side == "BUY":
            return min(stop_price, entry_price - dmin)
        return max(stop_price, entry_price + dmin)

    def enforce_target_distance(self, entry_price: float, target_price: float, side: str) -> float:
        dmin = self.min_stop_distance_price()
        if dmin <= 0:
            return target_price
        if side == "BUY":
            return max(target_price, entry_price + dmin)
        return min(target_price, entry_price - dmin)
