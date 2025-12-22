from __future__ import annotations

from dataclasses import dataclass

from ..mt5.broker_rules import BrokerRules


@dataclass(frozen=True)
class SizingConfig:
    risk_per_trade_usd: float


def size_by_stop_distance(
    *,
    rules: BrokerRules,
    risk_usd: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """Simple MT5 sizing using symbol tick_value/tick_size.

    loss_per_lot = (abs(entry-stop) / tick_size) * tick_value

    Notes:
    - Many brokers populate trade_tick_value and trade_tick_size.
    - If they are 0/invalid, we fall back to min lot (safest).
    """
    dist = abs(entry_price - stop_price)
    if dist <= 0:
        return rules.spec.volume_min

    tick_size = rules.spec.trade_tick_size
    tick_value = rules.spec.trade_tick_value
    if tick_size <= 0 or tick_value <= 0:
        return rules.spec.volume_min

    loss_per_lot = (dist / tick_size) * tick_value
    if loss_per_lot <= 0:
        return rules.spec.volume_min

    vol = risk_usd / loss_per_lot
    return rules.round_volume(vol)
