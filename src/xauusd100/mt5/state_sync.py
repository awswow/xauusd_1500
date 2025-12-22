from __future__ import annotations

from typing import Optional

import MetaTrader5 as mt5

from ..engine.models import PositionState, Side


def get_position_state(symbol: str, magic: int) -> PositionState:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        positions = []

    net_volume = 0.0
    weighted = 0.0
    side: Optional[Side] = None
    tickets: list[int] = []

    for p in positions:
        if int(getattr(p, "magic", -1)) != int(magic):
            continue
        vol = float(getattr(p, "volume", 0.0))
        ptype = int(getattr(p, "type", -1))
        price_open = float(getattr(p, "price_open", 0.0))
        ticket = int(getattr(p, "ticket", 0))

        if ptype == mt5.POSITION_TYPE_BUY:
            net_volume += vol
            side = Side.BUY
        elif ptype == mt5.POSITION_TYPE_SELL:
            net_volume += vol
            side = Side.SELL

        weighted += vol * price_open
        if ticket:
            tickets.append(ticket)

    avg_price = (weighted / net_volume) if net_volume > 0 else None
    if net_volume <= 0:
        side = None

    return PositionState(
        symbol=symbol,
        magic=magic,
        net_volume=net_volume,
        avg_price=avg_price,
        side=side,
        tickets=tickets,
    )
