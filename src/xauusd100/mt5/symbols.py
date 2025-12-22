from __future__ import annotations

from dataclasses import dataclass

import MetaTrader5 as mt5


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level_points: int
    freeze_level_points: int


class SymbolError(RuntimeError):
    pass


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise SymbolError(f"symbol_info returned None for {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise SymbolError(f"symbol_select failed for {symbol}: {mt5.last_error()}")


def get_symbol_spec(symbol: str) -> SymbolSpec:
    ensure_symbol(symbol)
    info = mt5.symbol_info(symbol)
    if info is None:
        raise SymbolError(f"symbol_info returned None for {symbol}")

    trade_tick_size = float(getattr(info, "trade_tick_size", info.point))
    trade_tick_value = float(getattr(info, "trade_tick_value", 0.0))

    return SymbolSpec(
        symbol=symbol,
        digits=int(info.digits),
        point=float(info.point),
        trade_tick_size=trade_tick_size,
        trade_tick_value=trade_tick_value,
        volume_min=float(info.volume_min),
        volume_step=float(info.volume_step),
        volume_max=float(info.volume_max),
        stops_level_points=int(getattr(info, "trade_stops_level", 0)),
        freeze_level_points=int(getattr(info, "trade_freeze_level", 0)),
    )
