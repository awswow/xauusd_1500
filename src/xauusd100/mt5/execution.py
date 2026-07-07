from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import MetaTrader5 as mt5

from ..engine.models import Fill, OrderRequest, Side
from .broker_rules import BrokerRules


FILLING_MAP = {
    "FOK": mt5.ORDER_FILLING_FOK,
    "IOC": mt5.ORDER_FILLING_IOC,
    "RETURN": mt5.ORDER_FILLING_RETURN,
}


@dataclass(frozen=True)
class ExecutionConfig:
    magic: int
    deviation_points: int
    filling_preference: Sequence[str]
    comment: str = "xauusd100"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class MT5Executor:
    def __init__(self, rules: BrokerRules, cfg: ExecutionConfig):
        self.rules = rules
        self.cfg = cfg

    def _market_price(self, symbol: str, side: Side) -> float:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick None for {symbol}")
        return float(tick.ask if side == Side.BUY else tick.bid)

    def _build_request(self, req: OrderRequest, filling: int) -> dict:
        price = self._market_price(req.symbol, req.side)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": float(req.volume),
            "type": mt5.ORDER_TYPE_BUY if req.side == Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "deviation": int(req.deviation_points),
            "magic": int(req.magic),
            "comment": req.comment or self.cfg.comment,
            "type_filling": filling,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if req.stop_price is not None:
            request["sl"] = float(self.rules.enforce_stop_distance(price, req.stop_price, req.side.value))
        if req.target_price is not None:
            request["tp"] = float(self.rules.enforce_target_distance(price, req.target_price, req.side.value))

        return request

    def send_market(self, req: OrderRequest) -> Fill:
        req.volume = self.rules.round_volume(req.volume)

        last_err = ""
        for key in self.cfg.filling_preference:
            filling = FILLING_MAP.get(key)
            if filling is None:
                continue

            request = self._build_request(req, filling)
            result = mt5.order_send(request)

            if result is None:
                last_err = f"order_send returned None: {mt5.last_error()}"
                continue

            retcode = int(result.retcode)
            msg = str(getattr(result, "comment", ""))

            if retcode == mt5.TRADE_RETCODE_DONE:
                return Fill(
                    time_utc=_now_utc(),
                    symbol=req.symbol,
                    side=req.side,
                    volume=req.volume,
                    price=float(getattr(result, "price", 0.0) or request["price"]),
                    order_ticket=int(getattr(result, "order", 0) or 0) or None,
                    deal_ticket=int(getattr(result, "deal", 0) or 0) or None,
                    retcode=retcode,
                    message=msg,
                    meta={"filling": key},
                )

            last_err = f"retcode={retcode} msg={msg} filling={key}"

        return Fill(
            time_utc=_now_utc(),
            symbol=req.symbol,
            side=req.side,
            volume=req.volume,
            price=0.0,
            order_ticket=None,
            deal_ticket=None,
            retcode=-1,
            message=last_err or "unknown execution failure",
            meta={},
        )

    def send_pending_buy_stop(
        self,
        *,
        symbol: str,
        volume: float,
        trigger_price: float,
        sl: float,
        tp: float,
        magic: int,
        comment: str = "",
        deviation_points: int = 30,
    ) -> Fill:
        """Place a BUY_STOP pending order at trigger_price with attached SL/TP."""
        volume = self.rules.round_volume(volume)
        sl = float(self.rules.enforce_stop_distance(trigger_price, sl, "BUY"))
        tp = float(self.rules.enforce_target_distance(trigger_price, tp, "BUY"))

        for key in self.cfg.filling_preference:
            filling = FILLING_MAP.get(key)
            if filling is None:
                continue
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY_STOP,
                "price": float(trigger_price),
                "sl": float(sl),
                "tp": float(tp),
                "deviation": int(deviation_points),
                "magic": int(magic),
                "comment": comment or self.cfg.comment,
                "type_filling": filling,
                "type_time": mt5.ORDER_TIME_GTC,
            }
            result = mt5.order_send(request)
            if result is None:
                continue
            retcode = int(result.retcode)
            msg = str(getattr(result, "comment", ""))
            if retcode == mt5.TRADE_RETCODE_DONE:
                return Fill(
                    time_utc=_now_utc(),
                    symbol=symbol,
                    side=Side.BUY,
                    volume=float(volume),
                    price=float(trigger_price),
                    order_ticket=int(getattr(result, "order", 0) or 0) or None,
                    deal_ticket=None,
                    retcode=retcode,
                    message=msg,
                    meta={"pending": True, "filling": key},
                )

        return Fill(
            time_utc=_now_utc(),
            symbol=symbol,
            side=Side.BUY,
            volume=float(volume),
            price=0.0,
            order_ticket=None,
            deal_ticket=None,
            retcode=-1,
            message="pending_placement_failed",
            meta={},
        )

    def cancel_pending(self, ticket: int) -> bool:
        """Cancel a pending order by ticket number. Returns True if MT5 confirms removal."""
        result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)})
        if result is None:
            return False
        return int(result.retcode) == mt5.TRADE_RETCODE_DONE
