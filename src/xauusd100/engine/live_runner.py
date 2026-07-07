from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import yaml

from ..engine.models import Bar, OrderRequest, RunEvent
from ..mt5.connector import MT5Config, MT5Connector
from ..mt5.symbols import get_symbol_spec
from ..mt5.broker_rules import BrokerRules
from ..mt5.execution import ExecutionConfig, MT5Executor
from ..mt5.state_sync import get_position_state
from ..risk.guardrails import ExecutionGuardrails, can_trade_now
from ..risk.limits import RiskLimits, RiskState, can_trade_today, can_enter_cooldown
from ..risk.sizing import size_by_stop_distance
from ..risk_filters.trend_quality import TrendQualityParams, trend_quality_gate
from ..strategy.pullback_trend import PullbackTrendStrategy, PullbackTrendParams
from ..strategy.base import StrategyContext


TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _rates_to_bars(rates) -> list[Bar]:
    bars: list[Bar] = []
    for r in rates:
        t = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
        bars.append(
            Bar(
                time_utc=t,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                tick_volume=int(r["tick_volume"]) if "tick_volume" in r.dtype.names else None,
            )
        )
    return bars


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


@dataclass
class LiveApp:
    cfg_path: str

    def run(self) -> None:
        cfg = yaml.safe_load(Path(self.cfg_path).read_text(encoding="utf-8"))

        symbol = cfg["symbol"]
        timeframe = cfg["timeframe"]
        tf = TF_MAP[timeframe]

        run_name = cfg.get("run_name", "live")
        run_id = _utc_now().strftime("%Y-%m-%dT%H-%M-%SZ") + f"_{run_name}"

        out_base = Path(cfg.get("reporting", {}).get("out_dir", "data/derived/runs"))
        out_dir = out_base / run_id
        _safe_mkdir(out_dir)

        (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        mt5_cfg = MT5Config(**cfg.get("mt5", {}))
        MT5Connector(mt5_cfg).connect()

        spec = get_symbol_spec(symbol)
        rules = BrokerRules(spec)
        point = float(spec.point)

        sp = cfg["strategy"]["params"]
        strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

        rl = cfg.get("risk", {}) or {}
        ex_init = cfg.get("execution", {}) or {}
        limits = RiskLimits(
            max_trades_per_day=int(ex_init.get("max_trades_per_day", 4)),
            max_daily_loss_usd=float(ex_init.get("daily_loss_limit_usd", 45.0)),
            cooldown_bars=int(rl.get("cooldown_bars", 16)),
            time_stop_bars=int(rl.get("time_stop_bars", 0)),
        )
        risk_state = RiskState(day=_utc_now().date())

        ex = cfg["execution"]
        exec_cfg = ExecutionConfig(
            magic=int(ex["magic"]),
            deviation_points=int(ex.get("deviation_points", 30)),
            filling_preference=ex.get("filling_preference", ["FOK", "IOC", "RETURN"]),
            comment=str(ex.get("comment", "xauusd100")),
        )
        executor = MT5Executor(rules=rules, cfg=exec_cfg)

        guard = ExecutionGuardrails(max_spread_points=int(ex.get("max_spread_points", 9999)))
        trade_windows = cfg.get("trade_windows_utc", [])
        block_windows = cfg.get("block_windows_utc", [])
        pending_expiry_bars = int(ex.get("pending_expiry_bars", 4))
        convert_to_market = bool(ex.get("convert_to_market_on_breakout", True))
        target_r_param = float((cfg.get("strategy", {}) or {}).get("params", {}).get("target_r", 2.0))
        risk_usd = float(ex.get("risk_usd_per_trade", 15.0))
        monthly_loss_limit_r = float(rl.get("monthly_loss_limit_r", 0.0) or 0.0)

        events_path = out_dir / "events.jsonl"
        stats_path = out_dir / "live_state.json"

        last_closed_bar_time: datetime | None = None
        bar_index = 0

        # Pending order state — tracks a single outstanding BUY_STOP placed this session
        pending_ticket: int | None = None
        pending_created_bar_index: int = 0

        # Recover any pending order placed by a previous run of this bot
        existing_orders = mt5.orders_get(symbol=symbol)
        if existing_orders:
            for o in existing_orders:
                if (int(getattr(o, "magic", -1)) == exec_cfg.magic
                        and int(getattr(o, "type", -1)) == mt5.ORDER_TYPE_BUY_STOP):
                    pending_ticket = int(o.ticket)
                    _append_jsonl(events_path, RunEvent(
                        kind="info", time_utc=_utc_now(),
                        payload={"event": "startup_pending_recovered", "ticket": pending_ticket},
                    ).__dict__)
                    break

        while True:
            try:
                now = _utc_now()

                if risk_state.day != now.date():
                    risk_state = RiskState(day=now.date())

                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 250)
                if rates is None or len(rates) < 100:
                    time.sleep(1.0)
                    continue

                bars = _rates_to_bars(rates)
                closed_bar = bars[-2]

                if last_closed_bar_time is not None and closed_bar.time_utc <= last_closed_bar_time:
                    time.sleep(0.5)
                    continue

                last_closed_bar_time = closed_bar.time_utc
                bar_index += 1

                pos = get_position_state(symbol, exec_cfg.magic)

                # ── Pending order lifecycle ──────────────────────────────────────
                if pending_ticket is not None:
                    if pos.net_volume > 0:
                        # Broker filled the pending — position is now open
                        _append_jsonl(events_path, RunEvent(
                            kind="info", time_utc=now,
                            payload={"event": "pending_filled", "ticket": pending_ticket},
                        ).__dict__)
                        risk_state.trades_today += 1
                        risk_state.last_entry_bar_index = bar_index
                        pending_ticket = None
                    else:
                        # Check whether the order still exists in MT5
                        live_orders = mt5.orders_get(symbol=symbol)
                        live_tickets = {int(o.ticket) for o in (live_orders or [])}
                        if pending_ticket not in live_tickets:
                            # Broker removed it (requote, margin, broker expiry)
                            _append_jsonl(events_path, RunEvent(
                                kind="info", time_utc=now,
                                payload={"event": "pending_vanished", "ticket": pending_ticket},
                            ).__dict__)
                            pending_ticket = None
                        elif (bar_index - pending_created_bar_index) >= pending_expiry_bars:
                            # Our expiry: cancel the order
                            ok = executor.cancel_pending(pending_ticket)
                            _append_jsonl(events_path, RunEvent(
                                kind="info", time_utc=now,
                                payload={"event": "pending_expired", "ticket": pending_ticket, "cancelled": ok},
                            ).__dict__)
                            pending_ticket = None
                        else:
                            bars_left = pending_expiry_bars - (bar_index - pending_created_bar_index)
                            _append_jsonl(events_path, RunEvent(
                                kind="info", time_utc=now,
                                payload={"event": "pending_waiting", "ticket": pending_ticket, "bars_left": bars_left},
                            ).__dict__)

                    _write_json(stats_path, {"risk_state": risk_state.__dict__})
                    time.sleep(0.2)
                    continue

                # ── Already in a position ────────────────────────────────────────
                if pos.net_volume > 0:
                    _append_jsonl(events_path, RunEvent(
                        kind="info", time_utc=now,
                        payload={"pos": "in_market", "net_volume": pos.net_volume},
                    ).__dict__)
                    _write_json(stats_path, {"risk_state": risk_state.__dict__, "position": pos.__dict__})
                    time.sleep(0.2)
                    continue

                # ── Guardrails ───────────────────────────────────────────────────
                ok, why = can_trade_now(
                    symbol=symbol,
                    ts_utc=closed_bar.time_utc,
                    guard=guard,
                    trade_windows_utc=trade_windows,
                    block_windows_utc=block_windows,
                )
                if not ok:
                    _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": why}).__dict__)
                    continue

                # Update live daily PnL from MT5 history (fixes daily loss limit tracking)
                day_start_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
                day_deals = mt5.history_deals_get(day_start_dt, now)
                if day_deals is not None:
                    risk_state.pnl_today_usd = sum(
                        float(d.profit) for d in day_deals
                        if d.magic == exec_cfg.magic and d.symbol == symbol
                    )

                ok, why = can_trade_today(risk_state, limits)
                if not ok:
                    _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": why}).__dict__)
                    continue

                ok, why = can_enter_cooldown(risk_state, limits, bar_index)
                if not ok:
                    _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": why}).__dict__)
                    continue

                # ── Weekly regime gate ───────────────────────────────────────────
                regime_cfg = cfg.get("regime_filter", {}) or {}
                if regime_cfg.get("enabled", False):
                    sma_weeks = int(regime_cfg.get("sma_weeks", 40))
                    slope_lookback = int(regime_cfg.get("slope_lookback_weeks", 4))
                    w_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_W1, 0, sma_weeks + slope_lookback + 3)
                    if w_rates is None or len(w_rates) < sma_weeks + 1:
                        _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": "regime_no_data"}).__dict__)
                        continue
                    w_closes = [float(r["close"]) for r in w_rates[:-1]]
                    if len(w_closes) < sma_weeks:
                        _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": "regime_warmup"}).__dict__)
                        continue
                    sma_now = sum(w_closes[-sma_weeks:]) / sma_weeks
                    price_above = w_closes[-1] > sma_now
                    slope_ok = True
                    if slope_lookback > 0 and len(w_closes) >= sma_weeks + slope_lookback:
                        sma_prev = sum(w_closes[-(sma_weeks + slope_lookback):-slope_lookback]) / sma_weeks
                        slope_ok = sma_now > sma_prev
                    if not (price_above and slope_ok):
                        _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={
                            "skip": "weekly_regime_off",
                            "close": w_closes[-1],
                            "sma40w": round(sma_now, 2),
                            "slope_ok": slope_ok,
                        }).__dict__)
                        continue

                # ── Monthly performance brake ────────────────────────────────────
                if monthly_loss_limit_r > 0:
                    month_start_dt = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
                    deals = mt5.history_deals_get(month_start_dt, now)
                    if deals is not None and risk_usd > 0:
                        month_pnl = sum(
                            float(d.profit) for d in deals
                            if d.magic == exec_cfg.magic and d.symbol == symbol
                        )
                        month_r = month_pnl / risk_usd
                        if month_r <= -monthly_loss_limit_r:
                            _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={
                                "skip": "monthly_brake",
                                "month_r": round(month_r, 2),
                                "limit": monthly_loss_limit_r,
                            }).__dict__)
                            continue

                # ── Trend quality gate ───────────────────────────────────────────
                rf = cfg.get("risk_filters", {}) or {}
                if rf.get("trend_quality_gate", False):
                    tq_params = TrendQualityParams(
                        tq_adx_len=int(rf.get("tq_adx_len", 14)),
                        tq_atr_len=int(rf.get("tq_atr_len", 14)),
                        tq_adx_min=float(rf.get("tq_adx_min", 15.0)),
                        tq_adx_rise_bars=int(rf.get("tq_adx_rise_bars", 0)),
                        tq_atr_ref_window=int(rf.get("tq_atr_ref_window", 120)),
                        tq_atr_quantile=float(rf.get("tq_atr_quantile", 0.20)),
                    )
                    tq_ok, tq_reason, tq_meta = trend_quality_gate(bars[:-1], tq_params)
                    if not tq_ok:
                        _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": f"tq:{tq_reason}", **tq_meta}).__dict__)
                        continue

                # ── Strategy ────────────────────────────────────────────────────
                ctx = StrategyContext(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars[:-1],
                    bar_index=bar_index,
                    meta={"symbol_spec": spec.__dict__},
                )
                decision = strategy.on_bar(ctx)
                if decision is None:
                    _append_jsonl(events_path, RunEvent(kind="decision", time_utc=now, payload={"decision": None}).__dict__)
                    _write_json(stats_path, {"risk_state": risk_state.__dict__})
                    continue

                _append_jsonl(events_path, RunEvent(kind="decision", time_utc=now, payload={"decision": decision.__dict__}).__dict__)

                # ── Order placement (unified with backtest) ──────────────────────
                # Entry trigger = signal bar's high (matches backtest BUY_STOP trigger)
                entry_level = float(closed_bar.high)
                sl_price = float(decision.stop_price)
                tp_price = entry_level + target_r_param * (entry_level - sl_price)

                if not (sl_price < entry_level < tp_price):
                    _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={
                        "skip": "geometry_reject",
                        "entry_level": entry_level,
                        "sl": sl_price,
                        "tp": tp_price,
                    }).__dict__)
                    continue

                # Size on the pending trigger price (not current ask)
                volume = size_by_stop_distance(
                    rules=rules,
                    risk_usd=risk_usd,
                    entry_price=entry_level,
                    stop_price=sl_price,
                )

                # Check current ask to decide pending vs. market
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    time.sleep(0.5)
                    continue
                ask_now = float(tick.ask)

                if convert_to_market and ask_now >= entry_level:
                    # Price already through the trigger — market order (backtest "convert" path)
                    rebased_tp = ask_now + target_r_param * (ask_now - sl_price)
                    if not (sl_price < ask_now < rebased_tp):
                        _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={
                            "skip": "geometry_reject_market", "ask": ask_now, "sl": sl_price, "tp": rebased_tp,
                        }).__dict__)
                        continue
                    req = OrderRequest(
                        symbol=symbol,
                        side=decision.side,
                        volume=float(volume),
                        magic=exec_cfg.magic,
                        deviation_points=exec_cfg.deviation_points,
                        stop_price=sl_price,
                        target_price=rebased_tp,
                        comment=f"{run_name}",
                        meta={"reason": decision.reason, "entry_type": "market", **(decision.meta or {})},
                    )
                    fill = executor.send_market(req)
                    _append_jsonl(events_path, RunEvent(
                        kind="fill", time_utc=fill.time_utc,
                        payload={"fill": fill.__dict__, "entry_type": "market"},
                    ).__dict__)
                    if fill.retcode == mt5.TRADE_RETCODE_DONE:
                        risk_state.trades_today += 1
                        risk_state.last_entry_bar_index = bar_index
                else:
                    # Normal path: place BUY_STOP pending at signal bar's high
                    fill = executor.send_pending_buy_stop(
                        symbol=symbol,
                        volume=float(volume),
                        trigger_price=entry_level,
                        sl=sl_price,
                        tp=tp_price,
                        magic=exec_cfg.magic,
                        comment=f"{run_name}",
                        deviation_points=exec_cfg.deviation_points,
                    )
                    _append_jsonl(events_path, RunEvent(
                        kind="fill", time_utc=fill.time_utc,
                        payload={"fill": fill.__dict__, "entry_type": "pending"},
                    ).__dict__)
                    if fill.retcode == mt5.TRADE_RETCODE_DONE and fill.order_ticket:
                        pending_ticket = fill.order_ticket
                        pending_created_bar_index = bar_index

                _write_json(stats_path, {"risk_state": risk_state.__dict__})
                time.sleep(0.2)

            except Exception as exc:
                _append_jsonl(events_path, RunEvent(
                    kind="error", time_utc=_utc_now(),
                    payload={"error": str(exc), "type": type(exc).__name__},
                ).__dict__)
                time.sleep(1.0)
