from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
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
                tick_volume=int(r.get("tick_volume", 0)) if "tick_volume" in r.dtype.names else None,
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

        # Snapshot config
        (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        # Connect MT5
        mt5_cfg = MT5Config(**cfg.get("mt5", {}))
        MT5Connector(mt5_cfg).connect()

        spec = get_symbol_spec(symbol)
        rules = BrokerRules(spec)

        # Strategy
        sp = cfg["strategy"]["params"]
        strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

        # Risk
        rl = cfg["risk"]
        limits = RiskLimits(
            max_trades_per_day=int(rl["max_trades_per_day"]),
            max_daily_loss_usd=float(rl["max_daily_loss_usd"]),
            cooldown_bars=int(rl["cooldown_bars"]),
            time_stop_bars=int(rl["time_stop_bars"]),
        )
        risk_state = RiskState(day=_utc_now().date())

        # Execution
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

        events_path = out_dir / "events.jsonl"
        stats_path = out_dir / "live_state.json"

        last_closed_bar_time: datetime | None = None
        bar_index = 0

        while True:
            now = _utc_now()

            # Reset day counters (UTC day; change if you prefer broker day)
            if risk_state.day != now.date():
                risk_state = RiskState(day=now.date())

            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 250)
            if rates is None or len(rates) < 100:
                time.sleep(1.0)
                continue

            bars = _rates_to_bars(rates)
            closed_bar = bars[-2]  # last closed bar

            if last_closed_bar_time is not None and closed_bar.time_utc <= last_closed_bar_time:
                time.sleep(0.5)
                continue

            last_closed_bar_time = closed_bar.time_utc
            bar_index += 1

            # Guardrails
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

            ok, why = can_trade_today(risk_state, limits)
            if not ok:
                _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": why}).__dict__)
                continue

            ok, why = can_enter_cooldown(risk_state, limits, bar_index)
            if not ok:
                _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"skip": why}).__dict__)
                continue

            # Position check (single-position assumption)
            pos = get_position_state(symbol, exec_cfg.magic)
            if pos.net_volume > 0:
                _append_jsonl(events_path, RunEvent(kind="info", time_utc=now, payload={"pos": "in_market", "net_volume": pos.net_volume}).__dict__)
                _write_json(stats_path, {"risk_state": risk_state.__dict__, "position": pos.__dict__})
                continue

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

            # Size
            entry_price = float(mt5.symbol_info_tick(symbol).ask) if mt5.symbol_info_tick(symbol) else decision.meta.get("entry")
            if entry_price is None:
                entry_price = bars[-2].close

            volume = size_by_stop_distance(
                rules=rules,
                risk_usd=float(rl.get("risk_per_trade_usd", 1.0)),
                entry_price=float(entry_price),
                stop_price=float(decision.stop_price or entry_price),
            )

            req = OrderRequest(
                symbol=symbol,
                side=decision.side,
                volume=float(volume),
                magic=exec_cfg.magic,
                deviation_points=exec_cfg.deviation_points,
                stop_price=decision.stop_price,
                target_price=decision.target_price,
                comment=f"{run_name}",
                meta={"reason": decision.reason, **(decision.meta or {})},
            )

            fill = executor.send_market(req)
            _append_jsonl(events_path, RunEvent(kind="fill", time_utc=fill.time_utc, payload={"fill": fill.__dict__}).__dict__)

            if fill.retcode == mt5.TRADE_RETCODE_DONE:
                risk_state.trades_today += 1
                risk_state.last_entry_bar_index = bar_index

            _write_json(stats_path, {"risk_state": risk_state.__dict__})

            time.sleep(0.2)
