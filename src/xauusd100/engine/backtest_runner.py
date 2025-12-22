from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .models import Bar, Side
from .utils import utc_now, iso_ts
from ..strategy.pullback_trend import PullbackTrendStrategy, PullbackTrendParams
from ..strategy.base import StrategyContext
from ..reporting.export import write_csv, write_json


def _rates_to_bars_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _df_to_bars(df: pd.DataFrame) -> list[Bar]:
    out = []
    for _, r in df.iterrows():
        out.append(Bar(
            time_utc=r["time_utc"].to_pydatetime(),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            tick_volume=int(r.get("tick_volume", 0)),
        ))
    return out


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    stats: dict


def run_backtest_from_csv(cfg_path: str) -> Path:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))

    prices_csv = cfg["data"]["prices_csv"]
    df = pd.read_csv(prices_csv)

    # Accept either `time` (epoch seconds) or `time_utc` ISO
    if "time" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    elif "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    else:
        raise ValueError("CSV must have `time` (epoch seconds) or `time_utc` column")

    bars = _df_to_bars(df)

    sp = cfg["strategy"]["params"]
    strategy = PullbackTrendStrategy(PullbackTrendParams(**sp))

    out_root = Path(cfg.get("reporting", {}).get("out_dir", "data/derived/runs"))
    run_id = f"{iso_ts(utc_now())}_{cfg.get('run_name','backtest')}"
    run_dir = out_root/run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"config.snapshot.yaml").write_text(Path(cfg_path).read_text(encoding='utf-8'), encoding='utf-8')

    in_pos = False
    entry_i = None
    entry_price = None
    stop_price = None
    target_price = None

    trades = []

    for i in range(200, len(bars)):
        ctx = StrategyContext(
            symbol=cfg["symbol"],
            timeframe=cfg["timeframe"],
            bars=bars[:i],
            bar_index=i,
            meta={},
        )

        b = bars[i-1]  # latest closed bar

        if in_pos:
            # Check stop/target intrabar using high/low of the new closed bar
            hit_stop = (b.low <= stop_price)
            hit_target = (b.high >= target_price)

            exit_reason = None
            exit_price = None
            if hit_stop and hit_target:
                # Conservative: assume stop hit first
                exit_reason = "stop"
                exit_price = stop_price
            elif hit_stop:
                exit_reason = "stop"
                exit_price = stop_price
            elif hit_target:
                exit_reason = "target"
                exit_price = target_price

            if exit_reason:
                pnl_points = exit_price - entry_price
                trades.append({
                    "entry_time_utc": bars[entry_i].time_utc.isoformat(),
                    "exit_time_utc": b.time_utc.isoformat(),
                    "side": "BUY",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "pnl_points": pnl_points,
                    "exit_reason": exit_reason,
                })
                in_pos = False
                entry_i = None
                continue

        if not in_pos:
            d = strategy.on_bar(ctx)
            if d is None:
                continue
            # enter next bar open (simple)
            entry_i = i
            entry_price = b.close
            stop_price = float(d.stop_price)
            target_price = float(d.target_price)
            in_pos = True

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        stats = {"trades": 0}
    else:
        stats = {
            "trades": int(len(trades_df)),
            "win_rate": float((trades_df["pnl_points"] > 0).mean()),
            "avg_win": float(trades_df.loc[trades_df["pnl_points"] > 0, "pnl_points"].mean() or 0.0),
            "avg_loss": float(trades_df.loc[trades_df["pnl_points"] <= 0, "pnl_points"].mean() or 0.0),
            "total_points": float(trades_df["pnl_points"].sum()),
        }

    write_csv(run_dir/"trades.csv", trades_df)
    write_json(run_dir/"stats.json", stats)

    return run_dir
