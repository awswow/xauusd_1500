from __future__ import annotations

import argparse
import asyncio
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from aiogram import Bot

UTC = timezone.utc


@dataclass(frozen=True)
class WatchCfg:
    token: str
    chat_id: str
    process_keyword: str
    log_file: Path
    check_every_sec: int
    max_log_age_sec: int
    alert_cooldown_sec: int  # throttle alerts


def is_process_running_windows(keyword: str) -> bool:
    """
    Reliable check: search Win32_Process CommandLine for the script name.
    Works even though tasklist only shows python.exe.
    """
    ps = (
        "powershell -NoProfile -Command "
        f"\"$p = Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -and $_.CommandLine -like '*{keyword}*' }} | "
        "Select-Object -First 1; "
        "if ($p) { 'YES' } else { 'NO' }\""
    )
    try:
        out = subprocess.check_output(ps, shell=True, text=True, errors="ignore").strip().upper()
        return out == "YES"
    except Exception:
        return False


def log_is_fresh(path: Path, max_age_sec: int) -> Tuple[bool, Optional[int]]:
    """
    Returns (fresh, age_seconds).
    """
    if not path.exists():
        return False, None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age = int((datetime.now(tz=UTC) - mtime).total_seconds())
    return age <= max_age_sec, age


async def send(bot: Bot, chat_id: str, text: str) -> None:
    await bot.send_message(chat_id=chat_id, text=text)


async def main_async(cfg: WatchCfg, once: bool) -> None:
    bot = Bot(token=cfg.token)

    await send(
        bot,
        cfg.chat_id,
        "Watchdog started.\n"
        f"- process_keyword: {cfg.process_keyword}\n"
        f"- log_file: {cfg.log_file}\n"
        f"- check_every_sec: {cfg.check_every_sec}\n"
        f"- max_log_age_sec: {cfg.max_log_age_sec}\n"
        f"- alert_cooldown_sec: {cfg.alert_cooldown_sec}",
    )

    last_alert_ts: Optional[float] = None

    while True:
        running = is_process_running_windows(cfg.process_keyword)
        fresh, age = log_is_fresh(cfg.log_file, cfg.max_log_age_sec)

        ok = running and fresh

        if not ok:
            now_ts = datetime.now(tz=UTC).timestamp()
            can_alert = last_alert_ts is None or (now_ts - last_alert_ts) >= cfg.alert_cooldown_sec

            if can_alert:
                if not running:
                    await send(bot, cfg.chat_id, f"ALERT: Process not running: {cfg.process_keyword}")
                if not fresh:
                    if age is None:
                        await send(bot, cfg.chat_id, f"ALERT: Log file missing: {cfg.log_file}")
                    else:
                        await send(bot, cfg.chat_id, f"ALERT: Log stale: {cfg.log_file} (age={age}s)")
                last_alert_ts = now_ts

        if once:
            await send(bot, cfg.chat_id, f"Watchdog check: {'OK' if ok else 'NOT OK'}")
            break

        await asyncio.sleep(cfg.check_every_sec)

    await bot.session.close()


def build_cfg_from_args() -> tuple[WatchCfg, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Telegram bot token")
    ap.add_argument("--chat_id", required=True, help="Telegram chat_id (user or group)")
    ap.add_argument("--process_keyword", default="demo_executor_mt5.py")
    ap.add_argument("--log_file", default="data/derived/demo/demo_events.csv")
    ap.add_argument("--check_every_sec", type=int, default=60)
    ap.add_argument("--max_log_age_sec", type=int, default=240)  # heartbeat=60 => 240 is safe
    ap.add_argument("--alert_cooldown_sec", type=int, default=900)  # 15 min throttle
    ap.add_argument("--once", action="store_true", help="Run one check and exit")
    args = ap.parse_args()

    cfg = WatchCfg(
        token=args.token,
        chat_id=args.chat_id,
        process_keyword=args.process_keyword,
        log_file=Path(args.log_file),
        check_every_sec=int(args.check_every_sec),
        max_log_age_sec=int(args.max_log_age_sec),
        alert_cooldown_sec=int(args.alert_cooldown_sec),
    )
    return cfg, bool(args.once)


if __name__ == "__main__":
    cfg, once = build_cfg_from_args()
    asyncio.run(main_async(cfg, once=once))
