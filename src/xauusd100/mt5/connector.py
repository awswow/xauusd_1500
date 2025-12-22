from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5


@dataclass
class MT5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None
    timeout_ms: int = 10000


class MT5ConnectionError(RuntimeError):
    pass


class MT5Connector:
    def __init__(self, cfg: MT5Config):
        self.cfg = cfg

    def connect(self) -> None:
        kwargs = {"timeout": self.cfg.timeout_ms}

        # Only pass args if they are not None / empty
        if self.cfg.path:
            kwargs["path"] = self.cfg.path
        if self.cfg.login is not None:
            kwargs["login"] = int(self.cfg.login)
        if self.cfg.password:
            kwargs["password"] = self.cfg.password
        if self.cfg.server:
            kwargs["server"] = self.cfg.server

        ok = mt5.initialize(**kwargs)
        if not ok:
            raise MT5ConnectionError(f"mt5.initialize failed: {mt5.last_error()}")


    def shutdown(self) -> None:
        mt5.shutdown()
