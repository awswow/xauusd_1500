# src/xauusd100/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float
    peak_index: int
    trough_index: int
    recovery_index: Optional[int]


def equity_curve_from_r(r: pd.Series) -> pd.Series:
    """
    Returns equity curve in R units. Always returns a Series (possibly empty).
    """
    if r is None:
        return pd.Series(dtype=float)
    r = pd.Series(r, dtype=float)
    if r.empty:
        return pd.Series(dtype=float)
    return r.fillna(0.0).cumsum()


def max_drawdown_stats(equity_r: pd.Series) -> DrawdownStats:
    """
    Robust max drawdown stats.
    - If equity_r is empty: returns zeros and None recovery.
    - If equity_r has 1 point: drawdown is zero.
    """
    if equity_r is None:
        equity_r = pd.Series(dtype=float)

    equity_r = pd.Series(equity_r, dtype=float)

    if equity_r.empty:
        return DrawdownStats(
            max_drawdown=0.0,
            peak_index=0,
            trough_index=0,
            recovery_index=None,
        )

    peak = equity_r.cummax()
    dd = equity_r - peak

    # dd will be same length as equity_r; still guard for safety
    if dd.empty:
        return DrawdownStats(
            max_drawdown=0.0,
            peak_index=0,
            trough_index=0,
            recovery_index=None,
        )

    trough_i = int(dd.idxmin())
    max_dd = float(dd.loc[trough_i])

    # peak index: last peak at or before trough
    peak_i = int(peak.loc[:trough_i].idxmax())

    # recovery: first index after trough where equity >= previous peak
    prev_peak_val = float(peak.loc[peak_i])
    recovery_i = None
    after = equity_r.loc[trough_i:]
    rec = after[after >= prev_peak_val]
    if not rec.empty:
        recovery_i = int(rec.index[0])

    return DrawdownStats(
        max_drawdown=max_dd,
        peak_index=peak_i,
        trough_index=trough_i,
        recovery_index=recovery_i,
    )
