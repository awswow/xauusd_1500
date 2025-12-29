# src/xauusd100/strategy/pullback_trend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from ..engine.models import Decision, Side
from .base import Strategy, StrategyContext


# ----------------------------
# Indicator helpers
# ----------------------------

def sma(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(x) < n:
        return np.array([], dtype=float)
    w = np.ones(n, dtype=float) / float(n)
    return np.convolve(x, w, mode="valid")


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """
    Wilder-style ATR (RMA). Returns array aligned to input length with NaNs for warmup.
    """
    if n <= 0 or len(close) < n:
        return np.array([], dtype=float)

    tr = _true_range(high, low, close)
    out = np.full_like(tr, np.nan, dtype=float)

    out[n - 1] = float(np.mean(tr[:n]))
    alpha = 1.0 / float(n)
    for i in range(n, len(tr)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])

    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """
    Minimal Wilder ADX implementation. Good enough for gating and logging.
    Returns array aligned to input length with NaNs for warmup.
    """
    if n <= 0 or len(close) < (2 * n):
        return np.array([], dtype=float)

    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)

    def wilder_sum(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=float)
        out[n - 1] = float(np.sum(x[:n]))
        for i in range(n, len(x)):
            out[i] = out[i - 1] - (out[i - 1] / float(n)) + float(x[i])
        return out

    tr_s = wilder_sum(tr)
    p_dm_s = wilder_sum(plus_dm)
    m_dm_s = wilder_sum(minus_dm)

    plus_di = 100.0 * (p_dm_s / tr_s)
    minus_di = 100.0 * (m_dm_s / tr_s)

    denom = plus_di + minus_di
    dx = np.full_like(denom, np.nan, dtype=float)
    valid = denom > 0
    dx[valid] = 100.0 * (np.abs(plus_di[valid] - minus_di[valid]) / denom[valid])

    adx_out = np.full_like(dx, np.nan, dtype=float)

    # seed ADX at index 2n-1 as mean of DX over [n, 2n)
    seed_index = (2 * n) - 1
    adx_out[seed_index] = float(np.nanmean(dx[n:2 * n]))
    for i in range(2 * n, len(dx)):
        adx_out[i] = ((adx_out[i - 1] * (n - 1)) + dx[i]) / float(n)

    return adx_out


def _finite(x: Optional[float]) -> bool:
    return x is not None and np.isfinite(x)


# ----------------------------
# Strategy
# ----------------------------

@dataclass(frozen=True)
class PullbackTrendParams:
    # Core lengths
    atr_len: int = 14
    sma_len: int = 60

    # Regime / trend-quality gates
    adx_min: float = 20.0                 # ADX must be >= this
    sma_slope_min: float = 0.0            # SMA slope (points per bar) must be >= this
    sma_slope_min_atr_frac: float = 0.0   # optional: slope must be >= frac * ATR (normalized slope gate)
    atr_min_points: float = 0.0           # optional: ATR must be >= this (avoid compression)
    atr_max_points: float = 0.0           # optional: ATR must be <= this (avoid extreme volatility)

    # Entry logic
    pullback_atr: float = 0.30            # "touch near SMA" threshold in ATR units
    pullback_lookback: int = 3            # how many bars to look back for touch
    long_only: bool = True

    # Risk/targets
    stop_atr: float = 1.20
    target_r: float = 2.0


class PullbackTrendStrategy(Strategy):
    """
    Long-only pullback trend:
      - Regime gates: close > SMA, SMA slope positive/threshold, ADX >= adx_min, ATR not too low/high
      - Setup: recent low touched near SMA within pullback_atr * ATR
      - Decision produces stop/target and logs regime metrics in meta
      - Entry confirmation (break prev high) should be implemented in execution layer (backtest/live),
        but we still log enough here to build regime filters.
    """

    def __init__(self, params: PullbackTrendParams):
        self.p = params
        self._last_diag: Dict[str, Any] = {}

    def last_diag(self) -> Dict[str, Any]:
        return dict(self._last_diag or {})

    def _reject(self, reason: str, **kw) -> Optional[Decision]:
        self._last_diag = {"reason": reason, **kw}
        return None

    def on_bar(self, ctx: StrategyContext) -> Optional[Decision]:
        bars = list(ctx.bars)
        need = max(self.p.sma_len + 2, 2 * self.p.atr_len + 5)
        if len(bars) < need:
            return self._reject("warmup", need=need, have=len(bars))

        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)

        # Indicators
        sma_arr = sma(close, self.p.sma_len)
        if len(sma_arr) < 2:
            return self._reject("sma_insufficient", sma_len=int(self.p.sma_len), sma_n=int(len(sma_arr)))

        sma_last = float(sma_arr[-1])
        sma_prev = float(sma_arr[-2])
        sma_slope = sma_last - sma_prev

        atr_arr = atr(high, low, close, self.p.atr_len)
        adx_arr = adx(high, low, close, self.p.atr_len)

        atr_last = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else None
        adx_last = float(adx_arr[-1]) if len(adx_arr) and np.isfinite(adx_arr[-1]) else None

        last_close = float(close[-1])

        # Must have valid ATR/ADX
        if not _finite(atr_last) or float(atr_last) <= 0:
            return self._reject(
                "atr_invalid",
                atr=atr_last,
                adx=adx_last,
                sma=float(sma_last),
                close=float(last_close),
                sma_slope=float(sma_slope),
            )

        if not _finite(adx_last):
            return self._reject(
                "adx_invalid",
                atr=float(atr_last),
                adx=adx_last,
                sma=float(sma_last),
                close=float(last_close),
                sma_slope=float(sma_slope),
            )

        atr_last_f = float(atr_last)
        adx_last_f = float(adx_last)

        # Optional ATR compression / chaos filters
        if self.p.atr_min_points and atr_last_f < float(self.p.atr_min_points):
            return self._reject(
                "atr_below_min",
                atr=float(atr_last_f),
                atr_min=float(self.p.atr_min_points),
                adx=float(adx_last_f),
            )

        if self.p.atr_max_points and atr_last_f > float(self.p.atr_max_points):
            return self._reject(
                "atr_above_max",
                atr=float(atr_last_f),
                atr_max=float(self.p.atr_max_points),
                adx=float(adx_last_f),
            )

        # Core trend direction gate (LONG)
        if last_close <= sma_last:
            return self._reject(
                "close_below_sma",
                close=float(last_close),
                sma=float(sma_last),
                adx=float(adx_last_f),
                atr=float(atr_last_f),
                sma_slope=float(sma_slope),
            )

        # Slope gates (absolute and/or normalized)
        if sma_slope < float(self.p.sma_slope_min):
            return self._reject(
                "sma_slope_below_min",
                sma_slope=float(sma_slope),
                sma_slope_min=float(self.p.sma_slope_min),
                sma=float(sma_last),
                atr=float(atr_last_f),
                adx=float(adx_last_f),
            )

        if self.p.sma_slope_min_atr_frac:
            thresh = float(self.p.sma_slope_min_atr_frac) * float(atr_last_f)
            if sma_slope < thresh:
                return self._reject(
                    "sma_slope_below_atr_frac",
                    sma_slope=float(sma_slope),
                    thresh=float(thresh),
                    sma_slope_atr=float(sma_slope / atr_last_f) if atr_last_f else None,
                    atr=float(atr_last_f),
                    adx=float(adx_last_f),
                )

        # ADX gate
        if adx_last_f < float(self.p.adx_min):
            return self._reject(
                "adx_below_min",
                adx=float(adx_last_f),
                adx_min=float(self.p.adx_min),
                atr=float(atr_last_f),
                sma=float(sma_last),
                close=float(last_close),
                sma_slope=float(sma_slope),
            )

        # Pullback touch near SMA
        lookback = max(1, int(self.p.pullback_lookback))
        pullback_thresh = float(self.p.pullback_atr) * float(atr_last_f)

        recent_lows = low[-lookback:]
        touched = bool(np.any((sma_last - recent_lows) <= pullback_thresh))
        if not touched:
            return self._reject(
                "pullback_not_touched",
                lookback=int(lookback),
                pullback_atr=float(self.p.pullback_atr),
                pullback_thresh=float(pullback_thresh),
                sma=float(sma_last),
                min_recent_low=float(np.min(recent_lows)) if len(recent_lows) else None,
                adx=float(adx_last_f),
                atr=float(atr_last_f),
                close=float(last_close),
                sma_slope=float(sma_slope),
            )

        # Decision uses entry at last close as a reference; execution layer will decide actual entry price
        entry_ref = last_close
        stop = entry_ref - (float(self.p.stop_atr) * float(atr_last_f))
        risk_dist = entry_ref - stop
        if risk_dist <= 0:
            return self._reject("risk_dist_nonpositive", entry_ref=float(entry_ref), stop=float(stop), atr=float(atr_last_f))

        target = entry_ref + (float(self.p.target_r) * float(risk_dist))

        # Meta logging for regime analysis
        meta: Dict[str, Any] = {
            "adx": float(adx_last_f),
            "atr": float(atr_last_f),
            "sma": float(sma_last),
            "close": float(last_close),
            "sma_slope": float(sma_slope),
            "sma_slope_atr": float(sma_slope / atr_last_f) if atr_last_f else None,
            "pullback_thresh": float(pullback_thresh),
            "pullback_touched": touched,
        }

        # record pass diag too (useful to confirm what “good” looks like)
        self._last_diag = {"reason": "pass", **meta}

        return Decision(
            time_utc=bars[-1].time_utc,
            symbol=ctx.symbol,
            side=Side.BUY,
            stop_price=float(stop),
            target_price=float(target),
            reason="pullback_trend_long",
            meta=meta,
        )
