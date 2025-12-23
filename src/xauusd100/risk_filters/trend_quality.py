# src/xauusd100/risk_filters/trend_quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np

from xauusd100.engine.models import Bar


# ----------------------------
# Indicators (self-contained)
# ----------------------------
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder ATR (RMA). Aligned to input length with NaNs for warmup."""
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
    """Minimal Wilder ADX. Aligned to input length with NaNs for warmup."""
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
    seed_index = (2 * n) - 1
    adx_out[seed_index] = float(np.nanmean(dx[n : 2 * n]))
    for i in range(2 * n, len(dx)):
        adx_out[i] = ((adx_out[i - 1] * (n - 1)) + dx[i]) / float(n)

    return adx_out


# ----------------------------
# Trend-quality gate
# ----------------------------
@dataclass(frozen=True)
class TrendQualityParams:
    tq_adx_len: int = 14
    tq_atr_len: int = 14

    tq_adx_min: float = 15.0
    tq_adx_rise_bars: int = 0

    tq_atr_ref_window: int = 120
    tq_atr_quantile: float = 0.20


def _finite(x: Optional[float]) -> bool:
    return x is not None and np.isfinite(x)


def trend_quality_gate(
    bars: list[Bar],
    p: TrendQualityParams,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns: (pass, reason, meta)
      - pass=True  => gate allows trading
      - pass=False => block with reason string + meta diagnostics
    """
    need = max(2 * int(p.tq_adx_len) + 5, int(p.tq_atr_ref_window) + int(p.tq_atr_len) + 5)
    if len(bars) < need:
        return False, "insufficient_bars", {"need": need, "have": len(bars)}

    high = np.array([b.high for b in bars], dtype=float)
    low = np.array([b.low for b in bars], dtype=float)
    close = np.array([b.close for b in bars], dtype=float)

    atr_arr = atr(high, low, close, int(p.tq_atr_len))
    adx_arr = adx(high, low, close, int(p.tq_adx_len))

    atr_now = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else None
    adx_now = float(adx_arr[-1]) if len(adx_arr) and np.isfinite(adx_arr[-1]) else None

    if not _finite(atr_now) or atr_now <= 0:
        return False, "atr_invalid", {"atr_now": atr_now}
    if not _finite(adx_now):
        return False, "adx_invalid", {"adx_now": adx_now}

    # ADX threshold
    if float(adx_now) < float(p.tq_adx_min):
        return False, "adx_below_min", {"adx_now": adx_now, "adx_min": float(p.tq_adx_min)}

    # ADX rise (optional)
    rise_bars = int(p.tq_adx_rise_bars or 0)
    adx_prev = None
    if rise_bars > 0:
        if len(adx_arr) <= rise_bars or not np.isfinite(adx_arr[-1 - rise_bars]):
            return False, "adx_rise_unavailable", {"adx_now": adx_now, "rise_bars": rise_bars}
        adx_prev = float(adx_arr[-1 - rise_bars])
        if float(adx_now) < float(adx_prev):
            return False, "adx_not_rising", {"adx_now": adx_now, "adx_prev": adx_prev, "rise_bars": rise_bars}

    # ATR quantile gate (core)
    w = int(p.tq_atr_ref_window)
    q = float(p.tq_atr_quantile)
    if w > 0:
        # compute quantile over the last w ATR values (finite only), excluding current if you want;
        # including current is fine for gating.
        tail = atr_arr[-w:]
        tail = tail[np.isfinite(tail)]
        if len(tail) < max(10, int(0.5 * w)):
            return False, "atr_ref_insufficient", {"atr_ref_have": int(len(tail)), "atr_ref_window": w}
        atr_q = float(np.quantile(tail, q))
        if float(atr_now) < float(atr_q):
            return False, "atr_below_q", {"atr_now": float(atr_now), "q": float(atr_q), "quantile": q, "window": w}
    else:
        atr_q = None

    meta = {
        "adx_now": float(adx_now),
        "adx_prev": None if adx_prev is None else float(adx_prev),
        "atr_now": float(atr_now),
        "atr_ref_window": int(w),
        "atr_quantile": float(q),
    }
    if w > 0:
        meta["atr_q"] = float(atr_q)  # type: ignore[name-defined]
    return True, "ok", meta
