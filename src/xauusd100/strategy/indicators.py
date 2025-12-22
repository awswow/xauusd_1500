from __future__ import annotations

import numpy as np


def sma(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.astype(float)
    out = np.full_like(values, np.nan, dtype=float)
    c = np.cumsum(values.astype(float))
    out[length-1:] = (c[length-1:] - np.concatenate(([0.0], c[:-length])) ) / float(length)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Wilder style ATR (RMA)
    out = np.full_like(tr, np.nan, dtype=float)
    if len(tr) == 0:
        return out
    out[length-1] = np.nanmean(tr[:length])
    alpha = 1.0 / float(length)
    for i in range(length, len(tr)):
        out[i] = out[i-1] + alpha * (tr[i] - out[i-1])
    return out
