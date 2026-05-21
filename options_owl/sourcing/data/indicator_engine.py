"""Compute all technical indicators locally from raw OHLCV bars.

Pure functions — no I/O, no side effects, fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndicatorSet:
    """All computed indicators for a single ticker at a point in time."""

    ema9: float = 0.0
    ema21: float = 0.0
    ema200: float = 0.0
    ema_cross_strength: float = 0.0  # -1 to +1

    rsi9: float = 50.0
    macd_histogram: float = 0.0
    macd_signal: float = 0.0

    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_squeeze: bool = False

    vwap: float = 0.0
    vwap_slope: float = 0.0

    atr14: float = 0.0
    atr_expanding: bool = False

    keltner_upper: float = 0.0
    keltner_lower: float = 0.0

    volume_ratio: float = 1.0  # current / 20-bar avg
    obv_slope: float = 0.0
    mfi: float = 50.0

    adx: float = 0.0


def compute_indicators(candles: list[dict]) -> IndicatorSet:
    """Compute full indicator set from OHLCV candle data.

    Args:
        candles: List of dicts with keys: open, high, low, close, volume.
                 Ordered oldest-first.

    Returns:
        IndicatorSet with all computed values.
    """
    raise NotImplementedError("Phase 1: implement all indicator computations")
