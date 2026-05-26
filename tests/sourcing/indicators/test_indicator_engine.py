"""Tests for the sourcing indicator engine (pure numpy functions)."""

import numpy as np
import pytest

from options_owl.sourcing.data.indicator_engine import (
    IndicatorSet,
    calc_adx,
    calc_atr,
    calc_bollinger,
    calc_ema,
    calc_keltner,
    calc_macd,
    calc_obv_slope,
    calc_rsi,
    calc_vwap,
    compute_indicators,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trending_up(n: int = 50, start: float = 100.0, step: float = 0.5) -> np.ndarray:
    """Generate a steadily rising price series."""
    return np.array([start + i * step for i in range(n)])


def _trending_down(n: int = 50, start: float = 150.0, step: float = 0.5) -> np.ndarray:
    return np.array([start - i * step for i in range(n)])


def _flat(n: int = 50, price: float = 100.0) -> np.ndarray:
    return np.full(n, price)


def _make_candles(closes: np.ndarray, spread: float = 0.5) -> list[dict]:
    """Build OHLCV candle dicts from close prices."""
    candles = []
    for i, c in enumerate(closes):
        candles.append({
            "open": float(c - spread / 4),
            "high": float(c + spread),
            "low": float(c - spread),
            "close": float(c),
            "volume": 1000 + i * 10,
        })
    return candles


# ---------------------------------------------------------------------------
# calc_ema
# ---------------------------------------------------------------------------

class TestCalcEMA:
    def test_short_input_returns_nan(self):
        vals = np.array([1.0, 2.0])
        result = calc_ema(vals, period=5)
        assert np.all(np.isnan(result))

    def test_seed_is_sma(self):
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        result = calc_ema(vals, period=5)
        assert result[4] == pytest.approx(30.0)  # SMA of first 5

    def test_ema9_tracks_uptrend(self):
        closes = _trending_up(50)
        ema = calc_ema(closes, 9)
        # EMA should lag behind price in an uptrend
        assert ema[-1] < closes[-1]
        assert ema[-1] > closes[-10]  # but still rising

    def test_output_length_matches_input(self):
        vals = np.arange(100, dtype=np.float64)
        result = calc_ema(vals, 21)
        assert len(result) == 100

    def test_leading_nans(self):
        vals = np.arange(20, dtype=np.float64)
        result = calc_ema(vals, 9)
        assert np.all(np.isnan(result[:8]))
        assert not np.isnan(result[8])


# ---------------------------------------------------------------------------
# calc_rsi
# ---------------------------------------------------------------------------

class TestCalcRSI:
    def test_insufficient_data(self):
        assert calc_rsi(np.array([1.0, 2.0]), period=9) == 50.0

    def test_strong_uptrend_high_rsi(self):
        closes = _trending_up(50)
        rsi = calc_rsi(closes, 9)
        assert rsi > 70  # strong uptrend = overbought

    def test_strong_downtrend_low_rsi(self):
        closes = _trending_down(50)
        rsi = calc_rsi(closes, 9)
        assert rsi < 30  # strong downtrend = oversold

    def test_flat_market_neutral_rsi(self):
        closes = _flat(50)
        rsi = calc_rsi(closes, 9)
        # No gains or losses — should return 50 (or 100 if avg_loss==0)
        # With flat prices, no deltas, so avg_gain=0, avg_loss=0 → 100.0
        assert rsi == 100.0

    def test_rsi_bounded_0_100(self):
        # Random walk
        rng = np.random.default_rng(42)
        closes = 100 + np.cumsum(rng.normal(0, 1, 200))
        rsi = calc_rsi(closes, 9)
        assert 0 <= rsi <= 100


# ---------------------------------------------------------------------------
# calc_macd
# ---------------------------------------------------------------------------

class TestCalcMACD:
    def test_insufficient_data(self):
        line, signal, hist = calc_macd(np.array([1.0, 2.0, 3.0]))
        assert line == 0.0
        assert signal == 0.0
        assert hist == 0.0

    def test_uptrend_positive_macd(self):
        closes = _trending_up(50)
        line, signal, hist = calc_macd(closes, fast=5, slow=13, signal=4)
        assert line > 0  # fast > slow in uptrend

    def test_downtrend_negative_macd(self):
        closes = _trending_down(50)
        line, signal, hist = calc_macd(closes, fast=5, slow=13, signal=4)
        assert line < 0

    def test_returns_three_floats(self):
        closes = _trending_up(50)
        result = calc_macd(closes)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# calc_bollinger
# ---------------------------------------------------------------------------

class TestCalcBollinger:
    def test_insufficient_data(self):
        closes = np.array([100.0, 101.0])
        upper, mid, lower, width = calc_bollinger(closes, period=20)
        assert upper == mid == lower == 101.0
        assert width == 0.0

    def test_flat_market_zero_width(self):
        closes = _flat(50)
        upper, mid, lower, width = calc_bollinger(closes, 20)
        assert upper == mid == lower == 100.0
        assert width == pytest.approx(0.0)

    def test_upper_above_lower(self):
        rng = np.random.default_rng(42)
        closes = 100 + np.cumsum(rng.normal(0, 1, 50))
        upper, mid, lower, width = calc_bollinger(closes, 20)
        assert upper > mid > lower
        assert width > 0

    def test_width_increases_with_volatility(self):
        low_vol = _flat(50) + np.random.default_rng(1).normal(0, 0.1, 50)
        high_vol = _flat(50) + np.random.default_rng(1).normal(0, 5.0, 50)
        _, _, _, w_low = calc_bollinger(low_vol, 20)
        _, _, _, w_high = calc_bollinger(high_vol, 20)
        assert w_high > w_low


# ---------------------------------------------------------------------------
# calc_atr
# ---------------------------------------------------------------------------

class TestCalcATR:
    def test_insufficient_data(self):
        h = np.array([101.0, 102.0])
        l = np.array([99.0, 98.0])
        c = np.array([100.0, 101.0])
        atr = calc_atr(h, l, c, 14)
        assert atr > 0  # falls back to mean(H-L)

    def test_flat_market_low_atr(self):
        n = 50
        c = _flat(n)
        h = c + 0.1
        l = c - 0.1
        atr = calc_atr(h, l, c, 14)
        assert atr < 1.0

    def test_volatile_market_high_atr(self):
        n = 50
        c = _flat(n)
        h = c + 5.0
        l = c - 5.0
        atr = calc_atr(h, l, c, 14)
        assert atr > 5.0

    def test_atr_positive(self):
        rng = np.random.default_rng(42)
        c = 100 + np.cumsum(rng.normal(0, 1, 50))
        h = c + np.abs(rng.normal(0, 0.5, 50))
        l = c - np.abs(rng.normal(0, 0.5, 50))
        atr = calc_atr(h, l, c, 14)
        assert atr > 0


# ---------------------------------------------------------------------------
# calc_keltner
# ---------------------------------------------------------------------------

class TestCalcKeltner:
    def test_upper_above_lower(self):
        c = _trending_up(50)
        h = c + 1.0
        l = c - 1.0
        upper, lower = calc_keltner(c, h, l, 20, 1.5)
        assert upper > lower

    def test_insufficient_data(self):
        c = np.array([100.0])
        h = np.array([101.0])
        l = np.array([99.0])
        upper, lower = calc_keltner(c, h, l, 20)
        assert upper == lower == 100.0


# ---------------------------------------------------------------------------
# calc_vwap
# ---------------------------------------------------------------------------

class TestCalcVWAP:
    def test_empty_returns_zero(self):
        assert calc_vwap(np.array([]), np.array([]), np.array([]), np.array([])) == 0.0

    def test_zero_volume_returns_zero(self):
        c = np.array([100.0])
        assert calc_vwap(c + 1, c - 1, c, np.array([0.0])) == 0.0

    def test_equal_volume_returns_mean_typical(self):
        h = np.array([102.0, 104.0])
        l = np.array([98.0, 96.0])
        c = np.array([100.0, 100.0])
        v = np.array([100.0, 100.0])
        vwap = calc_vwap(h, l, c, v)
        # typical prices: (102+98+100)/3=100, (104+96+100)/3=100
        assert vwap == pytest.approx(100.0)

    def test_volume_weighted(self):
        h = np.array([110.0, 210.0])
        l = np.array([90.0, 190.0])
        c = np.array([100.0, 200.0])
        v = np.array([1000.0, 1.0])
        vwap = calc_vwap(h, l, c, v)
        # Heavily weighted toward first bar (typical = 100)
        assert vwap < 110


# ---------------------------------------------------------------------------
# calc_obv_slope
# ---------------------------------------------------------------------------

class TestCalcOBVSlope:
    def test_insufficient_data(self):
        assert calc_obv_slope(np.array([1.0, 2.0]), np.array([100, 200]), 5) == 0.0

    def test_rising_prices_positive_slope(self):
        closes = _trending_up(20)
        volumes = np.full(20, 1000.0)
        slope = calc_obv_slope(closes, volumes, 5)
        assert slope > 0

    def test_falling_prices_negative_slope(self):
        closes = _trending_down(20)
        volumes = np.full(20, 1000.0)
        slope = calc_obv_slope(closes, volumes, 5)
        assert slope < 0


# ---------------------------------------------------------------------------
# calc_adx
# ---------------------------------------------------------------------------

class TestCalcADX:
    def test_insufficient_data(self):
        c = np.array([100.0] * 10)
        assert calc_adx(c + 1, c - 1, c, 14) == 0.0

    def test_trending_market_high_adx(self):
        c = _trending_up(50)
        h = c + 0.5
        l = c - 0.5
        adx = calc_adx(h, l, c, 14)
        assert adx > 0  # trending = nonzero ADX

    def test_adx_nonnegative(self):
        rng = np.random.default_rng(42)
        c = 100 + np.cumsum(rng.normal(0, 1, 50))
        h = c + np.abs(rng.normal(0, 0.5, 50))
        l = c - np.abs(rng.normal(0, 0.5, 50))
        adx = calc_adx(h, l, c, 14)
        assert adx >= 0


# ---------------------------------------------------------------------------
# compute_indicators (integration)
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_empty_candles_returns_defaults(self):
        result = compute_indicators([])
        assert isinstance(result, IndicatorSet)
        assert result.rsi9 == 50.0
        assert result.ema9 == 0.0

    def test_few_candles_returns_defaults(self):
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}]
        result = compute_indicators(candles)
        assert result.rsi9 == 50.0

    def test_full_candles_all_fields_populated(self):
        closes = _trending_up(78, start=100, step=0.3)
        candles = _make_candles(closes)
        result = compute_indicators(candles)

        # EMA
        assert result.ema9 > 0
        assert result.ema21 > 0
        assert result.ema9 > result.ema21  # uptrend

        # RSI
        assert 50 < result.rsi9 <= 100

        # MACD
        assert result.macd_line > 0  # uptrend

        # Bollinger
        assert result.bb_upper > result.bb_mid > result.bb_lower
        assert result.bb_width > 0

        # ATR
        assert result.atr14 > 0

        # Keltner
        assert result.keltner_upper > result.keltner_lower

        # VWAP
        assert result.vwap > 0

        # Volume
        assert result.volume_ratio > 0

        # Price context
        assert result.last_close == pytest.approx(float(closes[-1]))

    def test_ema_cross_strength_bounded(self):
        closes = _trending_up(78, start=100, step=2.0)  # strong trend
        candles = _make_candles(closes)
        result = compute_indicators(candles)
        assert -1.0 <= result.ema_cross_strength <= 1.0

    def test_downtrend_bearish_indicators(self):
        closes = _trending_down(78, start=200, step=0.5)
        candles = _make_candles(closes, spread=1.0)
        result = compute_indicators(candles)

        assert result.ema9 < result.ema21  # bearish cross
        assert result.rsi9 < 50  # oversold-ish
        assert result.macd_line < 0  # bearish MACD
        assert result.ema_cross_strength < 0  # bearish

    def test_squeeze_detection(self):
        # Tight range = BB inside Keltner = squeeze
        closes = _flat(78) + np.random.default_rng(42).normal(0, 0.01, 78)
        candles = _make_candles(closes, spread=0.02)
        result = compute_indicators(candles)
        # With very tight prices, BB should be inside Keltner
        assert result.bb_squeeze == True

    def test_missing_volume_key_defaults_to_zero(self):
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(10)]
        result = compute_indicators(candles)
        assert result.volume_ratio == 1.0  # no division errors

    def test_ema200_with_sufficient_data(self):
        closes = _trending_up(250, start=50, step=0.1)
        candles = _make_candles(closes)
        result = compute_indicators(candles)
        assert result.ema200 > 0
        assert result.ema200 != result.last_close  # actually computed, not fallback

    def test_ema200_fallback_with_insufficient_data(self):
        closes = _trending_up(50)
        candles = _make_candles(closes)
        result = compute_indicators(candles)
        # Falls back to last close
        assert result.ema200 == pytest.approx(float(closes[-1]))
