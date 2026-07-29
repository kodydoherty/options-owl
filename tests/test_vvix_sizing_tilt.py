"""Tests for the VVIX sizing tilt (validated 2026 flow+ML, no-lookahead).

Covers the multiplier curve (vvix_size_mult), the trailing-percentile fetch (fetch_vvix_percentile,
yfinance mocked), and the safety guarantees: off by default, fail-open on a missing feed, and never
skips/zeroes a trade (it's a SIZING tilt, not a gate).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from options_owl.config.settings import Settings
from options_owl.risk import vix_regime
from options_owl.risk.vinny_strategy import vvix_size_mult


class TestVvixSizeMult:
    def test_low_pctile_sizes_up(self):
        mult, desc = vvix_size_mult(0.0)
        assert mult == 1.3  # calmest → hi
        assert "VVIX" in desc

    def test_high_pctile_sizes_down(self):
        assert vvix_size_mult(1.0)[0] == 0.7  # most elevated → lo

    def test_midpoint_is_neutral(self):
        # symmetric band 0.7-1.3 → pctile 0.5 = 1.0
        assert vvix_size_mult(0.5)[0] == pytest.approx(1.0)

    def test_monotonic_nonincreasing(self):
        vals = [vvix_size_mult(p)[0] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert vals == sorted(vals, reverse=True)

    def test_none_is_no_tilt(self):
        mult, desc = vvix_size_mult(None)
        assert mult == 1.0
        assert "unavailable" in desc

    def test_clamped_to_band(self):
        # out-of-range percentiles are clamped, never explode sizing
        assert vvix_size_mult(-0.5)[0] == 1.3
        assert vvix_size_mult(1.5)[0] == 0.7

    def test_never_zero_or_negative(self):
        # a sizing tilt must never zero out a trade (that would be a filter)
        for p in (None, 0.0, 0.5, 1.0, 2.0):
            assert vvix_size_mult(p)[0] > 0

    def test_custom_band_respected(self):
        assert vvix_size_mult(0.0, lo=0.5, hi=1.6)[0] == 1.6
        assert vvix_size_mult(1.0, lo=0.5, hi=1.6)[0] == 0.5


def _fake_hist(closes):
    return pd.DataFrame({"Close": closes})


class TestFetchVvixPercentile:
    def setup_method(self):
        vix_regime._vvix_cache = None  # clear cache between tests

    def teardown_method(self):
        vix_regime._vvix_cache = None

    def test_percentile_ranks_today_vs_trailing_window(self):
        # trailing window 0..99, today = 50 → ~50th percentile (excludes today from window)
        closes = list(range(100)) + [50.0]
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.return_value = _fake_hist(closes)
            res = vix_regime.fetch_vvix_percentile(lookback_days=100)
        assert res is not None
        level, pctile = res
        assert level == 50.0
        assert pctile == pytest.approx(0.5, abs=0.05)

    def test_calmest_reads_low_percentile(self):
        closes = list(range(50, 150)) + [40.0]  # today below the whole window
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.return_value = _fake_hist(closes)
            _, pctile = vix_regime.fetch_vvix_percentile(lookback_days=100)
        assert pctile == 0.0

    def test_elevated_reads_high_percentile(self):
        closes = list(range(50, 150)) + [200.0]  # today above the whole window
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.return_value = _fake_hist(closes)
            _, pctile = vix_regime.fetch_vvix_percentile(lookback_days=100)
        assert pctile == 1.0

    def test_too_few_bars_returns_none(self):
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.return_value = _fake_hist([90.0, 91.0])
            assert vix_regime.fetch_vvix_percentile() is None

    def test_fetch_exception_returns_none(self):
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.side_effect = RuntimeError("network")
            assert vix_regime.fetch_vvix_percentile() is None

    def test_result_is_cached(self):
        closes = list(range(100)) + [50.0]
        with patch.object(vix_regime, "yf") as ymock:
            ymock.Ticker.return_value.history.return_value = _fake_hist(closes)
            vix_regime.fetch_vvix_percentile(lookback_days=100)
            vix_regime.fetch_vvix_percentile(lookback_days=100)
            # second call served from cache → history fetched only once
            assert ymock.Ticker.return_value.history.call_count == 1


class TestFlagOffByDefault:
    def test_disabled_by_default(self):
        s = Settings()
        assert s.ENABLE_VVIX_SIZING_TILT is False

    def test_default_band_is_mild(self):
        s = Settings()
        assert s.VVIX_TILT_MIN == 0.7
        assert s.VVIX_TILT_MAX == 1.3
