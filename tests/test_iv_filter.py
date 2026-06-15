"""Tests for IV filter module."""

from __future__ import annotations

import time
from unittest.mock import patch

from options_owl.config.settings import Settings
from options_owl.signals import iv_filter
from options_owl.signals.iv_filter import (
    _compute_and_cache,
    check_iv_filter,
    fetch_iv_percentile,
    fetch_iv_rank,
)


def _make_settings(**overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "ENABLE_IV_FILTER": True,
        "IV_RANK_MIN": 20.0,
        "IV_RANK_MAX": 80.0,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# check_iv_filter
# ---------------------------------------------------------------------------


class TestCheckIVFilter:
    def test_disabled_filter_always_passes(self):
        settings = _make_settings(ENABLE_IV_FILTER=False)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is True
        assert "disabled" in reason.lower()

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=None)
    def test_no_iv_data_allows_trade(self, mock_rank):
        settings = _make_settings()
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is True
        assert "Could not fetch" in reason

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=50.0)
    def test_iv_rank_within_range_passes(self, mock_rank):
        settings = _make_settings(IV_RANK_MIN=20.0, IV_RANK_MAX=80.0)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is True
        assert "within acceptable range" in reason

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=10.0)
    def test_iv_rank_below_min_fails(self, mock_rank):
        settings = _make_settings(IV_RANK_MIN=20.0, IV_RANK_MAX=80.0)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is False
        assert "below minimum" in reason

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=90.0)
    def test_iv_rank_above_max_fails(self, mock_rank):
        settings = _make_settings(IV_RANK_MIN=20.0, IV_RANK_MAX=80.0)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is False
        assert "above maximum" in reason

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=20.0)
    def test_iv_rank_at_min_boundary_passes(self, mock_rank):
        settings = _make_settings(IV_RANK_MIN=20.0, IV_RANK_MAX=80.0)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is True

    @patch("options_owl.signals.iv_filter.fetch_iv_rank", return_value=80.0)
    def test_iv_rank_at_max_boundary_passes(self, mock_rank):
        settings = _make_settings(IV_RANK_MIN=20.0, IV_RANK_MAX=80.0)
        passes, reason = check_iv_filter("AAPL", settings)
        assert passes is True


# ---------------------------------------------------------------------------
# IV rank calculation
# ---------------------------------------------------------------------------


class TestIVRankCalculation:
    @patch("options_owl.signals.iv_filter._get_historical_realized_vol")
    @patch("options_owl.signals.iv_filter._get_current_iv")
    def test_iv_rank_computation(self, mock_current, mock_hist):
        # Clear cache
        iv_filter._iv_cache.clear()

        mock_current.return_value = 0.30  # 30% IV
        # Historical vols range from 0.10 to 0.50
        mock_hist.return_value = [0.10, 0.20, 0.30, 0.40, 0.50]

        result = _compute_and_cache("TEST")
        assert result is not None
        iv_rank, iv_percentile = result
        # IV Rank = (0.30 - 0.10) / (0.50 - 0.10) * 100 = 50%
        assert abs(iv_rank - 50.0) < 0.1

    @patch("options_owl.signals.iv_filter._get_historical_realized_vol")
    @patch("options_owl.signals.iv_filter._get_current_iv")
    def test_iv_percentile_computation(self, mock_current, mock_hist):
        iv_filter._iv_cache.clear()

        mock_current.return_value = 0.30
        # 2 out of 5 values are below 0.30
        mock_hist.return_value = [0.10, 0.20, 0.30, 0.40, 0.50]

        result = _compute_and_cache("TEST2")
        assert result is not None
        _, iv_percentile = result
        # Days below 0.30: 0.10, 0.20 -> 2/5 = 40%
        assert abs(iv_percentile - 40.0) < 0.1

    @patch("options_owl.signals.iv_filter._get_historical_realized_vol")
    @patch("options_owl.signals.iv_filter._get_current_iv")
    def test_iv_rank_with_equal_high_low(self, mock_current, mock_hist):
        iv_filter._iv_cache.clear()

        mock_current.return_value = 0.25
        mock_hist.return_value = [0.25, 0.25, 0.25]  # All same

        result = _compute_and_cache("FLAT")
        assert result is not None
        iv_rank, _ = result
        assert iv_rank == 50.0  # Default when no range

    @patch("options_owl.signals.iv_filter._get_current_iv", return_value=None)
    def test_returns_none_when_no_current_iv(self, mock_current):
        iv_filter._iv_cache.clear()
        result = _compute_and_cache("NOIV")
        assert result is None

    @patch("options_owl.signals.iv_filter._get_historical_realized_vol", return_value=[])
    @patch("options_owl.signals.iv_filter._get_current_iv", return_value=0.30)
    def test_returns_none_when_no_historical_data(self, mock_current, mock_hist):
        iv_filter._iv_cache.clear()
        result = _compute_and_cache("NOHIST")
        assert result is None


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestIVCache:
    @patch("options_owl.signals.iv_filter._get_historical_realized_vol")
    @patch("options_owl.signals.iv_filter._get_current_iv")
    def test_cache_hit(self, mock_current, mock_hist):
        iv_filter._iv_cache.clear()

        mock_current.return_value = 0.30
        mock_hist.return_value = [0.10, 0.20, 0.30, 0.40, 0.50]

        # First call populates cache
        result1 = _compute_and_cache("CACHED")
        assert result1 is not None
        assert mock_current.call_count == 1

        # Second call should use cache
        result2 = _compute_and_cache("CACHED")
        assert result2 is not None
        assert result1 == result2
        # Should NOT have called the external functions again
        assert mock_current.call_count == 1

    @patch("options_owl.signals.iv_filter._get_historical_realized_vol")
    @patch("options_owl.signals.iv_filter._get_current_iv")
    def test_cache_expires(self, mock_current, mock_hist):
        iv_filter._iv_cache.clear()

        mock_current.return_value = 0.30
        mock_hist.return_value = [0.10, 0.20, 0.30, 0.40, 0.50]

        # Populate cache with old timestamp
        iv_filter._iv_cache["STALE"] = (time.time() - 1000, 50.0, 40.0)

        # Should re-compute because cache is stale
        result = _compute_and_cache("STALE")
        assert result is not None
        assert mock_current.call_count == 1


# ---------------------------------------------------------------------------
# fetch_iv_rank / fetch_iv_percentile
# ---------------------------------------------------------------------------


class TestFetchFunctions:
    @patch("options_owl.signals.iv_filter._compute_and_cache", return_value=(45.0, 60.0))
    def test_fetch_iv_rank(self, mock_compute):
        rank = fetch_iv_rank("AAPL")
        assert rank == 45.0

    @patch("options_owl.signals.iv_filter._compute_and_cache", return_value=(45.0, 60.0))
    def test_fetch_iv_percentile(self, mock_compute):
        pct = fetch_iv_percentile("AAPL")
        assert pct == 60.0

    @patch("options_owl.signals.iv_filter._compute_and_cache", return_value=None)
    def test_fetch_iv_rank_returns_none(self, mock_compute):
        assert fetch_iv_rank("FAIL") is None

    @patch("options_owl.signals.iv_filter._compute_and_cache", return_value=None)
    def test_fetch_iv_percentile_returns_none(self, mock_compute):
        assert fetch_iv_percentile("FAIL") is None
