"""Tests for VIX regime detection."""

from __future__ import annotations

from unittest.mock import patch

from options_owl.config.settings import Settings
from options_owl.risk import vix_regime
from options_owl.risk.vix_regime import VixRegime, check_vix_regime, fetch_vix_level


def _make_settings(**overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "ENABLE_VIX_FILTER": True,
        "VIX_MAX": 35.0,
        "VIX_HIGH_THRESHOLD": 25.0,
        "VIX_POSITION_REDUCTION_PCT": 50.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# VIX regime detection
# ---------------------------------------------------------------------------


class TestVixRegimeDetection:
    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=12.0)
    def test_low_vix_regime(self, mock_fetch):
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.regime == "low"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 1.0
        assert regime.level == 12.0

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=18.0)
    def test_normal_vix_regime(self, mock_fetch):
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.regime == "normal"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 1.0

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=30.0)
    def test_high_vix_regime(self, mock_fetch):
        settings = _make_settings(VIX_HIGH_THRESHOLD=25.0, VIX_POSITION_REDUCTION_PCT=50.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "high"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 0.5  # 1.0 - 50/100

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=40.0)
    def test_extreme_vix_regime(self, mock_fetch):
        settings = _make_settings(VIX_MAX=35.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "extreme"
        assert regime.can_trade is False
        assert regime.position_size_multiplier == 0.0


# ---------------------------------------------------------------------------
# Position size multiplier
# ---------------------------------------------------------------------------


class TestPositionSizeMultiplier:
    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=28.0)
    def test_high_vix_reduces_position(self, mock_fetch):
        settings = _make_settings(
            VIX_HIGH_THRESHOLD=25.0,
            VIX_POSITION_REDUCTION_PCT=30.0,
        )
        regime = check_vix_regime(settings)
        assert abs(regime.position_size_multiplier - 0.7) < 0.01

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=15.0)
    def test_normal_vix_full_position(self, mock_fetch):
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.position_size_multiplier == 1.0


# ---------------------------------------------------------------------------
# Trading pause
# ---------------------------------------------------------------------------


class TestTradingPause:
    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=50.0)
    def test_extreme_vix_pauses_trading(self, mock_fetch):
        settings = _make_settings(VIX_MAX=35.0)
        regime = check_vix_regime(settings)
        assert regime.can_trade is False
        assert "paused" in regime.reason.lower()

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=34.9)
    def test_just_below_max_allows_trading(self, mock_fetch):
        settings = _make_settings(VIX_MAX=35.0)
        regime = check_vix_regime(settings)
        assert regime.can_trade is True

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=35.1)
    def test_just_above_max_pauses_trading(self, mock_fetch):
        settings = _make_settings(VIX_MAX=35.0)
        regime = check_vix_regime(settings)
        assert regime.can_trade is False


# ---------------------------------------------------------------------------
# Disabled VIX filter
# ---------------------------------------------------------------------------


class TestVixFilterDisabled:
    def test_disabled_returns_normal_regime(self):
        settings = _make_settings(ENABLE_VIX_FILTER=False)
        regime = check_vix_regime(settings)
        assert regime.regime == "normal"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 1.0
        assert "disabled" in regime.reason.lower()


# ---------------------------------------------------------------------------
# VIX fetch failure
# ---------------------------------------------------------------------------


class TestVixFetchFailure:
    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=None)
    def test_fetch_failure_defaults_to_normal(self, mock_fetch):
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.regime == "normal"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 1.0
        assert "Could not fetch" in regime.reason


# ---------------------------------------------------------------------------
# VIX boundary values
# ---------------------------------------------------------------------------


class TestVixBoundaries:
    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=15.0)
    def test_at_low_boundary(self, mock_fetch):
        """VIX at exactly 15.0 is normal (not low, since < 15 is low)."""
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.regime == "normal"

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=14.9)
    def test_just_below_low_boundary(self, mock_fetch):
        settings = _make_settings()
        regime = check_vix_regime(settings)
        assert regime.regime == "low"

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=25.0)
    def test_at_high_threshold(self, mock_fetch):
        """VIX at exactly 25.0 is normal (> 25 triggers high)."""
        settings = _make_settings(VIX_HIGH_THRESHOLD=25.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "normal"

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=25.1)
    def test_just_above_high_threshold(self, mock_fetch):
        settings = _make_settings(VIX_HIGH_THRESHOLD=25.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "high"

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=35.0)
    def test_at_vix_max(self, mock_fetch):
        """VIX at exactly 35.0 is high (> 35 triggers extreme)."""
        settings = _make_settings(VIX_MAX=35.0, VIX_HIGH_THRESHOLD=25.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "high"

    @patch("options_owl.risk.vix_regime.fetch_vix_level", return_value=35.1)
    def test_just_above_vix_max(self, mock_fetch):
        settings = _make_settings(VIX_MAX=35.0)
        regime = check_vix_regime(settings)
        assert regime.regime == "extreme"


# ---------------------------------------------------------------------------
# VixRegime model
# ---------------------------------------------------------------------------


class TestVixRegimeModel:
    def test_vix_regime_model_fields(self):
        regime = VixRegime(
            level=20.0,
            regime="normal",
            can_trade=True,
            position_size_multiplier=1.0,
            reason="test",
        )
        assert regime.level == 20.0
        assert regime.regime == "normal"
        assert regime.can_trade is True
        assert regime.position_size_multiplier == 1.0
        assert regime.reason == "test"


# ---------------------------------------------------------------------------
# fetch_vix_level cache
# ---------------------------------------------------------------------------


class TestFetchVixLevelCache:
    def test_cache_returns_stored_value(self):
        import time as time_mod

        vix_regime._vix_cache = (time_mod.time(), 22.5)
        result = fetch_vix_level()
        assert result == 22.5

    def test_stale_cache_refetches(self):
        import time as time_mod

        # Set cache to be very old
        vix_regime._vix_cache = (time_mod.time() - 600, 22.5)
        with patch("options_owl.risk.vix_regime.yf") as mock_yf:
            mock_ticker = mock_yf.Ticker.return_value
            mock_hist = mock_ticker.history.return_value
            mock_hist.empty = True
            result = fetch_vix_level()
            assert result is None  # Failed to fetch
            assert mock_yf.Ticker.called

    def teardown_method(self):
        # Reset cache after each test
        vix_regime._vix_cache = None
