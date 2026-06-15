"""Tests for V5Config — category-aware defaults from backtest_category_sweep."""

import pytest

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    TickerCategory,
    V5Config,
    categorize_ticker,
)


class TestV5ConfigDefaults:

    def test_grace_period_5min(self):
        cfg = V5Config()
        assert cfg.grace_period_min == 5.0

    def test_eod_cutoff_15min(self):
        cfg = V5Config()
        assert cfg.eod_cutoff_minutes_before_close == 15.0

    def test_bid_disappearance_30s(self):
        cfg = V5Config()
        assert cfg.defensive.bid_zero_timeout_sec == 30.0

    def test_scalp_trail_defaults(self):
        cfg = V5Config()
        assert cfg.scalp_peak_threshold_pct == 20.0
        assert cfg.scalp_fade_ratio == 0.6
        assert cfg.scalp_confirm_threshold == 0.2

    def test_checkpoint_defaults(self):
        cfg = V5Config()
        assert cfg.checkpoint_drop_pct == 15.0

    def test_graduated_stop_0dte(self):
        cfg = V5Config()
        assert cfg.tight_stop_0dte_pct == 15.0
        assert cfg.backstop_0dte_pct == 30.0

    def test_graduated_stop_multiday(self):
        cfg = V5Config()
        assert cfg.tight_stop_multiday_pct == 30.0
        assert cfg.backstop_multiday_pct == 50.0

    def test_underlying_against_threshold(self):
        cfg = V5Config()
        assert cfg.underlying_against_threshold == 0.5

    def test_soft_trail_defaults(self):
        cfg = V5Config()
        assert cfg.soft_trail_band_low_pct == 15.0
        assert cfg.soft_trail_band_high_pct == 50.0
        assert cfg.soft_trail_keep_pct == 0.60

    def test_profit_target_index_0dte(self):
        cfg = V5Config()
        assert cfg.profit_target_index_0dte_pct == 30.0

    def test_adaptive_highvol_tiers(self):
        cfg = V5Config()
        tiers = cfg.adaptive_highvol_tiers
        assert len(tiers) == 3
        assert tiers[0] == AdaptiveTier(400, 35)
        assert tiers[1] == AdaptiveTier(150, 55)
        assert tiers[2] == AdaptiveTier(40, 50)

    def test_adaptive_index_tiers(self):
        cfg = V5Config()
        tiers = cfg.adaptive_index_tiers
        assert len(tiers) == 3
        assert tiers[0] == AdaptiveTier(300, 25)
        assert tiers[1] == AdaptiveTier(100, 40)
        assert tiers[2] == AdaptiveTier(30, 35)

    def test_adaptive_standard_same_as_index(self):
        cfg = V5Config()
        assert cfg.adaptive_standard_tiers == cfg.adaptive_index_tiers

    def test_get_adaptive_tiers_highvol(self):
        cfg = V5Config()
        tiers = cfg.get_adaptive_tiers(TickerCategory.HIGH_VOL)
        assert tiers == cfg.adaptive_highvol_tiers

    def test_get_adaptive_tiers_index(self):
        cfg = V5Config()
        tiers = cfg.get_adaptive_tiers(TickerCategory.INDEX)
        assert tiers == cfg.adaptive_index_tiers

    def test_get_adaptive_tiers_standard(self):
        cfg = V5Config()
        tiers = cfg.get_adaptive_tiers(TickerCategory.STANDARD)
        assert tiers == cfg.adaptive_standard_tiers

    def test_theta_bleed_defaults(self):
        cfg = V5Config()
        assert cfg.theta_bleed_min == 120.0
        assert cfg.theta_bleed_drop_pct == 30.0

    def test_theta_timer_defaults(self):
        cfg = V5Config()
        assert cfg.theta_timer_minutes == 180.0
        assert cfg.theta_timer_loss_pct == 15.0


class TestTickerCategory:

    def test_highvol_tickers(self):
        # COIN and AVGO removed from HIGH_VOL (blocked tickers, 2026-05-30)
        for t in ("MSTR", "AMD", "TSLA", "NVDA", "META", "SMCI", "PLTR"):
            assert categorize_ticker(t) == TickerCategory.HIGH_VOL

    def test_index_tickers(self):
        for t in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"):
            assert categorize_ticker(t) == TickerCategory.INDEX

    def test_standard_tickers(self):
        for t in ("AAPL", "MSFT", "GOOGL", "AMZN"):
            assert categorize_ticker(t) == TickerCategory.STANDARD


class TestV5ConfigFrozen:

    def test_frozen(self):
        cfg = V5Config()
        with pytest.raises(AttributeError):
            cfg.grace_period_min = 60.0  # type: ignore[misc]
