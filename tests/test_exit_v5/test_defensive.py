"""Tests for defensive layer — bid disappearance detection."""

from options_owl.risk.exit_v5.config import DefensiveConfig
from options_owl.risk.exit_v5.defensive import check_bid_disappearance


class TestCheckBidDisappearance:

    def test_bid_positive_no_exit(self):
        result = check_bid_disappearance(bid=1.50, seconds_at_zero_bid=0.0)
        assert result["should_exit"] is False
        assert result["seconds_at_zero"] == 0.0

    def test_bid_zero_under_timeout(self):
        result = check_bid_disappearance(bid=0.0, seconds_at_zero_bid=15.0)
        assert result["should_exit"] is False

    def test_bid_zero_at_timeout(self):
        result = check_bid_disappearance(bid=0.0, seconds_at_zero_bid=30.0)
        assert result["should_exit"] is True
        assert "bid_zero" in result["reason"]
        assert result["seconds_at_zero"] == 30.0

    def test_bid_zero_over_timeout(self):
        result = check_bid_disappearance(bid=0.0, seconds_at_zero_bid=45.0)
        assert result["should_exit"] is True

    def test_custom_timeout(self):
        cfg = DefensiveConfig(bid_zero_timeout_sec=10.0)
        result = check_bid_disappearance(bid=0.0, seconds_at_zero_bid=10.0, cfg=cfg)
        assert result["should_exit"] is True

    def test_negative_bid_treated_as_zero(self):
        result = check_bid_disappearance(bid=-0.01, seconds_at_zero_bid=30.0)
        assert result["should_exit"] is True
