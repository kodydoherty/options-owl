"""Tests for theta decay exit logic."""

from __future__ import annotations

from datetime import timedelta

from options_owl.config.settings import Settings
from options_owl.risk.theta_manager import _now_et, calc_time_to_expiry_days, should_theta_exit


def _make_settings(**overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "ENABLE_THETA_DECAY_EXIT": True,
        "THETA_EXIT_DTE_THRESHOLD": 1,
        "THETA_EXIT_LOSS_PCT": 50.0,
        "THETA_EXIT_TIME_MINUTES": 60,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_trade(
    *,
    premium: float = 1.70,
    expiry_date: str | None = None,
    opened_minutes_ago: float = 30.0,
) -> dict:
    now = _now_et().replace(tzinfo=None)
    opened = now - timedelta(minutes=opened_minutes_ago)
    return {
        "premium_per_contract": premium,
        "expiry_date": expiry_date or now.strftime("%Y-%m-%d"),
        "opened_at": opened.isoformat(),
    }


# ---------------------------------------------------------------------------
# calc_time_to_expiry_days
# ---------------------------------------------------------------------------


class TestCalcTimeToExpiryDays:
    def test_today_returns_zero(self):
        today = _now_et().replace(tzinfo=None).strftime("%Y-%m-%d")
        assert calc_time_to_expiry_days(today) == 0.0

    def test_tomorrow_returns_one(self):
        tomorrow = (_now_et().replace(tzinfo=None) + timedelta(days=1)).strftime("%Y-%m-%d")
        assert calc_time_to_expiry_days(tomorrow) == 1.0

    def test_past_date_returns_zero(self):
        past = (_now_et().replace(tzinfo=None) - timedelta(days=5)).strftime("%Y-%m-%d")
        assert calc_time_to_expiry_days(past) == 0.0

    def test_invalid_string_returns_zero(self):
        assert calc_time_to_expiry_days("not-a-date") == 0.0

    def test_empty_string_returns_zero(self):
        assert calc_time_to_expiry_days("") == 0.0


# ---------------------------------------------------------------------------
# Disabled theta exit
# ---------------------------------------------------------------------------


class TestThetaExitDisabled:
    def test_disabled_never_exits(self):
        settings = _make_settings(ENABLE_THETA_DECAY_EXIT=False)
        trade = _make_trade(premium=2.0, opened_minutes_ago=120.0)
        # Even with huge loss
        should_exit, reason = should_theta_exit(trade, 0.01, settings)
        assert should_exit is False
        assert reason == ""


# ---------------------------------------------------------------------------
# Rule 1: 0DTE held too long with loss
# ---------------------------------------------------------------------------


class TestRule1ZeroDTETimeLimitLoss:
    def test_0dte_held_past_limit_while_losing(self):
        settings = _make_settings(THETA_EXIT_TIME_MINUTES=60)
        # 0DTE trade (expiry = today), held 90 minutes, losing
        trade = _make_trade(premium=2.00, opened_minutes_ago=90.0)
        # Current premium < entry -> losing
        should_exit, reason = should_theta_exit(trade, 1.50, settings)
        assert should_exit is True
        assert "0DTE theta exit" in reason

    def test_0dte_held_past_limit_while_winning_no_exit(self):
        settings = _make_settings(THETA_EXIT_TIME_MINUTES=60)
        trade = _make_trade(premium=2.00, opened_minutes_ago=90.0)
        # Current premium > entry -> winning
        should_exit, reason = should_theta_exit(trade, 2.50, settings)
        assert should_exit is False

    def test_0dte_held_under_limit_no_exit(self):
        settings = _make_settings(THETA_EXIT_TIME_MINUTES=60)
        trade = _make_trade(premium=2.00, opened_minutes_ago=30.0)
        # Losing but under time limit
        should_exit, reason = should_theta_exit(trade, 1.50, settings)
        # Rule 1 won't trigger (under 60 min)
        # Rule 3 checks loss > 50%: 25% loss < 50% -> no exit
        assert should_exit is False


# ---------------------------------------------------------------------------
# Rule 2: Near-expiry with large loss
# ---------------------------------------------------------------------------


class TestRule2NearExpiryLargeLoss:
    def test_1dte_with_loss_exceeding_threshold(self):
        settings = _make_settings(
            THETA_EXIT_DTE_THRESHOLD=1,
            THETA_EXIT_LOSS_PCT=50.0,
        )
        tomorrow = (_now_et().replace(tzinfo=None) + timedelta(days=1)).strftime("%Y-%m-%d")
        trade = _make_trade(premium=2.00, expiry_date=tomorrow, opened_minutes_ago=30.0)
        # 60% loss: (0.80 - 2.00) / 2.00 = -60%
        should_exit, reason = should_theta_exit(trade, 0.80, settings)
        assert should_exit is True
        assert "Theta decay exit" in reason

    def test_1dte_with_small_loss_no_exit(self):
        settings = _make_settings(
            THETA_EXIT_DTE_THRESHOLD=1,
            THETA_EXIT_LOSS_PCT=50.0,
        )
        tomorrow = (_now_et().replace(tzinfo=None) + timedelta(days=1)).strftime("%Y-%m-%d")
        trade = _make_trade(premium=2.00, expiry_date=tomorrow, opened_minutes_ago=30.0)
        # 10% loss -> under 50% threshold
        should_exit, reason = should_theta_exit(trade, 1.80, settings)
        assert should_exit is False

    def test_2dte_does_not_trigger_with_threshold_1(self):
        settings = _make_settings(THETA_EXIT_DTE_THRESHOLD=1, THETA_EXIT_LOSS_PCT=50.0)
        two_days = (_now_et().replace(tzinfo=None) + timedelta(days=2)).strftime("%Y-%m-%d")
        trade = _make_trade(premium=2.00, expiry_date=two_days, opened_minutes_ago=30.0)
        # Big loss but DTE > threshold
        should_exit, reason = should_theta_exit(trade, 0.50, settings)
        assert should_exit is False


# ---------------------------------------------------------------------------
# Rule 3: 0DTE loss exceeding limit regardless of time
# ---------------------------------------------------------------------------


class TestRule3ZeroDTELossLimit:
    def test_0dte_big_loss_triggers_immediately(self):
        settings = _make_settings(THETA_EXIT_LOSS_PCT=50.0)
        # 0DTE, only held 10 minutes
        trade = _make_trade(premium=2.00, opened_minutes_ago=10.0)
        # 60% loss
        should_exit, reason = should_theta_exit(trade, 0.80, settings)
        assert should_exit is True
        assert "0DTE loss exit" in reason

    def test_0dte_under_loss_limit_no_exit(self):
        settings = _make_settings(THETA_EXIT_LOSS_PCT=50.0)
        trade = _make_trade(premium=2.00, opened_minutes_ago=10.0)
        # 20% loss -> under 50% threshold
        should_exit, reason = should_theta_exit(trade, 1.60, settings)
        assert should_exit is False


# ---------------------------------------------------------------------------
# Within thresholds = no exit
# ---------------------------------------------------------------------------


class TestNoExit:
    def test_profitable_0dte_no_exit(self):
        settings = _make_settings()
        trade = _make_trade(premium=2.00, opened_minutes_ago=120.0)
        # Winning: current > entry
        should_exit, reason = should_theta_exit(trade, 3.00, settings)
        assert should_exit is False

    def test_long_dated_option_no_exit(self):
        settings = _make_settings(THETA_EXIT_DTE_THRESHOLD=1)
        far_future = (_now_et().replace(tzinfo=None) + timedelta(days=30)).strftime("%Y-%m-%d")
        trade = _make_trade(premium=2.00, expiry_date=far_future, opened_minutes_ago=120.0)
        # Even with loss, DTE >> threshold
        should_exit, reason = should_theta_exit(trade, 0.50, settings)
        assert should_exit is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestThetaEdgeCases:
    def test_zero_entry_premium_no_exit(self):
        settings = _make_settings()
        trade = _make_trade(premium=0.0, opened_minutes_ago=120.0)
        should_exit, reason = should_theta_exit(trade, 1.00, settings)
        assert should_exit is False

    def test_missing_opened_at_no_exit(self):
        settings = _make_settings()
        trade = {
            "premium_per_contract": 2.00,
            "expiry_date": _now_et().replace(tzinfo=None).strftime("%Y-%m-%d"),
            "opened_at": "",
        }
        should_exit, reason = should_theta_exit(trade, 1.00, settings)
        assert should_exit is False

    def test_missing_expiry_date_treated_as_0dte(self):
        """No expiry_date -> dte = 0, so 0DTE rules apply."""
        settings = _make_settings(THETA_EXIT_LOSS_PCT=50.0)
        trade = {
            "premium_per_contract": 2.00,
            "expiry_date": "",
            "opened_at": (_now_et().replace(tzinfo=None) - timedelta(minutes=120)).isoformat(),
        }
        # Big loss + 0DTE + held > limit
        should_exit, reason = should_theta_exit(trade, 0.80, settings)
        assert should_exit is True
