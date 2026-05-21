"""Tests for position monitor exit condition logic and premium estimation."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from options_owl.execution.position_monitor import (
    _check_exit_condition,
    _estimate_exit_premium,
    _resolve_expiry_for_lookup,
)


# ---------------------------------------------------------------------------
# _estimate_exit_premium
# ---------------------------------------------------------------------------


class TestEstimateExitPremium:
    def test_call_price_increase_raises_premium(self):
        # Call: underlying rises $2, delta 0.5 -> premium up $1
        result = _estimate_exit_premium(
            entry_premium=2.00,
            entry_price=100.0,
            current_price=102.0,
            option_type="call",
            delta=0.50,
        )
        assert abs(result - 3.00) < 0.01

    def test_call_price_decrease_lowers_premium(self):
        result = _estimate_exit_premium(
            entry_premium=2.00,
            entry_price=100.0,
            current_price=98.0,
            option_type="call",
            delta=0.50,
        )
        assert abs(result - 1.00) < 0.01

    def test_put_price_decrease_raises_premium(self):
        # Put: underlying drops $2, delta 0.5 -> premium up $1
        result = _estimate_exit_premium(
            entry_premium=2.00,
            entry_price=100.0,
            current_price=98.0,
            option_type="put",
            delta=0.50,
        )
        assert abs(result - 3.00) < 0.01

    def test_put_price_increase_lowers_premium(self):
        result = _estimate_exit_premium(
            entry_premium=2.00,
            entry_price=100.0,
            current_price=102.0,
            option_type="put",
            delta=0.50,
        )
        assert abs(result - 1.00) < 0.01

    def test_premium_floored_at_001(self):
        # Massive adverse move should floor at $0.01
        result = _estimate_exit_premium(
            entry_premium=1.00,
            entry_price=100.0,
            current_price=90.0,
            option_type="call",
            delta=0.50,
        )
        assert result == 0.01

    def test_no_price_change_returns_entry_premium(self):
        result = _estimate_exit_premium(
            entry_premium=2.50,
            entry_price=100.0,
            current_price=100.0,
            option_type="call",
        )
        assert abs(result - 2.50) < 0.01

    def test_different_delta_values(self):
        # High delta: more sensitive
        result = _estimate_exit_premium(
            entry_premium=2.00,
            entry_price=100.0,
            current_price=101.0,
            option_type="call",
            delta=0.80,
        )
        assert abs(result - 2.80) < 0.01


# ---------------------------------------------------------------------------
# _check_exit_condition — CALL trades
# ---------------------------------------------------------------------------


class TestCheckExitConditionCalls:
    def _call_trade(self, **overrides):
        defaults = {
            "option_type": "call",
            "target_1": 105.0,
            "target_2": 108.0,
            "stop_price": 95.0,
            "exit_by": None,
        }
        defaults.update(overrides)
        return defaults

    def test_call_below_stop_no_exit(self):
        # Underlying stops removed — price below stop should NOT trigger exit
        # (premium-based stops live in the exit pipeline, not here)
        trade = self._call_trade()
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, desc = _check_exit_condition(trade, current_price=94.0)
            assert reason is None

    def test_call_t2_hit(self):
        trade = self._call_trade()
        reason, desc = _check_exit_condition(trade, current_price=110.0)
        assert reason == "t2_hit"

    def test_call_t2_exact(self):
        trade = self._call_trade()
        reason, desc = _check_exit_condition(trade, current_price=108.0)
        assert reason == "t2_hit"

    def test_call_t1_hit(self):
        trade = self._call_trade()
        reason, desc = _check_exit_condition(trade, current_price=106.0)
        assert reason == "t1_hit"

    def test_call_t1_exact(self):
        trade = self._call_trade()
        reason, desc = _check_exit_condition(trade, current_price=105.0)
        assert reason == "t1_hit"

    def test_call_between_entry_and_t1_no_exit(self):
        trade = self._call_trade()
        # Price at 102 — above stop, below T1
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            # Mid-day, not near EOD
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=102.0)
            assert reason is None


# ---------------------------------------------------------------------------
# _check_exit_condition — PUT trades
# ---------------------------------------------------------------------------


class TestCheckExitConditionPuts:
    def _put_trade(self, **overrides):
        defaults = {
            "option_type": "put",
            "target_1": 95.0,
            "target_2": 92.0,
            "stop_price": 105.0,
            "exit_by": None,
        }
        defaults.update(overrides)
        return defaults

    def test_put_above_stop_no_exit(self):
        # Underlying stops removed — price above stop should NOT trigger exit
        trade = self._put_trade()
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, desc = _check_exit_condition(trade, current_price=106.0)
            assert reason is None

    def test_put_t2_hit(self):
        trade = self._put_trade()
        reason, desc = _check_exit_condition(trade, current_price=90.0)
        assert reason == "t2_hit"

    def test_put_t2_exact(self):
        trade = self._put_trade()
        reason, desc = _check_exit_condition(trade, current_price=92.0)
        assert reason == "t2_hit"

    def test_put_t1_hit(self):
        trade = self._put_trade()
        reason, desc = _check_exit_condition(trade, current_price=94.0)
        assert reason == "t1_hit"

    def test_put_between_entry_and_t1_no_exit(self):
        trade = self._put_trade()
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=98.0)
            assert reason is None


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------


class TestTimeBasedExit:
    def test_exit_by_time_reached(self):
        trade = {
            "option_type": "call",
            "target_1": 200.0,
            "target_2": 210.0,
            "stop_price": 50.0,
            "exit_by": "10:30",
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 10, 45, 0)
            reason, desc = _check_exit_condition(trade, current_price=100.0)
            assert reason == "time_expiry"
            assert "10:30" in desc

    def test_exit_by_time_not_reached(self):
        trade = {
            "option_type": "call",
            "target_1": 200.0,
            "target_2": 210.0,
            "stop_price": 50.0,
            "exit_by": "14:00",
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=100.0)
            assert reason is None


# ---------------------------------------------------------------------------
# EOD auto-close for 0DTE (15:45 ET cutoff)
# ---------------------------------------------------------------------------


class TestEODAutoClose:
    def test_eod_close_after_1545(self):
        trade = {
            "option_type": "call",
            "target_1": 200.0,
            "target_2": 210.0,
            "stop_price": 50.0,
            "exit_by": None,
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 15, 50, 0)
            reason, desc = _check_exit_condition(trade, current_price=100.0)
            assert reason == "eod_expiry"
            assert "End-of-day" in desc

    def test_no_eod_close_before_1545(self):
        trade = {
            "option_type": "call",
            "target_1": 200.0,
            "target_2": 210.0,
            "stop_price": 50.0,
            "exit_by": None,
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 14, 30, 0)
            reason, _ = _check_exit_condition(trade, current_price=100.0)
            assert reason is None

    def test_eod_close_exactly_at_1545(self):
        trade = {
            "option_type": "call",
            "target_1": 200.0,
            "target_2": 210.0,
            "stop_price": 50.0,
            "exit_by": None,
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 15, 45, 0)
            reason, desc = _check_exit_condition(trade, current_price=100.0)
            assert reason == "eod_expiry"


# ---------------------------------------------------------------------------
# Exit priority: stop > T2 > T1
# ---------------------------------------------------------------------------


class TestExitPriority:
    def test_no_underlying_stop_in_legacy_function(self):
        """Underlying stops removed from legacy function — premium stops live in exit pipeline."""
        trade = {
            "option_type": "call",
            "target_1": 105.0,
            "target_2": 108.0,
            "stop_price": 95.0,
            "exit_by": None,
        }
        # Price below stop but no targets hit — should return EOD or None
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=90.0)
            assert reason is None  # no underlying stop anymore

    def test_t2_takes_priority_over_t1(self):
        """Price far above T2 -> T2 hit first."""
        trade = {
            "option_type": "call",
            "target_1": 105.0,
            "target_2": 108.0,
            "stop_price": 95.0,
            "exit_by": None,
        }
        reason, _ = _check_exit_condition(trade, current_price=110.0)
        assert reason == "t2_hit"


# ---------------------------------------------------------------------------
# None targets
# ---------------------------------------------------------------------------


class TestNoneTargets:
    def test_no_stop_price_skips_stop_check(self):
        trade = {
            "option_type": "call",
            "target_1": 105.0,
            "target_2": 108.0,
            "stop_price": None,
            "exit_by": None,
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=80.0)
            assert reason is None

    def test_no_targets_only_stop(self):
        trade = {
            "option_type": "call",
            "target_1": None,
            "target_2": None,
            "stop_price": 95.0,
            "exit_by": None,
        }
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            reason, _ = _check_exit_condition(trade, current_price=100.0)
            assert reason is None


# ---------------------------------------------------------------------------
# _resolve_expiry_for_lookup
# ---------------------------------------------------------------------------


class TestResolveExpiryForLookup:
    def test_uses_expiry_date_column(self):
        trade = {"expiry_date": "2025-04-18", "opened_at": "2025-04-18T10:00:00"}
        assert _resolve_expiry_for_lookup(trade) == "2025-04-18"

    def test_fallback_to_today_for_0dte(self):
        with patch("options_owl.execution.position_monitor._now_et") as mock_now:
            mock_now.return_value = datetime(2025, 6, 15, 12, 0, 0)
            trade = {"expiry_date": None, "opened_at": "2025-06-15T10:00:00"}
            assert _resolve_expiry_for_lookup(trade) == "2025-06-15"

    def test_returns_none_for_old_trade_without_expiry(self):
        trade = {"expiry_date": None, "opened_at": "2020-01-01T10:00:00"}
        assert _resolve_expiry_for_lookup(trade) is None

    def test_empty_expiry_date(self):
        trade = {"expiry_date": "", "opened_at": "2020-01-01T10:00:00"}
        # Empty string is falsy, falls through to legacy check
        assert _resolve_expiry_for_lookup(trade) is None
