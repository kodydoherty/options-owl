"""Tests for market hours utility."""

from datetime import datetime
from zoneinfo import ZoneInfo

from options_owl.sourcing.utils.market_hours import is_market_open

ET = ZoneInfo("America/New_York")


def test_market_open_during_hours():
    # Tuesday 10:30 AM ET
    now = datetime(2026, 5, 19, 10, 30, 0, tzinfo=ET)
    assert is_market_open(now) is True


def test_market_closed_before_open():
    # Tuesday 9:00 AM ET (before 9:30)
    now = datetime(2026, 5, 19, 9, 0, 0, tzinfo=ET)
    assert is_market_open(now) is False


def test_market_closed_after_close():
    # Tuesday 4:30 PM ET (after 4:00)
    now = datetime(2026, 5, 19, 16, 30, 0, tzinfo=ET)
    assert is_market_open(now) is False


def test_market_closed_weekend():
    # Saturday
    now = datetime(2026, 5, 23, 11, 0, 0, tzinfo=ET)
    assert is_market_open(now) is False


def test_market_open_at_boundary():
    # Exactly 9:30 AM ET
    now = datetime(2026, 5, 19, 9, 30, 0, tzinfo=ET)
    assert is_market_open(now) is True


def test_market_open_at_close_boundary():
    # Exactly 4:00 PM ET
    now = datetime(2026, 5, 19, 16, 0, 0, tzinfo=ET)
    assert is_market_open(now) is True
