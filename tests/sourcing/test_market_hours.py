"""Tests for market hours utility."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from options_owl.sourcing.utils import market_hours
from options_owl.sourcing.utils.market_hours import (
    NYSE_HOLIDAYS_2026,
    NYSE_HOLIDAYS_2027,
    is_market_open,
    is_trading_day,
)

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


# ---------------------------------------------------------------------------
# Holiday awareness — regression for the 2026-06-19 Juneteenth incident, where
# Juneteenth was missing from the calendar AND three _is_market_open() helpers
# were weekday-only, so the bots treated a closed market as open and the LIVE
# startup self-test crash-looped (churning a Webull token over 19 restarts).
# ---------------------------------------------------------------------------


def test_juneteenth_2026_is_closed():
    # 2026-06-19 is a Friday — the exact day of the incident. MUST be closed.
    now = datetime(2026, 6, 19, 11, 0, 0, tzinfo=ET)
    assert is_trading_day(now) is False
    assert is_market_open(now) is False


def test_juneteenth_2027_observed_is_closed():
    # 2027-06-19 is a Saturday → observed Friday 2027-06-18.
    assert is_market_open(datetime(2027, 6, 18, 11, 0, 0, tzinfo=ET)) is False


@pytest.mark.parametrize("date_str", sorted(NYSE_HOLIDAYS_2026 | NYSE_HOLIDAYS_2027))
def test_all_holidays_are_closed_at_midday(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    now = datetime(y, m, d, 11, 0, 0, tzinfo=ET)
    assert is_trading_day(now) is False, f"{date_str} should be a holiday"
    assert is_market_open(now) is False, f"{date_str} 11:00 ET should be closed"


def test_is_trading_day_normal_weekday():
    # Tuesday, not a holiday
    assert is_trading_day(datetime(2026, 5, 19, 8, 0, 0, tzinfo=ET)) is True


def test_is_trading_day_ignores_time_of_day():
    # is_trading_day is date-only: a normal weekday is a trading day even at 3 AM.
    assert is_trading_day(datetime(2026, 5, 19, 3, 0, 0, tzinfo=ET)) is True
    # ...but a holiday is never a trading day, any hour.
    assert is_trading_day(datetime(2026, 6, 19, 3, 0, 0, tzinfo=ET)) is False


def test_uncovered_year_fails_open_with_warning(caplog):
    # A year with no calendar entries must NOT silently treat holidays as open
    # without at least warning. Picks a far-future year guaranteed uncovered.
    market_hours._warned_years.discard(2099)
    now = datetime(2099, 6, 19, 11, 0, 0, tzinfo=ET)
    # Fails open (treated as a trading day) but is expected to warn.
    assert is_trading_day(now) is True
    assert 2099 in market_hours._warned_years


def test_calendar_has_juneteenth_every_covered_year():
    # Guard against the original bug recurring: every covered year must include
    # a Juneteenth closure (Jun 19, or its observed weekday).
    assert "2026-06-19" in NYSE_HOLIDAYS_2026
    assert "2027-06-18" in NYSE_HOLIDAYS_2027
