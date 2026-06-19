"""Market hours check, holiday calendar, pre-market/post-market windows.

SINGLE SOURCE OF TRUTH for "is the market open / is today a trading day".
Every `_is_market_open()` helper in the codebase (bot_runner, scanner, ml_pipeline)
and the startup Polygon entitlement self-test MUST route their weekend/holiday
decision through `is_trading_day()` here — do NOT reimplement a weekday-only check,
which silently treats market holidays (e.g. Juneteenth) as open days.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

ET = ZoneInfo("America/New_York")

# NYSE full-day closures. Keep this current — when the year rolls over, an
# uncovered year is treated as ALL trading days (see the warning in
# is_trading_day), which would run the bots on holidays. Verified against
# nyse.com/markets/hours-calendars.
NYSE_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth National Independence Day
    "2026-07-03",  # Independence Day (observed — Jul 4 is Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}

NYSE_HOLIDAYS_2027 = {
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # MLK Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observed — Jun 19 is Saturday)
    "2027-07-05",  # Independence Day (observed — Jul 4 is Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed — Dec 25 is Saturday)
}

NYSE_HOLIDAYS = NYSE_HOLIDAYS_2026 | NYSE_HOLIDAYS_2027

# Years for which NYSE_HOLIDAYS is authoritative. A date in an uncovered year
# is NOT treated as a holiday (fails open) but logs a warning so the gap is
# visible instead of silently trading on a holiday.
_COVERED_YEARS = {2026, 2027}
_warned_years: set[int] = set()


def is_trading_day(now: datetime | None = None) -> bool:
    """True if `now` (ET) is a weekday that is not an NYSE full-day holiday.

    This is the weekend/holiday decision ONLY — it does not consider the
    time of day. Callers apply their own session window on top.
    """
    if now is None:
        now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        return False
    if now.year not in _COVERED_YEARS and now.year not in _warned_years:
        _warned_years.add(now.year)
        logger.warning(
            f"NYSE holiday calendar has no entries for {now.year} — holidays will be "
            f"misclassified as trading days. Update NYSE_HOLIDAYS in market_hours.py."
        )
    return now.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS


def is_market_open(now: datetime | None = None) -> bool:
    """Check if NYSE regular session is currently open (9:30 AM - 4:00 PM ET)."""
    if now is None:
        now = datetime.now(tz=ET)
    if not is_trading_day(now):
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close
