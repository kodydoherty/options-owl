"""Market hours check, holiday calendar, pre-market/post-market windows."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# NYSE holidays for 2026 (add more years as needed)
NYSE_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}


def is_market_open(now: datetime | None = None) -> bool:
    """Check if NYSE is currently open."""
    if now is None:
        now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        return False
    date_str = now.strftime("%Y-%m-%d")
    if date_str in NYSE_HOLIDAYS_2026:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close
