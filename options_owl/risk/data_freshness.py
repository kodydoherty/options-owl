"""Runtime market-data freshness guard.

Blocks NEW entries (never the sell path) when the harvester's live feed goes stale during market
hours, and surfaces it loudly. Catches the failure a reconnect loop CANNOT see: frozen prices with
fresh write-timestamps (the 2026-06-19 Juneteenth case — the harvester kept rewriting the last-known
SPY price every minute, so a naive "is a row present + recent" check looked healthy while the data was
yesterday's close). We therefore check the SPY 1m candle's bar_time age AND that there is actual volume.

Fails OPEN (returns fresh) on any error or outside market hours, so the guard itself can never become a
silent death-sentence — a CONFIRMED stale feed during RTH is the only thing that blocks.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

from loguru import logger

from options_owl.db import postgres
from options_owl.sourcing.utils.market_hours import is_market_open

ET = ZoneInfo("America/New_York")
_UTC = datetime.timezone.utc


async def check_market_data_fresh(
    max_age_sec: int = 180, now: datetime.datetime | None = None,
) -> tuple[bool, str]:
    """Return (is_fresh, reason).

    Fresh = the newest SPY 1m candle in the harvester's Postgres is younger than ``max_age_sec`` AND
    the last few bars carry real volume (so a frozen-but-recently-rewritten price reads as STALE).
    Only enforced during regular market hours (``is_market_open`` is holiday-aware) — outside RTH a
    frozen feed is expected, so this returns fresh to avoid false halts. Fails OPEN on any error.
    """
    now = now or datetime.datetime.now(tz=ET)
    if not is_market_open(now):
        return True, "market closed (guard not enforced)"

    try:
        rows = await postgres.read_stock_candles("SPY", "1m", limit=3)
    except Exception as exc:  # transient PG hiccup must NOT halt trading — fail open
        logger.warning(f"DATA_FRESHNESS: check errored ({exc}) — failing open")
        return True, f"check error, failing open: {exc}"

    if not rows:
        return False, "no SPY 1m candle in harvester PG — feed down"

    newest = rows[-1]
    age = (datetime.datetime.now(tz=_UTC) - newest["bar_time"].astimezone(_UTC)).total_seconds()
    if age > max_age_sec:
        return False, f"SPY 1m newest bar is {age:.0f}s old (>{max_age_sec}s) — harvester feed stalled"

    recent_vol = sum((r.get("volume") or 0) for r in rows)
    if recent_vol <= 0:
        return False, (
            f"SPY 1m bars have 0 volume (newest age {age:.0f}s) — frozen price, "
            f"feed not live (Juneteenth-style stale)"
        )

    return True, f"fresh (newest age {age:.0f}s, recent_vol {recent_vol})"
