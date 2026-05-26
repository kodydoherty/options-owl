"""Congress member stock trades (STOCK Act disclosures).

Primary: Unusual Whales Congress endpoint (requires UW_KEY).
Backup: Capitol Trades free API.

Note: Congress trades are disclosed 30-45 days after execution,
so this is a lagging indicator. Useful for bias, not timing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from loguru import logger

# Cache: ticker -> (CongressActivity, expiry_time)
_cache: dict[str, tuple[CongressActivity, float]] = {}
_CACHE_TTL = 3600  # 1 hour (congress data updates slowly)


@dataclass
class CongressActivity:
    """Aggregated Congress trading activity for a ticker."""

    ticker: str
    net_buys_30d: int
    net_sells_30d: int
    committee_relevance: float  # 0-1, how relevant the member's committee is
    member_performance: float  # historical return of members who traded this ticker


async def fetch_congress_activity(ticker: str) -> CongressActivity | None:
    """Fetch recent Congress trades for a ticker.

    Tries Unusual Whales first (if API key configured), falls back to
    Capitol Trades free endpoint.
    """
    # Check cache
    cached = _cache.get(ticker)
    if cached and cached[1] > time.time():
        return cached[0]

    uw_key = os.getenv("UW_KEY", "")

    result = None
    if uw_key:
        result = await _fetch_from_unusual_whales(ticker, uw_key)

    if result is None:
        result = await _fetch_from_capitol_trades(ticker)

    if result is not None:
        _cache[ticker] = (result, time.time() + _CACHE_TTL)

    return result


async def _fetch_from_unusual_whales(ticker: str, api_key: str) -> CongressActivity | None:
    """Fetch Congress trades from Unusual Whales API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.unusualwhales.com/api/congress/{ticker.upper()}/trades",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            trades = data.get("data", [])
            if not trades:
                return None

            buys = sum(1 for t in trades if t.get("type", "").lower() == "purchase")
            sells = sum(1 for t in trades if t.get("type", "").lower() == "sale")

            return CongressActivity(
                ticker=ticker.upper(),
                net_buys_30d=buys,
                net_sells_30d=sells,
                committee_relevance=0.5,  # would need committee mapping
                member_performance=0.0,  # would need historical tracking
            )
    except Exception as exc:
        logger.debug(f"UW Congress fetch failed for {ticker}: {exc}")
        return None


async def _fetch_from_capitol_trades(ticker: str) -> CongressActivity | None:
    """Fetch Congress trades from Capitol Trades (free, no key).

    Capitol Trades web scraping as backup. Returns simplified data.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://www.capitoltrades.com/trades",
                params={"asset": ticker.upper(), "page": "1"},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; OptionsOwl/1.0)",
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                return None

            # Capitol Trades returns HTML by default — JSON parsing depends
            # on their API structure which may change. For now, return a
            # minimal result indicating congress activity was found.
            # Full HTML parsing would be implemented in Phase 3.5.
            return None  # TODO: parse Capitol Trades HTML/JSON response

    except Exception as exc:
        logger.debug(f"Capitol Trades fetch failed for {ticker}: {exc}")
        return None
