"""SEC EDGAR Form 4 insider trades (officers, directors, 10%+ owners).

Free API, no key required. Rate limit: 10 req/sec.
User-Agent header required by SEC (they block default user agents).

EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=%22TICKER%22&dateRange=custom&startdt=2026-05-14&enddt=2026-05-21&forms=4
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
from loguru import logger

# Simple in-memory cache: ticker -> (InsiderActivity, expiry_time)
_cache: dict[str, tuple[InsiderActivity, float]] = {}
_CACHE_TTL = 3600  # 1 hour


@dataclass
class InsiderActivity:
    """Aggregated insider trading activity for a ticker."""

    ticker: str
    net_buys_7d: int
    net_sells_7d: int
    largest_buy_dollars: float
    insider_buy_ratio: float  # buys / (buys + sells), 0-1


# Mapping of common tickers to SEC CIK numbers (central index key).
# This avoids an extra API call to resolve ticker -> CIK.
_TICKER_TO_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "AMD": "0000002488",
    "SPY": "",  # ETFs don't have insider filings
    "QQQ": "",
    "IWM": "",
    "AVGO": "0001649338",
    "MSTR": "0001050446",
    "PLTR": "0001321655",
    "MU": "0000723125",
    "COIN": "0001679788",
    "SMCI": "0001375365",
}


async def fetch_insider_activity(ticker: str) -> InsiderActivity | None:
    """Fetch and aggregate recent Form 4 filings from SEC EDGAR.

    Uses EDGAR full-text search API (no auth required, 10 req/sec).
    Cached for 1 hour since filings come in batches, not real-time.
    """
    # Skip ETFs (no insider filings)
    if ticker.upper() in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"):
        return None

    # Check cache
    cached = _cache.get(ticker)
    if cached and cached[1] > time.time():
        return cached[0]

    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=10) as client:
            # EDGAR full-text search for Form 4 filings mentioning this ticker
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{ticker.upper()}"',
                    "forms": "4",
                    "dateRange": "custom",
                    "startdt": start_date,
                    "enddt": end_date,
                },
                headers={
                    "User-Agent": "OptionsOwl/1.0 (kody@optionsowl.com)",
                    "Accept": "application/json",
                },
            )

            if resp.status_code != 200:
                logger.debug(f"SEC EDGAR: {resp.status_code} for {ticker}")
                return None

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                return None

            # Parse Form 4 filings for buy/sell counts
            buys = 0
            sells = 0
            largest_buy = 0.0

            for hit in hits:
                source = hit.get("_source", {})
                # Form 4 filings contain transaction codes:
                # P = purchase, S = sale, A = grant/award
                form_type = source.get("form_type", "")
                if form_type != "4":
                    continue

                # Count based on filing presence (detailed parsing would
                # require fetching each XML filing — too slow for scan loop)
                # For now, use filing count as a proxy signal
                buys += 1  # placeholder — will refine with XML parsing

            total = buys + sells
            ratio = buys / total if total > 0 else 0.5

            activity = InsiderActivity(
                ticker=ticker.upper(),
                net_buys_7d=buys,
                net_sells_7d=sells,
                largest_buy_dollars=largest_buy,
                insider_buy_ratio=ratio,
            )

            _cache[ticker] = (activity, time.time() + _CACHE_TTL)
            return activity

    except Exception as exc:
        logger.debug(f"SEC EDGAR fetch failed for {ticker}: {exc}")
        return None
