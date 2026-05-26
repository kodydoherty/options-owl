"""StockTwits retail sentiment (contrarian signal).

Free API, no key required. Rate limit: 200 req/hr.
Endpoint: https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json

The contrarian logic: when retail is overwhelmingly bullish (>80% bulls),
that's actually a bearish signal (crowded trade). And vice versa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from loguru import logger

# Simple in-memory cache: ticker -> (RetailSentiment, expiry_time)
_cache: dict[str, tuple[RetailSentiment, float]] = {}
_CACHE_TTL = 180  # 3 minutes (one scan cycle)


@dataclass
class RetailSentiment:
    """StockTwits sentiment data for a ticker."""

    ticker: str
    bull_ratio: float  # 0.0 to 1.0
    total_messages: int
    msg_velocity: float  # messages per hour (estimated)
    contrarian_signal: str  # "bullish_contrarian" / "bearish_contrarian" / "neutral"


def _classify_contrarian(bull_ratio: float, total_messages: int) -> str:
    """Classify sentiment as contrarian signal.

    High conviction retail consensus = contrarian opportunity.
    Low message count = not enough data to be meaningful.
    """
    if total_messages < 5:
        return "neutral"  # not enough data

    if bull_ratio > 0.80:
        return "bearish_contrarian"  # retail too bullish → fade
    if bull_ratio < 0.20:
        return "bullish_contrarian"  # retail too bearish → fade
    return "neutral"


async def fetch_sentiment(ticker: str) -> RetailSentiment | None:
    """Fetch current retail sentiment from StockTwits.

    Returns None if API fails or ticker not found.
    Cached for 3 minutes (one scan cycle).
    """
    # Check cache
    cached = _cache.get(ticker)
    if cached and cached[1] > time.time():
        return cached[0]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json",
            )

            if resp.status_code != 200:
                logger.debug(f"StockTwits: {resp.status_code} for {ticker}")
                return None

            data = resp.json()
            symbol_data = data.get("symbol", {})
            messages = data.get("messages", [])

            # Extract sentiment from symbol watchlist stats
            sentiment_data = symbol_data.get("watchlist_count", 0)

            # Count bull/bear from recent messages
            bulls = 0
            bears = 0
            for msg in messages:
                entities = msg.get("entities", {})
                sentiment = entities.get("sentiment", {})
                if sentiment:
                    basic = sentiment.get("basic", "")
                    if basic == "Bullish":
                        bulls += 1
                    elif basic == "Bearish":
                        bears += 1

            total = bulls + bears
            bull_ratio = bulls / total if total > 0 else 0.5

            # Estimate velocity (messages per hour from the batch)
            msg_velocity = len(messages) * 2.0  # rough estimate (30 msgs = ~60/hr)

            contrarian = _classify_contrarian(bull_ratio, total)

            result = RetailSentiment(
                ticker=ticker.upper(),
                bull_ratio=round(bull_ratio, 2),
                total_messages=total,
                msg_velocity=round(msg_velocity, 1),
                contrarian_signal=contrarian,
            )

            _cache[ticker] = (result, time.time() + _CACHE_TTL)

            if contrarian != "neutral":
                logger.info(
                    f"StockTwits {ticker}: {contrarian} "
                    f"(bull_ratio={bull_ratio:.0%}, msgs={total})"
                )

            return result

    except Exception as exc:
        logger.debug(f"StockTwits fetch failed for {ticker}: {exc}")
        return None
