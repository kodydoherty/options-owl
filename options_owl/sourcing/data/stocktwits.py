"""StockTwits retail sentiment (contrarian signal).

Free API, no key required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetailSentiment:
    """StockTwits sentiment data for a ticker."""

    ticker: str
    bull_ratio: float  # 0.0 to 1.0
    total_messages: int
    msg_velocity: float  # messages per hour
    contrarian_signal: str  # "bullish_contrarian" / "bearish_contrarian" / "neutral"


async def fetch_sentiment(ticker: str) -> RetailSentiment | None:
    """Fetch current retail sentiment from StockTwits.

    Cached per scan cycle (3 min).
    """
    raise NotImplementedError("Phase 3.5: implement StockTwits API")
