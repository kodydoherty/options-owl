"""Polygon news sentiment + real-time news sentinel for open positions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NewsHeadline:
    title: str
    published_at: str
    source: str
    ticker: str
    sentiment: str  # positive / negative / neutral


async def fetch_news(ticker: str, limit: int = 5) -> list[NewsHeadline]:
    """Fetch recent news headlines from Polygon News API.

    Cached for 60 seconds to avoid rate limits during sentinel polling.
    """
    raise NotImplementedError("Phase 3: wire up Polygon news API")


def classify_headline(title: str, direction: str) -> str:
    """Classify headline sentiment relative to trade direction.

    Returns: 'strongly_negative', 'negative', 'neutral', 'positive'
    """
    raise NotImplementedError("Phase 3.5: implement keyword + ML classification")
