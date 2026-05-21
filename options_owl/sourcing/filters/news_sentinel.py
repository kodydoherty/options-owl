"""Real-time news monitoring for open positions.

Polls Polygon News API every 60s for tickers with open trades.
Triggers emergency exit or trail tightening on breaking negative news.
"""

from __future__ import annotations

from options_owl.sourcing.data.news_provider import classify_headline, fetch_news


async def check_news_sentinel(ticker: str, direction: str) -> dict | None:
    """Check for breaking news that affects an open position.

    Returns:
        None if no actionable news.
        {"action": "exit", "reason": "..."} for immediate exit.
        {"action": "tighten", "multiplier": 0.5} for trail tightening.
    """
    raise NotImplementedError("Phase 3.5: implement news sentinel gate")
