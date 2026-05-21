"""SEC EDGAR Form 4 insider trades (officers, directors, 10%+ owners).

Free API, no key required. Rate limit: 10 req/sec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InsiderActivity:
    """Aggregated insider trading activity for a ticker."""

    ticker: str
    net_buys_7d: int  # net insider buys in last 7 days
    net_sells_7d: int
    largest_buy_dollars: float
    insider_buy_ratio: float  # buys / (buys + sells)


async def fetch_insider_activity(ticker: str) -> InsiderActivity | None:
    """Fetch and aggregate recent Form 4 filings from SEC EDGAR.

    Cached for 1 hour (filings come in batches, not real-time).
    """
    raise NotImplementedError("Phase 3.5: implement SEC EDGAR Form 4 parsing")
