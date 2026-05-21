"""Congress member stock trades (STOCK Act disclosures).

Backup for UW Congress endpoint. Free tier available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CongressActivity:
    """Aggregated Congress trading activity for a ticker."""

    ticker: str
    net_buys_30d: int
    committee_relevance: float  # 0-1, how relevant the member's committee is
    member_performance: float  # historical return of members who traded this ticker


async def fetch_congress_activity(ticker: str) -> CongressActivity | None:
    """Fetch recent Congress trades for a ticker."""
    raise NotImplementedError("Phase 3.5: implement Capitol Trades API")
