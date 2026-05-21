"""Fetch options chain snapshots from Polygon API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptionsChain:
    """Validated options chain data for a specific strike/expiry."""

    strike: float
    expiry: str
    bid: float
    ask: float
    mid: float
    spread_pct: float
    volume: int
    open_interest: int
    iv: float


async def fetch_options_chain(ticker: str, direction: str, expiry: str | None = None) -> OptionsChain | None:
    """Fetch and validate ATM options chain from Polygon.

    Returns None if no valid contract found.
    """
    raise NotImplementedError("Phase 3: wire up Polygon options API")
