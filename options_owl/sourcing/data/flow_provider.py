"""Unusual Whales options flow + dark pool + GEX data."""

from __future__ import annotations


async def fetch_flow(ticker: str) -> dict | None:
    """Fetch net premium flow from Unusual Whales."""
    raise NotImplementedError("Phase 3: wire up UW flow API")


async def fetch_dark_pool(ticker: str) -> dict | None:
    """Fetch dark pool prints from Unusual Whales."""
    raise NotImplementedError("Phase 3: wire up UW dark pool API")


async def fetch_gex(ticker: str) -> dict | None:
    """Fetch gamma exposure from Unusual Whales."""
    raise NotImplementedError("Phase 3: wire up UW GEX API")


async def fetch_congress_trades(ticker: str) -> dict | None:
    """Fetch recent Congress member trades from Unusual Whales."""
    raise NotImplementedError("Phase 3.5: wire up UW Congress API")
