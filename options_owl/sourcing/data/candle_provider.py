"""Read candles from harvester DB (primary) or Twelve Data (fallback)."""

from __future__ import annotations


async def fetch_candles(ticker: str, interval: str = "5min", bars: int = 78) -> list[dict] | None:
    """Fetch OHLCV candle data for a ticker.

    Priority:
        1. Harvester DB (WAL mode, zero API cost)
        2. Twelve Data REST API (fallback if harvester stale)

    Returns None if both sources fail.
    """
    raise NotImplementedError("Phase 1: wire up harvester DB reads")
