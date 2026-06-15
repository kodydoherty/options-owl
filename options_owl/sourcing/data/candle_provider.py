"""Read candles from PostgreSQL (primary data store).

Written by owlet-harvester's candle collector. All trading agents and
the sourcing agent read from PG concurrently.
"""

from __future__ import annotations

from loguru import logger


async def fetch_candles(
    ticker: str,
    interval: str = "5min",
    bars: int = 78,
    db_path: str = "",
) -> list[dict] | None:
    """Fetch OHLCV candle data for a ticker from PostgreSQL.

    Args:
        ticker: Stock symbol (e.g. "SPY").
        interval: Candle timeframe. Maps "5min" -> "5m" for DB query.
        bars: Number of bars to fetch.
        db_path: Ignored (kept for backward compat).

    Returns:
        List of candle dicts (oldest-first) with keys:
        open, high, low, close, volume. Returns None on failure.
    """
    # Map interval names to DB timeframe values
    tf_map = {"5min": "5m", "15min": "15m", "1min": "1m", "5m": "5m", "15m": "15m", "1m": "1m"}
    tf = tf_map.get(interval, interval)

    try:
        from options_owl.db import postgres as pg
        if not pg.is_connected():
            logger.debug(f"PG not connected for candle read {ticker}/{tf}")
            return None

        candles = await pg.read_stock_candles(ticker, tf, bars)
        if not candles:
            logger.debug(f"No candles for {ticker}/{tf} in PG")
            return None

        logger.debug(f"PG candles: {len(candles)} bars for {ticker}/{tf}")
        return candles

    except Exception as exc:
        logger.warning(f"Failed to read candles for {ticker}/{tf}: {exc}")
        return None
