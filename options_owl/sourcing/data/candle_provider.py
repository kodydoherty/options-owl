"""Read candles from harvester DB (primary) or Polygon REST (fallback).

The harvester DB is a shared SQLite file (WAL mode) written by owlet-harvester.
All trading agents and the sourcing agent read from it concurrently.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from loguru import logger


async def fetch_candles(
    ticker: str,
    interval: str = "5min",
    bars: int = 78,
    db_path: str = "/app/shared_harvester/options_data.db",
) -> list[dict] | None:
    """Fetch OHLCV candle data for a ticker from the harvester DB.

    Uses WAL mode with busy_timeout for concurrent reads alongside the
    harvester's writes. Never copies the DB file (it's 7GB+).

    Args:
        ticker: Stock symbol (e.g. "SPY").
        interval: Candle timeframe. Maps "5min" -> "5m" for DB query.
        bars: Number of bars to fetch.
        db_path: Path to the shared harvester SQLite DB.

    Returns:
        List of candle dicts (oldest-first) with keys:
        open, high, low, close, volume. Returns None on failure.
    """
    path = Path(db_path)
    if not path.exists():
        logger.debug(f"Harvester DB not found: {db_path}")
        return None

    # Map interval names to DB timeframe values
    tf_map = {"5min": "5m", "15min": "15m", "1min": "1m", "5m": "5m", "15m": "15m", "1m": "1m"}
    tf = tf_map.get(interval, interval)

    try:
        async with aiosqlite.connect(str(path)) as conn:
            await conn.execute("PRAGMA busy_timeout = 5000")
            await conn.execute("PRAGMA journal_mode = WAL")
            conn.row_factory = aiosqlite.Row

            cursor = await conn.execute(
                """SELECT bar_start_ts, open, high, low, close, volume, vwap
                   FROM stock_candles
                   WHERE ticker = ? AND timeframe = ?
                   ORDER BY bar_start_ts DESC
                   LIMIT ?""",
                (ticker.upper(), tf, bars),
            )
            rows = await cursor.fetchall()

        if not rows:
            logger.debug(f"No candles for {ticker}/{tf} in harvester DB")
            return None

        # Reverse to oldest-first order
        candles = [
            {
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"] or 0),
            }
            for r in reversed(rows)
        ]
        return candles

    except Exception as exc:
        logger.warning(f"Failed to read candles for {ticker}: {exc}")
        return None
