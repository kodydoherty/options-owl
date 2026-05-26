"""Read real option data from the shared harvester DB for ML features.

The harvester captures option snapshots (bid/ask/IV/delta/volume) every ~90s.
This module reads the most recent snapshot for a ticker's ATM contract,
giving the ML model REAL option-level features instead of stock candle proxies.

The harvester DB is mounted read-write (WAL mode) at SHARED_CANDLE_DB path,
which is the same options_data.db used for candle reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import aiosqlite
from loguru import logger
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass
class OptionSnapshot:
    """Recent option data for ML feature computation."""

    ticker: str
    strike: float
    option_type: str  # "call" or "put"
    expiry: str
    underlying_price: float
    bid: float
    ask: float
    midpoint: float
    volume: int
    iv: float
    delta: float
    theta: float
    vega: float
    captured_at: str


@dataclass
class OptionHistory:
    """Time series of recent option snapshots for premium history."""

    snapshots: list[OptionSnapshot]

    @property
    def premium_history(self) -> list[float]:
        return [s.midpoint for s in self.snapshots if s.midpoint > 0]

    @property
    def volume_history(self) -> list[int]:
        return [s.volume for s in self.snapshots]

    @property
    def underlying_history(self) -> list[float]:
        return [s.underlying_price for s in self.snapshots if s.underlying_price > 0]


async def fetch_atm_option_snapshot(
    ticker: str,
    direction: str,
    db_path: str,
) -> OptionSnapshot | None:
    """Fetch the most recent ATM option snapshot from harvester DB.

    Args:
        ticker: Stock symbol (e.g. "SPY")
        direction: "CALL" or "PUT"
        db_path: Path to the shared harvester options_data.db
    """
    option_type = "call" if direction.upper() == "CALL" else "put"

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode = WAL")
            await db.execute("PRAGMA busy_timeout = 5000")

            row = await db.execute_fetchall(
                """SELECT hs.contract_ticker, hc.strike, hc.option_type, hc.expiry_date,
                          hs.underlying_price, hs.bid, hs.ask, hs.midpoint,
                          hs.day_volume, hs.implied_volatility, hs.delta, hs.theta, hs.vega,
                          hs.captured_at
                   FROM harvest_snapshots hs
                   JOIN harvest_contracts hc ON hs.contract_ticker = hc.contract_ticker
                   WHERE hc.underlying = ? AND hc.option_type = ?
                     AND ABS(hc.strike - hs.underlying_price) < 3
                     AND hs.bid > 0 AND hs.ask > 0
                   ORDER BY hs.captured_at DESC, ABS(hc.strike - hs.underlying_price)
                   LIMIT 1""",
                (ticker, option_type),
            )

            if not row:
                return None

            r = row[0]
            return OptionSnapshot(
                ticker=ticker,
                strike=float(r[1]),
                option_type=str(r[2]),
                expiry=str(r[3]),
                underlying_price=float(r[4] or 0),
                bid=float(r[5] or 0),
                ask=float(r[6] or 0),
                midpoint=float(r[7] or 0),
                volume=int(r[8] or 0),
                iv=float(r[9] or 0),
                delta=float(r[10] or 0) if r[10] else 0.0,
                theta=float(r[11] or 0) if r[11] else 0.0,
                vega=float(r[12] or 0) if r[12] else 0.0,
                captured_at=str(r[13]),
            )
    except Exception as exc:
        logger.debug(f"Harvester option snapshot failed for {ticker}: {exc}")
        return None


async def fetch_option_history(
    ticker: str,
    direction: str,
    db_path: str,
    lookback_minutes: int = 75,
    max_snapshots: int = 15,
) -> OptionHistory | None:
    """Fetch recent option snapshot history for premium/volume time series.

    Gets up to max_snapshots of the ATM contract from the last lookback_minutes.
    Used to build premium_history, volume_history for ML features.
    """
    option_type = "call" if direction.upper() == "CALL" else "put"

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode = WAL")
            await db.execute("PRAGMA busy_timeout = 5000")

            # First find the current ATM strike
            atm_row = await db.execute_fetchall(
                """SELECT hc.strike, hc.contract_ticker
                   FROM harvest_snapshots hs
                   JOIN harvest_contracts hc ON hs.contract_ticker = hc.contract_ticker
                   WHERE hc.underlying = ? AND hc.option_type = ?
                     AND ABS(hc.strike - hs.underlying_price) < 3
                     AND hs.bid > 0 AND hs.ask > 0
                   ORDER BY hs.captured_at DESC, ABS(hc.strike - hs.underlying_price)
                   LIMIT 1""",
                (ticker, option_type),
            )

            if not atm_row:
                return None

            contract_ticker = atm_row[0][1]
            strike = atm_row[0][0]

            # Now get history for that specific contract
            rows = await db.execute_fetchall(
                """SELECT hs.underlying_price, hs.bid, hs.ask, hs.midpoint,
                          hs.day_volume, hs.implied_volatility, hs.delta, hs.theta, hs.vega,
                          hs.captured_at, hc.expiry_date
                   FROM harvest_snapshots hs
                   JOIN harvest_contracts hc ON hs.contract_ticker = hc.contract_ticker
                   WHERE hs.contract_ticker = ?
                     AND hs.bid > 0 AND hs.ask > 0
                   ORDER BY hs.captured_at DESC
                   LIMIT ?""",
                (contract_ticker, max_snapshots),
            )

            if not rows:
                return None

            snapshots = []
            for r in reversed(rows):  # oldest first
                snapshots.append(OptionSnapshot(
                    ticker=ticker,
                    strike=float(strike),
                    option_type=option_type,
                    expiry=str(r[10]),
                    underlying_price=float(r[0] or 0),
                    bid=float(r[1] or 0),
                    ask=float(r[2] or 0),
                    midpoint=float(r[3] or 0),
                    volume=int(r[4] or 0),
                    iv=float(r[5] or 0),
                    delta=float(r[6] or 0) if r[6] else 0.0,
                    theta=float(r[7] or 0) if r[7] else 0.0,
                    vega=float(r[8] or 0) if r[8] else 0.0,
                    captured_at=str(r[9]),
                ))

            return OptionHistory(snapshots=snapshots)
    except Exception as exc:
        logger.debug(f"Harvester option history failed for {ticker}: {exc}")
        return None
