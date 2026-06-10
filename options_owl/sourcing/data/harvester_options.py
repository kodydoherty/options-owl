"""Read real option data from PostgreSQL for ML features.

The harvester captures option snapshots (bid/ask/IV/delta/volume) every ~60s
into the PG option_ticks table. This module reads the most recent snapshot
for a ticker's ATM contract, giving the ML model REAL option-level features.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


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
    bid_size: int
    ask_size: int
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
    db_path: str = "",
) -> OptionSnapshot | None:
    """Fetch the most recent ATM option snapshot from PostgreSQL.

    Args:
        ticker: Stock symbol (e.g. "SPY")
        direction: "CALL" or "PUT"
        db_path: Ignored (kept for backward compat)
    """
    option_type = "call" if direction.upper() == "CALL" else "put"

    try:
        from options_owl.db import postgres as pg
        if not pg.is_connected():
            return None

        rows = await pg.fetch(
            """SELECT strike, option_type, expiry_date,
                      underlying_price, bid, ask, bid_size, ask_size, mid, volume,
                      iv, delta, theta, vega, captured_at
               FROM option_ticks
               WHERE ticker = $1 AND option_type = $2
                 AND ABS(strike - underlying_price) < 3
                 AND bid > 0 AND ask > 0
               ORDER BY captured_at DESC, ABS(strike - underlying_price)
               LIMIT 1""",
            ticker, option_type,
        )

        if not rows:
            return None

        r = rows[0]
        return OptionSnapshot(
            ticker=ticker,
            strike=float(r["strike"]),
            option_type=str(r["option_type"]),
            expiry=str(r["expiry_date"]),
            underlying_price=float(r["underlying_price"] or 0),
            bid=float(r["bid"] or 0),
            ask=float(r["ask"] or 0),
            bid_size=int(r["bid_size"] or 0),
            ask_size=int(r["ask_size"] or 0),
            midpoint=float(r["mid"] or 0),
            volume=int(r["volume"] or 0),
            iv=float(r["iv"] or 0),
            delta=float(r["delta"] or 0),
            theta=float(r["theta"] or 0),
            vega=float(r["vega"] or 0),
            captured_at=str(r["captured_at"]),
        )
    except Exception as exc:
        logger.debug(f"PG option snapshot failed for {ticker}: {exc}")
        return None


async def fetch_option_history(
    ticker: str,
    direction: str,
    db_path: str = "",
    lookback_minutes: int = 75,
    max_snapshots: int = 15,
) -> OptionHistory | None:
    """Fetch recent option snapshot history for premium/volume time series.

    Gets up to max_snapshots of the ATM contract from the last lookback_minutes.
    """
    option_type = "call" if direction.upper() == "CALL" else "put"

    try:
        from options_owl.db import postgres as pg
        if not pg.is_connected():
            return None

        # Find current ATM contract
        atm_rows = await pg.fetch(
            """SELECT strike, expiry_date
               FROM option_ticks
               WHERE ticker = $1 AND option_type = $2
                 AND ABS(strike - underlying_price) < 3
                 AND bid > 0 AND ask > 0
               ORDER BY captured_at DESC, ABS(strike - underlying_price)
               LIMIT 1""",
            ticker, option_type,
        )

        if not atm_rows:
            return None

        strike = atm_rows[0]["strike"]
        expiry = atm_rows[0]["expiry_date"]

        # Get history for that specific contract
        rows = await pg.fetch(
            """SELECT underlying_price, bid, ask, bid_size, ask_size, mid, volume,
                      iv, delta, theta, vega, captured_at, expiry_date
               FROM option_ticks
               WHERE ticker = $1 AND option_type = $2
                 AND strike = $3 AND expiry_date = $4
                 AND bid > 0 AND ask > 0
                 AND captured_at >= NOW() - INTERVAL '%s minutes'
               ORDER BY captured_at DESC
               LIMIT $5""" % lookback_minutes,
            ticker, option_type, strike, expiry, max_snapshots,
        )

        if not rows:
            return None

        snapshots = []
        for r in reversed(rows):  # oldest first
            snapshots.append(OptionSnapshot(
                ticker=ticker,
                strike=float(strike),
                option_type=option_type,
                expiry=str(r["expiry_date"]),
                underlying_price=float(r["underlying_price"] or 0),
                bid=float(r["bid"] or 0),
                ask=float(r["ask"] or 0),
                bid_size=int(r["bid_size"] or 0),
                ask_size=int(r["ask_size"] or 0),
                midpoint=float(r["mid"] or 0),
                volume=int(r["volume"] or 0),
                iv=float(r["iv"] or 0),
                delta=float(r["delta"] or 0),
                theta=float(r["theta"] or 0),
                vega=float(r["vega"] or 0),
                captured_at=str(r["captured_at"]),
            ))

        return OptionHistory(snapshots=snapshots)
    except Exception as exc:
        logger.debug(f"PG option history failed for {ticker}: {exc}")
        return None
