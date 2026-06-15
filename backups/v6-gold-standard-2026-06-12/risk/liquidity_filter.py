"""Liquidity filter — rejects options with low open interest, volume, or wide bid-ask spreads.

Thin options have wide spreads which lead to bad fills and phantom profits in paper trading.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger
from pydantic import BaseModel

from options_owl.config.settings import Settings


class OptionLiquidity(BaseModel):
    """Snapshot of liquidity metrics for a single option contract."""

    ticker: str
    strike: float
    expiry: str
    option_type: str  # "call" or "put"

    open_interest: int | None = None
    volume: int | None = None
    bid: float | None = None
    ask: float | None = None
    bid_ask_spread_pct: float | None = None  # spread as % of midpoint


async def fetch_option_liquidity(
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
    settings: Settings,
) -> OptionLiquidity:
    """Fetch liquidity data for an option contract.

    Tries Polygon REST API first (if POLYGON_API_KEY is set), then falls back
    to yfinance option chain data. Returns whatever data is available.
    """
    liq = OptionLiquidity(
        ticker=ticker, strike=strike, expiry=expiry, option_type=option_type,
    )

    # Try Polygon first
    if settings.POLYGON_API_KEY:
        polygon_liq = await _fetch_polygon_liquidity(ticker, strike, expiry, option_type, settings)
        if polygon_liq is not None:
            return polygon_liq

    # Fallback to yfinance
    yf_liq = await _fetch_yfinance_liquidity(ticker, strike, expiry, option_type)
    if yf_liq is not None:
        return yf_liq

    return liq


async def _fetch_polygon_liquidity(
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
    settings: Settings,
) -> OptionLiquidity | None:
    """Fetch liquidity from Polygon REST /v3/snapshot/options/{ticker}/{contract}."""
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed — skipping Polygon liquidity lookup")
        return None

    opt_char = "C" if option_type == "call" else "P"
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    expiry_str = expiry_dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    contract = f"O:{ticker}{expiry_str}{opt_char}{strike_int:08d}"

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}/{contract}"
    params = {"apiKey": settings.POLYGON_API_KEY}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.debug(f"Polygon liquidity {resp.status_code} for {contract}")
                return None

            data = resp.json()
            result = data.get("results", {})

            open_interest = result.get("open_interest")
            day = result.get("day", {})
            volume = day.get("volume")

            last_quote = result.get("last_quote", {})
            bid = last_quote.get("bid")
            ask = last_quote.get("ask")

            spread_pct = _calc_spread_pct(bid, ask)

            return OptionLiquidity(
                ticker=ticker,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                open_interest=int(open_interest) if open_interest is not None else None,
                volume=int(volume) if volume is not None else None,
                bid=float(bid) if bid is not None else None,
                ask=float(ask) if ask is not None else None,
                bid_ask_spread_pct=spread_pct,
            )
    except Exception as e:
        logger.debug(f"Polygon liquidity lookup failed for {contract}: {e}")
        return None


async def _fetch_yfinance_liquidity(
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
) -> OptionLiquidity | None:
    """Fetch liquidity from yfinance option chain."""
    import math

    def _sync() -> OptionLiquidity | None:
        try:
            import yfinance as yf

            tk = yf.Ticker(ticker)
            chain = tk.option_chain(expiry)
            df = chain.calls if option_type == "call" else chain.puts
            if df is None or df.empty:
                return None

            matches = df[abs(df["strike"] - strike) < 0.01]
            if matches.empty:
                return None

            row = matches.iloc[0]

            oi = row.get("openInterest")
            vol = row.get("volume")
            bid = row.get("bid")
            ask = row.get("ask")

            # Clean NaN values
            if isinstance(oi, float) and math.isnan(oi):
                oi = None
            if isinstance(vol, float) and math.isnan(vol):
                vol = None
            if isinstance(bid, float) and math.isnan(bid):
                bid = None
            if isinstance(ask, float) and math.isnan(ask):
                ask = None

            spread_pct = _calc_spread_pct(bid, ask)

            return OptionLiquidity(
                ticker=ticker,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                open_interest=int(oi) if oi is not None else None,
                volume=int(vol) if vol is not None else None,
                bid=float(bid) if bid is not None else None,
                ask=float(ask) if ask is not None else None,
                bid_ask_spread_pct=spread_pct,
            )
        except Exception as e:
            logger.debug(f"yfinance liquidity lookup failed for {ticker}: {e}")
            return None

    return await asyncio.to_thread(_sync)


def _calc_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Calculate bid-ask spread as a percentage of the midpoint."""
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    midpoint = (bid + ask) / 2.0
    if midpoint <= 0:
        return None
    return round((ask - bid) / midpoint * 100.0, 2)


def check_liquidity(
    liquidity: OptionLiquidity,
    settings: Settings,
) -> tuple[bool, str]:
    """Check whether an option meets minimum liquidity thresholds.

    Returns (passes, reason). If no data is available for a check, that check
    is skipped (we don't block on missing data).
    """
    min_oi = settings.MIN_OPEN_INTEREST
    min_vol = settings.MIN_VOLUME
    max_spread = settings.MAX_BID_ASK_SPREAD_PCT

    # Check 1: open interest
    if liquidity.open_interest is not None:
        if liquidity.open_interest < min_oi:
            return (
                False,
                f"Open interest {liquidity.open_interest} < min {min_oi}",
            )

    # Check 2: volume
    if liquidity.volume is not None:
        if liquidity.volume < min_vol:
            return (
                False,
                f"Volume {liquidity.volume} < min {min_vol}",
            )

    # Check 3: bid-ask spread
    if liquidity.bid_ask_spread_pct is not None:
        if liquidity.bid_ask_spread_pct > max_spread:
            return (
                False,
                f"Bid-ask spread {liquidity.bid_ask_spread_pct:.1f}% > max {max_spread:.1f}%",
            )

    # Build a summary of what passed
    parts: list[str] = []
    if liquidity.open_interest is not None:
        parts.append(f"OI={liquidity.open_interest}")
    if liquidity.volume is not None:
        parts.append(f"Vol={liquidity.volume}")
    if liquidity.bid_ask_spread_pct is not None:
        parts.append(f"Spread={liquidity.bid_ask_spread_pct:.1f}%")

    if not parts:
        return True, "No liquidity data available — passed by default"

    return True, f"Liquidity OK: {', '.join(parts)}"
