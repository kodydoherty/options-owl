"""Fetch options chain snapshots from Polygon API.

Reuses the existing polygon_options module for the actual API call,
wrapping it in a clean interface for the sourcing agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import httpx
from loguru import logger


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


async def fetch_options_chain(
    ticker: str,
    direction: str,
    underlying_price: float,
    expiry: str | None = None,
) -> OptionsChain | None:
    """Fetch ATM options chain from Polygon snapshot API.

    Args:
        ticker: Stock symbol.
        direction: "CALL" or "PUT".
        underlying_price: Current stock price (for ATM strike selection).
        expiry: Target expiry date (YYYY-MM-DD). Defaults to today.

    Returns OptionsChain or None if no valid contract found.
    """
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.debug("No POLYGON_API_KEY — skipping options chain lookup")
        return None

    if expiry is None:
        expiry = datetime.now().strftime("%Y-%m-%d")

    option_type = "call" if direction.upper() == "CALL" else "put"

    # Round to nearest $1 for ATM strike (most liquid)
    atm_strike = round(underlying_price)

    # Try ATM, then 1 strike OTM
    strikes_to_try = [atm_strike]
    if option_type == "call":
        strikes_to_try.append(atm_strike + 1)
    else:
        strikes_to_try.append(atm_strike - 1)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            for strike in strikes_to_try:
                chain = await _try_snapshot(client, api_key, ticker, strike, expiry, option_type)
                if chain is not None:
                    return chain
    except Exception as exc:
        logger.warning(f"Options chain fetch failed for {ticker}: {exc}")

    return None


async def _try_snapshot(
    client: httpx.AsyncClient,
    api_key: str,
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
) -> OptionsChain | None:
    """Try to fetch a single contract snapshot from Polygon."""
    from options_owl.collectors.polygon_options import build_option_contract_ticker

    contract = build_option_contract_ticker(ticker, strike, expiry, option_type)
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}/{contract}"

    resp = await client.get(url, params={"apiKey": api_key})
    if resp.status_code != 200:
        return None

    result = resp.json().get("results", {})
    quote = result.get("last_quote", {})
    day = result.get("day", {})
    greeks = result.get("greeks", {})

    bid = float(quote.get("bid") or 0)
    ask = float(quote.get("ask") or 0)
    mid = round((bid + ask) / 2.0, 2) if bid > 0 and ask > 0 else 0.0

    if mid <= 0:
        return None

    spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999.0

    return OptionsChain(
        strike=strike,
        expiry=expiry,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_pct=round(spread_pct, 1),
        volume=int(day.get("volume") or 0),
        open_interest=int(result.get("open_interest") or 0),
        iv=float(greeks.get("implied_volatility") or 0),
    )
