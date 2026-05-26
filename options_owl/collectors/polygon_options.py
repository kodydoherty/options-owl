"""Polygon.io option premium lookup — standalone, no class dependencies.

Used by both the entry path (_fill_missing_premium) and the exit path
(position_monitor) to fetch real bid/ask quotes for option contracts.

Supports two tiers:
- Options add-on ($29/mo): uses /v3/snapshot for real-time bid/ask
- Free/Starter tier: falls back to /v2/aggs for previous close
"""

from __future__ import annotations

from datetime import datetime

import httpx
from loguru import logger


def build_option_contract_ticker(
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
) -> str:
    """Build a Polygon OCC-format option ticker.

    Example: O:SPY260409C00675000
    """
    opt_char = "C" if option_type == "call" else "P"
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    expiry_str = expiry_dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"O:{ticker.upper()}{expiry_str}{opt_char}{strike_int:08d}"


async def _snapshot_quote(
    client: httpx.AsyncClient,
    api_key: str,
    ticker: str,
    contract: str,
) -> dict | None:
    """Try the /v3/snapshot endpoint (requires Options add-on).

    Returns dict with bid, ask, mid, last_trade, day_close keys,
    or None if endpoint fails.
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}/{contract}"
    resp = await client.get(url, params={"apiKey": api_key})
    if resp.status_code != 200:
        return None

    result = resp.json().get("results", {})

    quote = result.get("last_quote", {})
    bid = float(quote.get("bid") or 0)
    ask = float(quote.get("ask") or 0)
    mid = round((bid + ask) / 2.0, 2) if bid > 0 and ask > 0 else 0.0
    last_trade = float(result.get("last_trade", {}).get("price") or 0)
    day_close = float(result.get("day", {}).get("close") or 0)

    if mid > 0 or last_trade > 0 or day_close > 0:
        logger.debug(
            f"Polygon snapshot {contract}: bid=${bid}, ask=${ask}, "
            f"mid=${mid}, last=${last_trade}, close=${day_close}"
        )
        return {
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": mid,
            "last_trade": round(last_trade, 2),
            "day_close": round(day_close, 2),
        }

    return None


async def _snapshot_premium(
    client: httpx.AsyncClient,
    api_key: str,
    ticker: str,
    contract: str,
) -> float | None:
    """Try the /v3/snapshot endpoint — returns midpoint for backward compat."""
    q = await _snapshot_quote(client, api_key, ticker, contract)
    if not q:
        return None
    if q["mid"] > 0:
        return q["mid"]
    if q["last_trade"] > 0:
        return q["last_trade"]
    if q["day_close"] > 0:
        return q["day_close"]
    return None


async def _aggs_premium(
    client: httpx.AsyncClient,
    api_key: str,
    contract: str,
) -> float | None:
    """Try the /v2/aggs prev close endpoint (available on free/Starter tier)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{contract}/prev"
    resp = await client.get(url, params={"apiKey": api_key, "adjusted": "true"})
    if resp.status_code != 200:
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    bar = results[0]
    close = bar.get("c")
    if close and close > 0:
        logger.debug(f"Polygon aggs {contract}: prev close=${close}, vol={bar.get('v', 0)}")
        return round(float(close), 2)

    return None


async def polygon_option_premium(
    api_key: str,
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
) -> float | None:
    """Fetch option premium via Polygon REST.

    Tries snapshot (real-time bid/ask) first, falls back to aggregates
    (previous close) for free-tier API keys.
    """
    if not api_key:
        return None

    # Try exact strike, then nearby strikes ($0.50, $1, $2.50 away)
    # Some tickers use $2.50/$5 spacing so the exact strike may not exist
    strikes_to_try = [strike]
    for offset in [0.5, 1.0, 2.5]:
        strikes_to_try.append(round(strike - offset, 2))
        strikes_to_try.append(round(strike + offset, 2))

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            for s in strikes_to_try:
                contract = build_option_contract_ticker(ticker, s, expiry, option_type)

                # Try real-time snapshot first (Options add-on)
                premium = await _snapshot_premium(client, api_key, ticker, contract)
                if premium is not None:
                    if s != strike:
                        logger.info(f"Polygon: {ticker} exact ${strike} not found, used ${s}")
                    return premium

                # Fallback: previous day aggregates (free tier)
                premium = await _aggs_premium(client, api_key, contract)
                if premium is not None:
                    if s != strike:
                        logger.info(f"Polygon: {ticker} exact ${strike} not found, used ${s}")
                    return premium

            logger.debug(f"Polygon {ticker} ${strike} {option_type}: no data at any nearby strike")
            return None

    except Exception as e:
        logger.debug(f"Polygon option lookup failed for {ticker} ${strike}: {e}")
        return None


async def polygon_option_quote(
    api_key: str,
    ticker: str,
    strike: float,
    expiry: str,
    option_type: str,
) -> dict | None:
    """Fetch full bid/ask/mid quote for an option via Polygon REST.

    Returns dict with keys: bid, ask, mid, last_trade, day_close, used_strike.
    Only tries the exact strike — no nearby-strike fallback (which can return
    misleading prices from a different contract).
    """
    if not api_key:
        return None

    contract = build_option_contract_ticker(ticker, strike, expiry, option_type)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            q = await _snapshot_quote(client, api_key, ticker, contract)
            if q:
                q["used_strike"] = strike
                return q

            # Fallback: prev close (no bid/ask available)
            premium = await _aggs_premium(client, api_key, contract)
            if premium:
                return {
                    "bid": 0.0,
                    "ask": 0.0,
                    "mid": premium,
                    "last_trade": premium,
                    "day_close": premium,
                    "used_strike": strike,
                }

        logger.debug(f"Polygon quote {ticker} ${strike} {option_type}: no data")
        return None

    except Exception as e:
        logger.debug(f"Polygon quote lookup failed for {ticker} ${strike}: {e}")
        return None


async def polygon_option_chain(
    api_key: str,
    ticker: str,
    expiry: str,
    option_type: str | None = None,
    strike: float | None = None,
) -> list[dict]:
    """Fetch option chain snapshots from Polygon for a ticker + expiry.

    Returns a list of dicts with keys: strike, bid, ask, mid, volume,
    open_interest, last_price, option_type.
    """
    if not api_key:
        return []

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
    params: dict = {
        "apiKey": api_key,
        "expiration_date": expiry,
        "limit": 250,
    }
    if option_type:
        params["contract_type"] = option_type
    if strike is not None:
        params["strike_price"] = strike

    contracts: list[dict] = []
    max_pages = 20  # safety limit to prevent infinite pagination

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            page = 0
            while url and page < max_pages:
                page += 1
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.debug(f"Polygon chain {resp.status_code} for {ticker}")
                    break

                data = resp.json()
                for item in data.get("results", []):
                    details = item.get("details", {})
                    quote = item.get("last_quote", {})
                    day = item.get("day", {})
                    last_trade = item.get("last_trade", {})

                    bid = quote.get("bid", 0) or 0
                    ask = quote.get("ask", 0) or 0
                    mid = round((bid + ask) / 2.0, 2) if bid > 0 and ask > 0 else 0
                    last_price = last_trade.get("price", 0) or day.get("close", 0) or 0

                    contracts.append({
                        "strike": details.get("strike_price", 0),
                        "option_type": details.get("contract_type", "").lower(),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid or round(float(last_price), 2),
                        "last_price": round(float(last_price), 2),
                        "volume": day.get("volume", 0),
                        "open_interest": item.get("open_interest", 0),
                    })

                # Pagination
                next_url = data.get("next_url")
                if next_url:
                    url = next_url
                    params = {"apiKey": api_key}  # next_url includes other params
                else:
                    break

    except Exception as e:
        logger.debug(f"Polygon chain fetch failed for {ticker}: {e}")

    logger.debug(f"Polygon chain: {ticker} exp={expiry} → {len(contracts)} contracts")
    return contracts
