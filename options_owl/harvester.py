"""OptionsOwl Harvester — standalone options-data collector.

Pulls rich per-contract snapshots from Polygon Options Advanced on a timer
and stores them in a dedicated SQLite DB for later ML training and backtesting.

Captures (per contract, per poll):
- Real-time bid, ask, bid_size, ask_size, midpoint
- Last trade price + timestamp
- Day OHLC, volume, vwap
- Open interest
- Locally computed IV + greeks (delta, gamma, theta, vega) via Black-Scholes,
  since Options Advanced tier currently returns empty greeks/IV fields.

Universe, poll interval, DTE window, and strike window are all env-driven so
the container can be reconfigured without code changes.

Entry point: `python -m options_owl.harvester`
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import date, datetime

import httpx
import yfinance as yf
from loguru import logger

from options_owl.collectors.candle_collector import CandleCollector
from options_owl.collectors.flow_collector import FlowCollector
from options_owl.main import LOG_DIR, configure_logging, write_heartbeat
from options_owl.risk.greeks import (
    calc_delta,
    calc_gamma,
    calc_iv_from_premium,
    calc_theta,
    calc_vega,
)

# ---------------------------------------------------------------------------
# Config — all env-driven so we can tune without rebuilds
# ---------------------------------------------------------------------------

UNIVERSE = [
    t.strip().upper()
    for t in os.getenv(
        "HARVEST_UNIVERSE",
        "SPY,QQQ,IWM,AAPL,TSLA,NVDA,META,AMD,AMZN,GOOGL,MSFT,MU,MSTR,VIX",
    ).split(",")
    if t.strip()
]
POLL_INTERVAL = int(os.getenv("HARVEST_POLL_INTERVAL", "60"))
MAX_DTE = int(os.getenv("HARVEST_MAX_DTE", "7"))
STRIKE_WINDOW = int(os.getenv("HARVEST_STRIKE_WINDOW", "5"))
MARKET_HOURS_ONLY = os.getenv("HARVEST_MARKET_HOURS_ONLY", "true").lower() == "true"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
RISK_FREE_RATE = float(os.getenv("HARVEST_RISK_FREE_RATE", "0.045"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("HARVEST_MAX_CONCURRENT", "4"))

# Reuse position_monitor's market-hours helper to stay consistent
from options_owl.execution.position_monitor import _is_market_hours  # noqa: E402

# ---------------------------------------------------------------------------
# Schema — all market data now stored in PostgreSQL only
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Underlying price fetcher (yfinance — Polygon underlying is delayed on our plan)
# ---------------------------------------------------------------------------

_price_cache: dict[str, tuple[dict, float]] = {}  # ticker -> (quote_dict, unix_ts)
PRICE_CACHE_TTL = 15  # seconds


def _get_underlying_quote(ticker: str) -> dict | None:
    """Fetch current underlying quote (price + bid/ask/volume) via yfinance.

    Returns dict with keys: price, bid, ask, volume, vwap. Cached for PRICE_CACHE_TTL.
    """
    now = time.time()
    cached = _price_cache.get(ticker)
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0]
    try:
        yf_symbol = f"^{ticker}" if ticker in ("VIX", "GSPC", "DJI", "IXIC") else ticker
        tk = yf.Ticker(yf_symbol)
        fi = tk.fast_info
        price = fi.get("lastPrice") or fi.get("last_price")
        if not price or price <= 0:
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
        price = float(price)
        quote = {
            "price": price,
            "bid": float(fi.get("bid", 0) or 0),
            "ask": float(fi.get("ask", 0) or 0),
            "volume": int(fi.get("lastVolume", 0) or fi.get("volume", 0) or 0),
            "vwap": float(fi.get("dayHigh", 0) + fi.get("dayLow", 0)) / 2
            if fi.get("dayHigh") and fi.get("dayLow")
            else 0,
        }
        _price_cache[ticker] = (quote, now)
        return quote
    except Exception as exc:
        logger.warning(f"yfinance quote fetch failed for {ticker}: {exc}")
        return None


def _get_underlying_price(ticker: str) -> float | None:
    """Legacy wrapper — returns just the price for backward compat."""
    quote = _get_underlying_quote(ticker)
    return quote["price"] if quote else None


# ---------------------------------------------------------------------------
# Polygon chain fetcher
# ---------------------------------------------------------------------------


async def _fetch_chain(
    client: httpx.AsyncClient,
    ticker: str,
    strike_low: float,
    strike_high: float,
    expiry_lo: str,
    expiry_hi: str,
) -> list[dict]:
    """Fetch filtered option chain snapshot for a ticker.

    Uses Polygon's strike_price and expiration_date filters to keep the
    response small (ATM ± window, near-term expiries only).
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
    params = {
        "strike_price.gte": strike_low,
        "strike_price.lte": strike_high,
        "expiration_date.gte": expiry_lo,
        "expiration_date.lte": expiry_hi,
        "limit": 250,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        resp = await client.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(
                f"Polygon chain fetch failed for {ticker}: "
                f"HTTP {resp.status_code} {resp.text[:160]}"
            )
            return []
        return resp.json().get("results", []) or []
    except Exception as exc:
        logger.warning(f"Polygon chain fetch error for {ticker}: {type(exc).__name__}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Snapshot parsing + greeks computation
# ---------------------------------------------------------------------------


def _parse_contract(result: dict, underlying_price: float | None) -> dict | None:
    """Convert a Polygon chain result into a flat row ready for DB insertion.

    Computes IV and greeks locally from the Black-Scholes model since Polygon
    Options Advanced returns empty greeks on our plan.
    """
    details = result.get("details") or {}
    contract_ticker = details.get("ticker")
    if not contract_ticker:
        return None

    strike = details.get("strike_price")
    expiry = details.get("expiration_date")
    option_type = (details.get("contract_type") or "").lower()
    if strike is None or not expiry or option_type not in ("call", "put"):
        return None

    quote = result.get("last_quote") or {}
    trade = result.get("last_trade") or {}
    day = result.get("day") or {}

    bid = quote.get("bid")
    ask = quote.get("ask")
    midpoint = quote.get("midpoint")
    if midpoint is None and bid and ask:
        midpoint = round((bid + ask) / 2.0, 4)

    # Local BS greeks only if we have underlying price + a reasonable premium
    iv = delta = gamma = theta = vega = None
    if underlying_price and midpoint and midpoint > 0:
        try:
            expiry_dt = date.fromisoformat(expiry)
            today = date.today()
            days_to_expiry = (expiry_dt - today).days
            # For 0DTE use fractional time remaining (hours / 24)
            if days_to_expiry <= 0:
                now = datetime.now()
                close_hour = 16  # 4 PM ET-ish; close enough for 0DTE greeks
                hours_left = max(close_hour - now.hour - (now.minute / 60.0), 0.1)
                T = hours_left / (24.0 * 365.0)
            else:
                T = days_to_expiry / 365.0

            if T > 0:
                iv = calc_iv_from_premium(
                    midpoint, underlying_price, strike, T, RISK_FREE_RATE, option_type
                )
                if iv and 0.001 < iv < 5.0:
                    delta = round(
                        calc_delta(underlying_price, strike, T, RISK_FREE_RATE, iv, option_type),
                        4,
                    )
                    gamma = round(
                        calc_gamma(underlying_price, strike, T, RISK_FREE_RATE, iv), 6
                    )
                    theta = round(
                        calc_theta(underlying_price, strike, T, RISK_FREE_RATE, iv, option_type),
                        4,
                    )
                    vega = round(
                        calc_vega(underlying_price, strike, T, RISK_FREE_RATE, iv), 4
                    )
                    iv = round(iv, 4)
        except Exception as exc:
            logger.debug(f"Greeks computation failed for {contract_ticker}: {exc}")

    return {
        "contract_ticker": contract_ticker,
        "underlying": contract_ticker.split(":")[1][:-15] if ":" in contract_ticker else None,
        "strike": float(strike),
        "expiry_date": expiry,
        "option_type": option_type,
        "underlying_price": underlying_price,
        "bid": bid,
        "ask": ask,
        "bid_size": quote.get("bid_size"),
        "ask_size": quote.get("ask_size"),
        "midpoint": midpoint,
        "last_trade_price": trade.get("price"),
        "last_trade_ts_ns": trade.get("sip_timestamp"),
        "day_open": day.get("open"),
        "day_high": day.get("high"),
        "day_low": day.get("low"),
        "day_close": day.get("close"),
        "day_volume": day.get("volume"),
        "day_vwap": day.get("vwap"),
        "open_interest": result.get("open_interest"),
        "implied_volatility": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


async def _persist_rows(ticker: str, rows: list[dict]) -> int:
    """Write option snapshots + stock tick to PostgreSQL. Returns snapshot count."""
    if not rows:
        return 0
    try:
        from options_owl.db import postgres as pg
        if not pg.is_connected():
            logger.warning(f"PG not connected — dropping {len(rows)} rows for {ticker}")
            return 0

        # Write option ticks
        option_ticks = [
            {
                "ticker": ticker,
                "option_type": r["option_type"],
                "strike": r["strike"],
                "expiry_date": r["expiry_date"],
                "bid": r["bid"],
                "ask": r["ask"],
                "bid_size": r.get("bid_size"),
                "ask_size": r.get("ask_size"),
                "mid": r["midpoint"],
                "last": r["last_trade_price"],
                "volume": r["day_volume"],
                "open_interest": r["open_interest"],
                "iv": r["implied_volatility"],
                "delta": r["delta"],
                "gamma": r["gamma"],
                "theta": r["theta"],
                "vega": r["vega"],
                "underlying_price": r["underlying_price"],
            }
            for r in rows
        ]
        await pg.write_option_ticks_batch(option_ticks)

        # Write stock tick (one per ticker per cycle, with real bid/ask/volume)
        _quote = _price_cache.get(ticker)
        _q = _quote[0] if _quote else {}
        stock_price = rows[0]["underlying_price"]
        await pg.write_stock_tick(
            ticker=ticker,
            price=stock_price,
            bid=_q.get("bid", 0),
            ask=_q.get("ask", 0),
            volume=_q.get("volume", 0),
            vwap=_q.get("vwap", 0),
        )

        # Publish to Redis for real-time delivery to all agents
        try:
            from options_owl.db import redis_client
            if redis_client.is_connected():
                # Stock price
                await redis_client.publish_price(ticker, stock_price)
                # Full option snapshots (greeks + premium for ML models)
                for r in rows:
                    contract_key = (
                        f"{ticker}:{r['option_type']}:{r['strike']}:"
                        f"{r['expiry_date']}"
                    )
                    # Legacy premium-only (backward compat for position monitor)
                    await redis_client.publish_option_premium(
                        contract_key, r["bid"], r["ask"], r["midpoint"],
                    )
                    # Full snapshot for ML scan loop
                    await redis_client.publish_option_snapshot(
                        contract_key,
                        {
                            "bid": r["bid"],
                            "ask": r["ask"],
                            "mid": r["midpoint"],
                            "iv": r["implied_volatility"],
                            "delta": r["delta"],
                            "gamma": r["gamma"],
                            "theta": r["theta"],
                            "vega": r["vega"],
                            "volume": r["day_volume"],
                            "open_interest": r["open_interest"],
                            "underlying_price": r["underlying_price"],
                            "bid_size": r["bid_size"],
                            "ask_size": r["ask_size"],
                            "strike": r["strike"],
                            "expiry_date": r["expiry_date"],
                            "option_type": r["option_type"],
                        },
                    )
        except Exception:
            pass  # fire-and-forget

        return len(rows)
    except Exception as exc:
        logger.error(f"PG persist failed for {ticker} ({len(rows)} rows): {exc}")
        return 0


# ---------------------------------------------------------------------------
# Per-ticker harvest step
# ---------------------------------------------------------------------------

# Tickers that are index-only (no standard equity options chain on Polygon)
INDEX_ONLY_TICKERS = {"VIX"}


async def _persist_stock_tick_only(ticker: str, quote: dict) -> None:
    """Write just a stock tick to PG for index-only tickers (no option chain)."""
    try:
        from options_owl.db import postgres as pg
        if not pg.is_connected():
            return
        await pg.write_stock_tick(
            ticker=ticker,
            price=quote["price"],
            bid=quote.get("bid", 0),
            ask=quote.get("ask", 0),
            volume=quote.get("volume", 0),
            vwap=quote.get("vwap", 0),
        )
    except Exception as exc:
        logger.debug(f"PG stock tick write failed for {ticker}: {exc}")


async def _harvest_ticker(
    client: httpx.AsyncClient,
    ticker: str,
    semaphore: asyncio.Semaphore,
    candle_collector: CandleCollector | None = None,
) -> int:
    """Fetch + parse + persist one ticker. Returns snapshot count."""
    async with semaphore:
        quote = await asyncio.to_thread(_get_underlying_quote, ticker)
        if quote is None:
            logger.warning(f"{ticker}: no underlying price, skipping")
            return 0
        underlying_price = quote["price"]

        # Feed price to candle collector for 5m bar building
        if candle_collector is not None and underlying_price > 0:
            candle_collector.record_price(ticker, underlying_price)

        # Index-only tickers: write stock tick but skip option chain fetch
        if ticker in INDEX_ONLY_TICKERS:
            await _persist_stock_tick_only(ticker, quote)
            logger.debug(f"{ticker}: ${underlying_price:.2f} (index-only, no chain)")
            return 0

        strike_low = round(underlying_price * (1 - 0.02 * STRIKE_WINDOW), 2)
        strike_high = round(underlying_price * (1 + 0.02 * STRIKE_WINDOW), 2)
        today = date.today()
        expiry_lo = today.isoformat()
        expiry_hi = (today.fromordinal(today.toordinal() + MAX_DTE)).isoformat()

        raw = await _fetch_chain(
            client, ticker, strike_low, strike_high, expiry_lo, expiry_hi
        )
        if not raw:
            return 0

        rows = [p for p in (_parse_contract(r, underlying_price) for r in raw) if p]
        if not rows:
            return 0

        written = await _persist_rows(ticker, rows)
        logger.info(
            f"{ticker}: {written} contracts @ ${underlying_price:.2f} "
            f"(strikes {strike_low:.0f}-{strike_high:.0f}, exp≤{expiry_hi})"
        )
        return written


# ---------------------------------------------------------------------------
# WS Health Watchdog + Smoke Test
# ---------------------------------------------------------------------------

_WS_HEALTH_KEY = "owl:ws_health"


async def _ws_health_watchdog(
    candle_collector: CandleCollector,
    flow_collector: FlowCollector | None,
) -> None:
    """Check WS health every 30s during market hours. Publish status to Redis.

    If either WS is disconnected or stale (no messages for 60s), logs CRITICAL
    and publishes an unhealthy status to Redis so all bots can see it.
    """
    import json

    while True:
        try:
            await asyncio.sleep(30)
            if not _is_market_hours():
                continue

            issues: list[str] = []

            if not candle_collector.ws_connected:
                issues.append("candle_ws DISCONNECTED")
            elif candle_collector.last_msg_age > 60:
                issues.append(
                    f"candle_ws stale ({candle_collector.last_msg_age:.0f}s)"
                )

            if flow_collector is not None:
                if not flow_collector.ws_connected:
                    issues.append("flow_ws DISCONNECTED")
                elif flow_collector.last_msg_age > 60:
                    issues.append(
                        f"flow_ws stale ({flow_collector.last_msg_age:.0f}s)"
                    )

            health = {
                "healthy": len(issues) == 0,
                "issues": issues,
                "candle_ws": candle_collector.ws_connected,
                "flow_ws": flow_collector.ws_connected if flow_collector else False,
                "candle_age": round(candle_collector.last_msg_age, 1),
                "flow_age": round(flow_collector.last_msg_age, 1) if flow_collector else 0,
                "t": time.time(),
            }

            if issues:
                logger.critical(f"WS HEALTH ALERT: {', '.join(issues)}")

            # Publish to Redis so all bots can check WS health
            try:
                from options_owl.db import redis_client
                if redis_client.is_connected():
                    await redis_client._redis.set(  # type: ignore[union-attr]
                        _WS_HEALTH_KEY,
                        json.dumps(health),
                        ex=120,
                    )
            except Exception:
                pass

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"WS watchdog error: {exc}")


async def _ws_smoke_test(
    candle_collector: CandleCollector,
    flow_collector: FlowCollector | None,
    timeout: int = 60,
    retry_interval: int = 30,
) -> None:
    """Verify both WS connections deliver data before entering the poll loop.

    Retries every retry_interval seconds. During non-market hours, skips
    immediately (WS won't have data until market opens). Never blocks
    forever — logs progress and continues after data flows.
    """
    if not _is_market_hours():
        logger.info(
            "WS smoke test: outside market hours — skipping "
            "(WS will connect when market opens)"
        )
        return

    logger.info(f"WS smoke test: waiting up to {timeout}s for live WS data...")
    start = time.time()

    while time.time() - start < timeout:
        candle_ok = candle_collector.ws_connected and candle_collector.last_msg_age < 30
        flow_ok = (
            flow_collector is None
            or (flow_collector.ws_connected and flow_collector.last_msg_age < 30)
        )

        if candle_ok and flow_ok:
            elapsed = time.time() - start
            logger.info(
                f"WS smoke test PASSED in {elapsed:.0f}s — "
                f"candle_ws=connected (age={candle_collector.last_msg_age:.0f}s), "
                f"flow_ws={'connected' if flow_collector else 'N/A'}"
            )
            return

        status = []
        if not candle_ok:
            status.append(
                f"candle_ws={'connected' if candle_collector.ws_connected else 'connecting'}"
                f" (age={candle_collector.last_msg_age:.0f}s)"
            )
        if flow_collector and not flow_ok:
            status.append(
                f"flow_ws={'connected' if flow_collector.ws_connected else 'connecting'}"
                f" (age={flow_collector.last_msg_age:.0f}s)"
            )
        elapsed = time.time() - start
        logger.info(
            f"WS smoke test: waiting... ({elapsed:.0f}s/{timeout}s) — "
            f"{', '.join(status)}"
        )
        await asyncio.sleep(5)

    # Timed out — log warning but don't block. The WS reconnect logic will
    # keep trying, and the watchdog will alert if it stays down.
    logger.warning(
        f"WS smoke test: timed out after {timeout}s — proceeding with poll loop. "
        f"WS reconnect will continue in background."
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_harvester() -> None:
    """Main async loop: poll every POLL_INTERVAL seconds during market hours."""
    if not POLYGON_API_KEY:
        logger.critical("POLYGON_API_KEY is not set — harvester cannot run")
        sys.exit(2)

    # Initialize PostgreSQL — required for all data storage
    try:
        from options_owl.db import postgres as pg
        await pg.init_pool(os.environ.get("DATABASE_URL"))
        logger.info("PostgreSQL connected — primary data store")
    except Exception as exc:
        logger.critical(f"PostgreSQL init failed — harvester cannot run without PG: {exc}")
        sys.exit(2)

    # Initialize Redis — streams real-time data to all trading bots
    redis_url = os.environ.get("REDIS_URL", "")
    if os.environ.get("ENABLE_REDIS", "").lower() == "true" and redis_url:
        try:
            from options_owl.db import redis_client
            await redis_client.init_redis(redis_url)
        except Exception as exc:
            logger.warning(f"Redis init failed (non-fatal): {exc}")

    # Initialize candle collector for real-time WS minute bars → 5m/15m/1h candles.
    # Connects to wss://socket.polygon.io/stocks — SEPARATE endpoint from
    # the options WS (wss://socket.polygon.io/options).
    candle_collector = CandleCollector(UNIVERSE)
    await candle_collector.start_ws(POLYGON_API_KEY)

    # Centralized flow capture — uses dedicated key for /options WS.
    # All trading bots have ENABLE_POLYGON_WS=false, so this key's slot is free.
    # Flow data is written to PG + published to Redis for all agents.
    flow_api_key = os.environ.get("FLOW_POLYGON_API_KEY", "")
    flow_collector = None
    if flow_api_key:
        # Pass price_getter so FlowCollector can compute greeks from WS quotes
        def _get_price(ticker: str) -> float | None:
            cached = _price_cache.get(ticker)
            if cached and time.time() - cached[1] < 60:
                return cached[0]["price"]
            return None

        flow_collector = FlowCollector(UNIVERSE, price_getter=_get_price)
        await flow_collector.start_ws(flow_api_key)

    logger.info(
        f"Harvester started — universe={UNIVERSE}, poll={POLL_INTERVAL}s, "
        f"DTE≤{MAX_DTE}, strikes=ATM±{STRIKE_WINDOW}, "
        f"candle_ws={'on' if candle_collector.ws_connected else 'starting'}, "
        f"flow_ws={'on' if (flow_collector and flow_api_key and flow_collector.ws_connected) else 'off'}"
    )

    # WS Connection Smoke Test: verify both WS connections deliver data
    # within 60s before entering the poll loop. Retry indefinitely during
    # market hours — early boot before market open is expected.
    await _ws_smoke_test(candle_collector, flow_collector)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        logger.info("Harvester received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            signal.signal(sig, _shutdown)

    # Launch WS health watchdog — monitors WS connections during market hours
    watchdog_task = asyncio.create_task(
        _ws_health_watchdog(candle_collector, flow_collector)
    )

    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            write_heartbeat()

            if MARKET_HOURS_ONLY and not _is_market_hours():
                logger.debug("Outside market hours — sleeping 5 min")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    pass
                continue

            t0 = time.time()
            try:
                # Priority tickers first (actively traded) — ensures Redis
                # has fresh premiums for position monitoring even if later
                # tickers fail due to rate limits or timeouts.
                _PRIORITY = {
                    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "META",
                    "AAPL", "AMZN", "GOOGL", "AMD", "MSTR", "PLTR",
                }
                priority = [t for t in UNIVERSE if t in _PRIORITY]
                secondary = [t for t in UNIVERSE if t not in _PRIORITY]

                results_p = await asyncio.gather(
                    *[
                        _harvest_ticker(client, t, semaphore, candle_collector)
                        for t in priority
                    ],
                    return_exceptions=True,
                )
                results_s = await asyncio.gather(
                    *[
                        _harvest_ticker(client, t, semaphore, candle_collector)
                        for t in secondary
                    ],
                    return_exceptions=True,
                )
                results = list(results_p) + list(results_s)
                total = sum(r for r in results if isinstance(r, int))
                errors = [r for r in results if isinstance(r, Exception)]

                # Flush completed 5m candles + option flow to DB + Redis
                candle_count = await candle_collector.flush()
                flow_count = 0
                if flow_collector:
                    flow_count = await flow_collector.flush()

                elapsed = time.time() - t0

                # WS health status
                candle_ws_status = "connected" if candle_collector.ws_connected else "DISCONNECTED"
                candle_age = candle_collector.last_msg_age
                flow_ws_status = ""
                if flow_collector:
                    fc_connected = "connected" if flow_collector.ws_connected else "DISCONNECTED"
                    fc_age = flow_collector.last_msg_age
                    flow_ws_status = f", flow_ws={fc_connected} (last_msg={fc_age:.0f}s ago)"

                logger.info(
                    f"Poll complete: {total} snapshots across {len(UNIVERSE)} "
                    f"tickers in {elapsed:.1f}s ({len(errors)} errors"
                    f"{f', {candle_count} candle bars' if candle_count else ''}"
                    f"{f', {flow_count} flow trades' if flow_count else ''}) "
                    f"| candle_ws={candle_ws_status} (last_msg={candle_age:.0f}s ago)"
                    f"{flow_ws_status}"
                )

                # Publish ALL timeframe candle bars for ALL tickers to Redis
                # so bots read shared data instead of per-bot Polygon REST
                # calls (avoids 429 rate limits). Timeframes: 5m,15m,30m,1h,4h.
                try:
                    from options_owl.db import redis_client
                    if redis_client.is_connected():
                        _TF_MINUTES = {
                            "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
                        }
                        pub_count = 0
                        for ticker in UNIVERSE:
                            minute_bars = candle_collector._minute_bars.get(ticker)
                            if not minute_bars or len(minute_bars) < 2:
                                continue
                            for tf_label, tf_min in _TF_MINUTES.items():
                                bucket_ms = tf_min * 60 * 1000
                                bars_json: list[dict] = []
                                current_bucket = None
                                o = h = lo = c = vol = vw = 0.0
                                for mb in minute_bars:
                                    bucket = (mb.ts_ms // bucket_ms) * bucket_ms
                                    if current_bucket is not None and bucket != current_bucket:
                                        bars_json.append({
                                            "t": current_bucket, "o": o, "h": h,
                                            "l": lo, "c": c, "v": vol, "vw": vw,
                                        })
                                        current_bucket = None
                                    if current_bucket is None:
                                        current_bucket = bucket
                                        o, h, lo, vol, vw = (
                                            mb.open, mb.high, mb.low, 0.0, 0.0,
                                        )
                                    h = max(h, mb.high)
                                    lo = min(lo, mb.low)
                                    c = mb.close
                                    vol += mb.volume
                                    if mb.vwap > 0:
                                        vw = mb.vwap
                                if current_bucket is not None:
                                    bars_json.append({
                                        "t": current_bucket, "o": o, "h": h,
                                        "l": lo, "c": c, "v": vol, "vw": vw,
                                    })
                                if bars_json:
                                    await redis_client.publish_candle_bars(
                                        ticker, tf_label, bars_json,
                                    )
                                    pub_count += 1

                        if pub_count:
                            logger.debug(
                                f"Redis candle publish: {pub_count} ticker/TF combos"
                            )

                        # Also publish SPY change-from-open for the PUT direction gate
                        spy_bars = candle_collector._minute_bars.get("SPY")
                        if spy_bars and len(spy_bars) >= 2:
                            spy_open = spy_bars[0].open
                            spy_last = spy_bars[-1].close
                            if spy_open > 0:
                                spy_change = ((spy_last - spy_open) / spy_open) * 100
                                await redis_client.publish_spy_change(
                                    spy_change, spy_open, spy_last,
                                )
                except Exception:
                    pass  # non-critical — bots fall back to Polygon REST
                if errors:
                    for exc in errors[:3]:
                        logger.warning(f"  harvest error: {exc}")
            except Exception as exc:
                logger.error(f"Harvest cycle failed: {exc}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass

    watchdog_task.cancel()
    try:
        await watchdog_task
    except asyncio.CancelledError:
        pass
    await candle_collector.stop_ws()
    if flow_collector:
        await flow_collector.stop_ws()
    logger.info("Harvester shut down cleanly")


def main() -> None:
    configure_logging(verbose=False)
    # Drop a heartbeat immediately so Docker healthcheck passes during startup
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_heartbeat()
    try:
        asyncio.run(run_harvester())
    except KeyboardInterrupt:
        logger.info("Harvester interrupted")


if __name__ == "__main__":
    main()
