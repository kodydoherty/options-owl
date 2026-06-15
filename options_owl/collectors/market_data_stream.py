"""Real-time market data feed with WebSocket streaming and polling fallback.

Supports multiple data feed providers:
- **yfinance** (default): Polls prices every N seconds. No API key needed.
- **polygon**: Streams real-time trades via Polygon.io WebSocket. Requires API key.
- **webull**: Reserved for future Webull streaming integration.

Integration with position_monitor.py
-------------------------------------
To replace the current yfinance polling in ``position_monitor.py``:

1. Instantiate ``MarketDataStream`` with the app's ``Settings`` object::

       from options_owl.collectors.market_data_stream import MarketDataStream
       stream = MarketDataStream(settings)

2. Start the stream before the monitor loop (e.g. in ``on_ready``)::

       await stream.start()

3. Replace ``_fetch_price_async(ticker)`` calls with::

       price = await stream.get_price(ticker)

4. Replace ``_lookup_premium_from_chain(...)`` calls with::

       premium = await stream.get_option_premium(ticker, strike, expiry, option_type)

5. Subscribe to tickers as positions are opened::

       await stream.subscribe("AAPL")

6. Unsubscribe when positions are fully closed::

       await stream.unsubscribe("AAPL")

7. Stop the stream on shutdown::

       await stream.stop()
"""

from __future__ import annotations

import asyncio
import json as _json
import math
import os
import time as _time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import yfinance as yf
from loguru import logger

from options_owl.config.settings import Settings


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class DataFeedProvider(str, Enum):
    """Supported market data feed providers."""

    YFINANCE = "yfinance"
    POLYGON = "polygon"
    WEBULL = "webull"


# ---------------------------------------------------------------------------
# Price cache entry
# ---------------------------------------------------------------------------

PriceCacheEntry = tuple[float, float]  # (price, unix_timestamp)


# ---------------------------------------------------------------------------
# MarketDataStream
# ---------------------------------------------------------------------------


class MarketDataStream:
    """Unified market data interface supporting polling and streaming modes.

    Parameters
    ----------
    settings : Settings
        Application settings. Uses ``DATA_FEED_PROVIDER``,
        ``DATA_FEED_POLL_INTERVAL``, and ``POLYGON_API_KEY``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = DataFeedProvider(settings.DATA_FEED_PROVIDER.lower())
        self.poll_interval: int = settings.DATA_FEED_POLL_INTERVAL

        # Price cache: ticker -> (price, unix_timestamp)
        self._price_cache: dict[str, PriceCacheEntry] = {}

        # Option premium cache: contract_ticker -> (premium, unix_timestamp)
        # e.g. "O:SPY260414C00691000" -> (1.25, 1776179389.0)
        self._option_cache: dict[str, PriceCacheEntry] = {}

        # Map from (ticker, strike, expiry, option_type) -> contract_ticker
        # for reverse lookup when get_option_premium is called
        self._option_contract_map: dict[tuple[str, float, str, str], str] = {}

        # Active subscriptions
        self._subscriptions: set[str] = set()
        self._option_subscriptions: set[str] = set()  # contract tickers

        # Minute-bar buffer: ticker -> list of (timestamp_ms, o, h, l, c, v, vw)
        # Populated from AM events on the Polygon WS.  CandleCache can read
        # these to build higher-TF candles without extra REST calls.
        from collections import deque
        self._minute_bars: dict[str, deque[tuple[float, float, float, float, float, float, float]]] = {}
        self._MAX_MINUTE_BARS = 500  # ~8 hours of 1-min bars per ticker

        # Lifecycle
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._ws: Any = None  # websockets connection object

        # Reconnection backoff state
        self._reconnect_attempts = 0
        self._max_reconnect_delay = 300  # 5 minutes cap

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the data feed. Launches a background task for the chosen provider."""
        if self._running:
            logger.warning("MarketDataStream already running")
            return

        self._running = True
        logger.info(f"Starting MarketDataStream (provider={self.provider.value})")

        # Initialize tick logger for 1s option premium recording
        try:
            from options_owl.collectors.tick_logger import init_tick_logger
            journal_dir = os.path.dirname(self.settings.DB_PATH) or "journal"
            init_tick_logger(journal_dir)
        except Exception as exc:
            logger.warning(f"Tick logger init failed (non-fatal): {exc}")

        if self.provider == DataFeedProvider.POLYGON:
            if not self.settings.POLYGON_API_KEY:
                logger.error(
                    "POLYGON_API_KEY not set — falling back to yfinance polling"
                )
                self.provider = DataFeedProvider.YFINANCE
                self._poll_task = asyncio.create_task(self._polling_loop())
            elif not getattr(self.settings, "ENABLE_POLYGON_WS", True):
                logger.info(
                    "ENABLE_POLYGON_WS=false — using yfinance polling "
                    "(Polygon REST still available for option premiums)"
                )
                self.provider = DataFeedProvider.YFINANCE
                self._poll_task = asyncio.create_task(self._polling_loop())
            else:
                self._ws_task = asyncio.create_task(self._polygon_ws_loop())
        elif self.provider == DataFeedProvider.WEBULL:
            logger.warning(
                "Webull streaming not yet implemented — falling back to yfinance polling"
            )
            self.provider = DataFeedProvider.YFINANCE
            self._poll_task = asyncio.create_task(self._polling_loop())
        else:
            # yfinance polling
            self._poll_task = asyncio.create_task(self._polling_loop())

    async def stop(self) -> None:
        """Gracefully stop the data feed and release resources."""
        logger.info("Stopping MarketDataStream")
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._price_cache.clear()
        self._option_cache.clear()
        self._option_contract_map.clear()
        self._option_subscriptions.clear()
        self._reconnect_attempts = 0
        logger.info("MarketDataStream stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(self, ticker: str) -> None:
        """Subscribe to price updates for a ticker.

        For Polygon, this sends a WebSocket subscribe message immediately.
        For yfinance polling, the ticker is added to the next poll cycle.
        """
        ticker = ticker.upper()
        if ticker in self._subscriptions:
            return

        self._subscriptions.add(ticker)
        logger.info(f"Subscribed to {ticker} (total: {len(self._subscriptions)})")

        # If Polygon WS is live, subscribe on the wire immediately
        if self.provider == DataFeedProvider.POLYGON and self._ws is not None:
            await self._polygon_send_subscribe([ticker])

    async def unsubscribe(self, ticker: str) -> None:
        """Unsubscribe from price updates for a ticker."""
        ticker = ticker.upper()
        self._subscriptions.discard(ticker)
        self._price_cache.pop(ticker, None)
        logger.info(f"Unsubscribed from {ticker} (total: {len(self._subscriptions)})")

        if self.provider == DataFeedProvider.POLYGON and self._ws is not None:
            await self._polygon_send_unsubscribe([ticker])

    def get_minute_bars(self, ticker: str) -> list[tuple[float, float, float, float, float, float, float]]:
        """Return buffered 1-minute OHLCV bars for a ticker from the WS feed.

        Each tuple: (timestamp_ms, open, high, low, close, volume, vwap).
        Returns empty list if no bars available.
        """
        bars = self._minute_bars.get(ticker.upper())
        return list(bars) if bars else []

    @staticmethod
    def build_option_contract_ticker(
        ticker: str, strike: float, expiry: str, option_type: str,
    ) -> str:
        """Build a Polygon option contract ticker.

        Format: O:SPY260414C00691000
        """
        opt_char = "C" if option_type.lower() == "call" else "P"
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        expiry_str = expiry_dt.strftime("%y%m%d")
        strike_int = int(strike * 1000)
        return f"O:{ticker.upper()}{expiry_str}{opt_char}{strike_int:08d}"

    async def subscribe_option(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
    ) -> None:
        """Subscribe to real-time quotes for a specific option contract.

        Subscribes to both Q (quotes) and T (trades) channels for the contract
        on the Polygon options WebSocket.
        """
        contract = self.build_option_contract_ticker(ticker, strike, expiry, option_type)
        key = (ticker.upper(), strike, expiry, option_type.lower())
        self._option_contract_map[key] = contract

        if contract in self._option_subscriptions:
            return

        self._option_subscriptions.add(contract)
        logger.info(f"Subscribed to option quotes: {contract}")

        if self.provider == DataFeedProvider.POLYGON and self._ws is not None:
            await self._polygon_send_option_subscribe([contract])

    async def unsubscribe_option(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
    ) -> None:
        """Unsubscribe from a specific option contract's quotes."""
        contract = self.build_option_contract_ticker(ticker, strike, expiry, option_type)
        key = (ticker.upper(), strike, expiry, option_type.lower())
        self._option_contract_map.pop(key, None)
        self._option_subscriptions.discard(contract)
        self._option_cache.pop(contract, None)
        logger.info(f"Unsubscribed from option quotes: {contract}")

        if self.provider == DataFeedProvider.POLYGON and self._ws is not None:
            await self._polygon_send_option_unsubscribe([contract])

    async def get_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker.

        Returns the cached price if fresh (within 2x poll interval),
        otherwise fetches synchronously via yfinance as a fallback.
        """
        ticker = ticker.upper()

        # Ensure we're subscribed
        if ticker not in self._subscriptions:
            await self.subscribe(ticker)

        # Check local cache first
        entry = self._price_cache.get(ticker)
        if entry is not None:
            price, ts = entry
            staleness = _time.time() - ts
            max_age = self.poll_interval * 2 if self.provider == DataFeedProvider.YFINANCE else 30
            if staleness <= max_age:
                return price
            logger.debug(f"Cache stale for {ticker} ({staleness:.0f}s old)")

        # Redis cache (centralized harvester publishes stock prices)
        try:
            from options_owl.db import redis_client
            if redis_client.is_connected():
                result = await redis_client.get_price(ticker, max_age=60)
                if result is not None:
                    redis_price, _age = result
                    self._update_cache(ticker, redis_price)
                    return redis_price
        except Exception:
            pass

        # Fallback: synchronous yfinance fetch
        price = await self._yfinance_fetch_price(ticker)
        if price is not None:
            self._update_cache(ticker, price)
        return price

    async def get_option_premium(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
    ) -> float | None:
        """Fetch the current premium for an option contract.

        Parameters
        ----------
        ticker : str
            Underlying ticker symbol (e.g. "SPY").
        strike : float
            Option strike price.
        expiry : str
            Expiration date as "YYYY-MM-DD".
        option_type : str
            "call" or "put".

        Returns
        -------
        float | None
            Bid/ask midpoint premium, or None if unavailable.

        For Polygon streaming, this queries the REST API as a supplement
        since the WS stream provides underlying prices. For yfinance,
        it uses the option chain endpoint.
        """
        ticker = ticker.upper()

        # 0. Redis cache (real-time from centralized harvester)
        try:
            from options_owl.db import redis_client
            if redis_client.is_connected():
                contract_key = f"{ticker}:{option_type.lower()}:{strike}:{expiry}"
                data = await redis_client.get_option_premium(contract_key)
                if data and (_time.time() - data.get("t", 0)) < 120:
                    return data["mid"]
        except Exception:
            pass

        # 1. Check WS option cache (real-time when Polygon WS is enabled)
        key = (ticker, strike, expiry, option_type.lower())
        contract = self._option_contract_map.get(key)
        if contract:
            entry = self._option_cache.get(contract)
            if entry is not None:
                premium, ts = entry
                age = _time.time() - ts
                if age <= 30:  # fresh within 30s
                    return premium
                logger.debug(f"Option WS cache stale for {contract} ({age:.0f}s)")

        # 2. Polygon REST snapshot fallback
        if self.settings.POLYGON_API_KEY:
            premium = await self._polygon_rest_option_premium(
                ticker, strike, expiry, option_type
            )
            if premium is not None:
                return premium

        # 3. yfinance option chain fallback
        return await self._yfinance_fetch_option_premium(
            ticker, strike, expiry, option_type
        )

    # ------------------------------------------------------------------
    # yfinance polling
    # ------------------------------------------------------------------

    async def _polling_loop(self) -> None:
        """Poll yfinance for all subscribed tickers at a fixed interval."""
        logger.info(f"yfinance polling loop started (interval={self.poll_interval}s)")
        try:
            while self._running:
                tickers = list(self._subscriptions)
                if tickers:
                    await self._poll_yfinance_batch(tickers)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.debug("yfinance polling loop cancelled")
            raise

    async def _poll_yfinance_batch(self, tickers: list[str]) -> None:
        """Fetch prices for a batch of tickers via yfinance."""
        try:
            prices = await asyncio.to_thread(self._yfinance_batch_sync, tickers)
            for ticker, price in prices.items():
                if price is not None:
                    self._update_cache(ticker, price)
        except Exception:
            logger.exception("Error polling yfinance batch")

    @staticmethod
    def _yfinance_batch_sync(tickers: list[str]) -> dict[str, float | None]:
        """Synchronous batch price fetch using yfinance (runs in thread)."""
        results: dict[str, float | None] = {}
        for ticker in tickers:
            try:
                tk = yf.Ticker(ticker)
                price = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
                if price and price > 0:
                    results[ticker] = float(price)
                else:
                    hist = tk.history(period="1d", interval="1m")
                    if not hist.empty:
                        results[ticker] = float(hist["Close"].iloc[-1])
                    else:
                        results[ticker] = None
            except Exception as e:
                logger.warning(f"yfinance fetch failed for {ticker}: {e}")
                results[ticker] = None
        return results

    @staticmethod
    async def _yfinance_fetch_price(ticker: str) -> float | None:
        """Fetch a single ticker price via yfinance (async wrapper)."""
        def _sync() -> float | None:
            try:
                tk = yf.Ticker(ticker)
                price = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
                if price and price > 0:
                    return float(price)
                hist = tk.history(period="1d", interval="1m")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
                return None
            except Exception as e:
                logger.warning(f"yfinance single fetch failed for {ticker}: {e}")
                return None

        return await asyncio.to_thread(_sync)

    @staticmethod
    async def _yfinance_fetch_option_premium(
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
    ) -> float | None:
        """Fetch option premium from yfinance option chain (async wrapper)."""
        def _sync() -> float | None:
            try:
                tk = yf.Ticker(ticker)
                chain = tk.option_chain(expiry)
                df = chain.calls if option_type == "call" else chain.puts
                if df is None or df.empty:
                    return None

                matches = df[abs(df["strike"] - strike) < 0.01]
                if matches.empty:
                    return None

                row = matches.iloc[0]
                bid = row.get("bid")
                ask = row.get("ask")

                if (
                    bid is not None
                    and ask is not None
                    and not (isinstance(bid, float) and math.isnan(bid))
                    and not (isinstance(ask, float) and math.isnan(ask))
                    and bid > 0
                    and ask > 0
                ):
                    return round((bid + ask) / 2.0, 2)

                last = row.get("lastPrice")
                if (
                    last is not None
                    and not (isinstance(last, float) and math.isnan(last))
                    and last > 0
                ):
                    return round(float(last), 2)

                return None
            except Exception as e:
                logger.warning(f"yfinance option chain failed for {ticker}: {e}")
                return None

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Polygon.io WebSocket streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _is_market_hours() -> bool:
        """Check if US equity market is open (9:25 AM - 4:05 PM ET, weekdays).

        Uses a 5-min buffer on each side to ensure we're connected before open
        and stay connected through close.
        """
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except ImportError:
            _et = timezone(timedelta(hours=-5))
        et_now = datetime.now(_et)
        if et_now.weekday() >= 5:  # Saturday/Sunday
            return False
        hour, minute = et_now.hour, et_now.minute
        t = hour * 60 + minute
        return 9 * 60 + 25 <= t <= 16 * 60 + 5  # 9:25 AM - 4:05 PM ET

    @staticmethod
    def _seconds_until_market_open() -> float:
        """Seconds until next market open (9:25 AM ET). Returns 0 if market is open."""
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except ImportError:
            _et = timezone(timedelta(hours=-5))
        et_now = datetime.now(_et)
        open_time = et_now.replace(hour=9, minute=25, second=0, microsecond=0)

        # If weekend, advance to Monday
        days_ahead = 0
        weekday = et_now.weekday()
        if weekday == 5:  # Saturday
            days_ahead = 2
        elif weekday == 6:  # Sunday
            days_ahead = 1

        if days_ahead == 0 and et_now >= open_time.replace(hour=16, minute=5):
            # Past close today, next open is tomorrow (or Monday)
            days_ahead = 1
            if et_now.weekday() == 4:  # Friday after close
                days_ahead = 3

        target = open_time + timedelta(days=days_ahead)
        if target <= et_now:
            return 0
        return (target - et_now).total_seconds()

    async def _polygon_ws_loop(self) -> None:
        """Connect to Polygon.io WebSocket and process messages with reconnection."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package not installed — falling back to yfinance polling. "
                "Install with: pip install websockets"
            )
            self.provider = DataFeedProvider.YFINANCE
            self._poll_task = asyncio.create_task(self._polling_loop())
            return

        url = "wss://socket.polygon.io/options"

        while self._running:
            # --- Wait for market hours to avoid hammering WS post-market ---
            if not self._is_market_hours():
                wait_secs = self._seconds_until_market_open()
                wait_mins = wait_secs / 60
                if wait_mins > 60:
                    logger.info(
                        f"Polygon WS: market closed — sleeping {wait_secs/3600:.1f}h until next open"
                    )
                else:
                    logger.info(
                        f"Polygon WS: market closed — sleeping {wait_mins:.0f}m until next open"
                    )
                # Sleep in 60s chunks so we can respond to shutdown
                while wait_secs > 0 and self._running:
                    chunk = min(wait_secs, 60)
                    await asyncio.sleep(chunk)
                    wait_secs -= chunk
                if not self._running:
                    break
                continue

            try:
                # Ensure any stale connection is fully closed before reconnecting
                # to avoid hitting Polygon's max_connections limit.
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

                logger.info(f"Connecting to Polygon.io WebSocket at {url}")
                async with websockets.connect(url, ping_interval=30) as ws:
                    self._ws = ws

                    # Wait for connection confirmation before authenticating
                    conn_resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    conn_data = _json_loads(conn_resp)

                    # Check for max_connections error
                    if isinstance(conn_data, list):
                        for item in conn_data:
                            if item.get("status") == "max_connections":
                                logger.warning(
                                    "Polygon WS: max connections reached — "
                                    "waiting 30s for stale connections to expire"
                                )
                                await asyncio.sleep(30)
                                continue

                    logger.debug(f"Polygon WS connection msg: {conn_data}")

                    # Authenticate
                    await ws.send(
                        _json_dumps({"action": "auth", "params": self.settings.POLYGON_API_KEY})
                    )
                    auth_resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    auth_data = _json_loads(auth_resp)

                    if not _polygon_auth_ok(auth_data):
                        logger.error(f"Polygon auth failed: {auth_data}")
                        await self._backoff_sleep()
                        continue

                    logger.info("Polygon.io WebSocket authenticated")

                    # Subscribe to SPY aggregate trades as a keepalive —
                    # Polygon kicks idle connections with no subscriptions after ~10s.
                    # AM.* gets aggregate-per-minute for all option trades (lightweight).
                    await ws.send(
                        _json_dumps({"action": "subscribe", "params": "AM.*"})
                    )
                    logger.info("Polygon WS: subscribed to AM.* (keepalive)")

                    # Subscribe to all current underlying tickers
                    if self._subscriptions:
                        await self._polygon_send_subscribe(list(self._subscriptions))

                    # Re-subscribe to any active option contracts
                    if self._option_subscriptions:
                        await self._polygon_send_option_subscribe(
                            list(self._option_subscriptions)
                        )

                    # Message loop — reset backoff only after first real data
                    got_data = False
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        if not got_data:
                            got_data = True
                            self._reconnect_attempts = 0
                        self._process_polygon_message(raw_msg)

            except asyncio.CancelledError:
                logger.debug("Polygon WS loop cancelled")
                raise
            except Exception as e:
                # Explicitly close to free the connection slot on Polygon's side
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                self._ws = None
                if not self._running:
                    break
                logger.warning(f"Polygon WebSocket error: {e}")
                await self._backoff_sleep()

        self._ws = None
        logger.info("Polygon WS loop exited")

    async def _polygon_send_subscribe(self, tickers: list[str]) -> None:
        """Send subscribe message for tickers on the Polygon WebSocket."""
        if self._ws is None:
            return
        # Polygon options format: T.O:TICKER for trades, Q.O:TICKER for quotes
        # For underlying equity prices: T.TICKER
        params = ",".join(f"T.{t}" for t in tickers)
        try:
            await self._ws.send(_json_dumps({"action": "subscribe", "params": params}))
            logger.debug(f"Polygon subscribe sent: {params}")
        except Exception as e:
            logger.warning(f"Failed to send Polygon subscribe: {e}")

    async def _polygon_send_unsubscribe(self, tickers: list[str]) -> None:
        """Send unsubscribe message for tickers on the Polygon WebSocket."""
        if self._ws is None:
            return
        params = ",".join(f"T.{t}" for t in tickers)
        try:
            await self._ws.send(_json_dumps({"action": "unsubscribe", "params": params}))
            logger.debug(f"Polygon unsubscribe sent: {params}")
        except Exception as e:
            logger.warning(f"Failed to send Polygon unsubscribe: {e}")

    async def _polygon_send_option_subscribe(self, contracts: list[str]) -> None:
        """Subscribe to option quote + trade channels on the Polygon WS.

        contracts: list of Polygon option tickers like "O:SPY260414C00691000"
        """
        if self._ws is None:
            return
        # Subscribe to both quotes (Q.) and trades (T.) for each contract
        params = ",".join(
            f"Q.{c},T.{c}" for c in contracts
        )
        try:
            await self._ws.send(_json_dumps({"action": "subscribe", "params": params}))
            logger.info(f"Polygon WS option subscribe: {params}")
        except Exception as e:
            logger.warning(f"Failed to send Polygon option subscribe: {e}")

    async def _polygon_send_option_unsubscribe(self, contracts: list[str]) -> None:
        """Unsubscribe from option channels."""
        if self._ws is None:
            return
        params = ",".join(
            f"Q.{c},T.{c}" for c in contracts
        )
        try:
            await self._ws.send(_json_dumps({"action": "unsubscribe", "params": params}))
            logger.info(f"Polygon WS option unsubscribe: {params}")
        except Exception as e:
            logger.warning(f"Failed to send Polygon option unsubscribe: {e}")

    def _process_polygon_message(self, raw_msg: str | bytes) -> None:
        """Parse an incoming Polygon WebSocket message and update the price cache."""
        try:
            data = _json_loads(raw_msg)
        except Exception:
            logger.debug(f"Non-JSON Polygon message: {raw_msg!r:.100}")
            return

        # Polygon sends an array of event objects
        events = data if isinstance(data, list) else [data]

        for event in events:
            ev_type = event.get("ev")
            sym = event.get("sym", "")

            # Option events have sym starting with "O:" e.g. "O:SPY260414C00691000"
            is_option = sym.startswith("O:")

            if ev_type == "T":
                price = event.get("p")
                ts_ms = event.get("t")
                if sym and price and price > 0:
                    ts = (ts_ms / 1000.0) if ts_ms else _time.time()
                    if is_option:
                        self._option_cache[sym] = (round(float(price), 2), ts)
                        # Log option trade tick for backtesting
                        from options_owl.collectors.tick_logger import record_tick
                        record_tick(sym, ts, mid=round(float(price), 2), source="T")
                    else:
                        self._update_cache(sym, float(price), ts)
            elif ev_type == "Q":
                bid = event.get("bp")
                ask = event.get("ap")
                ts_ms = event.get("t")
                if sym and bid and ask and bid > 0 and ask > 0:
                    mid = round((bid + ask) / 2.0, 2)
                    ts = (ts_ms / 1000.0) if ts_ms else _time.time()
                    if is_option:
                        self._option_cache[sym] = (mid, ts)
                        # Log option quote tick for backtesting
                        from options_owl.collectors.tick_logger import record_tick
                        record_tick(sym, ts, bid=round(float(bid), 2),
                                    ask=round(float(ask), 2), mid=mid, source="Q")
                    else:
                        self._update_cache(sym, mid, ts)
            elif ev_type == "AM":
                # Aggregate minute bar — extract underlying price + store OHLCV
                price = event.get("c")  # close of the bar
                ts_ms = event.get("e")  # end timestamp
                if sym and price and price > 0:
                    ts = (ts_ms / 1000.0) if ts_ms else _time.time()
                    self._update_cache(sym, float(price), ts)

                    # Store full OHLCV bar for candle aggregation
                    if not is_option:
                        from collections import deque
                        if sym not in self._minute_bars:
                            self._minute_bars[sym] = deque(maxlen=self._MAX_MINUTE_BARS)
                        self._minute_bars[sym].append((
                            float(event.get("s", ts_ms or 0)),  # start timestamp ms
                            float(event.get("o", price)),       # open
                            float(event.get("h", price)),       # high
                            float(event.get("l", price)),       # low
                            float(price),                        # close
                            float(event.get("v", 0)),           # volume
                            float(event.get("vw", 0)),          # vwap
                        ))
            elif ev_type == "status":
                logger.debug(f"Polygon status: {event.get('message', event)}")

    async def _polygon_rest_option_premium(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
    ) -> float | None:
        """Fetch option premium from Polygon.io REST API.

        Uses the /v3/snapshot/options/{underlyingAsset} endpoint.
        """
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not installed — skipping Polygon REST lookup")
            return None

        # Build the Polygon option contract ticker
        # Format: O:SPY251231C00600000
        opt_char = "C" if option_type == "call" else "P"
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        expiry_str = expiry_dt.strftime("%y%m%d")
        strike_int = int(strike * 1000)
        contract = f"O:{ticker}{expiry_str}{opt_char}{strike_int:08d}"

        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}/{contract}"
        params = {"apiKey": self.settings.POLYGON_API_KEY}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.debug(
                        f"Polygon REST {resp.status_code} for {contract}"
                    )
                    return None

                data = resp.json()
                result = data.get("results", {})

                # Try last_quote bid/ask midpoint (most accurate)
                last_quote = result.get("last_quote", {})
                bid = last_quote.get("bid")
                ask = last_quote.get("ask")
                if bid and ask and bid > 0 and ask > 0:
                    return round((bid + ask) / 2.0, 2)

                # Try last_trade price
                last_trade = result.get("last_trade", {})
                trade_price = last_trade.get("price")
                if trade_price and trade_price > 0:
                    return round(float(trade_price), 2)

                # Fallback to day close
                day = result.get("day", {})
                close = day.get("close")
                if close and close > 0:
                    return round(float(close), 2)

                return None
        except Exception as e:
            logger.debug(f"Polygon REST option lookup failed for {contract}: {e}")
            return None

    # ------------------------------------------------------------------
    # Reconnection backoff
    # ------------------------------------------------------------------

    async def _backoff_sleep(self) -> None:
        """Sleep with exponential backoff between reconnection attempts."""
        self._reconnect_attempts += 1
        # Min 5s delay to avoid stacking connections at Polygon
        delay = max(5, min(2 ** self._reconnect_attempts, self._max_reconnect_delay))
        logger.info(
            f"Reconnecting in {delay}s (attempt #{self._reconnect_attempts})"
        )
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _update_cache(
        self, ticker: str, price: float, ts: float | None = None
    ) -> None:
        """Update the price cache for a ticker."""
        self._price_cache[ticker] = (price, ts or _time.time())

    @property
    def cache_snapshot(self) -> dict[str, PriceCacheEntry]:
        """Return a copy of the current price cache (for diagnostics)."""
        return dict(self._price_cache)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> str:
    return _json.dumps(obj)


def _json_loads(s: str | bytes) -> Any:
    return _json.loads(s)


def _polygon_auth_ok(data: Any) -> bool:
    """Check if a Polygon auth response indicates success."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("status") == "auth_success":
                return True
        return False
    if isinstance(data, dict):
        return data.get("status") == "auth_success"
    return False
