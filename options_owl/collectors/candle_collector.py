"""5-minute candle builder for the harvester.

Builds 5m OHLCV bars from two data sources:
1. Polygon Stocks Advanced WebSocket AM.* minute bars (real-time, preferred)
2. Polled price observations (60s intervals, fallback)

The WS connects to ``wss://socket.polygon.io/stocks`` — this is a DIFFERENT
endpoint from the options WS (``wss://socket.polygon.io/options``) that the
trading agents use.  No connection conflict.

Completed bars are written to ``stock_candles`` in the shared harvester DB.
Trading agents mount this DB read-only to get candle data without extra
Polygon REST calls or WebSocket connections.

Usage in harvester::

    collector = CandleCollector(db_path, ["SPY", "QQQ", "TSLA", ...])
    await collector.init_db()
    await collector.start_ws(api_key)   # real-time minute bars

    # Also feed prices from existing poll loop (fallback)
    collector.record_price("SPY", 542.30)

    # Every ~60s, flush completed bars to DB
    await collector.flush()

    # On shutdown
    await collector.stop_ws()
"""

from __future__ import annotations

import asyncio
import json as _json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
from loguru import logger

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CANDLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS stock_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    bar_start_ts INTEGER NOT NULL,
    bar_start TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    vwap REAL DEFAULT 0,
    source TEXT DEFAULT 'poll',
    UNIQUE(ticker, timeframe, bar_start_ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON stock_candles(ticker, timeframe, bar_start_ts);
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MinuteBar:
    """Single 1-minute OHLCV bar (from WebSocket AM.* events)."""

    ts_ms: float  # bar start timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    vwap: float = 0.0


@dataclass(slots=True)
class PriceObs:
    """Single price observation from polling."""

    ts_ms: float  # timestamp in milliseconds
    price: float


# ---------------------------------------------------------------------------
# CandleCollector
# ---------------------------------------------------------------------------

# 5 minutes in milliseconds
_5M_MS = 5 * 60 * 1000

# Keep enough observations to cover a full trading day
_MAX_OBS = 500
_MAX_MINUTE_BARS = 500


class CandleCollector:
    """Builds 5m candles from price observations and minute bars.

    Thread-safe for single-writer async usage (one harvester task).
    Multiple readers can query the DB concurrently via WAL mode.
    """

    def __init__(self, db_path: Path, tickers: list[str]) -> None:
        self._db_path = db_path
        self._tickers = frozenset(t.upper() for t in tickers)

        # Minute bar buffers (from WS): ticker -> deque[MinuteBar]
        self._minute_bars: dict[str, deque[MinuteBar]] = {
            t: deque(maxlen=_MAX_MINUTE_BARS) for t in self._tickers
        }

        # Price observation buffers (from polling): ticker -> deque[PriceObs]
        self._poll_obs: dict[str, deque[PriceObs]] = {
            t: deque(maxlen=_MAX_OBS) for t in self._tickers
        }

        # Track last flushed bar to avoid re-writing
        self._last_flushed_ts: dict[str, float] = {}

        # WebSocket state
        self._ws_running = False
        self._ws_task: asyncio.Task[None] | None = None
        self._ws_connected = False
        self._flush_task: asyncio.Task[None] | None = None

    async def init_db(self) -> None:
        """Create the stock_candles table if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(CANDLE_SCHEMA)
            await conn.commit()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_price(
        self, ticker: str, price: float, ts_ms: float | None = None
    ) -> None:
        """Record a price observation from the polling loop."""
        ticker = ticker.upper()
        buf = self._poll_obs.get(ticker)
        if buf is not None and price > 0:
            buf.append(PriceObs(ts_ms=ts_ms or time.time() * 1000, price=price))

    def ingest_minute_bar(self, ticker: str, bar: MinuteBar) -> None:
        """Accept a 1-minute bar from the WebSocket feed."""
        ticker = ticker.upper()
        buf = self._minute_bars.get(ticker)
        if buf is not None and bar.close > 0:
            buf.append(bar)

    # ------------------------------------------------------------------
    # Candle building
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_start(ts_ms: float) -> int:
        """Floor a timestamp (ms) to the start of its 5-minute bucket."""
        return int((ts_ms // _5M_MS) * _5M_MS)

    def _build_from_minute_bars(self, ticker: str) -> list[dict]:
        """Aggregate buffered WS minute bars into completed 5m candles."""
        bars = self._minute_bars.get(ticker, deque())
        if not bars:
            return []

        last_flushed = self._last_flushed_ts.get(ticker, 0)
        now_ms = time.time() * 1000
        current_bucket = self._bucket_start(now_ms)

        # Group into 5m buckets
        buckets: dict[int, list[MinuteBar]] = {}
        for bar in bars:
            bucket = self._bucket_start(bar.ts_ms)
            # Only completed buckets, not yet flushed
            if bucket < current_bucket and bucket > last_flushed:
                buckets.setdefault(bucket, []).append(bar)

        candles = []
        for bucket_ts in sorted(buckets):
            mb = buckets[bucket_ts]
            candles.append(
                {
                    "ticker": ticker,
                    "timeframe": "5m",
                    "bar_start_ts": bucket_ts,
                    "bar_start": datetime.fromtimestamp(
                        bucket_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "open": mb[0].open,
                    "high": max(b.high for b in mb),
                    "low": min(b.low for b in mb),
                    "close": mb[-1].close,
                    "volume": sum(b.volume for b in mb),
                    "vwap": mb[-1].vwap if mb[-1].vwap else 0,
                    "source": "ws",
                }
            )
        return candles

    def _build_from_polls(self, ticker: str) -> list[dict]:
        """Build approximate 5m candles from polled price observations."""
        obs = self._poll_obs.get(ticker, deque())
        if not obs:
            return []

        last_flushed = self._last_flushed_ts.get(ticker, 0)
        now_ms = time.time() * 1000
        current_bucket = self._bucket_start(now_ms)

        buckets: dict[int, list[PriceObs]] = {}
        for o in obs:
            bucket = self._bucket_start(o.ts_ms)
            if bucket < current_bucket and bucket > last_flushed:
                buckets.setdefault(bucket, []).append(o)

        candles = []
        for bucket_ts in sorted(buckets):
            points = buckets[bucket_ts]
            prices = [p.price for p in points]
            candles.append(
                {
                    "ticker": ticker,
                    "timeframe": "5m",
                    "bar_start_ts": bucket_ts,
                    "bar_start": datetime.fromtimestamp(
                        bucket_ts / 1000, tz=timezone.utc
                    ).isoformat(),
                    "open": points[0].price,
                    "high": max(prices),
                    "low": min(prices),
                    "close": points[-1].price,
                    "volume": 0,
                    "vwap": 0,
                    "source": "poll",
                }
            )
        return candles

    def build_candles(self, ticker: str) -> list[dict]:
        """Build completed 5m candles from best available source.

        Prefers WS minute bars (true OHLCV). Falls back to poll-based
        approximation when WS data isn't available.
        """
        ticker = ticker.upper()
        candles = self._build_from_minute_bars(ticker)
        if candles:
            return candles
        return self._build_from_polls(ticker)

    def build_all_candles(self) -> list[dict]:
        """Build completed candles for all tickers."""
        all_candles = []
        for ticker in self._tickers:
            all_candles.extend(self.build_candles(ticker))
        return all_candles

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    async def flush(self) -> int:
        """Write all completed candles to the database.

        Returns the number of new bars written.
        """
        all_candles = self.build_all_candles()
        if not all_candles:
            return 0

        written = 0
        try:
            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL")
                cursor = await conn.executemany(
                    """INSERT OR IGNORE INTO stock_candles
                       (ticker, timeframe, bar_start_ts, bar_start,
                        open, high, low, close, volume, vwap, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            c["ticker"],
                            c["timeframe"],
                            c["bar_start_ts"],
                            c["bar_start"],
                            c["open"],
                            c["high"],
                            c["low"],
                            c["close"],
                            c["volume"],
                            c["vwap"],
                            c["source"],
                        )
                        for c in all_candles
                    ],
                )
                await conn.commit()
                written = cursor.rowcount if cursor.rowcount > 0 else len(all_candles)
        except Exception as exc:
            logger.error(f"Candle flush failed: {exc}")
            return 0

        # Update last-flushed timestamps
        for c in all_candles:
            ticker = c["ticker"]
            ts = c["bar_start_ts"]
            if ts > self._last_flushed_ts.get(ticker, 0):
                self._last_flushed_ts[ticker] = ts

        if written:
            tickers_written = {c["ticker"] for c in all_candles}
            logger.info(
                f"Candle flush: {written} bars for "
                f"{', '.join(sorted(tickers_written))}"
            )

            # Fire-and-forget: also write candles to PostgreSQL
            try:
                from options_owl.db import postgres as pg
                if pg.is_connected():
                    from datetime import datetime as _dt, timezone as _tz
                    for c in all_candles:
                        bar_time = _dt.fromtimestamp(c["bar_start_ts"], tz=_tz.utc)
                        await pg.write_stock_candle(
                            ticker=c["ticker"],
                            timeframe=c["timeframe"],
                            open_=c["open"],
                            high=c["high"],
                            low=c["low"],
                            close=c["close"],
                            volume=c["volume"],
                            vwap=c.get("vwap", 0),
                            bar_time=bar_time,
                        )
            except Exception as exc:
                logger.debug(f"PG candle write failed: {exc}")

        return written

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_old_bars(self, keep_hours: int = 48) -> int:
        """Remove candle bars older than keep_hours from the database."""
        cutoff_ms = int((time.time() - keep_hours * 3600) * 1000)
        try:
            async with aiosqlite.connect(self._db_path) as conn:
                cursor = await conn.execute(
                    "DELETE FROM stock_candles WHERE bar_start_ts < ?",
                    (cutoff_ms,),
                )
                await conn.commit()
                deleted = cursor.rowcount
                if deleted:
                    logger.info(f"Candle cleanup: removed {deleted} bars older than {keep_hours}h")
                return deleted
        except Exception as exc:
            logger.error(f"Candle cleanup failed: {exc}")
            return 0


    # ------------------------------------------------------------------
    # WebSocket: Polygon Stocks Advanced (real-time minute bars)
    # ------------------------------------------------------------------
    # Connects to wss://socket.polygon.io/stocks — SEPARATE endpoint from
    # the options WS (wss://socket.polygon.io/options) used by trading agents.
    # No connection conflict.

    async def start_ws(self, api_key: str) -> None:
        """Start the Stocks WS for real-time minute bars + auto-flush loop."""
        if self._ws_running:
            return
        self._ws_running = True
        self._ws_task = asyncio.create_task(self._ws_loop(api_key))
        self._flush_task = asyncio.create_task(self._auto_flush_loop())
        logger.info("Candle collector: WS + auto-flush started")

    async def stop_ws(self) -> None:
        """Stop the WS connection and flush loop."""
        self._ws_running = False
        for task in (self._ws_task, self._flush_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._ws_task = None
        self._flush_task = None
        self._ws_connected = False
        logger.info("Candle collector: WS stopped")

    @property
    def ws_connected(self) -> bool:
        """Whether the stocks WS is currently connected and receiving data."""
        return self._ws_connected

    async def _auto_flush_loop(self) -> None:
        """Flush completed candles to DB every 60s."""
        try:
            while self._ws_running:
                await asyncio.sleep(60)
                try:
                    count = await self.flush()
                    if count:
                        logger.debug(f"Auto-flush wrote {count} candle bars")
                except Exception as exc:
                    logger.error(f"Auto-flush error: {exc}")
        except asyncio.CancelledError:
            # Final flush on shutdown
            try:
                await self.flush()
            except Exception:
                pass
            raise

    async def _ws_loop(self, api_key: str) -> None:
        """Connect to Polygon Stocks WS and collect AM.* minute bars.

        Uses ``wss://socket.polygon.io/stocks`` — completely separate from
        the options WS that trading agents connect to. One Polygon API key
        can hold one stocks WS + one options WS simultaneously.
        """
        try:
            import websockets
        except ImportError:
            logger.warning(
                "websockets not installed — candle collector using poll-only mode"
            )
            return

        url = "wss://socket.polygon.io/stocks"
        reconnect_attempts = 0
        max_reconnect_delay = 300

        while self._ws_running:
            # Wait for market hours
            if not _is_market_hours():
                self._ws_connected = False
                wait = _seconds_until_premarket()
                if wait > 3600:
                    logger.info(
                        f"Candle WS: market closed — sleeping "
                        f"{wait / 3600:.1f}h until pre-market"
                    )
                else:
                    logger.info(
                        f"Candle WS: market closed — sleeping "
                        f"{wait / 60:.0f}m until pre-market"
                    )
                # Sleep in 60s chunks for shutdown responsiveness
                while wait > 0 and self._ws_running:
                    chunk = min(wait, 60)
                    await asyncio.sleep(chunk)
                    wait -= chunk
                if not self._ws_running:
                    break
                continue

            try:
                # Close any lingering connection before reconnecting
                logger.info(f"Candle WS: connecting to {url}")
                async with websockets.connect(url, ping_interval=30) as ws:
                    # Wait for connection message
                    conn_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    conn_data = _json.loads(conn_msg)

                    # Check for max_connections
                    if isinstance(conn_data, list):
                        for item in conn_data:
                            if isinstance(item, dict) and item.get("status") == "max_connections":
                                logger.warning(
                                    "Candle WS: max connections — "
                                    "waiting 30s for stale connections to clear"
                                )
                                await asyncio.sleep(30)
                                continue

                    # Authenticate
                    await ws.send(_json.dumps({"action": "auth", "params": api_key}))
                    auth_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    auth_data = _json.loads(auth_msg)

                    if not _ws_auth_ok(auth_data):
                        logger.error(f"Candle WS auth failed: {auth_data}")
                        await asyncio.sleep(min(2 ** reconnect_attempts, max_reconnect_delay))
                        reconnect_attempts += 1
                        continue

                    logger.info("Candle WS: authenticated on Polygon Stocks Advanced")

                    # Subscribe to AM.{ticker} for each tracked ticker
                    subs = ",".join(f"AM.{t}" for t in sorted(self._tickers))
                    await ws.send(_json.dumps({"action": "subscribe", "params": subs}))
                    logger.info(
                        f"Candle WS: subscribed to {len(self._tickers)} tickers"
                    )

                    # Message loop
                    got_data = False
                    async for raw_msg in ws:
                        if not self._ws_running:
                            break
                        if not got_data:
                            got_data = True
                            reconnect_attempts = 0
                            self._ws_connected = True
                            logger.info("Candle WS: receiving live data")
                        self._process_ws_message(raw_msg)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._ws_connected = False
                if not self._ws_running:
                    break
                reconnect_attempts += 1
                delay = min(5 * (2 ** min(reconnect_attempts, 6)), max_reconnect_delay)
                logger.warning(
                    f"Candle WS error: {e} — reconnecting in {delay}s "
                    f"(attempt #{reconnect_attempts})"
                )
                await asyncio.sleep(delay)

        self._ws_connected = False
        logger.info("Candle WS loop exited")

    def _process_ws_message(self, raw_msg: str | bytes) -> None:
        """Parse a Polygon Stocks WS message and extract AM minute bars."""
        try:
            data = _json.loads(raw_msg)
        except Exception:
            return

        events = data if isinstance(data, list) else [data]

        for event in events:
            ev_type = event.get("ev")
            sym = event.get("sym", "")

            if ev_type != "AM" or not sym:
                continue

            # Skip anything that looks like an option symbol
            if sym.startswith("O:"):
                continue

            ticker = sym.upper()
            if ticker not in self._tickers:
                continue

            close_price = event.get("c", 0)
            if not close_price or close_price <= 0:
                continue

            bar = MinuteBar(
                ts_ms=float(event.get("s", event.get("e", 0))),
                open=float(event.get("o", close_price)),
                high=float(event.get("h", close_price)),
                low=float(event.get("l", close_price)),
                close=float(close_price),
                volume=float(event.get("v", 0)),
                vwap=float(event.get("vw", 0)),
            )
            self.ingest_minute_bar(ticker, bar)


# ---------------------------------------------------------------------------
# WS helpers
# ---------------------------------------------------------------------------


def _ws_auth_ok(data: object) -> bool:
    """Check if a Polygon WS auth response indicates success."""
    if isinstance(data, list):
        return any(
            isinstance(d, dict) and d.get("status") == "auth_success"
            for d in data
        )
    if isinstance(data, dict):
        return data.get("status") == "auth_success"
    return False


def _is_market_hours() -> bool:
    """Check if US equity market is open (9:25 AM - 4:05 PM ET, weekdays)."""
    try:
        from zoneinfo import ZoneInfo

        _et = ZoneInfo("America/New_York")
    except ImportError:
        _et = timezone(timedelta(hours=-5))
    et_now = datetime.now(_et)
    if et_now.weekday() >= 5:
        return False
    t = et_now.hour * 60 + et_now.minute
    return 9 * 60 + 25 <= t <= 16 * 60 + 5


def _seconds_until_premarket() -> float:
    """Seconds until 9:25 AM ET on the next trading day."""
    try:
        from zoneinfo import ZoneInfo

        _et = ZoneInfo("America/New_York")
    except ImportError:
        _et = timezone(timedelta(hours=-5))
    et_now = datetime.now(_et)
    target = et_now.replace(hour=9, minute=25, second=0, microsecond=0)

    if et_now >= target or et_now.weekday() >= 5:
        days_ahead = 1
        if et_now.weekday() == 4 and et_now >= target:
            days_ahead = 3
        elif et_now.weekday() == 5:
            days_ahead = 2
        elif et_now.weekday() == 6:
            days_ahead = 1
        target += timedelta(days=days_ahead)

    return max(0, (target - et_now).total_seconds())


# ---------------------------------------------------------------------------
# DB reader (for agents to query the shared DB)
# ---------------------------------------------------------------------------


async def read_candles_from_db(
    db_path: Path | str,
    ticker: str,
    timeframe: str = "5m",
    limit: int = 50,
) -> list[dict]:
    """Read recent candle bars from the shared harvester DB.

    Returns a list of dicts with keys: bar_start_ts, open, high, low, close,
    volume, vwap, source.  Sorted ascending by bar_start_ts.

    The harvester DB uses WAL mode, so we must checkpoint WAL data into the
    main DB before reading.  On read-only Docker mounts, we copy the DB +
    WAL to a temp location first so SQLite can open it normally (it needs
    write access for the -shm file even in read-only mode).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    try:
        # Read directly from the WAL-mode DB with a busy timeout.
        # WAL mode allows concurrent readers even while the harvester writes.
        # busy_timeout tells SQLite to retry for up to 5 seconds if locked.
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("PRAGMA busy_timeout = 5000")
            await conn.execute("PRAGMA journal_mode = WAL")
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """SELECT bar_start_ts, bar_start, open, high, low, close,
                          volume, vwap, source
                   FROM stock_candles
                   WHERE ticker = ? AND timeframe = ?
                   ORDER BY bar_start_ts DESC
                   LIMIT ?""",
                (ticker.upper(), timeframe, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in reversed(rows)]  # ascending order
    except Exception as exc:
        logger.debug(f"Failed to read candles from {db_path}: {exc}")
        return []
