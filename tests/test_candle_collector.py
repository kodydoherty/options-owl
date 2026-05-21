"""Tests for the candle collector module.

Covers:
- Price observation recording and buffering
- Minute bar ingestion
- 5m candle building from minute bars (WS source)
- 5m candle building from poll observations (fallback)
- Bucket alignment (5-minute boundaries)
- DB initialization, flushing, and reading
- Edge cases: empty data, single observation, boundary crossing
- Cleanup of old bars
- Source priority (WS preferred over polls)
- Integration: full cycle of record -> build -> flush -> read
- Safety: harvester changes don't affect options WS connection
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from options_owl.collectors.candle_collector import (
    CANDLE_SCHEMA,
    CandleCollector,
    MinuteBar,
    PriceObs,
    _5M_MS,
    _ws_auth_ok,
    read_candles_from_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ms(hour: int, minute: int, second: int = 0) -> float:
    """Create a timestamp in ms for today at the given HH:MM:SS UTC."""
    dt = datetime(2026, 5, 14, hour, minute, second, tzinfo=timezone.utc)
    return dt.timestamp() * 1000


def _make_minute_bar(ts_ms: float, price: float, volume: float = 1000) -> MinuteBar:
    """Create a MinuteBar at the given timestamp with OHLC around price."""
    return MinuteBar(
        ts_ms=ts_ms,
        open=price - 0.05,
        high=price + 0.10,
        low=price - 0.10,
        close=price,
        volume=volume,
        vwap=price,
    )


# ---------------------------------------------------------------------------
# Bucket alignment
# ---------------------------------------------------------------------------

class TestBucketStart:
    def test_exact_boundary(self):
        """Timestamp exactly on a 5m boundary stays there."""
        # 14:00:00 UTC
        ts = _ts_ms(14, 0)
        assert CandleCollector._bucket_start(ts) == int((ts // _5M_MS) * _5M_MS)

    def test_mid_bucket(self):
        """Timestamp in the middle of a bucket floors to bucket start."""
        ts_14_02 = _ts_ms(14, 2, 30)  # 14:02:30
        ts_14_00 = _ts_ms(14, 0)      # expected bucket
        assert CandleCollector._bucket_start(ts_14_02) == CandleCollector._bucket_start(ts_14_00)

    def test_end_of_bucket(self):
        """Timestamp at 14:04:59 floors to 14:00."""
        ts_14_04_59 = _ts_ms(14, 4, 59)
        ts_14_00 = _ts_ms(14, 0)
        assert CandleCollector._bucket_start(ts_14_04_59) == CandleCollector._bucket_start(ts_14_00)

    def test_next_bucket(self):
        """Timestamp at 14:05:00 starts a new bucket."""
        ts_14_05 = _ts_ms(14, 5)
        ts_14_00 = _ts_ms(14, 0)
        bucket_14_05 = CandleCollector._bucket_start(ts_14_05)
        bucket_14_00 = CandleCollector._bucket_start(ts_14_00)
        assert bucket_14_05 > bucket_14_00
        assert bucket_14_05 - bucket_14_00 == _5M_MS

    def test_consecutive_buckets(self):
        """Multiple consecutive 5m buckets are evenly spaced."""
        buckets = [
            CandleCollector._bucket_start(_ts_ms(14, m))
            for m in range(0, 20, 5)
        ]
        for i in range(1, len(buckets)):
            assert buckets[i] - buckets[i - 1] == _5M_MS


# ---------------------------------------------------------------------------
# Price observation recording
# ---------------------------------------------------------------------------

class TestRecordPrice:
    def test_record_basic(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("SPY", 542.30)
        assert len(collector._poll_obs["SPY"]) == 1
        assert collector._poll_obs["SPY"][0].price == 542.30

    def test_case_insensitive(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("spy", 542.30)
        assert len(collector._poll_obs["SPY"]) == 1

    def test_unknown_ticker_ignored(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("TSLA", 200.0)
        assert "TSLA" not in collector._poll_obs

    def test_zero_price_ignored(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("SPY", 0)
        assert len(collector._poll_obs["SPY"]) == 0

    def test_negative_price_ignored(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("SPY", -1.0)
        assert len(collector._poll_obs["SPY"]) == 0

    def test_custom_timestamp(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        ts = _ts_ms(14, 0)
        collector.record_price("SPY", 542.30, ts_ms=ts)
        assert collector._poll_obs["SPY"][0].ts_ms == ts

    def test_auto_timestamp(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        before = time.time() * 1000
        collector.record_price("SPY", 542.30)
        after = time.time() * 1000
        obs_ts = collector._poll_obs["SPY"][0].ts_ms
        assert before <= obs_ts <= after

    def test_buffer_limit(self):
        """Buffer doesn't grow unbounded."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        for i in range(600):
            collector.record_price("SPY", 542.0 + i * 0.01, ts_ms=_ts_ms(10, 0) + i * 60000)
        # deque maxlen is 500
        assert len(collector._poll_obs["SPY"]) == 500


# ---------------------------------------------------------------------------
# Minute bar ingestion
# ---------------------------------------------------------------------------

class TestIngestMinuteBar:
    def test_basic(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        bar = _make_minute_bar(_ts_ms(14, 0), 542.30)
        collector.ingest_minute_bar("SPY", bar)
        assert len(collector._minute_bars["SPY"]) == 1

    def test_zero_close_ignored(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        bar = MinuteBar(ts_ms=_ts_ms(14, 0), open=0, high=0, low=0, close=0)
        collector.ingest_minute_bar("SPY", bar)
        assert len(collector._minute_bars["SPY"]) == 0

    def test_unknown_ticker_ignored(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        bar = _make_minute_bar(_ts_ms(14, 0), 200.0)
        collector.ingest_minute_bar("TSLA", bar)
        assert "TSLA" not in collector._minute_bars


# ---------------------------------------------------------------------------
# Building candles from minute bars
# ---------------------------------------------------------------------------

class TestBuildFromMinuteBars:
    def test_single_complete_bucket(self):
        """5 minute bars in one bucket produce 1 candle."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        # Simulate a completed 5m bucket (14:00 - 14:04)
        # We need "now" to be past the bucket boundary (14:05+)
        prices = [542.0, 543.0, 544.0, 541.0, 542.5]
        for i, p in enumerate(prices):
            bar = MinuteBar(
                ts_ms=_ts_ms(14, i), open=p - 0.1, high=p + 0.5,
                low=p - 0.5, close=p, volume=1000 * (i + 1),
            )
            collector.ingest_minute_bar("SPY", bar)

        # Mock time to be past the 14:05 boundary
        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        c = candles[0]
        assert c["ticker"] == "SPY"
        assert c["timeframe"] == "5m"
        assert c["source"] == "ws"
        assert c["open"] == 542.0 - 0.1  # first bar's open
        assert c["high"] == max(p + 0.5 for p in prices)  # highest high
        assert c["low"] == min(p - 0.5 for p in prices)   # lowest low
        assert c["close"] == 542.5  # last bar's close
        assert c["volume"] == sum(1000 * (i + 1) for i in range(5))

    def test_current_bucket_excluded(self):
        """In-progress bucket is NOT included in output."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        # Place a bar at 14:00 (bucket starts at 14:00)
        collector.ingest_minute_bar("SPY", _make_minute_bar(_ts_ms(14, 0), 542.0))

        # "Now" is 14:03 — still in the same bucket
        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 3) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 0  # not yet complete

    def test_multiple_buckets(self):
        """Bars spanning two 5m buckets produce 2 candles."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        # Bucket 1: 14:00-14:04
        for i in range(5):
            collector.ingest_minute_bar(
                "SPY", _make_minute_bar(_ts_ms(14, i), 542.0 + i)
            )
        # Bucket 2: 14:05-14:09
        for i in range(5):
            collector.ingest_minute_bar(
                "SPY", _make_minute_bar(_ts_ms(14, 5 + i), 550.0 + i)
            )

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 11) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 2
        assert candles[0]["bar_start_ts"] < candles[1]["bar_start_ts"]

    def test_already_flushed_excluded(self):
        """Previously flushed candles are not re-built."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        for i in range(5):
            collector.ingest_minute_bar(
                "SPY", _make_minute_bar(_ts_ms(14, i), 542.0)
            )

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")
            assert len(candles) == 1

            # Simulate flush by updating last_flushed
            collector._last_flushed_ts["SPY"] = candles[0]["bar_start_ts"]

            # Build again — should be empty
            candles2 = collector.build_candles("SPY")
            assert len(candles2) == 0


# ---------------------------------------------------------------------------
# Building candles from poll observations
# ---------------------------------------------------------------------------

class TestBuildFromPolls:
    def test_single_bucket(self):
        """Poll observations in one 5m bucket produce 1 candle."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        prices = [542.0, 543.5, 541.0, 542.8, 543.2]
        for i, p in enumerate(prices):
            collector.record_price("SPY", p, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        c = candles[0]
        assert c["source"] == "poll"
        assert c["open"] == 542.0
        assert c["high"] == 543.5
        assert c["low"] == 541.0
        assert c["close"] == 543.2
        assert c["volume"] == 0  # no volume from polls

    def test_empty_observations(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")
        assert candles == []


# ---------------------------------------------------------------------------
# Source priority
# ---------------------------------------------------------------------------

class TestSourcePriority:
    def test_ws_preferred_over_polls(self):
        """When both WS minute bars and polls exist, WS wins."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        # Add both WS and poll data for the same bucket
        for i in range(3):
            collector.ingest_minute_bar(
                "SPY", _make_minute_bar(_ts_ms(14, i), 542.0)
            )
            collector.record_price("SPY", 540.0, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        assert candles[0]["source"] == "ws"

    def test_poll_fallback_when_no_ws(self):
        """When no WS data, falls back to poll observations."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        for i in range(3):
            collector.record_price("SPY", 542.0 + i, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        assert candles[0]["source"] == "poll"


# ---------------------------------------------------------------------------
# DB integration (init, flush, read, cleanup)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_candles.db"


class TestDBIntegration:
    @pytest.mark.asyncio
    async def test_init_db_creates_table(self, tmp_db):
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()
        assert tmp_db.exists()

        import aiosqlite
        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_candles'"
            )
            row = await cursor.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_init_db_idempotent(self, tmp_db):
        """Calling init_db twice doesn't error."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()
        await collector.init_db()

    @pytest.mark.asyncio
    async def test_flush_writes_candles(self, tmp_db):
        collector = CandleCollector(tmp_db, ["SPY", "QQQ"])
        await collector.init_db()

        # Add data for completed buckets
        for i in range(5):
            collector.record_price("SPY", 542.0 + i, ts_ms=_ts_ms(14, i))
            collector.record_price("QQQ", 480.0 + i, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            written = await collector.flush()

        assert written >= 2  # at least 1 per ticker

        import aiosqlite
        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM stock_candles")
            count = (await cursor.fetchone())[0]
            assert count >= 2

    @pytest.mark.asyncio
    async def test_flush_idempotent(self, tmp_db):
        """Flushing twice doesn't duplicate rows."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        for i in range(5):
            collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            w1 = await collector.flush()
            w2 = await collector.flush()

        # Second flush should write 0 (already flushed)
        assert w1 >= 1
        assert w2 == 0

    @pytest.mark.asyncio
    async def test_read_candles_from_db(self, tmp_db):
        """read_candles_from_db returns bars in ascending order."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        # Two complete buckets
        for i in range(5):
            collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, i))
        for i in range(5):
            collector.record_price("SPY", 543.0, ts_ms=_ts_ms(14, 5 + i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 11) / 1000
            await collector.flush()

        rows = await read_candles_from_db(tmp_db, "SPY", "5m", limit=10)
        assert len(rows) >= 2
        # Verify ascending order
        for i in range(1, len(rows)):
            assert rows[i]["bar_start_ts"] > rows[i - 1]["bar_start_ts"]

    @pytest.mark.asyncio
    async def test_read_candles_nonexistent_db(self, tmp_path):
        """Reading from a nonexistent DB returns empty list."""
        rows = await read_candles_from_db(tmp_path / "nope.db", "SPY")
        assert rows == []

    @pytest.mark.asyncio
    async def test_read_candles_wrong_ticker(self, tmp_db):
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        for i in range(5):
            collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            await collector.flush()

        rows = await read_candles_from_db(tmp_db, "TSLA")
        assert rows == []

    @pytest.mark.asyncio
    async def test_cleanup_old_bars(self, tmp_db):
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        # Insert a bar with a very old timestamp
        import aiosqlite
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute(
                """INSERT INTO stock_candles
                   (ticker, timeframe, bar_start_ts, bar_start, open, high, low, close)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("SPY", "5m", 1000000, "2000-01-01T00:00:00+00:00", 100, 101, 99, 100),
            )
            # Insert a recent bar
            recent_ts = int(time.time() * 1000)
            await conn.execute(
                """INSERT INTO stock_candles
                   (ticker, timeframe, bar_start_ts, bar_start, open, high, low, close)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("SPY", "5m", recent_ts, "2026-05-14T14:00:00+00:00", 542, 543, 541, 542),
            )
            await conn.commit()

        deleted = await collector.cleanup_old_bars(keep_hours=1)
        assert deleted >= 1

        # Recent bar should survive
        rows = await read_candles_from_db(tmp_db, "SPY")
        assert len(rows) >= 1
        assert rows[-1]["bar_start_ts"] == recent_ts


# ---------------------------------------------------------------------------
# CandleCache shared DB integration
# ---------------------------------------------------------------------------

class TestCandleCacheSharedDB:
    @pytest.mark.asyncio
    async def test_shared_db_used_before_rest(self, tmp_db):
        """CandleCache checks shared DB before making REST calls."""
        from options_owl.collectors.candle_cache import CandleCache

        # Set up shared DB with candle data
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        for i in range(5):
            collector.record_price("SPY", 542.0 + i * 0.1, ts_ms=_ts_ms(14, i))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            await collector.flush()

        # Create CandleCache with shared_db_path (no API key needed)
        cache = CandleCache(api_key="", shared_db_path=str(tmp_db))
        bars = await cache.get_candles("SPY", "5m")

        assert len(bars) >= 1
        assert bars[0].close > 0

    @pytest.mark.asyncio
    async def test_shared_db_missing_falls_through(self, tmp_path):
        """When shared DB doesn't exist, CandleCache falls through to next source."""
        from options_owl.collectors.candle_cache import CandleCache

        cache = CandleCache(
            api_key="",
            shared_db_path=str(tmp_path / "nonexistent.db"),
        )
        bars = await cache.get_candles("SPY", "5m")
        # Should return empty (no REST key either), not crash
        assert bars == []


# ---------------------------------------------------------------------------
# Build all candles (multi-ticker)
# ---------------------------------------------------------------------------

class TestBuildAllCandles:
    def test_multiple_tickers(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY", "QQQ", "TSLA"])

        for i in range(5):
            collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, i))
            collector.record_price("QQQ", 480.0, ts_ms=_ts_ms(14, i))
            # TSLA has no data

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            all_candles = collector.build_all_candles()

        tickers = {c["ticker"] for c in all_candles}
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "TSLA" not in tickers  # no data


# ---------------------------------------------------------------------------
# Safety: harvester integration doesn't break options flow
# ---------------------------------------------------------------------------

class TestHarvesterSafety:
    """Verify harvester candle changes are isolated from options data flow.

    CRITICAL: The harvester's candle WS connects to wss://socket.polygon.io/stocks.
    Trading agents connect to wss://socket.polygon.io/options.
    These are DIFFERENT endpoints — no conflict.
    """

    def test_ws_uses_stocks_endpoint_not_options(self):
        """Candle WS connects to /stocks, NOT /options (agents use /options)."""
        import inspect
        import options_owl.collectors.candle_collector as mod

        # Check the _ws_loop method specifically (not docstrings/comments)
        ws_loop_source = inspect.getsource(mod.CandleCollector._ws_loop)
        # Must use the stocks WS endpoint
        assert 'wss://socket.polygon.io/stocks' in ws_loop_source
        # Must NOT connect to the options WS endpoint
        assert 'wss://socket.polygon.io/options' not in ws_loop_source

    def test_agent_ws_uses_options_endpoint(self):
        """Trading agents connect to /options, confirming no conflict."""
        import inspect
        import options_owl.collectors.market_data_stream as mod

        source = inspect.getsource(mod)
        assert 'wss://socket.polygon.io/options' in source
        # Agents must NOT connect to /stocks (harvester owns that)
        assert 'wss://socket.polygon.io/stocks' not in source

    def test_candle_collector_does_not_import_market_data_stream(self):
        """CandleCollector doesn't import MarketDataStream (options WS)."""
        import inspect
        import options_owl.collectors.candle_collector as mod

        source = inspect.getsource(mod)
        assert "market_data_stream" not in source
        assert "polygon_options" not in source

    def test_candle_collector_no_http_calls(self):
        """CandleCollector only writes via aiosqlite, no HTTP calls."""
        import inspect
        import options_owl.collectors.candle_collector as mod

        source = inspect.getsource(mod)
        assert "httpx" not in source
        assert "requests" not in source
        assert "yfinance" not in source

    def test_harvester_import_order(self):
        """Verify CandleCollector import in harvester is before run_harvester."""
        import inspect
        import options_owl.harvester as mod

        source = inspect.getsource(mod)
        import_pos = source.find("from options_owl.collectors.candle_collector")
        run_pos = source.find("async def run_harvester")
        assert import_pos > 0
        assert run_pos > import_pos

    def test_candle_schema_has_unique_constraint(self):
        """Verify UNIQUE constraint prevents duplicate candle bars."""
        assert "UNIQUE(ticker, timeframe, bar_start_ts)" in CANDLE_SCHEMA

    def test_candle_schema_uses_wal(self):
        """WAL mode required for concurrent reads from multiple agents."""
        import inspect
        import options_owl.collectors.candle_collector as mod

        source = inspect.getsource(mod)
        assert "PRAGMA journal_mode=WAL" in source

    def test_docker_compose_harvester_db_readonly(self):
        """Agent containers mount harvester DB as read-only."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        # Each agent should have :ro mount of harvester journal
        assert "shared_harvester:ro" in compose
        # Harvester itself should NOT have :ro on its own journal
        harvester_section = compose[compose.find("owlet-harvester:"):]
        assert ":ro" not in harvester_section.split("volumes:")[1].split("\n")[1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_observation_in_bucket(self):
        """A single price observation still produces a valid candle."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, 0))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        c = candles[0]
        assert c["open"] == c["high"] == c["low"] == c["close"] == 542.0

    def test_observations_across_day_boundary(self):
        """Observations near midnight are bucketed correctly."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])

        # Just before midnight bucket and just after
        late_ts = _ts_ms(23, 58)
        early_ts = _ts_ms(0, 1)  # next bucket after midnight

        collector.record_price("SPY", 542.0, ts_ms=late_ts)
        collector.record_price("SPY", 543.0, ts_ms=early_ts)

        bucket_late = CandleCollector._bucket_start(late_ts)
        bucket_early = CandleCollector._bucket_start(early_ts)
        # These should be different buckets
        assert bucket_late != bucket_early

    def test_build_candles_unknown_ticker(self):
        """Building candles for untracked ticker returns empty."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        candles = collector.build_candles("UNKNOWN")
        assert candles == []

    def test_ws_auth_ok_success(self):
        assert _ws_auth_ok([{"status": "auth_success"}]) is True

    def test_ws_auth_ok_failure(self):
        assert _ws_auth_ok([{"status": "auth_failed"}]) is False

    def test_ws_auth_ok_dict(self):
        assert _ws_auth_ok({"status": "auth_success"}) is True

    def test_ws_auth_ok_garbage(self):
        assert _ws_auth_ok("garbage") is False
        assert _ws_auth_ok(None) is False
        assert _ws_auth_ok([]) is False

    def test_ws_connected_property(self):
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        assert collector.ws_connected is False

    def test_bar_start_is_valid_iso(self):
        """bar_start field is a valid ISO 8601 timestamp."""
        collector = CandleCollector(Path("/tmp/test.db"), ["SPY"])
        collector.record_price("SPY", 542.0, ts_ms=_ts_ms(14, 0))

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            candles = collector.build_candles("SPY")

        assert len(candles) == 1
        # Should parse without error
        dt = datetime.fromisoformat(candles[0]["bar_start"])
        assert dt.tzinfo is not None  # must be timezone-aware


# ---------------------------------------------------------------------------
# Full integration cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    @pytest.mark.asyncio
    async def test_record_build_flush_read(self, tmp_db):
        """Complete cycle: record -> build -> flush -> read."""
        tickers = ["SPY", "QQQ", "TSLA"]
        collector = CandleCollector(tmp_db, tickers)
        await collector.init_db()

        # Simulate 15 minutes of polling (3 complete 5m buckets)
        for minute in range(15):
            for ticker, base in [("SPY", 542), ("QQQ", 480), ("TSLA", 200)]:
                price = base + minute * 0.1
                collector.record_price(
                    ticker, price, ts_ms=_ts_ms(14, minute)
                )

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            # "Now" is 14:16 — so 14:00, 14:05, 14:10 buckets are complete
            mock_time.time.return_value = _ts_ms(14, 16) / 1000
            written = await collector.flush()

        assert written >= 9  # 3 tickers * 3 buckets

        # Read back
        for ticker in tickers:
            rows = await read_candles_from_db(tmp_db, ticker)
            assert len(rows) >= 3
            # Verify chronological order
            for i in range(1, len(rows)):
                assert rows[i]["bar_start_ts"] > rows[i - 1]["bar_start_ts"]
            # Verify OHLC makes sense
            for r in rows:
                assert r["high"] >= r["open"]
                assert r["high"] >= r["close"]
                assert r["low"] <= r["open"]
                assert r["low"] <= r["close"]

    @pytest.mark.asyncio
    async def test_ws_message_processing(self, tmp_db):
        """Test _process_ws_message correctly parses Polygon AM events."""
        collector = CandleCollector(tmp_db, ["SPY", "TSLA"])
        await collector.init_db()

        import json

        # Valid AM event for SPY
        msg = json.dumps([{
            "ev": "AM",
            "sym": "SPY",
            "s": _ts_ms(14, 0),
            "o": 542.10,
            "h": 542.50,
            "l": 541.80,
            "c": 542.30,
            "v": 150000,
            "vw": 542.15,
        }])
        collector._process_ws_message(msg)
        assert len(collector._minute_bars["SPY"]) == 1
        bar = collector._minute_bars["SPY"][0]
        assert bar.open == 542.10
        assert bar.high == 542.50
        assert bar.low == 541.80
        assert bar.close == 542.30
        assert bar.volume == 150000

    @pytest.mark.asyncio
    async def test_ws_ignores_option_symbols(self, tmp_db):
        """WS processor ignores option symbols (O:SPY...)."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        import json

        msg = json.dumps([{
            "ev": "AM",
            "sym": "O:SPY260514C00542000",
            "s": _ts_ms(14, 0),
            "o": 2.50, "h": 2.60, "l": 2.40, "c": 2.55, "v": 500,
        }])
        collector._process_ws_message(msg)
        assert len(collector._minute_bars["SPY"]) == 0

    @pytest.mark.asyncio
    async def test_ws_ignores_untracked_tickers(self, tmp_db):
        """WS processor ignores tickers not in the universe."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        import json

        msg = json.dumps([{
            "ev": "AM", "sym": "AAPL",
            "s": _ts_ms(14, 0),
            "o": 190, "h": 191, "l": 189, "c": 190.5, "v": 10000,
        }])
        collector._process_ws_message(msg)
        assert len(collector._minute_bars["SPY"]) == 0

    @pytest.mark.asyncio
    async def test_ws_ignores_non_am_events(self, tmp_db):
        """WS processor ignores non-AM events (T, Q, status)."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        import json

        # Trade event
        msg = json.dumps([{"ev": "T", "sym": "SPY", "p": 542.30}])
        collector._process_ws_message(msg)
        assert len(collector._minute_bars["SPY"]) == 0

        # Status event
        msg = json.dumps([{"ev": "status", "message": "connected"}])
        collector._process_ws_message(msg)
        assert len(collector._minute_bars["SPY"]) == 0

    @pytest.mark.asyncio
    async def test_ws_handles_malformed_json(self, tmp_db):
        """WS processor doesn't crash on malformed messages."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        collector._process_ws_message("not json at all")
        collector._process_ws_message(b"\x00\x01\x02")
        collector._process_ws_message("")
        # Should not crash
        assert len(collector._minute_bars["SPY"]) == 0

    @pytest.mark.asyncio
    async def test_ws_bars_produce_better_candles(self, tmp_db):
        """WS minute bars produce candles with real OHLCV (not just close)."""
        collector = CandleCollector(tmp_db, ["SPY"])
        await collector.init_db()

        # Add WS bars with distinct OHLC
        for i in range(5):
            bar = MinuteBar(
                ts_ms=_ts_ms(14, i),
                open=540.0 + i,
                high=545.0 + i,
                low=538.0 + i,
                close=542.0 + i,
                volume=100000 * (i + 1),
                vwap=541.0 + i,
            )
            collector.ingest_minute_bar("SPY", bar)

        with patch("options_owl.collectors.candle_collector.time") as mock_time:
            mock_time.time.return_value = _ts_ms(14, 6) / 1000
            written = await collector.flush()

        assert written >= 1
        rows = await read_candles_from_db(tmp_db, "SPY")
        assert len(rows) >= 1
        r = rows[-1]
        assert r["volume"] > 0  # real volume from WS
        assert r["high"] > r["low"]  # distinct OHLC
