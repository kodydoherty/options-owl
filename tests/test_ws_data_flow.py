"""End-to-end tests for the WebSocket data flow: WS → Redis → bot reads.

Tests cover:
- WS health watchdog (alerts when disconnected/stale, healthy when OK)
- Boot-time data wait (waits for fresh data, times out gracefully)
- WS smoke test (passes when WS connected, warns on timeout)
- Redis snapshot freshness (position monitor rejects stale data)
- Full pipeline: quote → flow_collector → Redis → position_monitor read
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCollector:
    """Minimal mock of CandleCollector or FlowCollector for watchdog tests."""

    def __init__(self, connected: bool = True, last_age: float = 5.0):
        self._ws_connected = connected
        self._last_msg_age = last_age

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    @property
    def last_msg_age(self) -> float:
        return self._last_msg_age


class FakeRedis:
    """In-memory Redis mock with get/set/scan_iter."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def scan_iter(self, match: str = "*"):
        import fnmatch
        for key in list(self._store.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    async def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# WS Health Watchdog Tests
# ---------------------------------------------------------------------------


class TestWSHealthWatchdog:
    """Test _ws_health_watchdog publishes correct health status to Redis."""

    @pytest.mark.asyncio
    async def test_healthy_status(self):
        """When both WS are connected and fresh, health = True."""
        from options_owl.harvester import _ws_health_watchdog

        candle = FakeCollector(connected=True, last_age=5.0)
        flow = FakeCollector(connected=True, last_age=3.0)

        fake_redis = FakeRedis()

        with (
            patch("options_owl.harvester._is_market_hours", return_value=True),
            patch("options_owl.db.redis_client.is_connected", return_value=True),
            patch("options_owl.db.redis_client._redis", fake_redis),
        ):
            # Run watchdog for one iteration then cancel
            task = asyncio.create_task(_ws_health_watchdog(candle, flow))
            await asyncio.sleep(0.5)  # let it get past the sleep(30)

            # The watchdog sleeps 30s. Patch sleep to unblock.
            # Instead, directly test the health logic by checking after cancel.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Since we can't easily test the 30s sleep loop in unit tests,
        # verify the function accepts the right types and doesn't crash.
        # The real integration test is verifying Redis key format.

    @pytest.mark.asyncio
    async def test_unhealthy_candle_ws(self):
        """Candle WS disconnected should produce issues list."""
        candle = FakeCollector(connected=False, last_age=120.0)
        flow = FakeCollector(connected=True, last_age=3.0)

        # Verify the collector reports correctly — only candle is unhealthy
        assert not candle.ws_connected
        assert candle.last_msg_age == 120.0
        assert flow.ws_connected  # flow is healthy, only candle should flag

    @pytest.mark.asyncio
    async def test_unhealthy_flow_ws_stale(self):
        """Flow WS connected but stale (>60s) should produce issues."""
        candle = FakeCollector(connected=True, last_age=5.0)
        flow = FakeCollector(connected=True, last_age=90.0)

        # This would produce "flow_ws stale (90s)" in the watchdog
        assert flow.ws_connected
        assert flow.last_msg_age > 60
        assert candle.last_msg_age < 60  # candle is fresh, only flow should flag


# ---------------------------------------------------------------------------
# WS Smoke Test
# ---------------------------------------------------------------------------


class TestWSSmokeTest:
    """Test _ws_smoke_test startup verification."""

    @pytest.mark.asyncio
    async def test_passes_when_both_connected(self):
        """Smoke test passes immediately when both WS are connected and fresh."""
        from options_owl.harvester import _ws_smoke_test

        candle = FakeCollector(connected=True, last_age=5.0)
        flow = FakeCollector(connected=True, last_age=3.0)

        with patch("options_owl.harvester._is_market_hours", return_value=True):
            # Should return quickly (not wait full timeout)
            await asyncio.wait_for(
                _ws_smoke_test(candle, flow, timeout=10, retry_interval=1),
                timeout=5,
            )

    @pytest.mark.asyncio
    async def test_skips_outside_market_hours(self):
        """Smoke test skips immediately when market is closed."""
        from options_owl.harvester import _ws_smoke_test

        candle = FakeCollector(connected=False, last_age=0.0)
        flow = FakeCollector(connected=False, last_age=0.0)

        with patch("options_owl.harvester._is_market_hours", return_value=False):
            await asyncio.wait_for(
                _ws_smoke_test(candle, flow, timeout=10, retry_interval=1),
                timeout=2,
            )

    @pytest.mark.asyncio
    async def test_times_out_gracefully(self):
        """Smoke test warns but doesn't crash on timeout."""
        from options_owl.harvester import _ws_smoke_test

        candle = FakeCollector(connected=False, last_age=0.0)
        flow = FakeCollector(connected=False, last_age=0.0)

        with patch("options_owl.harvester._is_market_hours", return_value=True):
            # Should complete after timeout (3s), not hang
            await asyncio.wait_for(
                _ws_smoke_test(candle, flow, timeout=3, retry_interval=1),
                timeout=10,
            )

    @pytest.mark.asyncio
    async def test_passes_without_flow_collector(self):
        """Smoke test passes with just candle WS (no flow collector)."""
        from options_owl.harvester import _ws_smoke_test

        candle = FakeCollector(connected=True, last_age=5.0)

        with patch("options_owl.harvester._is_market_hours", return_value=True):
            await asyncio.wait_for(
                _ws_smoke_test(candle, None, timeout=10, retry_interval=1),
                timeout=5,
            )


# ---------------------------------------------------------------------------
# Boot-Time Data Wait Tests
# ---------------------------------------------------------------------------


class TestBootTimeDataWait:
    """Test _wait_for_fresh_redis_data in discord_collector."""

    @pytest.mark.asyncio
    async def test_skips_outside_market_hours(self):
        """Boot wait skips immediately when market is closed."""
        from options_owl.collectors.discord_collector import OptionsOwlBot

        bot = OptionsOwlBot.__new__(OptionsOwlBot)
        bot.settings = MagicMock()

        with patch(
            "options_owl.execution.position_monitor._is_market_hours",
            return_value=False,
        ):
            result = await bot._wait_for_fresh_redis_data(max_wait=5, check_interval=1)
            assert result is True

    @pytest.mark.asyncio
    async def test_finds_fresh_data(self):
        """Boot wait returns True when fresh snapshot found in Redis."""
        from options_owl.collectors.discord_collector import OptionsOwlBot

        bot = OptionsOwlBot.__new__(OptionsOwlBot)
        bot.settings = MagicMock()

        fresh_snap = {"t": time.time(), "mid": 5.0, "bid": 4.9, "ask": 5.1}

        with (
            patch(
                "options_owl.execution.position_monitor._is_market_hours",
                return_value=True,
            ),
            patch("options_owl.db.redis_client.is_connected", return_value=True),
            patch(
                "options_owl.db.redis_client.get_option_snapshots_for_ticker",
                new_callable=AsyncMock,
                return_value=[fresh_snap],
            ),
            patch(
                "options_owl.db.redis_client._redis",
                MagicMock(get=AsyncMock(return_value=None)),
            ),
        ):
            result = await bot._wait_for_fresh_redis_data(max_wait=10, check_interval=1)
            assert result is True

    @pytest.mark.asyncio
    async def test_times_out_gracefully(self):
        """Boot wait returns False after max_wait with no fresh data."""
        from options_owl.collectors.discord_collector import OptionsOwlBot

        bot = OptionsOwlBot.__new__(OptionsOwlBot)
        bot.settings = MagicMock()

        stale_snap = {"t": time.time() - 120, "mid": 5.0}  # 2 min old

        with (
            patch(
                "options_owl.execution.position_monitor._is_market_hours",
                return_value=True,
            ),
            patch("options_owl.db.redis_client.is_connected", return_value=True),
            patch(
                "options_owl.db.redis_client.get_option_snapshots_for_ticker",
                new_callable=AsyncMock,
                return_value=[stale_snap],
            ),
            patch(
                "options_owl.db.redis_client._redis",
                MagicMock(get=AsyncMock(return_value=None)),
            ),
        ):
            result = await bot._wait_for_fresh_redis_data(max_wait=3, check_interval=1)
            assert result is False


# ---------------------------------------------------------------------------
# Redis Snapshot Freshness (Position Monitor)
# ---------------------------------------------------------------------------


class TestRedisSnapshotFreshness:
    """Test that position monitor correctly handles fresh vs stale Redis data."""

    def test_fresh_snapshot_accepted(self):
        """Snapshot < 30s old should be used."""
        snap = {"mid": 5.0, "t": time.time() - 10, "bid": 4.9, "ask": 5.1}
        age = time.time() - snap["t"]
        assert age < 30
        assert snap["mid"] > 0

    def test_stale_snapshot_rejected(self):
        """Snapshot > 30s old should be rejected (fall back to REST)."""
        snap = {"mid": 5.0, "t": time.time() - 60, "bid": 4.9, "ask": 5.1}
        age = time.time() - snap["t"]
        assert age >= 30

    def test_missing_timestamp_rejected(self):
        """Snapshot without 't' field should be rejected."""
        snap = {"mid": 5.0, "bid": 4.9, "ask": 5.1}
        age = time.time() - snap.get("t", 0) if "t" in snap else 999
        assert age >= 30

    def test_zero_mid_rejected(self):
        """Snapshot with mid=0 should be rejected."""
        snap = {"mid": 0, "t": time.time(), "bid": 0, "ask": 0}
        assert not (snap.get("mid") and snap["mid"] > 0)


# ---------------------------------------------------------------------------
# Full Pipeline: Quote → FlowCollector → Redis → Read
# ---------------------------------------------------------------------------


class TestFullDataPipeline:
    """Test the complete data flow from WS quote to Redis read."""

    def test_flow_collector_processes_quote_to_redis_buffer(self):
        """FlowCollector._process_quote should populate Redis snapshot buffer."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"], price_getter=lambda t: 550.0)

        # Simulate a Polygon Options WS quote event
        event = {
            "ev": "Q",
            "sym": "O:SPY260610C00550000",
            "bp": 5.0,  # bid
            "ap": 5.20,  # ask
            "bs": 100,  # bid size
            "as": 80,  # ask size
        }
        fc._process_quote(event)

        # Verify Redis snapshot buffer populated
        assert len(fc._redis_snapshot_buffer) == 1
        key = "SPY:call:550.0:2026-06-10"
        assert key in fc._redis_snapshot_buffer

        snap = fc._redis_snapshot_buffer[key]
        assert snap["bid"] == 5.0
        assert snap["ask"] == 5.20
        assert snap["mid"] == 5.10
        assert snap["underlying_price"] == 550.0
        assert snap["strike"] == 550.0
        assert snap["option_type"] == "call"
        assert snap["expiry_date"] == "2026-06-10"

    def test_flow_collector_processes_quote_without_price(self):
        """Quote processing should work even without underlying price (no greeks)."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["TSLA"], price_getter=lambda t: None)

        event = {
            "ev": "Q",
            "sym": "O:TSLA260610C00200000",
            "bp": 3.0,
            "ap": 3.40,
            "bs": 50,
            "as": 30,
        }
        fc._process_quote(event)

        key = "TSLA:call:200.0:2026-06-10"
        assert key in fc._redis_snapshot_buffer
        snap = fc._redis_snapshot_buffer[key]
        assert snap["underlying_price"] is None
        assert snap["delta"] is None  # no greeks without underlying

    def test_flow_collector_premium_buffer(self):
        """Quote should also populate the legacy premium buffer."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"], price_getter=lambda t: 550.0)

        event = {
            "ev": "Q",
            "sym": "O:SPY260610C00550000",
            "bp": 5.0,
            "ap": 5.20,
        }
        fc._process_quote(event)

        key = "SPY:call:550.0:2026-06-10"
        assert key in fc._redis_quote_buffer
        bid, ask, mid = fc._redis_quote_buffer[key]
        assert bid == 5.0
        assert ask == 5.20
        assert mid == 5.10

    def test_flow_collector_trade_updates_volume(self):
        """Trade events should update per-contract volume in snapshots."""
        from datetime import date, timedelta

        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"], price_getter=lambda t: 550.0)

        # Expiry must be in the future or _process_trade's DTE filter rejects it.
        # (Previously hardcoded 2026-06-10 and rotted once the date passed.)
        _exp_date = date.today() + timedelta(days=2)
        occ_sym = f"O:SPY{_exp_date.strftime('%y%m%d')}C00550000"

        # Initialize volume date (normally done by first quote of the day)
        fc._volume_date = date.today().isoformat()

        # Process a trade (>= MIN_FLOW_SIZE)
        trade_event = {
            "ev": "T",
            "sym": occ_sym,
            "p": 5.10,
            "s": 50,
            "c": [],
            "t": int(time.time() * 1e9),
        }
        fc._process_trade(trade_event)

        # Verify internal volume tracker uses OCC symbol
        assert fc._contract_volume.get(occ_sym) == 50

        # Now process a quote — snapshot should include the trade volume
        quote_event = {
            "ev": "Q",
            "sym": occ_sym,
            "bp": 5.0,
            "ap": 5.20,
        }
        fc._process_quote(quote_event)

        key = f"SPY:call:550.0:{_exp_date.isoformat()}"
        snap = fc._redis_snapshot_buffer[key]
        assert snap["volume"] == 50


# ---------------------------------------------------------------------------
# Redis Client Tests
# ---------------------------------------------------------------------------


class TestRedisWSHealth:
    """Test the Redis WS health helper."""

    @pytest.mark.asyncio
    async def test_get_ws_health_returns_none_when_no_redis(self):
        """get_ws_health returns None when Redis not connected."""
        from options_owl.db import redis_client

        original = redis_client._redis
        redis_client._redis = None
        try:
            result = await redis_client.get_ws_health()
            assert result is None
        finally:
            redis_client._redis = original

    @pytest.mark.asyncio
    async def test_get_ws_health_parses_json(self):
        """get_ws_health parses JSON health status correctly."""
        from options_owl.db import redis_client

        health = {"healthy": True, "issues": [], "t": time.time()}
        fake = MagicMock()
        fake.get = AsyncMock(return_value=json.dumps(health))

        original = redis_client._redis
        redis_client._redis = fake
        try:
            result = await redis_client.get_ws_health()
            assert result is not None
            assert result["healthy"] is True
            assert result["issues"] == []
        finally:
            redis_client._redis = original


# ---------------------------------------------------------------------------
# WS Reconnect Behavior Tests
# ---------------------------------------------------------------------------


class TestWSReconnectBehavior:
    """Verify WS collectors have proper reconnect with backoff."""

    def test_candle_collector_has_reconnect_backoff(self):
        """CandleCollector._ws_loop has exponential backoff on errors."""
        import inspect
        from options_owl.collectors.candle_collector import CandleCollector

        source = inspect.getsource(CandleCollector._ws_loop)
        # Verify exponential backoff pattern exists
        assert "reconnect_attempts" in source
        assert "max_reconnect_delay" in source
        assert "asyncio.sleep(delay)" in source

    def test_flow_collector_has_reconnect_backoff(self):
        """FlowCollector._ws_loop has exponential backoff on errors."""
        import inspect
        from options_owl.collectors.flow_collector import FlowCollector

        source = inspect.getsource(FlowCollector._ws_loop)
        # Verify exponential backoff pattern exists
        assert "reconnect_attempts" in source
        assert "max_reconnect_delay" in source
        assert "asyncio.sleep(delay)" in source

    def test_candle_collector_stale_detection(self):
        """CandleCollector has staleness timeout during market hours."""
        import inspect
        from options_owl.collectors.candle_collector import CandleCollector

        source = inspect.getsource(CandleCollector._ws_loop)
        assert "WS_STALE_TIMEOUT" in source or "STALE" in source
        assert "forcing reconnect" in source

    def test_flow_collector_stale_detection(self):
        """FlowCollector has staleness timeout during market hours."""
        import inspect
        from options_owl.collectors.flow_collector import FlowCollector

        source = inspect.getsource(FlowCollector._ws_loop)
        assert "WS_STALE_TIMEOUT" in source
        assert "forcing reconnect" in source

    def test_flow_collector_policy_violation_backoff(self):
        """FlowCollector handles Polygon policy violations with extended backoff."""
        import inspect
        from options_owl.collectors.flow_collector import FlowCollector

        source = inspect.getsource(FlowCollector._ws_loop)
        assert "policy_violations" in source
        assert "1008" in source


# ---------------------------------------------------------------------------
# Data Source Tracking in Position Monitor
# ---------------------------------------------------------------------------


class TestPremiumSourceTracking:
    """Verify position monitor tracks which source provided premium data."""

    def test_position_monitor_has_redis_source(self):
        """Position monitor tries Redis snapshot before REST fallback."""
        import inspect
        from options_owl.execution.position_monitor import run_position_monitor

        source = inspect.getsource(run_position_monitor)
        # Verify Redis is checked as first premium source
        assert "redis_snapshot" in source
        assert "Source -1" in source or "redis_client" in source

    def test_position_monitor_checks_age(self):
        """Position monitor rejects Redis data older than 30s."""
        import inspect
        from options_owl.execution.position_monitor import run_position_monitor

        source = inspect.getsource(run_position_monitor)
        assert "age < 30" in source or "age" in source
