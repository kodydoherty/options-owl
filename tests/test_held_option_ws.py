"""Tests for the held-contract options-WS feed (real-time exit premiums).

Covers the pure/orchestration logic without a live Polygon WS: contract-key parsing, the harvester
collector's subscribe/unsubscribe diffing + fresh-quote publishing (mocked stream + Redis), and the
settings defaults. The feed is strictly additive + flag-gated — these lock that it no-ops when off and
never blows up on malformed input.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.collectors.held_option_ws import _parse_contract_key, run_held_option_ws
from options_owl.config.settings import Settings


class TestParseContractKey:
    def test_valid_call(self):
        assert _parse_contract_key("SPY:call:500.0:2026-07-22") == ("SPY", "call", 500.0, "2026-07-22")

    def test_valid_put_lowercases_type_uppercases_ticker(self):
        assert _parse_contract_key("qqq:PUT:400:2026-07-22") == ("QQQ", "put", 400.0, "2026-07-22")

    def test_bad_field_count(self):
        assert _parse_contract_key("SPY:call:500") is None
        assert _parse_contract_key("SPY:call:500:exp:extra") is None

    def test_bad_strike(self):
        assert _parse_contract_key("SPY:call:abc:2026-07-22") is None

    def test_bad_type(self):
        assert _parse_contract_key("SPY:straddle:500:2026-07-22") is None

    def test_empty_fields(self):
        assert _parse_contract_key(":call:500:2026-07-22") is None


class TestSettings:
    def test_disabled_by_default(self):
        assert Settings().ENABLE_HELD_OPTION_WS is False

    def test_refresh_default(self):
        assert Settings().HELD_OPTION_WS_REFRESH_SEC == 3.0


def _settings(**kw):
    s = MagicMock()
    s.ENABLE_HELD_OPTION_WS = kw.get("ENABLE_HELD_OPTION_WS", True)
    s.POLYGON_API_KEY = kw.get("POLYGON_API_KEY", "key")
    s.HELD_OPTION_WS_REFRESH_SEC = kw.get("HELD_OPTION_WS_REFRESH_SEC", 3.0)
    s.model_copy = lambda update=None: s
    return s


class TestRunHeldOptionWs:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self):
        # returns immediately, never constructs a stream
        with patch("options_owl.collectors.held_option_ws.MarketDataStream") as MS:
            await run_held_option_ws(_settings(ENABLE_HELD_OPTION_WS=False))
            MS.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_without_polygon_key(self):
        with patch("options_owl.collectors.held_option_ws.MarketDataStream") as MS:
            await run_held_option_ws(_settings(POLYGON_API_KEY=""))
            MS.assert_not_called()

    def _mock_stream(self, cache):
        from options_owl.collectors.market_data_stream import MarketDataStream as RealMS
        stream = MagicMock()
        stream.start = AsyncMock(); stream.stop = AsyncMock()
        stream.subscribe_option = AsyncMock(); stream.unsubscribe_option = AsyncMock()
        stream._option_cache = cache
        return stream, RealMS.build_option_contract_ticker

    async def _drive(self, stream, real_build, held, pub):
        rc_mod = __import__("options_owl.db.redis_client", fromlist=["x"])
        stop = asyncio.Event()

        async def _run():
            with patch("options_owl.collectors.held_option_ws.MarketDataStream") as MS, \
                 patch.object(rc_mod, "get_all_held_contracts", AsyncMock(return_value=held)), \
                 patch.object(rc_mod, "publish_optquote", pub):
                MS.return_value = stream
                MS.build_option_contract_ticker = staticmethod(real_build)
                await run_held_option_ws(_settings(HELD_OPTION_WS_REFRESH_SEC=0.01), stop)

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    @pytest.mark.asyncio
    async def test_subscribes_held_and_publishes_fresh_quote(self):
        import time as _t
        from options_owl.collectors.market_data_stream import MarketDataStream as RealMS
        sym = RealMS.build_option_contract_ticker("SPY", 500.0, "2026-07-22", "call")
        stream, real_build = self._mock_stream({sym: (2.35, _t.time())})  # fresh
        pub = AsyncMock()
        await self._drive(stream, real_build, {"SPY:call:500.0:2026-07-22"}, pub)
        stream.subscribe_option.assert_awaited()
        pub.assert_awaited()
        assert pub.await_args.kwargs["mid"] == 2.35

    @pytest.mark.asyncio
    async def test_stale_quote_not_published(self):
        from options_owl.collectors.market_data_stream import MarketDataStream as RealMS
        sym = RealMS.build_option_contract_ticker("SPY", 500.0, "2026-07-22", "call")
        stream, real_build = self._mock_stream({sym: (2.35, 0.0)})  # ts=0 → ancient → stale
        pub = AsyncMock()
        await self._drive(stream, real_build, {"SPY:call:500.0:2026-07-22"}, pub)
        pub.assert_not_awaited()  # stale → never published (monitor falls back to snapshot)
