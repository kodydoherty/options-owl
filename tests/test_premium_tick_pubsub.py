"""Phase 1 tests for the event-driven monitor's enabling layer (premium-tick pub/sub).

Additive + flag-gated: the harvester publishes held-contract premium ticks to a Redis channel so the monitor
can react per-tick. These verify the publish helper targets the right channel/payload and no-ops safely when
Redis is down, plus the FlowCollector only publishes when enabled AND the contract is held.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.config.settings import Settings


class TestPublishPremiumTick:
    @pytest.mark.asyncio
    async def test_publishes_to_channel(self, monkeypatch):
        import options_owl.db.redis_client as rc
        fake = MagicMock()
        fake.publish = AsyncMock()
        monkeypatch.setattr(rc, "_redis", fake)
        await rc.publish_premium_tick("SPY:call:500:2026-07-24", mid=2.40, bid=2.38, ask=2.42)
        fake.publish.assert_awaited_once()
        channel, payload = fake.publish.await_args.args
        assert channel == rc.PREMIUM_TICK_CHANNEL
        d = json.loads(payload)
        assert d["mid"] == 2.40 and d["bid"] == 2.38 and d["ask"] == 2.42 and "t" in d

    @pytest.mark.asyncio
    async def test_noop_when_redis_down(self, monkeypatch):
        import options_owl.db.redis_client as rc
        monkeypatch.setattr(rc, "_redis", None)
        await rc.publish_premium_tick("SPY:call:500:2026-07-24", 2.4, 2.38, 2.42)  # must not raise

    @pytest.mark.asyncio
    async def test_pubsub_none_when_redis_down(self, monkeypatch):
        import options_owl.db.redis_client as rc
        monkeypatch.setattr(rc, "_redis", None)
        assert rc.premium_tick_pubsub() is None


class TestSettings:
    def test_flags_off_by_default(self):
        s = Settings()
        assert s.ENABLE_PREMIUM_TICK_PUBLISH is False
        assert s.ENABLE_EVENT_DRIVEN_MONITOR is False


class TestFlowCollectorGating:
    def test_publish_ticks_defaults_off(self):
        from options_owl.collectors.flow_collector import FlowCollector
        fc = FlowCollector(["SPY"])
        assert fc._publish_ticks is False
        assert fc._held_contracts == set()
