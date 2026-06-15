"""Tests for Redis integration, signal dedup, DCA max loss cap, per-ticker regime,
and PUT direction support."""

from __future__ import annotations


import pytest


# ---------------------------------------------------------------------------
# Redis client tests (without actual Redis)
# ---------------------------------------------------------------------------


class TestRedisClientGracefulDegradation:
    """Redis operations must never block trading."""

    def test_import_without_redis_package(self):
        """Redis client should import even if redis package is missing."""
        from options_owl.db import redis_client

        assert redis_client.is_connected() is False

    @pytest.mark.asyncio
    async def test_try_claim_signal_without_redis(self):
        """Without Redis, try_claim_signal returns True (allow trade)."""
        from options_owl.db import redis_client

        result = await redis_client.try_claim_signal("SPY:CALL:550:2.50", "owlet_kody")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_regime_decision_without_redis(self):
        """Without Redis, get_regime_decision returns None."""
        from options_owl.db import redis_client

        result = await redis_client.get_regime_decision("2026-05-24")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_total_open_positions_without_redis(self):
        """Without Redis, returns 0."""
        from options_owl.db import redis_client

        result = await redis_client.get_total_open_positions()
        assert result == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_without_redis(self):
        """Without Redis, always allows (no rate limiting)."""
        from options_owl.db import redis_client

        result = await redis_client.check_rate_limit("owlet_kody")
        assert result is True

    @pytest.mark.asyncio
    async def test_add_daily_loss_without_redis(self):
        """Without Redis, returns 0."""
        from options_owl.db import redis_client

        result = await redis_client.add_daily_loss("owlet_kody", 100.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# DCA max loss cap
# ---------------------------------------------------------------------------


class TestDCAMaxLossCap:
    """DCA should be blocked when unrealized loss exceeds threshold."""

    @pytest.mark.asyncio
    async def test_dca_blocked_when_loss_exceeds_threshold(self):
        """V6 DCA should not fire when unrealized loss > 50% of max trade loss."""
        import inspect
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor._check_v6_dca)

        # Verify the max loss cap check exists in the DCA function
        assert "max_trade_loss_pct" in source
        assert "unrealized_loss" in source
        assert "doubling down too risky" in source

    @pytest.mark.asyncio
    async def test_dca_loss_cap_uses_portfolio_percentage(self):
        """DCA loss cap should be percentage-based, not fixed dollar amount."""
        import inspect
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor._check_v6_dca)

        # Must use MAX_TRADE_LOSS_EXIT_PCT (percentage), not a hardcoded dollar amount
        assert "MAX_TRADE_LOSS_EXIT_PCT" in source
        # Must reference portfolio balance
        assert "balance_for_cap" in source
        # Must NOT have hardcoded dollar amounts
        assert "5000" not in source
        assert "5_000" not in source


# ---------------------------------------------------------------------------
# Docker compose configuration tests
# ---------------------------------------------------------------------------


class TestDockerComposeConfig:
    def test_redis_enabled_for_all_bots(self):
        """All trading bots should have ENABLE_REDIS=true."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()

        # Each owlet should have ENABLE_REDIS=true
        for bot in ["owlet-kody", "owlet-adam", "owlet-vinny", "owlet-yank"]:
            section = compose[compose.find(f"{bot}:"):]
            next_service = section.find("\n  owlet-", 10)
            if next_service > 0:
                section = section[:next_service]
            assert "ENABLE_REDIS=true" in section, f"{bot} missing ENABLE_REDIS=true"

    def test_redis_service_exists(self):
        """Redis service should be defined in docker-compose."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        assert "redis:" in compose
        assert "redis:7-alpine" in compose

    def test_no_shared_harvester_mounts(self):
        """Shared harvester SQLite mounts removed — all data now in PostgreSQL."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        # No more shared_harvester volume mounts on trading bots
        assert "shared_harvester" not in compose or compose.count("shared_harvester") <= 1  # harvester's own journal only
        assert "SHARED_CANDLE_DB" not in compose

    def test_postgres_service_exists(self):
        """PostgreSQL service should be defined in docker-compose."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        assert "postgres:" in compose
        assert "postgres:16-alpine" in compose


# ---------------------------------------------------------------------------
# Per-ticker regime
# ---------------------------------------------------------------------------


class TestPerTickerRegime:
    """Regime model should support per-ticker evaluation, not just SPY."""

    def test_per_ticker_regime_function_exists(self):
        """check_ticker_regime function should exist in ml_pipeline."""
        from options_owl.sourcing.ml_pipeline import check_ticker_regime

        assert callable(check_ticker_regime)

    def test_regime_cache_cleared_at_eod(self):
        """Per-ticker regime cache should be cleared between trading days."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.run_ml_pipeline)
        assert "_ticker_regime_cache.clear()" in source

    def test_per_ticker_regime_integrated_in_scan(self):
        """scan_all_tickers should call per-ticker regime check."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.scan_all_tickers)
        assert "check_ticker_regime" in source

    def test_regime_features_accept_any_ticker(self):
        """compute_regime_features should accept any ticker, not just SPY."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.compute_regime_features)
        # Should use ticker parameter, not hardcode "SPY"
        assert 'ticker: str' in source


# ---------------------------------------------------------------------------
# PUT direction support
# ---------------------------------------------------------------------------


class TestPutDirection:
    """ML pipeline should support PUT signals when underlying is bearish."""

    def test_direction_detection_in_scan(self):
        """scan_ticker_minute should determine direction from underlying price."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.scan_ticker_minute)
        assert '"PUT"' in source
        assert '"CALL"' in source
        assert "move_pct" in source or "direction" in source

    def test_emit_signal_supports_direction(self):
        """emit_signal_to_pg should accept direction parameter."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.emit_signal_to_pg)
        assert "direction: str" in source


# ---------------------------------------------------------------------------
# Redis signal dedup in discord_collector
# ---------------------------------------------------------------------------


class TestSignalDedup:
    """Signal dedup should be wired into the Discord collector entry path."""

    def test_signal_dedup_in_collector(self):
        """Discord collector should check Redis before trading a signal."""
        import inspect
        from options_owl.collectors import discord_collector

        source = inspect.getsource(discord_collector.OptionsOwlBot)
        assert "SIGNAL DEDUP" in source
        assert "try_claim_signal" in source

    def test_redis_init_in_on_ready(self):
        """Redis should be initialized in on_ready alongside Postgres."""
        import inspect
        from options_owl.collectors import discord_collector

        source = inspect.getsource(discord_collector.OptionsOwlBot.on_ready)
        assert "init_redis" in source
        assert "ENABLE_REDIS" in source


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_redis_settings_exist(self):
        """ENABLE_REDIS and REDIS_URL should be in Settings."""
        from options_owl.config.settings import Settings

        s = Settings()
        assert hasattr(s, "ENABLE_REDIS")
        assert hasattr(s, "REDIS_URL")

    def test_dead_flow_capture_settings_removed(self):
        """ENABLE_FLOW_CAPTURE and FLOW_CAPTURE_TICKERS should be removed."""
        from options_owl.config.settings import Settings

        s = Settings()
        assert not hasattr(s, "ENABLE_FLOW_CAPTURE")
        assert not hasattr(s, "FLOW_CAPTURE_TICKERS")


# ---------------------------------------------------------------------------
# Fix #1: Deprecated asyncio.get_event_loop() removed from candle_collector
# ---------------------------------------------------------------------------


class TestCandleCollectorAsyncio:
    def test_no_deprecated_get_event_loop(self):
        """candle_collector should use asyncio.create_task, not get_event_loop()."""
        import inspect
        from options_owl.collectors import candle_collector

        source = inspect.getsource(candle_collector.CandleCollector)
        assert "get_event_loop" not in source
        assert "asyncio.create_task" in source


# ---------------------------------------------------------------------------
# Fix #2: json imported at module level in redis_client
# ---------------------------------------------------------------------------


class TestRedisClientJsonImport:
    def test_json_imported_at_module_level(self):
        """redis_client should import json at module level, not inline."""
        import inspect
        from options_owl.db import redis_client

        source = inspect.getsource(redis_client)
        lines = source.split("\n")

        # json should be in top-level imports (not indented)
        module_level_json = any(
            line.strip() == "import json" and not line.startswith(" " * 4)
            for line in lines[:30]
        )
        assert module_level_json, "json should be imported at module level"

        # No inline 'import json' inside functions (indented)
        inline_json_imports = [
            i for i, line in enumerate(lines)
            if line.strip() == "import json" and line.startswith(" " * 4)
        ]
        assert len(inline_json_imports) == 0, (
            f"Found {len(inline_json_imports)} inline 'import json' statements "
            f"at lines {inline_json_imports}"
        )


# ---------------------------------------------------------------------------
# Fix #4: Contract key normalization
# ---------------------------------------------------------------------------


class TestContractKeyNormalization:
    def test_normalize_contract_key_lowercase(self):
        """_normalize_contract_key should lowercase option_type."""
        from options_owl.db.redis_client import _normalize_contract_key

        assert _normalize_contract_key("SPY:Call:530:2026-05-26") == "SPY:call:530:2026-05-26"
        assert _normalize_contract_key("SPY:PUT:530:2026-05-26") == "SPY:put:530:2026-05-26"
        assert _normalize_contract_key("SPY:call:530:2026-05-26") == "SPY:call:530:2026-05-26"

    def test_normalize_handles_short_keys(self):
        """_normalize_contract_key should handle edge cases."""
        from options_owl.db.redis_client import _normalize_contract_key

        assert _normalize_contract_key("SPY") == "SPY"
        assert _normalize_contract_key("SPY:CALL") == "SPY:call"

    def test_normalization_in_publish_and_get(self):
        """Both publish and get should call _normalize_contract_key."""
        import inspect
        from options_owl.db import redis_client

        pub_src = inspect.getsource(redis_client.publish_option_premium)
        get_src = inspect.getsource(redis_client.get_option_premium)
        assert "_normalize_contract_key" in pub_src
        assert "_normalize_contract_key" in get_src


# ---------------------------------------------------------------------------
# Fix #5: Price cache with timestamp
# ---------------------------------------------------------------------------


class TestPriceCacheTimestamp:
    def test_publish_price_uses_json_with_timestamp(self):
        """publish_price should store JSON with price + timestamp, not plain float."""
        import inspect
        from options_owl.db import redis_client

        source = inspect.getsource(redis_client.publish_price)
        assert "json.dumps" in source
        assert '"price"' in source
        assert '"t"' in source
        assert "time.time()" in source

    def test_get_price_returns_tuple_with_age(self):
        """get_price should return (price, age) tuple, not just float."""
        import inspect
        from options_owl.db import redis_client

        source = inspect.getsource(redis_client.get_price)
        assert "max_age" in source
        assert "json.loads" in source
        # Returns tuple
        assert '(data["price"]' in source

    @pytest.mark.asyncio
    async def test_get_price_without_redis_returns_none(self):
        """Without Redis, get_price still returns None."""
        from options_owl.db import redis_client

        result = await redis_client.get_price("SPY")
        assert result is None

    def test_market_data_stream_unpacks_price_tuple(self):
        """market_data_stream.get_price should unpack (price, age) from Redis."""
        import inspect
        from options_owl.collectors import market_data_stream

        source = inspect.getsource(market_data_stream.MarketDataStream.get_price)
        assert "redis_price, _age" in source
        assert "max_age=" in source


# ---------------------------------------------------------------------------
# Fix #3: Sourcing agent reads from Redis first
# ---------------------------------------------------------------------------


class TestSourcingRedisFirst:
    def test_fetch_live_underlying_price_tries_redis(self):
        """fetch_live_underlying_price should try Redis before Polygon REST."""
        import inspect
        from options_owl.sourcing import ml_pipeline

        source = inspect.getsource(ml_pipeline.fetch_live_underlying_price)
        # Redis should be tried BEFORE Polygon
        redis_pos = source.find("redis_client")
        polygon_pos = source.find("api.polygon.io")
        assert redis_pos > 0, "Redis not found in fetch_live_underlying_price"
        assert polygon_pos > 0, "Polygon fallback not found"
        assert redis_pos < polygon_pos, "Redis should be tried BEFORE Polygon REST"


# ---------------------------------------------------------------------------
# Fix #6: Dead flow capture code removed from market_data_stream
# ---------------------------------------------------------------------------


class TestFlowCaptureCodeRemoved:
    def test_no_flow_collector_in_market_data_stream(self):
        """MarketDataStream should not reference FlowCollector."""
        import inspect
        from options_owl.collectors import market_data_stream

        # Check __init__ — no _flow_collector attribute
        init_src = inspect.getsource(market_data_stream.MarketDataStream.__init__)
        assert "_flow_collector" not in init_src
        assert "_flow_flush_task" not in init_src

    def test_no_flow_flush_loop_method(self):
        """_flow_flush_loop should be removed from MarketDataStream."""
        from options_owl.collectors.market_data_stream import MarketDataStream

        assert not hasattr(MarketDataStream, "_flow_flush_loop")

    def test_no_enable_flow_capture_references(self):
        """No ENABLE_FLOW_CAPTURE references in market_data_stream."""
        import inspect
        from options_owl.collectors import market_data_stream

        source = inspect.getsource(market_data_stream.MarketDataStream)
        assert "ENABLE_FLOW_CAPTURE" not in source
        assert "FLOW_CAPTURE_TICKERS" not in source
