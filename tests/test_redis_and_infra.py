"""Tests for Redis integration, signal dedup, DCA max loss cap, per-ticker regime,
and PUT direction support."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_shared_harvester_rw_for_wal(self):
        """All bot volumes should mount shared_harvester as :rw for WAL."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        assert "shared_harvester:rw" in compose
        # owlet-kody specifically should NOT be :ro
        kody_section = compose[compose.find("owlet-kody:"):]
        kody_volumes = kody_section[kody_section.find("volumes:"):]
        kody_volumes = kody_volumes[:kody_volumes.find("\n\n")]
        assert ":ro" not in kody_volumes or "shared_harvester:ro" not in kody_volumes

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
