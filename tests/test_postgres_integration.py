"""Tests for PostgreSQL dual-write layer and signal consumer.

Uses mocks — does not require a running Postgres instance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.db import postgres as pg


class TestPostgresModule:
    """Tests for options_owl.db.postgres module."""

    def setup_method(self):
        pg._pool = None

    def test_is_connected_false_initially(self):
        assert pg.is_connected() is False

    @pytest.mark.asyncio
    async def test_write_trade_open_returns_none_when_disconnected(self):
        result = await pg.write_trade_open("test_agent", 1, {"ticker": "SPY"})
        assert result is None

    @pytest.mark.asyncio
    async def test_write_trade_close_returns_false_when_disconnected(self):
        result = await pg.write_trade_close(
            "test_agent", 1, 2.50, "scalp_trail", 100.0, 10.0, 30.0,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_write_trade_event_returns_none_when_disconnected(self):
        result = await pg.write_trade_event("test_agent", 1, "entry", {"foo": "bar"})
        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_signals_returns_empty_when_disconnected(self):
        result = await pg.get_pending_signals("test_agent")
        assert result == []

    @pytest.mark.asyncio
    async def test_emit_ml_signal_returns_none_when_disconnected(self):
        result = await pg.emit_ml_signal({"ticker": "SPY", "direction": "CALL"})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_agent_state_noop_when_disconnected(self):
        # Should not raise
        await pg.update_agent_state("test_agent", 10000.0)

    @pytest.mark.asyncio
    async def test_write_trade_open_with_mock_pool(self):
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)

        # Patch the acquire context manager at the module level
        pg._pool = MagicMock()  # just needs to be truthy for is_connected()

        with patch.object(pg, "fetchval", new_callable=AsyncMock, return_value=42):
            result = await pg.write_trade_open("owlet_kody", 123, {
                "ticker": "SPY",
                "direction": "CALL",
                "sentiment": "bullish",
                "score": 95,
                "strength": "strong",
                "bot_source": "ml_sourcing",
                "entry_price": 550.0,
                "strike": 550.0,
                "option_type": "call",
                "contracts": 5,
                "premium_per_contract": 2.50,
                "total_cost": 1250.0,
                "opened_at": datetime.now(),
            })

        assert result == 42
        pg._pool = None

    @pytest.mark.asyncio
    async def test_write_trade_close_with_mock_pool(self):
        pg._pool = MagicMock()

        with patch.object(pg, "execute", new_callable=AsyncMock, return_value="UPDATE 1"):
            result = await pg.write_trade_close(
                "owlet_kody", 123, 3.50, "scalp_trail",
                100.0, 40.0, 25.0, "ai", 4.0, 60.0,
            )

        assert result is True
        pg._pool = None


class TestSignalConsumer:
    """Tests for the signal consumer bridge."""

    def test_score_to_strength(self):
        from options_owl.collectors.signal_consumer import _score_to_strength
        from options_owl.models.signals import SignalStrength

        assert _score_to_strength(160) == SignalStrength.ELITE
        assert _score_to_strength(130) == SignalStrength.STRONG
        assert _score_to_strength(95) == SignalStrength.GOOD
        assert _score_to_strength(80) == SignalStrength.MODERATE
        assert _score_to_strength(50) == SignalStrength.MARGINAL

    @pytest.mark.asyncio
    async def test_poll_and_route_no_signals(self):
        from options_owl.collectors.signal_consumer import _poll_and_route
        from options_owl.config.settings import Settings

        settings = Settings(AGENT_ID="test_bot", MIN_SCORE=78)
        paper_trader = MagicMock()

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock, return_value=[]):
            await _poll_and_route(paper_trader, settings, "test_bot")

        # No signals = no trades attempted
        paper_trader.evaluate_and_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_and_route_with_signal(self):
        from options_owl.collectors.signal_consumer import _poll_and_route
        from options_owl.config.settings import Settings

        settings = Settings(AGENT_ID="test_bot", MIN_SCORE=78)
        paper_trader = AsyncMock()
        paper_trader.evaluate_and_trade = AsyncMock(return_value={"trade_id": 99})

        mock_signal = {
            "id": 1,
            "ticker": "SPY",
            "direction": "CALL",
            "score": 85,
            "ml_confidence": 0.75,
            "ml_threshold": 0.5,
            "premium": 2.50,
            "strike": 550.0,
            "expiry_date": "2026-05-21",
        }

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock, return_value=[mock_signal]), \
             patch("options_owl.db.postgres.mark_signal_consumed", new_callable=AsyncMock):
            await _poll_and_route(paper_trader, settings, "test_bot")

        paper_trader.evaluate_and_trade.assert_called_once()
        call_args = paper_trader.evaluate_and_trade.call_args
        trade_signal = call_args[0][0]
        assert trade_signal.ticker == "SPY"
        assert trade_signal.score == 85
        assert trade_signal.bot_source.value == "ml_sourcing"

    @pytest.mark.asyncio
    async def test_poll_and_route_skips_low_score(self):
        from options_owl.collectors.signal_consumer import _poll_and_route
        from options_owl.config.settings import Settings

        settings = Settings(AGENT_ID="test_bot", MIN_SCORE=78)
        paper_trader = AsyncMock()

        mock_signal = {
            "id": 1,
            "ticker": "SPY",
            "direction": "CALL",
            "score": 50,  # below MIN_SCORE
            "ml_confidence": 0.3,
            "premium": 2.50,
            "strike": 550.0,
            "expiry_date": None,
        }

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock, return_value=[mock_signal]), \
             patch("options_owl.db.postgres.mark_signal_consumed", new_callable=AsyncMock):
            await _poll_and_route(paper_trader, settings, "test_bot")

        paper_trader.evaluate_and_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_consumer_disabled_without_postgres(self):
        from options_owl.collectors.signal_consumer import run_signal_consumer
        from options_owl.config.settings import Settings

        settings = Settings(ENABLE_POSTGRES=False)
        paper_trader = MagicMock()

        # Should return immediately (not loop forever)
        await run_signal_consumer(paper_trader, settings)


class TestBotSourceEnum:
    """Test that ML_SOURCING was added to BotSource."""

    def test_ml_sourcing_exists(self):
        from options_owl.models.signals import BotSource
        assert BotSource.ML_SOURCING.value == "ml_sourcing"
