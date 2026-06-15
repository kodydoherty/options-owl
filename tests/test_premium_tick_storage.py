"""Tests for Spec 01: Premium Tick Storage — PG schema, write, read, throttle."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPremiumTickSchema:
    def test_pg_schema_has_premium_ticks_table(self):
        """trade_premium_ticks table should be in PG schema."""
        from options_owl.db.postgres import SCHEMA_SQL

        assert "trade_premium_ticks" in SCHEMA_SQL
        assert "agent_id TEXT NOT NULL" in SCHEMA_SQL
        assert "trade_id INTEGER NOT NULL" in SCHEMA_SQL
        assert "premium REAL NOT NULL" in SCHEMA_SQL
        assert "source TEXT NOT NULL" in SCHEMA_SQL

    def test_pg_schema_has_premium_ticks_indexes(self):
        """Premium tick indexes should exist for efficient reads."""
        from options_owl.db.postgres import SCHEMA_SQL

        assert "idx_prem_ticks_trade" in SCHEMA_SQL
        assert "idx_prem_ticks_ticker" in SCHEMA_SQL


# ---------------------------------------------------------------------------
# Write/read function tests
# ---------------------------------------------------------------------------


class TestPremiumTickFunctions:
    def test_write_function_exists(self):
        """write_premium_ticks_batch should be importable."""
        from options_owl.db.postgres import write_premium_ticks_batch

        assert callable(write_premium_ticks_batch)

    def test_read_function_exists(self):
        """read_premium_ticks should be importable."""
        from options_owl.db.postgres import read_premium_ticks

        assert callable(read_premium_ticks)

    @pytest.mark.asyncio
    async def test_write_skips_when_disconnected(self):
        """write_premium_ticks_batch should no-op when PG is not connected."""
        from options_owl.db.postgres import write_premium_ticks_batch

        with patch("options_owl.db.postgres.is_connected", return_value=False):
            # Should not raise
            await write_premium_ticks_batch([
                {"agent_id": "test", "trade_id": 1, "ticker": "SPY",
                 "premium": 2.50, "source": "polygon_ws"},
            ])

    @pytest.mark.asyncio
    async def test_write_skips_empty_list(self):
        """write_premium_ticks_batch should no-op on empty list."""
        from options_owl.db.postgres import write_premium_ticks_batch

        with patch("options_owl.db.postgres.is_connected", return_value=True):
            # Should not raise even with empty list
            await write_premium_ticks_batch([])

    @pytest.mark.asyncio
    async def test_read_returns_none_when_disconnected(self):
        """read_premium_ticks should return None when PG is not connected."""
        from options_owl.db.postgres import read_premium_ticks

        with patch("options_owl.db.postgres.is_connected", return_value=False):
            result = await read_premium_ticks("owlet-kody", 225)
            assert result is None


# ---------------------------------------------------------------------------
# Position monitor integration
# ---------------------------------------------------------------------------


class TestPremiumTickInMonitor:
    def test_tick_buffer_exists(self):
        """position_monitor should have the premium tick buffer."""
        from options_owl.execution import position_monitor

        assert hasattr(position_monitor, "_premium_tick_buffer")
        assert hasattr(position_monitor, "_premium_tick_last_write")
        assert hasattr(position_monitor, "_PREMIUM_TICK_INTERVAL")
        assert hasattr(position_monitor, "_PREMIUM_TICK_FLUSH_SIZE")

    def test_tick_interval_is_15s(self):
        """Premium ticks should be throttled to every 15 seconds."""
        from options_owl.execution.position_monitor import _PREMIUM_TICK_INTERVAL

        assert _PREMIUM_TICK_INTERVAL == 15.0

    def test_flush_size_is_reasonable(self):
        """Flush threshold should batch enough ticks for efficiency."""
        from options_owl.execution.position_monitor import _PREMIUM_TICK_FLUSH_SIZE

        assert 10 <= _PREMIUM_TICK_FLUSH_SIZE <= 50

    def test_write_premium_ticks_function_exists(self):
        """_write_premium_ticks async helper should exist."""
        from options_owl.execution.position_monitor import _write_premium_ticks

        assert asyncio.iscoroutinefunction(_write_premium_ticks)

    def test_cleanup_clears_tick_state(self):
        """_cleanup_trade_state should clear premium tick tracking."""
        source = inspect.getsource(
            __import__(
                "options_owl.execution.position_monitor", fromlist=["_cleanup_trade_state"]
            )._cleanup_trade_state
        )
        assert "_premium_tick_last_write" in source

    def test_premium_source_tracked_in_cascade(self):
        """Monitor should set _prem_source for each premium source in cascade."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        # Should track source at each stage
        assert '_prem_source: str = "unknown"' in source
        assert '_prem_source = "polygon_ws"' in source
        assert '_prem_source = "polygon_rest"' in source
        assert '_prem_source = "yfinance"' in source
        assert '_prem_source = "delta_approx"' in source

    def test_tick_buffer_uses_source(self):
        """Tick buffer entries should include the source field."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        # The tick buffer append should use the tracked _prem_source
        assert '"source": _prem_source' in source

    def test_flush_is_fire_and_forget(self):
        """Premium tick flush should use asyncio.create_task (non-blocking)."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        assert "asyncio.create_task(_write_premium_ticks(" in source

    @pytest.mark.asyncio
    async def test_write_premium_ticks_handles_pg_failure(self):
        """_write_premium_ticks should silently handle PG errors."""
        from options_owl.execution.position_monitor import _write_premium_ticks

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.write_premium_ticks_batch",
                   side_effect=Exception("PG down")):
            # Should not raise
            await _write_premium_ticks([
                {"agent_id": "test", "trade_id": 1, "ticker": "SPY",
                 "premium": 2.50, "source": "polygon_ws"},
            ])

    @pytest.mark.asyncio
    async def test_write_premium_ticks_calls_pg(self):
        """_write_premium_ticks should call pg.write_premium_ticks_batch."""
        from options_owl.execution.position_monitor import _write_premium_ticks

        mock_write = AsyncMock()
        ticks = [
            {"agent_id": "owlet-kody", "trade_id": 1, "ticker": "SPY",
             "premium": 2.50, "source": "polygon_ws"},
        ]
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.write_premium_ticks_batch", mock_write):
            await _write_premium_ticks(ticks)
            mock_write.assert_awaited_once_with(ticks)
