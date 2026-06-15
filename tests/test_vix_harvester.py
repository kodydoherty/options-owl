"""Tests for Spec 02: VIX in Harvester — universe, chain skip, candle WS skip."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest


class TestVixInUniverse:
    def test_vix_in_default_universe(self):
        """VIX should be in the default HARVEST_UNIVERSE."""
        from options_owl import harvester

        # Check the source code default string (not runtime value which depends on env)
        source = inspect.getsource(harvester)
        assert "VIX" in source
        # Specifically in the default universe string
        assert "MSTR,VIX" in source or "VIX," in source or ",VIX" in source

    def test_index_only_tickers_contains_vix(self):
        """INDEX_ONLY_TICKERS should include VIX."""
        from options_owl.harvester import INDEX_ONLY_TICKERS

        assert "VIX" in INDEX_ONLY_TICKERS


class TestVixSkipsOptionChain:
    def test_harvest_ticker_has_index_skip(self):
        """_harvest_ticker should skip chain fetch for INDEX_ONLY_TICKERS."""
        from options_owl import harvester

        source = inspect.getsource(harvester._harvest_ticker)
        assert "INDEX_ONLY_TICKERS" in source
        assert "_persist_stock_tick_only" in source

    def test_persist_stock_tick_only_exists(self):
        """_persist_stock_tick_only helper should exist."""
        from options_owl.harvester import _persist_stock_tick_only

        assert callable(_persist_stock_tick_only)

    @pytest.mark.asyncio
    async def test_persist_stock_tick_only_writes_to_pg(self):
        """_persist_stock_tick_only should call pg.write_stock_tick."""
        from options_owl.harvester import _persist_stock_tick_only

        mock_write = AsyncMock()
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.write_stock_tick", mock_write):
            await _persist_stock_tick_only("VIX", {
                "price": 18.5, "bid": 18.4, "ask": 18.6, "volume": 1000, "vwap": 18.5
            })
            mock_write.assert_awaited_once()
            call_kwargs = mock_write.call_args
            assert call_kwargs[1]["ticker"] == "VIX"
            assert call_kwargs[1]["price"] == 18.5


class TestVixYfinanceQuote:
    def test_vix_uses_caret_prefix(self):
        """_get_underlying_quote should use ^VIX for yfinance."""
        from options_owl import harvester

        source = inspect.getsource(harvester._get_underlying_quote)
        assert '"VIX"' in source
        assert "^{ticker}" in source or "^VIX" in source


class TestVixNotInCandleWS:
    def test_candle_ws_skips_index_tickers(self):
        """CandleCollector WS subscription should skip VIX and other indices."""
        from options_owl.collectors import candle_collector

        source = inspect.getsource(candle_collector.CandleCollector._ws_loop)
        assert "_INDEX_TICKERS" in source
        assert "VIX" in source

    def test_candle_ws_index_set_contents(self):
        """_INDEX_TICKERS in WS loop should contain VIX."""
        from options_owl.collectors import candle_collector

        source = inspect.getsource(candle_collector.CandleCollector._ws_loop)
        assert '"VIX"' in source
