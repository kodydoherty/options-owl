"""Tests for Spec 04: Options Flow Capture — FlowCollector, PG schema, parsing."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# PG Schema
# ---------------------------------------------------------------------------


class TestFlowSchema:
    def test_option_flow_table_exists(self):
        """option_flow table should be in PG schema."""
        from options_owl.db.postgres import SCHEMA_SQL

        assert "option_flow" in SCHEMA_SQL
        assert "trade_size INTEGER NOT NULL" in SCHEMA_SQL
        assert "is_sweep BOOLEAN" in SCHEMA_SQL

    def test_option_flow_5m_table_exists(self):
        """option_flow_5m table should be in PG schema."""
        from options_owl.db.postgres import SCHEMA_SQL

        assert "option_flow_5m" in SCHEMA_SQL
        assert "call_volume INTEGER" in SCHEMA_SQL
        assert "net_flow_dollars REAL" in SCHEMA_SQL
        assert "sweep_ratio REAL" in SCHEMA_SQL

    def test_flow_indexes_exist(self):
        """Flow tables should have efficient indexes."""
        from options_owl.db.postgres import SCHEMA_SQL

        assert "idx_flow_ticker" in SCHEMA_SQL
        assert "idx_flow_size" in SCHEMA_SQL
        assert "idx_flow_sweep" in SCHEMA_SQL
        assert "idx_flow_5m_ticker" in SCHEMA_SQL


# ---------------------------------------------------------------------------
# PG Functions
# ---------------------------------------------------------------------------


class TestFlowPGFunctions:
    def test_write_flow_batch_exists(self):
        from options_owl.db.postgres import write_option_flow_batch
        assert callable(write_option_flow_batch)

    def test_write_flow_5m_exists(self):
        from options_owl.db.postgres import write_option_flow_5m
        assert callable(write_option_flow_5m)

    def test_read_flow_5m_exists(self):
        from options_owl.db.postgres import read_option_flow_5m
        assert callable(read_option_flow_5m)

    @pytest.mark.asyncio
    async def test_write_skips_when_disconnected(self):
        from options_owl.db.postgres import write_option_flow_batch

        with patch("options_owl.db.postgres.is_connected", return_value=False):
            await write_option_flow_batch([{"ticker": "SPY"}])  # should not raise

    @pytest.mark.asyncio
    async def test_read_returns_none_when_disconnected(self):
        from options_owl.db.postgres import read_option_flow_5m

        with patch("options_owl.db.postgres.is_connected", return_value=False):
            result = await read_option_flow_5m("SPY")
            assert result is None


# ---------------------------------------------------------------------------
# OCC Ticker Parsing
# ---------------------------------------------------------------------------


class TestOCCParsing:
    def test_parse_call(self):
        from options_owl.collectors.flow_collector import parse_occ_ticker

        result = parse_occ_ticker("O:SPY260526C00530000")
        assert result is not None
        assert result["ticker"] == "SPY"
        assert result["expiry"] == "2026-05-26"
        assert result["type"] == "call"
        assert result["strike"] == 530.0

    def test_parse_put(self):
        from options_owl.collectors.flow_collector import parse_occ_ticker

        result = parse_occ_ticker("O:TSLA260526P00180000")
        assert result is not None
        assert result["ticker"] == "TSLA"
        assert result["type"] == "put"
        assert result["strike"] == 180.0

    def test_parse_fractional_strike(self):
        from options_owl.collectors.flow_collector import parse_occ_ticker

        result = parse_occ_ticker("O:AAPL260526C00192500")
        assert result is not None
        assert result["strike"] == 192.5

    def test_parse_invalid(self):
        from options_owl.collectors.flow_collector import parse_occ_ticker

        assert parse_occ_ticker("invalid") is None
        assert parse_occ_ticker("") is None
        assert parse_occ_ticker("O:") is None


# ---------------------------------------------------------------------------
# Aggressor Detection
# ---------------------------------------------------------------------------


class TestAggressorDetection:
    def test_buyer_at_ask(self):
        from options_owl.collectors.flow_collector import detect_aggressor

        above, below = detect_aggressor(2.50, 2.40, 2.50)
        assert above is True
        assert below is False

    def test_seller_at_bid(self):
        from options_owl.collectors.flow_collector import detect_aggressor

        above, below = detect_aggressor(2.40, 2.40, 2.50)
        assert above is False
        assert below is True

    def test_neutral_between(self):
        from options_owl.collectors.flow_collector import detect_aggressor

        above, below = detect_aggressor(2.45, 2.40, 2.50)
        assert above is False
        assert below is False

    def test_no_nbbo(self):
        from options_owl.collectors.flow_collector import detect_aggressor

        above, below = detect_aggressor(2.45, 0, 0)
        assert above is None
        assert below is None


# ---------------------------------------------------------------------------
# FlowCollector filtering
# ---------------------------------------------------------------------------


class TestFlowCollectorFiltering:
    def test_filters_small_trades(self):
        """Trades with size < MIN_FLOW_SIZE should be dropped."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        # Simulate a small trade event
        today = date.today().strftime("%y%m%d")
        event = {
            "ev": "OT",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 5,  # below MIN_FLOW_SIZE of 10
            "c": [],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc._process_trade(event)
        assert len(fc._trade_buffer) == 0

    def test_accepts_large_trades(self):
        """Trades with size >= MIN_FLOW_SIZE should be buffered."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        event = {
            "ev": "OT",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc._process_trade(event)
        assert len(fc._trade_buffer) == 1
        assert fc._trade_buffer[0].trade_size == 25

    def test_detects_sweeps(self):
        """Condition code 12 should set is_sweep=True."""
        from options_owl.collectors.flow_collector import FlowCollector, SWEEP_CONDITION

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        event = {
            "ev": "OT",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 50,
            "c": [SWEEP_CONDITION, 41],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc._process_trade(event)
        assert len(fc._trade_buffer) == 1
        assert fc._trade_buffer[0].is_sweep is True

    def test_filters_expired_contracts(self):
        """Trades on expired contracts (DTE < 0) should be dropped."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        event = {
            "ev": "OT",
            "sym": "O:SPY250101C00530000",  # 2025-01-01 — expired
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc._process_trade(event)
        assert len(fc._trade_buffer) == 0

    def test_filters_far_dte(self):
        """Trades on contracts with DTE > MAX_DTE should be dropped."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        # 60 days out
        from datetime import timedelta
        far_date = (date.today() + timedelta(days=60)).strftime("%y%m%d")
        event = {
            "ev": "OT",
            "sym": f"O:SPY{far_date}C00530000",
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc._process_trade(event)
        assert len(fc._trade_buffer) == 0

    def test_skips_vix(self):
        """FlowCollector should exclude VIX from tickers."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY", "VIX", "QQQ"])
        assert "VIX" not in fc._tickers
        assert "SPY" in fc._tickers
        assert "QQQ" in fc._tickers


# ---------------------------------------------------------------------------
# 5-minute aggregation
# ---------------------------------------------------------------------------


class TestFlowAggregation:
    def test_5m_bar_aggregation(self):
        """Multiple trades should aggregate into a single 5m bar."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        ts = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)

        # 3 call trades
        for size in [20, 30, 50]:
            fc._process_trade({
                "ev": "OT",
                "sym": f"O:SPY{today}C00530000",
                "p": 2.50,
                "s": size,
                "c": [12] if size == 50 else [],
                "t": ts,
            })

        # 1 put trade
        fc._process_trade({
            "ev": "OT",
            "sym": f"O:SPY{today}P00525000",
            "p": 1.80,
            "s": 40,
            "c": [],
            "t": ts,
        })

        assert len(fc._trade_buffer) == 4
        assert len(fc._flow_bars) == 1

        bar = list(fc._flow_bars.values())[0]
        assert bar.call_volume == 100  # 20 + 30 + 50
        assert bar.call_sweeps == 1   # only the 50-contract trade had sweep
        assert bar.call_large_trades == 1  # 50 >= LARGE_TRADE_SIZE
        assert bar.put_volume == 40
        assert bar.put_sweeps == 0

    def test_5m_bar_to_dict(self):
        """FlowBar5m.to_dict() should compute derived fields."""
        from options_owl.collectors.flow_collector import FlowBar5m

        bar = FlowBar5m(
            ticker="SPY",
            bar_time=datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc),
            call_volume=100, call_value=25000.0,
            call_total_count=3, call_buyer_count=2, call_sweeps=1,
            put_volume=40, put_value=7200.0,
            put_total_count=1, put_buyer_count=0,
        )
        d = bar.to_dict()
        assert d["call_put_ratio"] == 100 / 40
        assert d["net_flow_dollars"] == 25000.0 - 7200.0
        assert d["call_buyer_aggressor_pct"] == pytest.approx(66.67, abs=0.1)
        assert d["sweep_ratio"] == pytest.approx(1 / 4, abs=0.01)


# ---------------------------------------------------------------------------
# Harvester integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_flow_collector_in_harvester(self):
        """Harvester should initialize FlowCollector with FLOW_POLYGON_API_KEY."""
        from options_owl import harvester

        source = inspect.getsource(harvester.run_harvester)
        assert "FlowCollector" in source
        assert "FLOW_POLYGON_API_KEY" in source
        assert "flow_collector.flush()" in source
        assert "flow_collector.stop_ws()" in source

    def test_flow_collector_import_in_harvester(self):
        """FlowCollector should be imported in harvester."""
        from options_owl import harvester

        source = inspect.getsource(harvester)
        assert "from options_owl.collectors.flow_collector import FlowCollector" in source

    def test_flow_capture_not_in_market_data_stream(self):
        """Flow capture is centralized on harvester — not piggybacked on bot WS."""
        from options_owl.collectors import market_data_stream

        init_src = inspect.getsource(market_data_stream.MarketDataStream.__init__)
        assert "_flow_collector" not in init_src

    def test_process_event_method_exists(self):
        """FlowCollector should have a process_event method for external WS."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        assert hasattr(fc, "process_event")
        assert callable(fc.process_event)

    def test_process_event_forwards_trades(self):
        """process_event should handle T events (from /options WS)."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        event = {
            "ev": "T",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        }
        fc.process_event(event)
        assert len(fc._trade_buffer) == 1


# ---------------------------------------------------------------------------
# Flush to PG
# ---------------------------------------------------------------------------


class TestFlowFlush:
    @pytest.mark.asyncio
    async def test_flush_writes_trades_and_bars(self):
        """flush() should write both trade buffer and 5m bars to PG."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        ts = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)

        fc._process_trade({
            "ev": "OT",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": ts,
        })

        mock_flow_write = AsyncMock()
        mock_5m_write = AsyncMock()
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.write_option_flow_batch", mock_flow_write), \
             patch("options_owl.db.postgres.write_option_flow_5m", mock_5m_write):
            count = await fc.flush()

        assert count == 1
        mock_flow_write.assert_awaited_once()
        mock_5m_write.assert_awaited_once()
        # Buffer should be cleared
        assert len(fc._trade_buffer) == 0
        assert len(fc._flow_bars) == 0

    @pytest.mark.asyncio
    async def test_flush_handles_pg_failure(self):
        """flush() should handle PG errors gracefully."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        today = date.today().strftime("%y%m%d")
        ts = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)

        fc._process_trade({
            "ev": "OT",
            "sym": f"O:SPY{today}C00530000",
            "p": 2.50,
            "s": 25,
            "c": [],
            "t": ts,
        })

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.write_option_flow_batch",
                   side_effect=Exception("PG down")):
            count = await fc.flush()  # should not raise
            assert count == 0

    @pytest.mark.asyncio
    async def test_flush_empty_is_noop(self):
        """flush() with no data should return 0."""
        from options_owl.collectors.flow_collector import FlowCollector

        fc = FlowCollector(["SPY"])
        count = await fc.flush()
        assert count == 0
