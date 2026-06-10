"""Regression tests for structured Webull sell outcomes (FIX 1 + FIX 2).

FIX 1: close_webull_position must return a categorized SellResult so the monitor
only counts POSITION_NOT_FOUND toward the manual-close abandonment budget. A
transient Webull outage must NOT force-close a still-open live position as
'manual' (which previously triggered orphan recovery → +5000% garbage P&L).

FIX 2: SDK round-trips in the sell path are wrapped in asyncio.wait_for so a hung
Webull call returns gracefully (categorized as transient) instead of stalling the
monitor loop for every other open trade.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.execution.paper_trader import (
    PaperTrader,
    SellOutcome,
    SellResult,
)
from options_owl.execution.webull_executor import OrderResult


# ---------------------------------------------------------------------------
# SellResult value type
# ---------------------------------------------------------------------------

class TestSellResultType:
    def test_filled_is_truthy_and_success(self):
        r = SellResult(SellOutcome.FILLED)
        assert r.success is True
        assert bool(r) is True

    def test_failures_are_falsy(self):
        for outcome in (
            SellOutcome.POSITION_NOT_FOUND,
            SellOutcome.NOT_FILLED,
            SellOutcome.TRANSIENT_ERROR,
        ):
            r = SellResult(outcome)
            assert r.success is False
            assert bool(r) is False

    def test_backward_compat_if_result(self):
        """Partial-close callers use `if result:` — must keep working."""
        assert bool(SellResult(SellOutcome.FILLED))
        assert not bool(SellResult(SellOutcome.TRANSIENT_ERROR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trader(executor) -> PaperTrader:
    settings = MagicMock()
    settings.PAPER_TRADE = False
    settings.POLYGON_API_KEY = ""
    trader = PaperTrader.__new__(PaperTrader)  # bypass __init__/DB setup
    trader.settings = settings
    trader.webull_executor = executor
    trader.db_path = ":memory:"
    return trader


def _trade() -> dict:
    return {
        "id": 42,
        "ticker": "SPY",
        "strike": 500.0,
        "option_type": "call",
        "expiry_date": "2026-06-10",
        "contracts": 3,
        "webull_order_id": "WB123",
        "sell_retry_count": 0,
        "premium_per_contract": 1.00,
    }


async def _no_fresh_bid(self, trade):  # patch out network bid fetch
    return None


async def _noop_db(*args, **kwargs):
    return None


@pytest.fixture(autouse=True)
def _patch_io(monkeypatch):
    # Avoid real network + DB during these unit tests.
    monkeypatch.setattr(PaperTrader, "_get_fresh_option_bid", _no_fresh_bid)
    monkeypatch.setattr(
        "options_owl.execution.paper_trader._db_execute_with_retry", _noop_db
    )


# ---------------------------------------------------------------------------
# FIX 1: outcome categorization
# ---------------------------------------------------------------------------

class TestSellOutcomeCategorization:
    @pytest.mark.asyncio
    async def test_no_executor_returns_filled(self):
        trader = _make_trader(None)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.FILLED

    @pytest.mark.asyncio
    async def test_position_not_found_categorized(self):
        executor = MagicMock()
        executor.sell_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="No Webull position found for SPY $500 CALL — nothing to close",
            fill_status="UNKNOWN",
        ))
        trader = _make_trader(executor)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.POSITION_NOT_FOUND

    @pytest.mark.asyncio
    async def test_position_lookup_failed_categorized(self):
        executor = MagicMock()
        executor.sell_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="Position lookup failed for SPY $500 CALL — blocked sell",
        ))
        trader = _make_trader(executor)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.POSITION_NOT_FOUND

    @pytest.mark.asyncio
    async def test_order_rejected_is_transient(self):
        executor = MagicMock()
        executor.sell_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="OAUTH_OPENAPI rate limited",
            fill_status="FAILED",
        ))
        trader = _make_trader(executor)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.TRANSIENT_ERROR

    @pytest.mark.asyncio
    async def test_not_filled_is_not_position_gone(self):
        executor = MagicMock()
        executor.sell_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="Order not filled after 10s (status=SUBMITTED), cancelled",
            fill_status="SUBMITTED",
        ))
        trader = _make_trader(executor)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.NOT_FILLED
        # Must NOT be counted as a manual close.
        assert result.outcome is not SellOutcome.POSITION_NOT_FOUND

    @pytest.mark.asyncio
    async def test_exception_is_transient(self):
        executor = MagicMock()
        executor.sell_option = AsyncMock(side_effect=ValueError("no active connection"))
        trader = _make_trader(executor)
        result = await trader.close_webull_position(_trade(), 1.00)
        assert result.outcome is SellOutcome.TRANSIENT_ERROR


# ---------------------------------------------------------------------------
# FIX 2: hung SDK call returns gracefully
# ---------------------------------------------------------------------------

class TestSellTimeout:
    @pytest.mark.asyncio
    async def test_hung_sell_returns_transient(self):
        async def _hang(**kwargs):
            await asyncio.sleep(120)  # never returns within wrapper window
            return OrderResult(success=True)

        executor = MagicMock()
        executor.sell_option = _hang
        trader = _make_trader(executor)

        # Patch the 45s wrapper down so the test is fast but still exercises the
        # asyncio.wait_for branch.
        import options_owl.execution.paper_trader as pt

        orig_wait_for = asyncio.wait_for

        async def _fast_wait_for(coro, timeout):
            return await orig_wait_for(coro, timeout=0.05)

        pt.asyncio.wait_for = _fast_wait_for
        try:
            result = await trader.close_webull_position(_trade(), 1.00)
        finally:
            pt.asyncio.wait_for = orig_wait_for

        assert result.outcome is SellOutcome.TRANSIENT_ERROR
        assert "timed out" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# FIX 2: SDK call sites are wrapped in asyncio.wait_for (source-level safety)
# ---------------------------------------------------------------------------

class TestSdkCallsWrapped:
    def test_close_webull_position_wraps_sdk_calls(self):
        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "asyncio.wait_for(" in src
        # sell_option, get_open_orders, get_fill_price all wrapped
        assert src.count("asyncio.wait_for(") >= 3

    def test_monitor_wraps_sync_subscribe_reconcile(self):
        import options_owl.execution.position_monitor as pm

        src = inspect.getsource(pm.run_position_monitor)
        assert "sync_portfolio_from_webull(), timeout=15" in src
        assert "subscribe_option(" in src and "timeout=15" in src

        recon_src = inspect.getsource(pm._reconcile_positions)
        assert "get_open_option_positions(), timeout=15" in recon_src
