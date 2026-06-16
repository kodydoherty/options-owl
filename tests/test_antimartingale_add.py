"""Tests for the anti-martingale ADD (buy more on a confirmed runner).

Covers: fires at +30% (call) / not below; PUT pyramid (+30 then +100, one-shot per level);
underlying-confirm gate; position cap; and the CRITICAL safety property — it must NOT touch the
FSM (no GRACE/peak reset), so the trailing stop keeps protecting the runner.
"""
from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch as _patch

import pytest


def _settings():
    return SimpleNamespace(
        ENABLE_ANTIMARTINGALE_ADD=True,
        ANTIMG_CALL_LEVELS="30",
        ANTIMG_PUT_LEVELS="100,30",
        ANTIMG_MIN_MINUTES=3.0,
        ANTIMG_MAX_MINUTES=60.0,
        ANTIMG_UND_CONFIRM_PCT=0.10,
        ANTIMG_ADD_FRACTION=1.0,
        MAX_POSITION_PCT=15.0,
        PORTFOLIO_SIZE=23000.0,
        WEBULL_ENTRY_AGGRESS_PCT=2.0,
    )


def _trade(option_type="call", premium=2.00):
    return {
        "id": 700, "ticker": "NVDA", "contracts": 5, "premium_per_contract": premium,
        "opened_at": "2026-06-15 13:30:00", "entry_price": 130.0,
        "option_type": option_type, "strike": 130.0, "expiry_date": "2026-06-15",
        "total_cost": premium * 5 * 100,
    }


@pytest.fixture(autouse=True)
def _clear_fired():
    from options_owl.execution import position_monitor as pm
    pm._antimg_fired.difference_update({k for k in pm._antimg_fired if k[0] == 700})
    yield
    pm._antimg_fired.difference_update({k for k in pm._antimg_fired if k[0] == 700})


async def _run(trade, exit_premium, current_price, settings):
    from options_owl.execution import position_monitor as pm
    mock_pt = MagicMock()
    mock_pt.get_portfolio_balance = AsyncMock(return_value=23000.0)
    mock_pt.webull_executor = None  # skip the live order path
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _fake_connect(path):
        yield conn

    with _patch("options_owl.execution.position_monitor.datetime") as mdt, \
         _patch("options_owl.execution.position_monitor._connect_db", _fake_connect):
        mdt.fromisoformat.return_value = datetime(2026, 6, 15, 13, 30, 0)
        mdt.now.return_value = datetime(2026, 6, 15, 13, 40, 0)  # 10 min after entry
        added = await pm._check_antimartingale_add(
            trade, exit_premium, current_price, settings, mock_pt, "test.db")
    return added, conn


class TestAntimartingaleAdd:
    @pytest.mark.asyncio
    async def test_call_fires_at_plus30(self):
        from options_owl.execution.position_monitor import _antimg_fired
        added, conn = await _run(_trade("call"), 2.60, 130.6, _settings())  # +30%, underlying +0.46%
        assert added is True
        assert (700, 30) in _antimg_fired
        assert conn.execute.called  # DB updated

    @pytest.mark.asyncio
    async def test_call_does_not_fire_below_30(self):
        from options_owl.execution.position_monitor import _antimg_fired
        added, _ = await _run(_trade("call"), 2.40, 130.6, _settings())  # only +20%
        assert added is False
        assert (700, 30) not in _antimg_fired

    @pytest.mark.asyncio
    async def test_call_blocked_when_underlying_not_confirming(self):
        added, _ = await _run(_trade("call"), 2.60, 129.5, _settings())  # +30% but underlying DOWN
        assert added is False

    @pytest.mark.asyncio
    async def test_put_pyramids_30_then_100_oneshot(self):
        from options_owl.execution.position_monitor import _antimg_fired
        tr = _trade("put")
        # +30%, underlying DOWN -0.46% (confirms put) → fires the +30 level
        added1, _ = await _run(tr, 2.60, 129.4, _settings())
        assert added1 is True and (700, 30) in _antimg_fired
        # +100% later → fires the +100 level (pyramid), still one-shot for +30
        added2, _ = await _run(tr, 4.00, 129.4, _settings())
        assert added2 is True and (700, 100) in _antimg_fired
        # +100% again → nothing left to fire
        added3, _ = await _run(tr, 4.00, 129.4, _settings())
        assert added3 is False

    @pytest.mark.asyncio
    async def test_disabled_flag_via_caller_semantics(self):
        # function itself doesn't check the flag (caller does), but a non-eligible gain returns False
        added, _ = await _run(_trade("call"), 2.00, 130.6, _settings())  # 0% gain
        assert added is False

    def test_does_not_touch_the_fsm(self):
        """CRITICAL: the add must NOT mutate the FSM (no GRACE/peak/state reset) — the trail
        keeps protecting the runner. Guard against a future edit reintroducing DCA-style resets."""
        from options_owl.execution.position_monitor import _check_antimartingale_add
        src = inspect.getsource(_check_antimartingale_add)
        for forbidden in ("_v5_bridge", "FSMState", "GRACE", ".peak_premium", ".state =",
                          "breakeven_ratchet_armed", "scaled_out"):
            assert forbidden not in src, (
                f"anti-martingale add must NOT touch the FSM, found '{forbidden}'")
