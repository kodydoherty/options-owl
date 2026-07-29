"""Tests for the event-driven monitor (P2/P3) — the SELL PATH, so the focus is concurrency safety.

The whole design rests on one invariant: the poll loop and the event task can NEVER double-close a trade.
Both funnel through the locked, dedup-guarded `_finalize_full_close`. These verify that dedup, that the
per-tick evaluator uses the shared bridge + chokepoint, never raises, and is fully inert when disabled.
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import options_owl.execution.position_monitor as pm


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    pm._trade_close_locks.clear()
    yield
    pm._trade_close_locks.clear()


class TestCloseDedup:
    @pytest.mark.asyncio
    async def test_skips_inner_when_already_closed(self, monkeypatch):
        """If the trade is no longer open (other path closed it), the locked wrapper must NOT close again."""
        inner = AsyncMock(return_value=True)
        monkeypatch.setattr(pm, "_finalize_full_close_inner", inner)
        monkeypatch.setattr(pm, "_trade_is_open", AsyncMock(return_value=False))  # already closed
        r = await pm._finalize_full_close(MagicMock(), {"id": 1, "ticker": "SPY"}, 100.0, 2.0, "x", ":m:", None)
        assert r is True
        inner.assert_not_awaited()  # deduped — no double-close

    @pytest.mark.asyncio
    async def test_closes_when_still_open(self, monkeypatch):
        inner = AsyncMock(return_value=True)
        monkeypatch.setattr(pm, "_finalize_full_close_inner", inner)
        monkeypatch.setattr(pm, "_trade_is_open", AsyncMock(return_value=True))
        r = await pm._finalize_full_close(MagicMock(), {"id": 2, "ticker": "SPY"}, 100.0, 2.0, "x", ":m:", None)
        assert r is True
        inner.assert_awaited_once()

    def test_per_trade_lock_reused(self):
        a = pm._get_trade_close_lock(7)
        b = pm._get_trade_close_lock(7)
        c = pm._get_trade_close_lock(8)
        assert a is b and a is not c  # same lock per trade_id, distinct across trades


class TestEvaluateOnTick:
    def _bridge(self, reason):
        b = MagicMock()
        b.evaluate = MagicMock(return_value=(reason, "desc"))
        return b

    @pytest.mark.asyncio
    async def test_no_bridge_no_close(self, monkeypatch):
        monkeypatch.setattr(pm, "_v5_bridge", None)
        fin = AsyncMock()
        monkeypatch.setattr(pm, "_finalize_full_close", fin)
        r = await pm._evaluate_trade_on_tick(MagicMock(), {"id": 1, "ticker": "SPY"}, 2.0, 1.9, 2.1,
                                             MagicMock(), None, ":m:")
        assert r is False
        fin.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fsm_hold_no_close(self, monkeypatch):
        monkeypatch.setattr(pm, "_v5_bridge", self._bridge(None))  # FSM says HOLD
        fin = AsyncMock()
        monkeypatch.setattr(pm, "_finalize_full_close", fin)
        ms = MagicMock(); ms.get_price = AsyncMock(return_value=100.0)
        r = await pm._evaluate_trade_on_tick(MagicMock(), {"id": 1, "ticker": "SPY"}, 2.0, 1.9, 2.1,
                                             ms, None, ":m:")
        assert r is False
        fin.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fsm_exit_closes_via_chokepoint(self, monkeypatch):
        monkeypatch.setattr(pm, "_v5_bridge", self._bridge("premium_hardstop"))  # FSM fires
        fin = AsyncMock(return_value=True)
        monkeypatch.setattr(pm, "_finalize_full_close", fin)  # the locked/deduped chokepoint
        ms = MagicMock(); ms.get_price = AsyncMock(return_value=100.0)
        r = await pm._evaluate_trade_on_tick(MagicMock(), {"id": 1, "ticker": "SPY"}, 1.5, 1.4, 1.6,
                                             ms, None, ":m:")
        assert r is True
        fin.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_underlying_no_close(self, monkeypatch):
        monkeypatch.setattr(pm, "_v5_bridge", self._bridge("premium_hardstop"))
        fin = AsyncMock()
        monkeypatch.setattr(pm, "_finalize_full_close", fin)
        ms = MagicMock(); ms.get_price = AsyncMock(return_value=None)  # WS cache miss
        r = await pm._evaluate_trade_on_tick(MagicMock(), {"id": 1, "ticker": "SPY"}, 1.5, 1.4, 1.6,
                                             ms, None, ":m:")
        assert r is False
        fin.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_raises(self, monkeypatch):
        b = MagicMock(); b.evaluate = MagicMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(pm, "_v5_bridge", b)
        ms = MagicMock(); ms.get_price = AsyncMock(return_value=100.0)
        r = await pm._evaluate_trade_on_tick(MagicMock(), {"id": 1, "ticker": "SPY"}, 1.5, 1.4, 1.6,
                                             ms, None, ":m:")
        assert r is False  # swallowed


class TestEventTaskGating:
    @pytest.mark.asyncio
    async def test_disabled_returns_immediately(self):
        pt = SimpleNamespace(settings=SimpleNamespace(ENABLE_EVENT_DRIVEN_MONITOR=False))
        # returns without subscribing / looping
        await pm._run_event_driven_exits(pt, MagicMock(), None, ":m:")


class TestSourceSafety:
    def test_event_path_uses_locked_chokepoint(self):
        """The per-tick evaluator MUST close via _finalize_full_close (locked+deduped), never the inner."""
        src = inspect.getsource(pm._evaluate_trade_on_tick)
        assert "_finalize_full_close(" in src
        assert "_finalize_full_close_inner" not in src

    def test_wrapper_locks_and_rechecks(self):
        src = inspect.getsource(pm._finalize_full_close)
        assert "_get_trade_close_lock" in src and "_trade_is_open" in src

    def test_monitor_launches_event_task_gated(self):
        src = inspect.getsource(pm.run_position_monitor)
        assert "_run_event_driven_exits" in src
        assert "ENABLE_EVENT_DRIVEN_MONITOR" in src
