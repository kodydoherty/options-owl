"""B1 — per-cycle FSM telemetry snapshot for the dashboard timeline.

The exit engine already computes state/gain/peak/reason every cycle; B1 stashes a cheap
in-memory snapshot (monitor_bridge._LAST_FSM_SNAPSHOT) that position_monitor piggybacks onto
the existing fire-and-forget premium-tick flush. It is PURE TELEMETRY — never read back into an
exit decision — and every write is wrapped so a bug here cannot touch the money/sell path.

These tests lock that safety in.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import options_owl.risk.exit_v5.monitor_bridge as mb
from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge, get_fsm_snapshot


class FakeSettings:
    pass


def _trade(trade_id=1, premium=1.00, **kw):
    d = {
        "id": trade_id, "ticker": "SPY", "option_type": "call",
        "premium_per_contract": premium, "contracts": 5,
        "opened_at": "2026-04-28T14:00:00", "score": 85,
        "entry_price": 500.0, "mfe_premium": None, "strike": 500.0, "status": "open",
    }
    d.update(kw)
    return d


def _now_et(hour=10, minute=30):
    return datetime(2026, 4, 28, hour, minute, 0)


class TestSnapshotMechanics:
    def test_none_before_any_evaluate(self):
        assert get_fsm_snapshot(999) is None

    def test_snapshot_written_on_hold(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _trade(trade_id=1)
        reason, _ = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        snap = get_fsm_snapshot(1)
        assert snap is not None
        # every field the tick capture reads must be present
        for k in ("fsm_state", "gain_pct", "peak_gain_pct", "active_gate"):
            assert k in snap
        # at-entry hold: ~0% gain, holding
        assert reason is None
        assert snap["active_gate"] == "hold"
        assert abs(snap["gain_pct"]) < 0.01

    def test_gain_pct_tracks_premium(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _trade(trade_id=2)
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        bridge.evaluate(trade, 1.50, 505.0, _now_et(10, 1))  # +50%
        snap = get_fsm_snapshot(2)
        assert snap["gain_pct"] == 50.0
        assert snap["peak_gain_pct"] >= 50.0  # peak captured

    def test_active_gate_matches_hold_vs_exit_invariant(self):
        """active_gate == 'hold' IFF the engine held — true for ANY scenario."""
        bridge = V5MonitorBridge(FakeSettings())
        # 0DTE at EOD → the eod_cutoff gate should force an exit (gate #1, pre-grace)
        trade = _trade(trade_id=3, expiry_date="2026-04-28")
        reason, _ = bridge.evaluate(trade, 0.90, 499.0, _now_et(15, 59))
        snap = get_fsm_snapshot(3)
        if reason is None:
            assert snap["active_gate"] == "hold"
        else:
            assert snap["active_gate"] and snap["active_gate"] != "hold"

    def test_cleanup_pops_snapshot(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _trade(trade_id=4)
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert get_fsm_snapshot(4) is not None
        bridge.cleanup_trade(4)
        assert get_fsm_snapshot(4) is None


class TestSnapshotIsMoneyPathSafe:
    def test_broken_snapshot_store_cannot_break_exit(self, monkeypatch):
        """THE safety test: if writing telemetry raises, evaluate() must still return the
        correct exit decision and NEVER propagate the error into the sell path."""
        class Boom(dict):
            def __setitem__(self, k, v):
                raise RuntimeError("telemetry backend exploded")

        monkeypatch.setattr(mb, "_LAST_FSM_SNAPSHOT", Boom())
        bridge = V5MonitorBridge(FakeSettings())
        trade = _trade(trade_id=5)
        # must not raise despite the store blowing up on every write
        reason, desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert reason is None  # a fresh at-entry trade still correctly holds
        assert isinstance(desc, str)

    def test_snapshot_write_is_wrapped_in_try_except(self):
        src = inspect.getsource(V5MonitorBridge.evaluate)
        # the snapshot assignment must be inside a try/except so it can't escape
        assert "_LAST_FSM_SNAPSHOT[state.trade_id]" in src
        idx = src.index("_LAST_FSM_SNAPSHOT[state.trade_id]")
        assert "try:" in src[:idx]
        assert "except Exception" in src[idx:]


class TestPersistencePlumbing:
    def test_write_batch_includes_new_columns_and_is_backward_compatible(self):
        from options_owl.db import postgres
        src = inspect.getsource(postgres.write_premium_ticks_batch)
        for col in ("fsm_state", "gain_pct", "peak_gain_pct", "active_gate"):
            assert col in src
        # must use .get() for the new fields so a tick lacking them still inserts (NULL)
        assert 't.get("fsm_state")' in src

    def test_schema_migration_adds_nullable_columns(self):
        from options_owl.db import postgres
        for col in ("fsm_state", "gain_pct", "peak_gain_pct", "active_gate"):
            assert (f"ADD COLUMN IF NOT EXISTS {col}" in postgres.SCHEMA_SQL)

    def test_dashboard_query_selects_new_columns(self):
        from options_owl.dashboard import db
        src = inspect.getsource(db.get_premium_ticks)
        assert "fsm_state" in src and "active_gate" in src

    def test_position_monitor_tick_capture_is_wrapped(self):
        """The tick-append merge in the monitor loop must be wrapped + use .get so a
        telemetry miss can never break the sell loop."""
        from options_owl.execution import position_monitor
        src = inspect.getsource(position_monitor)
        i = src.index("get_fsm_snapshot(trade_id)")
        # a try: precedes the call and an except swallows any failure right after
        assert "try:" in src[max(0, i - 300):i]
        assert "except Exception:" in src[i:i + 800]
