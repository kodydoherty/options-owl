"""Integration tests — full trade lifecycle simulations through v5 FSM.

Simulates real trade scenarios with category-aware, DTE-aware exits.
"""

from datetime import datetime, timedelta

from options_owl.risk.exit_v5.config import V5Config
from options_owl.risk.exit_v5.fsm import ExitFSM, FSMState, TradeState
from options_owl.risk.exit_v5.types import ExitReason


def _now_et(hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, 4, 28, hour, minute, 0)


def _make_state(**kwargs) -> TradeState:
    defaults = dict(
        trade_id=1, ticker="AAPL", option_type="call",
        entry_premium=1.00, entry_time=_now_et(10, 0),
        contracts=5, peak_premium=1.00,
    )
    defaults.update(kwargs)
    return TradeState(**defaults)


def _simulate(fsm, state, premium_path, start_time=None, dt_seconds=5,
              current_underlying=0.0):
    """Run FSM over a sequence of (premium, bid, ask) tuples."""
    t = start_time or state.entry_time
    history = []
    action = None
    for i, (prem, bid, ask) in enumerate(premium_path):
        now = t + timedelta(seconds=i * dt_seconds)
        minutes_to_close = max(0, (16 * 60 - (now.hour * 60 + now.minute)))
        action = fsm.evaluate(state, prem, bid, ask, now,
                              current_underlying=current_underlying,
                              minutes_to_close=minutes_to_close)
        elapsed = (now - state.entry_time).total_seconds()
        history.append((elapsed, state.state.value, action.reason.value, action.detail))
        if action.should_exit:
            return action, history
    return action, history


class TestQuickLoser0DTE:
    """0DTE trade drops, underlying against → checkpoint fires first at 30%."""

    def test_checkpoint_fires(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)

        # Past 5min grace (60 ticks × 5s = 300s = 5min)
        grace = [(1.00, 0.95, 1.05)] * 60
        # Drop to -36% (> 30% checkpoint AND > 35% tight stop)
        drop = [(0.64, 0.59, 0.69)] * 5

        # underlying down 0.6% → against for call
        action, _ = _simulate(fsm, state, grace + drop, current_underlying=99.4)
        assert action.should_exit
        assert action.reason == ExitReason.CHECKPOINT_CUT


class TestBackstopWithoutConfirm:
    """0DTE: big drop but underlying not against → backstop at 65%."""

    def test_backstop_fires(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)

        grace = [(1.00, 0.95, 1.05)] * 60
        # Drop to -66% (> 65% backstop), underlying flat
        drop = [(0.34, 0.29, 0.39)] * 5

        action, _ = _simulate(fsm, state, grace + drop, current_underlying=100.1)
        assert action.should_exit
        assert action.reason == ExitReason.HARD_STOP


class TestSoftTrailAfterGrace:
    """Soft trail fires after 5min grace."""

    def test_soft_trail_exits(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)

        # 5min grace (60 ticks)
        grace = [(1.00, 0.95, 1.05)] * 60
        # Ramp to +30%
        ramp = []
        for i in range(20):
            p = 1.00 + i * 0.015
            ramp.append((p, p - 0.05, p + 0.05))
        peak = [(1.30, 1.25, 1.35)] * 5
        # Floor = 1.00 + 0.30*0.60 = 1.18. Drop below floor.
        retrace = [(1.10, 1.05, 1.15)] * 3

        # underlying up 1% → confirms call direction, so scalp won't fire
        action, _ = _simulate(fsm, state, grace + ramp + peak + retrace,
                              current_underlying=101.0)
        assert action.should_exit
        assert action.reason == ExitReason.SOFT_TRAIL


class TestScalpWith0DTE:
    """0DTE scalp: peaked, faded, underlying doesn't confirm."""

    def test_scalp_exits(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)

        # Past grace
        grace = [(1.00, 0.95, 1.05)] * 60
        # Pump to +25%
        pump = [(1.25, 1.20, 1.30)] * 10
        # Fade to +8%
        fade = [(1.08, 1.03, 1.13)] * 5

        # underlying flat → doesn't confirm (+0.2% needed)
        action, _ = _simulate(fsm, state, grace + pump + fade, current_underlying=100.1)
        assert action.should_exit
        assert action.reason == ExitReason.SCALP_TRAIL


class TestMultidayWiderStops:
    """Multi-day trade gets wider stops and no checkpoint/theta bleed."""

    def test_multiday_survives_25pct_drop(self):
        """Multi-day: 25% drop with underlying against → holds (tight=30%)."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")

        grace = [(1.00, 0.95, 1.05)] * 60
        drop = [(0.75, 0.70, 0.80)] * 5  # -25% < 30% tight

        action, _ = _simulate(fsm, state, grace + drop, current_underlying=99.4)
        assert not action.should_exit


class TestProfitTargetIndex0DTE:
    """Index 0DTE takes profit at 30%."""

    def test_spy_profit_target(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="SPY", entry_underlying_price=500.0)

        grace = [(1.00, 0.95, 1.05)] * 60
        # Ramp to +32%
        ramp = [(1.32, 1.30, 1.34)] * 3

        action, _ = _simulate(fsm, state, grace + ramp, current_underlying=501.0)
        assert action.should_exit
        assert action.reason == ExitReason.PROFIT_TARGET

    def test_non_index_no_profit_target(self):
        """AAPL doesn't get profit target gate."""
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="AAPL", entry_underlying_price=200.0)

        grace = [(1.00, 0.95, 1.05)] * 60
        ramp = [(1.32, 1.30, 1.34)] * 3

        action, _ = _simulate(fsm, state, grace + ramp, current_underlying=201.0)
        # Should hold (no profit target for non-index)
        assert not action.should_exit


class TestMultidayThetaTimer:
    """Multi-day theta timer cuts stale losers."""

    def test_theta_timer_fires(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")

        # Hold for 180min (2160 ticks × 5s), slowly declining
        hold = [(0.90, 0.85, 0.95)] * 2160
        # Drop to -20% (> 15% threshold)
        drop = [(0.80, 0.75, 0.85)] * 5

        action, _ = _simulate(fsm, state, hold + drop, current_underlying=100.1)
        assert action.should_exit
        assert action.reason == ExitReason.THETA_TIMER


class TestEODCutoff0DTE:
    """0DTE trade near close → forced exit."""

    def test_eod_forces_exit(self):
        fsm = ExitFSM(V5Config())
        state = _make_state()
        action = fsm.evaluate(state, 1.50, 1.45, 1.55, _now_et(15, 50),
                              minutes_to_close=10.0)
        assert action.should_exit
        assert action.reason == ExitReason.EOD_CUTOFF


class TestThetaBleed0DTE:
    """0DTE trade held 120min+ while down → theta bleed."""

    def test_theta_bleed(self):
        # Use wider backstop so graduated stop doesn't fire first at -32%
        cfg = V5Config(backstop_0dte_pct=65.0)
        fsm = ExitFSM(cfg)
        state = _make_state(entry_underlying_price=100.0)

        # Hold for 120min (1440 ticks), slowly declining
        hold = [(0.95, 0.90, 1.00)] * 1440
        # Drop to -32%
        drop = [(0.68, 0.63, 0.73)] * 5

        # underlying flat → won't trigger graduated stop backstop (65%)
        action, _ = _simulate(fsm, state, hold + drop, current_underlying=100.1)
        assert action.should_exit
        assert action.reason == ExitReason.THETA_BLEED


class TestFullStateProgression:
    """Complete lifecycle through all states."""

    def test_all_three_states(self):
        fsm = ExitFSM(V5Config())
        state = _make_state()
        states_seen = set()

        # Grace (0-5min = 60 ticks)
        for i in range(60):
            now = state.entry_time + timedelta(seconds=i * 5)
            fsm.evaluate(state, 1.00, 0.95, 1.05, now, minutes_to_close=350.0)
            states_seen.add(state.state)

        assert FSMState.GRACE in states_seen

        # Developing (after 5min, gain < 40%)
        for i in range(30):
            now = state.entry_time + timedelta(seconds=300 + i * 5)
            fsm.evaluate(state, 1.10, 1.05, 1.15, now, minutes_to_close=340.0)
            states_seen.add(state.state)

        assert FSMState.DEVELOPING in states_seen

        # Trailing (push past +40%)
        for i in range(10):
            now = state.entry_time + timedelta(seconds=450 + i * 5)
            p = 1.40 + i * 0.02
            fsm.evaluate(state, p, p - 0.05, p + 0.05, now, minutes_to_close=330.0)
            states_seen.add(state.state)

        assert FSMState.TRAILING in states_seen
        assert len(states_seen) == 3
