"""Tests for the V6 early-pop gate — tighter backstop for trades that peak early then fade.

Covers:
  1. _is_early_pop pure function: all conditions, edge cases
  2. FSM integration: gate fires correctly and produces right exit reason
  3. Gate is OFF when ENABLE_V6_EARLY_POP_GATE=False (no behavior change)
  4. Gate does not fire during grace period
  5. Gate does not affect trades that don't match the pattern
  6. TradeState.peak_elapsed_min tracking
  7. Config defaults are correct
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from options_owl.risk.exit_v5.config import V5Config
from options_owl.risk.exit_v5.fsm import (
    ExitFSM,
    TradeState,
    _is_early_pop,
)
from options_owl.risk.exit_v5.types import ExitReason


def _now_et(hour: int = 10, minute: int = 30) -> datetime:
    return datetime(2026, 5, 14, hour, minute, 0)


def _make_state(
    entry_premium: float = 1.00,
    contracts: int = 5,
    ticker: str = "AMZN",
    entry_time: datetime | None = None,
    option_type: str = "call",
    **kwargs,
) -> TradeState:
    return TradeState(
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=entry_premium,
        entry_time=entry_time or _now_et(10, 0),
        contracts=contracts,
        peak_premium=kwargs.pop("peak_premium", entry_premium),
        entry_underlying_price=kwargs.pop("entry_underlying_price", 200.0),
        **kwargs,
    )


def _settings(**overrides):
    """Settings with early-pop gate enabled by default for testing."""
    defaults = {
        "ENABLE_V6_EARLY_POP_GATE": True,
        "ENABLE_V6_BREAKEVEN_RATCHET": False,
        "V6_BREAKEVEN_TRIGGER_PCT": 20.0,
        "ENABLE_V6_SCALEOUT": False,
        "V6_SCALEOUT_GAIN_PCT": 20.0,
        "V6_SCALEOUT_FRACTION": 0.333,
        "V6_SCALEOUT_MIN_CONTRACTS": 3,
        "ENABLE_V6_2PM_TIGHTEN": False,
        "V6_2PM_TRAIL_TIGHTEN_FACTOR": 0.7,
        "V6_2PM_SOFT_TRAIL_BOOST": 0.15,
        "ENABLE_V6_SIDEWAYS_SCALP": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ══════════════════════════════════════════════════════════════════════════
# 1. _is_early_pop pure function tests
# ══════════════════════════════════════════════════════════════════════════


class TestIsEarlyPop:
    """Test the _is_early_pop detection function directly."""

    def _state_with_pop(self, peak_min=5.0, peak_gain_pct=10.0,
                        fade_pct=15.0, elapsed_min=13.0):
        """Build a TradeState that looks like an early-pop trade."""
        entry = 1.00
        peak = entry * (1 + peak_gain_pct / 100)
        current = peak * (1 - fade_pct / 100)
        state = _make_state(entry_premium=entry, peak_premium=peak)
        state.peak_elapsed_min = peak_min
        state.premium_history = [current]
        return state, elapsed_min, peak_gain_pct

    def test_classic_early_pop_detected(self):
        """Peak at 5min, faded 15% by minute 13 — should detect."""
        state, elapsed, peak_gain = self._state_with_pop()
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, peak_gain, cfg) is True

    def test_not_detected_before_check_time(self):
        """At minute 8 (before check_after=12), should NOT detect."""
        state, _, peak_gain = self._state_with_pop(elapsed_min=8.0)
        cfg = V5Config()
        assert _is_early_pop(state, 8.0, peak_gain, cfg) is False

    def test_not_detected_when_peak_was_late(self):
        """Peak at 15min (after peak_window=12) — should NOT detect."""
        state, elapsed, peak_gain = self._state_with_pop(peak_min=15.0)
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, peak_gain, cfg) is False

    def test_not_detected_when_peak_too_small(self):
        """Peak was only +2% (below min_peak_gain=3%) — should NOT detect."""
        state, elapsed, _ = self._state_with_pop(peak_gain_pct=2.0)
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, 2.0, cfg) is False

    def test_not_detected_when_fade_too_small(self):
        """Only faded 5% from peak (below fade_pct=10%) — should NOT detect."""
        state, elapsed, peak_gain = self._state_with_pop(fade_pct=5.0)
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, peak_gain, cfg) is False

    def test_detected_at_exact_thresholds(self):
        """Exactly at all thresholds — should detect.

        Uses fade_pct=10.1 instead of 10.0 to avoid IEEE 754 floating-point
        rounding at the boundary (e.g. 1.03 * 0.9 = 0.9269999...).
        """
        state, elapsed, peak_gain = self._state_with_pop(
            peak_min=12.0, peak_gain_pct=3.0, fade_pct=10.1, elapsed_min=12.0)
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, peak_gain, cfg) is True

    def test_not_detected_with_empty_history(self):
        """No premium history — should NOT detect (fails safe)."""
        state = _make_state(peak_premium=1.10)
        state.peak_elapsed_min = 5.0
        state.premium_history = []
        cfg = V5Config()
        assert _is_early_pop(state, 13.0, 10.0, cfg) is False

    def test_not_detected_with_zero_peak(self):
        """Peak premium is zero — should NOT detect."""
        state = _make_state(peak_premium=0.0)
        state.peak_elapsed_min = 5.0
        state.premium_history = [0.5]
        cfg = V5Config()
        assert _is_early_pop(state, 13.0, 0.0, cfg) is False

    def test_peak_at_time_zero_detected(self):
        """Peak at entry (time 0) is valid — the trade never went higher."""
        state, elapsed, peak_gain = self._state_with_pop(peak_min=0.0)
        cfg = V5Config()
        assert _is_early_pop(state, elapsed, peak_gain, cfg) is True

    def test_custom_config_thresholds(self):
        """Custom config values are respected."""
        cfg = V5Config(
            early_pop_peak_window_min=5.0,
            early_pop_fade_pct=20.0,
            early_pop_check_after_min=6.0,
            early_pop_min_peak_gain_pct=5.0,
        )
        # Peak at 4min, +6%, faded 25%, at minute 7 — should match custom config
        state, _, _ = self._state_with_pop(
            peak_min=4.0, peak_gain_pct=6.0, fade_pct=25.0, elapsed_min=7.0)
        assert _is_early_pop(state, 7.0, 6.0, cfg) is True

        # Peak at 6min — outside custom window of 5min
        state2, _, _ = self._state_with_pop(
            peak_min=6.0, peak_gain_pct=6.0, fade_pct=25.0, elapsed_min=7.0)
        assert _is_early_pop(state2, 7.0, 6.0, cfg) is False


# ══════════════════════════════════════════════════════════════════════════
# 2. FSM integration tests
# ══════════════════════════════════════════════════════════════════════════


class TestEarlyPopFSMIntegration:
    """Test early-pop gate fires correctly within the full FSM evaluate loop."""

    def _run_fsm_scenario(self, settings_obj, entry=1.00, ticks=None):
        """Simulate a trade through the FSM with given tick sequence.

        ticks: list of (minutes_after_entry, premium, underlying)
        Returns list of (minutes, action) pairs.
        """
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=settings_obj)
        entry_time = _now_et(10, 0)
        state = TradeState(
            trade_id=1, ticker="AMZN", option_type="call",
            entry_premium=entry, entry_time=entry_time,
            contracts=5, peak_premium=entry,
            entry_underlying_price=200.0, dte=0,
            expiry_date="2026-05-14",
        )

        results = []
        for minutes, premium, underlying in (ticks or []):
            now = entry_time + timedelta(minutes=minutes)
            action = fsm.evaluate(
                state, premium, premium, premium, now,
                current_underlying=underlying,
                minutes_to_close=390 - minutes,
            )
            results.append((minutes, action))
        return results

    def test_early_pop_tightens_backstop(self):
        """Trade peaks at +10% in first 5min, then crashes — tighter backstop fires."""
        ticks = []
        # Grace period: premium rises to 1.10 (peak at +10%)
        for m in range(1, 6):
            ticks.append((m, 1.00 + 0.02 * m, 200.0))  # 1.02, 1.04, 1.06, 1.08, 1.10

        # After grace: premium starts fading
        for m in range(6, 13):
            prem = 1.10 - 0.03 * (m - 5)  # 1.07, 1.04, 1.01, 0.98, 0.95, 0.92, 0.89
            ticks.append((m, prem, 200.0))

        # At minute 13+: premium crashes to trigger backstop
        # With tighter backstop at 35%: entry=1.00, 35% drop = 0.65
        # With default backstop at 65%: entry=1.00, 65% drop = 0.35
        for m in range(13, 25):
            prem = 0.89 - 0.05 * (m - 13)
            if prem < 0.10:
                prem = 0.10
            ticks.append((m, prem, 200.0))

        # With early-pop ON: should exit at 35% drop (premium ~0.65)
        results_on = self._run_fsm_scenario(_settings(), ticks=ticks)
        exits_on = [(m, a) for m, a in results_on if a.should_exit]

        # With early-pop OFF: should exit at 65% drop (premium ~0.35)
        results_off = self._run_fsm_scenario(
            _settings(ENABLE_V6_EARLY_POP_GATE=False), ticks=ticks)
        exits_off = [(m, a) for m, a in results_off if a.should_exit]

        assert len(exits_on) > 0, "Should exit with early-pop ON"
        assert len(exits_off) > 0, "Should exit with early-pop OFF"
        # Early-pop should exit EARLIER (lower minute number)
        assert exits_on[0][0] < exits_off[0][0], \
            f"Early-pop should exit sooner: {exits_on[0][0]}min vs {exits_off[0][0]}min"

    def test_gate_off_no_behavior_change(self):
        """When ENABLE_V6_EARLY_POP_GATE=False, no behavior change."""
        ticks = []
        for m in range(1, 6):
            ticks.append((m, 1.00 + 0.02 * m, 200.0))
        for m in range(6, 20):
            prem = max(0.10, 1.10 - 0.06 * (m - 5))
            ticks.append((m, prem, 200.0))

        results_off = self._run_fsm_scenario(
            _settings(ENABLE_V6_EARLY_POP_GATE=False), ticks=ticks)
        results_none = self._run_fsm_scenario(
            SimpleNamespace(), ticks=ticks)  # no settings at all

        # Both should produce identical exit points
        exits_off = [(m, a.reason) for m, a in results_off if a.should_exit]
        exits_none = [(m, a.reason) for m, a in results_none if a.should_exit]
        assert exits_off == exits_none

    def test_no_fire_during_grace(self):
        """Early-pop should NOT fire during grace period."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())
        entry_time = _now_et(10, 0)
        state = TradeState(
            trade_id=1, ticker="AMZN", option_type="call",
            entry_premium=1.00, entry_time=entry_time,
            contracts=5, peak_premium=1.10,
            entry_underlying_price=200.0, dte=0,
            expiry_date="2026-05-14",
        )
        state.peak_elapsed_min = 2.0

        # At minute 3 (during grace), premium crashed to 0.60 (-40%)
        now = entry_time + timedelta(minutes=3)
        action = fsm.evaluate(state, 0.60, 0.60, 0.60, now,
                              current_underlying=200.0,
                              minutes_to_close=387)

        # Should be GRACE hold or GRACE backstop — not early-pop backstop
        # (at -40%, grace backstop at 65% won't fire, so it should HOLD)
        assert not action.should_exit or "GRACE" in action.detail

    def test_runner_not_affected(self):
        """A trade that peaks at +100% (runner) should NOT trigger early-pop."""
        ticks = []
        # Quick initial pop to +8% at minute 3
        for m in range(1, 4):
            ticks.append((m, 1.00 + 0.027 * m, 200.0))  # 1.027, 1.054, 1.081

        # Then keeps running higher (runner behavior)
        for m in range(4, 30):
            prem = 1.08 + 0.03 * (m - 3)
            ticks.append((m, prem, 200.5 + 0.1 * m))

        results = self._run_fsm_scenario(_settings(), ticks=ticks)
        # Should NOT exit via hard_stop (early-pop backstop)
        hard_exits = [(m, a) for m, a in results
                      if a.should_exit and a.reason == ExitReason.HARD_STOP]
        assert len(hard_exits) == 0, "Runner should not be stopped by early-pop"

    def test_late_peak_not_affected(self):
        """Trade that peaks at minute 20 should NOT trigger early-pop."""
        ticks = []
        # Slow build: premium gradually rises
        for m in range(1, 25):
            prem = 1.00 + 0.005 * m  # very slow rise
            ticks.append((m, prem, 200.0 + 0.05 * m))

        # Then drops
        for m in range(25, 50):
            prem = max(0.10, 1.12 - 0.04 * (m - 25))
            ticks.append((m, prem, 200.0))

        results = self._run_fsm_scenario(_settings(), ticks=ticks)

        # Check debug context — early_pop should NOT be set
        for _, action in results:
            if action.debug.get("early_pop"):
                pytest.fail("early_pop should not be True for late-peak trade")


# ══════════════════════════════════════════════════════════════════════════
# 3. TradeState peak_elapsed_min tracking
# ══════════════════════════════════════════════════════════════════════════


class TestPeakElapsedTracking:
    """Verify peak_elapsed_min is updated correctly in the FSM."""

    def test_peak_time_tracks_new_highs(self):
        """peak_elapsed_min should update each time a new high is reached."""
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=_settings())
        entry_time = _now_et(10, 0)
        state = TradeState(
            trade_id=1, ticker="AAPL", option_type="call",
            entry_premium=1.00, entry_time=entry_time,
            contracts=5, peak_premium=1.00,
            entry_underlying_price=200.0, dte=0,
            expiry_date="2026-05-14",
        )

        # Minute 2: new high at 1.05
        fsm.evaluate(state, 1.05, 1.05, 1.05,
                      entry_time + timedelta(minutes=2),
                      current_underlying=200.0)
        assert abs(state.peak_elapsed_min - 2.0) < 0.01

        # Minute 3: not a new high (1.03)
        fsm.evaluate(state, 1.03, 1.03, 1.03,
                      entry_time + timedelta(minutes=3),
                      current_underlying=200.0)
        assert abs(state.peak_elapsed_min - 2.0) < 0.01  # unchanged

        # Minute 7: new high at 1.10
        fsm.evaluate(state, 1.10, 1.10, 1.10,
                      entry_time + timedelta(minutes=7),
                      current_underlying=200.0)
        assert abs(state.peak_elapsed_min - 7.0) < 0.01

    def test_peak_time_starts_at_zero(self):
        """Default peak_elapsed_min should be 0 (peak is entry premium)."""
        state = _make_state()
        assert state.peak_elapsed_min == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 4. Config defaults
# ══════════════════════════════════════════════════════════════════════════


class TestEarlyPopConfigDefaults:
    """Verify V5Config early-pop defaults match backtested optimal values."""

    def test_default_values(self):
        cfg = V5Config()
        assert cfg.early_pop_peak_window_min == 12.0
        assert cfg.early_pop_fade_pct == 10.0
        assert cfg.early_pop_check_after_min == 12.0
        assert cfg.early_pop_min_peak_gain_pct == 3.0
        assert cfg.early_pop_backstop_0dte_pct == 25.0
        assert cfg.early_pop_backstop_multiday_pct == 40.0

    def test_per_ticker_configs_inherit_defaults(self):
        """Per-ticker configs should inherit early-pop defaults."""
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS
        for ticker, cfg in TICKER_CONFIGS.items():
            assert cfg.early_pop_peak_window_min == 12.0, \
                f"{ticker} should inherit early_pop_peak_window_min"
            assert cfg.early_pop_backstop_0dte_pct == 25.0, \
                f"{ticker} should inherit early_pop_backstop_0dte_pct"


# ══════════════════════════════════════════════════════════════════════════
# 5. Source code safety — verify variable initialization
# ══════════════════════════════════════════════════════════════════════════


class TestSourceCodeSafety:
    """Inspect the actual source to catch uninitialized variable patterns.

    This class of test catches the UnboundLocalError bug pattern:
    a variable used after a conditional block must be initialized before it.
    """

    def test_grad_cfg_initialized_before_conditional(self):
        """grad_cfg must be initialized to cfg BEFORE the early-pop conditional."""
        import inspect
        from options_owl.risk.exit_v5.fsm import ExitFSM
        source = inspect.getsource(ExitFSM.evaluate)
        # grad_cfg = cfg must appear BEFORE the if block that may override it
        assign_pos = source.find("grad_cfg = cfg")
        if_pos = source.find("ENABLE_V6_EARLY_POP_GATE")
        assert assign_pos > 0, "grad_cfg = cfg assignment not found"
        assert if_pos > 0, "ENABLE_V6_EARLY_POP_GATE check not found"
        assert assign_pos < if_pos, \
            "grad_cfg must be initialized BEFORE the early-pop conditional"

    def test_check_graduated_stop_uses_grad_cfg(self):
        """check_graduated_stop must use grad_cfg, not cfg."""
        import inspect
        from options_owl.risk.exit_v5.fsm import ExitFSM
        source = inspect.getsource(ExitFSM.evaluate)
        # Find the check_graduated_stop call
        call_pos = source.find("check_graduated_stop(")
        assert call_pos > 0
        # The line should reference grad_cfg, not bare cfg
        call_line_end = source.find("\n", call_pos + 50)
        call_snippet = source[call_pos:call_line_end + 80]
        assert "grad_cfg" in call_snippet, \
            f"check_graduated_stop should use grad_cfg: {call_snippet!r}"
