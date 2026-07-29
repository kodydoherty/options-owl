"""Tests for adaptive near-stop polling (position_monitor).

When a held position nears its stop, the monitor polls faster so a fast 0DTE dive is caught closer to the
-25% stop (shrinks the ~$2k poll-gap slippage). Safety-critical (sell path) — these verify the helper only
ever SHORTENS the sleep, is off by default, and that the loop flag is initialized before the trade loop
(no conditional-only assignment — the class of bug that once froze the monitor).
"""

import inspect

from options_owl.config.settings import Settings
from options_owl.execution.position_monitor import (
    _compute_adaptive_sleep,
    run_position_monitor,
)


class _S:
    def __init__(self, **kw):
        self.ENABLE_ADAPTIVE_POLL = kw.get("ENABLE_ADAPTIVE_POLL", True)
        self.ADAPTIVE_POLL_FAST_SEC = kw.get("ADAPTIVE_POLL_FAST_SEC", 1.0)


class TestComputeAdaptiveSleep:
    def test_shortens_when_near_stop_and_enabled(self):
        assert _compute_adaptive_sleep(True, 3.0, _S()) == 1.0

    def test_unchanged_when_not_near_stop(self):
        assert _compute_adaptive_sleep(False, 3.0, _S()) == 3.0

    def test_unchanged_when_disabled(self):
        assert _compute_adaptive_sleep(True, 3.0, _S(ENABLE_ADAPTIVE_POLL=False)) == 3.0

    def test_never_lengthens(self):
        # a fast_sec larger than the base interval must not slow the loop down
        assert _compute_adaptive_sleep(True, 3.0, _S(ADAPTIVE_POLL_FAST_SEC=10.0)) == 3.0

    def test_floor_half_second(self):
        # never poll faster than 0.5s (API-load guard)
        assert _compute_adaptive_sleep(True, 3.0, _S(ADAPTIVE_POLL_FAST_SEC=0.1)) == 0.5

    def test_missing_attrs_safe_default(self):
        # a settings object without the attrs must not crash and must not change behavior
        assert _compute_adaptive_sleep(True, 3.0, object()) == 3.0


class TestSettings:
    def test_disabled_by_default(self):
        assert Settings().ENABLE_ADAPTIVE_POLL is False

    def test_defaults(self):
        s = Settings()
        assert s.ADAPTIVE_POLL_FAST_SEC == 1.0
        assert s.ADAPTIVE_POLL_NEAR_STOP_GAIN_PCT == -17.0

    def test_upside_running_default(self):
        # UPSIDE fast-poll: a position up >= this gain% also polls fast (catches the round-trip)
        assert Settings().ADAPTIVE_POLL_RUNNING_GAIN_PCT == 10.0


class TestUpsideFastPoll:
    """Upside adaptive poll (2026-07-22): poll fast when RUNNING (up big), not just near a stop —
    the +17%->-31% round-trip-through-the-floor case."""

    def test_running_condition_in_loop(self):
        """The monitor loop must set the fast-poll flag when pnl_pct >= ADAPTIVE_POLL_RUNNING_GAIN_PCT
        (the upside branch), in addition to the near-stop branch."""
        src = inspect.getsource(run_position_monitor)
        assert "ADAPTIVE_POLL_RUNNING_GAIN_PCT" in src, "upside running-gain condition missing"
        # near-stop (<=), running (>=), AND never-green danger-zone branches must set the flag
        assert src.count("near_stop_this_cycle = True") >= 3, "stop + running + never-green branches must fire"
        assert "NEVERGREEN_CUT_LOSS_PCT" in src, "never-green fast-poll condition missing"


class TestSourceCodeSafety:
    def test_near_stop_flag_initialized_before_trade_loop(self):
        """near_stop_this_cycle must be assigned BEFORE `for trade in trades:` so the end-of-cycle
        sleep can never hit an unbound variable (the conditional-only-assignment bug class)."""
        src = inspect.getsource(run_position_monitor)
        init_idx = src.find("near_stop_this_cycle = False")
        loop_idx = src.find("for trade in trades:")
        assert init_idx != -1, "near_stop_this_cycle init not found"
        assert loop_idx != -1
        assert init_idx < loop_idx, "flag must be initialized before the trade loop"

    def test_sleep_uses_adaptive_helper(self):
        src = inspect.getsource(run_position_monitor)
        assert "_compute_adaptive_sleep(near_stop_this_cycle" in src
