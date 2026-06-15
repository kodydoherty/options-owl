"""Tests for two new features:
1. Scalp Target gate (V5 exit FSM) — takes +25% profit unless confirmed runner.
2. Morning Cutoff (entry pipeline TimeOfDayGate) — blocks entries after 11:00 AM ET.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch


from options_owl.risk.exit_v5.config import V5Config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.gates import check_scalp_target
from options_owl.risk.exit_v5.types import ExitReason
from options_owl.risk.pipeline import GateResult, TimeOfDayGate


# ============================================================================
# Helpers
# ============================================================================

def _debug() -> dict:
    return {}


def _now_et(hour: int = 10, minute: int = 30) -> datetime:
    return datetime(2026, 5, 22, hour, minute, 0)


def _make_state(
    entry_premium: float = 1.00,
    contracts: int = 5,
    ticker: str = "AAPL",
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
        peak_premium=entry_premium,
        **kwargs,
    )


def _make_settings(**overrides) -> SimpleNamespace:
    """Create a mock settings object with sensible defaults for FSM tests."""
    defaults = dict(
        ENABLE_V6_BREAKEVEN_RATCHET=False,
        ENABLE_V6_SCALEOUT=False,
        ENABLE_V6_2PM_TIGHTEN=False,
        ENABLE_V6_EARLY_POP_GATE=False,
        ENABLE_V6_SIDEWAYS_SCALP=False,
        ENABLE_PUT_TRADING=True,
        ENABLE_SCALP_TARGET=False,
        SCALP_TARGET_PCT=25.0,
        SCALP_RUNNER_CONFIRM_PCT=40.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_signal(score: int = 100) -> SimpleNamespace:
    """Create a mock signal for pipeline gate tests."""
    return SimpleNamespace(score=score)


def _make_tod_settings(**overrides) -> SimpleNamespace:
    """Create a mock settings object with sensible defaults for TimeOfDayGate tests."""
    defaults = dict(
        ENABLE_VINNY_STRATEGY=True,
        ENTRY_HARD_CUTOFF_HOUR=15,
        ENTRY_HARD_CUTOFF_MINUTE=55,
        ENABLE_MORNING_CUTOFF=True,
        ENTRY_MORNING_CUTOFF_HOUR=11,
        ENTRY_MORNING_CUTOFF_MINUTE=0,
        TOD_EARLY_CUTOFF_HOUR=9,
        TOD_EARLY_CUTOFF_MINUTE=45,
        TOD_EARLY_MIN_SCORE=85,
        TOD_LATE_CUTOFF_HOUR=14,
        TOD_LATE_CUTOFF_MINUTE=0,
        TOD_LATE_MIN_SCORE=85,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_gate_at_time(settings, mock_time: datetime, score: int = 100):
    """Run TimeOfDayGate with a mocked datetime.now() and return GateOutcome.

    The TimeOfDayGate imports datetime locally inside evaluate():
        from datetime import datetime, timedelta
        now = datetime.now(tz=et)

    To control 'now', we patch the datetime module's datetime class so that
    datetime.now() returns our mock_time while still allowing .replace() etc.
    """
    gate = TimeOfDayGate()
    ctx = {
        "settings": settings,
        "signal": _make_signal(score=score),
    }

    # Subclass datetime to override now() while keeping all other methods
    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return mock_time.astimezone(tz)
            return mock_time

    # Patch in the datetime module itself, which is where the local import resolves
    loop = asyncio.new_event_loop()
    try:
        with patch("datetime.datetime", _FakeDatetime):
            outcome = loop.run_until_complete(gate.evaluate(ctx))
    finally:
        loop.close()
    return outcome


# ============================================================================
# Feature 1: Scalp Target Gate (unit tests on check_scalp_target)
# ============================================================================

class TestScalpTargetGate:

    def test_scalp_target_fires_at_threshold(self):
        """Gain >= 25%, no runner signals -> exit with SCALP_TARGET."""
        action = check_scalp_target(
            gain=27.0,
            peak_gain=27.0,
            elapsed_min=10.0,
            underlying_confirms=False,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is not None
        assert action.should_exit
        assert action.reason == ExitReason.SCALP_TARGET

    def test_scalp_target_below_threshold(self):
        """Gain < 25% -> None (hold)."""
        action = check_scalp_target(
            gain=20.0,
            peak_gain=20.0,
            elapsed_min=10.0,
            underlying_confirms=False,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is None

    def test_scalp_target_skips_confirmed_runner(self):
        """Peak gain >= 40% (runner confirmed) -> None (let it ride)."""
        action = check_scalp_target(
            gain=30.0,
            peak_gain=45.0,
            elapsed_min=15.0,
            underlying_confirms=False,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is None

    def test_scalp_target_skips_enrg_hold(self):
        """Candle data with ENRG HOLD -> None (momentum intact, skip scalp)."""
        candle_data = {"indicators": {"rsi_5m": 55.0}}
        with patch(
            "options_owl.collectors.candle_cache.evaluate_enrg",
            return_value=("HOLD", "momentum intact"),
        ) as mock_enrg:
            action = check_scalp_target(
                gain=28.0,
                peak_gain=28.0,
                elapsed_min=10.0,
                underlying_confirms=False,
                candle_data=candle_data,
                option_type="call",
                scalp_target_pct=25.0,
                runner_confirm_pct=40.0,
                debug=_debug(),
            )
            assert action is None
            mock_enrg.assert_called_once_with(candle_data, "call")

    def test_scalp_target_skips_underlying_confirms(self):
        """Underlying confirms + gain near target (80%-150% of 25%) -> None."""
        # gain=26.0 is between 20.0 (80% of 25) and 37.5 (150% of 25)
        action = check_scalp_target(
            gain=26.0,
            peak_gain=26.0,
            elapsed_min=10.0,
            underlying_confirms=True,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is None

    def test_scalp_target_fires_despite_underlying_when_way_above_target(self):
        """Gain >= 37.5% (150% of 25%) + underlying confirms -> still exits.

        The underlying_confirms skip only works when gain < 150% of target.
        At 38%, the trade is too far above target -- take profit.
        """
        action = check_scalp_target(
            gain=38.0,
            peak_gain=38.0,
            elapsed_min=10.0,
            underlying_confirms=True,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is not None
        assert action.should_exit
        assert action.reason == ExitReason.SCALP_TARGET

    def test_scalp_target_no_candle_data(self):
        """No candle data, no other runner signals -> takes scalp."""
        action = check_scalp_target(
            gain=30.0,
            peak_gain=30.0,
            elapsed_min=10.0,
            underlying_confirms=False,
            candle_data=None,
            option_type="call",
            scalp_target_pct=25.0,
            runner_confirm_pct=40.0,
            debug=_debug(),
        )
        assert action is not None
        assert action.should_exit
        assert action.reason == ExitReason.SCALP_TARGET

    def test_scalp_target_enrg_non_hold(self):
        """Candle data with ENRG action != 'HOLD' -> takes scalp."""
        candle_data = {"indicators": {"rsi_5m": 25.0}}
        with patch(
            "options_owl.collectors.candle_cache.evaluate_enrg",
            return_value=("IMMEDIATE_EXIT", "bearish reversal"),
        ):
            action = check_scalp_target(
                gain=28.0,
                peak_gain=28.0,
                elapsed_min=10.0,
                underlying_confirms=False,
                candle_data=candle_data,
                option_type="call",
                scalp_target_pct=25.0,
                runner_confirm_pct=40.0,
                debug=_debug(),
            )
            assert action is not None
            assert action.should_exit
            assert action.reason == ExitReason.SCALP_TARGET


# ============================================================================
# Feature 1: Scalp Target -- FSM integration tests
# ============================================================================

class TestFSMScalpTargetIntegration:

    def test_fsm_scalp_target_gate_enabled(self):
        """Full FSM evaluate() with ENABLE_SCALP_TARGET=True, gain at +30%,
        no runner signals -> should get SCALP_TARGET exit."""
        settings = _make_settings(ENABLE_SCALP_TARGET=True)
        fsm = ExitFSM(V5Config(), settings=settings)

        # Entry at $1.00, 10 min ago
        state = _make_state(
            entry_premium=1.00,
            ticker="AAPL",
            entry_time=_now_et(10, 0),
        )

        # Current premium at $1.30 = +30% gain, past grace period
        now = _now_et(10, 10)
        result = fsm.evaluate(
            state,
            current_premium=1.30,
            bid=1.28,
            ask=1.32,
            now_et=now,
            current_underlying=0.0,
            minutes_to_close=330.0,
            candle_data=None,
        )

        assert result.should_exit
        assert result.reason == ExitReason.SCALP_TARGET

    def test_fsm_scalp_target_gate_disabled(self):
        """ENABLE_SCALP_TARGET=False -> scalp target gate does not fire."""
        settings = _make_settings(ENABLE_SCALP_TARGET=False)
        fsm = ExitFSM(V5Config(), settings=settings)

        state = _make_state(
            entry_premium=1.00,
            ticker="AAPL",
            entry_time=_now_et(10, 0),
        )

        # Current premium at $1.30 = +30%, past grace
        now = _now_et(10, 10)
        result = fsm.evaluate(
            state,
            current_premium=1.30,
            bid=1.28,
            ask=1.32,
            now_et=now,
            current_underlying=0.0,
            minutes_to_close=330.0,
            candle_data=None,
        )

        # Should NOT get SCALP_TARGET -- gate is disabled
        assert result.reason != ExitReason.SCALP_TARGET


# ============================================================================
# Feature 2: Morning Cutoff (TimeOfDayGate in entry pipeline)
# ============================================================================

class TestMorningCutoff:

    def _et_time(self, hour: int, minute: int = 0) -> datetime:
        """Create a timezone-aware datetime in ET."""
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            et = None
        return datetime(2026, 5, 22, hour, minute, 0, tzinfo=et)

    def test_morning_cutoff_blocks_after_11am(self):
        """11:30 AM ET with ENABLE_MORNING_CUTOFF=True -> FAIL."""
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=True)
        outcome = _run_gate_at_time(settings, self._et_time(11, 30))
        assert outcome.result == GateResult.FAIL
        assert "Morning cutoff" in outcome.reason

    def test_morning_cutoff_allows_before_11am(self):
        """10:00 AM ET with ENABLE_MORNING_CUTOFF=True -> PASS."""
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=True)
        outcome = _run_gate_at_time(settings, self._et_time(10, 0))
        # 10:00 AM is after early cutoff (9:45) and before morning cutoff (11:00)
        assert outcome.result == GateResult.PASS

    def test_morning_cutoff_disabled(self):
        """ENABLE_MORNING_CUTOFF=False, time is 2:00 PM -> passes morning check.

        Score 100 >= TOD_LATE_MIN_SCORE (85), so late-cutoff also passes.
        """
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=False)
        outcome = _run_gate_at_time(settings, self._et_time(14, 0), score=100)
        # Morning cutoff disabled; at 14:00 with score 100 >= 85 late threshold -> PASS
        assert outcome.result == GateResult.PASS

    def test_morning_cutoff_exact_boundary(self):
        """Exactly 11:00 AM ET with ENABLE_MORNING_CUTOFF=True -> FAIL.

        The gate uses `now >= morning_cutoff`, so exactly 11:00 is blocked.
        """
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=True)
        outcome = _run_gate_at_time(settings, self._et_time(11, 0))
        assert outcome.result == GateResult.FAIL
        assert "Morning cutoff" in outcome.reason

    def test_morning_cutoff_custom_time(self):
        """Set ENTRY_MORNING_CUTOFF_HOUR=12 -> allows at 11:30 AM ET."""
        settings = _make_tod_settings(
            ENABLE_MORNING_CUTOFF=True,
            ENTRY_MORNING_CUTOFF_HOUR=12,
            ENTRY_MORNING_CUTOFF_MINUTE=0,
        )
        outcome = _run_gate_at_time(settings, self._et_time(11, 30))
        # 11:30 AM < 12:00 PM custom cutoff -> PASS
        assert outcome.result == GateResult.PASS

    def _run_gate_flow_call(self, settings, mock_time: datetime):
        """Run TimeOfDayGate with a flow-sourced CALL signal (bot_source.value='uw_flow')."""
        gate = TimeOfDayGate()
        flow_signal = SimpleNamespace(
            score=90, bot_source=SimpleNamespace(value="uw_flow"),
            direction=SimpleNamespace(name="CALL"),
        )
        ctx = {"settings": settings, "signal": flow_signal}

        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return mock_time.astimezone(tz) if tz is not None else mock_time

        loop = asyncio.new_event_loop()
        try:
            with patch("datetime.datetime", _FakeDatetime):
                return loop.run_until_complete(gate.evaluate(ctx))
        finally:
            loop.close()

    def test_flow_call_bypasses_morning_cutoff(self):
        """UW flow CALL at 1:00 PM ET -> SKIP (all-day source). Regression: the $6.25M
        SPY whale call on 2026-06-15 was wrongly blocked by the 11:00 ET morning cutoff."""
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=True)
        outcome = self._run_gate_flow_call(settings, self._et_time(13, 0))
        assert outcome.result == GateResult.SKIP
        assert "flow" in outcome.reason.lower()

    def test_flow_call_still_blocked_by_eod_hard_cutoff(self):
        """Flow does NOT escape the EOD hard cutoff (3:55 PM ET) — theta crush still applies."""
        settings = _make_tod_settings(ENABLE_MORNING_CUTOFF=True)
        outcome = self._run_gate_flow_call(settings, self._et_time(15, 56))
        assert outcome.result == GateResult.FAIL
        assert "after 15:55 ET" in outcome.reason
