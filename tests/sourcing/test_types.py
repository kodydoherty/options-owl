"""Tests for scoring types and FSM state management."""

from options_owl.sourcing.scoring.types import (
    Direction,
    ScoredSignal,
    SignalContext,
    SignalState,
    TierResult,
)


def test_signal_context_default_state():
    ctx = SignalContext(ticker="NVDA", scan_time="2026-05-21T10:00:00")
    assert ctx.state == SignalState.INIT
    assert ctx.direction is None
    assert ctx.score_total == 0
    assert ctx.rejection_reason == ""


def test_signal_context_state_transitions():
    ctx = SignalContext(ticker="NVDA", scan_time="2026-05-21T10:00:00")
    ctx.state = SignalState.CANDLE_READY
    assert ctx.state == SignalState.CANDLE_READY
    ctx.state = SignalState.INDICATED
    assert ctx.state == SignalState.INDICATED


def test_signal_context_rejection():
    ctx = SignalContext(ticker="NVDA", scan_time="2026-05-21T10:00:00")
    ctx.state = SignalState.REJECTED
    ctx.rejection_reason = "insufficient_volume"
    ctx.rejection_stage = "SCORED"
    assert ctx.state == SignalState.REJECTED


def test_tier_result():
    t = TierResult(total=25, max_possible=40, components={"ema": 15, "tf": 10}, reasons=["strong cross"])
    assert t.total == 25
    assert t.components["ema"] == 15


def test_scored_signal_rejected():
    s = ScoredSignal(score=0, rejected=True, reject_reason="insufficient_volume")
    assert s.rejected
    assert s.score == 0


def test_scored_signal_passed():
    s = ScoredSignal(score=73, direction=Direction.CALL)
    assert not s.rejected
    assert s.score == 73
    assert s.direction == Direction.CALL


def test_direction_enum():
    assert Direction.CALL.value == "CALL"
    assert Direction.PUT.value == "PUT"


def test_signal_state_enum():
    assert SignalState.INIT.value == "INIT"
    assert SignalState.EMITTED.value == "EMITTED"
    assert SignalState.REJECTED.value == "REJECTED"
