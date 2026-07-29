"""Tests for the CALL underlying-UP trigger (2026-07-22) — mirror of the put trigger.

Require the CALL's own underlying to NOT be falling at entry (>= CALL_UNDERLYING_UP_TRIGGER% from
open), applied to ML *and* flow. Real live data: calls into a falling ticker lost -$3,199/6wk.
Fail-OPEN on missing data (main revenue book — only block when we can CONFIRM the ticker is falling)."""
import asyncio
from unittest.mock import MagicMock

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction
from options_owl.risk.pipeline import DirectionalRegimeGate, GateResult


def _call(ticker="NVDA", flow=False):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.CALL
    sig.bot_source = MagicMock(value="uw_flow" if flow else "ml_sourcing")
    return sig


def _settings(**ov):
    d = dict(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
             ENABLE_CALL_UNDERLYING_TRIGGER=True, CALL_UNDERLYING_UP_TRIGGER=0.0,
             ENABLE_DIRECTIONAL_REGIME=True)
    d.update(ov)
    return Settings(**d)


def _run(sig, ticker_change, **s):
    # candle_cache omitted; _ticker_change_from_open reads ctx["ticker_change_from_open"] first.
    ctx = {"signal": sig, "settings": _settings(**s), "ticker_change_from_open": ticker_change}
    return asyncio.run(DirectionalRegimeGate().evaluate(ctx))


class TestCallUnderlyingTrigger:
    def test_ml_call_blocked_when_ticker_falling(self):
        """The bleeder: ML call bought when the ticker is -0.8% (falling) -> FAIL."""
        r = _run(_call("NVDA"), ticker_change=-0.8)
        assert r.result == GateResult.FAIL
        assert "not rising" in r.reason

    def test_flow_call_blocked_when_ticker_falling_no_bypass(self):
        """Flow calls must ALSO be blocked into a falling ticker (evaluated before the flow bypass)."""
        r = _run(_call("TSLA", flow=True), ticker_change=-1.0)
        assert r.result == GateResult.FAIL

    def test_call_allowed_when_ticker_rising(self):
        """Ticker +0.6% (rising) -> not blocked by the trigger."""
        r = _run(_call("NVDA"), ticker_change=+0.6, ticker_change_from_open=None) if False else \
            _run(_call("NVDA"), ticker_change=+0.6)
        assert r.result != GateResult.FAIL

    def test_missing_data_fails_open(self):
        """No underlying data -> ALLOW (fail-open; main revenue book, only block on confirmed falling)."""
        r = _run(_call("NVDA"), ticker_change=None)
        assert r.result != GateResult.FAIL

    def test_exactly_at_trigger_passes(self):
        """Ticker exactly 0.0% (== trigger) is allowed (need >= 0)."""
        r = _run(_call("NVDA"), ticker_change=0.0)
        assert r.result != GateResult.FAIL

    def test_flag_off_does_not_block(self):
        """Flag off -> the trigger is skipped; a falling-ticker call is not blocked by it."""
        r = _run(_call("NVDA"), ticker_change=-1.0, ENABLE_CALL_UNDERLYING_TRIGGER=False)
        # (may still pass/skip via the rest of the gate, but must NOT fail on the trigger reason)
        assert not (r.result == GateResult.FAIL and "not rising" in (r.reason or ""))

    def test_custom_threshold(self):
        """CALL_UNDERLYING_UP_TRIGGER=+0.5 blocks a call at +0.3% (not rising enough)."""
        r = _run(_call("NVDA"), ticker_change=+0.3, CALL_UNDERLYING_UP_TRIGGER=0.5)
        assert r.result == GateResult.FAIL

    def test_put_unaffected(self):
        """A PUT is never touched by the CALL trigger."""
        sig = _call("NVDA")
        sig.direction = Direction.PUT
        r = _run(sig, ticker_change=-1.0)
        # put path may fail for other reasons but NOT the call-trigger reason
        assert not (r.result == GateResult.FAIL and "not rising" in (r.reason or ""))
