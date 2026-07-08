"""Tests for the flow-CALL MARKET-DIRECTION filter (2026-07-08).

The symmetric twin of the flow-put filter. Flow normally bypasses directional_regime (own
whitelist), but a flow CALL bought into a FALLING tape is a counter-trend loser (the 2026-07-08
TSLA/NVDA-call-into-a-red-SPY case). The light re-application blocks flow CALLs when SPY is down
> FLOW_CALL_MKT_DIR_MAX_DROP% from the open. Validated on 848 flow calls (SPY-broad -0.5% =
+$3,386/+21%, PF 1.23->1.32; skipped 59 averaged -8%). Unlike the put twin, the edge is NOT
index-only, so this filters single-name flow calls too.
"""
import asyncio
from unittest.mock import MagicMock

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction
from options_owl.risk.pipeline import DirectionalRegimeGate, GateResult


def _flow_call(ticker="TSLA"):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.CALL
    sig.bot_source = MagicMock(value="uw_flow")
    return sig


def _flow_put(ticker="SPY"):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.PUT
    sig.bot_source = MagicMock(value="uw_flow")
    return sig


def _settings(**ov):
    d = dict(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
             ENABLE_DIRECTIONAL_REGIME=True,
             ENABLE_FLOW_CALL_MKT_DIR=True, FLOW_CALL_MKT_DIR_MAX_DROP=0.5)
    d.update(ov)
    return Settings(**d)


def _run(sig, spy_change, **s):
    ctx = {"signal": sig, "settings": _settings(**s), "spy_change_from_open": spy_change}
    return asyncio.run(DirectionalRegimeGate().evaluate(ctx))


class TestFlowCallMarketDirection:
    def test_call_blocked_on_falling_tape(self):
        """Flow call + SPY down -0.86% (the 2026-07-08 case) -> FAIL."""
        r = _run(_flow_call("TSLA"), -0.86)
        assert r.result == GateResult.FAIL
        assert "falling" in r.reason.lower() or "counter-trend" in r.reason.lower()

    def test_call_allowed_on_mild_dip(self):
        """SPY down only -0.2% (within -0.5) -> bypassed/allowed (SKIP) — don't skip dips."""
        r = _run(_flow_call("TSLA"), -0.2)
        assert r.result == GateResult.SKIP

    def test_call_allowed_on_green_tape(self):
        """SPY up +0.4% -> bypassed/allowed (calls want a green tape)."""
        r = _run(_flow_call("TSLA"), 0.4)
        assert r.result == GateResult.SKIP

    def test_boundary_not_blocked_at_exactly_threshold(self):
        """SPY exactly -0.5% is NOT < -0.5 -> allowed."""
        r = _run(_flow_call("TSLA"), -0.5)
        assert r.result == GateResult.SKIP

    def test_single_name_call_also_filtered(self):
        """Unlike the put twin, single-name flow calls ARE filtered (edge not index-only)."""
        r = _run(_flow_call("NVDA"), -0.9)
        assert r.result == GateResult.FAIL

    def test_index_call_filtered(self):
        """SPY (index) flow call on a falling tape -> FAIL (strongest sub-book)."""
        r = _run(_flow_call("SPY"), -0.9)
        assert r.result == GateResult.FAIL

    def test_disabled_flag_keeps_full_bypass(self):
        """Flag off -> flow call bypassed even on a falling tape."""
        r = _run(_flow_call("TSLA"), -1.5, ENABLE_FLOW_CALL_MKT_DIR=False)
        assert r.result == GateResult.SKIP

    def test_no_spy_data_does_not_block_flow(self):
        """Missing SPY change -> can't judge -> keep the flow bypass (don't fail-closed)."""
        r = _run(_flow_call("TSLA"), None)
        assert r.result == GateResult.SKIP

    def test_flow_put_not_touched_by_call_filter(self):
        """A flow PUT on a falling tape is not blocked by the CALL filter (bypasses normally)."""
        r = _run(_flow_put("SPY"), -0.9)
        assert r.result == GateResult.SKIP
