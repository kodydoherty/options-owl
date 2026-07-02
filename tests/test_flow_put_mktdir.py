"""Tests for the flow-put MARKET-DIRECTION filter (2026-07-01).

Flow normally bypasses put_market_direction (own whitelist), but a flow INDEX put bought while
SPY is rallying is a counter-trend loser (the 2026-07-01 SPY-put -54% case). The light
re-application blocks flow INDEX puts when SPY is up > FLOW_PUT_MKT_DIR_MAX_CHG% from the open.
Validated on 727 flow puts (SPY +$6,326 / PF 1.32->1.59 at +0.5%).
"""
import asyncio
from unittest.mock import MagicMock

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction
from options_owl.risk.pipeline import GateResult, PutMarketDirectionGate


def _flow_put(ticker="SPY"):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.PUT
    sig.bot_source = MagicMock(value="uw_flow")
    return sig


def _settings(**ov):
    d = dict(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
             ENABLE_FLOW_PUT_MKT_DIR=True, FLOW_PUT_MKT_DIR_MAX_CHG=0.5)
    d.update(ov)
    return Settings(**d)


def _run(gate, sig, spy_change, **s):
    ctx = {"signal": sig, "settings": _settings(**s), "spy_change_from_open": spy_change}
    return asyncio.run(gate.evaluate(ctx))


class TestFlowPutMarketDirection:
    def test_index_put_blocked_on_rally(self):
        """SPY flow put + SPY up +0.86% (the 2026-07-01 case) -> FAIL."""
        r = _run(PutMarketDirectionGate(), _flow_put("SPY"), 0.86)
        assert r.result == GateResult.FAIL
        assert "rally" in r.reason.lower() or "counter-trend" in r.reason.lower()

    def test_index_put_allowed_on_flat_tape(self):
        """SPY up only +0.2% (<= +0.5%) -> bypassed/allowed (SKIP)."""
        r = _run(PutMarketDirectionGate(), _flow_put("SPY"), 0.2)
        assert r.result == GateResult.SKIP

    def test_index_put_allowed_on_red_tape(self):
        """SPY down -0.4% -> bypassed/allowed (puts want a red tape)."""
        r = _run(PutMarketDirectionGate(), _flow_put("SPY"), -0.4)
        assert r.result == GateResult.SKIP

    def test_boundary_not_blocked_at_exactly_threshold(self):
        """SPY exactly +0.5% is NOT > 0.5 -> allowed."""
        r = _run(PutMarketDirectionGate(), _flow_put("SPY"), 0.5)
        assert r.result == GateResult.SKIP

    def test_non_index_flow_put_keeps_bypass(self):
        """A non-index flow put (MU) is NOT filtered even on a rally (edge was index-only)."""
        r = _run(PutMarketDirectionGate(), _flow_put("MU"), 1.5)
        assert r.result == GateResult.SKIP

    def test_disabled_flag_keeps_full_bypass(self):
        """Flag off -> flow index put bypassed even on a rally."""
        r = _run(PutMarketDirectionGate(), _flow_put("SPY"), 1.5, ENABLE_FLOW_PUT_MKT_DIR=False)
        assert r.result == GateResult.SKIP

    def test_no_spy_data_does_not_block_flow(self):
        """Missing SPY change -> can't judge direction -> keep the flow bypass (don't fail-closed)."""
        ctx = {"signal": _flow_put("SPY"), "settings": _settings(), "spy_change_from_open": None}
        r = asyncio.run(PutMarketDirectionGate().evaluate(ctx))
        assert r.result == GateResult.SKIP
