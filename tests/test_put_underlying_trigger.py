"""Tests for the PUT underlying-DOWN trigger (2026-07-20).

Require the PUT's OWN underlying to be falling >= |PUT_UNDERLYING_DOWN_TRIGGER|% from open at entry,
applied to ML *and* flow puts (flow otherwise bypasses put_market_direction). Real live data: puts
bought when the ticker was flat/up lost -$2,013/6wk (55% of all put losses)."""
import asyncio
from unittest.mock import MagicMock

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction
from options_owl.risk.pipeline import GateResult, PutMarketDirectionGate


def _put(ticker="NVDA", flow=False):
    sig = MagicMock()
    sig.ticker = ticker
    sig.direction = Direction.PUT
    sig.bot_source = MagicMock(value="uw_flow" if flow else "ml_sourcing")
    return sig


def _settings(**ov):
    d = dict(DISCORD_TOKEN="t", DISCORD_CHANNEL_ID=1,
             ENABLE_PUT_UNDERLYING_TRIGGER=True, PUT_UNDERLYING_DOWN_TRIGGER=-0.5)
    d.update(ov)
    return Settings(**d)


def _run(sig, ticker_change, spy_change=-0.6, **s):
    ctx = {"signal": sig, "settings": _settings(**s),
           "spy_change_from_open": spy_change, "ticker_change_from_open": ticker_change}
    return asyncio.run(PutMarketDirectionGate().evaluate(ctx))


class TestPutUnderlyingTrigger:
    def test_ml_put_blocked_when_ticker_flat_or_up(self):
        """The bleeder: ML put bought when the ticker is +0.3% (not falling) -> FAIL."""
        r = _run(_put("NVDA"), ticker_change=+0.3)
        assert r.result == GateResult.FAIL
        assert "not falling" in r.reason

    def test_flow_put_blocked_when_ticker_up_no_bypass(self):
        """Flow puts must ALSO be blocked when the ticker isn't falling (stops the bypass leak)."""
        r = _run(_put("TSLA", flow=True), ticker_change=+0.2)
        assert r.result == GateResult.FAIL

    def test_put_allowed_when_ticker_falling(self):
        """Ticker down -1.0% (past the -0.5% trigger) -> not blocked by the trigger (ML PUT passes)."""
        r = _run(_put("NVDA"), ticker_change=-1.0)
        assert r.result != GateResult.FAIL

    def test_flow_put_falling_reaches_flow_path(self):
        """Falling flow put clears the trigger and hits the flow bypass (SKIP), not a FAIL."""
        r = _run(_put("TSLA", flow=True), ticker_change=-1.2, ENABLE_FLOW_PUT_MKT_DIR=False)
        assert r.result != GateResult.FAIL

    def test_unknown_ticker_change_fails_closed(self):
        """No underlying data -> block (fail-closed; puts on non-falling names are the bleeder)."""
        r = _run(_put("NVDA"), ticker_change=None)
        assert r.result == GateResult.FAIL

    def test_trigger_off_does_not_block_flat_put(self):
        """Flag off -> the new check is skipped; a flat-ticker flow put bypasses as before."""
        r = _run(_put("TSLA", flow=True), ticker_change=+0.3, ENABLE_PUT_UNDERLYING_TRIGGER=False)
        assert r.result != GateResult.FAIL

    def test_exactly_at_trigger_passes(self):
        """Ticker exactly -0.5% (== trigger) is allowed (need <= -0.5%)."""
        r = _run(_put("NVDA"), ticker_change=-0.5)
        assert r.result != GateResult.FAIL

    def test_call_unaffected(self):
        """A CALL is never touched by the PUT underlying trigger."""
        sig = _put("NVDA")
        sig.direction = Direction.CALL
        r = _run(sig, ticker_change=+0.5)
        assert r.result != GateResult.FAIL
