"""Flat premium cap on FLOW CALLS (FLOW_CALL_MAX_PREMIUM).

Validated 2026-06-24: expensive flow calls are net losers; a $9 cap added +$3.3k/64d (PF 1.40->1.50)
and blocks the LRCX-type −$445 single-trade tail. Calls only, flow only, independent of the tiered
V6 cap (which stays disabled).
"""

from types import SimpleNamespace

import pytest

from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)
from options_owl.risk import pipeline


def _gate():
    for name in dir(pipeline):
        obj = getattr(pipeline, name)
        if isinstance(obj, type) and getattr(obj, "name", "") == "v6_premium_cap":
            return obj()
    raise AssertionError("v6_premium_cap gate not found")


def _sig(direction, premium, source, ticker="LRCX"):
    return TradeSignal(
        ticker=ticker, sentiment=Sentiment.BULLISH, direction=direction, score=90,
        strength=SignalStrength.STRONG, entry_price=372.5, target_price=380,
        expected_move_pct=0.5, strike=372.5, expiry="2026-06-26", risk_reward=2.0,
        atm_strike=372.5, atm_premium=premium, otm_strike=375, otm_premium=premium * 0.4,
        bot_source=source,
    )


async def _run(sig, cap=9.0):
    st = SimpleNamespace(FLOW_CALL_MAX_PREMIUM=cap, ENABLE_V6_PREMIUM_CAP=False)
    return (await _gate().evaluate({"settings": st, "signal": sig})).result.name


@pytest.mark.asyncio
async def test_flow_call_over_cap_blocked():
    # The LRCX-type case: flow call at $12.55 > $9 → FAIL (blocked).
    assert await _run(_sig(Direction.CALL, 12.55, BotSource.UW_FLOW)) == "FAIL"


@pytest.mark.asyncio
async def test_flow_call_under_cap_allowed():
    # $8 flow call <= $9 → not failed by the flat cap (SKIP since tiered cap disabled).
    assert await _run(_sig(Direction.CALL, 8.00, BotSource.UW_FLOW)) == "SKIP"


@pytest.mark.asyncio
async def test_ml_call_exempt():
    # Cap is flow-only — an ML call at $12.55 is NOT blocked.
    assert await _run(_sig(Direction.CALL, 12.55, BotSource.ML_SOURCING)) == "SKIP"


@pytest.mark.asyncio
async def test_flow_put_exempt():
    # Cap is calls-only — a flow PUT at $12.55 is NOT blocked.
    assert await _run(_sig(Direction.PUT, 12.55, BotSource.UW_FLOW)) == "SKIP"


@pytest.mark.asyncio
async def test_cap_off_allows_expensive_flow_call():
    # FLOW_CALL_MAX_PREMIUM=0 (default/off) → no flat cap, expensive flow call passes.
    assert await _run(_sig(Direction.CALL, 12.55, BotSource.UW_FLOW), cap=0.0) == "SKIP"
