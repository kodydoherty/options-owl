"""The runner_v1 Q1 floor gate must actually block. It had no test.

WHY THIS EXISTS
---------------
RUNNER_V1_MIN_P=0.39 is LIVE on kody and dennis and SKIPS a trade outright when
P(runner) falls below it. It had never fired in production and had no test, so there
was no evidence in either direction that the wiring worked.

It turned out not to have fired for an innocent reason: rebuilding P(runner) for every
August trade shows the sub-0.39 ones all landed on 08-04 and 08-06, and nothing below
0.61 has appeared since the gate deployed on 08-07. Pre-gate they were ~16% of trades,
so the opportunity is real and simply has not recurred yet.

That is an argument for testing it rather than waiting: a gate on the live-money entry
path that has never executed is indistinguishable from a broken one until it matters.
These drive the real evaluate_and_trade so the gate is exercised where it actually sits,
not re-implemented in the test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.test_partial_profits import (
    _attach_mock_candle_cache,
    _make_settings,
    _make_signal,
)

from options_owl.execution.paper_trader import PaperTrader


def _settings(tmp_db_path: str, floor: float):
    """Vinny/score sizing ON so the runner_v1 block is reachable, plus the floor."""
    return _make_settings(
        tmp_db_path,
        ENABLE_VINNY_STRATEGY=True,
        ENABLE_SCORE_SIZING=True,
        ENABLE_RUNNER_V1_SIZING=True,
        RUNNER_V1_MIN_P=floor,
    )


async def _trader(tmp_db_path: str, floor: float) -> PaperTrader:
    trader = PaperTrader(_settings(tmp_db_path, floor))
    await trader.init()
    _attach_mock_candle_cache(trader)
    return trader


class TestRunnerV1FloorGate:
    @pytest.mark.asyncio
    async def test_below_floor_is_skipped(self, tmp_db_path):
        """p_runner under the floor must abort the entry, not merely shrink it.

        The comment in paper_trader is explicit that x0.7 sizing still books the loss,
        so a 'sized down but taken' outcome is a real failure, not a near-miss.
        """
        trader = await _trader(tmp_db_path, 0.39)
        with patch(
            "options_owl.risk.flow_runner.compute_runner_v1_p",
            new=AsyncMock(return_value=0.20),
        ):
            result = await trader.evaluate_and_trade(_make_signal(score=130), signal_id=1)
        assert result is None, "sub-floor P(runner) must skip the trade entirely"

    @pytest.mark.asyncio
    async def test_above_floor_is_not_blocked_by_this_gate(self, tmp_db_path):
        """A healthy P(runner) must not be rejected FOR THIS REASON.

        Asserted on the rejection reason rather than on a trade being opened: other
        gates may independently decline in a test fixture, and asserting 'a trade
        happened' would make this test fail for unrelated changes.
        """
        trader = await _trader(tmp_db_path, 0.39)
        with patch(
            "options_owl.risk.flow_runner.compute_runner_v1_p",
            new=AsyncMock(return_value=0.85),
        ):
            await trader.evaluate_and_trade(_make_signal(score=130), signal_id=2)

        import aiosqlite

        async with aiosqlite.connect(tmp_db_path) as conn:
            rows = await conn.execute(
                "SELECT COUNT(*) FROM trade_events WHERE detail LIKE '%runner_v1_floor%'"
            )
            (n,) = await rows.fetchone()
        assert n == 0, "above-floor P(runner) was rejected by the runner_v1 floor"

    @pytest.mark.asyncio
    async def test_floor_of_zero_disables_the_gate(self, tmp_db_path):
        """RUNNER_V1_MIN_P=0 is the documented off switch — it must not block anything."""
        trader = await _trader(tmp_db_path, 0.0)
        with patch(
            "options_owl.risk.flow_runner.compute_runner_v1_p",
            new=AsyncMock(return_value=0.01),
        ):
            await trader.evaluate_and_trade(_make_signal(score=130), signal_id=3)

        import aiosqlite

        async with aiosqlite.connect(tmp_db_path) as conn:
            rows = await conn.execute(
                "SELECT COUNT(*) FROM trade_events WHERE detail LIKE '%runner_v1_floor%'"
            )
            (n,) = await rows.fetchone()
        assert n == 0, "floor=0 must disable the gate, but it still rejected"

    @pytest.mark.asyncio
    async def test_unavailable_p_runner_does_not_block(self, tmp_db_path):
        """compute_runner_v1_p returns None on missing data — that must stay a no-op.

        The scorer abstains (no greeks, PUT, harvester gap) far more often than it
        returns a low score. If None were treated as 'below the floor' the gate would
        silently halt trading whenever Redis or the candle feed hiccuped.
        """
        trader = await _trader(tmp_db_path, 0.39)
        with patch(
            "options_owl.risk.flow_runner.compute_runner_v1_p",
            new=AsyncMock(return_value=None),
        ):
            await trader.evaluate_and_trade(_make_signal(score=130), signal_id=4)

        import aiosqlite

        async with aiosqlite.connect(tmp_db_path) as conn:
            rows = await conn.execute(
                "SELECT COUNT(*) FROM trade_events WHERE detail LIKE '%runner_v1_floor%'"
            )
            (n,) = await rows.fetchone()
        assert n == 0, "None P(runner) must not be treated as sub-floor"
