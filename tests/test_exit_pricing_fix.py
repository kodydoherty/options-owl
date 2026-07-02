"""Exit-pricing fixes (2026-07-02) — root cause of adam's orphaned MU #399 bleed.

MU $985 put: Polygon quoted a stale ~$12 bid while Webull's real market was ~$5.25.
The exit priced sells off the $12 (bid-20% = $9.76), which never crossed the real
$5.25 book → 39 no-fills → the position orphaned and bled from -19% to -51%, and the
DB flip-flopped a fabricated "+$59 profit_lock" close over a live -$629 position.

Fix #1: _get_fresh_option_bid sources the exit bid from Webull's OWN venue quote
        (what actually fills) first, Polygon only as a fallback.
Fix #2: the sell escalation keeps crossing DEEPER on repeated no-fills (past -20%,
        down to a -50% floor) so a thin/wide book always fills instead of orphaning.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.execution.paper_trader import (
    PaperTrader,
    SellOutcome,
)
from options_owl.execution.webull_executor import OrderResult


def _make_trader(executor) -> PaperTrader:
    settings = MagicMock()
    settings.PAPER_TRADE = False
    settings.POLYGON_API_KEY = ""  # force Polygon fallback to no-op unless overridden
    trader = PaperTrader.__new__(PaperTrader)  # bypass __init__/DB
    trader.settings = settings
    trader.webull_executor = executor
    trader.db_path = ":memory:"
    return trader


def _trade(retry=0) -> dict:
    return {
        "id": 399,
        "ticker": "MU",
        "strike": 985.0,
        "option_type": "put",
        "expiry_date": "2026-07-02",
        "contracts": 1,
        "webull_order_id": "WB399",
        "sell_retry_count": retry,
        "premium_per_contract": 11.54,
    }


# ---------------------------------------------------------------------------
# Fix #1 — exit bid sourced from the Webull venue, not stale Polygon
# ---------------------------------------------------------------------------

class TestFreshBidPrefersWebullVenue:
    @pytest.mark.asyncio
    async def test_uses_webull_bid_when_available(self):
        """The real venue bid ($5.25) must win over anything Polygon would say."""
        executor = MagicMock()
        executor.get_option_quote = AsyncMock(
            return_value={"bid": 5.25, "ask": 5.60, "mid": 5.42}
        )
        trader = _make_trader(executor)
        bid = await trader._get_fresh_option_bid(_trade())
        assert bid == 5.25
        executor.get_option_quote.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_webull_mid_discounted_when_no_bid(self):
        executor = MagicMock()
        executor.get_option_quote = AsyncMock(
            return_value={"bid": 0, "ask": 5.60, "mid": 5.00}
        )
        trader = _make_trader(executor)
        bid = await trader._get_fresh_option_bid(_trade())
        assert bid == pytest.approx(5.00 * 0.98)

    @pytest.mark.asyncio
    async def test_falls_back_to_polygon_when_venue_unavailable(self):
        """No executor → must still try Polygon (returns None here since no API key)."""
        trader = _make_trader(None)
        bid = await trader._get_fresh_option_bid(_trade())
        assert bid is None  # no venue, no polygon key → None (caller uses exit_premium)

    @pytest.mark.asyncio
    async def test_venue_exception_falls_through_to_polygon(self):
        executor = MagicMock()
        executor.get_option_quote = AsyncMock(side_effect=RuntimeError("socket dead"))
        trader = _make_trader(executor)
        # No polygon key → falls through to None, but must NOT raise.
        bid = await trader._get_fresh_option_bid(_trade())
        assert bid is None


# ---------------------------------------------------------------------------
# Fix #2 — escalation keeps crossing deeper so a thin book always fills
# ---------------------------------------------------------------------------

class TestExitEscalationDeepens:
    async def _sell_price_at_retry(self, retry: int, base_bid: float = 5.25) -> float:
        """Drive close_webull_position once at a given retry count and capture the
        limit price it submits to the executor."""
        captured = {}

        async def fake_sell(**kwargs):
            captured["limit"] = kwargs["limit_price"]
            # Report not-filled so the ladder logic is what we're measuring.
            return OrderResult(success=False, error="not filled", fill_status="SUBMITTED")

        executor = MagicMock()
        executor.sell_option = AsyncMock(side_effect=fake_sell)
        executor.get_open_orders = AsyncMock(return_value=[])
        executor.cancel_order = AsyncMock(return_value=True)
        trader = _make_trader(executor)

        # Pin the fresh bid to a known venue value so we measure the ladder, not the quote.
        async def fixed_bid(self, trade):
            return base_bid
        trader._get_fresh_option_bid = fixed_bid.__get__(trader, PaperTrader)

        # Avoid real DB writes.
        import options_owl.execution.paper_trader as pt
        orig = pt._db_execute_with_retry

        async def _noop(*a, **k):
            return None
        pt._db_execute_with_retry = _noop
        try:
            res = await trader.close_webull_position(_trade(retry=retry), base_bid)
        finally:
            pt._db_execute_with_retry = orig
        assert res.outcome is SellOutcome.NOT_FILLED
        return captured["limit"]

    @pytest.mark.asyncio
    async def test_ladder_matches_known_tiers(self):
        base = 5.25
        assert await self._sell_price_at_retry(0, base) == pytest.approx(base)         # fresh bid
        assert await self._sell_price_at_retry(1, base) == pytest.approx(base * 0.95)  # -5%
        assert await self._sell_price_at_retry(2, base) == pytest.approx(base * 0.90)  # -10%
        assert await self._sell_price_at_retry(3, base) == pytest.approx(base * 0.85)  # -15%
        assert await self._sell_price_at_retry(4, base) == pytest.approx(base * 0.80)  # -20%

    @pytest.mark.asyncio
    async def test_ladder_keeps_deepening_past_retry_4(self):
        """The MU #399 bug: parked at -20% forever. Now it must keep crossing deeper."""
        base = 5.25
        assert await self._sell_price_at_retry(5, base) == pytest.approx(base * 0.75)  # -25%
        assert await self._sell_price_at_retry(6, base) == pytest.approx(base * 0.70)  # -30%
        assert await self._sell_price_at_retry(10, base) == pytest.approx(base * 0.50)  # -50% floor

    @pytest.mark.asyncio
    async def test_ladder_floors_at_minus_50pct(self):
        base = 5.25
        # Retry 20 would be -100% naively; must clamp at the -50% floor.
        assert await self._sell_price_at_retry(20, base) == pytest.approx(base * 0.50)
