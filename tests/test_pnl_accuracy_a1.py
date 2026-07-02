"""A1 forward-capture — recorded P&L must match the REAL Webull fill (2026-07-02).

Root cause of the adam +$538-DB-vs--$125-real gap: on a Webull close, close_trade booked an
approximate/simulated exit_premium (and credited the portfolio with it), while
close_webull_position later fixed pnl_dollars from the real fill but LEFT exit_premium at the
approximation — so the two disagreed by construction, and any consumer reading exit_premium saw
a fiction.

A1a fix: close_webull_position now writes exit_premium = the real Webull exit fill in the same
UPDATE that fixes pnl_dollars — for the main row, the scale-out child row, and the
entry-fill-unavailable branch. These tests lock that in with a real-DB round-trip.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from options_owl.execution.paper_trader import (
    PaperTrader,
    SellOutcome,
    init_paper_db,
)
from options_owl.execution.webull_executor import OrderResult

_REQ = ("signal_id, direction, sentiment, score, strength, bot_source, entry_price, opened_at")
_REQV = "1, 'PUT', 'bearish', 90, 'strong', 'uw_flow', 985.0, '2026-07-02T10:00:00'"


async def _seed_closed_trade(db, *, tid=500, contracts=1, entry_fill=11.54,
                             stale_exit=12.00, stale_pnl=46.0, parent=None):
    """Insert a CLOSED trade as close_trade would have left it — with a STALE exit_premium
    (the market approximation) and possibly-wrong pnl_dollars, plus a real entry fill."""
    async with aiosqlite.connect(db) as conn:
        cols = ("id, ticker, option_type, strike, contracts, premium_per_contract, "
                "webull_entry_fill_price, exit_premium, pnl_dollars, total_cost, strategy, "
                "status, webull_order_id, expiry_date, " + _REQ)
        parent_col = ", parent_trade_id" if parent is not None else ""
        parent_val = f", {parent}" if parent is not None else ""
        await conn.execute(
            f"INSERT INTO paper_trades ({cols}{parent_col}) VALUES "
            f"({tid}, 'MU', 'put', 985.0, {contracts}, {entry_fill}, {entry_fill}, "
            f"{stale_exit}, {stale_pnl}, {entry_fill * contracts * 100}, 'A', 'closed', "
            f"'WB{tid}', '2026-07-02', {_REQV}{parent_val})"
        )
        await conn.commit()


def _trader(db, executor):
    settings = MagicMock()
    settings.PAPER_TRADE = False
    settings.ENABLE_FAST_EXIT_CHASE = False  # MagicMock getattr is truthy — must pin it off
    settings.POLYGON_API_KEY = ""
    t = PaperTrader.__new__(PaperTrader)
    t.settings = settings
    t.webull_executor = executor
    t.db_path = db
    return t


def _filled_executor(exit_fill: float, contracts: int = 1):
    ex = MagicMock()
    ex.sell_option = AsyncMock(return_value=OrderResult(
        success=True, order_id="EXIT1", client_order_id="COID1",
        fill_status="FILLED", filled_quantity=contracts))
    ex.get_fill_price = AsyncMock(return_value=exit_fill)
    ex.get_open_orders = AsyncMock(return_value=[])
    ex.cancel_order = AsyncMock(return_value=True)
    return ex


async def _row(db, tid):
    async with aiosqlite.connect(db) as conn:
        conn.row_factory = aiosqlite.Row
        return dict(await (await conn.execute(
            "SELECT exit_premium, pnl_dollars, webull_exit_fill_price FROM paper_trades "
            "WHERE id = ?", (tid,))).fetchone())


class TestExitPremiumAlignsToRealFill:
    @pytest.mark.asyncio
    async def test_main_row_exit_premium_matches_fill(self, tmp_path):
        """The core fix: exit_premium must become the REAL fill, consistent with pnl_dollars."""
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed_closed_trade(db, tid=500, entry_fill=11.54, stale_exit=12.00, stale_pnl=46.0)

        trader = _trader(db, _filled_executor(5.50))
        # pin the fresh bid so we exercise the fill-reconcile, not the quote path
        trader._get_fresh_option_bid = (lambda self, trade: _aval(5.50)).__get__(trader, PaperTrader)

        res = await trader.close_webull_position(
            {"id": 500, "ticker": "MU", "strike": 985.0, "option_type": "put",
             "expiry_date": "2026-07-02", "contracts": 1, "webull_order_id": "WB500",
             "sell_retry_count": 0, "premium_per_contract": 11.54},
            5.50,
        )
        assert res.outcome is SellOutcome.FILLED
        row = await _row(db, 500)
        # exit_premium was $12.00 (stale) → must now equal the real $5.50 fill
        assert row["exit_premium"] == pytest.approx(5.50)
        assert row["webull_exit_fill_price"] == pytest.approx(5.50)
        # pnl_dollars recomputed from real fills: (5.50 - 11.54) * 1 * 100 = -604
        assert row["pnl_dollars"] == pytest.approx(-604.0)
        # and the two now AGREE: pnl == (exit_premium - entry) * ct * 100
        assert row["pnl_dollars"] == pytest.approx((row["exit_premium"] - 11.54) * 1 * 100)

    @pytest.mark.asyncio
    async def test_scaleout_child_row_also_aligns(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed_closed_trade(db, tid=600)                       # parent
        await _seed_closed_trade(db, tid=601, stale_exit=9.99, parent=600)  # child

        trader = _trader(db, _filled_executor(5.50))
        trader._get_fresh_option_bid = (lambda self, trade: _aval(5.50)).__get__(trader, PaperTrader)

        await trader.close_webull_position(
            {"id": 600, "ticker": "MU", "strike": 985.0, "option_type": "put",
             "expiry_date": "2026-07-02", "contracts": 1, "webull_order_id": "WB600",
             "sell_retry_count": 0, "premium_per_contract": 11.54},
            5.50, child_trade_id=601,
        )
        child = await _row(db, 601)
        assert child["exit_premium"] == pytest.approx(5.50)         # child aligned too
        assert child["webull_exit_fill_price"] == pytest.approx(5.50)

    def test_all_three_close_updates_set_exit_premium(self):
        """Source-safety: every UPDATE in the fill-reconcile block that writes
        pnl_dollars/webull_exit_fill_price must ALSO write exit_premium — so no close
        path can leave the stale approximation behind again."""
        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "webull_exit_fill_price" in src
        # every UPDATE that writes webull_exit_fill_price must ALSO write exit_premium
        n_exit_fill = src.count("SET webull_exit_fill_price")
        n_with_exit_prem = sum(
            1 for part in src.split("SET webull_exit_fill_price")[1:]
            if "exit_premium = ?" in part.split("WHERE id")[0]
        )
        assert n_with_exit_prem == n_exit_fill, (
            f"{n_exit_fill} exit-fill UPDATEs but only {n_with_exit_prem} also set exit_premium — "
            "a close path can leave exit_premium stale"
        )


async def _aval(v):
    return v
