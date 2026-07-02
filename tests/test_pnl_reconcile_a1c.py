"""A1c — in-process Webull P&L reconcile (2026-07-02).

When a Webull close 429'd/timed out on the fill lookup, close_webull_position stored the sell
order id (webull_exit_order_id) but left webull_exit_fill_price NULL and exit_premium at the
market approximation. A1c re-fetches the real fill from Webull ORDER HISTORY (one call) and
penny-corrects exit_premium + pnl_dollars + pnl_pct.

Safety contract (locked here): no-op when the flag is off / paper mode / no executor / fill not
yet in history / already reconciled. It only ever UPDATEs P&L display columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from options_owl.execution.paper_trader import PaperTrader, init_paper_db

_REQ = "signal_id, direction, sentiment, score, strength, bot_source, entry_price, opened_at"
_REQV = "1, 'PUT', 'bearish', 90, 'strong', 'uw_flow', 985.0, '2026-07-02T10:00:00'"


async def _seed(db, *, tid=700, contracts=2, entry_fill=2.00, stale_exit=9.99,
                stale_pnl=1598.0, exit_oid="WBEXIT700", exit_fill_captured=None,
                closed_at=None):
    """A CLOSED trade as a 429'd close would leave it: sell order id present, exit fill NULL,
    exit_premium at the stale approximation."""
    closed_at = closed_at or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    async with aiosqlite.connect(db) as conn:
        efc = "NULL" if exit_fill_captured is None else str(exit_fill_captured)
        await conn.execute(
            "INSERT INTO paper_trades (id, ticker, option_type, strike, contracts, "
            "premium_per_contract, webull_entry_fill_price, exit_premium, pnl_dollars, "
            "total_cost, strategy, status, webull_order_id, webull_exit_order_id, "
            "webull_exit_fill_price, closed_at, expiry_date, " + _REQ + ") VALUES "
            f"({tid}, 'SPY', 'call', 500.0, {contracts}, {entry_fill}, {entry_fill}, "
            f"{stale_exit}, {stale_pnl}, {entry_fill * contracts * 100}, 'A', 'closed', "
            f"'WB{tid}', '{exit_oid}', {efc}, '{closed_at}', '2026-07-02', {_REQV})"
        )
        await conn.commit()


def _trader(db, *, history=None, flag=True, paper=False, executor=True):
    settings = MagicMock()
    settings.PAPER_TRADE = paper
    settings.ENABLE_WEBULL_PNL_RECONCILE = flag
    t = PaperTrader.__new__(PaperTrader)
    t.settings = settings
    t.db_path = db
    if executor:
        ex = MagicMock()
        ex.get_order_history = AsyncMock(return_value=history or [])
        t.webull_executor = ex
    else:
        t.webull_executor = None
    return t


async def _row(db, tid):
    async with aiosqlite.connect(db) as conn:
        conn.row_factory = aiosqlite.Row
        return dict(await (await conn.execute(
            "SELECT exit_premium, pnl_dollars, pnl_pct, webull_exit_fill_price "
            "FROM paper_trades WHERE id = ?", (tid,))).fetchone())


class TestA1cReconcile:
    @pytest.mark.asyncio
    async def test_corrects_from_real_fill(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        # stale: exit_premium 9.99, pnl 1598; real exit fill was 5.50
        await _seed(db, tid=700, contracts=2, entry_fill=2.00, stale_exit=9.99, stale_pnl=1598.0)
        history = [{"order_id": "WBEXIT700", "avgFilledPrice": "5.50"}]
        trader = _trader(db, history=history)

        n = await trader.reconcile_closed_pnl_from_webull()
        assert n == 1
        row = await _row(db, 700)
        assert row["webull_exit_fill_price"] == pytest.approx(5.50)
        assert row["exit_premium"] == pytest.approx(5.50)                 # aligned to fill
        # pnl = (5.50 - 2.00) * 2 * 100 = 700
        assert row["pnl_dollars"] == pytest.approx(700.0)
        assert row["pnl_pct"] == pytest.approx((5.50 - 2.00) / 2.00 * 100)

    @pytest.mark.asyncio
    async def test_matches_by_client_order_id_too(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=701, exit_oid="COID701")
        history = [{"client_order_id": "COID701", "filledPrice": 3.0}]
        n = await _trader(db, history=history).reconcile_closed_pnl_from_webull()
        assert n == 1

    @pytest.mark.asyncio
    async def test_noop_when_flag_off(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=702)
        trader = _trader(db, history=[{"order_id": "WBEXIT702", "avgFilledPrice": 5.5}], flag=False)
        assert await trader.reconcile_closed_pnl_from_webull() == 0
        trader.webull_executor.get_order_history.assert_not_called()   # never touches Webull

    @pytest.mark.asyncio
    async def test_noop_paper_mode_and_no_executor(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=703)
        assert await _trader(db, paper=True, history=[]).reconcile_closed_pnl_from_webull() == 0
        assert await _trader(db, executor=False).reconcile_closed_pnl_from_webull() == 0

    @pytest.mark.asyncio
    async def test_skips_when_fill_absent_from_history(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=704, exit_oid="WBEXIT704", stale_exit=9.99)
        # history has a DIFFERENT order — no match → leave the trade untouched
        trader = _trader(db, history=[{"order_id": "SOMETHING_ELSE", "avgFilledPrice": 5.5}])
        assert await trader.reconcile_closed_pnl_from_webull() == 0
        row = await _row(db, 704)
        assert row["webull_exit_fill_price"] is None          # unchanged
        assert row["exit_premium"] == pytest.approx(9.99)     # stale, but not corrupted

    @pytest.mark.asyncio
    async def test_idempotent_already_reconciled(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        # already has a captured exit fill → not eligible
        await _seed(db, tid=705, exit_fill_captured=5.50)
        trader = _trader(db, history=[{"order_id": "WBEXIT705", "avgFilledPrice": 6.0}])
        assert await trader.reconcile_closed_pnl_from_webull() == 0
        assert (await _row(db, 705))["webull_exit_fill_price"] == pytest.approx(5.50)
