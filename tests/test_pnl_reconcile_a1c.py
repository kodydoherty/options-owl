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


class TestA1cResilienceAndEdges:
    """It must NEVER raise, and must handle every messy real-world shape as a safe no-op."""

    @pytest.mark.asyncio
    async def test_never_raises_when_history_call_fails(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=710)
        trader = _trader(db, history=[])
        trader.webull_executor.get_order_history = AsyncMock(side_effect=RuntimeError("boom"))
        # must swallow the error and return 0 — never propagate into the monitor loop
        assert await trader.reconcile_closed_pnl_from_webull() == 0
        assert (await _row(db, 710))["webull_exit_fill_price"] is None   # untouched

    @pytest.mark.asyncio
    async def test_malformed_history_no_crash(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=711, exit_oid="WBEXIT711")
        # non-dicts, missing price, unrelated order — none match, nothing crashes
        trader = _trader(db, history=["nope", 42, {"no_price": True}, {"order_id": "OTHER"}])
        assert await trader.reconcile_closed_pnl_from_webull() == 0

    @pytest.mark.asyncio
    async def test_nested_combo_leg_extraction(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=712, exit_oid="WBEXIT712", entry_fill=2.0, contracts=1)
        history = [{"orders": [{"order_id": "WBEXIT712", "avgFilledPrice": 4.0}]}]  # nested leg
        assert await _trader(db, history=history).reconcile_closed_pnl_from_webull() == 1
        assert (await _row(db, 712))["exit_premium"] == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_dca_uses_blended_entry_basis(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        # DCA: webull_entry_fill_price is only the FIRST fill (2.0); blended avg is 3.0
        async with aiosqlite.connect(db) as conn:
            await conn.execute(
                "INSERT INTO paper_trades (id, ticker, option_type, strike, contracts, "
                "premium_per_contract, webull_entry_fill_price, dca_total_contracts, "
                "total_cost, strategy, status, webull_order_id, webull_exit_order_id, "
                "closed_at, expiry_date, " + _REQ + ") VALUES "
                f"(713, 'SPY', 'call', 500, 2, 3.0, 2.0, 4, 600, 'A', 'closed', 'WB713', "
                f"'WBEXIT713', '{datetime.now(tz=timezone.utc):%Y-%m-%dT%H:%M:%S}', "
                f"'2026-07-02', {_REQV})"
            )
            await conn.commit()
        history = [{"order_id": "WBEXIT713", "avgFilledPrice": 5.0}]
        assert await _trader(db, history=history).reconcile_closed_pnl_from_webull() == 1
        row = await _row(db, 713)
        # pnl uses BLENDED 3.0 not raw 2.0: (5.0 - 3.0) * 2 * 100 = 400
        assert row["pnl_dollars"] == pytest.approx(400.0)

    @pytest.mark.asyncio
    async def test_ignores_open_trades(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        async with aiosqlite.connect(db) as conn:
            await conn.execute(
                "INSERT INTO paper_trades (id, ticker, option_type, strike, contracts, "
                "premium_per_contract, webull_entry_fill_price, total_cost, status, "
                "webull_order_id, webull_exit_order_id, expiry_date, " + _REQ + ") VALUES "
                f"(714, 'SPY', 'call', 500, 1, 2.0, 2.0, 200, 'open', 'WB714', 'WBEXIT714', "
                f"'2026-07-02', {_REQV})"
            )
            await conn.commit()
        # even though history has the fill, an OPEN trade is never touched
        trader = _trader(db, history=[{"order_id": "WBEXIT714", "avgFilledPrice": 9.0}])
        assert await trader.reconcile_closed_pnl_from_webull() == 0

    @pytest.mark.asyncio
    async def test_does_not_mutate_unrelated_columns(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        await _seed(db, tid=715, contracts=3, entry_fill=2.0, exit_oid="WBEXIT715")
        before = await _cols(db, 715, "status, contracts, webull_order_id, webull_exit_order_id, "
                                      "premium_per_contract, strike, entry_price")
        await _trader(db, history=[{"order_id": "WBEXIT715", "avgFilledPrice": 4.0}]
                      ).reconcile_closed_pnl_from_webull()
        after = await _cols(db, 715, "status, contracts, webull_order_id, webull_exit_order_id, "
                                     "premium_per_contract, strike, entry_price")
        assert before == after   # identity/state columns untouched — only P&L changed


class TestA1cCannotBreakTrading:
    """Source-safety invariants — PROVE A1c can never affect the trading path (the pattern the
    UnboundLocalError bug taught us to enforce statically)."""

    def _src(self):
        import inspect
        return inspect.getsource(PaperTrader.reconcile_closed_pnl_from_webull)

    def test_only_updates_pnl_display_columns(self):
        src = self._src()
        forbidden = ("status", "contracts", "webull_order_id", "webull_exit_order_id",
                     "closed_at", "entry_price", "strike", "premium_per_contract",
                     "opened_at", "expiry_date")
        for chunk in src.split("UPDATE paper_trades SET")[1:]:
            head = chunk.split("WHERE")[0]
            for col in forbidden:
                assert f"{col} =" not in head, f"reconcile UPDATE writes '{col}' — trading column!"

    def test_only_reads_order_history_never_places_orders(self):
        import re
        src = self._src()
        calls = set(re.findall(r"webull_executor\.(\w+)", src))
        assert calls <= {"get_order_history"}, f"A1c calls unexpected executor method(s): {calls}"
        for danger in ("sell_option", "buy_option", "place_option_order", "place_order",
                       "cancel_order", "close_webull_position"):
            assert danger not in src, f"A1c references order-placement call '{danger}'"

    def test_guards_run_before_any_webull_call(self):
        src = self._src()
        gi = src.index("get_order_history")
        assert src.index("ENABLE_WEBULL_PNL_RECONCILE") < gi  # flag gate first
        assert src.index("PAPER_TRADE") < gi                  # paper/executor gate first

    def test_body_wrapped_so_it_never_raises(self):
        src = self._src()
        assert "try:" in src and "except Exception" in src, "reconcile body not exception-wrapped"

    def test_monitor_hook_is_gated_timeout_bounded_and_wrapped(self):
        import inspect

        from options_owl.execution import position_monitor
        src = inspect.getsource(position_monitor.run_position_monitor)
        assert "reconcile_closed_pnl_from_webull" in src
        i = src.index("reconcile_closed_pnl_from_webull")
        window = src[max(0, i - 800):i + 500]
        assert "ENABLE_WEBULL_PNL_RECONCILE" in window, "A1c hook not flag-gated"
        assert "asyncio.wait_for" in window, "A1c hook not timeout-bounded"
        assert "except" in window, "A1c hook not exception-wrapped"


async def _cols(db, tid, cols):
    async with aiosqlite.connect(db) as conn:
        conn.row_factory = aiosqlite.Row
        return dict(await (await conn.execute(
            f"SELECT {cols} FROM paper_trades WHERE id = ?", (tid,))).fetchone())
