"""Orphan-safety for the scale-out / partial close paths (2026-07-02).

Before this fix, only the main close-all path reopened a trade when its live Webull
sell failed. The scale-out / target / milestone / tranche paths recorded the DB close
(full or partial) and IGNORED the Webull sell result — so a failed sell left the
position (or the scaled-out fraction) orphaned on Webull while the DB thought it sold.

Now every close routes through _finalize_full_close / _finalize_partial_close:
  - full close: revert + reopen the trade on a transient sell failure
  - partial close: REVERT the partial (restore parent contracts, delete child) on a
    transient sell failure so the scaled-out fraction can't orphan
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from options_owl.execution import position_monitor as pm
from options_owl.execution.paper_trader import (
    PaperTrader,
    SellOutcome,
    SellResult,
    init_paper_db,
)


# --- a fake async _connect_db so the helpers' DB reads/writes are no-ops -------

class _FakeCursor:
    def __init__(self, row=None):
        self._row = row

    async def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row=None):
        self._row = row
        self.executed = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return _FakeCursor(self._row)

    async def commit(self):
        pass


class _FakeCM:
    def __init__(self, conn):
        self._c = conn

    async def __aenter__(self):
        return self._c

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def fake_db(monkeypatch):
    conn = _FakeConn(row=None)  # contract re-read returns nothing → no DCA adjust
    monkeypatch.setattr(pm, "_connect_db", lambda p: _FakeCM(conn))
    return conn


def _trade(**kw):
    d = {
        "id": 1, "ticker": "MU", "strike": 985.0, "option_type": "put",
        "contracts": 3, "webull_order_id": "WB1", "sell_retry_count": 0,
        "premium_per_contract": 11.54,
    }
    d.update(kw)
    return d


def _pt():
    pt = MagicMock()
    pt.settings = MagicMock()
    return pt


# ---------------------------------------------------------------------------
# _finalize_full_close
# ---------------------------------------------------------------------------

class TestFinalizeFullClose:
    @pytest.mark.asyncio
    async def test_reopens_and_reverts_on_transient_sell_failure(self, fake_db):
        pt = _pt()
        pt.close_trade = AsyncMock(return_value={"strategy": "B", "proceeds": 100.0, "pnl": -50.0})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.TRANSIENT_ERROR))
        pt.revert_close_effects = AsyncMock()
        done = await pm._finalize_full_close(pt, _trade(), 986.0, 5.25, "premium_hardstop", ":m:", None)
        assert done is False  # reopened for retry → caller continues
        pt.revert_close_effects.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_done_and_no_revert_on_fill(self, fake_db):
        pt = _pt()
        pt.close_trade = AsyncMock(return_value={"strategy": "B", "proceeds": 100.0, "pnl": -50.0})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.FILLED))
        pt.revert_close_effects = AsyncMock()
        done = await pm._finalize_full_close(pt, _trade(), 986.0, 5.25, "stop", ":m:", None)
        assert done is True
        pt.revert_close_effects.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_paper_trade_no_webull_id_is_done(self, fake_db):
        pt = _pt()
        pt.close_trade = AsyncMock(return_value={"strategy": "B", "proceeds": 0.0, "pnl": 0.0})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.TRANSIENT_ERROR))
        pt.revert_close_effects = AsyncMock()
        done = await pm._finalize_full_close(pt, _trade(webull_order_id=None), 986.0, 5.25, "stop", ":m:", None)
        assert done is True  # paper trade — nothing to reopen
        pt.revert_close_effects.assert_not_awaited()


# ---------------------------------------------------------------------------
# _finalize_partial_close
# ---------------------------------------------------------------------------

class TestFinalizePartialClose:
    @pytest.mark.asyncio
    async def test_reverts_partial_on_transient_sell_failure(self, fake_db):
        pt = _pt()
        pt.partial_close_trade = AsyncMock(return_value={"contracts_closed": 1, "child_trade_id": 99})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.TRANSIENT_ERROR))
        pt.revert_partial_close = AsyncMock()
        done = await pm._finalize_partial_close(pt, _trade(), 986.0, 5.25, "t1_hit", 33.0, ":m:", None)
        assert done is True  # parent restored to full size; monitor re-evaluates
        pt.revert_partial_close.assert_awaited_once_with(99)

    @pytest.mark.asyncio
    async def test_no_revert_on_fill(self, fake_db):
        pt = _pt()
        pt.partial_close_trade = AsyncMock(return_value={"contracts_closed": 1, "child_trade_id": 99})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.FILLED))
        pt.revert_partial_close = AsyncMock()
        done = await pm._finalize_partial_close(pt, _trade(), 986.0, 5.25, "t1_hit", 33.0, ":m:", None)
        assert done is True
        pt.revert_partial_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_revert_when_position_genuinely_gone(self, fake_db):
        pt = _pt()
        pt.partial_close_trade = AsyncMock(return_value={"contracts_closed": 1, "child_trade_id": 99})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.POSITION_NOT_FOUND))
        pt.revert_partial_close = AsyncMock()
        done = await pm._finalize_partial_close(pt, _trade(), 986.0, 5.25, "t1_hit", 33.0, ":m:", None)
        assert done is True  # the fraction is really gone from Webull — DB reflects reality
        pt.revert_partial_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rounding_collapse_delegates_to_guarded_full_close(self, fake_db):
        """When partial rounds to a full close, it must use the guarded full-close
        path (which reopens on failure), not an unguarded sell."""
        pt = _pt()
        pt.partial_close_trade = AsyncMock(return_value={"trade_id": 1})  # no contracts_closed → full close
        pt.close_trade = AsyncMock(return_value={"strategy": "B", "proceeds": 0.0, "pnl": 0.0})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.TRANSIENT_ERROR))
        pt.revert_close_effects = AsyncMock()
        done = await pm._finalize_partial_close(pt, _trade(), 986.0, 5.25, "t1_hit", 33.0, ":m:", None)
        assert done is False  # delegated full close reopened → caller continues
        pt.close_trade.assert_awaited_once()
        pt.revert_close_effects.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_applies_pre_sell_updates_before_the_sell(self, fake_db):
        pt = _pt()
        pt.partial_close_trade = AsyncMock(return_value={"contracts_closed": 1, "child_trade_id": 99})
        pt.close_webull_position = AsyncMock(return_value=SellResult(SellOutcome.FILLED))
        pt.revert_partial_close = AsyncMock()
        await pm._finalize_partial_close(
            pt, _trade(), 986.0, 5.25, "t2_hit", 33.0, ":m:", None,
            pre_sell_updates=("UPDATE paper_trades SET last_target_hit = ? WHERE id = ?", (2, 1)),
        )
        assert any("last_target_hit" in sql for sql, _ in fake_db.executed)


# ---------------------------------------------------------------------------
# revert_partial_close — real DB round-trip
# ---------------------------------------------------------------------------

class TestRevertPartialCloseRealDB:
    @pytest.mark.asyncio
    async def test_restores_parent_and_deletes_child(self, tmp_path):
        db = str(tmp_path / "t.db")
        await init_paper_db(db)
        pt = PaperTrader.__new__(PaperTrader)
        pt.settings = MagicMock()
        pt.db_path = db
        pt.webull_executor = None

        async with aiosqlite.connect(db) as conn:
            # portfolio: strategy B, one recorded loss + credited proceeds from the partial
            await conn.execute(
                "INSERT INTO paper_portfolio (strategy, starting_balance, current_balance, "
                "daily_pnl, wins, losses, created_at) "
                "VALUES ('B', 10000, 10550, -50, 0, 1, '2026-07-02T10:00:00')"
            )
            _req = ("signal_id, direction, sentiment, score, strength, bot_source, "
                    "entry_price, opened_at")
            _reqv = "1, 'PUT', 'bearish', 90, 'strong', 'uw_flow', 986.0, '2026-07-02T10:00:00'"
            # parent: was 3 contracts, reduced to 2 by the partial
            await conn.execute(
                f"INSERT INTO paper_trades (id, ticker, option_type, strike, contracts, "
                f"premium_per_contract, total_cost, strategy, status, {_req}) "
                f"VALUES (1, 'MU', 'put', 985.0, 2, 11.54, 2308.0, 'B', 'open', {_reqv})"
            )
            # child: the closed 1-contract partial (exit 5.50, pnl -604), parent_trade_id=1
            await conn.execute(
                f"INSERT INTO paper_trades (id, ticker, option_type, strike, contracts, "
                f"premium_per_contract, exit_premium, pnl_dollars, strategy, status, "
                f"parent_trade_id, total_cost, {_req}) "
                f"VALUES (99, 'MU', 'put', 985.0, 1, 11.54, 5.50, -604.0, 'B', 'closed', "
                f"1, 1154.0, {_reqv})"
            )
            await conn.commit()

        await pt.revert_partial_close(99)

        async with aiosqlite.connect(db) as conn:
            conn.row_factory = aiosqlite.Row
            parent = dict(await (await conn.execute(
                "SELECT contracts, total_cost FROM paper_trades WHERE id = 1")).fetchone())
            child = await (await conn.execute(
                "SELECT id FROM paper_trades WHERE id = 99")).fetchone()
            port = dict(await (await conn.execute(
                "SELECT current_balance, daily_pnl, losses FROM paper_portfolio WHERE strategy='B'")).fetchone())

        assert parent["contracts"] == 3                       # 2 + 1 restored
        assert parent["total_cost"] == pytest.approx(2308.0 + 1154.0)  # + 1 * 11.54 * 100
        assert child is None                                  # child row deleted
        # proceeds credited by the partial = 5.50 * 1 * 100 = 550 → reversed
        assert port["current_balance"] == pytest.approx(10550 - 550)
        assert port["daily_pnl"] == pytest.approx(-50 - (-604.0))
        assert port["losses"] == 0                            # the recorded loss reversed
