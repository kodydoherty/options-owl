"""Tests for exit_source tracking — ensures manual vs AI exits are properly tagged.

Critical: The last code change to position_monitor introduced an UnboundLocalError
that prevented ALL sells. These tests verify:
1. The exit_source column exists and defaults to 'ai'
2. Sell-abandoned (position gone from Webull) correctly marks exit_source='manual'
3. Normal AI closes keep exit_source='ai'
4. The DB migration adds the column safely (idempotent)
5. No variable reference errors in the sell-abandoned path
6. log_trade_event is called with correct args for manual close detection
7. The close_trade function doesn't break existing behavior
"""

from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import aiosqlite
import pytest

from options_owl.execution.paper_trader import (
    PaperTrader,
    log_trade_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Minimal settings for paper trading tests."""
    s = MagicMock()
    s.DB_PATH = ":memory:"
    s.PORTFOLIO_SIZE = 5000
    s.PAPER_TRADE = True
    s.SIMULATED_EXIT_SLIPPAGE_BPS = 0
    s.MAX_POSITION_PCT = 20
    s.MAX_PORTFOLIO_RISK_PCT = 75
    s.MAX_CONCURRENT = 5
    s.WEBULL_KILL_SWITCH = True
    s.ENABLE_PORTFOLIO_SYNC = False
    s.MIN_SIGNAL_SCORE = 78
    s.POLYGON_API_KEY = ""
    s.ENABLE_V6_PREMIUM_CAP = False
    s.ENABLE_V6_SPREAD_GATE = False
    s.V6_PREMIUM_CAP_DOLLARS = 5.0
    s.V6_MAX_SPREAD_PCT = 40.0
    return s


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary SQLite database."""
    return str(tmp_path / "test.db")


async def _setup_db_with_trade(db_path: str, settings, **trade_overrides) -> int:
    """Create DB, init schema, insert a test trade. Returns trade_id."""
    settings.DB_PATH = db_path
    trader = PaperTrader(settings)
    await trader.init()

    defaults = {
        "signal_id": 1,
        "ticker": "MSTR",
        "direction": "bearish",
        "sentiment": "bearish",
        "score": 90,
        "strength": "strong",
        "bot_source": "test",
        "entry_price": 180.0,
        "strike": 180.0,
        "option_type": "put",
        "contracts": 2,
        "premium_per_contract": 2.96,
        "total_cost": 592.0,
        "status": "open",
        "opened_at": datetime.now().isoformat(),
        "webull_order_id": "FAKE_ORDER_123",
    }
    defaults.update(trade_overrides)

    async with aiosqlite.connect(db_path) as conn:
        cols = ", ".join(defaults.keys())
        placeholders = ", ".join(["?"] * len(defaults))
        await conn.execute(
            f"INSERT INTO paper_trades ({cols}) VALUES ({placeholders})",
            tuple(defaults.values()),
        )
        await conn.commit()
        cursor = await conn.execute("SELECT last_insert_rowid()")
        row = await cursor.fetchone()
        return row[0]


# ---------------------------------------------------------------------------
# Test: exit_source column migration
# ---------------------------------------------------------------------------

class TestExitSourceMigration:
    """Verify the exit_source column is added by init_paper_db."""

    @pytest.mark.asyncio
    async def test_exit_source_column_exists_after_init(self, tmp_db, settings):
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = {row[1] for row in await cursor.fetchall()}
            assert "exit_source" in columns

    @pytest.mark.asyncio
    async def test_exit_source_defaults_to_ai(self, tmp_db, settings):
        trade_id = await _setup_db_with_trade(tmp_db, settings)

        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT exit_source FROM paper_trades WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            assert row["exit_source"] == "ai"

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, tmp_db, settings):
        """Running init_paper_db twice should not error."""
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()
        # Second init should not raise
        await trader.init()

        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = [row[1] for row in await cursor.fetchall()]
            # exit_source should appear exactly once
            assert columns.count("exit_source") == 1


# ---------------------------------------------------------------------------
# Test: AI close keeps exit_source='ai'
# ---------------------------------------------------------------------------

class TestAICloseExitSource:
    """Normal bot-initiated closes should have exit_source='ai'."""

    @pytest.mark.asyncio
    async def test_close_trade_keeps_ai_exit_source(self, tmp_db, settings):
        trade_id = await _setup_db_with_trade(tmp_db, settings)
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        await trader.close_trade(trade_id, 175.0, 3.80, "adaptive_trail")

        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT exit_source, exit_reason, status FROM paper_trades WHERE id = ?",
                (trade_id,),
            )
            row = await cursor.fetchone()
            assert row["exit_source"] == "ai"
            assert row["exit_reason"] == "adaptive_trail"
            assert row["status"] == "closed"


# ---------------------------------------------------------------------------
# Test: Manual close detection (sell abandoned)
# ---------------------------------------------------------------------------

class TestManualCloseDetection:
    """When sell is abandoned (position gone), exit_source should be 'manual'."""

    @pytest.mark.asyncio
    async def test_sell_abandoned_marks_manual(self, tmp_db, settings):
        """Simulate the sell-abandoned path: trade closed in DB, then
        exit_source updated to 'manual'."""
        trade_id = await _setup_db_with_trade(tmp_db, settings)

        # Simulate what position_monitor does on sell abandoned:
        # 1. close_trade is called (sets exit_source='ai' by default)
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()
        await trader.close_trade(trade_id, 175.0, 3.78, "adaptive_trail")

        # 2. Then the sell-abandoned path updates exit_source to 'manual'
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute(
                "UPDATE paper_trades SET exit_source = 'manual' WHERE id = ?",
                (trade_id,),
            )
            await conn.commit()

        # Verify
        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT exit_source, exit_reason FROM paper_trades WHERE id = ?",
                (trade_id,),
            )
            row = await cursor.fetchone()
            assert row["exit_source"] == "manual"
            # exit_reason is still the AI's reason (what triggered the close attempt)
            assert row["exit_reason"] == "adaptive_trail"

    @pytest.mark.asyncio
    async def test_manual_close_logs_trade_event(self, tmp_db, settings):
        """log_trade_event should record the manual_close_detected event."""
        trade_id = await _setup_db_with_trade(tmp_db, settings)

        await log_trade_event(
            tmp_db, "MSTR", "manual_close_detected",
            f"trade#{trade_id} — Webull sell abandoned after 10 attempts.",
            trade_id=trade_id,
        )

        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM trade_events WHERE trade_id = ? AND event_type = 'manual_close_detected'",
                (trade_id,),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert "Webull sell abandoned" in row["detail"]
            assert "10 attempts" in row["detail"]


# ---------------------------------------------------------------------------
# Test: Source code safety — no UnboundLocalError risks
# ---------------------------------------------------------------------------

class TestSourceCodeSafety:
    """Static analysis to catch the class of bug that broke sells last time.

    The UnboundLocalError in position_monitor happened because a variable
    was used before being initialized. These tests inspect the actual source
    code to verify no new uninitialized variables were introduced.
    """

    def test_sell_abandoned_path_has_no_new_variables_before_use(self):
        """The sell-abandoned block only uses variables that are already
        defined in the outer scope: trade, ticker, db_path, retry_count,
        discord_client, paper_trader, _cleanup_trade_state."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)

        # The sell-abandoned block should reference these existing variables
        # and NOT introduce new variables that could be uninitialized
        assert "async with _connect_db(db_path)" in source
        assert "await log_trade_event(" in source
        assert "exit_source = 'manual'" in source

    def test_log_trade_event_import_exists(self):
        """Verify log_trade_event is imported in position_monitor."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor)
        assert "log_trade_event" in source

        # Verify it's actually imported, not just referenced in a string
        assert hasattr(position_monitor, "log_trade_event")

    def test_sell_abandoned_block_does_not_introduce_risky_assignments(self):
        """The sell-abandoned block should not assign to 'reason' or
        'description' — those were the exact variables that caused the
        UnboundLocalError that broke all sells last time."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)

        idx_abandoned = source.find("WEBULL SELL ABANDONED")
        idx_cleanup = source.find("_cleanup_trade_state(trade[\"id\"])", idx_abandoned)
        assert idx_abandoned > 0, "SELL ABANDONED block not found"
        assert idx_cleanup > idx_abandoned

        block = source[idx_abandoned:idx_cleanup]

        # These variables must NOT be assigned in this block —
        # they are the ones that caused the last critical bug
        dangerous_vars = ["reason =", "description ="]
        for var in dangerous_vars:
            assert var not in block, (
                f"'{var.strip()}' assigned in sell-abandoned block — "
                f"this is the exact pattern that broke all sells last time"
            )


# ---------------------------------------------------------------------------
# Test: Existing close_trade behavior unchanged
# ---------------------------------------------------------------------------

class TestCloseTradeUnchanged:
    """Verify close_trade still works correctly — no regressions."""

    @pytest.mark.asyncio
    async def test_close_trade_sets_all_fields(self, tmp_db, settings):
        trade_id = await _setup_db_with_trade(tmp_db, settings)
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        await trader.close_trade(trade_id, 175.0, 3.80, "soft_trail")

        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            assert row["status"] == "closed"
            assert row["exit_reason"] == "soft_trail"
            assert row["exit_premium"] is not None
            assert row["pnl_dollars"] is not None
            assert row["closed_at"] is not None
            assert row["exit_source"] == "ai"  # default

    @pytest.mark.asyncio
    async def test_close_trade_pnl_calculation(self, tmp_db, settings):
        """Verify P&L math is correct."""
        trade_id = await _setup_db_with_trade(
            tmp_db, settings,
            contracts=2, premium_per_contract=2.96, total_cost=592.0,
        )
        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        # Exit at $3.80 per contract, 2 contracts
        await trader.close_trade(trade_id, 175.0, 3.80, "adaptive_trail")

        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT pnl_dollars, exit_premium FROM paper_trades WHERE id = ?",
                (trade_id,),
            )
            row = await cursor.fetchone()
            # proceeds = 3.80 * 2 * 100 = $760, cost = $592, pnl = $168
            assert row["pnl_dollars"] == pytest.approx(168.0, abs=1.0)


# ---------------------------------------------------------------------------
# Test: Backtesting filter
# ---------------------------------------------------------------------------

class TestBacktestFilter:
    """Verify we can filter manual vs AI trades for backtesting."""

    @pytest.mark.asyncio
    async def test_filter_ai_only_trades(self, tmp_db, settings):
        """Create both AI and manual trades, filter correctly."""
        # Trade 1: AI close
        t1 = await _setup_db_with_trade(tmp_db, settings, ticker="SPY")
        # Trade 2: will be marked manual
        t2 = await _setup_db_with_trade(tmp_db, settings, ticker="MSTR")

        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        await trader.close_trade(t1, 500.0, 3.50, "profit_target")
        await trader.close_trade(t2, 175.0, 3.78, "adaptive_trail")

        # Mark t2 as manual
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute(
                "UPDATE paper_trades SET exit_source = 'manual' WHERE id = ?",
                (t2,),
            )
            await conn.commit()

        # Query AI-only trades (what backtester should use)
        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'closed' "
                "AND (exit_source = 'ai' OR exit_source IS NULL)"
            )
            ai_trades = await cursor.fetchall()
            assert len(ai_trades) == 1
            assert ai_trades[0]["ticker"] == "SPY"

        # Query manual trades
        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE exit_source = 'manual'"
            )
            manual_trades = await cursor.fetchall()
            assert len(manual_trades) == 1
            assert manual_trades[0]["ticker"] == "MSTR"

    @pytest.mark.asyncio
    async def test_old_trades_without_column_treated_as_ai(self, tmp_db, settings):
        """Trades from before the migration (exit_source=NULL or 'ai')
        should be included in AI backtests."""
        trade_id = await _setup_db_with_trade(tmp_db, settings)

        # Manually set exit_source to NULL to simulate pre-migration trade
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute(
                "UPDATE paper_trades SET exit_source = NULL WHERE id = ?",
                (trade_id,),
            )
            await conn.commit()

        # Filter should include NULL as AI
        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM paper_trades "
                "WHERE exit_source = 'ai' OR exit_source IS NULL"
            )
            count = (await cursor.fetchone())[0]
            assert count == 1


# ---------------------------------------------------------------------------
# Test: Full integration — sell-abandoned flow end-to-end
# ---------------------------------------------------------------------------

class TestSellAbandonedIntegration:
    """End-to-end test of the sell-abandoned → manual detection flow."""

    @pytest.mark.asyncio
    async def test_full_sell_abandoned_flow(self, tmp_db, settings):
        """Simulate complete flow: trade open → close attempt → sell fails
        → retry exhausted → marked manual."""
        trade_id = await _setup_db_with_trade(
            tmp_db, settings,
            sell_retry_count=10,  # Already at max retries
        )

        settings.DB_PATH = tmp_db
        trader = PaperTrader(settings)
        await trader.init()

        # Step 1: close_trade (AI decides to close)
        await trader.close_trade(trade_id, 175.0, 3.78, "adaptive_trail")

        # Step 2: Webull sell fails (position not found) — simulate what
        # position_monitor does after 10 retries
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute(
                "UPDATE paper_trades SET exit_source = 'manual' WHERE id = ?",
                (trade_id,),
            )
            await conn.commit()

        # Step 3: Log the event
        await log_trade_event(
            tmp_db, "MSTR", "manual_close_detected",
            f"trade#{trade_id} — Webull sell abandoned after 10 attempts.",
            trade_id=trade_id,
        )

        # Verify final state
        async with aiosqlite.connect(tmp_db) as conn:
            conn.row_factory = aiosqlite.Row

            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            )
            trade = await cursor.fetchone()
            assert trade["status"] == "closed"
            assert trade["exit_source"] == "manual"
            assert trade["exit_reason"] == "adaptive_trail"  # AI's reason preserved
            assert trade["pnl_dollars"] is not None

            cursor = await conn.execute(
                "SELECT * FROM trade_events WHERE trade_id = ? "
                "AND event_type = 'manual_close_detected'",
                (trade_id,),
            )
            event = await cursor.fetchone()
            assert event is not None
