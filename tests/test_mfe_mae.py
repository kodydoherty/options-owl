"""Tests for MFE/MAE (Max Favorable / Max Adverse Excursion) tracking."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from options_owl.execution.paper_trader import init_paper_db
from options_owl.execution.position_monitor import _update_mfe_mae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_open_trade(db_path: str, entry_premium: float = 2.00) -> int:
    """Insert a minimal open trade and return its id."""
    await init_paper_db(db_path)
    async with aiosqlite.connect(db_path) as conn:
        # Ensure portfolio exists
        await conn.execute(
            "INSERT INTO paper_portfolio (starting_balance, current_balance, created_at) "
            "VALUES (10000, 10000, '2025-01-01T00:00:00')"
        )
        cursor = await conn.execute(
            "INSERT INTO paper_trades "
            "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
            "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
            "status, opened_at) "
            "VALUES (1, 'NVDA', 'put', 'bearish', 90, 'strong', 'captain_hook', "
            "170.0, 170.0, 'put', 2, ?, ?, 'open', '2025-01-01T09:30:00')",
            (entry_premium, entry_premium * 2 * 100),
        )
        trade_id = cursor.lastrowid
        await conn.commit()
    return trade_id


async def _get_trade(db_path: str, trade_id: int) -> dict:
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
        return dict(await cursor.fetchone())


# ---------------------------------------------------------------------------
# _update_mfe_mae
# ---------------------------------------------------------------------------


class TestUpdateMfeMae:
    @pytest.mark.asyncio
    async def test_mfe_updates_when_premium_increases(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        # Premium goes up to 3.00
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=3.00, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mfe_premium"] == 3.00
        assert trade["mfe_pnl_pct"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_mae_updates_when_premium_decreases(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        # Premium drops to 1.00
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=1.00, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mae_premium"] == 1.00
        assert trade["mae_pnl_pct"] == pytest.approx(-50.0)

    @pytest.mark.asyncio
    async def test_mfe_does_not_regress(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        # MFE goes to 4.00, then premium drops to 2.50
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=4.00, entry_premium=2.00)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=2.50, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mfe_premium"] == 4.00  # should NOT have regressed

    @pytest.mark.asyncio
    async def test_mae_does_not_regress(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        # MAE goes to 0.50, then premium recovers to 1.80
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=0.50, entry_premium=2.00)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=1.80, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mae_premium"] == 0.50  # should NOT have regressed upward

    @pytest.mark.asyncio
    async def test_initial_values_default_to_entry_premium(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        # First call with a premium equal to entry — should set both MFE and MAE
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=2.00, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        # No change from entry, so values stay None (no DB write needed when equal)
        assert trade["mfe_premium"] is None
        assert trade["mae_premium"] is None

        # First call with a premium above entry
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=2.10, entry_premium=2.00)
        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mfe_premium"] == 2.10
        # MAE defaults to entry_premium (2.00), and 2.10 > 2.00 so MAE stays at entry
        assert trade["mae_premium"] is None or trade["mae_premium"] == 2.00

    @pytest.mark.asyncio
    async def test_pnl_pct_calculation(self, tmp_db_path):
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=4.00)

        # Premium goes to 6.00 (MFE = +50%) and later drops to 2.00 (MAE = -50%)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=6.00, entry_premium=4.00)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=2.00, entry_premium=4.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mfe_pnl_pct"] == pytest.approx(50.0)
        assert trade["mae_pnl_pct"] == pytest.approx(-50.0)

    @pytest.mark.asyncio
    async def test_simultaneous_mfe_and_mae_tracking(self, tmp_db_path):
        """Simulate a sequence: entry at 2.00, up to 3.00, down to 1.00, settle at 1.50."""
        trade_id = await _insert_open_trade(tmp_db_path, entry_premium=2.00)

        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=3.00, entry_premium=2.00)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=1.00, entry_premium=2.00)
        await _update_mfe_mae(tmp_db_path, trade_id, exit_premium=1.50, entry_premium=2.00)

        trade = await _get_trade(tmp_db_path, trade_id)
        assert trade["mfe_premium"] == 3.00
        assert trade["mae_premium"] == 1.00
        assert trade["mfe_pnl_pct"] == pytest.approx(50.0)
        assert trade["mae_pnl_pct"] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# DB migration
# ---------------------------------------------------------------------------


class TestDbMigration:
    @pytest.mark.asyncio
    async def test_migration_adds_mfe_mae_columns(self, tmp_db_path):
        await init_paper_db(tmp_db_path)

        # Verify the columns exist by querying them
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute(
                "SELECT mfe_premium, mae_premium, mfe_pnl_pct, mae_pnl_pct "
                "FROM paper_trades LIMIT 0"
            )
            col_names = [desc[0] for desc in cursor.description]

        assert "mfe_premium" in col_names
        assert "mae_premium" in col_names
        assert "mfe_pnl_pct" in col_names
        assert "mae_pnl_pct" in col_names

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, tmp_db_path):
        """Running init_paper_db twice should not raise errors."""
        await init_paper_db(tmp_db_path)
        await init_paper_db(tmp_db_path)  # should not raise


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------


class TestReportExcursionSection:
    def test_report_includes_excursion_section_when_data_exists(self, tmp_db_path):
        """Verify the report renders MFE/MAE analysis for closed trades."""
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

        # Set up DB with closed trades that have MFE/MAE data
        conn = sqlite3.connect(tmp_db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                starting_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                daily_pnl REAL NOT NULL DEFAULT 0,
                last_trade_date TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                score INTEGER NOT NULL,
                strength TEXT NOT NULL,
                bot_source TEXT NOT NULL,
                entry_price REAL NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                contracts INTEGER NOT NULL,
                premium_per_contract REAL NOT NULL,
                total_cost REAL NOT NULL,
                target_1 REAL, target_2 REAL, stop_price REAL,
                exit_by TEXT, expiry_date TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                exit_price REAL, exit_premium REAL, exit_reason TEXT,
                pnl_dollars REAL, pnl_pct REAL,
                opened_at TEXT NOT NULL, closed_at TEXT,
                parent_trade_id INTEGER,
                signal_premium REAL, entry_slippage REAL, exit_slippage REAL,
                mfe_premium REAL, mae_premium REAL, mfe_pnl_pct REAL, mae_pnl_pct REAL
            )"""
        )
        conn.execute(
            "INSERT INTO paper_portfolio VALUES (1, 10000, 10200, 2, 1, 1, 0, '2025-01-01', '2025-01-01')"
        )
        # Insert a winning closed trade with MFE/MAE
        conn.execute(
            "INSERT INTO paper_trades VALUES "
            "(1, 1, 'NVDA', 'put', 'bearish', 90, 'strong', 'captain_hook', "
            "170.0, 170.0, 'put', 2, 2.00, 400.0, "
            "168.0, 167.0, 171.0, NULL, '2025-01-01', "
            "'closed', 168.0, 3.00, 't1_hit', 200.0, 50.0, "
            "'2025-01-01T09:30:00', '2025-01-01T11:00:00', "
            "NULL, NULL, NULL, NULL, "
            "3.50, 1.50, 75.0, -25.0)"
        )
        # Insert a losing closed trade with MFE/MAE
        conn.execute(
            "INSERT INTO paper_trades VALUES "
            "(2, 2, 'TSLA', 'call', 'bullish', 85, 'strong', 'captain_hook', "
            "250.0, 250.0, 'call', 1, 3.00, 300.0, "
            "255.0, 260.0, 245.0, NULL, '2025-01-01', "
            "'closed', 248.0, 2.00, 'stop_hit', -100.0, -33.3, "
            "'2025-01-01T09:30:00', '2025-01-01T12:00:00', "
            "NULL, NULL, NULL, NULL, "
            "3.20, 1.80, 6.7, -40.0)"
        )
        conn.commit()
        conn.close()

        # Run the report using the test DB
        import scripts.paper_report as pr

        original_db = pr.DB_PATH
        pr.DB_PATH = tmp_db_path
        try:
            report = pr.run_report()
        finally:
            pr.DB_PATH = original_db

        assert "EXCURSION ANALYSIS" in report
        assert "Avg MFE" in report
        assert "Avg MAE" in report
        assert "Left on Table" in report
        assert "Heat Taken" in report
        assert "Worst MAE" in report

    def test_report_no_excursion_section_without_data(self, tmp_db_path):
        """Report should not crash when no MFE/MAE data exists."""
        conn = sqlite3.connect(tmp_db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                starting_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                daily_pnl REAL NOT NULL DEFAULT 0,
                last_trade_date TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                score INTEGER NOT NULL,
                strength TEXT NOT NULL,
                bot_source TEXT NOT NULL,
                entry_price REAL NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                contracts INTEGER NOT NULL,
                premium_per_contract REAL NOT NULL,
                total_cost REAL NOT NULL,
                target_1 REAL, target_2 REAL, stop_price REAL,
                exit_by TEXT, expiry_date TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                exit_price REAL, exit_premium REAL, exit_reason TEXT,
                pnl_dollars REAL, pnl_pct REAL,
                opened_at TEXT NOT NULL, closed_at TEXT,
                parent_trade_id INTEGER,
                signal_premium REAL, entry_slippage REAL, exit_slippage REAL,
                mfe_premium REAL, mae_premium REAL, mfe_pnl_pct REAL, mae_pnl_pct REAL
            )"""
        )
        conn.execute(
            "INSERT INTO paper_portfolio VALUES (1, 10000, 10000, 0, 0, 0, 0, NULL, '2025-01-01')"
        )
        conn.commit()
        conn.close()

        import scripts.paper_report as pr

        original_db = pr.DB_PATH
        pr.DB_PATH = tmp_db_path
        try:
            report = pr.run_report()
        finally:
            pr.DB_PATH = original_db

        # Should not contain excursion section when no data
        assert "EXCURSION ANALYSIS" not in report
