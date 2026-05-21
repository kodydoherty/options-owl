"""Tests for slippage tracking in paper trading."""

from __future__ import annotations

import aiosqlite
import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, init_paper_db
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)


def _make_settings(tmp_db_path: str, **overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": tmp_db_path,
        "PORTFOLIO_SIZE": 10000.0,
        "MAX_POSITION_PCT": 5.0,
        "MAX_CONCURRENT": 3,
        "MIN_SCORE": 75,
        "DAILY_LOSS_LIMIT_PCT": 10.0,
        "ENABLE_RISK_MANAGER": False,
        "ENABLE_SMART_ENTRY": False,
        "ENABLE_VINNY_STRATEGY": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_signal(**overrides) -> TradeSignal:
    defaults = dict(
        ticker="NVDA",
        sentiment=Sentiment.BEARISH,
        direction=Direction.PUT,
        score=90,
        strength=SignalStrength.STRONG,
        entry_price=170.0,
        target_price=167.0,
        expected_move_pct=1.8,
        strike=170.0,
        expiry="0DTE",
        risk_reward=1.5,
        target_1=168.0,
        target_2=167.0,
        stop_price=171.0,
        atm_strike=170.0,
        atm_premium=1.70,
        otm_strike=167.5,
        otm_premium=0.46,
        bot_source=BotSource.CAPTAIN_HOOK,
        is_elite=True,
    )
    defaults.update(overrides)
    return TradeSignal(**defaults)


class TestEntrySlippage:
    @pytest.mark.asyncio
    async def test_entry_slippage_applied(self, tmp_db_path):
        """Entry premium should be higher than signal premium by the slippage amount."""
        settings = _make_settings(tmp_db_path, SIMULATED_ENTRY_SLIPPAGE_BPS=50.0)
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)

        assert result is not None
        # Signal premium is 1.70, slippage is 50 bps = 0.5%
        expected_premium = 1.70 * (1 + 50.0 / 10000)  # 1.70850
        assert abs(result["premium"] - expected_premium) < 0.0001
        assert result["signal_premium"] == 1.70
        assert abs(result["entry_slippage"] - (expected_premium - 1.70)) < 0.0001

    @pytest.mark.asyncio
    async def test_entry_slippage_stored_in_db(self, tmp_db_path):
        """Slippage columns should be populated in the DB."""
        settings = _make_settings(tmp_db_path, SIMULATED_ENTRY_SLIPPAGE_BPS=100.0)
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=2.00)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None

        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT signal_premium, entry_slippage, premium_per_contract "
                "FROM paper_trades WHERE id = ?",
                (result["trade_id"],),
            )
            row = dict(await cursor.fetchone())

        assert row["signal_premium"] == 2.00
        # 100 bps = 1%
        expected_premium = 2.00 * 1.01
        assert abs(row["premium_per_contract"] - expected_premium) < 0.0001
        assert abs(row["entry_slippage"] - (expected_premium - 2.00)) < 0.0001

    @pytest.mark.asyncio
    async def test_entry_slippage_increases_total_cost(self, tmp_db_path):
        """Total cost should reflect the slippage-adjusted premium."""
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_POSITION_PCT=20.0,
            SIMULATED_ENTRY_SLIPPAGE_BPS=50.0,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None

        expected_premium = 1.70 * (1 + 50.0 / 10000)
        expected_cost = result["contracts"] * expected_premium * 100
        assert abs(result["total_cost"] - expected_cost) < 0.01


class TestExitSlippage:
    @pytest.mark.asyncio
    async def test_exit_slippage_applied(self, tmp_db_path):
        """Exit premium stored should be reduced by the slippage amount."""
        settings = _make_settings(
            tmp_db_path,
            SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=50.0,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        nominal_exit = 2.50
        await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=nominal_exit,
            reason="t1_hit",
        )

        # Check exit slippage was stored in DB
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT exit_premium, exit_slippage FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            row = dict(await cursor.fetchone())

        expected_actual_exit = nominal_exit * (1 - 50.0 / 10000)
        expected_exit_slippage = nominal_exit - expected_actual_exit
        assert abs(row["exit_premium"] - expected_actual_exit) < 0.0001
        assert abs(row["exit_slippage"] - expected_exit_slippage) < 0.0001

    @pytest.mark.asyncio
    async def test_exit_slippage_reduces_pnl(self, tmp_db_path):
        """P&L should be lower with exit slippage than without."""
        # With slippage
        settings_slip = _make_settings(
            tmp_db_path,
            SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=100.0,  # 1%
        )
        trader_slip = PaperTrader(settings_slip)
        await trader_slip.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        opened = await trader_slip.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        result_slip = await trader_slip.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
        )

        # Without slippage (separate DB)
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db2 = os.path.join(tmpdir, "test2.db")
            settings_no = _make_settings(
                db2,
                SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
                SIMULATED_EXIT_SLIPPAGE_BPS=0.0,
            )
            trader_no = PaperTrader(settings_no)
            await trader_no.init()

            sig2 = _make_signal(score=90, atm_premium=1.70)
            opened2 = await trader_no.evaluate_and_trade(sig2, signal_id=1)
            assert opened2 is not None

            result_no = await trader_no.close_trade(
                trade_id=opened2["trade_id"],
                exit_price=168.0,
                exit_premium=2.50,
                reason="t1_hit",
            )

        # Slippage reduces PnL
        assert result_slip["pnl"] < result_no["pnl"]


class TestSlippageImpactOnPnL:
    @pytest.mark.asyncio
    async def test_combined_slippage_impact(self, tmp_db_path):
        """Both entry and exit slippage should compound to reduce P&L."""
        settings = _make_settings(
            tmp_db_path,
            SIMULATED_ENTRY_SLIPPAGE_BPS=50.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=50.0,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
        )

        contracts = opened["contracts"]
        # Entry: pay more (1.70 * 1.005)
        entry_premium = 1.70 * (1 + 50.0 / 10000)
        entry_cost = entry_premium * contracts * 100
        # Exit: get less (2.50 * 0.995)
        actual_exit = 2.50 * (1 - 50.0 / 10000)
        proceeds = actual_exit * contracts * 100
        expected_pnl = proceeds - entry_cost

        assert abs(result["pnl"] - expected_pnl) < 0.01


class TestZeroSlippage:
    @pytest.mark.asyncio
    async def test_zero_entry_slippage(self, tmp_db_path):
        """With zero slippage settings, premium should match signal premium exactly."""
        settings = _make_settings(
            tmp_db_path,
            SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=0.0,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        assert result["premium"] == 1.70
        assert result["signal_premium"] == 1.70
        assert result["entry_slippage"] == 0.0

    @pytest.mark.asyncio
    async def test_zero_exit_slippage(self, tmp_db_path):
        """With zero exit slippage, exit premium should be unchanged."""
        settings = _make_settings(
            tmp_db_path,
            SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=0.0,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = _make_signal(score=90, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
        )

        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT exit_premium, exit_slippage FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            row = dict(await cursor.fetchone())

        assert row["exit_premium"] == 2.50
        assert row["exit_slippage"] == 0.0


class TestSlippageColumnsExist:
    @pytest.mark.asyncio
    async def test_columns_exist_in_new_db(self, tmp_db_path):
        """Slippage columns should exist after init_paper_db."""
        await init_paper_db(tmp_db_path)

        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = {row[1] for row in await cursor.fetchall()}

        assert "signal_premium" in columns
        assert "entry_slippage" in columns
        assert "exit_slippage" in columns

    @pytest.mark.asyncio
    async def test_columns_added_via_migration(self, tmp_db_path):
        """Slippage columns should be added to an existing DB via migration."""
        import os

        os.makedirs(os.path.dirname(tmp_db_path) or ".", exist_ok=True)

        # Create a DB with the old schema (no slippage columns)
        async with aiosqlite.connect(tmp_db_path) as conn:
            await conn.execute("""
                CREATE TABLE paper_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    starting_balance REAL NOT NULL,
                    current_balance REAL NOT NULL,
                    total_trades INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    daily_pnl REAL NOT NULL DEFAULT 0,
                    last_trade_date TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE paper_trades (
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
                    target_1 REAL,
                    target_2 REAL,
                    stop_price REAL,
                    exit_by TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    exit_price REAL,
                    exit_premium REAL,
                    exit_reason TEXT,
                    pnl_dollars REAL,
                    pnl_pct REAL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT
                )
            """)
            await conn.commit()

        # Run init_paper_db which should add the columns via migration
        await init_paper_db(tmp_db_path)

        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = {row[1] for row in await cursor.fetchall()}

        assert "signal_premium" in columns
        assert "entry_slippage" in columns
        assert "exit_slippage" in columns
