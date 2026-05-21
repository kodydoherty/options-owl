"""Tests for the dynamic Kelly Criterion position sizing module."""

from __future__ import annotations

import os
import tempfile

import aiosqlite
import pytest

from options_owl.risk.kelly import (
    compute_dynamic_position_pct,
    compute_kelly_fraction,
    get_kelly_summary,
)


# ---------------------------------------------------------------------------
# compute_kelly_fraction — pure math tests
# ---------------------------------------------------------------------------


class TestComputeKellyFraction:
    def test_coin_flip_fair(self):
        """50% win rate with equal payoffs => Kelly = 0."""
        result = compute_kelly_fraction(0.5, 10.0, 10.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_positive_edge(self):
        """60% win rate with equal payoffs => positive Kelly."""
        # f* = (p*b - q) / b = (0.6*1 - 0.4) / 1 = 0.2
        result = compute_kelly_fraction(0.6, 10.0, 10.0)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_negative_edge(self):
        """40% win rate with equal payoffs => negative Kelly."""
        result = compute_kelly_fraction(0.4, 10.0, 10.0)
        assert result == pytest.approx(-0.2, abs=1e-9)

    def test_asymmetric_payoff(self):
        """50% win rate but wins are 2x losses => positive edge."""
        # b = 20/10 = 2, f* = (0.5*2 - 0.5) / 2 = 0.25
        result = compute_kelly_fraction(0.5, 20.0, 10.0)
        assert result == pytest.approx(0.25, abs=1e-9)

    def test_zero_avg_loss(self):
        """Zero avg_loss returns 0 (avoids division by zero)."""
        assert compute_kelly_fraction(0.6, 10.0, 0.0) == 0.0

    def test_zero_avg_win(self):
        """Zero avg_win returns 0."""
        assert compute_kelly_fraction(0.6, 0.0, 10.0) == 0.0

    def test_high_win_rate(self):
        """90% win rate => large Kelly fraction."""
        # b = 1, f* = (0.9*1 - 0.1) / 1 = 0.8
        result = compute_kelly_fraction(0.9, 10.0, 10.0)
        assert result == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# Helpers for async tests with a temp DB
# ---------------------------------------------------------------------------


class FakeSettings:
    """Minimal settings object for testing."""

    ENABLE_KELLY_SIZING: bool = True
    KELLY_FRACTION: float = 0.25
    KELLY_MIN_PCT: float = 5.0
    KELLY_MAX_PCT: float = 25.0
    KELLY_MIN_TRADES: int = 20
    KELLY_DRAWDOWN_HALVE_PCT: float = 10.0
    MAX_POSITION_PCT: float = 5.0
    PORTFOLIO_SIZE: float = 10000.0


async def _create_test_db(db_path: str) -> None:
    """Create the paper_trades and paper_portfolio tables in a temp DB."""
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                ticker TEXT,
                direction TEXT,
                sentiment TEXT,
                score INTEGER,
                strength TEXT,
                bot_source TEXT NOT NULL,
                entry_price REAL,
                strike REAL,
                option_type TEXT,
                contracts INTEGER,
                premium_per_contract REAL,
                total_cost REAL,
                target_1 REAL,
                target_2 REAL,
                stop_price REAL,
                exit_by TEXT,
                expiry_date TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                exit_price REAL,
                exit_premium REAL,
                exit_reason TEXT,
                pnl_dollars REAL,
                pnl_pct REAL,
                opened_at TEXT,
                closed_at TEXT
            )"""
        )
        await conn.execute(
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
        await conn.commit()


async def _insert_trades(
    db_path: str,
    bot_source: str,
    wins: int,
    losses: int,
    avg_win_pnl: float = 30.0,
    avg_loss_pnl: float = -20.0,
) -> None:
    """Insert closed trades for a bot with specified win/loss counts."""
    async with aiosqlite.connect(db_path) as conn:
        for i in range(wins):
            await conn.execute(
                "INSERT INTO paper_trades "
                "(bot_source, status, pnl_pct, pnl_dollars, closed_at) "
                "VALUES (?, 'closed', ?, ?, datetime('now'))",
                (bot_source, avg_win_pnl, avg_win_pnl),
            )
        for i in range(losses):
            await conn.execute(
                "INSERT INTO paper_trades "
                "(bot_source, status, pnl_pct, pnl_dollars, closed_at) "
                "VALUES (?, 'closed', ?, ?, datetime('now'))",
                (bot_source, avg_loss_pnl, avg_loss_pnl),
            )
        await conn.commit()


async def _set_portfolio_balance(
    db_path: str, starting: float, current: float,
) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO paper_portfolio "
            "(starting_balance, current_balance, created_at) "
            "VALUES (?, ?, datetime('now'))",
            (starting, current),
        )
        await conn.commit()


# ---------------------------------------------------------------------------
# compute_dynamic_position_pct — async integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestDynamicPositionPct:
    @pytest.mark.asyncio
    async def test_fallback_not_enough_trades(self, tmp_db):
        """With fewer trades than KELLY_MIN_TRADES, returns MAX_POSITION_PCT."""
        await _create_test_db(tmp_db)
        await _insert_trades(tmp_db, "Captain Hook", wins=5, losses=3)
        settings = FakeSettings()
        result = await compute_dynamic_position_pct(tmp_db, "Captain Hook", settings)
        assert result == settings.MAX_POSITION_PCT

    @pytest.mark.asyncio
    async def test_positive_edge_clamped(self, tmp_db):
        """With enough trades and positive edge, returns a clamped Kelly pct."""
        await _create_test_db(tmp_db)
        # 15 wins, 5 losses = 75% win rate, avg_win=30%, avg_loss=20%
        await _insert_trades(tmp_db, "Captain Hook", wins=15, losses=5)
        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10
        result = await compute_dynamic_position_pct(tmp_db, "Captain Hook", settings)
        # Should be between min and max
        assert settings.KELLY_MIN_PCT <= result <= settings.KELLY_MAX_PCT

    @pytest.mark.asyncio
    async def test_clamp_at_minimum(self, tmp_db):
        """Negative edge should return KELLY_MIN_PCT."""
        await _create_test_db(tmp_db)
        # 5 wins, 15 losses = 25% win rate => negative edge
        await _insert_trades(tmp_db, "Captain Hook", wins=5, losses=15)
        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10
        result = await compute_dynamic_position_pct(tmp_db, "Captain Hook", settings)
        assert result == settings.KELLY_MIN_PCT

    @pytest.mark.asyncio
    async def test_clamp_at_maximum(self, tmp_db):
        """Very high edge with full Kelly should be clamped to KELLY_MAX_PCT."""
        await _create_test_db(tmp_db)
        # 19 wins, 1 loss = 95% win rate with big payoffs
        await _insert_trades(
            tmp_db, "Captain Hook", wins=19, losses=1,
            avg_win_pnl=50.0, avg_loss_pnl=-5.0,
        )
        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10
        settings.KELLY_FRACTION = 1.0  # full Kelly to push past max
        result = await compute_dynamic_position_pct(tmp_db, "Captain Hook", settings)
        assert result == settings.KELLY_MAX_PCT

    @pytest.mark.asyncio
    async def test_drawdown_halves_position(self, tmp_db):
        """When portfolio is down > KELLY_DRAWDOWN_HALVE_PCT, position is halved."""
        await _create_test_db(tmp_db)
        # 15W/5L = 75% win rate
        await _insert_trades(tmp_db, "Captain Hook", wins=15, losses=5)

        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10

        # First compute without drawdown (no portfolio row => defaults to PORTFOLIO_SIZE)
        result_no_drawdown = await compute_dynamic_position_pct(
            tmp_db, "Captain Hook", settings
        )

        # Now insert a portfolio that is down 15%
        await _set_portfolio_balance(tmp_db, 10000.0, 8500.0)

        result_with_drawdown = await compute_dynamic_position_pct(
            tmp_db, "Captain Hook", settings
        )

        # With drawdown, position should be halved (but clamped to min)
        expected = max(settings.KELLY_MIN_PCT, result_no_drawdown * 0.5)
        assert result_with_drawdown == pytest.approx(expected, abs=0.1)

    @pytest.mark.asyncio
    async def test_no_drawdown_reduction_within_threshold(self, tmp_db):
        """When drawdown is within threshold, no halving occurs."""
        await _create_test_db(tmp_db)
        await _insert_trades(tmp_db, "Captain Hook", wins=15, losses=5)
        # Only 5% drawdown (below 10% threshold)
        await _set_portfolio_balance(tmp_db, 10000.0, 9500.0)

        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10

        # Compute with 5% drawdown (should NOT halve)
        result = await compute_dynamic_position_pct(tmp_db, "Captain Hook", settings)

        # Same result as no portfolio entry (defaults to PORTFOLIO_SIZE)
        async with aiosqlite.connect(tmp_db) as conn:
            await conn.execute("DELETE FROM paper_portfolio")
            await conn.commit()
        result_default = await compute_dynamic_position_pct(
            tmp_db, "Captain Hook", settings
        )
        assert result == pytest.approx(result_default, abs=0.1)


class TestGetKellySummary:
    @pytest.mark.asyncio
    async def test_disabled(self, tmp_db):
        settings = FakeSettings()
        settings.ENABLE_KELLY_SIZING = False
        result = await get_kelly_summary(tmp_db, settings)
        assert "DISABLED" in result

    @pytest.mark.asyncio
    async def test_enabled_with_data(self, tmp_db):
        await _create_test_db(tmp_db)
        await _insert_trades(tmp_db, "Captain Hook", wins=15, losses=5)
        settings = FakeSettings()
        settings.KELLY_MIN_TRADES = 10
        result = await get_kelly_summary(tmp_db, settings)
        assert "Captain Hook" in result
        assert "Kelly Position Sizing" in result
