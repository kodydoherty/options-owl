"""Tests for the circuit breaker module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import init_paper_db
from options_owl.risk.circuit_breaker import CircuitBreaker


def _make_settings(tmp_db_path: str, **overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": tmp_db_path,
        "PORTFOLIO_SIZE": 10000.0,
        "ENABLE_CIRCUIT_BREAKERS": True,
        "CB_MAX_CONSECUTIVE_LOSSES": 3,
        "CB_MAX_DRAWDOWN_PCT": 15.0,
        "CB_OPENING_BUFFER_MINUTES": 10,
        "CB_CLOSING_BUFFER_MINUTES": 15,
        "CB_INTRADAY_LOSS_HALT_PCT": 5.0,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


async def _insert_closed_trade(
    db_path: str,
    pnl: float,
    closed_at: str | None = None,
) -> None:
    if closed_at is None:
        # Use UTC to match circuit breaker's UTC-based date filtering
        from zoneinfo import ZoneInfo
        closed_at = datetime.now(tz=ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO paper_trades "
            "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
            "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
            "status, pnl_dollars, pnl_pct, opened_at, closed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', ?, ?, ?, ?)",
            (
                1, "TEST", "put", "bearish", 90, "strong", "Captain Hook",
                100.0, 100.0, "put", 1, 1.0, 100.0,
                pnl, pnl / 100.0 * 100.0,
                datetime.now().isoformat(), closed_at,
            ),
        )
        await conn.commit()


async def _insert_open_trade(db_path: str, total_cost: float = 100.0) -> int:
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "INSERT INTO paper_trades "
            "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
            "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
            "status, opened_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
            (
                1, "TEST", "put", "bearish", 90, "strong", "Captain Hook",
                100.0, 100.0, "put", 1, total_cost / 100.0, total_cost,
                datetime.now().isoformat(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def _set_portfolio_balance(db_path: str, current: float, starting: float = 10000.0) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO paper_portfolio (starting_balance, current_balance, created_at) "
            "VALUES (?, ?, ?)",
            (starting, current, datetime.now().isoformat()),
        )
        await conn.commit()


# ---------------------------------------------------------------------------
# Consecutive losses
# ---------------------------------------------------------------------------


class TestConsecutiveLosses:
    @pytest.mark.asyncio
    async def test_three_losses_blocks(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        for _ in range(3):
            await _insert_closed_trade(tmp_db_path, -50.0)

        blocked, reason = await CircuitBreaker.check_consecutive_losses(tmp_db_path, 3)
        assert blocked is True
        assert "Consecutive-loss" in reason

    @pytest.mark.asyncio
    async def test_two_losses_does_not_block(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        await _insert_closed_trade(tmp_db_path, -50.0)
        await _insert_closed_trade(tmp_db_path, -50.0)

        blocked, reason = await CircuitBreaker.check_consecutive_losses(tmp_db_path, 3)
        assert blocked is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_win_breaks_streak(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        # Oldest first (insert order matters — we ORDER BY closed_at DESC)
        t1 = datetime.now() - timedelta(minutes=3)
        t2 = datetime.now() - timedelta(minutes=2)
        t3 = datetime.now() - timedelta(minutes=1)

        await _insert_closed_trade(tmp_db_path, -50.0, closed_at=t1.isoformat())
        await _insert_closed_trade(tmp_db_path, 100.0, closed_at=t2.isoformat())  # win
        await _insert_closed_trade(tmp_db_path, -50.0, closed_at=t3.isoformat())

        blocked, reason = await CircuitBreaker.check_consecutive_losses(tmp_db_path, 3)
        assert blocked is False

    @pytest.mark.asyncio
    async def test_no_trades_does_not_block(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        blocked, reason = await CircuitBreaker.check_consecutive_losses(tmp_db_path, 3)
        assert blocked is False


# ---------------------------------------------------------------------------
# Drawdown from peak
# ---------------------------------------------------------------------------


class TestDrawdownFromPeak:
    @pytest.mark.asyncio
    async def test_blocks_when_drawdown_exceeded(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        # Portfolio started at 10000, now at 8000 -> 20% drawdown > 15% limit
        await _set_portfolio_balance(tmp_db_path, current=8000.0, starting=10000.0)

        blocked, reason = await CircuitBreaker.check_drawdown_from_peak(
            tmp_db_path, portfolio_size=10000.0, max_drawdown_pct=15.0,
        )
        assert blocked is True
        assert "Drawdown breaker" in reason

    @pytest.mark.asyncio
    async def test_allows_when_drawdown_within_limit(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        # Portfolio at 9500 -> 5% drawdown < 15% limit
        await _set_portfolio_balance(tmp_db_path, current=9500.0, starting=10000.0)

        blocked, reason = await CircuitBreaker.check_drawdown_from_peak(
            tmp_db_path, portfolio_size=10000.0, max_drawdown_pct=15.0,
        )
        assert blocked is False

    @pytest.mark.asyncio
    async def test_no_portfolio_does_not_block(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        blocked, reason = await CircuitBreaker.check_drawdown_from_peak(
            tmp_db_path, portfolio_size=10000.0, max_drawdown_pct=15.0,
        )
        assert blocked is False


# ---------------------------------------------------------------------------
# Opening buffer
# ---------------------------------------------------------------------------


class TestOpeningBuffer:
    def test_blocks_during_opening_buffer(self):
        """Mock time to 9:35 ET on a weekday — should block."""
        from options_owl.risk import circuit_breaker as cb_mod

        # Monday 9:35 ET
        fake_now = datetime(2026, 3, 30, 9, 35, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_opening_buffer(minutes=10)
        assert blocked is True
        assert "Opening buffer" in reason

    def test_allows_after_opening_buffer(self):
        from options_owl.risk import circuit_breaker as cb_mod

        # Monday 9:41 ET — past the 10-min buffer
        fake_now = datetime(2026, 3, 30, 9, 41, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_opening_buffer(minutes=10)
        assert blocked is False

    def test_allows_on_weekend(self):
        from options_owl.risk import circuit_breaker as cb_mod

        # Saturday 9:35 ET — weekend, no market
        fake_now = datetime(2026, 3, 28, 9, 35, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_opening_buffer(minutes=10)
        assert blocked is False


# ---------------------------------------------------------------------------
# Closing buffer
# ---------------------------------------------------------------------------


class TestClosingBuffer:
    def test_blocks_during_closing_buffer(self):
        from options_owl.risk import circuit_breaker as cb_mod

        # Monday 15:50 ET — within last 15 min
        fake_now = datetime(2026, 3, 30, 15, 50, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_closing_buffer(minutes=15)
        assert blocked is True
        assert "Closing buffer" in reason

    def test_allows_before_closing_buffer(self):
        from options_owl.risk import circuit_breaker as cb_mod

        # Monday 15:30 ET — still 30 min to close
        fake_now = datetime(2026, 3, 30, 15, 30, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_closing_buffer(minutes=15)
        assert blocked is False

    def test_allows_on_weekend(self):
        from options_owl.risk import circuit_breaker as cb_mod

        # Saturday 15:50 ET
        fake_now = datetime(2026, 3, 28, 15, 50, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            blocked, reason = CircuitBreaker.check_closing_buffer(minutes=15)
        assert blocked is False


# ---------------------------------------------------------------------------
# check_all combined
# ---------------------------------------------------------------------------


class TestCheckAll:
    @pytest.mark.asyncio
    async def test_all_pass_when_clean(self, tmp_db_path):
        """No losses, no drawdown, time outside buffers -> approved."""
        from options_owl.risk import circuit_breaker as cb_mod

        await init_paper_db(tmp_db_path)
        await _set_portfolio_balance(tmp_db_path, current=10000.0, starting=10000.0)

        settings = _make_settings(tmp_db_path)

        # Mock time to midday on a weekday
        fake_now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            approved, reasons = await CircuitBreaker.check_all(tmp_db_path, settings)

        assert approved is True
        assert reasons == []

    @pytest.mark.asyncio
    async def test_multiple_breakers_fire(self, tmp_db_path):
        """Consecutive losses + drawdown should both appear in reasons."""
        from options_owl.risk import circuit_breaker as cb_mod

        await init_paper_db(tmp_db_path)
        # 3 consecutive losses
        for _ in range(3):
            await _insert_closed_trade(tmp_db_path, -50.0)
        # Drawdown: 8000 of 10000 -> 20% > 15%
        await _set_portfolio_balance(tmp_db_path, current=8000.0, starting=10000.0)

        settings = _make_settings(tmp_db_path)

        # Midday weekday — no time buffers
        fake_now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=cb_mod.ET)
        with patch.object(cb_mod, "_now_et", return_value=fake_now):
            approved, reasons = await CircuitBreaker.check_all(tmp_db_path, settings)

        assert approved is False
        assert len(reasons) >= 2
        assert any("Consecutive-loss" in r for r in reasons)
        assert any("Drawdown" in r for r in reasons)


# ---------------------------------------------------------------------------
# Emergency close all
# ---------------------------------------------------------------------------


class TestEmergencyCloseAll:
    @pytest.mark.asyncio
    async def test_closes_all_open_positions(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        await _set_portfolio_balance(tmp_db_path, current=10000.0, starting=10000.0)
        await _insert_open_trade(tmp_db_path, total_cost=100.0)
        await _insert_open_trade(tmp_db_path, total_cost=200.0)

        # Create a mock paper_trader with the real close_trade behavior
        paper_trader = AsyncMock()
        paper_trader.db_path = tmp_db_path

        closed = await CircuitBreaker.emergency_close_all(paper_trader)
        assert closed == 2
        assert paper_trader.close_trade.call_count == 2

        # Verify each call used reason "emergency_circuit_breaker"
        for call in paper_trader.close_trade.call_args_list:
            assert call.kwargs["reason"] == "emergency_circuit_breaker"
            assert call.kwargs["exit_premium"] == 0.01

    @pytest.mark.asyncio
    async def test_no_open_positions(self, tmp_db_path):
        await init_paper_db(tmp_db_path)

        paper_trader = AsyncMock()
        paper_trader.db_path = tmp_db_path

        closed = await CircuitBreaker.emergency_close_all(paper_trader)
        assert closed == 0
        paper_trader.close_trade.assert_not_called()
