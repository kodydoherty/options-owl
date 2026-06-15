"""Tests for the unified risk manager."""

from __future__ import annotations

from datetime import datetime, timedelta

import aiosqlite
import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import init_paper_db
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)
from options_owl.risk.manager import RiskManager


def _make_settings(tmp_db_path: str, **overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": tmp_db_path,
        "PORTFOLIO_SIZE": 10000.0,
        "ENABLE_RISK_MANAGER": True,
        "MAX_PORTFOLIO_RISK_PCT": 20.0,
        "MAX_LOSS_PER_TRADE_PCT": 2.0,
        "WEEKLY_LOSS_LIMIT_PCT": 20.0,
        "ENABLE_IV_FILTER": False,
        "ENABLE_VIX_FILTER": False,
        "ENABLE_ANALYST_FILTER": False,
        "ENABLE_DIRECTIONAL_REGIME": False,
        "ENABLE_PUT_TRADING": True,
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
        bot_source=BotSource.CAPTAIN_HOOK,
    )
    defaults.update(overrides)
    return TradeSignal(**defaults)


async def _insert_open_trade(db_path: str, total_cost: float, ticker: str = "SPY") -> None:
    """Insert a fake open trade for testing portfolio limits."""
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO paper_trades "
            "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
            "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
            "status, opened_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
            (1, ticker, "put", "bearish", 90, "strong", "Captain Hook",
             100.0, 100.0, "put", 1, total_cost / 100.0, total_cost,
             datetime.now().isoformat()),
        )
        await conn.commit()


async def _insert_closed_trade(
    db_path: str, pnl: float, bot_source: str = "Captain Hook",
    closed_at: str | None = None,
) -> None:
    """Insert a fake closed trade for testing weekly loss limits."""
    if closed_at is None:
        # Use UTC to match risk manager's UTC-based date filtering
        from zoneinfo import ZoneInfo
        closed_at = datetime.now(tz=ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO paper_trades "
            "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
            "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
            "status, pnl_dollars, pnl_pct, opened_at, closed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', ?, ?, ?, ?)",
            (1, "TEST", "put", "bearish", 90, "strong", bot_source,
             100.0, 100.0, "put", 1, 1.0, 100.0,
             pnl, pnl / 100.0 * 100.0,
             datetime.now().isoformat(), closed_at),
        )
        await conn.commit()


# ---------------------------------------------------------------------------
# Disabled risk manager
# ---------------------------------------------------------------------------


class TestRiskManagerDisabled:
    @pytest.mark.asyncio
    async def test_disabled_risk_manager_approves_all(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, ENABLE_RISK_MANAGER=False)
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True
        assert reasons == []


# ---------------------------------------------------------------------------
# Portfolio risk limit (Check 1)
# ---------------------------------------------------------------------------


class TestPortfolioRiskLimit:
    @pytest.mark.asyncio
    async def test_blocks_when_portfolio_risk_exceeded(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_PORTFOLIO_RISK_PCT=10.0,
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Insert open trades totaling $900 (9% of 10k)
        await _insert_open_trade(tmp_db_path, 900.0)

        # New signal costs 1.70 * 100 = $170 -> total $1070 -> 10.7% > 10%
        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is False
        assert any("Portfolio risk" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_allows_when_under_portfolio_limit(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_PORTFOLIO_RISK_PCT=20.0,
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # No existing trades; new signal cost = 1.70 * 100 = $170 -> 1.7% < 20%
        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True
        assert reasons == []


# ---------------------------------------------------------------------------
# Per-trade risk limit (Check 2)
# ---------------------------------------------------------------------------


class TestPerTradeRiskLimit:
    @pytest.mark.asyncio
    async def test_blocks_expensive_single_trade(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_LOSS_PER_TRADE_PCT=1.0,  # Only $100 per trade
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Signal cost = 1.70 * 100 = $170 -> 1.7% > 1.0%
        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is False
        assert any("Trade risk" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_allows_cheap_single_trade(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_LOSS_PER_TRADE_PCT=5.0,  # $500 per trade
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Signal cost = 1.70 * 100 = $170 -> 1.7% < 5.0%
        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True


# ---------------------------------------------------------------------------
# Weekly loss limit (Check 3)
# ---------------------------------------------------------------------------


class TestWeeklyLossLimit:
    @pytest.mark.asyncio
    async def test_blocks_after_weekly_losses_hit_limit(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            WEEKLY_LOSS_LIMIT_PCT=5.0,  # $500 weekly loss limit
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Insert closed trades with losses totaling -$600 this week
        # Use ET time to match risk manager's ET-based week boundary calculation
        from zoneinfo import ZoneInfo
        now_iso = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
        await _insert_closed_trade(tmp_db_path, -300.0, closed_at=now_iso)
        await _insert_closed_trade(tmp_db_path, -300.0, closed_at=now_iso)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is False
        assert any("Weekly loss" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_allows_when_weekly_losses_under_limit(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            WEEKLY_LOSS_LIMIT_PCT=20.0,  # $2000 limit
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Only -$100 losses this week -> 1% < 20%
        await _insert_closed_trade(tmp_db_path, -100.0)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True

    @pytest.mark.asyncio
    async def test_old_losses_dont_count(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            WEEKLY_LOSS_LIMIT_PCT=5.0,
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Insert a loss from 2 weeks ago
        old_date = (datetime.now() - timedelta(days=14)).isoformat()
        await _insert_closed_trade(tmp_db_path, -5000.0, closed_at=old_date)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True


# ---------------------------------------------------------------------------
# Multiple checks interact
# ---------------------------------------------------------------------------


class TestMultipleChecks:
    @pytest.mark.asyncio
    async def test_multiple_failures_collected(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_PORTFOLIO_RISK_PCT=1.0,   # Very tight
            MAX_LOSS_PER_TRADE_PCT=0.5,   # Very tight
            WEEKLY_LOSS_LIMIT_PCT=1.0,    # Very tight
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        # Add weekly losses
        await _insert_closed_trade(tmp_db_path, -200.0)

        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is False
        # Should have at least portfolio risk + per-trade risk + weekly loss
        assert len(reasons) >= 3

    @pytest.mark.asyncio
    async def test_all_pass_when_limits_generous(self, tmp_db_path):
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=100000.0,
            MAX_PORTFOLIO_RISK_PCT=50.0,
            MAX_LOSS_PER_TRADE_PCT=50.0,
            WEEKLY_LOSS_LIMIT_PCT=50.0,
        )
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        signal = _make_signal(atm_premium=1.70)
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert approved is True
        assert reasons == []


# ---------------------------------------------------------------------------
# Position size multiplier
# ---------------------------------------------------------------------------


class TestPositionSizeMultiplier:
    def test_multiplier_is_1_when_vix_disabled(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, ENABLE_VIX_FILTER=False)
        rm = RiskManager(settings)
        assert rm.get_position_size_multiplier() == 1.0


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------


class TestRiskSummary:
    @pytest.mark.asyncio
    async def test_risk_summary_returns_string(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        summary = await rm.get_risk_summary(tmp_db_path)
        assert "Risk Summary" in summary
        assert "Open exposure" in summary
        assert "Weekly losses" in summary


# ---------------------------------------------------------------------------
# Settings flags enable/disable checks
# ---------------------------------------------------------------------------


class TestSettingsFlags:
    @pytest.mark.asyncio
    async def test_iv_filter_skipped_when_disabled(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, ENABLE_IV_FILTER=False)
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        # IV filter disabled -> no IV-related reasons
        assert not any("IV filter" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_vix_filter_skipped_when_disabled(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, ENABLE_VIX_FILTER=False)
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert not any("VIX regime" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_analyst_filter_skipped_when_disabled(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, ENABLE_ANALYST_FILTER=False)
        rm = RiskManager(settings)
        await init_paper_db(tmp_db_path)

        signal = _make_signal()
        approved, reasons = await rm.check_trade(signal, tmp_db_path)
        assert not any("Analyst filter" in r for r in reasons)
