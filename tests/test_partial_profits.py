"""Tests for partial profit-taking (Feature 10)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, get_open_trades, init_paper_db
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)

# Pin time to morning so late-0DTE sizing caps don't interfere with contract counts
_MORNING_ET = datetime(2026, 4, 28, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture(autouse=True)
def _pin_morning_time():
    with patch("options_owl.execution.paper_trader._today_et", return_value=_MORNING_ET):
        yield


def _make_settings(tmp_db_path: str, **overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": tmp_db_path,
        "PORTFOLIO_SIZE": 10000.0,
        "MAX_POSITION_PCT": 20.0,
        "MAX_CONCURRENT": 5,
        "MIN_SCORE": 75,
        "DAILY_LOSS_LIMIT_PCT": 10.0,
        "ENABLE_RISK_MANAGER": False,
        "ENABLE_PARTIAL_PROFITS": True,
        "PARTIAL_CLOSE_PCT": 50.0,
        "SIMULATED_ENTRY_SLIPPAGE_BPS": 0.0,
        "SIMULATED_EXIT_SLIPPAGE_BPS": 0.0,
        "ENABLE_DCA": False,
        "ENABLE_VINNY_STRATEGY": False,
        "ENABLE_SCORE_SIZING": False,
        "ENABLE_SMART_ENTRY": False,
        "ENABLE_PUT_TRADING": True,
        "ENABLE_DIRECTIONAL_REGIME": False,
        "ENTRY_HARD_CUTOFF_HOUR": 23,
        "ENTRY_HARD_CUTOFF_MINUTE": 59,
        "TOD_LATE_CUTOFF_HOUR": 23,
        "TOD_LATE_CUTOFF_MINUTE": 59,
        "ENABLE_MORNING_CUTOFF": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _attach_mock_candle_cache(trader: PaperTrader) -> None:
    """Attach a mock candle cache with bearish data so PUT gates pass."""
    from options_owl.collectors.candle_cache import CandleBar
    bearish_bars = [
        CandleBar(0, 522.0, 522.5, 519.0, 519.5, 1000, vwap=521.0),
        CandleBar(0, 520.0, 520.5, 518.0, 518.5, 1000, vwap=520.0),
        CandleBar(0, 519.0, 519.5, 517.0, 517.5, 1000, vwap=519.0),
        CandleBar(0, 518.0, 518.5, 516.0, 516.5, 1000, vwap=518.0),
        CandleBar(0, 517.0, 517.5, 515.0, 515.5, 1000, vwap=517.0),
        CandleBar(0, 516.0, 516.5, 514.0, 514.5, 1000, vwap=516.0),
    ]
    mock_cc = AsyncMock()
    mock_cc.get_candle_data = AsyncMock(return_value={
        "5m": bearish_bars,
        "indicators": {"5m": {"rsi": 38.0, "ema9": 515.0, "ema21": 518.0}},
    })
    trader._candle_cache = mock_cc


def _make_signal(**overrides) -> TradeSignal:
    defaults = dict(
        ticker="NVDA",
        sentiment=Sentiment.BEARISH,
        direction=Direction.PUT,
        score=130,
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


# ---------------------------------------------------------------------------
# partial_close_trade
# ---------------------------------------------------------------------------


class TestPartialCloseTradeSpitsCorrectly:
    @pytest.mark.asyncio
    async def test_partial_close_splits_trade(self, tmp_db_path):
        """Closing 50% of a 4-contract trade should create a 2-contract closed child
        and leave 2 contracts open on the original."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Open a trade — with 20% of $10k = $2000, premium $1.70 -> 11 contracts
        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        total_contracts = opened["contracts"]
        assert total_contracts > 1  # need > 1 to split

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        expected_closed = round(total_contracts * 50.0 / 100)
        expected_remaining = total_contracts - expected_closed

        assert result["contracts_closed"] == expected_closed
        assert result["contracts_remaining"] == expected_remaining

        # Verify the original trade row is still open with reduced contracts
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == opened["trade_id"]
        assert open_trades[0]["contracts"] == expected_remaining
        assert open_trades[0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_partial_close_creates_child_row(self, tmp_db_path):
        """The child row should be closed, linked to parent, and have correct PnL."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        child_id = result["child_trade_id"]

        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE id = ?", (child_id,))
            child = dict(await cursor.fetchone())

        assert child["status"] == "closed"
        assert child["parent_trade_id"] == opened["trade_id"]
        assert child["exit_reason"] == "t1_hit"
        assert child["contracts"] == result["contracts_closed"]

        # PnL check: (2.50 - 1.70) * contracts_closed * 100
        expected_pnl = (2.50 - 1.70) * result["contracts_closed"] * 100
        assert abs(child["pnl_dollars"] - expected_pnl) < 0.01


class TestPartialCloseFallsBackToFullClose:
    """Regression: when close_pct rounds to 0 contracts, partial_close_trade
    should fall back to close_trade and the caller must detect this to avoid
    orphaning contracts on Webull (bug found 2026-04-27)."""

    @pytest.mark.asyncio
    async def test_20pct_of_2_contracts_falls_back_to_full_close(self, tmp_db_path):
        """20% of 2 contracts rounds to 0 → should fall back to full close."""
        settings = _make_settings(
            tmp_db_path, PORTFOLIO_SIZE=5000.0, ENABLE_SCORE_SIZING=False,
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Open with 2 contracts
        sig = _make_signal(score=130, atm_premium=5.00)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Force to exactly 2 contracts
        async with aiosqlite.connect(tmp_db_path) as conn:
            await conn.execute(
                "UPDATE paper_trades SET contracts = 2, total_cost = 1000.0 WHERE id = ?",
                (opened["trade_id"],),
            )
            await conn.commit()

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=6.00,
            reason="t2_hit",
            close_pct=20.0,  # round(2 * 0.2) = round(0.4) = 0 → full close
        )

        # Should have fallen back to full close (no "contracts_closed" key)
        assert "contracts_closed" not in result, (
            "20% of 2 contracts should fall back to close_trade, "
            "not partial_close_trade"
        )

        # Trade should be fully closed
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0

    @pytest.mark.asyncio
    async def test_caller_detects_full_close_fallback(self, tmp_db_path):
        """The result from partial_close_trade tells the caller whether it was
        a real partial (has contracts_closed) or full close (no contracts_closed).
        This is critical for selling the right amount on Webull."""
        settings = _make_settings(
            tmp_db_path, PORTFOLIO_SIZE=5000.0, ENABLE_SCORE_SIZING=False,
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=5.00)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # 4 contracts, 50% close = real partial
        async with aiosqlite.connect(tmp_db_path) as conn:
            await conn.execute(
                "UPDATE paper_trades SET contracts = 4, total_cost = 2000.0 WHERE id = ?",
                (opened["trade_id"],),
            )
            await conn.commit()

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=6.00,
            reason="t1_hit",
            close_pct=50.0,
        )

        # Real partial — has contracts_closed
        assert "contracts_closed" in result
        assert result["contracts_closed"] == 2
        assert result["contracts_remaining"] == 2


class TestRemainingContractsStillTracked:
    @pytest.mark.asyncio
    async def test_remaining_open_after_partial(self, tmp_db_path):
        """After a partial close, the remaining position should still show up as open."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        total_contracts = opened["contracts"]

        await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 1
        remaining = open_trades[0]
        expected_remaining = total_contracts - round(total_contracts * 50.0 / 100)
        assert remaining["contracts"] == expected_remaining
        # total_cost should be proportionally reduced
        expected_cost = expected_remaining * 1.70 * 100
        assert abs(remaining["total_cost"] - expected_cost) < 0.01


class TestT2ClosesRemainder:
    @pytest.mark.asyncio
    async def test_full_close_after_partial(self, tmp_db_path):
        """After partial close at T1, a full close at T2 should close the remainder."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        total_contracts = opened["contracts"]
        expected_remaining = total_contracts - round(total_contracts * 50.0 / 100)

        # Partial close at T1
        await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        # Full close at T2
        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=167.0,
            exit_premium=3.00,
            reason="t2_hit",
        )

        assert result["reason"] == "t2_hit"
        # PnL should be based on remaining contracts and their adjusted cost
        expected_proceeds = 3.00 * expected_remaining * 100
        expected_cost = expected_remaining * 1.70 * 100
        expected_pnl = expected_proceeds - expected_cost
        assert abs(result["pnl"] - expected_pnl) < 0.01

        # No open trades left
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0


class TestStopCloses100Percent:
    @pytest.mark.asyncio
    async def test_stop_closes_full_position(self, tmp_db_path):
        """Stop-loss should close the entire position, not partial."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Direct full close (simulating stop_hit which always does full close)
        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=172.0,
            exit_premium=0.50,
            reason="stop_hit",
        )

        assert result["reason"] == "stop_hit"
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0


class TestDisabledFeatureDoesNotSplit:
    @pytest.mark.asyncio
    async def test_disabled_partial_profits(self, tmp_db_path):
        """When ENABLE_PARTIAL_PROFITS=False, partial_close_trade with 100%
        should just do a full close (this tests the PaperTrader method itself;
        the monitor logic is what actually checks the feature flag)."""
        settings = _make_settings(tmp_db_path, ENABLE_PARTIAL_PROFITS=False)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Even calling partial_close_trade with 100% should do a full close
        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=100.0,
        )

        # Full close: no child_trade_id, and no remaining contracts
        assert "child_trade_id" not in result  # regular close_trade return
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0


class TestEdgeCaseSingleContract:
    @pytest.mark.asyncio
    async def test_single_contract_cannot_split(self, tmp_db_path):
        """With only 1 contract, partial close should fall back to full close."""
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=500.0,
            MAX_POSITION_PCT=5.0,  # $25 max -> 1 contract at $1.70 * 100 = $170
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # 1 contract: 500 * 5% = $25 -> int(25/170) = 0, max(1, 0) = 1
        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        assert opened["contracts"] == 1

        # 50% of 1 = round(0.5) = 0 -> contracts_to_close <= 0 -> full close
        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        # Should have done a full close (no child_trade_id in result)
        assert "child_trade_id" not in result
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0


class TestBalanceUpdatesOnPartialClose:
    @pytest.mark.asyncio
    async def test_balance_reflects_partial_proceeds(self, tmp_db_path):
        """Balance should increase by the proceeds of the closed portion only."""
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        balance_after_open = opened["balance"]
        total_contracts = opened["contracts"]
        contracts_to_close = round(total_contracts * 50.0 / 100)

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        expected_proceeds = 2.50 * contracts_to_close * 100
        expected_balance = balance_after_open + expected_proceeds
        assert abs(result["balance"] - expected_balance) < 0.01


class TestDbMigration:
    @pytest.mark.asyncio
    async def test_parent_trade_id_column_exists(self, tmp_db_path):
        """The parent_trade_id column should exist after init_paper_db."""
        await init_paper_db(tmp_db_path)
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = [row[1] for row in await cursor.fetchall()]
        assert "parent_trade_id" in columns

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, tmp_db_path):
        """Running init_paper_db twice should not fail."""
        await init_paper_db(tmp_db_path)
        await init_paper_db(tmp_db_path)  # should not raise


# ---------------------------------------------------------------------------
# Vinny + DCA interaction
# ---------------------------------------------------------------------------


class TestVinnyDCAInteraction:
    @pytest.mark.asyncio
    async def test_vinny_dca_disabled(self, tmp_db_path):
        """ENABLE_VINNY_STRATEGY=True + ENABLE_DCA=True → DCA should be skipped.

        Vinny's strategy always places all contracts at once (no DCA for 0DTE).
        A score >= 130 should yield 3 contracts (score-tier cap).
        """
        settings = _make_settings(
            tmp_db_path,
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_SCORE_SIZING=True,
            ENABLE_DCA=True,
            DCA_TRANCHES=3,
            DCA_FIRST_PCT=40.0,
            PORTFOLIO_SIZE=10000.0,
            MAX_POSITION_PCT=20.0,
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        # Mock _get_current_price to return a price near entry (avoids anti-chase rejection)
        with patch.object(trader, "_get_current_price", return_value=170.05):
            opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Flat 85% × PUT 50%: slot=$1600, 85%×50%=$680, raw=4, pos_cap=20%→11 → 4
        assert opened["contracts"] == 4

        # Verify DCA is skipped — dca_tranches_remaining should be 0
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT dca_tranches_remaining, dca_total_contracts FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            row = dict(await cursor.fetchone())
        assert row["dca_tranches_remaining"] == 0
        assert row["dca_total_contracts"] == 4

    @pytest.mark.asyncio
    async def test_non_vinny_dca_still_works(self, tmp_db_path):
        """ENABLE_VINNY_STRATEGY=False + ENABLE_DCA=True → DCA should split normally.

        With default sizing (20% of $10k = $2000) and premium $1.70 ($170/contract),
        total_contracts = 11. DCA first tranche = max(1, int(11 * 40/100)) = 4 contracts.
        """
        settings = _make_settings(
            tmp_db_path,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_SCORE_SIZING=False,
            ENABLE_DCA=True,
            DCA_TRANCHES=3,
            DCA_FIRST_PCT=40.0,
            PORTFOLIO_SIZE=10000.0,
            MAX_POSITION_PCT=20.0,
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Default sizing: 10000 * 20% = $2000, $2000 / $170 = 11 (no hard cap)
        # DCA first tranche: max(1, int(11 * 40/100)) = max(1, 4) = 4
        assert opened["contracts"] == 4

        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT dca_tranches_remaining, dca_total_contracts FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            row = dict(await cursor.fetchone())
        # 3 tranches - 1 already placed = 2 remaining
        assert row["dca_tranches_remaining"] == 2
        assert row["dca_total_contracts"] == 11


# ---------------------------------------------------------------------------
# Scale-out rounding edge cases
# ---------------------------------------------------------------------------


class TestScaleOutRounding:
    @pytest.mark.asyncio
    async def test_scale_out_20pct_on_5_contracts(self, tmp_db_path):
        """5 contracts, close 20% → round(5*0.20)=1 contract closed, 4 remaining."""
        # Flat 85% × PUT 50%: $12600 × 80% / 5 = $2016, 85%×50% = $856 / $170 = 5 contracts
        sig = _make_signal(score=150, atm_premium=1.70)
        settings_vinny = _make_settings(
            tmp_db_path,
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_SCORE_SIZING=True,
            PORTFOLIO_SIZE=12600.0,
        )
        trader_v = PaperTrader(settings_vinny)
        await trader_v.init()
        _attach_mock_candle_cache(trader_v)
        with patch.object(trader_v, "_get_current_price", return_value=170.05):
            opened = await trader_v.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        assert opened["contracts"] == 5

        result = await trader_v.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=20.0,
        )

        # round(5 * 0.20) = round(1.0) = 1 contract closed
        assert result["contracts_closed"] == 1
        assert result["contracts_remaining"] == 4

    @pytest.mark.asyncio
    async def test_scale_out_20pct_on_2_contracts(self, tmp_db_path):
        """2 contracts, close 20% → round(2*0.20)=0 → falls back to full close."""
        _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=1000.0,
            MAX_POSITION_PCT=5.0,  # $50 max -> floor(50/170)=0 -> max(1,0)=1
        )
        # We need exactly 2 contracts. Use default sizing with tight budget.
        settings2 = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_POSITION_PCT=4.0,  # $400 -> 400/170 = 2 contracts
        )
        trader = PaperTrader(settings2)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        assert opened["contracts"] == 2

        # round(2 * 0.20) = round(0.4) = 0 → falls back to full close
        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=20.0,
        )

        # Full close — no child_trade_id
        assert "child_trade_id" not in result
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0

    @pytest.mark.asyncio
    async def test_scale_out_25pct_on_4_contracts(self, tmp_db_path):
        """4 contracts, close 25% → round(4*0.25)=1 contract closed, 3 remaining."""
        settings = _make_settings(
            tmp_db_path,
            PORTFOLIO_SIZE=10000.0,
            MAX_POSITION_PCT=7.0,  # $700 -> 700/170 = 4 contracts
        )
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None
        assert opened["contracts"] == 4

        result = await trader.partial_close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=25.0,
        )

        # round(4 * 0.25) = round(1.0) = 1 contract closed
        assert result["contracts_closed"] == 1
        assert result["contracts_remaining"] == 3
