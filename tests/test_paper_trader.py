"""Comprehensive tests for the paper trading engine."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from options_owl.collectors.candle_cache import CandleBar
from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, _select_trade_premium, get_open_trades, get_portfolio, init_paper_db
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
        "SIMULATED_ENTRY_SLIPPAGE_BPS": 0.0,
        "SIMULATED_EXIT_SLIPPAGE_BPS": 0.0,
        "ENABLE_DCA": False,
        "ENABLE_VINNY_STRATEGY": False,
        "ENABLE_SCORE_SIZING": False,
        "ENABLE_SMART_ENTRY": False,
        "ENABLE_PUT_TRADING": True,
        "ENABLE_DIRECTIONAL_REGIME": False,
        "CB_CLOSING_BUFFER_MINUTES": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _attach_mock_candle_cache(trader: PaperTrader) -> None:
    """Attach a mock candle cache with bearish data so PUT gates pass.

    Provides: 6 bearish bars (close < open, close < vwap), RSI < 45, ema9 < ema21.
    This satisfies PutMarketDirectionGate (SPY green) and PutBearishConfirmGate.
    """
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
# should_trade
# ---------------------------------------------------------------------------


class TestShouldTrade:
    def test_signal_below_min_score(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, MIN_SCORE=75)
        trader = PaperTrader(settings)
        sig = _make_signal(score=60)
        ok, reason = trader.should_trade(sig)
        assert ok is False
        assert "Score 60" in reason

    def test_signal_at_min_score(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, MIN_SCORE=75)
        trader = PaperTrader(settings)
        sig = _make_signal(score=75)
        ok, reason = trader.should_trade(sig)
        assert ok is True
        assert reason == "Signal meets criteria"

    def test_signal_above_min_score(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, MIN_SCORE=75)
        trader = PaperTrader(settings)
        sig = _make_signal(score=164)
        ok, reason = trader.should_trade(sig)
        assert ok is True

    def test_no_atm_premium(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        sig = _make_signal(atm_premium=None)
        ok, reason = trader.should_trade(sig)
        assert ok is False
        assert "ATM premium" in reason

    def test_zero_atm_premium(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        sig = _make_signal(atm_premium=0.0)
        ok, reason = trader.should_trade(sig)
        assert ok is False
        assert "ATM premium" in reason

    def test_no_stop_price(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        sig = _make_signal(stop_price=None)
        ok, reason = trader.should_trade(sig)
        assert ok is False
        assert "stop price" in reason.lower()


# ---------------------------------------------------------------------------
# evaluate_and_trade
# ---------------------------------------------------------------------------


class TestEvaluateAndTrade:
    @pytest.mark.asyncio
    async def test_opens_trade_for_qualifying_signal(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)

        assert result is not None
        assert result["ticker"] == "NVDA"
        assert result["option_type"] == "put"
        assert result["strike"] == 170.0
        assert result["contracts"] >= 1
        assert result["premium"] == 1.70
        assert result["total_cost"] > 0
        assert result["balance"] < settings.PORTFOLIO_SIZE

    @pytest.mark.asyncio
    async def test_skips_low_score_signal(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, MIN_SCORE=75)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=50)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_max_concurrent_positions(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, MAX_CONCURRENT=2)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Open two trades (hit the limit) — use tickers that allow PUTs
        sig1 = _make_signal(ticker="NVDA", score=130, atm_premium=1.70)
        sig2 = _make_signal(ticker="TSLA", score=110, atm_premium=0.93)
        r1 = await trader.evaluate_and_trade(sig1, signal_id=1)
        r2 = await trader.evaluate_and_trade(sig2, signal_id=2)
        assert r1 is not None
        assert r2 is not None

        # Third trade should be skipped (concurrent limit hit)
        sig3 = _make_signal(ticker="META", score=150, atm_premium=2.00)
        r3 = await trader.evaluate_and_trade(sig3, signal_id=3)
        assert r3 is None

    @pytest.mark.asyncio
    async def test_respects_daily_loss_limit(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=2000.0, DAILY_LOSS_LIMIT_PCT=10.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Open and close a trade with a big loss to trigger the daily loss limit
        # With $2k portfolio, 5% max position = $100, so 1 contract at $0.90 = $90
        # Loss of ~$90 won't trigger 10% ($200) limit alone, so use high premium
        sig1 = _make_signal(ticker="NVDA", score=130, atm_premium=2.50)
        r1 = await trader.evaluate_and_trade(sig1, signal_id=1)
        assert r1 is not None

        # Mark as Webull trade so daily_loss_limit gate counts it
        import aiosqlite
        async with aiosqlite.connect(tmp_db_path) as conn:
            await conn.execute(
                "UPDATE paper_trades SET webull_order_id = 'TEST_ORDER_1' WHERE id = ?",
                (r1["trade_id"],),
            )
            await conn.commit()

        # Close with a huge loss: entry premium 2.50, exit premium 0.01
        # PnL = (0.01 * 1 * 100) - 250 = -$249, exceeds 10% of $2k ($200)
        await trader.close_trade(
            trade_id=r1["trade_id"],
            exit_price=175.0,
            exit_premium=0.01,
            reason="stop_hit",
        )

        # Now another signal should be rejected due to daily loss limit
        sig2 = _make_signal(ticker="TSLA", score=150, atm_premium=0.93)
        r2 = await trader.evaluate_and_trade(sig2, signal_id=2)
        assert r2 is None

    @pytest.mark.asyncio
    async def test_no_duplicate_same_ticker(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig1 = _make_signal(ticker="NVDA", score=130, atm_premium=1.70)
        r1 = await trader.evaluate_and_trade(sig1, signal_id=1)
        assert r1 is not None

        # Same ticker again should be skipped
        sig2 = _make_signal(ticker="NVDA", score=150, atm_premium=2.00)
        r2 = await trader.evaluate_and_trade(sig2, signal_id=2)
        assert r2 is None

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, tmp_db_path):
        # Tiny portfolio that cannot afford even one contract
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10.0, MAX_POSITION_PCT=100.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # 1 contract costs 1.70 * 100 = $170, far exceeds $10 balance
        sig = _make_signal(score=130, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_call_direction_sets_option_type(self, tmp_db_path):
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(
            ticker="TSLA",
            direction=Direction.CALL,
            sentiment=Sentiment.BULLISH,
            score=130,
            atm_premium=0.93,
        )
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        assert result["option_type"] == "call"


# ---------------------------------------------------------------------------
# close_trade
# ---------------------------------------------------------------------------


class TestCloseTrade:
    @pytest.mark.asyncio
    async def test_close_winning_trade(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Close at higher premium (winner)
        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=2.50,
            reason="t1_hit",
        )
        assert result["pnl"] > 0
        assert result["pnl_pct"] > 0
        assert result["reason"] == "t1_hit"
        assert result["balance"] > opened["balance"]

    @pytest.mark.asyncio
    async def test_close_losing_trade(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        # Close at lower premium (loser)
        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=172.0,
            exit_premium=0.50,
            reason="stop_hit",
        )
        assert result["pnl"] < 0
        assert result["pnl_pct"] < 0
        assert result["reason"] == "stop_hit"

    @pytest.mark.asyncio
    async def test_balance_updates_correctly_after_close(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(score=130, atm_premium=1.70)
        opened = await trader.evaluate_and_trade(sig, signal_id=1)
        assert opened is not None

        contracts = opened["contracts"]
        entry_cost = opened["total_cost"]
        exit_premium = 2.50
        expected_proceeds = exit_premium * contracts * 100

        result = await trader.close_trade(
            trade_id=opened["trade_id"],
            exit_price=168.0,
            exit_premium=exit_premium,
            reason="t1_hit",
        )

        # After opening: balance = 10000 - entry_cost
        # After closing: balance = (10000 - entry_cost) + proceeds
        expected_balance = (10000.0 - entry_cost) + expected_proceeds
        assert abs(result["balance"] - expected_balance) < 0.01

    @pytest.mark.asyncio
    async def test_portfolio_win_loss_counters(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=50000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Open and close a winning trade
        sig1 = _make_signal(ticker="NVDA", score=130, atm_premium=1.70)
        r1 = await trader.evaluate_and_trade(sig1, signal_id=1)
        assert r1 is not None
        await trader.close_trade(r1["trade_id"], 168.0, 2.50, "t1_hit")

        # Open and close a losing trade
        sig2 = _make_signal(ticker="TSLA", score=110, atm_premium=0.93)
        r2 = await trader.evaluate_and_trade(sig2, signal_id=2)
        assert r2 is not None
        await trader.close_trade(r2["trade_id"], 175.0, 0.10, "stop_hit")

        portfolio = await get_portfolio(tmp_db_path, settings.PORTFOLIO_SIZE)
        assert portfolio["wins"] == 1
        assert portfolio["losses"] == 1


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    @pytest.fixture(autouse=True)
    def _pin_morning_time(self):
        """Pin _today_et to 10:30 AM ET so late-0DTE sizing caps don't interfere."""
        from unittest.mock import patch
        from datetime import datetime
        from zoneinfo import ZoneInfo
        morning = datetime(2026, 4, 28, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("options_owl.execution.paper_trader._today_et", return_value=morning):
            yield

    @pytest.mark.asyncio
    async def test_contracts_calculated_from_max_position(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0, MAX_POSITION_PCT=20.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Max position = 10000 * 20% = $2000
        # Cost per contract = 1.70 * 100 = $170
        # contracts = int(2000 / 170) = 11
        sig = _make_signal(score=130, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        assert result["contracts"] == 11
        assert result["total_cost"] == 11 * 170.0

    @pytest.mark.asyncio
    async def test_minimum_one_contract(self, tmp_db_path):
        # Very small max position that cannot afford 1 full contract at max_position level
        # but balance is enough for 1 contract
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=500.0, MAX_POSITION_PCT=1.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        # Max position = 500 * 1% = $5
        # Cost per contract = 1.70 * 100 = $170
        # int(5 / 170) = 0, but max(1, 0) = 1
        # total_cost = $170 which is < $500 balance -> ok
        sig = _make_signal(score=130, atm_premium=1.70)
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        assert result["contracts"] == 1


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_get_status_returns_formatted_string(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()

        status = await trader.get_status()
        assert "OPTIONS OWL" in status
        assert "PAPER PORTFOLIO" in status
        assert "$10000.00" in status

    @pytest.mark.asyncio
    async def test_get_status_shows_open_positions(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=10000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(ticker="NVDA", score=130, atm_premium=1.70)
        await trader.evaluate_and_trade(sig, signal_id=1)

        status = await trader.get_status()
        assert "NVDA" in status
        assert "Open Positions" in status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.asyncio
    async def test_get_open_trades(self, tmp_db_path):
        settings = _make_settings(tmp_db_path, PORTFOLIO_SIZE=50000.0)
        trader = PaperTrader(settings)
        await trader.init()
        _attach_mock_candle_cache(trader)

        sig = _make_signal(ticker="NVDA", score=130, atm_premium=1.70)
        await trader.evaluate_and_trade(sig, signal_id=1)

        trades = await get_open_trades(tmp_db_path)
        assert len(trades) == 1
        assert trades[0]["ticker"] == "NVDA"
        assert trades[0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_portfolio_creates_if_missing(self, tmp_db_path):
        await init_paper_db(tmp_db_path)
        portfolio = await get_portfolio(tmp_db_path, 5000.0)
        assert portfolio["starting_balance"] == 5000.0
        assert portfolio["current_balance"] == 5000.0


# ---------------------------------------------------------------------------
# _select_trade_premium
# ---------------------------------------------------------------------------


class TestSelectTradePremium:
    """Tests for the premium selection logic that picks the correct
    ATM/OTM premium based on strike matching and penny-option rejection."""

    def test_strike_matches_atm_uses_atm(self):
        """Signal strike == atm_strike with a reasonable ATM premium → keeps ATM."""
        sig = _make_signal(
            strike=170.0,
            atm_strike=170.0,
            atm_premium=1.70,
            otm_strike=167.5,
            otm_premium=0.46,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 1.70
        assert result.atm_strike == 170.0

    def test_strike_matches_otm_penny_atm_uses_otm(self):
        """Strike matches OTM strike, ATM is a penny option → switches to OTM."""
        sig = _make_signal(
            strike=260.0,
            entry_price=260.22,
            atm_strike=265.0,
            atm_premium=0.02,
            otm_strike=260.0,
            otm_premium=0.70,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 0.70
        assert result.atm_strike == 260.0

    def test_strike_matches_otm_otm_cheaper_uses_otm(self):
        """Strike matches OTM strike, OTM is cheaper than ATM → uses OTM."""
        sig = _make_signal(
            strike=167.5,
            atm_strike=170.0,
            atm_premium=1.70,
            otm_strike=167.5,
            otm_premium=0.46,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 0.46
        assert result.atm_strike == 167.5

    def test_strike_matches_otm_otm_more_expensive_keeps_atm(self):
        """Strike matches OTM strike but OTM is MORE expensive and ATM is not
        penny → does NOT swap (keeps ATM)."""
        sig = _make_signal(
            strike=167.5,
            entry_price=170.0,
            atm_strike=170.0,
            atm_premium=1.70,
            otm_strike=167.5,
            otm_premium=2.50,  # OTM more expensive — likely mislabeled ITM
        )
        result = _select_trade_premium(sig)
        # OTM is more expensive and ATM is not penny, so it should NOT swap
        assert result.atm_premium == 1.70
        assert result.atm_strike == 170.0

    def test_neither_strike_matches_picks_closest_to_entry(self):
        """Neither strike matches → uses the one closest to entry_price."""
        sig = _make_signal(
            strike=168.0,
            entry_price=168.5,
            atm_strike=170.0,
            atm_premium=1.70,
            otm_strike=167.5,
            otm_premium=0.46,
        )
        result = _select_trade_premium(sig)
        # otm_strike=167.5 is closer to entry=168.5 (dist=1.0) than atm=170.0 (dist=1.5)
        assert result.atm_premium == 0.46
        assert result.atm_strike == 167.5

    def test_penny_rejection_uses_only_good_premium(self):
        """ATM is $0.02 (penny), OTM is $0.70 → uses $0.70."""
        sig = _make_signal(
            strike=260.0,
            entry_price=260.0,
            atm_strike=265.0,
            atm_premium=0.02,
            otm_strike=260.0,
            otm_premium=0.70,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 0.70

    def test_both_penny_keeps_atm(self):
        """Both premiums < $0.05 → keeps ATM (can't do better)."""
        sig = _make_signal(
            strike=170.0,
            atm_strike=170.0,
            atm_premium=0.02,
            otm_strike=167.5,
            otm_premium=0.03,
        )
        result = _select_trade_premium(sig)
        # Both are pennies; function returns signal unchanged
        assert result.atm_premium == 0.02
        assert result.atm_strike == 170.0

    def test_no_otm_data_keeps_atm(self):
        """otm_premium is None → keeps ATM."""
        sig = _make_signal(
            strike=170.0,
            atm_strike=170.0,
            atm_premium=1.70,
            otm_strike=None,
            otm_premium=None,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 1.70
        assert result.atm_strike == 170.0

    def test_no_atm_data_promotes_otm(self):
        """atm_premium is None, strike matches OTM → promotes OTM premium to ATM."""
        sig = _make_signal(
            strike=167.5,
            atm_strike=None,
            atm_premium=None,
            otm_strike=167.5,
            otm_premium=0.46,
        )
        result = _select_trade_premium(sig)
        # No ATM data but strike matches OTM; OTM gets promoted to ATM fields
        assert result.atm_premium == 0.46
        assert result.atm_strike == 167.5

    def test_no_atm_no_otm_match_returns_unchanged(self):
        """atm_premium is None, OTM strike doesn't match → returns signal as-is."""
        sig = _make_signal(
            strike=170.0,
            atm_strike=None,
            atm_premium=None,
            otm_strike=167.5,
            otm_premium=0.46,
        )
        result = _select_trade_premium(sig)
        # Neither matches strike; function hits Case 4 — OTM >= 0.05, ATM < 0.05 (0)
        # So it switches to OTM premium
        assert result.atm_premium == 0.46
        assert result.atm_strike == 167.5

    def test_iwm_real_scenario(self):
        """Exact IWM bug case: entry=$260.22, atm_strike=$265, atm_prem=$0.02,
        otm_strike=$260, otm_prem=$0.70, strike=$260 → should use $0.70."""
        sig = _make_signal(
            ticker="IWM",
            direction=Direction.CALL,
            sentiment=Sentiment.BULLISH,
            strike=260.0,
            entry_price=260.22,
            target_price=262.31,
            expected_move_pct=0.8,
            atm_strike=265.0,
            atm_premium=0.02,
            otm_strike=260.0,
            otm_premium=0.70,
            target_1=260.75,
            target_2=261.27,
            stop_price=260.01,
        )
        result = _select_trade_premium(sig)
        assert result.atm_premium == 0.70
        assert result.atm_strike == 260.0
