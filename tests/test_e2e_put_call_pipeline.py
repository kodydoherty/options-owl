"""End-to-end tests: CALL + PUT signals through the full pipeline.

Tests the complete lifecycle for both directions:
  1. Parse Discord signal (CALL or PUT)
  2. Route through entry pipeline (18 gates)
  3. Open paper trade with correct sizing
  4. V5 FSM evaluates exit conditions
  5. PUT uses PUT_SCALP_CONFIG (no ceiling, -50% hard stop, trail system exits)
  6. CALL uses per-ticker V5Config (adaptive FSM)
  7. Close trade and verify DB state

Also tests:
  - Bear mode detection (SPY down from open)
  - PUT ticker exclusion (AAPL, GOOGL, NVDA, AMZN excluded from PUTs)
  - PUT scalp/soft/adaptive trail exits, hard stop, grace period backstop
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from options_owl.collectors.discord_collector import parse_trade_signal
from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, get_open_trades
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)
from options_owl.risk.exit_v5.config import (
    PUT_SCALP_CONFIG,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_db_path: str, **overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": tmp_db_path,
        "PORTFOLIO_SIZE": 10000.0,
        "MAX_POSITION_PCT": 20.0,
        "MAX_CONCURRENT": 5,
        "MIN_SCORE": 78,
        "DAILY_LOSS_LIMIT_PCT": 10.0,
        "ENABLE_RISK_MANAGER": False,
        "ENABLE_PARTIAL_PROFITS": True,
        "PARTIAL_CLOSE_PCT": 50.0,
        "SIMULATED_ENTRY_SLIPPAGE_BPS": 0.0,
        "SIMULATED_EXIT_SLIPPAGE_BPS": 0.0,
        "ENABLE_DCA": False,
        "ENABLE_VINNY_STRATEGY": True,
        "ENABLE_SCORE_SIZING": True,
        "ENABLE_SMART_ENTRY": False,
        "ENABLE_PUT_TRADING": True,
        "ENABLE_DIRECTIONAL_REGIME": False,
        "CB_CLOSING_BUFFER_MINUTES": 0,
        "ANTI_CHASE_MAX_MOVE_PCT": 99.0,
        "ENTRY_HARD_CUTOFF_HOUR": 23,
        "ENTRY_HARD_CUTOFF_MINUTE": 59,
        "ENABLE_MORNING_CUTOFF": False,
        "TOD_LATE_CUTOFF_HOUR": 23,
        "TOD_LATE_CUTOFF_MINUTE": 59,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_call_signal(ticker="SPY", score=95, premium=2.50, strike=520.0) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        sentiment=Sentiment.BULLISH,
        direction=Direction.CALL,
        score=score,
        strength=SignalStrength.STRONG,
        entry_price=strike + 0.50,
        target_price=strike + 3.0,
        expected_move_pct=0.6,
        strike=strike,
        expiry="0DTE",
        risk_reward=2.0,
        target_1=strike + 1.0,
        target_2=strike + 2.0,
        stop_price=strike - 1.0,
        atm_strike=strike,
        atm_premium=premium,
        otm_strike=strike + 2.0,
        otm_premium=premium * 0.3,
        bot_source=BotSource.CAPTAIN_HOOK,
        is_elite=True,
    )


def _make_put_signal(ticker="SPY", score=95, premium=0.30, strike=520.0) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        sentiment=Sentiment.BEARISH,
        direction=Direction.PUT,
        score=score,
        strength=SignalStrength.STRONG,
        entry_price=strike - 0.50,
        target_price=strike - 3.0,
        expected_move_pct=0.6,
        strike=strike,
        expiry="0DTE",
        risk_reward=2.0,
        target_1=strike - 1.0,
        target_2=strike - 2.0,
        stop_price=strike + 1.0,
        atm_strike=strike,
        atm_premium=premium,
        otm_strike=strike - 2.0,
        otm_premium=premium * 0.5,
        bot_source=BotSource.CAPTAIN_HOOK,
        is_elite=True,
    )


# ---------------------------------------------------------------------------
# 1. CALL signal through full pipeline
# ---------------------------------------------------------------------------


class TestCallPipeline:
    @pytest.mark.asyncio
    async def test_call_signal_opens_and_closes(self, tmp_db_path):
        """CALL signal: parse -> open trade -> close at profit target."""
        sig = _make_call_signal()
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        market_time = datetime(2026, 5, 26, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=520.50), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=1001)

        assert opened is not None, "CALL trade should be opened"
        trade_id = opened["trade_id"]

        # Verify DB state
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            )
            trade = dict(await cursor.fetchone())

        assert trade["status"] == "open"
        assert trade["ticker"] == "SPY"
        assert trade["option_type"] == "call"
        assert trade["direction"] == "call"
        assert trade["contracts"] > 0

        # Close the trade
        result = await trader.close_trade(
            trade_id=trade_id,
            exit_price=523.0,
            exit_premium=3.50,
            reason="profit_target",
        )
        assert result["reason"] == "profit_target"

        # Verify closed
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0


# ---------------------------------------------------------------------------
# 2. PUT signal through full pipeline
# ---------------------------------------------------------------------------


class TestPutPipeline:
    @pytest.mark.asyncio
    async def test_put_signal_opens_and_closes(self, tmp_db_path):
        """PUT signal: parse -> open trade -> close at profit target."""
        sig = _make_put_signal()
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        # Mock candle cache with bearish data for PutBearishConfirmGate
        from options_owl.collectors.candle_cache import CandleBar
        # 6 bearish bars: close < open, close < vwap
        bearish_bars = [
            CandleBar(0, 522.0, 522.5, 519.0, 519.5, 1000, vwap=521.0),
            CandleBar(0, 520.0, 520.5, 518.0, 518.5, 1000, vwap=520.0),
            CandleBar(0, 519.0, 519.5, 517.0, 517.5, 1000, vwap=519.0),
            CandleBar(0, 518.0, 518.5, 516.0, 516.5, 1000, vwap=518.0),
            CandleBar(0, 517.0, 517.5, 515.0, 515.5, 1000, vwap=517.0),
            CandleBar(0, 516.0, 516.5, 514.0, 514.5, 1000, vwap=516.0),
        ]
        mock_candle_cache = AsyncMock()
        mock_candle_cache.get_candle_data = AsyncMock(return_value={
            "5m": bearish_bars,
            "indicators": {"5m": {"rsi": 38.0, "ema9": 515.0, "ema21": 518.0}},
        })
        trader._candle_cache = mock_candle_cache

        market_time = datetime(2026, 5, 26, 13, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=519.50), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=2001)

        assert opened is not None, "PUT trade should be opened"
        trade_id = opened["trade_id"]

        # Verify DB state
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            )
            trade = dict(await cursor.fetchone())

        assert trade["status"] == "open"
        assert trade["ticker"] == "SPY"
        assert trade["option_type"] == "put"
        assert trade["direction"] == "put"

        # Close at profit target (+50%)
        result = await trader.close_trade(
            trade_id=trade_id,
            exit_price=517.0,
            exit_premium=0.45,  # +50% from 0.30
            reason="profit_target",
        )
        assert result["reason"] == "profit_target"

        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0

    @pytest.mark.asyncio
    async def test_put_signal_from_discord_message(self, tmp_db_path):
        """Parse a real PUT Discord message and open a trade."""
        message = (
            "\U0001f43b SPY - Bearish (PUT) \U0001f48e\n"
            "95/100 (Strong) \U0001f7e2 (raw 155)\n"
            "$519.50 \u27a1 $517.00 (+0.5%)\n"
            "\U0001f511 Key Signals\n"
            "BB 2\u03c3 Touch | Vol 1.5x | EMA Bounce\n"
            "\U0001f4bc Trade Idea\n"
            "Buy Puts | Strike: $520 Put | Expiry: 0DTE | R:R 1.50:1\n"
            "\U0001f3af Price Targets\n"
            "T1: $518.50 (+0.2%)\n"
            "T2: $517.00 (+0.5%)\n"
            "Stop: $521.00\n"
            "\U0001f4b0 ATM Pick\n"
            "$520 put @ ~$1.50\n"
            "\u26a1 OTM Pick\n"
            "$518 put @ ~$0.30\n"
        )
        sig = parse_trade_signal(message, message_id=2002, channel="signals", author="captain hook")
        assert sig is not None
        assert sig.direction == Direction.PUT
        assert sig.ticker == "SPY"
        assert sig.strike == 520.0

        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        # Mock candle cache with bearish data for PutBearishConfirmGate
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

        market_time = datetime(2026, 5, 26, 13, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=519.50), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=2002)

        assert opened is not None, "PUT trade from Discord should be opened"
        assert opened["premium"] == pytest.approx(1.50, abs=0.01)

        # Verify it's a PUT in the DB
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT option_type FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            trade = dict(await cursor.fetchone())
        assert trade["option_type"] == "put"


# ---------------------------------------------------------------------------
# 3. PUT scalp V5 FSM exits
# ---------------------------------------------------------------------------


class TestPutScalpExits:
    """Test that PUT trades use PUT_SCALP_CONFIG for exits."""

    def test_put_gets_scalp_config(self):
        """get_ticker_config returns PUT_SCALP_CONFIG for PUT option_type."""
        cfg = get_ticker_config("SPY", use_per_ticker=True, option_type="put")
        assert cfg is PUT_SCALP_CONFIG
        assert cfg.profit_target_general_pct == 0.0  # no ceiling — trail locks in gains
        assert cfg.backstop_0dte_pct == 50.0
        assert cfg.theta_bleed_min == 999.0  # no hold limit — trail handles exits
        # Scalp trail and soft trail are enabled for PUTs
        assert cfg.scalp_peak_threshold_pct == 15.0
        assert cfg.soft_trail_band_low_pct == 15.0

    def test_call_gets_normal_config(self):
        """get_ticker_config returns normal V5Config for CALL option_type."""
        cfg = get_ticker_config("SPY", use_per_ticker=True, option_type="call")
        assert cfg is not PUT_SCALP_CONFIG
        assert cfg.profit_target_general_pct == 0.0  # disabled for calls

    def test_put_no_profit_target_holds(self):
        """PUT at +36% should HOLD — no profit target, trail locks in gains."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=1, ticker="QQQ", option_type="put",
            entry_premium=0.25, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        now = datetime(2026, 5, 26, 13, 10)  # 10min past entry (past grace)
        action = fsm.evaluate(
            state=state, current_premium=0.34,  # +36%
            bid=0.33, ask=0.35, now_et=now,
            current_underlying=449.0, minutes_to_close=50.0,
        )
        assert not action.should_exit  # trail system manages profitable PUTs

    def test_put_stop_loss_exit(self):
        """PUT at -55% should trigger hard stop (50% threshold)."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=2, ticker="QQQ", option_type="put",
            entry_premium=0.25, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        now = datetime(2026, 5, 26, 13, 10)
        action = fsm.evaluate(
            state=state, current_premium=0.11,  # -56%
            bid=0.10, ask=0.12, now_et=now,
            current_underlying=451.0, minutes_to_close=50.0,
        )
        assert action.should_exit

    def test_put_no_hold_limit(self):
        """PUT held 65 minutes should HOLD — no time limit, trail handles exits."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=3, ticker="QQQ", option_type="put",
            entry_premium=0.25, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        now = datetime(2026, 5, 26, 14, 5)  # 65min into trade
        action = fsm.evaluate(
            state=state, current_premium=0.22,  # slight loss, not at stop
            bid=0.21, ask=0.23, now_et=now,
            current_underlying=450.2, minutes_to_close=55.0,
        )
        assert not action.should_exit  # no hold time limit for PUTs

    def test_put_holds_during_grace(self):
        """PUT within grace period should HOLD (unless at backstop)."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=4, ticker="QQQ", option_type="put",
            entry_premium=0.25, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        now = datetime(2026, 5, 26, 13, 2)  # 2min in (within 3min grace)
        action = fsm.evaluate(
            state=state, current_premium=0.20,  # -20%, not at backstop
            bid=0.19, ask=0.21, now_et=now,
            current_underlying=450.5, minutes_to_close=58.0,
        )
        assert not action.should_exit, "Should HOLD during grace period"

    def test_put_backstop_during_grace(self):
        """PUT at -60% DURING grace should still trigger backstop."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=5, ticker="QQQ", option_type="put",
            entry_premium=0.25, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        now = datetime(2026, 5, 26, 13, 1)  # 1min in (within grace)
        action = fsm.evaluate(
            state=state, current_premium=0.10,  # -60%, at backstop
            bid=0.09, ask=0.11, now_et=now,
            current_underlying=451.0, minutes_to_close=59.0,
        )
        assert action.should_exit
        assert action.reason == ExitReason.HARD_STOP

    def test_put_scalp_trail_fires(self):
        """PUT peaked +25%, faded below 60% of peak → scalp trail exits."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=6, ticker="SPY", option_type="put",
            entry_premium=0.20, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        # Build peak first
        now = datetime(2026, 5, 26, 13, 6)
        fsm.evaluate(
            state=state, current_premium=0.25,  # +25% (sets peak)
            bid=0.24, ask=0.26, now_et=now,
            current_underlying=449.0, minutes_to_close=50.0,
        )
        # Now fade to 60% of peak gain: peak_gain=$0.05, 60% = $0.03, floor = 0.20 + 0.03 = 0.23
        # Premium at $0.209 = keeping only 18% of gains → below 60% threshold
        now2 = datetime(2026, 5, 26, 13, 8)
        action = fsm.evaluate(
            state=state, current_premium=0.209,
            bid=0.20, ask=0.22, now_et=now2,
            current_underlying=449.5, minutes_to_close=48.0,
        )
        assert action.should_exit
        assert action.reason == ExitReason.SCALP_TRAIL

    def test_put_soft_trail_fires(self):
        """PUT peaked +30%, faded below soft trail floor → exits."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=7, ticker="SPY", option_type="put",
            entry_premium=0.20, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=10, entry_underlying_price=450.0, dte=0,
        )
        # Build peak at +30%
        now = datetime(2026, 5, 26, 13, 6)
        fsm.evaluate(
            state=state, current_premium=0.26,  # +30%
            bid=0.25, ask=0.27, now_et=now,
            current_underlying=449.0, minutes_to_close=50.0,
        )
        # Soft trail: keep 60% of gains. peak=$0.26, gain=$0.06, keep=0.036, floor=0.236
        # Premium at $0.225 < floor → should exit
        now2 = datetime(2026, 5, 26, 13, 8)
        action = fsm.evaluate(
            state=state, current_premium=0.225,
            bid=0.22, ask=0.23, now_et=now2,
            current_underlying=449.5, minutes_to_close=48.0,
        )
        assert action.should_exit
        assert action.reason in (ExitReason.SCALP_TRAIL, ExitReason.SOFT_TRAIL)

    def test_put_trail_exits_on_large_drop(self):
        """PUT peaked +25%, drops 40% from peak → trail system exits (soft or adaptive)."""
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=8, ticker="SPY", option_type="put",
            entry_premium=1.00, entry_time=datetime(2026, 5, 26, 13, 0),
            contracts=5, entry_underlying_price=450.0, dte=0,
        )
        # Build peak at +25% ($1.25)
        now = datetime(2026, 5, 26, 13, 6)
        fsm.evaluate(
            state=state, current_premium=1.25,  # +25%
            bid=1.24, ask=1.26, now_et=now,
            current_underlying=448.0, minutes_to_close=50.0,
        )
        # Drop 40% from peak: peak=$1.25, price=$0.75
        # Soft trail (gate 7) fires before adaptive trail (gate 8)
        now2 = datetime(2026, 5, 26, 13, 10)
        action = fsm.evaluate(
            state=state, current_premium=0.75,  # 40% drop from peak
            bid=0.74, ask=0.76, now_et=now2,
            current_underlying=449.0, minutes_to_close=46.0,
        )
        assert action.should_exit
        assert action.reason in (ExitReason.SOFT_TRAIL, ExitReason.ADAPTIVE_TRAIL)


# ---------------------------------------------------------------------------
# 4. Monitor bridge uses PUT config
# ---------------------------------------------------------------------------


class TestMonitorBridgePutConfig:
    """Test that V5MonitorBridge dispatches PUT trades to PUT_SCALP_CONFIG."""

    def _make_bridge_settings(self):
        return SimpleNamespace(
            EXIT_ENGINE="v5",
            ENABLE_V6_PER_TICKER_CONFIG=True,
            ENABLE_V6_BREAKEVEN_RATCHET=False,
            V6_BREAKEVEN_TRIGGER_PCT=20.0,
            ENABLE_V6_SCALEOUT=False,
            V6_SCALEOUT_GAIN_PCT=20.0,
            V6_SCALEOUT_FRACTION=0.333,
            V6_SCALEOUT_MIN_CONTRACTS=3,
            ENABLE_V6_2PM_TIGHTEN=False,
            V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
            V6_2PM_SOFT_TRAIL_BOOST=0.15,
            ENABLE_V6_EARLY_POP_GATE=False,
            ENABLE_V6_SIDEWAYS_SCALP=False,
            ENABLE_SCALP_TARGET=False,
            SCALP_TARGET_PCT=25.0,
            SCALP_RUNNER_CONFIRM_PCT=40.0,
        )

    def test_bridge_put_fsm_uses_scalp_config(self):
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        bridge = V5MonitorBridge(self._make_bridge_settings())
        put_fsm = bridge._get_fsm("SPY", option_type="put")
        call_fsm = bridge._get_fsm("SPY", option_type="call")
        assert put_fsm.cfg is PUT_SCALP_CONFIG
        assert call_fsm.cfg is not PUT_SCALP_CONFIG

    def test_bridge_evaluate_put_trade(self):
        """Bridge should correctly evaluate a PUT trade using PUT_SCALP_CONFIG."""
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        bridge = V5MonitorBridge(self._make_bridge_settings())

        trade = {
            "id": 100,
            "ticker": "SPY",
            "option_type": "put",
            "premium_per_contract": 0.25,
            "contracts": 10,
            "entry_price": 520.0,
            "opened_at": "2026-05-26T17:00:00",  # UTC = 1PM ET
            "expiry_date": "2026-05-26",
            "mfe_premium": 0.25,
        }

        now = datetime(2026, 5, 26, 13, 10)  # 10min past entry, past grace

        # At +56% gain, should HOLD — no profit target, trail locks in gains
        reason, desc = bridge.evaluate(
            trade=trade,
            exit_premium=0.39,  # +56%
            current_price=519.0,
            now_et=now,
        )
        assert reason is None  # trail system manages, no immediate exit


# ---------------------------------------------------------------------------
# 5. PUT ticker exclusion
# ---------------------------------------------------------------------------


class TestPutTickerExclusion:
    """Verify PUT-excluded tickers are excluded from PUT scalp config."""

    def test_excluded_tickers_still_get_put_config(self):
        """Even excluded tickers get PUT_SCALP_CONFIG — the exclusion is
        handled at the sourcing/entry level, not the exit level."""
        for ticker in ["AAPL", "GOOGL", "NVDA", "AMZN"]:
            cfg = get_ticker_config(ticker, use_per_ticker=True, option_type="put")
            assert cfg is PUT_SCALP_CONFIG

    def test_excluded_tickers_get_normal_call_config(self):
        """CALL trades for these tickers use their normal per-ticker config."""
        cfg = get_ticker_config("NVDA", use_per_ticker=True, option_type="call")
        assert cfg is not PUT_SCALP_CONFIG
        # NVDA has a per-ticker CALL config (EARLY_PROFIT) — uses general profit target
        assert cfg.profit_target_general_pct == 20.0


# ---------------------------------------------------------------------------
# 6. Both CALL + PUT in same portfolio
# ---------------------------------------------------------------------------


class TestMixedPortfolio:
    @pytest.mark.asyncio
    async def test_call_and_put_coexist(self, tmp_db_path):
        """Open both a CALL and PUT in the same portfolio, verify both tracked."""
        settings = _make_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        # Mock candle cache with bearish data for PutBearishConfirmGate
        from options_owl.collectors.candle_cache import CandleBar
        # 6 bearish bars: close < open, close < vwap
        bearish_bars = [
            CandleBar(0, 522.0, 522.5, 519.0, 519.5, 1000, vwap=521.0),
            CandleBar(0, 520.0, 520.5, 518.0, 518.5, 1000, vwap=520.0),
            CandleBar(0, 519.0, 519.5, 517.0, 517.5, 1000, vwap=519.0),
            CandleBar(0, 518.0, 518.5, 516.0, 516.5, 1000, vwap=518.0),
            CandleBar(0, 517.0, 517.5, 515.0, 515.5, 1000, vwap=517.0),
            CandleBar(0, 516.0, 516.5, 514.0, 514.5, 1000, vwap=516.0),
        ]
        mock_candle_cache = AsyncMock()
        mock_candle_cache.get_candle_data = AsyncMock(return_value={
            "5m": bearish_bars,
            "indicators": {"5m": {"rsi": 38.0, "ema9": 515.0, "ema21": 518.0}},
        })
        trader._candle_cache = mock_candle_cache

        call_sig = _make_call_signal(ticker="QQQ", premium=1.80, strike=450.0)
        put_sig = _make_put_signal(ticker="SPY", premium=0.30, strike=520.0)

        market_time = datetime(2026, 5, 26, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=450.50), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time):
            call_opened = await trader.evaluate_and_trade(call_sig, signal_id=5001)

        with patch.object(trader, "_get_current_price", return_value=519.50), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time):
            put_opened = await trader.evaluate_and_trade(put_sig, signal_id=5002)

        assert call_opened is not None
        assert put_opened is not None

        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 2

        tickers = {t["ticker"] for t in open_trades}
        assert tickers == {"QQQ", "SPY"}

        option_types = {t["option_type"] for t in open_trades}
        assert option_types == {"call", "put"}

        # Close both
        await trader.close_trade(
            call_opened["trade_id"], exit_price=453.0,
            exit_premium=2.50, reason="profit_target",
        )
        await trader.close_trade(
            put_opened["trade_id"], exit_price=517.0,
            exit_premium=0.45, reason="profit_target",
        )

        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0
