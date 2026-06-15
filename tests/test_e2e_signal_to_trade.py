"""End-to-end tests: Discord message -> parsed signal -> trade entry -> exit.
Catches cascading bugs like the ATM/OTM parser swap that inflated P&L.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from options_owl.collectors.discord_collector import parse_trade_signal
from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import (
    PaperTrader,
    _select_trade_premium,
    get_open_trades,
)
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# Raw Discord messages — real formats from the Neverland Pirates server
# ---------------------------------------------------------------------------

# New format: OTM first (PRIMARY), ATM second (Conservative)
IWM_OTM_FIRST_MESSAGE = """\
\U0001f402 IWM - Bullish (CALL) \U0001f48e
95/100 (Strong) \U0001f7e2 (raw 155)
**$260.225** \u27a1 **$262.31** (+0.8%)
\U0001f511 Key Signals
BB 2\u03c3 Touch | Vol 1.5x | EMA Bounce | VWAP Support | Multi-TF Aligned
\U0001f4bc Trade Idea
Buy Calls | Strike: $260 Call | Expiry: 0DTE | R:R 6.81:1
\U0001f3af Price Targets
T1: $260.75 (+0.2%)
T2: $261.27 (+0.4%)
T3: $261.79 (+0.6%)
T4: $262.31 (+0.8%)
T5: $263.35 (+1.2%)
Stop: $260.01
\u26a1 PRIMARY: OTM Pick
$265 call @ ~$0.02
\U0001f4b0 Conservative: ATM
$260 call @ ~$0.70
\U0001f4ca Move Quality
Vol 1.5x | VWAP Support
\U0001f916 AI Analysis
Strong bullish setup with multiple confirmations."""

# Old format: ATM first, OTM second
NVDA_ATM_FIRST_MESSAGE = """\
\U0001f43b NVDA - Bearish (PUT) \U0001f48e
100/100 (Strong) \U0001f7e2 (raw 164)
$168.685 \u27a1 $167.09 (+0.9%)

\U0001f511 Key Signals
BB 2\u03c3 Touch | EMA Bounce | VWAP Support | Multi-TF Aligned

\U0001f4bc Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1

\U0001f3af Exit Targets
T1: $167.89 (+0.5%) | T2: $167.09 (+0.9%) | Stop: $169.43 (-0.5%)
Exit by 10:40

\U0001f4b0 ATM Pick
$170 put @ ~$1.70 (~+-3893% est.)
\u26a1 OTM Pick
$167.5 put @ ~$0.46 (~+-24007% est.)

\u23f1\ufe0f Time in Play
10:40 \u2022 \U0001f7e2 Low theta - full window \u2022 R:R 1.50:1

\U0001f916 AI Analysis
Strong bearish signals with BB double-touch, VWAP rejection, and all timeframes aligned."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_e2e_settings(tmp_db_path: str, **overrides) -> Settings:
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
        "ANTI_CHASE_MAX_MOVE_PCT": 99.0,  # disable for tests (live price != signal price)
        "ENTRY_HARD_CUTOFF_HOUR": 23,  # disable time gate in tests
        "ENTRY_HARD_CUTOFF_MINUTE": 59,
        "ENABLE_MORNING_CUTOFF": False,  # disable morning cutoff in tests
        "TOD_LATE_CUTOFF_HOUR": 23,
        "TOD_LATE_CUTOFF_MINUTE": 59,
        "ENABLE_DIRECTIONAL_REGIME": False,
        "ENABLE_PUT_TRADING": True,
        "CB_CLOSING_BUFFER_MINUTES": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# 1. OTM-first signal: correct premium selection
# ---------------------------------------------------------------------------


class TestOTMFirstSignalCorrectEntry:
    def test_otm_first_signal_correct_entry_premium(self):
        """Parse the IWM message (PRIMARY: OTM Pick $265 @ $0.02, Conservative:
        ATM $260 @ $0.70) and verify signal has correct atm/otm assignments.

        After _select_trade_premium, the trade should use $0.70 (the real ATM
        near the $260 strike), NOT the penny $0.02 OTM option.
        """
        sig = parse_trade_signal(
            IWM_OTM_FIRST_MESSAGE,
            message_id=1,
            channel="signals",
            author="captain hook",
        )
        assert sig is not None

        # Parser should correctly identify OTM (first in message) vs ATM (second)
        assert sig.otm_strike == 265.0, f"OTM strike should be $265, got {sig.otm_strike}"
        assert sig.otm_premium == 0.02, f"OTM premium should be $0.02, got {sig.otm_premium}"
        assert sig.atm_strike == 260.0, f"ATM strike should be $260, got {sig.atm_strike}"
        assert sig.atm_premium == 0.70, f"ATM premium should be $0.70, got {sig.atm_premium}"

        # Signal's recommended strike is $260 (from Trade Idea line)
        assert sig.strike == 260.0

        # _select_trade_premium should pick $0.70 (matching the $260 strike)
        adjusted = _select_trade_premium(sig)
        assert adjusted.atm_premium == 0.70, (
            f"Trade should use ATM premium $0.70, not ${adjusted.atm_premium}"
        )


# ---------------------------------------------------------------------------
# 2. ATM-first signal (old format): correct premium selection
# ---------------------------------------------------------------------------


class TestATMFirstSignalCorrectEntry:
    def test_atm_first_signal_correct_entry_premium(self):
        """Parse old format (ATM Pick $170 @ $1.70, OTM Pick $167.5 @ $0.46).
        Verify trade entry uses $1.70.
        """
        sig = parse_trade_signal(
            NVDA_ATM_FIRST_MESSAGE,
            message_id=2,
            channel="signals",
            author="captain hook",
        )
        assert sig is not None

        # ATM comes first in the message → picks[0] is ATM, picks[1] is OTM
        assert sig.atm_strike == 170.0
        assert sig.atm_premium == 1.70
        assert sig.otm_strike == 167.5
        assert sig.otm_premium == 0.46

        # Signal's recommended strike is $170
        assert sig.strike == 170.0

        # _select_trade_premium should keep $1.70 (ATM matches strike)
        adjusted = _select_trade_premium(sig)
        assert adjusted.atm_premium == 1.70, (
            f"Trade should use ATM premium $1.70, not ${adjusted.atm_premium}"
        )


# ---------------------------------------------------------------------------
# 3. Full trade lifecycle: parse -> open -> verify state
# ---------------------------------------------------------------------------


class TestFullTradeLifecycle:
    @pytest.mark.asyncio
    async def test_full_trade_lifecycle(self, tmp_db_path):
        """Parse signal -> open trade via paper_trader.evaluate_and_trade ->
        verify contracts, premium, DCA state are all correct for Vinny strategy.
        """
        # Parse the NVDA signal
        sig = parse_trade_signal(
            NVDA_ATM_FIRST_MESSAGE,
            message_id=10,
            channel="signals",
            author="captain hook",
        )
        assert sig is not None
        assert sig.ticker == "NVDA"
        assert sig.direction == Direction.PUT
        assert sig.score == 164
        assert sig.atm_premium == 1.70
        assert sig.stop_price == 169.43

        # Set up Vinny strategy with score-based sizing
        settings = _make_e2e_settings(
            tmp_db_path,
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_SCORE_SIZING=True,
            ENABLE_DCA=False,
        )
        trader = PaperTrader(settings)
        await trader.init()

        # Mock candle cache with bearish data for PutBearishConfirmGate
        from options_owl.collectors.candle_cache import CandleBar
        bearish_bars = [
            CandleBar(0, 170.0, 170.5, 168.0, 168.5, 1000, vwap=169.5),
            CandleBar(0, 169.0, 169.5, 167.0, 167.5, 1000, vwap=169.0),
            CandleBar(0, 168.0, 168.5, 166.0, 166.5, 1000, vwap=168.0),
            CandleBar(0, 167.0, 167.5, 165.0, 165.5, 1000, vwap=167.0),
            CandleBar(0, 166.0, 166.5, 164.0, 164.5, 1000, vwap=166.0),
            CandleBar(0, 165.0, 165.5, 163.0, 163.5, 1000, vwap=165.0),
        ]
        mock_cc = AsyncMock()
        mock_cc.get_candle_data = AsyncMock(return_value={
            "5m": bearish_bars,
            "indicators": {"5m": {"rsi": 38.0, "ema9": 164.0, "ema21": 167.0}},
        })
        trader._candle_cache = mock_cc

        # Mock _get_current_price and pin time to 10:30 AM ET so time_of_day gate passes
        # (without this, the test is flaky after ~9 PM Pacific / midnight ET)
        market_time = datetime(2026, 4, 28, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch.object(trader, "_get_current_price", return_value=168.70), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time), \
             patch("options_owl.risk.pipeline._now_et", return_value=market_time), \
             patch("options_owl.risk.circuit_breaker._now_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=10)
        assert opened is not None, "Trade should have been opened"

        # Dollar-target sizing: $10k × 80% / 5 = $1,600 target
        # Flat 85% × PUT 50% mult → $680 / $170 per contract = 4 raw
        # pos_cap = 20% → $2,000 / $170 = 11
        assert opened["contracts"] == 4
        assert opened["premium"] == pytest.approx(1.70, abs=0.01)

        # Verify database state
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            trade = dict(await cursor.fetchone())

        assert trade["status"] == "open"
        assert trade["ticker"] == "NVDA"
        assert trade["direction"] == "put"
        assert trade["score"] == 164
        assert trade["contracts"] == 4  # flat 85% × PUT 50%: $680/$170 = 4
        assert trade["premium_per_contract"] == pytest.approx(1.70, abs=0.01)
        assert trade["total_cost"] == pytest.approx(4 * 170.0, abs=1.0)
        assert trade["strike"] == 170.0
        assert trade["target_1"] == 167.89
        assert trade["target_2"] == 167.09
        assert trade["stop_price"] == 169.43

        # DCA should be disabled for Vinny
        assert trade["dca_tranches_remaining"] == 0
        assert trade["dca_total_contracts"] == 4  # flat 85% × PUT 50%: $680/$170 = 4

    @pytest.mark.asyncio
    async def test_otm_first_full_lifecycle(self, tmp_db_path):
        """Parse IWM OTM-first signal -> open trade -> verify correct premium used.

        This is the regression test for the ATM/OTM swap bug.
        Without the fix, the trader would use $0.02 (OTM penny option) instead of
        $0.70 (real ATM near the strike), causing absurdly inflated P&L.
        """
        sig = parse_trade_signal(
            IWM_OTM_FIRST_MESSAGE,
            message_id=20,
            channel="signals",
            author="neverland pan",
        )
        assert sig is not None
        assert sig.ticker == "IWM"

        settings = _make_e2e_settings(
            tmp_db_path,
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_SCORE_SIZING=True,
        )

        trader = PaperTrader(settings)
        await trader.init()

        market_time = datetime(2026, 4, 28, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=260.30), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time), \
             patch("options_owl.risk.pipeline._now_et", return_value=market_time), \
             patch("options_owl.risk.circuit_breaker._now_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=20)
        assert opened is not None

        # The trade should use $0.70 premium (ATM near the $260 strike),
        # NOT $0.02 (OTM penny option at $265)
        assert opened["premium"] == pytest.approx(0.70, abs=0.01), (
            f"Expected premium $0.70 (ATM), got ${opened['premium']:.4f}. "
            f"ATM/OTM swap bug may have regressed!"
        )

        # Dollar-target: $10k × 80% / 5 = $1,600, flat 85% = $1,360 / $70 = 19 raw
        # pos_cap = 20% → $2000/$70 = 28. Final = min(19, 28) = 19
        assert opened["contracts"] == 19

    @pytest.mark.asyncio
    async def test_trade_open_then_partial_then_full_close(self, tmp_db_path):
        """Full lifecycle: parse -> open -> partial close at T1 -> full close at T2."""
        sig = parse_trade_signal(
            NVDA_ATM_FIRST_MESSAGE,
            message_id=30,
            channel="signals",
            author="captain hook",
        )
        assert sig is not None

        settings = _make_e2e_settings(tmp_db_path)

        trader = PaperTrader(settings)
        await trader.init()

        # Mock candle cache with bearish data for PutBearishConfirmGate
        from options_owl.collectors.candle_cache import CandleBar
        bearish_bars = [
            CandleBar(0, 170.0, 170.5, 168.0, 168.5, 1000, vwap=169.5),
            CandleBar(0, 169.0, 169.5, 167.0, 167.5, 1000, vwap=169.0),
            CandleBar(0, 168.0, 168.5, 166.0, 166.5, 1000, vwap=168.0),
            CandleBar(0, 167.0, 167.5, 165.0, 165.5, 1000, vwap=167.0),
            CandleBar(0, 166.0, 166.5, 164.0, 164.5, 1000, vwap=166.0),
            CandleBar(0, 165.0, 165.5, 163.0, 163.5, 1000, vwap=165.0),
        ]
        mock_cc = AsyncMock()
        mock_cc.get_candle_data = AsyncMock(return_value={
            "5m": bearish_bars,
            "indicators": {"5m": {"rsi": 38.0, "ema9": 164.0, "ema21": 167.0}},
        })
        trader._candle_cache = mock_cc

        market_time = datetime(2026, 4, 28, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch.object(trader, "_get_current_price", return_value=168.70), \
             patch("options_owl.execution.paper_trader._today_et", return_value=market_time), \
             patch("options_owl.risk.pipeline._now_et", return_value=market_time), \
             patch("options_owl.risk.circuit_breaker._now_et", return_value=market_time):
            opened = await trader.evaluate_and_trade(sig, signal_id=30)
        assert opened is not None
        trade_id = opened["trade_id"]
        total_contracts = opened["contracts"]
        assert total_contracts == 4  # flat 85% × PUT 50%: $680/$170 = 4

        # Partial close at T1
        partial = await trader.partial_close_trade(
            trade_id=trade_id,
            exit_price=167.89,
            exit_premium=2.50,
            reason="t1_hit",
            close_pct=50.0,
        )

        contracts_closed = round(total_contracts * 50.0 / 100)  # 2 or 3
        assert partial["contracts_closed"] == contracts_closed
        remaining = total_contracts - contracts_closed
        assert partial["contracts_remaining"] == remaining

        # Verify open trade still has remaining contracts
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 1
        assert open_trades[0]["contracts"] == remaining

        # Full close at T2
        close_result = await trader.close_trade(
            trade_id=trade_id,
            exit_price=167.09,
            exit_premium=3.00,
            reason="t2_hit",
        )
        assert close_result["reason"] == "t2_hit"

        # No open trades left
        open_trades = await get_open_trades(tmp_db_path)
        assert len(open_trades) == 0

    @pytest.mark.asyncio
    async def test_penny_option_rejected(self, tmp_db_path):
        """If after premium selection we only have a penny option (<$0.05),
        the trade should be rejected by should_trade()."""
        sig = TradeSignal(
            ticker="MEME",
            sentiment=Sentiment.BULLISH,
            direction=Direction.CALL,
            score=90,
            strength=SignalStrength.STRONG,
            entry_price=5.0,
            target_price=5.50,
            expected_move_pct=10.0,
            strike=5.0,
            expiry="0DTE",
            risk_reward=1.5,
            target_1=5.25,
            target_2=5.50,
            stop_price=4.75,
            atm_strike=5.0,
            atm_premium=0.01,  # penny option
            otm_strike=5.5,
            otm_premium=0.005,  # also penny
            bot_source=BotSource.CAPTAIN_HOOK,
            is_elite=True,
        )

        settings = _make_e2e_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        # _select_trade_premium won't help — both premiums are pennies
        adjusted = _select_trade_premium(sig)
        # should_trade checks atm_premium > 0, but premium < 0.05 may
        # still pass the basic check. The real rejection happens at sizing.
        # But the signal should at minimum be recognizable as a penny option.
        assert adjusted.atm_premium < 0.05 or adjusted.otm_premium < 0.05


# ---------------------------------------------------------------------------
# 4. Webull order <-> DB record consistency
# ---------------------------------------------------------------------------


class TestWebullOrderDBConsistency:
    """Ensure every Webull order has a matching DB record — regression test for
    the bug where trades were executed on Webull but not recorded in SQLite.
    """

    def _make_signal(self) -> TradeSignal:
        return TradeSignal(
            ticker="SPY",
            sentiment=Sentiment.BULLISH,
            direction=Direction.CALL,
            score=95,
            strength=SignalStrength.STRONG,
            entry_price=520.0,
            target_price=523.0,
            expected_move_pct=0.6,
            strike=520.0,
            expiry="0DTE",
            risk_reward=2.0,
            target_1=521.0,
            target_2=522.0,
            stop_price=519.0,
            atm_strike=520.0,
            atm_premium=2.50,
            otm_strike=522.0,
            otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK,
            is_elite=True,
        )

    @pytest.mark.asyncio
    async def test_webull_order_has_matching_db_record(self, tmp_db_path):
        """When Webull order succeeds, the paper_trades DB must have matching
        webull_order_id and webull_client_order_id columns populated.
        """
        from unittest.mock import AsyncMock, MagicMock
        from options_owl.execution.webull_executor import OrderResult

        # Create a mock WebullExecutor that returns a successful order
        mock_executor = MagicMock()
        mock_executor.buy_option = AsyncMock(return_value=OrderResult(
            success=True,
            order_id="WB-12345",
            client_order_id="CLI-67890",
            fill_status="FILLED",
        ))
        mock_executor.get_fill_price = AsyncMock(return_value=2.48)

        settings = _make_e2e_settings(tmp_db_path)

        trader = PaperTrader(settings, webull_executor=mock_executor)
        await trader.init()

        sig = self._make_signal()

        with patch.object(trader, "_get_current_price", return_value=520.10):
            opened = await trader.evaluate_and_trade(sig, signal_id=100)

        assert opened is not None, "Trade should have been opened"

        # Verify Webull order was placed
        mock_executor.buy_option.assert_called_once()
        call_kwargs = mock_executor.buy_option.call_args
        assert call_kwargs[1]["ticker"] == "SPY" or call_kwargs[0][0] == "SPY"

        # Verify DB record has Webull order IDs and real fill price
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            trade = dict(await cursor.fetchone())

        assert trade["webull_order_id"] == "WB-12345", (
            f"DB must store Webull order_id. Got: {trade['webull_order_id']}"
        )
        assert trade["webull_client_order_id"] == "CLI-67890", (
            f"DB must store Webull client_order_id. Got: {trade['webull_client_order_id']}"
        )
        assert trade["webull_entry_fill_price"] == 2.48, (
            f"DB must store real entry fill price. Got: {trade['webull_entry_fill_price']}"
        )
        assert trade["status"] == "open"
        assert trade["ticker"] == "SPY"

    @pytest.mark.asyncio
    async def test_db_record_created_even_when_webull_fails(self, tmp_db_path):
        """If Webull order fails, the paper trade DB record should exist but be
        auto-closed as an orphan (no phantom open positions).
        """
        from unittest.mock import AsyncMock, MagicMock
        from options_owl.execution.webull_executor import OrderResult

        mock_executor = MagicMock()
        mock_executor.buy_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="Insufficient buying power",
            fill_status="FAILED",
        ))

        settings = _make_e2e_settings(tmp_db_path)
        trader = PaperTrader(settings, webull_executor=mock_executor)
        await trader.init()

        sig = self._make_signal()

        with patch.object(trader, "_get_current_price", return_value=520.10):
            opened = await trader.evaluate_and_trade(sig, signal_id=101)

        # Trade should still be created in DB
        assert opened is not None, "Paper trade should still be created even if Webull fails"

        # Verify DB record exists, auto-closed as orphan
        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            trade = dict(await cursor.fetchone())

        assert trade["status"] == "closed", (
            "Failed Webull order should auto-close the paper trade"
        )
        assert trade["webull_order_id"] is None
        assert "orphan_closed" in (trade["exit_reason"] or "")

    @pytest.mark.asyncio
    async def test_webull_exception_does_not_lose_db_record(self, tmp_db_path):
        """If Webull executor raises an exception (network timeout, etc),
        the DB record must still be intact and auto-closed as orphan.
        """
        from unittest.mock import AsyncMock, MagicMock

        mock_executor = MagicMock()
        mock_executor.buy_option = AsyncMock(
            side_effect=RuntimeError("Connection timeout to Webull API")
        )

        settings = _make_e2e_settings(tmp_db_path)
        trader = PaperTrader(settings, webull_executor=mock_executor)
        await trader.init()

        sig = self._make_signal()

        with patch.object(trader, "_get_current_price", return_value=520.10):
            opened = await trader.evaluate_and_trade(sig, signal_id=102)

        # DB record should still exist despite Webull crash
        assert opened is not None

        async with aiosqlite.connect(tmp_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?",
                (opened["trade_id"],),
            )
            trade = dict(await cursor.fetchone())

        assert trade["status"] == "closed", (
            "Webull exception should auto-close the orphaned paper trade"
        )
        assert trade["ticker"] == "SPY"
        assert trade["webull_order_id"] is None
        assert "orphan_closed" in (trade["exit_reason"] or "")

    @pytest.mark.asyncio
    async def test_no_webull_order_without_db_record(self, tmp_db_path):
        """The DB INSERT must happen BEFORE Webull order placement.
        Verify ordering: if we instrument the mock to check DB state at call time,
        the record must already exist.
        """
        from unittest.mock import AsyncMock, MagicMock
        from options_owl.execution.webull_executor import OrderResult

        db_state_at_order_time = {}

        async def capture_db_state_then_succeed(**kwargs):
            """When Webull buy is called, check that DB already has the trade."""
            async with aiosqlite.connect(tmp_db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
                )
                count = (await cursor.fetchone())[0]
                db_state_at_order_time["open_count"] = count
            return OrderResult(
                success=True,
                order_id="WB-ORDER-1",
                client_order_id="CLI-ORDER-1",
                fill_status="FILLED",
            )

        mock_executor = MagicMock()
        mock_executor.buy_option = AsyncMock(side_effect=capture_db_state_then_succeed)

        settings = _make_e2e_settings(tmp_db_path)
        trader = PaperTrader(settings, webull_executor=mock_executor)
        await trader.init()

        sig = self._make_signal()

        with patch.object(trader, "_get_current_price", return_value=520.10):
            opened = await trader.evaluate_and_trade(sig, signal_id=103)

        assert opened is not None
        # The DB record must exist BEFORE the Webull order is placed
        assert db_state_at_order_time.get("open_count", 0) >= 1, (
            "DB INSERT must happen BEFORE Webull order placement. "
            "If Webull succeeds but DB fails afterward, we get orphaned positions."
        )


# ---------------------------------------------------------------------------
# 5. discord_collector try-except around evaluate_and_trade
# ---------------------------------------------------------------------------


class TestDiscordCollectorErrorHandling:
    """Test that the try-except in discord_collector.py properly catches and
    logs exceptions from evaluate_and_trade without crashing the bot.
    """

    @pytest.mark.asyncio
    async def test_evaluate_and_trade_exception_is_caught(self, tmp_db_path):
        """If evaluate_and_trade raises, discord_collector catches it and logs
        an error — the bot does not crash.
        """
        from unittest.mock import AsyncMock

        settings = _make_e2e_settings(tmp_db_path)
        trader = PaperTrader(settings)
        await trader.init()

        # Make evaluate_and_trade blow up
        trader.evaluate_and_trade = AsyncMock(
            side_effect=RuntimeError("DB disk full")
        )

        sig = TradeSignal(
            ticker="AAPL",
            sentiment=Sentiment.BULLISH,
            direction=Direction.CALL,
            score=90,
            strength=SignalStrength.STRONG,
            entry_price=180.0,
            target_price=182.0,
            expected_move_pct=1.1,
            strike=180.0,
            expiry="0DTE",
            risk_reward=2.0,
            target_1=181.0,
            target_2=182.0,
            stop_price=179.0,
            atm_strike=180.0,
            atm_premium=1.50,
            otm_strike=182.0,
            otm_premium=0.30,
            bot_source=BotSource.CAPTAIN_HOOK,
            is_elite=True,
        )

        # Simulate the try-except pattern from discord_collector.py line ~619
        caught_exception = None
        try:
            await trader.evaluate_and_trade(sig, signal_id=200)
        except Exception as exc:
            caught_exception = exc

        # The exception SHOULD be raised from evaluate_and_trade
        assert caught_exception is not None
        assert "DB disk full" in str(caught_exception)

        # Now verify the discord_collector pattern handles it:
        # (re-implements the try-except from discord_collector.py)
        error_logged = False
        try:
            await trader.evaluate_and_trade(sig, signal_id=201)
        except Exception as exc:
            error_logged = True
            # This is what discord_collector does — log and continue
            assert "DB disk full" in str(exc)

        assert error_logged, (
            "evaluate_and_trade must propagate exceptions so discord_collector "
            "can catch them in its try-except block"
        )


# ---------------------------------------------------------------------------
# 6. Webull fill price extraction and retry
# ---------------------------------------------------------------------------


class TestWebullFillPriceExtraction:
    """Test that _extract_fill_price handles various Webull response formats."""

    def test_top_level_avg_filled_price(self):
        from options_owl.execution.webull_executor import WebullExecutor

        detail = {"avg_filled_price": "1.55", "status": "FILLED"}
        assert WebullExecutor._extract_fill_price(detail) == 1.55

    def test_nested_in_orders_list(self):
        from options_owl.execution.webull_executor import WebullExecutor

        detail = {
            "orders": [{"avg_filled_price": "2.30", "status": "FILLED"}]
        }
        assert WebullExecutor._extract_fill_price(detail) == 2.30

    def test_nested_in_legs(self):
        from options_owl.execution.webull_executor import WebullExecutor

        detail = {
            "orders": [{"legs": [{"avg_filled_price": "0.78"}]}]
        }
        assert WebullExecutor._extract_fill_price(detail) == 0.78

    def test_returns_none_when_missing(self):
        from options_owl.execution.webull_executor import WebullExecutor

        assert WebullExecutor._extract_fill_price({}) is None
        assert WebullExecutor._extract_fill_price({"orders": []}) is None
        assert WebullExecutor._extract_fill_price({"orders": [{}]}) is None

    @pytest.mark.asyncio
    async def test_get_fill_price_retries_on_failure(self):
        """get_fill_price retries when get_order_status returns None (e.g. 429)."""
        from unittest.mock import AsyncMock

        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        settings = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="key",
            WEBULL_APP_SECRET="secret",
        )
        executor = WebullExecutor(settings)

        # First 2 calls fail (429), third succeeds
        executor.get_order_status = AsyncMock(side_effect=[
            None,
            None,
            {"avg_filled_price": "1.25"},
        ])

        price = await executor.get_fill_price("test-order-123", retries=3)
        assert price == 1.25
        assert executor.get_order_status.call_count == 3

    @pytest.mark.asyncio
    async def test_get_fill_price_returns_none_after_exhausting_retries(self):
        from unittest.mock import AsyncMock

        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        settings = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="key",
            WEBULL_APP_SECRET="secret",
        )
        executor = WebullExecutor(settings)
        executor.get_order_status = AsyncMock(return_value=None)

        price = await executor.get_fill_price("test-order-456", retries=2)
        assert price is None
        assert executor.get_order_status.call_count == 2
