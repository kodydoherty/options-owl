from datetime import datetime, timezone

import pytest

from options_owl.collectors.discord_collector import (
    is_stand_down,
    parse_message,
    parse_performance,
    parse_trade_signal,
    parse_watchlist,
)
from options_owl.config.settings import Settings
from options_owl.journal import db
from options_owl.models.signals import (
    DailyPerformance,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)

# ---------------------------------------------------------------------------
# Sample messages from the Neverland Pirates Discord
# ---------------------------------------------------------------------------

CAPTAIN_HOOK_BEARISH = """🐻 NVDA - Bearish (PUT) 💎
100/100 (Strong) 🟢
$168.685 ➡ $167.09 (+0.9%)

🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support | Multi-TF Aligned

💼 Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $167.89 (+0.5%) | T2: $167.09 (+0.9%) | Stop: $169.43 (-0.5%)
Exit by 10:40

💰 ATM Pick
$170 put @ ~$1.70 (~+-3893% est.)
⚡ OTM Pick
$167.5 put @ ~$0.46 (~+-24007% est.)

⏱️ Time in Play
10:40 • 🟢 Low theta - full window • R:R 1.50:1

🤖 AI Analysis
Strong bearish signals with BB double-touch, VWAP rejection, and all timeframes aligned."""

NEVERLAND_PAN_BULLISH = """🐂 TSLA - Bullish (CALL) 💎
92/100 (Strong) 🟢
$368.1 ➡ $370.03 (+0.5%)

🔑 Key Signals
BB 2σ Touch | MACD Cross | EMA Bounce | VWAP Support | Multi-TF Aligned | strong directional

💼 Trade Idea
Buy Calls | Strike: $367.5 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $369.06 (+0.3%) | T2: $370.03 (+0.5%) | Stop: $367.20 (-0.3%)
Exit by 11:50

💰 ATM Pick
$367.5 call @ ~$0.93 (~+6947% est.)
⚡ OTM Pick
$370 call @ ~$0.39 (~+19112% est.)

⏱️ Time in Play
11:50 • 🟡 Moderate theta - take profits within 15 min • R:R 1.50:1

🤖 AI Analysis
Strong bullish signals with 9/21 EMA crossover, BB upper double-touch, and volume surge."""

TINKER_ELITE = """💫 TSLA - Elite Reversal (PUT) 💎
100/100 (Strong) 🟢
$363.24 ➡ $362.05 (+0.3%)

🔑 Key Signals
BB 2σ Touch | MACD Bear Cross | EMA Bounce | VWAP Support | Multi-TF Aligned | strong directional

💼 Trade Idea
Buy Puts | Strike: $362.5 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $362.64 (+0.2%) | T2: $362.05 (+0.3%) | Stop: $363.80 (-0.2%)
Exit by 12:55

💰 ATM Pick
$362.5 put @ ~$0.78 (~+-5812% est.)
⚡ OTM Pick
$360 put @ ~$0.32 (~+-16505% est.)

⏱️ Time in Play
12:55 • 🟡 Moderate theta - take profits within 15 min • R:R 1.50:1

🤖 AI Analysis
Strong bearish signals with BB double-touch, VWAP rejection, and all timeframes aligned."""

MARGINAL_SIGNAL = """🐂 MU - Bullish (CALL)
58/100 (Marginal) 🟠
$364 ➡ $367.35 (+0.9%)

🔑 Key Signals
MACD Cross | EMA Bounce | Multi-TF Aligned | strong directional

💼 Trade Idea
Buy Calls | Strike: $365 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $365.68 (+0.5%) | T2: $367.35 (+0.9%) | Stop: $362.44 (-0.5%)
Exit by 11:15

💰 ATM Pick
$365 call @ ~$1.93 (~+3129% est.)
⚡ OTM Pick
$370 call @ ~$0.86 (~+8331% est.)

🤖 AI Analysis
Strong bullish signals with all timeframes aligned."""

NEVERLAND_PAN_OTM_FIRST = """🐂 IWM - Bullish (CALL) 💎
95/100 (Strong) 🟢
**$260.225** ➡ **$262.31** (+0.8%)
🔑 Key Signals
BB 2σ Touch | Vol 1.5x | EMA Bounce | VWAP Support | Multi-TF Aligned
💼 Trade Idea
Buy Calls | Strike: $260 Call | Expiry: 0DTE | R:R 6.81:1
🎯 Price Targets
T1: $260.75 (+0.2%)
T2: $261.27 (+0.4%)
T3: $261.79 (+0.6%)
T4: $262.31 (+0.8%)
T5: $263.35 (+1.2%)
Stop: $260.01
⚡ PRIMARY: OTM Pick
$265 call @ ~$0.02
💰 Conservative: ATM
$260 call @ ~$0.70
📊 Move Quality
Vol 1.5x | VWAP Support
🤖 AI Analysis
Strong bullish setup with multiple confirmations."""

RUFIO_WATCHLIST = """🚀 Catalyst Sentinel 🛸
🌅 PRE-MARKET INTELLIGENCE BRIEF

Active Watchlist (5):
ENPH: Stage 1 (bearish, score 80)
GEMI: Stage 1 (bearish, score 80)
IMMP: Stage 1 (bearish, score 80)
BEAM: Stage 1 (bullish, score 85)
MNDY: Stage 1 (bearish, score 80)

Overnight Catalysts:
ENPH: Enphase Energy, Inc. lawsuit notice (NUCLEAR)

VIX Regime: NORMAL (15)
Macro: BEAR / RISK_OFF"""

SMEE_PERFORMANCE = """📊 DAILY PERFORMANCE SUMMARY
Today
6W / 1L (86%) | Avg PnL: 0.89%
Trades
✅ QQQ bearish | Score: 87 | PnL: 0.99%
✅ MSFT bearish | Score: 91 | PnL: 0.64%
✅ SPY bearish | Score: 81 | PnL: 0.97%
✅ MU bearish | Score: 87 | PnL: 2.06%
❌ AMD bearish | Score: 85 | PnL: -0.07%
✅ TSLA bearish | Score: 92 | PnL: 1.26%
✅ QQQ bearish | Score: 88 | PnL: 0.37%
All-Time
8/10 (80%) across 10 trades"""

SMEE_STANDDOWN = """🌔 STAND DOWN MODE
No high-edge setups. Patience is profit.
🚫 Filters
Score 46 < 75,Volume too low (need 1.3x+ avg)
📊 Status
Monitoring 12 tickers
⏱️ Next
Scanning every 5min
💡
In stillness, strategy sharpens."""

SINGLE_OTM_PICK_ONLY = """🐂 SPY - Bullish (CALL) 💎
95/100 (Strong) 🟢
$540.50 ➡ $543.20 (+0.5%)

🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support | Multi-TF Aligned

💼 Trade Idea
Buy Calls | Strike: $541 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $541.80 (+0.2%) | T2: $543.20 (+0.5%) | Stop: $539.80 (-0.1%)
Exit by 11:00

⚡ PRIMARY: OTM Pick
$545 call @ ~$0.35 (~+15000% est.)

🤖 AI Analysis
Strong bullish setup with multiple confirmations."""

SINGLE_ATM_PICK_ONLY = """🐻 AAPL - Bearish (PUT) 💎
88/100 (Strong) 🟢
$195.30 ➡ $193.50 (+0.9%)

🔑 Key Signals
BB 2σ Touch | MACD Bear Cross | EMA Bounce

💼 Trade Idea
Buy Puts | Strike: $195 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $194.40 (+0.5%) | T2: $193.50 (+0.9%) | Stop: $196.00 (-0.4%)
Exit by 12:30

💰 ATM Pick
$195 put @ ~$1.20 (~+5000% est.)

🤖 AI Analysis
Strong bearish signals with BB double-touch."""

BOTH_PICKS_SAME_STRIKE = """🐂 AMD - Bullish (CALL) 💎
90/100 (Strong) 🟢
$150.00 ➡ $152.00 (+1.3%)

🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support

💼 Trade Idea
Buy Calls | Strike: $150 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $151.00 (+0.7%) | T2: $152.00 (+1.3%) | Stop: $149.00 (-0.7%)
Exit by 11:30

💰 ATM Pick
$150 call @ ~$1.50 (~+5000% est.)
⚡ OTM Pick
$150 call @ ~$0.80 (~+10000% est.)

🤖 AI Analysis
Strong bullish signals with all timeframes aligned."""

MULTILINE_TARGETS_OTM_FIRST = """🐂 QQQ - Bullish (CALL) 💎
97/100 (Strong) 🟢
**$445.10** ➡ **$448.50** (+0.8%)
🔑 Key Signals
BB 2σ Touch | Vol 1.5x | EMA Bounce | VWAP Support | Multi-TF Aligned
💼 Trade Idea
Buy Calls | Strike: $445 Call | Expiry: 0DTE | R:R 5.50:1
🎯 Price Targets
T1: $445.80 (+0.2%)
T2: $446.50 (+0.3%)
T3: $447.20 (+0.5%)
T4: $448.50 (+0.8%)
T5: $449.80 (+1.1%)
Stop: $444.50
⚡ PRIMARY: OTM Pick
$450 call @ ~$0.10
💰 Conservative: ATM
$445 call @ ~$1.25
📊 Move Quality
Vol 1.5x | VWAP Support
🤖 AI Analysis
Strong bullish setup with multiple confirmations."""

GENERAL_CHAT = "good morning everyone, how's the market looking today?"


# ---------------------------------------------------------------------------
# Trade signal parsing
# ---------------------------------------------------------------------------


class TestTradeSignalParsing:
    def test_captain_hook_bearish_put(self):
        sig = parse_trade_signal(CAPTAIN_HOOK_BEARISH, author="Captain Hook 🗡")
        assert sig is not None
        assert sig.ticker == "NVDA"
        assert sig.sentiment == Sentiment.BEARISH
        assert sig.direction == Direction.PUT
        assert sig.score == 100
        assert sig.strength == SignalStrength.STRONG
        assert sig.is_elite is True
        assert sig.entry_price == 168.685
        assert sig.target_price == 167.09
        assert sig.strike == 170.0
        assert sig.expiry == "0DTE"
        assert sig.risk_reward == 1.50
        assert sig.target_1 == 167.89
        assert sig.target_2 == 167.09
        assert sig.stop_price == 169.43
        assert sig.exit_by == "10:40"
        assert sig.atm_strike == 170.0
        assert sig.atm_premium == 1.70
        assert sig.otm_strike == 167.5
        assert sig.otm_premium == 0.46
        assert "BB 2σ Touch" in sig.key_signals

    def test_neverland_pan_bullish_call(self):
        sig = parse_trade_signal(NEVERLAND_PAN_BULLISH, author="Neverland Pan 💸")
        assert sig is not None
        assert sig.ticker == "TSLA"
        assert sig.sentiment == Sentiment.BULLISH
        assert sig.direction == Direction.CALL
        assert sig.score == 92
        assert sig.is_elite is True
        assert sig.strike == 367.5
        assert sig.atm_premium == 0.93

    def test_tinker_elite_reversal(self):
        sig = parse_trade_signal(TINKER_ELITE, author="Tinker 🛎")
        assert sig is not None
        assert sig.ticker == "TSLA"
        assert sig.sentiment == Sentiment.BEARISH
        assert sig.direction == Direction.PUT
        assert sig.score == 100
        assert sig.is_elite is True

    def test_marginal_no_diamond(self):
        sig = parse_trade_signal(MARGINAL_SIGNAL)
        assert sig is not None
        assert sig.ticker == "MU"
        assert sig.strength == SignalStrength.MARGINAL
        assert sig.score == 58
        assert sig.is_elite is False

    def test_otm_first_format_assigns_labels_correctly(self):
        """New Discord format: PRIMARY: OTM Pick comes before Conservative: ATM."""
        sig = parse_trade_signal(NEVERLAND_PAN_OTM_FIRST, author="Neverland Pan 💸")
        assert sig is not None
        assert sig.ticker == "IWM"
        assert sig.strike == 260.0
        # OTM pick ($265 @ $0.02) comes first in text but should be assigned to otm_*
        assert sig.otm_strike == 265.0
        assert sig.otm_premium == 0.02
        # ATM pick ($260 @ $0.70) comes second but should be assigned to atm_*
        assert sig.atm_strike == 260.0
        assert sig.atm_premium == 0.70

    def test_atm_first_format_assigns_labels_correctly(self):
        """Old Discord format: ATM Pick comes before OTM Pick."""
        sig = parse_trade_signal(CAPTAIN_HOOK_BEARISH, author="Captain Hook 🗡")
        assert sig is not None
        # ATM ($170 @ $1.70) comes first → assigned to atm_*
        assert sig.atm_strike == 170.0
        assert sig.atm_premium == 1.70
        # OTM ($167.5 @ $0.46) comes second → assigned to otm_*
        assert sig.otm_strike == 167.5
        assert sig.otm_premium == 0.46

    def test_single_otm_pick_no_atm(self):
        """Only OTM pick present → otm_strike/otm_premium set, atm_* are None."""
        sig = parse_trade_signal(SINGLE_OTM_PICK_ONLY, author="Neverland Pan 💸")
        assert sig is not None
        assert sig.ticker == "SPY"
        assert sig.otm_strike == 545.0
        assert sig.otm_premium == 0.35
        assert sig.atm_strike is None
        assert sig.atm_premium is None

    def test_single_atm_pick_no_otm(self):
        """Only ATM pick present → atm_strike/atm_premium set, otm_* are None."""
        sig = parse_trade_signal(SINGLE_ATM_PICK_ONLY, author="Captain Hook 🗡")
        assert sig is not None
        assert sig.ticker == "AAPL"
        assert sig.atm_strike == 195.0
        assert sig.atm_premium == 1.20
        assert sig.otm_strike is None
        assert sig.otm_premium is None

    def test_both_picks_same_strike(self):
        """Both picks have the same strike → assigned based on label order."""
        sig = parse_trade_signal(BOTH_PICKS_SAME_STRIKE, author="Captain Hook 🗡")
        assert sig is not None
        assert sig.ticker == "AMD"
        # ATM label appears before OTM label → picks[0] is ATM, picks[1] is OTM
        assert sig.atm_strike == 150.0
        assert sig.atm_premium == 1.50
        assert sig.otm_strike == 150.0
        assert sig.otm_premium == 0.80

    def test_multiline_targets_with_otm_first(self):
        """Multi-line T1-T5 format with OTM first — verify all targets + ATM/OTM."""
        sig = parse_trade_signal(MULTILINE_TARGETS_OTM_FIRST, author="Neverland Pan 💸")
        assert sig is not None
        assert sig.ticker == "QQQ"
        assert sig.score == 97
        assert sig.strike == 445.0
        assert sig.risk_reward == 5.50
        # All 5 targets parsed
        assert sig.target_1 == 445.80
        assert sig.target_2 == 446.50
        assert sig.target_3 == 447.20
        assert sig.target_4 == 448.50
        assert sig.target_5 == 449.80
        assert sig.stop_price == 444.50
        # OTM label comes first in message → OTM fields get the $450 pick
        assert sig.otm_strike == 450.0
        assert sig.otm_premium == 0.10
        # ATM label comes second → ATM fields get the $445 pick
        assert sig.atm_strike == 445.0
        assert sig.atm_premium == 1.25

    def test_non_signal_returns_none(self):
        assert parse_trade_signal(GENERAL_CHAT) is None
        assert parse_trade_signal(SMEE_STANDDOWN) is None


# ---------------------------------------------------------------------------
# Watchlist parsing (Rufio)
# ---------------------------------------------------------------------------


class TestWatchlistParsing:
    def test_rufio_watchlist(self):
        entries = parse_watchlist(RUFIO_WATCHLIST)
        assert len(entries) == 5
        assert entries[0].ticker == "ENPH"
        assert entries[0].sentiment == Sentiment.BEARISH
        assert entries[0].score == 80
        assert entries[3].ticker == "BEAM"
        assert entries[3].sentiment == Sentiment.BULLISH
        assert entries[3].score == 85

    def test_non_watchlist_returns_empty(self):
        assert parse_watchlist(GENERAL_CHAT) == []


# ---------------------------------------------------------------------------
# Performance parsing (Smee)
# ---------------------------------------------------------------------------


class TestPerformanceParsing:
    def test_smee_daily_summary(self):
        perf = parse_performance(SMEE_PERFORMANCE)
        assert perf is not None
        assert perf.wins == 6
        assert perf.losses == 1
        assert perf.win_rate_pct == 86.0
        assert perf.avg_pnl_pct == 0.89
        assert len(perf.trades) == 7
        assert perf.trades[0].ticker == "QQQ"
        assert perf.trades[0].won is True
        assert perf.trades[0].pnl_pct == 0.99
        assert perf.trades[4].ticker == "AMD"
        assert perf.trades[4].won is False
        assert perf.all_time_wins == 8
        assert perf.all_time_total == 10

    def test_non_performance_returns_none(self):
        assert parse_performance(GENERAL_CHAT) is None


# ---------------------------------------------------------------------------
# Stand-down detection
# ---------------------------------------------------------------------------


class TestStandDown:
    def test_stand_down_detected(self):
        assert is_stand_down(SMEE_STANDDOWN) is True

    def test_non_stand_down(self):
        assert is_stand_down(GENERAL_CHAT) is False


# ---------------------------------------------------------------------------
# Unified parse_message
# ---------------------------------------------------------------------------


class TestParseMessage:
    def test_trade_signal(self):
        result = parse_message(CAPTAIN_HOOK_BEARISH, author="Captain Hook 🗡")
        assert isinstance(result, TradeSignal)

    def test_performance(self):
        result = parse_message(SMEE_PERFORMANCE)
        assert isinstance(result, DailyPerformance)

    def test_watchlist(self):
        result = parse_message(RUFIO_WATCHLIST)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_general_chat_returns_none(self):
        assert parse_message(GENERAL_CHAT) is None


# ---------------------------------------------------------------------------
# DB tests
# ---------------------------------------------------------------------------


class TestDatabase:
    @pytest.mark.asyncio
    async def test_init_and_save_message(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await db.save_message(
            tmp_db_path,
            guild_id=123,
            channel_id=456,
            author_id=789,
            author_name="Captain Hook 🗡",
            content=CAPTAIN_HOOK_BEARISH,
            timestamp=datetime.now(timezone.utc),
        )
        assert msg_id is not None
        row = await db.get_message(tmp_db_path, msg_id)
        assert row is not None
        assert "NVDA" in row["content"]

    @pytest.mark.asyncio
    async def test_save_signal_roundtrip(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await db.save_message(
            tmp_db_path,
            guild_id=123,
            channel_id=456,
            author_id=789,
            author_name="Captain Hook 🗡",
            content=CAPTAIN_HOOK_BEARISH,
            timestamp=datetime.now(timezone.utc),
        )
        sig_id = await db.save_signal(
            tmp_db_path,
            message_id=msg_id,
            ticker="NVDA",
            strike=170.0,
            expiry="0DTE",
            direction="put",
            premium=1.70,
            action="buy",
            confidence=1.0,
        )
        assert sig_id is not None
        signals = await db.get_signals_for_message(tmp_db_path, msg_id)
        assert len(signals) == 1
        assert signals[0]["ticker"] == "NVDA"
        assert signals[0]["strike"] == 170.0


class TestTradeSignalDB:
    @pytest.mark.asyncio
    async def test_save_and_get_trade_signal(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await db.save_message(
            tmp_db_path,
            guild_id=123,
            channel_id=456,
            author_id=789,
            author_name="Captain Hook 🗡",
            content=CAPTAIN_HOOK_BEARISH,
            timestamp=datetime.now(timezone.utc),
        )
        sig = parse_trade_signal(CAPTAIN_HOOK_BEARISH, author="Captain Hook 🗡")
        assert sig is not None
        sig_id = await db.save_trade_signal(
            tmp_db_path,
            message_id=msg_id,
            signal=sig.model_dump(mode="json"),
        )
        row = await db.get_trade_signal(tmp_db_path, sig_id)
        assert row is not None
        assert row["ticker"] == "NVDA"
        assert row["score"] == 100
        assert row["entry_price"] == 168.685
        assert row["target_1"] == 167.89
        assert row["stop_price"] == 169.43
        assert row["is_elite"] is True
        assert "BB 2σ Touch" in row["key_signals"]

    @pytest.mark.asyncio
    async def test_unresolved_signals(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await db.save_message(
            tmp_db_path,
            guild_id=123,
            channel_id=456,
            author_id=789,
            author_name="Captain Hook 🗡",
            content="test",
            timestamp=datetime.now(timezone.utc),
        )
        await db.save_trade_signal(
            tmp_db_path,
            message_id=msg_id,
            signal={
                "bot_source": "Captain Hook",
                "ticker": "NVDA",
                "sentiment": "bearish",
                "direction": "put",
                "score": 100,
                "strength": "strong",
                "entry_price": 170.0,
                "target_price": 167.0,
                "expected_move_pct": 0.9,
                "strike": 170.0,
                "expiry": "0DTE",
                "risk_reward": 1.5,
                "key_signals": [],
                "is_elite": True,
            },
        )
        unresolved = await db.get_unresolved_signals(tmp_db_path)
        assert len(unresolved) == 1
        assert unresolved[0]["ticker"] == "NVDA"


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DISCORD_TOKEN", raising=False)
        settings = Settings(DISCORD_TOKEN="fake-token")
        assert settings.PAPER_TRADE is True
        assert 1469404711613497591 in settings.guild_ids
        assert settings.channel_ids == []
