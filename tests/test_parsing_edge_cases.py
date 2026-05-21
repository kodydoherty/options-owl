"""Edge case tests for signal parsing — malformed messages, partial data, unusual formats."""

from options_owl.collectors.discord_collector import (
    parse_message,
    parse_performance,
    parse_trade_signal,
    parse_watchlist,
)
from options_owl.models.signals import Direction, Sentiment, TradeSignal


# ---------------------------------------------------------------------------
# Malformed / partial signals
# ---------------------------------------------------------------------------


class TestMalformedSignals:
    def test_empty_string(self):
        assert parse_trade_signal("") is None

    def test_only_header_no_body(self):
        text = "🐻 SPY - Bearish (PUT) 💎"
        assert parse_trade_signal(text) is None

    def test_header_and_score_but_no_trade(self):
        text = """🐻 SPY - Bearish (PUT) 💎
100/100 (Strong) 🟢
$550.00 ➡ $548.00 (+0.4%)"""
        assert parse_trade_signal(text) is None

    def test_missing_score(self):
        text = """🐻 SPY - Bearish (PUT)
$550.00 ➡ $548.00 (+0.4%)
Buy Puts | Strike: $550 Put | Expiry: 0DTE | R:R 1.50:1"""
        assert parse_trade_signal(text) is None

    def test_garbage_text(self):
        assert parse_trade_signal("🚀🚀🚀 to the moon!!!") is None

    def test_looks_like_signal_but_isnt(self):
        text = "SPY 550C 0DTE looking good, might buy at open"
        assert parse_trade_signal(text) is None


# ---------------------------------------------------------------------------
# T1-T5 new format parsing
# ---------------------------------------------------------------------------


NEW_FORMAT_SIGNAL = """🐂 SPY - Bullish (CALL) 💎
95/100 (Strong) 🟢
$575.50 ➡ $579.00 (+0.6%)

🔑 Key Signals
BB 2σ Touch | MACD Cross | EMA Bounce

💼 Trade Idea
Buy Calls | Strike: $576 Call | Expiry: 0DTE | R:R 2.00:1

🎯 Exit Targets
T1: $576.90 | T2: $577.50 | T3: $578.00 | T4: $578.31 | T5: $579.00 | Stop: $574.18
Exit by 11:00

💰 ATM Pick
$576 call @ ~$1.50 (~+5000% est.)"""


class TestNewFormatTargets:
    def test_all_five_targets_parsed(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL, author="Captain Hook")
        assert sig is not None
        assert sig.target_1 == 576.90
        assert sig.target_2 == 577.50
        assert sig.target_3 == 578.00
        assert sig.target_4 == 578.31
        assert sig.target_5 == 579.00
        assert sig.stop_price == 574.18

    def test_direction_is_call(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL)
        assert sig.direction == Direction.CALL

    def test_rr_parsed(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL)
        assert sig.risk_reward == 2.00


# ---------------------------------------------------------------------------
# Bot source detection
# ---------------------------------------------------------------------------


class TestBotDetection:
    def test_captain_hook(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL, author="Captain Hook 🗡")
        assert sig.bot_source.value == "Captain Hook"

    def test_neverland_pan(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL, author="Neverland Pan 💸")
        assert sig.bot_source.value == "Neverland Pan"

    def test_tinker(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL, author="Tinker 🛎")
        assert sig.bot_source.value == "Tinker"

    def test_unknown_author(self):
        sig = parse_trade_signal(NEW_FORMAT_SIGNAL, author="SomeRandomBot")
        assert sig.bot_source.value == "unknown"


# ---------------------------------------------------------------------------
# parse_message routing
# ---------------------------------------------------------------------------


class TestMessageRouting:
    def test_trade_signal_prioritized_over_watchlist(self):
        """Trade signals should be returned even if watchlist keywords present."""
        result = parse_message(NEW_FORMAT_SIGNAL, author="Captain Hook")
        assert isinstance(result, TradeSignal)

    def test_empty_returns_none(self):
        assert parse_message("") is None

    def test_whitespace_only(self):
        assert parse_message("   \n\n  ") is None

    def test_stand_down_not_parsed_as_signal(self):
        text = "🌔 STAND DOWN MODE\nNo high-edge setups."
        result = parse_message(text)
        assert result is None


# ---------------------------------------------------------------------------
# Performance edge cases
# ---------------------------------------------------------------------------


class TestPerformanceEdge:
    def test_no_all_time_section(self):
        text = """📊 DAILY PERFORMANCE SUMMARY
Today
3W / 0L (100%) | Avg PnL: 1.50%
Trades
✅ SPY bearish | Score: 90 | PnL: 1.50%
✅ QQQ bullish | Score: 85 | PnL: 1.20%
✅ AAPL bearish | Score: 88 | PnL: 1.80%"""
        perf = parse_performance(text)
        assert perf is not None
        assert perf.wins == 3
        assert perf.losses == 0
        assert perf.all_time_wins is None

    def test_all_losses(self):
        text = """📊 DAILY PERFORMANCE SUMMARY
Today
0W / 3L (0%) | Avg PnL: -1.50%
Trades
❌ SPY bearish | Score: 80 | PnL: -2.00%
❌ QQQ bullish | Score: 75 | PnL: -1.50%
❌ AAPL bearish | Score: 78 | PnL: -1.00%"""
        perf = parse_performance(text)
        assert perf is not None
        assert perf.wins == 0
        assert perf.losses == 3
        assert perf.avg_pnl_pct == -1.50


# ---------------------------------------------------------------------------
# Watchlist edge cases
# ---------------------------------------------------------------------------


class TestWatchlistEdge:
    def test_single_entry(self):
        text = """🚀 Catalyst Sentinel 🛸
Active Watchlist (1):
TSLA: Stage 2 (bullish, score 90)"""
        entries = parse_watchlist(text)
        assert len(entries) == 1
        assert entries[0].ticker == "TSLA"
        assert entries[0].sentiment == Sentiment.BULLISH
        assert entries[0].score == 90

    def test_empty_watchlist_text(self):
        text = """🚀 Catalyst Sentinel 🛸
Active Watchlist (0):
No catalysts detected."""
        entries = parse_watchlist(text)
        assert entries == []
