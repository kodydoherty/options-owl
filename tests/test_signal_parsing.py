"""Extended tests for signal parsing — complex formats, edge cases, expiry parsing."""

from __future__ import annotations

from datetime import datetime

from options_owl.collectors.discord_collector import (
    _detect_bot,
    parse_message,
    parse_trade_signal,
    parse_watchlist,
)
from options_owl.execution.paper_trader import resolve_expiry_date
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------


class TestBotDetection:
    def test_captain_hook(self):
        assert _detect_bot("Captain Hook 🗡") == BotSource.CAPTAIN_HOOK

    def test_neverland_pan(self):
        assert _detect_bot("Neverland Pan 💸") == BotSource.NEVERLAND_PAN

    def test_tinker(self):
        assert _detect_bot("Tinker 🛎") == BotSource.TINKER

    def test_smee(self):
        assert _detect_bot("Smee 📊") == BotSource.SMEE

    def test_rufio(self):
        assert _detect_bot("Rufio 🚀") == BotSource.RUFIO

    def test_unknown(self):
        assert _detect_bot("RandomBot") == BotSource.UNKNOWN

    def test_case_insensitive(self):
        assert _detect_bot("CAPTAIN HOOK") == BotSource.CAPTAIN_HOOK
        assert _detect_bot("neverland pan") == BotSource.NEVERLAND_PAN


# ---------------------------------------------------------------------------
# Expiry date parsing (resolve_expiry_date from paper_trader)
# ---------------------------------------------------------------------------


class TestExpiryDateParsing:
    def _today_et_str(self):
        from options_owl.execution.paper_trader import _today_et
        return _today_et().strftime("%Y-%m-%d")

    def test_0dte(self):
        result = resolve_expiry_date("0DTE")
        assert result == self._today_et_str()

    def test_1dte(self):
        result = resolve_expiry_date("1DTE")
        assert result is not None
        from options_owl.execution.paper_trader import _today_et
        from datetime import timedelta
        expected = (_today_et() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_already_a_date(self):
        result = resolve_expiry_date("2025-04-18")
        assert result == "2025-04-18"

    def test_none_input(self):
        assert resolve_expiry_date(None) is None

    def test_empty_string(self):
        assert resolve_expiry_date("") is None

    def test_unrecognized_format(self):
        assert resolve_expiry_date("next_friday") is None

    def test_case_insensitive_dte(self):
        result = resolve_expiry_date("0dte")
        assert result == self._today_et_str()

    def test_whitespace_handling(self):
        result = resolve_expiry_date("  0DTE  ")
        assert result == self._today_et_str()

    def test_today(self):
        result = resolve_expiry_date("today")
        assert result == self._today_et_str()

    def test_tomorrow(self):
        from options_owl.execution.paper_trader import _today_et
        from datetime import timedelta
        result = resolve_expiry_date("tomorrow")
        expected = (_today_et() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_friday(self):
        result = resolve_expiry_date("Friday")
        assert result is not None
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 4  # Should be a Friday
        # On a Friday, should return today (0DTE)
        from options_owl.execution.paper_trader import _today_et
        today = _today_et()
        if today.weekday() == 4:  # if today is Friday
            assert result == today.strftime("%Y-%m-%d")

    def test_same_day(self):
        result = resolve_expiry_date("same day")
        assert result == self._today_et_str()

    def test_intraday(self):
        result = resolve_expiry_date("intraday")
        assert result == self._today_et_str()


# ---------------------------------------------------------------------------
# Various signal message formats
# ---------------------------------------------------------------------------


class TestSignalFormats:
    def test_good_score(self):
        text = """🐂 AAPL - Bullish (CALL)
78/100 (Good) 🟡
$190.50 ➡ $192.00 (+0.8%)

🔑 Key Signals
MACD Cross | EMA Bounce

💼 Trade Idea
Buy Calls | Strike: $191 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $191.50 (+0.5%) | T2: $192.00 (+0.8%) | Stop: $189.50 (-0.5%)
Exit by 11:00

💰 ATM Pick
$191 call @ ~$1.20 (~+5000% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.ticker == "AAPL"
        assert sig.strength == SignalStrength.GOOD
        assert sig.score == 78
        assert sig.direction == Direction.CALL

    def test_solid_score(self):
        """'Solid' strength tier (new format from Discord bots)."""
        text = """🐻 TSLA - Bearish (PUT)
69/100 (Solid) 🟡 (raw 128)
**$369.55** ➡ **$365.85** (+1.0%)

🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support | Multi-TF Aligned

💼 Trade Idea
Buy Puts | Strike: $370 Put | Expiry: 0DTE | R:R 1.80:1

🎯 Price Targets
T1: $368.63 (+0.3%)
T2: $367.70 (+0.5%)
Stop: $370.99
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.ticker == "TSLA"
        assert sig.strength == SignalStrength.SOLID
        assert sig.score == 128  # raw score preferred
        assert sig.direction == Direction.PUT

    def test_bold_entry_target(self):
        """Bold markdown format: **$168.685** -> **$167.09** (+0.9%)."""
        text = """🐻 NVDA - Bearish (PUT) 💎
100/100 (Strong) 🟢
**$168.685** ➡ **$167.09** (+0.9%)

🔑 Key Signals
BB 2σ Touch

💼 Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $167.89 (+0.5%) | T2: $167.09 (+0.9%) | Stop: $169.43 (-0.5%)

💰 ATM Pick
$170 put @ ~$1.70 (~+3893% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.entry_price == 168.685
        assert sig.target_price == 167.09

    def test_no_otm_pick(self):
        """Signal with only ATM pick, no OTM."""
        text = """🐂 SPY - Bullish (CALL)
85/100 (Good) 🟡
$520.00 ➡ $523.00 (+0.6%)

🔑 Key Signals
EMA Bounce

💼 Trade Idea
Buy Calls | Strike: $520 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $521.50 (+0.3%) | T2: $523.00 (+0.6%) | Stop: $518.50 (-0.3%)

💰 ATM Pick
$520 call @ ~$2.50 (~+2000% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.atm_premium == 2.50
        assert sig.otm_premium is None
        assert sig.otm_strike is None

    def test_no_exit_by_time(self):
        """Signal without Exit by time."""
        text = """🐻 AMD - Bearish (PUT)
80/100 (Good) 🟡
$160.00 ➡ $158.00 (+1.3%)

🔑 Key Signals
VWAP Support

💼 Trade Idea
Buy Puts | Strike: $160 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $159.00 (+0.6%) | T2: $158.00 (+1.3%) | Stop: $161.00 (-0.6%)

💰 ATM Pick
$160 put @ ~$1.50 (~+4000% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.exit_by is None
        assert sig.stop_price == 161.0

    def test_elite_reversal_put_derives_bearish(self):
        """Elite Reversal (PUT) should derive bearish sentiment."""
        text = """💫 TSLA - Elite Reversal (PUT) 💎
100/100 (Strong) 🟢
$363.24 ➡ $362.05 (+0.3%)

🔑 Key Signals
BB 2σ Touch | MACD Bear Cross

💼 Trade Idea
Buy Puts | Strike: $362.5 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $362.64 (+0.2%) | T2: $362.05 (+0.3%) | Stop: $363.80 (-0.2%)

💰 ATM Pick
$362.5 put @ ~$0.78 (~+5812% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.sentiment == Sentiment.BEARISH
        assert sig.direction == Direction.PUT

    def test_raw_score_preferred_over_capped(self):
        """When (raw N) is present, use the uncapped raw score."""
        text = """🐂 SPY - Bullish (CALL)
100/100 (Strong) 🟢 (raw 164)
$575.00 ➡ $577.00 (+0.3%)

💼 Trade Idea
Buy Calls | Strike: $576 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $576.50 | T2: $577.00 | Stop: $574.00
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.score == 164  # raw, not capped 100

    def test_no_raw_score_uses_display(self):
        """When (raw N) is absent, use the display score."""
        text = """🐂 SPY - Bullish (CALL)
85/100 (Good) 🟡
$575.00 ➡ $577.00 (+0.3%)

💼 Trade Idea
Buy Calls | Strike: $576 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $576.50 | T2: $577.00 | Stop: $574.00
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.score == 85

    def test_message_without_score_returns_none(self):
        text = """🐂 MSFT - Bullish (CALL)
$380.00 ➡ $382.00 (+0.5%)

💼 Trade Idea
Buy Calls | Strike: $380 Call | Expiry: 0DTE | R:R 1.50:1
"""
        sig = parse_trade_signal(text)
        assert sig is None

    def test_message_without_header_returns_none(self):
        text = """100/100 (Strong) 🟢
$168.685 ➡ $167.09 (+0.9%)

💼 Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1
"""
        sig = parse_trade_signal(text)
        assert sig is None


# ---------------------------------------------------------------------------
# Parse multiple key signals
# ---------------------------------------------------------------------------


class TestKeySignals:
    def test_multiple_key_signals_parsed(self):
        text = """🐻 NVDA - Bearish (PUT) 💎
100/100 (Strong) 🟢
$168.685 ➡ $167.09 (+0.9%)

🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support | Multi-TF Aligned

💼 Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $167.89 (+0.5%) | T2: $167.09 (+0.9%) | Stop: $169.43 (-0.5%)

💰 ATM Pick
$170 put @ ~$1.70 (~+3893% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert len(sig.key_signals) == 4
        assert "BB 2σ Touch" in sig.key_signals
        assert "Multi-TF Aligned" in sig.key_signals


# ---------------------------------------------------------------------------
# Watchlist edge cases
# ---------------------------------------------------------------------------


class TestWatchlistEdgeCases:
    def test_watchlist_without_keyword(self):
        text = "AAPL: Stage 1 (bullish, score 90)"
        entries = parse_watchlist(text)
        assert entries == []

    def test_watchlist_with_catalyst_sentinel(self):
        text = """🚀 Catalyst Sentinel 🛸
AAPL: Stage 1 (bullish, score 90)
GOOG: Stage 2 (bearish, score 75)
"""
        entries = parse_watchlist(text)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# parse_message unified handler
# ---------------------------------------------------------------------------


class TestParseMessageUnified:
    def test_trade_signal_returns_trade_signal(self):
        text = """🐂 MSFT - Bullish (CALL)
88/100 (Good) 🟡
$410.00 ➡ $412.00 (+0.5%)

🔑 Key Signals
MACD Cross

💼 Trade Idea
Buy Calls | Strike: $410 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $411.00 (+0.2%) | T2: $412.00 (+0.5%) | Stop: $409.00 (-0.2%)

💰 ATM Pick
$410 call @ ~$2.00 (~+3000% est.)
"""
        result = parse_message(text)
        assert isinstance(result, TradeSignal)

    def test_garbage_returns_none(self):
        result = parse_message("lol nothing here")
        assert result is None

    def test_author_metadata_passed_through(self):
        text = """🐻 NVDA - Bearish (PUT)
90/100 (Strong) 🟢
$170.00 ➡ $168.00 (+1.2%)

🔑 Key Signals
BB 2σ Touch

💼 Trade Idea
Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $169.00 (+0.6%) | T2: $168.00 (+1.2%) | Stop: $171.00 (-0.6%)

💰 ATM Pick
$170 put @ ~$1.70 (~+3000% est.)
"""
        result = parse_message(
            text,
            message_id=42,
            channel="signals",
            author="Captain Hook 🗡",
        )
        assert isinstance(result, TradeSignal)
        assert result.source_message_id == 42
        assert result.source_channel == "signals"
        assert result.bot_source == BotSource.CAPTAIN_HOOK


# ---------------------------------------------------------------------------
# Strike price extraction formats
# ---------------------------------------------------------------------------


class TestStrikePriceFormats:
    def test_integer_strike(self):
        text = """🐂 SPY - Bullish (CALL)
85/100 (Good) 🟡
$520.00 ➡ $523.00 (+0.6%)

🔑 Key Signals
EMA Bounce

💼 Trade Idea
Buy Calls | Strike: $520 Call | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $521.50 (+0.3%) | T2: $523.00 (+0.6%) | Stop: $518.50 (-0.3%)

💰 ATM Pick
$520 call @ ~$2.50 (~+2000% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.strike == 520.0

    def test_half_strike(self):
        text = """🐻 TSLA - Bearish (PUT)
90/100 (Strong) 🟢
$363.24 ➡ $362.05 (+0.3%)

🔑 Key Signals
BB 2σ Touch

💼 Trade Idea
Buy Puts | Strike: $362.5 Put | Expiry: 0DTE | R:R 1.50:1

🎯 Exit Targets
T1: $362.64 (+0.2%) | T2: $362.05 (+0.3%) | Stop: $363.80 (-0.2%)

💰 ATM Pick
$362.5 put @ ~$0.78 (~+5812% est.)
"""
        sig = parse_trade_signal(text)
        assert sig is not None
        assert sig.strike == 362.5
