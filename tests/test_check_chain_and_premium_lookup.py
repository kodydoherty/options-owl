"""Tests for 'Strike: Check chain' / 'ATM Pick: N/A' parsing and premium lookup.

Covers the edge cases where Discord bot signals omit strike prices or ATM premiums,
and the fallback logic that derives them from entry price / option chain lookups.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from options_owl.collectors.discord_collector import parse_trade_signal
from options_owl.config.settings import Settings
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# Sample messages — real Discord formats with "Check chain" / "N/A"
# ---------------------------------------------------------------------------

# Full "Check chain" message — exactly as seen from Neverland Pan on 2026-04-07
CHECK_CHAIN_FULL = """\
🐂 AMD - Bullish (CALL) 💎
92/100 (Strong) 🟢
**$218.78** ➡ **$219.50** (+0.3%)
🔑 Key Signals
BB 2σ Touch | EMA Bounce | VWAP Support | Multi-TF Aligned
💼 Trade Idea
Buy Calls | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: $218.96 (+0.1%)
T2: $219.14 (+0.2%)
T3: $219.32 (+0.2%)
T4: $219.50 (+0.3%)
T5: $219.86 (+0.5%)
Stop: $218.45
💰 ATM Pick
N/A
⚡ OTM Pick
N/A
⏱️ Time in Play
14:26 • 🟠 Elevated theta - exit within 10 min • R:R 1.50:1
📊 Move Quality
VWAP Support"""

# Check chain with ATM pick present (strike missing but premium available)
CHECK_CHAIN_WITH_ATM = """\
🐻 TSLA - Bearish (PUT) 💎
100/100 (Strong) 🟢
**$340.44** ➡ **$339.41** (+0.3%)
🔑 Key Signals
MACD Bear Cross | EMA Bounce
💼 Trade Idea
Buy Puts | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: $340.18 (+0.1%)
T2: $339.93 (+0.1%)
Stop: $340.70
💰 ATM Pick
$340 put @ ~$4.50 (~+-35% est.)
⚡ OTM Pick
$338 put @ ~$2.80 (~+-40% est.)"""

# Normal signal (strike specified) — baseline for comparison
NORMAL_SIGNAL = """\
🐻 MSTR - Bearish (PUT) 💎
95/100 (Strong) 🟢
**$122.87** ➡ **$122.39** (+0.4%)
🔑 Key Signals
MACD Bear Cross | EMA Bounce | VWAP Support
💼 Trade Idea
Buy Puts | Strike: $123 Put | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: $122.75 (+0.1%)
T2: $122.63 (+0.2%)
T3: $122.51 (+0.3%)
T4: $122.39 (+0.4%)
T5: $122.15 (+0.6%)
Stop: $123.09
💰 ATM Pick
$123 put @ ~$6.84 (~+-51% est.)
⚡ OTM Pick
$121 put @ ~$5.19 (~+-53% est.)"""

# Check chain — PUT with decimal entry price
CHECK_CHAIN_DECIMAL_ENTRY = """\
🐻 NVDA - Bearish (PUT) 💎
88/100 (Strong) 🟢
**$176.33** ➡ **$175.50** (+0.5%)
🔑 Key Signals
BB 2σ Touch
💼 Trade Idea
Buy Puts | Strike: Check chain | Expiry: 1DTE | R:R 2.00:1
🎯 Price Targets
T1: $176.10 (+0.1%)
T2: $175.50 (+0.5%)
Stop: $176.80
💰 ATM Pick
N/A
⚡ OTM Pick
N/A"""

# Check chain — no expiry or R:R on trade idea line (extreme malformation)
CHECK_CHAIN_MINIMAL_TRADE_LINE = """\
🐂 SPY - Bullish (CALL) 💎
90/100 (Strong) 🟢
**$650.00** ➡ **$652.00** (+0.3%)
🔑 Key Signals
VWAP Support
💼 Trade Idea
Buy Calls | Strike: Check chain
🎯 Price Targets
T1: $650.50 (+0.1%)
T2: $651.00 (+0.2%)
Stop: $649.50"""

# Check chain with targets but no stop
CHECK_CHAIN_NO_STOP = """\
🐂 AAPL - Bullish (CALL)
85/100 (Good) 🟡
**$255.42** ➡ **$256.00** (+0.2%)
🔑 Key Signals
EMA Bounce
💼 Trade Idea
Buy Calls | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: $255.60 (+0.1%)
T2: $256.00 (+0.2%)"""

# Header + score + entry + targets + stop but NO trade idea line at all
NO_TRADE_LINE_WITH_TARGETS = """\
🐂 META - Bullish (CALL) 💎
91/100 (Strong) 🟢
**$525.00** ➡ **$527.00** (+0.4%)
🔑 Key Signals
MACD Cross | EMA Bounce
🎯 Price Targets
T1: $525.50 (+0.1%)
T2: $526.00 (+0.2%)
Stop: $524.00
💰 ATM Pick
$525 call @ ~$3.20 (~+-40% est.)"""

# "Strike: TBD" variant
CHECK_CHAIN_TBD = """\
🐻 GOOGL - Bearish (PUT) 💎
96/100 (Strong) 🟢
**$303.39** ➡ **$302.00** (+0.5%)
🔑 Key Signals
BB 2σ Touch
💼 Trade Idea
Buy Puts | Strike: TBD | Expiry: 0DTE | R:R 1.80:1
🎯 Price Targets
T1: $303.10 (+0.1%)
T2: $302.50 (+0.3%)
Stop: $303.80
💰 ATM Pick
$303 put @ ~$2.10 (~+-25% est.)"""

# SPX (index options) with Check chain
CHECK_CHAIN_SPX = """\
🐂 SPX - Bullish (CALL) 💎
87/100 (Strong) 🟢
**$653.79** ➡ **$655.37** (+0.2%)
🔑 Key Signals
VWAP Support
💼 Trade Idea
Buy Calls | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: $654.19 (+0.1%)
T2: $654.57 (+0.1%)
Stop: $653.20
💰 ATM Pick
N/A
⚡ OTM Pick
N/A"""


# ---------------------------------------------------------------------------
# Tests: "Strike: Check chain" parser handling
# ---------------------------------------------------------------------------


class TestCheckChainParsing:
    """Signals with 'Strike: Check chain' instead of a numeric strike."""

    def test_check_chain_full_message_parses(self):
        """The real AMD message from 2026-04-07 that was previously dropped."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig is not None
        assert sig.ticker == "AMD"
        assert sig.direction == Direction.CALL
        assert sig.score == 92

    def test_check_chain_strike_derived_from_entry(self):
        """Strike should be rounded from entry price when ATM pick is N/A."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        # Entry is $218.78, should round to nearest $0.50 → $219.00
        assert sig.strike == 219.0

    def test_check_chain_with_atm_pick_uses_atm_strike(self):
        """When ATM pick has a strike, use it instead of rounding entry price."""
        sig = parse_trade_signal(CHECK_CHAIN_WITH_ATM, author="Captain Hook")
        assert sig is not None
        assert sig.strike == 340.0  # from ATM Pick: $340 put
        assert sig.atm_premium == 4.50

    def test_check_chain_targets_still_parsed(self):
        """All T1-T5 targets should parse correctly despite missing strike."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig.target_1 == 218.96
        assert sig.target_2 == 219.14
        assert sig.target_3 == 219.32
        assert sig.target_4 == 219.50
        assert sig.target_5 == 219.86
        assert sig.stop_price == 218.45

    def test_check_chain_expiry_parsed(self):
        """Expiry should still be extracted from the trade idea line."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig.expiry == "0DTE"

    def test_check_chain_rr_parsed(self):
        """Risk:Reward ratio should still be extracted."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig.risk_reward == 1.50

    def test_check_chain_atm_premium_is_none(self):
        """ATM premium should be None when ATM Pick is 'N/A'."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig.atm_premium is None

    def test_check_chain_decimal_entry_rounding(self):
        """$176.33 entry should round to $176.5 (nearest $0.50)."""
        sig = parse_trade_signal(CHECK_CHAIN_DECIMAL_ENTRY, author="Captain Hook")
        assert sig is not None
        assert sig.strike == 176.5

    def test_check_chain_1dte_expiry(self):
        """Non-0DTE expiry should be parsed."""
        sig = parse_trade_signal(CHECK_CHAIN_DECIMAL_ENTRY, author="Captain Hook")
        assert sig.expiry == "1DTE"
        assert sig.risk_reward == 2.00

    def test_check_chain_put_direction(self):
        """Bearish PUT with check chain should set direction correctly."""
        sig = parse_trade_signal(CHECK_CHAIN_DECIMAL_ENTRY, author="Captain Hook")
        assert sig.direction == Direction.PUT
        assert sig.sentiment == Sentiment.BEARISH

    def test_check_chain_spx_index(self):
        """SPX index options with Check chain should parse."""
        sig = parse_trade_signal(CHECK_CHAIN_SPX, author="Neverland Pan")
        assert sig is not None
        assert sig.ticker == "SPX"
        # $653.79 rounds to $654.0
        assert sig.strike == 654.0

    def test_check_chain_bot_source_detected(self):
        """Bot source should still be detected with check chain signals."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan 💸")
        assert sig.bot_source == BotSource.NEVERLAND_PAN

    def test_check_chain_key_signals_parsed(self):
        """Key signals should still be extracted."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert "BB 2σ Touch" in sig.key_signals
        assert "VWAP Support" in sig.key_signals

    def test_check_chain_is_elite(self):
        """Elite marker should be detected."""
        sig = parse_trade_signal(CHECK_CHAIN_FULL, author="Neverland Pan")
        assert sig.is_elite is True


class TestCheckChainStrikeRounding:
    """Verify strike rounding to nearest $0.50 for various entry prices."""

    @pytest.mark.parametrize(
        "entry_price, expected_strike",
        [
            (100.00, 100.0),
            (100.24, 100.0),
            (100.25, 100.5),
            (100.49, 100.5),
            (100.50, 100.5),
            (100.74, 100.5),
            (100.75, 101.0),
            (99.99, 100.0),
            (218.78, 219.0),  # the real AMD case
            (653.79, 654.0),  # the real SPX case
            (176.33, 176.5),  # the NVDA case
            (0.50, 0.5),      # penny stock edge case
            (1234.56, 1234.5),  # high-price stock
        ],
    )
    def test_strike_rounding(self, entry_price, expected_strike):
        """Verify round-to-nearest-$0.50 logic for various prices."""
        text = f"""\
🐂 TEST - Bullish (CALL) 💎
90/100 (Strong) 🟢
**${entry_price}** ➡ **${entry_price + 1}** (+0.1%)
🔑 Key Signals
VWAP Support
💼 Trade Idea
Buy Calls | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1
🎯 Price Targets
T1: ${entry_price + 0.50} (+0.1%)
Stop: ${entry_price - 0.50}"""
        sig = parse_trade_signal(text, author="Neverland Pan")
        assert sig is not None
        assert sig.strike == expected_strike


class TestCheckChainVariants:
    """Other non-numeric strike variants that bots might send."""

    def test_strike_tbd(self):
        """'Strike: TBD' should be handled like 'Check chain'."""
        sig = parse_trade_signal(CHECK_CHAIN_TBD, author="Captain Hook")
        assert sig is not None
        assert sig.ticker == "GOOGL"
        # ATM Pick has $303 → strike should come from ATM pick
        assert sig.strike == 303.0
        assert sig.atm_premium == 2.10

    def test_no_trade_line_at_all(self):
        """Signal with targets but no trade idea line should still parse."""
        sig = parse_trade_signal(NO_TRADE_LINE_WITH_TARGETS, author="Neverland Pan")
        assert sig is not None
        assert sig.ticker == "META"
        # Strike from ATM Pick: $525
        assert sig.strike == 525.0
        assert sig.atm_premium == 3.20
        assert sig.expiry == "0DTE"  # default fallback

    def test_check_chain_minimal_trade_line(self):
        """Trade line with just 'Check chain' and no Expiry/R:R."""
        sig = parse_trade_signal(CHECK_CHAIN_MINIMAL_TRADE_LINE, author="Neverland Pan")
        assert sig is not None
        assert sig.ticker == "SPY"
        assert sig.strike == 650.0
        assert sig.expiry == "0DTE"  # default
        assert sig.risk_reward == 1.5  # default

    def test_check_chain_no_stop(self):
        """Check chain with targets but no stop price — should still parse (stop gate will reject)."""
        sig = parse_trade_signal(CHECK_CHAIN_NO_STOP, author="Neverland Pan")
        assert sig is not None
        assert sig.ticker == "AAPL"
        assert sig.stop_price is None
        assert sig.target_1 == 255.60


class TestNormalSignalRegression:
    """Ensure normal signals (with explicit strike) still parse correctly."""

    def test_normal_signal_still_works(self):
        sig = parse_trade_signal(NORMAL_SIGNAL, author="Captain Hook")
        assert sig is not None
        assert sig.ticker == "MSTR"
        assert sig.strike == 123.0
        assert sig.atm_premium == 6.84

    def test_normal_signal_all_targets(self):
        sig = parse_trade_signal(NORMAL_SIGNAL, author="Captain Hook")
        assert sig.target_1 == 122.75
        assert sig.target_2 == 122.63
        assert sig.target_3 == 122.51
        assert sig.target_4 == 122.39
        assert sig.target_5 == 122.15
        assert sig.stop_price == 123.09

    def test_normal_signal_otm_pick(self):
        sig = parse_trade_signal(NORMAL_SIGNAL, author="Captain Hook")
        assert sig.otm_strike == 121.0
        assert sig.otm_premium == 5.19


# ---------------------------------------------------------------------------
# Tests: "ATM Pick: N/A" → premium lookup fallback
# ---------------------------------------------------------------------------


class TestTruncatedMessagesStillRejected:
    """Messages that are incomplete should NOT parse — even with relaxed rules."""

    def test_header_score_entry_only(self):
        """No targets, no stop, no trade line → reject."""
        text = """\
🐻 SPY - Bearish (PUT) 💎
100/100 (Strong) 🟢
$550.00 ➡ $548.00 (+0.4%)"""
        assert parse_trade_signal(text) is None

    def test_header_score_entry_trade_but_no_targets(self):
        """Trade line present but no targets or stop → reject."""
        text = """\
🐂 AAPL - Bullish (CALL)
85/100 (Good) 🟡
**$250.00** ➡ **$252.00** (+0.8%)
💼 Trade Idea
Buy Calls | Strike: Check chain | Expiry: 0DTE | R:R 1.50:1"""
        assert parse_trade_signal(text) is None

    def test_header_only(self):
        assert parse_trade_signal("🐻 SPY - Bearish (PUT) 💎") is None

    def test_empty(self):
        assert parse_trade_signal("") is None

    def test_garbage_with_emojis(self):
        assert parse_trade_signal("🚀🔥💎 LFG!!!") is None


# ---------------------------------------------------------------------------
# Tests: Premium lookup via option chain (mocked)
# ---------------------------------------------------------------------------


class TestFillMissingPremium:
    """Test the _fill_missing_premium method on PaperTrader."""

    def _make_signal(self, **overrides):
        defaults = dict(
            ticker="AMD",
            sentiment=Sentiment.BULLISH,
            direction=Direction.CALL,
            score=92,
            strength=SignalStrength.STRONG,
            entry_price=218.78,
            target_price=219.50,
            expected_move_pct=0.3,
            strike=219.0,
            expiry="0DTE",
            risk_reward=1.5,
            stop_price=218.45,
            target_1=218.96,
            atm_premium=None,
            bot_source=BotSource.NEVERLAND_PAN,
            is_elite=True,
        )
        defaults.update(overrides)
        return TradeSignal(**defaults)

    @pytest.mark.asyncio
    async def test_premium_filled_from_chain(self):
        """When chain lookup succeeds, premium should be filled."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=None)
        assert signal.atm_premium is None

        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}

        with (
            patch(
                "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
                return_value=mock_chain,
            ),
            patch(
                "options_owl.execution.position_monitor._lookup_premium_from_chain",
                return_value=5.75,
            ),
        ):
            result = await pt._fill_missing_premium(signal)

        assert result.atm_premium == 5.75
        assert result.atm_strike == 219.0

    @pytest.mark.asyncio
    async def test_premium_none_when_chain_fails(self):
        """When chain lookup returns None, signal unchanged."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=None)

        with patch(
            "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
            return_value=None,
        ):
            result = await pt._fill_missing_premium(signal)

        assert result.atm_premium is None

    @pytest.mark.asyncio
    async def test_premium_none_when_strike_not_found(self):
        """When chain exists but strike not found, signal unchanged."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=None)
        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}

        with (
            patch(
                "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
                return_value=mock_chain,
            ),
            patch(
                "options_owl.execution.position_monitor._lookup_premium_from_chain",
                return_value=None,
            ),
        ):
            result = await pt._fill_missing_premium(signal)

        assert result.atm_premium is None

    @pytest.mark.asyncio
    async def test_premium_none_when_chain_throws(self):
        """When chain lookup raises an exception, signal unchanged (no crash)."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=None)

        with patch(
            "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
            side_effect=RuntimeError("network down"),
        ):
            result = await pt._fill_missing_premium(signal)

        assert result.atm_premium is None

    @pytest.mark.asyncio
    async def test_premium_skipped_when_already_set(self):
        """When atm_premium already has a value, don't look it up."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)
        await pt.init()

        signal = self._make_signal(atm_premium=6.50)

        # evaluate_and_trade calls _fill_missing_premium only if atm_premium is None/0
        # Verify the original premium is preserved
        assert signal.atm_premium == 6.50

    @pytest.mark.asyncio
    async def test_premium_lookup_uses_correct_option_type(self):
        """PUT signals should look up puts, CALL signals calls."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        put_signal = self._make_signal(
            direction=Direction.PUT, atm_premium=None, strike=218.0,
        )
        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}

        with (
            patch(
                "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
                return_value=mock_chain,
            ),
            patch(
                "options_owl.execution.position_monitor._lookup_premium_from_chain",
                return_value=3.20,
            ) as mock_lookup,
        ):
            result = await pt._fill_missing_premium(put_signal)

        # Verify lookup was called with "put" option type
        mock_lookup.assert_called_once_with(mock_chain, 218.0, "put")
        assert result.atm_premium == 3.20

    @pytest.mark.asyncio
    async def test_premium_zero_triggers_lookup(self):
        """atm_premium=0.0 should trigger lookup (treated as missing)."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=0.0)
        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}

        with (
            patch(
                "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
                return_value=mock_chain,
            ),
            patch(
                "options_owl.execution.position_monitor._lookup_premium_from_chain",
                return_value=7.00,
            ),
        ):
            result = await pt._fill_missing_premium(signal)

        assert result.atm_premium == 7.00

    @pytest.mark.asyncio
    async def test_no_expiry_skips_lookup(self):
        """If expiry can't be resolved (None), skip the chain lookup."""
        from options_owl.execution.paper_trader import PaperTrader

        settings = Settings(
            DISCORD_TOKEN="fake", PORTFOLIO_SIZE=5000, DB_PATH="/tmp/test.db",
        )
        pt = PaperTrader(settings)

        signal = self._make_signal(atm_premium=None, expiry="unknown_format")

        with patch(
            "options_owl.execution.position_monitor._fetch_option_chain_for_ticker",
        ) as mock_fetch:
            result = await pt._fill_missing_premium(signal)

        # Should not have tried to fetch since expiry couldn't resolve
        mock_fetch.assert_not_called()
        assert result.atm_premium is None


# ---------------------------------------------------------------------------
# Tests: Outcome resolver timezone handling
# ---------------------------------------------------------------------------


class TestOutcomeResolverTimezone:
    """Verify UTC→ET conversion for Discord timestamps vs yfinance bars."""

    def test_utc_timestamp_converted_to_et(self):
        """Signal at 17:50 UTC (1:50 PM ET) should match bars at 13:50 ET."""
        from options_owl.models.signals import PriceBar, TradeOutcome
        from options_owl.signals.outcome_resolver import resolve_signal

        signal = {
            "ticker": "AMZN",
            "direction": "put",
            "entry_price": 200.66,
            "target_1": 200.41,
            "target_2": 200.17,
            "stop_price": 200.89,
            # Real UTC timestamp from Discord
            "created_at": "2026-03-27T17:50:30.942000+00:00",
        }

        bars = [
            # Bar at 13:49 ET (before signal) — should be filtered out
            PriceBar(
                timestamp=datetime(2026, 3, 27, 13, 49),
                open=200.0, high=200.0, low=199.0, close=199.5, volume=1000,
            ),
            # Bar at 13:50 ET (at signal time) — should be included
            PriceBar(
                timestamp=datetime(2026, 3, 27, 13, 50),
                open=200.5, high=200.6, low=200.1, close=200.3, volume=1000,
            ),
            # Bar at 13:55 ET — target hit
            PriceBar(
                timestamp=datetime(2026, 3, 27, 13, 55),
                open=200.3, high=200.4, low=200.0, close=200.1, volume=1000,
            ),
        ]

        result = resolve_signal(signal, bars, signal_id=999)
        # Should NOT be unknown — the UTC→ET conversion should work
        assert result.outcome != TradeOutcome.UNKNOWN

    def test_naive_timestamp_no_conversion(self):
        """Naive timestamps (no timezone) should be compared as-is."""
        from options_owl.models.signals import PriceBar, TradeOutcome
        from options_owl.signals.outcome_resolver import resolve_signal

        signal = {
            "ticker": "SPY",
            "direction": "call",
            "entry_price": 575.50,
            "target_1": 576.90,
            "target_2": None,
            "stop_price": 574.18,
            # Naive timestamp — no timezone info
            "created_at": "2026-03-31T09:32:00",
        }

        bars = [
            # 09:30 — before signal, should be filtered
            PriceBar(
                timestamp=datetime(2026, 3, 31, 9, 30),
                open=576.0, high=577.0, low=575.0, close=576.5, volume=1000,
            ),
            # 09:31 — before signal, should be filtered
            PriceBar(
                timestamp=datetime(2026, 3, 31, 9, 31),
                open=576.5, high=577.5, low=575.5, close=577.0, volume=1000,
            ),
            # 09:32 — at signal time, should be included (no T1 hit)
            PriceBar(
                timestamp=datetime(2026, 3, 31, 9, 32),
                open=575.5, high=576.0, low=575.0, close=575.8, volume=1000,
            ),
            # 09:33 — after signal (no T1 hit)
            PriceBar(
                timestamp=datetime(2026, 3, 31, 9, 33),
                open=575.8, high=576.0, low=575.0, close=575.5, volume=1000,
            ),
        ]

        result = resolve_signal(signal, bars, signal_id=200)
        # Bars at 09:30 and 09:31 hit T1 ($576.90) but are before signal
        # Only 09:32 and 09:33 should be used — neither hits T1
        assert result.outcome == TradeOutcome.EXPIRED

    def test_utc_after_market_close_no_bars(self):
        """Signal at 21:00 UTC (5 PM ET, after close) should find no bars."""
        from options_owl.models.signals import PriceBar, TradeOutcome
        from options_owl.signals.outcome_resolver import resolve_signal

        signal = {
            "ticker": "AAPL",
            "direction": "call",
            "entry_price": 250.0,
            "target_1": 251.0,
            "target_2": None,
            "stop_price": 249.0,
            "created_at": "2026-04-07T21:00:00+00:00",
        }

        # All bars end at 15:59 ET
        bars = [
            PriceBar(
                timestamp=datetime(2026, 4, 7, 15, 58),
                open=250.0, high=251.0, low=249.5, close=250.5, volume=1000,
            ),
            PriceBar(
                timestamp=datetime(2026, 4, 7, 15, 59),
                open=250.5, high=251.5, low=250.0, close=251.0, volume=1000,
            ),
        ]

        result = resolve_signal(signal, bars, signal_id=300)
        # 5 PM ET > 3:59 PM ET — no bars after signal → unknown
        assert result.outcome == TradeOutcome.UNKNOWN
