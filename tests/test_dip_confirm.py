"""Tests for dip-confirm entry feature."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, init_paper_db
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
        "MAX_POSITION_PCT": 15.0,
        "MAX_CONCURRENT": 4,
        "MIN_SCORE": 78,
        "DAILY_LOSS_LIMIT_PCT": 10.0,
        "ENABLE_RISK_MANAGER": False,
        "SIMULATED_ENTRY_SLIPPAGE_BPS": 0.0,
        "SIMULATED_EXIT_SLIPPAGE_BPS": 0.0,
        "ENABLE_DCA": False,
        "ENABLE_VINNY_STRATEGY": False,
        "ENABLE_SCORE_SIZING": False,
        "ENABLE_SMART_ENTRY": False,
        "ENABLE_DIP_CONFIRM": True,
        "DIP_CONFIRM_MAX_POLLS": 3,
        "DIP_CONFIRM_POLL_SEC": 0.01,  # fast for tests
        "DIP_CONFIRM_FADE_PCT": 1.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


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
        expiry="2026-05-18",
        risk_reward=1.5,
        target_1=168.0,
        target_2=167.0,
        stop_price=171.0,
        atm_strike=170.0,
        atm_premium=2.00,
        otm_strike=167.5,
        otm_premium=0.46,
        bot_source=BotSource.CAPTAIN_HOOK,
        is_elite=True,
    )
    defaults.update(overrides)
    return TradeSignal(**defaults)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
async def trader(tmp_db):
    settings = _make_settings(tmp_db)
    pt = PaperTrader(settings)
    await init_paper_db(tmp_db)
    return pt


class TestDipConfirmDisabled:
    """When ENABLE_DIP_CONFIRM=False, feature is a no-op."""

    @pytest.mark.asyncio
    async def test_disabled_returns_immediately(self, tmp_db):
        settings = _make_settings(tmp_db, ENABLE_DIP_CONFIRM=False)
        pt = PaperTrader(settings)
        await init_paper_db(tmp_db)
        signal = _make_signal()
        confirmed, premium = await pt._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None


class TestDipConfirmNoStream:
    """When market_stream is None, feature falls through."""

    @pytest.mark.asyncio
    async def test_no_stream_enters_immediately(self, trader):
        signal = _make_signal()
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None


class TestDipConfirmStablePremium:
    """When premium is stable/rising, enter immediately."""

    @pytest.mark.asyncio
    async def test_stable_premium_enters_immediately(self, trader):
        stream = AsyncMock()
        # t1 = same as t0 → not fading
        stream.get_option_premium = AsyncMock(return_value=2.00)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        # Premium not cheaper, so None (use signal price)
        assert premium is None

    @pytest.mark.asyncio
    async def test_rising_premium_enters_immediately(self, trader):
        stream = AsyncMock()
        # t1 higher than t0 → rising
        stream.get_option_premium = AsyncMock(return_value=2.10)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_slight_dip_below_threshold_enters(self, trader):
        stream = AsyncMock()
        # t1 dipped 0.5% < 1% threshold → not fading enough
        stream.get_option_premium = AsyncMock(return_value=1.99)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        # Slightly cheaper, so returns the cheaper price
        assert premium == 1.99


class TestDipConfirmFadingWithUptick:
    """When premium fades then upticks, enter at the cheaper price."""

    @pytest.mark.asyncio
    async def test_uptick_after_fade(self, trader):
        stream = AsyncMock()
        # t1 check: fading (1.90 = -5% from 2.00)
        # poll 1: 1.85 (still dropping, prev=1.90)
        # poll 2: 1.87 (uptick! 1.87 > 1.85)
        premiums = [1.90, 1.85, 1.87]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium == 1.87

    @pytest.mark.asyncio
    async def test_immediate_uptick_poll1(self, trader):
        stream = AsyncMock()
        # t1 check: fading (1.95 = -2.5%)
        # poll 1: 1.96 (uptick! 1.96 > 1.95)
        premiums = [1.95, 1.96]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium == 1.96


class TestDipConfirmFadingNoUptick:
    """When premium fades continuously with no uptick, skip the trade."""

    @pytest.mark.asyncio
    async def test_continuous_fade_skips(self, trader):
        stream = AsyncMock()
        # t1: 1.90 (fading -5%)
        # poll 1: 1.85, poll 2: 1.80, poll 3: 1.75 — never upticks
        premiums = [1.90, 1.85, 1.80, 1.75]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is False
        assert premium is None

    @pytest.mark.asyncio
    async def test_flat_fade_skips(self, trader):
        stream = AsyncMock()
        # t1: 1.90 (fading -5%)
        # polls: all 1.90 — flat, no uptick (not > prev)
        premiums = [1.90, 1.90, 1.90, 1.90]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is False
        assert premium is None


class TestDipConfirmEdgeCases:
    """Edge cases: no WS data, subscribe failure, no expiry."""

    @pytest.mark.asyncio
    async def test_no_ws_premium_enters_immediately(self, trader):
        stream = AsyncMock()
        stream.get_option_premium = AsyncMock(return_value=None)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_subscribe_failure_enters_immediately(self, trader):
        stream = AsyncMock()
        stream.subscribe_option = AsyncMock(side_effect=Exception("WS down"))
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_no_expiry_date_enters_immediately(self, trader):
        stream = AsyncMock()
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00, expiry="weird_format")
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_zero_premium_enters_immediately(self, trader):
        stream = AsyncMock()
        trader.market_stream = stream

        signal = _make_signal(atm_premium=0)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_poll_error_falls_through(self, trader):
        stream = AsyncMock()
        # First get_option_premium works (fading), then errors
        stream.get_option_premium = AsyncMock(
            side_effect=[1.90, Exception("network error")]
        )
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        # Error during polling → fall through to immediate entry
        assert confirmed is True
        assert premium is None

    @pytest.mark.asyncio
    async def test_unsubscribe_called_on_completion(self, trader):
        stream = AsyncMock()
        stream.get_option_premium = AsyncMock(return_value=2.00)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        await trader._wait_for_entry_confirmation(signal)

        stream.unsubscribe_option.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsubscribe_called_on_skip(self, trader):
        stream = AsyncMock()
        premiums = [1.90, 1.85, 1.80, 1.75]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        signal = _make_signal(atm_premium=2.00)
        await trader._wait_for_entry_confirmation(signal)

        stream.unsubscribe_option.assert_called_once()


class TestDipConfirmCallDirection:
    """Verify it works for CALL signals too."""

    @pytest.mark.asyncio
    async def test_call_direction(self, trader):
        stream = AsyncMock()
        stream.get_option_premium = AsyncMock(return_value=2.00)
        trader.market_stream = stream

        signal = _make_signal(direction=Direction.CALL, atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)
        assert confirmed is True

        # Verify subscribe was called with "call"
        stream.subscribe_option.assert_called_once_with(
            "NVDA", 170.0, "2026-05-18", "call",
        )


class TestDipConfirmSupportAware:
    """Test support/VWAP-aware logic in the smart dip-confirm."""

    @pytest.mark.asyncio
    async def test_above_vwap_call_enters_despite_fade(self, trader):
        """Call + fading + above VWAP → enter immediately (premium decay, not trend)."""
        stream = AsyncMock()
        # t1 = 1.90 → fading -5% from 2.00
        stream.get_option_premium = AsyncMock(return_value=1.90)
        trader.market_stream = stream

        # Mock candle cache: underlying above VWAP
        mock_cache = AsyncMock()
        mock_bars = [MagicMock(close=150.0, high=151.0, low=149.0, volume=1000)]
        mock_bars = mock_bars * 10
        mock_cache.get_candle_data = AsyncMock(return_value={
            "5m": mock_bars,
        })
        trader._candle_cache = mock_cache

        # Patch _check_support_level to return above_vwap=True
        with patch.object(trader, "_check_support_level",
                          return_value=(False, True, "price=$150 vwap=$148")):
            signal = _make_signal(direction=Direction.CALL, atm_premium=2.00)
            confirmed, premium = await trader._wait_for_entry_confirmation(signal)

        assert confirmed is True
        assert premium == 1.90  # entered at the cheaper faded price

    @pytest.mark.asyncio
    async def test_below_vwap_put_enters_despite_fade(self, trader):
        """Put + fading + below VWAP → enter (bearish structure is correct for puts)."""
        stream = AsyncMock()
        stream.get_option_premium = AsyncMock(return_value=1.90)
        trader.market_stream = stream

        with patch.object(trader, "_check_support_level",
                          return_value=(False, False, "price=$148 vwap=$150")):
            signal = _make_signal(direction=Direction.PUT, atm_premium=2.00)
            confirmed, premium = await trader._wait_for_entry_confirmation(signal)

        assert confirmed is True
        assert premium == 1.90

    @pytest.mark.asyncio
    async def test_below_vwap_call_fading_blocked(self, trader):
        """Call + fading + below VWAP → VWAP direction block (counter-trend)."""
        stream = AsyncMock()
        # t1: 1.90 (fading -5%)
        premiums = [1.90]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        with patch.object(trader, "_check_support_level",
                          return_value=(True, False, "price=$148 vwap=$150")):
            signal = _make_signal(direction=Direction.CALL, atm_premium=2.00)
            confirmed, premium = await trader._wait_for_entry_confirmation(signal)

        # Call below VWAP = counter-trend → blocked
        assert confirmed is False
        assert premium is None

    @pytest.mark.asyncio
    async def test_above_vwap_put_fading_blocked(self, trader):
        """Put + fading + above VWAP → VWAP direction block (counter-trend)."""
        stream = AsyncMock()
        # t1: 1.90 (fading -5%)
        premiums = [1.90]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream

        with patch.object(trader, "_check_support_level",
                          return_value=(False, True, "price=$152 vwap=$150")):
            signal = _make_signal(direction=Direction.PUT, atm_premium=2.00)
            confirmed, premium = await trader._wait_for_entry_confirmation(signal)

        # Put above VWAP = counter-trend → blocked
        assert confirmed is False
        assert premium is None

    @pytest.mark.asyncio
    async def test_no_candle_data_falls_through_to_polling(self, trader):
        """No candle cache → falls through to timer-based polling."""
        stream = AsyncMock()
        # Fading, then uptick
        premiums = [1.90, 1.85, 1.88]
        stream.get_option_premium = AsyncMock(side_effect=premiums)
        trader.market_stream = stream
        # No _candle_cache attribute → _check_support_level returns None

        signal = _make_signal(direction=Direction.CALL, atm_premium=2.00)
        confirmed, premium = await trader._wait_for_entry_confirmation(signal)

        assert confirmed is True
        assert premium == 1.88

    @pytest.mark.asyncio
    async def test_check_support_level_no_cache(self, trader):
        """_check_support_level returns None when no candle cache."""
        result = await trader._check_support_level("SPY", "call")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_support_level_with_bars(self, trader):
        """_check_support_level computes support/VWAP from candle bars."""
        mock_cache = AsyncMock()
        bars = []
        for i in range(10):
            bar = MagicMock()
            bar.close = 150.0 + i * 0.1  # rising from 150.0 to 150.9
            bar.high = bar.close + 0.5
            bar.low = bar.close - 0.3
            bar.volume = 1000
            bars.append(bar)

        mock_cache.get_candle_data = AsyncMock(return_value={"5m": bars})
        trader._candle_cache = mock_cache

        result = await trader._check_support_level("SPY", "call")
        assert result is not None
        at_support, above_vwap, detail = result
        assert isinstance(at_support, bool)
        assert isinstance(above_vwap, bool)
        assert "price=" in detail
        assert "vwap=" in detail


class TestSettingsDefaults:
    """Verify settings have correct defaults."""

    def test_dip_confirm_defaults(self):
        s = Settings(DISCORD_TOKEN="fake")
        assert s.ENABLE_DIP_CONFIRM is False
        assert s.DIP_CONFIRM_MAX_POLLS == 6
        assert s.DIP_CONFIRM_POLL_SEC == 5.0
        assert s.DIP_CONFIRM_FADE_PCT == 1.0


class TestVWAPDirectionBlockSafety:
    """Source code safety: verify VWAP direction block exists and is correct."""

    def test_vwap_block_put_above_vwap_exists(self):
        """paper_trader must block puts when above VWAP (counter-trend)."""
        import inspect
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader._wait_for_entry_confirmation)
        assert 'above_vwap and option_type == "put"' in source, (
            "Missing VWAP block for puts above VWAP — this was the GOOGL bug"
        )
        # Must return False (block), not True (enter)
        assert "return False, None" in source

    def test_vwap_block_call_below_vwap_exists(self):
        """paper_trader must block calls when below VWAP (counter-trend)."""
        import inspect
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader._wait_for_entry_confirmation)
        assert 'not above_vwap and option_type == "call"' in source, (
            "Missing VWAP block for calls below VWAP — counter-trend entry"
        )

    def test_vwap_block_comes_before_uptick_polling(self):
        """VWAP direction block must come BEFORE the uptick polling loop."""
        import inspect
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader._wait_for_entry_confirmation)
        # The VWAP block for puts should appear before "polling for uptick"
        put_block_pos = source.find('above_vwap and option_type == "put"')
        poll_pos = source.find("polling for uptick")
        assert put_block_pos < poll_pos, (
            "VWAP block must come BEFORE uptick polling — "
            "otherwise counter-trend trades bypass the block"
        )
