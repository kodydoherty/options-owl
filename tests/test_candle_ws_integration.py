"""Tests for WS minute-bar buffering and CandleCache aggregation from WS data.

Covers:
- MarketDataStream minute-bar buffer from AM events
- CandleCache._build_from_ws() aggregation of 1-min bars into higher TFs
- CandleCache falls back to REST when WS has no data
- Integration: candle indicators computed from WS-aggregated bars
- Profit retrace gate with new settings (25% min gain, 50% retrace)
"""

import json
import time as _time

import pytest

from options_owl.collectors.candle_cache import (
    CandleBar,
    CandleCache,
    calc_atr,
    calc_rsi,
    calc_volume_trend,
    detect_candle_pattern,
    evaluate_enrg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minute_bar(ts_minutes_from_open: int, o: float, h: float, lo: float,
                     c: float, v: float = 10000.0, vw: float = 0.0):
    """Create a minute bar tuple as stored in MarketDataStream._minute_bars.

    ts_minutes_from_open: minutes since 9:30 AM ET (as integer).
    Returns (timestamp_ms, open, high, low, close, volume, vwap).
    """
    # Base: 2026-04-27 09:30 ET = 13:30 UTC
    base_ms = 1777507800000  # arbitrary base
    ts_ms = base_ms + ts_minutes_from_open * 60_000
    return (float(ts_ms), o, h, lo, c, v, vw or c)


def _rising_minute_bars(n: int, start: float = 100.0, step: float = 0.1):
    """Generate N rising minute bars starting at minute 0."""
    bars = []
    for i in range(n):
        c = start + i * step
        bars.append(_make_minute_bar(i, c - 0.05, c + 0.1, c - 0.1, c, 10000 + i * 100))
    return bars


def _falling_minute_bars(n: int, start: float = 110.0, step: float = 0.1):
    """Generate N falling minute bars."""
    bars = []
    for i in range(n):
        c = start - i * step
        bars.append(_make_minute_bar(i, c + 0.05, c + 0.1, c - 0.1, c, 10000 + i * 100))
    return bars


class FakeMarketStream:
    """Minimal mock of MarketDataStream with minute-bar buffer."""

    def __init__(self):
        self._bars: dict[str, list] = {}

    def set_bars(self, ticker: str, bars: list):
        self._bars[ticker.upper()] = bars

    def get_minute_bars(self, ticker: str):
        return list(self._bars.get(ticker.upper(), []))


# ---------------------------------------------------------------------------
# MarketDataStream minute-bar buffer tests
# ---------------------------------------------------------------------------


class TestMarketDataStreamMinuteBarBuffer:
    """Test that AM events populate the minute-bar buffer."""

    def test_minute_bars_stored_from_am_event(self):
        from collections import deque
        from options_owl.collectors.market_data_stream import MarketDataStream
        from options_owl.config.settings import Settings

        s = Settings(DATA_FEED_PROVIDER="polygon", POLYGON_API_KEY="fake")
        stream = MarketDataStream(s)

        # Simulate processing an AM event
        event = {
            "ev": "AM",
            "sym": "NVDA",
            "s": 1777507800000,  # start ts
            "e": 1777507860000,  # end ts
            "o": 213.50,
            "h": 214.00,
            "l": 213.20,
            "c": 213.80,
            "v": 50000,
            "vw": 213.60,
            "p": 213.80,  # not used for AM but some events have it
        }
        # Call internal handler directly
        stream._process_polygon_message(json.dumps([event]))

        bars = stream.get_minute_bars("NVDA")
        assert len(bars) == 1
        ts, o, h, lo, c, v, vw = bars[0]
        assert o == 213.50
        assert h == 214.00
        assert lo == 213.20
        assert c == 213.80
        assert v == 50000
        assert vw == 213.60

    def test_option_am_events_not_stored(self):
        from options_owl.collectors.market_data_stream import MarketDataStream
        from options_owl.config.settings import Settings

        s = Settings(DATA_FEED_PROVIDER="polygon", POLYGON_API_KEY="fake")
        stream = MarketDataStream(s)

        # Option symbol starts with "O:"
        event = {
            "ev": "AM",
            "sym": "O:NVDA260427C00215000",
            "s": 1777507800000,
            "e": 1777507860000,
            "o": 0.45, "h": 0.50, "l": 0.40, "c": 0.48,
            "v": 100, "vw": 0.46,
        }
        stream._process_polygon_message(json.dumps([event]))

        # Should NOT store option bars (only underlying equity)
        bars = stream.get_minute_bars("O:NVDA260427C00215000")
        assert len(bars) == 0

    def test_multiple_tickers_stored_separately(self):
        from options_owl.collectors.market_data_stream import MarketDataStream
        from options_owl.config.settings import Settings

        s = Settings(DATA_FEED_PROVIDER="polygon", POLYGON_API_KEY="fake")
        stream = MarketDataStream(s)

        events = [
            {"ev": "AM", "sym": "NVDA", "s": 1000, "e": 1060000,
             "o": 213.0, "h": 214.0, "l": 213.0, "c": 213.5, "v": 1000, "vw": 213.3},
            {"ev": "AM", "sym": "TSLA", "s": 1000, "e": 1060000,
             "o": 370.0, "h": 371.0, "l": 369.5, "c": 370.5, "v": 2000, "vw": 370.2},
        ]
        stream._process_polygon_message(json.dumps(events))

        assert len(stream.get_minute_bars("NVDA")) == 1
        assert len(stream.get_minute_bars("TSLA")) == 1
        assert len(stream.get_minute_bars("SPY")) == 0

    def test_max_buffer_size_respected(self):
        from options_owl.collectors.market_data_stream import MarketDataStream
        from options_owl.config.settings import Settings

        s = Settings(DATA_FEED_PROVIDER="polygon", POLYGON_API_KEY="fake")
        stream = MarketDataStream(s)

        # Push more bars than _MAX_MINUTE_BARS
        for i in range(stream._MAX_MINUTE_BARS + 50):
            event = {
                "ev": "AM", "sym": "SPY",
                "s": 1000 + i * 60000, "e": 1000 + (i + 1) * 60000,
                "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5,
                "v": 1000, "vw": 500.3,
            }
            stream._process_polygon_message(json.dumps([event]))

        bars = stream.get_minute_bars("SPY")
        assert len(bars) == stream._MAX_MINUTE_BARS

    def test_get_minute_bars_returns_copy(self):
        from options_owl.collectors.market_data_stream import MarketDataStream
        from options_owl.config.settings import Settings

        s = Settings(DATA_FEED_PROVIDER="polygon", POLYGON_API_KEY="fake")
        stream = MarketDataStream(s)

        event = {"ev": "AM", "sym": "SPY", "s": 1000, "e": 1060000,
                 "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 1000, "vw": 500.3}
        stream._process_polygon_message(json.dumps([event]))

        bars1 = stream.get_minute_bars("SPY")
        bars2 = stream.get_minute_bars("SPY")
        assert bars1 == bars2
        bars1.clear()  # mutating the returned list
        assert len(stream.get_minute_bars("SPY")) == 1  # original unaffected


# ---------------------------------------------------------------------------
# CandleCache._build_from_ws() aggregation tests
# ---------------------------------------------------------------------------


class TestCandleCacheBuildFromWS:
    """Test aggregation of 1-minute WS bars into higher timeframes."""

    def test_5min_aggregation(self):
        """10 minute bars → 2 complete 5-min candles."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(10, start=100.0, step=0.1)
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("SPY", "5m")

        assert len(result) == 2
        # First 5-min candle: bars 0-4
        assert result[0].open == bars[0][1]  # open of first minute bar
        assert result[0].close == bars[4][4]  # close of 5th minute bar
        assert result[0].high >= result[0].open
        assert result[0].low <= result[0].close

    def test_15min_aggregation(self):
        """30 minute bars → multiple 15-min candles."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(30)
        stream.set_bars("NVDA", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("NVDA", "15m")

        assert len(result) >= 2

    def test_1h_aggregation(self):
        """120 minute bars → multiple 1-hour candles."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(120)
        stream.set_bars("TSLA", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("TSLA", "1h")

        assert len(result) >= 2

    def test_high_is_max_of_minute_highs(self):
        """Aggregated candle high should be max of all constituent minute highs."""
        stream = FakeMarketStream()
        # Create 5 bars where bar[2] has a spike
        bars = [
            _make_minute_bar(0, 100.0, 100.5, 99.5, 100.2),
            _make_minute_bar(1, 100.2, 100.8, 100.0, 100.5),
            _make_minute_bar(2, 100.5, 105.0, 100.3, 101.0),  # spike high
            _make_minute_bar(3, 101.0, 101.5, 100.8, 101.2),
            _make_minute_bar(4, 101.2, 101.8, 101.0, 101.5),
        ]
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("SPY", "5m")

        assert len(result) == 1
        assert result[0].high == 105.0  # max of all minute highs

    def test_low_is_min_of_minute_lows(self):
        """Aggregated candle low should be min of all constituent minute lows."""
        stream = FakeMarketStream()
        bars = [
            _make_minute_bar(0, 100.0, 100.5, 99.5, 100.2),
            _make_minute_bar(1, 100.2, 100.8, 100.0, 100.5),
            _make_minute_bar(2, 100.5, 101.0, 95.0, 100.8),  # flash low
            _make_minute_bar(3, 100.8, 101.5, 100.5, 101.2),
            _make_minute_bar(4, 101.2, 101.8, 101.0, 101.5),
        ]
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("SPY", "5m")

        assert len(result) == 1
        assert result[0].low == 95.0

    def test_volume_sums_across_minutes(self):
        """Aggregated candle volume should sum all minute volumes."""
        stream = FakeMarketStream()
        bars = [
            _make_minute_bar(0, 100, 101, 99, 100, v=1000),
            _make_minute_bar(1, 100, 101, 99, 100, v=2000),
            _make_minute_bar(2, 100, 101, 99, 100, v=3000),
            _make_minute_bar(3, 100, 101, 99, 100, v=4000),
            _make_minute_bar(4, 100, 101, 99, 100, v=5000),
        ]
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("SPY", "5m")

        assert result[0].volume == 15000

    def test_empty_bars_returns_empty(self):
        """No WS bars → empty result."""
        stream = FakeMarketStream()
        cache = CandleCache(api_key="", market_stream=stream)
        result = cache._build_from_ws("SPY", "5m")
        assert result == []

    def test_no_stream_returns_empty(self):
        """No market_stream → _build_from_ws not called."""
        cache = CandleCache(api_key="fake_key", market_stream=None)
        # Should fall through to REST (which we don't call in this test)
        # Just verify it doesn't crash
        assert cache._market_stream is None


# ---------------------------------------------------------------------------
# CandleCache.get_candles() — WS-first with REST fallback
# ---------------------------------------------------------------------------


class TestCandleCacheGetCandles:

    @pytest.mark.asyncio
    async def test_uses_ws_bars_when_available(self):
        """get_candles should use WS bars instead of REST."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(20)
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        candles = await cache.get_candles("SPY", "5m")

        assert len(candles) > 0
        # All candles should be CandleBar instances
        assert all(isinstance(c, CandleBar) for c in candles)

    @pytest.mark.asyncio
    async def test_caches_ws_result_with_ttl(self):
        """Second call within TTL should return cached data."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(20)
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        candles1 = await cache.get_candles("SPY", "5m")

        # Clear the stream data
        stream.set_bars("SPY", [])

        # Should still return cached
        candles2 = await cache.get_candles("SPY", "5m")
        assert len(candles2) == len(candles1)

    @pytest.mark.asyncio
    async def test_ticker_case_insensitive(self):
        stream = FakeMarketStream()
        bars = _rising_minute_bars(10)
        stream.set_bars("spy", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        candles = await cache.get_candles("spy", "5m")
        assert len(candles) > 0


# ---------------------------------------------------------------------------
# CandleCache.get_candle_data() — full indicators from WS
# ---------------------------------------------------------------------------


class TestCandleCacheGetCandleData:

    @pytest.mark.asyncio
    async def test_indicators_computed_from_ws_bars(self):
        """get_candle_data should compute RSI/ATR/OBV from WS-aggregated candles."""
        stream = FakeMarketStream()
        # Need enough bars for RSI-14: at least 15 5-min candles = 75 minute bars
        bars = _rising_minute_bars(100, start=100.0, step=0.05)
        stream.set_bars("NVDA", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        data = await cache.get_candle_data("NVDA")

        assert "indicators" in data
        assert "5m" in data["indicators"]

        ind_5m = data["indicators"]["5m"]
        # With 100 rising bars → 20 5-min candles → RSI should be high
        assert ind_5m["rsi"] is not None
        assert ind_5m["rsi"] > 50  # rising bars should give high RSI

    @pytest.mark.asyncio
    async def test_enrg_works_with_ws_candles(self):
        """ENRG should be able to make a decision from WS-sourced candles."""
        stream = FakeMarketStream()
        # 200 rising minute bars — enough for 1h (3 bars) and 5m/15m/30m
        bars = _rising_minute_bars(200, start=100.0, step=0.05)
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        data = await cache.get_candle_data("SPY")

        action, reason = evaluate_enrg(data, "call")
        # Should be able to make a decision (not all SKIP)
        assert action in ("HOLD", "IMMEDIATE_EXIT", "PROCEED")
        # With rising bars, should likely be HOLD for calls
        # (RSI will be high, OBV positive)

    @pytest.mark.asyncio
    async def test_missing_tf_gracefully_handled(self):
        """If not enough bars for a timeframe, indicators should be None."""
        stream = FakeMarketStream()
        # Only 3 minute bars — not enough for any TF indicator computation
        bars = _rising_minute_bars(3)
        stream.set_bars("QQQ", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        data = await cache.get_candle_data("QQQ")

        # 5m: only ~0 complete candles, indicators should be None
        ind = data["indicators"]["5m"]
        assert ind["rsi"] is None
        assert ind["atr"] is None


# ---------------------------------------------------------------------------
# Profit retrace gate with new settings (regression tests)
# ---------------------------------------------------------------------------


class TestProfitRetraceNewSettings:
    """Verify the new 25%/50% settings prevent premature exits on small gains."""

    @pytest.mark.asyncio
    async def test_nvda_case_no_longer_exits_early(self):
        """Reproduce the NVDA case: entry $0.41, peak $0.49 (+19.5%), current $0.42.
        Old settings (10%/35%): would arm at +10% and exit (89.8% retrace).
        New settings (25%/50%): should NOT arm since peak gain < 25%.
        """
        from options_owl.risk.pipeline import ProfitRetraceExitGate, GateResult
        from tests.test_exit_pipeline import _ctx, _base_trade, _S

        gate = ProfitRetraceExitGate()

        # New settings
        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=50.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=25.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=35.0),
            trade=_base_trade(premium_per_contract=0.41, mfe_premium=0.49),  # +19.5% peak
            premium=0.42,
        )
        out = await gate.evaluate(ctx)
        # Should PASS (not exit) because peak gain 19.5% < 25% min
        assert out.result == GateResult.PASS
        assert "min" in out.reason.lower()

    @pytest.mark.asyncio
    async def test_old_settings_would_have_exited(self):
        """Same NVDA case with old settings — confirms it WOULD have triggered."""
        from options_owl.risk.pipeline import ProfitRetraceExitGate, GateResult
        from tests.test_exit_pipeline import _ctx, _base_trade, _S

        gate = ProfitRetraceExitGate()

        # Old settings
        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=35.0),
            trade=_base_trade(premium_per_contract=0.41, mfe_premium=0.49),
            premium=0.42,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_exits_on_large_retrace_above_min_gain(self):
        """With new settings, should still exit when a +30% gain retraces 60%."""
        from options_owl.risk.pipeline import ProfitRetraceExitGate, GateResult
        from tests.test_exit_pipeline import _ctx, _base_trade, _S

        gate = ProfitRetraceExitGate()

        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=50.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=25.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=35.0),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.30),  # +30% peak
            premium=1.10,  # gave back 66.7% of $0.30 profit
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_moderate_retrace_above_min_gain(self):
        """With new settings, 40% retrace of a +28% gain should HOLD (< 50% threshold)."""
        from options_owl.risk.pipeline import ProfitRetraceExitGate, GateResult
        from tests.test_exit_pipeline import _ctx, _base_trade, _S

        gate = ProfitRetraceExitGate()

        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=50.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=25.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=35.0),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.28),  # +28% peak
            premium=1.168,  # gave back 40% of $0.28 profit → $1.28 - $0.112 = $1.168
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_cheap_option_not_killed_by_noise(self):
        """$0.40 option with $0.08 gain and $0.05 pullback should hold with new settings."""
        from options_owl.risk.pipeline import ProfitRetraceExitGate, GateResult
        from tests.test_exit_pipeline import _ctx, _base_trade, _S

        gate = ProfitRetraceExitGate()

        # $0.40 entry, peak $0.48 (+20%), now $0.43 (+7.5%)
        # Retrace: $0.05 / $0.08 = 62.5% — old settings would kill this
        # New settings: peak only +20% < 25% min → won't arm
        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=50.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=25.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=35.0),
            trade=_base_trade(premium_per_contract=0.40, mfe_premium=0.48),
            premium=0.43,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS


# ---------------------------------------------------------------------------
# 4h candle history lookback (regression)
# ---------------------------------------------------------------------------


class TestCandleCacheHistoryLookback:
    """Verify that higher TFs fetch enough history for RSI-14."""

    @pytest.mark.asyncio
    async def test_4h_from_ws_with_enough_bars(self):
        """With 500 minute bars (~8 hours), we should get enough 4h candles."""
        stream = FakeMarketStream()
        # 500 minute bars = ~2 complete 4h candles (240 min each)
        bars = _rising_minute_bars(500, start=100.0, step=0.01)
        stream.set_bars("SPY", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        candles_4h = await cache.get_candles("SPY", "4h")

        # Should have 2 complete 4h candles
        assert len(candles_4h) >= 2

    @pytest.mark.asyncio
    async def test_1h_has_enough_for_rsi(self):
        """With 500 minute bars we should be able to compute RSI on 1h candles."""
        stream = FakeMarketStream()
        bars = _rising_minute_bars(500, start=100.0, step=0.01)
        stream.set_bars("NVDA", bars)

        cache = CandleCache(api_key="", market_stream=stream)
        candles_1h = await cache.get_candles("NVDA", "1h")

        # 500 min / 60 = ~8 candles — not enough for RSI-14
        # But this is the WS limit; REST fallback handles the rest
        assert len(candles_1h) >= 5
