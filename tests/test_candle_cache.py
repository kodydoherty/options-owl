"""Tests for candle_cache module — indicators, patterns, and exhaustion detection."""

from options_owl.collectors.candle_cache import (
    CandleBar,
    calc_atr,
    calc_obv,
    calc_rsi,
    calc_volume_trend,
    check_exhaustion,
    detect_candle_pattern,
    enrg_vote_tf,
    evaluate_enrg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(o: float, h: float, lo: float, c: float, v: float = 1000.0) -> CandleBar:
    return CandleBar(timestamp=0, open=o, high=h, low=lo, close=c, volume=v)


def _rising_bars(n: int = 20, start: float = 100.0, step: float = 0.5) -> list[CandleBar]:
    """Generate N bars with steadily rising closes."""
    bars = []
    for i in range(n):
        c = start + i * step
        bars.append(_bar(c - 0.2, c + 0.3, c - 0.4, c, 1000 + i * 10))
    return bars


def _falling_bars(n: int = 20, start: float = 110.0, step: float = 0.5) -> list[CandleBar]:
    """Generate N bars with steadily falling closes."""
    bars = []
    for i in range(n):
        c = start - i * step
        bars.append(_bar(c + 0.2, c + 0.4, c - 0.3, c, 1000 + i * 10))
    return bars


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

class TestATR:
    def test_returns_none_for_few_bars(self):
        assert calc_atr([_bar(1, 2, 0.5, 1.5)] * 5) is None

    def test_returns_positive_value(self):
        bars = _rising_bars(20)
        atr = calc_atr(bars)
        assert atr is not None
        assert atr > 0

    def test_high_volatility_bars(self):
        # Bars with large ranges
        bars = [_bar(100, 105, 95, 102, 1000) for _ in range(20)]
        atr = calc_atr(bars)
        assert atr is not None
        assert atr >= 5.0  # range is at least 10


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_returns_none_for_few_bars(self):
        assert calc_rsi([_bar(1, 2, 0.5, 1.5)] * 5) is None

    def test_overbought_on_rising(self):
        bars = _rising_bars(20)
        rsi = calc_rsi(bars)
        assert rsi is not None
        assert rsi > 60  # should be high since all bars are rising

    def test_oversold_on_falling(self):
        bars = _falling_bars(20)
        rsi = calc_rsi(bars)
        assert rsi is not None
        assert rsi < 40

    def test_returns_100_when_no_losses(self):
        # All gains, no losses
        bars = [_bar(100 + i, 101 + i, 99 + i, 101 + i) for i in range(20)]
        rsi = calc_rsi(bars)
        assert rsi == 100.0


# ---------------------------------------------------------------------------
# OBV tests
# ---------------------------------------------------------------------------

class TestOBV:
    def test_returns_none_for_single_bar(self):
        assert calc_obv([_bar(1, 2, 0.5, 1.5)]) is None

    def test_positive_obv_on_rising(self):
        bars = _rising_bars(10)
        obv = calc_obv(bars)
        assert obv is not None
        assert obv > 0

    def test_negative_obv_on_falling(self):
        bars = _falling_bars(10)
        obv = calc_obv(bars)
        assert obv is not None
        assert obv < 0


# ---------------------------------------------------------------------------
# Volume trend tests
# ---------------------------------------------------------------------------

class TestVolumeTrend:
    def test_returns_none_for_few_bars(self):
        assert calc_volume_trend([_bar(1, 2, 0.5, 1.5)] * 3) is None

    def test_rising_volume(self):
        bars = []
        for i in range(10):
            bars.append(_bar(100, 101, 99, 100.5, 100 + i * 50))
        assert calc_volume_trend(bars) == "rising"

    def test_falling_volume(self):
        bars = []
        for i in range(10):
            bars.append(_bar(100, 101, 99, 100.5, 500 - i * 50))
        assert calc_volume_trend(bars) == "falling"


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------

class TestCandlePattern:
    def test_no_bars(self):
        assert detect_candle_pattern([]) is None

    def test_doji(self):
        # Open ~= Close, but has range
        bar = _bar(100.0, 101.0, 99.0, 100.05)
        assert detect_candle_pattern([bar]) == "doji"

    def test_shooting_star(self):
        # Long upper wick, meaningful body at bottom, small lower wick
        # body = 0.8, range = 5.2, upper_wick = 4.0, lower_wick = 0.4
        bar = _bar(100.4, 105.2, 100.0, 101.2)
        assert detect_candle_pattern([bar]) == "shooting_star"

    def test_hammer(self):
        # Long lower wick, meaningful body at top, small upper wick
        # body = 0.8, range = 5.2, lower_wick = 4.0, upper_wick = 0.4
        bar = _bar(104.0, 105.2, 100.0, 104.8)
        assert detect_candle_pattern([bar]) == "hammer"

    def test_engulfing_bearish(self):
        prev = _bar(100, 101, 99.5, 100.5)
        curr = _bar(101, 102, 99, 99.2)  # opens high, closes below prev
        assert detect_candle_pattern([prev, curr]) == "engulfing_bearish"

    def test_engulfing_bullish(self):
        prev = _bar(100.5, 101, 99.5, 100)
        curr = _bar(99, 102, 98.5, 101.5)  # opens low, closes above prev
        assert detect_candle_pattern([prev, curr]) == "engulfing_bullish"

    def test_normal_bar_no_pattern(self):
        bar = _bar(100, 102, 99, 101)  # regular green bar
        assert detect_candle_pattern([bar]) is None


# ---------------------------------------------------------------------------
# Exhaustion detection tests
# ---------------------------------------------------------------------------

class TestCheckExhaustion:
    def _make_candle_data(self, rsi_5m=None, pattern=None, vol_trend=None, rsi_15m=None):
        return {
            "indicators": {
                "5m": {
                    "atr": 1.5,
                    "rsi": rsi_5m,
                    "obv": 50000,
                    "pattern": pattern,
                    "volume_trend": vol_trend,
                },
                "15m": {
                    "atr": 2.0,
                    "rsi": rsi_15m,
                    "obv": 100000,
                    "pattern": None,
                    "volume_trend": None,
                },
                "1h": {
                    "atr": None, "rsi": None, "obv": None,
                    "pattern": None, "volume_trend": None,
                },
            }
        }

    def test_no_exhaustion_below_min_gain(self):
        data = self._make_candle_data(rsi_5m=75, pattern="shooting_star")
        exhausted, _ = check_exhaustion(data, "call", peak_gain_pct=20.0)
        assert not exhausted

    def test_no_exhaustion_with_zero_signals(self):
        data = self._make_candle_data(rsi_5m=50, pattern=None, vol_trend=None)
        exhausted, _ = check_exhaustion(data, "call", peak_gain_pct=50.0)
        assert not exhausted

    def test_no_exhaustion_with_one_signal(self):
        # Only RSI overbought, need 2+ signals
        data = self._make_candle_data(rsi_5m=75, pattern=None, vol_trend=None)
        exhausted, _ = check_exhaustion(data, "call", peak_gain_pct=50.0)
        assert not exhausted

    def test_exhaustion_with_two_signals_call(self):
        # RSI overbought + shooting star
        data = self._make_candle_data(rsi_5m=75, pattern="shooting_star")
        exhausted, reason = check_exhaustion(data, "call", peak_gain_pct=50.0)
        assert exhausted
        assert "RSI=75" in reason
        assert "shooting_star" in reason

    def test_exhaustion_rsi_plus_volume(self):
        # RSI overbought + volume declining
        data = self._make_candle_data(rsi_5m=72, vol_trend="falling")
        exhausted, reason = check_exhaustion(data, "call", peak_gain_pct=50.0)
        assert exhausted
        assert "volume declining" in reason

    def test_exhaustion_with_15m_confirmation(self):
        # 5m pattern + 15m RSI
        data = self._make_candle_data(pattern="doji", rsi_15m=68)
        exhausted, reason = check_exhaustion(data, "call", peak_gain_pct=50.0)
        assert exhausted
        assert "15m RSI" in reason

    def test_exhaustion_for_puts(self):
        # For puts: RSI < 30 and hammer pattern
        data = self._make_candle_data(rsi_5m=25, pattern="hammer")
        exhausted, reason = check_exhaustion(data, "put", peak_gain_pct=50.0)
        assert exhausted
        assert "RSI=25" in reason

    def test_no_exhaustion_wrong_direction(self):
        # RSI overbought on a PUT — should not trigger
        data = self._make_candle_data(rsi_5m=75, pattern="shooting_star")
        exhausted, _ = check_exhaustion(data, "put", peak_gain_pct=50.0)
        assert not exhausted

    def test_empty_candle_data(self):
        exhausted, reason = check_exhaustion({}, "call", peak_gain_pct=50.0)
        assert not exhausted
        assert "no candle data" in reason

    def test_three_signals_detected(self):
        # RSI + pattern + volume
        data = self._make_candle_data(rsi_5m=72, pattern="shooting_star", vol_trend="falling")
        exhausted, reason = check_exhaustion(data, "call", peak_gain_pct=60.0)
        assert exhausted
        assert "3 signals" in reason


# ---------------------------------------------------------------------------
# ENRG per-TF vote tests
# ---------------------------------------------------------------------------

class TestENRGVoteTF:
    def test_bullish_call_high_rsi_positive_obv(self):
        ind = {"rsi": 55, "obv": 5000, "pattern": None}
        assert enrg_vote_tf(ind, "call") == "BULLISH"

    def test_bearish_call_low_rsi(self):
        ind = {"rsi": 35, "obv": 5000, "pattern": None}
        assert enrg_vote_tf(ind, "call") == "BEARISH"

    def test_bearish_call_shooting_star(self):
        ind = {"rsi": 55, "obv": 5000, "pattern": "shooting_star"}
        assert enrg_vote_tf(ind, "call") == "BEARISH"

    def test_neutral_call_no_rsi(self):
        ind = {"rsi": None, "obv": 5000, "pattern": None}
        assert enrg_vote_tf(ind, "call") == "NEUTRAL"

    def test_neutral_call_mid_rsi_negative_obv(self):
        # RSI > 40 but OBV negative and no bullish pattern → NEUTRAL
        ind = {"rsi": 50, "obv": -5000, "pattern": None}
        assert enrg_vote_tf(ind, "call") == "NEUTRAL"

    def test_bullish_put_low_rsi_negative_obv(self):
        ind = {"rsi": 45, "obv": -5000, "pattern": None}
        assert enrg_vote_tf(ind, "put") == "BULLISH"

    def test_bearish_put_high_rsi(self):
        ind = {"rsi": 65, "obv": -5000, "pattern": None}
        assert enrg_vote_tf(ind, "put") == "BEARISH"

    def test_bearish_put_hammer_pattern(self):
        # For puts, hammer is a bearish signal (reversal against put direction)
        ind = {"rsi": 55, "obv": -5000, "pattern": "hammer"}
        assert enrg_vote_tf(ind, "put") == "BEARISH"


# ---------------------------------------------------------------------------
# ENRG evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluateENRG:
    def _make_enrg_data(self, tf_overrides=None):
        """Build candle_data dict with indicators for all 5 timeframes."""
        defaults = {
            "5m":  {"rsi": 50, "obv": 1000, "pattern": None, "atr": 1.0, "volume_trend": None},
            "15m": {"rsi": 50, "obv": 1000, "pattern": None, "atr": 1.5, "volume_trend": None},
            "30m": {"rsi": 50, "obv": 1000, "pattern": None, "atr": 2.0, "volume_trend": None},
            "1h":  {"rsi": 50, "obv": 1000, "pattern": None, "atr": 2.5, "volume_trend": None},
            "4h":  {"rsi": 50, "obv": 1000, "pattern": None, "atr": 3.0, "volume_trend": None},
        }
        if tf_overrides:
            for tf, overrides in tf_overrides.items():
                defaults[tf].update(overrides)
        return {"indicators": defaults}

    def test_no_data_returns_proceed(self):
        action, _ = evaluate_enrg({}, "call")
        assert action == "PROCEED"

    def test_all_bullish_returns_hold(self):
        # All TFs: RSI > 40, OBV > 0 → all vote BULLISH → HOLD
        data = self._make_enrg_data()
        action, reason = evaluate_enrg(data, "call")
        assert action == "HOLD"
        assert "thesis intact" in reason

    def test_all_bearish_returns_exit(self):
        # All TFs: RSI < 40 → all vote BEARISH → EXIT
        overrides = {tf: {"rsi": 30} for tf in ("5m", "15m", "30m", "1h", "4h")}
        data = self._make_enrg_data(overrides)
        action, reason = evaluate_enrg(data, "call")
        assert action == "IMMEDIATE_EXIT"
        assert "thesis broken" in reason

    def test_extreme_pattern_1h_overrides(self):
        # engulfing_bearish on 1h → IMMEDIATE_EXIT even if other TFs are bullish
        data = self._make_enrg_data({"1h": {"pattern": "engulfing_bearish"}})
        action, reason = evaluate_enrg(data, "call")
        assert action == "IMMEDIATE_EXIT"
        assert "extreme override" in reason
        assert "1h" in reason

    def test_extreme_pattern_4h_overrides(self):
        data = self._make_enrg_data({"4h": {"pattern": "shooting_star"}})
        action, reason = evaluate_enrg(data, "call")
        assert action == "IMMEDIATE_EXIT"
        assert "4h" in reason

    def test_extreme_pattern_put_direction(self):
        # For puts: hammer on 4h is the extreme reversal
        data = self._make_enrg_data({"4h": {"pattern": "hammer"}})
        action, reason = evaluate_enrg(data, "put")
        assert action == "IMMEDIATE_EXIT"

    def test_extreme_pattern_wrong_direction_no_override(self):
        # shooting_star on 1h for a PUT should NOT trigger extreme override
        # (shooting_star is bearish for underlying, which is bullish for puts)
        data = self._make_enrg_data({"1h": {"pattern": "shooting_star"}})
        action, _ = evaluate_enrg(data, "put")
        assert action != "IMMEDIATE_EXIT" or "extreme" not in _

    def test_higher_tf_weights_matter(self):
        # 5m + 15m + 30m bearish (weight 3) vs 1h + 4h bullish (weight 4) → HOLD
        overrides = {
            "5m": {"rsi": 30},
            "15m": {"rsi": 30},
            "30m": {"rsi": 30},
            "1h": {"rsi": 55, "obv": 5000},
            "4h": {"rsi": 55, "obv": 5000},
        }
        data = self._make_enrg_data(overrides)
        action, reason = evaluate_enrg(data, "call")
        assert action == "HOLD"
        assert "bullish=4" in reason

    def test_tie_returns_proceed(self):
        # 5m bullish (1) + 15m bearish (1) + rest NEUTRAL → tie
        overrides = {
            "5m": {"rsi": 55, "obv": 5000},
            "15m": {"rsi": 30},
            "30m": {"rsi": None},
            "1h": {"rsi": None},
            "4h": {"rsi": None},
        }
        data = self._make_enrg_data(overrides)
        action, reason = evaluate_enrg(data, "call")
        assert action == "PROCEED"
        assert "inconclusive" in reason

    def test_put_all_bullish_hold(self):
        # For puts: RSI < 60 and OBV < 0 → BULLISH
        overrides = {tf: {"rsi": 45, "obv": -5000} for tf in ("5m", "15m", "30m", "1h", "4h")}
        data = self._make_enrg_data(overrides)
        action, _ = evaluate_enrg(data, "put")
        assert action == "HOLD"
