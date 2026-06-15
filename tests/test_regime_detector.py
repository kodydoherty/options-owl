"""Tests for intraday regime detector — classification, hysteresis, gating."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from options_owl.risk.regime_detector import (
    RegimeDetector,
    RegimeState,
    _adx,
    _ema,
    _rsi,
    compute_conviction_multiplier,
    get_allowed_directions,
    get_direction_slots,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeBar:
    open: float = 100.0
    high: float = 101.0
    low: float = 99.0
    close: float = 100.5
    vwap: float = 100.3
    volume: int = 1000


def _make_bars(closes, *, vwap=None, highs=None, lows=None, open_price=100.0):
    """Build a list of FakeBars from close prices."""
    bars = []
    for i, c in enumerate(closes):
        bars.append(FakeBar(
            open=open_price if i == 0 else closes[i - 1],
            high=highs[i] if highs else c + 0.5,
            low=lows[i] if lows else c - 0.5,
            close=c,
            vwap=vwap if vwap is not None else c - 0.1,
        ))
    return bars


def _mock_cache(bars):
    cache = AsyncMock()
    cache.get_candles = AsyncMock(return_value=bars)
    return cache


def _now():
    return datetime(2026, 5, 29, 10, 30, tzinfo=ET)


# ---------------------------------------------------------------------------
# _classify (via direct call on detector instance)
# ---------------------------------------------------------------------------

class TestClassify:
    def test_all_bullish(self):
        d = RegimeDetector()
        ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        assert d._classify(ind) == RegimeState.BULLISH

    def test_all_bearish(self):
        d = RegimeDetector()
        ind = {"price": 99, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 45, "adx": 25}
        assert d._classify(ind) == RegimeState.BEARISH

    def test_low_adx_choppy(self):
        d = RegimeDetector()
        ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 60, "adx": 15}
        assert d._classify(ind) == RegimeState.CHOPPY

    def test_two_bullish_one_bearish(self):
        d = RegimeDetector()
        # price > vwap (bull), ema9 > ema21 (bull), rsi < 50 (bear)
        ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 45, "adx": 25}
        assert d._classify(ind) == RegimeState.BULLISH

    def test_one_bullish_two_bearish(self):
        d = RegimeDetector()
        # price > vwap (bull), ema9 < ema21 (bear), rsi < 50 (bear)
        ind = {"price": 101, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 45, "adx": 25}
        assert d._classify(ind) == RegimeState.BEARISH

    def test_tie_returns_choppy(self):
        d = RegimeDetector()
        # price > vwap (bull), ema9 < ema21 (bear), rsi == 50 (neither)
        ind = {"price": 101, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 50, "adx": 25}
        assert d._classify(ind) == RegimeState.CHOPPY

    def test_zero_price_and_vwap_ignored(self):
        d = RegimeDetector()
        ind = {"price": 0, "vwap": 0, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        # ema bull + rsi bull = 2 → BULLISH
        assert d._classify(ind) == RegimeState.BULLISH

    def test_zero_ema_ignored(self):
        d = RegimeDetector()
        ind = {"price": 99, "vwap": 100, "ema9": 0, "ema21": 0, "rsi": 45, "adx": 25}
        # price bear + rsi bear = 2 → BEARISH
        assert d._classify(ind) == RegimeState.BEARISH

    def test_adx_exactly_at_threshold_is_choppy(self):
        d = RegimeDetector()
        ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 20}
        # adx < 20 is choppy, adx == 20 is NOT < 20 → trending
        assert d._classify(ind) == RegimeState.BULLISH


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    @pytest.mark.asyncio
    async def test_single_reading_does_not_flip(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        # Start CHOPPY, feed one BULLISH reading
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        d._check_hard_reversal = MagicMock(return_value=False)

        result = await d.update(None, now_et=_now())
        assert result == RegimeState.CHOPPY

    @pytest.mark.asyncio
    async def test_two_consecutive_readings_confirm_flip(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        d._check_hard_reversal = MagicMock(return_value=False)

        t1 = _now()
        t2 = t1 + timedelta(minutes=5)
        await d.update(None, now_et=t1)
        result = await d.update(None, now_et=t2)
        assert result == RegimeState.BULLISH

    @pytest.mark.asyncio
    async def test_min_hold_prevents_reevaluation(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=15)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        bearish_ind = {"price": 99, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 45, "adx": 25}

        d._check_hard_reversal = MagicMock(return_value=False)

        # First: confirm BULLISH (need 2 readings, no hold yet since state_since is None)
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        t1 = _now()
        await d.update(None, now_et=t1)
        t2 = t1 + timedelta(minutes=5)
        await d.update(None, now_et=t2)
        assert d.state == RegimeState.BULLISH

        # Now feed bearish but within min_hold_minutes → should stay BULLISH
        d._get_spy_indicators = AsyncMock(return_value=bearish_ind)
        t3 = t2 + timedelta(minutes=10)  # only 10 min since state_since
        result = await d.update(None, now_et=t3)
        assert result == RegimeState.BULLISH

    @pytest.mark.asyncio
    async def test_different_pending_resets_counter(self):
        d = RegimeDetector(hysteresis_checks=3, min_hold_minutes=0)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        bearish_ind = {"price": 99, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 45, "adx": 25}

        d._check_hard_reversal = MagicMock(return_value=False)

        # Feed 2 bullish (need 3 to confirm)
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        t = _now()
        await d.update(None, now_et=t)
        t += timedelta(minutes=5)
        await d.update(None, now_et=t)
        assert d.state == RegimeState.CHOPPY
        assert d._pending_count == 2

        # Now feed bearish → resets pending to bearish with count=1
        d._get_spy_indicators = AsyncMock(return_value=bearish_ind)
        t += timedelta(minutes=5)
        await d.update(None, now_et=t)
        assert d._pending_state == RegimeState.BEARISH
        assert d._pending_count == 1

    @pytest.mark.asyncio
    async def test_matching_state_resets_pending(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        choppy_ind = {"price": 100, "vwap": 100, "ema9": 50, "ema21": 50, "rsi": 50, "adx": 15}
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}

        d._check_hard_reversal = MagicMock(return_value=False)

        # Feed 1 bullish (pending count = 1)
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        await d.update(None, now_et=_now())
        assert d._pending_count == 1

        # Feed choppy (matches state) → pending resets
        d._get_spy_indicators = AsyncMock(return_value=choppy_ind)
        await d.update(None, now_et=_now() + timedelta(minutes=5))
        assert d._pending_state is None
        assert d._pending_count == 0

    @pytest.mark.asyncio
    async def test_empty_indicators_preserves_state(self):
        d = RegimeDetector()
        d._get_spy_indicators = AsyncMock(return_value={})
        d._check_hard_reversal = MagicMock(return_value=False)
        result = await d.update(None, now_et=_now())
        assert result == RegimeState.CHOPPY
        assert d._regime_changed is False


# ---------------------------------------------------------------------------
# allows_direction
# ---------------------------------------------------------------------------

class TestAllowsDirection:
    def test_bullish_allows_call(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.allows_direction("call") is True

    def test_bullish_blocks_put(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.allows_direction("put") is False

    def test_bearish_allows_put(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        assert d.allows_direction("put") is True

    def test_bearish_blocks_call(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        assert d.allows_direction("call") is False

    def test_choppy_allows_both(self):
        d = RegimeDetector(state=RegimeState.CHOPPY)
        assert d.allows_direction("call") is True
        assert d.allows_direction("put") is True

    def test_case_insensitive(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.allows_direction("CALL") is True
        assert d.allows_direction("Call") is True


# ---------------------------------------------------------------------------
# get_size_multiplier
# ---------------------------------------------------------------------------

class TestGetSizeMultiplier:
    def test_choppy_returns_configured_mult(self):
        d = RegimeDetector(state=RegimeState.CHOPPY, choppy_size_mult=0.6)
        assert d.get_size_multiplier() == 0.6

    def test_choppy_custom_mult(self):
        d = RegimeDetector(state=RegimeState.CHOPPY, choppy_size_mult=0.5)
        assert d.get_size_multiplier() == 0.5

    def test_bullish_returns_1(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.get_size_multiplier() == 1.0

    def test_bearish_returns_1(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        assert d.get_size_multiplier() == 1.0


# ---------------------------------------------------------------------------
# is_counter_trend
# ---------------------------------------------------------------------------

class TestIsCounterTrend:
    def test_call_during_bearish(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        assert d.is_counter_trend("call") is True

    def test_put_during_bullish(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.is_counter_trend("put") is True

    def test_call_during_bullish(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        assert d.is_counter_trend("call") is False

    def test_put_during_bearish(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        assert d.is_counter_trend("put") is False

    def test_choppy_never_counter_trend(self):
        d = RegimeDetector(state=RegimeState.CHOPPY)
        assert d.is_counter_trend("call") is False
        assert d.is_counter_trend("put") is False


# ---------------------------------------------------------------------------
# get_tighten_factor
# ---------------------------------------------------------------------------

class TestGetTightenFactor:
    def test_no_regime_change_returns_1(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        d._regime_changed = False
        assert d.get_tighten_factor("call") == 1.0

    def test_counter_trend_after_change(self):
        d = RegimeDetector(state=RegimeState.BEARISH)
        d._regime_changed = True
        assert d.get_tighten_factor("call") == 0.60

    def test_choppy_after_change(self):
        d = RegimeDetector(state=RegimeState.CHOPPY)
        d._regime_changed = True
        assert d.get_tighten_factor("call") == 0.80

    def test_aligned_after_change(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        d._regime_changed = True
        assert d.get_tighten_factor("call") == 1.0

    def test_put_counter_trend_after_change(self):
        d = RegimeDetector(state=RegimeState.BULLISH)
        d._regime_changed = True
        assert d.get_tighten_factor("put") == 0.60


# ---------------------------------------------------------------------------
# regime_changed property
# ---------------------------------------------------------------------------

class TestRegimeChanged:
    @pytest.mark.asyncio
    async def test_regime_changed_flag_set_on_flip(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        d._check_hard_reversal = MagicMock(return_value=False)

        t = _now()
        await d.update(None, now_et=t)
        await d.update(None, now_et=t + timedelta(minutes=5))
        assert d.regime_changed is True

    @pytest.mark.asyncio
    async def test_regime_changed_clears_when_state_matches(self):
        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
        d._check_hard_reversal = MagicMock(return_value=False)

        t = _now()
        await d.update(None, now_et=t)
        await d.update(None, now_et=t + timedelta(minutes=5))
        assert d.regime_changed is True

        # Third reading still bullish → state matches → _regime_changed = False
        await d.update(None, now_et=t + timedelta(minutes=20))
        assert d.regime_changed is False


# ---------------------------------------------------------------------------
# get_allowed_directions
# ---------------------------------------------------------------------------

class TestGetAllowedDirections:
    def test_opening_buffer(self):
        assert get_allowed_directions(0, RegimeState.BULLISH) == []
        assert get_allowed_directions(4, RegimeState.BEARISH) == []

    def test_morning_bullish(self):
        assert get_allowed_directions(30, RegimeState.BULLISH) == ["call"]

    def test_morning_bearish(self):
        assert get_allowed_directions(30, RegimeState.BEARISH) == ["put"]

    def test_morning_choppy_defaults_call(self):
        assert get_allowed_directions(30, RegimeState.CHOPPY) == ["call"]

    def test_midday_requires_extended(self):
        assert get_allowed_directions(150, RegimeState.BULLISH, extended_scan_enabled=False) == []

    def test_midday_bullish_extended(self):
        assert get_allowed_directions(150, RegimeState.BULLISH, extended_scan_enabled=True) == ["call"]

    def test_midday_bearish_extended(self):
        assert get_allowed_directions(150, RegimeState.BEARISH, extended_scan_enabled=True) == ["put"]

    def test_midday_choppy_extended_empty(self):
        assert get_allowed_directions(150, RegimeState.CHOPPY, extended_scan_enabled=True) == []

    def test_afternoon_includes_put(self):
        dirs = get_allowed_directions(250, RegimeState.BEARISH, extended_scan_enabled=True)
        assert "put" in dirs

    def test_afternoon_bullish_includes_both(self):
        dirs = get_allowed_directions(250, RegimeState.BULLISH, extended_scan_enabled=True)
        assert "put" in dirs
        assert "call" in dirs

    def test_late_bearish_put_only(self):
        assert get_allowed_directions(320, RegimeState.BEARISH, extended_scan_enabled=True) == ["put"]

    def test_late_bullish_empty(self):
        assert get_allowed_directions(320, RegimeState.BULLISH, extended_scan_enabled=True) == []

    def test_past_3pm_empty(self):
        assert get_allowed_directions(340, RegimeState.BEARISH, extended_scan_enabled=True) == []
        assert get_allowed_directions(340, RegimeState.BULLISH, extended_scan_enabled=True) == []

    def test_boundary_minute_5(self):
        assert get_allowed_directions(5, RegimeState.BULLISH) == ["call"]

    def test_boundary_minute_90(self):
        assert get_allowed_directions(90, RegimeState.BULLISH) == ["call"]

    def test_boundary_minute_91_no_extended(self):
        assert get_allowed_directions(91, RegimeState.BULLISH, extended_scan_enabled=False) == []

    def test_boundary_minute_330(self):
        assert get_allowed_directions(330, RegimeState.BEARISH, extended_scan_enabled=True) == ["put"]

    def test_boundary_minute_331(self):
        assert get_allowed_directions(331, RegimeState.BEARISH, extended_scan_enabled=True) == []


# ---------------------------------------------------------------------------
# get_direction_slots
# ---------------------------------------------------------------------------

class TestGetDirectionSlots:
    def test_dynamic_disabled_default(self):
        slots = get_direction_slots(RegimeState.BULLISH, max_concurrent=8, dynamic_puts_enabled=False)
        assert slots == {"call": 6, "put": 2}

    def test_dynamic_disabled_any_regime(self):
        for regime in RegimeState:
            slots = get_direction_slots(regime, max_concurrent=8, dynamic_puts_enabled=False)
            assert slots == {"call": 6, "put": 2}

    def test_bullish_dynamic(self):
        slots = get_direction_slots(RegimeState.BULLISH, max_concurrent=8, dynamic_puts_enabled=True)
        assert slots["call"] > slots["put"]
        assert slots["call"] + slots["put"] <= 8

    def test_bearish_dynamic(self):
        slots = get_direction_slots(RegimeState.BEARISH, max_concurrent=8, dynamic_puts_enabled=True)
        assert slots["put"] > slots["call"]
        assert slots["call"] + slots["put"] <= 8

    def test_choppy_dynamic_reduced_total(self):
        slots = get_direction_slots(RegimeState.CHOPPY, max_concurrent=8, dynamic_puts_enabled=True)
        total = slots["call"] + slots["put"]
        assert total < 8
        assert total >= 4

    def test_bullish_dynamic_values(self):
        slots = get_direction_slots(RegimeState.BULLISH, max_concurrent=8, dynamic_puts_enabled=True)
        # put_slots = max(2, 8//4) = 2, call = 8 - 2 = 6
        assert slots == {"call": 6, "put": 2}

    def test_bearish_dynamic_values(self):
        slots = get_direction_slots(RegimeState.BEARISH, max_concurrent=8, dynamic_puts_enabled=True)
        # call_slots = max(2, 8//4) = 2, put = 8 - 2 = 6
        assert slots == {"call": 2, "put": 6}

    def test_choppy_dynamic_values(self):
        slots = get_direction_slots(RegimeState.CHOPPY, max_concurrent=8, dynamic_puts_enabled=True)
        # total = max(4, int(8*0.75)) = 6, half = 3
        assert slots == {"call": 3, "put": 3}

    def test_small_max_concurrent(self):
        slots = get_direction_slots(RegimeState.BEARISH, max_concurrent=4, dynamic_puts_enabled=True)
        # call_slots = max(2, 4//4) = 2, put = 4 - 2 = 2
        assert slots == {"call": 2, "put": 2}


# ---------------------------------------------------------------------------
# compute_conviction_multiplier
# ---------------------------------------------------------------------------

class TestConvictionMultiplier:
    def test_disabled_returns_1(self):
        result = compute_conviction_multiplier(0.5, RegimeState.CHOPPY, "call", 200, conviction_enabled=False)
        assert result == 1.0

    def test_high_confidence_aligned_morning(self):
        result = compute_conviction_multiplier(
            0.96, RegimeState.BULLISH, "call", 30, conviction_enabled=True
        )
        # base=1.0, regime=1.0, time=1.0 → 1.0
        assert result == 1.0

    def test_low_confidence_choppy_afternoon(self):
        result = compute_conviction_multiplier(
            0.80, RegimeState.CHOPPY, "call", 250, conviction_enabled=True
        )
        # base=0.70, regime=0.70, time=0.75 → 0.3675 → clamped to 0.40
        assert result == 0.40

    def test_counter_trend_halved(self):
        result = compute_conviction_multiplier(
            0.96, RegimeState.BEARISH, "call", 30, conviction_enabled=True
        )
        # base=1.0, regime=0.50, time=1.0 → 0.50
        assert result == 0.50

    def test_mid_confidence_aligned_midday(self):
        result = compute_conviction_multiplier(
            0.92, RegimeState.BULLISH, "call", 150, conviction_enabled=True
        )
        # base=0.85, regime=1.0, time=0.80 → 0.68
        assert result == pytest.approx(0.68, abs=0.01)

    def test_floor_at_040(self):
        result = compute_conviction_multiplier(
            0.80, RegimeState.BEARISH, "call", 250, conviction_enabled=True
        )
        # base=0.70, regime=0.50, time=0.75 → 0.2625 → clamped to 0.40
        assert result == 0.40

    def test_ceiling_at_1(self):
        result = compute_conviction_multiplier(
            0.99, RegimeState.BULLISH, "call", 10, conviction_enabled=True
        )
        assert result <= 1.0

    def test_time_decay_brackets(self):
        # minute <= 60 → 1.0
        r1 = compute_conviction_multiplier(0.96, RegimeState.BULLISH, "call", 60, conviction_enabled=True)
        # minute <= 120 → 0.90
        r2 = compute_conviction_multiplier(0.96, RegimeState.BULLISH, "call", 120, conviction_enabled=True)
        # minute <= 210 → 0.80
        r3 = compute_conviction_multiplier(0.96, RegimeState.BULLISH, "call", 210, conviction_enabled=True)
        # minute > 210 → 0.75
        r4 = compute_conviction_multiplier(0.96, RegimeState.BULLISH, "call", 250, conviction_enabled=True)
        assert r1 == 1.0
        assert r2 == 0.90
        assert r3 == 0.80
        assert r4 == 0.75

    def test_confidence_tiers(self):
        # >= 0.95 → base 1.0
        r1 = compute_conviction_multiplier(0.95, RegimeState.BULLISH, "call", 30, conviction_enabled=True)
        # >= 0.90 → base 0.85
        r2 = compute_conviction_multiplier(0.90, RegimeState.BULLISH, "call", 30, conviction_enabled=True)
        # < 0.90 → base 0.70
        r3 = compute_conviction_multiplier(0.89, RegimeState.BULLISH, "call", 30, conviction_enabled=True)
        assert r1 == 1.0
        assert r2 == 0.85
        assert r3 == 0.70


# ---------------------------------------------------------------------------
# _ema helper
# ---------------------------------------------------------------------------

class TestEma:
    def test_known_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _ema(data, 3)
        # EMA(3) starting from data[-3]=3.0, k=2/(3+1)=0.5
        # ema = 3.0
        # ema = 4.0*0.5 + 3.0*0.5 = 3.5
        # ema = 5.0*0.5 + 3.5*0.5 = 4.25
        assert result == pytest.approx(4.25)

    def test_insufficient_data(self):
        assert _ema([1.0, 2.0], 5) == 0.0

    def test_exact_period_length(self):
        data = [10.0, 20.0, 30.0]
        result = _ema(data, 3)
        # starts at 10, k=0.5
        # ema = 10 → 20*0.5+10*0.5=15 → 30*0.5+15*0.5=22.5
        assert result == pytest.approx(22.5)

    def test_period_1_returns_last(self):
        data = [5.0, 10.0, 15.0]
        assert _ema(data, 1) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# _rsi helper
# ---------------------------------------------------------------------------

class TestRsi:
    def test_flat_data_returns_50(self):
        data = [100.0] * 20
        assert _rsi(data, 14) == 50.0

    def test_consistently_rising(self):
        data = list(range(20))  # 0,1,2,...,19
        result = _rsi(data, 14)
        # All gains, no losses → RSI = 100
        assert result == 100.0

    def test_consistently_falling(self):
        data = list(range(20, 0, -1))  # 20,19,...,1
        result = _rsi(data, 14)
        # All losses, no gains → avg_gain=0 → RSI = 50 (no gains branch)
        # Actually: avg_loss > 0, avg_gain = 0, rs = 0 → RSI = 100 - 100/1 = 0
        # Wait: code says if avg_loss == 0 return 100 if avg_gain > 0 else 50
        # But here avg_loss > 0 and avg_gain == 0 → rs = 0 → RSI = 100 - 100/1 = 0
        assert result == pytest.approx(0.0)

    def test_insufficient_data(self):
        data = [100.0] * 10
        assert _rsi(data, 14) == 50.0

    def test_mixed_data(self):
        data = [100, 102, 101, 103, 102, 104, 103, 105,
                104, 106, 105, 107, 106, 108, 107]
        result = _rsi(data, 14)
        assert 0 <= result <= 100


# ---------------------------------------------------------------------------
# _adx helper
# ---------------------------------------------------------------------------

class TestAdx:
    def test_insufficient_data(self):
        assert _adx([1, 2], [1, 2], [1, 2], 14) == 0.0

    def test_trending_data_above_20(self):
        n = 30
        highs = [100 + i * 1.0 for i in range(n)]
        lows = [99 + i * 1.0 for i in range(n)]
        closes = [99.5 + i * 1.0 for i in range(n)]
        result = _adx(highs, lows, closes, 14)
        assert result > 20

    def test_flat_data_low_adx(self):
        n = 30
        highs = [100.5] * n
        lows = [99.5] * n
        closes = [100.0] * n
        result = _adx(highs, lows, closes, 14)
        assert result == 0.0

    def test_returns_non_negative(self):
        n = 30
        highs = [100 + (i % 3) for i in range(n)]
        lows = [98 + (i % 3) for i in range(n)]
        closes = [99 + (i % 3) for i in range(n)]
        result = _adx(highs, lows, closes, 14)
        assert result >= 0


# ---------------------------------------------------------------------------
# Integration: update() with mock candle_cache
# ---------------------------------------------------------------------------

class TestUpdateIntegration:
    @pytest.mark.asyncio
    async def test_update_with_bullish_candles(self):
        # Build bars with price > vwap, ema9 > ema21, rsi > 50
        closes = [100 + i * 0.5 for i in range(30)]  # trending up
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.3 for c in closes]
        bars = _make_bars(closes, vwap=closes[-1] - 1.0, highs=highs, lows=lows)
        cache = _mock_cache(bars)

        d = RegimeDetector(hysteresis_checks=2, min_hold_minutes=0)
        t1 = _now()
        await d.update(cache, now_et=t1)
        t2 = t1 + timedelta(minutes=5)
        result = await d.update(cache, now_et=t2)
        assert result == RegimeState.BULLISH

    @pytest.mark.asyncio
    async def test_update_with_insufficient_bars(self):
        bars = _make_bars([100, 101, 102])  # only 3 bars, need 22+
        cache = _mock_cache(bars)

        d = RegimeDetector()
        result = await d.update(cache, now_et=_now())
        assert result == RegimeState.CHOPPY

    @pytest.mark.asyncio
    async def test_update_exception_preserves_state(self):
        cache = AsyncMock()
        cache.get_candles = AsyncMock(side_effect=RuntimeError("network error"))

        d = RegimeDetector(state=RegimeState.BULLISH)
        d.state_since = _now() - timedelta(minutes=30)
        d._last_update = _now() - timedelta(minutes=10)
        result = await d.update(cache, now_et=_now())
        assert result == RegimeState.BULLISH

    @pytest.mark.asyncio
    async def test_history_trimmed_at_50(self):
        d = RegimeDetector(hysteresis_checks=1, min_hold_minutes=0)
        bullish_ind = {"price": 101, "vwap": 100, "ema9": 50, "ema21": 49, "rsi": 55, "adx": 25}
        bearish_ind = {"price": 99, "vwap": 100, "ema9": 48, "ema21": 50, "rsi": 45, "adx": 25}

        d._check_hard_reversal = MagicMock(return_value=False)

        t = _now()
        for i in range(60):
            # Alternate to keep flipping and adding history entries
            if i % 2 == 0:
                d._get_spy_indicators = AsyncMock(return_value=bullish_ind)
            else:
                d._get_spy_indicators = AsyncMock(return_value=bearish_ind)
            t += timedelta(minutes=5)
            await d.update(None, now_et=t)

        assert len(d._history) <= 50
