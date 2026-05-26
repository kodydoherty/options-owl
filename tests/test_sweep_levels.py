"""Tests for institutional levels (PDH/PDL, PWH/PWL) and sweep detection."""

from __future__ import annotations

import pytest

from options_owl.sourcing.data.indicator_engine import (
    BARS_PER_DAY,
    IndicatorSet,
    _split_into_days,
    calc_institutional_levels,
    compute_indicators,
    detect_sweeps,
)
from options_owl.sourcing.scoring.amplifiers import _score_sweep_levels, tier3_amplifiers
from options_owl.sourcing.scoring.types import Direction, SignalContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    open_: float, high: float, low: float, close: float,
    volume: float = 1000, timestamp: str | None = None,
) -> dict:
    bar = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    if timestamp is not None:
        bar["timestamp"] = timestamp
    return bar


def _make_day_bars(
    base_price: float, n: int = BARS_PER_DAY,
    day_high_offset: float = 2.0, day_low_offset: float = 2.0,
    timestamp_prefix: str | None = None,
) -> list[dict]:
    """Generate n bars centered around base_price for one 'day'."""
    bars = []
    for i in range(n):
        h = base_price + day_high_offset * ((i + 1) / n)
        l = base_price - day_low_offset * ((i + 1) / n)
        c = base_price + (day_high_offset - day_low_offset) * 0.3
        ts = f"{timestamp_prefix}T{9 + i // 12}:{(i % 12) * 5:02d}:00" if timestamp_prefix else None
        bars.append(_make_bar(base_price, h, l, c, 1000, timestamp=ts))
    return bars


# ---------------------------------------------------------------------------
# IndicatorSet new fields
# ---------------------------------------------------------------------------


class TestIndicatorSetFields:
    def test_new_fields_exist_with_defaults(self):
        ind = IndicatorSet()
        assert ind.pdh == 0.0
        assert ind.pdl == 0.0
        assert ind.pwh == 0.0
        assert ind.pwl == 0.0
        assert ind.session_high == 0.0
        assert ind.session_low == 0.0
        assert ind.sweep_pdh is False
        assert ind.sweep_pdl is False
        assert ind.sweep_pwh is False
        assert ind.sweep_pwl is False
        assert ind.sweep_session_high is False
        assert ind.sweep_session_low is False

    def test_fields_settable(self):
        ind = IndicatorSet(pdh=150.0, sweep_pdl=True)
        assert ind.pdh == 150.0
        assert ind.sweep_pdl is True


# ---------------------------------------------------------------------------
# _split_into_days
# ---------------------------------------------------------------------------


class TestSplitIntoDays:
    def test_empty(self):
        assert _split_into_days([]) == []

    def test_heuristic_split(self):
        bars = [_make_bar(100, 101, 99, 100) for _ in range(BARS_PER_DAY * 2 + 10)]
        days = _split_into_days(bars)
        assert len(days) == 3
        assert len(days[0]) == BARS_PER_DAY
        assert len(days[1]) == BARS_PER_DAY
        assert len(days[2]) == 10

    def test_timestamp_split(self):
        bars = (
            [_make_bar(100, 101, 99, 100, timestamp="2026-05-19T10:00:00") for _ in range(5)]
            + [_make_bar(100, 101, 99, 100, timestamp="2026-05-20T10:00:00") for _ in range(3)]
        )
        days = _split_into_days(bars)
        assert len(days) == 2
        assert len(days[0]) == 5
        assert len(days[1]) == 3

    def test_epoch_timestamp_split(self):
        # Epoch ms for two different days
        epoch_day1 = 1716100800000  # some day
        epoch_day2 = epoch_day1 + 86400 * 1000  # next day
        bars = [
            _make_bar(100, 101, 99, 100, timestamp=epoch_day1),
            _make_bar(100, 101, 99, 100, timestamp=epoch_day2),
        ]
        days = _split_into_days(bars)
        assert len(days) == 2


# ---------------------------------------------------------------------------
# calc_institutional_levels
# ---------------------------------------------------------------------------


class TestCalcInstitutionalLevels:
    def test_empty_candles(self):
        levels = calc_institutional_levels([])
        assert levels["pdh"] == 0.0
        assert levels["session_high"] == 0.0

    def test_single_day_no_pdh(self):
        bars = _make_day_bars(100.0, n=BARS_PER_DAY)
        levels = calc_institutional_levels(bars)
        # Only one day, so no previous day
        assert levels["pdh"] == 0.0
        assert levels["pdl"] == 0.0
        assert levels["session_high"] > 0.0
        assert levels["session_low"] > 0.0

    def test_two_days_pdh_pdl(self):
        day1 = _make_day_bars(100.0, n=BARS_PER_DAY, day_high_offset=5.0, day_low_offset=3.0)
        day2 = _make_day_bars(110.0, n=BARS_PER_DAY, day_high_offset=2.0, day_low_offset=2.0)
        bars = day1 + day2
        levels = calc_institutional_levels(bars)
        # PDH = max high of day1
        assert levels["pdh"] == max(c["high"] for c in day1)
        assert levels["pdl"] == min(c["low"] for c in day1)
        assert levels["session_high"] == max(c["high"] for c in day2)
        assert levels["session_low"] == min(c["low"] for c in day2)

    def test_pwh_pwl_multiple_days(self):
        days = []
        for i in range(6):
            days.extend(_make_day_bars(
                100.0 + i * 10, n=BARS_PER_DAY,
                day_high_offset=5.0, day_low_offset=3.0,
            ))
        levels = calc_institutional_levels(days)
        # PWH/PWL should cover days 1-5 (previous 5 days, excluding today which is day 6)
        assert levels["pwh"] > 0.0
        assert levels["pwl"] > 0.0
        # PWH should be the max high from previous 5 days (days 1-5)
        # Day 5 (index 4) has highest base: 140, same as PDH since it IS the prev day
        assert levels["pwh"] >= levels["pdh"]  # weekly includes prev day + more

    def test_timestamp_based_levels(self):
        day1 = [_make_bar(100, 110, 90, 100, timestamp="2026-05-19T10:00:00") for _ in range(5)]
        day2 = [_make_bar(105, 115, 95, 105, timestamp="2026-05-20T10:00:00") for _ in range(5)]
        levels = calc_institutional_levels(day1 + day2)
        assert levels["pdh"] == 110.0
        assert levels["pdl"] == 90.0
        assert levels["session_high"] == 115.0
        assert levels["session_low"] == 95.0


# ---------------------------------------------------------------------------
# detect_sweeps
# ---------------------------------------------------------------------------


class TestDetectSweeps:
    def test_no_sweep(self):
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 104.5, "session_low": 95.5}
        # All bars within levels — no sweep
        bars = [_make_bar(100, 104, 96, 100) for _ in range(5)]
        sweeps = detect_sweeps(bars, levels)
        assert not any(sweeps.values())

    def test_sweep_pdh(self):
        """High goes above PDH but close comes back below."""
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(103, 106, 102, 104),  # high > 105, close < 105 → sweep!
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_pdh"] is True
        assert sweeps["sweep_pdl"] is False

    def test_sweep_pdl(self):
        """Low goes below PDL but close comes back above."""
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(100, 100, 94, 96),  # low < 95, close > 95 → sweep!
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_pdl"] is True

    def test_sweep_pwh(self):
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(108, 111, 107, 109),  # high > 110, close < 110
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_pwh"] is True

    def test_sweep_pwl(self):
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(92, 92, 89, 91),  # low < 90, close > 90
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_pwl"] is True

    def test_sweep_session_high(self):
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(102, 104, 101, 102),  # high > 103, close < 103
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_session_high"] is True

    def test_sweep_session_low(self):
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [
            _make_bar(100, 100, 99, 100),
            _make_bar(98, 98, 96, 98),  # low < 97, close > 97
        ]
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_session_low"] is True

    def test_not_a_sweep_when_close_stays_beyond(self):
        """If price goes above PDH and STAYS above, that's not a sweep."""
        levels = {"pdh": 105.0, "pdl": 95.0, "pwh": 110.0, "pwl": 90.0,
                  "session_high": 103.0, "session_low": 97.0}
        bars = [_make_bar(104, 107, 104, 106)]  # high > 105, close > 105 → NOT a sweep
        sweeps = detect_sweeps(bars, levels)
        assert sweeps["sweep_pdh"] is False

    def test_empty_candles(self):
        sweeps = detect_sweeps([], {"pdh": 100.0})
        assert not any(sweeps.values())

    def test_zero_levels_no_false_sweeps(self):
        """Levels at 0.0 should not trigger sweeps."""
        levels = {"pdh": 0.0, "pdl": 0.0, "pwh": 0.0, "pwl": 0.0,
                  "session_high": 0.0, "session_low": 0.0}
        bars = [_make_bar(100, 105, 95, 100)]
        sweeps = detect_sweeps(bars, levels)
        assert not any(sweeps.values())


# ---------------------------------------------------------------------------
# _score_sweep_levels (amplifier scoring)
# ---------------------------------------------------------------------------


class TestScoreSweepLevels:
    def test_call_pdl_sweep_5pts(self):
        ind = IndicatorSet(sweep_pdl=True)
        assert _score_sweep_levels(ind, is_call=True) == 5

    def test_call_pwl_sweep_5pts(self):
        ind = IndicatorSet(sweep_pwl=True)
        assert _score_sweep_levels(ind, is_call=True) == 5

    def test_put_pdh_sweep_5pts(self):
        ind = IndicatorSet(sweep_pdh=True)
        assert _score_sweep_levels(ind, is_call=False) == 5

    def test_put_pwh_sweep_5pts(self):
        ind = IndicatorSet(sweep_pwh=True)
        assert _score_sweep_levels(ind, is_call=False) == 5

    def test_call_session_low_3pts(self):
        ind = IndicatorSet(sweep_session_low=True)
        assert _score_sweep_levels(ind, is_call=True) == 3

    def test_put_session_high_3pts(self):
        ind = IndicatorSet(sweep_session_high=True)
        assert _score_sweep_levels(ind, is_call=False) == 3

    def test_no_sweep_0pts(self):
        ind = IndicatorSet()
        assert _score_sweep_levels(ind, is_call=True) == 0
        assert _score_sweep_levels(ind, is_call=False) == 0

    def test_wrong_direction_0pts(self):
        """CALL with PDH sweep (bearish) should score 0."""
        ind = IndicatorSet(sweep_pdh=True)
        assert _score_sweep_levels(ind, is_call=True) == 0

    def test_pdl_takes_precedence_over_session(self):
        """PDL sweep (5pts) should win over session_low (3pts)."""
        ind = IndicatorSet(sweep_pdl=True, sweep_session_low=True)
        assert _score_sweep_levels(ind, is_call=True) == 5


# ---------------------------------------------------------------------------
# tier3_amplifiers max_possible updated
# ---------------------------------------------------------------------------


class TestTier3MaxPossible:
    def test_max_possible_is_20(self):
        ctx = SignalContext(direction=Direction.CALL)
        result = tier3_amplifiers(ctx)
        assert result.max_possible == 20

    def test_no_indicators_max_20(self):
        ctx = SignalContext()
        result = tier3_amplifiers(ctx)
        assert result.max_possible == 20
        assert "no_indicators" in result.reasons


# ---------------------------------------------------------------------------
# Integration: compute_indicators populates sweep fields
# ---------------------------------------------------------------------------


class TestComputeIndicatorsIntegration:
    def test_indicators_include_levels(self):
        """compute_indicators with 2+ days of data sets PDH/PDL."""
        day1 = _make_day_bars(100.0, n=BARS_PER_DAY, day_high_offset=5.0, day_low_offset=3.0)
        day2 = _make_day_bars(105.0, n=BARS_PER_DAY, day_high_offset=2.0, day_low_offset=2.0)
        ind = compute_indicators(day1 + day2)
        assert ind.pdh > 0.0
        assert ind.pdl > 0.0
        assert ind.session_high > 0.0
        assert ind.session_low > 0.0

    def test_indicators_sweep_detected(self):
        """Build candles where last bar sweeps PDL then reverses."""
        day1 = [_make_bar(100, 105, 95, 100) for _ in range(BARS_PER_DAY)]
        # Day 2: normal bars + last bar sweeps below PDL (95) then closes above
        day2_normal = [_make_bar(100, 102, 98, 100) for _ in range(BARS_PER_DAY - 1)]
        sweep_bar = _make_bar(97, 98, 94, 96)  # low=94 < pdl=95, close=96 > pdl=95
        day2 = day2_normal + [sweep_bar]
        ind = compute_indicators(day1 + day2)
        assert ind.sweep_pdl is True
