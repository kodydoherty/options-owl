"""Tests for Tier 5 calibration — session timing, losing streak, daily bias."""

from __future__ import annotations

from datetime import datetime, timedelta


from options_owl.sourcing.scoring.calibration import (
    _score_daily_bias,
    _score_losing_streak,
    _score_session,
    _score_time_of_day,
    tier5_calibration,
)
from options_owl.sourcing.scoring.types import Direction, SignalContext


# ---------------------------------------------------------------------------
# _score_time_of_day — Simpsons agent data
# ---------------------------------------------------------------------------

class TestScoreTimeOfDay:
    def test_pre_market_returns_zero(self):
        assert _score_time_of_day(-10) == 0

    def test_ny_open_killzone_start(self):
        """0 min since open (9:30 AM) -> 5 pts."""
        assert _score_time_of_day(0) == 5

    def test_ny_open_killzone_end(self):
        """60 min since open (10:30 AM) -> 5 pts."""
        assert _score_time_of_day(60) == 5

    def test_late_morning_start(self):
        """61 min since open -> 3 pts."""
        assert _score_time_of_day(61) == 3

    def test_late_morning_end(self):
        """150 min since open (12:00 PM) -> 3 pts."""
        assert _score_time_of_day(150) == 3

    def test_midday_neutral(self):
        """200 min since open -> 0 pts (neutral)."""
        assert _score_time_of_day(200) == 0

    def test_early_afternoon_danger_zone(self):
        """270 min since open (1:30-3:00 PM) -> -5 pts (PENALTY)."""
        assert _score_time_of_day(270) == -5

    def test_early_afternoon_start_boundary(self):
        """241 min -> danger zone."""
        assert _score_time_of_day(241) == -5

    def test_early_afternoon_end_boundary(self):
        """330 min -> still danger zone."""
        assert _score_time_of_day(330) == -5

    def test_power_hour(self):
        """331 min since open (3:01 PM) -> 1 pt."""
        assert _score_time_of_day(331) == 1

    def test_power_hour_end(self):
        """390 min (4:00 PM) -> 1 pt."""
        assert _score_time_of_day(390) == 1

    def test_after_close(self):
        """391 min -> 0 pts."""
        assert _score_time_of_day(391) == 0


# ---------------------------------------------------------------------------
# _score_session
# ---------------------------------------------------------------------------

class TestScoreSession:
    def test_opening_drive_first_15(self):
        assert _score_session(10) == 5

    def test_opening_drive_15_to_45(self):
        assert _score_session(30) == 4

    def test_power_hour_session(self):
        assert _score_session(350) == 4

    def test_afternoon_danger_zone_gives_zero(self):
        """Session bonus must NOT stack during danger zone."""
        assert _score_session(270) == 0

    def test_danger_zone_boundary_240(self):
        assert _score_session(240) == 0

    def test_danger_zone_boundary_330(self):
        assert _score_session(330) == 0

    def test_midday_returns_one(self):
        assert _score_session(200) == 1


# ---------------------------------------------------------------------------
# _score_losing_streak
# ---------------------------------------------------------------------------

def _make_outcomes(outcomes: list[str], base_time: datetime | None = None) -> list[dict]:
    """Helper: build recent_signal_outcomes list."""
    if base_time is None:
        base_time = datetime(2026, 5, 19, 14, 0, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    result = []
    for i, outcome in enumerate(outcomes):
        closed = base_time - timedelta(minutes=10 * (len(outcomes) - 1 - i))
        result.append({"outcome": outcome, "closed_at": closed.isoformat()})
    return result


class TestScoringLosingStreak:
    def test_none_outcomes_returns_zero(self):
        ctx = SignalContext(recent_signal_outcomes=None)
        assert _score_losing_streak(ctx) == 0

    def test_empty_outcomes_returns_zero(self):
        ctx = SignalContext(recent_signal_outcomes=[])
        assert _score_losing_streak(ctx) == 0

    def test_one_loss_returns_zero(self):
        ctx = SignalContext(
            scan_time="2026-05-19T14:05:00-04:00",
            recent_signal_outcomes=_make_outcomes(["loss"]),
        )
        assert _score_losing_streak(ctx) == 0

    def test_two_consecutive_losses(self):
        ctx = SignalContext(
            scan_time="2026-05-19T14:05:00-04:00",
            recent_signal_outcomes=_make_outcomes(["loss", "loss"]),
        )
        assert _score_losing_streak(ctx) == -1

    def test_three_consecutive_losses(self):
        ctx = SignalContext(
            scan_time="2026-05-19T14:05:00-04:00",
            recent_signal_outcomes=_make_outcomes(["loss", "loss", "loss"]),
        )
        assert _score_losing_streak(ctx) == -3

    def test_four_consecutive_losses_still_minus_three(self):
        ctx = SignalContext(
            scan_time="2026-05-19T14:05:00-04:00",
            recent_signal_outcomes=_make_outcomes(["loss", "loss", "loss", "loss"]),
        )
        assert _score_losing_streak(ctx) == -3

    def test_win_breaks_streak(self):
        """loss, win, loss, loss -> only 2 consecutive (most recent)."""
        ctx = SignalContext(
            scan_time="2026-05-19T14:05:00-04:00",
            recent_signal_outcomes=_make_outcomes(["loss", "win", "loss", "loss"]),
        )
        assert _score_losing_streak(ctx) == -1

    def test_old_losses_outside_2h_window_ignored(self):
        """Losses from 3 hours ago should not count."""
        from zoneinfo import ZoneInfo
        now = datetime(2026, 5, 19, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        old = now - timedelta(hours=3)
        outcomes = [
            {"outcome": "loss", "closed_at": old.isoformat()},
            {"outcome": "loss", "closed_at": old.isoformat()},
            {"outcome": "loss", "closed_at": old.isoformat()},
        ]
        ctx = SignalContext(
            scan_time=now.isoformat(),
            recent_signal_outcomes=outcomes,
        )
        assert _score_losing_streak(ctx) == 0


# ---------------------------------------------------------------------------
# _score_daily_bias
# ---------------------------------------------------------------------------

class _FakeIndicators:
    def __init__(self, ema200=100.0, adx=30.0, last_close=110.0):
        self.ema200 = ema200
        self.adx = adx
        self.last_close = last_close


class TestScoreDailyBias:
    def test_no_indicators_returns_zero(self):
        ctx = SignalContext(direction=Direction.CALL, indicators=None)
        assert _score_daily_bias(ctx) == 0

    def test_no_direction_returns_zero(self):
        ctx = SignalContext(direction=None, indicators=_FakeIndicators())
        assert _score_daily_bias(ctx) == 0

    def test_call_above_ema200_strong_adx(self):
        """CALL with price above EMA200 and ADX>=25 -> 3 pts."""
        ctx = SignalContext(
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=100, adx=30, last_close=110),
        )
        assert _score_daily_bias(ctx) == 3

    def test_put_below_ema200_strong_adx(self):
        """PUT with price below EMA200 and ADX>=25 -> 3 pts."""
        ctx = SignalContext(
            direction=Direction.PUT,
            indicators=_FakeIndicators(ema200=100, adx=25, last_close=90),
        )
        assert _score_daily_bias(ctx) == 3

    def test_call_above_ema200_moderate_adx(self):
        """CALL aligned, ADX between 15-25 -> 1 pt."""
        ctx = SignalContext(
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=100, adx=20, last_close=110),
        )
        assert _score_daily_bias(ctx) == 1

    def test_call_above_ema200_weak_adx(self):
        """CALL aligned, ADX < 15 -> 1 pt (on correct side)."""
        ctx = SignalContext(
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=100, adx=10, last_close=110),
        )
        assert _score_daily_bias(ctx) == 1

    def test_counter_trend_returns_zero(self):
        """CALL but price below EMA200 -> 0."""
        ctx = SignalContext(
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=100, adx=30, last_close=90),
        )
        assert _score_daily_bias(ctx) == 0

    def test_ema200_zero_returns_zero(self):
        ctx = SignalContext(
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=0, adx=30, last_close=110),
        )
        assert _score_daily_bias(ctx) == 0


# ---------------------------------------------------------------------------
# tier5_calibration integration — danger zone produces negative totals
# ---------------------------------------------------------------------------

class TestTier5Integration:
    def test_danger_zone_total_is_negative(self):
        """Early afternoon (2:00 PM = 270 min) with no mitigating factors -> negative."""
        ctx = SignalContext(
            scan_time="2026-05-19T14:00:00-04:00",
            direction=Direction.CALL,
            indicators=None,
            recent_signal_outcomes=None,
        )
        result = tier5_calibration(ctx)
        # time_of_day=-5, session=0, dow=3(Mon), gap=0, streak=0, bias=0 => -2
        assert result.total < 0
        assert result.components["time_of_day"] == -5
        assert result.components["session"] == 0
        assert "danger_zone_afternoon" in result.reasons

    def test_danger_zone_plus_losing_streak_very_negative(self):
        """Afternoon danger + 3-loss streak -> heavily negative."""
        from zoneinfo import ZoneInfo
        now = datetime(2026, 5, 19, 14, 30, tzinfo=ZoneInfo("America/New_York"))
        outcomes = _make_outcomes(["loss", "loss", "loss"], base_time=now - timedelta(minutes=5))
        ctx = SignalContext(
            scan_time=now.isoformat(),
            direction=Direction.CALL,
            indicators=None,
            recent_signal_outcomes=outcomes,
        )
        result = tier5_calibration(ctx)
        # time_of_day=-5, session=0, dow=3, streak=-3, bias=0 => -5
        assert result.total <= -5
        assert result.components["losing_streak"] == -3

    def test_max_possible_is_18(self):
        ctx = SignalContext()
        result = tier5_calibration(ctx)
        assert result.max_possible == 18

    def test_killzone_with_aligned_bias_high_score(self):
        """9:45 AM CALL with strong bullish trend -> high score."""
        ctx = SignalContext(
            scan_time="2026-05-20T09:45:00-04:00",  # Tuesday, 15 min since open
            direction=Direction.CALL,
            indicators=_FakeIndicators(ema200=100, adx=30, last_close=110),
            recent_signal_outcomes=None,
        )
        result = tier5_calibration(ctx)
        # time_of_day=5, session=5, dow=3, gap=0, streak=0, bias=3 => 16
        assert result.total >= 14
        assert result.components["daily_bias"] == 3
