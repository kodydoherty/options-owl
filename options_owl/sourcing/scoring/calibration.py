"""Tier 5: Calibration (-8 to 18 points).

Context bonuses and penalties: time-of-day, day-of-week, session activity,
losing-streak regime detection, and daily bias alignment.

Time-of-day scoring is based on Simpsons agent backtest data (459 callouts)
which shows dramatic win-rate differences by session window.  The early
afternoon (1:30-3:00 PM ET, 240-330 min since open) has only 36% WR and
now carries a penalty instead of bonus points.
"""

from __future__ import annotations

from datetime import datetime

from zoneinfo import ZoneInfo

from options_owl.sourcing.scoring.types import Direction, SignalContext, TierResult

ET = ZoneInfo("America/New_York")


def tier5_calibration(ctx: SignalContext) -> TierResult:
    """Apply calibration bonuses based on market context.

    Sub-signals:
        - Time-of-day bonus/penalty: -5 to 5
        - Power hour / opening drive: 0-5
        - Day-of-week tendency: 0-3
        - Gap analysis: 0-2
        - Losing streak regime: -3 to 0
        - Daily bias alignment: 0-3
    Range: -8 to 18 points.
    """
    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []

    now_et = datetime.now(tz=ET)
    if ctx.scan_time:
        try:
            now_et = datetime.fromisoformat(ctx.scan_time).astimezone(ET)
        except (ValueError, TypeError):
            pass

    minutes_since_open = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)

    # --- Time-of-day (-5 to 5) ---
    tod_pts = _score_time_of_day(minutes_since_open)
    components["time_of_day"] = tod_pts
    total += tod_pts
    if tod_pts >= 4:
        reasons.append("favorable_time_window")
    if tod_pts < 0:
        reasons.append("danger_zone_afternoon")

    # --- Opening drive / power hour (0-5) ---
    session_pts = _score_session(minutes_since_open)
    components["session"] = session_pts
    total += session_pts
    if session_pts >= 4:
        reasons.append("high_activity_session")

    # --- Day-of-week (0-3) ---
    dow_pts = _score_day_of_week(now_et.weekday())
    components["day_of_week"] = dow_pts
    total += dow_pts

    # --- Gap analysis placeholder (0-2) ---
    gap_pts = 0  # requires open price comparison to prev close
    components["gap"] = gap_pts
    total += gap_pts

    # --- Losing streak regime (-3 to 0) ---
    streak_pts = _score_losing_streak(ctx)
    components["losing_streak"] = streak_pts
    total += streak_pts
    if streak_pts <= -3:
        reasons.append("losing_streak_regime")

    # --- Daily bias alignment (0-3) ---
    bias_pts = _score_daily_bias(ctx)
    components["daily_bias"] = bias_pts
    total += bias_pts
    if bias_pts >= 3:
        reasons.append("daily_bias_aligned")

    result = TierResult(total=total, max_possible=18, components=components, reasons=reasons)
    ctx.tier5_calibration = result
    return result


def _score_time_of_day(minutes_since_open: int) -> int:
    """Score time-of-day window for 0DTE options: -5 to 5 points.

    Based on Simpsons agent backtest data (459 callouts):
      NY Open Killzone (0-60min):    72% WR -> 5 pts
      Late Morning (60-150min):      63% WR -> 3 pts
      Midday (150-240min):           55% WR -> 0 pts (neutral)
      Early Afternoon (240-330min):  36% WR -> -5 pts (PENALTY)
      Power Hour (330-390min):       58% WR -> 1 pt
    """
    if minutes_since_open < 0:
        return 0
    if minutes_since_open <= 60:
        return 5
    if minutes_since_open <= 150:
        return 3
    if minutes_since_open <= 240:
        return 0
    if minutes_since_open <= 330:
        return -5
    if minutes_since_open <= 390:
        return 1
    return 0


def _score_session(minutes_since_open: int) -> int:
    """Score opening drive and power hour: 0-5 points.

    Returns 0 during the early afternoon danger zone (240-330min)
    to avoid stacking any bonus on top of the time_of_day penalty.
    """
    if 240 <= minutes_since_open <= 330:
        return 0
    if 0 <= minutes_since_open <= 15:
        return 5
    if 15 < minutes_since_open <= 45:
        return 4
    if 330 <= minutes_since_open <= 390:
        return 4
    if 300 <= minutes_since_open <= 330:
        return 3
    return 1


def _score_day_of_week(weekday: int) -> int:
    """Score day-of-week tendency: 0-3 points.

    Monday: tricky (gap risk from weekend), 1 pt.
    Tuesday-Thursday: prime trading days, 3 pts.
    Friday: 0DTE golden day (all tickers have options), 2 pts.
    """
    if weekday == 0:
        return 1
    if 1 <= weekday <= 3:
        return 3
    if weekday == 4:
        return 2
    return 0


def _score_losing_streak(ctx: SignalContext) -> int:
    """Score recent losing streak regime: -3 to 0.

    Checks ctx.recent_signal_outcomes for consecutive losses in
    the last 2 hours.  Each entry is a dict with at least
    ``{"outcome": "win"|"loss", "closed_at": "ISO-timestamp"}``.

    - 3+ consecutive losses: -3 (regime veto)
    - 2 consecutive losses:  -1
    - Otherwise:              0
    """
    if not ctx.recent_signal_outcomes:
        return 0

    now_et = datetime.now(tz=ET)
    if ctx.scan_time:
        try:
            now_et = datetime.fromisoformat(ctx.scan_time).astimezone(ET)
        except (ValueError, TypeError):
            pass

    # Filter to last 2 hours and sort newest-first
    recent: list[dict] = []
    for entry in ctx.recent_signal_outcomes:
        closed_at = entry.get("closed_at")
        if closed_at:
            try:
                ts = datetime.fromisoformat(closed_at).astimezone(ET)
                diff = (now_et - ts).total_seconds()
                if 0 <= diff <= 7200:  # last 2 hours
                    recent.append(entry)
            except (ValueError, TypeError):
                continue

    if not recent:
        return 0

    # Sort newest-first by closed_at
    try:
        recent.sort(
            key=lambda e: datetime.fromisoformat(e["closed_at"]),
            reverse=True,
        )
    except (KeyError, ValueError, TypeError):
        return 0

    # Count consecutive losses from the most recent trade
    consecutive_losses = 0
    for entry in recent:
        if entry.get("outcome") == "loss":
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= 3:
        return -3
    if consecutive_losses >= 2:
        return -1
    return 0


def _score_daily_bias(ctx: SignalContext) -> int:
    """Score daily bias alignment: 0-3 points.

    Checks whether the signal direction aligns with the daily trend
    using EMA200 position and ADX from ctx.indicators.

    - Strong trend (ADX >= 25) AND direction aligned with EMA200: +3
    - Moderate alignment (ADX >= 15 or price on correct side of EMA200): +1
    - Counter-trend or no data: 0
    """
    if ctx.indicators is None or ctx.direction is None:
        return 0

    indicators = ctx.indicators
    ema200 = getattr(indicators, "ema200", None)
    adx = getattr(indicators, "adx", None)
    last_close = getattr(indicators, "last_close", None)

    if ema200 is None or last_close is None or ema200 == 0:
        return 0

    # Determine if price is above or below EMA200
    price_above_ema200 = last_close > ema200

    # Direction alignment: CALL wants price above EMA200, PUT wants below
    is_aligned = (
        (ctx.direction == Direction.CALL and price_above_ema200)
        or (ctx.direction == Direction.PUT and not price_above_ema200)
    )

    if not is_aligned:
        return 0

    # Check trend strength via ADX
    if adx is not None and adx >= 25:
        return 3
    if adx is not None and adx >= 15:
        return 1
    # Price on correct side of EMA200 but weak ADX
    return 1
