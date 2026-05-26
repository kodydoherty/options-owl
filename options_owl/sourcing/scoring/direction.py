"""Tier 1: Direction Confidence (0-40 points).

Is the underlying actually moving in the signal's direction?
Uses only indicators from the IndicatorSet — no I/O.
"""

from __future__ import annotations

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.types import Direction, SignalContext, TierResult


def tier1_direction(ctx: SignalContext) -> TierResult:
    """Evaluate direction confidence from indicators.

    Sub-signals:
        - EMA 9/21 crossover strength: 0-15
        - VWAP position: 0-10
        - Trend regime (ADX + EMA200): 0-10
        - MACD confirmation: 0-5
    Max: 40 points.
    """
    ind: IndicatorSet | None = ctx.indicators
    if ind is None:
        return TierResult(total=0, max_possible=40, reasons=["no_indicators"])

    direction = ctx.direction
    if direction is None:
        return TierResult(total=0, max_possible=40, reasons=["no_direction"])

    is_call = direction == Direction.CALL
    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []

    # --- EMA 9/21 crossover strength (0-15) ---
    ema_pts = _score_ema_cross(ind, is_call)
    components["ema_cross"] = ema_pts
    total += ema_pts
    if ema_pts >= 10:
        reasons.append(f"strong_ema_{'bullish' if is_call else 'bearish'}")
    elif ema_pts == 0:
        reasons.append("ema_cross_against")

    # --- VWAP position (0-10) ---
    vwap_pts = _score_vwap(ind, is_call)
    components["vwap"] = vwap_pts
    total += vwap_pts
    if vwap_pts >= 7:
        reasons.append("price_favorable_vs_vwap")

    # --- Trend regime: ADX + EMA200 (0-10) ---
    trend_pts = _score_trend_regime(ind, is_call)
    components["trend_regime"] = trend_pts
    total += trend_pts
    if trend_pts >= 7:
        reasons.append("strong_trend_regime")

    # --- MACD confirmation (0-5) ---
    macd_pts = _score_macd(ind, is_call)
    components["macd"] = macd_pts
    total += macd_pts
    if macd_pts >= 4:
        reasons.append("macd_confirming")

    result = TierResult(total=total, max_possible=40, components=components, reasons=reasons)
    ctx.tier1_direction = result
    return result


def _score_ema_cross(ind: IndicatorSet, is_call: bool) -> int:
    """Score EMA 9/21 cross strength: 0-15 points.

    ema_cross_strength is (ema9 - ema21) / atr, clipped to [-1, +1].
    Positive = bullish, negative = bearish.
    """
    strength = ind.ema_cross_strength
    if is_call:
        if strength >= 0.7:
            return 15
        if strength >= 0.4:
            return 12
        if strength >= 0.1:
            return 8
        if strength >= 0:
            return 4
        return 0  # EMA cross is bearish for a CALL signal
    else:
        if strength <= -0.7:
            return 15
        if strength <= -0.4:
            return 12
        if strength <= -0.1:
            return 8
        if strength <= 0:
            return 4
        return 0  # EMA cross is bullish for a PUT signal


def _score_vwap(ind: IndicatorSet, is_call: bool) -> int:
    """Score price position relative to VWAP: 0-10 points.

    CALL: price above VWAP is bullish. PUT: price below VWAP is bearish.
    VWAP slope adds/subtracts conviction.
    """
    if ind.vwap == 0 or ind.last_close == 0:
        return 0

    pct_from_vwap = (ind.last_close - ind.vwap) / ind.vwap
    slope = ind.vwap_slope

    if is_call:
        if pct_from_vwap > 0.005:  # above VWAP by 0.5%+
            pts = 7
            if slope > 0:
                pts += 3  # rising VWAP confirms
            return min(10, pts)
        if pct_from_vwap > -0.002:  # near VWAP
            return 4
        return 0  # well below VWAP for a CALL
    else:
        if pct_from_vwap < -0.005:  # below VWAP by 0.5%+
            pts = 7
            if slope < 0:
                pts += 3  # falling VWAP confirms
            return min(10, pts)
        if pct_from_vwap < 0.002:  # near VWAP
            return 4
        return 0  # well above VWAP for a PUT


def _score_trend_regime(ind: IndicatorSet, is_call: bool) -> int:
    """Score trend regime from ADX + EMA200 position: 0-10 points.

    ADX > 25 = trending (good).
    Price above EMA200 = long-term bullish; below = bearish.
    """
    pts = 0

    # ADX component (0-5): is the market trending at all?
    if ind.adx >= 35:
        pts += 5
    elif ind.adx >= 25:
        pts += 3
    elif ind.adx >= 15:
        pts += 1
    # ADX < 15 = choppy, no points

    # EMA200 component (0-5): long-term trend alignment
    if ind.ema200 > 0 and ind.last_close > 0:
        above_200 = ind.last_close > ind.ema200
        if is_call and above_200:
            pts += 5
        elif not is_call and not above_200:
            pts += 5
        elif is_call and not above_200:
            pts += 1  # counter-trend CALL, small credit
        elif not is_call and above_200:
            pts += 1  # counter-trend PUT, small credit

    return min(10, pts)


def _score_macd(ind: IndicatorSet, is_call: bool) -> int:
    """Score MACD alignment: 0-5 points.

    CALL: positive MACD line + positive histogram = confirming.
    PUT: negative MACD line + negative histogram = confirming.
    """
    line = ind.macd_line
    hist = ind.macd_histogram

    if is_call:
        if line > 0 and hist > 0:
            return 5
        if line > 0 or hist > 0:
            return 3
        return 0
    else:
        if line < 0 and hist < 0:
            return 5
        if line < 0 or hist < 0:
            return 3
        return 0
