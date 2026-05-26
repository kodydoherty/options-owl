"""Tier 2: Timing Quality (0-30 points).

Is THIS the right moment to enter, or are we late/early?
Uses only indicators from the IndicatorSet — no I/O.
"""

from __future__ import annotations

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.types import Direction, SignalContext, TierResult


def tier2_timing(ctx: SignalContext) -> TierResult:
    """Evaluate timing quality from indicators.

    Sub-signals:
        - Volume confirmation: 0-10 (mandatory min 3/10 for any signal)
        - RSI positioning: 0-5
        - MACD histogram momentum: 0-5
        - Bollinger/squeeze setup: 0-5
        - ATR volatility regime: 0-5
    Max: 30 points.
    """
    ind: IndicatorSet | None = ctx.indicators
    if ind is None:
        return TierResult(total=0, max_possible=30, reasons=["no_indicators"])

    direction = ctx.direction
    is_call = direction == Direction.CALL if direction else True

    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []

    # --- Volume confirmation (0-10) [MANDATORY: min 3] ---
    vol_pts = _score_volume(ind)
    components["volume"] = vol_pts
    total += vol_pts
    if vol_pts >= 7:
        reasons.append("strong_volume")
    elif vol_pts < 3:
        reasons.append("insufficient_volume")

    # --- RSI positioning (0-5) ---
    rsi_pts = _score_rsi(ind, is_call)
    components["rsi"] = rsi_pts
    total += rsi_pts
    if rsi_pts >= 4:
        reasons.append("rsi_favorable")

    # --- MACD histogram momentum (0-5) ---
    macd_pts = _score_macd_momentum(ind, is_call)
    components["macd_momentum"] = macd_pts
    total += macd_pts
    if macd_pts >= 4:
        reasons.append("macd_accelerating")

    # --- Bollinger/squeeze setup (0-5) ---
    bb_pts = _score_bollinger(ind, is_call)
    components["bollinger"] = bb_pts
    total += bb_pts
    if bb_pts >= 4:
        reasons.append("squeeze_breakout" if ind.bb_squeeze else "bb_favorable")

    # --- ATR volatility regime (0-5) ---
    atr_pts = _score_atr_regime(ind)
    components["atr_regime"] = atr_pts
    total += atr_pts
    if atr_pts >= 4:
        reasons.append("expanding_volatility")

    result = TierResult(total=total, max_possible=30, components=components, reasons=reasons)
    ctx.tier2_timing = result
    return result


def _score_volume(ind: IndicatorSet) -> int:
    """Score volume confirmation: 0-10 points.

    volume_ratio = current / 20-bar avg. Higher = more conviction.
    OBV slope confirms volume is flowing in the right direction.
    """
    ratio = ind.volume_ratio
    obv = ind.obv_slope

    if ratio >= 2.0:
        pts = 8
    elif ratio >= 1.5:
        pts = 6
    elif ratio >= 1.0:
        pts = 4
    elif ratio >= 0.7:
        pts = 2
    else:
        pts = 1  # very low volume

    # OBV slope bonus
    if abs(obv) > 0.1:
        pts += 2

    return min(10, pts)


def _score_rsi(ind: IndicatorSet, is_call: bool) -> int:
    """Score RSI positioning: 0-5 points.

    CALL: RSI 40-70 = healthy uptrend. RSI > 80 = overbought (risky entry).
    PUT: RSI 30-60 = healthy downtrend. RSI < 20 = oversold (risky entry).
    """
    rsi = ind.rsi9

    if is_call:
        if 50 <= rsi <= 65:
            return 5  # sweet spot: trending up, not overextended
        if 40 <= rsi <= 70:
            return 3
        if rsi > 80:
            return 0  # overbought, poor timing
        if rsi < 30:
            return 1  # oversold bounce possible but risky for CALL entry
        return 2
    else:
        if 35 <= rsi <= 50:
            return 5  # sweet spot: trending down, not oversold
        if 30 <= rsi <= 60:
            return 3
        if rsi < 20:
            return 0  # oversold, poor timing for PUT
        if rsi > 70:
            return 1  # overbought reversal possible but risky for PUT entry
        return 2


def _score_macd_momentum(ind: IndicatorSet, is_call: bool) -> int:
    """Score MACD histogram momentum: 0-5 points.

    Histogram getting more positive/negative = accelerating momentum.
    """
    hist = ind.macd_histogram

    if is_call:
        if hist > 0.3:
            return 5
        if hist > 0.1:
            return 3
        if hist > 0:
            return 2
        return 0
    else:
        if hist < -0.3:
            return 5
        if hist < -0.1:
            return 3
        if hist < 0:
            return 2
        return 0


def _score_bollinger(ind: IndicatorSet, is_call: bool) -> int:
    """Score Bollinger Band setup: 0-5 points.

    Squeeze (BB inside Keltner) = energy building, breakout imminent.
    Price near band edge in direction = momentum.
    """
    pts = 0

    # Squeeze bonus: energy building
    if ind.bb_squeeze:
        pts += 3

    # Band position
    if ind.bb_upper > 0 and ind.bb_lower > 0 and ind.last_close > 0:
        bb_range = ind.bb_upper - ind.bb_lower
        if bb_range > 0:
            position = (ind.last_close - ind.bb_lower) / bb_range  # 0=lower, 1=upper

            if is_call and position > 0.7:
                pts += 2  # near upper band, momentum
            elif is_call and 0.4 <= position <= 0.6:
                pts += 1  # middle, neutral
            elif not is_call and position < 0.3:
                pts += 2  # near lower band, momentum
            elif not is_call and 0.4 <= position <= 0.6:
                pts += 1

    return min(5, pts)


def _score_atr_regime(ind: IndicatorSet) -> int:
    """Score ATR volatility regime: 0-5 points.

    Expanding ATR = market is moving, options premiums justified.
    Contracting ATR = quiet market, harder to profit from 0DTE.
    """
    if ind.atr_expanding:
        pts = 4
        if ind.atr14 > 0 and ind.last_close > 0:
            atr_pct = ind.atr14 / ind.last_close
            if atr_pct > 0.015:  # >1.5% ATR = active market
                pts = 5
        return pts

    # Not expanding but still reasonable
    if ind.atr14 > 0 and ind.last_close > 0:
        atr_pct = ind.atr14 / ind.last_close
        if atr_pct > 0.01:
            return 2
    return 1
