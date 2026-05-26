"""Tier 3: Amplifiers (0-15 points).

Bonus signals that increase confidence: squeeze breakout, insider flow, sentiment.
"""

from __future__ import annotations

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.types import Direction, SignalContext, TierResult


def tier3_amplifiers(ctx: SignalContext) -> TierResult:
    """Evaluate amplifier signals.

    Sub-signals:
        - Squeeze breakout setup: 0-5
        - OBV divergence/confirmation: 0-3
        - Multi-timeframe momentum: 0-3
        - Alpha source bonus (insider/congress/sentiment): 0-4
        - Institutional sweep levels: 0-5
    Max: 20 points.
    """
    ind: IndicatorSet | None = ctx.indicators
    if ind is None:
        return TierResult(total=0, max_possible=20, reasons=["no_indicators"])

    is_call = ctx.direction == Direction.CALL if ctx.direction else True
    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []

    # --- Squeeze breakout (0-5) ---
    sq_pts = _score_squeeze(ind, is_call)
    components["squeeze"] = sq_pts
    total += sq_pts
    if sq_pts >= 4:
        reasons.append("squeeze_firing")

    # --- OBV confirmation (0-3) ---
    obv_pts = _score_obv(ind, is_call)
    components["obv"] = obv_pts
    total += obv_pts
    if obv_pts >= 2:
        reasons.append("obv_confirming")

    # --- Multi-TF alignment (0-3) ---
    mtf_pts = _score_multi_tf(ctx)
    components["multi_tf"] = mtf_pts
    total += mtf_pts
    if mtf_pts >= 2:
        reasons.append("multi_tf_aligned")

    # --- Alpha source bonus (0-4) ---
    alpha_pts = _score_alpha(ctx)
    components["alpha"] = alpha_pts
    total += alpha_pts
    if alpha_pts >= 2:
        reasons.append("alpha_source_confirming")

    # --- Institutional sweep levels (0-5) ---
    sweep_pts = _score_sweep_levels(ind, is_call)
    components["sweep"] = sweep_pts
    total += sweep_pts
    if sweep_pts >= 3:
        reasons.append("sweep_level_detected")

    result = TierResult(total=total, max_possible=20, components=components, reasons=reasons)
    ctx.tier3_amplifiers = result
    return result


def _score_squeeze(ind: IndicatorSet, is_call: bool) -> int:
    """Bollinger squeeze with directional breakout: 0-5."""
    if not ind.bb_squeeze:
        return 0

    # Squeeze detected — check if breaking out in direction
    if ind.bb_upper > 0 and ind.bb_lower > 0 and ind.last_close > 0:
        bb_range = ind.bb_upper - ind.bb_lower
        if bb_range > 0:
            position = (ind.last_close - ind.bb_lower) / bb_range
            if is_call and position > 0.6:
                return 5  # squeeze + breaking upward
            if not is_call and position < 0.4:
                return 5  # squeeze + breaking downward
            return 3  # squeeze but no directional breakout yet

    return 2  # squeeze detected but can't determine position


def _score_obv(ind: IndicatorSet, is_call: bool) -> int:
    """OBV slope confirmation: 0-3."""
    slope = ind.obv_slope
    if is_call and slope > 0.1:
        return 3
    if not is_call and slope < -0.1:
        return 3
    if abs(slope) < 0.05:
        return 1  # neutral
    return 0  # OBV against direction


def _score_multi_tf(ctx: SignalContext) -> int:
    """Multi-timeframe alignment bonus: 0-3.

    If 15m candles are available, check if direction aligns with 5m.
    """
    if ctx.candles_15m is None or len(ctx.candles_15m) < 5:
        return 1  # no 15m data, give a small default

    c15 = ctx.candles_15m
    recent = [bar["close"] for bar in c15[-3:]]
    if len(recent) < 3:
        return 1

    trending_up = recent[-1] > recent[0]
    is_call = ctx.direction == Direction.CALL if ctx.direction else True

    if is_call and trending_up:
        return 3
    if not is_call and not trending_up:
        return 3
    return 0  # 15m against 5m direction


def _score_alpha(ctx: SignalContext) -> int:
    """Alpha source bonus from insider/congress/sentiment: 0-4.

    Phase 3.5 — currently returns 0 until alpha providers are wired up.
    """
    pts = 0

    if ctx.insider_activity is not None:
        pts += 2

    if ctx.congress_activity is not None:
        pts += 1

    if ctx.retail_sentiment is not None:
        pts += 1

    return min(4, pts)


def _score_sweep_levels(ind: IndicatorSet, is_call: bool) -> int:
    """Institutional sweep level scoring: 0-5.

    PDH/PDL and PWH/PWL sweeps indicate institutional manipulation reversals.
    - CALL + sweep below (PDL/PWL): 5 pts (bear trap → bullish reversal)
    - PUT + sweep above (PDH/PWH): 5 pts (bull trap → bearish reversal)
    - Session sweep in direction: 3 pts (intraday liquidity grab)
    """
    if is_call:
        # Bullish: want sweeps below key levels (bear traps)
        if ind.sweep_pdl or ind.sweep_pwl:
            return 5
        if ind.sweep_session_low:
            return 3
    else:
        # Bearish: want sweeps above key levels (bull traps)
        if ind.sweep_pdh or ind.sweep_pwh:
            return 5
        if ind.sweep_session_high:
            return 3

    return 0
