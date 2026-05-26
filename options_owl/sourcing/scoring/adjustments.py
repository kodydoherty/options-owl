"""Tier 4: Risk Adjustments (-15 to 0 points).

Penalty signals that reduce confidence: overextension, poor structure, gap risk.
"""

from __future__ import annotations

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.types import Direction, SignalContext, TierResult


def tier4_risk(ctx: SignalContext) -> TierResult:
    """Apply risk adjustment penalties.

    Sub-signals (all negative or zero):
        - RSI overextension: -5 to 0
        - Wide Bollinger (chasing extended move): -3 to 0
        - Low ADX (choppy market): -3 to 0
        - Spread penalty (wide bid-ask): -4 to 0
    Min: -15.
    """
    ind: IndicatorSet | None = ctx.indicators
    if ind is None:
        return TierResult(total=0, max_possible=0, reasons=["no_indicators"])

    is_call = ctx.direction == Direction.CALL if ctx.direction else True
    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []

    # --- RSI overextension (-5 to 0) ---
    rsi_adj = _penalize_rsi(ind, is_call)
    components["rsi_overextend"] = rsi_adj
    total += rsi_adj
    if rsi_adj < -2:
        reasons.append("rsi_overextended")

    # --- Wide BB (chasing) (-3 to 0) ---
    bb_adj = _penalize_wide_bb(ind)
    components["wide_bb"] = bb_adj
    total += bb_adj
    if bb_adj < -1:
        reasons.append("wide_bands_chasing")

    # --- Low ADX (choppy) (-3 to 0) ---
    adx_adj = _penalize_low_adx(ind)
    components["low_adx"] = adx_adj
    total += adx_adj
    if adx_adj < -1:
        reasons.append("choppy_market")

    # --- Spread penalty (-4 to 0) ---
    spread_adj = _penalize_spread(ctx)
    components["spread"] = spread_adj
    total += spread_adj
    if spread_adj < -1:
        reasons.append("wide_spread")

    total = max(-15, total)
    result = TierResult(total=total, max_possible=0, components=components, reasons=reasons)
    ctx.tier4_risk = result
    return result


def _penalize_rsi(ind: IndicatorSet, is_call: bool) -> int:
    """Penalize entries when RSI is overextended: -5 to 0."""
    rsi = ind.rsi9
    if is_call:
        if rsi > 85:
            return -5
        if rsi > 75:
            return -3
        if rsi > 70:
            return -1
    else:
        if rsi < 15:
            return -5
        if rsi < 25:
            return -3
        if rsi < 30:
            return -1
    return 0


def _penalize_wide_bb(ind: IndicatorSet) -> int:
    """Penalize wide Bollinger Bands (chasing an extended move): -3 to 0."""
    if ind.bb_width > 0.06:  # >6% width = very extended
        return -3
    if ind.bb_width > 0.04:
        return -1
    return 0


def _penalize_low_adx(ind: IndicatorSet) -> int:
    """Penalize low ADX (choppy, no trend): -3 to 0."""
    if ind.adx < 10:
        return -3
    if ind.adx < 15:
        return -2
    if ind.adx < 20:
        return -1
    return 0


def _penalize_spread(ctx: SignalContext) -> int:
    """Penalize wide option bid-ask spread: -4 to 0."""
    if ctx.spread_pct is None:
        return 0
    if ctx.spread_pct > 30:
        return -4
    if ctx.spread_pct > 20:
        return -2
    if ctx.spread_pct > 10:
        return -1
    return 0
