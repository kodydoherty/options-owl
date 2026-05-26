"""Post-scoring critical penalty combo veto.

Catches dangerous combos that the scoring tiers individually allow
but together signal a bad trade. Pure function — no I/O.
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


def check_penalty_veto(ctx: SignalContext) -> bool:
    """Returns True if signal should be VETOED (blocked).

    Veto conditions (any one triggers block):
    1. RSI overextended AND low ADX (chasing in choppy market)
    2. Wide spread AND low volume (illiquid, will get slipped)
    3. Direction score < 10 AND risk adjustment heavy (no conviction + penalties)
    """
    if ctx.tier4_risk is None:
        return False

    risk = ctx.tier4_risk.components

    # Combo 1: RSI overextended + choppy market
    rsi_penalty = risk.get("rsi_overextend", 0)
    adx_penalty = risk.get("low_adx", 0)
    if rsi_penalty <= -3 and adx_penalty <= -2:
        ctx.filter_result = "vetoed"
        ctx.filter_reason = "veto: overextended_rsi + choppy_market"
        return True

    # Combo 2: Wide spread + no volume
    spread_penalty = risk.get("spread", 0)
    vol_score = 0
    if ctx.tier2_timing:
        vol_score = ctx.tier2_timing.components.get("volume", 0)
    if spread_penalty <= -2 and vol_score < 4:
        ctx.filter_result = "vetoed"
        ctx.filter_reason = "veto: wide_spread + low_volume (illiquid)"
        return True

    # Combo 3: No directional conviction + heavy penalties
    dir_score = ctx.tier1_direction.total if ctx.tier1_direction else 0
    if dir_score < 10 and ctx.tier4_risk.total <= -5:
        ctx.filter_result = "vetoed"
        ctx.filter_reason = "veto: weak_direction + heavy_penalties"
        return True

    return False
