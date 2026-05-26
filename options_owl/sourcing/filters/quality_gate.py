"""Quality gate: minimum score, multi-tier contribution, circuit breaker.

Pure function — reads from SignalContext, writes filter_result/filter_reason.
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


def check_quality_gate(ctx: SignalContext, threshold: int = 60) -> bool:
    """Returns True if signal passes quality gate.

    Checks:
    1. Score >= threshold
    2. At least 2 tiers contribute meaningful points (prevents single-tier flukes)
    3. Direction tier >= 10 (must have SOME directional conviction)
    """
    # Gate 1: score threshold
    if ctx.score_total < threshold:
        ctx.filter_result = "rejected"
        ctx.filter_reason = f"score {ctx.score_total} < threshold {threshold}"
        return False

    # Gate 2: multi-tier contribution
    tiers_with_signal = 0
    if ctx.tier1_direction and ctx.tier1_direction.total >= 10:
        tiers_with_signal += 1
    if ctx.tier2_timing and ctx.tier2_timing.total >= 8:
        tiers_with_signal += 1
    if ctx.tier3_amplifiers and ctx.tier3_amplifiers.total >= 3:
        tiers_with_signal += 1
    if ctx.tier5_calibration and ctx.tier5_calibration.total >= 5:
        tiers_with_signal += 1

    if tiers_with_signal < 2:
        ctx.filter_result = "rejected"
        ctx.filter_reason = f"single_tier_fluke: only {tiers_with_signal} tiers contributing"
        return False

    # Gate 3: minimum directional conviction
    if ctx.tier1_direction and ctx.tier1_direction.total < 10:
        ctx.filter_result = "rejected"
        ctx.filter_reason = f"weak_direction: tier1={ctx.tier1_direction.total}/40"
        return False

    ctx.filter_result = "passed"
    return True
