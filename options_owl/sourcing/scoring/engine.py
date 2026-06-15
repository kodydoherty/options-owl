"""Scoring engine: orchestrate all tiers, produce final 0-100 score + direction."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import (
    ScoredSignal,
    SignalContext,
)


def compute_score(ctx: SignalContext) -> ScoredSignal:
    """Run all 5 scoring tiers and produce a final 0-100 score.

    Each tier is a pure function that reads indicators from the context.
    """
    from options_owl.sourcing.scoring.direction import tier1_direction
    from options_owl.sourcing.scoring.timing import tier2_timing
    from options_owl.sourcing.scoring.amplifiers import tier3_amplifiers
    from options_owl.sourcing.scoring.adjustments import tier4_risk
    from options_owl.sourcing.scoring.calibration import tier5_calibration

    t1 = tier1_direction(ctx)
    t2 = tier2_timing(ctx)
    t3 = tier3_amplifiers(ctx)
    t4 = tier4_risk(ctx)
    t5 = tier5_calibration(ctx)

    # Mandatory volume gate
    volume_pts = t2.components.get("volume", 0)
    if volume_pts < 3:
        return ScoredSignal(
            score=0,
            direction=ctx.direction,
            rejected=True,
            reject_reason="insufficient_volume",
            breakdown={"direction": t1, "timing": t2, "amplifiers": t3, "risk": t4, "calibration": t5},
        )

    raw = t1.total + t2.total + t3.total + t4.total + t5.total
    score = max(0, min(100, raw))

    return ScoredSignal(
        score=score,
        direction=ctx.direction,
        breakdown={"direction": t1, "timing": t2, "amplifiers": t3, "risk": t4, "calibration": t5},
    )
