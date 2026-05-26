"""Full scoring breakdown to scoring_audit table (always runs)."""

from __future__ import annotations

import json
from datetime import datetime

from loguru import logger

from options_owl.sourcing import db
from options_owl.sourcing.scoring.types import SignalContext


async def log_audit(ctx: SignalContext, scan_duration_ms: int = 0) -> None:
    """Log full scoring context to audit table, regardless of pass/fail."""
    try:
        reasons = []
        if ctx.tier1_direction:
            reasons.extend(ctx.tier1_direction.reasons)
        if ctx.tier2_timing:
            reasons.extend(ctx.tier2_timing.reasons)
        if ctx.tier3_amplifiers:
            reasons.extend(ctx.tier3_amplifiers.reasons)
        if ctx.tier4_risk:
            reasons.extend(ctx.tier4_risk.reasons)

        await db.execute(
            """
            INSERT INTO scoring_audit (
                scan_time, ticker, direction, score_total,
                score_direction, score_timing, score_amplifiers,
                score_adjustments, score_calibration,
                reasons, filter_result, filter_reason,
                options_strike, options_premium, options_spread_pct,
                scan_duration_ms
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9,
                $10, $11, $12, $13, $14, $15, $16
            )
            """,
            datetime.fromisoformat(ctx.scan_time) if ctx.scan_time else datetime.utcnow(),
            ctx.ticker,
            ctx.direction.value if ctx.direction else None,
            ctx.score_total,
            ctx.tier1_direction.total if ctx.tier1_direction else None,
            ctx.tier2_timing.total if ctx.tier2_timing else None,
            ctx.tier3_amplifiers.total if ctx.tier3_amplifiers else None,
            ctx.tier4_risk.total if ctx.tier4_risk else None,
            ctx.tier5_calibration.total if ctx.tier5_calibration else None,
            json.dumps(reasons),
            ctx.filter_result or None,
            ctx.filter_reason or None,
            ctx.strike,
            ctx.premium,
            ctx.spread_pct,
            scan_duration_ms,
        )
    except Exception:
        logger.exception(f"Failed to write audit for {ctx.ticker}")
