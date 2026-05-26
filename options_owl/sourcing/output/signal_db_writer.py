"""Write scored signals to PostgreSQL signals table.

The signals table is the handoff point between the sourcing agent
and the buy/sell agents. Trading bots poll this table for new signals.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from loguru import logger


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that json.dumps can't serialize natively."""

    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

from options_owl.sourcing import db
from options_owl.sourcing.scoring.types import SignalContext


async def emit_signal_db(ctx: SignalContext) -> int | None:
    """Write scored signal to PostgreSQL signals table.

    Returns the signal ID on success, None on failure.
    """
    try:
        # Build score breakdown for JSON storage
        breakdown = {}
        if ctx.tier1_direction:
            breakdown["direction"] = {
                "total": ctx.tier1_direction.total,
                "components": ctx.tier1_direction.components,
            }
        if ctx.tier2_timing:
            breakdown["timing"] = {
                "total": ctx.tier2_timing.total,
                "components": ctx.tier2_timing.components,
            }
        if ctx.tier3_amplifiers:
            breakdown["amplifiers"] = {
                "total": ctx.tier3_amplifiers.total,
                "components": ctx.tier3_amplifiers.components,
            }
        if ctx.tier4_risk:
            breakdown["risk"] = {
                "total": ctx.tier4_risk.total,
                "components": ctx.tier4_risk.components,
            }
        if ctx.tier5_calibration:
            breakdown["calibration"] = {
                "total": ctx.tier5_calibration.total,
                "components": ctx.tier5_calibration.components,
            }

        # Build indicators summary
        indicators_json = None
        if ctx.indicators is not None:
            ind = ctx.indicators
            indicators_json = json.dumps({
                "ema_cross_strength": round(float(getattr(ind, "ema_cross_strength", 0)), 3),
                "rsi9": round(float(getattr(ind, "rsi9", 0)), 1),
                "macd_line": round(float(getattr(ind, "macd_line", 0)), 4),
                "macd_histogram": round(float(getattr(ind, "macd_histogram", 0)), 4),
                "bb_squeeze": bool(getattr(ind, "bb_squeeze", False)),
                "atr14": round(float(getattr(ind, "atr14", 0)), 4),
                "volume_ratio": round(float(getattr(ind, "volume_ratio", 0)), 2),
                "adx": round(float(getattr(ind, "adx", 0)), 1),
                "vwap_slope": round(float(getattr(ind, "vwap_slope", 0)), 5),
            }, cls=_NumpyEncoder)

        # Collect alpha source names
        alpha_sources = []
        if ctx.insider_activity is not None:
            alpha_sources.append("sec_insider")
        if ctx.congress_activity is not None:
            alpha_sources.append("congress")
        if ctx.retail_sentiment is not None:
            alpha_sources.append("stocktwits")

        # Write to ml_signals (the table trading bots poll via signal_consumer)
        # Also write to signals (sourcing audit) for historical tracking
        emitted_at = datetime.fromisoformat(ctx.scan_time) if ctx.scan_time else datetime.utcnow()
        breakdown_json = json.dumps(breakdown, cls=_NumpyEncoder)

        signal_id = await db.fetchval(
            """
            INSERT INTO ml_signals (
                ticker, direction, score, premium, strike,
                indicators, score_breakdown, emitted_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8
            )
            RETURNING id
            """,
            ctx.ticker,
            ctx.direction.value if ctx.direction else None,
            ctx.score_total,
            ctx.premium,
            ctx.strike,
            indicators_json,
            breakdown_json,
            emitted_at,
        )

        # Also write to signals table for sourcing audit (best-effort)
        try:
            await db.execute(
                """
                INSERT INTO signals (
                    ticker, direction, score, score_breakdown,
                    strike, expiry, premium, spread_pct,
                    indicators, alpha_sources, emitted_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                """,
                ctx.ticker,
                ctx.direction.value if ctx.direction else None,
                ctx.score_total,
                breakdown_json,
                ctx.strike,
                None,
                ctx.premium,
                ctx.spread_pct,
                indicators_json,
                json.dumps(alpha_sources, cls=_NumpyEncoder) if alpha_sources else None,
                emitted_at,
            )
        except Exception:
            pass  # audit table write is non-critical

        logger.info(
            f"SIGNAL DB: #{signal_id} {ctx.ticker} {ctx.direction.value if ctx.direction else '?'} "
            f"score={ctx.score_total}"
        )
        return signal_id

    except Exception:
        logger.exception(f"Failed to write signal for {ctx.ticker}")
        return None
