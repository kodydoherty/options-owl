"""Format and POST signal embeds to a Discord webhook.

Uses a simple httpx POST — no discord.py dependency needed.
The webhook URL is configured via SOURCING_DISCORD_WEBHOOK_URL env var.
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx
from loguru import logger

from options_owl.sourcing.scoring.types import SignalContext


async def emit_discord(ctx: SignalContext) -> bool:
    """Format signal as Discord embed and POST to webhook.

    Returns True if posted successfully, False otherwise.
    """
    webhook_url = os.getenv("SOURCING_DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        logger.debug("No SOURCING_DISCORD_WEBHOOK_URL — skipping Discord output")
        return False

    try:
        embed = _build_embed(ctx)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                webhook_url,
                json={"embeds": [embed]},
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code in (200, 204):
                logger.info(f"DISCORD: posted signal for {ctx.ticker}")
                return True
            logger.warning(f"DISCORD: webhook returned {resp.status_code}: {resp.text[:200]}")
            return False

    except Exception as exc:
        logger.warning(f"DISCORD: webhook failed for {ctx.ticker}: {exc}")
        return False


def _build_embed(ctx: SignalContext) -> dict:
    """Build a Discord embed dict from a SignalContext."""
    direction = ctx.direction.value if ctx.direction else "?"
    is_call = direction == "CALL"

    color = 0x00FF00 if is_call else 0xFF0000  # green for CALL, red for PUT

    # Build tier breakdown string
    tiers = []
    if ctx.tier1_direction:
        tiers.append(f"Direction: {ctx.tier1_direction.total}/40")
    if ctx.tier2_timing:
        tiers.append(f"Timing: {ctx.tier2_timing.total}/30")
    if ctx.tier3_amplifiers:
        tiers.append(f"Amplifiers: {ctx.tier3_amplifiers.total}/15")
    if ctx.tier4_risk:
        tiers.append(f"Risk: {ctx.tier4_risk.total}/0")
    if ctx.tier5_calibration:
        tiers.append(f"Calibration: {ctx.tier5_calibration.total}/15")

    # Build indicator summary
    ind_lines = []
    if ctx.indicators is not None:
        ind = ctx.indicators
        ind_lines.append(f"EMA Cross: {getattr(ind, 'ema_cross_strength', 0):.2f}")
        ind_lines.append(f"RSI(9): {getattr(ind, 'rsi9', 0):.1f}")
        ind_lines.append(f"MACD: {getattr(ind, 'macd_line', 0):.3f}")
        ind_lines.append(f"Vol Ratio: {getattr(ind, 'volume_ratio', 0):.1f}x")
        ind_lines.append(f"ADX: {getattr(ind, 'adx', 0):.0f}")
        if getattr(ind, "bb_squeeze", False):
            ind_lines.append("BB Squeeze: YES")

    # Collect reasons
    reasons = []
    for tier in [ctx.tier1_direction, ctx.tier2_timing, ctx.tier3_amplifiers,
                 ctx.tier4_risk, ctx.tier5_calibration]:
        if tier and tier.reasons:
            reasons.extend(tier.reasons)

    fields = [
        {"name": "Score Breakdown", "value": "\n".join(tiers) or "N/A", "inline": True},
        {"name": "Indicators", "value": "\n".join(ind_lines) or "N/A", "inline": True},
    ]

    if ctx.premium:
        fields.append({"name": "Premium", "value": f"${ctx.premium:.2f}", "inline": True})
    if ctx.strike:
        fields.append({"name": "Strike", "value": f"${ctx.strike:.2f}", "inline": True})
    if reasons:
        fields.append({"name": "Reasons", "value": ", ".join(reasons[:8]), "inline": False})

    return {
        "title": f"{'CALL' if is_call else 'PUT'} {ctx.ticker} — Score {ctx.score_total}/100",
        "color": color,
        "fields": fields,
        "footer": {"text": "owlet-sourcing"},
        "timestamp": ctx.scan_time or datetime.utcnow().isoformat(),
    }
