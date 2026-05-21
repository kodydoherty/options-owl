"""Format and send Discord embeds (Phase 1 output)."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


async def emit_discord(ctx: SignalContext) -> None:
    """Format signal as Discord embed and POST to webhook."""
    raise NotImplementedError("Phase 4: implement Discord webhook output")
