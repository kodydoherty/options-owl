"""Full scoring breakdown to scoring_audit table (always runs)."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


async def log_audit(ctx: SignalContext, db_path: str) -> None:
    """Log full scoring context to audit table, regardless of pass/fail."""
    raise NotImplementedError("Phase 4: implement audit logger")
