"""Write signals to shared signals.db (Phase 2 output)."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


async def emit_signal_db(ctx: SignalContext, db_path: str) -> None:
    """Write scored signal to signals.db for direct owlet consumption."""
    raise NotImplementedError("Phase 6: implement signal DB writer")
