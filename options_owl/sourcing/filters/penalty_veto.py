"""Post-scoring critical penalty combo veto."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


def check_penalty_veto(ctx: SignalContext) -> bool:
    """Returns True if signal is vetoed by critical penalty combination."""
    raise NotImplementedError("Phase 2: implement penalty veto")
