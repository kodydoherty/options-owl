"""Quality gate: minimum score, multi-tier contribution, circuit breaker."""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext


def check_quality_gate(ctx: SignalContext, threshold: int = 60) -> bool:
    """Returns True if signal passes quality gate."""
    raise NotImplementedError("Phase 2: implement quality gate")
