"""Shared types for the v5 exit engine.

Extracted here to break the circular import between fsm.py and gates.py.
Both modules import ExitReason, ExitAction, _exit, and _hold from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExitReason(Enum):
    """Exit reasons — each maps to a specific gate."""
    HOLD = "hold"
    BID_DISAPPEARANCE = "bid_disappearance"
    HARD_STOP = "hard_stop"
    SOFT_TRAIL = "soft_trail"
    EOD_CUTOFF = "eod_cutoff"
    SCALP_TRAIL = "scalp_trail"
    CHECKPOINT_CUT = "checkpoint_cut"
    CONFIRMED_STOP = "confirmed_stop"
    ADAPTIVE_TRAIL = "adaptive_trail"
    THETA_BLEED = "theta_bleed"
    THETA_TIMER = "theta_timer"
    PROFIT_TARGET = "profit_target"
    BREAKEVEN_RATCHET = "breakeven_ratchet"
    SCALEOUT = "scaleout"
    SIDEWAYS_SCALP = "sideways_scalp"


@dataclass
class ExitAction:
    """Return type from FSM evaluation."""
    should_exit: bool
    reason: ExitReason
    detail: str
    contracts_to_close: int = 0  # 0 = close all, >0 = partial (reserved for future use)
    debug: dict = field(default_factory=dict)


def _exit(reason: ExitReason, detail: str, debug: dict | None = None) -> ExitAction:
    """Create an exit action."""
    return ExitAction(
        should_exit=True, reason=reason, detail=detail,
        debug=debug or {},
    )


def _hold(detail: str, debug: dict | None = None) -> ExitAction:
    """Create a hold action."""
    return ExitAction(
        should_exit=False, reason=ExitReason.HOLD, detail=detail,
        debug=debug or {},
    )
