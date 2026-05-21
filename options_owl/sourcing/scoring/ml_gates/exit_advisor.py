"""ML Gate 5: Exit timing advisory (survival analysis).

Advisory overlay on V5 FSM — logs recommendation but does NOT override gates.
"""

from __future__ import annotations


def predict_should_exit(exit_features: dict) -> float:
    """Predict P(should_exit_now) from trade context.

    Returns 0.0-1.0. Advisory only — V5 FSM makes the final decision.
    """
    raise NotImplementedError("Phase 5+: train on 200+ closed trades with tick data")
