"""ML Gate 3: Signal quality predictor.

When enabled, replaces the entire hand-tuned 5-tier scoring system.
P(win) becomes the score.
"""

from __future__ import annotations


def predict_win_probability(raw_features: dict) -> float:
    """Predict P(win) from raw indicator + alpha source features.

    Returns 0.0-1.0. Multiply by 100 for score.
    """
    raise NotImplementedError("Phase 5+: train after 300+ trades on new system")
