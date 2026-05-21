"""ML Gate 4: Regime-aware dynamic source weighting."""

from __future__ import annotations


def predict_source_weights(regime_features: dict) -> dict[str, float]:
    """Predict optimal weight multiplier per source category.

    Returns dict like {"technical": 1.2, "flow": 0.8, "sentiment": 1.5, "macro": 0.5}
    """
    raise NotImplementedError("Phase 6+: requires 3+ months of scored trade data")
