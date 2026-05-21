"""ML Gate 1: Smart money flow classifier.

Classifies raw UW flow data as smart money vs noise using LightGBM.
"""

from __future__ import annotations


def predict_smart_money(flow_features: dict) -> float:
    """Predict P(smart_money) from flow features.

    Returns 0.0-1.0 probability.
    """
    raise NotImplementedError("Phase 3.5: train and deploy flow classifier")
