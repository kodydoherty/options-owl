"""ML Gate 1: Smart money flow classifier.

Rule-based heuristic that estimates P(smart_money) from UW flow data.
Will be replaced with a trained LightGBM model once sufficient labeled
data is available. Until then, uses sweep counts, block volume, dark pool
percentage, net premium flow, and unusual volume as scoring signals.
"""

from __future__ import annotations


def predict_smart_money(flow_features: dict) -> float:
    """Predict P(smart_money) from flow features.

    Returns 0.0-1.0 probability.  Currently rule-based; no trained model.
    """
    score = 0.5  # neutral baseline

    sweep_count = flow_features.get("sweep_count", 0)
    if sweep_count > 3:
        score += 0.15
    if sweep_count > 8:
        score += 0.10

    if flow_features.get("block_trade_volume", 0) > 10_000:
        score += 0.10

    if flow_features.get("dark_pool_pct", 0.0) > 0.4:
        score += 0.10

    if abs(flow_features.get("net_premium_flow", 0.0)) > 1_000_000:
        score += 0.05

    if flow_features.get("unusual_volume_ratio", 0.0) > 3.0:
        score += 0.05

    return max(0.0, min(1.0, score))
