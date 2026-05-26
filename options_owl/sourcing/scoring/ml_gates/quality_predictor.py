"""ML Gate 3: Signal quality predictor.

Rule-based heuristic ensemble that estimates P(win) from technical score,
ML confidence, flow score, volume, spread, and time-of-day features.
Will be replaced with a trained model once 300+ trades on the new sourcing
system are available. Until then, blends existing signals with penalties
for wide spreads, opening chaos, and end-of-day theta decay.
"""

from __future__ import annotations


def predict_win_probability(raw_features: dict) -> float:
    """Predict P(win) from raw indicator + alpha source features.

    Returns 0.0-1.0. Multiply by 100 for score.
    Currently rule-based; no trained model.
    """
    technical_score = raw_features.get("technical_score", 50)
    ml_confidence = raw_features.get("ml_confidence")
    flow_score = raw_features.get("flow_score")
    volume_ratio = raw_features.get("volume_ratio", 0.0)
    spread_pct = raw_features.get("spread_pct", 0.0)
    minutes_since_open = raw_features.get("minutes_since_open", 60)

    base = technical_score / 100.0

    if ml_confidence is not None:
        blend = 0.6 * ml_confidence + 0.4 * base
    else:
        blend = base

    if flow_score is not None and flow_score > 0.6:
        blend += 0.05

    if volume_ratio > 2.0:
        blend += 0.03

    if spread_pct > 20:
        blend -= 0.10

    if minutes_since_open < 10:
        blend -= 0.05

    if minutes_since_open > 360:
        blend -= 0.05

    return max(0.0, min(1.0, blend))
