"""ML Gate 4: Regime-aware dynamic source weighting.

Rule-based heuristic that adjusts source category weights based on market
regime indicators (VIX level, SPY movement, ADX trend strength, Bollinger
squeeze). Will be replaced with a trained model once 3+ months of scored
trade data is available.
"""

from __future__ import annotations


def predict_source_weights(regime_features: dict) -> dict[str, float]:
    """Predict optimal weight multiplier per source category.

    Returns dict like {"technical": 1.2, "flow": 0.8, "sentiment": 1.5, "macro": 0.5}.
    Currently rule-based; no trained model.
    """
    weights: dict[str, float] = {
        "technical": 1.0,
        "flow": 1.0,
        "sentiment": 1.0,
        "macro": 1.0,
    }

    vix = regime_features.get("vix", 20.0)
    spy_change_1d = regime_features.get("spy_change_1d", 0.0)
    adx = regime_features.get("adx", 20.0)
    bb_squeeze = regime_features.get("bb_squeeze", False)

    # VIX regime
    if vix > 30:
        weights["technical"] *= 0.7
        weights["flow"] *= 1.3
        weights["macro"] *= 1.5
    elif vix < 15:
        weights["technical"] *= 1.2
        weights["flow"] *= 0.8

    # Big move day
    if abs(spy_change_1d) > 2.0:
        weights["flow"] *= 1.3
        weights["sentiment"] *= 0.7

    # Trend strength
    if adx > 25:
        weights["technical"] *= 1.2
    elif adx < 15:
        weights["technical"] *= 0.8
        weights["flow"] *= 1.2

    # Bollinger squeeze
    if bb_squeeze:
        weights["technical"] *= 1.1

    # Round to 2 decimal places
    return {k: round(v, 2) for k, v in weights.items()}
