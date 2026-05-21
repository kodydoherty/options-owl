"""Tier 1: Direction Confidence (0-40 points).

Is the underlying actually moving in the signal's direction?
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext, TierResult


def tier1_direction(ctx: SignalContext) -> TierResult:
    """Evaluate direction confidence from indicators.

    Sub-signals:
        - EMA 9/21 crossover strength: 0-15
        - Multi-timeframe alignment: 0-10
        - Trend regime (ADX + EMA200): 0-5
        - VWAP position: 0-5
        - Key level proximity: 0-5
    """
    raise NotImplementedError("Phase 2: implement tier 1 direction scoring")
