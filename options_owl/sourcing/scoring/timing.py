"""Tier 2: Timing Quality (0-30 points).

Is THIS the right moment to enter, or are we late/early?
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext, TierResult


def tier2_timing(ctx: SignalContext) -> TierResult:
    """Evaluate timing quality from indicators.

    Sub-signals:
        - Volume confirmation: 0-10 (mandatory min 3/10)
        - RSI positioning: 0-5
        - MACD alignment: 0-5
        - Entry velocity: 0-5
        - Volatility regime (ATR): 0-5
    """
    raise NotImplementedError("Phase 2: implement tier 2 timing scoring")
