"""Tier 4: Risk Adjustments (-10 to 0 points).

Are there reasons this trade is riskier than the signals suggest?
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext, TierResult


def tier4_risk(ctx: SignalContext) -> TierResult:
    """Evaluate risk penalties.

    Penalties:
        - RSI extreme against direction: -5
        - Chase / extended move: -5
        - Late-day theta bleed: -3
        - Negative news sentiment: -5 to BLOCK
        - Earnings proximity: BLOCK
    """
    raise NotImplementedError("Phase 2: implement tier 4 risk adjustments")
