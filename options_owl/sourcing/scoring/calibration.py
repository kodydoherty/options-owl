"""Tier 5: Calibration Bonus (0-10 points).

Does historical data suggest this setup outperforms the base rate?
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext, TierResult


def tier5_calibration(ctx: SignalContext) -> TierResult:
    """Evaluate calibration bonus from historical data.

    Sub-signals:
        - Bayesian signature match: 0-5
        - Ticker-specific lift: 0-5
    """
    raise NotImplementedError("Phase 5: implement tier 5 calibration")
