"""Tier 3: Edge Amplifiers (0-20 points).

Extra confluence beyond direction + timing. Includes alpha sources.
"""

from __future__ import annotations

from options_owl.sourcing.scoring.types import SignalContext, TierResult


def tier3_amplifiers(ctx: SignalContext) -> TierResult:
    """Evaluate edge amplifiers from indicators + alpha sources.

    Sub-signals:
        - ORB confirmation: 0-3
        - Candlestick pattern: 0-2
        - Relative strength vs SPY: 0-2
        - News/catalyst alignment: 0-3
        - Insider/Congress bias: 0-4 (alpha)
        - Contrarian sentiment: 0-3 (alpha)
        - Smart money flow quality: 0-3 (ML)
    """
    raise NotImplementedError("Phase 2-3.5: implement tier 3 amplifier scoring")
