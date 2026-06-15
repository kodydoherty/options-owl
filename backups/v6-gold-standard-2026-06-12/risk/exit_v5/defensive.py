"""Defensive safety gate — bid disappearance detection.

Pure function, no side effects. Used by Gate 2 in the v5 FSM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from options_owl.risk.exit_v5.config import DefensiveConfig


def check_bid_disappearance(
    bid: float,
    seconds_at_zero_bid: float,
    cfg: DefensiveConfig | None = None,
) -> dict:
    """Check if bid has been zero for too long.

    When the bid disappears (no buyers), the option is likely worthless.
    After 30s of consecutive zero bids, trigger an exit.

    Args:
        bid: Current best bid price.
        seconds_at_zero_bid: Consecutive seconds the bid has been 0.
        cfg: Defensive config (uses defaults if None).

    Returns:
        {"should_exit": bool, "reason": str, "seconds_at_zero": float}
    """
    if cfg is None:
        from options_owl.risk.exit_v5.config import DefensiveConfig
        cfg = DefensiveConfig()

    timeout = cfg.bid_zero_timeout_sec
    is_zero = bid <= 0.0
    should_exit = is_zero and seconds_at_zero_bid >= timeout

    return {
        "should_exit": should_exit,
        "reason": f"bid_zero_{seconds_at_zero_bid:.0f}s" if should_exit else "",
        "seconds_at_zero": seconds_at_zero_bid if is_zero else 0.0,
    }
