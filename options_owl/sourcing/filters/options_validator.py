"""Options chain validation: spread, premium cap, volume.

Matches production V6 entry gates (premium cap, spread gate).
Pure function — no I/O.
"""

from __future__ import annotations

from options_owl.sourcing.data.options_provider import OptionsChain


def validate_chain(chain: OptionsChain, score: int) -> tuple[bool, str]:
    """Validate options chain data against entry gates.

    Returns (is_valid, reason_if_invalid).

    Gates (match production V6):
    1. Premium cap: $6 base, $7 for score 120+, $9 for score 150+
    2. Spread gate: bid-ask spread > 40% = reject
    3. Minimum liquidity: volume + OI must show some activity
    """
    # Gate 1: Premium cap (tiered by score)
    premium_cap = 6.0
    if score >= 150:
        premium_cap = 9.0
    elif score >= 120:
        premium_cap = 7.0

    if chain.mid > premium_cap:
        return False, f"premium_too_high: ${chain.mid:.2f} > ${premium_cap:.2f} cap (score={score})"

    # Gate 2: Minimum premium floor
    if chain.mid < 0.05:
        return False, f"premium_too_low: ${chain.mid:.2f} (likely worthless)"

    # Gate 3: Spread gate
    if chain.spread_pct > 40.0:
        return False, f"spread_too_wide: {chain.spread_pct:.1f}% > 40% max"

    # Gate 4: Minimum liquidity (soft — warn but allow if volume exists)
    if chain.volume == 0 and chain.open_interest < 10:
        return False, f"no_liquidity: vol={chain.volume} OI={chain.open_interest}"

    return True, ""
