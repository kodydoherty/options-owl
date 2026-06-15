"""Target-anchored OTM strike selection.

Scores each available OTM strike using four criteria:
1. Reachability — is the strike within the signal's price targets?
2. Affordability — is the premium in the ideal leverage range?
3. Gamma zone — is the delta in the peak acceleration band?
4. Tiebreaker — cheapest option wins when scores are equal.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class ScoredStrike:
    """A candidate OTM strike with its scoring breakdown."""

    strike: float
    premium: float
    delta: float | None
    reach_score: int
    afford_score: int
    gamma_score: int
    tiebreak_score: float
    total_score: float


def score_otm_strikes(
    strikes: list[dict],
    entry_price: float,
    target_price: float,
    direction: str,
    t2_price: float | None = None,
) -> list[ScoredStrike]:
    """Score and rank OTM strikes for a given signal.

    Parameters
    ----------
    strikes
        List of dicts, each with keys: ``strike``, ``premium``, and
        optionally ``delta``.  Premium is per-share (e.g. $0.80), not
        per-contract.
    entry_price
        Current / entry price of the underlying.
    target_price
        The signal's final target price (T4/T5 or last target).
    direction
        ``"call"`` or ``"put"``.
    t2_price
        Halfway target (T2).  If *None*, computed as midpoint between
        *entry_price* and *target_price*.

    Returns
    -------
    list[ScoredStrike]
        Strikes sorted by total_score descending (best first).
    """
    is_call = direction.lower() in ("call", "c")

    # Compute T2 midpoint if not provided
    if t2_price is None:
        t2_price = (entry_price + target_price) / 2

    scored: list[ScoredStrike] = []

    for s in strikes:
        strike = s["strike"]
        premium = s["premium"]
        delta = s.get("delta")

        # Skip strikes with no premium data
        if premium is None or premium <= 0:
            continue

        # --- 1. Reachability ---
        reach = _score_reachability(strike, entry_price, target_price, t2_price, is_call)

        # --- 2. Affordability ---
        afford = _score_affordability(premium)

        # --- 3. Gamma zone ---
        gamma = _score_gamma_zone(delta)

        # --- 4. Tiebreaker: cheaper is better (capped to avoid dominating) ---
        tiebreak = min(5.0 * (1.0 / premium), 10.0)

        total = reach + afford + gamma + tiebreak

        scored.append(ScoredStrike(
            strike=strike,
            premium=premium,
            delta=delta,
            reach_score=reach,
            afford_score=afford,
            gamma_score=gamma,
            tiebreak_score=round(tiebreak, 2),
            total_score=round(total, 2),
        ))

    scored.sort(key=lambda x: x.total_score, reverse=True)
    return scored


def select_best_otm(
    strikes: list[dict],
    entry_price: float,
    target_price: float,
    direction: str,
    t2_price: float | None = None,
) -> ScoredStrike | None:
    """Return the single best OTM strike, or None if no viable candidates.

    Filters out strikes with a negative total score before selecting.
    """
    ranked = score_otm_strikes(strikes, entry_price, target_price, direction, t2_price)
    if not ranked:
        return None

    best = ranked[0]

    # Reject if the best score is still negative (all strikes are bad)
    if best.total_score < 0:
        logger.info(
            f"No viable OTM strike: best candidate ${best.strike} scored {best.total_score}"
        )
        return None

    logger.info(
        f"OTM selected: ${best.strike} @ ${best.premium:.2f} "
        f"(score={best.total_score}, reach={best.reach_score}, "
        f"afford={best.afford_score}, gamma={best.gamma_score})"
    )
    return best


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_reachability(
    strike: float,
    entry: float,
    target: float,
    t2: float,
    is_call: bool,
) -> int:
    """Score how reachable the strike is relative to price targets.

    For calls: strike should be above entry but within targets.
    For puts:  strike should be below entry but within targets.
    """
    if is_call:
        if strike <= entry:
            return -10  # ITM or ATM, not OTM
        if strike <= t2:
            return 35  # Within T2 — highest win rate
        if strike <= target:
            return 20  # Within full target — wins if target reached
        return -10  # Beyond target — needs overshoot
    else:
        if strike >= entry:
            return -10  # ITM or ATM, not OTM
        if strike >= t2:
            return 35  # Within T2
        if strike >= target:
            return 20  # Within full target
        return -10  # Beyond target


def _score_affordability(premium: float) -> int:
    """Score the premium range for leverage vs. lottery-ticket avoidance."""
    if premium < 0.10:
        return -25  # Lottery ticket
    if premium <= 0.30:
        return 20  # Lower end of good range
    if premium <= 1.00:
        return 35  # Maximum leverage sweet spot
    if premium <= 2.00:
        return 20  # Good range
    if premium <= 3.00:
        return 5   # Acceptable
    return -20  # Too expensive for OTM


def _score_gamma_zone(delta: float | None) -> int:
    """Score the delta for gamma acceleration potential."""
    if delta is None:
        return 0  # No delta data — neutral score
    delta = abs(delta)
    if delta < 0.08:
        return -20  # Too far out, won't gain meaningful value
    if delta < 0.10:  # 0.08-0.099
        return 10  # Acceptable low end
    if delta <= 0.30:
        return 20  # Peak gamma acceleration zone
    if delta <= 0.40:
        return 10  # Acceptable
    return -10  # Too deep ITM for an "OTM" pick
