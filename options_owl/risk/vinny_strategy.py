"""Vinny's 0DTE strategy — phase-based trailing stops, VIX-adjusted trails,
score-based position sizing, and comprehensive exit rules.

This module implements the complete strategy specification from Vinny:
- 7-phase trailing stop system with VIX adjustments
- Time decay zone rules (after 45 min or 3 PM)
- Anti-chase price validation
- Score-based position sizing (5/3/1 contracts)
- Theta bleed exit
- Consecutive loser pause
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone, timedelta
    _ET = timezone(timedelta(hours=-5))


def _now_et() -> datetime:
    """Current time in Eastern Time (DST-aware)."""
    return datetime.now(tz=_ET)


# ---------------------------------------------------------------------------
# Phase-based trailing stop
# ---------------------------------------------------------------------------

# Trail percentages by phase (how far premium can drop from peak before exit)
PHASE_TRAILS: dict[int, float] = {
    0: 40.0,  # Initial hold — no targets hit yet (widened from 25%; 0DTE swings 25% routinely)
    1: 20.0,  # T1 hit
    2: 18.0,  # T2 hit
    3: 15.0,  # T3 hit
    4: 12.0,  # T4 hit
    5: 10.0,  # T5 hit
    6: 8.0,   # Beyond T5 — riding the runner
}


@dataclass
class TrailResult:
    """Result of trailing stop evaluation."""

    should_exit: bool
    phase: int
    trail_pct: float
    drop_from_peak_pct: float
    reason: str


def compute_vix_adjusted_trail(base_trail: float, current_vix: float) -> float:
    """Adjust trail percentage based on VIX level.

    When VIX > 20: trail widens (more room for volatility)
    When VIX < 20: trail tightens (less noise expected)

    Formula: trailAdjust = (currentVIX - 20) × 0.5
    """
    adjustment = (current_vix - 20.0) * 0.5
    adjusted = base_trail + adjustment
    # Floor at 5% — never trail tighter than 5%
    return max(adjusted, 5.0)


def get_current_phase(last_target_hit: int | None) -> int:
    """Determine the current trailing stop phase from last target hit."""
    if last_target_hit is None or last_target_hit <= 0:
        return 0
    return min(last_target_hit, 6)


# ---------------------------------------------------------------------------
# Dollar-based stair-step trailing stop (replaces velocity exit)
# ---------------------------------------------------------------------------

# Vinny's dollar trail: activate at 10% profit, then ratchet the stop up
# in steps that scale with entry cost.  Step sizes are % of entry cost per
# contract so they work for cheap ($0.50) and expensive ($5.00) options alike.
#
# Default steps (match Vinny's $20/$10/$50 for a $2.00 option):
#   small_step_pct=10% → $20 on $200, $5 on $50, $50 on $500
#   step_threshold_pct=25% → $50 on $200, $12.50 on $50, $125 on $500
#   large_step_pct=5%  → $10 on $200, $2.50 on $50, $25 on $500


@dataclass
class DollarTrailResult:
    """Result of dollar-based stair-step trailing stop."""

    should_exit: bool
    active: bool  # whether the trail has activated
    stop_level: float  # current stop level (dollar profit per contract)
    current_profit: float  # current dollar profit per contract
    peak_profit: float  # highest dollar profit per contract
    reason: str


def evaluate_dollar_trail(
    entry_premium: float,
    current_premium: float,
    peak_premium: float,
    activation_pct: float = 10.0,
    small_step_pct: float = 10.0,
    step_threshold_pct: float = 25.0,
    large_step_pct: float = 5.0,
) -> DollarTrailResult:
    """Evaluate the dollar-based stair-step trailing stop.

    Parameters
    ----------
    entry_premium : float
        Premium paid at entry (per share, e.g. 2.00).
    current_premium : float
        Current option premium.
    peak_premium : float
        Highest premium seen (MFE).
    activation_pct : float
        Activate the trail when profit reaches this % of entry cost.
    small_step_pct : float
        Step size as % of entry cost (below threshold). 10% = $20 on $200.
    step_threshold_pct : float
        Switch to tighter steps at this % of entry cost. 25% = $50 on $200.
    large_step_pct : float
        Tighter step size as % of entry cost (above threshold). 5% = $10 on $200.

    All dollar amounts are per contract (1 contract = 100 shares).
    Premium of $2.00 = $200 cost per contract; $0.20 premium move = $20/contract.
    """
    if entry_premium <= 0 or peak_premium <= 0:
        return DollarTrailResult(
            should_exit=False, active=False, stop_level=0.0,
            current_profit=0.0, peak_profit=0.0,
            reason="No valid premium data",
        )

    cost_per_contract = entry_premium * 100
    current_profit = (current_premium - entry_premium) * 100  # per contract
    peak_profit = (peak_premium - entry_premium) * 100  # per contract

    # Scale steps to entry cost
    activation_dollars = cost_per_contract * (activation_pct / 100)
    small_step = cost_per_contract * (small_step_pct / 100)
    step_threshold = cost_per_contract * (step_threshold_pct / 100)
    large_step = cost_per_contract * (large_step_pct / 100)

    # Not yet activated
    if peak_profit < activation_dollars:
        return DollarTrailResult(
            should_exit=False, active=False, stop_level=0.0,
            current_profit=current_profit, peak_profit=peak_profit,
            reason=(
                f"Dollar trail dormant: peak profit ${peak_profit:.0f} "
                f"< activation ${activation_dollars:.0f}"
            ),
        )

    # Calculate the stop level based on highest profit reached (peak)
    stop_level = _compute_stair_stop(
        peak_profit, activation_dollars, small_step, step_threshold, large_step,
    )

    if current_profit < stop_level or abs(current_profit - stop_level) < 0.01:
        return DollarTrailResult(
            should_exit=True, active=True, stop_level=stop_level,
            current_profit=current_profit, peak_profit=peak_profit,
            reason=(
                f"Dollar trail hit: profit ${current_profit:.0f}/contract "
                f"<= stop ${stop_level:.0f} "
                f"(peak ${peak_profit:.0f}, entry ${entry_premium:.2f})"
            ),
        )

    return DollarTrailResult(
        should_exit=False, active=True, stop_level=stop_level,
        current_profit=current_profit, peak_profit=peak_profit,
        reason=(
            f"Dollar trail holding: profit ${current_profit:.0f} "
            f"> stop ${stop_level:.0f} (peak ${peak_profit:.0f})"
        ),
    )


def _compute_stair_stop(
    peak_profit: float,
    activation_dollars: float,
    small_step: float,
    step_threshold: float,
    large_step: float,
) -> float:
    """Compute the stair-step stop level for a given peak profit.

    The stop ratchets up in increments:
    - From activation to step_threshold: small_step increments ($20)
    - Above step_threshold: large_step increments ($10)

    The stop sits at the highest completed step below peak_profit.
    """
    if peak_profit < activation_dollars:
        return 0.0

    def _floor_steps(value: float, step: float) -> int:
        """Integer floor division robust to floating point."""
        return max(0, int(value / step + 1e-9))

    # Phase 1: $20 steps from activation up to step_threshold
    if peak_profit < step_threshold:
        steps = _floor_steps(peak_profit - activation_dollars, small_step)
        return activation_dollars + steps * small_step

    # Phase 2: peak is above step_threshold — compute stop in the $10 zone
    # First, figure out where phase 1 ended (the last $20 step at or below threshold)
    phase1_steps = _floor_steps(step_threshold - activation_dollars, small_step)
    phase1_top = activation_dollars + phase1_steps * small_step

    # Then count $10 steps above that
    above_phase1 = peak_profit - phase1_top
    phase2_steps = _floor_steps(above_phase1, large_step)
    return phase1_top + phase2_steps * large_step


# ---------------------------------------------------------------------------
# Adaptive 3-stage trailing stop (v2.1)
# ---------------------------------------------------------------------------

# Trail stages: as peak gain grows, the trail width adapts.
#   DORMANT  — below activation: no trail, let the trade develop
#   ACTIVE   — moderate gain: standard trail width
#   RUNNER   — large gain: wider trail to let runners breathe
#   MOONSHOT — huge gain: tighten to lock in outsized profits


@dataclass
class AdaptiveTrailResult:
    """Result of adaptive trailing stop evaluation."""

    should_exit: bool
    stage: str  # DORMANT, ACTIVE, RUNNER, MOONSHOT
    trail_width_pct: float  # effective trail width as %
    drop_from_peak_pct: float
    peak_gain_pct: float
    reason: str


def evaluate_adaptive_trail(
    entry_premium: float,
    current_premium: float,
    peak_premium: float,
    activation_pct: float = 40.0,
    active_width: float = 35.0,
    runner_threshold: float = 150.0,
    runner_width: float = 45.0,
    moonshot_threshold: float = 400.0,
    moonshot_width: float = 30.0,
) -> AdaptiveTrailResult:
    """Evaluate the 3-stage adaptive trailing stop.

    Stages (based on peak gain % from entry):
      DORMANT  — peak < activation_pct: no trail
      ACTIVE   — activation_pct <= peak < runner_threshold: trail = active_width
      RUNNER   — runner_threshold <= peak < moonshot_threshold: trail = runner_width (wider)
      MOONSHOT — peak >= moonshot_threshold: trail = moonshot_width (tighter)

    Trail is measured as % drop from peak premium (not from entry).
    """
    if entry_premium <= 0 or peak_premium <= 0:
        return AdaptiveTrailResult(
            should_exit=False, stage="DORMANT", trail_width_pct=0.0,
            drop_from_peak_pct=0.0, peak_gain_pct=0.0,
            reason="No valid premium data",
        )

    peak_gain_pct = (peak_premium - entry_premium) / entry_premium * 100
    drop_from_peak_pct = (peak_premium - current_premium) / peak_premium * 100 if peak_premium > 0 else 0.0

    # Determine stage and width
    if peak_gain_pct < activation_pct:
        return AdaptiveTrailResult(
            should_exit=False, stage="DORMANT", trail_width_pct=0.0,
            drop_from_peak_pct=drop_from_peak_pct, peak_gain_pct=peak_gain_pct,
            reason=f"Dormant: peak gain +{peak_gain_pct:.1f}% < activation +{activation_pct:.0f}%",
        )

    if peak_gain_pct >= moonshot_threshold:
        stage = "MOONSHOT"
        width = moonshot_width
    elif peak_gain_pct >= runner_threshold:
        stage = "RUNNER"
        width = runner_width
    else:
        stage = "ACTIVE"
        width = active_width

    if drop_from_peak_pct >= width:
        return AdaptiveTrailResult(
            should_exit=True, stage=stage, trail_width_pct=width,
            drop_from_peak_pct=drop_from_peak_pct, peak_gain_pct=peak_gain_pct,
            reason=(
                f"Adaptive trail ({stage}): prem ${current_premium:.2f} dropped "
                f"{drop_from_peak_pct:.1f}% from peak ${peak_premium:.2f} "
                f"(width={width:.0f}%, peak gain +{peak_gain_pct:.0f}%)"
            ),
        )

    return AdaptiveTrailResult(
        should_exit=False, stage=stage, trail_width_pct=width,
        drop_from_peak_pct=drop_from_peak_pct, peak_gain_pct=peak_gain_pct,
        reason=(
            f"{stage}: drop {drop_from_peak_pct:.1f}% < trail {width:.0f}% "
            f"(peak ${peak_premium:.2f}, gain +{peak_gain_pct:.0f}%)"
        ),
    )


# ---------------------------------------------------------------------------
# Underlying-anchored trail (v2.1 §5)
# ---------------------------------------------------------------------------

# Default tiers: gain% -> trail width on underlying price
UNDERLYING_TRAIL_TIERS_DEFAULT = [
    (100.0, 0.0050),
    (50.0, 0.0040),
    (15.0, 0.0030),
    (0.0, 0.0020),
]


def parse_underlying_trail_tiers(tiers_str: str) -> list[tuple[float, float]]:
    """Parse 'gain:trail,gain:trail,...' into sorted tier list."""
    tiers = []
    for part in tiers_str.split(","):
        gain_s, trail_s = part.strip().split(":")
        tiers.append((float(gain_s), float(trail_s) / 100.0))
    return sorted(tiers, key=lambda x: -x[0])


def evaluate_underlying_trail(
    entry_premium: float,
    current_premium: float,
    peak_premium: float,
    current_underlying: float,
    peak_underlying: float,
    direction: str,
    tiers: list[tuple[float, float]] | None = None,
    activation_pct: float = 35.0,
) -> tuple[bool, str]:
    """Check if underlying price has trailed past the allowed width.

    Only active when premium gain has reached the adaptive trail activation threshold.
    Returns (should_exit, reason).
    """
    if entry_premium <= 0 or peak_premium <= 0:
        return False, "no premium data"
    if current_underlying <= 0 or peak_underlying <= 0:
        return False, "no underlying data"

    peak_gain = (peak_premium - entry_premium) / entry_premium * 100
    if peak_gain < activation_pct:
        return False, f"dormant: peak gain +{peak_gain:.1f}% < {activation_pct:.0f}%"

    tiers = tiers or UNDERLYING_TRAIL_TIERS_DEFAULT
    trail_pct = tiers[-1][1]  # default to loosest tier
    for min_gain, pct in tiers:
        if peak_gain >= min_gain:
            trail_pct = pct
            break

    is_call = direction in ("call", "bullish", "long")
    if is_call:
        threshold = peak_underlying * (1.0 - trail_pct)
        if current_underlying < threshold:
            return True, (
                f"Underlying trail (call): ${current_underlying:.2f} < "
                f"${threshold:.2f} (peak ${peak_underlying:.2f} - {trail_pct*100:.2f}%, "
                f"prem gain +{peak_gain:.0f}%)"
            )
    else:
        threshold = peak_underlying * (1.0 + trail_pct)
        if current_underlying > threshold:
            return True, (
                f"Underlying trail (put): ${current_underlying:.2f} > "
                f"${threshold:.2f} (peak ${peak_underlying:.2f} + {trail_pct*100:.2f}%, "
                f"prem gain +{peak_gain:.0f}%)"
            )

    return False, (
        f"Underlying trail OK: ${current_underlying:.2f} "
        f"(trail {trail_pct*100:.2f}% of peak ${peak_underlying:.2f})"
    )


# ---------------------------------------------------------------------------
# Volume-peak modifier (v2.1 §6)
# ---------------------------------------------------------------------------


def check_volume_peak(
    underlying_prices: list[float],
    direction: str,
    divergence_threshold: float = 0.001,
) -> str | None:
    """Detect exhaustion via underlying price momentum divergence.

    Compares recent 3-bar avg to previous 3-bar avg. If underlying is moving
    against the trade direction, returns 'tighten'. Otherwise None.
    """
    if len(underlying_prices) < 6:
        return None

    recent = underlying_prices[-6:]
    first_avg = sum(recent[:3]) / 3
    second_avg = sum(recent[3:]) / 3

    is_call = direction in ("call", "bullish", "long")
    if is_call:
        if second_avg < first_avg * (1.0 - divergence_threshold):
            return "tighten"
    else:
        if second_avg > first_avg * (1.0 + divergence_threshold):
            return "tighten"

    return None


# ---------------------------------------------------------------------------
# Time decay zone
# ---------------------------------------------------------------------------


def is_time_decay_zone(
    opened_at: str | datetime,
    now: datetime | None = None,
    max_hold_minutes: float = 45.0,
    afternoon_hour: int = 15,
    afternoon_minute: int = 0,
) -> bool:
    """Check if we're in the time decay zone.

    Time decay zone activates when EITHER:
    - Trade has been open > max_hold_minutes (default 45 min)
    - Current time is after afternoon_hour:afternoon_minute (default 3:00 PM ET)
    """
    if now is None:
        now = _now_et()

    # Check afternoon cutoff
    afternoon_cutoff = now.replace(
        hour=afternoon_hour, minute=afternoon_minute, second=0, microsecond=0,
    )
    if now >= afternoon_cutoff:
        return True

    # Check hold duration
    if isinstance(opened_at, str):
        try:
            opened_at = datetime.fromisoformat(opened_at)
        except (ValueError, TypeError):
            return False

    # Align timezone awareness — opened_at from DB may be naive
    if now.tzinfo is not None and opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and opened_at.tzinfo is not None:
        now = now.replace(tzinfo=opened_at.tzinfo)

    elapsed_min = (now - opened_at).total_seconds() / 60
    return elapsed_min > max_hold_minutes


def check_time_decay_no_new_high(
    current_premium: float,
    peak_premium: float,
    last_new_high_at: str | datetime | None,
    now: datetime | None = None,
    stale_minutes: float = 5.0,
) -> tuple[bool, str]:
    """In time decay zone: exit if no new premium high in stale_minutes.

    Returns (should_exit, reason).
    """
    if now is None:
        now = _now_et()

    if last_new_high_at is None:
        return False, "No high timestamp tracked"

    if isinstance(last_new_high_at, str):
        try:
            last_new_high_at = datetime.fromisoformat(last_new_high_at)
        except (ValueError, TypeError):
            return False, "Cannot parse last_new_high_at"

    # Normalize tz-awareness for safe subtraction
    if now.tzinfo and last_new_high_at.tzinfo is None:
        last_new_high_at = last_new_high_at.replace(tzinfo=now.tzinfo)
    elif last_new_high_at.tzinfo and now.tzinfo is None:
        now = now.replace(tzinfo=last_new_high_at.tzinfo)

    minutes_since_high = (now - last_new_high_at).total_seconds() / 60

    if minutes_since_high >= stale_minutes:
        return True, (
            f"Time decay zone: no new high in {minutes_since_high:.0f}m "
            f"(limit {stale_minutes:.0f}m), prem=${current_premium:.2f} "
            f"vs peak=${peak_premium:.2f}"
        )

    return False, f"Last high {minutes_since_high:.0f}m ago (limit {stale_minutes:.0f}m)"


# ---------------------------------------------------------------------------
# Anti-chase check
# ---------------------------------------------------------------------------


def check_anti_chase(
    alert_price: float,
    current_price: float,
    max_move_pct: float = 0.3,
) -> tuple[bool, str]:
    """Reject if the underlying has moved too far from the alert price.

    Returns (passed, reason). True = OK to trade, False = chase detected.
    """
    if alert_price <= 0:
        return True, "No alert price"

    move_pct = abs(current_price - alert_price) / alert_price * 100

    if move_pct > max_move_pct:
        return False, (
            f"Anti-chase: underlying moved {move_pct:.2f}% from alert "
            f"${alert_price:.2f} → ${current_price:.2f} (max {max_move_pct:.1f}%)"
        )

    return True, f"Price within {move_pct:.2f}% of alert (max {max_move_pct:.1f}%)"


# ---------------------------------------------------------------------------
# Score-based position sizing
# ---------------------------------------------------------------------------



# Flat sizing: scores above 78 get equal allocation.
# Backtested 2026-05-20: tiered multipliers don't improve P&L because scores
# don't predict outcomes. The 78 floor is the real filter; above that, flat
# sizing avoids under-sizing trades that happen to score low but win big.
_SCORE_BUDGET_MULT = 0.85  # 85% of per-slot budget for all qualifying trades
_SCORE_FLOOR = 78  # below this, trade is rejected


def score_to_contracts(
    score: int,
    cost_per_contract: float | None = None,
    balance: float | None = None,
    max_position_pct: float = 15.0,
    max_concurrent: int = 4,
    max_portfolio_risk_pct: float = 75.0,
) -> int:
    """Flat sizing — equal allocation for all trades above score floor.

    1. Score floor: < 78 = rejected (0 contracts)
    2. Dollar target: balance × risk_cap / max_concurrent = slot budget
    3. Flat multiplier: 85% of slot budget for all qualifying trades
    4. Final = min(scaled_target, position_cap)

    Scores don't predict outcomes (backtested 2026-05-20), so every trade
    above the 78 floor gets the same allocation. The real edge comes from
    the V5 exit engine, not entry sizing.
    """
    if score < _SCORE_FLOOR:
        logger.info(f"SIZING: score {score} < {_SCORE_FLOOR} → 0 contracts (rejected)")
        return 0

    score_mult = _SCORE_BUDGET_MULT

    if cost_per_contract is not None and balance is not None and cost_per_contract > 0:
        # Primary: target each trade to fit max_concurrent trades in the risk cap
        total_deployable = balance * (max_portfolio_risk_pct / 100)
        target_per_trade = total_deployable / max(1, max_concurrent)

        # Flat budget scaling
        scaled_target = target_per_trade * score_mult
        raw_contracts = int(scaled_target / cost_per_contract)

        # Hard cap: never exceed position cap
        max_spend = balance * (max_position_pct / 100)
        max_by_position = int(max_spend / cost_per_contract)

        # Apply position % cap
        final_contracts = min(raw_contracts, max_by_position)

        # If 1 contract exceeds the position cap, skip this trade entirely.
        if max_by_position == 0:
            logger.info(
                f"SIZING: score={score} balance=${balance:.2f} cost/contract=${cost_per_contract:.2f} "
                f"| pos_cap({max_position_pct}%=${max_spend:.2f}) < 1 contract → SKIP"
            )
            return 0

        # Floor: at least 1 contract if within risk limits
        final_contracts = max(1, final_contracts)

        logger.info(
            f"SIZING: score={score} balance=${balance:.2f} cost/contract=${cost_per_contract:.2f} "
            f"| risk_cap={max_portfolio_risk_pct}% deployable=${total_deployable:.2f} "
            f"| max_concurrent={max_concurrent} target/slot=${target_per_trade:.2f} "
            f"| flat_mult={score_mult:.0%} scaled=${scaled_target:.2f} raw={raw_contracts} "
            f"| pos_cap({max_position_pct}%=${max_spend:.2f})={max_by_position} "
            f"→ {final_contracts} contracts "
            f"(total=${final_contracts * cost_per_contract:.2f})"
        )
        return final_contracts

    # Fallback when no cost/balance info
    return max(1, int(5 * score_mult))


# ---------------------------------------------------------------------------
# Theta bleed exit
# ---------------------------------------------------------------------------


def check_theta_bleed(
    entry_premium: float,
    current_premium: float,
    opened_at: str | datetime,
    now: datetime | None = None,
    max_hold_minutes: float = 45.0,
    max_loss_pct: float = 30.0,
) -> tuple[bool, str]:
    """Exit if held too long AND losing too much — theta is eating the position.

    Returns (should_exit, reason).
    """
    if now is None:
        now = _now_et()

    if isinstance(opened_at, str):
        try:
            opened_at = datetime.fromisoformat(opened_at)
        except (ValueError, TypeError):
            return False, "Cannot parse opened_at"

    # Normalize tz-awareness for safe subtraction
    if now.tzinfo and opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=now.tzinfo)
    elif opened_at.tzinfo and now.tzinfo is None:
        now = now.replace(tzinfo=opened_at.tzinfo)

    elapsed_min = (now - opened_at).total_seconds() / 60

    if elapsed_min < max_hold_minutes:
        return False, f"Only {elapsed_min:.0f}m held (check at {max_hold_minutes:.0f}m)"

    if entry_premium <= 0:
        return False, "No entry premium"

    loss_pct = (entry_premium - current_premium) / entry_premium * 100

    if loss_pct >= max_loss_pct:
        return True, (
            f"Theta bleed: held {elapsed_min:.0f}m and down {loss_pct:.1f}% "
            f"(${current_premium:.2f} from ${entry_premium:.2f})"
        )

    return False, f"Down {loss_pct:.1f}% after {elapsed_min:.0f}m (limit {max_loss_pct:.0f}%)"


# ---------------------------------------------------------------------------
# Consecutive loser pause
# ---------------------------------------------------------------------------


def check_consecutive_loser_pause(
    consecutive_losses: int,
    last_loss_at: str | datetime | None,
    now: datetime | None = None,
    max_consecutive: int = 2,
    pause_minutes: float = 15.0,
) -> tuple[bool, str]:
    """Pause trading after N consecutive losses for a cooldown period.

    Returns (can_trade, reason). True = OK to trade.
    """
    if consecutive_losses < max_consecutive:
        return True, f"{consecutive_losses} consecutive losses < {max_consecutive}"

    if now is None:
        now = _now_et()

    if last_loss_at is None:
        return True, "No last_loss_at timestamp"

    if isinstance(last_loss_at, str):
        try:
            last_loss_at = datetime.fromisoformat(last_loss_at)
        except (ValueError, TypeError):
            return True, "Cannot parse last_loss_at"

    # Normalize tz-awareness for safe subtraction
    if now.tzinfo and last_loss_at.tzinfo is None:
        last_loss_at = last_loss_at.replace(tzinfo=now.tzinfo)
    elif last_loss_at.tzinfo and now.tzinfo is None:
        now = now.replace(tzinfo=last_loss_at.tzinfo)

    elapsed = (now - last_loss_at).total_seconds() / 60

    if elapsed < pause_minutes:
        return False, (
            f"Consecutive loser pause: {consecutive_losses} losses in a row, "
            f"cooling down {elapsed:.0f}m / {pause_minutes:.0f}m"
        )

    return True, (
        f"Consecutive loser cooldown expired ({elapsed:.0f}m >= {pause_minutes:.0f}m)"
    )
