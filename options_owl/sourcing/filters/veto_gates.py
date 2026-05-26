"""Simpsons-inspired hard veto gates.

These are HARD BLOCKS (not score penalties). Based on the Simpsons v10
"Surgical Filter" that turned 39.4% WR into 85.9% by removing 84.5% of
losers without cutting winners.

Each gate returns (blocked: bool, reason: str). First block wins.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger
from zoneinfo import ZoneInfo

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.types import Direction, SignalContext

ET = ZoneInfo("America/New_York")


def run_veto_gates(ctx: SignalContext) -> tuple[bool, str]:
    """Run all Simpsons-inspired veto gates on a scored signal.

    Returns (blocked, reason). If blocked=True, the signal should be rejected.
    Gates are ordered cheapest-first.
    """
    gates = [
        _veto_afternoon_danger_zone,
        _veto_wide_spread,
        # NOTE: ML warmup (_veto_ml_warmup) is disabled as a hard block.
        # Sweep data shows killzone entries at 9:30 with 1 observation are
        # the most profitable — tech scoring (0.8 weight) handles filtering.
        # ML warmup should lower ML weight, not block entirely.
        #
        # NOTE: Sweep, volume, and momentum gates need fresh indicators
        # per entry (not stale 9:30 values). Disabled until indicators
        # are recomputed at each scan_ticker() call.
    ]

    for gate in gates:
        blocked, reason = gate(ctx)
        if blocked:
            logger.info(f"VETO {ctx.ticker}: {reason}")
            return True, reason

    return False, ""


# ---------------------------------------------------------------------------
# Individual veto gates
# ---------------------------------------------------------------------------


def _veto_afternoon_danger_zone(ctx: SignalContext) -> tuple[bool, str]:
    """Block ALL signals between 1:30 PM and 3:00 PM ET.

    Simpsons data: 36% WR in this window vs 72% in killzone.
    This is a HARD BLOCK, not a score penalty.
    """
    now_et = datetime.now(tz=ET)
    minutes_since_open = max(0, int((now_et - now_et.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60))

    # 1:30 PM = 240 min after open, 3:00 PM = 330 min after open
    if 240 <= minutes_since_open <= 330:
        return True, f"afternoon_danger_zone (min_since_open={minutes_since_open})"
    return False, ""


def _veto_ml_warmup(ctx: SignalContext) -> tuple[bool, str]:
    """Block ML-gated signals until >= 10 premium observations exist.

    At 9:30 AM, ML has 1 premium observation. Features like
    premium_volatility and premium_momentum are zero — the model
    is effectively blind. Wait until it has real data.

    Tech-only signals (no ML) are NOT blocked by this gate.
    """
    # Only applies when ML was used
    if ctx.ml_model_source in ("", "none") or ctx.ml_confidence is None:
        return False, ""

    option_hist = getattr(ctx, "_option_history", None)
    if option_hist and hasattr(option_hist, "snapshots"):
        obs_count = len(option_hist.snapshots) if option_hist.snapshots else 0
    else:
        obs_count = 0

    if obs_count < 10:
        return True, f"ml_warmup: only {obs_count} premium observations (need >= 10)"
    return False, ""


def _veto_no_institutional_sweep(ctx: SignalContext) -> tuple[bool, str]:
    """Prefer sweep of institutional level (PDH/PDL/PWH/PWL).

    Simpsons rule #3: institutional sweeps indicate smart money is moving.

    NOTE: On 5-minute candles, sweeps only fire ~9% of the time.
    Simpsons used 1-minute candles where sweeps are more visible.
    We make this a soft check: require sweep OR strong volume (>= 2x).
    This prevents blocking 95% of trades while still filtering noise.

    Direction-aware:
    - CALL prefers sweep below (PDL/PWL/session_low = bear trap → bullish)
    - PUT prefers sweep above (PDH/PWH/session_high = bull trap → bearish)
    """
    ind: IndicatorSet | None = ctx.indicators  # type: ignore[assignment]
    if ind is None:
        return False, ""  # no data to check, let other gates handle

    is_call = ctx.direction == Direction.CALL if ctx.direction else True

    if is_call:
        has_sweep = ind.sweep_pdl or ind.sweep_pwl or ind.sweep_session_low
    else:
        has_sweep = ind.sweep_pdh or ind.sweep_pwh or ind.sweep_session_high

    # Accept if sweep detected OR strong volume confirms the move
    if has_sweep or ind.volume_ratio >= 2.0:
        return False, ""

    return True, "no_institutional_sweep_or_volume (no sweep AND volume_ratio < 2.0x)"


def _veto_low_atr_chop(ctx: SignalContext) -> tuple[bool, str]:
    """Block when ATR is too low (choppy, range-bound market).

    Simpsons rule #6: skip if ATR < 1.0.
    We use a relative check: ATR should indicate the market is moving enough
    for options premiums to develop.
    """
    ind: IndicatorSet | None = ctx.indicators  # type: ignore[assignment]
    if ind is None:
        return False, ""

    # Choppy market: very low ATR AND very low ADX (no trend at all)
    # Relaxed from 0.5/15 — only block truly dead markets
    if ind.atr14 > 0 and ind.adx > 0:
        if ind.atr14 < 0.2 and ind.adx < 12:
            return True, f"low_atr_chop: atr={ind.atr14:.2f} adx={ind.adx:.1f}"
    return False, ""


def _veto_wide_spread(ctx: SignalContext) -> tuple[bool, str]:
    """Block if bid-ask spread > 30% of premium.

    Simpsons rule #8: wide spread = illiquid, will get slipped on entry AND exit.
    This is a HARD BLOCK, not the -2 to -4 pt penalty from risk tier.
    """
    if ctx.spread_pct is not None and ctx.spread_pct > 30:
        return True, f"spread_too_wide: {ctx.spread_pct:.1f}% > 30%"

    # Also check from option snapshot if available
    option_snap = getattr(ctx, "_option_snapshot", None)
    if option_snap and option_snap.bid > 0 and option_snap.ask > 0:
        mid = (option_snap.bid + option_snap.ask) / 2
        if mid > 0:
            spread_pct = ((option_snap.ask - option_snap.bid) / mid) * 100
            if spread_pct > 30:
                return True, f"spread_too_wide: {spread_pct:.1f}% > 30% (bid=${option_snap.bid:.2f} ask=${option_snap.ask:.2f})"

    return False, ""


def _veto_sweep_no_volume(ctx: SignalContext) -> tuple[bool, str]:
    """Block sweep signals without volume confirmation.

    Simpsons rule #7: sweep candle must have >= 1.5x avg volume.
    A sweep without volume is a fake breakout, not institutional activity.
    """
    ind: IndicatorSet | None = ctx.indicators  # type: ignore[assignment]
    if ind is None:
        return False, ""

    # Only applies when a sweep was detected
    has_any_sweep = (
        ind.sweep_pdh or ind.sweep_pdl
        or ind.sweep_pwh or ind.sweep_pwl
        or ind.sweep_session_high or ind.sweep_session_low
    )
    if not has_any_sweep:
        return False, ""  # no sweep to validate

    if ind.volume_ratio < 1.5:
        return True, f"sweep_no_volume_confirm: vol_ratio={ind.volume_ratio:.2f} < 1.5x"
    return False, ""


def _veto_no_momentum_confirm(ctx: SignalContext) -> tuple[bool, str]:
    """Block if underlying price is not moving in signal direction on 5m chart.

    Simpsons rule #14: underlying must have positive momentum in direction.
    Check last 3 candles — price should be trending with the signal.
    """
    if not ctx.candles_5m or len(ctx.candles_5m) < 3:
        return False, ""  # not enough data to check

    is_call = ctx.direction == Direction.CALL if ctx.direction else True
    recent = ctx.candles_5m[-3:]

    # Check if last 3 candles show momentum in direction
    first_close = recent[0].get("close", 0)
    last_close = recent[-1].get("close", 0)

    if first_close <= 0:
        return False, ""

    pct_change = ((last_close - first_close) / first_close) * 100

    # CALL needs positive momentum, PUT needs negative
    if is_call and pct_change < -0.1:
        return True, f"no_momentum_confirm: CALL but 5m trend={pct_change:+.2f}%"
    if not is_call and pct_change > 0.1:
        return True, f"no_momentum_confirm: PUT but 5m trend={pct_change:+.2f}%"

    return False, ""
