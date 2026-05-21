"""V5 exit gates — each gate is a pure function returning an ExitAction or None.

Gate priority (first match wins):
    1. EOD cutoff         — 0DTE only, 15min before close
    2. Bid disappearance  — no buyers for 30s
    3. [5min grace]       — skip all exits in first 5 minutes
    4. Profit target      — index 0DTE: take gains at 30%
    5. Scalp trail        — peaked +20%, faded <60%, underlying doesn't confirm
    6. Checkpoint cut     — 0DTE: down 30%+ AND underlying against 0.5%+
    7. Graduated stop     — underlying-based: 35% if against, 65% backstop (0DTE)
    8. Soft trail         — 10-50% peak band, keep 60% of peak gain
    9. Adaptive trail     — category-aware tiers (high-vol wider, index tighter)
   10. Theta exit         — 0DTE: bleed at 120min/30%; multi-day: timer at 180min/15%

Each gate function returns ExitAction if triggered, or None to pass to next gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from options_owl.risk.exit_v5.defensive import check_bid_disappearance
from options_owl.risk.exit_v5.types import ExitAction, ExitReason, _exit

if TYPE_CHECKING:
    from options_owl.risk.exit_v5.config import AdaptiveTier, V5Config


def check_eod_cutoff(
    is_0dte: bool,
    minutes_to_close: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 1: Close 0DTE positions before market close. Multi-day skipped."""
    if is_0dte and minutes_to_close <= cfg.eod_cutoff_minutes_before_close:
        return _exit(
            ExitReason.EOD_CUTOFF,
            f"EOD cutoff: {minutes_to_close:.0f}min to close",
            debug=debug,
        )
    return None


def check_bid_disappearance_gate(
    bid: float,
    seconds_at_zero_bid: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 2: Exit when bid has been zero for too long (no buyers)."""
    result = check_bid_disappearance(bid, seconds_at_zero_bid, cfg.defensive)
    if result["should_exit"]:
        debug["bid_check"] = result
        return _exit(ExitReason.BID_DISAPPEARANCE, result["reason"], debug=debug)
    return None


def check_profit_target(
    gain: float,
    is_0dte: bool,
    is_index: bool,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 3: Take profit at fixed % — index 0DTE only.

    Indexes (SPY, QQQ, IWM) are more predictable. Locking in 30% gains
    on 0DTE produces 100% win rate on this gate (10 trades, $1,401 total
    in backtest).
    """
    if not is_0dte or not is_index:
        return None
    if cfg.profit_target_index_0dte_pct <= 0:
        return None
    if gain >= cfg.profit_target_index_0dte_pct:
        return _exit(
            ExitReason.PROFIT_TARGET,
            f"Profit target: +{gain:.1f}% >= {cfg.profit_target_index_0dte_pct}% (index 0DTE)",
            debug=debug,
        )
    return None


def check_scalp_trail(
    peak_gain: float,
    gain: float,
    is_0dte: bool,
    underlying_confirms: bool,
    underlying_against: bool,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 4: Quick scalp exit when early peak fades.

    Peaked at +20%+, faded to <60% of peak, gain still positive.
    DTE-aware:
      0DTE: exit if underlying NOT confirming (stricter)
      Multi-day: exit only if underlying actively AGAINST (more patient)
    """
    if peak_gain < cfg.scalp_peak_threshold_pct:
        return None
    if gain <= 0 or gain >= peak_gain * cfg.scalp_fade_ratio:
        return None

    # 0DTE: exit when underlying doesn't confirm the move
    if is_0dte and not underlying_confirms:
        return _exit(
            ExitReason.SCALP_TRAIL,
            f"Scalp (0DTE): peaked +{peak_gain:.0f}%, faded to +{gain:.1f}%, "
            f"underlying not confirming",
            debug=debug,
        )

    # Multi-day: only exit if underlying is actively against
    if not is_0dte and underlying_against:
        return _exit(
            ExitReason.SCALP_TRAIL,
            f"Scalp (multi-day): peaked +{peak_gain:.0f}%, faded to +{gain:.1f}%, "
            f"underlying against",
            debug=debug,
        )

    return None


def check_checkpoint_cut(
    is_0dte: bool,
    drop_entry: float,
    has_underlying: bool,
    underlying_against: bool,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 5: Checkpoint cut — 0DTE only.

    Exit when premium is down 30%+ AND underlying confirms against 0.5%+.
    Disabled for multi-day trades (they recover from temporary dips).
    """
    if not is_0dte:
        return None
    if drop_entry < cfg.checkpoint_drop_pct:
        return None
    if not has_underlying or not underlying_against:
        return None

    return _exit(
        ExitReason.CHECKPOINT_CUT,
        f"Checkpoint: premium -{drop_entry:.1f}% AND underlying against",
        debug=debug,
    )


def check_graduated_stop(
    drop_entry: float,
    is_0dte: bool,
    underlying_against: bool,
    u_move: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 6: Graduated stop — underlying-based, DTE-aware.

    Two thresholds per DTE:
      If underlying against: tight stop (35% 0DTE, 52% multi-day)
      If underlying NOT against: backstop (65% 0DTE, 75% multi-day)
    """
    if is_0dte:
        tight = cfg.tight_stop_0dte_pct
        backstop = cfg.backstop_0dte_pct
    else:
        tight = cfg.tight_stop_multiday_pct
        backstop = cfg.backstop_multiday_pct

    if underlying_against:
        if drop_entry >= tight:
            return _exit(
                ExitReason.CONFIRMED_STOP,
                f"Confirmed stop: premium -{drop_entry:.1f}% >= {tight}% "
                f"AND underlying {u_move:+.2f}% against",
                debug=debug,
            )
    else:
        if drop_entry >= backstop:
            return _exit(
                ExitReason.HARD_STOP,
                f"Backstop: premium -{drop_entry:.1f}% >= {backstop}% "
                f"(underlying {u_move:+.2f}% not against)",
                debug=debug,
            )

    return None


def check_soft_trail(
    current_premium: float,
    entry_premium: float,
    peak_premium: float,
    peak_gain: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 7: Soft trail for the 10-50% peak gain band.

    Floor at 60% of peak-to-entry gain. Protects early gains.
    (Sweep showed 60% keep is more consistent than 50%.)
    """
    if not (cfg.soft_trail_band_low_pct <= peak_gain < cfg.soft_trail_band_high_pct):
        return None

    floor = entry_premium + (peak_premium - entry_premium) * cfg.soft_trail_keep_pct

    # Don't "protect gains" when the trade is currently at a loss.
    # This prevents DCA (which lowers entry) from immediately triggering
    # soft trail on the same cycle when the position is still underwater.
    if floor <= entry_premium:
        return None

    if current_premium <= floor:
        return _exit(
            ExitReason.SOFT_TRAIL,
            f"Soft trail: peak +{peak_gain:.0f}%, "
            f"floor=${floor:.2f}, current=${current_premium:.2f}",
            debug=debug,
        )
    return None


def check_adaptive_trail(
    peak_gain: float,
    drop_peak: float,
    tiers: tuple[AdaptiveTier, ...],
    debug: dict,
) -> ExitAction | None:
    """Gate 8: Category-aware adaptive trailing stop.

    Tiers are checked highest-first (moonshot → runner → active).
    High-vol tickers get wider trails, indexes get tighter ones.
    """
    for tier in tiers:
        if peak_gain >= tier.min_peak_gain and drop_peak >= tier.trail_width:
            return _exit(
                ExitReason.ADAPTIVE_TRAIL,
                f"Adaptive trail: peak +{peak_gain:.0f}% (tier {tier.min_peak_gain}%), "
                f"dropped {drop_peak:.1f}% >= {tier.trail_width}%",
                debug=debug,
            )
    return None


def check_breakeven_ratchet(
    gain: float,
    current_premium: float,
    entry_premium: float,
    armed: bool,
    trigger_pct: float,
    debug: dict,
) -> tuple[ExitAction | None, bool]:
    """V6 Gate: Break-even ratchet.

    Once a trade reaches +trigger_pct% gain, the stop floor moves to entry price.
    If premium subsequently drops below entry, exit at break-even.

    Returns (action_or_None, new_armed_state). Caller must persist the armed
    state on TradeState to avoid re-arming checks every cycle.
    """
    new_armed = armed
    if not armed and gain >= trigger_pct:
        new_armed = True

    if new_armed and current_premium < entry_premium:
        return _exit(
            ExitReason.BREAKEVEN_RATCHET,
            f"Break-even ratchet: was +{trigger_pct:.0f}%, now ${current_premium:.2f} "
            f"< entry ${entry_premium:.2f}",
            debug=debug,
        ), new_armed

    return None, new_armed


def check_scaleout(
    gain: float,
    contracts: int,
    already_scaled: bool,
    scaleout_gain_pct: float,
    scaleout_fraction: float,
    min_contracts: int,
    debug: dict,
) -> ExitAction | None:
    """V6 Gate: Scale-out at +N% gain.

    Sells a fraction of contracts when gain first reaches the threshold.
    One-shot: once fired, does not re-fire (tracked by already_scaled).

    Returns ExitAction with contracts_to_close set for partial exit, or None.
    """
    if already_scaled:
        return None
    if contracts < min_contracts:
        return None
    if gain < scaleout_gain_pct:
        return None

    close_qty = max(1, int(contracts * scaleout_fraction))
    return ExitAction(
        should_exit=True,
        reason=ExitReason.SCALEOUT,
        detail=(
            f"V6 scale-out: +{gain:.1f}% >= {scaleout_gain_pct}%, "
            f"closing {close_qty}/{contracts} contracts"
        ),
        contracts_to_close=close_qty,
        debug=debug,
    )


def check_sideways_scalp(
    gain: float,
    peak_gain_from_history: float,
    premium_history: list[float],
    timestamp_history: list[float],
    underlying_history: list[float],
    entry_premium: float,
    entry_underlying: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate: Sideways scalp — detect range-bound trades and take small profits.

    Fires when ALL of:
      1. Current gain >= sideways_take_profit_pct (trade is profitable)
      2. Peak gain from history < sideways_peak_cap_pct (trade hasn't trended)
      3. Enough history accumulated (>= sideways_min_ticks)
      4. At least sideways_signals_needed of 4 indicators agree trade is sideways

    The 4 sideways indicators:
      A. Premium range-bound: (max-min)/entry over lookback window < range_pct%
      B. No new highs: peak premium was hit >= N minutes ago
      C. Underlying flat: moved < Y% from entry
      D. Entry crosses: premium crossed entry price N+ times (choppy)

    Mitigations for known issues:
      - Issue #1 (restart): history is empty after restart, gate silently disabled
        until min_ticks accumulate (~2.5min at 5s poll). Safe: fails to HOLD.
      - Issue #2 (memory): history capped at MAX_HISTORY_LEN in TradeState (bounded).
      - Issue #4 (DCA): uses current entry_premium (blended after DCA), which is
        correct — gain and crosses are relative to actual cost basis.
      - Issue #5 (scaleout): can fire on remaining contracts after scaleout. Desirable —
        if remaining position goes sideways, scalp it rather than wait for soft_trail.
      - Issue #6 (ordering): placed before soft_trail in gate priority. Higher-priority
        gates (profit_target, breakeven, scaleout, scalp_trail, graduated_stop) still
        fire first. Sideways scalp only catches what those gates miss.

    timestamp_history: elapsed seconds from entry for each premium tick (monotonic).
    """
    n = len(premium_history)
    if n < cfg.sideways_min_ticks:
        return None

    # Must be profitable enough to scalp
    if gain < cfg.sideways_take_profit_pct:
        return None

    # Don't scalp trades that trended — let adaptive trail handle those
    if peak_gain_from_history >= cfg.sideways_peak_cap_pct:
        return None

    lookback = cfg.sideways_lookback
    window = premium_history[-lookback:] if n > lookback else premium_history
    signals_hit = 0

    # Indicator A: Premium range-bound
    if entry_premium > 0:
        prem_range = (max(window) - min(window)) / entry_premium * 100
    else:
        prem_range = 999.0
    if prem_range < cfg.sideways_range_pct:
        signals_hit += 1

    # Indicator B: No new highs (use actual elapsed seconds, not tick count)
    peak_val = max(premium_history)
    peak_idx = premium_history.index(peak_val)
    if len(timestamp_history) == n and peak_idx < n:
        sec_since_peak = timestamp_history[-1] - timestamp_history[peak_idx]
        min_since_peak = sec_since_peak / 60.0
    else:
        # Fallback: estimate from tick count at 5s intervals
        min_since_peak = (n - 1 - peak_idx) * 5.0 / 60.0
    if min_since_peak >= cfg.sideways_no_new_high_min:
        signals_hit += 1

    # Indicator C: Underlying flat
    if underlying_history and entry_underlying > 0:
        u_move = abs(underlying_history[-1] - entry_underlying) / entry_underlying * 100
        if u_move < cfg.sideways_underlying_flat_pct:
            signals_hit += 1
    # If no underlying data, this indicator simply doesn't contribute

    # Indicator D: Entry cross count
    crosses = 0
    above = premium_history[0] >= entry_premium
    for p in premium_history[1:]:
        now_above = p >= entry_premium
        if now_above != above:
            crosses += 1
            above = now_above
    if crosses >= cfg.sideways_cross_count:
        signals_hit += 1

    if signals_hit < cfg.sideways_signals_needed:
        return None

    debug["sideways"] = {
        "signals_hit": signals_hit,
        "prem_range_pct": round(prem_range, 1),
        "min_since_peak": round(min_since_peak, 1),
        "crosses": crosses,
        "history_len": n,
        "peak_gain_history": round(peak_gain_from_history, 1),
    }
    return _exit(
        ExitReason.SIDEWAYS_SCALP,
        f"Sideways scalp: +{gain:.1f}% gain, {signals_hit}/4 indicators "
        f"(range={prem_range:.0f}%, no_high={min_since_peak:.0f}min, crosses={crosses})",
        debug=debug,
    )


def check_theta_exit(
    is_0dte: bool,
    elapsed_min: float,
    drop_entry: float,
    cfg: V5Config,
    debug: dict,
) -> ExitAction | None:
    """Gate 9: Theta exit — DTE-aware stale trade cutter.

    0DTE (theta bleed): 120min+ held and down 30%+ from entry.
    Multi-day (theta timer): 180min+ held and down 15%+ from entry.
      Multi-day theta timer prevents holding underwater positions forever.
      Backtested: cuts stale multi-day losers that v5b held too long.
    """
    if is_0dte:
        if elapsed_min >= cfg.theta_bleed_min and drop_entry >= cfg.theta_bleed_drop_pct:
            return _exit(
                ExitReason.THETA_BLEED,
                f"Theta bleed (0DTE): {elapsed_min:.0f}min held, down {drop_entry:.1f}%",
                debug=debug,
            )
    else:
        if (cfg.theta_timer_minutes > 0
                and elapsed_min >= cfg.theta_timer_minutes
                and drop_entry >= cfg.theta_timer_loss_pct):
            return _exit(
                ExitReason.THETA_TIMER,
                f"Theta timer (multi-day): {elapsed_min:.0f}min held, down {drop_entry:.1f}%",
                debug=debug,
            )

    return None
