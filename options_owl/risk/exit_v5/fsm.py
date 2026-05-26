"""Exit FSM v5 — category-aware, DTE-aware exit engine.

Gate priority (first match wins):
    1. EOD cutoff (0DTE only, 15min before close)
    2. Bid disappearance (30s zero bid)
    3. [5min grace — skip all exits]
    4. Profit target (index 0DTE: lock 30% gains)
    5. Scalp trail (peaked +20%, faded <60%, underlying doesn't confirm)
    6. Checkpoint cut (0DTE: down 30% AND underlying against 0.5%)
    7. Graduated stop (underlying-based: 35% if against, 65% backstop for 0DTE)
    8. Soft trail (10-50% peak band, keep 60%)
    9. Adaptive trail (category-aware: high-vol wider, index tighter)
   10. Theta exit (0DTE: 120min/30%; multi-day: 180min/15%)

States (informational, for logging):
    GRACE      → First 5min. Only EOD and bid disappearance can exit.
    DEVELOPING → After 5min, peak gain < 40%.
    TRAILING   → After 5min, peak gain >= 40%.

Categories (per-ticker adaptive trail widths):
    HIGH_VOL   → MSTR, AMD, TSLA, NVDA, etc. — wider trails
    INDEX      → SPY, QQQ, IWM — tighter trails + profit target
    STANDARD   → everything else — moderate trails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    TickerCategory,
    categorize_ticker,
)
from options_owl.risk.exit_v5.gates import (
    check_adaptive_trail,
    check_bid_disappearance_gate,
    check_breakeven_ratchet,
    check_checkpoint_cut,
    check_eod_cutoff,
    check_graduated_stop,
    check_profit_target,
    check_scalp_target,
    check_scalp_trail,
    check_scaleout,
    check_sideways_scalp,
    check_soft_trail,
    check_theta_exit,
)
from options_owl.risk.exit_v5.types import ExitAction, ExitReason, _exit, _hold

if TYPE_CHECKING:
    from options_owl.risk.exit_v5.config import V5Config


class FSMState(Enum):
    """Informational FSM states — used for logging, not gate logic."""
    GRACE = "GRACE"
    DEVELOPING = "DEVELOPING"
    TRAILING = "TRAILING"


# POLL_INTERVAL_SEC: assumed interval between evaluate() calls.
# Used for bid disappearance tracking. Must match position_monitor's loop.
POLL_INTERVAL_SEC = 5.0


@dataclass
class TradeState:
    """Mutable per-trade state. Created at fill, updated each evaluation cycle."""

    # Identity
    trade_id: int
    ticker: str
    option_type: str  # "call" or "put"

    # Entry conditions
    entry_premium: float = 0.0
    entry_time: datetime | None = None
    contracts: int = 1

    # Running state (updated each cycle)
    state: FSMState = FSMState.GRACE
    peak_premium: float = 0.0
    seconds_at_zero_bid: float = 0.0

    # Underlying tracking
    entry_underlying_price: float = 0.0
    last_underlying_price: float = 0.0

    # DTE awareness
    dte: int = 0
    expiry_date: str = ""

    # Category (set once on first evaluate() call via categorize_ticker)
    category: TickerCategory = TickerCategory.STANDARD
    _category_initialized: bool = False

    # V6: break-even ratchet (armed once gain first hits trigger %)
    breakeven_ratchet_armed: bool = False
    # V6: scale-out at +20% (one-shot, does not re-fire)
    scaled_out: bool = False

    # V6: early-pop tracking — records when peak premium was reached (minutes from entry)
    peak_elapsed_min: float = 0.0

    # Sideways scalp history — accumulated each evaluate() cycle.
    # Bounded by MAX_HISTORY_LEN to cap memory (~200 entries = ~17min at 5s poll).
    # Empty after restart (fails safe: gate disabled until min_ticks accumulate).
    premium_history: list[float] = field(default_factory=list)
    # Elapsed seconds from entry for each premium tick (monotonic, avoids datetime overhead)
    elapsed_sec_history: list[float] = field(default_factory=list)
    underlying_history: list[float] = field(default_factory=list)

    MAX_HISTORY_LEN: int = 200  # ~17 minutes at 5s poll interval


# ── FSM ─────────────────────────────────────────────────────────────────


class ExitFSM:
    """Stateless exit engine — v5 category-aware strategy with V6 enhancements.

    One instance per bot (or per-ticker when V6 per-ticker configs are enabled).
    All mutable state lives in TradeState.
    Call evaluate() every poll cycle (5s) with current market snapshot.

    V6 enhancements (all gated behind ENABLE_V6_* settings):
      - Break-even ratchet: once +20%, stop floor = entry price
      - Scale-out at +20%: sell 1/3 of contracts (partial exit)
      - 2PM trail tightening: tighten adaptive trails 30% after 2PM ET
    """

    def __init__(self, cfg: V5Config, settings=None):
        self.cfg = cfg
        self._settings = settings

    def evaluate(
        self,
        state: TradeState,
        current_premium: float,
        bid: float,
        ask: float,
        now_et: datetime,
        current_underlying: float = 0.0,
        minutes_to_close: float = 390.0,
        candle_data: dict | None = None,
    ) -> ExitAction:
        """Evaluate all exit gates in priority order. First match wins."""
        cfg = self.cfg

        # ── Guard: zero entry_premium ────────────────────────────────
        if state.entry_premium <= 0:
            return _exit(ExitReason.HARD_STOP,
                         "entry_premium <= 0 — force exit",
                         debug={"error": "zero_entry_premium"})

        # ── Update running state ─────────────────────────────────────
        if current_premium > state.peak_premium:
            state.peak_premium = current_premium
            state.peak_elapsed_min = _elapsed_minutes(state, now_et)

        # Set category once (idempotent after first call)
        if not state._category_initialized:
            state.category = categorize_ticker(state.ticker)
            state._category_initialized = True

        elapsed_min = _elapsed_minutes(state, now_et)
        gain = _gain_pct(current_premium, state.entry_premium)
        peak_gain = _gain_pct(state.peak_premium, state.entry_premium)
        drop_entry = max(0, -gain)
        drop_peak = (
            (state.peak_premium - current_premium) / state.peak_premium * 100
            if state.peak_premium > 0 else 0
        )

        # Bid disappearance tracking (assumes 5s poll interval)
        if bid <= 0:
            state.seconds_at_zero_bid += POLL_INTERVAL_SEC
        else:
            state.seconds_at_zero_bid = 0.0

        # Accumulate sideways history (bounded to MAX_HISTORY_LEN)
        state.premium_history.append(current_premium)
        elapsed_sec = elapsed_min * 60.0
        state.elapsed_sec_history.append(elapsed_sec)
        effective_u = current_underlying if current_underlying > 0 else (
            state.underlying_history[-1] if state.underlying_history
            else state.entry_underlying_price
        )
        state.underlying_history.append(effective_u)
        if len(state.premium_history) > state.MAX_HISTORY_LEN:
            state.premium_history = state.premium_history[-state.MAX_HISTORY_LEN:]
            state.elapsed_sec_history = state.elapsed_sec_history[-state.MAX_HISTORY_LEN:]
            state.underlying_history = state.underlying_history[-state.MAX_HISTORY_LEN:]

        # Underlying move + confirmation
        u_move, underlying_against, underlying_confirms, has_underlying = (
            _compute_underlying_move(state, current_underlying,
                                     cfg.underlying_against_threshold,
                                     cfg.scalp_confirm_threshold))

        # DTE (recompute from expiry_date each cycle)
        dte = _compute_dte(state, now_et)
        is_0dte = dte == 0
        is_index = state.category == TickerCategory.INDEX

        # State transition (informational)
        state.state = _compute_state(elapsed_min, peak_gain, cfg.grace_period_min)

        # Debug context
        debug = {
            "state": state.state.value,
            "category": state.category.value,
            "elapsed_min": round(elapsed_min, 1),
            "current_premium": current_premium,
            "peak_premium": state.peak_premium,
            "gain_pct": round(gain, 1),
            "peak_gain_pct": round(peak_gain, 1),
            "drop_entry_pct": round(drop_entry, 1),
            "drop_peak_pct": round(drop_peak, 1),
            "bid": bid, "ask": ask,
            "minutes_to_close": minutes_to_close,
            "dte": dte,
            "underlying_move": round(u_move, 2),
            "underlying_against": underlying_against,
            "underlying_confirms": underlying_confirms,
        }

        # ── Run gates in priority order ──────────────────────────────

        # Gate 1: EOD cutoff (0DTE only)
        action = check_eod_cutoff(is_0dte, minutes_to_close, cfg, debug)
        if action:
            return action

        # Gate 2: Bid disappearance
        action = check_bid_disappearance_gate(bid, state.seconds_at_zero_bid, cfg, debug)
        if action:
            return action

        # ── 5-minute grace — skip most exits, but backstop still fires ──
        if elapsed_min < cfg.grace_period_min:
            # Never let grace protect a catastrophic loss. The backstop fires
            # even during grace so a -95% position doesn't sit for 5 min.
            backstop_pct = cfg.backstop_multiday_pct if not is_0dte else cfg.backstop_0dte_pct
            if drop_entry >= backstop_pct:
                # Smart backstop: consult candle data before firing.
                # If candles say thesis is intact (HOLD), widen backstop by 15%
                # and let grace continue. This prevents cutting trades that
                # dip early but recover (e.g. AAPL -50% then back to profit).
                enrg_action = "PROCEED"
                enrg_reason = ""
                if candle_data and candle_data.get("indicators"):
                    from options_owl.collectors.candle_cache import evaluate_enrg
                    enrg_action, enrg_reason = evaluate_enrg(
                        candle_data, state.option_type,
                    )
                    debug["enrg_action"] = enrg_action
                    debug["enrg_reason"] = enrg_reason

                if enrg_action == "HOLD":
                    # Thesis intact — widen backstop by 15% and continue grace
                    widened = backstop_pct * 1.15
                    if drop_entry >= widened:
                        return _exit(
                            ExitReason.HARD_STOP,
                            f"GRACE BACKSTOP (ENRG-widened): down {drop_entry:.0f}% "
                            f">= {widened:.0f}% | {enrg_reason}",
                            debug=debug,
                        )
                    return _hold(
                        f"GRACE+ENRG HOLD: down {drop_entry:.0f}% but candles "
                        f"bullish, widened backstop to {widened:.0f}% | {enrg_reason}",
                        debug=debug,
                    )
                elif enrg_action == "IMMEDIATE_EXIT":
                    # Candles confirm reversal — exit faster
                    return _exit(
                        ExitReason.HARD_STOP,
                        f"GRACE BACKSTOP (ENRG-confirmed): down {drop_entry:.0f}% "
                        f"AND candles bearish | {enrg_reason}",
                        debug=debug,
                    )
                else:
                    # No candle data or inconclusive — fire backstop as before
                    return _exit(
                        ExitReason.HARD_STOP,
                        f"GRACE BACKSTOP: down {drop_entry:.0f}% >= {backstop_pct:.0f}% backstop",
                        debug=debug,
                    )
            return _hold(
                f"GRACE: {elapsed_min:.1f}min elapsed, waiting for {cfg.grace_period_min}min",
                debug=debug)

        # Gate 3: Profit target (index 0DTE only — V6 per-ticker may change threshold)
        action = check_profit_target(gain, is_0dte, is_index, cfg, debug)
        if action:
            return action

        # V6 Gate 3.5: Break-even ratchet — once +N%, exit if drops below entry
        if self._settings and getattr(self._settings, "ENABLE_V6_BREAKEVEN_RATCHET", False):
            trigger_pct = getattr(self._settings, "V6_BREAKEVEN_TRIGGER_PCT", 20.0)
            action, state.breakeven_ratchet_armed = check_breakeven_ratchet(
                gain, current_premium, state.entry_premium,
                state.breakeven_ratchet_armed, trigger_pct, debug,
            )
            if action:
                return action

        # V6 Gate 3.7: Scale-out at +20% — sell fraction of contracts (partial exit)
        if self._settings and getattr(self._settings, "ENABLE_V6_SCALEOUT", False):
            action = check_scaleout(
                gain, state.contracts, state.scaled_out,
                scaleout_gain_pct=getattr(self._settings, "V6_SCALEOUT_GAIN_PCT", 20.0),
                scaleout_fraction=getattr(self._settings, "V6_SCALEOUT_FRACTION", 0.333),
                min_contracts=getattr(self._settings, "V6_SCALEOUT_MIN_CONTRACTS", 3),
                debug=debug,
            )
            if action:
                state.scaled_out = True
                return action

        # Gate 3.8: Scalp target — take +25% profit unless confirmed runner
        # Uses candle data to avoid nuking genuine runners
        if self._settings and getattr(self._settings, "ENABLE_SCALP_TARGET", False):
            action = check_scalp_target(
                gain=gain,
                peak_gain=peak_gain,
                elapsed_min=elapsed_min,
                underlying_confirms=underlying_confirms,
                candle_data=candle_data,
                option_type=state.option_type,
                scalp_target_pct=getattr(self._settings, "SCALP_TARGET_PCT", 25.0),
                runner_confirm_pct=getattr(self._settings, "SCALP_RUNNER_CONFIRM_PCT", 40.0),
                debug=debug,
            )
            if action:
                return action

        # Gate 4: Scalp trail (underlying-aware)
        action = check_scalp_trail(
            peak_gain, gain, is_0dte, underlying_confirms,
            underlying_against, cfg, debug)
        if action:
            return action

        # Gate 5: Checkpoint cut (0DTE only, underlying-confirmed)
        action = check_checkpoint_cut(
            is_0dte, drop_entry, has_underlying, underlying_against, cfg, debug)
        if action:
            return action

        # Gate 6: Graduated stop (underlying-based, DTE-aware)
        # V6: Early-pop backstop tightening — if trade peaked early and is now
        # fading, use a tighter backstop to cut losses sooner.
        # This creates a local config override (same pattern as 2PM tightening)
        # and does NOT mutate self.cfg.
        grad_cfg = cfg
        if (self._settings
                and getattr(self._settings, "ENABLE_V6_EARLY_POP_GATE", False)
                and _is_early_pop(state, elapsed_min, peak_gain, cfg)):
            from dataclasses import replace
            grad_cfg = replace(
                cfg,
                backstop_0dte_pct=cfg.early_pop_backstop_0dte_pct,
                backstop_multiday_pct=cfg.early_pop_backstop_multiday_pct,
            )
            debug["early_pop"] = True

        action = check_graduated_stop(
            drop_entry, is_0dte, underlying_against, u_move, grad_cfg, debug)
        if action:
            return action

        # Gate 6.5: Sideways scalp — detect choppy trades and take small profits
        if self._settings and getattr(self._settings, "ENABLE_V6_SIDEWAYS_SCALP", False):
            # Compute peak gain from actual history (not state.peak_premium which
            # may include pre-DCA peaks). This ensures the peak_cap guard is accurate.
            if state.premium_history and state.entry_premium > 0:
                hist_peak = max(state.premium_history)
                peak_gain_hist = (hist_peak - state.entry_premium) / state.entry_premium * 100
            else:
                peak_gain_hist = peak_gain

            action = check_sideways_scalp(
                gain=gain,
                peak_gain_from_history=peak_gain_hist,
                premium_history=state.premium_history,
                timestamp_history=state.elapsed_sec_history,
                underlying_history=state.underlying_history,
                entry_premium=state.entry_premium,
                entry_underlying=state.entry_underlying_price,
                cfg=cfg,
                debug=debug,
            )
            if action:
                return action

        # V6: 2PM trail tightening — tighter exits in the gamma death zone
        # Creates a local config override; does NOT mutate self.cfg.
        active_cfg = cfg
        if (self._settings
                and getattr(self._settings, "ENABLE_V6_2PM_TIGHTEN", False)
                and now_et.hour >= 14):
            from dataclasses import replace
            tighten = getattr(self._settings, "V6_2PM_TRAIL_TIGHTEN_FACTOR", 0.7)
            boost = getattr(self._settings, "V6_2PM_SOFT_TRAIL_BOOST", 0.15)
            # Tighten adaptive tiers for the ticker's category
            base_tiers = cfg.get_adaptive_tiers(state.category)
            tight_tiers = tuple(
                AdaptiveTier(t.min_peak_gain, t.trail_width * tighten)
                for t in base_tiers
            )
            new_keep = min(0.80, cfg.soft_trail_keep_pct + boost)
            kw = {"soft_trail_keep_pct": new_keep}
            if state.category == TickerCategory.HIGH_VOL:
                kw["adaptive_highvol_tiers"] = tight_tiers
            elif state.category == TickerCategory.INDEX:
                kw["adaptive_index_tiers"] = tight_tiers
            else:
                kw["adaptive_standard_tiers"] = tight_tiers
            active_cfg = replace(cfg, **kw)

        # Gate 7: Soft trail (10-50% peak band, keep 60%)
        action = check_soft_trail(
            current_premium, state.entry_premium, state.peak_premium,
            peak_gain, active_cfg, debug)
        if action:
            return action

        # Gate 8: Adaptive trail (category-aware tiers)
        tiers = active_cfg.get_adaptive_tiers(state.category)
        action = check_adaptive_trail(peak_gain, drop_peak, tiers, debug)
        if action:
            return action

        # Gate 9: Theta exit (0DTE: bleed; multi-day: timer)
        action = check_theta_exit(is_0dte, elapsed_min, drop_entry, cfg, debug)
        if action:
            return action

        # ── HOLD ─────────────────────────────────────────────────────
        return _hold(
            f"{state.state.value}: gain={gain:.1f}% peak={peak_gain:.1f}% "
            f"elapsed={elapsed_min:.0f}min dte={dte} cat={state.category.value}",
            debug=debug)


# ── Pure computation helpers ─────────────────────────────────────────────


def _compute_state(elapsed_min: float, peak_gain: float, grace_min: float) -> FSMState:
    """Determine FSM state based on time and peak gain."""
    if elapsed_min < grace_min:
        return FSMState.GRACE
    if peak_gain >= 40:
        return FSMState.TRAILING
    return FSMState.DEVELOPING


def _compute_underlying_move(
    state: TradeState,
    current_underlying: float,
    against_threshold: float,
    confirm_threshold: float,
) -> tuple[float, bool, bool, bool]:
    """Compute underlying price move relative to entry.

    Updates state.last_underlying_price as a side effect (caches last
    known good price for gaps in underlying data).

    Returns (u_move_pct, underlying_against, underlying_confirms, has_underlying).
    """
    effective = current_underlying
    if current_underlying > 0:
        state.last_underlying_price = current_underlying
    elif state.last_underlying_price > 0:
        effective = state.last_underlying_price

    has = state.entry_underlying_price > 0 and effective > 0
    if not has:
        return 0.0, False, False, False

    u_move = (effective - state.entry_underlying_price) / state.entry_underlying_price * 100
    is_call = state.option_type.lower() in ("call", "bullish", "long")

    if is_call:
        against = u_move < -against_threshold
        confirms = u_move > confirm_threshold
    else:
        against = u_move > against_threshold
        confirms = u_move < -confirm_threshold

    return u_move, against, confirms, True


def _compute_dte(state: TradeState, now_et: datetime) -> int:
    """Compute days to expiration from expiry_date string."""
    if not state.expiry_date:
        return state.dte
    try:
        exp = datetime.strptime(state.expiry_date, "%Y-%m-%d").date()
        return max(0, (exp - now_et.date()).days)
    except (ValueError, TypeError):
        return state.dte


def _elapsed_minutes(state: TradeState, now_et: datetime) -> float:
    """Compute minutes since trade entry."""
    if state.entry_time is None:
        return 0.0
    entry = state.entry_time.replace(tzinfo=None) if state.entry_time.tzinfo else state.entry_time
    now = now_et.replace(tzinfo=None) if now_et.tzinfo else now_et
    return max(0.0, (now - entry).total_seconds()) / 60.0


def _gain_pct(premium: float, entry: float) -> float:
    """Compute gain percentage: (premium - entry) / entry * 100."""
    if entry <= 0:
        return 0.0
    return (premium - entry) / entry * 100.0


def _is_early_pop(state: TradeState, elapsed_min: float,
                  peak_gain: float, cfg) -> bool:
    """Detect the early-pop-then-fade pattern for backstop tightening.

    Returns True when ALL conditions are met:
      1. We're past the check time (enough data to evaluate)
      2. Peak was reached within the early window
      3. Peak was meaningful (above min threshold)
      4. Premium has faded significantly from peak

    This is intentionally conservative — only fires when the pattern is clear.
    Backtested: +$1,737 at zero cost across 192 trades.
    """
    # Only evaluate after the check time
    if elapsed_min < cfg.early_pop_check_after_min:
        return False

    # Peak must have occurred within the early window
    if state.peak_elapsed_min > cfg.early_pop_peak_window_min:
        return False

    # Peak must be meaningful (not just noise)
    if peak_gain < cfg.early_pop_min_peak_gain_pct:
        return False

    # Premium must have faded from peak
    if state.peak_premium <= 0:
        return False
    current_prem = state.premium_history[-1] if state.premium_history else 0
    if current_prem <= 0:
        return False
    fade_pct = (state.peak_premium - current_prem) / state.peak_premium * 100
    return fade_pct >= cfg.early_pop_fade_pct
