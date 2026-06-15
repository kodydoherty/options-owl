"""Tests for regime-triggered stop tightening (spec 08) in V5 FSM exit engine.

The regime tighten feature allows position_monitor to tighten adaptive trail
widths when the market regime flips against open positions. It stacks
multiplicatively with the existing 2PM tightening.
"""

from datetime import datetime
from types import SimpleNamespace


from options_owl.risk.exit_v5.config import V5Config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_settings(**overrides):
    """Create a mock settings object with V6 defaults."""
    defaults = {
        "ENABLE_V6_BREAKEVEN_RATCHET": True,
        "V6_BREAKEVEN_TRIGGER_PCT": 20.0,
        "ENABLE_V6_SCALEOUT": False,
        "ENABLE_V6_2PM_TIGHTEN": True,
        "V6_2PM_TRAIL_TIGHTEN_FACTOR": 0.7,
        "V6_2PM_SOFT_TRAIL_BOOST": 0.15,
        "ENABLE_SCALP_TARGET": False,
        "ENABLE_V6_EARLY_POP_GATE": False,
        "ENABLE_V6_SIDEWAYS_SCALP": False,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_trade_state(
    ticker="MSFT",
    entry_premium=1.00,
    peak_premium=1.50,
    entry_time=None,
    option_type="call",
    dte=0,
    expiry_date="2026-05-29",
    contracts=5,
    entry_underlying=100.0,
):
    """Create a TradeState with sensible defaults for testing."""
    if entry_time is None:
        entry_time = datetime(2026, 5, 29, 10, 0)
    return TradeState(
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=entry_premium,
        entry_time=entry_time,
        contracts=contracts,
        peak_premium=peak_premium,
        dte=dte,
        expiry_date=expiry_date,
        entry_underlying_price=entry_underlying,
        last_underlying_price=entry_underlying,
    )


PRE_2PM = datetime(2026, 5, 29, 10, 30)
POST_2PM = datetime(2026, 5, 29, 14, 30)

# For 0DTE tests, underlying must confirm the move (price UP for a CALL)
# so that scalp_trail (gate 4) does not fire before adaptive_trail (gate 8).
# Scalp trail fires when: peak >= 20%, gain > 0, gain < peak * 0.6, underlying NOT confirming.
# Setting underlying 1% above entry satisfies the confirm threshold (0.2%).
CONFIRMING_UNDERLYING = 101.0  # 1% above entry_underlying of 100.0


# ── apply_regime_tighten / clear_regime_tighten ─────────────────────────────


class TestApplyRegimeTighten:
    """Tests for the regime tighten API on ExitFSM."""

    def test_default_regime_tighten_factor_is_1(self):
        """Default _regime_tighten_factor is 1.0 (no tightening)."""
        fsm = ExitFSM(V5Config())
        assert fsm._regime_tighten_factor == 1.0

    def test_apply_regime_tighten_sets_factor(self):
        """apply_regime_tighten(0.60) sets factor to 0.60."""
        fsm = ExitFSM(V5Config())
        fsm.apply_regime_tighten(0.60)
        assert fsm._regime_tighten_factor == 0.60

    def test_clear_regime_tighten_resets_to_1(self):
        """clear_regime_tighten() resets factor to 1.0."""
        fsm = ExitFSM(V5Config())
        fsm.apply_regime_tighten(0.60)
        assert fsm._regime_tighten_factor == 0.60
        fsm.clear_regime_tighten()
        assert fsm._regime_tighten_factor == 1.0


# ── FSM evaluation with regime tightening ────────────────────────────────────


class TestRegimeTightenAdaptiveTrail:
    """Tests that regime tightening affects adaptive trail widths during FSM evaluation.

    Key setup notes:
    - Underlying must confirm the CALL direction (price > entry + 0.2%) to bypass
      scalp_trail (gate 4) which fires before adaptive_trail (gate 8).
    - Peak gain must be >= 50% to bypass soft_trail (gate 7, band 15-50%).
    - Trade must be past grace period (5 min).
    """

    def test_regime_tighten_produces_tighter_trail(self):
        """Trade with regime tighten 0.60 should have tighter adaptive trails than without.

        STANDARD adaptive tiers: active = AdaptiveTier(30, 35).
        With 0.60 regime factor: trail_width becomes 35 * 0.60 = 21.0.
        A drop_from_peak of 25% should EXIT with regime tighten but HOLD without.
        """
        cfg = V5Config()
        settings = _make_settings(ENABLE_V6_2PM_TIGHTEN=False)

        # entry=1.00, peak=1.60 (+60%, above soft_trail_band_high 50%)
        # current=1.20 (drop from peak = 25%), gain=+20%
        # Underlying confirms => scalp_trail won't fire

        # Without regime tighten: trail_width = 35%, drop = 25% < 35% => HOLD
        fsm_normal = ExitFSM(cfg, settings=settings)
        action_normal = fsm_normal.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=1.20,
            bid=1.19, ask=1.21,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert not action_normal.should_exit, (
            f"Expected HOLD without regime tighten, got exit: {action_normal.reason}"
        )

        # With regime tighten 0.60: trail_width = 35 * 0.60 = 21%, drop = 25% >= 21% => EXIT
        fsm_tightened = ExitFSM(cfg, settings=settings)
        fsm_tightened.apply_regime_tighten(0.60)
        action_tightened = fsm_tightened.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=1.20,
            bid=1.19, ask=1.21,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert action_tightened.should_exit
        assert action_tightened.reason == ExitReason.ADAPTIVE_TRAIL

    def test_regime_and_2pm_stack_multiplicatively(self):
        """Regime tighten + 2PM tighten stack multiplicatively (0.60 x 0.70 = 0.42).

        STANDARD active tier trail_width = 35%.
        Combined factor = 0.42 => effective trail = 35 * 0.42 = 14.7%.
        A 16% drop from peak should trigger with both, but not with either alone.
        """
        cfg = V5Config()

        # entry=1.00, peak=1.60 (+60%), current=1.344 (drop from peak = 16%)
        # gain = +34.4%, underlying confirms
        current_premium = 1.344

        # 2PM only (factor = 0.70): effective trail = 35 * 0.70 = 24.5%. Drop 16% < 24.5% => HOLD
        fsm_2pm_only = ExitFSM(cfg, settings=_make_settings(ENABLE_V6_2PM_TIGHTEN=True))
        action_2pm = fsm_2pm_only.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=current_premium,
            bid=1.33, ask=1.35,
            now_et=POST_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert not action_2pm.should_exit, (
            f"Expected HOLD with 2PM-only tighten, got exit: {action_2pm.reason}"
        )

        # Regime only (factor = 0.60): effective trail = 35 * 0.60 = 21%. Drop 16% < 21% => HOLD
        fsm_regime_only = ExitFSM(cfg, settings=_make_settings(ENABLE_V6_2PM_TIGHTEN=False))
        fsm_regime_only.apply_regime_tighten(0.60)
        action_regime = fsm_regime_only.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=current_premium,
            bid=1.33, ask=1.35,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert not action_regime.should_exit, (
            f"Expected HOLD with regime-only tighten, got exit: {action_regime.reason}"
        )

        # Both stacked (0.60 * 0.70 = 0.42): effective trail = 35 * 0.42 = 14.7%.
        # Drop 16% >= 14.7% => EXIT
        fsm_both = ExitFSM(cfg, settings=_make_settings(ENABLE_V6_2PM_TIGHTEN=True))
        fsm_both.apply_regime_tighten(0.60)
        action_both = fsm_both.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=current_premium,
            bid=1.33, ask=1.35,
            now_et=POST_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert action_both.should_exit
        assert action_both.reason == ExitReason.ADAPTIVE_TRAIL

    def test_trade_survives_normal_trail_exits_with_regime_tighten(self):
        """A trade that holds under normal trail exits with regime-tightened trail.

        STANDARD active tier: trail_width = 35%.
        With regime 0.60: trail_width = 21%.
        Trade with 30% drop from peak: survives 35%, fails 21%.
        """
        cfg = V5Config()
        settings = _make_settings(ENABLE_V6_2PM_TIGHTEN=False)

        # entry=1.00, peak=1.60 (+60%), current=1.12 (drop from peak = 30%)
        # gain = +12%, underlying confirms
        current_premium = 1.12

        # Normal: 30% drop < 35% trail => HOLD
        fsm_normal = ExitFSM(cfg, settings=settings)
        action_normal = fsm_normal.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=current_premium,
            bid=1.11, ask=1.13,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert not action_normal.should_exit

        # Regime tightened: 30% drop >= 21% trail => EXIT
        fsm_tight = ExitFSM(cfg, settings=settings)
        fsm_tight.apply_regime_tighten(0.60)
        action_tight = fsm_tight.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=current_premium,
            bid=1.11, ask=1.13,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert action_tight.should_exit
        assert action_tight.reason == ExitReason.ADAPTIVE_TRAIL


# ── Integration with TradeState ──────────────────────────────────────────────


class TestRegimeTightenTradeStateIntegration:
    """Tests for regime tightening interacting with other trade state features."""

    def test_counter_trend_call_with_regime_tighten_exits_sooner(self):
        """Counter-trend CALL position with regime tighten fires adaptive trail sooner.

        Use multi-day (dte=1) so that scalp_trail (gate 4) only fires when
        underlying is actively AGAINST (not just "not confirming" as for 0DTE).
        For multi-day scalp trail: fires only when underlying_against AND fade.
        With underlying at 99.0 (against threshold 0.5%), underlying IS against,
        so we use a moderate drop that scalp trail won't catch but adaptive will
        with regime tightening.

        We avoid scalp_trail by keeping gain >= peak_gain * scalp_fade_ratio (0.6).
        peak_gain=60%, gain must be >= 36%. We set gain=37% (current=1.37).
        Drop from peak = (1.60 - 1.37) / 1.60 = 14.4%.
        Normal trail = 35%, 14.4% < 35% => HOLD.
        Regime 0.40 => trail = 35 * 0.40 = 14.0%, 14.4% >= 14.0% => EXIT.
        """
        cfg = V5Config()
        settings = _make_settings(ENABLE_V6_2PM_TIGHTEN=False)

        state_kwargs = dict(
            ticker="MSFT",
            entry_premium=1.00,
            peak_premium=1.60,
            option_type="call",
            entry_underlying=100.0,
            dte=1,
            expiry_date="2026-05-30",
        )

        # Normal: 14.4% drop < 35% trail => HOLD
        fsm_normal = ExitFSM(cfg, settings=settings)
        action_normal = fsm_normal.evaluate(
            state=_make_trade_state(**state_kwargs),
            current_premium=1.37,
            bid=1.36, ask=1.38,
            now_et=PRE_2PM,
            current_underlying=99.0,
        )
        assert not action_normal.should_exit, (
            f"Expected HOLD without regime tighten, got exit: {action_normal.reason}"
        )

        # With regime tighten 0.40: trail = 35 * 0.40 = 14.0%, 14.4% >= 14.0% => EXIT
        fsm_regime = ExitFSM(cfg, settings=settings)
        fsm_regime.apply_regime_tighten(0.40)
        action_regime = fsm_regime.evaluate(
            state=_make_trade_state(**state_kwargs),
            current_premium=1.37,
            bid=1.36, ask=1.38,
            now_et=PRE_2PM,
            current_underlying=99.0,
        )
        assert action_regime.should_exit
        assert action_regime.reason == ExitReason.ADAPTIVE_TRAIL

    def test_breakeven_ratchet_respected_with_regime_tighten(self):
        """Breakeven ratchet still fires even with regime tightening active.

        Once a trade hits +20% gain, the breakeven ratchet arms. If the trade
        then drops below entry, it should exit via breakeven ratchet regardless
        of regime tighten state. Breakeven ratchet (gate 3.5) fires before
        adaptive trail (gate 8), so it takes priority.
        """
        cfg = V5Config()
        settings = _make_settings(
            ENABLE_V6_BREAKEVEN_RATCHET=True,
            V6_BREAKEVEN_TRIGGER_PCT=20.0,
            ENABLE_V6_2PM_TIGHTEN=False,
        )

        # Trade that peaked at +25% (ratchet armed), now below entry
        state = _make_trade_state(
            entry_premium=1.00,
            peak_premium=1.25,
        )
        state.breakeven_ratchet_armed = True  # already armed from prior cycle

        fsm = ExitFSM(cfg, settings=settings)
        fsm.apply_regime_tighten(0.60)

        # Current premium = 0.95 (below entry of 1.00)
        action = fsm.evaluate(
            state=state,
            current_premium=0.95,
            bid=0.94, ask=0.96,
            now_et=PRE_2PM,
            current_underlying=100.0,
        )
        assert action.should_exit
        assert action.reason == ExitReason.BREAKEVEN_RATCHET

    def test_regime_tighten_debug_context(self):
        """When regime tighten is active, debug dict includes regime_tighten factor."""
        cfg = V5Config()
        settings = _make_settings(ENABLE_V6_2PM_TIGHTEN=False)

        fsm = ExitFSM(cfg, settings=settings)
        fsm.apply_regime_tighten(0.60)

        # Trade that will HOLD (small drop from peak, well within even tightened trail)
        action = fsm.evaluate(
            state=_make_trade_state(entry_premium=1.00, peak_premium=1.60),
            current_premium=1.55,
            bid=1.54, ask=1.56,
            now_et=PRE_2PM,
            current_underlying=CONFIRMING_UNDERLYING,
        )
        assert action.debug.get("regime_tighten") == 0.60
