"""Tests for individual gate functions — each gate tested in isolation.

Tests match v5 category-aware logic (DTE-aware, underlying-based, per-category trails).
"""

from options_owl.risk.exit_v5.config import AdaptiveTier, TickerCategory, V5Config
from options_owl.risk.exit_v5.types import ExitReason
from options_owl.risk.exit_v5.gates import (
    check_adaptive_trail,
    check_bid_disappearance_gate,
    check_checkpoint_cut,
    check_eod_cutoff,
    check_graduated_stop,
    check_profit_target,
    check_scalp_trail,
    check_soft_trail,
    check_theta_exit,
)


def _cfg() -> V5Config:
    return V5Config()


def _debug() -> dict:
    return {}


# ── Gate 1: EOD cutoff ──────────────────────────────────────────────────

class TestCheckEodCutoff:

    def test_0dte_within_cutoff(self):
        action = check_eod_cutoff(is_0dte=True, minutes_to_close=10, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.EOD_CUTOFF

    def test_0dte_outside_cutoff(self):
        action = check_eod_cutoff(is_0dte=True, minutes_to_close=30, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_no_cutoff(self):
        """Multi-day trades skip EOD cutoff entirely."""
        action = check_eod_cutoff(is_0dte=False, minutes_to_close=10, cfg=_cfg(), debug=_debug())
        assert action is None


# ── Gate 2: Bid disappearance ────────────────────────────────────────────

class TestCheckBidDisappearanceGate:

    def test_zero_bid_over_timeout(self):
        action = check_bid_disappearance_gate(bid=0.0, seconds_at_zero_bid=35, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.BID_DISAPPEARANCE

    def test_positive_bid_no_exit(self):
        action = check_bid_disappearance_gate(bid=1.0, seconds_at_zero_bid=0, cfg=_cfg(), debug=_debug())
        assert action is None


# ── Gate 3: Profit target (index 0DTE only) ──────────────────────────────

class TestCheckProfitTarget:

    def test_fires_index_0dte_at_30pct(self):
        action = check_profit_target(gain=32.0, is_0dte=True, is_index=True, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.PROFIT_TARGET

    def test_holds_below_target(self):
        action = check_profit_target(gain=25.0, is_0dte=True, is_index=True, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_disabled_for_non_index(self):
        action = check_profit_target(gain=50.0, is_0dte=True, is_index=False, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_disabled_for_multiday(self):
        action = check_profit_target(gain=50.0, is_0dte=False, is_index=True, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_disabled_when_zero(self):
        cfg = V5Config(profit_target_index_0dte_pct=0.0)
        action = check_profit_target(gain=50.0, is_0dte=True, is_index=True, cfg=cfg, debug=_debug())
        assert action is None


# ── Gate 4: Scalp trail (underlying-aware) ───────────────────────────────

class TestCheckScalpTrail:

    def test_0dte_scalp_fires_without_confirm(self):
        """0DTE: exit when underlying doesn't confirm the move."""
        action = check_scalp_trail(
            peak_gain=25.0, gain=5.0, is_0dte=True,
            underlying_confirms=False, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.SCALP_TRAIL

    def test_0dte_scalp_holds_when_confirms(self):
        """0DTE: hold when underlying confirms the direction."""
        action = check_scalp_trail(
            peak_gain=25.0, gain=5.0, is_0dte=True,
            underlying_confirms=True, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_scalp_fires_when_against(self):
        """Multi-day: only exit if underlying is actively against."""
        action = check_scalp_trail(
            peak_gain=25.0, gain=5.0, is_0dte=False,
            underlying_confirms=False, underlying_against=True,
            cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.SCALP_TRAIL

    def test_multiday_scalp_holds_when_not_against(self):
        """Multi-day: hold even without confirmation — more patient."""
        action = check_scalp_trail(
            peak_gain=25.0, gain=5.0, is_0dte=False,
            underlying_confirms=False, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is None

    def test_scalp_no_fire_low_peak(self):
        action = check_scalp_trail(
            peak_gain=15.0, gain=5.0, is_0dte=True,
            underlying_confirms=False, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is None

    def test_scalp_no_fire_zero_gain(self):
        action = check_scalp_trail(
            peak_gain=25.0, gain=0.0, is_0dte=True,
            underlying_confirms=False, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is None

    def test_scalp_no_fire_gain_above_threshold(self):
        """gain = 16% >= 60% of peak 25% = 15% → hold."""
        action = check_scalp_trail(
            peak_gain=25.0, gain=16.0, is_0dte=True,
            underlying_confirms=False, underlying_against=False,
            cfg=_cfg(), debug=_debug())
        assert action is None


# ── Gate 5: Checkpoint cut (0DTE only, underlying-confirmed) ─────────────

class TestCheckCheckpointCut:

    def test_fires_when_down_30_and_against(self):
        """0DTE: exit when premium -30%+ AND underlying against."""
        action = check_checkpoint_cut(
            is_0dte=True, drop_entry=32, has_underlying=True,
            underlying_against=True, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.CHECKPOINT_CUT

    def test_holds_when_not_against(self):
        """Even at -30%, don't exit if underlying isn't against."""
        action = check_checkpoint_cut(
            is_0dte=True, drop_entry=32, has_underlying=True,
            underlying_against=False, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_holds_when_drop_below_threshold(self):
        action = check_checkpoint_cut(
            is_0dte=True, drop_entry=25, has_underlying=True,
            underlying_against=True, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_disabled_for_multiday(self):
        """Multi-day trades skip checkpoint entirely."""
        action = check_checkpoint_cut(
            is_0dte=False, drop_entry=50, has_underlying=True,
            underlying_against=True, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_no_underlying_no_exit(self):
        """Can't confirm checkpoint without underlying data."""
        action = check_checkpoint_cut(
            is_0dte=True, drop_entry=35, has_underlying=False,
            underlying_against=False, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_fires_repeatedly(self):
        """Checkpoint is NOT one-shot — fires every tick while conditions hold."""
        for _ in range(3):
            action = check_checkpoint_cut(
                is_0dte=True, drop_entry=32, has_underlying=True,
                underlying_against=True, cfg=_cfg(), debug=_debug())
            assert action is not None


# ── Gate 6: Graduated stop (underlying-based, DTE-aware) ─────────────────

class TestCheckGraduatedStop:

    def test_0dte_confirmed_stop_at_35pct(self):
        """0DTE: tight stop at 35% when underlying against."""
        action = check_graduated_stop(
            drop_entry=36, is_0dte=True, underlying_against=True,
            u_move=-0.6, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.CONFIRMED_STOP

    def test_0dte_backstop_at_65pct(self):
        """0DTE: backstop at 65% when underlying NOT against."""
        action = check_graduated_stop(
            drop_entry=66, is_0dte=True, underlying_against=False,
            u_move=0.1, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.HARD_STOP

    def test_0dte_holds_between_35_and_65(self):
        """0DTE: between tight and backstop, underlying not against → hold."""
        action = check_graduated_stop(
            drop_entry=50, is_0dte=True, underlying_against=False,
            u_move=0.1, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_confirmed_stop_at_52pct(self):
        """Multi-day: wider tight stop at 52%."""
        action = check_graduated_stop(
            drop_entry=53, is_0dte=False, underlying_against=True,
            u_move=-0.6, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.CONFIRMED_STOP

    def test_multiday_backstop_at_75pct(self):
        """Multi-day: wider backstop at 75%."""
        action = check_graduated_stop(
            drop_entry=76, is_0dte=False, underlying_against=False,
            u_move=0.1, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.HARD_STOP

    def test_multiday_holds_at_60pct(self):
        """Multi-day: 60% drop but not against → hold (tight=52, backstop=75)."""
        action = check_graduated_stop(
            drop_entry=60, is_0dte=False, underlying_against=False,
            u_move=0.1, cfg=_cfg(), debug=_debug())
        assert action is None


# ── Gate 7: Soft trail ───────────────────────────────────────────────────

class TestCheckSoftTrail:

    def test_fires_in_band(self):
        # peak 30% in [10, 50] band, floor = 1.00 + 0.30*0.60 = 1.18
        action = check_soft_trail(
            current_premium=1.10, entry_premium=1.00,
            peak_premium=1.30, peak_gain=30.0, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.SOFT_TRAIL

    def test_holds_above_floor(self):
        # floor = 1.00 + 0.30*0.60 = 1.18, current 1.20 > floor
        action = check_soft_trail(
            current_premium=1.20, entry_premium=1.00,
            peak_premium=1.30, peak_gain=30.0, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_no_fire_below_band(self):
        # peak gain 8% < 10% band low
        action = check_soft_trail(
            current_premium=1.05, entry_premium=1.00,
            peak_premium=1.08, peak_gain=8.0, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_no_fire_above_band(self):
        action = check_soft_trail(
            current_premium=1.40, entry_premium=1.00,
            peak_premium=1.60, peak_gain=60.0, cfg=_cfg(), debug=_debug())
        assert action is None


# ── Gate 8: Adaptive trail (category-aware tiers) ────────────────────────

class TestCheckAdaptiveTrail:

    def test_standard_active_stage(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.STANDARD)
        # standard active: peak >= 30%, drop >= 35%
        action = check_adaptive_trail(peak_gain=50.0, drop_peak=36, tiers=tiers, debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.ADAPTIVE_TRAIL

    def test_standard_runner_stage(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.STANDARD)
        action = check_adaptive_trail(peak_gain=120.0, drop_peak=41, tiers=tiers, debug=_debug())
        assert action is not None

    def test_standard_moonshot_stage(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.STANDARD)
        action = check_adaptive_trail(peak_gain=350.0, drop_peak=26, tiers=tiers, debug=_debug())
        assert action is not None

    def test_highvol_wider_trails(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.HIGH_VOL)
        # High-vol active: peak >= 40%, drop >= 50%
        # 45% drop < 50% → hold (wider trail)
        action = check_adaptive_trail(peak_gain=80.0, drop_peak=45, tiers=tiers, debug=_debug())
        assert action is None
        # 51% drop >= 50% → exit
        action = check_adaptive_trail(peak_gain=80.0, drop_peak=51, tiers=tiers, debug=_debug())
        assert action is not None

    def test_index_tighter_trails(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.INDEX)
        # Index active: peak >= 30%, drop >= 35%
        action = check_adaptive_trail(peak_gain=50.0, drop_peak=36, tiers=tiers, debug=_debug())
        assert action is not None

    def test_holds_below_threshold(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.STANDARD)
        # peak 50% but drop 30% < 35% trail width
        action = check_adaptive_trail(peak_gain=50.0, drop_peak=30, tiers=tiers, debug=_debug())
        assert action is None

    def test_below_min_peak_no_trail(self):
        tiers = _cfg().get_adaptive_tiers(TickerCategory.STANDARD)
        # peak 25% < 30% min
        action = check_adaptive_trail(peak_gain=25.0, drop_peak=50, tiers=tiers, debug=_debug())
        assert action is None


# ── Gate 9: Theta exit (DTE-aware) ───────────────────────────────────────

class TestCheckThetaExit:

    def test_0dte_bleed_fires(self):
        action = check_theta_exit(is_0dte=True, elapsed_min=125, drop_entry=32, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.THETA_BLEED

    def test_0dte_holds_before_120min(self):
        action = check_theta_exit(is_0dte=True, elapsed_min=115, drop_entry=32, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_0dte_holds_if_not_down_enough(self):
        action = check_theta_exit(is_0dte=True, elapsed_min=125, drop_entry=25, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_timer_fires(self):
        """Multi-day: theta timer at 180min+ and down 15%+."""
        action = check_theta_exit(is_0dte=False, elapsed_min=185, drop_entry=18, cfg=_cfg(), debug=_debug())
        assert action is not None
        assert action.reason == ExitReason.THETA_TIMER

    def test_multiday_holds_before_180min(self):
        action = check_theta_exit(is_0dte=False, elapsed_min=170, drop_entry=18, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_holds_if_not_down_enough(self):
        action = check_theta_exit(is_0dte=False, elapsed_min=185, drop_entry=10, cfg=_cfg(), debug=_debug())
        assert action is None

    def test_multiday_timer_disabled_when_zero(self):
        cfg = V5Config(theta_timer_minutes=0.0)
        action = check_theta_exit(is_0dte=False, elapsed_min=300, drop_entry=20, cfg=cfg, debug=_debug())
        assert action is None
