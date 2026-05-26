"""Tests for sourcing filters: quality gate, penalty veto, options validator."""

import pytest

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.data.options_provider import OptionsChain
from options_owl.sourcing.filters.options_validator import validate_chain
from options_owl.sourcing.filters.penalty_veto import check_penalty_veto
from options_owl.sourcing.filters.quality_gate import check_quality_gate
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState, TierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_ctx() -> SignalContext:
    """Context that should pass all gates."""
    ctx = SignalContext(
        ticker="NVDA",
        scan_time="2026-05-21T10:30:00-04:00",
        state=SignalState.SCORED,
        direction=Direction.CALL,
        score_total=72,
        tier1_direction=TierResult(total=28, max_possible=40, components={"ema_cross": 12}),
        tier2_timing=TierResult(total=20, max_possible=30, components={"volume": 8}),
        tier3_amplifiers=TierResult(total=6, max_possible=15, components={}),
        tier4_risk=TierResult(total=-2, max_possible=0, components={
            "rsi_overextend": 0, "wide_bb": 0, "low_adx": -2, "spread": 0,
        }),
        tier5_calibration=TierResult(total=10, max_possible=15, components={}),
    )
    return ctx


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_passes_good_signal(self):
        ctx = _good_ctx()
        assert check_quality_gate(ctx, threshold=60) is True
        assert ctx.filter_result == "passed"

    def test_rejects_below_threshold(self):
        ctx = _good_ctx()
        ctx.score_total = 55
        assert check_quality_gate(ctx, threshold=60) is False
        assert "score 55" in ctx.filter_reason

    def test_rejects_single_tier_fluke(self):
        ctx = _good_ctx()
        ctx.tier2_timing = TierResult(total=2, max_possible=30)
        ctx.tier3_amplifiers = TierResult(total=1, max_possible=15)
        ctx.tier5_calibration = TierResult(total=2, max_possible=15)
        # Only tier1 is meaningful — single tier fluke
        assert check_quality_gate(ctx, threshold=30) is False
        assert "single_tier_fluke" in ctx.filter_reason

    def test_rejects_weak_direction(self):
        ctx = _good_ctx()
        ctx.tier1_direction = TierResult(total=5, max_possible=40)
        assert check_quality_gate(ctx, threshold=30) is False
        assert "weak_direction" in ctx.filter_reason


# ---------------------------------------------------------------------------
# Penalty Veto
# ---------------------------------------------------------------------------

class TestPenaltyVeto:
    def test_no_veto_clean_setup(self):
        ctx = _good_ctx()
        assert check_penalty_veto(ctx) is False

    def test_veto_overextended_plus_choppy(self):
        ctx = _good_ctx()
        ctx.tier4_risk = TierResult(
            total=-8, max_possible=0,
            components={"rsi_overextend": -5, "low_adx": -3, "wide_bb": 0, "spread": 0},
        )
        assert check_penalty_veto(ctx) is True
        assert "overextended_rsi" in ctx.filter_reason

    def test_veto_wide_spread_low_volume(self):
        ctx = _good_ctx()
        ctx.tier4_risk = TierResult(
            total=-4, max_possible=0,
            components={"rsi_overextend": 0, "low_adx": 0, "wide_bb": 0, "spread": -4},
        )
        ctx.tier2_timing = TierResult(
            total=5, max_possible=30,
            components={"volume": 2},  # low volume
        )
        assert check_penalty_veto(ctx) is True
        assert "wide_spread" in ctx.filter_reason

    def test_veto_weak_direction_heavy_penalties(self):
        ctx = _good_ctx()
        ctx.tier1_direction = TierResult(total=5, max_possible=40)
        ctx.tier4_risk = TierResult(
            total=-6, max_possible=0,
            components={"rsi_overextend": -2, "low_adx": -1, "wide_bb": -3, "spread": 0},
        )
        assert check_penalty_veto(ctx) is True
        assert "weak_direction" in ctx.filter_reason

    def test_no_veto_without_risk_tier(self):
        ctx = _good_ctx()
        ctx.tier4_risk = None
        assert check_penalty_veto(ctx) is False


# ---------------------------------------------------------------------------
# Options Validator
# ---------------------------------------------------------------------------

class TestOptionsValidator:
    def test_good_chain_passes(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=2.50, ask=2.70,
            mid=2.60, spread_pct=7.7, volume=500, open_interest=2000, iv=0.35,
        )
        valid, reason = validate_chain(chain, score=85)
        assert valid is True
        assert reason == ""

    def test_premium_too_high_base(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=6.80, ask=7.20,
            mid=7.00, spread_pct=5.7, volume=500, open_interest=2000, iv=0.35,
        )
        valid, reason = validate_chain(chain, score=85)
        assert valid is False
        assert "premium_too_high" in reason

    def test_premium_cap_raised_for_high_score(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=6.80, ask=7.20,
            mid=7.00, spread_pct=5.7, volume=500, open_interest=2000, iv=0.35,
        )
        # Score 120+ raises cap to $7
        valid, _ = validate_chain(chain, score=120)
        assert valid is True

    def test_premium_cap_highest_tier(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=8.80, ask=9.20,
            mid=9.00, spread_pct=4.3, volume=500, open_interest=2000, iv=0.35,
        )
        # Score 150+ raises cap to $9
        valid, _ = validate_chain(chain, score=150)
        assert valid is True

    def test_premium_too_low(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=0.01, ask=0.03,
            mid=0.02, spread_pct=100.0, volume=0, open_interest=0, iv=0.0,
        )
        valid, reason = validate_chain(chain, score=85)
        assert valid is False
        assert "premium_too_low" in reason

    def test_wide_spread_rejected(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=1.00, ask=2.00,
            mid=1.50, spread_pct=66.7, volume=100, open_interest=500, iv=0.45,
        )
        valid, reason = validate_chain(chain, score=85)
        assert valid is False
        assert "spread_too_wide" in reason

    def test_no_liquidity_rejected(self):
        chain = OptionsChain(
            strike=550.0, expiry="2026-05-21", bid=2.50, ask=2.70,
            mid=2.60, spread_pct=7.7, volume=0, open_interest=5, iv=0.35,
        )
        valid, reason = validate_chain(chain, score=85)
        assert valid is False
        assert "no_liquidity" in reason
