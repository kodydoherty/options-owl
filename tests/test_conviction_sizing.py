"""Tests for conviction-based position sizing (spec 09).

Covers:
  - compute_conviction_multiplier() in regime_detector.py
  - score_to_contracts() conviction_mult parameter in vinny_strategy.py
"""


from options_owl.risk.regime_detector import RegimeState, compute_conviction_multiplier
from options_owl.risk.vinny_strategy import score_to_contracts


# ---------------------------------------------------------------------------
# compute_conviction_multiplier tests
# ---------------------------------------------------------------------------


class TestConvictionMultiplierDisabled:
    def test_disabled_returns_1(self):
        """conviction_enabled=False always returns 1.0 regardless of inputs."""
        result = compute_conviction_multiplier(
            ml_confidence=0.50,
            regime=RegimeState.CHOPPY,
            direction="call",
            minute=300,
            conviction_enabled=False,
        )
        assert result == 1.0


class TestConvictionConfidenceTiers:
    def test_high_confidence_bullish_call_morning(self):
        """High conf (>=0.95) + aligned regime + morning -> 1.0."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=30,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=1.0 -> 1.0
        assert result == 1.0

    def test_medium_confidence_bullish_call_morning(self):
        """Medium conf (0.90-0.95) + aligned regime + morning -> 0.85."""
        result = compute_conviction_multiplier(
            ml_confidence=0.91,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=30,
            conviction_enabled=True,
        )
        # base=0.85, regime=1.0, time=1.0 -> 0.85
        assert result == 0.85

    def test_low_confidence_bullish_call_morning(self):
        """Low conf (<0.90) + aligned regime + morning -> 0.70."""
        result = compute_conviction_multiplier(
            ml_confidence=0.86,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=30,
            conviction_enabled=True,
        )
        # base=0.70, regime=1.0, time=1.0 -> 0.70
        assert result == 0.70


class TestConvictionRegimeAlignment:
    def test_choppy_regime_multiplies_by_070(self):
        """CHOPPY regime applies 0.70 multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.CHOPPY,
            direction="call",
            minute=30,
            conviction_enabled=True,
        )
        # base=1.0, regime=0.70, time=1.0 -> 0.70
        assert result == 0.70

    def test_counter_trend_bearish_call_multiplies_by_050(self):
        """Counter-trend (BEARISH + call) applies 0.50 multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BEARISH,
            direction="call",
            minute=30,
            conviction_enabled=True,
        )
        # base=1.0, regime=0.50, time=1.0 -> 0.50
        assert result == 0.50

    def test_counter_trend_bullish_put(self):
        """Counter-trend (BULLISH + put) applies 0.50 multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BULLISH,
            direction="put",
            minute=30,
            conviction_enabled=True,
        )
        # base=1.0, regime=0.50, time=1.0 -> 0.50
        assert result == 0.50

    def test_aligned_bearish_put(self):
        """Aligned regime (BEARISH + put) -> regime_mult=1.0."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BEARISH,
            direction="put",
            minute=30,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=1.0 -> 1.0
        assert result == 1.0


class TestConvictionTimeOfDay:
    def test_afternoon_multiplies_by_075(self):
        """Afternoon (minute > 210) applies 0.75 time multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=250,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=0.75 -> 0.75
        assert result == 0.75

    def test_late_morning_multiplies_by_090(self):
        """Late morning (61-120 min) applies 0.90 time multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=90,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=0.90 -> 0.90
        assert result == 0.90

    def test_midday_multiplies_by_080(self):
        """Midday (121-210 min) applies 0.80 time multiplier."""
        result = compute_conviction_multiplier(
            ml_confidence=0.96,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=150,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=0.80 -> 0.80
        assert result == 0.80


class TestConvictionFloorAndCap:
    def test_floor_worst_case_is_040(self):
        """Worst case: low conf + counter-trend + afternoon -> clamped to 0.40."""
        result = compute_conviction_multiplier(
            ml_confidence=0.80,
            regime=RegimeState.BEARISH,
            direction="call",
            minute=250,
            conviction_enabled=True,
        )
        # base=0.70, regime=0.50, time=0.75 -> 0.2625 -> clamped to 0.40
        assert result == 0.40

    def test_cap_best_case_never_exceeds_1(self):
        """Best case never exceeds 1.0."""
        result = compute_conviction_multiplier(
            ml_confidence=0.99,
            regime=RegimeState.BULLISH,
            direction="call",
            minute=10,
            conviction_enabled=True,
        )
        # base=1.0, regime=1.0, time=1.0 -> 1.0
        assert result == 1.0
        assert result <= 1.0


# ---------------------------------------------------------------------------
# score_to_contracts with conviction_mult tests
# ---------------------------------------------------------------------------

# Common sizing params for reproducible tests
_BALANCE = 23000.0
_COST = 200.0  # $2.00 premium * 100
_MAX_POS_PCT = 15.0
_MAX_CONCURRENT = 5
_RISK_PCT = 75.0


class TestScoreToContractsConviction:
    def test_conviction_1_backward_compatible(self):
        """conviction_mult=1.0 gives same result as default (no arg)."""
        baseline = score_to_contracts(
            score=90,
            cost_per_contract=_COST,
            balance=_BALANCE,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
        )
        with_mult = score_to_contracts(
            score=90,
            cost_per_contract=_COST,
            balance=_BALANCE,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
            conviction_mult=1.0,
        )
        assert baseline == with_mult

    def test_conviction_050_halves_contracts(self):
        """conviction_mult=0.50 should roughly halve the number of contracts."""
        full = score_to_contracts(
            score=90,
            cost_per_contract=_COST,
            balance=_BALANCE,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
            conviction_mult=1.0,
        )
        half = score_to_contracts(
            score=90,
            cost_per_contract=_COST,
            balance=_BALANCE,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
            conviction_mult=0.50,
        )
        # Full should be significantly more than half.
        # Due to int() truncation and the floor of 1, check the relationship holds.
        assert full > 0
        assert half > 0
        assert half <= full
        # With $23K balance and $200 cost, full ~ 14-17 contracts,
        # half ~ 7-8 contracts. Verify roughly halved.
        assert half <= (full // 2) + 1

    def test_conviction_040_small_balance_still_1_contract(self):
        """conviction_mult=0.40 with small balance still gives at least 1 contract."""
        result = score_to_contracts(
            score=90,
            cost_per_contract=200.0,
            balance=2000.0,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
            conviction_mult=0.40,
        )
        # balance=2000, deployable=1500, per_slot=300, *0.85(or ml)*0.40=~102
        # 102/200=0 -> floor to 1 (as long as position cap allows 1 contract)
        assert result >= 1

    def test_conviction_does_not_override_score_floor(self):
        """Score < 62 is still rejected regardless of conviction_mult."""
        result = score_to_contracts(
            score=61,
            cost_per_contract=_COST,
            balance=_BALANCE,
            max_position_pct=_MAX_POS_PCT,
            max_concurrent=_MAX_CONCURRENT,
            max_portfolio_risk_pct=_RISK_PCT,
            conviction_mult=1.0,
        )
        assert result == 0
