"""Tests for all 5 scoring tiers + the scoring engine."""


from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.scoring.adjustments import tier4_risk
from options_owl.sourcing.scoring.amplifiers import tier3_amplifiers
from options_owl.sourcing.scoring.calibration import tier5_calibration
from options_owl.sourcing.scoring.direction import tier1_direction
from options_owl.sourcing.scoring.engine import compute_score
from options_owl.sourcing.scoring.timing import tier2_timing
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bullish_indicators() -> IndicatorSet:
    """Indicators for a strong bullish setup."""
    return IndicatorSet(
        ema9=105.0,
        ema21=100.0,
        ema200=90.0,
        ema_cross_strength=0.6,
        rsi9=58.0,
        macd_line=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        bb_upper=108.0,
        bb_lower=97.0,
        bb_mid=102.5,
        bb_width=0.03,
        bb_squeeze=False,
        vwap=101.0,
        vwap_slope=0.002,
        atr14=1.5,
        atr_expanding=True,
        keltner_upper=106.0,
        keltner_lower=96.0,
        volume_ratio=1.8,
        obv_slope=0.15,
        adx=30.0,
        last_close=104.0,
        last_high=105.0,
        last_low=103.0,
    )


def _bearish_indicators() -> IndicatorSet:
    """Indicators for a strong bearish setup."""
    return IndicatorSet(
        ema9=95.0,
        ema21=100.0,
        ema200=110.0,
        ema_cross_strength=-0.6,
        rsi9=38.0,
        macd_line=-0.5,
        macd_signal=-0.3,
        macd_histogram=-0.2,
        bb_upper=103.0,
        bb_lower=92.0,
        bb_mid=97.5,
        bb_width=0.03,
        bb_squeeze=False,
        vwap=99.0,
        vwap_slope=-0.002,
        atr14=1.5,
        atr_expanding=True,
        keltner_upper=105.0,
        keltner_lower=95.0,
        volume_ratio=1.6,
        obv_slope=-0.15,
        adx=28.0,
        last_close=96.0,
        last_high=97.0,
        last_low=95.0,
    )


def _make_ctx(direction: Direction, indicators: IndicatorSet | None = None) -> SignalContext:
    return SignalContext(
        ticker="SPY",
        scan_time="2026-05-21T10:30:00-04:00",
        state=SignalState.INDICATED,
        direction=direction,
        indicators=indicators,
    )


# ---------------------------------------------------------------------------
# Tier 1: Direction
# ---------------------------------------------------------------------------

class TestTier1Direction:
    def test_no_indicators(self):
        ctx = _make_ctx(Direction.CALL, indicators=None)
        result = tier1_direction(ctx)
        assert result.total == 0
        assert "no_indicators" in result.reasons

    def test_no_direction(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.direction = None
        result = tier1_direction(ctx)
        assert result.total == 0

    def test_bullish_call_high_score(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier1_direction(ctx)
        assert result.total >= 25  # strong bullish should score well
        assert result.max_possible == 40

    def test_bearish_put_high_score(self):
        ctx = _make_ctx(Direction.PUT, _bearish_indicators())
        result = tier1_direction(ctx)
        assert result.total >= 25

    def test_bearish_call_low_score(self):
        """CALL signal with bearish indicators should score poorly."""
        ctx = _make_ctx(Direction.CALL, _bearish_indicators())
        result = tier1_direction(ctx)
        assert result.total < 15  # misaligned direction

    def test_ema_cross_component_present(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier1_direction(ctx)
        assert "ema_cross" in result.components

    def test_stores_result_on_context(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        tier1_direction(ctx)
        assert ctx.tier1_direction is not None
        assert ctx.tier1_direction.total > 0


# ---------------------------------------------------------------------------
# Tier 2: Timing
# ---------------------------------------------------------------------------

class TestTier2Timing:
    def test_no_indicators(self):
        ctx = _make_ctx(Direction.CALL, indicators=None)
        result = tier2_timing(ctx)
        assert result.total == 0

    def test_strong_volume_high_score(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier2_timing(ctx)
        assert result.components.get("volume", 0) >= 6  # 1.8x volume ratio

    def test_volume_mandatory_gate(self):
        """Volume < 3 causes rejection in compute_score."""
        ind = _bullish_indicators()
        ind.volume_ratio = 0.3  # very low
        ind.obv_slope = 0.0
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier2_timing(ctx)
        # Low volume should score < 3 (mandatory gate threshold)
        assert result.components.get("volume", 0) < 3

    def test_rsi_sweet_spot(self):
        ind = _bullish_indicators()
        ind.rsi9 = 55.0  # ideal for CALL
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier2_timing(ctx)
        assert result.components.get("rsi", 0) == 5

    def test_rsi_overbought_penalized(self):
        ind = _bullish_indicators()
        ind.rsi9 = 82.0
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier2_timing(ctx)
        assert result.components.get("rsi", 0) == 0

    def test_expanding_atr_bonus(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier2_timing(ctx)
        assert result.components.get("atr_regime", 0) >= 4


# ---------------------------------------------------------------------------
# Tier 3: Amplifiers
# ---------------------------------------------------------------------------

class TestTier3Amplifiers:
    def test_no_indicators(self):
        ctx = _make_ctx(Direction.CALL, indicators=None)
        result = tier3_amplifiers(ctx)
        assert result.total == 0

    def test_squeeze_bonus(self):
        ind = _bullish_indicators()
        ind.bb_squeeze = True
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier3_amplifiers(ctx)
        assert result.components.get("squeeze", 0) >= 3

    def test_obv_confirming(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier3_amplifiers(ctx)
        assert result.components.get("obv", 0) >= 2

    def test_no_15m_gives_default(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.candles_15m = None
        result = tier3_amplifiers(ctx)
        assert result.components.get("multi_tf", 0) == 1

    def test_alpha_sources_bonus(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.insider_activity = {"placeholder": True}
        ctx.congress_activity = {"placeholder": True}
        result = tier3_amplifiers(ctx)
        assert result.components.get("alpha", 0) >= 2


# ---------------------------------------------------------------------------
# Tier 4: Risk Adjustments
# ---------------------------------------------------------------------------

class TestTier4Risk:
    def test_no_indicators(self):
        ctx = _make_ctx(Direction.CALL, indicators=None)
        result = tier4_risk(ctx)
        assert result.total == 0

    def test_no_penalties_clean_setup(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        result = tier4_risk(ctx)
        assert result.total >= -2  # minimal penalties for clean setup

    def test_rsi_overextended_penalty(self):
        ind = _bullish_indicators()
        ind.rsi9 = 88.0
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier4_risk(ctx)
        assert result.components.get("rsi_overextend", 0) == -5

    def test_wide_bb_penalty(self):
        ind = _bullish_indicators()
        ind.bb_width = 0.07
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier4_risk(ctx)
        assert result.components.get("wide_bb", 0) == -3

    def test_low_adx_penalty(self):
        ind = _bullish_indicators()
        ind.adx = 8.0
        ctx = _make_ctx(Direction.CALL, ind)
        result = tier4_risk(ctx)
        assert result.components.get("low_adx", 0) == -3

    def test_wide_spread_penalty(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.spread_pct = 35.0
        result = tier4_risk(ctx)
        assert result.components.get("spread", 0) == -4

    def test_capped_at_minus_15(self):
        ind = _bullish_indicators()
        ind.rsi9 = 90.0
        ind.bb_width = 0.08
        ind.adx = 5.0
        ctx = _make_ctx(Direction.CALL, ind)
        ctx.spread_pct = 40.0
        result = tier4_risk(ctx)
        assert result.total >= -15


# ---------------------------------------------------------------------------
# Tier 5: Calibration
# ---------------------------------------------------------------------------

class TestTier5Calibration:
    def test_tuesday_10am(self):
        """Tuesday at 10:00 AM ET = prime time."""
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.scan_time = "2026-05-19T10:00:00-04:00"  # Tuesday
        result = tier5_calibration(ctx)
        assert result.components.get("day_of_week", 0) == 3  # Tue
        assert result.components.get("time_of_day", 0) >= 4  # 30min after open

    def test_monday_penalty(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.scan_time = "2026-05-18T10:30:00-04:00"  # Monday
        result = tier5_calibration(ctx)
        assert result.components.get("day_of_week", 0) == 1

    def test_opening_drive(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        ctx.scan_time = "2026-05-20T09:35:00-04:00"  # 5min after open
        result = tier5_calibration(ctx)
        assert result.components.get("session", 0) == 5

    def test_stores_result_on_context(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        tier5_calibration(ctx)
        assert ctx.tier5_calibration is not None


# ---------------------------------------------------------------------------
# Full scoring engine (integration)
# ---------------------------------------------------------------------------

class TestScoringEngine:
    def test_bullish_signal_scores_above_threshold(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        scored = compute_score(ctx)
        assert scored.score >= 40  # strong bullish should clear threshold
        assert not scored.rejected

    def test_bearish_signal_aligned(self):
        ctx = _make_ctx(Direction.PUT, _bearish_indicators())
        scored = compute_score(ctx)
        assert scored.score >= 40
        assert not scored.rejected

    def test_low_volume_rejected(self):
        ind = _bullish_indicators()
        ind.volume_ratio = 0.2
        ind.obv_slope = 0.0
        ctx = _make_ctx(Direction.CALL, ind)
        scored = compute_score(ctx)
        assert scored.rejected
        assert scored.reject_reason == "insufficient_volume"

    def test_score_bounded_0_100(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        scored = compute_score(ctx)
        assert 0 <= scored.score <= 100

    def test_breakdown_has_all_tiers(self):
        ctx = _make_ctx(Direction.CALL, _bullish_indicators())
        scored = compute_score(ctx)
        assert "direction" in scored.breakdown
        assert "timing" in scored.breakdown
        assert "amplifiers" in scored.breakdown
        assert "risk" in scored.breakdown
        assert "calibration" in scored.breakdown

    def test_misaligned_direction_low_score(self):
        """CALL signal with bearish indicators should score poorly."""
        ctx = _make_ctx(Direction.CALL, _bearish_indicators())
        scored = compute_score(ctx)
        # Misaligned direction + risk penalties = low score
        assert scored.score < 50
