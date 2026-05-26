"""Tests for rule-based ML gate stubs (flow, quality, regime)."""

from options_owl.sourcing.scoring.ml_gates.flow_classifier import predict_smart_money
from options_owl.sourcing.scoring.ml_gates.quality_predictor import predict_win_probability
from options_owl.sourcing.scoring.ml_gates.regime_weighter import predict_source_weights


# ── flow_classifier ──────────────────────────────────────────────────────────


class TestFlowClassifier:
    def test_neutral_with_no_features(self):
        """Empty dict should return the neutral baseline (~0.5)."""
        result = predict_smart_money({})
        assert result == 0.5

    def test_high_sweep_count_boosts(self):
        result = predict_smart_money({"sweep_count": 5})
        assert result == 0.65  # 0.5 + 0.15

    def test_very_high_sweep_count(self):
        result = predict_smart_money({"sweep_count": 10})
        assert result == 0.75  # 0.5 + 0.15 + 0.10

    def test_combined_smart_money_signals(self):
        result = predict_smart_money({
            "sweep_count": 10,
            "block_trade_volume": 20_000,
            "dark_pool_pct": 0.5,
            "net_premium_flow": 2_000_000,
            "unusual_volume_ratio": 4.0,
        })
        # 0.5 + 0.15 + 0.10 + 0.10 + 0.10 + 0.05 + 0.05 = 1.05 → clamped to 1.0
        assert result == 1.0

    def test_clamped_to_range(self):
        # Max possible should not exceed 1.0
        result = predict_smart_money({
            "sweep_count": 100,
            "block_trade_volume": 999_999,
            "dark_pool_pct": 0.99,
            "net_premium_flow": 50_000_000,
            "unusual_volume_ratio": 100.0,
        })
        assert 0.0 <= result <= 1.0

    def test_below_thresholds_stays_neutral(self):
        result = predict_smart_money({
            "sweep_count": 2,
            "block_trade_volume": 5000,
            "dark_pool_pct": 0.2,
            "net_premium_flow": 100_000,
            "unusual_volume_ratio": 1.5,
        })
        assert result == 0.5


# ── quality_predictor ────────────────────────────────────────────────────────


class TestQualityPredictor:
    def test_technical_only(self):
        """No ML confidence, no flow — pure technical base."""
        result = predict_win_probability({"technical_score": 80})
        assert result == 0.8

    def test_ml_blended(self):
        """ML confidence blends 60/40 with technical base."""
        result = predict_win_probability({
            "technical_score": 50,
            "ml_confidence": 0.9,
        })
        # 0.6 * 0.9 + 0.4 * 0.5 = 0.54 + 0.20 = 0.74
        assert abs(result - 0.74) < 1e-9

    def test_flow_boost(self):
        result = predict_win_probability({
            "technical_score": 70,
            "flow_score": 0.8,
        })
        assert abs(result - 0.75) < 1e-9  # 0.7 + 0.05

    def test_flow_below_threshold_no_boost(self):
        result = predict_win_probability({
            "technical_score": 70,
            "flow_score": 0.5,
        })
        assert abs(result - 0.70) < 1e-9

    def test_wide_spread_penalty(self):
        result = predict_win_probability({
            "technical_score": 70,
            "spread_pct": 25,
        })
        assert abs(result - 0.60) < 1e-9  # 0.7 - 0.10

    def test_opening_chaos_penalty(self):
        result = predict_win_probability({
            "technical_score": 70,
            "minutes_since_open": 5,
        })
        assert abs(result - 0.65) < 1e-9  # 0.7 - 0.05

    def test_eod_theta_penalty(self):
        result = predict_win_probability({
            "technical_score": 70,
            "minutes_since_open": 380,
        })
        assert abs(result - 0.65) < 1e-9  # 0.7 - 0.05

    def test_clamped_to_range(self):
        # Very low score with penalties
        result = predict_win_probability({
            "technical_score": 5,
            "spread_pct": 50,
            "minutes_since_open": 5,
        })
        assert 0.0 <= result <= 1.0

        # Very high score with all bonuses
        result = predict_win_probability({
            "technical_score": 100,
            "ml_confidence": 1.0,
            "flow_score": 0.9,
            "volume_ratio": 5.0,
        })
        assert 0.0 <= result <= 1.0


# ── regime_weighter ──────────────────────────────────────────────────────────


class TestRegimeWeighter:
    def test_normal_regime_default_weights(self):
        """VIX 20, ADX 20, no squeeze → all weights 1.0."""
        result = predict_source_weights({"vix": 20, "adx": 20})
        assert result == {
            "technical": 1.0,
            "flow": 1.0,
            "sentiment": 1.0,
            "macro": 1.0,
        }

    def test_high_vix_regime(self):
        result = predict_source_weights({"vix": 35, "adx": 20})
        assert result["technical"] == 0.7
        assert result["flow"] == 1.3
        assert result["macro"] == 1.5
        assert result["sentiment"] == 1.0

    def test_low_vix_regime(self):
        result = predict_source_weights({"vix": 12, "adx": 20})
        assert result["technical"] == 1.2
        assert result["flow"] == 0.8
        assert result["sentiment"] == 1.0
        assert result["macro"] == 1.0

    def test_trending_market(self):
        result = predict_source_weights({"vix": 20, "adx": 30})
        assert result["technical"] == 1.2

    def test_choppy_market(self):
        result = predict_source_weights({"vix": 20, "adx": 10})
        assert result["technical"] == 0.8
        assert result["flow"] == 1.2

    def test_big_move_day(self):
        result = predict_source_weights({"vix": 20, "adx": 20, "spy_change_1d": -3.0})
        assert result["flow"] == 1.3
        assert result["sentiment"] == 0.7

    def test_bb_squeeze(self):
        result = predict_source_weights({"vix": 20, "adx": 20, "bb_squeeze": True})
        assert result["technical"] == 1.1

    def test_combined_high_vix_trending(self):
        """High VIX + trending: technical gets both multipliers."""
        result = predict_source_weights({"vix": 35, "adx": 30})
        # technical: 1.0 * 0.7 * 1.2 = 0.84
        assert result["technical"] == 0.84
        assert result["flow"] == 1.3
        assert result["macro"] == 1.5

    def test_weights_rounded_to_2_decimals(self):
        result = predict_source_weights({"vix": 35, "adx": 30, "bb_squeeze": True})
        for v in result.values():
            assert v == round(v, 2)
