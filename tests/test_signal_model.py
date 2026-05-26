"""Tests for ML signal model gate and scanner integration.

Covers:
  1. compute_option_features_from_live() — feature engineering from live data
  2. predict_entry_confidence() — model loading and prediction
  3. Scanner ML gate integration — veto/pass behavior
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from options_owl.sourcing.scoring.ml_gates.signal_model import (
    _model_cache,
    compute_option_features_from_live,
    predict_entry_confidence,
)


# ---------------------------------------------------------------------------
# compute_option_features_from_live
# ---------------------------------------------------------------------------


class TestComputeOptionFeaturesFromLive:
    """Tests for building feature dicts from live market data."""

    def test_returns_dict(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
        )
        assert isinstance(f, dict)
        assert len(f) > 20

    def test_time_features(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=15, is_call=True,
        )
        assert f["minutes_since_open"] == 15
        assert f["hour_bucket"] == 0
        assert f["is_first_30min"] == 1
        assert f["is_last_hour"] == 0

    def test_last_hour_flag(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=350, is_call=True,
        )
        assert f["is_first_30min"] == 0
        assert f["is_last_hour"] == 1

    def test_premium_change_with_history(self):
        history = [2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45]
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
            premium_history=history,
        )
        # premium_change_5m = (2.50 / history[-5] - 1) * 100
        expected_5m = (2.50 / 2.25 - 1) * 100
        assert abs(f["premium_change_5m"] - expected_5m) < 0.01

    def test_premium_change_no_history(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
        )
        assert f["premium_change_5m"] == 0
        assert f["premium_change_10m"] == 0
        assert f["premium_change_15m"] == 0

    def test_spread_features(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.40, ask=2.60,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
        )
        assert f["spread"] == pytest.approx(0.20)
        mid = (2.40 + 2.60) / 2
        assert f["spread_pct"] == pytest.approx(0.20 / mid * 100, rel=0.01)

    def test_greeks_passed_through(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.35, delta=-0.45, theta=-0.08, vega=0.12,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=False,
        )
        assert f["iv"] == 0.35
        assert f["delta"] == 0.45  # abs(delta)
        assert f["theta"] == -0.08
        assert f["vega"] == 0.12
        assert f["is_call"] == 0

    def test_volume_features_with_history(self):
        vol_history = [100, 150, 200, 250, 300, 500, 600, 700]
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
            volume_history=vol_history,
        )
        avg_vol = np.mean(vol_history)
        assert abs(f["volume_ratio"] - 1000 / avg_vol) < 0.01
        assert f["volume_trend"] > 0

    def test_underlying_change_with_history(self):
        underlying_hist = [545, 546, 547, 548, 549, 550]
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=552.0,
            minutes_since_open=60, is_call=True,
            underlying_history=underlying_hist,
        )
        # history[-5] = 546 (6 elements, index 1)
        expected_5m = (552.0 / 546 - 1) * 100
        assert f["underlying_change_5m"] == pytest.approx(expected_5m, rel=0.01)

    def test_coiled_spring_pattern(self):
        # Low volatility + high volume ratio = coiled_spring
        history = [2.50, 2.50, 2.50, 2.51, 2.50]  # very flat
        vol_history = [100, 100, 100, 100, 100]
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=500, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
            premium_history=history,
            volume_history=vol_history,
        )
        # premium_volatility should be very low, volume_ratio = 500/100 = 5.0
        assert f["volume_ratio"] == pytest.approx(5.0, rel=0.1)
        assert f["coiled_spring"] == 1

    def test_consecutive_bars(self):
        # 3 consecutive up bars
        history = [2.00, 1.95, 2.00, 2.10, 2.20, 2.30]
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.40, bid=2.35, ask=2.45,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
            premium_history=history,
        )
        assert f["consecutive_up_bars"] >= 3

    def test_no_nan_in_features(self):
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
            premium_history=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            volume_history=[100, 200, 300, 400, 500, 600],
            underlying_history=[545, 546, 547, 548, 549, 550],
        )
        for k, v in f.items():
            assert not (isinstance(v, float) and np.isnan(v)), f"NaN in feature {k}"


# ---------------------------------------------------------------------------
# predict_entry_confidence
# ---------------------------------------------------------------------------


class TestPredictEntryConfidence:
    """Tests for model loading and prediction."""

    def setup_method(self):
        _model_cache.clear()

    def test_no_model_returns_none_source(self):
        with patch("options_owl.sourcing.scoring.ml_gates.signal_model.MODELS_DIR",
                    Path("/nonexistent")):
            _model_cache.clear()
            result = predict_entry_confidence("SPY", {"premium": 2.5})
            assert result["model_source"] == "none"
            assert result["is_signal"] is False
            assert result["confidence"] == 0.0

    def test_model_loaded_and_predicts(self):
        """Test with a mock LightGBM booster."""
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.75])
        meta = {
            "features": ["premium", "delta", "volume_ratio"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }

        # No direction → cache key is just "SPY"
        _model_cache["SPY"] = (mock_booster, meta)

        result = predict_entry_confidence("SPY", {
            "premium": 2.5, "delta": 0.50, "volume_ratio": 1.5,
        })
        assert result["confidence"] == 0.75
        assert result["threshold"] == 0.5
        assert result["is_signal"] is True
        assert result["model_source"] == "per_ticker"
        mock_booster.predict.assert_called_once()

    def test_below_threshold_not_signal(self):
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.30])
        meta = {
            "features": ["premium"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_booster, meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5})
        assert result["is_signal"] is False
        assert result["confidence"] == 0.30

    def test_generic_fallback_model(self):
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.60])
        meta = {
            "features": ["premium"],
            "optimal_threshold": 0.45,
            "ticker": "GENERIC",
        }
        _model_cache["TSLA"] = (mock_booster, meta)

        result = predict_entry_confidence("TSLA", {"premium": 3.0})
        assert result["model_source"] == "generic"
        assert result["is_signal"] is True

    def test_runner_model_score(self):
        # Entry model
        mock_entry = MagicMock()
        mock_entry.predict.return_value = np.array([0.80])
        entry_meta = {
            "features": ["premium", "delta"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_entry, entry_meta)

        # Runner model (no direction → key is "runner_SPY")
        mock_runner = MagicMock()
        mock_runner.predict.return_value = np.array([0.65])
        runner_meta = {"features": ["premium", "volume_ratio"]}
        _model_cache["runner_SPY"] = (mock_runner, runner_meta)

        result = predict_entry_confidence("SPY", {
            "premium": 2.5, "delta": 0.50, "volume_ratio": 1.5,
        })
        assert result["runner_score"] == 0.65

    def test_missing_features_default_to_zero(self):
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.55])
        meta = {
            "features": ["premium", "delta", "nonexistent_feature"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_booster, meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5, "delta": 0.5})
        assert result["is_signal"] is True
        # Check the model was called with 0 for missing feature
        call_args = mock_booster.predict.call_args[0][0]
        assert call_args[0][2] == 0  # nonexistent_feature defaults to 0

    def test_empty_features_in_meta(self):
        mock_booster = MagicMock()
        meta = {"features": [], "optimal_threshold": 0.5, "ticker": "SPY"}
        _model_cache["SPY"] = (mock_booster, meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5})
        assert result["model_source"] == "none"
        assert result["is_signal"] is False

    # --- Combined model tests (direction-specific models removed — halved training data) ---

    def test_direction_param_uses_combined_model(self):
        """Direction param is accepted but combined model is always used."""
        mock_combined = MagicMock()
        mock_combined.predict.return_value = np.array([0.85])
        combined_meta = {
            "features": ["premium", "delta"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_combined, combined_meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5, "delta": 0.5}, direction="CALL")
        assert result["confidence"] == 0.85
        assert result["model_source"] == "per_ticker"
        assert result["is_signal"] is True

    def test_put_direction_uses_combined_model(self):
        """PUT direction also uses combined model (is_call feature handles direction)."""
        mock_combined = MagicMock()
        mock_combined.predict.return_value = np.array([0.40])
        combined_meta = {
            "features": ["premium"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_combined, combined_meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5}, direction="PUT")
        assert result["confidence"] == 0.40
        assert result["model_source"] == "per_ticker"
        assert result["is_signal"] is False  # below threshold

    def test_runner_model_uses_combined(self):
        """Runner model uses combined (non-direction-specific) model."""
        mock_entry = MagicMock()
        mock_entry.predict.return_value = np.array([0.80])
        entry_meta = {
            "features": ["premium"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_entry, entry_meta)

        # Combined runner model
        mock_runner = MagicMock()
        mock_runner.predict.return_value = np.array([0.72])
        runner_meta = {"features": ["premium"]}
        _model_cache["runner_SPY"] = (mock_runner, runner_meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5}, direction="CALL")
        assert result["runner_score"] == 0.72

    def test_no_direction_uses_combined(self):
        """When direction is empty, uses combined model (backward compat)."""
        mock_combined = MagicMock()
        mock_combined.predict.return_value = np.array([0.65])
        combined_meta = {
            "features": ["premium"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache["SPY"] = (mock_combined, combined_meta)

        result = predict_entry_confidence("SPY", {"premium": 2.5}, direction="")
        assert result["confidence"] == 0.65
        assert result["model_source"] == "per_ticker"


# ---------------------------------------------------------------------------
# Scanner ML gate integration
# ---------------------------------------------------------------------------


class TestScannerMLGate:
    """Test the ML gate integration in scan_ticker."""

    def _make_ctx(self):
        """Build a minimal SignalContext with candle data."""
        from options_owl.sourcing.scoring.types import (
            Direction,
            SignalContext,
            SignalState,
        )

        candles = [
            {"open": 550 + i * 0.1, "high": 551 + i * 0.1,
             "low": 549 + i * 0.1, "close": 550.5 + i * 0.1,
             "volume": 1000 + i * 10}
            for i in range(20)
        ]

        indicators = MagicMock()
        indicators.iv = 0.30
        indicators.ema_cross_strength = 0.1
        indicators.macd_line = 0.5
        indicators.vwap = 550.0
        indicators.last_close = 552.0
        indicators.rsi9 = 55.0
        indicators.volume_ratio = 1.5

        return SignalContext(
            ticker="SPY",
            scan_time="2026-05-21T14:00:00Z",
            state=SignalState.SCORED,
            direction=Direction.CALL,
            candles_5m=candles,
            indicators=indicators,
            score_total=75,
        )

    def test_run_ml_gate_returns_prediction(self):
        from options_owl.sourcing.scanner import _run_ml_gate

        ctx = self._make_ctx()

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.70])
        meta = {
            "features": ["premium", "delta", "minutes_since_open"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache.clear()
        # Combined model — cache key is just "SPY" (direction-specific removed)
        _model_cache["SPY"] = (mock_booster, meta)

        result = _run_ml_gate(ctx)
        assert result["confidence"] == 0.70
        assert result["is_signal"] is True

    def test_run_ml_gate_no_model(self):
        from options_owl.sourcing.scanner import _run_ml_gate

        ctx = self._make_ctx()
        _model_cache.clear()

        with patch("options_owl.sourcing.scoring.ml_gates.signal_model.MODELS_DIR",
                    Path("/nonexistent")):
            _model_cache.clear()
            result = _run_ml_gate(ctx)

        assert result["model_source"] == "none"

    def test_run_ml_gate_no_candles(self):
        from options_owl.sourcing.scanner import _run_ml_gate
        from options_owl.sourcing.scoring.types import (
            Direction,
            SignalContext,
            SignalState,
        )

        ctx = SignalContext(
            ticker="SPY",
            state=SignalState.SCORED,
            direction=Direction.CALL,
            candles_5m=None,
        )
        result = _run_ml_gate(ctx)
        assert result["model_source"] == "none"

    @pytest.mark.asyncio
    async def test_scan_ticker_ml_veto(self):
        """ML model below threshold should reject the signal."""
        from options_owl.sourcing.config import SourcingSettings
        from options_owl.sourcing.scanner import scan_ticker
        from options_owl.sourcing.scoring.types import SignalState

        settings = SourcingSettings(
            ENABLE_ML_SIGNAL_MODEL=True,
            SCORE_THRESHOLD=50,
        )

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.20])  # below threshold
        meta = {
            "features": ["premium", "delta"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache.clear()
        # Combined model — cache key is just "SPY"
        _model_cache["SPY"] = (mock_booster, meta)

        candles = [
            {"open": 550, "high": 552, "low": 549, "close": 551, "volume": 5000}
            for _ in range(20)
        ]

        with patch("options_owl.sourcing.scanner.fetch_candles", return_value=candles), \
             patch("options_owl.sourcing.scanner.compute_indicators") as mock_ind, \
             patch("options_owl.sourcing.scanner.compute_score") as mock_score:

            mock_ind.return_value = MagicMock(
                ema_cross_strength=0.1, macd_line=0.5, vwap=550.0,
                last_close=552.0, rsi9=55.0, volume_ratio=1.5, iv=0.3,
            )

            from options_owl.sourcing.scoring.types import ScoredSignal
            mock_score.return_value = ScoredSignal(score=75, rejected=False)

            ctx = await scan_ticker("SPY", settings)

        assert ctx is not None
        assert ctx.state == SignalState.REJECTED
        assert ctx.filter_result == "ml_veto"
        assert ctx.ml_confidence == 0.20

    @pytest.mark.asyncio
    async def test_scan_ticker_ml_pass(self):
        """ML model above threshold should not reject."""
        from options_owl.sourcing.config import SourcingSettings
        from options_owl.sourcing.scanner import scan_ticker
        from options_owl.sourcing.scoring.types import SignalState

        settings = SourcingSettings(
            ENABLE_ML_SIGNAL_MODEL=True,
            SCORE_THRESHOLD=50,
        )

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.80])  # above threshold
        meta = {
            "features": ["premium", "delta"],
            "optimal_threshold": 0.5,
            "ticker": "SPY",
        }
        _model_cache.clear()
        # Combined model — cache key is just "SPY"
        _model_cache["SPY"] = (mock_booster, meta)

        candles = [
            {"open": 550, "high": 552, "low": 549, "close": 551, "volume": 5000}
            for _ in range(20)
        ]

        with patch("options_owl.sourcing.scanner.fetch_candles", return_value=candles), \
             patch("options_owl.sourcing.scanner.compute_indicators") as mock_ind, \
             patch("options_owl.sourcing.scanner.compute_score") as mock_score, \
             patch("options_owl.sourcing.scanner.check_quality_gate", return_value=True), \
             patch("options_owl.sourcing.scanner.check_penalty_veto", return_value=False), \
             patch("options_owl.sourcing.scanner.run_veto_gates", return_value=(False, "")):

            mock_ind.return_value = MagicMock(
                ema_cross_strength=0.1, macd_line=0.5, vwap=550.0,
                last_close=552.0, rsi9=55.0, volume_ratio=1.5, iv=0.3,
            )

            from options_owl.sourcing.scoring.types import ScoredSignal
            mock_score.return_value = ScoredSignal(score=75, rejected=False)

            ctx = await scan_ticker("SPY", settings)

        assert ctx is not None
        assert ctx.state == SignalState.FILTERED  # passed through
        assert ctx.ml_confidence == 0.80
        assert ctx.ml_is_signal is True

    @pytest.mark.asyncio
    async def test_scan_ticker_ml_disabled(self):
        """ML gate disabled — signal passes without ML fields."""
        from options_owl.sourcing.config import SourcingSettings
        from options_owl.sourcing.scanner import scan_ticker

        settings = SourcingSettings(
            ENABLE_ML_SIGNAL_MODEL=False,
            SCORE_THRESHOLD=50,
        )

        candles = [
            {"open": 550, "high": 552, "low": 549, "close": 551, "volume": 5000}
            for _ in range(20)
        ]

        with patch("options_owl.sourcing.scanner.fetch_candles", return_value=candles), \
             patch("options_owl.sourcing.scanner.compute_indicators") as mock_ind, \
             patch("options_owl.sourcing.scanner.compute_score") as mock_score, \
             patch("options_owl.sourcing.scanner.check_quality_gate", return_value=True), \
             patch("options_owl.sourcing.scanner.check_penalty_veto", return_value=False), \
             patch("options_owl.sourcing.scanner.run_veto_gates", return_value=(False, "")):

            mock_ind.return_value = MagicMock(
                ema_cross_strength=0.1, macd_line=0.5, vwap=550.0,
                last_close=552.0, rsi9=55.0, volume_ratio=1.5, iv=0.3,
            )

            from options_owl.sourcing.scoring.types import ScoredSignal
            mock_score.return_value = ScoredSignal(score=75, rejected=False)

            ctx = await scan_ticker("SPY", settings)

        assert ctx is not None
        assert ctx.ml_confidence is None  # not set
        assert ctx.ml_is_signal is None
