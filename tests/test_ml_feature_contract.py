"""Feature-contract tests for the ML entry models (FIX 3).

These tests enforce that:
  1. The live serve-time feature builder (the SINGLE source of truth in
     signal_model.compute_option_features_from_live) supplies EVERY feature
     the deployed signal_GENERIC model was trained on — no silent .get(f, 0)
     zero-fills. This test would have caught the serve-time skew where 8 of
     40 features were hardcoded to 0.
  2. The ml_pipeline V2 adapter (_build_v2_signal_features) covers the same
     contract from TickerScanState data.
  3. Model loading fails loudly when meta["features"] != booster.feature_name().
  4. Previously-zeroed features now compute real values that match the
     training definitions in scripts/train_option_signals_v2.py.
  5. ML-sourced signals are exempt from Discord-scale score gates/tiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from options_owl.sourcing.scoring.ml_gates.signal_model import (
    _feature_contract_ok,
    _missing_feature_warned,
    _warn_missing_features,
    compute_option_features_from_live,
)

# Canonical feature contract of the deployed signal_GENERIC model
# (journal/models/signal_ml_v2/signal_GENERIC_meta.json, trained 2026-05-28).
GENERIC_MODEL_FEATURES = [
    "minutes_since_open", "hour_bucket", "is_first_30min", "is_last_hour",
    "premium", "premium_change_5m", "premium_change_10m", "premium_change_15m",
    "premium_volatility",
    "current_volume", "volume_ratio", "volume_trend", "volume_zscore",
    "spread", "spread_pct", "spread_tightening", "bid_size", "ask_size",
    "size_imbalance",
    "iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend",
    "underlying_price", "underlying_change_5m", "underlying_change_15m",
    "underlying_volatility", "vwap_deviation",
    "daily_trend_pct", "daily_range_position", "atr_pct",
    "pre_move_underlying_5m",
    "sweep_high", "sweep_low", "near_key_level",
    "coiled_spring", "iv_expanding",
    "is_call",
]

GENERIC_META_PATH = (
    Path(__file__).resolve().parent.parent
    / "journal" / "models" / "signal_ml_v2" / "signal_GENERIC_meta.json"
)


def _full_live_features(**overrides):
    """Build a feature dict the way the scanner/ml_pipeline serve paths do."""
    kwargs = dict(
        ticker="SPY",
        premium=2.50,
        bid=2.45,
        ask=2.55,
        iv=0.30,
        delta=0.52,
        theta=-0.05,
        vega=0.11,
        volume=1200,
        underlying_price=550.0,
        minutes_since_open=45,
        is_call=True,
        premium_history=[2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45],
        volume_history=[500, 600, 700, 800, 900, 1000, 1100, 1150, 1180, 1190],
        underlying_history=[548.0, 548.5, 549.0, 549.2, 549.5, 549.8, 550.1,
                            550.0, 549.9, 550.0],
        bid_size=40.0,
        ask_size=25.0,
        spread_history=[0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09, 0.10],
        iv_history=[0.28, 0.282, 0.285, 0.287, 0.29, 0.292, 0.295, 0.30],
    )
    kwargs.update(overrides)
    return compute_option_features_from_live(**kwargs)


# ---------------------------------------------------------------------------
# 1. Live feature dict covers every model feature
# ---------------------------------------------------------------------------


class TestFeatureCoverage:
    def test_live_builder_covers_every_generic_model_feature(self):
        """Every feature the deployed model expects must be supplied live."""
        f = _full_live_features()
        missing = [name for name in GENERIC_MODEL_FEATURES if name not in f]
        assert missing == [], (
            f"Serve-time feature dict is missing model features: {missing}. "
            f"These would be silently zero-filled at predict — fix the builder."
        )

    def test_live_builder_covers_meta_file_features_if_present(self):
        """Same check against the actual deployed meta file (if available)."""
        if not GENERIC_META_PATH.exists():
            pytest.skip("signal_GENERIC_meta.json not present in this checkout")
        with open(GENERIC_META_PATH) as fh:
            meta = json.load(fh)
        f = _full_live_features()
        missing = [name for name in meta.get("features", []) if name not in f]
        assert missing == [], f"Live dict missing deployed-model features: {missing}"

    def test_live_builder_covers_contract_even_with_minimal_inputs(self):
        """Even without histories, the builder must emit every key (zeros are
        then the TRAINING convention for missing data, not silent skew)."""
        f = compute_option_features_from_live(
            ticker="SPY", premium=2.50, bid=2.45, ask=2.55,
            iv=0.30, delta=0.50, theta=-0.05, vega=0.10,
            volume=1000, underlying_price=550.0,
            minutes_since_open=60, is_call=True,
        )
        missing = [name for name in GENERIC_MODEL_FEATURES if name not in f]
        assert missing == []

    def test_pipeline_v2_adapter_covers_every_generic_model_feature(self):
        """_build_v2_signal_features (TickerScanState path) must also cover
        the full contract."""
        from options_owl.sourcing.ml_pipeline import (
            TickerScanState,
            _build_v2_signal_features,
        )

        state = TickerScanState(expiry="2026-06-10")
        for minute in range(20):
            state.append_snapshot(
                {
                    "mid": 2.0 + minute * 0.03,
                    "bid": 1.95 + minute * 0.03,
                    "ask": 2.05 + minute * 0.03,
                    "iv": 0.28 + minute * 0.001,
                    "delta": 0.5,
                    "theta": -0.05,
                    "vega": 0.1,
                    "volume": 500 + minute * 40,
                    "underlying_price": 549.0 + minute * 0.05,
                    "bid_size": 30 + minute,
                    "ask_size": 20 + minute,
                },
                minute,
            )

        feat = _build_v2_signal_features(
            state, len(state.closes) - 1, GENERIC_MODEL_FEATURES, "CALL"
        )
        assert feat is not None
        assert set(feat.keys()) == set(GENERIC_MODEL_FEATURES)
        # Previously-zeroed features must now carry real values
        assert feat["bid_size"] > 0
        assert feat["ask_size"] > 0
        assert feat["size_imbalance"] != 0
        assert feat["iv_change_15m"] != 0
        assert feat["iv_trend"] != 0
        assert feat["vwap_deviation"] != 0
        # volume_trend is a RATIO (training definition) — rising volumes > 1,
        # never the old difference-of-means (which would be in the hundreds)
        assert 0.5 < feat["volume_trend"] < 5.0


# ---------------------------------------------------------------------------
# 2. Previously-zeroed features compute training-exact values
# ---------------------------------------------------------------------------


class TestRealFeatureValues:
    def test_spread_tightening_matches_training_definition(self):
        # training: mean(first half) - mean(second half), positive = tightening
        spreads = [0.20, 0.20, 0.18, 0.18, 0.10, 0.10, 0.08, 0.08]
        f = _full_live_features(spread_history=spreads)
        expected = float(np.mean(spreads[:4]) - np.mean(spreads[4:]))
        assert f["spread_tightening"] == pytest.approx(expected)
        assert f["spread_tightening"] > 0

    def test_spread_tightening_zero_without_history(self):
        f = _full_live_features(spread_history=None)
        assert f["spread_tightening"] == 0

    def test_quote_sizes_and_imbalance(self):
        f = _full_live_features(bid_size=60.0, ask_size=20.0)
        assert f["bid_size"] == 60.0
        assert f["ask_size"] == 20.0
        assert f["size_imbalance"] == pytest.approx((60 - 20) / 80)

    def test_iv_changes_match_training_definition(self):
        ivs = [0.28, 0.282, 0.285, 0.287, 0.29, 0.292, 0.295, 0.30]
        f = _full_live_features(iv_history=ivs)
        # iv_change_5m = ivs[-1] - ivs[-6]; iv_change_15m = ivs[-1] - ivs[0]
        assert f["iv_change_5m"] == pytest.approx(0.30 - 0.285)  # ivs[-6] = 0.285
        assert f["iv_change_15m"] == pytest.approx(0.30 - 0.28)
        slope = np.polyfit(range(len(ivs)), ivs, 1)[0]
        assert f["iv_trend"] == pytest.approx(float(slope))

    def test_iv_changes_zero_with_short_history(self):
        # training guard: needs > 3 valid IV observations
        f = _full_live_features(iv_history=[0.30, 0.31])
        assert f["iv_change_5m"] == 0
        assert f["iv_change_15m"] == 0
        assert f["iv_trend"] == 0

    def test_iv_expanding_derives_from_real_iv_change(self):
        rising = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.30]  # 5m change 0.06 > 0.02
        f = _full_live_features(iv_history=rising)
        assert f["iv_expanding"] == 1
        flat = [0.30] * 8
        f2 = _full_live_features(iv_history=flat)
        assert f2["iv_expanding"] == 0

    def test_vwap_deviation_uses_trailing_window_mean_like_training(self):
        hist = [548.0, 549.0, 550.0, 551.0, 552.0]
        f = _full_live_features(underlying_history=hist, underlying_price=552.0)
        expected = (552.0 / np.mean(hist) - 1) * 100
        assert f["vwap_deviation"] == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# 3. Feature-contract validation at load
# ---------------------------------------------------------------------------


class TestFeatureContractValidation:
    def _mock_model(self, names):
        m = MagicMock()
        m.feature_name.return_value = list(names)
        return m

    def test_contract_ok_when_matching(self):
        model = self._mock_model(["a", "b", "c"])
        assert _feature_contract_ok("test", model, {"features": ["a", "b", "c"]})

    def test_contract_fails_on_mismatch(self):
        model = self._mock_model(["a", "b", "c"])
        assert not _feature_contract_ok("test", model, {"features": ["a", "b"]})

    def test_contract_fails_on_wrong_order(self):
        # LightGBM consumes positional arrays — order IS the contract
        model = self._mock_model(["a", "b"])
        assert not _feature_contract_ok("test", model, {"features": ["b", "a"]})

    def test_contract_skipped_without_meta_features(self):
        model = self._mock_model(["a", "b"])
        assert _feature_contract_ok("test", model, {})

    def test_pipeline_check_raises_for_required_model(self):
        from options_owl.sourcing.ml_pipeline import _check_feature_contract

        model = self._mock_model(["a", "b"])
        with pytest.raises(ValueError, match="FEATURE CONTRACT MISMATCH"):
            _check_feature_contract("pattern_entry", model, ["a", "x"], required=True)

    def test_pipeline_check_disables_optional_model(self):
        from options_owl.sourcing.ml_pipeline import _check_feature_contract

        model = self._mock_model(["a", "b"])
        assert _check_feature_contract("optional", model, ["a", "x"]) is False
        assert _check_feature_contract("optional", model, ["a", "b"]) is True

    def test_missing_feature_warning_fires_once_per_model(self):
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            _missing_feature_warned.discard("warn_test_model")
            _warn_missing_features("warn_test_model", ["a", "b"], {"a": 1})
            _warn_missing_features("warn_test_model", ["a", "b"], {"a": 1})
        finally:
            logger.remove(sink_id)

        hits = [m for m in messages if "warn_test_model" in m]
        assert len(hits) == 1  # once per model, not per predict
        assert "'b'" in hits[0]

    def test_no_warning_when_all_features_supplied(self):
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            _missing_feature_warned.discard("warn_test_model_2")
            _warn_missing_features("warn_test_model_2", ["a"], {"a": 1})
        finally:
            logger.remove(sink_id)

        assert not any("warn_test_model_2" in m for m in messages)


# ---------------------------------------------------------------------------
# 4. Pattern threshold comes from the model meta (FIX 2)
# ---------------------------------------------------------------------------


class TestPatternThreshold:
    def test_threshold_from_model_meta_by_default(self):
        from options_owl.sourcing.ml_pipeline import (
            MLModels,
            MLPipelineSettings,
            resolve_pattern_threshold,
        )

        settings = MLPipelineSettings()  # ML_PATTERN_THRESHOLD defaults to 0.0
        models = MLModels(pattern_meta={"best_threshold": 0.80})
        assert resolve_pattern_threshold(settings, models) == 0.80

    def test_env_override_wins(self):
        from options_owl.sourcing.ml_pipeline import (
            MLModels,
            MLPipelineSettings,
            resolve_pattern_threshold,
        )

        settings = MLPipelineSettings(ML_PATTERN_THRESHOLD=0.85)
        models = MLModels(pattern_meta={"best_threshold": 0.80})
        assert resolve_pattern_threshold(settings, models) == 0.85

    def test_fallback_to_default_without_meta(self):
        from options_owl.sourcing.ml_pipeline import (
            DEFAULT_PATTERN_THRESHOLD,
            MLModels,
            MLPipelineSettings,
            resolve_pattern_threshold,
        )

        settings = MLPipelineSettings()
        models = MLModels(pattern_meta={})
        assert resolve_pattern_threshold(settings, models) == DEFAULT_PATTERN_THRESHOLD


# ---------------------------------------------------------------------------
# 5. ML-sourced signals vs Discord-scale score gates (FIX 2)
# ---------------------------------------------------------------------------


def _signal(score, source_value):
    return SimpleNamespace(
        ticker="NVDA",
        score=score,
        atm_premium=2.50,
        entry_price=550.0,
        bot_source=SimpleNamespace(value=source_value),
    )


class TestScoreGateMLExemption:
    @pytest.mark.asyncio
    async def test_ml_signal_in_old_dead_band_now_passes(self):
        """conf 0.74 → score 74 was emitted but ALWAYS rejected by MIN_SCORE=78."""
        from options_owl.risk.pipeline import GateResult, ScoreGate

        settings = SimpleNamespace(MIN_SCORE=78, ML_MIN_SCORE=60)
        ctx = {"signal": _signal(74, "ml_sourcing"), "settings": settings}
        outcome = await ScoreGate().evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_ml_signal_below_ml_floor_fails(self):
        from options_owl.risk.pipeline import GateResult, ScoreGate

        settings = SimpleNamespace(MIN_SCORE=78, ML_MIN_SCORE=60)
        ctx = {"signal": _signal(50, "ml_sourcing"), "settings": settings}
        outcome = await ScoreGate().evaluate(ctx)
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_discord_signal_still_uses_min_score(self):
        from options_owl.risk.pipeline import GateResult, ScoreGate

        settings = SimpleNamespace(MIN_SCORE=78, ML_MIN_SCORE=60)
        ctx = {"signal": _signal(74, "Captain Hook"), "settings": settings}
        outcome = await ScoreGate().evaluate(ctx)
        assert outcome.result == GateResult.FAIL


class TestPremiumCapMLExemption:
    @pytest.mark.asyncio
    async def test_ml_signal_always_uses_base_cap(self):
        """ML scores are capped at 100 — the 120/150 tiers must never apply,
        even if a future change inflated ML scores."""
        from options_owl.risk.pipeline import GateResult, PremiumCapGate

        settings = SimpleNamespace(
            ENABLE_V6_PREMIUM_CAP=True,
            V6_PREMIUM_CAP=6.0, V6_PREMIUM_CAP_MID=7.0, V6_PREMIUM_CAP_HIGH=9.0,
        )
        sig = _signal(150, "ml_sourcing")
        sig.atm_premium = 6.50  # above base cap, below high cap
        outcome = await PremiumCapGate().evaluate({"signal": sig, "settings": settings})
        assert outcome.result == GateResult.FAIL  # base cap applied

        discord_sig = _signal(150, "Captain Hook")
        discord_sig.atm_premium = 6.50
        outcome2 = await PremiumCapGate().evaluate(
            {"signal": discord_sig, "settings": settings}
        )
        assert outcome2.result == GateResult.PASS  # high tier (score 150)


class TestAntiChaseMLExemption:
    @pytest.mark.asyncio
    async def test_ml_signal_uses_base_move(self):
        from options_owl.risk.pipeline import AntiChaseGate, GateResult

        settings = SimpleNamespace(
            ENABLE_VINNY_STRATEGY=True, ANTI_CHASE_MAX_MOVE_PCT=0.3,
        )
        # Underlying moved +0.6% from alert: within the 0.75% elite tier but
        # beyond the 0.3% base move.
        sig = _signal(150, "ml_sourcing")
        ctx = {"signal": sig, "settings": settings, "current_price": 553.3}
        outcome = await AntiChaseGate().evaluate(ctx)
        assert outcome.result == GateResult.FAIL  # base move for ML

        discord_sig = _signal(150, "Captain Hook")
        ctx2 = {"signal": discord_sig, "settings": settings, "current_price": 553.3}
        outcome2 = await AntiChaseGate().evaluate(ctx2)
        assert outcome2.result == GateResult.PASS  # 0.75% tier for score 150
