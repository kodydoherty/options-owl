"""ML v2 — Entry filter, peak prediction, and regime classification.

Three models trained on 5 years of 0DTE data (81K+ samples):

1. Entry Filter: Should we take this trade? (binary classifier, AUC=0.80)
2. Peak Predictor: What MFE will this trade achieve? (regressor, R²=0.40)
3. Regime Classifier: Is today trending or choppy? (classifier, AUC=0.71)

Usage in the pipeline:
    from options_owl.risk.ml_v2 import predict_entry, predict_peak, predict_regime

    # At entry time — reject low-confidence trades
    entry = predict_entry(ticker, entry_premium, underlying_price, hour, minute, ...)
    if not entry.should_enter:
        reject("ML entry filter: low confidence")

    # During trade — set dynamic targets based on predicted MFE
    peak = predict_peak(ticker, entry_premium, underlying_price, hour, minute, ...)
    if peak.predicted_mfe > 100:
        use wider trails  # big winner predicted

    # At market open — classify today's regime
    regime = predict_regime(ticker, first_30min_bars)
    if regime.is_choppy:
        use tighter trails
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger

MODELS_DIR = os.environ.get("ML_MODELS_DIR", os.path.join(os.getcwd(), "journal", "models"))

# Lazy-loaded model cache
_entry_model = None
_entry_meta = None
_entry_afternoon_model = None
_entry_afternoon_meta = None
_peak_model = None
_peak_timing_model = None
_peak_meta = None
_regime_model = None
_regime_dr_model = None
_regime_meta = None


@dataclass
class EntrySignal:
    """Result from the entry filter model."""
    should_enter: bool
    confidence: float  # probability of being a winner
    reason: str
    model_used: str


@dataclass
class PeakSignal:
    """Result from the peak predictor model."""
    predicted_mfe_pct: float  # expected max favorable excursion
    predicted_peak_minutes: float  # expected minutes to peak
    suggested_trail_width: float  # dynamic trail width based on prediction
    reason: str


@dataclass
class RegimeSignal:
    """Result from the regime classifier."""
    is_trending: bool
    trending_probability: float
    directional_ratio: float  # 0=choppy, 1=perfectly directional
    suggested_trail_multiplier: float  # multiply trail widths by this
    reason: str


def _load_entry_model(afternoon: bool = False):
    global _entry_model, _entry_meta, _entry_afternoon_model, _entry_afternoon_meta

    if afternoon:
        if _entry_afternoon_model is not None:
            return _entry_afternoon_model, _entry_afternoon_meta
    else:
        if _entry_model is not None:
            return _entry_model, _entry_meta

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — ML v2 entry filter disabled")
        return None, None

    suffix = "_afternoon" if afternoon else ""
    model_path = os.path.join(MODELS_DIR, f"entry_filter_v2{suffix}.lgb")
    meta_path = os.path.join(MODELS_DIR, f"entry_filter_v2{suffix}_meta.json")

    if not os.path.exists(model_path):
        logger.warning(f"Entry filter model not found at {model_path}")
        return None, None

    loaded_model = lgb.Booster(model_file=model_path)
    loaded_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            loaded_meta = json.load(f)

    if afternoon:
        _entry_afternoon_model = loaded_model
        _entry_afternoon_meta = loaded_meta
    else:
        _entry_model = loaded_model
        _entry_meta = loaded_meta

    logger.info(f"Loaded ML v2 entry filter: {model_path}")
    return loaded_model, loaded_meta


def _load_peak_model():
    global _peak_model, _peak_timing_model, _peak_meta
    if _peak_model is not None:
        return _peak_model, _peak_timing_model, _peak_meta

    try:
        import lightgbm as lgb
    except ImportError:
        return None, None, None

    model_path = os.path.join(MODELS_DIR, "peak_predictor_v2.lgb")
    timing_path = os.path.join(MODELS_DIR, "peak_timing_v2.lgb")
    meta_path = os.path.join(MODELS_DIR, "peak_predictor_v2_meta.json")

    if not os.path.exists(model_path):
        logger.warning(f"Peak predictor model not found at {model_path}")
        return None, None, None

    _peak_model = lgb.Booster(model_file=model_path)
    if os.path.exists(timing_path):
        _peak_timing_model = lgb.Booster(model_file=timing_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _peak_meta = json.load(f)
    else:
        _peak_meta = {}

    logger.info(f"Loaded ML v2 peak predictor: {model_path}")
    return _peak_model, _peak_timing_model, _peak_meta


def _load_regime_model():
    global _regime_model, _regime_dr_model, _regime_meta
    if _regime_model is not None:
        return _regime_model, _regime_dr_model, _regime_meta

    try:
        import lightgbm as lgb
    except ImportError:
        return None, None, None

    model_path = os.path.join(MODELS_DIR, "regime_classifier_v2.lgb")
    dr_path = os.path.join(MODELS_DIR, "regime_direction_v2.lgb")
    meta_path = os.path.join(MODELS_DIR, "regime_classifier_v2_meta.json")

    if not os.path.exists(model_path):
        logger.warning(f"Regime classifier not found at {model_path}")
        return None, None, None

    _regime_model = lgb.Booster(model_file=model_path)
    if os.path.exists(dr_path):
        _regime_dr_model = lgb.Booster(model_file=dr_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _regime_meta = json.load(f)
    else:
        _regime_meta = {}

    logger.info(f"Loaded ML v2 regime classifier: {model_path}")
    return _regime_model, _regime_dr_model, _regime_meta


def _encode_ticker(ticker: str, meta: dict) -> int:
    """Encode ticker using the same mapping as training."""
    classes = meta.get("ticker_classes", [])
    ticker_upper = ticker.upper()
    if ticker_upper in classes:
        return classes.index(ticker_upper)
    return 0  # unknown ticker gets index 0


def predict_entry(
    ticker: str,
    entry_premium: float,
    underlying_price: float,
    hour: int,
    minute: int,
    day_of_week: int,
    underlying_momentum_5m: float = 0.0,
    underlying_momentum_10m: float = 0.0,
    underlying_momentum_15m: float = 0.0,
    underlying_momentum_30m: float = 0.0,
    underlying_volatility: float = 0.0,
    volume_avg: float = 0.0,
    volume_vs_avg: float = 1.0,
    volume_trend: float = 1.0,
    price_position_in_range: float = 0.5,
    avg_bar_range_pct: float = 0.0,
    vwap_deviation_pct: float = 0.0,
    consec_up_bars: int = 0,
    consec_down_bars: int = 0,
    day_open_to_now_pct: float = 0.0,
    premium_momentum_5m: float = 0.0,
    premium_momentum_10m: float = 0.0,
    option_bar_range_pct: float = 0.0,
    option_volume: float = 0.0,
    option_vol_vs_avg: float = 1.0,
    option_num_trades: float = 0.0,
    is_call: bool = True,
    threshold: float = 0.50,
    afternoon: bool = False,
) -> EntrySignal:
    """Run the entry filter model. Returns whether to take the trade.

    If afternoon=True, uses the afternoon-calibrated model (trained on 1PM+ entries only).
    """
    model, meta = _load_entry_model(afternoon=afternoon)
    variant = "afternoon" if afternoon else "allday"
    if model is None:
        return EntrySignal(
            should_enter=True, confidence=0.5,
            reason="Entry filter not available", model_used="none",
        )

    minutes_since_open = (hour - 9) * 60 + (minute - 30)
    premium_to_underlying = (entry_premium / underlying_price * 100) if underlying_price > 0 else 0

    features = np.array([[
        hour, minute, day_of_week, minutes_since_open, 1 if is_call else 0,
        underlying_momentum_5m, underlying_momentum_10m,
        underlying_momentum_15m, underlying_momentum_30m,
        underlying_volatility,
        volume_avg, volume_vs_avg, volume_trend,
        price_position_in_range, avg_bar_range_pct, vwap_deviation_pct,
        consec_up_bars, consec_down_bars, day_open_to_now_pct,
        entry_premium, premium_to_underlying,
        premium_momentum_5m, premium_momentum_10m,
        option_bar_range_pct, option_volume, option_vol_vs_avg, option_num_trades,
        _encode_ticker(ticker, meta),
    ]])

    prob = float(model.predict(features)[0])
    should_enter = prob >= threshold

    if should_enter:
        reason = f"ML entry OK: {prob:.0%} confidence"
    else:
        reason = f"ML entry REJECT: {prob:.0%} confidence < {threshold:.0%} threshold"

    return EntrySignal(
        should_enter=should_enter,
        confidence=prob,
        reason=reason,
        model_used=f"entry_filter_v2_{variant}",
    )


def predict_peak(
    ticker: str,
    entry_premium: float,
    underlying_price: float,
    hour: int,
    minute: int,
    day_of_week: int,
    is_call: bool = True,
    underlying_momentum_5m: float = 0.0,
    underlying_momentum_10m: float = 0.0,
    underlying_momentum_15m: float = 0.0,
    underlying_momentum_30m: float = 0.0,
    underlying_volatility: float = 0.0,
    volume_avg: float = 0.0,
    volume_vs_avg: float = 1.0,
    volume_trend: float = 1.0,
    price_position_in_range: float = 0.5,
    avg_bar_range_pct: float = 0.0,
    vwap_deviation_pct: float = 0.0,
    consec_up_bars: int = 0,
    consec_down_bars: int = 0,
    day_open_to_now_pct: float = 0.0,
    premium_momentum_5m: float = 0.0,
    premium_momentum_10m: float = 0.0,
    option_bar_range_pct: float = 0.0,
    option_volume: float = 0.0,
    option_vol_vs_avg: float = 1.0,
    option_num_trades: float = 0.0,
) -> PeakSignal:
    """Predict the MFE and optimal hold time for a trade."""
    model, timing_model, meta = _load_peak_model()
    if model is None:
        return PeakSignal(
            predicted_mfe_pct=50.0, predicted_peak_minutes=60.0,
            suggested_trail_width=35.0,
            reason="Peak predictor not available",
        )

    minutes_since_open = (hour - 9) * 60 + (minute - 30)
    premium_to_underlying = (entry_premium / underlying_price * 100) if underlying_price > 0 else 0

    features = np.array([[
        hour, minute, day_of_week, minutes_since_open, 1 if is_call else 0,
        underlying_momentum_5m, underlying_momentum_10m,
        underlying_momentum_15m, underlying_momentum_30m,
        underlying_volatility,
        volume_avg, volume_vs_avg, volume_trend,
        price_position_in_range, avg_bar_range_pct, vwap_deviation_pct,
        consec_up_bars, consec_down_bars, day_open_to_now_pct,
        entry_premium, premium_to_underlying,
        premium_momentum_5m, premium_momentum_10m,
        option_bar_range_pct, option_volume, option_vol_vs_avg, option_num_trades,
        _encode_ticker(ticker, meta),
    ]])

    mfe_pred = float(model.predict(features)[0])
    peak_minutes = 60.0  # default
    if timing_model is not None:
        peak_minutes = float(timing_model.predict(features)[0])

    # Dynamic trail width based on predicted MFE
    if mfe_pred >= 100:
        trail_width = 50.0  # big winner predicted — very wide trail
    elif mfe_pred >= 50:
        trail_width = 40.0  # moderate winner — wide trail
    elif mfe_pred >= 20:
        trail_width = 35.0  # normal — standard trail
    else:
        trail_width = 25.0  # low MFE predicted — tight trail, take profits fast

    reason = f"Predicted MFE={mfe_pred:+.0f}%, peak in ~{peak_minutes:.0f}m → trail={trail_width:.0f}%"

    return PeakSignal(
        predicted_mfe_pct=mfe_pred,
        predicted_peak_minutes=peak_minutes,
        suggested_trail_width=trail_width,
        reason=reason,
    )


def predict_regime(
    ticker: str,
    day_of_week: int,
    first30_move_pct: float = 0.0,
    first30_abs_move_pct: float = 0.0,
    first30_vol: float = 0.0,
    first30_range_pct: float = 0.0,
    first30_avg_volume: float = 0.0,
    first30_vol_ratio: float = 1.0,
    first30_direction_changes: int = 0,
    first30_max_run: int = 1,
    first30_vwap_trend: float = 0.0,
) -> RegimeSignal:
    """Classify today's market regime based on first 30 min of trading."""
    model, dr_model, meta = _load_regime_model()
    if model is None:
        return RegimeSignal(
            is_trending=True, trending_probability=0.5,
            directional_ratio=0.5, suggested_trail_multiplier=1.0,
            reason="Regime classifier not available",
        )

    features = np.array([[
        day_of_week,
        first30_move_pct, first30_abs_move_pct, first30_vol,
        first30_range_pct, first30_avg_volume, first30_vol_ratio,
        first30_direction_changes, first30_max_run, first30_vwap_trend,
        _encode_ticker(ticker, meta),
    ]])

    trend_prob = float(model.predict(features)[0])
    is_trending = trend_prob >= 0.5

    directional_ratio = 0.5
    if dr_model is not None:
        directional_ratio = float(dr_model.predict(features)[0])
        directional_ratio = max(0.0, min(1.0, directional_ratio))

    # Regime-based trail multiplier
    if is_trending and trend_prob >= 0.7:
        multiplier = 1.3  # strong trending day — 30% wider trails
    elif is_trending:
        multiplier = 1.1  # mild trending — 10% wider
    elif trend_prob < 0.3:
        multiplier = 0.7  # strong choppy — 30% tighter trails
    else:
        multiplier = 0.9  # mild choppy — 10% tighter

    label = "TRENDING" if is_trending else "CHOPPY"
    reason = f"Regime: {label} ({trend_prob:.0%}), directional={directional_ratio:.2f} → trail×{multiplier:.1f}"

    return RegimeSignal(
        is_trending=is_trending,
        trending_probability=trend_prob,
        directional_ratio=directional_ratio,
        suggested_trail_multiplier=multiplier,
        reason=reason,
    )
