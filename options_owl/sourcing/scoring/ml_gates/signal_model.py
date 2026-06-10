"""ML Signal Model Gate — uses trained LightGBM entry models from V2 pipeline.

Loads per-ticker (or generic fallback) models trained by train_option_signals_v2.py.
Computes option-price features from live Polygon data and predicts entry confidence.

This is the bridge between the offline ML training pipeline and live signal generation.
Models are in journal/models/signal_ml_v2/signal_{TICKER}.lgb.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

MODELS_DIR = Path(os.environ.get(
    "ML_SIGNAL_MODELS_DIR",
    os.path.join(os.getcwd(), "journal", "models", "signal_ml_v2"),
))

# Lazy-loaded model cache: ticker -> (booster, meta_dict)
_model_cache: dict[str, tuple] = {}

# Once-per-model warning registry for missing serve-time features
_missing_feature_warned: set[str] = set()


def _feature_contract_ok(name: str, model, meta: dict) -> bool:
    """Validate that meta['features'] exactly matches the booster's features.

    A mismatch means serve-time feature vectors would be built in the wrong
    order / with wrong columns — silently corrupting every prediction.
    Fail loudly and refuse to use the model.
    """
    meta_features = meta.get("features")
    if not meta_features:
        # No meta features — predict path will bail out (returns model_source=none)
        return True
    try:
        booster_features = list(model.feature_name())
    except Exception:
        return True  # cannot introspect (e.g. mocks) — skip validation
    if list(meta_features) != booster_features:
        logger.error(
            f"ML_SIGNAL: FEATURE CONTRACT MISMATCH for {name} — "
            f"meta has {len(meta_features)} features, booster has "
            f"{len(booster_features)}. meta_only={set(meta_features) - set(booster_features)} "
            f"booster_only={set(booster_features) - set(meta_features)}. REFUSING to load."
        )
        return False
    return True


def _warn_missing_features(model_name: str, feature_names: list, features: dict) -> None:
    """Log a once-per-model WARNING for any meta feature the live dict didn't supply.

    Silent .get(f, 0) defaults hide serve-time feature skew — this makes it loud.
    """
    if model_name in _missing_feature_warned:
        return
    missing = [f for f in feature_names if f not in features]
    if missing:
        logger.warning(
            f"ML_SIGNAL: {model_name} — live feature dict is missing "
            f"{len(missing)}/{len(feature_names)} model features (defaulting to 0): "
            f"{missing}"
        )
    _missing_feature_warned.add(model_name)


def _load_model(ticker: str, direction: str = ""):
    """Lazy-load a signal model with fallback chain.

    Fallback order (combined models only — direction-specific were worse):
      1. signal_{TICKER} (combined per-ticker with is_call feature)
      2. signal_GENERIC (generic fallback)

    Args:
        direction: Accepted for API compatibility but ignored for model selection.
    """
    cache_key = ticker  # Always use combined model
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("ML_SIGNAL: lightgbm not installed — gate disabled")
        _model_cache[cache_key] = (None, None)
        return None, None

    # Combined models only — direction-specific halved training data and lost AUC
    candidates = []
    candidates.append(f"signal_{ticker}")
    candidates.append("signal_GENERIC")

    for name in candidates:
        model_path = MODELS_DIR / f"{name}.lgb"
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if model_path.exists():
            try:
                model = lgb.Booster(model_file=str(model_path))
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                if not _feature_contract_ok(name, model, meta):
                    continue  # try next candidate — never serve a skewed model
                _model_cache[cache_key] = (model, meta)
                logger.info(f"ML_SIGNAL: loaded {name} (threshold={meta.get('optimal_threshold', 0.5):.2f})")
                return model, meta
            except Exception as e:
                logger.warning(f"ML_SIGNAL: failed to load {name}: {e}")

    logger.debug(f"ML_SIGNAL: no model for {cache_key}")
    _model_cache[cache_key] = (None, None)
    return None, None


def _load_runner_model(ticker: str, direction: str = ""):
    """Lazy-load runner classifier with fallback chain.

    Fallback: runner_{TICKER} → runner_GENERIC (combined models only).
    """
    cache_key = f"runner_{ticker}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import lightgbm as lgb
    except ImportError:
        _model_cache[cache_key] = (None, None)
        return None, None

    candidates = []
    candidates.append(f"runner_{ticker}")
    candidates.append("runner_GENERIC")

    for name in candidates:
        model_path = MODELS_DIR / f"{name}.lgb"
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if model_path.exists():
            try:
                model = lgb.Booster(model_file=str(model_path))
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                if not _feature_contract_ok(name, model, meta):
                    continue
                _model_cache[cache_key] = (model, meta)
                return model, meta
            except Exception:
                pass

    _model_cache[cache_key] = (None, None)
    return None, None


def predict_entry_confidence(ticker: str, features: dict, direction: str = "") -> dict:
    """Predict entry confidence using the trained V2 signal model.

    Args:
        ticker: e.g. "SPY"
        features: dict of option-price features (same keys as FEATURE_COLS in training script)
        direction: "CALL", "PUT", or "" — uses direction-specific model if available

    Returns:
        {
            "confidence": float (0-1),
            "threshold": float (model's optimal threshold),
            "is_signal": bool (confidence >= threshold),
            "runner_score": float (0-1, probability of being a runner),
            "model_source": str ("per_ticker_CALL", "per_ticker_PUT", "per_ticker", "generic", "none"),
        }
    """
    dir_key = direction.upper() if direction else ""
    model, meta = _load_model(ticker, dir_key)
    if model is None:
        return {
            "confidence": 0.0,
            "threshold": 1.0,
            "is_signal": False,
            "runner_score": 0.0,
            "model_source": "none",
        }

    feature_names = meta.get("features", [])
    if not feature_names:
        return {
            "confidence": 0.0,
            "threshold": 1.0,
            "is_signal": False,
            "runner_score": 0.0,
            "model_source": "none",
        }

    import numpy as np

    _warn_missing_features(f"signal_{meta.get('ticker', ticker)}", feature_names, features)
    X = np.array([[features.get(f, 0) for f in feature_names]])
    confidence = float(model.predict(X)[0])
    threshold = meta.get("optimal_threshold", 0.5)

    # Runner model (same direction fallback chain)
    runner_score = 0.0
    runner_model, runner_meta = _load_runner_model(ticker, dir_key)
    if runner_model is not None:
        runner_features = runner_meta.get("features", feature_names)
        _warn_missing_features(f"runner_{runner_meta.get('ticker', ticker)}", runner_features, features)
        X_runner = np.array([[features.get(f, 0) for f in runner_features]])
        runner_score = float(runner_model.predict(X_runner)[0])

    model_name = meta.get("ticker", "GENERIC")
    model_dir = meta.get("direction", "BOTH")
    if model_name == ticker and model_dir != "BOTH":
        source = f"per_ticker_{model_dir}"
    elif model_name == ticker:
        source = "per_ticker"
    else:
        source = "generic"

    return {
        "confidence": confidence,
        "threshold": threshold,
        "is_signal": confidence >= threshold,
        "runner_score": runner_score,
        "model_source": source,
    }


def compute_option_features_from_live(
    ticker: str,
    premium: float,
    bid: float,
    ask: float,
    iv: float,
    delta: float,
    theta: float,
    vega: float,
    volume: int,
    underlying_price: float,
    minutes_since_open: int,
    is_call: bool,
    premium_history: list[float] | None = None,
    volume_history: list[int] | None = None,
    underlying_history: list[float] | None = None,
    bid_size: float = 0.0,
    ask_size: float = 0.0,
    spread_history: list[float] | None = None,
    iv_history: list[float] | None = None,
) -> dict:
    """Build feature dict from live market data for model prediction.

    This is the SINGLE source of truth for serve-time V2 signal features.
    Both the sourcing scanner (scanner._run_ml_gate) and the ML pipeline
    (ml_pipeline._build_v2_signal_features) MUST build features through this
    function so serving matches training (train_option_signals_v2.py
    compute_setup_features) exactly.

    History conventions (matching training windows):
      - premium_history / volume_history: trailing values OLDEST→NEWEST,
        EXCLUDING the current snapshot (current values are passed as
        `premium` / `volume`).
      - spread_history / iv_history / underlying_history: trailing values
        oldest→newest INCLUDING the current snapshot as the last element
        (training computes these from the full lookback window whose last
        row is the decision candle).
    """
    import numpy as np

    f: dict = {}

    # Time
    f["minutes_since_open"] = minutes_since_open
    f["hour_bucket"] = minutes_since_open // 60
    f["is_first_30min"] = 1 if minutes_since_open <= 30 else 0
    f["is_last_hour"] = 1 if minutes_since_open >= 330 else 0

    # Premium action
    f["premium"] = premium
    prices = premium_history or []
    if len(prices) >= 5 and prices[-5] > 0:
        f["premium_change_5m"] = (premium / prices[-5] - 1) * 100
    else:
        f["premium_change_5m"] = 0
    if len(prices) >= 10 and prices[-10] > 0:
        f["premium_change_10m"] = (premium / prices[-10] - 1) * 100
    else:
        f["premium_change_10m"] = 0
    if len(prices) >= 15 and prices[0] > 0:
        f["premium_change_15m"] = (premium / prices[0] - 1) * 100
    else:
        f["premium_change_15m"] = 0

    # Premium volatility
    if len(prices) >= 3 and all(p > 0 for p in prices[:-1]):
        arr = np.array(prices)
        returns = np.diff(arr) / arr[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
        f["premium_skew"] = float(np.mean(returns > 0) - 0.5) * 2  # simplified skew
    else:
        f["premium_volatility"] = 0
        f["premium_skew"] = 0

    # Range position
    if prices:
        lo, hi = min(prices), max(prices)
        f["range_position"] = (premium - lo) / (hi - lo) if hi > lo else 0.5
    else:
        f["range_position"] = 0.5
    f["near_low"] = 1 if f["range_position"] < 0.25 else 0
    f["near_high"] = 1 if f["range_position"] > 0.85 else 0

    # Consecutive bars
    if len(prices) >= 2:
        diffs = np.diff(prices)
        consecutive_up = 0
        for d in reversed(diffs):
            if d > 0:
                consecutive_up += 1
            else:
                break
        consecutive_down = 0
        for d in reversed(diffs):
            if d < 0:
                consecutive_down += 1
            else:
                break
        f["consecutive_up_bars"] = consecutive_up
        f["consecutive_down_bars"] = consecutive_down
    else:
        f["consecutive_up_bars"] = 0
        f["consecutive_down_bars"] = 0

    # Volume
    f["current_volume"] = volume
    vols = volume_history or []
    avg_vol = np.mean(vols) if vols else 1
    f["volume_ratio"] = volume / max(avg_vol, 1)
    if len(vols) > 5:
        f["volume_trend"] = np.mean(vols[-5:]) / max(np.mean(vols[:-5]), 1)
    else:
        f["volume_trend"] = 1.0
    if len(vols) > 5 and np.std(vols[:-1]) > 0:
        f["volume_zscore"] = (volume - np.mean(vols[:-1])) / np.std(vols[:-1])
    else:
        f["volume_zscore"] = 0

    # Bid/ask
    mid = (bid + ask) / 2 if (bid + ask) > 0 else premium
    f["spread"] = max(0, ask - bid)
    f["spread_pct"] = f["spread"] / mid * 100 if mid > 0 else 0

    # Spread tightening — training: mean(first half) - mean(second half) of the
    # window's spreads (positive = tightening). Requires > 3 spread observations.
    sp_hist = [s for s in (spread_history or []) if s is not None and s >= 0]
    if len(sp_hist) > 3:
        sp_arr = np.array(sp_hist, dtype=np.float64)
        first_half = float(np.mean(sp_arr[: len(sp_arr) // 2]))
        second_half = float(np.mean(sp_arr[len(sp_arr) // 2 :]))
        f["spread_tightening"] = first_half - second_half
    else:
        f["spread_tightening"] = 0

    # Quote sizes — real values from the live quote (training: bid_size/ask_size
    # from option_quotes; size_imbalance = (bid-ask)/max(bid+ask, 1))
    f["bid_size"] = float(bid_size or 0)
    f["ask_size"] = float(ask_size or 0)
    f["size_imbalance"] = (f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1)

    # Greeks
    f["iv"] = iv if iv is not None else 0
    f["delta"] = abs(delta) if delta is not None else 0
    f["theta"] = theta if theta is not None else 0
    f["vega"] = vega if vega is not None else 0

    # IV dynamics — training (with ivs = window IVs incl. decision candle, NaNs dropped):
    #   iv_change_5m  = ivs[-1] - ivs[-6]  (if > 5 obs, else 0)
    #   iv_change_15m = ivs[-1] - ivs[0]
    #   iv_trend      = linear slope over the window (if > 2 obs)
    # Requires > 3 valid IV observations (training guard), else all zero.
    ivs = [v for v in (iv_history or []) if v is not None and v > 0]
    if len(ivs) > 3:
        f["iv_change_5m"] = float(ivs[-1] - ivs[-6]) if len(ivs) > 5 else 0
        f["iv_change_15m"] = float(ivs[-1] - ivs[0])
        f["iv_trend"] = float(np.polyfit(range(len(ivs)), ivs, 1)[0]) if len(ivs) > 2 else 0
    else:
        f["iv_change_5m"] = 0
        f["iv_change_15m"] = 0
        f["iv_trend"] = 0

    # Underlying
    f["underlying_price"] = underlying_price
    if underlying_history and len(underlying_history) >= 5 and underlying_history[-5] > 0:
        f["underlying_change_5m"] = (underlying_price / underlying_history[-5] - 1) * 100
    else:
        f["underlying_change_5m"] = 0
    if underlying_history and len(underlying_history) >= 15 and underlying_history[0] > 0:
        f["underlying_change_15m"] = (underlying_price / underlying_history[0] - 1) * 100
    else:
        f["underlying_change_15m"] = 0
    if underlying_history and len(underlying_history) > 2:
        arr = np.array(underlying_history)
        f["underlying_volatility"] = float(np.std(np.diff(arr) / arr[:-1]) * 100)
        # NOTE: training defines vwap_deviation as deviation from the MEAN of the
        # trailing stock-close window (train_option_signals_v2.py:634), NOT true
        # session VWAP. Serving must match training, so we use the same
        # trailing-window mean here. If the model is retrained on real session
        # VWAP (Polygon AM-bar vwap is now captured in ml_pipeline.MinuteBar),
        # update this to match.
        vwap = np.mean(arr)
        f["vwap_deviation"] = (underlying_price / vwap - 1) * 100 if vwap > 0 else 0
    else:
        f["underlying_volatility"] = 0
        f["vwap_deviation"] = 0

    # Market regime features (defaults for live — underlying_history serves as proxy)
    if underlying_history and len(underlying_history) > 10 and underlying_history[0] > 0:
        f["daily_trend_pct"] = (underlying_price / underlying_history[0] - 1) * 100
        day_lo = min(underlying_history)
        day_hi = max(underlying_history)
        f["daily_range_position"] = (underlying_price - day_lo) / (day_hi - day_lo) if day_hi > day_lo else 0.5
        if len(underlying_history) >= 14:
            # Simplified ATR proxy from underlying history
            diffs = [abs(underlying_history[i] - underlying_history[i - 1]) for i in range(1, len(underlying_history))]
            f["atr_pct"] = float(np.mean(diffs[-14:]) / underlying_price * 100) if underlying_price > 0 else 0
        else:
            f["atr_pct"] = 0
        if len(underlying_history) >= 5:
            f["pre_move_underlying_5m"] = (underlying_price / underlying_history[-5] - 1) * 100
        else:
            f["pre_move_underlying_5m"] = 0
    else:
        f["daily_trend_pct"] = 0
        f["daily_range_position"] = 0.5
        f["atr_pct"] = 0
        f["pre_move_underlying_5m"] = 0

    # Institutional sweep features (simplified for live — detect recent key level sweeps)
    if underlying_history and len(underlying_history) > 5:
        recent_high = max(underlying_history[-5:])
        recent_low = min(underlying_history[-5:])
        f["sweep_high"] = 1 if (underlying_price < recent_high and max(underlying_history[-2:]) >= recent_high) else 0
        f["sweep_low"] = 1 if (underlying_price > recent_low and min(underlying_history[-2:]) <= recent_low) else 0
        f["near_key_level"] = 1 if (
            abs(underlying_price - recent_high) / max(recent_high, 1) < 0.001
            or abs(underlying_price - recent_low) / max(recent_low, 1) < 0.001
        ) else 0
    else:
        f["sweep_high"] = 0
        f["sweep_low"] = 0
        f["near_key_level"] = 0

    # Computed patterns
    f["coiled_spring"] = 1 if (f["premium_volatility"] < 2 and f["volume_ratio"] > 1.5) else 0
    f["volume_breakout"] = 1 if (f["volume_zscore"] > 2 and f.get("near_high", 0)) else 0
    f["bounce_setup"] = 1 if (f.get("near_low", 0) and f["spread_tightening"] > 0) else 0
    f["iv_expanding"] = 1 if f.get("iv_change_5m", 0) > 0.02 else 0
    f["momentum_ignition"] = 1 if (f.get("consecutive_up_bars", 0) >= 3 and f["volume_trend"] > 1.3) else 0

    # Direction
    f["is_call"] = 1 if is_call else 0

    return f
