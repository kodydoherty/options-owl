"""ML-powered sell timing for 0DTE options.

Loads pre-trained LightGBM models (per-ticker for SPY/QQQ/IWM, generic fallback
for everything else) and provides a should_sell() function that the exit pipeline
calls each monitoring cycle.

The model uses the trade's current state (PnL, time held, momentum, etc.) to
predict whether holding has positive expected value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger

# Feature constants (must match training)
SWEET_SPOT_START_MIN = 45
SWEET_SPOT_END_MIN = 150
HARD_DEADLINE_MIN = 240

SELL_THRESHOLD = 0.4  # classifier probability threshold

MODELS_DIR = os.environ.get("ML_MODELS_DIR", os.path.join(os.getcwd(), "journal", "models"))

# Per-ticker models available (trained on harvester data with real entry times)
PER_TICKER_MODELS = {
    'AAPL', 'AMD', 'AMZN', 'COIN', 'GOOGL', 'IWM', 'JPM', 'META',
    'MSFT', 'MSTR', 'MU', 'NVDA', 'PLTR', 'QQQ', 'SPY', 'TLT',
    'TSLA', 'XLF',
}

# Lazy-loaded model cache
_model_cache: dict[str, tuple] = {}  # key -> (clf, reg)


@dataclass
class MLSellSignal:
    """Result from the ML sell model."""
    should_sell: bool
    sell_probability: float
    expected_future_pnl: float
    reason: str
    model_used: str


def _load_models(ticker: str) -> tuple | None:
    """Load classifier + regressor for a ticker. Returns (clf, reg) or None."""
    cache_key = ticker.upper() if ticker.upper() in PER_TICKER_MODELS else "generic"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — ML exit disabled")
        _model_cache[cache_key] = None
        return None

    clf_path = os.path.join(MODELS_DIR, f"{cache_key.lower()}_clf.lgb")
    reg_path = os.path.join(MODELS_DIR, f"{cache_key.lower()}_reg.lgb")

    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        logger.warning(f"ML model not found at {clf_path} — ML exit disabled for {cache_key}")
        _model_cache[cache_key] = None
        return None

    clf = lgb.Booster(model_file=clf_path)
    reg = lgb.Booster(model_file=reg_path)
    _model_cache[cache_key] = (clf, reg)
    logger.info(f"Loaded ML model: {cache_key} (clf={clf.best_iteration} iters, reg={reg.best_iteration} iters)")
    return (clf, reg)


def compute_features(
    entry_premium: float,
    current_premium: float,
    peak_premium: float,
    minutes_since_entry: float,
    now_hour: int,
    now_minute: int,
    ticker: str,
    is_call: bool,
    premium_history: list[float] | None = None,
    underlying_entry: float | None = None,
    underlying_current: float | None = None,
    volume: float = 0,
    bar_range_pct: float = 0,
) -> dict:
    """Compute features for a single prediction from current trade state.

    premium_history: list of recent premiums (newest last), used for velocity/volatility.
    If None, velocities default to 0.
    """
    if entry_premium <= 0:
        entry_premium = 0.01

    pnl_pct = (current_premium - entry_premium) / entry_premium * 100
    mfe_pct = (peak_premium - entry_premium) / entry_premium * 100 if peak_premium > entry_premium else 0
    dd_pct = (peak_premium - current_premium) / peak_premium * 100 if peak_premium > 0 else 0

    mins = minutes_since_entry
    in_sweet = 1 if SWEET_SPOT_START_MIN <= mins <= SWEET_SPOT_END_MIN else 0
    past_sweet = 1 if mins > SWEET_SPOT_END_MIN else 0
    mins_past = max(0, mins - SWEET_SPOT_END_MIN)
    time_pressure = min(1.0, mins / HARD_DEADLINE_MIN)

    # Velocities from premium history
    v5, v10, v15 = 0.0, 0.0, 0.0
    accel = 0.0
    roll_vol = 0.0
    consec_down = 0
    bars_since_high = 0

    if premium_history and len(premium_history) >= 2:
        hist = premium_history

        def _vel(lookback):
            if len(hist) > lookback and hist[-lookback-1] > 0:
                return (hist[-1] - hist[-lookback-1]) / hist[-lookback-1] * 100 / lookback
            return 0.0

        v5 = _vel(5)
        v10 = _vel(10)
        v15 = _vel(15)

        if len(hist) >= 10:
            # Acceleration
            p5 = hist[-6] if len(hist) > 5 else hist[0]
            p10 = hist[-11] if len(hist) > 10 else hist[0]
            if p5 > 0 and p10 > 0:
                vb = (p5 - p10) / p10 * 100 / 5
                accel = v5 - vb

            # Rolling volatility
            rets = []
            for j in range(max(0, len(hist)-10), len(hist)):
                if j > 0 and hist[j-1] > 0:
                    rets.append((hist[j] - hist[j-1]) / hist[j-1] * 100)
            if rets:
                import numpy as np
                roll_vol = float(np.std(rets))

        # Consecutive down bars
        for j in range(len(hist)-1, 0, -1):
            if hist[j] < hist[j-1]:
                consec_down += 1
            else:
                break

        # Bars since new high
        peak_val = max(hist)
        for j in range(len(hist)-1, -1, -1):
            if hist[j] >= peak_val * 0.999:  # within 0.1% of peak
                bars_since_high = len(hist) - 1 - j
                break

    # MFE retracement
    mfe_retrace = 0.0
    if mfe_pct > 1.0 and mfe_pct > 0:
        mfe_retrace = min(dd_pct / mfe_pct, 2.0)

    # Volume vs avg (default 1.0 if no history)
    vol_vs_avg = 1.0

    # Underlying
    u_pnl = 0.0
    u_vel = 0.0
    if underlying_entry and underlying_current and underlying_entry > 0:
        u_pnl = (underlying_current - underlying_entry) / underlying_entry * 100

    rr = mfe_pct / max(dd_pct, 1.0)

    features = {
        "pnl_pct": pnl_pct,
        "mfe_pct": mfe_pct,
        "drawdown_from_peak_pct": dd_pct,
        "minutes_since_entry": mins,
        "hour_of_day": now_hour,
        "minute_of_hour": now_minute,
        "in_sweet_spot": in_sweet,
        "past_sweet_spot": past_sweet,
        "minutes_past_sweet_spot": mins_past,
        "time_pressure": time_pressure,
        "premium_velocity_5m": v5,
        "premium_velocity_10m": v10,
        "premium_velocity_15m": v15,
        "pnl_acceleration": accel,
        "mfe_retracement_ratio": mfe_retrace,
        "bar_range_pct": bar_range_pct,
        "volume": volume,
        "volume_vs_avg": vol_vs_avg,
        "rolling_volatility_10m": roll_vol,
        "underlying_pnl_pct": u_pnl,
        "underlying_velocity_5m": u_vel,
        "is_call": 1 if is_call else 0,
        "entry_premium": entry_premium,
        "bars_since_new_high": bars_since_high,
        "consecutive_down_bars": consec_down,
        "risk_reward_ratio": rr,
    }

    return features


def predict_sell(
    ticker: str,
    entry_premium: float,
    current_premium: float,
    peak_premium: float,
    minutes_since_entry: float,
    now_hour: int,
    now_minute: int,
    is_call: bool,
    premium_history: list[float] | None = None,
    underlying_entry: float | None = None,
    underlying_current: float | None = None,
) -> MLSellSignal:
    """Run the ML model and return a sell/hold signal."""
    models = _load_models(ticker)
    if models is None:
        return MLSellSignal(
            should_sell=False, sell_probability=0.0, expected_future_pnl=0.0,
            reason="ML model not available", model_used="none",
        )

    clf, reg = models
    model_name = ticker.upper() if ticker.upper() in PER_TICKER_MODELS else "generic"

    features = compute_features(
        entry_premium=entry_premium,
        current_premium=current_premium,
        peak_premium=peak_premium,
        minutes_since_entry=minutes_since_entry,
        now_hour=now_hour,
        now_minute=now_minute,
        ticker=ticker,
        is_call=is_call,
        premium_history=premium_history,
        underlying_entry=underlying_entry,
        underlying_current=underlying_current,
    )

    # Build feature array in correct order
    if model_name == "generic":
        # Generic model needs ticker_encoded
        from sklearn.preprocessing import LabelEncoder
        ALL_TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "META", "AMD",
                       "AMZN", "GOOGL", "MSFT", "IWM", "MU", "MSTR"]
        te = LabelEncoder()
        te.fit(ALL_TICKERS)
        tc = int(te.transform([ticker.upper()])[0]) if ticker.upper() in ALL_TICKERS else 0
        features["ticker_encoded"] = tc
        feature_order = [
            "pnl_pct", "mfe_pct", "drawdown_from_peak_pct",
            "minutes_since_entry", "hour_of_day", "minute_of_hour",
            "in_sweet_spot", "past_sweet_spot", "minutes_past_sweet_spot",
            "time_pressure",
            "premium_velocity_5m", "premium_velocity_10m", "premium_velocity_15m",
            "pnl_acceleration", "mfe_retracement_ratio",
            "bar_range_pct", "volume", "volume_vs_avg",
            "rolling_volatility_10m",
            "underlying_pnl_pct", "underlying_velocity_5m",
            "is_call", "entry_premium",
            "bars_since_new_high", "consecutive_down_bars",
            "risk_reward_ratio",
            "ticker_encoded",
        ]
    else:
        feature_order = [
            "pnl_pct", "mfe_pct", "drawdown_from_peak_pct",
            "minutes_since_entry", "hour_of_day", "minute_of_hour",
            "in_sweet_spot", "past_sweet_spot", "minutes_past_sweet_spot",
            "time_pressure",
            "premium_velocity_5m", "premium_velocity_10m", "premium_velocity_15m",
            "pnl_acceleration", "mfe_retracement_ratio",
            "bar_range_pct", "volume", "volume_vs_avg",
            "rolling_volatility_10m",
            "underlying_pnl_pct", "underlying_velocity_5m",
            "is_call", "entry_premium",
            "bars_since_new_high", "consecutive_down_bars",
            "risk_reward_ratio",
        ]

    import numpy as np
    X = np.array([[features[f] for f in feature_order]])

    sell_prob = float(clf.predict(X, num_iteration=clf.best_iteration)[0])
    exp_future = float(reg.predict(X, num_iteration=reg.best_iteration)[0])

    # Combo logic (same as backtest)
    should_sell = False
    reason = "hold"

    if sell_prob > SELL_THRESHOLD and exp_future < 2.0:
        should_sell = True
        reason = f"ML combo: P(sell)={sell_prob:.2f}, E[future]={exp_future:+.1f}%"
    elif minutes_since_entry >= 10 and exp_future < -10.0:
        should_sell = True
        reason = f"ML regressor bearish: E[future]={exp_future:+.1f}%"

    return MLSellSignal(
        should_sell=should_sell,
        sell_probability=sell_prob,
        expected_future_pnl=exp_future,
        reason=reason,
        model_used=model_name,
    )
