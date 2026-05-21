"""Train 3 new ML models on 5 years of historical 0DTE data.

Model 1: Entry Filter — binary classifier "should we take this trade?"
Model 2: Peak Predictor — regressor "what MFE will this trade achieve?"
Model 3: Regime Classifier — "is today trending or choppy?"

Usage:
    python scripts/train_ml_v2.py              # train all 3
    python scripts/train_ml_v2.py --model entry  # train just entry filter
    python scripts/train_ml_v2.py --model peak   # train just peak predictor
    python scripts/train_ml_v2.py --model regime  # train just regime classifier
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pathlib import Path

# Timestamps in DB are milliseconds since epoch (UTC)
# Market hours: 9:30 AM - 4:00 PM ET = 13:30 - 20:00 UTC (EST) or 13:30-20:00
ET_OFFSET_HOURS = -4  # EDT (April-November)... approximate, most of our data is EDT

HISTORICAL_DB = os.environ.get("HISTORICAL_DB", "journal/historical_0dte.db")
HARVESTER_DB = os.environ.get("HARVESTER_DB", "journal/owlet-harvester/options_data.db")
MODELS_DIR = os.environ.get("ML_MODELS_DIR", "journal/models")

# Entry filter thresholds
ENTRY_WIN_THRESHOLD_PCT = 20.0  # "winner" = option went up 20%+ within window
ENTRY_WINDOW_MINUTES = 120  # look-ahead window for winner classification

# Peak prediction settings
PEAK_WINDOW_MINUTES = 240  # max look-ahead for MFE calculation (4 hours, rest of day)

# Regime settings
REGIME_FIRST_N_MINUTES = 30  # use first 30 min to predict regime
REGIME_TREND_THRESHOLD_PCT = 0.5  # day is "trending" if underlying moved > 0.5% directionally


def ts_to_et_hour_min(ts_ms: int) -> tuple[int, int, int]:
    """Convert ms timestamp to (hour, minute, day_of_week) in approximate ET."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    et = dt + timedelta(hours=ET_OFFSET_HOURS)
    return et.hour, et.minute, et.weekday()


def load_underlying_bars(conn: sqlite3.Connection, ticker: str, date: str) -> pd.DataFrame:
    """Load underlying minute bars for a ticker/date, sorted by time."""
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM underlying_bars
        WHERE ticker = ? AND date = ?
        ORDER BY timestamp
    """, (ticker, date)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "vwap"])
    return df


def load_option_bars(conn: sqlite3.Connection, contract_ticker: str) -> pd.DataFrame:
    """Load option minute bars for a contract, sorted by time."""
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "vwap", "num_trades"])
    return df


def compute_underlying_features(u_bars: pd.DataFrame, entry_idx: int) -> dict:
    """Compute features from underlying bars up to entry_idx."""
    if entry_idx < 5:
        return {}

    close = u_bars["close"].values
    volume = u_bars["volume"].values
    high = u_bars["high"].values
    low = u_bars["low"].values
    entry_price = close[entry_idx]

    if entry_price <= 0:
        return {}

    features = {}

    # Price momentum (lookback periods)
    for lb in [5, 10, 15, 30]:
        if entry_idx >= lb:
            prev = close[entry_idx - lb]
            if prev > 0:
                features[f"underlying_mom_{lb}m"] = (entry_price - prev) / prev * 100
            else:
                features[f"underlying_mom_{lb}m"] = 0.0
        else:
            features[f"underlying_mom_{lb}m"] = 0.0

    # Volatility (std of 1-min returns)
    lookback = min(30, entry_idx)
    if lookback >= 5:
        rets = np.diff(close[entry_idx - lookback:entry_idx + 1]) / close[entry_idx - lookback:entry_idx]
        rets = rets[np.isfinite(rets)]
        features["underlying_vol_30m"] = float(np.std(rets) * 100) if len(rets) > 1 else 0.0
    else:
        features["underlying_vol_30m"] = 0.0

    # Volume profile
    vol_window = min(30, entry_idx)
    if vol_window >= 5:
        recent_vol = volume[entry_idx - vol_window:entry_idx + 1]
        avg_vol = np.mean(recent_vol)
        features["volume_avg_30m"] = float(avg_vol)
        features["volume_current_vs_avg"] = float(volume[entry_idx] / avg_vol) if avg_vol > 0 else 1.0
        # Volume trend
        first_half = np.mean(recent_vol[:vol_window // 2])
        second_half = np.mean(recent_vol[vol_window // 2:])
        features["volume_trend"] = float(second_half / first_half) if first_half > 0 else 1.0
    else:
        features["volume_avg_30m"] = 0.0
        features["volume_current_vs_avg"] = 1.0
        features["volume_trend"] = 1.0

    # Day range position: where is price relative to today's high/low so far
    day_high = np.max(high[:entry_idx + 1])
    day_low = np.min(low[:entry_idx + 1])
    if day_high > day_low:
        features["price_position_in_range"] = (entry_price - day_low) / (day_high - day_low)
    else:
        features["price_position_in_range"] = 0.5

    # Bar range (avg high-low as % of close)
    if lookback >= 5:
        ranges = (high[entry_idx - lookback:entry_idx + 1] - low[entry_idx - lookback:entry_idx + 1]) / close[entry_idx - lookback:entry_idx + 1] * 100
        features["avg_bar_range_pct"] = float(np.mean(ranges[np.isfinite(ranges)]))
    else:
        features["avg_bar_range_pct"] = 0.0

    # VWAP deviation
    if u_bars["vwap"].values[entry_idx] > 0:
        features["vwap_deviation_pct"] = (entry_price - u_bars["vwap"].values[entry_idx]) / u_bars["vwap"].values[entry_idx] * 100
    else:
        features["vwap_deviation_pct"] = 0.0

    # Consecutive up/down bars
    consec_up, consec_down = 0, 0
    for i in range(entry_idx, max(0, entry_idx - 10), -1):
        if i > 0 and close[i] > close[i - 1]:
            consec_up += 1
        elif i > 0 and close[i] < close[i - 1]:
            consec_down += 1
        else:
            break
    features["consec_up_bars"] = consec_up
    features["consec_down_bars"] = consec_down

    # Gap from previous day close (approximated by first bar open vs entry)
    open_price = close[0] if len(close) > 0 else entry_price
    features["day_open_to_now_pct"] = (entry_price - open_price) / open_price * 100 if open_price > 0 else 0.0

    return features


def compute_option_features(o_bars: pd.DataFrame, entry_idx: int, u_entry_price: float) -> dict:
    """Compute option-specific features at entry point."""
    if entry_idx < 1:
        return {}

    close = o_bars["close"].values
    volume = o_bars["volume"].values
    entry_premium = close[entry_idx]

    if entry_premium <= 0:
        return {}

    features = {}

    # Premium level (proxy for moneyness/IV)
    features["entry_premium"] = entry_premium
    features["premium_to_underlying_pct"] = (entry_premium / u_entry_price * 100) if u_entry_price > 0 else 0.0

    # Premium momentum
    for lb in [5, 10]:
        if entry_idx >= lb and close[entry_idx - lb] > 0:
            features[f"premium_mom_{lb}m"] = (entry_premium - close[entry_idx - lb]) / close[entry_idx - lb] * 100
        else:
            features[f"premium_mom_{lb}m"] = 0.0

    # Bid-ask spread proxy (use bar range as proxy)
    high = o_bars["high"].values
    low = o_bars["low"].values
    if entry_premium > 0:
        features["option_bar_range_pct"] = (high[entry_idx] - low[entry_idx]) / entry_premium * 100
    else:
        features["option_bar_range_pct"] = 0.0

    # Option volume
    features["option_volume"] = float(volume[entry_idx])
    lookback = min(10, entry_idx)
    if lookback >= 3:
        avg_vol = np.mean(volume[entry_idx - lookback:entry_idx + 1])
        features["option_vol_vs_avg"] = float(volume[entry_idx] / avg_vol) if avg_vol > 0 else 1.0
    else:
        features["option_vol_vs_avg"] = 1.0

    # Num trades
    if "num_trades" in o_bars.columns:
        features["option_num_trades"] = float(o_bars["num_trades"].values[entry_idx])
    else:
        features["option_num_trades"] = 0.0

    return features


def compute_labels(o_bars: pd.DataFrame, entry_idx: int, window_minutes: int) -> dict:
    """Compute labels: did the option go up 20%+? What was the MFE? etc."""
    close = o_bars["close"].values
    entry_premium = close[entry_idx]

    if entry_premium <= 0:
        return {}

    end_idx = min(entry_idx + window_minutes, len(close))
    if end_idx <= entry_idx + 5:
        return {}  # not enough future data

    future = close[entry_idx + 1:end_idx]
    if len(future) < 5:
        return {}

    mfe = float(np.max(future))
    mae = float(np.min(future))
    mfe_pct = (mfe - entry_premium) / entry_premium * 100
    mae_pct = (mae - entry_premium) / entry_premium * 100
    exit_premium = future[-1]
    exit_pnl_pct = (exit_premium - entry_premium) / entry_premium * 100

    # Peak timing (minutes after entry to reach MFE)
    peak_idx = int(np.argmax(future))

    return {
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "exit_pnl_pct": exit_pnl_pct,
        "is_winner": 1 if mfe_pct >= ENTRY_WIN_THRESHOLD_PCT else 0,
        "peak_minutes": peak_idx + 1,
        "mfe_premium": mfe,
    }


def build_dataset(conn: sqlite3.Connection, sample_interval_min: int = 15,
                  max_days: int = 0, ticker_filter: str | None = None) -> pd.DataFrame:
    """Build the main training dataset from historical_0dte.db.

    For each trading day and each option type (call/put), samples entry points
    every sample_interval_min minutes and computes features + labels.
    """
    # Get all trading days
    query = "SELECT date, ticker, atm_call_ticker, atm_put_ticker, atm_strike, open_price, close_price FROM trading_days"
    params = []
    if ticker_filter:
        query += " WHERE ticker = ?"
        params.append(ticker_filter)
    query += " ORDER BY date"

    days = conn.execute(query, params).fetchall()
    if max_days > 0:
        days = days[:max_days]

    print(f"Processing {len(days)} trading days...")

    all_rows = []
    processed = 0

    for date, ticker, call_ticker, put_ticker, atm_strike, day_open, day_close in days:
        # Load bars
        u_bars = load_underlying_bars(conn, ticker, date)
        if u_bars.empty:
            continue

        # Filter to market hours (9:30 AM - 4:00 PM ET ~ timestamps between specific hours)
        # Market open at 9:30 ET = 13:30 UTC
        market_u = u_bars[(u_bars["ts"] >= u_bars["ts"].min())].copy()
        if len(market_u) < 60:
            continue

        for opt_type, contract_ticker in [("call", call_ticker), ("put", put_ticker)]:
            if not contract_ticker:
                continue

            o_bars = load_option_bars(conn, contract_ticker)
            if o_bars.empty or len(o_bars) < 60:
                continue

            # Align option bars with underlying bars by timestamp
            # Build timestamp -> index maps
            u_ts_to_idx = {ts: i for i, ts in enumerate(market_u["ts"].values)}

            # Sample entry points every N minutes through the option bars
            o_timestamps = o_bars["ts"].values
            o_close = o_bars["close"].values

            # Start 30 min in (need lookback), end 60 min before close (need future data)
            start_idx = 30
            end_idx = len(o_bars) - 60

            for entry_idx in range(start_idx, end_idx, sample_interval_min):
                entry_ts = o_timestamps[entry_idx]
                entry_premium = o_close[entry_idx]

                if entry_premium <= 0.01:
                    continue

                # Time features
                hour, minute, dow = ts_to_et_hour_min(entry_ts)

                # Skip pre/post market
                if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
                    continue

                # Find matching underlying bar
                u_idx = u_ts_to_idx.get(entry_ts)
                if u_idx is None:
                    # Find closest
                    diffs = np.abs(market_u["ts"].values - entry_ts)
                    u_idx = int(np.argmin(diffs))
                    if diffs[u_idx] > 120000:  # > 2 min apart
                        continue

                u_entry_price = market_u["close"].values[u_idx]
                if u_entry_price <= 0:
                    continue

                # Compute features
                u_feats = compute_underlying_features(market_u, u_idx)
                if not u_feats:
                    continue

                o_feats = compute_option_features(o_bars, entry_idx, u_entry_price)
                if not o_feats:
                    continue

                # Compute labels
                labels = compute_labels(o_bars, entry_idx, PEAK_WINDOW_MINUTES)
                if not labels:
                    continue

                # Combine
                row = {
                    "date": date,
                    "ticker": ticker,
                    "option_type": opt_type,
                    "hour": hour,
                    "minute": minute,
                    "day_of_week": dow,
                    "minutes_since_open": (hour - 9) * 60 + (minute - 30),
                    "atm_strike": atm_strike,
                    "is_call": 1 if opt_type == "call" else 0,
                    **u_feats,
                    **o_feats,
                    **labels,
                }
                all_rows.append(row)

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(days)} days, {len(all_rows):,} samples so far...")

    print(f"Dataset complete: {len(all_rows):,} samples from {processed} trading days")
    return pd.DataFrame(all_rows)


def build_regime_dataset(conn: sqlite3.Connection, max_days: int = 0) -> pd.DataFrame:
    """Build regime classification dataset.

    For each trading day, use first 30 min of underlying data to predict
    whether the rest of the day was trending or choppy.
    """
    days = conn.execute("""
        SELECT DISTINCT date, ticker FROM trading_days ORDER BY date
    """).fetchall()
    if max_days > 0:
        days = days[:max_days]

    print(f"Building regime dataset from {len(days)} day/ticker combos...")

    all_rows = []

    for date, ticker in days:
        u_bars = load_underlying_bars(conn, ticker, date)
        if u_bars.empty or len(u_bars) < 200:
            continue

        close = u_bars["close"].values
        volume = u_bars["volume"].values
        high = u_bars["high"].values
        low = u_bars["low"].values
        ts = u_bars["ts"].values

        # Find market open (9:30 AM ET)
        open_idx = None
        for i, t in enumerate(ts):
            h, m, _ = ts_to_et_hour_min(t)
            if h == 9 and m >= 30:
                open_idx = i
                break
        if open_idx is None or open_idx + REGIME_FIRST_N_MINUTES + 60 >= len(close):
            continue

        # FEATURES: first 30 minutes after open
        first30_end = open_idx + REGIME_FIRST_N_MINUTES
        first30_close = close[open_idx:first30_end]
        first30_vol = volume[open_idx:first30_end]
        first30_high = high[open_idx:first30_end]
        first30_low = low[open_idx:first30_end]

        if len(first30_close) < 20 or first30_close[0] <= 0:
            continue

        # Direction of first 30 min
        move_pct = (first30_close[-1] - first30_close[0]) / first30_close[0] * 100
        abs_move_pct = abs(move_pct)

        # Volatility of first 30 min
        rets = np.diff(first30_close) / first30_close[:-1]
        rets = rets[np.isfinite(rets)]
        vol_30m = float(np.std(rets) * 100) if len(rets) > 1 else 0.0

        # Range in first 30 min
        range_30m = (np.max(first30_high) - np.min(first30_low)) / first30_close[0] * 100

        # Volume in first 30 min
        avg_vol_30m = float(np.mean(first30_vol))
        vol_first15 = float(np.mean(first30_vol[:15]))
        vol_last15 = float(np.mean(first30_vol[15:]))
        vol_ratio = vol_last15 / vol_first15 if vol_first15 > 0 else 1.0

        # Number of direction changes (sign changes in returns)
        signs = np.sign(rets[rets != 0])
        direction_changes = int(np.sum(np.abs(np.diff(signs)) > 0)) if len(signs) > 1 else 0

        # Max consecutive same-direction bars
        max_run = 1
        current_run = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Vwap trend
        vwap_vals = u_bars["vwap"].values[open_idx:first30_end]
        vwap_trend = 0.0
        if len(vwap_vals) > 5 and vwap_vals[0] > 0:
            vwap_trend = (vwap_vals[-1] - vwap_vals[0]) / vwap_vals[0] * 100

        # LABELS: rest of day behavior (after first 30 min)
        rest_close = close[first30_end:]
        if len(rest_close) < 30:
            continue

        # Total move rest-of-day
        rod_move = (rest_close[-1] - rest_close[0]) / rest_close[0] * 100
        rod_range = (np.max(close[first30_end:]) - np.min(close[first30_end:])) / rest_close[0] * 100

        # Is trending? big directional move in rest of day
        is_trending = 1 if abs(rod_move) >= REGIME_TREND_THRESHOLD_PCT else 0

        # Directional ratio: how much of the range was directional vs choppy
        # trending: |move| / range is high. choppy: |move| / range is low
        directional_ratio = abs(rod_move) / rod_range if rod_range > 0 else 0.5

        # MFE for call options (how much could you make rest of day)
        rod_mfe_up = (np.max(rest_close) - rest_close[0]) / rest_close[0] * 100
        rod_mfe_down = (rest_close[0] - np.min(rest_close)) / rest_close[0] * 100

        hour, minute, dow = ts_to_et_hour_min(ts[open_idx])

        row = {
            "date": date,
            "ticker": ticker,
            "day_of_week": dow,
            # First 30 min features
            "first30_move_pct": move_pct,
            "first30_abs_move_pct": abs_move_pct,
            "first30_vol": vol_30m,
            "first30_range_pct": range_30m,
            "first30_avg_volume": avg_vol_30m,
            "first30_vol_ratio": vol_ratio,
            "first30_direction_changes": direction_changes,
            "first30_max_run": max_run,
            "first30_vwap_trend": vwap_trend,
            # Labels
            "rod_move_pct": rod_move,
            "rod_range_pct": rod_range,
            "is_trending": is_trending,
            "directional_ratio": directional_ratio,
            "rod_mfe_up_pct": rod_mfe_up,
            "rod_mfe_down_pct": rod_mfe_down,
        }
        all_rows.append(row)

    print(f"Regime dataset: {len(all_rows):,} day/ticker combos")
    return pd.DataFrame(all_rows)


def train_entry_filter(df: pd.DataFrame, models_dir: str, suffix: str = ""):
    """Train Model 1: Entry Filter (binary classifier)."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    print("\n" + "=" * 80)
    print("MODEL 1: ENTRY FILTER — Should we take this trade?")
    print("=" * 80)

    feature_cols = [
        "hour", "minute", "day_of_week", "minutes_since_open", "is_call",
        "underlying_mom_5m", "underlying_mom_10m", "underlying_mom_15m", "underlying_mom_30m",
        "underlying_vol_30m",
        "volume_avg_30m", "volume_current_vs_avg", "volume_trend",
        "price_position_in_range", "avg_bar_range_pct", "vwap_deviation_pct",
        "consec_up_bars", "consec_down_bars", "day_open_to_now_pct",
        "entry_premium", "premium_to_underlying_pct",
        "premium_mom_5m", "premium_mom_10m",
        "option_bar_range_pct", "option_volume", "option_vol_vs_avg", "option_num_trades",
    ]

    # Encode ticker
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["ticker_encoded"] = le.fit_transform(df["ticker"])
    feature_cols.append("ticker_encoded")

    label_col = "is_winner"

    # Drop rows with missing values
    valid = df.dropna(subset=feature_cols + [label_col])
    print(f"Training samples: {len(valid):,}")
    print(f"Winner rate: {valid[label_col].mean():.1%}")

    X = valid[feature_cols].values
    y = valid[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, y_test, feature_name=feature_cols, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Loser", "Winner"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop 15 Features:")
    for i in sorted_idx[:15]:
        print(f"  {feature_cols[i]:<35} {importance[i]:>10.0f}")

    # Threshold analysis — find optimal threshold for different precision targets
    print("\nThreshold Analysis (for trade rejection):")
    print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>8} {'Trades Taken':>14} {'Winners Caught':>16}")
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        pred = (y_pred_proba >= thresh).astype(int)
        taken = pred.sum()
        true_pos = ((pred == 1) & (y_test == 1)).sum()
        prec = true_pos / taken if taken > 0 else 0
        recall = true_pos / y_test.sum() if y_test.sum() > 0 else 0
        print(f"  {thresh:<12.2f} {prec:>9.1%} {recall:>7.1%} {taken:>14,} / {len(y_test):,} {true_pos:>10,} / {int(y_test.sum()):,}")

    # Save
    model_path = os.path.join(models_dir, f"entry_filter_v2{suffix}.lgb")
    model.save_model(model_path)
    print(f"\nSaved: {model_path}")

    # Save feature list and ticker encoder
    import json
    meta = {
        "feature_cols": feature_cols,
        "ticker_classes": le.classes_.tolist(),
        "win_threshold_pct": ENTRY_WIN_THRESHOLD_PCT,
        "window_minutes": ENTRY_WINDOW_MINUTES,
        "best_iteration": model.best_iteration,
        "afternoon_only": suffix == "_afternoon",
        "min_minutes_since_open": 210 if suffix == "_afternoon" else 0,
    }
    meta_path = os.path.join(models_dir, f"entry_filter_v2{suffix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    return model


def train_peak_predictor(df: pd.DataFrame, models_dir: str, suffix: str = ""):
    """Train Model 2: Peak Predictor (regressor)."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\n" + "=" * 80)
    print("MODEL 2: PEAK PREDICTOR — What MFE will this trade achieve?")
    print("=" * 80)

    feature_cols = [
        "hour", "minute", "day_of_week", "minutes_since_open", "is_call",
        "underlying_mom_5m", "underlying_mom_10m", "underlying_mom_15m", "underlying_mom_30m",
        "underlying_vol_30m",
        "volume_avg_30m", "volume_current_vs_avg", "volume_trend",
        "price_position_in_range", "avg_bar_range_pct", "vwap_deviation_pct",
        "consec_up_bars", "consec_down_bars", "day_open_to_now_pct",
        "entry_premium", "premium_to_underlying_pct",
        "premium_mom_5m", "premium_mom_10m",
        "option_bar_range_pct", "option_volume", "option_vol_vs_avg", "option_num_trades",
    ]

    if "ticker_encoded" not in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["ticker_encoded"] = le.fit_transform(df["ticker"])
    feature_cols.append("ticker_encoded")

    label_col = "mfe_pct"

    valid = df.dropna(subset=feature_cols + [label_col])
    # Clip extreme MFE values for better training
    valid = valid[valid[label_col].between(-50, 500)].copy()

    print(f"Training samples: {len(valid):,}")
    print(f"MFE distribution: mean={valid[label_col].mean():.1f}%, median={valid[label_col].median():.1f}%, "
          f"p25={valid[label_col].quantile(0.25):.1f}%, p75={valid[label_col].quantile(0.75):.1f}%")

    X = valid[feature_cols].values
    y = valid[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, y_test, feature_name=feature_cols, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_test)

    print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.2f}%")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # Bucketized accuracy: how well does it separate low vs high MFE trades?
    print("\nPredicted MFE Buckets vs Actual:")
    print(f"  {'Predicted MFE':<20} {'Count':>7} {'Actual Mean MFE':>16} {'Actual Median':>14} {'Win Rate (>20%)':>16}")
    for lo, hi in [(-999, 5), (5, 15), (15, 30), (30, 50), (50, 100), (100, 999)]:
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            actual = y_test[mask]
            wr = (actual >= 20).mean()
            label = f"[{lo:+d}%, {hi:+d}%)" if lo > -999 else f"[<{hi}%)"
            if hi == 999:
                label = f"[{lo}%+)"
            print(f"  {label:<20} {mask.sum():>7,} {actual.mean():>+15.1f}% {np.median(actual):>+13.1f}% {wr:>15.1%}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop 15 Features:")
    for i in sorted_idx[:15]:
        print(f"  {feature_cols[i]:<35} {importance[i]:>10.0f}")

    # Also train a peak_minutes regressor (how long to hold)
    print("\n--- Training peak timing sub-model ---")
    peak_label = "peak_minutes"
    valid_peak = valid.dropna(subset=[peak_label])
    y_peak = valid_peak[peak_label].values
    X_peak = valid_peak[feature_cols].values

    X_pt, X_ptest, y_pt, y_ptest = train_test_split(X_peak, y_peak, test_size=0.2, random_state=42)
    dtrain_p = lgb.Dataset(X_pt, y_pt, feature_name=feature_cols)
    dtest_p = lgb.Dataset(X_ptest, y_ptest, feature_name=feature_cols, reference=dtrain_p)

    params_p = {**params, "metric": "mae"}
    model_timing = lgb.train(
        params_p, dtrain_p,
        num_boost_round=500,
        valid_sets=[dtest_p],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    y_timing_pred = model_timing.predict(X_ptest)
    print(f"Peak Timing MAE: {mean_absolute_error(y_ptest, y_timing_pred):.1f} minutes")

    # Save
    model_path = os.path.join(models_dir, f"peak_predictor_v2{suffix}.lgb")
    model.save_model(model_path)
    print(f"\nSaved: {model_path}")

    timing_path = os.path.join(models_dir, f"peak_timing_v2{suffix}.lgb")
    model_timing.save_model(timing_path)
    print(f"Saved: {timing_path}")

    import json
    meta = {
        "feature_cols": feature_cols,
        "peak_window_minutes": PEAK_WINDOW_MINUTES,
        "mfe_mae": float(mean_absolute_error(y_test, y_pred)),
        "mfe_r2": float(r2_score(y_test, y_pred)),
        "timing_mae_minutes": float(mean_absolute_error(y_ptest, y_timing_pred)),
        "afternoon_only": suffix == "_afternoon",
    }
    meta_path = os.path.join(models_dir, f"peak_predictor_v2{suffix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    return model, model_timing


def train_regime_classifier(regime_df: pd.DataFrame, models_dir: str):
    """Train Model 3: Regime Classifier (trending vs choppy)."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    print("\n" + "=" * 80)
    print("MODEL 3: REGIME CLASSIFIER — Is today trending or choppy?")
    print("=" * 80)

    feature_cols = [
        "day_of_week",
        "first30_move_pct", "first30_abs_move_pct", "first30_vol",
        "first30_range_pct", "first30_avg_volume", "first30_vol_ratio",
        "first30_direction_changes", "first30_max_run", "first30_vwap_trend",
    ]

    # Encode ticker
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    regime_df["ticker_encoded"] = le.fit_transform(regime_df["ticker"])
    feature_cols.append("ticker_encoded")

    label_col = "is_trending"

    valid = regime_df.dropna(subset=feature_cols + [label_col])
    print(f"Training samples: {len(valid):,}")
    print(f"Trending rate: {valid[label_col].mean():.1%}")

    X = valid[feature_cols].values
    y = valid[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Also store rest-of-day metrics for evaluation
    rod_mfe_up = valid["rod_mfe_up_pct"].values
    _, rod_mfe_test, _, _ = train_test_split(rod_mfe_up, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, y_test, feature_name=feature_cols, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Choppy", "Trending"]))
    try:
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    except ValueError:
        print("AUC-ROC: N/A (single class)")

    # How useful is regime for trading?
    print("\nRegime Impact on Option MFE:")
    print(f"  {'Predicted':<12} {'Count':>7} {'Avg MFE Up':>12} {'Suggestion'}")
    for label, name in [(0, "Choppy"), (1, "Trending")]:
        mask = y_pred == label
        if mask.sum() > 0:
            avg_mfe = rod_mfe_test[mask].mean()
            suggestion = "tighter trails" if label == 0 else "wider trails"
            print(f"  {name:<12} {mask.sum():>7,} {avg_mfe:>+11.2f}% {suggestion}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop Features:")
    for i in sorted_idx[:10]:
        print(f"  {feature_cols[i]:<35} {importance[i]:>10.0f}")

    # Also train a directional_ratio regressor (how trending vs choppy)
    print("\n--- Training directional ratio sub-model ---")
    dr_label = "directional_ratio"
    valid_dr = valid.dropna(subset=[dr_label])
    y_dr = valid_dr[dr_label].values
    X_dr = valid_dr[feature_cols].values

    X_dt, X_dtest, y_dt, y_dtest = train_test_split(X_dr, y_dr, test_size=0.2, random_state=42)
    dtrain_d = lgb.Dataset(X_dt, y_dt, feature_name=feature_cols)
    dtest_d = lgb.Dataset(X_dtest, y_dtest, feature_name=feature_cols, reference=dtrain_d)

    params_d = {**params, "objective": "regression", "metric": "mae"}
    model_dr = lgb.train(
        params_d, dtrain_d,
        num_boost_round=500,
        valid_sets=[dtest_d],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    from sklearn.metrics import mean_absolute_error
    y_dr_pred = model_dr.predict(X_dtest)
    print(f"Directional Ratio MAE: {mean_absolute_error(y_dtest, y_dr_pred):.4f}")

    # Save
    model_path = os.path.join(models_dir, "regime_classifier_v2.lgb")
    model.save_model(model_path)
    print(f"\nSaved: {model_path}")

    dr_path = os.path.join(models_dir, "regime_direction_v2.lgb")
    model_dr.save_model(dr_path)
    print(f"Saved: {dr_path}")

    import json
    meta = {
        "feature_cols": feature_cols,
        "ticker_classes": le.classes_.tolist(),
        "first_n_minutes": REGIME_FIRST_N_MINUTES,
        "trend_threshold_pct": REGIME_TREND_THRESHOLD_PCT,
    }
    meta_path = os.path.join(models_dir, "regime_classifier_v2_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    return model, model_dr


def main():
    parser = argparse.ArgumentParser(description="Train ML v2 models")
    parser.add_argument("--model", choices=["entry", "peak", "regime", "all"], default="all")
    parser.add_argument("--max-days", type=int, default=0, help="Limit trading days (0=all)")
    parser.add_argument("--sample-interval", type=int, default=15, help="Minutes between entry samples")
    parser.add_argument("--ticker", type=str, default=None, help="Train on single ticker only")
    parser.add_argument("--afternoon-only", action="store_true",
                        help="Only train on afternoon entries (after 1 PM ET / 210 min since open)")
    parser.add_argument("--min-minutes-since-open", type=int, default=0,
                        help="Only include entries at least this many minutes after market open")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    conn = sqlite3.connect(HISTORICAL_DB)

    # Determine minimum minutes since open filter
    min_mso = args.min_minutes_since_open
    if args.afternoon_only:
        min_mso = max(min_mso, 210)  # 1:00 PM ET = 210 min after 9:30
        print(f"AFTERNOON-ONLY MODE: filtering to entries >= {min_mso} min after open (1:00 PM ET+)")

    if args.model in ("entry", "peak", "all"):
        print("Building main dataset (entry + peak)...")
        df = build_dataset(conn, sample_interval_min=args.sample_interval,
                          max_days=args.max_days, ticker_filter=args.ticker)

        if min_mso > 0:
            before = len(df)
            df = df[df["minutes_since_open"] >= min_mso].copy()
            print(f"Filtered to entries >= {min_mso} min since open: {before:,} → {len(df):,} samples")

        if len(df) < 100:
            print("ERROR: Too few samples. Check database path.")
            sys.exit(1)

        # Save dataset for debugging
        suffix = f"_afternoon" if min_mso >= 210 else ""
        csv_path = os.path.join(MODELS_DIR, f"ml_v2_dataset{suffix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved dataset: {csv_path} ({len(df):,} rows)")

        model_suffix = "_afternoon" if min_mso >= 210 else ""

        if args.model in ("entry", "all"):
            train_entry_filter(df, MODELS_DIR, suffix=model_suffix)

        if args.model in ("peak", "all"):
            train_peak_predictor(df, MODELS_DIR, suffix=model_suffix)

    if args.model in ("regime", "all"):
        print("\nBuilding regime dataset...")
        regime_df = build_regime_dataset(conn, max_days=args.max_days)

        if len(regime_df) < 100:
            print("ERROR: Too few regime samples.")
            sys.exit(1)

        if args.model in ("regime", "all"):
            train_regime_classifier(regime_df, MODELS_DIR)

    conn.close()
    print("\n" + "=" * 80)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
