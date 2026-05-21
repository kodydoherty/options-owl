"""Train ML peak detector — predict when a runner is near its peak.

Uses tick-by-tick harvester data with greeks, IV, volume to build a
LightGBM classifier that predicts: "will premium drop 20%+ from here
within the next N minutes?"

Two models:
  1. Binary classifier: is this tick near-peak? (premium will drop 20%+ soon)
  2. Regressor: what % of eventual peak is this tick? (0-100%)

Output: trained model + feature importance + backtest simulation.

Usage:
    python scripts/train_peak_detector.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from backtest_ladder_report import load_signals, load_ticks, size_contracts
from options_owl.risk.exit_v5.config import categorize_ticker

HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Label parameters
DROP_THRESHOLD_PCT = 20  # "near peak" = premium drops 20%+ from here
DROP_WINDOW_MINUTES = 10  # within this many minutes
MIN_GAIN_TO_INCLUDE = 10  # only include ticks where trade is +10%+ (in the money)


def extract_features(df, entry_premium, ticker, direction, dte):
    """Extract per-tick features for peak detection.

    Returns DataFrame with one row per tick, including features and labels.
    """
    rows = []
    category = categorize_ticker(ticker).value
    is_call = direction in ("bullish", "call")

    # Pre-compute underlying entry
    first_u = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_u = float(u)
            break

    # Get actual peak premium for the whole trade
    trade_peak = df["premium"].max()

    premiums = df["premium"].values
    timestamps = pd.to_datetime(df["ts"]).values

    for idx in range(5, len(df)):  # need at least 5 prior ticks
        premium = premiums[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        gain_pct = (premium - entry_premium) / entry_premium * 100
        if gain_pct < MIN_GAIN_TO_INCLUDE:
            continue

        now = timestamps[idx]
        now_dt = pd.Timestamp(now)
        et_hour = now_dt.hour - 4
        if et_hour < 0:
            et_hour += 24

        # --- Premium velocity/acceleration ---
        lookbacks = [3, 5, 10, 20]  # ticks back
        velocities = {}
        for lb in lookbacks:
            if idx >= lb:
                prev_prem = premiums[idx - lb]
                if prev_prem > 0:
                    velocities[f"vel_{lb}"] = (premium - prev_prem) / prev_prem * 100
                else:
                    velocities[f"vel_{lb}"] = 0
            else:
                velocities[f"vel_{lb}"] = 0

        # Acceleration: change in velocity
        accel_5 = velocities["vel_3"] - velocities["vel_5"] if "vel_3" in velocities else 0

        # Premium relative to recent range
        window = min(20, idx)
        recent = premiums[idx - window:idx + 1]
        recent_valid = recent[~np.isnan(recent)]
        if len(recent_valid) > 1:
            prem_percentile = (premium - recent_valid.min()) / (recent_valid.max() - recent_valid.min()) \
                if recent_valid.max() > recent_valid.min() else 0.5
            prem_std = np.std(recent_valid)
        else:
            prem_percentile = 0.5
            prem_std = 0

        # --- Greeks ---
        def safe_float(val, default=0.0):
            try:
                if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        delta = safe_float(df["delta"].iloc[idx])
        gamma = safe_float(df["gamma"].iloc[idx])
        theta = safe_float(df["theta"].iloc[idx])
        vega = safe_float(df["vega"].iloc[idx])
        iv = safe_float(df["iv"].iloc[idx])

        # Delta change rate
        if idx >= 5:
            prev_delta = safe_float(df["delta"].iloc[idx - 5], delta)
            delta_change = delta - prev_delta
        else:
            delta_change = 0

        # IV change rate
        if idx >= 5:
            prev_iv = safe_float(df["iv"].iloc[idx - 5], iv)
            iv_change = iv - prev_iv
        else:
            iv_change = 0

        # --- Volume ---
        vol = safe_float(df["volume"].iloc[idx])
        if idx >= 10:
            prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
            prev_vols = [v for v in prev_vols if v > 0]
            avg_vol = np.mean(prev_vols) if prev_vols else vol
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # --- Underlying ---
        underlying = safe_float(df["underlying_price"].iloc[idx])
        u_move_from_entry = 0
        if first_u > 0 and underlying > 0:
            u_move_from_entry = (underlying - first_u) / first_u * 100

        # Underlying velocity
        if idx >= 5:
            prev_u = safe_float(df["underlying_price"].iloc[idx - 5], underlying)
            u_vel = (underlying - prev_u) / prev_u * 100 if prev_u > 0 else 0
        else:
            u_vel = 0

        # Premium-underlying divergence (premium rising but underlying fading)
        prem_vel = velocities.get("vel_5", 0)
        if is_call:
            divergence = prem_vel - u_vel * 50  # scaled: 1% underlying ~ 50% premium for 0DTE
        else:
            divergence = prem_vel + u_vel * 50

        # --- Bid-ask spread ---
        bid = safe_float(df["bid"].iloc[idx], premium)
        ask = safe_float(df["ask"].iloc[idx], premium)
        spread_pct = (ask - bid) / premium * 100 if premium > 0 else 0

        # --- Time features ---
        elapsed_min = (now - timestamps[0]).astype("timedelta64[s]").astype(float) / 60
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now_dt.minute))

        # --- Labels ---
        # Label 1: Will premium drop DROP_THRESHOLD_PCT% within DROP_WINDOW_MINUTES?
        future_window = timestamps[idx:] <= now + np.timedelta64(DROP_WINDOW_MINUTES, "m")
        future_prems = premiums[idx:][future_window]
        if len(future_prems) > 1:
            future_min = np.nanmin(future_prems[1:])  # exclude current tick
            drop_from_here = (premium - future_min) / premium * 100
            near_peak = 1 if drop_from_here >= DROP_THRESHOLD_PCT else 0
        else:
            near_peak = -1  # unknown (end of data)

        # Label 2: % of eventual trade peak
        pct_of_peak = premium / trade_peak * 100 if trade_peak > 0 else 0

        # Label 3: Max future premium (for regression)
        future_all = premiums[idx:]
        future_max = np.nanmax(future_all) if len(future_all) > 0 else premium
        upside_remaining = (future_max - premium) / premium * 100

        row = {
            # Features
            "gain_pct": gain_pct,
            "vel_3": velocities["vel_3"],
            "vel_5": velocities["vel_5"],
            "vel_10": velocities["vel_10"],
            "vel_20": velocities["vel_20"],
            "accel_5": accel_5,
            "prem_percentile": prem_percentile,
            "prem_std": prem_std,
            "delta": abs(delta),  # abs because puts have negative delta
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
            "delta_change": delta_change,
            "iv_change": iv_change,
            "volume": vol,
            "vol_ratio": vol_ratio,
            "u_move_entry": u_move_from_entry,
            "u_vel": u_vel,
            "divergence": divergence,
            "spread_pct": spread_pct,
            "elapsed_min": elapsed_min,
            "minutes_to_close": minutes_to_close,
            "et_hour": et_hour,
            "dte": dte,
            "is_highvol": 1 if category == "high_vol" else 0,
            "is_index": 1 if category == "index" else 0,
            # Labels
            "near_peak": near_peak,
            "pct_of_peak": pct_of_peak,
            "upside_remaining": upside_remaining,
            # Metadata
            "ticker": ticker,
            "premium": premium,
            "entry_premium": entry_premium,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    print(f"\n{'=' * 100}")
    print("PEAK DETECTOR — Feature Extraction & Model Training")
    print(f"{'=' * 100}\n")

    all_features = []
    trade_ids = []
    trade_id = 0

    for sig in signals:
        score = sig["score"] or 80
        if score < 78:
            continue
        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue

        ticker = sig["ticker"]
        direction = (sig.get("direction") or "bullish").lower()
        entry = df["premium"].iloc[0]
        if entry <= 0:
            continue

        peak = df["premium"].max()
        peak_gain = (peak - entry) / entry * 100
        if peak_gain < 20:  # only runners
            continue

        dte = sig.get("_dte", 0)
        feat_df = extract_features(df, entry, ticker, direction, dte)
        if len(feat_df) == 0:
            continue

        feat_df["trade_id"] = trade_id
        all_features.append(feat_df)
        trade_ids.append(trade_id)
        trade_id += 1

    harvester_conn.close()

    if not all_features:
        print("No feature data extracted!")
        return

    full_df = pd.concat(all_features, ignore_index=True)
    print(f"Total feature rows: {len(full_df):,}")
    print(f"Trades: {trade_id}")
    print(f"Near-peak labels: {(full_df['near_peak'] == 1).sum():,} positive, "
          f"{(full_df['near_peak'] == 0).sum():,} negative, "
          f"{(full_df['near_peak'] == -1).sum():,} unknown")

    # Remove unknown labels
    train_df = full_df[full_df["near_peak"] >= 0].copy()
    print(f"Training rows (known labels): {len(train_df):,}")
    print(f"Positive rate: {train_df['near_peak'].mean():.1%}")

    # Features
    feature_cols = [
        "gain_pct", "vel_3", "vel_5", "vel_10", "vel_20", "accel_5",
        "prem_percentile", "prem_std",
        "delta", "gamma", "theta", "vega", "iv", "delta_change", "iv_change",
        "volume", "vol_ratio",
        "u_move_entry", "u_vel", "divergence", "spread_pct",
        "elapsed_min", "minutes_to_close", "et_hour", "dte",
        "is_highvol", "is_index",
    ]

    X = train_df[feature_cols].values
    y_cls = train_df["near_peak"].values
    y_reg = train_df["pct_of_peak"].values
    groups = train_df["trade_id"].values

    # === CLASSIFIER: near-peak detection ===
    print(f"\n{'=' * 100}")
    print("MODEL 1: Near-Peak Classifier (will premium drop 20%+ within 10min?)")
    print(f"{'=' * 100}\n")

    gkf = GroupKFold(n_splits=5)
    cls_aucs = []
    cls_accs = []
    all_preds = np.zeros(len(y_cls))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_cls, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cls[train_idx], y_cls[val_idx]

        dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": (y_train == 0).sum() / max(1, (y_train == 1).sum()),
            "verbose": -1,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        all_preds[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        acc = accuracy_score(y_val, (preds > 0.5).astype(int))
        cls_aucs.append(auc)
        cls_accs.append(acc)

        n_trades_val = len(set(groups[val_idx]))
        print(f"  Fold {fold+1}: AUC={auc:.3f} Acc={acc:.3f} "
              f"({n_trades_val} trades, {len(val_idx):,} ticks)")

    print(f"\n  Mean AUC: {np.mean(cls_aucs):.3f} (+/- {np.std(cls_aucs):.3f})")
    print(f"  Mean Acc: {np.mean(cls_accs):.3f}")

    # Train final classifier on all data
    dtrain_full = lgb.Dataset(X, y_cls, feature_name=feature_cols)
    final_cls = lgb.train(
        {**params, "verbose": -1}, dtrain_full, num_boost_round=200,
    )

    # Feature importance
    importance = final_cls.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features (classifier):")
    for feat, imp in feat_imp[:15]:
        print(f"    {feat:<25} {imp:>10,.0f}")

    # === REGRESSOR: % of peak ===
    print(f"\n{'=' * 100}")
    print("MODEL 2: Peak % Regressor (what % of eventual peak is this tick?)")
    print(f"{'=' * 100}\n")

    reg_maes = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_reg, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_reg[train_idx], y_reg[val_idx]

        dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols, reference=dtrain)

        params_reg = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "verbose": -1,
        }

        model = lgb.train(
            params_reg, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        reg_maes.append(mae)

        n_trades_val = len(set(groups[val_idx]))
        print(f"  Fold {fold+1}: MAE={mae:.1f}% ({n_trades_val} trades)")

    print(f"\n  Mean MAE: {np.mean(reg_maes):.1f}% (+/- {np.std(reg_maes):.1f}%)")

    # Train final regressor
    dtrain_full_reg = lgb.Dataset(X, y_reg, feature_name=feature_cols)
    final_reg = lgb.train(
        {**params_reg, "verbose": -1}, dtrain_full_reg, num_boost_round=200,
    )

    importance_reg = final_reg.feature_importance(importance_type="gain")
    feat_imp_reg = sorted(zip(feature_cols, importance_reg), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features (regressor):")
    for feat, imp in feat_imp_reg[:15]:
        print(f"    {feat:<25} {imp:>10,.0f}")

    # === SIMULATION: How would this work in practice? ===
    print(f"\n{'=' * 100}")
    print("SIMULATION: Sell when classifier says 'near peak' (prob > threshold)")
    print(f"{'=' * 100}\n")

    # Use cross-validated predictions
    train_df["pred_prob"] = all_preds

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        total_sold_gain = 0
        total_actual_peak_gain = 0
        total_eventual_gain = 0
        n_sold = 0
        n_trades_sold = 0
        trades_seen = set()

        for trade_id_val in train_df["trade_id"].unique():
            trade_ticks = train_df[train_df["trade_id"] == trade_id_val].sort_values("elapsed_min")
            # Find first tick where model says "near peak"
            trigger = trade_ticks[trade_ticks["pred_prob"] >= threshold]
            if len(trigger) == 0:
                continue

            first_trigger = trigger.iloc[0]
            sell_gain = first_trigger["gain_pct"]
            peak_gain = first_trigger["pct_of_peak"]
            upside = first_trigger["upside_remaining"]

            total_sold_gain += sell_gain
            total_actual_peak_gain += peak_gain
            total_eventual_gain += upside
            n_sold += 1
            n_trades_sold += 1

        if n_sold > 0:
            avg_sell = total_sold_gain / n_sold
            avg_peak_pct = total_actual_peak_gain / n_sold
            avg_upside = total_eventual_gain / n_sold
            print(f"  Threshold {threshold:.1f}: {n_trades_sold:>3} trades triggered | "
                  f"avg sell at +{avg_sell:.0f}% gain | "
                  f"avg {avg_peak_pct:.0f}% of peak captured | "
                  f"avg {avg_upside:.0f}% upside left on table")
        else:
            print(f"  Threshold {threshold:.1f}: 0 trades triggered")

    # Save models
    cls_path = MODEL_DIR / "peak_detector_cls.txt"
    reg_path = MODEL_DIR / "peak_detector_reg.txt"
    final_cls.save_model(str(cls_path))
    final_reg.save_model(str(reg_path))
    print(f"\n  Models saved:")
    print(f"    Classifier: {cls_path}")
    print(f"    Regressor:  {reg_path}")

    # Save feature list
    feat_path = MODEL_DIR / "peak_detector_features.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"    Features:   {feat_path}")


if __name__ == "__main__":
    main()
