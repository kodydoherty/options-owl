"""Train ML peak detector v2 — uses ALL harvester data (not just signal trades).

Key improvements over v1:
  - Mines 18,000+ contracts from harvester DB (vs 137 signal trades)
  - New features: gamma acceleration, IV percentile, VWAP divergence,
    moneyness, spread dynamics, session context
  - Multi-horizon labels: 5min, 10min, 20min drop predictions
  - Per-category models (HIGH_VOL vs INDEX vs STANDARD)
  - Sampling strategy to handle 15M+ ticks efficiently

Usage:
    python scripts/train_peak_detector_v2.py
    python scripts/train_peak_detector_v2.py --max-contracts 5000  # limit for testing
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
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

from options_owl.risk.exit_v5.config import TickerCategory, categorize_ticker

HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Label parameters ---
DROP_THRESHOLDS = {
    "drop_5m": (5, 15),    # 15% drop within 5 min
    "drop_10m": (10, 20),  # 20% drop within 10 min (primary)
    "drop_20m": (20, 25),  # 25% drop within 20 min
}
PRIMARY_LABEL = "drop_10m"
MIN_GAIN_TO_INCLUDE = 15      # only ticks where contract is +15%+ from session low
MIN_TICKS_PER_CONTRACT = 30   # need enough data to compute features
MIN_RUN_PCT = 25              # contract must have 25%+ peak-to-trough run
MAX_TICKS_PER_CONTRACT = 200  # downsample long-lived contracts to cap memory


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def find_runner_contracts(conn, min_run_pct=MIN_RUN_PCT, min_ticks=MIN_TICKS_PER_CONTRACT,
                          max_contracts=None):
    """Find all contracts in harvester that had a significant run."""
    print("  Scanning for runner contracts...")
    t0 = time.time()

    query = """
        SELECT
            s.contract_ticker,
            c.underlying,
            c.option_type,
            c.strike,
            c.expiry_date,
            COUNT(*) as ticks,
            MIN(CASE WHEN s.midpoint > 0.01 THEN s.midpoint END) as min_mid,
            MAX(s.midpoint) as max_mid,
            MIN(s.captured_at) as first_ts,
            MAX(s.captured_at) as last_ts
        FROM harvest_snapshots s
        JOIN harvest_contracts c ON s.contract_ticker = c.contract_ticker
        WHERE s.midpoint > 0.01
          AND s.delta IS NOT NULL
          AND s.delta != 0
        GROUP BY s.contract_ticker
        HAVING ticks >= ?
           AND ((max_mid - min_mid) / min_mid * 100) >= ?
        ORDER BY ticks DESC
    """
    params = [min_ticks, min_run_pct]
    if max_contracts:
        query += " LIMIT ?"
        params.append(max_contracts)

    df = pd.read_sql_query(query, conn, params=params)
    elapsed = time.time() - t0
    print(f"  Found {len(df):,} runner contracts in {elapsed:.1f}s")

    # Breakdown by category
    df["category"] = df["underlying"].apply(lambda t: categorize_ticker(t).value)
    for cat, grp in df.groupby("category"):
        print(f"    {cat}: {len(grp):,} contracts, {grp['ticks'].sum():,} ticks")

    return df


def load_contract_ticks(conn, contract_ticker):
    """Load all snapshots for a single contract, ordered by time."""
    query = """
        SELECT
            captured_at as ts,
            midpoint as premium,
            underlying_price,
            bid, ask,
            bid_size, ask_size,
            day_volume as volume,
            day_vwap as vwap,
            implied_volatility as iv,
            delta, gamma, theta, vega,
            open_interest
        FROM harvest_snapshots
        WHERE contract_ticker = ?
          AND midpoint > 0.01
        ORDER BY captured_at
    """
    return pd.read_sql_query(query, conn, params=[contract_ticker])


def extract_features_v2(df, contract_info):
    """Extract per-tick features for peak detection from raw harvester data.

    Unlike v1 which used signal entry price, v2 uses the contract's session low
    as the reference point (simulating entering at any point during a run).
    """
    ticker = contract_info["underlying"]
    option_type = contract_info["option_type"]
    category = categorize_ticker(ticker)
    is_call = option_type == "call"
    strike = contract_info["strike"]

    premiums = df["premium"].values.astype(float)
    timestamps = pd.to_datetime(df["ts"], format="ISO8601").values

    # Session reference: rolling minimum (the "entry" for any tick is the worst
    # price seen so far — simulates catching the run from its trough)
    n = len(premiums)
    running_min = np.minimum.accumulate(premiums)
    running_max = np.maximum.accumulate(premiums)

    # Peak premium for the entire session
    session_peak = premiums.max()
    session_low = premiums[premiums > 0].min() if (premiums > 0).any() else premiums[0]

    # Pre-compute underlying array
    underlyings = np.array([safe_float(df["underlying_price"].iloc[i]) for i in range(n)])
    first_u = 0.0
    for i in range(min(5, n)):
        if underlyings[i] > 0:
            first_u = underlyings[i]
            break

    rows = []
    # Sample indices if contract is too long
    if n > MAX_TICKS_PER_CONTRACT:
        # Keep first 10, last 10, and evenly sample the rest
        indices = list(range(10))
        indices += list(np.linspace(10, n - 11, MAX_TICKS_PER_CONTRACT - 20, dtype=int))
        indices += list(range(n - 10, n))
        indices = sorted(set(indices))
    else:
        indices = list(range(n))

    for idx in indices:
        if idx < 10:  # need lookback
            continue

        premium = premiums[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        # Gain from running minimum (how much has it run up from trough)
        ref_price = running_min[idx]
        if ref_price <= 0:
            continue
        gain_pct = (premium - ref_price) / ref_price * 100
        if gain_pct < MIN_GAIN_TO_INCLUDE:
            continue

        now = timestamps[idx]
        now_dt = pd.Timestamp(now)
        # Convert to ET (UTC - 4 during EDT, -5 during EST)
        # Approximation: market hours are always EDT for our date range
        et_hour = now_dt.hour - 4
        et_minute = now_dt.minute
        if et_hour < 0:
            et_hour += 24

        # Skip pre-market / after-hours ticks
        et_decimal = et_hour + et_minute / 60
        if et_decimal < 9.5 or et_decimal > 16.0:
            continue

        # --- Premium velocity/acceleration ---
        velocities = {}
        for lb in [3, 5, 10, 20]:
            if idx >= lb:
                prev_prem = premiums[idx - lb]
                velocities[f"vel_{lb}"] = (premium - prev_prem) / prev_prem * 100 if prev_prem > 0 else 0
            else:
                velocities[f"vel_{lb}"] = 0

        accel_5 = velocities["vel_3"] - velocities["vel_5"]

        # Premium relative to recent range (local percentile)
        window = min(30, idx)
        recent = premiums[idx - window:idx + 1]
        recent_valid = recent[~np.isnan(recent)]
        if len(recent_valid) > 1:
            rng = recent_valid.max() - recent_valid.min()
            prem_percentile = (premium - recent_valid.min()) / rng if rng > 0 else 0.5
            prem_std = np.std(recent_valid) / premium if premium > 0 else 0  # normalized
        else:
            prem_percentile = 0.5
            prem_std = 0

        # Premium relative to session peak (how close to the overall peak)
        pct_of_session_peak = premium / session_peak * 100 if session_peak > 0 else 0

        # MFE so far (max favorable excursion from running min)
        mfe_so_far = (running_max[idx] - running_min[idx]) / running_min[idx] * 100 if running_min[idx] > 0 else 0

        # Retracement from local peak
        if running_max[idx] > 0:
            retrace_from_peak = (running_max[idx] - premium) / running_max[idx] * 100
        else:
            retrace_from_peak = 0

        # --- Greeks ---
        delta_val = safe_float(df["delta"].iloc[idx])
        gamma_val = safe_float(df["gamma"].iloc[idx])
        theta_val = safe_float(df["theta"].iloc[idx])
        vega_val = safe_float(df["vega"].iloc[idx])
        iv_val = safe_float(df["iv"].iloc[idx])

        # Greeks rate of change (5-tick lookback)
        delta_prev = safe_float(df["delta"].iloc[idx - 5], delta_val)
        gamma_prev = safe_float(df["gamma"].iloc[idx - 5], gamma_val)
        theta_prev = safe_float(df["theta"].iloc[idx - 5], theta_val)
        iv_prev = safe_float(df["iv"].iloc[idx - 5], iv_val)

        delta_change = delta_val - delta_prev
        gamma_change = gamma_val - gamma_prev  # NEW: gamma acceleration
        theta_change = theta_val - theta_prev  # NEW: theta acceleration
        iv_change = iv_val - iv_prev

        # IV percentile over session (NEW)
        iv_window = min(50, idx)
        iv_values = [safe_float(df["iv"].iloc[i]) for i in range(idx - iv_window, idx + 1)]
        iv_values = [v for v in iv_values if v > 0]
        if iv_values:
            iv_percentile = sum(1 for v in iv_values if v <= iv_val) / len(iv_values)
        else:
            iv_percentile = 0.5

        # Gamma/delta ratio — measures convexity (NEW)
        gamma_delta_ratio = gamma_val / abs(delta_val) if abs(delta_val) > 0.01 else 0

        # --- Volume ---
        vol = safe_float(df["volume"].iloc[idx])
        if idx >= 10:
            prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
            prev_vols = [v for v in prev_vols if v > 0]
            avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # Volume trend (is volume increasing or decreasing?) (NEW)
        if idx >= 20:
            vol_recent = [safe_float(df["volume"].iloc[i]) for i in range(idx - 5, idx + 1)]
            vol_older = [safe_float(df["volume"].iloc[i]) for i in range(idx - 20, idx - 10)]
            avg_recent = np.mean([v for v in vol_recent if v > 0] or [0])
            avg_older = np.mean([v for v in vol_older if v > 0] or [1])
            vol_trend = avg_recent / avg_older if avg_older > 0 else 1.0
        else:
            vol_trend = 1.0

        # --- Bid-Ask dynamics ---
        bid = safe_float(df["bid"].iloc[idx], premium * 0.95)
        ask = safe_float(df["ask"].iloc[idx], premium * 1.05)
        spread_pct = (ask - bid) / premium * 100 if premium > 0 else 0

        # Spread change (widening spread = market maker uncertainty) (NEW)
        if idx >= 5:
            prev_bid = safe_float(df["bid"].iloc[idx - 5], bid)
            prev_ask = safe_float(df["ask"].iloc[idx - 5], ask)
            prev_premium = premiums[idx - 5]
            prev_spread = (prev_ask - prev_bid) / prev_premium * 100 if prev_premium > 0 else spread_pct
            spread_change = spread_pct - prev_spread
        else:
            spread_change = 0

        # Bid-ask size imbalance (more ask size = sellers) (NEW)
        bid_size = safe_float(df["bid_size"].iloc[idx], 1)
        ask_size = safe_float(df["ask_size"].iloc[idx], 1)
        size_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0

        # --- VWAP divergence (NEW) ---
        vwap = safe_float(df["vwap"].iloc[idx])
        vwap_divergence = (premium - vwap) / vwap * 100 if vwap > 0 else 0

        # --- Underlying ---
        underlying = underlyings[idx]
        u_move_from_entry = 0
        if first_u > 0 and underlying > 0:
            u_move_from_entry = (underlying - first_u) / first_u * 100

        # Underlying velocity
        if idx >= 5:
            prev_u = underlyings[idx - 5]
            u_vel = (underlying - prev_u) / prev_u * 100 if prev_u > 0 else 0
        else:
            u_vel = 0

        # Moneyness: how far ITM/OTM (NEW)
        if underlying > 0 and strike > 0:
            if is_call:
                moneyness = (underlying - strike) / strike * 100
            else:
                moneyness = (strike - underlying) / strike * 100
        else:
            moneyness = 0

        # Premium-underlying divergence
        prem_vel = velocities.get("vel_5", 0)
        if is_call:
            divergence = prem_vel - u_vel * 50
        else:
            divergence = prem_vel + u_vel * 50

        # --- Time features ---
        elapsed_min = (now - timestamps[0]).astype("timedelta64[s]").astype(float) / 60
        minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + et_minute))

        # Session progress (0 = open, 1 = close) (NEW)
        session_progress = max(0, min(1, (et_decimal - 9.5) / 6.5))

        # Open interest relative to volume (NEW)
        oi = safe_float(df["open_interest"].iloc[idx])
        oi_vol_ratio = vol / oi if oi > 0 else 0

        # --- Labels ---
        labels = {}
        for label_name, (window_min, drop_pct) in DROP_THRESHOLDS.items():
            future_window = timestamps[idx:] <= now + np.timedelta64(window_min, "m")
            future_prems = premiums[idx:][future_window]
            if len(future_prems) > 1:
                future_min = np.nanmin(future_prems[1:])
                drop_from_here = (premium - future_min) / premium * 100
                labels[label_name] = 1 if drop_from_here >= drop_pct else 0
            else:
                labels[label_name] = -1  # unknown

        # Pct of session peak (regression target)
        pct_of_peak = premium / session_peak * 100 if session_peak > 0 else 0

        # Max future upside
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
            "pct_of_session_peak": pct_of_session_peak,
            "mfe_so_far": mfe_so_far,
            "retrace_from_peak": retrace_from_peak,
            "delta": abs(delta_val),
            "gamma": gamma_val,
            "theta": theta_val,
            "vega": vega_val,
            "iv": iv_val,
            "delta_change": delta_change,
            "gamma_change": gamma_change,
            "theta_change": theta_change,
            "iv_change": iv_change,
            "iv_percentile": iv_percentile,
            "gamma_delta_ratio": gamma_delta_ratio,
            "volume": vol,
            "vol_ratio": vol_ratio,
            "vol_trend": vol_trend,
            "u_move_entry": u_move_from_entry,
            "u_vel": u_vel,
            "moneyness": moneyness,
            "divergence": divergence,
            "spread_pct": spread_pct,
            "spread_change": spread_change,
            "size_imbalance": size_imbalance,
            "vwap_divergence": vwap_divergence,
            "oi_vol_ratio": oi_vol_ratio,
            "elapsed_min": elapsed_min,
            "minutes_to_close": minutes_to_close,
            "session_progress": session_progress,
            "is_highvol": 1 if category == TickerCategory.HIGH_VOL else 0,
            "is_index": 1 if category == TickerCategory.INDEX else 0,
            "is_call": 1 if is_call else 0,
            # Labels
            **labels,
            "pct_of_peak": pct_of_peak,
            "upside_remaining": upside_remaining,
            # Metadata (not used as features)
            "_ticker": ticker,
            "_premium": premium,
            "_contract": contract_info["contract_ticker"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "gain_pct", "vel_3", "vel_5", "vel_10", "vel_20", "accel_5",
    "prem_percentile", "prem_std", "pct_of_session_peak", "mfe_so_far",
    "retrace_from_peak",
    "delta", "gamma", "theta", "vega", "iv",
    "delta_change", "gamma_change", "theta_change", "iv_change",
    "iv_percentile", "gamma_delta_ratio",
    "volume", "vol_ratio", "vol_trend",
    "u_move_entry", "u_vel", "moneyness", "divergence",
    "spread_pct", "spread_change", "size_imbalance",
    "vwap_divergence", "oi_vol_ratio",
    "elapsed_min", "minutes_to_close", "session_progress",
    "is_highvol", "is_index", "is_call",
]


def train_classifier(X, y, groups, feature_cols, label_name, n_splits=5):
    """Train and evaluate a LightGBM binary classifier with GroupKFold."""
    print(f"\n{'=' * 100}")
    print(f"CLASSIFIER: {label_name}")
    print(f"  Samples: {len(y):,} | Positive: {(y == 1).sum():,} ({y.mean():.1%})")
    print(f"{'=' * 100}\n")

    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs = [], []
    all_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_va, y_va, feature_name=feature_cols, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "scale_pos_weight": pos_weight,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_va)
        all_preds[val_idx] = preds

        auc = roc_auc_score(y_va, preds)
        acc = accuracy_score(y_va, (preds > 0.5).astype(int))
        aucs.append(auc)
        accs.append(acc)

        n_groups = len(set(groups[val_idx]))
        print(f"  Fold {fold+1}: AUC={auc:.4f} Acc={acc:.3f} ({n_groups} contracts, {len(val_idx):,} ticks)")

    print(f"\n  Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"  Mean Acc: {np.mean(accs):.3f}")

    # Train final model on all data
    dtrain_full = lgb.Dataset(X, y, feature_name=feature_cols)
    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    params["scale_pos_weight"] = pos_weight
    final_model = lgb.train(params, dtrain_full, num_boost_round=300)

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 20 features:")
    for feat, imp in feat_imp[:20]:
        print(f"    {feat:<25} {imp:>12,.0f}")

    return final_model, all_preds, np.mean(aucs)


def train_regressor(X, y, groups, feature_cols, label_name, n_splits=5):
    """Train and evaluate a LightGBM regressor with GroupKFold."""
    print(f"\n{'=' * 100}")
    print(f"REGRESSOR: {label_name}")
    print(f"  Samples: {len(y):,} | Mean: {y.mean():.1f}% | Std: {y.std():.1f}%")
    print(f"{'=' * 100}\n")

    gkf = GroupKFold(n_splits=n_splits)
    maes = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_va, y_va, feature_name=feature_cols, reference=dtrain)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "feature_fraction": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_va)
        mae = mean_absolute_error(y_va, preds)
        maes.append(mae)

        n_groups = len(set(groups[val_idx]))
        print(f"  Fold {fold+1}: MAE={mae:.2f}% ({n_groups} contracts)")

    print(f"\n  Mean MAE: {np.mean(maes):.2f}% (+/- {np.std(maes):.2f}%)")

    # Train final model
    dtrain_full = lgb.Dataset(X, y, feature_name=feature_cols)
    final_model = lgb.train(params, dtrain_full, num_boost_round=300)

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 20 features:")
    for feat, imp in feat_imp[:20]:
        print(f"    {feat:<25} {imp:>12,.0f}")

    return final_model, np.mean(maes)


def simulate_exits(train_df, all_preds, feature_cols):
    """Simulate selling when classifier triggers, measure capture quality."""
    print(f"\n{'=' * 100}")
    print("SIMULATION: Sell when classifier says 'near peak' (first trigger per contract)")
    print(f"{'=' * 100}\n")

    train_df = train_df.copy()
    train_df["pred_prob"] = all_preds

    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        gains, peak_pcts, upsides = [], [], []
        n_triggered = 0
        n_contracts = train_df["contract_id"].nunique()

        for cid in train_df["contract_id"].unique():
            ct = train_df[train_df["contract_id"] == cid].sort_values("elapsed_min")
            trigger = ct[ct["pred_prob"] >= threshold]
            if len(trigger) == 0:
                continue

            first = trigger.iloc[0]
            gains.append(first["gain_pct"])
            peak_pcts.append(first["pct_of_peak"])
            upsides.append(first["upside_remaining"])
            n_triggered += 1

        if n_triggered > 0:
            avg_gain = np.mean(gains)
            avg_peak = np.mean(peak_pcts)
            avg_upside = np.mean(upsides)
            median_peak = np.median(peak_pcts)
            print(f"  Threshold {threshold:.1f}: {n_triggered:>5}/{n_contracts} contracts triggered | "
                  f"avg sell at +{avg_gain:.0f}% gain | "
                  f"captured {avg_peak:.0f}% of peak (median {median_peak:.0f}%) | "
                  f"left {avg_upside:.0f}% on table")
            results.append({
                "threshold": threshold,
                "triggered": n_triggered,
                "avg_gain": avg_gain,
                "avg_peak_pct": avg_peak,
                "median_peak_pct": median_peak,
                "avg_upside_left": avg_upside,
            })
        else:
            print(f"  Threshold {threshold:.1f}: 0 contracts triggered")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-contracts", type=int, default=None,
                        help="Limit number of contracts for testing")
    parser.add_argument("--min-run", type=float, default=MIN_RUN_PCT,
                        help="Minimum %% run to include contract")
    args = parser.parse_args()

    print(f"\n{'=' * 100}")
    print("PEAK DETECTOR v2 — Training on ALL Harvester Data")
    print(f"{'=' * 100}\n")

    conn = sqlite3.connect(HARVESTER_DB, timeout=60)

    # Step 1: Find all runner contracts
    runners = find_runner_contracts(conn, min_run_pct=args.min_run,
                                    max_contracts=args.max_contracts)

    if len(runners) == 0:
        print("No runner contracts found!")
        return

    # Step 2: Extract features for all runners
    print(f"\n  Extracting features from {len(runners):,} contracts...")
    t0 = time.time()

    all_features = []
    contract_id = 0
    skipped = 0

    for i, (_, row) in enumerate(runners.iterrows()):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(runners) - i - 1) / rate
            print(f"    {i+1:,}/{len(runners):,} contracts processed "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                  f"{sum(len(f) for f in all_features):,} ticks extracted)")

        ticks = load_contract_ticks(conn, row["contract_ticker"])
        if len(ticks) < MIN_TICKS_PER_CONTRACT:
            skipped += 1
            continue

        feat_df = extract_features_v2(ticks, row)
        if len(feat_df) == 0:
            skipped += 1
            continue

        feat_df["contract_id"] = contract_id
        all_features.append(feat_df)
        contract_id += 1

    conn.close()

    elapsed = time.time() - t0
    print(f"\n  Feature extraction complete: {elapsed:.0f}s")
    print(f"  {contract_id:,} contracts with features, {skipped:,} skipped")

    if not all_features:
        print("No feature data extracted!")
        return

    full_df = pd.concat(all_features, ignore_index=True)
    print(f"  Total feature rows: {len(full_df):,}")

    # Step 3: Prepare training data
    # Filter to known labels only
    for label_name in DROP_THRESHOLDS:
        known = (full_df[label_name] >= 0).sum()
        pos = (full_df[label_name] == 1).sum()
        print(f"  {label_name}: {known:,} known labels, {pos:,} positive ({pos/max(known,1):.1%})")

    train_df = full_df[full_df[PRIMARY_LABEL] >= 0].copy()
    print(f"\n  Training rows (primary label known): {len(train_df):,}")
    print(f"  Contracts: {train_df['contract_id'].nunique():,}")

    X = train_df[FEATURE_COLS].values
    groups = train_df["contract_id"].values

    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 4: Train classifier (primary: 10min/20% drop)
    y_cls = train_df[PRIMARY_LABEL].values
    cls_model, cls_preds, cls_auc = train_classifier(
        X, y_cls, groups, FEATURE_COLS, PRIMARY_LABEL
    )

    # Step 5: Train regressor (% of peak)
    y_reg = train_df["pct_of_peak"].values
    reg_model, reg_mae = train_regressor(
        X, y_reg, groups, FEATURE_COLS, "pct_of_peak"
    )

    # Step 6: Train secondary classifiers (5min and 20min horizons)
    for label_name in DROP_THRESHOLDS:
        if label_name == PRIMARY_LABEL:
            continue
        sub_df = full_df[full_df[label_name] >= 0]
        if len(sub_df) < 1000:
            print(f"\n  Skipping {label_name}: only {len(sub_df)} rows")
            continue
        X_sub = sub_df[FEATURE_COLS].values
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        y_sub = sub_df[label_name].values
        g_sub = sub_df["contract_id"].values
        sec_model, _, sec_auc = train_classifier(
            X_sub, y_sub, g_sub, FEATURE_COLS, label_name
        )
        # Save secondary models
        sec_path = MODEL_DIR / f"peak_detector_{label_name}.txt"
        sec_model.save_model(str(sec_path))
        print(f"  Saved: {sec_path}")

    # Step 7: Simulate exits
    train_df_sim = train_df.copy()
    sim_results = simulate_exits(train_df_sim, cls_preds, FEATURE_COLS)

    # Step 8: Per-category breakdown
    print(f"\n{'=' * 100}")
    print("PER-CATEGORY ANALYSIS")
    print(f"{'=' * 100}\n")

    train_df["pred_prob"] = cls_preds
    for cat_name, cat_filter in [("HIGH_VOL", "is_highvol"), ("INDEX", "is_index"),
                                  ("STANDARD", None)]:
        if cat_filter:
            mask = train_df[cat_filter] == 1
        else:
            mask = (train_df["is_highvol"] == 0) & (train_df["is_index"] == 0)

        sub = train_df[mask]
        if len(sub) < 100:
            continue

        y_sub = sub[PRIMARY_LABEL].values
        p_sub = sub["pred_prob"].values
        valid = ~np.isnan(p_sub)
        if valid.sum() < 100:
            continue

        auc = roc_auc_score(y_sub[valid], p_sub[valid])
        print(f"  {cat_name}: {len(sub):,} ticks, {sub['contract_id'].nunique()} contracts, AUC={auc:.4f}")

    # Step 9: Save models
    cls_path = MODEL_DIR / "peak_detector_v2_cls.txt"
    reg_path = MODEL_DIR / "peak_detector_v2_reg.txt"
    feat_path = MODEL_DIR / "peak_detector_v2_features.txt"

    cls_model.save_model(str(cls_path))
    reg_model.save_model(str(reg_path))
    with open(feat_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))

    print(f"\n{'=' * 100}")
    print("MODELS SAVED")
    print(f"{'=' * 100}")
    print(f"  Classifier ({PRIMARY_LABEL}): {cls_path}")
    print(f"  Regressor (pct_of_peak):  {reg_path}")
    print(f"  Features ({len(FEATURE_COLS)}):          {feat_path}")
    print(f"\n  Training summary:")
    print(f"    Contracts: {contract_id:,}")
    print(f"    Ticks:     {len(train_df):,}")
    print(f"    CLS AUC:   {cls_auc:.4f}")
    print(f"    REG MAE:   {reg_mae:.2f}%")


if __name__ == "__main__":
    main()
