"""Peak detector v3 — signal-matched training data from harvester.

Key difference from v2: instead of using every contract's full session,
we find "signal-like entry points" in harvester data:
  1. Filter to our signal tickers only (SPY, QQQ, TSLA, NVDA, etc.)
  2. Filter to signal-like premiums ($0.50-$10)
  3. Find momentum entry points: ticks where premium starts rising
     after a trough (simulating when a signal would fire)
  4. Only train on the run FROM that entry point forward
  5. Label "peak" = the highest premium from entry, not session peak
  6. Temporal split: train on older data, test on recent data

This produces a model that sees the world the same way our bot does:
"I just entered at this premium, the trade is running, when should I
lock profits?"

Usage:
    python scripts/train_peak_detector_v3.py
    python scripts/train_peak_detector_v3.py --max-contracts 500
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

# Signal-matching filters
SIGNAL_TICKERS = [
    "SPY", "QQQ", "IWM",  # index
    "TSLA", "NVDA", "AAPL", "AMZN", "META", "GOOGL", "MSFT",  # mega cap
    "AVGO", "AMD", "MSTR", "PLTR", "MU",  # high vol
]
PREMIUM_MIN = 0.30   # signal premiums are $0.50-$10, be slightly wider
PREMIUM_MAX = 15.0
ENTRY_HOUR_MIN = 9    # signals come in 9:30-14:00 ET
ENTRY_HOUR_MAX = 14

# Training params
DROP_WINDOW_MINUTES = 10
DROP_THRESHOLD_PCT = 20
MIN_RUN_FROM_ENTRY = 20      # entry must lead to at least 20% gain
MIN_TICKS_AFTER_ENTRY = 20   # need enough ticks after entry
MOMENTUM_LOOKBACK = 5        # ticks to look back for momentum detection
MAX_ENTRIES_PER_CONTRACT = 3  # multiple entry points per contract

# Temporal split
TEMPORAL_SPLIT_DATE = "2026-05-01"  # train on before, test on after


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def find_signal_like_contracts(conn, max_contracts=None):
    """Find contracts on our tickers with signal-like characteristics."""
    print("  Scanning for signal-like contracts...")
    t0 = time.time()

    ticker_list = ",".join(f"'{t}'" for t in SIGNAL_TICKERS)

    query = f"""
        SELECT
            s.contract_ticker,
            c.underlying,
            c.option_type,
            c.strike,
            c.expiry_date,
            COUNT(*) as ticks,
            MIN(s.captured_at) as first_ts,
            MAX(s.captured_at) as last_ts
        FROM harvest_snapshots s
        JOIN harvest_contracts c ON s.contract_ticker = c.contract_ticker
        WHERE c.underlying IN ({ticker_list})
          AND s.midpoint BETWEEN {PREMIUM_MIN} AND {PREMIUM_MAX}
          AND s.delta IS NOT NULL AND s.delta != 0
        GROUP BY s.contract_ticker
        HAVING ticks >= 30
        ORDER BY first_ts
    """
    if max_contracts:
        query += f" LIMIT {max_contracts}"

    df = pd.read_sql_query(query, conn)
    elapsed = time.time() - t0
    print(f"  Found {len(df):,} signal-like contracts in {elapsed:.1f}s")

    df["category"] = df["underlying"].apply(lambda t: categorize_ticker(t).value)
    for cat, grp in df.groupby("category"):
        print(f"    {cat}: {len(grp):,} contracts")

    df["date"] = df["first_ts"].str[:10]
    return df


def load_contract_ticks(conn, contract_ticker):
    """Load all snapshots for a contract."""
    query = """
        SELECT
            captured_at as ts,
            midpoint as premium,
            underlying_price,
            bid, ask, bid_size, ask_size,
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


def find_entry_points(premiums, timestamps):
    """Find signal-like entry points: where premium starts a momentum run.

    Looks for ticks where:
      - Premium has been flat/declining (trough)
      - Next N ticks show acceleration (premium rising)
      - Subsequent run reaches at least MIN_RUN_FROM_ENTRY%

    Returns list of entry indices.
    """
    entries = []
    n = len(premiums)
    i = MOMENTUM_LOOKBACK

    while i < n - MIN_TICKS_AFTER_ENTRY and len(entries) < MAX_ENTRIES_PER_CONTRACT:
        # Check if this is a local trough + momentum start
        prem = premiums[i]
        if prem <= 0 or np.isnan(prem):
            i += 1
            continue

        # Time filter: entry should be during signal hours (ET)
        ts = pd.Timestamp(timestamps[i])
        et_hour = ts.hour - 4
        if et_hour < 0:
            et_hour += 24
        if et_hour < ENTRY_HOUR_MIN or et_hour > ENTRY_HOUR_MAX:
            i += 1
            continue

        # Is this a trough? Previous ticks declining or flat
        prev_prems = premiums[i - MOMENTUM_LOOKBACK:i]
        if np.any(np.isnan(prev_prems)) or np.any(prev_prems <= 0):
            i += 1
            continue

        # Check: price is near recent low (within 10% of min)
        recent_min = np.min(prev_prems)
        if prem > recent_min * 1.10:  # not a trough, price already elevated
            i += 1
            continue

        # Check: next few ticks show momentum (rising)
        if i + 5 >= n:
            break
        next_prems = premiums[i:i + 5]
        if np.any(np.isnan(next_prems)):
            i += 5
            continue
        if next_prems[-1] <= prem * 1.03:  # need at least 3% rise in 5 ticks
            i += 1
            continue

        # Check: subsequent run reaches MIN_RUN_FROM_ENTRY%
        future_prems = premiums[i:]
        future_valid = future_prems[~np.isnan(future_prems)]
        if len(future_valid) < MIN_TICKS_AFTER_ENTRY:
            break

        peak_future = np.max(future_valid)
        run_pct = (peak_future - prem) / prem * 100
        if run_pct < MIN_RUN_FROM_ENTRY:
            i += 5
            continue

        entries.append(i)
        # Skip ahead past this entry's run to avoid overlapping entries
        # Find the peak, then skip to 10 ticks after
        peak_idx = i + np.argmax(future_valid[:200])  # look within 200 ticks
        i = peak_idx + 10

    return entries


def extract_features_from_entry(df, entry_idx, contract_info):
    """Extract features for ticks AFTER the entry point.

    The entry premium is the premium at entry_idx — this simulates
    our bot entering a trade at that price.
    """
    ticker = contract_info["underlying"]
    option_type = contract_info["option_type"]
    category = categorize_ticker(ticker)
    is_call = option_type == "call"
    strike = contract_info["strike"]

    premiums = df["premium"].values.astype(float)
    timestamps = pd.to_datetime(df["ts"], format="ISO8601").values
    n = len(premiums)

    entry_premium = premiums[entry_idx]
    if entry_premium <= 0 or np.isnan(entry_premium):
        return pd.DataFrame()

    # Peak premium from entry forward
    future_prems = premiums[entry_idx:]
    trade_peak = np.nanmax(future_prems)

    # Underlying at entry
    entry_underlying = safe_float(df["underlying_price"].iloc[entry_idx])

    rows = []
    # Sample ticks from entry forward (cap at 150 ticks per entry)
    end_idx = min(n, entry_idx + 500)
    ticks_available = end_idx - entry_idx
    if ticks_available > 150:
        indices = list(range(entry_idx + 5, entry_idx + 15))  # first 10 after entry
        indices += list(np.linspace(entry_idx + 15, end_idx - 1, 140, dtype=int))
        indices = sorted(set(indices))
    else:
        indices = list(range(entry_idx + 5, end_idx))

    for idx in indices:
        if idx >= n or idx < 10:
            continue

        premium = premiums[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        # Gain from ENTRY (not from session low)
        gain_pct = (premium - entry_premium) / entry_premium * 100
        if gain_pct < -50:  # skip ticks where we'd have stopped out
            continue

        now = timestamps[idx]
        now_dt = pd.Timestamp(now)
        et_hour = now_dt.hour - 4
        et_minute = now_dt.minute
        if et_hour < 0:
            et_hour += 24
        et_decimal = et_hour + et_minute / 60
        if et_decimal < 9.5 or et_decimal > 16.0:
            continue

        # --- Premium velocity from ENTRY perspective ---
        velocities = {}
        for lb in [3, 5, 10, 20]:
            if idx >= lb:
                prev_prem = premiums[idx - lb]
                velocities[f"vel_{lb}"] = (premium - prev_prem) / prev_prem * 100 if prev_prem > 0 else 0
            else:
                velocities[f"vel_{lb}"] = 0

        accel_5 = velocities["vel_3"] - velocities["vel_5"]

        # Premium percentile (local, from entry forward)
        window = min(30, idx - entry_idx)
        if window < 3:
            window = min(30, idx)
        recent = premiums[max(entry_idx, idx - window):idx + 1]
        recent_valid = recent[~np.isnan(recent)]
        if len(recent_valid) > 1:
            rng = recent_valid.max() - recent_valid.min()
            prem_percentile = (premium - recent_valid.min()) / rng if rng > 0 else 0.5
            prem_std = np.std(recent_valid) / premium if premium > 0 else 0
        else:
            prem_percentile = 0.5
            prem_std = 0

        # MFE from entry (max favorable excursion)
        entry_forward = premiums[entry_idx:idx + 1]
        mfe = np.nanmax(entry_forward)
        mfe_pct = (mfe - entry_premium) / entry_premium * 100

        # Retrace from local peak (from entry forward)
        retrace = (mfe - premium) / mfe * 100 if mfe > 0 else 0

        # Pct of peak from entry
        pct_of_run_peak = premium / mfe * 100 if mfe > 0 else 100

        # Greeks
        delta_val = safe_float(df["delta"].iloc[idx])
        gamma_val = safe_float(df["gamma"].iloc[idx])
        theta_val = safe_float(df["theta"].iloc[idx])
        vega_val = safe_float(df["vega"].iloc[idx])
        iv_val = safe_float(df["iv"].iloc[idx])

        # Greeks rate of change
        delta_prev = safe_float(df["delta"].iloc[idx - 5], delta_val) if idx >= 5 else delta_val
        gamma_prev = safe_float(df["gamma"].iloc[idx - 5], gamma_val) if idx >= 5 else gamma_val
        theta_prev = safe_float(df["theta"].iloc[idx - 5], theta_val) if idx >= 5 else theta_val
        iv_prev = safe_float(df["iv"].iloc[idx - 5], iv_val) if idx >= 5 else iv_val

        delta_change = delta_val - delta_prev
        gamma_change = gamma_val - gamma_prev
        theta_change = theta_val - theta_prev
        iv_change = iv_val - iv_prev

        # IV percentile over trade
        iv_start = max(entry_idx, idx - 50)
        iv_values = [safe_float(df["iv"].iloc[i]) for i in range(iv_start, idx + 1)]
        iv_values = [v for v in iv_values if v > 0]
        iv_percentile = sum(1 for v in iv_values if v <= iv_val) / len(iv_values) if iv_values else 0.5

        gamma_delta_ratio = gamma_val / abs(delta_val) if abs(delta_val) > 0.01 else 0

        # Volume
        vol = safe_float(df["volume"].iloc[idx])
        if idx >= 10:
            prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
            prev_vols = [v for v in prev_vols if v > 0]
            avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        if idx >= 20:
            vol_recent = [safe_float(df["volume"].iloc[i]) for i in range(idx - 5, idx + 1)]
            vol_older = [safe_float(df["volume"].iloc[i]) for i in range(idx - 20, idx - 10)]
            avg_recent = np.mean([v for v in vol_recent if v > 0] or [0])
            avg_older = np.mean([v for v in vol_older if v > 0] or [1])
            vol_trend = avg_recent / avg_older if avg_older > 0 else 1.0
        else:
            vol_trend = 1.0

        # Bid-ask
        bid = safe_float(df["bid"].iloc[idx], premium * 0.95)
        ask = safe_float(df["ask"].iloc[idx], premium * 1.05)
        spread_pct = (ask - bid) / premium * 100 if premium > 0 else 0

        if idx >= 5:
            prev_bid = safe_float(df["bid"].iloc[idx - 5], bid)
            prev_ask = safe_float(df["ask"].iloc[idx - 5], ask)
            prev_p = premiums[idx - 5]
            prev_spread = (prev_ask - prev_bid) / prev_p * 100 if prev_p > 0 else spread_pct
            spread_change = spread_pct - prev_spread
        else:
            spread_change = 0

        bid_size = safe_float(df["bid_size"].iloc[idx] if "bid_size" in df.columns else 1, 1)
        ask_size = safe_float(df["ask_size"].iloc[idx] if "ask_size" in df.columns else 1, 1)
        size_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0

        # VWAP
        vwap = safe_float(df["vwap"].iloc[idx] if "vwap" in df.columns else 0)
        vwap_divergence = (premium - vwap) / vwap * 100 if vwap > 0 else 0

        # OI
        oi = safe_float(df["open_interest"].iloc[idx] if "open_interest" in df.columns else 0)
        oi_vol_ratio = vol / oi if oi > 0 else 0

        # Underlying
        underlying = safe_float(df["underlying_price"].iloc[idx])
        u_move = (underlying - entry_underlying) / entry_underlying * 100 if entry_underlying > 0 else 0

        if idx >= 5:
            prev_u = safe_float(df["underlying_price"].iloc[idx - 5], underlying)
            u_vel = (underlying - prev_u) / prev_u * 100 if prev_u > 0 else 0
        else:
            u_vel = 0

        moneyness = 0
        if underlying > 0 and strike > 0:
            moneyness = ((underlying - strike) / strike * 100) if is_call else ((strike - underlying) / strike * 100)

        prem_vel = velocities.get("vel_5", 0)
        divergence = (prem_vel - u_vel * 50) if is_call else (prem_vel + u_vel * 50)

        # Time from ENTRY (not session start)
        elapsed_from_entry = (now - timestamps[entry_idx]).astype("timedelta64[s]").astype(float) / 60
        minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + et_minute))
        session_progress = max(0, min(1, (et_decimal - 9.5) / 6.5))

        # --- Labels (from ENTRY perspective) ---
        # Primary: will premium drop 20% from HERE within 10 min?
        future_window = timestamps[idx:] <= now + np.timedelta64(DROP_WINDOW_MINUTES, "m")
        future_prems = premiums[idx:][future_window]
        if len(future_prems) > 1:
            future_min = np.nanmin(future_prems[1:])
            drop_from_here = (premium - future_min) / premium * 100
            near_peak = 1 if drop_from_here >= DROP_THRESHOLD_PCT else 0
        else:
            near_peak = -1

        # Pct of trade peak (from entry forward)
        pct_of_peak = premium / trade_peak * 100 if trade_peak > 0 else 0

        # Upside remaining from here
        remaining_prems = premiums[idx:]
        future_max = np.nanmax(remaining_prems) if len(remaining_prems) > 0 else premium
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
            "mfe_pct": mfe_pct,
            "retrace_from_peak": retrace,
            "pct_of_run_peak": pct_of_run_peak,
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
            "u_move": u_move,
            "u_vel": u_vel,
            "moneyness": moneyness,
            "divergence": divergence,
            "spread_pct": spread_pct,
            "spread_change": spread_change,
            "size_imbalance": size_imbalance,
            "vwap_divergence": vwap_divergence,
            "oi_vol_ratio": oi_vol_ratio,
            "elapsed_from_entry": elapsed_from_entry,
            "minutes_to_close": minutes_to_close,
            "session_progress": session_progress,
            "is_highvol": 1 if category == TickerCategory.HIGH_VOL else 0,
            "is_index": 1 if category == TickerCategory.INDEX else 0,
            "is_call": 1 if is_call else 0,
            # Labels
            "near_peak": near_peak,
            "pct_of_peak": pct_of_peak,
            "upside_remaining": upside_remaining,
            # Meta
            "_ticker": ticker,
            "_premium": premium,
            "_entry_premium": entry_premium,
            "_date": contract_info.get("date", ""),
        }
        rows.append(row)

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "gain_pct", "vel_3", "vel_5", "vel_10", "vel_20", "accel_5",
    "prem_percentile", "prem_std", "mfe_pct", "retrace_from_peak",
    "pct_of_run_peak",
    "delta", "gamma", "theta", "vega", "iv",
    "delta_change", "gamma_change", "theta_change", "iv_change",
    "iv_percentile", "gamma_delta_ratio",
    "volume", "vol_ratio", "vol_trend",
    "u_move", "u_vel", "moneyness", "divergence",
    "spread_pct", "spread_change", "size_imbalance",
    "vwap_divergence", "oi_vol_ratio",
    "elapsed_from_entry", "minutes_to_close", "session_progress",
    "is_highvol", "is_index", "is_call",
]


def train_and_eval(X_train, y_train, groups_train, X_test, y_test, groups_test,
                    feature_cols, label_name, is_classifier=True):
    """Train on train set, evaluate on held-out test set."""
    print(f"\n{'=' * 100}")
    if is_classifier:
        print(f"CLASSIFIER: {label_name}")
        pos_rate_train = y_train.mean()
        pos_rate_test = y_test.mean()
        print(f"  Train: {len(y_train):,} ticks, {(y_train==1).sum():,} pos ({pos_rate_train:.1%})")
        print(f"  Test:  {len(y_test):,} ticks, {(y_test==1).sum():,} pos ({pos_rate_test:.1%})")
    else:
        print(f"REGRESSOR: {label_name}")
        print(f"  Train: {len(y_train):,} ticks, mean={y_train.mean():.1f}%")
        print(f"  Test:  {len(y_test):,} ticks, mean={y_test.mean():.1f}%")
    print(f"{'=' * 100}")

    # Also do GroupKFold CV on train for internal validation
    n_unique_groups = len(set(groups_train))
    n_splits = min(5, n_unique_groups)
    if n_splits < 2:
        print("  Not enough groups for CV, training on all data")
        n_splits = 2

    gkf = GroupKFold(n_splits=n_splits)

    if is_classifier:
        pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 100, "feature_fraction": 0.7,
            "bagging_fraction": 0.7, "bagging_freq": 5,
            "scale_pos_weight": pos_weight,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
        }
    else:
        params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 100, "feature_fraction": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
        }

    # CV on train
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]

        dtrain = lgb.Dataset(Xtr, ytr, feature_name=feature_cols)
        dval = lgb.Dataset(Xva, yva, feature_name=feature_cols, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(Xva)
        if is_classifier:
            score = roc_auc_score(yva, preds)
            cv_scores.append(score)
            print(f"    CV Fold {fold+1}: AUC={score:.4f}")
        else:
            score = mean_absolute_error(yva, preds)
            cv_scores.append(score)
            print(f"    CV Fold {fold+1}: MAE={score:.2f}%")

    metric_name = "AUC" if is_classifier else "MAE"
    print(f"  CV Mean {metric_name}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Train final model on ALL train data
    dtrain_full = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    final_model = lgb.train(params, dtrain_full, num_boost_round=400)

    # Evaluate on HELD-OUT test set
    test_preds = final_model.predict(X_test)
    if is_classifier:
        test_auc = roc_auc_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
        print(f"\n  ** TEST SET: AUC={test_auc:.4f}, Acc={test_acc:.3f} **")
    else:
        test_mae = mean_absolute_error(y_test, test_preds)
        print(f"\n  ** TEST SET: MAE={test_mae:.2f}% **")

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 features:")
    for feat, imp in feat_imp[:15]:
        print(f"    {feat:<25} {imp:>12,.0f}")

    return final_model, test_preds


def simulate_on_test(test_df, test_preds):
    """Simulate selling on first ML trigger for test set trades."""
    print(f"\n{'=' * 100}")
    print("SIMULATION ON TEST SET (out-of-sample)")
    print(f"{'=' * 100}")

    test_df = test_df.copy()
    test_df["pred_prob"] = test_preds

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gains, peak_pcts, upsides = [], [], []
        n_triggered = 0
        n_trades = test_df["trade_id"].nunique()

        for tid in test_df["trade_id"].unique():
            ct = test_df[test_df["trade_id"] == tid].sort_values("elapsed_from_entry")
            # Only consider ticks where gain is positive (we're in profit)
            profitable = ct[ct["gain_pct"] > 10]
            trigger = profitable[profitable["pred_prob"] >= threshold]
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
            # How many were RIGHT (upside < 10%)?
            right = sum(1 for u in upsides if u < 10)
            print(f"  Threshold {threshold:.1f}: {n_triggered:>4}/{n_trades} trades | "
                  f"sell at +{avg_gain:.0f}% | "
                  f"{avg_peak:.0f}% of peak (med {median_peak:.0f}%) | "
                  f"{avg_upside:.0f}% left | "
                  f"correct: {right}/{n_triggered} ({right/n_triggered*100:.0f}%)")
        else:
            print(f"  Threshold {threshold:.1f}: 0 trades triggered")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-contracts", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{'=' * 100}")
    print("PEAK DETECTOR v3 — Signal-Matched Training from Harvester")
    print(f"  Tickers: {', '.join(SIGNAL_TICKERS)}")
    print(f"  Premium filter: ${PREMIUM_MIN}-${PREMIUM_MAX}")
    print(f"  Entry hours: {ENTRY_HOUR_MIN}-{ENTRY_HOUR_MAX} ET")
    print(f"  Temporal split: train < {TEMPORAL_SPLIT_DATE}, test >= {TEMPORAL_SPLIT_DATE}")
    print(f"{'=' * 100}\n")

    conn = sqlite3.connect(HARVESTER_DB, timeout=60)

    # Step 1: Find signal-like contracts
    contracts = find_signal_like_contracts(conn, max_contracts=args.max_contracts)

    if len(contracts) == 0:
        print("No contracts found!")
        return

    # Step 2: Extract features with signal-like entry points
    print(f"\n  Extracting features with momentum entry detection...")
    t0 = time.time()

    all_features = []
    trade_id = 0
    n_entries_found = 0
    n_contracts_with_entries = 0

    for i, (_, row) in enumerate(contracts.iterrows()):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(contracts) - i - 1) / rate
            total_ticks = sum(len(f) for f in all_features)
            print(f"    {i+1:,}/{len(contracts):,} contracts | "
                  f"{n_entries_found} entries found | "
                  f"{total_ticks:,} ticks | "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        ticks = load_contract_ticks(conn, row["contract_ticker"])
        if len(ticks) < 30:
            continue

        premiums = ticks["premium"].values.astype(float)
        timestamps = pd.to_datetime(ticks["ts"], format="ISO8601").values

        # Find entry points
        entries = find_entry_points(premiums, timestamps)
        if not entries:
            continue

        n_contracts_with_entries += 1
        contract_date = row.get("date", row["first_ts"][:10])

        for entry_idx in entries:
            feat_df = extract_features_from_entry(ticks, entry_idx, row)
            if len(feat_df) == 0:
                continue

            feat_df["trade_id"] = trade_id
            feat_df["_date"] = contract_date
            all_features.append(feat_df)
            trade_id += 1
            n_entries_found += 1

    conn.close()

    elapsed = time.time() - t0
    print(f"\n  Extraction complete: {elapsed:.0f}s")
    print(f"  Contracts with entries: {n_contracts_with_entries:,}")
    print(f"  Entry points found: {n_entries_found:,}")
    print(f"  Trades (entries): {trade_id:,}")

    if not all_features:
        print("No features extracted!")
        return

    full_df = pd.concat(all_features, ignore_index=True)
    print(f"  Total ticks: {len(full_df):,}")

    # Filter to known labels
    train_ready = full_df[full_df["near_peak"] >= 0].copy()
    print(f"  Labeled ticks: {len(train_ready):,}")
    print(f"  Positive rate: {train_ready['near_peak'].mean():.1%}")
    print(f"  Trades: {train_ready['trade_id'].nunique()}")

    # Step 3: Temporal split
    train_mask = train_ready["_date"] < TEMPORAL_SPLIT_DATE
    test_mask = train_ready["_date"] >= TEMPORAL_SPLIT_DATE

    train_set = train_ready[train_mask]
    test_set = train_ready[test_mask]

    print(f"\n  Temporal split:")
    print(f"    Train (< {TEMPORAL_SPLIT_DATE}): {len(train_set):,} ticks, "
          f"{train_set['trade_id'].nunique()} trades, "
          f"{train_set['near_peak'].mean():.1%} positive")
    print(f"    Test  (>= {TEMPORAL_SPLIT_DATE}): {len(test_set):,} ticks, "
          f"{test_set['trade_id'].nunique()} trades, "
          f"{test_set['near_peak'].mean():.1%} positive")

    if len(train_set) < 100 or len(test_set) < 100:
        print("  Not enough data for temporal split, falling back to GroupKFold on all data")
        train_set = train_ready
        test_set = train_ready  # will still use CV for validation

    X_train = np.nan_to_num(train_set[FEATURE_COLS].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_cls = train_set["near_peak"].values
    y_train_reg = train_set["pct_of_peak"].values
    groups_train = train_set["trade_id"].values

    X_test = np.nan_to_num(test_set[FEATURE_COLS].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_cls = test_set["near_peak"].values
    y_test_reg = test_set["pct_of_peak"].values
    groups_test = test_set["trade_id"].values

    # Step 4: Train classifier
    cls_model, cls_test_preds = train_and_eval(
        X_train, y_train_cls, groups_train,
        X_test, y_test_cls, groups_test,
        FEATURE_COLS, "near_peak (20% drop in 10min)", is_classifier=True
    )

    # Step 5: Train regressor
    reg_model, reg_test_preds = train_and_eval(
        X_train, y_train_reg, groups_train,
        X_test, y_test_reg, groups_test,
        FEATURE_COLS, "pct_of_peak", is_classifier=False
    )

    # Step 6: Simulate on test set
    simulate_on_test(test_set, cls_test_preds)

    # Step 7: Accuracy analysis on test set
    print(f"\n{'=' * 100}")
    print("ACCURACY ANALYSIS ON TEST SET")
    print(f"{'=' * 100}")

    test_with_preds = test_set.copy()
    test_with_preds["pred_prob"] = cls_test_preds

    # For each trade, find first trigger and measure accuracy
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        results = []
        for tid in test_with_preds["trade_id"].unique():
            ct = test_with_preds[test_with_preds["trade_id"] == tid].sort_values("elapsed_from_entry")
            profitable = ct[ct["gain_pct"] > 10]
            trigger = profitable[profitable["pred_prob"] >= threshold]
            if len(trigger) == 0:
                continue
            first = trigger.iloc[0]
            results.append({
                "ticker": first["_ticker"],
                "gain": first["gain_pct"],
                "upside": first["upside_remaining"],
                "pct_peak": first["pct_of_peak"],
            })

        if results:
            rdf = pd.DataFrame(results)
            right = (rdf["upside"] < 15).sum()
            mostly_right = (rdf["upside"] < 30).sum()
            wrong = (rdf["upside"] > 50).sum()
            print(f"\n  Threshold {threshold:.1f}: {len(rdf)} triggers")
            print(f"    Correct (upside < 15%): {right}/{len(rdf)} ({right/len(rdf)*100:.0f}%)")
            print(f"    Close   (upside < 30%): {mostly_right}/{len(rdf)} ({mostly_right/len(rdf)*100:.0f}%)")
            print(f"    Wrong   (upside > 50%): {wrong}/{len(rdf)} ({wrong/len(rdf)*100:.0f}%)")
            print(f"    Avg gain at trigger: +{rdf['gain'].mean():.0f}%")
            print(f"    Avg upside left: +{rdf['upside'].mean():.0f}%")
            print(f"    Avg % of peak: {rdf['pct_peak'].mean():.0f}%")

    # Step 8: Save models
    cls_path = MODEL_DIR / "peak_detector_v3_cls.txt"
    reg_path = MODEL_DIR / "peak_detector_v3_reg.txt"
    feat_path = MODEL_DIR / "peak_detector_v3_features.txt"

    cls_model.save_model(str(cls_path))
    reg_model.save_model(str(reg_path))
    with open(feat_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))

    print(f"\n{'=' * 100}")
    print("MODELS SAVED")
    print(f"{'=' * 100}")
    print(f"  Classifier: {cls_path}")
    print(f"  Regressor:  {reg_path}")
    print(f"  Features:   {feat_path}")


if __name__ == "__main__":
    main()
