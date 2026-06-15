"""Afternoon pattern entry model: learn 1:00 PM - 3:00 PM ET setups.

Trains TWO separate models:
  - pattern_afternoon_call.txt  (CALL dip+bounce patterns)
  - pattern_afternoon_put.txt   (PUT spike+dump patterns — inverted logic)

Does NOT overwrite morning or midday models.

Usage:
    python scripts/train_pattern_afternoon.py
    python scripts/train_pattern_afternoon.py --calls-only
    python scripts/train_pattern_afternoon.py --puts-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

N_WORKERS = min(os.cpu_count() or 4, 16)

# Afternoon window: minutes 210-330 after open (1:00 PM - 3:00 PM ET)
AFTERNOON_START_MIN = 210
AFTERNOON_END_MIN = 330
MIN_GAIN_CALL = 12.0     # Lower than morning — afternoon moves smaller but faster
MIN_GAIN_PUT = 12.0      # PUT: premium gain when underlying drops
LOW_PROXIMITY_PCT = 5.0
MIN_CANDLES = 30


# ── Feature Computation ───────────────────────────────────────────────────

def compute_afternoon_features(closes, volumes, ivs, deltas, thetas, underlyings,
                                bids, asks, idx, reference_price, option_type="call"):
    """Compute features for afternoon model at position idx.

    reference_price is the premium at minute 210 (1:00 PM).
    """
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)

    pre5 = closes[w5_start:idx]
    pre10 = closes[w10_start:idx]
    pre5_v = volumes[w5_start:idx]
    pre5_iv = ivs[w5_start:idx]
    pre5_u = underlyings[w5_start:idx]

    valid5 = pre5[~np.isnan(pre5)]
    valid10 = pre10[~np.isnan(pre10)]
    valid5_v = pre5_v[~np.isnan(pre5_v)]
    valid5_iv = pre5_iv[~np.isnan(pre5_iv)]
    valid5_u = pre5_u[~np.isnan(pre5_u)]

    if len(valid5) < 3 or valid5[0] <= 0:
        return None

    current = closes[idx]
    if np.isnan(current) or current <= 0:
        return None

    f = {}

    # Premium trajectory
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0

    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    # Volume
    f["volume_avg_5"] = float(np.mean(valid5_v)) if len(valid5_v) > 0 else 0
    w20_start = max(0, idx - 20)
    vol20 = volumes[w20_start:idx]
    vol20_valid = vol20[~np.isnan(vol20)]
    avg20 = float(np.mean(vol20_valid)) if len(vol20_valid) > 0 else 1
    f["volume_ratio"] = f["volume_avg_5"] / max(avg20, 1)

    if len(valid5_v) >= 3:
        f["volume_trend"] = float(valid5_v[-1] / max(valid5_v[0], 1))
    else:
        f["volume_trend"] = 1.0

    # IV
    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    # Underlying
    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    # Drop from reference (1:00 PM price)
    f["drop_from_open"] = (current / reference_price - 1) * 100 if reference_price > 0 else 0

    # Spread
    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0

    # Greeks
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0

    # Time
    f["minutes_since_open"] = idx
    f["minutes_to_close"] = max(0, 390 - idx)  # 3:30 PM close area

    # Premium level
    f["premium"] = float(current)

    # ── Context features: morning + midday session history ──
    # Morning session (0-90 min)
    if closes[0] > 0 and not np.isnan(closes[0]):
        morning_slice = closes[:90]
        valid_morning = morning_slice[~np.isnan(morning_slice) & (morning_slice > 0)]
        if len(valid_morning) >= 10:
            f["morning_range_pct"] = (float(np.max(valid_morning)) - float(np.min(valid_morning))) / float(np.max(valid_morning)) * 100
            f["morning_return_pct"] = (float(valid_morning[-1]) / float(valid_morning[0]) - 1) * 100
        else:
            f["morning_range_pct"] = 0
            f["morning_return_pct"] = 0
    else:
        f["morning_range_pct"] = 0
        f["morning_return_pct"] = 0

    # Full session low relative to current
    session_slice = closes[:idx]
    valid_session = session_slice[~np.isnan(session_slice) & (session_slice > 0)]
    if len(valid_session) >= 10:
        f["prem_vs_session_low"] = (current / float(np.min(valid_session)) - 1) * 100
        f["prem_vs_session_high"] = (current / float(np.max(valid_session)) - 1) * 100
    else:
        f["prem_vs_session_low"] = 0
        f["prem_vs_session_high"] = 0

    # Underlying morning+midday return
    u_morning = underlyings[:210]
    valid_u = u_morning[~np.isnan(u_morning) & (u_morning > 0)]
    if len(valid_u) >= 10:
        f["und_session_return"] = (float(valid_u[-1]) / float(valid_u[0]) - 1) * 100
    else:
        f["und_session_return"] = 0

    # Underlying current direction (last 30 min)
    u_recent = underlyings[max(0, idx - 30):idx + 1]
    valid_u_recent = u_recent[~np.isnan(u_recent) & (u_recent > 0)]
    if len(valid_u_recent) >= 5:
        f["und_30min_return"] = (float(valid_u_recent[-1]) / float(valid_u_recent[0]) - 1) * 100
    else:
        f["und_30min_return"] = 0

    # Theta decay acceleration (afternoon specific — theta burns faster)
    if idx >= 210:
        midday_theta = thetas[max(0, 180):210]
        afternoon_theta = thetas[max(0, idx - 10):idx]
        valid_mid_t = midday_theta[~np.isnan(midday_theta)]
        valid_aft_t = afternoon_theta[~np.isnan(afternoon_theta)]
        if len(valid_mid_t) > 0 and len(valid_aft_t) > 0:
            f["theta_accel"] = float(np.mean(valid_aft_t) - np.mean(valid_mid_t))
        else:
            f["theta_accel"] = 0
    else:
        f["theta_accel"] = 0

    return f


# ── Worker Functions ──────────────────────────────────────────────────────

def _worker_call(item):
    """Process one ticker-day for CALL afternoon patterns (dip → bounce)."""
    return _worker_generic(item, option_type="call", min_gain=MIN_GAIN_CALL)


def _worker_put(item):
    """Process one ticker-day for PUT afternoon patterns (spike → dump)."""
    return _worker_generic(item, option_type="put", min_gain=MIN_GAIN_PUT)


def _worker_generic(item, option_type, min_gain):
    """Process one ticker-day: find afternoon low, label candles, extract features."""
    ticker = item["ticker"]
    dt = item["date"]

    closes = np.array(item["closes"], dtype=np.float64)
    volumes = np.array(item["volumes"], dtype=np.float64)
    ivs = np.array(item["ivs"], dtype=np.float64)
    deltas = np.array(item["deltas"], dtype=np.float64)
    thetas = np.array(item["thetas"], dtype=np.float64)
    underlyings = np.array(item["underlyings"], dtype=np.float64)
    bids = np.array(item["bids"], dtype=np.float64)
    asks = np.array(item["asks"], dtype=np.float64)

    n = len(closes)
    if n < AFTERNOON_END_MIN:
        return []

    # Reference price: premium at start of afternoon window
    reference_price = 0
    for c in closes[max(0, AFTERNOON_START_MIN - 3):AFTERNOON_START_MIN + 3]:
        if not np.isnan(c) and c > 0:
            reference_price = c
            break
    if reference_price <= 0:
        return []

    # Find the afternoon low (minutes 210-330)
    afternoon_slice = closes[AFTERNOON_START_MIN:AFTERNOON_END_MIN]
    valid_mask = ~np.isnan(afternoon_slice) & (afternoon_slice > 0)
    if valid_mask.sum() < 10:
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_values = afternoon_slice[valid_mask]
    low_rel = np.argmin(valid_values)
    low_idx_abs = AFTERNOON_START_MIN + valid_indices[low_rel]
    low_price = valid_values[low_rel]

    if low_price <= 0:
        return []

    # Compute gain from low to subsequent peak (rest of day)
    future = closes[low_idx_abs:]
    future_valid = future[~np.isnan(future) & (future > 0)]
    if len(future_valid) < 3:
        return []
    peak_after = np.nanmax(future_valid)
    gain_from_low = (peak_after / low_price - 1) * 100

    has_move = gain_from_low >= min_gain

    # Label candles in the afternoon window
    rows = []
    for i in range(AFTERNOON_START_MIN + 5, min(AFTERNOON_END_MIN, n)):
        if np.isnan(closes[i]) or closes[i] <= 0:
            continue

        if has_move and low_price > 0:
            dist_from_low = (closes[i] / low_price - 1) * 100
            is_near_low = dist_from_low < LOW_PROXIMITY_PCT
        else:
            is_near_low = False

        features = compute_afternoon_features(
            closes, volumes, ivs, deltas, thetas, underlyings,
            bids, asks, i, reference_price, option_type,
        )
        if features is None:
            continue

        features["label"] = 1 if is_near_low else 0
        features["ticker"] = ticker
        features["date"] = dt
        features["gain_from_low"] = round(gain_from_low, 1)
        rows.append(features)

    return rows


# ── Data Preloading ────────────────────────────────────────────────────────

def preload_ticker_data(ticker: str, option_type: str = "CALL") -> list[dict]:
    """Load all day data for a ticker into memory for parallel processing."""
    right = option_type.upper()
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? AND right=? ORDER BY 1",
        (ticker, right),
    ).fetchall()]

    items = []
    for dt in dates:
        # ATM strike — for PUTs, find ATM put strike
        atm = conn.execute("""
            SELECT oohlc.strike FROM option_ohlc oohlc
            JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right=?
                AND og.underlying_price > 0
            GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
        """, (ticker, dt, right)).fetchone()
        if not atm:
            continue
        strike = atm[0]

        rows = conn.execute("""
            SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                   COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                   COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                   oohlc.volume
            FROM option_ohlc oohlc
            LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
            LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right=? AND oohlc.strike=?
            ORDER BY oohlc.timestamp
        """, (ticker, dt, right, strike)).fetchall()

        if len(rows) < AFTERNOON_END_MIN:
            continue

        items.append({
            "ticker": ticker,
            "date": dt,
            "closes": [float(r[0]) if r[0] else float("nan") for r in rows],
            "underlyings": [float(r[1]) if r[1] else float("nan") for r in rows],
            "ivs": [float(r[2]) if r[2] else float("nan") for r in rows],
            "bids": [float(r[3]) if r[3] else 0 for r in rows],
            "asks": [float(r[4]) if r[4] else 0 for r in rows],
            "deltas": [float(r[5]) if r[5] else float("nan") for r in rows],
            "thetas": [float(r[6]) if r[6] else float("nan") for r in rows],
            "volumes": [float(r[7]) if r[7] else 0 for r in rows],
        })

    conn.close()
    return items


# ── Training ───────────────────────────────────────────────────────────────

def train_model(tickers: list[str], option_type: str):
    """Train afternoon model for given option type."""
    label = option_type.upper()
    worker_fn = _worker_call if option_type == "call" else _worker_put

    print(f"\n{'=' * 70}")
    print(f"AFTERNOON PATTERN ENTRY MODEL — {label}")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Window: minutes {AFTERNOON_START_MIN}-{AFTERNOON_END_MIN} (1:00 PM - 3:00 PM ET)")
    print(f"  Min gain: {MIN_GAIN_CALL if option_type == 'call' else MIN_GAIN_PUT}%")
    print(f"{'=' * 70}\n")

    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker} {label}...", end="", flush=True)
        items = preload_ticker_data(ticker, option_type)
        elapsed = time.time() - t0
        print(f" {len(items)} days in {elapsed:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker} ({len(items)} days, {N_WORKERS} workers)...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(worker_fn, items, chunksize=4)
        for r in results:
            all_rows.extend(r)
        elapsed = time.time() - t0
        print(f" {len(all_rows)} samples ({elapsed:.0f}s)", flush=True)

    if not all_rows:
        print("  No training data!")
        return None

    df = pd.DataFrame(all_rows)

    meta_cols = ["ticker", "date", "gain_from_low"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"\n  Total samples: {len(df):,}")
    print(f"  Positive (near afternoon low): {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"  Negative: {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # Time-based split
    all_dates = sorted(df["date"].unique())
    split_idx = int(len(all_dates) * 0.8)
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"\n  Train: {len(X_train):,}, dates {all_dates[0]} to {all_dates[split_idx-1]}")
    print(f"  Test:  {len(X_test):,}, dates {all_dates[split_idx]} to {all_dates[-1]}")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": neg_count / max(pos_count, 1),
        "max_depth": 6,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
    }

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(50),
        ],
    )

    # Evaluate
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — AFTERNOON {label}")
    print(f"{'=' * 70}")
    print(f"  AUC: {auc:.4f}")

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_labels = (preds >= thresh).astype(int)
        tp = ((pred_labels == 1) & (y_test == 1)).sum()
        fp = ((pred_labels == 1) & (y_test == 0)).sum()
        fn = ((pred_labels == 0) & (y_test == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_labels = (preds >= best_thresh).astype(int)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)
    acc = accuracy_score(y_test, pred_labels)

    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1: {best_f1:.3f}")

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print(f"\n  Top features:")
    for name, gain in imp[:15]:
        print(f"    {name}: {gain:.0f}")

    # Per-ticker AUC
    print(f"\n  Per-ticker AUC (test set):")
    for ticker in tickers:
        t_mask = df.loc[test_mask, "ticker"] == ticker
        if t_mask.sum() < 50:
            continue
        t_preds = preds[t_mask.values]
        t_labels = y_test[t_mask.values]
        if len(set(t_labels)) < 2:
            continue
        t_auc = roc_auc_score(t_labels, t_preds)
        print(f"    {ticker}: AUC={t_auc:.3f} ({t_mask.sum()} samples)")

    # Save model
    suffix = "call" if option_type == "call" else "put"
    model_path = str(MODEL_DIR / f"pattern_afternoon_{suffix}.txt")
    model.save_model(model_path)

    meta = {
        "features": feature_cols,
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "best_threshold": float(best_thresh),
        "f1": float(best_f1),
        "option_type": option_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "window_start_min": AFTERNOON_START_MIN,
        "window_end_min": AFTERNOON_END_MIN,
        "window_description": "1:00 PM - 3:00 PM ET",
        "min_gain_pct": MIN_GAIN_CALL if option_type == "call" else MIN_GAIN_PUT,
        "train_dates": f"{all_dates[0]} to {all_dates[split_idx-1]}",
        "test_dates": f"{all_dates[split_idx]} to {all_dates[-1]}",
        "tickers": tickers,
    }
    meta_path = str(MODEL_DIR / f"pattern_afternoon_{suffix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved: {model_path}")
    print(f"  Meta saved:  {meta_path}")

    return {"auc": auc, "threshold": best_thresh, "precision": prec, "recall": rec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls-only", action="store_true")
    parser.add_argument("--puts-only", action="store_true")
    parser.add_argument("--ticker", help="Single ticker")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    results = {}

    if not args.puts_only:
        results["call"] = train_model(tickers, "call")

    if not args.calls_only:
        results["put"] = train_model(tickers, "put")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for otype, r in results.items():
        if r:
            print(f"  {otype.upper()}: AUC={r['auc']:.4f}, threshold={r['threshold']:.2f}, "
                  f"precision={r['precision']:.3f}, recall={r['recall']:.3f}")
    print(f"\n  Models saved to: {MODEL_DIR}")
    print(f"  Morning model: UNTOUCHED")
    print(f"  Midday model: UNTOUCHED")


if __name__ == "__main__":
    main()
