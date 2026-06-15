"""Midday pattern entry model: learn 11:00 AM - 1:00 PM ET setups.

Same approach as the morning pattern_entry model but shifted to the
midday window. Looks for dip+bounce patterns in the 90-210 minute range
(11:00 AM - 1:00 PM ET) after the morning session settles.

Saves to separate model files (pattern_midday.txt) — does NOT overwrite
the morning pattern_entry model.

Usage:
    python scripts/train_pattern_midday.py
    python scripts/train_pattern_midday.py --ticker SPY
    python scripts/train_pattern_midday.py --evaluate
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

# Midday window: minutes 90-210 after open (11:00 AM - 1:00 PM ET)
MIDDAY_START_MIN = 90
MIDDAY_END_MIN = 210
MIN_GAIN_FROM_LOW = 15.0    # Slightly lower than morning (moves smaller midday)
LOW_PROXIMITY_PCT = 5.0
MIN_CANDLES = 30            # Need at least 30 candles in the full day


# ── Feature Computation ───────────────────────────────────────────────────
# Reuse same features as morning model — the patterns are similar
# but add midday-specific features

def compute_trailing_features(closes, volumes, ivs, deltas, thetas, underlyings,
                               bids, asks, idx, reference_price):
    """Compute features from trailing candles at position idx.

    reference_price is the price at the start of the midday window (minute 90),
    NOT the 9:30 open.
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

    # ── Premium trajectory ──
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

    # ── Volume ──
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

    # ── IV ──
    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    # ── Underlying ──
    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    # ── Drop from reference (11:00 AM price, not 9:30 open) ──
    if reference_price > 0:
        f["drop_from_open"] = (current / reference_price - 1) * 100
    else:
        f["drop_from_open"] = 0

    # ── Spread ──
    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    if ask > 0 and bid >= 0:
        f["spread_pct"] = (ask - bid) / ask * 100
    else:
        f["spread_pct"] = 0

    # ── Greeks ──
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0

    # ── Time (minutes since market open) ──
    f["minutes_since_open"] = idx

    # ── Premium level ──
    f["premium"] = float(current)

    # ── Midday-specific features ──
    # Morning session context: how much has the premium already moved today?
    if idx >= MIDDAY_START_MIN and closes[0] > 0 and not np.isnan(closes[0]):
        # Morning range as % of open
        morning_slice = closes[:MIDDAY_START_MIN]
        valid_morning = morning_slice[~np.isnan(morning_slice) & (morning_slice > 0)]
        if len(valid_morning) >= 10:
            f["morning_range_pct"] = (float(np.max(valid_morning)) - float(np.min(valid_morning))) / float(np.max(valid_morning)) * 100
            f["morning_return_pct"] = (float(valid_morning[-1]) / float(valid_morning[0]) - 1) * 100
            # Is premium at session low?
            f["prem_vs_session_low"] = (current / float(np.min(valid_morning)) - 1) * 100
        else:
            f["morning_range_pct"] = 0
            f["morning_return_pct"] = 0
            f["prem_vs_session_low"] = 0
    else:
        f["morning_range_pct"] = 0
        f["morning_return_pct"] = 0
        f["prem_vs_session_low"] = 0

    # Underlying momentum from morning
    if idx >= MIDDAY_START_MIN:
        u_morning = underlyings[:MIDDAY_START_MIN]
        valid_u_morning = u_morning[~np.isnan(u_morning) & (u_morning > 0)]
        if len(valid_u_morning) >= 10:
            f["und_morning_return"] = (float(valid_u_morning[-1]) / float(valid_u_morning[0]) - 1) * 100
        else:
            f["und_morning_return"] = 0
    else:
        f["und_morning_return"] = 0

    return f


# ── Worker Function ────────────────────────────────────────────────────────


def _worker_midday(item):
    """Process one ticker-day: find midday low, label candles, extract features."""
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
    # Need enough candles to cover the midday window
    if n < MIDDAY_END_MIN:
        return []

    # Reference price: premium at start of midday window
    reference_price = 0
    for c in closes[max(0, MIDDAY_START_MIN - 3):MIDDAY_START_MIN + 3]:
        if not np.isnan(c) and c > 0:
            reference_price = c
            break
    if reference_price <= 0:
        return []

    # Find the midday low (minutes 90-210)
    midday_slice = closes[MIDDAY_START_MIN:MIDDAY_END_MIN]
    valid_mask = ~np.isnan(midday_slice) & (midday_slice > 0)
    if valid_mask.sum() < 10:
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_values = midday_slice[valid_mask]
    low_rel = np.argmin(valid_values)
    low_idx_abs = MIDDAY_START_MIN + valid_indices[low_rel]  # absolute index
    low_price = valid_values[low_rel]

    if low_price <= 0:
        return []

    # Compute gain from low to subsequent peak (rest of day)
    future = closes[low_idx_abs:]
    future_valid = future[~np.isnan(future) & (future > 0)]
    if len(future_valid) < 5:
        return []
    peak_after = np.nanmax(future_valid)
    gain_from_low = (peak_after / low_price - 1) * 100

    has_move = gain_from_low >= MIN_GAIN_FROM_LOW

    # Label candles in the midday window
    rows = []
    for i in range(MIDDAY_START_MIN + 5, min(MIDDAY_END_MIN, n)):
        if np.isnan(closes[i]) or closes[i] <= 0:
            continue

        if has_move and low_price > 0:
            dist_from_low = (closes[i] / low_price - 1) * 100
            is_near_low = dist_from_low < LOW_PROXIMITY_PCT
        else:
            is_near_low = False

        features = compute_trailing_features(
            closes, volumes, ivs, deltas, thetas, underlyings,
            bids, asks, i, reference_price,
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


def preload_ticker_data(ticker: str) -> list[dict]:
    """Load all day data for a ticker into memory for parallel processing."""
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
        (ticker,),
    ).fetchall()]

    items = []
    for dt in dates:
        atm = conn.execute("""
            SELECT oohlc.strike FROM option_ohlc oohlc
            JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL'
                AND og.underlying_price > 0
            GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
        """, (ticker, dt)).fetchone()
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
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
            ORDER BY oohlc.timestamp
        """, (ticker, dt, strike)).fetchall()

        # Need enough candles to cover midday window (210+ minutes)
        if len(rows) < MIDDAY_END_MIN:
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


def train(tickers: list[str]):
    """Train the midday pattern entry model."""
    print(f"\n{'=' * 70}")
    print("MIDDAY PATTERN ENTRY MODEL (11:00 AM - 1:00 PM ET)")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Window: minutes {MIDDAY_START_MIN}-{MIDDAY_END_MIN} (11:00 AM - 1:00 PM ET)")
    print(f"  Min gain from low: {MIN_GAIN_FROM_LOW}%")
    print(f"  Low proximity: {LOW_PROXIMITY_PCT}%")
    print(f"{'=' * 70}\n")

    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker}...", end="", flush=True)
        items = preload_ticker_data(ticker)
        elapsed = time.time() - t0
        print(f" {len(items)} days in {elapsed:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker} ({len(items)} days, {N_WORKERS} workers)...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker_midday, items, chunksize=4)
        for r in results:
            all_rows.extend(r)
        elapsed = time.time() - t0
        print(f" {len(all_rows)} samples ({elapsed:.0f}s)", flush=True)

    if not all_rows:
        print("  No training data!")
        return

    df = pd.DataFrame(all_rows)

    meta_cols = ["ticker", "date", "gain_from_low"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"\n  Total samples: {len(df):,}")
    print(f"  Positive (near midday low): {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"  Negative: {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")

    print(f"\n  Per-ticker class balance:")
    for ticker in tickers:
        t_df = df[df["ticker"] == ticker]
        if len(t_df) > 0:
            t_pos = t_df["label"].sum()
            print(f"    {ticker}: {len(t_df):,} samples, {t_pos:,} positive ({t_pos/len(t_df)*100:.1f}%)")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # Time-based split: train on first 80% of dates, test on last 20%
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
    print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test positive rate:  {y_test.mean()*100:.1f}%")

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
    print(f"RESULTS — MIDDAY PATTERN ENTRY")
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

    # Threshold sweep with trading impact
    print(f"\n  Threshold sweep (trading impact):")
    print(f"  {'Thresh':<8} {'Signals':<8} {'Precision':<10} {'Recall':<8} {'Avg Gain':<10}")
    test_df = pd.DataFrame({
        "pred": preds, "label": y_test,
        "gain": df.loc[test_mask, "gain_from_low"].values,
        "minute": df.loc[test_mask, "minutes_since_open"].values,
    })
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90]:
        signaled = test_df[test_df["pred"] >= thresh]
        if len(signaled) == 0:
            continue
        tp = (signaled["label"] == 1).sum()
        prec_t = tp / len(signaled)
        rec_t = tp / max((test_df["label"] == 1).sum(), 1)
        avg_gain = signaled[signaled["label"] == 1]["gain"].mean() if tp > 0 else 0
        print(f"  {thresh:<8.2f} {len(signaled):<8} {prec_t:<10.3f} {rec_t:<8.3f} {avg_gain:<10.1f}%")

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print(f"\n  Top features:")
    for name, gain in imp[:15]:
        print(f"    {name}: {gain:.0f}")

    # Per-ticker test performance
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

    # Save model — SEPARATE files from morning model
    model_path = str(MODEL_DIR / "pattern_midday.txt")
    model.save_model(model_path)

    meta = {
        "features": feature_cols,
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "best_threshold": float(best_thresh),
        "f1": float(best_f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "window_start_min": MIDDAY_START_MIN,
        "window_end_min": MIDDAY_END_MIN,
        "window_description": "11:00 AM - 1:00 PM ET",
        "min_gain_pct": MIN_GAIN_FROM_LOW,
        "low_proximity_pct": LOW_PROXIMITY_PCT,
        "train_dates": f"{all_dates[0]} to {all_dates[split_idx-1]}",
        "test_dates": f"{all_dates[split_idx]} to {all_dates[-1]}",
        "tickers": tickers,
    }
    meta_path = str(MODEL_DIR / "pattern_midday_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved: {model_path}")
    print(f"  Meta saved:  {meta_path}")
    print(f"  Morning model: UNTOUCHED (pattern_entry.txt)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Train on specific ticker only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    if args.evaluate:
        model_path = MODEL_DIR / "pattern_midday.txt"
        if not model_path.exists():
            print(f"No midday model at {model_path} — train first")
            sys.exit(1)
        print("Evaluate-only mode not yet implemented for midday model")
        sys.exit(0)

    train(tickers)


if __name__ == "__main__":
    main()
