"""Train a PUT-specific entry timing model using ThetaData PUT chain data.

Adapted from train_pattern_entry.py (CALL entry timing model).

Key differences from CALL version:
  - Queries right='PUT' instead of right='CALL'
  - Scan window: 5-360 min (all day) — PUTs can set up anytime
  - Same labeling: find PUT premium low on days with 20%+ gain, label candles
    within 5% of low as POSITIVE (good entry point)
  - Same features: trailing premium slopes, volume, IV, spread, greeks
  - Saves as put_entry_timing.txt / put_entry_timing_meta.json

For PUTs: buy when premium is cheap (near low), profit when underlying drops
and PUT premium spikes. The entry timing model learns the price action pattern
that precedes PUT premium lows (stabilization after decline, volume surge, etc).

Usage:
    python scripts/train_put_entry_timing.py                  # all tickers
    python scripts/train_put_entry_timing.py --ticker SPY     # single ticker
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
SCAN_END_MINUTES = 360     # full day (9:30-3:30, skip last 30 min for theta)
MIN_GAIN_FROM_LOW = 20.0   # PUT premium must gain 20%+ from local low
LOW_PROXIMITY_PCT = 5.0    # candles within 5% of low are positive
MIN_CANDLES = 30


# -- Feature Computation -----------------------------------------------------


def compute_trailing_features(closes, volumes, ivs, deltas, thetas, underlyings,
                               bids, asks, idx, opening_price):
    """Compute features from trailing candles at position idx.

    Same features as CALL entry timing model — price action patterns are
    direction-agnostic (we're looking for premium near its low in both cases).
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

    # -- Premium trajectory --
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    # Acceleration: is the decline slowing?
    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    # Stabilization: range of last 3 candles as % of price
    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    if max(last3) > 0:
        f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100
    else:
        f["prem_stabilizing"] = 0

    # Premium volatility (std of returns)
    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    # -- Volume --
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

    # -- IV --
    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    # -- Underlying --
    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    # -- Drop from open --
    if opening_price > 0:
        f["drop_from_open"] = (current / opening_price - 1) * 100
    else:
        f["drop_from_open"] = 0

    # -- Spread --
    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    if ask > 0 and bid >= 0:
        f["spread_pct"] = (ask - bid) / ask * 100
    else:
        f["spread_pct"] = 0

    # -- Greeks --
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0

    # -- Time --
    f["minutes_since_open"] = idx

    # -- Premium level --
    f["premium"] = float(current)

    # -- PUT-specific: underlying momentum (bearish = good for PUTs) --
    if len(valid5_u) >= 5 and valid5_u[0] > 0:
        # Underlying 5-candle return (negative = bearish = PUT-favorable)
        f["und_momentum_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_momentum_5"] = 0

    # Underlying 10-candle return
    w10_u_start = max(0, idx - 10)
    u10 = underlyings[w10_u_start:idx]
    u10_valid = u10[~np.isnan(u10)]
    if len(u10_valid) >= 5 and u10_valid[0] > 0:
        f["und_momentum_10"] = (u10_valid[-1] / u10_valid[0] - 1) * 100
    else:
        f["und_momentum_10"] = 0

    # Time bucket (afternoon PUTs may behave differently)
    f["hour_bucket"] = idx // 60  # 0=first hour, 1=second, etc.
    f["is_afternoon"] = 1 if idx >= 210 else 0  # after 1PM ET

    return f


# -- Worker Function ----------------------------------------------------------


def _worker_put_entry(item):
    """Process one ticker-day: find PUT premium lows, label candles, extract features."""
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
    if n < MIN_CANDLES:
        return []

    opening_price = 0
    for c in closes[:5]:
        if not np.isnan(c) and c > 0:
            opening_price = c
            break
    if opening_price <= 0:
        return []

    # For PUTs scanning all day, find the global low in the scan window
    scan_end = min(SCAN_END_MINUTES, n)
    scan = closes[:scan_end]
    valid_mask = ~np.isnan(scan) & (scan > 0)
    if valid_mask.sum() < 10:
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_values = scan[valid_mask]
    low_rel = np.argmin(valid_values)
    low_idx = valid_indices[low_rel]
    low_price = valid_values[low_rel]

    if low_price <= 0:
        return []

    # Compute gain from low to subsequent peak
    future = closes[low_idx:]
    future_valid = future[~np.isnan(future) & (future > 0)]
    if len(future_valid) < 5:
        return []
    peak_after = np.nanmax(future_valid)
    gain_from_low = (peak_after / low_price - 1) * 100

    has_move = gain_from_low >= MIN_GAIN_FROM_LOW

    rows = []

    # Sample every candle in the full scan window
    for i in range(5, scan_end):
        if np.isnan(closes[i]) or closes[i] <= 0:
            continue

        # Label: near the low on a day with a tradeable move
        if has_move and low_price > 0:
            dist_from_low = (closes[i] / low_price - 1) * 100
            is_near_low = dist_from_low < LOW_PROXIMITY_PCT
        else:
            is_near_low = False

        features = compute_trailing_features(
            closes, volumes, ivs, deltas, thetas, underlyings,
            bids, asks, i, opening_price,
        )
        if features is None:
            continue

        features["label"] = 1 if is_near_low else 0
        features["ticker"] = ticker
        features["date"] = dt
        features["gain_from_low"] = round(gain_from_low, 1)
        rows.append(features)

    return rows


# -- Data Loading --------------------------------------------------------------


def preload_ticker_data(ticker: str) -> list[dict]:
    """Load all PUT day data for a ticker into memory for parallel processing."""
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? AND right='PUT' ORDER BY 1",
        (ticker,),
    ).fetchall()]

    items = []
    for dt in dates:
        # Find ATM strike for PUTs
        atm = conn.execute("""
            SELECT oohlc.strike FROM option_ohlc oohlc
            JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='PUT'
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
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='PUT' AND oohlc.strike=?
            ORDER BY oohlc.timestamp
        """, (ticker, dt, strike)).fetchall()

        if len(rows) < MIN_CANDLES:
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


# -- Training ------------------------------------------------------------------


def train(tickers: list[str]):
    """Train the PUT entry timing model."""
    print(f"\n{'=' * 70}")
    print("PUT ENTRY TIMING MODEL")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Scan window: 5-{SCAN_END_MINUTES} min (full day)")
    print(f"  Min gain from low: {MIN_GAIN_FROM_LOW}%")
    print(f"  Low proximity: {LOW_PROXIMITY_PCT}%")
    print(f"{'=' * 70}\n")

    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker} (PUT)...", end="", flush=True)
        items = preload_ticker_data(ticker)
        elapsed = time.time() - t0
        print(f" {len(items)} days in {elapsed:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker} ({len(items)} days, {N_WORKERS} workers)...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker_put_entry, items, chunksize=4)
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
    print(f"  Positive (near low): {pos:,} ({pos/len(df)*100:.1f}%)")
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

    train_mask = df["date"].isin(train_dates)
    test_mask = ~train_mask

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
    print(f"RESULTS")
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
    print(f"  Precision: {prec:.3f} (of predicted entries, how many are near the low)")
    print(f"  Recall: {rec:.3f} (of actual lows, how many do we catch)")
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
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        signaled = test_df[test_df["pred"] >= thresh]
        if len(signaled) == 0:
            continue
        tp = (signaled["label"] == 1).sum()
        prec_t = tp / len(signaled)
        rec_t = tp / max((test_df["label"] == 1).sum(), 1)
        avg_gain = signaled[signaled["label"] == 1]["gain"].mean() if tp > 0 else 0
        print(f"  {thresh:<8.1f} {len(signaled):<8} {prec_t:<10.3f} {rec_t:<8.3f} {avg_gain:<10.1f}%")

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

    # Save model
    model_path = str(MODEL_DIR / "put_entry_timing.txt")
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
        "scan_window_minutes": SCAN_END_MINUTES,
        "min_gain_pct": MIN_GAIN_FROM_LOW,
        "low_proximity_pct": LOW_PROXIMITY_PCT,
        "train_dates": f"{all_dates[0]} to {all_dates[split_idx-1]}",
        "test_dates": f"{all_dates[split_idx]} to {all_dates[-1]}",
    }
    with open(str(MODEL_DIR / "put_entry_timing_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {model_path}")
    return model, meta


def main():
    parser = argparse.ArgumentParser(description="Train PUT entry timing model")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    t0 = time.time()
    train(tickers)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
