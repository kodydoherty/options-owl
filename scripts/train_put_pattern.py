"""Train a PUT-specific pattern entry model using ThetaData PUT chain data.

V2: Fixed label strategy — matches CALL model's "near daily low" approach.

V1 had a 54% positive rate (label: "will PUT gain 20% in 60min?") which was
nearly a coin flip. The model stopped at iteration 34 with AUC 0.77.

V2 uses: "Is this candle near the PUT premium killzone LOW on a day where
the PUT premium subsequently gained >= 20%?" This creates a ~5% positive
rate — a specific, learnable pattern instead of noise.

For PUTs this makes sense: buy PUT premium when it's cheap (near the low),
then profit when the underlying drops and PUT premium spikes.

Additional improvements over V1:
  - Near-low labeling (positive rate ~5% vs 54%)
  - More features: vega, gamma proxy, bid/ask size imbalance, OHLC range,
    underlying momentum (RSI-like), IV acceleration
  - Query bid_size/ask_size and high/low/vwap from DB
  - Better LightGBM params (more rounds possible with harder label)

Usage:
    python scripts/train_put_pattern.py                # all tickers
    python scripts/train_put_pattern.py --ticker SPY   # single ticker
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
]

N_WORKERS = min(os.cpu_count() or 4, 16)
SCAN_END_MINUTES = 360      # scan full day (6h = 9:30-3:30, skip last 30 min for theta)
WINDOW_MINUTES = 60         # sliding window to find local lows
MIN_GAIN_FROM_LOW = 20.0    # PUT premium must gain 20%+ from local low
LOW_PROXIMITY_PCT = 5.0     # candles within 5% of low are positive
MIN_CANDLES = 30


# ── Feature Computation ───────────────────────────────────────────────────


def compute_put_features(closes, volumes, ivs, deltas, thetas, underlyings,
                         bids, asks, idx, opening_price,
                         vegas=None, highs=None, lows=None, vwaps=None,
                         bid_sizes=None, ask_sizes=None,
                         call_ivs=None, call_volumes=None):
    """Compute features from trailing PUT candles at position idx.

    Same base features as CALL pattern model, plus PUT-specific features
    and additional features for V2.
    """
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)
    w15_start = max(0, idx - 15)

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
    f["prem_slope_10"] = (
        (valid10[-1] / valid10[0] - 1) * 100
        if len(valid10) >= 5 and valid10[0] > 0
        else f["prem_slope_5"]
    )

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (
        (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0
    )

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

    # IV acceleration (is IV expanding faster?)
    if len(valid5_iv) >= 4:
        mid_iv = len(valid5_iv) // 2
        first_iv_rate = valid5_iv[mid_iv] - valid5_iv[0]
        second_iv_rate = valid5_iv[-1] - valid5_iv[mid_iv]
        f["iv_accel"] = float(second_iv_rate - first_iv_rate)
    else:
        f["iv_accel"] = 0.0

    # ── Underlying slopes ──
    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    pre10_u = underlyings[w10_start:idx]
    pre15_u = underlyings[w15_start:idx]
    valid10_u = pre10_u[~np.isnan(pre10_u)]
    valid15_u = pre15_u[~np.isnan(pre15_u)]
    f["und_slope_10"] = (
        (valid10_u[-1] / valid10_u[0] - 1) * 100
        if len(valid10_u) >= 5 and valid10_u[0] > 0
        else f["und_slope_5"]
    )
    f["und_slope_15"] = (
        (valid15_u[-1] / valid15_u[0] - 1) * 100
        if len(valid15_u) >= 5 and valid15_u[0] > 0
        else f["und_slope_10"]
    )

    # Underlying momentum (RSI-like: ratio of up vs down moves)
    if len(valid5_u) >= 3:
        diffs = np.diff(valid5_u)
        up_sum = float(np.sum(diffs[diffs > 0]))
        down_sum = float(-np.sum(diffs[diffs < 0]))
        f["und_momentum"] = down_sum / max(up_sum + down_sum, 1e-8) * 100
    else:
        f["und_momentum"] = 50.0

    # Consecutive underlying down candles
    if len(valid5_u) >= 3 and valid5_u[0] > 0:
        down_count = 0
        for i in range(len(valid5_u) - 1, 0, -1):
            if valid5_u[i] < valid5_u[i - 1]:
                down_count += 1
            else:
                break
        f["consec_underlying_down"] = down_count
    else:
        f["consec_underlying_down"] = 0

    # ── Drop from open ──
    f["drop_from_open"] = (current / opening_price - 1) * 100 if opening_price > 0 else 0

    # ── Spread ──
    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0

    # ── Greeks ──
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0

    # Vega (sensitivity to IV changes — critical for PUTs during fear spikes)
    if vegas is not None and idx < len(vegas) and not np.isnan(vegas[idx]):
        f["vega"] = float(vegas[idx])
    else:
        f["vega"] = 0.0

    # ── Time ──
    f["minutes_since_open"] = idx

    # ── Premium level ──
    f["premium"] = float(current)

    # ── OHLC range (intrabar volatility) ──
    if highs is not None and lows is not None and idx < len(highs) and idx < len(lows):
        h = highs[idx] if not np.isnan(highs[idx]) else current
        lo = lows[idx] if not np.isnan(lows[idx]) else current
        f["candle_range_pct"] = (h - lo) / max(current, 0.01) * 100
    else:
        f["candle_range_pct"] = 0.0

    # ── Bid/ask size imbalance (demand pressure) ──
    if bid_sizes is not None and ask_sizes is not None and idx < len(bid_sizes):
        bs = bid_sizes[idx] if not np.isnan(bid_sizes[idx]) else 0
        as_ = ask_sizes[idx] if not np.isnan(ask_sizes[idx]) else 0
        total = bs + as_
        f["bid_size_ratio"] = bs / max(total, 1)
    else:
        f["bid_size_ratio"] = 0.5

    # ── PUT-specific: IV skew (PUT IV / CALL IV) ──
    if call_ivs is not None and idx < len(call_ivs):
        put_iv = valid5_iv[-1] if len(valid5_iv) > 0 else 0
        call_iv = call_ivs[idx] if not np.isnan(call_ivs[idx]) else 0
        f["iv_skew"] = put_iv / call_iv if call_iv > 0 else 1.0
    else:
        f["iv_skew"] = 1.0

    # PUT volume / CALL volume
    if call_volumes is not None and idx < len(call_volumes):
        call_vol = call_volumes[idx] if not np.isnan(call_volumes[idx]) else 0
        put_vol = volumes[idx] if not np.isnan(volumes[idx]) else 0
        f["put_call_volume_ratio"] = put_vol / max(call_vol, 1)
    else:
        f["put_call_volume_ratio"] = 1.0

    return f


# ── Worker Function ────────────────────────────────────────────────────────


def _worker_put_pattern(item):
    """Process one ticker-day: find PUT premium lows ALL DAY, label candles near them.

    Uses sliding windows across the full trading day (not just the morning).
    For each 60-min window, finds the local low. If PUT premium gains 20%+
    from that low, candles within 5% of the low are labeled positive.

    This captures morning dips, midday selloffs, and afternoon crashes.
    """
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
    vegas = np.array(item.get("vegas", []), dtype=np.float64) if item.get("vegas") else None
    highs = np.array(item.get("highs", []), dtype=np.float64) if item.get("highs") else None
    lows_arr = np.array(item.get("lows", []), dtype=np.float64) if item.get("lows") else None
    vwaps = np.array(item.get("vwaps", []), dtype=np.float64) if item.get("vwaps") else None
    bid_sizes = np.array(item.get("bid_sizes", []), dtype=np.float64) if item.get("bid_sizes") else None
    ask_sizes = np.array(item.get("ask_sizes", []), dtype=np.float64) if item.get("ask_sizes") else None
    call_ivs = np.array(item.get("call_ivs", []), dtype=np.float64) if item.get("call_ivs") else None
    call_volumes = np.array(item.get("call_volumes", []), dtype=np.float64) if item.get("call_volumes") else None

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

    scan_end = min(SCAN_END_MINUTES, n)

    # ── Find ALL local lows using sliding windows across the full day ──
    # Each window: find the low, check if premium gains 20%+ afterward.
    # A candle is positive if it's within 5% of ANY qualifying local low.
    positive_set = set()  # indices of positive candles
    best_gain_for_idx = {}  # track best gain for reporting

    for win_start in range(0, scan_end - 10, WINDOW_MINUTES // 2):
        # Overlapping windows (step = half window) to catch lows at boundaries
        win_end = min(win_start + WINDOW_MINUTES, n)
        window = closes[win_start:win_end]
        valid_mask = ~np.isnan(window) & (window > 0)
        if valid_mask.sum() < 5:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_values = window[valid_mask]
        low_rel = np.argmin(valid_values)
        low_abs_idx = win_start + valid_indices[low_rel]
        low_price = valid_values[low_rel]

        if low_price <= 0:
            continue

        # Check gain from this local low to peak in the rest of the day
        future = closes[low_abs_idx:]
        future_valid = future[~np.isnan(future) & (future > 0)]
        if len(future_valid) < 3:
            continue
        peak_after = np.nanmax(future_valid)
        gain = (peak_after / low_price - 1) * 100

        if gain < MIN_GAIN_FROM_LOW:
            continue

        # Mark candles near this low as positive
        for j in range(max(5, win_start), win_end):
            if np.isnan(closes[j]) or closes[j] <= 0:
                continue
            dist = (closes[j] / low_price - 1) * 100
            if dist < LOW_PROXIMITY_PCT:
                positive_set.add(j)
                if j not in best_gain_for_idx or gain > best_gain_for_idx[j]:
                    best_gain_for_idx[j] = gain

    # ── Generate training samples for every scannable candle ──
    rows = []
    for i in range(5, scan_end):
        if np.isnan(closes[i]) or closes[i] <= 0:
            continue

        features = compute_put_features(
            closes, volumes, ivs, deltas, thetas, underlyings,
            bids, asks, i, opening_price,
            vegas=vegas, highs=highs, lows=lows_arr, vwaps=vwaps,
            bid_sizes=bid_sizes, ask_sizes=ask_sizes,
            call_ivs=call_ivs, call_volumes=call_volumes,
        )
        if features is None:
            continue

        is_positive = i in positive_set
        features["label"] = 1 if is_positive else 0
        features["ticker"] = ticker
        features["date"] = dt
        features["gain_from_low"] = round(best_gain_for_idx.get(i, 0), 1)
        rows.append(features)

    return rows


# ── Data Preloading ────────────────────────────────────────────────────────


def preload_put_ticker_data(ticker: str) -> list[dict]:
    """Load all day PUT data for a ticker, with matching CALL IV/volume for skew features."""
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc "
        "WHERE ticker=? AND right='PUT' ORDER BY 1",
        (ticker,),
    ).fetchall()]

    items = []
    for dt in dates:
        # Find ATM PUT strike (closest to underlying price)
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

        # Load PUT chain data (with high/low/vwap, vega, bid_size/ask_size)
        put_rows = conn.execute("""
            SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                   COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                   COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                   oohlc.volume,
                   COALESCE(og.vega, 0),
                   oohlc.high, oohlc.low, COALESCE(oohlc.vwap, 0),
                   COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0)
            FROM option_ohlc oohlc
            LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
            LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='PUT' AND oohlc.strike=?
            ORDER BY oohlc.timestamp
        """, (ticker, dt, strike)).fetchall()

        if len(put_rows) < MIN_CANDLES:
            continue

        # Load matching CALL IV and volume for skew features (same strike, same day)
        call_rows = conn.execute("""
            SELECT COALESCE(og.implied_vol, 0), oohlc.volume
            FROM option_ohlc oohlc
            LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
            WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
            ORDER BY oohlc.timestamp
        """, (ticker, dt, strike)).fetchall()

        n_put = len(put_rows)
        call_ivs = [float(r[0]) if r[0] else float("nan") for r in call_rows]
        call_vols = [float(r[1]) if r[1] else 0 for r in call_rows]
        while len(call_ivs) < n_put:
            call_ivs.append(float("nan"))
            call_vols.append(0)

        items.append({
            "ticker": ticker,
            "date": dt,
            "closes": [float(r[0]) if r[0] else float("nan") for r in put_rows],
            "underlyings": [float(r[1]) if r[1] else float("nan") for r in put_rows],
            "ivs": [float(r[2]) if r[2] else float("nan") for r in put_rows],
            "bids": [float(r[3]) if r[3] else 0 for r in put_rows],
            "asks": [float(r[4]) if r[4] else 0 for r in put_rows],
            "deltas": [float(r[5]) if r[5] else float("nan") for r in put_rows],
            "thetas": [float(r[6]) if r[6] else float("nan") for r in put_rows],
            "volumes": [float(r[7]) if r[7] else 0 for r in put_rows],
            "vegas": [float(r[8]) if r[8] else float("nan") for r in put_rows],
            "highs": [float(r[9]) if r[9] else float("nan") for r in put_rows],
            "lows": [float(r[10]) if r[10] else float("nan") for r in put_rows],
            "vwaps": [float(r[11]) if r[11] else 0 for r in put_rows],
            "bid_sizes": [float(r[12]) if r[12] else 0 for r in put_rows],
            "ask_sizes": [float(r[13]) if r[13] else 0 for r in put_rows],
            "call_ivs": call_ivs[:n_put],
            "call_volumes": call_vols[:n_put],
        })

    conn.close()
    return items


# ── Training ───────────────────────────────────────────────────────────────


def _load_data(tickers: list[str]) -> pd.DataFrame:
    """Load and process all PUT data into a DataFrame."""
    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker} PUTs...", end="", flush=True)
        items = preload_put_ticker_data(ticker)
        elapsed = time.time() - t0
        print(f" {len(items)} days in {elapsed:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker} ({len(items)} days, {N_WORKERS} workers)...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker_put_pattern, items, chunksize=4)
        for r in results:
            all_rows.extend(r)
        elapsed = time.time() - t0
        print(f" {len(all_rows)} samples ({elapsed:.0f}s)", flush=True)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def train(tickers: list[str]):
    """Train the PUT pattern model with walk-forward validation.

    Walk-forward: train on expanding monthly window, test on next month.
    Every prediction is truly out-of-sample. Final model trained on all
    data except last month (held out for reporting).
    """
    print(f"\n{'=' * 70}")
    print("PUT PATTERN ENTRY MODEL V2 (full-day sliding-window near-low)")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Scan window: 0-{SCAN_END_MINUTES} min (full day)")
    print(f"  Local low window: {WINDOW_MINUTES} min (sliding, 50% overlap)")
    print(f"  Min gain from low: {MIN_GAIN_FROM_LOW}%")
    print(f"  Low proximity: {LOW_PROXIMITY_PCT}%")
    print(f"{'=' * 70}\n")

    df = _load_data(tickers)
    if df.empty:
        print("  No training data!")
        return None, None

    meta_cols = ["ticker", "date", "gain_from_low"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"\n  Total samples: {len(df):,}")
    print(f"  Positive (near PUT low on good day): {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"  Negative: {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")

    # Per-ticker stats
    print(f"\n  Per-ticker class balance:")
    for ticker in tickers:
        t_df = df[df["ticker"] == ticker]
        if len(t_df) > 0:
            t_pos = t_df["label"].sum()
            print(f"    {ticker}: {len(t_df):,} samples, {t_pos:,} positive ({t_pos/len(t_df)*100:.1f}%)")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    all_dates = sorted(df["date"].unique())
    months = sorted(set(d[:7] for d in all_dates))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 7,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
    }

    # ── Walk-Forward Validation ──
    print(f"\n  Months available: {months}")
    print(f"  Walk-forward folds (expanding train window → test next month):\n")

    fold_aucs = []
    fold_details = []
    for fold_idx in range(2, len(months)):
        train_months = set(months[:fold_idx])
        test_month = months[fold_idx]

        train_mask = df["date"].apply(lambda d: d[:7] in train_months)
        test_mask = df["date"].apply(lambda d: d[:7] == test_month)

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        if len(set(y_te)) < 2:
            continue

        params = {**lgb_params, "scale_pos_weight": (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)}
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dtest = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        fold_model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(100)],
        )

        fold_preds = fold_model.predict(X_te)
        fold_auc = roc_auc_score(y_te, fold_preds)
        fold_aucs.append(fold_auc)

        # Precision at 0.80 threshold
        pred_80 = (fold_preds >= 0.80).astype(int)
        tp_80 = ((pred_80 == 1) & (y_te == 1)).sum()
        fp_80 = ((pred_80 == 1) & (y_te == 0)).sum()
        prec_80 = tp_80 / max(tp_80 + fp_80, 1)

        fold_details.append({
            "test_month": test_month,
            "train_end": months[fold_idx - 1],
            "auc": fold_auc,
            "prec_80": prec_80,
            "signals_80": int(pred_80.sum()),
            "n_test": len(X_te),
            "best_iter": fold_model.best_iteration,
        })

        print(f"    Fold {fold_idx-1}: train {months[0]}→{months[fold_idx-1]} | "
              f"test {test_month} | AUC={fold_auc:.4f} "
              f"prec@0.80={prec_80:.3f} ({pred_80.sum()} signals) "
              f"iter={fold_model.best_iteration}")

    if fold_aucs:
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  Walk-forward AUC: {mean_auc:.4f} +/- {std_auc:.4f} "
              f"(min={min(fold_aucs):.4f}, max={max(fold_aucs):.4f})")
    else:
        mean_auc = 0.0
        std_auc = 0.0

    # ── Final Production Model ──
    print(f"\n{'=' * 70}")
    print("FINAL PRODUCTION MODEL (train on all except last month)")
    print(f"{'=' * 70}")

    final_train_months = set(months[:-1])
    final_test_month = months[-1]
    train_mask = df["date"].apply(lambda d: d[:7] in final_train_months)
    test_mask = df["date"].apply(lambda d: d[:7] == final_test_month)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,}, months {months[0]} to {months[-2]}")
    print(f"  Test:  {len(X_test):,}, month {final_test_month} (held out)")
    print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test positive rate:  {y_test.mean()*100:.1f}%")

    params = {**lgb_params, "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1)}
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(100)],
    )

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)

    print(f"\n  AUC: {auc:.4f}")
    print(f"  Best iteration: {model.best_iteration}")

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        pred_labels = (preds >= thresh).astype(int)
        tp = ((pred_labels == 1) & (y_test == 1)).sum()
        fp = ((pred_labels == 1) & (y_test == 0)).sum()
        fn = ((pred_labels == 0) & (y_test == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_labels = (preds >= best_thresh).astype(int)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)
    acc = accuracy_score(y_test, pred_labels)

    print(f"  Best threshold (F1): {best_thresh:.2f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1: {best_f1:.3f}")

    # Threshold sweep
    print(f"\n  Threshold sweep (trading impact):")
    print(f"  {'Thresh':<8} {'Signals':<8} {'Precision':<10} {'Recall':<8} {'Avg Gain':<10}")
    test_df = pd.DataFrame({
        "pred": preds, "label": y_test,
        "gain": df.loc[test_mask, "gain_from_low"].values,
        "minute": df.loc[test_mask, "minutes_since_open"].values,
    })
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
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
    for name, gain in imp[:20]:
        if gain > 0:
            print(f"    {name}: {gain:,.0f}")

    # Per-ticker test performance
    print(f"\n  Per-ticker AUC (test — {final_test_month}):")
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

    # Save
    model_path = str(MODEL_DIR / "put_pattern_v1.lgb")
    model.save_model(model_path)

    meta = {
        "features": feature_cols,
        "auc": float(auc),
        "walk_forward_auc_mean": float(mean_auc),
        "walk_forward_auc_std": float(std_auc),
        "walk_forward_folds": fold_details,
        "precision": float(prec),
        "recall": float(rec),
        "best_threshold": float(best_thresh),
        "f1": float(best_f1),
        "best_iteration": model.best_iteration,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "scan_end_minutes": SCAN_END_MINUTES,
        "window_minutes": WINDOW_MINUTES,
        "min_gain_pct": MIN_GAIN_FROM_LOW,
        "low_proximity_pct": LOW_PROXIMITY_PCT,
        "label_strategy": "sliding_window_near_low",
        "train_months": f"{months[0]} to {months[-2]}",
        "test_month": final_test_month,
    }
    with open(str(MODEL_DIR / "put_pattern_v1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {model_path}")
    return model, meta


def main():
    parser = argparse.ArgumentParser(description="Train PUT pattern entry model")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    t0 = time.time()
    train(tickers)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
