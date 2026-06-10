"""Ablation study: fix training pipeline issues one at a time, measure impact.

Identified problems:
  A) Same-candle leakage: features include entry candle (idx), should use idx-1
  B) Easy negatives: random timestamps far from moves, model learns "is this alive"
  C) Trivial threshold: +15% is too easy for 0DTE (72% base positive rate on SPY)
  D) Class imbalance: 72% positive makes high AUC trivial

This script tests each fix independently, then combines them, sweeping
MIN_MOVE_PCT from 15% to 50% in 1% steps. Runs on 1-2 tickers (SPY + TSLA)
for speed, outputs a CSV table of results.

Usage:
    python scripts/ablation_study.py                      # full study on SPY
    python scripts/ablation_study.py --ticker TSLA         # single ticker
    python scripts/ablation_study.py --quick               # fewer thresholds (15,20,25,30,40,50)
    python scripts/ablation_study.py --fix A               # only test Fix A
    python scripts/ablation_study.py --fix B               # only test Fix B
    python scripts/ablation_study.py --sweep               # threshold sweep with all fixes applied
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.train_option_signals_v2 import (
    COOLDOWN_MIN,
    FEATURE_COLS,
    MOVE_WINDOW_MIN,
    PRE_MOVE_LOOKBACK,
    THETADATA_DB,
    _V6_SETTINGS,
    simulate_with_production_fsm,
)

# ---------------------------------------------------------------------------
# Fix A: Feature extraction WITHOUT same-candle leakage
# ---------------------------------------------------------------------------

def compute_features_no_leakage(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    stock: pd.DataFrame,
    idx: int,
    lookback: int = PRE_MOVE_LOOKBACK,
) -> dict | None:
    """Compute features using data STRICTLY BEFORE the entry candle.

    Fix A: window is [idx-lookback, idx) — excludes idx itself.
    Entry price comes from the PREVIOUS candle's close (decision point).
    """
    if idx < lookback + 1 or idx >= len(ohlc):
        return None

    # Decision candle is idx-1 (the last candle we've FULLY seen)
    prev = ohlc.iloc[idx - 1]
    window = ohlc.iloc[max(0, idx - lookback - 1):idx]  # up to but NOT including idx
    decision_price = prev.get("close", 0) or 0
    if decision_price <= 0:
        return None

    f = {}

    # --- Time of day (from decision candle, not entry) ---
    try:
        ts = pd.Timestamp(prev["timestamp"])
        if ts.tzinfo:
            ts = ts.tz_convert("America/New_York")
        f["minutes_since_open"] = max(0, (ts.hour - 9) * 60 + ts.minute - 30)
    except Exception:
        f["minutes_since_open"] = 0
    f["hour_bucket"] = f["minutes_since_open"] // 60
    f["is_first_30min"] = 1 if f["minutes_since_open"] <= 30 else 0
    f["is_last_hour"] = 1 if f["minutes_since_open"] >= 330 else 0

    # --- Premium action (from window BEFORE entry) ---
    prices = window["close"].dropna().values
    if len(prices) < 3:
        return None

    f["premium"] = float(decision_price)
    f["premium_change_5m"] = float((prices[-1] / prices[max(-6, -len(prices))] - 1) * 100) if prices[max(-6, -len(prices))] > 0 else 0
    f["premium_change_10m"] = float((prices[-1] / prices[max(-11, -len(prices))] - 1) * 100) if prices[max(-11, -len(prices))] > 0 else 0
    f["premium_change_15m"] = float((prices[-1] / prices[0] - 1) * 100) if prices[0] > 0 else 0

    if len(prices) > 2 and all(prices[:-1] > 0):
        returns = np.diff(prices) / prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
        f["premium_skew"] = float(pd.Series(returns).skew()) if len(returns) > 3 else 0
    else:
        f["premium_volatility"] = 0
        f["premium_skew"] = 0

    f["range_position"] = float((prices[-1] - prices.min()) / (prices.max() - prices.min())) if prices.max() > prices.min() else 0.5
    f["near_low"] = 1 if f["range_position"] < 0.25 else 0
    f["near_high"] = 1 if f["range_position"] > 0.85 else 0

    if len(prices) > 1:
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

    # --- Volume ---
    vols = window["volume"].fillna(0).values if "volume" in window.columns else np.zeros(len(window))
    f["current_volume"] = float(vols[-1])
    avg_vol = float(np.mean(vols[:-1])) if len(vols) > 1 else 1
    f["volume_ratio"] = float(vols[-1] / max(avg_vol, 1))
    f["volume_trend"] = float(np.mean(vols[-5:]) / max(np.mean(vols[:max(len(vols)-5, 1)]), 1)) if len(vols) > 5 else 1.0
    if len(vols) > 5 and np.std(vols[:-1]) > 0:
        f["volume_zscore"] = float((vols[-1] - np.mean(vols[:-1])) / np.std(vols[:-1]))
    else:
        f["volume_zscore"] = 0

    # --- Bid/ask (from decision candle) ---
    if len(quotes) > idx - 1:
        q_window = quotes.iloc[max(0, idx - lookback - 1):idx]
        if len(q_window) > 0:
            q = q_window.iloc[-1]
            bid = q.get("bid", 0) or 0
            ask = q.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else decision_price
            f["spread"] = float(ask - bid) if ask > bid else 0
            f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
            if len(q_window) > 3:
                spreads = (q_window["ask"].fillna(0) - q_window["bid"].fillna(0)).values
                spreads = spreads[spreads >= 0]
                if len(spreads) > 3:
                    first_half = spreads[:len(spreads) // 2].mean()
                    second_half = spreads[len(spreads) // 2:].mean()
                    f["spread_tightening"] = float(first_half - second_half)
                else:
                    f["spread_tightening"] = 0
            else:
                f["spread_tightening"] = 0
            f["bid_size"] = float(q.get("bid_size", 0) or 0)
            f["ask_size"] = float(q.get("ask_size", 0) or 0)
            f["size_imbalance"] = float((f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1))
        else:
            for k in ["spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance"]:
                f[k] = 0
    else:
        for k in ["spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance"]:
            f[k] = 0

    # --- Greeks (from decision candle) ---
    if len(greeks) > idx - 1:
        g_window = greeks.iloc[max(0, idx - lookback - 1):idx]
        if len(g_window) > 0:
            g = g_window.iloc[-1]
            f["iv"] = float(g.get("implied_vol", 0) or 0)
            f["delta"] = float(abs(g.get("delta", 0) or 0))
            f["theta"] = float(g.get("theta", 0) or 0)
            f["vega"] = float(g.get("vega", 0) or 0)
            if len(g_window) > 3 and g_window["implied_vol"].notna().sum() > 3:
                ivs = g_window["implied_vol"].dropna().values
                f["iv_change_5m"] = float(ivs[-1] - ivs[max(-6, -len(ivs))]) if len(ivs) > 5 else 0
                f["iv_change_15m"] = float(ivs[-1] - ivs[0])
                f["iv_trend"] = float(np.polyfit(range(len(ivs)), ivs, 1)[0]) if len(ivs) > 2 else 0
            else:
                f["iv_change_5m"] = 0
                f["iv_change_15m"] = 0
                f["iv_trend"] = 0
            f["underlying_price"] = float(g.get("underlying_price", 0) or 0)
        else:
            for k in ["iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend", "underlying_price"]:
                f[k] = 0
    else:
        for k in ["iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend", "underlying_price"]:
            f[k] = 0

    # --- Underlying ---
    if len(stock) > 0:
        s_idx = min(idx - 1, len(stock) - 1)
        s_window = stock.iloc[max(0, s_idx - lookback):s_idx + 1]
        if len(s_window) > 1:
            s_closes = s_window["close"].dropna().values
            if len(s_closes) > 1 and all(s_closes > 0):
                f["underlying_change_5m"] = float((s_closes[-1] / s_closes[max(-6, -len(s_closes))] - 1) * 100)
                f["underlying_change_15m"] = float((s_closes[-1] / s_closes[0] - 1) * 100)
                f["underlying_volatility"] = float(np.std(np.diff(s_closes) / s_closes[:-1]) * 100)
                vwap = np.mean(s_closes)
                f["vwap_deviation"] = float((s_closes[-1] / vwap - 1) * 100) if vwap > 0 else 0
            else:
                for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
                    f[k] = 0
        else:
            for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
                f[k] = 0
    else:
        for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
            f[k] = 0

    # --- Computed patterns ---
    f["coiled_spring"] = 1 if (f["premium_volatility"] < 2 and f["volume_ratio"] > 1.5) else 0
    f["volume_breakout"] = 1 if (f["volume_zscore"] > 2 and f["near_high"]) else 0
    f["bounce_setup"] = 1 if (f["near_low"] and f["spread_tightening"] > 0) else 0
    f["iv_expanding"] = 1 if (f.get("iv_change_5m", 0) > 0.02) else 0
    f["momentum_ignition"] = 1 if (f["consecutive_up_bars"] >= 3 and f["volume_trend"] > 1.3) else 0

    right_val = prev.get("right", "CALL")
    f["is_call"] = 1 if str(right_val).upper() == "CALL" else 0

    return f


# ---------------------------------------------------------------------------
# Fix B: Hard negatives — timestamps that LOOKED good but LOST money
# ---------------------------------------------------------------------------

def find_losing_entries(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    ticker: str,
    min_move_pct: float,
    cooldown: int = COOLDOWN_MIN,
    dte: int = 0,
    expiry_date: str = "",
) -> list[dict]:
    """Find entries where the FSM would have LOST money (hard negatives).

    These are timestamps where someone might have entered but the trade
    was unprofitable — the model needs to learn to AVOID these.
    """
    losers = []
    last_idx = -cooldown
    closes = ohlc["close"].values
    n = len(closes)

    for i in range(PRE_MOVE_LOOKBACK, n - 10):
        if i - last_idx < cooldown:
            continue

        entry = closes[i]
        if not entry or entry <= 0 or np.isnan(entry):
            continue

        result = simulate_with_production_fsm(
            ohlc, quotes, greeks, i,
            ticker=ticker, dte=dte, expiry_date=expiry_date,
        )
        if result is None:
            continue

        # This entry LOST money (or made less than threshold)
        if result["pnl_pct"] < min_move_pct:
            losers.append({
                "idx": i,
                "entry_price": float(entry),
                "pnl_pct": result["pnl_pct"],
                "peak_gain": result["peak_gain"],
                "hold_minutes": result["hold_minutes"],
                "exit_reason": result["reason"],
                "right": ohlc.iloc[i].get("right", "CALL"),
            })
            last_idx = i

    return losers


# ---------------------------------------------------------------------------
# Build dataset with configurable fixes
# ---------------------------------------------------------------------------

def build_dataset_ablation(
    conn: sqlite3.Connection,
    ticker: str,
    min_move_pct: float = 15.0,
    fix_leakage: bool = False,     # Fix A: no same-candle features
    hard_negatives: bool = False,   # Fix B: use losing trades as negatives
    balanced: bool = False,         # Fix D: force 50/50 class ratio
    neg_ratio: int = 2,
    max_days: int | None = None,    # limit days for speed
) -> pd.DataFrame:
    """Build labeled dataset with configurable fixes applied."""

    # Import original feature extractor for baseline
    from scripts.train_option_signals_v2 import compute_setup_features as original_features

    feature_fn = compute_features_no_leakage if fix_leakage else original_features

    dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
        (ticker,),
    ).fetchall()]

    if not dates:
        return pd.DataFrame()

    if max_days:
        dates = dates[:max_days]

    all_rows = []
    total_moves = 0
    total_losers = 0

    for dt in dates:
        ohlc = pd.read_sql_query(
            "SELECT * FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )
        quotes = pd.read_sql_query(
            "SELECT * FROM option_quotes WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )
        greeks = pd.read_sql_query(
            "SELECT * FROM option_greeks WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )
        stock = pd.read_sql_query(
            "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )

        for right in ["CALL", "PUT"]:
            ohlc_side = ohlc[ohlc["right"].str.upper() == right].reset_index(drop=True)
            quotes_side = quotes[quotes["right"].str.upper() == right].reset_index(drop=True) if not quotes.empty else pd.DataFrame()
            greeks_side = greeks[greeks["right"].str.upper() == right].reset_index(drop=True) if not greeks.empty else pd.DataFrame()

            if len(ohlc_side) < 30:
                continue

            # Find winning entries (positives)
            from scripts.train_option_signals_v2 import find_profitable_moves
            moves = find_profitable_moves(
                ohlc_side, quotes_side, greeks_side,
                ticker=ticker, min_move_pct=min_move_pct,
                dte=0, expiry_date=dt,
            )
            total_moves += len(moves)

            # Positive samples
            move_indices = set()
            for move in moves:
                idx = move["idx"]
                features = feature_fn(ohlc_side, quotes_side, greeks_side, stock, idx)
                if features:
                    features["label"] = 1
                    features["peak_pct"] = move.get("peak_gain", move.get("pnl_pct", 0))
                    features["pnl_pct"] = move.get("pnl_pct", 0)
                    features["ticker"] = ticker
                    features["date"] = dt
                    all_rows.append(features)
                    move_indices.add(idx)

            # Negative samples
            if hard_negatives:
                # Fix B: use actual losing entries as negatives
                losers = find_losing_entries(
                    ohlc_side, quotes_side, greeks_side,
                    ticker=ticker, min_move_pct=min_move_pct,
                    dte=0, expiry_date=dt,
                )
                total_losers += len(losers)

                for loser in losers:
                    idx = loser["idx"]
                    if idx in move_indices:
                        continue
                    features = feature_fn(ohlc_side, quotes_side, greeks_side, stock, idx)
                    if features:
                        features["label"] = 0
                        features["peak_pct"] = loser.get("peak_gain", 0)
                        features["pnl_pct"] = loser.get("pnl_pct", 0)
                        features["ticker"] = ticker
                        features["date"] = dt
                        all_rows.append(features)
            else:
                # Original: random non-move timestamps
                if moves and len(ohlc_side) > 30:
                    n_neg = len(moves) * neg_ratio
                    excluded = set()
                    for m in moves:
                        for offset in range(-COOLDOWN_MIN, COOLDOWN_MIN):
                            excluded.add(m["idx"] + offset)

                    candidates = [
                        i for i in range(PRE_MOVE_LOOKBACK, len(ohlc_side) - 10)
                        if i not in excluded
                    ]
                    if candidates:
                        np.random.seed(42 + hash(dt + right) % 10000)
                        neg_indices = np.random.choice(
                            candidates,
                            size=min(n_neg, len(candidates)),
                            replace=False,
                        )
                        for idx in neg_indices:
                            features = feature_fn(ohlc_side, quotes_side, greeks_side, stock, idx)
                            if features:
                                features["label"] = 0
                                features["peak_pct"] = 0
                                features["pnl_pct"] = 0
                                features["ticker"] = ticker
                                features["date"] = dt
                                all_rows.append(features)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Fix D: balance classes
    if balanced:
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        min_count = min(len(pos_df), len(neg_df))
        if min_count > 0:
            pos_df = pos_df.sample(n=min_count, random_state=42)
            neg_df = neg_df.sample(n=min_count, random_state=42)
            df = pd.concat([pos_df, neg_df]).sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Train + evaluate (same LightGBM setup, time-based split)
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, ticker: str) -> dict:
    """Train LightGBM on dataset and return metrics. Time-based 80/20 split."""
    import lightgbm as lgb
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    if len(df) < 50:
        return {"error": f"only {len(df)} samples"}

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["label"].values

    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_test) < 10 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "insufficient test data or single class"}

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 24,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "scale_pos_weight": scale_pos,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    y_prob = model.predict(X_test)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.8, 0.05):
        y_bin = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

    return {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(best_f1),
        "auc": float(auc),
        "threshold": float(best_thresh),
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Ablation experiments
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "baseline": {
        "desc": "Original (all bugs present)",
        "fix_leakage": False,
        "hard_negatives": False,
        "balanced": False,
    },
    "fix_A": {
        "desc": "Fix A: no same-candle leakage",
        "fix_leakage": True,
        "hard_negatives": False,
        "balanced": False,
    },
    "fix_B": {
        "desc": "Fix B: hard negatives (losing trades)",
        "fix_leakage": False,
        "hard_negatives": True,
        "balanced": False,
    },
    "fix_AB": {
        "desc": "Fix A+B: no leakage + hard negatives",
        "fix_leakage": True,
        "hard_negatives": True,
        "balanced": False,
    },
    "fix_ABD": {
        "desc": "Fix A+B+D: no leakage + hard neg + balanced",
        "fix_leakage": True,
        "hard_negatives": True,
        "balanced": True,
    },
    "fix_AD": {
        "desc": "Fix A+D: no leakage + balanced (no hard neg)",
        "fix_leakage": True,
        "hard_negatives": False,
        "balanced": True,
    },
}


def run_experiment(
    conn: sqlite3.Connection,
    ticker: str,
    experiment: dict,
    min_move_pct: float,
    max_days: int | None = None,
) -> dict:
    """Run a single experiment configuration and return results."""
    df = build_dataset_ablation(
        conn, ticker,
        min_move_pct=min_move_pct,
        fix_leakage=experiment["fix_leakage"],
        hard_negatives=experiment["hard_negatives"],
        balanced=experiment["balanced"],
        max_days=max_days,
    )

    if df.empty or len(df) < 50:
        return {"error": f"insufficient data ({len(df)} samples)"}

    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()

    metrics = train_and_evaluate(df, ticker)
    metrics["total_samples"] = len(df)
    metrics["positives"] = int(pos)
    metrics["negatives"] = int(neg)

    return metrics


def run_ablation(
    conn: sqlite3.Connection,
    ticker: str,
    experiments: dict | None = None,
    min_move_pct: float = 15.0,
    max_days: int | None = None,
) -> list[dict]:
    """Run all ablation experiments for one ticker at one threshold."""
    if experiments is None:
        experiments = EXPERIMENTS

    results = []
    for name, config in experiments.items():
        print(f"\n  [{name}] {config['desc']} (min_move={min_move_pct}%)")
        start = time.time()
        metrics = run_experiment(conn, ticker, config, min_move_pct, max_days)
        elapsed = time.time() - start

        row = {
            "ticker": ticker,
            "experiment": name,
            "description": config["desc"],
            "min_move_pct": min_move_pct,
            "elapsed_s": round(elapsed, 1),
        }
        row.update(metrics)
        results.append(row)

        if "error" in metrics:
            print(f"    ERROR: {metrics['error']} ({elapsed:.1f}s)")
        else:
            print(
                f"    samples={metrics['total_samples']} "
                f"(+{metrics['positives']}/-{metrics['negatives']}) "
                f"pos_rate={metrics['positive_rate']:.1%}"
            )
            print(
                f"    AUC={metrics['auc']:.3f} "
                f"F1={metrics['f1']:.3f} "
                f"Prec={metrics['precision']:.1%} "
                f"Rec={metrics['recall']:.1%} "
                f"({elapsed:.1f}s)"
            )
            if metrics.get("top_features"):
                feats = ", ".join(f"{f[0]}" for f in metrics["top_features"][:3])
                print(f"    top: {feats}")

    return results


def run_threshold_sweep(
    conn: sqlite3.Connection,
    ticker: str,
    thresholds: list[float],
    experiment_config: dict,
    experiment_name: str,
    max_days: int | None = None,
) -> list[dict]:
    """Sweep MIN_MOVE_PCT thresholds for a single experiment config."""
    results = []

    for pct in thresholds:
        print(f"\n  [{experiment_name}] threshold={pct}%")
        start = time.time()
        metrics = run_experiment(conn, ticker, experiment_config, pct, max_days)
        elapsed = time.time() - start

        row = {
            "ticker": ticker,
            "experiment": experiment_name,
            "min_move_pct": pct,
            "elapsed_s": round(elapsed, 1),
        }
        row.update(metrics)
        results.append(row)

        if "error" in metrics:
            print(f"    ERROR: {metrics['error']} ({elapsed:.1f}s)")
        else:
            print(
                f"    samples={metrics['total_samples']} "
                f"(+{metrics['positives']}/-{metrics['negatives']}) "
                f"pos_rate={metrics['positive_rate']:.1%} "
                f"AUC={metrics['auc']:.3f} F1={metrics['f1']:.3f} "
                f"({elapsed:.1f}s)"
            )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*110}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*110}")
    print(
        f"{'Experiment':<12} {'Move%':>5} {'Samples':>8} "
        f"{'Pos%':>5} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} "
        f"{'Thresh':>6} {'Time':>6}  Description"
    )
    print("-" * 110)

    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<12} {r['min_move_pct']:>5.0f} {'ERROR':>8}  {r.get('error', '')}")
            continue

        print(
            f"{r['experiment']:<12} {r['min_move_pct']:>5.0f} "
            f"{r.get('total_samples', 0):>8} "
            f"{r.get('positive_rate', 0)*100:>4.1f}% "
            f"{r.get('auc', 0):>5.3f} "
            f"{r.get('f1', 0):>5.3f} "
            f"{r.get('precision', 0)*100:>5.1f}% "
            f"{r.get('recall', 0)*100:>5.1f}% "
            f"{r.get('threshold', 0):>5.2f} "
            f"{r.get('elapsed_s', 0):>5.1f}s"
            f"  {r.get('description', '')}"
        )


def save_csv(results: list[dict], path: str) -> None:
    """Save results to CSV for further analysis."""
    if not results:
        return

    keys = ["ticker", "experiment", "description", "min_move_pct",
            "total_samples", "positives", "negatives", "positive_rate",
            "auc", "f1", "precision", "recall", "accuracy", "threshold",
            "train_samples", "test_samples", "elapsed_s"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML training ablation study")
    parser.add_argument("--ticker", default="SPY", help="Ticker to test (default: SPY)")
    parser.add_argument("--db", default=THETADATA_DB, help="Path to thetadata DB")
    parser.add_argument("--quick", action="store_true", help="Fewer thresholds for speed")
    parser.add_argument("--fix", type=str, help="Test only one fix: A, B, AB, ABD")
    parser.add_argument("--sweep", action="store_true", help="Threshold sweep with all fixes")
    parser.add_argument("--max-days", type=int, default=None, help="Limit days for speed")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    print(f"Ablation study: {args.ticker} from {args.db}")
    conn = sqlite3.connect(args.db)

    all_results = []

    if args.sweep:
        # Threshold sweep with all fixes applied
        if args.quick:
            thresholds = [15, 20, 25, 30, 35, 40, 50]
        else:
            thresholds = list(range(15, 51))

        config = EXPERIMENTS["fix_ABD"]
        print(f"\nThreshold sweep: {thresholds[0]}% to {thresholds[-1]}% ({config['desc']})")
        results = run_threshold_sweep(
            conn, args.ticker, thresholds, config, "fix_ABD", args.max_days
        )
        all_results.extend(results)

    elif args.fix:
        # Test one specific fix at the default threshold
        fix_name = f"fix_{args.fix}" if not args.fix.startswith("fix_") else args.fix
        if fix_name not in EXPERIMENTS:
            print(f"Unknown fix: {args.fix}. Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)

        results = run_ablation(
            conn, args.ticker,
            experiments={"baseline": EXPERIMENTS["baseline"], fix_name: EXPERIMENTS[fix_name]},
            min_move_pct=15.0,
            max_days=args.max_days,
        )
        all_results.extend(results)

    else:
        # Full ablation: all experiments at 15%, then sweep with best config
        print(f"\n{'='*60}")
        print(f"PHASE 1: Ablation at 15% threshold")
        print(f"{'='*60}")

        results = run_ablation(conn, args.ticker, min_move_pct=15.0, max_days=args.max_days)
        all_results.extend(results)

        print(f"\n{'='*60}")
        print(f"PHASE 2: Threshold sweep with all fixes (fix_ABD)")
        print(f"{'='*60}")

        if args.quick:
            thresholds = [15, 20, 25, 30, 35, 40, 50]
        else:
            thresholds = list(range(15, 51))

        config = EXPERIMENTS["fix_ABD"]
        sweep_results = run_threshold_sweep(
            conn, args.ticker, thresholds, config, "fix_ABD", args.max_days
        )
        all_results.extend(sweep_results)

    conn.close()

    print_report(all_results)

    # Save CSV
    output_path = args.output or f"journal/ablation_{args.ticker}_{datetime.now():%Y%m%d_%H%M}.csv"
    save_csv(all_results, output_path)

    # Find optimal threshold from sweep results
    sweep_rows = [r for r in all_results if r["experiment"] == "fix_ABD" and "error" not in r]
    if sweep_rows:
        # Sort by F1 then AUC to find optimal
        best = max(sweep_rows, key=lambda r: (r.get("f1", 0), r.get("auc", 0)))
        print(f"\n{'='*60}")
        print(f"OPTIMAL THRESHOLD: {best['min_move_pct']}%")
        print(f"  AUC={best['auc']:.3f} F1={best['f1']:.3f} "
              f"Prec={best['precision']:.1%} Rec={best['recall']:.1%}")
        print(f"  Samples: {best['total_samples']} "
              f"(+{best['positives']}/-{best['negatives']}) "
              f"pos_rate={best['positive_rate']:.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
