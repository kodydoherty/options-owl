"""Train per-ticker ML sell timing models + a generic fallback.

For each ticker with enough data (MIN_TRAINABLE_DAYS), trains a dedicated
classifier + regressor pair. Tickers with insufficient data fall back to
the generic model trained on all tickers combined.

Output: journal/models/{ticker}_clf.lgb and {ticker}_reg.lgb for each ticker,
        plus generic_clf.lgb and generic_reg.lgb as fallback.

Usage:
    python scripts/train_per_ticker_models.py
    python scripts/train_per_ticker_models.py --ticker SPY   # train just one
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not installed: pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Reuse feature extraction from the main training script
from scripts.ml_sell_model import (
    DB_PATH, FEATURE_COLS, MIN_OPTION_BARS, TICKERS as OLD_TICKERS,
    extract_features_for_trade, find_deadline_idx, find_entry_idx,
    get_db_connection, load_all_trading_days, load_option_bars,
    load_underlying_bars, ts_to_et,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
MODELS_DIR = os.path.join(PROJECT_DIR, "journal", "models")

MIN_TRAINABLE_DAYS = 20  # need at least 20 days with option bars

SELL_THRESHOLD = 0.4

# Per-ticker models don't need ticker_encoded feature
PER_TICKER_FEATURES = [f for f in FEATURE_COLS if f != "ticker_encoded"]


def get_all_tickers_with_data():
    """Find all tickers that have enough trainable days."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT ticker,
               SUM(CASE WHEN call_bars >= ? AND put_bars >= ? THEN 1 ELSE 0 END) as good_days
        FROM trading_days
        GROUP BY ticker
        HAVING good_days >= ?
        ORDER BY good_days DESC
    """, (MIN_OPTION_BARS, MIN_OPTION_BARS, MIN_TRAINABLE_DAYS)).fetchall()
    conn.close()
    return [(r["ticker"], r["good_days"]) for r in rows]


def build_ticker_dataset(ticker):
    """Build feature dataset for a single ticker."""
    conn = get_db_connection()

    days = conn.execute("""
        SELECT ticker, date, open_price, close_price, high_price, low_price,
               atm_call_ticker, atm_put_ticker, atm_strike,
               call_bars, put_bars, underlying_bars
        FROM trading_days
        WHERE ticker = ? AND call_bars >= ? AND put_bars >= ?
        ORDER BY date
    """, (ticker, MIN_OPTION_BARS, MIN_OPTION_BARS)).fetchall()
    days = [dict(d) for d in days]

    all_rows = []
    trades_meta = []
    processed = 0

    for day in days:
        date = day["date"]
        u_bars = load_underlying_bars(conn, ticker, date)

        for direction, contract_key in [("call", "atm_call_ticker"),
                                        ("put", "atm_put_ticker")]:
            contract_ticker = day[contract_key]
            if not contract_ticker:
                continue

            o_bars = load_option_bars(conn, contract_ticker)
            if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                continue

            entry_idx = find_entry_idx(o_bars)
            deadline_idx = find_deadline_idx(o_bars)
            if entry_idx is None or deadline_idx is None:
                continue

            is_call = direction == "call"
            rows = extract_features_for_trade(
                o_bars, u_bars, entry_idx, deadline_idx,
                0,  # ticker_code not used for per-ticker
                is_call,
            )
            if rows is None or len(rows) < 10:
                continue

            start_idx = len(all_rows)
            all_rows.extend(rows)
            trades_meta.append({
                "date": date,
                "ticker": ticker,
                "direction": direction,
                "num_rows": len(rows),
                "start_idx": start_idx,
                "end_idx": start_idx + len(rows),
            })
            processed += 1

    conn.close()

    if not all_rows:
        return None, None

    df = pd.DataFrame(all_rows)
    trades_df = pd.DataFrame(trades_meta)
    return df, trades_df


def train_lgb_pair(df, trades_df, feature_cols, label=""):
    """Train classifier + regressor pair. Returns (clf, reg) or (None, None)."""
    unique_dates = sorted(trades_df["date"].unique())
    n_dates = len(unique_dates)

    if n_dates < 10:
        print(f"    Only {n_dates} dates — too few to train")
        return None, None

    train_end = int(n_dates * 0.70)
    val_end = int(n_dates * 0.85)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    # Map rows to dates
    row_dates = []
    for _, trade in trades_df.iterrows():
        row_dates.extend([trade["date"]] * trade["num_rows"])
    row_dates = np.array(row_dates)

    train_mask = np.isin(row_dates, list(train_dates))
    val_mask = np.isin(row_dates, list(val_dates))
    test_mask = np.isin(row_dates, list(test_dates))

    X = df[feature_cols]
    y = df["should_sell"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) < 100 or len(X_val) < 50:
        print(f"    Insufficient samples: train={len(X_train)}, val={len(X_val)}")
        return None, None

    print(f"    Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"    Dates: {unique_dates[0]} to {unique_dates[-1]}")

    # Classifier
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.01,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "min_child_samples": 50,  # less conservative for per-ticker (less data)
        "max_depth": 8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    clf = lgb.train(
        clf_params, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(80)],
    )

    # Regressor
    y_reg_train = df.loc[train_mask, "expected_future_pnl"]
    y_reg_val = df.loc[val_mask, "expected_future_pnl"]

    dtrain_reg = lgb.Dataset(X_train, label=y_reg_train)
    dval_reg = lgb.Dataset(X_val, label=y_reg_val, reference=dtrain_reg)

    reg_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.01,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "min_child_samples": 50,
        "max_depth": 8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    reg = lgb.train(
        reg_params, dtrain_reg,
        num_boost_round=2000,
        valid_sets=[dtrain_reg, dval_reg],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(80)],
    )

    # Evaluate
    test_pred_proba = clf.predict(X_test, num_iteration=clf.best_iteration)
    test_pred = (test_pred_proba > SELL_THRESHOLD).astype(int)

    test_reg_pred = reg.predict(X_test, num_iteration=reg.best_iteration)
    test_reg_actual = df.loc[test_mask, "expected_future_pnl"].values
    mae = np.mean(np.abs(test_reg_pred - test_reg_actual))

    wins = y_test.sum()
    total = len(y_test)
    sell_pct = wins / total * 100 if total > 0 else 0

    print(f"    CLF best iter: {clf.best_iteration} | REG best iter: {reg.best_iteration}")
    print(f"    REG MAE: {mae:.2f}% | Labels: {sell_pct:.0f}% sell")

    if len(X_test) > 0:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, test_pred)
        print(f"    CLF accuracy: {acc:.1%}")

    return clf, reg


def build_generic_dataset(all_tickers):
    """Build combined dataset across all tickers for generic model."""
    print("\n  Building generic dataset across all tickers...")

    all_rows = []
    trades_meta = []
    ticker_encoder = LabelEncoder()
    ticker_names = [t for t, _ in all_tickers]
    ticker_encoder.fit(ticker_names)

    conn = get_db_connection()

    for ticker, _ in all_tickers:
        days = conn.execute("""
            SELECT ticker, date, open_price, close_price, high_price, low_price,
                   atm_call_ticker, atm_put_ticker, atm_strike,
                   call_bars, put_bars, underlying_bars
            FROM trading_days
            WHERE ticker = ? AND call_bars >= ? AND put_bars >= ?
            ORDER BY date
        """, (ticker, MIN_OPTION_BARS, MIN_OPTION_BARS)).fetchall()
        days = [dict(d) for d in days]

        ticker_code = int(ticker_encoder.transform([ticker])[0])

        for day in days:
            date = day["date"]
            u_bars = load_underlying_bars(conn, ticker, date)

            for direction, contract_key in [("call", "atm_call_ticker"),
                                            ("put", "atm_put_ticker")]:
                contract_ticker = day[contract_key]
                if not contract_ticker:
                    continue

                o_bars = load_option_bars(conn, contract_ticker)
                if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                    continue

                entry_idx = find_entry_idx(o_bars)
                deadline_idx = find_deadline_idx(o_bars)
                if entry_idx is None or deadline_idx is None:
                    continue

                is_call = direction == "call"
                rows = extract_features_for_trade(
                    o_bars, u_bars, entry_idx, deadline_idx,
                    ticker_code, is_call,
                )
                if rows is None or len(rows) < 10:
                    continue

                start_idx = len(all_rows)
                all_rows.extend(rows)
                trades_meta.append({
                    "date": date,
                    "ticker": ticker,
                    "direction": direction,
                    "num_rows": len(rows),
                    "start_idx": start_idx,
                    "end_idx": start_idx + len(rows),
                })

    conn.close()

    if not all_rows:
        return None, None, ticker_encoder

    df = pd.DataFrame(all_rows)
    trades_df = pd.DataFrame(trades_meta)
    return df, trades_df, ticker_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Train just one ticker")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    start_time = time.time()
    print("=" * 70)
    print("  OptionsOwl — Per-Ticker ML Sell Timing Models")
    print("=" * 70)
    print(f"  Database: {DB_PATH}")
    print(f"  Output:   {MODELS_DIR}")
    print()

    if not os.path.exists(DB_PATH):
        print(f"  ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    # Find all trainable tickers
    all_tickers = get_all_tickers_with_data()
    print(f"  Found {len(all_tickers)} tickers with >= {MIN_TRAINABLE_DAYS} trainable days:")
    for t, d in all_tickers:
        print(f"    {t:<8} {d:>4} days")
    print()

    if args.ticker:
        all_tickers = [(t, d) for t, d in all_tickers if t == args.ticker.upper()]
        if not all_tickers:
            print(f"  ERROR: {args.ticker} not found or insufficient data")
            sys.exit(1)

    # Train per-ticker models
    trained = []
    failed = []

    for ticker, num_days in all_tickers:
        print(f"\n{'='*60}")
        print(f"  Training {ticker} ({num_days} days)")
        print(f"{'='*60}")

        t0 = time.time()
        df, trades_df = build_ticker_dataset(ticker)

        if df is None or len(df) < 500:
            print(f"    Skipping {ticker}: insufficient rows ({len(df) if df is not None else 0})")
            failed.append(ticker)
            continue

        print(f"    Dataset: {len(df)} rows, {len(trades_df)} trades")

        clf, reg = train_lgb_pair(df, trades_df, PER_TICKER_FEATURES, label=ticker)

        if clf is None:
            failed.append(ticker)
            continue

        # Save
        clf_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_clf.lgb")
        reg_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_reg.lgb")
        clf.save_model(clf_path)
        reg.save_model(reg_path)

        elapsed = time.time() - t0
        print(f"    Saved: {clf_path}")
        print(f"    Time: {elapsed:.0f}s")
        trained.append(ticker)

    # Train generic model (all tickers combined)
    if not args.ticker:
        print(f"\n{'='*60}")
        print(f"  Training GENERIC model (all {len(all_tickers)} tickers)")
        print(f"{'='*60}")

        t0 = time.time()
        df, trades_df, ticker_encoder = build_generic_dataset(all_tickers)

        if df is not None and len(df) > 1000:
            print(f"    Dataset: {len(df)} rows, {len(trades_df)} trades")
            clf, reg = train_lgb_pair(df, trades_df, FEATURE_COLS, label="generic")

            if clf is not None:
                clf.save_model(os.path.join(MODELS_DIR, "generic_clf.lgb"))
                reg.save_model(os.path.join(MODELS_DIR, "generic_reg.lgb"))
                print(f"    Saved generic model ({time.time()-t0:.0f}s)")
                trained.append("GENERIC")

    # Update PER_TICKER_MODELS in ml_exit.py
    if trained and not args.ticker:
        update_per_ticker_set(trained)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trained: {', '.join(trained)} ({len(trained)} models)")
    if failed:
        print(f"  Failed:  {', '.join(failed)} (will use generic)")
    print(f"  Time:    {elapsed/60:.1f} minutes")
    print(f"  Models:  {MODELS_DIR}")
    print(f"{'='*60}")


def update_per_ticker_set(trained_tickers):
    """Update PER_TICKER_MODELS in ml_exit.py with newly trained tickers."""
    ml_exit_path = os.path.join(PROJECT_DIR, "options_owl", "risk", "ml_exit.py")
    if not os.path.exists(ml_exit_path):
        return

    ticker_set = [t for t in trained_tickers if t != "GENERIC"]
    new_line = f'PER_TICKER_MODELS = {{{", ".join(repr(t) for t in sorted(ticker_set))}}}'

    with open(ml_exit_path, "r") as f:
        content = f.read()

    import re
    content = re.sub(
        r"PER_TICKER_MODELS = \{[^}]*\}",
        new_line,
        content,
    )

    with open(ml_exit_path, "w") as f:
        f.write(content)

    print(f"  Updated ml_exit.py: PER_TICKER_MODELS = {ticker_set}")


if __name__ == "__main__":
    main()
