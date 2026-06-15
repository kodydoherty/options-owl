"""V3 ML Model Suite — 6 specialized models for the full trade lifecycle.

Models:
  1. entry_timing   — Predict the LOW before a +38% run (optimal entry point)
  2. exit_timing    — Hold vs sell at any point in an active trade
  3. regime         — Is today a trending day or a chop day?
  4. ticker_select  — Which tickers to focus on today?
  5. stop_calibrate — Optimal stop width for this specific entry
  6. signal_quality — Predict magnitude of move (regression, not binary)

Usage:
    python scripts/train_ml_models_v3.py                    # train all 6 models
    python scripts/train_ml_models_v3.py --model exit_timing  # single model
    python scripts/train_ml_models_v3.py --model entry_timing --ticker SPY
    python scripts/train_ml_models_v3.py --evaluate         # evaluate all models
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402
from options_owl.sourcing.features.regime_features import (  # noqa: E402
    REGIME_FEATURE_ORDER,
    compute_regime_feature_vector,
    load_training_inputs,
    rth_bars_by_date_from_rows,
)

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    # New tickers (added 2026-05-28)
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

N_WORKERS = min(os.cpu_count() or 4, 16)

# Production V6 settings
_V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True,
    V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True,
    V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333,
    V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True,
    V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15,
    ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_PREMIUM_CAP=True,
    V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True,
    V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

MIN_MOVE_PCT = 38.0
MOVE_WINDOW_MIN = 120
PRE_MOVE_LOOKBACK = 15
COOLDOWN_MIN = 30

TICKER_MOVE_PCT = {
    "SPY": 38.0, "QQQ": 38.0, "IWM": 38.0,
    "TSLA": 45.0, "MSTR": 45.0, "AMD": 45.0,
    "NVDA": 40.0, "AVGO": 40.0, "META": 40.0,
    "AAPL": 35.0, "MSFT": 35.0, "GOOGL": 35.0,
    "AMZN": 38.0, "PLTR": 45.0,
}


# ===========================================================================
# Shared: date-based walk-forward splitting (NEVER split by row)
# ===========================================================================
#
# All samples within a trading day are highly correlated (same underlying
# path, overlapping forward-looking label windows). A shuffled
# train_test_split leaks test-day information into training and wildly
# inflates AUC. We split by DATE only, using the proven expanding-window
# walk-forward harness from scripts/train_put_pattern.py (~565-628):
#   - fold k: train on months[0..k-1], test on months[k]
#   - final production model: train on all but the last month, test on the
#     last month
# Models whose labels look FORWARD in time (entry_timing +120min window,
# exit_timing +10min window, signal_quality +120min window, stop_calibration
# forward simulation) additionally get a 1-day embargo: the last train date
# immediately preceding the test period is dropped so a label window cannot
# touch the test period.


def _apply_embargo(train_mask: pd.Series, df: pd.DataFrame, embargo_days: int) -> pd.Series:
    """Drop the last `embargo_days` distinct train dates (the ones adjacent to
    the test period) from the train mask."""
    if embargo_days <= 0:
        return train_mask
    train_dates = sorted(df.loc[train_mask, "date"].unique())
    if len(train_dates) <= embargo_days:
        return train_mask
    embargo_dates = set(train_dates[-embargo_days:])
    return train_mask & ~df["date"].isin(embargo_dates)


def _walk_forward_validate(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    params: dict,
    is_regression: bool,
    embargo_days: int = 0,
    num_boost_round: int = 500,
) -> list[float]:
    """Expanding-window walk-forward by month. Returns per-fold scores
    (AUC for classification, MAE for regression)."""
    months = sorted(set(str(d)[:7] for d in df["date"].unique()))
    month_series = df["date"].astype(str).str[:7]
    print(f"  Months available: {months}")
    print("  Walk-forward folds (expanding train window → test next month):")

    fold_scores: list[float] = []
    for fold_idx in range(2, len(months)):
        train_months = set(months[:fold_idx])
        test_month = months[fold_idx]
        train_mask = month_series.isin(train_months)
        test_mask = month_series == test_month
        train_mask = _apply_embargo(train_mask, df, embargo_days)

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        X_tr, y_tr = X[train_mask.values], y[train_mask.values]
        X_te, y_te = X[test_mask.values], y[test_mask.values]

        fold_params = dict(params)
        if not is_regression:
            if len(set(y_te)) < 2:
                print(f"    Fold {fold_idx - 1}: SKIPPED — test month {test_month} "
                      "is single-class (AUC undefined)")
                continue
            fold_params["scale_pos_weight"] = float(
                (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            )

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dtest = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        fold_model = lgb.train(
            fold_params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        fold_preds = fold_model.predict(X_te)

        if is_regression:
            score = float(mean_absolute_error(y_te, fold_preds))
            print(f"    Fold {fold_idx - 1}: train {months[0]}→{months[fold_idx - 1]} | "
                  f"test {test_month} | MAE={score:.3f} (n={len(X_te)})")
        else:
            score = float(roc_auc_score(y_te, fold_preds))
            print(f"    Fold {fold_idx - 1}: train {months[0]}→{months[fold_idx - 1]} | "
                  f"test {test_month} | AUC={score:.4f} (n={len(X_te)})")
        fold_scores.append(score)

    if fold_scores:
        metric = "MAE" if is_regression else "AUC"
        print(f"  Walk-forward {metric}: {np.mean(fold_scores):.4f} +/- "
              f"{np.std(fold_scores):.4f} "
              f"(min={min(fold_scores):.4f}, max={max(fold_scores):.4f})")
    return fold_scores


def _final_date_split(df: pd.DataFrame, embargo_days: int = 0) -> tuple[pd.Series, pd.Series]:
    """Final production split: train on all but the last month, test on the
    last month (by DATE, never by row). Falls back to the last 20% of distinct
    dates if fewer than 3 months of data exist."""
    months = sorted(set(str(d)[:7] for d in df["date"].unique()))
    month_series = df["date"].astype(str).str[:7]
    if len(months) >= 3:
        test_mask = month_series == months[-1]
    else:
        dates = sorted(df["date"].unique())
        cutoff = dates[max(1, int(len(dates) * 0.8))]
        test_mask = df["date"] >= cutoff
    train_mask = ~test_mask
    train_mask = _apply_embargo(train_mask, df, embargo_days)
    return train_mask, test_mask


# ===========================================================================
# Shared: robust validation for SMALL day-level datasets
# ===========================================================================
#
# regime_classifier and ticker_selection have ~1 sample per ticker-day
# (~2k rows total on the current 126-day DB). A naive last-month holdout can
# end up single-class (AUC=NaN, accuracy=1.0 — exactly what happened to
# ticker_selection). For these models we instead:
#   1. Try an expanding date-based holdout: start with the last ~20% of
#      distinct dates and grow it (up to 40%) until the test set contains
#      BOTH classes (ideally >= 5 of the minority) and a sane minimum size.
#   2. If no such holdout exists, fall back to stratified-by-label k-fold
#      over DATES (purged + embargoed) and report mean +/- std AUC.
# AUC is always guarded: single-class folds are skipped and logged, never
# emitted as NaN.


def _safe_auc(y_true, preds) -> float | None:
    """AUC that returns None (instead of raising / NaN) on single-class y."""
    if len(np.unique(np.asarray(y_true))) < 2:
        return None
    return float(roc_auc_score(y_true, preds))


def _robust_final_date_split(
    df: pd.DataFrame,
    label_col: str = "label",
    embargo_days: int = 0,
    min_test_frac: float = 0.2,
    max_test_frac: float = 0.4,
    min_test_samples: int = 30,
    min_minority: int = 5,
) -> tuple[pd.Series, pd.Series] | None:
    """Date-based final holdout that GUARANTEES a multi-class test set.

    Sizes the test window by ROW fraction but cuts only on DATE boundaries
    (never splits a date — per-ticker date coverage can be very uneven, e.g.
    SPY has 4x the history of other tickers). Starts at the latest cutoff
    where the test set holds >= min_test_frac of rows, then expands earlier
    (up to max_test_frac of rows) until it has >= min_minority samples of
    each class (second pass relaxes to >= 1). Returns (train_mask, test_mask)
    or None if even the largest allowed window is single-class (caller should
    fall back to k-fold CV).
    """
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    n_rows = len(df)
    if n_dates < 10:
        return None

    rows_per_date = df["date"].value_counts()
    min_test_rows = max(min_test_samples, int(round(n_rows * min_test_frac)))
    max_test_rows = int(round(n_rows * max_test_frac))

    # Latest cutoff where the test window reaches min_test_rows
    test_rows = 0
    start_cut = None
    for cut in range(n_dates - 1, 0, -1):
        test_rows += int(rows_per_date.get(dates[cut], 0))
        if test_rows >= min_test_rows:
            start_cut = cut
            break
    if start_cut is None or test_rows > max_test_rows:
        return None

    for required_minority in (min_minority, 1):
        for cut in range(start_cut, 0, -1):
            cutoff = dates[cut]
            test_mask = df["date"] >= cutoff
            n_test = int(test_mask.sum())
            if n_test > max_test_rows:
                break  # expanding further only makes the test set bigger
            y_te = df.loc[test_mask, label_col].values.astype(int)
            counts = np.bincount(y_te, minlength=2)
            if counts.min() >= required_minority:
                train_mask = _apply_embargo(~test_mask, df, embargo_days)
                print(f"  Final holdout: last {n_dates - cut}/{n_dates} distinct dates "
                      f"(>= {cutoff}) | test n={n_test} "
                      f"class_counts={counts.tolist()}")
                return train_mask, test_mask
    print("  No multi-class date holdout found (even at "
          f"{max_test_frac:.0%} of rows) — will fall back to k-fold CV")
    return None


def _stratified_date_kfold_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    params: dict,
    label_col: str = "label",
    n_splits: int = 5,
    embargo_days: int = 1,
    num_boost_round: int = 300,
) -> list[float]:
    """Fallback CV when no valid date holdout exists.

    Folds are built over DATES (never rows): dates are sorted by their
    positive-label rate and dealt round-robin into folds, so every fold gets
    a representative label mix (stratified-by-label over dates). Train dates
    within `embargo_days` positions of any test date are purged (embargo).
    Single-class folds are skipped and logged — never NaN.
    """
    dates = sorted(df["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    date_rates = df.groupby("date")[label_col].mean()
    ordered = sorted(dates, key=lambda d: (date_rates[d], d))
    folds: list[list] = [[] for _ in range(n_splits)]
    for i, d in enumerate(ordered):
        folds[i % n_splits].append(d)

    print(f"  Stratified-by-label date k-fold (k={n_splits}, "
          f"embargo_days={embargo_days}, purged):")
    scores: list[float] = []
    for k, test_dates in enumerate(folds):
        test_set = set(test_dates)
        banned = set()
        for d in test_dates:
            i = date_pos[d]
            for off in range(-embargo_days, embargo_days + 1):
                j = i + off
                if 0 <= j < len(dates):
                    banned.add(dates[j])
        test_mask = df["date"].isin(test_set)
        train_mask = ~df["date"].isin(banned | test_set)
        if test_mask.sum() < 10 or train_mask.sum() < 50:
            print(f"    Fold {k + 1}: SKIPPED (too few samples)")
            continue
        y_tr = y[train_mask.values]
        y_te = y[test_mask.values]
        if len(np.unique(y_tr)) < 2:
            print(f"    Fold {k + 1}: SKIPPED (single-class train set)")
            continue
        if len(np.unique(y_te)) < 2:
            print(f"    Fold {k + 1}: SKIPPED (single-class test set)")
            continue

        fold_params = dict(params)
        fold_params["scale_pos_weight"] = float(
            (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        )
        dtrain = lgb.Dataset(X[train_mask.values], label=y_tr, feature_name=feature_cols)
        dtest = lgb.Dataset(X[test_mask.values], label=y_te, reference=dtrain)
        fold_model = lgb.train(
            fold_params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = fold_model.predict(X[test_mask.values])
        auc = _safe_auc(y_te, preds)
        if auc is None:
            print(f"    Fold {k + 1}: SKIPPED (AUC undefined)")
            continue
        print(f"    Fold {k + 1}: AUC={auc:.4f} (n_test={int(test_mask.sum())}, "
              f"test_dates={len(test_dates)})")
        scores.append(auc)

    if scores:
        print(f"  K-fold AUC: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
              f"({len(scores)}/{n_splits} valid folds)")
    return scores


def _train_day_level_model(
    df: pd.DataFrame,
    feature_cols: list,
    params: dict,
    model_name: str,
    label_col: str = "label",
    embargo_days: int = 1,
    num_boost_round: int = 300,
):
    """Train + robustly evaluate a small day-level binary classifier.

    Runs the informational expanding walk-forward, then either a guaranteed
    multi-class date holdout or (fallback) stratified date k-fold.
    Returns (model, meta) or None if training is impossible.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    if len(np.unique(y)) < 2:
        print(f"  {model_name}: label is single-class over the WHOLE dataset — "
              "cannot train. Check the label definition.")
        return None

    # Informational expanding-window walk-forward by month
    wf_scores = _walk_forward_validate(
        df, X, y, feature_cols, params, is_regression=False,
        embargo_days=embargo_days, num_boost_round=num_boost_round,
    )

    meta: dict = {
        "features": feature_cols,
        "split": "date_only_never_row",
        "embargo_days": embargo_days,
        "walk_forward_folds": len(wf_scores),
        "walk_forward_auc_mean": float(np.mean(wf_scores)) if wf_scores else None,
        "walk_forward_auc_std": float(np.std(wf_scores)) if wf_scores else None,
    }

    split = _robust_final_date_split(df, label_col=label_col, embargo_days=embargo_days)
    if split is not None:
        train_mask, test_mask = split
        X_train, y_train = X[train_mask.values], y[train_mask.values]
        X_test, y_test = X[test_mask.values], y[test_mask.values]
        print(f"  Final split: train={len(X_train)} test={len(X_test)} (by date)")

        fit_params = dict(params)
        fit_params["scale_pos_weight"] = float(
            (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        )
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        model = lgb.train(fit_params, dtrain, num_boost_round=num_boost_round,
                          valid_sets=[dtest], callbacks=[lgb.log_evaluation(100)])

        preds = model.predict(X_test)
        auc = _safe_auc(y_test, preds)
        if auc is None:  # cannot happen by construction, but never emit NaN
            print("  WARNING: holdout unexpectedly single-class — "
                  "falling back to k-fold CV")
        else:
            pred_labels = (preds > 0.5).astype(int)
            meta.update({
                "validation": "expanding_date_holdout",
                "auc": auc,
                "accuracy": float(accuracy_score(y_test, pred_labels)),
                "precision": float(precision_score(y_test, pred_labels, zero_division=0)),
                "recall": float(recall_score(y_test, pred_labels, zero_division=0)),
                "n_train": len(X_train),
                "n_test": len(X_test),
            })
            print(f"\n  {model_name}: AUC={auc:.3f} Acc={meta['accuracy']:.3f} "
                  f"Prec={meta['precision']:.3f} Recall={meta['recall']:.3f}")
            return model, meta

    # Fallback: stratified-by-label k-fold over dates (purged, embargoed).
    cv_scores = _stratified_date_kfold_cv(
        df, X, y, feature_cols, params, label_col=label_col,
        embargo_days=max(embargo_days, 1), num_boost_round=num_boost_round,
    )
    if not cv_scores:
        print(f"  {model_name}: no valid CV folds — not saving a model.")
        return None

    # Final production model trains on ALL data (no holdout exists);
    # honest metrics come from the k-fold CV.
    fit_params = dict(params)
    fit_params["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model = lgb.train(fit_params, dtrain, num_boost_round=num_boost_round)
    meta.update({
        "validation": "stratified_date_kfold",
        "auc": float(np.mean(cv_scores)),
        "auc_std": float(np.std(cv_scores)),
        "cv_folds_valid": len(cv_scores),
        "n_train": len(X),
        "n_test": 0,
    })
    print(f"\n  {model_name}: CV AUC={meta['auc']:.3f} +/- {meta['auc_std']:.3f} "
          f"(final model trained on all {len(X)} samples)")
    return model, meta


# ===========================================================================
# Shared: DB loading + FSM simulation (reused from v2)
# ===========================================================================

def _connect_theta():
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _connect_uw():
    conn = sqlite3.connect(UW_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def find_atm_strike(conn, ticker, dt, right):
    row = conn.execute(
        "SELECT underlying_price FROM option_greeks "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? AND underlying_price > 0 "
        "ORDER BY timestamp LIMIT 1",
        (ticker, f"{dt}%", right),
    ).fetchone()
    if not row:
        return None
    underlying = row[0]
    strikes = [r[0] for r in conn.execute(
        "SELECT DISTINCT strike FROM option_ohlc "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? ORDER BY strike",
        (ticker, f"{dt}%", right),
    ).fetchall()]
    if not strikes:
        return None
    if right.upper() == "CALL":
        otm = [s for s in strikes if s >= underlying]
        return min(otm, key=lambda s: s - underlying) if otm else min(strikes, key=lambda s: abs(s - underlying))
    else:
        otm = [s for s in strikes if s <= underlying]
        return max(otm, key=lambda s: underlying - s) if otm else min(strikes, key=lambda s: abs(s - underlying))


def load_day_data(conn, ticker, dt, right, strike):
    """Load OHLC, quotes, greeks, stock for one ticker/date/right/strike."""
    ohlc = pd.read_sql_query(
        "SELECT * FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    quotes = pd.read_sql_query(
        "SELECT * FROM option_quotes WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    greeks = pd.read_sql_query(
        "SELECT * FROM option_greeks WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    stock = pd.read_sql_query(
        "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%"),
    )
    return ohlc, quotes, greeks, stock


def simulate_fsm(ohlc, quotes, greeks, entry_idx, ticker, dte=0, expiry_date="", contracts=5):
    """Run production V5 FSM from entry_idx. Returns full trajectory + result."""
    if entry_idx >= len(ohlc) - 10:
        return None

    entry_price = ohlc.iloc[entry_idx]["close"]
    if not entry_price or entry_price <= 0 or (isinstance(entry_price, float) and np.isnan(entry_price)):
        return None

    right_val = str(ohlc.iloc[entry_idx].get("right", "CALL")).upper()
    option_type = "call" if right_val == "CALL" else "put"

    entry_ts_raw = ohlc.iloc[entry_idx]["timestamp"]
    entry_ts = pd.Timestamp(entry_ts_raw)
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.tz_localize(None)
    entry_ts = entry_ts.to_pydatetime()

    first_underlying = 0.0
    if len(greeks) > entry_idx:
        u = greeks.iloc[entry_idx].get("underlying_price", 0)
        if u and u > 0:
            first_underlying = float(u)

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_price, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_price,
        entry_underlying_price=first_underlying, dte=dte,
        expiry_date=expiry_date,
    )

    trajectory = []  # (minute_offset, premium, pnl_pct, underlying, bid, ask)
    locked_pnl = 0.0
    remaining = contracts
    end_idx = min(entry_idx + MOVE_WINDOW_MIN, len(ohlc))

    for idx in range(entry_idx + 1, end_idx):
        row = ohlc.iloc[idx]
        premium = row.get("close", 0)
        if not premium or premium <= 0 or (isinstance(premium, float) and np.isnan(premium)):
            continue

        bid, ask = premium, premium
        if len(quotes) > idx:
            q = quotes.iloc[idx]
            b = q.get("bid", 0)
            a = q.get("ask", 0)
            if b and not (isinstance(b, float) and np.isnan(b)) and b > 0:
                bid = float(b)
            if a and not (isinstance(a, float) and np.isnan(a)) and a > 0:
                ask = float(a)

        now_raw = row["timestamp"]
        now = pd.Timestamp(now_raw)
        if now.tzinfo is not None:
            now = now.tz_localize(None)
        now = now.to_pydatetime()

        underlying = first_underlying
        if len(greeks) > idx:
            u = greeks.iloc[idx].get("underlying_price", 0)
            if u and u > 0:
                underlying = float(u)

        elapsed = (now - entry_ts).total_seconds() / 60
        pnl_pct = (premium / entry_price - 1) * 100

        trajectory.append({
            "minute": elapsed,
            "premium": premium,
            "pnl_pct": pnl_pct,
            "underlying": underlying,
            "bid": bid,
            "ask": ask,
            "idx": idx,
        })

        et_hour = now.hour
        if et_hour >= 13:
            et_hour = now.hour - 4
            if et_hour < 0:
                et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (premium - entry_price) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            peak_gain = (state.peak_premium - entry_price) / entry_price * 100
            pnl = locked_pnl + (premium - entry_price) * remaining * 100

            return {
                "pnl_pct": pnl_pct,
                "pnl_dollars": pnl,
                "reason": action.reason.value,
                "hold_minutes": elapsed,
                "exit_premium": premium,
                "peak_gain": peak_gain,
                "trajectory": trajectory,
                "entry_price": entry_price,
                "entry_idx": entry_idx,
            }

    last_row = ohlc.iloc[end_idx - 1]
    last_prem = last_row.get("close", entry_price)
    if not last_prem or (isinstance(last_prem, float) and np.isnan(last_prem)):
        last_prem = entry_price
    peak_gain = (state.peak_premium - entry_price) / entry_price * 100
    pnl = locked_pnl + (last_prem - entry_price) * remaining * 100

    return {
        "pnl_pct": (last_prem / entry_price - 1) * 100,
        "pnl_dollars": pnl,
        "reason": "eod_data_end",
        "hold_minutes": (end_idx - entry_idx),
        "exit_premium": last_prem,
        "peak_gain": peak_gain,
        "trajectory": trajectory,
        "entry_price": entry_price,
        "entry_idx": entry_idx,
    }


def compute_pre_entry_features(ohlc, quotes, greeks, stock, idx, lookback=15):
    """Features describing what the market looks like BEFORE entry at idx."""
    if idx < lookback + 1 or idx >= len(ohlc):
        return None

    curr = ohlc.iloc[idx - 1]
    window = ohlc.iloc[max(0, idx - lookback - 1):idx]
    entry_price = curr.get("close", 0) or 0
    if entry_price <= 0:
        return None

    f = {}

    # Time of day
    try:
        ts = pd.Timestamp(curr["timestamp"])
        if ts.tzinfo:
            ts = ts.tz_convert("America/New_York")
        f["minutes_since_open"] = max(0, (ts.hour - 9) * 60 + ts.minute - 30)
    except Exception:
        f["minutes_since_open"] = 0
    f["hour_bucket"] = f["minutes_since_open"] // 60
    f["is_first_30min"] = 1 if f["minutes_since_open"] <= 30 else 0

    # Premium action
    prices = window["close"].dropna().values
    if len(prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = float((prices[-1] / prices[max(-6, -len(prices))] - 1) * 100) if prices[max(-6, -len(prices))] > 0 else 0
    f["premium_change_10m"] = float((prices[-1] / prices[max(-11, -len(prices))] - 1) * 100) if prices[max(-11, -len(prices))] > 0 else 0
    f["premium_change_15m"] = float((prices[-1] / prices[0] - 1) * 100) if prices[0] > 0 else 0

    if len(prices) > 2 and all(prices[:-1] > 0):
        returns = np.diff(prices) / prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    # Volume
    vols = window["volume"].fillna(0).values if "volume" in window.columns else np.zeros(len(window))
    f["current_volume"] = float(vols[-1])
    avg_vol = float(np.mean(vols[:-1])) if len(vols) > 1 else 1
    f["volume_ratio"] = float(vols[-1] / max(avg_vol, 1))
    if len(vols) > 5 and np.std(vols[:-1]) > 0:
        f["volume_zscore"] = float((vols[-1] - np.mean(vols[:-1])) / np.std(vols[:-1]))
    else:
        f["volume_zscore"] = 0

    # Bid/ask dynamics
    if len(quotes) > idx - 1:
        q_window = quotes.iloc[max(0, idx - lookback - 1):idx]
        if len(q_window) > 0:
            q = q_window.iloc[-1]
            bid = q.get("bid", 0) or 0
            ask = q.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price
            f["spread"] = float(ask - bid) if ask > bid else 0
            f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
            f["bid_size"] = float(q.get("bid_size", 0) or 0)
            f["ask_size"] = float(q.get("ask_size", 0) or 0)
            f["size_imbalance"] = float((f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1))
        else:
            for k in ["spread", "spread_pct", "bid_size", "ask_size", "size_imbalance"]:
                f[k] = 0
    else:
        for k in ["spread", "spread_pct", "bid_size", "ask_size", "size_imbalance"]:
            f[k] = 0

    # Greeks
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
                f["iv_change_15m"] = float(ivs[-1] - ivs[0])
            else:
                f["iv_change_15m"] = 0
            f["underlying_price"] = float(g.get("underlying_price", 0) or 0)
        else:
            for k in ["iv", "delta", "theta", "vega", "iv_change_15m", "underlying_price"]:
                f[k] = 0
    else:
        for k in ["iv", "delta", "theta", "vega", "iv_change_15m", "underlying_price"]:
            f[k] = 0

    # Underlying price action
    if len(stock) > 0:
        s_window = stock.iloc[max(0, min(idx - 1, len(stock)) - lookback):min(idx, len(stock))]
        if len(s_window) > 1:
            s_closes = s_window["close"].dropna().values
            if len(s_closes) > 1 and all(s_closes > 0):
                f["underlying_change_5m"] = float((s_closes[-1] / s_closes[max(-6, -len(s_closes))] - 1) * 100)
                f["underlying_change_15m"] = float((s_closes[-1] / s_closes[0] - 1) * 100)
                f["underlying_volatility"] = float(np.std(np.diff(s_closes) / s_closes[:-1]) * 100)
            else:
                f["underlying_change_5m"] = 0
                f["underlying_change_15m"] = 0
                f["underlying_volatility"] = 0
        else:
            f["underlying_change_5m"] = 0
            f["underlying_change_15m"] = 0
            f["underlying_volatility"] = 0

        # Daily trend context
        s_all = stock.iloc[:min(idx, len(stock))]
        if len(s_all) > 10:
            s_closes_all = s_all["close"].dropna().values
            if len(s_closes_all) > 10 and s_closes_all[0] > 0:
                f["daily_trend_pct"] = float((s_closes_all[-1] / s_closes_all[0] - 1) * 100)
            else:
                f["daily_trend_pct"] = 0
            if len(s_closes_all) > 1:
                day_lo = s_closes_all.min()
                day_hi = s_closes_all.max()
                f["daily_range_position"] = float((s_closes_all[-1] - day_lo) / (day_hi - day_lo)) if day_hi > day_lo else 0.5
            else:
                f["daily_range_position"] = 0.5
            if len(s_all) > 14 and "high" in s_all.columns:
                highs = s_all["high"].dropna().values[-14:]
                lows = s_all["low"].dropna().values[-14:]
                if len(highs) >= 14:
                    f["atr_pct"] = float(np.mean(highs - lows) / s_closes_all[-1] * 100) if s_closes_all[-1] > 0 else 0
                else:
                    f["atr_pct"] = 0
            else:
                f["atr_pct"] = 0
        else:
            f["daily_trend_pct"] = 0
            f["daily_range_position"] = 0.5
            f["atr_pct"] = 0
    else:
        for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility",
                   "daily_trend_pct", "daily_range_position", "atr_pct"]:
            f[k] = 0

    return f


# ===========================================================================
# Preload data for a ticker (all dates, both directions)
# ===========================================================================

def preload_ticker_data(ticker):
    """Load all dates + ATM strikes for a ticker. Returns list of work items."""
    conn = _connect_theta()

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
        (ticker,),
    ).fetchall()]

    items = []
    for dt in dates:
        for right in ["CALL", "PUT"]:
            strike = find_atm_strike(conn, ticker, dt, right)
            if strike is None:
                continue

            ohlc, quotes, greeks, stock = load_day_data(conn, ticker, dt, right, strike)
            if len(ohlc) < 30:
                continue

            items.append({
                "ticker": ticker, "date": dt, "right": right, "strike": strike,
                "ohlc": ohlc.to_dict("list"),
                "quotes": quotes.to_dict("list"),
                "greeks": greeks.to_dict("list"),
                "stock": stock.to_dict("list"),
            })

    conn.close()
    return items


# ===========================================================================
# MODEL 1: Entry Timing — predict the LOW before a +38% run
# ===========================================================================

def _worker_entry_timing(item):
    """Find the optimal entry point (the low) before each profitable move."""
    ticker = item["ticker"]
    dt = item["date"]
    right = item["right"]
    min_move = TICKER_MOVE_PCT.get(ticker, MIN_MOVE_PCT)

    ohlc = pd.DataFrame(item["ohlc"])
    quotes = pd.DataFrame(item["quotes"])
    greeks = pd.DataFrame(item["greeks"])
    stock = pd.DataFrame(item["stock"])

    if len(ohlc) < 30:
        return []

    closes = ohlc["close"].values
    n = len(closes)
    rows = []

    # Phase 1: Find all profitable moves and their preceding lows
    move_lows = []  # (low_idx, move_start_idx, peak_pct)

    for i in range(PRE_MOVE_LOOKBACK, n - 10):
        entry = closes[i]
        if not entry or entry <= 0 or (isinstance(entry, float) and np.isnan(entry)):
            continue

        # Look forward: find peak gain within window
        future = closes[i:min(i + MOVE_WINDOW_MIN, n)]
        if len(future) < 5:
            continue
        peak = np.nanmax(future)
        if peak <= 0 or entry <= 0:
            continue
        peak_pct = (peak / entry - 1) * 100

        if peak_pct >= min_move:
            # This is a profitable entry. Now find the ACTUAL low in the
            # preceding 15 minutes — the optimal entry point.
            search_start = max(0, i - PRE_MOVE_LOOKBACK)
            search_end = i + 5  # allow 5 min into the move (the real low might be right at start)
            search_window = closes[search_start:min(search_end, n)]
            valid = [(j + search_start, closes[j + search_start])
                     for j in range(len(search_window))
                     if search_window[j] > 0 and not np.isnan(search_window[j])]
            if valid:
                low_idx, low_price = min(valid, key=lambda x: x[1])
                # Only count if the low gives an even better gain
                low_peak_pct = (peak / low_price - 1) * 100 if low_price > 0 else 0
                if low_peak_pct >= min_move:
                    move_lows.append((low_idx, i, low_peak_pct))

    if not move_lows:
        return []

    # Phase 2: Label each candle
    # Positive: within 5% of the low price before a profitable move
    # Negative: random candles NOT near any move low
    low_indices = set()
    for low_idx, _, _ in move_lows:
        # Mark candles within 3 bars of the low as positive
        for offset in range(-2, 3):
            idx = low_idx + offset
            if PRE_MOVE_LOOKBACK < idx < n - 10:
                price = closes[idx]
                low_price = closes[low_idx]
                if price > 0 and low_price > 0:
                    # Only if within 5% of the low
                    if (price / low_price - 1) * 100 < 5.0:
                        low_indices.add(idx)

    if not low_indices:
        return []

    # Generate features for positives
    for idx in low_indices:
        features = compute_pre_entry_features(ohlc, quotes, greeks, stock, idx)
        if features:
            # How much premium drops from recent peak to this point
            recent = closes[max(0, idx - 10):idx + 1]
            valid_recent = recent[~np.isnan(recent) & (recent > 0)]
            if len(valid_recent) > 0:
                features["prem_drop_from_recent_peak"] = float(
                    (closes[idx] / np.max(valid_recent) - 1) * 100
                )
            else:
                features["prem_drop_from_recent_peak"] = 0

            # Rate of premium decline (is it accelerating or decelerating?)
            if len(valid_recent) >= 3:
                first_half = valid_recent[:len(valid_recent)//2]
                second_half = valid_recent[len(valid_recent)//2:]
                if len(first_half) > 0 and len(second_half) > 0:
                    first_change = (first_half[-1] / first_half[0] - 1) * 100 if first_half[0] > 0 else 0
                    second_change = (second_half[-1] / second_half[0] - 1) * 100 if second_half[0] > 0 else 0
                    features["decline_deceleration"] = float(second_change - first_change)
                else:
                    features["decline_deceleration"] = 0
            else:
                features["decline_deceleration"] = 0

            features["label"] = 1
            features["ticker"] = ticker
            features["date"] = dt
            features["right"] = right
            rows.append(features)

    # Generate negatives: random candles NOT near any low
    neg_candidates = [i for i in range(PRE_MOVE_LOOKBACK + 1, n - 10)
                      if i not in low_indices and abs(i - min(low_indices)) > COOLDOWN_MIN]
    np.random.seed(hash(f"{ticker}{dt}{right}") % 2**31)
    n_neg = min(len(neg_candidates), len(low_indices) * 3)
    if n_neg > 0 and neg_candidates:
        neg_sample = np.random.choice(neg_candidates, size=n_neg, replace=False)
        for idx in neg_sample:
            features = compute_pre_entry_features(ohlc, quotes, greeks, stock, idx)
            if features:
                recent = closes[max(0, idx - 10):idx + 1]
                valid_recent = recent[~np.isnan(recent) & (recent > 0)]
                if len(valid_recent) > 0:
                    features["prem_drop_from_recent_peak"] = float(
                        (closes[idx] / np.max(valid_recent) - 1) * 100
                    )
                else:
                    features["prem_drop_from_recent_peak"] = 0
                if len(valid_recent) >= 3:
                    first_half = valid_recent[:len(valid_recent)//2]
                    second_half = valid_recent[len(valid_recent)//2:]
                    if len(first_half) > 0 and len(second_half) > 0:
                        first_change = (first_half[-1] / first_half[0] - 1) * 100 if first_half[0] > 0 else 0
                        second_change = (second_half[-1] / second_half[0] - 1) * 100 if second_half[0] > 0 else 0
                        features["decline_deceleration"] = float(second_change - first_change)
                    else:
                        features["decline_deceleration"] = 0
                else:
                    features["decline_deceleration"] = 0
                features["label"] = 0
                features["ticker"] = ticker
                features["date"] = dt
                features["right"] = right
                rows.append(features)

    return rows


# ===========================================================================
# MODEL 2: Exit Timing — hold vs sell at each point in a trade
# ===========================================================================

def _worker_exit_timing(item):
    """Generate exit timing training data from simulated trades.

    For each candle in a simulated trade:
    - Label 1 (SELL) if future premium will be LOWER than current (selling now is better)
    - Label 0 (HOLD) if future premium will be HIGHER (holding is better)

    Features: current trade state (P&L, time in trade, momentum, greeks changes)
    """
    ticker = item["ticker"]
    dt = item["date"]

    ohlc = pd.DataFrame(item["ohlc"])
    quotes = pd.DataFrame(item["quotes"])
    greeks = pd.DataFrame(item["greeks"])

    if len(ohlc) < 30:
        return []

    closes = ohlc["close"].values
    n = len(closes)
    rows = []

    # Sample entry points every COOLDOWN_MIN candles
    for entry_idx in range(PRE_MOVE_LOOKBACK, n - 20, COOLDOWN_MIN):
        entry_price = closes[entry_idx]
        if not entry_price or entry_price <= 0 or np.isnan(entry_price):
            continue

        # Look forward for the trade window
        end_idx = min(entry_idx + MOVE_WINDOW_MIN, n)
        future_closes = closes[entry_idx:end_idx]
        if len(future_closes) < 10:
            continue

        # Get underlying prices
        underlying_prices = []
        for j in range(entry_idx, end_idx):
            u = 0.0
            if len(greeks) > j:
                u = greeks.iloc[j].get("underlying_price", 0) or 0
            underlying_prices.append(float(u))

        # For each minute in the trade, decide: hold or sell?
        for offset in range(1, len(future_closes) - 5):
            curr_idx = entry_idx + offset
            curr_premium = future_closes[offset]
            if not curr_premium or curr_premium <= 0 or np.isnan(curr_premium):
                continue

            # Future premium: best achievable in next 10 minutes
            future_window = future_closes[offset + 1:min(offset + 11, len(future_closes))]
            if len(future_window) < 3:
                continue
            future_best = np.nanmax(future_window)
            future_worst = np.nanmin(future_window)

            # Label: sell if expected future value is worse than current
            # More nuanced: sell if the risk-adjusted future is negative
            future_expected = np.nanmean(future_window)
            upside = (future_best / curr_premium - 1) * 100
            downside = (future_worst / curr_premium - 1) * 100

            # SELL if expected future is < current (momentum dying)
            # or if downside risk > 2x upside potential
            should_sell = (future_expected < curr_premium * 0.99) or (abs(downside) > 2 * max(upside, 0.01))
            label = 1 if should_sell else 0

            # Features for exit decision
            f = {}
            f["minutes_in_trade"] = float(offset)
            f["current_pnl_pct"] = float((curr_premium / entry_price - 1) * 100)
            f["peak_pnl_pct"] = float((np.nanmax(future_closes[:offset + 1]) / entry_price - 1) * 100)
            f["drop_from_peak_pct"] = f["peak_pnl_pct"] - f["current_pnl_pct"]

            # Premium momentum (last 5 candles)
            recent = future_closes[max(0, offset - 5):offset + 1]
            if len(recent) > 1 and recent[0] > 0:
                f["premium_momentum_5m"] = float((recent[-1] / recent[0] - 1) * 100)
                f["premium_velocity"] = float(np.mean(np.diff(recent) / recent[:-1]) * 100) if all(recent[:-1] > 0) else 0
            else:
                f["premium_momentum_5m"] = 0
                f["premium_velocity"] = 0

            # Premium acceleration (is momentum changing?)
            if len(recent) > 3:
                first_half_mom = (recent[len(recent)//2] / recent[0] - 1) * 100 if recent[0] > 0 else 0
                second_half_mom = (recent[-1] / recent[len(recent)//2] - 1) * 100 if recent[len(recent)//2] > 0 else 0
                f["premium_acceleration"] = float(second_half_mom - first_half_mom)
            else:
                f["premium_acceleration"] = 0

            # Underlying momentum
            if offset < len(underlying_prices) and underlying_prices[offset] > 0:
                f["underlying_price"] = underlying_prices[offset]
                u_start = underlying_prices[max(0, offset - 5)]
                if u_start > 0:
                    f["underlying_momentum_5m"] = float((underlying_prices[offset] / u_start - 1) * 100)
                else:
                    f["underlying_momentum_5m"] = 0
                u_entry = underlying_prices[0]
                if u_entry > 0:
                    f["underlying_from_entry"] = float((underlying_prices[offset] / u_entry - 1) * 100)
                else:
                    f["underlying_from_entry"] = 0
            else:
                f["underlying_price"] = 0
                f["underlying_momentum_5m"] = 0
                f["underlying_from_entry"] = 0

            # Bid/ask at current point
            if len(quotes) > curr_idx:
                q = quotes.iloc[curr_idx]
                bid = q.get("bid", 0) or 0
                ask = q.get("ask", 0) or 0
                mid = (bid + ask) / 2 if (bid + ask) > 0 else curr_premium
                f["current_spread_pct"] = float((ask - bid) / mid * 100) if mid > 0 and ask > bid else 0
            else:
                f["current_spread_pct"] = 0

            # Greeks at current point
            if len(greeks) > curr_idx:
                g = greeks.iloc[curr_idx]
                f["current_iv"] = float(g.get("implied_vol", 0) or 0)
                f["current_delta"] = float(abs(g.get("delta", 0) or 0))
                f["current_theta"] = float(g.get("theta", 0) or 0)
            else:
                f["current_iv"] = 0
                f["current_delta"] = 0
                f["current_theta"] = 0

            # Volume at current point
            if "volume" in ohlc.columns and curr_idx < len(ohlc):
                f["current_volume"] = float(ohlc.iloc[curr_idx].get("volume", 0) or 0)
            else:
                f["current_volume"] = 0

            # Time of day
            try:
                ts = pd.Timestamp(ohlc.iloc[curr_idx]["timestamp"])
                if ts.tzinfo:
                    ts = ts.tz_convert("America/New_York")
                f["minutes_since_open"] = max(0, (ts.hour - 9) * 60 + ts.minute - 30)
            except Exception:
                f["minutes_since_open"] = 0

            f["is_call"] = 1 if item["right"] == "CALL" else 0
            f["label"] = label
            f["ticker"] = ticker
            f["date"] = dt
            rows.append(f)

    return rows


# ===========================================================================
# Shared: serve-time-safe day-level features (regime + ticker_selection)
# ===========================================================================
#
# Both day-level models are served ~9:45 ET (morning-features design), so
# every feature must be computable from data that exists by 9:45:
#   - RTH bars 09:30-09:44 of the SAME day (completed before 9:45)
#   - anything from PRIOR days (closes, ranges, volumes, realized vol)
#   - UW greek_exposure for the day (OI-based; OI is fixed at the open)
# stock_ohlc includes premarket bars from 04:00 — all intraday computations
# below filter to RTH explicitly.
# VIX is NOT in any local DB (checked thetadata_options.db + uw_historical.db
# 2026-06-10) — skipped until a source is backfilled.
# minutes-since-open is constant (=15) at the fixed 9:45 serve time for these
# one-prediction-per-day models, so it is intentionally not a feature.

RTH_START = "09:30"
RTH_END = "16:00"
EARLY_END = "09:45"  # day-level models serve at ~9:45 ET

# Per-ticker serve-time-safe features produced by _load_daily_context().
# Names match compute_regime_features() in options_owl/sourcing/ml_pipeline.py
# where the semantic is identical, so the live serving path needs minimal
# changes to feed the new model.
_TICKER_CONTEXT_FEATURES = [
    "morning_range_pct",   # 09:30-09:44 high-low range %
    "morning_volume",      # 09:30-09:44 share volume
    "morning_direction",   # 09:30 open -> 09:44 close return %
    "morning_body_ratio",  # |close-open| / (high-low) of the 15m window
    "morning_vol_15m",     # realized vol of 1-min returns, 09:30-09:44
    "overnight_gap_pct",   # today's RTH open vs yesterday's RTH close
    "prev_range_pct",      # yesterday's RTH range %
    "prev_volume",         # yesterday's RTH volume
    "prev_day_ret",        # yesterday's open->close return %
    "prev_close_pos",      # where yesterday closed in its range (0..1)
    "avg_3d_range",        # mean RTH range % of prior 3 days
    "vol_5d",              # realized vol of prior 5 close-to-close returns
    "range_trend",         # morning_range_pct / avg_3d_range - 1
    "volume_vs_prev",      # morning volume extrapolated vs prior 5-day avg
]

# Cross-market context (SPY/QQQ early morning) — all from bars <= 09:44 of
# the same day or prior days.
_MARKET_CONTEXT_COLS = [
    "morning_direction", "morning_range_pct", "morning_vol_15m",
    "overnight_gap_pct", "prev_day_ret", "vol_5d",
]

# OI-based greek exposure from UW. OI is set at the open (prior day's
# clearing), so the same-day greek_exposure row is known by 9:45. Falls back
# to the most recent prior date; zeros if stale (> 5 days; UW coverage
# currently ends 2026-05-21).
_GEX_COLS = [
    "call_gamma", "put_gamma", "net_gamma",
    "call_delta", "put_delta", "net_delta",
    "call_charm", "put_charm", "net_charm",
    "call_vanna", "put_vanna", "net_vanna",
]


def _load_daily_context(conn, ticker: str) -> pd.DataFrame | None:
    """Per-date daily context for a ticker from stock_ohlc (RTH bars only).

    Returns a DataFrame indexed by date string containing:
      - rth_range_pct / rth_close_pos: SAME-DAY label ingredients —
        NEVER to be used as features (that's the day_range_pct leak).
      - _TICKER_CONTEXT_FEATURES: all computable by 9:45 ET (early-morning
        window uses only bars before 09:45; the rest are prior-day lags).
    """
    bars = pd.read_sql_query(
        "SELECT substr(timestamp, 1, 10) AS d, substr(timestamp, 12, 5) AS tm, "
        "open, high, low, close, volume FROM stock_ohlc "
        "WHERE ticker=? ORDER BY timestamp",
        conn, params=(ticker,),
    )
    if bars.empty:
        return None
    # stock_ohlc includes premarket (04:00+) — keep regular hours only
    rth = bars[(bars["tm"] >= RTH_START) & (bars["tm"] <= RTH_END)]
    if rth.empty:
        return None

    g = rth.groupby("d")
    daily = pd.DataFrame({
        "rth_open": g["open"].first(),
        "rth_close": g["close"].last(),
        "rth_high": g["high"].max(),
        "rth_low": g["low"].min(),
        "rth_volume": g["volume"].sum(),
    })

    # Early-morning window: bars completed before the 9:45 serve time
    early_feats: dict[str, dict] = {}
    for d, grp in rth[rth["tm"] < EARLY_END].groupby("d"):
        closes = grp["close"].dropna().values
        opens = grp["open"].dropna().values
        highs = grp["high"].dropna().values
        lows = grp["low"].dropna().values
        row = {"morning_volume": float(grp["volume"].fillna(0).sum())}
        if (len(closes) >= 5 and len(opens) > 0 and len(highs) > 0 and len(lows) > 0
                and opens[0] > 0 and np.min(lows) > 0):
            m_open = float(opens[0])
            m_close = float(closes[-1])
            m_high = float(np.max(highs))
            m_low = float(np.min(lows))
            row["morning_range_pct"] = (m_high - m_low) / m_low * 100
            row["morning_direction"] = (m_close / m_open - 1) * 100
            rng = m_high - m_low
            row["morning_body_ratio"] = abs(m_close - m_open) / rng if rng > 0 else 0.0
            if np.all(closes[:-1] > 0):
                row["morning_vol_15m"] = float(
                    np.std(np.diff(closes) / closes[:-1]) * 100
                )
            else:
                row["morning_vol_15m"] = 0.0
        early_feats[d] = row
    daily = daily.join(pd.DataFrame.from_dict(early_feats, orient="index"), how="left")

    # SAME-DAY label ingredients (never features)
    rng_pct = (daily["rth_high"] - daily["rth_low"]) / daily["rth_low"] * 100
    span = (daily["rth_high"] - daily["rth_low"]).replace(0, np.nan)
    daily["rth_range_pct"] = rng_pct
    daily["rth_close_pos"] = ((daily["rth_close"] - daily["rth_low"]) / span).fillna(0.5)

    # Prior-day lag features (shift(1) = strictly past data at 9:45 today)
    prev_close = daily["rth_close"].shift(1)
    daily["overnight_gap_pct"] = (daily["rth_open"] / prev_close - 1) * 100
    daily["prev_range_pct"] = rng_pct.shift(1)
    daily["prev_volume"] = daily["rth_volume"].shift(1)
    daily["prev_day_ret"] = ((daily["rth_close"] / daily["rth_open"] - 1) * 100).shift(1)
    daily["prev_close_pos"] = daily["rth_close_pos"].shift(1)
    daily["avg_3d_range"] = rng_pct.shift(1).rolling(3).mean()
    daily["vol_5d"] = (daily["rth_close"].pct_change().rolling(5).std() * 100).shift(1)
    daily["range_trend"] = (
        daily["morning_range_pct"] / daily["avg_3d_range"].clip(lower=0.01) - 1
    )
    avg_prev_vol = daily["rth_volume"].shift(1).rolling(5).mean()
    daily["volume_vs_prev"] = daily["morning_volume"] * 26 / avg_prev_vol.clip(lower=1)
    return daily


def _load_rth_bars_by_date(conn, ticker: str) -> dict:
    """Grouped RTH 1-min bars {date: [bar,...]} for one ticker from stock_ohlc.

    Feeds the SHARED feature module (regime_features) so the trainer builds the
    morning/prior-day/market features with the exact same math as the live
    serving path — zero train/serve skew. Premarket bars are dropped inside
    rth_bars_by_date_from_rows.
    """
    bars = pd.read_sql_query(
        "SELECT substr(timestamp, 1, 10) AS d, substr(timestamp, 12, 5) AS tm, "
        "open, high, low, close, volume FROM stock_ohlc "
        "WHERE ticker=? ORDER BY timestamp",
        conn, params=(ticker,),
    )
    if bars.empty:
        return {}
    rows = bars.to_dict("records")
    return rth_bars_by_date_from_rows(rows)


def _load_market_bars(theta_conn) -> dict[str, dict]:
    """SPY + QQQ grouped RTH bars for cross-market context (shared module)."""
    return {
        mkt: _load_rth_bars_by_date(theta_conn, mkt) for mkt in ("SPY", "QQQ")
    }


def _load_market_context(theta_conn) -> dict[str, pd.DataFrame]:
    """SPY + QQQ daily context for cross-market features."""
    out = {}
    for mkt in ("SPY", "QQQ"):
        ctx = _load_daily_context(theta_conn, mkt)
        if ctx is not None:
            out[mkt] = ctx
    return out


def _market_features_for_date(market_ctx: dict[str, pd.DataFrame], dt: str) -> dict:
    """spy_*/qqq_* early-morning features for a date (zeros when missing)."""
    f = {}
    for mkt in ("SPY", "QQQ"):
        prefix = f"{mkt.lower()}_"
        ctx = market_ctx.get(mkt)
        if ctx is not None and dt in ctx.index:
            row = ctx.loc[dt]
            for col in _MARKET_CONTEXT_COLS:
                v = row.get(col)
                f[prefix + col] = float(v) if v is not None and pd.notna(v) else 0.0
        else:
            for col in _MARKET_CONTEXT_COLS:
                f[prefix + col] = 0.0
    return f


def _gex_features(uw_conn, ticker: str, dt: str, max_staleness_days: int = 5) -> dict:
    """OI-based UW greek exposure for ticker/date (serve-time-safe: OI is
    fixed at the open). Most recent date <= dt; zeros when missing/stale."""
    f = {k: 0.0 for k in _GEX_COLS}
    row = uw_conn.execute(
        "SELECT date, call_gamma, put_gamma, call_delta, put_delta, "
        "call_charm, put_charm, call_vanna, put_vanna "
        "FROM greek_exposure WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
        (ticker, dt),
    ).fetchone()
    if not row:
        return f
    staleness = (
        datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(row[0], "%Y-%m-%d")
    ).days
    if staleness > max_staleness_days:
        return f
    f["call_gamma"] = float(row[1] or 0)
    f["put_gamma"] = float(row[2] or 0)
    f["net_gamma"] = f["call_gamma"] - f["put_gamma"]
    f["call_delta"] = float(row[3] or 0)
    f["put_delta"] = float(row[4] or 0)
    f["net_delta"] = f["call_delta"] - f["put_delta"]
    f["call_charm"] = float(row[5] or 0)
    f["put_charm"] = float(row[6] or 0)
    f["net_charm"] = f["call_charm"] - f["put_charm"]
    f["call_vanna"] = float(row[7] or 0)
    f["put_vanna"] = float(row[8] or 0)
    f["net_vanna"] = f["call_vanna"] - f["put_vanna"]
    return f


# LightGBM params sized for ~2k-sample day-level datasets
_DAY_LEVEL_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l2": 1.0,
}


# ===========================================================================
# MODEL 3: Regime Classification — trending vs chop day
# ===========================================================================

def train_regime_model():
    """Train regime classifier from RTH stock OHLC + UW GEX data.

    Label: 1 = trending day (RTH range > 1.5% AND close near high/low),
           0 = chop day. Computed from RTH bars only (premarket excluded).

    Every feature is serve-time-safe at ~9:45 ET: early-morning RTH bars
    (09:30-09:44), prior-day lags, SPY/QQQ cross-market context, and OI-based
    greek exposure. The old full-day leak features (day_range_pct, day_volume)
    remain dropped; the old same-day options_volume features (put_call_ratio,
    net premiums, OI) were also full-day information AND the UW table only
    has a single date of data, so they are dropped too.
    """
    print("\n" + "=" * 70)
    print("MODEL 3: Regime Classification (trending vs chop)")
    print("=" * 70)

    theta_conn = _connect_theta()
    uw_conn = _connect_uw()

    # SHARED feature module: build features from grouped RTH bars (same math as
    # live serving). _load_daily_context is kept ONLY to derive the LABEL
    # (full-day RTH range/close-pos — same-day info that is never a feature).
    market_ctx = _load_market_context(theta_conn)        # label/date frames
    market_bars = _load_market_bars(theta_conn)          # shared-module inputs

    rows = []
    for ticker in TICKERS:
        ctx = market_ctx.get(ticker)
        if ctx is None:
            ctx = _load_daily_context(theta_conn, ticker)
        if ctx is None or len(ctx) < 20:
            print(f"  {ticker}: skipped (no/insufficient stock data)")
            continue

        by_date = (
            market_bars.get(ticker)
            if ticker in market_bars
            else _load_rth_bars_by_date(theta_conn, ticker)
        )
        if not by_date:
            print(f"  {ticker}: skipped (no RTH bars)")
            continue

        n_ticker = 0
        for dt, day in ctx.iterrows():
            if (pd.isna(day["rth_high"]) or pd.isna(day["rth_low"])
                    or day["rth_low"] <= 0 or pd.isna(day["rth_close"])):
                continue
            # Require a usable early-morning window (serving needs >= 5 bars)
            if pd.isna(day.get("morning_direction")):
                continue

            # Label from RTH-only range/close position
            is_trending = (
                day["rth_range_pct"] > 1.5
                and (day["rth_close_pos"] < 0.2 or day["rth_close_pos"] > 0.8)
            )

            raw_inputs = load_training_inputs(
                ticker, dt,
                by_date=by_date,
                market_by_date=market_bars,
                gex_row=_gex_features(uw_conn, ticker, dt),
            )
            f = compute_regime_feature_vector(raw_inputs)

            f["label"] = 1 if is_trending else 0
            f["ticker"] = ticker
            f["date"] = dt
            rows.append(f)
            n_ticker += 1
        print(f"  {ticker}: {n_ticker} days")

    theta_conn.close()
    uw_conn.close()

    if not rows:
        print("  No regime data collected!")
        return

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} day-ticker samples "
          f"({df['label'].sum()} trending, {(~df['label'].astype(bool)).sum()} chop)")

    # Feature order is the SHARED module's canonical REGIME_FEATURE_ORDER, so
    # the saved meta["features"] matches compute_regime_features() exactly.
    leak_cols = ["day_range_pct", "day_volume", "rth_range_pct", "rth_close_pos"]
    feature_cols = list(REGIME_FEATURE_ORDER)
    assert not set(leak_cols) & set(feature_cols), "leak features must stay dropped"

    result = _train_day_level_model(
        df, feature_cols, dict(_DAY_LEVEL_PARAMS), "Regime Model",
        embargo_days=1, num_boost_round=300,
    )
    if result is None:
        return
    model, meta = result

    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    model_path = str(MODEL_DIR / "regime_classifier.txt")
    model.save_model(model_path)
    meta["dropped_leak_features"] = leak_cols + ["options_volume_same_day_features"]
    with open(str(MODEL_DIR / "regime_classifier_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {model_path}")


# ===========================================================================
# MODEL 4: Ticker Selection — which tickers to focus on today
# ===========================================================================

def _ticker_day_opportunity(theta_conn, ticker: str, dt: str) -> dict | None:
    """Realistic tradable-opportunity label for one ticker-day.

    Old label (best entry in first 60min -> best full-day exit > 30%) was
    ~99% positive for 0DTE ATM options — degenerate, which is why the
    last-month holdout was single-class and AUC was NaN.

    New label: from the first ATM bar at/after 09:45 ET (the model's serve
    time), did the premium peak >= the per-ticker min_move (TICKER_MOVE_PCT)
    within the next 120 minutes (MOVE_WINDOW_MIN), on EITHER the CALL or PUT
    side? ~78% positive on the current DB. Also returns serve-time-safe
    CALL-side features (9:30 open premium/IV, 9:30->9:45 premium change).
    """
    min_move = TICKER_MOVE_PCT.get(ticker, MIN_MOVE_PCT)
    best_gain = None
    feats: dict = {}

    for right in ("CALL", "PUT"):
        strike = find_atm_strike(theta_conn, ticker, dt, right)
        if strike is None:
            continue

        opt = pd.read_sql_query(
            "SELECT substr(timestamp, 12, 5) AS tm, close FROM option_ohlc "
            "WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? "
            "ORDER BY timestamp",
            theta_conn, params=(ticker, f"{dt}%", right, strike),
        )
        opt = opt[(opt["close"].notna()) & (opt["close"] > 0)]
        if len(opt) < 30:
            continue

        # Entry = first valid bar at/after the 9:45 serve time
        entry_rows = opt[opt["tm"] >= EARLY_END]
        if entry_rows.empty:
            continue
        entry_tm = entry_rows.iloc[0]["tm"]
        entry_price = float(entry_rows.iloc[0]["close"])
        if entry_price <= 0:
            continue

        # Peak within the next MOVE_WINDOW_MIN minutes
        eh, em = int(entry_tm[:2]), int(entry_tm[3:])
        end_min = eh * 60 + em + MOVE_WINDOW_MIN
        end_tm = f"{end_min // 60:02d}:{end_min % 60:02d}"
        window = entry_rows.iloc[1:]
        window = window[window["tm"] <= end_tm]
        if len(window) < 5:
            continue
        gain = (float(window["close"].max()) / entry_price - 1) * 100
        best_gain = gain if best_gain is None else max(best_gain, gain)

        if right == "CALL":
            # Serve-time-safe CALL-side features (bars <= 9:45 only)
            feats["opening_premium"] = float(opt.iloc[0]["close"])
            feats["call_prem_change_15m"] = (
                (entry_price / feats["opening_premium"] - 1) * 100
                if feats["opening_premium"] > 0 else 0.0
            )
            greeks_first = theta_conn.execute(
                "SELECT implied_vol, delta FROM option_greeks "
                "WHERE ticker=? AND timestamp LIKE ? AND right='CALL' AND strike=? "
                "ORDER BY timestamp LIMIT 1",
                (ticker, f"{dt}%", strike),
            ).fetchone()
            feats["opening_iv"] = float(greeks_first[0] or 0) if greeks_first else 0.0
            feats["opening_delta"] = (
                float(abs(greeks_first[1] or 0)) if greeks_first else 0.0
            )

    if best_gain is None:
        return None
    for k in ("opening_premium", "call_prem_change_15m", "opening_iv", "opening_delta"):
        feats.setdefault(k, 0.0)
    feats["label"] = 1 if best_gain >= min_move else 0
    feats["max_pnl_pct"] = best_gain
    return feats


def train_ticker_selection_model():
    """Train model to predict which tickers will be tradable today.

    For each ticker-day: features are early-morning (<= 9:45 ET) data only;
    label is whether a realistic 9:45 ATM entry (CALL or PUT) had a
    min_move-sized peak within the next 120 minutes (see
    _ticker_day_opportunity).
    """
    print("\n" + "=" * 70)
    print("MODEL 4: Ticker Selection (which tickers to trade today)")
    print("=" * 70)

    theta_conn = _connect_theta()
    uw_conn = _connect_uw()

    market_ctx = _load_market_context(theta_conn)

    rows = []
    for ticker in TICKERS:
        print(f"  Processing {ticker}...", flush=True)

        ctx = market_ctx.get(ticker)
        if ctx is None:
            ctx = _load_daily_context(theta_conn, ticker)

        dates = [r[0] for r in theta_conn.execute(
            "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
            (ticker,),
        ).fetchall()]

        for dt in dates:
            opp = _ticker_day_opportunity(theta_conn, ticker, dt)
            if opp is None:
                continue

            f = {
                "ticker_idx": TICKERS.index(ticker),
                "day_of_week": datetime.strptime(dt, "%Y-%m-%d").weekday(),
            }

            # Per-ticker early-morning + prior-day context (RTH only —
            # the old version's "early" features used premarket bars)
            if ctx is not None and dt in ctx.index:
                day = ctx.loc[dt]
                for col in _TICKER_CONTEXT_FEATURES:
                    v = day.get(col)
                    f[col] = float(v) if v is not None and pd.notna(v) else 0.0
            else:
                for col in _TICKER_CONTEXT_FEATURES:
                    f[col] = 0.0

            # OI-based greek exposure (known at open). The old same-day
            # options_volume features (put_call_ratio, net_premium_flow)
            # were full-day info AND the table only has one date — dropped.
            f.update(_gex_features(uw_conn, ticker, dt))

            # SPY/QQQ early-morning market context
            f.update(_market_features_for_date(market_ctx, dt))

            f.update(opp)
            f["ticker"] = ticker
            f["date"] = dt
            rows.append(f)

    theta_conn.close()
    uw_conn.close()

    if not rows:
        print("  No ticker selection data!")
        return

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} ticker-day samples "
          f"({df['label'].sum()} tradable, {(~df['label'].astype(bool)).sum()} not)")

    meta_cols = ["ticker", "date", "max_pnl_pct"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    result = _train_day_level_model(
        df, feature_cols, dict(_DAY_LEVEL_PARAMS), "Ticker Selection",
        embargo_days=1, num_boost_round=300,
    )
    if result is None:
        return
    model, meta = result

    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    model_path = str(MODEL_DIR / "ticker_selection.txt")
    model.save_model(model_path)
    meta["label"] = (
        "peak >= per-ticker min_move within 120min of a 9:45 ATM entry "
        "(CALL or PUT)"
    )
    with open(str(MODEL_DIR / "ticker_selection_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {model_path}")


# ===========================================================================
# MODEL 5: Stop Calibration — optimal stop width per entry
# ===========================================================================

def _worker_stop_calibrate(item):
    """Test multiple stop widths for each entry and find the optimal one.

    For each entry point, simulate with 5 different stop widths (20%, 30%, 40%, 50%, 65%)
    and record which stop width produced the best P&L.
    """
    ticker = item["ticker"]
    dt = item["date"]

    ohlc = pd.DataFrame(item["ohlc"])
    quotes = pd.DataFrame(item["quotes"])
    greeks = pd.DataFrame(item["greeks"])
    stock = pd.DataFrame(item["stock"])

    if len(ohlc) < 30:
        return []

    closes = ohlc["close"].values
    n = len(closes)
    rows = []
    stop_widths = [20.0, 30.0, 40.0, 50.0, 65.0]

    for entry_idx in range(PRE_MOVE_LOOKBACK, n - 20, COOLDOWN_MIN):
        entry_price = closes[entry_idx]
        if not entry_price or entry_price <= 0 or np.isnan(entry_price):
            continue

        # Simulate with each stop width
        results = {}
        for stop_pct in stop_widths:
            # Simple stop simulation: exit when premium drops stop_pct% from entry
            # or when premium hits +100% (profit target)
            end_idx = min(entry_idx + MOVE_WINDOW_MIN, n)
            exit_pnl = 0
            for j in range(entry_idx + 1, end_idx):
                prem = closes[j]
                if not prem or prem <= 0 or np.isnan(prem):
                    continue
                pnl_pct = (prem / entry_price - 1) * 100
                if pnl_pct <= -stop_pct:
                    exit_pnl = -stop_pct
                    break
                if pnl_pct >= 100:
                    exit_pnl = 100
                    break
                exit_pnl = pnl_pct  # last seen P&L if no stop/target hit
            results[stop_pct] = exit_pnl

        if not results:
            continue

        # Best stop = highest P&L
        best_stop = max(results, key=results.get)

        # Features
        features = compute_pre_entry_features(ohlc, quotes, greeks, stock, entry_idx)
        if not features:
            continue

        features["is_call"] = 1 if item["right"] == "CALL" else 0
        # Label: optimal stop width (as category index)
        features["optimal_stop_pct"] = best_stop
        features["best_pnl"] = results[best_stop]
        features["worst_stop_pnl"] = min(results.values())
        features["ticker"] = ticker
        features["date"] = dt
        rows.append(features)

    return rows


# ===========================================================================
# MODEL 6: Signal Quality — predict magnitude of move (regression)
# ===========================================================================

# Quality threshold: an entry is "high quality" if its premium peaks at least
# this much within the forward window. 50% is a meaningful 0DTE scalp move and
# yields a usable, non-degenerate positive rate (vs the old regression target
# whose corr collapsed 0.52 -> 0.068 once leakage was removed).
SIGNAL_QUALITY_THRESHOLD_PCT = 50.0


def _worker_signal_quality(item):
    """Classify whether an ATM entry is HIGH QUALITY (binary, serve-time-safe).

    REWORK (2026-06-10): the old target was a regression on peak_gain_pct that
    was a leakage artifact (corr 0.52 -> 0.068 once the forward leak was
    removed) and uninformative (MAE 158). It is replaced — same medicine as
    regime/ticker_selection — with a binary classification:

      Label  : 1 if premium peaks >= SIGNAL_QUALITY_THRESHOLD_PCT within the
               next MOVE_WINDOW_MIN minutes, else 0  (forward window -> needs
               the 1-day embargo, which train_model_with_pool applies).
      Features: compute_pre_entry_features — STRICTLY-PAST trailing windows
               (premium/volume/greeks/underlying up to idx-1). No forward bars,
               no full-day aggregates. Serve-time-safe at the entry minute.

    Entries are sampled only at/after 30 minutes since the open so the trailing
    feature windows are fully populated (mirrors the live scanner's warm-up).
    """
    ticker = item["ticker"]
    dt = item["date"]

    ohlc = pd.DataFrame(item["ohlc"])
    quotes = pd.DataFrame(item["quotes"])
    greeks = pd.DataFrame(item["greeks"])
    stock = pd.DataFrame(item["stock"])

    if len(ohlc) < 30:
        return []

    closes = ohlc["close"].values
    n = len(closes)
    rows = []

    for entry_idx in range(PRE_MOVE_LOOKBACK, n - 10, 10):  # every 10 candles
        entry_price = closes[entry_idx]
        if not entry_price or entry_price <= 0 or np.isnan(entry_price):
            continue

        # Forward peak gain over the next MOVE_WINDOW_MIN minutes (label only).
        future = closes[entry_idx:min(entry_idx + MOVE_WINDOW_MIN, n)]
        if len(future) < 5:
            continue
        peak = np.nanmax(future)
        if peak <= 0:
            continue
        peak_gain_pct = (peak / entry_price - 1) * 100

        features = compute_pre_entry_features(ohlc, quotes, greeks, stock, entry_idx)
        if not features:
            continue

        # Warm-up gate: require a fully-populated trailing window (matches live).
        if features.get("minutes_since_open", 0) < 30:
            continue

        features["is_call"] = 1 if item["right"] == "CALL" else 0
        features["label"] = 1 if peak_gain_pct >= SIGNAL_QUALITY_THRESHOLD_PCT else 0
        features["peak_gain_pct"] = peak_gain_pct  # meta only (not a feature)
        features["ticker"] = ticker
        features["date"] = dt
        rows.append(features)

    return rows


# ===========================================================================
# Training orchestration
# ===========================================================================

def train_model_with_pool(model_name, worker_fn, tickers, label_col="label",
                          is_regression=False, extra_meta_cols=None,
                          embargo_days=0, robust_split=False, model_meta=None):
    """Generic training pipeline: preload data → parallel workers → train LightGBM.

    When ``robust_split`` is True the assembled (binary) dataset is routed
    through _train_day_level_model, which GUARANTEES a multi-class date holdout
    (or falls back to stratified date k-fold) and never emits NaN AUC — the same
    robust validation used by regime/ticker_selection. ``model_meta`` extra keys
    are merged into the saved meta JSON.

    Splits by DATE (expanding-window walk-forward + last-month holdout), never
    by row. Set embargo_days=1 for models whose labels look forward in time.
    """
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker}...", end="", flush=True)
        items = preload_ticker_data(ticker)
        print(f" {len(items)} day-sides in {time.time() - t0:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker} ({len(items)} items, {N_WORKERS} workers)...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(worker_fn, items, chunksize=4)
        for r in results:
            all_rows.extend(r)
        print(f" {len(all_rows)} samples so far ({time.time() - t0:.0f}s)", flush=True)

    if not all_rows:
        print("  No training data collected!")
        return None

    df = pd.DataFrame(all_rows)
    meta_cols = ["ticker", "date", "right"] + (extra_meta_cols or [])
    meta_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols + [label_col]]

    print(f"\n  Total samples: {len(df)}")
    if not is_regression:
        pos = df[label_col].sum()
        neg = len(df) - pos
        print(f"  Positive: {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg} ({neg/len(df)*100:.1f}%)")

    # Robust path: guaranteed multi-class date holdout (or stratified k-fold),
    # no NaN AUC. Used by the reworked signal_quality model.
    if robust_split and not is_regression:
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l2": 1.0,
        }
        result = _train_day_level_model(
            df, feature_cols, params, model_name, label_col=label_col,
            embargo_days=embargo_days, num_boost_round=500,
        )
        if result is None:
            print(f"  {model_name}: not saved (no valid validation).")
            return None
        model, meta = result
        if model_meta:
            meta.update(model_meta)
        imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
        print("  Top features:")
        for name, gain in imp[:10]:
            print(f"    {name}: {gain:.0f}")
        safe_name = model_name.replace(" ", "_").lower()
        model.save_model(str(MODEL_DIR / f"{safe_name}.txt"))
        with open(str(MODEL_DIR / f"{safe_name}_meta.json"), "w") as mf:
            json.dump(meta, mf, indent=2)
        print(f"  Saved to {MODEL_DIR / f'{safe_name}.txt'}")
        return model

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    if is_regression:
        params = {
            "objective": "regression", "metric": "mae", "verbosity": -1,
            "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        }
    else:
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        }

    # Date-based expanding-window walk-forward validation (never split by row)
    fold_scores = _walk_forward_validate(
        df, X, y, feature_cols, params, is_regression,
        embargo_days=embargo_days, num_boost_round=500,
    )

    # Final production model: train on all but the last month, test on it
    train_mask, test_mask = _final_date_split(df, embargo_days=embargo_days)
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    print(f"\n  Final split: train={len(X_train)} test={len(X_test)} "
          f"(by date, embargo_days={embargo_days})")

    if not is_regression:
        params["scale_pos_weight"] = float(
            (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        )

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=500,
                       valid_sets=[dtest], callbacks=[lgb.log_evaluation(100)])

    preds = model.predict(X_test)

    meta = {
        "features": feature_cols, "n_train": len(X_train), "n_test": len(X_test),
        "split": "date_walk_forward", "embargo_days": embargo_days,
        "walk_forward_folds": len(fold_scores),
        "walk_forward_mean": float(np.mean(fold_scores)) if fold_scores else None,
        "walk_forward_std": float(np.std(fold_scores)) if fold_scores else None,
    }

    if is_regression:
        mae = mean_absolute_error(y_test, preds)
        corr = np.corrcoef(y_test, preds)[0, 1] if len(y_test) > 2 else 0
        print(f"\n  {model_name}: MAE={mae:.2f} Correlation={corr:.3f}")
        meta["mae"] = mae
        meta["correlation"] = corr

        # Bucket analysis: does the model rank moves correctly?
        pred_df = pd.DataFrame({"pred": preds, "actual": y_test})
        for q_label, (lo, hi) in [("bottom_20%", (0, 0.2)), ("mid_60%", (0.2, 0.8)), ("top_20%", (0.8, 1.0))]:
            q_lo = pred_df["pred"].quantile(lo)
            q_hi = pred_df["pred"].quantile(hi)
            subset = pred_df[(pred_df["pred"] >= q_lo) & (pred_df["pred"] < q_hi)]
            if len(subset) > 0:
                print(f"    {q_label}: avg_pred={subset['pred'].mean():.1f}% avg_actual={subset['actual'].mean():.1f}% n={len(subset)}")
    else:
        auc = roc_auc_score(y_test, preds)
        pred_labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, pred_labels)
        prec = precision_score(y_test, pred_labels, zero_division=0)
        rec = recall_score(y_test, pred_labels, zero_division=0)
        print(f"\n  {model_name}: AUC={auc:.3f} Acc={acc:.3f} Prec={prec:.3f} Recall={rec:.3f}")
        meta["auc"] = auc
        meta["accuracy"] = acc
        meta["precision"] = prec
        meta["recall"] = rec

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    # Save
    safe_name = model_name.replace(" ", "_").lower()
    model_path = str(MODEL_DIR / f"{safe_name}.txt")
    model.save_model(model_path)
    with open(str(MODEL_DIR / f"{safe_name}_meta.json"), "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"  Saved to {model_path}")

    return model


def train_stop_calibration_model(tickers):
    """Train stop calibration as multi-class: predict optimal stop width bucket."""
    print(f"\n{'=' * 70}")
    print("MODEL 5: Stop Calibration (optimal stop width)")
    print(f"{'=' * 70}")

    all_rows = []
    for ticker in tickers:
        t0 = time.time()
        print(f"  Preloading {ticker}...", end="", flush=True)
        items = preload_ticker_data(ticker)
        print(f" {len(items)} day-sides in {time.time() - t0:.0f}s", flush=True)

        if not items:
            continue

        print(f"  Processing {ticker}...", end="", flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker_stop_calibrate, items, chunksize=4)
        for r in results:
            all_rows.extend(r)
        print(f" {len(all_rows)} samples ({time.time() - t0:.0f}s)", flush=True)

    if not all_rows:
        print("  No data!")
        return

    df = pd.DataFrame(all_rows)
    meta_cols = ["ticker", "date", "optimal_stop_pct", "best_pnl", "worst_stop_pnl"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"\n  Total samples: {len(df)}")
    print("  Stop distribution:")
    for stop in sorted(df["optimal_stop_pct"].unique()):
        n = (df["optimal_stop_pct"] == stop).sum()
        avg_pnl = df[df["optimal_stop_pct"] == stop]["best_pnl"].mean()
        print(f"    {stop:.0f}%: {n} ({n/len(df)*100:.1f}%) avg_pnl={avg_pnl:+.1f}%")

    # Train as regression: predict optimal stop width directly
    X = df[feature_cols].values.astype(np.float32)
    y = df["optimal_stop_pct"].values.astype(np.float32)

    params = {
        "objective": "regression", "metric": "mae", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    }

    # Date-based expanding-window walk-forward (never split by row).
    # 1-day embargo: the label simulates 120min forward from each entry.
    _walk_forward_validate(df, X, y, feature_cols, params, is_regression=True,
                           embargo_days=1, num_boost_round=300)

    train_mask, test_mask = _final_date_split(df, embargo_days=1)
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    print(f"\n  Final split: train={len(X_train)} test={len(X_test)} "
          f"(by date, embargo_days=1)")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=300,
                       valid_sets=[dtest], callbacks=[lgb.log_evaluation(100)])

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    corr = np.corrcoef(y_test, preds)[0, 1] if len(y_test) > 2 else 0

    print(f"\n  Stop Calibration: MAE={mae:.1f}% Correlation={corr:.3f}")

    # Does using the model's predicted stop beat the fixed stop?
    # Compare predicted vs fixed 35% stop
    print(f"  Mean predicted stop: {preds.mean():.1f}% vs actual best: {y_test.mean():.1f}%")

    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    model_path = str(MODEL_DIR / "stop_calibration.txt")
    model.save_model(model_path)
    meta = {"features": feature_cols, "mae": mae, "correlation": corr,
            "n_train": len(X_train), "n_test": len(X_test)}
    with open(str(MODEL_DIR / "stop_calibration_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {model_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train V3 ML model suite")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "entry_timing", "exit_timing", "regime",
                                 "ticker_select", "stop_calibrate", "signal_quality"],
                        help="Which model to train (default: all)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker (default: all)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate existing models")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS
    model = args.model

    print(f"V3 ML Model Suite — {len(tickers)} tickers, {N_WORKERS} workers")
    print(f"Models: {model}")
    print(f"ThetaData DB: {THETADATA_DB}")
    print(f"UW DB: {UW_DB}")
    print(f"Output: {MODEL_DIR}")
    print()

    t_start = time.time()

    # embargo_days=1 for models whose labels look forward in time
    # (entry_timing: 120-min move window; exit_timing: 10-min hold/sell window;
    # signal_quality: 120-min peak-gain window)
    if model in ("all", "entry_timing"):
        train_model_with_pool(
            "entry_timing", _worker_entry_timing, tickers,
            label_col="label", is_regression=False, embargo_days=1,
        )

    if model in ("all", "exit_timing"):
        train_model_with_pool(
            "exit_timing", _worker_exit_timing, tickers,
            label_col="label", is_regression=False, embargo_days=1,
        )

    if model in ("all", "regime"):
        train_regime_model()

    if model in ("all", "ticker_select"):
        train_ticker_selection_model()

    # stop_calibration is recorded-only at serve time (ml_pipeline computes stop_pct → audit log
    # ONLY; never applied to the FSM config). Under V7 wide-trail exits the graduated_stop gate
    # never binds before the trails fire, so calibrating it changes nothing (verified byte-identical
    # in the 2026-06-13 stop_cal test). DROPPED from "all" retrains by default. The fixed
    # graduated_stop backstop (per-ticker config) is untouched and still the catastrophic-loss net.
    _train_stop = model == "stop_calibrate" or (
        model == "all" and os.getenv("RETRAIN_INCLUDE_STOP_CALIBRATION", "0") == "1")
    if _train_stop:
        train_stop_calibration_model(tickers)

    # signal_quality is recorded-only (it gates nothing in V7 OR the gs baseline), so it's
    # DROPPED from "all" retrains by default to save compute. Set RETRAIN_INCLUDE_SIGNAL_QUALITY=1
    # to re-include, or train explicitly with --model signal_quality.
    _train_sq = model == "signal_quality" or (
        model == "all" and os.getenv("RETRAIN_INCLUDE_SIGNAL_QUALITY", "0") == "1")
    if _train_sq:
        # Reworked: binary "high-quality entry" classifier (was a leak-inflated
        # regression). Serve-time-safe pre-entry features + robust date split.
        train_model_with_pool(
            "signal_quality", _worker_signal_quality, tickers,
            label_col="label", is_regression=False,
            extra_meta_cols=["peak_gain_pct"], embargo_days=1,
            robust_split=True,
            model_meta={
                "label": (
                    f"premium peaks >= {SIGNAL_QUALITY_THRESHOLD_PCT:.0f}% within "
                    f"{MOVE_WINDOW_MIN}min of an entry (binary); features are "
                    "strictly-past pre-entry trailing windows"
                ),
                "quality_threshold_pct": SIGNAL_QUALITY_THRESHOLD_PCT,
            },
        )

    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
