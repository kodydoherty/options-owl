"""Train ML sell models using harvester data with real signal entry times.

Unlike ml_sell_model.py which uses fixed 10AM entries from historical_0dte.db,
this script uses actual Discord signal timestamps as entry points and harvester
minute-by-minute snapshots as price data. This produces models that actually
match the production distribution of entry times (mostly afternoon signals).

Additionally, this script can multi-sample each day's harvester data: for each
contract that has enough bars, we create synthetic trades at multiple entry
times (every 15 minutes from first bar to 90 min before last bar). This
dramatically increases the training set from ~100 real signals to thousands
of training trades.

Usage:
    python scripts/train_on_harvester.py
    python scripts/train_on_harvester.py --signal-only  # only real signal entries
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not installed: pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
HARVESTER_DB = os.path.join(PROJECT_DIR, "journal", "owlet-harvester", "options_data.db")
SIGNALS_DBS = [
    os.path.join(PROJECT_DIR, "journal", "raw_messages.db"),
    os.path.join(PROJECT_DIR, "journal", "owlet-kody", "raw_messages.db"),
]
MODEL_DIR = os.path.join(PROJECT_DIR, "journal", "models")

ET_TZ = ZoneInfo("America/New_York")

# ─── Constants ───────────────────────────────────────────────────────────────
# Sweet spot timing (relative to entry, not absolute clock time)
SWEET_SPOT_START_MIN = 30   # shorter than 10AM model — afternoon trades move faster
SWEET_SPOT_END_MIN = 120
HARD_DEADLINE_MIN = 180     # 3 hours max hold (afternoon entries have less runway)

# Label thresholds
UPSIDE_THRESHOLD_PCT = 3.0
REGRET_THRESHOLD_PCT = -10.0

# Minimum bars to consider a trade
MIN_BARS = 15

# Multi-sampling: create synthetic entries every N minutes
MULTI_SAMPLE_INTERVAL_MIN = 15
MIN_BARS_AFTER_ENTRY = 30   # need at least 30 min of data after each entry

ALL_TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "META", "AMD",
               "AMZN", "GOOGL", "MSFT", "IWM", "MU", "MSTR",
               "COIN", "JPM", "TLT", "XLF", "PLTR"]

FEATURE_COLS = [
    "pnl_pct", "mfe_pct", "drawdown_from_peak_pct",
    "minutes_since_entry", "hour_of_day", "minute_of_hour",
    "in_sweet_spot", "past_sweet_spot", "minutes_past_sweet_spot",
    "time_pressure",
    "premium_velocity_5m", "premium_velocity_10m", "premium_velocity_15m",
    "pnl_acceleration", "mfe_retracement_ratio",
    "bar_range_pct", "volume", "volume_vs_avg",
    "rolling_volatility_10m",
    "underlying_pnl_pct", "underlying_velocity_5m",
    "is_call", "entry_premium",
    "bars_since_new_high", "consecutive_down_bars",
    "risk_reward_ratio",
]

# Per-ticker models also get ticker_encoded
GENERIC_FEATURE_COLS = FEATURE_COLS + ["ticker_encoded"]


# ─── Data Loading ────────────────────────────────────────────────────────────

def _to_et(utc_dt: datetime) -> datetime:
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(ET_TZ)


def load_harvester_snapshots(contract_ticker: str, date: str, conn) -> list[dict]:
    """Load minute-by-minute snapshots for a contract on a date."""
    rows = conn.execute("""
        SELECT captured_at, midpoint, last_trade_price, underlying_price,
               day_high, day_low, day_volume, day_open, day_close
        FROM harvest_snapshots
        WHERE contract_ticker = ? AND DATE(captured_at) = ?
        ORDER BY captured_at
    """, (contract_ticker, date)).fetchall()

    bars = []
    for r in rows:
        captured_at, midpoint, last_trade, underlying, high, low, vol, opn, close = r
        price = midpoint if midpoint and midpoint > 0 else (last_trade if last_trade and last_trade > 0 else None)
        if price is None or price <= 0:
            continue
        dt = datetime.fromisoformat(captured_at)
        et = _to_et(dt)
        bars.append({
            "price": price,
            "underlying_price": underlying,
            "high": high or price,
            "low": low or price,
            "volume": vol or 0,
            "open": opn or price,
            "close": close or price,
            "hour": et.hour,
            "minute": et.minute,
            "timestamp_utc": dt,
        })
    return bars


def load_all_signals() -> list[dict]:
    """Load all Discord signals from both DBs."""
    signals = []
    for db_path in SIGNALS_DBS:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("""
                SELECT ticker, direction, strike, expiry, created_at
                FROM trade_signals WHERE score >= 75
                ORDER BY created_at
            """).fetchall()
            for ticker, direction, strike, expiry, created_at in rows:
                if expiry == "0DTE":
                    expiry_date = created_at[:10]
                else:
                    expiry_date = expiry
                option_type = "call" if direction.lower() in ("bullish", "call", "long") else "put"
                signals.append({
                    "ticker": ticker,
                    "strike": float(strike),
                    "expiry_date": expiry_date,
                    "option_type": option_type,
                    "created_at": created_at,
                    "date": created_at[:10],
                })
        except Exception:
            pass
        conn.close()
    return signals


def build_contract_ticker(underlying, expiry_date, strike, option_type):
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{date_str}{opt_char}{strike_int:08d}"


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_features(bars: list[dict], entry_idx: int, ticker_code: int,
                     is_call: bool) -> list[dict] | None:
    """Extract features from entry_idx to end of bars."""
    if entry_idx >= len(bars) - MIN_BARS:
        return None

    trade_bars = bars[entry_idx:]
    if len(trade_bars) < MIN_BARS:
        return None

    entry_price = trade_bars[0]["price"]
    if entry_price <= 0:
        return None

    underlying_entry = trade_bars[0].get("underlying_price")
    close_prices = [b["price"] for b in trade_bars]
    close_arr = np.array(close_prices, dtype=np.float64)
    pnl_arr = (close_arr - entry_price) / entry_price * 100.0

    rows = []
    peak_price = entry_price
    peak_idx = 0
    consecutive_down = 0

    for i in range(len(trade_bars)):
        bar = trade_bars[i]
        price = close_prices[i]

        if price > peak_price:
            peak_price = price
            peak_idx = i
            consecutive_down = 0
        elif i > 0 and price < close_prices[i - 1]:
            consecutive_down += 1
        elif i > 0:
            consecutive_down = 0

        pnl_pct = pnl_arr[i]
        mfe_pct = (peak_price - entry_price) / entry_price * 100.0
        dd_pct = (peak_price - price) / peak_price * 100.0 if peak_price > 0 else 0.0
        mins = i  # minutes since entry (1 bar ≈ 1 min)

        def velocity(lookback):
            if i < lookback or close_prices[i - lookback] <= 0:
                return 0.0
            return (price - close_prices[i - lookback]) / close_prices[i - lookback] * 100.0 / lookback

        v5 = velocity(5)
        v10 = velocity(10)
        v15 = velocity(15)

        bar_range_pct = ((bar["high"] - bar["low"]) / price * 100.0
                         if price > 0 and bar["high"] > 0 and bar["low"] > 0 else 0.0)

        vol = bar["volume"] or 0
        vol_window = [trade_bars[j]["volume"] for j in range(max(0, i - 19), i + 1)
                      if trade_bars[j]["volume"] and trade_bars[j]["volume"] > 0]
        avg_vol = np.mean(vol_window) if vol_window else 1.0
        volume_vs_avg = vol / avg_vol if avg_vol > 0 else 0.0

        u_price = bar.get("underlying_price")
        underlying_pnl_pct = 0.0
        if u_price and underlying_entry and underlying_entry > 0:
            underlying_pnl_pct = (u_price - underlying_entry) / underlying_entry * 100.0

        underlying_velocity_5m = 0.0
        if i >= 5:
            u_past = trade_bars[i - 5].get("underlying_price")
            if u_past and u_price and u_past > 0:
                underlying_velocity_5m = (u_price - u_past) / u_past * 100.0 / 5

        in_sweet = 1 if SWEET_SPOT_START_MIN <= mins <= SWEET_SPOT_END_MIN else 0
        past_sweet = 1 if mins > SWEET_SPOT_END_MIN else 0
        mins_past = max(0, mins - SWEET_SPOT_END_MIN)
        time_pressure = min(1.0, mins / HARD_DEADLINE_MIN)

        accel = 0.0
        if i >= 10:
            p5 = close_prices[i - 5] if i >= 5 else close_prices[0]
            p10 = close_prices[i - 10] if i >= 10 else close_prices[0]
            if p5 > 0 and p10 > 0:
                vb = (p5 - p10) / p10 * 100.0 / 5
                accel = v5 - vb

        mfe_retrace = 0.0
        if mfe_pct > 1.0:
            mfe_retrace = min(dd_pct / mfe_pct, 2.0) if mfe_pct > 0 else 0.0

        roll_vol = 0.0
        if i >= 10:
            rets = []
            for j in range(i - 9, i + 1):
                if j > 0 and close_prices[j - 1] > 0:
                    rets.append((close_prices[j] - close_prices[j - 1]) / close_prices[j - 1] * 100)
            if rets:
                roll_vol = float(np.std(rets))

        rr = mfe_pct / max(dd_pct, 1.0)
        bars_since_high = i - peak_idx

        # ─── Labels (hindsight) ──────────────────────────────────────
        future_pnls = pnl_arr[i:]
        valid_future = future_pnls[~np.isnan(future_pnls)]
        if len(valid_future) < 3:
            continue

        future_max = float(np.nanmax(valid_future))
        future_min = float(np.nanmin(valid_future))
        current_pnl = float(pnl_pct)
        upside = future_max - current_pnl
        regret = future_min - current_pnl

        should_sell = 0
        if upside < UPSIDE_THRESHOLD_PCT:
            should_sell = 1
        elif mins > SWEET_SPOT_END_MIN and upside < 10.0:
            should_sell = 1
        elif mins > SWEET_SPOT_END_MIN and regret < -20.0:
            should_sell = 1
        elif mins > HARD_DEADLINE_MIN and upside < 20.0:
            should_sell = 1
        elif current_pnl > 20.0 and mfe_retrace > 0.5 and upside < current_pnl * 0.3:
            should_sell = 1
        elif current_pnl < -30.0 and upside < 15.0:
            should_sell = 1

        expected_future_pnl = float(np.nanmean(valid_future)) - current_pnl

        rows.append({
            "pnl_pct": pnl_pct, "mfe_pct": mfe_pct,
            "drawdown_from_peak_pct": dd_pct,
            "minutes_since_entry": mins,
            "hour_of_day": bar["hour"], "minute_of_hour": bar["minute"],
            "in_sweet_spot": in_sweet, "past_sweet_spot": past_sweet,
            "minutes_past_sweet_spot": mins_past, "time_pressure": time_pressure,
            "premium_velocity_5m": v5, "premium_velocity_10m": v10,
            "premium_velocity_15m": v15, "pnl_acceleration": accel,
            "mfe_retracement_ratio": mfe_retrace,
            "bar_range_pct": bar_range_pct, "volume": vol,
            "volume_vs_avg": volume_vs_avg,
            "rolling_volatility_10m": roll_vol,
            "underlying_pnl_pct": underlying_pnl_pct,
            "underlying_velocity_5m": underlying_velocity_5m,
            "is_call": 1 if is_call else 0,
            "entry_premium": entry_price,
            "bars_since_new_high": bars_since_high,
            "consecutive_down_bars": consecutive_down,
            "risk_reward_ratio": rr,
            "ticker_encoded": ticker_code,
            # Labels
            "should_sell": should_sell,
            "expected_future_pnl": expected_future_pnl,
            # Metadata
            "_current_pnl": current_pnl,
            "_price": price,
            "_entry_price": entry_price,
        })

    return rows if len(rows) >= MIN_BARS else None


# ─── Dataset Building ────────────────────────────────────────────────────────

def build_dataset(signal_only: bool = False):
    """Build training dataset from harvester data.

    If signal_only=True, only use real Discord signal entry times.
    Otherwise, multi-sample each contract's data at 15-min intervals.
    """
    print("=" * 70)
    print("  Building ML dataset from harvester data")
    print("=" * 70)

    ticker_encoder = LabelEncoder()
    ticker_encoder.fit(ALL_TICKERS)

    h_conn = sqlite3.connect(HARVESTER_DB)

    # Get (contract_ticker, date) pairs from harvester
    # Filter: enough bars, has volume (actually traded), and ticker in our list
    print("  Querying harvester DB...")
    ticker_pattern = "|".join(ALL_TICKERS)
    all_contracts = h_conn.execute("""
        SELECT contract_ticker, DATE(captured_at) as d, COUNT(*) as n,
               AVG(day_volume) as avg_vol
        FROM harvest_snapshots
        WHERE day_volume > 0 OR last_trade_price > 0
        GROUP BY contract_ticker, d
        HAVING n >= ?
        ORDER BY d, contract_ticker
    """, (MIN_BARS,)).fetchall()
    # Filter to known tickers
    filtered = []
    for ct, d, n, avg_vol in all_contracts:
        parts = ct.split(":")
        if len(parts) != 2:
            continue
        raw = parts[1]
        ticker = ""
        for ci, ch in enumerate(raw):
            if ch.isdigit():
                ticker = raw[:ci]
                break
        if ticker in ALL_TICKERS:
            filtered.append((ct, d))
    all_contracts = filtered
    print(f"  Harvester: {len(all_contracts)} (contract, date) pairs with >= {MIN_BARS} bars and known tickers")

    # Load real signals for matching
    signals = load_all_signals()
    signal_contracts = set()
    for s in signals:
        ct = build_contract_ticker(s["ticker"], s["expiry_date"], s["strike"], s["option_type"])
        signal_contracts.add((ct, s["date"]))
    print(f"  Real signals: {len(signals)} → {len(signal_contracts)} unique (contract, date) pairs")

    all_rows = []
    trades_meta = []
    processed = 0
    skipped = 0

    for idx, (contract_ticker, date) in enumerate(all_contracts):
        # Parse ticker from contract_ticker: O:SPY260417C00709000
        parts = contract_ticker.split(":")
        if len(parts) != 2:
            skipped += 1
            continue
        raw = parts[1]
        # Find where the date digits start (6 digits YYMMDD)
        ticker = ""
        for ci, ch in enumerate(raw):
            if ch.isdigit():
                ticker = raw[:ci]
                break
        if not ticker or ticker not in ALL_TICKERS:
            skipped += 1
            continue

        is_call = "C" in raw[len(ticker) + 6:len(ticker) + 7]
        ticker_code = int(ticker_encoder.transform([ticker])[0])

        bars = load_harvester_snapshots(contract_ticker, date, h_conn)
        if len(bars) < MIN_BARS:
            skipped += 1
            continue

        is_real_signal = (contract_ticker, date) in signal_contracts

        if signal_only and not is_real_signal:
            skipped += 1
            continue

        # Determine entry points
        if signal_only:
            # Only use the real signal entry time (first bar)
            entry_indices = [0]
        else:
            # Multi-sample: create entries every MULTI_SAMPLE_INTERVAL_MIN minutes
            # ensuring at least MIN_BARS_AFTER_ENTRY bars after each entry
            max_entry = len(bars) - MIN_BARS_AFTER_ENTRY
            entry_indices = list(range(0, max(1, max_entry), MULTI_SAMPLE_INTERVAL_MIN))

        for entry_idx in entry_indices:
            rows = extract_features(bars, entry_idx, ticker_code, is_call)
            if rows is None:
                continue

            start_idx = len(all_rows)
            all_rows.extend(rows)
            trades_meta.append({
                "date": date,
                "ticker": ticker,
                "direction": "call" if is_call else "put",
                "num_rows": len(rows),
                "start_idx": start_idx,
                "end_idx": start_idx + len(rows),
                "is_real_signal": is_real_signal,
                "entry_bar": entry_idx,
            })
            processed += 1

        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx + 1}/{len(all_contracts)} contracts "
                  f"({processed} trades, {len(all_rows):,} rows)")

    h_conn.close()

    print(f"\n  Done: {processed} trades, {len(all_rows):,} feature rows "
          f"({skipped} skipped)")

    if not all_rows:
        print("  ERROR: No data extracted!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    trades_df = pd.DataFrame(trades_meta)

    sell_pct = df["should_sell"].mean() * 100
    real_trades = trades_df[trades_df["is_real_signal"]].shape[0]
    print(f"  Label distribution: SELL={sell_pct:.1f}%, HOLD={100-sell_pct:.1f}%")
    print(f"  Date range: {trades_df['date'].min()} to {trades_df['date'].max()}")
    print(f"  Real signal trades: {real_trades} / {processed} total")
    print(f"  Unique tickers: {trades_df['ticker'].nunique()}")

    return df, trades_df, ticker_encoder


# ─── Training ────────────────────────────────────────────────────────────────

def train_models(df, trades_df, feature_cols, model_prefix="generic"):
    """Train classifier + regressor and save models."""
    print(f"\n{'='*70}")
    print(f"  Training models: {model_prefix}")
    print(f"{'='*70}")

    # Walk-forward date split
    unique_dates = sorted(trades_df["date"].unique())
    n = len(unique_dates)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    print(f"  Split: train={unique_dates[0]}..{unique_dates[train_end-1]} ({train_end}d), "
          f"val={unique_dates[train_end]}..{unique_dates[val_end-1]} ({val_end-train_end}d), "
          f"test={unique_dates[val_end]}..{unique_dates[-1]} ({n-val_end}d)")

    row_dates = []
    for _, trade in trades_df.iterrows():
        row_dates.extend([trade["date"]] * trade["num_rows"])
    row_dates = np.array(row_dates)

    train_mask = np.isin(row_dates, list(train_dates))
    val_mask = np.isin(row_dates, list(val_dates))
    test_mask = np.isin(row_dates, list(test_dates))

    X = df[feature_cols]
    y_clf = df["should_sell"]
    y_reg = df["expected_future_pnl"]

    X_train, y_train = X[train_mask], y_clf[train_mask]
    X_val, y_val = X[val_mask], y_clf[val_mask]
    X_test, y_test = X[test_mask], y_clf[test_mask]

    print(f"  Samples: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    if len(X_train) < 100 or len(X_val) < 50:
        print(f"  SKIP: not enough data for {model_prefix}")
        return None, None

    # ─── Classifier ──────────────────────────────────────────────────
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
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
        "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    print("  Training classifier...")
    clf = lgb.train(
        clf_params, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(80)],
    )
    print(f"  Classifier best iteration: {clf.best_iteration}")

    # ─── Regressor ───────────────────────────────────────────────────
    y_reg_train = y_reg[train_mask]
    y_reg_val = y_reg[val_mask]
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

    print("  Training regressor...")
    reg = lgb.train(
        reg_params, dtrain_reg,
        num_boost_round=2000,
        valid_sets=[dtrain_reg, dval_reg],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(80)],
    )
    print(f"  Regressor best iteration: {reg.best_iteration}")

    # ─── Evaluate ────────────────────────────────────────────────────
    SELL_THRESHOLD = 0.4
    test_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
    test_binary = (test_pred > SELL_THRESHOLD).astype(int)

    print(f"\n  Test results (threshold={SELL_THRESHOLD}):")
    print(classification_report(y_test, test_binary, target_names=["HOLD", "SELL"]))

    test_reg_pred = reg.predict(X_test, num_iteration=reg.best_iteration)
    test_reg_actual = y_reg[test_mask].values
    mae = np.mean(np.abs(test_reg_pred - test_reg_actual))
    print(f"  Regressor MAE: {mae:.2f}%")

    # Feature importance
    importance = clf.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\n  Top 10 features:")
    for name, imp in feat_imp[:10]:
        bar = "#" * int(imp / feat_imp[0][1] * 30)
        print(f"    {name:30s} {imp:>10.0f}  {bar}")

    # ─── Save ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    clf_path = os.path.join(MODEL_DIR, f"{model_prefix}_clf.lgb")
    reg_path = os.path.join(MODEL_DIR, f"{model_prefix}_reg.lgb")

    clf.save_model(clf_path, num_iteration=clf.best_iteration)
    reg.save_model(reg_path, num_iteration=reg.best_iteration)
    print(f"\n  Saved: {clf_path}")
    print(f"  Saved: {reg_path}")

    return clf, reg


def main():
    parser = argparse.ArgumentParser(description="Train ML models on harvester data")
    parser.add_argument("--signal-only", action="store_true",
                        help="Only use real signal entry times (no multi-sampling)")
    args = parser.parse_args()

    df, trades_df, ticker_encoder = build_dataset(signal_only=args.signal_only)

    # Train generic model (all tickers)
    print("\n" + "=" * 70)
    print("  TRAINING GENERIC MODEL (all tickers)")
    print("=" * 70)
    train_models(df, trades_df, GENERIC_FEATURE_COLS, model_prefix="generic")

    # Train per-ticker models for tickers with enough data
    PER_TICKER_MIN_TRADES = 20
    ticker_counts = trades_df["ticker"].value_counts()
    print(f"\n  Ticker trade counts:")
    for ticker, count in ticker_counts.items():
        print(f"    {ticker}: {count} trades")

    for ticker, count in ticker_counts.items():
        if count < PER_TICKER_MIN_TRADES:
            print(f"\n  SKIP {ticker}: only {count} trades (need {PER_TICKER_MIN_TRADES})")
            continue

        ticker_trades = trades_df[trades_df["ticker"] == ticker]

        # Build per-ticker dataset (subset of rows)
        ticker_indices = []
        for _, trade in ticker_trades.iterrows():
            ticker_indices.extend(range(trade["start_idx"], trade["end_idx"]))

        ticker_df = df.iloc[ticker_indices].reset_index(drop=True)
        ticker_trades_df = ticker_trades.reset_index(drop=True)

        # Recompute start/end indices for the subset
        new_meta = []
        offset = 0
        for _, trade in ticker_trades_df.iterrows():
            n = trade["num_rows"]
            new_meta.append({
                "date": trade["date"],
                "ticker": trade["ticker"],
                "direction": trade["direction"],
                "num_rows": n,
                "start_idx": offset,
                "end_idx": offset + n,
                "is_real_signal": trade["is_real_signal"],
            })
            offset += n
        new_trades_df = pd.DataFrame(new_meta)

        train_models(ticker_df, new_trades_df, FEATURE_COLS,
                     model_prefix=ticker.lower())

    print(f"\n{'='*70}")
    print(f"  ALL MODELS TRAINED")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
