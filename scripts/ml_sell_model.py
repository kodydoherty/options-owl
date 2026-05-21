"""ML-based optimal sell timing model for 0DTE options — V2 (tuned).

Key insight from data analysis: optimal profits are in the 60-150 minute
window after entry. Before 60 min the move hasn't developed; after 150 min
theta decay accelerates and probability of loss climbs rapidly.

V2 improvements over V1:
- Smarter labels using time-aware optimal exit windows
- Regression target (expected future PnL) + binary classifier
- More features: time_in_sweet_spot, pnl_momentum, mfe_retracement_ratio
- Better hyperparams (deeper trees, more rounds, lower LR)
- Lower sell threshold (0.4 instead of 0.5) for earlier exits
- Two-model approach: when-to-sell classifier + expected-upside regressor

Usage:
    python scripts/ml_sell_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import sqlite3
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# Check for lightgbm
try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm is not installed. Install it with:")
    print("  pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "journal", "historical_0dte.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "journal", "ml_sell_model.lgb")
IMPORTANCE_PATH = os.path.join(PROJECT_DIR, "journal", "ml_feature_importance.png")
BACKTEST_PATH = os.path.join(PROJECT_DIR, "journal", "ml_backtest_comparison.csv")

# ─── Constants ───────────────────────────────────────────────────────────────
# Entry at 10:00 AM ET = 14:00 UTC (EDT offset = -4h)
ENTRY_HOUR_ET = 10
ENTRY_MINUTE_ET = 0
ENTRY_UTC_MS = None  # computed per day from bars

# Deadline 3:30 PM ET = 19:30 UTC
DEADLINE_HOUR_ET = 15
DEADLINE_MINUTE_ET = 30

# Label thresholds — V2 tuned
UPSIDE_THRESHOLD_PCT = 3.0      # sell if upside < 3% (tighter than v1's 5%)
REGRET_THRESHOLD_PCT = -10.0    # sell if regret < -10% (tighter than v1's -15%)

# Time-based sweet spot (from data analysis)
SWEET_SPOT_START_MIN = 45       # move starts developing
SWEET_SPOT_PEAK_MIN = 90        # peak avg PnL zone
SWEET_SPOT_END_MIN = 150        # theta decay kicks in hard
HARD_DEADLINE_MIN = 240         # after 4 hours, almost always sell

MIN_OPTION_BARS = 30            # skip days with fewer bars

TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "META", "AMD",
           "AMZN", "GOOGL", "MSFT", "IWM", "MU", "MSTR"]

FEATURE_COLS = [
    # Core PnL features
    "pnl_pct", "mfe_pct", "drawdown_from_peak_pct",
    # Time features
    "minutes_since_entry", "hour_of_day", "minute_of_hour",
    "in_sweet_spot", "past_sweet_spot", "minutes_past_sweet_spot",
    "time_pressure",  # 0 to 1, increases as we approach hard deadline
    # Momentum features
    "premium_velocity_5m", "premium_velocity_10m", "premium_velocity_15m",
    "pnl_acceleration",  # velocity change (2nd derivative)
    "mfe_retracement_ratio",  # how much of MFE have we given back? 0=at peak, 1=back to entry
    # Volatility / volume
    "bar_range_pct", "volume", "volume_vs_avg",
    "rolling_volatility_10m",  # stddev of 1-min returns over 10 bars
    # Underlying
    "underlying_pnl_pct", "underlying_velocity_5m",
    # Identifiers
    "ticker_encoded", "is_call", "entry_premium",
    # Staleness
    "bars_since_new_high", "consecutive_down_bars",
    # Risk/reward ratio features
    "risk_reward_ratio",  # mfe_pct / max(drawdown, 1) — are we giving back gains?
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ts_to_et(timestamp_ms):
    """Convert unix ms timestamp to ET datetime (UTC - 4h for EDT)."""
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def get_db_connection():
    """Get a read-only SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_trading_days():
    """Load all trading days sorted by date, grouped by (ticker, date)."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT ticker, date, open_price, close_price, high_price, low_price,
               atm_call_ticker, atm_put_ticker, atm_strike,
               call_bars, put_bars, underlying_bars
        FROM trading_days
        WHERE call_bars >= ? AND put_bars >= ?
        ORDER BY date, ticker
    """, (MIN_OPTION_BARS, MIN_OPTION_BARS)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_option_bars(conn, contract_ticker):
    """Load 1-minute option bars for a contract."""
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    return [dict(r) for r in rows]


def load_underlying_bars(conn, ticker, date):
    """Load 1-minute underlying bars for a ticker on a date."""
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM underlying_bars
        WHERE ticker = ? AND date = ?
        ORDER BY timestamp
    """, (ticker, date)).fetchall()
    return [dict(r) for r in rows]


def find_entry_idx(bars, entry_hour_et=ENTRY_HOUR_ET, entry_minute_et=ENTRY_MINUTE_ET):
    """Find first bar at or after entry time (10:00 AM ET)."""
    target_minutes = entry_hour_et * 60 + entry_minute_et
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_minutes = et.hour * 60 + et.minute
        if bar_minutes >= target_minutes:
            return i
    return None


def find_deadline_idx(bars, deadline_hour_et=DEADLINE_HOUR_ET,
                      deadline_minute_et=DEADLINE_MINUTE_ET):
    """Find last bar before the deadline (3:30 PM ET)."""
    target_minutes = deadline_hour_et * 60 + deadline_minute_et
    last_valid = None
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_minutes = et.hour * 60 + et.minute
        if bar_minutes < target_minutes:
            last_valid = i
        elif bar_minutes >= target_minutes:
            # Include this bar (it's the deadline bar) then stop searching
            return i
    return last_valid


def match_underlying_price(underlying_bars, target_ts):
    """Find underlying close price closest to the target timestamp."""
    if not underlying_bars:
        return None
    best = None
    best_diff = float("inf")
    for bar in underlying_bars:
        diff = abs(bar["timestamp"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = bar["close"]
    return best


# ─── Step 1 & 2: Feature extraction and label generation ────────────────────

def extract_features_for_trade(option_bars, underlying_bars, entry_idx,
                               deadline_idx, ticker_code, is_call):
    """Extract per-minute features and labels for a single trade.

    Returns a list of dicts (one per bar from entry to deadline), or None
    if insufficient data.
    """
    if deadline_idx is None or entry_idx is None:
        return None
    if deadline_idx <= entry_idx:
        return None

    bars = option_bars[entry_idx:deadline_idx + 1]
    if len(bars) < 2:
        return None

    entry_price = bars[0]["close"]
    if entry_price <= 0:
        # Try vwap
        entry_price = bars[0]["vwap"] if bars[0]["vwap"] > 0 else 0
    if entry_price <= 0:
        return None

    # Get underlying entry price
    underlying_entry_price = match_underlying_price(
        underlying_bars, bars[0]["timestamp"]
    )

    # Pre-compute close prices for label generation
    close_prices = []
    for b in bars:
        c = b["close"] if b["close"] > 0 else b["vwap"]
        close_prices.append(c if c > 0 else np.nan)
    close_arr = np.array(close_prices, dtype=np.float64)

    # Pre-compute PnL % array
    pnl_arr = (close_arr - entry_price) / entry_price * 100.0

    # Build underlying price lookup: map each option bar to closest underlying
    underlying_prices = []
    for b in bars:
        up = match_underlying_price(underlying_bars, b["timestamp"])
        underlying_prices.append(up)

    rows = []
    peak_price = entry_price
    peak_idx_in_window = 0  # index within bars where peak was set
    consecutive_down = 0

    for i in range(len(bars)):
        bar = bars[i]
        price = close_prices[i]
        if np.isnan(price):
            continue

        # Track peak
        if price > peak_price:
            peak_price = price
            peak_idx_in_window = i
            consecutive_down = 0
        else:
            if i > 0 and not np.isnan(close_prices[i - 1]) and price < close_prices[i - 1]:
                consecutive_down += 1
            elif i > 0 and not np.isnan(close_prices[i - 1]) and price >= close_prices[i - 1]:
                consecutive_down = 0

        pnl_pct = pnl_arr[i]
        mfe_pct = (peak_price - entry_price) / entry_price * 100.0
        drawdown_from_peak_pct = (peak_price - price) / peak_price * 100.0 if peak_price > 0 else 0.0

        et = ts_to_et(bar["timestamp"])
        minutes_since_entry = i
        hour_of_day = et.hour
        minute_of_hour = et.minute

        # Premium velocities
        def velocity(lookback):
            if i < lookback:
                return 0.0
            past = close_prices[i - lookback]
            if np.isnan(past) or past == 0:
                return 0.0
            return (price - past) / past * 100.0 / lookback

        premium_velocity_5m = velocity(5)
        premium_velocity_10m = velocity(10)
        premium_velocity_15m = velocity(15)

        # Bar range
        bar_range_pct = ((bar["high"] - bar["low"]) / price * 100.0
                         if price > 0 and bar["high"] > 0 and bar["low"] > 0
                         else 0.0)

        # Volume
        vol = bar["volume"] if bar["volume"] else 0
        # Rolling 20-bar average volume
        vol_window = [bars[j]["volume"] for j in range(max(0, i - 19), i + 1)
                      if bars[j]["volume"] and bars[j]["volume"] > 0]
        avg_vol = np.mean(vol_window) if vol_window else 1.0
        volume_vs_avg = vol / avg_vol if avg_vol > 0 else 0.0

        # Underlying
        u_price = underlying_prices[i]
        underlying_pnl_pct = 0.0
        if u_price is not None and underlying_entry_price is not None and underlying_entry_price > 0:
            underlying_pnl_pct = (u_price - underlying_entry_price) / underlying_entry_price * 100.0

        # Underlying velocity (5-bar)
        underlying_velocity_5m = 0.0
        if i >= 5:
            u_past = underlying_prices[i - 5]
            if u_past is not None and u_price is not None and u_past > 0:
                underlying_velocity_5m = (u_price - u_past) / u_past * 100.0 / 5

        bars_since_new_high = i - peak_idx_in_window

        # ─── New V2 features ────────────────────────────────────────
        # Time-aware features
        in_sweet_spot = 1 if SWEET_SPOT_START_MIN <= minutes_since_entry <= SWEET_SPOT_END_MIN else 0
        past_sweet_spot = 1 if minutes_since_entry > SWEET_SPOT_END_MIN else 0
        minutes_past_sweet_spot = max(0, minutes_since_entry - SWEET_SPOT_END_MIN)

        # Time pressure: 0 at entry, 1 at hard deadline
        time_pressure = min(1.0, minutes_since_entry / HARD_DEADLINE_MIN)

        # PnL acceleration (2nd derivative of premium)
        pnl_acceleration = 0.0
        if i >= 10:
            vel_now = velocity(5)
            past5 = close_prices[i - 5]
            past10 = close_prices[i - 10]
            if not np.isnan(past5) and not np.isnan(past10) and past10 > 0 and past5 > 0:
                vel_before = (past5 - past10) / past10 * 100.0 / 5
                pnl_acceleration = vel_now - vel_before

        # MFE retracement ratio: 0 = at peak, 1 = given back all gains to entry
        if mfe_pct > 1.0:  # only meaningful if there was a real gain
            mfe_retracement_ratio = drawdown_from_peak_pct / mfe_pct if mfe_pct > 0 else 0.0
            mfe_retracement_ratio = min(mfe_retracement_ratio, 2.0)  # cap at 2x
        else:
            mfe_retracement_ratio = 0.0

        # Rolling volatility (stddev of 1-min returns over 10 bars)
        rolling_volatility_10m = 0.0
        if i >= 10:
            returns = []
            for j in range(i - 9, i + 1):
                if j > 0 and not np.isnan(close_prices[j]) and not np.isnan(close_prices[j-1]) and close_prices[j-1] > 0:
                    returns.append((close_prices[j] - close_prices[j-1]) / close_prices[j-1] * 100)
            if returns:
                rolling_volatility_10m = float(np.std(returns))

        # Risk/reward ratio
        risk_reward_ratio = mfe_pct / max(drawdown_from_peak_pct, 1.0)

        # ─── Labels (hindsight) — V2 smarter labels ─────────────────
        future_pnls = pnl_arr[i:]
        valid_future = future_pnls[~np.isnan(future_pnls)]
        if len(valid_future) == 0:
            continue

        future_max_pnl = float(np.nanmax(valid_future))
        future_min_pnl = float(np.nanmin(valid_future))
        current_pnl = pnl_pct

        upside_of_holding = future_max_pnl - current_pnl
        regret_of_holding = future_min_pnl - current_pnl

        # V2 label logic: time-aware sell signals
        should_sell = 0

        # Rule 1: No upside left — always sell
        if upside_of_holding < UPSIDE_THRESHOLD_PCT:
            should_sell = 1

        # Rule 2: Past sweet spot + losing momentum — sell
        elif minutes_since_entry > SWEET_SPOT_END_MIN:
            if upside_of_holding < 10.0:  # not much juice left
                should_sell = 1
            elif regret_of_holding < -20.0:  # big downside risk
                should_sell = 1

        # Rule 3: Way past deadline (4+ hours) — almost always sell
        elif minutes_since_entry > HARD_DEADLINE_MIN:
            if upside_of_holding < 20.0:
                should_sell = 1

        # Rule 4: Giving back gains — sell to protect profits
        elif current_pnl > 20.0 and mfe_retracement_ratio > 0.5:
            if upside_of_holding < current_pnl * 0.3:  # upside < 30% of current gain
                should_sell = 1

        # Rule 5: Deep in the hole with no recovery coming
        elif current_pnl < -30.0 and upside_of_holding < 15.0:
            should_sell = 1

        # Regression target: expected future PnL change from this point
        # (positive = holding is good, negative = should have sold)
        expected_future_pnl = float(np.nanmean(valid_future)) - current_pnl

        row = {
            # Core PnL
            "pnl_pct": pnl_pct,
            "mfe_pct": mfe_pct,
            "drawdown_from_peak_pct": drawdown_from_peak_pct,
            # Time
            "minutes_since_entry": minutes_since_entry,
            "hour_of_day": hour_of_day,
            "minute_of_hour": minute_of_hour,
            "in_sweet_spot": in_sweet_spot,
            "past_sweet_spot": past_sweet_spot,
            "minutes_past_sweet_spot": minutes_past_sweet_spot,
            "time_pressure": time_pressure,
            # Momentum
            "premium_velocity_5m": premium_velocity_5m,
            "premium_velocity_10m": premium_velocity_10m,
            "premium_velocity_15m": premium_velocity_15m,
            "pnl_acceleration": pnl_acceleration,
            "mfe_retracement_ratio": mfe_retracement_ratio,
            # Volatility / volume
            "bar_range_pct": bar_range_pct,
            "volume": vol,
            "volume_vs_avg": volume_vs_avg,
            "rolling_volatility_10m": rolling_volatility_10m,
            # Underlying
            "underlying_pnl_pct": underlying_pnl_pct,
            "underlying_velocity_5m": underlying_velocity_5m,
            # Identifiers
            "ticker_encoded": ticker_code,
            "is_call": 1 if is_call else 0,
            "entry_premium": entry_price,
            # Staleness
            "bars_since_new_high": bars_since_new_high,
            "consecutive_down_bars": consecutive_down,
            # Risk/reward
            "risk_reward_ratio": risk_reward_ratio,
            # Labels
            "should_sell": should_sell,
            "expected_future_pnl": expected_future_pnl,
            "future_max_pnl": future_max_pnl,
            "future_min_pnl": future_min_pnl,
            "upside_of_holding": upside_of_holding,
            "regret_of_holding": regret_of_holding,
            # Metadata (not features)
            "_current_pnl": current_pnl,
            "_price": price,
            "_entry_price": entry_price,
            "_timestamp": bar["timestamp"],
        }
        rows.append(row)

    return rows if rows else None


def build_dataset():
    """Build the full feature/label dataset from all trading days."""
    print("=" * 70)
    print("  STEP 1 & 2: Feature extraction + label generation")
    print("=" * 70)

    trading_days = load_all_trading_days()
    print(f"  Loaded {len(trading_days)} trading day/ticker combos")

    # Build ticker encoder
    ticker_encoder = LabelEncoder()
    ticker_encoder.fit(TICKERS)

    all_rows = []
    trades_meta = []  # (date, ticker, direction, num_rows, start_idx)
    skipped = 0
    processed = 0

    conn = get_db_connection()

    for day_idx, day in enumerate(trading_days):
        ticker = day["ticker"]
        date = day["date"]

        # Encode ticker
        if ticker not in TICKERS:
            skipped += 1
            continue
        ticker_code = int(ticker_encoder.transform([ticker])[0])

        # Load underlying bars for this day
        u_bars = load_underlying_bars(conn, ticker, date)

        # Process both call and put
        for direction, contract_key in [("call", "atm_call_ticker"),
                                        ("put", "atm_put_ticker")]:
            contract_ticker = day[contract_key]
            if not contract_ticker:
                skipped += 1
                continue

            o_bars = load_option_bars(conn, contract_ticker)
            if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                skipped += 1
                continue

            entry_idx = find_entry_idx(o_bars)
            deadline_idx = find_deadline_idx(o_bars)
            if entry_idx is None or deadline_idx is None:
                skipped += 1
                continue

            is_call = direction == "call"
            rows = extract_features_for_trade(
                o_bars, u_bars, entry_idx, deadline_idx,
                ticker_code, is_call
            )
            if rows is None or len(rows) < 10:
                skipped += 1
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

        if (day_idx + 1) % 100 == 0:
            print(f"    Processed {day_idx + 1}/{len(trading_days)} days "
                  f"({processed} trades, {len(all_rows)} rows so far)")

    conn.close()

    print(f"\n  Done: {processed} trades, {len(all_rows)} feature rows "
          f"({skipped} skipped)")

    if not all_rows:
        print("  ERROR: No data extracted!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    trades_df = pd.DataFrame(trades_meta)

    # Stats
    sell_pct = df["should_sell"].mean() * 100
    print(f"  Label distribution: SELL={sell_pct:.1f}%, HOLD={100-sell_pct:.1f}%")
    print(f"  Date range: {trades_df['date'].min()} to {trades_df['date'].max()}")

    return df, trades_df, ticker_encoder


# ─── Step 3: Train model ────────────────────────────────────────────────────

def train_model(df, trades_df):
    """Train LightGBM with walk-forward split on dates."""
    print("\n" + "=" * 70)
    print("  STEP 3: Train LightGBM model")
    print("=" * 70)

    # Sort trades by date for walk-forward split
    unique_dates = sorted(trades_df["date"].unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * 0.70)
    val_end = int(n_dates * 0.85)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    print(f"  Walk-forward split:")
    print(f"    Train: {unique_dates[0]} to {unique_dates[train_end - 1]} "
          f"({train_end} dates)")
    print(f"    Val:   {unique_dates[train_end]} to {unique_dates[val_end - 1]} "
          f"({val_end - train_end} dates)")
    print(f"    Test:  {unique_dates[val_end]} to {unique_dates[-1]} "
          f"({n_dates - val_end} dates)")

    # Map each row to its trade's date
    row_dates = []
    for _, trade in trades_df.iterrows():
        d = trade["date"]
        row_dates.extend([d] * trade["num_rows"])
    row_dates = np.array(row_dates)

    train_mask = np.isin(row_dates, list(train_dates))
    val_mask = np.isin(row_dates, list(val_dates))
    test_mask = np.isin(row_dates, list(test_dates))

    X = df[FEATURE_COLS]
    y = df["should_sell"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n  Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # LightGBM dataset
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,         # deeper trees (v1 was 63)
        "learning_rate": 0.01,     # slower learning for better generalization
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "min_child_samples": 100,  # more conservative splits
        "max_depth": 8,            # prevent overfitting
        "lambda_l1": 0.1,          # L1 regularization
        "lambda_l2": 1.0,          # L2 regularization
        "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    print("  Training LightGBM classifier...")
    callbacks = [
        lgb.log_evaluation(period=200),
        lgb.early_stopping(stopping_rounds=100),  # more patience (v1 was 50)
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,      # more rounds (v1 was 1000)
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")

    # ─── Also train a regressor for expected future PnL ────────────
    print("\n  Training LightGBM regressor (expected future PnL)...")
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
        "min_child_samples": 100,
        "max_depth": 8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    reg_callbacks = [
        lgb.log_evaluation(period=200),
        lgb.early_stopping(stopping_rounds=100),
    ]

    reg_model = lgb.train(
        reg_params,
        dtrain_reg,
        num_boost_round=3000,
        valid_sets=[dtrain_reg, dval_reg],
        valid_names=["train", "val"],
        callbacks=reg_callbacks,
    )

    print(f"  Regressor best iteration: {reg_model.best_iteration}")

    # ─── Evaluate classifier ─────────────────────────────────────────
    # Use lower threshold (0.4) to catch more sell signals
    SELL_THRESHOLD = 0.4

    print(f"\n  --- Validation Set Results (threshold={SELL_THRESHOLD}) ---")
    val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = (val_pred_proba > SELL_THRESHOLD).astype(int)
    print(classification_report(y_val, val_pred, target_names=["HOLD", "SELL"]))

    print(f"  --- Test Set Results (threshold={SELL_THRESHOLD}) ---")
    test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    test_pred = (test_pred_proba > SELL_THRESHOLD).astype(int)
    print(classification_report(y_test, test_pred, target_names=["HOLD", "SELL"]))

    # ─── Regressor evaluation ─────────────────────────────────────
    test_reg_pred = reg_model.predict(X_test, num_iteration=reg_model.best_iteration)
    test_reg_actual = df.loc[test_mask, "expected_future_pnl"].values
    mae = np.mean(np.abs(test_reg_pred - test_reg_actual))
    print(f"\n  Regressor MAE on test set: {mae:.2f}% (predicting future PnL change)")
    print(f"  Avg predicted upside when classifier says HOLD: "
          f"{np.mean(test_reg_pred[test_pred == 0]):+.1f}%")
    print(f"  Avg predicted upside when classifier says SELL: "
          f"{np.mean(test_reg_pred[test_pred == 1]):+.1f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print("  Confusion Matrix (test):")
    print(f"              Predicted")
    print(f"              HOLD   SELL")
    print(f"  Actual HOLD {cm[0][0]:>6} {cm[0][1]:>6}")
    print(f"  Actual SELL {cm[1][0]:>6} {cm[1][1]:>6}")

    # ─── Feature importance ──────────────────────────────────────────
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    print("\n  Feature Importance (top 15):")
    for _, row in feat_imp.head(15).iterrows():
        bar = "#" * int(row["importance"] / feat_imp["importance"].max() * 40)
        print(f"    {row['feature']:30s} {row['importance']:>10.0f}  {bar}")

    return model, reg_model, test_mask, test_pred_proba, test_reg_pred, feat_imp, SELL_THRESHOLD


# ─── Step 4: Backtest ───────────────────────────────────────────────────────

def backtest(df, trades_df, test_mask, model, reg_model, test_pred_proba, test_reg_pred, sell_threshold):
    """Backtest the model against baseline strategies on the test set."""
    print("\n" + "=" * 70)
    print("  STEP 4: Backtest comparison")
    print("=" * 70)

    # Get test trades
    row_dates = []
    for _, trade in trades_df.iterrows():
        row_dates.extend([trade["date"]] * trade["num_rows"])
    row_dates = np.array(row_dates)

    unique_dates = sorted(trades_df["date"].unique())
    n_dates = len(unique_dates)
    val_end = int(n_dates * 0.85)
    test_dates = set(unique_dates[val_end:])

    # Identify test-set trades
    test_trades = trades_df[trades_df["date"].isin(test_dates)].reset_index(drop=True)

    if len(test_trades) == 0:
        print("  No test trades found!")
        return None

    print(f"  Test set: {len(test_trades)} trades across "
          f"{len(test_dates)} dates")

    # For each trade in test set, simulate different strategies
    results = {
        "ml_classifier": [],
        "ml_combo": [],         # classifier + regressor combined
        "hold_to_eod": [],
        "exit_at_90min": [],    # exit at sweet spot peak
        "exit_at_150min": [],   # exit at sweet spot end
        "trailing_25pct_10min": [],
        "trailing_50pct": [],
        "fixed_stop_25pct_10min": [],
        "vinny_phase_trail": [],  # current Vinny strategy
    }

    # Get the test portion of predictions
    test_indices = np.where(test_mask)[0]
    pred_map = {}       # global_idx -> classifier proba
    reg_pred_map = {}   # global_idx -> regressor prediction
    for local_i, global_i in enumerate(test_indices):
        pred_map[global_i] = test_pred_proba[local_i]
        reg_pred_map[global_i] = test_reg_pred[local_i]

    for _, trade in test_trades.iterrows():
        start = trade["start_idx"]
        end = trade["end_idx"]
        trade_rows = df.iloc[start:end]

        if len(trade_rows) < 2:
            continue

        entry_price = trade_rows.iloc[0]["_entry_price"]
        if entry_price <= 0:
            continue

        prices = trade_rows["_price"].values
        pnls = trade_rows["_current_pnl"].values
        eod_pnl = pnls[-1]

        # 1) Hold to EOD
        results["hold_to_eod"].append(eod_pnl)

        # 2) ML Classifier only — exit when P(sell) > threshold
        ml_pnl = eod_pnl
        for i in range(len(trade_rows)):
            global_idx = start + i
            prob = pred_map.get(global_idx, 0.0)
            if prob > sell_threshold:
                ml_pnl = pnls[i]
                break
        results["ml_classifier"].append(ml_pnl)

        # 3) ML Combo — classifier says sell AND regressor says negative future
        combo_pnl = eod_pnl
        for i in range(len(trade_rows)):
            global_idx = start + i
            prob = pred_map.get(global_idx, 0.0)
            reg_val = reg_pred_map.get(global_idx, 10.0)
            # Sell when: classifier confident OR regressor predicts negative future
            if prob > sell_threshold and reg_val < 2.0:
                combo_pnl = pnls[i]
                break
            # Also sell if regressor is very bearish even if classifier isn't sure
            if i >= 10 and reg_val < -10.0:
                combo_pnl = pnls[i]
                break
        results["ml_combo"].append(combo_pnl)

        # 4) Exit at 90 min (sweet spot peak)
        exit_90 = eod_pnl
        if len(pnls) > 90:
            exit_90 = pnls[90]
        results["exit_at_90min"].append(exit_90)

        # 5) Exit at 150 min (end of sweet spot)
        exit_150 = eod_pnl
        if len(pnls) > 150:
            exit_150 = pnls[150]
        results["exit_at_150min"].append(exit_150)

        # 6) 25% trailing stop + 10 min hold
        trail25_pnl = eod_pnl
        peak = entry_price
        for i in range(len(trade_rows)):
            p = prices[i]
            if p > peak:
                peak = p
            if i >= 10 and peak > 0:
                drop = (peak - p) / peak * 100
                if drop >= 25.0:
                    trail25_pnl = pnls[i]
                    break
        results["trailing_25pct_10min"].append(trail25_pnl)

        # 7) 50% trailing stop
        trail_pnl = eod_pnl
        peak = entry_price
        for i in range(len(trade_rows)):
            p = prices[i]
            if p > peak:
                peak = p
            if peak > 0:
                drop = (peak - p) / peak * 100
                if drop >= 50.0:
                    trail_pnl = pnls[i]
                    break
        results["trailing_50pct"].append(trail_pnl)

        # 8) 25% fixed stop + 10 min hold
        fixed_pnl = eod_pnl
        for i in range(len(trade_rows)):
            if i < 10:
                continue
            if pnls[i] <= -25.0:
                fixed_pnl = pnls[i]
                break
        results["fixed_stop_25pct_10min"].append(fixed_pnl)

        # 9) Vinny phase trail: 25% initial trail, tighten after gains
        vinny_pnl = eod_pnl
        peak = entry_price
        phase_trail = 25.0  # phase 0
        for i in range(len(trade_rows)):
            p = prices[i]
            if p > peak:
                peak = p
                # Tighten trail as we hit targets
                gain = (peak - entry_price) / entry_price * 100
                if gain >= 200:
                    phase_trail = 10.0
                elif gain >= 100:
                    phase_trail = 15.0
                elif gain >= 50:
                    phase_trail = 20.0
            if i >= 5 and peak > 0:  # 5 min grace
                drop = (peak - p) / peak * 100
                if drop >= phase_trail:
                    vinny_pnl = pnls[i]
                    break
            # Time decay zone: after 150 min, tighten to 10%
            if i >= 150 and peak > 0:
                drop = (peak - p) / peak * 100
                if drop >= 10.0:
                    vinny_pnl = pnls[i]
                    break
        results["vinny_phase_trail"].append(vinny_pnl)

    # ─── Comparison table ────────────────────────────────────────────
    print("\n  Strategy Comparison (test set):")
    print(f"  {'Strategy':<30s} {'Trades':>6} {'Total PnL%':>12} "
          f"{'Avg PnL%':>10} {'Win Rate':>10} {'Avg Win%':>10} {'Avg Loss%':>10}")
    print("  " + "-" * 90)

    comparison_rows = []
    for name, pnls in results.items():
        pnls = np.array(pnls)
        n = len(pnls)
        total = np.sum(pnls)
        avg = np.mean(pnls) if n > 0 else 0
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        print(f"  {name:<30s} {n:>6} {total:>+11.1f}% "
              f"{avg:>+9.2f}% {wr:>9.1f}% {avg_win:>+9.2f}% {avg_loss:>+9.2f}%")

        comparison_rows.append({
            "strategy": name,
            "num_trades": n,
            "total_pnl_pct": round(total, 2),
            "avg_pnl_pct": round(avg, 2),
            "win_rate_pct": round(wr, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
        })

    comp_df = pd.DataFrame(comparison_rows)
    return comp_df


# ─── Step 5: Save outputs ───────────────────────────────────────────────────

REG_MODEL_PATH = os.path.join(PROJECT_DIR, "journal", "ml_sell_regressor.lgb")


def save_outputs(model, reg_model, feat_imp, comp_df):
    """Save model, feature importance chart, and backtest comparison."""
    print("\n" + "=" * 70)
    print("  STEP 5: Save outputs")
    print("=" * 70)

    # Save models
    model.save_model(MODEL_PATH)
    print(f"  Classifier saved to: {MODEL_PATH}")
    reg_model.save_model(REG_MODEL_PATH)
    print(f"  Regressor saved to: {REG_MODEL_PATH}")

    # Save feature importance chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        feat_imp_sorted = feat_imp.sort_values("importance", ascending=True)
        ax.barh(feat_imp_sorted["feature"], feat_imp_sorted["importance"])
        ax.set_xlabel("Importance (gain)")
        ax.set_title("LightGBM Feature Importance — 0DTE Sell Timing Model")
        plt.tight_layout()
        fig.savefig(IMPORTANCE_PATH, dpi=150)
        plt.close(fig)
        print(f"  Feature importance chart saved to: {IMPORTANCE_PATH}")
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping chart")

    # Save backtest comparison
    if comp_df is not None:
        comp_df.to_csv(BACKTEST_PATH, index=False)
        print(f"  Backtest comparison saved to: {BACKTEST_PATH}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    print()
    print("=" * 70)
    print("  OptionsOwl — ML Sell Timing Model for 0DTE Options")
    print("=" * 70)
    print(f"  Database: {DB_PATH}")
    print(f"  Entry: 10:00 AM ET | Deadline: 3:30 PM ET")
    print()

    # Check DB exists
    if not os.path.exists(DB_PATH):
        print(f"  ERROR: Database not found at {DB_PATH}")
        print("  Run download_historical_0dte.py first.")
        sys.exit(1)

    # Step 1 & 2: Build dataset
    df, trades_df, ticker_encoder = build_dataset()

    # Step 3: Train model
    model, reg_model, test_mask, test_pred_proba, test_reg_pred, feat_imp, sell_threshold = train_model(df, trades_df)

    # Step 4: Backtest
    comp_df = backtest(df, trades_df, test_mask, model, reg_model, test_pred_proba, test_reg_pred, sell_threshold)

    # Step 5: Save
    save_outputs(model, reg_model, feat_imp, comp_df)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print("  Done!")


if __name__ == "__main__":
    main()
