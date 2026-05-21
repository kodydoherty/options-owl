"""Per-ticker ML sell models + backtest on real Discord signals.

Trains specialized models for SPY, QQQ, IWM (500+ days each),
falls back to generic model for all other tickers.

Then backtests all models on the actual 39 Discord signals we've received.

Usage:
    python scripts/ml_per_ticker.py
"""

import warnings
warnings.filterwarnings("ignore")

import csv
import os
import sys
import sqlite3
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "journal", "historical_0dte.db")
SIGNALS_CSV = os.path.join(PROJECT_DIR, "journal", "all_signals_export.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "journal", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Constants
ENTRY_HOUR_ET = 10
DEADLINE_HOUR_ET = 15
DEADLINE_MINUTE_ET = 30
MIN_OPTION_BARS = 30

SWEET_SPOT_START_MIN = 45
SWEET_SPOT_END_MIN = 150
HARD_DEADLINE_MIN = 240

SELL_THRESHOLD = 0.4

TICKERS_WITH_DATA = ["SPY", "QQQ", "IWM"]  # 500+ days each
ALL_TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "META", "AMD",
               "AMZN", "GOOGL", "MSFT", "IWM", "MU", "MSTR"]

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

# Per-ticker models don't need ticker_encoded
FEATURE_COLS_GENERIC = FEATURE_COLS + ["ticker_encoded"]


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_option_bars(conn, contract_ticker):
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close, volume, vwap, num_trades "
        "FROM option_bars WHERE contract_ticker = ? ORDER BY timestamp",
        (contract_ticker,)
    ).fetchall()
    return [dict(r) for r in rows]


def load_underlying_bars(conn, ticker, date):
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close, volume, vwap "
        "FROM underlying_bars WHERE ticker = ? AND date = ? ORDER BY timestamp",
        (ticker, date)
    ).fetchall()
    return [dict(r) for r in rows]


def find_entry_idx(bars, hour=ENTRY_HOUR_ET, minute=0):
    target = hour * 60 + minute
    for i, b in enumerate(bars):
        et = ts_to_et(b["timestamp"])
        if et.hour * 60 + et.minute >= target:
            return i
    return None


def find_deadline_idx(bars):
    target = DEADLINE_HOUR_ET * 60 + DEADLINE_MINUTE_ET
    last = None
    for i, b in enumerate(bars):
        et = ts_to_et(b["timestamp"])
        bm = et.hour * 60 + et.minute
        if bm < target:
            last = i
        elif bm >= target:
            return i
    return last


def match_underlying(u_bars, ts):
    if not u_bars:
        return None
    best, best_diff = None, float("inf")
    for b in u_bars:
        d = abs(b["timestamp"] - ts)
        if d < best_diff:
            best_diff = d
            best = b["close"]
    return best


def extract_features(option_bars, u_bars, entry_idx, deadline_idx, is_call, ticker_code=None):
    """Extract per-minute features for one trade."""
    if deadline_idx is None or entry_idx is None or deadline_idx <= entry_idx:
        return None

    bars = option_bars[entry_idx:deadline_idx + 1]
    if len(bars) < 2:
        return None

    entry_price = bars[0]["close"]
    if entry_price <= 0:
        entry_price = bars[0]["vwap"] if bars[0]["vwap"] > 0 else 0
    if entry_price <= 0:
        return None

    u_entry = match_underlying(u_bars, bars[0]["timestamp"])

    close_prices = []
    for b in bars:
        c = b["close"] if b["close"] > 0 else b["vwap"]
        close_prices.append(c if c > 0 else np.nan)
    close_arr = np.array(close_prices, dtype=np.float64)
    pnl_arr = (close_arr - entry_price) / entry_price * 100.0

    u_prices = [match_underlying(u_bars, b["timestamp"]) for b in bars]

    rows = []
    peak_price = entry_price
    peak_idx = 0
    consec_down = 0

    for i in range(len(bars)):
        bar = bars[i]
        price = close_prices[i]
        if np.isnan(price):
            continue

        if price > peak_price:
            peak_price = price
            peak_idx = i
            consec_down = 0
        elif i > 0 and not np.isnan(close_prices[i-1]) and price < close_prices[i-1]:
            consec_down += 1
        elif i > 0 and not np.isnan(close_prices[i-1]):
            consec_down = 0

        pnl_pct = pnl_arr[i]
        mfe_pct = (peak_price - entry_price) / entry_price * 100.0
        dd_pct = (peak_price - price) / peak_price * 100.0 if peak_price > 0 else 0.0

        et = ts_to_et(bar["timestamp"])
        mins = i

        def vel(lb):
            if i < lb:
                return 0.0
            p = close_prices[i - lb]
            if np.isnan(p) or p == 0:
                return 0.0
            return (price - p) / p * 100.0 / lb

        v5, v10, v15 = vel(5), vel(10), vel(15)

        accel = 0.0
        if i >= 10:
            p5 = close_prices[i-5]
            p10 = close_prices[i-10]
            if not np.isnan(p5) and not np.isnan(p10) and p10 > 0 and p5 > 0:
                vb = (p5 - p10) / p10 * 100.0 / 5
                accel = v5 - vb

        mfe_retrace = 0.0
        if mfe_pct > 1.0:
            mfe_retrace = min(dd_pct / mfe_pct, 2.0) if mfe_pct > 0 else 0.0

        roll_vol = 0.0
        if i >= 10:
            rets = []
            for j in range(i-9, i+1):
                if j > 0 and not np.isnan(close_prices[j]) and not np.isnan(close_prices[j-1]) and close_prices[j-1] > 0:
                    rets.append((close_prices[j] - close_prices[j-1]) / close_prices[j-1] * 100)
            if rets:
                roll_vol = float(np.std(rets))

        br_pct = (bar["high"] - bar["low"]) / price * 100.0 if price > 0 and bar["high"] > 0 and bar["low"] > 0 else 0.0
        vol = bar["volume"] if bar["volume"] else 0
        vw = [bars[j]["volume"] for j in range(max(0, i-19), i+1) if bars[j]["volume"] and bars[j]["volume"] > 0]
        avg_vol = np.mean(vw) if vw else 1.0
        vol_vs = vol / avg_vol if avg_vol > 0 else 0.0

        up = u_prices[i]
        u_pnl = (up - u_entry) / u_entry * 100.0 if up and u_entry and u_entry > 0 else 0.0
        u_vel = 0.0
        if i >= 5 and u_prices[i-5] and up and u_prices[i-5] > 0:
            u_vel = (up - u_prices[i-5]) / u_prices[i-5] * 100.0 / 5

        rr = mfe_pct / max(dd_pct, 1.0)

        # Labels
        future = pnl_arr[i:]
        valid = future[~np.isnan(future)]
        if len(valid) == 0:
            continue
        fmax = float(np.nanmax(valid))
        fmin = float(np.nanmin(valid))
        upside = fmax - pnl_pct
        regret = fmin - pnl_pct

        should_sell = 0
        if upside < 3.0:
            should_sell = 1
        elif mins > SWEET_SPOT_END_MIN and (upside < 10.0 or regret < -20.0):
            should_sell = 1
        elif mins > HARD_DEADLINE_MIN and upside < 20.0:
            should_sell = 1
        elif pnl_pct > 20.0 and mfe_retrace > 0.5 and upside < pnl_pct * 0.3:
            should_sell = 1
        elif pnl_pct < -30.0 and upside < 15.0:
            should_sell = 1

        exp_future = float(np.nanmean(valid)) - pnl_pct

        row = {
            "pnl_pct": pnl_pct, "mfe_pct": mfe_pct, "drawdown_from_peak_pct": dd_pct,
            "minutes_since_entry": mins, "hour_of_day": et.hour, "minute_of_hour": et.minute,
            "in_sweet_spot": 1 if SWEET_SPOT_START_MIN <= mins <= SWEET_SPOT_END_MIN else 0,
            "past_sweet_spot": 1 if mins > SWEET_SPOT_END_MIN else 0,
            "minutes_past_sweet_spot": max(0, mins - SWEET_SPOT_END_MIN),
            "time_pressure": min(1.0, mins / HARD_DEADLINE_MIN),
            "premium_velocity_5m": v5, "premium_velocity_10m": v10, "premium_velocity_15m": v15,
            "pnl_acceleration": accel, "mfe_retracement_ratio": mfe_retrace,
            "bar_range_pct": br_pct, "volume": vol, "volume_vs_avg": vol_vs,
            "rolling_volatility_10m": roll_vol,
            "underlying_pnl_pct": u_pnl, "underlying_velocity_5m": u_vel,
            "is_call": 1 if is_call else 0, "entry_premium": entry_price,
            "bars_since_new_high": i - peak_idx, "consecutive_down_bars": consec_down,
            "risk_reward_ratio": rr,
            "should_sell": should_sell, "expected_future_pnl": exp_future,
            "_current_pnl": pnl_pct, "_price": price, "_entry_price": entry_price,
            "_timestamp": bar["timestamp"],
        }
        if ticker_code is not None:
            row["ticker_encoded"] = ticker_code
        rows.append(row)

    return rows if rows else None


def build_ticker_dataset(ticker):
    """Build dataset for a single ticker."""
    conn = get_db()
    days = conn.execute(
        "SELECT date, atm_call_ticker, atm_put_ticker FROM trading_days "
        "WHERE ticker = ? AND call_bars >= ? AND put_bars >= ? ORDER BY date",
        (ticker, MIN_OPTION_BARS, MIN_OPTION_BARS)
    ).fetchall()

    all_rows = []
    trades_meta = []

    for day in days:
        u_bars = load_underlying_bars(conn, ticker, day["date"])
        for direction, key in [("call", "atm_call_ticker"), ("put", "atm_put_ticker")]:
            ct = day[key]
            if not ct:
                continue
            o_bars = load_option_bars(conn, ct)
            if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                continue
            ei = find_entry_idx(o_bars)
            di = find_deadline_idx(o_bars)
            if ei is None or di is None:
                continue
            rows = extract_features(o_bars, u_bars, ei, di, direction == "call")
            if not rows or len(rows) < 10:
                continue
            start = len(all_rows)
            all_rows.extend(rows)
            trades_meta.append({"date": day["date"], "ticker": ticker,
                                "direction": direction, "num_rows": len(rows),
                                "start_idx": start, "end_idx": start + len(rows)})

    conn.close()
    if not all_rows:
        return None, None
    return pd.DataFrame(all_rows), pd.DataFrame(trades_meta)


def build_generic_dataset():
    """Build dataset for all tickers (generic fallback model)."""
    ticker_enc = LabelEncoder()
    ticker_enc.fit(ALL_TICKERS)

    conn = get_db()
    all_rows = []
    trades_meta = []

    for ticker in ALL_TICKERS:
        days = conn.execute(
            "SELECT date, atm_call_ticker, atm_put_ticker FROM trading_days "
            "WHERE ticker = ? AND call_bars >= ? AND put_bars >= ? ORDER BY date",
            (ticker, MIN_OPTION_BARS, MIN_OPTION_BARS)
        ).fetchall()

        tc = int(ticker_enc.transform([ticker])[0])
        for day in days:
            u_bars = load_underlying_bars(conn, ticker, day["date"])
            for direction, key in [("call", "atm_call_ticker"), ("put", "atm_put_ticker")]:
                ct = day[key]
                if not ct:
                    continue
                o_bars = load_option_bars(conn, ct)
                if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                    continue
                ei = find_entry_idx(o_bars)
                di = find_deadline_idx(o_bars)
                if ei is None or di is None:
                    continue
                rows = extract_features(o_bars, u_bars, ei, di, direction == "call", ticker_code=tc)
                if not rows or len(rows) < 10:
                    continue
                start = len(all_rows)
                all_rows.extend(rows)
                trades_meta.append({"date": day["date"], "ticker": ticker,
                                    "direction": direction, "num_rows": len(rows),
                                    "start_idx": start, "end_idx": start + len(rows)})

    conn.close()
    if not all_rows:
        return None, None
    return pd.DataFrame(all_rows), pd.DataFrame(trades_meta)


def train_model(df, trades_df, feature_cols, model_name):
    """Train a classifier + regressor. Returns (clf, reg)."""
    dates = sorted(trades_df["date"].unique())
    n = len(dates)
    train_end = int(n * 0.75)
    val_end = int(n * 0.90)

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])

    row_dates = []
    for _, t in trades_df.iterrows():
        row_dates.extend([t["date"]] * t["num_rows"])
    row_dates = np.array(row_dates)

    train_mask = np.isin(row_dates, list(train_dates))
    val_mask = np.isin(row_dates, list(val_dates))
    test_mask = np.isin(row_dates, list(test_dates))

    X = df[feature_cols]
    y = df["should_sell"]

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    if len(X_tr) < 100 or len(X_val) < 50:
        print(f"    Insufficient data for {model_name}, skipping")
        return None, None, None, None

    print(f"    {model_name}: train={len(X_tr):,}, val={len(X_val):,}, test={len(X_te):,}")
    print(f"    Dates: train {dates[0]}→{dates[train_end-1]}, "
          f"val {dates[train_end]}→{dates[val_end-1]}, "
          f"test {dates[val_end]}→{dates[-1]}")

    # Classifier
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    clf_params = {
        "objective": "binary", "metric": "binary_logloss",
        "num_leaves": 63, "learning_rate": 0.02, "max_depth": 7,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 3,
        "min_child_samples": 50, "lambda_l1": 0.1, "lambda_l2": 1.0,
        "scale_pos_weight": len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1),
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }

    clf = lgb.train(clf_params, dtrain, num_boost_round=2000,
                    valid_sets=[dval], valid_names=["val"],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    # Regressor
    y_reg_tr = df.loc[train_mask, "expected_future_pnl"]
    y_reg_val = df.loc[val_mask, "expected_future_pnl"]
    dtrain_r = lgb.Dataset(X_tr, label=y_reg_tr)
    dval_r = lgb.Dataset(X_val, label=y_reg_val, reference=dtrain_r)

    reg_params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 63, "learning_rate": 0.02, "max_depth": 7,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 3,
        "min_child_samples": 50, "lambda_l1": 0.1, "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }

    reg = lgb.train(reg_params, dtrain_r, num_boost_round=2000,
                    valid_sets=[dval_r], valid_names=["val"],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    # Test set eval
    te_proba = clf.predict(X_te, num_iteration=clf.best_iteration)
    te_pred = (te_proba > SELL_THRESHOLD).astype(int)

    if len(y_te) > 0:
        wr = sum(y_te == te_pred) / len(y_te) * 100
        sell_recall = sum((te_pred == 1) & (y_te == 1)) / max(sum(y_te == 1), 1) * 100
        sell_prec = sum((te_pred == 1) & (y_te == 1)) / max(sum(te_pred == 1), 1) * 100
        print(f"    Test: accuracy={wr:.0f}%, sell_precision={sell_prec:.0f}%, sell_recall={sell_recall:.0f}%")
        print(f"    Classifier iters: {clf.best_iteration}, Regressor iters: {reg.best_iteration}")

    # Save
    clf_path = os.path.join(MODELS_DIR, f"{model_name}_clf.lgb")
    reg_path = os.path.join(MODELS_DIR, f"{model_name}_reg.lgb")
    clf.save_model(clf_path)
    reg.save_model(reg_path)
    print(f"    Saved: {clf_path}")

    # Feature importance top 5
    imp = sorted(zip(feature_cols, clf.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])[:5]
    print(f"    Top features: {', '.join(f'{n}' for n, _ in imp)}")

    return clf, reg, test_mask, te_proba


def simulate_ml_exit(pnls, prices, entry_price, clf_proba, reg_preds, threshold=SELL_THRESHOLD):
    """Simulate ML combo exit: returns (exit_pnl, exit_minute, exit_reason)."""
    for i in range(len(pnls)):
        prob = clf_proba[i] if i < len(clf_proba) else 0
        reg_val = reg_preds[i] if i < len(reg_preds) else 10

        # Combo logic
        if prob > threshold and reg_val < 2.0:
            return pnls[i], i, "ml_combo"
        if i >= 10 and reg_val < -10.0:
            return pnls[i], i, "ml_reg_bearish"

    return pnls[-1], len(pnls) - 1, "eod"


def backtest_on_signals(models, generic_clf, generic_reg):
    """Backtest all models on our actual Discord signals."""
    print("\n" + "=" * 100)
    print("  BACKTEST ON REAL DISCORD SIGNALS (last 2-3 weeks)")
    print("=" * 100)

    # Load signals
    if not os.path.exists(SIGNALS_CSV):
        print(f"  No signals CSV at {SIGNALS_CSV}")
        return

    with open(SIGNALS_CSV) as f:
        signals = list(csv.DictReader(f))

    print(f"  Loaded {len(signals)} signals from {SIGNALS_CSV}")

    conn = get_db()
    results = []

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"].lower()
        is_call = direction in ("call", "c")
        entry_price_str = sig.get("entry_price", "0")
        try:
            entry_underlying = float(entry_price_str)
        except (ValueError, TypeError):
            continue

        sig_date = sig["created_at"][:10]
        score = int(sig.get("score", 0))
        target_1 = sig.get("target_1", "")
        target_2 = sig.get("target_2", "")
        stop_price = sig.get("stop_price", "")

        # Try to find matching option data in our historical DB
        # Look for this ticker on this date
        day = conn.execute(
            "SELECT atm_call_ticker, atm_put_ticker, atm_strike FROM trading_days "
            "WHERE ticker = ? AND date = ?",
            (ticker, sig_date)
        ).fetchone()

        if not day:
            # No historical data for this ticker/date
            results.append({
                "date": sig_date, "ticker": ticker, "direction": direction,
                "score": score, "entry_price": entry_underlying,
                "has_data": False, "ml_pnl": None, "eod_pnl": None,
                "exit_90m_pnl": None, "vinny_pnl": None, "ml_exit_min": None,
            })
            continue

        ct_key = "atm_call_ticker" if is_call else "atm_put_ticker"
        ct = day[ct_key]
        if not ct:
            continue

        o_bars = load_option_bars(conn, ct)
        if not o_bars or len(o_bars) < 30:
            continue

        u_bars = load_underlying_bars(conn, ticker, sig_date)
        ei = find_entry_idx(o_bars)
        di = find_deadline_idx(o_bars)
        if ei is None or di is None:
            continue

        # Extract features
        feat_cols = FEATURE_COLS
        ticker_code = None
        if ticker not in TICKERS_WITH_DATA:
            feat_cols = FEATURE_COLS_GENERIC
            te = LabelEncoder()
            te.fit(ALL_TICKERS)
            if ticker in ALL_TICKERS:
                ticker_code = int(te.transform([ticker])[0])
            else:
                ticker_code = 0

        rows = extract_features(o_bars, u_bars, ei, di, is_call, ticker_code=ticker_code)
        if not rows or len(rows) < 10:
            continue

        row_df = pd.DataFrame(rows)
        pnls = row_df["_current_pnl"].values
        prices = row_df["_price"].values
        ep = row_df.iloc[0]["_entry_price"]

        # Choose model
        if ticker in models and models[ticker][0] is not None:
            clf, reg = models[ticker]
            X = row_df[FEATURE_COLS]
        else:
            clf, reg = generic_clf, generic_reg
            X = row_df[FEATURE_COLS_GENERIC]

        clf_proba = clf.predict(X, num_iteration=clf.best_iteration)
        reg_preds = reg.predict(X, num_iteration=reg.best_iteration)

        # ML combo exit
        ml_pnl, ml_min, ml_reason = simulate_ml_exit(pnls, prices, ep, clf_proba, reg_preds)

        # EOD exit
        eod_pnl = pnls[-1]

        # 90 min exit
        exit_90 = pnls[90] if len(pnls) > 90 else eod_pnl

        # Vinny trail
        vinny_pnl = eod_pnl
        peak = ep
        phase_trail = 25.0
        for i in range(len(prices)):
            p = prices[i]
            if p > peak:
                peak = p
                gain = (peak - ep) / ep * 100
                if gain >= 200: phase_trail = 10.0
                elif gain >= 100: phase_trail = 15.0
                elif gain >= 50: phase_trail = 20.0
            if i >= 5 and peak > 0:
                drop = (peak - p) / peak * 100
                if drop >= phase_trail:
                    vinny_pnl = pnls[i]
                    break
            if i >= 150 and peak > 0:
                drop = (peak - p) / peak * 100
                if drop >= 10.0:
                    vinny_pnl = pnls[i]
                    break

        # MFE
        mfe = max(pnls) if len(pnls) > 0 else 0

        results.append({
            "date": sig_date, "ticker": ticker, "direction": direction,
            "score": score, "entry_price": entry_underlying,
            "has_data": True, "entry_premium": ep,
            "ml_pnl": round(ml_pnl, 1), "ml_exit_min": ml_min,
            "ml_reason": ml_reason,
            "eod_pnl": round(eod_pnl, 1),
            "exit_90m_pnl": round(exit_90, 1),
            "vinny_pnl": round(vinny_pnl, 1),
            "mfe_pct": round(mfe, 1),
            "model_used": "per_ticker" if ticker in models and models[ticker][0] else "generic",
        })

    conn.close()

    # Print results
    with_data = [r for r in results if r["has_data"]]
    no_data = [r for r in results if not r["has_data"]]

    print(f"\n  Signals with historical data: {len(with_data)}")
    print(f"  Signals without data (no backtest possible): {len(no_data)}")
    if no_data:
        missing = [r["ticker"] + " (" + r["date"] + ")" for r in no_data]
        print(f"    Missing: {', '.join(missing)}")

    if not with_data:
        print("  No backtestable signals!")
        return

    print(f"\n  {'Date':>12} {'Ticker':>6} {'Dir':>4} {'Scr':>4} {'Model':>10} "
          f"{'ML%':>8} {'@Min':>5} {'EOD%':>8} {'90m%':>8} {'Vinny%':>8} {'MFE%':>8} {'ML Reason':>14}")
    print("  " + "-" * 110)

    for r in sorted(with_data, key=lambda x: x["date"]):
        print(f"  {r['date']:>12} {r['ticker']:>6} {r['direction']:>4} {r['score']:>4} "
              f"{r['model_used']:>10} "
              f"{r['ml_pnl']:>+7.1f}% {r['ml_exit_min']:>4}m "
              f"{r['eod_pnl']:>+7.1f}% {r['exit_90m_pnl']:>+7.1f}% "
              f"{r['vinny_pnl']:>+7.1f}% {r['mfe_pct']:>+7.1f}% "
              f"{r['ml_reason']:>14}")

    # Totals
    print("  " + "-" * 110)
    for strategy, key in [("ML Combo", "ml_pnl"), ("Hold EOD", "eod_pnl"),
                          ("Exit 90m", "exit_90m_pnl"), ("Vinny Trail", "vinny_pnl")]:
        vals = [r[key] for r in with_data if r[key] is not None]
        if vals:
            total = sum(vals)
            avg = np.mean(vals)
            wins = sum(1 for v in vals if v > 0)
            wr = wins / len(vals) * 100
            print(f"  {strategy:>20}: total={total:>+8.1f}%  avg={avg:>+7.1f}%  "
                  f"win_rate={wr:.0f}% ({wins}/{len(vals)})")

    # Save CSV
    csv_path = os.path.join(PROJECT_DIR, "journal", "ml_signal_backtest.csv")
    if with_data:
        keys = with_data[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(with_data)
        print(f"\n  Results saved to {csv_path}")


def main():
    start = time.time()
    print()
    print("=" * 100)
    print("  OptionsOwl — Per-Ticker ML Sell Models + Signal Backtest")
    print("=" * 100)

    # ── Step 1: Train per-ticker models ──────────────────────────────
    print("\n  STEP 1: Train per-ticker models (SPY, QQQ, IWM)")
    print("  " + "-" * 60)

    models = {}  # ticker -> (clf, reg)

    for ticker in TICKERS_WITH_DATA:
        print(f"\n  Training {ticker}...")
        df, trades_df = build_ticker_dataset(ticker)
        if df is None:
            print(f"    No data for {ticker}")
            continue
        print(f"    Dataset: {len(df):,} rows, {len(trades_df)} trades")
        clf, reg, _, _ = train_model(df, trades_df, FEATURE_COLS, ticker.lower())
        models[ticker] = (clf, reg)

    # ── Step 2: Train generic fallback model ─────────────────────────
    print(f"\n\n  STEP 2: Train generic fallback model (all {len(ALL_TICKERS)} tickers)")
    print("  " + "-" * 60)

    gdf, gtrades = build_generic_dataset()
    print(f"    Dataset: {len(gdf):,} rows, {len(gtrades)} trades")
    generic_clf, generic_reg, _, _ = train_model(gdf, gtrades, FEATURE_COLS_GENERIC, "generic")

    # ── Step 3: Backtest on Discord signals ──────────────────────────
    backtest_on_signals(models, generic_clf, generic_reg)

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("  Done!")


if __name__ == "__main__":
    main()
