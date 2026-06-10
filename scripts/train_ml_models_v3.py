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
# MODEL 3: Regime Classification — trending vs chop day
# ===========================================================================

def train_regime_model():
    """Train regime classifier using daily stock OHLC + UW GEX data.

    Label: 1 = trending day (intraday range > 1.5% AND close near high/low),
           0 = chop day (range < 1% OR close near middle of range)
    """
    print("\n" + "=" * 70)
    print("MODEL 3: Regime Classification (trending vs chop)")
    print("=" * 70)

    theta_conn = _connect_theta()
    uw_conn = _connect_uw()

    rows = []
    for ticker in TICKERS:
        # Get daily OHLC from stock data (aggregate 1-min into daily)
        daily = pd.read_sql_query(
            """SELECT substr(timestamp, 1, 10) as date,
                      MIN(open) as day_open,
                      MAX(high) as day_high,
                      MIN(low) as day_low,
                      SUM(volume) as day_volume
               FROM stock_ohlc WHERE ticker=?
               GROUP BY substr(timestamp, 1, 10)
               ORDER BY date""",
            theta_conn, params=(ticker,),
        )
        if len(daily) < 20:
            continue

        # Get first and last close per day
        for _, day_row in daily.iterrows():
            dt = day_row["date"]
            day_open = day_row["day_open"]
            day_high = day_row["day_high"]
            day_low = day_row["day_low"]
            day_volume = day_row["day_volume"] or 0

            if not day_high or not day_low or day_high <= 0 or day_low <= 0:
                continue

            # Get close (last bar of day)
            close_row = theta_conn.execute(
                "SELECT close FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1",
                (ticker, f"{dt}%"),
            ).fetchone()
            if not close_row:
                continue
            day_close = close_row[0]
            if not day_close or day_close <= 0:
                continue

            # Get open (first bar)
            open_row = theta_conn.execute(
                "SELECT open FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp ASC LIMIT 1",
                (ticker, f"{dt}%"),
            ).fetchone()
            if open_row and open_row[0] and open_row[0] > 0:
                day_open = open_row[0]
            _ = day_open  # kept for clarity; label uses range/close position only

            day_range_pct = (day_high - day_low) / day_low * 100
            close_position = (day_close - day_low) / (day_high - day_low) if day_high > day_low else 0.5

            # Label: trending if range > 1.5% AND close is near extreme (< 0.2 or > 0.8)
            is_trending = day_range_pct > 1.5 and (close_position < 0.2 or close_position > 0.8)

            f = {
                "ticker_idx": TICKERS.index(ticker),
                "day_range_pct": day_range_pct,
            }

            # GEX data for this ticker/date
            gex_row = uw_conn.execute(
                "SELECT call_gamma, put_gamma, call_delta, put_delta FROM greek_exposure WHERE ticker=? AND date=?",
                (ticker, dt),
            ).fetchone()
            if gex_row:
                f["call_gamma"] = float(gex_row[0] or 0)
                f["put_gamma"] = float(gex_row[1] or 0)
                f["net_gamma"] = f["call_gamma"] - f["put_gamma"]
                f["call_delta"] = float(gex_row[2] or 0)
                f["put_delta"] = float(gex_row[3] or 0)
                f["net_delta"] = f["call_delta"] - f["put_delta"]
            else:
                for k in ["call_gamma", "put_gamma", "net_gamma", "call_delta", "put_delta", "net_delta"]:
                    f[k] = 0

            # Options volume data
            vol_row = uw_conn.execute(
                "SELECT call_volume, put_volume, net_call_premium, net_put_premium, "
                "put_open_interest, call_open_interest FROM options_volume WHERE ticker=? AND date=?",
                (ticker, dt),
            ).fetchone()
            if vol_row:
                call_vol = vol_row[0] or 0
                put_vol = vol_row[1] or 0
                f["put_call_ratio"] = float(put_vol / max(call_vol, 1))
                f["net_call_premium"] = float(vol_row[2] or 0)
                f["net_put_premium"] = float(vol_row[3] or 0)
                f["put_oi"] = float(vol_row[4] or 0)
                f["call_oi"] = float(vol_row[5] or 0)
            else:
                for k in ["put_call_ratio", "net_call_premium", "net_put_premium", "put_oi", "call_oi"]:
                    f[k] = 0

            # Previous day's action (lag features)
            prev_rows = daily[daily["date"] < dt].tail(5)
            if len(prev_rows) > 0:
                prev = prev_rows.iloc[-1]
                f["prev_range_pct"] = float((prev["day_high"] - prev["day_low"]) / prev["day_low"] * 100) if prev["day_low"] > 0 else 0
                f["prev_volume"] = float(prev["day_volume"] or 0)
                if len(prev_rows) >= 3:
                    recent_ranges = [(r["day_high"] - r["day_low"]) / r["day_low"] * 100
                                     for _, r in prev_rows.tail(3).iterrows() if r["day_low"] > 0]
                    f["avg_3d_range"] = float(np.mean(recent_ranges)) if recent_ranges else 0
                else:
                    f["avg_3d_range"] = 0
            else:
                f["prev_range_pct"] = 0
                f["prev_volume"] = 0
                f["avg_3d_range"] = 0

            f["day_volume"] = float(day_volume)
            f["day_of_week"] = datetime.strptime(dt, "%Y-%m-%d").weekday()

            f["label"] = 1 if is_trending else 0
            f["ticker"] = ticker
            f["date"] = dt
            rows.append(f)

    theta_conn.close()
    uw_conn.close()

    if not rows:
        print("  No regime data collected!")
        return

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} day-ticker samples ({df['label'].sum()} trending, {(~df['label'].astype(bool)).sum()} chop)")

    meta_cols = ["ticker", "date"]
    # TARGET LEAKAGE FIX: the label is derived from day_range_pct (trending =
    # day_range_pct > 1.5 AND close near extreme), and day_volume is full-day
    # information not knowable at prediction time (the model is served at 9:45
    # AM). Including them lets the model read the answer off the features.
    leak_cols = ["day_range_pct", "day_volume"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"] + leak_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    }

    # Date-based expanding-window walk-forward (never split by row)
    _walk_forward_validate(df, X, y, feature_cols, params, is_regression=False,
                           num_boost_round=300)

    train_mask, test_mask = _final_date_split(df)
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    print(f"\n  Final split: train={len(X_train)} test={len(X_test)} (by date)")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=300,
                       valid_sets=[dtest], callbacks=[lgb.log_evaluation(50)])

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)

    print(f"\n  Regime Model: AUC={auc:.3f} Acc={acc:.3f} Prec={prec:.3f} Recall={rec:.3f}")

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    # Save
    model_path = str(MODEL_DIR / "regime_classifier.txt")
    model.save_model(model_path)
    meta = {"features": feature_cols, "auc": auc, "accuracy": acc,
            "precision": prec, "recall": rec, "n_train": len(X_train), "n_test": len(X_test),
            "split": "date_walk_forward", "dropped_leak_features": leak_cols}
    with open(str(MODEL_DIR / "regime_classifier_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {model_path}")


# ===========================================================================
# MODEL 4: Ticker Selection — which tickers to focus on today
# ===========================================================================

def train_ticker_selection_model():
    """Train model to predict which tickers will be profitable today.

    For each ticker-day: features are pre-market/early data, label is whether
    the ticker had net positive P&L that day (from FSM backtests).
    """
    print("\n" + "=" * 70)
    print("MODEL 4: Ticker Selection (which tickers to trade today)")
    print("=" * 70)

    theta_conn = _connect_theta()
    uw_conn = _connect_uw()

    # First, compute daily P&L per ticker from option data
    # Use ATM option intraday to compute "would this have been profitable?"
    rows = []
    for ticker in TICKERS:
        print(f"  Processing {ticker}...", flush=True)

        dates = [r[0] for r in theta_conn.execute(
            "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
            (ticker,),
        ).fetchall()]

        for dt in dates:
            # Quick daily P&L estimate: ATM call from first bar to best exit
            strike = find_atm_strike(theta_conn, ticker, dt, "CALL")
            if strike is None:
                continue

            day_ohlc = pd.read_sql_query(
                "SELECT close FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? AND right='CALL' AND strike=? ORDER BY timestamp",
                theta_conn, params=(ticker, f"{dt}%", strike),
            )
            if len(day_ohlc) < 20:
                continue

            closes = day_ohlc["close"].dropna().values
            if len(closes) < 20 or closes[0] <= 0:
                continue

            # Simple metric: max gain achievable from any 9:30-10:30 entry
            # (killzone entries)
            early_entries = closes[:min(60, len(closes))]
            best_entry = np.min(early_entries)
            best_exit = np.max(closes)
            if best_entry <= 0:
                continue
            max_pnl_pct = (best_exit / best_entry - 1) * 100

            # Label: profitable if best achievable gain > 30%
            is_profitable = max_pnl_pct > 30

            # Features (pre-market / early morning)
            f = {"ticker_idx": TICKERS.index(ticker)}

            # GEX
            gex = uw_conn.execute(
                "SELECT call_gamma, put_gamma, call_delta, put_delta FROM greek_exposure WHERE ticker=? AND date=?",
                (ticker, dt),
            ).fetchone()
            if gex:
                f["call_gamma"] = float(gex[0] or 0)
                f["put_gamma"] = float(gex[1] or 0)
                f["net_gamma"] = f["call_gamma"] - f["put_gamma"]
                f["net_delta"] = float((gex[2] or 0) - (gex[3] or 0))
            else:
                f["call_gamma"] = 0
                f["put_gamma"] = 0
                f["net_gamma"] = 0
                f["net_delta"] = 0

            # Options volume
            vol = uw_conn.execute(
                "SELECT call_volume, put_volume, net_call_premium, net_put_premium FROM options_volume WHERE ticker=? AND date=?",
                (ticker, dt),
            ).fetchone()
            if vol:
                f["put_call_ratio"] = float((vol[1] or 0) / max(vol[0] or 1, 1))
                f["net_premium_flow"] = float((vol[2] or 0) - (vol[3] or 0))
            else:
                f["put_call_ratio"] = 0
                f["net_premium_flow"] = 0

            # Stock data: previous day and early today
            stock = pd.read_sql_query(
                "SELECT open, high, low, close, volume FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp LIMIT 30",
                theta_conn, params=(ticker, f"{dt}%"),
            )
            if len(stock) > 5:
                s = stock["close"].dropna().values
                if len(s) > 5 and s[0] > 0:
                    f["early_momentum"] = float((s[min(5, len(s)-1)] / s[0] - 1) * 100)
                    f["early_volatility"] = float(np.std(np.diff(s[:min(10, len(s))]) / s[:min(9, len(s)-1)]) * 100) if all(s[:min(9, len(s)-1)] > 0) else 0
                else:
                    f["early_momentum"] = 0
                    f["early_volatility"] = 0
                if "volume" in stock.columns:
                    f["early_volume"] = float(stock["volume"].iloc[:5].sum() or 0)
                else:
                    f["early_volume"] = 0
            else:
                f["early_momentum"] = 0
                f["early_volatility"] = 0
                f["early_volume"] = 0

            # Opening premium and IV
            greeks_first = theta_conn.execute(
                "SELECT implied_vol, delta FROM option_greeks "
                "WHERE ticker=? AND timestamp LIKE ? AND right='CALL' AND strike=? "
                "ORDER BY timestamp LIMIT 1",
                (ticker, f"{dt}%", strike),
            ).fetchone()
            if greeks_first:
                f["opening_iv"] = float(greeks_first[0] or 0)
                f["opening_delta"] = float(abs(greeks_first[1] or 0))
            else:
                f["opening_iv"] = 0
                f["opening_delta"] = 0

            f["opening_premium"] = float(closes[0])
            f["day_of_week"] = datetime.strptime(dt, "%Y-%m-%d").weekday()

            f["label"] = 1 if is_profitable else 0
            f["ticker"] = ticker
            f["date"] = dt
            f["max_pnl_pct"] = max_pnl_pct
            rows.append(f)

    theta_conn.close()
    uw_conn.close()

    if not rows:
        print("  No ticker selection data!")
        return

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} ticker-day samples ({df['label'].sum()} profitable, {(~df['label'].astype(bool)).sum()} unprofitable)")

    meta_cols = ["ticker", "date", "max_pnl_pct"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    }

    # Date-based expanding-window walk-forward (never split by row)
    _walk_forward_validate(df, X, y, feature_cols, params, is_regression=False,
                           num_boost_round=300)

    train_mask, test_mask = _final_date_split(df)
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    print(f"\n  Final split: train={len(X_train)} test={len(X_test)} (by date)")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=300,
                       valid_sets=[dtest], callbacks=[lgb.log_evaluation(50)])

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)

    print(f"\n  Ticker Selection: AUC={auc:.3f} Acc={acc:.3f}")

    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    model_path = str(MODEL_DIR / "ticker_selection.txt")
    model.save_model(model_path)
    meta = {"features": feature_cols, "auc": auc, "accuracy": acc,
            "n_train": len(X_train), "n_test": len(X_test)}
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

def _worker_signal_quality(item):
    """Predict the peak gain % achievable from each entry point.

    Label: peak_gain_pct (regression — continuous value, not binary)
    Features: same pre-entry features as entry model
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

        # Peak gain in forward window
        future = closes[entry_idx:min(entry_idx + MOVE_WINDOW_MIN, n)]
        if len(future) < 5:
            continue
        peak = np.nanmax(future)
        if peak <= 0:
            continue
        peak_gain_pct = (peak / entry_price - 1) * 100

        # Min gain (worst drawdown)
        trough = np.nanmin(future)
        min_gain_pct = (trough / entry_price - 1) * 100 if trough > 0 else -100

        features = compute_pre_entry_features(ohlc, quotes, greeks, stock, entry_idx)
        if not features:
            continue

        features["is_call"] = 1 if item["right"] == "CALL" else 0
        features["peak_gain_pct"] = peak_gain_pct
        features["max_drawdown_pct"] = min_gain_pct
        features["ticker"] = ticker
        features["date"] = dt
        rows.append(features)

    return rows


# ===========================================================================
# Training orchestration
# ===========================================================================

def train_model_with_pool(model_name, worker_fn, tickers, label_col="label",
                          is_regression=False, extra_meta_cols=None,
                          embargo_days=0):
    """Generic training pipeline: preload data → parallel workers → train LightGBM.

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

    if model in ("all", "stop_calibrate"):
        train_stop_calibration_model(tickers)

    if model in ("all", "signal_quality"):
        train_model_with_pool(
            "signal_quality", _worker_signal_quality, tickers,
            label_col="peak_gain_pct", is_regression=True,
            extra_meta_cols=["max_drawdown_pct"], embargo_days=1,
        )

    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
