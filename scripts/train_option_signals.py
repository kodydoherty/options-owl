"""Train per-ticker ML models to detect profitable 0DTE entry patterns from option price data.

Instead of relying on Discord analysts or rule-based scoring, this learns directly from
historical option prices what patterns precede profitable trades.

Features extracted from ThetaData (1-min resolution):
  - IV dynamics: current IV, IV rank (percentile), IV acceleration, IV vs HV spread
  - Volume/flow: option volume surge, put/call volume ratio, volume vs OI
  - Bid/ask microstructure: spread width, spread tightening rate, bid size imbalance
  - Greeks dynamics: delta, gamma exposure, theta decay rate
  - Price action: underlying momentum, VWAP deviation, recent range position
  - Time features: minutes since open, time-of-day buckets

Labels: simulate entry → run V5 FSM → P&L outcome (win/loss/magnitude)

Usage:
    python scripts/train_option_signals.py                    # train all tickers
    python scripts/train_option_signals.py --ticker SPY       # single ticker
    python scripts/train_option_signals.py --evaluate         # evaluate only (no retrain)
    python scripts/train_option_signals.py --backtest         # run backtest with trained models
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "signal_ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# V5 FSM simulation thresholds (simplified for labeling)
WIN_THRESHOLD_PCT = 15.0    # +15% premium gain = win
LOSS_THRESHOLD_PCT = -35.0  # -35% premium loss = loss
MAX_HOLD_MINUTES = 120      # max hold time for simulation


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def load_day_data(conn: sqlite3.Connection, ticker: str, trade_date: str) -> dict:
    """Load all option + stock data for a ticker on a single day."""
    ohlc = pd.read_sql_query(
        "SELECT * FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
        conn, params=(ticker, f"{trade_date}%"),
    )
    quotes = pd.read_sql_query(
        "SELECT * FROM option_quotes WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
        conn, params=(ticker, f"{trade_date}%"),
    )
    greeks = pd.read_sql_query(
        "SELECT * FROM option_greeks WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
        conn, params=(ticker, f"{trade_date}%"),
    )
    stock = pd.read_sql_query(
        "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
        conn, params=(ticker, f"{trade_date}%"),
    )
    return {"ohlc": ohlc, "quotes": quotes, "greeks": greeks, "stock": stock}


def compute_features_at_timestamp(
    data: dict,
    right: str,  # "call" or "put"
    timestamp_idx: int,
    lookback: int = 15,  # minutes of lookback for rolling features
) -> dict | None:
    """Compute ML features at a specific timestamp for a specific option side.

    Returns None if insufficient data.
    """
    ohlc = data["ohlc"]
    quotes = data["quotes"]
    greeks = data["greeks"]
    stock = data["stock"]

    # Filter to the right side (call or put) — DB may store as CALL/PUT or call/put
    right_upper = right.upper()
    ohlc_side = ohlc[ohlc["right"].str.upper() == right_upper].reset_index(drop=True)
    quotes_side = quotes[quotes["right"].str.upper() == right_upper].reset_index(drop=True)
    greeks_side = greeks[greeks["right"].str.upper() == right_upper].reset_index(drop=True)

    if len(ohlc_side) < lookback + 5 or timestamp_idx < lookback:
        return None
    if timestamp_idx >= len(ohlc_side):
        return None

    # Current row and lookback window
    curr = ohlc_side.iloc[timestamp_idx]
    window = ohlc_side.iloc[max(0, timestamp_idx - lookback):timestamp_idx + 1]

    features = {}

    # --- Time features ---
    try:
        ts = pd.Timestamp(curr["timestamp"])
        minutes_since_open = max(0, (ts.hour - 9) * 60 + ts.minute - 30) if ts.tzinfo is None else \
            max(0, (ts.tz_convert("America/New_York").hour - 9) * 60 + ts.tz_convert("America/New_York").minute - 30)
        features["minutes_since_open"] = minutes_since_open
        features["time_bucket"] = min(minutes_since_open // 30, 12)  # 30-min buckets, cap at 12
    except Exception:
        features["minutes_since_open"] = 0
        features["time_bucket"] = 0

    # --- Option price features ---
    current_price = curr.get("close", 0) or 0
    features["option_price"] = current_price

    if len(window) > 1 and window["close"].notna().sum() > 1:
        prices = window["close"].dropna().values
        features["price_momentum_5m"] = (prices[-1] / prices[max(0, -6)] - 1) * 100 if len(prices) > 5 and prices[max(0, -6)] > 0 else 0
        features["price_momentum_15m"] = (prices[-1] / prices[0] - 1) * 100 if prices[0] > 0 else 0
        features["price_volatility"] = np.std(np.diff(prices) / prices[:-1]) * 100 if len(prices) > 2 and all(prices[:-1] > 0) else 0
        features["price_range_position"] = (prices[-1] - prices.min()) / (prices.max() - prices.min()) if prices.max() > prices.min() else 0.5
    else:
        features["price_momentum_5m"] = 0
        features["price_momentum_15m"] = 0
        features["price_volatility"] = 0
        features["price_range_position"] = 0.5

    # --- Volume features ---
    if "volume" in window.columns and window["volume"].notna().sum() > 0:
        vols = window["volume"].fillna(0).values
        features["current_volume"] = float(vols[-1])
        features["avg_volume"] = float(np.mean(vols)) if len(vols) > 0 else 0
        features["volume_surge"] = float(vols[-1] / max(np.mean(vols[:-1]), 1)) if len(vols) > 1 else 1.0
        features["volume_acceleration"] = float(np.mean(vols[-3:]) / max(np.mean(vols[:-3]), 1)) if len(vols) > 5 else 1.0
    else:
        features["current_volume"] = 0
        features["avg_volume"] = 0
        features["volume_surge"] = 1.0
        features["volume_acceleration"] = 1.0

    # --- Bid/ask spread features ---
    if len(quotes_side) > timestamp_idx:
        q = quotes_side.iloc[timestamp_idx]
        q_window = quotes_side.iloc[max(0, timestamp_idx - lookback):timestamp_idx + 1]
        bid = q.get("bid", 0) or 0
        ask = q.get("ask", 0) or 0
        mid = (bid + ask) / 2 if (bid + ask) > 0 else current_price

        features["bid_ask_spread"] = (ask - bid) if ask > bid else 0
        features["spread_pct"] = ((ask - bid) / mid * 100) if mid > 0 else 0
        features["bid_size"] = q.get("bid_size", 0) or 0
        features["ask_size"] = q.get("ask_size", 0) or 0
        features["size_imbalance"] = (features["bid_size"] - features["ask_size"]) / max(features["bid_size"] + features["ask_size"], 1)

        # Spread tightening (positive = tightening = institutional interest)
        if len(q_window) > 3:
            spreads = (q_window["ask"].fillna(0) - q_window["bid"].fillna(0)).values
            spreads = spreads[spreads >= 0]
            if len(spreads) > 3:
                features["spread_trend"] = float(spreads[:len(spreads)//2].mean() - spreads[len(spreads)//2:].mean())
            else:
                features["spread_trend"] = 0
        else:
            features["spread_trend"] = 0
    else:
        features["bid_ask_spread"] = 0
        features["spread_pct"] = 0
        features["bid_size"] = 0
        features["ask_size"] = 0
        features["size_imbalance"] = 0
        features["spread_trend"] = 0

    # --- Greeks features ---
    if len(greeks_side) > timestamp_idx:
        g = greeks_side.iloc[timestamp_idx]
        g_window = greeks_side.iloc[max(0, timestamp_idx - lookback):timestamp_idx + 1]

        features["iv"] = g.get("implied_vol", 0) or 0
        features["delta"] = abs(g.get("delta", 0) or 0)
        features["theta"] = g.get("theta", 0) or 0
        features["vega"] = g.get("vega", 0) or 0

        # IV dynamics
        if len(g_window) > 3 and g_window["implied_vol"].notna().sum() > 3:
            ivs = g_window["implied_vol"].dropna().values
            features["iv_momentum"] = float(ivs[-1] - ivs[0]) if len(ivs) > 1 else 0
            features["iv_percentile"] = float(np.searchsorted(np.sort(ivs), ivs[-1]) / len(ivs))
            features["iv_acceleration"] = float(np.mean(ivs[-3:]) - np.mean(ivs[:-3])) if len(ivs) > 5 else 0
        else:
            features["iv_momentum"] = 0
            features["iv_percentile"] = 0.5
            features["iv_acceleration"] = 0

        # Underlying price from greeks
        features["underlying_price"] = g.get("underlying_price", 0) or 0
    else:
        features["iv"] = 0
        features["delta"] = 0.5
        features["theta"] = 0
        features["vega"] = 0
        features["iv_momentum"] = 0
        features["iv_percentile"] = 0.5
        features["iv_acceleration"] = 0
        features["underlying_price"] = 0

    # --- Underlying price features (from stock_ohlc) ---
    if len(stock) > timestamp_idx:
        s_window = stock.iloc[max(0, timestamp_idx - lookback):timestamp_idx + 1]
        if len(s_window) > 1:
            closes = s_window["close"].dropna().values
            if len(closes) > 1:
                features["underlying_momentum_5m"] = float((closes[-1] / closes[max(0, -6)] - 1) * 100) if closes[max(0, -6)] > 0 else 0
                features["underlying_momentum_15m"] = float((closes[-1] / closes[0] - 1) * 100) if closes[0] > 0 else 0
                features["underlying_volatility"] = float(np.std(np.diff(closes) / closes[:-1]) * 100) if all(closes[:-1] > 0) else 0

                # VWAP deviation (approximate — use close as proxy)
                vwap = np.mean(closes)
                features["vwap_deviation"] = float((closes[-1] / vwap - 1) * 100) if vwap > 0 else 0
            else:
                features["underlying_momentum_5m"] = 0
                features["underlying_momentum_15m"] = 0
                features["underlying_volatility"] = 0
                features["vwap_deviation"] = 0
        else:
            features["underlying_momentum_5m"] = 0
            features["underlying_momentum_15m"] = 0
            features["underlying_volatility"] = 0
            features["vwap_deviation"] = 0
    else:
        features["underlying_momentum_5m"] = 0
        features["underlying_momentum_15m"] = 0
        features["underlying_volatility"] = 0
        features["vwap_deviation"] = 0

    # --- Cross-features (option vs underlying) ---
    if features["underlying_price"] > 0 and current_price > 0:
        # Option leverage: how much option moves per underlying move
        if features["underlying_momentum_5m"] != 0:
            features["leverage_ratio"] = features["price_momentum_5m"] / features["underlying_momentum_5m"]
        else:
            features["leverage_ratio"] = 0
    else:
        features["leverage_ratio"] = 0

    # Direction indicator
    features["is_call"] = 1 if right == "call" else 0

    return features


def simulate_trade_outcome(
    ohlc_side: pd.DataFrame,
    entry_idx: int,
    max_hold: int = MAX_HOLD_MINUTES,
) -> dict:
    """Simulate a trade entry at entry_idx and compute outcome.

    Returns: {peak_pct, trough_pct, final_pct, hold_minutes, label}
    """
    if entry_idx >= len(ohlc_side) - 5:
        return {"peak_pct": 0, "trough_pct": 0, "final_pct": 0, "hold_minutes": 0, "label": 0}

    entry_price = ohlc_side.iloc[entry_idx]["close"]
    if not entry_price or entry_price <= 0:
        return {"peak_pct": 0, "trough_pct": 0, "final_pct": 0, "hold_minutes": 0, "label": 0}

    end_idx = min(entry_idx + max_hold, len(ohlc_side))
    future = ohlc_side.iloc[entry_idx + 1:end_idx]

    if len(future) == 0:
        return {"peak_pct": 0, "trough_pct": 0, "final_pct": 0, "hold_minutes": 0, "label": 0}

    future_closes = future["close"].dropna()
    if len(future_closes) == 0:
        return {"peak_pct": 0, "trough_pct": 0, "final_pct": 0, "hold_minutes": 0, "label": 0}

    changes = (future_closes / entry_price - 1) * 100

    peak_pct = float(changes.max())
    trough_pct = float(changes.min())
    final_pct = float(changes.iloc[-1])

    # Label: simplified V5-style exit simulation
    # Check if we'd hit profit target (+15%) before stop (-35%)
    for i, pct in enumerate(changes):
        if pct >= WIN_THRESHOLD_PCT:
            return {
                "peak_pct": peak_pct,
                "trough_pct": float(changes.iloc[:i+1].min()),
                "final_pct": float(pct),
                "hold_minutes": i + 1,
                "label": 1,  # WIN
            }
        if pct <= LOSS_THRESHOLD_PCT:
            return {
                "peak_pct": float(changes.iloc[:i+1].max()),
                "trough_pct": float(pct),
                "final_pct": float(pct),
                "hold_minutes": i + 1,
                "label": 0,  # LOSS
            }

    # Didn't hit either threshold — classify based on final P&L
    label = 1 if final_pct > 5 else 0
    return {
        "peak_pct": peak_pct,
        "trough_pct": trough_pct,
        "final_pct": final_pct,
        "hold_minutes": len(future_closes),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset_for_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    sample_interval: int = 5,  # sample every N minutes (avoid correlated samples)
) -> pd.DataFrame:
    """Generate feature+label dataset for a single ticker across all available dates."""

    # Get all dates with data
    dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
        (ticker,),
    ).fetchall()]

    if not dates:
        print(f"  {ticker}: no data in thetadata_options.db")
        return pd.DataFrame()

    print(f"  {ticker}: {len(dates)} trading days with data")

    all_rows = []
    for dt in dates:
        data = load_day_data(conn, ticker, dt)
        if data["ohlc"].empty:
            continue

        for right in ["call", "put"]:
            ohlc_side = data["ohlc"][data["ohlc"]["right"].str.upper() == right.upper()].reset_index(drop=True)
            if len(ohlc_side) < 30:
                continue

            # Sample every N minutes, starting after 15-min warmup
            for idx in range(15, len(ohlc_side) - 10, sample_interval):
                features = compute_features_at_timestamp(data, right, idx)
                if features is None:
                    continue

                outcome = simulate_trade_outcome(ohlc_side, idx)
                if outcome["hold_minutes"] == 0:
                    continue

                row = {**features, **outcome, "ticker": ticker, "date": dt}
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    wins = (df["label"] == 1).sum()
    total = len(df)
    print(f"  {ticker}: {total} samples, {wins} wins ({wins/total*100:.1f}% WR)")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "minutes_since_open", "time_bucket",
    "option_price", "price_momentum_5m", "price_momentum_15m",
    "price_volatility", "price_range_position",
    "current_volume", "avg_volume", "volume_surge", "volume_acceleration",
    "bid_ask_spread", "spread_pct", "bid_size", "ask_size",
    "size_imbalance", "spread_trend",
    "iv", "delta", "theta", "vega",
    "iv_momentum", "iv_percentile", "iv_acceleration",
    "underlying_price", "underlying_momentum_5m", "underlying_momentum_15m",
    "underlying_volatility", "vwap_deviation",
    "leverage_ratio", "is_call",
]


def train_model(df: pd.DataFrame, ticker: str) -> dict | None:
    """Train a LightGBM model for a single ticker. Returns metrics dict."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  ERROR: pip install lightgbm")
        return None

    if len(df) < 100:
        print(f"  {ticker}: only {len(df)} samples — skipping (need >= 100)")
        return None

    # Use available feature columns (some might be missing)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["label"].values

    # Time-based split: train on first 80%, test on last 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(np.unique(y_train)) < 2:
        print(f"  {ticker}: only one class in training set — skipping")
        return None

    # LightGBM with conservative params to avoid overfitting
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(20), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        valid_sets=[test_data],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    acc = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        auc = 0.5

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    metrics = {
        "ticker": ticker,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_win_rate": float(y_train.mean()),
        "test_win_rate": float(y_test.mean()),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "top_features": top_features,
    }

    # Save model
    model_path = MODEL_DIR / f"signal_{ticker}.lgb"
    model.save_model(str(model_path))

    # Save metadata
    meta = {
        "ticker": ticker,
        "features": available,
        "metrics": {k: v for k, v in metrics.items() if k != "top_features"},
        "top_features": [(f, float(v)) for f, v in top_features],
        "trained_at": datetime.utcnow().isoformat(),
        "samples": len(df),
    }
    meta_path = MODEL_DIR / f"signal_{ticker}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  {ticker} MODEL RESULTS:")
    print(f"    Train: {len(y_train)} samples, {y_train.mean()*100:.1f}% base WR")
    print(f"    Test:  {len(y_test)} samples, {y_test.mean()*100:.1f}% base WR")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {precision*100:.1f}% (of predicted wins, how many actually won)")
    print(f"    Recall:    {recall*100:.1f}% (of actual wins, how many we caught)")
    print(f"    AUC:       {auc:.3f}")
    print(f"    Top features:")
    for feat, imp in top_features[:5]:
        print(f"      {feat}: {imp:.0f}")
    print(f"    Saved: {model_path}")

    return metrics


def train_generic_model(all_dfs: dict[str, pd.DataFrame]) -> dict | None:
    """Train a generic model on all tickers combined (fallback for unknown tickers)."""
    combined = pd.concat([df for df in all_dfs.values() if len(df) > 0], ignore_index=True)
    if len(combined) < 200:
        print("  Not enough combined data for generic model")
        return None

    print(f"\n  GENERIC model: {len(combined)} total samples across {len(all_dfs)} tickers")
    return train_model(combined, "GENERIC")


# ---------------------------------------------------------------------------
# Backtest with trained models
# ---------------------------------------------------------------------------

def run_backtest(
    conn: sqlite3.Connection,
    ticker: str,
    threshold: float = 0.55,
    portfolio: float = 20000,
) -> dict:
    """Backtest a trained model on out-of-sample data."""
    try:
        import lightgbm as lgb
    except ImportError:
        return {}

    model_path = MODEL_DIR / f"signal_{ticker}.lgb"
    meta_path = MODEL_DIR / f"signal_{ticker}_meta.json"

    if not model_path.exists():
        # Try generic
        model_path = MODEL_DIR / "signal_GENERIC.lgb"
        meta_path = MODEL_DIR / "signal_GENERIC_meta.json"
        if not model_path.exists():
            print(f"  {ticker}: no model found")
            return {}

    model = lgb.Booster(model_name=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)

    features = meta["features"]

    # Generate dataset (last 20% is test set — same as training eval)
    df = generate_dataset_for_ticker(conn, ticker, sample_interval=5)
    if df.empty:
        return {}

    # Use last 20% for backtest
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[[c for c in features if c in test_df.columns]].fillna(0).values
    y_pred = model.predict(X_test)

    # Simulate trading only when model confidence > threshold
    trades = []
    balance = portfolio
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if y_pred[i] < threshold:
            continue

        # Size: $500 per trade (simplified)
        trade_size = min(500, balance * 0.05)
        entry_price = row.get("option_price", 1) or 1
        contracts = max(1, int(trade_size / (entry_price * 100)))

        pnl_pct = row["final_pct"]
        pnl_dollars = contracts * entry_price * 100 * (pnl_pct / 100)
        balance += pnl_dollars

        trades.append({
            "date": row.get("date", ""),
            "direction": "call" if row.get("is_call", 1) else "put",
            "confidence": float(y_pred[i]),
            "entry_price": entry_price,
            "pnl_pct": pnl_pct,
            "pnl_dollars": pnl_dollars,
            "balance": balance,
            "label": int(row["label"]),
        })

    if not trades:
        print(f"  {ticker}: 0 trades above threshold {threshold}")
        return {}

    wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
    total_pnl = sum(t["pnl_dollars"] for t in trades)

    result = {
        "ticker": ticker,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100,
        "total_pnl": total_pnl,
        "final_balance": balance,
        "avg_confidence": np.mean([t["confidence"] for t in trades]),
        "threshold": threshold,
    }

    print(f"\n  {ticker} BACKTEST (threshold={threshold}):")
    print(f"    Trades: {len(trades)}, Wins: {wins} ({result['win_rate']:.1f}%)")
    print(f"    P&L: ${total_pnl:,.0f}")
    print(f"    Balance: ${portfolio:,.0f} → ${balance:,.0f}")
    print(f"    Avg confidence: {result['avg_confidence']:.3f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train per-ticker ML signal models from option price data")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    parser.add_argument("--db", type=str, default=THETADATA_DB, help="ThetaData DB path")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing models only")
    parser.add_argument("--backtest", action="store_true", help="Run backtest with trained models")
    parser.add_argument("--threshold", type=float, default=0.55, help="Backtest confidence threshold")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sample every N minutes")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: ThetaData DB not found at {args.db}")
        print(f"Run: python scripts/download_thetadata.py first")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"\n{'='*70}")
    print(f"ML SIGNAL TRAINING — Per-Ticker Option Pattern Detection")
    print(f"{'='*70}")
    print(f"DB:       {args.db}")
    print(f"Tickers:  {', '.join(tickers)}")
    print(f"Models:   {MODEL_DIR}")
    print(f"{'='*70}\n")

    if args.backtest:
        print("BACKTEST MODE\n")
        all_results = []
        for ticker in tickers:
            result = run_backtest(conn, ticker, threshold=args.threshold)
            if result:
                all_results.append(result)

        if all_results:
            print(f"\n{'='*70}")
            print(f"BACKTEST SUMMARY (threshold={args.threshold})")
            print(f"{'='*70}")
            total_trades = sum(r["trades"] for r in all_results)
            total_wins = sum(r["wins"] for r in all_results)
            total_pnl = sum(r["total_pnl"] for r in all_results)
            print(f"Total trades: {total_trades}")
            print(f"Total wins:   {total_wins} ({total_wins/total_trades*100:.1f}%)")
            print(f"Total P&L:    ${total_pnl:,.0f}")
        return

    # Generate datasets
    print("GENERATING DATASETS\n")
    all_dfs = {}
    for ticker in tickers:
        df = generate_dataset_for_ticker(conn, ticker, sample_interval=args.sample_interval)
        if not df.empty:
            all_dfs[ticker] = df

    if not all_dfs:
        print("\nNo data found. Run download_thetadata.py first.")
        conn.close()
        return

    # Train models
    print(f"\n{'='*70}")
    print("TRAINING MODELS\n")

    all_metrics = []
    for ticker, df in all_dfs.items():
        metrics = train_model(df, ticker)
        if metrics:
            all_metrics.append(metrics)

    # Train generic fallback
    generic = train_generic_model(all_dfs)
    if generic:
        all_metrics.append(generic)

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Samples':>8} {'Base WR':>8} {'Acc':>6} {'Prec':>6} {'AUC':>6}")
    print("-" * 50)
    for m in all_metrics:
        print(f"{m['ticker']:<8} {m['test_samples']:>8} {m['test_win_rate']*100:>7.1f}% {m['accuracy']*100:>5.1f}% {m['precision']*100:>5.1f}% {m['auc']:>5.3f}")

    conn.close()
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
