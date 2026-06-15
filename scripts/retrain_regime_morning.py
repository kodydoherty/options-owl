"""Retrain regime classifier using ONLY features available at 9:45 AM.

The original model used full-day `day_range_pct` which leaks the answer.
This version uses:
  - First 15-min range (proxy, scaled 2x)
  - First 15-min volume (extrapolated)
  - Overnight gap (open vs prev close)
  - Previous day stats (range, volume, 3-day avg range)
  - GEX from prior day close
  - Day of week

Label: 1 = trending day (intraday range > 1.5% AND close near high/low)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
THETA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]


def main():
    print("=" * 70)
    print("REGIME MODEL RETRAINING — Morning-Only Features")
    print("=" * 70)

    theta_conn = sqlite3.connect(THETA_DB)
    theta_conn.execute("PRAGMA journal_mode = WAL")
    theta_conn.execute("PRAGMA busy_timeout = 5000")

    uw_conn = sqlite3.connect(UW_DB) if Path(UW_DB).exists() else None

    rows = []
    for ticker in TICKERS:
        print(f"  Processing {ticker}...", end="", flush=True)

        # Get all dates with stock data
        daily = pd.read_sql_query(
            """SELECT substr(timestamp, 1, 10) as date,
                      MIN(open) as day_open,
                      MAX(high) as day_high,
                      MIN(low) as day_low,
                      SUM(volume) as day_volume
               FROM stock_ohlc WHERE ticker=?
               GROUP BY substr(timestamp, 1, 10)
               HAVING day_high > 0 AND day_low > 0
               ORDER BY date""",
            theta_conn, params=(ticker,),
        )
        if len(daily) < 20:
            print(f" skip ({len(daily)} days)")
            continue

        ticker_count = 0
        for i, day_row in daily.iterrows():
            dt = day_row["date"]
            day_high = day_row["day_high"]
            day_low = day_row["day_low"]

            if not day_high or not day_low or day_high <= 0 or day_low <= 0:
                continue

            # Get close (last bar)
            close_row = theta_conn.execute(
                "SELECT close FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1",
                (ticker, f"{dt}%"),
            ).fetchone()
            if not close_row or not close_row[0] or close_row[0] <= 0:
                continue
            day_close = close_row[0]

            # Get open (first bar)
            open_row = theta_conn.execute(
                "SELECT open FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp ASC LIMIT 1",
                (ticker, f"{dt}%"),
            ).fetchone()
            if not open_row or not open_row[0] or open_row[0] <= 0:
                continue
            day_open = open_row[0]

            # LABEL: trending if range > 1.5% AND close near extreme
            day_range_pct = (day_high - day_low) / day_low * 100
            close_position = (day_close - day_low) / (day_high - day_low) if day_high > day_low else 0.5
            is_trending = day_range_pct > 1.5 and (close_position < 0.2 or close_position > 0.8)

            # === FEATURES (available at 9:45 AM) ===
            f = {}
            f["ticker_idx"] = TICKERS.index(ticker)
            f["day_of_week"] = datetime.strptime(dt, "%Y-%m-%d").weekday()

            # First 15 minutes data
            morning_rows = theta_conn.execute(
                "SELECT open, high, low, close, volume FROM stock_ohlc "
                "WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp LIMIT 15",
                (ticker, dt),
            ).fetchall()

            if len(morning_rows) < 5:
                continue

            morning_high = max(r[1] for r in morning_rows if r[1] and r[1] > 0)
            morning_low = min(r[2] for r in morning_rows if r[2] and r[2] > 0)
            morning_open = morning_rows[0][0] if morning_rows[0][0] else day_open
            morning_close = morning_rows[-1][3] if morning_rows[-1][3] else morning_open
            morning_volume = sum(r[4] for r in morning_rows if r[4])

            if morning_low <= 0:
                continue

            # Morning range (scaled — first 15 min is ~40-60% of full day)
            f["morning_range_pct"] = (morning_high - morning_low) / morning_low * 100
            f["morning_volume"] = float(morning_volume)
            # Morning direction: positive = bullish open
            f["morning_direction"] = (morning_close / morning_open - 1) * 100 if morning_open > 0 else 0
            # Morning body vs range (doji detection)
            morning_body = abs(morning_close - morning_open)
            morning_range = morning_high - morning_low
            f["morning_body_ratio"] = morning_body / morning_range if morning_range > 0 else 0

            # Overnight gap (today's open vs yesterday's close)
            prev_idx = daily.index.get_loc(i) if i in daily.index else None
            if prev_idx is not None and prev_idx > 0:
                prev_row = daily.iloc[prev_idx - 1]
                prev_close_row = theta_conn.execute(
                    "SELECT close FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1",
                    (ticker, f"{prev_row['date']}%"),
                ).fetchone()
                if prev_close_row and prev_close_row[0] and prev_close_row[0] > 0:
                    f["overnight_gap_pct"] = (morning_open / prev_close_row[0] - 1) * 100
                else:
                    f["overnight_gap_pct"] = 0
            else:
                f["overnight_gap_pct"] = 0

            # Previous day stats
            prev_days = daily.iloc[max(0, daily.index.get_loc(i) - 5):daily.index.get_loc(i)] if i in daily.index else pd.DataFrame()
            if len(prev_days) > 0:
                prev = prev_days.iloc[-1]
                f["prev_range_pct"] = (prev["day_high"] - prev["day_low"]) / prev["day_low"] * 100 if prev["day_low"] > 0 else 0
                f["prev_volume"] = float(prev["day_volume"] or 0)

                if len(prev_days) >= 3:
                    recent_ranges = [(r["day_high"] - r["day_low"]) / r["day_low"] * 100
                                     for _, r in prev_days.tail(3).iterrows() if r["day_low"] > 0]
                    f["avg_3d_range"] = float(np.mean(recent_ranges)) if recent_ranges else 0
                    f["range_trend"] = f["morning_range_pct"] / max(f["avg_3d_range"], 0.01) - 1
                else:
                    f["avg_3d_range"] = 0
                    f["range_trend"] = 0

                # Volume trend vs prev days
                prev_vols = [float(r["day_volume"] or 0) for _, r in prev_days.iterrows()]
                avg_prev_vol = np.mean(prev_vols) if prev_vols else 1
                f["volume_vs_prev"] = morning_volume * 26 / max(avg_prev_vol, 1)  # extrapolate to full day
            else:
                f["prev_range_pct"] = 0
                f["prev_volume"] = 0
                f["avg_3d_range"] = 0
                f["range_trend"] = 0
                f["volume_vs_prev"] = 0

            # GEX data from UW (previous day close)
            if uw_conn:
                gex_row = uw_conn.execute(
                    "SELECT call_gamma, put_gamma, call_delta, put_delta FROM greek_exposure "
                    "WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1",
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
            else:
                for k in ["call_gamma", "put_gamma", "net_gamma", "call_delta", "put_delta", "net_delta"]:
                    f[k] = 0

            f["label"] = 1 if is_trending else 0
            f["ticker"] = ticker
            f["date"] = dt
            rows.append(f)
            ticker_count += 1

        print(f" {ticker_count} days")

    theta_conn.close()
    if uw_conn:
        uw_conn.close()

    if not rows:
        print("No data collected!")
        return

    df = pd.DataFrame(rows)
    n_trend = df["label"].sum()
    n_chop = len(df) - n_trend
    print(f"\nCollected {len(df)} samples: {n_trend} trending ({n_trend/len(df)*100:.1f}%), {n_chop} chop")

    meta_cols = ["ticker", "date"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["label"]]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": n_chop / max(n_trend, 1),  # Balance classes
    }

    model = lgb.train(
        params, dtrain, num_boost_round=300,
        valid_sets=[dtest],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(30)],
    )

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)

    print(f"\nRegime Model (Morning): AUC={auc:.3f} Acc={acc:.3f} Prec={prec:.3f} Recall={rec:.3f}")
    print(f"  Predictions: mean={preds.mean():.3f}, min={preds.min():.3f}, max={preds.max():.3f}")
    print(f"  P>0.5: {(preds > 0.5).sum()}/{len(preds)}")
    print(f"  P>0.3: {(preds > 0.3).sum()}/{len(preds)}")

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("\n  Top features:")
    for name, gain in imp[:10]:
        print(f"    {name}: {gain:.0f}")

    # Save (overwrite the old broken model)
    model_path = str(MODEL_DIR / "regime_classifier.txt")
    model.save_model(model_path)
    meta = {
        "features": feature_cols,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "version": "morning_v2",
        "note": "Retrained with morning-only features (no day_range_pct leakage)",
    }
    with open(str(MODEL_DIR / "regime_classifier_meta.json"), "w") as f_out:
        json.dump(meta, f_out, indent=2)
    print(f"\n  Saved to {model_path}")


if __name__ == "__main__":
    main()
