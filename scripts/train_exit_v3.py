"""Train v3 ML exit model on real signals + harvester data.

Uses actual trade signals matched to harvester snapshots (with Greeks),
plus underlying price confirmation features. Trains a "should I exit now?"
classifier and an "expected future P&L" regressor.

Key improvements over v2:
  - Real Greeks from harvester (theta, delta, gamma, IV)
  - Underlying confirmation (direction, magnitude)
  - DTE-aware (0DTE vs multi-day)
  - Premium trajectory features (velocity, acceleration, recovery)
  - Trained on our actual signal entries, not random sample points

Usage:
    python scripts/train_exit_v3.py                  # train + backtest
    python scripts/train_exit_v3.py --backtest-only   # just backtest existing models
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIGNALS_DB = os.environ.get("SIGNALS_DB", str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db"))
HARVESTER_DB = os.environ.get("HARVESTER_DB", str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db"))
MODELS_DIR = os.environ.get("ML_MODELS_DIR", str(PROJECT_DIR / "journal" / "models"))

# Label thresholds
EXIT_LOOKAHEAD_MINUTES = 30  # look 30 min ahead to decide if exiting now is smart
GOOD_EXIT_THRESHOLD_PCT = -5.0  # if future P&L < -5%, exiting was correct


def load_signals() -> list[dict]:
    """Load all parsed trade signals with tick data available."""
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, ticker, direction, score,
               atm_premium, otm_premium, strike,
               created_at, sentiment, expiry
        FROM trade_signals
        WHERE score >= 70
        ORDER BY created_at
    """).fetchall()

    signals = []
    for r in rows:
        sig = dict(r)
        sig["premium"] = sig["atm_premium"] or sig["otm_premium"]
        # Map sentiment to option_type for contract ticker building
        sent = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        sig["option_type"] = "put" if sent in ("bearish", "put") else "call"
        if sig["premium"] and sig["premium"] > 0 and sig["strike"]:
            signals.append(sig)

    conn.close()
    print(f"Loaded {len(signals)} signals from {SIGNALS_DB}")
    return signals


def build_contract_ticker(ticker: str, expiry: str, strike: float,
                          option_type: str) -> str:
    """Build Polygon-style contract ticker like O:TSLA260429C00250000."""
    if not expiry or not strike:
        return ""
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type and option_type.lower() in ("call", "bullish", "c") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def load_ticks_for_signal(harvester_conn: sqlite3.Connection, signal: dict) -> pd.DataFrame | None:
    """Load harvester snapshots for a signal's option contract.

    Tries multiple expiry dates since not all tickers have 0DTE every day.
    """
    ticker = signal["ticker"]
    strike = signal["strike"]
    created_at = signal["created_at"]
    option_type = signal.get("option_type", "call")
    direction = signal.get("direction", "bullish")

    if not isinstance(created_at, str):
        return None

    sig_date = created_at[:10]

    # Try multiple expiry dates: same day, +1, +2, Friday
    from datetime import date
    sig_dt = datetime.strptime(sig_date, "%Y-%m-%d").date()
    candidates = [sig_dt]
    for delta in range(1, 6):
        d = sig_dt + timedelta(days=delta)
        if d.weekday() < 5:  # skip weekends
            candidates.append(d)
            if len(candidates) >= 4:
                break

    rows = None
    used_expiry = None
    for exp_date in candidates:
        expiry = exp_date.strftime("%Y-%m-%d")
        ct = build_contract_ticker(ticker, expiry, strike, option_type)
        if not ct:
            continue

        rows = harvester_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume, open_interest
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()

        if rows and len(rows) >= 10:
            used_expiry = expiry
            break
        rows = None

    if not rows or len(rows) < 10:
        return None

    # Store DTE on the signal for later use
    if used_expiry:
        exp_d = datetime.strptime(used_expiry, "%Y-%m-%d").date()
        sig_d = datetime.strptime(sig_date, "%Y-%m-%d").date()
        signal["_dte"] = (exp_d - sig_d).days
        signal["_expiry"] = used_expiry
    else:
        signal["_dte"] = 0

    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega",
        "volume", "open_interest"
    ])

    # Use midpoint as premium, fall back to (bid+ask)/2
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0,
                                          (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])

    if len(df) < 10:
        return None

    # Parse timestamps
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def compute_exit_features(df: pd.DataFrame, idx: int, entry_idx: int,
                          entry_premium: float, direction: str,
                          dte: int) -> dict | None:
    """Compute features at a given point in the trade for exit decision."""
    if idx < entry_idx + 2 or entry_premium <= 0:
        return None

    current = df.iloc[idx]
    entry_row = df.iloc[entry_idx]
    premium = current["premium"]

    if premium <= 0 or np.isnan(premium):
        return None

    # Basic P&L
    pnl_pct = (premium - entry_premium) / entry_premium * 100

    # Peak/trough since entry
    premiums_since_entry = df["premium"].iloc[entry_idx:idx + 1].values
    peak_premium = float(np.nanmax(premiums_since_entry))
    trough_premium = float(np.nanmin(premiums_since_entry))
    mfe_pct = (peak_premium - entry_premium) / entry_premium * 100
    mae_pct = (trough_premium - entry_premium) / entry_premium * 100
    dd_from_peak = (peak_premium - premium) / peak_premium * 100 if peak_premium > 0 else 0

    # Time features
    elapsed_seconds = (current["ts"] - entry_row["ts"]).total_seconds()
    elapsed_minutes = elapsed_seconds / 60

    if elapsed_minutes < 1:
        return None

    # Time of day (ET approximation — EDT offset)
    ts_utc = current["ts"]
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_et = ts_utc - timedelta(hours=4)
    hour_et = ts_et.hour
    minute_et = ts_et.minute
    minutes_since_open = (hour_et - 9) * 60 + (minute_et - 30)

    # Premium velocity (rate of change)
    lookbacks = {"5": 5, "10": 10, "20": 20}
    velocities = {}
    for name, lb in lookbacks.items():
        lb_idx = max(entry_idx, idx - lb)
        if lb_idx < idx and df["premium"].iloc[lb_idx] > 0:
            velocities[f"vel_{name}"] = (premium - df["premium"].iloc[lb_idx]) / df["premium"].iloc[lb_idx] * 100
        else:
            velocities[f"vel_{name}"] = 0.0

    # Acceleration (change in velocity)
    if idx >= entry_idx + 10:
        v_now = velocities["vel_5"]
        prev_idx = max(entry_idx, idx - 10)
        prev_5_idx = max(entry_idx, prev_idx - 5)
        if df["premium"].iloc[prev_5_idx] > 0:
            v_prev = (df["premium"].iloc[prev_idx] - df["premium"].iloc[prev_5_idx]) / df["premium"].iloc[prev_5_idx] * 100
        else:
            v_prev = 0
        accel = v_now - v_prev
    else:
        accel = 0.0

    # Consecutive down ticks
    consec_down = 0
    for j in range(idx, max(entry_idx, idx - 20), -1):
        if j > 0 and df["premium"].iloc[j] < df["premium"].iloc[j - 1]:
            consec_down += 1
        else:
            break

    # Bars since new high
    bars_since_high = 0
    for j in range(idx, entry_idx - 1, -1):
        if df["premium"].iloc[j] >= peak_premium * 0.995:
            bars_since_high = idx - j
            break

    # Recovery: bouncing from recent low?
    recent_window = max(entry_idx, idx - 10)
    recent_low = float(np.nanmin(df["premium"].iloc[recent_window:idx + 1].values))
    recovery_pct = (premium - recent_low) / recent_low * 100 if recent_low > 0 else 0

    # MFE retracement ratio
    mfe_retrace = dd_from_peak / mfe_pct if mfe_pct > 1 else 0

    # Greeks (from harvester — real data!)
    theta = current.get("theta", 0) or 0
    delta = current.get("delta", 0) or 0
    gamma = current.get("gamma", 0) or 0
    iv = current.get("iv", 0) or 0
    vega = current.get("vega", 0) or 0

    # Entry Greeks for comparison
    entry_theta = entry_row.get("theta", 0) or 0
    entry_delta = entry_row.get("delta", 0) or 0
    entry_iv = entry_row.get("iv", 0) or 0

    theta_change = theta - entry_theta
    delta_change = delta - entry_delta
    iv_change = iv - entry_iv

    # Underlying confirmation
    u_current = current.get("underlying_price", 0) or 0
    u_entry = entry_row.get("underlying_price", 0) or 0
    u_pnl = 0.0
    u_confirming = 0

    if u_current > 0 and u_entry > 0:
        u_move = (u_current - u_entry) / u_entry * 100
        if direction in ("bullish", "call", "long"):
            u_pnl = u_move
            u_confirming = 1 if u_move > 0.1 else (-1 if u_move < -0.1 else 0)
        else:
            u_pnl = -u_move
            u_confirming = 1 if u_move < -0.1 else (-1 if u_move > 0.1 else 0)

    # Underlying velocity (last 5 ticks)
    u_vel = 0.0
    if idx >= entry_idx + 5:
        u_prev = df["underlying_price"].iloc[idx - 5]
        if u_prev and u_prev > 0 and u_current > 0:
            u_vel = (u_current - u_prev) / u_prev * 100

    # Volume
    vol = current.get("volume", 0) or 0
    entry_vol = entry_row.get("volume", 0) or 0
    vol_ratio = vol / entry_vol if entry_vol and entry_vol > 0 else 1.0

    features = {
        # P&L state
        "pnl_pct": pnl_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "dd_from_peak": dd_from_peak,
        "mfe_retrace": min(mfe_retrace, 3.0),
        "recovery_pct": recovery_pct,

        # Time
        "elapsed_minutes": elapsed_minutes,
        "minutes_since_open": minutes_since_open,
        "hour_et": hour_et,

        # Premium dynamics
        "vel_5": velocities["vel_5"],
        "vel_10": velocities["vel_10"],
        "vel_20": velocities["vel_20"],
        "accel": accel,
        "consec_down": consec_down,
        "bars_since_high": bars_since_high,

        # Greeks (real from harvester)
        "theta": theta,
        "delta": abs(delta),  # magnitude matters more than sign
        "gamma": gamma,
        "iv": iv,
        "vega": vega,
        "theta_change": theta_change,
        "delta_change": delta_change,
        "iv_change": iv_change,

        # Underlying confirmation
        "u_pnl": u_pnl,
        "u_confirming": u_confirming,
        "u_vel": u_vel,

        # DTE
        "is_0dte": 1 if dte == 0 else 0,
        "dte": dte,

        # Volume
        "vol_ratio": min(vol_ratio, 10.0),

        # Entry context
        "entry_premium": entry_premium,
    }

    return features


def compute_exit_label(df: pd.DataFrame, idx: int, entry_premium: float,
                       lookahead: int = 30) -> dict | None:
    """Compute label: was exiting NOW the right call?

    Looks ahead N ticks. If future P&L drops significantly,
    exiting was correct (label=1). If future P&L improves,
    holding was correct (label=0).
    """
    end_idx = min(idx + lookahead, len(df))
    if end_idx <= idx + 3:
        return None

    current_premium = df["premium"].iloc[idx]
    future_premiums = df["premium"].iloc[idx + 1:end_idx].values
    future_premiums = future_premiums[~np.isnan(future_premiums)]

    if len(future_premiums) < 3:
        return None

    # Best and worst future premium
    future_best = float(np.nanmax(future_premiums))
    future_worst = float(np.nanmin(future_premiums))
    future_end = float(future_premiums[-1])

    # Future P&L relative to current
    future_best_pct = (future_best - current_premium) / current_premium * 100 if current_premium > 0 else 0
    future_worst_pct = (future_worst - current_premium) / current_premium * 100 if current_premium > 0 else 0
    future_end_pct = (future_end - current_premium) / current_premium * 100 if current_premium > 0 else 0

    # "Should exit" if:
    # 1. Premium goes lower from here (future end < current)
    # 2. OR the best future gain is tiny but worst drawdown is significant
    should_exit = 1 if future_end_pct < GOOD_EXIT_THRESHOLD_PCT else 0

    # Also: expected future value relative to entry
    future_best_from_entry = (future_best - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

    return {
        "should_exit": should_exit,
        "future_pnl_pct": future_end_pct,
        "future_best_pct": future_best_pct,
        "future_worst_pct": future_worst_pct,
        "future_best_from_entry": future_best_from_entry,
    }


def build_dataset() -> pd.DataFrame:
    """Build training dataset from real signals + harvester data."""
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    all_rows = []
    matched = 0
    no_ticks = 0

    for sig in signals:
        created_day = sig["created_at"][:10]

        direction = sig["direction"]
        if not direction:
            direction = "bullish" if sig.get("option_type", "call").lower() in ("call", "c") else "bearish"

        # Load ticks (also finds actual expiry)
        df = load_ticks_for_signal(harvester_conn, sig)
        if df is None:
            no_ticks += 1
            continue

        # DTE from the actual expiry found by load_ticks
        dte = sig.get("_dte", 0)

        matched += 1
        entry_premium = sig["premium"]

        # Find entry point (first tick near signal time)
        entry_idx = 0

        # Sample every 5 ticks (not every single tick — too correlated)
        sample_interval = 5
        for idx in range(entry_idx + sample_interval, len(df) - 30, sample_interval):
            features = compute_exit_features(df, idx, entry_idx, entry_premium,
                                             direction, dte)
            if features is None:
                continue

            label = compute_exit_label(df, idx, entry_premium)
            if label is None:
                continue

            row = {
                "signal_id": sig["id"],
                "ticker": sig["ticker"],
                "direction": direction,
                "score": sig["score"],
                **features,
                **label,
            }
            all_rows.append(row)

    harvester_conn.close()

    print(f"\nDataset: {len(all_rows):,} samples from {matched} signals ({no_ticks} signals had no tick data)")

    return pd.DataFrame(all_rows)


FEATURE_COLS = [
    "pnl_pct", "mfe_pct", "mae_pct", "dd_from_peak", "mfe_retrace", "recovery_pct",
    "elapsed_minutes", "minutes_since_open", "hour_et",
    "vel_5", "vel_10", "vel_20", "accel", "consec_down", "bars_since_high",
    "theta", "delta", "gamma", "iv", "vega",
    "theta_change", "delta_change", "iv_change",
    "u_pnl", "u_confirming", "u_vel",
    "is_0dte", "dte",
    "vol_ratio", "entry_premium",
]


def train_exit_classifier(df: pd.DataFrame):
    """Train exit classifier: should we exit now?"""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    print("\n" + "=" * 80)
    print("EXIT CLASSIFIER — Should we exit now?")
    print("=" * 80)

    valid = df.dropna(subset=FEATURE_COLS + ["should_exit"])
    print(f"Training samples: {len(valid):,}")
    print(f"Exit rate: {valid['should_exit'].mean():.1%}")

    X = valid[FEATURE_COLS].values
    y = valid["should_exit"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0

    dtrain = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    dtest = lgb.Dataset(X_test, y_test, feature_name=FEATURE_COLS, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["HOLD", "EXIT"]))
    try:
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    except ValueError:
        pass

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop 15 Features (exit classifier):")
    for i in sorted_idx[:15]:
        print(f"  {FEATURE_COLS[i]:<30} {importance[i]:>10.0f}")

    # Threshold analysis
    print("\nThreshold Analysis:")
    print(f"  {'Thresh':<8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Exits':>7}")
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        pred = (y_pred_proba >= thresh).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {thresh:<8.2f} {prec:>5.1%} {rec:>6.1%} {f1:>5.1%} {pred.sum():>7,}/{len(pred):,}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "exit_clf_v3.lgb")
    model.save_model(model_path)
    print(f"\nSaved: {model_path}")

    return model


def train_future_regressor(df: pd.DataFrame):
    """Train regressor: what's the expected future P&L from here?"""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\n" + "=" * 80)
    print("FUTURE P&L REGRESSOR — What happens if we hold?")
    print("=" * 80)

    valid = df.dropna(subset=FEATURE_COLS + ["future_pnl_pct"])
    # Clip extremes
    valid = valid[valid["future_pnl_pct"].between(-100, 200)].copy()

    print(f"Training samples: {len(valid):,}")
    print(f"Future P&L: mean={valid['future_pnl_pct'].mean():.1f}%, "
          f"median={valid['future_pnl_pct'].median():.1f}%")

    X = valid[FEATURE_COLS].values
    y = valid["future_pnl_pct"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = lgb.Dataset(X_train, y_train, feature_name=FEATURE_COLS)
    dtest = lgb.Dataset(X_test, y_test, feature_name=FEATURE_COLS, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_test)

    print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.2f}%")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop 15 Features (future regressor):")
    for i in sorted_idx[:15]:
        print(f"  {FEATURE_COLS[i]:<30} {importance[i]:>10.0f}")

    # Bucket analysis
    print("\nPredicted Future P&L Buckets vs Actual:")
    print(f"  {'Predicted':>20} {'N':>6} {'Actual Mean':>12} {'Actual Med':>11} {'% Exit Correct':>15}")
    for lo, hi in [(-999, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 999)]:
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            actual = y_test[mask]
            exit_correct = (actual < GOOD_EXIT_THRESHOLD_PCT).mean()
            label = f"[{lo:+d}%, {hi:+d}%)"
            if lo <= -999:
                label = f"[<{hi}%)"
            if hi >= 999:
                label = f"[{lo}%+)"
            print(f"  {label:>20} {mask.sum():>6,} {actual.mean():>+11.1f}% "
                  f"{np.median(actual):>+10.1f}% {exit_correct:>14.1%}")

    # Save
    model_path = os.path.join(MODELS_DIR, "exit_reg_v3.lgb")
    model.save_model(model_path)
    print(f"\nSaved: {model_path}")

    return model


def backtest_ml_on_signals(clf, reg):
    """Backtest the trained models on actual signals — simulate exit decisions."""
    print("\n" + "=" * 80)
    print("BACKTEST: ML Exit on Real Signals")
    print("=" * 80)

    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Simulate each signal with ML-guided exits vs v5b baseline
    results_ml = []
    results_baseline = []

    for sig in signals:
        expiry = sig.get("expiry") or sig["created_at"][:10]
        created_day = sig["created_at"][:10]
        try:
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() -
                   datetime.strptime(created_day, "%Y-%m-%d").date()).days
        except (ValueError, TypeError):
            dte = 0

        direction = sig["direction"] or "bullish"
        df = load_ticks_for_signal(harvester_conn, sig)
        if df is None:
            continue

        entry_premium = sig["premium"]
        entry_idx = 0
        ticker = sig["ticker"]
        score = sig["score"] or 80
        day = sig["created_at"][:10]

        # Determine contracts (v5b sizing)
        cost_per = entry_premium * 100
        portfolio = 8000
        target = portfolio * 0.75 / 5
        if score >= 95:
            mult = 1.0
        elif score >= 90:
            mult = 0.75
        elif score >= 85:
            mult = 0.5
        else:
            mult = 0.25
        scaled = target * mult
        contracts = max(1, int(scaled / cost_per)) if cost_per > 0 else 1
        contracts = min(contracts, int(portfolio * 0.15 / cost_per)) if cost_per > 0 else contracts

        # === ML EXIT SIMULATION ===
        ml_exit_idx = None
        ml_exit_reason = "eod"
        peak_ml = entry_premium

        for idx in range(entry_idx + 3, len(df)):
            premium = df["premium"].iloc[idx]
            if np.isnan(premium) or premium <= 0:
                continue
            peak_ml = max(peak_ml, premium)

            features = compute_exit_features(df, idx, entry_idx, entry_premium,
                                             direction, dte)
            if features is None:
                continue

            X = np.array([[features[f] for f in FEATURE_COLS]])
            sell_prob = float(clf.predict(X)[0])
            exp_future = float(reg.predict(X)[0])

            # ML exit logic: combo of classifier + regressor
            if sell_prob > 0.55 and exp_future < -3.0:
                ml_exit_idx = idx
                ml_exit_reason = f"ml_combo(p={sell_prob:.2f},f={exp_future:+.1f}%)"
                break
            elif features["elapsed_minutes"] > 10 and exp_future < -15.0:
                ml_exit_idx = idx
                ml_exit_reason = f"ml_reg_bearish(f={exp_future:+.1f}%)"
                break

            # Hard backstop
            drop_pct = (entry_premium - premium) / entry_premium * 100
            if drop_pct >= 55:
                ml_exit_idx = idx
                ml_exit_reason = "backstop_55%"
                break

        if ml_exit_idx is None:
            ml_exit_idx = len(df) - 1

        ml_exit_prem = df["premium"].iloc[ml_exit_idx]
        ml_pnl = (ml_exit_prem - entry_premium) * contracts * 100
        ml_hold = (df["ts"].iloc[ml_exit_idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60

        results_ml.append({
            "ticker": ticker, "day": day, "score": score,
            "entry": entry_premium, "contracts": contracts,
            "exit_prem": ml_exit_prem,
            "pnl": ml_pnl, "hold": ml_hold,
            "reason": ml_exit_reason, "dte": dte,
            "direction": direction,
        })

        # === BASELINE (v5b) SIMULATION ===
        bl_exit_idx = None
        bl_reason = "eod"
        peak_bl = entry_premium
        trough_bl = entry_premium

        for idx in range(entry_idx + 3, len(df)):
            premium = df["premium"].iloc[idx]
            if np.isnan(premium) or premium <= 0:
                continue
            peak_bl = max(peak_bl, premium)
            trough_bl = min(trough_bl, premium)

            gain_pct = (premium - entry_premium) / entry_premium * 100
            drop_pct = (entry_premium - premium) / entry_premium * 100
            peak_gain = (peak_bl - entry_premium) / entry_premium * 100
            dd = (peak_bl - premium) / peak_bl * 100 if peak_bl > 0 else 0

            elapsed = (df["ts"].iloc[idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60

            # v5b graduated stop (tight=35%, wide=50%)
            if elapsed >= 5:
                u_cur = df["underlying_price"].iloc[idx] or 0
                u_ent = df["underlying_price"].iloc[entry_idx] or 0
                if u_cur > 0 and u_ent > 0:
                    u_move = (u_cur - u_ent) / u_ent * 100
                    if direction in ("bullish", "call"):
                        u_against = u_move < -0.4
                    else:
                        u_against = u_move > 0.4
                else:
                    u_against = False

                stop = 35 if u_against else 50
                if drop_pct >= stop:
                    bl_exit_idx = idx
                    bl_reason = f"hard_stop({stop}%)"
                    break

                # Backstop
                if drop_pct >= 65:
                    bl_exit_idx = idx
                    bl_reason = "backstop_65%"
                    break

            # Scalp trail
            if peak_gain >= 20:
                fade = (peak_bl - premium) / (peak_bl - entry_premium) * 100 if peak_bl > entry_premium else 0
                if fade >= 60:
                    # Check underlying
                    u_cur = df["underlying_price"].iloc[idx] or 0
                    u_ent = df["underlying_price"].iloc[entry_idx] or 0
                    u_confirm = False
                    if u_cur > 0 and u_ent > 0:
                        u_m = (u_cur - u_ent) / u_ent * 100
                        u_confirm = (u_m > 0.2) if direction in ("bullish", "call") else (u_m < -0.2)
                    if not u_confirm:
                        bl_exit_idx = idx
                        bl_reason = "scalp_trail"
                        break

            # Adaptive trail
            if peak_gain >= 35:
                if peak_gain >= 400:
                    trail_w = 30
                elif peak_gain >= 150:
                    trail_w = 45
                else:
                    trail_w = 35
                if dd >= trail_w:
                    bl_exit_idx = idx
                    bl_reason = f"adaptive({trail_w}%)"
                    break

            # Checkpoint (v5b: -30%)
            if elapsed >= 5 and drop_pct >= 30:
                u_cur = df["underlying_price"].iloc[idx] or 0
                u_ent = df["underlying_price"].iloc[entry_idx] or 0
                if u_cur > 0 and u_ent > 0:
                    u_m = (u_cur - u_ent) / u_ent * 100
                    if direction in ("bullish", "call"):
                        u_ag = u_m < -0.3
                    else:
                        u_ag = u_m > 0.3
                    if u_ag:
                        bl_exit_idx = idx
                        bl_reason = "checkpoint"
                        break

        if bl_exit_idx is None:
            bl_exit_idx = len(df) - 1

        bl_exit_prem = df["premium"].iloc[bl_exit_idx]
        bl_pnl = (bl_exit_prem - entry_premium) * contracts * 100
        bl_hold = (df["ts"].iloc[bl_exit_idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60

        results_baseline.append({
            "ticker": ticker, "day": day, "score": score,
            "entry": entry_premium, "contracts": contracts,
            "exit_prem": bl_exit_prem,
            "pnl": bl_pnl, "hold": bl_hold,
            "reason": bl_reason, "dte": dte,
        })

    harvester_conn.close()

    # === COMPARE ===
    print(f"\n{'Metric':<25} {'ML Exit':>12} {'v5b Baseline':>12}")
    print("-" * 55)

    ml_pnls = [r["pnl"] for r in results_ml]
    bl_pnls = [r["pnl"] for r in results_baseline]

    ml_wins = sum(1 for p in ml_pnls if p > 0)
    bl_wins = sum(1 for p in bl_pnls if p > 0)
    n = len(ml_pnls)

    print(f"{'Trades':<25} {n:>12} {len(bl_pnls):>12}")
    print(f"{'Total P&L':<25} ${sum(ml_pnls):>+10,.0f} ${sum(bl_pnls):>+10,.0f}")
    print(f"{'Win Rate':<25} {ml_wins/n:.1%} ({ml_wins}W/{n-ml_wins}L){'':>1} "
          f"{bl_wins/n:.1%} ({bl_wins}W/{n-bl_wins}L)")
    print(f"{'Avg Win':<25} ${np.mean([p for p in ml_pnls if p > 0]):>+10,.0f} "
          f"${np.mean([p for p in bl_pnls if p > 0]):>+10,.0f}")
    print(f"{'Avg Loss':<25} ${np.mean([p for p in ml_pnls if p <= 0]):>+10,.0f} "
          f"${np.mean([p for p in bl_pnls if p <= 0]):>+10,.0f}")
    print(f"{'Avg Hold':<25} {np.mean([r['hold'] for r in results_ml]):>10.0f}m "
          f"{np.mean([r['hold'] for r in results_baseline]):>10.0f}m")

    # Daily breakdown
    days = sorted(set(r["day"] for r in results_ml))
    print(f"\n{'Day':<12} {'ML P&L':>10} {'ML WR':>7} {'v5b P&L':>10} {'v5b WR':>7} {'Winner':>8}")
    print("-" * 60)
    for day in days:
        ml_day = [r for r in results_ml if r["day"] == day]
        bl_day = [r for r in results_baseline if r["day"] == day]
        ml_p = sum(r["pnl"] for r in ml_day)
        bl_p = sum(r["pnl"] for r in bl_day)
        ml_w = sum(1 for r in ml_day if r["pnl"] > 0) / len(ml_day) if ml_day else 0
        bl_w = sum(1 for r in bl_day if r["pnl"] > 0) / len(bl_day) if bl_day else 0
        winner = "ML" if ml_p > bl_p else "v5b"
        print(f"{day:<12} ${ml_p:>+9,.0f} {ml_w:>6.0%} ${bl_p:>+9,.0f} {bl_w:>6.0%} {winner:>8}")

    ml_cum = sum(ml_pnls)
    bl_cum = sum(bl_pnls)
    print(f"{'TOTAL':<12} ${ml_cum:>+9,.0f}         ${bl_cum:>+9,.0f}")

    # 0DTE vs multi-day breakdown
    print(f"\n{'DTE':>4} {'ML P&L':>10} {'ML WR':>7} {'ML N':>5} {'v5b P&L':>10} {'v5b WR':>7}")
    print("-" * 50)
    for dte_val in sorted(set(r["dte"] for r in results_ml)):
        ml_d = [r for r in results_ml if r["dte"] == dte_val]
        bl_d = [r for r in results_baseline if r["dte"] == dte_val]
        ml_p = sum(r["pnl"] for r in ml_d)
        bl_p = sum(r["pnl"] for r in bl_d)
        ml_w = sum(1 for r in ml_d if r["pnl"] > 0) / len(ml_d) if ml_d else 0
        bl_w = sum(1 for r in bl_d if r["pnl"] > 0) / len(bl_d) if bl_d else 0
        print(f"{dte_val:>4} ${ml_p:>+9,.0f} {ml_w:>6.0%} {len(ml_d):>5} ${bl_p:>+9,.0f} {bl_w:>6.0%}")

    # Gate fire breakdown for ML
    print(f"\n{'ML Gate':<30} {'Fires':>6} {'P&L':>10} {'WR':>6}")
    print("-" * 55)
    gate_stats = {}
    for r in results_ml:
        reason = r["reason"].split("(")[0]  # simplify
        if reason not in gate_stats:
            gate_stats[reason] = {"fires": 0, "pnl": 0, "wins": 0}
        gate_stats[reason]["fires"] += 1
        gate_stats[reason]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            gate_stats[reason]["wins"] += 1
    for gate, stats in sorted(gate_stats.items(), key=lambda x: -x[1]["fires"]):
        wr = stats["wins"] / stats["fires"] if stats["fires"] > 0 else 0
        print(f"{gate:<30} {stats['fires']:>6} ${stats['pnl']:>+9,.0f} {wr:>5.0%}")

    # Save meta
    meta = {
        "feature_cols": FEATURE_COLS,
        "exit_lookahead_minutes": EXIT_LOOKAHEAD_MINUTES,
        "good_exit_threshold_pct": GOOD_EXIT_THRESHOLD_PCT,
        "n_signals": n,
        "ml_total_pnl": ml_cum,
        "baseline_total_pnl": bl_cum,
        "ml_win_rate": ml_wins / n if n > 0 else 0,
    }
    meta_path = os.path.join(MODELS_DIR, "exit_v3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta: {meta_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train v3 ML exit model")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Only run backtest with existing models")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.backtest_only:
        import lightgbm as lgb
        clf = lgb.Booster(model_file=os.path.join(MODELS_DIR, "exit_clf_v3.lgb"))
        reg = lgb.Booster(model_file=os.path.join(MODELS_DIR, "exit_reg_v3.lgb"))
        backtest_ml_on_signals(clf, reg)
        return

    print("Building dataset from signals + harvester...")
    df = build_dataset()

    if len(df) < 100:
        print(f"ERROR: Only {len(df)} samples — need at least 100. Check harvester data.")
        sys.exit(1)

    # Save dataset
    csv_path = os.path.join(MODELS_DIR, "exit_v3_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset: {csv_path}")

    # Print 0DTE vs multi-day stats
    print(f"\n--- Dataset DTE breakdown ---")
    for dte_val in sorted(df["dte"].unique()):
        subset = df[df["dte"] == dte_val]
        print(f"  DTE={dte_val}: {len(subset):,} samples, "
              f"exit_rate={subset['should_exit'].mean():.1%}, "
              f"avg_future_pnl={subset['future_pnl_pct'].mean():+.1f}%")

    clf = train_exit_classifier(df)
    reg = train_future_regressor(df)

    backtest_ml_on_signals(clf, reg)


if __name__ == "__main__":
    main()
