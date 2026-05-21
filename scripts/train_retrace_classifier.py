"""Retrace classifier: will this pullback recover or is it the final peak?

The RIGHT question for our trading system:
  When premium retraces 15%+ from a local high during a run,
  will it RECOVER and go higher, or is this the FINAL peak?

This is exactly what the adaptive trail needs to know:
  - Temporary retrace → HOLD (keep wide trail)
  - Final peak retrace → TIGHTEN (lock profits now)

Training data: every 15%+ retrace event in harvester data for our tickers.
Label: 0 = temporary (recovered), 1 = final (never exceeded the high).

Usage:
    python scripts/train_retrace_classifier.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from options_owl.risk.exit_v5.config import TickerCategory, categorize_ticker

HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_TICKERS = [
    "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "AMZN",
    "META", "GOOGL", "MSFT", "AVGO", "AMD", "MSTR", "PLTR", "MU",
]

RETRACE_THRESHOLD = 15  # % drop from local high to count as retrace
MIN_GAIN_AT_HIGH = 15   # local high must be +15% from entry to matter


def safe_float(v, d=0.0):
    try:
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return d
        return float(v)
    except:
        return d


FEATURE_COLS = [
    # Retrace characteristics
    "retrace_depth",          # how deep is this pullback (%)
    "gain_at_high",           # how much was the run before retrace (%)
    "gain_at_retrace_low",    # current gain from entry at retrace low
    "pct_of_session_peak",    # retrace low / session peak so far
    # Speed of retrace
    "retrace_speed",          # how fast did it drop (% per tick)
    "retrace_ticks",          # how many ticks into the retrace
    # Premium dynamics at retrace low
    "vel_3", "vel_5", "vel_10",
    "prem_std",               # premium volatility during retrace
    # Greeks at retrace low
    "delta", "gamma", "theta", "vega", "iv",
    "delta_change", "iv_change",
    "gamma_delta_ratio",
    # Volume at retrace low
    "volume", "vol_ratio",
    "vol_trend",              # volume trend (increasing = capitulation?)
    # Bid-ask at retrace
    "spread_pct",
    "spread_change",          # spread widening during retrace?
    "size_imbalance",         # bid vs ask size
    # Underlying
    "u_move",                 # underlying move from entry
    "u_vel",                  # underlying velocity
    "u_retrace_pct",          # underlying retrace from its high
    "moneyness",
    "divergence",             # premium vs underlying divergence
    # Time
    "minutes_to_close",
    "session_progress",
    "elapsed_min",            # minutes from start
    # Context
    "is_highvol", "is_index", "is_call",
    # Prior retrace history
    "n_prior_retraces",       # how many retraces already happened
    "avg_prior_retrace_depth", # avg depth of prior retraces (pattern)
    "max_prior_retrace_depth", # deepest prior retrace
]


def extract_retrace_features(ticks_df, premiums, timestamps, retrace_low_idx,
                              local_high, local_high_idx, entry_prem, entry_u,
                              ticker, category, is_call, strike,
                              prior_retraces):
    """Extract features at a retrace low point."""
    idx = retrace_low_idx
    n = len(premiums)
    if idx < 10 or idx >= n:
        return None

    prem = premiums[idx]
    if np.isnan(prem) or prem <= 0:
        return None

    now = timestamps[idx]
    now_dt = pd.Timestamp(now)
    et_hour = now_dt.hour - 4
    if et_hour < 0:
        et_hour += 24
    et_minute = now_dt.minute
    et_decimal = et_hour + et_minute / 60
    if et_decimal < 9.5 or et_decimal > 16.0:
        return None

    retrace_depth = (local_high - prem) / local_high * 100
    gain_at_high = (local_high - entry_prem) / entry_prem * 100
    gain_at_low = (prem - entry_prem) / entry_prem * 100
    pct_of_session_peak = prem / local_high * 100  # local_high IS session peak so far

    # Retrace speed
    retrace_ticks = idx - local_high_idx
    retrace_speed = retrace_depth / max(retrace_ticks, 1)

    # Premium velocities
    vels = {}
    for lb in [3, 5, 10]:
        if idx >= lb:
            pp = premiums[idx - lb]
            vels[f"vel_{lb}"] = (prem - pp) / pp * 100 if pp > 0 else 0
        else:
            vels[f"vel_{lb}"] = 0

    # Premium std during retrace
    retrace_window = premiums[max(0, local_high_idx):idx + 1]
    retrace_valid = retrace_window[~np.isnan(retrace_window)]
    prem_std = np.std(retrace_valid) / prem if len(retrace_valid) > 1 and prem > 0 else 0

    # Greeks
    delta = safe_float(ticks_df["delta"].iloc[idx])
    gamma = safe_float(ticks_df["gamma"].iloc[idx])
    theta = safe_float(ticks_df["theta"].iloc[idx])
    vega = safe_float(ticks_df["vega"].iloc[idx])
    iv = safe_float(ticks_df["iv"].iloc[idx])

    d5 = safe_float(ticks_df["delta"].iloc[idx - 5], delta) if idx >= 5 else delta
    iv5 = safe_float(ticks_df["iv"].iloc[idx - 5], iv) if idx >= 5 else iv
    delta_change = delta - d5
    iv_change = iv - iv5
    gdr = gamma / abs(delta) if abs(delta) > 0.01 else 0

    # Volume
    vol = safe_float(ticks_df["volume"].iloc[idx])
    if idx >= 10:
        prev_vols = [safe_float(ticks_df["volume"].iloc[i]) for i in range(idx - 10, idx)]
        prev_vols = [v for v in prev_vols if v > 0]
        avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    if idx >= 20:
        vr = [safe_float(ticks_df["volume"].iloc[i]) for i in range(idx - 5, idx + 1)]
        vo = [safe_float(ticks_df["volume"].iloc[i]) for i in range(idx - 20, idx - 10)]
        ar = np.mean([v for v in vr if v > 0] or [0])
        ao = np.mean([v for v in vo if v > 0] or [1])
        vol_trend = ar / ao if ao > 0 else 1.0
    else:
        vol_trend = 1.0

    # Bid-ask
    bid = safe_float(ticks_df["bid"].iloc[idx], prem * 0.95)
    ask = safe_float(ticks_df["ask"].iloc[idx], prem * 1.05)
    spread_pct = (ask - bid) / prem * 100 if prem > 0 else 0

    if idx >= 5:
        pb = safe_float(ticks_df["bid"].iloc[idx - 5], bid)
        pa = safe_float(ticks_df["ask"].iloc[idx - 5], ask)
        pp = premiums[idx - 5]
        ps = (pa - pb) / pp * 100 if pp > 0 else spread_pct
        spread_change = spread_pct - ps
    else:
        spread_change = 0

    bs = safe_float(ticks_df["bid_size"].iloc[idx] if "bid_size" in ticks_df.columns else 1, 1)
    ask_s = safe_float(ticks_df["ask_size"].iloc[idx] if "ask_size" in ticks_df.columns else 1, 1)
    size_imbalance = (bs - ask_s) / (bs + ask_s) if (bs + ask_s) > 0 else 0

    # Underlying
    u = safe_float(ticks_df["underlying_price"].iloc[idx])
    u_move = (u - entry_u) / entry_u * 100 if entry_u > 0 else 0

    if idx >= 5:
        pu = safe_float(ticks_df["underlying_price"].iloc[idx - 5], u)
        u_vel = (u - pu) / pu * 100 if pu > 0 else 0
    else:
        u_vel = 0

    # Underlying retrace from its own high
    u_prices = [safe_float(ticks_df["underlying_price"].iloc[i])
                for i in range(min(idx + 1, n))]
    u_valid = [x for x in u_prices if x > 0]
    if u_valid:
        u_high = max(u_valid) if is_call else min(u_valid)
        u_retrace = abs(u - u_high) / u_high * 100 if u_high > 0 else 0
    else:
        u_retrace = 0

    moneyness = 0
    if u > 0 and strike > 0:
        moneyness = ((u - strike) / strike * 100) if is_call else ((strike - u) / strike * 100)

    prem_vel = vels.get("vel_5", 0)
    divergence = (prem_vel - u_vel * 50) if is_call else (prem_vel + u_vel * 50)

    # Time
    minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + et_minute))
    session_progress = max(0, min(1, (et_decimal - 9.5) / 6.5))
    elapsed = (now - timestamps[0]).astype("timedelta64[s]").astype(float) / 60

    # Prior retrace history
    n_prior = len(prior_retraces)
    avg_prior = np.mean(prior_retraces) if prior_retraces else 0
    max_prior = max(prior_retraces) if prior_retraces else 0

    return {
        "retrace_depth": retrace_depth,
        "gain_at_high": gain_at_high,
        "gain_at_retrace_low": gain_at_low,
        "pct_of_session_peak": pct_of_session_peak,
        "retrace_speed": retrace_speed,
        "retrace_ticks": retrace_ticks,
        "vel_3": vels["vel_3"],
        "vel_5": vels["vel_5"],
        "vel_10": vels["vel_10"],
        "prem_std": prem_std,
        "delta": abs(delta),
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "iv": iv,
        "delta_change": delta_change,
        "iv_change": iv_change,
        "gamma_delta_ratio": gdr,
        "volume": vol,
        "vol_ratio": vol_ratio,
        "vol_trend": vol_trend,
        "spread_pct": spread_pct,
        "spread_change": spread_change,
        "size_imbalance": size_imbalance,
        "u_move": u_move,
        "u_vel": u_vel,
        "u_retrace_pct": u_retrace,
        "moneyness": moneyness,
        "divergence": divergence,
        "minutes_to_close": minutes_to_close,
        "session_progress": session_progress,
        "elapsed_min": elapsed,
        "is_highvol": 1 if category == TickerCategory.HIGH_VOL else 0,
        "is_index": 1 if category == TickerCategory.INDEX else 0,
        "is_call": 1 if is_call else 0,
        "n_prior_retraces": n_prior,
        "avg_prior_retrace_depth": avg_prior,
        "max_prior_retrace_depth": max_prior,
    }


def process_contract(ticks_df, cinfo):
    """Find all retrace events in a contract and extract features."""
    premiums = ticks_df["premium"].values.astype(float)
    timestamps = pd.to_datetime(ticks_df["ts"], format="ISO8601").values
    n = len(premiums)

    session_peak = np.nanmax(premiums)
    entry_prem = premiums[0]
    if entry_prem <= 0:
        return []

    ticker = cinfo["underlying"]
    category = categorize_ticker(ticker)
    is_call = cinfo["option_type"] == "call"
    strike = cinfo["strike"]

    entry_u = 0
    for i in range(min(5, n)):
        u = safe_float(ticks_df["underlying_price"].iloc[i])
        if u > 0:
            entry_u = u
            break

    # Scan for retrace events
    running_high = premiums[0]
    running_high_idx = 0
    in_retrace = False
    retrace_start_idx = 0
    prior_retrace_depths = []
    events = []

    for i in range(1, n):
        p = premiums[i]
        if np.isnan(p) or p <= 0:
            continue

        if p > running_high:
            # New high — check if we were in a retrace
            if in_retrace:
                # TEMPORARY retrace — it recovered
                retrace_window = premiums[retrace_start_idx:i]
                retrace_low_idx = retrace_start_idx + np.nanargmin(retrace_window)
                retrace_depth = (running_high - premiums[retrace_low_idx]) / running_high * 100

                gain_at_high = (running_high - entry_prem) / entry_prem * 100

                if retrace_depth >= RETRACE_THRESHOLD and gain_at_high >= MIN_GAIN_AT_HIGH:
                    feat = extract_retrace_features(
                        ticks_df, premiums, timestamps, retrace_low_idx,
                        running_high, running_high_idx, entry_prem, entry_u,
                        ticker, category, is_call, strike,
                        prior_retrace_depths.copy()
                    )
                    if feat:
                        feat["is_final"] = 0  # recovered
                        events.append(feat)

                prior_retrace_depths.append(retrace_depth)
                in_retrace = False

            running_high = p
            running_high_idx = i

        elif not in_retrace:
            drop = (running_high - p) / running_high * 100
            if drop >= RETRACE_THRESHOLD:
                in_retrace = True
                retrace_start_idx = i

    # End of data: if in retrace from the session peak, this is FINAL
    if in_retrace and running_high_idx == np.nanargmax(premiums):
        retrace_window = premiums[retrace_start_idx:]
        retrace_low_idx = retrace_start_idx + np.nanargmin(retrace_window)
        retrace_depth = (running_high - premiums[retrace_low_idx]) / running_high * 100
        gain_at_high = (running_high - entry_prem) / entry_prem * 100

        if retrace_depth >= RETRACE_THRESHOLD and gain_at_high >= MIN_GAIN_AT_HIGH:
            feat = extract_retrace_features(
                ticks_df, premiums, timestamps, retrace_low_idx,
                running_high, running_high_idx, entry_prem, entry_u,
                ticker, category, is_call, strike,
                prior_retrace_depths.copy()
            )
            if feat:
                feat["is_final"] = 1  # final peak
                events.append(feat)

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-contracts", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{'=' * 120}")
    print("RETRACE CLASSIFIER: Will this pullback recover or is it the final peak?")
    print(f"  Retrace threshold: {RETRACE_THRESHOLD}% drop from local high")
    print(f"  Min gain at high: {MIN_GAIN_AT_HIGH}%")
    print(f"  Tickers: {', '.join(SIGNAL_TICKERS)}")
    print(f"{'=' * 120}\n")

    conn = sqlite3.connect(HARVESTER_DB, timeout=60)
    ticker_list = ",".join(f"'{t}'" for t in SIGNAL_TICKERS)

    query = f"""
        SELECT s.contract_ticker, c.underlying, c.option_type, c.strike,
               COUNT(*) as ticks
        FROM harvest_snapshots s
        JOIN harvest_contracts c ON s.contract_ticker = c.contract_ticker
        WHERE c.underlying IN ({ticker_list})
          AND s.midpoint BETWEEN 0.3 AND 15.0
          AND s.delta IS NOT NULL AND s.delta != 0
        GROUP BY s.contract_ticker
        HAVING ticks >= 40
        ORDER BY s.contract_ticker
    """
    if args.max_contracts:
        query += f" LIMIT {args.max_contracts}"

    contracts = pd.read_sql_query(query, conn)
    print(f"  Found {len(contracts):,} contracts\n")

    # Extract retrace events
    all_events = []
    trade_id = 0
    t0 = time.time()

    for ci, (_, cinfo) in enumerate(contracts.iterrows()):
        if (ci + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            remaining = (len(contracts) - ci - 1) / rate
            n_temp = sum(1 for e in all_events if e["is_final"] == 0)
            n_final = sum(1 for e in all_events if e["is_final"] == 1)
            print(f"    {ci+1:,}/{len(contracts):,} | "
                  f"{len(all_events):,} events ({n_temp} temp, {n_final} final) | "
                  f"{elapsed:.0f}s, ~{remaining:.0f}s left")

        ticks = pd.read_sql_query("""
            SELECT captured_at as ts, midpoint as premium, underlying_price,
                   bid, ask, bid_size, ask_size,
                   implied_volatility as iv, delta, gamma, theta, vega,
                   day_volume as volume, open_interest
            FROM harvest_snapshots WHERE contract_ticker = ? AND midpoint > 0.01
            ORDER BY captured_at
        """, conn, params=[cinfo["contract_ticker"]])

        if len(ticks) < 40:
            continue

        events = process_contract(ticks, cinfo)
        for e in events:
            e["trade_id"] = trade_id
        all_events.extend(events)
        trade_id += 1

    conn.close()

    elapsed = time.time() - t0
    print(f"\n  Extraction complete: {elapsed:.0f}s")

    if not all_events:
        print("No retrace events found!")
        return

    df = pd.DataFrame(all_events)
    n_temp = (df["is_final"] == 0).sum()
    n_final = (df["is_final"] == 1).sum()
    print(f"  Total retrace events: {len(df):,}")
    print(f"    Temporary (recovered): {n_temp:,} ({n_temp/len(df)*100:.0f}%)")
    print(f"    Final (peak done):     {n_final:,} ({n_final/len(df)*100:.0f}%)")
    print(f"  Unique contracts: {df['trade_id'].nunique()}")

    # Train classifier
    X = np.nan_to_num(df[FEATURE_COLS].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["is_final"].values
    groups = df["trade_id"].values

    print(f"\n{'=' * 120}")
    print("TRAINING: Is this retrace final or temporary?")
    print(f"{'=' * 120}\n")

    gkf = GroupKFold(n_splits=5)
    aucs, accs = [], []
    all_preds = np.full(len(y), np.nan)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        pos_w = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        dtrain = lgb.Dataset(X[tr], y[tr], feature_name=FEATURE_COLS)
        dval = lgb.Dataset(X[va], y[va], feature_name=FEATURE_COLS, reference=dtrain)

        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.7,
            "bagging_fraction": 0.7, "bagging_freq": 5,
            "scale_pos_weight": pos_w, "reg_alpha": 0.1, "verbose": -1,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X[va])
        all_preds[va] = preds
        auc = roc_auc_score(y[va], preds)
        acc = accuracy_score(y[va], (preds > 0.5).astype(int))
        aucs.append(auc)
        accs.append(acc)
        n_groups = len(set(groups[va]))
        print(f"  Fold {fold+1}: AUC={auc:.4f} Acc={acc:.3f} ({n_groups} contracts)")

    print(f"\n  Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"  Mean Acc: {np.mean(accs):.3f}")

    # Final model
    dtrain_full = lgb.Dataset(X, y, feature_name=FEATURE_COLS)
    pos_w = (y == 0).sum() / max(1, (y == 1).sum())
    params["scale_pos_weight"] = pos_w
    final_model = lgb.train(params, dtrain_full, num_boost_round=400)

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 20 features:")
    for feat, imp in feat_imp[:20]:
        print(f"    {feat:<30} {imp:>12,.0f}")

    # Accuracy analysis
    print(f"\n{'=' * 120}")
    print("ACCURACY ANALYSIS")
    print(f"{'=' * 120}")

    df["pred"] = all_preds

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        predicted_final = df[df["pred"] >= threshold]
        predicted_temp = df[df["pred"] < threshold]

        if len(predicted_final) > 0:
            true_final = (predicted_final["is_final"] == 1).sum()
            false_final = (predicted_final["is_final"] == 0).sum()
            precision = true_final / len(predicted_final) * 100
        else:
            precision = 0
            true_final = false_final = 0

        if len(predicted_temp) > 0:
            true_temp = (predicted_temp["is_final"] == 0).sum()
            false_temp = (predicted_temp["is_final"] == 1).sum()
            neg_precision = true_temp / len(predicted_temp) * 100
        else:
            neg_precision = 0
            true_temp = false_temp = 0

        print(f"\n  Threshold {threshold:.1f}:")
        print(f"    Says FINAL: {len(predicted_final):>5} "
              f"({true_final} correct, {false_final} wrong) "
              f"→ precision {precision:.0f}%")
        print(f"    Says TEMP:  {len(predicted_temp):>5} "
              f"({true_temp} correct, {false_temp} wrong) "
              f"→ precision {neg_precision:.0f}%")

        # What are the characteristics of correct final calls?
        if true_final > 0:
            correct_final = predicted_final[predicted_final["is_final"] == 1]
            print(f"    Correct FINAL calls: avg retrace {correct_final['retrace_depth'].mean():.0f}%, "
                  f"avg gain at high +{correct_final['gain_at_high'].mean():.0f}%")

    # Per-category breakdown
    print(f"\n{'=' * 120}")
    print("PER-CATEGORY ACCURACY")
    print(f"{'=' * 120}")
    for cat_col, cat_name in [("is_highvol", "HIGH_VOL"), ("is_index", "INDEX")]:
        mask = df[cat_col] == 1
        if mask.sum() < 50:
            continue
        sub = df[mask]
        auc = roc_auc_score(sub["is_final"], sub["pred"])
        print(f"\n  {cat_name}: {len(sub)} events, AUC={auc:.4f}")

    std_mask = (df["is_highvol"] == 0) & (df["is_index"] == 0)
    if std_mask.sum() >= 50:
        sub = df[std_mask]
        auc = roc_auc_score(sub["is_final"], sub["pred"])
        print(f"  STANDARD: {len(sub)} events, AUC={auc:.4f}")

    # Save
    model_path = MODEL_DIR / "retrace_classifier.txt"
    feat_path = MODEL_DIR / "retrace_classifier_features.txt"
    final_model.save_model(str(model_path))
    with open(feat_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))

    print(f"\n{'=' * 120}")
    print("MODEL SAVED")
    print(f"{'=' * 120}")
    print(f"  Classifier: {model_path}")
    print(f"  Features:   {feat_path}")


if __name__ == "__main__":
    main()
