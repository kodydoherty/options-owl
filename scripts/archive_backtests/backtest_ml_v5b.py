"""Backtest ML entry filter + v5b exit logic on all signals.

Compares 3 strategies side by side:
  1. v5b (current production — all signals, v5b exits)
  2. ML+v5b (ML entry filter rejects low-confidence signals, v5b exits on rest)
  3. ML+v5b+regime (regime classifier adjusts trail widths)

Shows daily P&L table for each.

Usage:
    python scripts/backtest_ml_v5b.py
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
SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
MODELS_DIR = str(PROJECT_DIR / "journal" / "models")

PORTFOLIO = 8000
SLIPPAGE_PCT = 15


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 70
        ORDER BY created_at
    """).fetchall()
    signals = []
    for r in rows:
        sig = dict(r)
        sig["premium"] = sig["atm_premium"] or sig["otm_premium"]
        sent = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        sig["option_type"] = "put" if sent in ("bearish", "put") else "call"
        if sig["premium"] and sig["premium"] > 0 and sig["strike"]:
            signals.append(sig)
    conn.close()
    return signals


def build_contract_ticker(ticker, expiry, strike, option_type):
    if not expiry:
        return ""
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        return ""
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type.lower() in ("call", "bullish", "c") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def load_ticks(harvester_conn, signal):
    ticker = signal["ticker"]
    strike = signal["strike"]
    created_at = signal["created_at"]
    option_type = signal["option_type"]
    sig_date = created_at[:10]

    sig_dt = datetime.strptime(sig_date, "%Y-%m-%d").date()
    candidates = [sig_dt]
    for delta in range(1, 6):
        d = sig_dt + timedelta(days=delta)
        if d.weekday() < 5:
            candidates.append(d)
            if len(candidates) >= 4:
                break

    for exp_date in candidates:
        expiry = exp_date.strftime("%Y-%m-%d")
        ct = build_contract_ticker(ticker, expiry, strike, option_type)
        if not ct:
            continue
        rows = harvester_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()
        if rows and len(rows) >= 10:
            signal["_dte"] = (exp_date - sig_dt).days
            break
    else:
        return None

    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega", "volume"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def compute_ml_entry_features(df, entry_idx, signal):
    """Compute features matching entry_filter_v2 model format."""
    if entry_idx < 5:
        return None

    premium = df["premium"].iloc[entry_idx]
    u_price = df["underlying_price"].iloc[entry_idx]
    if premium <= 0 or not u_price or u_price <= 0:
        return None

    # Time features
    ts = df["ts"].iloc[entry_idx]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    et = ts - timedelta(hours=4)
    hour = et.hour
    minute = et.minute
    dow = et.weekday()
    mso = (hour - 9) * 60 + (minute - 30)

    # Underlying momentum
    u_prices = df["underlying_price"].iloc[:entry_idx + 1].values
    u_prices = u_prices[u_prices > 0]

    def u_mom(lookback):
        if len(u_prices) > lookback and u_prices[-lookback - 1] > 0:
            return (u_prices[-1] - u_prices[-lookback - 1]) / u_prices[-lookback - 1] * 100
        return 0.0

    # Underlying volatility
    lookback = min(30, len(u_prices) - 1)
    if lookback >= 5:
        rets = np.diff(u_prices[-lookback - 1:]) / u_prices[-lookback - 1:-1]
        rets = rets[np.isfinite(rets)]
        u_vol = float(np.std(rets) * 100) if len(rets) > 1 else 0.0
    else:
        u_vol = 0.0

    # Volume (use option volume as proxy)
    vol = df["volume"].iloc[entry_idx] or 0

    # Premium momentum
    premiums = df["premium"].iloc[:entry_idx + 1].values
    def p_mom(lb):
        if len(premiums) > lb and premiums[-lb - 1] > 0:
            return (premiums[-1] - premiums[-lb - 1]) / premiums[-lb - 1] * 100
        return 0.0

    # Day range position
    u_all = df["underlying_price"].iloc[:entry_idx + 1].values
    u_all = u_all[u_all > 0]
    if len(u_all) > 1:
        day_high = np.max(u_all)
        day_low = np.min(u_all)
        pos_in_range = (u_price - day_low) / (day_high - day_low) if day_high > day_low else 0.5
    else:
        pos_in_range = 0.5

    # VWAP deviation (approximate with mean underlying)
    u_mean = np.mean(u_all) if len(u_all) > 0 else u_price
    vwap_dev = (u_price - u_mean) / u_mean * 100 if u_mean > 0 else 0

    # Day open to now
    u_open = u_all[0] if len(u_all) > 0 else u_price
    day_open_pct = (u_price - u_open) / u_open * 100 if u_open > 0 else 0

    # Consecutive up/down
    consec_up, consec_down = 0, 0
    for j in range(len(u_all) - 1, 0, -1):
        if u_all[j] > u_all[j - 1]:
            consec_up += 1
        elif u_all[j] < u_all[j - 1]:
            consec_down += 1
        else:
            break
        if consec_up > 0 and consec_down > 0:
            break

    # Ticker encoding (match training)
    ALL_TICKERS = ["AAPL", "AMD", "AMZN", "BA", "COIN", "DIA", "GLD", "GOOGL",
                   "IWM", "JPM", "META", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
                   "PLTR", "QQQ", "SLV", "SMCI", "SPY", "TLT", "TSLA", "XLF", "XLK"]
    ticker = signal["ticker"]
    try:
        ticker_enc = ALL_TICKERS.index(ticker)
    except ValueError:
        ticker_enc = 0

    is_call = 1 if signal["option_type"].lower() in ("call", "bullish") else 0

    features = [
        hour,                           # hour
        minute,                         # minute
        dow,                            # day_of_week
        mso,                            # minutes_since_open
        is_call,                        # is_call
        u_mom(5),                       # underlying_mom_5m
        u_mom(10),                      # underlying_mom_10m
        u_mom(15),                      # underlying_mom_15m
        u_mom(30),                      # underlying_mom_30m
        u_vol,                          # underlying_vol_30m
        float(vol),                     # volume_avg_30m (proxy)
        1.0,                            # volume_current_vs_avg
        1.0,                            # volume_trend
        pos_in_range,                   # price_position_in_range
        u_vol * 2,                      # avg_bar_range_pct (proxy)
        vwap_dev,                       # vwap_deviation_pct
        consec_up,                      # consec_up_bars
        consec_down,                    # consec_down_bars
        day_open_pct,                   # day_open_to_now_pct
        premium,                        # entry_premium
        premium / u_price * 100,        # premium_to_underlying_pct
        p_mom(5),                       # premium_mom_5m
        p_mom(10),                      # premium_mom_10m
        0.0,                            # option_bar_range_pct (no bar data)
        float(vol),                     # option_volume
        1.0,                            # option_vol_vs_avg
        0.0,                            # option_num_trades
        ticker_enc,                     # ticker_encoded
    ]
    return np.array([features])


def v5b_simulate(df, entry_idx, entry_premium, contracts, direction, dte=0):
    """Simulate v5b exit logic — DTE-aware version.

    v5b params: checkpoint_drop=30, checkpoint_u=0.5,
                tight_stop=35, wide_stop=50, backstop=65
    DTE adjustments:
      - checkpoint_cut: disabled for DTE>0 (noise kills multi-day trades)
      - graduated stop: widened 1.5x for DTE>0
      - theta bleed: disabled for DTE>0
      - scalp trail: requires underlying_against for DTE>0 (not just !confirms)
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    is_call = direction in ("bullish", "call", "long")
    peak = entry_premium
    entry_underlying = None

    # DTE-aware thresholds
    checkpoint_drop = 30
    checkpoint_u = 0.5
    tight_stop = 35 if dte == 0 else 52    # 35 * 1.5 ≈ 52 for multi-day
    backstop = 65 if dte == 0 else 75       # wider for multi-day

    for idx in range(entry_idx + 1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        if premium > peak:
            peak = premium

        elapsed = (df["ts"].iloc[idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60
        gain_pct = (premium - entry_premium) / entry_premium * 100
        drop_entry = max(0, (entry_premium - premium) / entry_premium * 100)
        drop_peak = (peak - premium) / peak * 100 if peak > 0 else 0
        peak_gain = (peak - entry_premium) / entry_premium * 100

        # Underlying
        underlying = df["underlying_price"].iloc[idx] or 0
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying

        u_move = 0.0
        underlying_against = False
        underlying_confirms = False
        has_underlying = False
        if entry_underlying and underlying and underlying > 0:
            has_underlying = True
            u_move = (underlying - entry_underlying) / entry_underlying * 100
            if is_call:
                underlying_against = u_move < -checkpoint_u
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > checkpoint_u
                underlying_confirms = u_move < -0.2

        # EOD cutoff (0DTE only)
        if dte == 0:
            ts = df["ts"].iloc[idx]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts - timedelta(hours=4)
            if et.hour >= 15 and et.minute >= 45:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "eod_cutoff", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        if elapsed < 5:
            continue

        # SCALP TRAIL: peaked 20%+, faded to <60% of peak gain
        # 0DTE: exit if underlying NOT confirming
        # Multi-day: only exit if underlying actively AGAINST (more patient)
        if peak_gain >= 20 and gain_pct > 0 and gain_pct < peak_gain * 0.6:
            if dte == 0 and not underlying_confirms:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "scalp_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
            elif dte > 0 and underlying_against:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "scalp_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # CHECKPOINT: premium down 30%+ AND underlying against 0.5%+
        # DISABLED for DTE>0 — multi-day trades recover from temporary dips
        if dte == 0 and drop_entry >= checkpoint_drop:
            if has_underlying and underlying_against:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "checkpoint_cut", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # GRADUATED STOP: tight when underlying confirms against, backstop otherwise
        if underlying_against:
            if drop_entry >= tight_stop:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "confirmed_stop", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
        else:
            if drop_entry >= backstop:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "hard_stop", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # SOFT TRAIL (15-50% band): protect 50% of gains
        if 15 <= peak_gain < 50:
            floor = entry_premium + (peak - entry_premium) * 0.50
            if premium <= floor:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "soft_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # ADAPTIVE TRAIL
        if peak_gain >= 400:
            if drop_peak >= 30:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "adaptive_moonshot", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
        elif peak_gain >= 150:
            if drop_peak >= 45:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "adaptive_runner", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
        elif peak_gain >= 40:
            if drop_peak >= 40:
                pnl = (premium - entry_premium) * contracts * 100
                return {"pnl": pnl, "reason": "adaptive_active", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # THETA BLEED (0DTE only — multi-day theta is negligible)
        if dte == 0 and elapsed >= 120 and drop_entry >= 30:
            pnl = (premium - entry_premium) * contracts * 100
            return {"pnl": pnl, "reason": "theta_bleed", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

    # End of data
    last_idx = len(df) - 1
    exit_prem = df["premium"].iloc[last_idx]
    elapsed = (df["ts"].iloc[last_idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60
    pnl = (exit_prem - entry_premium) * contracts * 100
    peak_g = (peak - entry_premium) / entry_premium * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": exit_prem, "peak_gain": peak_g}


def main():
    import lightgbm as lgb

    # Load ML models
    clf_path = os.path.join(MODELS_DIR, "entry_filter_v2.lgb")
    if not os.path.exists(clf_path):
        print(f"ERROR: Entry filter model not found at {clf_path}")
        print("Run: python scripts/train_ml_v2.py --model entry")
        sys.exit(1)

    entry_clf = lgb.Booster(model_file=clf_path)
    print(f"Loaded entry filter: {entry_clf.num_feature()} features")

    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Results for 3 strategies
    results = {"v5b": [], "ml_v5b": [], "ml_v5b_strict": []}
    ml_rejected = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        entry_premium = sig["premium"]

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)

        # Sizing — match backtest_compare.py exactly (fixed by score tier)
        if score >= 95:
            contracts = 5
        elif score >= 90:
            contracts = 4
        elif score >= 85:
            contracts = 3
        else:
            contracts = 1

        entry_idx = 0

        # Use actual market price from harvester (first ask), not signal premium + slippage
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = entry_premium

        # --- v5b: trade everything ---
        v5b_result = v5b_simulate(df, entry_idx, adj_entry, contracts, direction, dte)
        v5b_result.update({"ticker": ticker, "day": day, "score": score,
                           "entry": adj_entry, "contracts": contracts,
                           "direction": direction, "dte": dte})
        results["v5b"].append(v5b_result)

        # --- ML entry filter ---
        # Compute features from harvester data at entry point
        ml_features = compute_ml_entry_features(df, min(5, len(df) - 1), sig)
        if ml_features is not None:
            win_prob = float(entry_clf.predict(ml_features)[0])
        else:
            # Fallback: use basic features when can't compute from ticks
            u_price = df["underlying_price"].iloc[entry_idx]
            if u_price and u_price > 0:
                prem_pct = entry_premium / u_price * 100
            else:
                prem_pct = entry_premium / 200 * 100  # approximate

            ts = df["ts"].iloc[entry_idx]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts - timedelta(hours=4)
            is_call = 1 if direction in ("bullish", "call") else 0

            ALL_TICKERS = ["AAPL", "AMD", "AMZN", "BA", "COIN", "DIA", "GLD", "GOOGL",
                           "IWM", "JPM", "META", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
                           "PLTR", "QQQ", "SLV", "SMCI", "SPY", "TLT", "TSLA", "XLF", "XLK"]
            try:
                ticker_enc = ALL_TICKERS.index(ticker)
            except ValueError:
                ticker_enc = 0

            ml_features = np.array([[
                et.hour, et.minute, et.weekday(),
                (et.hour - 9) * 60 + (et.minute - 30), is_call,
                0, 0, 0, 0, 0.5, 1000, 1.0, 1.0, 0.5, 0.3, 0, 0, 0, 0,
                entry_premium, prem_pct, 0, 0, 0, 0, 1.0, 0, ticker_enc,
            ]])
            win_prob = float(entry_clf.predict(ml_features)[0])

        # ML+v5b (threshold 0.40 — moderate filter)
        if win_prob >= 0.40:
            ml_result = v5b_simulate(df, entry_idx, adj_entry, contracts, direction, dte)
            ml_result.update({"ticker": ticker, "day": day, "score": score,
                              "entry": adj_entry, "contracts": contracts,
                              "direction": direction, "dte": dte,
                              "win_prob": win_prob})
            results["ml_v5b"].append(ml_result)
        else:
            ml_rejected.append({
                "ticker": ticker, "day": day, "score": score,
                "win_prob": win_prob, "direction": direction,
                "would_pnl": v5b_result["pnl"],
            })

        # ML+v5b strict (threshold 0.55 — aggressive filter)
        if win_prob >= 0.55:
            strict_result = v5b_simulate(df, entry_idx, adj_entry, contracts, direction, dte)
            strict_result.update({"ticker": ticker, "day": day, "score": score,
                                  "entry": adj_entry, "contracts": contracts,
                                  "direction": direction, "dte": dte,
                                  "win_prob": win_prob})
            results["ml_v5b_strict"].append(strict_result)

    harvester_conn.close()

    # === PRINT RESULTS ===
    print(f"\n{'=' * 100}")
    print(f"ML + v5b BACKTEST — {len(signals)} signals, {no_data} no tick data")
    print(f"{'=' * 100}")

    # Overall summary
    print(f"\n{'Metric':<25} {'v5b (all)':>15} {'ML+v5b (0.40)':>15} {'ML+v5b (0.55)':>15}")
    print("-" * 72)

    for name in ["v5b", "ml_v5b", "ml_v5b_strict"]:
        res = results[name]
        if not res:
            continue
        pnls = [r["pnl"] for r in res]
        wins = sum(1 for p in pnls if p > 0)
        n = len(pnls)
        total = sum(pnls)
        avg_w = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_l = np.mean([p for p in pnls if p <= 0]) if (n - wins) > 0 else 0
        avg_h = np.mean([r["hold"] for r in res])

        if name == "v5b":
            label = "v5b (all)"
        elif name == "ml_v5b":
            label = "ML+v5b (0.40)"
        else:
            label = "ML+v5b (0.55)"

    # Print as table
    strats = ["v5b", "ml_v5b", "ml_v5b_strict"]
    labels = ["v5b (all)", "ML+v5b (0.40)", "ML+v5b (0.55)"]

    for metric in ["Trades", "Total P&L", "Win Rate", "Avg Win", "Avg Loss", "Avg Hold"]:
        row = f"{metric:<25}"
        for name in strats:
            res = results[name]
            pnls = [r["pnl"] for r in res]
            wins = sum(1 for p in pnls if p > 0)
            n = len(pnls)
            if metric == "Trades":
                row += f"{n:>15}"
            elif metric == "Total P&L":
                row += f"${sum(pnls):>+13,.0f}"
            elif metric == "Win Rate":
                wr = wins / n * 100 if n > 0 else 0
                row += f"{wr:>13.0f}% ({wins}W/{n - wins}L)"[:15].rjust(15)
            elif metric == "Avg Win":
                avg = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
                row += f"${avg:>+13,.0f}"
            elif metric == "Avg Loss":
                losses = [p for p in pnls if p <= 0]
                avg = np.mean(losses) if losses else 0
                row += f"${avg:>+13,.0f}"
            elif metric == "Avg Hold":
                avg = np.mean([r["hold"] for r in res])
                row += f"{avg:>13.0f}m"
        print(row)

    # Daily P&L table
    all_days = sorted(set(r["day"] for r in results["v5b"]))

    print(f"\n{'=' * 100}")
    print("DAILY P&L")
    print(f"{'=' * 100}")
    print(f"{'Day':<12} {'Sigs':>4}  {'v5b P&L':>10} {'v5b WR':>6}  "
          f"{'ML40 P&L':>10} {'ML40 WR':>6} {'ML40 N':>5}  "
          f"{'ML55 P&L':>10} {'ML55 WR':>6} {'ML55 N':>5}  {'Best':>6}")
    print("-" * 100)

    cum = {"v5b": 0, "ml_v5b": 0, "ml_v5b_strict": 0}
    day_wins = {"v5b": 0, "ml_v5b": 0, "ml_v5b_strict": 0}

    for day in all_days:
        v5b_day = [r for r in results["v5b"] if r["day"] == day]
        ml40_day = [r for r in results["ml_v5b"] if r["day"] == day]
        ml55_day = [r for r in results["ml_v5b_strict"] if r["day"] == day]

        v5b_pnl = sum(r["pnl"] for r in v5b_day)
        ml40_pnl = sum(r["pnl"] for r in ml40_day)
        ml55_pnl = sum(r["pnl"] for r in ml55_day)

        v5b_wr = sum(1 for r in v5b_day if r["pnl"] > 0) / len(v5b_day) * 100 if v5b_day else 0
        ml40_wr = sum(1 for r in ml40_day if r["pnl"] > 0) / len(ml40_day) * 100 if ml40_day else 0
        ml55_wr = sum(1 for r in ml55_day if r["pnl"] > 0) / len(ml55_day) * 100 if ml55_day else 0

        cum["v5b"] += v5b_pnl
        cum["ml_v5b"] += ml40_pnl
        cum["ml_v5b_strict"] += ml55_pnl

        if v5b_pnl > 0: day_wins["v5b"] += 1
        if ml40_pnl > 0: day_wins["ml_v5b"] += 1
        if ml55_pnl > 0: day_wins["ml_v5b_strict"] += 1

        best_pnl = max(v5b_pnl, ml40_pnl, ml55_pnl)
        best = "v5b" if best_pnl == v5b_pnl else ("ML40" if best_pnl == ml40_pnl else "ML55")

        print(f"{day:<12} {len(v5b_day):>4}  ${v5b_pnl:>+9,.0f} {v5b_wr:>5.0f}%  "
              f"${ml40_pnl:>+9,.0f} {ml40_wr:>5.0f}% {len(ml40_day):>5}  "
              f"${ml55_pnl:>+9,.0f} {ml55_wr:>5.0f}% {len(ml55_day):>5}  {best:>6}")

    print("-" * 100)
    print(f"{'CUMULATIVE':<12} {'':>4}  ${cum['v5b']:>+9,.0f} {'':>6}  "
          f"${cum['ml_v5b']:>+9,.0f} {'':>6} {'':>5}  "
          f"${cum['ml_v5b_strict']:>+9,.0f}")
    print(f"{'Days won':<12} {'':>4}  {day_wins['v5b']:>6}/{len(all_days)} {'':>6}  "
          f"{day_wins['ml_v5b']:>6}/{len(all_days)} {'':>6} {'':>5}  "
          f"{day_wins['ml_v5b_strict']:>6}/{len(all_days)}")

    # ML rejected signals analysis
    print(f"\n{'=' * 100}")
    print(f"ML REJECTED SIGNALS — {len(ml_rejected)} trades blocked by ML filter (prob < 0.40)")
    print(f"{'=' * 100}")

    if ml_rejected:
        rej_pnls = [r["would_pnl"] for r in ml_rejected]
        rej_wins = sum(1 for p in rej_pnls if p > 0)
        print(f"Would-be P&L if traded: ${sum(rej_pnls):>+,.0f}")
        print(f"Would-be WR: {rej_wins}/{len(rej_pnls)} ({rej_wins / len(rej_pnls) * 100:.0f}%)")
        print(f"Avg would-be P&L: ${np.mean(rej_pnls):>+,.0f}")
        print(f"\nRejected signals:")
        print(f"{'Ticker':<7} {'Day':<12} {'Score':>5} {'Prob':>5} {'Dir':<6} {'Would P&L':>10}")
        print("-" * 50)
        for r in sorted(ml_rejected, key=lambda x: x["would_pnl"]):
            print(f"{r['ticker']:<7} {r['day']:<12} {r['score']:>5} {r['win_prob']:>4.0%} "
                  f"{r['direction'][:5]:<6} ${r['would_pnl']:>+9,.0f}")

    # Gate breakdown for v5b
    print(f"\n{'=' * 100}")
    print("GATE FIRE BREAKDOWN — v5b")
    print(f"{'=' * 100}")
    gate_stats = {}
    for r in results["v5b"]:
        reason = r["reason"].split("(")[0]
        if reason not in gate_stats:
            gate_stats[reason] = {"fires": 0, "pnl": 0, "wins": 0}
        gate_stats[reason]["fires"] += 1
        gate_stats[reason]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            gate_stats[reason]["wins"] += 1

    print(f"{'Gate':<25} {'Fires':>6} {'%':>5} {'P&L':>10} {'WR':>6} {'AvgHold':>7}")
    print("-" * 65)
    n_total = len(results["v5b"])
    for gate, stats in sorted(gate_stats.items(), key=lambda x: -x[1]["fires"]):
        wr = stats["wins"] / stats["fires"] * 100 if stats["fires"] > 0 else 0
        pct = stats["fires"] / n_total * 100
        avg_hold_trades = [r["hold"] for r in results["v5b"] if r["reason"].startswith(gate)]
        avg_h = np.mean(avg_hold_trades) if avg_hold_trades else 0
        print(f"{gate:<25} {stats['fires']:>6} {pct:>4.0f}% ${stats['pnl']:>+9,.0f} {wr:>5.0f}% {avg_h:>6.0f}m")

    # Per-trade detail for biggest disagreements
    print(f"\n{'=' * 100}")
    print("TOP DISAGREEMENTS: ML rejected but would have been profitable")
    print(f"{'=' * 100}")

    profitable_rejected = [r for r in ml_rejected if r["would_pnl"] > 100]
    if profitable_rejected:
        print(f"{'Ticker':<7} {'Day':<12} {'Score':>5} {'Prob':>5} {'Would P&L':>10}")
        for r in sorted(profitable_rejected, key=lambda x: -x["would_pnl"]):
            print(f"{r['ticker']:<7} {r['day']:<12} {r['score']:>5} {r['win_prob']:>4.0%} "
                  f"${r['would_pnl']:>+9,.0f}")

    # Return on portfolio
    print(f"\n{'=' * 100}")
    print("PORTFOLIO RETURN")
    print(f"{'=' * 100}")
    for name, label in zip(strats, labels):
        total = sum(r["pnl"] for r in results[name])
        ret = total / PORTFOLIO * 100
        n = len(results[name])
        wr = sum(1 for r in results[name] if r["pnl"] > 0) / n * 100 if n > 0 else 0
        print(f"  {label:<20} ${total:>+9,.0f} ({ret:>+.1f}%)  WR={wr:.0f}%  {n} trades")


if __name__ == "__main__":
    main()
