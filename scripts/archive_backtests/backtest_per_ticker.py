"""Backtest per-ticker ML models on their test sets.

Simulates trades using each ticker's dedicated ML model vs baseline strategies
(hold-to-EOD, Vinny phase trail, simple trailing stops).

Shows per-ticker and aggregate results.

Usage:
    python scripts/backtest_per_ticker.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb

from scripts.ml_sell_model import (
    DB_PATH, MIN_OPTION_BARS, FEATURE_COLS,
    extract_features_for_trade, find_deadline_idx, find_entry_idx,
    get_db_connection, load_option_bars, load_underlying_bars,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
MODELS_DIR = os.path.join(PROJECT_DIR, "journal", "models")

SELL_THRESHOLD = 0.4

PER_TICKER_FEATURES = [f for f in FEATURE_COLS if f != "ticker_encoded"]


def load_model(ticker):
    """Load per-ticker model, fallback to generic."""
    clf_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_clf.lgb")
    reg_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_reg.lgb")

    if not os.path.exists(clf_path):
        clf_path = os.path.join(MODELS_DIR, "generic_clf.lgb")
        reg_path = os.path.join(MODELS_DIR, "generic_reg.lgb")
        if not os.path.exists(clf_path):
            return None, None, "none"
        return lgb.Booster(model_file=clf_path), lgb.Booster(model_file=reg_path), "generic"

    return lgb.Booster(model_file=clf_path), lgb.Booster(model_file=reg_path), ticker.lower()


def simulate_trade(prices, pnls, clf, reg, features_df, model_name):
    """Simulate different exit strategies on one trade. Returns dict of strategy -> exit_pnl."""
    n = len(prices)
    entry_price = prices[0]
    if entry_price <= 0 or n < 2:
        return None

    eod_pnl = pnls[-1]
    results = {}

    # 1) Hold to EOD
    results["hold_to_eod"] = eod_pnl

    # 2) ML per-ticker model
    if clf is not None and features_df is not None:
        use_generic = model_name == "generic"
        feat_cols = FEATURE_COLS if use_generic else PER_TICKER_FEATURES
        X = features_df[feat_cols].values

        sell_probs = clf.predict(X, num_iteration=clf.best_iteration)
        reg_preds = reg.predict(X, num_iteration=reg.best_iteration)

        ml_pnl = eod_pnl
        for i in range(len(X)):
            if sell_probs[i] > SELL_THRESHOLD and reg_preds[i] < 2.0:
                ml_pnl = pnls[i]
                break
            if i >= 10 and reg_preds[i] < -10.0:
                ml_pnl = pnls[i]
                break
        results["ml_per_ticker"] = ml_pnl

    # 3) Vinny phase trail (25% -> tighten on gains)
    vinny_pnl = eod_pnl
    peak = entry_price
    phase_trail = 25.0
    for i in range(n):
        p = prices[i]
        if p > peak:
            peak = p
            gain = (peak - entry_price) / entry_price * 100
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
    results["vinny_trail"] = vinny_pnl

    # 4) Simple 25% trailing stop (10 min grace)
    trail_pnl = eod_pnl
    peak = entry_price
    for i in range(n):
        p = prices[i]
        if p > peak: peak = p
        if i >= 10 and peak > 0:
            drop = (peak - p) / peak * 100
            if drop >= 25.0:
                trail_pnl = pnls[i]
                break
    results["trail_25pct"] = trail_pnl

    # 5) Exit at 90 min
    results["exit_90min"] = pnls[min(90, n-1)]

    return results


def main():
    print("=" * 70)
    print("  OptionsOwl — Per-Ticker ML Backtest")
    print("=" * 70)
    print(f"  Models: {MODELS_DIR}")
    print()

    conn = get_db_connection()

    # Get all tickers with models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_clf.lgb")]
    tickers = sorted(set(f.replace("_clf.lgb", "").upper() for f in model_files if f != "generic_clf.lgb"))

    print(f"  Testing {len(tickers)} per-ticker models + generic fallback")
    print()

    # Use last 15% of each ticker's data as test set (same as training)
    all_results = {s: [] for s in ["hold_to_eod", "ml_per_ticker", "vinny_trail", "trail_25pct", "exit_90min"]}
    ticker_summaries = []

    for ticker in tickers:
        clf, reg, model_name = load_model(ticker)
        if clf is None:
            continue

        days = conn.execute("""
            SELECT ticker, date, open_price, close_price, high_price, low_price,
                   atm_call_ticker, atm_put_ticker, atm_strike,
                   call_bars, put_bars, underlying_bars
            FROM trading_days
            WHERE ticker = ? AND call_bars >= ? AND put_bars >= ?
            ORDER BY date
        """, (ticker, MIN_OPTION_BARS, MIN_OPTION_BARS)).fetchall()
        days = [dict(d) for d in days]

        if len(days) < 10:
            continue

        # Test set = last 15% of dates
        test_start = int(len(days) * 0.85)
        test_days = days[test_start:]

        ticker_results = {s: [] for s in all_results}
        trades_tested = 0

        for day in test_days:
            date = day["date"]
            u_bars = load_underlying_bars(conn, ticker, date)

            for direction, contract_key in [("call", "atm_call_ticker"),
                                            ("put", "atm_put_ticker")]:
                contract_ticker = day[contract_key]
                if not contract_ticker:
                    continue

                o_bars = load_option_bars(conn, contract_ticker)
                if not o_bars or len(o_bars) < MIN_OPTION_BARS:
                    continue

                entry_idx = find_entry_idx(o_bars)
                deadline_idx = find_deadline_idx(o_bars)
                if entry_idx is None or deadline_idx is None:
                    continue

                is_call = direction == "call"
                rows = extract_features_for_trade(
                    o_bars, u_bars, entry_idx, deadline_idx, 0, is_call
                )
                if rows is None or len(rows) < 10:
                    continue

                df = pd.DataFrame(rows)
                prices = df["_price"].values
                entry_price = prices[0]
                if entry_price <= 0:
                    continue
                pnls = df["_current_pnl"].values

                res = simulate_trade(prices, pnls, clf, reg, df, model_name)
                if res is None:
                    continue

                for s, v in res.items():
                    ticker_results[s].append(v)
                    all_results[s].append(v)
                trades_tested += 1

        if trades_tested < 2:
            continue

        # Per-ticker summary
        ml_pnls = np.array(ticker_results["ml_per_ticker"])
        hold_pnls = np.array(ticker_results["hold_to_eod"])
        vinny_pnls = np.array(ticker_results["vinny_trail"])

        ml_wr = np.sum(ml_pnls > 0) / len(ml_pnls) * 100 if len(ml_pnls) > 0 else 0
        hold_wr = np.sum(hold_pnls > 0) / len(hold_pnls) * 100 if len(hold_pnls) > 0 else 0

        summary = {
            "ticker": ticker,
            "trades": trades_tested,
            "ml_avg": np.mean(ml_pnls),
            "ml_total": np.sum(ml_pnls),
            "ml_wr": ml_wr,
            "hold_avg": np.mean(hold_pnls),
            "hold_total": np.sum(hold_pnls),
            "vinny_avg": np.mean(vinny_pnls),
            "ml_vs_hold": np.mean(ml_pnls) - np.mean(hold_pnls),
            "ml_vs_vinny": np.mean(ml_pnls) - np.mean(vinny_pnls),
        }
        ticker_summaries.append(summary)

        arrow = "▲" if summary["ml_vs_hold"] > 0 else "▼"
        print(f"  {ticker:<6} | {trades_tested:>3} trades | ML avg: {summary['ml_avg']:>+7.1f}% "
              f"WR: {ml_wr:>4.0f}% | Hold: {summary['hold_avg']:>+7.1f}% | "
              f"Vinny: {summary['vinny_avg']:>+7.1f}% | ML vs Hold: {summary['ml_vs_hold']:>+5.1f}% {arrow}")

    conn.close()

    # Aggregate results
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({sum(s['trades'] for s in ticker_summaries)} trades)")
    print(f"{'='*70}")

    print(f"\n  {'Strategy':<20} {'Trades':>6} {'Total PnL%':>12} {'Avg PnL%':>10} "
          f"{'Win Rate':>10} {'Avg Win%':>10} {'Avg Loss%':>10}")
    print(f"  {'-'*80}")

    for name in ["hold_to_eod", "ml_per_ticker", "vinny_trail", "trail_25pct", "exit_90min"]:
        pnls = np.array(all_results[name])
        n = len(pnls)
        if n == 0:
            continue
        total = np.sum(pnls)
        avg = np.mean(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / n * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        marker = " ★" if name == "ml_per_ticker" else ""
        print(f"  {name:<20} {n:>6} {total:>+11.1f}% {avg:>+9.2f}% "
              f"{wr:>9.1f}% {avg_win:>+9.2f}% {avg_loss:>+9.2f}%{marker}")

    # Per-ticker leaderboard
    if ticker_summaries:
        print(f"\n  Per-Ticker ML Model Leaderboard (by avg PnL):")
        print(f"  {'Ticker':<8} {'Trades':>6} {'ML Avg%':>8} {'ML WR%':>7} "
              f"{'vs Hold':>8} {'vs Vinny':>9}")
        print(f"  {'-'*52}")
        for s in sorted(ticker_summaries, key=lambda x: x["ml_avg"], reverse=True):
            print(f"  {s['ticker']:<8} {s['trades']:>6} {s['ml_avg']:>+7.1f}% {s['ml_wr']:>6.0f}% "
                  f"{s['ml_vs_hold']:>+7.1f}% {s['ml_vs_vinny']:>+8.1f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
