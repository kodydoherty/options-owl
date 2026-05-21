"""Compare generic vs per-ticker ML models for each ticker.

For each ticker that has a per-ticker model, runs the test set through BOTH
the per-ticker model and the generic model, then shows which one wins.

Usage:
    python scripts/compare_generic_vs_ticker.py
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


def load_both_models(ticker):
    """Load both per-ticker and generic models. Returns (ticker_clf, ticker_reg, generic_clf, generic_reg)."""
    ticker_clf_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_clf.lgb")
    ticker_reg_path = os.path.join(MODELS_DIR, f"{ticker.lower()}_reg.lgb")
    generic_clf_path = os.path.join(MODELS_DIR, "generic_clf.lgb")
    generic_reg_path = os.path.join(MODELS_DIR, "generic_reg.lgb")

    ticker_clf = ticker_reg = None
    if os.path.exists(ticker_clf_path):
        ticker_clf = lgb.Booster(model_file=ticker_clf_path)
        ticker_reg = lgb.Booster(model_file=ticker_reg_path)

    generic_clf = generic_reg = None
    if os.path.exists(generic_clf_path):
        generic_clf = lgb.Booster(model_file=generic_clf_path)
        generic_reg = lgb.Booster(model_file=generic_reg_path)

    return ticker_clf, ticker_reg, generic_clf, generic_reg


def simulate_ml(prices, pnls, clf, reg, features_df, feat_cols):
    """Simulate ML exit. Returns exit PnL."""
    n = len(prices)
    eod_pnl = pnls[-1]
    X = features_df[feat_cols].values

    sell_probs = clf.predict(X, num_iteration=clf.best_iteration)
    reg_preds = reg.predict(X, num_iteration=reg.best_iteration)

    for i in range(len(X)):
        if sell_probs[i] > SELL_THRESHOLD and reg_preds[i] < 2.0:
            return pnls[i]
        if i >= 10 and reg_preds[i] < -10.0:
            return pnls[i]
    return eod_pnl


def main():
    print("=" * 70)
    print("  Generic vs Per-Ticker ML Model Comparison")
    print("=" * 70)

    conn = get_db_connection()

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_clf.lgb")]
    tickers = sorted(set(f.replace("_clf.lgb", "").upper() for f in model_files if f != "generic_clf.lgb"))

    print(f"  Testing {len(tickers)} tickers\n")

    results = []

    for ticker in tickers:
        ticker_clf, ticker_reg, generic_clf, generic_reg = load_both_models(ticker)
        if ticker_clf is None or generic_clf is None:
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

        test_start = int(len(days) * 0.85)
        test_days = days[test_start:]

        ticker_pnls = []
        generic_pnls = []
        hold_pnls = []

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

                # Per-ticker model
                t_pnl = simulate_ml(prices, pnls, ticker_clf, ticker_reg, df, PER_TICKER_FEATURES)
                ticker_pnls.append(t_pnl)

                # Generic model
                g_pnl = simulate_ml(prices, pnls, generic_clf, generic_reg, df, FEATURE_COLS)
                generic_pnls.append(g_pnl)

                # Hold baseline
                hold_pnls.append(pnls[-1])

        if len(ticker_pnls) < 2:
            continue

        t_arr = np.array(ticker_pnls)
        g_arr = np.array(generic_pnls)
        h_arr = np.array(hold_pnls)

        t_avg = np.mean(t_arr)
        g_avg = np.mean(g_arr)
        h_avg = np.mean(h_arr)
        t_wr = np.sum(t_arr > 0) / len(t_arr) * 100
        g_wr = np.sum(g_arr > 0) / len(g_arr) * 100

        winner = "GENERIC" if g_avg > t_avg else "TICKER"
        diff = g_avg - t_avg

        results.append({
            "ticker": ticker,
            "trades": len(ticker_pnls),
            "ticker_avg": t_avg,
            "ticker_wr": t_wr,
            "ticker_total": np.sum(t_arr),
            "generic_avg": g_avg,
            "generic_wr": g_wr,
            "generic_total": np.sum(g_arr),
            "hold_avg": h_avg,
            "diff": diff,
            "winner": winner,
        })

        arrow = "◀ GENERIC" if winner == "GENERIC" else "TICKER ▶"
        print(f"  {ticker:<6} | {len(ticker_pnls):>3} trades | "
              f"Ticker: {t_avg:>+7.1f}% WR:{t_wr:>4.0f}% | "
              f"Generic: {g_avg:>+7.1f}% WR:{g_wr:>4.0f}% | "
              f"Δ {diff:>+6.1f}% {arrow}")

    conn.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION — Use generic model for these tickers:")
    print(f"{'='*70}")

    generic_wins = [r for r in results if r["winner"] == "GENERIC"]
    ticker_wins = [r for r in results if r["winner"] == "TICKER"]

    if generic_wins:
        print(f"\n  Switch to GENERIC ({len(generic_wins)} tickers):")
        for r in sorted(generic_wins, key=lambda x: x["diff"], reverse=True):
            print(f"    {r['ticker']:<6} generic beats ticker by {r['diff']:>+6.1f}% avg PnL")

    if ticker_wins:
        print(f"\n  Keep PER-TICKER ({len(ticker_wins)} tickers):")
        for r in sorted(ticker_wins, key=lambda x: x["ticker_avg"], reverse=True):
            print(f"    {r['ticker']:<6} ticker beats generic by {-r['diff']:>+6.1f}% avg PnL")

    # Show what PER_TICKER_MODELS should be
    keep_tickers = sorted(r["ticker"] for r in ticker_wins)
    print(f"\n  Recommended PER_TICKER_MODELS set:")
    print(f"    {{{', '.join(repr(t) for t in keep_tickers)}}}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
