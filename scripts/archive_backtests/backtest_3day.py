"""Backtest last 3 days using PG-exported data with new ML config.

Patches the gold standard backtest to use the PG export DB.
Run inside owlet-kody container.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Patch the DB path BEFORE importing
import scripts.backtest_gold_standard as gs
gs.THETADATA_DB = "/app/journal/pg_export.db"
# Also disable UW DB (not exported)
gs.UW_DB = "/dev/null"

from scripts.backtest_gold_standard import load_models, run_backtest, TICKERS, EXCLUDED_TICKERS
import json
import sqlite3


def main():
    print("=" * 80)
    print("3-DAY BACKTEST WITH NEW CONFIG (pattern=0.74, entry=0.80)")
    print("=" * 80)

    conn = sqlite3.connect(gs.THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1
    """).fetchall()]
    conn.close()

    if not all_dates:
        print("ERROR: No data found")
        sys.exit(1)

    start_date = all_dates[0]
    end_date = all_dates[-1]
    tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    print(f"Period: {start_date} to {end_date} ({len(all_dates)} days)")
    print(f"Tickers: {len(tickers)} — {', '.join(tickers)}")
    print(f"Config: pattern=0.74, entry=0.80, regime=0.19")

    # Load models
    print("\nLoading models...")
    pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model, signal_model = load_models(
        use_entry_filter=True, use_regime=True
    )

    # New config
    PATTERN_THRESHOLD = 0.74
    ENTRY_THRESHOLD = 0.80
    REGIME_THRESHOLD = 0.19

    r = run_backtest(
        pattern_model, pattern_meta, entry_model, entry_features,
        PATTERN_THRESHOLD, ENTRY_THRESHOLD, tickers, start_date, end_date,
        stop_model, regime_model, REGIME_THRESHOLD, signal_model
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Trades:        {r['trades']}")
    print(f"Win Rate:      {r['win_rate']:.1f}%")
    print(f"Total P&L:     ${r['total_pnl']:+,.0f}")
    print(f"Profit Factor: {r['profit_factor']:.2f}")
    print(f"Sharpe:        {r['sharpe']:.2f}")
    print(f"Max Drawdown:  {r['max_drawdown_pct']:.1f}%")
    print(f"Avg Win:       ${r['avg_win']:+,.0f}")
    print(f"Avg Loss:      ${r['avg_loss']:+,.0f}")
    avg_pnl = r['total_pnl'] / r['trades'] if r['trades'] > 0 else 0
    print(f"$/Trade:       ${avg_pnl:+,.0f}")

    # Per-day breakdown
    if r.get("trade_details"):
        print("\n" + "=" * 80)
        print("DAILY TRADE LOG")
        print("=" * 80)
        print(f"{'Day':<12} {'Ticker':<8} {'Min':<5} {'Contracts':<10} {'Entry':>7} {'Exit':>7} "
              f"{'P&L':>10} {'Hold':>5} {'Peak%':>6} {'Exit Reason':<25}")
        print("-" * 110)

        current_day = None
        day_pnl = 0
        for t in r["trade_details"]:
            if t["day"] != current_day:
                if current_day is not None:
                    print(f"{'':>50} Day total: ${day_pnl:+,.0f}")
                    print()
                current_day = t["day"]
                day_pnl = 0

            pnl = t["pnl"]
            day_pnl += pnl
            contracts = t.get("contracts", 1)
            entry = t.get("effective_entry", t.get("entry", 0))
            exit_prem = t.get("exit_prem", 0)
            hold = t.get("hold_min", 0)
            peak = t.get("peak_gain", 0)
            reason = t.get("reason", "unknown")
            pattern_c = t.get("pattern_conf", 0)

            win = "W" if pnl > 0 else "L"
            print(f"{t['day']:<12} {t['ticker']:<8} {t['minute']:<5} {contracts:<10} "
                  f"${entry:>6.2f} ${exit_prem:>6.2f} ${pnl:>+9,.0f} {hold:>4}m "
                  f"{peak:>+5.0f}% {reason:<25} [{win}] conf={pattern_c:.2f}")

        if current_day is not None:
            print(f"{'':>50} Day total: ${day_pnl:+,.0f}")

    # Also compare with old config (0.85/0.70)
    print("\n" + "=" * 80)
    print("COMPARISON: OLD CONFIG (pattern=0.85, entry=0.70)")
    print("=" * 80)

    r_old = run_backtest(
        pattern_model, pattern_meta, entry_model, entry_features,
        0.85, 0.70, tickers, start_date, end_date,
        stop_model, regime_model, 0.19, signal_model
    )

    print(f"\n{'Metric':<20} {'Old (0.85/0.70)':>18} {'New (0.74/0.80)':>18}")
    print("-" * 58)
    print(f"{'Trades':<20} {r_old['trades']:>18} {r['trades']:>18}")
    print(f"{'Win Rate':<20} {r_old['win_rate']:>17.1f}% {r['win_rate']:>17.1f}%")
    print(f"{'Total P&L':<20} ${r_old['total_pnl']:>+16,.0f} ${r['total_pnl']:>+16,.0f}")
    print(f"{'Profit Factor':<20} {r_old['profit_factor']:>18.2f} {r['profit_factor']:>18.2f}")
    print(f"{'Sharpe':<20} {r_old['sharpe']:>18.2f} {r['sharpe']:>18.2f}")
    old_avg = r_old['total_pnl'] / r_old['trades'] if r_old['trades'] > 0 else 0
    new_avg = r['total_pnl'] / r['trades'] if r['trades'] > 0 else 0
    print(f"{'$/Trade':<20} ${old_avg:>+16,.0f} ${new_avg:>+16,.0f}")


if __name__ == "__main__":
    main()
