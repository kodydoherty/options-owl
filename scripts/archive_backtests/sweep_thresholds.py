"""Fine-grained threshold sweep for pattern + entry timing models.

Sweeps pattern threshold 0.70-0.92 (step 0.02) x entry threshold OFF, 0.50-0.80 (step 0.10).
Uses the gold standard backtest infrastructure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.backtest_gold_standard import (
    EXCLUDED_TICKERS,
    TICKERS,
    load_models,
    run_backtest,
    THETADATA_DB,
)
import sqlite3


def main():
    print("=" * 80)
    print("FINE-GRAINED THRESHOLD SWEEP")
    print("Pattern: 0.70-0.92 (step 0.02) x Entry: OFF, 0.50, 0.60, 0.70, 0.80")
    print("=" * 80)

    # Date range — last 60 trading days
    conn = sqlite3.connect(THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1 DESC
    """).fetchall()]
    conn.close()

    end_date = all_dates[0]
    start_date = all_dates[min(59, len(all_dates) - 1)]
    tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    print(f"Period: {start_date} to {end_date}")
    print(f"Tickers: {len(tickers)}")

    # Load models
    print("\nLoading models...")
    pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model, signal_model = load_models(
        use_entry_filter=True, use_regime=True
    )

    # Pattern thresholds: 0.70 to 0.92 step 0.02
    pattern_thresholds = [round(0.70 + i * 0.02, 2) for i in range(12)]  # 0.70, 0.72, ..., 0.92
    # Entry thresholds: OFF (None), 0.50, 0.60, 0.70, 0.80
    entry_thresholds = [None, 0.50, 0.60, 0.70, 0.80]

    results = []

    header = (f"{'PatTh':<7} {'EntTh':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} "
              f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgWin':>8} {'AvgLoss':>8} {'$/Trade':>8}")
    print(f"\n{header}")
    print("-" * 90)

    t0 = time.time()
    total_runs = len(pattern_thresholds) * len(entry_thresholds)
    run_num = 0

    for pt in pattern_thresholds:
        for et_raw in entry_thresholds:
            run_num += 1
            em = entry_model if et_raw is not None else None
            ef = entry_features if et_raw is not None else None
            et = et_raw if et_raw is not None else 0.0

            r = run_backtest(
                pattern_model, pattern_meta, em, ef,
                pt, et, tickers, start_date, end_date,
                stop_model, regime_model, 0.02, signal_model
            )

            et_str = f"{et_raw:.2f}" if et_raw is not None else "OFF"
            pnl_str = f"${r['total_pnl']:+,.0f}"
            avg_pnl = r['total_pnl'] / r['trades'] if r['trades'] > 0 else 0
            print(f"{pt:<7.2f} {et_str:<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
                  f"{pnl_str:>11} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown_pct']:>6.1f}% {r['avg_win']:>+7.0f} {r['avg_loss']:>+7.0f} "
                  f"{avg_pnl:>+7.0f}")

            results.append({
                "pattern": pt, "entry": et_raw, "trades": r["trades"],
                "win_rate": r["win_rate"], "pnl": r["total_pnl"],
                "pf": r["profit_factor"], "sharpe": r["sharpe"],
                "max_dd": r["max_drawdown_pct"], "avg_win": r["avg_win"],
                "avg_loss": r["avg_loss"], "avg_pnl": avg_pnl,
            })

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"Sweep complete: {total_runs} configs in {elapsed:.0f}s")

    # Top 10 by P&L
    print(f"\n{'=' * 90}")
    print("TOP 10 BY TOTAL P&L")
    print(f"{'=' * 90}")
    by_pnl = sorted(results, key=lambda x: x["pnl"], reverse=True)[:10]
    print(f"{'PatTh':<7} {'EntTh':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} {'Sharpe':>7} {'$/Trade':>8}")
    print("-" * 65)
    for r in by_pnl:
        et_str = f"{r['entry']:.2f}" if r['entry'] is not None else "OFF"
        print(f"{r['pattern']:<7.2f} {et_str:<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
              f"${r['pnl']:>+10,.0f} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['avg_pnl']:>+7.0f}")

    # Top 10 by Sharpe (min 20 trades)
    print(f"\n{'=' * 90}")
    print("TOP 10 BY SHARPE (min 20 trades)")
    print(f"{'=' * 90}")
    qualified = [r for r in results if r["trades"] >= 20]
    by_sharpe = sorted(qualified, key=lambda x: x["sharpe"], reverse=True)[:10]
    print(f"{'PatTh':<7} {'EntTh':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} {'Sharpe':>7} {'$/Trade':>8}")
    print("-" * 65)
    for r in by_sharpe:
        et_str = f"{r['entry']:.2f}" if r['entry'] is not None else "OFF"
        print(f"{r['pattern']:<7.2f} {et_str:<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
              f"${r['pnl']:>+10,.0f} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['avg_pnl']:>+7.0f}")

    # Top 10 by profit factor (min 20 trades)
    print(f"\n{'=' * 90}")
    print("TOP 10 BY PROFIT FACTOR (min 20 trades)")
    print(f"{'=' * 90}")
    by_pf = sorted(qualified, key=lambda x: x["pf"], reverse=True)[:10]
    print(f"{'PatTh':<7} {'EntTh':<7} {'Trades':<7} {'WR%':<6} {'P&L':>11} {'PF':>6} {'Sharpe':>7} {'$/Trade':>8}")
    print("-" * 65)
    for r in by_pf:
        et_str = f"{r['entry']:.2f}" if r['entry'] is not None else "OFF"
        print(f"{r['pattern']:<7.2f} {et_str:<7} {r['trades']:<7} {r['win_rate']:<6.1f} "
              f"${r['pnl']:>+10,.0f} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['avg_pnl']:>+7.0f}")

    # Trade count summary
    print(f"\n{'=' * 90}")
    print("TRADE COUNT BY THRESHOLD (entry=OFF)")
    print(f"{'=' * 90}")
    no_entry = [r for r in results if r["entry"] is None]
    for r in sorted(no_entry, key=lambda x: x["pattern"]):
        bar = "#" * (r["trades"] // 2)
        print(f"  {r['pattern']:.2f}  {r['trades']:>4} trades  {bar}")

    print(f"\nTRADE COUNT BY THRESHOLD (entry=0.70)")
    entry70 = [r for r in results if r["entry"] == 0.70]
    for r in sorted(entry70, key=lambda x: x["pattern"]):
        bar = "#" * (r["trades"] // 2)
        print(f"  {r['pattern']:.2f}  {r['trades']:>4} trades  {bar}")


if __name__ == "__main__":
    main()
