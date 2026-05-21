"""Export closed paper trades to CSV for backtesting analysis.

Outputs all 10 tracked fields:
  1. Ticker
  2. Date of purchase
  3. Strike price
  4. Cost per contract (premium)
  5. Time of purchase
  6. Time of sell
  7. Duration (minutes)
  8. P&L %
  9. Sell reason
  10. Direction (Call/Put)

Plus extra columns: strategy, contracts, total_cost, pnl_dollars, entry_price,
exit_premium, targets, stop_price, bot_source, score.

Usage:
    python scripts/backtest_export.py                    # all closed trades
    python scripts/backtest_export.py --strategy B       # only strategy B
    python scripts/backtest_export.py --output trades.csv
"""

import csv
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.getenv("DB_PATH", "journal/raw_messages.db")


def export(strategy: str | None = None, output: str | None = None) -> str:
    db_path = os.path.join(os.path.dirname(__file__), "..", DB_PATH)
    if not os.path.exists(db_path):
        return f"ERROR: Database not found at {db_path}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            ticker,
            opened_at,
            strike,
            premium_per_contract,
            closed_at,
            duration_minutes,
            pnl_pct,
            pnl_dollars,
            exit_reason,
            option_type,
            strategy,
            contracts,
            total_cost,
            entry_price,
            exit_price,
            exit_premium,
            target_1, target_2, target_3, target_4, target_5,
            stop_price,
            bot_source,
            score,
            signal_premium,
            entry_slippage,
            exit_slippage,
            parent_trade_id
        FROM paper_trades
        WHERE status = 'closed'
    """
    params: list = []
    if strategy:
        query += " AND COALESCE(strategy, 'B') = ?"
        params.append(strategy)
    query += " ORDER BY closed_at ASC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return "No closed trades found."

    headers = [
        "ticker",
        "purchase_date",
        "strike_price",
        "cost_per_contract",
        "purchase_time",
        "sell_time",
        "duration_minutes",
        "pnl_pct",
        "pnl_dollars",
        "sell_reason",
        "direction",
        "strategy",
        "contracts",
        "total_cost",
        "underlying_entry",
        "underlying_exit",
        "exit_premium",
        "target_1",
        "target_2",
        "target_3",
        "target_4",
        "target_5",
        "stop_price",
        "bot_source",
        "score",
        "signal_premium",
        "entry_slippage",
        "exit_slippage",
        "is_partial_close",
    ]

    csv_rows = []
    for r in rows:
        opened = r["opened_at"] or ""
        closed = r["closed_at"] or ""
        csv_rows.append([
            r["ticker"],
            opened[:10] if opened else "",           # purchase_date
            r["strike"],
            r["premium_per_contract"],
            opened,                                   # full purchase_time
            closed,                                   # full sell_time
            r["duration_minutes"],
            round(r["pnl_pct"] or 0, 2),
            round(r["pnl_dollars"] or 0, 2),
            r["exit_reason"],
            (r["option_type"] or "").upper(),          # Call/Put
            r["strategy"] or "B",
            r["contracts"],
            r["total_cost"],
            r["entry_price"],
            r["exit_price"],
            r["exit_premium"],
            r["target_1"],
            r["target_2"],
            r["target_3"],
            r["target_4"],
            r["target_5"],
            r["stop_price"],
            r["bot_source"],
            r["score"],
            r["signal_premium"],
            r["entry_slippage"],
            r["exit_slippage"],
            "yes" if r["parent_trade_id"] else "no",
        ])

    out_path = output or os.path.join(
        os.path.dirname(__file__), "..", "journal", "backtest_trades.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    return f"Exported {len(csv_rows)} trades to {out_path}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trades for backtesting")
    parser.add_argument("--strategy", choices=["A", "B"], help="Filter by strategy")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()
    print(export(strategy=args.strategy, output=args.output))
