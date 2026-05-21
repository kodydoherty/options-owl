"""Generate a paper trading report for the last N hours (default 6)."""

import asyncio
import sqlite3
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.getenv("DB_PATH", "journal/raw_messages.db")
REPORT_HOURS = int(os.getenv("REPORT_HOURS", "6"))
REPORT_FILE = os.path.join(os.path.dirname(__file__), "..", "journal", "last_report.txt")


def run_report() -> str:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=REPORT_HOURS)
    window_start_iso = window_start.isoformat()
    now_iso = now.isoformat()

    db_path = os.path.join(os.path.dirname(__file__), "..", DB_PATH)
    if not os.path.exists(db_path):
        return "ERROR: Database not found at " + db_path

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    lines = []
    lines.append("=" * 60)
    lines.append(f"  OptionsOwl Paper Trading Report")
    lines.append(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Window: last {REPORT_HOURS} hours")
    lines.append("=" * 60)

    # Portfolio snapshot — both strategies (handle legacy DBs without strategy column)
    try:
        portfolios = conn.execute("SELECT * FROM paper_portfolio ORDER BY strategy ASC").fetchall()
    except sqlite3.OperationalError:
        portfolios = conn.execute("SELECT * FROM paper_portfolio").fetchall()
    if portfolios:
        lines.append("")
        lines.append("PORTFOLIO COMPARISON")
        for row in portfolios:
            strat = row["strategy"] if "strategy" in row.keys() else "B"
            label = "A (Old)" if strat == "A" else "B (New: DCA+ScaleOut)"
            balance = row["current_balance"]
            starting = row["starting_balance"]
            total_return = balance - starting
            total_return_pct = (total_return / starting * 100) if starting else 0
            wr = (row["wins"] / (row["wins"] + row["losses"]) * 100) if (row["wins"] + row["losses"]) > 0 else 0
            lines.append(f"  [{label}]")
            lines.append(f"    Balance:   ${balance:,.2f}  (start: ${starting:,.2f})")
            lines.append(f"    P&L:       ${total_return:+,.2f} ({total_return_pct:+.2f}%)")
            lines.append(f"    W/L:       {row['wins']}W / {row['losses']}L  ({wr:.1f}%)")
            lines.append(f"    Trades:    {row['total_trades']}")
            lines.append("")

    # Trades closed in this window
    closed_trades = conn.execute(
        """SELECT * FROM paper_trades
           WHERE status='closed' AND closed_at >= ?
           ORDER BY closed_at ASC""",
        (window_start_iso,),
    ).fetchall()

    lines.append("")
    lines.append(f"TRADES CLOSED (last {REPORT_HOURS}h): {len(closed_trades)}")

    if closed_trades:
        window_pnl = sum(t["pnl_dollars"] for t in closed_trades)
        window_wins = sum(1 for t in closed_trades if t["pnl_dollars"] >= 0)
        window_losses = len(closed_trades) - window_wins
        lines.append(f"  Window P&L:  ${window_pnl:+,.2f}")
        lines.append(f"  Window W/L:  {window_wins}W / {window_losses}L")
        lines.append("")

        for t in closed_trades:
            pnl = t["pnl_dollars"]
            pnl_pct = t["pnl_pct"] or 0
            icon = "+" if pnl >= 0 else "-"
            strat = t["strategy"] if "strategy" in t.keys() else "B"
            lines.append(
                f"  {icon} [{strat}] {t['ticker']:6s} {t['direction']:4s} "
                f"{t['contracts']}x @ ${t['premium_per_contract']:.2f} → ${t['exit_premium']:.2f}  "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)  [{t['exit_reason']}]"
            )
    else:
        lines.append("  No trades closed in this window.")

    # Trades opened in this window
    opened_trades = conn.execute(
        """SELECT * FROM paper_trades
           WHERE opened_at >= ?
           ORDER BY opened_at ASC""",
        (window_start_iso,),
    ).fetchall()

    lines.append("")
    lines.append(f"TRADES OPENED (last {REPORT_HOURS}h): {len(opened_trades)}")
    for t in opened_trades:
        status_tag = "OPEN" if t["status"] == "open" else "CLOSED"
        strat = t["strategy"] if "strategy" in t.keys() else "B"
        lines.append(
            f"  [{strat}] {t['ticker']:6s} {t['direction']:4s} "
            f"strike=${t['strike']:.0f} {t['contracts']}x @ ${t['premium_per_contract']:.2f}  "
            f"[{status_tag}] src={t['bot_source']}"
        )

    # Currently open positions
    open_positions = conn.execute(
        "SELECT * FROM paper_trades WHERE status='open' ORDER BY opened_at ASC"
    ).fetchall()

    lines.append("")
    lines.append(f"OPEN POSITIONS: {len(open_positions)}")
    if open_positions:
        for t in open_positions:
            strat = t["strategy"] if "strategy" in t.keys() else "B"
            lines.append(
                f"  [{strat}] {t['ticker']:6s} {t['direction']:4s} "
                f"strike=${t['strike']:.0f} {t['contracts']}x @ ${t['premium_per_contract']:.2f}  "
                f"stop=${t['stop_price']:.2f} T1=${t['target_1']:.2f}"
            )
    else:
        lines.append("  No open positions.")

    # Bot performance breakdown (all time)
    # Strategy comparison (skip if DB doesn't have strategy column)
    try:
        strat_stats = conn.execute(
            """SELECT COALESCE(strategy, 'B') as strat,
                  COUNT(*) as total,
                  SUM(CASE WHEN pnl_dollars >= 0 THEN 1 ELSE 0 END) as wins,
                  SUM(CASE WHEN pnl_dollars < 0 THEN 1 ELSE 0 END) as losses,
                  SUM(pnl_dollars) as total_pnl,
                  AVG(pnl_pct) as avg_pnl_pct
           FROM paper_trades WHERE status='closed'
           GROUP BY strat ORDER BY strat ASC"""
        ).fetchall()
    except sqlite3.OperationalError:
        strat_stats = []

    if strat_stats and len(strat_stats) > 1:
        lines.append("")
        lines.append("STRATEGY A vs B (closed trades)")
        for s in strat_stats:
            label = "A (Old)" if s["strat"] == "A" else "B (New)"
            wr = (s["wins"] / s["total"] * 100) if s["total"] > 0 else 0
            lines.append(
                f"  [{label}]  "
                f"{s['total']}T {s['wins']}W/{s['losses']}L ({wr:.0f}%)  "
                f"P&L: ${s['total_pnl']:+,.2f}  avg: {s['avg_pnl_pct']:+.1f}%"
            )

    bot_stats = conn.execute(
        """SELECT bot_source,
                  COUNT(*) as total,
                  SUM(CASE WHEN pnl_dollars >= 0 THEN 1 ELSE 0 END) as wins,
                  SUM(CASE WHEN pnl_dollars < 0 THEN 1 ELSE 0 END) as losses,
                  SUM(pnl_dollars) as total_pnl,
                  AVG(pnl_pct) as avg_pnl_pct
           FROM paper_trades WHERE status='closed'
           GROUP BY bot_source ORDER BY total_pnl DESC"""
    ).fetchall()

    if bot_stats:
        lines.append("")
        lines.append("BOT PERFORMANCE (all time)")
        for b in bot_stats:
            wr = (b["wins"] / b["total"] * 100) if b["total"] > 0 else 0
            lines.append(
                f"  {b['bot_source'] or 'unknown':20s}  "
                f"{b['total']}T {b['wins']}W/{b['losses']}L ({wr:.0f}%)  "
                f"P&L: ${b['total_pnl']:+,.2f}  avg: {b['avg_pnl_pct']:+.1f}%"
            )

    # Slippage analysis (all time, closed trades only)
    slippage_row = conn.execute(
        """SELECT
               COALESCE(SUM(entry_slippage * contracts * 100), 0) as total_entry_slippage,
               COALESCE(SUM(exit_slippage * contracts * 100), 0) as total_exit_slippage,
               COUNT(*) as trade_count,
               COALESCE(SUM(pnl_dollars), 0) as total_pnl
           FROM paper_trades WHERE status='closed'"""
    ).fetchone()

    if slippage_row and slippage_row["trade_count"] > 0:
        total_entry_slip = slippage_row["total_entry_slippage"]
        total_exit_slip = slippage_row["total_exit_slippage"]
        combined_slip = total_entry_slip + total_exit_slip
        total_pnl_for_slip = slippage_row["total_pnl"]
        trade_count = slippage_row["trade_count"]
        avg_slip = combined_slip / trade_count if trade_count > 0 else 0
        slip_pct_of_pnl = (
            (combined_slip / abs(total_pnl_for_slip) * 100)
            if total_pnl_for_slip != 0
            else 0
        )

        lines.append("")
        lines.append("SLIPPAGE (all time)")
        lines.append(f"  Entry slippage:    ${total_entry_slip:+,.2f}")
        lines.append(f"  Exit slippage:     ${total_exit_slip:+,.2f}")
        lines.append(f"  Combined:          ${combined_slip:+,.2f}")
        lines.append(f"  Avg per trade:     ${avg_slip:+,.2f}")
        lines.append(f"  As % of total P&L: {slip_pct_of_pnl:+.1f}%")

    # Best / worst trades all time
    best = conn.execute(
        "SELECT ticker, direction, pnl_dollars, pnl_pct FROM paper_trades WHERE status='closed' ORDER BY pnl_dollars DESC LIMIT 1"
    ).fetchone()
    worst = conn.execute(
        "SELECT ticker, direction, pnl_dollars, pnl_pct FROM paper_trades WHERE status='closed' ORDER BY pnl_dollars ASC LIMIT 1"
    ).fetchone()

    if best and worst:
        lines.append("")
        lines.append("EXTREMES (all time)")
        lines.append(f"  Best:  {best['ticker']} {best['direction']} ${best['pnl_dollars']:+,.2f} ({best['pnl_pct']:+.1f}%)")
        lines.append(f"  Worst: {worst['ticker']} {worst['direction']} ${worst['pnl_dollars']:+,.2f} ({worst['pnl_pct']:+.1f}%)")

    # Excursion analysis (MFE / MAE) for closed trades with data
    excursion_trades = conn.execute(
        """SELECT ticker, direction, pnl_pct, mfe_pnl_pct, mae_pnl_pct
           FROM paper_trades
           WHERE status='closed' AND mfe_pnl_pct IS NOT NULL AND mae_pnl_pct IS NOT NULL"""
    ).fetchall()

    if excursion_trades:
        mfe_vals = [t["mfe_pnl_pct"] for t in excursion_trades]
        mae_vals = [t["mae_pnl_pct"] for t in excursion_trades]
        pnl_vals = [t["pnl_pct"] or 0 for t in excursion_trades]
        left_on_table = [m - p for m, p in zip(mfe_vals, pnl_vals)]

        avg_mfe = sum(mfe_vals) / len(mfe_vals)
        avg_mae = sum(mae_vals) / len(mae_vals)
        avg_left = sum(left_on_table) / len(left_on_table)
        avg_heat = avg_mae

        # Worst MAE trade
        worst_mae_trade = min(excursion_trades, key=lambda t: t["mae_pnl_pct"])

        lines.append("")
        lines.append("EXCURSION ANALYSIS (MFE/MAE)")
        lines.append(f"  Trades with data: {len(excursion_trades)}")
        lines.append(f"  Avg MFE:          {avg_mfe:+.1f}%  (avg max profit available)")
        lines.append(f"  Avg MAE:          {avg_mae:+.1f}%  (avg max drawdown during trade)")
        lines.append(f"  Avg Left on Table:{avg_left:+.1f}%  (MFE - actual exit P&L)")
        lines.append(f"  Avg Heat Taken:   {avg_heat:+.1f}%  (how deep trades went against you)")
        lines.append(
            f"  Worst MAE:        {worst_mae_trade['ticker']} {worst_mae_trade['direction']} "
            f"MAE={worst_mae_trade['mae_pnl_pct']:+.1f}% "
            f"(exited at {worst_mae_trade['pnl_pct']:+.1f}%)"
        )

    lines.append("")
    lines.append("=" * 60)

    conn.close()
    report = "\n".join(lines)

    # Also save to file
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    return report


if __name__ == "__main__":
    print(run_report())
