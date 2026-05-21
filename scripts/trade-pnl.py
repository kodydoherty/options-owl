#!/usr/bin/env python3
"""Trade P&L analysis — correct math for DCA, scaleout, and Webull fills.

Usage:
    python scripts/trade-pnl.py                     # all closed trades summary
    python scripts/trade-pnl.py NVDA                # specific ticker
    python scripts/trade-pnl.py --date 2026-05-19   # specific date
    python scripts/trade-pnl.py --id 226            # specific trade family
    python scripts/trade-pnl.py --recent 10         # last N trade families
    python scripts/trade-pnl.py --webull-only        # only Webull-executed trades

Runs against local DB by default. Use --droplet to query the droplet via SSH.

P&L Calculation Rules:
  1. A "trade family" = parent + scaleout children
  2. Parent P&L = (exit_premium - entry_premium) × parent_contracts × 100
  3. Child P&L  = (exit_premium - entry_premium) × child_contracts × 100
  4. Family P&L = parent P&L + SUM(children P&L)
  5. When Webull fills exist, use those instead of paper premiums
  6. DCA: parent's premium_per_contract is already blended (paper avg)
     BUT webull_entry_fill_price is only the FIRST fill — not blended
  7. For Webull accuracy: use paper blended entry for DCA trades,
     Webull exit fill for the exit (best available data)
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "journal" / "owlet-kody" / "raw_messages.db"

DROPLET_CMD = (
    "ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 "
    "sqlite3 -json /root/options-owl/journal/owlet-kody/raw_messages.db"
)


def query_db(sql: str, droplet: bool = False) -> list[dict]:
    """Run a query and return rows as dicts."""
    if droplet:
        import base64
        import re
        # Strip SQL comments before flattening (-- comments eat rest of line)
        sql_no_comments = re.sub(r"--[^\n]*", "", sql)
        flat_sql = " ".join(sql_no_comments.split()).rstrip().rstrip(";") + ";"
        b64 = base64.b64encode(flat_sql.encode()).decode()
        remote_cmd = (
            f"echo {b64} | base64 -d | "
            f"sqlite3 -json /root/options-owl/journal/owlet-kody/raw_messages.db"
        )
        cmd = [
            "ssh", "-i", str(Path.home() / ".ssh" / "id_ed25519_do"),
            "root@129.212.138.145",
            remote_cmd,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"SSH error: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        if not result.stdout.strip():
            return []
        return json.loads(result.stdout)
    else:
        if not DB_PATH.exists():
            print(f"Local DB not found: {DB_PATH}", file=sys.stderr)
            print("Use --droplet to query the droplet instead.", file=sys.stderr)
            sys.exit(1)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_trade_families(
    ticker: str | None = None,
    date: str | None = None,
    trade_id: int | None = None,
    recent: int | None = None,
    webull_only: bool = False,
    droplet: bool = False,
) -> list[dict]:
    """Get trade families with correct P&L aggregation."""

    where_clauses = ["p.status = 'closed'", "p.parent_trade_id IS NULL"]
    if ticker:
        where_clauses.append(f"p.ticker = '{ticker.upper()}'")
    if date:
        where_clauses.append(f"date(p.closed_at) = '{date}'")
    if trade_id:
        where_clauses.append(f"p.id = {trade_id}")
    if webull_only:
        where_clauses.append("p.webull_order_id IS NOT NULL")

    where = " AND ".join(where_clauses)
    limit = f"LIMIT {recent}" if recent else ""

    sql = f"""
    SELECT
        p.id,
        p.ticker,
        p.direction,
        p.option_type,
        p.strike,
        p.score,
        p.contracts as final_contracts,
        p.dca_total_contracts,
        p.premium_per_contract as blended_entry,
        p.webull_entry_fill_price as wb_first_entry,
        p.exit_premium as paper_exit,
        p.webull_exit_fill_price as wb_exit,
        p.total_cost,
        p.pnl_dollars as db_pnl,
        p.pnl_pct as db_pnl_pct,
        p.exit_reason,
        p.exit_source,
        p.opened_at,
        p.closed_at,
        p.duration_minutes,
        p.mfe_premium,
        p.mae_premium,
        p.webull_order_id,
        CASE WHEN p.webull_order_id IS NOT NULL THEN 1 ELSE 0 END as is_webull,
        -- Scaleout children aggregation
        COALESCE(c.child_count, 0) as scaleout_count,
        COALESCE(c.child_contracts, 0) as scaleout_contracts,
        COALESCE(c.child_pnl_sum, 0.0) as scaleout_pnl,
        -- DCA detection
        CASE WHEN p.dca_total_contracts > 0
             AND p.dca_total_contracts != p.contracts
             AND p.dca_total_contracts < p.contracts
        THEN 1 ELSE 0 END as had_dca
    FROM paper_trades p
    LEFT JOIN (
        SELECT
            parent_trade_id,
            COUNT(*) as child_count,
            SUM(contracts) as child_contracts,
            SUM(pnl_dollars) as child_pnl_sum
        FROM paper_trades
        WHERE parent_trade_id IS NOT NULL AND status = 'closed'
        GROUP BY parent_trade_id
    ) c ON c.parent_trade_id = p.id
    WHERE {where}
    ORDER BY p.id DESC
    {limit}
    """
    return query_db(sql, droplet=droplet)


def calc_correct_pnl(trade: dict) -> dict:
    """Calculate the most accurate P&L for a trade family.

    Priority for entry price:
      1. If DCA happened: use blended_entry (paper avg of all entries)
         - webull_entry_fill_price is only the FIRST entry, not blended
      2. If no DCA: use webull_entry_fill_price if available, else blended_entry

    Priority for exit price:
      1. webull_exit_fill_price if available
      2. paper exit_premium

    For scaleout children: use their db_pnl as-is (already computed per-child).
    """
    had_dca = trade["had_dca"]
    contracts = trade["final_contracts"]

    # Best entry price
    if had_dca:
        entry = trade["blended_entry"] or 0
        entry_source = "blended (DCA)"
    elif trade["wb_first_entry"] and trade["wb_first_entry"] > 0:
        entry = trade["wb_first_entry"]
        entry_source = "webull"
    else:
        entry = trade["blended_entry"] or 0
        entry_source = "paper"

    # Best exit price
    if trade["wb_exit"] and trade["wb_exit"] > 0:
        exit_p = trade["wb_exit"]
        exit_source = "webull"
    else:
        exit_p = trade["paper_exit"] or 0
        exit_source = "paper"

    # Parent P&L from clean math
    parent_pnl = (exit_p - entry) * contracts * 100 if entry > 0 else 0
    parent_pnl_pct = ((exit_p - entry) / entry * 100) if entry > 0 else 0

    # Scaleout child P&L (use DB values — they were computed at close time)
    child_pnl = trade["scaleout_pnl"] or 0

    # Family total
    family_pnl = parent_pnl + child_pnl

    # Original total contracts (before scaleout reduced parent)
    orig_contracts = contracts + (trade["scaleout_contracts"] or 0)

    # Total cost (what was actually spent to open all contracts)
    # For DCA: total_cost in DB includes DCA additions
    total_cost = trade["total_cost"] or (entry * orig_contracts * 100)

    return {
        "entry": entry,
        "entry_source": entry_source,
        "exit": exit_p,
        "exit_source": exit_source,
        "parent_contracts": contracts,
        "orig_contracts": orig_contracts,
        "parent_pnl": round(parent_pnl, 2),
        "parent_pnl_pct": round(parent_pnl_pct, 1),
        "scaleout_pnl": round(child_pnl, 2),
        "family_pnl": round(family_pnl, 2),
        "family_pnl_pct": round(family_pnl / total_cost * 100, 1) if total_cost > 0 else 0,
        "total_cost": round(total_cost, 2),
        "db_pnl": trade["db_pnl"] or 0,
        "pnl_diff": round(family_pnl - (trade["db_pnl"] or 0) - child_pnl, 2),
    }


def print_trade_detail(trade: dict, pnl: dict) -> None:
    """Print detailed trade info."""
    t = trade
    tid = t["id"]
    ticker = t["ticker"]
    otype = (t["option_type"] or "?").upper()[0]
    strike = t["strike"] or 0
    score = t["score"] or 0
    wb = "WB" if t["is_webull"] else "PP"

    print(f"\n{'='*70}")
    print(f"  #{tid} {ticker} {otype}{strike:.0f} | score={score} | {wb} | {t['exit_reason'] or '?'}")
    print(f"  opened: {t['opened_at'][:16]}  closed: {(t['closed_at'] or '')[:16]}"
          f"  duration: {t['duration_minutes'] or 0:.0f}min")
    print(f"{'='*70}")

    dca_note = " (DCA)" if t["had_dca"] else ""
    print(f"  Entry:  ${pnl['entry']:.2f} ({pnl['entry_source']}){dca_note}")
    print(f"  Exit:   ${pnl['exit']:.2f} ({pnl['exit_source']})")
    print(f"  Contracts: {pnl['orig_contracts']}"
          + (f" (scaleout sold {t['scaleout_contracts']},"
             f" closed {pnl['parent_contracts']})" if t["scaleout_count"] else ""))

    if t["had_dca"]:
        orig = t["dca_total_contracts"] or 0
        print(f"  DCA: started {orig} -> doubled to {pnl['orig_contracts']}")
        if t["wb_first_entry"] and t["wb_first_entry"] > 0:
            print(f"  Webull first fill: ${t['wb_first_entry']:.2f}"
                  f" (blended avg: ${t['blended_entry']:.2f})")

    print(f"\n  Parent P&L:   ${pnl['parent_pnl']:>+9.2f} ({pnl['parent_pnl_pct']:+.1f}%)")
    if t["scaleout_count"]:
        print(f"  Scaleout P&L: ${pnl['scaleout_pnl']:>+9.2f}")
    print(f"  Family P&L:   ${pnl['family_pnl']:>+9.2f} ({pnl['family_pnl_pct']:+.1f}%)")

    if abs(pnl["pnl_diff"]) > 1.0:
        print(f"  DB P&L:       ${pnl['db_pnl']:>+9.2f}  [MISMATCH: diff=${pnl['pnl_diff']:+.2f}]")

    if t["mfe_premium"] and t["mfe_premium"] > 0:
        mfe_pct = ((t["mfe_premium"] - pnl["entry"]) / pnl["entry"] * 100) if pnl["entry"] > 0 else 0
        mae_pct = ((t["mae_premium"] - pnl["entry"]) / pnl["entry"] * 100) if pnl["entry"] > 0 and t["mae_premium"] else 0
        print(f"  MFE: ${t['mfe_premium']:.2f} ({mfe_pct:+.1f}%)  "
              f"MAE: ${t['mae_premium'] or 0:.2f} ({mae_pct:+.1f}%)")


def print_summary(trades: list[dict], pnls: list[dict]) -> None:
    """Print summary stats."""
    if not pnls:
        print("\nNo trades found.")
        return

    total_pnl = sum(p["family_pnl"] for p in pnls)
    total_cost = sum(p["total_cost"] for p in pnls)
    winners = sum(1 for p in pnls if p["family_pnl"] >= 0)
    losers = len(pnls) - winners
    win_rate = winners / len(pnls) * 100 if pnls else 0

    avg_win = (sum(p["family_pnl"] for p in pnls if p["family_pnl"] >= 0) / winners) if winners else 0
    avg_loss = (sum(p["family_pnl"] for p in pnls if p["family_pnl"] < 0) / losers) if losers else 0

    webull_trades = [p for t, p in zip(trades, pnls) if t["is_webull"]]
    wb_pnl = sum(p["family_pnl"] for p in webull_trades) if webull_trades else 0

    dca_trades = [p for t, p in zip(trades, pnls) if t["had_dca"]]
    dca_pnl = sum(p["family_pnl"] for p in dca_trades) if dca_trades else 0

    scaleout_trades = [p for t, p in zip(trades, pnls) if t["scaleout_count"] > 0]
    scaleout_saved = sum(p["scaleout_pnl"] for p in scaleout_trades) if scaleout_trades else 0

    mismatches = sum(1 for p in pnls if abs(p["pnl_diff"]) > 1.0)

    print(f"\n{'='*60}")
    print(f"  TRADE SUMMARY — {len(pnls)} trade families")
    print(f"{'='*60}")
    print(f"  Total P&L:     ${total_pnl:>+10.2f}")
    print(f"  Win/Loss:      {winners}W / {losers}L ({win_rate:.1f}%)")
    print(f"  Avg Win:       ${avg_win:>+10.2f}")
    print(f"  Avg Loss:      ${avg_loss:>+10.2f}")
    if webull_trades:
        print(f"  Webull P&L:    ${wb_pnl:>+10.2f} ({len(webull_trades)} trades)")
    if dca_trades:
        print(f"  DCA trades:    {len(dca_trades)} (P&L: ${dca_pnl:+.2f})")
    if scaleout_trades:
        print(f"  Scaleout P&L:  ${scaleout_saved:>+10.2f} ({len(scaleout_trades)} trades)")
    if mismatches:
        print(f"  DB mismatches: {mismatches} trades (recalc differs from stored P&L)")
    print(f"{'='*60}")


def print_table(trades: list[dict], pnls: list[dict]) -> None:
    """Print compact table of trades."""
    print(f"\n{'ID':>4} {'Ticker':<6} {'Type':>4} {'Cts':>4} {'Entry':>7} {'Exit':>7} "
          f"{'P&L':>10} {'%':>7} {'Exit Reason':<16} {'Src':>3}")
    print("-" * 80)
    for t, p in zip(trades, pnls):
        otype = (t["option_type"] or "?")[0].upper()
        wb = "WB" if t["is_webull"] else "PP"
        dca = "*" if t["had_dca"] else " "
        so = "+" if t["scaleout_count"] else " "
        flags = f"{dca}{so}"
        print(f"{t['id']:>4} {t['ticker']:<6} {otype}{flags} {p['orig_contracts']:>4} "
              f"${p['entry']:>6.2f} ${p['exit']:>6.2f} "
              f"${p['family_pnl']:>+9.2f} {p['family_pnl_pct']:>+6.1f}% "
              f"{(t['exit_reason'] or '?'):<16} {wb}")
    print()
    print("Flags: * = DCA, + = scaleout")


def main():
    parser = argparse.ArgumentParser(description="Trade P&L analysis with correct DCA/scaleout math")
    parser.add_argument("ticker", nargs="?", help="Filter by ticker")
    parser.add_argument("--date", help="Filter by close date (YYYY-MM-DD)")
    parser.add_argument("--id", type=int, help="Show specific trade family")
    parser.add_argument("--recent", type=int, help="Show last N trade families")
    parser.add_argument("--webull-only", action="store_true", help="Only Webull-executed trades")
    parser.add_argument("--droplet", action="store_true", help="Query droplet via SSH")
    parser.add_argument("--detail", action="store_true", help="Show detailed per-trade breakdown")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary stats")
    args = parser.parse_args()

    if not args.recent and not args.id and not args.ticker and not args.date and not args.webull_only:
        args.recent = 20  # default to last 20

    trades = get_trade_families(
        ticker=args.ticker,
        date=args.date,
        trade_id=args.id,
        recent=args.recent,
        webull_only=args.webull_only,
        droplet=args.droplet,
    )

    pnls = [calc_correct_pnl(t) for t in trades]

    if args.detail or args.id:
        for t, p in zip(trades, pnls):
            print_trade_detail(t, p)
    else:
        print_table(trades, pnls)

    if not args.no_summary:
        print_summary(trades, pnls)


if __name__ == "__main__":
    main()
