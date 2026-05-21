#!/usr/bin/env python3
"""Backfill P&L for DCA trades using consistent math.

Problem: V6 DCA used paper premium (monitor quote) as the DCA fill price,
not the actual Webull fill. The blended premium_per_contract is wrong for
17 of 24 DCA trades, causing pnl_dollars to be inaccurate.

What this script does:
  1. For each DCA trade, recalculate pnl_dollars using:
     - Entry: premium_per_contract (blended, best we have — includes paper DCA price)
     - Exit: webull_exit_fill_price if available, else exit_premium
     - Contracts: current contracts value
  2. Also recalculate for scaleout children
  3. Show the diff, optionally apply with --fix

This makes P&L internally consistent even though the DCA entry price
is still paper-based. The forward fix (capturing real Webull DCA fills)
prevents this issue for new trades.

Usage:
    python scripts/backfill_dca_pnl.py              # dry run, show diffs
    python scripts/backfill_dca_pnl.py --fix         # apply fixes
    python scripts/backfill_dca_pnl.py --all         # fix ALL trades, not just DCA
    python scripts/backfill_dca_pnl.py --droplet     # run on droplet DB via SSH
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "journal" / "owlet-kody" / "raw_messages.db"


def get_conn(droplet: bool = False):
    if droplet:
        return None  # will use SSH
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query_droplet(sql: str) -> list[dict]:
    import base64
    import re
    sql_clean = re.sub(r"--[^\n]*", "", sql)
    flat = " ".join(sql_clean.split()).rstrip().rstrip(";") + ";"
    b64 = base64.b64encode(flat.encode()).decode()
    cmd = [
        "ssh", "-i", str(Path.home() / ".ssh" / "id_ed25519_do"),
        "root@129.212.138.145",
        f"echo {b64} | base64 -d | sqlite3 -json /root/options-owl/journal/owlet-kody/raw_messages.db",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SSH error: {r.stderr}", file=sys.stderr)
        return []
    return json.loads(r.stdout) if r.stdout.strip() else []


def exec_droplet(sql: str) -> None:
    import base64
    import re
    sql_clean = re.sub(r"--[^\n]*", "", sql)
    flat = " ".join(sql_clean.split()).rstrip().rstrip(";") + ";"
    b64 = base64.b64encode(flat.encode()).decode()
    cmd = [
        "ssh", "-i", str(Path.home() / ".ssh" / "id_ed25519_do"),
        "root@129.212.138.145",
        f"echo {b64} | base64 -d | sqlite3 /root/options-owl/journal/owlet-kody/raw_messages.db",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SSH exec error: {r.stderr}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    parser.add_argument("--all", action="store_true", help="Fix all closed trades, not just DCA")
    parser.add_argument("--droplet", action="store_true", help="Run against droplet")
    args = parser.parse_args()

    # Get all closed trades (parents only)
    where = "status = 'closed'"
    if not args.all:
        where += " AND dca_total_contracts > 0 AND dca_total_contracts < contracts"

    sql = f"""
        SELECT id, ticker, contracts, premium_per_contract,
            webull_entry_fill_price, exit_premium, webull_exit_fill_price,
            total_cost, pnl_dollars, pnl_pct,
            dca_total_contracts, parent_trade_id
        FROM paper_trades
        WHERE {where}
        ORDER BY id
    """

    if args.droplet:
        trades = query_droplet(sql)
    else:
        conn = get_conn()
        trades = [dict(r) for r in conn.execute(sql).fetchall()]

    fixes = []
    total_diff = 0.0

    for t in trades:
        tid = t["id"]
        contracts = t["contracts"]
        is_child = t["parent_trade_id"] is not None

        # Best entry: for DCA parents, use blended premium_per_contract
        # For children (scaleout), use premium_per_contract (inherited from parent)
        entry = t["premium_per_contract"] or 0
        if not is_child and t["webull_entry_fill_price"] and t["dca_total_contracts"] in (0, None):
            # Non-DCA trade: use Webull first fill if available
            entry = t["webull_entry_fill_price"]

        # Best exit: Webull fill > paper
        exit_p = t["webull_exit_fill_price"] if t["webull_exit_fill_price"] and t["webull_exit_fill_price"] > 0 else (t["exit_premium"] or 0)

        if entry <= 0 or exit_p <= 0:
            continue

        # Recalculate
        new_pnl = (exit_p - entry) * contracts * 100
        new_pnl_pct = (exit_p - entry) / entry * 100
        old_pnl = t["pnl_dollars"] or 0

        diff = new_pnl - old_pnl
        if abs(diff) < 0.50:
            continue  # close enough

        total_diff += diff
        fixes.append({
            "id": tid,
            "ticker": t["ticker"],
            "contracts": contracts,
            "entry": entry,
            "exit": exit_p,
            "old_pnl": old_pnl,
            "new_pnl": round(new_pnl, 2),
            "new_pnl_pct": round(new_pnl_pct, 2),
            "diff": round(diff, 2),
            "is_dca": (t["dca_total_contracts"] or 0) > 0 and (t["dca_total_contracts"] or 0) < contracts,
        })

    if not fixes:
        print("No P&L fixes needed.")
        return

    print(f"\n{'ID':>4} {'Ticker':<6} {'Cts':>4} {'Entry':>7} {'Exit':>7} "
          f"{'Old P&L':>10} {'New P&L':>10} {'Diff':>9} {'DCA':>3}")
    print("-" * 75)

    for f in fixes:
        dca = "Y" if f["is_dca"] else ""
        print(f"{f['id']:>4} {f['ticker']:<6} {f['contracts']:>4} "
              f"${f['entry']:>6.2f} ${f['exit']:>6.2f} "
              f"${f['old_pnl']:>+9.2f} ${f['new_pnl']:>+9.2f} ${f['diff']:>+8.2f} {dca:>3}")

    print(f"\n{len(fixes)} trades to fix, total P&L shift: ${total_diff:+.2f}")

    if not args.fix:
        print("\nDry run — use --fix to apply.")
        return

    print("\nApplying fixes...")
    if args.droplet:
        for f in fixes:
            exec_droplet(
                f"UPDATE paper_trades SET pnl_dollars = {f['new_pnl']}, "
                f"pnl_pct = {f['new_pnl_pct']} WHERE id = {f['id']}"
            )
    else:
        conn = get_conn()
        for f in fixes:
            conn.execute(
                "UPDATE paper_trades SET pnl_dollars = ?, pnl_pct = ? WHERE id = ?",
                (f["new_pnl"], f["new_pnl_pct"], f["id"]),
            )
        conn.commit()
        conn.close()

    print(f"Fixed {len(fixes)} trades.")


if __name__ == "__main__":
    main()
