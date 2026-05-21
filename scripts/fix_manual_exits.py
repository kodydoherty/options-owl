#!/usr/bin/env python3
"""Fix manual-close trades using real Webull sell fill prices.

Problem: When user sells manually on Webull, the bot detects "position gone"
and records an approximate exit_premium (market price at detection time).
The real Webull fill price can be significantly different.

This script:
  1. Pulls Webull order history CSV from the droplet
  2. Matches SELL orders to manual-close trades by ticker + date + qty
  3. Updates webull_exit_fill_price, exit_premium, pnl_dollars, pnl_pct

Usage:
    python scripts/fix_manual_exits.py              # dry run
    python scripts/fix_manual_exits.py --fix        # apply fixes
"""

import argparse
import base64
import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path


def ssh_cmd(cmd: str) -> str:
    r = subprocess.run(
        ["ssh", "-i", str(Path.home() / ".ssh" / "id_ed25519_do"),
         "root@129.212.138.145", cmd],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"SSH error: {r.stderr}", file=sys.stderr)
    return r.stdout


def query_droplet(sql: str) -> list[dict]:
    sql_clean = re.sub(r"--[^\n]*", "", sql)
    flat = " ".join(sql_clean.split()).rstrip().rstrip(";") + ";"
    b64 = base64.b64encode(flat.encode()).decode()
    out = ssh_cmd(
        f"echo {b64} | base64 -d | sqlite3 -json "
        f"/root/options-owl/journal/owlet-kody/raw_messages.db"
    )
    return json.loads(out) if out.strip() else []


def exec_droplet(sql: str) -> None:
    sql_clean = re.sub(r"--[^\n]*", "", sql)
    flat = " ".join(sql_clean.split()).rstrip().rstrip(";") + ";"
    b64 = base64.b64encode(flat.encode()).decode()
    ssh_cmd(
        f"echo {b64} | base64 -d | sqlite3 "
        f"/root/options-owl/journal/owlet-kody/raw_messages.db"
    )


def pull_webull_sells(csv_path: str | None = None) -> list[dict]:
    """Pull Webull sell orders from history CSV (cached or live)."""
    if csv_path:
        text = open(csv_path).read()
    else:
        print("Pulling Webull order history from droplet...", file=sys.stderr)
        text = ssh_cmd(
            "cd /root/options-owl && "
            "docker compose exec -T owlet-kody python scripts/webull_history.py 2>/dev/null"
        )

    # Skip SDK noise lines before CSV header
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("day,"):
            break
    else:
        print("No CSV header found in Webull history output", file=sys.stderr)
        return []

    csv_text = "".join(lines[i:])
    sells = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        if row.get("side") == "SELL":
            sells.append(row)
    return sells


def match_sells_to_trade(trade: dict, sells: list[dict]) -> dict | None:
    """Match a manual-close trade to its Webull sell order(s).

    Returns dict with real_exit_price and matched_qty, or None.
    For multi-tranche sells, returns weighted average price.
    """
    ticker = trade["ticker"]
    dt = trade["opened_date"]
    cts = trade["contracts"]

    # Find sells for same ticker on same date
    candidates = [s for s in sells if s["symbol"] == ticker and s["day"] == dt]

    if not candidates:
        return None

    # Try exact qty match first
    exact = [c for c in candidates if int(c["qty"]) == cts]
    if exact:
        return {
            "real_exit_price": float(exact[0]["price"]),
            "matched_qty": int(exact[0]["qty"]),
            "order_ids": [exact[0]["order_id"]],
            "tranches": 1,
        }

    # Try combining multiple sells (user may have sold in tranches)
    # Sort by time to combine sequential sells
    candidates.sort(key=lambda x: x["time_utc"])
    total_qty = sum(int(c["qty"]) for c in candidates)
    if total_qty >= cts:
        # Weighted average price across tranches that sum to our qty
        running_qty = 0
        running_value = 0
        used = []
        for c in candidates:
            q = int(c["qty"])
            p = float(c["price"])
            take = min(q, cts - running_qty)
            running_qty += take
            running_value += take * p
            used.append(c)
            if running_qty >= cts:
                break

        if running_qty >= cts:
            avg_price = running_value / cts
            return {
                "real_exit_price": round(avg_price, 2),
                "matched_qty": cts,
                "order_ids": [c["order_id"] for c in used],
                "tranches": len(used),
            }

    # Partial match — use best single candidate
    closest = min(candidates, key=lambda c: abs(int(c["qty"]) - cts))
    return {
        "real_exit_price": float(closest["price"]),
        "matched_qty": int(closest["qty"]),
        "order_ids": [closest["order_id"]],
        "tranches": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply fixes to droplet DB")
    parser.add_argument("--csv", help="Use cached CSV instead of pulling from droplet")
    args = parser.parse_args()

    # Get manual-close trades
    trades = query_droplet("""
        SELECT id, ticker, contracts,
            premium_per_contract, webull_entry_fill_price,
            exit_premium, webull_exit_fill_price,
            pnl_dollars, pnl_pct, exit_source,
            date(opened_at) as opened_date
        FROM paper_trades
        WHERE status = 'closed' AND exit_source = 'manual'
        ORDER BY id
    """)

    if not trades:
        print("No manual-close trades found.")
        return

    sells = pull_webull_sells(csv_path=args.csv)
    if not sells:
        print("No Webull sell orders found.")
        return

    print(f"\n{len(trades)} manual closes, {len(sells)} Webull sells\n")
    print(f"{'ID':>4} {'Ticker':<6} {'Cts':>4} {'Entry':>7} {'DB Exit':>8} {'WB Exit':>8} "
          f"{'Old P&L':>10} {'New P&L':>10} {'Diff':>9} {'Match':>6}")
    print("-" * 90)

    fixes = []
    total_shift = 0

    for t in trades:
        tid = t["id"]
        entry = t["premium_per_contract"] or 0
        cts = t["contracts"]
        old_exit = t["webull_exit_fill_price"] or t["exit_premium"] or 0
        old_pnl = t["pnl_dollars"] or 0

        match = match_sells_to_trade(t, sells)
        if not match:
            print(f"{tid:>4} {t['ticker']:<6} {cts:>4} ${entry:>6.2f} ${old_exit:>7.2f} "
                  f"{'NO MATCH':>8} ${old_pnl:>+9.2f} {'':>10} {'':>9} {'':>6}")
            continue

        wb_exit = match["real_exit_price"]
        new_pnl = (wb_exit - entry) * cts * 100
        new_pnl_pct = ((wb_exit - entry) / entry * 100) if entry > 0 else 0
        diff = new_pnl - old_pnl
        total_shift += diff
        tranche_note = f"{match['tranches']}T" if match["tranches"] > 1 else "1:1"

        print(f"{tid:>4} {t['ticker']:<6} {cts:>4} ${entry:>6.2f} ${old_exit:>7.2f} "
              f"${wb_exit:>7.2f} ${old_pnl:>+9.2f} ${new_pnl:>+9.2f} ${diff:>+8.2f} {tranche_note:>6}")

        if abs(diff) >= 0.50:
            fixes.append({
                "id": tid,
                "wb_exit": wb_exit,
                "new_pnl": round(new_pnl, 2),
                "new_pnl_pct": round(new_pnl_pct, 2),
            })

    print(f"\n{len(fixes)} trades to fix, total P&L shift: ${total_shift:+.2f}")

    if not args.fix:
        print("\nDry run — use --fix to apply.")
        return

    print("\nApplying fixes...")
    for f in fixes:
        exec_droplet(
            f"UPDATE paper_trades SET "
            f"webull_exit_fill_price = {f['wb_exit']}, "
            f"exit_premium = {f['wb_exit']}, "
            f"pnl_dollars = {f['new_pnl']}, "
            f"pnl_pct = {f['new_pnl_pct']} "
            f"WHERE id = {f['id']}"
        )
    print(f"Fixed {len(fixes)} trades.")


if __name__ == "__main__":
    main()
