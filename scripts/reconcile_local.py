"""Local reconciliation: compare webull_orders.csv with DB paper_trades.

Usage:
    python scripts/reconcile_local.py                    # dry run
    python scripts/reconcile_local.py --fix              # apply P&L fixes to DB
    python scripts/reconcile_local.py --date 2026-05-13  # detail for one date
    python scripts/reconcile_local.py -v                 # verbose all trades
"""
import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("/tmp/webull_orders.csv")
DB_PATH = Path("journal/owlet-kody/raw_messages.db")


def load_webull_csv(path: Path) -> list[dict]:
    orders = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        row["qty"] = int(row["qty"])
        row["price"] = float(row["price"])
        row["value"] = float(row["value"])
        row["strike"] = float(row["strike"]) if row.get("strike") else 0
        orders.append(row)
    return orders


def load_db_trades(path: Path) -> list[dict]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, score, strike, option_type, contracts,
               premium_per_contract, total_cost, exit_premium, exit_reason,
               pnl_dollars, pnl_pct, opened_at, closed_at, status,
               webull_order_id, webull_client_order_id,
               webull_entry_fill_price, webull_exit_fill_price,
               webull_exit_order_id, exit_source,
               dca_total_contracts, dca_tranches_remaining,
               signal_premium, expiry_date
        FROM paper_trades
        ORDER BY id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def group_webull_trades(wb_orders: list[dict]) -> list[dict]:
    """Group Webull orders into logical trades (buy+sell groups per symbol/strike/expiry/day).

    A 'trade' is all BUY and SELL orders for the same symbol+strike+expiry on the same day.
    This handles DCA (multiple buys) and scaleout (multiple sells) correctly.
    """
    # Key: (day, symbol, option_type, strike, expiry) → list of orders
    groups = defaultdict(list)
    for o in wb_orders:
        key = (o["day"], o["symbol"], o.get("option_type", ""), o.get("strike", 0), o.get("expiry", ""))
        groups[key].append(o)

    trades = []
    for (day, symbol, opt_type, strike, expiry), orders in groups.items():
        buys = [o for o in orders if o["side"] == "BUY"]
        sells = [o for o in orders if o["side"] == "SELL"]

        total_buy_qty = sum(b["qty"] for b in buys)
        total_buy_cost = sum(b["qty"] * b["price"] * 100 for b in buys)
        avg_buy_price = (total_buy_cost / (total_buy_qty * 100)) if total_buy_qty > 0 else 0

        total_sell_qty = sum(s["qty"] for s in sells)
        total_sell_proceeds = sum(s["qty"] * s["price"] * 100 for s in sells)
        avg_sell_price = (total_sell_proceeds / (total_sell_qty * 100)) if total_sell_qty > 0 else 0

        # P&L = proceeds - cost (for matched qty)
        matched_qty = min(total_buy_qty, total_sell_qty)
        # More accurate: use actual cost basis and actual proceeds for matched quantities
        pnl = total_sell_proceeds - total_buy_cost if total_buy_qty > 0 and total_sell_qty > 0 else None

        trades.append({
            "day": day,
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "buys": buys,
            "sells": sells,
            "total_buy_qty": total_buy_qty,
            "total_sell_qty": total_sell_qty,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "total_buy_cost": total_buy_cost,
            "total_sell_proceeds": total_sell_proceeds,
            "pnl": pnl,
            "buy_order_ids": {b["order_id"] for b in buys},
            "sell_order_ids": {s["order_id"] for s in sells},
            "all_order_ids": {o["order_id"] for o in orders},
            "all_client_ids": {o["client_order_id"] for o in orders if o.get("client_order_id")},
        })

    return trades


def match_trades(wb_trades: list[dict], db_trades: list[dict]) -> dict:
    """Match grouped Webull trades to DB trades.

    When multiple DB trades map to the same Webull trade group (e.g. scaleout
    children), the Webull P&L is split proportionally by contract count.
    """

    # Index Webull trades by order_id and client_id for quick lookup
    wb_by_oid = {}  # order_id → wb_trade
    wb_by_cid = {}  # client_order_id → wb_trade
    for wt in wb_trades:
        for oid in wt["all_order_ids"]:
            wb_by_oid[oid] = wt
        for cid in wt["all_client_ids"]:
            wb_by_cid[cid] = wt

    matched_wb_trades = set()  # track by id(wb_trade)
    # First pass: find which Webull group each DB trade maps to
    db_to_wt = {}  # trade_id → wb_trade
    wt_to_db = defaultdict(list)  # id(wb_trade) → [db_trades]

    for trade in db_trades:
        if not trade["webull_order_id"]:
            continue
        if trade["status"] != "closed":
            continue

        oid = trade["webull_order_id"]
        cid = trade.get("webull_client_order_id")
        exit_oid = trade.get("webull_exit_order_id")

        wt = wb_by_oid.get(oid)
        if not wt and cid:
            wt = wb_by_cid.get(cid)
        if not wt and exit_oid:
            wt = wb_by_oid.get(exit_oid)

        if wt:
            db_to_wt[trade["id"]] = wt
            wt_to_db[id(wt)].append(trade)
            matched_wb_trades.add(id(wt))

    results = {
        "matched": [],
        "pnl_mismatches": [],
        "db_no_wb": [],
        "wb_unmatched": [],
        "fixes": [],
    }

    for trade in db_trades:
        if not trade["webull_order_id"]:
            continue
        if trade["status"] != "closed":
            continue

        wt = db_to_wt.get(trade["id"])
        if not wt:
            results["db_no_wb"].append(trade)
            continue

        # Split Webull P&L proportionally when multiple DB trades share the group
        siblings = wt_to_db[id(wt)]
        total_db_contracts = sum(s["contracts"] for s in siblings)
        my_fraction = trade["contracts"] / total_db_contracts if total_db_contracts > 0 else 1.0

        db_pnl = trade["pnl_dollars"] or 0
        if wt["pnl"] is not None:
            wb_pnl_share = wt["pnl"] * my_fraction
        else:
            wb_pnl_share = None

        info = {
            "trade": trade,
            "wb_trade": wt,
            "db_pnl": db_pnl,
            "wb_pnl": wb_pnl_share,
            "wb_pnl_total": wt["pnl"],
            "pnl_diff": (wb_pnl_share - db_pnl) if wb_pnl_share is not None else None,
            "fraction": my_fraction,
            "num_siblings": len(siblings),
        }
        results["matched"].append(info)

        if wb_pnl_share is not None and abs(wb_pnl_share - db_pnl) > 1.0:
            info["reasons"] = diagnose(trade, wt)
            results["pnl_mismatches"].append(info)
            results["fixes"].append({
                "trade_id": trade["id"],
                "ticker": trade["ticker"],
                "old_pnl": db_pnl,
                "new_pnl": wb_pnl_share,
                "wb_avg_entry": wt["avg_buy_price"],
                "wb_avg_exit": wt["avg_sell_price"],
                "wb_buy_qty": wt["total_buy_qty"],
                "wb_sell_qty": wt["total_sell_qty"],
                "wb_buy_cost": wt["total_buy_cost"],
                "wb_sell_proceeds": wt["total_sell_proceeds"],
                "fraction": my_fraction,
            })

    # Unmatched Webull trades
    for wt in wb_trades:
        if id(wt) not in matched_wb_trades:
            results["wb_unmatched"].append(wt)

    return results


def diagnose(trade: dict, wt: dict) -> list[str]:
    reasons = []
    db_entry = trade.get("webull_entry_fill_price") or 0
    db_exit = trade.get("webull_exit_fill_price") or trade.get("exit_premium") or 0
    db_contracts = trade["contracts"]
    dca_qty = trade.get("dca_total_contracts") or 0

    if db_entry and abs(db_entry - wt["avg_buy_price"]) > 0.01:
        reasons.append(f"entry_price: DB=${db_entry:.2f} vs WB_avg=${wt['avg_buy_price']:.2f}")
    if wt["avg_sell_price"] and db_exit and abs(db_exit - wt["avg_sell_price"]) > 0.01:
        reasons.append(f"exit_price: DB=${db_exit:.2f} vs WB_avg=${wt['avg_sell_price']:.2f}")
    if wt["total_buy_qty"] != db_contracts:
        reasons.append(f"buy_qty: DB={db_contracts} vs WB={wt['total_buy_qty']} "
                       f"({len(wt['buys'])} buy orders)")
    if wt["total_sell_qty"] != db_contracts:
        reasons.append(f"sell_qty: DB={db_contracts} vs WB={wt['total_sell_qty']} "
                       f"({len(wt['sells'])} sell orders)")
    if dca_qty:
        reasons.append(f"DCA trade (target={dca_qty}, {len(wt['buys'])} buys in WB)")
    if not trade.get("webull_exit_fill_price"):
        reasons.append("no webull_exit_fill_price — simulated exit used")
    if trade.get("exit_source") == "manual":
        reasons.append("manual close")
    if not reasons:
        reasons.append("rounding/slippage")
    return reasons


def apply_fixes(db_path: Path, fixes: list[dict]) -> int:
    """Update DB P&L from Webull grouped trade data (proportionally split)."""
    conn = sqlite3.connect(str(db_path))
    n = 0
    for fix in fixes:
        if fix["wb_sell_qty"] == 0:
            continue
        pnl = fix["new_pnl"]  # already proportionally split
        pct = ((fix["wb_avg_exit"] - fix["wb_avg_entry"]) / fix["wb_avg_entry"] * 100) if fix["wb_avg_entry"] > 0 else 0
        conn.execute(
            "UPDATE paper_trades SET pnl_dollars = ?, pnl_pct = ?, "
            "webull_entry_fill_price = ?, webull_exit_fill_price = ? "
            "WHERE id = ?",
            (pnl, pct, fix["wb_avg_entry"], fix["wb_avg_exit"], fix["trade_id"]),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--date", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    wb_orders = load_webull_csv(CSV_PATH)
    db_trades = load_db_trades(DB_PATH)
    webull_db = [t for t in db_trades if t["webull_order_id"]]
    paper_db = [t for t in db_trades if not t["webull_order_id"]]

    print(f"Loaded {len(wb_orders)} Webull orders, {len(webull_db)} DB Webull trades, {len(paper_db)} paper-only")

    # Group Webull orders into trades
    wb_trades = group_webull_trades(wb_orders)
    print(f"Grouped into {len(wb_trades)} Webull trade groups")

    results = match_trades(wb_trades, db_trades)

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("RECONCILIATION SUMMARY (grouped matching)")
    print(f"{'=' * 80}")
    print(f"Matched trades:          {len(results['matched'])}")
    print(f"P&L mismatches (>$1):    {len(results['pnl_mismatches'])}")
    print(f"DB trades w/o WB match:  {len(results['db_no_wb'])}")
    print(f"WB trade groups w/o DB:  {len(results['wb_unmatched'])}")

    # ── P&L totals ──
    matched_with_pnl = [m for m in results["matched"] if m["wb_pnl"] is not None]
    total_db = sum(m["db_pnl"] for m in matched_with_pnl)
    total_wb = sum(m["wb_pnl"] for m in matched_with_pnl)
    print(f"\nP&L for matched trades:")
    print(f"  DB total:     ${total_db:>+10,.2f}")
    print(f"  Webull total: ${total_wb:>+10,.2f}")
    print(f"  Difference:   ${total_wb - total_db:>+10,.2f}")

    # ── Unmatched Webull ──
    if results["wb_unmatched"]:
        print(f"\n{'─' * 80}")
        print(f"WEBULL TRADES NOT IN DB ({len(results['wb_unmatched'])} groups):")
        print(f"{'─' * 80}")
        unmatched_pnl = 0
        for wt in sorted(results["wb_unmatched"], key=lambda x: x["day"]):
            pnl_str = f"${wt['pnl']:+,.2f}" if wt["pnl"] is not None else "open"
            if wt["pnl"] is not None:
                unmatched_pnl += wt["pnl"]
            print(f"  {wt['day']} {wt['symbol']:<6} ${wt['strike']:<8} "
                  f"BUY x{wt['total_buy_qty']} avg@${wt['avg_buy_price']:.2f} → "
                  f"SELL x{wt['total_sell_qty']} avg@${wt['avg_sell_price']:.2f} "
                  f"= {pnl_str}")
        print(f"\n  Unmatched WB P&L: ${unmatched_pnl:+,.2f}")

    # ── DB no match ──
    if results["db_no_wb"]:
        print(f"\n{'─' * 80}")
        print(f"DB TRADES NO WEBULL MATCH ({len(results['db_no_wb'])}):")
        print(f"{'─' * 80}")
        for t in results["db_no_wb"]:
            print(f"  #{t['id']:<4} {(t['opened_at'] or '')[:10]} {t['ticker']:<6} "
                  f"oid={t['webull_order_id']} pnl=${t['pnl_dollars'] or 0:+,.2f}")

    # ── Mismatches ──
    if results["pnl_mismatches"]:
        print(f"\n{'─' * 80}")
        print(f"P&L MISMATCHES ({len(results['pnl_mismatches'])} trades):")
        print(f"{'─' * 80}")
        for m in sorted(results["pnl_mismatches"], key=lambda x: abs(x["pnl_diff"] or 0), reverse=True):
            t = m["trade"]
            wt = m["wb_trade"]
            print(f"\n  #{t['id']:<4} {(t['opened_at'] or '')[:10]} {t['ticker']:<6} "
                  f"{t['option_type'].upper():<4} ${t['strike']}")
            print(f"    DB: ${m['db_pnl']:>+9,.2f}  WB: ${m['wb_pnl']:>+9,.2f}  "
                  f"Diff: ${m['pnl_diff']:>+9,.2f}")
            print(f"    WB: {len(wt['buys'])} buys (x{wt['total_buy_qty']} avg@${wt['avg_buy_price']:.2f} "
                  f"= ${wt['total_buy_cost']:.2f}) → "
                  f"{len(wt['sells'])} sells (x{wt['total_sell_qty']} avg@${wt['avg_sell_price']:.2f} "
                  f"= ${wt['total_sell_proceeds']:.2f})")
            print(f"    DB: x{t['contracts']} entry=${t.get('webull_entry_fill_price') or 0:.2f} "
                  f"exit=${t.get('webull_exit_fill_price') or 0:.2f} "
                  f"premium=${t['premium_per_contract']:.2f}")
            for r in m.get("reasons", []):
                print(f"    → {r}")

    # ── Day-by-day ──
    print(f"\n{'─' * 80}")
    print("DAY-BY-DAY P&L:")
    print(f"{'─' * 80}")
    print(f"  {'Date':<12} {'DB P&L':>10} {'WB P&L':>10} {'Diff':>10} {'#':>4}")

    day_db = defaultdict(float)
    day_wb = defaultdict(float)
    day_n = defaultdict(int)
    for m in results["matched"]:
        d = (m["trade"]["opened_at"] or "")[:10]
        day_db[d] += m["db_pnl"]
        if m["wb_pnl"] is not None:
            day_wb[d] += m["wb_pnl"]
        day_n[d] += 1

    for d in sorted(day_db.keys()):
        db = day_db[d]
        wb = day_wb.get(d, 0)
        diff = wb - db
        flag = " ***" if abs(diff) > 10 else ""
        print(f"  {d:<12} ${db:>+9,.2f} ${wb:>+9,.2f} ${diff:>+9,.2f} {day_n[d]:>4}{flag}")

    print(f"  {'─' * 50}")
    t_db = sum(day_db.values())
    t_wb = sum(day_wb.values())
    print(f"  {'TOTAL':<12} ${t_db:>+9,.2f} ${t_wb:>+9,.2f} ${t_wb - t_db:>+9,.2f}")

    # ── Date detail ──
    if args.date:
        print(f"\n{'─' * 80}")
        print(f"DETAIL FOR {args.date}:")
        print(f"{'─' * 80}")
        for m in results["matched"]:
            d = (m["trade"]["opened_at"] or "")[:10]
            if d == args.date:
                t = m["trade"]
                wb_str = f"${m['wb_pnl']:+,.2f}" if m["wb_pnl"] is not None else "N/A"
                diff_str = f" D=${m['pnl_diff']:+,.2f}" if m["pnl_diff"] and abs(m["pnl_diff"]) > 1 else ""
                print(f"  #{t['id']:<4} {t['ticker']:<6} {t['option_type'].upper():<4} ${t['strike']:<8} "
                      f"x{t['contracts']} | DB=${m['db_pnl']:>+8,.2f} WB={wb_str:>9}{diff_str}")

    # ── Verbose ──
    if args.verbose:
        print(f"\n{'─' * 80}")
        print("ALL MATCHED:")
        print(f"{'─' * 80}")
        for m in results["matched"]:
            t = m["trade"]
            d = (t["opened_at"] or "")[:10]
            wb_str = f"${m['wb_pnl']:>+8,.2f}" if m["wb_pnl"] is not None else "     N/A"
            diff = f" D=${m['pnl_diff']:+,.2f}" if m["pnl_diff"] and abs(m["pnl_diff"]) > 1 else ""
            print(f"  #{t['id']:<4} {d} {t['ticker']:<6} x{t['contracts']} "
                  f"DB=${m['db_pnl']:>+8,.2f} WB={wb_str}{diff}")

    # ── Fix ──
    if args.fix and results["fixes"]:
        print(f"\n{'─' * 80}")
        print(f"APPLYING {len(results['fixes'])} FIXES...")
        n = apply_fixes(DB_PATH, results["fixes"])
        print(f"Fixed {n} trades.")
        print("Re-run without --fix to verify.")
    elif results["fixes"]:
        print(f"\n  → {len(results['fixes'])} trades need correction. Run with --fix to apply.")


if __name__ == "__main__":
    main()
