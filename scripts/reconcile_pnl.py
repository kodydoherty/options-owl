"""Reconcile Webull order history with DB paper_trades — find and fix all P&L discrepancies.

Usage:
    # Dry run — show discrepancies only
    python -m scripts.reconcile_pnl

    # Fix DB records to match Webull reality
    python -m scripts.reconcile_pnl --fix

    # Show details for a specific date
    python -m scripts.reconcile_pnl --date 2026-05-13
"""
import asyncio
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Suppress Webull SDK logs
logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
for name in ("webull", "webull_trade_sdk", "urllib3", "httpx"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).handlers = []
os.environ.setdefault("WEBULL_LOG_LEVEL", "CRITICAL")

from options_owl.execution.webull_executor import WebullExecutor
from options_owl.config.settings import Settings

DB_PATH = Path(os.getenv("JOURNAL_DIR", "journal")) / "owlet-kody" / "raw_messages.db"


async def fetch_webull_orders(w: WebullExecutor) -> list[dict]:
    """Pull all Webull order history from API."""
    acct = w._account_id
    tc = w._trade_client
    all_items = []
    last_cid = None

    for page in range(30):
        if page > 0:
            time.sleep(2)
        kwargs = {"page_size": 100, "start_date": "2026-04-01", "end_date": "2026-05-19"}
        if last_cid:
            kwargs["last_client_order_id"] = last_cid
        try:
            resp = await asyncio.to_thread(
                tc.order_v2.get_order_history, acct, **kwargs
            )
            data = resp.json() if hasattr(resp, "json") else resp
        except Exception as e:
            print(f"  Page {page} error: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        if not isinstance(data, list) or not data:
            break
        all_items.extend(data)
        last_cid = data[-1].get("client_order_id", "")
        print(f"  Page {page}: {len(data)} order groups", file=sys.stderr)
        if len(data) < 100:
            break
    return all_items


def parse_webull_orders(raw_items: list[dict]) -> list[dict]:
    """Parse raw Webull order groups into flat order list."""
    orders = []
    for entry in raw_items:
        cid = entry.get("client_order_id", "")
        for order in entry.get("orders", []):
            status = order.get("status", "")
            if status == "CANCELLED":
                continue
            side = order.get("side", "")
            sym = order.get("symbol", "")
            qty = int(order.get("filled_qty", 0) or 0)
            price = float(order.get("filled_price", 0) or 0)
            if qty == 0 or price == 0:
                continue
            ft = order.get("place_time_at", "")
            day = ft[:10] if ft else "unknown"
            oid = order.get("order_id", "")
            val = qty * price * 100
            orders.append({
                "day": day,
                "side": side,
                "symbol": sym,
                "qty": qty,
                "price": price,
                "value": val,
                "order_id": oid,
                "client_order_id": cid,
                "status": status,
                "place_time": ft,
            })
    return orders


def load_db_trades(db_path: Path) -> list[dict]:
    """Load all closed paper trades that have Webull order IDs."""
    conn = sqlite3.connect(str(db_path))
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
        WHERE webull_order_id IS NOT NULL
        ORDER BY id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_all_db_trades(db_path: Path) -> list[dict]:
    """Load ALL paper trades (including paper-only) for completeness."""
    conn = sqlite3.connect(str(db_path))
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


def match_orders_to_trades(webull_orders: list[dict], db_trades: list[dict]) -> dict:
    """Match Webull orders to DB trades and identify discrepancies."""

    # Index webull orders by order_id and client_order_id
    wb_by_order_id = {}
    wb_by_client_id = defaultdict(list)
    for o in webull_orders:
        wb_by_order_id[o["order_id"]] = o
        if o["client_order_id"]:
            wb_by_client_id[o["client_order_id"]].append(o)

    # Index DB trades by webull_order_id
    db_by_order_id = {}
    db_by_client_id = defaultdict(list)
    for t in db_trades:
        if t["webull_order_id"]:
            db_by_order_id[t["webull_order_id"]] = t
        if t["webull_client_order_id"]:
            db_by_client_id[t["webull_client_order_id"]].append(t)

    results = {
        "matched": [],       # DB trade + Webull BUY + Webull SELL matched
        "db_no_wb_buy": [],  # DB has webull_order_id but it's not in Webull history
        "db_no_wb_sell": [], # DB has exit but no matching Webull sell order
        "wb_no_db": [],      # Webull has orders not in DB (manual trades)
        "pnl_mismatches": [],  # matched but P&L doesn't agree
        "fixes": [],         # proposed DB fixes
    }

    # Track which Webull orders are matched
    matched_wb_ids = set()

    for trade in db_trades:
        if trade["status"] != "closed":
            continue

        entry_oid = trade["webull_order_id"]
        entry_cid = trade["webull_client_order_id"]
        exit_oid = trade.get("webull_exit_order_id")

        # Find the BUY order in Webull
        wb_buy = None
        if entry_oid and entry_oid in wb_by_order_id:
            wb_buy = wb_by_order_id[entry_oid]
        elif entry_cid and entry_cid in wb_by_client_id:
            buys = [o for o in wb_by_client_id[entry_cid] if o["side"] == "BUY"]
            if buys:
                wb_buy = buys[0]

        if not wb_buy:
            results["db_no_wb_buy"].append(trade)
            continue

        matched_wb_ids.add(wb_buy["order_id"])

        # Find the SELL order in Webull
        wb_sell = None
        if exit_oid and exit_oid in wb_by_order_id:
            wb_sell = wb_by_order_id[exit_oid]
            matched_wb_ids.add(wb_sell["order_id"])
        elif entry_cid and entry_cid in wb_by_client_id:
            sells = [o for o in wb_by_client_id[entry_cid] if o["side"] == "SELL"]
            if sells:
                wb_sell = sells[0]
                matched_wb_ids.add(wb_sell["order_id"])

        # Calculate Webull P&L
        wb_entry_price = wb_buy["price"]
        wb_entry_qty = wb_buy["qty"]
        wb_entry_value = wb_entry_price * wb_entry_qty * 100

        wb_exit_price = wb_sell["price"] if wb_sell else None
        wb_exit_qty = wb_sell["qty"] if wb_sell else None
        wb_exit_value = (wb_exit_price * wb_exit_qty * 100) if wb_sell else None
        wb_pnl = (wb_exit_value - wb_entry_value) if wb_sell else None

        db_pnl = trade["pnl_dollars"] or 0

        match_info = {
            "trade": trade,
            "wb_buy": wb_buy,
            "wb_sell": wb_sell,
            "wb_entry_price": wb_entry_price,
            "wb_entry_qty": wb_entry_qty,
            "wb_exit_price": wb_exit_price,
            "wb_exit_qty": wb_exit_qty,
            "wb_pnl": wb_pnl,
            "db_pnl": db_pnl,
            "pnl_diff": (wb_pnl - db_pnl) if wb_pnl is not None else None,
        }

        results["matched"].append(match_info)

        # Check for P&L mismatch (tolerance $1)
        if wb_pnl is not None and abs(wb_pnl - db_pnl) > 1.0:
            match_info["discrepancy_reasons"] = diagnose_discrepancy(trade, wb_buy, wb_sell)
            results["pnl_mismatches"].append(match_info)
            # Propose fix
            results["fixes"].append({
                "trade_id": trade["id"],
                "ticker": trade["ticker"],
                "old_pnl": db_pnl,
                "new_pnl": wb_pnl,
                "wb_entry_price": wb_entry_price,
                "wb_exit_price": wb_exit_price,
                "wb_entry_qty": wb_entry_qty,
                "wb_exit_qty": wb_exit_qty,
                "reasons": match_info["discrepancy_reasons"],
            })

        if not wb_sell:
            results["db_no_wb_sell"].append(match_info)

    # Find Webull orders not matched to any DB trade
    for o in webull_orders:
        if o["order_id"] not in matched_wb_ids:
            results["wb_no_db"].append(o)

    return results


def diagnose_discrepancy(trade: dict, wb_buy: dict, wb_sell: dict | None) -> list[str]:
    """Diagnose why DB P&L differs from Webull P&L."""
    reasons = []

    db_entry = trade.get("webull_entry_fill_price") or 0
    db_exit = trade.get("webull_exit_fill_price") or trade.get("exit_premium") or 0
    db_contracts = trade["contracts"]
    db_ppc = trade["premium_per_contract"] or 0
    dca_qty = trade.get("dca_total_contracts") or 0

    wb_entry = wb_buy["price"]
    wb_exit = wb_sell["price"] if wb_sell else None
    wb_buy_qty = wb_buy["qty"]
    wb_sell_qty = wb_sell["qty"] if wb_sell else None

    # Entry price mismatch
    if db_entry and abs(db_entry - wb_entry) > 0.01:
        reasons.append(f"entry_price: DB=${db_entry:.2f} vs WB=${wb_entry:.2f}")

    # premium_per_contract vs actual fill
    if db_ppc and abs(db_ppc - wb_entry) > 0.01 and dca_qty == 0:
        reasons.append(f"premium_per_contract: DB=${db_ppc:.2f} vs WB=${wb_entry:.2f}")

    # Exit price mismatch
    if wb_exit and db_exit and abs(db_exit - wb_exit) > 0.01:
        reasons.append(f"exit_price: DB=${db_exit:.2f} vs WB=${wb_exit:.2f}")

    # Contract count mismatch (could be scaleout)
    if wb_buy_qty != db_contracts:
        reasons.append(f"entry_qty: DB={db_contracts} vs WB={wb_buy_qty}")

    if wb_sell and wb_sell_qty != db_contracts:
        reasons.append(f"exit_qty: DB={db_contracts} vs WB={wb_sell_qty}")

    # DCA trades often have issues
    if dca_qty and dca_qty > 0:
        reasons.append(f"DCA trade (total_target={dca_qty})")

    # Slippage applied
    slippage = trade.get("exit_slippage")
    if slippage and slippage > 0:
        reasons.append(f"simulated_slippage=${slippage:.4f} deducted from exit")

    # Missing exit fill
    if not trade.get("webull_exit_fill_price"):
        reasons.append("no webull_exit_fill_price — using simulated exit_premium")

    # Missing entry fill
    if not trade.get("webull_entry_fill_price"):
        reasons.append("no webull_entry_fill_price — using signal premium for entry")

    # Manual close
    if trade.get("exit_source") == "manual":
        reasons.append("manual close — exit_premium is approximate")

    if not reasons:
        reasons.append("unknown — all fields seem to match")

    return reasons


def apply_fixes(db_path: Path, fixes: list[dict]) -> int:
    """Apply P&L fixes to the DB based on Webull order history."""
    conn = sqlite3.connect(str(db_path))
    fixed = 0
    for fix in fixes:
        tid = fix["trade_id"]
        new_pnl = fix["new_pnl"]
        wb_entry = fix["wb_entry_price"]
        wb_exit = fix["wb_exit_price"]
        wb_entry_qty = fix["wb_entry_qty"]
        wb_exit_qty = fix["wb_exit_qty"]

        if wb_exit is None:
            # Can't fix without exit price
            continue

        # Use Webull quantities for P&L calc
        # For scaleout: buy may be N, sell may be N-1 (partial already sold)
        # P&L = (exit * sell_qty - entry * buy_qty) * 100
        # Actually for a matched trade: pnl = (exit - entry) * min(buy, sell) * 100
        # But typically they should be equal. Let's use the sell qty.
        qty = min(wb_entry_qty, wb_exit_qty) if wb_exit_qty else wb_entry_qty
        new_pnl_calc = (wb_exit - wb_entry) * qty * 100
        pnl_pct = ((wb_exit - wb_entry) / wb_entry * 100) if wb_entry > 0 else 0

        conn.execute(
            "UPDATE paper_trades SET "
            "pnl_dollars = ?, pnl_pct = ?, "
            "webull_entry_fill_price = ?, webull_exit_fill_price = ? "
            "WHERE id = ?",
            (new_pnl_calc, pnl_pct, wb_entry, wb_exit, tid),
        )
        fixed += 1

    conn.commit()
    conn.close()
    return fixed


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reconcile Webull orders vs DB trades")
    parser.add_argument("--fix", action="store_true", help="Apply fixes to DB")
    parser.add_argument("--date", type=str, help="Show details for specific date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all matched trades")
    args = parser.parse_args()

    print("Initializing Webull connection...", file=sys.stderr)
    s = Settings()
    w = WebullExecutor(s)
    await w.init()

    print("Fetching Webull order history...", file=sys.stderr)
    raw_items = await fetch_webull_orders(w)
    webull_orders = parse_webull_orders(raw_items)
    print(f"  Total Webull filled orders: {len(webull_orders)}", file=sys.stderr)

    print("Loading DB trades...", file=sys.stderr)
    db_trades = load_db_trades(DB_PATH)
    all_db_trades = load_all_db_trades(DB_PATH)
    print(f"  DB trades with Webull order ID: {len(db_trades)}", file=sys.stderr)
    print(f"  Total DB trades: {len(all_db_trades)}", file=sys.stderr)

    print("\nMatching orders to trades...", file=sys.stderr)
    results = match_orders_to_trades(webull_orders, db_trades)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("RECONCILIATION SUMMARY")
    print("=" * 80)

    print(f"\nMatched trades:        {len(results['matched'])}")
    print(f"P&L mismatches:        {len(results['pnl_mismatches'])}")
    print(f"DB has no WB buy:      {len(results['db_no_wb_buy'])}")
    print(f"DB has no WB sell:     {len(results['db_no_wb_sell'])}")
    print(f"WB orders not in DB:   {len(results['wb_no_db'])}")

    # ── P&L totals ──
    total_db_pnl = sum(m["db_pnl"] for m in results["matched"])
    total_wb_pnl = sum(m["wb_pnl"] for m in results["matched"] if m["wb_pnl"] is not None)
    matched_with_sell = [m for m in results["matched"] if m["wb_sell"]]

    print(f"\nP&L Totals (matched trades with both buy+sell):")
    print(f"  DB total:     ${total_db_pnl:+,.2f}")
    print(f"  Webull total: ${total_wb_pnl:+,.2f}")
    print(f"  Difference:   ${total_wb_pnl - total_db_pnl:+,.2f}")

    # ── Unmatched Webull orders (manual trades) ──
    if results["wb_no_db"]:
        print(f"\n{'─' * 80}")
        print("WEBULL ORDERS NOT IN DB (manual trades or missing order IDs):")
        print(f"{'─' * 80}")
        wb_unmatched_pnl = 0
        # Group by client_order_id to pair buys and sells
        by_cid = defaultdict(list)
        no_cid = []
        for o in results["wb_no_db"]:
            if o["client_order_id"]:
                by_cid[o["client_order_id"]].append(o)
            else:
                no_cid.append(o)

        for cid, orders in sorted(by_cid.items()):
            buys = [o for o in orders if o["side"] == "BUY"]
            sells = [o for o in orders if o["side"] == "SELL"]
            for b in buys:
                matching_sell = None
                for s in sells:
                    if s["symbol"] == b["symbol"]:
                        matching_sell = s
                        break
                if matching_sell:
                    pnl = (matching_sell["price"] - b["price"]) * min(b["qty"], matching_sell["qty"]) * 100
                    wb_unmatched_pnl += pnl
                    print(f"  {b['day']} {b['symbol']} BUY x{b['qty']} @${b['price']:.2f} → "
                          f"SELL x{matching_sell['qty']} @${matching_sell['price']:.2f} "
                          f"= ${pnl:+,.2f}")
                    sells.remove(matching_sell)
                else:
                    print(f"  {b['day']} {b['symbol']} BUY x{b['qty']} @${b['price']:.2f} "
                          f"(no matching sell)")
            for s in sells:
                print(f"  {s['day']} {s['symbol']} SELL x{s['qty']} @${s['price']:.2f} "
                      f"(no matching buy in unmatched — may be exit of DB trade)")

        for o in no_cid:
            print(f"  {o['day']} {o['symbol']} {o['side']} x{o['qty']} @${o['price']:.2f} "
                  f"(no client_order_id)")

        print(f"\n  Unmatched Webull P&L: ${wb_unmatched_pnl:+,.2f}")

    # ── P&L mismatches ──
    if results["pnl_mismatches"]:
        print(f"\n{'─' * 80}")
        print("P&L MISMATCHES (DB vs Webull):")
        print(f"{'─' * 80}")
        for m in sorted(results["pnl_mismatches"], key=lambda x: abs(x["pnl_diff"] or 0), reverse=True):
            t = m["trade"]
            date_str = (t["opened_at"] or "")[:10]
            print(f"\n  #{t['id']} {date_str} {t['ticker']} {t['option_type'].upper()} ${t['strike']}")
            print(f"    DB P&L:  ${m['db_pnl']:+,.2f}  |  WB P&L: ${m['wb_pnl']:+,.2f}  |  Diff: ${m['pnl_diff']:+,.2f}")
            print(f"    WB entry: ${m['wb_entry_price']:.2f} x{m['wb_entry_qty']}  →  "
                  f"WB exit: ${m['wb_exit_price']:.2f} x{m['wb_exit_qty']}" if m['wb_exit_price'] else
                  f"    WB entry: ${m['wb_entry_price']:.2f} x{m['wb_entry_qty']}  →  NO WB EXIT")
            print(f"    DB entry fill: ${t.get('webull_entry_fill_price') or 0:.2f}  |  "
                  f"DB exit fill: ${t.get('webull_exit_fill_price') or 0:.2f}  |  "
                  f"DB premium: ${t['premium_per_contract']:.2f}")
            for reason in m.get("discrepancy_reasons", []):
                print(f"    → {reason}")

    # ── By-day summary ──
    print(f"\n{'─' * 80}")
    print("DAY-BY-DAY P&L COMPARISON:")
    print(f"{'─' * 80}")
    print(f"  {'Date':<12} {'DB P&L':>10} {'WB P&L':>10} {'Diff':>10} {'Trades':>7}")

    day_db = defaultdict(float)
    day_wb = defaultdict(float)
    day_count = defaultdict(int)

    for m in results["matched"]:
        date_str = (m["trade"]["opened_at"] or "")[:10]
        day_db[date_str] += m["db_pnl"]
        if m["wb_pnl"] is not None:
            day_wb[date_str] += m["wb_pnl"]
        day_count[date_str] += 1

    all_dates = sorted(set(day_db.keys()) | set(day_wb.keys()))
    for d in all_dates:
        db_val = day_db.get(d, 0)
        wb_val = day_wb.get(d, 0)
        diff = wb_val - db_val
        cnt = day_count.get(d, 0)
        marker = " ⚠️" if abs(diff) > 5 else ""
        print(f"  {d:<12} ${db_val:>+9,.2f} ${wb_val:>+9,.2f} ${diff:>+9,.2f} {cnt:>6}{marker}")

    print(f"  {'─' * 55}")
    total_diff = sum(day_wb.get(d, 0) - day_db.get(d, 0) for d in all_dates)
    print(f"  {'TOTAL':<12} ${sum(day_db.values()):>+9,.2f} ${sum(day_wb.values()):>+9,.2f} ${total_diff:>+9,.2f}")

    # ── Date detail ──
    if args.date:
        print(f"\n{'─' * 80}")
        print(f"DETAIL FOR {args.date}:")
        print(f"{'─' * 80}")
        for m in results["matched"]:
            date_str = (m["trade"]["opened_at"] or "")[:10]
            if date_str == args.date:
                t = m["trade"]
                print(f"  #{t['id']} {t['ticker']} {t['option_type'].upper()} ${t['strike']} "
                      f"x{t['contracts']} | DB=${m['db_pnl']:+,.2f} WB=${m['wb_pnl']:+,.2f}" +
                      (f" DIFF=${m['pnl_diff']:+,.2f}" if m['pnl_diff'] and abs(m['pnl_diff']) > 1 else ""))

    # ── Verbose ──
    if args.verbose:
        print(f"\n{'─' * 80}")
        print("ALL MATCHED TRADES:")
        print(f"{'─' * 80}")
        for m in results["matched"]:
            t = m["trade"]
            date_str = (t["opened_at"] or "")[:10]
            diff_str = f" DIFF=${m['pnl_diff']:+,.2f}" if m['pnl_diff'] and abs(m['pnl_diff']) > 1 else ""
            print(f"  #{t['id']:<4} {date_str} {t['ticker']:<6} {t['option_type'].upper():<4} "
                  f"${t['strike']:<8} x{t['contracts']} | "
                  f"DB=${m['db_pnl']:>+8,.2f} WB=${m['wb_pnl']:>+8,.2f if m['wb_pnl'] is not None else 'N/A':>8}{diff_str}")

    # ── Apply fixes ──
    if args.fix and results["fixes"]:
        print(f"\n{'─' * 80}")
        print(f"APPLYING {len(results['fixes'])} FIXES...")
        print(f"{'─' * 80}")
        fixed = apply_fixes(DB_PATH, results["fixes"])
        print(f"  Fixed {fixed} trades in DB.")
        print("  Re-run without --fix to verify.")
    elif results["fixes"]:
        print(f"\n  {len(results['fixes'])} trades need P&L correction. Run with --fix to apply.")


asyncio.run(main())
