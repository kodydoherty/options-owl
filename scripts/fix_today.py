"""Fix today's trades by pulling ALL Webull fills and reconciling with DB.

Tries v3 API, v2 API, and v2 with start/end dates. Uses 5s delays between pages.
"""
import asyncio
import json
import sqlite3
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_DATE = "2026-05-12"
DELAY = 5


async def try_v3_history(w):
    """Try order_v3.get_order_history which uses only last_client_order_id for paging."""
    all_fills = []
    last_coid = None

    for page in range(20):
        if page > 0:
            time.sleep(DELAY)

        for attempt in range(3):
            try:
                kwargs = {"account_id": w._account_id, "page_size": 10}
                if last_coid:
                    kwargs["last_client_order_id"] = last_coid
                response = await asyncio.to_thread(
                    w._trade_client.order_v3.get_order_history, **kwargs)
                result = response.json() if hasattr(response, "json") else response
                orders = (result if isinstance(result, list)
                          else result.get("orders", result.get("data", [])))
                break
            except Exception as e:
                print(f"    v3 page {page+1} attempt {attempt+1}: {str(e)[:80]}")
                if attempt < 2:
                    time.sleep(DELAY)
                orders = []

        if not orders:
            break

        for combo in orders:
            for o in combo.get("orders", []):
                placed = o.get("place_time_at", "")
                if not placed.startswith(TARGET_DATE):
                    continue
                if o.get("status") != "FILLED":
                    continue
                legs = o.get("legs", [{}])
                all_fills.append({
                    "oid": o.get("order_id", ""),
                    "sym": o.get("symbol"),
                    "side": o.get("side"),
                    "qty": float(o.get("filled_quantity", 0)),
                    "fill": float(o.get("filled_price", 0)),
                    "placed": placed,
                    "otype": (legs[0].get("option_type", "").lower() if legs else ""),
                    "strike": float(legs[0].get("strike_price", 0)) if legs else 0,
                })

        last_combo = orders[-1]
        last_coid = last_combo.get("client_order_id")
        print(f"  v3 page {page+1}: {len(orders)} combos, total fills: {len(all_fills)}")

        if len(orders) < 10:
            break

    return all_fills


async def try_v2_history(w):
    """Try order_v2 with just last_client_order_id (not last_order_id)."""
    all_fills = []
    last_coid = None

    for page in range(20):
        if page > 0:
            time.sleep(DELAY)

        for attempt in range(3):
            try:
                kwargs = {"account_id": w._account_id}
                if last_coid:
                    kwargs["last_client_order_id"] = last_coid
                response = await asyncio.to_thread(
                    w._trade_client.order_v2.get_order_history, **kwargs)
                result = response.json() if hasattr(response, "json") else result
                orders = (result if isinstance(result, list)
                          else result.get("orders", result.get("data", [])))
                break
            except Exception as e:
                print(f"    v2 page {page+1} attempt {attempt+1}: {str(e)[:80]}")
                if attempt < 2:
                    time.sleep(DELAY)
                orders = []

        if not orders:
            break

        for combo in orders:
            for o in combo.get("orders", []):
                placed = o.get("place_time_at", "")
                if not placed.startswith(TARGET_DATE):
                    continue
                if o.get("status") != "FILLED":
                    continue
                legs = o.get("legs", [{}])
                all_fills.append({
                    "oid": o.get("order_id", ""),
                    "sym": o.get("symbol"),
                    "side": o.get("side"),
                    "qty": float(o.get("filled_quantity", 0)),
                    "fill": float(o.get("filled_price", 0)),
                    "placed": placed,
                    "otype": (legs[0].get("option_type", "").lower() if legs else ""),
                    "strike": float(legs[0].get("strike_price", 0)) if legs else 0,
                })

        last_combo = orders[-1]
        last_coid = last_combo.get("client_order_id")
        print(f"  v2 page {page+1}: {len(orders)} combos, total fills: {len(all_fills)}")

        if len(orders) < 10:
            break

    return all_fills


async def main():
    from options_owl.config.settings import Settings
    from options_owl.execution.webull_executor import WebullExecutor

    s = Settings()
    w = WebullExecutor(s)
    await w.init()

    # Try v3 first
    print("=== Trying v3 API ===")
    fills = await try_v3_history(w)

    if len(fills) < 15:
        print(f"\nv3 only got {len(fills)} fills, trying v2...")
        time.sleep(DELAY)
        fills2 = await try_v2_history(w)
        if len(fills2) > len(fills):
            fills = fills2

    # Deduplicate by order_id
    seen = set()
    unique_fills = []
    for f in fills:
        if f["oid"] not in seen:
            seen.add(f["oid"])
            unique_fills.append(f)
    fills = sorted(unique_fills, key=lambda x: x["placed"])

    buys = [f for f in fills if f["side"] == "BUY"]
    sells = [f for f in fills if f["side"] == "SELL"]
    print(f"\nTotal unique fills: {len(fills)} ({len(buys)} buys, {len(sells)} sells)\n")

    for f in fills:
        print(
            f"  {f['placed'][:19]} {f['side']:4s} {f['qty']:5.0f}x "
            f"{f['sym']:5s} ${f['strike']:7.1f} {f['otype']:4s} @ ${f['fill']:6.2f}"
        )

    # P&L by position
    buys_k = {}
    sells_k = {}
    for f in fills:
        key = (f["sym"], f["strike"], f["otype"])
        (buys_k if f["side"] == "BUY" else sells_k).setdefault(key, []).append(f)

    total = 0
    print(f"\n{'='*70}")
    print(f"  REAL P&L FROM WEBULL FILLS")
    print(f"{'='*70}")
    for key in sorted(set(list(buys_k.keys()) + list(sells_k.keys()))):
        bl = buys_k.get(key, [])
        sl = sells_k.get(key, [])
        bc = sum(f["qty"] * f["fill"] * 100 for f in bl)
        bq = sum(f["qty"] for f in bl)
        sr = sum(f["qty"] * f["fill"] * 100 for f in sl)
        sq = sum(f["qty"] for f in sl)
        pnl = sr - bc
        total += pnl
        bavg = bc / bq / 100 if bq else 0
        savg = sr / sq / 100 if sq else 0
        print(
            f"  {key[0]:5s} ${key[1]:7.1f} {key[2]:4s}  "
            f"B:{bq:3.0f} @ ${bavg:.2f} = ${bc:8.0f}  "
            f"S:{sq:3.0f} @ ${savg:.2f} = ${sr:8.0f}  "
            f"PnL: ${pnl:+9.2f}"
        )
    print(f"\n  WEBULL TOTAL P&L: ${total:+.2f}")

    # ── Update DB ──
    if len(sells) == 0:
        print("\nNo sells found — cannot reconcile DB.")
        return

    db = "journal/raw_messages.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    trades = conn.execute(
        "SELECT id, ticker, option_type, strike, contracts, "
        "premium_per_contract, exit_premium, pnl_dollars, "
        "webull_order_id, webull_exit_order_id, "
        "webull_entry_fill_price, exit_source "
        "FROM paper_trades "
        "WHERE date(opened_at) >= ? AND status = 'closed' "
        "AND webull_order_id IS NOT NULL ORDER BY id",
        (TARGET_DATE,),
    ).fetchall()
    trades = [dict(r) for r in trades]

    # Index sells for matching
    sell_by_oid = {f["oid"]: f for f in sells}
    avail_sells = {}
    for f in sells:
        key = (f["sym"], f["strike"], f["otype"])
        avail_sells.setdefault(key, []).append(f)

    # Index buys for entry fill matching
    buy_by_oid = {f["oid"]: f for f in buys}

    print(f"\n{'='*70}")
    print(f"  DB RECONCILIATION")
    print(f"{'='*70}")

    updates = []
    for t in trades:
        tid = t["id"]
        woid = t["webull_order_id"]
        exit_oid = t.get("webull_exit_order_id") or ""
        otype = (t["option_type"] or "call").lower()
        entry_fill = t["webull_entry_fill_price"] or t["premium_per_contract"]

        # Try to get real entry fill from buy orders
        if woid in buy_by_oid:
            entry_fill = buy_by_oid[woid]["fill"]

        # Find matching sell
        real_sell = None
        if exit_oid and exit_oid in sell_by_oid:
            real_sell = sell_by_oid[exit_oid]
        elif woid in sell_by_oid:
            real_sell = sell_by_oid[woid]
        else:
            key = (t["ticker"], float(t["strike"]), otype)
            candidates = avail_sells.get(key, [])
            for sf in candidates:
                if sf.get("_matched"):
                    continue
                real_sell = sf
                sf["_matched"] = True
                break

        old_exit = t["exit_premium"] or 0
        old_pnl = t["pnl_dollars"] or 0

        if real_sell:
            real_exit = real_sell["fill"]
            real_pnl = (real_exit - entry_fill) * t["contracts"] * 100
            changed = abs(real_exit - old_exit) > 0.005 or abs(real_pnl - old_pnl) > 0.50
            marker = " ***" if changed else ""
            print(
                f"  #{tid:3d} {t['ticker']:5s} x{t['contracts']:2d} "
                f"entry=${entry_fill:.2f} "
                f"exit: db=${old_exit:.2f} real=${real_exit:.2f} "
                f"pnl: db=${old_pnl:+9.2f} real=${real_pnl:+9.2f} "
                f"[{t['exit_source'] or 'ai'}]{marker}"
            )
            if changed:
                updates.append((tid, real_exit, real_pnl))
        else:
            print(
                f"  #{tid:3d} {t['ticker']:5s} x{t['contracts']:2d} "
                f"NO SELL FOUND  exit=${old_exit:.2f} pnl=${old_pnl:+.2f} "
                f"[{t['exit_source'] or 'ai'}]"
            )

    if updates:
        print(f"\n  Applying {len(updates)} updates...")
        for tid, ep, pnl in updates:
            conn.execute(
                "UPDATE paper_trades SET exit_premium=?, pnl_dollars=?, "
                "webull_exit_fill_price=? WHERE id=?",
                (ep, pnl, ep, tid),
            )
        conn.commit()

        row = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END), "
            "SUM(pnl_dollars) "
            "FROM paper_trades "
            "WHERE date(opened_at) >= ? AND status = 'closed' "
            "AND webull_order_id IS NOT NULL",
            (TARGET_DATE,),
        ).fetchone()
        print(
            f"\n  CORRECTED DB: {row[0]} trades, "
            f"{row[1]}W/{row[2]}L, P&L: ${row[3]:+.2f}"
        )
    else:
        print("\n  No updates needed.")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
