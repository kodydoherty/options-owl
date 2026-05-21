"""Fix today's DB by matching Webull fills to parent trades.

Scaleout records (no webull_order_id) are zeroed out — their P&L is
captured in the parent trade's real Webull fill totals.
"""
import asyncio
import sqlite3
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_DATE = "2026-05-12"
DELAY = 5


async def fetch_v3_fills(w):
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

        if len(orders) < 10:
            break

    # Deduplicate
    seen = set()
    unique = []
    for f in all_fills:
        if f["oid"] not in seen:
            seen.add(f["oid"])
            unique.append(f)
    return sorted(unique, key=lambda x: x["placed"])


async def main():
    from options_owl.config.settings import Settings
    from options_owl.execution.webull_executor import WebullExecutor

    s = Settings()
    w = WebullExecutor(s)
    await w.init()

    fills = await fetch_v3_fills(w)
    buys = [f for f in fills if f["side"] == "BUY"]
    sells = [f for f in fills if f["side"] == "SELL"]
    print(f"Webull fills: {len(fills)} ({len(buys)} buys, {len(sells)} sells)\n")

    # Group by position key
    positions = {}
    for f in fills:
        key = (f["sym"], f["strike"], f["otype"])
        positions.setdefault(key, {"buys": [], "sells": []})
        positions[key]["buys" if f["side"] == "BUY" else "sells"].append(f)

    # Compute real P&L per position
    real_pnl_by_key = {}
    print(f"{'='*70}")
    print(f"  REAL P&L FROM WEBULL FILLS")
    print(f"{'='*70}")
    total_real = 0
    for key in sorted(positions.keys()):
        p = positions[key]
        bc = sum(f["qty"] * f["fill"] * 100 for f in p["buys"])
        bq = sum(f["qty"] for f in p["buys"])
        sr = sum(f["qty"] * f["fill"] * 100 for f in p["sells"])
        sq = sum(f["qty"] for f in p["sells"])
        pnl = sr - bc
        total_real += pnl
        bavg = bc / bq / 100 if bq else 0
        savg = sr / sq / 100 if sq else 0
        real_pnl_by_key[key] = {
            "buy_cost": bc, "buy_qty": bq, "buy_avg": bavg,
            "sell_rev": sr, "sell_qty": sq, "sell_avg": savg,
            "pnl": pnl,
        }
        print(
            f"  {key[0]:5s} ${key[1]:7.1f} {key[2]:4s}  "
            f"B:{bq:3.0f} @ ${bavg:.2f} = ${bc:8.0f}  "
            f"S:{sq:3.0f} @ ${savg:.2f} = ${sr:8.0f}  "
            f"PnL: ${pnl:+9.2f}"
        )
    print(f"\n  WEBULL TOTAL P&L: ${total_real:+.2f}")

    # Load DB trades
    db = "journal/raw_messages.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    all_trades = conn.execute(
        "SELECT id, ticker, option_type, strike, contracts, "
        "premium_per_contract, exit_premium, pnl_dollars, "
        "webull_order_id, exit_source, exit_reason, status "
        "FROM paper_trades WHERE date(opened_at) >= ? ORDER BY id",
        (TARGET_DATE,),
    ).fetchall()
    all_trades = [dict(r) for r in all_trades]

    # Separate parent trades (have webull_order_id) from scaleouts (no webull_order_id)
    parent_trades = [t for t in all_trades if t["webull_order_id"]]
    scaleout_trades = [t for t in all_trades if not t["webull_order_id"]]

    print(f"\n{'='*70}")
    print(f"  DB: {len(parent_trades)} parent trades, {len(scaleout_trades)} scaleouts")
    print(f"{'='*70}")

    # Group parent trades by position key
    parents_by_key = {}
    for t in parent_trades:
        key = (t["ticker"], float(t["strike"]), (t["option_type"] or "call").lower())
        parents_by_key.setdefault(key, []).append(t)

    # Strategy: for each position key, distribute the real Webull P&L
    # across the parent trade(s), then zero out scaleouts for that key
    updates = []

    for key, real in real_pnl_by_key.items():
        parents = parents_by_key.get(key, [])
        if not parents:
            print(f"\n  WARNING: {key} has Webull fills but no parent trade in DB!")
            continue

        # If single parent, assign all P&L to it
        if len(parents) == 1:
            t = parents[0]
            total_contracts = int(real["buy_qty"])
            old_pnl = t["pnl_dollars"] or 0
            new_entry = real["buy_avg"]
            new_exit = real["sell_avg"]
            new_pnl = real["pnl"]

            changed = abs(new_pnl - old_pnl) > 0.50
            marker = " ***" if changed else ""
            print(
                f"\n  #{t['id']:3d} {key[0]:5s} ${key[1]:7.1f} {key[2]:4s} "
                f"x{t['contracts']}→x{total_contracts} "
                f"entry=${new_entry:.2f} exit=${new_exit:.2f} "
                f"pnl: db=${old_pnl:+.2f} real=${new_pnl:+.2f}{marker}"
            )
            if changed:
                updates.append({
                    "id": t["id"],
                    "contracts": total_contracts,
                    "premium_per_contract": new_entry,
                    "exit_premium": new_exit,
                    "pnl_dollars": new_pnl,
                    "webull_entry_fill_price": new_entry,
                    "webull_exit_fill_price": new_exit,
                })
        else:
            # Multiple parent trades for same key (e.g. two DCA buys)
            # Distribute P&L proportionally by contract count
            total_parent_contracts = sum(t["contracts"] for t in parents)
            for t in parents:
                share = t["contracts"] / total_parent_contracts if total_parent_contracts else 0
                new_pnl = real["pnl"] * share
                old_pnl = t["pnl_dollars"] or 0
                changed = abs(new_pnl - old_pnl) > 0.50
                marker = " ***" if changed else ""
                print(
                    f"\n  #{t['id']:3d} {key[0]:5s} ${key[1]:7.1f} {key[2]:4s} "
                    f"x{t['contracts']} (share={share:.0%}) "
                    f"pnl: db=${old_pnl:+.2f} real=${new_pnl:+.2f}{marker}"
                )
                if changed:
                    updates.append({
                        "id": t["id"],
                        "pnl_dollars": new_pnl,
                        "exit_premium": real["sell_avg"],
                        "webull_exit_fill_price": real["sell_avg"],
                    })

    # Zero out scaleout trades (their P&L is now in parent)
    scaleout_zero = []
    for t in scaleout_trades:
        if (t["pnl_dollars"] or 0) != 0:
            scaleout_zero.append(t["id"])
            print(
                f"  Zeroing scaleout #{t['id']:3d} {t['ticker']:5s} "
                f"x{t['contracts']} pnl=${t['pnl_dollars'] or 0:+.2f} → $0.00"
            )

    # Apply updates
    print(f"\n{'='*70}")
    print(f"  APPLYING {len(updates)} parent updates + {len(scaleout_zero)} scaleout zeroes")
    print(f"{'='*70}")

    for u in updates:
        if "contracts" in u:
            conn.execute(
                "UPDATE paper_trades SET contracts=?, premium_per_contract=?, "
                "exit_premium=?, pnl_dollars=?, webull_entry_fill_price=?, "
                "webull_exit_fill_price=? WHERE id=?",
                (u["contracts"], u["premium_per_contract"], u["exit_premium"],
                 u["pnl_dollars"], u["webull_entry_fill_price"],
                 u["webull_exit_fill_price"], u["id"]),
            )
        else:
            conn.execute(
                "UPDATE paper_trades SET exit_premium=?, pnl_dollars=?, "
                "webull_exit_fill_price=? WHERE id=?",
                (u["exit_premium"], u["pnl_dollars"],
                 u["webull_exit_fill_price"], u["id"]),
            )

    for tid in scaleout_zero:
        conn.execute(
            "UPDATE paper_trades SET pnl_dollars=0, exit_premium=0 WHERE id=?",
            (tid,),
        )

    conn.commit()

    # Final totals
    row = conn.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END), "
        "SUM(pnl_dollars) "
        "FROM paper_trades "
        "WHERE date(opened_at) >= ? AND status = 'closed'",
        (TARGET_DATE,),
    ).fetchone()
    print(
        f"\n  CORRECTED DB: {row[0]} trades, "
        f"{row[1]}W/{row[2]}L, Total P&L: ${row[3]:+.2f}"
    )
    print(f"  TARGET (Webull): ${total_real:+.2f}")
    print(f"  DIFFERENCE: ${row[3] - total_real:+.2f}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
