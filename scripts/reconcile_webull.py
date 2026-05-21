#!/usr/bin/env python3
"""Reconcile DB trades with actual Webull order history.

Pulls real fill prices from Webull API and updates paper_trades DB.
Fixes:
  - webull_entry_fill_price / webull_exit_fill_price (were $0 due to 429s)
  - exit_premium / pnl_dollars recomputed from real fills
  - Manual sell detection: matches sell fills to trades even without exit_order_id

Usage:
  # From droplet, inside a bot container:
  docker compose exec owlet-kody python scripts/reconcile_webull.py

  # With date filter:
  docker compose exec owlet-kody python scripts/reconcile_webull.py --date 2026-05-11

  # Dry run (show changes only):
  docker compose exec owlet-kody python scripts/reconcile_webull.py --dry-run
"""
import argparse
import asyncio
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_db_path(bot_name: str | None = None) -> str:
    if bot_name:
        return f"journal/{bot_name}/raw_messages.db"
    return os.environ.get("DB_PATH", "journal/raw_messages.db")


async def fetch_all_webull_fills(w, date_prefix: str) -> list[dict]:
    """Pull all filled orders from Webull order history, paginating through all pages."""
    all_combos = []
    last_coid = None
    last_oid = None

    for page in range(20):
        try:
            kwargs = {}
            if last_coid:
                kwargs["last_client_order_id"] = last_coid
                kwargs["last_order_id"] = last_oid
            response = await asyncio.to_thread(
                w._trade_client.order_v2.get_order_history,
                w._account_id, **kwargs,
            )
            result = response.json() if hasattr(response, "json") else response
            orders = (result if isinstance(result, list)
                      else result.get("orders", result.get("data", [])))
            if not orders:
                break
            all_combos.extend(orders)
            last_combo = orders[-1]
            last_coid = last_combo.get("client_order_id")
            inner = last_combo.get("orders", [])
            last_oid = inner[-1].get("order_id") if inner else None
            if len(orders) < 10:
                break
            time.sleep(1)
        except Exception:
            break

    fills = []
    for combo in all_combos:
        for o in combo.get("orders", []):
            placed = o.get("place_time_at", "")
            if not placed.startswith(date_prefix):
                continue
            if o.get("status") != "FILLED":
                continue
            legs = o.get("legs", [{}])
            fills.append({
                "oid": o.get("order_id", ""),
                "client_oid": combo.get("client_order_id", ""),
                "sym": o.get("symbol"),
                "side": o.get("side"),
                "qty": float(o.get("filled_quantity", 0)),
                "fill": float(o.get("filled_price", 0)),
                "placed": placed,
                "otype": (legs[0].get("option_type", "") if legs else "").lower(),
                "strike": float(legs[0].get("strike_price", 0)) if legs else 0,
            })

    fills.sort(key=lambda x: x["placed"])
    return fills


async def reconcile(bot_name: str | None = None, dry_run: bool = False,
                    target_date: str = "2026-05-11"):
    from options_owl.config.settings import Settings
    from options_owl.execution.webull_executor import WebullExecutor

    settings = Settings()
    db_path = get_db_path(bot_name)

    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return

    # Init Webull client
    w = WebullExecutor(settings)
    w._ensure_clients()
    print("Webull client initialized")

    # Pull fills from Webull
    fills = await fetch_all_webull_fills(w, target_date)
    print(f"\nWebull fills for {target_date}: {len(fills)}")

    buys = [f for f in fills if f["side"] == "BUY"]
    sells = [f for f in fills if f["side"] == "SELL"]
    print(f"  Buys: {len(buys)}, Sells: {len(sells)}\n")

    for f in fills:
        print(
            f"  {f['placed'][:19]} {f['side']:4s} {f['qty']:5.0f}x "
            f"{f['sym']:5s} ${f['strike']:7.1f} {f['otype']:4s} @ ${f['fill']:6.2f}  "
            f"oid={f['oid'][:16]}"
        )

    # ── Compute real P&L from Webull fills ──
    buys_by_key = {}
    sells_by_key = {}
    for f in fills:
        key = (f["sym"], f["strike"], f["otype"])
        (buys_by_key if f["side"] == "BUY" else sells_by_key).setdefault(key, []).append(f)

    total_pnl = 0.0
    print(f"\n{'='*70}")
    print(f"  REAL P&L FROM WEBULL FILLS")
    print(f"{'='*70}")

    for key in sorted(set(list(buys_by_key.keys()) + list(sells_by_key.keys()))):
        bl = buys_by_key.get(key, [])
        sl = sells_by_key.get(key, [])
        bc = sum(f["qty"] * f["fill"] * 100 for f in bl)
        bq = sum(f["qty"] for f in bl)
        sr = sum(f["qty"] * f["fill"] * 100 for f in sl)
        sq = sum(f["qty"] for f in sl)
        pnl = sr - bc
        total_pnl += pnl
        sym, strike, otype = key
        bavg = bc / bq / 100 if bq else 0
        savg = sr / sq / 100 if sq else 0
        print(
            f"  {sym:5s} ${strike:7.1f} {otype:4s}  "
            f"B: {bq:3.0f}x @ ${bavg:.2f} = ${bc:8.0f}  "
            f"S: {sq:3.0f}x @ ${savg:.2f} = ${sr:8.0f}  "
            f"PnL: ${pnl:+9.2f}"
        )

    print(f"\n  WEBULL TOTAL P&L: ${total_pnl:+.2f}")

    # ── Load DB trades and match to Webull fills ──
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = conn.execute(
        "SELECT id, ticker, option_type, strike, contracts, "
        "premium_per_contract, exit_premium, pnl_dollars, "
        "webull_order_id, webull_entry_fill_price, "
        "webull_exit_fill_price, webull_exit_order_id, "
        "exit_source, exit_reason, status "
        "FROM paper_trades "
        "WHERE date(opened_at) >= ? AND status = 'closed' "
        "AND webull_order_id IS NOT NULL "
        "ORDER BY id",
        (target_date,),
    ).fetchall()
    trades = [dict(r) for r in trades]

    print(f"\n{'='*70}")
    print(f"  DB TRADES vs WEBULL FILLS")
    print(f"{'='*70}")

    # Index sells by order_id for direct matching
    sell_by_oid = {f["oid"]: f for f in sells}

    # Index sells by (sym, strike, otype) for fuzzy matching
    available_sells = {}
    for f in sells:
        key = (f["sym"], f["strike"], f["otype"])
        available_sells.setdefault(key, []).append(f)

    updates = []
    for trade in trades:
        tid = trade["id"]
        ticker = trade["ticker"]
        strike = trade["strike"]
        contracts = trade["contracts"]
        entry_prem = trade["premium_per_contract"]
        old_exit = trade["exit_premium"] or 0
        old_pnl = trade["pnl_dollars"] or 0
        woid = trade["webull_order_id"]
        exit_oid = trade.get("webull_exit_order_id")
        otype = (trade["option_type"] or "call").lower()

        # Try to find the real sell fill:
        # 1. Direct match by exit_order_id
        # 2. Direct match by entry_order_id (if bot reuses ID)
        # 3. Fuzzy match by (ticker, strike, otype) + qty
        real_sell = None

        if exit_oid and exit_oid in sell_by_oid:
            real_sell = sell_by_oid[exit_oid]
        elif woid in sell_by_oid:
            real_sell = sell_by_oid[woid]
        else:
            # Fuzzy match: find unmatched sells for this position
            key = (ticker, float(strike), otype)
            candidates = available_sells.get(key, [])
            for sf in candidates:
                if sf.get("_matched"):
                    continue
                if sf["qty"] == contracts or sf["qty"] <= contracts:
                    real_sell = sf
                    sf["_matched"] = True
                    break

        if real_sell:
            real_exit = real_sell["fill"]
            # Use webull_entry_fill_price if available, otherwise premium_per_contract
            entry_price = trade.get("webull_entry_fill_price") or entry_prem
            if entry_price == 0:
                entry_price = entry_prem
            real_pnl = (real_exit - entry_price) * contracts * 100

            changed = abs(real_exit - old_exit) > 0.005 or abs(real_pnl - old_pnl) > 0.50
            marker = " ***" if changed else ""
            print(
                f"  #{tid:3d} {ticker:6s} ${strike:7.1f} {otype:4s} x{contracts:2d}  "
                f"entry=${entry_price:.2f}  "
                f"exit: DB=${old_exit:.2f} REAL=${real_exit:.2f}  "
                f"PnL: DB=${old_pnl:+.2f} REAL=${real_pnl:+.2f}{marker}"
            )
            if changed:
                updates.append({
                    "id": tid,
                    "exit_premium": real_exit,
                    "pnl_dollars": real_pnl,
                    "webull_exit_fill_price": real_exit,
                })
        else:
            print(
                f"  #{tid:3d} {ticker:6s} ${strike:7.1f} {otype:4s} x{contracts:2d}  "
                f"NO MATCHING SELL FOUND (exit=${old_exit:.2f} pnl=${old_pnl:+.2f})"
            )

    if updates:
        print(f"\n  {len(updates)} trades need updating")
        if dry_run:
            print("  DRY RUN — no changes applied")
        else:
            for u in updates:
                conn.execute(
                    "UPDATE paper_trades SET exit_premium = ?, pnl_dollars = ?, "
                    "webull_exit_fill_price = ? WHERE id = ?",
                    (u["exit_premium"], u["pnl_dollars"],
                     u["webull_exit_fill_price"], u["id"]),
                )
            conn.commit()
            print(f"  Applied {len(updates)} updates")

        # Show corrected totals
        row = conn.execute(
            "SELECT COUNT(*) as trades, "
            "SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END) as losses, "
            "printf('$%.2f', SUM(pnl_dollars)) as total_pnl "
            "FROM paper_trades "
            "WHERE date(opened_at) >= ? AND status = 'closed' "
            "AND webull_order_id IS NOT NULL",
            (target_date,),
        ).fetchone()
        print(f"\n  {'CORRECTED' if not dry_run else 'PROJECTED'} DB TOTALS: "
              f"{row[0]} trades, {row[1]}W/{row[2]}L, P&L: {row[3]}")
    else:
        print("\n  All exit prices already match Webull fills!")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reconcile DB with Webull fills")
    parser.add_argument("--bot", help="Bot name (e.g. owlet-kody)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", default="2026-05-11",
                        help="Target date (yyyy-mm-dd)")
    args = parser.parse_args()

    asyncio.run(reconcile(bot_name=args.bot, dry_run=args.dry_run,
                          target_date=args.date))


if __name__ == "__main__":
    main()
