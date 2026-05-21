"""Pull real Webull fill prices and update DB entries.

Matches Webull order history to DB trades via order_id,
then updates webull_entry_fill_price, webull_exit_fill_price,
and recalculates pnl_dollars/pnl_pct from real fills.

Usage:
    docker compose run --rm --no-deps \
      -v /root/options-owl/scripts:/app/scripts \
      owlet-kody python scripts/fix_db_fills.py
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta

DB_PATH = os.environ.get("DB_PATH", "journal/raw_messages.db")
START_DATE = os.environ.get("START_DATE", "2026-04-10")
END_DATE = os.environ.get("END_DATE", "2026-04-23")


def get_webull_orders():
    """Pull all order history from Webull and return parsed fills + balance."""
    from webull.core.client import ApiClient
    from webull.trade.trade_client import TradeClient

    app_key = os.environ.get("WEBULL_APP_KEY", "")
    app_secret = os.environ.get("WEBULL_APP_SECRET", "")
    account_id = os.environ.get("WEBULL_ACCOUNT_ID", "")

    if not app_key or not app_secret:
        print("ERROR: WEBULL_APP_KEY and WEBULL_APP_SECRET required")
        return [], {}

    api = ApiClient(app_key=app_key, app_secret=app_secret, region_id="us")
    tc = TradeClient(api)

    # Auto-detect account ID if not set
    if not account_id:
        try:
            resp = tc.account_v2.get_account_list()
            accounts = resp.json() if hasattr(resp, "json") else resp
            if isinstance(accounts, dict):
                account_list = accounts.get("accounts", accounts.get("data", []))
            elif isinstance(accounts, list):
                account_list = accounts
            else:
                account_list = []
            for acct in account_list:
                acct_type = acct.get("account_type", "").upper()
                if acct_type == "CASH":
                    account_id = str(acct.get("account_id", ""))
                    print(f"Auto-detected account ID: {account_id}")
                    break
            if not account_id:
                print("ERROR: Could not auto-detect WEBULL_ACCOUNT_ID")
                return [], {}
        except Exception as e:
            print(f"ERROR: Failed to auto-detect account: {e}")
            return [], {}

    # Get account balance
    balance_info = {}
    try:
        resp = tc.account_v2.get_account_balance(account_id)
        bal = resp.json() if hasattr(resp, "json") else resp
        if isinstance(bal, dict):
            balance_info = {
                "total_value": bal.get("total_net_liquidation_value", "N/A"),
                "day_pnl": bal.get("total_day_profit_loss", "N/A"),
            }
            for ca in bal.get("account_currency_assets", []):
                balance_info["cash"] = ca.get("cash_balance", "N/A")
                balance_info["buying_power"] = ca.get("buying_power", "N/A")
    except Exception as e:
        print(f"Balance error: {e}")

    # Get order history
    resp = tc.order_v2.get_order_history(account_id, 100, start_date=START_DATE, end_date=END_DATE)
    data = resp.json() if hasattr(resp, "json") else resp
    raw_orders = data if isinstance(data, list) else data.get("orders", data.get("data", [data]))

    # Parse fills — the real data is at the inner order level
    # Structure: { client_order_id, orders: [{ order_id, side, status,
    #   filled_price, filled_quantity, place_time_at, filled_time_at, legs: [...] }] }
    fills = []
    for o in raw_orders:
        coid = o.get("client_order_id", "")
        combo_id = o.get("combo_order_id", "")

        for order in o.get("orders", []):
            order_id = order.get("order_id", "")
            side = order.get("side", "")
            status = order.get("status", "")
            filled_price = order.get("filled_price")
            filled_qty = order.get("filled_quantity", "0")
            place_time = order.get("place_time_at", "")
            filled_time = order.get("filled_time_at", "")

            # Get leg details
            legs = order.get("legs", [])
            leg = legs[0] if legs else {}

            fills.append({
                "client_order_id": coid,
                "order_id": order_id,
                "combo_order_id": combo_id,
                "symbol": leg.get("symbol", order.get("symbol", "")),
                "side": side,
                "status": status,
                "strike": leg.get("strike_price", ""),
                "expiry": leg.get("option_expire_date", ""),
                "option_type": leg.get("option_type", ""),
                "qty": int(float(filled_qty or 0)),
                "filled_price": float(filled_price) if filled_price else None,
                "place_time": place_time,
                "filled_time": filled_time,
            })

    return fills, balance_info


def match_and_update():
    """Match Webull fills to DB trades and update fill prices."""
    fills, balance_info = get_webull_orders()

    if balance_info:
        print(f"Account balance: ${balance_info.get('total_value', 'N/A')}")
        print(f"Day P&L: ${balance_info.get('day_pnl', 'N/A')}")
        print()

    # Index fills by order_id (matches DB webull_order_id)
    fills_by_oid = {}
    for f in fills:
        oid = f["order_id"]
        if oid:
            fills_by_oid[oid] = f

    # Index fills by client_order_id (matches DB webull_client_order_id)
    fills_by_coid = {}
    for f in fills:
        coid = f["client_order_id"]
        if coid:
            fills_by_coid.setdefault(coid, []).append(f)

    # Build lookup by (symbol, strike, option_type, side) for fallback matching
    fills_by_key = {}
    for f in fills:
        if f["filled_price"] and f["status"] == "FILLED":
            key = (f["symbol"].upper(), f["strike"], f["option_type"].upper(), f["side"])
            fills_by_key.setdefault(key, []).append(f)

    print(f"Webull fills: {len(fills)} total, {len(fills_by_oid)} by order_id")

    # Get DB trades
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Add columns if needed
    cols = [r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()]
    for col in ("webull_entry_fill_price REAL", "webull_exit_fill_price REAL", "webull_exit_order_id TEXT"):
        col_name = col.split()[0]
        if col_name not in cols:
            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col}")
            print(f"Added column: {col_name}")

    trades = conn.execute("""
        SELECT * FROM paper_trades
        WHERE opened_at >= ?
        AND webull_order_id IS NOT NULL AND webull_order_id != ''
        ORDER BY opened_at
    """, (START_DATE,)).fetchall()

    print(f"DB trades with Webull orders: {len(trades)}")
    print()

    updated = 0
    for trade in trades:
        trade = dict(trade)
        tid = trade["id"]
        ticker = trade["ticker"].upper()
        strike = str(trade["strike"])
        # Normalize strike format (DB might have "709.0", Webull has "709.00")
        try:
            strike_norm = f"{float(strike):.2f}"
        except ValueError:
            strike_norm = strike
        otype = trade["option_type"].upper()
        contracts = trade["contracts"]
        db_oid = trade.get("webull_order_id", "")
        db_coid = trade.get("webull_client_order_id", "")

        # === Match ENTRY fill ===
        entry_fill = None

        # 1. Try by order_id (DB webull_order_id = Webull order_id)
        if db_oid and db_oid in fills_by_oid:
            f = fills_by_oid[db_oid]
            if f["side"] == "BUY" and f["filled_price"]:
                entry_fill = f["filled_price"]

        # 2. Try by client_order_id
        if entry_fill is None and db_coid and db_coid in fills_by_coid:
            for f in fills_by_coid[db_coid]:
                if f["side"] == "BUY" and f["filled_price"]:
                    entry_fill = f["filled_price"]
                    break

        # 3. Fallback: match by (ticker, strike, type, BUY)
        if entry_fill is None:
            key = (ticker, strike_norm, otype, "BUY")
            if key in fills_by_key:
                for f in fills_by_key[key]:
                    if f["qty"] == contracts:
                        entry_fill = f["filled_price"]
                        fills_by_key[key].remove(f)
                        break

        # === Match EXIT fill ===
        exit_fill = None
        key = (ticker, strike_norm, otype, "SELL")
        if key in fills_by_key:
            for f in fills_by_key[key]:
                if f["qty"] == contracts:
                    exit_fill = f["filled_price"]
                    fills_by_key[key].remove(f)
                    break
            # Try any qty match
            if exit_fill is None:
                for f in fills_by_key[key]:
                    if f["filled_price"]:
                        exit_fill = f["filled_price"]
                        fills_by_key[key].remove(f)
                        break

        # Calculate real P&L
        db_entry = trade["premium_per_contract"]
        db_exit = trade.get("exit_premium")
        db_pnl = trade.get("pnl_dollars", 0) or 0

        real_pnl = None
        real_pnl_pct = None
        if entry_fill and exit_fill:
            real_cost = entry_fill * contracts * 100
            real_proceeds = exit_fill * contracts * 100
            real_pnl = real_proceeds - real_cost
            real_pnl_pct = (real_pnl / real_cost * 100) if real_cost > 0 else 0

        win = (real_pnl is not None and real_pnl >= 0) or (real_pnl is None and db_pnl >= 0)
        icon = "W" if win else "L"

        entry_str = f"DB=${db_entry:.2f}"
        if entry_fill:
            entry_str += f" → WB=${entry_fill:.2f}"
        else:
            entry_str += " (no WB)"

        exit_str = f"DB=${db_exit:.2f}" if db_exit else "no exit"
        if exit_fill:
            exit_str += f" → WB=${exit_fill:.2f}"

        pnl_str = f"DB=${db_pnl:+.2f}"
        if real_pnl is not None:
            pnl_str += f" → REAL=${real_pnl:+.2f} ({real_pnl_pct:+.1f}%)"

        print(f"  [{icon}] #{tid} {ticker} ${strike} {otype} x{contracts} | {entry_str} | {exit_str} | {pnl_str}")

        # Update DB
        if entry_fill or exit_fill:
            updates = []
            params = []
            if entry_fill:
                updates.append("webull_entry_fill_price = ?")
                params.append(entry_fill)
            if exit_fill:
                updates.append("webull_exit_fill_price = ?")
                params.append(exit_fill)
            if real_pnl is not None:
                updates.append("pnl_dollars = ?")
                params.append(real_pnl)
                updates.append("pnl_pct = ?")
                params.append(real_pnl_pct)
            params.append(tid)
            conn.execute(f"UPDATE paper_trades SET {', '.join(updates)} WHERE id = ?", params)
            updated += 1

    conn.commit()
    conn.close()
    print(f"\nUpdated {updated}/{len(trades)} trades with real Webull fill data")


if __name__ == "__main__":
    match_and_update()
