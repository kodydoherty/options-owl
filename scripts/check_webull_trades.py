"""Pull real Webull trade history for a given date range.

Usage:
    # Inside docker on droplet:
    docker run --rm \
      -v /root/options-owl/scripts:/app/scripts \
      -e WEBULL_APP_KEY=$KODY_WEBULL_APP_KEY \
      -e WEBULL_APP_SECRET=$KODY_WEBULL_APP_SECRET \
      -e WEBULL_ACCOUNT_ID=$KODY_WEBULL_ACCOUNT_ID \
      options-owl-owlet-kody python scripts/check_webull_trades.py
"""

import os
import json
import sys

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

START_DATE = os.environ.get("START_DATE", "2026-04-10")
END_DATE = os.environ.get("END_DATE", "2026-04-20")

api = ApiClient(
    app_key=os.environ["WEBULL_APP_KEY"],
    app_secret=os.environ["WEBULL_APP_SECRET"],
    region_id="us",
)
tc = TradeClient(api)
acct_id = os.environ["WEBULL_ACCOUNT_ID"]

print(f"=== Webull Order History: {START_DATE} to {END_DATE} ===")
print(f"Account: {acct_id}")

# Get account balance
try:
    resp = tc.account_v2.get_account_balance(acct_id)
    bal = resp.json() if hasattr(resp, "json") else resp
    if isinstance(bal, dict):
        print(f"Balance: ${bal.get('total_net_liquidation_value', 'N/A')}")
        print(f"Day P&L: ${bal.get('total_day_profit_loss', 'N/A')}")
        for ca in bal.get("account_currency_assets", []):
            print(f"Cash: ${ca.get('cash_balance', 'N/A')}, Buying Power: ${ca.get('buying_power', 'N/A')}")
except Exception as e:
    print(f"Balance error: {e}")

print()

# Get order history
resp = tc.order_v2.get_order_history(acct_id, 100, start_date=START_DATE, end_date=END_DATE)
data = resp.json() if hasattr(resp, "json") else resp

# Full raw dump for debugging
raw_orders = data if isinstance(data, list) else data.get("orders", data.get("data", [data]))
print(f"Raw orders returned: {len(raw_orders)}")
print()

# Parse fills into paired trades
buys = []
sells = []

for o in raw_orders:
    coid = o.get("client_order_id", "")
    inner_orders = o.get("orders", [])
    for order in inner_orders:
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        status = order.get("status", "")
        placed_time = order.get("placed_time", "")
        filled_time = order.get("filled_time", "")
        avg_price = order.get("avg_filled_price", "")
        limit_price = order.get("limit_price", "")
        qty = order.get("filled_quantity", order.get("quantity", ""))

        for leg in order.get("legs", []):
            entry = {
                "symbol": leg.get("symbol", symbol),
                "side": leg.get("side", side),
                "strike": leg.get("strike_price", ""),
                "expiry": leg.get("option_expire_date", ""),
                "option_type": leg.get("option_type", ""),
                "qty": int(leg.get("filled_quantity", leg.get("quantity", qty) or 0) or 0),
                "price": float(leg.get("avg_filled_price", avg_price or limit_price or 0) or 0),
                "status": status,
                "placed": placed_time,
                "filled": filled_time,
                "client_order_id": coid,
            }
            if entry["side"] == "BUY":
                buys.append(entry)
            elif entry["side"] == "SELL":
                sells.append(entry)

# Match buys to sells by (symbol, strike, expiry, option_type)
matched = []
used_sells = set()

for buy in sorted(buys, key=lambda x: x["placed"]):
    key = (buy["symbol"], buy["strike"], buy["expiry"], buy["option_type"])
    for i, sell in enumerate(sells):
        if i in used_sells:
            continue
        sell_key = (sell["symbol"], sell["strike"], sell["expiry"], sell["option_type"])
        if key == sell_key and sell["qty"] == buy["qty"]:
            used_sells.add(i)
            matched.append((buy, sell))
            break

# Also handle partial matches
for buy in sorted(buys, key=lambda x: x["placed"]):
    key = (buy["symbol"], buy["strike"], buy["expiry"], buy["option_type"])
    already_matched = any(b is buy for b, s in matched)
    if already_matched:
        continue
    for i, sell in enumerate(sells):
        if i in used_sells:
            continue
        sell_key = (sell["symbol"], sell["strike"], sell["expiry"], sell["option_type"])
        if key == sell_key:
            used_sells.add(i)
            matched.append((buy, sell))
            break

# Print results by date
from collections import defaultdict
daily = defaultdict(list)

for buy, sell in matched:
    date = buy["placed"][:10] if buy["placed"] else "unknown"
    daily[date].append((buy, sell))

# Also show unmatched
unmatched_buys = [b for b in buys if not any(b is mb for mb, ms in matched)]
unmatched_sells = [s for i, s in enumerate(sells) if i not in used_sells]

total_pnl = 0
total_trades = 0

for date in sorted(daily.keys()):
    trades = daily[date]
    day_pnl = 0
    print(f"\n{'='*100}")
    print(f"  {date} — {len(trades)} round-trip trades")
    print(f"{'='*100}")

    for buy, sell in trades:
        qty = min(buy["qty"], sell["qty"])
        if qty > 0 and buy["price"] > 0 and sell["price"] > 0:
            pnl = (sell["price"] - buy["price"]) * qty * 100
            pnl_pct = (sell["price"] - buy["price"]) / buy["price"] * 100
            day_pnl += pnl
            win = "W" if pnl >= 0 else "L"
            print(
                f"  [{win}] {buy['symbol']:5} ${buy['strike']:>7} {buy['option_type']:4} "
                f"| {qty:>2}ct | buy=${buy['price']:.2f} sell=${sell['price']:.2f} "
                f"| PnL=${pnl:+.2f} ({pnl_pct:+.1f}%) "
                f"| {buy['placed'][11:19] if buy['placed'] else ''} → {sell['filled'][11:19] if sell['filled'] else ''}"
            )
        else:
            print(f"  [?] {buy['symbol']} ${buy['strike']} — incomplete fill data")

    wins = sum(1 for b, s in trades if s["price"] > b["price"])
    losses = len(trades) - wins
    total_pnl += day_pnl
    total_trades += len(trades)
    print(f"  Day total: ${day_pnl:+.2f} ({wins}W/{losses}L)")

if unmatched_buys:
    print(f"\n  Unmatched buys: {len(unmatched_buys)}")
    for b in unmatched_buys:
        print(f"    BUY {b['symbol']} ${b['strike']} {b['qty']}ct @ ${b['price']:.2f} — {b['placed']}")

if unmatched_sells:
    print(f"\n  Unmatched sells: {len(unmatched_sells)}")
    for s in unmatched_sells:
        print(f"    SELL {s['symbol']} ${s['strike']} {s['qty']}ct @ ${s['price']:.2f} — {s['placed']}")

print(f"\n{'='*100}")
print(f"  TOTAL: {total_trades} trades | P&L: ${total_pnl:+.2f}")
print(f"{'='*100}")
