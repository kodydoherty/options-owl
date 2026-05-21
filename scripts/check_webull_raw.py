"""Pull real Webull trade history with correct field parsing."""

import os
import json
import sys
from collections import defaultdict

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

api = ApiClient(
    app_key=os.environ["WEBULL_APP_KEY"],
    app_secret=os.environ["WEBULL_APP_SECRET"],
    region_id="us",
)
tc = TradeClient(api)
acct_id = os.environ["WEBULL_ACCOUNT_ID"]

# Fetch balance
resp = tc.account_v2.get_account_balance(acct_id)
bal = resp.json() if hasattr(resp, "json") else resp
if isinstance(bal, dict):
    print(f"Balance: ${bal.get('total_net_liquidation_value', 'N/A')}")
    print(f"Day P&L: ${bal.get('total_day_profit_loss', 'N/A')}")
print()

# Fetch all order history (paginate)
all_orders = []
last_client_order_id = None
for page in range(5):  # max 5 pages
    kwargs = {"page_size": 100}
    if last_client_order_id:
        kwargs["last_client_order_id"] = last_client_order_id
    resp = tc.order_v2.get_order_history(acct_id, **kwargs)
    data = resp.json() if hasattr(resp, "json") else resp
    if not isinstance(data, list) or len(data) == 0:
        break
    all_orders.extend(data)
    last_client_order_id = data[-1].get("client_order_id")
    if len(data) < 100:
        break

print(f"Total orders fetched: {len(all_orders)}")

# Parse into flat list of fills
fills = []
for o in all_orders:
    for order in o.get("orders", []):
        leg = order.get("legs", [{}])[0] if order.get("legs") else {}
        fills.append({
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "status": order.get("status", ""),
            "strike": leg.get("strike_price", ""),
            "expiry": leg.get("option_expire_date", ""),
            "option_type": leg.get("option_type", ""),
            "qty": int(float(order.get("filled_quantity", 0) or 0)),
            "price": float(order.get("filled_price", 0) or 0),
            "limit": float(order.get("limit_price", 0) or 0),
            "placed_at": order.get("place_time_at", ""),
            "filled_at": order.get("filled_time_at", ""),
            "client_id": order.get("client_order_id", ""),
        })

# Group by expiry date
by_expiry = defaultdict(list)
for f in fills:
    by_expiry[f["expiry"]].append(f)

# For each expiry date, match buys to sells
grand_total = 0
for expiry in sorted(by_expiry.keys(), reverse=True):
    day_fills = by_expiry[expiry]

    # Group by (symbol, strike, option_type)
    buys = defaultdict(list)
    sells = defaultdict(list)
    for f in day_fills:
        key = (f["symbol"], f["strike"], f["option_type"])
        if f["side"] == "BUY":
            buys[key].append(f)
        elif f["side"] == "SELL":
            sells[key].append(f)

    all_keys = sorted(set(list(buys.keys()) + list(sells.keys())))
    if not all_keys:
        continue

    print(f"\n{'='*100}")
    print(f"  EXPIRY: {expiry}")
    print(f"{'='*100}")

    day_pnl = 0
    day_wins = 0
    day_losses = 0

    for key in all_keys:
        sym, strike, otype = key
        buy_list = buys.get(key, [])
        sell_list = sells.get(key, [])

        total_buy_qty = sum(b["qty"] for b in buy_list)
        total_sell_qty = sum(s["qty"] for s in sell_list)
        avg_buy = sum(b["price"] * b["qty"] for b in buy_list) / total_buy_qty if total_buy_qty else 0
        avg_sell = sum(s["price"] * s["qty"] for s in sell_list) / total_sell_qty if total_sell_qty else 0

        qty = min(total_buy_qty, total_sell_qty)
        if qty > 0 and avg_buy > 0 and avg_sell > 0:
            pnl = (avg_sell - avg_buy) * qty * 100
            pnl_pct = (avg_sell - avg_buy) / avg_buy * 100
            day_pnl += pnl
            if pnl >= 0:
                day_wins += 1
            else:
                day_losses += 1
            win = "W" if pnl >= 0 else "L"

            buy_time = buy_list[0]["placed_at"][11:19] if buy_list and buy_list[0]["placed_at"] else ""
            sell_time = sell_list[0]["filled_at"][11:19] if sell_list and sell_list[0]["filled_at"] else ""

            print(
                f"  [{win}] {sym:5} ${strike:>7} {otype:4} "
                f"| {qty:>2}ct | buy=${avg_buy:.2f} sell=${avg_sell:.2f} "
                f"| PnL=${pnl:+.2f} ({pnl_pct:+.1f}%) "
                f"| {buy_time} → {sell_time}"
            )
        elif total_buy_qty > total_sell_qty:
            print(f"  [?] {sym:5} ${strike:>7} {otype:4} | {total_buy_qty}ct bought, only {total_sell_qty}ct sold")
        elif total_sell_qty > total_buy_qty:
            print(f"  [?] {sym:5} ${strike:>7} {otype:4} | {total_sell_qty}ct sold, only {total_buy_qty}ct bought")

    grand_total += day_pnl
    print(f"\n  Day: {day_wins}W/{day_losses}L | P&L: ${day_pnl:+.2f}")

print(f"\n{'='*100}")
print(f"  GRAND TOTAL P&L: ${grand_total:+.2f}")
print(f"{'='*100}")
