"""Pull filled Webull orders and compute P&L per day."""
import json, os, sys
from datetime import datetime, timezone
from collections import defaultdict

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

app_key = os.environ["WEBULL_APP_KEY"]
app_secret = os.environ["WEBULL_APP_SECRET"]
account_id = os.environ["WEBULL_ACCOUNT_ID"]

api = ApiClient(app_key=app_key, app_secret=app_secret, region_id="us")
tc = TradeClient(api)

start = sys.argv[1] if len(sys.argv) > 1 else "2026-04-07"
end = sys.argv[2] if len(sys.argv) > 2 else "2026-04-20"

resp = tc.order_v2.get_order_history(account_id, 100, start_date=start, end_date=end)
if hasattr(resp, "json"):
    data = resp.json()
elif isinstance(resp, list):
    data = resp
else:
    data = [resp]
if not isinstance(data, list):
    data = [data]

trades = []
for item in data:
    orders = item.get("orders", [])
    for o in orders:
        if o.get("status") != "FILLED":
            continue
        legs = o.get("legs", [{}])
        leg = legs[0] if legs else {}
        place_str = o.get("place_time_at", "")
        if place_str:
            dt = datetime.fromisoformat(place_str.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S UTC")
        else:
            date_str = "?"
            time_str = "?"

        qty = float(o.get("filled_quantity", 0))
        price = float(o.get("filled_price", 0))
        side = o.get("side", "?")
        intent = o.get("position_intent", "?")
        symbol = o.get("symbol", "?")
        strike = leg.get("strike_price", "?")
        opt_type = leg.get("option_type", "?")
        expiry = leg.get("option_expire_date", "?")

        trades.append({
            "date": date_str, "time": time_str, "symbol": symbol,
            "side": side, "intent": intent, "qty": qty, "price": price,
            "strike": strike, "type": opt_type, "expiry": expiry,
            "value": qty * price * 100,
        })

trades.sort(key=lambda x: (x["date"], x["time"]))

by_date = defaultdict(list)
for t in trades:
    by_date[t["date"]].append(t)

grand_total = 0
for date in sorted(by_date.keys()):
    day_trades = by_date[date]
    print(f"\n=== {date} ({len(day_trades)} filled orders) ===")
    buys = {}
    for t in day_trades:
        arrow = "BUY " if "BUY" in t["intent"] else "SELL"
        print(f"  {t['time']} {arrow} {t['qty']:.0f}x {t['symbol']} ${t['strike']}{t['type'][0]} exp={t['expiry']} @ ${t['price']:.2f}  (= ${t['value']:.0f})")

        key = f"{t['symbol']}_{t['strike']}_{t['type']}_{t['expiry']}"
        if "BUY" in t["intent"]:
            buys[key] = buys.get(key, 0) - t["value"]
        else:
            buys[key] = buys.get(key, 0) + t["value"]

    total = sum(buys.values())
    grand_total += total
    print(f"  --- Day P&L: ${total:+,.0f}")

print(f"\n{'='*50}")
print(f"TOTAL P&L ({start} to {end}): ${grand_total:+,.0f}")
