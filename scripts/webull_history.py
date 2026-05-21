"""Pull all Webull order history and output as CSV for reconciliation."""
import asyncio
import json
import logging
import os
import sys
import time

# Suppress ALL logging to stdout — redirect everything to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
for name in ("webull", "webull_trade_sdk", "urllib3", "httpx"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).handlers = []
os.environ.setdefault("WEBULL_LOG_LEVEL", "CRITICAL")

# Suppress loguru
from loguru import logger
logger.remove()

from options_owl.execution.webull_executor import WebullExecutor
from options_owl.config.settings import Settings


async def main():
    s = Settings()
    w = WebullExecutor(s)
    await w.init()
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
            print(f"# Page {page} error: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        if not isinstance(data, list) or not data:
            break
        all_items.extend(data)
        last_cid = data[-1].get("client_order_id", "")
        print(f"# Page {page}: {len(data)} orders", file=sys.stderr)
        if len(data) < 100:
            break

    # CSV header — includes option details from legs
    print("day,time_utc,side,symbol,option_type,strike,expiry,qty,price,value,order_id,client_order_id,status")

    for entry in all_items:
        cid = entry.get("client_order_id", "")
        for order in entry.get("orders", []):
            status = order.get("status", "")
            if status == "CANCELLED":
                continue
            side = order.get("side", "")
            sym = order.get("symbol", "")
            # Webull API uses "filled_quantity" (string), not "filled_qty"
            qty = int(order.get("filled_quantity", 0) or 0)
            price = float(order.get("filled_price", 0) or 0)
            if qty == 0 or price == 0:
                continue
            ft = order.get("place_time_at", "")
            day = ft[:10] if ft else "unknown"
            time_utc = ft[11:19] if len(ft) > 18 else ""
            oid = order.get("order_id", "")
            val = qty * price * 100

            # Extract option details from legs
            opt_type = ""
            strike = ""
            expiry = ""
            legs = order.get("legs", [])
            if legs:
                leg = legs[0]
                opt_type = leg.get("option_type", "")
                strike = leg.get("strike_price", "")
                expiry = leg.get("option_expire_date", "")

            print(f"{day},{time_utc},{side},{sym},{opt_type},{strike},{expiry},{qty},{price},{val:.2f},{oid},{cid},{status}")

    # Also dump raw JSON for debugging if --raw flag
    if "--raw" in sys.argv:
        print("\n# RAW JSON:", file=sys.stderr)
        for entry in all_items[:3]:
            print(json.dumps(entry, indent=2, default=str), file=sys.stderr)


asyncio.run(main())
