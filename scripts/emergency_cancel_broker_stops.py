#!/usr/bin/env python3
"""EMERGENCY: cancel ALL resting broker STOP_LOSS orders at Webull for this bot.

A resting SELL STOP_LOSS reserves the position's holding quantity, which BLOCKS the FSM's own exit
(OPTION_LONG_POSITION_MUST_BE_CLOSE_THAN_SELL_SHORT). Disabling ENABLE_BROKER_STOP stops NEW stops but leaves
existing ones resting. Run this INSIDE the bot container (docker exec) so it reuses the bot's stored Webull
session (no re-auth / token churn). Read-only except for cancelling STOP_LOSS orders.
"""
import asyncio

from options_owl.config.settings import Settings
from options_owl.execution.webull_executor import WebullExecutor


async def main() -> None:
    ex = WebullExecutor(Settings())
    try:
        orders = await asyncio.wait_for(ex.get_open_orders(), timeout=20)
    except Exception as exc:
        print(f"FAILED to list open orders: {exc}")
        return
    stops = [o for o in (orders or []) if str(o.get("order_type", "")).upper() == "STOP_LOSS"]
    print(f"found {len(stops)} resting STOP_LOSS order(s)")
    cancelled = 0
    for o in stops:
        coid = o.get("client_order_id")
        legs = o.get("legs") or [{}]
        sym = f"{legs[0].get('symbol')} {legs[0].get('strike_price')} {legs[0].get('option_type')}"
        if not coid:
            print(f"  SKIP (no client_order_id): {sym}")
            continue
        try:
            ok = await asyncio.wait_for(ex.cancel_order(coid), timeout=15)
            print(f"  cancel {sym} coid={coid} -> {ok}")
            cancelled += 1
        except Exception as exc:
            print(f"  cancel FAILED {sym} coid={coid}: {exc}")
    print(f"cancelled {cancelled}/{len(stops)} stop order(s)")


if __name__ == "__main__":
    asyncio.run(main())
