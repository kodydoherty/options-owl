#!/usr/bin/env python3
"""Debug script: pull real Webull positions and orders."""
import asyncio
import json
from options_owl.config.settings import Settings
from options_owl.execution.webull_executor import WebullExecutor

async def main():
    s = Settings()
    w = WebullExecutor(s)
    await w.init()

    info = await w.get_account_info()
    print("=== ACCOUNT ===")
    print(f"Total: ${info.total_asset:.2f}  Cash: ${info.cash_balance:.2f}  BP: ${info.buying_power:.2f}")
    print(f"\n=== POSITIONS ({len(info.positions)}) ===")
    for p in info.positions:
        print(json.dumps(p, indent=2, default=str))

    orders = await w.get_open_orders()
    print(f"\n=== OPEN ORDERS ({len(orders)}) ===")
    for o in orders:
        print(json.dumps(o, indent=2, default=str))

asyncio.run(main())
