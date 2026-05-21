#!/usr/bin/env python3
"""Manually sell orphaned positions on Webull."""
import asyncio
from options_owl.config.settings import Settings
from options_owl.execution.webull_executor import WebullExecutor


async def main():
    s = Settings()
    w = WebullExecutor(s)
    await w.init()

    # Wait to avoid rate limit
    await asyncio.sleep(3)

    # 1. Sell orphaned TSLA PUT $367.5 x1 (cut loss)
    print("=== Selling orphaned TSLA PUT $367.5 x1 ===")
    result = await w.sell_option(
        ticker="TSLA",
        strike=367.5,
        expiry_date="2026-04-27",
        option_type="PUT",
        contracts=1,
        limit_price=0.30,
    )
    if result.success:
        print(f"  SOLD — order_id={result.order_id}")
    else:
        print(f"  FAILED: {result.error}")

    await asyncio.sleep(3)

    # 2. Sell TSLA CALL $370 x4 (take the win)
    print("\n=== Selling TSLA CALL $370 x4 (taking profit) ===")
    result = await w.sell_option(
        ticker="TSLA",
        strike=370.0,
        expiry_date="2026-04-27",
        option_type="CALL",
        contracts=4,
        limit_price=2.20,
    )
    if result.success:
        print(f"  SOLD — order_id={result.order_id}")
        if result.client_order_id:
            await asyncio.sleep(3)
            fill = await w.get_fill_price(result.client_order_id)
            if fill:
                print(f"  Fill: ${fill:.2f} (total ${fill * 4 * 100:.2f})")
    else:
        print(f"  FAILED: {result.error}")


asyncio.run(main())
