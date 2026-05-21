"""Quick script to query Webull account balance and today's P&L."""
import asyncio
import json
from options_owl.execution.webull_executor import WebullExecutor
from options_owl.config.settings import Settings

async def main():
    s = Settings()
    w = WebullExecutor(s)
    await w.init()
    info = await w.get_account_info()
    print(f"Total asset: ${info.total_asset:,.2f}")
    print(f"Cash balance: ${info.cash_balance:,.2f}")
    print(f"Buying power: ${info.buying_power:,.2f}")
    print(f"Open positions: {len(info.positions)}")
    for p in info.positions:
        print(f"  {p}")

    # Also get raw balance JSON for day P&L fields
    bal_resp = await asyncio.to_thread(
        w._trade_client.account_v2.get_account_balance,
        w._account_id,
    )
    bal = bal_resp.json() if hasattr(bal_resp, "json") else bal_resp
    if isinstance(bal, dict):
        print("\n--- Raw balance fields ---")
        for k, v in sorted(bal.items()):
            if k == "account_currency_assets":
                continue
            print(f"  {k}: {v}")
        # Check currency assets for day P&L
        ca = bal.get("account_currency_assets", bal.get("accountCurrencyAssets", []))
        if ca:
            print("\n--- Currency assets ---")
            for item in ca:
                for k2, v2 in sorted(item.items()):
                    print(f"  {k2}: {v2}")

asyncio.run(main())
