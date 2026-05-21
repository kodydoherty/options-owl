"""End-to-end Webull API test — exercises all endpoints without placing real orders.

Tests:
1. Auth + account detection
2. Account balance
3. Account positions
4. Market data quotes (SPY)
5. Option order preview (does NOT place)
6. Open orders list

Usage: python scripts/test_webull_api.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

from options_owl.config.settings import Settings


def main():
    settings = Settings()

    if not settings.WEBULL_APP_KEY or not settings.WEBULL_APP_SECRET:
        print("ERROR: WEBULL_APP_KEY and WEBULL_APP_SECRET must be set in .env")
        sys.exit(1)

    print("=" * 60)
    print("Webull API End-to-End Test")
    print("=" * 60)

    # 1. Initialize client
    print("\n[1/6] Initializing API client...")
    api_client = ApiClient(
        app_key=settings.WEBULL_APP_KEY,
        app_secret=settings.WEBULL_APP_SECRET,
        region_id="us",
    )
    trade_client = TradeClient(api_client)
    print("  ✓ API client initialized")

    # 2. Get account list
    print("\n[2/6] Fetching account list...")
    try:
        resp = trade_client.account_v2.get_account_list()
        print(f"  Raw response type: {type(resp)}")
        if hasattr(resp, 'json'):
            data = resp.json()
        else:
            data = resp
        print(f"  Response: {data}")

        # Extract account ID
        if isinstance(data, dict):
            accounts = data.get("accounts", data.get("data", data.get("account_list", [])))
            if not accounts and "account_id" in data:
                accounts = [data]
        elif isinstance(data, list):
            accounts = data
        else:
            accounts = []

        if accounts:
            account_id = str(
                accounts[0].get("account_id",
                    accounts[0].get("accountId",
                        accounts[0].get("sec_account_id", "?")))
            )
            print(f"  ✓ Account ID: {account_id}")
        else:
            account_id = settings.WEBULL_ACCOUNT_ID
            print(f"  ⚠ No accounts in list response, using configured: {account_id}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        account_id = settings.WEBULL_ACCOUNT_ID
        print(f"  Using configured account ID: {account_id}")

    if not account_id:
        print("\nERROR: No account ID available. Set WEBULL_ACCOUNT_ID in .env")
        sys.exit(1)

    # 3. Get account balance
    print("\n[3/6] Fetching account balance...")
    try:
        resp = trade_client.account_v2.get_account_balance(account_id)
        if hasattr(resp, 'json'):
            bal = resp.json()
        else:
            bal = resp
        print(f"  Response: {bal}")

        if isinstance(bal, dict):
            total = bal.get("total_asset", bal.get("totalAsset", "?"))
            cash = bal.get("total_cash_balance", bal.get("totalCashBalance", "?"))
            print(f"  ✓ Total: ${total}, Cash: ${cash}")
        else:
            print(f"  ✓ Got balance response")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 4. Get account positions
    print("\n[4/6] Fetching positions...")
    try:
        resp = trade_client.account_v2.get_account_position(account_id)
        if hasattr(resp, 'json'):
            pos = resp.json()
        else:
            pos = resp
        print(f"  Response: {pos}")

        if isinstance(pos, dict):
            holdings = pos.get("holdings", pos.get("data", []))
            print(f"  ✓ {len(holdings)} positions")
        else:
            print(f"  ✓ Got positions response")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 5. Preview an option order (does NOT place)
    print("\n[5/6] Previewing option order (SPY $580 CALL, 1 contract, $0.01)...")
    try:
        preview_payload = [{
            "client_order_id": "test_preview_001",
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "quantity": "1",
            "limit_price": "0.01",
            "option_strategy": "SINGLE",
            "side": "BUY",
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "legs": [{
                "side": "BUY",
                "quantity": "1",
                "symbol": "SPY",
                "strike_price": "580",
                "option_expire_date": "2026-04-08",
                "instrument_type": "OPTION",
                "option_type": "CALL",
                "market": "US",
            }],
        }]
        resp = trade_client.order_v2.preview_option(account_id, preview_payload)
        if hasattr(resp, 'json'):
            preview = resp.json()
        else:
            preview = resp
        print(f"  Response: {preview}")
        print(f"  ✓ Preview endpoint works")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"    (This may fail if market is closed or account has no buying power)")

    # 6. Get open orders
    print("\n[6/6] Fetching open orders...")
    try:
        resp = trade_client.order_v2.get_order_open(account_id, 100)
        if hasattr(resp, 'json'):
            orders = resp.json()
        else:
            orders = resp
        print(f"  Response: {orders}")

        if isinstance(orders, dict):
            order_list = orders.get("orders", orders.get("data", []))
            print(f"  ✓ {len(order_list)} open orders")
        else:
            print(f"  ✓ Got orders response")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 7. Try market data
    print("\n[BONUS] Testing market data...")
    try:
        from webull.data.data_client import DataClient
        data_client = DataClient(api_client)
        # Try getting a stock snapshot
        resp = data_client.market_data.get_snapshot("SPY", "US_STOCK")
        if hasattr(resp, 'json'):
            snap = resp.json()
        else:
            snap = resp
        if isinstance(snap, dict):
            price = snap.get("close", snap.get("last_price", snap.get("lastPrice", "?")))
            print(f"  SPY snapshot: {snap}")
            print(f"  ✓ Market data works")
        else:
            print(f"  Response: {snap}")
    except Exception as e:
        print(f"  ✗ Market data failed: {e}")
        print(f"    (May need different category or symbol format)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
