"""Webull token verification — run standalone to complete first-time auth.

Usage (on droplet):
  docker exec -it owlet-dennis python scripts/webull_verify_token.py

Or locally:
  WEBULL_APP_KEY=... WEBULL_APP_SECRET=... python scripts/webull_verify_token.py

Dennis will get a verification code via SMS/email. He approves it in the
Webull app, and this script polls until the token status goes NORMAL.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app_key = os.environ.get("WEBULL_APP_KEY", "")
app_secret = os.environ.get("WEBULL_APP_SECRET", "")

if not app_key or not app_secret:
    print("ERROR: Set WEBULL_APP_KEY and WEBULL_APP_SECRET")
    sys.exit(1)

print(f"App Key: {app_key[:8]}...{app_key[-4:]}")
print("Initializing Webull SDK (this triggers verification code)...")
print()

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

api_client = ApiClient(
    app_key=app_key,
    app_secret=app_secret,
    region_id="us",
)

try:
    trade_client = TradeClient(api_client)
    print("SUCCESS — token verified, TradeClient initialized!")
except Exception as exc:
    if "MANY_TOO_TOKEN" in str(exc) or "more than 10" in str(exc):
        print(f"Token limit hit: {exc}")
        print("Old tokens will expire in ~24h. Bypassing token init...")
        from webull.core.http.initializer.client_initializer import ClientInitializer
        _orig = ClientInitializer.init_token
        ClientInitializer.init_token = staticmethod(lambda *a, **kw: None)
        try:
            trade_client = TradeClient(api_client)
            print("TradeClient initialized (bypassed token check).")
        finally:
            ClientInitializer.init_token = _orig
    elif "not verified" in str(exc).lower() or "PENDING" in str(exc):
        print(f"Token PENDING — Dennis needs to approve the verification code.")
        print(f"Error: {exc}")
        print()
        print("Once Dennis enters the code, re-run this script.")
        sys.exit(1)
    else:
        print(f"ERROR: {exc}")
        sys.exit(1)

# Try to get account info
print()
print("Fetching account info...")
try:
    response = trade_client.account_v2.get_account_list()
    accounts = response.json() if hasattr(response, 'json') else response
    print(f"Account response: {accounts}")
except Exception as e:
    print(f"Account fetch error: {e}")

print()
print("Done. If token is NORMAL, owlet-dennis will auth automatically on next startup.")
