"""Webull account setup — login, get device ID, verify options access."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from webull import webull

wb = webull()

email = os.getenv("WEBULL_EMAIL", "")
password = os.getenv("WEBULL_PASSWORD", "")
trade_pin = os.getenv("WEBULL_TRADE_PIN", "")

if not email or not password:
    print("ERROR: Set WEBULL_EMAIL and WEBULL_PASSWORD in .env")
    sys.exit(1)

print(f"Email: {email}")
print(f"Auto-generated Device ID: {wb._did}")
print()

# Try direct login first (no MFA)
print("Attempting direct login (no MFA)...")
try:
    login_result = wb.login(email, password)
    if isinstance(login_result, dict) and login_result.get("accessToken"):
        print("  Direct login SUCCESS!")
    else:
        print(f"  Direct login result: {login_result}")
        print()

        # Use phone for MFA (email MFA doesn't work for this account)
        print("Direct login failed. Need MFA via phone.")
        phone = os.getenv("WEBULL_PHONE", "+19258192663")
        mfa_target = phone

        print(f"\nSending MFA code to {mfa_target}...")
        mfa_result = wb.get_mfa(mfa_target)
        print(f"  MFA result: {mfa_result}")

        if not mfa_result:
            print("  MFA send failed. Trying with phone number format...")
            phone = input("Enter your Webull phone number (e.g. +11234567890): ").strip()
            mfa_result = wb.get_mfa(phone)
            print(f"  MFA result (phone): {mfa_result}")
            mfa_target = phone

        mfa_code = input("\nEnter the 6-digit MFA code: ").strip()

        # Check for security questions
        try:
            security = wb.get_security(mfa_target)
            if security:
                print(f"\nSecurity questions:")
                for q in security:
                    print(f"  ID {q.get('questionId')}: {q.get('questionName')}")
                qid = input("Enter question ID: ").strip()
                answer = input("Enter your answer: ").strip()
                login_result = wb.login(mfa_target, password, "OptionsOwl", mfa_code, qid, answer)
            else:
                login_result = wb.login(mfa_target, password, "OptionsOwl", mfa_code)
        except Exception:
            login_result = wb.login(mfa_target, password, "OptionsOwl", mfa_code)

        if isinstance(login_result, dict) and login_result.get("accessToken"):
            print("  Login SUCCESS!")
        else:
            print(f"  Login result: {login_result}")
            print("  Login may have failed. Continuing to check...")

except Exception as e:
    print(f"  Login error: {e}")
    sys.exit(1)

# Set trade token
if trade_pin:
    print(f"\nSetting trade token...")
    try:
        wb.get_trade_token(trade_pin)
        print("  Trade token set!")
    except Exception as e:
        print(f"  Trade token error: {e}")

# Get account info
print("\nGetting account info...")
try:
    account_id = wb.get_account_id()
    print(f"  Account ID: {account_id}")
    account = wb.get_account()
    if isinstance(account, dict):
        print(f"  Net Liquidation: ${account.get('netLiquidation', '?')}")
        print(f"  Buying Power: ${account.get('optionBuyingPower', account.get('dayBuyingPower', '?'))}")
except Exception as e:
    print(f"  Account error: {e}")

# Test options
print("\nTesting options chain (SPY)...")
try:
    exps = wb.get_options_expiration_dates(stock="SPY", count=3)
    if exps:
        dates = [e.get("date") for e in exps[:3]]
        print(f"  Expirations: {dates}")
        chain = wb.get_options(stock="SPY", count=2, direction="call", expireDate=dates[0])
        if chain:
            print(f"  Got {len(chain)} entries - OPTIONS ACCESS WORKS!")
        else:
            print("  Empty chain returned")
    else:
        print("  No expirations returned")
except Exception as e:
    print(f"  Options error: {e}")

print(f"\n{'='*50}")
print(f"DEVICE ID: {wb._did}")
print(f"{'='*50}")
print(f"\nAdd to .env: WEBULL_DEVICE_ID={wb._did}")
