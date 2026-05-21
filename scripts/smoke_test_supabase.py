"""Smoke test: verify Supabase writes work with agent_id (migration 005).

Tests each table (fills, closes, execution_decisions, account_state) with
each agent_id. Uses fake alert_ids so rows are harmless — Vince can delete them.

Usage:
    python scripts/smoke_test_supabase.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_WEBULL_JWT = os.getenv("SUPABASE_WEBULL_JWT", "")

AGENT_IDS = ["owlet_kody", "owlet_adam", "owlet_vinny", "owlet_yank"]

WRITE_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_WEBULL_JWT}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

READ_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_WEBULL_JWT}",
    "Content-Type": "application/json",
}


async def test_write(client: httpx.AsyncClient, table: str, payload: dict) -> tuple[str, int, str]:
    """POST to a table, return (table, status, detail)."""
    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        json=payload,
        headers=WRITE_HEADERS,
    )
    detail = "OK" if resp.status_code in (200, 201) else resp.text[:120]
    return table, resp.status_code, detail


async def test_rpc(client: httpx.AsyncClient) -> tuple[str, int, str]:
    """Test RPC match_alert_for_fill exists and responds."""
    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_alert_for_fill",
        json={
            "p_ticker": "SPY",
            "p_direction": "bullish",
            "p_fill_time": "2026-05-01T10:00:00Z",
            "p_window_min": 5,
        },
        headers=READ_HEADERS,
    )
    detail = "OK" if resp.status_code == 200 else resp.text[:120]
    return "rpc/match_alert_for_fill", resp.status_code, detail


async def smoke_test() -> None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY or not SUPABASE_WEBULL_JWT:
        print("ERROR: Missing Supabase credentials in .env")
        sys.exit(1)

    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Testing {len(AGENT_IDS)} agent_ids...\n")

    passed = 0
    failed = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Test RPC first
        table, status, detail = await test_rpc(client)
        icon = "PASS" if status == 200 else "FAIL"
        print(f"  [{icon}] {table}: HTTP {status} — {detail}")
        if status == 200:
            passed += 1
        else:
            failed += 1

        # Test each agent_id against each writable table
        for agent_id in AGENT_IDS:
            smoke_alert_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            tests = [
                ("fills", {
                    "alert_id": smoke_alert_id,
                    "agent_id": agent_id,
                    "broker_order_id": f"SMOKE-{smoke_alert_id[:8]}",
                    "fill_time": now,
                    "fill_price": 1.00,
                    "fill_quantity": 1,
                    "strike_filled": 500.0,
                }),
                ("closes", {
                    "alert_id": smoke_alert_id,
                    "agent_id": agent_id,
                    "close_time": now,
                    "close_price": 1.50,
                    "close_reason": "target_hit",
                    "real_pnl_pct": 50.0,
                    "real_pnl_usd": 50.0,
                }),
                ("execution_decisions", {
                    "alert_id": smoke_alert_id,
                    "agent_id": agent_id,
                    "decision": "executed",
                    "reason": "executed_normal",
                    "actual_contracts": 1,
                    "actual_strike": 500.0,
                    "notes": "smoke test — safe to delete",
                }),
                ("account_state", {
                    "agent_id": agent_id,
                    "equity_usd": 1000.0,
                    "cash_usd": 1000.0,
                    "open_positions": 0,
                    "daily_pnl_usd": 0.0,
                    "daily_pnl_pct": 0.0,
                }),
            ]

            print(f"\n  Agent: {agent_id}")
            for table, payload in tests:
                tbl, status, detail = await test_write(client, table, payload)
                ok = status in (200, 201, 409)
                icon = "PASS" if ok else "FAIL"
                print(f"    [{icon}] {tbl}: HTTP {status} — {detail}")
                if ok:
                    passed += 1
                else:
                    failed += 1

    print(f"\n{'='*50}")
    print(f"SMOKE TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        print("\nFIX failures before running backfill!")
        sys.exit(1)
    else:
        print("\nAll green — cleared for backfill.")


if __name__ == "__main__":
    asyncio.run(smoke_test())
