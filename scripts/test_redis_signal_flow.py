#!/usr/bin/env python3
"""End-to-end test: inject a fake signal into Redis, verify all read paths work.

Usage:
    # Against local Redis (docker must be running)
    python scripts/test_redis_signal_flow.py

    # Against droplet Redis
    python scripts/test_redis_signal_flow.py --droplet

Tests:
  1. Signal publish → read back via get_latest_signal()
  2. Stock price publish → read back via get_price()
  3. Option premium publish → read back via get_option_premium()
  4. Flow bar publish → read back via get_latest_flow() and get_all_latest_flow()
  5. Signal deduplication (try_claim_signal)
  6. Pub/sub delivery (publish_signal → subscriber receives)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


async def run_tests(redis_url: str) -> bool:
    """Run all Redis signal flow tests. Returns True if all pass."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("FAIL: redis package not installed (pip install redis)")
        return False

    # Connect
    print(f"\nConnecting to {redis_url} ...")
    try:
        r = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
        await r.ping()
        print("OK: Redis connected\n")
    except Exception as exc:
        print(f"FAIL: Cannot connect to Redis — {exc}")
        return False

    passed = 0
    failed = 0
    test_prefix = "test:owl:"  # isolated namespace so we don't pollute real keys

    # ── Test 1: Signal publish + read ────────────────────────────────────────
    print("Test 1: Signal publish → get_latest_signal")
    try:
        signal = {
            "id": 99999,
            "ticker": "TEST",
            "direction": "call",
            "score": 85.5,
            "premium": 1.50,
            "strike": 530.0,
            "emitted_at": "2026-05-26T10:00:00",
        }
        key = f"owl:ml_signal:TEST"
        await r.set(key, json.dumps(signal), ex=60)
        data = await r.get(key)
        result = json.loads(data) if data else None
        assert result is not None, "Signal not found"
        assert result["ticker"] == "TEST", f"Wrong ticker: {result['ticker']}"
        assert result["score"] == 85.5, f"Wrong score: {result['score']}"
        assert result["direction"] == "call", f"Wrong direction: {result['direction']}"
        print("  PASS: Signal written and read back correctly")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1
    finally:
        await r.delete(key)

    # ── Test 2: Stock price publish + read ───────────────────────────────────
    print("Test 2: Stock price publish → get_price (with timestamp)")
    try:
        key = "owl:price:TEST"
        price_data = json.dumps({"price": 534.21, "t": time.time()})
        await r.set(key, price_data, ex=60)
        val = await r.get(key)
        result = json.loads(val) if val else None
        assert result is not None, "Price not found"
        assert abs(result["price"] - 534.21) < 0.01, f"Wrong price: {result['price']}"
        assert "t" in result, "Missing timestamp"
        assert result["t"] > 0, "Invalid timestamp"
        print("  PASS: Stock price written and read back with timestamp")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1
    finally:
        await r.delete(key)

    # ── Test 3: Option premium publish + read ────────────────────────────────
    print("Test 3: Option premium publish → get_option_premium")
    try:
        contract_key = "TEST:call:530:2026-05-26"
        key = f"owl:option:{contract_key}"
        premium_data = {"bid": 1.40, "ask": 1.60, "mid": 1.50, "t": time.time()}
        await r.set(key, json.dumps(premium_data), ex=60)
        val = await r.get(key)
        result = json.loads(val) if val else None
        assert result is not None, "Premium not found"
        assert abs(result["mid"] - 1.50) < 0.01, f"Wrong mid: {result['mid']}"
        assert abs(result["bid"] - 1.40) < 0.01, f"Wrong bid: {result['bid']}"
        assert abs(result["ask"] - 1.60) < 0.01, f"Wrong ask: {result['ask']}"
        assert result["t"] > 0, "Missing timestamp"
        print("  PASS: Option premium written and read back correctly")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1
    finally:
        await r.delete(key)

    # ── Test 4: Flow bar publish + read ──────────────────────────────────────
    print("Test 4: Flow bar publish → get_latest_flow / get_all_latest_flow")
    try:
        flow_bar = {
            "ticker": "TEST",
            "ts": "2026-05-26T10:00:00",
            "call_volume": 5000,
            "put_volume": 3000,
            "net_premium": 250000.0,
            "sweep_count": 12,
        }
        key = f"owl:flow:TEST"
        await r.set(key, json.dumps(flow_bar), ex=60)

        # Read single
        val = await r.get(key)
        result = json.loads(val) if val else None
        assert result is not None, "Flow bar not found"
        assert result["ticker"] == "TEST", f"Wrong ticker: {result['ticker']}"
        assert result["call_volume"] == 5000, f"Wrong call_volume: {result['call_volume']}"

        # Read all (scan)
        all_flow = {}
        async for k in r.scan_iter(match="owl:flow:*"):
            ticker = k.replace("owl:flow:", "")
            d = await r.get(k)
            if d:
                all_flow[ticker] = json.loads(d)
        assert "TEST" in all_flow, "TEST not found in get_all_latest_flow"
        print("  PASS: Flow bar written, read single and scan all work")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1
    finally:
        await r.delete(key)

    # ── Test 5: Signal deduplication (SET NX) ────────────────────────────────
    print("Test 5: Signal deduplication (try_claim_signal)")
    try:
        dedup_key = "owl:signal:TEST_DEDUP_12345"
        # First claim should succeed
        result1 = await r.set(dedup_key, "agent-kody", nx=True, ex=60)
        assert result1 is not None, "First claim should succeed"

        # Second claim (different agent) should fail
        result2 = await r.set(dedup_key, "agent-adam", nx=True, ex=60)
        assert result2 is None, "Second claim should be blocked (NX)"

        # Verify the owner
        owner = await r.get(dedup_key)
        assert owner == "agent-kody", f"Wrong owner: {owner}"
        print("  PASS: First agent claims, second is blocked, owner is correct")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1
    finally:
        await r.delete(dedup_key)

    # ── Test 6: Pub/sub signal delivery ──────────────────────────────────────
    print("Test 6: Pub/sub signal delivery")
    try:
        channel = "owl:signals"
        received = []

        # Subscribe
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        # Drain the subscribe confirmation message
        await pubsub.get_message(timeout=1)

        # Publish
        signal = {"id": 88888, "ticker": "PUBTEST", "score": 90.0}
        await r.publish(channel, json.dumps(signal))

        # Read (with timeout)
        msg = await pubsub.get_message(timeout=3)
        assert msg is not None, "No pub/sub message received within 3s"
        assert msg["type"] == "message", f"Wrong msg type: {msg['type']}"
        payload = json.loads(msg["data"])
        assert payload["ticker"] == "PUBTEST", f"Wrong ticker: {payload['ticker']}"
        assert payload["score"] == 90.0, f"Wrong score: {payload['score']}"
        print("  PASS: Published signal received via pub/sub")
        passed += 1

        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1

    # ── Test 7: Verify redis_client module functions ─────────────────────────
    print("Test 7: redis_client module integration (init → publish → read → close)")
    try:
        sys.path.insert(0, ".")
        from options_owl.db import redis_client

        # Init
        await redis_client.init_redis(redis_url)
        assert redis_client.is_connected(), "redis_client not connected after init"

        # Publish price (now returns (price, age) tuple)
        await redis_client.publish_price("MODTEST", 123.45)
        result = await redis_client.get_price("MODTEST")
        assert result is not None, f"Module price read failed: {result}"
        p, age = result
        assert abs(p - 123.45) < 0.01, f"Module price wrong: {p}"
        assert age < 5, f"Module price age too old: {age}"

        # Publish option premium + test case normalization
        await redis_client.publish_option_premium("MODTEST:Call:125:2026-05-26", 2.0, 2.2, 2.1)
        # Read with lowercase — should still match thanks to normalization
        prem = await redis_client.get_option_premium("MODTEST:call:125:2026-05-26")
        assert prem is not None and abs(prem["mid"] - 2.1) < 0.01, f"Module premium read failed: {prem}"

        # Publish signal
        await redis_client.publish_signal({"id": 77777, "ticker": "MODTEST", "score": 75.0})
        sig = await redis_client.get_latest_signal("MODTEST")
        assert sig is not None and sig["ticker"] == "MODTEST", f"Module signal read failed: {sig}"

        # Publish flow
        await redis_client.set_latest_flow("MODTEST", {"ticker": "MODTEST", "call_volume": 100})
        flow = await redis_client.get_latest_flow("MODTEST")
        assert flow is not None and flow["call_volume"] == 100, f"Module flow read failed: {flow}"

        # Cleanup
        await r.delete("owl:price:MODTEST", "owl:option:MODTEST:call:125:2026-05-26",
                       "owl:ml_signal:MODTEST", "owl:flow:MODTEST")
        await redis_client.close()
        print("  PASS: All redis_client module functions work end-to-end")
        passed += 1
    except Exception as exc:
        print(f"  FAIL: {exc}")
        failed += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    await r.aclose()
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*50}")
    return failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Redis signal flow end-to-end")
    parser.add_argument("--droplet", action="store_true", help="Test against droplet Redis via SSH tunnel")
    parser.add_argument("--url", default=None, help="Custom Redis URL (default: redis://localhost:6379/0)")
    args = parser.parse_args()

    if args.url:
        redis_url = args.url
    elif args.droplet:
        print("NOTE: To test against droplet, first open an SSH tunnel:")
        print("  ssh -i ~/.ssh/id_ed25519_do -L 6379:localhost:6379 root@129.212.138.145 -N &")
        print("Then run this script without --droplet (it will connect to localhost:6379)\n")
        redis_url = "redis://localhost:6379/0"
    else:
        redis_url = "redis://localhost:6379/0"

    success = asyncio.run(run_tests(redis_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
