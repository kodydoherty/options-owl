"""Webull order-status throttle (429 fix): per-order cache + min-interval + 429 cooldown + batch.
Order placement still works; we just check status less aggressively so the per-account rate limit
isn't blown during the morning burst."""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from options_owl.execution.webull_executor import WebullExecutor, _is_webull_rate_limit


def _exec():
    e = WebullExecutor(SimpleNamespace(WEBULL_ACCOUNT_ID="acct"))
    e._ensure_clients = lambda: None
    e._os_min_interval = 0.0   # don't actually sleep in tests
    e._trade_client = MagicMock()
    return e


def test_is_rate_limit_detection():
    assert _is_webull_rate_limit(Exception("HTTP Status: 429, Code: TOO_MANY_REQUESTS"))
    assert _is_webull_rate_limit(Exception("429"))
    assert not _is_webull_rate_limit(Exception("connection reset by peer"))


@pytest.mark.asyncio
async def test_cache_dedupes_rapid_calls():
    e = _exec()
    n = {"calls": 0}

    def _detail(acct, coid):
        n["calls"] += 1
        return {"status": "SUBMITTED"}

    e._trade_client.order_v2.get_order_detail = _detail
    r1 = await e.get_order_status("abc")
    r2 = await e.get_order_status("abc")   # within TTL → served from cache, no 2nd API call
    assert n["calls"] == 1
    assert r1 == r2 == {"status": "SUBMITTED"}


@pytest.mark.asyncio
async def test_429_sets_cooldown_and_serves_cached():
    e = _exec()
    e._os_cache_ttl = 0.0                  # force a re-call each time (isolate the cooldown path)
    seq = [{"status": "SUBMITTED"}, Exception("429 TOO_MANY_REQUESTS")]

    def _detail(acct, coid):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    e._trade_client.order_v2.get_order_detail = _detail
    await e.get_order_status("abc")        # caches SUBMITTED
    r2 = await e.get_order_status("abc")   # 429 → cooldown set, serve stale cache
    assert e._os_cooldown_until > time.monotonic()
    assert r2 == {"status": "SUBMITTED"}
    remaining = len(seq)
    r3 = await e.get_order_status("abc")   # in cooldown → NO API call, serve cache
    assert len(seq) == remaining           # no new call made
    assert r3 == {"status": "SUBMITTED"}


@pytest.mark.asyncio
async def test_positions_skipped_during_cooldown():
    e = _exec()
    e._os_cooldown_until = time.monotonic() + 10   # active cooldown
    e._trade_client.account_v2.get_account_position = MagicMock(
        side_effect=AssertionError("should NOT call positions during cooldown"))
    out = await e.get_open_option_positions()
    assert out == []                        # returned empty without hitting the API
