"""L5 — send_alert must bound its Discord I/O so a stalled gateway send can never freeze
the 5s monitor loop (the sell path). A hung user.send() times out and is logged, not awaited
forever."""

import asyncio
from unittest.mock import MagicMock

import pytest

from options_owl.execution import alerts


def _settings():
    s = MagicMock()
    s.DISCORD_ALERT_USER_IDS = "12345"
    return s


class _HangingUser:
    async def send(self, *a, **k):
        await asyncio.sleep(3600)  # simulate a stalled Discord gateway send


class _HangingClient:
    async def fetch_user(self, uid):
        return _HangingUser()


@pytest.mark.asyncio
async def test_send_alert_times_out_and_returns(monkeypatch):
    # Shrink the internal 5s bound so the test is fast but still exercises the guard.
    real_wait_for = asyncio.wait_for

    async def fast_wait_for(coro, timeout):
        return await real_wait_for(coro, timeout=0.05)

    monkeypatch.setattr(alerts.asyncio, "wait_for", fast_wait_for)
    alerts._alerted_trades.clear()

    # Bound the test with the UNPATCHED wait_for so we don't depend on the patched one.
    # send_alert must return promptly (its internal guard times out the hung send).
    await real_wait_for(
        alerts.send_alert(_HangingClient(), _settings(), "T", "msg", force=True),
        timeout=2,
    )


@pytest.mark.asyncio
async def test_send_alert_no_users_just_logs():
    s = MagicMock()
    s.DISCORD_ALERT_USER_IDS = ""
    # No users configured -> no I/O, returns immediately (regression guard).
    await asyncio.wait_for(
        alerts.send_alert(_HangingClient(), s, "T", "msg", force=True), timeout=1
    )
