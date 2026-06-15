"""Tests for the BUY-side fill-escalation ladder in WebullExecutor.

These guard the CRITICAL real-money fix: entry (BUY) limit orders that do not
fill within the per-attempt wait must be CHASED UP (re-priced toward/through the
ask) instead of cancelled-and-given-up.  The #1 risk is a DOUBLE FILL — every
re-priced attempt must cancel-and-confirm the prior order first, and a fill on
any rung must stop the ladder immediately.

The Webull SDK is mocked at the executor's internal seams:
- ``_submit_order_payload``  → returns (order_id, raw_result, error)
- ``_wait_for_fill``         → returns a status string
- ``cancel_order`` / ``get_order_status`` → drive ``_confirm_cancelled``
- ``_fetch_ask``             → fresh ask used to chase the current ask
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.execution.webull_executor import WebullExecutor


def _make_settings(**overrides):
    settings = MagicMock()
    defaults = {
        "WEBULL_APP_KEY": "test_key",
        "WEBULL_APP_SECRET": "test_secret",
        "WEBULL_ACCOUNT_ID": "12345",
        "WEBULL_KILL_SWITCH": False,
        "PAPER_TRADE": False,  # allow real-order code path
        "WEBULL_ENTRY_AGGRESS_PCT": 5.0,
        "WEBULL_ENTRY_FILL_ATTEMPTS": 3,
        "WEBULL_ENTRY_MAX_CHASE_PCT": 15.0,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _executor(**settings_overrides):
    ex = WebullExecutor(_make_settings(**settings_overrides))
    # Neutralize the real SDK / safety preamble — we test the ladder, not auth.
    ex._ensure_clients = MagicMock()
    ex._check_kill_switch = AsyncMock()
    ex._check_safety_limits = MagicMock()
    # Default: no fresh quote available (chase falls back to derived ask).
    ex._fetch_ask = AsyncMock(return_value=None)
    return ex


# ask = 2.00 → aggress 5% → caller-supplied entry limit = 2.10
ENTRY_LIMIT = 2.10
BASE_ASK = 2.00


async def _buy(ex, contracts=2):
    return await ex.buy_option(
        ticker="NVDA",
        strike=900.0,
        expiry_date="2026-06-12",
        option_type="PUT",
        contracts=contracts,
        limit_price=ENTRY_LIMIT,
    )


class TestFillsOnFirstAttempt:
    @pytest.mark.asyncio
    async def test_fills_attempt_one_no_chase(self):
        ex = _executor()
        ex._submit_order_payload = AsyncMock(return_value=("ORD1", {"order_id": "ORD1"}, None))
        ex._wait_for_fill = AsyncMock(return_value="FILLED")
        cancel = ex.cancel_order = AsyncMock(return_value=True)

        result = await _buy(ex)

        assert result.success is True
        assert result.fill_status == "FILLED"
        assert result.order_id == "ORD1"
        # Filled on attempt 1 → exactly one submission, never cancelled, no chase.
        assert ex._submit_order_payload.await_count == 1
        cancel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_attempt_limit_is_caller_price(self):
        """Attempt 1 should price at ask×1.05 (the caller-supplied entry)."""
        ex = _executor()
        captured = []

        async def _submit(payload):
            captured.append(float(payload[0]["limit_price"]))
            return ("ORD1", {"order_id": "ORD1"}, None)

        ex._submit_order_payload = AsyncMock(side_effect=_submit)
        ex._wait_for_fill = AsyncMock(return_value="FILLED")
        ex.cancel_order = AsyncMock(return_value=True)

        await _buy(ex)
        # ask 2.00 * (1 + 5%) = 2.10
        assert captured[0] == pytest.approx(2.10, abs=0.01)


class TestEscalatesAndFills:
    @pytest.mark.asyncio
    async def test_no_fill_then_fills_on_third_attempt_at_higher_limit(self):
        ex = _executor()
        # Each rung gets a distinct order id.
        order_ids = ["ORD1", "ORD2", "ORD3"]
        submit_calls = {"n": 0}
        captured_limits = []

        async def _submit(payload):
            i = submit_calls["n"]
            submit_calls["n"] += 1
            captured_limits.append(float(payload[0]["limit_price"]))
            oid = order_ids[i]
            return (oid, {"order_id": oid}, None)

        ex._submit_order_payload = AsyncMock(side_effect=_submit)
        # Miss, miss, fill.
        ex._wait_for_fill = AsyncMock(side_effect=["SUBMITTED", "SUBMITTED", "FILLED"])
        # _confirm_cancelled is invoked between rungs and must report a clean cancel.
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")

        result = await _buy(ex)

        assert result.success is True
        assert result.fill_status == "FILLED"
        assert result.order_id == "ORD3"
        assert ex._submit_order_payload.await_count == 3
        # Limits must escalate UP across attempts.
        assert captured_limits[0] < captured_limits[2]
        # Prior two orders must each be cancel-confirmed before re-submitting.
        assert ex._confirm_cancelled.await_count == 2

    @pytest.mark.asyncio
    async def test_chases_fresh_ask_when_available(self):
        """When a fresh (higher) ask is available, the chase uses it."""
        ex = _executor()
        # Fresh ask jumps to 3.00 on retries.
        ex._fetch_ask = AsyncMock(return_value=3.00)
        captured = []

        async def _submit(payload):
            captured.append(float(payload[0]["limit_price"]))
            return (f"ORD{len(captured)}", {}, None)

        ex._submit_order_payload = AsyncMock(side_effect=_submit)
        ex._wait_for_fill = AsyncMock(side_effect=["SUBMITTED", "FILLED"])
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")

        result = await _buy(ex)
        assert result.success is True
        # Attempt 2 priced off the fresh 3.00 ask (well above attempt-1's ~2.10).
        assert captured[1] > captured[0]
        assert captured[1] >= 3.00


class TestNeverFills:
    @pytest.mark.asyncio
    async def test_never_fills_returns_not_filled_no_exception(self):
        ex = _executor()
        ex._submit_order_payload = AsyncMock(
            side_effect=[("ORD1", {}, None), ("ORD2", {}, None), ("ORD3", {}, None)]
        )
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        confirm = ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")

        result = await _buy(ex)

        assert result.success is False
        assert result.fill_status in ("SUBMITTED", "UNKNOWN")
        assert "not filled" in (result.error or "").lower()
        assert ex._submit_order_payload.await_count == 3
        # Every placed order (all 3) must be cancel-confirmed — none left working.
        assert confirm.await_count == 3

    @pytest.mark.asyncio
    async def test_respects_attempts_setting(self):
        ex = _executor(WEBULL_ENTRY_FILL_ATTEMPTS=2)
        ex._submit_order_payload = AsyncMock(
            side_effect=[("ORD1", {}, None), ("ORD2", {}, None)]
        )
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")

        result = await _buy(ex)
        assert result.success is False
        assert ex._submit_order_payload.await_count == 2


class TestDoubleFillGuard:
    @pytest.mark.asyncio
    async def test_never_resubmits_without_confirming_cancel_first(self):
        """CRITICAL: each re-submit must be preceded by a confirmed cancel of
        the prior order.  Assert the ordering of submit/confirm events."""
        ex = _executor()
        events = []

        async def _submit(payload):
            events.append(("submit", payload[0]["client_order_id"]))
            return (f"ORD{len(events)}", {}, None)

        async def _confirm(client_order_id, timeout_seconds=6.0):
            events.append(("confirm", client_order_id))
            return "CANCELLED"

        ex._submit_order_payload = AsyncMock(side_effect=_submit)
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        ex._confirm_cancelled = AsyncMock(side_effect=_confirm)

        await _buy(ex)

        # Sequence must be submit, confirm, submit, confirm, submit, confirm.
        kinds = [e[0] for e in events]
        assert kinds == ["submit", "confirm", "submit", "confirm", "submit", "confirm"]
        # Every re-submit must follow a confirm of the IMMEDIATELY prior order.
        for i in range(1, len(events)):
            if events[i][0] == "submit":
                assert events[i - 1][0] == "confirm", (
                    "re-submitted before confirming prior cancel — DOUBLE-FILL RISK"
                )

    @pytest.mark.asyncio
    async def test_fill_during_cancel_race_is_honored_and_halts(self):
        """If the prior order fills in the race with our cancel, honor that
        fill and STOP — never submit another rung on top of a live fill."""
        ex = _executor()
        ex._submit_order_payload = AsyncMock(
            side_effect=[("ORD1", {}, None), ("ORD2", {}, None), ("ORD3", {}, None)]
        )
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        # Cancel attempt reveals the order actually FILLED.
        ex._confirm_cancelled = AsyncMock(return_value="FILLED")

        result = await _buy(ex)

        assert result.success is True
        assert result.fill_status == "FILLED"
        assert result.order_id == "ORD1"
        # Halted immediately — only the first order was ever submitted.
        assert ex._submit_order_payload.await_count == 1

    @pytest.mark.asyncio
    async def test_unconfirmed_cancel_aborts_chase(self):
        """If a prior order's cancel cannot be confirmed (still working), we must
        NOT submit another order — abort to avoid stacking two live orders."""
        ex = _executor()
        ex._submit_order_payload = AsyncMock(
            side_effect=[("ORD1", {}, None), ("ORD2", {}, None)]
        )
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        # Cancel never confirms — order still appears to be working.
        ex._confirm_cancelled = AsyncMock(return_value="SUBMITTED")

        result = await _buy(ex)

        assert result.success is False
        assert "double-fill" in (result.error or "").lower()
        # Aborted after the first rung — no second submission.
        assert ex._submit_order_payload.await_count == 1


class TestMaxChaseCap:
    @pytest.mark.asyncio
    async def test_never_exceeds_max_chase_ceiling(self):
        """No attempt may price above ask × (1 + WEBULL_ENTRY_MAX_CHASE_PCT/100).

        With 5 allowed attempts and a 15% cap, later rungs (which would naively
        want 25%, 30% over ask) must be clamped to the 15% ceiling.
        """
        ex = _executor(WEBULL_ENTRY_FILL_ATTEMPTS=5, WEBULL_ENTRY_MAX_CHASE_PCT=15.0)
        captured = []

        async def _submit(payload):
            captured.append(float(payload[0]["limit_price"]))
            return (f"ORD{len(captured)}", {}, None)

        ex._submit_order_payload = AsyncMock(side_effect=_submit)
        ex._wait_for_fill = AsyncMock(return_value="SUBMITTED")
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        # No fresh quote → ceiling anchored to derived ask (2.00).
        ex._fetch_ask = AsyncMock(return_value=None)

        await _buy(ex)

        ceiling = BASE_ASK * 1.15  # 2.30
        assert len(captured) == 5
        for limit in captured:
            # Allow a half-cent of rounding slack (Webull $0.01 increments < $3).
            assert limit <= ceiling + 0.011, f"limit {limit} exceeded ceiling {ceiling}"
        # And the later rungs must actually reach the cap (escalation works).
        assert captured[-1] == pytest.approx(ceiling, abs=0.011)
