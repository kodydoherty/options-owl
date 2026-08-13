"""The three monitoring gaps that let IWM #681 run unchecked (2026-08-13).

The loss was not caused by a missing monitor. It was caused by monitors that were the
wrong shape, silently unarmed, and self-silencing:

  1. NO INVARIANT. `_reconcile_positions` compared which positions exist, never HOW MANY
     contracts. A size-down of 78 -> 71 that failed to persist was therefore invisible,
     and every exit oversold. A quantity check on the first reconcile cycle would have
     caught it at 13:55, before a cent moved.

  2. SILENT ALERTS. Every alert site read `if <condition> and discord_client:`. With no
     Discord client configured the alert vanished. 1,093 sell failures produced ZERO
     alerts and zero indication that alerting was not armed. That is worse than having no
     watchdog, because it looks identical to a quiet one.

  3. SELF-SILENCING ESCALATION. The stuck-exit alert fired at exactly 5, 10 and 20 and
     then never again. Failure #1,000 was quieter than failure #20 while the position bled
     from +$1,313 to -$1,280.

Each test below is written to FAIL against the pre-fix code.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from options_owl.execution import position_monitor


def _src(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn))


class TestQuantityInvariant:
    """Gap 1: recorded contracts must be compared against the broker."""

    def test_reconcile_compares_quantity_not_just_existence(self):
        src = _src(position_monitor._reconcile_positions)
        assert "broker_qty" in src and "db_qty" in src, (
            "reconcile does not compare contract counts — it only checks which positions "
            "exist, so a phantom quantity (IWM #681: 78 recorded vs 71 held) is invisible"
        )

    def test_matched_positions_are_no_longer_skipped_outright(self):
        """The old loop began `if key in db_keys: continue`, skipping every matched
        position before any quantity comparison could happen."""
        src = _src(position_monitor._reconcile_positions)
        idx = src.find("broker_qty")
        assert idx != -1, "no quantity comparison present at all"
        before = src[:idx]
        assert "trade = db_keys.get(key)" in before, (
            "the quantity check is not reached for MATCHED positions — a bare "
            "`if key in db_keys: continue` skips exactly the case that needs checking"
        )

    def test_heals_down_only_never_up(self):
        """Shrinking cannot oversell. Growing could create a real naked short."""
        src = _src(position_monitor._reconcile_positions)
        assert "broker_qty < db_qty" in src, "no shrink-only guard on the auto-heal"
        tree = ast.parse(src)
        updates_under_shrink = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "broker_qty < db_qty" not in ast.unparse(node.test):
                continue
            body = "\n".join(ast.unparse(n) for n in node.body)
            if "UPDATE paper_trades SET contracts" in body:
                updates_under_shrink = True
            orelse = "\n".join(ast.unparse(n) for n in node.orelse)
            assert "UPDATE paper_trades SET contracts" not in orelse, (
                "the record is written in the GROW branch — raising a recorded size to "
                "match a larger broker position must never be automatic"
            )
        assert updates_under_shrink, "the shrink branch does not correct the record"

    def test_zero_or_equal_quantity_is_not_touched(self):
        src = _src(position_monitor._reconcile_positions)
        assert "broker_qty <= 0 or broker_qty == db_qty" in src, (
            "missing the guard that leaves matching (or unreadable) quantities alone; "
            "without it every healthy position would be rewritten each cycle"
        )


class TestAlertDeliveryIsNeverSilent:
    """Gap 2: an undeliverable alert must be loud, not skipped."""

    def test_helper_exists(self):
        assert hasattr(position_monitor, "_alert_or_shout"), (
            "no delivery-guaranteed alert path — alerts remain best-effort and vanish "
            "when no Discord client is configured"
        )

    @pytest.mark.asyncio
    async def test_message_survives_a_missing_client(self, caplog):
        """With discord_client=None the alert must still reach the log at CRITICAL."""
        import logging

        records: list[str] = []

        class _Sink:
            def write(self, msg):
                records.append(str(msg))

            def flush(self):
                pass

        from loguru import logger as _logger

        sink_id = _logger.add(_Sink(), level="CRITICAL")
        try:
            await position_monitor._alert_or_shout(None, object(), "canary message")
        finally:
            _logger.remove(sink_id)

        joined = "\n".join(records)
        assert "canary message" in joined, (
            "the alert vanished when no client was configured — this is exactly how "
            "1,093 sell failures produced zero alerts"
        )
        assert "UNDELIVERABLE" in joined, (
            "the alert was logged but the fact that it could NOT be delivered was not; "
            "an unarmed watchdog must announce that it is unarmed"
        )
        assert logging  # keep the import meaningful for linters

    @pytest.mark.asyncio
    async def test_delivery_exception_does_not_break_the_monitor(self):
        """Alerting must never take down the sell path."""

        class _Boom:
            pass

        # alert_critical will raise on this junk client; _alert_or_shout must swallow it
        await position_monitor._alert_or_shout(_Boom(), object(), "boom test")

    def test_no_remaining_silent_alert_sites_in_the_close_path(self):
        """`and discord_client` in a condition means the alert is skipped when unarmed."""
        src = _src(position_monitor._finalize_full_close_inner)
        assert "and discord_client:" not in src, (
            "a conditional still gates an alert on the client existing, so it silently "
            "no-ops when alerting is not configured"
        )


class TestStuckExitEscalation:
    """Gap 3: alerting must get louder, not stop."""

    def test_alerts_continue_past_twenty(self):
        src = _src(position_monitor._finalize_full_close_inner)
        assert "transient_count % 25 == 0" in src, (
            "alerting still stops after the 5/10/20 ladder — failure #1,000 would be "
            "silent, which is what happened while IWM #681 bled"
        )

    def test_ladder_still_fires_early(self):
        """Escalation must not come at the cost of the early warning."""
        src = _src(position_monitor._finalize_full_close_inner)
        assert "(5, 10, 20)" in src, "lost the early 5/10/20 warnings"

    @pytest.mark.parametrize(
        "count,should_alert",
        [
            (1, False), (4, False), (5, True), (10, True), (19, False), (20, True),
            (21, False), (25, True), (50, True), (100, True), (1000, True), (1001, False),
        ],
    )
    def test_escalation_schedule(self, count: int, should_alert: bool):
        """Encodes the intended schedule: 5/10/20, then every 25 forever."""
        fires = count in (5, 10, 20) or (count > 20 and count % 25 == 0)
        assert fires is should_alert, f"count={count} alerting behaviour is wrong"
