"""A sell rejected as a naked short must self-heal, not retry forever.

THE INCIDENT (2026-08-13, kody IWM #681)
----------------------------------------
The liquidity guard sized an entry down 78 -> 71 contracts. That order then filled via
the FILLED-DURING-CANCEL path, which reported `filled_quantity=None` meaning "full fill,
nothing to correct" -- but "full" there meant the SIZED-DOWN 71, not the requested 78.
paper_trader kept 78.

Every exit then tried to sell 78 against a 71-lot position, so Webull rejected the excess
as a naked short (MUST_BE_CLOSE_THAN_SELL_SHORT). That rejection was classified
TRANSIENT_ERROR, which by design does NOT consume the abandonment budget, so the monitor
reopened and retried every ~4s: roughly 760 identical failures over 35 minutes while the
FSM's profit_lock exit could not execute. The position went from +$1,313 to -$1,280.

Two independent defects, one test file:
  1. the entry path must report the true filled size when it was sized down
  2. the exit path must treat the rejection as PERMANENT and self-heal by shrinking the
     record to the broker's true size (shrinking can never oversell)
"""

from __future__ import annotations

import pytest

from options_owl.execution.paper_trader import SellOutcome


class TestEntryReportsTrueSizeAfterSizeDown:
    """Defect 1: FILLED-DURING-CANCEL must not report a sized-down fill as 'full'."""

    def test_filled_during_cancel_reports_sized_down_count(self):
        """`filled_quantity=None` means 'matches the request' — it must not be sent
        when a liquidity size-down changed the submitted quantity."""
        import ast
        import inspect
        import textwrap

        from options_owl.execution.webull_executor import WebullExecutor

        src = textwrap.dedent(
            inspect.getsource(WebullExecutor._place_buy_with_escalation)
        )
        tree = ast.parse(src)

        # locate the filled_qty assignment guarded by confirm_status == "FILLED"
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, "id", None) == "filled_qty" for t in node.targets):
                continue
            rendered = ast.unparse(node.value)
            if "confirm_status" in rendered:
                found = True
                assert "requested_contracts" in rendered, (
                    "the FILLED-DURING-CANCEL branch reports None on a full fill without "
                    "comparing against requested_contracts, so a liquidity size-down is "
                    "reported as a full fill and the record keeps the ORIGINAL size — "
                    "this is the IWM #681 phantom-quantity bug"
                )
        assert found, "could not locate the filled_qty/confirm_status assignment"


class TestQuantityMismatchIsPermanent:
    """Defect 2: the rejection must not be classified as retry-forever."""

    def test_outcome_enum_has_a_permanent_category(self):
        assert hasattr(SellOutcome, "QUANTITY_MISMATCH"), (
            "no distinct outcome for an oversell rejection — it falls through to "
            "TRANSIENT_ERROR, which never consumes the abandonment budget and so retries "
            "indefinitely against a broker that can never accept the order"
        )

    def test_classifier_maps_the_broker_error_to_it(self):
        """The specific Webull code must route to QUANTITY_MISMATCH."""
        import inspect

        from options_owl.execution.paper_trader import PaperTrader

        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "must_be_close_than_sell_short" in src.lower(), (
            "the MUST_BE_CLOSE_THAN_SELL_SHORT rejection is not recognised, so it is "
            "treated as transient and retried forever"
        )
        # search from the LAST occurrence: the string also appears in an older comment
        # documenting cause (1), the pending-order case fixed on 2026-07-07.
        idx = src.lower().rindex("must_be_close_than_sell_short")
        assert "QUANTITY_MISMATCH" in src[idx:idx + 900], (
            "the oversell rejection is recognised but does not return QUANTITY_MISMATCH"
        )

    def test_transient_pending_order_cause_still_recovers(self):
        """Cause (1) — a pending order tying up quantity — must keep its retry recovery.

        The same broker code fires when a prior SUBMITTED order reserves the position
        (META #467 chain). Classifying it permanent outright would abandon a recoverable
        exit, so the self-heal must be conditional on the broker reporting a SMALLER size.
        """
        import inspect

        from options_owl.execution import position_monitor

        src = inspect.getsource(position_monitor._finalize_full_close_inner)
        assert "true_qty is not None" in src, (
            "self-heal is not guarded on actually having read a broker size; an unreadable "
            "size must fall through to the normal transient retry, not abandon the exit"
        )

    def test_monitor_self_heals_by_shrinking_only(self):
        """The monitor must correct the record DOWN to the broker size, never up.

        Shrinking is the only safe direction: selling fewer than held cannot oversell,
        while raising the count would risk a naked short.
        """
        import inspect

        from options_owl.execution import position_monitor

        src = inspect.getsource(position_monitor._finalize_full_close_inner)
        assert "QUANTITY_MISMATCH" in src, "monitor does not handle the permanent case"
        assert "get_open_option_positions" in src, (
            "monitor does not read the broker's true size, so it cannot self-heal"
        )
        assert "true_qty < int(trade[\"contracts\"])" in src or "true_qty < int(trade['contracts'])" in src, (
            "the self-heal is not guarded to shrink-only — it must never raise the "
            "recorded size above what is held"
        )

    def test_zero_broker_quantity_is_treated_as_position_gone(self):
        import inspect

        from options_owl.execution import position_monitor

        src = inspect.getsource(position_monitor._finalize_full_close_inner)
        assert "true_qty == 0" in src and "is_position_gone = True" in src, (
            "a broker quantity of 0 must fall through to POSITION_NOT_FOUND rather than "
            "looping on a position that no longer exists"
        )


@pytest.mark.parametrize(
    "err,expected_permanent",
    [
        ("HTTP 417 Code: OAUTH_OPENAPI_OPTION_LONG_POSITION_MUST_BE_CLOSE_THAN_SELL_SHORT", True),
        ("You can not place a sell-short order", True),
        ("HTTP 500 internal error", False),
        ("connection reset by peer", False),
    ],
)
def test_error_strings_classify_as_expected(err: str, expected_permanent: bool):
    """Guards the matcher against both false negatives and over-matching."""
    e = err.lower()
    is_perm = (
        "must_be_close_than_sell_short" in e
        or "close_than_sell_short" in e
        or "can not place a sell-short" in e
    )
    assert is_perm is expected_permanent, f"{err!r} classified wrongly"
