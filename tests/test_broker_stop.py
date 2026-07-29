"""Phase 1 tests — broker-side STOP_LOSS executor capability (webull_executor.place_stop_loss).

The resting-stop payload sits on the LIVE sell path, so these lock the schema Webull's options API
requires (order_type=STOP_LOSS, stop_price on a legal increment, side=SELL, DAY, single leg) and the
safety behavior of ``place_stop_loss``: it is a NO-OP when the flag is off, and it NEVER submits a stop
without a close_contracts position_id (which would risk a sell-to-open / naked short). No live API is
touched — the submit + position lookup are mocked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.config.settings import Settings
from options_owl.execution.webull_executor import WebullExecutor


@pytest.fixture(autouse=True)
def _reset_broker_stop_fuse():
    """The circuit breaker + tracked-stop registry are module globals — clear both before AND after
    every test for isolation."""
    from options_owl.execution import broker_stop as _bs
    _bs.reset_broker_stops_kill()
    _bs._ACTIVE_STOP_CLIENT_IDS.clear()
    yield
    _bs.reset_broker_stops_kill()
    _bs._ACTIVE_STOP_CLIENT_IDS.clear()


def _make_settings(**overrides):
    settings = MagicMock()
    defaults = {
        "WEBULL_APP_KEY": "test_key",
        "WEBULL_APP_SECRET": "test_secret",
        "WEBULL_ACCOUNT_ID": "12345",
        "WEBULL_KILL_SWITCH": False,
        "PAPER_TRADE": True,
        "MARGIN_ACCOUNT": False,
        "ENABLE_PUT_TRADING": True,
        "ENABLE_BROKER_STOP": True,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


class TestStopPayloadSchema:
    """The STOP_LOSS payload must match Webull's documented options schema exactly."""

    def _payload(self, **kw):
        defaults = dict(
            client_order_id="abc123",
            ticker="spy",
            strike=500.0,
            expiry_date="2026-07-22",
            option_type="call",
            contracts=3,
            stop_price=2.40,
            close_contracts=[{"position_id": "P1", "quantity": "3"}],
        )
        defaults.update(kw)
        return WebullExecutor._build_stop_order_payload(**defaults)[0]

    def test_order_type_is_stop_loss(self):
        assert self._payload()["order_type"] == "STOP_LOSS"

    def test_side_is_always_sell(self):
        o = self._payload()
        assert o["side"] == "SELL"
        assert o["legs"][0]["side"] == "SELL"

    def test_carries_stop_price_not_limit_price(self):
        o = self._payload(stop_price=2.40)
        assert o["stop_price"] == "2.40"
        assert "limit_price" not in o

    def test_day_time_in_force(self):
        # Webull options stops are DAY-only.
        assert self._payload()["time_in_force"] == "DAY"

    def test_single_leg_uppercased(self):
        o = self._payload(ticker="spy", option_type="call")
        assert len(o["legs"]) == 1
        leg = o["legs"][0]
        assert leg["symbol"] == "SPY"
        assert leg["option_type"] == "CALL"
        assert leg["instrument_type"] == "OPTION"

    def test_carries_close_contracts(self):
        o = self._payload(close_contracts=[{"position_id": "P9", "quantity": "3"}])
        assert o["close_contracts"] == [{"position_id": "P9", "quantity": "3"}]

    def test_stop_price_snaps_to_legal_increment(self):
        # >= $3 must sit on a nickel; a resting stop rounds DOWN (SELL direction).
        o = self._payload(stop_price=3.07)
        assert o["stop_price"] == "3.05"
        cents = round(float(o["stop_price"]) * 100)
        assert cents % 5 == 0

    def test_stop_price_penny_under_three(self):
        o = self._payload(stop_price=2.97)
        assert o["stop_price"] == "2.97"


class TestReplaceStopLoss:
    @pytest.mark.asyncio
    async def test_noop_when_flag_off(self):
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=False))
        ex._ensure_clients = MagicMock()
        r = await ex.replace_stop_loss(
            client_order_id="C1", ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.fill_status == "DISABLED"

    @pytest.mark.asyncio
    async def test_modify_keeps_client_order_id_and_new_stop(self):
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=True))
        ex._ensure_clients = MagicMock()
        ex._account_id = "ACCT"
        captured = {}

        def _replace(account_id, modify_orders):
            captured["orders"] = modify_orders
            resp = MagicMock()
            resp.json.return_value = {"order_id": "OID9"}
            return resp
        ex._trade_client = MagicMock()
        ex._trade_client.order_v2.replace_option = _replace
        r = await ex.replace_stop_loss(
            client_order_id="C1", ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.success is True
        o = captured["orders"][0]
        assert o["client_order_id"] == "C1"        # identifies the order to modify
        assert o["order_type"] == "STOP_LOSS"
        assert o["stop_price"] == "2.40"           # the new price


class TestPlaceStopLossSafety:
    @pytest.mark.asyncio
    async def test_noop_when_flag_off(self):
        """Flag off → returns a DISABLED no-op and never touches the client."""
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=False))
        ex._ensure_clients = MagicMock()
        ex._find_position_id = AsyncMock(return_value="P1")
        ex._submit_order_payload = AsyncMock(return_value=("OID", {}, None))
        r = await ex.place_stop_loss(
            ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.success is False
        assert r.fill_status == "DISABLED"
        ex._submit_order_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocks_when_no_position(self):
        """No live Webull position → NEVER submit (a SELL stop w/o close_contracts risks a naked short)."""
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=True))
        ex._ensure_clients = MagicMock()
        ex._check_kill_switch = AsyncMock()
        ex._find_position_id = AsyncMock(return_value=None)
        ex._submit_order_payload = AsyncMock(return_value=("OID", {}, None))
        r = await ex.place_stop_loss(
            ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.success is False
        assert r.fill_status == "NO_POSITION"
        ex._submit_order_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_places_with_close_contracts_on_success(self):
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=True))
        ex._ensure_clients = MagicMock()
        ex._check_kill_switch = AsyncMock()
        ex._find_position_id = AsyncMock(return_value="POS42")
        ex._submit_order_payload = AsyncMock(return_value=("ORD99", {}, None))
        r = await ex.place_stop_loss(
            ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.success is True
        assert r.order_id == "ORD99"
        assert r.client_order_id  # caller needs this to cancel/replace
        # The submitted payload must carry the position_id as sell-to-close.
        submitted = ex._submit_order_payload.call_args[0][0][0]
        assert submitted["order_type"] == "STOP_LOSS"
        assert submitted["close_contracts"][0]["position_id"] == "POS42"

    @pytest.mark.asyncio
    async def test_rejected_submit_returns_failure(self):
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=True))
        ex._ensure_clients = MagicMock()
        ex._check_kill_switch = AsyncMock()
        ex._find_position_id = AsyncMock(return_value="POS42")
        ex._submit_order_payload = AsyncMock(return_value=(None, {}, "OPTION_PRICE_STEP_GTE"))
        r = await ex.place_stop_loss(
            ticker="SPY", strike=500.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=3, stop_price=2.40,
        )
        assert r.success is False
        assert r.fill_status == "REJECTED"

    @pytest.mark.asyncio
    async def test_clamps_oversized_stop_to_venue_cap(self):
        ex = WebullExecutor(_make_settings(ENABLE_BROKER_STOP=True))
        ex._ensure_clients = MagicMock()
        ex._check_kill_switch = AsyncMock()
        ex._find_position_id = AsyncMock(return_value="POS42")
        ex._submit_order_payload = AsyncMock(return_value=("ORD99", {}, None))
        await ex.place_stop_loss(
            ticker="SMCI", strike=60.0, expiry_date="2026-07-22",
            option_type="CALL", contracts=120, stop_price=2.40,
        )
        submitted = ex._submit_order_payload.call_args[0][0][0]
        assert submitted["quantity"] == "100"  # clamped to MAX_ORDER_CONTRACTS
        assert submitted["legs"][0]["quantity"] == "100"


class TestBrokerStopSettings:
    def test_disabled_by_default(self):
        assert Settings().ENABLE_BROKER_STOP is False

    def test_default_knobs(self):
        s = Settings()
        assert s.BROKER_STOP_ENTRY_FRAC == 0.75  # stop at -25%
        assert s.BROKER_STOP_MIN_STEP_FRAC == 0.05
        assert s.BROKER_STOP_MIN_REPLACE_SEC == 30.0
        assert s.BROKER_STOP_MAX_ATTEMPTS == 3


# ===========================================================================
# UNIT — BrokerStopManager (placement / tracking / cleanup / fallback)
# ===========================================================================

import asyncio  # noqa: E402

from options_owl.execution.broker_stop import (  # noqa: E402
    BrokerStopManager,
    compute_desired_stop,
    compute_stop_price,
)
from options_owl.execution.paper_trader import (  # noqa: E402
    PaperTrader,
    SellOutcome,
)
from options_owl.execution.webull_executor import OrderResult  # noqa: E402


def _mgr_settings(**overrides):
    s = MagicMock()
    defaults = {
        "ENABLE_BROKER_STOP": True,
        "PAPER_TRADE": False,
        "BROKER_STOP_ENTRY_FRAC": 0.75,
        "BROKER_STOP_MAX_ATTEMPTS": 3,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _live_trade(**kw):
    t = {
        "id": 7,
        "ticker": "SPY",
        "strike": 500.0,
        "option_type": "call",
        "expiry_date": "2026-07-22",
        "contracts": 4,
        "webull_order_id": "WB1",
        "premium_per_contract": 2.00,
    }
    t.update(kw)
    return t


class TestComputeStopPrice:
    def test_default_frac_is_minus_25pct(self):
        assert compute_stop_price(2.00, _mgr_settings()) == 1.50

    def test_custom_frac(self):
        assert compute_stop_price(2.00, _mgr_settings(BROKER_STOP_ENTRY_FRAC=0.6)) == 1.20

    def test_zero_or_negative_entry_returns_none(self):
        assert compute_stop_price(0.0, _mgr_settings()) is None
        assert compute_stop_price(-1.0, _mgr_settings()) is None

    def test_bad_entry_returns_none(self):
        assert compute_stop_price(None, _mgr_settings()) is None
        assert compute_stop_price("x", _mgr_settings()) is None

    def test_bad_frac_falls_back(self):
        s = _mgr_settings()
        s.BROKER_STOP_ENTRY_FRAC = "nope"
        assert compute_stop_price(2.00, s) == 1.50


def _ok_stop():
    return OrderResult(success=True, order_id="ORD1", client_order_id="COID1", fill_status="SUBMITTED")


class TestManagerEnsureStop:
    @pytest.mark.asyncio
    async def test_places_and_tracks_on_success(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), ex)
        ex.place_stop_loss.assert_awaited_once()
        # stop price passed = entry(2.00) * 0.75
        assert ex.place_stop_loss.call_args.kwargs["stop_price"] == 1.50
        assert mgr.active_stops() == {7: "COID1"}

    @pytest.mark.asyncio
    async def test_idempotent_when_already_placed(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), ex)
        await mgr.ensure_stop(_live_trade(), ex)  # second cycle — must NOT re-place
        ex.place_stop_loss.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_disabled(self):
        ex = MagicMock(); ex.place_stop_loss = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings(ENABLE_BROKER_STOP=False))
        await mgr.ensure_stop(_live_trade(), ex)
        ex.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_paper(self):
        ex = MagicMock(); ex.place_stop_loss = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings(PAPER_TRADE=True))
        await mgr.ensure_stop(_live_trade(), ex)
        ex.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_no_executor(self):
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), None)  # must not raise
        assert mgr.active_stops() == {}

    @pytest.mark.asyncio
    async def test_noop_for_paper_only_trade(self):
        ex = MagicMock(); ex.place_stop_loss = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(webull_order_id=None), ex)
        ex.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_then_gives_up_to_poll_only(self):
        """Bounded retry: failure retries across cycles, then STOPS calling (poll-only fallback)."""
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=False, error="boom", fill_status="REJECTED",
        ))
        mgr = BrokerStopManager(_mgr_settings(BROKER_STOP_MAX_ATTEMPTS=3))
        for _ in range(6):  # six cycles, but only 3 attempts allowed
            await mgr.ensure_stop(_live_trade(), ex)
        assert ex.place_stop_loss.await_count == 3
        assert mgr.active_stops() == {}  # never placed → nothing to cancel; FSM protects

    @pytest.mark.asyncio
    async def test_never_raises_when_executor_throws(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(side_effect=RuntimeError("api down"))
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), ex)  # must swallow
        assert mgr.active_stops() == {}

    @pytest.mark.asyncio
    async def test_skips_when_no_entry_premium(self):
        ex = MagicMock(); ex.place_stop_loss = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(premium_per_contract=0.0), ex)
        ex.place_stop_loss.assert_not_called()


# ===========================================================================
# PHASE 3 — replace-to-trail (ratchet) + orphan reconcile
# ===========================================================================

def _ratchet_settings(**overrides):
    return _mgr_settings(**{
        "BROKER_STOP_RATCHET_ARM_PCT": 20.0,
        "BROKER_STOP_TRAIL_KEEP_FRAC": 0.75,
        "BROKER_STOP_MIN_STEP_FRAC": 0.05,
        "BROKER_STOP_MIN_REPLACE_SEC": 0.0,  # disable the interval churn-guard for deterministic tests
        **overrides,
    })


class TestComputeDesiredStop:
    def test_base_when_peak_below_arm(self):
        # entry 2.00, peak 2.30 = +15% < 20% arm → base -25% floor
        assert compute_desired_stop(2.00, 2.30, _ratchet_settings()) == 1.50

    def test_ratchets_to_trail_when_armed(self):
        # entry 2.00, peak 3.00 = +50% armed → max(1.50, 2.00, 3.00*0.75=2.25) = 2.25
        assert compute_desired_stop(2.00, 3.00, _ratchet_settings()) == 2.25

    def test_never_below_breakeven_once_armed(self):
        # entry 2.00, peak 2.50 = +25% armed, trail 2.50*0.75=1.875 < entry → breakeven 2.00
        assert compute_desired_stop(2.00, 2.50, _ratchet_settings()) == 2.00

    def test_none_peak_returns_base(self):
        assert compute_desired_stop(2.00, None, _ratchet_settings()) == 1.50

    def test_monotonic_nondecreasing_in_peak(self):
        s = _ratchet_settings()
        vals = [compute_desired_stop(2.00, p, s) for p in (2.0, 2.4, 3.0, 4.0, 5.0)]
        assert vals == sorted(vals)

    def test_bad_entry_none(self):
        assert compute_desired_stop(0.0, 3.0, _ratchet_settings()) is None


class TestRatchetReplace:
    """The ratchet MODIFIES the stop in place (replace_option) — NEVER cancel+place (the 2026-07-23 storm)."""

    @pytest.mark.asyncio
    async def test_places_base_then_modifies_in_place(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.replace_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.cancel_order = AsyncMock()
        mgr = BrokerStopManager(_ratchet_settings())
        # cycle 1: premium at entry → base stop 1.50 placed
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.00)
        assert ex.place_stop_loss.call_args.kwargs["stop_price"] == 1.50
        # cycle 2: premium runs to 3.00 → armed, MODIFY up to 2.25 (same order, no cancel/replace)
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=3.00)
        ex.replace_stop_loss.assert_awaited_once()
        assert ex.replace_stop_loss.await_args.kwargs["client_order_id"] == "C1"
        assert ex.replace_stop_loss.await_args.kwargs["stop_price"] == 2.25
        ex.cancel_order.assert_not_called()          # NEVER cancel+replace
        assert ex.place_stop_loss.await_count == 1   # only the initial placement
        assert mgr.active_stops() == {7: "C1"}       # same order, ratcheted in place

    @pytest.mark.asyncio
    async def test_ratchet_only_moves_up_never_down(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.replace_stop_loss = AsyncMock(return_value=OrderResult(success=True, client_order_id="C1"))
        mgr = BrokerStopManager(_ratchet_settings())
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=3.00)  # → 2.25
        ex.replace_stop_loss.reset_mock()
        # premium falls back to 2.40 — peak stays 3.0, desired unchanged → NO modify
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.40)
        ex.replace_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_min_step_blocks_tiny_modify(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.replace_stop_loss = AsyncMock(return_value=OrderResult(success=True, client_order_id="C1"))
        mgr = BrokerStopManager(_ratchet_settings(BROKER_STOP_MIN_STEP_FRAC=0.5))  # min move $1.00
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.50)  # base 2.00
        ex.replace_stop_loss.reset_mock()
        # desired 2.10, move 0.10 < 1.00 min step → blocked
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.80)
        ex.replace_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_modify_failure_keeps_existing_stop(self):
        """CRITICAL: a failed modify NEVER cancels — the old stop stays resting, just doesn't trail up."""
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.replace_stop_loss = AsyncMock(return_value=OrderResult(
            success=False, error="boom", fill_status="REJECTED"))
        ex.cancel_order = AsyncMock()
        mgr = BrokerStopManager(_ratchet_settings())
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.00)  # base 1.50
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=3.00)  # modify fails
        ex.cancel_order.assert_not_called()           # never cancels
        assert mgr.active_stops() == {7: "C1"}        # old stop still resting
        # stop_price stays at the base (didn't trail)
        assert mgr._state[7].stop_price == 1.50

    @pytest.mark.asyncio
    async def test_modify_exception_keeps_existing_stop(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.replace_stop_loss = AsyncMock(side_effect=RuntimeError("api down"))
        ex.cancel_order = AsyncMock()
        mgr = BrokerStopManager(_ratchet_settings())
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.00)
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=3.00)  # raises
        ex.cancel_order.assert_not_called()
        assert mgr.active_stops() == {7: "C1"}        # untouched, still protected


class TestCircuitBreaker:
    """The fuse: one blocked-sell disables broker stops process-wide + cancels all resting stops."""

    @pytest.mark.asyncio
    async def test_kill_disables_and_cancels_all(self):
        from options_owl.execution.broker_stop import broker_stops_killed, kill_broker_stops
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="O1", client_order_id="C1", fill_status="SUBMITTED"))
        ex.cancel_order = AsyncMock(return_value=True)
        mgr = BrokerStopManager(_ratchet_settings())
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=2.00)
        assert mgr.enabled is True
        assert mgr.active_stops() == {7: "C1"}
        # trip the fuse
        kill_broker_stops("test blocked sell")
        assert broker_stops_killed() is True
        assert mgr.enabled is False                    # disabled process-wide
        # next ensure_stop cancels every resting stop and drops tracking
        await mgr.ensure_stop(_live_trade(premium_per_contract=2.00), ex, current_premium=4.00)
        ex.cancel_order.assert_awaited_with("C1")
        assert mgr.active_stops() == {}

    @pytest.mark.asyncio
    async def test_killed_manager_places_nothing(self):
        from options_owl.execution.broker_stop import kill_broker_stops
        kill_broker_stops("test")
        ex = MagicMock(); ex.place_stop_loss = AsyncMock()
        mgr = BrokerStopManager(_ratchet_settings())
        await mgr.ensure_stop(_live_trade(), ex, current_premium=2.00)
        ex.place_stop_loss.assert_not_called()

    def test_sell_path_trips_fuse_on_blocked_error(self):
        """The sell chokepoint must trip the fuse when a sell is blocked by reserved qty."""
        import inspect

        from options_owl.execution.paper_trader import PaperTrader
        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "MUST_BE_CLOSE_THAN_SELL_SHORT" in src
        assert "kill_broker_stops" in src


class TestReconcileOrphans:
    def _stop_order(self, coid, symbol="SPY", strike="500.0", exp="2026-07-22", ot="CALL"):
        return {
            "order_type": "STOP_LOSS", "client_order_id": coid, "order_id": "OID",
            "stop_price": "1.50",
            "legs": [{"symbol": symbol, "strike_price": strike,
                      "option_expire_date": exp, "option_type": ot}],
        }

    @pytest.mark.asyncio
    async def test_cancels_orphan_readopts_match(self):
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(return_value=[
            self._stop_order("MATCH"),                       # matches the open trade
            self._stop_order("ORPHAN", symbol="TSLA", strike="400.0"),  # no open trade
            {"order_type": "LIMIT", "client_order_id": "IGNORE", "legs": []},  # not a stop → ignored
        ])
        ex.cancel_order = AsyncMock(return_value=True)
        mgr = BrokerStopManager(_mgr_settings())
        open_trades = [_live_trade(id=7, strike=500.0)]
        await mgr.reconcile_orphans(open_trades, ex)
        ex.cancel_order.assert_awaited_once_with("ORPHAN")  # only the orphan cancelled
        assert mgr.active_stops() == {7: "MATCH"}           # match re-adopted into tracking

    @pytest.mark.asyncio
    async def test_reconcile_noop_when_disabled(self):
        ex = MagicMock(); ex.get_open_orders = AsyncMock(return_value=[])
        mgr = BrokerStopManager(_mgr_settings(ENABLE_BROKER_STOP=False))
        await mgr.reconcile_orphans([_live_trade()], ex)
        ex.get_open_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_noop_when_paper(self):
        ex = MagicMock(); ex.get_open_orders = AsyncMock(return_value=[])
        mgr = BrokerStopManager(_mgr_settings(PAPER_TRADE=True))
        await mgr.reconcile_orphans([_live_trade()], ex)
        ex.get_open_orders.assert_not_called()


class TestTrackedStopRegistry:
    """The 2026-07-27 collision fix: the sell path cancels the resting stop by its TRACKED client_id
    (not fuzzy leg-matching, which missed on GOOG #633/#378 and tripped the fuse)."""

    @pytest.mark.asyncio
    async def test_place_registers_tracked_client_id(self):
        from options_owl.execution.broker_stop import get_active_stop_client_id
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        assert get_active_stop_client_id(7) == "COID1"  # reachable without a manager ref

    @pytest.mark.asyncio
    async def test_release_forgets_tracked_id(self):
        from options_owl.execution.broker_stop import get_active_stop_client_id
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex.cancel_order = AsyncMock(return_value=True)
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        await mgr.release(7, ex)
        assert get_active_stop_client_id(7) is None

    @pytest.mark.asyncio
    async def test_mark_failed_forgets_tracked_id(self):
        from options_owl.execution.broker_stop import get_active_stop_client_id
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        assert get_active_stop_client_id(7) == "COID1"
        # a subsequent placement failure clears the tracked id
        st = mgr._state[7]
        mgr._mark_failed(7, st, "some error")
        assert get_active_stop_client_id(7) is None

    @pytest.mark.asyncio
    async def test_release_and_confirm_cancels_by_tracked_id(self):
        from options_owl.execution.broker_stop import get_active_stop_client_id
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        ok = await mgr.release_and_confirm(7, ex)
        assert ok is True
        ex._confirm_cancelled.assert_awaited_once_with("COID1", timeout_seconds=6.0)
        assert get_active_stop_client_id(7) is None       # forgotten
        assert 7 not in mgr._state                          # dropped from tracking

    @pytest.mark.asyncio
    async def test_release_and_confirm_false_when_none_tracked(self):
        ex = MagicMock(); ex._confirm_cancelled = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings())
        ok = await mgr.release_and_confirm(999, ex)  # no stop for this trade
        assert ok is False
        ex._confirm_cancelled.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_and_confirm_never_raises_on_executor_error(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex._confirm_cancelled = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        ok = await mgr.release_and_confirm(7, ex)  # must swallow and return False
        assert ok is False

    def test_sell_path_cancels_by_tracked_id_before_fuzzy_sweep(self):
        """The sell chokepoint must look the stop up by tracked id (get_active_stop_client_id) and
        confirm-cancel it BEFORE the fuzzy get_open_orders sweep."""
        import inspect

        from options_owl.execution.paper_trader import PaperTrader
        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "get_active_stop_client_id" in src
        assert "_confirm_cancelled" in src
        # tracked-id cancel must appear before the fuzzy open-orders sweep
        assert src.index("get_active_stop_client_id") < src.index("get_open_orders")

    @pytest.mark.asyncio
    async def test_reconcile_swallows_open_orders_error(self):
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(side_effect=RuntimeError("api down"))
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.reconcile_orphans([_live_trade()], ex)  # must not raise


class TestManagerCleanup:
    @pytest.mark.asyncio
    async def test_release_cancels_and_drops(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex.cancel_order = AsyncMock(return_value=True)
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), ex)
        await mgr.release(7, ex)
        ex.cancel_order.assert_awaited_once_with("COID1")
        assert mgr.active_stops() == {}

    @pytest.mark.asyncio
    async def test_release_unknown_is_noop(self):
        ex = MagicMock(); ex.cancel_order = AsyncMock()
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.release(999, ex)
        ex.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_swallows_cancel_error(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex.cancel_order = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(), ex)
        await mgr.release(7, ex)  # must not raise
        assert mgr.active_stops() == {}  # still dropped from tracking

    @pytest.mark.asyncio
    async def test_prune_releases_vanished_only(self):
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=_ok_stop())
        ex.cancel_order = AsyncMock(return_value=True)
        mgr = BrokerStopManager(_mgr_settings())
        await mgr.ensure_stop(_live_trade(id=7), ex)
        await mgr.ensure_stop(_live_trade(id=8, webull_order_id="WB2"), ex)
        await mgr.prune_closed({7}, ex)  # 8 vanished
        assert set(mgr.active_stops().keys()) == {7}
        ex.cancel_order.assert_awaited_once_with("COID1")


# ===========================================================================
# INTEGRATION — the double-fill guard in the REAL sell path
# (close_webull_position must cancel a resting stop before selling, and ALWAYS
#  fall back to the legacy sell when the guard fails — the critical path.)
# ===========================================================================

def _guard_settings(**overrides):
    s = MagicMock()
    defaults = {
        "PAPER_TRADE": False,
        "POLYGON_API_KEY": "",
        "ENABLE_BROKER_STOP": True,
        "ENABLE_FAST_EXIT_CHASE": False,
        "WEBULL_EXIT_PER_ATTEMPT_SEC": 2.5,
        "WEBULL_EXIT_POLL_SEC": 1.0,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _guard_trader(executor, settings):
    trader = PaperTrader.__new__(PaperTrader)
    trader.settings = settings
    trader.webull_executor = executor
    trader.db_path = ":memory:"
    return trader


def _resting_stop_order():
    return {
        "client_order_id": "STOP-XYZ",
        "order_type": "STOP_LOSS",
        "legs": [{
            "symbol": "SPY", "strike_price": "500.0",
            "option_expire_date": "2026-07-22", "option_type": "CALL",
        }],
    }


def _guard_trade(retry=0):
    return {
        "id": 42, "ticker": "SPY", "strike": 500.0, "option_type": "call",
        "expiry_date": "2026-07-22", "contracts": 3, "webull_order_id": "WB123",
        "sell_retry_count": retry, "premium_per_contract": 2.00,
    }


@pytest.fixture
def _patch_sell_io(monkeypatch):
    async def _no_fresh_bid(self, trade):
        return None

    async def _noop_db(*a, **k):
        return None

    monkeypatch.setattr(PaperTrader, "_get_fresh_option_bid", _no_fresh_bid)
    monkeypatch.setattr("options_owl.execution.paper_trader._db_execute_with_retry", _noop_db)


class TestSellPathGuard:
    @pytest.mark.asyncio
    async def test_cancels_resting_stop_before_sell_on_first_attempt(self, _patch_sell_io):
        """Flag ON: attempt 0 must cancel-and-confirm the resting stop BEFORE selling."""
        calls = []
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(side_effect=lambda: (calls.append("get_open"), [_resting_stop_order()])[1])
        ex._confirm_cancelled = AsyncMock(side_effect=lambda coid, timeout_seconds=6.0: (calls.append(f"cancel:{coid}"), "CANCELLED")[1])
        ex.sell_option = AsyncMock(side_effect=lambda **k: (calls.append("sell"), OrderResult(success=True, order_id="S1", fill_status="FILLED", filled_quantity=3))[1])
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=True))
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        assert res.outcome is SellOutcome.FILLED
        # ordering: cancel the resting stop, THEN sell (no double-fill)
        assert calls.index("cancel:STOP-XYZ") < calls.index("sell")

    @pytest.mark.asyncio
    async def test_flag_off_is_legacy_identical_on_first_attempt(self, _patch_sell_io):
        """Flag OFF: attempt 0 must NOT touch open orders — byte-identical to the legacy path."""
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(return_value=[_resting_stop_order()])
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        ex.sell_option = AsyncMock(return_value=OrderResult(success=True, order_id="S1", fill_status="FILLED", filled_quantity=3))
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=False))
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        assert res.outcome is SellOutcome.FILLED
        ex.get_open_orders.assert_not_called()  # legacy: no guard on attempt 0
        ex.sell_option.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guard_failure_still_sells_fallback_to_legacy(self, _patch_sell_io):
        """CRITICAL PATH: if the guard (get_open_orders) throws, the sell MUST still happen."""
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(side_effect=RuntimeError("api down"))
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        ex.sell_option = AsyncMock(return_value=OrderResult(success=True, order_id="S1", fill_status="FILLED", filled_quantity=3))
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=True))
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        assert res.outcome is SellOutcome.FILLED
        ex.sell_option.assert_awaited_once()  # fell through to the legacy sell

    @pytest.mark.asyncio
    async def test_guard_timeout_still_sells(self, _patch_sell_io):
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(side_effect=asyncio.TimeoutError)
        ex.sell_option = AsyncMock(return_value=OrderResult(success=True, order_id="S1", fill_status="FILLED", filled_quantity=3))
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=True))
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        assert res.outcome is SellOutcome.FILLED
        ex.sell_option.assert_awaited_once()


# ===========================================================================
# E2E — full sequence, simulated, ZERO live API
# ===========================================================================

class TestBrokerStopE2E:
    @pytest.mark.asyncio
    async def test_open_place_exit_cancel_prune(self, _patch_sell_io):
        """Lifecycle: open live trade → stop placed → monitor exit cancels stop → sell fills →
        prune drops tracking. No double-fill; nothing orphaned."""
        ex = MagicMock()
        ex.place_stop_loss = AsyncMock(return_value=OrderResult(
            success=True, order_id="ORD1", client_order_id="STOP-XYZ", fill_status="SUBMITTED",
        ))
        ex.get_open_orders = AsyncMock(return_value=[_resting_stop_order()])
        ex._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        ex.sell_option = AsyncMock(return_value=OrderResult(
            success=True, order_id="S1", fill_status="FILLED", filled_quantity=3))
        ex.cancel_order = AsyncMock(return_value=True)

        mgr = BrokerStopManager(_mgr_settings())
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=True))

        # 1) monitor sees the live trade → places a resting stop
        await mgr.ensure_stop(_live_trade(id=42, strike=500.0, contracts=3, premium_per_contract=2.00), ex)
        assert mgr.active_stops() == {42: "STOP-XYZ"}

        # 2) monitor decides to exit → sell path cancels the resting stop first, then sells
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        assert res.outcome is SellOutcome.FILLED
        ex._confirm_cancelled.assert_awaited()  # the resting stop was cancelled before the sell

        # 3) next cycle: trade no longer open → prune releases + drops tracking (self-cleanup)
        await mgr.prune_closed(set(), ex)
        assert mgr.active_stops() == {}

    @pytest.mark.asyncio
    async def test_stop_fired_race_sell_finds_no_position_no_double_fill(self, _patch_sell_io):
        """The crash fires the broker stop between polls. The monitor's follow-up sell finds no
        position (sell-to-close lookup fails) → POSITION_NOT_FOUND, closed ONCE. No naked short."""
        ex = MagicMock()
        ex.get_open_orders = AsyncMock(return_value=[_resting_stop_order()])
        # the resting stop already FILLED during the cancel race
        ex._confirm_cancelled = AsyncMock(return_value="FILLED")
        # the subsequent sell-to-close finds nothing (position gone) → blocked (never a naked short)
        ex.sell_option = AsyncMock(return_value=OrderResult(
            success=False,
            error="No Webull position found for SPY $500 CALL — nothing to close",
            fill_status="UNKNOWN",
        ))
        trader = _guard_trader(ex, _guard_settings(ENABLE_BROKER_STOP=True))
        res = await trader.close_webull_position(_guard_trade(retry=0), 2.00)
        # position genuinely gone → categorized as not-found (monitor marks it closed once)
        assert res.outcome is SellOutcome.POSITION_NOT_FOUND
        ex.sell_option.assert_awaited_once()  # exactly one sell attempt — no double sell


# ===========================================================================
# SOURCE-CODE SAFETY — lock the critical-path invariants against reordering
# ===========================================================================

import inspect  # noqa: E402

from options_owl.execution.position_monitor import run_position_monitor  # noqa: E402


class TestCriticalPathSafety:
    def test_guard_is_wrapped_so_sell_always_follows(self):
        """The broker-stop cancel guard must sit in a try/except that logs+proceeds, so a guard
        failure can NEVER prevent the legacy sell (the critical path). The sell call must appear
        AFTER the guard's except handlers."""
        src = inspect.getsource(PaperTrader.close_webull_position)
        guard_idx = src.find("BROKER-STOP double-fill guard")
        assert guard_idx != -1, "broker-stop guard marker missing"
        # the guard's failure handlers both fall through (no return/raise that skips the sell)
        assert "proceeding with sell" in src
        sell_idx = src.find("self.webull_executor.sell_option")
        assert sell_idx > guard_idx, "the sell must come after the guard"

    def test_guard_gated_on_flag_or_retry(self):
        src = inspect.getsource(PaperTrader.close_webull_position)
        assert "retry_count > 0 or _broker_stop_on" in src, \
            "guard must run on retries OR when broker-stops enabled (never unconditionally on the legacy path)"

    def test_monitor_wires_ensure_prune_reconcile(self):
        src = inspect.getsource(run_position_monitor)
        assert "broker_stops.ensure_stop(" in src, "monitor must place/ratchet stops per live trade"
        assert "broker_stops.prune_closed(" in src, "monitor must self-clean vanished stops"
        assert "broker_stops.reconcile_orphans(" in src, "monitor must reconcile orphaned stops on startup"
        assert "current_premium=exit_premium" in src, "ratchet needs the fresh premium fed in"
