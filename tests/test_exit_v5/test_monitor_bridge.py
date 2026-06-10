"""Tests for V5MonitorBridge — the integration layer between position_monitor and FSM."""

from datetime import datetime

from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge


class FakeSettings:
    """Minimal Settings stub for bridge initialization."""
    pass


def _make_trade(
    trade_id: int = 1,
    ticker: str = "SPY",
    premium: float = 1.00,
    contracts: int = 5,
    opened_at: str = "2026-04-28T14:00:00",  # UTC (= 10:00 AM ET)
    score: int = 85,
    **kwargs,
) -> dict:
    """Create a trade dict matching DB schema. Times are UTC (as stored in DB)."""
    d = {
        "id": trade_id,
        "ticker": ticker,
        "option_type": "call",
        "premium_per_contract": premium,
        "contracts": contracts,
        "opened_at": opened_at,
        "score": score,
        "entry_price": 500.0,
        "mfe_premium": None,
        "strike": 500.0,
        "status": "open",
    }
    d.update(kwargs)
    return d


def _now_et(hour: int = 10, minute: int = 30) -> datetime:
    return datetime(2026, 4, 28, hour, minute, 0)


class TestBridgeStateCreation:

    def test_creates_state_on_first_call(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade()
        reason, desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert 1 in bridge._states
        assert bridge._states[1].ticker == "SPY"
        assert bridge._states[1].entry_premium == 1.00

    def test_reuses_state_on_subsequent_calls(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade()
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        state1 = bridge._states[1]
        bridge.evaluate(trade, 1.10, 501.0, _now_et(10, 0))
        state2 = bridge._states[1]
        assert state1 is state2  # same object

    def test_category_assigned_on_create(self):
        from options_owl.risk.exit_v5.config import TickerCategory
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(ticker="TSLA")
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert bridge._states[1].category == TickerCategory.HIGH_VOL

    def test_cleanup_removes_state(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade()
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert 1 in bridge._states
        bridge.cleanup_trade(1)
        assert 1 not in bridge._states


class TestBridgeEvaluation:

    def test_hold_returns_none(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade()
        reason, desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert reason is None
        assert desc == ""

    def test_hard_stop_returns_reason(self):
        bridge = V5MonitorBridge(FakeSettings())
        # No underlying data → backstop at 65% (no confirmed stop without underlying)
        trade = _make_trade(premium=2.00, opened_at="2026-04-28T13:30:00",
                            entry_price=0.0)
        # 26min after fill, premium dropped from 2.00 to 0.60 = -70% > 65% backstop
        reason, desc = bridge.evaluate(trade, 0.60, 0.0, _now_et(9, 56))
        assert reason == "stop_loss"
        assert "[V5]" in desc

    def test_eod_cutoff_returns_reason(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(opened_at="2026-04-28T14:00:00")
        # Near close → EOD cutoff (minutes_to_close = 10)
        reason, desc = bridge.evaluate(trade, 1.50, 505.0, _now_et(15, 50))
        assert reason == "eod_cutoff"

    def test_peak_premium_tracked_across_calls(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(opened_at="2026-04-28T13:30:00")
        # Price rises
        bridge.evaluate(trade, 1.50, 505.0, _now_et(9, 32))
        assert bridge._states[1].peak_premium == 1.50
        # Price falls
        bridge.evaluate(trade, 1.30, 502.0, _now_et(9, 33))
        assert bridge._states[1].peak_premium == 1.50  # still 1.50

    def test_contracts_sync_from_db(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(contracts=10, opened_at="2026-04-28T13:30:00")
        bridge.evaluate(trade, 1.00, 500.0, _now_et(9, 32))
        assert bridge._states[1].contracts == 10
        # Simulate partial close in DB
        trade["contracts"] = 7
        bridge.evaluate(trade, 1.10, 501.0, _now_et(9, 33))
        assert bridge._states[1].contracts == 7

    def test_profit_target_reason_mapped(self):
        """PROFIT_TARGET maps to 'profit_target' for DB."""
        bridge = V5MonitorBridge(FakeSettings())
        # SPY = index, 0DTE, gain > 30%
        trade = _make_trade(ticker="SPY", premium=1.00, opened_at="2026-04-28T13:30:00")
        # Past grace (6min), up 35%
        reason, desc = bridge.evaluate(trade, 1.35, 501.0, _now_et(9, 36))
        assert reason == "profit_target"
        assert "[V5]" in desc

    def test_theta_timer_reason_mapped(self):
        """THETA_TIMER maps to 'theta_timer' for DB."""
        from options_owl.risk.exit_v5.monitor_bridge import _REASON_MAP
        from options_owl.risk.exit_v5.types import ExitReason
        assert _REASON_MAP[ExitReason.THETA_TIMER] == "theta_timer"

    def test_reason_map_covers_all_exit_reasons(self):
        """Every ExitReason must have a mapping — prevents silent fallthrough."""
        from options_owl.risk.exit_v5.monitor_bridge import _REASON_MAP
        from options_owl.risk.exit_v5.types import ExitReason
        for reason in ExitReason:
            assert reason in _REASON_MAP, (
                f"ExitReason.{reason.name} missing from _REASON_MAP — "
                f"add it to monitor_bridge.py"
            )


class _V6Settings:
    """Settings stub with V6 ratchet/scaleout enabled."""

    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0
    ENABLE_V6_SCALEOUT = True
    V6_SCALEOUT_GAIN_PCT = 20.0
    V6_SCALEOUT_FRACTION = 0.333
    V6_SCALEOUT_MIN_CONTRACTS = 3
    ENABLE_V6_PER_TICKER_CONFIG = False


class TestRestartDurability:
    """FIX 3: TradeState rebuilt from the DB after a restart must be correct
    for DCA'd trades, must not double-scaleout, and must keep ratchet protection
    armed from peak gain (not current gain)."""

    def test_dca_trade_uses_blended_entry(self):
        """A DCA'd trade must use premium_per_contract (blended avg), NOT the
        original webull_entry_fill_price."""
        bridge = V5MonitorBridge(FakeSettings())
        # Original fill $2.00, then DCA'd down → blended $1.50. dca_last_add_at set.
        trade = _make_trade(
            premium=1.50,  # premium_per_contract = blended avg
            webull_entry_fill_price=2.00,  # original first fill
            dca_last_add_at="2026-04-28T14:05:00",
        )
        bridge.evaluate(trade, 1.50, 500.0, _now_et(10, 0))
        assert bridge._states[1].entry_premium == 1.50  # blended, not 2.00

    def test_dca_detected_by_price_divergence(self):
        """Even without dca_last_add_at, a blended price diverging from the
        first fill beyond rounding signals a DCA → use blended."""
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(
            premium=1.40,
            webull_entry_fill_price=2.00,
        )
        bridge.evaluate(trade, 1.40, 500.0, _now_et(10, 0))
        assert bridge._states[1].entry_premium == 1.40

    def test_non_dca_trade_uses_webull_fill(self):
        """Normal (non-DCA) trade still prefers webull_entry_fill_price."""
        bridge = V5MonitorBridge(FakeSettings())
        # Fill matches premium within rounding → no DCA detected.
        trade = _make_trade(
            premium=1.83,
            webull_entry_fill_price=1.83,
        )
        bridge.evaluate(trade, 1.83, 500.0, _now_et(10, 0))
        assert bridge._states[1].entry_premium == 1.83

    def test_scaled_out_restored_no_double_scaleout(self):
        """A trade that already scaled out (child row flagged) must NOT scale
        out again after restart."""
        bridge = V5MonitorBridge(_V6Settings())
        trade = _make_trade(
            premium=1.00, contracts=6,
            opened_at="2026-04-28T13:30:00",
            _scaled_out_restore=True,  # monitor injects this on restart
        )
        # +25% gain past grace would normally trigger scaleout — but it already did.
        reason, _desc = bridge.evaluate(trade, 1.25, 505.0, _now_et(9, 40))
        assert bridge._states[1].scaled_out is True
        assert reason != "scaleout_20"

    def test_no_scaleout_flag_allows_scaleout(self):
        """Control: without the restore flag, scaleout still fires normally."""
        bridge = V5MonitorBridge(_V6Settings())
        trade = _make_trade(
            premium=1.00, contracts=6,
            opened_at="2026-04-28T13:30:00",
        )
        reason, _desc = bridge.evaluate(trade, 1.25, 505.0, _now_et(9, 40))
        assert reason == "scaleout_20"

    def test_ratchet_armed_from_peak_on_restart(self):
        """A trade that peaked >= +20% (mfe) but now sits at a loss must keep
        breakeven-ratchet protection armed after restart → exits at breakeven."""
        bridge = V5MonitorBridge(_V6Settings())
        # entry $1.00, peaked $1.30 (+30%) per mfe, now sitting at $0.90 (-10%).
        trade = _make_trade(
            premium=1.00,
            webull_entry_fill_price=1.00,
            mfe_premium=1.30,
            opened_at="2026-04-28T13:30:00",
        )
        reason, desc = bridge.evaluate(trade, 0.90, 505.0, _now_et(9, 40))
        assert bridge._states[1].breakeven_ratchet_armed is True
        assert reason == "breakeven_ratchet"

    def test_ratchet_not_armed_when_peak_below_trigger(self):
        """A trade that never peaked to the trigger must not pre-arm the
        ratchet on restart."""
        bridge = V5MonitorBridge(_V6Settings())
        # peaked only +10% (mfe 1.10), now at -5%.
        trade = _make_trade(
            premium=1.00,
            webull_entry_fill_price=1.00,
            mfe_premium=1.10,
            opened_at="2026-04-28T13:30:00",
        )
        reason, _desc = bridge.evaluate(trade, 0.95, 505.0, _now_et(9, 40))
        assert bridge._states[1].breakeven_ratchet_armed is False
        assert reason != "breakeven_ratchet"

    def test_peak_never_below_entry(self):
        """Guard: a stale/low mfe must not produce peak < entry."""
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(premium=1.00, mfe_premium=0.50)
        bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
        assert bridge._states[1].peak_premium >= bridge._states[1].entry_premium


class TestBidDisappearanceGate:
    """Regression: a REAL zero bid must reach the gate and trigger an exit.

    The old bridge did `trade.get("bid", 0.0) or 0.0` then synthesized a
    positive bid whenever bid <= 0, so a genuine no-bid (the exact condition
    the gate exists to catch) was masked and the gate could never fire.
    """

    def test_real_zero_bid_triggers_exit(self):
        bridge = V5MonitorBridge(FakeSettings())
        # Real NBBO present, bid collapsed to 0 (no buyers) while the ask is
        # still quoted wide. Premium (mid) flat vs entry so no stop/backstop
        # fires first — isolating the bid-disappearance gate.
        trade = _make_trade(premium=1.00, bid=0.0, ask=2.00)
        reason = None
        # >= 30s of zero bid at 5s polls → bid_disappearance.
        for _ in range(8):
            reason, _desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
            if reason:
                break
        assert reason == "bid_disappearance"

    def test_absent_bid_does_not_trigger(self):
        bridge = V5MonitorBridge(FakeSettings())
        # No NBBO supplied → bridge estimates a positive bid; gate stays inert.
        trade = _make_trade(premium=1.00)
        assert "bid" not in trade
        for _ in range(10):
            reason, _desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
            assert reason != "bid_disappearance"

    def test_positive_real_bid_does_not_trigger(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(premium=1.00, bid=0.98, ask=1.02)
        for _ in range(10):
            reason, _desc = bridge.evaluate(trade, 1.00, 500.0, _now_et(10, 0))
            assert reason != "bid_disappearance"
