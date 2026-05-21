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
