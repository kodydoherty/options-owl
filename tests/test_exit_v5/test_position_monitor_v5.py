"""Tests for v5 FSM routing in position_monitor.

Covers the EXIT_ENGINE=v5 dispatch path, cleanup on full close,
and bridge evaluation format.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock


from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSettings:
    EXIT_ENGINE = "v5"
    ENABLE_VINNY_STRATEGY = False
    ENABLE_PARTIAL_PROFITS = False
    ENABLE_SCALE_OUT = False


def _make_trade(
    trade_id: int = 1,
    ticker: str = "SPY",
    premium: float = 1.00,
    contracts: int = 5,
    opened_at: str = "2026-04-28T14:00:00",  # UTC (= 10:00 AM ET)
    score: int = 85,
    **kwargs,
) -> dict:
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


# ---------------------------------------------------------------------------
# V5 routing dispatch
# ---------------------------------------------------------------------------

class TestV5RoutingDispatch:
    """Verify EXIT_ENGINE=v5 causes bridge.evaluate() to be called."""

    def test_v5_bridge_evaluate_called(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(opened_at="2026-04-28T13:30:00")
        now = datetime(2026, 4, 28, 9, 32, 0)
        reason, desc = bridge.evaluate(trade, 1.00, 500.0, now)
        # Just opened, should HOLD
        assert reason is None
        assert desc == ""

    def test_v3_setting_does_not_create_bridge(self):
        settings = FakeSettings()
        settings.EXIT_ENGINE = "v3"
        engine = getattr(settings, "EXIT_ENGINE", "v3")
        use_v5 = engine in ("v4", "v5")
        assert use_v5 is False

    def test_v5_setting_activates_bridge(self):
        settings = FakeSettings()
        settings.EXIT_ENGINE = "v5"
        engine = getattr(settings, "EXIT_ENGINE", "v3")
        use_v5 = engine in ("v4", "v5")
        assert use_v5 is True

    def test_v4_setting_also_activates_bridge(self):
        """v4 accepted for backward compatibility."""
        settings = FakeSettings()
        settings.EXIT_ENGINE = "v4"
        engine = getattr(settings, "EXIT_ENGINE", "v3")
        use_v5 = engine in ("v4", "v5")
        assert use_v5 is True

    def test_missing_exit_engine_defaults_to_v3(self):
        settings = MagicMock(spec=[])  # no attributes
        engine = getattr(settings, "EXIT_ENGINE", "v3")
        use_v5 = engine in ("v4", "v5")
        assert use_v5 is False


# ---------------------------------------------------------------------------
# Bridge cleanup
# ---------------------------------------------------------------------------

class TestBridgeCleanupIntegration:
    """Verify cleanup removes FSM state correctly."""

    def test_cleanup_after_full_close(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade()
        now = datetime(2026, 4, 28, 10, 0, 0)
        bridge.evaluate(trade, 1.00, 500.0, now)
        assert 1 in bridge._states
        bridge.cleanup_trade(1)
        assert 1 not in bridge._states

    def test_cleanup_idempotent(self):
        bridge = V5MonitorBridge(FakeSettings())
        bridge.cleanup_trade(999)  # non-existent — should not raise

    def test_multiple_trades_independent_cleanup(self):
        bridge = V5MonitorBridge(FakeSettings())
        now = datetime(2026, 4, 28, 10, 0, 0)
        bridge.evaluate(_make_trade(trade_id=1), 1.00, 500.0, now)
        bridge.evaluate(_make_trade(trade_id=2, ticker="QQQ"), 2.00, 450.0, now)
        assert len(bridge._states) == 2
        bridge.cleanup_trade(1)
        assert 1 not in bridge._states
        assert 2 in bridge._states


# ---------------------------------------------------------------------------
# Full exit lifecycle: entry → evaluate → exit reason → cleanup
# ---------------------------------------------------------------------------

class TestV5ExitLifecycle:
    """End-to-end: create state, evaluate through to exit, verify reason format."""

    def test_hard_stop_lifecycle(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(premium=2.00, contracts=5, opened_at="2026-04-28T13:30:00",
                            entry_price=0.0)

        # 26min after fill (9:30 → 9:56), premium dropped from 2.00 to 0.60 = -70% > 65% backstop
        now = datetime(2026, 4, 28, 9, 56, 0)
        reason, desc = bridge.evaluate(trade, 0.60, 0.0, now)
        assert reason == "stop_loss"
        assert "[V5]" in desc

        # Cleanup
        bridge.cleanup_trade(trade["id"])
        assert trade["id"] not in bridge._states

    def test_eod_lifecycle(self):
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(ticker="AAPL", opened_at="2026-04-28T14:00:00")

        # Hold through day
        now_hold = datetime(2026, 4, 28, 14, 0, 0)
        reason, _ = bridge.evaluate(trade, 1.50, 505.0, now_hold)
        assert reason is None  # HOLD

        # EOD
        now_eod = datetime(2026, 4, 28, 15, 50, 0)
        reason, desc = bridge.evaluate(trade, 1.50, 505.0, now_eod)
        assert reason == "eod_cutoff"

        bridge.cleanup_trade(trade["id"])

    def test_adaptive_trail_lifecycle(self):
        """Trade peaks then drops → adaptive trail fires."""
        bridge = V5MonitorBridge(FakeSettings())
        trade = _make_trade(premium=1.00, opened_at="2026-04-28T13:30:00")

        # Get past grace
        now = datetime(2026, 4, 28, 9, 32, 0)
        bridge.evaluate(trade, 1.00, 500.0, now)

        # Push peak to +80%, set underlying to confirm so scalp doesn't fire
        bridge._states[1].peak_premium = 1.80
        bridge._states[1].entry_underlying_price = 500.0
        now2 = datetime(2026, 4, 28, 10, 16, 0)

        # Drop 42% from peak (1.80 * 0.58 = 1.044)
        # underlying up 1% (505) → confirms call, so scalp won't intercept
        reason, desc = bridge.evaluate(trade, 1.03, 505.0, now2)
        assert reason == "adaptive_trail"
        assert "[V5]" in desc

        bridge.cleanup_trade(trade["id"])
