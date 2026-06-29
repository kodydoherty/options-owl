"""Stepping-tier profit lock (2026-06-26): every N% of peak gain, ratchet a HARD floor up
so a big winner can't round-trip to zero (the "+300% SPY → 0" complaint). Monotonic;
applies to calls AND puts; layered on the V7 trail.
"""

from datetime import datetime

from options_owl.risk.exit_v5.gates import check_profit_step_lock
from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
from options_owl.risk.exit_v5.types import ExitReason


class TestStepLockGate:
    """Pure gate: floor = one step below the highest step the peak reached, monotonic."""

    def test_no_arm_below_one_step(self):
        action, floor = check_profit_step_lock(40, 49, 50.0, 0.0, {})
        assert action is None and floor == 0.0

    def test_floor_inert_in_first_step_band(self):
        # peak +90% → highest step +50% → floor 0 (break-even ratchet covers this band)
        action, floor = check_profit_step_lock(10, 90, 50.0, 0.0, {})
        assert action is None and floor == 0.0

    def test_arms_and_holds_above_floor(self):
        # peak +120% → floor +50%; gain +60% is above → HOLD, floor armed at 50
        action, floor = check_profit_step_lock(60, 120, 50.0, 0.0, {})
        assert action is None and floor == 50.0

    def test_exits_below_locked_floor(self):
        # peak +120% → floor +50%; gain +40% < +50% → EXIT
        action, floor = check_profit_step_lock(40, 120, 50.0, 0.0, {})
        assert action is not None and action.should_exit
        assert action.reason == ExitReason.PROFIT_STEP_LOCK
        assert floor == 50.0

    def test_big_winner_locks_high_floor(self):
        # the headline case: a +300% peak locks a +250% floor — can't give back to 0
        action, floor = check_profit_step_lock(200, 300, 50.0, 0.0, {})
        assert action is not None and floor == 250.0  # +200% < +250% floor → exit

    def test_floor_is_monotonic(self):
        # already locked +150%; peak now reads +120% (noise) → floor must NOT drop
        action, floor = check_profit_step_lock(100, 120, 50.0, 150.0, {})
        assert floor == 150.0
        assert action is not None and action.should_exit  # +100% < +150% locked floor

    def test_step_100_variant(self):
        # step=100: peak +300% → highest step +300% → floor +200%
        action, floor = check_profit_step_lock(150, 300, 100.0, 0.0, {})
        assert floor == 200.0 and action is not None  # +150% < +200%


class _Settings:
    ENABLE_PROFIT_STEP_LOCK = True
    PROFIT_STEP_LOCK_PCT = 50.0


def _trade(option_type="call", premium=1.00):
    # AAPL (STANDARD category) isolates the step-lock: SPY/QQQ (INDEX) have the
    # profit_target gate that preempts at +30% under default config (V7 wide-trail
    # removes that ceiling in prod, so step-lock then applies to index too).
    return {
        "id": 1, "ticker": "AAPL", "option_type": option_type,
        "premium_per_contract": premium, "contracts": 5,
        "opened_at": "2026-04-28T14:00:00", "score": 85, "entry_price": 230.0,
        "mfe_premium": None, "strike": 230.0, "status": "open",
        "webull_entry_fill_price": premium,
    }


def _now(h=10, m=0):
    return datetime(2026, 4, 28, h, m, 0)


class TestStepLockInFSM:
    """End-to-end through the bridge: a peaked trade that gives back exits via step-lock."""

    def _peak_then_dip(self, option_type):
        bridge = V5MonitorBridge(_Settings())
        t = _trade(option_type=option_type, premium=1.00)
        bridge.evaluate(t, 1.00, 230.0, _now())          # entry
        bridge.evaluate(t, 3.00, 230.0, _now(10, 5))     # +200% peak → floor +150%
        # dip to +100% (premium 2.00) — below the +150% locked floor
        reason, desc = bridge.evaluate(t, 2.00, 230.0, _now(10, 6))
        return reason

    def test_call_exits_on_stepback(self):
        assert self._peak_then_dip("call") == "profit_step_lock"

    def test_put_exits_on_stepback(self):
        # puts get the step lock too (no V7 profit-lock for puts — this is their ratchet)
        assert self._peak_then_dip("put") == "profit_step_lock"

    def test_disabled_does_not_fire(self):
        class Off:
            ENABLE_PROFIT_STEP_LOCK = False
        bridge = V5MonitorBridge(Off())
        t = _trade(premium=1.00)
        bridge.evaluate(t, 1.00, 500.0, _now())
        bridge.evaluate(t, 3.00, 500.0, _now(10, 5))
        reason, _ = bridge.evaluate(t, 2.00, 500.0, _now(10, 6))
        assert reason != "profit_step_lock"
