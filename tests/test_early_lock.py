"""Tests for the early small-win floor gate (FSM gate 3.55, ENABLE_EARLY_LOCK).

Plugs the "peaked +10-15% then round-tripped to the -25% hardstop" gap. Validated 2026-07-14
(flow +$6.4k, ML rescue +$11.9k, ~zero runner clip at arm +12%).
"""

from options_owl.config.settings import Settings
from options_owl.risk.exit_v5.gates import check_early_lock
from options_owl.risk.exit_v5.types import ExitReason


class TestCheckEarlyLock:
    def test_not_armed_below_arm_threshold(self):
        # peaked only +8%, never reached +12% arm → no floor, even at a loss
        assert check_early_lock(gain=-5, peak_gain=8, arm_pct=12, floor_pct=3, debug={}) is None

    def test_armed_but_still_above_floor_holds(self):
        # peaked +20%, currently +8% (> floor +3) → still climbing/holding, don't exit
        assert check_early_lock(gain=8, peak_gain=20, arm_pct=12, floor_pct=3, debug={}) is None

    def test_armed_and_falls_to_floor_exits(self):
        act = check_early_lock(gain=3, peak_gain=15, arm_pct=12, floor_pct=3, debug={})
        assert act is not None and act.should_exit
        assert act.reason == ExitReason.EARLY_LOCK

    def test_armed_and_round_trips_below_floor_exits(self):
        # the exact pain case: peaked +14%, now +1% → lock the small win before it goes red
        act = check_early_lock(gain=1, peak_gain=14, arm_pct=12, floor_pct=3, debug={})
        assert act is not None and act.should_exit

    def test_armed_and_negative_exits(self):
        act = check_early_lock(gain=-2, peak_gain=13, arm_pct=12, floor_pct=3, debug={})
        assert act is not None and act.should_exit

    def test_climbing_runner_not_clipped(self):
        # a trade that peaks and keeps climbing never falls to the floor → never fires (no runner clip)
        for g in (12, 25, 60, 150):
            assert check_early_lock(gain=g, peak_gain=g, arm_pct=12, floor_pct=3, debug={}) is None

    def test_arm_boundary_inclusive(self):
        # peak exactly at arm arms it; gain exactly at floor exits (<=)
        assert check_early_lock(gain=3, peak_gain=12, arm_pct=12, floor_pct=3, debug={}) is not None
        # peak just below arm does not
        assert check_early_lock(gain=3, peak_gain=11.9, arm_pct=12, floor_pct=3, debug={}) is None

    def test_breakeven_floor_variant(self):
        # floor_pct=0 = "never let a green trade go red"
        assert check_early_lock(gain=0, peak_gain=15, arm_pct=12, floor_pct=0, debug={}) is not None
        assert check_early_lock(gain=2, peak_gain=15, arm_pct=12, floor_pct=0, debug={}) is None


class TestSettings:
    def test_disabled_by_default(self):
        assert Settings().ENABLE_EARLY_LOCK is False

    def test_default_arm_and_floor(self):
        s = Settings()
        assert s.EARLY_LOCK_ARM_PCT == 12.0
        assert s.EARLY_LOCK_FLOOR_PCT == 3.0
