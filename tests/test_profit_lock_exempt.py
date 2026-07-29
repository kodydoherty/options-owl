"""Tests for the profit-lock moonshot exemption (V7_PROFIT_LOCK_PEAK_EXEMPT_PCT).

A leg peaking past the exemption threshold SKIPS the tight keep-fraction lock and rides the wide "let it
run" trail (gate 8) instead of being clipped on a 20%-of-peak dip. Validated 2026-07-14 flow sweep:
keep 0.8 + exempt >150% = +$9,875 (+41%) at +$295 DD, helps calls AND puts. Default 0 = off = unchanged.
"""

from options_owl.config.settings import Settings
from options_owl.risk.exit_v5.gates import check_profit_lock
from options_owl.risk.exit_v5.types import ExitReason


class TestPeakExempt:
    def test_off_by_default_behaves_as_before(self):
        # exempt=0: a peaked winner that fades below the keep floor still locks
        act = check_profit_lock(gain=50, peak_gain=100, keep_frac=0.8, activate_pct=25,
                                debug={}, peak_exempt_pct=0.0)
        assert act is not None and act.reason == ExitReason.PROFIT_LOCK

    def test_moonshot_exempted_rides_trail(self):
        # peak 200% >= exempt 150% → return None (hand to the wide trail), even though gain < keep floor
        act = check_profit_lock(gain=120, peak_gain=200, keep_frac=0.8, activate_pct=25,
                                debug={}, peak_exempt_pct=150.0)
        assert act is None

    def test_below_exempt_still_locks(self):
        # peak 120% < exempt 150% → the tight lock still governs the common winners
        act = check_profit_lock(gain=80, peak_gain=120, keep_frac=0.8, activate_pct=25,
                                debug={}, peak_exempt_pct=150.0)
        assert act is not None and act.reason == ExitReason.PROFIT_LOCK

    def test_exempt_boundary_inclusive(self):
        # peak exactly at the threshold is exempt (>=)
        assert check_profit_lock(gain=10, peak_gain=150, keep_frac=0.8, activate_pct=25,
                                 debug={}, peak_exempt_pct=150.0) is None
        # just below is not
        assert check_profit_lock(gain=10, peak_gain=149.9, keep_frac=0.8, activate_pct=25,
                                 debug={}, peak_exempt_pct=150.0) is not None

    def test_not_yet_activated_still_none(self):
        # peak below activate_pct → None regardless of exemption
        assert check_profit_lock(gain=5, peak_gain=20, keep_frac=0.8, activate_pct=25,
                                 debug={}, peak_exempt_pct=150.0) is None


class TestSettings:
    def test_exempt_off_by_default(self):
        assert Settings().V7_PROFIT_LOCK_PEAK_EXEMPT_PCT == 0.0
