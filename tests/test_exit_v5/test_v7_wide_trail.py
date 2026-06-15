"""Phase A — V7 wide-trail exits (ENABLE_V7_WIDE_TRAIL), CALL-only.

Locks in: flag OFF = baseline parity; flag ON = CALL configs transformed
(no ceiling, widened tiers, faster stall, scaleout/2PM off, ratchet kept);
PUTs untouched; the shared Settings object is never mutated.
"""

from options_owl.config.settings import Settings
from options_owl.risk.exit_v5.config import (
    apply_v7_wide_trail_exits,
    get_ticker_config,
)
from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge


def _bridge(**overrides):
    base = dict(ENABLE_V6_PER_TICKER_CONFIG=True, ENABLE_V6_SCALEOUT=True,
                ENABLE_V6_2PM_TIGHTEN=True, ENABLE_V6_BREAKEVEN_RATCHET=True)
    base.update(overrides)
    return V5MonitorBridge(Settings(**base))


class TestV7WideTrailTransform:
    def test_transform_zeroes_ceilings_and_widens(self):
        base = get_ticker_config("NVDA", use_per_ticker=True, option_type="call")
        v7 = apply_v7_wide_trail_exits(base)
        assert v7.profit_target_general_pct == 0.0
        assert v7.profit_target_index_0dte_pct == 0.0
        assert v7.theta_bleed_min == 60.0
        assert v7.theta_bleed_drop_pct == 25.0
        # every tier widened, clamped to <= 90
        for tier in v7.adaptive_highvol_tiers:
            assert tier.trail_width <= 90.0
        # moonshot (>=300) widened most (x1.5)
        base_moon = next(t for t in base.adaptive_highvol_tiers if t.min_peak_gain >= 300)
        v7_moon = next(t for t in v7.adaptive_highvol_tiers if t.min_peak_gain >= 300)
        assert v7_moon.trail_width == min(90.0, base_moon.trail_width * 1.5)

    def test_transform_is_pure(self):
        base = get_ticker_config("NVDA", use_per_ticker=True, option_type="call")
        before = base.theta_bleed_min
        apply_v7_wide_trail_exits(base)
        assert base.theta_bleed_min == before  # frozen dataclass, replace() not mutate


class TestV7WideTrailWiring:
    def test_flag_off_is_baseline(self):
        off = _bridge(ENABLE_V7_WIDE_TRAIL=False)
        fsm = off._get_fsm("NVDA", "call")
        assert fsm.cfg.theta_bleed_min == 120.0  # untouched baseline
        assert off._v7_settings.ENABLE_V6_SCALEOUT is True

    def test_flag_on_transforms_calls(self):
        on = _bridge(ENABLE_V7_WIDE_TRAIL=True)
        fsm = on._get_fsm("NVDA", "call")
        assert fsm.cfg.theta_bleed_min == 60.0
        assert fsm.cfg.profit_target_general_pct == 0.0
        assert on._v7_settings.ENABLE_V6_SCALEOUT is False
        assert on._v7_settings.ENABLE_V6_2PM_TIGHTEN is False
        assert on._v7_settings.ENABLE_V6_BREAKEVEN_RATCHET is True

    def test_flag_on_puts_widen_but_keep_no_limit(self):
        on = _bridge(ENABLE_V7_WIDE_TRAIL=True)
        base = get_ticker_config("NVDA", use_per_ticker=True, option_type="put")
        put = on._get_fsm("NVDA", "put")
        # PUTs KEEP their no-hold-limit (theta untouched) so they can ride crashes...
        assert put.cfg.theta_bleed_min == base.theta_bleed_min  # not forced to 60
        # ...but DO get the widened trail + scaleout-off (let the down-day tail run)
        assert put.cfg.profit_target_general_pct == 0.0
        widened = any(
            p.trail_width > b.trail_width
            for p, b in zip(put.cfg.adaptive_standard_tiers, base.adaptive_standard_tiers)
        )
        assert widened
        assert put._settings.ENABLE_V6_SCALEOUT is False

    def test_does_not_mutate_shared_settings(self):
        s = Settings(ENABLE_V7_WIDE_TRAIL=True, ENABLE_V6_PER_TICKER_CONFIG=True,
                     ENABLE_V6_SCALEOUT=True)
        V5MonitorBridge(s)
        assert s.ENABLE_V6_SCALEOUT is True  # original object never mutated
