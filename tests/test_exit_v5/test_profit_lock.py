"""Tests for the V7 CALL-only profit-lock gate (FSM gate 3.6).

Once a CALL peaks >= activate%, exit when the current gain falls below keep_frac of the peak gain
(locks profit instead of riding the wide trail to break-even). PUTs are exempt (caller-gated) — a
tight give-back clips their slow-building crashes, so puts keep the V7 wide trail.
"""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.gates import check_profit_lock
from options_owl.risk.exit_v5.types import ExitReason

ET = ZoneInfo("America/New_York")


def _v7_cfg(is_put=False):
    """Prod-faithful config: V7 wide trail (profit targets disabled), HIGH_VOL ticker."""
    return apply_v7_wide_trail_exits(
        get_ticker_config("TSLA", use_per_ticker=True, option_type="put" if is_put else "call"),
        is_put=is_put)


def _now(h=10, m=30):
    return datetime(2026, 6, 16, h, m, tzinfo=ET)


def _settings(**ov):
    d = dict(ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
             V7_PROFIT_LOCK_ACTIVATE_PCT=30.0, ENABLE_V6_PER_TICKER_CONFIG=False)
    d.update(ov)
    return SimpleNamespace(**d)


def _state(option_type="call", entry=1.0):
    return TradeState(trade_id=1, ticker="TSLA", option_type=option_type, entry_premium=entry,
                      entry_time=_now(10, 0), contracts=5, peak_premium=entry,
                      entry_underlying_price=100.0, dte=0, expiry_date="2026-06-16")


class TestProfitLockGate:
    def test_fires_when_gain_below_keep_of_peak(self):
        # peak +100%, keep 60% -> floor +60%; gain +50% < 60 -> exit
        a = check_profit_lock(gain=50.0, peak_gain=100.0, keep_frac=0.6, activate_pct=30.0, debug={})
        assert a is not None and a.should_exit
        assert a.reason == ExitReason.PROFIT_LOCK

    def test_holds_above_floor(self):
        a = check_profit_lock(gain=70.0, peak_gain=100.0, keep_frac=0.6, activate_pct=30.0, debug={})
        assert a is None

    def test_not_armed_below_activate(self):
        # only peaked +25% (< 30 activate) -> never arms even after a fade
        a = check_profit_lock(gain=5.0, peak_gain=25.0, keep_frac=0.6, activate_pct=30.0, debug={})
        assert a is None


class TestProfitLockFSM:
    def test_call_locks_profit_after_fade(self):
        fsm = ExitFSM(_v7_cfg(), settings=_settings())
        st = _state("call", entry=1.0)
        # past grace, peak to +100%
        fsm.evaluate(st, 2.0, 1.95, 2.05, _now(10, 10), current_underlying=101.0, minutes_to_close=120)
        # fade to +50% (< 60% of the +100% peak) -> profit-lock fires
        a = fsm.evaluate(st, 1.5, 1.45, 1.55, _now(10, 12), current_underlying=100.5, minutes_to_close=120)
        assert a.should_exit and a.reason == ExitReason.PROFIT_LOCK

    def test_put_is_exempt(self):
        """Same fade on a PUT must NOT trigger profit-lock by default (call-only)."""
        fsm = ExitFSM(_v7_cfg(is_put=True), settings=_settings())
        st = _state("put", entry=1.0)
        fsm.evaluate(st, 2.0, 1.95, 2.05, _now(10, 10), current_underlying=99.0, minutes_to_close=120)
        a = fsm.evaluate(st, 1.5, 1.45, 1.55, _now(10, 12), current_underlying=99.5, minutes_to_close=120)
        assert a.reason != ExitReason.PROFIT_LOCK

    def test_put_locks_when_opted_in(self):
        """V7_PROFIT_LOCK_PUTS=True opts PUTs into profit-lock (the 2026-06-29 canary)."""
        fsm = ExitFSM(_v7_cfg(is_put=True), settings=_settings(V7_PROFIT_LOCK_PUTS=True))
        st = _state("put", entry=1.0)
        fsm.evaluate(st, 2.0, 1.95, 2.05, _now(10, 10), current_underlying=99.0, minutes_to_close=120)
        a = fsm.evaluate(st, 1.5, 1.45, 1.55, _now(10, 12), current_underlying=99.5, minutes_to_close=120)
        assert a.should_exit and a.reason == ExitReason.PROFIT_LOCK

    def test_disabled_flag_no_lock(self):
        fsm = ExitFSM(_v7_cfg(), settings=_settings(ENABLE_V7_PROFIT_LOCK=False))
        st = _state("call", entry=1.0)
        fsm.evaluate(st, 2.0, 1.95, 2.05, _now(10, 10), current_underlying=101.0, minutes_to_close=120)
        a = fsm.evaluate(st, 1.5, 1.45, 1.55, _now(10, 12), current_underlying=100.5, minutes_to_close=120)
        assert a.reason != ExitReason.PROFIT_LOCK
