"""Tests for the multi-day DEAD-ON-ARRIVAL stall cut + EOD close-all (2026-07-01).

Stall cut: cut a MULTI-DAY leg that never worked (held >= N min, down >= X%, peak < Y%) —
fixes the 1-DTE put/call that bleeds an hour to the wide 50% graduated backstop (adam META
2026-07-01: peaked +1%, -47%, held ~1hr). EOD close-all: force-close multi-day at the EOD
cutoff so nothing is held overnight.
"""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

ET = ZoneInfo("America/New_York")


def _cfg(is_put=False):
    return apply_v7_wide_trail_exits(
        get_ticker_config("META", use_per_ticker=True, option_type="put" if is_put else "call"),
        is_put=is_put)


def _settings(**ov):
    d = dict(ENABLE_STALL_CUT=True, STALL_CUT_MIN_MINUTES=30.0, STALL_CUT_LOSS_PCT=30.0,
             STALL_CUT_PEAK_PCT=10.0, ENABLE_EOD_CLOSE_ALL=False,
             ENABLE_V6_PER_TICKER_CONFIG=False)
    d.update(ov)
    return SimpleNamespace(**d)


def _state(option_type="put", entry=5.0, dte=1):
    return TradeState(trade_id=361, ticker="META", option_type=option_type, entry_premium=entry,
                      entry_time=datetime(2026, 7, 1, 12, 39, tzinfo=ET), contracts=1,
                      peak_premium=entry, entry_underlying_price=557.0, dte=dte,
                      expiry_date="2026-07-02")


class TestStallCut:
    def test_dead_on_arrival_put_cut(self):
        """adam META repro: 1-DTE put, never peaked, down ~47%, held ~40min -> STALL_CUT."""
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings())
        st = _state("put", entry=5.47, dte=1)
        now = datetime(2026, 7, 1, 13, 19, tzinfo=ET)  # 40 min after entry
        a = fsm.evaluate(st, 2.90, 2.85, 2.95, now, current_underlying=558.0, minutes_to_close=160)
        assert a.should_exit and a.reason == ExitReason.STALL_CUT

    def test_spares_ran_up_then_faded(self):
        """A leg that peaked +40% then faded to -30% must NOT stall-cut (it worked)."""
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings())
        st = _state("put", entry=5.0, dte=1)
        fsm.evaluate(st, 7.0, 6.9, 7.1, datetime(2026, 7, 1, 12, 50, tzinfo=ET),
                     current_underlying=550.0, minutes_to_close=190)  # peak +40%
        a = fsm.evaluate(st, 3.5, 3.45, 3.55, datetime(2026, 7, 1, 13, 19, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=160)  # fade to -30%
        assert a.reason != ExitReason.STALL_CUT

    def test_not_before_min_minutes(self):
        """Down 30%, peaked <10%, but only 10 min held -> stall cut not yet armed."""
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings())
        st = _state("put", entry=5.0, dte=1)
        a = fsm.evaluate(st, 3.5, 3.45, 3.55, datetime(2026, 7, 1, 12, 49, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=190)
        assert a.reason != ExitReason.STALL_CUT

    def test_0dte_exempt(self):
        """0DTE is covered by the -25% premium hardstop, not the stall cut.

        is_0dte is derived from expiry_date vs the eval date, so a true 0DTE leg expires
        the SAME day (2026-07-01).
        """
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings())
        st = TradeState(trade_id=1, ticker="META", option_type="put", entry_premium=5.0,
                        entry_time=datetime(2026, 7, 1, 12, 39, tzinfo=ET), contracts=1,
                        peak_premium=5.0, entry_underlying_price=557.0, dte=0,
                        expiry_date="2026-07-01")
        a = fsm.evaluate(st, 3.5, 3.45, 3.55, datetime(2026, 7, 1, 13, 19, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=160)
        assert a.reason != ExitReason.STALL_CUT

    def test_disabled(self):
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings(ENABLE_STALL_CUT=False))
        st = _state("put", entry=5.47, dte=1)
        a = fsm.evaluate(st, 2.90, 2.85, 2.95, datetime(2026, 7, 1, 13, 19, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=160)
        assert a.reason != ExitReason.STALL_CUT


class TestEODCloseAll:
    def test_multiday_closed_at_eod_when_enabled(self):
        fsm = ExitFSM(_cfg(), settings=_settings(ENABLE_EOD_CLOSE_ALL=True))
        st = _state("call", entry=5.0, dte=1)
        a = fsm.evaluate(st, 5.1, 5.05, 5.15, datetime(2026, 7, 1, 15, 50, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=10)  # within EOD cutoff
        assert a.should_exit and a.reason == ExitReason.EOD_CUTOFF

    def test_multiday_held_when_disabled(self):
        fsm = ExitFSM(_cfg(), settings=_settings(ENABLE_EOD_CLOSE_ALL=False))
        st = _state("call", entry=5.0, dte=1)
        a = fsm.evaluate(st, 5.1, 5.05, 5.15, datetime(2026, 7, 1, 15, 50, tzinfo=ET),
                         current_underlying=558.0, minutes_to_close=10)
        assert a.reason != ExitReason.EOD_CUTOFF
