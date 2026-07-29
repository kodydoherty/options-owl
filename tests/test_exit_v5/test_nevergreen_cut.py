"""Tests for the never-green early cut (2026-07-23).

A trade down NEVERGREEN_CUT_LOSS_PCT that has NEVER reached +NEVERGREEN_MAX_PEAK_PCT is a ~3%-win-rate
dead trade (validated on 21d of live kody+dennis: book swings positive, ~1-2 winners clipped). Covers 0DTE
AND multi-day, fires early (before grace). Must SPARE dip-after-green recoverers (they peaked higher).
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
        get_ticker_config("NVDA", use_per_ticker=True, option_type="put" if is_put else "call"),
        is_put=is_put)


def _settings(**ov):
    d = dict(ENABLE_NEVERGREEN_CUT=True, NEVERGREEN_CUT_LOSS_PCT=8.0, NEVERGREEN_MAX_PEAK_PCT=8.0,
             NEVERGREEN_MIN_MINUTES=2.0, ENABLE_STALL_CUT=False, ENABLE_EOD_CLOSE_ALL=False,
             ENABLE_V6_PER_TICKER_CONFIG=False, ENABLE_0DTE_PREMIUM_HARDSTOP=False)
    d.update(ov)
    return SimpleNamespace(**d)


def _state(entry=2.0, dte=0):
    return TradeState(trade_id=999, ticker="NVDA", option_type="call", entry_premium=entry,
                      entry_time=datetime(2026, 7, 23, 12, 0, tzinfo=ET), contracts=2,
                      peak_premium=entry, entry_underlying_price=120.0, dte=dte,
                      expiry_date="2026-07-23" if dte == 0 else "2026-07-25")


class TestNeverGreenCut:
    def test_0dte_dead_trade_cut(self):
        """0DTE down -8%, never green, held >= 2min -> NEVERGREEN_CUT."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state(entry=2.0, dte=0)
        now = datetime(2026, 7, 23, 12, 3, tzinfo=ET)  # 3 min
        a = fsm.evaluate(st, 1.82, 1.81, 1.83, now, current_underlying=119.5, minutes_to_close=120)  # -9%
        assert a.should_exit and a.reason == ExitReason.NEVERGREEN_CUT

    def test_multiday_dead_trade_cut(self):
        """Multi-day also covered (unlike the 0DTE-only stall behavior)."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state(entry=2.0, dte=2)
        now = datetime(2026, 7, 23, 12, 3, tzinfo=ET)
        a = fsm.evaluate(st, 1.83, 1.82, 1.84, now, current_underlying=119.5, minutes_to_close=120)
        assert a.should_exit and a.reason == ExitReason.NEVERGREEN_CUT

    def test_spares_dip_after_green(self):
        """Went green (+15%) first, THEN dipped -8% -> must NOT never-green cut (it worked)."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state(entry=2.0, dte=0)
        # first tick: peak to +15% ($2.30)
        fsm.evaluate(st, 2.30, 2.29, 2.31, datetime(2026, 7, 23, 12, 1, tzinfo=ET),
                     current_underlying=121.0, minutes_to_close=125)
        # later: dip to -8%
        a = fsm.evaluate(st, 1.84, 1.83, 1.85, datetime(2026, 7, 23, 12, 4, tzinfo=ET),
                         current_underlying=119.5, minutes_to_close=120)
        assert a.reason != ExitReason.NEVERGREEN_CUT

    def test_not_before_min_minutes(self):
        """Down -8%, never green, but only 1 min held -> not armed yet."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state(entry=2.0, dte=0)
        now = datetime(2026, 7, 23, 12, 1, tzinfo=ET)  # 1 min
        a = fsm.evaluate(st, 1.84, 1.83, 1.85, now, current_underlying=119.5, minutes_to_close=120)
        assert a.reason != ExitReason.NEVERGREEN_CUT

    def test_not_cut_if_only_small_loss(self):
        """Never green but only down -4% (< 8%) -> not cut (give it room)."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state(entry=2.0, dte=0)
        now = datetime(2026, 7, 23, 12, 3, tzinfo=ET)
        a = fsm.evaluate(st, 1.92, 1.91, 1.93, now, current_underlying=119.8, minutes_to_close=120)
        assert a.reason != ExitReason.NEVERGREEN_CUT

    def test_disabled_by_default(self):
        """Flag off -> never fires."""
        fsm = ExitFSM(_cfg(), settings=_settings(ENABLE_NEVERGREEN_CUT=False))
        st = _state(entry=2.0, dte=0)
        now = datetime(2026, 7, 23, 12, 3, tzinfo=ET)
        a = fsm.evaluate(st, 1.84, 1.83, 1.85, now, current_underlying=119.5, minutes_to_close=120)
        assert a.reason != ExitReason.NEVERGREEN_CUT
