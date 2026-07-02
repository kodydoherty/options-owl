"""Multi-day CALL premium hard-stop (2026-07-02).

Extends the 0DTE -25% premium hardstop to MULTI-DAY CALL legs, which otherwise ride the
wide 30/50% graduated backstop (live confirmed_stop losers averaged -43%). CALLS ONLY —
puts ride slow-building crashes and a tight cut clips those winners (3mo sweep: best PUT
cap = none; best CALL cap = -25%, +$3,730 / +8%). Fires even during grace, top priority.
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
    d = dict(ENABLE_MULTIDAY_CALL_HARDSTOP=True, MULTIDAY_CALL_HARDSTOP_PCT=25.0,
             ENABLE_0DTE_PREMIUM_HARDSTOP=True, PREMIUM_HARDSTOP_0DTE_PCT=25.0,
             ENABLE_STALL_CUT=False, ENABLE_EOD_CLOSE_ALL=False,
             ENABLE_V6_PER_TICKER_CONFIG=False)
    d.update(ov)
    return SimpleNamespace(**d)


def _state(option_type="call", entry=5.0, dte=1, expiry="2026-07-02"):
    return TradeState(trade_id=400, ticker="META", option_type=option_type, entry_premium=entry,
                      entry_time=datetime(2026, 7, 1, 12, 39, tzinfo=ET), contracts=1,
                      peak_premium=entry, entry_underlying_price=557.0, dte=dte,
                      expiry_date=expiry)


class TestMultidayCallHardstop:
    def test_multiday_call_cut_at_25pct(self):
        """Multi-day CALL down -25% from entry → MULTIDAY_CALL_HARDSTOP."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state("call", entry=5.0, dte=1)
        # 3.75 = -25%. Eval 20 min after entry (past grace).
        a = fsm.evaluate(st, 3.75, 3.70, 3.80, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert a.should_exit and a.reason == ExitReason.MULTIDAY_CALL_HARDSTOP

    def test_fires_even_during_grace(self):
        """The hard-stop is top priority — it must fire even inside the grace window."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state("call", entry=5.0, dte=1)
        # 2 min after entry (well inside 5-min grace), down -30%.
        a = fsm.evaluate(st, 3.50, 3.45, 3.55, datetime(2026, 7, 1, 12, 41, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert a.should_exit and a.reason == ExitReason.MULTIDAY_CALL_HARDSTOP

    def test_puts_excluded(self):
        """PUTs must NOT hit this gate — they ride slow crashes (best cap = none)."""
        fsm = ExitFSM(_cfg(is_put=True), settings=_settings())
        st = _state("put", entry=5.0, dte=1)
        a = fsm.evaluate(st, 3.50, 3.45, 3.55, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=560.0, minutes_to_close=180)
        assert a.reason != ExitReason.MULTIDAY_CALL_HARDSTOP

    def test_above_threshold_holds(self):
        """Down only -20% → below the -25% cut, must NOT fire."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state("call", entry=5.0, dte=1)
        a = fsm.evaluate(st, 4.00, 3.95, 4.05, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=556.0, minutes_to_close=180)
        assert a.reason != ExitReason.MULTIDAY_CALL_HARDSTOP

    def test_0dte_uses_the_0dte_gate_not_this_one(self):
        """A true 0DTE call at -25% exits via the 0DTE premium hardstop, not the multi-day gate."""
        fsm = ExitFSM(_cfg(), settings=_settings())
        st = _state("call", entry=5.0, dte=0, expiry="2026-07-01")
        a = fsm.evaluate(st, 3.75, 3.70, 3.80, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert a.should_exit and a.reason == ExitReason.PREMIUM_HARDSTOP

    def test_disabled(self):
        fsm = ExitFSM(_cfg(), settings=_settings(ENABLE_MULTIDAY_CALL_HARDSTOP=False))
        st = _state("call", entry=5.0, dte=1)
        a = fsm.evaluate(st, 3.75, 3.70, 3.80, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert a.reason != ExitReason.MULTIDAY_CALL_HARDSTOP

    def test_custom_threshold(self):
        """Threshold is configurable (e.g. -30%)."""
        fsm = ExitFSM(_cfg(), settings=_settings(MULTIDAY_CALL_HARDSTOP_PCT=30.0))
        st = _state("call", entry=5.0, dte=1)
        # -25% should NOT fire when the cut is set to -30%...
        a = fsm.evaluate(st, 3.75, 3.70, 3.80, datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert a.reason != ExitReason.MULTIDAY_CALL_HARDSTOP
        # ...but -30% does.
        b = fsm.evaluate(_state("call", entry=5.0, dte=1), 3.50, 3.45, 3.55,
                         datetime(2026, 7, 1, 12, 59, tzinfo=ET),
                         current_underlying=555.0, minutes_to_close=180)
        assert b.should_exit and b.reason == ExitReason.MULTIDAY_CALL_HARDSTOP
