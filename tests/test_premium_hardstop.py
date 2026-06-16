"""0DTE premium hard-stop (FSM gate 2.5): cut a 0DTE contract down >= X% from ENTRY regardless of the
underlying — fixes the NVDA #34 case (premium melted -42% while the stock barely moved, FSM held it)."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

ET = ZoneInfo("America/New_York")


def _now(h=11, m=0):
    return datetime(2026, 6, 16, h, m, tzinfo=ET)


def _settings(**ov):
    d = dict(ENABLE_0DTE_PREMIUM_HARDSTOP=True, PREMIUM_HARDSTOP_0DTE_PCT=25.0,
             ENABLE_V6_PER_TICKER_CONFIG=False)
    d.update(ov)
    return SimpleNamespace(**d)


def _cfg(is_put=False):
    return apply_v7_wide_trail_exits(
        get_ticker_config("NVDA", use_per_ticker=True, option_type="put" if is_put else "call"), is_put=is_put)


def _state(dte=0, entry=2.0):
    expiry = "2026-06-16" if dte == 0 else "2026-06-19"   # DTE is derived from expiry vs now
    return TradeState(trade_id=1, ticker="NVDA", option_type="call", entry_premium=entry,
                      entry_time=_now(10, 0), contracts=8, peak_premium=entry,
                      entry_underlying_price=210.0, dte=dte, expiry_date=expiry)


def test_fires_on_0dte_down_25_even_with_flat_underlying():
    """The NVDA case: premium -30% from entry, underlying basically flat → hard-stop must fire."""
    fsm = ExitFSM(_cfg(), settings=_settings())
    st = _state(dte=0, entry=2.0)
    # premium $1.40 = -30% off $2.00; underlying barely moved (210 -> 209.5)
    a = fsm.evaluate(st, 1.40, 1.36, 1.44, _now(11, 30), current_underlying=209.5, minutes_to_close=120)
    assert a.should_exit and a.reason == ExitReason.PREMIUM_HARDSTOP


def test_holds_above_threshold():
    fsm = ExitFSM(_cfg(), settings=_settings())
    st = _state(dte=0, entry=2.0)
    a = fsm.evaluate(st, 1.70, 1.66, 1.74, _now(11, 30), current_underlying=209.5, minutes_to_close=120)
    assert a.reason != ExitReason.PREMIUM_HARDSTOP   # -15% > -25% floor


def test_multiday_exempt():
    """Multi-day trades are NOT subject to the 0DTE hard-stop (premiums move slower)."""
    fsm = ExitFSM(_cfg(), settings=_settings())
    st = _state(dte=3, entry=2.0)
    a = fsm.evaluate(st, 1.40, 1.36, 1.44, _now(11, 30), current_underlying=209.5, minutes_to_close=120)
    assert a.reason != ExitReason.PREMIUM_HARDSTOP


def test_disabled_flag():
    fsm = ExitFSM(_cfg(), settings=_settings(ENABLE_0DTE_PREMIUM_HARDSTOP=False))
    st = _state(dte=0, entry=2.0)
    a = fsm.evaluate(st, 1.40, 1.36, 1.44, _now(11, 30), current_underlying=209.5, minutes_to_close=120)
    assert a.reason != ExitReason.PREMIUM_HARDSTOP
