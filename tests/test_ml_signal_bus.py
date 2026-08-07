"""Unit tests for the ML signal bus payload contract (centralize-ml-signals spec).

These are the pure-function safety tests: serialize, freshness-guard, validation, heartbeat.
No Redis / no live deps — the transport is tested separately.
"""
import math

import pytest

from options_owl.collectors import ml_signal_bus as bus


def _good_signal():
    return {
        "ticker": "IWM", "direction": "CALL", "score": 68, "premium": 0.62,
        "strike": 298.0, "expiry": "2026-08-04", "ml_confidence": 0.685,
        "underlying_price": 297.6,
        # an unknown field that must NOT survive into the payload:
        "internal_debug": {"foo": "bar"},
    }


# ── build_signal_payload ────────────────────────────────────────────────
def test_build_signal_payload_whitelists_fields():
    p = bus.build_signal_payload(_good_signal(), now=1000.0)
    assert p["type"] == bus.TYPE_SIGNAL
    assert p["t"] == 1000.0
    for k in bus.SIGNAL_FIELDS:
        assert p[k] == _good_signal()[k]
    assert "internal_debug" not in p       # unknown field dropped


def test_build_signal_payload_drops_missing_optional_fields():
    p = bus.build_signal_payload({"ticker": "SPY", "direction": "PUT", "strike": 500.0}, now=5.0)
    assert p["ticker"] == "SPY" and p["direction"] == "PUT" and p["strike"] == 500.0
    assert "ml_confidence" not in p         # not provided → simply absent


def test_build_signal_payload_stamps_time_when_none():
    p = bus.build_signal_payload(_good_signal())
    assert isinstance(p["t"], float) and p["t"] > 0


# ── heartbeat ───────────────────────────────────────────────────────────
def test_heartbeat_shape_and_detection():
    hb = bus.build_heartbeat(now=42.0)
    assert hb == {"type": bus.TYPE_HEARTBEAT, "t": 42.0}
    assert bus.is_heartbeat(hb) is True
    assert bus.is_signal(hb) is False


def test_signal_is_not_heartbeat():
    p = bus.build_signal_payload(_good_signal(), now=1.0)
    assert bus.is_signal(p) is True
    assert bus.is_heartbeat(p) is False


# ── freshness ───────────────────────────────────────────────────────────
def test_age_and_freshness():
    p = bus.build_signal_payload(_good_signal(), now=1000.0)
    assert bus.payload_age_sec(p, now=1000.0) == 0.0
    assert bus.payload_age_sec(p, now=1010.0) == 10.0
    assert bus.is_fresh(p, max_age_sec=15.0, now=1010.0) is True
    assert bus.is_fresh(p, max_age_sec=15.0, now=1020.0) is False   # 20s > 15s


def test_age_missing_or_bad_timestamp_is_infinite():
    assert math.isinf(bus.payload_age_sec({"type": "signal"}, now=1.0))
    assert math.isinf(bus.payload_age_sec({"type": "signal", "t": "not-a-number"}, now=1.0))
    assert math.isinf(bus.payload_age_sec("garbage", now=1.0))


def test_future_timestamp_clamped_to_zero_age():
    # clock skew: a payload stamped slightly in the future must not read as negative age
    assert bus.payload_age_sec({"t": 1005.0}, now=1000.0) == 0.0


# ── parse_signal (the actionability gate) ───────────────────────────────
def test_parse_valid_signal_returns_trade_kwargs():
    p = bus.build_signal_payload(_good_signal(), now=1000.0)
    kw = bus.parse_signal(p, max_age_sec=15.0, now=1005.0)
    assert kw is not None
    assert kw["ticker"] == "IWM" and kw["direction"] == "CALL" and kw["strike"] == 298.0
    assert "type" not in kw and "t" not in kw           # only trade-signal kwargs
    assert "internal_debug" not in kw


def test_parse_rejects_heartbeat():
    assert bus.parse_signal(bus.build_heartbeat(now=1.0), max_age_sec=15.0, now=1.0) is None


def test_parse_rejects_stale_signal():
    p = bus.build_signal_payload(_good_signal(), now=1000.0)
    assert bus.parse_signal(p, max_age_sec=15.0, now=1030.0) is None   # 30s old


def test_parse_rejects_missing_required_fields():
    for missing in ("ticker", "direction", "strike"):
        sig = _good_signal()
        del sig[missing]
        p = bus.build_signal_payload(sig, now=1000.0)
        assert bus.parse_signal(p, max_age_sec=15.0, now=1000.0) is None, f"should reject missing {missing}"


def test_parse_rejects_bad_direction():
    sig = _good_signal(); sig["direction"] = "SIDEWAYS"
    p = bus.build_signal_payload(sig, now=1000.0)
    assert bus.parse_signal(p, max_age_sec=15.0, now=1000.0) is None


def test_parse_rejects_empty_ticker():
    sig = _good_signal(); sig["ticker"] = "  "
    p = bus.build_signal_payload(sig, now=1000.0)
    assert bus.parse_signal(p, max_age_sec=15.0, now=1000.0) is None


def test_parse_never_raises_on_garbage():
    for junk in (None, {}, {"type": "signal"}, {"type": "x", "t": 1}, [], "str", 42):
        assert bus.parse_signal(junk, max_age_sec=15.0, now=1.0) is None


def test_roundtrip_matches_trade_signal_kwargs():
    """The parsed kwargs must be exactly what _ml_signal_to_trade_signal accepts — guards drift."""
    import inspect
    from options_owl.bot_runner import _ml_signal_to_trade_signal
    accepted = set(inspect.signature(_ml_signal_to_trade_signal).parameters)
    kw = bus.parse_signal(bus.build_signal_payload(_good_signal(), now=1.0), max_age_sec=15.0, now=1.0)
    assert set(kw).issubset(accepted), f"payload fields {set(kw)} not all accepted by trade-signal builder {accepted}"
