"""E2E guard for the 2026-06-19 Juneteenth incident.

Two bugs took the LIVE bots down on a market holiday:
  1. Juneteenth was missing from the NYSE holiday calendar.
  2. Three `_is_market_open()` helpers were weekday-only (no holiday awareness),
     and the startup Polygon self-test ignored the calendar entirely — so it
     crash-looped on a closed day, churning a Webull token over 19 restarts.

These tests ensure every market-open gate routes its weekend/holiday decision
through the single holiday-aware source of truth, and that the LIVE startup
self-test idles (not crash-loops) when the market is closed.
"""

from types import SimpleNamespace

import pytest

from options_owl.sourcing.utils import market_hours


@pytest.fixture
def closed_market(monkeypatch):
    """Force the shared calendar to report the market closed (e.g. a holiday)."""
    monkeypatch.setattr(market_hours, "is_trading_day", lambda now=None: False)
    monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)


def test_bot_runner_market_gate_honors_holiday(closed_market):
    # bot_runner._run_ml_scan_loop gates on this — the live bot's main entry gate.
    from options_owl import bot_runner

    assert bot_runner._is_market_open() is False


def test_scanner_market_gate_honors_holiday(closed_market):
    from options_owl.sourcing import scanner

    assert scanner._is_market_open() is False


def test_ml_pipeline_market_gate_honors_holiday(closed_market):
    from options_owl.sourcing import ml_pipeline

    assert ml_pipeline._is_market_open() is False


def test_entitlement_check_skips_when_market_closed(monkeypatch):
    """On a closed day the LIVE self-test must return cleanly — no SystemExit,
    no network call — so the bots idle instead of crash-looping."""
    from options_owl import main

    monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)

    def _no_network(*_a, **_k):
        raise AssertionError("entitlement check must not hit Polygon when market closed")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    settings = SimpleNamespace(POLYGON_API_KEY="dummy", PAPER_TRADE=False)
    # Must not raise SystemExit and must not touch the network.
    main.check_polygon_realtime_entitlement(settings)


def test_entitlement_check_proceeds_when_market_open(monkeypatch):
    """When open, the self-test must get past the calendar guard and actually
    probe Polygon (proven here by the network shim being reached)."""
    from options_owl import main

    monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: True)

    calls = {"n": 0}

    def _shim(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("stop after first probe")

    monkeypatch.setattr("urllib.request.urlopen", _shim)

    settings = SimpleNamespace(POLYGON_API_KEY="dummy", PAPER_TRADE=True)
    # PAPER mode swallows errors and continues; the point is it reached the probe.
    main.check_polygon_realtime_entitlement(settings)
    assert calls["n"] >= 1, "open-market self-test should probe Polygon"
