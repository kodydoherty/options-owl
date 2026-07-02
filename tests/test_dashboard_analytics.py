"""Unit tests for the verbose trade-detail helpers (dashboard analytics_util).

Pure functions — no DB, no clock — so they're exhaustively testable in isolation. They power
the verbose detail view off data that already exists (trade_premium_ticks + trade_events).
"""

from __future__ import annotations

from datetime import datetime

from options_owl.dashboard.analytics_util import build_timeline, summarize_ticks


def _tick(prem, bid=None, ask=None, under=None, t="2026-07-02T10:00:00"):
    return {"captured_at": t, "premium": prem, "bid": bid, "ask": ask, "underlying_price": under}


class TestSummarizeTicks:
    def test_empty_series_is_safe(self):
        s = summarize_ticks([], entry_premium=2.0)
        assert s["n"] == 0
        # every downstream key present and None so the template never KeyErrors
        for k in ("peak_premium", "peak_gain_pct", "drawdown_from_peak_pct", "avg_spread_pct"):
            assert s[k] is None

    def test_peak_trough_and_gain(self):
        ticks = [_tick(2.0), _tick(3.0), _tick(5.0), _tick(4.0), _tick(2.5)]
        s = summarize_ticks(ticks, entry_premium=2.0, contracts=3)
        assert s["n"] == 5
        assert s["premium_max"] == 5.0
        assert s["premium_min"] == 2.0
        assert s["premium_last"] == 2.5
        assert s["peak_premium"] == 5.0
        assert s["peak_idx"] == 2
        assert s["peak_gain_pct"] == 150.0          # (5-2)/2
        assert s["trough_gain_pct"] == 0.0
        # $ swings at 3 contracts: peak (5-2)*3*100 = 900, last (2.5-2)*3*100 = 150
        assert s["peak_dollars"] == 900.0
        assert s["last_dollars"] == 150.0

    def test_drawdown_from_peak(self):
        # rises to 5 then falls to 2 → max give-back = (5-2)/5 = 60%
        s = summarize_ticks([_tick(2.0), _tick(5.0), _tick(2.0)], entry_premium=2.0)
        assert s["drawdown_from_peak_pct"] == 60.0

    def test_spread_stats(self):
        ticks = [_tick(2.0, bid=1.9, ask=2.1), _tick(3.0, bid=2.8, ask=3.2)]
        s = summarize_ticks(ticks, entry_premium=2.0)
        # spread1 = 0.2/2.0 = 10%, spread2 = 0.4/3.0 = 13.33% → max 13.33
        assert s["max_spread_pct"] > 13.0
        assert s["avg_spread_pct"] > 11.0

    def test_underlying_move(self):
        s = summarize_ticks(
            [_tick(2.0, under=100.0), _tick(3.0, under=102.0)], entry_premium=2.0
        )
        assert s["underlying_first"] == 100.0
        assert s["underlying_last"] == 102.0
        assert s["underlying_move_pct"] == 2.0

    def test_none_and_garbage_values_ignored(self):
        ticks = [_tick(None), _tick("bad"), _tick(0), _tick(4.0), _tick(2.0)]
        s = summarize_ticks(ticks, entry_premium=2.0)
        # only 4.0 and 2.0 count; n still counts all rows
        assert s["n"] == 5
        assert s["peak_premium"] == 4.0
        assert s["premium_min"] == 2.0

    def test_all_bad_premiums_returns_safe(self):
        s = summarize_ticks([_tick(None), _tick(0)], entry_premium=2.0)
        assert s["n"] == 2
        assert s["peak_premium"] is None  # nothing usable → safe, no crash

    def test_zero_entry_no_divide_by_zero(self):
        s = summarize_ticks([_tick(2.0)], entry_premium=0.0)
        assert s["peak_gain_pct"] is None  # entry<=0 → gain undefined, not a crash


class TestBuildTimeline:
    def _trade(self, **kw):
        base = {
            "ticker": "MU", "direction": "put", "contracts": 2,
            "premium_per_contract": 11.54, "opened_at": datetime(2026, 7, 2, 10, 0),
            "status": "open",
        }
        base.update(kw)
        return base

    def test_entry_is_first(self):
        tl = build_timeline(self._trade(), events=[])
        assert tl[0]["kind"] == "entry"
        assert "MU" in tl[0]["label"] and "PUT" in tl[0]["label"]

    def test_exit_is_last_and_toned_by_pnl(self):
        trade = self._trade(status="closed", exit_premium=5.5, pnl_dollars=-604.0,
                            exit_reason="multiday_put_hardstop",
                            closed_at=datetime(2026, 7, 2, 11, 0))
        tl = build_timeline(trade, events=[])
        assert tl[-1]["kind"] == "exit"
        assert tl[-1]["tone"] == "bad"
        assert "hardstop" in tl[-1]["label"]

    def test_winning_exit_is_good_tone(self):
        trade = self._trade(status="closed", exit_premium=20.0, pnl_dollars=1692.0,
                            closed_at=datetime(2026, 7, 2, 11, 0))
        assert build_timeline(trade, [])[-1]["tone"] == "good"

    def test_events_are_tone_coded_and_ordered(self):
        events = [
            {"event_type": "webull_filled", "created_at": datetime(2026, 7, 2, 10, 1), "details": {}},
            {"event_type": "webull_rejected", "created_at": datetime(2026, 7, 2, 10, 2), "details": {}},
            {"event_type": "dca_add", "created_at": datetime(2026, 7, 2, 10, 3), "details": {}},
        ]
        tl = build_timeline(self._trade(), events)
        by_kind = {i["kind"]: i["tone"] for i in tl}
        assert by_kind["webull_filled"] == "good"
        assert by_kind["webull_rejected"] == "bad"
        assert by_kind["dca_add"] == "warn"
        # chronological
        stamps = [i["sort"] for i in tl]
        assert stamps == sorted(stamps)

    def test_peak_milestone_added_when_profitable(self):
        stats = {"peak_gain_pct": 48.0, "peak_premium": 17.1,
                 "peak_at": datetime(2026, 7, 2, 10, 30).isoformat()}
        tl = build_timeline(self._trade(), events=[], tick_stats=stats)
        peaks = [i for i in tl if i["kind"] == "peak"]
        assert len(peaks) == 1
        assert "+48.0%" in peaks[0]["label"]

    def test_no_peak_milestone_when_never_green(self):
        stats = {"peak_gain_pct": -5.0, "peak_at": datetime(2026, 7, 2, 10, 30).isoformat()}
        tl = build_timeline(self._trade(), events=[], tick_stats=stats)
        assert not any(i["kind"] == "peak" for i in tl)


# ---------------------------------------------------------------------------
# Integration: the verbose trade_detail template renders with the helpers
# ---------------------------------------------------------------------------


def _render_detail(trade, ticks, events):
    """Render trade_detail.html in isolation through the real Jinja env + filters +
    helpers — catches template syntax/logic errors without needing a DB."""
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    from options_owl.dashboard import app as dash_app
    from options_owl.dashboard.analytics_util import build_timeline, summarize_ticks

    base = Path(dash_app.__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(base)), autoescape=True)
    env.filters["money"] = dash_app._fmt_money
    env.filters["pct"] = dash_app._fmt_pct
    env.filters["ftime"] = dash_app._fmt_time
    env.filters["fdate"] = dash_app._fmt_date
    env.filters["pnl_class"] = dash_app._pnl_class

    # match the route: JSON-safe ticks
    ticks = [
        {k: (v.isoformat() if hasattr(v, "isoformat")
             else float(v) if hasattr(v, "__float__") else v) for k, v in t.items()}
        for t in ticks
    ]
    ts = summarize_ticks(ticks, trade.get("premium_per_contract"), trade.get("contracts"))
    tl = build_timeline(trade, events, ts)
    return env.get_template("trade_detail.html").render(
        trade=trade, ticks=ticks, events=events, tick_stats=ts, timeline=tl,
        user={"sub": "kody", "agent_id": "owlet_kody"},
    )


class TestTradeDetailRender:
    def _closed_trade(self):
        return {
            "id": 399, "sqlite_id": 399, "ticker": "MU", "direction": "put",
            "strike": 985.0, "expiry_date": "2026-07-02", "contracts": 2,
            "premium_per_contract": 11.54, "exit_premium": 5.50, "total_cost": 2308.0,
            "pnl_dollars": -1208.0, "pnl_pct": -52.3, "status": "closed",
            "exit_reason": "multiday_put_hardstop", "exit_source": "ai", "hold_minutes": 63,
            "score": 88, "bot_source": "uw_flow", "webull_order_id": "WB399",
            "webull_entry_fill_price": 11.54, "opened_at": datetime(2026, 7, 2, 10, 0),
            "closed_at": datetime(2026, 7, 2, 11, 3),
        }

    def _ticks(self):
        from decimal import Decimal  # prove Decimal (asyncpg type) round-trips
        return [
            {"captured_at": datetime(2026, 7, 2, 10, 0), "premium": Decimal("11.5"),
             "bid": Decimal("11.3"), "ask": Decimal("11.7"), "underlying_price": Decimal("985")},
            {"captured_at": datetime(2026, 7, 2, 10, 30), "premium": Decimal("17.1"),
             "bid": Decimal("16.9"), "ask": Decimal("17.3"), "underlying_price": Decimal("979")},
            {"captured_at": datetime(2026, 7, 2, 11, 0), "premium": Decimal("5.5"),
             "bid": Decimal("5.3"), "ask": Decimal("5.7"), "underlying_price": Decimal("990")},
        ]

    def test_renders_full_closed_trade(self):
        html = _render_detail(self._closed_trade(), self._ticks(), events=[])
        assert "MU" in html
        assert "premium-chart" in html          # chart canvas present
        assert "Peak Gain" in html               # tick_stats surfaced
        assert "Timeline" in html
        assert "multiday_put_hardstop" in html   # exit reason in timeline + header
        assert "cdn.jsdelivr.net/npm/chart.js" in html

    def test_open_trade_without_ticks(self):
        trade = self._closed_trade()
        trade.update(status="open", exit_premium=None, pnl_dollars=None,
                     closed_at=None, exit_reason=None)
        html = _render_detail(trade, ticks=[], events=[])
        assert "No premium ticks captured" in html
        assert "LIVE" in html
        assert "premium-chart" not in html       # no chart without ticks

    def test_timeline_shows_events(self):
        events = [
            {"event_type": "webull_filled", "created_at": datetime(2026, 7, 2, 10, 1),
             "details": {"order_id": "WB399", "fill": 11.54}},
            {"event_type": "webull_rejected", "created_at": datetime(2026, 7, 2, 10, 2),
             "details": {"error": "step"}},
        ]
        html = _render_detail(self._closed_trade(), self._ticks(), events)
        assert "webull filled" in html           # underscores humanized
        assert "webull rejected" in html
        assert "WB399" in html                    # details rendered

    def test_bare_open_trade_full_schema_no_crash(self):
        """A freshly-opened trade as asyncpg dict(row) hands it over: every column present,
        None where unset (the real prod shape — NOT a partial dict). Must render cleanly."""
        trade = dict.fromkeys([
            "sqlite_id", "exit_premium", "pnl_dollars", "pnl_pct", "hold_minutes",
            "peak_gain_pct", "exit_reason", "exit_source", "webull_order_id",
            "webull_entry_fill_price", "webull_exit_order_id", "webull_exit_fill_price",
            "dca_count", "original_premium", "expiry_date", "bot_source", "score",
            "total_cost", "closed_at",
        ])
        trade.update({
            "id": 1, "ticker": "SPY", "direction": "call", "strike": 500.0,
            "contracts": 1, "premium_per_contract": 2.0, "status": "open",
            "opened_at": datetime(2026, 7, 2, 10, 0),
        })
        html = _render_detail(trade, ticks=[], events=[])  # must not raise
        assert "SPY" in html
        assert "LIVE" in html
