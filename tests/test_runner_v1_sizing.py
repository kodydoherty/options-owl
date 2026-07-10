"""Tests for runner_v1 P(runner) CALL sizing (Stage D serve path).

Covers the quartile sizing curve (runner_v1_size_mult) and the serve-time scorer fidelity
(_score_runner_v1 consumes the meta's 18-feature schema and returns a valid probability).
"""

from pathlib import Path

import pytest

from options_owl.risk.vinny_strategy import runner_v1_size_mult

MODEL = Path(__file__).resolve().parents[1] / "journal" / "models" / "ml_v3" / "runner_v1.lgb"


class TestRunnerSizeMultQuartiles:
    def test_q1_shrinks(self):
        mult, desc = runner_v1_size_mult(0.40)
        assert mult == 0.5
        assert "Q1" in desc

    def test_q2(self):
        assert runner_v1_size_mult(0.60)[0] == 0.85

    def test_q3(self):
        assert runner_v1_size_mult(0.65)[0] == 1.15

    def test_q4_sizes_up(self):
        mult, desc = runner_v1_size_mult(0.80)
        assert mult == 1.5
        assert "Q4" in desc

    def test_boundaries_are_lower_inclusive(self):
        # cut at q1=0.580: 0.579→Q1, 0.580→Q2
        assert runner_v1_size_mult(0.579)[0] == 0.5
        assert runner_v1_size_mult(0.580)[0] == 0.85
        # cut at q3=0.670: 0.669→Q3, 0.670→Q4
        assert runner_v1_size_mult(0.669)[0] == 1.15
        assert runner_v1_size_mult(0.670)[0] == 1.5

    def test_monotonic_nondecreasing(self):
        vals = [runner_v1_size_mult(p)[0] for p in (0.1, 0.59, 0.64, 0.68, 0.99)]
        assert vals == sorted(vals)

    def test_custom_cuts_respected(self):
        assert runner_v1_size_mult(0.50, q1=0.55, m_q1=0.25)[0] == 0.25

    def test_returns_float_and_desc(self):
        mult, desc = runner_v1_size_mult(0.72)
        assert isinstance(mult, float)
        assert "P(runner)=0.720" in desc


@pytest.mark.skipif(not MODEL.exists(), reason="runner_v1.lgb not present")
class TestScorerFidelity:
    def _feat(self):
        return {
            "entry_premium": 2.50, "log_premium": 0.916, "delta": 0.52, "iv": 0.45,
            "vega": 0.08, "theta": -0.30, "moneyness": 1.001, "spread_pct": 3.0,
            "und_move_pct": 0.4, "und_slope_5": 0.1, "und_rvol_15": 0.2, "opt_vol_5": 500.0,
            "gap_pct": 0.3, "prior_range_pct": 2.1, "dte": 0, "entry_min": 35,
            "ticker": "TSLA", "day_of_week": 2,
        }

    def test_score_in_range_and_deterministic(self):
        from options_owl.risk.flow_runner import _score_runner_v1
        p1 = _score_runner_v1(self._feat())
        p2 = _score_runner_v1(self._feat())
        assert 0.0 <= p1 <= 1.0
        assert p1 == p2  # deterministic

    def test_consumes_full_meta_schema(self):
        # All 18 meta features must be present in the vector we build; a missing one would
        # silently degrade the score. Guards against schema drift.
        from options_owl.risk.flow_runner import _load_runner_v1
        _, meta = _load_runner_v1()
        assert set(meta["features"]) <= set(self._feat().keys())
        assert set(meta["cat_features"]) == {"ticker", "day_of_week"}


# ---------------------------------------------------------------------------
# UW REST header helper (2026-07-10) — UW now requires UW-CLIENT-API-ID + UA,
# without which every REST call 401s and the feature silently dies (the enabled
# market-tide gate went dead in prod until this fix).
# ---------------------------------------------------------------------------
class TestUwRestHeaders:
    def test_required_headers_present(self):
        from options_owl.risk.flow_runner import _uw_rest_headers
        h = _uw_rest_headers("some-token")
        assert h["Authorization"] == "Bearer some-token"
        # The two headers UW added that we were missing:
        assert h["UW-CLIENT-API-ID"] == "100001"
        assert "Mozilla" in h.get("User-Agent", "")  # browser-like UA to pass Cloudflare

    def test_live_callers_use_the_helper(self):
        import inspect

        from options_owl.risk import flow_runner
        from options_owl.sourcing.data import capitol_trades
        # Both live UW REST callers must route through the helper, not a bare Bearer dict.
        assert "_uw_rest_headers(api_key)" in inspect.getsource(flow_runner.get_market_tide_bias)
        assert "_uw_rest_headers(api_key)" in inspect.getsource(capitol_trades._fetch_from_unusual_whales)
