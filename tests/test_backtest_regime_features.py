"""Regression tests: backtest regime scoring uses the SHARED feature module.

Guards against the "40 vs 18" zero-fill skew — the gold-standard backtest's
compute_regime_score must produce EVERY feature the regime model expects via
options_owl.sourcing.features.regime_features (no silent .get(feat, 0)).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import scripts.backtest_gold_standard as gs
from options_owl.sourcing.features.regime_features import REGIME_FEATURE_ORDER

DATE = "2026-06-09"
PRIOR_DATES = [
    "2026-06-01", "2026-06-02", "2026-06-03",
    "2026-06-04", "2026-06-05", "2026-06-08",
]


class StubRegimeModel:
    """LightGBM Booster stand-in that records the X it was given."""

    def __init__(self, feature_names):
        self._features = list(feature_names)
        self.last_X = None

    def feature_name(self):
        return list(self._features)

    def predict(self, X):
        self.last_X = X
        return np.array([0.42])


def _make_theta_conn(tickers=("NVDA", "SPY", "QQQ"), morning_bars=15):
    """In-memory stock_ohlc with synthetic RTH 1-min bars (today + prior days)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE stock_ohlc (ticker TEXT, timestamp TEXT, open REAL, "
        "high REAL, low REAL, close REAL, volume REAL)"
    )
    for ticker in tickers:
        base = 100.0 if ticker != "SPY" else 600.0
        for d_i, d in enumerate([*PRIOR_DATES, DATE]):
            n = morning_bars if d == DATE else 20
            for i in range(n):
                px = base + d_i * 0.5 + i * 0.1
                conn.execute(
                    "INSERT INTO stock_ohlc VALUES (?,?,?,?,?,?,?)",
                    (ticker, f"{d} 09:{30 + i:02d}:00-04:00",
                     px, px + 0.2, px - 0.2, px + 0.1, 1000.0),
                )
            # A late-session bar so prior-day close/range aggregates are sane
            if d != DATE:
                px = base + d_i * 0.5 + 3
                conn.execute(
                    "INSERT INTO stock_ohlc VALUES (?,?,?,?,?,?,?)",
                    (ticker, f"{d} 15:59:00-04:00",
                     px, px + 0.3, px - 0.3, px + 0.2, 2000.0),
                )
    conn.commit()
    return conn


def _make_uw_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE greek_exposure (ticker TEXT, date TEXT, "
        "call_gamma REAL, put_gamma REAL, call_delta REAL, put_delta REAL, "
        "call_charm REAL, put_charm REAL, call_vanna REAL, put_vanna REAL)"
    )
    conn.execute(
        "INSERT INTO greek_exposure VALUES "
        "('NVDA', '2026-06-08', 5e6, -2e6, 4e8, -3e8, 1e5, -5e4, 2e7, -1e7)"
    )
    conn.commit()
    return conn


class TestRegimeScoreSharedModule:
    def test_all_40_features_populated_no_zero_fill(self):
        """Every REGIME_FEATURE_ORDER feature reaches the model — shape (1, 40)."""
        model = StubRegimeModel(REGIME_FEATURE_ORDER)
        conn = _make_theta_conn()
        uw = _make_uw_conn()
        score = gs.compute_regime_score(model, "NVDA", DATE, conn, uw, {}, {})

        assert score == pytest.approx(0.42)
        assert model.last_X is not None
        assert model.last_X.shape == (1, len(REGIME_FEATURE_ORDER))
        # Non-trivial coverage: morning, lag, GEX and market features all live
        vec = dict(zip(REGIME_FEATURE_ORDER, model.last_X[0].tolist()))
        for feat in ("morning_range_pct", "prev_day_ret", "net_gamma",
                     "spy_morning_direction", "qqq_vol_5d"):
            assert vec[feat] != 0.0, f"{feat} unexpectedly zero — zero-fill regression?"

    def test_missing_model_feature_raises(self):
        """A model feature the shared module can't produce must fail LOUDLY."""
        model = StubRegimeModel([*REGIME_FEATURE_ORDER, "bogus_feature"])
        conn = _make_theta_conn()
        uw = _make_uw_conn()
        with pytest.raises(AssertionError, match="REGIME FEATURE SKEW"):
            gs.compute_regime_score(model, "NVDA", DATE, conn, uw, {}, {})

    def test_insufficient_morning_bars_returns_zero(self):
        """Early-return guard unchanged: < 5 morning bars -> 0.0, no predict."""
        model = StubRegimeModel(REGIME_FEATURE_ORDER)
        conn = _make_theta_conn(morning_bars=3)
        uw = _make_uw_conn()
        assert gs.compute_regime_score(model, "NVDA", DATE, conn, uw, {}, {}) == 0.0
        assert model.last_X is None

    def test_no_uw_conn_degrades_to_zero_gex(self):
        """uw_conn=None -> GEX legs zero (deterministic), score still computed."""
        model = StubRegimeModel(REGIME_FEATURE_ORDER)
        conn = _make_theta_conn()
        score = gs.compute_regime_score(model, "NVDA", DATE, conn, None, {}, {})
        assert score == pytest.approx(0.42)
        vec = dict(zip(REGIME_FEATURE_ORDER, model.last_X[0].tolist()))
        assert vec["net_gamma"] == 0.0

    def test_deployed_meta_matches_shared_feature_order(self):
        """The retrained regime model's meta must equal REGIME_FEATURE_ORDER."""
        meta_path = (
            Path(gs.PROJECT_DIR) / "journal" / "models" / "ml_v3"
            / "regime_classifier_meta.json"
        )
        if not meta_path.exists():
            pytest.skip("regime_classifier_meta.json not present in this checkout")
        meta = json.loads(meta_path.read_text())
        assert meta["features"] == list(REGIME_FEATURE_ORDER)
