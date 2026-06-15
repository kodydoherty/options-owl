"""Train/serve feature-parity tests for the shared regime feature module.

These tests are the guard rail against re-introducing the "40 vs 18" train/serve
skew (spec §3, §8). They MUST fail if anyone makes the trainer and the live
serving path compute different regime features.

Covered:
  - The shared module yields the SAME feature names AND order for the trainer
    (load_training_inputs) and serving (load_serving_inputs) paths.
  - compute_regime_features() output covers EVERY feature in the deployed
    regime model meta (no silent omission / zero-fill).
  - GEX aggregation math + empty/None edge cases -> 0, never NaN.
  - Serve-time-safety: banned leak features are never produced.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from options_owl.sourcing.features import regime_features as rf
from options_owl.sourcing.features.regime_features import (
    REGIME_FEATURE_ORDER,
    compute_gex_features,
    compute_regime_feature_vector,
    load_serving_inputs,
    load_training_inputs,
    rth_bars_by_date_from_rows,
)

ET = ZoneInfo("America/New_York")
PROJECT_DIR = Path(__file__).resolve().parent.parent
REGIME_META = PROJECT_DIR / "journal" / "models" / "ml_v3" / "regime_classifier_meta.json"


# ---------------------------------------------------------------------------
# Synthetic data shared by trainer + serving fixtures
# ---------------------------------------------------------------------------

TICKER = "NVDA"
TODAY = "2026-06-09"
PRIOR_DATES = ["2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06"]


def _bar(tm, base):
    return {
        "tm": tm,
        "open": base,
        "high": base + 0.5,
        "low": base - 0.4,
        "close": base + 0.2,
        "volume": 1000 + int(base),
    }


def _flat_rows():
    """Flat (d, tm, ohlcv) rows for the OWN ticker: 5 prior RTH days + today AM."""
    rows = []
    # Prior full RTH days (just enough bars to compute daily aggregates + lags)
    for i, d in enumerate(PRIOR_DATES):
        for j, tm in enumerate(["09:30", "10:00", "12:00", "15:00", "16:00"]):
            b = _bar(tm, 100 + i + j * 0.3)
            rows.append({"d": d, **b})
        # premarket bar that MUST be dropped (serve-time safety)
        rows.append({"d": d, **_bar("08:00", 999.0)})
    # Today: early-morning window 09:30-09:44 (>= 5 bars) + later bars
    for k, tm in enumerate(["09:30", "09:35", "09:40", "09:42", "09:44"]):
        rows.append({"d": TODAY, **_bar(tm, 105 + k * 0.2)})
    # Same-day full-day bars (must NOT influence morning features)
    rows.append({"d": TODAY, **_bar("11:00", 130.0)})
    rows.append({"d": TODAY, **_bar("15:30", 90.0)})
    return rows


def _gex_legs():
    return {
        "call_gamma": 1200.0,
        "put_gamma": 800.0,
        "call_delta": 50.0,
        "put_delta": -30.0,
        "call_charm": 5.0,
        "put_charm": 2.0,
        "call_vanna": 3.0,
        "put_vanna": 1.0,
    }


# ---------------------------------------------------------------------------
# 1. Trainer-path raw_inputs -> full ordered vector
# ---------------------------------------------------------------------------


def _training_vector():
    own_by_date = rth_bars_by_date_from_rows(_flat_rows())
    # SPY/QQQ reuse the own grouped bars for determinism in this test
    market_by_date = {"SPY": own_by_date, "QQQ": own_by_date}
    raw = load_training_inputs(
        TICKER, TODAY,
        by_date=own_by_date,
        market_by_date=market_by_date,
        gex_row=_gex_legs(),
    )
    return compute_regime_feature_vector(raw)


def test_training_vector_matches_canonical_order():
    vec = _training_vector()
    assert list(vec.keys()) == REGIME_FEATURE_ORDER
    assert len(vec) == 40
    assert all(isinstance(v, float) and not math.isnan(v) for v in vec.values())


# ---------------------------------------------------------------------------
# 2. Serving-path raw_inputs (mocked Postgres) -> full ordered vector
# ---------------------------------------------------------------------------


def _pg_rows_for(sym):
    """Postgres stock_candles rows (with tz-aware bar_time) for one symbol."""
    rows = []
    for r in _flat_rows():
        # build a tz-aware ET timestamp from d + tm
        dt = datetime.strptime(f"{r['d']} {r['tm']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        rows.append(
            {
                "bar_time": dt,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            }
        )
    return rows


async def _serving_vector(gex_row=None):
    """Run load_serving_inputs with Postgres fully mocked, then build the vector."""

    async def fake_fetch(query, *args):
        # All ticker stock_candles reads return the same synthetic rows.
        return _pg_rows_for(args[0])

    async def fake_fetchrow(query, *args):
        if gex_row is None:
            return None
        return gex_row

    with patch("options_owl.db.postgres.fetch", new=AsyncMock(side_effect=fake_fetch)), \
         patch("options_owl.db.postgres.fetchrow", new=AsyncMock(side_effect=fake_fetchrow)):
        now_et = datetime(2026, 6, 9, 9, 45, tzinfo=ET)
        raw = await load_serving_inputs(TICKER, now_et, tz_et=ET)
    return compute_regime_feature_vector(raw)


async def test_serving_vector_matches_canonical_order():
    vec = await _serving_vector()
    assert list(vec.keys()) == REGIME_FEATURE_ORDER
    assert len(vec) == 40
    assert all(not math.isnan(v) for v in vec.values())


# ---------------------------------------------------------------------------
# 3. PARITY: trainer and serving produce identical names/order (the guard)
# ---------------------------------------------------------------------------


async def test_train_serve_feature_parity():
    train_vec = _training_vector()
    serve_vec = await _serving_vector()

    # Identical feature names AND order — this is the anti-skew assertion.
    assert list(train_vec.keys()) == list(serve_vec.keys())
    assert list(train_vec.keys()) == REGIME_FEATURE_ORDER

    # The non-GEX features are computed from the same synthetic bars on both
    # paths, so their VALUES must also agree (the math is defined once).
    gex_keys = set(rf._GEX_COLS)
    for k in REGIME_FEATURE_ORDER:
        if k in gex_keys:
            continue  # GEX legs are sourced differently (UW sqlite vs gex_ticks)
        assert train_vec[k] == pytest.approx(serve_vec[k], rel=1e-9, abs=1e-9), (
            f"feature {k} drifted: train={train_vec[k]} serve={serve_vec[k]}"
        )


# ---------------------------------------------------------------------------
# 4. compute_regime_features() covers every model-meta feature (no zero-fill)
# ---------------------------------------------------------------------------


async def test_compute_regime_features_covers_model_meta():
    assert REGIME_META.exists(), f"regime meta not found at {REGIME_META}"
    meta = json.loads(REGIME_META.read_text())
    model_features = meta["features"]

    # The deployed model's feature list must equal our canonical order.
    assert model_features == REGIME_FEATURE_ORDER, (
        "regime model meta features != REGIME_FEATURE_ORDER — retrain or sync the "
        "shared module (train/serve skew)"
    )

    from options_owl.sourcing import ml_pipeline

    async def fake_fetch(query, *args):
        return _pg_rows_for(args[0])

    with patch("options_owl.db.postgres.fetch", new=AsyncMock(side_effect=fake_fetch)), \
         patch("options_owl.db.postgres.fetchrow", new=AsyncMock(return_value=None)):
        now_et = datetime(2026, 6, 9, 9, 45, tzinfo=ET)
        feat = await ml_pipeline.compute_regime_features(TICKER, now_et, model_features)

    assert feat is not None
    # Every meta feature is present (no silent omission).
    for name in model_features:
        assert name in feat, f"serving output missing model feature {name}"


async def test_missing_feature_warner_stays_quiet():
    """The once-per-model missing-feature warner must not fire for regime."""
    from options_owl.sourcing import ml_pipeline
    from options_owl.sourcing.scoring.ml_gates import signal_model

    # reset the warned set so this test is independent of ordering
    signal_model._missing_feature_warned.discard("regime_classifier")

    async def fake_fetch(query, *args):
        return _pg_rows_for(args[0])

    with patch("options_owl.db.postgres.fetch", new=AsyncMock(side_effect=fake_fetch)), \
         patch("options_owl.db.postgres.fetchrow", new=AsyncMock(return_value=None)), \
         patch.object(signal_model.logger, "warning") as mock_warn:
        now_et = datetime(2026, 6, 9, 9, 45, tzinfo=ET)
        await ml_pipeline.compute_regime_features(TICKER, now_et, REGIME_FEATURE_ORDER)

    missing_warnings = [
        c for c in mock_warn.call_args_list
        if c.args and "missing" in str(c.args[0]) and "regime" in str(c.args[0])
    ]
    assert not missing_warnings, f"unexpected missing-feature warning: {missing_warnings}"


# ---------------------------------------------------------------------------
# 5. GEX math + edge cases (empty / None -> 0, never NaN); serve-time safety
# ---------------------------------------------------------------------------


def test_gex_net_legs_derived_once():
    g = compute_gex_features(_gex_legs())
    assert g["net_gamma"] == pytest.approx(1200.0 - 800.0)
    assert g["net_delta"] == pytest.approx(50.0 - (-30.0))
    assert g["net_charm"] == pytest.approx(5.0 - 2.0)
    assert g["net_vanna"] == pytest.approx(3.0 - 1.0)


def test_gex_empty_returns_zeros_not_nan():
    for empty in (None, {}, {"call_gamma": None}):
        g = compute_gex_features(empty)
        assert set(g.keys()) == set(rf._GEX_COLS)
        assert all(v == 0.0 for v in g.values())
        assert all(not math.isnan(v) for v in g.values())


async def test_serving_empty_gex_table_one_warning_and_zeros():
    """gex_ticks empty/absent -> GEX features 0, exactly one warning, model loads."""
    rf._gex_empty_warned.clear()

    async def fake_fetch(query, *args):
        return _pg_rows_for(args[0])

    # fetchrow raises (table absent during rollout) on the GEX query
    async def fake_fetchrow(query, *args):
        raise RuntimeError('relation "gex_ticks" does not exist')

    with patch("options_owl.db.postgres.fetch", new=AsyncMock(side_effect=fake_fetch)), \
         patch("options_owl.db.postgres.fetchrow", new=AsyncMock(side_effect=fake_fetchrow)), \
         patch.object(rf, "logger", create=True):
        now_et = datetime(2026, 6, 9, 9, 45, tzinfo=ET)
        # patch the loguru logger inside the function's import scope
        with patch("loguru.logger.warning") as mock_warn:
            raw1 = await load_serving_inputs("AMD", now_et, tz_et=ET)
            await load_serving_inputs("AMD", now_et, tz_et=ET)  # second call

    vec = compute_regime_feature_vector(raw1)
    for col in rf._GEX_COLS:
        assert vec[col] == 0.0

    amd_warnings = [
        c for c in mock_warn.call_args_list
        if c.args and "gex_ticks" in str(c.args[0]) and "AMD" in str(c.args[0])
    ]
    assert len(amd_warnings) == 1, (
        f"expected exactly one gex_ticks warning for AMD, got {len(amd_warnings)}"
    )


def test_no_leak_features_ever_produced():
    vec = _training_vector()
    banned = {"day_range_pct", "day_volume", "rth_range_pct", "rth_close_pos"}
    assert not (banned & set(vec)), "banned leak features must never appear"


def test_premarket_bars_excluded_from_morning():
    """RTH-only filter: the 08:00 premarket bar (price 999) must not leak in."""
    by_date = rth_bars_by_date_from_rows(_flat_rows())
    today = by_date[TODAY]
    assert all(b["tm"] >= "09:30" for b in today)
    # morning window stays in the 105-ish range, NOT polluted by 999 / 130 / 90
    raw = load_training_inputs(
        TICKER, TODAY, by_date=by_date,
        market_by_date={"SPY": by_date, "QQQ": by_date}, gex_row={},
    )
    vec = compute_regime_feature_vector(raw)
    # morning_range_pct from a tight 105-ish window is small (< 5%); a 999/90
    # leak would blow it up massively.
    assert vec["morning_range_pct"] < 5.0
