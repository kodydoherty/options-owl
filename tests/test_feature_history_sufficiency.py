"""compute_pattern_features must REFUSE short history, not fabricate it.

THE BUG (2026-08-14)
--------------------
The guard was `if idx < 5: return None`, but the function computes 10- and 20-bar
features using `max(0, idx - N)` slices. A short history therefore TRUNCATED silently
instead of failing: at idx=5 the 20-bar volume window held the same 5 bars as the 5-bar
window, so `volume_ratio = volume_avg_5 / avg20` was ~1.0 BY CONSTRUCTION and measured
nothing. Other short windows fell back to 0.

The model is trained on rows with full windows, so serving it truncated rows is
out-of-distribution scoring dressed up as a real prediction.

Live evidence: ml_sourcing's first 30 minutes lost -$6,633 of a -$7,157 total at a
24-35% win rate, negative in 3/3 months, and the win rate climbed as the windows filled
(31% -> 39% -> 67%). A skipped trade costs nothing; a fabricated score costs money.

Both the production and harness copies are covered -- they are documented as EXACT
copies, and a fix applied to only one would silently reintroduce backtest/live drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from options_owl.sourcing.ml_pipeline import MIN_HISTORY_BARS, compute_pattern_features


def _series(n: int = 80):
    return dict(
        closes=np.linspace(1.0, 1.5, n), volumes=np.linspace(100, 300, n),
        ivs=np.full(n, 0.4), deltas=np.full(n, 0.4), thetas=np.full(n, -0.05),
        underlyings=np.linspace(500, 506, n),
        bids=np.linspace(1.0, 1.5, n) - 0.02, asks=np.linspace(1.0, 1.5, n) + 0.02,
    )


def _call(fn, idx, n=80):
    s = _series(n)
    return fn(s["closes"], s["volumes"], s["ivs"], s["deltas"], s["thetas"],
              s["underlyings"], s["bids"], s["asks"], idx, 500.0)


class TestRefusesInsufficientHistory:
    @pytest.mark.parametrize("idx", [0, 1, 4, 5, 10, 15, MIN_HISTORY_BARS - 1])
    def test_short_history_returns_none(self, idx):
        assert _call(compute_pattern_features, idx) is None, (
            f"idx={idx} was scored with fewer than {MIN_HISTORY_BARS} bars — the long "
            "windows would be truncated and the features partly fabricated"
        )

    @pytest.mark.parametrize("idx", [MIN_HISTORY_BARS, MIN_HISTORY_BARS + 5, 40, 60])
    def test_sufficient_history_is_scored(self, idx):
        f = _call(compute_pattern_features, idx)
        assert f is not None, f"idx={idx} has full windows and must be scored"
        assert len(f) == 17, f"expected 17 features, got {len(f)}"

    def test_threshold_matches_longest_window(self):
        """MIN_HISTORY_BARS must cover the longest window the function actually uses."""
        import inspect
        import re
        src = inspect.getsource(compute_pattern_features)
        windows = {int(m) for m in re.findall(r"idx\s*-\s*(\d+)", src)}
        assert windows, "no trailing windows found — did the implementation change?"
        assert MIN_HISTORY_BARS >= max(windows), (
            f"MIN_HISTORY_BARS={MIN_HISTORY_BARS} is smaller than the longest window "
            f"({max(windows)}), so that feature is still computed from truncated data"
        )

    def test_volume_ratio_is_not_degenerate(self):
        """The tell of the original bug: both windows drawing on the same bars."""
        f = _call(compute_pattern_features, MIN_HISTORY_BARS + 10)
        assert f is not None
        assert f["volume_ratio"] != pytest.approx(1.0, abs=1e-9), (
            "volume_ratio is exactly 1.0 — the 5-bar and 20-bar windows are drawing on "
            "the same data, which is what made this feature meaningless early in a session"
        )


class TestHarnessCopyMatches:
    """The harness copy must carry the same guard, or backtests diverge from live."""

    def _harness_module(self):
        p = Path("scripts/backtest_gold_standard.py")
        if not p.exists():
            pytest.skip("harness not present")
        spec = importlib.util.spec_from_file_location("_bgs", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_bgs"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 - heavy optional deps
            pytest.skip(f"harness not importable: {exc}")
        return mod

    def test_harness_has_the_same_threshold(self):
        mod = self._harness_module()
        assert getattr(mod, "MIN_HISTORY_BARS", None) == MIN_HISTORY_BARS, (
            "harness MIN_HISTORY_BARS differs from production — the backtest would score "
            "trades live would refuse, reintroducing the very drift these copies exist to avoid"
        )

    def test_harness_refuses_short_history_too(self):
        mod = self._harness_module()
        assert _call(mod.compute_pattern_features, 5) is None
        assert _call(mod.compute_pattern_features, MIN_HISTORY_BARS) is not None
