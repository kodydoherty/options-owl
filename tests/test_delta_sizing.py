"""Tests for the delta-sized budget haircut (scale-invariant cheap-CALL tail brake).

High-gamma OTM lottery calls (low |delta|) get a proportional budget haircut = min(1, |delta|/ref);
ATM (|delta| >= ref) passes through at 1.0. CALLS only; fail-open when delta is missing.
"""

import pytest

from options_owl.risk.vinny_strategy import delta_size_haircut


class TestDeltaHaircut:
    def test_atm_passes_through(self):
        # |delta| >= ref → no haircut
        assert delta_size_haircut(0.50, ref=0.45) == 1.0
        assert delta_size_haircut(0.45, ref=0.45) == 1.0
        assert delta_size_haircut(0.90, ref=0.45) == 1.0  # clamped to 1.0

    def test_otm_lottery_gets_cut(self):
        # SMCI failure mode: delta ~0.18 → 0.18/0.45 = 0.40
        assert delta_size_haircut(0.18, ref=0.45) == pytest.approx(0.40)
        # delta 0.225 → exactly half
        assert delta_size_haircut(0.225, ref=0.45) == pytest.approx(0.5)

    def test_monotonic_lower_delta_smaller(self):
        vals = [delta_size_haircut(d) for d in (0.05, 0.18, 0.30, 0.45, 0.60)]
        assert vals == sorted(vals)
        assert vals[-1] == 1.0  # ATM full

    def test_uses_absolute_delta(self):
        # negative delta (sign-agnostic) → same as positive
        assert delta_size_haircut(-0.18, ref=0.45) == pytest.approx(0.40)

    def test_fail_open_on_missing_delta(self):
        assert delta_size_haircut(None) == 1.0

    def test_guard_on_bad_ref(self):
        assert delta_size_haircut(0.18, ref=0.0) == 1.0

    def test_custom_ref(self):
        assert delta_size_haircut(0.30, ref=0.60) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(delta_size_haircut(0.2), float)
