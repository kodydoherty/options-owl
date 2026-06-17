"""Tests for the regime-aware call/put budget multiplier (validated 2.5yr,
scripts/backtest_2yr_regime.py). When SPY drifts down intraday, cut calls and favor puts;
when SPY is up, the reverse. Folds into the conviction multiplier in score_to_contracts.
"""

from options_owl.risk.vinny_strategy import regime_budget_mult


class TestRegimeBudgetDown:
    """SPY down → cut calls hard, favor puts."""

    def test_call_down_cut_hard(self):
        mult, desc = regime_budget_mult(is_put=False, spy_pct_move=-0.5)
        assert mult == 0.25
        assert "DOWN" in desc and "call" in desc

    def test_put_down_full_size(self):
        mult, _ = regime_budget_mult(is_put=True, spy_pct_move=-0.5)
        assert mult == 2.0

    def test_threshold_boundary_just_below(self):
        # -0.11 < -0.1 → down regime
        assert regime_budget_mult(is_put=False, spy_pct_move=-0.11)[0] == 0.25


class TestRegimeBudgetUp:
    """SPY up → full calls, cut puts."""

    def test_call_up_full(self):
        assert regime_budget_mult(is_put=False, spy_pct_move=0.5)[0] == 1.0

    def test_put_up_cut(self):
        assert regime_budget_mult(is_put=True, spy_pct_move=0.5)[0] == 0.5


class TestRegimeBudgetFlat:
    """SPY flat (between thresholds) → mild de-risk on calls, mild lift on puts."""

    def test_call_flat(self):
        assert regime_budget_mult(is_put=False, spy_pct_move=0.0)[0] == 0.6

    def test_put_flat(self):
        assert regime_budget_mult(is_put=True, spy_pct_move=0.05)[0] == 1.2

    def test_exact_down_threshold_is_flat(self):
        # -0.1 is NOT < -0.1 → flat, not down
        assert regime_budget_mult(is_put=False, spy_pct_move=-0.1)[0] == 0.6

    def test_exact_up_threshold_is_flat(self):
        # 0.1 is NOT > 0.1 → flat, not up
        assert regime_budget_mult(is_put=True, spy_pct_move=0.1)[0] == 1.2


class TestRegimeBudgetConfigurable:
    """Knobs are overridable (env-configurable per project convention)."""

    def test_custom_multipliers_respected(self):
        mult, _ = regime_budget_mult(
            is_put=False, spy_pct_move=-1.0, call_down=0.1, down_thresh=-0.2
        )
        assert mult == 0.1

    def test_custom_threshold_shifts_regime(self):
        # With a wider down threshold, -0.15 is now flat, not down
        assert regime_budget_mult(is_put=False, spy_pct_move=-0.15, down_thresh=-0.2)[0] == 0.6


class TestRegimeBudgetReturnType:
    def test_returns_float_and_str(self):
        mult, desc = regime_budget_mult(is_put=False, spy_pct_move=0.3)
        assert isinstance(mult, float)
        assert isinstance(desc, str)
        assert "SPY" in desc
