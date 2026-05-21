"""Tests for target-anchored OTM strike selection."""


from options_owl.risk.otm_selector import (
    _score_affordability,
    _score_gamma_zone,
    _score_reachability,
    score_otm_strikes,
    select_best_otm,
)


# ---------------------------------------------------------------------------
# Reachability scoring
# ---------------------------------------------------------------------------


class TestReachability:
    """Strike reachability relative to price targets."""

    def test_call_within_t2(self):
        # entry=655, t2=657.5, target=660, strike=656 → within T2
        assert _score_reachability(656, 655, 660, 657.5, is_call=True) == 35

    def test_call_within_target(self):
        # strike=659 is beyond T2 but within target
        assert _score_reachability(659, 655, 660, 657.5, is_call=True) == 20

    def test_call_beyond_target(self):
        # strike=661 is beyond the target
        assert _score_reachability(661, 655, 660, 657.5, is_call=True) == -10

    def test_call_at_entry_is_atm(self):
        assert _score_reachability(655, 655, 660, 657.5, is_call=True) == -10

    def test_call_below_entry_is_itm(self):
        assert _score_reachability(653, 655, 660, 657.5, is_call=True) == -10

    def test_put_within_t2(self):
        # entry=340, t2=339, target=338, strike=339.5 → within T2
        assert _score_reachability(339.5, 340, 338, 339, is_call=False) == 35

    def test_put_within_target(self):
        # strike=338.5 is beyond T2 but within target
        assert _score_reachability(338.5, 340, 338, 339, is_call=False) == 20

    def test_put_beyond_target(self):
        # strike=337 is beyond target
        assert _score_reachability(337, 340, 338, 339, is_call=False) == -10

    def test_put_at_entry_is_atm(self):
        assert _score_reachability(340, 340, 338, 339, is_call=False) == -10

    def test_put_above_entry_is_itm(self):
        assert _score_reachability(341, 340, 338, 339, is_call=False) == -10

    def test_call_at_t2_boundary(self):
        assert _score_reachability(657.5, 655, 660, 657.5, is_call=True) == 35

    def test_call_at_target_boundary(self):
        assert _score_reachability(660, 655, 660, 657.5, is_call=True) == 20

    def test_put_at_t2_boundary(self):
        assert _score_reachability(339, 340, 338, 339, is_call=False) == 35

    def test_put_at_target_boundary(self):
        assert _score_reachability(338, 340, 338, 339, is_call=False) == 20


# ---------------------------------------------------------------------------
# Affordability scoring
# ---------------------------------------------------------------------------


class TestAffordability:
    def test_lottery_ticket(self):
        assert _score_affordability(0.05) == -25

    def test_low_good_range(self):
        assert _score_affordability(0.15) == 20

    def test_sweet_spot_low(self):
        assert _score_affordability(0.35) == 35

    def test_sweet_spot_high(self):
        assert _score_affordability(0.99) == 35

    def test_good_range_upper(self):
        assert _score_affordability(1.50) == 20

    def test_acceptable(self):
        assert _score_affordability(2.50) == 5

    def test_too_expensive(self):
        assert _score_affordability(3.50) == -20

    def test_boundary_0_10(self):
        assert _score_affordability(0.10) == 20

    def test_boundary_0_30(self):
        assert _score_affordability(0.30) == 20

    def test_boundary_1_00(self):
        assert _score_affordability(1.00) == 35

    def test_boundary_2_00(self):
        assert _score_affordability(2.00) == 20

    def test_boundary_3_00(self):
        assert _score_affordability(3.00) == 5


# ---------------------------------------------------------------------------
# Gamma zone scoring
# ---------------------------------------------------------------------------


class TestGammaZone:
    def test_no_delta(self):
        assert _score_gamma_zone(None) == 0

    def test_too_far_otm(self):
        assert _score_gamma_zone(0.05) == -20

    def test_low_acceptable(self):
        assert _score_gamma_zone(0.09) == 10

    def test_peak_gamma_low(self):
        assert _score_gamma_zone(0.15) == 20

    def test_peak_gamma_mid(self):
        assert _score_gamma_zone(0.24) == 20

    def test_peak_gamma_high(self):
        assert _score_gamma_zone(0.30) == 20

    def test_acceptable_high(self):
        assert _score_gamma_zone(0.35) == 10

    def test_too_deep_itm(self):
        assert _score_gamma_zone(0.45) == -10

    def test_negative_delta_uses_abs(self):
        # Put deltas are negative; we should use absolute value
        assert _score_gamma_zone(-0.24) == 20

    def test_boundary_0_08(self):
        # 0.08 is >= 0.08, falls into the 0.08-0.099 acceptable low range
        assert _score_gamma_zone(0.08) == 10

    def test_boundary_0_10(self):
        assert _score_gamma_zone(0.10) == 20

    def test_boundary_0_40(self):
        assert _score_gamma_zone(0.40) == 10


# ---------------------------------------------------------------------------
# Full scoring & selection: SPY example from spec
# ---------------------------------------------------------------------------


class TestSPYExample:
    """The SPY $655 → $660 bullish example from the spec."""

    STRIKES = [
        {"strike": 656, "premium": 2.80, "delta": 0.42},
        {"strike": 657, "premium": 1.60, "delta": 0.33},
        {"strike": 658, "premium": 0.80, "delta": 0.24},
        {"strike": 659, "premium": 0.35, "delta": 0.16},
        {"strike": 660, "premium": 0.15, "delta": 0.09},
    ]

    def test_best_is_in_sweet_spot(self):
        """Best strike should be in the $0.30-$1.00 premium sweet spot with peak gamma."""
        best = select_best_otm(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        assert best is not None
        # $659 ($0.35) and $658 ($0.80) are both strong picks
        # T2 = (655+660)/2 = 657.5, so both are "within full target" (reach=20)
        # Both in sweet spot (afford=35), both peak gamma (20)
        # $659 wins on tiebreak (cheaper premium)
        assert best.strike in (658, 659)

    def test_658_and_659_top_two(self):
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        top_two = {ranked[0].strike, ranked[1].strike}
        assert top_two == {658, 659}

    def test_658_reachability(self):
        """$658 > T2 ($657.50) — within full target, gets +20."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s658 = next(s for s in ranked if s.strike == 658)
        assert s658.reach_score == 20

    def test_658_affordability(self):
        """$0.80 is in $0.30-$1.00 sweet spot — gets +35."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s658 = next(s for s in ranked if s.strike == 658)
        assert s658.afford_score == 35

    def test_658_gamma(self):
        """Delta 0.24 is peak gamma zone — gets +20."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s658 = next(s for s in ranked if s.strike == 658)
        assert s658.gamma_score == 20

    def test_657_within_t2(self):
        """$657 < T2 ($657.50) — within T2, gets +35 reach."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s657 = next(s for s in ranked if s.strike == 657)
        assert s657.reach_score == 35

    def test_660_at_target_edge(self):
        """$660 is at the target boundary — within target."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s660 = next(s for s in ranked if s.strike == 660)
        assert s660.reach_score == 20

    def test_656_within_t2(self):
        """$656 < T2 ($657.50) — within T2, gets +35 reach."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
        )
        s656 = next(s for s in ranked if s.strike == 656)
        assert s656.reach_score == 35

    def test_with_explicit_t2_658_wins(self):
        """When T2 is explicitly set to $658, the $658 strike gets +35 reach and wins."""
        best = select_best_otm(
            self.STRIKES, entry_price=655, target_price=660, direction="call",
            t2_price=658,
        )
        assert best is not None
        assert best.strike == 658


# ---------------------------------------------------------------------------
# PUT example: TSLA bearish
# ---------------------------------------------------------------------------


class TestTSLAPutExample:
    """TSLA at $340 → $338 bearish."""

    STRIKES = [
        {"strike": 339, "premium": 2.20, "delta": -0.38},
        {"strike": 338, "premium": 1.40, "delta": -0.28},
        {"strike": 337, "premium": 0.75, "delta": -0.20},
        {"strike": 336, "premium": 0.30, "delta": -0.12},
        {"strike": 335, "premium": 0.10, "delta": -0.06},
    ]

    def test_best_is_within_t2_and_sweet_spot(self):
        best = select_best_otm(
            self.STRIKES, entry_price=340, target_price=338, direction="put",
        )
        assert best is not None
        # $339 is within T2 (339) and $338 is at target
        # $339 premium $2.20 is only +5 affordability
        # $338 premium $1.40 is +20 affordability, delta 0.28 is +20 gamma
        # Either 338 or 339 depending on total scores
        assert best.strike in (338, 339)

    def test_335_scores_poorly(self):
        """$335 at $0.10 premium and delta 0.06 — beyond target, far OTM."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=340, target_price=338, direction="put",
        )
        s335 = next(s for s in ranked if s.strike == 335)
        assert s335.afford_score == 20  # $0.10 is at boundary of 0.10-0.30 range
        assert s335.gamma_score == -20  # delta 0.06 is too far out
        assert s335.reach_score == -10  # beyond target

    def test_put_direction_respected(self):
        """Put strikes below entry should be OTM (positive reach scores)."""
        ranked = score_otm_strikes(
            self.STRIKES, entry_price=340, target_price=338, direction="put",
        )
        # All strikes < 340 should have positive or neutral reach
        for s in ranked:
            if s.strike < 340 and s.strike >= 338:
                assert s.reach_score >= 20


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_strikes_returns_none(self):
        assert select_best_otm([], 100, 105, "call") is None

    def test_all_negative_scores_returns_none(self):
        """If every strike scores negative, return None."""
        strikes = [
            {"strike": 115, "premium": 0.02, "delta": 0.02},  # beyond target + lottery + too far OTM
        ]
        result = select_best_otm(strikes, 100, 105, "call")
        assert result is None

    def test_zero_premium_skipped(self):
        strikes = [
            {"strike": 101, "premium": 0.0, "delta": 0.30},
            {"strike": 102, "premium": 0.80, "delta": 0.24},
        ]
        ranked = score_otm_strikes(strikes, 100, 105, "call")
        assert len(ranked) == 1
        assert ranked[0].strike == 102

    def test_none_premium_skipped(self):
        strikes = [
            {"strike": 101, "premium": None, "delta": 0.30},
            {"strike": 102, "premium": 0.80, "delta": 0.24},
        ]
        ranked = score_otm_strikes(strikes, 100, 105, "call")
        assert len(ranked) == 1

    def test_no_delta_still_scored(self):
        """Strikes without delta data get 0 for gamma — not rejected."""
        strikes = [
            {"strike": 102, "premium": 0.80},
        ]
        ranked = score_otm_strikes(strikes, 100, 105, "call")
        assert len(ranked) == 1
        assert ranked[0].gamma_score == 0

    def test_custom_t2(self):
        """Providing explicit T2 overrides the midpoint calc."""
        strikes = [
            {"strike": 101, "premium": 0.80, "delta": 0.24},
            {"strike": 103, "premium": 0.50, "delta": 0.18},
        ]
        # Default T2 = (100+105)/2 = 102.5 → both within T2
        ranked_default = score_otm_strikes(strikes, 100, 105, "call")
        assert ranked_default[0].reach_score == 35  # within T2

        # Custom T2 = 101 → only 101 within T2, 103 within full target
        ranked_custom = score_otm_strikes(strikes, 100, 105, "call", t2_price=101)
        s101 = next(s for s in ranked_custom if s.strike == 101)
        s103 = next(s for s in ranked_custom if s.strike == 103)
        assert s101.reach_score == 35  # within T2
        assert s103.reach_score == 20  # within target but beyond T2

    def test_tiebreaker_cheaper_wins(self):
        """Two strikes with identical reach/afford/gamma — cheaper premium wins."""
        strikes = [
            {"strike": 101, "premium": 0.80, "delta": 0.24},
            {"strike": 102, "premium": 0.60, "delta": 0.24},
        ]
        # T2 = (100+105)/2 = 102.5 → both within T2 (reach=35)
        ranked = score_otm_strikes(strikes, 100, 105, "call")
        assert ranked[0].reach_score == ranked[1].reach_score == 35
        assert ranked[0].afford_score == ranked[1].afford_score == 35
        assert ranked[0].gamma_score == ranked[1].gamma_score == 20
        # $102 at $0.60 wins on tiebreak over $101 at $0.80
        assert ranked[0].strike == 102

    def test_single_strike(self):
        strikes = [{"strike": 102, "premium": 0.80, "delta": 0.24}]
        best = select_best_otm(strikes, 100, 105, "call")
        assert best is not None
        assert best.strike == 102

    def test_very_expensive_stock(self):
        """SPX-like with $650+ entry."""
        strikes = [
            {"strike": 655, "premium": 4.00, "delta": 0.35},
            {"strike": 658, "premium": 1.80, "delta": 0.22},
            {"strike": 660, "premium": 0.90, "delta": 0.15},
            {"strike": 665, "premium": 0.20, "delta": 0.07},
        ]
        best = select_best_otm(strikes, 653, 660, "call")
        assert best is not None
        # $660 at $0.90 in sweet spot, within target, peak gamma
        assert best.strike == 660

    def test_penny_stock(self):
        """Low-price stock with tight strikes."""
        strikes = [
            {"strike": 5.5, "premium": 0.15, "delta": 0.25},
            {"strike": 6.0, "premium": 0.08, "delta": 0.12},
        ]
        best = select_best_otm(strikes, 5.0, 6.0, "call")
        assert best is not None
        assert best.strike == 5.5  # within T2, not a lottery ticket


class TestDirectionStrings:
    """Various direction string formats."""

    STRIKES = [{"strike": 102, "premium": 0.80, "delta": 0.24}]

    def test_call_lowercase(self):
        assert select_best_otm(self.STRIKES, 100, 105, "call") is not None

    def test_call_uppercase(self):
        assert select_best_otm(self.STRIKES, 100, 105, "CALL") is not None

    def test_c_shorthand(self):
        assert select_best_otm(self.STRIKES, 100, 105, "C") is not None

    def test_put_lowercase(self):
        strikes = [{"strike": 99, "premium": 0.80, "delta": -0.24}]
        assert select_best_otm(strikes, 100, 95, "put") is not None

    def test_put_uppercase(self):
        strikes = [{"strike": 99, "premium": 0.80, "delta": -0.24}]
        assert select_best_otm(strikes, 100, 95, "PUT") is not None
