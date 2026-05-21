"""Tests for Black-Scholes Greeks calculations."""

from __future__ import annotations

import math

from options_owl.risk.greeks import (
    _bs_price,
    calc_delta,
    calc_gamma,
    calc_iv_from_premium,
    calc_theta,
    calc_vega,
    norm_cdf,
    norm_pdf,
)


# ---------------------------------------------------------------------------
# norm_cdf / norm_pdf sanity checks
# ---------------------------------------------------------------------------


class TestNormFunctions:
    def test_norm_cdf_at_zero(self):
        assert abs(norm_cdf(0.0) - 0.5) < 1e-10

    def test_norm_cdf_large_positive(self):
        assert norm_cdf(6.0) > 0.999

    def test_norm_cdf_large_negative(self):
        assert norm_cdf(-6.0) < 0.001

    def test_norm_cdf_symmetry(self):
        x = 1.5
        assert abs(norm_cdf(x) + norm_cdf(-x) - 1.0) < 1e-10

    def test_norm_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(norm_pdf(0.0) - expected) < 1e-10

    def test_norm_pdf_positive(self):
        assert norm_pdf(1.0) > 0
        assert norm_pdf(1.0) < norm_pdf(0.0)

    def test_norm_pdf_symmetry(self):
        assert abs(norm_pdf(1.5) - norm_pdf(-1.5)) < 1e-10


# ---------------------------------------------------------------------------
# Known Black-Scholes values
# ---------------------------------------------------------------------------

# Standard test case: S=100, K=100, T=1, r=0.05, sigma=0.20
# Published BS values (approx):
#   call price ~10.4506, put price ~5.5735
#   call delta ~0.6368,  put delta ~-0.3632
#   gamma ~0.01876
#   call theta per day ~-0.01445 (approx), put theta per day ~-0.00817 (approx)
#   vega per 1% ~0.03752


class TestKnownBSValues:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    def test_bs_call_price(self):
        price = _bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert abs(price - 10.4506) < 0.01

    def test_bs_put_price(self):
        price = _bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(price - 5.5735) < 0.01

    def test_call_delta(self):
        delta = calc_delta(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert abs(delta - 0.6368) < 0.005

    def test_put_delta(self):
        delta = calc_delta(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(delta - (-0.3632)) < 0.005

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = 1 (for European options)."""
        call_d = calc_delta(self.S, self.K, self.T, self.r, self.sigma, "call")
        put_d = calc_delta(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs((call_d - put_d) - 1.0) < 1e-6

    def test_gamma(self):
        gamma = calc_gamma(self.S, self.K, self.T, self.r, self.sigma)
        assert abs(gamma - 0.01876) < 0.001

    def test_vega(self):
        vega = calc_vega(self.S, self.K, self.T, self.r, self.sigma)
        # vega per 1% change: S * N'(d1) * sqrt(T) / 100
        assert abs(vega - 0.3752) < 0.01

    def test_theta_call_is_negative(self):
        theta = calc_theta(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert theta < 0

    def test_theta_put_is_negative(self):
        theta = calc_theta(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert theta < 0


# ---------------------------------------------------------------------------
# Put-call parity: C - P = S - K*e^(-rT)
# ---------------------------------------------------------------------------


class TestPutCallParity:
    def test_put_call_parity_atm(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        C = _bs_price(S, K, T, r, sigma, "call")
        P = _bs_price(S, K, T, r, sigma, "put")
        expected_diff = S - K * math.exp(-r * T)
        assert abs((C - P) - expected_diff) < 1e-6

    def test_put_call_parity_itm_call(self):
        S, K, T, r, sigma = 120, 100, 0.5, 0.03, 0.30
        C = _bs_price(S, K, T, r, sigma, "call")
        P = _bs_price(S, K, T, r, sigma, "put")
        expected_diff = S - K * math.exp(-r * T)
        assert abs((C - P) - expected_diff) < 1e-6

    def test_put_call_parity_otm_call(self):
        S, K, T, r, sigma = 80, 100, 0.25, 0.04, 0.40
        C = _bs_price(S, K, T, r, sigma, "call")
        P = _bs_price(S, K, T, r, sigma, "put")
        expected_diff = S - K * math.exp(-r * T)
        assert abs((C - P) - expected_diff) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_deep_itm_call_delta_near_one(self):
        delta = calc_delta(200, 100, 1.0, 0.05, 0.20, "call")
        assert delta > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        delta = calc_delta(50, 100, 1.0, 0.05, 0.20, "call")
        assert delta < 0.01

    def test_deep_itm_put_delta_near_neg_one(self):
        delta = calc_delta(50, 100, 1.0, 0.05, 0.20, "put")
        assert delta < -0.99

    def test_deep_otm_put_delta_near_zero(self):
        delta = calc_delta(200, 100, 1.0, 0.05, 0.20, "put")
        assert delta > -0.01

    def test_atm_call_delta_near_half(self):
        delta = calc_delta(100, 100, 1.0, 0.05, 0.20, "call")
        assert 0.5 < delta < 0.7

    def test_near_expiry_atm_delta_close_to_half(self):
        # Very short time, ATM call delta approaches 0.5
        delta = calc_delta(100, 100, 1 / 365, 0.05, 0.20, "call")
        assert 0.45 < delta < 0.6

    def test_zero_time_returns_zero(self):
        assert calc_delta(100, 100, 0, 0.05, 0.20, "call") == 0.0
        assert calc_gamma(100, 100, 0, 0.05, 0.20) == 0.0
        assert calc_theta(100, 100, 0, 0.05, 0.20, "call") == 0.0
        assert calc_vega(100, 100, 0, 0.05, 0.20) == 0.0

    def test_negative_time_returns_zero(self):
        assert calc_delta(100, 100, -0.1, 0.05, 0.20, "call") == 0.0

    def test_gamma_positive_for_all_moneyness(self):
        for S in [80, 100, 120]:
            assert calc_gamma(S, 100, 1.0, 0.05, 0.20) > 0

    def test_vega_positive(self):
        for S in [80, 100, 120]:
            assert calc_vega(S, 100, 1.0, 0.05, 0.20) > 0

    def test_vega_peaks_near_atm(self):
        vega_atm = calc_vega(100, 100, 1.0, 0.05, 0.20)
        vega_itm = calc_vega(120, 100, 1.0, 0.05, 0.20)
        vega_otm = calc_vega(80, 100, 1.0, 0.05, 0.20)
        assert vega_atm > vega_itm
        assert vega_atm > vega_otm


# ---------------------------------------------------------------------------
# IV from premium via bisection
# ---------------------------------------------------------------------------


class TestIVBisection:
    def test_recovers_known_iv(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.30
        call_price = _bs_price(S, K, T, r, sigma, "call")
        iv = calc_iv_from_premium(call_price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_recovers_known_iv_for_put(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        put_price = _bs_price(S, K, T, r, sigma, "put")
        iv = calc_iv_from_premium(put_price, S, K, T, r, "put")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_iv_returns_none_for_zero_time(self):
        assert calc_iv_from_premium(5.0, 100, 100, 0, 0.05, "call") is None

    def test_iv_returns_none_for_zero_premium(self):
        assert calc_iv_from_premium(0.0, 100, 100, 1.0, 0.05, "call") is None

    def test_iv_returns_none_for_negative_premium(self):
        assert calc_iv_from_premium(-1.0, 100, 100, 1.0, 0.05, "call") is None

    def test_iv_high_vol(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 1.50
        price = _bs_price(S, K, T, r, sigma, "call")
        iv = calc_iv_from_premium(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 0.01

    def test_iv_low_vol(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.05
        price = _bs_price(S, K, T, r, sigma, "call")
        iv = calc_iv_from_premium(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_iv_itm_option(self):
        S, K, T, r, sigma = 110, 100, 0.5, 0.05, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        iv = calc_iv_from_premium(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_iv_otm_option(self):
        S, K, T, r, sigma = 90, 100, 0.5, 0.05, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        iv = calc_iv_from_premium(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 0.001
