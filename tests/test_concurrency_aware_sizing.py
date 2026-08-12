"""Concurrency-aware sizing: bigger per trade, but total exposure still capped."""
from options_owl.risk.vinny_strategy import score_to_contracts

BAL, COST = 21000.0, 200.0          # $21k account, $2.00 contract
DEPLOYABLE = BAL * 0.75             # 15,750


class TestConcurrencyAwareSizing:
    def _size(self, **kw):
        return score_to_contracts(
            95, cost_per_contract=COST, balance=BAL, max_position_pct=100.0,
            max_concurrent=8, max_portfolio_risk_pct=75.0, **kw)

    def test_disabled_is_a_noop(self):
        assert self._size(concurrency_slots=0) == self._size()

    def test_realistic_slots_sizes_up(self):
        base = self._size(concurrency_slots=0)
        bigger = self._size(concurrency_slots=4)
        assert bigger > base, "dividing by 4 instead of 8 must size up"
        # halving the denominator roughly doubles the size (int truncation aside);
        # asserted as a ratio so the score/confidence multiplier stays out of it
        assert 1.8 <= bigger / base <= 2.2, f"expected ~2x, got {base} -> {bigger}"

    def test_clamped_to_uncommitted_capital(self):
        """The safety property: cannot exceed the risk cap however many legs are open."""
        # 12,000 of 15,750 already committed -> only 3,750 left, less than 15750/4
        n = self._size(concurrency_slots=4, deployed_dollars=12000.0)
        assert n * COST <= DEPLOYABLE - 12000.0 + 1e-6

    def test_fully_committed_skips(self):
        assert self._size(concurrency_slots=4, deployed_dollars=DEPLOYABLE) == 0

    def test_overcommitted_skips_not_negative(self):
        assert self._size(concurrency_slots=4, deployed_dollars=DEPLOYABLE * 2) == 0

    def test_total_exposure_bounded_across_sequential_opens(self):
        """Open trades one after another; the sum must never breach the risk cap."""
        deployed = 0.0
        for _ in range(12):
            n = self._size(concurrency_slots=4, deployed_dollars=deployed)
            deployed += n * COST
        assert deployed <= DEPLOYABLE + 1e-6, f"breached risk cap: {deployed} > {DEPLOYABLE}"
