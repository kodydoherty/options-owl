"""Tests for select_flow_strike — ATM default vs validated-combo OTM strike selection."""
from __future__ import annotations

from options_owl.bot_runner import select_flow_strike


def _chain(side):
    """A small chain around spot=100. mid rises toward ATM, falls going OTM."""
    if side == "call":
        return [
            {"strike": 100, "mid": 4.00, "option_type": "call"},  # ATM
            {"strike": 102, "mid": 2.10, "option_type": "call"},  # OTM ~$2
            {"strike": 105, "mid": 0.90, "option_type": "call"},  # further OTM
            {"strike": 98, "mid": 6.00, "option_type": "call"},   # ITM
        ]
    return [
        {"strike": 100, "mid": 4.00, "option_type": "put"},   # ATM
        {"strike": 98, "mid": 2.05, "option_type": "put"},    # OTM ~$2
        {"strike": 95, "mid": 0.80, "option_type": "put"},    # further OTM
        {"strike": 102, "mid": 6.00, "option_type": "put"},   # ITM
    ]


class TestSelectFlowStrike:
    def test_atm_default_call(self):
        strike, mode = select_flow_strike(_chain("call"), 100, False, otm_mode=False, target=2.0)
        assert strike == 100 and mode == "ATM"

    def test_atm_default_put(self):
        strike, mode = select_flow_strike(_chain("put"), 100, True, otm_mode=False, target=2.0)
        assert strike == 100 and mode == "ATM"

    def test_otm_call_picks_above_spot_near_target(self):
        strike, mode = select_flow_strike(_chain("call"), 100, False, otm_mode=True, target=2.0)
        assert strike == 102 and mode == "OTM"  # mid 2.10 closest to $2, above spot

    def test_otm_put_picks_below_spot_near_target(self):
        strike, mode = select_flow_strike(_chain("put"), 100, True, otm_mode=True, target=2.0)
        assert strike == 98 and mode == "OTM"   # mid 2.05 closest to $2, below spot

    def test_otm_falls_back_to_atm_when_no_otm_side(self):
        # All strikes ITM/ATM for a call (none above spot) → ATM fallback
        chain = [{"strike": 100, "mid": 4.0}, {"strike": 98, "mid": 6.0}]
        strike, mode = select_flow_strike(chain, 100, False, otm_mode=True, target=2.0)
        assert strike == 100 and mode == "ATM"

    def test_empty_chain(self):
        assert select_flow_strike([], 100, False, True, 2.0) == (None, None)

    def test_zero_spot(self):
        assert select_flow_strike(_chain("call"), 0, False, True, 2.0) == (None, None)
