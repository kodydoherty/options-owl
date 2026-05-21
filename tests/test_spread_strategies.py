"""Tests for spread strategy construction, P&L, and close logic."""

import pytest

from options_owl.execution.spread_strategies import (
    SpreadStrategy,
    StrategyType,
    build_bear_put_spread,
    build_bull_call_spread,
    build_iron_condor,
    compute_spread_pnl,
    should_close_spread,
)


# ---------------------------------------------------------------------------
# Option chain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def call_chain() -> dict[float, float]:
    """Simulated call option chain: strike -> premium."""
    return {
        95.0: 8.00,
        100.0: 5.00,
        105.0: 3.00,
        110.0: 1.50,
        115.0: 0.75,
    }


@pytest.fixture
def put_chain() -> dict[float, float]:
    """Simulated put option chain: strike -> premium."""
    return {
        85.0: 0.50,
        90.0: 1.00,
        95.0: 2.50,
        100.0: 5.00,
        105.0: 8.00,
    }


@pytest.fixture
def iron_condor_chain() -> dict[str, dict[float, float]]:
    """Option chain for iron condor with calls and puts."""
    return {
        "calls": {
            100.0: 5.00,
            105.0: 3.00,
            110.0: 1.50,
            115.0: 0.75,
        },
        "puts": {
            85.0: 0.50,
            90.0: 1.00,
            95.0: 2.50,
            100.0: 5.00,
        },
    }


# ---------------------------------------------------------------------------
# Bull call spread tests
# ---------------------------------------------------------------------------

class TestBullCallSpread:
    def test_construction(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(
            entry_strike=100.0,
            entry_premium=5.00,
            width=5.0,
            option_chain=call_chain,
        )
        assert spread.strategy_type == StrategyType.VERTICAL_SPREAD
        assert len(spread.legs) == 2

        long_leg = spread.legs[0]
        assert long_leg.strike == 100.0
        assert long_leg.action == "buy"
        assert long_leg.option_type == "call"
        assert long_leg.premium == 5.00

        short_leg = spread.legs[1]
        assert short_leg.strike == 105.0
        assert short_leg.action == "sell"
        assert short_leg.option_type == "call"
        assert short_leg.premium == 3.00

    def test_max_profit_and_loss(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(
            entry_strike=100.0,
            entry_premium=5.00,
            width=5.0,
            option_chain=call_chain,
        )
        # Net debit = 5.00 - 3.00 = 2.00
        # Max profit = (5 - 2) * 100 = $300
        # Max loss = 2 * 100 = $200
        assert spread.max_profit == pytest.approx(300.0)
        assert spread.max_loss == pytest.approx(200.0)

    def test_breakeven(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(
            entry_strike=100.0,
            entry_premium=5.00,
            width=5.0,
            option_chain=call_chain,
        )
        # Breakeven = 100 + 2.00 = 102.00
        assert spread.breakeven == pytest.approx(102.0)

    def test_net_debit(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(
            entry_strike=100.0,
            entry_premium=5.00,
            width=5.0,
            option_chain=call_chain,
        )
        # Debit spread: net_credit_or_debit should be negative
        assert spread.net_credit_or_debit == pytest.approx(-2.0)

    def test_missing_short_strike_raises(self) -> None:
        chain = {100.0: 5.00}  # missing 105
        with pytest.raises(ValueError, match="not found"):
            build_bull_call_spread(100.0, 5.00, 5.0, chain)

    def test_pnl_at_profit(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(100.0, 5.00, 5.0, call_chain)
        # Premiums moved in our favor: long up, short stayed
        current = {100.0: 7.00, 105.0: 3.00}
        pnl = compute_spread_pnl(spread, current)
        # Long: (7.00 - 5.00) * 100 = +200
        # Short: (3.00 - 3.00) * 100 = 0
        assert pnl == pytest.approx(200.0)

    def test_pnl_at_loss(self, call_chain: dict[float, float]) -> None:
        spread = build_bull_call_spread(100.0, 5.00, 5.0, call_chain)
        # Both legs lost value (underlying dropped)
        current = {100.0: 2.00, 105.0: 1.00}
        pnl = compute_spread_pnl(spread, current)
        # Long: (2.00 - 5.00) * 100 = -300
        # Short: (3.00 - 1.00) * 100 = +200
        assert pnl == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Bear put spread tests
# ---------------------------------------------------------------------------

class TestBearPutSpread:
    def test_construction(self, put_chain: dict[float, float]) -> None:
        spread = build_bear_put_spread(
            entry_strike=100.0,
            entry_premium=5.00,
            width=5.0,
            option_chain=put_chain,
        )
        assert spread.strategy_type == StrategyType.VERTICAL_SPREAD
        assert len(spread.legs) == 2

        long_leg = spread.legs[0]
        assert long_leg.strike == 100.0
        assert long_leg.action == "buy"
        assert long_leg.option_type == "put"

        short_leg = spread.legs[1]
        assert short_leg.strike == 95.0
        assert short_leg.action == "sell"
        assert short_leg.option_type == "put"

    def test_max_profit_and_loss(self, put_chain: dict[float, float]) -> None:
        spread = build_bear_put_spread(100.0, 5.00, 5.0, put_chain)
        # Net debit = 5.00 - 2.50 = 2.50
        # Max profit = (5 - 2.50) * 100 = $250
        # Max loss = 2.50 * 100 = $250
        assert spread.max_profit == pytest.approx(250.0)
        assert spread.max_loss == pytest.approx(250.0)

    def test_breakeven(self, put_chain: dict[float, float]) -> None:
        spread = build_bear_put_spread(100.0, 5.00, 5.0, put_chain)
        # Breakeven = 100 - 2.50 = 97.50
        assert spread.breakeven == pytest.approx(97.5)

    def test_pnl_calculation(self, put_chain: dict[float, float]) -> None:
        spread = build_bear_put_spread(100.0, 5.00, 5.0, put_chain)
        # Underlying dropped, puts gained value
        current = {100.0: 8.00, 95.0: 4.00}
        pnl = compute_spread_pnl(spread, current)
        # Long: (8.00 - 5.00) * 100 = +300
        # Short: (2.50 - 4.00) * 100 = -150
        assert pnl == pytest.approx(150.0)

    def test_missing_short_strike_raises(self) -> None:
        chain = {100.0: 5.00}
        with pytest.raises(ValueError, match="not found"):
            build_bear_put_spread(100.0, 5.00, 5.0, chain)


# ---------------------------------------------------------------------------
# Iron condor tests
# ---------------------------------------------------------------------------

class TestIronCondor:
    def test_construction(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            ticker="SPY",
            current_price=100.0,
            expiry="2026-04-17",
            option_chain=iron_condor_chain,
            wing_width=5.0,
        )
        assert spread.strategy_type == StrategyType.IRON_CONDOR
        assert len(spread.legs) == 4

    def test_is_net_credit(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        # Short call 105 @ 3.00, short put 95 @ 2.50 => credit = 5.50
        # Long call 110 @ 1.50, long put 90 @ 1.00 => debit = 2.50
        # Net credit = 3.00
        assert spread.net_credit_or_debit == pytest.approx(3.0)

    def test_max_profit(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        # Max profit = net credit * 100 = $300
        assert spread.max_profit == pytest.approx(300.0)

    def test_max_loss(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        # Max loss = (wing_width - net_credit) * 100 = (5 - 3) * 100 = $200
        assert spread.max_loss == pytest.approx(200.0)

    def test_breakevens(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        assert isinstance(spread.breakeven, list)
        # Lower breakeven = 95 - 3.00 = 92.00
        # Upper breakeven = 105 + 3.00 = 108.00
        assert spread.breakeven[0] == pytest.approx(92.0)
        assert spread.breakeven[1] == pytest.approx(108.0)

    def test_missing_strike_raises(self) -> None:
        chain = {"calls": {105.0: 3.00}, "puts": {95.0: 2.50}}
        with pytest.raises(ValueError, match="not found"):
            build_iron_condor("SPY", 100.0, "2026-04-17", chain, wing_width=5.0)


# ---------------------------------------------------------------------------
# Spread close decision tests
# ---------------------------------------------------------------------------

class TestShouldCloseSpread:
    def test_close_at_50pct_profit(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        # Max profit = $300, 50% = $150
        assert should_close_spread(spread, current_pnl=150.0) is True
        assert should_close_spread(spread, current_pnl=149.0) is False

    def test_close_at_custom_pct(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        # Max profit = $300, 75% = $225
        assert should_close_spread(spread, 225.0, max_profit_pct=75.0) is True
        assert should_close_spread(spread, 224.0, max_profit_pct=75.0) is False

    def test_no_close_on_loss(
        self, iron_condor_chain: dict[str, dict[float, float]]
    ) -> None:
        spread = build_iron_condor(
            "SPY", 100.0, "2026-04-17", iron_condor_chain, wing_width=5.0
        )
        assert should_close_spread(spread, current_pnl=-50.0) is False

    def test_zero_max_profit_returns_false(self) -> None:
        """Edge case: strategy with zero max profit should not trigger close."""
        spread = SpreadStrategy(
            strategy_type=StrategyType.VERTICAL_SPREAD,
            legs=[],
            max_profit=0.0,
            max_loss=100.0,
            breakeven=100.0,
            net_credit_or_debit=0.0,
        )
        assert should_close_spread(spread, current_pnl=0.0) is False
