"""Spread strategy support for options trading.

Provides models and functions for constructing and managing multi-leg option
strategies: vertical spreads (bull call, bear put) and iron condors.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class StrategyType(str, Enum):
    SINGLE_LEG = "single_leg"
    VERTICAL_SPREAD = "vertical_spread"
    IRON_CONDOR = "iron_condor"


class SpreadLeg(BaseModel):
    """A single leg of a spread strategy."""

    strike: float
    option_type: str  # "call" or "put"
    premium: float
    action: str  # "buy" or "sell"
    contracts: int = 1


class SpreadStrategy(BaseModel):
    """A complete spread strategy with computed risk/reward metrics."""

    strategy_type: StrategyType
    legs: list[SpreadLeg]
    max_profit: float
    max_loss: float
    breakeven: float | list[float]
    net_credit_or_debit: float  # positive = credit, negative = debit


def build_bull_call_spread(
    entry_strike: float,
    entry_premium: float,
    width: float,
    option_chain: dict[float, float],
) -> SpreadStrategy:
    """Build a bull call spread (buy lower call, sell higher call).

    Args:
        entry_strike: The strike to buy (lower leg).
        entry_premium: Premium for the long call at entry_strike.
        width: Distance between strikes.
        option_chain: Mapping of strike -> premium for available options.

    Returns:
        A SpreadStrategy for the bull call spread.

    Raises:
        ValueError: If the short leg strike is not found in option_chain.
    """
    short_strike = entry_strike + width
    if short_strike not in option_chain:
        raise ValueError(
            f"Short leg strike {short_strike} not found in option chain"
        )

    short_premium = option_chain[short_strike]

    long_leg = SpreadLeg(
        strike=entry_strike,
        option_type="call",
        premium=entry_premium,
        action="buy",
    )
    short_leg = SpreadLeg(
        strike=short_strike,
        option_type="call",
        premium=short_premium,
        action="sell",
    )

    net_debit = entry_premium - short_premium  # cost to enter
    max_profit = (width - net_debit) * 100  # per contract, in dollars
    max_loss = net_debit * 100
    breakeven = entry_strike + net_debit

    return SpreadStrategy(
        strategy_type=StrategyType.VERTICAL_SPREAD,
        legs=[long_leg, short_leg],
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        net_credit_or_debit=-net_debit,  # negative = debit
    )


def build_bear_put_spread(
    entry_strike: float,
    entry_premium: float,
    width: float,
    option_chain: dict[float, float],
) -> SpreadStrategy:
    """Build a bear put spread (buy higher put, sell lower put).

    Args:
        entry_strike: The strike to buy (higher leg).
        entry_premium: Premium for the long put at entry_strike.
        width: Distance between strikes.
        option_chain: Mapping of strike -> premium for available options.

    Returns:
        A SpreadStrategy for the bear put spread.

    Raises:
        ValueError: If the short leg strike is not found in option_chain.
    """
    short_strike = entry_strike - width
    if short_strike not in option_chain:
        raise ValueError(
            f"Short leg strike {short_strike} not found in option chain"
        )

    short_premium = option_chain[short_strike]

    long_leg = SpreadLeg(
        strike=entry_strike,
        option_type="put",
        premium=entry_premium,
        action="buy",
    )
    short_leg = SpreadLeg(
        strike=short_strike,
        option_type="put",
        premium=short_premium,
        action="sell",
    )

    net_debit = entry_premium - short_premium
    max_profit = (width - net_debit) * 100
    max_loss = net_debit * 100
    breakeven = entry_strike - net_debit

    return SpreadStrategy(
        strategy_type=StrategyType.VERTICAL_SPREAD,
        legs=[long_leg, short_leg],
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        net_credit_or_debit=-net_debit,
    )


def build_iron_condor(
    ticker: str,
    current_price: float,
    expiry: str,
    option_chain: dict[str, dict[float, float]],
    wing_width: float = 5,
) -> SpreadStrategy:
    """Build an iron condor around the current price.

    Sells an OTM put spread and an OTM call spread to collect a net credit.

    Args:
        ticker: The underlying ticker symbol (for reference).
        current_price: Current price of the underlying.
        expiry: Expiration date string (for reference).
        option_chain: Mapping with keys "calls" and "puts", each mapping
            strike -> premium.
        wing_width: Width of each wing (distance between long and short strikes).

    Returns:
        A SpreadStrategy for the iron condor.

    Raises:
        ValueError: If required strikes are not found in the option chain.
    """
    calls = option_chain["calls"]
    puts = option_chain["puts"]

    # Short strikes: closest OTM strikes
    # Short call above current price, short put below
    short_call_strike = current_price + wing_width
    short_put_strike = current_price - wing_width
    long_call_strike = short_call_strike + wing_width
    long_put_strike = short_put_strike - wing_width

    for label, strike, chain in [
        ("Short call", short_call_strike, calls),
        ("Long call", long_call_strike, calls),
        ("Short put", short_put_strike, puts),
        ("Long put", long_put_strike, puts),
    ]:
        if strike not in chain:
            raise ValueError(f"{label} strike {strike} not found in option chain")

    short_call = SpreadLeg(
        strike=short_call_strike,
        option_type="call",
        premium=calls[short_call_strike],
        action="sell",
    )
    long_call = SpreadLeg(
        strike=long_call_strike,
        option_type="call",
        premium=calls[long_call_strike],
        action="buy",
    )
    short_put = SpreadLeg(
        strike=short_put_strike,
        option_type="put",
        premium=puts[short_put_strike],
        action="sell",
    )
    long_put = SpreadLeg(
        strike=long_put_strike,
        option_type="put",
        premium=puts[long_put_strike],
        action="buy",
    )

    # Net credit = premiums received - premiums paid
    credit_received = short_call.premium + short_put.premium
    debit_paid = long_call.premium + long_put.premium
    net_credit = credit_received - debit_paid

    max_profit = net_credit * 100
    max_loss = (wing_width - net_credit) * 100

    upper_breakeven = short_call_strike + net_credit
    lower_breakeven = short_put_strike - net_credit

    return SpreadStrategy(
        strategy_type=StrategyType.IRON_CONDOR,
        legs=[short_put, long_put, short_call, long_call],
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=[lower_breakeven, upper_breakeven],
        net_credit_or_debit=net_credit,
    )


def compute_spread_pnl(
    strategy: SpreadStrategy,
    current_premiums: dict[float, float],
) -> float:
    """Compute current P&L for a spread strategy.

    Args:
        strategy: The spread strategy to evaluate.
        current_premiums: Mapping of strike -> current premium for each leg.

    Returns:
        Current P&L in dollars (per contract).
    """
    pnl = 0.0
    for leg in strategy.legs:
        if leg.strike not in current_premiums:
            continue
        current = current_premiums[leg.strike]
        if leg.action == "buy":
            # Long leg: profit when premium increases
            pnl += (current - leg.premium) * 100 * leg.contracts
        else:
            # Short leg: profit when premium decreases
            pnl += (leg.premium - current) * 100 * leg.contracts
    return pnl


def should_close_spread(
    strategy: SpreadStrategy,
    current_pnl: float,
    max_profit_pct: float = 50.0,
) -> bool:
    """Decide whether to close a spread based on profit target.

    Standard practice for credit spreads is to close at 50% of max profit
    to lock in gains and avoid gamma risk near expiration.

    Args:
        strategy: The spread strategy.
        current_pnl: Current P&L in dollars.
        max_profit_pct: Close when this percentage of max profit is reached.

    Returns:
        True if the spread should be closed.
    """
    if strategy.max_profit <= 0:
        return False
    profit_ratio = (current_pnl / strategy.max_profit) * 100
    return profit_ratio >= max_profit_pct
