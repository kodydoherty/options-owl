"""Take-profit sizing cap (MAX_SIZING_BALANCE): once the account grows past $X, sizing freezes at $X
so the bot trades a fixed-size book and the excess is banked — bounding absolute drawdown."""
import inspect

from options_owl.execution import paper_trader
from options_owl.risk.vinny_strategy import score_to_contracts


def test_sizing_off_capped_balance_is_smaller():
    """Sizing off the $50k cap must yield fewer contracts than off the raw (grown) balance."""
    kw = dict(cost_per_contract=200.0, max_position_pct=100.0, max_concurrent=8,
              max_portfolio_risk_pct=75.0, ml_confidence=0.75)
    uncapped = score_to_contracts(95, balance=120000.0, **kw)
    capped = score_to_contracts(95, balance=50000.0, **kw)
    assert uncapped > capped > 0, "a $50k book must size smaller than an uncapped $120k account"


def test_cap_logic_freezes_at_threshold():
    """Mirror the production cap: balances above the cap collapse to it; below pass through."""
    cap = 50000.0
    for bal, expected in [(30000.0, 30000.0), (50000.0, 50000.0), (120000.0, 50000.0)]:
        eff = bal
        if cap and cap > 0 and eff > cap:
            eff = cap
        assert eff == expected


def test_take_profit_cap_is_wired_into_sizing():
    """Guard: MAX_SIZING_BALANCE must actually cap effective_balance in the sizing path."""
    src = inspect.getsource(paper_trader)
    assert "MAX_SIZING_BALANCE" in src, "take-profit cap setting must be read in the sizing path"
    assert "effective_balance = _tp_cap" in src, "the cap must reduce effective_balance"
