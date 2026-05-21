"""Exit Engine v5 — category-aware, DTE-aware FSM exit strategy.

Public API:
    ExitFSM      — stateless evaluator, one per bot
    TradeState   — mutable per-trade state, created at fill
    ExitAction   — return type from FSM evaluation
    ExitReason   — enum of exit reasons (one per gate)
    V5Config     — typed configuration (frozen dataclass)
    FSMState     — enum of FSM states (GRACE, DEVELOPING, TRAILING)
    TickerCategory — HIGH_VOL, INDEX, STANDARD
    AdaptiveTier — one tier of the adaptive trailing stop
"""

from options_owl.risk.exit_v5.config import (
    TICKER_CONFIGS,
    AdaptiveTier,
    TickerCategory,
    V5Config,
    categorize_ticker,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, FSMState, TradeState
from options_owl.risk.exit_v5.types import ExitAction, ExitReason

__all__ = [
    "AdaptiveTier",
    "ExitAction",
    "ExitFSM",
    "ExitReason",
    "FSMState",
    "TICKER_CONFIGS",
    "TickerCategory",
    "TradeState",
    "V5Config",
    "categorize_ticker",
    "get_ticker_config",
]
