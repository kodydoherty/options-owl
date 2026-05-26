from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Direction(str, Enum):
    CALL = "call"
    PUT = "put"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class SignalStrength(str, Enum):
    ELITE = "elite"
    STRONG = "strong"
    GOOD = "good"
    SOLID = "solid"
    MODERATE = "moderate"
    MARGINAL = "marginal"


class BotSource(str, Enum):
    CAPTAIN_HOOK = "Captain Hook"
    NEVERLAND_PAN = "Neverland Pan"
    TINKER = "Tinker"
    SMEE = "Smee"
    RUFIO = "Rufio"
    UNKNOWN = "unknown"
    ML_SOURCING = "ml_sourcing"


class TradeOutcome(str, Enum):
    T1_HIT = "t1_hit"
    T2_HIT = "t2_hit"
    STOP_HIT = "stop_hit"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class TradeSignal(BaseModel):
    """A parsed trade signal from a Neverland Pirates bot."""

    ticker: str
    sentiment: Sentiment
    direction: Direction
    score: int
    strength: SignalStrength
    entry_price: float
    target_price: float
    expected_move_pct: float

    # Trade idea
    strike: float
    expiry: str
    risk_reward: float

    # Exit targets
    target_1: float | None = None
    target_1_pct: float | None = None
    target_2: float | None = None
    target_2_pct: float | None = None
    target_3: float | None = None
    target_3_pct: float | None = None
    target_4: float | None = None
    target_4_pct: float | None = None
    target_5: float | None = None
    target_5_pct: float | None = None
    stop_price: float | None = None
    stop_pct: float | None = None
    exit_by: str | None = None

    # Option picks
    atm_strike: float | None = None
    atm_premium: float | None = None
    otm_strike: float | None = None
    otm_premium: float | None = None

    # Key signals (technical indicators)
    key_signals: list[str] = []

    # Metadata
    bot_source: BotSource
    is_elite: bool = False
    source_message_id: int = 0
    source_channel: str = ""
    author: str = ""
    timestamp: datetime | None = None
    raw_text: str = ""


class PriceBar(BaseModel):
    """A single intraday price bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class ResolvedSignal(BaseModel):
    """The outcome of a trade signal after price data is analyzed."""

    signal_id: int
    outcome: TradeOutcome
    hit_price: float | None = None
    hit_time: datetime | None = None
    pnl_underlying_pct: float = 0.0
    pnl_atm_est: float | None = None
    pnl_otm_est: float | None = None
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0


class BotPerformanceReport(BaseModel):
    """Aggregated performance stats for a single bot."""

    bot_source: BotSource
    total_signals: int
    resolved_signals: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_pnl_atm: float | None = None
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_score: float = 0.0
    elite_win_rate_pct: float | None = None
    strong_win_rate_pct: float | None = None
    smee_reported_win_rate: float | None = None
    smee_reported_avg_pnl: float | None = None


class WatchlistEntry(BaseModel):
    """An entry from Rufio's pre-market watchlist."""

    ticker: str
    stage: int
    sentiment: Sentiment
    score: int
    catalyst: str | None = None


class PerformanceEntry(BaseModel):
    """A trade result from Smee's daily performance summary."""

    ticker: str
    sentiment: Sentiment
    score: int
    pnl_pct: float
    won: bool


class DailyPerformance(BaseModel):
    """Smee's daily performance summary."""

    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    trades: list[PerformanceEntry] = []
    all_time_wins: int | None = None
    all_time_total: int | None = None
