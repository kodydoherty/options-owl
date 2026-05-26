"""Shared types for the scoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class SignalState(str, Enum):
    """FSM states for the signal scoring pipeline."""

    INIT = "INIT"
    CANDLE_READY = "CANDLE_READY"
    INDICATED = "INDICATED"
    SCORED = "SCORED"
    FILTERED = "FILTERED"
    CHAIN_VALIDATED = "CHAIN_VALIDATED"
    EMITTED = "EMITTED"
    REJECTED = "REJECTED"


@dataclass
class TierResult:
    """Result from a single scoring tier."""

    total: int = 0
    max_possible: int = 0
    components: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScoredSignal:
    """Final scored signal with full breakdown."""

    score: int = 0
    direction: Direction | None = None
    rejected: bool = False
    reject_reason: str = ""
    breakdown: dict[str, TierResult] = field(default_factory=dict)


@dataclass
class SignalContext:
    """Full state object passed through the FSM pipeline.

    Every gate reads from this and writes its contribution.
    The audit log serializes this at EMITTED or REJECTED.
    """

    # Identity
    ticker: str = ""
    scan_time: str = ""
    state: SignalState = SignalState.INIT

    # Direction
    direction: Direction | None = None

    # Stage 1: Candle data
    candles_5m: list[dict] | None = None
    candles_15m: list[dict] | None = None
    candles_1m: list[dict] | None = None
    candle_source: str = ""

    # Stage 2: Indicators (filled by indicator_engine)
    indicators: object | None = None  # IndicatorSet

    # Stage 3: Scoring (each tier writes its result)
    tier1_direction: TierResult | None = None
    tier2_timing: TierResult | None = None
    tier3_amplifiers: TierResult | None = None
    tier4_risk: TierResult | None = None
    tier5_calibration: TierResult | None = None
    score_total: int = 0

    # Stage 3a: ML signal model
    ml_confidence: float | None = None
    ml_threshold: float | None = None
    ml_is_signal: bool | None = None
    ml_runner_score: float | None = None
    ml_model_source: str = ""

    # Stage 3b: Alpha sources
    insider_activity: object | None = None  # InsiderActivity
    congress_activity: object | None = None  # CongressActivity
    retail_sentiment: object | None = None  # RetailSentiment
    flow_quality: float | None = None

    # Stage 4: Filters
    filter_result: str = ""
    filter_reason: str = ""

    # Stage 5: Options chain
    strike: float | None = None
    premium: float | None = None
    spread_pct: float | None = None

    # Stage 6: Output
    output_channel: str = ""
    output_timestamp: str = ""

    # Recent outcomes (for regime detection)
    recent_signal_outcomes: list[dict] | None = None

    # Rejection tracking
    rejection_reason: str = ""
    rejection_stage: str = ""
