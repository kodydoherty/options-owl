# Signal Source A/B Testing Plan

## Problem Statement

The 0DTE options signal scanner aggregates scores from 16+ data sources across 4 external providers (Twelve Data, Polygon, Unusual Whales, Grok AI) plus derived/computed indicators. The current win rate hovers around 60% -- effectively a coin flip with edge. The hypothesis: many of these sources overlap, contradict each other, or inject noise that dilutes the two signals that actually matter (EMA crossover direction + volume confirmation). We need to systematically measure each source's marginal contribution and remove the ones that hurt or add nothing.

The scoring system awards up to ~160+ raw points across all signals, then caps/normalizes to 100. When 16 components each contribute small amounts, no single signal dominates the score -- the result is a blended average that washes out strong directional conviction. A trade with a perfect EMA cross but weak RSI/MACD/news gets the same score as a trade with mediocre everything. Both score ~85 and get identical sizing. That is the core problem.

---

## 1. A/B Testing Philosophy

### Principles

1. **Isolate marginal contribution.** Each source must prove it improves outcomes ABOVE the baseline. "Interesting data" is not the same as "profitable data."
2. **Baseline is king.** The baseline is the simplest signal set that captures directional conviction: 9/21 EMA crossover (direction) + volume spike (confirmation). Every other source must justify its existence relative to this pair.
3. **Ablation over addition.** Rather than building up from nothing, we also tear down from the current system. Both directions reveal different things -- a source might help in isolation but hurt when combined with others (multicollinearity).
4. **Per-category analysis.** HIGH_VOL tickers (MSTR, TSLA, NVDA) behave differently from INDEX (SPY, QQQ) and STANDARD. A source might help for one category and hurt another.
5. **Statistical rigor over gut feel.** Minimum sample sizes, confidence intervals, and effect sizes. No deploying changes based on 12 trades.
6. **Cost-aware.** Each external API call has latency and dollar cost. A source that improves win rate by 1% but adds 2 seconds of latency and $50/month in API fees may not be worth it.

### What We Are NOT Testing

- **Exit engine parameters** -- V5 FSM is separately tuned and backtested.
- **Position sizing** -- flat 85% budget is a separate decision.
- **Entry pipeline gates** (score threshold, premium cap, spread gate, etc.) -- these are risk filters, not signal sources. They stay as-is.

We are ONLY testing which data sources contribute to the raw score that determines whether a signal clears the MIN_SCORE threshold and how confidently.

---

## 2. Current Source Inventory

### External Data Sources

| # | Source | Provider | API Call | Points (max) | Penalty (max) | Latency | Cost |
|---|--------|----------|----------|--------------|---------------|---------|------|
| S1 | 5min candles (OHLCV) | Twelve Data | time_series | 20 (EMA) + 15 (vol) + 10 (candle) + 10 (ATR) + 5 (momentum) | -3 to -8 | ~200ms | Included |
| S2 | 15min candles | Twelve Data | time_series | 15 (multi-TF) | -- | ~200ms | Included |
| S3 | 1min candles | Twelve Data | time_series | 10 (entry timing) | -- | ~200ms | Included |
| S4 | EMA(200) | Twelve Data | ema | 5 (macro bonus) | -- | ~150ms | Included |
| S5 | Bollinger Bands (20,2) | Twelve Data | bbands | Part of squeeze | -- | ~150ms | Included |
| S6 | RSI(9) | Twelve Data | rsi | 10 | -- | ~150ms | Included |
| S7 | MACD(5,13,1) | Twelve Data | macd | 10 | -- | ~150ms | Included |
| S8 | Keltner Channels | Twelve Data | keltner | Part of squeeze | -- | ~150ms | Included |
| S9 | News headlines | Polygon | reference/news | 5 | -8 / VETO | ~300ms | Included |
| S10 | Options chain snapshot | Polygon | snapshot/options | Greeks/OI/vol/spread | -- | ~500ms | Included |
| S11 | Minute aggs | Polygon | aggs | Recovery data | -- | ~200ms | Included |
| S12 | Options flow | Unusual Whales | flow/ticker | TBD | -- | ~400ms | $99/mo |
| S13 | Dark pool prints | Unusual Whales | darkpool/ticker | TBD | -- | ~400ms | $99/mo |
| S14 | AI analysis | Grok (xAI) | chat/completions | Confidence adj | -- | ~2-5s | ~$0.05/call |

### Alpha Data Sources (NEW — Smart Money, Insider, Sentiment)

These are non-technical signals that capture WHO is trading, not just WHAT the price is doing. They provide directional bias rather than entry timing.

| # | Source | Provider | API Call | Points (max) | Penalty (max) | Latency | Cost |
|---|--------|----------|----------|--------------|---------------|---------|------|
| S15 | Congress trades | Unusual Whales | `/api/congress/trades` | 5 (directional bias) | -- | ~400ms | Included in $99/mo UW sub |
| S16 | SEC insider trades (Form 4) | SEC EDGAR | EDGAR full-text search | 8 (net insider buys) | -- | ~300ms | Free |
| S17 | Retail sentiment | StockTwits | `/api/2/streams/symbol` | 5 (contrarian) | -3 (if crowd agrees) | ~200ms | Free |
| S18 | Congress trades (backup) | Capitol Trades | REST API | Same as S15 | -- | ~300ms | Free tier |
| S19 | Smart money flow (ML) | ML Gate 1 (trained on UW flow + Polygon) | Local inference | 5 (P(smart_money)) | -- | ~5ms | $0 (local model) |

**Why these sources are different from S1-S14:**
- S1-S14 are all **technical/quantitative** — they measure price, volume, and derivatives of price. Every algo does this.
- S15-S19 measure **information asymmetry** — insiders know things the market doesn't. Congress members sit on committees that regulate these companies. Retail sentiment extremes are reliable contrarian indicators.
- These sources provide **directional bias** (bullish/bearish lean), not entry timing. They're most useful when they confirm a technical signal.

### Derived/Computed Signals (no external call, computed from S1-S3 data)

| # | Signal | Input Source | Points (max) | Penalty (max) |
|---|--------|-------------|--------------|---------------|
| D1 | VWAP | S1 (5min candles) | Part of other calcs | -- |
| D2 | ATR expansion | S1 (5min candles) | 10 | -8 |
| D3 | ORB (Opening Range Breakout) | S1 (5min candles) | 15 | -8 |
| D4 | Candlestick patterns | S1 (5min candles) | 10 | -3 |
| D5 | Relative strength vs SPY | S1 + SPY candles | 10 | -5 |
| D6 | Key levels (S/R) | S1 (5min candles) | 5 | -- |
| D7 | Multi-candle momentum | S1 (5min candles) | 5 | -6 |
| D8 | Momentum cluster cap | Aggregate | 25 max | -- |
| D9 | Time-of-day adjustments | Clock | Variable | Variable |
| D10 | BB + Keltner squeeze | S5 + S8 | Squeeze signal | -- |

### Current Score Composition

```
MAX POSSIBLE (approx):
  EMA crossover:        20
  Multi-TF alignment:   15
  Volume spike:         15  (mandatory -- zero volume = reject)
  ORB:                  15
  RSI extreme:          10
  MACD crossover:       10
  Candlestick patterns: 10
  Entry timing (1min):  10
  ATR expansion:        10
  Rel strength vs SPY:  10
  News sentiment:        5
  Key levels:            5
  Multi-candle momentum: 5
  EMA(200) bonus:        5
  Momentum cluster cap: 25
  ─────────────────────────
  THEORETICAL MAX:     ~170 raw  → capped to 100

MIN POSSIBLE:
  Various penalties:   -8 (news VETO) -8 (ORB against) -8 (ATR contraction)
                       -5 (rel strength against) -6 (momentum fade) -3 (candle against)
  ─────────────────────────
  THEORETICAL MIN:     ~-38 penalty points
```

**The problem is clear:** With 170 possible raw points, the EMA crossover (20 pts) is only 12% of the total. A strong EMA cross can be drowned out by weak readings on 10 other indicators. The score becomes a popularity contest rather than a conviction measure.

---

## 3. Testing Methodology

### 3.1 Approach: Parallel Ablation + Isolation

For each source or signal under test, we run THREE variants against historical data:

| Variant | Description | What It Measures |
|---------|-------------|------------------|
| **Control** | All sources enabled (current production scoring) | Baseline comparison |
| **Ablation** | Remove THIS source only, keep everything else | Marginal contribution when other sources present |
| **Isolation** | ONLY this source + baseline (EMA cross + volume) | Standalone predictive power |

**Why all three?** A source might look great in isolation (it predicts direction) but add nothing when combined with others (because EMA cross already captures the same information). Conversely, a source might look useless alone but provide valuable filtering when combined. The ablation test catches both cases.

### 3.2 Metrics

For each variant, compute:

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **Win rate** | `wins / total_trades` | Core edge measure |
| **Average P&L per trade** | `mean(pnl_dollars)` | Dollar impact |
| **Median P&L per trade** | `median(pnl_dollars)` | Robust to outliers |
| **Sharpe ratio** | `mean(pnl) / std(pnl)` | Risk-adjusted return |
| **Max drawdown** | Largest peak-to-trough in cumulative P&L | Tail risk |
| **Signal frequency** | Trades per day that pass MIN_SCORE | Are we over-filtering? |
| **False positive rate** | `trades losing > 30% / total_trades` | Catastrophic loss rate |
| **Score-outcome correlation** | `pearson(score, pnl)` | Does score predict outcome? |
| **Time-to-peak (minutes)** | Median time from entry to peak premium | Signal timing quality |
| **Entry price improvement** | `(signal_premium - fill_premium) / signal_premium` | For timing-related sources |

### 3.3 Statistical Rigor

- **Minimum sample size:** 50 trades per variant per category. At ~8-12 signals/day, this requires ~5-6 days of data minimum. Use all available historical data (April 10 - present, ~200+ trades).
- **Significance test:** Two-proportion z-test for win rate differences, paired t-test for P&L differences. Report p-values.
- **Effect size:** Cohen's h for win rate, Cohen's d for P&L. Minimum meaningful effect: h >= 0.2 (small) for keep, h >= 0.5 (medium) for weight increase.
- **Confidence intervals:** 95% CI on win rate and mean P&L for each variant.
- **Multiple comparison correction:** With 20 tests, apply Bonferroni correction (alpha = 0.05/20 = 0.0025) or use Benjamini-Hochberg FDR.
- **Category stratification:** Run each test separately for HIGH_VOL, INDEX, and STANDARD tickers. A source that helps INDEX but hurts HIGH_VOL should be conditionally enabled.

### 3.4 Data Available for Backtesting

| Source | Location | Contents | Size |
|--------|----------|----------|------|
| Harvester DB | `journal/owlet-harvester/options_data.db` | Polygon options chain snapshots + stock candles (5min), captured live since April 2026 | ~7GB |
| Signal DB | `journal/owlet-kody/raw_messages.db` | All Discord signals (parsed), trade outcomes, trade events | ~50MB |
| Supabase | Cloud | Cross-agent trade data, Vince's alerts with convictions | ~500 trades |
| N8N logs | N8N workflow history | Raw API responses from Twelve Data, Polygon, UW, Grok | Varies |

**Critical limitation:** The harvester DB has options chain + stock candle data, but does NOT have the raw Twelve Data indicator values (RSI, MACD, BB, etc.) at signal time. For a true ablation test, we need to either:

1. **Reconstruct indicators** from raw candle data (preferred -- deterministic, reproducible)
2. **Capture indicator snapshots** going forward alongside signals (needed for live A/B)
3. **Use N8N API response logs** if available (historical, but may have gaps)

Option 1 is the path forward: compute all technical indicators from the harvester's stock candle data using the same parameters as the N8N workflow.

---

## 4. Test Matrix

### Phase 1: Core Signal Validation (Week 1-2)

These tests validate the baseline and the signals most likely to have strong effects.

| Test ID | Source Under Test | Baseline | Test Variant | Primary Metric | Hypothesis |
|---------|------------------|----------|--------------|----------------|------------|
| **A1** | 9/21 EMA crossover (5min) | None | EMA only | Direction accuracy (%) | ESSENTIAL -- sets direction. Expect 55-60% alone. |
| **A2** | Volume spike | EMA only | EMA + volume | Win rate delta | ESSENTIAL -- filters noise. Expect +5-10% WR. |
| **A3** | Multi-TF alignment (15min) | EMA + vol | + 15min confirm | Win rate delta | LIKELY HELPFUL -- multi-TF is a known edge. Expect +2-4%. |
| **A4** | ORB (Opening Range) | EMA + vol | + ORB | Morning WR delta | LIKELY HELPFUL -- but ONLY in first 30-45 minutes. |
| **A5** | RSI(9) extremes | EMA + vol | + RSI | Win rate delta | LIKELY HELPFUL -- extreme readings predict reversals. Expect +2-3%. |
| **A6** | MACD(5,13,1) crossover | EMA + vol | + MACD | Win rate delta | SUSPECT REDUNDANT -- may overlap with EMA cross. Test for multicollinearity. |

### Phase 2: Secondary Signals (Week 2-3)

These signals have weaker hypotheses or are more likely to be noise.

| Test ID | Source Under Test | Baseline | Test Variant | Primary Metric | Hypothesis |
|---------|------------------|----------|--------------|----------------|------------|
| **A7** | 1min entry timing | EMA + vol | + 1min candles | Entry price improvement (%) | POSSIBLY HARMFUL -- too noisy for 3-min scan interval. May cause overfitting. |
| **A8** | ATR expansion | EMA + vol | + ATR | Win rate delta | UNCLEAR -- volatility context helps in theory, but penalty (-8) may over-filter. |
| **A9** | Bollinger + Keltner squeeze | EMA + vol | + BB/Keltner squeeze | Squeeze trade WR | NARROW USE -- only relevant for squeeze breakouts. May add noise otherwise. |
| **A10** | Relative strength vs SPY | EMA + vol | + RelStr | Win rate delta | LIKELY HELPFUL for stock tickers, IRRELEVANT for index tickers. |
| **A11** | Candlestick patterns | EMA + vol | + candle patterns | Win rate delta | LIKELY WEAK -- 5min bars have limited pattern reliability. |
| **A12** | Key levels (S/R) | EMA + vol | + levels | Win rate delta | UNCLEAR -- useful if accurate, but computed S/R is often imprecise. |
| **A13** | Multi-candle momentum | EMA + vol | + momentum | Win rate delta | LIKELY REDUNDANT with EMA cross. |
| **A14** | EMA(200) macro trend | EMA + vol | + EMA200 | Win rate delta | MARGINAL -- 5pts bonus only, unlikely to move the needle. |
| **A15** | Time-of-day adjustments | EMA + vol | + ToD scoring | Signal frequency + WR by hour | CALIBRATION -- not a signal, but a filter. Check if it over-restricts. |

### Phase 3: External / Expensive Sources (Week 3)

These sources have external API costs, latency, or are untested.

| Test ID | Source Under Test | Baseline | Test Variant | Primary Metric | Hypothesis |
|---------|------------------|----------|--------------|----------------|------------|
| **A16** | Polygon news sentiment | EMA + vol | + news | Win rate delta + VETO accuracy | SUSPECT -- sentiment is lagging vs price. VETO (-8) may kill good trades. |
| **A17** | Unusual Whales flow | EMA + vol | + UW flow | Win rate delta | UNKNOWN -- flow data valuable in theory, UW data quality unverified. |
| **A18** | Unusual Whales dark pool | EMA + vol | + UW dark pool | Win rate delta | UNKNOWN -- institutional positioning signal, but latency may be too high for 0DTE. |
| **A19** | Grok AI analysis | EMA + vol | + Grok confidence | Win rate delta | SUSPECT -- LLM adds 2-5s latency, may just echo technical signals. |
| **A20** | Options chain Greeks | EMA + vol | + delta/gamma/IV | Strike selection quality | LIKELY HELPFUL for strike selection, NOT for direction. |

### Phase 4: Interaction Effects (Week 3-4)

After individual tests, check for harmful interactions.

| Test ID | Description | Test Variant | Primary Metric |
|---------|-------------|--------------|----------------|
| **A21** | Remove ALL penalties | All signals, no penalties | Over-filtering check: are penalties killing good trades? |
| **A22** | Remove news VETO only | All signals, remove VETO flag | VETO accuracy: how many vetoed trades would have won? |
| **A23** | Penalties-only scoring | EMA + vol + all penalties, no bonus signals | Is the penalty system doing the heavy lifting? |
| **A24** | Top-3 sources only | EMA + vol + best 2 from Phase 1 | Minimal viable scoring: what's the floor? |
| **A25** | Kitchen sink minus worst 3 | All sources minus 3 worst from Phase 2-3 | Incremental cleanup: does removing bad sources help? |

### Phase 5: Alpha Sources — Smart Money, Insider, Sentiment (Week 4-5)

These test the new non-technical data sources. Unlike Phases 1-3 (which remove existing sources), these ADDITION tests measure whether new sources add edge on top of the best technical baseline from Phases 1-4.

| Test ID | Source Under Test | Baseline | Test Variant | Primary Metric | Hypothesis |
|---------|------------------|----------|--------------|----------------|------------|
| **A26** | SEC insider trades (Form 4) | Best baseline from A24 | + insider bias (7d net buys) | Win rate delta | LIKELY HELPFUL — insiders have material info. Net buying in past 7 days = bullish bias confirmation. |
| **A27** | Congress trades (UW) | Best baseline from A24 | + congress bias (30d net buys) | Win rate delta | LIKELY HELPFUL — Congress members sit on oversight committees. But 45-day disclosure lag dilutes signal. |
| **A28** | StockTwits contrarian sentiment | Best baseline from A24 | + contrarian signal | Win rate delta | LIKELY HELPFUL — extreme retail bullishness (>80%) is a reliable fade signal. Moderate readings (40-60%) should be neutral. |
| **A29** | Combined alpha stack | Best baseline from A24 | + insider + congress + contrarian | Win rate delta + Sharpe | THE BIG TEST — do all three alpha sources together improve the best technical baseline? |
| **A30** | Alpha vs technical head-to-head | All technicals (no alpha) | All alpha sources + minimal tech (EMA + vol) | Win rate + profit factor | Can alpha sources with minimal TA beat full TA with no alpha? |

**Special considerations for alpha source testing:**

1. **Survivorship bias in insider data:** SEC filings are disclosed AFTER the trade. We must verify that the filing date (not trade date) falls BEFORE our signal timestamp. Otherwise we're using future data.

2. **Congress trade lag:** The STOCK Act allows 45-day disclosure. Most members file within 30 days. The signal is "someone who knows this industry bought recently" — a directional lean, not a timing signal.

3. **StockTwits contrarian validity:** Only test the CONTRARIAN signal (extreme readings). Testing sentiment-as-confirmation (bullish sentiment + bullish signal = more bullish) is likely noise.

4. **Data availability:** SEC EDGAR and StockTwits have historical data. We can reconstruct insider/sentiment state at the time of each historical trade for backtesting. Congress trades via UW may require forward-only testing.

### Phase 6: ML Gates (Week 5-8)

These test the ML-powered scoring and timing models. Each gate is trained on historical data and evaluated via walk-forward validation (train on months 1-2, predict month 3).

| Test ID | ML Gate Under Test | Baseline | Test Variant | Primary Metric | Hypothesis |
|---------|-------------------|----------|--------------|----------------|------------|
| **A31** | Flow Classifier (Gate 1) | Best from A29 | + P(smart_money) as Tier 3 | Win rate on flow-confirmed trades | LIKELY HELPFUL — classifies raw UW flow as smart vs noise. Should improve UW signal quality from A17. |
| **A32** | Entry Optimizer (Gate 2) | Best from A29 | Delay entry when model predicts dip | Entry price improvement (%) | LIKELY HELPFUL — learned version of dip-confirm gate. Should save 5-15% on premium. |
| **A33** | Quality Predictor (Gate 3) | Hand-tuned 5-tier score | ML P(win) as score | Score-outcome monotonicity (Spearman) | THE BIG BET — if P(win) has higher Spearman correlation with outcomes than hand-tuned score, switch permanently. |
| **A34** | Regime Weighter (Gate 4) | Fixed source weights | Dynamic weights by regime | Sharpe ratio improvement | UNCLEAR — requires 3+ months of data. May overfit to recent market regime. Test with strict walk-forward. |
| **A35** | Exit Advisor (Gate 5) | V5 FSM (current) | V5 FSM + ML advisory | Average P&L per trade | LOW RISK — advisory only. Log ML recommendation vs FSM decision, measure theoretical improvement. |

**ML testing protocol:**

1. **Walk-forward validation is mandatory.** No test using the same data for training and evaluation. Train on Jan-Feb, predict Mar. Train on Jan-Mar, predict Apr. Etc.
2. **Minimum 100 trades in test set.** If we don't have enough data, defer the test.
3. **Compare against random baseline.** Every ML gate must beat a random coin flip by a statistically significant margin.
4. **Feature importance audit.** After training, inspect top-10 features. If the model relies on a single feature (e.g., time_of_day alone), it's fragile and shouldn't be deployed.
5. **Overfitting check:** Train accuracy should NOT be more than 15pp higher than test accuracy. If it is, the model is memorizing, not learning.

---

## 5. Backtest Infrastructure

### 5.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│              SourceAblationBacktest                  │
│                                                     │
│  Input:                                             │
│    - Historical signals (raw_messages.db)            │
│    - Market data at signal time (options_data.db)    │
│    - enabled_sources: set[str]                      │
│    - score_weights: dict[str, float]                │
│                                                     │
│  Process:                                           │
│    1. For each historical signal:                   │
│       a. Load candle data at signal timestamp       │
│       b. Compute indicators (only enabled sources)  │
│       c. Score the signal                           │
│       d. Apply MIN_SCORE threshold                  │
│       e. Look up actual trade outcome               │
│    2. Aggregate metrics per variant                 │
│                                                     │
│  Output:                                            │
│    - Per-variant metrics table                      │
│    - Statistical comparison vs control              │
│    - Per-category breakdown                         │
│    - Recommendation (keep/demote/remove)            │
└─────────────────────────────────────────────────────┘
```

### 5.2 Implementation Plan

**File:** `scripts/backtest_source_ablation.py`

```python
"""Source ablation backtest -- measures marginal contribution of each signal source.

Usage:
    # Run all tests
    python scripts/backtest_source_ablation.py

    # Run specific test
    python scripts/backtest_source_ablation.py --test A3

    # Run specific test with verbose output
    python scripts/backtest_source_ablation.py --test A3 --verbose

    # Export results to CSV
    python scripts/backtest_source_ablation.py --csv results/ablation.csv
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Source registry ──────────────────────────────────────────────────

class Source(str, Enum):
    """Every scorable data source in the system."""
    EMA_CROSS = "ema_cross"           # 9/21 EMA crossover (5min)
    VOLUME_SPIKE = "volume_spike"     # Volume confirmation (5min)
    MULTI_TF = "multi_tf"            # 15min alignment
    ORB = "orb"                      # Opening range breakout
    RSI = "rsi"                      # RSI(9) extremes
    MACD = "macd"                    # MACD(5,13,1) crossover
    ENTRY_1MIN = "entry_1min"        # 1min entry timing
    ATR = "atr"                      # ATR expansion
    BB_KELTNER = "bb_keltner"        # Bollinger + Keltner squeeze
    REL_STRENGTH = "rel_strength"    # Relative strength vs SPY
    CANDLE_PATTERN = "candle_pattern" # Candlestick patterns
    KEY_LEVELS = "key_levels"        # Support/resistance
    MOMENTUM = "momentum"            # Multi-candle momentum
    EMA200 = "ema200"                # EMA(200) macro trend
    TOD = "tod"                      # Time-of-day adjustments
    NEWS = "news"                    # Polygon news sentiment
    UW_FLOW = "uw_flow"             # Unusual Whales options flow
    UW_DARKPOOL = "uw_darkpool"     # Unusual Whales dark pool
    GROK_AI = "grok_ai"             # Grok AI analysis
    OPTIONS_GREEKS = "options_greeks" # Options chain Greeks


# Baseline: the two sources we believe are essential
BASELINE = {Source.EMA_CROSS, Source.VOLUME_SPIKE}

# All sources (current production)
ALL_SOURCES = set(Source)


@dataclass
class TestVariant:
    """A specific A/B test configuration."""
    test_id: str
    name: str
    enabled_sources: set[Source]
    description: str


@dataclass
class VariantResult:
    """Metrics for a single test variant."""
    test_id: str
    variant_name: str
    total_signals: int          # signals evaluated
    trades_taken: int           # signals that passed score threshold
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    median_pnl: float
    sharpe: float
    max_drawdown: float
    false_positive_rate: float  # trades losing > 30%
    score_outcome_corr: float
    avg_time_to_peak_min: float
    signals_per_day: float

    # Per-category breakdown
    high_vol_wr: float | None = None
    index_wr: float | None = None
    standard_wr: float | None = None


# ── Test matrix ──────────────────────────────────────────────────────

TEST_MATRIX: dict[str, TestVariant] = {
    "A1": TestVariant("A1", "EMA only", {Source.EMA_CROSS}, "Direction accuracy baseline"),
    "A2": TestVariant("A2", "EMA + Volume", BASELINE, "Core baseline"),
    "A3": TestVariant("A3", "EMA + Vol + 15min", BASELINE | {Source.MULTI_TF}, "Multi-TF alignment"),
    "A4": TestVariant("A4", "EMA + Vol + ORB", BASELINE | {Source.ORB}, "Opening range breakout"),
    "A5": TestVariant("A5", "EMA + Vol + RSI", BASELINE | {Source.RSI}, "RSI extremes"),
    "A6": TestVariant("A6", "EMA + Vol + MACD", BASELINE | {Source.MACD}, "MACD crossover"),
    # ... (Phase 2, 3, 4 variants follow same pattern)
}


class SourceScorer:
    """Score a signal using only enabled sources.

    Each score_* method returns (points, penalty) and is only called
    if its source is in self.enabled_sources.
    """

    def __init__(self, enabled_sources: set[Source]):
        self.sources = enabled_sources

    def score(self, market_data: dict[str, Any]) -> int:
        total = 0

        if Source.EMA_CROSS in self.sources:
            total += self._score_ema_cross(market_data)

        if Source.VOLUME_SPIKE in self.sources:
            total += self._score_volume(market_data)

        if Source.MULTI_TF in self.sources:
            total += self._score_multi_tf(market_data)

        if Source.RSI in self.sources:
            total += self._score_rsi(market_data)

        if Source.MACD in self.sources:
            total += self._score_macd(market_data)

        # ... all other sources ...

        return self._normalize(total)

    def _normalize(self, raw: int) -> int:
        """Normalize raw score relative to number of enabled sources.

        KEY INSIGHT: When testing with fewer sources, the raw score
        must be scaled so MIN_SCORE threshold still makes sense.
        Otherwise, 'EMA only' (max 20 pts) can never pass MIN_SCORE=78.

        Approach: scale by (max_possible_all_sources / max_possible_enabled).
        """
        max_all = 170  # approx max with all sources
        max_enabled = self._max_possible()
        if max_enabled == 0:
            return 0
        return int(raw * (max_all / max_enabled))

    def _max_possible(self) -> int:
        """Max possible score given enabled sources."""
        maxes = {
            Source.EMA_CROSS: 20, Source.VOLUME_SPIKE: 15,
            Source.MULTI_TF: 15, Source.ORB: 15,
            Source.RSI: 10, Source.MACD: 10,
            Source.CANDLE_PATTERN: 10, Source.ENTRY_1MIN: 10,
            Source.ATR: 10, Source.REL_STRENGTH: 10,
            Source.NEWS: 5, Source.KEY_LEVELS: 5,
            Source.MOMENTUM: 5, Source.EMA200: 5,
        }
        return sum(maxes.get(s, 5) for s in self.sources)

    # Individual scorers -- each mirrors the N8N workflow logic
    def _score_ema_cross(self, data: dict) -> int: ...
    def _score_volume(self, data: dict) -> int: ...
    def _score_multi_tf(self, data: dict) -> int: ...
    def _score_rsi(self, data: dict) -> int: ...
    def _score_macd(self, data: dict) -> int: ...
    # ... etc
```

### 5.3 Indicator Reconstruction

Since the harvester DB stores raw 5min stock candles, we can reconstruct all Twelve Data indicators locally:

```python
"""Reconstruct technical indicators from raw candle data.

Uses the same parameters as the N8N workflow to ensure consistency:
  - EMA: 9, 21 periods on 5min candles
  - RSI: 9 periods
  - MACD: fast=5, slow=13, signal=1
  - Bollinger Bands: 20 periods, 2 sigma
  - ATR: 14 periods
  - EMA(200): 200 periods on 5min candles (~16 hours of data needed)
"""

import pandas as pd
import numpy as np


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=5, slow=13, signal=1):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol
```

### 5.4 Outcome Lookup

For each historical signal, we need the actual trade outcome. Two paths:

**Path A -- Use actual trade outcomes (preferred for live trades):**
```sql
-- Join signals to paper_trades outcomes
SELECT
    ts.id as signal_id,
    ts.ticker,
    ts.score,
    ts.direction,
    ts.created_at,
    pt.premium_per_contract as entry_premium,
    pt.exit_premium,
    pt.pnl_dollars,
    pt.exit_reason,
    pt.exit_source
FROM trade_signals ts
JOIN paper_trades pt ON pt.signal_id = ts.id
WHERE pt.status = 'closed'
  AND (pt.exit_source = 'ai' OR pt.exit_source IS NULL)  -- exclude manual closes
ORDER BY ts.created_at
```

**Path B -- Simulate outcome from options chain data (for signals not taken):**
```sql
-- Get options chain snapshots after signal timestamp
SELECT
    captured_at,
    bid, ask, mid,
    underlying_price
FROM option_snapshots
WHERE ticker = ? AND strike = ? AND expiry = ?
  AND captured_at BETWEEN ? AND ?
ORDER BY captured_at
```

For ablation tests, we MUST use Path B because some variants will "take" trades that production didn't (different score threshold), and vice versa. The harvester DB provides the option premium trajectory needed to simulate V5 FSM exits.

### 5.5 Score Normalization Strategy

**The fundamental problem with ablation testing of scores:** When you remove a source worth 15 points, raw scores drop by ~15 points across the board. If MIN_SCORE is 78, many previously-passing signals will fail. This doesn't mean the removed source was valuable -- it means the threshold is calibrated for the full source set.

**Solution: Normalize scores to a common scale.**

For each variant, compute the score as a PERCENTAGE of the maximum possible for that variant's enabled sources:

```
normalized_score = (raw_score / max_possible_for_variant) * 100
```

Then apply MIN_SCORE to the normalized score. This way, a 16/20 with EMA-only (80%) is comparable to a 130/170 with all sources (76%).

**Alternative: Fixed threshold approach.** Use a percentile-based threshold: "take the top N% of signals by score, regardless of absolute score." This avoids the normalization problem entirely. Set N to match current trade frequency (~8-12 signals/day taken).

**Recommendation:** Use the percentile approach for the backtest, then validate with a fixed normalized threshold. If results match, the normalization is sound.

---

## 6. Live A/B Testing (Post-Backtest)

### 6.1 Shadow Scoring Architecture

After backtests identify promising variants, deploy shadow scoring to production:

```python
# In discord_collector.py, after parsing signal:

async def _shadow_score(self, signal: TradeSignal, market_data: dict):
    """Compute scores for all active A/B test variants. Log but don't act."""
    variants = {
        "production": ALL_SOURCES,
        "minimal": {Source.EMA_CROSS, Source.VOLUME_SPIKE, Source.MULTI_TF},
        "no_news": ALL_SOURCES - {Source.NEWS},
        "no_grok": ALL_SOURCES - {Source.GROK_AI},
    }

    for name, sources in variants.items():
        scorer = SourceScorer(sources)
        score = scorer.score(market_data)
        would_trade = score >= settings.MIN_SCORE

        logger.info(
            f"SHADOW_SCORE: {signal.ticker} variant={name} "
            f"score={score} would_trade={would_trade} "
            f"production_score={signal.score}"
        )

        # Persist to trade_events for analysis
        await db.insert_trade_event(
            db_path, "shadow_score",
            {"ticker": signal.ticker, "variant": name,
             "score": score, "would_trade": would_trade,
             "production_score": signal.score}
        )
```

### 6.2 Feature Flags

Add per-source feature flags to `settings.py`:

```python
# Signal source toggles (all default True for backward compatibility)
ENABLE_SOURCE_MULTI_TF: bool = True         # 15min candle alignment
ENABLE_SOURCE_RSI: bool = True              # RSI(9) extremes
ENABLE_SOURCE_MACD: bool = True             # MACD crossover
ENABLE_SOURCE_1MIN_ENTRY: bool = True       # 1min entry timing
ENABLE_SOURCE_ORB: bool = True              # Opening range breakout
ENABLE_SOURCE_ATR: bool = True              # ATR expansion
ENABLE_SOURCE_BB_KELTNER: bool = True       # Bollinger + Keltner squeeze
ENABLE_SOURCE_REL_STRENGTH: bool = True     # Relative strength vs SPY
ENABLE_SOURCE_CANDLE_PATTERN: bool = True   # Candlestick patterns
ENABLE_SOURCE_KEY_LEVELS: bool = True       # Support/resistance
ENABLE_SOURCE_MOMENTUM: bool = True         # Multi-candle momentum
ENABLE_SOURCE_EMA200: bool = True           # EMA(200) macro trend
ENABLE_SOURCE_NEWS: bool = True             # Polygon news sentiment
ENABLE_SOURCE_UW_FLOW: bool = False         # Unusual Whales flow (not yet integrated)
ENABLE_SOURCE_UW_DARKPOOL: bool = False     # Unusual Whales dark pool (not yet integrated)
ENABLE_SOURCE_GROK_AI: bool = True          # Grok AI analysis
ENABLE_SOURCE_OPTIONS_GREEKS: bool = True   # Options chain Greeks
```

These can be overridden per-bot in `docker-compose.yml` to run different variants on different owlets:

```yaml
owlet-kody:
  environment:
    - ENABLE_SOURCE_NEWS=false       # test: disable news
    - ENABLE_SOURCE_GROK_AI=false    # test: disable Grok

owlet-adam:
  environment:
    # adam runs production (control group)
```

### 6.3 Live Test Protocol

1. **Week 1-2:** Shadow scoring only. All bots run production scoring but log alternative scores.
2. **Week 3:** Analyze shadow data. Identify variants where `would_trade` differs from production and check actual outcomes.
3. **Week 4:** Deploy best variant to ONE bot (owlet-vinny, smallest portfolio). Keep others on production.
4. **Week 5:** Compare owlet-vinny results vs others. If owlet-vinny outperforms, roll out to all bots.
5. **Week 6:** Full cutover or rollback based on data.

---

## 7. Hypotheses and Expected Outcomes

### Tier 1: Almost Certainly Essential (keep)

| Source | Hypothesis | Expected Outcome | Confidence |
|--------|-----------|-------------------|------------|
| **EMA crossover (5min)** | Sets trade direction. Without it, we have no directional thesis. | Baseline WR 55-60% alone. Removing it drops WR below 50%. | 95% |
| **Volume spike** | Filters low-conviction moves. Volume confirms institutional participation. | +5-10% WR over EMA alone. Removing it increases false positives significantly. | 90% |

### Tier 2: Likely Helpful (expect to keep, possibly adjust weight)

| Source | Hypothesis | Expected Outcome | Confidence |
|--------|-----------|-------------------|------------|
| **Multi-TF alignment (15min)** | Multi-timeframe confirmation is a known edge in technical analysis. When 5min and 15min agree, conviction is higher. | +2-4% WR over baseline. More helpful for STANDARD tickers than HIGH_VOL. | 80% |
| **ORB (Opening Range)** | Strong edge in the first 30-45 minutes of trading. Opening range breakouts are well-documented. | +3-5% WR for morning trades only. Neutral or negative after 10:30 AM ET. | 75% |
| **RSI(9) extremes** | Extreme RSI readings (oversold/overbought) predict mean reversion. At RSI < 20 or > 80, the signal is strong. | +2-3% WR. Most helpful for HIGH_VOL tickers that overshoot. | 70% |
| **Options chain Greeks** | Delta, gamma, and IV inform strike selection and premium pricing. Not a directional signal but a quality filter. | Not a WR signal. Should improve average P&L by selecting better strikes. | 75% |

### Tier 3: Unclear (test to decide)

| Source | Hypothesis | Expected Outcome | Confidence |
|--------|-----------|-------------------|------------|
| **MACD(5,13,1)** | Fast MACD may be redundant with 9/21 EMA cross. Both measure short-term momentum crossovers. If redundant, it adds 10 points of noise without information. | LIKELY REDUNDANT. Ablation test will show removing MACD has <1% WR impact. | 60% |
| **ATR expansion** | Volatility context is useful, but the -8 penalty for contraction may over-filter. Options need volatility, so expansion is good, but ATR may already be captured by volume. | MIXED. Expansion bonus may help (+1-2%), but contraction penalty may hurt more (-2-3%). Net effect unclear. | 50% |
| **BB + Keltner squeeze** | Squeeze detection identifies compression before explosive moves. Theory is sound, but on 5min bars with the N8N scan interval (~3 min), the squeeze may resolve before we can act. | NARROW. Expect +3-5% WR on squeeze trades specifically, but only ~10% of signals involve squeezes. Net portfolio impact < 1%. | 50% |
| **Relative strength vs SPY** | Stocks outperforming SPY have institutional inflows. Good theory, but on 0DTE timeframes, relative strength may not be meaningful. | CATEGORY-DEPENDENT. Expect helpful for STANDARD tickers, irrelevant for INDEX (comparing SPY to SPY), potentially misleading for HIGH_VOL. | 55% |
| **Key levels (S/R)** | Support/resistance levels provide context for entries. But computed S/R from limited intraday data is noisy. | MARGINAL. Only 5 points max. Expect <1% WR impact either way. May not be worth the computation. | 45% |

### Tier 4: Suspect (likely remove or heavily demote)

| Source | Hypothesis | Expected Outcome | Confidence |
|--------|-----------|-------------------|------------|
| **1min entry timing** | 1-minute candles are too noisy for a 3-minute scan interval. By the time N8N processes the signal, the 1min candle that triggered it is 2-3 candles old. This is fitting to noise. | HARMFUL. Expect 0% WR improvement, possibly negative. The 10 points add score variance without signal. | 70% |
| **Candlestick patterns** | On 5min bars with limited history (1-2 hours), candlestick patterns lack the context they need. A "hammer" on a 5min chart is not the same as on a daily chart. | WEAK. Expect <1% WR impact. The 10 points and -3 penalty add noise. | 65% |
| **Multi-candle momentum** | If EMA cross and volume already capture momentum, this is triple-counting the same information. 5 points max, with a -6 penalty. | REDUNDANT. Removing it will have <0.5% WR impact. The penalty may be actively harmful. | 70% |
| **News sentiment** | By the time Polygon news API returns results, price has already moved. On 0DTE timeframes, news is priced in within seconds. The 5-point bonus is tiny, but the -8 penalty and VETO can kill good trades. | NET NEGATIVE. The VETO flag likely kills more winners than losers (news arrives after price moves). Expect +1-2% WR when news source is removed. | 65% |
| **Grok AI analysis** | LLM analysis adds 2-5 seconds of latency per signal. On 0DTE, 5 seconds matters. The AI is likely just summarizing the technical signals it's given -- an echo chamber, not new information. | HARMFUL VIA LATENCY. Even if the confidence assessment is slightly predictive, the latency cost outweighs it. Expect neutral WR impact but measurably worse entry prices. | 60% |
| **EMA(200)** | Only 5 points as a bonus. The 200-period EMA on 5min bars represents ~16 hours of data -- roughly 2 trading days. This is a micro-macro hybrid that fits neither timeframe well. | NEGLIGIBLE. 5-point bonus is too small to affect scoring meaningfully. Keep or remove, no measurable difference. | 60% |
| **Time-of-day** | Not a signal source but a filter. Requires higher scores at open/close. May be over-restricting during the most volatile (and often most profitable) periods. | OVER-FILTERING. Morning volatility is highest, which is good for 0DTE. Requiring higher scores at 9:30-10:00 may block the best setups. Test separately. | 55% |

### Tier 5: Unknown / Not Yet Integrated

| Source | Hypothesis | Expected Outcome | Confidence |
|--------|-----------|-------------------|------------|
| **Unusual Whales flow** | Block/sweep detection shows institutional intent. Theoretically the strongest non-price signal. But UW data quality varies, and free tier has latency. | HIGH POTENTIAL if data quality is good. Need to evaluate UW API before scoring. | 40% |
| **Unusual Whales dark pool** | Dark pool prints show where institutions are positioned. On 0DTE this may be less relevant (dark pool trades are often hedges). | UNCLEAR. Need data evaluation first. | 30% |

### Penalty System Hypothesis

The current penalty system applies -3 to -8 points across 6 different sources, with a -8 VETO from news. Total potential penalty: -38 points.

**Hypothesis:** Penalties are over-applied. A trade with a strong EMA cross (+20), volume (+15), and multi-TF (+15) = 50 raw points could be dragged down to 12 points by penalties (-38), failing the 78-point threshold despite strong directional conviction.

**Test A21-A23** will determine:
1. How many trades are killed solely by penalties?
2. What is the win rate of penalty-killed trades? (If > 60%, penalties are too aggressive.)
3. Which specific penalties are accurate filters vs noise?

---

## 8. Decision Framework

After backtesting, classify each source using this matrix:

```
                    WR Improvement (ablation - control)

                    Negative        Neutral (< 1%)    Positive (>= 2%)
                 ┌──────────────┬──────────────────┬──────────────────┐
   p < 0.05      │   REMOVE     │   REMOVE         │   KEEP           │
   (significant)  │   (harmful)  │   (dead weight)  │   (proven edge)  │
                 ├──────────────┼──────────────────┼──────────────────┤
   p >= 0.05     │   DEMOTE     │   DEMOTE         │   INVESTIGATE    │
   (not signif.) │   (suspect)  │   (noise)        │   (promising)    │
                 └──────────────┴──────────────────┴──────────────────┘
```

### Actions by Classification

| Classification | Action | Implementation |
|---------------|--------|----------------|
| **KEEP** | Retain at current weight or increase. This source has proven predictive power. | No change, or increase weight by 25-50%. |
| **INVESTIGATE** | Promising but not statistically significant. Collect more data before deciding. | Keep enabled, shadow-score with higher weight for 2 more weeks. |
| **DEMOTE** | Marginal or noisy. Reduce influence but don't remove yet. | Cut weight by 50%. If next round still shows no improvement, remove. |
| **REMOVE** | Proven harmful or dead weight. Adding noise to scores without information. | Set `ENABLE_SOURCE_X=false` in production. Remove from N8N workflow to save API calls. |

### Scoring Weight Rebalance

After removing/demoting sources, rebalance remaining source weights to maintain score distribution:

```
Current:  20 + 15 + 15 + 15 + 10 + 10 + 10 + 10 + 10 + 10 + 5 + 5 + 5 + 5 = 145 (+ 25 cluster cap = 170)
After:    Remaining sources should sum to ~100 raw (matching the 0-100 output scale)
```

**Goal state:** 3-5 sources, each worth 20-30 points, where the score is dominated by the signals that actually predict outcomes. A score of 80 should mean "3 out of 4 strong signals agree" rather than "8 out of 16 weak signals agree."

---

## 9. Timeline

| Week | Phase | Activities | Deliverables |
|------|-------|------------|-------------|
| **Week 1** | Infrastructure | Build `backtest_source_ablation.py` with indicator reconstruction from harvester candles. Implement `SourceScorer` with per-source toggle. Validate indicator values match N8N output for 10 random signals. | Working backtest script, validated indicator reconstruction. |
| **Week 2** | Phase 1 Tests | Run A1-A6 (core signal validation). Compute all metrics. Statistical comparison of each variant vs control and vs baseline. | Phase 1 results table with p-values and CIs. |
| **Week 3** | Phase 2-3 Tests | Run A7-A20 (secondary + external sources). Cross-reference with Phase 1 results. Identify interaction effects. | Full test matrix results. Preliminary keep/remove recommendations. |
| **Week 4** | Phase 4 + Analysis | Run A21-A25 (interaction + penalty tests). Compile final recommendations. Propose new weight distribution. | Decision document: which sources to keep, demote, remove. New proposed scoring formula. |
| **Week 5** | Shadow Deploy | Deploy shadow scoring to production. Feature flags for all sources. Log alternative scores alongside production scores. | Shadow scoring data accumulating. One week of side-by-side data. |
| **Week 6** | Live Cutover | Analyze shadow results. Deploy optimized scoring to owlet-vinny (smallest portfolio). Monitor for 5 trading days. | Live A/B data. Go/no-go decision for full rollout. |
| **Week 7** | Full Rollout | If owlet-vinny outperforms or matches control, roll out to all bots. Remove dead-weight API calls from N8N to reduce latency and cost. | Production deployment of optimized signal set. |

### Success Criteria

The project is successful if ANY of these are achieved:

1. **Win rate improvement >= 3%** (from ~60% to >= 63%) with statistical significance (p < 0.05).
2. **Average P&L per trade improvement >= 10%** with no increase in max drawdown.
3. **Signal latency reduction >= 1 second** by removing unnecessary API calls (Grok, UW, etc.) without WR degradation.
4. **False positive rate reduction >= 5%** (fewer trades losing > 30%).
5. **Score-outcome correlation improvement** from current near-zero to r >= 0.15 (scores become predictive again).

The last criterion is arguably the most important: if higher scores don't predict better outcomes, the entire scoring system is noise. The goal is to make score a meaningful signal again.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Insufficient historical data** | Can't reach statistical significance for rare signals (squeeze, ORB) | Extend backtest window. For rare signals, use wider CI and label as "inconclusive." |
| **Indicator reconstruction mismatch** | Our computed RSI/MACD differs from Twelve Data's due to different warmup/smoothing | Validate 20+ data points against Twelve Data API. Accept 1% tolerance. |
| **Regime dependency** | Source X helps in trending markets but hurts in choppy markets. Backtest period may not cover both. | Segment results by VIX regime (low < 15, medium 15-25, high > 25). |
| **Overfitting the backtest** | Optimizing for April-May 2026 data may not generalize | Use walk-forward validation: train on first 2/3 of data, test on last 1/3. |
| **Correlated sources mask each other** | Removing source A shows no effect because source B captures the same info. But removing BOTH would degrade. | After individual ablation, test removing the "redundancy cluster" together (e.g., remove MACD + momentum + candle patterns simultaneously). |
| **Score normalization artifacts** | Percentile-based threshold behaves differently with fewer sources | Run each variant twice: once with percentile threshold, once with normalized fixed threshold. Flag discrepancies. |
| **Live deployment risk** | Deploying a worse scoring model to a live trading bot | Always deploy to smallest portfolio first (owlet-vinny, $500). Set circuit breaker to halt after 3 consecutive losses. |

---

## Appendix A: Existing Entry Pipeline Gates (Not Under Test)

These pipeline gates are RISK FILTERS, not signal sources. They remain active regardless of which scoring sources are enabled:

| Gate | Purpose |
|------|---------|
| BlockedTickerGate | Historically unprofitable tickers |
| ScoreGate | MIN_SCORE threshold (78) |
| PremiumGate | Valid premium exists |
| PremiumCapGate | Non-index premium < $5-9 (V6) |
| SpreadCostGate | Bid-ask spread < 40% (V6) |
| StopPriceGate | Stop price exists in signal |
| AntiChaseGate | Price hasn't moved >0.3% since signal |
| MomentumConfirmGate | Underlying candle momentum confirms direction |
| TimeOfDayGate | Higher score required at open/close |
| ConsecutiveLoserGate | Pause after consecutive losses |
| DailyLossGate | Daily loss limit |
| ConcurrentPositionsGate | Max 5 concurrent trades |
| DuplicateTickerGate | No duplicate ticker |
| CorrelationCapGate | Max 3 same-direction per group |
| CircuitBreakerGate | Time buffers, streaks, drawdown |
| PortfolioRiskGate | Portfolio-level risk cap |
| PerTradeRiskGate | Per-trade risk cap |
| LiquidityGate | OI / volume / spread minimums |
| WeeklyLossGate | Weekly loss limit |
| IVFilterGate | IV rank/percentile filter |
| VIXRegimeGate | VIX regime check |
| AnalystFilterGate | Bot performance filter |
| BalanceGate | Sufficient account balance |

**Note:** MomentumConfirmGate and TimeOfDayGate sit at the intersection of signal scoring and risk filtering. They are included in the test matrix (A15, A5) to evaluate their signal value, but their risk-filtering function remains separate.

## Appendix B: Quick-Start Commands

```bash
# Run the full ablation backtest (all phases)
python scripts/backtest_source_ablation.py

# Run a single test with verbose output
python scripts/backtest_source_ablation.py --test A3 --verbose

# Compare two specific variants
python scripts/backtest_source_ablation.py --compare A2 A6

# Export full results to CSV for analysis
python scripts/backtest_source_ablation.py --csv results/source_ablation_results.csv

# Run Phase 1 only (core signals)
python scripts/backtest_source_ablation.py --phase 1

# Run with walk-forward validation
python scripts/backtest_source_ablation.py --walk-forward --train-pct 0.67

# Shadow scoring analysis (after live deployment)
python scripts/analyze_shadow_scores.py --days 7
```
