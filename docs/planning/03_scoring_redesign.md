# Scoring System Redesign: 0-100 Calibrated Scale

**Status:** Planning
**Author:** Kody
**Date:** 2026-05-21

---

## 1. Problem Statement

The current N8N signal scanner produces an uncapped raw score from ~21 signals and ~10 penalties. Typical range is 40-170+, capped at 100 for display but the raw value is forwarded to OptionsOwl via Discord (`(raw 164)` annotation).

### What is broken

1. **Unbounded scale.** Max theoretical ~160, min could go negative. A "score of 95" has no intrinsic meaning.
2. **Signal overlap.** EMA crossover + MACD crossover + multi-candle momentum all measure the same thing (price going up). A strong move triple-counts itself.
3. **Penalties are band-aids.** Exhaustion, extended move, and RSI direction penalties try to claw back inflation caused by overlap. They partially cancel signals that should never have been summed.
4. **No calibration.** Score 85 does not mean 85% win probability. Backtest (2026-05-20, 188 trades) showed scores above 78 have no correlation with win rate or P&L.
5. **Downstream ignores nuance.** OptionsOwl rejects `< 78` and applies flat 85% budget to everything above. The 78-170 range carries zero information for sizing.
6. **Threshold creep.** Time-of-day gates require 85+, premium caps tier at 120+ and 150+, anti-chase relaxes at 120+/150+. These thresholds were chosen arbitrarily and interact unpredictably.

### Current signal inventory (N8N Fetch & Score node)

| Signal | Max Pts | Overlap Group |
|--------|---------|---------------|
| 9/21 EMA Crossover (5min) | 20 | Trend/Momentum |
| Multi-TF Alignment (15min) | 15 | Trend/Momentum |
| Volume Spike (mandatory) | 15 | Activity |
| Opening Range Breakout | 15/-8 | Pattern |
| RSI(9) Extreme + Divergence | 10 | Oscillator |
| MACD(5,13,1) Crossover | 10 | Trend/Momentum |
| Candlestick Patterns | 10/-3 | Pattern |
| Entry Timing (1min) | 10 | Timing |
| ATR Expansion | 10/-8 | Activity |
| Relative Strength vs SPY | 10/-5 | Context |
| News Sentiment | 5/-8/VETO | Catalyst |
| Key Level Detection | 5 | Level |
| Multi-Candle Momentum | 5/-6 | Trend/Momentum |
| Momentum Cluster Cap | 25 max | Aggregate cap |
| EMA(200) Macro | 5 | Trend/Momentum |
| Time-of-Day | variable | Timing |
| Recent Streak Penalty | -variable | Risk |
| Exhaustion Chase | -variable | Risk |
| Extended Move | -variable | Risk |
| RSI Direction Confirmation | -8 | Risk |

**Overlap groups with combined weight:**
- **Trend/Momentum:** EMA(20) + Multi-TF(15) + MACD(10) + Multi-Candle(5) + EMA200(5) = 55 pts from one concept
- **Activity:** Volume(15) + ATR(10) = 25 pts from one concept
- **Risk penalties** partially cancel the inflated momentum group

---

## 2. Design Goals

| Goal | Metric |
|------|--------|
| **Bounded 0-100 scale** | `0 <= score <= 100` always, no clipping needed |
| **Calibrated** | Score bucket N should produce ~N% win rate (within 10pp) |
| **Monotonic** | Higher score = higher expected P&L (verified quarterly) |
| **Orthogonal signals** | No two signals measure the same market dynamic |
| **Minimal** | Fewer, stronger signals. Each must independently add information |
| **Debuggable** | Full breakdown logged: `{tier: points, max, reasons}` |
| **Backward compatible** | Transition period emits both old and new scores |

---

## 3. Proposed 0-100 Architecture

Five tiers, each answering a distinct question. Points within each tier are capped at the tier maximum.

### Tier 1: Direction Confidence (0-40 points)

**Question: Is the underlying actually moving in the signal's direction?**

This is the most important tier. A correct directional call is necessary (though not sufficient) for profit.

| Sub-signal | Points | Source | Notes |
|------------|--------|--------|-------|
| EMA 9/21 crossover strength | 0-15 | Twelve Data 5min | Slope of cross, not just presence. Replaces binary 0/20 |
| Multi-timeframe alignment | 0-10 | Twelve Data 15min | 5m + 15m + 1h agreement. No double-count with EMA cross |
| Trend regime (ADX + EMA200) | 0-5 | Twelve Data | ADX > 25 = trending, price relative to EMA200 = macro context |
| VWAP position | 0-5 | Computed | Price above/below VWAP confirming direction |
| Key level proximity | 0-5 | Computed | Near support (calls) or resistance (puts) |

**What changed from current system:**
- MACD removed from direction tier (it overlaps with EMA cross). Moved to Tier 2 as timing confirmation.
- Multi-candle momentum removed (overlaps with EMA cross slope).
- EMA200 merged into trend regime (was standalone 5 pts).
- EMA cross rescaled from 0/20 binary to 0-15 continuous (stronger cross = more points).

### Tier 2: Timing Quality (0-30 points)

**Question: Is THIS the right moment to enter, or are we late/early?**

| Sub-signal | Points | Source | Notes |
|------------|--------|--------|-------|
| Volume confirmation | 0-10 | Twelve Data | **Mandatory minimum 3/10 to pass.** Current bar volume vs 20-bar avg |
| RSI positioning | 0-5 | Twelve Data | Not overbought for calls / not oversold for puts. Replaces RSI extreme + divergence |
| MACD alignment | 0-5 | Twelve Data | Histogram direction confirms entry. Demoted from 10 pts to 5 (it mostly echoes EMA cross) |
| Entry velocity | 0-5 | Twelve Data 1min | Price acceleration on 1min chart. Merges current "entry timing" + "multi-candle momentum" |
| Volatility regime (ATR) | 0-5 | Computed | ATR expanding = opportunity. Merged with volume into "activity" concept, scored independently |

**Mandatory volume gate:** If volume confirmation < 3/10 points, the entire signal is rejected (score = 0). Volume is non-negotiable for 0DTE options.

**What changed:**
- Volume reduced from 15 to 10 (was overweighted). Made mandatory via floor.
- ATR reduced from 10 to 5 (merged conceptually with volume as "activity").
- MACD demoted from primary (10 pts) to confirmation (5 pts).
- Entry timing and multi-candle momentum merged into single "entry velocity" (5 pts).

### Tier 3: Edge Amplifiers (0-20 points)

**Question: Is there extra confluence beyond direction + timing?**

These are bonus signals. A trade can score 70/100 with zero amplifier points and still be excellent.

| Sub-signal | Points | Source | Notes |
|------------|--------|--------|-------|
| ORB confirmation | 0-3 | Computed | Morning only (9:30-10:30 ET). Breakout direction matches signal |
| Candlestick pattern | 0-2 | Computed | Engulfing, hammer, shooting star at key level. Reduced from 10 |
| Relative strength vs SPY | 0-2 | Computed | Ticker outperforming/underperforming SPY in signal direction. Reduced from 10 |
| News/catalyst alignment | 0-3 | Polygon | **A/B test first.** Only award points if historical data shows news alignment improves WR. Otherwise set to 0 |
| **Insider/Congress bias** | **0-4** | **SEC EDGAR + UW Congress** | **NEW.** Net insider buys (7d) + Congress member buys (30d) confirm directional lean. See below. |
| **Contrarian sentiment** | **0-3** | **StockTwits** | **NEW.** Extreme retail bullishness (>80%) = bearish signal. Extreme bearishness (<20%) = bullish. Moderate = 0 pts. |
| **Smart money flow quality** | **0-3** | **ML Gate 1** | **NEW.** LightGBM classifier scores UW flow as smart vs noise. `P(smart_money) > 0.7` = 3 pts. Behind `ENABLE_ML_FLOW_CLASSIFIER` flag. |

**What changed:**
- ORB reduced from 15 to 3 (it's confirming, not primary).
- Candlestick reduced from 10 to 2 (low signal-to-noise on 0DTE).
- Relative strength reduced from 10 to 2.
- News gated behind A/B test results (currently it's a net negative: the VETO is useful but the +5 is noise).
- **NEW: Insider/Congress bias (0-4).** SEC EDGAR Form 4 filings + UW Congress trade data provide directional confirmation that no technical indicator can. Insiders know things the market doesn't.
- **NEW: Contrarian sentiment (0-3).** Extreme StockTwits sentiment is a reliable contrarian indicator. When everyone is bullish, fade the crowd.
- **NEW: Smart money flow quality (0-3).** ML classifier (LightGBM) trained on UW flow data distinguishes institutional hedges from speculative bets. Raw flow data is noisy; the classifier filters it.

**Insider/Congress bias scoring logic:**

```python
def score_insider_bias(insider: InsiderActivity, congress: CongressActivity, direction: Direction) -> int:
    """Score insider/Congress directional confirmation (0-4 points)."""
    points = 0

    # SEC Form 4: net insider buys in last 7 days
    if direction == Direction.CALL and insider.net_buys_7d >= 3:
        points += 2  # 3+ insiders buying = strong bullish lean
    elif direction == Direction.CALL and insider.net_buys_7d >= 1:
        points += 1  # at least 1 insider buying
    elif direction == Direction.PUT and insider.net_sells_7d >= 3:
        points += 2  # 3+ insiders selling = bearish lean

    # Congress trades: net buys in last 30 days
    if direction == Direction.CALL and congress.net_buys_30d >= 2:
        points += 2  # multiple Congress members buying
    elif direction == Direction.CALL and congress.net_buys_30d >= 1:
        points += 1

    return min(points, 4)  # cap at tier allocation
```

**Contrarian sentiment scoring logic:**

```python
def score_contrarian_sentiment(sentiment: RetailSentiment, direction: Direction) -> int:
    """Score contrarian retail sentiment (0-3 points).

    Extreme crowd agreement AGAINST our direction = contrarian confirmation.
    Extreme crowd agreement WITH our direction = warning (no points).
    """
    if sentiment.total_messages < 10:
        return 0  # insufficient data, no signal

    bull_ratio = sentiment.bull_ratio  # 0.0 to 1.0

    if direction == Direction.CALL and bull_ratio < 0.20:
        return 3  # crowd is extremely bearish, contrarian bullish = strong
    elif direction == Direction.CALL and bull_ratio < 0.35:
        return 1  # crowd is bearish, mild contrarian
    elif direction == Direction.PUT and bull_ratio > 0.80:
        return 3  # crowd is extremely bullish, contrarian bearish = strong
    elif direction == Direction.PUT and bull_ratio > 0.65:
        return 1  # crowd is bullish, mild contrarian

    return 0  # moderate sentiment, no edge
```

### Tier 4: Risk Adjustments (-10 to 0 points)

**Question: Are there reasons this trade is riskier than the signals suggest?**

Only penalties that represent genuine, independent risk factors. Not band-aids for inflated signals.

| Penalty | Points | Trigger | Notes |
|---------|--------|---------|-------|
| RSI extreme against direction | -5 | RSI(9) > 80 for calls, < 20 for puts | Overbought call / oversold put |
| Chase / extended move | -5 | Price moved > 1.5 ATR in signal direction in last 30min | You're late. Single penalty replaces both "exhaustion chase" and "extended move" |
| Late-day theta bleed | -3 | After 2:00 PM ET | 0DTE theta accelerates. Replaces time-of-day score gate |
| Earnings proximity | BLOCK | Earnings within 1 trading day | Hard reject, not a point penalty. Moved from pipeline gate |
| **Negative news sentiment** | **-5 to BLOCK** | **Polygon News API detects negative headline for ticker** | **Entry-time check only. For MID-TRADE news, see News Sentinel below.** |

**What changed:**
- Streak penalty removed (this is a portfolio-level concern, not a signal quality concern; circuit breaker handles it).
- RSI direction confirmation (-8) merged into RSI extreme (-5) with cleaner logic.
- Exhaustion chase and extended move collapsed into single "chase" penalty (-5).
- Time-of-day is now a flat -3 penalty after 2 PM, not a separate minimum-score gate.

### Tier 5: Calibration Bonus (0-10 points)

**Question: Does historical data suggest this setup outperforms the base rate?**

This tier is ML-derived and should be retrained monthly. It is the only tier that uses backward-looking trade outcomes.

| Sub-signal | Points | Source | Notes |
|------------|--------|--------|-------|
| Bayesian signature match | 0-5 | Lookup table | Hash the tier 1-3 signal combination, look up historical WR for that signature. More wins = more points |
| Ticker-specific lift | 0-5 | Lookup table | Some tickers have structurally higher WR (e.g., SPY 0DTE). Apply lift/drag based on 90-day WR |

**Implementation:** Start with simple lookup tables built from the `paper_trades` table. Graduate to LightGBM if the lookup tables plateau.

### Future: ML Quality Predictor (replaces all 5 tiers)

**Gate 3 (`ENABLE_ML_QUALITY_PREDICTOR`)** is the long-term replacement for the entire hand-tuned 5-tier scoring system. When enabled, a single LightGBM model takes ALL raw indicator values (not point scores) as features and outputs `P(win)` as the score.

```python
# When ENABLE_ML_QUALITY_PREDICTOR=True:
def compute_score_ml(raw_features: dict) -> ScoredSignal:
    """ML-based scoring: single model replaces 5 tiers.

    Features include raw indicator values (RSI, MACD histogram, EMA spread,
    volume ratio, ATR, VWAP distance, OBV slope) PLUS alpha source data
    (insider_bias, congress_bias, contrarian_sentiment, flow_quality).
    """
    p_win = ml_model.predict_proba(raw_features)  # 0.0 to 1.0
    score = int(p_win * 100)  # naturally calibrated: score 70 means ~70% WR

    return ScoredSignal(
        score=score,
        breakdown={"ml_p_win": p_win, "top_features": ml_model.top_features(raw_features)},
    )
```

**Why this is better than hand-tuned tiers:**
- Learns non-linear interactions (e.g., "RSI 35 + high volume = 78% WR" vs "RSI 35 + low volume = 45%")
- Naturally calibrated — P(win) = 0.70 means ~70% of such trades win
- No overlap problem — the model figures out which features are redundant
- Automatically adapts to new data sources (just add features and retrain)

**Deployment criteria (all must be met):**
1. Walk-forward validation: ML score has higher Spearman correlation with outcomes than hand-tuned score
2. ML P(win) is monotonic with actual win rate across deciles
3. At least 300 trades in the training set
4. Feature importance shows no single feature dominates (>50% importance)
5. A/B test (A33) shows statistically significant improvement

**Rollout:** Behind `ENABLE_ML_QUALITY_PREDICTOR` flag. During transition, log BOTH hand-tuned and ML scores. Switch when criteria are met.

**Calibration loop:**
1. After 200+ trades on new scoring, bucket scores into deciles (0-10, 10-20, ..., 90-100)
2. Compute actual win rate per bucket
3. If score 60 bucket has 45% WR, reduce Tier 5 bonus for those signal signatures
4. Re-run monthly or after regime shifts

---

## 4. Score Computation

### Recommended: Weighted Sum with Hard Cap (Option A)

```python
def compute_score(signals: dict) -> ScoredSignal:
    t1 = tier1_direction(signals)    # 0-40
    t2 = tier2_timing(signals)       # 0-30
    t3 = tier3_amplifiers(signals)   # 0-20
    t4 = tier4_risk(signals)         # -10 to 0
    t5 = tier5_calibration(signals)  # 0-10

    # Volume gate: if volume < 3/10, reject entirely
    if t2.volume_points < 3:
        return ScoredSignal(score=0, rejected=True, reason="insufficient_volume")

    raw = t1.total + t2.total + t3.total + t4.total + t5.total
    score = max(0, min(100, raw))

    return ScoredSignal(
        score=score,
        breakdown={
            "direction": t1,   # .total, .max, .reasons
            "timing": t2,
            "amplifiers": t3,
            "risk": t4,
            "calibration": t5,
        },
    )
```

**Why not sigmoid (Option B)?** Sigmoid normalization maps any distribution to 0-100, but individual signal contributions become opaque. A +5 in Tier 1 might map to +2 or +8 on the final scale depending on where you are on the curve. For debugging ("why did this trade score 73?"), linear sum with hard cap is far easier to reason about. The cap rarely binds anyway -- a perfect score across all tiers is 100 by construction.

### Score interpretation

| Range | Label | Expected WR | Trading action |
|-------|-------|-------------|----------------|
| 0-39 | Weak | < 40% | Reject. Direction unclear |
| 40-54 | Below threshold | 40-50% | Reject. Coin flip |
| 55-64 | Marginal | 50-60% | Trade with minimum size (if calibration confirms) |
| 65-79 | Solid | 60-70% | Standard allocation |
| 80-89 | Strong | 70-80% | Standard allocation (flat sizing still applies) |
| 90-100 | Elite | 80%+ | Standard allocation (resist the urge to oversize) |

**Threshold for OptionsOwl:** Replace `MIN_SCORE=78` with `MIN_SCORE=55` on the new scale. The 55 floor on a calibrated scale means "more likely to win than lose." Flat sizing remains -- scores above the floor still do not determine contract count.

---

## 5. Signal Orthogonality Analysis

The core problem: the current system has 55+ points allocated to "trend/momentum" signals that all fire together on any strong move. Here is how the redesign eliminates overlap.

### Overlap group: Trend/Momentum (current: 55 pts from 5 signals)

| Current Signal | Issue | Redesign |
|----------------|-------|----------|
| 9/21 EMA Cross (20 pts) | Primary trend signal | **Keep.** Tier 1, 0-15 pts continuous |
| Multi-TF Alignment (15 pts) | Confirms trend across timeframes | **Keep.** Tier 1, 0-10 pts. Independent axis (timeframe agreement vs cross strength) |
| MACD Cross (10 pts) | Lagging derivative of price, overlaps EMA | **Demote.** Tier 2, 0-5 pts as timing confirmation only |
| Multi-Candle Momentum (5 pts) | Recent price velocity, same info as EMA slope | **Remove.** Absorbed into EMA cross strength (slope component) |
| EMA(200) Macro (5 pts) | Long-term context | **Merge.** Into Tier 1 "trend regime" with ADX |

**Result:** 55 pts from 5 overlapping signals becomes 35 pts from 3 orthogonal signals.

### Overlap group: Activity (current: 25 pts from 2 signals)

| Current Signal | Issue | Redesign |
|----------------|-------|----------|
| Volume Spike (15 pts) | Activity confirmation | **Keep.** Tier 2, 0-10 pts + mandatory floor |
| ATR Expansion (10 pts) | Volatility = implied activity | **Keep but separate.** Tier 2, 0-5 pts. ATR measures volatility regime, volume measures participation. Distinct axes |

**Result:** 25 pts from 2 partially-overlapping signals becomes 15 pts from 2 independent signals.

### Overlap group: Risk penalties (current: ~30 pts of penalties)

| Current Penalty | Issue | Redesign |
|-----------------|-------|----------|
| Exhaustion Chase (-var) | Overlaps with extended move | **Merge** both into single "chase" penalty (-5) |
| Extended Move (-var) | Overlaps with exhaustion | **Merge** (see above) |
| RSI Direction (-8) | Overlaps with RSI extreme | **Merge** into RSI extreme (-5) |
| Recent Streak (-var) | Portfolio concern, not signal quality | **Remove.** Circuit breaker handles this at the pipeline level |

**Result:** ~30 pts of penalty from 4 overlapping checks becomes -10 max from 2 clean checks.

---

## 6. FSM-Based Signal Pipeline (State Machine Architecture)

### Why FSM for Signal Scoring

The V5 exit engine already uses an FSM (Finite State Machine) with named states and gate-based transitions. We apply the same pattern to the ENTRY scoring pipeline for the same reasons:

1. **Testability:** Each state and transition is independently testable. Mock a state, fire a gate, assert the transition.
2. **Debuggability:** At any point, the signal's current state explains exactly what happened and why.
3. **No hidden state:** All data flows through an explicit `SignalContext` object — no globals, no side effects.
4. **Gate isolation:** Each scoring gate is a pure function. Adding/removing a gate requires zero changes to other gates.

### Signal Pipeline FSM States

```
INIT → CANDLE_READY → INDICATED → SCORED → FILTERED → CHAIN_VALIDATED → EMITTED
  |         |              |          |          |              |
  |    (candle fetch  (indicators  (scoring  (quality    (options chain
  |     failed)       failed)     below     gate/veto    missing/invalid)
  |         |              |      threshold)  blocks)          |
  +-----> REJECTED ←------+----------+----------+-----------+
```

| State | Description | Data Available |
|---|---|---|
| `INIT` | Ticker selected for evaluation | ticker, scan_time |
| `CANDLE_READY` | Candle data fetched successfully | + candle_data (OHLCV) |
| `INDICATED` | All indicators computed | + indicators (EMA, RSI, MACD, etc.) |
| `SCORED` | All 5 tiers evaluated | + score, breakdown, direction |
| `FILTERED` | Passed quality gate + penalty veto + cooldown | + filter_result |
| `CHAIN_VALIDATED` | Options chain fetched and validated | + strike, premium, spread |
| `EMITTED` | Signal sent to Discord / signal DB | + output_timestamp |
| `REJECTED` | Signal blocked at any stage | + rejection_reason, rejection_stage |

### SignalContext (Full State Object)

```python
@dataclass
class SignalContext:
    """Immutable-ish context passed through the FSM pipeline.

    Every gate reads from this and writes its contribution.
    The audit log serializes this at EMITTED or REJECTED.
    """
    # Identity
    ticker: str
    scan_time: datetime
    direction: Direction | None = None

    # Stage 1: Candle data
    candles_5m: list[CandleBar] | None = None
    candles_15m: list[CandleBar] | None = None
    candles_1m: list[CandleBar] | None = None
    candle_source: str = ""  # "harvester" or "twelve_data"

    # Stage 2: Indicators
    indicators: IndicatorSet | None = None

    # Stage 3: Scoring (each tier writes its result)
    tier1_direction: TierResult | None = None
    tier2_timing: TierResult | None = None
    tier3_amplifiers: TierResult | None = None
    tier4_risk: TierResult | None = None
    tier5_calibration: TierResult | None = None
    score_total: int = 0

    # Stage 3b: Alpha sources (optional, feature-flagged)
    insider_activity: InsiderActivity | None = None
    congress_activity: CongressActivity | None = None
    retail_sentiment: RetailSentiment | None = None
    flow_quality: float | None = None  # ML Gate 1 output

    # Stage 4: Filter results
    filter_result: str = ""  # PASS / COOLDOWN / QUALITY / VETO / CHAIN_FAIL
    filter_reason: str = ""

    # Stage 5: Options chain
    strike: float | None = None
    premium: float | None = None
    spread_pct: float | None = None
    chain_source: str = ""

    # Stage 6: Output
    output_channel: str = ""  # "discord" / "signal_db"
    output_timestamp: datetime | None = None

    # Final state
    state: str = "INIT"
    rejection_reason: str = ""
    rejection_stage: str = ""
```

### Gate Interface

Every gate in the pipeline follows the same interface:

```python
class ScoringGate(Protocol):
    """Protocol for all scoring pipeline gates."""
    name: str

    async def evaluate(self, ctx: SignalContext) -> SignalContext:
        """Evaluate this gate and return updated context.

        If the gate rejects the signal, set ctx.state = "REJECTED"
        and populate ctx.rejection_reason / ctx.rejection_stage.
        """
        ...
```

This is identical in spirit to the V5 FSM exit gates. Adding a new gate = implement the protocol, add to the gate list. Removing a gate = delete from the list. No other code changes needed.

### Benefits for Testing

```python
# Unit test: single gate in isolation
def test_tier1_direction_strong_cross():
    ctx = SignalContext(ticker="NVDA", scan_time=now)
    ctx.indicators = make_indicators(ema9=105, ema21=100, adx=35)
    ctx = tier1_direction_gate.evaluate(ctx)
    assert ctx.tier1_direction.total >= 25
    assert ctx.state != "REJECTED"

# Integration test: full pipeline with mocked data
async def test_full_pipeline_emits_signal():
    ctx = SignalContext(ticker="NVDA", scan_time=now)
    with mock_candle_source(), mock_polygon_chain():
        ctx = await run_pipeline(ctx, gates=ALL_GATES)
    assert ctx.state == "EMITTED"
    assert ctx.score_total >= 55

# Property test: no gate can transition to an invalid state
@given(st.sampled_from(ALL_GATES))
def test_gate_never_corrupts_state(gate):
    ctx = make_random_context()
    result = gate.evaluate(ctx)
    assert result.state in VALID_STATES
```

## 7. Signal Pruning — Eliminating Redundant Signals

### Principle: Every Signal Must Independently Improve Win Rate

If removing a signal does not decrease win rate (tested via ablation, Doc 02 Phases 1-4), that signal is noise. Noise signals actively HURT because they:
1. Inflate scores for bad trades (false confidence)
2. Dilute the weight of signals that actually predict outcomes
3. Add code complexity and maintenance burden
4. Create multicollinearity that makes ML models fragile

### Signals Flagged for Removal (pending A/B test confirmation)

| Signal | Current Points | Why It's Likely Redundant | Keep/Remove/Demote |
|---|---|---|---|
| Multi-candle momentum | 5/-6 | Measures same thing as EMA cross slope | **REMOVE** (absorbed into EMA cross strength) |
| EMA(200) macro | 5 | Only 5 pts, too slow for intraday 0DTE | **MERGE** into trend regime |
| MACD crossover | 10 | Lagging derivative of EMA cross | **DEMOTE** to 5 pts (timing confirmation only) |
| SuperTrend | +6/-5 | Lagging trend indicator, overlaps EMA | **REMOVE** (pending A/B test A-series) |
| MFI | +6/-5 | Partially overlaps with volume + RSI | **REMOVE** (pending A/B test) |
| OBV slope | +10-15/-12 | Partially overlaps with volume spike | **DEMOTE** to 5 pts max |
| Candlestick patterns | +3-10/-3 | Low reliability on 5-min bars | **DEMOTE** to 2 pts |
| News sentiment (+5) | +5 | No proven edge for positive news | **REMOVE** (keep VETO only) |

### Signals That MUST Stay (proven or structurally necessary)

| Signal | Why It Stays |
|---|---|
| 9/21 EMA crossover | Primary direction signal. Sets the trade direction. Non-negotiable. |
| Volume spike | Mandatory gate. Zero-volume trades always lose. |
| Multi-TF alignment (15min) | Independent axis: timeframe agreement vs cross strength. Proven +2-4% WR. |
| RSI(9) extremes | Oscillator measures overbought/oversold, different axis from trend. |
| ORB (morning only) | Narrow use case but proven for first 30-45 min. |
| Key level (S/R) | Structural levels are independent of momentum indicators. |
| Time-of-day adjustments | Risk factor, not a signal. Theta acceleration is real. |

### New Signals Being Added (proven alpha)

| Signal | Points | Why It Adds Edge |
|---|---|---|
| Insider/Congress bias | 0-4 | Information asymmetry. Insiders know things price doesn't reflect yet. |
| Contrarian sentiment | 0-3 | Crowd psychology. Extreme sentiment is a reliable reversal indicator. |
| Smart money flow (ML) | 0-3 | ML classifier separates institutional bets from noise in raw flow data. |

### Net Result

| Category | Current | After Pruning | Change |
|---|---|---|---|
| Total signals | 46 | 28 | -18 (39% reduction) |
| Max possible points | ~170 | 100 | Natural 0-100 scale |
| Correlated momentum signals | 5 (55 pts) | 2 (25 pts) | -3 signals, -30 pts |
| Alpha (non-technical) signals | 0 | 3 (10 pts) | New source of edge |
| ML-derived signals | 0 | 1 (3 pts) + Tier 5 | Learned, not hand-tuned |

## 8. Implementation Plan

### Phase 1: Scoring module with FSM pipeline (week 1)

Create `options_owl/sourcing/scoring/` with pure functions that can be tested independently of N8N.

```
options_owl/sourcing/
    __init__.py
    scoring/
        __init__.py
        types.py          # ScoredSignal, TierResult, SignalBreakdown
        tier1_direction.py
        tier2_timing.py
        tier3_amplifiers.py
        tier4_risk.py
        tier5_calibration.py
        aggregator.py     # compute_score() — sums tiers, clips, logs
        thresholds.py     # MIN_SCORE, label mapping, WR targets
```

Each tier function signature:

```python
def tier1_direction(
    ema_cross_strength: float,      # -1 to +1 (negative = against)
    tf_alignment: dict[str, bool],  # {"5m": True, "15m": True, "1h": False}
    adx: float,                     # 0-100
    ema200_position: float,         # price / ema200 ratio
    vwap_position: float,           # price / vwap ratio
    key_level_distance: float,      # distance to nearest support/resistance in ATR units
    direction: Direction,           # CALL or PUT
) -> TierResult:
    """Returns TierResult(total=0-40, max=40, components={...}, reasons=[...])"""
```

### Phase 2: Score breakdown storage (week 1)

Add `score_breakdown` JSON column to `paper_trades` table:

```sql
ALTER TABLE paper_trades ADD COLUMN score_breakdown TEXT;
-- JSON: {"direction": 32, "timing": 22, "amplifiers": 8, "risk": -3, "calibration": 4}
```

Log full breakdown at INFO level on every signal evaluation:

```
SCORE: NVDA CALL score=63 | direction=32/40 (ema=15 tf=10 regime=3 vwap=4 level=0)
  | timing=22/30 (vol=8 rsi=4 macd=3 velocity=4 atr=3)
  | amplifiers=8/20 (orb=5 candle=0 relstr=3 news=0 flow=0)
  | risk=-3/0 (chase=-3) | calibration=4/10 (sig=2 ticker=2)
```

### Phase 3: Parallel scoring in N8N (week 2)

Run the new scorer alongside the existing one in the N8N Fetch & Score node. Emit both scores in the Discord message:

```
Score: 85/100 (Strong)  [v2: 63/100]
```

OptionsOwl parser (`discord_collector.py`) already extracts `(raw N)` -- extend it to extract `[v2: N]`:

```python
V2_SCORE_RE = re.compile(r"\[v2:\s*(\d+)/100\]")
```

During transition, the bot uses whichever score the `SCORING_VERSION` setting selects:

```python
# settings.py
SCORING_VERSION: str = "v1"  # "v1" = raw score, "v2" = calibrated score
```

### Phase 4: Threshold migration (week 3)

Once parallel scoring has run for 5+ trading days with logged outcomes:

1. Compute WR per decile for both v1 and v2 scores
2. Verify v2 score 55+ has >= 50% WR (the minimum useful threshold)
3. Verify v2 is monotonic (higher bucket = higher WR) -- if not, adjust tier weights
4. Switch `SCORING_VERSION=v2` and `MIN_SCORE=55`

**Settings changes when switching to v2:**

| Setting | Current (v1) | New (v2) | Notes |
|---------|-------------|----------|-------|
| `MIN_SCORE` | 78 | 55 | Calibrated: 55 means >50% WR |
| `TOD_EARLY_MIN_SCORE` | 85 | 65 | Or remove entirely (Tier 4 handles this) |
| `TOD_LATE_MIN_SCORE` | 85 | 65 | Or remove entirely |
| `V6_PREMIUM_CAP_MID` threshold | score 120+ | N/A | Remove tiered caps (score is 0-100) |
| `V6_PREMIUM_CAP_HIGH` threshold | score 150+ | N/A | Remove tiered caps |
| Anti-chase relaxation | score 120+/150+ | Remove | Tier 4 risk penalty handles this |
| `ENABLE_SCORE_SIZING` | True | False | Flat sizing stays, but the setting is moot |

### Phase 5: Calibration loop (ongoing, monthly)

After 200+ trades on v2 scoring:

```python
# scripts/calibrate_scoring.py
import sqlite3
import numpy as np

db = sqlite3.connect("journal/owlet-kody/raw_messages.db")
trades = db.execute("""
    SELECT score, pnl_dollars, score_breakdown
    FROM paper_trades
    WHERE status = 'closed'
      AND exit_source = 'ai'
      AND score_breakdown IS NOT NULL
    ORDER BY id
""").fetchall()

# Bucket into deciles
for low in range(0, 100, 10):
    high = low + 10
    bucket = [t for t in trades if low <= t[0] < high]
    wins = sum(1 for t in bucket if t[1] > 0)
    wr = wins / len(bucket) if bucket else 0
    print(f"Score {low}-{high}: {len(bucket)} trades, {wr:.0%} WR (target: {low+5}%)")
```

If a bucket's WR diverges from its target by more than 15pp, adjust the tier weights that contribute most to that bucket's scores. Specifically:

- If high-score bucket has low WR: reduce amplifier weights (Tier 3) -- they are adding false confidence
- If low-score bucket has high WR: increase direction weights (Tier 1) -- the core signal is being undervalued
- If mid-range is miscalibrated: adjust Tier 5 calibration bonus

---

## 9. Backward Compatibility

### OptionsOwl signal parsing

The `parse_trade_signal()` function in `discord_collector.py` currently:
1. Extracts display score from `(\d{1,3})/100\s*\((\w+)\)` regex
2. Prefers raw score from `\(raw\s+(\d+)\)` annotation
3. Maps strength word to `SignalStrength` enum

**Transition plan:**

| Phase | Discord message format | What OptionsOwl uses |
|-------|----------------------|---------------------|
| Phase 3 (parallel) | `85/100 (Strong) (raw 164) [v2: 63/100]` | Raw 164 (v1 behavior) |
| Phase 4 (switch) | `63/100 (Solid) [v1: 164]` | 63 (v2 is now primary) |
| Phase 5 (cleanup) | `63/100 (Solid)` | 63 (v1 annotation removed) |

### Pipeline gate changes

These gates reference score directly and need updating:

| Gate | Current behavior | v2 behavior |
|------|-----------------|-------------|
| `ScoreGate` | Rejects < 78 | Rejects < 55 (new `MIN_SCORE`) |
| `PremiumCapGate` | Tiers at score 120/150 | Single cap (score is 0-100). Remove tier logic |
| `AntiChaseGate` | Relaxes max_move for score 120+/150+ | Remove score-based relaxation. Tier 4 risk penalty handles chase |
| `TimeOfDayGate` | Requires 85+ before 9:45 / after 14:00 | Simplify: Tier 4 applies -3 for late day. Gate can use 65+ or be removed |
| `score_to_contracts()` | Flat 85% above floor 78 | Flat 85% above floor 55. No code change needed beyond threshold |

### SignalStrength enum mapping

| v2 Score Range | SignalStrength | Old equivalent |
|----------------|---------------|----------------|
| 0-39 | MARGINAL | Not traded |
| 40-54 | MODERATE | Not traded |
| 55-64 | SOLID | ~78-89 raw |
| 65-79 | GOOD | ~90-110 raw |
| 80-89 | STRONG | ~110-140 raw |
| 90-100 | ELITE | ~140+ raw |

---

## 10. Testing Strategy

### Unit tests: each tier scorer

```python
# tests/test_scoring_tier1.py
def test_strong_ema_cross_with_tf_alignment():
    result = tier1_direction(
        ema_cross_strength=0.8,       # strong cross
        tf_alignment={"5m": True, "15m": True, "1h": True},
        adx=35,                       # trending
        ema200_position=1.02,         # above 200
        vwap_position=1.005,          # above VWAP
        key_level_distance=0.3,       # near support
        direction=Direction.CALL,
    )
    assert result.total >= 30         # strong direction should score high
    assert result.total <= 40         # capped at tier max

def test_weak_cross_against_trend():
    result = tier1_direction(
        ema_cross_strength=0.1,       # barely crossed
        tf_alignment={"5m": True, "15m": False, "1h": False},
        adx=12,                       # choppy
        ema200_position=0.97,         # below 200 for a call
        vwap_position=0.995,          # below VWAP for a call
        key_level_distance=2.0,       # far from levels
        direction=Direction.CALL,
    )
    assert result.total <= 15         # weak direction
```

### Property tests: invariants

```python
from hypothesis import given, strategies as st

@given(st.floats(-1, 1), st.booleans(), st.booleans(), st.booleans(),
       st.floats(0, 100), st.floats(0.8, 1.2), st.floats(0.95, 1.05),
       st.floats(0, 5))
def test_score_always_bounded(cross, tf5, tf15, tf1h, adx, ema200, vwap, level):
    result = compute_score(...)
    assert 0 <= result.score <= 100

def test_score_monotonic_in_direction():
    """Stronger direction signal = higher score, all else equal."""
    base = make_signals(ema_cross_strength=0.3)
    strong = make_signals(ema_cross_strength=0.9)
    assert compute_score(strong).score >= compute_score(base).score
```

### Calibration tests: historical replay

```python
# tests/test_scoring_calibration.py
def test_historical_win_rate_correlation():
    """Re-score all historical trades, verify score correlates with outcome."""
    trades = load_historical_trades()  # from paper_trades DB
    for trade in trades:
        new_score = compute_score(trade.signal_data)
        trade.v2_score = new_score.score

    # Bucket and check monotonicity
    buckets = bucket_by_decile(trades)
    win_rates = [bucket_win_rate(b) for b in buckets]
    # Each bucket's WR should be >= previous bucket's WR (with tolerance)
    for i in range(1, len(win_rates)):
        assert win_rates[i] >= win_rates[i-1] - 0.10  # 10pp tolerance
```

### Regression tests: no good trades lost

```python
def test_no_profitable_trades_rejected():
    """Verify that trades which were profitable under v1 are not rejected by v2."""
    profitable_v1 = load_trades(pnl_dollars__gt=0, v1_score__gte=78)
    rejected_by_v2 = [t for t in profitable_v1 if compute_score(t.signals).score < 55]
    # Allow up to 5% of profitable trades to be rejected (noise)
    assert len(rejected_by_v2) / len(profitable_v1) < 0.05
```

---

## 11. Data Requirements

### What N8N must send (minimum viable)

For OptionsOwl to compute v2 scores locally (Phase 1), the Discord message or a sidecar API must provide these raw indicator values:

| Field | Type | Source |
|-------|------|--------|
| `ema9`, `ema21` (5min) | float | Twelve Data |
| `ema9_slope`, `ema21_slope` | float | Computed |
| `tf_alignment` | dict | Twelve Data (5m/15m/1h) |
| `adx` | float | Twelve Data |
| `ema200` | float | Twelve Data |
| `vwap` | float | Twelve Data |
| `volume_ratio` | float | Current bar vol / 20-bar avg |
| `rsi9` | float | Twelve Data |
| `macd_histogram` | float | Twelve Data |
| `atr14` | float | Twelve Data |
| `price_velocity_1m` | float | Computed from 1min bars |
| `nearest_level` | float | Computed S/R level |
| `orb_high`, `orb_low` | float | Opening range (first 5/15min) |
| `spy_return_5m` | float | For relative strength |
| `ticker_return_5m` | float | For relative strength |

**Alternative:** If N8N cannot send raw indicators, it computes the v2 score itself using the tier functions ported to JavaScript. OptionsOwl receives only the final score + breakdown JSON.

### What OptionsOwl already has

OptionsOwl's `candle_cache.py` and `market_data_stream.py` already have access to:
- 5m/15m/30m/1h/4h candles with EMA, RSI, MACD, ATR, VWAP, ADX
- Real-time premium via Polygon WebSocket
- SPY candles for relative strength

This means OptionsOwl could compute the v2 score independently as a **verification layer**, even if N8N sends its own v2 score. Discrepancies between N8N's score and OptionsOwl's independent score would be a strong rejection signal.

---

## 12. Migration Risks

| Risk | Mitigation |
|------|------------|
| New scoring rejects trades that would have been profitable | Regression test (Section 8). Phase 3 parallel scoring catches this before switch |
| Calibration bonus overfits to recent market regime | Cap Tier 5 at 10 pts. Retrain monthly. Use only 90-day lookback |
| N8N and OptionsOwl scores diverge during transition | Log both, alert on divergence > 10 pts |
| Threshold change (78 to 55) lets in more noise | Phase 4 requires 5 days of parallel data proving 55+ is profitable |
| Score breakdown logging increases DB size | JSON column, ~200 bytes per trade. Negligible |

---

## 13. Success Criteria

Before declaring v2 scoring production-ready:

1. **Bounded:** Zero trades with score outside 0-100 in 5 days of parallel scoring
2. **Monotonic:** Win rate increases with score decile (Spearman correlation > 0.5)
3. **Calibrated:** Average WR per decile is within 15pp of the decile midpoint
4. **No regression:** < 5% of historically profitable trades would be rejected by new threshold
5. **Orthogonal:** Correlation between any two tier totals < 0.6 (measured on 200+ trades)
6. **Debuggable:** Every score has a logged breakdown that a human can read and verify

---

## 14. Open Questions

1. **Should OptionsOwl compute v2 independently or trust N8N?** If independently, we need raw indicators in the Discord message or a sidecar API. If trusting N8N, we just parse the score.

2. **Volume mandatory gate: reject or penalize?** Current design rejects (score=0) if volume < 3/10. Alternative: heavy penalty (-20) but allow the trade. Historical data should decide.

3. **Tier 5 calibration: lookup table or LightGBM?** Start with lookup (simpler, debuggable). Graduate to LightGBM only if the lookup plateaus. The existing per-ticker LightGBM models in `journal/models/` could be repurposed.

4. **Should score affect sizing?** Current answer is no (flat 85%). But on a calibrated scale, it might make sense to size up on 90+ scores. Defer until calibration is proven monotonic.

5. **News signal: keep or drop?** The VETO (hard block on negative news) is valuable. The +5 for positive news is unproven. Recommendation: keep VETO as a pipeline gate, drop the +5 until A/B tested.

---

## 15. News Sentinel — Real-Time News Monitoring for Open Positions

### Problem

The current system checks news only at signal entry time (every 3-min scan via Polygon News API). Once a trade is open, there is **zero news monitoring**. Breaking news during a live trade — like the META layoff article that nuked a position mid-trade — causes losses that no technical exit gate can prevent because the price moves before any indicator updates.

This is the single biggest unaddressed risk in the system.

### Proposed Solution: News Sentinel Gate in V5 FSM

Add a new exit gate to the V5 FSM that polls for breaking news on tickers with open positions.

```python
# In exit_v5/gates.py — new gate
async def news_sentinel_gate(trade, context) -> ExitAction | None:
    """Emergency exit gate: check for breaking negative news on open positions.

    Runs every monitor cycle (5s) but only hits the API every 60s (cached).
    """
    headlines = await news_cache.get_recent(trade.ticker, max_age_seconds=60)
    if not headlines:
        return None

    for headline in headlines:
        sentiment = classify_headline(headline.title, trade.direction)
        if sentiment == "strongly_negative":
            return ExitAction(
                reason=ExitReason.NEWS_SENTINEL,
                description=f"Breaking negative news: {headline.title[:80]}",
                urgency="immediate",
            )
        elif sentiment == "negative":
            # Don't exit immediately, but tighten the trailing stop
            context.tighten_trail_multiplier = 0.5  # cut trail width in half
            return None

    return None
```

### Gate Priority

Insert between bid_disappearance (#2) and grace period:

| # | Gate | What it does |
|---|---|---|
| 1 | `eod_cutoff` | 0DTE only, 15min before close |
| 2 | `bid_disappearance` | No buyers for 30s |
| **2.5** | **`news_sentinel`** | **Breaking negative news → immediate exit or tighten trail** |
| — | **5min grace** | Skip remaining gates |

### News Sources for Sentinel

| Source | Latency | Cost | Implementation |
|---|---|---|---|
| **Polygon News API** (already have) | ~30s-2min delay | Included | Poll every 60s for open position tickers only |
| **Benzinga Pro API** | ~5-15s delay | $99/mo | Real-time headline streaming. Worth it if Polygon too slow |
| **Twitter/X firehose** | ~1-5s delay | $5K+/mo | Fastest but extremely expensive. Phase 3+ |
| **Finnhub News API** | ~1-5min delay | Free tier | Backup only, too slow for sentinel |

**Recommended Phase 1:** Polygon News API (already have access, poll every 60s). This catches articles within 1-2 minutes of publication. Not ideal for flash crashes but catches sustained news events like the META layoffs.

**Phase 2:** Add Benzinga Pro for near-real-time headlines if Polygon proves too slow.

### Headline Sentiment Classification

Option A: **Keyword-based** (fast, no API cost, deterministic)

```python
NEGATIVE_PATTERNS = [
    r"layoff|laid off|job cuts|workforce reduction",
    r"SEC investigation|subpoena|fraud|indictment",
    r"downgrade|price target cut|sell rating",
    r"earnings miss|revenue miss|guidance cut",
    r"recall|safety concern|FDA reject",
    r"tariff|sanction|ban|restrict",
    r"crash|plunge|plummet|tank|collapse",
]
```

Option B: **LightGBM text classifier** (more accurate, needs training data)
- Train on historical Polygon headlines + 5-min post-headline price move
- Features: TF-IDF of headline, ticker mention count, time_of_day, market_regime
- Label: did price drop > 1% in next 5 minutes?

Option C: **Grok AI** (most accurate, but 2-5s latency + API cost)
- Send headline + trade context to Grok: "Is this headline likely to cause {ticker} to move against a {direction} position?"
- Behind `ENABLE_GROK_NEWS_SENTINEL` flag

**Recommendation:** Start with Option A (keyword-based). It catches the obvious cases (layoffs, SEC, earnings miss) with zero latency and zero cost. Graduate to Option B after collecting headline→price-move training data.

### Impact on Existing Trades

When the sentinel fires:
- **Strongly negative news:** Immediate market sell. Don't wait for trailing stops.
- **Negative news:** Tighten all trail widths by 50%. This means if adaptive trail was allowing a 40% drop from peak, it now allows only 20%.
- **Neutral/positive news:** No action.

### Testing

```python
# Unit test: headline classification
def test_meta_layoff_headline():
    headline = "Meta Platforms to Lay Off 10,000 Employees in New Round of Cuts"
    sentiment = classify_headline(headline, Direction.CALL)
    assert sentiment == "strongly_negative"

# Integration test: gate fires on negative news
async def test_news_sentinel_exits_on_negative():
    trade = make_trade(ticker="META", direction="CALL")
    with mock_news_api(headlines=[{"title": "META layoffs announced"}]):
        action = await news_sentinel_gate(trade, context)
    assert action is not None
    assert action.reason == ExitReason.NEWS_SENTINEL
```

### Open Questions

1. **Should the sentinel also BLOCK entries?** If breaking news just hit, should we refuse to enter? Probably yes — add a news recency check to the entry pipeline.
2. **How to handle conflicting signals?** Breaking negative news + strong technical momentum up. The news should win — fundamentals override technicals in the short term.
3. **Rate limiting:** Polygon News API has a 5 req/sec limit. With 5 open trades polling every 60s, that's 5 req/min — well within limits.
