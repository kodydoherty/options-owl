# V3 ML Model Evaluation Spec

## Purpose

Track every ML experiment, what we tested, what we learned, and how each approach compares to the current baseline end-to-end. This is the single source of truth for ML strategy decisions.

---

## Current Baseline (Production as of 2026-05-24)

| Metric | Value |
|---|---|
| Signal source | Discord (Neverland Pirates) |
| Entry filter | 0.8 tech + 0.2 ML, score >= 78 |
| Exit engine | V5 FSM (category-aware, DTE-aware) |
| Stops | Per-ticker configs (wide: 35/65 for 0DTE) |
| Sizing | Flat 85% budget for all scores >= 78 |
| Live P&L (Webull) | ~$9,842 over ~4 months |
| Backtest P&L (V5) | $21,650 over 188 trades (60.1% WR) |

### Baseline in Combined Scoring Sweep (97 trading days, ThetaData)

| Config | Trades | WR% | P&L | PF | Sharpe |
|---|---|---|---|---|---|
| Best quality (0.8T+0.2ML, t55, moderate) | 71 | 73.2% | varies | 3.76 | 5.15 |
| Best P&L (0.0T+1.0ML, t50, wide) | 355 | 69.6% | $3.63M* | 1.59 | N/A |
| Unfiltered ML (all candidates, wide) | 1,088+ | ~51% | -$80K | <1.0 | negative |

*Compounding inflated. Flat-sized equivalent ~$132K.

---

## The 9:30 AM Bias Problem (CRITICAL)

### What We See

94% of ML-generated candidates (1,210 of 1,285) fire at exactly 9:30 AM — the first minute of market open. Only 75 candidates across 97 days fire after 9:30.

### Why This Happens

1. **ThetaData scanning generates 1 snapshot per ticker at 9:30** — the first minute bar
2. **ML model has 1 premium observation** — features like `premium_volatility`, `premium_momentum` are all zero
3. **Tech indicators (EMA, MACD, RSI, VWAP)** are computed from the opening candle — minimal signal
4. **The model passes almost everything at 9:30** because the features don't differentiate good from bad

### Is This Survivorship Bias?

**YES, this is likely biased.** Here's why:

1. **Morning momentum correlation**: In the last ~2 months of training data, morning moves (first 30 min) have been directional. The model may be learning "enter at open, ride the morning move" — which works in trending markets but dies in chop.

2. **No negative example contrast at 9:30**: Since we enter almost every ticker at 9:30, we can't distinguish "9:30 entries that work" from "9:30 entries that fail" — the model sees opening conditions as always-positive because the FILTER (0.8 tech weight) is doing the real work downstream.

3. **Recency bias**: ThetaData goes back to Jan 2023, but market regimes change. If 2023-2024 was more mean-reverting and 2025-2026 more momentum-driven, our model learns the recent regime and overfits.

4. **No intraday re-scan**: Unlike Neverland signals that arrive 10:00-13:30 ET throughout the day, our ML scanner only fires once per ticker per day (at 9:30). We literally CAN'T test later entries because they don't exist in the data.

### How to Test for This Bias

The V3 models can partially address this:

| Test | What it Proves |
|---|---|
| **Regime model on walk-forward** | If trending days correlate with profitable 9:30 entries, we have regime-dependent alpha (not pure 9:30 magic) |
| **Ticker selection model** | If certain tickers consistently profit at 9:30 and others don't, there's signal beyond timing |
| **Entry timing retrained on lows** | If the model can find the actual low in the 15-min window, 9:30 may genuinely be "the low" sometimes |
| **Out-of-sample backtest** | Train on 2023-2024, test on 2025-2026. If 9:30 entries still work, it's not recency bias |

---

## V3 Model Suite — Design & Rationale

### Model 1: Entry Timing (Redesigned)

**Question**: "Is this candle near the LOW before a +38% run?"

**Old approach (V2)**: "Will premium go up 38% sometime in the next 120 minutes?" — this labeled any candle in a 2-hour window as positive, creating massive label noise.

**New approach (V3)**: Find the actual LOW point in the 15-min window before each +38% move. Only candles within 5% of that low are labeled positive. This is much harder to predict but much more useful — it tells us WHEN to buy, not just WHETHER.

**Why it matters for 9:30 bias**: If the model learns that 9:30 IS genuinely the low (morning gap-down that recovers), that's legitimate signal. If it just says "yes" to everything at 9:30, it's useless.

**Features**: Pre-entry premium trajectory, IV, delta/theta/vega, underlying momentum, minutes since open, bid-ask spread, volume profile.

**Isolation test**: Gate that blocks entries unless the model says we're near a local low. Test at thresholds 0.3-0.7. Compare trades taken, win rate, P&L, and average entry timing vs baseline.

### Model 2: Exit Timing

**Question**: "Should I HOLD or SELL right now?"

At every minute of an open trade, the model predicts whether the future is better (HOLD) or worse (SELL) than current premium.

**Why it's different from V5 FSM**: FSM uses fixed rules (35% stop, 20% scalp, etc). The exit model looks at current market conditions — IV expansion/contraction, underlying momentum, premium trajectory — and makes a dynamic call.

**Isolation test**: Replace V5 FSM with model-only exit decisions in the backtest simulator. Requires re-simulating exits (can't use pre-computed candidates). This is the most expensive test to run.

**Note**: Exit timing model is NOT testable against pre-computed sweep candidates (those have FSM-specific P&L baked in). We need a separate simulator for this. Deferred to Phase 2 evaluation.

### Model 3: Regime Classification

**Question**: "Is today a TRENDING day or a CHOP day?"

**Features** (daily-level, available pre-market or at 9:30):
- GEX (call/put gamma, net gamma, net delta) from UW historical
- Options volume (put/call ratio, net premium flow)
- Previous day's range, volume, 3-day avg range
- Day of week
- Early morning stock data (first 5 candles: momentum, volatility, volume)

**Label**: "trending" = at least one ATM option moved +30% in the killzone; "chop" = no ATM option moved +30%.

**Why this addresses 9:30 bias**: Regime features are available BEFORE 9:30 (GEX, prior day data) or at 9:30 (early candles). If chop days consistently lose money on 9:30 entries, skipping them improves consistency WITHOUT needing later entries.

**Isolation test**: Gate that blocks ALL trades on days the model predicts as chop. Test at thresholds 0.3-0.7. Key metric: does blocking chop days improve Sharpe ratio and reduce max drawdown, even if it reduces total P&L?

### Model 4: Ticker Selection

**Question**: "Will THIS TICKER be profitable today?"

**Features**: Per-ticker-day features (GEX, options volume, opening IV, opening delta, opening premium, early momentum, early volatility, early volume).

**Label**: "profitable" = best achievable gain on ATM options from killzone entry > 30%.

**Why this helps**: We already know some tickers are consistent losers (TSLA, AAPL, GOOGL, MSFT — see sweep_results.md). But those are ALWAYS excluded. The real question is: on any given day, should we trade NVDA or skip it? Some days NVDA is gold, other days it chops. This model makes a per-day call.

**Addresses 9:30 bias**: The model uses pre-market GEX + early morning data to decide WHICH tickers to trade at 9:30, not WHETHER to trade at 9:30. It's an orthogonal filter.

**Isolation test**: Gate that blocks tickers the model says won't be profitable today. Test at thresholds 0.3-0.7. Key metric: does it filter out the daily losers while keeping the winners?

### Model 5: Stop Calibration

**Question**: "What stop width should I use for THIS specific entry?"

**Design**: For each entry point, simulate 5 different stop widths (20%, 30%, 40%, 50%, 65%) and record which produced the best P&L. Train a regression model to predict the optimal stop.

**Why this is different from fixed stops**: The sweep showed wide stops (35/65) dominate. But that's the AVERAGE. Some entries need tight stops (quick reversals) while others need wide stops (volatile runners). If the model can distinguish, it outperforms any fixed width.

**Isolation test**: Use model-predicted stop config instead of fixed "wide". Compare P&L, max drawdown, and win rate. The key question: does right-sizing stops improve risk-adjusted returns?

### Model 6: Signal Quality (Regression)

**Question**: "How BIG will this move be?"

**Design**: Predict the peak gain % achievable from each entry point. Unlike binary classifiers, this gives a magnitude — "this is a 20% move" vs "this is a 100% move".

**Why it helps**: Position sizing. A predicted 100% move gets more contracts than a 20% move. Also acts as a filter: predicted <10% moves aren't worth the spread cost.

**Isolation test**: Gate that blocks entries below various magnitude thresholds (10%, 20%, 30%, 40%, 50%). Also test as a sizing multiplier: allocate contracts proportional to predicted magnitude.

---

## Evaluation Framework

### Phase 1: Isolation Tests (scripts/evaluate_v3_models.py)

Each model tested alone against the pre-computed sweep candidates (1,285 candidates, 97 trading days, 14 tickers).

**Standard test for each model:**
1. Load the model + its metadata
2. Run portfolio simulation with model as filter/override
3. Compare to 3 baselines:
   - BASELINE: 0.8 tech + 0.2 ML, threshold 50, wide stops
   - ML_ONLY: 1.0 ML, threshold 50, wide stops
   - UNFILTERED: every candidate, wide stops

**Metrics tracked per test:**
- Trades taken (and how many blocked by model)
- Win rate %
- Total P&L ($)
- Profit Factor
- Sharpe ratio (annualized)
- Max drawdown %
- Average win / average loss
- Per-ticker breakdown
- Delta from baseline (how much does the model add/remove?)

### Phase 2: Combo Tests

All pairs, triples, and full combination of models tested together:
- entry_timing + regime
- entry_timing + ticker_select
- regime + ticker_select
- entry_timing + regime + ticker_select
- All models combined
- Baseline + stop_calibrate only
- Each combo with and without stop_calibrate override

### Phase 3: Walk-Forward Validation (Future)

Train on 2023-2024 data, test on 2025-2026. This directly tests whether:
1. 9:30 entries work out-of-sample
2. Regime model generalizes across market conditions
3. Ticker selection isn't just overfitting to recent winners

### Phase 4: Exit Timing Model (Future)

Requires re-simulating exits (not pre-computed). Build a separate backtester that:
1. Takes entry from the candidate list
2. Loads 1-min option bars from ThetaData
3. At each minute, asks exit_timing model HOLD vs SELL
4. Compares total P&L and holding time vs V5 FSM

---

## Results Tracking

### Experiment Log

| # | Date | Experiment | Result | Key Finding |
|---|---|---|---|---|
| 1 | 2026-05-23 | Combined scoring sweep (21,600 combos) | Completed | Tech 0.8 dominates; wide stops win; ML warmup kills profit |
| 2 | 2026-05-23 | Simpsons veto gates (7 gates) | Completed | Most gates don't work with our data; only afternoon+spread kept |
| 3 | 2026-05-23 | ML-only after real stock data retrain | Completed | ML-only now profitable ($3.6M compound) — real stock OHLC was key |
| 4 | 2026-05-24 | V3 model training (6 models) | IN PROGRESS | ~6-8 hours, entry_timing running |
| 5 | 2026-05-24 | V3 isolation tests | PENDING | Waiting for training |
| 6 | 2026-05-24 | V3 combo strategies | PENDING | Waiting for training |
| 7 | 2026-05-24 | Pattern analysis (1,380 days, 8 tickers) | Completed | 43% of lows at min 60-90; volume surge d=0.88; 9:30 is WRONG time |
| 8 | 2026-05-24 | Pattern-based entry model training | **COMPLETED** | AUC=0.890, 14 tickers 0.877-0.899, top feature=drop_from_open |
| 9 | 2026-05-24 | Pattern model backtest (end-to-end) | **COMPLETED** | t=0.80 best: +$164K, 453 trades, 62.9% WR, PF=1.49, Sharpe=2.41, avg entry min 46 |
| 10 | TBD | Walk-forward validation | NOT STARTED | Train 2023-2024, test 2025-2026 |
| 11 | TBD | Exit timing model backtest | NOT STARTED | Requires separate simulator |

### Combined Scoring Sweep Results (Experiment 1)

Full results in `journal/sweep_candidates.json` and `memory/sweep_results.md`.

**Top configs by P&L:**

| Config | Trades | WR% | P&L | PF |
|---|---|---|---|---|
| 0.0T+1.0ML, t50, wide | 355 | 69.6% | $3.63M* | 1.59 |
| 0.8T+0.2ML, t50, wide | 339 | 64.3% | $132K** | 1.83 |
| 0.8T+0.2ML, t55, moderate | 71 | 73.2% | varies | 3.76 |

*Compound. **Flat.

**Top configs by quality (Sharpe/PF):**

| Config | Trades | WR% | PF | Sharpe |
|---|---|---|---|---|
| 0.8T+0.2ML, t55, moderate, no_midday | 71 | 73.2% | 3.76 | 5.15 |
| 0.8T+0.2ML, t60, moderate, killzone_only | 52 | 75.0% | 4.02 | 5.43 |
| 0.6T+0.4ML, t55, wide, morning_only | 89 | 70.8% | 2.91 | 4.22 |

### Simpsons Veto Gates (Experiment 2)

| Gate | Block Rate | Effect on P&L | Decision |
|---|---|---|---|
| Afternoon (1:30-3PM) | 0% | Neutral | KEEP (safety) |
| ML warmup (<10 obs) | 82% | DESTROYS profit | DISABLED |
| Sweep required | 95% | Blocks everything | DISABLED |
| Wide spread >30% | 0% | None | KEEP (safety) |
| Momentum confirm | 20% | Blocks good reversals | DISABLED |

### V3 Isolation Tests (Experiment 5) — PENDING

*Will be filled by `scripts/evaluate_v3_models.py` output*

| Model | Config | Trades | Blocked | WR% | P&L | PF | Sharpe | Delta vs Baseline |
|---|---|---|---|---|---|---|---|---|
| entry_timing_t30 | - | - | - | - | - | - | - | - |
| entry_timing_t50 | - | - | - | - | - | - | - | - |
| entry_timing_t70 | - | - | - | - | - | - | - | - |
| regime_t30 | - | - | - | - | - | - | - | - |
| regime_t50 | - | - | - | - | - | - | - | - |
| regime_t70 | - | - | - | - | - | - | - | - |
| ticker_select_t30 | - | - | - | - | - | - | - | - |
| ticker_select_t50 | - | - | - | - | - | - | - | - |
| ticker_select_t70 | - | - | - | - | - | - | - | - |
| signal_quality_m20 | - | - | - | - | - | - | - | - |
| signal_quality_m30 | - | - | - | - | - | - | - | - |
| signal_quality_m50 | - | - | - | - | - | - | - | - |
| stop_calibrate | - | - | - | - | - | - | - | - |

### V3 Combo Strategies (Experiment 6) — PENDING

*Will be filled by `scripts/evaluate_v3_models.py` output*

| Combo | Trades | WR% | P&L | PF | Sharpe | Delta vs Baseline |
|---|---|---|---|---|---|---|
| regime + ticker_select | - | - | - | - | - | - |
| regime + entry_timing | - | - | - | - | - | - |
| ticker_select + signal_quality | - | - | - | - | - | - |
| regime + ticker_select + entry_timing | - | - | - | - | - | - |
| ALL MODELS | - | - | - | - | - | - |
| baseline + stop_calibrate | - | - | - | - | - | - |

---

## Pattern Analysis Results (2026-05-24)

### Finding 1: +30% moves happen EVERY day on EVERY ticker

SPY, QQQ, IWM: 100% of days have a +30% ATM move. Even TSLA/META: 99%.
The 30% threshold is useless as a filter — ATM 0DTE options are volatile enough
that a 30% swing is the norm, not the exception.

**Implication**: The V3 entry_timing model (trained on +38% moves) is learning
"when does the daily move START" not "does a move happen." The question needs
to be "where is the LOW" — because there's almost always a move AFTER it.

### Finding 2: The daily low is NOT at 9:30

| Low Minute | % of Days | Avg Gain from Low |
|---|---|---|
| 0-5 min | 7% | 359% |
| 5-15 min | 14% | 553% |
| 15-30 min | 13% | 316% |
| 30-60 min | 23% | 314% |
| 60-90 min | 43% | 310% |

**43% of daily lows occur 60-90 minutes after open.** Only 7% are in the first 5 minutes.
Our current system enters at 9:30 (minute 0) — it's buying BEFORE the dip, not at the bottom.

The best gains (553%) come from lows at minute 5-15, but those are only 14% of days.
The most COMMON low is 60-90 min (43% of days) — this is the pullback after the opening rush.

**This directly contradicts our 9:30 entry bias.** We should be WAITING for the dip.

### Finding 3: What the pattern looks like before the low

**GREAT entries (+50% gain) vs POOR entries (<20% gain):**

| Feature | GREAT | POOR | Separability |
|---|---|---|---|
| Premium slope (5min) | -14.1% | -6.4% | MODERATE (d=0.70) |
| Volume surge | 553 avg | 74 avg | **STRONG (d=0.88)** |
| Low minute (from open) | 43 median | 79 median | **STRONG (d=1.48)** |
| IV change (5min) | +0.039 | +0.004 | WEAK (d=0.31) |
| Drop to reach low | -44% | -35% | WEAK (d=0.42) |
| Premium stabilizing | 16.1% | 14.2% | NONE |
| Underlying slope | -0.18% | -0.18% | NONE |

**The two strongest predictors of a GREAT entry:**

1. **Volume surge (d=0.88)** — option volume spikes 7.5x before great entries vs poor.
   This is institutional flow. Someone is loading up at the bottom.

2. **Earlier low minute (d=1.48)** — great entries happen at minute 43 median,
   poor entries at minute 79. Earlier dips = more time for the move to develop.

**The pattern before a great entry:**
- Premium is FALLING sharply (-14% over 5 candles)
- Volume is SURGING (553 avg contracts)
- IV is EXPANDING (+0.039)
- The drop is DEEP (-44% from open)

In plain English: **"Premium crashes hard with heavy volume while IV expands =
someone is buying the fear dip."**

### Finding 4: The 9:30 bias IS a problem

Our current system:
- Enters at 9:30 (minute 0)
- Has 1 premium observation (no pattern to detect)
- Can't see if premium is falling, stabilizing, or surging
- Can't detect volume patterns
- Is often buying BEFORE the dip, not at the bottom

**What we SHOULD do:**
1. Scan continuously from 9:30-11:00 (90 min window)
2. At each minute, compute the 5-candle premium slope, volume, IV change
3. Look for the "crash + volume surge + IV expansion" pattern
4. WAIT until we see it, THEN enter
5. If no pattern by 11:00, skip the day (it's chop)

### Finding 5: The dip-confirm gate was right (but too simple)

Our V6 `ENABLE_DIP_CONFIRM=true` gate (wait 60s for premium dip + support) was
pointing in the right direction. But 60 seconds is far too short — the real dip
takes 30-60 MINUTES on most days. And it was only checking for a small pullback,
not the full "crash + volume + IV" pattern.

---

## Revised Training Approach: Pattern-Based Models

Based on the pattern analysis, the V3 models need to be retrained with a
fundamentally different approach:

### Old approach (time-based)
"At 9:30, should I enter?" → YES/NO

### New approach (pattern-based)
"Given the last 5-10 candles of premium/volume/IV/underlying data,
am I looking at the setup for a profitable entry?" → YES/NO + magnitude

**Key change: The model runs CONTINUOUSLY, not once at 9:30.**
Every minute from 9:30-11:00, compute features from the trailing 5-10 candles.
The model fires when it detects the pattern, not at a fixed time.

### Pattern-Based Feature Set

| Feature | What it captures | From analysis |
|---|---|---|
| `prem_slope_5` | Premium trajectory (last 5 candles) | d=0.70 MODERATE |
| `prem_slope_10` | Premium trajectory (last 10 candles) | Extended window |
| `prem_accel` | Is decline slowing? (2nd derivative) | d=0.30 WEAK |
| `prem_stabilizing` | Last 3 candles range (% of price) | Consolidation |
| `volume_surge` | Average volume in last 5 candles | **d=0.88 STRONG** |
| `volume_ratio` | Current volume vs 20-candle average | Volume spike |
| `iv_change_5` | IV expansion/contraction (5 candles) | d=0.31 WEAK |
| `iv_level` | Absolute IV level | d=1.10 STRONG |
| `und_slope_5` | Underlying price trajectory | d=0.00 NONE |
| `drop_from_open` | How far premium dropped from day's high | d=0.42 WEAK |
| `minutes_since_open` | Time context | **d=1.48 STRONG** |
| `bid_ask_spread` | Liquidity indicator | d=0.96 STRONG |
| `delta` | Option moneyness | Structural |
| `theta` | Time decay rate | Structural |

### Pattern-Based Labels

Instead of "will this go up 38%", use:
- **Label = 1**: This candle is within 5% of the day's killzone low AND
  the subsequent move from here is >= 20% (FSM-realistic, not raw peak)
- **Label = 0**: Everything else

This means:
- NOT every day has a positive label (if the low happens at minute 90 and
  the move is only 15%, it's negative)
- Multiple candles near the low can be positive (the "entry window")
- The model learns the PATTERN, not the time

### Training Data Generation

For each ticker-day:
1. Load full day of 1-min option OHLC + quotes + greeks + stock OHLC
2. Find the killzone low (minutes 0-90)
3. Compute the gain from low to subsequent peak
4. If gain >= 20%: label candles within 5% of low as positive
5. For EVERY candle (positive and negative), compute the trailing-window features
6. This creates a balanced training set with clear pattern labels

### Continuous Scanning in Production

In production, this model runs every minute 9:30-11:00:
1. Get latest option quote (premium, bid, ask, IV, delta, volume)
2. Compute trailing 5-10 candle features
3. If model confidence > threshold → ENTER
4. If no signal by 11:00 → skip this ticker today

This is fundamentally different from "check once at 9:30" — it's
**continuous pattern recognition** with a 90-minute observation window.

---

## Open Questions

### Q1: Is the 9:30 entry the actual optimal time?

**Hypothesis**: Morning gap continuations (open → first 15 min) are the dominant edge. By 10:00+ the move is over.

**Test**: Walk-forward on different market regimes. If 9:30 entries fail in choppy months (Aug-Sep typically), the edge is regime-dependent.

**Counter-hypothesis**: 9:30 works because options have the most extrinsic value at open (full day of theta). As the day progresses, theta decay makes profitable exits harder.

### Q2: Should we scan more than once per day?

**Current state**: 1 scan per ticker per day (at 9:30).

**Alternative**: Scan every 5 minutes, 9:30-11:30 (24 scans per ticker). This would:
- Give the ML model more data points (premium_history, volume_history fill up)
- Allow entries when morning dip bottoms out (10:00-10:30)
- Match Neverland's pattern of spread-out entries

**Cost**: 14 tickers × 24 scans = 336 evaluations per day (vs 14 today). Phase 1 would take 24x longer.

**Decision**: DEFER until V3 results are in. If regime/ticker models add enough signal, multi-scan may not be needed.

### Q3: Do we need different models per market regime?

**Current**: One model for all conditions.

**Alternative**: Train separate models for trending vs choppy markets (identified by regime model). Use regime model to select which entry/exit model to apply.

**Decision**: DEFER. Test regime as a simple gate first.

### Q4: Is the 38% move threshold right?

The entry timing model labels +38% moves as "winners." But:
- Our V5 FSM often exits at +20% (breakeven ratchet) or +25% (scalp target)
- A +38% move may never be realized if FSM exits early
- Maybe we should label based on FSM exit P&L, not raw premium trajectory

**Test**: Compare entry_timing labels using 38% raw threshold vs 20% FSM-realistic threshold.

### Q5: How much does theta decay bias 0DTE entries?

At 9:30, a 0DTE ATM call has ~6.5 hours of extrinsic value. By 12:00, it has ~4 hours. Same underlying move = less premium gain at noon vs 9:30.

This creates a structural advantage for early entries that has nothing to do with ML prediction quality. The model might just be learning "early = more extrinsic = more profit."

---

## Decision Framework

For each V3 model, we classify it as:

| Classification | Criteria | Action |
|---|---|---|
| **DEPLOY** | Improves Sharpe by >0.5 AND doesn't reduce P&L by >10% | Integrate into production pipeline |
| **CONDITIONAL** | Improves one metric but hurts another | Test in paper trading for 2 weeks |
| **NEUTRAL** | No measurable impact (delta < 5%) | Do not deploy, but keep for combos |
| **HARMFUL** | Reduces Sharpe or P&L by >10% | Discard and analyze why |

The final production strategy will be the combo of DEPLOY + CONDITIONAL models that maximizes Sharpe on the 97-day backtest WHILE maintaining >50 trades (statistical significance).

---

## Pattern Entry Model — Backtest Results (Experiment 9)

170 trading days (Sep 2025 - May 2026), $23K compounding, 10 tickers, V5 FSM exits.

| Threshold | Trades | WR% | P&L | PF | Sharpe | MaxDD | Avg Entry Min |
|---|---|---|---|---|---|---|---|
| 0.30 | 425 | 51.8% | -$22,570 | 0.75 | -1.49 | 99.6% | 15 |
| 0.50 | 500 | 53.0% | -$22,381 | 0.79 | -1.60 | 97.9% | 23 |
| 0.60 | 498 | 58.4% | -$7,048 | 0.95 | -0.33 | 88.2% | 28 |
| 0.70 | 485 | 59.2% | +$4,370 | 1.02 | 0.13 | 65.1% | 33 |
| 0.75 | 478 | 59.4% | +$21,104 | 1.07 | 0.49 | 70.0% | 38 |
| **0.80** | **453** | **62.9%** | **+$163,817** | **1.49** | **2.41** | 64.2% | **46** |
| **0.85** | **397** | **67.3%** | **+$160,952** | **1.45** | **1.88** | **56.4%** | 56 |
| 0.90 | 219 | 68.5% | +$24,921 | 1.28 | 1.80 | 41.4% | 72 |

**Key finding:** The model DOES learn to wait for the dip. As threshold increases,
avg entry minute moves from 15 → 72 and win rate from 52% → 69%. The sweet spot
is t=0.80 (best Sharpe) or t=0.85 (best quality, lowest MaxDD).

**vs Baseline comparison:**
- Current production (Discord signals, 9:30 entry): $21,650 over 188 trades
- Pattern model t=0.80 (continuous scan, dip entry): $163,817 over 453 trades
- Pattern model uses NO Discord signals — fully autonomous scanning

**Caveats:**
1. Compounding inflates results (0.80 went to $183K but had 64% drawdown)
2. Only tests ATM calls — no PUT direction
3. No regime/ticker selection filter yet — adding those should reduce drawdown
4. 170-day test period may not capture all market regimes

---

## File Inventory

| File | Purpose |
|---|---|
| `scripts/train_ml_models_v3.py` | Train all 6 V3 models |
| `scripts/evaluate_v3_models.py` | Test models in isolation + combos, generate reports |
| `scripts/watch_and_evaluate_v3.sh` | Auto-run evaluation when training completes |
| `scripts/sweep_combined_scoring.py` | Combined scoring parameter sweep |
| `journal/sweep_candidates.json` | Pre-computed candidate trades (1,285 candidates) |
| `journal/models/ml_v3/` | Trained V3 model files + metadata |
| `journal/v3_eval_results/` | Evaluation reports + raw JSON results |
| `specs/v3-ml-evaluation-spec.md` | This file — comprehensive tracking |
| `scripts/train_pattern_entry.py` | Train pattern-based continuous-scan entry model |
| `scripts/backtest_pattern_entry.py` | End-to-end backtest with V5 FSM exits |
| `specs/lessons-learned-and-roadmap.md` | Previous lessons and roadmap |
| `memory/sweep_results.md` | Detailed sweep result tables |
