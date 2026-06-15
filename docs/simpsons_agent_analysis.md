# Simpsons Agent Analysis — Yancy Adams' n8n Trading Bot

**Analyzed:** 2026-05-21
**Source:** n8n workflow JSON (37 nodes) + Combined Backtest Report (Side A + Side B)

## Overview

The "Simpsons" agent is an n8n-based 0DTE options callout system built by Yancy Adams. It runs as a set of webhooks and scheduled triggers that scan for institutional flow patterns, score them, and emit trade alerts. It's fundamentally different from our Discord-relay approach — it generates its own signals from market structure.

**Key stats:**
- Side A (Distribution Alerts): 387 callouts, 66.7% WR, 93.1% avg peak gain
- Side B (Daily Trends): 72 callouts, 58.8% WR, 69.0% avg peak gain
- Combined: 459 callouts analyzed

## Architecture (4 Sections)

### RED — Daily Recap (cron: 7:30 PM ET)
- Summarizes day's performance, calculates running stats
- Sends Discord recap embed

### BLUE — News/Trump Filter (cron: 9:00 AM ET)
- Pre-market news scan for market-moving events
- Trump policy/tariff filter — downgrades bias or skips trading on high-impact news days
- Sets a daily `news_sentiment` flag consumed by scoring

### BROWN — Side A: Distribution Alerts (Homer webhook)
- **This is the core signal engine**
- Triggered by Homer's webhook on each candle close
- Uses AMD+iFVG (Accumulation/Manipulation/Distribution + implied Fair Value Gap)
- Session-level scanning for institutional sweep patterns

### YELLOW — Side B: Daily Trends (Bart/Nelson/Maggie webhooks)
- Slower-timeframe trend following
- Multiple sub-agents each tracking different timeframes/patterns
- Feeds into same scoring/emission pipeline

## Side A Deep Dive — Key Algorithms

### Session Level Scanner (v11, ~1200 lines)

Scans for price sweeps of key institutional levels:

| Level | Description | Win Rate |
|---|---|---|
| PDH / PDL | Previous Day High/Low | 66% / 71% |
| PWH / PWL | Previous Week High/Low | 65% / **80%** |
| PMH / PML | Previous Month High/Low | 68% / 72% |
| NY High/Low | New York session high/low | **83%** / 70% |
| London High | London session high | 74% |

**AMD Pattern Detection:**
1. **Accumulation** — price consolidates in a range
2. **Manipulation** — price sweeps beyond the range (stop hunt)
3. **Distribution** — price reverses back through the range (the real move)

The scanner looks for price to sweep a key level (manipulation), then form an iFVG (implied Fair Value Gap) in the opposite direction — this is the entry signal.

**iFVG (implied Fair Value Gap):**
- Standard FVG = gap between candle 1 high and candle 3 low (bullish) or candle 1 low and candle 3 high (bearish)
- "Implied" = uses body-to-body gaps, not just wick-to-wick, catching subtler institutional footprints
- Minimum gap size: 0.3× ATR (filters noise)

### Signal Scorer (v6.4, ~370 lines)

Multi-factor scoring with these components:

| Factor | Weight | Notes |
|---|---|---|
| Session timing | High | NY Open Killzone (9:30-10:30) = best window |
| Sweep level importance | High | PWL/NY High get bonus points |
| Daily bias alignment | Medium | +5/+10 when trade aligns with macro bias |
| iFVG quality | Medium | Size relative to ATR, clean vs messy |
| Volume confirmation | Low | Spike on sweep candle |
| Multi-TF alignment | Low | Higher TF trend in same direction |

**Session Timing Win Rates:**
- NY Open Killzone (9:30-10:30): **72% WR**
- Late Morning (10:30-12:00): 63% WR
- Midday (12:00-1:30): 55% WR
- Early Afternoon (1:30-3:00): **36% WR** ← danger zone
- Power Hour (3:00-4:00): 58% WR

### ML Score Adjuster (v7, ~390 lines)

Uses Pinecone vector DB for historical pattern matching:
1. Embeds current signal features into a vector
2. Queries Pinecone for similar historical signals
3. Adjusts score based on historical win rate of similar setups
4. Per-ticker adjustment (e.g., TSLA AMD patterns behave differently than SPY)

This is essentially a nearest-neighbor feedback loop — signals that resemble past winners get boosted, past losers get penalized.

### v10 Surgical Filter (CRITICAL — 85.9% WR)

The v10 filter is the most impressive component. Starting from a 39.4% baseline WR, it applies 14 rules found via combinatorial feature search:

**Result:** 85.9% WR on filtered signals (removed 84.5% of losers with zero winner casualties)

Key filter rules identified:
1. Skip early-morning signals before 9:35 ET (pre-market noise)
2. Skip signals during 12:00-1:30 dead zone
3. Require sweep level to be PDH/PDL/PWH/PWL (not minor levels)
4. Require iFVG gap ≥ 0.5× ATR (strong institutional footprint)
5. Require daily bias alignment (no counter-trend trades)
6. Skip if ATR < 1.0 (low volatility = random chop)
7. Require volume spike on sweep candle (≥ 1.5× avg)
8. Skip if spread > 30% of premium (liquidity filter)
9. Require minimum score threshold after ML adjustment
10. Skip if 3+ losing signals in last 2 hours (regime detection)
11. Require NY session (no pre-market or after-hours)
12. Maximum 3 concurrent positions
13. Skip duplicate ticker within 15 minutes (cooldown)
14. Require positive underlying momentum on 5m chart

## Side B Deep Dive — Daily Trends

Side B is simpler — trend-following on 15m/1h timeframes:
- **Bart** tracks EMA cross + MACD divergence
- **Nelson** tracks VWAP reclaims/rejections
- **Maggie** tracks volume profile (POC, VAH, VAL)

Per-ticker TP targets calibrated from historical data:
- SPY: 25% TP / 15% stop
- TSLA: 40% TP / 25% stop
- NVDA: 35% TP / 20% stop

## What's Worth Adapting to OptionsOwl

### HIGH VALUE — Implement These

1. **Session Timing as Scoring Factor**
   - Their data clearly shows 72% WR during NY Open Killzone vs 36% early afternoon
   - We should add a time-of-day bonus/penalty to our sourcing scorer
   - Easy to implement: add `session_timing_score()` to `scoring/engine.py`
   - Penalty multiplier after 1:30 PM ET would filter our worst trades

2. **v10 Surgical Filter Approach (Combinatorial Feature Search)**
   - Their process: enumerate all possible filter rule combinations, test each against historical data, keep rules that cut losers without cutting winners
   - We have the backtest data to do this — run combinatorial search on our 188+ historical trades
   - Could be a new `scripts/optimize_filters.py` that brute-forces filter thresholds

3. **Sweep Level Detection (PDH/PDL/PWH/PWL)**
   - Add previous day/week high-low as features in our indicator engine
   - These are trivial to compute from our existing candle data
   - When price sweeps beyond PDH then reverses → strong signal for puts (and vice versa)
   - Add to `sourcing/data/indicator_engine.py`

4. **Losing Streak Regime Detection**
   - Their rule #10: skip if 3+ losses in last 2 hours
   - We already have circuit breaker but it's daily. A shorter-window "regime detector" could avoid bad market conditions
   - Simple to add to cooldown_manager or as a new filter

5. **Session-Based Position Limits**
   - Their 3 concurrent max is session-aware — resets each day
   - Our MAX_CONCURRENT is static. Adding time-of-day awareness would help

### MEDIUM VALUE — Worth Investigating

6. **AMD+iFVG Pattern Detection**
   - This is their core edge — institutional flow detection
   - Significantly more complex to implement (needs multi-candle pattern recognition)
   - Could be added as an optional ML feature rather than a full scanner
   - Start with FVG detection in `indicator_engine.py`, use as a scoring bonus

7. **Pinecone/Vector DB Score Adjustment**
   - Historical pattern matching via embeddings is clever
   - We could implement something similar with our existing LightGBM features
   - Simpler approach: k-nearest-neighbor lookup in our trade history table
   - Lower priority — our ML model already does feature-based prediction

8. **Per-Ticker TP Calibration (Side B)**
   - Their per-ticker targets are data-driven
   - We already have per-ticker V5 configs, but our TP targets are percentage-based
   - Could calibrate from our backtest data

### LOW VALUE — Nice to Know But Skip

9. **News/Trump Filter** — Too manual, requires NLP pipeline we don't have
10. **n8n Workflow Architecture** — We're already async Python, no benefit to switching
11. **Daily Recap Bot** — Nice for Discord but doesn't improve trading
12. **iFVG specifically** — Complex candle math, marginal benefit over our existing indicators

## Key Takeaways

1. **Their biggest edge is filtering, not signal generation.** The v10 filter turned a 39.4% WR into 85.9% by aggressively cutting losers. We should focus more energy on post-scoring filters.

2. **Time of day matters enormously.** 72% vs 36% WR based solely on session timing. This is free alpha we're leaving on the table.

3. **Institutional levels (PDH/PDL/PWH/PWL) are strong features.** 80% WR on PWL sweeps. These are trivial to compute and should be added to our indicator engine.

4. **Combinatorial filter optimization is a proven approach.** Rather than hand-tuning thresholds, systematically search the space. We have the data for this.

5. **Their AMD+iFVG is fundamentally different from our signal source.** It's self-generated from market structure, not relying on Discord bots. Long-term, this is the direction we should move — our sourcing scanner is already heading there.

## Comparison: Simpsons vs OptionsOwl

| Aspect | Simpsons | OptionsOwl |
|---|---|---|
| Signal Source | Self-generated (AMD+iFVG) | Discord (Neverland) + ML sourcing |
| Scoring | Multi-factor + ML vector adjustment | Multi-factor + LightGBM |
| Entry | Fixed targets from score | Smart entry + dip confirm + 18-gate pipeline |
| Exit | Fixed TP/stop per ticker | V5 FSM with 10 gates, DTE-aware, category-aware |
| Position Sizing | Fixed contract counts | Dollar-target with portfolio % caps |
| Risk Management | 3 concurrent max, session cooldown | Circuit breaker, graduated stops, DCA |
| Filtering | v10 surgical (14 rules, 85.9% WR) | Quality gate + penalty veto + ML veto |
| Backtest | 66.7% WR (Side A), 93.1% avg peak | 60.1% WR, $21,650 over 188 trades |
| Execution | Callouts only (manual trading) | Full auto (Webull API) |

**Their strength:** Signal generation + aggressive filtering
**Our strength:** Execution automation + risk management + exit engine

The ideal system combines both: their signal quality with our execution pipeline.
