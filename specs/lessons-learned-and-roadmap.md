# OptionsOwl: Lessons Learned & Roadmap Spec

## Goal

Build a fully autonomous ML-powered options sourcing + buy/sell agent that:
- Replaces Discord (Neverland) signals entirely with self-generated ML signals
- Produces **consistent daily profit** ($500-$1,500/day target on $23K portfolio)
- Runs end-to-end: ThetaData -> Sourcing Scanner -> Scoring -> Entry Pipeline -> Webull Execution -> V5 FSM Exit -> Postgres/Redis state -> Supabase analytics
- Is fully tested with realistic backtests (no compounding inflation, no outlier reliance)

---

## Part 1: What We Learned (Hard Data)

### 1.1 The ML-Only System Is NOT Profitable

We ran a 97-day sweep (Jan 2 - May 21, 2026) across 1,297 ML-generated candidates, 14 tickers, 4 stop configs, 5 warmup delays, 8 TOD rules, and 9 scoring weight combos (21,600 total parameter combos).

**Raw results (flat $23K sizing, no compounding):**

| Stop Config | Trades | WR% | Total P&L | PF | Avg Loss |
|---|---|---|---|---|---|
| ultra_tight (10/20) | 339 | 47.5% | $89,727 | 1.59 | -$862 |
| tight (15/30) | 339 | 54.0% | $95,684 | 1.58 | -$1,054 |
| moderate (25/45) | 339 | 60.8% | $118,969 | 1.73 | -$1,226 |
| wide (35/65) | 339 | 64.3% | $132,334 | 1.83 | -$1,318 |

These numbers look good BUT they are inflated by scoring filter (threshold >= 50) selecting only 339 of 1,088 candidates. The unfiltered system loses money across ALL stop configs.

**Without the scoring filter, ALL 1,088 ML signals are net negative:**
- tight: -$85,686
- wide: -$80,702

The scoring filter is doing the heavy lifting, not the ML model.

### 1.2 Real Neverland Trades vs ML Signals — Key Differences

| Metric | Neverland (real Webull) | ML Sweep |
|---|---|---|
| Total trades | 131 | 1,297 |
| Win rate | 57.3% | 51% (unfiltered) |
| Total P&L | +$683 | -$85K (unfiltered) |
| Trades/day | 1.4 | 13.4 |
| Entry window | 10:00-13:30 ET (spread out) | 96% at 9:30 ET |
| Best exit | soft_trail (73% WR, +$3,750) | soft_trail (83% WR, +$11K) |
| Worst exit | checkpoint_cut (0% WR, -$4,531) | confirmed_stop (0% WR, -$27K) |
| Premium sweet spot | $2-3 (78% WR) | Not filtered |

**Critical insight: Neverland was SELECTIVE.** 1.4 trades/day, spread across 10:00-13:30 ET, with human judgment filtering bad setups. ML fires 13+ signals at 9:30 AM with 1 premium observation and no discretion.

### 1.3 Entry Timing Problem

**ML at 9:30 AM is flying blind on options data:**
- 1,088 of 1,297 candidates (84%) enter at minute 0 (9:30 AM)
- ALL minute-0 entries have premium_hist_len = 1 (single observation)
- ML features like `premium_volatility`, `premium_momentum` are zero
- The model has delta/theta/vega from the first snapshot, but no price action

**Neverland entries were spread across the day:**
- 10:xx ET: 40 trades (best: +$2,597, 57% WR)
- 11:xx ET: 38 trades (-$2,107, 53% WR)
- 12:xx ET: 19 trades (-$270, 74% WR)
- 13:xx ET: 28 trades (+$563, 54% WR)

The human callers waited for setups to develop. They didn't fire at the opening bell with zero data.

### 1.4 Stop Loss Paradox

User hypothesis: "If entries are precise, cut losses fast."
Data says: Wide stops (35/65) produce +$37K more than tight (15/30).

**But this is a symptom, not a cause.** The entries are NOT precise — that's why wider stops help. Wider stops give mediocre entries room to recover. If entries WERE precise, tight stops would work because the trade moves in your favor immediately.

**Exit reason breakdown reveals the real problem:**

| Exit Reason | Tight Stop Trades | Tight P&L | Wide Stop Trades | Wide P&L |
|---|---|---|---|---|
| hard_stop | 329 | -$24,348 | 75 | -$7,072 |
| confirmed_stop | 202 | -$26,806 | 116 | -$23,821 |
| checkpoint_cut | 36 | -$3,822 | 197 | -$24,182 |
| theta_timer | 10 | -$928 | 53 | -$6,325 |

With tight stops, `hard_stop` fires 329 times (-$24K). With wide stops, only 75 fire (-$7K) — but checkpoint_cut catches the rest later (-$24K). **The money lost is similar, just from different gates.** Wide stops let more trades recover from initial dips but trades that don't recover get cut by checkpoint instead.

**The real fix is not wider stops — it's better entries that don't need to recover.**

### 1.5 Per-Ticker Reality

**ML sweep (all candidates, tight stops):**
- Only MSTR is net positive (+$753)
- Worst: PLTR (-$3,358), AVGO (-$3,004), GOOGL (-$2,967), MSFT (-$2,217)

**Real Neverland trades:**
- Best: SPY (+$1,960), TSLA (+$1,401), AAPL (+$1,096), QQQ (+$730)
- Worst: AVGO (-$2,514), MSFT (-$1,925), NVDA (-$693)

**AVGO and MSFT are losers in both systems.** They should be excluded or given much tighter filters.

### 1.6 ML Confidence Does NOT Predict Outcomes

| ML Confidence | Count | WR% | Avg P&L |
|---|---|---|---|
| 0.4-0.5 | 29 | 44.8% | -$44 |
| 0.5-0.6 | 39 | 51.3% | -$20 |
| 0.6-0.7 | 183 | 45.4% | -$24 |
| 0.7-0.8 | 626 | 53.7% | -$8 |
| 0.8-0.9 | 303 | 47.5% | -$15 |
| 0.9+ | 117 | 52.1% | -$29 |

The 0.7-0.8 bin is the "least bad" but still negative. Above 0.8, performance DEGRADES — the model is overconfident on setups that mean-revert. ML confidence should be ONE input, not the primary filter.

### 1.7 DCA and Tick Data

**Yes, we have full tick data.** ThetaData has 1-min option bars (24.6M rows, 14 tickers, Jan 2023 - May 2026). The production `backtest_ml_e2e.py` runs full FSM with DCA support. The sweep script runs FSM with 4 stop configs per candidate — each gets full tick-by-tick simulation through the V5 exit engine.

**DCA was NOT tested in the sweep** because phase 1 pre-computes per-contract P&L. DCA requires tracking portfolio state across trades (which phase 2 does for sizing but not for mid-trade re-entry). To test DCA properly, we'd need to run the full `backtest_ml_e2e.py` pipeline per configuration.

### 1.8 Session Consistency (Daily P&L)

Using wide stops, scoring filter (threshold >= 50):
- 97 trading days, only **36 positive** (37.1%)
- Median daily P&L: **-$185**
- Best day: +$960, Worst day: -$1,634

**The system is not consistently profitable.** Nearly 2 in 3 days lose money. This is the opposite of the Neverland experience ($500-$1,500/day).

---

## Part 2: What Made Neverland & Simpsons Work

### 2.1 Neverland: Human Curation + Fixed Structure

1. **Selective entry** — 1.4 trades/day, not 13+
2. **Spread across the day** — entries from 10:00-13:30 ET, not all at 9:30
3. **Human judgment** — callers waited for setups to develop, skipped choppy periods
4. **ATM premium selection** — careful strike/expiry matching
5. **Multi-timeframe validation** — signals only when multiple TFs aligned
6. **The exit engine worked** — soft_trail (73% WR, +$3,750) was the primary profitable exit

### 2.2 Simpsons: Aggressive Filtering (85.9% WR)

The v10 "Surgical Filter" turned 39.4% baseline WR into 85.9% by applying 14 combinatorial rules that **removed 84.5% of losers without cutting any winners**:

1. Skip signals before 9:35 AM (pre-market noise)
2. Skip 12:00-1:30 PM dead zone
3. Require sweep of institutional level (PDH/PDL/PWH/PWL)
4. iFVG gap >= 0.5x ATR
5. Daily bias alignment required
6. Skip if ATR < 1.0 (low vol = chop)
7. Volume spike >= 1.5x on sweep candle
8. Spread <= 30% of premium
9. Min score threshold after ML adjustment
10. Skip after 3+ losses in last 2 hours (regime detection)
11. NY session only
12. Max 3 concurrent positions
13. 15-min cooldown on same ticker
14. Require positive underlying momentum on 5m chart

**Key insight: "Their biggest edge is filtering, not signal generation."**

### 2.3 Session Timing (from Simpsons data, 459 callouts)

| Session | Time | Win Rate | Action |
|---|---|---|---|
| NY Open Killzone | 9:30-10:30 | 72% | TRADE |
| Late Morning | 10:30-12:00 | 63% | TRADE |
| Midday | 12:00-1:30 | 55% | NEUTRAL |
| Early Afternoon | 1:30-3:00 | **36%** | **BLOCK** |
| Power Hour | 3:00-4:00 | 58% | TRADE |

**The 9:30-10:30 vs 1:30-3:00 spread is 100% WR difference.** But our ML fires almost entirely at 9:30 before the data develops. The answer isn't "only trade killzone" — it's "wait for the setup to develop within the killzone."

---

## Part 3: What's Wrong With Our Current Approach

### 3.1 ML Fires Too Early (Blind Entries)

The ML model checks every 5 minutes starting at 9:30. At the first check, it has 1 premium observation. Features like premium_volatility and premium_momentum are meaningless. It fires on delta/theta/vega alone — which is basically a coin flip on direction.

**Fix: Minimum warmup period.** Don't let ML fire until it has >= 10 premium observations (roughly 10-15 minutes of data). OR weight ML confidence lower in the first 15 minutes and let tech score dominate.

### 3.2 Too Many Signals (13+ per day vs 1-4)

The system generates 13+ candidates per day. Even with MAX_CONCURRENT=4 and scoring filters, it's entering far too many trades. Neverland averaged 1.4/day with better results.

**Fix: Tighten filters aggressively.** The Simpsons approach — 14 veto rules that cut 84.5% of signals — is the model to follow.

### 3.3 No Institutional Flow Detection

The tech score uses standard indicators (EMA, MACD, RSI, VWAP, BB). These detect momentum but not WHY the momentum exists. Simpsons detected **institutional sweeps** — price taking out PDH/PDL/PWH/PWL levels, indicating smart money is moving.

Our scoring already computes sweep levels (pdh, pdl, pwh, pwl in indicators) but they're worth only +5 points in Tier 3. They should be **required filters**, not bonuses.

### 3.4 Filters Are Too Gentle

Current system uses score adjustments (-3 to -5 points) where it should use **hard vetos**:
- Afternoon penalty: -5 pts (should be BLOCK after 1:30 PM)
- Losing streak: -3 pts (should be STOP after 2+ consecutive losses)
- Low volume: minor adjustment (should be BLOCK if < 1.5x on sweep candle)
- Wide spread: -2 to -4 pts (should be BLOCK if > 30%)

### 3.5 No Regime Detection

When the market is choppy (low ADX, range-bound), every signal is a coin flip. We have no mechanism to detect "this is a bad regime, stop trading" beyond the weak -3 pt losing streak penalty.

### 3.6 Missing Exit Intelligence

The exit engine (V5 FSM) is actually good — soft_trail, adaptive_trail, and scalp_trail are consistently profitable exits. The problem is **what goes IN, not how it comes OUT.**

`confirmed_stop` and `hard_stop` together account for -$51K on tight stops. These are trades that immediately went against us and never recovered. Better entry filtering would eliminate most of these.

---

## Part 4: The Path Forward

### 4.1 Architecture: Full End-to-End System

```
ThetaData (1-min options bars)
  -> Harvester (captures to Postgres)
  -> Sourcing Scanner (3-min intervals, market hours)
    -> Indicator Engine (multi-TF: 5m, 15m, 1h stock candles)
    -> Institutional Sweep Detector (PDH/PDL/PWH/PWL sweep + iFVG)
    -> 5-Tier Scoring (direction + timing + amplifiers + risk + calibration)
    -> ML Signal Model (LightGBM confidence, needs >= 10 premium observations)
    -> Combined Score = tech_weight × tech + ml_weight × ml_confidence
    -> Veto Gates (14 Simpsons-style rules)
    -> Quality Gate + Penalty Veto
  -> Entry Pipeline (18 gates, premium cap, spread gate, GFV, circuit breaker)
  -> Dip-Confirm Gate (wait for premium stabilization)
  -> Paper Trader (Postgres trade records)
  -> Webull Executor (live orders)
  -> Position Monitor (5s loop)
    -> V5 FSM Exit Engine (10 gates, DCA, scaleout)
  -> Redis (real-time state: open positions, daily P&L, regime status)
  -> Postgres (trade history, ML signals, agent state)
  -> Supabase (cross-agent analytics, fill matching)
```

### 4.2 Postgres Integration (Currently Configured, Not Active)

Status: PostgreSQL 16 is in docker-compose.yml with `ENABLE_POSTGRES=false`. Schema exists in `options_owl/db/postgres.py`. Signal consumer exists in `options_owl/collectors/signal_consumer.py`.

**Next steps:**
1. Set `ENABLE_POSTGRES=true` in docker-compose per-bot overrides
2. Rebuild and verify schema auto-creates
3. Migrate trade writes to dual-write (SQLite + Postgres) for transition period
4. Add ML signal publishing to `ml_signals` table
5. Signal consumer polls `ml_signals` for cross-agent signal sharing

### 4.3 Redis Integration (Not Yet Built)

**Purpose:** Real-time state that multiple agents need to read/write fast:
- Current open positions per agent
- Daily realized P&L per agent (for circuit breaker)
- Regime status (choppy/trending/unknown)
- Cooldown timers (ticker + direction)
- Losing streak counter (last N trades)

**Design:**
- Redis container in docker-compose (already has postgres, add redis)
- `options_owl/state/redis_state.py` — read/write state
- Fire-and-forget writes (never block trading on Redis)
- Fallback to in-memory state if Redis is down

### 4.4 Proposed Veto Gates (Simpsons-Inspired)

These replace soft score penalties with hard blocks:

```python
# In scanner.py, before emitting signal:

# Gate V1: Warmup — ML needs data to be useful
if minutes_since_open < 15 and ml_confidence_used:
    skip("ml_warmup: only 1 premium observation")

# Gate V2: Session veto — 36% WR zone
if 240 <= minutes_since_open <= 330:  # 1:30-3:00 PM ET
    skip("afternoon_danger_zone")

# Gate V3: Institutional sweep required
if not any([sweep_pdh, sweep_pdl, sweep_pwh, sweep_pwl]):
    skip("no_institutional_sweep")

# Gate V4: Volume confirmation on sweep
if sweep_detected and volume_ratio < 1.5:
    skip("sweep_no_volume_confirm")

# Gate V5: Spread veto (not just penalty)
if spread_pct > 30:
    skip("spread_too_wide")

# Gate V6: ATR floor — no chop
if atr < atr_20_percentile:  # bottom 20% of ATR range
    skip("low_atr_chop")

# Gate V7: Losing streak regime
if consecutive_losses_last_90min >= 2:
    skip("losing_streak_regime")

# Gate V8: Daily bias alignment
if not bias_aligns_with_direction:
    skip("against_daily_bias")

# Gate V9: Momentum confirmation on 5m
if not underlying_momentum_confirms:
    skip("no_momentum_confirm")

# Gate V10: Cooldown per ticker+direction
if same_ticker_same_direction_within_30min:
    skip("cooldown_active")
```

### 4.5 Two-Phase Entry: Tech First, ML Second

Instead of ML firing at 9:30 with zero data:

**Phase 1 (9:30-9:45): Tech-only scanning**
- Use multi-day candle indicators (EMA, MACD, RSI, VWAP)
- Identify direction and candidate tickers
- DO NOT trade yet — just build watchlist

**Phase 2 (9:45+): ML-enhanced entry**
- ML model now has 15+ premium observations
- Premium volatility, momentum, pattern features are populated
- Combined score: tech (0.6) × tech_score + ml (0.4) × ml_confidence
- Veto gates apply
- Only fire when combined score >= 55

This gives the ML model data to work with while still catching the killzone (9:30-10:30).

### 4.6 Smarter Loss Mitigation (Not Just Wider Stops)

Instead of debating stop widths, implement **adaptive early exits**:

1. **Direction confirmation within 3 minutes** — If underlying moves against entry direction by > 0.3% in first 3 min, the thesis is wrong. Exit immediately at small loss instead of waiting for -30% premium drop.

2. **Premium velocity check** — If premium drops > 5% in first 2 minutes, the entry was mistimed. Exit with -5% loss instead of riding to -30%.

3. **Bid disappearance detection** (already exists) — Keep the 30s bid-zero timeout.

4. **Partial exit on stall** — If premium hasn't moved +5% after 10 minutes, sell half. Reduces exposure on trades that go nowhere.

5. **Breakeven ratchet** (already exists) — Once +20%, floor = entry. Keep this.

The point: **Cut wrong trades in the first 2-3 minutes based on price action, not arbitrary % thresholds.** If the entry is right, the trade moves immediately. If it doesn't move, it's probably wrong.

### 4.7 Per-Ticker Strategy

Based on data from both real trades and ML sweep:

| Ticker | Action | Rationale |
|---|---|---|
| SPY | TRADE (index, high liquidity) | +$1,960 real, consistent |
| QQQ | TRADE (index, high liquidity) | +$730 real, 74% WR |
| TSLA | TRADE (high vol, momentum) | +$1,401 real, 79% WR |
| AAPL | TRADE (standard, moderate) | +$1,096 real, 60% WR |
| AMZN | TRADE (standard, moderate) | +$362 real, 60% WR |
| META | TRADE (high vol, be selective) | -$145 real but small sample |
| MSTR | TRADE (high vol, runners) | Only ML winner (+$753) |
| AMD | TRADE (high vol, selective) | Near breakeven both systems |
| PLTR | CAUTION (losers in ML) | +$667 real (3 trades) but -$3.4K ML |
| NVDA | CAUTION (losers in ML + real) | -$693 real, needs tighter filters |
| GOOGL | EXCLUDE or tight filter | -$165 real, -$3K ML, worst WR |
| MSFT | EXCLUDE or tight filter | -$1,925 real, -$2.2K ML |
| AVGO | EXCLUDE or tight filter | -$2,514 real, -$3K ML |
| IWM | CAUTION (small sample) | -$286 real, noisy |

**Reduce from 14 to 10 tickers initially.** Exclude GOOGL, MSFT, AVGO. Add them back only when per-ticker models prove profitable in backtest.

### 4.8 Consistent Daily Profit Target

To hit $500-$1,500/day on $23K:
- Need 2-4 trades/day (not 13+)
- Need 60%+ WR with PF > 1.5
- Average winner ~$400, average loser ~-$250
- This requires: 2 wins ($800) + 1 loss (-$250) = $550/day

**The math only works with high selectivity.** Taking 13 trades with 52% WR produces churn. Taking 3 trades with 65% WR produces consistent profit.

---

## Part 5: Immediate Action Items

### P0: Fix Entry Quality (Week 1)

1. **Add ML warmup gate** — Don't fire ML until >= 10 premium observations (~15 min after open)
2. **Add afternoon veto** — Block ALL signals 1:30-3:00 PM ET (hard veto, not -5 pts)
3. **Add losing streak veto** — Stop trading after 2 consecutive losses in 90 min
4. **Reduce ticker list** — Remove GOOGL, MSFT, AVGO from scanner watchlist
5. **Tighten spread veto** — Block if spread > 30% (currently just -2 to -4 pts penalty)

### P1: Add Institutional Flow Detection (Week 2-3)

6. **Require sweep of key level** — Signal must have swept PDH/PDL/PWH/PWL
7. **Volume confirmation on sweep** — Require 1.5x avg volume on sweep candle
8. **ATR floor** — Skip ticker if ATR is in bottom 20th percentile (chop)
9. **Direction confirmation** — Underlying must be moving in signal direction on 5m chart

### P2: Infrastructure (Week 2-3, parallel)

10. **Enable Postgres** — Set ENABLE_POSTGRES=true, verify schema, dual-write
11. **Add Redis** — Container in docker-compose, state module, regime tracking
12. **Backtest harness** — Build realistic backtest that tests veto gates, NO compounding, flat sizing

### P3: Adaptive Early Exits (Week 3-4)

13. **3-min direction check** — Exit if underlying moves > 0.3% against within 3 min
14. **Premium velocity exit** — Exit if premium drops > 5% in first 2 min
15. **Partial exit on stall** — Sell half if no movement after 10 min

### P4: Validate & Deploy (Week 4)

16. **Run per-ticker backtests** with all veto gates on ThetaData
17. **Compare daily P&L consistency** — target 60%+ winning days
18. **Deploy to paper trading** for 1 week validation
19. **Deploy to live** with reduced position sizes initially

---

## Part 6: Open Questions

1. **Is the ML model trained on the right labels?** It was trained using V5 FSM exits as labels. If the FSM itself has issues (like not catching early losers), the model learns to predict entries that feed into a flawed exit system. We may need to retrain with better exit labels (e.g., "did premium go up 15% at any point in the next 30 min?").

2. **Should ML run at market open at all?** Maybe tech-only scoring for the first 15 min is better, with ML kicking in after 9:45 when it has data. The 2-phase approach in 4.5.

3. **Is 14 tickers too many?** Real Neverland profits came from 5-6 tickers (SPY, TSLA, AAPL, QQQ, AMZN, MSTR). Spreading across 14 dilutes signal quality with tickers we can't predict.

4. **Do we need a fundamentally different ML model for open vs midday?** The feature importances are different at 9:30 (delta/theta/vega dominate) vs 10:30+ (premium patterns dominate). Two models might outperform one.

5. **Should we add options flow data (UW)?** GEX bias, dark pool levels, and sweep flow could replace the "human judgment" that Neverland had. The UW API is already integrated but not used in the scoring pipeline.

---

## Part 7: Success Criteria

| Metric | Current | Target |
|---|---|---|
| Daily win rate (% positive days) | 37% | 60%+ |
| Trades per day | 13+ | 2-4 |
| Trade win rate | 52% | 65%+ |
| Profit factor | 1.0 (breakeven) | 1.5+ |
| Daily P&L (median) | -$185 | +$500 |
| Max daily loss | -$1,634 | -$500 cap |
| Tickers traded | 14 | 10 (initially) |
| Average hold time | 29 min | 15-45 min |
| Entries at 9:30 with zero data | 84% | 0% |

---

*Created: 2026-05-23*
*Based on: 97-day sweep (21,600 combos), 131 real Webull trades, Simpsons v10 analysis*
