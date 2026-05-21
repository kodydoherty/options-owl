# Errata & Cross-Document Corrections

**Status:** Pre-implementation review findings
**Date:** 2026-05-21
**Scope:** Issues found across Docs 01-05 that must be resolved before coding begins

---

## Critical Issues (Must Fix Before Phase 1)

### 1. Indicator Parameter Conflicts

**Affected:** Doc 03 (Scoring Redesign) vs Doc 05 (Architecture Overview)

Doc 03 specifies:
- RSI(14) — standard period
- MACD(12, 26, 9) — standard parameters
- EMA crossover: 9/21

Doc 05 specifies:
- RSI(9) — non-standard period
- MACD(5, 13, 1) — non-standard parameters
- EMA crossover: 9/21

**Resolution:** Use the N8N source code as ground truth. The original workflow uses:
- RSI(9) for short-term momentum (matches N8N `Fetch & Score All Tickers` node)
- MACD(5, 13, 1) for fast crossover detection (matches N8N)
- EMA 9/21 crossover (both docs agree)

**Action:** Update Doc 03 Section 3.1 to use RSI(9) and MACD(5, 13, 1). Add a note that these are intentionally non-standard for 0DTE intraday trading where standard periods are too slow.

---

### 2. Signal Overlap — 55 Points from Redundant Momentum Signals

**Affected:** Doc 03 (Scoring Redesign)

The current N8N scoring has 5 different signals all measuring "price is going up":
1. RSI momentum (+15 pts)
2. MACD crossover (+15 pts)
3. EMA 9/21 crossover (+10 pts)
4. Price vs VWAP (+8 pts)
5. ADX trend strength (+7 pts)

Total: ~55 of ~170 possible points (32%) from correlated momentum indicators.

**Resolution:** Doc 03 Section 4.2 proposes collapsing to 3 orthogonal signals (35 pts max):
1. Momentum composite: best-of(RSI, MACD) — 15 pts max
2. Trend confirmation: EMA crossover — 10 pts max
3. Volume-price alignment: VWAP + ADX — 10 pts max

This is correct but needs explicit mapping from old signals to new composite scores, with a transition table showing how historical scores would change.

**Action:** Add Appendix A to Doc 03 with old-to-new score mapping table for all 220 historical trades.

---

### 3. Statistical Power Problem for A/B Testing

**Affected:** Doc 02 (Source A/B Testing Plan)

With 220 historical trades:
- Detectable effect size: ~10% win rate difference (not the 2% claimed)
- Required sample for 2% detection: ~2,400 trades
- At 10-15 trades/day, that's 160-240 trading days (8-12 months)

**Resolution:** Restructure the ablation study:
1. **Phase 1 (immediate):** Use 220 trades to detect large effects only (>8% WR delta). This eliminates clearly harmful sources.
2. **Phase 2 (ongoing):** Accumulate data over 3 months for medium effects (>5% WR delta).
3. **Phase 3 (long-term):** Only attempt 2% detection after 6+ months of data.

Use **Bayesian sequential testing** instead of fixed-sample frequentist tests — allows early stopping when evidence is strong.

**Action:** Update Doc 02 Section 4.1 hypothesis table with realistic detectable effect sizes. Add sequential testing protocol.

---

## High-Severity Issues

### 4. Missing Market Hours Gate Complexity

**Affected:** Doc 01 (Migration Plan)

The N8N `Market Hours Gate` node is 2,118 lines of JS — not a simple time check. It includes:
- Pre-market data prefetch (8:30-9:30 AM ET)
- Staggered ticker polling (avoid rate limits)
- Holiday calendar (NYSE closures + early closes)
- Circuit breaker pause (halt trading on extreme VIX)
- Retry logic with exponential backoff for failed API calls

Doc 01 Phase 2 treats this as a one-line `if market_open()` check.

**Action:** Add a dedicated subtask in Doc 01 Phase 2 for `MarketHoursManager` class covering all 5 responsibilities. Estimate 2-3 days, not 2 hours.

---

### 5. Missing Exit Monitor ML Tracking

**Affected:** Doc 01 (Migration Plan)

The N8N `Exit Monitor` node (1,095 lines) includes ML feature logging:
- Tracks 12 features at each exit decision point
- Logs "would have been profitable to hold" counterfactual
- Saves feature vectors for offline model training

Doc 01 Phase 5 mentions exit logic but not the ML tracking pipeline.

**Action:** Add ML feature logging to Doc 01 Phase 5. This feeds into future exit optimization (separate from entry scoring).

---

### 6. Candle Resolution Gap

**Affected:** Doc 04 (Backtest Plan)

Harvester DB stores 5-minute candles. The N8N workflow uses 1-minute candles for:
- VWAP calculation (intraday, needs tick-level or 1-min)
- RSI(9) on 1-min timeframe for entry timing
- Precise entry/exit timestamps

5-minute candles introduce up to 4:59 of timing error on entries and exits.

**Resolution:**
- VWAP: Approximate from 5-min candles using `typical_price × volume` per bar. Document expected error margin (~1-3%).
- RSI(9) 1-min: Cannot reconstruct. Use RSI(9) on 5-min as substitute. Document this limitation.
- Entry/exit timing: Use bar open/close as bounds, not point estimates. Report P&L as ranges, not precise values.

**Action:** Add Section 3.5 to Doc 04 documenting resolution limitations and error bounds for each affected indicator.

---

## Medium-Severity Issues

### 7. Tier Assignment Mismatch (Scoring)

**Affected:** Doc 03 vs Doc 05

Doc 03 Section 5 defines score tiers:
- Elite (90-100): Reduce position size (over-conviction)
- Strong (75-89): Full allocation
- Moderate (60-74): Half allocation
- Weak (40-59): Paper-only
- Reject (0-39): No trade

Doc 05 Section 4.2 defines different tiers:
- 85-100: Maximum confidence
- 70-84: High confidence
- 50-69: Moderate confidence
- Below 50: Low/no trade

**Resolution:** Use Doc 03's tiers as canonical (more detailed, has the "elite over-conviction" insight from backtesting). Update Doc 05 to reference Doc 03's tier table.

**Action:** Replace Doc 05 Section 4.2 tier table with a cross-reference to Doc 03 Section 5.

---

### 8. Missing Data Sources in Architecture

**Affected:** Doc 05 (Architecture Overview)

Doc 05 lists 4 data sources: Twelve Data, Polygon, Unusual Whales, Grok AI.

The N8N workflow also uses:
- **Yahoo Finance** (options chain fallback, already in OptionsOwl codebase)
- **TradingView webhooks** (alert-based signals for specific patterns)

**Action:** Add Yahoo Finance and TradingView to Doc 05 data source inventory. Note that TradingView webhooks are Phase 3+ (requires webhook receiver endpoint).

---

## Lessons Learned (Pre-Implementation)

1. **N8N is more complex than it looks.** The 30-node graph hides 10,000+ lines of JS. Two nodes alone (Market Hours Gate + Fetch & Score) account for 5,870 lines. Budget accordingly.

2. **Signal overlap is the root cause of coin-flip accuracy.** 32% of scoring points come from 5 correlated momentum signals. Fixing this alone could move win rate from 60% to 65-70%.

3. **220 trades is insufficient for fine-grained A/B testing.** Only large effects (>8% WR) are detectable. Plan for months of data collection before drawing conclusions about individual sources.

4. **Indicator parameters are intentionally non-standard.** RSI(9) and MACD(5,13,1) are tuned for intraday 0DTE trading. Don't "fix" them to textbook values.

5. **The harvester DB is an asset.** 7GB of 5-min candles + options snapshots enables backtesting without re-fetching. But 5-min resolution limits VWAP and short-timeframe indicator accuracy.

6. **Phase 1 parity is non-negotiable.** The new Python sourcing bot must produce identical signals to N8N before any scoring changes. A/B test the source, not the source + new scoring simultaneously.

7. **Technical analysis alone is a coin flip.** Every algo shop does the same TA faster. The edge comes from non-technical data: insider trades (SEC Form 4), Congress trades (STOCK Act), and contrarian retail sentiment (StockTwits). These are the alpha sources that most retail traders ignore.

8. **ML gates must be deployed incrementally.** Start with hand-tuned scoring (Phase 2). Add ML flow classifier (Phase 3.5). Only replace the entire scoring engine with ML Quality Predictor (Gate 3) after 300+ trades prove it beats hand-tuned via walk-forward validation.

9. **Free alpha sources exist.** SEC EDGAR (insider trades), StockTwits (retail sentiment), and Capitol Trades (Congress) are all free APIs. The UW subscription already includes Congress trade data. Zero additional cost for potentially the highest-edge signals.

10. **Breaking news during live trades is the biggest unaddressed risk.** The current system has no real-time news monitoring for open positions. A META layoff article can nuke a position before any exit gate fires. The exit engine needs a news-triggered emergency gate (see Doc 03 additions).

---

## Recommended Reading Order

For implementation:
1. Doc 05 (Architecture) — understand the system shape
2. Doc 01 (Migration) — follow the phases
3. Doc 03 (Scoring) — apply after Phase 1 parity achieved
4. Doc 02 (A/B Testing) — run alongside Phase 2-3
5. Doc 04 (Backtest) — validate changes before production
6. **This doc** — cross-reference corrections throughout
