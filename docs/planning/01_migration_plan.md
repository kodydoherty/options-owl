# Migration Plan: N8N 0DTE Scanner to owlet-sourcing

**Status:** Planning
**Created:** 2026-05-21
**Original workflow:** `docs/original/n8n_0dte_scanner.json` (30 nodes, ~10,000 lines JS)

---

## 1. Executive Summary

### Why We Are Migrating

The 0DTE Main Scanner currently runs as an N8N Cloud workflow -- a 30-node, ~10,000-line JavaScript pipeline that fires every 3 minutes during market hours. It fetches data from 5 external APIs, scores 13 tickers across 21+ signal dimensions, filters through multiple quality gates, and publishes trade alerts to Discord. The trading owlets (owlet-kody, owlet-adam, etc.) consume these Discord alerts as their signal source.

This architecture has served us well but has hit its limits:

1. **No testability.** The scoring logic lives in N8N code nodes -- giant JS blobs with no unit tests, no type safety, and no way to run regression tests. Bugs are discovered in production.
2. **No version control.** Changes to N8N nodes are not diffable. The only "backup" is exporting the full JSON. There is no PR review, no blame history, no rollback beyond restoring a prior JSON export.
3. **API cost overhead.** The N8N workflow makes ~7 Twelve Data API calls per ticker per scan (91 calls/scan for 13 tickers). Most of these fetch pre-computed indicators that we could compute locally from a single candle fetch, reducing to ~4 calls/ticker (52 calls/scan). Some indicators (BBands, RSI, MACD, Keltner) are already computed locally in the current workflow but the API calls for MFI, SuperTrend, and OBV remain.
4. **Discord as a bus.** Signals flow through Discord webhooks, then the trading owlets parse Discord messages back into structured data. This adds latency (~1-3s), introduces parsing fragility, and creates a single point of failure (Discord outage = no trading).
5. **No A/B testing.** We cannot run two scoring variants side-by-side. Every change is all-or-nothing.
6. **Operational fragility.** N8N Cloud has had outages. The workflow uses N8N's `$getWorkflowStaticData('global')` for state -- a black box that has lost state on platform updates.
7. **Duplication.** The trading owlets already have Polygon API access, candle data from the harvester DB, and their own entry pipeline with 18 gates. The N8N workflow duplicates some of this work.

### Current State vs Target State

| Dimension | Current (N8N) | Target (owlet-sourcing) |
|---|---|---|
| Runtime | N8N Cloud (managed) | Docker container on droplet (self-hosted) |
| Language | JavaScript (untyped, no LSP) | Python 3.12+ (typed, async, same stack as owlets) |
| Testing | None | Full unit/integration/E2E test suite |
| Version control | JSON export snapshots | Git, PR reviews, diffable |
| Signal delivery | Discord webhook -> Discord parse | Phase 1: Discord (drop-in). Phase 2: direct DB/channel |
| State management | N8N staticData (opaque) | SQLite + pydantic models (inspectable, backed up) |
| Indicator computation | Mix of API-fetched and local | All local (from candle data) |
| API calls per scan | ~91 Twelve Data + ~65 UW + ~13 Polygon news | ~13-26 Twelve Data (candles only) + UW/Polygon behind flags |
| Observability | N8N execution logs (retained 7 days) | Loguru (persisted, 90-day retention, grep-able) |
| Cost | N8N Cloud subscription + API plans | Droplet (already paid) + API plans |

---

## 2. Architecture: owlet-sourcing

### Container Design

`owlet-sourcing` is a new Docker container in `docker-compose.yml`, running alongside the existing owlets. It uses the same base image, Python version, and dependency set.

```yaml
# docker-compose.yml addition
owlet-sourcing:
  build: .
  container_name: owlet-sourcing
  command: python -m options_owl.sourcing.scanner
  restart: always
  env_file: .env
  environment:
    - SOURCING_SCAN_INTERVAL=180          # 3 minutes
    - SOURCING_DISCORD_OUTPUT=true        # Phase 1: write to Discord
    - SOURCING_DB_OUTPUT=true             # Phase 2: write to shared DB
    - ENABLE_UNUSUAL_WHALES=true
    - ENABLE_GROK_AI=false                # Start disabled, enable after validation
    - ENABLE_FINNHUB_EARNINGS=true
  volumes:
    - ./journal/owlet-sourcing:/app/journal/owlet-sourcing:rw
    - ./journal/owlet-harvester:/app/journal/owlet-harvester:rw   # read candle data
    - ./journal/models:/app/journal/models:ro
```

### Data Flow

```
Phase 1 (drop-in replacement):
  owlet-sourcing (3-min loop)
    -> fetch candles (Twelve Data or harvester DB)
    -> compute indicators locally
    -> score 13 tickers
    -> quality gate + penalty veto
    -> fetch options chain (Polygon)
    -> build alert
    -> POST to Discord webhooks (same format as N8N)
  Trading owlets read from Discord (unchanged)

Phase 2 (direct feed):
  owlet-sourcing (3-min loop)
    -> same pipeline
    -> write signals to shared SQLite DB (signals.db)
    -> optionally POST to Discord for human visibility
  Trading owlets read from signals.db instead of Discord
    -> eliminate Discord parsing
    -> structured data from the start
    -> sub-100ms latency
```

### Interaction with Existing Owlets

In Phase 1, nothing changes for the trading owlets. They continue reading from Discord. `owlet-sourcing` replaces N8N as the Discord webhook poster.

In Phase 2, `discord_collector.py` gains a new signal source: direct DB reads from `journal/owlet-sourcing/signals.db`. The signal parser is bypassed entirely -- data arrives as structured pydantic models.

### Scan Loop

The main loop mirrors N8N's 3-minute trigger:

```python
async def scan_loop():
    while True:
        if not is_market_open():
            await asyncio.sleep(60)
            continue

        scan_start = time.monotonic()

        # 1. Load/update state (cooldowns, budgets, regime)
        state = await state_manager.refresh(today)

        # 2. Fetch candle data for all tickers (parallel)
        candle_data = await fetch_all_candles(TICKERS)

        # 3. Score all tickers (parallel)
        scored = await asyncio.gather(*[
            score_ticker(ticker, candle_data[ticker], state)
            for ticker in TICKERS
        ])

        # 4. Quality gate filter
        passed = quality_gate.filter(scored, state)

        # 5. For each passing signal: options chain + refinement + output
        for signal in passed:
            chain = await fetch_options_chain(signal)
            refined = refine_chain(signal, chain)
            if penalty_veto.check(refined):
                continue
            await output.emit(refined)

        # 6. Update state (cooldowns, counters)
        await state_manager.post_scan_update(passed)

        # 7. Sleep remainder of 3-minute interval
        elapsed = time.monotonic() - scan_start
        await asyncio.sleep(max(0, 180 - elapsed))
```

---

## 3. Module Decomposition

### N8N Node to Python Module Mapping

| N8N Node | Lines | Python Module | Notes |
|---|---|---|---|
| 3-Min Market Hours Trigger | - | `scanner.py` (main loop) | `asyncio.sleep` replaces N8N scheduler |
| Market Hours Gate | 2119 | `scanner.py` + `utils/market_hours.py` | ET timezone, DST, weekend check. Most of the 2119 lines are pre-market brief, daily summary, pulse alerts -- these become separate output modules |
| Load State & Weekly Budget | 346 | `state.py` | SQLite-backed state instead of N8N staticData. Cooldowns, regime data, earnings calendar, prior levels |
| Fetch & Score All Tickers | 3753 | `indicators/*` + `scoring/*` + `sources/*` | The monster node. Decomposed below |
| Quality Gate Filter | 767 | `filters/quality_gate.py` | Circuit breaker, sector rotation detector, session context gate |
| Fetch 0DTE Options Chain | HTTP | `sources/polygon.py` | Polygon options snapshot API |
| Options Chain Refinement | 484 | `filters/options_chain.py` | Strike selection, Greeks, GEX/flow analysis, premium estimation |
| Grok AI Analysis | HTTP | `sources/grok.py` | xAI chat completions (behind feature flag) |
| Build Final Alert Data | 434 | `output/formatter.py` | Assembles alert payload, premium recovery |
| Penalty Veto Filter | 157 | `filters/penalty_veto.py` | Post-scoring critical penalty combo veto |
| ML v4 Inference | 199 | `scoring/ml_model.py` | Logistic regression P(win), advisory only |
| Build Webull Context | 188 | `output/formatter.py` | Bayesian signature lookup, historical WR |
| Format Entry Embed | 207 | `output/discord.py` | Discord rich embed formatting |
| Send Discord Entry Alert | HTTP | `output/discord.py` | Discord webhook POST |
| Update State Post-Alert | 308 | `state.py` | Cooldown/counter updates, trajectory tracking |
| Exit Monitor | 1096 | **NOT MIGRATED** | Owlets already have V5 FSM exit engine. N8N exit monitor is vestigial |
| Stand Down Check | 63 | `output/discord.py` | "No trades today" message |
| API Cost Monitor | 15 | `utils/api_cost.py` | Simple counter |
| Process Webull Close | 132 | **NOT MIGRATED** | Owlets handle their own Webull close |

### Proposed Directory Structure

```
options_owl/sourcing/
|-- __init__.py
|-- scanner.py              # Main 3-min scan loop, market hours gate
|-- state.py                # State management (cooldowns, budgets, regime, earnings)
|-- config.py               # Sourcing-specific settings (tickers, thresholds, feature flags)
|
|-- indicators/
|   |-- __init__.py
|   |-- ema.py              # calcEMA (9, 21, 200), crossover detection
|   |-- bollinger.py        # calcBBands (period=20, sd=2), double-touch, TTM squeeze
|   |-- rsi.py              # calcRSI (period=9), Wilder smoothing, divergence, curl
|   |-- macd.py             # calcMACD (5, 13, 4), histogram crossover
|   |-- keltner.py          # calcKeltner (period=20, multiplier=1.5)
|   |-- vwap.py             # Session VWAP from OHLCV, slope detection
|   |-- atr.py              # ATR expansion/contraction
|   |-- obv.py              # On-Balance Volume slope + divergence
|   |-- mfi.py              # Money Flow Index (period=14)
|   |-- supertrend.py       # SuperTrend confirmation
|   |-- candles.py          # Candlestick pattern detection (engulfing, pin bar, hammer, etc.)
|   `-- volume.py           # Volume spike detection, time-weighted normalization
|
|-- sources/
|   |-- __init__.py
|   |-- twelve_data.py      # Twelve Data REST API (candles + pre-computed indicators)
|   |-- polygon_data.py     # Polygon (news, options chain, daily aggs, SPY regime)
|   |-- unusual_whales.py   # UW (flow, dark pool, GEX, sector tide, 0DTE flow, max pain, Congress trades)
|   |-- grok.py             # xAI Grok chat completions
|   |-- finnhub.py          # Finnhub earnings calendar
|   |-- harvester.py        # Read candle data from shared harvester DB (WAL mode)
|   |-- sec_edgar.py        # SEC EDGAR Form 4 insider trades (officers, directors, 10%+ owners)
|   |-- capitol_trades.py   # Congress member stock trades (STOCK Act disclosures)
|   `-- stocktwits.py       # StockTwits retail sentiment (bullish/bearish ratio, contrarian signal)
|
|-- scoring/
|   |-- __init__.py
|   |-- signals.py          # Individual signal scorers (21+ signals, each a pure function)
|   |-- penalties.py        # Penalty functions (exhaustion, extended move, RSI curl, etc.)
|   |-- aggregator.py       # Combines raw signal points, applies penalties, normalizes 0-100
|   |-- fast_alert.py       # Fast Alert path (direction-independent, VWAP+1min based)
|   |-- gap_fade.py         # Gap-and-Fade detector + Gap-Down Reversal detector
|   |-- regime.py           # Market regime classification (BULL/BEAR EXTENSION/NORMAL/SIDEWAYS)
|   |-- ml_model.py         # ML v4 logistic regression P(win) (advisory, no veto)
|   `-- ml_gates/
|       |-- __init__.py
|       |-- flow_classifier.py    # ML Gate 1: Smart money flow classifier (LightGBM)
|       |-- entry_optimizer.py    # ML Gate 2: Entry timing optimizer (predict premium dip)
|       |-- quality_predictor.py  # ML Gate 3: Signal quality predictor (replaces hand-tuned score)
|       |-- regime_weighter.py    # ML Gate 4: Dynamic source weighting by market regime
|       `-- exit_advisor.py       # ML Gate 5: Exit timing advisory (survival analysis)
|
|-- filters/
|   |-- __init__.py
|   |-- quality_gate.py     # Circuit breaker, sector rotation, session context, FOMC
|   |-- penalty_veto.py     # Post-scoring critical penalty combo veto
|   |-- options_chain.py    # Options chain refinement, strike selection, Greeks, GEX
|   |-- cooldown.py         # Ticker cooldown (90min same dir, 30min opposite, cross-index)
|   |-- regime_gate.py      # Counter-trend hard blocks based on market regime
|   `-- news_sentinel.py    # Real-time news monitoring for open positions (exit gate integration)
|
|-- output/
|   |-- __init__.py
|   |-- discord.py          # Discord webhook output (bullish/bearish/reversal/standdown channels)
|   |-- db_writer.py        # SQLite signal writer (Phase 2)
|   `-- formatter.py        # Alert data assembly, Webull context, embed formatting
|
`-- utils/
    |-- __init__.py
    |-- market_hours.py     # ET timezone, DST calculation, market open check
    |-- api_cost.py         # API call counter + warning
    `-- helpers.py          # safeFloat, HTTP helpers, rate limiting
```

### Scoring Signals Inventory

The following signals are extracted from the N8N `Fetch & Score All Tickers` node (3753 lines). Each becomes a pure function in `scoring/signals.py` or its own module:

| # | Signal | Points | Direction | Source Data | Module |
|---|---|---|---|---|---|
| 1 | 9/21 EMA Crossover (5min) | 10 cross / 8 spread / 0 tight | Sets direction | 5min closes | `indicators/ema.py` |
| 2 | 200 EMA Macro Context | 5 (aligned) / 3 (near) | Bonus only | Daily EMA200 | `indicators/ema.py` |
| 3 | BB Double-Touch + TTM Squeeze | 10-13 | Confirms | 5min OHLC | `indicators/bollinger.py` |
| 4 | VWAP Confluence + Slope | 10-15 reclaim / 4-6 approach / +5 slope / -5 opposing slope | Confirms | 5min OHLCV | `indicators/vwap.py` |
| 5 | Multi-TF Alignment | 15 (3/3) / 3 (2/3) | Confirms | 1min + 5min + 15min | `scoring/signals.py` |
| 6 | RSI Extreme + Divergence | 5-10 | Confirms | RSI(9) on 5min | `indicators/rsi.py` |
| 7 | MFI (Money Flow Index) | +6 / -5 | Confirms/penalizes | MFI(14) on 5min | `indicators/mfi.py` |
| 8 | SuperTrend | +6 / -5 | Confirms/penalizes | SuperTrend on 5min | `indicators/supertrend.py` |
| 9 | OBV Volume Pressure | +10-15 / -12 divergence | Confirms/penalizes | OBV on 5min | `indicators/obv.py` |
| 10 | UW Net Premium Flow | 0 (confirm, zeroed) / -10 (diverge) | Penalizes only | Unusual Whales | `sources/unusual_whales.py` |
| 11 | UW Dark Pool | +10 / -4 | Confirms/penalizes | Unusual Whales | `sources/unusual_whales.py` |
| 12 | UW GEX (Gamma Exposure) | +18 (negative GEX) | Bonus | Unusual Whales | `sources/unusual_whales.py` |
| 13 | UW Sector Tide | +8 / -8 | Confirms/penalizes | Unusual Whales | `sources/unusual_whales.py` |
| 14 | UW 0DTE Flow | +5-12 / -8 | Confirms/penalizes | Unusual Whales | `sources/unusual_whales.py` |
| 15 | Max Pain Proximity | -5 | Penalty only | Unusual Whales | `sources/unusual_whales.py` |
| 16 | Trend Alignment (VWAP) | -8 | Penalty only | VWAP + direction | `scoring/penalties.py` |
| 17 | Fast Alert Path | 80+ (override) | Independent direction | 1min candles + VWAP + ORB | `scoring/fast_alert.py` |
| 18 | Gap-and-Fade | 82+ (override) | Fade direction | Day open + prev close | `scoring/gap_fade.py` |
| 19 | Gap-Down Reversal | 82+ (override) | Bullish override | Session low + volume | `scoring/gap_fade.py` |
| 20 | RSI Curl Direction | +3 / -5 / -15 | Confirms/penalizes | RSI delta | `indicators/rsi.py` |
| 21 | Move Exhaustion | -3 to -18 | Penalty | Move from open | `scoring/penalties.py` |
| 22 | Catalyst Sentinel Boost | +25-35 | Boost/override | Cross-workflow bridge | `scoring/signals.py` |
| 23 | MACD Crossover | 6 (zero-line cross) | Confirms | MACD(5,13,4) | `indicators/macd.py` |
| 24 | Candlestick Patterns | +3-10 / -3 | Confirms/penalizes | 5min OHLC | `indicators/candles.py` |
| 25 | RSI Extreme Reversal | +12 | Confluence bonus | RSI + candle/EMA | `scoring/signals.py` |
| 26 | Entry Timing (1min) | up to 10 | Confirms | 1min momentum | `scoring/signals.py` |
| 27 | 1-Min Volume Spike | varies | Confirms | 1min volume | `indicators/volume.py` |
| 28 | News Sentiment | +5 / -8 / VETO | Confirms/vetoes | Polygon news | `sources/polygon_data.py` |
| 29 | Volume Spike (mandatory) | 15 (>= 1.5x) | Gate + bonus | 5min volume | `indicators/volume.py` |
| 30 | ATR Expansion | +10 / -8 | Confirms/penalizes | ATR on 5min | `indicators/atr.py` |
| 31 | Volume Breakout Tier | +8 (>= 2.5x) | Bonus | 5min volume | `indicators/volume.py` |
| 32 | Multi-Candle Momentum | +5 / -6 | Confirms/penalizes | 5min closes | `scoring/signals.py` |
| 33 | Opening Range Breakout | +15 / -8 | Confirms/penalizes | First 15min range | `scoring/signals.py` |
| 34 | Key Level Detection | +5 | Bonus | S/R levels | `scoring/signals.py` |
| 35 | S/R Proximity Scoring | varies | Confirms/penalizes | Prior day H/L/pivots | `scoring/signals.py` |
| 36 | Relative Strength vs SPY | +10 / -5 | Confirms/penalizes | SPY change % | `scoring/signals.py` |
| 37 | Momentum Cluster Cap | 25 max | Cap | Correlated tickers | `scoring/aggregator.py` |
| 38 | RSI Direction Confirm | -8 | Penalty | RSI extremes | `scoring/penalties.py` |
| 39 | Time-of-Day Scoring | varies | Modifier | ET time | `scoring/signals.py` |
| 40 | Recent Streak Penalty | exponential decay | Penalty | Trade history | `scoring/penalties.py` |
| 41 | Exhaustion Chase Penalty | varies | Penalty | Move analysis | `scoring/penalties.py` |
| 42 | Extended Move Penalty | varies | Penalty | Move from open | `scoring/penalties.py` |
| 43 | Bearish Conviction Framework | varies | Modifier | Multiple signals | `scoring/signals.py` |
| 44 | SPX Power Fade | varies | Detects bearish PM pattern | SPY + volume | `scoring/signals.py` |
| 45 | Mid-Day Reversal | varies | Detects reversal | Price action | `scoring/signals.py` |
| 46 | Afternoon Momentum | varies | Detects PM trend | Price action + volume | `scoring/signals.py` |

### Additional Detectors (from scoring node lines 2000+)

| Detector | Lines | Description | Module |
|---|---|---|---|
| Cross-index opposite-dir block | 848-887 | Hard-skip if SPY/QQQ/IWM whipsaw within 30 min | `filters/cooldown.py` |
| Counter-trend regime block | 889-921 | Hard-skip bearish in BULL regime (and vice versa) | `filters/regime_gate.py` |
| Stage 2.7b skip gates | 2228-2250 | Shadow hard-skip gates being validated | `filters/quality_gate.py` |
| Shadow reversal veto | 2315-2480 | Resistance/support rejection (shadow mode) | `filters/quality_gate.py` |
| Intraday exhaustion reversal-flip | 2481-2589 | Shadow: flip direction on exhaustion | `scoring/gap_fade.py` |

---

## 4. Data Source Inventory

### API Calls Per Scan Cycle (Current N8N)

| Source | Endpoint | Data | Calls/Scan | Rate Limit | Cost Tier | Essential? | Migration Plan |
|---|---|---|---|---|---|---|---|
| **Twelve Data** | `time_series` (5min, outputsize=78) | OHLCV candles | 13 | 55 credits/min (Grow) | Included | YES | Keep -- primary candle source |
| **Twelve Data** | `time_series` (15min, outputsize=30) | Multi-TF candles | 13 | included | Included | YES | Keep |
| **Twelve Data** | `time_series` (1min, outputsize=20) | Entry timing | 13 | included | Included | YES | Keep |
| **Twelve Data** | `ema` (200, daily) | Macro trend | 13 | included | Included | NO | Compute locally from daily candles |
| **Twelve Data** | `mfi` (14, 5min) | Money Flow Index | 13 | included | Included | MAYBE | Compute locally from OHLCV |
| **Twelve Data** | `obv` (5min) | On-Balance Volume | 13 | included | Included | MAYBE | Compute locally from close+volume |
| **Twelve Data** | `supertrend` (5min) | Trend confirm | 13 | included | Included | MAYBE | Compute locally from OHLC |
| **Polygon** | `/v2/reference/news` | Headlines/sentiment | 13 (cached 10min) | 5/sec | Per-call | MAYBE | Keep, cache aggressively |
| **Polygon** | `/v3/snapshot/options` | Chain/Greeks/OI | per-alert | 5/sec | Per-call | YES | Keep |
| **Polygon** | `/v2/aggs/ticker/.../range` (1min options) | Premium recovery | per-alert | 5/sec | Per-call | YES | Keep |
| **Polygon** | `/v2/aggs/ticker/SPY/range/1/day` | Regime data (daily) | 1 (cached daily) | 5/sec | Per-call | YES | Keep |
| **Polygon** | `/v2/aggs/ticker/.../range/1/day` (prior levels) | Yesterday H/L/pivots | 13 (cached daily) | 5/sec | Per-call | YES | Keep |
| **Polygon** | `/v2/aggs/ticker/.../range/5/minute` (pre-market) | Pre-market H/L | 13 (once/day) | 5/sec | Per-call | YES | Keep |
| **Unusual Whales** | `/api/stock/{ticker}/net-prem-ticks` | Net premium flow | 13 (if score >= 30) | Unknown | Subscription | YES | Keep behind flag |
| **Unusual Whales** | `/api/darkpool/{ticker}` | Dark pool prints | 13 (if score >= 30) | Unknown | Subscription | YES | Keep behind flag |
| **Unusual Whales** | `/api/stock/{ticker}/greek-exposure` | GEX | 13 (if score >= 30) | Unknown | Subscription | YES | Keep behind flag |
| **Unusual Whales** | `/api/market/{sector}/sector-tide` | Sector flow | 13 (if score >= 30) | Unknown | Subscription | MAYBE | Keep behind flag |
| **Unusual Whales** | `/api/stock/{ticker}/flow-recent` | 0DTE flow | 13 (if score >= 30) | Unknown | Subscription | MAYBE | Keep behind flag |
| **Unusual Whales** | `/api/stock/{ticker}/max-pain` | Max pain strike | 13 (if score >= 30) | Unknown | Subscription | MAYBE | Keep behind flag |
| **Grok (xAI)** | Chat completions | AI trade analysis | per-alert | RPM-based | Per-call | NO | Behind flag, Phase 3+ |
| **Finnhub** | `/api/v1/calendar/earnings` | Earnings calendar | 1 (cached 1hr) | Unknown | Free tier | YES | Keep |
| **UW Congress** | `/api/congress/trades` | Congress member trades | 1 (cached 1hr) | Included | Included w/ UW sub | YES | Keep behind flag. Directional bias signal |
| **SEC EDGAR** | `/cgi-bin/browse-edgar` (Form 4) | Insider buys/sells (officers, directors, 10%+ owners) | 13 (cached 1hr) | 10/sec | Free | YES | New source. Strong alpha for directional bias |
| **Capitol Trades** | REST API | Congress trades with historical performance | 1 (cached 1hr) | Unknown | Free tier | MAYBE | Backup/enrichment for UW Congress data |
| **StockTwits** | `/api/2/streams/symbol` | Retail sentiment (bull/bear ratio) | 13 | Unknown | Free | YES | Contrarian signal: extreme retail = fade |
| **Yahoo Finance** | Options chain fallback | Chain data when Polygon misses | per-alert | No limit | Free | YES | Already in codebase (yfinance). Fallback only |

### API Call Reduction Opportunities

**Current:** ~91 Twelve Data + ~78 UW + ~40 Polygon = ~209 API calls per scan (worst case)

**Optimized:** Computing BBands, RSI, MACD, Keltner, MFI, OBV, SuperTrend locally from candle data (the workflow already does this for BBands/RSI/MACD/Keltner). Remaining Twelve Data calls: 3 per ticker (5min + 15min + 1min candles) + 1 daily EMA200 = ~40 Twelve Data calls. Net savings: ~51 Twelve Data API calls/scan.

**Further optimization (Phase 2+):** If we can source 5min/15min/1min candles from the harvester DB (Polygon WebSocket data already captured by owlet-harvester), we eliminate Twelve Data entirely during market hours. This requires verifying that the harvester captures intraday candles with sufficient granularity and freshness.

### API Keys Required

| Service | Key Variable | Where Stored | Notes |
|---|---|---|---|
| Twelve Data | `TWELVE_DATA_KEY` | `.env` (new) | Grow plan, 55 credits/min |
| Polygon | `POLYGON_KEY` | `.env` (existing) | Already used by owlets; sourcing needs its own or shares |
| Unusual Whales | `UW_KEY` | `.env` (new) | Bearer token, subscription required |
| xAI (Grok) | `GROK_KEY` | `.env` (new) | Chat completions API |
| Finnhub | `FINNHUB_KEY` | `.env` (new) | Free tier, earnings calendar |
| Discord | `WEBHOOK_BULLISH`, `WEBHOOK_BEARISH`, `WEBHOOK_REVERSAL`, `WEBHOOK_STANDDOWN` | `.env` or `config.py` | 4 webhook URLs for different alert channels |

---

## 5. Migration Phases

### Phase 0: Preparation (Week 1)

**Goal:** Infrastructure ready, skeleton in place, no behavioral changes.

Tasks:
- [x] Copy N8N JSON to `docs/original/n8n_0dte_scanner.json`
- [ ] Create `options_owl/sourcing/` module skeleton with all directories and `__init__.py` files
- [ ] Add `owlet-sourcing` service to `docker-compose.yml` (initially with `command: echo "not yet"`)
- [ ] Add sourcing-specific settings to `options_owl/config/settings.py` (new `SourcingSettings` class)
- [ ] Extract and document all API keys/credentials; add to `.env.example`
- [ ] Create test directory structure: `tests/sourcing/`
- [ ] Write `scanner.py` skeleton with market hours check and empty scan loop
- [ ] Write `config.py` with ticker list, sector map, correlation groups, thresholds

**Deliverables:**
- Module skeleton that imports and does nothing
- Docker container that starts and logs "sourcing not yet implemented"
- All API keys in `.env` on droplet

### Phase 1: Core Indicators (Week 2)

**Goal:** All technical indicators computed locally from candle data, with full test coverage.

Tasks:
- [ ] Port `calcEMA()` to `indicators/ema.py` -- 9/21 EMA crossover, 200 EMA, spread detection
- [ ] Port `calcBBands()` to `indicators/bollinger.py` -- period=20, sd=2, double-touch, TTM squeeze
- [ ] Port `calcRSI()` to `indicators/rsi.py` -- period=9, Wilder smoothing, divergence, curl direction
- [ ] Port `calcMACD()` to `indicators/macd.py` -- fast=5, slow=13, signal=4, histogram crossover
- [ ] Port `calcKeltner()` to `indicators/keltner.py` -- period=20, multiplier=1.5
- [ ] Implement VWAP calculation in `indicators/vwap.py` -- session VWAP from full-day OHLCV, slope detection
- [ ] Implement ATR in `indicators/atr.py` -- expansion/contraction detection
- [ ] Implement OBV in `indicators/obv.py` -- slope, divergence, volume surge
- [ ] Implement MFI in `indicators/mfi.py` -- period=14
- [ ] Implement SuperTrend in `indicators/supertrend.py`
- [ ] Implement candlestick patterns in `indicators/candles.py` -- engulfing, pin bar, hammer, shooting star, doji
- [ ] Implement volume analysis in `indicators/volume.py` -- time-weighted normalization, spike detection

**Critical implementation note:** All indicator functions must be pure -- they take numpy arrays or lists of floats and return computed values. No API calls, no state, no side effects. This makes them trivially testable.

**Test strategy:** Each indicator gets test vectors computed from known data. For validation during migration, we will feed the same candle data to both the N8N JS implementation and the Python implementation and diff the outputs.

```python
# Example: indicators/ema.py
def calc_ema(closes: list[float], period: int) -> list[float]:
    """Compute EMA from oldest-first closing prices.

    Args:
        closes: List of closing prices, oldest first.
        period: EMA period.

    Returns:
        List of EMA values, same length convention as input minus warmup.
    """
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    result = [ema]
    for close in closes[period:]:
        ema = close * k + ema * (1 - k)
        result.append(ema)
    return result
```

**Deliverables:**
- All 12 indicator modules with full implementations
- Unit tests for every indicator with deterministic test vectors
- Validation notebook comparing JS vs Python outputs on same input data

### Phase 2: Scoring Engine (Week 3)

**Goal:** Full scoring pipeline ported, producing identical scores to N8N on the same input data.

Tasks:
- [ ] Implement all 21+ scoring signals in `scoring/signals.py` as pure functions
- [ ] Implement penalty functions in `scoring/penalties.py`
- [ ] Implement Fast Alert path in `scoring/fast_alert.py`
- [ ] Implement Gap-and-Fade + Gap-Down Reversal in `scoring/gap_fade.py`
- [ ] Implement market regime classification in `scoring/regime.py`
- [ ] Implement score aggregation in `scoring/aggregator.py`:
  - Raw score accumulation (uncapped)
  - Override paths (Fast Alert, Gap-Fade, Gap-Down Reversal, Catalyst)
  - Final 0-100 normalization
- [ ] Implement Bearish Conviction Framework
- [ ] Implement SPX Power Fade, Mid-Day Reversal, Afternoon Momentum detectors
- [ ] Port ML v4 logistic regression to `scoring/ml_model.py`
- [ ] Implement quality gate filter in `filters/quality_gate.py`:
  - Circuit breaker check
  - Sector rotation detector
  - Session context gate (FOMC, per-ticker AVOID)
- [ ] Implement penalty veto filter in `filters/penalty_veto.py`
- [ ] Implement cooldown management in `filters/cooldown.py`:
  - 90-min same direction, 30-min opposite direction
  - Cross-index opposite-direction block (SPY/QQQ/IWM)
  - Correlated ticker group cooldowns
- [ ] Implement regime gate in `filters/regime_gate.py`:
  - Counter-trend hard blocks (BULL regime + bearish = block)
  - RSI + SPY confirmation thresholds

**Test strategy:**
- Unit tests for every scoring function with known inputs and expected point outputs
- Integration test: feed a complete ticker data snapshot through the full scoring pipeline and assert the final score and direction
- Historical replay: take 50+ historical N8N alerts, feed the same candle data through the Python pipeline, and verify scores are within +/-5 points

**Deliverables:**
- Complete scoring engine
- 200+ unit tests
- Historical replay comparison report

### Phase 3: Data Source Integration (Week 4)

**Goal:** Wire up all external APIs, each behind its own feature flag.

Tasks:
- [ ] Implement `sources/twelve_data.py`:
  - `fetch_candles(ticker, interval, outputsize)` -- returns OHLCV list
  - Rate limiting (55 credits/min)
  - Error handling with retries
- [ ] Implement `sources/polygon_data.py`:
  - `fetch_news(ticker, limit)` -- with 10-min cache
  - `fetch_options_snapshot(ticker, expiry)` -- for chain
  - `fetch_option_aggs(occ_symbol, date)` -- for premium recovery
  - `fetch_daily_aggs(ticker, from_date, to_date)` -- for regime + prior levels
  - Rate limiting (5 calls/sec, batched 5-at-a-time with 1.1s delay)
- [ ] Implement `sources/unusual_whales.py`:
  - `fetch_flow(ticker, date)` -- net premium ticks
  - `fetch_dark_pool(ticker, date)` -- large prints
  - `fetch_gex(ticker, date)` -- gamma exposure
  - `fetch_sector_tide(sector, date)` -- sector flow
  - `fetch_0dte_flow(ticker)` -- recent flow
  - `fetch_max_pain(ticker, date)` -- max pain strike
  - All behind `ENABLE_UNUSUAL_WHALES` flag
  - Only called when pre-UW score >= 30 (matching N8N behavior)
- [ ] Implement `sources/grok.py`:
  - Chat completions call with trade context
  - Behind `ENABLE_GROK_AI` flag
- [ ] Implement `sources/finnhub.py`:
  - Earnings calendar with 1-hour cache
- [ ] Implement `sources/harvester.py`:
  - Read 5min/15min candles from `journal/owlet-harvester/options_data.db`
  - WAL mode reads with `PRAGMA busy_timeout = 5000`
  - Fallback to Twelve Data if harvester data is stale
- [ ] Implement state management in `state.py`:
  - SQLite-backed state (replaces N8N staticData)
  - Cooldown tracking per ticker + direction + timestamp
  - Daily/weekly counters
  - Regime data cache (daily)
  - Prior levels cache (daily)
  - Earnings calendar cache (hourly)
  - Trade history (last 50 resolved for ML adjustment)

**Feature flags (all in settings):**

```python
class SourcingSettings(BaseSettings):
    # Data sources
    ENABLE_UNUSUAL_WHALES: bool = True
    ENABLE_GROK_AI: bool = False
    ENABLE_FINNHUB_EARNINGS: bool = True
    ENABLE_POLYGON_NEWS: bool = True
    ENABLE_HARVESTER_CANDLES: bool = False  # Phase 2+

    # Alpha sources (new — high-edge, non-technical signals)
    ENABLE_SEC_INSIDER: bool = True           # SEC Form 4 insider trades
    ENABLE_UW_CONGRESS: bool = True           # Congress member trades (via UW)
    ENABLE_CAPITOL_TRADES: bool = False       # Capitol Trades API (backup)
    ENABLE_STOCKTWITS_SENTIMENT: bool = True  # Retail sentiment (contrarian)

    # ML gates (new — learned models for scoring/timing)
    ENABLE_ML_FLOW_CLASSIFIER: bool = False   # Gate 1: smart money flow classifier
    ENABLE_ML_ENTRY_OPTIMIZER: bool = False   # Gate 2: entry timing optimizer
    ENABLE_ML_QUALITY_PREDICTOR: bool = False # Gate 3: replace hand-tuned score with ML
    ENABLE_ML_REGIME_WEIGHTER: bool = False   # Gate 4: dynamic source weighting
    ENABLE_ML_EXIT_ADVISOR: bool = False      # Gate 5: exit timing advisory

    # Scoring features
    ENABLE_FAST_ALERT: bool = True
    ENABLE_GAP_FADE: bool = True
    ENABLE_GAP_DOWN_REVERSAL: bool = True
    ENABLE_CATALYST_BOOST: bool = True
    ENABLE_REGIME_GATE: bool = True
    ENABLE_SECTOR_ROTATION: bool = True

    # Output
    SOURCING_DISCORD_OUTPUT: bool = True
    SOURCING_DB_OUTPUT: bool = True
    SOURCING_SCAN_INTERVAL: int = 180  # seconds
```

**Deliverables:**
- All data source modules with error handling and rate limiting
- State management with SQLite persistence
- Integration tests with mocked API responses
- Feature flags for every data source

### Phase 3.5: Alpha Sources + ML Gates (Week 4-5)

**Goal:** Integrate high-alpha non-technical data sources and build ML-powered scoring/timing gates.

This phase runs in parallel with Phase 3 API wiring. Alpha sources feed into Tier 3 (amplifiers) and Tier 5 (calibration) of the scoring engine. ML gates are trained on historical data and deployed behind feature flags.

#### New Data Sources (Smart Money / Insider / Sentiment)

**Why these sources matter:** The current system is 100% technical analysis — it only looks at price, volume, and indicators. Every algo shop in the world does the same TA faster. The edge comes from data that most retail traders don't use: who is buying (insiders, Congress, institutions) and what is the crowd doing (contrarian sentiment).

Tasks:
- [ ] Implement `sources/sec_edgar.py`:
  - Parse SEC EDGAR Form 4 filings (insider buys/sells)
  - Track: officer name, title, transaction type (buy/sell), shares, dollar value
  - Cache hourly (filings come in batches, not real-time)
  - Output: `InsiderActivity(ticker, net_insider_buys_7d, largest_buy_dollars, insider_buy_ratio)`
  - Scoring: net insider buys in last 7 days → +3 to +8 directional bias points
  - **Free API, no key required.** Rate limit: 10 req/sec

- [ ] Implement `sources/capitol_trades.py` (or use UW Congress endpoint):
  - Congress member stock trades (STOCK Act, disclosed within 45 days)
  - UW already includes `/api/congress/trades` — check if our subscription covers it
  - Track: member name, party, committee, trade date, ticker, amount, buy/sell
  - Cache hourly
  - Output: `CongressActivity(ticker, net_congress_buys_30d, committee_relevance, member_performance)`
  - Scoring: recent Congress buys on our ticker → +3 to +5 directional bias
  - **Already included in UW subscription ($0 extra)**

- [ ] Implement `sources/stocktwits.py`:
  - Retail sentiment via StockTwits API (free, no key required)
  - Track: bullish_count, bearish_count, total_messages, sentiment_ratio
  - **Use as CONTRARIAN signal:** extreme bullish (>80%) = bearish bias, extreme bearish (<20%) = bullish bias
  - Moderate readings (40-60%) = neutral, no signal
  - Output: `RetailSentiment(ticker, bull_ratio, msg_velocity, contrarian_signal)`
  - Scoring: extreme contrarian signal → +3 to +5 points
  - Cache per scan cycle (3 min)

- [ ] Wire alpha sources into scoring engine:
  - Tier 3 (amplifiers): insider_bias (0-5), congress_bias (0-3), contrarian_sentiment (0-3)
  - Update `tier3_amplifiers.py` to accept new source data
  - All behind individual feature flags — disabled sources contribute 0 points

#### ML Gates (Trained Models for Scoring + Timing)

**Why ML:** Hand-tuned scoring (21 signals × fixed point weights) can't capture non-linear interactions. ML learns that "RSI(35) + MACD crossing up + high volume = 78% WR" while "RSI(65) + MACD crossing up + low volume = 45% WR" — your current scoring gives both similar scores.

All ML gates use **LightGBM** (already in the stack for exit models). Training data comes from the 220+ historical trades in `paper_trades` + harvester candle data.

Tasks:
- [ ] **ML Gate 1: Smart Money Flow Classifier** (`scoring/ml_gates/flow_classifier.py`)
  - **Problem:** Raw UW flow data is noisy. A $1M call could be a hedge, spec bet, or institutional roll.
  - **Features:** order_size_vs_avg, time_of_day, sweep_vs_block, delta_of_contract, DTE, bid_vs_ask_side
  - **Label:** Did underlying move in flow direction within 30 min?
  - **Model:** LightGBM binary classifier → `P(smart_money)` score
  - **Integration:** Feed P(smart_money) into Tier 3 as `flow_quality` (0-5 pts)
  - **Training data:** UW flow snapshots from harvester + Polygon price data (both already collected)
  - **Flag:** `ENABLE_ML_FLOW_CLASSIFIER`

- [ ] **ML Gate 2: Entry Timing Optimizer** (`scoring/ml_gates/entry_optimizer.py`)
  - **Problem:** Buying at signal time vs 2 min later can save 10-20% on premium
  - **Features:** current_premium_vs_5min_avg, bid_ask_spread, seconds_since_signal, underlying_velocity_60s, rsi9_1min, volume_vs_avg, vix, time_of_day
  - **Label:** Was premium lower in next 1-5 min? By how much?
  - **Model:** LightGBM regressor → `expected_savings_pct`
  - **Integration:** If expected_savings > 3%, delay entry 60-120s (smarter version of dip-confirm gate)
  - **Training data:** Tick logs from `journal/owlet-*/logs/` (premium ticks already captured by tick logger)
  - **Flag:** `ENABLE_ML_ENTRY_OPTIMIZER`

- [ ] **ML Gate 3: Signal Quality Predictor** (`scoring/ml_gates/quality_predictor.py`)
  - **Problem:** Hand-tuned 0-100 score uses fixed weights. ML can learn non-linear interactions.
  - **Features:** All 21 current signals as RAW VALUES (not point scores) — RSI value (0-100), MACD histogram value, EMA spread, volume ratio, ATR percentile, VWAP distance, OBV slope + flow_score (Gate 1), insider_bias, sentiment_ratio
  - **Label:** Trade outcome (win/loss, or P&L bucket: big_loss / small_loss / small_win / big_win)
  - **Model:** LightGBM classifier → `P(win)` as the score (0-100, naturally calibrated)
  - **Integration:** When enabled, REPLACES the hand-tuned 5-tier score. `P(win) × 100` IS the score.
  - **Training data:** 220+ trades from paper_trades + indicator values reconstructed from harvester candles
  - **Bootstrapping:** Start with hand-tuned scoring (Phase 2). After 200+ trades on new system, train Gate 3. Phase in via A/B test.
  - **Flag:** `ENABLE_ML_QUALITY_PREDICTOR`

- [ ] **ML Gate 4: Regime-Aware Source Weighting** (`scoring/ml_gates/regime_weighter.py`)
  - **Problem:** On FOMC days, news dominates. On quiet days, technicals dominate. On meme days, flow dominates. Fixed weights can't adapt.
  - **Features:** vix_level, vix_change, spy_gap, is_fomc_cpi_nfp, sector_rotation_score, premarket_volume_vs_avg, day_of_week
  - **Label:** Which source group was most predictive today? (technical / flow / sentiment / macro)
  - **Model:** Multi-class LightGBM → dynamic weight multipliers per source category
  - **Integration:** Multiply each tier's contribution by the regime-specific weight before summing
  - **Training data:** Requires 3+ months of trade data with full indicator breakdowns. Phase 3+ only.
  - **Flag:** `ENABLE_ML_REGIME_WEIGHTER`

- [ ] **ML Gate 5: Exit Timing Advisory** (`scoring/ml_gates/exit_advisor.py`)
  - **Problem:** V5 FSM uses fixed % thresholds for trailing stops. Optimal exit depends on entry context.
  - **Features:** entry_score_components, time_of_day_at_entry, current_gain_pct, time_held, atr, premium_decay_rate, vix, market_regime
  - **Label:** Optimal hold time from historical trades
  - **Model:** LightGBM with survival analysis objective → `P(should_exit_now)`
  - **Integration:** Advisory overlay on V5 FSM — logs recommendation but does NOT override FSM gates until validated
  - **Training data:** 220+ closed trades with full premium history from tick logs
  - **Flag:** `ENABLE_ML_EXIT_ADVISOR`

#### ML Training Infrastructure

- [ ] Create `scripts/train_ml_gates.py`:
  - Reconstructs indicator values at signal time from harvester candle data
  - Joins with trade outcomes from paper_trades
  - Trains each gate model independently
  - Saves models to `journal/models/ml_gates/` (same pattern as existing LightGBM models)
  - Outputs: model file, feature importance, cross-validation metrics, confusion matrix
  - Intended to run monthly (or after 50+ new trades)

- [ ] Create `scripts/backtest_ml_gates.py`:
  - Walk-forward validation: train on months 1-2, predict month 3, slide forward
  - Prevents overfitting to recent data
  - Reports: accuracy, precision, recall, profit factor, comparison vs hand-tuned baseline

#### Rollout Strategy

ML gates roll out in strict order based on data requirements and risk:

| Gate | Data Needed | Earliest Deploy | Risk |
|---|---|---|---|
| Gate 1 (flow classifier) | UW flow + price data (have both) | Phase 3.5 | Low — advisory only, adds to Tier 3 |
| Gate 2 (entry optimizer) | Premium tick logs (have them) | Phase 3.5 | Low — delay mechanism, worse case = buy at signal time |
| Gate 3 (quality predictor) | 200+ trades on new scoring (need to accumulate) | Phase 5+ (after 200 trades) | Medium — replaces entire scoring engine |
| Gate 4 (regime weighter) | 3+ months of scored trades with breakdowns | Phase 6+ | Medium — changes all source weights dynamically |
| Gate 5 (exit advisor) | 200+ trades with tick-level premium history | Phase 5+ | Low — advisory only, V5 FSM still makes decisions |

**Deliverables:**
- 3 new data source modules (SEC EDGAR, Capitol Trades/UW Congress, StockTwits)
- 5 ML gate modules (feature extraction + model inference)
- Training and backtesting scripts
- Feature flags for every source and gate
- Integration tests with mocked source data and pre-trained model fixtures

### Phase 4: Output + Shadow Mode (Week 5-6)

**Goal:** Full pipeline running in shadow mode alongside N8N, comparing outputs.

Tasks:
- [ ] Implement `output/discord.py`:
  - Format rich embeds matching N8N format exactly (fields, colors, emoji, layout)
  - POST to same Discord webhook URLs
  - Bullish/bearish/reversal/stand-down channel routing
- [ ] Implement `output/formatter.py`:
  - Build final alert data (premium recovery from Polygon aggs)
  - Webull context (Bayesian signature, historical WR)
  - Score breakdown formatting
- [ ] Implement `output/db_writer.py`:
  - Write signals to `journal/owlet-sourcing/signals.db`
  - Schema: `signals` table with ticker, direction, score, strike, premium, signals JSON, penalties JSON, timestamp
- [ ] Implement stand-down message at end of day
- [ ] Implement daily summary (3:50 PM ET)
- [ ] **Shadow mode infrastructure:**
  - `owlet-sourcing` runs full pipeline but writes to a **shadow Discord channel** (not the live ones)
  - Shadow signals also written to `journal/owlet-sourcing/shadow_signals.db`
  - Comparison script: reads N8N alerts from `raw_messages.db` and shadow signals, diffs scores/directions/timing
  - Daily comparison report posted to admin Discord channel

**Shadow mode validation criteria (must ALL pass before cutover):**
- [ ] 95%+ of N8N alerts are also generated by owlet-sourcing (within same 3-min window)
- [ ] Score difference < 10 points on 90%+ of matched alerts
- [ ] Direction matches on 98%+ of matched alerts
- [ ] No false negatives on alerts that resulted in profitable trades
- [ ] Zero crashes or unhandled exceptions over 5 trading days

**Deliverables:**
- Complete output pipeline
- Shadow mode running for 1+ weeks
- Daily comparison reports
- Discrepancy resolution log

### Phase 5: Cutover (Week 6)

**Goal:** owlet-sourcing becomes the sole signal source.

Tasks:
- [ ] Verify all shadow mode criteria are met
- [ ] Switch Discord webhook URLs from shadow to live channels
- [ ] Disable N8N workflow (pause, do not delete)
- [ ] Monitor owlet-sourcing for full trading day
- [ ] Verify trading owlets receive and process signals correctly
- [ ] Keep N8N paused (not deleted) for 2 weeks as rollback option

**Rollback plan:**
1. Re-enable N8N workflow (< 1 minute)
2. Stop owlet-sourcing container
3. N8N immediately resumes signal generation

**Rollback triggers:**
- owlet-sourcing crashes during market hours
- Signal generation stops for > 10 minutes
- Score quality degrades (manual review of first 3 alerts)
- Any trading owlet fails to process a signal

**Deliverables:**
- N8N disabled, owlet-sourcing live
- 1-week monitoring period with daily review

### Phase 6: Direct Feed (Week 7+)

**Goal:** Eliminate Discord as a signal bus.

Tasks:
- [ ] Extend `discord_collector.py` with a `SignalDBReader` that reads from `journal/owlet-sourcing/signals.db`
- [ ] Add `SIGNAL_SOURCE` setting: `discord` (default) or `db`
- [ ] When `SIGNAL_SOURCE=db`, owlet reads signals directly from sourcing DB
  - Poll every 5 seconds for new rows
  - Structured data -- no parsing needed
  - Sub-100ms latency (vs 1-3s via Discord)
- [ ] Keep Discord output enabled for human visibility (but owlets no longer depend on it)
- [ ] Add WebSocket option (future): owlet-sourcing publishes signals on a local Unix socket, owlets subscribe

**Deliverables:**
- Trading owlets can read signals from either Discord or DB
- Discord output continues for human monitoring
- Latency reduced from ~2s to <100ms

---

## 6. Score Normalization (0-100)

### Current System (N8N)

The N8N workflow uses an uncapped raw score that typically ranges from 40 to 180+. The raw score is clipped to 0-100 only at the very end for display. Internally, override paths (Fast Alert, Gap-Fade) set the score to 80-100 directly, bypassing the normal accumulation.

Key thresholds in the current system:
- `minScoreThreshold = 50` (quality gate floor, fixed)
- Score >= 30 triggers UW data fetch (institutional overlay)
- Fast Alert threshold: 50-60 (time-dependent)
- Gap-Fade threshold: 40
- Trading owlets' entry pipeline requires score >= 78

### Target System (owlet-sourcing)

Maintain backward compatibility with the trading owlets' score >= 78 floor. The normalization should produce scores that are directly comparable to the current system.

**Approach:** Sigmoid normalization of the raw score, calibrated against the historical distribution of N8N raw scores.

```python
def normalize_score(raw_score: float, max_raw: float = 150.0) -> int:
    """Normalize raw score to 0-100 using sigmoid curve.

    Calibrated so that:
    - raw 50 -> ~35 (below threshold, filtered)
    - raw 75 -> ~55 (moderate)
    - raw 100 -> ~75 (strong)
    - raw 120 -> ~85 (very strong)
    - raw 150+ -> ~95 (elite)

    The trading owlets use score >= 78 as the entry floor.
    """
    if raw_score <= 0:
        return 0
    # Sigmoid centered at raw_score = 85 (median of "tradeable" signals)
    x = (raw_score - 85) / 25  # scale factor
    sigmoid = 1 / (1 + math.exp(-x))
    return max(0, min(100, int(sigmoid * 100)))
```

**Score meaning after normalization:**

| Normalized Score | Raw Score (approx) | Meaning | Action |
|---|---|---|---|
| 0-30 | < 50 | Noise / no signal | Filtered (not emitted) |
| 30-50 | 50-70 | Weak signal | Below threshold |
| 50-70 | 70-95 | Moderate signal | May pass quality gate |
| 70-85 | 95-120 | Strong signal | Passes entry pipeline |
| 85-100 | 120+ | Elite signal | High conviction |

**Calibration plan:** During shadow mode (Phase 4), we will collect the mapping between raw scores and trade outcomes. If the sigmoid needs adjustment, we retune the center and scale parameters.

---

## 7. Testing Strategy

### Unit Tests (indicators + scoring)

Every indicator and scoring function is a pure function. Test with deterministic inputs:

```python
# tests/sourcing/indicators/test_ema.py
def test_ema_crossover_bullish():
    closes = [100, 99, 98, 97, 98, 99, 100, 101, 102, 103,
              104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
              114, 115]
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    # After sustained uptrend, 9 EMA should be above 21 EMA
    assert ema9[-1] > ema21[-1]

def test_ema_matches_n8n():
    """Cross-validate against known N8N output."""
    closes = [...]  # exact data from N8N execution
    expected = [...]  # N8N calcEMA output
    result = calc_ema(closes, 9)
    for py, js in zip(result, expected):
        assert abs(py - js) < 0.001
```

**Target: 400+ unit tests across all indicator and scoring modules.**

### Integration Tests (full scan cycle)

Mock all API responses, run a complete scan cycle, verify output:

```python
# tests/sourcing/test_scan_cycle.py
async def test_full_scan_produces_alert():
    """Feed mocked candle data that should trigger a bullish NVDA alert."""
    mock_candles = load_fixture("nvda_bullish_setup.json")
    with mock_apis(candles=mock_candles, chain=mock_chain):
        signals = await run_scan_cycle()
    assert len(signals) == 1
    assert signals[0].ticker == "NVDA"
    assert signals[0].direction == "bullish"
    assert signals[0].score >= 78
```

### E2E / Historical Replay Tests

Replay historical signal data through the pipeline:

```python
# tests/sourcing/test_replay.py
@pytest.mark.parametrize("fixture", load_historical_fixtures())
def test_historical_replay(fixture):
    """Replay historical N8N candle data and verify score/direction match."""
    result = score_ticker(fixture.ticker, fixture.candle_data, fixture.state)
    assert result.direction == fixture.expected_direction
    assert abs(result.raw_score - fixture.expected_raw_score) < 10
```

**Data source for replay tests:** Extract candle snapshots from harvester DB + N8N execution logs. Store as JSON fixtures in `tests/sourcing/fixtures/`.

### Shadow Testing (Phase 4)

Run owlet-sourcing alongside N8N for 1+ weeks. Automated comparison:

```python
# scripts/compare_sourcing_signals.py
# Reads N8N alerts from raw_messages.db and owlet-sourcing signals from shadow_signals.db
# Produces daily report: matched, missed, score diff, direction diff, timing diff
```

### Regression Tests

After cutover, maintain a corpus of "golden" signals -- known inputs that must produce known outputs. Run on every PR:

```bash
pytest tests/sourcing/ -x -q  # must pass before deploy
```

---

## 8. Risk Mitigation

### Feature Flags

Every data source and major scoring feature is behind a feature flag. This allows:
- Disabling a broken API without redeploying
- A/B testing signal variants
- Gradual rollout of new features

### Shadow Mode

Phase 4 runs owlet-sourcing in shadow mode for 1+ weeks before cutover. The shadow pipeline:
- Writes to a **separate Discord channel** (not live trading channels)
- Writes to a **separate SQLite DB** (shadow_signals.db)
- Does NOT affect trading owlets in any way
- Automated daily comparison against N8N signals

### Rollback Plan

**Time to rollback: < 5 minutes.**

1. SSH to droplet
2. Re-enable N8N workflow (already paused, not deleted)
3. `docker compose stop owlet-sourcing`
4. N8N resumes signal generation within 3 minutes (next scheduled trigger)

The N8N workflow will remain paused (not deleted) for at least 2 weeks after cutover.

### No Trading Owlet Changes Until Proven

Trading owlets (`owlet-kody`, `owlet-adam`, etc.) are NOT modified until Phase 6. In Phases 1-5, they continue consuming signals from Discord exactly as they do today. The only change is who posts to Discord (N8N vs owlet-sourcing).

### Monitoring

After cutover, monitor for:
- **Signal count:** Compare daily alert count to N8N historical average (typically 2-6/day)
- **Score distribution:** Compare score histogram to N8N historical distribution
- **Timing:** Signals should appear within the same 3-minute window
- **API errors:** Any data source failure should log and continue (not crash)
- **Container health:** Docker healthcheck (heartbeat in scanner loop)

### Known Risk: State Migration

N8N's `$getWorkflowStaticData('global')` contains accumulated state:
- `tickerCooldowns` -- active cooldowns with timestamps
- `tradeHistory` -- last 50 resolved trades for ML adjustment
- `weeklyAdjustments` -- score threshold, signal effectiveness
- `openPositions` -- N8N's exit monitor state (not needed -- owlets handle exits)

**Migration approach:** Start owlet-sourcing with fresh state. Cooldowns reset naturally within 90 minutes. Trade history can be seeded from `raw_messages.db` (historical alerts). Weekly adjustments are a minor optimization that will re-converge within 1-2 weeks.

---

## 9. Dependencies and Prerequisites

### API Keys and Credentials

| Credential | Source | Status | Action |
|---|---|---|---|
| `TWELVE_DATA_KEY` | Twelve Data Grow plan | Have it (from N8N) | Add to droplet `.env` |
| `POLYGON_KEY` | Polygon subscription | Have it (existing) | Already in `.env`; may need separate key for rate limits |
| `UW_KEY` | Unusual Whales subscription | Have it (from N8N) | Add to droplet `.env` |
| `GROK_KEY` | xAI API | Have it (from N8N) | Add to droplet `.env` |
| `FINNHUB_KEY` | Finnhub free tier | Have it (from N8N) | Add to droplet `.env` |
| `WEBHOOK_BULLISH` | Discord | Have it (from N8N) | Add to `config.py` or `.env` |
| `WEBHOOK_BEARISH` | Discord | Have it (from N8N) | Add to `config.py` or `.env` |
| `WEBHOOK_REVERSAL` | Discord | Have it (from N8N) | Add to `config.py` or `.env` |
| `WEBHOOK_STANDDOWN` | Discord | Have it (from N8N) | Add to `config.py` or `.env` |

### Data Dependencies

| Data | Source | Status | Notes |
|---|---|---|---|
| 5min/15min/1min candles | Twelve Data API | Available | Primary candle source in Phase 1 |
| Harvester candle data | `journal/owlet-harvester/options_data.db` | Available | 7GB+, stock candles. Phase 2+ alternative to Twelve Data |
| Historical signals | `journal/owlet-*/raw_messages.db` | Available | For replay tests and trade history seeding |
| Options chain snapshots | Polygon API | Available | Already used by trading owlets |
| Historical signal corpus | N8N execution logs | Need to export | For regression test fixtures |

### Python Dependencies (already in project)

- `aiohttp` -- async HTTP client (for API calls)
- `aiosqlite` -- async SQLite (for state + signal DB)
- `pydantic` / `pydantic-settings` -- typed config and models
- `loguru` -- logging (same as owlets)
- `numpy` -- indicator computation (optional but faster)

### New Dependencies (may need to add)

- `ta-lib` or `pandas-ta` -- optional, for validating indicator implementations against industry-standard libraries. Not required for production (we compute indicators from scratch for transparency).

### Infrastructure

| Resource | Status | Notes |
|---|---|---|
| Droplet (129.212.138.145) | Running | Has capacity for 1 more container |
| Docker Compose | Ready | Add new service block |
| Journal directory | Create | `journal/owlet-sourcing/` with logs/ subdirectory |
| Shadow Discord channel | Create | Separate webhook for shadow mode testing |

---

## Appendix A: N8N Node Dependency Graph

```
3-Min Trigger
  -> Market Hours Gate
    -> IF Market Open
      -> Load State & Weekly Budget
        -> Fetch & Score All Tickers
          -> Quality Gate Filter
            -> IF Alerts Exist
              -> Fetch 0DTE Options Chain
                -> Options Chain Refinement
                  -> Grok AI Analysis (HTTP)
                    -> Build Final Alert Data
                      -> Penalty Veto Filter
                        -> ML v4 Inference
                          -> Build Webull Context
                            -> IF Final Alert
                              -> Format Entry Embed
                                -> Send Discord Entry Alert
                                  -> Update State Post-Alert
                                    -> API Cost Monitor
                                      -> IF API Warning
                                        -> Send API Warning
              -> (also) Stand Down Check
                -> IF Send Stand Down
                  -> Send Stand Down Embed
        -> Exit Monitor
          -> IF Exits Needed
            -> Send Exit Alert

Webull Close Webhook (separate entry point)
  -> Process Webull Close
    -> Respond OK
```

## Appendix B: Ticker Configuration

```python
TICKERS = ['SPY', 'QQQ', 'AVGO', 'NVDA', 'TSLA', 'AAPL', 'AMZN',
           'META', 'MSFT', 'AMD', 'GOOGL', 'PLTR', 'MSTR']

SECTOR_MAP = {
    'SPY': 'index', 'QQQ': 'index',
    'AVGO': 'semi', 'NVDA': 'semi', 'AMD': 'semi',
    'TSLA': 'auto',
    'AAPL': 'tech', 'AMZN': 'tech', 'META': 'tech',
    'MSFT': 'tech', 'GOOGL': 'tech',
    'PLTR': 'tech',
    'MSTR': 'crypto',
}

CORRELATION_GROUPS = {
    'Index': ['SPY', 'QQQ'],
    'Semiconductors': ['NVDA', 'AMD', 'AVGO'],
    'MegaCap Tech': ['AAPL', 'MSFT', 'GOOGL', 'META'],
}

# UW sector mapping (different from our sector map)
UW_SECTOR_MAP = {
    'AAPL': 'technology', 'MSFT': 'technology', 'NVDA': 'technology',
    'AMD': 'technology', 'PLTR': 'technology',
    'GOOGL': 'communication-services', 'META': 'communication-services',
    'TSLA': 'consumer-discretionary', 'AMZN': 'consumer-discretionary',
    'MSTR': 'financial',
}
```

## Appendix C: N8N Exit Monitor (NOT Migrated)

The N8N Exit Monitor (1096 lines) is **not migrated** because the trading owlets already have the V5 FSM exit engine, which is significantly more sophisticated:

| Feature | N8N Exit Monitor | Owlet V5 FSM |
|---|---|---|
| Poll interval | 3 min (N8N trigger) | 5 sec (async loop) |
| Exit gates | ~5 basic checks | 10 gates, category-aware |
| DTE awareness | Minimal | Full (0DTE vs multi-day) |
| Per-ticker config | None | Yes (V6) |
| Breakeven ratchet | No | Yes |
| Scaleout | No | Yes |
| DCA | No | Yes |
| Bid disappearance | No | Yes |
| Premium sources | 1 (Polygon) | 4 (WebSocket, Polygon, yfinance, delta approx) |

The N8N exit monitor exists because the original system was N8N-only. Now that the owlets handle their own exits, this code is vestigial.

## Appendix D: Indicator Parameter Reference

Quick reference for all indicator parameters used in the N8N workflow:

| Indicator | Parameters | Notes |
|---|---|---|
| EMA (fast) | period=9 | Primary signal, 5min timeframe |
| EMA (slow) | period=21 | Primary signal, 5min timeframe |
| EMA (macro) | period=200 | Daily timeframe, 5pt bonus only |
| Bollinger Bands | period=20, sd=2 | 5min timeframe, double-touch detection |
| RSI | period=9 | Wilder smoothing (not SMA-based) |
| MACD | fast=5, slow=13, signal=4 | Faster than standard (12,26,9) |
| Keltner Channels | period=20, multiplier=1.5 | Used for TTM Squeeze detection |
| ATR | period=14 (implied) | Expansion/contraction filter |
| MFI | period=14 | Money Flow Index, 5min |
| SuperTrend | default params | Twelve Data default |
| OBV | N/A | Slope over 5 bars |
| VWAP | session (full day) | Computed from OHLCV, not indicator API |
| Volume ratio | 10-bar average | Time-weighted before 10:00 AM ET |

## Appendix E: Twelve Data API Credit Usage

Current N8N usage per scan cycle (13 tickers):

| Endpoint | Credits/Call | Calls/Scan | Total Credits |
|---|---|---|---|
| `time_series` (5min) | 1 | 13 | 13 |
| `time_series` (15min) | 1 | 13 | 13 |
| `time_series` (1min) | 1 | 13 | 13 |
| `ema` (200, daily) | 1 | 13 | 13 |
| `mfi` (14, 5min) | 1 | 13 | 13 |
| `obv` (5min) | 1 | 13 | 13 |
| `supertrend` (5min) | 1 | 13 | 13 |
| **Total** | | **91** | **91 credits/scan** |

Grow plan limit: 55 credits/minute. At 91 credits/scan with a 3-minute interval, peak usage is ~30 credits/minute (within limit). But back-to-back scans during catch-up could hit the cap.

**After migration (Phase 1):** Computing MFI, OBV, SuperTrend locally from 5min candles:

| Endpoint | Credits/Call | Calls/Scan | Total Credits |
|---|---|---|---|
| `time_series` (5min) | 1 | 13 | 13 |
| `time_series` (15min) | 1 | 13 | 13 |
| `time_series` (1min) | 1 | 13 | 13 |
| `ema` (200, daily) | 1 | 13 | 13 |
| **Total** | | **52** | **52 credits/scan** |

**After Phase 2 (harvester candles):** If we source 5min/15min candles from the harvester DB and compute EMA200 locally from daily data:

| Endpoint | Credits/Call | Calls/Scan | Total Credits |
|---|---|---|---|
| `time_series` (1min) | 1 | 13 | 13 |
| **Total** | | **13** | **13 credits/scan** |

Or potentially **0** if we add 1min candle capture to the harvester.
