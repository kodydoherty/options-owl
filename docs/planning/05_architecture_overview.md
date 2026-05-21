# owlet-sourcing Architecture Overview

Definitive architectural reference for the owlet-sourcing signal generation bot. This document covers the system context, internal design, data flows, integration points, and operational concerns.

---

## 1. System Context Diagram

owlet-sourcing replaces the N8N cloud workflow as the signal generation engine. It runs as a new Docker container alongside the existing trading owlets and harvester on the same DigitalOcean droplet.

```
                          EXTERNAL APIs
                 +-----------------------------+
                 |  Polygon (options chain)     |  Required
                 |  Twelve Data (candle fallback)|  Feature-flagged
                 |  Unusual Whales (flow+congress)| Feature-flagged
                 |  Polygon News (sentiment)    |  Feature-flagged
                 |  Grok AI (analysis)          |  Feature-flagged
                 |  SEC EDGAR (insider trades)  |  Feature-flagged (FREE)
                 |  StockTwits (retail sent.)   |  Feature-flagged (FREE)
                 |  Capitol Trades (congress)   |  Feature-flagged (FREE)
                 +-------------+---------------+
                               |
                               v
+---------------------------------------------------------------------+
|                     DigitalOcean Droplet (129.212.138.145)           |
|                                                                     |
|  +---------------------+       +------------------------------+     |
|  | owlet-harvester     |       | owlet-sourcing (NEW)         |     |
|  | - Options snapshots |       | - 3-min scan loop            |     |
|  | - 5m stock candles  |       | - Local indicator engine     |     |
|  | - Writes            |       | - 0-100 scoring engine       |     |
|  |   options_data.db   +------>| - Filter pipeline            |     |
|  |   (~7GB, WAL)       | reads | - Discord webhook output (P1)|     |
|  +---------------------+       | - Signal DB output (P2)      |     |
|                                +--+----------+----------------+     |
|                                   |          |                      |
|                          Phase 1  |          | Phase 2              |
|                        (webhook)  |          | (direct DB)          |
|                                   v          v                      |
|                         +------------------+                        |
|                         | Discord Channel  |  <-- Phase 1 only      |
|                         +--------+---------+                        |
|                                  |                                  |
|            +---------------------+---------------------+            |
|            |                     |                     |            |
|            v                     v                     v            |
|  +----------------+   +----------------+   +----------------+       |
|  | owlet-kody     |   | owlet-adam     |   | owlet-vinny   |  ...  |
|  | $23K portfolio |   | $4.7K          |   | $3.1K         |       |
|  | discord_coll.  |   | discord_coll.  |   | discord_coll. |       |
|  | signal parser  |   | signal parser  |   | signal parser |       |
|  | 18-gate pipe.  |   | 18-gate pipe.  |   | 18-gate pipe. |       |
|  | Webull exec.   |   | Webull exec.   |   | Webull exec.  |       |
|  | V5 FSM exits   |   | V5 FSM exits   |   | V5 FSM exits  |       |
|  +----------------+   +----------------+   +----------------+       |
+---------------------------------------------------------------------+
```

**What is new:** owlet-sourcing container. Everything else remains unchanged during Phase 1.

**What it replaces:** N8N Cloud "0DTE Main Scanner" workflow (runs every 3 minutes, fetches from 4 APIs, sends Discord alerts). The N8N workflow will run in parallel during shadow testing, then be disabled.

**What it does NOT replace (Phase 1):** The trading owlets' existing Discord collector, signal parser/scorer, entry pipeline, execution, or exit engine. Those remain untouched. owlet-sourcing is purely a signal emitter.

---

## 2. owlet-sourcing Internal Architecture

```
options_owl/sourcing/
|
+-- scanner.py              Main loop: timer, market-hours guard, ticker iteration
|
+-- data/
|   +-- candle_provider.py   Read candles from harvester DB (primary) or Twelve Data (fallback)
|   +-- indicator_engine.py  Compute EMA, BB, RSI, MACD, VWAP, ATR, Keltner from raw bars
|   +-- options_provider.py  Fetch options chain snapshots from Polygon API
|   +-- news_provider.py     Polygon news sentiment (feature-flagged)
|   +-- flow_provider.py     Unusual Whales options flow (feature-flagged)
|   +-- ai_provider.py       Grok AI analysis (feature-flagged)
|
+-- scoring/
|   +-- engine.py            Orchestrate all tiers, produce final 0-100 score + direction
|   +-- direction.py         Tier 1: Direction Confidence (0-40)
|   +-- timing.py            Tier 2: Timing Quality (0-30)
|   +-- amplifiers.py        Tier 3: Edge Amplifiers (0-20)
|   +-- adjustments.py       Tier 4: Risk Adjustments (-10 to 0)
|   +-- calibration.py       Tier 5: Calibration Bonus (0-10)
|
+-- filters/
|   +-- quality_gate.py      Minimum indicator alignment threshold
|   +-- cooldown_manager.py  Per-ticker, per-direction cooldown enforcement
|   +-- penalty_veto.py      Block signals with critical disqualifiers
|   +-- options_validator.py Chain liquidity, spread, premium validation
|
+-- output/
|   +-- discord_webhook.py   Format and send Discord embeds (Phase 1)
|   +-- signal_db_writer.py  Write to shared signals.db (Phase 2)
|   +-- audit_logger.py      Full scoring breakdown to scoring_audit table
|
+-- state/
|   +-- state_manager.py     SQLite-backed ticker cooldowns, counters, circuit breaker
|   +-- models.py            Pydantic models for scan results, scores, state
|
+-- config.py                Sourcing-specific settings (pydantic-settings, env-driven)
```

### Module Responsibilities

**scanner.py** -- Entry point. Runs an async loop that fires every `SCAN_INTERVAL_SECONDS` (default 180). Guards against weekends, pre-market, and post-market. Iterates the ticker watchlist, orchestrates data fetch, scoring, filtering, and output. Writes heartbeat for Docker healthcheck.

**data/** -- Each provider is a standalone async class with a unified interface: `async def fetch(ticker: str) -> SomeDataModel | None`. Providers that fail return `None` and log a warning. The scanner never crashes on a single provider failure.

**scoring/** -- Pure functions. Every tier receives structured data and returns a numeric sub-score plus a list of human-readable reasons. No side effects, no I/O, fully deterministic given the same inputs. This makes them trivially unit-testable and backtestable.

**filters/** -- Stateful gates that decide whether a scored signal should be emitted. Cooldown manager reads/writes state. Quality gate and penalty veto are pure predicates.

**output/** -- Fire-and-forget emitters. Discord webhook sends an HTTP POST. Signal DB writer inserts a row. Audit logger writes every scoring computation regardless of pass/fail.

**state/** -- Single SQLite database at `journal/owlet-sourcing/state.db`. WAL mode for crash safety. Stores cooldowns, daily/weekly alert counters, circuit breaker state, and scan history.

---

## 3. Data Flow Diagram

### Single Scan Cycle (every 3 minutes)

```
Timer fires (180s interval)
    |
    v
[1] Market hours check
    |-- Weekend?  --> skip, sleep until next interval
    |-- Before 9:33 AM ET?  --> skip
    |-- After 3:57 PM ET?  --> skip
    |-- Market holiday?  --> skip
    |
    v
[2] For each ticker in watchlist (13+ tickers):
    |
    +--[2a] Fetch candle data
    |   |-- PRIMARY: Read last 78 five-minute candles from harvester DB
    |   |   (options_data.db via WAL mode, same as trading owlets)
    |   |-- STALENESS CHECK: If newest candle > 10 min old --> fallback
    |   +-- FALLBACK: Twelve Data REST API (if ENABLE_SOURCE_TWELVE_DATA=true)
    |
    +--[2b] Compute indicators locally
    |   |-- EMA (8, 21, 50)
    |   |-- Bollinger Bands (20, 2.0)
    |   |-- RSI (14)
    |   |-- MACD (12, 26, 9)
    |   |-- VWAP (session)
    |   |-- ATR (14)
    |   +-- Keltner Channels (20, 1.5)
    |
    +--[2c] Check cooldowns
    |   |-- Same direction: 90 min since last alert?
    |   +-- Opposite direction: 30 min since last alert?
    |       |-- On cooldown --> skip ticker, continue loop
    |
    +--[2d] Run scoring engine
    |   |-- Tier 1: Direction Confidence  (0-40 pts)
    |   |   |-- EMA alignment, crossover recency
    |   |   |-- BB position, squeeze breakout
    |   |   +-- RSI momentum, divergence
    |   |
    |   |-- Tier 2: Timing Quality  (0-30 pts)
    |   |   |-- MACD histogram slope, zero-line cross
    |   |   |-- VWAP position (price vs VWAP)
    |   |   +-- Candle pattern (engulfing, hammer, etc.)
    |   |
    |   |-- Tier 3: Edge Amplifiers  (0-20 pts)
    |   |   |-- Volume surge (vs 20-bar avg)
    |   |   |-- Multi-timeframe alignment
    |   |   +-- Optional: news sentiment, flow data, AI signal
    |   |
    |   |-- Tier 4: Risk Adjustments  (-10 to 0 pts)
    |   |   |-- Chop detection (low ADX, tight range)
    |   |   |-- Overextension penalty (RSI > 80 or < 20)
    |   |   +-- Earnings/event proximity penalty
    |   |
    |   +-- Tier 5: Calibration Bonus  (0-10 pts)
    |       +-- Historical ticker-specific edge (win rate, avg return)
    |
    |   Result: score (0-100) + direction (BULLISH/BEARISH)
    |
    +--[2e] Apply quality gate
    |   |-- Score < SCORE_THRESHOLD (default 60)?  --> skip ticker
    |   +-- Fewer than 3 tiers contributing?  --> skip ticker
    |
    +--[2f] Fetch options chain (only for passing signals)
    |   |-- Polygon /v3/snapshot/options/{ticker}
    |   |-- Select ATM strike for direction + correct expiry
    |   +-- Returns: bid, ask, mid, volume, OI, spread
    |
    +--[2g] Validate options chain
    |   |-- Bid-ask spread > 40%?  --> skip
    |   |-- Premium > tiered cap ($6/$7/$9)?  --> skip
    |   |-- Volume < minimum?  --> skip
    |   +-- No valid contract found?  --> skip
    |
    +--[2h] Apply penalty veto
    |   |-- Circuit breaker tripped?  --> skip
    |   +-- Daily alert cap reached?  --> skip
    |
    +--[2i] EMIT SIGNAL
        |
        +-- Discord webhook (Phase 1)
        |   |-- Format embed: ticker, direction, score, strike, premium
        |   +-- POST to bullish or bearish webhook URL
        |
        +-- Signal DB (Phase 2)
        |   |-- INSERT into signals.db: full score breakdown, chain data
        |   +-- Trading owlets poll this table
        |
        +-- Audit log (always)
            +-- INSERT into scoring_audit: all sub-scores, reasons, pass/fail
    |
    v
[3] Update state
    |-- Record cooldowns for emitted signals
    |-- Increment daily/weekly alert counters
    +-- Update circuit breaker state

[4] Log scan summary
    |-- Duration, tickers scanned, signals emitted, filters triggered
    +-- Write heartbeat file for Docker healthcheck
```

### Data Source Priority

```
+--------------------+----------+------------------+-------------------------+
| Data               | Priority | Source           | Fallback                |
+--------------------+----------+------------------+-------------------------+
| 5-min candles      | 1        | Harvester DB     | Twelve Data REST API    |
| Technical ind.     | N/A      | Computed locally | (no fallback needed)    |
| Options chain      | 1        | Polygon REST     | (required, no fallback) |
| News sentiment     | Optional | Polygon News     | Skip source             |
| Options flow       | Optional | Unusual Whales   | Skip source             |
| AI analysis        | Optional | Grok AI          | Skip source             |
| Insider trades     | Optional | SEC EDGAR        | Skip source (free API)  |
| Congress trades    | Optional | UW Congress       | Capitol Trades (free)   |
| Retail sentiment   | Optional | StockTwits       | Skip source (free API)  |
| ML flow scoring    | Optional | Local LightGBM   | Raw UW flow (no ML)    |
| ML quality score   | Optional | Local LightGBM   | Hand-tuned 5-tier score |
+--------------------+----------+------------------+-------------------------+
```

---

## 4. Key Design Decisions

### Local Indicator Computation

**Decision:** Compute all technical indicators locally from raw 5-minute candle data instead of fetching pre-computed indicators from Twelve Data.

**Why:**
- Eliminates Twelve Data as a required dependency (4 API calls per ticker per scan = 52 calls per cycle = 0 with local compute)
- Indicators become deterministic pure functions: same candles always produce the same EMA/RSI/MACD
- Enables offline backtesting against the 7GB+ harvester candle archive
- Removes a failure mode (Twelve Data outage = no signals)

**How:** `indicator_engine.py` takes a list of `CandleBar` objects and returns a typed `IndicatorSet` with all computed values. Uses standard formulas (Wilder's RSI, standard MACD 12/26/9, 20-period BB with 2.0 stdev).

**Tradeoff:** Must validate once that local computations match Twelve Data outputs. Minor numerical differences (floating point, lookback initialization) are acceptable as long as directional signals agree.

### Harvester DB as Primary Candle Source

**Decision:** Read 5-minute candles from the shared `options_data.db` (written by owlet-harvester) rather than making API calls.

**Why:**
- Data already exists -- the harvester captures candles for all 13+ tickers at 5-minute intervals
- Zero additional API calls for the primary data source
- Same access pattern already battle-tested by all 4 trading owlets
- Candle data is the foundation of all indicator calculations

**How:** Use `aiosqlite` with WAL mode (`PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout = 5000`), identical to how trading owlets read candles. Docker volume mount maps the harvester journal directory as read-write (WAL requires `-shm` sidecar file creation even for readers).

**Staleness guard:** If the newest candle in the DB is older than 10 minutes, fall back to Twelve Data API. This handles harvester outages without silent degradation.

**Tradeoff:** Limited to 5-minute resolution. The N8N workflow could fetch 1-minute candles from Twelve Data. For 0DTE scoring, 5-minute bars are sufficient -- the trading owlets already make all entry/exit decisions on 5-minute data.

### Feature Flags for Every External Source

**Decision:** Every data source beyond the harvester DB is gated behind an `ENABLE_SOURCE_*` environment variable.

```
ENABLE_SOURCE_HARVESTER_CANDLES=true    # Primary candle source
ENABLE_SOURCE_TWELVE_DATA=false         # Fallback candle source
ENABLE_SOURCE_POLYGON_NEWS=false        # News sentiment (Tier 3)
ENABLE_SOURCE_UNUSUAL_WHALES=false      # Options flow (Tier 3)
ENABLE_SOURCE_GROK_AI=false             # AI analysis (Tier 3)
ENABLE_SOURCE_POLYGON_OPTIONS=true      # Options chain (required for output)

# Alpha sources (smart money / insider / sentiment)
ENABLE_SEC_INSIDER=true                 # SEC EDGAR Form 4 insider trades (FREE)
ENABLE_UW_CONGRESS=true                 # Congress member trades via UW (included)
ENABLE_CAPITOL_TRADES=false             # Capitol Trades API backup (FREE)
ENABLE_STOCKTWITS_SENTIMENT=true        # Retail sentiment — contrarian (FREE)

# ML gates (trained models, deployed incrementally)
ENABLE_ML_FLOW_CLASSIFIER=false         # Gate 1: smart money flow scoring
ENABLE_ML_ENTRY_OPTIMIZER=false         # Gate 2: entry timing optimization
ENABLE_ML_QUALITY_PREDICTOR=false       # Gate 3: ML-based score (replaces 5 tiers)
ENABLE_ML_REGIME_WEIGHTER=false         # Gate 4: dynamic source weighting
ENABLE_ML_EXIT_ADVISOR=false            # Gate 5: exit timing advisory
```

**Why:**
- Enables incremental rollout: start with candles + options only, add sources one at a time
- Each source can be independently disabled if its API is down, rate-limited, or costing money
- Shadow testing can compare score quality with different source combinations
- Simplifies local development: run with zero external APIs using mock candle data

**Impact on scoring:** When a Tier 3 source is disabled, the scoring engine treats its contribution as 0 points. The score ceiling is effectively lower (max 80 instead of 100), but the threshold adjusts accordingly. A signal scoring 65/80 is functionally equivalent to 81/100.

### SQLite State Management

**Decision:** Persist all scanner state in `journal/owlet-sourcing/state.db` (SQLite, WAL mode).

**What is stored:**
- Per-ticker, per-direction cooldown timestamps
- Daily and weekly alert emission counters
- Circuit breaker state (trip count, reset time)
- Last scan result per ticker (for trend tracking)
- Scoring audit trail (every computation, pass or fail)

**Why:**
- Survives container restarts (Docker `restart: always`)
- Replaces N8N's `$getWorkflowStaticData('global')` which is volatile and lost on workflow restart
- Same operational pattern as trading owlets' `raw_messages.db`
- WAL mode for crash safety (write in progress does not corrupt)

**Schema design:** Separate tables for each concern (cooldowns, counters, audit). No cross-table joins needed in hot path. The audit table will grow large but is append-only and can be pruned on a 30-day retention.

### Phased Output Strategy

**Phase 1: Discord Webhook (drop-in replacement)**

```
owlet-sourcing --> Discord webhook POST --> Discord channel
                                              |
                                    4 trading owlets read via discord_collector.py
```

- Output format is identical to N8N's Discord embeds
- Trading owlets require zero code changes
- Can run owlet-sourcing in parallel with N8N for shadow comparison
- Transition: disable N8N, owlet-sourcing becomes sole signal source
- Risk: minimal. If owlet-sourcing fails, re-enable N8N within seconds.

**Phase 2: Direct Signal DB (eliminates Discord dependency)**

```
owlet-sourcing --> signals.db (SQLite, WAL)
                      |
            4 trading owlets poll signals.db directly
```

- Trading owlets get a new `signal_db_collector.py` that replaces `discord_collector.py`
- Eliminates: Discord message parsing, webhook latency (~200-500ms), format coupling
- Enables: richer signal data (full score breakdown, all indicator values, chain data)
- Enables: direct signal replay for backtesting
- Risk: requires changes to all 4 trading owlets. Phase 1 proves the scoring engine first.

---

## 5. Docker Compose Integration

### Container Definition

```yaml
owlet-sourcing:
    <<: *bot-common
    container_name: owlet-sourcing
    command: ["python", "-m", "options_owl.sourcing.scanner"]
    environment:
      - PYTHONUNBUFFERED=1
      - SOURCING_MODE=true
      - SCAN_INTERVAL_SECONDS=180
      - SCORE_THRESHOLD=60
      # Data sources
      - ENABLE_SOURCE_HARVESTER_CANDLES=true
      - ENABLE_SOURCE_TWELVE_DATA=false
      - ENABLE_SOURCE_POLYGON_OPTIONS=true
      - ENABLE_SOURCE_POLYGON_NEWS=false
      - ENABLE_SOURCE_UNUSUAL_WHALES=false
      - ENABLE_SOURCE_GROK_AI=false
      # API keys
      - POLYGON_API_KEY=${POLYGON_KEY_SOURCING}
      # Output
      - DISCORD_WEBHOOK_BULLISH=${SOURCING_WEBHOOK_BULLISH}
      - DISCORD_WEBHOOK_BEARISH=${SOURCING_WEBHOOK_BEARISH}
      # Ticker watchlist
      - SOURCING_TICKERS=SPY,QQQ,NVDA,TSLA,META,AAPL,AMZN,GOOGL,MSFT,AMD,MSTR,PLTR,AVGO
      # State
      - AGENT_ID=owlet_sourcing
      - SHARED_CANDLE_DB=/app/shared_harvester/options_data.db
    volumes:
      - ./journal/owlet-sourcing:/app/journal:rw
      - ./journal/owlet-harvester:/app/shared_harvester:rw  # WAL requires :rw
    deploy:
      resources:
        limits:
          memory: 512M  # No Webull SDK, no Discord bot — lighter than trading owlets
```

### Volume Layout

```
journal/
+-- owlet-sourcing/
|   +-- state.db          Cooldowns, counters, circuit breaker (WAL mode)
|   +-- logs/
|   |   +-- sourcing_YYYY-MM-DD.log    Daily human-readable (DEBUG)
|   |   +-- sourcing.json              JSON structured (INFO, 50MB rotation)
|   +-- heartbeat         Docker healthcheck file
|
+-- owlet-harvester/      (read by owlet-sourcing, written by harvester)
    +-- options_data.db   5-min candles + options snapshots (~7GB, WAL mode)
```

### Relationship to Existing Containers

owlet-sourcing has no dependency on the 4 trading owlets. It does not need Webull credentials, a Discord bot token, or any trading configuration. It is a pure signal emitter.

It depends only on:
1. **owlet-harvester** -- for candle data (via shared volume mount)
2. **Polygon API** -- for options chain validation
3. **Discord webhook** -- for signal delivery (Phase 1 only)

It can start, stop, and restart independently of all other containers. If it crashes, the trading owlets continue operating on whatever signals were last received. There is no coupling.

---

## 6. Monitoring and Observability

### Logging

Follows the same loguru pattern as all other owlets:

| Output | Level | Rotation | Retention | Purpose |
|---|---|---|---|---|
| `journal/owlet-sourcing/logs/sourcing_YYYY-MM-DD.log` | DEBUG | Daily | 90 days | Human-readable, full detail |
| `journal/owlet-sourcing/logs/sourcing.json` | INFO | 50 MB | 90 days | Machine-parseable structured log |
| Docker `json-file` driver | INFO | 10 MB x 5 | Ephemeral | Live tailing only (wiped on rebuild) |

**Critical:** Persisted log files are the source of truth, not `docker compose logs`.

### Scoring Audit Table

Every scoring computation is logged to `state.db:scoring_audit`, regardless of whether the signal passes filters:

```sql
CREATE TABLE scoring_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_time TEXT NOT NULL,           -- ISO 8601 UTC
    ticker TEXT NOT NULL,
    direction TEXT,                    -- BULLISH / BEARISH / NULL (no signal)
    score_total INTEGER,
    score_direction INTEGER,           -- Tier 1
    score_timing INTEGER,              -- Tier 2
    score_amplifiers INTEGER,          -- Tier 3
    score_adjustments INTEGER,         -- Tier 4
    score_calibration INTEGER,         -- Tier 5
    reasons TEXT,                      -- JSON array of human-readable reasons
    filter_result TEXT,                -- PASS / COOLDOWN / QUALITY / VETO / CHAIN_FAIL
    filter_reason TEXT,                -- Why it was filtered (if applicable)
    options_strike REAL,
    options_premium REAL,
    options_spread_pct REAL,
    scan_duration_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
```

This table enables:
- Post-hoc analysis of scoring quality ("what did we miss?")
- Shadow comparison with N8N outputs during parallel run
- Threshold tuning (lower/raise `SCORE_THRESHOLD` based on audit data)
- Backtesting with historical harvester candles

### Shadow Testing (Phase 1 Transition)

During the parallel run period, both N8N and owlet-sourcing emit signals. A comparison script diffs the outputs:

```
For each 3-minute scan window:
  - Did N8N fire a signal?  Did owlet-sourcing fire?
  - If both fired: compare ticker, direction, approximate score
  - If only one fired: log the discrepancy with full scoring audit
  - Track agreement rate over days
```

Target: > 90% agreement on signal/no-signal decisions before disabling N8N. Score magnitudes will differ (different scoring scales) but directional agreement is what matters.

### Health Metrics (logged every scan)

```
SCAN_SUMMARY: duration=12.3s tickers=13 signals=2 filtered=4 skipped=7
  emitted: NVDA BULLISH (score=78) TSLA BEARISH (score=64)
  filtered: SPY (cooldown) AMD (quality<60) MSTR (chain_fail) META (penalty_veto)
  skipped: AAPL AMZN GOOGL MSFT PLTR AVGO QQQ (no signal)
```

### Alerts

Discord DM alerts to the operator for:
- Scan failure (exception in main loop, 2 consecutive failures)
- Harvester data stale (no candles updated in 15+ minutes)
- API error rate (> 50% of Polygon calls failing in a scan)
- Zero signals for 2+ hours during market hours (possible miscalibration)

---

## 7. Error Handling and Resilience

### Per-Source Graceful Degradation

Each data source failure is isolated. The scanner never crashes because one source is unavailable.

```
Source Failure Behavior:
+---------------------------+-----------------------------------------------+
| Failure                   | Behavior                                      |
+---------------------------+-----------------------------------------------+
| Harvester DB unreachable  | Fall back to Twelve Data (if enabled).        |
|                           | If Twelve Data also disabled/fails, skip scan.|
| Harvester data stale      | Fall back to Twelve Data. Log warning.         |
| Twelve Data API error     | Skip ticker for this scan. Log warning.        |
| Polygon options API error | Skip signal emission (can't validate chain).  |
| Polygon news API error    | Score without news component (0 for that sub).|
| Unusual Whales API error  | Score without flow component.                  |
| Grok AI API error         | Score without AI component.                    |
| SEC EDGAR API error       | Score without insider component (0 for sub).   |
| StockTwits API error      | Score without sentiment component.             |
| Capitol Trades API error  | Fall back to UW Congress data.                 |
| ML model file missing     | Fall back to hand-tuned scoring (no ML gate).  |
| ML inference error        | Fall back to hand-tuned scoring. Log warning.  |
+---------------------------+-----------------------------------------------+
```

### Database Resilience

- WAL mode on all SQLite databases
- `PRAGMA busy_timeout = 5000` (5-second retry on lock contention)
- All DB operations wrapped in `asyncio.wait_for(..., timeout=15)` to prevent event loop freeze
- State DB is small (< 100 MB) and append-heavy -- no risk of the 7GB copy problem

### Crash Recovery

- Docker `restart: always` handles process crashes
- State in SQLite survives restarts -- cooldowns, counters, circuit breaker are intact
- First scan after restart: full scan of all tickers (no stale cache to worry about)
- Heartbeat file written every scan cycle -- Docker healthcheck catches stuck loops

### Market Hours Guard

```
Scan windows (Eastern Time):
  First scan:  9:33 AM  (3 min after open -- let opening volatility settle)
  Last scan:   3:57 PM  (3 min before close -- final signals still actionable)
  Weekends:    No scans (sleep until Monday)
  Holidays:    No scans (market calendar check)
```

If the container starts outside market hours, it logs "waiting for market open" and sleeps until the next valid scan window. No crash-loop on weekends (unlike trading owlets which crash on stale Polygon quotes).

---

## 8. Security Considerations

| Concern | Mitigation |
|---|---|
| API keys | Stored in `.env` on droplet, injected via `env_file` + `environment` in docker-compose. Never in code or images. |
| Discord webhooks | Write-only URLs. Cannot read channel history or messages. Separate URLs for bullish/bearish channels. |
| SQLite files | Inside Docker volume, not exposed to network. No authentication needed (single-tenant, local only). |
| Network exposure | Container has no exposed ports. Outbound-only: HTTPS to APIs, HTTPS to Discord webhook. |
| Secrets in logs | API keys, webhook URLs, and credentials are never logged. Log sanitization applied. |
| Image contents | `.dockerignore` excludes `.env`, `journal/`, and credentials from the build context. |

---

## 9. Performance Requirements

### Scan Cycle Budget

```
Total budget per scan cycle: < 60 seconds (scan interval is 180s)
Target for typical cycle:    < 30 seconds

Breakdown (13 tickers):
+-------------------------------+-----------+--------+
| Operation                     | Per-tick  | Total  |
+-------------------------------+-----------+--------+
| Read candles from harvester   | ~50ms     | 0.7s   |
| Compute indicators            | ~10ms     | 0.1s   |
| Check cooldowns (DB read)     | ~5ms      | 0.1s   |
| Run scoring engine            | ~5ms      | 0.1s   |
| Apply filters                 | ~2ms      | 0.03s  |
+-------------------------------+-----------+--------+
| Subtotal (all tickers)        |           | ~1s    |
+-------------------------------+-----------+--------+

Signals that pass filters (typically 1-3 per cycle):
+-------------------------------+-----------+--------+
| Fetch options chain (Polygon) | ~2-5s     | 5-15s  |
| Validate chain                | ~2ms      | 0.01s  |
| Send Discord webhook          | ~200ms    | 0.6s   |
| Write audit log               | ~5ms      | 0.02s  |
+-------------------------------+-----------+--------+
| Subtotal (signal emission)    |           | ~6-16s |
+-------------------------------+-----------+--------+

Typical total: 7-17s per scan cycle
Worst case (all 13 tickers pass): ~50s (still within budget)
```

### Resource Limits

| Resource | Limit | Rationale |
|---|---|---|
| Memory | 512 MB | No Webull SDK, no Discord bot, no LightGBM models. Candle data read on-demand, not cached in bulk. |
| CPU | Shared (no limit) | Indicator computation is trivial. No ML inference. |
| Disk (state.db) | < 100 MB | Audit table pruned at 30-day retention. |
| Network | Outbound only | HTTPS to Polygon, Discord. No inbound ports. |

### Concurrency

The scanner is single-threaded async (asyncio). Within a scan cycle, ticker processing is sequential to avoid Polygon rate limits. Options chain fetches for passing signals are also sequential (typically 1-3 per cycle, not worth parallelizing).

If Polygon rate limits become an issue with more tickers, add a `SCAN_BATCH_SIZE` setting to process tickers in batches with a delay between batches.

---

## 10. Testing Strategy

### Unit Tests

Every scoring tier, indicator computation, and filter is a pure function. Unit tests cover:

```
tests/sourcing/
+-- test_indicator_engine.py    Verify EMA, RSI, MACD, BB against known values
+-- test_scoring_direction.py   Tier 1 with crafted indicator inputs
+-- test_scoring_timing.py      Tier 2 with crafted indicator inputs
+-- test_scoring_amplifiers.py  Tier 3 with/without optional sources
+-- test_scoring_adjustments.py Tier 4 penalty scenarios
+-- test_scoring_calibration.py Tier 5 historical edge lookup
+-- test_quality_gate.py        Threshold and multi-tier contribution checks
+-- test_cooldown_manager.py    Cooldown enforcement and expiry
+-- test_penalty_veto.py        Circuit breaker, daily caps
+-- test_options_validator.py   Spread, premium cap, volume checks
+-- test_discord_webhook.py     Embed formatting, correct webhook URL selection
+-- test_state_manager.py       SQLite state read/write/prune
```

**Indicator validation:** One-time test that compares local indicator outputs against Twelve Data API outputs for the same candle series. Ensures EMA/RSI/MACD/BB computations are correct. Run manually during development, not in CI.

### Integration Tests

Full scan cycle with mocked external APIs:

```python
# Mock harvester DB with known candle data
# Mock Polygon options API with known chain
# Run scanner.scan_once()
# Assert: correct signals emitted, correct signals filtered
# Assert: audit log contains all 13 ticker evaluations
# Assert: cooldowns updated for emitted signals
# Assert: scan completes within time budget
```

### Shadow Tests

During Phase 1 parallel run:

```
1. Capture N8N Discord output (via webhook logger or Discord bot)
2. Capture owlet-sourcing Discord output (audit table)
3. Diff script compares:
   - Signal agreement rate (same ticker + direction within same 3-min window)
   - False negatives (N8N fired, sourcing didn't -- and the trade was profitable)
   - False positives (sourcing fired, N8N didn't -- would the trade have worked?)
4. Report daily: agreement %, missed winners, avoided losers
```

### Backtest Harness

The scoring engine can be run offline against historical harvester candle data:

```python
# Load candles from harvester DB for date range
# For each 3-min window, run scoring engine
# Compare emitted signals against actual trade outcomes
# Compute: precision, recall, expected P&L
```

This reuses the same pure scoring functions as production. No mocking needed -- just feed historical candles to the indicator engine and scoring tiers.

---

## Appendix A: Ticker Watchlist

Initial watchlist (matches current N8N scanner):

| Ticker | Category | 0DTE Schedule |
|---|---|---|
| SPY | INDEX | Daily |
| QQQ | INDEX | Daily |
| NVDA | HIGH_VOL | Mon/Wed/Fri |
| TSLA | HIGH_VOL | Mon/Wed/Fri |
| META | HIGH_VOL | Mon/Wed/Fri |
| AAPL | STANDARD | Mon/Wed/Fri |
| AMZN | STANDARD | Mon/Wed/Fri |
| GOOGL | STANDARD | Mon/Wed/Fri |
| MSFT | STANDARD | Mon/Wed/Fri |
| AMD | HIGH_VOL | Friday only (weekly) |
| MSTR | HIGH_VOL | Friday only (weekly) |
| PLTR | HIGH_VOL | Friday only (weekly) |
| AVGO | HIGH_VOL | Mon/Wed/Fri |

The watchlist is env-driven (`SOURCING_TICKERS`). Adding a ticker requires only a docker-compose change and a rebuild.

## Appendix B: Scoring Engine Range Mapping

```
N8N Score (0-177)  -->  owlet-sourcing Score (0-100)

Tier 1: Direction Confidence    0-40 pts   (40% weight)
Tier 2: Timing Quality          0-30 pts   (30% weight)
Tier 3: Edge Amplifiers          0-20 pts   (20% weight)
Tier 4: Risk Adjustments       -10-0  pts   (penalty only)
Tier 5: Calibration Bonus        0-10 pts   (10% weight)
                               ----------
Theoretical range:             -10 to 100
Practical range:                 0 to 100   (floor at 0)

Trading owlet compatibility:
  owlet-sourcing score 60  ~=  N8N score 78 (current entry threshold)
  The trading owlets' signal parser will map the new score range
  to their existing 78-177 scale during Phase 1 (Discord parsing).
  Phase 2 (direct DB) uses the 0-100 score natively.
```

## Appendix C: State Database Schema

```sql
-- Cooldown tracking (per-ticker, per-direction)
CREATE TABLE cooldowns (
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,       -- BULLISH / BEARISH
    last_alert_at TEXT NOT NULL,   -- ISO 8601 UTC
    cooldown_minutes INTEGER NOT NULL DEFAULT 90,
    PRIMARY KEY (ticker, direction)
);

-- Daily/weekly alert counters
CREATE TABLE alert_counters (
    date TEXT NOT NULL,             -- YYYY-MM-DD
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, ticker, direction)
);

-- Circuit breaker state
CREATE TABLE circuit_breaker (
    id INTEGER PRIMARY KEY DEFAULT 1,
    is_tripped INTEGER NOT NULL DEFAULT 0,
    trip_count INTEGER NOT NULL DEFAULT 0,
    tripped_at TEXT,
    reset_at TEXT,
    reason TEXT
);

-- Scoring audit (append-only, pruned at 30 days)
-- Schema defined in Section 6 above
```
