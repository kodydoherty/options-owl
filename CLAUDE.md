# OptionsOwl

0DTE options trading system: Discord signal collector → signal parser/scorer → risk pipeline → Webull executor.

## Quick Reference

```bash
# Local development
pip install -e ".[dev]"
ruff check options_owl/ tests/
pytest tests/                         # 1621 tests, ~5min
python -m options_owl.main            # run locally (uses .env)

# Deploy to droplet (ALWAYS use rebuild.sh — never just rsync without rebuilding)
./scripts/rebuild.sh                  # sync + rebuild ALL agents (no cache)
./scripts/rebuild.sh owlet-kody       # sync + rebuild just one agent

# Droplet management (NEVER run docker locally)
ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145
cd /root/options-owl

docker compose ps                     # check all agents
docker compose logs owlet-kody --tail 50 -f   # follow logs (EPHEMERAL — wiped on rebuild)
docker compose restart owlet-kody     # restart one agent (uses existing image)
docker compose up -d                  # start all (uses existing images)

# Trade P&L analysis (CORRECT math for DCA + scaleout)
python scripts/trade-pnl.py --droplet              # last 20 trade families
python scripts/trade-pnl.py --droplet NVDA          # specific ticker
python scripts/trade-pnl.py --droplet --id 226      # specific trade family detail
python scripts/trade-pnl.py --droplet --date 2026-05-19  # specific date
python scripts/trade-pnl.py --droplet --webull-only  # all Webull trades
python scripts/trade-pnl.py --droplet --detail       # verbose per-trade breakdown

# Trade investigation (uses PERSISTED logs — survives rebuilds)
./scripts/trade-log.sh                # today's trade events from DB
./scripts/trade-log.sh trades         # all trades with WEBULL/PAPER status
./scripts/trade-log.sh webull         # Webull order logs for today
./scripts/trade-log.sh logs           # full trade lifecycle logs for today
./scripts/trade-log.sh 2026-04-23     # specific date's trade events
```

## Production Architecture

### Agents (docker-compose.yml)

| Container | Purpose | Portfolio | Key Overrides |
|---|---|---|---|
| `owlet-kody` | Kody's live trading bot | $23,000 | Own Webull creds, PAPER_TRADE=false |
| `owlet-adam` | Adam's live trading bot | $2,500 | Own Webull + Polygon creds |
| `owlet-vinny` | Vinny's live trading bot | $500 | Own Webull + Polygon creds |
| `owlet-yank` | Yank's live trading bot | $500 | Own Webull + Polygon creds |
| `owlet-harvester` | Data collection only | N/A | Captures options chain snapshots for ML/backtesting |

Each trading bot runs the same code with different portfolio sizes and credentials. The `.env` file holds shared config; `docker-compose.yml` overrides per-bot values (PORTFOLIO_SIZE, credentials, PAPER_TRADE=false).

### Data Flow

```
Discord (Neverland Pirates) → discord_collector.py → signal parser/scorer
  → smart entry (verify live premium + resolve expiry) → entry pipeline (18 gates)
  → dip-confirm gate (optional: wait for premium dip + support confirmation)
  → paper_trader.py (DB record + Webull order placement)
  → position_monitor.py (5s poll loop) → V5 FSM exit engine (10 gates)
  → close via Webull API (with auto-reconnect on stale connection)
```

### Databases (SQLite WAL mode, per-agent in journal/)

- `journal/owlet-{name}/raw_messages.db` — Discord messages + parsed trade signals + paper trades + trade_events audit log
- `journal/owlet-harvester/options_data.db` — Polygon options snapshots + stock candles (shared across all bots via volume mount, ~7GB)
- `journal/models/` — Pre-trained LightGBM models (per-ticker + generic)

**CRITICAL: Shared harvester DB uses WAL mode.** The harvester writes continuously while all 4 trading bots read candle data. WAL mode (`PRAGMA journal_mode = WAL`) allows concurrent readers without blocking. All bot volume mounts must be `:rw` (not `:ro`) because WAL creates `-shm` and `-wal` sidecar files that need write access even for readers.

### Logging (persisted in journal/, survives rebuilds)

Logs are written to **two places**:
1. **`docker compose logs`** — goes to Docker's json-file driver. **WIPED on every rebuild.** Good for live tailing, useless for post-mortem.
2. **`journal/owlet-{name}/logs/`** — loguru file output. **PERSISTED across rebuilds** via volume mounts. This is the source of truth.

Log files in `journal/owlet-kody/logs/`:
- `options_owl_YYYY-MM-DD.log` — daily human-readable log (DEBUG level, rotated daily, kept 90 days)
- `options_owl.json` — JSON structured log (INFO level, rotated at 50MB, kept 90 days)

**CRITICAL: Always use the persisted log files for troubleshooting, not `docker compose logs`.**

```bash
# Read persisted logs on droplet
ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145
cat /root/options-owl/journal/owlet-kody/logs/options_owl_2026-04-23.log

# Or use the helper script locally
./scripts/trade-log.sh logs 2026-04-23
./scripts/trade-log.sh webull 2026-04-23
```

### Trade Events Audit Table

Every trade lifecycle decision is persisted in the `trade_events` table in `raw_messages.db`:

| event_type | What it captures |
|---|---|
| `smart_entry` | Premium lookup result, strike, expiry, live vs signal premium |
| `rejected` | Why a trade was blocked (smart entry, pipeline gate, no chain) |
| `pipeline_approved` | Signal passed all entry gates — score, premium, strike |
| `pipeline_rejected` | Which entry gates failed and why |
| `webull_filled` | Webull order accepted — order_id, fill price |
| `webull_rejected` | Webull API rejected the order — error message, params |
| `webull_error` | Exception during Webull API call |

Query trade events:
```bash
# From droplet
sqlite3 journal/owlet-kody/raw_messages.db "SELECT * FROM trade_events WHERE date(created_at) = '2026-04-23' ORDER BY id"

# From local
./scripts/trade-log.sh 2026-04-23
```

## Code Structure

```
options_owl/
├── collectors/
│   ├── discord_collector.py    # Discord bot, on_ready (guarded), message ingestion
│   ├── market_data_stream.py   # Polygon WS + REST price/premium feeds
│   ├── polygon_options.py      # Polygon options snapshot API
│   ├── candle_cache.py         # Multi-TF candle data (5m/15m/30m/1h/4h) + indicators + ENRG
│   └── candle_collector.py    # Reads stock candles from shared harvester DB (WAL mode)
├── signals/                    # Signal parsing and scoring engine
├── risk/
│   ├── pipeline.py             # Entry pipeline (18 gates)
│   ├── vinny_strategy.py       # Position sizing (score-to-contracts)
│   ├── ml_exit.py              # LightGBM sell timing (disabled)
│   └── exit_v5/               # V5 FSM exit engine (active)
│       ├── fsm.py              # ExitFSM class, gate orchestration, state machine
│       ├── gates.py            # Individual gate functions (pure, testable)
│       ├── config.py           # V5Config, per-ticker configs, category classification
│       ├── monitor_bridge.py   # Bridge between position_monitor and V5 FSM
│       ├── defensive.py        # Bid disappearance detection
│       └── types.py            # ExitAction, ExitReason (shared types)
├── execution/
│   ├── paper_trader.py         # Trade DB, open/close/partial, Webull order, trade_events audit
│   ├── position_monitor.py     # Main monitoring loop (5s), premium fetching, exit decisions
│   ├── webull_executor.py      # Webull API wrapper (auth, orders, fills)
│   └── alerts.py               # Discord DM alerts for critical events
├── config/
│   └── settings.py             # All config via pydantic-settings (env + .env file)
├── models/                     # Pydantic data models
├── main.py                     # Entry point, Polygon freshness check, retry loop, logging setup
└── harvester.py                # Standalone data harvester (options chain snapshots)
```

## Trade Lifecycle (Entry Path)

Understanding the full entry path is critical for debugging "why didn't this trade execute on Webull":

```
1. Discord signal received → discord_collector.py on_message()
2. Signal parsed + scored → signals/ module
3. Premium lookup → _fill_missing_premium() if ATM premium missing/suspicious
4. Smart entry → _verify_live_premium()
   a. Try Polygon quote for today's expiry
   b. If 0DTE and no contract exists → try next 2 business days (near-expiry fallback)
   c. Fall back to yfinance chain (also tries multiple expiry dates)
   d. If no chain found in LIVE mode → BLOCK trade (contract doesn't exist)
   e. Compare live vs signal premium, reject if deviation > 75%
5. Entry pipeline → run_entry_pipeline() — 18 gates, ALL logged at INFO
6. Position sizing → score_to_contracts() in vinny_strategy.py
   - target_per_trade = balance × MAX_PORTFOLIO_RISK_PCT / MAX_CONCURRENT
   - Flat 85% budget multiplier for all scores >= 78 (scores don't predict outcomes)
   - scaled_target = target_per_trade × 0.85, contracts = scaled_target / cost_per_contract
   - Capped by MAX_POSITION_PCT (no fixed contract cap — scales with portfolio)
7. Dip-confirm gate (if ENABLE_DIP_CONFIRM=true) → _wait_for_entry_confirmation()
   - Monitors live premium via WebSocket for up to 60s
   - Waits for premium to dip (stabilize or drop) before buying
   - Checks underlying support level (5m candle lows) and VWAP
   - If premium upticks from dip AND price above support → BUY
   - If timeout (60s) → buy anyway (signal was already approved)
   - Typically saves 2-5% on entry price vs immediate buy
8. Paper trade created in DB
9. Webull order placed → _place_webull_order()
   - If webull_executor is None → logged as "NO WEBULL EXECUTOR — trade is PAPER ONLY"
   - If Webull rejects → logged as "WEBULL ENTRY FAILED" with error
   - If Webull fills → order_id saved to DB
10. All decisions logged to trade_events table
```

## V5 Exit Engine (EXIT_ENGINE=v5) — Category-Aware FSM

**This is the active exit engine deployed to ALL owlets.** Code in `options_owl/risk/exit_v5/`.

The V5 FSM runs every 5 seconds per open trade. First gate to trigger wins. States are informational (GRACE/DEVELOPING/TRAILING).

### Gate Priority (first match exits)

| # | Gate | What it does | Key Thresholds |
|---|---|---|---|
| 1 | `eod_cutoff` | 0DTE only, 15min before close | 15min |
| 2 | `bid_disappearance` | No buyers for 30s | 30s zero bid |
| — | **5min grace** | Skip all gates below (BUT backstop still fires) | 5min (TSLA/QQQ: 8min) |
| 3 | `profit_target` | Index 0DTE: lock gains at 30% | SPY/QQQ/IWM only |
| 3.5 | `breakeven_ratchet` | V6: once +20%, floor = entry price | `ENABLE_V6_BREAKEVEN_RATCHET=true` |
| 3.7 | `scaleout` | V6: sell 1/3 at +20% (one-shot) | `ENABLE_V6_SCALEOUT=true` |
| 4 | `scalp_trail` | Peaked +20%, faded <60% of peak | DTE-aware: 0DTE strict, multi-day patient |
| 5 | `checkpoint_cut` | 0DTE: down 30% AND underlying against 0.5% | 0DTE only |
| 6 | `graduated_stop` | Tight stop if underlying against, backstop otherwise | 0DTE: 35%/65%, multi-day: 52%/75% |
| 7 | `soft_trail` | 15-50% peak band, keep 60% of gain (70% MSTR/NVDA) | floor = entry + 60-70% of (peak - entry) |
| 8 | `adaptive_trail` | Category-aware trailing stop (primary exit) | See tiers below |
| 9 | `theta_exit` | 0DTE: 120min+down 30%; multi-day: 180min+down 15% | Cuts stale losers |

### Grace Period Backstop (CRITICAL)

Grace period (5min) does NOT protect catastrophic losses. The backstop fires DURING grace:
- 0DTE: backstop at -65%
- Multi-day: backstop at -75%

This prevents the old bug where a -95% trade sat untouched for 5 minutes.

### Category-Aware Adaptive Trail Tiers (gate #8)

Tickers classified as HIGH_VOL, INDEX, or STANDARD. Each gets different trail widths.

| Category | Tickers | Active (40%+) | Runner (150%+) | Moonshot (400%+) |
|---|---|---|---|---|
| **HIGH_VOL** | MSTR, AMD, TSLA, NVDA, AVGO, META, COIN, SMCI, PLTR | 50% drop | 55% drop | 35% drop |
| **INDEX** | SPY, QQQ, IWM, DIA, XLF, XLK | 35% drop | 40% drop | 25% drop |
| **STANDARD** | Everything else | 35% drop | 40% drop | 25% drop |

**Per-ticker overrides** (V6, `ENABLE_V6_PER_TICKER_CONFIG=true`):
- MSTR: TIGHT+QUICK — 35% active trail, 15% scalp threshold, 70% soft keep
- NVDA/AVGO/MSFT: EARLY_PROFIT — 20% profit target, 70% soft keep
- TSLA/QQQ: LONG_GRACE — 8min grace period
- GOOGL/IWM: WIDE_STOP — 45%/75% stops, 40% checkpoint
- META/AAPL: DEFENSIVE — 25%/50% stops, 90min theta bleed
- AMZN: TIGHT_TRAIL — tighter adaptive tiers

### V6 Enhancements (all gated behind ENABLE_V6_* settings)

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_V6_BREAKEVEN_RATCHET` | true | Once +20%, stop floor = entry price. Cannot go negative. |
| `ENABLE_V6_SCALEOUT` | true | Sell 1/3 contracts at +20% (one-shot, min 3 contracts) |
| `ENABLE_V6_2PM_TIGHTEN` | true | After 2PM ET, tighten adaptive trails by 30% |
| `ENABLE_V6_PER_TICKER_CONFIG` | true | Use per-ticker optimal FSM configs |
| `ENABLE_V6_PREMIUM_CAP` | true | Block entries with premium > tiered cap ($6/$7/$9) |
| `ENABLE_V6_SPREAD_GATE` | true | Block entries with bid-ask spread > 40% |
| `ENABLE_V6_DCA` | true | Auto-double position when premium dips 15-35% from entry |
| `ENABLE_V6_EARLY_POP_GATE` | true | Block entries on initial spike (wait for pullback) |
| `ENABLE_DIP_CONFIRM` | true | Wait for premium dip + support confirmation before buying |
| `ENABLE_SUPABASE_BRAIN` | true | Sync trade data to Supabase for cross-agent analytics |

### DTE Awareness

V5 is fully DTE-aware — 0DTE and multi-day trades get different treatment:

| Parameter | 0DTE | Multi-day |
|---|---|---|
| Tight stop (underlying against) | 35% | 52% |
| Backstop (underlying neutral) | 65% | 75% |
| Checkpoint cut | Active | Disabled |
| Scalp trail | Exit if underlying NOT confirming | Exit only if underlying AGAINST |
| Theta exit | 120min + down 30% | 180min + down 15% |

### How to Check What's Protecting a Trade

```bash
# Live: tail the monitor log for a specific trade
grep '#120 MSTR' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log | tail -5
# Output: EXIT_FSM: #120 MSTR state=TRAILING HOLD prem=$4.90 (+65.3%) peak=$5.30

# Check which adaptive tier is active:
# peak gain >= 40% → active tier
# peak gain >= 150% → runner tier
# peak gain >= 400% → moonshot tier
# Trail fires when drop_from_peak >= tier's trail_width
```

### Exit Source Tracking (manual vs AI)

The `exit_source` column in `paper_trades` tracks who closed the trade:

| Value | Meaning |
|---|---|
| `ai` (default) | Bot closed via V5 FSM gate |
| `manual` | User sold on Webull manually; bot detected position was gone |

**How manual detection works:** When the bot tries to sell but Webull returns "no position" after 10 retries, the position was already sold/expired. The bot marks `exit_source='manual'` and logs a `manual_close_detected` trade event. The `exit_premium` in the DB is approximate (market price at detection time), NOT the user's actual fill.

**For backtesting:** Filter with `WHERE exit_source = 'ai' OR exit_source IS NULL` to exclude manual trades.

```bash
# Show manual vs AI exits
sqlite3 journal/owlet-kody/raw_messages.db "
  SELECT id, ticker, exit_reason, exit_source,
    printf('\$%.2f', pnl_dollars) as pnl
  FROM paper_trades WHERE status='closed'
  ORDER BY id DESC LIMIT 20
" -column -header
```

## Key Settings (.env)

### Critical Settings (must be correct)

```bash
PAPER_TRADE=true              # .env default; docker-compose overrides to false for live bots
WEBULL_KILL_SWITCH=true       # .env default; docker-compose overrides to false per bot
EXIT_ENGINE=v5                # V5 FSM exit engine (category-aware, DTE-aware)
ENABLE_PORTFOLIO_SYNC=true    # Auto-pull live Webull balance daily

# V6 enhancements (all enabled in production)
ENABLE_V6_PER_TICKER_CONFIG=true    # Per-ticker optimal FSM configs
ENABLE_V6_BREAKEVEN_RATCHET=true    # Once +20%, floor = entry price
ENABLE_V6_2PM_TIGHTEN=true         # Tighten trails 30% after 2PM ET
ENABLE_V6_SCALEOUT=true            # Sell 1/3 at +20% (one-shot)
ENABLE_V6_PREMIUM_CAP=true         # Block entries > tiered cap ($6 base, $7 score 120+, $9 score 150+)
ENABLE_V6_SPREAD_GATE=true         # Block entries with wide bid-ask spread
```

### Per-Bot Overrides (in docker-compose.yml)

Each bot overrides these from docker-compose.yml `environment:` section:
- `PORTFOLIO_SIZE` — $23000 (kody), $4685 (adam), $3123 (vinny), $3600 (yank)
- `PAPER_TRADE=false` — enables live Webull execution
- `WEBULL_KILL_SWITCH=false` — allows orders
- `MAX_CONCURRENT=5` — max simultaneous trades
- `MAX_POSITION_PCT=15` — max % of portfolio per trade
- `MAX_PORTFOLIO_RISK_PCT=75` — total deployable capital as % of portfolio
- `MAX_LOSS_PER_TRADE_PCT=25` — max loss per single trade
- Webull credentials (`WEBULL_APP_KEY`, `WEBULL_APP_SECRET`, `WEBULL_ACCOUNT_ID`)
- Polygon API key (per-user for rate limits)

### Position Sizing (Flat Budget)

Flat sizing — all trades above score 78 get equal allocation. Backtested: scores don't predict outcomes, so tiered sizing was removed.

```
target_per_trade = balance × MAX_PORTFOLIO_RISK_PCT / MAX_CONCURRENT
scaled_target = target_per_trade × 0.85       # flat 85% for all qualifying trades
contracts = int(scaled_target / cost_per_contract)
final = max(1, min(contracts, position_cap))   # capped by MAX_POSITION_PCT
```

| Score | Budget Mult | Result |
|---|---|---|
| >= 78 | 85% | Equal allocation for all qualifying trades |
| < 78 | 0 (rejected) | Not traded |

For Kody's $23K portfolio: deployable = $17,250 (75%), per-slot = $4,312, 85% = $3,665, position cap = $3,450 (15%).
No fixed contract cap or liquidity cap — sizing scales with portfolio.

## Troubleshooting

### Log Access — ALWAYS Use Persisted Logs

**`docker compose logs` is ephemeral — wiped on every rebuild.** For troubleshooting, always use the persisted log files:

```bash
# Quick: use the helper script
./scripts/trade-log.sh trades         # all trades with WEBULL vs PAPER status
./scripts/trade-log.sh webull         # today's Webull order attempts/fills/errors
./scripts/trade-log.sh logs           # today's full trade lifecycle
./scripts/trade-log.sh 2026-04-23     # trade events for specific date

# Direct: SSH to droplet and read log files
ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145
# Daily logs (DEBUG level, human-readable)
cat /root/options-owl/journal/owlet-kody/logs/options_owl_2026-04-23.log
# Grep for specific events
grep 'WEBULL ORDER ERROR' /root/options-owl/journal/owlet-kody/logs/options_owl_2026-04-23.log
grep 'TradeLifecycle' /root/options-owl/journal/owlet-kody/logs/options_owl_2026-04-23.log
grep 'SIZING' /root/options-owl/journal/owlet-kody/logs/options_owl_2026-04-23.log
grep 'ENRG' /root/options-owl/journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log

# Query trade events audit table
sqlite3 /root/options-owl/journal/owlet-kody/raw_messages.db \
  "SELECT * FROM trade_events WHERE date(created_at) = '2026-04-23' ORDER BY id"
```

### Trades not reaching Webull (paper-only)

This is the #1 issue. Check the `webull_order_id` column in `paper_trades`:

```bash
# Show which trades hit Webull vs paper-only
sqlite3 journal/owlet-kody/raw_messages.db "
  SELECT id, ticker, date(opened_at), time(opened_at),
    CASE WHEN webull_order_id IS NOT NULL THEN 'WEBULL' ELSE 'PAPER' END as exec_type,
    exit_reason, printf('\$%.2f', pnl_dollars) as pnl
  FROM paper_trades WHERE status='closed' ORDER BY id DESC LIMIT 20
" -column -header
```

**Common causes and how to diagnose:**

#### 1. Contract doesn't exist for today's expiry (MOST COMMON)
**Symptom:** Webull returns `HTTP 417 OAUTH_OPENAPI_PARAM_ERR: Parameter error, invalid market,symbol,instrument_type,option_type,strike_price`

**Why:** Not all tickers have daily 0DTE options. The actual schedule is:
- **SPY, QQQ** — daily 0DTE (Mon-Fri)
- **NVDA, TSLA, META, AAPL, AMZN, GOOGL, MSFT, AVGO** — 0DTE on Mon/Wed/Fri only. On Tue/Thu they have 2-day contracts (higher premium)
- **AMD, PLTR, MSTR** — weekly only (Friday expiry). 0DTE only on Fridays. Mon-Thu contracts expire Friday (higher premium, decreasing through the week)
- **All tickers** have 0DTE on Fridays (golden day)

**Fix (updated 2026-04-26):** Smart entry now uses per-ticker expiry schedules. MWF tickers on Tue/Thu try the next day's contract. Weekly tickers try Friday. Unknown tickers try next 2 business days + Friday as fallback.

#### 2. Webull stale connection (FIXED 2026-05-07)
**Symptom:** Log shows `ValueError: no active connection` on order placement.
**Why:** Webull SDK HTTP session drops after hours of idle time (no signals during slow market).
**Fix:** `webull_executor.py` has `_reconnect()` method — tears down stale clients, reinitializes, retries the order. Applied to `place_option_order`, `get_account_info`, `get_account_balance`.

#### 3. Webull executor is None (no live connection)
**Symptom:** Log shows `NO WEBULL EXECUTOR — trade is PAPER ONLY`
**Fix:** Webull init retries 3x with backoff. `on_ready()` guarded against re-entry.

#### 4. Smart entry blocked the trade (no option chain found)
**Symptom:** Trade events show `smart_entry_blocked: no_chain_blocked_live`

#### 5. Discord reconnect wiped Webull connection (FIXED 2026-04-23)
**Fix:** `_initialized` guard prevents re-initialization.

#### 6. User sold manually on Webull (bot can't find position)
**Symptom:** Log shows `WEBULL SELL BLOCKED (no position_id)` repeated 10 times, then `WEBULL SELL ABANDONED`.
**Why:** User already closed the position on Webull app. Bot tries to sell, can't find position_id.
**Behavior:** After 10 failed attempts, bot force-closes in DB with `exit_source='manual'`. The `exit_premium` is approximate (market price at time of detection), NOT the user's actual fill price.
**Query:** `SELECT * FROM paper_trades WHERE exit_source = 'manual'`

### Bot frozen / position monitor not selling (CRITICAL)

**This is the highest-severity issue — a frozen monitor means trades can't exit, causing unlimited losses.**

**Symptoms:** Log output stops for minutes at a time, trades don't exit despite hitting stop levels, Docker healthcheck still passes (heartbeat is separate from monitor loop).

**Root causes (all fixed 2026-05-19):**

1. **7GB file copy on candle read (ROOT CAUSE):** `candle_collector.py` was using `shutil.copy2()` to copy the entire 7GB harvester DB to a temp file on every candle read. This blocked the asyncio event loop for minutes.
   - **Fix:** Rewrote `read_candles_from_db()` to use direct WAL reads with `PRAGMA busy_timeout = 5000`. No temp file copy.

2. **No timeout on candle fetches:** `get_candle_data()` calls in pipeline.py, position_monitor.py, and paper_trader.py had no timeout. If Polygon REST or DB read hung, the entire event loop blocked.
   - **Fix:** All 4 call sites now use `asyncio.wait_for(..., timeout=15)` with graceful fallback on timeout.

3. **Read-only Docker mount blocked WAL:** Shared harvester volume was mounted `:ro`, but WAL mode needs to create `-shm` sidecar files.
   - **Fix:** Changed all 4 bot mounts from `:ro` to `:rw` in docker-compose.yml.

**How to diagnose a freeze:**
```bash
# Check if logs are still being written (gap = freeze)
tail -20 journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log

# Look for timeout warnings (healthy behavior — means timeouts are working)
grep 'timed out (15s)' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log

# Check candle reads are using WAL (should see "WAL" not "shutil")
grep 'candle' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log | head -5
```

**Prevention pattern — asyncio.wait_for:**
Any async call that touches external I/O (DB, REST API, WebSocket) in the monitor loop or entry pipeline MUST be wrapped:
```python
try:
    data = await asyncio.wait_for(some_async_call(), timeout=15)
except asyncio.TimeoutError:
    logger.warning("call timed out (15s)")
    # graceful fallback — never block the event loop
```

### Bot crash-looping on weekends/off-hours

**Expected behavior.** The `check_polygon_realtime_entitlement()` function aborts startup when Polygon quotes are stale. On weekends → crash → docker `restart: always` retries. Bots auto-recover Monday at 9:30 AM ET.

### Premium/price data issues

Position monitor tries 4 sources in order:
1. **Market stream** (Polygon WebSocket) — real-time
2. **Polygon REST** (`polygon_option_premium()`) — near real-time
3. **yfinance option chain** — delayed
4. **Delta approximation** — last resort estimate

After 3 consecutive premium failures, the bot sends a Discord DM alert.

### Position sizing seems wrong

```bash
grep 'SIZING' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log
```

The log shows: `SIZING: score=95 balance=$23000.00 cost/contract=$200.00 | risk_cap=75% deployable=$17250.00 | max_concurrent=5 target/slot=$3450.00 | flat_mult=85% scaled=$2932.50 raw=14 | pos_cap(15%=$3450.00)=17 → 14 contracts (total=$2800.00)`

### P&L numbers look wrong

**CRITICAL: Do NOT use `SUM(pnl_dollars)` from paper_trades directly.** The `pnl_dollars` column has known issues:

1. **DCA entry price mismatch**: V6 DCA updates `premium_per_contract` with paper blended avg, but Webull fills differ. The Webull reconcile path uses this paper blended price, not the actual DCA fill.
2. **Scaleout double-counting**: Naive `SUM(pnl_dollars)` across all rows includes both parent and child P&L, but parent P&L was already computed on only remaining contracts.
3. **46 trades have mismatched P&L** as of 2026-05-20.

**Always use `scripts/trade-pnl.py`** for correct P&L:
```bash
python scripts/trade-pnl.py --droplet              # last 20 families
python scripts/trade-pnl.py --droplet --webull-only # all Webull trades
python scripts/trade-pnl.py --droplet NVDA          # per-ticker
```

The script groups parent + scaleout children into "trade families", uses the best available fill prices (Webull > paper), and flags mismatches.

### ENRG not firing

ENRG only fires when: (1) ENABLE_ENRG=true, (2) within grace period, (3) position is negative, (4) candle data available. Check:
```bash
grep 'ENRG\|enrg' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log
```

### Emergency: stop all trading immediately

```bash
# Option 1: Kill switch via docker-compose (persists across restarts)
ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145
cd /root/options-owl
# Edit docker-compose.yml: set WEBULL_KILL_SWITCH=true for the target bot
docker compose up -d owlet-kody  # restart just that bot

# Option 2: Stop containers (temporary)
docker compose stop owlet-kody owlet-adam owlet-vinny owlet-yank

# Option 3: Nuclear — stop everything
docker compose down
```

## Timezone Handling

**All timestamps in the database are UTC.** Signal `created_at` and harvester `captured_at` are both UTC.

Production code converts to ET correctly:
- `position_monitor.py` uses `_now_et()` → `datetime.now(tz=ZoneInfo("America/New_York"))`
- `is_time_decay_zone()` receives ET times from position_monitor
- Market hours: 9:30 AM – 4:00 PM ET
- EOD cutoff: 3:45 PM ET (hardcoded in `EODExitGate`)
- Expiry safety: 10 min before 4:00 PM ET

**Replay scripts** must convert UTC→ET before any time-of-day logic. Never assume DB timestamps are in ET.

## Deploying Changes

**CRITICAL: ALWAYS use the `/deploy` skill (below) or `scripts/rebuild.sh` — NEVER manually rsync, docker build, or docker compose up.**

### Production Droplet

| Field | Value |
|---|---|
| **IP** | `129.212.138.145` |
| **User** | `root` |
| **SSH Key** | `~/.ssh/id_ed25519_do` |
| **Project Dir** | `/root/options-owl` |
| **SSH Command** | `ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145` |

### Deploy Pipeline (`scripts/rebuild.sh`)

The rebuild script enforces a controlled local DevOps pipeline:

```
Step 1a: pytest tests/ -x -q        ← MUST pass or deploy is blocked
Step 1b: ruff check options_owl/     ← lint warning (non-blocking)
Step 2:  rsync local → droplet       ← syncs all code (excludes .env, journal data, .git)
Step 3:  docker compose build --no-cache  ← always fresh images
Step 4:  restart-staggered.sh        ← 15s between bots (avoids Webull 429)
Step 5:  docker compose ps           ← verify all containers healthy
```

```bash
# Standard deploy (runs tests first, blocks on failure)
./scripts/rebuild.sh

# Deploy just one agent
./scripts/rebuild.sh owlet-kody

# Emergency hotfix ONLY (skips tests — use sparingly)
./scripts/rebuild.sh --skip-tests
```

**RULES:**
- Tests MUST pass before code reaches the droplet. No exceptions except `--skip-tests` for emergencies.
- Restarts are ALWAYS staggered (15s between bots) to avoid Webull 429 rate limits.
- NEVER run `docker compose up -d` directly — it starts all bots simultaneously.
- NEVER run docker locally — only on the droplet.
- Webull auth tokens SURVIVE rebuilds — no phone re-approval needed.

### /deploy Skill

**When the user says "deploy", "rebuild", "push to prod", or "ship it" — ALWAYS follow this exact workflow:**

1. Run `./scripts/rebuild.sh` (or `./scripts/rebuild.sh <target>` for single agent)
2. The script handles everything: tests → sync → build → staggered restart → verify
3. After the script completes, tail logs to confirm bots are operational:
   ```bash
   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
     "cd /root/options-owl && docker compose logs owlet-kody --tail 10"
   ```
4. If any bot shows errors, investigate immediately — do NOT leave broken bots running during market hours

**NEVER skip the rebuild script. NEVER deploy by manually rsyncing or running docker commands.**

## Key Decisions (Why Things Are The Way They Are)

- **V5 FSM is the active exit engine**: Category-aware (HIGH_VOL/INDEX/STANDARD), DTE-aware (0DTE vs multi-day), with per-ticker optimal configs. Backtested: $21,685 over 161 trades.
- **Break-even ratchet guarantees no loss**: Once a trade hits +20%, the stop floor moves to entry price. User literally cannot lose money after that point.
- **2PM trail tightening**: After 2PM ET, adaptive trails tighten by 30% — locks in more profit during gamma death zone.
- **Grace backstop prevents catastrophic holds**: Grace period (5min) still allows backstop to fire at -65% (0DTE) / -75% (multi-day). Old bug held -95% trades for 5 minutes.
- **Per-ticker configs from backtest**: Each high-activity ticker has its own optimal FSM params (grace length, stop widths, trail tiers). Unknown tickers use sensible defaults.
- **Flat sizing above score 78**: Backtested (2026-05-20) — scores don't predict outcomes, so all qualifying trades get equal 85% budget allocation. The 78 floor is the real filter.
- **Per-ticker expiry schedule**: Smart entry knows each ticker's actual options schedule instead of blindly trying 0DTE.
- **Stale quote abort**: Intentional crash-on-stale-quote. Better to not trade than trade on yesterday's prices.
- **on_ready guard**: Prevents Discord reconnects from creating duplicate monitors or losing Webull connection.
- **Auto-reconnect on stale Webull**: SDK connection silently dies after idle periods. Bot auto-reconnects and retries.
- **Manual vs AI exit tracking**: `exit_source` column distinguishes bot exits from user manual sells for clean backtesting.
- **Dip-confirm entry gate**: Waits up to 60s for premium to stabilize/dip before buying. Checks underlying support (5m candle lows) and VWAP. Typically saves 2-5% on entry. Falls through to immediate buy on timeout.
- **WAL mode for shared harvester DB**: Harvester writes continuously, 4 bots read concurrently. WAL allows this without blocking. Docker mounts must be `:rw` (not `:ro`) for WAL sidecar files.
- **asyncio.wait_for on all external I/O in monitor loop**: Every DB read, REST call, and candle fetch in the critical path has a 15s hard timeout. Prevents event loop freezes that block sells.
- **V6 DCA (Dollar Cost Average)**: When premium dips 15-35% from entry, auto-doubles position at lower price. Blends entry price down. One-shot per trade.

## Known Issues & Historical Bugs

### 0DTE Expiry Mismatch (discovered 2026-04-23, improved 2026-04-26)
**Bug:** Bot assumed all tickers have daily 0DTE options.
**Impact:** ~60% of trades were paper-only because Webull rejected non-existent contracts.
**Fix:** Per-ticker expiry schedules + near-expiry fallback.

### Discord on_ready Re-initialization (fixed 2026-04-23)
**Fix:** `_initialized` flag prevents re-init. Webull init retries 3x with backoff.

### Velocity Exit Too Aggressive (fixed 2026-04-20)
**Fix:** Replaced with dollar_trail (stair-step trailing stop with % of entry cost).

### Stale Code After Deploy (fixed 2026-04-20)
**Fix:** `rebuild.sh` always uses `--no-cache`. Never rsync without rebuilding.

### Webull Stale Connection — All Bots Lost Trades (fixed 2026-05-07)
**Bug:** Webull SDK internal HTTP session silently dies after hours of idle time. All 4 bots hit `ValueError: no active connection` simultaneously when signals arrived after a quiet period.
**Impact:** MSTR, AMZN, AAPL signals all approved by pipeline but crashed on Webull order placement. All trades were paper-only.
**Fix:** `_reconnect()` method in webull_executor.py — catches connection errors, tears down stale clients, reinitializes, retries the order.

### UnboundLocalError in Position Monitor — No Sells for Hours (fixed 2026-05-07)
**Bug:** Adding `MAX_TRADE_LOSS_EXIT_PCT` check introduced `reason` variable that was only initialized conditionally but referenced unconditionally. Crashed monitor every 15-second cycle.
**Impact:** NO trades could exit. QQQ went from -35% (stop should have fired) to -88%. All positions were stuck.
**Fix:** Initialize `reason = None` and `description = ""` before the conditional block. Added source code safety tests that inspect the actual function to verify variable initialization order.
**Lesson:** ALWAYS run integration tests that exercise the full monitor loop before deploying. Static source analysis tests catch this class of bug.

### Manual Closes Not Tracked (fixed 2026-05-08)
**Bug:** When users sell on Webull manually, bot force-closes in DB at approximate market price. No way to distinguish AI vs manual exits, poisoning backtests.
**Fix:** Added `exit_source` column (`ai`/`manual`) to paper_trades. Sell-abandoned path (position gone from Webull after 10 retries) now marks `exit_source='manual'`.

### 7GB Temp File Copy Froze All Bots (fixed 2026-05-19)
**Bug:** `candle_collector.py:read_candles_from_db()` used `shutil.copy2()` to copy the entire 7GB harvester DB to a temp file before reading candles. This blocked the asyncio event loop for 2-4 minutes.
**Impact:** Position monitor couldn't sell. QQQ peaked at +21% but couldn't exit, eventually closed negative (-$1,620 loss). Multiple freezes in one trading day.
**Fix:** Rewrote to use direct WAL reads (`aiosqlite.connect` + `PRAGMA journal_mode = WAL` + `PRAGMA busy_timeout = 5000`). Added `asyncio.wait_for(..., timeout=15)` to all 4 `get_candle_data()` call sites (pipeline.py ×2, position_monitor.py ×1, paper_trader.py ×1). Changed Docker volume mounts from `:ro` to `:rw` for WAL sidecar file creation.
**Lesson:** NEVER copy large SQLite files for reads. Use WAL mode for concurrent access. ALWAYS wrap external I/O in asyncio.wait_for with a hard timeout in the monitor loop.

### P&L Reconciliation Mismatch (fixed 2026-05-18)
**Bug:** DB tracked +$1,939 for Webull trades but actual Webull P&L was ~$9,842.
**Root causes:** (1) scaleout child rows had no Webull IDs, used simulated P&L; (2) DCA blended entry price wrong; (3) manual closes with approximate exits.
**Fix:** Reconciled 52 trade records via `scripts/reconcile_local.py`. Forward fix: `close_webull_position` now accepts `child_trade_id` param, `partial_close_trade` copies Webull IDs to child rows.

## Development Notes

- Python 3.12+, all async via asyncio + aiosqlite
- Discord server: Neverland Pirates (ID: 1469404711613497591)
- LightGBM requires `libgomp1` (installed in Dockerfile)
- Tests: `pytest tests/` — 1621 tests, ~5min
- Lint: `ruff check options_owl/ tests/`
- Never run docker locally — always on the droplet
- Daily portfolio sync: `ENABLE_PORTFOLIO_SYNC=true` — fetches live Webull balance once per day
- After rebuild, use `scripts/restart-staggered.sh` (15s delay between bots) to avoid Webull 429 rate limits
- Webull auth tokens SURVIVE rebuilds — do NOT tell user to re-approve on phone after rebuild

## Code Change Safety Rules

**CRITICAL: The UnboundLocalError bug (2026-05-07) cost real money.** Follow these rules:

1. **Never introduce conditional-only variable assignments** — if a variable is used after a conditional block, initialize it BEFORE the block.
2. **Run the full test suite before deploying** — `pytest tests/ -q` must pass.
3. **Integration tests for position_monitor changes are mandatory** — the monitor loop is the most critical code path. See `tests/test_monitor_integration.py` and `tests/test_exit_source_tracking.py`.
4. **Source code safety tests**: `TestSourceCodeSafety` in test files inspects actual function source to catch uninitialized variable patterns.
5. **Changes to position_monitor.py are HIGH RISK** — this is the sell path. A bug here means trades can't exit, which means unlimited losses.
6. **All external I/O in the monitor loop MUST have asyncio.wait_for timeout** — DB reads, REST calls, candle fetches. A hung call freezes the entire event loop, preventing ALL sells. Use `asyncio.wait_for(..., timeout=15)` with graceful fallback.
7. **Never copy large SQLite files for reads** — use WAL mode (`PRAGMA journal_mode = WAL`) with `PRAGMA busy_timeout = 5000` for concurrent access. The harvester DB is 7GB+.
