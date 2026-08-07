# Centralize ML Signal Generation (Harvester → Redis → Bots)

**Status:** BUILT (flag-gated, inert by default) 2026-08-04 — awaiting shadow-mode deploy + validation

## Implementation status (2026-08-04)
Code complete, all behind flags (default OFF = today's exact behavior). 33 tests green; e2e trade path
unchanged; bot_runner + harvester import clean.
- **Redis transport** — `redis_client.publish_ml_signal` / `subscribe_ml_signals` (mirror of flow pub/sub).
- **Payload bus** — `collectors/ml_signal_bus.py` (pure: build/parse/heartbeat/freshness). 16 unit tests.
- **Settings** — `ENABLE_CENTRAL_ML_SIGNALS`, `ENABLE_CENTRAL_ML_PUBLISH`, `ML_CENTRAL_SHADOW`,
  `ML_CENTRAL_FALLBACK_SEC`, `ML_SIGNAL_MAX_AGE_SEC`, `ML_SCAN_INTERVAL_SEC` (all default off/safe).
- **Scan-loop hooks** — `_run_ml_scan_loop(..., signal_sink=None, central_state=None)`: signal_sink =
  harvester publishes instead of trading; central_state = bot defers to central when alive, falls back when
  silent. Defaults None = byte-for-byte today's behavior.
- **Harvester publisher** — gated `ENABLE_CENTRAL_ML_PUBLISH` task runs the scan via a bare shim (coupling is
  a single `evaluate_and_trade`, bypassed by signal_sink) + publishes signals + heartbeats.
- **Bot consumer** — gated `ENABLE_CENTRAL_ML_SIGNALS` task; shadow logs parity, active trades + local defers.
- **NOT yet done:** the faster `ML_SCAN_INTERVAL_SEC` cadence inside the loop (v1 reuses the existing scan
  cadence — already a big win once the Polygon-fallback latency is gone in the harvester). Add after shadow.

**Deploy sequence (no more code — this is validation):** (1) harvester `ENABLE_CENTRAL_ML_PUBLISH=true` +
one paper bot `ENABLE_CENTRAL_ML_SIGNALS=true ML_CENTRAL_SHADOW=true` → watch `ML_CENTRAL[shadow]` parity vs
local ≥3 days. (2) paper bot shadow→active. (3) dennis. (4) kody.

*(original design below)*
**Status (design):** DESIGN — ready for review, then flag-gated build + paper canary
**Created:** 2026-08-04
**Risk:** HIGH — touches the live-money entry/signal path (kody + dennis live). Flag-gated, paper-canary mandatory.

## Problem (verified 2026-08-04)
Each bot runs its **own** ML scan loop (`bot_runner._run_ml_scan_loop` → `_scan_one_ticker` → `_run_ml_for_ticker`),
rotating through ~22 tickers on an **unsynchronized clock**. 0DTE pattern signals are **transient** (valid a few
minutes). Whether a bot catches one depends on where its rotation happens to be at that instant.

**Live proof:** 2026-08-04 dennis polled IWM at 13:47:39 → caught `pattern=0.685` → +$546. kody polled IWM at
13:35 (too early) and not again until 15:45 → **missed the same winner entirely.** Identical code, different
polling phase. This is the kody/adam divergence too, and it gets **worse the more tickers we add** (slower
rotation → more missed transient signals). Directly undermines the ticker-expansion work.

## Current architecture
```
each bot:  _run_ml_scan_loop (own clock) → _run_ml_for_ticker (pattern model) → signal dict
           → run_entry_pipeline (gates) → score_to_contracts (sizing) → paper_trader (execute)
```
Signal generation is DUPLICATED per bot and unsynchronized. Everything downstream (pipeline/sizing/execution)
is correctly per-bot (account-dependent).

## Target architecture (mirror the UW-flow design — already proven)
The flow book ALREADY solved this exact class of problem: harvester is the sole computor, bots consume from
Redis. Apply the identical pattern to ML signals.
```
HARVESTER (single clock):  ml scan loop → _run_ml_for_ticker (pattern model) → SIGNAL
                           → redis.publish("owl:ml:signals", signal_payload)     [NEW]
each BOT (consumer):       subscribe owl:ml:signals → _on_ml_signal(payload)      [NEW]
                           → build TradeSignal → run_entry_pipeline → sizing → execute  [UNCHANGED]
```
**One signal, published once, at one instant → every bot sees the identical signal at the identical time.**
The fleet stops diverging on *which* trades it takes. Bots still decide *whether/how much* to trade
independently (their own caps, capital, sizing, fills) — autonomy preserved, only the SIGNAL is synchronized.

### The exact mapping (flow template → ML)
| Flow (existing) | ML (to build) |
|---|---|
| `run_uw_flow_collector` (harvester) | move `_run_ml_scan_loop` into harvester (or a harvester-side scanner task) |
| `_publish_flow` → `publish_flow_signal` → `owl:flow:signals` | `publish_ml_signal` → `owl:ml:signals` [new redis_client fn] |
| `_consume_flow` / `_on_flow_signal` (bot) | `_consume_ml` / `_on_ml_signal` (bot) — deserialize → TradeSignal → `evaluate_and_trade` |
| `_flow_factory` subscribes | `_ml_factory` subscribes |
| `ENABLE_UW_FLOW_SIGNAL` gate | `ENABLE_CENTRAL_ML_SIGNALS` gate |

## What moves vs what stays
- **Moves to harvester:** the pattern-model inference (`_run_ml_for_ticker` + `compute_pattern_features`), the
  scan cadence, model loading (`ml_pipeline.load_models`). Harvester already holds Redis + the option snapshots
  the scan reads — natural home.
- **Stays per-bot (unchanged):** `run_entry_pipeline` (spread/delta/premium/regime/tod/**position-cap**/
  **capital** gates), `score_to_contracts` sizing, Webull execution. These are account-dependent and MUST stay
  per-bot. The harvester publishes the *raw pattern signal*; bots pipeline+size+execute it themselves.

## Cycle time / "check every tick" (the OTHER half of the fix)
Centralizing alone fixes *divergence* but NOT *latency* — if the harvester's scan still cycles slowly, the whole
fleet misses transient signals together. The real root of missed signals is **cycle latency**, and it's fixable
in the same move:

**Why the current cycle is slow (and variable per bot):** the scan is already concurrent (`asyncio.gather` all
tickers), BUT on a Redis miss each ticker does a **synchronous Polygon REST fetch** (~10s, `Semaphore(5)`
throttle, 25s outer timeout — see the "timed out at minute 134" logs). A few slow tickers stretch the whole
cycle far past the intended 1 minute, and *how far* depends on each bot's momentary Redis-hit rate → kody's
effective IWM interval blew out to ~2h while dennis's stayed tight. Same code, different luck.

**The fix — harvester-side scan on in-memory data, no network in the hot path:**
- The harvester IS the Polygon-WS holder and already maintains fresh option snapshots in-process (flushed to
  Redis ~every 1s). Running the scan THERE means it reads **local, already-fresh** data — **zero synchronous
  Polygon fetches in the scan path.** No `Semaphore(5)`, no 10s stalls, no 25s timeouts.
- With no network round-trips, all tickers scan concurrently in **sub-second**, so the scan can run **every
  ~1-2s** (or fire on each snapshot flush) instead of a bloated minute+. A ticker's transient signal is caught
  within seconds of appearing — not minutes later, and not missed.
- Net: the fleet checks **every ticker on every tick** (bounded by the ~1s WS flush cadence), uniformly. This is
  the "don't miss a few-second/minute gap" you asked for — it falls out of moving the scan to where the data
  already lives.

**Guardrail:** keep the pattern-model inference cheap (it already is — one LightGBM predict per ticker/minute).
If scanning every 1-2s proves too hot on the harvester, throttle to every 5s (still ~12-60× tighter than today)
— configurable via `ML_SCAN_INTERVAL_SEC`.

## Redis contract (owl:ml:signals payload)
Serialize the signal dict `_run_ml_for_ticker` returns + resolved contract: `{ticker, direction, minute,
pattern_conf, score, strike, expiry, entry_price, stop_price, underlying, ts_epoch}`. Bot rebuilds a TradeSignal
(BotSource.ML) exactly like `flow_signal_to_trade_signal`. Include `ts_epoch` so bots can **drop stale signals**
(> N sec old = the bot was slow/restarting; don't act on a stale pattern).

## Fallback design (the current ML scan is KEPT as a hot standby)
The per-bot local scan is NOT deleted — it's retained as an automatic fallback, so a harvester/Redis failure
can never leave a bot signal-blind:
- The harvester publishes a **heartbeat** on `owl:ml:signals` every scan tick (even a "no signal this tick"
  beat carries a timestamp). Bots track last-heartbeat time.
- If a consuming bot sees **no heartbeat for `ML_CENTRAL_FALLBACK_SEC`** (e.g. 90s) during market hours, it
  logs LOUD (Discord alert) and **automatically re-enables its local scan** until heartbeats resume. On
  resume, it hands signal duty back to the central feed. No manual intervention, no silent outage.
- `ENABLE_CENTRAL_ML_SIGNALS=false` → local scan is primary (today's behavior, untouched).

## Testing strategy — "test the shit out of it" before it acts
Three gates before the central feed is allowed to place a single trade:
1. **Unit** — Redis publish/subscribe round-trip, payload serialize/deserialize/validate, freshness-guard drop,
   fallback trigger on missed heartbeat. Pure, no live deps.
2. **SHADOW MODE (`ML_CENTRAL_SHADOW=true`)** — the harvester publishes, and a bot runs BOTH its local scan
   AND the central consumer, but the central consumer **only logs** (does NOT call `evaluate_and_trade`). It
   records, per signal: did central and local agree on {ticker, direction, minute, ~pattern score}? Emit a
   daily parity report. **Target: central ⊇ local** (central catches everything local did, ideally MORE —
   the transient signals local was missing). Run shadow ≥3 trading days until parity is clean.
3. **Paper canary** — only after shadow parity is clean, let the central feed actually trade on ONE paper bot.

Shadow mode is the key safety valve: it proves the centralized signals are correct against the live local
scan on real market data, with zero trading risk, before we trust it with money.

## Rollout (mandatory — live money)
1. **Build flag-gated** (`ENABLE_CENTRAL_ML_SIGNALS`, default FALSE). When false → current per-bot scan (zero
   change). When true → per-bot scan DISABLED, consume from Redis.
2. **Fallback:** if the Redis subscription drops or no signal arrives for T minutes during market hours, log LOUD
   + optionally fall back to local scan (avoid a silent no-signal outage).
3. **Harvester self-test:** confirm harvester loads the pattern model + publishes on a known signal before any
   bot consumes.
4. **Paper canary FIRST:** enable on one PAPER bot (e.g. yank), run ≥1 week, confirm it takes the SAME signals
   as a still-local paper bot (should now match) with no missed/dup/stale issues.
5. **Then live:** enable on dennis, watch a few days, then kody. Compare kody-vs-dennis trade OVERLAP (should
   jump to ~100% of ML signals vs today's divergence).
6. Never overwrite the per-bot path until the canary proves parity.

## Risks & mitigations
- **Harvester becomes a single point of failure for ML signals** → it already is for flow + all data; supervise
  the scan task (auto-restart), + the per-bot fallback in step 2.
- **Serialization drift** (signal dict fields) → version the payload; bot validates + skips bad payloads (like
  `_consume_flow` already does).
- **Stale signals** on a slow/restarting bot → `ts_epoch` freshness guard.
- **Double-publish / dedup** → each bot processes each signal once (Redis pub/sub delivers once per subscriber);
  bots already de-dup concurrent same-ticker via DuplicateTickerGate.
- **Harvester CPU** (now runs model inference for the whole fleet once, instead of N bots each) → net LESS total
  compute; but monitor harvester load.

## Open decisions (for review)
1. Harvester runs the scan **task in-process**, or a **separate `ml_signal_publisher.py`** service? (in-process
   = simpler, shares the option snapshots; separate = isolation). **Lean: in-process harvester task** (mirrors
   how `run_uw_flow_collector` is a harvester task).
2. Publish **raw pattern signals** (bots run full pipeline) vs **pre-vetted** (harvester runs account-independent
   gates)? **Lean: raw first** (simplest, preserves all per-bot gate behavior); optimize later.
3. Does harvester need per-bot Polygon fallback for the scan, or is Redis/PG enough? (Harvester already has the
   data it publishes — should be self-sufficient.)

## Related
[[ticker-expansion-download-2026-08-05]] (this fix is a prerequisite for scaling the ticker count),
flow architecture (`collectors/uw_flow_collector.py`, `bot_runner._on_flow_signal`, `harvester.py:780`).
