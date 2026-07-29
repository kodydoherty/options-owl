# Broker-Side Stop-Loss Orders (catch the peak our 5s poll misses)

**Status:** active (building) · **Created:** 2026-07-22 · **Owner:** Kody

## Problem (measured)
Our position monitor polls every 5s. Fast 0DTE moves round-trip *between* polls — measured **avg 19pt give-back**
(peak% − exit%), worst cases +17%→−31% (the crash blew through the +3% profit-lock floor in <5s). This is *why*
the harness (clean minute data, instant FSM) shows 67% WR while live is 34% — the harness catches the peak; live
can't by polling. Scale-out was REFUTED as a fix (−$368; our losers peak too low). The real fix is a resting order
at the venue that fills the microsecond price hits the level — zero poll latency. See
[[put-fix-canary-2026-07-20]] context + the give-back diagnostic.

## API feasibility — CONFIRMED (Webull docs)
- ✅ **STOP_LOSS supported for options** (sell-side, triggers a market sell when premium ≤ stop_price). Exact
  schema: `order_type: "STOP_LOSS"`, `stop_price: "3.00"`, `side: "SELL"`, `time_in_force: "DAY"`, same legs as
  our LIMIT sell. (developer.webull.com options docs, AAPL example.)
- ❌ **TRAILING stop NOT supported for options** → we emulate a trail with a FIXED stop + cancel/replace higher
  as the position runs.
- ⚠️ **Live preview deferred:** Webull has a TOKEN LIMIT; a fresh-auth preview could churn kody's live token
  overnight unattended. Build on the confirmed docs schema; confirm execution via paper-sim + one MONITORED live
  test (market hours), not an unattended preview.

## Design
Place a resting broker STOP_LOSS at the current floor for each open leg; the broker catches the fast crash.
- **Floor logic:** stop at the −25% hardstop level on entry; once the profit-lock arms (+12% → +3% floor),
  cancel+replace the stop up at the floor. Emulated trail via periodic replace (rate-limited to avoid churn).
- **The monitor still runs** (for trail logic + replace) but the HARD exit is the broker stop.

## ⚠️ Safety hazards (the whole reason to go slow) — MUST all be handled
1. **Double-fill:** a resting broker stop + the monitor both selling = sell twice on live money. Guard: while a
   broker stop is ACTIVE for a leg, the monitor must NOT place its own sell; if the monitor must exit for a
   non-stop reason (EOD, signal-flip), it CANCELS the broker stop first and confirms cancel before selling.
2. **Orphaned stops on restart:** in-memory tracking lost on restart → stop keeps resting at Webull. On startup,
   query Webull open orders, match to open positions, re-adopt or cancel orphans.
3. **Order churn / rate limits:** cancel+replace only when the floor moves materially (a min-step + min-interval),
   never every poll.
4. **Partial fills / already-closed:** stop fills while monitor is mid-decision → reconcile via the existing
   position-gone detection.

## ROBUSTNESS CONTRACT (Kody, non-negotiable — this is the live sell path)
The broker stop is **strictly ADDITIVE**. Every failure mode degrades to the **legacy poll-only exit**:
- **Retry, bounded.** Placement retries across cycles up to `BROKER_STOP_MAX_ATTEMPTS` (3), then gives up →
  poll-only. The sell path keeps its existing per-cycle retry/escalation ladder UNTOUCHED.
- **Clean up after itself.** Closed/vanished trades' stops are cancelled + forgotten (`release`/`prune_closed`);
  orphans caught by the restart reconcile (Phase 3).
- **Fall back to legacy on ANY guard failure.** The cancel-before-sell guard is wrapped in try/except that
  logs + proceeds to the legacy sell — a guard failure can NEVER block a sell. Double-fill is *structurally*
  impossible: the sell-to-close position lookup blocks selling a position that's already gone (no naked short).
- **Flag off = byte-identical legacy.** With `ENABLE_BROKER_STOP=false`, attempt-0 skips the guard entirely.
- The manager **never raises into the monitor loop** (every public coroutine swallows+logs).

## Build phases
- **Phase 1 (executor capability) — ✅ DONE.** `_build_stop_order_payload` + `place_stop_loss()` (flag-gated,
  never submits without a position_id). 15 unit tests. Inert.
- **Phase 2 (monitor integration + double-fill guard) — ✅ DONE.** `broker_stop.BrokerStopManager` (place/track/
  clean-up, bounded-retry, fail-safe) wired into `run_position_monitor` (`ensure_stop` per live trade,
  `prune_closed` each cycle). Guard added at the sell chokepoint `close_webull_position` (cancel-and-confirm the
  resting stop on attempt 0 when enabled, reusing the proven 2026-07-07 cancel-confirm block). **42 tests:
  unit (manager) + integration (real sell path + fallback) + e2e (full lifecycle + stop-fired race) + source-
  safety invariants.** Still inert in prod (flag off).
- **Phase 3 (replace-to-trail + orphan reconcile) — ✅ DONE.** `compute_desired_stop` (base −25% floor →, once
  peak gain ≥ `RATCHET_ARM_PCT`=20%, ratchet up to `max(base, breakeven, peak × TRAIL_KEEP_FRAC=0.75)` — never
  below breakeven, monotonic in peak). `ensure_stop(..., current_premium=)` tracks the peak and cancel+replaces
  the resting stop UP, rate-limited by `MIN_STEP_FRAC` (0.05×entry) + `MIN_REPLACE_SEC` (30s). Robustness: only
  moves up; cancel-fails → keeps old stop (never two live); cancel-then-place-fails → FSM covers + retry.
  `reconcile_orphans` startup sweep: cancels resting STOP_LOSS on closed contracts, re-adopts valid ones. Wired
  into `run_position_monitor` (ratchet fed the fresh `exit_premium`; one-time startup reconcile). **+14 tests
  (56 total): desired-stop math, ratchet up/down/min-step/cancel-fail, reconcile adopt+cancel+guards.** Still
  inert (flag off).
- **Phase 4 (validate) — NEXT:** paper-canary (flip `ENABLE_BROKER_STOP=true` on ONE paper bot — yank/vinny —
  no live Webull) → ONE monitored live test (market hours) → measure give-back vs a poll-only control (the
  scoreboard from trade_premium_ticks, already live, 128k rows).

## Go-live criteria
Give-back (peak−exit) on the broker-stop bot drops materially vs a poll-only control, with ZERO double-fills
observed in the paper-sim + the monitored live test.

## Files
`execution/webull_executor.py` (place_stop_loss + payload), `execution/position_monitor.py` (integration +
guard, Phase 2), `config/settings.py` (ENABLE_BROKER_STOP + BROKER_STOP_* knobs). Tests: `test_broker_stop.py`.
