# Event-Driven Monitor (react at data-speed, not on a 3–5s timer)

**Status:** active (building, phased) · **Created:** 2026-07-23 · **Owner:** Kody

## Why
The premium data is already streamed (harvester FlowCollector → Redis ~1s), but the monitor is a **poll loop**
that only *evaluates* the FSM every 3–5s. So we hold fresh data and look at it late. Reacting per-tick takes
every exit (incl. the new never-green cut) from ~3–5s → ~1s. Ceiling: bounded by the ~1s data cadence, and the
broker stop still wins the absolute venue-tail — but ~1s beats 3–5s everywhere.

## ⚠️ This is the SELL PATH — the highest-risk file. Build slow.
On 2026-07-23 a rushed sell-path change (broker stop) cost ~$563 live. This must be flag-gated, concurrency-safe,
and paper-canaried before any live bot. The poll loop STAYS as the backstop; event-driven is additive.

## Design
1. **Harvester publishes held-contract premium ticks to a pub/sub channel** `owl:premium:ticks` (currently it
   only SETs keys). Only held contracts (reuse the `owl:held:*` registry) to bound traffic. Flag `ENABLE_PREMIUM_TICK_PUBLISH`.
2. **Monitor runs an event task** subscribed to that channel. On each tick for a held trade, it evaluates the FSM
   immediately (out-of-band from the poll loop) and exits if triggered.
3. **Concurrency guard (the critical safety piece):** a per-trade `asyncio.Lock` (or a "processing" set) shared
   by the poll loop and the event task — only ONE path evaluates/exits a given trade at a time. Prevents a
   double-exit (poll + event both selling the same trade). The sell itself already guards double-sell via the
   sell-to-close position lookup, but we add the per-trade lock so we never even attempt it twice.
4. **Fallback:** `ENABLE_EVENT_DRIVEN_MONITOR` off → only the poll loop runs (today's exact behavior). Event task
   failure → poll loop still covers everything. Strictly additive.

## Phases
- **P1 (enabling infra) — safe, additive:** `redis_client.publish_premium_tick` + `subscribe_premium_ticks`;
  harvester publishes held-contract ticks (flag-gated). No monitor behavior change yet. Unit tests. ← THIS TURN.
- **P2 (per-trade eval extraction):** pull the per-trade FSM evaluate+exit block out of `run_position_monitor`
  into a reusable `evaluate_and_maybe_exit(trade, ...)` callable from BOTH the poll loop and the event task, with
  the per-trade lock. Refactor is behavior-preserving; full test suite must stay green.
- **P3 (event task):** the subscriber task, flag-gated, running alongside the poll loop with the lock.
- **P4 (validate):** paper-canary on one bot (measure reaction latency + zero double-exits) → then live.

## Risks
- Double-exit (poll+event) → per-trade lock + the existing sell-to-close guard.
- Pub/sub flood (busy 0DTE ticks fast) → publish only held contracts + a min-interval per contract if needed.
- The per-trade extraction (P2) touches the sell path — behavior-preserving refactor, full-suite gated.

## Files
`db/redis_client.py` (publish/subscribe ticks), `harvester.py` / `collectors/flow_collector.py` (publish held
ticks), `execution/position_monitor.py` (extract eval + event task + lock), `config/settings.py` (flags).
