# Held-Contract Options WS Feed (real-time premiums for the monitor)

**Status:** ❌ REJECTED / NOT NEEDED (2026-07-22) · **Created:** 2026-07-22 · **Owner:** Kody

## ⛔ OUTCOME — do NOT build a second options WS
Two findings killed this as designed AND showed it's largely unnecessary:
1. **Polygon account has a HARD max-connections limit (1 per cluster).** The harvester's `FlowCollector`
   ALREADY holds the single `wss://socket.polygon.io/options` connection (subscribes `T.*,Q.*`, filters to
   the 36-ticker universe client-side). A second `/options` WS → `status: max_connections` → the two fight and
   reconnect-war, disrupting the LIVE flow feed. (Observed live 2026-07-22; reverted immediately.)
2. **The real-time feed already exists.** FlowCollector's `_redis_quote_flush_loop` publishes `owl:option:` +
   `owl:snapshot:` to Redis **every 1s** for every quoting universe contract. Empirically (today's kody log)
   the monitor reads premiums at **age 0–10s** (median ~5s), capped by `EXIT_SNAPSHOT_MAX_AGE_SEC=10` — NOT the
   15s REST loop (that's a supplement for greeks/OI/thin coverage). The earlier "15s stale" premise was wrong.
3. **No coverage gap.** `HARVEST_UNIVERSE` already includes all 36 traded tickers (ML + flow + expansion).

**Remaining staleness** = thin contracts that STOP quoting (no NBBO ticks >10s) — a market reality no WS can fix
(the contract simply isn't quoting). That tail is exactly what the **broker-side stop** (venue-resident) covers.

Code shipped flag-gated OFF (`ENABLE_HELD_OPTION_WS=false` everywhere): `collectors/held_option_ws.py`,
`redis_client.publish_optquote/get_optquote/register_held_contracts`, monitor Source -1.5. Left dormant (would
only be usable on a higher Polygon connection tier). If freshness ever needs improving, the lever is INSIDE
FlowCollector (faster flush / per-held-contract priority), NOT a new connection.

---
(original design below — retained for context)

## Problem
The monitor measures every exit gate (trail, profit-lock, stops) on the **option premium**, but that premium is
**not real-time**. Today's path:
- Underlying price = Polygon **/stocks WS** (per-tick, real-time). ✅
- Option premium = harvester pulls **Polygon REST `/v3/snapshot/options` every 15s** → Redis; the monitor
  rejects snapshots >10s old and falls to a per-contract Polygon REST call. So premium freshness is **~15s /
  best-effort REST**, not per-tick.

Consequence: the give-back (+17%→−31% between polls) is **bounded by data freshness, not loop cadence** — polling
faster can't help when the underlying premium data only refreshes every 15s. The full options-WS plumbing already
exists (`market_data_stream._polygon_send_option_subscribe` → `Q.O:`/`T.O:` NBBO+trades,
`_process_polygon_message`, `subscribe_option`) but is **disabled on bots** (`ENABLE_POLYGON_WS=false`) under the
"single WS holder = harvester" rule (per-bot WS storms caused 429s — the retired flow-shadow lesson).

## Key insight
Full-chain options WS is impractical (thousands of contracts) — which is why the harvester scans via REST. But
the **monitor only needs the 5–30 contracts currently held** across all bots. Subscribing a WS to *just those* is
trivially within Polygon's subscription cap and is **one** connection on the harvester — the WS-storm rule is
satisfied.

## Design
1. **Held-contract registry (Redis).** Each bot writes its currently-open contract keys to a Redis set
   (`owl:held_contracts`, per-bot sub-keys w/ TTL refreshed each monitor cycle). Union = the live subscription set.
2. **Harvester opens a Polygon /options WS** (in addition to /stocks) subscribed to `Q.O:<contract>` (NBBO) +
   `T.O:<contract>` (trades) for the union. Diffs the set each cycle → subscribe new / unsubscribe closed. One
   connection, small dynamic set. Reuse the existing `_polygon_send_option_subscribe` logic (lift into harvester).
3. **Harvester publishes per-tick NBBO** to Redis `owl:optquote:<contract>` (bid/ask/mid + ts), fresh <1s.
4. **Monitor reads `owl:optquote:` first** (ahead of the 15s snapshot), tightening `EXIT_SNAPSHOT_MAX_AGE_SEC`
   reliance. Fresh premium → the adaptive 1s poll (or an event-driven variant) reacts in ~sub-second.

## Phases
- **P1:** held-contract registry (bots write, harvester reads the union). No WS yet — validate the set is correct.
- **P2:** harvester /options WS + `owl:optquote:` publish (behind `ENABLE_HELD_OPTION_WS`, default off). Log tick
  rate + freshness; confirm no 429 / connection issues on the harvester.
- **P3:** monitor consumes `owl:optquote:` first; measure give-back vs the 15s-snapshot control (trade_premium_ticks).
- **P4 (optional):** event-driven monitor — evaluate the FSM on each WS tick for held contracts instead of polling.

## Relationship to the broker stop (complementary)
- **This** makes ALL exit logic real-time (trail/profit-lock react per-tick), monitor-side, ~sub-second latency.
- **Broker stop** is the zero-latency hard floor resting AT the venue (beats any network round-trip on the tail).
- End-state = both: WS for real-time exit logic, broker stop as the instant floor beneath it.

## Risks / open questions
- Confirm the Polygon plan is Options **Advanced** (real-time WS, not delayed) — REST snapshots already are, so WS
  should be included. Verify.
- Harvester WS subscription churn as trades open/close (rate-limit subscribe/unsubscribe messages).
- Redis `owl:optquote:` write volume at high tick rates (a busy SPY 0DTE can tick fast) — bound with a min-publish
  interval per contract if needed.

## Files (anticipated)
`harvester.py` (+options WS + optquote publish), `collectors/market_data_stream.py` (lift `_polygon_send_option_*`),
`execution/position_monitor.py` (read optquote first + register held contracts), `db/redis_client.py` (optquote
get/set + held-contract set), `config/settings.py` (`ENABLE_HELD_OPTION_WS`).
