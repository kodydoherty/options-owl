# E2E Signal Pipeline Test

## Purpose

Verify the full sourcing-to-trade pipeline works end-to-end across all agents. Catches broken database connections, stale consumers, missing containers, and harvester data gaps — before market open.

## Architecture

```
scripts/e2e_signal_test.py (local machine)
  │
  │ SSH + psql via Docker
  ▼
┌──────────────────────────────────────────────────────────┐
│  Droplet (129.212.138.145)                               │
│                                                          │
│  PostgreSQL (options-owl-db)                             │
│    └── ml_signals table ← inject test signal here        │
│                                                          │
│  owlet-kody ──┐                                          │
│  owlet-adam ──┤ signal_consumer.py (polls every 30s)     │
│  owlet-vinny ┤ reads ml_signals → marks consumed_by     │
│  owlet-yank ──┘                                          │
│                                                          │
│  owlet-harvester → stock_ticks, option_ticks,            │
│                     stock_candles (PG tables)             │
│                                                          │
│  owlet-sourcing → ml_signals (real signals)              │
└──────────────────────────────────────────────────────────┘
```

## What It Tests

| Step | Check | Pass Criteria |
|------|-------|---------------|
| 0 | PostgreSQL reachable | `SELECT 1` succeeds |
| 0 | Containers running | All 6 containers (4 bots + sourcing + harvester) are `Up` |
| 1 | Signal injection | INSERT into `ml_signals` returns an ID |
| 2 | Agent consumption | All running trading bots add their `agent_id` to `consumed_by` within 150s |
| 3 | Trade events | `trade_events` rows appear for the test ticker (rejected is expected — proves pipeline ran) |
| 4 | Tick data | `stock_ticks`, `option_ticks`, `stock_candles` have recent rows from harvester |
| 5 | Cleanup | Test signals removed from `ml_signals` |

## Signal Flow Being Tested

```
1. Test signal inserted into ml_signals (ticker="TESTowl", score=95)
2. Each bot's signal_consumer polls ml_signals every 30s
3. Consumer finds signal where agent_id NOT IN consumed_by
4. Consumer converts to TradeSignal, calls paper_trader.evaluate_and_trade()
5. Entry pipeline rejects (no option chain for "TESTowl") → trade_event logged
6. Consumer marks signal consumed (appends agent_id to consumed_by array)
7. After all 4 agents consume → status changes to "consumed"
```

The test ticker `TESTowl` is intentionally fake. It will be rejected by smart entry (no option chain exists), which is perfect — we verify the full consumption path without placing real trades.

## Commands

```bash
# Full E2E (default) — inject → wait → check → cleanup
python3 scripts/e2e_signal_test.py --droplet

# Quick health check — containers, recent signals, tick data
python3 scripts/e2e_signal_test.py --droplet --status

# Just inject a signal (don't wait)
python3 scripts/e2e_signal_test.py --droplet --inject-only

# Remove leftover test signals
python3 scripts/e2e_signal_test.py --droplet --cleanup
```

## Dependencies

**None.** The script uses `psql` inside the Docker postgres container via SSH. No Python packages needed on the host or locally beyond the standard library.

## When To Run

- **Monday before market open** (9:00 AM ET) — verify all bots recovered from weekend crash-loop
- **After any deployment** (`scripts/rebuild.sh`) — verify bots are consuming signals
- **After infrastructure changes** — new containers, PG schema changes, Redis config
- **Debugging "no trades"** — quick `--status` shows if signals exist and who consumed them

## Expected Output (Healthy System)

```
============================================================
  Step 0: Infrastructure check
============================================================
  [OK]   PostgreSQL reachable
  [..]   Running containers: ['options-owl-db', 'options-owl-redis', 'owlet-harvester', 'owlet-kody', 'owlet-adam', 'owlet-vinny', 'owlet-yank', 'owlet-sourcing']
  [OK]   Active trading agents: ['owlet_kody', 'owlet_adam', 'owlet_vinny', 'owlet_yank']

============================================================
  Step 1: Inject test signal
============================================================
  [OK]   Injected signal #42 (TESTowl CALL score=95)

============================================================
  Step 2: Wait for agent consumption
============================================================
  [..]   Waiting up to 150s for 4 agent(s)...
  [..]   30s: consumed_by=['owlet_kody', 'owlet_adam'] (2/4)
  [..]   60s: consumed_by=['owlet_kody', 'owlet_adam', 'owlet_vinny', 'owlet_yank'] (4/4)
  [OK]   All 4 agent(s) consumed signal #42!

============================================================
  Step 3: Check trade_events
============================================================
  [OK]   Found 4 trade event(s):
  [..]     owlet_kody|smart_entry_blocked|2026-05-26 13:30:05+00
  [..]     owlet_adam|smart_entry_blocked|2026-05-26 13:30:12+00
  [..]     owlet_vinny|smart_entry_blocked|2026-05-26 13:30:18+00
  [..]     owlet_yank|smart_entry_blocked|2026-05-26 13:30:25+00

============================================================
  Step 4: Harvester tick data
============================================================
  [OK]   stock_ticks: 125,000 rows, 42 recent (10min), latest=2026-05-26 13:29:50+00
  [OK]   option_ticks: 890,000 rows, 215 recent (10min), latest=2026-05-26 13:29:55+00
  [OK]   stock_candles: 15,200 rows, 8 recent (10min), latest=2026-05-26 13:25:00+00

============================================================
  Step 5: Cleanup
============================================================
  [OK]   Removed 1 test signal(s)

============================================================
  RESULT
============================================================
  [OK]   ALL CHECKS PASSED
```

## Expected Output (Sunday / Off-Hours)

```
  [FAIL] No trading bot containers running! Agents won't consume signals.
  [..]   Trading bots crash-loop on weekends (expected). Try Monday after 9:30 AM ET.
  [..]   stock_ticks: empty (market closed or harvester hasn't written yet)
```

This is normal. Trading bots intentionally crash when Polygon quotes are stale. They auto-recover Monday at 9:30 AM ET via Docker `restart: always`.

## Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| PostgreSQL unreachable | PG container down | `docker compose up -d postgres` |
| Signal injection fails | Schema not created | Restart any bot (auto-creates schema) |
| Agent never consumes | `signal_consumer` not started, or `ENABLE_POSTGRES=false` | Check docker-compose.yml env vars |
| Only some agents consume | One bot crashed or isn't polling | `docker compose logs owlet-X --tail 20` |
| No trade_events | Phase 1 dual-write — events still go to SQLite only | Check SQLite: `./scripts/trade-log.sh` |
| Tick data empty (during market hours) | Harvester not writing to PG, or `ENABLE_POSTGRES=false` | Check harvester logs |
| Timeout (150s) | Bots are slow to poll, or consumer erroring | Check bot logs for exceptions |
