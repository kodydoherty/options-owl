# Spec 01: Premium Tick Storage

**Priority**: 1 (highest)
**Effort**: Low (1-2 hours)
**Impact**: Directly enables better exit timing ML model training

## Problem

Position monitor fetches live option premium every 5 seconds for each open trade but only keeps it in memory (`_premium_histories` dict in `position_monitor.py:59`). When the process restarts or a trade closes, this granular premium path data is lost forever.

This is the single most valuable data we're NOT storing. Every 5s premium tick per trade gives us:
- Exact premium paths for training exit timing models
- Drawdown/recovery patterns per ticker category
- Real slippage measurement (signal premium vs actual path)
- DCA timing optimization (when dips actually bottom)

## Current State

- `position_monitor.py` polls every 5s, fetches premium, stores in `_premium_histories[trade_id]`
- Premium comes from: Polygon WS → Polygon REST → yfinance → delta approximation (4-source cascade)
- Data is used for velocity/deceleration calculations within the FSM, then discarded on trade close
- `tick_logger.py` logs raw WS ticks to flat files — not queryable, not in PG

## Design

### New PG Table: `trade_premium_ticks`

```sql
CREATE TABLE IF NOT EXISTS trade_premium_ticks (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    trade_id INTEGER NOT NULL,          -- SQLite paper_trades.id
    ticker TEXT NOT NULL,
    premium REAL NOT NULL,
    bid REAL,
    ask REAL,
    underlying_price REAL,
    source TEXT NOT NULL,                -- 'polygon_ws', 'polygon_rest', 'yfinance', 'delta_approx', 'webull'
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prem_ticks_trade ON trade_premium_ticks(agent_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_prem_ticks_ticker ON trade_premium_ticks(ticker, captured_at);
```

### Write Path (fire-and-forget)

In `position_monitor.py`, after premium is fetched successfully for a trade, fire-and-forget write to PG:

```python
# After premium is obtained (around line ~700 in _monitor_single_trade)
asyncio.create_task(_write_premium_tick(
    agent_id=settings.AGENT_ID,
    trade_id=trade["id"],
    ticker=trade["ticker"],
    premium=current_premium,
    bid=bid_price,        # if available from source
    ask=ask_price,        # if available from source
    underlying_price=underlying_price,
    source=premium_source,  # track which of the 4 sources provided it
))
```

### Throttling

Don't write every 5s tick — that's 720 rows/hour per trade. Write every **15 seconds** (every 3rd poll cycle). Still captures premium path with high fidelity. With 5 max concurrent trades × 4 agents × 240 ticks/hour = ~4,800 rows/hour. Manageable.

### Batch Writes

Buffer ticks and flush every 30s (6 ticks per trade per flush) using `pg.executemany()` to minimize PG round-trips.

## Files to Modify

| File | Change |
|---|---|
| `options_owl/db/postgres.py` | Add `trade_premium_ticks` to SCHEMA_SQL, add `write_premium_ticks_batch()` and `read_premium_ticks()` functions |
| `options_owl/execution/position_monitor.py` | Add tick buffer dict, write premium after each successful fetch (throttled to 15s), flush buffer periodically |
| `docker-compose.yml` | No changes needed (PG already running) |

## Tests

| Test | What it validates |
|---|---|
| `test_premium_tick_schema_exists` | PG schema includes `trade_premium_ticks` table |
| `test_premium_tick_write_batch` | `write_premium_ticks_batch()` inserts rows correctly |
| `test_premium_tick_throttle` | Only writes every 3rd poll cycle (15s), not every 5s |
| `test_premium_tick_fire_and_forget` | PG write failure doesn't block monitor loop |
| `test_premium_tick_source_tracking` | Source field correctly identifies polygon_ws vs polygon_rest vs yfinance vs delta_approx |
| `test_premium_tick_read` | `read_premium_ticks(trade_id)` returns chronological premium path |
| `test_premium_tick_cleanup` | Optional: old ticks (>90 days) can be pruned |

## ML Training Usage

Once populated, premium tick data enables:

```python
# Load premium path for a closed trade
ticks = await pg.read_premium_ticks(agent_id="owlet-kody", trade_id=225)
# Returns: [{"premium": 2.50, "captured_at": ..., "source": "polygon_ws"}, ...]

# Features extractable per-trade:
# - max_drawdown_from_entry (how deep did it dip before recovery)
# - time_to_peak (how quickly did premium peak)
# - recovery_speed (if it dipped, how fast did it bounce)
# - premium_volatility (std dev of 15s changes)
# - source_reliability (which source was most accurate before fills)
```

## Rollout

1. Add schema + write functions to `postgres.py`
2. Add throttled tick writer to `position_monitor.py`
3. Run tests locally
4. Deploy — data starts accumulating immediately on next trading day
5. After 1 week of data: build exit timing features from premium ticks

## Success Criteria

- Premium ticks flowing to PG for all open trades across all 4 agents
- ~4,800 rows/hour during market hours (5 trades × 4 agents × 240 ticks)
- Zero impact on monitor loop latency (fire-and-forget)
- Premium source breakdown visible in data (what % comes from each source)
