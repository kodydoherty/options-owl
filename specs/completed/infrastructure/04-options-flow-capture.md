# Spec 04: Options Flow Capture (Unusual Activity Detection)

**Priority**: 4
**Effort**: Medium-High (6-8 hours)
**Impact**: Best new ML feature — institutional flow predicts 0DTE momentum

## Problem

We currently capture option chain **snapshots** (bid/ask/greeks every 60s) but NOT individual option **trades**. Large options trades (>50 contracts) from institutions and whales are the single strongest predictor of short-term directional movement. This is the data that paid flow services like Unusual Whales, Cheddar Flow, and FlowAlgo sell for $50-200/month.

We can capture this directly from Polygon's options WebSocket for free (included with Options Advanced subscription on the harvester key).

## Current State

### What Polygon WS Channels Exist

| Channel | What it provides | Currently used? |
|---|---|---|
| `AM.*` | Stock aggregate minute bars | YES — candle_collector for 5m bars |
| `A.*` | Stock per-second aggregates | No |
| `T.*` | Stock trades (every print) | No |
| `Q.*` | Stock quotes (NBBO changes) | No |
| `O.T.*` | **Option trades** (every print) | **NO — this is what we need** |
| `O.Q.*` | Option quote changes | Partially — via market_data_stream for subscribed contracts |
| `O.A.*` | Option aggregate minute bars | No |

### Polygon Options Trade Event (`O.T.*`)

Each event contains:
```json
{
    "ev": "OT",                    // event type
    "sym": "O:SPY260526C00530000", // OCC contract ticker
    "p": 2.45,                     // trade price
    "s": 150,                      // trade size (contracts)
    "c": [12, 41],                 // condition codes (12=intermarket sweep, 41=opening)
    "t": 1748275200000,            // SIP timestamp (ms)
    "x": 301                       // exchange ID
}
```

**Key fields for ML**: `s` (size — large = institutional), `p` (price vs bid/ask = aggressor detection), `c` (conditions — sweeps indicate urgency).

## Design

### Architecture

```
Polygon WS (options) ──→ FlowCollector ──→ filter ──→ PG: option_flow
      O.T.{universe}         │                            │
                              ▼                            ▼
                     aggregate by ticker/5min      ML features:
                     detect unusual activity       - net_flow_dollars
                                                   - sweep_ratio
                                                   - call_put_flow_ratio
                                                   - large_trade_count
```

### New PG Table: `option_flow`

```sql
CREATE TABLE IF NOT EXISTS option_flow (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,                -- underlying (SPY, TSLA, etc.)
    contract TEXT NOT NULL,              -- full OCC ticker (O:SPY260526C00530000)
    option_type TEXT NOT NULL,           -- 'call' or 'put'
    strike REAL NOT NULL,
    expiry_date TEXT NOT NULL,
    trade_price REAL NOT NULL,
    trade_size INTEGER NOT NULL,         -- number of contracts
    trade_value REAL NOT NULL,           -- price × size × 100 (dollar notional)
    conditions INTEGER[],               -- Polygon condition codes
    is_sweep BOOLEAN DEFAULT FALSE,     -- condition 12 = intermarket sweep
    is_above_ask BOOLEAN,               -- buyer aggressor (bullish)
    is_below_bid BOOLEAN,               -- seller aggressor (bearish)
    exchange_id INTEGER,
    sip_timestamp TIMESTAMPTZ NOT NULL,  -- exchange timestamp
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_flow_ticker ON option_flow(ticker, sip_timestamp);
CREATE INDEX IF NOT EXISTS idx_flow_size ON option_flow(trade_size) WHERE trade_size >= 50;
CREATE INDEX IF NOT EXISTS idx_flow_sweep ON option_flow(ticker, sip_timestamp) WHERE is_sweep = TRUE;
```

### New PG Table: `option_flow_5m` (Aggregated)

Pre-aggregated 5-minute flow summaries for efficient ML feature reads:

```sql
CREATE TABLE IF NOT EXISTS option_flow_5m (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    -- Call flow
    call_volume INTEGER NOT NULL DEFAULT 0,
    call_value REAL NOT NULL DEFAULT 0,
    call_sweeps INTEGER NOT NULL DEFAULT 0,
    call_large_trades INTEGER NOT NULL DEFAULT 0,  -- size >= 50
    call_buyer_aggressor_pct REAL,                  -- % of volume above ask
    -- Put flow
    put_volume INTEGER NOT NULL DEFAULT 0,
    put_value REAL NOT NULL DEFAULT 0,
    put_sweeps INTEGER NOT NULL DEFAULT 0,
    put_large_trades INTEGER NOT NULL DEFAULT 0,
    put_buyer_aggressor_pct REAL,
    -- Derived
    call_put_ratio REAL,                            -- call_volume / put_volume
    net_flow_dollars REAL,                          -- call_value - put_value
    sweep_ratio REAL,                               -- sweeps / total_trades
    UNIQUE(ticker, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_flow_5m_ticker ON option_flow_5m(ticker, bar_time);
```

### FlowCollector (New Module)

New file: `options_owl/collectors/flow_collector.py`

```python
class FlowCollector:
    """Captures option trade flow from Polygon WS and stores to PostgreSQL.

    Runs in the harvester process alongside CandleCollector.
    Subscribes to O.T.{ticker} for all universe tickers.
    Filters trades: only stores size >= MIN_FLOW_SIZE (default 10 contracts).
    Aggregates 5-minute flow summaries and flushes to PG.
    """

    MIN_FLOW_SIZE = 10          # ignore tiny trades (retail noise)
    LARGE_TRADE_SIZE = 50       # flag as "large" for ML features
    FLUSH_INTERVAL = 30         # seconds between PG batch writes
    AGGREGATE_INTERVAL = 300    # 5 minutes for flow_5m bars
```

### WS Connection Strategy

The harvester already connects to `wss://socket.polygon.io/stocks` for stock minute bars via CandleCollector. Options flow requires the **separate** options WS endpoint: `wss://socket.polygon.io/options`.

**Important**: Polygon allows 1 concurrent WS connection per endpoint per API key. The harvester uses the stocks WS; we need a second connection to the options WS. This is allowed — they're different endpoints.

```python
# In harvester.py run_harvester():
flow_collector = FlowCollector(UNIVERSE)
await flow_collector.start_ws(POLYGON_API_KEY)  # connects to /options WS
```

### Filtering Strategy

Raw options flow for 13 tickers generates ~50,000-100,000 trades/day. We filter:

1. **Size filter**: Only store trades with `size >= 10` contracts (~5-10% of trades)
2. **Universe filter**: Only subscribed tickers (no random small-caps)
3. **Market hours only**: Skip pre/post-market flow (unreliable)
4. **Near-term only**: Only 0-7 DTE contracts (matches our trading window)

After filtering: ~2,000-5,000 trades/day stored, ~4,000 rows in flow_5m.

### Aggressor Detection

Determine if a trade is buyer-initiated or seller-initiated:
```python
def _detect_aggressor(trade_price: float, bid: float, ask: float) -> str:
    """Detect trade aggressor from price vs NBBO.

    - Price >= ask → buyer aggressor (bullish for calls, bearish for puts)
    - Price <= bid → seller aggressor (bearish for calls, bullish for puts)
    - Price between → indeterminate
    """
    if bid > 0 and trade_price <= bid:
        return "seller"
    elif ask > 0 and trade_price >= ask:
        return "buyer"
    return "neutral"
```

Need the current bid/ask for aggressor detection — use the option_ticks table (harvester snapshots every 60s) or maintain a local NBBO cache from `O.Q.*` events.

**Decision**: Subscribe to `O.Q.*` for universe tickers too, maintain local NBBO cache. This adds minimal overhead and enables accurate aggressor detection.

## Files to Create

| File | Purpose |
|---|---|
| `options_owl/collectors/flow_collector.py` | FlowCollector class — WS connection, filtering, aggregation, PG writes |

## Files to Modify

| File | Change |
|---|---|
| `options_owl/db/postgres.py` | Add `option_flow` + `option_flow_5m` tables to schema, add write/read functions |
| `options_owl/harvester.py` | Initialize FlowCollector, start options WS, flush alongside candle flushes |
| `docker-compose.yml` | No changes needed (harvester already has Polygon key) |

## Tests

| Test | What it validates |
|---|---|
| `test_flow_collector_filters_small_trades` | Trades with size < 10 are dropped |
| `test_flow_collector_detects_sweeps` | Condition code 12 sets `is_sweep=True` |
| `test_flow_collector_aggressor_detection` | Price at/above ask = buyer, at/below bid = seller |
| `test_flow_5m_aggregation` | 5-minute bars correctly aggregate call/put volume, sweeps, large trades |
| `test_flow_collector_pg_write` | Batch write to `option_flow` table works |
| `test_flow_5m_pg_write` | Upsert to `option_flow_5m` works |
| `test_flow_collector_ws_reconnect` | Reconnects on WS disconnect (same pattern as CandleCollector) |
| `test_flow_near_term_filter` | Only stores flow for contracts with DTE <= 7 |
| `test_flow_schema_exists` | PG schema includes both flow tables |

## ML Features Extractable

Once flow data is accumulating, these features can be computed per-ticker per-5min:

| Feature | Description | Expected Signal |
|---|---|---|
| `net_flow_dollars` | call_value - put_value | Positive = bullish consensus |
| `call_put_ratio` | call_volume / put_volume | >2 = strongly bullish |
| `sweep_count_5m` | Intermarket sweeps in last 5 min | High = urgent institutional buying |
| `large_trade_pct` | % of volume from trades >= 50 contracts | High = institutional, low = retail |
| `buyer_aggressor_pct` | % of volume traded at/above ask | High = buyers paying up = bullish |
| `flow_acceleration` | Change in net_flow vs prior 5m bar | Spike = momentum shift |
| `relative_flow` | Today's flow vs 5-day avg at same time | Unusual = signal |
| `put_sweep_spike` | Put sweeps > 3x normal | Bearish — institutions hedging |

### Integration with Existing Models

- **pattern_entry model**: Add `net_flow_dollars`, `sweep_count_5m`, `call_put_ratio` as features
- **regime_classifier**: Add `flow_acceleration`, `relative_flow` — detects institutional activity shifts
- **entry_timing model**: Add `buyer_aggressor_pct` — confirms momentum direction
- **PUT detection**: `put_sweep_spike` directly signals bearish institutional flow

## Rollout Plan

### Phase 1: Capture Raw Flow (Week 1)
1. Create FlowCollector + PG tables
2. Wire into harvester
3. Deploy — start accumulating data
4. Monitor: row counts, data quality, WS stability

### Phase 2: Aggregation (Week 1-2)
1. Add 5-minute aggregation
2. Add `read_flow_5m()` function to postgres.py
3. Validate aggregated data quality

### Phase 3: ML Integration (Week 3+)
1. Add flow features to training scripts
2. Retrain pattern_entry with flow features
3. A/B test: with-flow vs without-flow model accuracy
4. If improved, deploy to sourcing scanner

## Data Volume Estimates

| Metric | Estimate |
|---|---|
| Raw trades/day (pre-filter) | ~50,000-100,000 |
| Stored trades/day (size >= 10) | ~2,000-5,000 |
| Flow_5m rows/day | ~4,000 (13 tickers × ~78 5min bars × ~4 active bars/ticker) |
| PG storage/day | ~5-10 MB |
| PG storage/month | ~200 MB |
| WS bandwidth | Minimal (text events, ~1KB each) |

## Status: BLOCKER — Polygon Options WS Not Included in Plan

**Discovered 2026-05-26**: The harvester's Polygon API key authenticates successfully on `wss://socket.polygon.io/options`, but gets disconnected with **1008 policy violation** ~10 seconds after subscribing. This means the subscription tier includes Options REST (chain snapshots work) but NOT Options WebSocket streaming.

### Fix Applied
- FlowCollector now detects consecutive 1008 policy violations and backs off aggressively (180s+ between retries) instead of rapid-looping every 1s. The harvester is stable — FlowCollector just sleeps quietly.

### Options to Unblock
1. **Upgrade Polygon plan** — Add Options Advanced WebSocket add-on (check pricing at polygon.io/pricing)
2. **REST polling fallback** — Poll Polygon's `/v3/trades/O:{ticker}` endpoint every 30-60s for recent trades. Lower fidelity but no additional cost. Captures large trades with delay.
3. **Wait for market open** — Today is Memorial Day (markets closed). Polygon may reject Options WS connections when markets are closed. Test again on Tuesday 2026-05-27 during market hours before upgrading.

**Recommendation**: Test on Tuesday first (Option 3). If still 1008, implement REST polling fallback (Option 2) as it's free and captures 80% of the value.

## Risks

| Risk | Mitigation |
|---|---|
| **Options WS not in Polygon plan** | **ACTIVE BLOCKER** — 1008 policy violation. See Status section above. |
| Options WS adds second connection — Polygon limits? | Stocks and options are separate endpoints, each allows 1 concurrent connection |
| High-volume tickers flood PG | Size filter (>= 10 contracts) drops 90% of noise. Can raise threshold if needed |
| Aggressor detection inaccurate without real-time NBBO | Subscribe to O.Q.* for NBBO cache. Fallback: use option_ticks bid/ask (60s stale) |
| Flow data not predictive for 0DTE | Unlikely — institutional 0DTE flow is well-documented as predictive. Validate in backtest |
| Latency on options WS | Non-critical — flow data is for feature computation, not real-time trading decisions |

## Success Criteria

- Option flow data accumulating in PG for all 13 universe tickers
- 2,000-5,000 filtered trades/day stored
- 5-minute aggregated flow bars computed and available
- At least 2 flow features show >0.05 correlation with trade P&L in backtesting
- Pattern entry model AUC improves by >0.02 when flow features are added
