# Spec 03: Webull as Primary Data Source for Trading Bots

**Priority**: 3
**Effort**: Medium (4-6 hours)
**Impact**: Saves $600-800/month by eliminating per-agent Polygon API keys

## Problem

Each of the 4 trading agents has its own Polygon API key for real-time option pricing. At $200/month per Options Advanced plan, that's $800/month just for option quotes. Meanwhile, every agent already has an authenticated Webull connection with a working `DataClient` that provides the same bid/ask/mid data — from the **same exchange** where we're executing trades.

Using Webull quotes also eliminates the "estimation gap" — currently Polygon quotes may differ from Webull fills because they're different data sources.

## Current State

### Data Sources in Trading Bots (4 agents)

| Usage | Current Source | File | Polygon Required? |
|---|---|---|---|
| **Entry premium verification** | Polygon REST `polygon_option_quote()` | `paper_trader.py:513,2190` | Yes ($200/mo each) |
| **Exit premium monitoring** | Polygon WS → Polygon REST → yfinance → delta | `position_monitor.py:697,1039` | Yes (primary) |
| **Stock price streaming** | Polygon WS `AM.*` minute bars | `market_data_stream.py` | Yes |
| **Dip-confirm premium** | Polygon WS option quotes | `market_data_stream.py` | Yes |
| **Option chain (sourcing)** | Polygon REST `polygon_option_chain()` | `ml_pipeline.py:928` | Yes (sourcing only) |
| **Bulk chain snapshots** | Polygon `/v3/snapshot/options` | `harvester.py` | Yes (harvester only) |

### What Webull Already Provides

`webull_executor.py` already has:
- `get_option_quote()` (line 1032) — real-time bid/ask/mid for any option contract
- `_lookup_instrument_id()` — resolves ticker+strike+expiry to Webull instrument_id
- `_ensure_data_client()` — lazy-inits `DataClient` for market data
- `_quote_cache` with 3s TTL — prevents hammering the API
- Two fallback methods: `get_snapshot` → `get_quotes`

### What We Can Drop Per Agent

After this change, trading agents need **zero Polygon API keys**. Only the harvester and sourcing agent need Polygon (1 key each for bulk chain pulls and scanning).

**Savings: 4 agents × $200/mo = $800/mo → $0/mo for agent quotes**

(Keep 1 Polygon key for harvester bulk snapshots = $200/mo, 1 for sourcing = $200/mo)

## Design

### Phase 1: Webull Option Quotes as Primary (position_monitor + paper_trader)

Replace the Polygon REST calls in the premium cascade with Webull `get_option_quote()`.

#### position_monitor.py — Premium Cascade Change

Current cascade (4 sources):
```
1. Polygon WS (market_data_stream) → fast, real-time
2. Polygon REST (polygon_option_premium) → near real-time
3. yfinance option chain → delayed 15-30s
4. Delta approximation → last resort estimate
```

New cascade (4 sources, Webull-first):
```
1. Webull DataClient (get_option_quote) → real-time, same venue as execution
2. Polygon WS (market_data_stream) → keep as fallback if Webull fails
3. yfinance option chain → delayed fallback
4. Delta approximation → last resort estimate
```

The position_monitor needs access to the `WebullExecutor` instance (it already has it for selling — stored as `webull_executor` in the monitor context). We just wire `get_option_quote()` into the premium fetch cascade.

#### paper_trader.py — Entry Premium Verification

`_verify_live_premium()` and `_fill_missing_premium()` use `polygon_option_quote()`. Replace with Webull quote, fall back to Polygon if Webull unavailable.

```python
# In _verify_live_premium():
# Try Webull first (same venue = most accurate)
if webull_executor:
    quote = await webull_executor.get_option_quote(ticker, strike, expiry, option_type)
    if quote and quote.get("mid"):
        return quote["mid"]

# Fallback to Polygon REST
from options_owl.collectors.polygon_options import polygon_option_quote
quote = await polygon_option_quote(api_key, ticker, strike, expiry_date, option_type)
```

### Phase 2: Webull Stock Price Streaming (market_data_stream.py)

Replace Polygon WS stock price streaming with Webull's market data. The `DataClient` supports:
- `market_data.get_snapshot(instrument_id, Category.US_STOCK)` — current price
- `market_data.get_quotes(instrument_id, Category.US_STOCK)` — bid/ask/depth

For stock prices, yfinance polling (current fallback) is actually fine — stock prices are only used for underlying tracking, not exit decisions. Option premium is what matters for exits.

**Decision: Keep yfinance for stock prices, Webull for option premiums.** This is simpler and stock price accuracy is less critical than option premium accuracy.

### Phase 3: Remove Per-Agent Polygon Keys

Once Webull is stable as primary:
1. Remove `POLYGON_API_KEY` from trading bot env vars in docker-compose.yml
2. Keep `POLYGON_API_KEY` only in harvester and sourcing containers
3. Set `ENABLE_POLYGON_WS=false` for trading bots (no WS connection needed)
4. Remove Polygon WS connection code from trading bot startup

### Webull Rate Limits

Webull OpenAPI rate limits (from SDK docs):
- Market data: **120 requests/minute** per app (shared across all agents using same APP_KEY)
- Each agent has its own APP_KEY, so 120 req/min per agent
- At 5s poll × 5 trades = 60 requests/min per agent → within limits
- Quote cache (3s TTL) prevents duplicate requests within same cycle

**Risk mitigation**: If Webull rate-limits us, fall back to Polygon/yfinance transparently. The cascade already handles this.

### Error Handling

Webull `get_option_quote()` already returns `None` on failure. The cascade pattern means any failure transparently falls through to the next source. No new error handling needed — just wire Webull as source #1.

## Files to Modify

| File | Change | Risk |
|---|---|---|
| `options_owl/execution/position_monitor.py` | Add Webull quote as first source in premium cascade | **HIGH** — this is the sell path. Must test thoroughly. |
| `options_owl/execution/paper_trader.py` | Add Webull quote in `_verify_live_premium()` and `_fill_missing_premium()` | Medium — entry path, less critical |
| `options_owl/collectors/market_data_stream.py` | Add `DataFeedProvider.WEBULL` implementation for option premiums | Medium |
| `options_owl/config/settings.py` | Add `WEBULL_PRIMARY_QUOTES: bool = True` feature flag | Low |
| `docker-compose.yml` | Phase 3: remove POLYGON_API_KEY from trading bot envs | Low (after validation) |

## Feature Flag

Gate behind `WEBULL_PRIMARY_QUOTES=true` (default false initially). This allows:
1. Deploy code to all agents
2. Enable on 1 agent (owlet-kody) for validation
3. Monitor for 2-3 trading days
4. Enable on all agents
5. Remove Polygon keys after 1 week stable

## Tests

| Test | What it validates |
|---|---|
| `test_webull_quote_in_premium_cascade` | position_monitor tries Webull before Polygon when flag enabled |
| `test_webull_quote_fallback_to_polygon` | When Webull returns None, falls through to Polygon REST |
| `test_webull_quote_in_entry_verification` | paper_trader uses Webull quote for premium verification |
| `test_webull_rate_limit_respected` | Quote cache prevents >120 req/min per agent |
| `test_feature_flag_disabled` | When WEBULL_PRIMARY_QUOTES=false, cascade unchanged |
| `test_premium_source_tracking` | Premium tick storage (spec 01) records "webull" as source |
| `test_no_polygon_key_graceful` | Trading bot works with no POLYGON_API_KEY when Webull is primary |

## Integration Tests

| Test | What it validates |
|---|---|
| `test_webull_quote_vs_polygon_accuracy` | Compare Webull vs Polygon quotes for same contract over 1 day — measure spread |
| `test_webull_quote_latency` | Webull quote returns in <1s (comparable to Polygon REST) |
| `test_full_monitor_loop_webull_primary` | End-to-end monitor loop with Webull as primary source |

## Rollout Plan

### Week 1: Deploy with Flag Off
1. Implement all code changes
2. Add `WEBULL_PRIMARY_QUOTES=false` to settings
3. Deploy to all agents — no behavior change
4. Verify no regressions

### Week 2: Validate on One Agent
1. Set `WEBULL_PRIMARY_QUOTES=true` on owlet-kody only
2. Monitor premium source distribution in logs: `grep 'premium_source' logs/`
3. Compare Webull vs Polygon fill accuracy
4. Check rate limits aren't being hit

### Week 3: Enable All + Remove Polygon Keys
1. Enable on all 4 agents
2. Monitor for 2 trading days
3. Remove POLYGON_API_KEY from trading bot docker-compose envs
4. Set ENABLE_POLYGON_WS=false for trading bots
5. Cancel 3 Polygon subscriptions ($600/mo saved)

### Keep Running
- 1 Polygon key for harvester (bulk chain snapshots — Webull can't do this)
- 1 Polygon key for sourcing (option chain scanning)
- Total: $400/mo → down from $1,000/mo = **$600/mo saved**

## Risks

| Risk | Mitigation |
|---|---|
| Webull quote API goes down | Cascade falls through to Polygon/yfinance — same as today |
| Webull rate limits during volatile sessions | Quote cache (3s TTL) + per-agent APP_KEY = 120 req/min headroom |
| Webull quotes differ from Polygon | They SHOULD — Webull is the execution venue, so Webull quotes are more accurate for our fills |
| Webull SDK connection stale | Already handled by `_reconnect()` in webull_executor.py |

## Success Criteria

- All 4 trading agents using Webull as primary option quote source
- Premium source logs show >90% of quotes from Webull
- No increase in premium blackout alerts
- Fill accuracy improves (Webull quote = Webull fill, no estimation gap)
- 3 Polygon subscriptions cancelled = $600/mo saved
