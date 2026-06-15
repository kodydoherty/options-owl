# Spec 02: VIX in Harvester Universe

**Priority**: 2
**Effort**: Trivial (15 minutes)
**Impact**: Regime model improvement (currently AUC=0.616 — worst performing model)

## Problem

The regime classifier has the lowest AUC of all V3 models (0.616). It tries to predict "chop days" to skip trading but only uses per-ticker price features. VIX is the single most important volatility regime indicator and we're not capturing it in real-time.

The harvester polls every 60s and captures option chain snapshots + stock ticks for its universe. VIX is not in the universe, so we have no real-time VIX data in PostgreSQL. The regime model was retrained with "morning-only" features but still can't see actual market fear levels.

## Current State

- Harvester universe: `SPY,QQQ,IWM,AAPL,TSLA,NVDA,META,AMD,AMZN,GOOGL,MSFT,MU,MSTR`
- VIX is NOT included
- Regime model uses: `drop_from_open`, `range_vs_atr`, `volume_ratio`, `vwap_distance` — all per-ticker
- No VIX option chain needed — just the underlying VIX index level
- yfinance uses `^VIX` ticker format for index symbols

## Design

### Change 1: Add VIX to Harvester Universe

In `harvester.py`, the universe is env-driven:
```python
UNIVERSE = [t.strip().upper() for t in os.getenv(
    "HARVEST_UNIVERSE",
    "SPY,QQQ,IWM,AAPL,TSLA,NVDA,META,AMD,AMZN,GOOGL,MSFT,MU,MSTR",
).split(",") if t.strip()]
```

Add VIX to the default:
```python
"SPY,QQQ,IWM,AAPL,TSLA,NVDA,META,AMD,AMZN,GOOGL,MSFT,MU,MSTR,VIX"
```

### Change 2: Handle VIX in `_get_underlying_quote()`

VIX already works — `_get_underlying_quote()` has special handling:
```python
yf_symbol = f"^{ticker}" if ticker in ("VIX", "GSPC", "DJI", "IXIC") else ticker
```

So stock ticks will flow to PG automatically.

### Change 3: Skip VIX Option Chain

VIX options trade under a different ticker (VIX options are on CBOE, not standard equity options). The Polygon chain fetch will return empty results for VIX, which is fine — `_harvest_ticker()` handles empty chains gracefully. But to avoid wasted API calls:

```python
# In _harvest_ticker(), skip chain fetch for index-only tickers
INDEX_ONLY_TICKERS = {"VIX"}  # no standard equity options chain
if ticker in INDEX_ONLY_TICKERS:
    # Still write stock tick, skip option chain
    await _persist_stock_tick_only(ticker, quote)
    return 0
```

### Change 4: Add VIX to CandleCollector WS

The `CandleCollector` subscribes to Polygon stock WS for minute bars. VIX doesn't have a stock WS stream (it's a calculated index), so candle_collector should skip VIX WS subscription. The yfinance poll in harvester is sufficient for VIX (60s granularity is fine for regime detection).

### Change 5: Update docker-compose.yml

Update `HARVEST_UNIVERSE` env var in the harvester service to include VIX (or rely on the new default).

## Files to Modify

| File | Change |
|---|---|
| `options_owl/harvester.py` | Add VIX to default UNIVERSE, add INDEX_ONLY_TICKERS skip for chain fetch, add `_persist_stock_tick_only()` helper |
| `options_owl/collectors/candle_collector.py` | Skip VIX in WS subscription (no stock WS for indices) |
| `docker-compose.yml` | Update HARVEST_UNIVERSE if explicitly set (or let new default take effect) |

## Tests

| Test | What it validates |
|---|---|
| `test_vix_in_default_universe` | VIX is in default HARVEST_UNIVERSE |
| `test_vix_skips_option_chain` | `_harvest_ticker("VIX", ...)` writes stock tick but skips Polygon chain fetch |
| `test_vix_yfinance_quote` | `_get_underlying_quote("VIX")` returns valid price (uses `^VIX` format) |
| `test_vix_not_in_candle_ws` | CandleCollector skips VIX for WS subscription |

## ML Training Usage

With VIX stock_ticks in PG, regime model can use:
- `vix_level` — absolute VIX value (>20 = elevated, >30 = fear, >40 = panic)
- `vix_change_pct` — intraday VIX movement (VIX spike = regime shift)
- `vix_vs_5d_avg` — VIX relative to recent average (mean reversion signal)
- `vix_at_open` — morning VIX for day classification

These features alone should significantly improve regime AUC. VIX spike days ARE the chop days — it's the direct signal the model is missing.

## Rollout

1. Update harvester.py + candle_collector.py
2. Run tests
3. Deploy — VIX ticks start flowing immediately
4. After 5 trading days: retrain regime model with VIX features

## Success Criteria

- VIX stock_ticks in PG updating every 60s during market hours
- No wasted Polygon API calls on VIX option chain
- Regime model AUC improves from 0.616 to >0.75 after retraining with VIX features
