# ML Signal Sourcing — Plan & Progress

## Goal
Replace reliance on Discord analyst signals with ML models that detect profitable 0DTE option entry patterns directly from market data. Per-ticker models because each ticker has unique volatility/flow characteristics.

## Architecture

```
ThetaData (historical) ──┐
                          ├─→ Feature Engineering ──→ LightGBM per-ticker ──→ Signal
Live Polygon + UW Flow ──┘
```

**Training:** Historical option OHLC + quotes + greeks → feature extraction → production V5 FSM simulation for labels → LightGBM classifier per ticker.

**Production:** Real-time data → same features → model predict → if confidence > threshold → trigger entry → V5 FSM manages exit.

## Data Sources

| Source | Status | What it provides |
|---|---|---|
| ThetaData Options Standard ($80/mo) | Active | 1-min option OHLC, bid/ask, IV/delta/theta/vega, underlying price |
| Polygon REST/WS | Active | Real-time quotes, options snapshots |
| Unusual Whales | Key obtained | Historical flow, dark pool, GEX, max pain |
| Twelve Data | Active | Technical indicators (5m/15m candles) |

### API Keys
All API keys stored in `.env` on the droplet — never commit secrets to git.

## Results So Far

### SPY (34 training days, 9 test days) — 2026-05-21
- **AUC: 0.814** (strong signal quality)
- **Precision: 65.8%** — of predicted setups, 2/3 actually moved +15%
- **Recall: 82.1%** — catches 4 out of 5 real moves
- **Backtest P&L: +$15,192** ($20K → $35K in 9 days, 63.6% WR)
- Uses real production V5 FSM for exit simulation (not simplified thresholds)
- Top features: delta (0.50 = ATM sweet spot), premium volatility, premium price, theta, vega, time of day

### Key finding: Real exit engine is critical
- Simplified +15%/-35% thresholds: **-$2,374** (money-losing)
- Production V5 FSM: **+$15,192** (highly profitable)
- V5's scaleouts, breakeven ratchet, and adaptive trails capture gains the simple sim misses

## Feature Set (V2)

### Premium Action
- `premium`, `premium_change_5m/10m/15m` — momentum
- `premium_volatility`, `premium_skew` — is premium calm (coiled) or noisy?
- `range_position` — where within recent range (0=low, 1=high)
- `consecutive_up/down_bars` — trend strength

### Volume
- `volume_ratio` — current vs average
- `volume_trend` — recent vs older volume
- `volume_zscore` — how unusual is current volume

### Bid/Ask Microstructure
- `spread`, `spread_pct` — tightness
- `spread_tightening` — institutional interest indicator
- `size_imbalance` — bid vs ask pressure

### Greeks
- `iv`, `delta`, `theta`, `vega` — current values
- `iv_change_5m/15m`, `iv_trend` — IV dynamics (rising IV = smart money)

### Underlying
- `underlying_change_5m/15m` — stock momentum
- `underlying_volatility` — recent stock volatility
- `vwap_deviation` — distance from intraday average

### Computed Patterns
- `coiled_spring` — low vol + building volume = breakout coming
- `volume_breakout` — big surge + price at high
- `bounce_setup` — price at low + spread tightening
- `iv_expanding` — IV rising (positioning)
- `momentum_ignition` — consecutive up + rising volume

## What Works

| Approach | Result | Notes |
|---|---|---|
| V2 ML + real V5 FSM | +$15K / 9 days | Best so far. Real exit engine is critical. |
| V1 ML + simplified exits | -$2.4K / 7 days | Simplified exits lose money even with good entries |
| Rule-based sourcing (technical only) | -$20K / 68 days | Scores without alpha data have no edge (~44% WR) |
| Rule-based sourcing (rescaled) | Same as raw | Rescaling shifts thresholds but doesn't improve quality |

## What Doesn't Work

| Approach | Why |
|---|---|
| Simplified +15%/-35% exits | Misses scaleout profits, doesn't adapt to ticker/DTE/time-of-day |
| Technical-only scoring (no alpha) | 44% win rate — essentially random. Missing UW flow, congress, sentiment data. |
| Random timestamp sampling (V1) | Creates noisy labels. Better to find actual moves, then learn what preceded them. |
| Universal model for all tickers | MSTR moves 5-10% daily, SPY moves 0.5% — need per-ticker models |

## Roadmap

### Phase 1: Data Collection (IN PROGRESS)
- [x] ThetaData download script with retry, batching, OTM strikes
- [x] SPY 34 days downloaded (Jan-May partial)
- [x] TSLA 1 day downloaded (with 5 OTM strikes)
- [x] Full 14-ticker download started — 1.16M rows, day 21/100 (Jan 26)
- [ ] Full 14-ticker download running: `tail -f /tmp/thetadata_download_full.log` (PID 62744)

### Phase 1b: UW Flow Data (IN PROGRESS)
- [x] UW API tested — key works, endpoints mapped
- [x] Download script: `scripts/download_uw_historical.py`
- [x] Daily GEX (greek-exposure): 251 days × 14 tickers downloaded (~1yr history)
- [x] Congress trades: 200 records downloaded
- [ ] Intraday net premium ticks (1-min): downloading 30 days × 14 tickers
- [ ] Intraday spot GEX (1-min): downloading 30 days × 14 tickers
- [ ] Dark pool trades: downloading 30 days × 14 tickers
- [ ] Flow alerts (unusual activity): downloading 30 days × 14 tickers
- [ ] Options volume (daily): downloading 30 days × 14 tickers
- [ ] Max pain: downloading 30 days × 14 tickers
- [ ] Download running: `tail -f /tmp/uw_download.log` (PID 65219)
- [ ] DB: `journal/uw_historical.db`

#### UW API — What's Available

| Endpoint | Data | History | Resolution |
|---|---|---|---|
| `stock/{ticker}/greek-exposure` | GEX (gamma, delta, charm, vanna) | ~1yr (251 days) | Daily |
| `stock/{ticker}/options-volume` | Call/put volume, premium, OI | 30 trading days | Daily |
| `stock/{ticker}/net-prem-ticks` | Net call/put premium + volume | 30 trading days | 1-min |
| `stock/{ticker}/spot-exposures` | Per-minute GEX | 30 trading days | 1-min |
| `stock/{ticker}/flow-alerts` | Unusual flow (sweeps, repeats) | 30 trading days | Per-alert |
| `darkpool/{ticker}` | Dark pool trades | 30 trading days | Per-trade |
| `stock/{ticker}/max-pain` | Max pain by expiry | 30 trading days | Per-expiry |
| `congress/recent-trades` | Congressional buy/sell | Unlimited | Per-trade |

**Limitation:** Intraday endpoints only go back 30 trading days. For full historic access, email dev@unusualwhales.com.

### Phase 2: Per-Ticker Models (NEXT)
- [x] SPY model trained — AUC 0.814, +$15K backtest
- [ ] Train TSLA, NVDA, MSTR, META, QQQ, AAPL, AMZN, GOOGL, MSFT, AMD, PLTR, AVGO, IWM
- [ ] Compare per-ticker vs generic model performance
- [ ] Determine minimum data per ticker for reliable model (30 days? 60?)

### Phase 3: Add UW Flow Features
- [x] UW API key verified: `0294df1c-4517-4c0a-bae9-f037a39aa5ef`
- [x] Download script built with retry, resume, rate limiting
- [ ] Merge UW daily features (GEX bias, put/call ratio, flow sentiment) into training dataset
- [ ] Merge UW intraday features (net premium flow, spot GEX) where dates overlap (Apr 10+)
- [ ] Add features: net premium flow direction, dark pool size vs avg, GEX flip detection
- [ ] Retrain models with UW features — expect significant precision improvement
- [ ] Congress trades as lagging indicator (buy what congress bought)

### Phase 4: Production Integration
- [ ] Wire trained models into sourcing agent scan loop
- [ ] Real-time feature computation from Polygon WS + UW API
- [ ] Model predict → if confidence > threshold → emit signal to Discord + Supabase
- [ ] A/B test: ML signals vs Discord signals (run both, compare P&L)

### Phase 5: Live Trading
- [ ] Route ML signals through existing entry pipeline (smart entry, premium cap, etc.)
- [ ] Paper trade for 2 weeks before live
- [ ] Start with SPY/QQQ (highest data, most liquid) then add per-ticker models

## Scripts

| Script | Purpose |
|---|---|
| `scripts/download_thetadata.py` | Download historical options data from ThetaData |
| `scripts/download_uw_historical.py` | Download historical UW flow/GEX/dark pool data |
| `scripts/train_option_signals_v2.py` | Train per-ticker ML models + backtest with real V5 FSM |
| `scripts/train_option_signals.py` | V1 (deprecated) — random sampling, simplified exits |
| `scripts/backtest_sourcing_pnl.py` | Rule-based sourcing backtest (confirmed no edge without alpha) |

## Databases

### ThetaData Options — `journal/thetadata_options.db` (SQLite WAL)

| Table | Contents |
|---|---|
| `option_ohlc` | 1-min option OHLC (open/high/low/close/volume/vwap) |
| `option_quotes` | 1-min bid/ask + sizes |
| `option_greeks` | 1-min IV/delta/theta/vega + underlying_price |
| `stock_ohlc` | Underlying prices (extracted from greeks) |
| `download_log` | Tracks what's been downloaded (for resume) |

### Unusual Whales — `journal/uw_historical.db` (SQLite WAL)

| Table | Contents |
|---|---|
| `greek_exposure` | Daily GEX: gamma, delta, charm, vanna (call+put) — ~1yr history |
| `options_volume` | Daily volume, premium, OI, bearish/bullish split |
| `net_prem_ticks` | Intraday (1-min) net premium flow + volume + delta |
| `spot_gex` | Intraday (1-min) spot GEX by OI and volume |
| `flow_alerts` | Unusual flow detections (sweeps, repeated hits, floor trades) |
| `darkpool` | Dark pool trades with NBBO context |
| `max_pain` | Max pain per expiry per day |
| `congress_trades` | Congressional buy/sell disclosures |
| `download_log` | Resume tracking |

Models saved to: `journal/models/signal_ml_v2/signal_{TICKER}.lgb`
