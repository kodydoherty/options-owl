# 04 - Backtest Plan: Signal Sourcing Bot

## Objective

Validate that `owlet-sourcing` produces equal or better signals than the N8N workflow it replaces. Specifically:

1. Reproduce N8N's historical signals with the new scoring engine
2. A/B test each data source's marginal contribution to signal quality
3. Reverse-engineer historical market states to backfill a ground-truth dataset
4. Optimize the new 0-100 scoring system so score correlates with win rate

## Available Historical Data

### Harvester Options DB (`journal/owlet-harvester/options_data.db`, ~7GB)

| Table | Row Count | Date Range | Key Columns |
|---|---|---|---|
| `harvest_snapshots` | ~23M | 2026-03-27 to present | `contract_ticker, captured_at, underlying_price, bid, ask, midpoint, implied_volatility, delta, gamma, theta, vega, open_interest, day_volume` |
| `harvest_contracts` | (lookup) | — | `contract_ticker, underlying, strike, expiry_date, option_type` |
| `stock_candles` | ~10K | 2026-05-14 to present | `ticker, timeframe, bar_start_ts, bar_start, open, high, low, close, volume, vwap` |

**Important:** `stock_candles` only covers the last week. For candle data before 2026-05-14, we must pull from Polygon REST API and cache locally (or rely on the 23M options snapshots which have `underlying_price` as a proxy).

**Tickers in harvester:** AAPL, AMD, AMZN, BA, COIN, DIA, GLD, GOOGL, IWM, JPM, META, MSFT, MSTR, MU, NFLX, NVDA, PLTR, QQQ, SLV, SMCI, SPY, TLT, TSLA, VIX, XLF, XLK (26 tickers).

### Trade History DB (`journal/owlet-kody/raw_messages.db`)

| Table | Row Count | Date Range | Key Columns |
|---|---|---|---|
| `trade_signals` | 330 | 2026-04-10 to 2026-05-18 | `bot_source, ticker, sentiment, direction, score, entry_price, strike, expiry, atm_premium, key_signals, created_at` |
| `paper_trades` | 220 closed | 2026-04-13 to 2026-05-18 | `signal_id, ticker, direction, score, premium_per_contract, contracts, entry_price, strike, exit_premium, exit_reason, pnl_dollars, pnl_pct, webull_order_id, exit_source, opened_at, closed_at` |
| `trade_events` | (audit) | — | `trade_id, ticker, event_type, detail, created_at` |
| `raw_messages` | (all) | — | `content, timestamp, author_name, channel_id` |

### Supabase

- `alerts` table (via `v_alerts_with_conviction` view) -- N8N signal history with Bayesian signatures, ML scores, timestamps
- `fills` table -- Webull fill data synced from owlets
- `closes` table -- Exit data

### Polygon REST API (backfillable)

- 5-minute candles for any ticker, any historical date (within plan limits)
- News sentiment for any ticker/date
- Options chain snapshots (redundant with harvester but available for gap-filling)

### What We Cannot Backfill

| Source | Status | Impact |
|---|---|---|
| Unusual Whales flow | Not stored historically | Test marginal value going forward only |
| Grok AI analysis | Not stored historically | Assume neutral (no direction change) for backtest |
| N8N internal state (cooldowns, trade history) | Ephemeral, lost on restart | Reconstruct from `trade_signals` + `paper_trades` |
| Twelve Data indicator values | N8N used these, we use Polygon | Must verify our local computation matches closely enough |

---

## Backtest Architecture

```
Historical Market Data (harvester DB + Polygon REST)
         |
         v
    +------------------+
    |  Reconstruct     | <-- For each timestamp:
    |  Market State    |     - Pull 5min candles (78 bars = full session)
    |                  |     - Compute EMA, BB, RSI, MACD, VWAP, ATR locally
    |                  |     - Pull options chain snapshot (nearest to signal time)
    +--------+---------+
             |
             v
    +------------------+
    |  Score Signal    | <-- Apply scoring engine with configurable sources
    |  (new engine)    |     - Feature flags: which sources enabled
    |                  |     - Returns: 0-100 score, breakdown, direction
    +--------+---------+
             |
             v
    +------------------+
    |  Simulate Trade  | <-- Would this signal have been traded?
    |  Decision        |     - Apply entry pipeline gates
    |                  |     - Record: would_trade, score, direction
    +--------+---------+
             |
             v
    +------------------+
    |  Evaluate        | <-- Look up actual outcome
    |  Outcome         |     - From paper_trades: real P&L (exit_source='ai')
    |                  |     - From harvest_snapshots: premium at signal+N min
    +--------+---------+
             |
             v
    +------------------+
    |  Aggregate       | <-- Compute metrics
    |  Metrics         |     - Win rate, avg P&L, Sharpe, false positive rate
    +------------------+
```

---

## Phase 1: Data Reconstruction Pipeline (Week 1)

### Step 1a: Build Historical Candle Cache

The harvester `stock_candles` table only covers 2026-05-14 onward. For the full backtest window (Apr 10 - present), we need 5-minute candles from Polygon REST.

**Script: `scripts/build_candle_cache.py`**

```python
"""Download and cache 5-min candles from Polygon for the full backtest window."""

import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path
import requests

POLYGON_API_KEY = "..."
CACHE_DB = Path("journal/backtest_candles.db")

# All tickers that appear in trade_signals
TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "META", "AMD",
    "AMZN", "GOOGL", "MSFT", "MU", "MSTR", "AVGO", "PLTR", "SMCI",
    "COIN", "BA", "NFLX", "JPM",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles_5m (
    ticker TEXT NOT NULL,
    bar_ts_ms INTEGER NOT NULL,  -- unix epoch ms
    bar_time TEXT NOT NULL,       -- ISO 8601
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    vwap REAL DEFAULT 0,
    transactions INTEGER DEFAULT 0,
    UNIQUE(ticker, bar_ts_ms)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles_5m(ticker, bar_ts_ms);
"""

def fetch_day(ticker: str, day: date) -> list[dict]:
    """Fetch 5-min candles for one ticker/day from Polygon."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute"
        f"/{day.isoformat()}/{day.isoformat()}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    for attempt in range(3):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429:
            time.sleep(12)
            continue
        return resp.json().get("results", [])
    return []

def build_cache(start: date, end: date):
    conn = sqlite3.connect(CACHE_DB)
    conn.executescript(SCHEMA)

    current = start
    while current <= end:
        if current.weekday() >= 5:  # skip weekends
            current += timedelta(days=1)
            continue
        for ticker in TICKERS:
            bars = fetch_day(ticker, current)
            if not bars:
                continue
            rows = [
                (ticker, b["t"], b.get("t_str", ""), b["o"], b["h"],
                 b["l"], b["c"], b.get("v", 0), b.get("vw", 0),
                 b.get("n", 0))
                for b in bars
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO candles_5m VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            time.sleep(0.15)  # rate limit
        conn.commit()
        print(f"  cached {current}")
        current += timedelta(days=1)
    conn.close()
```

**Estimated size:** ~20 tickers x 40 trading days x 78 bars/day = ~62,400 rows. Trivial.

**Rate limits:** Polygon allows 5 req/min on free tier, unlimited on paid. With 0.15s sleep between requests, a full backfill takes ~20 tickers x 40 days x 0.15s = ~2 minutes on paid tier.

### Step 1b: Compute Technical Indicators

For each signal timestamp, compute all indicators from the candle cache. These must match what `owlet-sourcing` will compute in production.

**Module: `options_owl/sourcing/indicators.py`**

Indicator computation functions (all operate on lists of `(timestamp, open, high, low, close, volume, vwap)` tuples):

| Indicator | Parameters | Computation |
|---|---|---|
| EMA(9) | 9-period on closes | Standard EMA with smoothing = 2/(9+1) |
| EMA(21) | 21-period on closes | Standard EMA with smoothing = 2/(21+1) |
| EMA(200) | 200-period on closes | Need 200+ bars (use daily or extended 5m) |
| Bollinger Bands | (20, 2sigma) on closes | SMA(20) +/- 2 * StdDev(20) |
| RSI(9) | 9-period | Wilder's smoothing on gains/losses |
| MACD | (5, 13, 1) | EMA(5) - EMA(13), signal = EMA(1) of MACD |
| VWAP | session reset | cumsum(typical_price * volume) / cumsum(volume) |
| ATR(14) | 14-period | Wilder's smoothing on true range |
| Keltner Channels | (20, 1.5 * ATR) | EMA(20) +/- 1.5 * ATR(14) |
| OBV | cumulative | cumsum(volume * sign(close - prev_close)) |

**Critical: Twelve Data vs local computation verification.**

N8N used Twelve Data's pre-computed indicators. Our local computation from Polygon candles may differ due to:
- Different OHLCV source data (Twelve Data vs Polygon)
- Different smoothing algorithms (Twelve Data may use SMA seed for EMA)
- Different session boundaries for VWAP reset

**Verification approach:**
1. For 10 random signal timestamps, pull Twelve Data indicator values via API
2. Compute the same indicators locally from Polygon candles
3. Compare: acceptable if within 2% for oscillators (RSI, MACD), 0.5% for moving averages (EMA, BB)
4. If discrepancy > threshold, investigate and document the delta

### Step 1c: Reconstruct Market State Object

```python
@dataclass
class MarketState:
    """Complete market state at a point in time, sufficient for scoring."""

    ticker: str
    timestamp: datetime          # UTC
    timestamp_et: datetime       # Eastern Time

    # Underlying price data
    current_price: float
    session_open: float          # day's opening price
    session_high: float
    session_low: float
    session_volume: float

    # Technical indicators (from 5m candles)
    ema_9: float
    ema_21: float
    ema_200: float | None        # may not have enough bars
    bb_upper: float
    bb_lower: float
    bb_mid: float
    bb_pct_b: float              # (price - lower) / (upper - lower)
    rsi_9: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    vwap: float
    atr_14: float
    keltner_upper: float
    keltner_lower: float
    obv: float
    obv_slope: float             # OBV trend over last 5 bars

    # EMA cross state
    ema_cross: str               # "bullish" | "bearish" | "neutral"
    bars_since_cross: int

    # Candle patterns (last 3 bars)
    last_3_candles: list[dict]   # [{open, high, low, close, volume}, ...]

    # Relative strength (vs SPY)
    relative_strength: float     # ticker return - SPY return over session

    # Options chain data (from harvest_snapshots)
    atm_call_premium: float | None
    atm_put_premium: float | None
    atm_iv: float | None
    atm_delta: float | None
    put_call_ratio: float | None  # put OI / call OI near ATM
    total_call_volume: int | None
    total_put_volume: int | None
    max_oi_strike: float | None   # strike with highest open interest

    # SPY context (market regime)
    spy_price: float
    spy_rsi: float
    spy_vwap_position: str       # "above" | "below"
    spy_trend: str               # "up" | "down" | "flat" (based on EMA cross)

    # Data source availability flags
    has_options_chain: bool
    has_unusual_flow: bool       # always False for backtest
    has_news_sentiment: bool     # True if Polygon news available
    has_grok_analysis: bool      # always False for backtest
```

**Reconstruction function:**

```python
async def reconstruct_market_state(
    ticker: str,
    timestamp_utc: datetime,
    candle_db: str,
    harvester_db: str,
) -> MarketState:
    """Build a complete MarketState from historical data.

    Args:
        ticker: Stock ticker symbol
        timestamp_utc: The point in time to reconstruct
        candle_db: Path to backtest candle cache (candles_5m table)
        harvester_db: Path to harvester DB (harvest_snapshots + harvest_contracts)

    Steps:
        1. Query candles_5m for ticker, last 78 bars before timestamp
        2. Query candles_5m for SPY, same window
        3. Compute all technical indicators
        4. Query harvest_snapshots for nearest options chain
        5. Assemble MarketState
    """
```

**SQL for candle retrieval:**

```sql
-- Get 78 most recent 5m bars for ticker at or before signal time
SELECT bar_ts_ms, open, high, low, close, volume, vwap
FROM candles_5m
WHERE ticker = ? AND bar_ts_ms <= ?
ORDER BY bar_ts_ms DESC
LIMIT 78;
```

**SQL for options chain at signal time:**

```sql
-- Get ATM options snapshot nearest to signal time
-- First, find the underlying price at signal time
SELECT hs.underlying_price, hs.bid, hs.ask, hs.midpoint,
       hs.implied_volatility, hs.delta, hs.open_interest, hs.day_volume,
       hc.strike, hc.option_type, hc.expiry_date,
       hs.captured_at
FROM harvest_snapshots hs
JOIN harvest_contracts hc ON hs.contract_ticker = hc.contract_ticker
WHERE hc.underlying = ?
  AND hs.captured_at BETWEEN datetime(?, '-5 minutes') AND datetime(?, '+5 minutes')
  AND hc.expiry_date >= date(?)
ORDER BY ABS(julianday(hs.captured_at) - julianday(?))
LIMIT 50;
```

Then filter in Python for ATM strikes (closest to underlying_price) and compute put/call ratio, max OI strike, etc.

---

## Phase 2: Signal Replay (Week 2)

### Step 2a: Re-score Historical N8N Signals

For each of the 330 signals in `trade_signals`:

1. Reconstruct `MarketState` at `created_at` timestamp
2. Run the new scoring engine with all sources enabled
3. Record: `(signal_id, n8n_score, new_score, new_direction, breakdown)`
4. Compare scores

**Script: `scripts/backtest_sourcing.py --mode=replay`**

```python
async def run_signal_replay(
    signals_db: str,
    candle_db: str,
    harvester_db: str,
    enabled_sources: set[str] | None = None,
) -> pd.DataFrame:
    """Re-score all historical N8N signals with the new engine.

    Returns DataFrame with columns:
        signal_id, ticker, n8n_direction, n8n_score, new_direction,
        new_score, score_delta, direction_match, breakdown,
        actual_pnl_pct (from paper_trades if available)
    """
    conn = sqlite3.connect(signals_db)
    signals = conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.created_at,
               ts.entry_price, ts.strike, ts.expiry, ts.atm_premium,
               ts.key_signals,
               pt.pnl_pct, pt.exit_reason, pt.exit_source
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id
            AND pt.status = 'closed'
            AND (pt.exit_source = 'ai' OR pt.exit_source IS NULL)
        ORDER BY ts.created_at
    """).fetchall()

    results = []
    for sig in signals:
        state = await reconstruct_market_state(
            sig["ticker"], parse_utc(sig["created_at"]),
            candle_db, harvester_db,
        )
        result = score_signal(state, enabled_sources=enabled_sources)
        results.append({
            "signal_id": sig["id"],
            "ticker": sig["ticker"],
            "timestamp": sig["created_at"],
            "n8n_direction": sig["direction"],
            "n8n_score": sig["score"],
            "new_direction": result.direction,
            "new_score": result.score,
            "score_delta": result.score - sig["score"],
            "direction_match": result.direction == sig["direction"],
            "breakdown": result.breakdown,
            "actual_pnl_pct": sig["pnl_pct"],
        })

    return pd.DataFrame(results)
```

**Acceptance criteria for Phase 2:**
- Direction agreement: >= 90% of signals should have matching direction
- Score correlation: Pearson r >= 0.6 between N8N score and new score
- No catastrophic misses: signals that N8N scored 120+ should score 70+ with new engine

### Step 2b: Score vs Outcome Calibration

Using the 220 closed trades with known outcomes:

```python
def calibrate_scores(replay_df: pd.DataFrame):
    """Verify that new scores predict outcomes better than N8N scores.

    Bucket trades by score decile and compute win rate per bucket.
    A well-calibrated score system shows monotonically increasing WR.
    """
    # Filter to trades with actual outcomes
    df = replay_df.dropna(subset=["actual_pnl_pct"])

    # Bucket by new score
    df["score_bucket"] = pd.cut(df["new_score"], bins=range(0, 101, 10))
    df["is_win"] = df["actual_pnl_pct"] > 0

    calibration = df.groupby("score_bucket").agg(
        count=("is_win", "count"),
        win_rate=("is_win", "mean"),
        avg_pnl=("actual_pnl_pct", "mean"),
        median_pnl=("actual_pnl_pct", "median"),
    ).reset_index()

    # Check monotonicity
    win_rates = calibration["win_rate"].tolist()
    is_monotonic = all(
        win_rates[i] <= win_rates[i + 1]
        for i in range(len(win_rates) - 1)
        if not pd.isna(win_rates[i]) and not pd.isna(win_rates[i + 1])
    )

    return calibration, is_monotonic
```

---

## Phase 3: Full Interval Scanner Backtest (Week 3)

### Step 3a: Reverse-Engineer What Sourcing Would Have Fired

Instead of only re-scoring N8N's signals, scan every 3-minute interval to find signals the new engine would generate independently.

**Script: `scripts/backtest_sourcing.py --mode=scan`**

```python
async def run_full_scan(
    start_date: date,
    end_date: date,
    candle_db: str,
    harvester_db: str,
    tickers: list[str],
    interval_minutes: int = 3,
    score_threshold: int = 60,
    enabled_sources: set[str] | None = None,
) -> pd.DataFrame:
    """Scan every N-minute interval across all tickers.

    For each interval:
      1. Reconstruct MarketState for all tickers
      2. Score each ticker
      3. If score >= threshold, record as potential signal
      4. Apply cooldown (no repeat signal within 30 min for same ticker)

    Returns DataFrame of all generated signals with columns:
        timestamp, ticker, direction, score, breakdown,
        matched_n8n_signal_id (if N8N also fired within +/- 5 min),
        actual_outcome (if we have trade data)
    """
    # Market hours: 9:33 AM to 3:57 PM ET (first signal at 9:33, last at 3:57)
    # = 130 intervals per day at 3-min spacing
    # x 40 trading days x 13 tickers = ~67,600 market state reconstructions
    #
    # Each reconstruction: 2 SQL queries (candles + options chain)
    # Estimated runtime: ~45 minutes (cached candles, indexed harvester DB)

    signals = []
    cooldowns: dict[str, datetime] = {}  # ticker -> last signal time

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Generate timestamps: 9:33 AM ET through 3:57 PM ET, every 3 min
        market_open_et = datetime(
            current_date.year, current_date.month, current_date.day,
            9, 33, tzinfo=ET,
        )
        market_close_et = datetime(
            current_date.year, current_date.month, current_date.day,
            15, 57, tzinfo=ET,
        )

        t = market_open_et
        while t <= market_close_et:
            t_utc = t.astimezone(timezone.utc)

            for ticker in tickers:
                # Cooldown check
                if ticker in cooldowns:
                    if (t_utc - cooldowns[ticker]).total_seconds() < 1800:
                        continue

                state = await reconstruct_market_state(
                    ticker, t_utc, candle_db, harvester_db,
                )
                result = score_signal(state, enabled_sources=enabled_sources)

                if result.score >= score_threshold:
                    signals.append({
                        "timestamp": t_utc.isoformat(),
                        "ticker": ticker,
                        "direction": result.direction,
                        "score": result.score,
                        "breakdown": result.breakdown,
                    })
                    cooldowns[ticker] = t_utc

            t += timedelta(minutes=interval_minutes)

        current_date += timedelta(days=1)

    df = pd.DataFrame(signals)

    # Cross-reference with N8N signals
    df = _match_to_n8n_signals(df, signals_db)

    # Cross-reference with actual outcomes
    df = _match_to_outcomes(df, harvester_db)

    return df
```

### Step 3b: Outcome Evaluation from Harvester Data

For signals where we don't have a matching `paper_trades` entry (because N8N didn't fire them), we simulate the trade outcome using harvester tick data:

```python
async def evaluate_hypothetical_outcome(
    ticker: str,
    direction: str,
    entry_time_utc: datetime,
    entry_premium: float,
    harvester_db: str,
    hold_minutes: int = 120,  # max hold time for evaluation
) -> dict:
    """Simulate what would have happened if we took this trade.

    Uses harvest_snapshots to track ATM premium after entry.
    Applies simplified V5 exit logic (graduated stop at -35%, profit target
    at +30% for index, adaptive trail at +40%).

    Returns:
        {
            "max_gain_pct": float,    # peak premium gain
            "max_loss_pct": float,    # worst premium drawdown
            "exit_prem": float,       # premium at exit
            "exit_reason": str,       # what triggered exit
            "pnl_pct": float,         # (exit - entry) / entry * 100
            "hold_minutes": float,    # how long held
        }
    """
```

This reuses the same `load_ticks` + `simulate_with_production_fsm` pattern from `scripts/backtest_v5_production.py`, which already runs the live ExitFSM against harvester tick data.

---

## Phase 4: Backtest Scenarios (Week 4)

### Scenario 1: Reproduction Test

**Question:** Does the new engine score N8N signals similarly?

| Metric | Target |
|---|---|
| Direction agreement | >= 90% |
| Score correlation (Pearson r) | >= 0.6 |
| Signals scored 120+ by N8N that new engine scores < 70 | 0 |

**Implementation:** `scripts/backtest_sourcing.py --mode=replay`

### Scenario 2: Source Ablation Study

**Question:** What is each data source's marginal contribution?

Run the signal replay (Phase 2) multiple times, each with one source disabled:

```python
ABLATION_CONFIGS = {
    "all_sources":     {"ema", "bb", "rsi", "macd", "vwap", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_ema":          {"bb", "rsi", "macd", "vwap", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_bb":           {"ema", "rsi", "macd", "vwap", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_rsi":          {"ema", "bb", "macd", "vwap", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_macd":         {"ema", "bb", "rsi", "vwap", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_vwap":         {"ema", "bb", "rsi", "macd", "atr", "volume", "options_chain", "relative_strength", "candle_patterns"},
    "no_options":      {"ema", "bb", "rsi", "macd", "vwap", "atr", "volume", "relative_strength", "candle_patterns"},
    "no_volume":       {"ema", "bb", "rsi", "macd", "vwap", "atr", "options_chain", "relative_strength", "candle_patterns"},
    "no_rel_strength": {"ema", "bb", "rsi", "macd", "vwap", "atr", "volume", "options_chain", "candle_patterns"},
    "technicals_only": {"ema", "bb", "rsi", "macd", "vwap", "atr"},
    "minimal":         {"ema", "rsi", "vwap"},
}
```

**Output table:**

| Config | Signals | Win Rate | Avg P&L | Sharpe | Delta vs All |
|---|---|---|---|---|---|
| all_sources | ... | ... | ... | ... | baseline |
| no_ema | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

A source is "harmful" if removing it improves win rate. A source is "useless" if removing it changes nothing. Both should be removed from the scoring engine.

### Scenario 3: Score Threshold Sweep

**Question:** What is the optimal score cutoff?

```python
THRESHOLDS = [50, 55, 60, 65, 70, 75, 78, 80, 85, 90]

for threshold in THRESHOLDS:
    # Filter signals to score >= threshold
    # Compute: signal_count, win_rate, total_pnl, sharpe
    # Plot: precision-recall tradeoff
```

**Expected tradeoff:** Higher threshold = fewer signals + higher win rate. We want the "elbow" where win rate improvement plateaus.

Current N8N: fires ~3-5 signals/day at ~60% WR.
Target: 2-4 signals/day at 65-70% WR.

### Scenario 4: Missed Opportunity Analysis

**Question:** Are there profitable trades the new engine catches that N8N missed?

From the full scan (Phase 3), identify signals where:
- New engine scored >= 70
- N8N did NOT fire a signal within +/- 10 minutes
- Hypothetical outcome is profitable (simulated via harvester ticks)

This reveals untapped alpha. If the new engine finds consistent winners that N8N missed, it justifies the migration.

### Scenario 5: False Positive Reduction

**Question:** Can we reduce losing trades without cutting winners?

For each losing trade in the historical dataset (120+ trades with negative P&L):
1. Examine the MarketState at entry
2. Identify patterns common to losers but rare in winners
3. Propose additional gates (e.g., "RSI > 70 for calls = reject")
4. Backtest the proposed gate

Common false positive patterns to check:
- Entering calls when RSI > 70 (overbought)
- Entering puts when RSI < 30 (oversold)
- Entering against VWAP (calls below VWAP, puts above)
- Entering against EMA trend (9 EMA below 21 EMA for calls)
- Entering with low volume (session volume < 50% of average)
- Entering near session high/low (buying calls at session high)

### Scenario 6: Per-Ticker Performance

**Question:** Should different tickers use different scoring weights?

```python
TRACKED_TICKERS = [
    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "META", "AAPL",
    "AMZN", "GOOGL", "MSFT", "AMD", "MSTR", "PLTR",
]

for ticker in TRACKED_TICKERS:
    # Run signal replay filtered to this ticker
    # Compare win rate, avg P&L, signal frequency
    # Cross-reference with V5 FSM per-ticker configs (exit_v5/config.py)
```

If a ticker consistently underperforms (< 50% WR), it may need:
- Higher score threshold
- Different indicator weights
- Blocking entirely (add to BLOCKED_TICKERS)

---

## Phase 5: Implementation Details

### Script Structure: `scripts/backtest_sourcing.py`

```python
"""Backtest the owlet-sourcing signal engine against historical data.

Usage:
    # Re-score all historical N8N signals
    python scripts/backtest_sourcing.py --mode=replay

    # Full 3-min interval scan
    python scripts/backtest_sourcing.py --mode=scan --start=2026-04-10 --end=2026-05-18

    # Source ablation study
    python scripts/backtest_sourcing.py --mode=ablation

    # Score threshold sweep
    python scripts/backtest_sourcing.py --mode=threshold-sweep

    # Per-ticker breakdown
    python scripts/backtest_sourcing.py --mode=per-ticker

    # Export results to CSV
    python scripts/backtest_sourcing.py --mode=replay --output=results/replay.csv
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
CANDLE_CACHE_DB = str(PROJECT_DIR / "journal" / "backtest_candles.db")


@dataclass
class BacktestConfig:
    mode: str = "replay"
    start_date: date | None = None
    end_date: date | None = None
    score_threshold: int = 60
    enabled_sources: set[str] = field(default_factory=lambda: {
        "ema", "bb", "rsi", "macd", "vwap", "atr", "volume",
        "options_chain", "relative_strength", "candle_patterns",
    })
    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "NVDA", "TSLA", "META", "AAPL",
        "AMZN", "GOOGL", "MSFT", "AMD", "MSTR", "PLTR",
    ])
    output_path: str | None = None
    scan_interval_minutes: int = 3
    cooldown_minutes: int = 30


@dataclass
class ScoringResult:
    score: int                   # 0-100
    direction: str               # "bullish" | "bearish"
    breakdown: dict[str, float]  # source_name -> contribution


class SourcingBacktest:
    """Backtest the owlet-sourcing signal engine against historical data."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._candle_conn: sqlite3.Connection | None = None
        self._harvester_conn: sqlite3.Connection | None = None

    async def reconstruct_market_state(
        self, ticker: str, timestamp_utc: datetime
    ) -> MarketState:
        """Pull candles from cache and compute all indicators."""
        ...

    def score(
        self, state: MarketState, enabled_sources: set[str] | None = None
    ) -> ScoringResult:
        """Run scoring engine with configured sources."""
        ...

    async def run_replay(self) -> pd.DataFrame:
        """Re-score historical N8N signals."""
        ...

    async def run_scan(self) -> pd.DataFrame:
        """Scan every 3-min interval for signals."""
        ...

    async def run_ablation(self) -> pd.DataFrame:
        """Run source ablation study."""
        ...

    async def run_threshold_sweep(self) -> pd.DataFrame:
        """Sweep score thresholds."""
        ...

    async def run_per_ticker(self) -> pd.DataFrame:
        """Per-ticker performance breakdown."""
        ...
```

### Integration with Existing Backtest Infrastructure

The sourcing backtest plugs into the existing exit-strategy backtest:

1. **Sourcing backtest** answers: "Should we enter this trade?" (signal quality)
2. **V5 production backtest** (`backtest_v5_production.py`) answers: "How should we exit?" (exit strategy)

To run an end-to-end backtest:

```python
# Step 1: Generate signals with new sourcing engine
sourcing_signals = await sourcing_backtest.run_scan()

# Step 2: For each signal, simulate exit with production V5 FSM
# (reuse simulate_with_production_fsm from backtest_v5_production.py)
for signal in sourcing_signals.itertuples():
    ticks = load_ticks(harvester_conn, signal_dict)
    if ticks is None:
        continue
    result = simulate_with_production_fsm(
        ticks, entry_premium, contracts, direction, dte, expiry_date, ticker
    )
    # Record combined entry+exit outcome
```

This ensures we test the full pipeline: sourcing -> entry gates -> exit FSM.

---

## Metrics and Reporting

### Per-Run Summary

For each backtest run, compute and display:

| Metric | Formula |
|---|---|
| Total signals generated | count(score >= threshold) |
| Signals per day | total / trading_days |
| Win rate | wins / total * 100 |
| Average winner P&L | mean(pnl where pnl > 0) |
| Average loser P&L | mean(pnl where pnl < 0) |
| Profit factor | gross_wins / gross_losses |
| Sharpe ratio | mean(returns) / std(returns) * sqrt(252) |
| Sortino ratio | mean(returns) / downside_std * sqrt(252) |
| Max drawdown | peak_to_trough / peak * 100 |
| Score-outcome correlation | pearson_r(score, pnl_pct) |

### Visualizations

Generate as PNG files in `results/sourcing_backtest/`:

1. **Score distribution histogram** -- new scores vs N8N scores, side by side
2. **Score vs outcome scatter** -- x=score, y=pnl_pct, color=win/loss
3. **Win rate by score bucket** -- bar chart, 10-point buckets, with error bars
4. **Confusion matrix** -- signal fired Y/N x trade profitable Y/N
5. **Source contribution heatmap** -- rows=sources, columns=tickers, values=win_rate_delta when source removed
6. **Equity curve** -- cumulative P&L over time for new engine vs N8N baseline
7. **Per-ticker win rate** -- horizontal bar chart
8. **Time-of-day performance** -- win rate by hour (ET), highlight 9:30-10:30 and 2:00-3:30 zones
9. **Daily signal count** -- line chart of signals/day over backtest window

### Report Format

```
================================================================
  SOURCING BACKTEST REPORT — {mode} mode
  {start_date} to {end_date}
  Sources: {enabled_sources}
================================================================

  SIGNAL GENERATION
    Total signals:        {n}
    Per trading day:      {n/days:.1f}
    Direction split:      {bullish_pct:.0f}% bull / {bearish_pct:.0f}% bear

  PERFORMANCE (matched to actual outcomes)
    Trades evaluated:     {n_with_outcome}
    Win rate:             {wr:.1f}%  (target: >= 60%)
    Avg winner:           +{avg_win:.1f}%
    Avg loser:            {avg_loss:.1f}%
    Profit factor:        {pf:.2f}
    Sharpe:               {sharpe:.2f}

  SCORE CALIBRATION
    Score-outcome corr:   r={corr:.3f}  (target: > 0 and monotonic)
    [50-60) WR:           {wr_50_60:.0f}%  (n={n_50_60})
    [60-70) WR:           {wr_60_70:.0f}%  (n={n_60_70})
    [70-80) WR:           {wr_70_80:.0f}%  (n={n_70_80})
    [80-90) WR:           {wr_80_90:.0f}%  (n={n_80_90})
    [90-100] WR:          {wr_90_100:.0f}% (n={n_90_100})

  vs N8N BASELINE
    Direction agreement:  {dir_agree:.0f}%  (target: >= 90%)
    Score correlation:    r={score_corr:.3f}  (target: >= 0.6)
    N8N signals missed:   {n_missed}  (scored < threshold by new engine)

  PER-TICKER
    {ticker_table}

================================================================
```

---

## Validation Criteria

The new sourcing bot MUST demonstrate all of these before replacing N8N:

| # | Criterion | Threshold | Measured By |
|---|---|---|---|
| 1 | Win rate | >= 60% (N8N baseline) | Signal replay + outcome matching |
| 2 | Total P&L | >= N8N's over same period | End-to-end backtest with V5 FSM |
| 3 | Score-outcome monotonicity | Higher score -> higher WR | Score calibration analysis |
| 4 | Fewer false positives | Signal count <= N8N's at equal/better WR | Threshold sweep |
| 5 | Per-ticker consistency | No ticker < 50% WR | Per-ticker analysis |
| 6 | Per-week consistency | No week < 45% WR | Weekly bucketing |
| 7 | Direction agreement with N8N | >= 90% | Signal replay |
| 8 | No catastrophic misses | 0 signals scored 120+ by N8N and < 70 by new | Signal replay |

If any criterion fails, iterate on scoring weights before deploying.

---

## Data Gaps and Mitigations

### 3-Minute vs 5-Minute Resolution

N8N ran every 3 minutes; harvester captures every 5 minutes (for options) and every 1 minute (for candles, when available). Some signal timestamps will fall between harvester snapshots.

**Mitigation:** For options chain data, use the nearest snapshot within +/- 5 minutes. For candles, the 5-minute bar containing the signal timestamp is adequate since indicators (EMA, RSI) don't change meaningfully within a single bar.

### Twelve Data vs Polygon Candle Differences

N8N used Twelve Data for candles and pre-computed indicators. We use Polygon.

**Mitigation:**
1. Run the verification step in Phase 1 (Step 1b) for 10 random timestamps
2. If differences > 2%, adjust indicator parameters to match Twelve Data output
3. Document any systematic bias (e.g., "Polygon VWAP runs 0.1% higher than Twelve Data")

### Missing Sources for Full Scan

Unusual Whales flow and Grok AI analysis cannot be backfilled. This means:
- Full scan results will underestimate the new engine's capability (fewer sources = fewer high-conviction signals)
- OR full scan results will overestimate if those sources add noise

**Mitigation:** Run two full scan variants:
1. With all available sources (baseline)
2. After 2 weeks of production logging, re-run with Unusual Whales + Grok data from those 2 weeks to measure marginal value

### Harvester Snapshots Before Apr 10

Harvester data starts Mar 27, but trade signals start Apr 10. Options chain reconstruction is available for the full trade signal window.

---

## Timeline

| Week | Deliverable | Exit Criteria |
|---|---|---|
| 1 | Data reconstruction pipeline | `build_candle_cache.py` populates 62K+ rows; indicator computation verified within 2% of Twelve Data for 10 test points; `MarketState` reconstruction works for all 330 historical signals |
| 2 | Signal replay engine | All 330 signals re-scored; direction agreement >= 90%; score correlation report generated; score calibration chart shows general upward trend |
| 3 | Full interval scanner | 3-min scan over full date range completes in < 2 hours; missed opportunity analysis identifies >= 5 potential additional winners; false positive patterns documented |
| 4 | All scenarios run + reports | Source ablation table complete; threshold sweep identifies optimal cutoff; per-ticker report generated; all 8 validation criteria evaluated |
| 5 | Iterate and finalize | Any failing criteria addressed; final scoring weights locked; comparison report ready for review; green light to deploy or documented blockers |

---

## Appendix: Key SQL Queries

### A. All N8N signals with outcomes

```sql
SELECT
    ts.id, ts.ticker, ts.direction, ts.score, ts.created_at,
    ts.entry_price, ts.strike, ts.expiry, ts.atm_premium,
    ts.key_signals, ts.bot_source,
    pt.id AS trade_id, pt.premium_per_contract, pt.contracts,
    pt.exit_premium, pt.exit_reason, pt.pnl_dollars, pt.pnl_pct,
    pt.opened_at, pt.closed_at, pt.webull_order_id, pt.exit_source
FROM trade_signals ts
LEFT JOIN paper_trades pt ON pt.signal_id = ts.id
    AND pt.status = 'closed'
ORDER BY ts.created_at;
```

### B. Win rate by N8N score bucket

```sql
SELECT
    CASE
        WHEN ts.score >= 135 THEN '135+'
        WHEN ts.score >= 120 THEN '120-134'
        WHEN ts.score >= 100 THEN '100-119'
        WHEN ts.score >= 90  THEN '90-99'
        WHEN ts.score >= 78  THEN '78-89'
        ELSE '<78'
    END AS score_bucket,
    COUNT(*) AS trades,
    SUM(CASE WHEN pt.pnl_dollars > 0 THEN 1 ELSE 0 END) AS wins,
    ROUND(100.0 * SUM(CASE WHEN pt.pnl_dollars > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate,
    ROUND(AVG(pt.pnl_pct), 1) AS avg_pnl_pct
FROM trade_signals ts
JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.status = 'closed'
WHERE pt.exit_source = 'ai' OR pt.exit_source IS NULL
GROUP BY score_bucket
ORDER BY MIN(ts.score);
```

### C. Options chain at signal time (harvester)

```sql
-- Get all option snapshots for a ticker near a specific time
SELECT
    hc.underlying, hc.strike, hc.option_type, hc.expiry_date,
    hs.captured_at, hs.underlying_price,
    hs.bid, hs.ask, hs.midpoint,
    hs.implied_volatility, hs.delta, hs.gamma, hs.theta, hs.vega,
    hs.open_interest, hs.day_volume
FROM harvest_snapshots hs
JOIN harvest_contracts hc ON hs.contract_ticker = hc.contract_ticker
WHERE hc.underlying = :ticker
  AND hs.captured_at BETWEEN datetime(:signal_time, '-3 minutes')
                         AND datetime(:signal_time, '+3 minutes')
  AND hc.expiry_date >= date(:signal_time)
  AND hc.expiry_date <= date(:signal_time, '+7 days')
ORDER BY hc.option_type, hc.strike;
```

### D. 5-min candles for indicator computation

```sql
-- Get last 78 bars (full session) for a ticker from cache
SELECT bar_ts_ms, open, high, low, close, volume, vwap
FROM candles_5m
WHERE ticker = :ticker AND bar_ts_ms <= :signal_ts_ms
ORDER BY bar_ts_ms DESC
LIMIT 78;
```

### E. Signals that N8N fired but would fail new engine

```sql
-- After populating backtest_results table:
SELECT
    br.signal_id, br.ticker, br.n8n_score, br.new_score,
    br.n8n_direction, br.new_direction,
    br.actual_pnl_pct
FROM backtest_results br
WHERE br.n8n_score >= 78
  AND br.new_score < 60
ORDER BY br.n8n_score DESC;
```

### F. Trade events for debugging signal lifecycle

```sql
SELECT te.trade_id, te.ticker, te.event_type, te.detail, te.created_at
FROM trade_events te
WHERE te.ticker = :ticker
  AND date(te.created_at) = :date
ORDER BY te.id;
```
