"""Multi-timeframe candle data from Polygon REST API with caching.

Fetches OHLCV bars at 5m/15m/30m/1h/4h timeframes for underlying tickers.
Candles are cached in memory with TTL matching the timeframe so we don't
hammer Polygon on every 5-second poll cycle.

Technical indicators (ATR, RSI, OBV) are computed from the cached bars.

Usage in position_monitor::

    from options_owl.collectors.candle_cache import CandleCache
    cache = CandleCache(api_key="...")
    data = await cache.get_candle_data("SPY")
    # data = {"5m": [...], "15m": [...], ..., "indicators": {"5m": {...}, ...}}
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Candle bar dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CandleBar:
    """Single OHLCV bar."""
    timestamp: float  # unix epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0


# ---------------------------------------------------------------------------
# Timeframe config
# ---------------------------------------------------------------------------

TIMEFRAMES: dict[str, tuple[int, str, int]] = {
    # label -> (multiplier, span, cache_ttl_seconds)
    "5m":  (5,  "minute", 300),    # refresh every 5 min
    "15m": (15, "minute", 900),    # refresh every 15 min
    "30m": (30, "minute", 1800),   # refresh every 30 min
    "1h":  (1,  "hour",   3600),   # refresh every hour
    "4h":  (4,  "hour",   14400),  # refresh every 4 hours
}

# How many bars to request per timeframe (enough for ATR-14 + RSI-14 + buffer)
BARS_REQUESTED = 50


# ---------------------------------------------------------------------------
# CandleCache
# ---------------------------------------------------------------------------

class CandleCache:
    """Async candle fetcher with per-timeframe TTL caching.

    Data source priority:
    1. Shared harvester DB (``shared_db_path``) — zero API calls
    2. MarketDataStream WS minute bars — zero REST calls
    3. Polygon REST — direct API call (fallback)
    """

    def __init__(
        self,
        api_key: str,
        market_stream: object | None = None,
        shared_db_path: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._market_stream = market_stream  # MarketDataStream (optional)
        self._shared_db_path = shared_db_path  # Harvester DB (optional)
        # cache: (ticker, timeframe) -> (bars, fetched_at_unix)
        self._cache: dict[tuple[str, str], tuple[list[CandleBar], float]] = {}

    async def get_candles(
        self,
        ticker: str,
        timeframe: str = "5m",
    ) -> list[CandleBar]:
        """Get cached candles, fetching from WS buffer or Polygon REST."""
        ticker = ticker.upper()
        key = (ticker, timeframe)

        if timeframe not in TIMEFRAMES:
            return []

        mult, span, ttl = TIMEFRAMES[timeframe]

        # Check TTL cache first
        cached = self._cache.get(key)
        if cached is not None:
            bars, fetched_at = cached
            if _time.time() - fetched_at < ttl:
                return bars

        # 1. Try shared harvester DB (zero API calls, works for all agents)
        if self._shared_db_path:
            db_bars = await self._read_from_shared_db(ticker, timeframe)
            if db_bars:
                logger.debug(
                    f"Shared DB candles: {ticker} {timeframe} → {len(db_bars)} bars"
                )
                self._cache[key] = (db_bars, _time.time())
                return db_bars

        # 2. Try building from WS minute bars (free, real-time)
        if self._market_stream is not None:
            ws_bars = self._build_from_ws(ticker, timeframe)
            if ws_bars:
                self._cache[key] = (ws_bars, _time.time())
                return ws_bars

        # 3. Fall back to Polygon REST
        bars = await self._fetch_from_polygon(ticker, timeframe)
        self._cache[key] = (bars, _time.time())
        return bars

    async def _read_from_shared_db(
        self, ticker: str, timeframe: str
    ) -> list[CandleBar]:
        """Read candle bars from the shared harvester DB.

        The harvester only writes 5m bars. For higher timeframes (15m, 30m,
        1h, 4h) we read extra 5m bars and aggregate them locally — this
        avoids Polygon REST calls entirely for candle data.
        """
        from pathlib import Path

        db_path = Path(self._shared_db_path)  # type: ignore[arg-type]
        if not db_path.exists():
            return []

        try:
            from options_owl.collectors.candle_collector import read_candles_from_db

            if timeframe == "5m":
                rows = await read_candles_from_db(db_path, ticker, "5m", limit=50)
            else:
                # Read enough 5m bars to aggregate into the requested timeframe.
                # We need ~20 aggregated bars (ATR-14 + RSI-14 + buffer).
                # e.g. 15m: 20×3=60, 1h: 20×12=240, 4h: 20×48=960
                mult, span, _ = TIMEFRAMES[timeframe]
                tf_minutes = mult * 60 if span == "hour" else mult
                bars_per_candle = tf_minutes // 5
                rows = await read_candles_from_db(
                    db_path, ticker, "5m", limit=20 * bars_per_candle
                )

            if not rows:
                return []

            bars_5m = [
                CandleBar(
                    timestamp=r["bar_start_ts"],
                    open=r["open"],
                    high=r["high"],
                    low=r["low"],
                    close=r["close"],
                    volume=r.get("volume", 0) or 0,
                    vwap=r.get("vwap", 0) or 0,
                )
                for r in rows
            ]

            if timeframe == "5m":
                return bars_5m

            # Aggregate 5m bars into the requested timeframe
            return self._aggregate_bars(bars_5m, timeframe)

        except Exception as e:
            logger.debug(f"Shared DB read failed for {ticker} {timeframe}: {e}")
            return []

    @staticmethod
    def _aggregate_bars(
        bars_5m: list[CandleBar], timeframe: str
    ) -> list[CandleBar]:
        """Aggregate 5-minute bars into a higher timeframe."""
        mult, span, _ = TIMEFRAMES[timeframe]
        tf_minutes = mult * 60 if span == "hour" else mult
        bucket_ms = tf_minutes * 60 * 1000

        # Group 5m bars into buckets
        buckets: dict[int, list[CandleBar]] = {}
        for bar in bars_5m:
            bucket = int((bar.timestamp // bucket_ms) * bucket_ms)
            buckets.setdefault(bucket, []).append(bar)

        aggregated: list[CandleBar] = []
        for bucket_ts in sorted(buckets):
            group = buckets[bucket_ts]
            aggregated.append(CandleBar(
                timestamp=bucket_ts,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
                vwap=group[-1].vwap if group[-1].vwap else 0.0,
            ))
        return aggregated

    def _build_from_ws(self, ticker: str, timeframe: str) -> list[CandleBar]:
        """Aggregate WS 1-minute bars into the requested timeframe."""
        minute_bars = self._market_stream.get_minute_bars(ticker)  # type: ignore[union-attr]
        if not minute_bars:
            return []

        mult, span, _ = TIMEFRAMES[timeframe]

        # Convert timeframe to minutes
        if span == "hour":
            tf_minutes = mult * 60
        else:
            tf_minutes = mult

        # Aggregate minute bars into buckets
        aggregated: list[CandleBar] = []
        bucket_start: float | None = None
        o = h = lo = c = vol = vw = 0.0

        for ts_ms, bar_o, bar_h, bar_l, bar_c, bar_v, bar_vw in minute_bars:
            # Bucket by flooring timestamp to timeframe boundary
            bucket = (ts_ms // (tf_minutes * 60 * 1000)) * (tf_minutes * 60 * 1000)

            if bucket_start is not None and bucket != bucket_start:
                # Close previous bucket
                aggregated.append(CandleBar(
                    timestamp=bucket_start, open=o, high=h, low=lo, close=c,
                    volume=vol, vwap=vw,
                ))
                bucket_start = None

            if bucket_start is None:
                bucket_start = bucket
                o = bar_o
                h = bar_h
                lo = bar_l
                vol = 0.0
                vw = 0.0

            h = max(h, bar_h)
            lo = min(lo, bar_l)
            c = bar_c
            vol += bar_v
            if bar_vw > 0:
                vw = bar_vw  # use last vwap

        # Close final bucket
        if bucket_start is not None:
            aggregated.append(CandleBar(
                timestamp=bucket_start, open=o, high=h, low=lo, close=c,
                volume=vol, vwap=vw,
            ))

        return aggregated

    async def get_candle_data(self, ticker: str) -> dict:
        """Fetch all timeframes + compute indicators for a ticker.

        Returns::

            {
                "5m":  [CandleBar, ...],
                "15m": [CandleBar, ...],
                "1h":  [CandleBar, ...],
                "indicators": {
                    "5m":  {"atr": 1.23, "rsi": 45.2, "obv": 1234567, "pattern": None},
                    "15m": {...},
                    "1h":  {...},
                },
            }
        """
        result: dict = {}
        indicators: dict = {}

        for tf in TIMEFRAMES:
            bars = await self.get_candles(ticker, tf)
            result[tf] = bars
            if bars:
                indicators[tf] = {
                    "atr": calc_atr(bars),
                    "rsi": calc_rsi(bars),
                    "obv": calc_obv(bars),
                    "pattern": detect_candle_pattern(bars),
                    "volume_trend": calc_volume_trend(bars),
                }
            else:
                indicators[tf] = {
                    "atr": None, "rsi": None, "obv": None,
                    "pattern": None, "volume_trend": None,
                }

        result["indicators"] = indicators
        return result

    def invalidate(self, ticker: str) -> None:
        """Clear all cached candles for a ticker."""
        ticker = ticker.upper()
        keys_to_remove = [k for k in self._cache if k[0] == ticker]
        for k in keys_to_remove:
            del self._cache[k]

    async def _fetch_from_polygon(
        self,
        ticker: str,
        timeframe: str,
    ) -> list[CandleBar]:
        """Fetch bars from Polygon /v2/aggs endpoint."""
        if not self._api_key:
            return []

        mult, span, _ = TIMEFRAMES[timeframe]

        # Request enough history for indicators (RSI-14 needs 15+ bars).
        # For higher TFs (1h/4h) a single trading day doesn't have enough bars,
        # so we look back further.  For 5m we only need today.
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except ImportError:
            _et = timezone(timedelta(hours=-5))
        now = datetime.now(_et)
        to_str = now.strftime("%Y-%m-%d")

        if span == "hour" and mult >= 4:
            # 4h bars: need ~15 bars × 4h = 60h ≈ 10 trading days
            from_dt = now - timedelta(days=14)
        elif span == "hour":
            # 1h bars: need ~15 bars × 1h = 15h ≈ 3 trading days
            from_dt = now - timedelta(days=5)
        elif mult >= 30:
            # 30m bars: need ~15 bars × 30m = 7.5h ≈ 2 trading days
            from_dt = now - timedelta(days=3)
        else:
            # 5m/15m: today is enough
            from_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now < from_dt:
                from_dt -= timedelta(days=1)

        from_str = from_dt.strftime("%Y-%m-%d")

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range"
            f"/{mult}/{span}/{from_str}/{to_str}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": str(BARS_REQUESTED),
            "apiKey": self._api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.debug(
                        f"Polygon candles {resp.status_code} for {ticker} {timeframe}"
                    )
                    return []

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return []

                bars = []
                for bar in results:
                    bars.append(CandleBar(
                        timestamp=bar.get("t", 0),
                        open=float(bar.get("o", 0)),
                        high=float(bar.get("h", 0)),
                        low=float(bar.get("l", 0)),
                        close=float(bar.get("c", 0)),
                        volume=float(bar.get("v", 0)),
                        vwap=float(bar.get("vw", 0)),
                    ))

                logger.debug(
                    f"Polygon candles: {ticker} {timeframe} → {len(bars)} bars"
                )
                return bars

        except Exception as e:
            logger.debug(f"Polygon candle fetch failed for {ticker} {timeframe}: {e}")
            return []


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def calc_atr(bars: list[CandleBar], period: int = 14) -> float | None:
    """Average True Range over the last `period` bars."""
    if len(bars) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        h = bars[i].high
        lo = bars[i].low
        tr = max(h - lo, abs(h - prev_close), abs(lo - prev_close))
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Simple average of last `period` true ranges
    return sum(true_ranges[-period:]) / period


def calc_rsi(bars: list[CandleBar], period: int = 14) -> float | None:
    """Relative Strength Index over the last `period` bars."""
    if len(bars) < period + 1:
        return None

    changes = []
    for i in range(1, len(bars)):
        changes.append(bars[i].close - bars[i - 1].close)

    if len(changes) < period:
        return None

    recent = changes[-period:]
    gains = [c for c in recent if c > 0]
    losses = [-c for c in recent if c < 0]

    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_obv(bars: list[CandleBar]) -> float | None:
    """On-Balance Volume — cumulative volume weighted by price direction."""
    if len(bars) < 2:
        return None

    obv = 0.0
    for i in range(1, len(bars)):
        if bars[i].close > bars[i - 1].close:
            obv += bars[i].volume
        elif bars[i].close < bars[i - 1].close:
            obv -= bars[i].volume
        # Equal close → no change
    return obv


def calc_volume_trend(bars: list[CandleBar], lookback: int = 5) -> str | None:
    """Compare recent volume to earlier volume. Returns 'rising', 'falling', or None."""
    if len(bars) < lookback * 2:
        return None

    recent = bars[-lookback:]
    earlier = bars[-lookback * 2:-lookback]

    recent_avg = sum(b.volume for b in recent) / lookback
    earlier_avg = sum(b.volume for b in earlier) / lookback

    if earlier_avg <= 0:
        return None

    ratio = recent_avg / earlier_avg
    if ratio > 1.2:
        return "rising"
    elif ratio < 0.8:
        return "falling"
    return None


def detect_candle_pattern(bars: list[CandleBar]) -> str | None:
    """Detect exhaustion patterns in the most recent candle(s).

    Returns pattern name or None:
    - 'doji': tiny body relative to range (indecision)
    - 'hammer': long lower wick, small body at top (reversal)
    - 'shooting_star': long upper wick, small body at bottom (reversal)
    - 'engulfing_bearish': current bar body engulfs previous bar body, bearish
    - 'engulfing_bullish': current bar body engulfs previous bar body, bullish
    """
    if not bars:
        return None

    bar = bars[-1]
    body = abs(bar.close - bar.open)
    full_range = bar.high - bar.low

    if full_range <= 0:
        return None

    body_ratio = body / full_range
    upper_wick = bar.high - max(bar.open, bar.close)
    lower_wick = min(bar.open, bar.close) - bar.low

    # Doji: body < 10% of range
    if body_ratio < 0.10:
        return "doji"

    # Shooting star: upper wick > 2x body, lower wick < 30% of range
    if upper_wick > body * 2 and lower_wick < full_range * 0.3:
        return "shooting_star"

    # Hammer: lower wick > 2x body, upper wick < 30% of range
    if lower_wick > body * 2 and upper_wick < full_range * 0.3:
        return "hammer"

    # Engulfing patterns (need 2 bars)
    if len(bars) >= 2:
        prev = bars[-2]
        prev_body = abs(prev.close - prev.open)
        curr_body_top = max(bar.open, bar.close)
        curr_body_bot = min(bar.open, bar.close)
        prev_body_top = max(prev.open, prev.close)
        prev_body_bot = min(prev.open, prev.close)

        if (
            curr_body_top > prev_body_top
            and curr_body_bot < prev_body_bot
            and body > prev_body
        ):
            if bar.close < bar.open:
                return "engulfing_bearish"
            else:
                return "engulfing_bullish"

    return None


# ---------------------------------------------------------------------------
# High-level exhaustion signal (combines indicators + patterns)
# ---------------------------------------------------------------------------


def check_exhaustion(
    candle_data: dict,
    direction: str,
    peak_gain_pct: float,
    min_gain_pct: float = 35.0,
) -> tuple[bool, str]:
    """Check for exhaustion using multi-timeframe candle data.

    Returns (is_exhausted, reason).

    Exhaustion signals (for calls — reversed for puts):
    1. RSI > 70 on 5m (overbought)
    2. Bearish candle pattern (shooting_star, doji, engulfing_bearish) on 5m
    3. OBV divergence: price making highs but OBV falling
    4. Volume declining while price rising (distribution)

    Requires at least 2 of 4 signals to confirm exhaustion.
    """
    if peak_gain_pct < min_gain_pct:
        return False, f"peak gain +{peak_gain_pct:.1f}% < {min_gain_pct:.0f}%"

    indicators = candle_data.get("indicators", {})
    tf_5m = indicators.get("5m", {})

    if not any(tf_5m.get(k) is not None for k in ("rsi", "obv", "pattern")):
        return False, "no candle data available"

    is_call = direction in ("call", "bullish", "long")
    signals = []

    # Signal 1: RSI extremes
    rsi = tf_5m.get("rsi")
    if rsi is not None:
        if is_call and rsi > 70:
            signals.append(f"RSI={rsi:.0f} (overbought)")
        elif not is_call and rsi < 30:
            signals.append(f"RSI={rsi:.0f} (oversold)")

    # Signal 2: Candle patterns
    pattern = tf_5m.get("pattern")
    if pattern:
        if is_call and pattern in ("shooting_star", "doji", "engulfing_bearish"):
            signals.append(f"pattern={pattern}")
        elif not is_call and pattern in ("hammer", "doji", "engulfing_bullish"):
            signals.append(f"pattern={pattern}")

    # Signal 3: Volume trend (distribution = price up but volume falling)
    vol_trend = tf_5m.get("volume_trend")
    if vol_trend == "falling":
        signals.append("volume declining")

    # Signal 4: Check 15m for confirmation
    tf_15m = indicators.get("15m", {})
    rsi_15 = tf_15m.get("rsi")
    if rsi_15 is not None:
        if is_call and rsi_15 > 65:
            signals.append(f"15m RSI={rsi_15:.0f}")
        elif not is_call and rsi_15 < 35:
            signals.append(f"15m RSI={rsi_15:.0f}")

    # Require at least 2 confirming signals
    if len(signals) >= 2:
        reason = f"Exhaustion detected ({len(signals)} signals): {', '.join(signals)}"
        return True, reason

    return False, f"no exhaustion ({len(signals)}/2 signals)"


# ---------------------------------------------------------------------------
# ENRG — Early Negative Thesis Revalidation Gate
# ---------------------------------------------------------------------------

# Per-timeframe weights for the voting system
ENRG_TF_WEIGHTS: dict[str, int] = {
    "5m": 1,
    "15m": 1,
    "30m": 1,
    "1h": 2,
    "4h": 2,
}

# Extreme patterns that trigger IMMEDIATE_EXIT on higher timeframes
_EXTREME_BEARISH = {"engulfing_bearish", "shooting_star"}
_EXTREME_BULLISH = {"engulfing_bullish", "hammer"}


def enrg_vote_tf(
    indicators: dict,
    direction: str,
) -> str:
    """Vote for a single timeframe: 'BULLISH', 'BEARISH', or 'NEUTRAL'.

    For calls:
      BULLISH  = RSI > 40 AND (OBV > 0 OR bullish pattern)
      BEARISH  = RSI < 40 OR bearish pattern
    For puts: reversed.
    """
    rsi = indicators.get("rsi")
    obv = indicators.get("obv")
    pattern = indicators.get("pattern")

    if rsi is None:
        return "NEUTRAL"

    is_call = direction in ("call", "bullish", "long")

    if is_call:
        bearish_pattern = pattern in _EXTREME_BEARISH if pattern else False
        bullish_pattern = pattern in _EXTREME_BULLISH if pattern else False

        if rsi < 40 or bearish_pattern:
            return "BEARISH"
        if rsi > 40 and (obv is not None and obv > 0 or bullish_pattern):
            return "BULLISH"
    else:
        bearish_pattern = pattern in _EXTREME_BULLISH if pattern else False
        bullish_pattern = pattern in _EXTREME_BEARISH if pattern else False

        if rsi > 60 or bearish_pattern:
            return "BEARISH"
        if rsi < 60 and (obv is not None and obv < 0 or bullish_pattern):
            return "BULLISH"

    return "NEUTRAL"


def evaluate_enrg(
    candle_data: dict,
    direction: str,
) -> tuple[str, str]:
    """Run ENRG weighted voting across all timeframes.

    Returns (action, reason) where action is one of:
      'HOLD'           — thesis intact, widen stop +15%
      'IMMEDIATE_EXIT' — extreme reversal pattern on higher TF
      'PROCEED'        — not enough data or inconclusive, proceed to hard stop
    """
    indicators = candle_data.get("indicators", {})
    if not indicators:
        return "PROCEED", "no candle data for ENRG"

    is_call = direction in ("call", "bullish", "long")

    # Check for extreme pattern override on 1h/4h FIRST
    for tf in ("1h", "4h"):
        tf_ind = indicators.get(tf, {})
        pattern = tf_ind.get("pattern")
        if pattern:
            extreme_set = _EXTREME_BEARISH if is_call else _EXTREME_BULLISH
            if pattern in extreme_set:
                return (
                    "IMMEDIATE_EXIT",
                    f"ENRG extreme override: {tf} {pattern}",
                )

    # Weighted voting
    bullish_weight = 0
    bearish_weight = 0
    total_weight = 0
    votes: list[str] = []

    for tf, weight in ENRG_TF_WEIGHTS.items():
        tf_ind = indicators.get(tf, {})
        if not any(tf_ind.get(k) is not None for k in ("rsi", "obv", "pattern")):
            votes.append(f"{tf}:SKIP")
            continue

        vote = enrg_vote_tf(tf_ind, direction)
        votes.append(f"{tf}:{vote}")
        total_weight += weight

        if vote == "BULLISH":
            bullish_weight += weight
        elif vote == "BEARISH":
            bearish_weight += weight

    if total_weight == 0:
        return "PROCEED", "ENRG: no TF data available"

    vote_str = ", ".join(votes)

    # Thesis holds if bullish weight > bearish weight
    if bullish_weight > bearish_weight:
        return (
            "HOLD",
            f"ENRG HOLD (thesis intact): bullish={bullish_weight} > "
            f"bearish={bearish_weight} [{vote_str}]",
        )

    if bearish_weight > bullish_weight:
        return (
            "IMMEDIATE_EXIT",
            f"ENRG EXIT (thesis broken): bearish={bearish_weight} > "
            f"bullish={bullish_weight} [{vote_str}]",
        )

    # Tie → inconclusive, proceed to normal hard stop
    return (
        "PROCEED",
        f"ENRG inconclusive: bullish={bullish_weight} = "
        f"bearish={bearish_weight} [{vote_str}]",
    )
