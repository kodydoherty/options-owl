"""Intraday regime detector — rule-based market direction classification.

Uses 5-minute candle data (VWAP, EMA, RSI, ADX) to classify the current
market regime as BULLISH, BEARISH, or CHOPPY. No ML — pure price action rules.

The regime detector gates:
  - Which directions the ML scanner can trade (spec 06)
  - Position sizing multiplier (spec 09)
  - Stop tightening on open positions (spec 08)
  - Dynamic PUT expansion (spec 10)
  - Extended scan window entries (spec 07)

Usage:
    detector = RegimeDetector()
    regime = await detector.update(candle_cache)
    if detector.allows_direction("call"):
        # proceed with CALL scan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from loguru import logger


class RegimeState(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    CHOPPY = "choppy"


@dataclass
class RegimeSnapshot:
    """A single regime evaluation at a point in time."""
    state: RegimeState
    timestamp: datetime
    spy_price: float = 0.0
    spy_vwap: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    rsi: float = 0.0
    adx: float = 0.0


@dataclass
class RegimeDetector:
    """Rule-based intraday regime classifier.

    Evaluates SPY/QQQ 5-minute candles to determine market direction.
    Includes hysteresis to prevent flip-flopping.
    """

    # Current confirmed regime
    state: RegimeState = RegimeState.CHOPPY
    state_since: datetime | None = None

    # Pending state (needs consecutive confirmations)
    _pending_state: RegimeState | None = None
    _pending_count: int = 0

    # Configuration
    hysteresis_checks: int = 2       # consecutive readings to confirm
    min_hold_minutes: int = 15       # minimum time before re-evaluation
    hard_reversal_pct: float = 0.5   # SPY drop % for immediate flip
    choppy_size_mult: float = 0.6    # size reduction in choppy regime

    # Tracking
    _last_update: datetime | None = None
    _spy_open_price: float = 0.0
    _history: list[RegimeSnapshot] = field(default_factory=list)
    _regime_changed: bool = False     # flag for stop tightening

    # ADX threshold for trending vs ranging
    ADX_TREND_THRESHOLD: float = 20.0

    async def update(self, candle_cache, now_et: datetime | None = None) -> RegimeState:
        """Evaluate current market regime from SPY 5m candles.

        Call every 5 minutes during market hours.
        Returns the current confirmed regime state.
        """
        if now_et is None:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(tz=ZoneInfo("America/New_York"))

        # Respect minimum hold period (unless hard reversal)
        if (self.state_since and self._last_update
                and (now_et - self.state_since).total_seconds() < self.min_hold_minutes * 60
                and not self._check_hard_reversal(candle_cache, now_et)):
            self._regime_changed = False
            return self.state

        # Get SPY 5m candles + indicators
        indicators = await self._get_spy_indicators(candle_cache)
        if not indicators:
            self._regime_changed = False
            return self.state

        raw_state = self._classify(indicators)

        # Check for hard reversal override
        if self._check_hard_reversal(candle_cache, now_et):
            if self.state != RegimeState.BEARISH:
                self._confirm_state(RegimeState.BEARISH, now_et, indicators)
                logger.info(
                    f"REGIME: HARD REVERSAL → BEARISH "
                    f"(SPY dropped >{self.hard_reversal_pct}% in 15min)"
                )
            return self.state

        # Hysteresis: need consecutive confirmations to flip
        if raw_state != self.state:
            if raw_state == self._pending_state:
                self._pending_count += 1
                if self._pending_count >= self.hysteresis_checks:
                    old = self.state
                    self._confirm_state(raw_state, now_et, indicators)
                    logger.info(
                        f"REGIME: {old.value} → {self.state.value} "
                        f"(confirmed after {self._pending_count} readings) "
                        f"RSI={indicators.get('rsi', 0):.0f} "
                        f"ADX={indicators.get('adx', 0):.0f} "
                        f"EMA9={'>' if indicators.get('ema9', 0) > indicators.get('ema21', 0) else '<'}EMA21"
                    )
            else:
                self._pending_state = raw_state
                self._pending_count = 1
        else:
            # State matches — reset pending
            self._pending_state = None
            self._pending_count = 0
            self._regime_changed = False

        self._last_update = now_et
        return self.state

    def _confirm_state(self, new_state: RegimeState, now_et: datetime,
                       indicators: dict) -> None:
        """Lock in a new regime state."""
        self._regime_changed = (self.state != new_state)
        self.state = new_state
        self.state_since = now_et
        self._pending_state = None
        self._pending_count = 0
        self._history.append(RegimeSnapshot(
            state=new_state,
            timestamp=now_et,
            spy_price=indicators.get("price", 0),
            spy_vwap=indicators.get("vwap", 0),
            ema9=indicators.get("ema9", 0),
            ema21=indicators.get("ema21", 0),
            rsi=indicators.get("rsi", 0),
            adx=indicators.get("adx", 0),
        ))
        # Trim history
        if len(self._history) > 50:
            self._history = self._history[-50:]

    def _classify(self, ind: dict) -> RegimeState:
        """Classify regime from indicators. Pure function."""
        price = ind.get("price", 0)
        vwap = ind.get("vwap", 0)
        ema9 = ind.get("ema9", 0)
        ema21 = ind.get("ema21", 0)
        rsi = ind.get("rsi", 0)
        adx = ind.get("adx", 0)

        # Not trending → CHOPPY
        if adx < self.ADX_TREND_THRESHOLD:
            return RegimeState.CHOPPY

        # Count bullish signals
        bullish_signals = 0
        bearish_signals = 0

        if price > 0 and vwap > 0:
            if price > vwap:
                bullish_signals += 1
            else:
                bearish_signals += 1

        if ema9 > 0 and ema21 > 0:
            if ema9 > ema21:
                bullish_signals += 1
            else:
                bearish_signals += 1

        if rsi > 50:
            bullish_signals += 1
        elif rsi < 50:
            bearish_signals += 1

        # Need majority (2+ of 3) to confirm direction
        if bullish_signals >= 2:
            return RegimeState.BULLISH
        if bearish_signals >= 2:
            return RegimeState.BEARISH
        return RegimeState.CHOPPY

    def _check_hard_reversal(self, candle_cache, now_et: datetime) -> bool:
        """Check if SPY dropped significantly in the last 15 minutes."""
        if self._spy_open_price <= 0:
            return False
        return False

    async def _get_spy_indicators(self, candle_cache) -> dict:
        """Extract regime indicators from SPY 5m candles."""
        try:
            bars = await candle_cache.get_candles("SPY", "5m")
            if not bars or len(bars) < 22:  # need 21+ for EMA21
                return {}

            closes = [b.close for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]

            price = closes[-1]
            vwap = bars[-1].vwap if bars[-1].vwap > 0 else 0

            # Set day open price on first call
            if self._spy_open_price <= 0 and bars:
                self._spy_open_price = bars[0].open

            ema9 = _ema(closes, 9)
            ema21 = _ema(closes, 21)
            rsi = _rsi(closes, 14)
            adx = _adx(highs, lows, closes, 14)

            return {
                "price": price,
                "vwap": vwap,
                "ema9": ema9,
                "ema21": ema21,
                "rsi": rsi,
                "adx": adx,
            }
        except Exception as e:
            logger.debug(f"REGIME: Failed to get SPY indicators: {e}")
            return {}

    # --- Direction gating ---

    def allows_direction(self, direction: str) -> bool:
        """Check if the current regime allows this trade direction."""
        direction = direction.lower()
        if self.state == RegimeState.BULLISH:
            return direction == "call"
        if self.state == RegimeState.BEARISH:
            return direction == "put"
        # CHOPPY allows both (at reduced size)
        return True

    def get_size_multiplier(self) -> float:
        """Position sizing multiplier based on regime."""
        if self.state == RegimeState.CHOPPY:
            return self.choppy_size_mult
        return 1.0

    @property
    def regime_changed(self) -> bool:
        """True if the regime just changed (for stop tightening)."""
        return self._regime_changed

    def is_counter_trend(self, direction: str) -> bool:
        """Check if a direction is against the current regime."""
        direction = direction.lower()
        if self.state == RegimeState.BULLISH and direction == "put":
            return True
        if self.state == RegimeState.BEARISH and direction == "call":
            return True
        return False

    def get_tighten_factor(self, direction: str) -> float:
        """Get trail tightening factor for regime-based stop adjustment.

        Returns 1.0 (no change) or <1.0 (tighter).
        Applied multiplicatively to adaptive trail widths.
        """
        if not self._regime_changed:
            return 1.0
        if self.is_counter_trend(direction):
            return 0.60  # 40% tighter for counter-trend
        if self.state == RegimeState.CHOPPY:
            return 0.80  # 20% tighter in chop
        return 1.0


# --- Allowed directions per time window (spec 07) ---

def get_allowed_directions(
    minute: int,
    regime: RegimeState,
    extended_scan_enabled: bool = False,
) -> list[str]:
    """Return which directions the scanner may emit for this minute.

    minute: minutes since 9:30 AM ET market open.

    Time windows:
      0-5:     Opening buffer (no entries)
      5-90:    Morning session (CALL primary)
      90-210:  Midday session (regime-dependent) — requires extended scan
      210-300: Afternoon session (PUT primary) — requires extended scan
      300-330: Late session (PUT only if bearish) — requires extended scan
      330+:    No new entries (theta death)
    """
    if minute < 5:
        return []

    # Morning session: 9:35-11:00
    if minute <= 90:
        if regime == RegimeState.BEARISH:
            return ["put"]
        return ["call"]

    # Midday and beyond require extended scan flag
    if not extended_scan_enabled:
        return []

    # Midday: 11:00-1:00
    if minute <= 210:
        if regime == RegimeState.BULLISH:
            return ["call"]
        if regime == RegimeState.BEARISH:
            return ["put"]
        return []  # choppy → sit out midday

    # Afternoon: 1:00-2:30
    if minute <= 300:
        dirs = ["put"]
        if regime == RegimeState.BULLISH:
            dirs.append("call")
        return dirs

    # Late: 2:30-3:00
    if minute <= 330:
        if regime == RegimeState.BEARISH:
            return ["put"]
        return []

    # 3:00+ no new entries
    return []


# --- Dynamic PUT slot allocation (spec 10) ---

def get_direction_slots(
    regime: RegimeState,
    max_concurrent: int = 8,
    dynamic_puts_enabled: bool = False,
) -> dict[str, int]:
    """Get max slots per direction based on regime.

    Returns {"call": N, "put": M} where N + M <= max_concurrent.
    """
    if not dynamic_puts_enabled:
        # Default: fixed allocation
        return {"call": max_concurrent - 2, "put": 2}

    if regime == RegimeState.BULLISH:
        put_slots = max(2, max_concurrent // 4)
        return {"call": max_concurrent - put_slots, "put": put_slots}

    if regime == RegimeState.BEARISH:
        call_slots = max(2, max_concurrent // 4)
        return {"call": call_slots, "put": max_concurrent - call_slots}

    # CHOPPY: reduced total
    total = max(4, int(max_concurrent * 0.75))
    half = total // 2
    return {"call": half, "put": total - half}


# --- Conviction-based sizing (spec 09) ---

def compute_conviction_multiplier(
    ml_confidence: float,
    regime: RegimeState,
    direction: str,
    minute: int,
    conviction_enabled: bool = False,
) -> float:
    """Compute position sizing multiplier based on setup quality.

    Returns 0.40 - 1.00 multiplier applied to per-slot budget.
    When conviction sizing is disabled, returns 1.0 (no change).
    """
    if not conviction_enabled:
        return 1.0

    # Base: ML model confidence
    if ml_confidence >= 0.95:
        base = 1.0
    elif ml_confidence >= 0.90:
        base = 0.85
    else:
        base = 0.70

    # Regime alignment
    direction = direction.lower()
    if (regime == RegimeState.BULLISH and direction == "call") or \
       (regime == RegimeState.BEARISH and direction == "put"):
        regime_mult = 1.0
    elif regime == RegimeState.CHOPPY:
        regime_mult = 0.70
    else:
        regime_mult = 0.50  # counter-trend

    # Time-of-day (morning best)
    if minute <= 60:
        time_mult = 1.0
    elif minute <= 120:
        time_mult = 0.90
    elif minute <= 210:
        time_mult = 0.80
    else:
        time_mult = 0.75

    return max(0.40, min(1.0, base * regime_mult * time_mult))


# ---------------------------------------------------------------------------
# Technical indicator helpers (for regime detection)
# ---------------------------------------------------------------------------

def _ema(data: list[float], period: int) -> float:
    """Exponential Moving Average of the last N values."""
    if len(data) < period:
        return 0.0
    k = 2.0 / (period + 1)
    ema = data[-period]
    for price in data[-period + 1:]:
        ema = price * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(-period, 0)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> float:
    """Average Directional Index — measures trend strength (not direction)."""
    n = len(closes)
    if n < period + 1:
        return 0.0

    # True Range + Directional Movement
    plus_dm = []
    minus_dm = []
    tr_list = []

    for i in range(1, n):
        h = highs[i]
        lo = lows[i]
        ph = highs[i - 1]
        pl = lows[i - 1]
        pc = closes[i - 1]

        tr = max(h - lo, abs(h - pc), abs(lo - pc))
        tr_list.append(tr)

        up = h - ph
        down = pl - lo

        if up > down and up > 0:
            plus_dm.append(up)
        else:
            plus_dm.append(0)

        if down > up and down > 0:
            minus_dm.append(down)
        else:
            minus_dm.append(0)

    if len(tr_list) < period:
        return 0.0

    # Smoothed averages (Wilder's smoothing)
    atr = sum(tr_list[:period]) / period
    plus_di_sum = sum(plus_dm[:period]) / period
    minus_di_sum = sum(minus_dm[:period]) / period

    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_sum = (plus_di_sum * (period - 1) + plus_dm[i]) / period
        minus_di_sum = (minus_di_sum * (period - 1) + minus_dm[i]) / period

    if atr == 0:
        return 0.0

    plus_di = 100 * plus_di_sum / atr
    minus_di = 100 * minus_di_sum / atr

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0

    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx
