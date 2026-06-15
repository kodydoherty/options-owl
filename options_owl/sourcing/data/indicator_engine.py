"""Compute all technical indicators locally from raw OHLCV bars.

Pure functions — no I/O, no side effects, fully deterministic.
All formulas match the N8N workflow parameters (see Appendix D of Doc 01):
  - EMA: 9, 21, 200
  - RSI: period=9, Wilder smoothing
  - MACD: fast=5, slow=13, signal=4
  - Bollinger: period=20, sd=2
  - Keltner: period=20, multiplier=1.5
  - ATR: period=14
  - VWAP: session (full day)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IndicatorSet:
    """All computed indicators for a single ticker at a point in time."""

    # EMA
    ema9: float = 0.0
    ema21: float = 0.0
    ema200: float = 0.0
    ema_cross_strength: float = 0.0  # (ema9 - ema21) / atr, clipped -1 to +1

    # RSI (Wilder smoothing, period=9)
    rsi9: float = 50.0

    # MACD (5, 13, 4)
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Bollinger Bands (20, 2.0)
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_width: float = 0.0
    bb_squeeze: bool = False  # BB inside Keltner

    # VWAP
    vwap: float = 0.0
    vwap_slope: float = 0.0  # slope of VWAP over last 5 bars

    # ATR (14)
    atr14: float = 0.0
    atr_expanding: bool = False  # current ATR > 20-bar avg ATR

    # Keltner Channels (20, 1.5)
    keltner_upper: float = 0.0
    keltner_lower: float = 0.0

    # Volume
    volume_ratio: float = 1.0  # current / 20-bar avg
    obv_slope: float = 0.0  # OBV change over last 5 bars

    # ADX (14)
    adx: float = 0.0

    # Price context
    last_close: float = 0.0
    last_high: float = 0.0
    last_low: float = 0.0

    # Institutional levels (sweep detection)
    pdh: float = 0.0           # Previous Day High
    pdl: float = 0.0           # Previous Day Low
    pwh: float = 0.0           # Previous Week High
    pwl: float = 0.0           # Previous Week Low
    session_high: float = 0.0  # Today's session high
    session_low: float = 0.0   # Today's session low
    sweep_pdh: bool = False    # Price swept above PDH then reversed
    sweep_pdl: bool = False    # Price swept below PDL then reversed
    sweep_pwh: bool = False    # Price swept above PWH then reversed
    sweep_pwl: bool = False    # Price swept below PWL then reversed
    sweep_session_high: bool = False
    sweep_session_low: bool = False


# ---------------------------------------------------------------------------
# Core computation functions (pure, no side effects)
# ---------------------------------------------------------------------------


def calc_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA from oldest-first values. Returns array same length as input.

    First `period` values use SMA as seed, then exponential smoothing.
    """
    if len(values) < period:
        return np.full_like(values, np.nan)

    k = 2.0 / (period + 1)
    ema = np.empty_like(values)
    ema[:period - 1] = np.nan
    ema[period - 1] = np.mean(values[:period])

    for i in range(period, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)

    return ema


def calc_rsi(closes: np.ndarray, period: int = 9) -> float:
    """Compute RSI using Wilder smoothing (matches N8N/Twelve Data).

    Returns the most recent RSI value.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing: first avg is SMA, then exponential
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_macd(
    closes: np.ndarray, fast: int = 5, slow: int = 13, signal: int = 4
) -> tuple[float, float, float]:
    """Compute MACD line, signal line, and histogram.

    Uses N8N parameters: fast=5, slow=13, signal=4 (faster than standard 12/26/9).
    Returns (macd_line, signal_line, histogram) for the most recent bar.
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0

    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd_line = ema_fast - ema_slow

    # Signal line is EMA of MACD line
    valid_macd = macd_line[~np.isnan(macd_line)]
    if len(valid_macd) < signal:
        return float(macd_line[-1]), 0.0, float(macd_line[-1])

    signal_ema = calc_ema(valid_macd, signal)
    hist = valid_macd[-1] - signal_ema[-1]

    return float(valid_macd[-1]), float(signal_ema[-1]), float(hist)


def calc_bollinger(
    closes: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float, float]:
    """Compute Bollinger Bands.

    Returns (upper, middle, lower, width) for the most recent bar.
    """
    if len(closes) < period:
        c = float(closes[-1]) if len(closes) > 0 else 0.0
        return c, c, c, 0.0

    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid if mid > 0 else 0.0

    return upper, mid, lower, width


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range (Wilder smoothing)."""
    if len(closes) < period + 1:
        if len(highs) > 0:
            return float(np.mean(highs - lows))
        return 0.0

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )

    # Wilder smoothing
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return atr


def calc_keltner(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    period: int = 20, multiplier: float = 1.5,
) -> tuple[float, float]:
    """Compute Keltner Channel upper and lower bands.

    Returns (upper, lower).
    """
    if len(closes) < period:
        c = float(closes[-1]) if len(closes) > 0 else 0.0
        return c, c

    mid = float(np.mean(closes[-period:]))
    atr = calc_atr(highs, lows, closes, period=period)
    return mid + multiplier * atr, mid - multiplier * atr


def calc_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """Compute session VWAP from OHLCV data."""
    if len(closes) == 0 or np.sum(volumes) == 0:
        return 0.0

    typical_price = (highs + lows + closes) / 3.0
    return float(np.sum(typical_price * volumes) / np.sum(volumes))


def calc_obv_slope(closes: np.ndarray, volumes: np.ndarray, lookback: int = 5) -> float:
    """Compute OBV slope over the last `lookback` bars."""
    if len(closes) < lookback + 1:
        return 0.0

    obv = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    # Slope = change over lookback
    if obv[-1] == 0 and obv[-lookback] == 0:
        return 0.0
    denom = max(abs(obv[-lookback]), 1.0)
    return float((obv[-1] - obv[-lookback]) / denom)


def calc_adx(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> float:
    """Compute ADX (Average Directional Index)."""
    if len(closes) < period * 2:
        return 0.0

    up_move = np.diff(highs)
    down_move = -np.diff(lows)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = calc_atr(highs, lows, closes, period)
    if atr == 0:
        return 0.0

    # Wilder smoothing for +DI and -DI
    smooth_plus = float(np.mean(plus_dm[:period]))
    smooth_minus = float(np.mean(minus_dm[:period]))

    for i in range(period, len(plus_dm)):
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period

    plus_di = 100 * smooth_plus / atr if atr > 0 else 0
    minus_di = 100 * smooth_minus / atr if atr > 0 else 0

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0

    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx  # Simplified — full ADX would smooth DX over another period


# ---------------------------------------------------------------------------
# Institutional levels & sweep detection
# ---------------------------------------------------------------------------

BARS_PER_DAY = 78  # 5-minute bars in a 6.5-hour trading day


def _split_into_days(candles: list[dict]) -> list[list[dict]]:
    """Split candles into per-day groups.

    Uses the 'timestamp' or 'time' key if available for day boundaries;
    otherwise falls back to the heuristic that 78 bars = 1 trading day.
    """
    if not candles:
        return []

    # Try timestamp-based splitting first
    ts_key = None
    if "timestamp" in candles[0]:
        ts_key = "timestamp"
    elif "time" in candles[0]:
        ts_key = "time"

    if ts_key is not None:
        from collections import OrderedDict

        days: OrderedDict[str, list[dict]] = OrderedDict()
        for c in candles:
            raw = c[ts_key]
            # Support both epoch (int/float) and ISO string
            if isinstance(raw, (int, float)):
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(raw / 1000 if raw > 1e12 else raw, tz=timezone.utc)
                day_key = dt.strftime("%Y-%m-%d")
            else:
                day_key = str(raw)[:10]
            days.setdefault(day_key, []).append(c)
        return list(days.values())

    # Heuristic: chunk into groups of BARS_PER_DAY
    result = []
    for i in range(0, len(candles), BARS_PER_DAY):
        chunk = candles[i : i + BARS_PER_DAY]
        if chunk:
            result.append(chunk)
    return result


def calc_institutional_levels(candles: list[dict]) -> dict:
    """Compute PDH/PDL, PWH/PWL, and session high/low from candle data.

    Args:
        candles: List of OHLCV dicts, oldest-first (5m bars).

    Returns:
        Dict with keys: pdh, pdl, pwh, pwl, session_high, session_low.
        All default to 0.0 if insufficient data.
    """
    result = {
        "pdh": 0.0, "pdl": 0.0,
        "pwh": 0.0, "pwl": 0.0,
        "session_high": 0.0, "session_low": 0.0,
    }

    if not candles or len(candles) < 2:
        return result

    days = _split_into_days(candles)

    if len(days) < 1:
        return result

    # Session (today) = last day group
    today = days[-1]
    today_highs = [c["high"] for c in today]
    today_lows = [c["low"] for c in today]
    result["session_high"] = max(today_highs)
    result["session_low"] = min(today_lows)

    # PDH / PDL = previous day (second-to-last group)
    if len(days) >= 2:
        prev_day = days[-2]
        result["pdh"] = max(c["high"] for c in prev_day)
        result["pdl"] = min(c["low"] for c in prev_day)

    # PWH / PWL = previous 5 days (excluding today)
    # Use up to 5 previous day groups
    prev_days = days[:-1][-5:] if len(days) > 1 else []
    if prev_days:
        all_highs = [c["high"] for day in prev_days for c in day]
        all_lows = [c["low"] for day in prev_days for c in day]
        result["pwh"] = max(all_highs)
        result["pwl"] = min(all_lows)

    return result


def detect_sweeps(candles: list[dict], levels: dict) -> dict:
    """Detect sweep patterns in the last 3-5 candles.

    A "sweep" means price went BEYOND a level then the close came back:
      - sweep_pdh: high > PDH then close < PDH (bull trap / manipulation)
      - sweep_pdl: low < PDL then close > PDL (bear trap / reversal)
      - Same logic for PWH/PWL and session high/low.

    Args:
        candles: OHLCV list, oldest-first.
        levels: Dict from calc_institutional_levels().

    Returns:
        Dict of booleans for each sweep type.
    """
    sweeps = {
        "sweep_pdh": False,
        "sweep_pdl": False,
        "sweep_pwh": False,
        "sweep_pwl": False,
        "sweep_session_high": False,
        "sweep_session_low": False,
    }

    if not candles or len(candles) < 2:
        return sweeps

    # Check last 5 candles (or fewer if not enough data)
    window = candles[-5:]

    pdh = levels.get("pdh", 0.0)
    pdl = levels.get("pdl", 0.0)
    pwh = levels.get("pwh", 0.0)
    pwl = levels.get("pwl", 0.0)
    session_high = levels.get("session_high", 0.0)
    session_low = levels.get("session_low", 0.0)

    for bar in window:
        h = bar["high"]
        low = bar["low"]
        c = bar["close"]

        # Sweep above then close below = bull trap (bearish sweep)
        if pdh > 0 and h > pdh and c < pdh:
            sweeps["sweep_pdh"] = True
        if pwh > 0 and h > pwh and c < pwh:
            sweeps["sweep_pwh"] = True
        if session_high > 0 and h > session_high and c < session_high:
            sweeps["sweep_session_high"] = True

        # Sweep below then close above = bear trap (bullish sweep)
        if pdl > 0 and low < pdl and c > pdl:
            sweeps["sweep_pdl"] = True
        if pwl > 0 and low < pwl and c > pwl:
            sweeps["sweep_pwl"] = True
        if session_low > 0 and low < session_low and c > session_low:
            sweeps["sweep_session_low"] = True

    return sweeps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_indicators(candles: list[dict]) -> IndicatorSet:
    """Compute full indicator set from OHLCV candle data.

    Args:
        candles: List of dicts with keys: open, high, low, close, volume.
                 Ordered oldest-first. Minimum 30 bars for reliable output.

    Returns:
        IndicatorSet with all computed values.
    """
    if not candles or len(candles) < 5:
        return IndicatorSet()

    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    volumes = np.array([c.get("volume", 0) for c in candles], dtype=np.float64)

    # EMA
    ema9_arr = calc_ema(closes, 9)
    ema21_arr = calc_ema(closes, 21)
    ema9 = float(ema9_arr[-1]) if not np.isnan(ema9_arr[-1]) else float(closes[-1])
    ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else float(closes[-1])

    # EMA 200 (only meaningful with daily data, but compute from available bars)
    ema200 = float(closes[-1])
    if len(closes) >= 200:
        ema200_arr = calc_ema(closes, 200)
        ema200 = float(ema200_arr[-1])

    # ATR
    atr14 = calc_atr(highs, lows, closes, 14)

    # EMA cross strength: normalized by ATR
    ema_cross_strength = 0.0
    if atr14 > 0:
        raw = (ema9 - ema21) / atr14
        ema_cross_strength = max(-1.0, min(1.0, raw))

    # RSI
    rsi9 = calc_rsi(closes, 9)

    # MACD
    macd_line, macd_signal, macd_histogram = calc_macd(closes, 5, 13, 4)

    # Bollinger
    bb_upper, bb_mid, bb_lower, bb_width = calc_bollinger(closes, 20, 2.0)

    # Keltner
    keltner_upper, keltner_lower = calc_keltner(closes, highs, lows, 20, 1.5)

    # Squeeze: BB inside Keltner
    bb_squeeze = bb_upper < keltner_upper and bb_lower > keltner_lower

    # VWAP
    vwap = calc_vwap(highs, lows, closes, volumes)
    # VWAP slope over last 5 bars (approximate by computing VWAP at different windows)
    vwap_slope = 0.0
    if len(candles) >= 10:
        vwap_recent = calc_vwap(highs[-5:], lows[-5:], closes[-5:], volumes[-5:])
        vwap_prior = calc_vwap(highs[-10:-5], lows[-10:-5], closes[-10:-5], volumes[-10:-5])
        if vwap_prior > 0:
            vwap_slope = (vwap_recent - vwap_prior) / vwap_prior

    # ATR expansion
    atr_expanding = False
    if len(closes) >= 34:  # 20 bars of ATR history
        atr_values = []
        for i in range(20):
            end = len(closes) - i
            start = max(0, end - 15)
            if end - start >= 2:
                atr_values.append(calc_atr(highs[start:end], lows[start:end], closes[start:end], 14))
        if atr_values:
            atr_expanding = atr14 > np.mean(atr_values)

    # Volume ratio
    volume_ratio = 1.0
    if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0:
        volume_ratio = float(volumes[-1] / np.mean(volumes[-20:]))

    # OBV slope
    obv_slope = calc_obv_slope(closes, volumes, 5)

    # ADX
    adx = calc_adx(highs, lows, closes, 14)

    # Institutional levels & sweeps
    inst_levels = calc_institutional_levels(candles)
    sweep_flags = detect_sweeps(candles, inst_levels)

    return IndicatorSet(
        ema9=ema9,
        ema21=ema21,
        ema200=ema200,
        ema_cross_strength=ema_cross_strength,
        rsi9=rsi9,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_mid=bb_mid,
        bb_width=bb_width,
        bb_squeeze=bb_squeeze,
        vwap=vwap,
        vwap_slope=vwap_slope,
        atr14=atr14,
        atr_expanding=atr_expanding,
        keltner_upper=keltner_upper,
        keltner_lower=keltner_lower,
        volume_ratio=volume_ratio,
        obv_slope=obv_slope,
        adx=adx,
        last_close=float(closes[-1]),
        last_high=float(highs[-1]),
        last_low=float(lows[-1]),
        # Institutional levels
        pdh=inst_levels["pdh"],
        pdl=inst_levels["pdl"],
        pwh=inst_levels["pwh"],
        pwl=inst_levels["pwl"],
        session_high=inst_levels["session_high"],
        session_low=inst_levels["session_low"],
        # Sweep flags
        sweep_pdh=sweep_flags["sweep_pdh"],
        sweep_pdl=sweep_flags["sweep_pdl"],
        sweep_pwh=sweep_flags["sweep_pwh"],
        sweep_pwl=sweep_flags["sweep_pwl"],
        sweep_session_high=sweep_flags["sweep_session_high"],
        sweep_session_low=sweep_flags["sweep_session_low"],
    )
