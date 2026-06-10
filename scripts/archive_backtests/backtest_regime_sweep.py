"""Regime Feature Sweep — tests all 5 new features individually + in combos.

Builds on the gold standard combined backtest. Adds:
  1. Intraday regime detector (rule-based SPY candle classification)
  2. Extended scan window (midday/afternoon entries gated by regime)
  3. Regime-triggered stop tightening (tighten adaptive trail on regime flip)
  4. Conviction-based sizing (ML confidence × regime × TOD → budget mult)
  5. Dynamic PUT expansion (slot allocation by regime)

Usage:
    python scripts/backtest_regime_sweep.py                  # full sweep
    python scripts/backtest_regime_sweep.py --baseline-only  # just gold standard baseline
    python scripts/backtest_regime_sweep.py --days 90        # longer window
    python scripts/backtest_regime_sweep.py --individual     # individual feature tests only
    python scripts/backtest_regime_sweep.py --combos         # combo tests only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.regime_detector import (
    RegimeState,
    _adx,
    _ema,
    _rsi,
    compute_conviction_multiplier,
    get_allowed_directions,
    get_direction_slots,
)

ET = ZoneInfo("America/New_York")
THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"
UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")

# ── Portfolio Config (from gold standard) ─────────────────────────────────

PORTFOLIO_START = 23_000
MAX_RISK_PCT = 0.75
GFV_BUFFER_PCT = 15.0
DAILY_LOSS_CB_PCT = 15.0

# CALL config
CALL_TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]
CALL_EXCLUDED = {"MSFT"}
CALL_MAX_CONCURRENT = 4
CALL_MAX_POSITION_PCT = 0.15
CALL_MAX_POSITION_DOLLARS = 5_000
CALL_MAX_CONTRACTS = 200
CALL_PREMIUM_FLOOR = 0.20
CALL_PREMIUM_CAP = 6.0
CALL_SPREAD_GATE_PCT = 15.0
CALL_MAX_SAME_DIRECTION = 2
CALL_MAX_INDEX_CONCURRENT = 1
CALL_MIN_ENTRY_SPACING = 5
CALL_BAD_DAY_THRESHOLD = 0.90
CALL_LOSS_COUNT_THRESHOLD = 2

# ML thresholds
PATTERN_THRESHOLD = 0.85
ENTRY_THRESHOLD = 0.70
REGIME_THRESHOLD = 0.19

# Scan window (baseline)
CALL_SCAN_START = 5
CALL_SCAN_END = 90

# PUT config
PUT_TICKERS = ["SPY", "QQQ", "TSLA", "META", "IWM"]
PUT_EXCLUDED = {"PLTR", "AMD", "MSTR", "AVGO", "AAPL", "GOOGL", "NVDA", "AMZN"}
PUT_SLOTS = ["13:00", "13:30", "14:00", "14:30"]
PUT_PREMIUM_FLOOR = 0.05
PUT_PREMIUM_CAP = 0.50
PUT_TARGET_PCT = 50.0
PUT_STOP_PCT = 60.0
PUT_MAX_HOLD = 60
PUT_MAX_CONCURRENT = 2
PUT_MAX_POSITION_PCT = 0.15
PUT_MAX_POSITION_DOLLARS = 5_000
PUT_MAX_CONTRACTS = 200

# Bear mode
BEAR_MODE_THRESHOLD = -0.5
BEAR_PUT_MAX_CONCURRENT = 4
BEAR_SKIP_CALLS = True
BEAR_PUT_TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# Production gates (keep only the value-add ones)
ENABLE_AFTERNOON_DANGER = True
ENABLE_HARD_CUTOFF = True
AFTERNOON_DANGER_START = 240
AFTERNOON_DANGER_END = 330
HARD_CUTOFF_MIN = 385

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA"}

# V6 settings
_V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True,
    V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True,
    V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333,
    V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True,
    V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15,
    ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_PREMIUM_CAP=True,
    V6_PREMIUM_CAP=CALL_PREMIUM_CAP,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=CALL_SPREAD_GATE_PCT,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)


# ── Feature Config ────────────────────────────────────────────────────────


@dataclass
class FeatureConfig:
    """Toggles and params for all 5 regime features."""
    # Feature 1: Intraday regime detector
    regime_detector: bool = False
    adx_threshold: float = 20.0
    hysteresis_checks: int = 2
    min_hold_bars: int = 3       # 5m bars = 15 min

    # Feature 2: Extended scan window
    extended_scan: bool = False
    midday_scan_end: int = 210   # 11:00-1:00 (minutes after open)
    afternoon_scan_end: int = 300  # 1:00-2:30
    late_scan_end: int = 330     # 2:30-3:00

    # Feature 3: Regime stop tightening
    regime_stop_tighten: bool = False
    counter_trend_tighten: float = 0.60  # 40% tighter
    choppy_tighten: float = 0.80         # 20% tighter

    # Feature 4: Conviction sizing
    conviction_sizing: bool = False

    # Feature 5: Dynamic PUT expansion
    dynamic_puts: bool = False
    bullish_put_slots: int = 2
    bearish_put_slots: int = 6
    choppy_put_slots: int = 3

    def label(self) -> str:
        """Short label for this config."""
        parts = []
        if self.regime_detector:
            parts.append(f"RD(adx={self.adx_threshold:.0f})")
        if self.extended_scan:
            parts.append("EXT")
        if self.regime_stop_tighten:
            parts.append(f"RST({self.counter_trend_tighten:.0f}/{self.choppy_tighten:.0f})")
        if self.conviction_sizing:
            parts.append("CONV")
        if self.dynamic_puts:
            parts.append(f"DPUT(b={self.bearish_put_slots})")
        return "+".join(parts) if parts else "BASELINE"


# ── Intraday Regime Classification (from SPY 5m stock candles) ────────────


def classify_regime_from_bars(
    spy_closes_5m: list[float],
    spy_highs_5m: list[float],
    spy_lows_5m: list[float],
    spy_vwaps_5m: list[float],
    adx_threshold: float = 20.0,
) -> RegimeState:
    """Classify regime at current bar using EMA/RSI/ADX/VWAP.

    Args:
        spy_closes_5m: SPY 5-minute close prices (chronological)
        spy_highs_5m: SPY 5-minute high prices
        spy_lows_5m: SPY 5-minute low prices
        spy_vwaps_5m: SPY 5-minute VWAP prices
        adx_threshold: ADX below this = CHOPPY
    """
    if len(spy_closes_5m) < 22:
        return RegimeState.CHOPPY

    price = spy_closes_5m[-1]
    vwap = spy_vwaps_5m[-1] if spy_vwaps_5m[-1] > 0 else 0

    ema9 = _ema(spy_closes_5m, 9)
    ema21 = _ema(spy_closes_5m, 21)
    rsi = _rsi(spy_closes_5m, 14)
    adx = _adx(spy_highs_5m, spy_lows_5m, spy_closes_5m, 14)

    # Not trending → CHOPPY
    if adx < adx_threshold:
        return RegimeState.CHOPPY

    bullish = 0
    bearish = 0

    if price > 0 and vwap > 0:
        if price > vwap:
            bullish += 1
        else:
            bearish += 1

    if ema9 > 0 and ema21 > 0:
        if ema9 > ema21:
            bullish += 1
        else:
            bearish += 1

    if rsi > 50:
        bullish += 1
    elif rsi < 50:
        bearish += 1

    if bullish >= 2:
        return RegimeState.BULLISH
    if bearish >= 2:
        return RegimeState.BEARISH
    return RegimeState.CHOPPY


def compute_regime_timeline(conn, date_str: str, adx_threshold: float = 20.0,
                             hysteresis: int = 2, min_hold: int = 3) -> dict[int, RegimeState]:
    """Pre-compute regime state at every 5-minute bar for a trading day.

    Returns dict: minute_since_open → RegimeState.
    Uses hysteresis: needs `hysteresis` consecutive readings to flip.
    Uses min_hold: won't flip for at least `min_hold` bars after a change.
    """
    rows = conn.execute(
        "SELECT open, high, low, close, volume, vwap FROM stock_ohlc "
        "WHERE ticker='SPY' AND date(timestamp)=? ORDER BY timestamp",
        (date_str,),
    ).fetchall()

    if len(rows) < 30:
        return {}

    # Build 5-minute bars from 1-minute data
    closes_5m = []
    highs_5m = []
    lows_5m = []
    vwaps_5m = []

    for bar_start in range(0, len(rows), 5):
        bar_end = min(bar_start + 5, len(rows))
        chunk = rows[bar_start:bar_end]
        if not chunk:
            break
        closes_5m.append(chunk[-1][3])
        highs_5m.append(max(r[1] for r in chunk if r[1]))
        lows_5m.append(min(r[2] for r in chunk if r[2]))
        vwaps_5m.append(chunk[-1][5] if chunk[-1][5] else 0)

    # Build timeline with hysteresis
    timeline = {}
    current = RegimeState.CHOPPY
    pending = None
    pending_count = 0
    bars_since_change = 999

    for bar_idx in range(len(closes_5m)):
        minute = bar_idx * 5  # minutes since open

        if bar_idx < 4:
            # Need at least 22 bars of history for EMA21 — but we build incrementally
            # Use what we have, classify as CHOPPY if insufficient
            timeline[minute] = current
            continue

        # Classify with data up to this bar
        end = bar_idx + 1
        raw = classify_regime_from_bars(
            closes_5m[:end], highs_5m[:end], lows_5m[:end], vwaps_5m[:end],
            adx_threshold=adx_threshold,
        )

        bars_since_change += 1

        # Hysteresis logic
        if raw != current:
            if bars_since_change < min_hold:
                # Too soon to flip
                timeline[minute] = current
                continue
            if raw == pending:
                pending_count += 1
                if pending_count >= hysteresis:
                    current = raw
                    pending = None
                    pending_count = 0
                    bars_since_change = 0
            else:
                pending = raw
                pending_count = 1
        else:
            pending = None
            pending_count = 0

        timeline[minute] = current

    return timeline


# ── ML Model Loading (reused from combined backtest) ─────────────────────


def load_models():
    """Load ML models."""
    models = {}

    p_path = MODEL_DIR / "pattern_entry.txt"
    p_meta_path = MODEL_DIR / "pattern_entry_meta.json"
    if not p_path.exists():
        print(f"ERROR: No pattern model at {p_path}")
        sys.exit(1)
    models["pattern"] = lgb.Booster(model_file=str(p_path))
    with open(p_meta_path) as f:
        models["pattern_meta"] = json.load(f)

    et_path = MODEL_DIR / "entry_timing.txt"
    if et_path.exists():
        models["entry"] = lgb.Booster(model_file=str(et_path))
        models["entry_features"] = models["entry"].feature_name()

    r_path = MODEL_DIR / "regime_classifier.txt"
    if r_path.exists():
        models["regime"] = lgb.Booster(model_file=str(r_path))

    s_path = MODEL_DIR / "stop_calibration.txt"
    if s_path.exists():
        models["stop"] = lgb.Booster(model_file=str(s_path))

    return models


# ── Feature Computation (from combined backtest) ─────────────────────────


def compute_pattern_features(closes, volumes, ivs, deltas, thetas, underlyings,
                              bids, asks, idx, opening_price):
    """Compute trailing features for pattern model at position idx."""
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)

    pre5 = closes[w5_start:idx]
    pre10 = closes[w10_start:idx]
    pre5_v = volumes[w5_start:idx]
    pre5_iv = ivs[w5_start:idx]
    pre5_u = underlyings[w5_start:idx]

    valid5 = pre5[~np.isnan(pre5)]
    valid10 = pre10[~np.isnan(pre10)]
    valid5_v = pre5_v[~np.isnan(pre5_v)]
    valid5_iv = pre5_iv[~np.isnan(pre5_iv)]
    valid5_u = pre5_u[~np.isnan(pre5_u)]

    if len(valid5) < 3 or valid5[0] <= 0:
        return None

    current = closes[idx]
    if np.isnan(current) or current <= 0:
        return None

    f = {}
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0

    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    f["volume_avg_5"] = float(np.mean(valid5_v)) if len(valid5_v) > 0 else 0
    w20_start = max(0, idx - 20)
    vol20 = volumes[w20_start:idx]
    vol20_valid = vol20[~np.isnan(vol20)]
    avg20 = float(np.mean(vol20_valid)) if len(vol20_valid) > 0 else 1
    f["volume_ratio"] = f["volume_avg_5"] / max(avg20, 1)

    if len(valid5_v) >= 3:
        f["volume_trend"] = float(valid5_v[-1] / max(valid5_v[0], 1))
    else:
        f["volume_trend"] = 1.0

    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    f["drop_from_open"] = (current / opening_price - 1) * 100 if opening_price > 0 else 0

    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0
    f["minutes_since_open"] = idx
    f["premium"] = float(current)

    return f


def compute_entry_timing_features(closes, volumes, bids_arr, asks_arr, bid_sizes,
                                   ask_sizes, ivs, deltas, thetas, vegas,
                                   underlyings, stock_closes, stock_highs,
                                   stock_lows, idx, entry_features):
    """Compute entry_timing model features at position idx."""
    lookback = 15
    if idx < lookback + 1:
        return None

    entry_price = closes[idx]
    if np.isnan(entry_price) or entry_price <= 0:
        return None

    f = {}
    f["minutes_since_open"] = idx
    f["hour_bucket"] = idx // 60
    f["is_first_30min"] = 1 if idx <= 30 else 0

    prices = closes[max(0, idx - lookback):idx + 1]
    valid_prices = prices[~np.isnan(prices) & (prices > 0)]
    if len(valid_prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = float((valid_prices[-1] / valid_prices[max(-6, -len(valid_prices))] - 1) * 100) if valid_prices[max(-6, -len(valid_prices))] > 0 else 0
    f["premium_change_10m"] = float((valid_prices[-1] / valid_prices[max(-11, -len(valid_prices))] - 1) * 100) if valid_prices[max(-11, -len(valid_prices))] > 0 else 0
    f["premium_change_15m"] = float((valid_prices[-1] / valid_prices[0] - 1) * 100) if valid_prices[0] > 0 else 0

    if len(valid_prices) > 2 and all(valid_prices[:-1] > 0):
        returns = np.diff(valid_prices) / valid_prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    vols = volumes[max(0, idx - lookback):idx + 1]
    valid_vols = vols[~np.isnan(vols)]
    f["current_volume"] = float(volumes[idx]) if not np.isnan(volumes[idx]) else 0
    avg_vol = float(np.mean(valid_vols[:-1])) if len(valid_vols) > 1 else 1
    f["volume_ratio"] = float(f["current_volume"] / max(avg_vol, 1))
    if len(valid_vols) > 5 and np.std(valid_vols[:-1]) > 0:
        f["volume_zscore"] = float((valid_vols[-1] - np.mean(valid_vols[:-1])) / np.std(valid_vols[:-1]))
    else:
        f["volume_zscore"] = 0

    bid = float(bids_arr[idx]) if not np.isnan(bids_arr[idx]) else 0
    ask = float(asks_arr[idx]) if not np.isnan(asks_arr[idx]) else 0
    mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price
    f["spread"] = float(ask - bid) if ask > bid else 0
    f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
    f["bid_size"] = float(bid_sizes[idx]) if idx < len(bid_sizes) and not np.isnan(bid_sizes[idx]) else 0
    f["ask_size"] = float(ask_sizes[idx]) if idx < len(ask_sizes) and not np.isnan(ask_sizes[idx]) else 0
    f["size_imbalance"] = float((f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1))

    f["iv"] = float(ivs[idx]) if not np.isnan(ivs[idx]) else 0
    f["delta"] = float(abs(deltas[idx])) if not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if not np.isnan(thetas[idx]) else 0
    f["vega"] = float(vegas[idx]) if idx < len(vegas) and not np.isnan(vegas[idx]) else 0

    iv_window = ivs[max(0, idx - lookback):idx + 1]
    valid_iv = iv_window[~np.isnan(iv_window)]
    f["iv_change_15m"] = float(valid_iv[-1] - valid_iv[0]) if len(valid_iv) > 3 else 0

    f["underlying_price"] = float(underlyings[idx]) if not np.isnan(underlyings[idx]) else 0

    s_idx = min(idx, len(stock_closes) - 1)
    if s_idx > 5 and len(stock_closes) > 5:
        s_window = stock_closes[max(0, s_idx - lookback):s_idx + 1]
        s_valid = s_window[~np.isnan(s_window) & (s_window > 0)]
        if len(s_valid) > 1:
            f["underlying_change_5m"] = float((s_valid[-1] / s_valid[max(-6, -len(s_valid))] - 1) * 100)
            f["underlying_change_15m"] = float((s_valid[-1] / s_valid[0] - 1) * 100)
            if len(s_valid) > 2 and all(s_valid[:-1] > 0):
                f["underlying_volatility"] = float(np.std(np.diff(s_valid) / s_valid[:-1]) * 100)
            else:
                f["underlying_volatility"] = 0
        else:
            f["underlying_change_5m"] = 0
            f["underlying_change_15m"] = 0
            f["underlying_volatility"] = 0

        s_all = stock_closes[:s_idx + 1]
        s_all_valid = s_all[~np.isnan(s_all) & (s_all > 0)]
        if len(s_all_valid) > 10 and s_all_valid[0] > 0:
            f["daily_trend_pct"] = float((s_all_valid[-1] / s_all_valid[0] - 1) * 100)
        else:
            f["daily_trend_pct"] = 0

        if len(s_all_valid) > 1:
            day_lo = s_all_valid.min()
            day_hi = s_all_valid.max()
            f["daily_range_position"] = float((s_all_valid[-1] - day_lo) / (day_hi - day_lo)) if day_hi > day_lo else 0.5
        else:
            f["daily_range_position"] = 0.5

        if s_idx > 14 and len(stock_highs) > 14:
            h_window = stock_highs[max(0, s_idx - 14):s_idx]
            l_window = stock_lows[max(0, s_idx - 14):s_idx]
            h_valid = h_window[~np.isnan(h_window)]
            l_valid = l_window[~np.isnan(l_window)]
            if len(h_valid) >= 14 and len(l_valid) >= 14 and s_all_valid[-1] > 0:
                f["atr_pct"] = float(np.mean(h_valid[-14:] - l_valid[-14:]) / s_all_valid[-1] * 100)
            else:
                f["atr_pct"] = 0
        else:
            f["atr_pct"] = 0
    else:
        for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility",
                   "daily_trend_pct", "daily_range_position", "atr_pct"]:
            f[k] = 0

    recent = closes[max(0, idx - 10):idx + 1]
    valid_recent = recent[~np.isnan(recent) & (recent > 0)]
    if len(valid_recent) > 0:
        f["prem_drop_from_recent_peak"] = float((closes[idx] / np.max(valid_recent) - 1) * 100)
    else:
        f["prem_drop_from_recent_peak"] = 0

    if len(valid_recent) >= 3:
        first_half = valid_recent[:len(valid_recent) // 2]
        second_half = valid_recent[len(valid_recent) // 2:]
        if len(first_half) > 0 and len(second_half) > 0 and first_half[0] > 0 and second_half[0] > 0:
            first_change = (first_half[-1] / first_half[0] - 1) * 100
            second_change = (second_half[-1] / second_half[0] - 1) * 100
            f["decline_deceleration"] = float(second_change - first_change)
        else:
            f["decline_deceleration"] = 0
    else:
        f["decline_deceleration"] = 0

    return {k: f.get(k, 0) for k in entry_features}


def compute_regime_score(regime_model, ticker, date_str, conn, uw_conn,
                         stock_cache, prev_cache):
    """Compute regime model prediction for a ticker-day."""
    features = regime_model.feature_name()
    f = {}
    f["ticker_idx"] = CALL_TICKERS.index(ticker) if ticker in CALL_TICKERS else 0
    f["day_of_week"] = datetime.strptime(date_str, "%Y-%m-%d").weekday()

    cache_key = f"{ticker}_{date_str}"
    if cache_key not in stock_cache:
        stock_cache[cache_key] = conn.execute(
            "SELECT open, high, low, close, volume FROM stock_ohlc "
            "WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp LIMIT 15",
            (ticker, date_str),
        ).fetchall()

    stock_rows = stock_cache[cache_key]
    if not stock_rows or len(stock_rows) < 5:
        return 0.0

    morning_high = max(r[1] for r in stock_rows if r[1] and r[1] > 0)
    morning_low = min(r[2] for r in stock_rows if r[2] and r[2] > 0)
    morning_open = stock_rows[0][0] if stock_rows[0][0] else 0
    morning_close = stock_rows[-1][3] if stock_rows[-1][3] else morning_open
    morning_volume = sum(r[4] for r in stock_rows if r[4])

    if morning_low <= 0 or morning_open <= 0:
        return 0.0

    f["morning_range_pct"] = (morning_high - morning_low) / morning_low * 100
    f["morning_volume"] = float(morning_volume)
    f["morning_direction"] = (morning_close / morning_open - 1) * 100
    morning_body = abs(morning_close - morning_open)
    morning_range = morning_high - morning_low
    f["morning_body_ratio"] = morning_body / morning_range if morning_range > 0 else 0

    prev_key = f"{ticker}_prev_{date_str}"
    if prev_key not in prev_cache:
        prev_cache[prev_key] = conn.execute("""
            SELECT MAX(high), MIN(low), SUM(volume),
                   (SELECT close FROM stock_ohlc WHERE ticker=? AND date(timestamp)=date(so2.timestamp)
                    ORDER BY timestamp DESC LIMIT 1) as day_close
            FROM stock_ohlc so2
            WHERE ticker=? AND date(timestamp) < ? AND date(timestamp) >= date(?, '-7 days')
            GROUP BY date(timestamp) ORDER BY date(timestamp) DESC LIMIT 5
        """, (ticker, ticker, date_str, date_str)).fetchall()

    prev_rows = prev_cache[prev_key]
    if prev_rows:
        prev = prev_rows[0]
        f["prev_range_pct"] = (prev[0] - prev[1]) / prev[1] * 100 if prev[1] and prev[1] > 0 else 0
        f["prev_volume"] = float(prev[2] or 0)
        prev_close = prev[3] if prev[3] else 0
        f["overnight_gap_pct"] = (morning_open / prev_close - 1) * 100 if prev_close > 0 else 0
        if len(prev_rows) >= 3:
            recent_ranges = [(r[0] - r[1]) / r[1] * 100 for r in prev_rows[:3] if r[1] and r[1] > 0]
            f["avg_3d_range"] = float(np.mean(recent_ranges)) if recent_ranges else 0
        else:
            f["avg_3d_range"] = 0
        f["range_trend"] = f["morning_range_pct"] / max(f["avg_3d_range"], 0.01) - 1
        prev_vols = [float(r[2] or 0) for r in prev_rows]
        avg_prev_vol = np.mean(prev_vols) if prev_vols else 1
        f["volume_vs_prev"] = morning_volume * 26 / max(avg_prev_vol, 1)
    else:
        for k in ["prev_range_pct", "prev_volume", "overnight_gap_pct",
                   "avg_3d_range", "range_trend", "volume_vs_prev"]:
            f[k] = 0

    if uw_conn:
        gex_row = uw_conn.execute(
            "SELECT call_gamma, put_gamma, call_delta, put_delta FROM greek_exposure "
            "WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1",
            (ticker, date_str),
        ).fetchone()
        if gex_row:
            f["call_gamma"] = float(gex_row[0] or 0)
            f["put_gamma"] = float(gex_row[1] or 0)
            f["net_gamma"] = f["call_gamma"] - f["put_gamma"]
            f["call_delta"] = float(gex_row[2] or 0)
            f["put_delta"] = float(gex_row[3] or 0)
            f["net_delta"] = f["call_delta"] - f["put_delta"]
        else:
            for k in ["call_gamma", "put_gamma", "net_gamma", "call_delta", "put_delta", "net_delta"]:
                f[k] = 0
    else:
        for k in ["call_gamma", "put_gamma", "net_gamma", "call_delta", "put_delta", "net_delta"]:
            f[k] = 0

    X = np.array([[f.get(feat, 0) for feat in features]], dtype=np.float32)
    return float(regime_model.predict(X)[0])


# ── V5 FSM Exit Simulation ───────────────────────────────────────────────


def simulate_call_exit(closes, bids, asks, underlyings, entry_idx,
                       entry_premium, contracts, ticker, dte, expiry_date,
                       ml_stop_pct=None, regime_tighten_factor=1.0):
    """Run V5 FSM on remaining candles after entry."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "peak": 0}

    tcfg = get_ticker_config(ticker, use_per_ticker=True)
    if ml_stop_pct is not None:
        from dataclasses import replace
        clamped = max(15.0, min(55.0, ml_stop_pct))
        tcfg = replace(tcfg,
                       tight_stop_0dte_pct=clamped,
                       backstop_0dte_pct=min(clamped + 20, 65.0))
    fsm = ExitFSM(tcfg, settings=_V6_SETTINGS)

    # Apply regime tightening
    if regime_tighten_factor < 1.0:
        fsm.apply_regime_tighten(regime_tighten_factor)

    entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=entry_idx)

    underlying_0 = 0
    for i in range(entry_idx, min(entry_idx + 5, len(underlyings))):
        u = underlyings[i]
        if not np.isnan(u) and u > 0:
            underlying_0 = float(u)
            break

    state = TradeState(
        trade_id=1, ticker=ticker, option_type="call",
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=underlying_0,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts

    for idx in range(entry_idx + 1, len(closes)):
        prem = closes[idx]
        if np.isnan(prem) or prem <= 0:
            continue

        bid = float(bids[idx]) if idx < len(bids) and not np.isnan(bids[idx]) else prem
        ask = float(asks[idx]) if idx < len(asks) and not np.isnan(asks[idx]) else prem
        underlying = float(underlyings[idx]) if idx < len(underlyings) and not np.isnan(underlyings[idx]) else 0

        now = entry_ts + timedelta(minutes=(idx - entry_idx))
        mtc = max(0, (16 * 60) - (now.hour * 60 + now.minute))

        action = fsm.evaluate(state, prem, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=mtc, candle_data={})

        if action.should_exit:
            exit_p = bid if bid > 0 else prem
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (exit_p - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            peak = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (exit_p - entry_premium) * remaining * 100
            return {"pnl": round(pnl, 2), "reason": action.reason.value,
                    "hold": idx - entry_idx, "peak": round(peak, 1)}

    last_valid = entry_premium
    for i in range(len(closes) - 1, entry_idx, -1):
        if not np.isnan(closes[i]) and closes[i] > 0:
            last_valid = closes[i]
            break
    peak = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_valid - entry_premium) * remaining * 100
    return {"pnl": round(pnl, 2), "reason": "eod_data_end",
            "hold": len(closes) - entry_idx, "peak": round(peak, 1)}


def run_put_scalp(bars, entry_idx, entry_premium, contracts):
    """Simulate PUT scalp with fixed target/stop/maxhold."""
    peak_pct = 0.0
    for i in range(entry_idx + 1, min(entry_idx + 1 + PUT_MAX_HOLD, len(bars))):
        bar = bars[i]
        h, lo = bar["high"], bar["low"]
        if not h or h <= 0:
            continue

        pct_h = (h - entry_premium) / entry_premium * 100
        pct_l = (lo - entry_premium) / entry_premium * 100 if lo > 0 else 0
        if pct_h > peak_pct:
            peak_pct = pct_h

        if pct_h >= PUT_TARGET_PCT:
            exit_p = entry_premium * (1 + PUT_TARGET_PCT / 100) * 0.98
            return {"pnl": round((exit_p - entry_premium) * contracts * 100, 2),
                    "reason": "put_target", "hold": i - entry_idx, "peak": round(peak_pct, 1)}

        if pct_l <= -PUT_STOP_PCT:
            exit_p = entry_premium * (1 - PUT_STOP_PCT / 100) * 0.97
            return {"pnl": round((exit_p - entry_premium) * contracts * 100, 2),
                    "reason": "put_stop", "hold": i - entry_idx, "peak": round(peak_pct, 1)}

    last = bars[min(entry_idx + PUT_MAX_HOLD, len(bars) - 1)]
    exit_p = last["bid"] if last["bid"] > 0 else last["close"]
    if exit_p <= 0:
        exit_p = entry_premium
    return {"pnl": round((exit_p - entry_premium) * contracts * 100, 2),
            "reason": "put_maxhold", "hold": PUT_MAX_HOLD, "peak": round(peak_pct, 1)}


def load_put_data(conn, ticker, date_str):
    """Load ATM PUT OHLC for a ticker on a given day (0DTE)."""
    stock_open = conn.execute(
        "SELECT close FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp LIMIT 1",
        (ticker, date_str),
    ).fetchone()
    if not stock_open:
        return None, None

    atm = conn.execute("""
        SELECT DISTINCT strike FROM option_ohlc
        WHERE ticker=? AND right='PUT' AND expiration=? AND date(timestamp)=?
        ORDER BY ABS(strike - ?) LIMIT 1
    """, (ticker, date_str, date_str, stock_open[0])).fetchone()
    if not atm:
        return None, None

    rows = conn.execute("""
        SELECT time(timestamp), open, high, low, close, volume
        FROM option_ohlc
        WHERE ticker=? AND right='PUT' AND expiration=? AND date(timestamp)=? AND strike=?
        ORDER BY timestamp
    """, (ticker, date_str, date_str, atm[0])).fetchall()

    if not rows:
        return None, None

    bars = []
    for r in rows:
        c = r[4] or 0
        bars.append({
            "time": r[0][:5], "open": r[1] or 0, "high": r[2] or 0,
            "low": r[3] or 0, "close": c, "volume": r[5] or 0,
            "bid": c * 0.97, "ask": c * 1.03,
        })
    return atm[0], bars


# ── Main Backtest Runner ─────────────────────────────────────────────────


def run_backtest(conn, uw_conn, models, dates, fc: FeatureConfig) -> dict:
    """Run combined backtest with feature config.

    Returns summary dict with P&L, WR, PF, Sharpe, MaxDD, trades.
    """
    stock_cache = {}
    prev_cache = {}

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    all_trades = []
    regime_skipped_days = 0
    daily_pnls = {}

    # Regime timeline cache (date -> {minute -> RegimeState})
    regime_timelines = {}

    # Stats
    regime_blocks = 0
    extended_entries = 0
    conviction_adjustments = 0
    dynamic_put_adjustments = 0

    total_slots = CALL_MAX_CONCURRENT + PUT_MAX_CONCURRENT

    for day_idx, date_str in enumerate(dates):
        sod = portfolio
        day_pnl = 0.0
        day_trades = []
        day_spent = 0.0
        gfv_limit = sod * (1 - GFV_BUFFER_PCT / 100)
        day_realized = 0.0
        day_cb = False

        # ── Compute regime timeline for today ────────────────────────────
        regime_tl = {}
        if fc.regime_detector:
            if date_str not in regime_timelines:
                regime_timelines[date_str] = compute_regime_timeline(
                    conn, date_str,
                    adx_threshold=fc.adx_threshold,
                    hysteresis=fc.hysteresis_checks,
                    min_hold=fc.min_hold_bars,
                )
            regime_tl = regime_timelines[date_str]

        # ── Bear mode detection ──────────────────────────────────────────
        bear_mode = False
        spy_rows_morning = conn.execute(
            "SELECT close FROM stock_ohlc WHERE ticker='SPY' AND date(timestamp)=? ORDER BY timestamp LIMIT 61",
            (date_str,),
        ).fetchall()
        if spy_rows_morning and len(spy_rows_morning) > 30:
            spy_open = spy_rows_morning[0][0]
            spy_mid = spy_rows_morning[-1][0]
            if spy_open > 0 and spy_mid > 0:
                spy_change = (spy_mid / spy_open - 1) * 100
                if spy_change <= BEAR_MODE_THRESHOLD:
                    bear_mode = True

        day_put_max_concurrent = BEAR_PUT_MAX_CONCURRENT if bear_mode else PUT_MAX_CONCURRENT
        day_put_tickers = BEAR_PUT_TICKERS if bear_mode else PUT_TICKERS
        day_skip_calls = bear_mode and BEAR_SKIP_CALLS

        # Dynamic PUT slots (Feature 5)
        if fc.dynamic_puts and regime_tl:
            # Use regime at minute 60 (10:30 AM) for PUT slot allocation
            regime_at_60 = regime_tl.get(60, RegimeState.CHOPPY)
            slots = get_direction_slots(regime_at_60, total_slots, dynamic_puts_enabled=True)
            day_put_max_concurrent = slots["put"]
            if day_put_max_concurrent != PUT_MAX_CONCURRENT:
                dynamic_put_adjustments += 1

        # ── ML Regime daily filter (existing, not new feature) ───────────
        regime_skip = False
        if "regime" in models and REGIME_THRESHOLD > 0:
            regime_score = compute_regime_score(
                models["regime"], "SPY", date_str, conn, uw_conn,
                stock_cache, prev_cache,
            )
            if regime_score < REGIME_THRESHOLD:
                regime_skip = True
                regime_skipped_days += 1

        # ── Determine scan window ────────────────────────────────────────
        scan_end = CALL_SCAN_END  # default: 90 min (11:00)
        if fc.extended_scan and not regime_skip:
            # Extended scan only when regime detector is also enabled
            if fc.regime_detector:
                scan_end = fc.late_scan_end  # 330 min (3:00 PM)
            else:
                scan_end = CALL_SCAN_END  # fallback to baseline without regime data

        # ── CALL TRACK ───────────────────────────────────────────────────

        if not day_skip_calls and not regime_skip:
            call_open_tickers = []
            call_open_dirs = []
            last_entry_minute = -999
            day_losses = 0
            day_pattern_threshold = PATTERN_THRESHOLD

            call_tickers = [t for t in CALL_TICKERS if t not in CALL_EXCLUDED]

            for ticker in call_tickers:
                if ticker in call_open_tickers:
                    continue
                if day_cb:
                    break
                if len(call_open_tickers) >= CALL_MAX_CONCURRENT:
                    break
                if ticker in INDEX_TICKERS:
                    index_open = sum(1 for t in call_open_tickers if t in INDEX_TICKERS)
                    if index_open >= CALL_MAX_INDEX_CONCURRENT:
                        continue

                # Load day data
                atm = conn.execute("""
                    SELECT oohlc.strike FROM option_ohlc oohlc
                    JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                        AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                    WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL'
                        AND og.underlying_price > 0
                    GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
                """, (ticker, date_str)).fetchone()
                if not atm:
                    continue
                strike = atm[0]

                rows = conn.execute("""
                    SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                           COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                           COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                           oohlc.volume, oohlc.expiration,
                           COALESCE(og.vega, 0),
                           COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0)
                    FROM option_ohlc oohlc
                    LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                        AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
                    LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                        AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                    WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
                    ORDER BY oohlc.timestamp
                """, (ticker, date_str, strike)).fetchall()

                if len(rows) < 30:
                    continue

                closes = np.array([float(r[0]) if r[0] else np.nan for r in rows])
                underlyings = np.array([float(r[1]) if r[1] else np.nan for r in rows])
                ivs = np.array([float(r[2]) if r[2] else np.nan for r in rows])
                bids_arr = np.array([float(r[3]) if r[3] else 0 for r in rows])
                asks_arr = np.array([float(r[4]) if r[4] else 0 for r in rows])
                deltas_arr = np.array([float(r[5]) if r[5] else np.nan for r in rows])
                thetas_arr = np.array([float(r[6]) if r[6] else np.nan for r in rows])
                volumes_arr = np.array([float(r[7]) if r[7] else 0 for r in rows])
                expiry_date = rows[0][8] if rows else date_str
                vegas_arr = np.array([float(r[9]) if r[9] else np.nan for r in rows])
                bid_sizes = np.array([float(r[10]) if r[10] else 0 for r in rows])
                ask_sizes = np.array([float(r[11]) if r[11] else 0 for r in rows])

                stock_rows = conn.execute(
                    "SELECT close, high, low FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp",
                    (ticker, date_str),
                ).fetchall()
                stock_closes = np.array([float(r[0]) for r in stock_rows]) if stock_rows else np.array([])
                stock_highs = np.array([float(r[1]) for r in stock_rows]) if stock_rows else np.array([])
                stock_lows = np.array([float(r[2]) for r in stock_rows]) if stock_rows else np.array([])

                opening_price = 0
                for c in closes[:5]:
                    if not np.isnan(c) and c > 0:
                        opening_price = c
                        break
                if opening_price <= 0:
                    continue

                try:
                    exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    dte = max(0, (exp_dt - day_dt).days)
                except (ValueError, TypeError):
                    dte = 0

                entered = False
                effective_scan_end = min(scan_end, len(closes))
                p_features = models["pattern_meta"]["features"]

                for minute in range(CALL_SCAN_START, effective_scan_end):
                    if entered or day_cb:
                        break
                    if len(call_open_tickers) >= CALL_MAX_CONCURRENT:
                        break

                    if minute - last_entry_minute < CALL_MIN_ENTRY_SPACING:
                        continue

                    # ── Feature 1: Regime direction gating ────────────
                    if fc.regime_detector and regime_tl:
                        # Snap to nearest 5-min bar
                        bar_min = (minute // 5) * 5
                        regime_now = regime_tl.get(bar_min, RegimeState.CHOPPY)

                        # Feature 2: Extended scan uses direction gating
                        if fc.extended_scan and minute > CALL_SCAN_END:
                            allowed = get_allowed_directions(
                                minute, regime_now, extended_scan_enabled=True,
                            )
                            if "call" not in allowed:
                                regime_blocks += 1
                                continue
                        else:
                            # Morning session: block CALLs in bearish regime
                            if regime_now == RegimeState.BEARISH:
                                regime_blocks += 1
                                continue

                    # Afternoon danger gate (baseline behavior for extended scan)
                    if ENABLE_AFTERNOON_DANGER and AFTERNOON_DANGER_START <= minute <= AFTERNOON_DANGER_END:
                        if not fc.extended_scan:
                            continue
                        # With extended scan, afternoon is regime-gated (above), not blanket blocked

                    if ENABLE_HARD_CUTOFF and minute >= HARD_CUTOFF_MIN:
                        continue

                    # Pattern model
                    feat = compute_pattern_features(
                        closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                        bids_arr, asks_arr, minute, opening_price,
                    )
                    if feat is None:
                        continue

                    X_pat = np.array([[feat.get(f, 0) for f in p_features]], dtype=np.float32)
                    pattern_conf = float(models["pattern"].predict(X_pat)[0])

                    if pattern_conf < day_pattern_threshold:
                        continue

                    # Entry timing gate
                    et_feat = None
                    if "entry" in models:
                        ef = models["entry_features"]
                        et_feat = compute_entry_timing_features(
                            closes, volumes_arr, bids_arr, asks_arr, bid_sizes,
                            ask_sizes, ivs, deltas_arr, thetas_arr, vegas_arr,
                            underlyings, stock_closes, stock_highs, stock_lows,
                            minute, ef,
                        )
                        if et_feat is not None:
                            X_et = np.array([[et_feat.get(f, 0) for f in ef]], dtype=np.float32)
                            entry_conf = float(models["entry"].predict(X_et)[0])
                            if entry_conf < ENTRY_THRESHOLD:
                                continue

                    # Entry gates
                    entry_premium = float(asks_arr[minute]) if asks_arr[minute] > 0 else float(closes[minute])
                    if entry_premium <= 0 or np.isnan(entry_premium):
                        continue
                    if entry_premium < CALL_PREMIUM_FLOOR or entry_premium > CALL_PREMIUM_CAP:
                        continue

                    bid_val = float(bids_arr[minute]) if bids_arr[minute] > 0 else 0
                    if bid_val > 0 and entry_premium > 0:
                        spread = (entry_premium - bid_val) / entry_premium * 100
                        if spread > CALL_SPREAD_GATE_PCT:
                            continue

                    same_dir = sum(1 for d in call_open_dirs if d == "call")
                    if same_dir >= CALL_MAX_SAME_DIRECTION:
                        continue

                    # ── Feature 4: Conviction sizing ──────────────────
                    conv_mult = 1.0
                    if fc.conviction_sizing and fc.regime_detector and regime_tl:
                        bar_min = (minute // 5) * 5
                        regime_now = regime_tl.get(bar_min, RegimeState.CHOPPY)
                        conv_mult = compute_conviction_multiplier(
                            pattern_conf, regime_now, "call", minute,
                            conviction_enabled=True,
                        )
                        if conv_mult < 1.0:
                            conviction_adjustments += 1

                    # Position sizing
                    deployable = portfolio * MAX_RISK_PCT
                    per_slot = deployable / total_slots
                    position_cap = portfolio * CALL_MAX_POSITION_PCT
                    cost_per = entry_premium * 100

                    gfv_remaining = gfv_limit - day_spent
                    if gfv_remaining < cost_per:
                        continue

                    scaled = per_slot * 0.85 * conv_mult
                    raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
                    cap_ct = int(position_cap / cost_per) if cost_per > 0 else 1
                    gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                    dollar_ct = int(CALL_MAX_POSITION_DOLLARS / cost_per) if cost_per > 0 else 1
                    contracts = max(1, min(raw_ct, cap_ct, gfv_ct, dollar_ct, CALL_MAX_CONTRACTS))

                    day_spent += contracts * cost_per

                    # DCA check
                    effective_entry = entry_premium
                    effective_contracts = contracts
                    dca_window_start = minute + 8
                    dca_window_end = min(minute + 20, len(closes))
                    for dca_idx in range(dca_window_start, dca_window_end):
                        dca_prem = closes[dca_idx]
                        if np.isnan(dca_prem) or dca_prem <= 0:
                            continue
                        dip_pct = (entry_premium - dca_prem) / entry_premium * 100
                        if 15.0 <= dip_pct <= 35.0:
                            und_now = underlyings[dca_idx] if dca_idx < len(underlyings) and not np.isnan(underlyings[dca_idx]) else 0
                            und_entry = underlyings[minute] if minute < len(underlyings) and not np.isnan(underlyings[minute]) else 0
                            if und_entry > 0 and und_now > 0 and abs(und_now / und_entry - 1) * 100 > 0.5:
                                continue
                            dca_cost_per = dca_prem * 100
                            dca_ct = max(1, min(contracts, int(CALL_MAX_POSITION_DOLLARS / dca_cost_per), CALL_MAX_CONTRACTS - contracts))
                            dca_add_cost = dca_ct * dca_cost_per
                            if day_spent + dca_add_cost > gfv_limit:
                                continue
                            day_spent += dca_add_cost
                            effective_contracts = contracts + dca_ct
                            effective_entry = (entry_premium * contracts + dca_prem * dca_ct) / effective_contracts
                            break

                    # ── Feature 3: Regime stop tightening ─────────────
                    regime_tighten = 1.0
                    if fc.regime_stop_tighten and fc.regime_detector and regime_tl:
                        bar_min = (minute // 5) * 5
                        regime_now = regime_tl.get(bar_min, RegimeState.CHOPPY)
                        if regime_now == RegimeState.BEARISH:
                            regime_tighten = fc.counter_trend_tighten
                        elif regime_now == RegimeState.CHOPPY:
                            regime_tighten = fc.choppy_tighten

                    # ML stop calibration
                    ml_stop = None
                    if "stop" in models and et_feat is not None:
                        stop_features = models["stop"].feature_name()
                        X_stop = np.array([[et_feat.get(f, 0) for f in stop_features]], dtype=np.float32)
                        ml_stop = float(models["stop"].predict(X_stop)[0])

                    result = simulate_call_exit(
                        closes, bids_arr, asks_arr, underlyings,
                        minute, effective_entry, effective_contracts, ticker,
                        dte, expiry_date, ml_stop_pct=ml_stop,
                        regime_tighten_factor=regime_tighten,
                    )

                    trade_pnl = result["pnl"]
                    portfolio += trade_pnl
                    day_pnl += trade_pnl
                    day_realized += trade_pnl

                    call_open_tickers.append(ticker)
                    call_open_dirs.append("call")
                    entered = True
                    last_entry_minute = minute

                    if minute > CALL_SCAN_END:
                        extended_entries += 1

                    if trade_pnl < 0:
                        day_losses += 1
                        if day_losses >= CALL_LOSS_COUNT_THRESHOLD:
                            day_pattern_threshold = max(day_pattern_threshold, CALL_BAD_DAY_THRESHOLD)

                    if day_realized < 0 and abs(day_realized) > sod * DAILY_LOSS_CB_PCT / 100:
                        day_cb = True

                    day_trades.append({
                        "day": date_str, "type": "CALL", "ticker": ticker,
                        "entry": round(entry_premium, 2), "contracts": effective_contracts,
                        "pnl": trade_pnl, "reason": result["reason"],
                        "hold": result["hold"], "peak": result["peak"],
                        "conf": round(pattern_conf, 3),
                        "minute": minute,
                    })

        # ── PUT TRACK ────────────────────────────────────────────────────

        if not day_cb:
            put_count = 0
            for ticker in day_put_tickers:
                if put_count >= day_put_max_concurrent:
                    break

                strike, bars = load_put_data(conn, ticker, date_str)
                if not bars or len(bars) < 10:
                    continue

                time_to_idx = {b["time"]: i for i, b in enumerate(bars)}

                for slot in PUT_SLOTS:
                    if put_count >= day_put_max_concurrent:
                        break
                    if slot not in time_to_idx:
                        continue

                    entry_idx = time_to_idx[slot]
                    if len(bars) - entry_idx - 1 < 5:
                        continue

                    bar = bars[entry_idx]
                    entry_premium = bar["ask"] if bar["ask"] > 0 else bar["close"]
                    if not entry_premium or entry_premium <= 0:
                        continue
                    if entry_premium < PUT_PREMIUM_FLOOR or entry_premium > PUT_PREMIUM_CAP:
                        continue

                    cost_per = entry_premium * 100
                    gfv_remaining = gfv_limit - day_spent
                    if gfv_remaining < cost_per:
                        continue

                    deployable = portfolio * MAX_RISK_PCT
                    per_slot = deployable / total_slots
                    scaled = per_slot * 0.85
                    raw_ct = int(scaled / cost_per)
                    cap_ct = int(portfolio * PUT_MAX_POSITION_PCT / cost_per)
                    dollar_ct = int(PUT_MAX_POSITION_DOLLARS / cost_per)
                    gfv_ct = int(gfv_remaining / cost_per)
                    contracts = max(1, min(raw_ct, cap_ct, dollar_ct, gfv_ct, PUT_MAX_CONTRACTS))

                    day_spent += contracts * cost_per
                    put_count += 1

                    result = run_put_scalp(bars, entry_idx, entry_premium, contracts)
                    trade_pnl = result["pnl"]
                    portfolio += trade_pnl
                    day_pnl += trade_pnl
                    day_realized += trade_pnl

                    if day_realized < 0 and abs(day_realized) > sod * DAILY_LOSS_CB_PCT / 100:
                        day_cb = True

                    day_trades.append({
                        "day": date_str, "type": "PUT", "ticker": ticker,
                        "entry": round(entry_premium, 2), "contracts": contracts,
                        "pnl": trade_pnl, "reason": result["reason"],
                        "hold": result["hold"], "peak": result["peak"],
                        "slot": slot,
                    })

                    break

        if day_trades:
            all_trades.extend(day_trades)
            if date_str not in daily_pnls:
                daily_pnls[date_str] = 0
            daily_pnls[date_str] += day_pnl

        if portfolio > peak_portfolio:
            peak_portfolio = portfolio
        dd = (peak_portfolio - portfolio) / peak_portfolio * 100 if peak_portfolio > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # ── Compute results ──────────────────────────────────────────────────

    n = len(all_trades)
    if n == 0:
        return {"trades": 0, "pnl": 0, "wr": 0, "pf": 0, "sharpe": 0, "max_dd": 0,
                "portfolio": PORTFOLIO_START, "label": fc.label(),
                "regime_blocks": regime_blocks, "extended_entries": extended_entries,
                "conviction_adjustments": conviction_adjustments,
                "dynamic_put_adjustments": dynamic_put_adjustments,
                "regime_skipped_days": regime_skipped_days}

    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in all_trades)
    wr = len(wins) / n * 100
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else 999

    daily_returns = list(daily_pnls.values())
    sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0

    call_trades = [t for t in all_trades if t["type"] == "CALL"]
    put_trades = [t for t in all_trades if t["type"] == "PUT"]

    # Per-ticker breakdown
    per_ticker = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in all_trades:
        per_ticker[t["ticker"]]["n"] += 1
        per_ticker[t["ticker"]]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            per_ticker[t["ticker"]]["wins"] += 1

    # Exit reasons
    reasons = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in all_trades:
        reasons[t["reason"]]["n"] += 1
        reasons[t["reason"]]["pnl"] += t["pnl"]

    return {
        "trades": n,
        "calls": len(call_trades),
        "puts": len(put_trades),
        "pnl": round(total_pnl, 0),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 1),
        "portfolio": round(portfolio, 0),
        "return_pct": round(total_pnl / PORTFOLIO_START * 100, 1),
        "label": fc.label(),
        "regime_blocks": regime_blocks,
        "extended_entries": extended_entries,
        "conviction_adjustments": conviction_adjustments,
        "dynamic_put_adjustments": dynamic_put_adjustments,
        "regime_skipped_days": regime_skipped_days,
        "per_ticker": dict(per_ticker),
        "reasons": dict(reasons),
        "all_trades": all_trades,
    }


def print_result(r: dict, verbose: bool = False):
    """Print one-line result summary."""
    label = r["label"]
    print(f"  {label:<45} | {r['trades']:>4} ({r['calls']}C+{r['puts']}P) | "
          f"{r['wr']:>5.1f}% | ${r['pnl']:>+10,.0f} | "
          f"PF={r['pf']:>5.2f} | Sharpe={r['sharpe']:>5.2f} | DD={r['max_dd']:>4.1f}%",
          flush=True)
    if verbose:
        if r["regime_blocks"] > 0:
            print(f"    Regime blocks: {r['regime_blocks']}", flush=True)
        if r["extended_entries"] > 0:
            print(f"    Extended entries: {r['extended_entries']}", flush=True)
        if r["conviction_adjustments"] > 0:
            print(f"    Conviction adjustments: {r['conviction_adjustments']}", flush=True)
        if r["dynamic_put_adjustments"] > 0:
            print(f"    Dynamic PUT adjustments: {r['dynamic_put_adjustments']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Regime Feature Sweep Backtest")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--individual", action="store_true", help="Individual feature tests only")
    parser.add_argument("--combos", action="store_true", help="Combo tests only")
    parser.add_argument("--params", action="store_true", help="Parameter sweep for best features")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_individual = not args.combos or args.individual
    run_combos = not args.individual or args.combos
    if args.baseline_only:
        run_individual = False
        run_combos = False

    print("=" * 110)
    print("REGIME FEATURE SWEEP — Gold Standard + 5 New Features")
    print("=" * 110)

    # Load models
    print("\nLoading ML models...", flush=True)
    models = load_models()
    print(f"  Pattern: AUC={models['pattern_meta']['auc']:.4f}")
    if "entry" in models:
        print(f"  Entry timing: {len(models['entry_features'])} features")
    if "regime" in models:
        print(f"  Regime classifier: loaded")

    # Connect DB
    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA cache_size = -200000")

    uw_conn = None
    if "regime" in models and Path(UW_DB).exists():
        uw_conn = sqlite3.connect(UW_DB)

    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date(timestamp) FROM stock_ohlc WHERE ticker='SPY' ORDER BY 1"
    ).fetchall()]
    dates = all_dates[-args.days:]

    print(f"\n  Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Portfolio: ${PORTFOLIO_START:,}")

    results = []

    # ── 1. BASELINE ──────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("BASELINE (Gold Standard Combined)")
    print(f"{'='*110}")
    print(f"\n  {'Config':<45} | {'Trades':>12} | {'WR':>5} | {'P&L':>11} | "
          f"{'PF':>7} | {'Sharpe':>8} | {'DD':>6}", flush=True)
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*5}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}", flush=True)

    t0 = time.time()
    baseline = run_backtest(conn, uw_conn, models, dates, FeatureConfig())
    print_result(baseline, args.verbose)
    results.append(baseline)
    print(f"\n  Baseline runtime: {time.time() - t0:.0f}s", flush=True)

    if args.baseline_only:
        conn.close()
        return

    # ── 2. INDIVIDUAL FEATURE TESTS ─────────────────────────────────────
    if run_individual:
        print(f"\n{'='*110}")
        print("INDIVIDUAL FEATURE TESTS (one feature at a time)")
        print(f"{'='*110}")
        print(f"\n  {'Config':<45} | {'Trades':>12} | {'WR':>5} | {'P&L':>11} | "
              f"{'PF':>7} | {'Sharpe':>8} | {'DD':>6}", flush=True)
        print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*5}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}", flush=True)

        individual_configs = [
            # Feature 1: Regime detector alone
            FeatureConfig(regime_detector=True),
            FeatureConfig(regime_detector=True, adx_threshold=15.0),
            FeatureConfig(regime_detector=True, adx_threshold=25.0),

            # Feature 2: Extended scan (needs regime)
            FeatureConfig(regime_detector=True, extended_scan=True),

            # Feature 3: Regime stop tighten (needs regime)
            FeatureConfig(regime_detector=True, regime_stop_tighten=True),
            FeatureConfig(regime_detector=True, regime_stop_tighten=True,
                         counter_trend_tighten=0.50, choppy_tighten=0.70),
            FeatureConfig(regime_detector=True, regime_stop_tighten=True,
                         counter_trend_tighten=0.70, choppy_tighten=0.90),

            # Feature 4: Conviction sizing (needs regime)
            FeatureConfig(regime_detector=True, conviction_sizing=True),

            # Feature 5: Dynamic PUTs (needs regime)
            FeatureConfig(regime_detector=True, dynamic_puts=True),
            FeatureConfig(regime_detector=True, dynamic_puts=True, bearish_put_slots=4),
        ]

        for fc in individual_configs:
            r = run_backtest(conn, uw_conn, models, dates, fc)
            print_result(r, args.verbose)
            results.append(r)

    # ── 3. COMBO TESTS ──────────────────────────────────────────────────
    if run_combos:
        print(f"\n{'='*110}")
        print("COMBO TESTS (best individual features combined)")
        print(f"{'='*110}")
        print(f"\n  {'Config':<45} | {'Trades':>12} | {'WR':>5} | {'P&L':>11} | "
              f"{'PF':>7} | {'Sharpe':>8} | {'DD':>6}", flush=True)
        print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*5}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}", flush=True)

        combo_configs = [
            # Regime + Extended
            FeatureConfig(regime_detector=True, extended_scan=True),

            # Regime + Stop Tighten
            FeatureConfig(regime_detector=True, regime_stop_tighten=True),

            # Regime + Conviction
            FeatureConfig(regime_detector=True, conviction_sizing=True),

            # Regime + Extended + Stop Tighten
            FeatureConfig(regime_detector=True, extended_scan=True, regime_stop_tighten=True),

            # Regime + Extended + Conviction
            FeatureConfig(regime_detector=True, extended_scan=True, conviction_sizing=True),

            # Regime + Stop Tighten + Conviction
            FeatureConfig(regime_detector=True, regime_stop_tighten=True, conviction_sizing=True),

            # Regime + Extended + Stop Tighten + Conviction
            FeatureConfig(regime_detector=True, extended_scan=True,
                         regime_stop_tighten=True, conviction_sizing=True),

            # All 5 features
            FeatureConfig(regime_detector=True, extended_scan=True,
                         regime_stop_tighten=True, conviction_sizing=True,
                         dynamic_puts=True),

            # All 5 with tuned params
            FeatureConfig(regime_detector=True, adx_threshold=15.0,
                         extended_scan=True, regime_stop_tighten=True,
                         counter_trend_tighten=0.50, choppy_tighten=0.70,
                         conviction_sizing=True, dynamic_puts=True,
                         bearish_put_slots=4),

            FeatureConfig(regime_detector=True, adx_threshold=25.0,
                         extended_scan=True, regime_stop_tighten=True,
                         counter_trend_tighten=0.70, choppy_tighten=0.90,
                         conviction_sizing=True, dynamic_puts=True,
                         bearish_put_slots=6),
        ]

        for fc in combo_configs:
            r = run_backtest(conn, uw_conn, models, dates, fc)
            print_result(r, args.verbose)
            results.append(r)

    # ── 4. PARAMETER SWEEP (if requested) ────────────────────────────────
    if args.params:
        print(f"\n{'='*110}")
        print("PARAMETER SWEEP — ADX Threshold × Stop Tighten × Conviction")
        print(f"{'='*110}")
        print(f"\n  {'Config':<45} | {'Trades':>12} | {'WR':>5} | {'P&L':>11} | "
              f"{'PF':>7} | {'Sharpe':>8} | {'DD':>6}", flush=True)
        print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*5}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}", flush=True)

        for adx in [15.0, 20.0, 25.0, 30.0]:
            for ct_tighten in [0.50, 0.60, 0.70]:
                for ch_tighten in [0.70, 0.80, 0.90]:
                    fc = FeatureConfig(
                        regime_detector=True, adx_threshold=adx,
                        extended_scan=True, regime_stop_tighten=True,
                        counter_trend_tighten=ct_tighten, choppy_tighten=ch_tighten,
                        conviction_sizing=True, dynamic_puts=True,
                    )
                    r = run_backtest(conn, uw_conn, models, dates, fc)
                    print_result(r)
                    results.append(r)

    conn.close()
    if uw_conn:
        uw_conn.close()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("SWEEP SUMMARY — Ranked by Sharpe Ratio")
    print(f"{'='*110}")
    print(f"\n  {'Rank':>4} {'Config':<45} | {'Trades':>6} | {'WR':>5} | {'P&L':>11} | "
          f"{'PF':>7} | {'Sharpe':>8} | {'DD':>6}", flush=True)
    print(f"  {'-'*4} {'-'*45}-+-{'-'*6}-+-{'-'*5}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}", flush=True)

    sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    for i, r in enumerate(sorted_results[:15], 1):
        label = r["label"]
        marker = " *" if r["label"] == "BASELINE" else ""
        print(f"  {i:>4} {label:<45} | {r['trades']:>6} | {r['wr']:>5.1f}% | "
              f"${r['pnl']:>+10,.0f} | PF={r['pf']:>5.2f} | "
              f"Sharpe={r['sharpe']:>5.2f} | DD={r['max_dd']:>4.1f}%{marker}", flush=True)

    # Show best config details
    best = sorted_results[0]
    print(f"\n  BEST CONFIG: {best['label']}")
    print(f"    Trades: {best['trades']} ({best['calls']}C + {best['puts']}P)")
    print(f"    P&L: ${best['pnl']:+,.0f} ({best['return_pct']:+.1f}%)")
    print(f"    Win Rate: {best['wr']:.1f}%")
    print(f"    Profit Factor: {best['pf']:.2f}")
    print(f"    Sharpe: {best['sharpe']:.2f}")
    print(f"    Max DD: {best['max_dd']:.1f}%")

    if best.get("per_ticker"):
        print(f"\n    Per Ticker:")
        for tk, data in sorted(best["per_ticker"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = data["wins"] / data["n"] * 100 if data["n"] else 0
            print(f"      {tk:>6}: {data['n']:>3} trades, {wr:>4.0f}% WR, ${data['pnl']:>+8,.0f}")

    if best.get("reasons"):
        print(f"\n    Exit Reasons:")
        for reason, data in sorted(best["reasons"].items(), key=lambda x: -x[1]["n"]):
            print(f"      {reason:>20}: {data['n']:>4} trades, ${data['pnl']:>+8,.0f}")

    # Compare best to baseline
    if len(results) > 1:
        bl = results[0]
        print(f"\n  BEST vs BASELINE:")
        print(f"    P&L delta:    ${best['pnl'] - bl['pnl']:>+,.0f}")
        print(f"    WR delta:     {best['wr'] - bl['wr']:>+.1f}%")
        print(f"    PF delta:     {best['pf'] - bl['pf']:>+.2f}")
        print(f"    Sharpe delta: {best['sharpe'] - bl['sharpe']:>+.2f}")
        print(f"    DD delta:     {best['max_dd'] - bl['max_dd']:>+.1f}%")


if __name__ == "__main__":
    main()
