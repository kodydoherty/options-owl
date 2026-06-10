"""Simpsons AMD+iFVG Strategy Backtest — Python port of the n8n workflow.

Faithfully replicates the Simpsons Side A logic:
  1. Session Level Scanner — detects sweeps of PDH/PDL/PWH/PWL/PMH/PML + open sweeps
  2. AMD state machine (Accumulation → Manipulation → Distribution)
  3. iFVG detection (Inverse Fair Value Gap — proper ICT definition)
  4. Signal Scorer — grades signals A+ through F
  5. v10 surgical filter — 14 data-driven rules (85.9% WR in Simpsons backtest)
  6. Option-leg P&L simulation using ThetaData option bars

Data source: ThetaData DB (1-min stock bars resampled to 5-min, plus option bars).

Usage:
    python scripts/backtest_simpsons.py                      # last 60 trading days
    python scripts/backtest_simpsons.py --days 90            # last 90 days
    python scripts/backtest_simpsons.py --no-v10-filter      # disable v10 filter
    python scripts/backtest_simpsons.py --ticker TSLA        # single ticker
    python scripts/backtest_simpsons.py --stage1-only        # include Stage 1 Watch signals
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")

# ── Portfolio simulation constants ──────────────────────────────────────
PORTFOLIO_START = 23_000
MAX_CONCURRENT = 4
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
MAX_POSITION_DOLLARS = 3500
MAX_CONTRACTS = 20
PREMIUM_CAP = 6.0

# V6 settings (matches production)
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
    V6_PREMIUM_CAP=PREMIUM_CAP,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=40.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)

# ── Simpsons ticker config (from Generate Ticker List v4) ──────────────
TICKER_CONFIG = {
    "AMD":   {"tier": 1, "minATR": 1.5,  "sweepMult": 0.42, "type": "HIGH_VOL"},
    "TSLA":  {"tier": 1, "minATR": 2.0,  "sweepMult": 0.50, "type": "HIGHEST_VOL"},
    "NVDA":  {"tier": 2, "minATR": 0.6,  "sweepMult": 0.32, "type": "MED_VOL"},
    "META":  {"tier": 1, "minATR": 1.0,  "sweepMult": 0.38, "type": "HIGH_VOL"},
    "MSFT":  {"tier": 1, "minATR": 0.7,  "sweepMult": 0.30, "type": "LARGE_CAP"},
    "AAPL":  {"tier": 1, "minATR": 0.4,  "sweepMult": 0.28, "type": "LARGE_CAP"},
    "GOOGL": {"tier": 1, "minATR": 0.5,  "sweepMult": 0.25, "type": "LARGE_CAP"},
    "AMZN":  {"tier": 1, "minATR": 0.4,  "sweepMult": 0.28, "type": "LARGE_CAP"},
    "AVGO":  {"tier": 2, "minATR": 0.8,  "sweepMult": 0.35, "type": "MED_VOL"},
    "SPY":   {"tier": 1, "minATR": 0.4,  "sweepMult": 0.20, "type": "ETF"},
    "QQQ":   {"tier": 1, "minATR": 0.6,  "sweepMult": 0.22, "type": "ETF"},
}

TICKERS = list(TICKER_CONFIG.keys())


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_trading_days(db_path: str, n_days: int) -> list[str]:
    """Get the last N trading days from stock data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT DATE(timestamp) as d FROM stock_ohlc
        WHERE ticker = 'SPY' AND TIME(timestamp) >= '09:30' AND TIME(timestamp) <= '16:00'
        ORDER BY d DESC LIMIT ?
    """, (n_days,))
    days = sorted([r[0] for r in cur.fetchall()])
    conn.close()
    return days


def load_5min_candles(db_path: str, ticker: str, date_str: str) -> pd.DataFrame:
    """Load 1-min bars and resample to 5-min candles for a single day."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume
        FROM stock_ohlc
        WHERE ticker = ? AND DATE(timestamp) = ?
          AND TIME(timestamp) >= '09:30' AND TIME(timestamp) <= '16:00'
        ORDER BY timestamp
    """, conn, params=(ticker, date_str))
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df.set_index("timestamp", inplace=True)

    # Resample to 5-min bars
    resampled = df.resample("5min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    return resampled.reset_index()


def load_daily_candles(db_path: str, ticker: str, before_date: str, n_days: int = 30) -> pd.DataFrame:
    """Load daily OHLCV bars for prior days (for PDH/PDL/PWH/PWL/PMH/PML)."""
    conn = sqlite3.connect(db_path)
    # Build daily bars from 1-min data by grouping by date
    df = pd.read_sql_query("""
        SELECT DATE(timestamp) as date,
               MIN(CASE WHEN TIME(timestamp) = '09:30' THEN open END) as open,
               MAX(high) as high, MIN(low) as low,
               -- last close of the day
               close, volume
        FROM (
            SELECT DATE(timestamp) as dt, open, high, low, close, volume, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY DATE(timestamp) ORDER BY timestamp DESC) as rn
            FROM stock_ohlc
            WHERE ticker = ? AND DATE(timestamp) < ? AND DATE(timestamp) >= DATE(?, '-60 days')
              AND TIME(timestamp) >= '09:30' AND TIME(timestamp) <= '16:00'
        )
        WHERE rn = 1
        GROUP BY dt
        ORDER BY dt
    """, conn, params=(ticker, before_date, before_date))
    conn.close()

    if df.empty:
        # Fallback: simple aggregation
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT DATE(timestamp) as date,
                   MIN(open) as open, MAX(high) as high, MIN(low) as low,
                   MAX(close) as close, SUM(volume) as volume
            FROM stock_ohlc
            WHERE ticker = ? AND DATE(timestamp) < ? AND DATE(timestamp) >= DATE(?, '-60 days')
              AND TIME(timestamp) >= '09:30' AND TIME(timestamp) <= '16:00'
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp)
        """, conn, params=(ticker, before_date, before_date))
        conn.close()

    return df


def load_option_bars(db_path: str, ticker: str, date_str: str,
                     strike: float, direction: str, expiry: str) -> pd.DataFrame:
    """Load 1-min option bars for grading."""
    right = "CALL" if direction == "CALL" else "PUT"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume
        FROM option_ohlc
        WHERE ticker = ? AND DATE(timestamp) = ? AND strike = ? AND right = ?
          AND expiration = ?
          AND TIME(timestamp) >= '09:30' AND TIME(timestamp) <= '16:00'
        ORDER BY timestamp
    """, conn, params=(ticker, date_str, strike, right, expiry))
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════════════════
# Technical helpers (ported from Simpsons JS)
# ═══════════════════════════════════════════════════════════════════════

def calc_ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def calc_atr(candles: pd.DataFrame, period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles.iloc[i]["high"]
        l = candles.iloc[i]["low"]
        pc = candles.iloc[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def get_daily_levels(daily: pd.DataFrame, today_str: str) -> dict:
    """Compute PDH/PDL/PWH/PWL/PMH/PML from daily candles."""
    prior = daily[daily["date"] < today_str]
    if prior.empty:
        return {}

    levels = {}

    # PDH/PDL — yesterday
    yest = prior.iloc[-1]
    levels["pdh"] = float(yest["high"])
    levels["pdl"] = float(yest["low"])

    # PWH/PWL — previous calendar week
    today = datetime.strptime(today_str, "%Y-%m-%d").date()
    dow = today.weekday()  # 0=Mon
    this_monday = today - timedelta(days=dow)
    prev_week_start = this_monday - timedelta(days=7)
    prev_week_bars = prior[
        (prior["date"] >= str(prev_week_start)) & (prior["date"] < str(this_monday))
    ]
    if not prev_week_bars.empty:
        levels["pwh"] = float(prev_week_bars["high"].max())
        levels["pwl"] = float(prev_week_bars["low"].min())

    # PMH/PML — previous calendar month
    prev_month = (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    prev_month_bars = prior[prior["date"].str.startswith(prev_month)]
    if not prev_month_bars.empty:
        levels["pmh"] = float(prev_month_bars["high"].max())
        levels["pml"] = float(prev_month_bars["low"].min())

    return levels


def classify_ema_trend(closes: list[float]) -> str:
    if len(closes) < 53:
        return "NEUTRAL"
    ema9 = calc_ema(closes, 9)
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    prev = closes[:-3]
    pema9 = calc_ema(prev, 9)
    pema20 = calc_ema(prev, 20)
    pema50 = calc_ema(prev, 50)
    if not all([ema9, ema20, ema50, pema9, pema20, pema50]):
        return "NEUTRAL"
    if ema9 > ema20 > ema50 and ema9 > pema9 and ema20 > pema20:
        return "BULL"
    if ema9 < ema20 < ema50 and ema9 < pema9 and ema20 < pema20:
        return "BEAR"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# AMD + iFVG Scanner (core Simpsons logic)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    ticker: str
    date: str
    fire_time: str  # HH:MM ET
    direction: str  # CALL / PUT
    stage: str  # STAGE1_NEW / STAGE2
    sweep_level_name: str
    sweep_level_val: float
    amd_bias: str
    score: int
    grade: str
    current_price: float
    atr5m: float
    ifvg_detected: bool
    session_label: str
    v10_filter_reason: str | None
    # For P&L grading
    entry_price: float | None = None
    peak_pct: float | None = None
    pnl_dollars: float | None = None
    option_entry: float | None = None
    option_peak: float | None = None
    option_exit: float | None = None
    # V5 FSM results
    fsm_pnl: float | None = None
    fsm_reason: str | None = None
    fsm_hold_min: int | None = None
    fsm_peak_gain: float | None = None
    fsm_contracts: int = 0


def compute_daily_bias(daily: pd.DataFrame, today_str: str, current_price: float) -> tuple[str, int]:
    """Compute daily bias from price vs PDH/PDL midpoint + 5-day trend."""
    prior = daily[daily["date"] < today_str]
    if len(prior) < 5:
        return "NEUTRAL", 0

    pdh = float(prior.iloc[-1]["high"])
    pdl = float(prior.iloc[-1]["low"])
    pd_mid = (pdh + pdl) / 2

    sig1 = "BULLISH" if current_price > pd_mid else ("BEARISH" if current_price < pd_mid else "NEUTRAL")

    last5 = [float(prior.iloc[i]["close"]) for i in range(-5, 0)]
    slope = last5[-1] - last5[0]

    # ATR proxy for slope threshold
    if len(prior) >= 14:
        recent14 = prior.iloc[-14:]
        tr_sum = 0
        for i in range(1, len(recent14)):
            h = float(recent14.iloc[i]["high"])
            l = float(recent14.iloc[i]["low"])
            pc = float(recent14.iloc[i - 1]["close"])
            tr_sum += max(h - l, abs(h - pc), abs(l - pc))
        atr_day_local = tr_sum / (len(recent14) - 1)
    else:
        atr_day_local = 1.0

    slope_thresh = atr_day_local * 0.3
    sig2 = "BULLISH" if slope > slope_thresh else ("BEARISH" if slope < -slope_thresh else "NEUTRAL")

    if sig1 == sig2 and sig1 != "NEUTRAL":
        return sig1, 2
    elif sig1 != "NEUTRAL":
        return sig1, 1
    elif sig2 != "NEUTRAL":
        return sig2, 1
    return "NEUTRAL", 0


def detect_open_sweep(candles: pd.DataFrame, atr5m: float, sweep_mult: float) -> dict | None:
    """Improvement 1: Extreme open-bar fast detection (wick-based direction)."""
    if candles.empty or atr5m is None:
        return None

    # Filter to first 4 bars of the day (9:30-9:50)
    open_bars = candles[candles["timestamp"].dt.hour == 9].head(4)

    # Volume baseline
    if not open_bars.empty:
        avg_vol = open_bars["volume"].mean()
    else:
        avg_vol = candles.head(12)["volume"].mean()
    if avg_vol == 0:
        avg_vol = 1

    for _, bar in open_bars.iterrows():
        o, h, l, c, vol = bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]
        body = abs(c - o)
        rng = h - l
        atr_proxy = atr5m if atr5m else (c * 0.003)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if rng > atr_proxy * 1.3 and vol > avg_vol * 1.5:
            if upper_wick > lower_wick * 1.3 and upper_wick > body * 0.5:
                return {
                    "bias": "BEARISH",
                    "name": "Open Sweep (Upper Wick)",
                    "val": h,
                    "candle_time": bar["timestamp"],
                    "origin_high": h,
                    "origin_low": l,
                }
            elif lower_wick > upper_wick * 1.3 and lower_wick > body * 0.5:
                return {
                    "bias": "BULLISH",
                    "name": "Open Sweep (Lower Wick)",
                    "val": l,
                    "candle_time": bar["timestamp"],
                    "origin_high": h,
                    "origin_low": l,
                }
    return None


def detect_level_sweep(candles: pd.DataFrame, levels: dict, atr5m: float,
                       sweep_mult: float) -> dict | None:
    """Path B: Detect sweep of key levels in last 20 candles."""
    if candles.empty or atr5m is None:
        return None

    all_levels = []
    level_map = {
        "PDH": ("HIGH", levels.get("pdh")),
        "PDL": ("LOW", levels.get("pdl")),
        "PWH": ("HIGH", levels.get("pwh")),
        "PWL": ("LOW", levels.get("pwl")),
        "PMH": ("HIGH", levels.get("pmh")),
        "PML": ("LOW", levels.get("pml")),
    }
    for name, (lvl_type, val) in level_map.items():
        if val is not None:
            all_levels.append({"name": name, "type": lvl_type, "val": val})

    # Also add NY session levels from today's data
    ny_candles = candles[candles["timestamp"].dt.hour.between(9, 11)]
    if not ny_candles.empty:
        all_levels.append({"name": "NY High", "type": "HIGH", "val": float(ny_candles["high"].max())})
        all_levels.append({"name": "NY Low", "type": "LOW", "val": float(ny_candles["low"].min())})

    if not all_levels:
        return None

    lookback = candles.tail(20)
    for i in range(len(lookback) - 2, -1, -1):
        row = lookback.iloc[i]
        c_high, c_low, c_close = row["high"], row["low"], row["close"]

        for lvl in all_levels:
            if lvl["type"] == "LOW":
                # LOW sweep: candle straddles level, wick dips below
                if c_high >= lvl["val"] * 0.9995 and c_low < lvl["val"]:
                    depth = lvl["val"] - c_low
                    if depth > atr5m * sweep_mult:
                        # Reversal check A: close above level
                        if c_close >= lvl["val"]:
                            # Reversal check B: next 3 bars don't break down further
                            check_b = True
                            for k in range(i + 1, min(i + 4, len(lookback))):
                                if lookback.iloc[k]["close"] < c_low - 0.3 * atr5m:
                                    check_b = False
                                    break
                            if check_b:
                                return {
                                    "bias": "BULLISH",
                                    "name": lvl["name"],
                                    "val": lvl["val"],
                                    "candle_time": row["timestamp"],
                                    "origin_high": c_high,
                                    "origin_low": c_low,
                                }

            elif lvl["type"] == "HIGH":
                # HIGH sweep: candle straddles level, wick pierces above
                if c_low <= lvl["val"] * 1.0005 and c_high >= lvl["val"]:
                    depth = c_high - lvl["val"]
                    if depth > atr5m * sweep_mult:
                        # Reversal check A: close below level
                        if c_close <= lvl["val"]:
                            # Reversal check B: next 3 bars don't break up further
                            check_b = True
                            for k in range(i + 1, min(i + 4, len(lookback))):
                                if lookback.iloc[k]["close"] > c_high + 0.3 * atr5m:
                                    check_b = False
                                    break
                            if check_b:
                                return {
                                    "bias": "BEARISH",
                                    "name": lvl["name"],
                                    "val": lvl["val"],
                                    "candle_time": row["timestamp"],
                                    "origin_high": c_high,
                                    "origin_low": c_low,
                                }
    return None


def detect_ifvg(candles: pd.DataFrame, atr5m: float, current_price: float) -> dict | None:
    """Real iFVG (Inverted Fair Value Gap) detection — ICT definition."""
    if len(candles) < 5 or atr5m is None:
        return None

    start_idx = max(2, len(candles) - 30)

    # Step 1: Find all FVGs
    candidate_fvgs = []
    for i in range(start_idx, len(candles)):
        c1 = candles.iloc[i - 2]
        c3 = candles.iloc[i]

        if c3["low"] > c1["high"]:
            # Bullish FVG: gap UP
            candidate_fvgs.append({
                "idx": i, "kind": "BULL_FVG",
                "top": c3["low"], "bottom": c1["high"]
            })
        elif c3["high"] < c1["low"]:
            # Bearish FVG: gap DOWN
            candidate_fvgs.append({
                "idx": i, "kind": "BEAR_FVG",
                "top": c1["low"], "bottom": c3["high"]
            })

    # Step 2: Find inversions
    most_recent = None
    for fvg in candidate_fvgs:
        for j in range(fvg["idx"] + 1, len(candles)):
            cj_close = candles.iloc[j]["close"]
            if fvg["kind"] == "BULL_FVG" and cj_close < fvg["bottom"]:
                inv = {"top": fvg["top"], "bottom": fvg["bottom"],
                       "kind": "BEARISH", "inv_idx": j}
                if most_recent is None or j > most_recent["inv_idx"]:
                    most_recent = inv
                break
            if fvg["kind"] == "BEAR_FVG" and cj_close > fvg["top"]:
                inv = {"top": fvg["top"], "bottom": fvg["bottom"],
                       "kind": "BULLISH", "inv_idx": j}
                if most_recent is None or j > most_recent["inv_idx"]:
                    most_recent = inv
                break

    # Step 3: Check if price is in retest zone
    if most_recent:
        in_zone = most_recent["bottom"] <= current_price <= most_recent["top"]
        near = atr5m > 0 and (
            (current_price > most_recent["top"] and current_price - most_recent["top"] < 0.2 * atr5m) or
            (current_price < most_recent["bottom"] and most_recent["bottom"] - current_price < 0.2 * atr5m)
        )
        if in_zone or near:
            return {"type": most_recent["kind"], "top": most_recent["top"],
                    "bottom": most_recent["bottom"]}
    return None


# ═══════════════════════════════════════════════════════════════════════
# Signal Scorer (ported from Signal Scorer v6.4)
# ═══════════════════════════════════════════════════════════════════════

def score_signal(sweep: dict, ifvg: dict | None, atr5m: float, atr_day: float | None,
                 min_atr: float, candles: pd.DataFrame, direction: str,
                 daily_bias: str, daily_bias_conf: int,
                 session_label: str, minutes_since_s1: int | None) -> tuple[int, str]:
    """Score a signal using Simpsons scoring. Returns (score, grade)."""
    total = 0
    parts = []

    # 1. Session timing
    if "PRIME" in session_label or "Killzone" in session_label:
        total += 15; parts.append("Session PRIME +15")
    elif "London Close" in session_label or "Power Hour" in session_label:
        total += 10; parts.append("Session active +10")
    elif "Premarket" in session_label:
        total += 5; parts.append("Session premarket +5")

    # 2. ATR quality
    atr = atr5m
    atr_d = atr_day or 0
    if not atr or atr <= 0:
        atr = atr_d / 8 if atr_d > 0 else 0
    atr_typical = atr_d / 8 if atr_d > 0 else min_atr
    atr_elevated = atr_d / 5 if atr_d > 0 else min_atr * 1.6

    if atr >= atr_elevated:
        total += 10; parts.append("ATR elevated +10")
    elif atr >= atr_typical:
        total += 5; parts.append("ATR adequate +5")
    elif atr >= min_atr > 0:
        total += 2; parts.append("ATR above floor +2")
    elif atr > 0:
        total -= 2; parts.append("ATR weak -2")
    else:
        total -= 5; parts.append("ATR unavailable -5")

    # 3. Sweep level quality
    name = sweep["name"]
    if name in ("PDH", "PDL"):
        total += 10; parts.append(f"{name} swept +10")
    elif name in ("PWH", "PWL"):
        total += 8; parts.append(f"{name} swept +8")
    elif name in ("PMH", "PML"):
        total += 6; parts.append(f"{name} swept +6")
    elif "NY" in name:
        total += 4; parts.append(f"NY level swept +4")
    elif "Open Sweep" in name:
        pdh = sweep.get("pdh", 0)
        pdl = sweep.get("pdl", 0)
        sv = sweep["val"]
        if sv > 0 and pdh > 0 and sv >= pdh:
            total += 9; parts.append("Open Sweep above PDH +9")
        elif sv > 0 and pdl > 0 and sv <= pdl:
            total += 9; parts.append("Open Sweep below PDL +9")
        else:
            total += 4; parts.append("Open Sweep (no key level) +4")

    # 4. Sweep depth
    if atr and atr > 0:
        if sweep["bias"] == "BULLISH" and sweep.get("origin_low"):
            wick_depth = sweep["val"] - sweep["origin_low"]
        elif sweep["bias"] == "BEARISH" and sweep.get("origin_high"):
            wick_depth = sweep["origin_high"] - sweep["val"]
        else:
            wick_depth = 0
        if wick_depth > 0:
            ratio = wick_depth / atr
            if ratio >= 0.5:
                total += 10; parts.append("Sweep deep +10")
            elif ratio >= 0.3:
                total += 6; parts.append("Sweep moderate +6")
            elif ratio >= 0.1:
                total += 3; parts.append("Sweep shallow +3")

    # 5. iFVG
    if ifvg:
        aligned = (sweep["bias"] == "BULLISH" and ifvg["type"] == "BULLISH") or \
                  (sweep["bias"] == "BEARISH" and ifvg["type"] == "BEARISH")
        if aligned:
            total += 25; parts.append("iFVG aligned +25")
        else:
            total += 10; parts.append("iFVG detected +10")

    # 6. Volume confirmation
    if not candles.empty:
        last = candles.iloc[-1]
        vol_period = min(5, len(candles))
        avg_vol = candles.tail(vol_period)["volume"].mean()
        cur_vol = last["volume"]
        is_green = last["close"] > last["open"]
        is_red = last["close"] < last["open"]
        buy_ratio = cur_vol / avg_vol if avg_vol > 0 and is_green else 0
        sell_ratio = cur_vol / avg_vol if avg_vol > 0 and is_red else 0
        if sweep["bias"] == "BULLISH" and buy_ratio >= 1.2:
            total += 15; parts.append("Buy volume +15")
        elif sweep["bias"] == "BEARISH" and sell_ratio >= 1.2:
            total += 15; parts.append("Sell volume +15")

    # 7. Stage 2 time bonus
    if minutes_since_s1 is not None:
        if minutes_since_s1 < 10:
            total += 10; parts.append("Fast distribution +10")
        elif minutes_since_s1 < 20:
            total += 7; parts.append("Distribution 10-20min +7")
        elif minutes_since_s1 < 35:
            total += 4; parts.append("Distribution 20-35min +4")

    # 8. Daily bias
    if daily_bias != "NEUTRAL" and sweep["bias"] != "NEUTRAL":
        bias_dir = "BULLISH" if direction == "CALL" else "BEARISH"
        if daily_bias == bias_dir:
            bonus = 10 if daily_bias_conf >= 2 else 5
            total += bonus; parts.append(f"Daily bias aligned +{bonus}")
        else:
            total -= 5; parts.append("Daily bias counter -5")

    score = max(0, total)

    if score >= 75:
        grade = "A+"
    elif score >= 60:
        grade = "A"
    elif score >= 50:
        grade = "B+"
    elif score >= 40:
        grade = "B"
    elif score >= 25:
        grade = "C"
    else:
        grade = "F"

    return score, grade


# ═══════════════════════════════════════════════════════════════════════
# v10 Surgical Filter (14 rules)
# ═══════════════════════════════════════════════════════════════════════

def apply_v10_filter(candles: pd.DataFrame, direction: str, sweep: dict,
                     stage1_sweep_name: str, ifvg_detected: bool,
                     daily_bias: str, daily_bias_conf: int,
                     minutes_since_s1: int, today_date: date) -> str | None:
    """Apply v10 surgical filter. Returns None if passed, or rule code if blocked."""
    if candles.empty:
        return None

    is_call = direction == "CALL"
    last_bar = candles.iloc[-1]
    fb_open = last_bar["open"]
    fb_high = last_bar["high"]
    fb_low = last_bar["low"]
    fb_close = last_bar["close"]
    fb_vol = last_bar["volume"]

    # fire_bar_fav
    fb_move_pct = ((fb_close - fb_open) / fb_open * 100) if fb_open > 0 else 0
    fire_bar_fav = fb_move_pct if is_call else -fb_move_pct

    # fire_range
    fire_range = ((fb_high - fb_low) / fb_open * 100) if fb_open > 0 else 0

    # unfav_wick
    if is_call:
        unfav_wick = ((fb_open - fb_low) / fb_open * 100) if fb_open > fb_low else 0
    else:
        unfav_wick = ((fb_high - fb_open) / fb_open * 100) if fb_high > fb_open else 0

    # pre_30m_fav (6 bars before fire bar)
    pre_start = max(0, len(candles) - 7)
    pre_end = len(candles) - 1
    pre_30m_fav = 0
    if pre_end > pre_start:
        s_open = candles.iloc[pre_start]["open"]
        e_close = candles.iloc[pre_end - 1]["close"]
        if s_open > 0:
            pre_move = ((e_close - s_open) / s_open) * 100
            pre_30m_fav = pre_move if is_call else -pre_move

    # vol_ratio
    vol_ratio = 1.0
    if len(candles) >= 13:
        prior_vols = candles.iloc[-13:-1]["volume"].values
        avg_vol = prior_vols.mean()
        if avg_vol > 0:
            vol_ratio = fb_vol / avg_vol

    # DTE (days to Friday, 0-4)
    dow = today_date.weekday()  # 0=Mon
    dte = (4 - dow) if dow <= 4 else 4

    # DOW
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_name = dow_names[dow]

    # Hour
    hour = last_bar["timestamp"].hour if hasattr(last_bar["timestamp"], "hour") else 10

    # mss1
    mss1 = minutes_since_s1 or 0

    # bias_conf
    bias_conf = daily_bias_conf

    # Alignment
    is_aligned = (daily_bias == "BULLISH" and direction == "CALL") or \
                 (daily_bias == "BEARISH" and direction == "PUT")

    # S1 / S2 level names
    S1 = stage1_sweep_name or sweep["name"]
    S2 = sweep["name"]
    matched = S1 == S2 and S1 != ""
    no_ifvg = not ifvg_detected

    # S2 direction mismatch
    s2_dir_mismatch = (direction == "CALL" and S2 == "Open Sweep (Upper Wick)") or \
                      (direction == "PUT" and S2 == "Open Sweep (Lower Wick)")

    # ═══ Step 11 cuts (6 rules) ═══
    if bias_conf == 1 and vol_ratio < 0.3 and pre_30m_fav < 0:
        return "S11_A"
    if dow_name in ("Mon", "Fri") and S1 == "NY Low":
        return "S11_B"
    if no_ifvg and dte == 2 and S1 == "PWH":
        return "S11_C"
    if s2_dir_mismatch and hour == 14:
        return "S11_D"
    if dte == 1 and fire_range < 0.3 and hour == 14:
        return "S11_E"
    if bias_conf == 1 and dow_name == "Tue" and vol_ratio < 0.5:
        return "S11_F"

    # ═══ Step 12 cuts (8 rules) ═══
    if dte >= 2 and S1 in ("PML", "PMH") and pre_30m_fav > 0.5:
        return "S12_A"
    if bias_conf == 2 and S1 == "PDH" and fire_range >= 0.5:
        return "S12_B"
    if S2 == "PWH" and 0.5 <= vol_ratio < 1.0 and fire_bar_fav < -0.15:
        return "S12_C"
    if dte == 2 and S2 == "Open Sweep (Lower Wick)" and fire_bar_fav < 0:
        return "S12_D"
    if pre_30m_fav > 0.5 and hour == 12 and matched:
        return "S12_E"
    if direction == "CALL" and bias_conf == 2 and fire_bar_fav < -0.15 and mss1 <= 5:
        return "S12_F"
    if not is_aligned and matched and dte == 0 and fire_range >= 0.5:
        return "S12_G"
    if pre_30m_fav > 0 and fire_range < 0.3 and hour == 10 and 6 <= mss1 <= 15:
        return "S12_H"

    return None


# ═══════════════════════════════════════════════════════════════════════
# Option P&L grading
# ═══════════════════════════════════════════════════════════════════════

def find_atm_strike(price: float, ticker: str) -> float:
    """Find nearest ATM strike for a given price.

    SPY/QQQ have $1 intervals at all prices.
    Other tickers: $1 (< $50), $2.50 ($50-$100), $5 ($100-$500), $10 ($500+).
    """
    # ETFs have $1 intervals regardless of price
    if ticker in ("SPY", "QQQ", "IWM", "DIA"):
        interval = 1
    elif price < 50:
        interval = 1
    elif price < 100:
        interval = 2.5
    elif price < 500:
        interval = 5
    else:
        # Most $500+ stocks (META, AVGO) use $5 intervals
        interval = 5
    return round(price / interval) * interval


def find_option_expiry(today_str: str, ticker: str) -> str:
    """Find same-day expiry first (0DTE), then try Friday.

    ThetaData downloaded 0DTE contracts for all tickers.
    """
    return today_str  # Always try same-day first


def grade_option_trade(db_path: str, ticker: str, date_str: str,
                       fire_time_str: str, direction: str,
                       current_price: float) -> dict:
    """Grade a signal using actual option bars from ThetaData."""
    strike = find_atm_strike(current_price, ticker)
    expiry = date_str  # 0DTE first

    # Determine strike interval for this ticker/price
    if ticker in ("SPY", "QQQ", "IWM", "DIA"):
        interval = 1
    elif current_price < 50:
        interval = 1
    elif current_price < 100:
        interval = 2.5
    elif current_price < 500:
        interval = 5
    else:
        interval = 5

    # Try multiple strikes: ATM, then 1-2 OTM in signal direction, then 1 ITM
    strikes_to_try = [strike]
    if direction == "CALL":
        strikes_to_try += [strike + interval, strike + 2 * interval, strike - interval]
    else:
        strikes_to_try += [strike - interval, strike - 2 * interval, strike + interval]

    opt_df = pd.DataFrame()
    used_strike = strike
    for s in strikes_to_try:
        opt_df = load_option_bars(db_path, ticker, date_str, s, direction, expiry)
        if not opt_df.empty:
            used_strike = s
            break

    # If same-day didn't work, try Friday expiry
    if opt_df.empty:
        today = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_to_fri = (4 - today.weekday()) % 7
        if days_to_fri > 0:
            friday_expiry = (today + timedelta(days=days_to_fri)).strftime("%Y-%m-%d")
            for s in strikes_to_try:
                opt_df = load_option_bars(db_path, ticker, date_str, s, direction, friday_expiry)
                if not opt_df.empty:
                    used_strike = s
                    expiry = friday_expiry
                    break

    if opt_df.empty:
        return {"entry": None, "peak_pct": None, "pnl": None}

    opt_df["timestamp"] = pd.to_datetime(opt_df["timestamp"], utc=True).dt.tz_convert(ET)

    # Find entry bar (at or after fire time)
    fire_hour, fire_min = map(int, fire_time_str.split(":"))
    fire_dt = opt_df["timestamp"].iloc[0].replace(hour=fire_hour, minute=fire_min, second=0)

    entry_bars = opt_df[opt_df["timestamp"] >= fire_dt]
    if entry_bars.empty:
        return {"entry": None, "peak_pct": None, "pnl": None}

    # Use first available bar's close as entry (within 5 min)
    first_bar = entry_bars.iloc[0]
    if (first_bar["timestamp"] - fire_dt).total_seconds() > 300:
        return {"entry": None, "peak_pct": None, "pnl": None}

    entry_price = first_bar["close"]
    if entry_price <= 0:
        return {"entry": None, "peak_pct": None, "pnl": None}

    # Find peak from entry to EOD
    after_entry = opt_df[opt_df["timestamp"] >= first_bar["timestamp"]]
    peak_price = after_entry["high"].max()
    peak_pct = ((peak_price - entry_price) / entry_price) * 100

    # Exit at EOD close
    eod_price = after_entry.iloc[-1]["close"]
    eod_pct = ((eod_price - entry_price) / entry_price) * 100

    return {
        "entry": entry_price,
        "peak_pct": peak_pct,
        "pnl": eod_pct,
        "strike": used_strike,
        "expiry": expiry,
        "opt_df": after_entry,  # Full bars for V5 FSM simulation
    }


def simulate_v5_exit(opt_df: pd.DataFrame, entry_premium: float,
                     contracts: int, ticker: str, direction: str,
                     fire_time_str: str, date_str: str) -> dict:
    """Run V5 FSM on option bars after entry. Returns dollar P&L."""
    if entry_premium <= 0 or opt_df.empty:
        return {"pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0, "exit_prem": 0}

    option_type = "call" if direction == "CALL" else "put"
    tcfg = get_ticker_config(ticker, use_per_ticker=True, option_type=option_type)
    fsm = ExitFSM(tcfg, settings=_V6_SETTINGS)

    fire_hour, fire_min = map(int, fire_time_str.split(":"))
    entry_ts = datetime(2026, 1, 1, fire_hour, fire_min, tzinfo=ET)

    # Find underlying price from stock data (use close as proxy)
    underlying_0 = 0

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=underlying_0,
        dte=0, expiry_date=date_str,
    )

    locked_pnl = 0.0
    remaining = contracts

    for i in range(1, len(opt_df)):
        row = opt_df.iloc[i]
        prem = float(row["close"])
        if np.isnan(prem) or prem <= 0:
            continue

        bid = float(row["low"]) if not np.isnan(row["low"]) else prem  # conservative: use low as bid proxy
        ask = float(row["high"]) if not np.isnan(row["high"]) else prem

        ts = row["timestamp"]
        if hasattr(ts, 'hour'):
            now = datetime(2026, 1, 1, ts.hour, ts.minute, tzinfo=ET)
            minutes_to_close = max(0, 16 * 60 - (ts.hour * 60 + ts.minute))
        else:
            now = entry_ts + timedelta(minutes=i)
            minutes_to_close = max(0, 390 - i)

        action = fsm.evaluate(
            state, prem, bid, ask, now,
            current_underlying=0,
            minutes_to_close=minutes_to_close,
            candle_data={},
        )

        if action.should_exit:
            exit_price = bid if bid > 0 else prem

            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (exit_price - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            elapsed = i
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (exit_price - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold_min": elapsed, "peak_gain": peak_gain,
                "exit_prem": exit_price,
            }

    # EOD fallback
    last_prem = entry_premium
    for i in range(len(opt_df) - 1, 0, -1):
        v = float(opt_df.iloc[i]["close"])
        if not np.isnan(v) and v > 0:
            last_prem = v
            break
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold_min": len(opt_df),
        "peak_gain": peak_gain, "exit_prem": last_prem,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main scan loop (simulates Simpsons scanning every 5 min)
# ═══════════════════════════════════════════════════════════════════════

def get_session_label(hour: float) -> str:
    if 9.5 <= hour <= 10.75:
        return "NY Open Killzone (PRIME)"
    elif 10.75 < hour <= 12.0:
        return "London Close Window"
    elif 12.0 < hour <= 14.0:
        return "Midday Session"
    elif 14.0 < hour <= 14.5:
        return "Early Afternoon"
    elif 14.5 < hour <= 15.5:
        return "Power Hour"
    return "Off Hours"


def scan_day(db_path: str, date_str: str, tickers: list[str],
             use_v10_filter: bool = True,
             include_stage1: bool = False) -> list[Signal]:
    """Run the full Simpsons scan for one trading day."""
    signals = []
    today = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Per-ticker AMD state machine (persists across scan intervals within the day)
    stage1_state = {}  # ticker -> {bias, sweep_level, sweep_name, candle_time, origin_high, origin_low, fired_at_min}
    stage2_dedup = set()  # (ticker, bias, sweep_name)

    for ticker in tickers:
        cfg = TICKER_CONFIG.get(ticker, {"tier": 1, "minATR": 0.5, "sweepMult": 0.30})

        # Load daily candles for levels
        daily = load_daily_candles(db_path, ticker, date_str)
        if daily.empty:
            continue

        levels = get_daily_levels(daily, date_str)
        if not levels:
            continue

        # Load all 5-min candles for the day
        candles_5m = load_5min_candles(db_path, ticker, date_str)
        if candles_5m.empty or len(candles_5m) < 5:
            continue

        atr_day = calc_atr(candles_5m, 14) if len(candles_5m) >= 15 else None

        # Scan every 5 minutes from 9:30 to 15:30
        # We simulate by iterating over the candle indices
        for bar_idx in range(4, len(candles_5m)):
            bar = candles_5m.iloc[bar_idx]
            bar_time = bar["timestamp"]
            et_hour = bar_time.hour + bar_time.minute / 60

            # Only scan during market hours
            if et_hour < 9.5 or et_hour > 15.5:
                continue

            session_label = get_session_label(et_hour)

            # Candles up to current bar
            up_to_now = candles_5m.iloc[:bar_idx + 1]
            current_price = float(bar["close"])
            closes = up_to_now["close"].tolist()

            atr5m = calc_atr(up_to_now, 14)
            if atr5m is None or atr5m < cfg["minATR"] * 0.1:
                continue

            # Daily bias
            daily_bias, daily_bias_conf = compute_daily_bias(daily, date_str, current_price)

            # ── SWEEP DETECTION ──
            sweep = None

            # Try open sweep first (Improvement 1)
            if bar_idx <= 6:  # Only in first ~30 min
                sweep = detect_open_sweep(up_to_now, atr5m, cfg["sweepMult"])

            # Path B: level sweep
            if sweep is None:
                sweep = detect_level_sweep(up_to_now, levels, atr5m, cfg["sweepMult"])

            if sweep is None:
                # Check if existing Stage 1 can produce Stage 2
                if ticker in stage1_state:
                    s1 = stage1_state[ticker]
                    s1_age_min = (bar_idx - s1["bar_idx"]) * 5
                    if s1_age_min > 60:
                        del stage1_state[ticker]
                        continue

                    # Check for distribution breakout
                    momentum3 = closes[-1] - closes[-4] if len(closes) >= 4 else 0
                    min_mom = (atr5m or 0.5) * 0.35
                    ref_bias = s1["bias"]
                    bullish_bo = ref_bias == "BULLISH" and (
                        bar["high"] > s1["origin_high"] or current_price > s1["origin_high"] * 0.998
                    ) and momentum3 > min_mom
                    bearish_bo = ref_bias == "BEARISH" and (
                        bar["low"] < s1["origin_low"] or current_price < s1["origin_low"] * 1.002
                    ) and momentum3 < -min_mom

                    if bullish_bo or bearish_bo:
                        dedup_key = (ticker, ref_bias, s1["sweep_name"])
                        if dedup_key not in stage2_dedup:
                            stage2_dedup.add(dedup_key)
                            direction = "CALL" if ref_bias == "BULLISH" else "PUT"
                            sweep_info = {
                                "bias": ref_bias,
                                "name": s1["sweep_name"],
                                "val": s1["sweep_level"],
                                "origin_high": s1["origin_high"],
                                "origin_low": s1["origin_low"],
                            }
                            mss1 = s1_age_min
                            fire_time = f"{bar_time.hour:02d}:{bar_time.minute:02d}"

                            # iFVG detection
                            ifvg = detect_ifvg(up_to_now, atr5m, current_price)

                            # Score it
                            score, grade = score_signal(
                                sweep_info, ifvg, atr5m, atr_day, cfg["minATR"],
                                up_to_now, direction, daily_bias, daily_bias_conf,
                                session_label, mss1
                            )

                            if grade == "F":
                                continue

                            # v10 filter
                            v10_reason = None
                            if use_v10_filter:
                                v10_reason = apply_v10_filter(
                                    up_to_now, direction, sweep_info,
                                    s1["sweep_name"], ifvg is not None,
                                    daily_bias, daily_bias_conf, mss1, today
                                )

                            if v10_reason is None:
                                signals.append(Signal(
                                    ticker=ticker, date=date_str, fire_time=fire_time,
                                    direction=direction, stage="STAGE2",
                                    sweep_level_name=s1["sweep_name"],
                                    sweep_level_val=s1["sweep_level"],
                                    amd_bias=ref_bias, score=score, grade=grade,
                                    current_price=current_price, atr5m=atr5m,
                                    ifvg_detected=ifvg is not None,
                                    session_label=session_label,
                                    v10_filter_reason=v10_reason,
                                ))

                            del stage1_state[ticker]
                continue

            # ── We have a sweep — process AMD state machine ──
            bias = sweep["bias"]
            direction = "CALL" if bias == "BULLISH" else "PUT"
            dedup_key = (ticker, bias, sweep["name"])

            if dedup_key in stage2_dedup:
                continue

            # Check for open candle false sweep guard
            if bar_idx <= 2:
                # Require 2 confirming candles
                consec = 0
                for ci in range(bar_idx + 1, min(bar_idx + 4, len(candles_5m))):
                    move = candles_5m.iloc[ci]["close"] - candles_5m.iloc[ci]["open"]
                    if bias == "BULLISH" and move > 0:
                        consec += 1
                    elif bias == "BEARISH" and move < 0:
                        consec += 1
                    else:
                        break
                if consec < 2:
                    continue

            # Check if existing Stage 1 + new sweep → Stage 2
            if ticker in stage1_state:
                s1 = stage1_state[ticker]
                s1_age_min = (bar_idx - s1["bar_idx"]) * 5
                if s1_age_min <= 60:
                    momentum3 = closes[-1] - closes[-4] if len(closes) >= 4 else 0
                    min_mom = (atr5m or 0.5) * 0.35
                    ref_bias = s1["bias"]
                    bullish_bo = ref_bias == "BULLISH" and (
                        bar["high"] > s1["origin_high"] or current_price > s1["origin_high"] * 0.998
                    ) and momentum3 > min_mom
                    bearish_bo = ref_bias == "BEARISH" and (
                        bar["low"] < s1["origin_low"] or current_price < s1["origin_low"] * 1.002
                    ) and momentum3 < -min_mom

                    if bullish_bo or bearish_bo:
                        dedup_key2 = (ticker, ref_bias, s1["sweep_name"])
                        if dedup_key2 not in stage2_dedup:
                            stage2_dedup.add(dedup_key2)
                            ifvg = detect_ifvg(up_to_now, atr5m, current_price)
                            mss1 = s1_age_min
                            fire_time = f"{bar_time.hour:02d}:{bar_time.minute:02d}"

                            score, grade = score_signal(
                                sweep, ifvg, atr5m, atr_day, cfg["minATR"],
                                up_to_now, direction, daily_bias, daily_bias_conf,
                                session_label, mss1
                            )
                            if grade != "F":
                                v10_reason = None
                                if use_v10_filter:
                                    v10_reason = apply_v10_filter(
                                        up_to_now, direction, sweep,
                                        s1["sweep_name"], ifvg is not None,
                                        daily_bias, daily_bias_conf, mss1, today
                                    )
                                if v10_reason is None:
                                    signals.append(Signal(
                                        ticker=ticker, date=date_str, fire_time=fire_time,
                                        direction=direction, stage="STAGE2",
                                        sweep_level_name=s1["sweep_name"],
                                        sweep_level_val=s1["sweep_level"],
                                        amd_bias=ref_bias, score=score, grade=grade,
                                        current_price=current_price, atr5m=atr5m,
                                        ifvg_detected=ifvg is not None,
                                        session_label=session_label,
                                        v10_filter_reason=v10_reason,
                                    ))
                            del stage1_state[ticker]
                            continue
                else:
                    del stage1_state[ticker]

            # Register as Stage 1
            fire_time = f"{bar_time.hour:02d}:{bar_time.minute:02d}"
            stage1_state[ticker] = {
                "bias": bias,
                "sweep_level": sweep["val"],
                "sweep_name": sweep["name"],
                "origin_high": sweep.get("origin_high", bar["high"]),
                "origin_low": sweep.get("origin_low", bar["low"]),
                "bar_idx": bar_idx,
            }

            if include_stage1:
                ifvg = detect_ifvg(up_to_now, atr5m, current_price)
                score, grade = score_signal(
                    sweep, ifvg, atr5m, atr_day, cfg["minATR"],
                    up_to_now, direction, daily_bias, daily_bias_conf,
                    session_label, None
                )
                if grade != "F":
                    signals.append(Signal(
                        ticker=ticker, date=date_str, fire_time=fire_time,
                        direction=direction, stage="STAGE1_NEW",
                        sweep_level_name=sweep["name"],
                        sweep_level_val=sweep["val"],
                        amd_bias=bias, score=score, grade=grade,
                        current_price=current_price, atr5m=atr5m,
                        ifvg_detected=ifvg is not None,
                        session_label=session_label,
                        v10_filter_reason=None,
                    ))

    return signals


# ═══════════════════════════════════════════════════════════════════════
# Main backtest
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Simpsons AMD+iFVG Backtest")
    parser.add_argument("--days", type=int, default=60, help="Trading days to backtest")
    parser.add_argument("--no-v10-filter", action="store_true", help="Disable v10 filter")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    parser.add_argument("--stage1-only", action="store_true", help="Include Stage 1 signals")
    parser.add_argument("--no-option-grade", action="store_true", help="Skip option grading (faster)")
    parser.add_argument("--min-score", type=int, default=50, help="Min score for V5 sim (default 50)")
    parser.add_argument("--kz-only", action="store_true", help="NY Killzone only for V5 sim")
    parser.add_argument("--no-scalp", action="store_true", help="Disable scalp_target (let runners run)")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated tickers to exclude from V5 sim")
    args = parser.parse_args()

    use_v10 = not args.no_v10_filter
    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"Simpsons AMD+iFVG Backtest")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Days: {args.days}")
    print(f"  v10 filter: {'ON' if use_v10 else 'OFF'}")
    print(f"  Stages: {'Stage 1 + Stage 2' if args.stage1_only else 'Stage 2 only'}")
    print(f"  Option grading: {'ON' if not args.no_option_grade else 'OFF'}")
    print()

    days = load_trading_days(THETADATA_DB, args.days)
    print(f"  Trading days: {len(days)} ({days[0]} to {days[-1]})")

    # Score filter for V5 simulation (KZ + score >= threshold)
    min_score = args.min_score
    kz_only = args.kz_only
    excluded_tickers = set(t.strip().upper() for t in args.exclude.split(",")) if args.exclude else set()

    # Override V6 settings if --no-scalp
    if args.no_scalp:
        _V6_SETTINGS.ENABLE_SCALP_TARGET = False
        _V6_SETTINGS.SCALP_TARGET_PCT = 999.0  # effectively disabled

    all_signals = []
    t0 = time.time()

    # Portfolio simulation state
    portfolio = PORTFOLIO_START
    total_pnl = 0.0
    equity_curve = [(days[0], PORTFOLIO_START)]
    trade_results = []  # (signal, fsm_result)
    daily_pnl = defaultdict(float)

    for i, day in enumerate(days):
        day_signals = scan_day(
            THETADATA_DB, day, tickers,
            use_v10_filter=use_v10,
            include_stage1=args.stage1_only,
        )

        # Sort by fire_time so we process in order
        day_signals.sort(key=lambda s: s.fire_time)

        day_open = 0  # concurrent open positions this day
        day_open_tickers = set()
        day_spent = 0.0
        sod_balance = portfolio

        # Grade with option data + V5 FSM
        if not args.no_option_grade:
            for sig in day_signals:
                result = grade_option_trade(
                    THETADATA_DB, sig.ticker, sig.date,
                    sig.fire_time, sig.direction, sig.current_price
                )
                sig.option_entry = result.get("entry")
                sig.peak_pct = result.get("peak_pct")
                sig.pnl_dollars = result.get("pnl")
                opt_df = result.get("opt_df")

                # Apply score/session/exclusion filter for V5 simulation
                if min_score and sig.score < min_score:
                    continue
                if kz_only and sig.session_label != "NY Open Killzone (PRIME)":
                    continue
                if sig.ticker in excluded_tickers:
                    continue

                # Skip if no option data
                entry_prem = result.get("entry")
                if entry_prem is None or not np.isfinite(entry_prem) or entry_prem <= 0:
                    continue
                if opt_df is None or (isinstance(opt_df, pd.DataFrame) and opt_df.empty):
                    continue

                # Premium cap
                if entry_prem > PREMIUM_CAP:
                    continue

                # Concurrent limit
                if day_open >= MAX_CONCURRENT:
                    continue

                # No duplicate tickers same day
                if sig.ticker in day_open_tickers:
                    continue

                # Position sizing
                deployable = portfolio * MAX_RISK_PCT
                per_slot = deployable / MAX_CONCURRENT
                cost_per = entry_prem * 100
                scaled = per_slot * 0.85
                raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
                pos_cap_ct = int(portfolio * MAX_POSITION_PCT / cost_per) if cost_per > 0 else 1
                dollar_ct = int(MAX_POSITION_DOLLARS / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(raw_ct, pos_cap_ct, dollar_ct, MAX_CONTRACTS))

                trade_cost = contracts * cost_per
                day_spent += trade_cost

                # Run V5 FSM
                fsm_result = simulate_v5_exit(
                    opt_df, entry_prem, contracts, sig.ticker, sig.direction,
                    sig.fire_time, sig.date
                )

                sig.fsm_pnl = fsm_result["pnl"]
                sig.fsm_reason = fsm_result["reason"]
                sig.fsm_hold_min = fsm_result["hold_min"]
                sig.fsm_peak_gain = fsm_result["peak_gain"]
                sig.fsm_contracts = contracts

                day_open += 1
                day_open_tickers.add(sig.ticker)

                trade_results.append((sig, fsm_result))
                daily_pnl[day] += fsm_result["pnl"]

                # Assume position closes same day (0DTE)
                day_open -= 1

        all_signals.extend(day_signals)

        # Update portfolio
        portfolio += daily_pnl[day]
        total_pnl += daily_pnl[day]
        equity_curve.append((day, portfolio))

        if (i + 1) % 10 == 0 or i == len(days) - 1:
            elapsed = time.time() - t0
            trades_so_far = len(trade_results)
            print(f"  [{i+1}/{len(days)}] {day} — {len(day_signals)} signals, {trades_so_far} trades, P&L=${total_pnl:+,.0f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\nScan complete: {len(all_signals)} signals in {elapsed:.1f}s\n")

    if not all_signals:
        print("No signals found.")
        return

    # ── Results ──
    graded = [s for s in all_signals if s.peak_pct is not None and not np.isnan(s.peak_pct)]
    ungraded = [s for s in all_signals if s.peak_pct is None or (s.peak_pct is not None and np.isnan(s.peak_pct))]

    # Win = peak >= 40% (same as Simpsons grading)
    wins = [s for s in graded if s.peak_pct >= 40]
    losses = [s for s in graded if s.peak_pct < 40]

    print("=" * 70)
    print(f"RESULTS — Simpsons AMD+iFVG Backtest")
    print("=" * 70)
    print(f"  Period: {days[0]} to {days[-1]} ({len(days)} trading days)")
    print(f"  v10 filter: {'ON' if use_v10 else 'OFF'}")
    print(f"  Total signals: {len(all_signals)} ({len(graded)} graded, {len(ungraded)} ungraded)")
    print(f"  Wins (peak >= 40%): {len(wins)}")
    print(f"  Losses: {len(losses)}")
    if graded:
        wr = len(wins) / len(graded) * 100
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Avg peak %: {sum(s.peak_pct for s in graded) / len(graded):.1f}%")
        avg_win_peak = sum(s.peak_pct for s in wins) / len(wins) if wins else 0
        avg_loss_peak = sum(s.peak_pct for s in losses) / len(losses) if losses else 0
        print(f"  Avg winner peak: {avg_win_peak:.1f}%")
        print(f"  Avg loser peak: {avg_loss_peak:.1f}%")
        print(f"  Signals/day: {len(all_signals) / len(days):.1f}")

    # Per-ticker breakdown
    print(f"\n{'─' * 70}")
    print(f"{'Ticker':<8} {'Signals':>8} {'Graded':>8} {'Wins':>6} {'WR':>6} {'Avg Peak':>10} {'Sigs/Day':>8}")
    print(f"{'─' * 70}")
    for tk in sorted(set(s.ticker for s in all_signals)):
        tk_sigs = [s for s in all_signals if s.ticker == tk]
        tk_graded = [s for s in tk_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        tk_wins = [s for s in tk_graded if s.peak_pct >= 40]
        tk_wr = len(tk_wins) / len(tk_graded) * 100 if tk_graded else 0
        tk_avg_peak = sum(s.peak_pct for s in tk_graded) / len(tk_graded) if tk_graded else 0
        print(f"{tk:<8} {len(tk_sigs):>8} {len(tk_graded):>8} {len(tk_wins):>6} {tk_wr:>5.0f}% {tk_avg_peak:>9.1f}% {len(tk_sigs)/len(days):>7.1f}")

    # Session breakdown
    print(f"\n{'─' * 70}")
    print(f"{'Session':<30} {'Signals':>8} {'WR':>6} {'Avg Peak':>10}")
    print(f"{'─' * 70}")
    for sess in sorted(set(s.session_label for s in all_signals)):
        s_sigs = [s for s in all_signals if s.session_label == sess]
        s_graded = [s for s in s_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        s_wins = [s for s in s_graded if s.peak_pct >= 40]
        s_wr = len(s_wins) / len(s_graded) * 100 if s_graded else 0
        s_avg = sum(s.peak_pct for s in s_graded) / len(s_graded) if s_graded else 0
        print(f"{sess:<30} {len(s_sigs):>8} {s_wr:>5.0f}% {s_avg:>9.1f}%")

    # Sweep level breakdown
    print(f"\n{'─' * 70}")
    print(f"{'Sweep Level':<30} {'Signals':>8} {'WR':>6} {'Avg Peak':>10}")
    print(f"{'─' * 70}")
    for lvl in sorted(set(s.sweep_level_name for s in all_signals)):
        l_sigs = [s for s in all_signals if s.sweep_level_name == lvl]
        l_graded = [s for s in l_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        l_wins = [s for s in l_graded if s.peak_pct >= 40]
        l_wr = len(l_wins) / len(l_graded) * 100 if l_graded else 0
        l_avg = sum(s.peak_pct for s in l_graded) / len(l_graded) if l_graded else 0
        print(f"{lvl:<30} {len(l_sigs):>8} {l_wr:>5.0f}% {l_avg:>9.1f}%")

    # Direction breakdown
    print(f"\n{'─' * 70}")
    for d in ("CALL", "PUT"):
        d_sigs = [s for s in all_signals if s.direction == d]
        d_graded = [s for s in d_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        d_wins = [s for s in d_graded if s.peak_pct >= 40]
        d_wr = len(d_wins) / len(d_graded) * 100 if d_graded else 0
        print(f"{d}: {len(d_sigs)} signals, {d_wr:.0f}% WR")

    # Score band analysis (KEY: find the quality filter)
    print(f"\n{'─' * 70}")
    print("SCORE BAND ANALYSIS (graded signals only):")
    print(f"{'Score Band':<15} {'Signals':>8} {'Graded':>8} {'Wins':>6} {'WR':>6} {'Avg Peak':>10} {'Sigs/Day':>9}")
    print(f"{'─' * 70}")
    score_bands = [(0, 29, "0-29"), (30, 39, "30-39"), (40, 49, "40-49"),
                   (50, 59, "50-59"), (60, 69, "60-69"), (70, 79, "70-79"),
                   (80, 100, "80+")]
    for lo, hi, label in score_bands:
        b_sigs = [s for s in all_signals if lo <= s.score <= hi]
        b_graded = [s for s in b_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        b_wins = [s for s in b_graded if s.peak_pct >= 40]
        b_wr = len(b_wins) / len(b_graded) * 100 if b_graded else 0
        b_avg = sum(s.peak_pct for s in b_graded) / len(b_graded) if b_graded else 0
        print(f"{label:<15} {len(b_sigs):>8} {len(b_graded):>8} {len(b_wins):>6} {b_wr:>5.0f}% {b_avg:>9.1f}% {len(b_sigs)/len(days):>8.1f}")

    # Cumulative: score >= X
    print(f"\n{'─' * 70}")
    print("CUMULATIVE FILTER (score >= threshold, graded only):")
    print(f"{'Threshold':<12} {'Signals':>8} {'Graded':>8} {'Wins':>6} {'WR':>6} {'Sigs/Day':>9}")
    print(f"{'─' * 70}")
    for threshold in [30, 40, 50, 55, 60, 65, 70, 75, 80]:
        t_sigs = [s for s in all_signals if s.score >= threshold]
        t_graded = [s for s in t_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        t_wins = [s for s in t_graded if s.peak_pct >= 40]
        t_wr = len(t_wins) / len(t_graded) * 100 if t_graded else 0
        print(f">= {threshold:<9} {len(t_sigs):>8} {len(t_graded):>8} {len(t_wins):>6} {t_wr:>5.0f}% {len(t_sigs)/len(days):>8.1f}")

    # NY Killzone only + score filter
    print(f"\n{'─' * 70}")
    print("NY KILLZONE ONLY + SCORE FILTER (best combo for live trading):")
    print(f"{'Filter':<30} {'Signals':>8} {'Graded':>8} {'Wins':>6} {'WR':>6} {'Sigs/Day':>9}")
    print(f"{'─' * 70}")
    for threshold in [0, 40, 50, 55, 60, 65, 70]:
        nk_sigs = [s for s in all_signals
                   if s.session_label == "NY Open Killzone (PRIME)" and s.score >= threshold]
        nk_graded = [s for s in nk_sigs if s.peak_pct is not None and not np.isnan(s.peak_pct)]
        nk_wins = [s for s in nk_graded if s.peak_pct >= 40]
        nk_wr = len(nk_wins) / len(nk_graded) * 100 if nk_graded else 0
        label = f"KZ + score >= {threshold}" if threshold > 0 else "KZ (all scores)"
        print(f"{label:<30} {len(nk_sigs):>8} {len(nk_graded):>8} {len(nk_wins):>6} {nk_wr:>5.0f}% {len(nk_sigs)/len(days):>8.1f}")

    # Ungraded analysis
    if ungraded:
        print(f"\n{'─' * 70}")
        print(f"UNGRADED ANALYSIS ({len(ungraded)} signals without option data):")
        ug_sessions = defaultdict(int)
        ug_tickers = defaultdict(int)
        for s in ungraded:
            ug_sessions[s.session_label] += 1
            ug_tickers[s.ticker] += 1
        print("  By session:")
        for sess, cnt in sorted(ug_sessions.items(), key=lambda x: -x[1]):
            print(f"    {sess:<35} {cnt:>5}")
        print("  By ticker:")
        for tk, cnt in sorted(ug_tickers.items(), key=lambda x: -x[1]):
            print(f"    {tk:<35} {cnt:>5}")

    # v10 filter analysis (if disabled, show what it would have filtered)
    if not use_v10:
        print(f"\n{'─' * 70}")
        print("v10 FILTER ANALYSIS (what would be filtered):")
        today_dt = datetime.strptime(days[-1], "%Y-%m-%d").date()
        filtered_count = 0
        for sig in all_signals:
            reason = apply_v10_filter(
                pd.DataFrame(),  # Would need candles — just show count estimate
                sig.direction, {"name": sig.sweep_level_name, "val": sig.sweep_level_val,
                                "bias": sig.amd_bias},
                sig.sweep_level_name, sig.ifvg_detected,
                "NEUTRAL", 1, 0, today_dt
            )
            if reason:
                filtered_count += 1
        print(f"  Would filter: ~{filtered_count} of {len(all_signals)} signals")

    # Trade list (last 30)
    print(f"\n{'─' * 70}")
    print("RECENT SIGNALS:")
    print(f"{'Date':<12} {'Time':<6} {'Ticker':<6} {'Dir':<5} {'Stage':<7} {'Sweep':<25} {'Score':>5} {'Grade':<3} {'Peak%':>7}")
    print(f"{'─' * 70}")
    for s in all_signals[-30:]:
        peak_str = f"{s.peak_pct:.0f}%" if (s.peak_pct is not None and not np.isnan(s.peak_pct)) else "N/A"
        print(f"{s.date:<12} {s.fire_time:<6} {s.ticker:<6} {s.direction:<5} {s.stage:<7} {s.sweep_level_name:<25} {s.score:>5} {s.grade:<3} {peak_str:>7}")

    # Comparison with our ML scan
    print(f"\n{'=' * 70}")
    print("COMPARISON: Simpsons vs ML Scan")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'Simpsons':<20} {'ML Scan (gold std)':<20}")
    print(f"  {'─' * 65}")
    if graded:
        wr = len(wins) / len(graded) * 100
        print(f"  {'Signals/day':<25} {len(all_signals)/len(days):<20.1f} {'~1.1':<20}")
        print(f"  {'Win rate':<25} {f'{wr:.1f}%':<20} {'91.7%':<20}")
        print(f"  {'Avg peak %':<25} {f'{sum(s.peak_pct for s in graded)/len(graded):.1f}%':<20} {'N/A':<20}")
        print(f"  {'Scan window':<25} {'All day (9:30-3:30)':<20} {'9:35-11:00 only':<20}")

    # ══════════════════════════════════════════════════════════════════
    # V5 FSM PORTFOLIO SIMULATION RESULTS
    # ══════════════════════════════════════════════════════════════════
    if trade_results:
        print(f"\n{'=' * 70}")
        print("V5 FSM PORTFOLIO SIMULATION")
        print(f"{'=' * 70}")
        print(f"  Portfolio start: ${PORTFOLIO_START:,}")
        print(f"  Portfolio end: ${portfolio:,.0f}")
        print(f"  Total P&L: ${total_pnl:+,.0f} ({total_pnl/PORTFOLIO_START*100:+.1f}%)")
        print(f"  Total trades: {len(trade_results)}")
        print(f"  Trades/day: {len(trade_results)/len(days):.1f}")
        filter_desc = f"score >= {min_score}" + (" + KZ only" if kz_only else "")
        print(f"  Filter: {filter_desc}")

        fsm_wins = [t for t in trade_results if t[1]["pnl"] > 0]
        fsm_losses = [t for t in trade_results if t[1]["pnl"] <= 0]
        fsm_wr = len(fsm_wins) / len(trade_results) * 100 if trade_results else 0
        print(f"  Win rate: {fsm_wr:.1f}% ({len(fsm_wins)}W / {len(fsm_losses)}L)")

        avg_win = sum(t[1]["pnl"] for t in fsm_wins) / len(fsm_wins) if fsm_wins else 0
        avg_loss = sum(t[1]["pnl"] for t in fsm_losses) / len(fsm_losses) if fsm_losses else 0
        print(f"  Avg win: ${avg_win:+,.0f}  |  Avg loss: ${avg_loss:+,.0f}")

        gross_profit = sum(t[1]["pnl"] for t in fsm_wins)
        gross_loss = abs(sum(t[1]["pnl"] for t in fsm_losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        print(f"  Profit factor: {pf:.2f}")

        # Max drawdown
        peak_equity = PORTFOLIO_START
        max_dd = 0
        running = PORTFOLIO_START
        for sig, res in trade_results:
            running += res["pnl"]
            peak_equity = max(peak_equity, running)
            dd = (peak_equity - running) / peak_equity * 100
            max_dd = max(max_dd, dd)
        print(f"  Max drawdown: {max_dd:.1f}%")

        # Runners (peak gain >= 100%)
        runners = [t for t in trade_results if t[1]["peak_gain"] >= 100]
        big_runners = [t for t in trade_results if t[1]["peak_gain"] >= 200]
        print(f"  Runners (100%+): {len(runners)}  |  Big runners (200%+): {len(big_runners)}")

        # Best and worst trades
        sorted_trades = sorted(trade_results, key=lambda t: t[1]["pnl"], reverse=True)
        print(f"\n  TOP 10 TRADES:")
        print(f"  {'Ticker':<7} {'Date':<12} {'Dir':<5} {'Score':>5} {'Ctrs':>5} {'P&L':>10} {'Peak%':>7} {'Exit':>12} {'Hold':>5}")
        for sig, res in sorted_trades[:10]:
            print(f"  {sig.ticker:<7} {sig.date:<12} {sig.direction:<5} {sig.score:>5} {sig.fsm_contracts:>5} "
                  f"${res['pnl']:>+9,.0f} {res['peak_gain']:>6.0f}% {res['reason']:>12} {res['hold_min']:>4}m")

        print(f"\n  WORST 5 TRADES:")
        for sig, res in sorted_trades[-5:]:
            print(f"  {sig.ticker:<7} {sig.date:<12} {sig.direction:<5} {sig.score:>5} {sig.fsm_contracts:>5} "
                  f"${res['pnl']:>+9,.0f} {res['peak_gain']:>6.0f}% {res['reason']:>12} {res['hold_min']:>4}m")

        # Per-ticker P&L
        print(f"\n  PER-TICKER P&L:")
        print(f"  {'Ticker':<7} {'Trades':>7} {'Wins':>5} {'WR':>5} {'P&L':>10} {'Avg P&L':>10} {'Runners':>8}")
        tk_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "runners": 0})
        for sig, res in trade_results:
            tk_stats[sig.ticker]["trades"] += 1
            if res["pnl"] > 0:
                tk_stats[sig.ticker]["wins"] += 1
            tk_stats[sig.ticker]["pnl"] += res["pnl"]
            if res["peak_gain"] >= 100:
                tk_stats[sig.ticker]["runners"] += 1
        for tk in sorted(tk_stats, key=lambda t: tk_stats[t]["pnl"], reverse=True):
            s = tk_stats[tk]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
            avg = s["pnl"] / s["trades"]
            print(f"  {tk:<7} {s['trades']:>7} {s['wins']:>5} {wr:>4.0f}% ${s['pnl']:>+9,.0f} ${avg:>+9,.0f} {s['runners']:>8}")

        # Exit reason distribution
        print(f"\n  EXIT REASON DISTRIBUTION:")
        reason_counts = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for sig, res in trade_results:
            reason_counts[res["reason"]]["count"] += 1
            reason_counts[res["reason"]]["pnl"] += res["pnl"]
        for reason in sorted(reason_counts, key=lambda r: reason_counts[r]["count"], reverse=True):
            rc = reason_counts[reason]
            print(f"  {reason:<25} {rc['count']:>4} trades  ${rc['pnl']:>+10,.0f}")

        # Daily P&L histogram
        print(f"\n  DAILY P&L:")
        green_days = sum(1 for d in daily_pnl.values() if d > 0)
        red_days = sum(1 for d in daily_pnl.values() if d < 0)
        flat_days = len(days) - green_days - red_days
        print(f"  Green: {green_days}  Red: {red_days}  Flat: {flat_days}")
        if daily_pnl:
            best_day = max(daily_pnl.items(), key=lambda x: x[1])
            worst_day = min(daily_pnl.items(), key=lambda x: x[1])
            print(f"  Best day: {best_day[0]} ${best_day[1]:+,.0f}")
            print(f"  Worst day: {worst_day[0]} ${worst_day[1]:+,.0f}")
    else:
        print("\nNo trades passed V5 FSM simulation filters.")


if __name__ == "__main__":
    main()
