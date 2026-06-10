"""End-to-end backtest: ML signal models → V5 FSM exit → daily P&L.

Simulates what would happen if we used ONLY the ML models to generate entry signals
from raw candle/greeks data, then managed exits with the production V5 FSM.

Outputs:
  - Per-trade results with ML confidence scores
  - Daily P&L breakdown (for the last 2 months)
  - Score bucket analysis (confidence → win rate → optimal sizing)
  - Total cumulative P&L

Usage:
    python scripts/backtest_ml_signals.py                    # all tickers, last 60 days
    python scripts/backtest_ml_signals.py --ticker SPY       # single ticker
    python scripts/backtest_ml_signals.py --days 90          # last 90 days
    python scripts/backtest_ml_signals.py --portfolio 23000  # custom portfolio size
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "signal_ml_v2"


def find_atm_strike(
    conn: sqlite3.Connection,
    ticker: str,
    dt: str,
    right: str,
) -> float | None:
    """Find the ATM strike for ticker/date/right using underlying price from greeks.

    For CALLs: nearest strike >= underlying (slightly OTM).
    For PUTs: nearest strike <= underlying (slightly OTM).
    """
    row = conn.execute(
        "SELECT underlying_price FROM option_greeks "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? AND underlying_price > 0 "
        "ORDER BY timestamp LIMIT 1",
        (ticker, f"{dt}%", right),
    ).fetchone()
    if not row:
        return None
    underlying = row[0]

    strikes = [r[0] for r in conn.execute(
        "SELECT DISTINCT strike FROM option_ohlc "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? ORDER BY strike",
        (ticker, f"{dt}%", right),
    ).fetchall()]
    if not strikes:
        return None

    if right.upper() == "CALL":
        otm = [s for s in strikes if s >= underlying]
        return min(otm, key=lambda s: s - underlying) if otm else min(strikes, key=lambda s: abs(s - underlying))
    else:
        otm = [s for s in strikes if s <= underlying]
        return max(otm, key=lambda s: underlying - s) if otm else min(strikes, key=lambda s: abs(s - underlying))

# Production V6 settings
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
    V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True,
    V6_DCA_TICKERS="IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# Blocked tickers (historically unprofitable)
BLOCKED = {"MSFT"}

# Per-ticker PUT restrictions (backtested 2026-05-21):
# PUTs are net-negative on these tickers. Only allow CALLs.
# AMD/META/PLTR/AVGO PUTs are profitable — keep both directions.
CALLS_ONLY_TICKERS = {"SPY", "QQQ", "TSLA", "AAPL", "GOOGL", "IWM", "AMZN", "META"}


# ---------------------------------------------------------------------------
# Simpsons-inspired filters (from Yank's n8n agent — 85.9% WR on Side A)
# ---------------------------------------------------------------------------

def compute_daily_trend(
    conn: sqlite3.Connection, ticker: str, dt: str, lookback_days: int = 60,
) -> str:
    """Classify daily trend as BULLISH/BEARISH/RANGING using EMA(20) vs EMA(50).

    From Simpsons Side B: price > EMA20 > EMA50 = BULLISH, opposite = BEARISH.
    """
    rows = conn.execute(
        "SELECT close FROM stock_ohlc WHERE ticker=? AND substr(timestamp,1,10) < ? "
        "GROUP BY substr(timestamp,1,10) ORDER BY substr(timestamp,1,10) DESC LIMIT ?",
        (ticker, dt, lookback_days),
    ).fetchall()

    if len(rows) < 50:
        return "RANGING"

    closes = [r[0] for r in reversed(rows)]  # oldest to newest

    def ema(data: list, period: int) -> float | None:
        if len(data) < period:
            return None
        k = 2.0 / (period + 1)
        val = sum(data[:period]) / period
        for d in data[period:]:
            val = d * k + val * (1 - k)
        return val

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    last = closes[-1]

    if e20 is None or e50 is None:
        return "RANGING"
    if last > e20 and e20 > e50:
        return "BULLISH"
    if last < e20 and e20 < e50:
        return "BEARISH"
    return "RANGING"


def compute_opening_range_direction(
    stock: pd.DataFrame,
) -> str:
    """Compute opening range direction from first 15 min of stock data.

    From Simpsons Side B: OR > +0.15% = BULLISH, < -0.15% = BEARISH.
    """
    if len(stock) < 3:
        return "FLAT"

    # First 3 bars (assuming 5-min data) or first 15 bars (1-min data)
    # Detect bar frequency from timestamps
    try:
        ts0 = pd.Timestamp(stock.iloc[0]["timestamp"])
        ts1 = pd.Timestamp(stock.iloc[1]["timestamp"])
        interval_min = max(1, int((ts1 - ts0).total_seconds() / 60))
    except Exception:
        interval_min = 1

    or_bars = max(1, 15 // interval_min)
    or_data = stock.iloc[:or_bars]

    if len(or_data) < 1:
        return "FLAT"

    first_open = or_data.iloc[0].get("open", 0)
    last_close = or_data.iloc[-1].get("close", 0)

    if not first_open or first_open <= 0:
        return "FLAT"

    pct = (last_close - first_open) / first_open * 100

    if pct > 0.15:
        return "BULLISH"
    elif pct < -0.15:
        return "BEARISH"
    return "FLAT"


def simpsons_entry_filter(
    ohlc_side: pd.DataFrame,
    stock: pd.DataFrame,
    idx: int,
    right: str,
    daily_trend: str,
    or_direction: str,
) -> tuple[bool, str]:
    """Apply Simpsons-inspired entry filters. Returns (passed, reason).

    Filters applied:
    1. Daily trend must not oppose trade direction (from Side B F2)
    2. Opening range must not oppose trade direction (from Side B F1)
    3. Fire bar quality: trigger bar should move favorably (from Side A v10)
    4. Pre-30m momentum: last 30 min should favor trade direction (from Side A v10)
    5. Volume ratio: fire bar should have meaningful volume (from Side A v10)
    """
    is_call = right.upper() == "CALL"

    # F1: Daily trend gate (from Side B)
    if is_call and daily_trend == "BEARISH":
        return False, "daily_trend_against"
    if not is_call and daily_trend == "BULLISH":
        return False, "daily_trend_against"

    # F2: Opening range gate (from Side B)
    if is_call and or_direction == "BEARISH":
        return False, "or_direction_against"
    if not is_call and or_direction == "BULLISH":
        return False, "or_direction_against"

    # Fire bar quality: the bar at idx-1 (decision candle) should have moved favorably
    if idx >= 2 and idx < len(ohlc_side):
        fb = ohlc_side.iloc[idx - 1]
        fb_open = fb.get("open", 0) or 0
        fb_close = fb.get("close", 0) or 0
        if fb_open > 0:
            fb_move_pct = (fb_close - fb_open) / fb_open * 100
            fire_bar_fav = fb_move_pct if is_call else -fb_move_pct
            # Skip if fire bar strongly against us (from Side A S12_F)
            if fire_bar_fav < -0.15:
                return False, "fire_bar_against"

    # Pre-30m momentum: 6 bars (5-min) or 30 bars (1-min) before signal
    # Should show favorable momentum
    lookback = min(30, idx - 1)
    if lookback >= 3:
        start_close = ohlc_side.iloc[idx - lookback].get("close", 0)
        end_close = ohlc_side.iloc[idx - 1].get("close", 0)
        if start_close and start_close > 0:
            pre_move_pct = (end_close - start_close) / start_close * 100
            pre_30m_fav = pre_move_pct if is_call else -pre_move_pct
            # Skip if strong adverse momentum in last 30 candles
            # (from Side A S11_A: pre_30m_fav < 0 + low vol = skip)
            if pre_30m_fav < -2.0:  # 2% adverse premium move in last 30 min
                return False, "pre_30m_adverse"

    # Volume ratio: fire bar volume vs prior hour average
    if idx >= 13:
        vols = ohlc_side.iloc[idx - 13:idx - 1]["volume"].fillna(0).values
        fire_vol = ohlc_side.iloc[idx - 1].get("volume", 0) or 0
        avg_vol = np.mean(vols) if len(vols) > 0 else 1
        if avg_vol > 0:
            vol_ratio = fire_vol / avg_vol
            # Skip if volume is dead (from Side A S11_A/S11_F)
            if vol_ratio < 0.3:
                return False, "dead_volume"

    return True, "passed"

# Portfolio sizing
DEFAULT_PORTFOLIO = 23000
MAX_RISK_PCT = 75
MAX_CONCURRENT = 5
MAX_POSITION_PCT = 15

# Confidence-weighted sizing multipliers (backtested 2026-05-21)
# 0.70-0.80 = sweet spot (76% WR, +$13.8K) → full allocation
# 0.80-0.90 = weakest bucket (58% WR, -$1K) → minimum allocation
# 0.90+     = big movers (64% WR, +$10K) → near-full
CONFIDENCE_TIERS = [
    (0.90, 0.95),   # 95% budget — big movers, runners justify near-full
    (0.80, 0.60),   # 60% budget — weakest bucket (58% WR, negative P&L)
    (0.70, 1.00),   # 100% budget — sweet spot, highest WR + P&L
]
FLAT_MULT = 0.85  # fallback if confidence weighting disabled

# ML confidence thresholds for entry
MIN_CONFIDENCE = 0.70  # minimum ML probability to enter (70% = meaningful signal)


@dataclass
class Trade:
    ticker: str
    direction: str  # CALL or PUT
    entry_price: float
    entry_time: datetime
    contracts: int
    ml_confidence: float
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    peak_gain: float = 0.0
    hold_minutes: int = 0


def load_model(ticker: str):
    """Load per-ticker or GENERIC model."""
    import lightgbm as lgb

    ticker_path = MODEL_DIR / f"signal_{ticker}.lgb"
    generic_path = MODEL_DIR / "signal_GENERIC.lgb"

    if ticker_path.exists():
        return lgb.Booster(model_file=str(ticker_path)), ticker
    elif generic_path.exists():
        return lgb.Booster(model_file=str(generic_path)), "GENERIC"
    else:
        return None, None


def compute_features_for_prediction(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    stock: pd.DataFrame,
    idx: int,
    lookback: int = 15,
) -> dict | None:
    """Compute features for ML prediction at idx (uses idx-1 as decision candle)."""
    if idx < lookback + 1 or idx >= len(ohlc):
        return None

    curr = ohlc.iloc[idx - 1]  # decision candle
    window = ohlc.iloc[max(0, idx - lookback - 1):idx]
    entry_price = curr.get("close", 0) or 0
    if entry_price <= 0:
        return None

    f = {}

    # Time of day
    try:
        ts = pd.Timestamp(curr["timestamp"])
        if ts.tzinfo:
            ts = ts.tz_convert("America/New_York")
        f["minutes_since_open"] = max(0, (ts.hour - 9) * 60 + ts.minute - 30)
    except Exception:
        f["minutes_since_open"] = 0
    f["hour_bucket"] = f["minutes_since_open"] // 60
    f["is_first_30min"] = 1 if f["minutes_since_open"] <= 30 else 0
    f["is_last_hour"] = 1 if f["minutes_since_open"] >= 330 else 0

    # Premium action
    prices = window["close"].dropna().values
    if len(prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = float((prices[-1] / prices[max(-6, -len(prices))] - 1) * 100) if prices[max(-6, -len(prices))] > 0 else 0
    f["premium_change_10m"] = float((prices[-1] / prices[max(-11, -len(prices))] - 1) * 100) if prices[max(-11, -len(prices))] > 0 else 0
    f["premium_change_15m"] = float((prices[-1] / prices[0] - 1) * 100) if prices[0] > 0 else 0

    if len(prices) > 2 and all(prices[:-1] > 0):
        returns = np.diff(prices) / prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
    else:
        f["premium_volatility"] = 0

    # Volume
    vols = window["volume"].fillna(0).values if "volume" in window.columns else np.zeros(len(window))
    f["current_volume"] = float(vols[-1])
    avg_vol = float(np.mean(vols[:-1])) if len(vols) > 1 else 1
    f["volume_ratio"] = float(vols[-1] / max(avg_vol, 1))
    f["volume_trend"] = float(np.mean(vols[-5:]) / max(np.mean(vols[:max(len(vols)-5, 1)]), 1)) if len(vols) > 5 else 1.0
    if len(vols) > 5 and np.std(vols[:-1]) > 0:
        f["volume_zscore"] = float((vols[-1] - np.mean(vols[:-1])) / np.std(vols[:-1]))
    else:
        f["volume_zscore"] = 0

    # Bid/ask
    if len(quotes) > idx - 1:
        q_window = quotes.iloc[max(0, idx - lookback - 1):idx]
        if len(q_window) > 0:
            q = q_window.iloc[-1]
            bid = q.get("bid", 0) or 0
            ask = q.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price
            f["spread"] = float(ask - bid) if ask > bid else 0
            f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0
            if len(q_window) > 3:
                spreads = (q_window["ask"].fillna(0) - q_window["bid"].fillna(0)).values
                spreads = spreads[spreads >= 0]
                if len(spreads) > 3:
                    first_half = spreads[:len(spreads) // 2].mean()
                    second_half = spreads[len(spreads) // 2:].mean()
                    f["spread_tightening"] = float(first_half - second_half)
                else:
                    f["spread_tightening"] = 0
            else:
                f["spread_tightening"] = 0
            f["bid_size"] = float(q.get("bid_size", 0) or 0)
            f["ask_size"] = float(q.get("ask_size", 0) or 0)
            f["size_imbalance"] = float((f["bid_size"] - f["ask_size"]) / max(f["bid_size"] + f["ask_size"], 1))
        else:
            for k in ["spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance"]:
                f[k] = 0
    else:
        for k in ["spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance"]:
            f[k] = 0

    # Greeks
    if len(greeks) > idx - 1:
        g_window = greeks.iloc[max(0, idx - lookback - 1):idx]
        if len(g_window) > 0:
            g = g_window.iloc[-1]
            f["iv"] = float(g.get("implied_vol", 0) or 0)
            f["delta"] = float(abs(g.get("delta", 0) or 0))
            f["theta"] = float(g.get("theta", 0) or 0)
            f["vega"] = float(g.get("vega", 0) or 0)
            if len(g_window) > 3 and g_window["implied_vol"].notna().sum() > 3:
                ivs = g_window["implied_vol"].dropna().values
                f["iv_change_5m"] = float(ivs[-1] - ivs[max(-6, -len(ivs))]) if len(ivs) > 5 else 0
                f["iv_change_15m"] = float(ivs[-1] - ivs[0])
                f["iv_trend"] = float(np.polyfit(range(len(ivs)), ivs, 1)[0]) if len(ivs) > 2 else 0
            else:
                f["iv_change_5m"] = 0
                f["iv_change_15m"] = 0
                f["iv_trend"] = 0
            f["underlying_price"] = float(g.get("underlying_price", 0) or 0)
        else:
            for k in ["iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend", "underlying_price"]:
                f[k] = 0
    else:
        for k in ["iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend", "underlying_price"]:
            f[k] = 0

    # Underlying
    if len(stock) > 0:
        s_window = stock.iloc[max(0, min(idx - 1, len(stock)) - lookback):min(idx, len(stock))]
        if len(s_window) > 1:
            s_closes = s_window["close"].dropna().values
            if len(s_closes) > 1 and all(s_closes > 0):
                f["underlying_change_5m"] = float((s_closes[-1] / s_closes[max(-6, -len(s_closes))] - 1) * 100)
                f["underlying_change_15m"] = float((s_closes[-1] / s_closes[0] - 1) * 100)
                f["underlying_volatility"] = float(np.std(np.diff(s_closes) / s_closes[:-1]) * 100)
                vwap = np.mean(s_closes)
                f["vwap_deviation"] = float((s_closes[-1] / vwap - 1) * 100) if vwap > 0 else 0
            else:
                for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
                    f[k] = 0
        else:
            for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
                f[k] = 0
    else:
        for k in ["underlying_change_5m", "underlying_change_15m", "underlying_volatility", "vwap_deviation"]:
            f[k] = 0

    # Computed patterns
    f["coiled_spring"] = 1 if (f["premium_volatility"] < 2 and f["volume_ratio"] > 1.5) else 0
    f["iv_expanding"] = 1 if (f.get("iv_change_5m", 0) > 0.02) else 0

    return f


# Feature columns (must match training)
FEATURE_COLS = [
    "minutes_since_open", "hour_bucket", "is_first_30min", "is_last_hour",
    "premium", "premium_change_5m", "premium_change_10m", "premium_change_15m",
    "premium_volatility",
    "current_volume", "volume_ratio", "volume_trend", "volume_zscore",
    "spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance",
    "iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend",
    "underlying_price", "underlying_change_5m", "underlying_change_15m",
    "underlying_volatility", "vwap_deviation",
    "coiled_spring", "iv_expanding",
]


def simulate_trade_fsm(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    entry_idx: int,
    ticker: str,
    expiry_date: str = "",
    contracts: int = 5,
) -> dict | None:
    """Run V5 FSM from entry_idx forward, return trade outcome."""
    from zoneinfo import ZoneInfo

    entry_price = ohlc.iloc[entry_idx]["close"]
    if not entry_price or entry_price <= 0:
        return None

    try:
        entry_ts = pd.Timestamp(ohlc.iloc[entry_idx]["timestamp"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("America/New_York")
        else:
            entry_ts = entry_ts.tz_convert("America/New_York")
    except Exception:
        return None

    config = get_ticker_config(ticker, use_per_ticker=True)

    trade_state = TradeState(
        trade_id=0,
        ticker=ticker,
        option_type="CALL",
        entry_premium=float(entry_price),
        contracts=contracts,
        entry_time=entry_ts,
        dte=0,
        expiry_date=expiry_date,
    )

    fsm = ExitFSM(cfg=config, settings=_V6_SETTINGS)
    peak_gain = 0.0

    for j in range(entry_idx + 1, min(entry_idx + 390, len(ohlc))):
        row = ohlc.iloc[j]
        price = row["close"]
        if not price or price <= 0:
            continue

        try:
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
        except Exception:
            continue

        gain_pct = (price / entry_price - 1) * 100
        peak_gain = max(peak_gain, gain_pct)

        # Get bid/ask for the FSM
        bid = float(price)
        ask = float(price)
        if len(quotes) > j:
            b = quotes.iloc[j].get("bid", 0)
            a = quotes.iloc[j].get("ask", 0)
            if b and b > 0:
                bid = float(b)
            if a and a > 0:
                ask = float(a)

        # Get underlying
        underlying = 0.0
        if len(greeks) > j:
            u = greeks.iloc[j].get("underlying_price", 0)
            if u and u > 0:
                underlying = float(u)

        # Minutes to close (market closes at 4PM = 390min after 9:30)
        minutes_to_close = max(0, 390 - (ts.hour - 9) * 60 - ts.minute + 30)

        action = fsm.evaluate(
            trade_state, float(price), bid, ask, ts,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action and action.should_exit:
            exit_price = float(price)
            pnl_pct = (exit_price / entry_price - 1) * 100
            hold_min = int((ts - entry_ts).total_seconds() / 60)
            return {
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_dollars": (exit_price - entry_price) * contracts * 100,
                "peak_gain": peak_gain,
                "hold_minutes": hold_min,
                "reason": action.reason.value if hasattr(action.reason, "value") else str(action.reason),
            }

    # EOD — force close at last price
    if len(ohlc) > entry_idx + 1:
        last_price = ohlc.iloc[-1]["close"]
        if last_price and last_price > 0:
            pnl_pct = (last_price / entry_price - 1) * 100
            return {
                "exit_price": float(last_price),
                "pnl_pct": pnl_pct,
                "pnl_dollars": (last_price - entry_price) * contracts * 100,
                "peak_gain": peak_gain,
                "hold_minutes": 390,
                "reason": "eod_force",
            }

    return None


def confidence_to_mult(ml_confidence: float) -> float:
    """Map ML confidence to budget multiplier using backtested tiers.

    Sweet spot is 0.70-0.80 (71.9% WR) → gets full allocation.
    0.80-0.90 is weakest qualifying bucket → reduced allocation.
    0.90+ has big runners → near-full allocation.
    """
    for threshold, mult in CONFIDENCE_TIERS:
        if ml_confidence >= threshold:
            return mult
    return CONFIDENCE_TIERS[-1][1]  # lowest tier


def size_position(ml_confidence: float, entry_price: float, portfolio: float) -> int:
    """Calculate contracts using confidence-weighted sizing."""
    deployable = portfolio * MAX_RISK_PCT / 100
    per_slot = deployable / MAX_CONCURRENT
    mult = confidence_to_mult(ml_confidence)
    scaled = per_slot * mult
    position_cap = portfolio * MAX_POSITION_PCT / 100

    budget = min(scaled, position_cap)
    cost_per_contract = entry_price * 100
    if cost_per_contract <= 0 or np.isnan(cost_per_contract):
        return 1

    contracts = int(budget / cost_per_contract)
    if np.isnan(contracts):
        return 1
    return max(1, contracts)


def run_backtest(
    conn: sqlite3.Connection,
    tickers: list[str],
    days: int = 60,
    portfolio: float = DEFAULT_PORTFOLIO,
    min_confidence: float = MIN_CONFIDENCE,
    scan_interval: int = 15,  # check ML every N minutes
) -> list[Trade]:
    """Run end-to-end ML backtest across all tickers."""
    import lightgbm as lgb

    # Load models
    models = {}
    for ticker in tickers:
        model, source = load_model(ticker)
        if model:
            models[ticker] = (model, source)
            print(f"  Loaded model for {ticker} (source: {source})")

    if not models:
        print("ERROR: No models found!")
        return []

    # Get date range
    all_dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc ORDER BY 1 DESC"
    ).fetchall()]

    if not all_dates:
        print("ERROR: No data in DB")
        return []

    # Use last N days
    dates = sorted(all_dates[:days])
    print(f"\n  Backtesting {len(dates)} days: {dates[0]} → {dates[-1]}")
    print(f"  Portfolio: ${portfolio:,.0f} | Min confidence: {min_confidence:.0%}")
    print(f"  Scan interval: every {scan_interval} min | Max concurrent: {MAX_CONCURRENT}")
    print()

    all_trades = []
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(int)

    for dt_idx, dt in enumerate(dates):
        open_count = 0  # simulated concurrent positions
        ticker_traded_today = set()  # max 1 trade per ticker per day

        for ticker in tickers:
            if ticker in BLOCKED:
                continue
            if ticker not in models:
                continue
            if ticker in ticker_traded_today:
                continue

            model, _ = models[ticker]

            # Load stock data (not strike-specific)
            stock = pd.read_sql_query(
                "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
                conn, params=(ticker, f"{dt}%"),
            )

            # Simpsons filters: compute daily trend + opening range direction (once per ticker/day)
            daily_trend = compute_daily_trend(conn, ticker, dt)
            or_direction = compute_opening_range_direction(stock)

            # Scan for CALL and PUT separately (with per-ticker direction filter)
            directions = ["CALL", "PUT"]
            if ticker in CALLS_ONLY_TICKERS:
                directions = ["CALL"]  # PUTs are net-negative on these tickers
            for right in directions:
                # Pick ATM strike to avoid multi-strike data contamination
                strike = find_atm_strike(conn, ticker, dt, right)
                if strike is None:
                    continue

                ohlc_side = pd.read_sql_query(
                    "SELECT * FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
                    conn, params=(ticker, f"{dt}%", right, strike),
                )
                quotes_side = pd.read_sql_query(
                    "SELECT * FROM option_quotes WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
                    conn, params=(ticker, f"{dt}%", right, strike),
                )
                greeks_side = pd.read_sql_query(
                    "SELECT * FROM option_greeks WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
                    conn, params=(ticker, f"{dt}%", right, strike),
                )

                if len(ohlc_side) < 30:
                    continue

                # Scan every scan_interval minutes for ML signals
                last_entry_idx = -30  # cooldown
                for idx in range(16, len(ohlc_side) - 10, scan_interval):
                    if idx - last_entry_idx < 30:
                        continue
                    if open_count >= MAX_CONCURRENT:
                        break

                    # Compute features and get ML prediction
                    features = compute_features_for_prediction(
                        ohlc_side, quotes_side, greeks_side, stock, idx
                    )
                    if features is None:
                        continue

                    # --- ENTRY FILTERS (production pipeline) ---

                    entry_price = ohlc_side.iloc[idx]["close"]
                    if not entry_price or entry_price <= 0 or np.isnan(entry_price):
                        continue

                    # Premium cap (V6): block entries with premium > $6
                    if entry_price > 6.0:
                        continue

                    # Minimum premium: reject pennies (theta death)
                    if entry_price < 0.10:
                        continue

                    # Time filter: skip early afternoon danger zone (1:30-3:00 PM ET)
                    # Simpsons data: 36% WR in this window vs 72% during NY Open
                    mins = features.get("minutes_since_open", 0)
                    if 240 <= mins <= 330:
                        continue

                    # Skip last 15 min (EOD cutoff, insufficient time for move)
                    if mins >= 375:
                        continue

                    # Spread gate: reject wide spreads (> 40% of premium)
                    spread_pct = features.get("spread_pct", 0)
                    if spread_pct > 40:
                        continue

                    # Early pop gate: don't buy into a spike (wait for pullback)
                    premium_change_5m = features.get("premium_change_5m", 0)
                    if premium_change_5m > 30:  # >30% spike in 5min = chasing
                        continue

                    # Simpsons filters: daily trend, OR direction, fire bar, pre-30m, volume
                    simp_passed, simp_reason = simpsons_entry_filter(
                        ohlc_side, stock, idx, right, daily_trend, or_direction,
                    )
                    if not simp_passed:
                        continue

                    # PUT confirmation gate: PUTs need stronger evidence
                    # (17/23 "never positive" trades are PUTs with high ML confidence
                    #  but no actual downward momentum — model is miscalibrated on PUTs)
                    if right == "PUT" and idx >= 5:
                        # Require underlying to be actually dropping (not just ML prediction)
                        recent_stock_start = stock.iloc[max(0, idx - 5)]["close"] if idx < len(stock) else 0
                        recent_stock_end = stock.iloc[min(idx, len(stock) - 1)]["close"] if idx < len(stock) else 0
                        if recent_stock_start > 0 and recent_stock_end > 0:
                            stock_move = (recent_stock_end - recent_stock_start) / recent_stock_start * 100
                            # For PUTs: stock must be falling (or at least not rising)
                            if stock_move > 0.1:  # stock is up → don't buy puts
                                continue

                    # Build feature vector and get ML prediction
                    X = np.array([[features.get(c, 0) for c in FEATURE_COLS]])
                    confidence = float(model.predict(X)[0])

                    if confidence < min_confidence:
                        continue

                    # Session timing bonus/penalty (from Simpsons analysis)
                    session_mult = 1.0
                    if mins <= 60:       # NY Open Killzone: best window
                        session_mult = 1.1
                    elif mins <= 150:    # Late morning: good
                        session_mult = 1.0
                    elif mins <= 240:    # Midday: neutral
                        session_mult = 0.9
                    # (early afternoon already filtered out above)
                    elif mins >= 330:    # Power hour: decent
                        session_mult = 1.0

                    # Adjusted confidence with session timing
                    adjusted_confidence = confidence * session_mult

                    # Enter at the signal candle (no look-ahead bias)
                    # The ML model was trained on entries at THIS exact candle
                    contracts = size_position(adjusted_confidence, entry_price, portfolio)
                    result = simulate_trade_fsm(
                        ohlc_side, quotes_side, greeks_side,
                        entry_idx=idx, ticker=ticker, expiry_date=dt,
                        contracts=contracts,
                    )

                    if result is None:
                        continue

                    # Skip NaN results
                    if np.isnan(result["pnl_dollars"]) or np.isnan(result["exit_price"]):
                        continue

                    try:
                        entry_time = pd.Timestamp(ohlc_side.iloc[idx]["timestamp"])
                    except Exception:
                        entry_time = datetime.now()

                    trade = Trade(
                        ticker=ticker,
                        direction=right,
                        entry_price=float(entry_price),
                        entry_time=entry_time,
                        contracts=contracts,
                        ml_confidence=adjusted_confidence,
                        exit_price=result["exit_price"],
                        pnl_dollars=result["pnl_dollars"],
                        pnl_pct=result["pnl_pct"],
                        peak_gain=result["peak_gain"],
                        hold_minutes=result["hold_minutes"],
                        exit_reason=result["reason"],
                    )
                    all_trades.append(trade)
                    daily_pnl[dt] += trade.pnl_dollars
                    daily_trades[dt] += 1
                    open_count += 1
                    last_entry_idx = idx
                    ticker_traded_today.add(ticker)
                    break  # 1 trade per ticker per day per side

        if (dt_idx + 1) % 10 == 0:
            print(f"  ... {dt_idx + 1}/{len(dates)} days processed ({len(all_trades)} trades so far)")

    return all_trades


def print_results(trades: list[Trade], portfolio: float):
    """Print comprehensive backtest results."""
    if not trades:
        print("\nNo trades generated!")
        return

    print(f"\n{'='*90}")
    print("ML SIGNAL BACKTEST — END-TO-END RESULTS")
    print(f"{'='*90}")

    # Summary
    total_pnl = sum(t.pnl_dollars for t in trades)
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers = [t for t in trades if t.pnl_dollars <= 0]
    win_rate = len(winners) / len(trades) * 100

    print(f"\n  Total Trades:  {len(trades)}")
    print(f"  Win Rate:      {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
    print(f"  Total P&L:     ${total_pnl:,.0f} ({total_pnl/portfolio*100:+.1f}% of portfolio)")
    print(f"  Avg Win:       ${np.mean([t.pnl_dollars for t in winners]):,.0f}" if winners else "  Avg Win:       N/A")
    print(f"  Avg Loss:      ${np.mean([t.pnl_dollars for t in losers]):,.0f}" if losers else "  Avg Loss:      N/A")
    print(f"  Avg Hold:      {np.mean([t.hold_minutes for t in trades]):.0f} min")
    print(f"  Avg Peak:      {np.mean([t.peak_gain for t in trades]):.1f}%")

    # Daily P&L
    print(f"\n{'='*90}")
    print("DAILY P&L BREAKDOWN")
    print(f"{'='*90}")
    print(f"{'Date':<12} {'Trades':>6} {'P&L':>10} {'Cumulative':>12} {'W/L':>6}")
    print("-" * 50)

    daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in trades:
        dt = str(t.entry_time)[:10]
        daily[dt]["pnl"] += t.pnl_dollars
        daily[dt]["trades"] += 1
        if t.pnl_dollars > 0:
            daily[dt]["wins"] += 1

    cumulative = 0.0
    for dt in sorted(daily.keys()):
        d = daily[dt]
        cumulative += d["pnl"]
        wl = f"{d['wins']}/{d['trades']-d['wins']}"
        print(f"  {dt:<10} {d['trades']:>6} ${d['pnl']:>+9,.0f} ${cumulative:>+11,.0f} {wl:>6}")

    # Score bucket analysis
    print(f"\n{'='*90}")
    print("ML CONFIDENCE → WIN RATE (for score-weighted sizing)")
    print(f"{'='*90}")
    print(f"{'Confidence':>12} {'Mult':>5} {'Trades':>7} {'WR':>6} {'Avg P&L':>10} {'Total P&L':>12} {'Avg Peak':>9}")
    print("-" * 75)

    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        bucket_trades = [t for t in trades if lo <= t.ml_confidence < hi]
        if not bucket_trades:
            print(f"  {lo:.1f}-{hi:.1f}      {'—':>7}")
            continue
        bt_wins = [t for t in bucket_trades if t.pnl_dollars > 0]
        wr = len(bt_wins) / len(bucket_trades) * 100
        avg_pnl = np.mean([t.pnl_dollars for t in bucket_trades])
        total = sum(t.pnl_dollars for t in bucket_trades)
        avg_peak = np.mean([t.peak_gain for t in bucket_trades])
        mult = confidence_to_mult((lo + hi) / 2)
        print(f"  {lo:.1f}-{hi:.1f}    {mult:>4.0%} {len(bucket_trades):>7} {wr:>5.1f}% ${avg_pnl:>+9,.0f} ${total:>+11,.0f} {avg_peak:>8.1f}%")

    # Per-ticker breakdown
    print(f"\n{'='*90}")
    print("PER-TICKER RESULTS")
    print(f"{'='*90}")
    print(f"{'Ticker':<8} {'Trades':>7} {'WR':>6} {'P&L':>10} {'Avg Conf':>9}")
    print("-" * 45)

    ticker_groups = defaultdict(list)
    for t in trades:
        ticker_groups[t.ticker].append(t)

    for ticker in sorted(ticker_groups.keys(), key=lambda x: -sum(t.pnl_dollars for t in ticker_groups[x])):
        tg = ticker_groups[ticker]
        wins = [t for t in tg if t.pnl_dollars > 0]
        wr = len(wins) / len(tg) * 100
        total = sum(t.pnl_dollars for t in tg)
        avg_conf = np.mean([t.ml_confidence for t in tg])
        print(f"  {ticker:<6} {len(tg):>7} {wr:>5.1f}% ${total:>+9,.0f} {avg_conf:>8.1%}")

    # Exit reason distribution
    print(f"\n{'='*90}")
    print("EXIT REASONS")
    print(f"{'='*90}")
    reason_counts = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades:
        reason_counts[t.exit_reason]["count"] += 1
        reason_counts[t.exit_reason]["pnl"] += t.pnl_dollars

    for reason in sorted(reason_counts.keys(), key=lambda x: -reason_counts[x]["count"]):
        rc = reason_counts[reason]
        print(f"  {reason:<25} {rc['count']:>5} trades  ${rc['pnl']:>+10,.0f}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end ML signal backtest")
    parser.add_argument("--ticker", type=str, help="Single ticker to backtest")
    parser.add_argument("--days", type=int, default=60, help="Number of days to backtest (default: 60)")
    parser.add_argument("--portfolio", type=float, default=DEFAULT_PORTFOLIO, help="Portfolio size")
    parser.add_argument("--confidence", type=float, default=MIN_CONFIDENCE, help="Min ML confidence (default: 0.50)")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in minutes (default: 15)")
    parser.add_argument("--db", type=str, default=THETADATA_DB, help="Path to thetadata DB")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"ML Signal Backtest — {args.days} days")
    print(f"  DB: {args.db}")
    print(f"  Tickers: {', '.join(tickers)}")

    conn = sqlite3.connect(args.db)
    trades = run_backtest(
        conn, tickers,
        days=args.days,
        portfolio=args.portfolio,
        min_confidence=args.confidence,
        scan_interval=args.interval,
    )
    conn.close()

    print_results(trades, args.portfolio)

    # Save trades to CSV
    if trades:
        csv_path = PROJECT_DIR / "journal" / f"backtest_ml_{datetime.now():%Y%m%d_%H%M}.csv"
        df = pd.DataFrame([{
            "date": str(t.entry_time)[:10],
            "ticker": t.ticker,
            "direction": t.direction,
            "entry": t.entry_price,
            "exit": t.exit_price,
            "contracts": t.contracts,
            "confidence": t.ml_confidence,
            "pnl_dollars": t.pnl_dollars,
            "pnl_pct": t.pnl_pct,
            "peak_gain": t.peak_gain,
            "hold_min": t.hold_minutes,
            "exit_reason": t.exit_reason,
        } for t in trades])
        df.to_csv(csv_path, index=False)
        print(f"\n  Trades saved to {csv_path}")


if __name__ == "__main__":
    main()
