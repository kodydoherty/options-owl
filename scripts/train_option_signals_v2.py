"""V2: Per-ticker ML signal models — learn what option price patterns precede profitable moves.

Instead of relying on Discord analysts or rule-based scoring, this learns directly from
historical option prices what patterns precede profitable trades.

Uses the REAL production V5 FSM exit engine to determine trade outcomes — not simplified
thresholds. If the FSM code changes, labels automatically reflect production behavior.

Pipeline:
1. Scan historical 1-min option data to find every profitable move (per V5 FSM)
2. Capture what the market looked like 5-15 min BEFORE the move started
3. Label as "positive" (FSM exit = profit) or "negative" (random non-move timestamps)
4. Train per-ticker LightGBM models to recognize pre-move setups

Usage:
    python scripts/train_option_signals_v2.py                     # train all tickers with data
    python scripts/train_option_signals_v2.py --ticker SPY        # single ticker
    python scripts/train_option_signals_v2.py --backtest          # P&L backtest with models
    python scripts/train_option_signals_v2.py --scan              # show what model finds today
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "signal_ml_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    # New tickers (added 2026-05-28)
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]

# Production V6 settings (must match docker-compose.yml)
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
    V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

# --- Move detection thresholds ---
MIN_MOVE_PCT = 38.0       # minimum premium gain to count as a "move" (calibrated via ablation study)
MOVE_WINDOW_MIN = 120     # look forward this many minutes for the move
PRE_MOVE_LOOKBACK = 15    # capture features from this many minutes before entry
NEGATIVE_RATIO = 2        # negatives per positive (class balancing)
COOLDOWN_MIN = 30         # min gap between detected moves (avoid overlap)

# Per-ticker MIN_MOVE_PCT (from ablation study — volatile tickers need higher threshold)
TICKER_MOVE_PCT = {
    "SPY": 38.0, "QQQ": 38.0, "IWM": 38.0,         # index: tighter moves
    "TSLA": 45.0, "MSTR": 45.0, "AMD": 45.0,        # high-vol: bigger moves
    "NVDA": 40.0, "AVGO": 40.0, "META": 40.0,        # tech: moderate-high
    "AAPL": 35.0, "MSFT": 35.0, "GOOGL": 35.0,       # mega-cap: tighter
    "AMZN": 38.0, "PLTR": 45.0,                       # varies
}

# Features to EXCLUDE from training (identified as leaky in ablation study)
LEAKY_FEATURES = {
    "range_position",      # proxies for "buy at the low" — hindsight bias
    "near_low",            # derived from range_position
    "near_high",            # derived from range_position
    "consecutive_up_bars", # captures mid-move momentum — partially leaky
    "consecutive_down_bars",
    "premium_skew",        # strong distribution separation but unstable
    # NOTE: is_call was removed from leaky list — combined model needs direction info
}


# Parallelism — use all available cores
N_WORKERS = min(os.cpu_count() or 4, 16)  # cap at 16 to avoid SQLite contention


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _process_date_inmemory(args_tuple: tuple) -> list[dict]:
    """Process a single (ticker, date, right) combo using pre-loaded in-memory data.

    No DB access — all data passed in via args. Pure CPU work (FSM simulation).
    Returns list of feature dicts with labels.
    """
    ticker, dt, right, min_move, ohlc_dict, quotes_dict, greeks_dict, stock_dict = args_tuple

    # Reconstruct DataFrames from dicts (pickle-safe transfer)
    ohlc_side = pd.DataFrame(ohlc_dict) if ohlc_dict else pd.DataFrame()
    quotes_side = pd.DataFrame(quotes_dict) if quotes_dict else pd.DataFrame()
    greeks_side = pd.DataFrame(greeks_dict) if greeks_dict else pd.DataFrame()
    stock = pd.DataFrame(stock_dict) if stock_dict else pd.DataFrame()

    if len(ohlc_side) < 30:
        return []

    # Single FSM pass — classify each entry as winner or loser
    rows = []
    last_idx = -COOLDOWN_MIN
    closes = ohlc_side["close"].values
    n = len(closes)

    for i in range(PRE_MOVE_LOOKBACK, n - 10):
        if i - last_idx < COOLDOWN_MIN:
            continue

        entry = closes[i]
        if not entry or entry <= 0 or (isinstance(entry, float) and np.isnan(entry)):
            continue

        result = simulate_with_production_fsm(
            ohlc_side, quotes_side, greeks_side, i,
            ticker=ticker, dte=0, expiry_date=dt,
        )
        if result is None:
            continue

        features = compute_setup_features(ohlc_side, quotes_side, greeks_side, stock, i)
        if not features:
            continue

        is_winner = result["pnl_pct"] >= min_move
        features["label"] = 1 if is_winner else 0
        features["peak_pct"] = result.get("peak_gain", result.get("pnl_pct", 0))
        features["hold_minutes"] = result.get("hold_minutes", 0)
        features["exit_reason"] = result.get("exit_reason", result.get("reason", ""))
        features["ticker"] = ticker
        features["date"] = dt
        features["right"] = right
        rows.append(features)
        last_idx = i

    return rows


# ---------------------------------------------------------------------------
# Strike selection — pick ATM strike to avoid multi-strike data contamination
# ---------------------------------------------------------------------------

def find_atm_strike(
    conn: sqlite3.Connection,
    ticker: str,
    dt: str,
    right: str,
) -> float | None:
    """Find the ATM strike for ticker/date/right using underlying price from greeks.

    Returns the strike closest to the underlying price, or None if no data.
    For CALLs: nearest strike >= underlying (slightly OTM, typical 0DTE entry).
    For PUTs: nearest strike <= underlying (slightly OTM).
    """
    # Get underlying price from greeks
    row = conn.execute(
        "SELECT underlying_price FROM option_greeks "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? AND underlying_price > 0 "
        "ORDER BY timestamp LIMIT 1",
        (ticker, f"{dt}%", right),
    ).fetchone()
    if not row:
        return None
    underlying = row[0]

    # Get available strikes
    strikes = [r[0] for r in conn.execute(
        "SELECT DISTINCT strike FROM option_ohlc "
        "WHERE ticker=? AND timestamp LIKE ? AND right=? ORDER BY strike",
        (ticker, f"{dt}%", right),
    ).fetchall()]
    if not strikes:
        return None

    # Pick ATM: closest strike to underlying
    # For calls: prefer slightly OTM (>= underlying). For puts: slightly OTM (<= underlying).
    if right.upper() == "CALL":
        otm = [s for s in strikes if s >= underlying]
        return min(otm, key=lambda s: s - underlying) if otm else min(strikes, key=lambda s: abs(s - underlying))
    else:
        otm = [s for s in strikes if s <= underlying]
        return max(otm, key=lambda s: underlying - s) if otm else min(strikes, key=lambda s: abs(s - underlying))


def load_single_strike_data(
    conn: sqlite3.Connection,
    ticker: str,
    dt: str,
    right: str,
    strike: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load OHLC, quotes, and greeks for a single strike (no multi-strike contamination)."""
    ohlc = pd.read_sql_query(
        "SELECT * FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    quotes = pd.read_sql_query(
        "SELECT * FROM option_quotes WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    greeks = pd.read_sql_query(
        "SELECT * FROM option_greeks WHERE ticker=? AND timestamp LIKE ? AND right=? AND strike=? ORDER BY timestamp",
        conn, params=(ticker, f"{dt}%", right, strike),
    )
    return ohlc, quotes, greeks


# ---------------------------------------------------------------------------
# Production V5 FSM simulation (same code running on all owlets)
# ---------------------------------------------------------------------------

def simulate_with_production_fsm(
    ohlc_side: pd.DataFrame,
    quotes_side: pd.DataFrame,
    greeks_side: pd.DataFrame,
    entry_idx: int,
    ticker: str,
    dte: int = 0,
    expiry_date: str = "",
    contracts: int = 5,
) -> dict | None:
    """Run the ACTUAL production ExitFSM against historical tick data starting at entry_idx.

    Returns {pnl_pct, pnl_dollars, reason, hold_minutes, exit_premium, peak_gain} or None.
    """
    if entry_idx >= len(ohlc_side) - 10:
        return None

    entry_price = ohlc_side.iloc[entry_idx]["close"]
    if not entry_price or entry_price <= 0 or np.isnan(entry_price):
        return None

    right_val = str(ohlc_side.iloc[entry_idx].get("right", "CALL")).upper()
    option_type = "call" if right_val == "CALL" else "put"

    # Parse entry timestamp
    entry_ts_raw = ohlc_side.iloc[entry_idx]["timestamp"]
    entry_ts = pd.Timestamp(entry_ts_raw)
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.tz_localize(None)
    entry_ts = entry_ts.to_pydatetime()

    # Get underlying price from greeks
    first_underlying = 0.0
    if len(greeks_side) > entry_idx:
        u = greeks_side.iloc[entry_idx].get("underlying_price", 0)
        if u and u > 0:
            first_underlying = float(u)

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)

    state = TradeState(
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=entry_price,
        entry_time=entry_ts,
        contracts=contracts,
        peak_premium=entry_price,
        entry_underlying_price=first_underlying,
        dte=dte,
        expiry_date=expiry_date,
    )

    locked_pnl = 0.0
    remaining = contracts

    end_idx = min(entry_idx + MOVE_WINDOW_MIN, len(ohlc_side))

    for idx in range(entry_idx + 1, end_idx):
        row = ohlc_side.iloc[idx]
        premium = row.get("close", 0)
        if not premium or premium <= 0 or (isinstance(premium, float) and np.isnan(premium)):
            continue

        # Get bid/ask from quotes if available
        bid, ask = premium, premium
        if len(quotes_side) > idx:
            q = quotes_side.iloc[idx]
            b = q.get("bid", 0)
            a = q.get("ask", 0)
            if b and not (isinstance(b, float) and np.isnan(b)) and b > 0:
                bid = float(b)
            if a and not (isinstance(a, float) and np.isnan(a)) and a > 0:
                ask = float(a)

        now_raw = row["timestamp"]
        now = pd.Timestamp(now_raw)
        if now.tzinfo is not None:
            now = now.tz_localize(None)
        now = now.to_pydatetime()

        # Underlying price from greeks
        underlying = first_underlying
        if len(greeks_side) > idx:
            u = greeks_side.iloc[idx].get("underlying_price", 0)
            if u and u > 0:
                underlying = float(u)

        # Minutes to close (market closes at 16:00 ET)
        et_hour = now.hour
        if et_hour >= 13:  # timestamps may be in UTC (13:30 UTC = 9:30 ET)
            et_hour = now.hour - 4
            if et_hour < 0:
                et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            # Scaleout: partial exit
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_price) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_price) / entry_price * 100
            pnl = locked_pnl + (premium - entry_price) * remaining * 100
            pnl_pct = (premium / entry_price - 1) * 100

            return {
                "pnl_pct": pnl_pct,
                "pnl_dollars": pnl,
                "reason": action.reason.value,
                "hold_minutes": elapsed,
                "exit_premium": premium,
                "peak_gain": peak_gain,
            }

    # End of window — force close at last available tick
    last_row = ohlc_side.iloc[end_idx - 1]
    last_prem = last_row.get("close", entry_price)
    if not last_prem or (isinstance(last_prem, float) and np.isnan(last_prem)):
        last_prem = entry_price

    last_ts = pd.Timestamp(last_row["timestamp"])
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_localize(None)
    elapsed = (last_ts.to_pydatetime() - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_price) / entry_price * 100
    pnl = locked_pnl + (last_prem - entry_price) * remaining * 100

    return {
        "pnl_pct": (last_prem / entry_price - 1) * 100,
        "pnl_dollars": pnl,
        "reason": "eod_data_end",
        "hold_minutes": elapsed,
        "exit_premium": last_prem,
        "peak_gain": peak_gain,
    }


# ---------------------------------------------------------------------------
# Step 1: Find profitable moves using REAL V5 FSM
# ---------------------------------------------------------------------------

def find_profitable_moves(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    ticker: str,
    min_move_pct: float = MIN_MOVE_PCT,
    cooldown: int = COOLDOWN_MIN,
    dte: int = 0,
    expiry_date: str = "",
) -> list[dict]:
    """Scan 1-min OHLC and run production V5 FSM at each timestamp to find profitable entries.

    Returns list of {idx, entry_price, pnl_pct, pnl_dollars, reason, hold_minutes, peak_gain}.
    """
    moves = []
    last_move_idx = -cooldown

    closes = ohlc["close"].values
    n = len(closes)

    for i in range(PRE_MOVE_LOOKBACK, n - 10):
        if i - last_move_idx < cooldown:
            continue

        entry = closes[i]
        if not entry or entry <= 0 or np.isnan(entry):
            continue

        # Run production FSM to determine if this entry would be profitable
        result = simulate_with_production_fsm(
            ohlc, quotes, greeks, i,
            ticker=ticker, dte=dte, expiry_date=expiry_date,
        )
        if result is None:
            continue

        if result["pnl_pct"] >= min_move_pct:
            moves.append({
                "idx": i,
                "entry_price": float(entry),
                "pnl_pct": result["pnl_pct"],
                "pnl_dollars": result["pnl_dollars"],
                "peak_gain": result["peak_gain"],
                "hold_minutes": result["hold_minutes"],
                "exit_reason": result["reason"],
                "right": ohlc.iloc[i].get("right", "CALL"),
            })
            last_move_idx = i

    return moves


# ---------------------------------------------------------------------------
# Step 2: Feature engineering (captures the "setup" before a move)
# ---------------------------------------------------------------------------

def compute_setup_features(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    stock: pd.DataFrame,
    idx: int,
    lookback: int = PRE_MOVE_LOOKBACK,
) -> dict | None:
    """Compute features that describe what the market looks like BEFORE timestamp idx.

    Fix A (ablation study): Uses idx-1 as the decision candle, window excludes idx.
    This prevents same-candle leakage — we can only know features from candles
    we've fully observed before deciding to enter at idx.
    """
    if idx < lookback + 1 or idx >= len(ohlc):
        return None

    # Decision candle is idx-1 (last fully observed candle before entry)
    curr = ohlc.iloc[idx - 1]
    window = ohlc.iloc[max(0, idx - lookback - 1):idx]  # up to but NOT including idx
    entry_price = curr.get("close", 0) or 0
    if entry_price <= 0:
        return None

    f = {}

    # --- Time of day ---
    try:
        ts = pd.Timestamp(curr["timestamp"])
        if ts.tzinfo:
            ts = ts.tz_convert("America/New_York")
        f["minutes_since_open"] = max(0, (ts.hour - 9) * 60 + ts.minute - 30)
    except Exception:
        f["minutes_since_open"] = 0
    f["hour_bucket"] = f["minutes_since_open"] // 60  # 0=first hour, 1=second, etc.
    f["is_first_30min"] = 1 if f["minutes_since_open"] <= 30 else 0
    f["is_last_hour"] = 1 if f["minutes_since_open"] >= 330 else 0

    # --- Premium action (what the option price is doing) ---
    prices = window["close"].dropna().values
    if len(prices) < 3:
        return None

    f["premium"] = float(entry_price)
    f["premium_change_5m"] = float((prices[-1] / prices[max(-6, -len(prices))] - 1) * 100) if prices[max(-6, -len(prices))] > 0 else 0
    f["premium_change_10m"] = float((prices[-1] / prices[max(-11, -len(prices))] - 1) * 100) if prices[max(-11, -len(prices))] > 0 else 0
    f["premium_change_15m"] = float((prices[-1] / prices[0] - 1) * 100) if prices[0] > 0 else 0

    # Volatility of premium (noisy = uncertain, calm = coiled)
    if len(prices) > 2 and all(prices[:-1] > 0):
        returns = np.diff(prices) / prices[:-1]
        f["premium_volatility"] = float(np.std(returns) * 100)
        f["premium_skew"] = float(pd.Series(returns).skew()) if len(returns) > 3 else 0
    else:
        f["premium_volatility"] = 0
        f["premium_skew"] = 0

    # Range position (where are we within recent range)
    f["range_position"] = float((prices[-1] - prices.min()) / (prices.max() - prices.min())) if prices.max() > prices.min() else 0.5

    # Is premium near its low? (bounce setup)
    f["near_low"] = 1 if f["range_position"] < 0.25 else 0
    # Is premium breaking out? (momentum setup)
    f["near_high"] = 1 if f["range_position"] > 0.85 else 0

    # Consecutive up/down bars
    if len(prices) > 1:
        diffs = np.diff(prices)
        consecutive_up = 0
        for d in reversed(diffs):
            if d > 0:
                consecutive_up += 1
            else:
                break
        consecutive_down = 0
        for d in reversed(diffs):
            if d < 0:
                consecutive_down += 1
            else:
                break
        f["consecutive_up_bars"] = consecutive_up
        f["consecutive_down_bars"] = consecutive_down
    else:
        f["consecutive_up_bars"] = 0
        f["consecutive_down_bars"] = 0

    # --- Volume pattern ---
    vols = window["volume"].fillna(0).values if "volume" in window.columns else np.zeros(len(window))
    f["current_volume"] = float(vols[-1])
    avg_vol = float(np.mean(vols[:-1])) if len(vols) > 1 else 1
    f["volume_ratio"] = float(vols[-1] / max(avg_vol, 1))
    f["volume_trend"] = float(np.mean(vols[-5:]) / max(np.mean(vols[:max(len(vols)-5, 1)]), 1)) if len(vols) > 5 else 1.0

    # Volume spike detection (is volume unusually high?)
    if len(vols) > 5 and np.std(vols[:-1]) > 0:
        f["volume_zscore"] = float((vols[-1] - np.mean(vols[:-1])) / np.std(vols[:-1]))
    else:
        f["volume_zscore"] = 0

    # --- Bid/ask dynamics ---
    if len(quotes) > idx - 1:
        q_window = quotes.iloc[max(0, idx - lookback - 1):idx]  # exclude entry candle
        if len(q_window) > 0:
            q = q_window.iloc[-1]
            bid = q.get("bid", 0) or 0
            ask = q.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else entry_price

            f["spread"] = float(ask - bid) if ask > bid else 0
            f["spread_pct"] = float(f["spread"] / mid * 100) if mid > 0 else 0

            # Spread trend (tightening = institutions entering)
            if len(q_window) > 3:
                spreads = (q_window["ask"].fillna(0) - q_window["bid"].fillna(0)).values
                spreads = spreads[spreads >= 0]
                if len(spreads) > 3:
                    first_half = spreads[:len(spreads) // 2].mean()
                    second_half = spreads[len(spreads) // 2:].mean()
                    f["spread_tightening"] = float(first_half - second_half)  # positive = tightening
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

    # --- Greeks (IV, delta, gamma dynamics) ---
    if len(greeks) > idx - 1:
        g_window = greeks.iloc[max(0, idx - lookback - 1):idx]  # exclude entry candle
        if len(g_window) > 0:
            g = g_window.iloc[-1]
            f["iv"] = float(g.get("implied_vol", 0) or 0)
            f["delta"] = float(abs(g.get("delta", 0) or 0))
            f["theta"] = float(g.get("theta", 0) or 0)
            f["vega"] = float(g.get("vega", 0) or 0)

            # IV change (rising IV before a move = anticipation)
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

    # --- Underlying price action ---
    if len(stock) > 0:
        s_window = stock.iloc[max(0, min(idx - 1, len(stock)) - lookback):min(idx, len(stock))]  # exclude entry candle
        if len(s_window) > 1:
            s_closes = s_window["close"].dropna().values
            if len(s_closes) > 1 and all(s_closes > 0):
                f["underlying_change_5m"] = float((s_closes[-1] / s_closes[max(-6, -len(s_closes))] - 1) * 100)
                f["underlying_change_15m"] = float((s_closes[-1] / s_closes[0] - 1) * 100)
                f["underlying_volatility"] = float(np.std(np.diff(s_closes) / s_closes[:-1]) * 100)

                # VWAP deviation
                vwap = np.mean(s_closes)
                f["vwap_deviation"] = float((s_closes[-1] / vwap - 1) * 100) if vwap > 0 else 0
            else:
                f["underlying_change_5m"] = 0
                f["underlying_change_15m"] = 0
                f["underlying_volatility"] = 0
                f["vwap_deviation"] = 0
        else:
            f["underlying_change_5m"] = 0
            f["underlying_change_15m"] = 0
            f["underlying_volatility"] = 0
            f["vwap_deviation"] = 0
    else:
        f["underlying_change_5m"] = 0
        f["underlying_change_15m"] = 0
        f["underlying_volatility"] = 0
        f["vwap_deviation"] = 0

    # --- Market regime features (daily trend context) ---
    if len(stock) > 0:
        # Daily trend: how has the underlying moved today up to this point?
        s_all = stock.iloc[:min(idx, len(stock))]
        if len(s_all) > 10:
            s_closes_all = s_all["close"].dropna().values
            if len(s_closes_all) > 10 and s_closes_all[0] > 0:
                f["daily_trend_pct"] = float((s_closes_all[-1] / s_closes_all[0] - 1) * 100)
            else:
                f["daily_trend_pct"] = 0
            # Daily range position: where are we within today's range?
            if len(s_closes_all) > 1:
                day_lo = s_closes_all.min()
                day_hi = s_closes_all.max()
                f["daily_range_position"] = float((s_closes_all[-1] - day_lo) / (day_hi - day_lo)) if day_hi > day_lo else 0.5
            else:
                f["daily_range_position"] = 0.5
            # ADX proxy: average true range / price level (higher = more trending)
            if len(s_all) > 14 and "high" in s_all.columns and "low" in s_all.columns:
                highs = s_all["high"].dropna().values[-14:]
                lows = s_all["low"].dropna().values[-14:]
                if len(highs) >= 14 and len(lows) >= 14:
                    tr = highs - lows  # simplified true range
                    atr = float(np.mean(tr))
                    f["atr_pct"] = float(atr / s_closes_all[-1] * 100) if s_closes_all[-1] > 0 else 0
                else:
                    f["atr_pct"] = 0
            else:
                f["atr_pct"] = 0
            # Pre-move underlying momentum: 5-min underlying change leading into entry
            if len(s_closes_all) > 5:
                f["pre_move_underlying_5m"] = float((s_closes_all[-1] / s_closes_all[-6] - 1) * 100)
            else:
                f["pre_move_underlying_5m"] = 0
        else:
            f["daily_trend_pct"] = 0
            f["daily_range_position"] = 0.5
            f["atr_pct"] = 0
            f["pre_move_underlying_5m"] = 0
    else:
        f["daily_trend_pct"] = 0
        f["daily_range_position"] = 0.5
        f["atr_pct"] = 0
        f["pre_move_underlying_5m"] = 0

    # --- Institutional sweep features (from underlying price action) ---
    # These detect sweeps of key levels — highly predictive per Simpsons analysis
    if len(stock) > 0 and len(stock) > idx:
        s_window_sweep = stock.iloc[max(0, min(idx - 1, len(stock)) - lookback):min(idx, len(stock))]
        if len(s_window_sweep) > 1 and "high" in s_window_sweep.columns and "low" in s_window_sweep.columns:
            recent_high = s_window_sweep["high"].max()
            recent_low = s_window_sweep["low"].min()
            last_close_s = s_window_sweep["close"].iloc[-1] if "close" in s_window_sweep.columns else 0
            last_high_s = s_window_sweep["high"].iloc[-1]
            last_low_s = s_window_sweep["low"].iloc[-1]

            # Sweep high then reverse (price went above recent high but closed below it)
            f["sweep_high"] = 1 if (last_high_s > recent_high * 0.999 and last_close_s < recent_high) else 0
            # Sweep low then reverse (price went below recent low but closed above it)
            f["sweep_low"] = 1 if (last_low_s < recent_low * 1.001 and last_close_s > recent_low) else 0
            # Price near key level (within 0.1% of recent high or low)
            f["near_key_level"] = 1 if (
                abs(last_close_s - recent_high) / max(recent_high, 1) < 0.001
                or abs(last_close_s - recent_low) / max(recent_low, 1) < 0.001
            ) else 0
        else:
            f["sweep_high"] = 0
            f["sweep_low"] = 0
            f["near_key_level"] = 0
    else:
        f["sweep_high"] = 0
        f["sweep_low"] = 0
        f["near_key_level"] = 0

    # --- Computed setups (pattern indicators) ---
    # "Coiled spring": low volatility + volume building = about to break out
    f["coiled_spring"] = 1 if (f["premium_volatility"] < 2 and f["volume_ratio"] > 1.5) else 0

    # "Volume breakout": big volume surge + price near high
    f["volume_breakout"] = 1 if (f["volume_zscore"] > 2 and f.get("near_high", 0)) else 0

    # "Bounce setup": price near low + spread tightening (buyers arriving)
    f["bounce_setup"] = 1 if (f.get("near_low", 0) and f["spread_tightening"] > 0) else 0

    # "IV expansion": IV rising (smart money positioning)
    f["iv_expanding"] = 1 if (f.get("iv_change_5m", 0) > 0.02) else 0

    # "Momentum ignition": 3+ consecutive up bars with increasing volume
    f["momentum_ignition"] = 1 if (f.get("consecutive_up_bars", 0) >= 3 and f["volume_trend"] > 1.3) else 0

    # Direction
    right_val = curr.get("right", "CALL")
    f["is_call"] = 1 if str(right_val).upper() == "CALL" else 0

    return f


# ---------------------------------------------------------------------------
# Step 2b: Find losing entries (hard negatives for Fix B)
# ---------------------------------------------------------------------------

def _find_losing_entries(
    ohlc: pd.DataFrame,
    quotes: pd.DataFrame,
    greeks: pd.DataFrame,
    ticker: str,
    min_move_pct: float = MIN_MOVE_PCT,
    cooldown: int = COOLDOWN_MIN,
    dte: int = 0,
    expiry_date: str = "",
) -> list[dict]:
    """Find entries where the FSM would have LOST money (hard negatives).

    These are timestamps where someone might have entered but the trade was
    unprofitable — the model needs to learn to AVOID these.
    """
    losers = []
    last_idx = -cooldown
    closes = ohlc["close"].values
    n = len(closes)

    for i in range(PRE_MOVE_LOOKBACK, n - 10):
        if i - last_idx < cooldown:
            continue

        entry = closes[i]
        if not entry or entry <= 0 or np.isnan(entry):
            continue

        result = simulate_with_production_fsm(
            ohlc, quotes, greeks, i,
            ticker=ticker, dte=dte, expiry_date=expiry_date,
        )
        if result is None:
            continue

        # This entry LOST money or made less than the threshold
        if result["pnl_pct"] < min_move_pct:
            losers.append({
                "idx": i,
                "entry_price": float(entry),
                "pnl_pct": result["pnl_pct"],
                "peak_gain": result["peak_gain"],
                "hold_minutes": result["hold_minutes"],
                "exit_reason": result["reason"],
                "right": ohlc.iloc[i].get("right", "CALL"),
            })
            last_idx = i

    return losers


# ---------------------------------------------------------------------------
# Step 3: Build dataset — hard negatives + class balancing
# ---------------------------------------------------------------------------

def _collect_raw_samples(
    conn: sqlite3.Connection,
    ticker: str,
    right_filter: str | None = None,
    max_date: str | None = None,
    db_path: str = THETADATA_DB,
) -> list[dict]:
    """Collect raw feature samples for a ticker using multiprocessing.

    Strategy: Preload ALL data for the ticker into memory (sequential I/O),
    then parallelize the CPU-bound FSM simulation across dates using workers.
    This avoids SQLite contention from multiple processes reading the same DB.

    Args:
        right_filter: "CALL", "PUT", or None (both).
        max_date: If set, only use data up to this date (YYYY-MM-DD). For train/test isolation.
        db_path: Path to the ThetaData DB (needed for worker processes).
    """
    min_move = TICKER_MOVE_PCT.get(ticker, MIN_MOVE_PCT)

    if max_date:
        dates = [row[0] for row in conn.execute(
            "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? AND substr(timestamp, 1, 10) <= ? ORDER BY 1",
            (ticker, max_date),
        ).fetchall()]
    else:
        dates = [row[0] for row in conn.execute(
            "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
            (ticker,),
        ).fetchall()]

    if not dates:
        return []

    rights = [right_filter] if right_filter else ["CALL", "PUT"]

    # --- Phase 1: Preload all data into memory (sequential I/O, fast) ---
    t0 = time.time()
    print(f"    {ticker}: preloading {len(dates)} days from DB...", end="", flush=True)

    work_items = []
    skipped = 0
    for dt in dates:
        # Load stock data once per date (shared across rights)
        stock = pd.read_sql_query(
            "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )
        stock_dict = stock.to_dict("list") if len(stock) > 0 else None

        for right in rights:
            strike = find_atm_strike(conn, ticker, dt, right)
            if strike is None:
                skipped += 1
                continue

            ohlc, quotes, greeks = load_single_strike_data(conn, ticker, dt, right, strike)

            if len(ohlc) < 30:
                skipped += 1
                continue

            work_items.append((
                ticker, dt, right, min_move,
                ohlc.to_dict("list"),
                quotes.to_dict("list") if len(quotes) > 0 else None,
                greeks.to_dict("list") if len(greeks) > 0 else None,
                stock_dict,
            ))

    load_time = time.time() - t0
    print(f" {len(work_items)} items in {load_time:.0f}s (skipped {skipped})", flush=True)

    if not work_items:
        return []

    # --- Phase 2: Parallel FSM simulation (CPU-bound, no I/O) ---
    t1 = time.time()
    all_rows = []

    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_process_date_inmemory, work_items, chunksize=4)

    for batch in results:
        all_rows.extend(batch)

    sim_time = time.time() - t1
    total_time = time.time() - t0
    winners = sum(1 for r in all_rows if r.get("label") == 1)
    losers = sum(1 for r in all_rows if r.get("label") == 0)
    print(f"    {ticker}: {len(all_rows)} samples ({winners} winners, {losers} losers) "
          f"| load={load_time:.0f}s sim={sim_time:.0f}s total={total_time:.0f}s "
          f"({N_WORKERS} workers)", flush=True)

    return all_rows


def _balance_and_label(all_rows: list[dict], ticker: str, label: str = "") -> pd.DataFrame:
    """Balance classes 50/50 and add runner tier labels."""
    if not all_rows:
        print(f"  {ticker}{label}: no moves found")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()

    if pos > 0 and neg > 0:
        min_count = min(pos, neg)
        pos_df = df[df["label"] == 1].sample(n=min_count, random_state=42)
        neg_df = df[df["label"] == 0].sample(n=min_count, random_state=42)
        df = pd.concat([pos_df, neg_df]).sort_values("date").reset_index(drop=True)
        print(f"  {ticker}{label}: balanced to {min_count} per class (was +{pos}/-{neg})")
        pos, neg = min_count, min_count

    df["runner_tier"] = 0
    df.loc[(df["label"] == 1) & (df["peak_pct"] < 50), "runner_tier"] = 1
    df.loc[(df["label"] == 1) & (df["peak_pct"] >= 50) & (df["peak_pct"] < 100), "runner_tier"] = 2
    df.loc[(df["label"] == 1) & (df["peak_pct"] >= 100), "runner_tier"] = 3
    df["is_runner"] = (df["peak_pct"] >= 50).astype(int)

    tier_counts = df["runner_tier"].value_counts().sort_index()
    runners = (df["is_runner"] == 1).sum()
    print(f"  {ticker}{label}: {pos}+{neg} balanced samples")
    print(f"    Tiers: regular={tier_counts.get(1,0)}, strong={tier_counts.get(2,0)}, runner={tier_counts.get(3,0)} | {runners} runners total")
    return df


def build_dataset_for_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    neg_ratio: int = NEGATIVE_RATIO,
    max_date: str | None = None,
    db_path: str = THETADATA_DB,
) -> pd.DataFrame:
    """Build labeled dataset with hard negatives (actual losing trades) + class balancing.

    Fixes applied (from ablation study 2026-05-21):
      A) Features use idx-1 (decision candle), window excludes entry candle
      B) Hard negatives: actual losing FSM entries instead of random timestamps
      D) 50/50 class balance to prevent trivially high AUC
    """
    min_move = TICKER_MOVE_PCT.get(ticker, MIN_MOVE_PCT)
    suffix = f" (data through {max_date})" if max_date else ""
    print(f"  {ticker}: scanning (min_move={min_move}%){suffix}...", flush=True)
    all_rows = _collect_raw_samples(conn, ticker, right_filter=None, max_date=max_date, db_path=db_path)
    return _balance_and_label(all_rows, ticker)


def build_direction_datasets_for_ticker(
    conn: sqlite3.Connection,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build separate CALL and PUT datasets for direction-specific model training.

    Returns (call_df, put_df). Either may be empty if insufficient data.
    """
    min_move = TICKER_MOVE_PCT.get(ticker, MIN_MOVE_PCT)
    print(f"  {ticker}: scanning CALL+PUT separately (min_move={min_move}%)...")

    call_rows = _collect_raw_samples(conn, ticker, right_filter="CALL")
    put_rows = _collect_raw_samples(conn, ticker, right_filter="PUT")

    call_df = _balance_and_label(call_rows, ticker, label="_CALL")
    put_df = _balance_and_label(put_rows, ticker, label="_PUT")

    return call_df, put_df


# ---------------------------------------------------------------------------
# Step 4: Train per-ticker model
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Time
    "minutes_since_open", "hour_bucket", "is_first_30min", "is_last_hour",
    # Premium action
    "premium", "premium_change_5m", "premium_change_10m", "premium_change_15m",
    "premium_volatility",
    # Volume
    "current_volume", "volume_ratio", "volume_trend", "volume_zscore",
    # Bid/ask
    "spread", "spread_pct", "spread_tightening", "bid_size", "ask_size", "size_imbalance",
    # Greeks
    "iv", "delta", "theta", "vega", "iv_change_5m", "iv_change_15m", "iv_trend",
    # Underlying
    "underlying_price", "underlying_change_5m", "underlying_change_15m",
    "underlying_volatility", "vwap_deviation",
    # Market regime (daily context — helps model learn trending vs ranging days)
    "daily_trend_pct", "daily_range_position", "atr_pct", "pre_move_underlying_5m",
    # Institutional sweep features (key level sweeps from Simpsons analysis)
    "sweep_high", "sweep_low", "near_key_level",
    # Computed patterns
    "coiled_spring", "iv_expanding",
    # Direction (combined model needs to learn CALL vs PUT patterns)
    "is_call",
]


def train_ticker_model(df: pd.DataFrame, ticker: str, direction_suffix: str = "") -> dict | None:
    """Train entry model. direction_suffix is "" (combined), "_CALL", or "_PUT"."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
    except ImportError:
        print("  ERROR: pip install lightgbm scikit-learn")
        return None

    label = f"{ticker}{direction_suffix}"
    if len(df) < 50:
        print(f"  {label}: only {len(df)} samples — need >= 50")
        return None

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["label"].values

    # Time-based split (first 80% train, last 20% test)
    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_test) < 10 or len(np.unique(y_train)) < 2:
        print(f"  {ticker}: insufficient test data or single class")
        return None

    # Calculate positive weight for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 24,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "scale_pos_weight": scale_pos,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=callbacks,
    )

    # Evaluate
    y_prob = model.predict(X_test)

    # Find optimal threshold (maximize F1)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.8, 0.05):
        y_bin = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    # Save model — direction_suffix e.g. "_CALL", "_PUT", or "" (combined)
    model_name = f"signal_{ticker}{direction_suffix}"
    model_path = MODEL_DIR / f"{model_name}.lgb"
    model.save_model(str(model_path))

    meta = {
        "ticker": ticker,
        "direction": direction_suffix.lstrip("_") if direction_suffix else "BOTH",
        "features": available,
        "optimal_threshold": float(best_thresh),
        "metrics": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(best_f1),
            "auc": float(auc),
            "base_positive_rate": float(y_test.mean()),
        },
        "top_features": [(f, float(v)) for f, v in top_features],
        "trained_at": datetime.now().isoformat(),
        "min_move_pct": MIN_MOVE_PCT,
        "move_window_min": MOVE_WINDOW_MIN,
    }
    with open(MODEL_DIR / f"{model_name}_meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\n  {label} RESULTS (threshold={best_thresh:.2f}):")
    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Base positive rate: {y_test.mean()*100:.1f}%")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {prec*100:.1f}% ← of predicted setups, how many actually moved +{MIN_MOVE_PCT}%")
    print(f"    Recall:    {rec*100:.1f}% ← of actual moves, how many we detected")
    print(f"    F1:        {best_f1*100:.1f}%")
    print(f"    AUC:       {auc:.3f}")
    print(f"    Top features that predict moves:")
    for feat, imp in top_features[:7]:
        print(f"      {feat}: {imp:.0f}")
    print(f"    Saved: {model_path}")

    return meta


def train_runner_model(df: pd.DataFrame, ticker: str, direction_suffix: str = "") -> dict | None:
    """Train a runner classifier — predicts if a winning trade will be a runner (+50%+).

    Binary classification on positive samples only:
    - 0 = regular win (+15-50%)
    - 1 = runner (+50%+, where our V5 adaptive trail really shines)

    In production: entry model says "take this trade", runner model says "size up".
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
    except ImportError:
        return None

    label = f"{ticker}{direction_suffix}"

    # Only train on positive samples (entries the entry model would take)
    pos_df = df[df["label"] == 1].copy()
    if len(pos_df) < 50:
        print(f"  {label} RUNNER: only {len(pos_df)} positive samples — skipping")
        return None

    available = [c for c in FEATURE_COLS if c in pos_df.columns]
    X = pos_df[available].fillna(0).values
    y = pos_df["is_runner"].values  # binary: 0 = regular, 1 = runner (+50%+)

    if y.sum() < 5:
        print(f"  {label} RUNNER: only {y.sum()} runners in data — skipping (need >= 5)")
        return None

    # Time-based split
    dates = sorted(pos_df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = pos_df["date"] < split_date
    test_mask = pos_df["date"] >= split_date

    X_train, X_test = X[train_mask.values], X[test_mask.values]
    y_train, y_test = y[train_mask.values], y[test_mask.values]

    if len(X_test) < 10 or len(np.unique(y_train)) < 2:
        print(f"  {label} RUNNER: insufficient test data or single class")
        return None

    # Balance classes — runners are rarer
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 8,
        "scale_pos_weight": scale_pos,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=callbacks,
    )

    y_prob = model.predict(X_test)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.8, 0.05):
        y_bin = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    # What's the avg peak gain for predicted runners vs non-runners?
    test_peaks = pos_df[test_mask]["peak_pct"].values
    pred_runner_peaks = test_peaks[y_pred == 1] if y_pred.sum() > 0 else np.array([0])
    pred_non_runner_peaks = test_peaks[y_pred == 0] if (y_pred == 0).sum() > 0 else np.array([0])

    # Feature importance
    importance = dict(zip(available, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    # Save — direction_suffix e.g. "_CALL", "_PUT", or "" (combined)
    model_name = f"runner_{ticker}{direction_suffix}"
    model_path = MODEL_DIR / f"{model_name}.lgb"
    model.save_model(str(model_path))

    meta = {
        "ticker": ticker,
        "direction": direction_suffix.lstrip("_") if direction_suffix else "BOTH",
        "type": "runner_classifier",
        "features": available,
        "optimal_threshold": float(best_thresh),
        "metrics": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(best_f1),
            "auc": float(auc),
            "base_runner_rate": float(y_test.mean()),
            "avg_peak_predicted_runner": float(pred_runner_peaks.mean()),
            "avg_peak_predicted_regular": float(pred_non_runner_peaks.mean()),
        },
        "top_features": [(f, float(v)) for f, v in top_features],
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_DIR / f"{model_name}_meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\n  {label} RUNNER CLASSIFIER:")
    print(f"    Test: {X_test.shape[0]} wins, {y_test.sum()} actual runners ({y_test.mean()*100:.0f}% base rate)")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {prec*100:.1f}% ← of predicted runners, how many really ran")
    print(f"    Recall:    {rec*100:.1f}% ← of actual runners, how many we caught")
    print(f"    F1:        {best_f1*100:.1f}%  |  AUC: {auc:.3f}")
    print(f"    Avg peak gain — predicted runners: {pred_runner_peaks.mean():.1f}% vs regular: {pred_non_runner_peaks.mean():.1f}%")
    print(f"    Features that distinguish runners from regular wins:")
    for feat, imp in top_features[:7]:
        print(f"      {feat}: {imp:.0f}")
    print(f"    Saved: {model_path}")

    return meta


# ---------------------------------------------------------------------------
# Step 4b: UW Flow Score Adjustments (rule-based layer on top of ML)
# ---------------------------------------------------------------------------

class UWScoreAdjuster:
    """Load UW historical data and compute score adjustments for a given date/ticker.

    Architecture: ML model predicts entry quality from price action alone.
    UW data adjusts confidence up/down based on institutional flow context.
    This keeps the ML model trainable on maximum data (100+ days) while
    UW rules only need ~30 days of data to calibrate.
    """

    def __init__(self, uw_db_path: str = UW_DB):
        self.conn = None
        self.gex_cache: dict[tuple[str, str], dict] = {}
        self.volume_cache: dict[tuple[str, str], dict] = {}
        self.flow_alerts_cache: dict[tuple[str, str], list] = {}
        self.net_prem_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._enabled = False

        if not os.path.exists(uw_db_path):
            print("  UW DB not found — score adjustments disabled")
            return

        try:
            self.conn = sqlite3.connect(uw_db_path)
            self.conn.row_factory = sqlite3.Row
            count = self.conn.execute("SELECT COUNT(*) FROM greek_exposure").fetchone()[0]
            if count > 0:
                self._enabled = True
                print(f"  UW score adjuster loaded ({count} GEX records)")
            else:
                print("  UW DB empty — score adjustments disabled")
        except Exception as e:
            print(f"  UW DB error: {e} — score adjustments disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_gex(self, ticker: str, date: str) -> dict | None:
        key = (ticker, date)
        if key in self.gex_cache:
            return self.gex_cache[key]

        row = self.conn.execute(
            "SELECT * FROM greek_exposure WHERE ticker=? AND date=?",
            (ticker, date),
        ).fetchone()
        result = dict(row) if row else None
        self.gex_cache[key] = result
        return result

    def _get_volume(self, ticker: str, date: str) -> dict | None:
        key = (ticker, date)
        if key in self.volume_cache:
            return self.volume_cache[key]

        row = self.conn.execute(
            "SELECT * FROM options_volume WHERE ticker=? AND date=?",
            (ticker, date),
        ).fetchone()
        result = dict(row) if row else None
        self.volume_cache[key] = result
        return result

    def _get_flow_alerts(self, ticker: str, date: str) -> list:
        key = (ticker, date)
        if key in self.flow_alerts_cache:
            return self.flow_alerts_cache[key]

        rows = self.conn.execute(
            "SELECT * FROM flow_alerts WHERE ticker=? AND created_at LIKE ?",
            (ticker, f"{date}%"),
        ).fetchall()
        result = [dict(r) for r in rows]
        self.flow_alerts_cache[key] = result
        return result

    def _get_net_prem(self, ticker: str, date: str) -> pd.DataFrame:
        key = (ticker, date)
        if key in self.net_prem_cache:
            return self.net_prem_cache[key]

        df = pd.read_sql_query(
            "SELECT * FROM net_prem_ticks WHERE ticker=? AND date=? ORDER BY tape_time",
            self.conn, params=(ticker, date),
        )
        self.net_prem_cache[key] = df
        return df

    def compute_adjustment(
        self, ticker: str, date: str, option_type: str, entry_time_str: str = "",
    ) -> dict:
        """Compute UW-based score adjustment for a potential trade.

        Returns {adjustment: float (-0.2 to +0.2), reasons: list[str], details: dict}
        """
        if not self._enabled:
            return {"adjustment": 0.0, "reasons": [], "details": {}}

        adj = 0.0
        reasons = []
        details = {}
        is_call = option_type.upper() == "CALL"

        # --- Rule 1: GEX bias (daily) ---
        gex = self._get_gex(ticker, date)
        if gex:
            net_gamma = float(gex.get("call_gamma", 0)) + float(gex.get("put_gamma", 0))
            net_delta = float(gex.get("call_delta", 0)) + float(gex.get("put_delta", 0))
            details["net_gamma"] = net_gamma
            details["net_delta"] = net_delta

            # Negative gamma = dealers short gamma = more volatility = good for 0DTE
            if net_gamma < 0:
                adj += 0.05
                reasons.append("negative_gex_volatile_day")

            # Delta alignment: positive net delta favors calls, negative favors puts
            if (is_call and net_delta > 0) or (not is_call and net_delta < 0):
                adj += 0.05
                reasons.append("gex_delta_aligned")
            elif (is_call and net_delta < 0) or (not is_call and net_delta > 0):
                adj -= 0.05
                reasons.append("gex_delta_against")

        # --- Rule 2: Options volume sentiment (daily) ---
        vol = self._get_volume(ticker, date)
        if vol:
            bullish = float(vol.get("bullish_premium", 0))
            bearish = float(vol.get("bearish_premium", 0))
            total = bullish + bearish
            if total > 0:
                bull_ratio = bullish / total
                details["bullish_ratio"] = bull_ratio

                # Strong bullish flow + call signal = aligned
                if is_call and bull_ratio > 0.55:
                    adj += 0.05
                    reasons.append("bullish_flow_aligned")
                elif not is_call and bull_ratio < 0.45:
                    adj += 0.05
                    reasons.append("bearish_flow_aligned")
                elif is_call and bull_ratio < 0.40:
                    adj -= 0.05
                    reasons.append("flow_against_call")
                elif not is_call and bull_ratio > 0.60:
                    adj -= 0.05
                    reasons.append("flow_against_put")

            # Put/call volume ratio
            call_vol = int(vol.get("call_volume", 0))
            put_vol = int(vol.get("put_volume", 0))
            if call_vol + put_vol > 0:
                pc_ratio = put_vol / max(call_vol, 1)
                details["put_call_ratio"] = pc_ratio
                # High put/call ratio + put signal = institutions hedging = good for puts
                if not is_call and pc_ratio > 1.2:
                    adj += 0.03
                    reasons.append("high_pc_ratio_put")

        # --- Rule 3: Unusual flow alerts (same-day) ---
        alerts = self._get_flow_alerts(ticker, date)
        if alerts:
            # Count sweeps (aggressive orders, cross exchanges)
            sweeps = [a for a in alerts if a.get("has_sweep")]
            aligned_sweeps = [
                a for a in sweeps
                if (is_call and a.get("type") == "call") or (not is_call and a.get("type") == "put")
            ]
            details["total_alerts"] = len(alerts)
            details["sweep_count"] = len(sweeps)
            details["aligned_sweeps"] = len(aligned_sweeps)

            if len(aligned_sweeps) >= 2:
                adj += 0.08
                reasons.append(f"multiple_sweeps_aligned({len(aligned_sweeps)})")
            elif len(aligned_sweeps) == 1:
                adj += 0.04
                reasons.append("sweep_aligned")

            # Large premium alerts in our direction
            large_aligned = [
                a for a in alerts
                if float(a.get("total_premium", 0)) > 100000
                and ((is_call and a.get("type") == "call") or (not is_call and a.get("type") == "put"))
            ]
            if large_aligned:
                adj += 0.05
                reasons.append(f"large_premium_flow(${sum(float(a['total_premium']) for a in large_aligned):,.0f})")

        # --- Rule 4: Intraday net premium flow (if available and entry time given) ---
        if entry_time_str:
            net_prem = self._get_net_prem(ticker, date)
            if not net_prem.empty:
                # Look at flow in the 15min leading up to entry
                # tape_time is UTC, entry_time_str might need conversion
                try:
                    recent = net_prem[net_prem["tape_time"] <= entry_time_str].tail(15)
                    if len(recent) >= 5:
                        if is_call:
                            net_flow = recent["net_call_premium"].sum()
                        else:
                            net_flow = recent["net_put_premium"].sum()

                        details["recent_net_flow"] = float(net_flow)

                        # Strong flow in our direction in last 15 min
                        if net_flow > 500000:
                            adj += 0.05
                            reasons.append("strong_intraday_flow_aligned")
                        elif net_flow < -500000:
                            adj -= 0.05
                            reasons.append("strong_intraday_flow_against")
                except Exception:
                    pass

        # Clamp total adjustment to [-0.20, +0.20]
        adj = max(-0.20, min(0.20, adj))

        return {"adjustment": adj, "reasons": reasons, "details": details}

    def close(self):
        if self.conn:
            self.conn.close()


# ---------------------------------------------------------------------------
# Step 5: Backtest — simulate trading with model signals + runner sizing
# ---------------------------------------------------------------------------

def run_backtest(
    conn: sqlite3.Connection,
    ticker: str,
    portfolio: float = 20000,
    per_trade: float = 1000,
    uw_adjuster: UWScoreAdjuster | None = None,
) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        return {}

    # Load direction-specific models if available, with fallback to combined
    def _load_bt_model(prefix: str, ticker: str, direction: str = ""):
        """Load model with fallback: direction-specific → combined → generic."""
        candidates = []
        if direction:
            candidates.append(f"{prefix}_{ticker}_{direction}")
        candidates.append(f"{prefix}_{ticker}")
        candidates.append(f"{prefix}_GENERIC")
        for name in candidates:
            path = MODEL_DIR / f"{name}.lgb"
            mpath = MODEL_DIR / f"{name}_meta.json"
            if path.exists():
                m = lgb.Booster(model_file=str(path))
                mt = {}
                if mpath.exists():
                    with open(mpath) as f:
                        mt = json.load(f)
                return m, mt, name
        return None, None, None

    # Use combined model only (direction-specific models were worse — halved training data)
    model, meta, model_name = _load_bt_model("signal", ticker)
    if model is None:
        print(f"  {ticker}: no model")
        return {}

    threshold = meta.get("optimal_threshold", 0.5)
    features = meta["features"]

    # Load runner models
    runner_model = None
    runner_path = MODEL_DIR / f"runner_{ticker}.lgb"
    if not runner_path.exists():
        runner_path = MODEL_DIR / "runner_GENERIC.lgb"
    if runner_path.exists():
        runner_model = lgb.Booster(model_file=str(runner_path))
        print(f"  {ticker}: runner model loaded for position sizing")

    print(f"  {ticker}: using combined model ({model_name})")

    # Get test dates (last 20%)
    dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc WHERE ticker=? ORDER BY 1",
        (ticker,),
    ).fetchall()]
    if not dates:
        return {}

    test_dates = dates[int(len(dates) * 0.8):]
    print(f"  {ticker}: backtesting on {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    balance = portfolio
    trades = []

    for dt in test_dates:
        stock = pd.read_sql_query(
            "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
            conn, params=(ticker, f"{dt}%"),
        )

        for right in ["CALL", "PUT"]:
            # Pick ATM strike to avoid multi-strike data contamination
            strike = find_atm_strike(conn, ticker, dt, right)
            if strike is None:
                continue

            ohlc_side, quotes_side, greeks_side = load_single_strike_data(
                conn, ticker, dt, right, strike,
            )

            if len(ohlc_side) < 30:
                continue

            last_trade_idx = -COOLDOWN_MIN
            # Scan every 5 minutes
            for idx in range(PRE_MOVE_LOOKBACK, len(ohlc_side) - 10, 5):
                if idx - last_trade_idx < COOLDOWN_MIN:
                    continue

                feat = compute_setup_features(ohlc_side, quotes_side, greeks_side, stock, idx)
                if feat is None:
                    continue

                # Combined model with is_call feature for direction awareness
                X = np.array([[feat.get(c, 0) for c in features]])
                prob = model.predict(X)[0]

                # UW flow adjustment: informational only — NEVER boosts below-threshold trades
                # Analysis showed UW boost-in lost -$56K on 916 trades
                uw_adj = 0.0
                uw_reasons = []
                if uw_adjuster and uw_adjuster.enabled:
                    entry_ts_str = ""
                    try:
                        entry_ts_str = str(ohlc_side.iloc[idx]["timestamp"])
                    except Exception:
                        pass
                    uw_result = uw_adjuster.compute_adjustment(
                        ticker, dt, right, entry_time_str=entry_ts_str,
                    )
                    uw_adj = uw_result["adjustment"]
                    uw_reasons = uw_result["reasons"]

                # UW can only REDUCE confidence (veto), never boost in
                adjusted_prob = prob + min(uw_adj, 0)

                if adjusted_prob >= threshold:
                    entry_price = feat["premium"]
                    if not entry_price or entry_price <= 0 or np.isnan(entry_price):
                        continue

                    # Runner-aware sizing: classifier predicts if this is a runner setup
                    runner_score = 0.0
                    size_mult = 1.0
                    if runner_model is not None:
                        X_runner = np.array([[feat.get(c, 0) for c in features]])
                        runner_score = float(runner_model.predict(X_runner)[0])
                        # High runner probability → size up (let V5 adaptive trail ride it)
                        if runner_score >= 0.7:
                            size_mult = 2.0  # high confidence runner → 2x size
                        elif runner_score >= 0.5:
                            size_mult = 1.5  # moderate runner signal → 1.5x size

                    trade_budget = per_trade * size_mult
                    contracts = max(1, int(trade_budget / (entry_price * 100)))

                    # Run REAL production V5 FSM for exit
                    result = simulate_with_production_fsm(
                        ohlc_side, quotes_side, greeks_side, idx,
                        ticker=ticker, dte=0, expiry_date=dt,
                        contracts=contracts,
                    )
                    if result is None:
                        continue

                    pnl = result["pnl_dollars"]
                    balance += pnl

                    trades.append({
                        "date": dt,
                        "right": right,
                        "confidence": float(prob),
                        "uw_adj": float(uw_adj),
                        "adjusted_conf": float(adjusted_prob),
                        "uw_reasons": uw_reasons,
                        "runner_score": runner_score,
                        "size_mult": size_mult,
                        "entry_price": entry_price,
                        "exit_pct": result["pnl_pct"],
                        "exit_reason": result["reason"],
                        "pnl": pnl,
                        "hold_min": result["hold_minutes"],
                        "peak_gain": result["peak_gain"],
                        "contracts": contracts,
                        "balance": balance,
                    })
                    last_trade_idx = idx

    if not trades:
        print(f"  {ticker}: 0 trades triggered")
        return {}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)

    print(f"\n  {ticker} BACKTEST RESULTS:")
    print(f"    Trades:  {len(trades)}")
    print(f"    Wins:    {wins} ({wins/len(trades)*100:.1f}%)")
    print(f"    P&L:     ${total_pnl:,.0f}")
    print(f"    Balance: ${portfolio:,.0f} → ${balance:,.0f}")
    print(f"    Avg conf: {np.mean([t['confidence'] for t in trades]):.3f}")
    print(f"    Avg hold: {np.mean([t['hold_min'] for t in trades]):.0f} min")

    # Show exit reason breakdown
    reason_counts = {}
    reason_pnl = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
        reason_pnl[r] = reason_pnl.get(r, 0) + t["pnl"]
    print(f"\n    Exit reasons:")
    for r in sorted(reason_counts, key=lambda x: -reason_counts[x]):
        marker = "+" if reason_pnl[r] > 0 else ""
        print(f"      {r}: {reason_counts[r]} trades, {marker}${reason_pnl[r]:,.0f}")

    # Show daily breakdown
    trade_df = pd.DataFrame(trades)
    daily = trade_df.groupby("date").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
    )
    print(f"\n    Daily P&L:")
    for dt, row in daily.iterrows():
        marker = "+" if row["pnl"] > 0 else ""
        print(f"      {dt}: {int(row['trades'])} trades, {marker}${row['pnl']:,.0f}")

    # UW adjustment impact analysis
    uw_adjusted_trades = [t for t in trades if t.get("uw_adj", 0) != 0]
    if uw_adjusted_trades:
        uw_boosted = [t for t in trades if t.get("uw_adj", 0) > 0]
        uw_dampened = [t for t in trades if t.get("uw_adj", 0) < 0]
        # Trades that only entered BECAUSE of UW boost (below raw threshold)
        uw_enabled = [t for t in trades if t["confidence"] < threshold <= t.get("adjusted_conf", t["confidence"])]
        # Trades that would have entered but UW blocked (above raw threshold but adjusted below)
        # These don't appear in trades list, so count them separately below

        print(f"\n    UW Flow Impact:")
        print(f"      Trades with UW adjustment: {len(uw_adjusted_trades)}/{len(trades)}")
        if uw_boosted:
            boost_pnl = sum(t["pnl"] for t in uw_boosted)
            print(f"      Boosted by UW:  {len(uw_boosted)} trades, ${boost_pnl:+,.0f} P&L")
        if uw_dampened:
            damp_pnl = sum(t["pnl"] for t in uw_dampened)
            print(f"      Dampened by UW: {len(uw_dampened)} trades, ${damp_pnl:+,.0f} P&L")
        if uw_enabled:
            enabled_pnl = sum(t["pnl"] for t in uw_enabled)
            enabled_wins = sum(1 for t in uw_enabled if t["pnl"] > 0)
            print(f"      UW-ENABLED (below ML threshold, boosted in): {len(uw_enabled)} trades, "
                  f"{enabled_wins}/{len(uw_enabled)} wins, ${enabled_pnl:+,.0f} P&L")

        # Show most common UW reasons
        reason_freq: dict[str, int] = {}
        for t in uw_adjusted_trades:
            for r in t.get("uw_reasons", []):
                reason_freq[r] = reason_freq.get(r, 0) + 1
        if reason_freq:
            print(f"      Top UW reasons:")
            for r, c in sorted(reason_freq.items(), key=lambda x: -x[1])[:5]:
                print(f"        {r}: {c}x")

    return {
        "ticker": ticker,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100,
        "total_pnl": total_pnl,
        "final_balance": balance,
        "uw_adjusted_trades": len(uw_adjusted_trades) if uw_adjusted_trades else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2: Learn pre-move patterns from option price data")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--db", type=str, default=THETADATA_DB)
    parser.add_argument("--backtest", action="store_true", help="Run P&L backtest")
    parser.add_argument("--portfolio", type=float, default=20000)
    parser.add_argument("--scan", action="store_true", help="Show what model finds in latest data")
    parser.add_argument("--no-uw", action="store_true", help="Disable UW flow adjustments in backtest")
    parser.add_argument("--uw-db", type=str, default=UW_DB, help="UW historical DB path")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Max date for training data (YYYY-MM-DD). Data after this is unseen by model.")
    parser.add_argument("--direction-split", action="store_true",
                        help="Train separate CALL and PUT models per ticker (NOT recommended — halves data)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found at {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    # Check which tickers have data
    available = []
    for t in tickers:
        count = conn.execute("SELECT count(*) FROM option_ohlc WHERE ticker=?", (t,)).fetchone()[0]
        if count > 0:
            available.append(t)

    if not available:
        print("No tickers have data. Run download_thetadata.py first.")
        conn.close()
        return

    print(f"\n{'='*70}", flush=True)
    print(f"ML SIGNAL V2 — Pre-Move Pattern Detection (Per-Ticker)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Tickers with data: {', '.join(available)}", flush=True)
    print(f"Workers: {N_WORKERS} | Min move: +{MIN_MOVE_PCT}% | Window: {MOVE_WINDOW_MIN}min | Lookback: {PRE_MOVE_LOOKBACK}min", flush=True)
    print(f"{'='*70}\n", flush=True)

    if args.backtest:
        print("BACKTEST MODE\n")
        uw_adjuster = None
        if not args.no_uw:
            uw_adjuster = UWScoreAdjuster(args.uw_db)
        results = []
        for t in available:
            r = run_backtest(conn, t, portfolio=args.portfolio, uw_adjuster=uw_adjuster)
            if r:
                results.append(r)

        if results:
            total_trades = sum(r["trades"] for r in results)
            total_wins = sum(r["wins"] for r in results)
            total_pnl = sum(r["total_pnl"] for r in results)
            total_uw = sum(r.get("uw_adjusted_trades", 0) for r in results)
            print(f"\n{'='*70}")
            print(f"COMBINED RESULTS")
            print(f"Trades: {total_trades} | Wins: {total_wins} ({total_wins/total_trades*100:.1f}%) | P&L: ${total_pnl:,.0f}")
            if total_uw > 0:
                print(f"UW-adjusted trades: {total_uw}/{total_trades}")
        if uw_adjuster:
            uw_adjuster.close()
        conn.close()
        return

    # Build datasets + train
    all_dfs = {}
    direction_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    if args.direction_split:
        print(f"\n{'='*70}")
        print("DIRECTION-SPLIT MODE: Training separate CALL and PUT models")
        print(f"{'='*70}\n")
        for t in available:
            min_move = TICKER_MOVE_PCT.get(t, MIN_MOVE_PCT)
            print(f"  {t}: scanning CALL+PUT separately (min_move={min_move}%)...")
            call_rows = _collect_raw_samples(conn, t, right_filter="CALL", max_date=args.train_end, db_path=args.db)
            put_rows = _collect_raw_samples(conn, t, right_filter="PUT", max_date=args.train_end, db_path=args.db)

            call_df = _balance_and_label(call_rows, t, label="_CALL")
            put_df = _balance_and_label(put_rows, t, label="_PUT")

            if not call_df.empty or not put_df.empty:
                direction_dfs[t] = (call_df, put_df)

            # Merge CALL+PUT samples for combined model (no re-scan)
            combined_rows = call_rows + put_rows
            combined_df = _balance_and_label(combined_rows, t)
            if not combined_df.empty:
                all_dfs[t] = combined_df
    else:
        for t in available:
            df = build_dataset_for_ticker(conn, t, max_date=args.train_end, db_path=args.db)
            if not df.empty:
                all_dfs[t] = df

    if not all_dfs and not direction_dfs:
        print("No profitable moves found in any ticker.")
        conn.close()
        return

    print(f"\n{'='*70}")
    if args.direction_split:
        print("TRAINING DIRECTION-SPECIFIC MODELS (CALL + PUT per ticker)\n")
    else:
        print("TRAINING PER-TICKER MODELS\n")

    entry_results = []
    runner_results = []

    if args.direction_split:
        for t, (call_df, put_df) in direction_dfs.items():
            # Train CALL model
            if not call_df.empty and len(call_df) >= 50:
                meta = train_ticker_model(call_df, t, direction_suffix="_CALL")
                if meta:
                    entry_results.append(meta)
                rmeta = train_runner_model(call_df, t, direction_suffix="_CALL")
                if rmeta:
                    runner_results.append(rmeta)

            # Train PUT model
            if not put_df.empty and len(put_df) >= 50:
                meta = train_ticker_model(put_df, t, direction_suffix="_PUT")
                if meta:
                    entry_results.append(meta)
                rmeta = train_runner_model(put_df, t, direction_suffix="_PUT")
                if rmeta:
                    runner_results.append(rmeta)

            # Also train combined model as fallback
            if t in all_dfs:
                meta = train_ticker_model(all_dfs[t], t)
                if meta:
                    entry_results.append(meta)
                rmeta = train_runner_model(all_dfs[t], t)
                if rmeta:
                    runner_results.append(rmeta)
    else:
        for t, df in all_dfs.items():
            meta = train_ticker_model(df, t)
            if meta:
                entry_results.append(meta)
            rmeta = train_runner_model(df, t)
            if rmeta:
                runner_results.append(rmeta)

    # Generic fallback models (always combined)
    if all_dfs:
        combined = pd.concat(list(all_dfs.values()), ignore_index=True)
        if len(combined) >= 100:
            print(f"\n  GENERIC entry model ({len(combined)} samples)...")
            train_ticker_model(combined, "GENERIC")
            print(f"  GENERIC runner model...")
            train_runner_model(combined, "GENERIC")

    # Entry model summary
    print(f"\n{'='*70}")
    print("ENTRY MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Samples':>8} {'Prec':>7} {'Recall':>7} {'F1':>6} {'AUC':>6} {'Thresh':>7}")
    print("-" * 55)
    for r in entry_results:
        m = r["metrics"]
        print(f"{r['ticker']:<8} {m['test_samples']:>8} {m['precision']*100:>6.1f}% {m['recall']*100:>6.1f}% {m['f1']*100:>5.1f}% {m['auc']:>5.3f} {r['optimal_threshold']:>6.2f}")

    # Runner model summary
    if runner_results:
        print(f"\n{'='*70}")
        print("RUNNER CLASSIFIER SUMMARY")
        print(f"{'='*70}")
        print(f"{'Ticker':<8} {'Samples':>8} {'Prec':>7} {'Recall':>7} {'AUC':>6} {'RunPeak':>8} {'RegPeak':>8}")
        print("-" * 60)
        for r in runner_results:
            m = r["metrics"]
            print(f"{r['ticker']:<8} {m['test_samples']:>8} {m['precision']*100:>6.1f}% {m['recall']*100:>6.1f}% {m['auc']:>5.3f} {m['avg_peak_predicted_runner']:>7.1f}% {m['avg_peak_predicted_regular']:>7.1f}%")

    conn.close()
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # fork is faster than spawn; safe here (no CUDA/GUI)
    main()
