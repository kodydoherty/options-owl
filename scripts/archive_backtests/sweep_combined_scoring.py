"""Combined scoring sweep: find optimal tech_weight × ml_weight × threshold × TOD.

Phase 1: Pre-compute all candidate trades (ML signal + tech score + FSM result).
          This is the expensive part (~1-2 hours). Results cached to disk.
Phase 2: Sweep all parameter combos against pre-computed data (seconds).

Usage:
    # Full run (phase 1 + phase 2)
    python scripts/sweep_combined_scoring.py

    # Phase 1 only (pre-compute candidates)
    python scripts/sweep_combined_scoring.py --phase1-only

    # Phase 2 only (sweep from cached data)
    python scripts/sweep_combined_scoring.py --phase2-only

    # Custom date range
    python scripts/sweep_combined_scoring.py --start 2026-01-02 --end 2026-05-21
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.sourcing.data.indicator_engine import compute_indicators
from options_owl.sourcing.scoring.engine import compute_score
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState
from options_owl.sourcing.filters.quality_gate import check_quality_gate
from options_owl.sourcing.filters.penalty_veto import check_penalty_veto
from options_owl.sourcing.scoring.ml_gates.signal_model import (
    compute_option_features_from_live,
    predict_entry_confidence,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
CANDIDATES_CACHE = str(PROJECT_DIR / "journal" / "sweep_candidates.json")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

PORTFOLIO_START = 23_000
MAX_CONCURRENT = 4
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
PREMIUM_CAP = 6.0
SPREAD_GATE_PCT = 15.0
GFV_BUFFER_PCT = 15.0       # only deploy 85% of SOD balance (cash account GFV protection)
DAILY_LOSS_CB_PCT = 15.0    # stop trading after losing 15% of SOD balance
MAX_SAME_DIRECTION = 3       # max positions in same direction (correlation guard)

# V6 settings for FSM
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
    V6_MAX_SPREAD_PCT=SPREAD_GATE_PCT,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)


# ── Data Loaders (from backtest_ml_e2e.py) ──────────────────────────────────


def load_trading_days_theta(conn, start: str, end: str) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT date(timestamp) as d FROM option_ohlc
        WHERE ticker = 'SPY' AND date(timestamp) >= ? AND date(timestamp) <= ?
        ORDER BY d
    """, (start, end)).fetchall()
    return [r[0] for r in rows]


def load_stock_candles_theta(conn, ticker: str, date_str: str) -> list[dict]:
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM stock_ohlc
        WHERE ticker = ? AND date(timestamp) = ?
        ORDER BY timestamp
    """, (ticker, date_str)).fetchall()
    if not rows:
        return []
    candles_5m = []
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) == 5:
            ts = datetime.fromisoformat(batch[0][0])
            candles_5m.append({
                "timestamp": int(ts.timestamp() * 1000),
                "open": float(batch[0][1]),
                "high": max(float(b[2]) for b in batch),
                "low": min(float(b[3]) for b in batch),
                "close": float(batch[-1][4]),
                "volume": sum(float(b[5] or 0) for b in batch),
                "vwap": float(batch[-1][6] or 0),
            })
            batch = []
    return candles_5m


def load_stock_candles_theta_multi(conn, ticker: str, date_str: str, lookback: int = 5) -> list[dict]:
    day_dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (day_dt - timedelta(days=lookback + 2)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM stock_ohlc
        WHERE ticker = ? AND date(timestamp) >= ? AND date(timestamp) <= ?
        ORDER BY timestamp
    """, (ticker, start, date_str)).fetchall()
    if not rows:
        return []
    candles_5m = []
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) == 5:
            ts = datetime.fromisoformat(batch[0][0])
            candles_5m.append({
                "timestamp": int(ts.timestamp() * 1000),
                "open": float(batch[0][1]),
                "high": max(float(b[2]) for b in batch),
                "low": min(float(b[3]) for b in batch),
                "close": float(batch[-1][4]),
                "volume": sum(float(b[5] or 0) for b in batch),
                "vwap": float(batch[-1][6] or 0),
            })
            batch = []
    return candles_5m


def load_day_snapshots_theta(conn, date_str: str, ticker: str):
    expiries = conn.execute("""
        SELECT DISTINCT expiration FROM option_ohlc
        WHERE ticker = ? AND date(timestamp) = ?
        ORDER BY expiration
    """, (ticker, date_str)).fetchall()
    if not expiries:
        return []
    target_exp = None
    for exp_row in expiries:
        if exp_row[0] == date_str:
            target_exp = date_str
            break
    if not target_exp:
        target_exp = expiries[0][0]

    rows = conn.execute("""
        SELECT oohlc.timestamp,
               COALESCE(og.underlying_price, 0) as underlying_price,
               COALESCE(oq.bid, 0) as bid,
               COALESCE(oq.ask, 0) as ask,
               CASE
                   WHEN oq.bid > 0 AND oq.ask > 0 THEN (oq.bid + oq.ask) / 2.0
                   ELSE oohlc.close
               END as midpoint,
               COALESCE(og.implied_vol, 0) as iv,
               COALESCE(og.delta, 0) as delta,
               COALESCE(og.theta, 0) as theta,
               COALESCE(og.vega, 0) as vega,
               oohlc.volume,
               oohlc.strike,
               LOWER(oohlc.right) as option_type,
               oohlc.expiration as expiry_date,
               oohlc.ticker || '_' || oohlc.expiration || '_' || CAST(oohlc.strike AS TEXT) || '_' || oohlc.right as contract_ticker
        FROM option_ohlc oohlc
        LEFT JOIN option_quotes oq
          ON oohlc.ticker = oq.ticker AND oohlc.expiration = oq.expiration
          AND oohlc.strike = oq.strike AND oohlc.right = oq.right
          AND oohlc.timestamp = oq.timestamp
        LEFT JOIN option_greeks og
          ON oohlc.ticker = og.ticker AND oohlc.expiration = og.expiration
          AND oohlc.strike = og.strike AND oohlc.right = og.right
          AND oohlc.timestamp = og.timestamp
        WHERE oohlc.ticker = ? AND date(oohlc.timestamp) = ?
          AND oohlc.expiration = ?
        ORDER BY oohlc.timestamp, ABS(COALESCE(og.underlying_price, 0) - oohlc.strike)
    """, (ticker, date_str, target_exp)).fetchall()
    return rows


def load_contract_ticks_theta(conn, ticker, expiration, strike, right, after_ts):
    rows = conn.execute("""
        SELECT oohlc.timestamp,
               CASE
                   WHEN oq.bid > 0 AND oq.ask > 0 THEN (oq.bid + oq.ask) / 2.0
                   ELSE oohlc.close
               END as midpoint,
               COALESCE(oq.bid, oohlc.close) as bid,
               COALESCE(oq.ask, oohlc.close) as ask,
               COALESCE(og.underlying_price, 0) as underlying_price,
               COALESCE(og.implied_vol, 0) as iv,
               COALESCE(og.delta, 0) as delta,
               COALESCE(og.theta, 0) as theta,
               COALESCE(og.vega, 0) as vega,
               oohlc.volume
        FROM option_ohlc oohlc
        LEFT JOIN option_quotes oq
          ON oohlc.ticker = oq.ticker AND oohlc.expiration = oq.expiration
          AND oohlc.strike = oq.strike AND oohlc.right = oq.right
          AND oohlc.timestamp = oq.timestamp
        LEFT JOIN option_greeks og
          ON oohlc.ticker = og.ticker AND oohlc.expiration = og.expiration
          AND oohlc.strike = og.strike AND oohlc.right = og.right
          AND oohlc.timestamp = og.timestamp
        WHERE oohlc.ticker = ? AND oohlc.expiration = ?
          AND oohlc.strike = ? AND oohlc.right = ?
          AND oohlc.timestamp > ?
        ORDER BY oohlc.timestamp
    """, (ticker, expiration, strike, right, after_ts)).fetchall()

    if not rows or len(rows) < 5:
        return None
    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "theta", "vega", "volume"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 5:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"], format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ── Direction + Scoring ─────────────────────────────────────────────────────


def infer_direction(indicators) -> Direction:
    bullish = bearish = 0
    if indicators.ema_cross_strength > 0.05:
        bullish += 2
    elif indicators.ema_cross_strength < -0.05:
        bearish += 2
    if indicators.macd_line > 0:
        bullish += 1
    elif indicators.macd_line < 0:
        bearish += 1
    if indicators.vwap > 0 and indicators.last_close > indicators.vwap:
        bullish += 1
    elif indicators.vwap > 0 and indicators.last_close < indicators.vwap:
        bearish += 1
    return Direction.CALL if bullish >= bearish else Direction.PUT


def run_scoring(ticker, candles_5m, indicators, direction):
    """Run sourcing scoring pipeline. Returns (score, passed_quality, passed_penalty)."""
    ctx = SignalContext(
        ticker=ticker,
        scan_time=datetime.now(ET).isoformat(),
        state=SignalState.INDICATED,
        direction=direction,
        candles_5m=candles_5m,
        candle_source="thetadata",
        indicators=indicators,
    )
    scored = compute_score(ctx)
    if scored.rejected:
        return 0, False, False

    quality_ok = check_quality_gate(ctx, 0)  # threshold=0 to not block on score
    penalty_veto = check_penalty_veto(ctx)

    return scored.score, quality_ok, not penalty_veto


# ── FSM Simulation ──────────────────────────────────────────────────────────


def build_candle_data_for_fsm(indicators) -> dict:
    """Build candle_data dict matching what position_monitor passes to FSM."""
    if indicators is None:
        return {}
    ind_5m = {
        "ema_cross_strength": indicators.ema_cross_strength,
        "macd_line": indicators.macd_line,
        "macd_histogram": indicators.macd_histogram,
        "rsi9": indicators.rsi9,
        "vwap": indicators.vwap,
        "bb_squeeze": indicators.bb_squeeze,
        "atr_expanding": indicators.atr_expanding,
        "volume_ratio": indicators.volume_ratio,
        "adx": indicators.adx,
        "last_close": indicators.last_close,
        "last_high": indicators.last_high,
        "last_low": indicators.last_low,
        "session_high": indicators.session_high,
        "session_low": indicators.session_low,
        "pdh": indicators.pdh,
        "pdl": indicators.pdl,
        "pwh": indicators.pwh,
        "pwl": indicators.pwl,
    }
    return {"indicators": {"5m": ind_5m}}


from options_owl.risk.exit_v5.config import V5Config, AdaptiveTier

# Stop-loss configurations to sweep (phase 1 runs FSM for each)
STOP_CONFIGS = {
    "ultra_tight": V5Config(
        tight_stop_0dte_pct=10.0, backstop_0dte_pct=20.0,
        tight_stop_multiday_pct=15.0, backstop_multiday_pct=30.0,
        checkpoint_drop_pct=10.0,
        early_pop_backstop_0dte_pct=15.0, early_pop_backstop_multiday_pct=25.0,
    ),
    "tight": V5Config(
        tight_stop_0dte_pct=15.0, backstop_0dte_pct=30.0,
        tight_stop_multiday_pct=30.0, backstop_multiday_pct=50.0,
        checkpoint_drop_pct=15.0,
        early_pop_backstop_0dte_pct=25.0, early_pop_backstop_multiday_pct=40.0,
    ),
    "moderate": V5Config(
        tight_stop_0dte_pct=25.0, backstop_0dte_pct=45.0,
        tight_stop_multiday_pct=40.0, backstop_multiday_pct=60.0,
        checkpoint_drop_pct=20.0,
        early_pop_backstop_0dte_pct=30.0, early_pop_backstop_multiday_pct=45.0,
    ),
    "wide": V5Config(
        tight_stop_0dte_pct=35.0, backstop_0dte_pct=65.0,
        tight_stop_multiday_pct=52.0, backstop_multiday_pct=75.0,
        checkpoint_drop_pct=30.0,
        early_pop_backstop_0dte_pct=35.0, early_pop_backstop_multiday_pct=50.0,
    ),
}


def simulate_exit(df, entry_premium, contracts, option_type, dte, expiry_date, ticker,
                  candle_data=None, stop_config=None):
    """Run production V5 FSM against tick data. Returns dict with pnl, reason, etc."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    # Use provided stop config or per-ticker default
    if stop_config is not None:
        tcfg = stop_config
    else:
        tcfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(tcfg, settings=_V6_SETTINGS)

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.astimezone(ET).replace(tzinfo=None)
    else:
        entry_ts = entry_ts.replace(tzinfo=UTC).astimezone(ET).replace(tzinfo=None)

    first_underlying = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_underlying = float(u)
            break

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium

        now = df["ts"].iloc[idx]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        underlying = df["underlying_price"].iloc[idx] or 0.0

        if now.tzinfo is not None:
            now_et = now.astimezone(ET)
        else:
            now_et = now.replace(tzinfo=UTC).astimezone(ET)
        minutes_to_close = max(0, (16 * 60) - (now_et.hour * 60 + now_et.minute))
        now_naive = now_et.replace(tzinfo=None)

        action = fsm.evaluate(
            state, premium, bid, ask, now_naive,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
            candle_data=candle_data or {},
        )

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now_naive - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed, "exit_prem": premium, "peak_gain": peak_gain,
            }

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
        "exit_prem": last_prem, "peak_gain": peak_gain,
    }


# ── Phase 1: Pre-compute all candidate trades ──────────────────────────────


def phase1_precompute(start: str, end: str) -> list[dict]:
    """Scan all days/tickers, find ML signals, run FSM, store everything."""
    conn = sqlite3.connect(THETADATA_DB)
    trading_days = load_trading_days_theta(conn, start, end)
    print(f"Phase 1: Pre-computing candidates across {len(trading_days)} trading days")
    print(f"Date range: {trading_days[0]} to {trading_days[-1]}")

    candidates = []
    total_snapshots = 0

    for day_idx, date_str in enumerate(trading_days):
        day_dt = datetime.strptime(date_str, "%Y-%m-%d")
        market_open_et = day_dt.replace(hour=9, minute=30, tzinfo=ET)
        day_count = 0

        print(f"\n[Day {day_idx+1}/{len(trading_days)}] {date_str}:", end="", flush=True)

        for ticker in TICKERS:
            # Load candle data
            candles_5m = load_stock_candles_theta(conn, ticker, date_str)
            candles_multi = load_stock_candles_theta_multi(conn, ticker, date_str) if candles_5m else []
            has_candles = len(candles_5m) >= 10

            indicators = None
            direction = None
            tech_score = 0
            candle_data_fsm = {}

            if has_candles:
                ind_candles = candles_multi if len(candles_multi) >= 30 else candles_5m
                indicators = compute_indicators(ind_candles)
                direction = infer_direction(indicators)
                candle_data_fsm = build_candle_data_for_fsm(indicators)
                tech_score, quality_ok, penalty_ok = run_scoring(
                    ticker, candles_5m, indicators, direction
                )

            # Load option snapshots
            snapshots = load_day_snapshots_theta(conn, date_str, ticker)
            if not snapshots:
                continue

            # Build underlying history for fallback direction
            underlying_hist_all = []
            for s in snapshots:
                u = float(s[1] or 0)
                if u > 0 and (not underlying_hist_all or u != underlying_hist_all[-1]):
                    underlying_hist_all.append(u)

            if direction is None and len(underlying_hist_all) >= 10:
                prices = np.array(underlying_hist_all[-78:], dtype=np.float64)
                k9, k21 = 2.0 / 10, 2.0 / 22
                ema9 = ema21 = prices[0]
                for p in prices[1:]:
                    ema9 = p * k9 + ema9 * (1 - k9)
                    ema21 = p * k21 + ema21 * (1 - k21)
                direction = Direction.CALL if ema9 > ema21 else Direction.PUT

            if direction is None:
                continue

            dir_str = "call" if direction == Direction.CALL else "put"
            filtered_snaps = [s for s in snapshots if s[11] == dir_str]
            if not filtered_snaps:
                continue

            # Sample snapshots for ML entry
            premium_hist = []
            volume_hist = []
            underlying_hist = []
            last_check_ts = None
            found_entry = False
            is_call = direction == Direction.CALL

            for snap in filtered_snaps:
                mid = float(snap[4] or 0)
                vol = int(snap[9] or 0)
                und = float(snap[1] or 0)
                if mid > 0:
                    premium_hist.append(mid)
                if vol > 0:
                    volume_hist.append(vol)
                if und > 0:
                    underlying_hist.append(und)

                if found_entry:
                    continue

                try:
                    snap_dt = datetime.fromisoformat(snap[0])
                    if snap_dt.tzinfo is None:
                        snap_dt = snap_dt.replace(tzinfo=UTC)
                    snap_et = snap_dt.astimezone(ET)
                except (ValueError, TypeError):
                    continue

                # Rate limit: every 5 min
                if last_check_ts is not None:
                    elapsed = (snap_et - last_check_ts).total_seconds() / 60
                    if elapsed < 5:
                        continue
                last_check_ts = snap_et

                # Market hours
                if snap_et.hour < 9 or (snap_et.hour == 9 and snap_et.minute < 30):
                    continue
                if snap_et.hour >= 15 and snap_et.minute >= 30:
                    continue

                minutes_since_open = int((snap_et - market_open_et.replace(
                    year=snap_et.year, month=snap_et.month, day=snap_et.day
                )).total_seconds() / 60)
                if minutes_since_open < 0:
                    continue

                total_snapshots += 1

                # Run ML check
                captured_at = snap[0]
                underlying = float(snap[1] or 0)
                bid = float(snap[2] or 0)
                ask = float(snap[3] or 0)
                midpoint = float(snap[4] or 0)
                iv = float(snap[5] or 0)
                delta_val = float(snap[6] or 0)
                theta_val = float(snap[7] or 0)
                vega_val = float(snap[8] or 0)
                volume = int(snap[9] or 0)

                premium = midpoint if midpoint > 0 else (bid + ask) / 2
                if premium <= 0:
                    continue

                features = compute_option_features_from_live(
                    ticker=ticker, premium=premium, bid=bid, ask=ask,
                    iv=iv, delta=delta_val, theta=theta_val, vega=vega_val,
                    volume=volume, underlying_price=underlying,
                    minutes_since_open=minutes_since_open, is_call=is_call,
                    premium_history=premium_hist[-15:],
                    volume_history=volume_hist[-15:],
                    underlying_history=underlying_hist[-15:],
                )

                ml_direction = "CALL" if is_call else "PUT"
                ml_result = predict_entry_confidence(ticker, features, ml_direction)

                if not ml_result["is_signal"] or ml_result["confidence"] <= 0:
                    continue

                # ML says yes — now compute entry details
                entry_premium = ask if ask > 0 else premium
                if entry_premium <= 0:
                    continue

                # Premium cap (hard filter — always applied)
                if entry_premium > PREMIUM_CAP:
                    continue

                # Spread gate (hard filter)
                if bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / ask * 100
                    if spread_pct > SPREAD_GATE_PCT:
                        continue

                # Get expiry info
                expiry_date = snap[12]
                try:
                    exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    dte = max(0, (exp_dt - day_dt.date()).days)
                except (ValueError, TypeError):
                    dte = 0

                # Load tick data for FSM
                snap_strike = float(snap[10])
                snap_right = snap[11].upper()
                tick_df = load_contract_ticks_theta(
                    conn, ticker, expiry_date, snap_strike, snap_right, snap[0]
                )
                if tick_df is None:
                    continue

                # Size at 10 contracts for normalization (we'll rescale in phase 2)
                norm_contracts = 10

                # Run FSM for each stop-loss configuration
                stop_results = {}
                for stop_name, stop_cfg in STOP_CONFIGS.items():
                    r = simulate_exit(
                        tick_df, entry_premium, norm_contracts, dir_str,
                        dte, expiry_date, ticker,
                        candle_data=candle_data_fsm,
                        stop_config=stop_cfg,
                    )
                    ppc = r["pnl"] / norm_contracts if norm_contracts > 0 else 0
                    stop_results[stop_name] = {
                        "pnl_per_contract": round(ppc, 2),
                        "exit_reason": r["reason"],
                        "hold_min": round(r["hold"], 1),
                        "peak_gain_pct": round(r["peak_gain"], 1),
                    }

                # Use "tight" (current production) as the display result
                result = stop_results.get("tight", stop_results[list(stop_results.keys())[0]])
                pnl_per_contract = result["pnl_per_contract"]

                # Classify time-of-day session
                hour = snap_et.hour
                minute = snap_et.minute
                total_min = hour * 60 + minute
                if total_min < 630:  # before 10:30
                    session = "killzone"      # 9:30-10:30
                elif total_min < 720:
                    session = "late_morning"   # 10:30-12:00
                elif total_min < 810:
                    session = "midday"         # 12:00-1:30
                elif total_min < 900:
                    session = "early_afternoon"  # 1:30-3:00
                else:
                    session = "power_hour"     # 3:00-4:00

                # Compute veto gate fields from indicators (for phase 2 filtering)
                veto_fields = {}
                if indicators is not None:
                    veto_fields = {
                        "sweep_pdh": indicators.sweep_pdh,
                        "sweep_pdl": indicators.sweep_pdl,
                        "sweep_pwh": indicators.sweep_pwh,
                        "sweep_pwl": indicators.sweep_pwl,
                        "sweep_session_high": indicators.sweep_session_high,
                        "sweep_session_low": indicators.sweep_session_low,
                        "volume_ratio": round(indicators.volume_ratio, 2),
                        "atr14": round(indicators.atr14, 4),
                        "adx": round(indicators.adx, 1),
                    }
                    # 5m momentum: last 3 candle close trend
                    if candles_5m and len(candles_5m) >= 3:
                        c_first = candles_5m[-3]["close"]
                        c_last = candles_5m[-1]["close"]
                        if c_first > 0:
                            veto_fields["momentum_5m_pct"] = round(
                                ((c_last - c_first) / c_first) * 100, 3
                            )

                # Spread % from option data
                spread_pct_val = None
                if bid > 0 and ask > 0:
                    mid_val = (bid + ask) / 2
                    if mid_val > 0:
                        spread_pct_val = round(((ask - bid) / mid_val) * 100, 1)

                candidate = {
                    "day": date_str,
                    "ticker": ticker,
                    "direction": dir_str,
                    "entry_time_et": snap_et.strftime("%H:%M"),
                    "session": session,
                    "hour": hour,
                    "minute": minute,
                    "minutes_since_open": minutes_since_open,
                    "entry_premium": entry_premium,
                    "dte": dte,
                    "tech_score": tech_score,
                    "ml_confidence": round(ml_result["confidence"], 4),
                    "ml_threshold": round(ml_result["threshold"], 4),
                    "runner_score": round(ml_result.get("runner_score", 0), 4),
                    "model_source": ml_result.get("model_source", "unknown"),
                    "has_candles": has_candles,
                    "premium_hist_len": len(premium_hist),
                    "spread_pct": spread_pct_val,
                    # Veto gate fields (for Simpsons-style filtering in phase 2)
                    **veto_fields,
                    # Per-stop-config results (for phase 2 sweep)
                    "stop_results": stop_results,
                    # Legacy fields (from "tight" config for display)
                    "pnl_per_contract": round(pnl_per_contract, 2),
                    "exit_reason": result["exit_reason"],
                    "hold_min": result["hold_min"],
                    "peak_gain_pct": result["peak_gain_pct"],
                }
                candidates.append(candidate)
                found_entry = True
                day_count += 1

                pnl_str = f"+${pnl_per_contract:,.0f}" if pnl_per_contract >= 0 else f"-${abs(pnl_per_contract):,.0f}"
                print(f" {ticker}{'C' if is_call else 'P'}{pnl_str}", end="", flush=True)

        if day_count == 0:
            print(" (no trades)", end="")

    conn.close()

    print(f"\n\nPhase 1 complete: {len(candidates)} candidate trades from {total_snapshots:,} snapshots checked")

    # Cache to disk
    with open(CANDIDATES_CACHE, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Cached to {CANDIDATES_CACHE}")

    return candidates


# ── Phase 2: Sweep parameter combos ────────────────────────────────────────


@dataclass
class SweepResult:
    tech_weight: float
    ml_weight: float
    threshold: float
    tod_rule: str
    stop_config: str = "tight"
    min_warmup: int = 0
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    final_portfolio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0


# Time-of-day rules: which sessions are allowed + score adjustments
TOD_RULES = {
    "all_sessions": {},  # no restriction
    "no_midday": {"midday": "block"},  # skip 12:00-1:30
    "no_afternoon": {"midday": "block", "early_afternoon": "block"},  # skip 12:00-3:00
    "killzone_only": {
        "late_morning": "block", "midday": "block",
        "early_afternoon": "block", "power_hour": "block",
    },
    "killzone_bonus": {"killzone": 15, "midday": -15, "early_afternoon": -10},
    "morning_only": {"midday": "block", "early_afternoon": "block", "power_hour": "block"},
    "no_early_afternoon": {"early_afternoon": "block"},
    "session_weighted": {"killzone": 20, "late_morning": 5, "midday": -20, "early_afternoon": -15, "power_hour": -5},
}


def compute_combined_score(tech_score: float, ml_confidence: float,
                           tech_weight: float, ml_weight: float) -> float:
    """Combine tech score (0-100) and ML confidence (0-1) into a single 0-100 score."""
    ml_scaled = ml_confidence * 100
    return tech_weight * tech_score + ml_weight * ml_scaled


def _apply_veto_gates(c: dict) -> str | None:
    """Apply Simpsons-inspired veto gates to a candidate dict.

    Returns rejection reason string if vetoed, None if passed.
    """
    direction = c.get("direction", "call")
    is_call = direction == "call"

    # Gate V1: Afternoon danger zone (1:30-3:00 PM ET)
    session = c.get("session", "")
    if session == "early_afternoon":
        return "afternoon_danger_zone"

    # Gate V2: Wide spread > 30%
    spread = c.get("spread_pct")
    if spread is not None and spread > 30:
        return f"spread_too_wide: {spread}%"

    # NOTE: ML warmup, sweep, volume, momentum, ATR gates are disabled.
    # Sweep data proves killzone entries at 9:30 with 1 obs are most
    # profitable. Tech scoring (0.8 weight) handles filtering.
    # Sweep/volume/momentum need fresh indicators per entry.

    return None  # all gates passed


def run_sweep_combo(candidates: list[dict], tech_weight: float, ml_weight: float,
                    threshold: float, tod_rule_name: str,
                    stop_config_name: str = "tight",
                    min_minutes_after_open: int = 0,
                    enable_veto_gates: bool = False) -> SweepResult:
    """Evaluate one parameter combination against pre-computed candidates."""
    tod_rule = TOD_RULES[tod_rule_name]
    res = SweepResult(
        tech_weight=tech_weight, ml_weight=ml_weight,
        threshold=threshold, tod_rule=tod_rule_name,
        stop_config=stop_config_name,
        min_warmup=min_minutes_after_open,
    )

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    daily_pnls: dict[str, float] = {}
    # Per-day state (reset each day)
    open_today: dict[str, list[str]] = {}    # day -> tickers with open trades
    open_dirs: dict[str, list[str]] = {}     # day -> directions of open trades
    daily_spent: dict[str, float] = {}       # day -> total $ deployed
    daily_realized: dict[str, float] = {}    # day -> realized P&L
    daily_cb: dict[str, bool] = {}           # day -> circuit breaker tripped

    pnl_list = []

    for c in candidates:
        day = c["day"]
        ticker = c["ticker"]
        session = c["session"]
        direction = c["direction"]

        # Initialize daily state
        if day not in open_today:
            open_today[day] = []
            open_dirs[day] = []
            daily_spent[day] = 0.0
            daily_realized[day] = 0.0
            daily_cb[day] = False

        # Circuit breaker: stop trading after 15% daily loss
        if daily_cb[day]:
            continue

        # Warmup delay: skip entries too close to market open
        if min_minutes_after_open > 0:
            mso = c.get("minutes_since_open", 0)
            if mso < min_minutes_after_open:
                continue

        # Veto gates (Simpsons-style hard blocks)
        if enable_veto_gates:
            veto_reason = _apply_veto_gates(c)
            if veto_reason:
                continue

        # TOD rule: block or adjust
        tod_adj = tod_rule.get(session, 0)
        if tod_adj == "block":
            continue

        # Concurrent check (max 4 per day)
        if len(open_today[day]) >= MAX_CONCURRENT:
            continue
        if ticker in open_today[day]:
            continue

        # Correlation guard: max 3 same direction
        same_dir = sum(1 for d in open_dirs[day] if d == direction)
        if same_dir >= MAX_SAME_DIRECTION:
            continue

        # Combined score
        score = compute_combined_score(
            c["tech_score"], c["ml_confidence"],
            tech_weight, ml_weight,
        )
        if isinstance(tod_adj, (int, float)):
            score += tod_adj

        if score < threshold:
            continue

        # Position sizing (compounding)
        entry_premium = c["entry_premium"]
        deployable = portfolio * MAX_RISK_PCT
        per_slot = deployable / MAX_CONCURRENT
        position_cap = portfolio * MAX_POSITION_PCT
        cost_per = entry_premium * 100

        # GFV check: can we afford this trade with settled funds?
        sod_balance = portfolio  # simplified: use current portfolio as SOD
        gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
        gfv_remaining = gfv_limit - daily_spent[day]
        if gfv_remaining < cost_per:
            continue

        # Confidence-weighted sizing
        conf = c["ml_confidence"]
        if conf >= 0.90:
            mult = 0.95
        elif conf >= 0.80:
            mult = 0.60
        else:
            mult = 1.00

        scaled = per_slot * mult
        raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
        cap_ct = int(position_cap / cost_per) if cost_per > 0 else 1
        gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_ct, cap_ct, gfv_ct))

        # Track GFV spend
        trade_cost = contracts * cost_per
        daily_spent[day] += trade_cost

        # P&L using pre-computed per-contract result for this stop config
        stop_data = c.get("stop_results", {}).get(stop_config_name)
        if stop_data:
            pnl_pc = stop_data["pnl_per_contract"]
        else:
            pnl_pc = c["pnl_per_contract"]  # fallback to legacy
        trade_pnl = pnl_pc * contracts
        portfolio += trade_pnl
        pnl_list.append(trade_pnl)
        open_today[day].append(ticker)
        open_dirs[day].append(direction)

        if not daily_pnls.get(day):
            daily_pnls[day] = 0
        daily_pnls[day] += trade_pnl

        # Circuit breaker check
        daily_realized[day] += trade_pnl
        if daily_realized[day] < 0:
            loss_pct = abs(daily_realized[day]) / sod_balance * 100
            if loss_pct >= DAILY_LOSS_CB_PCT:
                daily_cb[day] = True

        # Drawdown
        if portfolio > peak_portfolio:
            peak_portfolio = portfolio
        dd = (peak_portfolio - portfolio) / peak_portfolio * 100 if peak_portfolio > 0 else 0
        if dd > max_dd:
            max_dd = dd

    if not pnl_list:
        return res

    pnl_arr = np.array(pnl_list)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]

    res.trades = len(pnl_arr)
    res.wins = len(wins)
    res.total_pnl = float(pnl_arr.sum())
    res.final_portfolio = portfolio
    res.win_rate = len(wins) / len(pnl_arr) * 100
    res.avg_win = float(wins.mean()) if len(wins) > 0 else 0
    res.avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    res.max_drawdown = max_dd

    gross_win = float(wins.sum()) if len(wins) > 0 else 0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 1
    res.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe from daily P&L
    daily_vals = list(daily_pnls.values())
    if len(daily_vals) > 1:
        mean_r = np.mean(daily_vals)
        std_r = np.std(daily_vals, ddof=1)
        res.sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0

    return res


EXCLUDED_TICKERS = {"GOOGL", "MSFT", "AVGO"}  # net losers in both ML sweep and real trades


def phase2_sweep(candidates: list[dict], enable_veto_gates: bool = False):
    """Sweep all parameter combinations and find optimal config."""
    # Filter excluded tickers
    orig_count = len(candidates)
    candidates = [c for c in candidates if c["ticker"] not in EXCLUDED_TICKERS]
    if orig_count != len(candidates):
        print(f"Excluded {orig_count - len(candidates)} candidates from {EXCLUDED_TICKERS}")

    veto_label = " [VETO GATES ON]" if enable_veto_gates else ""
    print(f"\nPhase 2: Sweeping parameter combos against {len(candidates)} candidates{veto_label}")

    # Parameter grid
    tech_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresholds = list(range(10, 85, 5))
    tod_rules = list(TOD_RULES.keys())
    stop_configs = list(STOP_CONFIGS.keys())
    warmup_delays = [0, 5, 15, 30, 60]  # minutes after market open before trading

    total_combos = (len(tech_weights) * len(thresholds) * len(tod_rules)
                    * len(stop_configs) * len(warmup_delays))
    print(f"Grid: {len(tech_weights)} weights × {len(thresholds)} thresholds × "
          f"{len(tod_rules)} TOD rules × {len(stop_configs)} stop configs × "
          f"{len(warmup_delays)} warmup delays = {total_combos:,} combos")

    results: list[SweepResult] = []
    t0 = time.time()

    for tw in tech_weights:
        mw = 1.0 - tw
        for thresh in thresholds:
            for tod in tod_rules:
                for sc in stop_configs:
                    for wu in warmup_delays:
                        r = run_sweep_combo(candidates, tw, mw, thresh, tod, sc, wu,
                                            enable_veto_gates=enable_veto_gates)
                        results.append(r)

    elapsed = time.time() - t0
    print(f"Sweep complete in {elapsed:.1f}s")

    # Filter to combos with >= 30 trades (statistical significance)
    valid = [r for r in results if r.trades >= 30]
    if not valid:
        print("WARNING: No combos had >= 30 trades. Lowering to >= 10.")
        valid = [r for r in results if r.trades >= 10]

    if not valid:
        print("ERROR: No combos produced any trades. Check ML models and data.")
        return

    # Sort by total P&L
    valid.sort(key=lambda r: r.total_pnl, reverse=True)

    # Top 30 by P&L
    hdr = (f"{'Rank':>4} {'TechW':>5} {'MLW':>5} {'Thresh':>6} {'Stops':<12} {'TOD Rule':<22} {'WU':>3} "
           f"{'Trades':>6} {'WR%':>5} {'P&L':>12} {'PF':>5} {'AvgWin':>8} {'AvgLoss':>8} "
           f"{'MaxDD%':>6} {'Sharpe':>6} {'Final$':>10}")
    print(f"\n{'='*150}")
    print(f"TOP 30 BY TOTAL P&L (from {len(valid)} valid combos with 30+ trades)")
    print(f"{'='*150}")
    print(hdr)
    print("-" * 150)
    for i, r in enumerate(valid[:30]):
        print(f"{i+1:>4} {r.tech_weight:>5.1f} {r.ml_weight:>5.1f} {r.threshold:>6.0f} "
              f"{r.stop_config:<12} {r.tod_rule:<22} {r.min_warmup:>3} "
              f"{r.trades:>6} {r.win_rate:>5.1f} ${r.total_pnl:>10,.0f} "
              f"{r.profit_factor:>5.2f} ${r.avg_win:>7,.0f} ${r.avg_loss:>7,.0f} "
              f"{r.max_drawdown:>5.1f}% {r.sharpe:>6.2f} ${r.final_portfolio:>9,.0f}")

    # Best by profit factor (min 30 trades)
    pf_sorted = sorted(valid, key=lambda r: r.profit_factor, reverse=True)
    print(f"\n{'='*120}")
    print(f"TOP 15 BY PROFIT FACTOR")
    print(f"{'='*120}")
    print(f"{'Rank':>4} {'TechW':>5} {'MLW':>5} {'Thresh':>6} {'Stops':<12} {'TOD Rule':<22} {'WU':>3} "
          f"{'Trades':>6} {'WR%':>5} {'P&L':>12} {'PF':>5} {'Sharpe':>6}")
    print("-" * 120)
    for i, r in enumerate(pf_sorted[:15]):
        print(f"{i+1:>4} {r.tech_weight:>5.1f} {r.ml_weight:>5.1f} {r.threshold:>6.0f} "
              f"{r.stop_config:<12} {r.tod_rule:<22} {r.min_warmup:>3} "
              f"{r.trades:>6} {r.win_rate:>5.1f} ${r.total_pnl:>10,.0f} "
              f"{r.profit_factor:>5.2f} {r.sharpe:>6.2f}")

    # Best by Sharpe
    sharpe_sorted = sorted(valid, key=lambda r: r.sharpe, reverse=True)
    print(f"\n{'='*120}")
    print(f"TOP 15 BY SHARPE RATIO")
    print(f"{'='*120}")
    print(f"{'Rank':>4} {'TechW':>5} {'MLW':>5} {'Thresh':>6} {'Stops':<12} {'TOD Rule':<22} {'WU':>3} "
          f"{'Trades':>6} {'WR%':>5} {'P&L':>12} {'PF':>5} {'Sharpe':>6}")
    print("-" * 120)
    for i, r in enumerate(sharpe_sorted[:15]):
        print(f"{i+1:>4} {r.tech_weight:>5.1f} {r.ml_weight:>5.1f} {r.threshold:>6.0f} "
              f"{r.stop_config:<12} {r.tod_rule:<22} {r.min_warmup:>3} "
              f"{r.trades:>6} {r.win_rate:>5.1f} ${r.total_pnl:>10,.0f} "
              f"{r.profit_factor:>5.2f} {r.sharpe:>6.2f}")

    # Analysis: effect of each parameter dimension
    print(f"\n{'='*80}")
    print("PARAMETER DIMENSION ANALYSIS")
    print(f"{'='*80}")

    # Tech weight effect
    print(f"\n--- Tech Weight Effect (averaged across all thresholds + TOD) ---")
    print(f"{'TechW':>6} {'Avg P&L':>10} {'Avg WR%':>8} {'Avg PF':>7} {'Avg Trades':>10}")
    for tw in tech_weights:
        subset = [r for r in valid if r.tech_weight == tw]
        if subset:
            avg_pnl = np.mean([r.total_pnl for r in subset])
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_pf = np.mean([r.profit_factor for r in subset])
            avg_trades = np.mean([r.trades for r in subset])
            print(f"{tw:>6.1f} ${avg_pnl:>9,.0f} {avg_wr:>7.1f}% {avg_pf:>7.2f} {avg_trades:>10.0f}")

    # Threshold effect
    print(f"\n--- Threshold Effect (averaged across all weights + TOD) ---")
    print(f"{'Thresh':>6} {'Avg P&L':>10} {'Avg WR%':>8} {'Avg PF':>7} {'Avg Trades':>10}")
    for t in thresholds:
        subset = [r for r in valid if r.threshold == t]
        if subset:
            avg_pnl = np.mean([r.total_pnl for r in subset])
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_pf = np.mean([r.profit_factor for r in subset])
            avg_trades = np.mean([r.trades for r in subset])
            print(f"{t:>6.0f} ${avg_pnl:>9,.0f} {avg_wr:>7.1f}% {avg_pf:>7.2f} {avg_trades:>10.0f}")

    # TOD rule effect
    print(f"\n--- TOD Rule Effect (averaged across all weights + thresholds) ---")
    print(f"{'TOD Rule':<22} {'Avg P&L':>10} {'Avg WR%':>8} {'Avg PF':>7} {'Avg Trades':>10}")
    for tod in tod_rules:
        subset = [r for r in valid if r.tod_rule == tod]
        if subset:
            avg_pnl = np.mean([r.total_pnl for r in subset])
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_pf = np.mean([r.profit_factor for r in subset])
            avg_trades = np.mean([r.trades for r in subset])
            print(f"{tod:<22} ${avg_pnl:>9,.0f} {avg_wr:>7.1f}% {avg_pf:>7.2f} {avg_trades:>10.0f}")

    # Stop config effect
    print(f"\n--- Stop Config Effect (averaged across all weights + thresholds + TOD) ---")
    print(f"{'Stops':<14} {'Avg P&L':>10} {'Avg WR%':>8} {'Avg PF':>7} {'Avg Trades':>10} {'AvgLoss':>10}")
    for sc in stop_configs:
        subset = [r for r in valid if r.stop_config == sc]
        if subset:
            avg_pnl = np.mean([r.total_pnl for r in subset])
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_pf = np.mean([r.profit_factor for r in subset])
            avg_trades = np.mean([r.trades for r in subset])
            avg_loss = np.mean([r.avg_loss for r in subset if r.avg_loss != 0])
            print(f"{sc:<14} ${avg_pnl:>9,.0f} {avg_wr:>7.1f}% {avg_pf:>7.2f} {avg_trades:>10.0f} ${avg_loss:>9,.0f}")

    # Warmup delay effect
    print(f"\n--- Warmup Delay Effect (min minutes after open before trading) ---")
    print(f"{'Warmup':>7} {'Avg P&L':>10} {'Avg WR%':>8} {'Avg PF':>7} {'Avg Trades':>10}")
    for wu in warmup_delays:
        subset = [r for r in valid if r.min_warmup == wu]
        if subset:
            avg_pnl = np.mean([r.total_pnl for r in subset])
            avg_wr = np.mean([r.win_rate for r in subset])
            avg_pf = np.mean([r.profit_factor for r in subset])
            avg_trades = np.mean([r.trades for r in subset])
            print(f"{wu:>5}min ${avg_pnl:>9,.0f} {avg_wr:>7.1f}% {avg_pf:>7.2f} {avg_trades:>10.0f}")

    # THE WINNER
    best = valid[0]
    print(f"\n{'='*80}")
    print(f"OPTIMAL CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Tech Weight:    {best.tech_weight:.1f}")
    print(f"  ML Weight:      {best.ml_weight:.1f}")
    print(f"  Threshold:      {best.threshold:.0f}")
    print(f"  Stop Config:    {best.stop_config}")
    print(f"  TOD Rule:       {best.tod_rule}")
    print(f"  Warmup Delay:   {best.min_warmup} min after open")
    sc_detail = STOP_CONFIGS[best.stop_config]
    print(f"  Stop Levels:    0DTE tight={sc_detail.tight_stop_0dte_pct}% backstop={sc_detail.backstop_0dte_pct}% "
          f"| multi tight={sc_detail.tight_stop_multiday_pct}% backstop={sc_detail.backstop_multiday_pct}%")
    print(f"  ---")
    print(f"  Trades:         {best.trades}")
    print(f"  Win Rate:       {best.win_rate:.1f}%")
    print(f"  Total P&L:      ${best.total_pnl:,.2f}")
    print(f"  Profit Factor:  {best.profit_factor:.2f}")
    print(f"  Sharpe:         {best.sharpe:.2f}")
    print(f"  Max Drawdown:   {best.max_drawdown:.1f}%")
    print(f"  Final Portfolio: ${best.final_portfolio:,.2f}")
    print(f"\n  Combined score formula:")
    print(f"    score = {best.tech_weight:.1f} × tech_score + {best.ml_weight:.1f} × (ml_confidence × 100)")
    if best.tod_rule != "all_sessions":
        print(f"    TOD adjustments: {TOD_RULES[best.tod_rule]}")
    print(f"    Accept trade when score >= {best.threshold:.0f}")

    # Save full results to CSV for further analysis
    csv_path = str(PROJECT_DIR / "journal" / "sweep_results.csv")
    rows = []
    for r in results:
        rows.append({
            "tech_weight": r.tech_weight,
            "ml_weight": r.ml_weight,
            "threshold": r.threshold,
            "stop_config": r.stop_config,
            "tod_rule": r.tod_rule,
            "min_warmup": r.min_warmup,
            "trades": r.trades,
            "wins": r.wins,
            "win_rate": round(r.win_rate, 1),
            "total_pnl": round(r.total_pnl, 2),
            "profit_factor": round(r.profit_factor, 2),
            "avg_win": round(r.avg_win, 2),
            "avg_loss": round(r.avg_loss, 2),
            "max_drawdown": round(r.max_drawdown, 1),
            "sharpe": round(r.sharpe, 2),
            "final_portfolio": round(r.final_portfolio, 2),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Combined scoring parameter sweep")
    parser.add_argument("--start", default="2026-01-02", help="Start date")
    parser.add_argument("--end", default="2026-05-21", help="End date")
    parser.add_argument("--phase1-only", action="store_true", help="Only pre-compute candidates")
    parser.add_argument("--phase2-only", action="store_true", help="Only run sweep (from cache)")
    parser.add_argument("--veto-gates", action="store_true",
                        help="Enable Simpsons-inspired veto gates (sweep, warmup, momentum, etc)")
    parser.add_argument("--compare", action="store_true",
                        help="Run sweep BOTH with and without veto gates and compare")
    args = parser.parse_args()

    if args.phase2_only:
        with open(CANDIDATES_CACHE) as f:
            candidates = json.load(f)
        print(f"Loaded {len(candidates)} cached candidates")

        if args.compare:
            print("\n" + "=" * 80)
            print("COMPARISON: WITHOUT VETO GATES")
            print("=" * 80)
            phase2_sweep(candidates, enable_veto_gates=False)

            print("\n\n" + "=" * 80)
            print("COMPARISON: WITH VETO GATES (Simpsons-inspired)")
            print("=" * 80)
            phase2_sweep(candidates, enable_veto_gates=True)
        else:
            phase2_sweep(candidates, enable_veto_gates=args.veto_gates)
        return

    candidates = phase1_precompute(args.start, args.end)

    if args.phase1_only:
        return

    if args.compare:
        print("\n" + "=" * 80)
        print("COMPARISON: WITHOUT VETO GATES")
        print("=" * 80)
        phase2_sweep(candidates, enable_veto_gates=False)

        print("\n\n" + "=" * 80)
        print("COMPARISON: WITH VETO GATES (Simpsons-inspired)")
        print("=" * 80)
        phase2_sweep(candidates, enable_veto_gates=True)
    else:
        phase2_sweep(candidates, enable_veto_gates=args.veto_gates)


if __name__ == "__main__":
    main()
