"""Backtest ML peak detector as a scaleout gate.

Compares two strategies:
  A) Production FSM (baseline) — current V5/V6 exit engine
  B) Production FSM + ML peak scaleout — when the peak detector says
     "near peak" (prob > threshold), sell a fraction of contracts early

The ML gate does NOT replace any existing exit logic — it adds a partial
profit-lock on top. If the model triggers at +80% gain, we sell 1/3 of
contracts at that point and let the rest ride with normal trailing.

Usage:
    python scripts/backtest_ml_peak_scaleout.py
    python scripts/backtest_ml_peak_scaleout.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    TickerCategory,
    V5Config,
    categorize_ticker,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

# --- Production settings (match docker-compose.yml) ---
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
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models"

PORTFOLIO = 8000

SCORE_TIERS = [
    (135, 1.00),
    (120, 0.85),
    (100, 0.85),
    (90, 0.50),
    (78, 0.25),
]


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# --- Data loading (from backtest_v5_production.py) ---

def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 70
        ORDER BY created_at
    """).fetchall()
    signals = []
    for r in rows:
        sig = dict(r)
        sig["premium"] = sig["atm_premium"] or sig["otm_premium"]
        sent = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        sig["option_type"] = "put" if sent in ("bearish", "put") else "call"
        if sig["premium"] and sig["premium"] > 0 and sig["strike"]:
            signals.append(sig)
    conn.close()
    return signals


def build_contract_ticker(ticker, expiry, strike, option_type):
    if not expiry:
        return ""
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        return ""
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type.lower() in ("call", "bullish", "c") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def load_ticks(harvester_conn, signal):
    ticker = signal["ticker"]
    strike = signal["strike"]
    created_at = signal["created_at"]
    option_type = signal["option_type"]
    sig_date = created_at[:10]

    sig_dt = datetime.strptime(sig_date, "%Y-%m-%d").date()
    candidates = [sig_dt]
    for delta in range(1, 6):
        d = sig_dt + timedelta(days=delta)
        if d.weekday() < 5:
            candidates.append(d)
            if len(candidates) >= 4:
                break

    for exp_date in candidates:
        expiry = exp_date.strftime("%Y-%m-%d")
        ct = build_contract_ticker(ticker, expiry, strike, option_type)
        if not ct:
            continue
        rows = harvester_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume, day_vwap, open_interest, bid_size, ask_size
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()
        if rows and len(rows) >= 10:
            signal["_dte"] = (exp_date - sig_dt).days
            signal["_expiry_date"] = expiry
            break
    else:
        return None

    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega", "volume",
        "vwap", "open_interest", "bid_size", "ask_size",
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"], format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# --- ML Feature extraction (must match train_peak_detector_v2.py) ---

def compute_ml_features(df, idx, entry_premium, ticker, is_call, strike):
    """Compute ML features for a single tick. Returns dict or None."""
    if idx < 10:
        return None

    premiums = df["premium"].values.astype(float)
    premium = premiums[idx]
    if np.isnan(premium) or premium <= 0:
        return None

    timestamps = pd.to_datetime(df["ts"]).values

    # Running min/max
    running_min = np.minimum.accumulate(premiums[:idx + 1])
    running_max = np.maximum.accumulate(premiums[:idx + 1])

    ref_price = running_min[idx]
    if ref_price <= 0:
        return None

    gain_pct = (premium - ref_price) / ref_price * 100
    if gain_pct < 15:  # MIN_GAIN_TO_INCLUDE from training
        return None

    now = timestamps[idx]
    now_dt = pd.Timestamp(now)
    et_hour = now_dt.hour - 4
    et_minute = now_dt.minute
    if et_hour < 0:
        et_hour += 24

    et_decimal = et_hour + et_minute / 60
    if et_decimal < 9.5 or et_decimal > 16.0:
        return None

    session_peak = premiums[:idx + 1].max()
    n = len(premiums)
    category = categorize_ticker(ticker)

    # Velocities
    velocities = {}
    for lb in [3, 5, 10, 20]:
        if idx >= lb:
            prev_prem = premiums[idx - lb]
            velocities[f"vel_{lb}"] = (premium - prev_prem) / prev_prem * 100 if prev_prem > 0 else 0
        else:
            velocities[f"vel_{lb}"] = 0

    accel_5 = velocities["vel_3"] - velocities["vel_5"]

    # Premium percentile
    window = min(30, idx)
    recent = premiums[idx - window:idx + 1]
    recent_valid = recent[~np.isnan(recent)]
    if len(recent_valid) > 1:
        rng = recent_valid.max() - recent_valid.min()
        prem_percentile = (premium - recent_valid.min()) / rng if rng > 0 else 0.5
        prem_std = np.std(recent_valid) / premium if premium > 0 else 0
    else:
        prem_percentile = 0.5
        prem_std = 0

    pct_of_session_peak = premium / session_peak * 100 if session_peak > 0 else 0
    mfe_so_far = (running_max[idx] - running_min[idx]) / running_min[idx] * 100 if running_min[idx] > 0 else 0
    retrace_from_peak = (running_max[idx] - premium) / running_max[idx] * 100 if running_max[idx] > 0 else 0

    # Greeks
    delta_val = safe_float(df["delta"].iloc[idx])
    gamma_val = safe_float(df["gamma"].iloc[idx])
    theta_val = safe_float(df["theta"].iloc[idx])
    vega_val = safe_float(df["vega"].iloc[idx])
    iv_val = safe_float(df["iv"].iloc[idx])

    delta_prev = safe_float(df["delta"].iloc[idx - 5], delta_val)
    gamma_prev = safe_float(df["gamma"].iloc[idx - 5], gamma_val)
    theta_prev = safe_float(df["theta"].iloc[idx - 5], theta_val)
    iv_prev = safe_float(df["iv"].iloc[idx - 5], iv_val)

    delta_change = delta_val - delta_prev
    gamma_change = gamma_val - gamma_prev
    theta_change = theta_val - theta_prev
    iv_change = iv_val - iv_prev

    iv_window = min(50, idx)
    iv_values = [safe_float(df["iv"].iloc[i]) for i in range(idx - iv_window, idx + 1)]
    iv_values = [v for v in iv_values if v > 0]
    iv_percentile = sum(1 for v in iv_values if v <= iv_val) / len(iv_values) if iv_values else 0.5

    gamma_delta_ratio = gamma_val / abs(delta_val) if abs(delta_val) > 0.01 else 0

    # Volume
    vol = safe_float(df["volume"].iloc[idx])
    if idx >= 10:
        prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
        prev_vols = [v for v in prev_vols if v > 0]
        avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    if idx >= 20:
        vol_recent = [safe_float(df["volume"].iloc[i]) for i in range(idx - 5, idx + 1)]
        vol_older = [safe_float(df["volume"].iloc[i]) for i in range(idx - 20, idx - 10)]
        avg_recent = np.mean([v for v in vol_recent if v > 0] or [0])
        avg_older = np.mean([v for v in vol_older if v > 0] or [1])
        vol_trend = avg_recent / avg_older if avg_older > 0 else 1.0
    else:
        vol_trend = 1.0

    # Bid-ask
    bid = safe_float(df["bid"].iloc[idx], premium * 0.95)
    ask = safe_float(df["ask"].iloc[idx], premium * 1.05)
    spread_pct = (ask - bid) / premium * 100 if premium > 0 else 0

    if idx >= 5:
        prev_bid = safe_float(df["bid"].iloc[idx - 5], bid)
        prev_ask = safe_float(df["ask"].iloc[idx - 5], ask)
        prev_premium = premiums[idx - 5]
        prev_spread = (prev_ask - prev_bid) / prev_premium * 100 if prev_premium > 0 else spread_pct
        spread_change = spread_pct - prev_spread
    else:
        spread_change = 0

    bid_size = safe_float(df.get("bid_size", pd.Series([1])).iloc[idx] if "bid_size" in df.columns else 1, 1)
    ask_size = safe_float(df.get("ask_size", pd.Series([1])).iloc[idx] if "ask_size" in df.columns else 1, 1)
    size_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0

    # VWAP
    vwap = safe_float(df["vwap"].iloc[idx] if "vwap" in df.columns else 0)
    vwap_divergence = (premium - vwap) / vwap * 100 if vwap > 0 else 0

    # OI
    oi = safe_float(df["open_interest"].iloc[idx] if "open_interest" in df.columns else 0)
    oi_vol_ratio = vol / oi if oi > 0 else 0

    # Underlying
    underlying = safe_float(df["underlying_price"].iloc[idx])
    first_u = 0.0
    for i in range(min(5, idx + 1)):
        u = safe_float(df["underlying_price"].iloc[i])
        if u > 0:
            first_u = u
            break

    u_move_entry = (underlying - first_u) / first_u * 100 if first_u > 0 else 0

    if idx >= 5:
        prev_u = safe_float(df["underlying_price"].iloc[idx - 5], underlying)
        u_vel = (underlying - prev_u) / prev_u * 100 if prev_u > 0 else 0
    else:
        u_vel = 0

    if underlying > 0 and strike > 0:
        moneyness = ((underlying - strike) / strike * 100) if is_call else ((strike - underlying) / strike * 100)
    else:
        moneyness = 0

    prem_vel = velocities.get("vel_5", 0)
    divergence = (prem_vel - u_vel * 50) if is_call else (prem_vel + u_vel * 50)

    elapsed_min = (now - timestamps[0]).astype("timedelta64[s]").astype(float) / 60
    minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + et_minute))
    session_progress = max(0, min(1, (et_decimal - 9.5) / 6.5))

    return [
        gain_pct, velocities["vel_3"], velocities["vel_5"],
        velocities["vel_10"], velocities["vel_20"], accel_5,
        prem_percentile, prem_std, pct_of_session_peak, mfe_so_far,
        retrace_from_peak,
        abs(delta_val), gamma_val, theta_val, vega_val, iv_val,
        delta_change, gamma_change, theta_change, iv_change,
        iv_percentile, gamma_delta_ratio,
        vol, vol_ratio, vol_trend,
        u_move_entry, u_vel, moneyness, divergence,
        spread_pct, spread_change, size_imbalance,
        vwap_divergence, oi_vol_ratio,
        elapsed_min, minutes_to_close, session_progress,
        1 if category == TickerCategory.HIGH_VOL else 0,
        1 if category == TickerCategory.INDEX else 0,
        1 if is_call else 0,
    ]


# --- Simulation ---

def simulate_baseline(df, entry_premium, contracts, direction, dte, expiry_date, ticker):
    """Run production FSM (baseline — no ML gate)."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

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
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def simulate_ml_scaleout(df, entry_premium, contracts, direction, dte, expiry_date,
                          ticker, ml_model, threshold, scaleout_fraction, min_gain_for_ml):
    """Run production FSM + ML peak scaleout gate.

    When ML model predicts near-peak (prob > threshold) AND trade is up > min_gain_for_ml,
    sell scaleout_fraction of remaining contracts at current price.
    ML scaleout fires at most once per trade.
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "ml_triggered": False, "ml_sell_gain": 0, "ml_prob": 0}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type == "call"
    strike = float(df.get("strike", pd.Series([0])).iloc[0]) if "strike" in df.columns else 0

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

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
    ml_triggered = False
    ml_sell_gain = 0
    ml_prob = 0
    ml_sell_prem = 0

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium

        now = df["ts"].iloc[idx]
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        # --- ML peak scaleout check (before FSM, fires once) ---
        if not ml_triggered and remaining >= 2:
            gain_pct = (premium - entry_premium) / entry_premium * 100
            if gain_pct >= min_gain_for_ml:
                # Get strike from signal (stored externally)
                features = compute_ml_features(df, idx, entry_premium, ticker, is_call,
                                                strike if strike > 0 else first_underlying)
                if features is not None:
                    features_arr = np.array([features])
                    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    prob = ml_model.predict(features_arr)[0]
                    if prob >= threshold:
                        # ML says near peak — sell fraction
                        to_sell = max(1, int(remaining * scaleout_fraction))
                        if to_sell < remaining:  # keep at least 1
                            locked_pnl += (premium - entry_premium) * to_sell * 100
                            remaining -= to_sell
                            state.contracts = remaining
                            ml_triggered = True
                            ml_sell_gain = gain_pct
                            ml_prob = prob
                            ml_sell_prem = premium

        # --- Normal FSM evaluation ---
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain,
                    "ml_triggered": ml_triggered, "ml_sell_gain": ml_sell_gain,
                    "ml_prob": ml_prob, "ml_sell_prem": ml_sell_prem}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain,
            "ml_triggered": ml_triggered, "ml_sell_gain": ml_sell_gain,
            "ml_prob": ml_prob, "ml_sell_prem": ml_sell_prem}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="ML probability threshold (default: sweep 0.3-0.8)")
    parser.add_argument("--fraction", type=float, default=0.333,
                        help="Fraction of contracts to sell on ML trigger")
    parser.add_argument("--min-gain", type=float, default=25.0,
                        help="Minimum gain %% before ML gate can fire")
    args = parser.parse_args()

    # Load ML model
    model_path = MODEL_DIR / "peak_detector_v2_cls.txt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python scripts/train_peak_detector_v2.py")
        return

    ml_model = lgb.Booster(model_file=str(model_path))
    print(f"Loaded ML model: {model_path}")

    # Load signals + harvester
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Prepare trades
    trades = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        strike = sig.get("strike", 0) or 0

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        # Sizing
        max_risk_pct = 0.75
        max_concurrent = 4
        max_position_pct = 0.15
        deployable = PORTFOLIO * max_risk_pct
        per_slot = deployable / max_concurrent
        position_cap = PORTFOLIO * max_position_pct

        score_mult = 0.25
        for thresh, mult in SCORE_TIERS:
            if score >= thresh:
                score_mult = mult
                break

        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_contracts, pos_cap_contracts))

        # Store strike in df for ML feature computation
        df["strike"] = strike

        trades.append({
            "sig": sig,
            "df": df,
            "ticker": ticker,
            "direction": direction,
            "score": score,
            "day": sig["created_at"][:10],
            "entry": adj_entry,
            "contracts": contracts,
            "dte": dte,
            "expiry_date": expiry_date,
            "strike": strike,
        })

    harvester_conn.close()
    print(f"Prepared {len(trades)} trades ({no_data} skipped, no tick data)")

    # --- Run threshold sweep or single threshold ---
    thresholds = [args.threshold] if args.threshold else [0.5, 0.6, 0.7, 0.8]

    # First compute baseline once
    print(f"\n{'=' * 110}")
    print("BASELINE: Production FSM (no ML gate)")
    print(f"{'=' * 110}")

    baseline_results = []
    for t in trades:
        result = simulate_baseline(
            t["df"], t["entry"], t["contracts"], t["direction"],
            t["dte"], t["expiry_date"], t["ticker"]
        )
        result["ticker"] = t["ticker"]
        result["day"] = t["day"]
        result["score"] = t["score"]
        result["entry"] = t["entry"]
        result["contracts"] = t["contracts"]
        baseline_results.append(result)

    bl_df = pd.DataFrame(baseline_results)
    bl_pnl = bl_df["pnl"].sum()
    bl_wins = (bl_df["pnl"] > 0).sum()
    bl_losses = (bl_df["pnl"] <= 0).sum()
    bl_wr = bl_wins / len(bl_df) * 100

    print(f"  Trades: {len(bl_df)} | P&L: ${bl_pnl:,.2f} | "
          f"WR: {bl_wr:.1f}% ({bl_wins}W/{bl_losses}L) | "
          f"Avg: ${bl_df['pnl'].mean():,.2f}")

    # --- ML Scaleout sweep ---
    print(f"\n{'=' * 110}")
    print(f"ML PEAK SCALEOUT — fraction={args.fraction:.0%}, min_gain={args.min_gain:.0f}%")
    print(f"{'=' * 110}")

    sweep_results = []

    for threshold in thresholds:
        ml_results = []
        for t in trades:
            result = simulate_ml_scaleout(
                t["df"], t["entry"], t["contracts"], t["direction"],
                t["dte"], t["expiry_date"], t["ticker"],
                ml_model, threshold, args.fraction, args.min_gain,
            )
            result["ticker"] = t["ticker"]
            result["day"] = t["day"]
            result["score"] = t["score"]
            result["entry"] = t["entry"]
            result["contracts"] = t["contracts"]
            ml_results.append(result)

        ml_df = pd.DataFrame(ml_results)
        ml_pnl = ml_df["pnl"].sum()
        ml_wins = (ml_df["pnl"] > 0).sum()
        ml_losses = (ml_df["pnl"] <= 0).sum()
        ml_wr = ml_wins / len(ml_df) * 100
        n_triggered = ml_df["ml_triggered"].sum()
        delta_pnl = ml_pnl - bl_pnl

        print(f"\n  Threshold {threshold:.1f}: "
              f"P&L ${ml_pnl:,.2f} ({'+' if delta_pnl >= 0 else ''}{delta_pnl:,.2f} vs baseline) | "
              f"WR {ml_wr:.1f}% | ML triggered {n_triggered}/{len(ml_df)} trades")

        # Per-trade detail for triggered trades
        triggered = ml_df[ml_df["ml_triggered"] == True]
        if len(triggered) > 0:
            avg_sell_gain = triggered["ml_sell_gain"].mean()
            avg_prob = triggered["ml_prob"].mean()
            print(f"    ML avg trigger: +{avg_sell_gain:.0f}% gain, {avg_prob:.2f} prob")

            # Compare triggered trades: ML vs baseline
            for i, (_, row) in enumerate(triggered.iterrows()):
                bl_row = baseline_results[ml_results.index(ml_results[row.name])] if row.name < len(baseline_results) else None
                idx_in_trades = next((j for j, r in enumerate(ml_results) if r is ml_results[row.name]), None)

        sweep_results.append({
            "threshold": threshold,
            "pnl": ml_pnl,
            "delta_pnl": delta_pnl,
            "win_rate": ml_wr,
            "n_triggered": n_triggered,
            "avg_trigger_gain": triggered["ml_sell_gain"].mean() if len(triggered) > 0 else 0,
        })

    # --- Summary table ---
    print(f"\n{'=' * 110}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 110}")
    print(f"\n  {'Threshold':>10} {'P&L':>12} {'vs Baseline':>14} {'WR':>7} {'ML Fires':>10} {'Avg Trigger':>13}")
    print(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*7} {'-'*10} {'-'*13}")
    print(f"  {'baseline':>10} ${bl_pnl:>10,.2f} {'—':>14} {bl_wr:>6.1f}% {'—':>10} {'—':>13}")
    for sr in sweep_results:
        sign = "+" if sr["delta_pnl"] >= 0 else ""
        print(f"  {sr['threshold']:>10.1f} ${sr['pnl']:>10,.2f} "
              f"{sign}${sr['delta_pnl']:>10,.2f} {sr['win_rate']:>6.1f}% "
              f"{sr['n_triggered']:>10} +{sr['avg_trigger_gain']:.0f}%")

    # --- Per-trade comparison for best threshold ---
    if sweep_results:
        best = max(sweep_results, key=lambda x: x["delta_pnl"])
        if best["delta_pnl"] > 0:
            print(f"\n{'=' * 110}")
            print(f"BEST: threshold={best['threshold']:.1f} → +${best['delta_pnl']:,.2f} vs baseline")
            print(f"{'=' * 110}")

            # Re-run best threshold for per-trade detail
            print(f"\n  {'Ticker':<8} {'Day':>12} {'Contracts':>10} {'Base P&L':>10} {'ML P&L':>10} "
                  f"{'Delta':>10} {'ML Gain%':>10} {'Peak%':>8}")
            print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

            for i, t in enumerate(trades):
                bl = baseline_results[i]
                ml = simulate_ml_scaleout(
                    t["df"], t["entry"], t["contracts"], t["direction"],
                    t["dte"], t["expiry_date"], t["ticker"],
                    ml_model, best["threshold"], args.fraction, args.min_gain,
                )
                delta = ml["pnl"] - bl["pnl"]
                if abs(delta) > 0.01:  # only show trades where ML made a difference
                    ml_gain_str = f"+{ml['ml_sell_gain']:.0f}%" if ml["ml_triggered"] else "—"
                    print(f"  {t['ticker']:<8} {t['day']:>12} {t['contracts']:>10} "
                          f"${bl['pnl']:>9,.2f} ${ml['pnl']:>9,.2f} "
                          f"{'+'if delta>=0 else ''}{delta:>9,.2f} "
                          f"{ml_gain_str:>10} "
                          f"{bl['peak_gain']:>7.0f}%")

            # Count improvements vs degradations
            improvements = sum(1 for i in range(len(trades)) if
                simulate_ml_scaleout(
                    trades[i]["df"], trades[i]["entry"], trades[i]["contracts"],
                    trades[i]["direction"], trades[i]["dte"], trades[i]["expiry_date"],
                    trades[i]["ticker"], ml_model, best["threshold"], args.fraction, args.min_gain
                )["pnl"] > baseline_results[i]["pnl"])
            degradations = sum(1 for i in range(len(trades)) if
                simulate_ml_scaleout(
                    trades[i]["df"], trades[i]["entry"], trades[i]["contracts"],
                    trades[i]["direction"], trades[i]["dte"], trades[i]["expiry_date"],
                    trades[i]["ticker"], ml_model, best["threshold"], args.fraction, args.min_gain
                )["pnl"] < baseline_results[i]["pnl"])
            print(f"\n  Trades improved: {improvements} | Trades hurt: {degradations} | "
                  f"Unchanged: {len(trades) - improvements - degradations}")
        else:
            print(f"\n  No threshold improved over baseline. ML scaleout not helpful for these signals.")


if __name__ == "__main__":
    main()
