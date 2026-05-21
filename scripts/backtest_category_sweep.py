"""Comprehensive category-aware strategy sweep.

Tests dozens of exit strategy configurations across trade categories:
  - 0DTE vs multi-day
  - High-vol memes (MSTR, AMD, TSLA, NVDA) vs indexes (SPY, QQQ, IWM)
  - Morning (before noon ET) vs afternoon entries
  - Score tiers (95+, 90-94, 85-89, <85)

Goal: find per-category exit configs that produce consistent 15-25%+ daily
returns WITHOUT relying on moonshot outliers.

Usage:
    python scripts/backtest_category_sweep.py
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000

# Ticker categories
HIGH_VOL = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEXES = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
# Everything else = "standard"


def categorize_ticker(ticker):
    if ticker in HIGH_VOL:
        return "high_vol"
    if ticker in INDEXES:
        return "index"
    return "standard"


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
                   implied_volatility, delta, gamma, theta, vega, day_volume
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()
        if rows and len(rows) >= 10:
            signal["_dte"] = (exp_date - sig_dt).days
            break
    else:
        return None

    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega", "volume"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def compute_candle_features(df, idx, lookback=30):
    """Compute simple candle/momentum features from tick data.

    Returns dict with: rsi_proxy, momentum_5m, momentum_15m, vol_trend,
    underlying_trend, is_exhausted
    """
    features = {}

    start = max(0, idx - lookback)
    prices = df["underlying_price"].iloc[start:idx+1].values
    prices = prices[prices > 0]
    premiums = df["premium"].iloc[start:idx+1].values
    premiums = premiums[~np.isnan(premiums)]

    # RSI proxy: ratio of up moves to total moves in lookback
    if len(prices) > 5:
        changes = np.diff(prices)
        ups = np.sum(changes[changes > 0])
        downs = -np.sum(changes[changes < 0])
        if ups + downs > 0:
            features["rsi_proxy"] = ups / (ups + downs) * 100
        else:
            features["rsi_proxy"] = 50.0
    else:
        features["rsi_proxy"] = 50.0

    # Momentum: % change over N ticks (each ~30s)
    def mom(arr, n):
        if len(arr) > n and arr[-n-1] > 0:
            return (arr[-1] - arr[-n-1]) / arr[-n-1] * 100
        return 0.0

    features["u_mom_5m"] = mom(prices, 10)   # ~5 min at 30s ticks
    features["u_mom_15m"] = mom(prices, 30)  # ~15 min
    features["p_mom_5m"] = mom(premiums, 10)
    features["p_mom_15m"] = mom(premiums, 30)

    # Volume trend (if available)
    vols = df["volume"].iloc[start:idx+1].values
    vols = vols[vols > 0] if len(vols) > 0 else np.array([0])
    if len(vols) > 10:
        recent = np.mean(vols[-5:])
        older = np.mean(vols[:5])
        features["vol_trend"] = recent / older if older > 0 else 1.0
    else:
        features["vol_trend"] = 1.0

    # Exhaustion signal: RSI > 70 AND premium momentum slowing AND volume spike
    features["is_exhausted"] = (
        features["rsi_proxy"] > 70 and
        features["p_mom_5m"] < features["p_mom_15m"] * 0.5 and
        features["vol_trend"] > 2.0
    )

    # IV from harvester
    iv = df["iv"].iloc[idx] if "iv" in df.columns else 0
    features["iv"] = float(iv) if iv and not np.isnan(iv) else 0.3

    return features


def simulate_trade(df, entry_idx, entry_premium, contracts, direction, dte, config):
    """Simulate a trade with configurable exit strategy.

    Config keys:
        hard_stop: float (% drop from entry to trigger hard stop)
        tight_stop: float (% drop with underlying against)
        backstop: float (% drop without underlying against, fallback)
        checkpoint_enabled: bool
        checkpoint_drop: float
        checkpoint_u: float (underlying threshold)
        scalp_peak_trigger: float (% peak gain to arm scalp trail)
        scalp_fade_pct: float (fade from peak to trigger, e.g., 0.6 = 60% of peak)
        scalp_0dte_mode: str ("not_confirming" or "against")
        scalp_multiday_mode: str ("against" or "disabled")
        soft_trail_min: float (min peak gain for soft trail)
        soft_trail_max: float (max peak gain for soft trail)
        soft_trail_keep: float (% of gains to keep)
        adaptive_tiers: list of (min_gain, trail_width) tuples
        theta_minutes: float (0 = disabled)
        theta_loss_pct: float (must be losing by this %)
        eod_0dte: bool
        max_loss_dollars: float (0 = disabled)
        profit_target_pct: float (0 = disabled, take profit at this %)
        exhaust_tighten: float (multiply trail by this when exhausted, e.g., 0.7)
        grace_minutes: float
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    is_call = direction in ("bullish", "call", "long")
    peak = entry_premium
    entry_underlying = None
    cost = entry_premium * contracts * 100

    hard_stop = config.get("hard_stop", 30)
    tight_stop = config.get("tight_stop", 35)
    backstop = config.get("backstop", 65)
    checkpoint_enabled = config.get("checkpoint_enabled", True)
    checkpoint_drop = config.get("checkpoint_drop", 30)
    checkpoint_u = config.get("checkpoint_u", 0.5)
    scalp_peak_trigger = config.get("scalp_peak_trigger", 20)
    scalp_fade_pct = config.get("scalp_fade_pct", 0.6)
    scalp_0dte = config.get("scalp_0dte_mode", "not_confirming")
    scalp_multi = config.get("scalp_multiday_mode", "against")
    soft_trail_min = config.get("soft_trail_min", 15)
    soft_trail_max = config.get("soft_trail_max", 50)
    soft_trail_keep = config.get("soft_trail_keep", 0.50)
    adaptive_tiers = config.get("adaptive_tiers", [
        (40, 40), (150, 45), (400, 30)
    ])
    theta_minutes = config.get("theta_minutes", 0)
    theta_loss_pct = config.get("theta_loss_pct", 0)
    eod_0dte = config.get("eod_0dte", True)
    max_loss_dollars = config.get("max_loss_dollars", 0)
    profit_target_pct = config.get("profit_target_pct", 0)
    exhaust_tighten = config.get("exhaust_tighten", 1.0)
    grace_minutes = config.get("grace_minutes", 5)

    for idx in range(entry_idx + 1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        if premium > peak:
            peak = premium

        elapsed = (df["ts"].iloc[idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60
        gain_pct = (premium - entry_premium) / entry_premium * 100
        drop_entry = max(0, (entry_premium - premium) / entry_premium * 100)
        drop_peak = (peak - premium) / peak * 100 if peak > 0 else 0
        peak_gain = (peak - entry_premium) / entry_premium * 100
        current_pnl = (premium - entry_premium) * contracts * 100

        # Underlying
        underlying = df["underlying_price"].iloc[idx] or 0
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying

        u_move = 0.0
        underlying_against = False
        underlying_confirms = False
        has_underlying = entry_underlying is not None and underlying > 0
        if has_underlying:
            u_move = (underlying - entry_underlying) / entry_underlying * 100
            if is_call:
                underlying_against = u_move < -checkpoint_u
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > checkpoint_u
                underlying_confirms = u_move < -0.2

        # EOD cutoff (0DTE only)
        if dte == 0 and eod_0dte:
            ts = df["ts"].iloc[idx]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts - timedelta(hours=4)
            if et.hour >= 15 and et.minute >= 45:
                return {"pnl": current_pnl, "reason": "eod_cutoff", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # Grace period
        if elapsed < grace_minutes:
            # During grace, only max loss cap applies
            if max_loss_dollars > 0 and current_pnl < -max_loss_dollars:
                return {"pnl": current_pnl, "reason": "max_loss_cap", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
            continue

        # --- MAX LOSS CAP (absolute dollar limit) ---
        if max_loss_dollars > 0 and current_pnl < -max_loss_dollars:
            return {"pnl": current_pnl, "reason": "max_loss_cap", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

        # --- PROFIT TARGET (take profit at fixed %) ---
        if profit_target_pct > 0 and gain_pct >= profit_target_pct:
            return {"pnl": current_pnl, "reason": "profit_target", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

        # --- CANDLE/MARKET FEATURES ---
        candle = None
        if exhaust_tighten < 1.0 and idx % 10 == 0:  # check every ~5min
            candle = compute_candle_features(df, idx)
        trail_mult = 1.0
        if candle and candle["is_exhausted"]:
            trail_mult = exhaust_tighten

        # --- SCALP TRAIL ---
        if peak_gain >= scalp_peak_trigger and gain_pct > 0 and gain_pct < peak_gain * scalp_fade_pct:
            should_scalp = False
            if dte == 0:
                if scalp_0dte == "not_confirming" and not underlying_confirms:
                    should_scalp = True
                elif scalp_0dte == "against" and underlying_against:
                    should_scalp = True
                elif scalp_0dte == "always":
                    should_scalp = True
            else:
                if scalp_multi == "against" and underlying_against:
                    should_scalp = True
                elif scalp_multi == "not_confirming" and not underlying_confirms:
                    should_scalp = True

            if should_scalp:
                return {"pnl": current_pnl, "reason": "scalp_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # --- CHECKPOINT (0DTE only by default) ---
        if checkpoint_enabled and dte == 0:
            if drop_entry >= checkpoint_drop and has_underlying and underlying_against:
                return {"pnl": current_pnl, "reason": "checkpoint_cut", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # --- GRADUATED STOP ---
        if has_underlying:
            if underlying_against:
                effective_tight = tight_stop
                if drop_entry >= effective_tight:
                    return {"pnl": current_pnl, "reason": "confirmed_stop", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
            else:
                if drop_entry >= backstop:
                    return {"pnl": current_pnl, "reason": "hard_stop", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
        else:
            # No underlying data — use simple hard stop
            if drop_entry >= hard_stop:
                return {"pnl": current_pnl, "reason": "hard_stop", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # --- SOFT TRAIL ---
        if soft_trail_min <= peak_gain < soft_trail_max:
            floor = entry_premium + (peak - entry_premium) * soft_trail_keep
            if premium <= floor:
                return {"pnl": current_pnl, "reason": "soft_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # --- ADAPTIVE TRAIL (tiered) ---
        for tier_min, tier_width in sorted(adaptive_tiers, key=lambda x: -x[0]):
            if peak_gain >= tier_min:
                effective_width = tier_width * trail_mult
                if drop_peak >= effective_width:
                    return {"pnl": current_pnl, "reason": f"adaptive_{tier_min}", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
                break

        # --- THETA TIMER ---
        if theta_minutes > 0 and elapsed >= theta_minutes:
            if gain_pct < -theta_loss_pct:
                return {"pnl": current_pnl, "reason": "theta_timer", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

    # End of data
    last_idx = len(df) - 1
    exit_prem = df["premium"].iloc[last_idx]
    elapsed = (df["ts"].iloc[last_idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60
    pnl = (exit_prem - entry_premium) * contracts * 100
    peak_g = (peak - entry_premium) / entry_premium * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": exit_prem, "peak_gain": peak_g}


def size_contracts(score, entry_premium, portfolio=PORTFOLIO):
    """Score-based sizing matching production."""
    if score >= 95:
        contracts = 5
    elif score >= 90:
        contracts = 4
    elif score >= 85:
        contracts = 3
    else:
        contracts = 1
    # Cap by portfolio risk
    max_cost = portfolio * 0.20  # 20% max per trade
    cost = entry_premium * contracts * 100
    while cost > max_cost and contracts > 1:
        contracts -= 1
        cost = entry_premium * contracts * 100
    return contracts


# ============================================================================
# STRATEGY CONFIGURATIONS
# ============================================================================

def make_config(name, **overrides):
    """Create a named config with defaults + overrides."""
    base = {
        "hard_stop": 30,
        "tight_stop": 35,
        "backstop": 65,
        "checkpoint_enabled": True,
        "checkpoint_drop": 30,
        "checkpoint_u": 0.5,
        "scalp_peak_trigger": 20,
        "scalp_fade_pct": 0.6,
        "scalp_0dte_mode": "not_confirming",
        "scalp_multiday_mode": "against",
        "soft_trail_min": 15,
        "soft_trail_max": 50,
        "soft_trail_keep": 0.50,
        "adaptive_tiers": [(40, 40), (150, 45), (400, 30)],
        "theta_minutes": 0,
        "theta_loss_pct": 0,
        "eod_0dte": True,
        "max_loss_dollars": 0,
        "profit_target_pct": 0,
        "exhaust_tighten": 1.0,
        "grace_minutes": 5,
    }
    base.update(overrides)
    base["_name"] = name
    return base


# --- Category-specific strategy menus ---

# 0DTE strategies (tighter stops, faster exits)
ZERO_DTE_CONFIGS = [
    make_config("0dte_base"),
    make_config("0dte_tight", hard_stop=25, tight_stop=30, backstop=50),
    make_config("0dte_theta60", theta_minutes=60, theta_loss_pct=5),
    make_config("0dte_theta45", theta_minutes=45, theta_loss_pct=5),
    make_config("0dte_theta90", theta_minutes=90, theta_loss_pct=10),
    make_config("0dte_profit25", profit_target_pct=25),
    make_config("0dte_profit40", profit_target_pct=40),
    make_config("0dte_profit60", profit_target_pct=60),
    make_config("0dte_maxloss300", max_loss_dollars=300),
    make_config("0dte_maxloss400", max_loss_dollars=400),
    make_config("0dte_maxloss500", max_loss_dollars=500),
    make_config("0dte_tight_theta60", hard_stop=25, tight_stop=30, backstop=50,
                theta_minutes=60, theta_loss_pct=5),
    make_config("0dte_tight_maxloss400", hard_stop=25, tight_stop=30, backstop=50,
                max_loss_dollars=400),
    make_config("0dte_scalp_always", scalp_0dte_mode="always", scalp_peak_trigger=15),
    make_config("0dte_conservative", hard_stop=25, tight_stop=30, backstop=50,
                theta_minutes=60, theta_loss_pct=5, max_loss_dollars=400,
                profit_target_pct=40),
    make_config("0dte_balanced", hard_stop=30, tight_stop=35, backstop=55,
                theta_minutes=90, theta_loss_pct=10, max_loss_dollars=500,
                soft_trail_keep=0.55),
    make_config("0dte_aggressive_trail", adaptive_tiers=[(30, 35), (100, 40), (300, 25)],
                soft_trail_min=10, soft_trail_keep=0.60),
    make_config("0dte_exhaust", exhaust_tighten=0.70),
    make_config("0dte_wide_soft", soft_trail_min=10, soft_trail_max=60, soft_trail_keep=0.55),
    make_config("0dte_target30_maxloss300", profit_target_pct=30, max_loss_dollars=300),
    make_config("0dte_target25_theta60", profit_target_pct=25, theta_minutes=60, theta_loss_pct=5),
]

# Multi-day strategies (wider stops, more patience)
MULTI_DAY_CONFIGS = [
    make_config("multi_base", tight_stop=52, backstop=75, checkpoint_enabled=False),
    make_config("multi_patient", tight_stop=52, backstop=75, checkpoint_enabled=False,
                scalp_multiday_mode="against", grace_minutes=15),
    make_config("multi_theta120", tight_stop=52, backstop=75, checkpoint_enabled=False,
                theta_minutes=120, theta_loss_pct=10),
    make_config("multi_theta180", tight_stop=52, backstop=75, checkpoint_enabled=False,
                theta_minutes=180, theta_loss_pct=15),
    make_config("multi_maxloss400", tight_stop=45, backstop=65, checkpoint_enabled=False,
                max_loss_dollars=400),
    make_config("multi_maxloss500", tight_stop=52, backstop=75, checkpoint_enabled=False,
                max_loss_dollars=500),
    make_config("multi_maxloss600", tight_stop=52, backstop=75, checkpoint_enabled=False,
                max_loss_dollars=600),
    make_config("multi_profit50", tight_stop=52, backstop=75, checkpoint_enabled=False,
                profit_target_pct=50),
    make_config("multi_profit75", tight_stop=52, backstop=75, checkpoint_enabled=False,
                profit_target_pct=75),
    make_config("multi_conservative", tight_stop=40, backstop=60, checkpoint_enabled=False,
                theta_minutes=120, theta_loss_pct=10, max_loss_dollars=400),
    make_config("multi_balanced", tight_stop=45, backstop=65, checkpoint_enabled=False,
                theta_minutes=150, theta_loss_pct=15, max_loss_dollars=500),
    make_config("multi_wide_trail", tight_stop=52, backstop=75, checkpoint_enabled=False,
                adaptive_tiers=[(40, 50), (150, 55), (400, 35)]),
    make_config("multi_tighter_stop", tight_stop=35, backstop=55, checkpoint_enabled=False,
                max_loss_dollars=400),
    make_config("multi_no_scalp", tight_stop=52, backstop=75, checkpoint_enabled=False,
                scalp_multiday_mode="disabled"),
]

# High-vol ticker strategies (wider trails for wild swings)
HIGHVOL_MODS = [
    {},  # no mod (use base)
    {"adaptive_tiers": [(40, 50), (150, 55), (400, 35)]},  # wider trails
    {"soft_trail_keep": 0.40},  # keep less (wider room)
    {"scalp_fade_pct": 0.5},   # scalp only on bigger fade
    {"scalp_peak_trigger": 30}, # higher peak before arming scalp
    {"exhaust_tighten": 0.70},  # tighten on exhaustion
]

# Index ticker strategies (tighter trails, more predictable)
INDEX_MODS = [
    {},  # no mod
    {"adaptive_tiers": [(35, 35), (100, 40), (300, 25)]},  # tighter trails
    {"soft_trail_keep": 0.60},  # keep more
    {"profit_target_pct": 30},  # take profits earlier
    {"scalp_peak_trigger": 15}, # arm scalp earlier
]


def run_sweep():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load all tick data
    print("Loading tick data for all signals...")
    trade_data = []
    no_data = 0
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        dte = sig.get("_dte", 0)

        # Entry price from harvester
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        contracts = size_contracts(score, adj_entry)

        # Determine time of day (ET)
        ts = df["ts"].iloc[0]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        et = ts - timedelta(hours=4)
        is_morning = et.hour < 12

        cat = categorize_ticker(ticker)

        trade_data.append({
            "sig": sig, "df": df, "ticker": ticker, "direction": direction,
            "score": score, "day": day, "dte": dte, "entry": adj_entry,
            "contracts": contracts, "is_morning": is_morning, "category": cat,
        })

    harvester_conn.close()
    print(f"Loaded {len(trade_data)} trades ({no_data} no data)")

    # Stats
    n_0dte = sum(1 for t in trade_data if t["dte"] == 0)
    n_multi = sum(1 for t in trade_data if t["dte"] > 0)
    n_hv = sum(1 for t in trade_data if t["category"] == "high_vol")
    n_idx = sum(1 for t in trade_data if t["category"] == "index")
    n_std = sum(1 for t in trade_data if t["category"] == "standard")
    print(f"  0DTE: {n_0dte}, Multi-day: {n_multi}")
    print(f"  High-vol: {n_hv}, Index: {n_idx}, Standard: {n_std}")

    all_days = sorted(set(t["day"] for t in trade_data))
    n_days = len(all_days)

    # ========================================================================
    # Phase 1: Find best BASE strategy per DTE category
    # ========================================================================
    print(f"\n{'='*120}")
    print("PHASE 1: Best base strategy per DTE category")
    print(f"{'='*120}")

    # Test each config on the appropriate DTE subset
    best_results = {}

    for dte_label, configs, dte_filter in [
        ("0DTE", ZERO_DTE_CONFIGS, lambda t: t["dte"] == 0),
        ("Multi-day", MULTI_DAY_CONFIGS, lambda t: t["dte"] > 0),
    ]:
        subset = [t for t in trade_data if dte_filter(t)]
        if not subset:
            print(f"\n  {dte_label}: No trades")
            continue

        print(f"\n  {dte_label} ({len(subset)} trades across {len(set(t['day'] for t in subset))} days)")
        print(f"  {'Config':<35} {'Total':>10} {'WR':>6} {'MaxLoss':>9} {'DaysWon':>8} {'Avg/Day':>8} {'NoOutlier':>10}")
        print(f"  {'-'*95}")

        config_scores = []
        for cfg in configs:
            results = []
            for t in subset:
                r = simulate_trade(t["df"], 0, t["entry"], t["contracts"],
                                   t["direction"], t["dte"], cfg)
                r.update({"ticker": t["ticker"], "day": t["day"], "score": t["score"],
                           "entry": t["entry"], "contracts": t["contracts"],
                           "dte": t["dte"], "category": t["category"]})
                results.append(r)

            pnls = [r["pnl"] for r in results]
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100 if pnls else 0
            max_loss = min(pnls) if pnls else 0

            # Daily P&L
            day_pnls = {}
            for r in results:
                day_pnls.setdefault(r["day"], 0)
                day_pnls[r["day"]] += r["pnl"]

            days_won = sum(1 for v in day_pnls.values() if v > 0)
            days_total = len(day_pnls)
            avg_daily = np.mean(list(day_pnls.values())) if day_pnls else 0

            # Without top outlier
            sorted_pnls = sorted(pnls, reverse=True)
            no_outlier = sum(sorted_pnls[1:]) if len(sorted_pnls) > 1 else 0

            # Consistency score: days_won ratio * total (reward consistency)
            consistency = (days_won / days_total if days_total > 0 else 0) * abs(avg_daily)
            if total < 0:
                consistency = -abs(consistency)

            config_scores.append((cfg["_name"], total, wr, max_loss, days_won,
                                  days_total, avg_daily, no_outlier, results, cfg, consistency))

            print(f"  {cfg['_name']:<35} ${total:>+8,.0f} {wr:>5.0f}% ${max_loss:>+8,.0f} "
                  f"{days_won:>2}/{days_total:<3} ${avg_daily:>+7,.0f} ${no_outlier:>+9,.0f}")

        # Rank by: (1) most days won, (2) highest no-outlier total, (3) best consistency
        config_scores.sort(key=lambda x: (x[4], x[7], x[10]), reverse=True)
        best = config_scores[0]
        best_results[dte_label] = best
        print(f"\n  >>> BEST {dte_label}: {best[0]} — ${best[1]:+,.0f}, {best[4]}/{best[5]} days won, "
              f"no-outlier ${best[7]:+,.0f}")

    # ========================================================================
    # Phase 2: Find best ticker-category overlay
    # ========================================================================
    print(f"\n{'='*120}")
    print("PHASE 2: Ticker-category overlays on best base configs")
    print(f"{'='*120}")

    category_configs = {}
    for dte_label, base_info in best_results.items():
        base_cfg = base_info[9]  # the config dict
        dte_filter = (lambda t: t["dte"] == 0) if dte_label == "0DTE" else (lambda t: t["dte"] > 0)

        for cat_name, mods_list in [("high_vol", HIGHVOL_MODS), ("index", INDEX_MODS), ("standard", [{}])]:
            subset = [t for t in trade_data if dte_filter(t) and t["category"] == cat_name]
            if not subset:
                continue

            print(f"\n  {dte_label} + {cat_name} ({len(subset)} trades)")
            best_mod = None
            best_mod_score = -float("inf")

            for i, mod in enumerate(mods_list):
                cfg = {**base_cfg, **mod}
                cfg["_name"] = f"{base_cfg['_name']}+{cat_name}_mod{i}"
                results = []
                for t in subset:
                    r = simulate_trade(t["df"], 0, t["entry"], t["contracts"],
                                       t["direction"], t["dte"], cfg)
                    r.update({"ticker": t["ticker"], "day": t["day"]})
                    results.append(r)

                total = sum(r["pnl"] for r in results)
                day_pnls = {}
                for r in results:
                    day_pnls.setdefault(r["day"], 0)
                    day_pnls[r["day"]] += r["pnl"]
                days_won = sum(1 for v in day_pnls.values() if v > 0)
                days_total = len(day_pnls)

                # Remove top outlier
                pnls = sorted([r["pnl"] for r in results], reverse=True)
                no_outlier = sum(pnls[1:]) if len(pnls) > 1 else 0

                mod_desc = str(mod) if mod else "base"
                score = days_won * 1000 + no_outlier
                print(f"    mod{i}: {mod_desc[:60]:<62} ${total:>+8,.0f} {days_won}/{days_total} days "
                      f"no-outlier=${no_outlier:>+7,.0f}")

                if score > best_mod_score:
                    best_mod_score = score
                    best_mod = cfg

            key = f"{dte_label}_{cat_name}"
            category_configs[key] = best_mod
            print(f"    >>> Best: {best_mod['_name']}")

    # ========================================================================
    # Phase 3: Run combined category-aware strategy
    # ========================================================================
    print(f"\n{'='*120}")
    print("PHASE 3: Combined category-aware strategy vs uniform strategies")
    print(f"{'='*120}")

    # Run category-aware
    cat_results = []
    for t in trade_data:
        dte_label = "0DTE" if t["dte"] == 0 else "Multi-day"
        key = f"{dte_label}_{t['category']}"
        cfg = category_configs.get(key)
        if cfg is None:
            # Fallback to base
            cfg = best_results.get(dte_label, (None,)*10)[9] or make_config("fallback")
        r = simulate_trade(t["df"], 0, t["entry"], t["contracts"],
                           t["direction"], t["dte"], cfg)
        r.update({"ticker": t["ticker"], "day": t["day"], "score": t["score"],
                   "entry": t["entry"], "contracts": t["contracts"],
                   "dte": t["dte"], "category": t["category"],
                   "config": cfg["_name"]})
        cat_results.append(r)

    # Run uniform v5b_base for comparison
    v5b_cfg = make_config("v5b_base")
    v5b_results = []
    for t in trade_data:
        r = simulate_trade(t["df"], 0, t["entry"], t["contracts"],
                           t["direction"], t["dte"], v5b_cfg)
        r.update({"ticker": t["ticker"], "day": t["day"]})
        v5b_results.append(r)

    # Run uniform conservative for comparison
    cons_cfg = make_config("conservative", hard_stop=25, tight_stop=30, backstop=50,
                           theta_minutes=60, theta_loss_pct=5, max_loss_dollars=400)
    cons_results = []
    for t in trade_data:
        r = simulate_trade(t["df"], 0, t["entry"], t["contracts"],
                           t["direction"], t["dte"], cons_cfg)
        r.update({"ticker": t["ticker"], "day": t["day"]})
        cons_results.append(r)

    # Print comparison
    for label, results in [("Category-Aware", cat_results), ("v5b Uniform", v5b_results),
                           ("Conservative Uniform", cons_results)]:
        total = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100 if results else 0
        max_loss = min(r["pnl"] for r in results) if results else 0
        max_win = max(r["pnl"] for r in results) if results else 0

        day_pnls = {}
        for r in results:
            day_pnls.setdefault(r["day"], 0)
            day_pnls[r["day"]] += r["pnl"]
        days_won = sum(1 for v in day_pnls.values() if v > 0)

        pnl_sorted = sorted([r["pnl"] for r in results], reverse=True)
        no_outlier = sum(pnl_sorted[1:]) if len(pnl_sorted) > 1 else 0
        ret_pct = total / PORTFOLIO * 100

        print(f"\n  {label}:")
        print(f"    Total: ${total:>+,.0f} ({ret_pct:>+.1f}% on ${PORTFOLIO:,})")
        print(f"    Trades: {len(results)}, WR: {wr:.0f}%")
        print(f"    Max single win: ${max_win:>+,.0f}, Max single loss: ${max_loss:>+,.0f}")
        print(f"    No-outlier total: ${no_outlier:>+,.0f}")
        print(f"    Days won: {days_won}/{len(day_pnls)}")

    # ========================================================================
    # Daily P&L comparison
    # ========================================================================
    print(f"\n{'='*120}")
    print("DAILY P&L COMPARISON")
    print(f"{'='*120}")
    print(f"{'Day':<12} {'Sigs':>4}  {'CatAware':>10} {'Uniform':>10} {'Conserv':>10}  {'Best':>10}")
    print("-" * 70)

    cum = {"cat": 0, "uni": 0, "con": 0}
    for day in all_days:
        cat_day = sum(r["pnl"] for r in cat_results if r["day"] == day)
        uni_day = sum(r["pnl"] for r in v5b_results if r["day"] == day)
        con_day = sum(r["pnl"] for r in cons_results if r["day"] == day)
        n_sigs = sum(1 for r in cat_results if r["day"] == day)
        cum["cat"] += cat_day
        cum["uni"] += uni_day
        cum["con"] += con_day
        best = max(cat_day, uni_day, con_day)
        best_label = "CatAware" if best == cat_day else ("Uniform" if best == uni_day else "Conserv")
        marker = " <<<" if cat_day > 0 else " XXX"
        print(f"{day:<12} {n_sigs:>4}  ${cat_day:>+9,.0f} ${uni_day:>+9,.0f} ${con_day:>+9,.0f}  "
              f"{best_label:>10}{marker}")

    print("-" * 70)
    print(f"{'CUMULATIVE':<12} {'':>4}  ${cum['cat']:>+9,.0f} ${cum['uni']:>+9,.0f} ${cum['con']:>+9,.0f}")

    # ========================================================================
    # Per-trade detail for category-aware
    # ========================================================================
    print(f"\n{'='*120}")
    print("CATEGORY-AWARE: Per-Trade Detail")
    print(f"{'='*120}")
    print(f"{'Day':<12} {'Ticker':<7} {'DTE':>3} {'Cat':<8} {'Score':>5} {'Entry':>6} {'Ct':>3} "
          f"{'P&L':>9} {'PkGain':>7} {'Reason':<20} {'Config':<35}")
    print("-" * 130)

    for day in all_days:
        day_trades = [r for r in cat_results if r["day"] == day]
        for r in sorted(day_trades, key=lambda x: x["pnl"]):
            cfg_name = r.get("config", "?")
            print(f"{r['day']:<12} {r['ticker']:<7} {r['dte']:>3} {r.get('category','?'):<8} "
                  f"{r['score']:>5} ${r['entry']:>5.2f} {r['contracts']:>3} "
                  f"${r['pnl']:>+8,.0f} {r['peak_gain']:>+6.0f}% {r['reason']:<20} {cfg_name:<35}")
        day_total = sum(r["pnl"] for r in day_trades)
        print(f"{'':>12} {'':>7} {'':>3} {'':>8} {'':>5} {'':>6} {'':>3} "
              f"${day_total:>+8,.0f} {'─── Day Total ───'}")
        print()

    # ========================================================================
    # Gate breakdown for category-aware
    # ========================================================================
    print(f"\n{'='*120}")
    print("GATE FIRE BREAKDOWN — Category-Aware")
    print(f"{'='*120}")
    gate_stats = {}
    for r in cat_results:
        reason = r["reason"]
        if reason not in gate_stats:
            gate_stats[reason] = {"fires": 0, "pnl": 0, "wins": 0, "trades": []}
        gate_stats[reason]["fires"] += 1
        gate_stats[reason]["pnl"] += r["pnl"]
        gate_stats[reason]["trades"].append(r["pnl"])
        if r["pnl"] > 0:
            gate_stats[reason]["wins"] += 1

    print(f"{'Gate':<25} {'Fires':>6} {'%':>5} {'P&L':>10} {'WR':>6} {'AvgP&L':>9} {'MaxLoss':>9}")
    print("-" * 75)
    for gate, stats in sorted(gate_stats.items(), key=lambda x: -x[1]["fires"]):
        wr = stats["wins"] / stats["fires"] * 100 if stats["fires"] > 0 else 0
        pct = stats["fires"] / len(cat_results) * 100
        avg = np.mean(stats["trades"])
        ml = min(stats["trades"])
        print(f"{gate:<25} {stats['fires']:>6} {pct:>4.0f}% ${stats['pnl']:>+9,.0f} {wr:>5.0f}% "
              f"${avg:>+8,.0f} ${ml:>+8,.0f}")

    # ========================================================================
    # Final recommendation
    # ========================================================================
    print(f"\n{'='*120}")
    print("RECOMMENDED CONFIGS PER CATEGORY")
    print(f"{'='*120}")
    for key, cfg in sorted(category_configs.items()):
        print(f"\n  {key}:")
        for k, v in sorted(cfg.items()):
            if k.startswith("_"):
                continue
            print(f"    {k}: {v}")


if __name__ == "__main__":
    run_sweep()
