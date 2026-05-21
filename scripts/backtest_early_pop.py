"""Backtest two strategies for "early pop then crash" trades.

Strategy 1: PRE-FSM MOMENTUM QUALITY (exit-side)
  After grace period ends, if peak was reached in first N minutes AND premium
  velocity is negative, tighten the backstop from 65% to a lower value.
  This catches crashers ~25% earlier.

Strategy 2: ENTRY-SIDE CHASING FILTER (entry-side)
  At entry time, check if the underlying already made a big move before the
  signal arrived. If the move is "done" (underlying moved significantly in the
  signal direction), the trade is likely chasing and gets blocked.

Both strategies are tested with multiple parameter sweeps independently.

Usage:
    python scripts/backtest_early_pop.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import (
    V5Config,
    AdaptiveTier,
    TickerCategory,
    get_ticker_config,
    categorize_ticker,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitAction, ExitReason, _exit, _hold

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
    ENABLE_V6_SIDEWAYS_SCALP=True,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}

SCORE_TIERS = [
    (135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25),
]


# ── Data loading (shared with other backtest scripts) ────────────────────────


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
            signal["_expiry_date"] = expiry
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


def momentum_blocked(df, direction):
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    underlying_prices = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))
    if len(underlying_prices) < 5:
        return False
    first_half = underlying_prices[:len(underlying_prices) // 2]
    second_half = underlying_prices[len(underlying_prices) // 2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    pct_move = (avg_second - avg_first) / avg_first * 100
    prem_start = df["premium"].iloc[0]
    prem_5 = df["premium"].iloc[min(4, len(df) - 1)]
    prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0
    neg_signals = 0
    if is_call and pct_move < -0.05:
        neg_signals += 1
    elif not is_call and pct_move > 0.05:
        neg_signals += 1
    if prem_fade < -5:
        neg_signals += 1
    against = 0
    for i in range(max(0, window - 3), window):
        if i == 0:
            continue
        prev_u = df["underlying_price"].iloc[i - 1]
        cur_u = df["underlying_price"].iloc[i]
        if prev_u and cur_u:
            if is_call and cur_u < prev_u:
                against += 1
            elif not is_call and cur_u > prev_u:
                against += 1
    if against >= 3:
        neg_signals += 1
    return neg_signals >= 2


# ── Simulation ───────────────────────────────────────────────────────────────


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker="SIM", cfg_override=None, settings_override=None):
    """Run actual production FSM with optional config overrides."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "peak_min": 0}

    cfg = cfg_override or get_ticker_config(ticker, use_per_ticker=True)
    settings = settings_override or _V6_SETTINGS
    fsm = ExitFSM(cfg, settings=settings)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
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
    peak_time_min = 0.0

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
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        # Track when peak was reached
        if premium >= state.peak_premium:
            peak_time_min = (now - entry_ts).total_seconds() / 60

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
                    "exit_prem": premium, "peak_gain": peak_gain,
                    "peak_min": peak_time_min}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain,
            "peak_min": peak_time_min}


# ── Strategy 1: Pre-FSM Momentum Quality Gate ───────────────────────────────
#
# Concept: After grace ends, if the trade peaked early (first N minutes) AND
# premium is now fading (velocity negative), tighten the backstop.
# This is implemented by modifying the V5Config for trades that match the
# "early peak" pattern at the time the backstop would fire.
#
# We can't inject a gate *inside* the FSM evaluate loop without modifying
# production code. Instead, we simulate with a MODIFIED config that has a
# tighter backstop, but ONLY for trades where the early-pop pattern is
# detected from tick data.


def detect_early_pop(df, entry_premium, direction, peak_window_min, fade_threshold_pct):
    """Detect if this trade shows the early-pop-then-fade pattern.

    Returns True if:
      1. Premium peaked within first peak_window_min minutes
      2. After peak, premium dropped by at least fade_threshold_pct from peak
      3. Peak was at least +5% above entry (to avoid noise)

    This is computed from the FULL tick history (cheating a bit — in production
    we'd use a rolling window). But for parameter tuning, this shows the max
    potential of the approach.
    """
    if len(df) < 10 or entry_premium <= 0:
        return False

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    # Find peak within the window
    peak_prem = entry_premium
    peak_idx = 0
    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed = (ts - entry_ts).total_seconds() / 60
        if elapsed > peak_window_min:
            break
        prem = df["premium"].iloc[i]
        if not np.isnan(prem) and prem > peak_prem:
            peak_prem = prem
            peak_idx = i

    # Peak must be meaningful (at least +5% above entry)
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    if peak_gain < 5.0:
        return False

    # Check if premium faded significantly after the peak
    # Look at the next 30 ticks after peak (roughly 2.5 min at 5s intervals)
    post_peak = df["premium"].iloc[peak_idx:peak_idx + 60].dropna()
    if len(post_peak) < 5:
        return False

    min_after_peak = post_peak.min()
    fade_from_peak = (peak_prem - min_after_peak) / peak_prem * 100
    return fade_from_peak >= fade_threshold_pct


def detect_early_pop_realtime(df, entry_premium, direction,
                               peak_window_min, fade_threshold_pct,
                               check_at_min):
    """Realistic version: only uses data available up to check_at_min minutes.

    This simulates what the bot would actually know at the time it needs to
    decide whether to tighten the backstop. Only uses ticks up to check_at_min.
    """
    if len(df) < 10 or entry_premium <= 0:
        return False

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    # Only use data up to check_at_min
    peak_prem = entry_premium
    peak_elapsed = 0.0
    current_prem = entry_premium
    current_elapsed = 0.0

    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed = (ts - entry_ts).total_seconds() / 60
        if elapsed > check_at_min:
            break

        prem = df["premium"].iloc[i]
        if np.isnan(prem) or prem <= 0:
            continue

        current_prem = prem
        current_elapsed = elapsed
        if prem > peak_prem:
            peak_prem = prem
            peak_elapsed = elapsed

    # Peak must have been reached within the early window
    if peak_elapsed > peak_window_min:
        return False

    # Peak must be meaningful
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    if peak_gain < 5.0:
        return False

    # Current premium must have faded from peak
    if peak_prem <= 0:
        return False
    fade = (peak_prem - current_prem) / peak_prem * 100
    return fade >= fade_threshold_pct


# ── Strategy 2: Entry-Side Chasing Filter ────────────────────────────────────
#
# Concept: At entry time, check if the underlying already moved significantly
# in the signal direction BEFORE the signal was sent. If so, the move may be
# "done" and we're chasing. Block the trade.
#
# We check the underlying price in the first few ticks of harvester data
# (which starts at signal time) and compare to the strike price or use
# the pre-signal underlying move implied by IV/delta.


def detect_chasing(df, direction, underlying_move_threshold_pct,
                   premium_velocity_threshold_pct, lookback_ticks=10):
    """Detect if this trade is chasing an already-completed move.

    Checks first `lookback_ticks` of data for signs of chasing:
      1. Underlying is already extended in the signal direction
         (e.g., call signal but stock already moved +0.5% in the last few ticks)
      2. Premium is already fading from what was likely a higher level
         (negative premium velocity in first few ticks = signal arrived late)

    Returns (is_chasing, details_dict)
    """
    if len(df) < lookback_ticks:
        return False, {}

    is_call = direction in ("bullish", "call")

    # Check underlying momentum in first N ticks
    underlying_prices = []
    for i in range(min(lookback_ticks, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))

    u_momentum = 0.0
    if len(underlying_prices) >= 3:
        first = np.mean(underlying_prices[:len(underlying_prices)//3])
        last = np.mean(underlying_prices[-len(underlying_prices)//3:])
        if first > 0:
            u_momentum = (last - first) / first * 100

    # Check premium velocity (are we buying into a fading premium?)
    premiums = []
    for i in range(min(lookback_ticks, len(df))):
        p = df["premium"].iloc[i]
        if not np.isnan(p) and p > 0:
            premiums.append(float(p))

    prem_velocity = 0.0
    if len(premiums) >= 3:
        first_prem = np.mean(premiums[:len(premiums)//3])
        last_prem = np.mean(premiums[-len(premiums)//3:])
        if first_prem > 0:
            prem_velocity = (last_prem - first_prem) / first_prem * 100

    details = {
        "u_momentum": round(u_momentum, 3),
        "prem_velocity": round(prem_velocity, 2),
        "u_prices": len(underlying_prices),
        "premiums": len(premiums),
    }

    # Chasing signals:
    chasing_signals = 0

    # Signal 1: Underlying already moved in signal direction
    if is_call and u_momentum < -underlying_move_threshold_pct:
        # Stock already dropping — call is chasing a bounce that may be done
        chasing_signals += 1
    elif not is_call and u_momentum > underlying_move_threshold_pct:
        chasing_signals += 1

    # Signal 2: Premium is fading at entry (bought the top)
    if prem_velocity < -premium_velocity_threshold_pct:
        chasing_signals += 1

    # Signal 3: Underlying moving AGAINST signal direction (stronger)
    if is_call and u_momentum < -(underlying_move_threshold_pct * 2):
        chasing_signals += 1
    elif not is_call and u_momentum > (underlying_move_threshold_pct * 2):
        chasing_signals += 1

    return chasing_signals >= 1, details


# ── Main ─────────────────────────────────────────────────────────────────────


def prepare_trades():
    """Load signals, match with harvester ticks, compute sizing."""
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    prepared = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue

        premium = sig["premium"]

        # Premium cap (tiered: $6/$7/$9)
        if ticker not in INDEX_TICKERS:
            if score >= 150:
                cap = 9.0
            elif score >= 120:
                cap = 7.0
            else:
                cap = 6.0
            if premium > cap:
                continue

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = premium

        # Dollar-target sizing
        deployable = PORTFOLIO * 0.75
        per_slot = deployable / 4
        position_cap = PORTFOLIO * 0.15

        score_mult = 0.25
        for threshold, mult in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                break

        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_contracts, pos_cap_contracts))

        is_blocked = momentum_blocked(df, direction)

        prepared.append({
            "ticker": ticker,
            "direction": direction,
            "score": score,
            "premium": adj_entry,
            "contracts": contracts,
            "df": df,
            "dte": dte,
            "expiry_date": expiry_date,
            "momentum_blocked": is_blocked,
            "day": sig["created_at"][:10],
            "signal_id": sig["id"],
        })

    harvester_conn.close()
    print(f"Prepared {len(prepared)} tradeable signals ({no_data} no tick data)")

    tradeable = [s for s in prepared if not s["momentum_blocked"]]
    print(f"After momentum gate: {len(tradeable)} trades\n")
    return tradeable


def run_strategy1(tradeable):
    """Strategy 1: Momentum Quality — tighter backstop for early-pop trades."""
    print("=" * 120)
    print("STRATEGY 1: PRE-FSM MOMENTUM QUALITY (exit-side)")
    print("Tighten backstop for trades that peak early then fade")
    print("=" * 120)

    # Parameter sweep
    params = [
        # (peak_window_min, fade_threshold_pct, check_at_min, tight_backstop_0dte, tight_backstop_multi, label)
        (10, 15, 8,  40, 55, "early10/fade15/bs40"),
        (10, 15, 8,  45, 60, "early10/fade15/bs45"),
        (10, 15, 8,  50, 65, "early10/fade15/bs50"),
        (10, 20, 8,  40, 55, "early10/fade20/bs40"),
        (10, 20, 8,  45, 60, "early10/fade20/bs45"),
        (15, 15, 10, 40, 55, "early15/fade15/bs40"),
        (15, 15, 10, 45, 60, "early15/fade15/bs45"),
        (15, 20, 10, 40, 55, "early15/fade20/bs40"),
        (15, 20, 10, 45, 60, "early15/fade20/bs45"),
        (8,  10, 7,  40, 55, "early8/fade10/bs40"),
        (8,  10, 7,  45, 60, "early8/fade10/bs45"),
        (8,  15, 7,  40, 55, "early8/fade15/bs40"),
        (8,  15, 7,  50, 65, "early8/fade15/bs50"),
        # Also test with tighter confirmed stop (underlying-against)
        (10, 15, 8,  40, 55, "early10/fade15/bs40+tight25"),
        (15, 15, 10, 40, 55, "early15/fade15/bs40+tight25"),
    ]

    # Baseline first
    print("\nRunning baseline...")
    baseline_results = []
    for sig in tradeable:
        result = simulate_trade(
            sig["df"], sig["premium"], sig["contracts"],
            sig["direction"], sig["dte"], sig["expiry_date"],
            ticker=sig["ticker"],
        )
        result["ticker"] = sig["ticker"]
        result["day"] = sig["day"]
        result["contracts"] = sig["contracts"]
        result["entry"] = sig["premium"]
        result["dte"] = sig["dte"]
        baseline_results.append(result)

    baseline_pnl = sum(r["pnl"] for r in baseline_results)
    baseline_wins = sum(1 for r in baseline_results if r["pnl"] > 0)
    baseline_wr = baseline_wins / len(baseline_results) * 100
    print(f"Baseline: {len(baseline_results)} trades, ${baseline_pnl:,.2f}, "
          f"{baseline_wr:.1f}% WR\n")

    # Run each param set
    scenario_results = {}
    for pw, ft, cam, bs0, bsm, label in params:
        use_tight25 = "tight25" in label

        results = []
        detected = 0
        for i, sig in enumerate(tradeable):
            ticker = sig["ticker"]
            df = sig["df"]
            direction = sig["direction"]
            entry = sig["premium"]

            # Detect early pop using realistic (causal) detection
            is_early_pop = detect_early_pop_realtime(
                df, entry, direction,
                peak_window_min=pw,
                fade_threshold_pct=ft,
                check_at_min=cam,
            )

            if is_early_pop:
                detected += 1
                # Tighter config for this trade
                base_cfg = get_ticker_config(ticker, use_per_ticker=True)
                overrides = {
                    "backstop_0dte_pct": bs0,
                    "backstop_multiday_pct": bsm,
                }
                if use_tight25:
                    overrides["tight_stop_0dte_pct"] = 25.0
                    overrides["tight_stop_multiday_pct"] = 40.0
                cfg = replace(base_cfg, **overrides)

                result = simulate_trade(
                    df, entry, sig["contracts"], direction,
                    sig["dte"], sig["expiry_date"],
                    ticker=ticker, cfg_override=cfg,
                )
            else:
                # Use baseline result (same as production FSM)
                result = baseline_results[i].copy()

            result["ticker"] = ticker
            result["day"] = sig["day"]
            result["contracts"] = sig["contracts"]
            result["entry"] = entry
            result["dte"] = sig["dte"]
            result["early_pop"] = is_early_pop
            results.append(result)

        scenario_results[label] = {
            "results": results,
            "detected": detected,
        }

    # Summary table
    print(f"\n{'Scenario':<30} {'Det':>4} {'Trades':>6} {'P&L':>12} {'vs Base':>10} "
          f"{'Win%':>6} {'AvgWin':>9} {'AvgLoss':>9}")
    print("-" * 100)

    print(f"{'BASELINE':<30} {'':>4} {len(baseline_results):>6} "
          f"${baseline_pnl:>10,.2f} {'$0':>10} {baseline_wr:>5.1f}%")

    best_scenario = None
    best_delta = -999999

    for label in [p[-1] for p in params]:
        s = scenario_results[label]
        results = s["results"]
        pnl = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100
        delta = pnl - baseline_pnl
        pnls = [r["pnl"] for r in results]
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0

        marker = " ***" if delta > 200 else (" *" if delta > 50 else "")
        print(f"{label:<30} {s['detected']:>4} {len(results):>6} "
              f"${pnl:>10,.2f} ${delta:>+8,.2f} {wr:>5.1f}% "
              f"${avg_win:>7,.2f} ${avg_loss:>7,.2f}{marker}")

        if delta > best_delta:
            best_delta = delta
            best_scenario = label

    # Show trade-level diff for best scenario
    if best_scenario and best_delta > 0:
        print(f"\n--- Best scenario: {best_scenario} (${best_delta:+,.2f}) ---")
        print(f"\nTrade-level differences (early-pop detected trades only):")
        print(f"{'Day':<12} {'Ticker':<7} {'DTE':>3} {'Entry':>6} {'Ct':>3} "
              f"{'Base P&L':>10} {'New P&L':>10} {'Delta':>9} {'BaseRsn':<22} {'NewRsn':<22}")
        print("-" * 120)

        best = scenario_results[best_scenario]["results"]
        diffs = []
        for br, nr in zip(baseline_results, best):
            if nr.get("early_pop"):
                diff = nr["pnl"] - br["pnl"]
                diffs.append((br, nr, diff))

        diffs.sort(key=lambda x: x[2], reverse=True)
        for br, nr, diff in diffs:
            print(f"{br['day']:<12} {br['ticker']:<7} {br['dte']:>3} "
                  f"${br['entry']:>5.2f} {br['contracts']:>3} "
                  f"${br['pnl']:>8,.2f} ${nr['pnl']:>8,.2f} ${diff:>+7,.2f} "
                  f"{br['reason']:<22} {nr['reason']:<22}")

        saved = sum(d for _, _, d in diffs if d > 0)
        cost = sum(-d for _, _, d in diffs if d < 0)
        print(f"\nSaved: ${saved:,.2f} | Cost: ${cost:,.2f} | Net: ${saved-cost:+,.2f}")

    return baseline_results, baseline_pnl


def run_strategy2(tradeable, baseline_results, baseline_pnl):
    """Strategy 2: Entry-side chasing filter — block trades chasing completed moves."""
    print("\n\n" + "=" * 120)
    print("STRATEGY 2: ENTRY-SIDE CHASING FILTER")
    print("Block trades where underlying already moved significantly before signal")
    print("=" * 120)

    # Parameter sweep
    params = [
        # (underlying_move_threshold_pct, premium_velocity_threshold_pct, lookback_ticks, label)
        (0.05, 3.0, 10, "u0.05/pv3/L10"),
        (0.05, 5.0, 10, "u0.05/pv5/L10"),
        (0.05, 3.0, 15, "u0.05/pv3/L15"),
        (0.10, 3.0, 10, "u0.10/pv3/L10"),
        (0.10, 5.0, 10, "u0.10/pv5/L10"),
        (0.10, 3.0, 15, "u0.10/pv3/L15"),
        (0.10, 5.0, 15, "u0.10/pv5/L15"),
        (0.15, 3.0, 10, "u0.15/pv3/L10"),
        (0.15, 5.0, 10, "u0.15/pv5/L10"),
        (0.15, 3.0, 15, "u0.15/pv3/L15"),
        (0.20, 3.0, 10, "u0.20/pv3/L10"),
        (0.20, 5.0, 10, "u0.20/pv5/L10"),
        (0.03, 2.0, 8,  "u0.03/pv2/L8"),
        (0.03, 3.0, 8,  "u0.03/pv3/L8"),
    ]

    scenario_results = {}

    for umt, pvt, lt, label in params:
        blocked = 0
        blocked_pnl = 0
        allowed_pnl = 0
        blocked_trades = []

        for i, sig in enumerate(tradeable):
            is_chasing, details = detect_chasing(
                sig["df"], sig["direction"],
                underlying_move_threshold_pct=umt,
                premium_velocity_threshold_pct=pvt,
                lookback_ticks=lt,
            )

            base_pnl = baseline_results[i]["pnl"]

            if is_chasing:
                blocked += 1
                blocked_pnl += base_pnl
                blocked_trades.append({
                    "ticker": sig["ticker"],
                    "day": sig["day"],
                    "pnl": base_pnl,
                    "reason": baseline_results[i]["reason"],
                    **details,
                })
            else:
                allowed_pnl += base_pnl

        scenario_results[label] = {
            "blocked": blocked,
            "allowed_pnl": allowed_pnl,
            "blocked_pnl": blocked_pnl,
            "blocked_trades": blocked_trades,
        }

    # Summary table
    print(f"\n{'Scenario':<22} {'Blocked':>7} {'Allowed P&L':>14} {'Blocked P&L':>14} "
          f"{'Net P&L':>12} {'vs Base':>10} {'Block WR':>9}")
    print("-" * 100)

    print(f"{'BASELINE (no filter)':<22} {0:>7} ${baseline_pnl:>12,.2f} "
          f"{'$0':>14} ${baseline_pnl:>10,.2f} {'$0':>10}")

    best_scenario = None
    best_delta = -999999

    for label in [p[-1] for p in params]:
        s = scenario_results[label]
        net = s["allowed_pnl"]
        delta = net - baseline_pnl  # = -blocked_pnl (we lose blocked trades)
        # Positive delta means we blocked losing trades (good)
        # Negative delta means we blocked winning trades (bad)
        blocked_losses = sum(1 for t in s["blocked_trades"] if t["pnl"] <= 0)
        blocked_wins = sum(1 for t in s["blocked_trades"] if t["pnl"] > 0)
        block_wr = (blocked_losses / s["blocked"] * 100) if s["blocked"] > 0 else 0

        marker = " ***" if delta > 200 else (" *" if delta > 50 else "")
        print(f"{label:<22} {s['blocked']:>7} ${s['allowed_pnl']:>12,.2f} "
              f"${s['blocked_pnl']:>12,.2f} ${net:>10,.2f} ${delta:>+8,.2f} "
              f"{block_wr:>7.0f}%{marker}")

        if delta > best_delta:
            best_delta = delta
            best_scenario = label

    # Show what the best filter blocked
    if best_scenario:
        s = scenario_results[best_scenario]
        print(f"\n--- Best filter: {best_scenario} (${best_delta:+,.2f}) ---")
        print(f"\nBlocked {s['blocked']} trades (blocked P&L: ${s['blocked_pnl']:,.2f}):")
        print(f"{'Day':<12} {'Ticker':<7} {'P&L':>10} {'Reason':<22} "
              f"{'U_Mom':>7} {'Prem_Vel':>9}")
        print("-" * 80)

        blocked = sorted(s["blocked_trades"], key=lambda t: t["pnl"])
        for t in blocked:
            marker = " SAVED" if t["pnl"] < -50 else (" LOST" if t["pnl"] > 50 else "")
            print(f"{t['day']:<12} {t['ticker']:<7} ${t['pnl']:>8,.2f} "
                  f"{t['reason']:<22} {t['u_momentum']:>6.3f}% {t['prem_velocity']:>8.2f}%"
                  f"{marker}")

        blocked_losses = sum(t["pnl"] for t in blocked if t["pnl"] <= 0)
        blocked_wins = sum(t["pnl"] for t in blocked if t["pnl"] > 0)
        print(f"\nBlocked losers: ${blocked_losses:,.2f} | "
              f"Blocked winners: ${blocked_wins:,.2f} | "
              f"Net: ${-s['blocked_pnl']:+,.2f}")

    return scenario_results


def run_combined(tradeable, baseline_results, baseline_pnl,
                 s1_params, s2_params):
    """Test best of Strategy 1 + best of Strategy 2 combined."""
    print("\n\n" + "=" * 120)
    print("COMBINED: BEST STRATEGY 1 + BEST STRATEGY 2")
    print(f"  S1 (exit): {s1_params}")
    print(f"  S2 (entry): {s2_params}")
    print("=" * 120)

    pw, ft, cam, bs0, bsm = s1_params
    umt, pvt, lt = s2_params

    results = []
    s1_triggered = 0
    s2_blocked = 0
    both = 0

    for i, sig in enumerate(tradeable):
        ticker = sig["ticker"]
        df = sig["df"]
        direction = sig["direction"]
        entry = sig["premium"]

        # Strategy 2: entry filter
        is_chasing, _ = detect_chasing(
            df, direction,
            underlying_move_threshold_pct=umt,
            premium_velocity_threshold_pct=pvt,
            lookback_ticks=lt,
        )

        if is_chasing:
            s2_blocked += 1
            # Trade is blocked — contributes $0
            results.append({
                "pnl": 0, "reason": "entry_blocked", "ticker": ticker,
                "day": sig["day"], "blocked": True, "early_pop": False,
                "base_pnl": baseline_results[i]["pnl"],
            })
            continue

        # Strategy 1: tighter backstop for early-pop
        is_early_pop = detect_early_pop_realtime(
            df, entry, direction,
            peak_window_min=pw,
            fade_threshold_pct=ft,
            check_at_min=cam,
        )

        if is_early_pop:
            s1_triggered += 1
            base_cfg = get_ticker_config(ticker, use_per_ticker=True)
            cfg = replace(base_cfg,
                          backstop_0dte_pct=bs0,
                          backstop_multiday_pct=bsm)
            result = simulate_trade(
                df, entry, sig["contracts"], direction,
                sig["dte"], sig["expiry_date"],
                ticker=ticker, cfg_override=cfg,
            )
        else:
            result = baseline_results[i].copy()

        result["ticker"] = ticker
        result["day"] = sig["day"]
        result["blocked"] = False
        result["early_pop"] = is_early_pop
        result["base_pnl"] = baseline_results[i]["pnl"]
        results.append(result)

    total_pnl = sum(r["pnl"] for r in results)
    delta = total_pnl - baseline_pnl
    trades_taken = sum(1 for r in results if not r.get("blocked"))
    wins = sum(1 for r in results if not r.get("blocked") and r["pnl"] > 0)
    wr = wins / trades_taken * 100 if trades_taken > 0 else 0

    print(f"\nBaseline: {len(tradeable)} trades, ${baseline_pnl:,.2f}")
    print(f"Combined: {trades_taken} trades taken ({s2_blocked} blocked), "
          f"${total_pnl:,.2f} ({delta:+,.2f})")
    print(f"Win rate: {wr:.1f}%")
    print(f"S1 triggered on {s1_triggered} trades, S2 blocked {s2_blocked} trades")

    # Show all changes
    print(f"\nAll affected trades:")
    print(f"{'Day':<12} {'Ticker':<7} {'Action':<12} {'Base P&L':>10} {'New P&L':>10} {'Delta':>9}")
    print("-" * 70)

    affected = [(r, r["pnl"] - r["base_pnl"]) for r in results
                if r.get("blocked") or r.get("early_pop")]
    affected.sort(key=lambda x: x[1], reverse=True)

    for r, diff in affected:
        action = "BLOCKED" if r.get("blocked") else "TIGHTER_BS"
        print(f"{r['day']:<12} {r['ticker']:<7} {action:<12} "
              f"${r['base_pnl']:>8,.2f} ${r['pnl']:>8,.2f} ${diff:>+7,.2f}")


def main():
    tradeable = prepare_trades()

    # Run Strategy 1 (exit-side)
    baseline_results, baseline_pnl = run_strategy1(tradeable)

    # Run Strategy 2 (entry-side)
    s2_results = run_strategy2(tradeable, baseline_results, baseline_pnl)

    # Find best params for each strategy
    # Best S1: pick the scenario with highest delta
    # (We'll hardcode a reasonable one for the combined test based on output)

    # Combined test with reasonable params from each
    # These are "middle of the road" params — adjust after seeing individual results
    run_combined(
        tradeable, baseline_results, baseline_pnl,
        s1_params=(10, 15, 8, 45, 60),   # early10/fade15/bs45
        s2_params=(0.10, 5.0, 10),        # u0.10/pv5/L10
    )

    # Also test with tightest S1
    run_combined(
        tradeable, baseline_results, baseline_pnl,
        s1_params=(10, 15, 8, 40, 55),   # early10/fade15/bs40
        s2_params=(0.10, 3.0, 10),        # u0.10/pv3/L10
    )

    print("\n\nDONE")


if __name__ == "__main__":
    main()
