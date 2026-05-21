"""Deep analysis of Strategy 1 (early-pop momentum quality gate).

Questions to answer:
1. Do any of the 25 detected trades become runners? (conflict check)
2. Can we take small profits at peak instead of just cutting losses?
3. What features distinguish crashers from recoverers? (ML training data)

Usage:
    python scripts/backtest_early_pop_deep.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

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
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals WHERE score >= 70 ORDER BY created_at
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


def detect_early_pop_realtime(df, entry_premium, direction, peak_window_min=10,
                               fade_threshold_pct=15, check_at_min=8):
    if len(df) < 10 or entry_premium <= 0:
        return False
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    peak_prem = entry_premium
    peak_elapsed = 0.0
    current_prem = entry_premium
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
        if prem > peak_prem:
            peak_prem = prem
            peak_elapsed = elapsed
    if peak_elapsed > peak_window_min:
        return False
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    if peak_gain < 5.0:
        return False
    if peak_prem <= 0:
        return False
    fade = (peak_prem - current_prem) / peak_prem * 100
    return fade >= fade_threshold_pct


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker="SIM", cfg_override=None):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "peak_min": 0, "max_gain_pct": 0}
    cfg = cfg_override or get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
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
    max_gain = 0.0
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
        gain_pct = (premium - entry_premium) / entry_premium * 100
        if gain_pct > max_gain:
            max_gain = gain_pct
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
                    "peak_min": peak_time_min, "max_gain_pct": max_gain}
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
            "peak_min": peak_time_min, "max_gain_pct": max_gain}


def extract_features(df, entry_premium, direction, check_at_min=8):
    """Extract ML-trainable features from first N minutes of tick data."""
    if len(df) < 5 or entry_premium <= 0:
        return None

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    is_call = direction in ("bullish", "call")

    # Collect data within the check window
    premiums = []
    underlyings = []
    timestamps = []
    ivs = []
    volumes = []

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

        premiums.append(float(prem))
        timestamps.append(elapsed)

        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlyings.append(float(u))

        iv = df["iv"].iloc[i]
        if iv and not pd.isna(iv) and iv > 0:
            ivs.append(float(iv))

        vol = df["volume"].iloc[i]
        if vol and not pd.isna(vol):
            volumes.append(float(vol))

    if len(premiums) < 3:
        return None

    # Premium features
    peak_prem = max(premiums)
    peak_idx = premiums.index(peak_prem)
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    current_prem = premiums[-1]
    current_gain = (current_prem - entry_premium) / entry_premium * 100
    fade_from_peak = (peak_prem - current_prem) / peak_prem * 100 if peak_prem > 0 else 0
    peak_time = timestamps[peak_idx] if peak_idx < len(timestamps) else 0

    # Premium velocity (last 1/3 vs first 1/3)
    n = len(premiums)
    third = max(1, n // 3)
    prem_velocity = 0.0
    if premiums[0] > 0:
        prem_velocity = (np.mean(premiums[-third:]) - np.mean(premiums[:third])) / premiums[0] * 100

    # Premium volatility (std/mean)
    prem_std = np.std(premiums) / np.mean(premiums) * 100 if np.mean(premiums) > 0 else 0

    # How quickly did it peak? (peak position as fraction of window)
    peak_position = peak_idx / max(1, n - 1)  # 0 = peaked immediately, 1 = peaked at end

    # Underlying features
    u_move = 0.0
    u_velocity = 0.0
    if len(underlyings) >= 3:
        u_third = max(1, len(underlyings) // 3)
        u_first = np.mean(underlyings[:u_third])
        u_last = np.mean(underlyings[-u_third:])
        if u_first > 0:
            u_move = (u_last - u_first) / u_first * 100
            # Direction-adjusted: positive = confirming, negative = against
            u_velocity = u_move if is_call else -u_move

    # IV features
    iv_change = 0.0
    if len(ivs) >= 2:
        iv_change = (ivs[-1] - ivs[0]) / ivs[0] * 100 if ivs[0] > 0 else 0

    # Volume
    avg_volume = np.mean(volumes) if volumes else 0
    vol_trend = 0.0
    if len(volumes) >= 3:
        v_third = max(1, len(volumes) // 3)
        v_first = np.mean(volumes[:v_third])
        v_last = np.mean(volumes[-v_third:])
        if v_first > 0:
            vol_trend = (v_last - v_first) / v_first * 100

    # Time of day (ET)
    entry_hour_et = entry_ts.hour - 4
    if entry_hour_et < 0:
        entry_hour_et += 24

    # Number of premium direction changes (choppiness)
    direction_changes = 0
    for j in range(1, len(premiums)):
        if j >= 2:
            prev_dir = premiums[j-1] - premiums[j-2]
            curr_dir = premiums[j] - premiums[j-1]
            if prev_dir * curr_dir < 0:
                direction_changes += 1

    return {
        "peak_gain": round(peak_gain, 2),
        "current_gain": round(current_gain, 2),
        "fade_from_peak": round(fade_from_peak, 2),
        "peak_time_min": round(peak_time, 2),
        "peak_position": round(peak_position, 3),
        "prem_velocity": round(prem_velocity, 2),
        "prem_std": round(prem_std, 2),
        "u_move": round(u_move, 4),
        "u_velocity": round(u_velocity, 4),
        "iv_change": round(iv_change, 2),
        "avg_volume": round(avg_volume, 0),
        "vol_trend": round(vol_trend, 2),
        "entry_hour_et": entry_hour_et,
        "direction_changes": direction_changes,
        "n_ticks": n,
        "entry_premium": entry_premium,
        "is_call": is_call,
    }


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Prepare all trades (same as main backtest)
    prepared = []
    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue
        premium = sig["premium"]
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
            continue
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = premium
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
        if is_blocked:
            continue
        prepared.append({
            "ticker": ticker, "direction": direction, "score": score,
            "premium": adj_entry, "contracts": contracts, "df": df,
            "dte": dte, "expiry_date": expiry_date, "day": sig["created_at"][:10],
        })

    harvester_conn.close()
    print(f"Prepared {len(prepared)} trades\n")

    # ── Part 1: Deep analysis of all 25 detected early-pop trades ────────────

    print("=" * 130)
    print("PART 1: FULL PROFILE OF ALL DETECTED EARLY-POP TRADES")
    print("Do any become runners? What's the max gain? When does the peak happen?")
    print("=" * 130)

    detected_trades = []
    all_results = []

    for sig in prepared:
        is_ep = detect_early_pop_realtime(
            sig["df"], sig["premium"], sig["direction"],
            peak_window_min=10, fade_threshold_pct=15, check_at_min=8,
        )
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
        result["early_pop"] = is_ep
        all_results.append(result)

        if is_ep:
            detected_trades.append(result)

    print(f"\nDetected {len(detected_trades)} early-pop trades out of {len(prepared)}")

    print(f"\n{'Day':<12} {'Ticker':<7} {'DTE':>3} {'Entry':>6} {'Ct':>3} "
          f"{'P&L':>10} {'MaxGain%':>9} {'PeakMin':>8} {'Hold':>6} {'Reason':<22} {'Runner?':<8}")
    print("-" * 130)

    runners_killed = 0
    for r in sorted(detected_trades, key=lambda x: x["pnl"], reverse=True):
        is_runner = r["max_gain_pct"] >= 40
        runner_tag = "RUNNER" if is_runner else ""
        if is_runner:
            runners_killed += 1
        print(f"{r['day']:<12} {r['ticker']:<7} {r['dte']:>3} "
              f"${r['entry']:>5.2f} {r['contracts']:>3} "
              f"${r['pnl']:>8,.2f} {r['max_gain_pct']:>8.1f}% {r['peak_min']:>7.1f}m "
              f"{r['hold']:>5.0f}m {r['reason']:<22} {runner_tag}")

    print(f"\nRunners (peak gain >= 40%) in detected set: {runners_killed}")
    print(f"Winners (P&L > 0): {sum(1 for r in detected_trades if r['pnl'] > 0)}")
    print(f"Losers (P&L <= 0): {sum(1 for r in detected_trades if r['pnl'] <= 0)}")
    total_winner_pnl = sum(r["pnl"] for r in detected_trades if r["pnl"] > 0)
    total_loser_pnl = sum(r["pnl"] for r in detected_trades if r["pnl"] <= 0)
    print(f"Winner P&L: ${total_winner_pnl:,.2f} | Loser P&L: ${total_loser_pnl:,.2f}")

    # ── Part 2: Can we take profits at peak? ─────────────────────────────────

    print(f"\n\n{'=' * 130}")
    print("PART 2: PROFIT-TAKING AT EARLY PEAK")
    print("If we sold at the early peak (first 10min peak), what would P&L be?")
    print("Compare: (A) baseline FSM, (B) tighter backstop (bs40), (C) sell at peak")
    print("=" * 130)

    print(f"\n{'Day':<12} {'Ticker':<7} {'Entry':>6} {'Ct':>3} "
          f"{'Base P&L':>10} {'BS40 P&L':>10} {'Peak P&L':>10} "
          f"{'PeakGain':>9} {'PeakMin':>8}")
    print("-" * 120)

    total_base = 0
    total_bs40 = 0
    total_peak = 0

    for sig in prepared:
        is_ep = detect_early_pop_realtime(
            sig["df"], sig["premium"], sig["direction"],
            peak_window_min=10, fade_threshold_pct=15, check_at_min=8,
        )
        if not is_ep:
            continue

        ticker = sig["ticker"]
        df = sig["df"]
        entry = sig["premium"]
        contracts = sig["contracts"]

        # A) Baseline
        base = simulate_trade(df, entry, contracts, sig["direction"],
                              sig["dte"], sig["expiry_date"], ticker=ticker)

        # B) Tighter backstop
        cfg_bs40 = replace(get_ticker_config(ticker, use_per_ticker=True),
                           backstop_0dte_pct=40.0, backstop_multiday_pct=55.0)
        bs40 = simulate_trade(df, entry, contracts, sig["direction"],
                              sig["dte"], sig["expiry_date"],
                              ticker=ticker, cfg_override=cfg_bs40)

        # C) Sell at the early peak premium
        entry_ts = df["ts"].iloc[0]
        if hasattr(entry_ts, "to_pydatetime"):
            entry_ts = entry_ts.to_pydatetime()
        if entry_ts.tzinfo is not None:
            entry_ts = entry_ts.replace(tzinfo=None)

        peak_prem = entry
        for i in range(len(df)):
            ts = df["ts"].iloc[i]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            elapsed = (ts - entry_ts).total_seconds() / 60
            if elapsed > 10:
                break
            p = df["premium"].iloc[i]
            if not np.isnan(p) and p > peak_prem:
                peak_prem = p

        peak_pnl = (peak_prem - entry) * contracts * 100
        peak_gain = (peak_prem - entry) / entry * 100

        total_base += base["pnl"]
        total_bs40 += bs40["pnl"]
        total_peak += peak_pnl

        print(f"{sig['day']:<12} {ticker:<7} ${entry:>5.2f} {contracts:>3} "
              f"${base['pnl']:>8,.2f} ${bs40['pnl']:>8,.2f} ${peak_pnl:>8,.2f} "
              f"{peak_gain:>8.1f}% {base['peak_min']:>7.1f}m")

    print(f"\n{'TOTALS':<12} {'':>7} {'':>6} {'':>3} "
          f"${total_base:>8,.2f} ${total_bs40:>8,.2f} ${total_peak:>8,.2f}")
    print(f"\n  Baseline:       ${total_base:>10,.2f}")
    print(f"  Tighter BS(40): ${total_bs40:>10,.2f} ({total_bs40-total_base:+,.2f})")
    print(f"  Sell at peak:   ${total_peak:>10,.2f} ({total_peak-total_base:+,.2f})")

    # ── Part 3: ML Feature extraction ────────────────────────────────────────

    print(f"\n\n{'=' * 130}")
    print("PART 3: ML FEATURES — CRASHERS vs RECOVERERS in early-pop set")
    print("What distinguishes the 3 losers from the 22 winners?")
    print("=" * 130)

    crashers = []
    recoverers = []

    for sig in prepared:
        is_ep = detect_early_pop_realtime(
            sig["df"], sig["premium"], sig["direction"],
            peak_window_min=10, fade_threshold_pct=15, check_at_min=8,
        )
        if not is_ep:
            continue

        result = simulate_trade(
            sig["df"], sig["premium"], sig["contracts"],
            sig["direction"], sig["dte"], sig["expiry_date"],
            ticker=sig["ticker"],
        )

        features = extract_features(sig["df"], sig["premium"], sig["direction"])
        if features is None:
            continue

        features["ticker"] = sig["ticker"]
        features["day"] = sig["day"]
        features["final_pnl"] = result["pnl"]
        features["final_reason"] = result["reason"]
        features["max_gain"] = result["max_gain_pct"]

        if result["pnl"] <= 0:
            crashers.append(features)
        else:
            recoverers.append(features)

    print(f"\nCrashers: {len(crashers)} | Recoverers: {len(recoverers)}")

    # Compare feature distributions
    numeric_features = [
        "peak_gain", "current_gain", "fade_from_peak", "peak_time_min",
        "peak_position", "prem_velocity", "prem_std", "u_move", "u_velocity",
        "iv_change", "avg_volume", "vol_trend", "entry_hour_et",
        "direction_changes", "entry_premium",
    ]

    print(f"\n{'Feature':<22} {'Crashers':>12} {'Recoverers':>12} {'Diff':>10} {'Useful?':<8}")
    print("-" * 70)

    useful_features = []
    for feat in numeric_features:
        c_vals = [f[feat] for f in crashers if feat in f]
        r_vals = [f[feat] for f in recoverers if feat in f]
        if not c_vals or not r_vals:
            continue
        c_mean = np.mean(c_vals)
        r_mean = np.mean(r_vals)
        diff = c_mean - r_mean

        # Is the difference significant relative to the spread?
        combined_std = np.std(c_vals + r_vals) if len(c_vals + r_vals) > 1 else 1
        signal_ratio = abs(diff) / combined_std if combined_std > 0 else 0

        useful = "YES" if signal_ratio > 0.5 else ""
        if useful:
            useful_features.append(feat)

        print(f"{feat:<22} {c_mean:>12.3f} {r_mean:>12.3f} {diff:>+10.3f} {useful}")

    # ── Part 4: Broader ML analysis — ALL trades ─────────────────────────────

    print(f"\n\n{'=' * 130}")
    print("PART 4: ML FEATURES — ALL TRADES (not just early-pop)")
    print("Can early-tick features at minute 8 predict final outcome?")
    print("=" * 130)

    all_features = []
    for sig in prepared:
        features = extract_features(sig["df"], sig["premium"], sig["direction"])
        if features is None:
            continue

        result = simulate_trade(
            sig["df"], sig["premium"], sig["contracts"],
            sig["direction"], sig["dte"], sig["expiry_date"],
            ticker=sig["ticker"],
        )

        features["ticker"] = sig["ticker"]
        features["day"] = sig["day"]
        features["final_pnl"] = result["pnl"]
        features["is_loser"] = 1 if result["pnl"] <= 0 else 0
        features["is_big_loser"] = 1 if result["pnl"] < -200 else 0
        features["early_pop"] = 1 if detect_early_pop_realtime(
            sig["df"], sig["premium"], sig["direction"]) else 0
        all_features.append(features)

    df_feat = pd.DataFrame(all_features)
    print(f"\nTotal trades with features: {len(df_feat)}")
    print(f"Losers: {df_feat['is_loser'].sum()} ({df_feat['is_loser'].mean()*100:.1f}%)")
    print(f"Big losers (>$200): {df_feat['is_big_loser'].sum()}")
    print(f"Early-pop detected: {df_feat['early_pop'].sum()}")

    # Correlation with losing
    print(f"\n{'Feature':<22} {'Corr w/ Loser':>14} {'Corr w/ BigLoser':>16}")
    print("-" * 55)

    for feat in numeric_features:
        if feat not in df_feat.columns:
            continue
        corr_loser = df_feat[feat].corr(df_feat["is_loser"])
        corr_big = df_feat[feat].corr(df_feat["is_big_loser"])
        marker = " ***" if abs(corr_big) > 0.15 else (" *" if abs(corr_loser) > 0.15 else "")
        print(f"{feat:<22} {corr_loser:>+14.3f} {corr_big:>+16.3f}{marker}")

    # Early pop as a predictor
    ep_trades = df_feat[df_feat["early_pop"] == 1]
    non_ep = df_feat[df_feat["early_pop"] == 0]
    print(f"\n  Early-pop trades: {len(ep_trades)} — "
          f"loser rate {ep_trades['is_loser'].mean()*100:.1f}%, "
          f"avg P&L ${ep_trades['final_pnl'].mean():,.2f}")
    print(f"  Non-early-pop:   {len(non_ep)} — "
          f"loser rate {non_ep['is_loser'].mean()*100:.1f}%, "
          f"avg P&L ${non_ep['final_pnl'].mean():,.2f}")

    # Try a simple decision tree to see if features can predict big losers
    print(f"\n\n{'=' * 130}")
    print("PART 5: SIMPLE RULES — Can we find a rule that catches big losers?")
    print("=" * 130)

    # Test combinations of features for identifying big losers
    rules = [
        ("fade>15 AND prem_vel<-5", lambda r: r["fade_from_peak"] > 15 and r["prem_velocity"] < -5),
        ("fade>15 AND u_vel<-0.05", lambda r: r["fade_from_peak"] > 15 and r["u_velocity"] < -0.05),
        ("fade>20 AND prem_vel<-8", lambda r: r["fade_from_peak"] > 20 and r["prem_velocity"] < -8),
        ("peak_pos<0.3 AND fade>10", lambda r: r["peak_position"] < 0.3 and r["fade_from_peak"] > 10),
        ("peak_pos<0.3 AND prem_vel<-5", lambda r: r["peak_position"] < 0.3 and r["prem_velocity"] < -5),
        ("peak_pos<0.5 AND fade>15 AND prem_vel<-5",
         lambda r: r["peak_position"] < 0.5 and r["fade_from_peak"] > 15 and r["prem_velocity"] < -5),
        ("early_pop AND fade>20", lambda r: r["early_pop"] and r["fade_from_peak"] > 20),
        ("early_pop AND u_vel<-0.1", lambda r: r["early_pop"] and r["u_velocity"] < -0.1),
        ("iv_change>5 AND fade>10", lambda r: r.get("iv_change", 0) > 5 and r["fade_from_peak"] > 10),
        ("dir_changes>3 AND fade>10", lambda r: r["direction_changes"] > 3 and r["fade_from_peak"] > 10),
    ]

    print(f"\n{'Rule':<48} {'Match':>5} {'BigLos':>6} {'Win':>4} "
          f"{'Precision':>9} {'P&L Saved':>10} {'P&L Lost':>10} {'Net':>10}")
    print("-" * 120)

    for rule_name, rule_fn in rules:
        matched = [f for f in all_features if rule_fn(f)]
        if not matched:
            print(f"{rule_name:<48} {0:>5}")
            continue
        big_losers_caught = sum(1 for f in matched if f["is_big_loser"])
        winners_caught = sum(1 for f in matched if f["final_pnl"] > 0)
        precision = big_losers_caught / len(matched) * 100
        pnl_saved = sum(-f["final_pnl"] for f in matched if f["final_pnl"] < 0)
        pnl_lost = sum(f["final_pnl"] for f in matched if f["final_pnl"] > 0)
        net = pnl_saved - pnl_lost

        marker = " ***" if net > 200 else ""
        print(f"{rule_name:<48} {len(matched):>5} {big_losers_caught:>6} {winners_caught:>4} "
              f"{precision:>8.0f}% ${pnl_saved:>8,.2f} ${pnl_lost:>8,.2f} ${net:>+8,.2f}{marker}")

    # Save features CSV for external ML training
    csv_path = PROJECT_DIR / "journal" / "early_pop_features.csv"
    df_feat.to_csv(csv_path, index=False)
    print(f"\n\nFeature CSV saved to: {csv_path}")
    print(f"  {len(df_feat)} rows x {len(df_feat.columns)} columns")
    print(f"  Use this to train LightGBM classifier for 'is_big_loser' prediction")


if __name__ == "__main__":
    main()
