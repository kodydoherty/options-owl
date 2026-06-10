"""Backtest premium cap sensitivity analysis.

Tests different premium cap levels against historical signals to find
the optimal cap that maximizes P&L without taking on excessive risk.

Usage:
    python scripts/backtest_premium_cap.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import get_ticker_config
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
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

# Premium cap scenarios to test
CAP_SCENARIOS = [
    ("$5 flat (current base)", 5.00, 5.00, 5.00),     # current production base
    ("$5/$6/$8 tiered", 5.00, 6.00, 8.00),             # current tiered
    ("$6/$7/$9 tiered", 6.00, 7.00, 9.00),             # relaxed
    ("$7/$8/$10 tiered", 7.00, 8.00, 10.00),           # wide
    ("$8 flat", 8.00, 8.00, 8.00),                     # generous flat
    ("$10 flat", 10.00, 10.00, 10.00),                  # very wide
    ("No cap", 999.0, 999.0, 999.0),                    # unlimited
]
# Tuple: (label, base_cap, mid_cap_score120, high_cap_score150)

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}


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


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM"):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
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
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def passes_premium_cap(premium, score, ticker, base_cap, mid_cap, high_cap):
    """Check if a signal passes a given premium cap configuration."""
    if ticker in INDEX_TICKERS:
        return True  # Index options exempt from premium cap
    if score >= 150:
        cap = high_cap
    elif score >= 120:
        cap = mid_cap
    else:
        cap = base_cap
    return premium <= cap


def momentum_blocked(df, direction):
    """Simulate MomentumConfirmGate — returns True if blocked."""
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


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-compute tick data and sizing for all signals
    prepared = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        premium = sig["premium"]

        if score < 78:
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
        max_risk_pct = 0.75
        max_concurrent = 4
        max_position_pct = 0.15
        deployable = PORTFOLIO * max_risk_pct
        per_slot = deployable / max_concurrent
        position_cap = PORTFOLIO * max_position_pct

        SCORE_TIERS = [
            (135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25),
        ]
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
        })

    harvester_conn.close()
    print(f"Prepared {len(prepared)} signals with tick data ({no_data} skipped)")

    # Run each cap scenario
    print(f"\n{'=' * 100}")
    print(f"PREMIUM CAP SENSITIVITY ANALYSIS — {len(prepared)} signals with tick data")
    print(f"{'=' * 100}")

    scenario_results = []

    for label, base_cap, mid_cap, high_cap in CAP_SCENARIOS:
        trades_taken = 0
        trades_blocked_cap = 0
        trades_blocked_momentum = 0
        total_pnl = 0.0
        wins = 0
        losses = 0
        blocked_would_pnl = 0.0
        blocked_details = []

        for sig in prepared:
            cap_ok = passes_premium_cap(
                sig["premium"], sig["score"], sig["ticker"],
                base_cap, mid_cap, high_cap
            )

            if not cap_ok:
                trades_blocked_cap += 1
                # Still simulate to see what we missed
                result = simulate_trade(
                    sig["df"], sig["premium"], sig["contracts"],
                    sig["direction"], sig["dte"], sig["expiry_date"],
                    ticker=sig["ticker"]
                )
                blocked_would_pnl += result["pnl"]
                blocked_details.append({
                    "ticker": sig["ticker"],
                    "premium": sig["premium"],
                    "score": sig["score"],
                    "would_pnl": result["pnl"],
                    "day": sig["day"],
                })
                continue

            if sig["momentum_blocked"]:
                trades_blocked_momentum += 1
                continue

            result = simulate_trade(
                sig["df"], sig["premium"], sig["contracts"],
                sig["direction"], sig["dte"], sig["expiry_date"],
                ticker=sig["ticker"]
            )
            total_pnl += result["pnl"]
            trades_taken += 1
            if result["pnl"] > 0:
                wins += 1
            else:
                losses += 1

        wr = wins / trades_taken * 100 if trades_taken > 0 else 0
        scenario_results.append({
            "label": label,
            "base": base_cap,
            "mid": mid_cap,
            "high": high_cap,
            "taken": trades_taken,
            "blocked_cap": trades_blocked_cap,
            "blocked_mom": trades_blocked_momentum,
            "pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "wr": wr,
            "missed_pnl": blocked_would_pnl,
            "blocked_details": blocked_details,
        })

    # Summary table
    print(f"\n{'Scenario':<25} {'Trades':>6} {'CapBlk':>7} {'MomBlk':>7} "
          f"{'Total P&L':>12} {'Win%':>6} {'Missed P&L':>12} {'Net If Taken':>12}")
    print("-" * 105)
    for s in scenario_results:
        net_if_taken = s["pnl"] + s["missed_pnl"]
        print(f"{s['label']:<25} {s['taken']:>6} {s['blocked_cap']:>7} {s['blocked_mom']:>7} "
              f"${s['pnl']:>10,.2f} {s['wr']:>5.1f}% ${s['missed_pnl']:>10,.2f} ${net_if_taken:>10,.2f}")

    # Detail on what the premium cap blocked (for current production cap)
    print(f"\n{'=' * 100}")
    print(f"TRADES BLOCKED BY CURRENT PRODUCTION CAP ($5/$6/$8 tiered)")
    print(f"{'=' * 100}")

    current = next(s for s in scenario_results if s["label"] == "$5/$6/$8 tiered")
    if current["blocked_details"]:
        print(f"\n{'Day':<12} {'Ticker':<7} {'Premium':>8} {'Score':>6} {'Would P&L':>10}")
        print("-" * 50)
        for d in sorted(current["blocked_details"], key=lambda x: x["would_pnl"]):
            print(f"{d['day']:<12} {d['ticker']:<7} ${d['premium']:>6.2f} {d['score']:>6} "
                  f"${d['would_pnl']:>9,.2f}")

        won = [d for d in current["blocked_details"] if d["would_pnl"] > 0]
        lost = [d for d in current["blocked_details"] if d["would_pnl"] <= 0]
        print(f"\nBlocked winners: {len(won)} (${sum(d['would_pnl'] for d in won):,.2f})")
        print(f"Blocked losers:  {len(lost)} (${sum(d['would_pnl'] for d in lost):,.2f})")
        print(f"Net missed:      ${current['missed_pnl']:,.2f}")
    else:
        print("No trades blocked by premium cap.")

    # Recommendation
    print(f"\n{'=' * 100}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 100}")
    best = max(scenario_results, key=lambda s: s["pnl"])
    print(f"Best P&L scenario: {best['label']} — ${best['pnl']:,.2f} "
          f"({best['taken']} trades, {best['wr']:.1f}% WR)")

    current_pnl = current["pnl"]
    if best["pnl"] > current_pnl * 1.05:
        delta = best["pnl"] - current_pnl
        print(f"Raising cap to {best['label']} would add ${delta:,.2f} (+{delta/current_pnl*100:.1f}%)")
    else:
        print(f"Current cap is near-optimal (within 5% of best)")


if __name__ == "__main__":
    main()
