"""Backtest: Progressive trail tightening — tighter trail as gains increase.

Instead of a fixed trail width at all gain levels, use a staircase:
  +100% gain → 30% trail (exit if drops to +70%)
  +150% gain → 25% trail (exit if drops to +112%)
  +200% gain → 20% trail (exit if drops to +160%)
  +300% gain → 15% trail (exit if drops to +255%)

The key insight: at +100%, a 50% trail means you exit at +50%.
But at +100% you've already proven the trade works — tighten to keep more.

Tests multiple tightening schedules to find the sweet spot.

Usage:
    python scripts/backtest_progressive_tighten.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    TickerCategory,
    V5Config,
    categorize_ticker,
    get_ticker_config,
)
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
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry, created_at
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
    return f"O:{ticker}{exp_str}{ot}{int(strike * 1000):08d}"


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
            SELECT captured_at, midpoint, bid, ask, underlying_price
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
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"], format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def score_to_contracts(score, premium):
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 5
    pos_cap = PORTFOLIO * 0.15
    mult = 0
    for tier_score, tier_mult in SCORE_TIERS:
        if score >= tier_score:
            mult = tier_mult
            break
    if mult == 0:
        return 0
    cost = premium * 100
    if cost <= 0:
        return 0
    return max(1, min(int(per_slot * mult / cost), int(pos_cap / cost)))


def _strip_tz(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def simulate_with_progressive_trail(df, entry_premium, contracts, direction, dte,
                                     expiry_date, ticker, trail_schedule):
    """Run FSM but override the adaptive trail with a progressive tightening schedule.

    trail_schedule: list of (peak_gain_pct, trail_width_pct) sorted descending by peak_gain.
    When peak gain exceeds a tier, use that tier's trail width.
    Below all tiers, use normal FSM behavior.

    The trail is measured as % drop from PEAK premium (not entry).
    """
    if entry_premium <= 0:
        return None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    entry_ts = _strip_tz(df["ts"].iloc[0])

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
    progressive_exit = None  # track if our progressive trail triggered

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium
        now = _strip_tz(df["ts"].iloc[idx])
        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        # Track peak
        peak_prem = state.peak_premium
        if premium > peak_prem:
            peak_prem = premium

        peak_gain = (peak_prem - entry_premium) / entry_premium * 100
        current_gain = (premium - entry_premium) / entry_premium * 100
        drop_from_peak = (peak_prem - premium) / peak_prem * 100 if peak_prem > 0 else 0

        # Check progressive trail BEFORE FSM
        # Find the active trail tier based on peak gain
        active_trail = None
        for tier_gain, tier_trail in trail_schedule:
            if peak_gain >= tier_gain:
                active_trail = tier_trail
                break

        if active_trail is not None and drop_from_peak >= active_trail:
            # Progressive trail triggered — exit
            elapsed = (now - entry_ts).total_seconds() / 60
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl,
                "reason": f"progressive_trail_{active_trail}%",
                "exit_gain": current_gain,
                "peak_gain": peak_gain,
                "drop_from_peak": drop_from_peak,
                "trail_width": active_trail,
                "progressive": True,
            }

        # Normal FSM evaluation
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (premium - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "exit_gain": current_gain,
                "peak_gain": peak_gain,
                "drop_from_peak": drop_from_peak,
                "trail_width": None,
                "progressive": False,
            }

    last_prem = df["premium"].iloc[-1]
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "last_tick",
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "peak_gain": peak_gain,
        "progressive": False,
    }


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals")

    hconn = sqlite3.connect(HARVESTER_DB)

    # Define progressive trail schedules to test
    # Format: list of (peak_gain_threshold, trail_width_pct) — highest first
    schedules = {
        "baseline": None,  # no progressive trail

        # Your suggestion: tighten as gains grow
        "aggressive": [
            (300, 15),   # +300% peak → 15% trail (exit at +255%)
            (200, 20),   # +200% peak → 20% trail (exit at +160%)
            (150, 25),   # +150% peak → 25% trail (exit at +112%)
            (100, 30),   # +100% peak → 30% trail (exit at +70%)
        ],

        # Moderate — wider trails, starts later
        "moderate": [
            (400, 20),   # +400% peak → 20% trail (exit at +320%)
            (300, 25),   # +300% peak → 25% trail (exit at +225%)
            (200, 30),   # +200% peak → 30% trail (exit at +140%)
            (150, 35),   # +150% peak → 35% trail (exit at +97%)
        ],

        # Conservative — only tighten at very high gains
        "conservative": [
            (400, 25),   # +400% peak → 25% trail (exit at +300%)
            (300, 30),   # +300% peak → 30% trail (exit at +210%)
            (200, 35),   # +200% peak → 35% trail (exit at +130%)
        ],

        # Ultra-tight at lower threshold
        "early_tight": [
            (300, 15),
            (200, 20),
            (100, 25),
            (75, 30),
        ],

        # Gentle — just barely tighter than current FSM
        "gentle": [
            (400, 25),
            (200, 30),
            (100, 40),
        ],

        # Only tighten above 200%
        "high_only": [
            (400, 20),
            (300, 25),
            (200, 30),
        ],
    }

    all_results = {}

    for sched_name, schedule in schedules.items():
        results = []
        matched = 0

        for i, sig in enumerate(signals):
            df = load_ticks(hconn, sig)
            if df is None:
                continue

            if sched_name == list(schedules.keys())[0]:
                matched += 1

            ticker = sig["ticker"]
            entry_premium = float(sig["premium"])
            score = sig.get("score", 85)
            contracts = score_to_contracts(score, entry_premium)
            if contracts <= 0:
                continue

            direction = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
            dte = sig.get("_dte", 0)
            expiry_date = sig.get("_expiry_date", "")

            if schedule is None:
                # Baseline — just FSM, no progressive trail
                r = simulate_with_progressive_trail(
                    df, entry_premium, contracts, direction, dte, expiry_date,
                    ticker, trail_schedule=[])
            else:
                r = simulate_with_progressive_trail(
                    df, entry_premium, contracts, direction, dte, expiry_date,
                    ticker, trail_schedule=schedule)

            if r:
                r["ticker"] = ticker
                r["contracts"] = contracts
                results.append(r)

        all_results[sched_name] = results
        if sched_name == list(schedules.keys())[0]:
            print(f"  Matched {matched} signals")

    hconn.close()

    # --- Summary ---
    print(f"\n{'=' * 110}")
    print(f"{'Schedule':<16} {'Total P&L':>12} {'Delta':>10} {'WR':>6} {'Trades':>7} "
          f"{'Prog Exits':>11} {'Avg Exit':>10} {'Avg Peak':>10}")
    print(f"{'=' * 110}")

    baseline_total = sum(r["pnl"] for r in all_results["baseline"])

    for name, results in all_results.items():
        total = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100 if results else 0
        delta = total - baseline_total
        prog_exits = sum(1 for r in results if r.get("progressive"))
        avg_exit = np.mean([r["exit_gain"] for r in results])
        avg_peak = np.mean([r["peak_gain"] for r in results])

        delta_str = f"${delta:>+9,.0f}" if name != "baseline" else "—"
        print(f"  {name:<14} ${total:>10,.0f} {delta_str:>10} {wr:>5.1f}% {len(results):>6} "
              f"{prog_exits:>10} {avg_exit:>9.0f}% {avg_peak:>9.0f}%")

    # --- Per-schedule detail on progressive exits ---
    for name, results in all_results.items():
        if name == "baseline":
            continue

        prog = [r for r in results if r.get("progressive")]
        if not prog:
            continue

        print(f"\n{'=' * 110}")
        print(f"PROGRESSIVE EXITS: {name}")
        print(f"{'=' * 110}")

        # Compare each progressive exit to baseline
        bl_results = all_results["baseline"]
        bl_by_idx = {i: r for i, r in enumerate(bl_results)}

        total_delta = 0
        for i, (r, bl) in enumerate(zip(results, bl_results)):
            if not r.get("progressive"):
                continue
            d = r["pnl"] - bl["pnl"]
            total_delta += d
            peak_str = f"+{r['peak_gain']:.0f}%"
            exit_str = f"+{r['exit_gain']:.0f}%"
            bl_exit_str = f"+{bl['exit_gain']:.0f}%"
            trail_str = f"{r.get('trail_width', '?')}%"
            status = "HELPED" if d > 0 else "HURT" if d < 0 else "SAME"
            print(f"  {r['ticker']:<8} peak {peak_str:>6} | "
                  f"prog exit {exit_str:>6} (trail {trail_str:>4}) vs "
                  f"baseline {bl_exit_str:>6} ({bl['reason']}) | "
                  f"Δ ${d:>+8,.0f} {status}")

        print(f"\n  Net delta from progressive exits: ${total_delta:>+,.0f}")
        helped = sum(1 for r, bl in zip(results, bl_results)
                    if r.get("progressive") and r["pnl"] > bl["pnl"])
        hurt = sum(1 for r, bl in zip(results, bl_results)
                  if r.get("progressive") and r["pnl"] < bl["pnl"])
        print(f"  Helped: {helped}, Hurt: {hurt}")

    # --- Per-ticker best schedule ---
    print(f"\n{'=' * 110}")
    print("PER-TICKER: Which schedule works best?")
    print(f"{'=' * 110}")

    tickers = sorted(set(r["ticker"] for r in all_results["baseline"]))
    for tk in tickers:
        tk_counts = sum(1 for r in all_results["baseline"] if r["ticker"] == tk)
        if tk_counts < 3:
            continue

        print(f"\n  {tk} ({tk_counts} trades):")
        for name, results in all_results.items():
            tk_pnl = sum(r["pnl"] for r in results if r["ticker"] == tk)
            bl_pnl = sum(r["pnl"] for r in all_results["baseline"] if r["ticker"] == tk)
            d = tk_pnl - bl_pnl
            prog = sum(1 for r in results if r["ticker"] == tk and r.get("progressive"))
            d_str = f"${d:>+7,.0f}" if name != "baseline" else "  base "
            print(f"    {name:<16}: ${tk_pnl:>8,.0f} ({d_str}) [{prog} prog exits]")


if __name__ == "__main__":
    main()
