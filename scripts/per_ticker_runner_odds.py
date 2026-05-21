"""Per-ticker runner probability analysis.

For each ticker, at each gain milestone (25%, 50%, 75%, 100%, 150%, 200%, 300%),
compute:
  - How many trades reached that milestone
  - Of those, what % went 50%+ higher from that point
  - Average additional upside
  - What the "runner threshold" is (gain % where >50% chance of running further)

This tells us: "Don't engage ML/partial-sell until this ticker-specific threshold."

Usage:
    python scripts/per_ticker_runner_odds.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import TickerCategory, categorize_ticker

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

MILESTONES = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500]
RUNNER_THRESHOLD = 50  # "runs further" = goes 50%+ higher from milestone


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry, created_at
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

    rows = None
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
            break
    else:
        return None

    if not rows or len(rows) < 10:
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


def analyze_trade(signal, df):
    """For a trade, compute milestone stats: at each gain milestone,
    what was the max additional upside after that point."""
    entry_premium = float(signal["premium"])
    if entry_premium <= 0:
        return None

    premiums = df["premium"].values.astype(float)
    # Track running peak from entry
    gains = (premiums - entry_premium) / entry_premium * 100

    # Session peak gain
    peak_gain = gains.max()
    peak_idx = gains.argmax()

    results = {
        "ticker": signal["ticker"],
        "category": categorize_ticker(signal["ticker"]).value,
        "entry_premium": entry_premium,
        "peak_gain": peak_gain,
        "milestones": {},
    }

    for milestone in MILESTONES:
        # Find first tick where gain >= milestone
        hit_indices = np.where(gains >= milestone)[0]
        if len(hit_indices) == 0:
            continue

        first_hit = hit_indices[0]
        premium_at_milestone = premiums[first_hit]

        # What happened AFTER hitting this milestone?
        remaining_premiums = premiums[first_hit:]
        remaining_gains_from_milestone = (remaining_premiums - premium_at_milestone) / premium_at_milestone * 100

        max_additional_upside = remaining_gains_from_milestone.max()
        max_drawdown_after = remaining_gains_from_milestone.min()

        # Did it go RUNNER_THRESHOLD% higher from this point?
        went_higher = max_additional_upside >= RUNNER_THRESHOLD

        # What was the final gain from milestone (last tick)?
        final_gain_from_milestone = remaining_gains_from_milestone[-1]

        results["milestones"][milestone] = {
            "premium_at_milestone": premium_at_milestone,
            "max_additional_upside": max_additional_upside,
            "max_drawdown_after": max_drawdown_after,
            "final_from_milestone": final_gain_from_milestone,
            "went_higher": went_higher,
            "ticks_remaining": len(remaining_premiums),
        }

    return results


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals with premium > 0")

    print("Connecting to harvester DB...")
    hconn = sqlite3.connect(HARVESTER_DB)

    # Analyze all trades
    all_results = []
    matched = 0
    for i, sig in enumerate(signals):
        df = load_ticks(hconn, sig)
        if df is None:
            continue
        matched += 1
        result = analyze_trade(sig, df)
        if result:
            all_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(signals)}, matched {matched}")

    hconn.close()
    print(f"\nMatched {matched} signals to harvester data, {len(all_results)} analyzed")

    # --- Aggregate by ticker ---
    ticker_stats = defaultdict(lambda: {"trades": 0, "peak_gains": [], "milestones": defaultdict(list)})

    for r in all_results:
        tk = r["ticker"]
        ticker_stats[tk]["trades"] += 1
        ticker_stats[tk]["peak_gains"].append(r["peak_gain"])
        ticker_stats[tk]["category"] = r["category"]
        for ms, data in r["milestones"].items():
            ticker_stats[tk]["milestones"][ms].append(data)

    # --- Per-ticker runner threshold ---
    print("\n" + "=" * 100)
    print("PER-TICKER RUNNER PROBABILITY AT EACH MILESTONE")
    print("=" * 100)
    print(f"'Runner' = goes {RUNNER_THRESHOLD}%+ higher from that gain milestone")
    print()

    # Sort tickers by trade count
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: -x[1]["trades"])

    # Header
    ms_headers = " ".join(f"{m:>6}%" for m in MILESTONES)
    print(f"{'Ticker':<8} {'Cat':<10} {'N':>4} {'Avg Peak':>9}  {ms_headers}")
    print("-" * 120)

    ticker_thresholds = {}

    for tk, stats in sorted_tickers:
        if stats["trades"] < 3:
            continue

        avg_peak = np.mean(stats["peak_gains"])
        cat = stats.get("category", "?")

        cells = []
        runner_threshold = None
        for ms in MILESTONES:
            ms_data = stats["milestones"].get(ms, [])
            if not ms_data:
                cells.append(f"{'--':>7}")
                continue
            n_hit = len(ms_data)
            n_ran = sum(1 for d in ms_data if d["went_higher"])
            pct = n_ran / n_hit * 100 if n_hit > 0 else 0
            cells.append(f"{pct:>4.0f}%/{n_hit:<2d}")

            # Find first milestone where >60% chance of running further
            if runner_threshold is None and pct >= 60 and n_hit >= 3:
                runner_threshold = ms

        ticker_thresholds[tk] = {
            "threshold": runner_threshold,
            "category": cat,
            "trades": stats["trades"],
            "avg_peak": avg_peak,
        }

        line = f"{tk:<8} {cat:<10} {stats['trades']:>4} {avg_peak:>8.0f}%  " + " ".join(cells)
        print(line)

    # --- Summary: Runner threshold per ticker ---
    print("\n" + "=" * 100)
    print("RUNNER THRESHOLD PER TICKER (first milestone where >60% go 50%+ higher, n>=3)")
    print("=" * 100)
    print()

    for tk, info in sorted(ticker_thresholds.items(), key=lambda x: (x[1]["threshold"] or 9999)):
        th = info["threshold"]
        th_str = f"+{th}%" if th else "NEVER (no milestone hit >60% with n>=3)"
        print(f"  {tk:<8} ({info['category']:<10}, {info['trades']:>3} trades, avg peak +{info['avg_peak']:.0f}%) → threshold: {th_str}")

    # --- Category-level analysis ---
    print("\n" + "=" * 100)
    print("CATEGORY-LEVEL RUNNER PROBABILITY")
    print("=" * 100)

    cat_stats = defaultdict(lambda: {"trades": 0, "milestones": defaultdict(list)})
    for r in all_results:
        cat = r["category"]
        cat_stats[cat]["trades"] += 1
        for ms, data in r["milestones"].items():
            cat_stats[cat]["milestones"][ms].append(data)

    for cat in ["HIGH_VOL", "INDEX", "STANDARD"]:
        cdata = cat_stats.get(cat)
        if not cdata:
            continue
        print(f"\n  {cat} ({cdata['trades']} trades)")
        for ms in MILESTONES:
            ms_data = cdata["milestones"].get(ms, [])
            if not ms_data:
                continue
            n_hit = len(ms_data)
            n_ran = sum(1 for d in ms_data if d["went_higher"])
            avg_upside = np.mean([d["max_additional_upside"] for d in ms_data])
            avg_drawdown = np.mean([d["max_drawdown_after"] for d in ms_data])
            pct = n_ran / n_hit * 100
            print(f"    +{ms:>4}%: {n_ran:>3}/{n_hit:<3} ({pct:>5.1f}%) went +{RUNNER_THRESHOLD}%+ higher | "
                  f"avg add'l upside: +{avg_upside:.0f}% | avg max drawdown: {avg_drawdown:.0f}%")

    # --- Detailed per-ticker milestone table for top tickers ---
    print("\n" + "=" * 100)
    print("DETAILED: AVG ADDITIONAL UPSIDE & AVG MAX DRAWDOWN AT EACH MILESTONE")
    print("=" * 100)

    for tk, stats in sorted_tickers:
        if stats["trades"] < 5:
            continue
        print(f"\n  {tk} ({stats.get('category','?')}, {stats['trades']} trades, avg peak +{np.mean(stats['peak_gains']):.0f}%)")
        for ms in MILESTONES:
            ms_data = stats["milestones"].get(ms, [])
            if not ms_data or len(ms_data) < 2:
                continue
            n_hit = len(ms_data)
            n_ran = sum(1 for d in ms_data if d["went_higher"])
            pct = n_ran / n_hit * 100
            avg_up = np.mean([d["max_additional_upside"] for d in ms_data])
            avg_dd = np.mean([d["max_drawdown_after"] for d in ms_data])
            avg_final = np.mean([d["final_from_milestone"] for d in ms_data])
            print(f"    +{ms:>4}%: {n_ran:>2}/{n_hit:<2} ({pct:>5.1f}%) run further | "
                  f"avg upside: +{avg_up:>5.0f}% | avg drawdown: {avg_dd:>6.0f}% | "
                  f"avg final from MS: {avg_final:>+6.0f}%")

    # --- Actionable config output ---
    print("\n" + "=" * 100)
    print("ACTIONABLE: Suggested per-ticker partial-sell thresholds")
    print("=" * 100)
    print()
    print("Tickers where ML/partial-sell should ONLY engage AFTER this gain %:")
    print("(Before threshold: hold all contracts. After threshold: consider partial sell on retrace)")
    print()

    for tk, info in sorted(ticker_thresholds.items(), key=lambda x: (x[1]["threshold"] or 9999)):
        th = info["threshold"]
        if th is None:
            print(f"  {tk:<8}: NO_PARTIAL_SELL  # never reliably runs, sell everything on retrace")
        else:
            print(f"  {tk:<8}: +{th}%  # after +{th}%, {RUNNER_THRESHOLD}%+ further upside likely")


if __name__ == "__main__":
    main()
