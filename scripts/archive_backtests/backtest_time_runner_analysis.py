"""Backtest: Time-of-day analysis + Runner vs Scalp strategy optimization.

Runs the full V5 FSM backtest over the last 60 days and analyzes:
1. Optimal entry time windows (when to buy options)
2. Runner detection accuracy (when to let trades ride)
3. Scalp profit-taking ranges (when to take quick profit on non-runners)

Usage:
    python scripts/backtest_time_runner_analysis.py
    python scripts/backtest_time_runner_analysis.py --days 60 --portfolio 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
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

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

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

MOVE_WINDOW_MIN = 120
PRE_MOVE_LOOKBACK = 15
COOLDOWN_MIN = 30


# ---- Import from training script ----
from scripts.train_option_signals_v2 import (
    find_atm_strike,
    load_single_strike_data,
    simulate_with_production_fsm,
    compute_setup_features,
    FEATURE_COLS,
    TICKER_MOVE_PCT,
    MIN_MOVE_PCT,
)


def _ts_to_minutes_since_open(ts_str: str) -> int:
    """Convert timestamp to minutes since market open (9:30 ET)."""
    try:
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo:
            ts = ts.tz_convert("America/New_York")
        return max(0, (ts.hour - 9) * 60 + ts.minute - 30)
    except Exception:
        return 0


def run_full_backtest(
    conn: sqlite3.Connection,
    days: int = 60,
    portfolio: float = 20000,
) -> list[dict]:
    """Run backtest across all tickers, collecting detailed trade data."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: pip install lightgbm")
        return []

    # Get test dates (last N days)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc ORDER BY 1 DESC LIMIT ?",
        (days,),
    ).fetchall()]
    all_dates.sort()
    print(f"Backtesting {len(all_dates)} days: {all_dates[0]} to {all_dates[-1]}")

    all_trades = []

    for ticker in TICKERS:
        # Load model
        model_path = MODEL_DIR / f"signal_{ticker}.lgb"
        meta_path = MODEL_DIR / f"signal_{ticker}_meta.json"
        if not model_path.exists():
            model_path = MODEL_DIR / "signal_GENERIC.lgb"
            meta_path = MODEL_DIR / "signal_GENERIC_meta.json"
        if not model_path.exists():
            print(f"  {ticker}: no model, skipping")
            continue

        model = lgb.Booster(model_file=str(model_path))
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        features = meta.get("features", FEATURE_COLS)
        threshold = meta.get("optimal_threshold", 0.5)

        # Runner model
        runner_model = None
        runner_features = features
        rpath = MODEL_DIR / f"runner_{ticker}.lgb"
        rmeta_path = MODEL_DIR / f"runner_{ticker}_meta.json"
        if not rpath.exists():
            rpath = MODEL_DIR / "runner_GENERIC.lgb"
            rmeta_path = MODEL_DIR / "runner_GENERIC_meta.json"
        if rpath.exists():
            runner_model = lgb.Booster(model_file=str(rpath))
            if rmeta_path.exists():
                with open(rmeta_path) as f:
                    rmeta = json.load(f)
                    runner_features = rmeta.get("features", features)

        ticker_dates = [d for d in all_dates if conn.execute(
            "SELECT 1 FROM option_ohlc WHERE ticker=? AND timestamp LIKE ? LIMIT 1",
            (ticker, f"{d}%"),
        ).fetchone()]

        print(f"  {ticker}: {len(ticker_dates)} days, model threshold={threshold:.2f}")

        for dt in ticker_dates:
            stock = pd.read_sql_query(
                "SELECT * FROM stock_ohlc WHERE ticker=? AND timestamp LIKE ? ORDER BY timestamp",
                conn, params=(ticker, f"{dt}%"),
            )

            for right in ["CALL", "PUT"]:
                strike = find_atm_strike(conn, ticker, dt, right)
                if strike is None:
                    continue

                ohlc, quotes, greeks = load_single_strike_data(conn, ticker, dt, right, strike)
                if len(ohlc) < 30:
                    continue

                last_trade_idx = -COOLDOWN_MIN

                # Scan every 5 minutes
                for idx in range(PRE_MOVE_LOOKBACK, len(ohlc) - 10, 5):
                    if idx - last_trade_idx < COOLDOWN_MIN:
                        continue

                    feat = compute_setup_features(ohlc, quotes, greeks, stock, idx)
                    if feat is None:
                        continue

                    X = np.array([[feat.get(c, 0) for c in features]])
                    prob = float(model.predict(X)[0])

                    if prob < threshold:
                        continue

                    entry_price = feat["premium"]
                    if not entry_price or entry_price <= 0 or np.isnan(entry_price):
                        continue

                    # Runner prediction
                    runner_score = 0.0
                    if runner_model is not None:
                        X_r = np.array([[feat.get(c, 0) for c in runner_features]])
                        runner_score = float(runner_model.predict(X_r)[0])

                    # Get entry time
                    entry_ts = str(ohlc.iloc[idx]["timestamp"])
                    minutes_since_open = _ts_to_minutes_since_open(entry_ts)

                    # Sizing (production: flat 85% budget)
                    per_trade = (portfolio * 0.75) / 4  # max_risk=75%, max_concurrent=4
                    trade_budget = per_trade * 0.85
                    contracts = max(1, int(trade_budget / (entry_price * 100)))

                    # Run REAL V5 FSM
                    result = simulate_with_production_fsm(
                        ohlc, quotes, greeks, idx,
                        ticker=ticker, dte=0, expiry_date=dt,
                        contracts=contracts,
                    )
                    if result is None:
                        continue

                    # Track premium path for peak analysis
                    peak_prem = entry_price
                    peak_minutes = 0
                    prem_at_10min = None
                    prem_at_20min = None
                    prem_at_30min = None
                    prem_at_60min = None

                    end_idx = min(idx + MOVE_WINDOW_MIN, len(ohlc))
                    for j in range(idx + 1, end_idx):
                        p = ohlc.iloc[j].get("close", 0)
                        if p and p > 0 and not np.isnan(p):
                            if p > peak_prem:
                                peak_prem = p
                                peak_minutes = j - idx

                            elapsed = j - idx
                            if elapsed == 10 and prem_at_10min is None:
                                prem_at_10min = p
                            if elapsed == 20 and prem_at_20min is None:
                                prem_at_20min = p
                            if elapsed == 30 and prem_at_30min is None:
                                prem_at_30min = p
                            if elapsed == 60 and prem_at_60min is None:
                                prem_at_60min = p

                    peak_gain_pct = (peak_prem / entry_price - 1) * 100

                    trade = {
                        "ticker": ticker,
                        "date": dt,
                        "right": right,
                        "entry_ts": entry_ts,
                        "minutes_since_open": minutes_since_open,
                        "hour_bucket": minutes_since_open // 60,
                        "entry_price": entry_price,
                        "confidence": prob,
                        "runner_score": runner_score,
                        "contracts": contracts,
                        "pnl_pct": result["pnl_pct"],
                        "pnl_dollars": result["pnl_dollars"],
                        "exit_reason": result["reason"],
                        "hold_minutes": result["hold_minutes"],
                        "peak_gain_pct": peak_gain_pct,
                        "peak_minutes": peak_minutes,
                        # Premium checkpoints (% change from entry)
                        "gain_at_10min": ((prem_at_10min / entry_price - 1) * 100) if prem_at_10min else None,
                        "gain_at_20min": ((prem_at_20min / entry_price - 1) * 100) if prem_at_20min else None,
                        "gain_at_30min": ((prem_at_30min / entry_price - 1) * 100) if prem_at_30min else None,
                        "gain_at_60min": ((prem_at_60min / entry_price - 1) * 100) if prem_at_60min else None,
                        "is_winner": result["pnl_dollars"] > 0,
                    }
                    all_trades.append(trade)
                    last_trade_idx = idx

    return all_trades


def analyze_time_of_day(trades: list[dict]):
    """Analyze win rate and P&L by time-of-day window."""
    print(f"\n{'='*80}")
    print("TIME-OF-DAY ANALYSIS — When should we be buying options?")
    print(f"{'='*80}\n")

    # Define windows
    windows = [
        ("9:30-10:00 (Open Kill Zone)", 0, 30),
        ("10:00-10:30 (Early Momentum)", 30, 60),
        ("10:30-11:30 (Late Morning)", 60, 120),
        ("11:30-12:30 (Midday Lull)", 120, 180),
        ("12:30-1:30 (Early Afternoon)", 180, 240),
        ("1:30-3:00 (Danger Zone)", 240, 330),
        ("3:00-4:00 (Power Hour)", 330, 390),
    ]

    print(f"{'Window':<35} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Avg P&L':>10} {'Total P&L':>12} {'Avg Peak':>9} {'Runners':>8}")
    print("-" * 105)

    for label, start, end in windows:
        w_trades = [t for t in trades if start <= t["minutes_since_open"] < end]
        if not w_trades:
            print(f"{label:<35} {'—':>7}")
            continue

        wins = sum(1 for t in w_trades if t["is_winner"])
        wr = wins / len(w_trades) * 100
        avg_pnl = np.mean([t["pnl_dollars"] for t in w_trades])
        total_pnl = sum(t["pnl_dollars"] for t in w_trades)
        avg_peak = np.mean([t["peak_gain_pct"] for t in w_trades])
        runners = sum(1 for t in w_trades if t["peak_gain_pct"] >= 50)

        marker = "***" if wr >= 55 else "   "
        print(f"{label:<35} {len(w_trades):>7} {wins:>6} {wr:>5.1f}% ${avg_pnl:>8,.0f} ${total_pnl:>10,.0f} {avg_peak:>8.1f}% {runners:>7} {marker}")

    # Finer 15-min buckets for the first 2 hours
    print(f"\n{'='*80}")
    print("FIRST 2 HOURS — 15-minute buckets (where the money is)")
    print(f"{'='*80}\n")

    print(f"{'Window':<25} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Avg P&L':>10} {'Total P&L':>12} {'Avg Peak':>9}")
    print("-" * 85)

    for start in range(0, 120, 15):
        end = start + 15
        hour = 9 + (start + 30) // 60
        minute = (start + 30) % 60
        hour_end = 9 + (end + 30) // 60
        minute_end = (end + 30) % 60
        label = f"{hour}:{minute:02d}-{hour_end}:{minute_end:02d}"

        w_trades = [t for t in trades if start <= t["minutes_since_open"] < end]
        if not w_trades:
            print(f"{label:<25} {'—':>7}")
            continue

        wins = sum(1 for t in w_trades if t["is_winner"])
        wr = wins / len(w_trades) * 100
        avg_pnl = np.mean([t["pnl_dollars"] for t in w_trades])
        total_pnl = sum(t["pnl_dollars"] for t in w_trades)
        avg_peak = np.mean([t["peak_gain_pct"] for t in w_trades])

        marker = "<<<" if wr >= 55 else ""
        print(f"{label:<25} {len(w_trades):>7} {wins:>6} {wr:>5.1f}% ${avg_pnl:>8,.0f} ${total_pnl:>10,.0f} {avg_peak:>8.1f}% {marker}")

    # Recommended cutoff analysis
    print(f"\n{'='*80}")
    print("CUMULATIVE CUTOFF — What if we stop trading after X minutes?")
    print(f"{'='*80}\n")

    print(f"{'Stop After':<20} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Total P&L':>12} {'$/Trade':>10}")
    print("-" * 70)

    for cutoff in [30, 45, 60, 90, 120, 150, 180, 240, 330, 390]:
        c_trades = [t for t in trades if t["minutes_since_open"] < cutoff]
        if not c_trades:
            continue
        wins = sum(1 for t in c_trades if t["is_winner"])
        wr = wins / len(c_trades) * 100
        total = sum(t["pnl_dollars"] for t in c_trades)
        avg = total / len(c_trades)
        hour = 9 + (cutoff + 30) // 60
        minute = (cutoff + 30) % 60
        print(f"Before {hour}:{minute:02d} ET{'':<8} {len(c_trades):>7} {wins:>6} {wr:>5.1f}% ${total:>10,.0f} ${avg:>8,.0f}")


def analyze_runners(trades: list[dict]):
    """Analyze runner behavior: detection, capture, and scalp opportunities."""
    print(f"\n{'='*80}")
    print("RUNNER vs SCALP ANALYSIS — When to let it ride vs take quick profit")
    print(f"{'='*80}\n")

    # Classify trades by peak gain
    tiers = [
        ("Losers (< 0%)", lambda t: t["peak_gain_pct"] < 0),
        ("Tiny (0-10%)", lambda t: 0 <= t["peak_gain_pct"] < 10),
        ("Small (10-20%)", lambda t: 10 <= t["peak_gain_pct"] < 20),
        ("Moderate (20-40%)", lambda t: 20 <= t["peak_gain_pct"] < 40),
        ("Strong (40-80%)", lambda t: 40 <= t["peak_gain_pct"] < 80),
        ("Runner (80-150%)", lambda t: 80 <= t["peak_gain_pct"] < 150),
        ("Moon (150%+)", lambda t: t["peak_gain_pct"] >= 150),
    ]

    print(f"{'Tier':<25} {'Count':>7} {'%':>6} {'Avg FSM P&L':>12} {'Total P&L':>12} {'Avg Peak':>9} {'Avg Hold':>9}")
    print("-" * 90)

    for label, fn in tiers:
        tier_trades = [t for t in trades if fn(t)]
        if not tier_trades:
            print(f"{label:<25} {0:>7}")
            continue

        avg_pnl = np.mean([t["pnl_dollars"] for t in tier_trades])
        total_pnl = sum(t["pnl_dollars"] for t in tier_trades)
        avg_peak = np.mean([t["peak_gain_pct"] for t in tier_trades])
        avg_hold = np.mean([t["hold_minutes"] for t in tier_trades])
        pct = len(tier_trades) / len(trades) * 100

        print(f"{label:<25} {len(tier_trades):>7} {pct:>5.1f}% ${avg_pnl:>10,.0f} ${total_pnl:>10,.0f} {avg_peak:>8.1f}% {avg_hold:>8.1f}m")

    # Runner score prediction accuracy
    print(f"\n{'='*80}")
    print("RUNNER SCORE vs ACTUAL OUTCOME — Does the runner model predict correctly?")
    print(f"{'='*80}\n")

    scored_trades = [t for t in trades if t["runner_score"] > 0]
    if scored_trades:
        # Bucket by runner score
        buckets = [
            ("Score < 0.3 (low)", lambda t: t["runner_score"] < 0.3),
            ("Score 0.3-0.5", lambda t: 0.3 <= t["runner_score"] < 0.5),
            ("Score 0.5-0.7", lambda t: 0.5 <= t["runner_score"] < 0.7),
            ("Score 0.7+ (high)", lambda t: t["runner_score"] >= 0.7),
        ]

        print(f"{'Runner Score':<25} {'Count':>7} {'Actual Runners':>15} {'Runner%':>8} {'Avg Peak':>9} {'Avg P&L':>10}")
        print("-" * 80)

        for label, fn in buckets:
            b_trades = [t for t in scored_trades if fn(t)]
            if not b_trades:
                continue
            actual_runners = sum(1 for t in b_trades if t["peak_gain_pct"] >= 50)
            runner_pct = actual_runners / len(b_trades) * 100
            avg_peak = np.mean([t["peak_gain_pct"] for t in b_trades])
            avg_pnl = np.mean([t["pnl_dollars"] for t in b_trades])
            print(f"{label:<25} {len(b_trades):>7} {actual_runners:>15} {runner_pct:>7.1f}% {avg_peak:>8.1f}% ${avg_pnl:>8,.0f}")
    else:
        print("  No runner scores available (no runner model loaded)")

    # How much of the peak do we capture?
    print(f"\n{'='*80}")
    print("PEAK CAPTURE — How much of the runner do we actually keep?")
    print(f"{'='*80}\n")

    winners = [t for t in trades if t["peak_gain_pct"] >= 20]
    if winners:
        print(f"{'Peak Range':<25} {'Count':>7} {'Avg Peak':>9} {'Avg Exit':>9} {'Capture%':>9} {'Avg P&L':>10}")
        print("-" * 75)

        peak_tiers = [
            ("20-40% peak", lambda t: 20 <= t["peak_gain_pct"] < 40),
            ("40-80% peak", lambda t: 40 <= t["peak_gain_pct"] < 80),
            ("80-150% peak", lambda t: 80 <= t["peak_gain_pct"] < 150),
            ("150%+ peak", lambda t: t["peak_gain_pct"] >= 150),
        ]

        for label, fn in peak_tiers:
            tier = [t for t in winners if fn(t)]
            if not tier:
                continue
            avg_peak = np.mean([t["peak_gain_pct"] for t in tier])
            avg_exit = np.mean([t["pnl_pct"] for t in tier])
            capture = avg_exit / avg_peak * 100 if avg_peak > 0 else 0
            avg_pnl = np.mean([t["pnl_dollars"] for t in tier])
            print(f"{label:<25} {len(tier):>7} {avg_peak:>8.1f}% {avg_exit:>8.1f}% {capture:>8.1f}% ${avg_pnl:>8,.0f}")


def analyze_scalp_strategy(trades: list[dict]):
    """Test different scalp profit-taking thresholds."""
    print(f"\n{'='*80}")
    print("SCALP STRATEGY — What if we take profit at X% for non-runners?")
    print(f"{'='*80}\n")

    # For each trade, compute what P&L would be if we exited at various thresholds
    # using the premium checkpoints we captured
    print("Simulated: if we had a hard take-profit at each level (using actual premium path):\n")

    print(f"{'Take-Profit':<15} {'Triggered':>10} {'Hit Rate':>10} {'Avg P&L':>10} {'Total P&L':>12} {'vs FSM':>12}")
    print("-" * 75)

    # FSM baseline
    fsm_total = sum(t["pnl_dollars"] for t in trades)
    fsm_avg = fsm_total / len(trades) if trades else 0

    for tp_pct in [10, 15, 20, 25, 30, 40, 50]:
        triggered = 0
        simulated_pnl = 0

        for t in trades:
            if t["peak_gain_pct"] >= tp_pct:
                # Would have hit take-profit
                triggered += 1
                # Simulated P&L: take profit at tp_pct
                simulated_trade_pnl = t["entry_price"] * (tp_pct / 100) * t["contracts"] * 100
                simulated_pnl += simulated_trade_pnl
            else:
                # Didn't reach TP — use actual FSM exit
                simulated_pnl += t["pnl_dollars"]

        hit_rate = triggered / len(trades) * 100 if trades else 0
        avg_pnl = simulated_pnl / len(trades) if trades else 0
        diff = simulated_pnl - fsm_total

        print(f"+{tp_pct}%{'':<10} {triggered:>10} {hit_rate:>9.1f}% ${avg_pnl:>8,.0f} ${simulated_pnl:>10,.0f} ${diff:>+10,.0f}")

    print(f"{'FSM (actual)':<15} {'':<10} {'':<10} ${fsm_avg:>8,.0f} ${fsm_total:>10,.0f}")

    # Now test HYBRID: scalp non-runners, let runners ride
    print(f"\n{'='*80}")
    print("HYBRID STRATEGY — Scalp non-runners, let runners ride")
    print(f"{'='*80}")
    print("Uses runner_score to decide: high score = let it ride, low score = scalp\n")

    runner_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    scalp_targets = [15, 20, 25, 30]

    print(f"{'Runner Thresh':<15} {'Scalp@':>7} {'Scalps':>7} {'Runs':>6} {'Total P&L':>12} {'vs FSM':>12} {'Run P&L':>10} {'Scalp P&L':>10}")
    print("-" * 95)

    for rt in runner_thresholds:
        for st in scalp_targets:
            scalp_count = 0
            run_count = 0
            scalp_pnl = 0
            run_pnl = 0

            for t in trades:
                if t["runner_score"] >= rt:
                    # Let it ride — use FSM exit
                    run_count += 1
                    run_pnl += t["pnl_dollars"]
                else:
                    # Scalp — take profit at st% or use FSM if it doesn't reach
                    scalp_count += 1
                    if t["peak_gain_pct"] >= st:
                        scalp_pnl += t["entry_price"] * (st / 100) * t["contracts"] * 100
                    else:
                        scalp_pnl += t["pnl_dollars"]

            total = scalp_pnl + run_pnl
            diff = total - fsm_total
            print(f"rs >= {rt:<9.1f} +{st}%{'':<3} {scalp_count:>7} {run_count:>6} ${total:>10,.0f} ${diff:>+10,.0f} ${run_pnl:>8,.0f} ${scalp_pnl:>8,.0f}")


def analyze_per_ticker(trades: list[dict]):
    """Per-ticker breakdown."""
    print(f"\n{'='*80}")
    print("PER-TICKER BREAKDOWN")
    print(f"{'='*80}\n")

    print(f"{'Ticker':<8} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Avg Peak':>9} {'Runners':>8} {'Best Window':>15}")
    print("-" * 100)

    for ticker in TICKERS:
        t_trades = [t for t in trades if t["ticker"] == ticker]
        if not t_trades:
            continue

        wins = sum(1 for t in t_trades if t["is_winner"])
        wr = wins / len(t_trades) * 100
        total_pnl = sum(t["pnl_dollars"] for t in t_trades)
        avg_pnl = np.mean([t["pnl_dollars"] for t in t_trades])
        avg_peak = np.mean([t["peak_gain_pct"] for t in t_trades])
        runners = sum(1 for t in t_trades if t["peak_gain_pct"] >= 50)

        # Best time window for this ticker
        best_window = ""
        best_wr = 0
        for start in range(0, 330, 60):
            end = start + 60
            w = [t for t in t_trades if start <= t["minutes_since_open"] < end]
            if len(w) >= 5:
                ww = sum(1 for t in w if t["is_winner"]) / len(w) * 100
                if ww > best_wr:
                    best_wr = ww
                    hour = 9 + (start + 30) // 60
                    best_window = f"{hour}:{'30' if (start+30)%60==30 else '00'} ({ww:.0f}%)"

        print(f"{ticker:<8} {len(t_trades):>7} {wins:>6} {wr:>5.1f}% ${total_pnl:>10,.0f} ${avg_pnl:>8,.0f} {avg_peak:>8.1f}% {runners:>7}  {best_window:>15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--portfolio", type=float, default=20000)
    parser.add_argument("--db", type=str, default=THETADATA_DB)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    print(f"\n{'='*80}")
    print(f"FULL BACKTEST — Last {args.days} days, ${args.portfolio:,.0f} portfolio")
    print(f"Using combined ML models + V5 FSM exit engine")
    print(f"{'='*80}\n")

    trades = run_full_backtest(conn, days=args.days, portfolio=args.portfolio)
    conn.close()

    if not trades:
        print("No trades generated!")
        return

    # Overall summary
    wins = sum(1 for t in trades if t["is_winner"])
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    print(f"\n{'='*80}")
    print(f"OVERALL: {len(trades)} trades | {wins} wins ({wins/len(trades)*100:.1f}%) | P&L: ${total_pnl:,.0f}")
    print(f"Avg P&L/trade: ${total_pnl/len(trades):,.0f} | Avg hold: {np.mean([t['hold_minutes'] for t in trades]):.0f}min")
    print(f"{'='*80}")

    analyze_time_of_day(trades)
    analyze_runners(trades)
    analyze_scalp_strategy(trades)
    analyze_per_ticker(trades)

    # Save raw trades for further analysis
    out_path = PROJECT_DIR / "journal" / "backtest_time_runner.json"
    with open(out_path, "w") as f:
        json.dump(trades, f, indent=2, default=str)
    print(f"\nRaw trades saved to: {out_path}")


if __name__ == "__main__":
    main()
