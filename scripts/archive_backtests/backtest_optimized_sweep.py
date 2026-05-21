"""Optimized category-aware strategy with circuit breakers.

Builds on Phase 3 results from backtest_category_sweep.py:
- Category-specific exit configs (0DTE high_vol/index/standard, multi-day)
- Adds daily loss circuit breaker (stop trading after N$ loss in a day)
- Adds multi-day max loss cap and score filter
- Adds sizing adjustments (reduce size for expensive multi-day options)
- Tests removing the worst-performing categories entirely

Goal: every day profitable, 15-25%+ daily on $8K.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000

HIGH_VOL = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEXES = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}


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


def simulate_trade(df, entry_idx, entry_premium, contracts, direction, dte, config):
    """Same simulator from category_sweep."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    is_call = direction in ("bullish", "call", "long")
    peak = entry_premium
    entry_underlying = None

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
    adaptive_tiers = config.get("adaptive_tiers", [(40, 40), (150, 45), (400, 30)])
    theta_minutes = config.get("theta_minutes", 0)
    theta_loss_pct = config.get("theta_loss_pct", 0)
    max_loss_dollars = config.get("max_loss_dollars", 0)
    profit_target_pct = config.get("profit_target_pct", 0)
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

        underlying = df["underlying_price"].iloc[idx] or 0
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying

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
        if dte == 0:
            ts = df["ts"].iloc[idx]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts - timedelta(hours=4)
            if et.hour >= 15 and et.minute >= 45:
                return {"pnl": current_pnl, "reason": "eod_cutoff", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        if elapsed < grace_minutes:
            if max_loss_dollars > 0 and current_pnl < -max_loss_dollars:
                return {"pnl": current_pnl, "reason": "max_loss_cap", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}
            continue

        if max_loss_dollars > 0 and current_pnl < -max_loss_dollars:
            return {"pnl": current_pnl, "reason": "max_loss_cap", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

        if profit_target_pct > 0 and gain_pct >= profit_target_pct:
            return {"pnl": current_pnl, "reason": "profit_target", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

        # Scalp trail
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
            if should_scalp:
                return {"pnl": current_pnl, "reason": "scalp_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # Checkpoint
        if checkpoint_enabled and dte == 0:
            if drop_entry >= checkpoint_drop and has_underlying and underlying_against:
                return {"pnl": current_pnl, "reason": "checkpoint_cut", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # Graduated stop
        if has_underlying:
            if underlying_against:
                if drop_entry >= tight_stop:
                    return {"pnl": current_pnl, "reason": "confirmed_stop", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
            else:
                if drop_entry >= backstop:
                    return {"pnl": current_pnl, "reason": "hard_stop", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
        else:
            if drop_entry >= hard_stop:
                return {"pnl": current_pnl, "reason": "hard_stop", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # Soft trail
        if soft_trail_min <= peak_gain < soft_trail_max:
            floor = entry_premium + (peak - entry_premium) * soft_trail_keep
            if premium <= floor:
                return {"pnl": current_pnl, "reason": "soft_trail", "hold": elapsed,
                        "exit_prem": premium, "peak_gain": peak_gain}

        # Adaptive trail
        for tier_min, tier_width in sorted(adaptive_tiers, key=lambda x: -x[0]):
            if peak_gain >= tier_min:
                if drop_peak >= tier_width:
                    return {"pnl": current_pnl, "reason": f"adaptive_{tier_min}", "hold": elapsed,
                            "exit_prem": premium, "peak_gain": peak_gain}
                break

        # Theta timer
        if theta_minutes > 0 and elapsed >= theta_minutes and gain_pct < -theta_loss_pct:
            return {"pnl": current_pnl, "reason": "theta_timer", "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

    last_idx = len(df) - 1
    exit_prem = df["premium"].iloc[last_idx]
    elapsed = (df["ts"].iloc[last_idx] - df["ts"].iloc[entry_idx]).total_seconds() / 60
    pnl = (exit_prem - entry_premium) * contracts * 100
    peak_g = (peak - entry_premium) / entry_premium * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": exit_prem, "peak_gain": peak_g}


# Category configs from Phase 3
CATEGORY_CONFIGS = {
    "0DTE_high_vol": {
        "adaptive_tiers": [(40, 50), (150, 55), (400, 35)],
        "backstop": 65, "checkpoint_enabled": True, "checkpoint_drop": 30,
        "checkpoint_u": 0.5, "grace_minutes": 5, "hard_stop": 30,
        "soft_trail_keep": 0.6, "soft_trail_max": 50, "soft_trail_min": 10,
        "tight_stop": 35, "scalp_0dte_mode": "not_confirming",
        "scalp_multiday_mode": "against", "scalp_peak_trigger": 20,
        "scalp_fade_pct": 0.6,
    },
    "0DTE_index": {
        "adaptive_tiers": [(30, 35), (100, 40), (300, 25)],
        "backstop": 65, "checkpoint_enabled": True, "checkpoint_drop": 30,
        "checkpoint_u": 0.5, "grace_minutes": 5, "hard_stop": 30,
        "profit_target_pct": 30,
        "soft_trail_keep": 0.6, "soft_trail_max": 50, "soft_trail_min": 10,
        "tight_stop": 35, "scalp_0dte_mode": "not_confirming",
        "scalp_multiday_mode": "against", "scalp_peak_trigger": 20,
        "scalp_fade_pct": 0.6,
    },
    "0DTE_standard": {
        "adaptive_tiers": [(30, 35), (100, 40), (300, 25)],
        "backstop": 65, "checkpoint_enabled": True, "checkpoint_drop": 30,
        "checkpoint_u": 0.5, "grace_minutes": 5, "hard_stop": 30,
        "soft_trail_keep": 0.6, "soft_trail_max": 50, "soft_trail_min": 10,
        "tight_stop": 35, "scalp_0dte_mode": "not_confirming",
        "scalp_multiday_mode": "against", "scalp_peak_trigger": 20,
        "scalp_fade_pct": 0.6,
    },
    "Multi-day_high_vol": {
        "adaptive_tiers": [(40, 40), (150, 45), (400, 30)],
        "backstop": 75, "checkpoint_enabled": False, "checkpoint_drop": 30,
        "checkpoint_u": 0.5, "grace_minutes": 5, "hard_stop": 30,
        "soft_trail_keep": 0.5, "soft_trail_max": 50, "soft_trail_min": 15,
        "theta_loss_pct": 15, "theta_minutes": 180, "tight_stop": 52,
        "scalp_0dte_mode": "not_confirming", "scalp_multiday_mode": "against",
        "scalp_peak_trigger": 20, "scalp_fade_pct": 0.6,
    },
    "Multi-day_standard": {
        "adaptive_tiers": [(40, 40), (150, 45), (400, 30)],
        "backstop": 75, "checkpoint_enabled": False, "checkpoint_drop": 30,
        "checkpoint_u": 0.5, "grace_minutes": 5, "hard_stop": 30,
        "soft_trail_keep": 0.5, "soft_trail_max": 50, "soft_trail_min": 15,
        "theta_loss_pct": 15, "theta_minutes": 180, "tight_stop": 52,
        "scalp_0dte_mode": "not_confirming", "scalp_multiday_mode": "against",
        "scalp_peak_trigger": 20, "scalp_fade_pct": 0.6,
    },
}


def get_config(dte, category):
    dte_label = "0DTE" if dte == 0 else "Multi-day"
    key = f"{dte_label}_{category}"
    return CATEGORY_CONFIGS.get(key, CATEGORY_CONFIGS.get(f"{dte_label}_standard", {}))


def run_strategy(trade_data, label, *, daily_loss_limit=0, multi_maxloss=0,
                 multi_min_score=0, multi_max_contracts=0, skip_multi=False,
                 skip_categories=None, extra_overrides=None):
    """Run category-aware strategy with optional circuit breakers."""
    results = []
    daily_pnl = defaultdict(float)
    daily_stopped = set()

    # Sort trades by entry time within each day
    sorted_trades = sorted(trade_data, key=lambda t: t["sig"]["created_at"])

    for t in sorted_trades:
        day = t["day"]

        # Daily circuit breaker
        if daily_loss_limit > 0 and daily_pnl[day] < -daily_loss_limit:
            if day not in daily_stopped:
                daily_stopped.add(day)
            results.append({"pnl": 0, "reason": "circuit_breaker", "hold": 0,
                            "exit_prem": 0, "peak_gain": 0, "ticker": t["ticker"],
                            "day": day, "score": t["score"], "entry": t["entry"],
                            "contracts": 0, "dte": t["dte"], "category": t["category"],
                            "skipped": True})
            continue

        # Skip multi-day entirely?
        if skip_multi and t["dte"] > 0:
            results.append({"pnl": 0, "reason": "skip_multi", "hold": 0,
                            "exit_prem": 0, "peak_gain": 0, "ticker": t["ticker"],
                            "day": day, "score": t["score"], "entry": t["entry"],
                            "contracts": 0, "dte": t["dte"], "category": t["category"],
                            "skipped": True})
            continue

        # Skip certain categories?
        if skip_categories and t["category"] in skip_categories:
            results.append({"pnl": 0, "reason": "skip_category", "hold": 0,
                            "exit_prem": 0, "peak_gain": 0, "ticker": t["ticker"],
                            "day": day, "score": t["score"], "entry": t["entry"],
                            "contracts": 0, "dte": t["dte"], "category": t["category"],
                            "skipped": True})
            continue

        # Multi-day score filter
        if t["dte"] > 0 and multi_min_score > 0 and t["score"] < multi_min_score:
            results.append({"pnl": 0, "reason": "score_filter", "hold": 0,
                            "exit_prem": 0, "peak_gain": 0, "ticker": t["ticker"],
                            "day": day, "score": t["score"], "entry": t["entry"],
                            "contracts": 0, "dte": t["dte"], "category": t["category"],
                            "skipped": True})
            continue

        cfg = get_config(t["dte"], t["category"]).copy()

        # Multi-day max loss override
        if t["dte"] > 0 and multi_maxloss > 0:
            cfg["max_loss_dollars"] = multi_maxloss

        # Extra overrides
        if extra_overrides:
            cfg.update(extra_overrides)

        contracts = t["contracts"]

        # Multi-day contract cap
        if t["dte"] > 0 and multi_max_contracts > 0:
            contracts = min(contracts, multi_max_contracts)

        # Also cap contracts if premium is expensive (>$5) for multi-day
        if t["dte"] > 0 and t["entry"] > 5.0:
            max_by_cost = max(1, int(400 / (t["entry"] * 100)))
            contracts = min(contracts, max_by_cost)

        r = simulate_trade(t["df"], 0, t["entry"], contracts, t["direction"], t["dte"], cfg)
        r.update({"ticker": t["ticker"], "day": day, "score": t["score"],
                   "entry": t["entry"], "contracts": contracts,
                   "dte": t["dte"], "category": t["category"], "skipped": False})
        results.append(r)
        daily_pnl[day] += r["pnl"]

    return results


def print_summary(label, results, all_days):
    active = [r for r in results if not r.get("skipped", False)]
    if not active:
        print(f"\n  {label}: No active trades")
        return

    total = sum(r["pnl"] for r in active)
    wins = sum(1 for r in active if r["pnl"] > 0)
    wr = wins / len(active) * 100
    max_loss = min(r["pnl"] for r in active)
    max_win = max(r["pnl"] for r in active)

    day_pnls = defaultdict(float)
    for r in active:
        day_pnls[r["day"]] += r["pnl"]
    days_won = sum(1 for v in day_pnls.values() if v > 0)
    days_total = len(day_pnls)

    pnl_sorted = sorted([r["pnl"] for r in active], reverse=True)
    no_outlier = sum(pnl_sorted[1:]) if len(pnl_sorted) > 1 else 0
    no_top3 = sum(pnl_sorted[3:]) if len(pnl_sorted) > 3 else 0

    ret_pct = total / PORTFOLIO * 100
    daily_ret = ret_pct / len(all_days) if all_days else 0

    skipped = sum(1 for r in results if r.get("skipped", False))

    print(f"\n  {label}:")
    print(f"    Total: ${total:>+,.0f} ({ret_pct:>+.1f}% on ${PORTFOLIO:,}) | {daily_ret:>+.1f}%/day avg")
    print(f"    Trades: {len(active)} active, {skipped} skipped | WR: {wr:.0f}%")
    print(f"    Max win: ${max_win:>+,.0f} | Max loss: ${max_loss:>+,.0f}")
    print(f"    No-outlier: ${no_outlier:>+,.0f} | No-top-3: ${no_top3:>+,.0f}")
    print(f"    Days won: {days_won}/{days_total}")

    # Daily breakdown
    print(f"    {'Day':<12} {'Trades':>6} {'P&L':>10} {'Cum':>10} {'Daily%':>7}")
    cum = 0
    for day in sorted(all_days):
        d_trades = [r for r in active if r["day"] == day]
        d_pnl = sum(r["pnl"] for r in d_trades)
        cum += d_pnl
        d_pct = d_pnl / PORTFOLIO * 100
        marker = " <<<" if d_pnl > 0 else " XXX"
        print(f"    {day:<12} {len(d_trades):>6} ${d_pnl:>+9,.0f} ${cum:>+9,.0f} {d_pct:>+6.1f}%{marker}")

    return total, days_won, days_total, no_outlier


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    print("Loading tick data...")
    trade_data = []
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        dte = sig.get("_dte", 0)
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        # Sizing
        if score >= 95:
            contracts = 5
        elif score >= 90:
            contracts = 4
        elif score >= 85:
            contracts = 3
        else:
            contracts = 1
        max_cost = PORTFOLIO * 0.20
        while contracts > 1 and adj_entry * contracts * 100 > max_cost:
            contracts -= 1

        cat = categorize_ticker(ticker)
        trade_data.append({
            "sig": sig, "df": df, "ticker": ticker, "direction": direction,
            "score": score, "day": day, "dte": dte, "entry": adj_entry,
            "contracts": contracts, "category": cat,
        })
    harvester_conn.close()

    all_days = sorted(set(t["day"] for t in trade_data))
    n_0dte = sum(1 for t in trade_data if t["dte"] == 0)
    n_multi = sum(1 for t in trade_data if t["dte"] > 0)
    print(f"Loaded {len(trade_data)} trades: {n_0dte} 0DTE, {n_multi} multi-day, {len(all_days)} days\n")

    # ====================================================================
    # Test many combinations
    # ====================================================================
    strategies = [
        ("A: CatAware (no breakers)",
         dict()),
        ("B: CatAware + daily limit $800",
         dict(daily_loss_limit=800)),
        ("C: CatAware + daily limit $500",
         dict(daily_loss_limit=500)),
        ("D: CatAware + daily limit $1000",
         dict(daily_loss_limit=1000)),
        ("E: CatAware + multi maxloss $400",
         dict(multi_maxloss=400)),
        ("F: CatAware + multi maxloss $300",
         dict(multi_maxloss=300)),
        ("G: CatAware + multi maxloss $500",
         dict(multi_maxloss=500)),
        ("H: CatAware + daily $800 + multi $400",
         dict(daily_loss_limit=800, multi_maxloss=400)),
        ("I: CatAware + daily $500 + multi $300",
         dict(daily_loss_limit=500, multi_maxloss=300)),
        ("J: CatAware + daily $800 + multi $400 + score>=95 multi",
         dict(daily_loss_limit=800, multi_maxloss=400, multi_min_score=95)),
        ("K: CatAware + skip multi entirely (0DTE only)",
         dict(skip_multi=True)),
        ("L: CatAware + skip multi + daily $500",
         dict(skip_multi=True, daily_loss_limit=500)),
        ("M: CatAware + multi maxcontracts=2",
         dict(multi_max_contracts=2)),
        ("N: CatAware + multi $400 + maxcontracts=2",
         dict(multi_maxloss=400, multi_max_contracts=2)),
        ("O: CatAware + daily $800 + multi $400 + maxcontracts=2",
         dict(daily_loss_limit=800, multi_maxloss=400, multi_max_contracts=2)),
        ("P: CatAware + daily $1000 + multi $500 + maxcontracts=2",
         dict(daily_loss_limit=1000, multi_maxloss=500, multi_max_contracts=2)),
        ("Q: CatAware + multi score>=90",
         dict(multi_min_score=90)),
        ("R: CatAware + multi score>=90 + multi $400",
         dict(multi_min_score=90, multi_maxloss=400)),
        ("S: CatAware + daily $800 + multi score>=90 + multi $400",
         dict(daily_loss_limit=800, multi_min_score=90, multi_maxloss=400)),
        ("T: CatAware + daily $600 + multi $300 + maxcontracts=2",
         dict(daily_loss_limit=600, multi_maxloss=300, multi_max_contracts=2)),
    ]

    print(f"{'='*120}")
    print(f"STRATEGY COMPARISON — {len(strategies)} variants")
    print(f"{'='*120}")

    all_summaries = []

    for label, kwargs in strategies:
        results = run_strategy(trade_data, label, **kwargs)
        info = print_summary(label, results, all_days)
        if info:
            all_summaries.append((label, *info, kwargs))

    # ====================================================================
    # Rankings
    # ====================================================================
    print(f"\n{'='*120}")
    print("RANKING TABLE")
    print(f"{'='*120}")
    print(f"{'#':>2} {'Strategy':<55} {'Total':>10} {'DaysWon':>8} {'NoOutlier':>10} {'Score':>8}")
    print("-" * 100)

    # Score: days_won * 1000 + no_outlier + total/10 (reward consistency most)
    ranked = sorted(all_summaries,
                    key=lambda x: (x[2], x[4], x[1]),  # days_won, no_outlier, total
                    reverse=True)

    for i, (label, total, days_won, days_total, no_outlier, kwargs) in enumerate(ranked):
        score = days_won * 1000 + no_outlier / 10 + total / 100
        marker = " *** BEST ***" if i == 0 else ""
        print(f"{i+1:>2} {label[:55]:<55} ${total:>+9,.0f} {days_won:>2}/{days_total} "
              f"${no_outlier:>+9,.0f} {score:>7.0f}{marker}")

    # ====================================================================
    # Best strategy — detailed per-trade
    # ====================================================================
    best_label, best_kwargs = ranked[0][0], ranked[0][5]
    print(f"\n{'='*120}")
    print(f"BEST STRATEGY DETAIL: {best_label}")
    print(f"{'='*120}")

    best_results = run_strategy(trade_data, best_label, **best_kwargs)
    active = [r for r in best_results if not r.get("skipped", False)]

    print(f"\n{'Day':<12} {'Ticker':<7} {'DTE':>3} {'Cat':<9} {'Sc':>3} {'Entry':>6} {'Ct':>3} "
          f"{'P&L':>9} {'Peak':>6} {'Reason':<20}")
    print("-" * 100)

    for day in all_days:
        day_trades = sorted([r for r in active if r["day"] == day], key=lambda x: x["pnl"])
        for r in day_trades:
            print(f"{r['day']:<12} {r['ticker']:<7} {r['dte']:>3} {r['category']:<9} "
                  f"{r['score']:>3} ${r['entry']:>5.2f} {r['contracts']:>3} "
                  f"${r['pnl']:>+8,.0f} {r['peak_gain']:>+5.0f}% {r['reason']:<20}")
        day_total = sum(r["pnl"] for r in day_trades)
        day_pct = day_total / PORTFOLIO * 100
        print(f"{'':>12} {'':>7} {'':>3} {'':>9} {'':>3} {'':>6} {'':>3} "
              f"${day_total:>+8,.0f} {day_pct:>+5.1f}% ─── Day Total ───\n")

    # Gate breakdown
    print(f"\nGATE BREAKDOWN:")
    gate_stats = defaultdict(lambda: {"fires": 0, "pnl": 0, "wins": 0})
    for r in active:
        gate_stats[r["reason"]]["fires"] += 1
        gate_stats[r["reason"]]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            gate_stats[r["reason"]]["wins"] += 1

    print(f"{'Gate':<22} {'Fires':>6} {'P&L':>10} {'WR':>6} {'AvgP&L':>9}")
    print("-" * 60)
    for gate, s in sorted(gate_stats.items(), key=lambda x: -x[1]["fires"]):
        wr = s["wins"] / s["fires"] * 100 if s["fires"] > 0 else 0
        avg = s["pnl"] / s["fires"] if s["fires"] > 0 else 0
        print(f"{gate:<22} {s['fires']:>6} ${s['pnl']:>+9,.0f} {wr:>5.0f}% ${avg:>+8,.0f}")


if __name__ == "__main__":
    main()
