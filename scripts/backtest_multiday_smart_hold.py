"""Backtest: Smart overnight hold — ticker-aware + momentum filter.

Pattern from data analysis:
  - AMD, TSLA, AAPL recover on day 2 most often (60-100% profitable)
  - META, MSTR, NVDA almost never recover (0% profitable)
  - Premium momentum into close (+20%+) predicts day 2 success (61%)
  - EOD gain +30%+ predicts day 2 continuation (59%)

Strategy: disable soft trail on day 1 for multi-day trades, but at EOD:
  - Always cut: META, MSTR, NVDA, GOOGL (0% d2 recovery in data)
  - Always hold: AMD (100% d2 recovery in data) — if not deeply underwater
  - Conditional hold: TSLA, AAPL, PLTR, AMZN, MSFT, AVGO — based on
    EOD gain level and/or momentum

Sweep parameters:
  - Which tickers to hold overnight
  - EOD gain threshold for conditional tickers
  - Day 1 momentum threshold
  - Underlying confirms requirement

Usage:
    python scripts/backtest_multiday_smart_hold.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import AdaptiveTier, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from backtest_ladder_report import (
    check_momentum_gate,
    load_signals,
    load_ticks,
    size_contracts,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

_V6 = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True, V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333, V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True, V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15, ENABLE_V6_PER_TICKER_CONFIG=True,
)

# Tickers that NEVER recover on day 2 — always cut at EOD day 1
ALWAYS_CUT = {"META", "MSTR", "NVDA", "GOOGL"}
# Tickers with strong day 2 recovery — hold with minimal filter
STRONG_HOLD = {"AMD"}
# Tickers with good day 2 recovery — hold with moderate filter
GOOD_HOLD = {"TSLA", "AAPL"}
# Conditional tickers — hold only if strong signal
CONDITIONAL = {"PLTR", "AMZN", "MSFT", "AVGO"}


def compute_d1_momentum(df, entry_date):
    """Compute late-day premium momentum on day 1.

    Returns (d1_eod_gain_pct, d1_momentum_pct, d1_last_premium).
    d1_momentum = (avg premium last 30min - avg premium first 30min) / first 30min avg
    """
    entry_prem = df["premium"].iloc[0]
    day1_mask = df["ts"].apply(
        lambda x: (pd.Timestamp(x).date() if not hasattr(x, "date") else x.date()) == entry_date
    )
    day1 = df[day1_mask]
    if len(day1) < 2:
        return 0.0, 0.0, entry_prem

    d1_last = day1["premium"].iloc[-1]
    d1_eod_gain = (d1_last - entry_prem) / entry_prem * 100 if entry_prem > 0 else 0

    first_ts = day1["ts"].iloc[0]
    last_ts = day1["ts"].iloc[-1]
    if hasattr(first_ts, "to_pydatetime"):
        first_ts = first_ts.to_pydatetime()
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()

    early = day1[day1["ts"] <= first_ts + pd.Timedelta(minutes=30)]
    late = day1[day1["ts"] >= last_ts - pd.Timedelta(minutes=30)]

    early_avg = early["premium"].mean() if len(early) > 0 else entry_prem
    late_avg = late["premium"].mean() if len(late) > 0 else d1_last
    momentum = (late_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0

    return d1_eod_gain, momentum, d1_last


def compute_d1_underlying_confirms(df, entry_date, direction):
    """Check if underlying is confirming the trade direction at EOD day 1."""
    day1_mask = df["ts"].apply(
        lambda x: (pd.Timestamp(x).date() if not hasattr(x, "date") else x.date()) == entry_date
    )
    day1 = df[day1_mask]

    first_u = 0.0
    for i in range(min(5, len(day1))):
        u = day1["underlying_price"].iloc[i]
        if u and u > 0:
            first_u = float(u)
            break
    last_u = 0.0
    for i in range(len(day1) - 1, -1, -1):
        u = day1["underlying_price"].iloc[i]
        if u and u > 0:
            last_u = float(u)
            break

    if first_u <= 0 or last_u <= 0:
        return False
    u_move = (last_u - first_u) / first_u * 100
    is_call = direction in ("bullish", "call")
    return (is_call and u_move > 0.1) or (not is_call and u_move < -0.1)


def should_hold_overnight(ticker, d1_eod_gain, d1_momentum, u_confirms,
                          eod_min_strong=-20, eod_min_good=0, eod_min_cond=10,
                          mom_min=0, require_u_confirms_cond=False):
    """Decide if a multi-day trade should hold overnight.

    Returns (hold: bool, reason: str).
    """
    if ticker in ALWAYS_CUT:
        return False, f"always_cut ({ticker})"

    if ticker in STRONG_HOLD:
        if d1_eod_gain >= eod_min_strong:
            return True, f"strong_hold ({ticker}, eod={d1_eod_gain:+.0f}%)"
        return False, f"strong_hold_cut ({ticker}, eod={d1_eod_gain:+.0f}% < {eod_min_strong}%)"

    if ticker in GOOD_HOLD:
        if d1_eod_gain >= eod_min_good or d1_momentum >= mom_min:
            return True, f"good_hold ({ticker}, eod={d1_eod_gain:+.0f}%, mom={d1_momentum:+.0f}%)"
        return False, f"good_hold_cut ({ticker}, eod={d1_eod_gain:+.0f}%, mom={d1_momentum:+.0f}%)"

    # Conditional tickers — need stronger signal
    if d1_eod_gain >= eod_min_cond:
        if not require_u_confirms_cond or u_confirms:
            return True, f"conditional_hold ({ticker}, eod={d1_eod_gain:+.0f}%)"
    if d1_momentum >= max(mom_min, 10):  # higher momentum bar for conditional
        return True, f"conditional_hold_momentum ({ticker}, mom={d1_momentum:+.0f}%)"

    return False, f"conditional_cut ({ticker}, eod={d1_eod_gain:+.0f}%, mom={d1_momentum:+.0f}%)"


def run_sim(df, entry, contracts, ticker, direction, dte, expiry,
            eod_min_strong, eod_min_good, eod_min_cond, mom_min,
            require_u_confirms_cond, disable_soft_d1=True):
    """Run FSM with smart overnight hold logic."""
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_multiday = dte > 0

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    entry_date = entry_ts.date()

    first_u = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_u = float(u)
            break

    def make_state():
        return TradeState(
            trade_id=1, ticker=ticker, option_type=option_type,
            entry_premium=entry, entry_time=entry_ts, contracts=contracts,
            peak_premium=entry, entry_underlying_price=first_u,
            dte=dte, expiry_date=expiry or "",
        )

    # Day 1 config: disable soft trail so trades develop
    day1_cfg = cfg
    if is_multiday and disable_soft_d1:
        day1_cfg = replace(cfg, soft_trail_band_low_pct=999)

    # --- Production ---
    fsm_prod = ExitFSM(cfg, settings=_V6)
    state_prod = make_state()
    locked_prod = 0.0
    rem_prod = contracts
    prod_result = None

    # --- Smart hold variant ---
    is_day1 = True
    fsm_smart = ExitFSM(day1_cfg if is_multiday else cfg, settings=_V6)
    state_smart = make_state()
    locked_smart = 0.0
    rem_smart = contracts
    smart_result = None
    hold_decision = None
    held_overnight = False
    eod_checked = False

    # Pre-compute day 1 stats for hold decision
    d1_eod_gain, d1_momentum, _ = compute_d1_momentum(df, entry_date) if is_multiday else (0, 0, 0)
    u_confirms = compute_d1_underlying_confirms(df, entry_date, direction) if is_multiday else False

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

        current_date = now.date()
        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_min = now.minute
        mtc = max(0, (16 * 60) - (et_hour * 60 + et_min))

        # --- Production ---
        if prod_result is None:
            action = fsm_prod.evaluate(
                state_prod, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_prod:
                    locked_prod += (premium - entry) * action.contracts_to_close * 100
                    rem_prod -= action.contracts_to_close
                    state_prod.contracts = rem_prod
                else:
                    pnl = locked_prod + (premium - entry) * rem_prod * 100
                    peak_g = (state_prod.peak_premium - entry) / entry * 100
                    prod_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": "day1" if current_date == entry_date else "day2+",
                    }

        # --- Smart hold ---
        if smart_result is None:
            # EOD day 1 hold/cut decision (~3:45 PM ET)
            if (is_day1 and is_multiday and current_date == entry_date
                    and et_hour == 15 and et_min >= 45 and not eod_checked):
                eod_checked = True
                hold, hold_reason = should_hold_overnight(
                    ticker, d1_eod_gain, d1_momentum, u_confirms,
                    eod_min_strong=eod_min_strong, eod_min_good=eod_min_good,
                    eod_min_cond=eod_min_cond, mom_min=mom_min,
                    require_u_confirms_cond=require_u_confirms_cond,
                )
                hold_decision = hold_reason

                if not hold:
                    # Cut at EOD day 1
                    pnl = locked_smart + (premium - entry) * rem_smart * 100
                    peak_g = (state_smart.peak_premium - entry) / entry * 100
                    smart_result = {
                        "pnl": pnl, "reason": f"smart_cut:{hold_reason}",
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": "day1", "held_overnight": False,
                        "hold_decision": hold_reason,
                    }
                    if prod_result is not None:
                        break
                    continue

            # Day transition
            if is_day1 and current_date > entry_date and is_multiday:
                is_day1 = False
                held_overnight = True
                fsm_smart = ExitFSM(cfg, settings=_V6)

            action = fsm_smart.evaluate(
                state_smart, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_smart:
                    locked_smart += (premium - entry) * action.contracts_to_close * 100
                    rem_smart -= action.contracts_to_close
                    state_smart.contracts = rem_smart
                else:
                    pnl = locked_smart + (premium - entry) * rem_smart * 100
                    peak_g = (state_smart.peak_premium - entry) / entry * 100
                    smart_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": "day1" if current_date == entry_date else "day2+",
                        "held_overnight": held_overnight,
                        "hold_decision": hold_decision or "n/a",
                    }

        if prod_result is not None and smart_result is not None:
            break

    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                       "exit_prem": last_prem, "exit_day": "data_end"}
    if smart_result is None:
        pnl = locked_smart + (last_prem - entry) * rem_smart * 100
        peak_g = (state_smart.peak_premium - entry) / entry * 100
        smart_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                        "exit_prem": last_prem, "exit_day": "data_end",
                        "held_overnight": held_overnight,
                        "hold_decision": hold_decision or "n/a"}

    return prod_result, smart_result


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    trades = []
    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue
        df_ticks = load_ticks(harvester_conn, sig)
        if df_ticks is None:
            continue
        first_ask = df_ticks["ask"].iloc[0]
        first_mid = df_ticks["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]
        contracts = size_contracts(score, adj_entry)
        if contracts <= 0:
            continue
        blocked, _ = check_momentum_gate(df_ticks, direction)
        if blocked:
            trades.append(None)
            continue
        dte = sig.get("_dte", 0)
        expiry = sig.get("_expiry_date", "")
        trades.append({
            "df": df_ticks, "entry": adj_entry, "contracts": contracts,
            "ticker": ticker, "direction": direction, "dte": dte, "expiry": expiry,
            "score": score, "day": sig["created_at"][:10],
            "is_multiday": dte > 0,
        })
    harvester_conn.close()

    # Production baseline
    prod_results = []
    for t in trades:
        if t is None:
            prod_results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0,
                                 "exit_day": "n/a"})
            continue
        prod, _ = run_sim(t["df"], t["entry"], t["contracts"], t["ticker"],
                          t["direction"], t["dte"], t["expiry"],
                          eod_min_strong=999, eod_min_good=999, eod_min_cond=999,
                          mom_min=999, require_u_confirms_cond=False,
                          disable_soft_d1=False)
        prod["ticker"] = t["ticker"]
        prod["day"] = t["day"]
        prod["contracts"] = t["contracts"]
        prod["is_multiday"] = t["is_multiday"]
        prod_results.append(prod)

    prod_total = sum(r["pnl"] for r in prod_results)
    prod_md = sum(r["pnl"] for r in prod_results if r.get("is_multiday"))
    n_trades = len(prod_results)

    print(f"\n{'=' * 140}")
    print("SMART OVERNIGHT HOLD — Ticker-aware + momentum filter")
    print(f"{'=' * 140}")
    print(f"\nProduction: ${prod_total:,.0f} total | Multi-day: ${prod_md:,.0f}")
    print(f"Always cut overnight: {', '.join(sorted(ALWAYS_CUT))}")
    print(f"Strong hold (AMD): easy threshold | Good hold (TSLA, AAPL): moderate | Conditional: strict\n")

    # Sweep configs
    # (name, eod_min_strong, eod_min_good, eod_min_cond, mom_min, require_u_cond, disable_soft)
    configs = [
        # Group A: Just disable soft trail, ticker-filter only (no EOD threshold)
        ("Ticker filter only (no soft d1)",      -99, -99, -99, -99, False, True),

        # Group B: Ticker filter + EOD thresholds (disable soft trail d1)
        ("Strong:-20, Good:0, Cond:+10",        -20,   0,  10,   0, False, True),
        ("Strong:-20, Good:0, Cond:+20",        -20,   0,  20,   0, False, True),
        ("Strong:-20, Good:0, Cond:+30",        -20,   0,  30,   0, False, True),
        ("Strong:-10, Good:+5, Cond:+10",       -10,   5,  10,   0, False, True),
        ("Strong:-10, Good:+5, Cond:+15",       -10,   5,  15,   0, False, True),
        ("Strong:-10, Good:+5, Cond:+20",       -10,   5,  20,   0, False, True),
        ("Strong:-10, Good:+10, Cond:+20",      -10,  10,  20,   0, False, True),
        ("Strong:0, Good:+10, Cond:+20",          0,  10,  20,   0, False, True),
        ("Strong:0, Good:+10, Cond:+30",          0,  10,  30,   0, False, True),

        # Group C: Add momentum requirement
        ("S:-20 G:0 C:+10 mom>0",               -20,   0,  10,   0, False, True),
        ("S:-20 G:0 C:+10 mom>10",              -20,   0,  10,  10, False, True),
        ("S:-20 G:0 C:+10 mom>20",              -20,   0,  10,  20, False, True),
        ("S:-10 G:+5 C:+15 mom>10",             -10,   5,  15,  10, False, True),
        ("S:-10 G:+5 C:+15 mom>20",             -10,   5,  15,  20, False, True),

        # Group D: Require underlying confirms for conditional
        ("S:-20 G:0 C:+10 u_confirm",           -20,   0,  10,   0, True,  True),
        ("S:-10 G:+5 C:+15 u_confirm",          -10,   5,  15,   0, True,  True),

        # Group E: Keep soft trail active (only hold via ticker filter)
        ("Ticker filter (keep soft trail)",      -20,   0,  10,   0, False, False),
        ("S:-10 G:+5 C:+15 (keep soft)",        -10,   5,  15,   0, False, False),

        # Group F: Very conservative — only hold AMD + TSLA
        ("Only AMD+TSLA overnight",              -20, -20,  999, 999, False, True),
        ("Only AMD+TSLA+AAPL overnight",         -20, -20,  999, 999, False, True),

        # Group G: Aggressive — hold everything except ALWAYS_CUT
        ("Hold all except META/MSTR/NVDA/GOOGL", -99, -99, -99, -99, False, True),
    ]

    header = (
        f"{'Strategy':<45} {'Total':>10} {'vs Prod':>10} {'MD P&L':>10} "
        f"{'vs MD':>10} {'WR':>6} {'Held':>5} {'Cut':>5} {'Sharpe':>7}"
    )
    print(header)
    print("-" * len(header))

    all_results = {}

    for name, eod_s, eod_g, eod_c, mom, u_req, no_soft in configs:
        results = []
        for t in trades:
            if t is None:
                results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0,
                                "exit_day": "n/a", "is_multiday": False,
                                "held_overnight": False, "hold_decision": "n/a"})
                continue
            _, smart = run_sim(
                t["df"], t["entry"], t["contracts"], t["ticker"],
                t["direction"], t["dte"], t["expiry"],
                eod_min_strong=eod_s, eod_min_good=eod_g, eod_min_cond=eod_c,
                mom_min=mom, require_u_confirms_cond=u_req,
                disable_soft_d1=no_soft,
            )
            smart["ticker"] = t["ticker"]
            smart["day"] = t["day"]
            smart["contracts"] = t["contracts"]
            smart["is_multiday"] = t["is_multiday"]
            results.append(smart)

        total = sum(r["pnl"] for r in results)
        md_pnl = sum(r["pnl"] for r in results if r.get("is_multiday"))
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / n_trades * 100
        diff = total - prod_total
        diff_md = md_pnl - prod_md
        held = sum(1 for r in results if r.get("held_overnight"))
        cut = sum(1 for r in results if "smart_cut" in str(r.get("reason", "")))

        daily = {}
        for r in results:
            d = r.get("day", "?")
            daily[d] = daily.get(d, 0) + r["pnl"]
        daily_s = pd.Series(list(daily.values()))
        sharpe = daily_s.mean() / daily_s.std() if daily_s.std() > 0 else 0

        marker = " ***" if diff > 0 else ""
        print(
            f"{name:<45} ${total:>9,.0f} ${diff:>+9,.0f} ${md_pnl:>9,.0f} "
            f"${diff_md:>+9,.0f} {wr:>5.1f}% {held:>5} {cut:>5} "
            f"{sharpe:>7.2f}{marker}"
        )

        all_results[name] = results

    # Winners
    winner_names = [name for name, _, _, _, _, _, _ in configs
                    if sum(r["pnl"] for r in all_results[name]) > prod_total]

    print(f"\n{'=' * 140}")
    if winner_names:
        print(f"WINNERS: {len(winner_names)} configs beat production!")
    else:
        print("No configs beat production. Showing best configs detail.")
        # Pick top 3 by total
        sorted_names = sorted(all_results.keys(),
                              key=lambda n: sum(r["pnl"] for r in all_results[n]), reverse=True)
        winner_names = sorted_names[:3]
    print(f"{'=' * 140}")

    for detail_name in winner_names[:5]:
        results = all_results[detail_name]
        total = sum(r["pnl"] for r in results)
        diff = total - prod_total

        diffs = []
        for p, r, t in zip(prod_results, results, trades):
            if t is None or not t["is_multiday"]:
                continue
            d = r["pnl"] - p["pnl"]
            if abs(d) > 1:
                diffs.append((p, r, t, d))

        if not diffs:
            continue

        print(f"\n--- {detail_name} (${diff:>+,.0f} vs prod) ---\n")

        print(f"{'Day':<12} {'Ticker':>6} {'Ctrs':>5} "
              f"{'Prod P&L':>10} {'Smart P&L':>10} {'Diff':>10} "
              f"{'Prod Exit':>18} {'Smart Exit':>18} {'HeldON':>6} {'Decision':>30}")
        print("-" * 155)

        for p, r, t, d in sorted(diffs, key=lambda x: x[3], reverse=True):
            held = "YES" if r.get("held_overnight") else "no"
            decision = str(r.get("hold_decision", ""))[:30]
            print(
                f"{t['day']:<12} {t['ticker']:>6} {t['contracts']:>5} "
                f"${p['pnl']:>9,.0f} ${r['pnl']:>9,.0f} ${d:>+9,.0f} "
                f"{p['reason']:>18} {str(r['reason'])[:18]:>18} "
                f"{held:>6} {decision:>30}"
            )

        better = [x for x in diffs if x[3] > 0]
        worse = [x for x in diffs if x[3] < 0]
        print(f"\n  Changed: {len(diffs)} trades")
        print(f"  Helped: {len(better)} (${sum(x[3] for x in better):>+,.0f})")
        print(f"  Hurt:   {len(worse)} (${sum(x[3] for x in worse):>+,.0f})")
        print(f"  Net:    ${sum(x[3] for x in diffs):>+,.0f}")

        held_trades = [(p, r) for p, r, t, d in diffs if r.get("held_overnight")]
        if held_trades:
            h_better = sum(1 for p, r in held_trades if r["pnl"] > p["pnl"])
            h_total_diff = sum(r["pnl"] - p["pnl"] for p, r in held_trades)
            print(f"\n  Held overnight: {len(held_trades)} trades, "
                  f"{h_better} improved ({h_better/len(held_trades)*100:.0f}%), "
                  f"net ${h_total_diff:>+,.0f}")


if __name__ == "__main__":
    main()
