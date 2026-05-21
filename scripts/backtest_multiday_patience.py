"""Backtest: Multi-day patience — wider stops on day 1, normal on day 2+.

Adam's insight: 14 real trades that were down 40-60% on day 1 but recovered on
day 2 and made money. The idea: for multi-day trades (DTE > 0), don't cut
losses aggressively on day 1. Let them hold overnight for the day 2 recovery.

What changes on day 1:
  - Backstop widened or disabled (currently 75% for multi-day)
  - Tight stop widened or disabled (currently 52% for multi-day)
  - Theta timer disabled on day 1 (currently 180min/15%)
  - Soft trail disabled on day 1 (prevents locking small gains that reverse)
  - Adaptive trail disabled on day 1 (prevents trailing out before day 2)

Day 2+: all gates return to normal production settings.

Usage:
    python scripts/backtest_multiday_patience.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
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


def run_sim(df, entry, contracts, ticker, direction, dte, expiry,
            day1_backstop=75, day1_tight=52, day1_disable_theta=False,
            day1_disable_soft_trail=False, day1_disable_adaptive=False,
            day1_disable_scaleout=False):
    """Run FSM with day-1 patience overrides for multi-day trades.

    Returns (prod_result, patient_result) dicts.
    """
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    entry_date = entry_ts.date() if hasattr(entry_ts, "date") else pd.Timestamp(entry_ts).date()

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

    # Build day-1 patient config (widened stops)
    day1_cfg = replace(cfg,
        backstop_multiday_pct=day1_backstop,
        tight_stop_multiday_pct=day1_tight,
    )
    if day1_disable_theta:
        day1_cfg = replace(day1_cfg, theta_timer_minutes=0)
    if day1_disable_soft_trail:
        day1_cfg = replace(day1_cfg, soft_trail_band_low_pct=999)  # effectively disable
    if day1_disable_adaptive:
        # Set adaptive tiers to impossibly wide trails
        from options_owl.risk.exit_v5.config import AdaptiveTier
        no_trail = (AdaptiveTier(0, 999),)
        day1_cfg = replace(day1_cfg,
            adaptive_highvol_tiers=no_trail,
            adaptive_index_tiers=no_trail,
            adaptive_standard_tiers=no_trail,
        )

    day1_v6 = _V6
    if day1_disable_scaleout:
        day1_v6 = SimpleNamespace(**{k: getattr(_V6, k) for k in dir(_V6) if not k.startswith("_")})
        day1_v6.ENABLE_V6_SCALEOUT = False

    # --- Production (baseline) ---
    fsm_prod = ExitFSM(cfg, settings=_V6)
    state_prod = make_state()
    locked_prod = 0.0
    rem_prod = contracts
    prod_result = None

    # --- Patient variant ---
    # Start with day1 config; will switch to normal on day 2
    is_day1 = True
    fsm_patient = ExitFSM(day1_cfg if dte > 0 else cfg, settings=day1_v6 if dte > 0 else _V6)
    state_patient = make_state()
    locked_patient = 0.0
    rem_patient = contracts
    patient_result = None
    patient_exit_day = None

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

        current_date = now.date() if hasattr(now, "date") else pd.Timestamp(now).date()

        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        mtc = max(0, (16 * 60) - (et_hour * 60 + now.minute))

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

        # --- Patient variant ---
        if patient_result is None:
            # Switch to normal config on day 2+
            if is_day1 and current_date > entry_date and dte > 0:
                is_day1 = False
                fsm_patient = ExitFSM(cfg, settings=_V6)
                # Carry forward state (peak_premium, etc are on state_patient)

            action = fsm_patient.evaluate(
                state_patient, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_patient:
                    locked_patient += (premium - entry) * action.contracts_to_close * 100
                    rem_patient -= action.contracts_to_close
                    state_patient.contracts = rem_patient
                else:
                    pnl = locked_patient + (premium - entry) * rem_patient * 100
                    peak_g = (state_patient.peak_premium - entry) / entry * 100
                    patient_exit_day = "day1" if current_date == entry_date else "day2+"
                    patient_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": patient_exit_day,
                    }

        if prod_result is not None and patient_result is not None:
            break

    # EOD close
    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                       "exit_prem": last_prem, "exit_day": "data_end"}
    if patient_result is None:
        pnl = locked_patient + (last_prem - entry) * rem_patient * 100
        peak_g = (state_patient.peak_premium - entry) / entry * 100
        patient_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                          "exit_prem": last_prem, "exit_day": "data_end"}

    return prod_result, patient_result


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-process trades
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

        # Check if has day 2 data
        first_day = df_ticks["ts"].iloc[0]
        if hasattr(first_day, "date"):
            first_day = first_day.date()
        else:
            first_day = pd.Timestamp(first_day).date()
        last_day = df_ticks["ts"].iloc[-1]
        if hasattr(last_day, "date"):
            last_day = last_day.date()
        else:
            last_day = pd.Timestamp(last_day).date()

        trades.append({
            "df": df_ticks, "entry": adj_entry, "contracts": contracts,
            "ticker": ticker, "direction": direction, "dte": dte, "expiry": expiry,
            "score": score, "day": sig["created_at"][:10],
            "has_day2": (last_day > first_day) if dte > 0 else False,
            "is_multiday": dte > 0,
        })
    harvester_conn.close()

    # Stats
    n_total = len(trades)
    n_blocked = sum(1 for t in trades if t is None)
    n_active = n_total - n_blocked
    n_multiday = sum(1 for t in trades if t and t["is_multiday"])
    n_multiday_d2 = sum(1 for t in trades if t and t["has_day2"])
    n_0dte = sum(1 for t in trades if t and not t["is_multiday"])

    print(f"\n{'=' * 130}")
    print("MULTI-DAY PATIENCE SWEEP — Wider stops on day 1 for multi-day trades")
    print(f"{'=' * 130}")
    print(f"\nTrades: {n_total} total, {n_blocked} blocked, {n_active} active")
    print(f"  0DTE: {n_0dte} | Multi-day: {n_multiday} ({n_multiday_d2} with day 2 data)")

    # Production baseline
    prod_results = []
    for t in trades:
        if t is None:
            prod_results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0, "exit_day": "n/a"})
            continue
        prod, _ = run_sim(
            t["df"], t["entry"], t["contracts"], t["ticker"], t["direction"],
            t["dte"], t["expiry"],
        )
        prod["ticker"] = t["ticker"]
        prod["day"] = t["day"]
        prod["contracts"] = t["contracts"]
        prod["is_multiday"] = t["is_multiday"]
        prod_results.append(prod)

    prod_total = sum(r["pnl"] for r in prod_results)
    prod_md = sum(r["pnl"] for r in prod_results if r.get("is_multiday"))
    prod_0d = sum(r["pnl"] for r in prod_results if not r.get("is_multiday") and r["reason"] != "momentum_blocked")

    print(f"\nProduction baseline: ${prod_total:,.0f} total")
    print(f"  0DTE: ${prod_0d:,.0f} | Multi-day: ${prod_md:,.0f}")

    # Count prod day-1 exits on multi-day trades
    prod_md_d1_exits = sum(1 for r in prod_results if r.get("is_multiday") and r.get("exit_day") == "day1")
    prod_md_d2_exits = sum(1 for r in prod_results if r.get("is_multiday") and r.get("exit_day") in ("day2+", "data_end"))
    print(f"  Multi-day prod exits: {prod_md_d1_exits} on day 1, {prod_md_d2_exits} on day 2+\n")

    # Show prod day-1 exits detail
    print(f"Production multi-day trades that EXIT on day 1 (these are what patience could save):")
    print(f"{'Day':<12} {'Ticker':>6} {'Ctrs':>5} {'DTE':>4} {'P&L':>10} {'Peak':>7} {'Reason':>20}")
    print("-" * 75)
    for r, t in zip(prod_results, trades):
        if t is None or not t["is_multiday"]:
            continue
        if r.get("exit_day") != "day1":
            continue
        print(f"{r.get('day','?'):<12} {r.get('ticker','?'):>6} {r.get('contracts',0):>5} "
              f"{t['dte']:>4} ${r['pnl']:>9,.0f} +{r['peak_gain']:>5.0f}% {r['reason']:>20}")

    # Sweep configurations
    configs = [
        # (name, backstop, tight, disable_theta, disable_soft, disable_adaptive, disable_scaleout)
        ("Production (baseline)",            75,  52, False, False, False, False),
        # Widen backstop only
        ("Backstop 85%",                     85,  52, False, False, False, False),
        ("Backstop 90%",                     90,  52, False, False, False, False),
        ("Backstop 95%",                     95,  52, False, False, False, False),
        ("Backstop 100% (no backstop d1)",  100,  52, False, False, False, False),
        # Widen both stops
        ("Both 85%/65%",                     85,  65, False, False, False, False),
        ("Both 90%/75%",                     90,  75, False, False, False, False),
        ("Both 95%/85%",                     95,  85, False, False, False, False),
        ("Both 100%/100% (no stops d1)",    100, 100, False, False, False, False),
        # No stops + disable theta
        ("No stops + no theta d1",          100, 100, True,  False, False, False),
        # No stops + no theta + no soft trail
        ("No stops/theta/soft d1",          100, 100, True,  True,  False, False),
        # Full patience: disable everything on day 1
        ("FULL PATIENCE d1 (stops only)",   100, 100, True,  True,  True,  False),
        ("FULL PATIENCE d1 (+ no scaleout)",100, 100, True,  True,  True,  True),
        # Moderate patience: widen but don't disable
        ("Moderate: 90%/70% + no theta",     90,  70, True,  False, False, False),
        ("Moderate: 90%/70% + no soft/theta",90,  70, True,  True,  False, False),
        # Just disable trailing on day 1
        ("Normal stops, no trail d1",        75,  52, False, False, True,  False),
        ("Normal stops, no soft+trail d1",   75,  52, False, True,  True,  False),
    ]

    print(f"\n{'=' * 130}")
    print("SWEEP RESULTS")
    print(f"{'=' * 130}\n")

    header = (f"{'Strategy':<40} {'Total P&L':>10} {'vs Prod':>10} {'MD P&L':>10} "
              f"{'vs MD Prod':>10} {'WR':>6} {'MD D1 Exit':>10} {'MD D2 Exit':>10} "
              f"{'Sharpe':>7}")
    print(header)
    print("-" * len(header))

    all_config_results = {}

    for name, backstop, tight, no_theta, no_soft, no_adapt, no_scale in configs:
        results = []
        for t in trades:
            if t is None:
                results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0,
                                "exit_day": "n/a", "is_multiday": False})
                continue
            _, patient = run_sim(
                t["df"], t["entry"], t["contracts"], t["ticker"], t["direction"],
                t["dte"], t["expiry"],
                day1_backstop=backstop, day1_tight=tight,
                day1_disable_theta=no_theta, day1_disable_soft_trail=no_soft,
                day1_disable_adaptive=no_adapt, day1_disable_scaleout=no_scale,
            )
            patient["ticker"] = t["ticker"]
            patient["day"] = t["day"]
            patient["contracts"] = t["contracts"]
            patient["is_multiday"] = t["is_multiday"]
            results.append(patient)

        total = sum(r["pnl"] for r in results)
        md_pnl = sum(r["pnl"] for r in results if r.get("is_multiday"))
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100
        diff_total = total - prod_total
        diff_md = md_pnl - prod_md

        md_d1 = sum(1 for r in results if r.get("is_multiday") and r.get("exit_day") == "day1")
        md_d2 = sum(1 for r in results if r.get("is_multiday") and r.get("exit_day") in ("day2+", "data_end"))

        daily = {}
        for r in results:
            d = r.get("day", "?")
            daily[d] = daily.get(d, 0) + r["pnl"]
        daily_s = pd.Series(list(daily.values()))
        sharpe = daily_s.mean() / daily_s.std() if daily_s.std() > 0 else 0

        print(
            f"{name:<40} ${total:>9,.0f} ${diff_total:>+9,.0f} ${md_pnl:>9,.0f} "
            f"${diff_md:>+9,.0f} {wr:>5.1f}% {md_d1:>10} {md_d2:>10} "
            f"{sharpe:>7.2f}"
        )

        all_config_results[name] = results

    # Detail on best configs
    for detail_name in ["FULL PATIENCE d1 (stops only)", "No stops/theta/soft d1",
                        "Both 100%/100% (no stops d1)", "Moderate: 90%/70% + no theta"]:
        if detail_name not in all_config_results:
            continue
        results = all_config_results[detail_name]

        # Only show multi-day trades where outcome differs
        diffs = []
        for p, r, t in zip(prod_results, results, trades):
            if t is None or not t["is_multiday"]:
                continue
            d = r["pnl"] - p["pnl"]
            if abs(d) > 1:
                diffs.append((p, r, t, d))

        if not diffs:
            continue

        print(f"\n{'=' * 130}")
        print(f"DETAIL: {detail_name} — multi-day trades that changed")
        print(f"{'=' * 130}\n")

        print(f"{'Day':<12} {'Ticker':>6} {'Ctrs':>5} {'DTE':>4} "
              f"{'Prod P&L':>10} {'Patient':>10} {'Diff':>10} "
              f"{'Prod Exit':>18} {'Pat Exit':>18} {'Prod Day':>10} {'Pat Day':>10}")
        print("-" * 140)

        for p, r, t, d in sorted(diffs, key=lambda x: x[3], reverse=True):
            print(
                f"{t['day']:<12} {t['ticker']:>6} {t['contracts']:>5} {t['dte']:>4} "
                f"${p['pnl']:>9,.0f} ${r['pnl']:>9,.0f} ${d:>+9,.0f} "
                f"{p['reason']:>18} {r['reason']:>18} "
                f"{p.get('exit_day','?'):>10} {r.get('exit_day','?'):>10}"
            )

        better = [x for x in diffs if x[3] > 0]
        worse = [x for x in diffs if x[3] < 0]
        print(f"\n  Changed: {len(diffs)} trades")
        print(f"  Helped: {len(better)} (${sum(x[3] for x in better):>+,.0f})")
        print(f"  Hurt:   {len(worse)} (${sum(x[3] for x in worse):>+,.0f})")
        print(f"  Net:    ${sum(x[3] for x in diffs):>+,.0f}")


if __name__ == "__main__":
    main()
