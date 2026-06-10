"""Smart Trail Parameter Sweep — find the sweet spot.

The broad smart trail (+50% peak, 0.15% underlying) showed +$8,110 but fires
on 50% of trades. The narrow version (+100%, 0.30%) fires 3 times and loses.

This sweep tests the FULL grid in between to find where the improvement is
actually coming from and the optimal activation point.

Smart trail logic:
    When peak_gain >= THRESHOLD and underlying moves AGAINST by >= X%,
    tighten the adaptive trail width by a FACTOR (e.g., 0.55 = 55% of normal).

Usage:
    python scripts/backtest_smart_trail_sweep.py
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

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    TickerCategory,
    categorize_ticker,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

# Import shared helpers
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
            smart_peak_pct=999, smart_underlying_pct=999, smart_tighten_factor=0.55):
    """Run production FSM with smart trail overlay.

    Smart trail: when peak_gain >= smart_peak_pct AND underlying is against
    by >= smart_underlying_pct, multiply all adaptive trail widths by
    smart_tighten_factor (< 1.0 = tighter).

    Returns (prod_result, smart_result) dicts.
    """
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

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

    # --- Production (baseline) ---
    fsm_prod = ExitFSM(cfg, settings=_V6)
    state_prod = make_state()
    locked_prod = 0.0
    rem_prod = contracts
    prod_result = None

    # --- Smart trail variant ---
    fsm_smart = ExitFSM(cfg, settings=_V6)
    state_smart = make_state()
    locked_smart = 0.0
    rem_smart = contracts
    smart_result = None
    smart_triggered = False  # track if smart trail ever fired
    # Track underlying peak (best price in trade direction)
    u_peak = first_u  # best underlying price seen

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
                    }

        # --- Smart trail ---
        if smart_result is None:
            # Track underlying peak (best price in trade direction)
            if underlying > 0:
                is_call = option_type in ("call", "bullish")
                if is_call and underlying > u_peak:
                    u_peak = underlying
                elif not is_call and underlying < u_peak:
                    u_peak = underlying

            # Check if smart trail should activate:
            # peak gain high enough AND underlying REVERSING from its peak
            smart_peak_gain = (state_smart.peak_premium - entry) / entry * 100
            should_tighten = False

            if smart_peak_gain >= smart_peak_pct and u_peak > 0 and underlying > 0:
                # Measure underlying reversal from its peak, not from entry
                u_reversal = abs(underlying - u_peak) / u_peak * 100
                if is_call and underlying < u_peak and u_reversal >= smart_underlying_pct:
                    should_tighten = True
                elif not is_call and underlying > u_peak and u_reversal >= smart_underlying_pct:
                    should_tighten = True

            # If smart trail conditions met, use tightened config
            if should_tighten and not smart_triggered:
                smart_triggered = True
                # Create tightened config — multiply all trail widths
                category = categorize_ticker(ticker)
                base_tiers = cfg.get_adaptive_tiers(category)
                tight_tiers = tuple(
                    AdaptiveTier(t.min_peak_gain, t.trail_width * smart_tighten_factor)
                    for t in base_tiers
                )
                kw = {}
                if category == TickerCategory.HIGH_VOL:
                    kw["adaptive_highvol_tiers"] = tight_tiers
                elif category == TickerCategory.INDEX:
                    kw["adaptive_index_tiers"] = tight_tiers
                else:
                    kw["adaptive_standard_tiers"] = tight_tiers
                tight_cfg = replace(cfg, **kw)
                fsm_smart = ExitFSM(tight_cfg, settings=_V6)

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
                        "smart_fired": smart_triggered,
                    }

        if prod_result is not None and smart_result is not None:
            break

    # EOD close
    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g, "exit_prem": last_prem}
    if smart_result is None:
        smart_peak_gain = (state_smart.peak_premium - entry) / entry * 100
        pnl = locked_smart + (last_prem - entry) * rem_smart * 100
        smart_result = {
            "pnl": pnl, "reason": "eod_data_end", "peak_gain": smart_peak_gain,
            "exit_prem": last_prem, "smart_fired": smart_triggered,
        }

    return prod_result, smart_result


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-process all trades once (shared across all configs)
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
            trades.append(None)  # placeholder for momentum-blocked
            continue
        dte = sig.get("_dte", 0)
        expiry = sig.get("_expiry_date", "")
        trades.append({
            "df": df_ticks, "entry": adj_entry, "contracts": contracts,
            "ticker": ticker, "direction": direction, "dte": dte, "expiry": expiry,
            "score": score, "day": sig["created_at"][:10],
        })
    harvester_conn.close()

    # Get production baseline once
    prod_results = []
    for t in trades:
        if t is None:
            prod_results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0})
            continue
        prod, _ = run_sim(
            t["df"], t["entry"], t["contracts"], t["ticker"], t["direction"],
            t["dte"], t["expiry"], smart_peak_pct=999, smart_underlying_pct=999,
        )
        prod["ticker"] = t["ticker"]
        prod["day"] = t["day"]
        prod["contracts"] = t["contracts"]
        prod_results.append(prod)

    prod_total = sum(r["pnl"] for r in prod_results)
    prod_wins = sum(1 for r in prod_results if r["pnl"] > 0)
    n_trades = len(prod_results)

    print(f"\n{'=' * 120}")
    print("SMART TRAIL PARAMETER SWEEP")
    print(f"{'=' * 120}")
    print(f"\nProduction baseline: ${prod_total:,.0f} | {prod_wins}/{n_trades} wins ({prod_wins/n_trades*100:.1f}% WR)")
    print(f"Trades with data: {sum(1 for t in trades if t is not None)} | Momentum blocked: {sum(1 for t in trades if t is None)}\n")

    # Sweep grid
    peak_thresholds = [40, 50, 60, 70, 80, 90, 100, 120, 150]
    underlying_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    tighten_factors = [0.40, 0.55, 0.70]  # 40%, 55%, 70% of normal width

    header = (
        f"{'Peak%':>6} {'UndAg%':>7} {'Tight':>6} | "
        f"{'Total P&L':>10} {'vs Prod':>10} {'WR':>6} {'Fired':>6} {'Changed':>8} | "
        f"{'Fired P&L':>10} {'Prod on F':>10} {'Net on F':>10}"
    )
    print(header)
    print("-" * len(header))

    best_configs = []

    for tighten in tighten_factors:
        for peak_pct in peak_thresholds:
            for u_pct in underlying_thresholds:
                smart_results = []
                for t in trades:
                    if t is None:
                        smart_results.append({
                            "pnl": 0, "reason": "momentum_blocked",
                            "peak_gain": 0, "smart_fired": False,
                        })
                        continue
                    _, smart = run_sim(
                        t["df"], t["entry"], t["contracts"], t["ticker"],
                        t["direction"], t["dte"], t["expiry"],
                        smart_peak_pct=peak_pct, smart_underlying_pct=u_pct,
                        smart_tighten_factor=tighten,
                    )
                    smart["ticker"] = t["ticker"]
                    smart["day"] = t["day"]
                    smart_results.append(smart)

                total = sum(r["pnl"] for r in smart_results)
                wins = sum(1 for r in smart_results if r["pnl"] > 0)
                wr = wins / n_trades * 100
                diff = total - prod_total
                fired = sum(1 for r in smart_results if r.get("smart_fired"))

                # Count trades where outcome actually changed
                changed = sum(
                    1 for p, s in zip(prod_results, smart_results)
                    if abs(p["pnl"] - s["pnl"]) > 1.0
                )

                # P&L on just the fired trades
                fired_smart_pnl = sum(
                    s["pnl"] for s in smart_results if s.get("smart_fired")
                )
                fired_prod_pnl = sum(
                    p["pnl"] for p, s in zip(prod_results, smart_results)
                    if s.get("smart_fired")
                )
                fired_net = fired_smart_pnl - fired_prod_pnl

                print(
                    f"{peak_pct:>5}% {u_pct:>6.2f}% {tighten:>5.2f} | "
                    f"${total:>9,.0f} ${diff:>+9,.0f} {wr:>5.1f}% {fired:>6} {changed:>8} | "
                    f"${fired_smart_pnl:>9,.0f} ${fired_prod_pnl:>9,.0f} ${fired_net:>+9,.0f}"
                )

                best_configs.append({
                    "peak": peak_pct, "underlying": u_pct, "tighten": tighten,
                    "total": total, "diff": diff, "wr": wr,
                    "fired": fired, "changed": changed,
                    "fired_net": fired_net,
                })

        print()  # blank line between tighten factors

    # Top 10 configs by total P&L
    best_configs.sort(key=lambda x: x["total"], reverse=True)
    print(f"\n{'=' * 120}")
    print("TOP 15 CONFIGS BY TOTAL P&L")
    print(f"{'=' * 120}\n")
    print(f"{'#':>3} {'Peak%':>6} {'UndAg%':>7} {'Tight':>6} | {'Total':>10} {'vs Prod':>10} {'WR':>6} {'Fired':>6} {'Changed':>8} {'Fired Net':>10}")
    print("-" * 90)
    for i, c in enumerate(best_configs[:15]):
        print(
            f"{i+1:>3} {c['peak']:>5}% {c['underlying']:>6.2f}% {c['tighten']:>5.2f} | "
            f"${c['total']:>9,.0f} ${c['diff']:>+9,.0f} {c['wr']:>5.1f}% "
            f"{c['fired']:>6} {c['changed']:>8} ${c['fired_net']:>+9,.0f}"
        )

    # Best configs that fire on 5-30 trades (sweet spot)
    sweet = [c for c in best_configs if 5 <= c["fired"] <= 30 and c["diff"] > 0]
    sweet.sort(key=lambda x: x["diff"], reverse=True)
    print(f"\n{'=' * 120}")
    print("SWEET SPOT: Configs that fire 5-30 trades AND improve P&L")
    print(f"{'=' * 120}\n")
    if sweet:
        print(f"{'#':>3} {'Peak%':>6} {'UndAg%':>7} {'Tight':>6} | {'Total':>10} {'vs Prod':>10} {'WR':>6} {'Fired':>6} {'Changed':>8} {'Fired Net':>10}")
        print("-" * 90)
        for i, c in enumerate(sweet[:15]):
            print(
                f"{i+1:>3} {c['peak']:>5}% {c['underlying']:>6.2f}% {c['tighten']:>5.2f} | "
                f"${c['total']:>9,.0f} ${c['diff']:>+9,.0f} {c['wr']:>5.1f}% "
                f"{c['fired']:>6} {c['changed']:>8} ${c['fired_net']:>+9,.0f}"
            )
    else:
        print("  No configs in the sweet spot (5-30 fires AND positive improvement)")

    # Detail on top sweet spot config
    if sweet:
        best = sweet[0]
        print(f"\n{'=' * 120}")
        print(f"DETAIL: Best sweet-spot config — peak={best['peak']}%, underlying={best['underlying']:.2f}%, tighten={best['tighten']:.2f}")
        print(f"{'=' * 120}\n")

        # Re-run best config to get per-trade detail
        detail_results = []
        for t in trades:
            if t is None:
                detail_results.append({
                    "pnl": 0, "reason": "momentum_blocked",
                    "peak_gain": 0, "smart_fired": False,
                })
                continue
            _, smart = run_sim(
                t["df"], t["entry"], t["contracts"], t["ticker"],
                t["direction"], t["dte"], t["expiry"],
                smart_peak_pct=best["peak"], smart_underlying_pct=best["underlying"],
                smart_tighten_factor=best["tighten"],
            )
            smart["ticker"] = t["ticker"]
            smart["day"] = t["day"]
            detail_results.append(smart)

        print(f"{'Day':<12} {'Ticker':>6} {'Peak':>7} {'Prod P&L':>10} {'Smart P&L':>10} {'Diff':>10} {'Prod Exit':>18} {'Smart Exit':>18}")
        print("-" * 105)
        for p, s, t in zip(prod_results, detail_results, trades):
            if not s.get("smart_fired"):
                continue
            d = s["pnl"] - p["pnl"]
            day = t["day"] if t else "?"
            tk = t["ticker"] if t else "?"
            print(
                f"{day:<12} {tk:>6} +{s['peak_gain']:>5.0f}% "
                f"${p['pnl']:>9,.0f} ${s['pnl']:>9,.0f} ${d:>+9,.0f} "
                f"{p['reason']:>18} {s['reason']:>18}"
            )

        fired_trades = [(p, s) for p, s in zip(prod_results, detail_results) if s.get("smart_fired")]
        if fired_trades:
            p_sum = sum(p["pnl"] for p, _ in fired_trades)
            s_sum = sum(s["pnl"] for _, s in fired_trades)
            wins_smart = sum(1 for _, s in fired_trades if s["pnl"] > 0)
            wins_prod = sum(1 for p, _ in fired_trades if p["pnl"] > 0)
            print(f"\n  Trades where smart trail fired: {len(fired_trades)}")
            print(f"  Production P&L on these: ${p_sum:>,.0f} ({wins_prod} wins)")
            print(f"  Smart trail P&L on these: ${s_sum:>,.0f} ({wins_smart} wins)")
            print(f"  Net improvement: ${s_sum - p_sum:>+,.0f}")

            # Wins vs losses breakdown
            better = [(p, s) for p, s in fired_trades if s["pnl"] > p["pnl"]]
            worse = [(p, s) for p, s in fired_trades if s["pnl"] < p["pnl"]]
            same = [(p, s) for p, s in fired_trades if abs(s["pnl"] - p["pnl"]) < 1]
            print(f"\n  Smart trail helped: {len(better)} trades (${sum(s['pnl']-p['pnl'] for p,s in better):>+,.0f})")
            print(f"  Smart trail hurt:   {len(worse)} trades (${sum(s['pnl']-p['pnl'] for p,s in worse):>+,.0f})")
            print(f"  No change:          {len(same)} trades")


if __name__ == "__main__":
    main()
