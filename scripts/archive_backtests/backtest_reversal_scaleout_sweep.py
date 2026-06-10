"""Reversal Scaleout Sweep — partial profit lock when underlying reverses.

Instead of tightening the trail (which causes premature exits), LOCK partial
profit by selling a fraction of contracts when the underlying reverses from
its peak. The remaining contracts ride with the normal wide trail.

This is like a second V6 scaleout but triggered by underlying reversal instead
of a fixed gain threshold.

Sweep parameters:
  - Peak gain threshold: 40-150% (when to arm the reversal scaleout)
  - Underlying reversal: 0.10-0.50% (how much underlying must reverse from peak)
  - Scaleout fraction: 0.25, 0.33, 0.50 (how many contracts to sell)
  - Min contracts: 2-3 (don't scaleout below this)

Usage:
    python scripts/backtest_reversal_scaleout_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
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
            rev_peak_pct=999, rev_underlying_pct=999, rev_fraction=0.333,
            rev_min_contracts=2):
    """Run production FSM with reversal scaleout overlay.

    Reversal scaleout: when peak_gain >= rev_peak_pct AND underlying reverses
    from its peak by >= rev_underlying_pct, sell rev_fraction of remaining
    contracts to lock profit. One-shot (only fires once).

    Returns (prod_result, rev_result) dicts.
    """
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

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

    # --- Reversal scaleout variant ---
    fsm_rev = ExitFSM(cfg, settings=_V6)
    state_rev = make_state()
    locked_rev = 0.0
    rem_rev = contracts
    rev_result = None
    rev_fired = False  # one-shot
    rev_armed = False  # peak gain threshold reached
    u_peak = first_u   # best underlying in trade direction

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

        # --- Reversal scaleout ---
        if rev_result is None:
            # Track underlying peak
            if underlying > 0:
                if is_call and underlying > u_peak:
                    u_peak = underlying
                elif not is_call and (u_peak == 0 or underlying < u_peak):
                    u_peak = underlying

            # Check if armed (peak gain threshold reached)
            rev_peak_gain = (state_rev.peak_premium - entry) / entry * 100
            if not rev_armed and rev_peak_gain >= rev_peak_pct:
                rev_armed = True

            # Check reversal scaleout trigger
            if (rev_armed and not rev_fired and u_peak > 0 and underlying > 0
                    and rem_rev >= rev_min_contracts):
                u_reversal = abs(underlying - u_peak) / u_peak * 100
                u_reversed = (is_call and underlying < u_peak) or (not is_call and underlying > u_peak)

                if u_reversed and u_reversal >= rev_underlying_pct:
                    rev_fired = True
                    close_qty = max(1, int(rem_rev * rev_fraction))
                    if close_qty < rem_rev:  # don't close everything
                        locked_rev += (premium - entry) * close_qty * 100
                        rem_rev -= close_qty
                        state_rev.contracts = rem_rev

            # Normal FSM evaluation
            action = fsm_rev.evaluate(
                state_rev, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_rev:
                    locked_rev += (premium - entry) * action.contracts_to_close * 100
                    rem_rev -= action.contracts_to_close
                    state_rev.contracts = rem_rev
                else:
                    pnl = locked_rev + (premium - entry) * rem_rev * 100
                    peak_g = (state_rev.peak_premium - entry) / entry * 100
                    rev_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                        "rev_fired": rev_fired,
                    }

        if prod_result is not None and rev_result is not None:
            break

    # EOD close
    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g, "exit_prem": last_prem}
    if rev_result is None:
        rev_peak_gain = (state_rev.peak_premium - entry) / entry * 100
        pnl = locked_rev + (last_prem - entry) * rem_rev * 100
        rev_result = {
            "pnl": pnl, "reason": "eod_data_end", "peak_gain": rev_peak_gain,
            "exit_prem": last_prem, "rev_fired": rev_fired,
        }

    return prod_result, rev_result


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-process all trades
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
        })
    harvester_conn.close()

    # Production baseline
    prod_results = []
    for t in trades:
        if t is None:
            prod_results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0})
            continue
        prod, _ = run_sim(
            t["df"], t["entry"], t["contracts"], t["ticker"], t["direction"],
            t["dte"], t["expiry"], rev_peak_pct=999,
        )
        prod["ticker"] = t["ticker"]
        prod["day"] = t["day"]
        prod["contracts"] = t["contracts"]
        prod_results.append(prod)

    prod_total = sum(r["pnl"] for r in prod_results)
    prod_wins = sum(1 for r in prod_results if r["pnl"] > 0)
    n_trades = len(prod_results)
    prod_daily = pd.DataFrame(prod_results).groupby(
        [r.get("day", "?") for r in prod_results]
    )["pnl"].sum()
    prod_std = prod_daily.std()
    prod_sharpe = prod_daily.mean() / prod_std if prod_std > 0 else 0

    print(f"\n{'=' * 130}")
    print("REVERSAL SCALEOUT SWEEP — Lock partial profit when underlying reverses")
    print(f"{'=' * 130}")
    print(f"\nProduction baseline: ${prod_total:,.0f} | {prod_wins}/{n_trades} wins "
          f"({prod_wins/n_trades*100:.1f}% WR) | Sharpe {prod_sharpe:.2f}")
    print(f"Trades with data: {sum(1 for t in trades if t is not None)} | "
          f"Momentum blocked: {sum(1 for t in trades if t is None)}\n")

    # Sweep grid
    peak_thresholds = [40, 50, 60, 70, 80, 100, 120, 150]
    underlying_reversals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    fractions = [0.25, 0.333, 0.50]
    min_contracts_list = [2, 3]

    header = (
        f"{'Peak%':>6} {'URev%':>6} {'Frac':>5} {'MinC':>4} | "
        f"{'Total P&L':>10} {'vs Prod':>10} {'WR':>6} {'Sharpe':>7} {'Fired':>6} {'Changed':>8} | "
        f"{'Fired P&L':>10} {'Prod on F':>10} {'Net on F':>10}"
    )
    print(header)
    print("-" * len(header))

    all_configs = []

    for frac in fractions:
        for min_c in min_contracts_list:
            for peak_pct in peak_thresholds:
                for u_rev in underlying_reversals:
                    rev_results = []
                    for t in trades:
                        if t is None:
                            rev_results.append({
                                "pnl": 0, "reason": "momentum_blocked",
                                "peak_gain": 0, "rev_fired": False,
                            })
                            continue
                        _, rev = run_sim(
                            t["df"], t["entry"], t["contracts"], t["ticker"],
                            t["direction"], t["dte"], t["expiry"],
                            rev_peak_pct=peak_pct, rev_underlying_pct=u_rev,
                            rev_fraction=frac, rev_min_contracts=min_c,
                        )
                        rev["ticker"] = t["ticker"]
                        rev["day"] = t["day"]
                        rev_results.append(rev)

                    total = sum(r["pnl"] for r in rev_results)
                    wins = sum(1 for r in rev_results if r["pnl"] > 0)
                    wr = wins / n_trades * 100
                    diff = total - prod_total
                    fired = sum(1 for r in rev_results if r.get("rev_fired"))
                    changed = sum(
                        1 for p, s in zip(prod_results, rev_results)
                        if abs(p["pnl"] - s["pnl"]) > 1.0
                    )

                    # Daily stats
                    rev_daily = {}
                    for r in rev_results:
                        d = r.get("day", "?")
                        rev_daily[d] = rev_daily.get(d, 0) + r["pnl"]
                    daily_vals = pd.Series(list(rev_daily.values()))
                    dstd = daily_vals.std()
                    sharpe = daily_vals.mean() / dstd if dstd > 0 else 0

                    # Fired trades detail
                    fired_rev_pnl = sum(
                        s["pnl"] for s in rev_results if s.get("rev_fired")
                    )
                    fired_prod_pnl = sum(
                        p["pnl"] for p, s in zip(prod_results, rev_results)
                        if s.get("rev_fired")
                    )
                    fired_net = fired_rev_pnl - fired_prod_pnl

                    print(
                        f"{peak_pct:>5}% {u_rev:>5.2f}% {frac:>5.2f} {min_c:>4} | "
                        f"${total:>9,.0f} ${diff:>+9,.0f} {wr:>5.1f}% {sharpe:>7.2f} "
                        f"{fired:>6} {changed:>8} | "
                        f"${fired_rev_pnl:>9,.0f} ${fired_prod_pnl:>9,.0f} ${fired_net:>+9,.0f}"
                    )

                    all_configs.append({
                        "peak": peak_pct, "underlying": u_rev, "frac": frac,
                        "min_c": min_c, "total": total, "diff": diff, "wr": wr,
                        "sharpe": sharpe, "fired": fired, "changed": changed,
                        "fired_net": fired_net,
                    })

            print()  # blank between min_contracts

        print("=" * 40)  # separator between fractions

    # Winners only
    winners = [c for c in all_configs if c["diff"] > 0]
    winners.sort(key=lambda x: x["diff"], reverse=True)

    print(f"\n{'=' * 130}")
    print("CONFIGS THAT BEAT PRODUCTION (sorted by improvement)")
    print(f"{'=' * 130}\n")

    if winners:
        print(f"{'#':>3} {'Peak%':>6} {'URev%':>6} {'Frac':>5} {'MinC':>4} | "
              f"{'Total':>10} {'vs Prod':>10} {'WR':>6} {'Sharpe':>7} {'Fired':>6} {'Changed':>8}")
        print("-" * 80)
        for i, c in enumerate(winners[:25]):
            print(
                f"{i+1:>3} {c['peak']:>5}% {c['underlying']:>5.2f}% {c['frac']:>5.2f} {c['min_c']:>4} | "
                f"${c['total']:>9,.0f} ${c['diff']:>+9,.0f} {c['wr']:>5.1f}% {c['sharpe']:>7.2f} "
                f"{c['fired']:>6} {c['changed']:>8}"
            )
    else:
        print("  No configs beat production.")

    # Best by Sharpe (stability)
    all_configs.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'=' * 130}")
    print("TOP 15 BY SHARPE RATIO (consistency)")
    print(f"{'=' * 130}\n")
    print(f"{'#':>3} {'Peak%':>6} {'URev%':>6} {'Frac':>5} {'MinC':>4} | "
          f"{'Total':>10} {'vs Prod':>10} {'WR':>6} {'Sharpe':>7} {'Fired':>6} {'Changed':>8}")
    print("-" * 80)
    for i, c in enumerate(all_configs[:15]):
        marker = " *" if c["diff"] > 0 else ""
        print(
            f"{i+1:>3} {c['peak']:>5}% {c['underlying']:>5.02f}% {c['frac']:>5.2f} {c['min_c']:>4} | "
            f"${c['total']:>9,.0f} ${c['diff']:>+9,.0f} {c['wr']:>5.1f}% {c['sharpe']:>7.2f} "
            f"{c['fired']:>6} {c['changed']:>8}{marker}"
        )

    # Detail on best winner
    best = winners[0] if winners else max(all_configs, key=lambda x: x["diff"])
    print(f"\n{'=' * 130}")
    label = "BEST WINNER" if winners else "LEAST WORST"
    print(f"DETAIL: {label} — peak={best['peak']}%, rev={best['underlying']:.2f}%, "
          f"frac={best['frac']:.2f}, min_c={best['min_c']}")
    print(f"{'=' * 130}\n")

    # Re-run to get per-trade detail
    detail_results = []
    for t in trades:
        if t is None:
            detail_results.append({
                "pnl": 0, "reason": "momentum_blocked",
                "peak_gain": 0, "rev_fired": False,
            })
            continue
        _, rev = run_sim(
            t["df"], t["entry"], t["contracts"], t["ticker"],
            t["direction"], t["dte"], t["expiry"],
            rev_peak_pct=best["peak"], rev_underlying_pct=best["underlying"],
            rev_fraction=best["frac"], rev_min_contracts=best["min_c"],
        )
        rev["ticker"] = t["ticker"]
        rev["day"] = t["day"]
        detail_results.append(rev)

    print(f"{'Day':<12} {'Ticker':>6} {'Ctrs':>5} {'Peak':>7} {'Prod P&L':>10} "
          f"{'Rev P&L':>10} {'Diff':>10} {'Prod Exit':>18} {'Rev Exit':>18}")
    print("-" * 115)
    for p, s, t in zip(prod_results, detail_results, trades):
        if not s.get("rev_fired"):
            continue
        d = s["pnl"] - p["pnl"]
        day = t["day"] if t else "?"
        tk = t["ticker"] if t else "?"
        ct = t["contracts"] if t else 0
        print(
            f"{day:<12} {tk:>6} {ct:>5} +{s['peak_gain']:>5.0f}% "
            f"${p['pnl']:>9,.0f} ${s['pnl']:>9,.0f} ${d:>+9,.0f} "
            f"{p['reason']:>18} {s['reason']:>18}"
        )

    fired_trades = [(p, s) for p, s in zip(prod_results, detail_results) if s.get("rev_fired")]
    if fired_trades:
        p_sum = sum(p["pnl"] for p, _ in fired_trades)
        s_sum = sum(s["pnl"] for _, s in fired_trades)
        better = [(p, s) for p, s in fired_trades if s["pnl"] > p["pnl"]]
        worse = [(p, s) for p, s in fired_trades if s["pnl"] < p["pnl"]]
        same = [(p, s) for p, s in fired_trades if abs(s["pnl"] - p["pnl"]) < 1]
        print(f"\n  Trades where reversal scaleout fired: {len(fired_trades)}")
        print(f"  Production P&L on these: ${p_sum:>,.0f}")
        print(f"  Reversal scaleout P&L:   ${s_sum:>,.0f}")
        print(f"  Net improvement:         ${s_sum - p_sum:>+,.0f}")
        print(f"\n  Helped: {len(better)} trades (${sum(s['pnl']-p['pnl'] for p,s in better):>+,.0f})")
        print(f"  Hurt:   {len(worse)} trades (${sum(s['pnl']-p['pnl'] for p,s in worse):>+,.0f})")
        print(f"  Same:   {len(same)} trades")


if __name__ == "__main__":
    main()
