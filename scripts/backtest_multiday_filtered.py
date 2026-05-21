"""Backtest: Filtered multi-day patience — hold overnight UNLESS too deep underwater.

Key idea: disable soft trail on day 1 so trades can develop, BUT add an EOD
day-1 filter: if the trade is down more than X% at 3:45 PM, cut it.
Only trades that are still reasonably healthy hold overnight.

Sweep parameters:
  - EOD day 1 max loss: what's the max acceptable loss to hold overnight?
    (e.g., -30% = cut anything down >30% at EOD, hold the rest)
  - EOD day 1 min gain: alternatively, only hold if you're UP at EOD
  - Which gates to disable on day 1: soft trail, theta, stops

Usage:
    python scripts/backtest_multiday_filtered.py
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


def run_sim(df, entry, contracts, ticker, direction, dte, expiry,
            eod_d1_max_loss_pct=None, disable_soft_d1=False,
            disable_theta_d1=False, disable_stops_d1=False,
            disable_adaptive_d1=False, eod_d1_min_gain_pct=None):
    """Run FSM with filtered multi-day patience.

    On day 1 for multi-day trades:
      - Optionally disable soft trail, theta, stops, adaptive trail
      - At EOD day 1 (~3:45 PM), check if trade should hold overnight:
        * If eod_d1_max_loss_pct set: cut if loss exceeds this %
        * If eod_d1_min_gain_pct set: cut if gain is below this %
      - Day 2+: all gates return to normal

    Returns (prod_result, filtered_result) dicts.
    """
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_multiday = dte > 0

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

    # Build day-1 patient config
    day1_cfg = cfg
    if is_multiday:
        kw = {}
        if disable_stops_d1:
            kw["backstop_multiday_pct"] = 100
            kw["tight_stop_multiday_pct"] = 100
        if disable_soft_d1:
            kw["soft_trail_band_low_pct"] = 999  # effectively disable
        if disable_theta_d1:
            kw["theta_timer_minutes"] = 0
        if disable_adaptive_d1:
            no_trail = (AdaptiveTier(0, 999),)
            kw["adaptive_highvol_tiers"] = no_trail
            kw["adaptive_index_tiers"] = no_trail
            kw["adaptive_standard_tiers"] = no_trail
        if kw:
            day1_cfg = replace(cfg, **kw)

    # --- Production (baseline) ---
    fsm_prod = ExitFSM(cfg, settings=_V6)
    state_prod = make_state()
    locked_prod = 0.0
    rem_prod = contracts
    prod_result = None

    # --- Filtered patient variant ---
    is_day1 = True
    fsm_filt = ExitFSM(day1_cfg if is_multiday else cfg, settings=_V6)
    state_filt = make_state()
    locked_filt = 0.0
    rem_filt = contracts
    filt_result = None
    held_overnight = False
    eod_cut = False

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

        # --- Filtered patient variant ---
        if filt_result is None:
            # Day transition: switch from patient day1 config to normal
            if is_day1 and current_date > entry_date and is_multiday:
                is_day1 = False
                held_overnight = True
                fsm_filt = ExitFSM(cfg, settings=_V6)

            # EOD day 1 filter: check at ~3:45 PM ET (15 min before close)
            if (is_day1 and is_multiday and current_date == entry_date
                    and et_hour == 15 and et_min >= 45
                    and not eod_cut):
                gain_pct = (premium - entry) / entry * 100

                should_cut = False
                if eod_d1_max_loss_pct is not None and gain_pct < -eod_d1_max_loss_pct:
                    should_cut = True
                if eod_d1_min_gain_pct is not None and gain_pct < eod_d1_min_gain_pct:
                    should_cut = True

                if should_cut:
                    eod_cut = True
                    pnl = locked_filt + (premium - entry) * rem_filt * 100
                    peak_g = (state_filt.peak_premium - entry) / entry * 100
                    filt_result = {
                        "pnl": pnl, "reason": "eod_d1_filter_cut",
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": "day1", "held_overnight": False,
                    }
                    if prod_result is not None:
                        break
                    continue

            action = fsm_filt.evaluate(
                state_filt, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_filt:
                    locked_filt += (premium - entry) * action.contracts_to_close * 100
                    rem_filt -= action.contracts_to_close
                    state_filt.contracts = rem_filt
                else:
                    pnl = locked_filt + (premium - entry) * rem_filt * 100
                    peak_g = (state_filt.peak_premium - entry) / entry * 100
                    filt_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                        "exit_day": "day1" if current_date == entry_date else "day2+",
                        "held_overnight": held_overnight,
                    }

        if prod_result is not None and filt_result is not None:
            break

    # EOD close
    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                       "exit_prem": last_prem, "exit_day": "data_end"}
    if filt_result is None:
        pnl = locked_filt + (last_prem - entry) * rem_filt * 100
        peak_g = (state_filt.peak_premium - entry) / entry * 100
        filt_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g,
                       "exit_prem": last_prem, "exit_day": "data_end",
                       "held_overnight": held_overnight}

    return prod_result, filt_result


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

    n_total = len(trades)
    n_blocked = sum(1 for t in trades if t is None)
    n_md = sum(1 for t in trades if t and t["is_multiday"])

    # Production baseline
    prod_results = []
    for t in trades:
        if t is None:
            prod_results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0,
                                 "exit_day": "n/a", "is_multiday": False})
            continue
        prod, _ = run_sim(t["df"], t["entry"], t["contracts"], t["ticker"],
                          t["direction"], t["dte"], t["expiry"])
        prod["ticker"] = t["ticker"]
        prod["day"] = t["day"]
        prod["contracts"] = t["contracts"]
        prod["is_multiday"] = t["is_multiday"]
        prod_results.append(prod)

    prod_total = sum(r["pnl"] for r in prod_results)
    prod_md = sum(r["pnl"] for r in prod_results if r.get("is_multiday"))

    print(f"\n{'=' * 140}")
    print("FILTERED MULTI-DAY PATIENCE — Hold overnight unless too deep underwater at EOD")
    print(f"{'=' * 140}")
    print(f"\nTrades: {n_total} total, {n_blocked} blocked, {n_md} multi-day")
    print(f"Production: ${prod_total:,.0f} total | Multi-day: ${prod_md:,.0f}\n")

    # Configs to test
    # (name, eod_max_loss, eod_min_gain, no_soft, no_theta, no_stops, no_adaptive)
    configs = [
        # --- Group A: Disable soft trail day 1 + EOD loss filter ---
        # "Let trades develop on day 1, cut deep losers at EOD, hold rest overnight"
        ("A: no soft d1, cut >60% loss EOD",    60, None, True, False, False, False),
        ("A: no soft d1, cut >50% loss EOD",    50, None, True, False, False, False),
        ("A: no soft d1, cut >40% loss EOD",    40, None, True, False, False, False),
        ("A: no soft d1, cut >30% loss EOD",    30, None, True, False, False, False),
        ("A: no soft d1, cut >20% loss EOD",    20, None, True, False, False, False),
        ("A: no soft d1, cut >10% loss EOD",    10, None, True, False, False, False),
        ("A: no soft d1, cut any loss EOD",      0, None, True, False, False, False),

        # --- Group B: Same + also disable theta/stops day 1 ---
        ("B: no soft/theta/stop, >60% cut",     60, None, True, True, True, False),
        ("B: no soft/theta/stop, >50% cut",     50, None, True, True, True, False),
        ("B: no soft/theta/stop, >40% cut",     40, None, True, True, True, False),
        ("B: no soft/theta/stop, >30% cut",     30, None, True, True, True, False),
        ("B: no soft/theta/stop, >20% cut",     20, None, True, True, True, False),
        ("B: no soft/theta/stop, >10% cut",     10, None, True, True, True, False),
        ("B: no soft/theta/stop, any loss cut",  0, None, True, True, True, False),

        # --- Group C: Full patience + EOD filter ---
        ("C: full patience, >50% cut",          50, None, True, True, True, True),
        ("C: full patience, >40% cut",          40, None, True, True, True, True),
        ("C: full patience, >30% cut",          30, None, True, True, True, True),
        ("C: full patience, >20% cut",          20, None, True, True, True, True),
        ("C: full patience, any loss cut",       0, None, True, True, True, True),

        # --- Group D: Only hold overnight if UP at EOD ---
        ("D: no soft d1, must be UP at EOD",    None, 0, True, False, False, False),
        ("D: no soft d1, must be +5% at EOD",   None, 5, True, False, False, False),
        ("D: no soft d1, must be +10% at EOD",  None, 10, True, False, False, False),
        ("D: no soft d1, must be +15% at EOD",  None, 15, True, False, False, False),
        ("D: no soft d1, must be +20% at EOD",  None, 20, True, False, False, False),

        # --- Group E: Full patience, must be up to hold ---
        ("E: full patience, must be UP EOD",    None, 0, True, True, True, True),
        ("E: full patience, must be +10% EOD",  None, 10, True, True, True, True),
        ("E: full patience, must be +20% EOD",  None, 20, True, True, True, True),
    ]

    header = (
        f"{'Strategy':<42} {'Total P&L':>10} {'vs Prod':>10} {'MD P&L':>10} "
        f"{'vs MD':>10} {'WR':>6} {'Held ON':>8} {'EOD Cut':>8} {'Sharpe':>7}"
    )
    print(header)
    print("-" * len(header))

    best_configs = []

    for name, max_loss, min_gain, no_soft, no_theta, no_stops, no_adapt in configs:
        results = []
        for t in trades:
            if t is None:
                results.append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0,
                                "exit_day": "n/a", "is_multiday": False,
                                "held_overnight": False})
                continue
            _, filt = run_sim(
                t["df"], t["entry"], t["contracts"], t["ticker"],
                t["direction"], t["dte"], t["expiry"],
                eod_d1_max_loss_pct=max_loss, eod_d1_min_gain_pct=min_gain,
                disable_soft_d1=no_soft, disable_theta_d1=no_theta,
                disable_stops_d1=no_stops, disable_adaptive_d1=no_adapt,
            )
            filt["ticker"] = t["ticker"]
            filt["day"] = t["day"]
            filt["contracts"] = t["contracts"]
            filt["is_multiday"] = t["is_multiday"]
            results.append(filt)

        total = sum(r["pnl"] for r in results)
        md_pnl = sum(r["pnl"] for r in results if r.get("is_multiday"))
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100
        diff_total = total - prod_total
        diff_md = md_pnl - prod_md
        held_on = sum(1 for r in results if r.get("held_overnight"))
        eod_cut_count = sum(1 for r in results if r.get("reason") == "eod_d1_filter_cut")

        daily = {}
        for r in results:
            d = r.get("day", "?")
            daily[d] = daily.get(d, 0) + r["pnl"]
        daily_s = pd.Series(list(daily.values()))
        sharpe = daily_s.mean() / daily_s.std() if daily_s.std() > 0 else 0

        print(
            f"{name:<42} ${total:>9,.0f} ${diff_total:>+9,.0f} ${md_pnl:>9,.0f} "
            f"${diff_md:>+9,.0f} {wr:>5.1f}% {held_on:>8} {eod_cut_count:>8} "
            f"{sharpe:>7.2f}"
        )

        best_configs.append({
            "name": name, "total": total, "diff": diff_total, "wr": wr,
            "sharpe": sharpe, "held_on": held_on, "eod_cut": eod_cut_count,
            "results": results,
        })

    # Winners
    winners = [c for c in best_configs if c["diff"] > 0]
    winners.sort(key=lambda x: x["diff"], reverse=True)

    print(f"\n{'=' * 140}")
    if winners:
        print(f"WINNERS — {len(winners)} configs beat production!")
    else:
        print("No configs beat production overall. Showing closest:")
        winners = sorted(best_configs, key=lambda x: x["diff"], reverse=True)[:5]
    print(f"{'=' * 140}\n")

    for c in winners[:10]:
        print(f"  {c['name']:<42} ${c['total']:>9,.0f} (${c['diff']:>+,.0f}) "
              f"WR={c['wr']:.1f}% Sharpe={c['sharpe']:.2f} "
              f"held_overnight={c['held_on']} eod_cut={c['eod_cut']}")

    # Detail on the best config
    best = winners[0]
    print(f"\n{'=' * 140}")
    print(f"DETAIL: {best['name']}")
    print(f"{'=' * 140}\n")

    results = best["results"]
    print(f"{'Day':<12} {'Ticker':>6} {'Ctrs':>5} {'DTE':>4} "
          f"{'Prod P&L':>10} {'Filt P&L':>10} {'Diff':>10} "
          f"{'Prod Exit':>18} {'Filt Exit':>18} {'P Day':>7} {'F Day':>7} {'HeldON':>6}")
    print("-" * 140)

    diffs = []
    for p, r, t in zip(prod_results, results, trades):
        if t is None or not t["is_multiday"]:
            continue
        d = r["pnl"] - p["pnl"]
        if abs(d) > 1:
            diffs.append((p, r, t, d))

    for p, r, t, d in sorted(diffs, key=lambda x: x[3], reverse=True):
        held = "YES" if r.get("held_overnight") else "no"
        print(
            f"{t['day']:<12} {t['ticker']:>6} {t['contracts']:>5} {t['dte']:>4} "
            f"${p['pnl']:>9,.0f} ${r['pnl']:>9,.0f} ${d:>+9,.0f} "
            f"{p['reason']:>18} {r['reason']:>18} "
            f"{p.get('exit_day','?'):>7} {r.get('exit_day','?'):>7} {held:>6}"
        )

    if diffs:
        better = [x for x in diffs if x[3] > 0]
        worse = [x for x in diffs if x[3] < 0]
        print(f"\n  Changed: {len(diffs)} multi-day trades")
        print(f"  Helped: {len(better)} (${sum(x[3] for x in better):>+,.0f})")
        print(f"  Hurt:   {len(worse)} (${sum(x[3] for x in worse):>+,.0f})")
        print(f"  Net:    ${sum(x[3] for x in diffs):>+,.0f}")

        # Win rate on held-overnight trades
        held_trades = [(p, r) for p, r, t, d in diffs if r.get("held_overnight")]
        if held_trades:
            held_wins = sum(1 for p, r in held_trades if r["pnl"] > p["pnl"])
            print(f"\n  Held overnight: {len(held_trades)} trades, "
                  f"{held_wins} improved ({held_wins/len(held_trades)*100:.0f}%)")


if __name__ == "__main__":
    main()
