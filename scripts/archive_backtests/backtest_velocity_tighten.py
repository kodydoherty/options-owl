"""Backtest: Velocity Tighten strategy for runner protection.

Instead of selling contracts early (ladder), detect FAST reversals using
premium drop speed and exit before giving back all gains.

Key insight from data: 68% of runner reversals lose half their peak gain
within 10 minutes. Consolidations (healthy pullbacks) are slower.

Strategy: when a trade has peaked +50%+ and drops 40% of its gain within
8 minutes → exit immediately. Otherwise, let the normal 55% trail handle it.
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

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

_V6 = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True, V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333, V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True, V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15,
)

# Import shared helpers
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from backtest_ladder_report import (
    check_momentum_gate,
    load_signals,
    load_ticks,
    size_contracts,
)


def run_sim(df, entry, contracts, ticker, direction, dte, expiry,
            velocity_drop_pct=40, velocity_window_min=8, velocity_peak_threshold=50):
    """Run production FSM with optional velocity tighten overlay.

    Returns (prod_result, vt_result) dicts.
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

    # Production
    fsm_prod = ExitFSM(cfg, settings=_V6)
    state_prod = make_state()
    locked_prod = 0.0
    rem_prod = contracts
    prod_result = None

    # Velocity Tighten
    fsm_vt = ExitFSM(cfg, settings=_V6)
    state_vt = make_state()
    locked_vt = 0.0
    rem_vt = contracts
    vt_result = None
    vt_peak_prem = entry
    vt_peak_time = entry_ts
    vt_above_thresh = False

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

        # --- Velocity Tighten ---
        if vt_result is None:
            # Track VT peak
            if premium > vt_peak_prem:
                vt_peak_prem = premium
                vt_peak_time = now

            vt_peak_gain = (vt_peak_prem - entry) / entry * 100
            if vt_peak_gain >= velocity_peak_threshold:
                vt_above_thresh = True

            # Check velocity reversal
            if vt_above_thresh and premium < vt_peak_prem and (vt_peak_prem - entry) > 0:
                giveback = (vt_peak_prem - premium) / (vt_peak_prem - entry) * 100
                mins_since_peak = (now - vt_peak_time).total_seconds() / 60

                if giveback >= velocity_drop_pct and 0 < mins_since_peak <= velocity_window_min:
                    pnl = locked_vt + (premium - entry) * rem_vt * 100
                    vt_result = {
                        "pnl": pnl, "reason": "velocity_exit",
                        "peak_gain": vt_peak_gain, "exit_prem": premium,
                    }
                    if prod_result is not None:
                        break
                    continue

            action = fsm_vt.evaluate(
                state_vt, premium, bid, ask, now,
                current_underlying=underlying, minutes_to_close=float(mtc),
            )
            if action.should_exit:
                if action.contracts_to_close > 0 and action.contracts_to_close < rem_vt:
                    locked_vt += (premium - entry) * action.contracts_to_close * 100
                    rem_vt -= action.contracts_to_close
                    state_vt.contracts = rem_vt
                else:
                    pnl = locked_vt + (premium - entry) * rem_vt * 100
                    peak_g = (state_vt.peak_premium - entry) / entry * 100
                    vt_result = {
                        "pnl": pnl, "reason": action.reason.value,
                        "peak_gain": peak_g, "exit_prem": premium,
                    }

        if prod_result is not None and vt_result is not None:
            break

    # EOD close
    last_prem = df["premium"].iloc[-1]
    if prod_result is None:
        pnl = locked_prod + (last_prem - entry) * rem_prod * 100
        peak_g = (state_prod.peak_premium - entry) / entry * 100
        prod_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": peak_g, "exit_prem": last_prem}
    if vt_result is None:
        pnl = locked_vt + (last_prem - entry) * rem_vt * 100
        vt_result = {"pnl": pnl, "reason": "eod_data_end", "peak_gain": vt_peak_gain, "exit_prem": last_prem}

    return prod_result, vt_result


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    configs = [
        # (name, drop_pct, window_min, peak_threshold)
        ("Production (baseline)", 999, 0, 999),  # never triggers
        ("VT: 40% drop in 8min, +50% peak", 40, 8, 50),
        ("VT: 50% drop in 10min, +50% peak", 50, 10, 50),
        ("VT: 40% drop in 5min, +50% peak", 40, 5, 50),
        ("VT: 30% drop in 8min, +50% peak", 30, 8, 50),
        ("VT: 40% drop in 8min, +35% peak", 40, 8, 35),
    ]

    all_results = {name: [] for name, _, _, _ in configs}

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
            all_results[configs[0][0]].append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0, "ticker": ticker, "day": sig["created_at"][:10]})
            for name, _, _, _ in configs[1:]:
                all_results[name].append({"pnl": 0, "reason": "momentum_blocked", "peak_gain": 0, "ticker": ticker, "day": sig["created_at"][:10]})
            continue

        dte = sig.get("_dte", 0)
        expiry = sig.get("_expiry_date", "")

        for name, drop_pct, window, peak_thresh in configs:
            _, vt_res = run_sim(
                df_ticks, adj_entry, contracts, ticker, direction, dte, expiry,
                velocity_drop_pct=drop_pct, velocity_window_min=window,
                velocity_peak_threshold=peak_thresh,
            )
            vt_res["ticker"] = ticker
            vt_res["day"] = sig["created_at"][:10]
            vt_res["contracts"] = contracts
            vt_res["entry"] = adj_entry
            all_results[name].append(vt_res)

    harvester_conn.close()

    # === Results ===
    print(f"\n{'=' * 100}")
    print("VELOCITY TIGHTEN SWEEP — Runner Reversal Detection")
    print(f"{'=' * 100}\n")

    header = f"{'Strategy':<40} {'Total P&L':>10} {'WR':>6} {'Trades':>6} {'Daily Std':>10} {'Sharpe':>7} {'Runner P&L':>11} {'VelExits':>8}"
    print(header)
    print("-" * len(header))

    for name, _, _, _ in configs:
        results = all_results[name]
        rdf = pd.DataFrame(results)
        total = rdf["pnl"].sum()
        wins = (rdf["pnl"] > 0).sum()
        wr = wins / len(rdf) * 100
        daily = rdf.groupby("day")["pnl"].sum()
        dstd = daily.std()
        sharpe = daily.mean() / dstd if dstd > 0 else 0
        runners = rdf[rdf["peak_gain"] >= 50]
        runner_pnl = runners["pnl"].sum()
        vel_exits = (rdf["reason"] == "velocity_exit").sum()

        print(f"{name:<40} ${total:>9,.0f} {wr:>5.1f}% {len(rdf):>6} ${dstd:>9,.0f} {sharpe:>7.2f} ${runner_pnl:>10,.0f} {vel_exits:>8}")

    # Detail on best VT config velocity exits
    # Show detail for each config that has velocity exits
    for cfg_idx in range(1, len(configs)):
        name = configs[cfg_idx][0]
        vt_results_cfg = all_results[name]
        vel_count = sum(1 for r in vt_results_cfg if r["reason"] == "velocity_exit")
        if vel_count == 0:
            continue

        print(f"\n{'=' * 100}")
        print(f"VELOCITY EXIT DETAIL — {name}")
        print(f"{'=' * 100}\n")

        prod_results_ref = all_results[configs[0][0]]
        for prod, vt in zip(prod_results_ref, vt_results_cfg):
            if vt["reason"] != "velocity_exit":
                continue
            diff = vt["pnl"] - prod["pnl"]
            print(
                f"  {vt['day']} {vt['ticker']:>5} peak=+{vt['peak_gain']:>5.0f}%  "
                f"VT=${vt['pnl']:>8,.0f}  Prod=${prod['pnl']:>8,.0f}  "
                f"diff=${diff:>+8,.0f}  prod_reason={prod['reason']}"
            )

        vel_trades = [(p, v) for p, v in zip(prod_results_ref, vt_results_cfg) if v["reason"] == "velocity_exit"]
        if vel_trades:
            prod_sum = sum(p["pnl"] for p, _ in vel_trades)
            vt_sum = sum(v["pnl"] for _, v in vel_trades)
            print(f"\n  Velocity exits:  {len(vel_trades)} trades")
            print(f"  Prod P&L on these: ${prod_sum:,.0f}")
            print(f"  VT P&L on these:   ${vt_sum:,.0f}")
            print(f"  Net savings:       ${vt_sum - prod_sum:+,.0f}")

    print(f"\n{'=' * 100}")
    print("VELOCITY EXIT DETAIL — 40% drop in 8min, +50% peak (original)")
    print(f"{'=' * 100}\n")

    prod_results = all_results[configs[0][0]]
    vt_results = all_results[configs[1][0]]

    for prod, vt in zip(prod_results, vt_results):
        if vt["reason"] != "velocity_exit":
            continue
        diff = vt["pnl"] - prod["pnl"]
        print(
            f"  {vt['day']} {vt['ticker']:>5} peak=+{vt['peak_gain']:>5.0f}%  "
            f"VT=${vt['pnl']:>8,.0f}  Prod=${prod['pnl']:>8,.0f}  "
            f"diff=${diff:>+8,.0f}  prod_reason={prod['reason']}"
        )

    # Net impact
    vel_trades = [(p, v) for p, v in zip(prod_results, vt_results) if v["reason"] == "velocity_exit"]
    if vel_trades:
        prod_sum = sum(p["pnl"] for p, _ in vel_trades)
        vt_sum = sum(v["pnl"] for _, v in vel_trades)
        print(f"\n  Velocity exits:  {len(vel_trades)} trades")
        print(f"  Prod P&L on these: ${prod_sum:,.0f}")
        print(f"  VT P&L on these:   ${vt_sum:,.0f}")
        print(f"  Net savings:       ${vt_sum - prod_sum:+,.0f}")


if __name__ == "__main__":
    main()
