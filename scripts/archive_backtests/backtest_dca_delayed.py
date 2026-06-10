"""Backtest DCA with DELAYED windows — add contracts after the initial chaos settles.

Instead of averaging down in the first 3-5 minutes (which just doubles losers),
test adding contracts after 5-15 minutes when the trade dips but the underlying
thesis is still intact.

Strategies:
  POST_GRACE_DIP   — Add after 5min grace, if dipped 15-30%, underlying OK
  MID_TRADE_DIP    — Add between 8-20min, if dipped 15-35%, underlying OK
  RECOVERY_ADD     — Add after 10min if dipped 20%+ then bounced back to -10%, underlying OK
  LATE_CONVICTION  — Add after 15min if still down 10-25% but underlying confirms direction

Usage:
    python scripts/backtest_dca_delayed.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 78
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
            signal["_expiry_date"] = expiry
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


def compute_contracts(entry_premium, score):
    max_risk_pct = 0.75
    max_concurrent = 5
    max_position_pct = 0.15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent
    position_cap = PORTFOLIO * max_position_pct
    if score >= 95:
        score_mult = 1.0
    elif score >= 90:
        score_mult = 0.75
    elif score >= 85:
        score_mult = 0.50
    else:
        score_mult = 0.25
    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, pos_cap, 20))


def simulate_delayed_dca(df, entry_premium, initial_contracts, direction, dte,
                         expiry_date, ticker, dca_mode="NONE"):
    """Run FSM with delayed DCA — add contracts AFTER initial chaos settles."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False, "dca_price": 0,
                "dca_elapsed_min": 0, "avg_entry": entry_premium,
                "total_contracts": initial_contracts,
                "total_cost": entry_premium * initial_contracts * 100}

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    first_underlying = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_underlying = float(u)
            break

    dca_fired = False
    dca_price = 0.0
    dca_elapsed = 0.0
    total_contracts = initial_contracts
    total_cost = entry_premium * initial_contracts * 100
    avg_entry = entry_premium

    # Track trough for RECOVERY_ADD
    trough_pct = 0.0  # worst dip seen so far

    # DCA window parameters per mode
    configs = {
        "POST_GRACE_DIP": {
            "window_start_min": 5.0,
            "window_end_min": 12.0,
            "min_dip_pct": 15.0,
            "max_dip_pct": 30.0,
            "check_underlying": True,
            "underlying_threshold": 0.3,  # underlying must not be >0.3% against
            "need_recovery": False,
        },
        "MID_TRADE_DIP": {
            "window_start_min": 8.0,
            "window_end_min": 20.0,
            "min_dip_pct": 15.0,
            "max_dip_pct": 35.0,
            "check_underlying": True,
            "underlying_threshold": 0.5,
            "need_recovery": False,
        },
        "RECOVERY_ADD": {
            "window_start_min": 10.0,
            "window_end_min": 25.0,
            "min_dip_pct": 5.0,    # currently down 5-15% ...
            "max_dip_pct": 15.0,
            "min_trough_pct": 20.0,  # ... but was down 20%+ at some point
            "check_underlying": True,
            "underlying_threshold": 0.3,
            "need_recovery": True,
        },
        "LATE_CONVICTION": {
            "window_start_min": 15.0,
            "window_end_min": 30.0,
            "min_dip_pct": 10.0,
            "max_dip_pct": 25.0,
            "check_underlying": True,
            "underlying_threshold": 0.0,  # underlying must CONFIRM direction
            "need_underlying_confirm": True,
            "need_recovery": False,
        },
    }

    cfg = configs.get(dca_mode) if dca_mode != "NONE" else None

    fsm = ExitFSM(V5Config())
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=avg_entry, entry_time=entry_ts,
        contracts=total_contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium

        now = df["ts"].iloc[idx]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        underlying = df["underlying_price"].iloc[idx] or 0.0
        elapsed_min = (now - entry_ts).total_seconds() / 60.0

        # Track trough
        dip_pct = (entry_premium - premium) / entry_premium * 100
        if dip_pct > trough_pct:
            trough_pct = dip_pct

        # ── Delayed DCA check ──
        if (not dca_fired and cfg is not None
                and cfg["window_start_min"] <= elapsed_min <= cfg["window_end_min"]):

            current_dip = (entry_premium - premium) / entry_premium * 100

            dip_ok = cfg["min_dip_pct"] <= current_dip <= cfg["max_dip_pct"]

            # Recovery check: was down 20%+, now only 5-15%
            recovery_ok = True
            if cfg.get("need_recovery"):
                recovery_ok = trough_pct >= cfg.get("min_trough_pct", 20)

            # Underlying check
            underlying_ok = True
            if cfg["check_underlying"] and first_underlying > 0 and underlying > 0:
                u_move = (underlying - first_underlying) / first_underlying * 100
                thresh = cfg["underlying_threshold"]

                if cfg.get("need_underlying_confirm"):
                    # Must CONFIRM direction (call needs up, put needs down)
                    if is_call:
                        underlying_ok = u_move > thresh
                    else:
                        underlying_ok = u_move < -thresh
                else:
                    # Must not be strongly against
                    if is_call:
                        underlying_ok = u_move > -thresh
                    else:
                        underlying_ok = u_move < thresh

            if dip_ok and recovery_ok and underlying_ok:
                dca_fired = True
                dca_price = premium
                dca_elapsed = elapsed_min

                # Add same number of contracts at cheaper price
                add_contracts = max(1, initial_contracts)
                total_cost += dca_price * add_contracts * 100
                total_contracts += add_contracts
                avg_entry = total_cost / (total_contracts * 100)

                state.entry_premium = avg_entry
                state.contracts = total_contracts
                state.peak_premium = max(avg_entry, premium)

        # Minutes to close
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
            pnl = (premium - avg_entry) * total_contracts * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed, "exit_prem": premium,
                "peak_gain": peak_gain, "dca_fired": dca_fired,
                "dca_price": dca_price, "dca_elapsed_min": dca_elapsed,
                "avg_entry": avg_entry, "total_contracts": total_contracts,
                "total_cost": total_cost, "trough_pct": trough_pct,
            }

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
    pnl = (last_prem - avg_entry) * total_contracts * 100
    return {
        "pnl": pnl, "reason": "eod_data_end",
        "hold": elapsed, "exit_prem": last_prem,
        "peak_gain": peak_gain, "dca_fired": dca_fired,
        "dca_price": dca_price, "dca_elapsed_min": dca_elapsed,
        "avg_entry": avg_entry, "total_contracts": total_contracts,
        "total_cost": total_cost, "trough_pct": trough_pct,
    }


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    modes = ["NONE", "POST_GRACE_DIP", "MID_TRADE_DIP", "RECOVERY_ADD", "LATE_CONVICTION"]
    all_results = {m: [] for m in modes}
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        contracts = compute_contracts(adj_entry, score)

        for mode in modes:
            result = simulate_delayed_dca(
                df, adj_entry, contracts, direction, dte, expiry_date,
                ticker=ticker, dca_mode=mode,
            )
            result.update({
                "ticker": ticker, "day": day, "score": score,
                "entry": adj_entry, "init_contracts": contracts,
                "direction": direction, "dte": dte,
            })
            all_results[mode].append(result)

    harvester_conn.close()

    if not all_results["NONE"]:
        print("No results — check DBs")
        return

    n_trades = len(all_results["NONE"])

    # ── Summary comparison ───────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print(f"DELAYED DCA BACKTEST — {n_trades} trades, {no_data} skipped")
    print(f"Adding contracts AFTER initial chaos (5-30min windows)")
    print(f"{'=' * 110}")

    print(f"\n{'Mode':<22} {'Total P&L':>12} {'Win Rate':>10} {'Avg Win':>10} {'Avg Loss':>10} "
          f"{'Fired':>6} {'Avg Cost':>10}")
    print("-" * 85)

    for mode in modes:
        df_r = pd.DataFrame(all_results[mode])
        pnls = df_r["pnl"]
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        total_pnl = pnls.sum()
        win_rate = wins / len(pnls) * 100
        avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
        avg_loss = pnls[pnls <= 0].mean() if losses > 0 else 0
        dca_count = df_r["dca_fired"].sum()
        avg_cost = df_r["total_cost"].mean()

        marker = " <-- BASELINE" if mode == "NONE" else ""
        print(f"{mode:<22} ${total_pnl:>10,.2f} {win_rate:>9.1f}% ${avg_win:>8,.2f} "
              f"${avg_loss:>8,.2f} {dca_count:>5} ${avg_cost:>8,.0f}{marker}")

    # ── When DCA fires — detailed comparison ────────────────────────────

    for mode in modes:
        if mode == "NONE":
            continue

        dca_results = all_results[mode]
        base_results = all_results["NONE"]
        fired_indices = [i for i, r in enumerate(dca_results) if r["dca_fired"]]

        if not fired_indices:
            print(f"\n{mode}: DCA never fired")
            continue

        dca_pnls = [dca_results[i]["pnl"] for i in fired_indices]
        base_pnls = [base_results[i]["pnl"] for i in fired_indices]

        dca_total = sum(dca_pnls)
        base_total = sum(base_pnls)
        improvement = dca_total - base_total

        dca_wins = sum(1 for p in dca_pnls if p > 0)
        base_wins = sum(1 for p in base_pnls if p > 0)
        helped = sum(1 for i in range(len(fired_indices))
                     if dca_pnls[i] > base_pnls[i])
        hurt = len(fired_indices) - helped

        print(f"\n{'=' * 110}")
        print(f"{mode}: DCA fired on {len(fired_indices)} trades "
              f"(helped {helped}, hurt {hurt})")
        print(f"{'=' * 110}")
        print(f"  {'':>30} {'Baseline':>12} {'With DCA':>12} {'Delta':>12}")
        print(f"  {'Total P&L':>30} ${base_total:>10,.2f} ${dca_total:>10,.2f} ${improvement:>10,.2f}")
        print(f"  {'Win Count':>30} {base_wins:>12} {dca_wins:>12} {dca_wins - base_wins:>+12}")
        print(f"  {'Win Rate':>30} {base_wins/len(fired_indices)*100:>11.1f}% "
              f"{dca_wins/len(fired_indices)*100:>11.1f}%")

        print(f"\n  {'Day':<12} {'Ticker':<6} {'Entry':>7} {'DCA@':>7} {'@Min':>5} "
              f"{'AvgE':>7} {'Ct':>3}→{'Ct':>3} {'Base P&L':>10} {'DCA P&L':>10} {'Delta':>10}")
        print(f"  {'-' * 95}")
        for i in fired_indices:
            b = base_results[i]
            d = dca_results[i]
            delta = d["pnl"] - b["pnl"]
            marker = " OK" if delta > 0 else " XX"
            print(f"  {d['day']:<12} {d['ticker']:<6} ${d['entry']:>5.2f} "
                  f"${d['dca_price']:>5.2f} {d['dca_elapsed_min']:>4.0f}m "
                  f"${d['avg_entry']:>5.2f} "
                  f"{b['total_contracts']:>3}→{d['total_contracts']:>3} "
                  f"${b['pnl']:>8,.2f} ${d['pnl']:>8,.2f} ${delta:>8,.2f}{marker}")

    # ── Best mode per ticker ─────────────────────────────────────────────

    print(f"\n\n{'=' * 110}")
    print("BEST DCA MODE PER TICKER")
    print(f"{'=' * 110}")

    tickers = sorted(set(r["ticker"] for r in all_results["NONE"]))
    print(f"\n{'Ticker':<8}", end="")
    for mode in modes:
        label = mode[:12]
        print(f" {label:>12}", end="")
    print(f" {'Best':>14}")
    print("-" * (8 + 13 * len(modes) + 15))

    for t in tickers:
        print(f"{t:<8}", end="")
        mode_pnls = {}
        for mode in modes:
            t_pnl = sum(r["pnl"] for r in all_results[mode] if r["ticker"] == t)
            mode_pnls[mode] = t_pnl
            print(f" ${t_pnl:>10,.0f}", end="")
        best = max(mode_pnls, key=mode_pnls.get)
        delta_vs_none = mode_pnls[best] - mode_pnls["NONE"]
        if best == "NONE":
            print(f" {'NONE':>14}")
        else:
            print(f" {best[:12]:>12} +${delta_vs_none:,.0f}")


if __name__ == "__main__":
    main()
