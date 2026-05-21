"""Backtest DCA (Dollar Cost Averaging) on dips — does adding contracts help?

Simulates: when a position dips in the first N minutes but the underlying
thesis is still intact (underlying not moving against), add more contracts
at the cheaper price. Compare vs baseline (no DCA).

Tests multiple DCA strategies:
  1. SINGLE_DIP: Add once when premium drops 15-30% in first 5min, underlying OK
  2. AGGRESSIVE_DIP: Add once at any 10%+ dip in first 3min
  3. TWO_TRANCHES: Split initial order — 60% at entry, 40% at first 15%+ dip
  4. NONE: Baseline (no DCA)

Usage:
    python scripts/backtest_dca.py
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


# ── Data loading (same as backtest_v5_production.py) ────────────────────────


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


def compute_contracts(entry_premium, score, budget_fraction=1.0):
    """Dollar-target sizing matching production."""
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
    scaled_target = per_slot * score_mult * budget_fraction
    raw = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, pos_cap, 20))


# ── DCA simulation ──────────────────────────────────────────────────────────


def simulate_dca(df, entry_premium, initial_contracts, direction, dte, expiry_date,
                 ticker, dca_mode="NONE"):
    """Run FSM with optional DCA add during early dip.

    DCA modes:
      NONE          — baseline, no averaging
      SINGLE_DIP    — add contracts once if premium dips 15-30% in first 5min, underlying OK
      AGGRESSIVE_DIP— add once at any 10%+ dip in first 3min
      TWO_TRANCHES  — enter with 60% of contracts, add remaining 40% at first 15%+ dip
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False, "dca_price": 0,
                "avg_entry": entry_premium, "total_contracts": initial_contracts,
                "total_cost": entry_premium * initial_contracts * 100}

    option_type = "put" if direction in ("bearish", "put") else "call"
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

    # TWO_TRANCHES: enter with 60% of contracts, hold 40% back
    if dca_mode == "TWO_TRANCHES":
        contracts_now = max(1, int(initial_contracts * 0.6))
        contracts_reserve = initial_contracts - contracts_now
    else:
        contracts_now = initial_contracts
        contracts_reserve = 0

    # DCA tracking
    dca_fired = False
    dca_price = 0.0
    total_contracts = contracts_now
    total_cost = entry_premium * contracts_now * 100  # total $ invested
    avg_entry = entry_premium

    # DCA parameters per mode
    if dca_mode == "SINGLE_DIP":
        dca_window_min = 5.0
        dca_min_dip_pct = 15.0
        dca_max_dip_pct = 35.0  # don't add if down >35% (probably broken)
        dca_check_underlying = True
    elif dca_mode == "AGGRESSIVE_DIP":
        dca_window_min = 3.0
        dca_min_dip_pct = 10.0
        dca_max_dip_pct = 40.0
        dca_check_underlying = False
    elif dca_mode == "TWO_TRANCHES":
        dca_window_min = 7.0
        dca_min_dip_pct = 15.0
        dca_max_dip_pct = 40.0
        dca_check_underlying = True
    else:
        dca_window_min = 0
        dca_min_dip_pct = 999
        dca_max_dip_pct = 999
        dca_check_underlying = False

    # FSM uses avg_entry as the effective entry for exit decisions
    fsm = ExitFSM(V5Config())
    state = TradeState(
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=avg_entry,
        entry_time=entry_ts,
        contracts=total_contracts,
        peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte,
        expiry_date=expiry_date or "",
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

        # ── DCA check: add contracts on dip ──
        if not dca_fired and dca_mode != "NONE" and elapsed_min <= dca_window_min:
            dip_pct = (entry_premium - premium) / entry_premium * 100
            if dca_min_dip_pct <= dip_pct <= dca_max_dip_pct:
                # Check underlying isn't moving against us
                underlying_ok = True
                if dca_check_underlying and first_underlying > 0 and underlying > 0:
                    u_move = (underlying - first_underlying) / first_underlying * 100
                    is_call = option_type in ("call", "bullish")
                    if is_call and u_move < -0.5:
                        underlying_ok = False
                    elif not is_call and u_move > 0.5:
                        underlying_ok = False

                if underlying_ok:
                    dca_fired = True
                    dca_price = premium

                    if dca_mode == "TWO_TRANCHES":
                        add_contracts = max(1, contracts_reserve)
                    else:
                        # Add same number of contracts at cheaper price
                        add_contracts = max(1, contracts_now)

                    total_cost += dca_price * add_contracts * 100
                    total_contracts += add_contracts
                    avg_entry = total_cost / (total_contracts * 100)

                    # Update FSM state with new average entry
                    state.entry_premium = avg_entry
                    state.contracts = total_contracts
                    # Reset peak to avg entry (we're effectively re-entering)
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
                "pnl": pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "dca_fired": dca_fired,
                "dca_price": dca_price,
                "avg_entry": avg_entry,
                "total_contracts": total_contracts,
                "total_cost": total_cost,
            }

    # End of data
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
        "pnl": pnl,
        "reason": "eod_data_end",
        "hold": elapsed,
        "exit_prem": last_prem,
        "peak_gain": peak_gain,
        "dca_fired": dca_fired,
        "dca_price": dca_price,
        "avg_entry": avg_entry,
        "total_contracts": total_contracts,
        "total_cost": total_cost,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    modes = ["NONE", "SINGLE_DIP", "AGGRESSIVE_DIP", "TWO_TRANCHES"]
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
            result = simulate_dca(
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

    # ── Summary comparison ───────────────────────────────────────────────

    n_trades = len(all_results["NONE"])
    print(f"\n{'=' * 100}")
    print(f"DCA BACKTEST — {n_trades} trades, {no_data} skipped (no tick data)")
    print(f"{'=' * 100}")

    print(f"\n{'Mode':<20} {'Total P&L':>12} {'Win Rate':>10} {'Avg Win':>10} {'Avg Loss':>10} "
          f"{'DCA Fired':>10} {'Avg Cost':>10}")
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

        print(f"{mode:<20} ${total_pnl:>10,.2f} {win_rate:>9.1f}% ${avg_win:>8,.2f} "
              f"${avg_loss:>8,.2f} {dca_count:>9} ${avg_cost:>8,.0f}")

    # ── DCA fired analysis: when DCA fires, does it help? ──────────────

    print(f"\n\n{'=' * 100}")
    print("WHEN DCA FIRES — Comparing same trades with and without DCA")
    print(f"{'=' * 100}")

    for mode in modes:
        if mode == "NONE":
            continue

        # Find trades where DCA actually fired
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

        print(f"\n{mode}: DCA fired on {len(fired_indices)} trades")
        print(f"  {'':>30} {'Baseline':>12} {'With DCA':>12} {'Delta':>12}")
        print(f"  {'Total P&L':>30} ${base_total:>10,.2f} ${dca_total:>10,.2f} ${improvement:>10,.2f}")
        print(f"  {'Win Count':>30} {base_wins:>12} {dca_wins:>12} {dca_wins - base_wins:>+12}")
        print(f"  {'Win Rate':>30} {base_wins/len(fired_indices)*100:>11.1f}% "
              f"{dca_wins/len(fired_indices)*100:>11.1f}%")

        # Per-trade detail where DCA fired
        print(f"\n  {'Day':<12} {'Ticker':<6} {'Entry':>7} {'DCA@':>7} {'AvgE':>7} "
              f"{'Ct':>3}→{'Ct':>3} {'Base P&L':>10} {'DCA P&L':>10} {'Delta':>10}")
        print(f"  {'-' * 90}")
        for i in fired_indices:
            b = base_results[i]
            d = dca_results[i]
            delta = d["pnl"] - b["pnl"]
            marker = " ✓" if delta > 0 else " ✗"
            print(f"  {d['day']:<12} {d['ticker']:<6} ${d['entry']:>5.2f} "
                  f"${d['dca_price']:>5.2f} ${d['avg_entry']:>5.2f} "
                  f"{b['total_contracts']:>3}→{d['total_contracts']:>3} "
                  f"${b['pnl']:>8,.2f} ${d['pnl']:>8,.2f} ${delta:>8,.2f}{marker}")

    # ── Per-ticker DCA analysis ──────────────────────────────────────────

    print(f"\n\n{'=' * 100}")
    print("PER-TICKER DCA IMPACT (SINGLE_DIP mode)")
    print(f"{'=' * 100}")

    dca_r = all_results["SINGLE_DIP"]
    base_r = all_results["NONE"]

    ticker_data = {}
    for i in range(len(dca_r)):
        t = dca_r[i]["ticker"]
        if t not in ticker_data:
            ticker_data[t] = {"base_pnl": 0, "dca_pnl": 0, "fired": 0, "total": 0}
        ticker_data[t]["base_pnl"] += base_r[i]["pnl"]
        ticker_data[t]["dca_pnl"] += dca_r[i]["pnl"]
        ticker_data[t]["total"] += 1
        if dca_r[i]["dca_fired"]:
            ticker_data[t]["fired"] += 1

    print(f"\n{'Ticker':<8} {'Trades':>6} {'Fired':>6} {'Base P&L':>12} {'DCA P&L':>12} {'Delta':>12}")
    print("-" * 60)
    for t in sorted(ticker_data, key=lambda x: ticker_data[x]["dca_pnl"] - ticker_data[x]["base_pnl"], reverse=True):
        d = ticker_data[t]
        delta = d["dca_pnl"] - d["base_pnl"]
        print(f"{t:<8} {d['total']:>6} {d['fired']:>6} ${d['base_pnl']:>10,.2f} "
              f"${d['dca_pnl']:>10,.2f} ${delta:>10,.2f}")

    print(f"\nTotal baseline: ${sum(d['base_pnl'] for d in ticker_data.values()):,.2f}")
    print(f"Total DCA:      ${sum(d['dca_pnl'] for d in ticker_data.values()):,.2f}")
    delta_all = sum(d['dca_pnl'] - d['base_pnl'] for d in ticker_data.values())
    print(f"Net DCA impact: ${delta_all:>+,.2f}")


if __name__ == "__main__":
    main()
