"""Backtest sizing sweep: find optimal MAX_POSITION_PCT / MAX_CONCURRENT
with daily GFV capital constraint.

The key constraint: in a cash account, total $ bought in a day cannot exceed
the starting settled balance without risking a Good Faith Violation. This means
smaller trades = more diversity = potentially better risk-adjusted returns.

Tests combos of position size (5-20%) and max concurrent (4-12) against
historical signals using the production V5 FSM exit engine.

Usage:
    python scripts/backtest_sizing_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
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

# Sweep parameters
PORTFOLIOS = {
    "kody_8k": 8000,
    "adam_2500": 2500,
}
POSITION_PCTS = [5, 8, 10, 12, 15, 20]
MAX_CONCURRENTS = [4, 6, 8, 10, 12]
RISK_PCT = 0.75  # MAX_PORTFOLIO_RISK_PCT

# GFV constraint: total $ bought in one day cannot exceed starting balance.
# This is the REAL limiter for cash accounts, not concurrent positions.

SCORE_TIERS = [
    (135, 1.00),
    (120, 0.85),
    (100, 0.85),
    (90, 0.50),
    (78, 0.25),
]
SCORE_FLOOR = 78


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


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM"):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    fsm = ExitFSM(V5Config())
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

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
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
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = (premium - entry_premium) * contracts * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = (last_prem - entry_premium) * contracts * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def size_trade(score, cost_per_contract, portfolio, max_pos_pct, max_concurrent):
    """Compute contracts for a given sizing config."""
    if score < SCORE_FLOOR:
        return 0
    score_mult = 0.25
    for threshold, mult in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break

    deployable = portfolio * RISK_PCT
    per_slot = deployable / max(1, max_concurrent)
    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per_contract) if cost_per_contract > 0 else 1

    max_spend = portfolio * (max_pos_pct / 100)
    pos_cap = int(max_spend / cost_per_contract) if cost_per_contract > 0 else 1

    return max(1, min(raw_contracts, pos_cap))


def run_sweep(signals, tick_cache, portfolio_name, portfolio_size):
    """Run all sizing combos for one portfolio size."""
    results = []

    for max_pos_pct in POSITION_PCTS:
        for max_conc in MAX_CONCURRENTS:
            daily_spend = defaultdict(float)  # day -> total $ bought (GFV tracker)
            daily_blocked_gfv = defaultdict(int)
            trade_results = []

            for sig in signals:
                ticker = sig["ticker"]
                score = sig["score"] or 80
                if score < SCORE_FLOOR:
                    continue

                day = sig["created_at"][:10]
                direction = (sig["direction"] or "bullish").lower()

                cache_key = (ticker, sig["strike"], sig["option_type"], day)
                if cache_key not in tick_cache:
                    continue
                df, adj_entry, dte, expiry_date = tick_cache[cache_key]

                cost_per = adj_entry * 100
                contracts = size_trade(score, cost_per, portfolio_size, max_pos_pct, max_conc)
                if contracts <= 0:
                    continue

                trade_cost = contracts * cost_per

                # GFV check: total $ bought today cannot exceed starting balance.
                # This is the REAL constraint — not concurrent positions.
                # Trades close intraday but the $ spent still counts.
                if daily_spend[day] + trade_cost > portfolio_size:
                    daily_blocked_gfv[day] += 1
                    # Simulate what WOULD have happened (for analysis)
                    would_result = simulate_fsm(df, adj_entry, contracts, direction, dte, expiry_date, ticker=ticker)
                    would_result["ticker"] = ticker
                    would_result["day"] = day
                    would_result["score"] = score
                    would_result["entry"] = adj_entry
                    would_result["contracts"] = contracts
                    would_result["cost"] = trade_cost
                    would_result["gfv_blocked"] = True
                    trade_results.append(would_result)
                    continue

                daily_spend[day] += trade_cost

                result = simulate_fsm(df, adj_entry, contracts, direction, dte, expiry_date, ticker=ticker)
                result["ticker"] = ticker
                result["day"] = day
                result["score"] = score
                result["entry"] = adj_entry
                result["contracts"] = contracts
                result["cost"] = trade_cost
                result["gfv_blocked"] = False
                trade_results.append(result)

            if not trade_results:
                continue

            df_r = pd.DataFrame(trade_results)
            traded = df_r[df_r["gfv_blocked"] == False]
            blocked = df_r[df_r["gfv_blocked"] == True]

            total_traded = len(traded)
            total_blocked = len(blocked)
            total_all = len(df_r)

            if total_traded == 0:
                continue

            traded_pnl = traded["pnl"].sum()
            blocked_pnl = blocked["pnl"].sum() if total_blocked > 0 else 0
            traded_wins = (traded["pnl"] > 0).sum()
            blocked_wins = (blocked["pnl"] > 0).sum() if total_blocked > 0 else 0
            win_rate = traded_wins / total_traded * 100
            blocked_wr = blocked_wins / total_blocked * 100 if total_blocked > 0 else 0

            avg_contracts = traded["contracts"].mean()
            avg_cost = traded["cost"].mean()
            trading_days = max(1, len(set(df_r["day"])))

            theoretical_max = int(portfolio_size / avg_cost) if avg_cost > 0 else 0

            results.append({
                "portfolio": portfolio_name,
                "portfolio_size": portfolio_size,
                "max_pos_pct": max_pos_pct,
                "max_concurrent": max_conc,
                "traded": total_traded,
                "blocked": total_blocked,
                "total_signals": total_all,
                "pnl": traded_pnl,
                "blocked_pnl": blocked_pnl,
                "win_rate": win_rate,
                "blocked_wr": blocked_wr,
                "avg_contracts": avg_contracts,
                "avg_cost": avg_cost,
                "theoretical_max_daily": theoretical_max,
                "traded_per_day": total_traded / trading_days,
                "avg_win": traded.loc[traded["pnl"] > 0, "pnl"].mean() if traded_wins > 0 else 0,
                "avg_loss": traded.loc[traded["pnl"] <= 0, "pnl"].mean() if (total_traded - traded_wins) > 0 else 0,
                "max_drawdown": traded.groupby("day")["pnl"].sum().min(),
            })

    return results


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load all tick data (shared across sizing combos)
    print("Loading tick data...")
    tick_cache = {}
    no_data = 0
    for sig in signals:
        score = sig["score"] or 80
        if score < SCORE_FLOOR:
            continue
        ticker = sig["ticker"]
        day = sig["created_at"][:10]
        direction = (sig["direction"] or "bullish").lower()
        cache_key = (ticker, sig["strike"], sig["option_type"], day)

        if cache_key in tick_cache:
            continue

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        tick_cache[cache_key] = (df, adj_entry, dte, expiry_date)

    harvester_conn.close()
    print(f"Cached {len(tick_cache)} unique signal/tick pairs ({no_data} had no data)\n")

    # Run sweep for each portfolio
    all_results = []
    for name, size in PORTFOLIOS.items():
        print(f"Sweeping {name} (${size:,})...")
        results = run_sweep(signals, tick_cache, name, size)
        all_results.extend(results)
        print(f"  {len(results)} combos tested")

    if not all_results:
        print("No results!")
        return

    df = pd.DataFrame(all_results)

    # Print results for each portfolio
    for name, size in PORTFOLIOS.items():
        pdf = df[df["portfolio"] == name].copy()
        if pdf.empty:
            continue

        pdf = pdf.sort_values("pnl", ascending=False)

        print(f"\n{'=' * 150}")
        print(f"SIZING SWEEP: {name} (${size:,}) — GFV constraint: total daily buys <= ${size:,}")
        print(f"{'=' * 150}")
        print(f"{'Pos%':>5} {'MaxC':>5} {'Traded':>7} {'Blocked':>8} {'Total':>6} "
              f"{'Traded P&L':>11} {'Blocked P&L':>12} {'TrWR%':>6} {'BlkWR%':>7} "
              f"{'AvgCt':>6} {'AvgCost':>8} {'MaxDay':>7} "
              f"{'AvgWin':>8} {'AvgLoss':>9} {'WorstDay':>9}")
        print("-" * 150)

        for _, r in pdf.iterrows():
            print(f"{r['max_pos_pct']:>4.0f}% {r['max_concurrent']:>5.0f} "
                  f"{r['traded']:>7.0f} {r['blocked']:>8.0f} {r['total_signals']:>6.0f} "
                  f"${r['pnl']:>9,.2f} ${r['blocked_pnl']:>10,.2f} "
                  f"{r['win_rate']:>5.1f}% {r['blocked_wr']:>6.1f}% "
                  f"{r['avg_contracts']:>5.1f} ${r['avg_cost']:>6,.0f} "
                  f"{r['theoretical_max_daily']:>7.0f} "
                  f"${r['avg_win']:>6,.0f} ${r['avg_loss']:>7,.0f} "
                  f"${r['max_drawdown']:>7,.0f}")

        # Summary insights
        best = pdf.iloc[0]
        print(f"\nBEST P&L: {best['max_pos_pct']:.0f}% / {best['max_concurrent']:.0f}conc → "
              f"${best['pnl']:,.2f} ({best['traded']:.0f} traded, "
              f"{best['blocked']:.0f} blocked = ${best['blocked_pnl']:,.2f} missed)")

        pdf["pnl_per_trade"] = pdf["pnl"] / pdf["traded"]
        best_adj = pdf.sort_values("pnl_per_trade", ascending=False).iloc[0]
        print(f"BEST $/trade: {best_adj['max_pos_pct']:.0f}% / {best_adj['max_concurrent']:.0f}conc → "
              f"${best_adj['pnl_per_trade']:,.2f}/trade")

        # Show the key tradeoff: what are we MISSING by blocking signals?
        high_block = pdf[pdf["blocked"] > 0].sort_values("blocked", ascending=False)
        if not high_block.empty:
            worst = high_block.iloc[0]
            print(f"\nMOST BLOCKED: {worst['max_pos_pct']:.0f}% / {worst['max_concurrent']:.0f}conc → "
                  f"{worst['blocked']:.0f} signals blocked, "
                  f"they would have earned ${worst['blocked_pnl']:,.2f} "
                  f"({worst['blocked_wr']:.0f}% win rate)")

        # Zero-blocked configs
        zero_blocked = pdf[pdf["blocked"] == 0].sort_values("pnl", ascending=False)
        if not zero_blocked.empty:
            zb = zero_blocked.iloc[0]
            print(f"BEST WITH ZERO BLOCKS: {zb['max_pos_pct']:.0f}% / {zb['max_concurrent']:.0f}conc → "
                  f"${zb['pnl']:,.2f} P&L, {zb['traded']:.0f} trades, "
                  f"{zb['win_rate']:.1f}% WR, ~{zb['theoretical_max_daily']:.0f} trades/day")


if __name__ == "__main__":
    main()
