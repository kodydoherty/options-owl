"""Backtest with compounding: portfolio grows/shrinks daily based on P&L.

Uses the new 8%/8 sizing with score-tiered position caps and GFV constraint.
Shows daily returns, compounded balance, and average daily return.

Usage:
    python scripts/backtest_compound.py
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

# (starting_balance, base_pos_pct, max_concurrent)
STARTING_PORTFOLIOS = {
    "kody_15pct_4c": (8227, 0.15, 4),
}

RISK_PCT = 0.75
SCORE_FLOOR = 78
CIRCUIT_BREAKER_PCT = 25.0  # stop trading when daily losses exceed this % of portfolio

# Match production: score-tiered budget mult + position cap
SCORE_TIERS = [
    (135, 1.00, 0.15),  # elite: 100% budget, 15% position cap
    (120, 0.85, 0.12),  # strong: 85%, 12%
    (100, 0.85, 0.08),  # standard: 85%, 8%
    (90, 0.50, 0.08),   # moderate: 50%, 8%
    (78, 0.25, 0.08),   # marginal: 25%, 8%
]


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
        return {"pnl": 0, "reason": "no_data", "hold": 0}

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
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            pnl = (premium - entry_premium) * contracts * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    pnl = (last_prem - entry_premium) * contracts * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed}


def size_trade(score, cost_per_contract, balance, base_pos_pct=0.08, max_conc=8):
    if score < SCORE_FLOOR or cost_per_contract <= 0 or balance <= 0:
        return 0, 0.0

    score_mult = 0.25
    tier_pos_pct = base_pos_pct
    for threshold, mult, cap in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            tier_pos_pct = max(cap, base_pos_pct)
            break

    deployable = balance * RISK_PCT
    per_slot = deployable / max_conc
    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per_contract)

    max_spend = balance * tier_pos_pct
    pos_cap = int(max_spend / cost_per_contract)

    # If 1 contract exceeds the position cap, skip (matches production)
    if pos_cap == 0:
        return 0, 0.0

    contracts = max(1, min(raw_contracts, pos_cap))
    cost = contracts * cost_per_contract
    return contracts, cost


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals\n")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load tick data
    tick_cache = {}
    no_data = 0
    for sig in signals:
        score = sig["score"] or 80
        if score < SCORE_FLOOR:
            continue
        cache_key = (sig["ticker"], sig["strike"], sig["option_type"], sig["created_at"][:10])
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
        tick_cache[cache_key] = (df, adj_entry, sig.get("_dte", 0), sig.get("_expiry_date", ""))
    harvester_conn.close()
    print(f"Cached {len(tick_cache)} signals ({no_data} no data)\n")

    # Run compounding simulation for each portfolio
    for name, (starting_balance, base_pos_pct, max_conc) in STARTING_PORTFOLIOS.items():
        balance = starting_balance
        daily_results = []
        daily_spend = defaultdict(float)
        trade_details = []
        current_day = None
        day_pnl = 0.0
        day_trades = 0
        day_blocked = 0

        for sig in signals:
            score = sig["score"] or 80
            if score < SCORE_FLOOR:
                continue

            day = sig["created_at"][:10]
            direction = (sig["direction"] or "bullish").lower()
            cache_key = (sig["ticker"], sig["strike"], sig["option_type"], day)

            if cache_key not in tick_cache:
                continue

            # New day — apply previous day's P&L to balance
            if current_day and day != current_day:
                daily_results.append({
                    "day": current_day,
                    "balance_start": balance - day_pnl,
                    "pnl": day_pnl,
                    "balance_end": balance,
                    "trades": day_trades,
                    "blocked": day_blocked,
                })
                daily_spend.clear()
                day_pnl = 0.0
                day_trades = 0
                day_blocked = 0
            current_day = day

            df, adj_entry, dte, expiry_date = tick_cache[cache_key]
            cost_per = adj_entry * 100

            # Circuit breaker: stop trading if day's realized losses exceed threshold
            cb_limit = balance * (CIRCUIT_BREAKER_PCT / 100)
            if day_pnl < -cb_limit:
                day_blocked += 1
                continue

            # Momentum gate: check if underlying is fading against our direction
            is_call = direction in ("bullish", "call")
            window = min(15, len(df))
            underlying_prices = []
            for i in range(window):
                u = df["underlying_price"].iloc[i]
                if u and u > 0:
                    underlying_prices.append(float(u))

            if len(underlying_prices) >= 5:
                first_half = underlying_prices[:len(underlying_prices)//2]
                second_half = underlying_prices[len(underlying_prices)//2:]
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)
                pct_move = (avg_second - avg_first) / avg_first * 100

                prem_start = df["premium"].iloc[0]
                prem_5 = df["premium"].iloc[min(4, len(df)-1)]
                prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0

                neg_signals = 0
                if is_call and pct_move < -0.05:
                    neg_signals += 1
                elif not is_call and pct_move > 0.05:
                    neg_signals += 1
                if prem_fade < -5:
                    neg_signals += 1
                against = 0
                for i in range(max(0, window-3), window):
                    if i == 0:
                        continue
                    prev_u = df["underlying_price"].iloc[i-1]
                    cur_u = df["underlying_price"].iloc[i]
                    if prev_u and cur_u:
                        if is_call and cur_u < prev_u:
                            against += 1
                        elif not is_call and cur_u > prev_u:
                            against += 1
                if against >= 3:
                    neg_signals += 1
                if neg_signals >= 2:
                    day_blocked += 1
                    continue

            contracts, trade_cost = size_trade(score, cost_per, balance, base_pos_pct, max_conc)
            if contracts <= 0:
                # size_trade returns 0 when 1 contract exceeds position cap
                day_blocked += 1
                continue

            # GFV check: total daily buys cannot exceed starting balance for the day
            if daily_spend[day] + trade_cost > balance:
                day_blocked += 1
                continue

            daily_spend[day] += trade_cost

            result = simulate_fsm(df, adj_entry, contracts, direction, dte, expiry_date, ticker=sig["ticker"])
            pnl = result["pnl"]

            balance += pnl
            day_pnl += pnl
            day_trades += 1

            # Track per-trade details for analysis
            if name == list(STARTING_PORTFOLIOS.keys())[0]:  # only first portfolio
                trade_details.append({
                    "day": day,
                    "ticker": sig["ticker"],
                    "score": score,
                    "direction": direction,
                    "contracts": contracts,
                    "cost": trade_cost,
                    "entry_prem": adj_entry,
                    "pnl": pnl,
                    "exit_reason": result.get("reason", ""),
                    "hold_min": result.get("hold", 0),
                })

        # Final day
        if current_day:
            daily_results.append({
                "day": current_day,
                "balance_start": balance - day_pnl,
                "pnl": day_pnl,
                "balance_end": balance,
                "trades": day_trades,
                "blocked": day_blocked,
            })

        if not daily_results:
            print(f"{name}: no results")
            continue

        df_d = pd.DataFrame(daily_results)
        total_pnl = df_d["pnl"].sum()
        total_trades = df_d["trades"].sum()
        total_blocked = df_d["blocked"].sum()
        trading_days = len(df_d)
        avg_daily_pnl = df_d["pnl"].mean()
        avg_daily_return = (df_d["pnl"] / df_d["balance_start"]).mean() * 100
        win_days = (df_d["pnl"] > 0).sum()
        final_balance = df_d["balance_end"].iloc[-1]
        total_return = (final_balance - starting_balance) / starting_balance * 100

        print(f"{'=' * 100}")
        print(f"COMPOUNDING BACKTEST: {name.upper()} — ${starting_balance:,} starting")
        print(f"{'=' * 100}")
        print(f"Final balance:    ${final_balance:,.2f} ({total_return:+.1f}%)")
        print(f"Total P&L:        ${total_pnl:,.2f}")
        print(f"Trading days:     {trading_days}")
        print(f"Total trades:     {total_trades} ({total_blocked} GFV-blocked)")
        print(f"Avg daily P&L:    ${avg_daily_pnl:,.2f}")
        print(f"Avg daily return: {avg_daily_return:+.2f}%")
        print(f"Win days:         {win_days}/{trading_days} ({win_days/trading_days*100:.0f}%)")
        print(f"Best day:         ${df_d['pnl'].max():,.2f}")
        print(f"Worst day:        ${df_d['pnl'].min():,.2f}")
        print(f"Max balance:      ${df_d['balance_end'].max():,.2f}")
        print(f"Min balance:      ${df_d['balance_end'].min():,.2f}")

        print(f"\n{'Day':<12} {'Start':>9} {'Trades':>7} {'Blkd':>5} {'Day P&L':>10} "
              f"{'End':>9} {'Return':>8}")
        print("-" * 70)
        for _, r in df_d.iterrows():
            ret = r["pnl"] / r["balance_start"] * 100 if r["balance_start"] > 0 else 0
            print(f"{r['day']:<12} ${r['balance_start']:>7,.0f} {r['trades']:>7} "
                  f"{r['blocked']:>5} ${r['pnl']:>8,.2f} ${r['balance_end']:>7,.0f} "
                  f"{ret:>+7.1f}%")

        # Per-trade analysis on losing days
        if trade_details:
            td = pd.DataFrame(trade_details)
            losers = td[td["pnl"] < 0].sort_values("pnl")
            winners = td[td["pnl"] > 0].sort_values("pnl", ascending=False)
            win_rate = len(winners) / len(td) * 100 if len(td) > 0 else 0
            avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
            avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0

            print(f"\n  Trade-level: {len(winners)}W / {len(losers)}L ({win_rate:.0f}% WR)")
            print(f"  Avg win: ${avg_win:,.2f}  |  Avg loss: ${avg_loss:,.2f}  |  Ratio: {abs(avg_win/avg_loss) if avg_loss else 0:.1f}x")

            # Biggest losers
            print(f"\n  Top 10 Losers:")
            print(f"  {'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} {'Ctrs':>4} {'Entry':>6} {'P&L':>9} {'Exit':>20} {'Hold':>5}")
            for _, t in losers.head(10).iterrows():
                print(f"  {t['day']:<12} {t['ticker']:<6} {t['direction']:<5} {t['score']:>5} "
                      f"{t['contracts']:>4} ${t['entry_prem']:>5.2f} ${t['pnl']:>8,.2f} "
                      f"{t['exit_reason']:>20} {t['hold_min']:>4.0f}m")

            # Exit reason breakdown
            reason_pnl = td.groupby("exit_reason").agg(
                count=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                wins=("pnl", lambda x: (x > 0).sum()),
            ).sort_values("total_pnl")
            print(f"\n  Exit Reason Breakdown:")
            print(f"  {'Reason':<25} {'Count':>5} {'Wins':>4} {'WR':>5} {'Total P&L':>10} {'Avg P&L':>9}")
            for reason, row in reason_pnl.iterrows():
                wr = row["wins"] / row["count"] * 100 if row["count"] > 0 else 0
                print(f"  {reason:<25} {row['count']:>5} {row['wins']:>4} {wr:>4.0f}% ${row['total_pnl']:>9,.2f} ${row['avg_pnl']:>8,.2f}")

        # Projected monthly/annual returns (compound)
        if avg_daily_return > 0:
            monthly = ((1 + avg_daily_return / 100) ** 21 - 1) * 100
            annual = ((1 + avg_daily_return / 100) ** 252 - 1) * 100
            print(f"\nProjected (compound): {avg_daily_return:+.2f}%/day → "
                  f"{monthly:+.1f}%/month → {annual:+.1f}%/year")
            print(f"  $8K → ${8000 * (1 + monthly/100):,.0f}/month, "
                  f"${8000 * (1 + annual/100):,.0f}/year")
        print()


if __name__ == "__main__":
    main()
