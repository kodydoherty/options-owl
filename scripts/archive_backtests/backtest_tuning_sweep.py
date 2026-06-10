"""Sweep three consistency levers across per-ticker signals.

Levers tested:
  1. Premium floor  — skip cheap lottery tickets (e.g. < $0.50)
  2. Max loss cap   — cap $ loss per trade as % of portfolio
  3. Tighter stops  — lower hard_stop / checkpoint thresholds in V5 config

Also shows per-ticker breakdown so we can tune per-ticker if needed.

Usage:
    python scripts/backtest_tuning_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from copy import deepcopy
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

STARTING_BALANCE = 8227
BASE_POS_PCT = 0.15
MAX_CONC = 4
RISK_PCT = 0.75
SCORE_FLOOR = 78
CIRCUIT_BREAKER_PCT = 25.0

SCORE_TIERS = [
    (135, 1.00, 0.15),
    (120, 0.85, 0.12),
    (100, 0.85, 0.08),
    (90, 0.50, 0.08),
    (78, 0.25, 0.08),
]


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT ticker, strike, direction, sentiment, score,
               atm_premium, otm_premium, expiry,
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


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date,
                 ticker="SIM", v5_config=None, max_loss_dollars=None):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0}

    cfg = v5_config or V5Config()
    fsm = ExitFSM(cfg)
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

        # Max loss cap: force exit if unrealized loss exceeds cap
        if max_loss_dollars and max_loss_dollars > 0:
            unrealized = (premium - entry_premium) * contracts * 100
            if unrealized < -max_loss_dollars:
                elapsed = (now - entry_ts).total_seconds() / 60
                return {"pnl": -max_loss_dollars, "reason": "max_loss_cap", "hold": elapsed}

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
    if max_loss_dollars and max_loss_dollars > 0 and pnl < -max_loss_dollars:
        pnl = -max_loss_dollars
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed}


def size_trade(score, cost_per_contract, balance, base_pos_pct=0.15, max_conc=4):
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
    if pos_cap == 0:
        return 0, 0.0
    contracts = max(1, min(raw_contracts, pos_cap))
    cost = contracts * cost_per_contract
    return contracts, cost


def momentum_gate(df, direction, is_call):
    """Return True if trade should be BLOCKED."""
    window = min(15, len(df))
    underlying_prices = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))
    if len(underlying_prices) < 5:
        return False

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
    return neg_signals >= 2


def run_scenario(signals, tick_cache, scenario_name, premium_floor=0.0,
                 max_loss_pct=0.0, v5_config=None):
    """Run one scenario and return trade-level results."""
    balance = STARTING_BALANCE
    daily_spend = defaultdict(float)
    current_day = None
    day_pnl = 0.0
    trades = []

    for sig in signals:
        score = sig["score"] or 80
        if score < SCORE_FLOOR:
            continue

        day = sig["created_at"][:10]
        direction = (sig["direction"] or "bullish").lower()
        cache_key = (sig["ticker"], sig["strike"], sig["option_type"], day)
        if cache_key not in tick_cache:
            continue

        # New day reset
        if current_day and day != current_day:
            daily_spend.clear()
            day_pnl = 0.0
        current_day = day

        df, adj_entry, dte, expiry_date = tick_cache[cache_key]
        cost_per = adj_entry * 100

        # Circuit breaker
        cb_limit = balance * (CIRCUIT_BREAKER_PCT / 100)
        if day_pnl < -cb_limit:
            continue

        # Premium floor
        if premium_floor > 0 and adj_entry < premium_floor:
            continue

        is_call = direction in ("bullish", "call")
        if momentum_gate(df, direction, is_call):
            continue

        contracts, trade_cost = size_trade(score, cost_per, balance)
        if contracts <= 0:
            continue

        # GFV check
        if daily_spend[day] + trade_cost > balance:
            continue
        daily_spend[day] += trade_cost

        # Max loss cap in dollars
        max_loss_dollars = None
        if max_loss_pct > 0:
            max_loss_dollars = balance * (max_loss_pct / 100)

        result = simulate_fsm(df, adj_entry, contracts, direction, dte,
                              expiry_date, ticker=sig["ticker"],
                              v5_config=v5_config, max_loss_dollars=max_loss_dollars)
        pnl = result["pnl"]
        balance += pnl
        day_pnl += pnl

        trades.append({
            "day": day,
            "ticker": sig["ticker"],
            "score": score,
            "direction": direction,
            "contracts": contracts,
            "entry_prem": adj_entry,
            "cost": trade_cost,
            "pnl": pnl,
            "exit_reason": result.get("reason", ""),
            "hold_min": result.get("hold", 0),
        })

    return trades, balance


def print_scenario(name, trades, final_balance):
    td = pd.DataFrame(trades)
    if td.empty:
        print(f"\n{name}: no trades")
        return

    total_pnl = td["pnl"].sum()
    winners = td[td["pnl"] > 0]
    losers = td[td["pnl"] < 0]
    win_rate = len(winners) / len(td) * 100
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    ratio = abs(avg_win / avg_loss) if avg_loss else float('inf')

    # Daily aggregation
    day_pnl = td.groupby("day")["pnl"].sum()
    win_days = (day_pnl > 0).sum()
    total_days = len(day_pnl)
    avg_daily = day_pnl.mean()
    worst_day = day_pnl.min()
    best_day = day_pnl.max()

    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"{'='*100}")
    print(f"  Final: ${final_balance:,.0f} ({total_return:+.1f}%)  |  "
          f"Trades: {len(td)} ({len(winners)}W/{len(losers)}L, {win_rate:.0f}% WR)  |  "
          f"Win/Loss ratio: {ratio:.1f}x")
    print(f"  Avg win: ${avg_win:,.0f}  |  Avg loss: ${avg_loss:,.0f}  |  "
          f"Win days: {win_days}/{total_days} ({win_days/total_days*100:.0f}%)")
    print(f"  Best day: ${best_day:,.0f}  |  Worst day: ${worst_day:,.0f}  |  "
          f"Avg daily: ${avg_daily:,.0f}")

    # Per-ticker breakdown
    ticker_stats = td.groupby("ticker").agg(
        count=("pnl", "count"),
        wins=("pnl", lambda x: (x > 0).sum()),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_entry=("entry_prem", "mean"),
        worst=("pnl", "min"),
        best=("pnl", "max"),
    ).sort_values("total_pnl")

    print(f"\n  Per-Ticker Breakdown:")
    print(f"  {'Ticker':<7} {'Trades':>6} {'Wins':>5} {'WR':>5} {'Total P&L':>10} "
          f"{'Avg P&L':>9} {'Avg Entry':>9} {'Worst':>9} {'Best':>9}")
    print(f"  {'-'*80}")
    for ticker, row in ticker_stats.iterrows():
        wr = row["wins"] / row["count"] * 100 if row["count"] > 0 else 0
        print(f"  {ticker:<7} {row['count']:>6} {row['wins']:>5} {wr:>4.0f}% "
              f"${row['total_pnl']:>9,.0f} ${row['avg_pnl']:>8,.0f} "
              f"${row['avg_entry']:>8.2f} ${row['worst']:>8,.0f} ${row['best']:>8,.0f}")

    # Exit reason breakdown
    reason_stats = td.groupby("exit_reason").agg(
        count=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    ).sort_values("total_pnl")
    print(f"\n  Exit Reasons:")
    print(f"  {'Reason':<25} {'Count':>5} {'Wins':>4} {'WR':>5} {'Total P&L':>10}")
    for reason, row in reason_stats.iterrows():
        wr = row["wins"] / row["count"] * 100
        print(f"  {reason:<25} {row['count']:>5} {row['wins']:>4} {wr:>4.0f}% ${row['total_pnl']:>9,.0f}")

    # Daily breakdown
    print(f"\n  Daily:")
    print(f"  {'Day':<12} {'Trades':>6} {'W':>3} {'L':>3} {'P&L':>10}")
    for day, group in td.groupby("day"):
        w = (group["pnl"] > 0).sum()
        l = (group["pnl"] < 0).sum()
        pnl = group["pnl"].sum()
        print(f"  {day:<12} {len(group):>6} {w:>3} {l:>3} ${pnl:>9,.0f}")


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
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

    # ── BASELINE ──
    print("\n" + "█" * 100)
    print("  BASELINE — current production config (15%/4c, no premium floor, no loss cap)")
    print("█" * 100)
    trades, bal = run_scenario(signals, tick_cache, "BASELINE")
    print_scenario("BASELINE", trades, bal)

    # ── LEVER 1: Premium Floor ──
    print("\n\n" + "█" * 100)
    print("  LEVER 1: PREMIUM FLOOR — skip cheap lottery tickets")
    print("█" * 100)
    for floor in [0.30, 0.50, 0.75, 1.00]:
        trades, bal = run_scenario(signals, tick_cache,
                                   f"Floor ${floor:.2f}",
                                   premium_floor=floor)
        print_scenario(f"Premium Floor ≥ ${floor:.2f}", trades, bal)

    # ── LEVER 2: Max Loss Cap ──
    print("\n\n" + "█" * 100)
    print("  LEVER 2: MAX LOSS CAP — cap max $ loss per trade as % of portfolio")
    print("█" * 100)
    for cap_pct in [3.0, 5.0, 7.0, 10.0]:
        trades, bal = run_scenario(signals, tick_cache,
                                   f"MaxLoss {cap_pct}%",
                                   max_loss_pct=cap_pct)
        print_scenario(f"Max Loss Cap {cap_pct}% of portfolio", trades, bal)

    # ── LEVER 3: Tighter Stops ──
    print("\n\n" + "█" * 100)
    print("  LEVER 3: TIGHTER STOPS — adjust V5 stop thresholds")
    print("█" * 100)

    # Baseline V5 config for reference
    base_cfg = V5Config()

    # Tighter graduated stop (default is 35% for 0DTE, 30% checkpoint)
    configs = [
        ("Stop25/CP20", {"tight_stop_0dte_pct": 25.0, "checkpoint_drop_pct": 20.0}),
        ("Stop30/CP25", {"tight_stop_0dte_pct": 30.0, "checkpoint_drop_pct": 25.0}),
        ("Stop40/CP35", {"tight_stop_0dte_pct": 40.0, "checkpoint_drop_pct": 35.0}),
        ("Stop45/CP40", {"tight_stop_0dte_pct": 45.0, "checkpoint_drop_pct": 40.0}),
    ]
    for label, overrides in configs:
        from dataclasses import replace
        cfg = replace(base_cfg, **overrides)
        trades, bal = run_scenario(signals, tick_cache, label, v5_config=cfg)
        print_scenario(f"{label} (default stop={base_cfg.tight_stop_0dte_pct:.0f}% cp={base_cfg.checkpoint_drop_pct:.0f}%)",
                       trades, bal)

    # ── COMBINED: Best of each lever ──
    print("\n\n" + "█" * 100)
    print("  COMBINED SCENARIOS — stacking levers")
    print("█" * 100)

    # Floor $0.50 + MaxLoss 5%
    trades, bal = run_scenario(signals, tick_cache,
                               "Floor+MaxLoss",
                               premium_floor=0.50, max_loss_pct=5.0)
    print_scenario("Floor ≥$0.50 + MaxLoss 5%", trades, bal)

    # Floor $0.50 + MaxLoss 5% + Tighter stops
    from dataclasses import replace
    cfg = replace(V5Config(), tight_stop_0dte_pct=30.0, checkpoint_drop_pct=25.0)
    trades, bal = run_scenario(signals, tick_cache,
                               "Floor+MaxLoss+TighterStop",
                               premium_floor=0.50, max_loss_pct=5.0,
                               v5_config=cfg)
    print_scenario("Floor ≥$0.50 + MaxLoss 5% + Stop30/CP25", trades, bal)

    # Floor $0.30 + MaxLoss 7%
    trades, bal = run_scenario(signals, tick_cache,
                               "Floor+MaxLoss7",
                               premium_floor=0.30, max_loss_pct=7.0)
    print_scenario("Floor ≥$0.30 + MaxLoss 7%", trades, bal)

    # Floor $0.50 + MaxLoss 3%
    trades, bal = run_scenario(signals, tick_cache,
                               "Floor+MaxLoss3",
                               premium_floor=0.50, max_loss_pct=3.0)
    print_scenario("Floor ≥$0.50 + MaxLoss 3%", trades, bal)


if __name__ == "__main__":
    main()
