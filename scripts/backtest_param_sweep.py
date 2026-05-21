"""Parameter sweep backtest — tests multiple adjustments to find optimal values.

Sweeps:
  1. Score floor: 78, 80, 83, 85
  2. Time-of-day sizing reduction: none, 11AM half, 11AM quarter, noon cutoff
  3. Max position cost cap: none, $2000, $1500, $1000
  4. Max concurrent: 3, 4, 5
  5. Max position pct: 10, 12, 15

Uses the LIVE production V5 FSM code. Each scenario runs the full backtest.

Usage:
    python scripts/backtest_param_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

_V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True,
    V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True,
    V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333,
    V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True,
    V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15,
    ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_PREMIUM_CAP=True,
    V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True,
    V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 23000


# ── Data loading ─────────────────────────────────────────────────────────────


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
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume
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


# ── Momentum gate (same as production backtest) ─────────────────────────────


def check_momentum(df, direction):
    is_call = direction in ("bullish", "call")
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


# ── FSM simulation ───────────────────────────────────────────────────────────


def simulate_with_production_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM"):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
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

    locked_pnl = 0.0
    remaining = contracts

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
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
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
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


# ── Scenario runner ──────────────────────────────────────────────────────────


# Production score tiers
SCORE_TIERS = [
    (135, 1.00, 0.15),
    (120, 0.85, 0.12),
    (100, 0.85, 0.08),
    (90, 0.50, 0.08),
    (78, 0.25, 0.08),
]


def get_et_hour(created_at_str):
    """Extract ET hour from UTC timestamp string."""
    try:
        if "T" in created_at_str:
            dt = datetime.strptime(created_at_str[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            dt = datetime.strptime(created_at_str[:19], "%Y-%m-%d %H:%M:%S")
        et_hour = dt.hour - 4
        if et_hour < 0:
            et_hour += 24
        return et_hour, dt.minute
    except (ValueError, TypeError):
        return 10, 0  # default to safe morning time


def run_scenario(signals, tick_cache, params):
    """Run one backtest scenario with given parameters.

    params dict keys:
      score_floor: int (78, 80, 83, 85)
      tod_mode: str ("none", "11am_half", "11am_quarter", "noon_cutoff", "11am_cutoff")
      max_cost_cap: float or None (None=no cap, 2000, 1500, 1000)
      max_concurrent: int (3, 4, 5)
      max_position_pct: float (0.10, 0.12, 0.15)
    """
    score_floor = params["score_floor"]
    tod_mode = params["tod_mode"]
    max_cost_cap = params["max_cost_cap"]
    max_concurrent = params["max_concurrent"]
    max_position_pct = params["max_position_pct"]

    results = []
    skipped_score = 0
    skipped_momentum = 0
    skipped_premium = 0
    skipped_spread = 0
    skipped_cost_cap = 0
    skipped_tod = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        entry_premium = sig["premium"]

        # Score floor
        if score < score_floor:
            skipped_score += 1
            continue

        cache_key = sig["id"]
        if cache_key not in tick_cache:
            continue
        df, dte, expiry_date = tick_cache[cache_key]

        # Entry price from harvester
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = entry_premium

        # V6 premium cap
        cap = 6.0
        if score >= 150:
            cap = 9.0
        elif score >= 120:
            cap = 7.0
        if adj_entry > cap:
            skipped_premium += 1
            continue

        # V6 spread gate
        first_bid = df["bid"].iloc[0]
        first_ask_val = df["ask"].iloc[0]
        if first_bid and first_ask_val and first_bid > 0 and first_ask_val > 0:
            spread_pct = (first_ask_val - first_bid) / first_ask_val * 100
            if spread_pct > 15.0:
                skipped_spread += 1
                continue

        # Momentum gate
        if check_momentum(df, direction):
            skipped_momentum += 1
            continue

        # Position sizing
        max_risk_pct = 0.75
        deployable = PORTFOLIO * max_risk_pct
        per_slot = deployable / max_concurrent

        score_mult = 0.25
        tier_pos_pct = 0.08
        for threshold, mult, pos_pct in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                tier_pos_pct = pos_pct
                break

        effective_pos_pct = max(tier_pos_pct, max_position_pct)
        position_cap = PORTFOLIO * effective_pos_pct

        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        if pos_cap_contracts == 0:
            continue
        contracts = max(1, min(raw_contracts, pos_cap_contracts))

        # Time-of-day sizing adjustment
        et_hour, et_min = get_et_hour(sig["created_at"])
        et_decimal = et_hour + et_min / 60.0

        if tod_mode == "11am_half" and et_decimal >= 11.0:
            contracts = max(1, contracts // 2)
        elif tod_mode == "11am_quarter" and et_decimal >= 11.0:
            contracts = max(1, contracts // 4)
        elif tod_mode == "noon_cutoff" and et_decimal >= 12.0:
            skipped_tod += 1
            continue
        elif tod_mode == "11am_cutoff" and et_decimal >= 11.0:
            skipped_tod += 1
            continue
        elif tod_mode == "11am_half_noon_stop":
            if et_decimal >= 12.0:
                skipped_tod += 1
                continue
            elif et_decimal >= 11.0:
                contracts = max(1, contracts // 2)
        elif tod_mode == "gradual":
            # Gradual reduction: 100% before 10:30, 75% at 11, 50% at 11:30, 25% at 12+
            if et_decimal >= 12.0:
                contracts = max(1, contracts // 4)
            elif et_decimal >= 11.5:
                contracts = max(1, contracts // 2)
            elif et_decimal >= 11.0:
                contracts = max(1, int(contracts * 0.75))

        # Late-session 0DTE reduction (existing production logic)
        if dte == 0 and contracts > 1:
            if et_decimal >= 14.0:  # 2 PM ET
                contracts = 1
            elif et_decimal >= 13.0:  # 1 PM ET
                contracts = max(1, contracts // 2)

        # Max position cost cap
        total_cost = contracts * cost_per
        if max_cost_cap is not None and total_cost > max_cost_cap:
            contracts = max(1, int(max_cost_cap / cost_per))
            total_cost = contracts * cost_per
            if total_cost > max_cost_cap and contracts == 1:
                # Even 1 contract exceeds cap — skip only if cost_cap is very tight
                # Keep 1 contract — the premium cap already limits per-contract cost
                pass

        result = simulate_with_production_fsm(
            df, adj_entry, contracts, direction, dte, expiry_date, ticker=ticker
        )
        result["ticker"] = ticker
        result["day"] = day
        result["score"] = score
        result["entry"] = adj_entry
        result["contracts"] = contracts
        result["direction"] = direction
        result["dte"] = dte
        result["et_hour"] = et_hour
        result["total_cost"] = contracts * cost_per
        results.append(result)

    return {
        "results": results,
        "skipped_score": skipped_score,
        "skipped_momentum": skipped_momentum,
        "skipped_premium": skipped_premium,
        "skipped_spread": skipped_spread,
        "skipped_cost_cap": skipped_cost_cap,
        "skipped_tod": skipped_tod,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals loaded")

    print("Loading tick data (this takes a minute)...")
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    tick_cache = {}
    no_data = 0
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is not None:
            tick_cache[sig["id"]] = (df, sig.get("_dte", 0), sig.get("_expiry_date", ""))
        else:
            no_data += 1
    harvester_conn.close()
    print(f"  {len(tick_cache)} signals have tick data, {no_data} skipped")

    # ── Define sweep parameters ──────────────────────────────────────────

    # Individual dimension sweeps (test one parameter at a time vs baseline)
    baseline = {
        "score_floor": 78,
        "tod_mode": "none",
        "max_cost_cap": None,
        "max_concurrent": 4,
        "max_position_pct": 0.15,
    }

    scenarios = []

    # 1. Score floor sweep
    for sf in [78, 80, 83, 85, 88, 90]:
        p = {**baseline, "score_floor": sf}
        p["_name"] = f"score_floor={sf}"
        p["_group"] = "score_floor"
        scenarios.append(p)

    # 2. Time-of-day sizing
    for tod in ["none", "11am_half", "11am_quarter", "gradual", "11am_half_noon_stop", "noon_cutoff", "11am_cutoff"]:
        p = {**baseline, "tod_mode": tod}
        p["_name"] = f"tod={tod}"
        p["_group"] = "time_of_day"
        scenarios.append(p)

    # 3. Max position cost cap
    for cap in [None, 2500, 2000, 1500, 1000, 750]:
        p = {**baseline, "max_cost_cap": cap}
        label = f"${cap}" if cap else "none"
        p["_name"] = f"cost_cap={label}"
        p["_group"] = "cost_cap"
        scenarios.append(p)

    # 4. Max concurrent trades
    for mc in [2, 3, 4, 5, 6]:
        p = {**baseline, "max_concurrent": mc}
        p["_name"] = f"max_concurrent={mc}"
        p["_group"] = "max_concurrent"
        scenarios.append(p)

    # 5. Max position pct
    for mp in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        p = {**baseline, "max_position_pct": mp}
        p["_name"] = f"max_pos={mp:.0%}"
        p["_group"] = "max_position"
        scenarios.append(p)

    # 6. Combined best candidates (will fill in after seeing individual results)
    combos = [
        {"score_floor": 83, "tod_mode": "gradual", "max_cost_cap": 2000,
         "max_concurrent": 4, "max_position_pct": 0.12,
         "_name": "combo_conservative", "_group": "combos"},
        {"score_floor": 80, "tod_mode": "11am_half", "max_cost_cap": 2000,
         "max_concurrent": 4, "max_position_pct": 0.15,
         "_name": "combo_moderate", "_group": "combos"},
        {"score_floor": 78, "tod_mode": "gradual", "max_cost_cap": 1500,
         "max_concurrent": 4, "max_position_pct": 0.10,
         "_name": "combo_tight", "_group": "combos"},
        {"score_floor": 85, "tod_mode": "11am_half", "max_cost_cap": None,
         "max_concurrent": 4, "max_position_pct": 0.15,
         "_name": "combo_high_floor", "_group": "combos"},
        {"score_floor": 80, "tod_mode": "11am_half_noon_stop", "max_cost_cap": 2000,
         "max_concurrent": 4, "max_position_pct": 0.12,
         "_name": "combo_balanced", "_group": "combos"},
    ]
    scenarios.extend(combos)

    # ── Run all scenarios ────────────────────────────────────────────────

    print(f"\nRunning {len(scenarios)} scenarios...")
    all_results = []

    for i, params in enumerate(scenarios):
        name = params.pop("_name")
        group = params.pop("_group")
        outcome = run_scenario(signals, tick_cache, params)
        r = outcome["results"]

        if r:
            df_r = pd.DataFrame(r)
            pnls = df_r["pnl"]
            total_pnl = pnls.sum()
            wins = (pnls > 0).sum()
            losses = (pnls <= 0).sum()
            win_rate = wins / len(pnls) * 100
            avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
            avg_loss = pnls[pnls <= 0].mean() if losses > 0 else 0
            max_loss = pnls.min()
            max_win = pnls.max()
            avg_cost = df_r["total_cost"].mean()
            max_cost = df_r["total_cost"].max()

            # Sharpe-like: avg daily pnl / std daily pnl
            daily_pnl = df_r.groupby("day")["pnl"].sum()
            sharpe = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0

            # Max drawdown from cumulative daily P&L
            cum = daily_pnl.cumsum()
            running_max = cum.cummax()
            drawdown = (cum - running_max).min()

            # Late-morning P&L (trades entered 11 AM+ ET)
            late = df_r[df_r["et_hour"] >= 11]
            late_pnl = late["pnl"].sum() if len(late) > 0 else 0
            late_count = len(late)
        else:
            total_pnl = 0
            wins = losses = 0
            win_rate = avg_win = avg_loss = max_loss = max_win = 0
            avg_cost = max_cost = 0
            sharpe = drawdown = 0
            late_pnl = late_count = 0

        row = {
            "group": group,
            "name": name,
            "trades": len(r),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "sharpe": sharpe,
            "max_dd": drawdown,
            "avg_cost": avg_cost,
            "max_cost": max_cost,
            "late_pnl": late_pnl,
            "late_trades": late_count,
            "skip_score": outcome["skipped_score"],
            "skip_mom": outcome["skipped_momentum"],
            "skip_tod": outcome["skipped_tod"],
        }
        all_results.append(row)
        params["_name"] = name
        params["_group"] = group

        pct = (i + 1) / len(scenarios) * 100
        print(f"  [{pct:5.1f}%] {name:<35} → ${total_pnl:>9,.2f}  {len(r)} trades  "
              f"{win_rate:.0f}% WR  late=${late_pnl:>7,.2f}")

    # ── Summary tables ───────────────────────────────────────────────────

    df_all = pd.DataFrame(all_results)

    # Find baseline P&L for comparison
    baseline_row = df_all[df_all["name"] == "score_floor=78"]
    baseline_pnl = baseline_row["total_pnl"].iloc[0] if len(baseline_row) > 0 else 0

    print(f"\n{'=' * 130}")
    print(f"PARAMETER SWEEP RESULTS — Baseline P&L: ${baseline_pnl:,.2f}")
    print(f"{'=' * 130}")

    for group_name in ["score_floor", "time_of_day", "cost_cap", "max_concurrent", "max_position", "combos"]:
        group = df_all[df_all["group"] == group_name]
        if len(group) == 0:
            continue

        print(f"\n── {group_name.upper()} ─────────────────────────────────────────────────")
        print(f"{'Scenario':<35} {'Trades':>6} {'Total P&L':>11} {'vs Base':>9} "
              f"{'Win%':>5} {'AvgWin':>8} {'AvgLoss':>8} {'MaxLoss':>8} "
              f"{'Sharpe':>6} {'MaxDD':>8} {'LatePnL':>9} {'LateTr':>6}")
        print("-" * 130)

        for _, r in group.sort_values("total_pnl", ascending=False).iterrows():
            diff = r["total_pnl"] - baseline_pnl
            marker = " ***" if r["total_pnl"] == group["total_pnl"].max() else ""
            print(f"{r['name']:<35} {r['trades']:>6} ${r['total_pnl']:>9,.2f} "
                  f"${diff:>+7,.0f} {r['win_rate']:>4.0f}% "
                  f"${r['avg_win']:>6,.0f} ${r['avg_loss']:>6,.0f} "
                  f"${r['max_loss']:>6,.0f} {r['sharpe']:>5.2f} "
                  f"${r['max_dd']:>6,.0f} ${r['late_pnl']:>7,.0f} {r['late_trades']:>5}{marker}")

    # ── Best overall ─────────────────────────────────────────────────────

    print(f"\n{'=' * 130}")
    print("TOP 10 SCENARIOS BY TOTAL P&L")
    print(f"{'=' * 130}")
    top10 = df_all.sort_values("total_pnl", ascending=False).head(10)
    print(f"{'Rank':>4} {'Scenario':<35} {'Trades':>6} {'Total P&L':>11} {'vs Base':>9} "
          f"{'Win%':>5} {'Sharpe':>6} {'MaxDD':>8} {'AvgCost':>8} {'MaxCost':>8}")
    print("-" * 110)
    for rank, (_, r) in enumerate(top10.iterrows(), 1):
        diff = r["total_pnl"] - baseline_pnl
        print(f"{rank:>4} {r['name']:<35} {r['trades']:>6} ${r['total_pnl']:>9,.2f} "
              f"${diff:>+7,.0f} {r['win_rate']:>4.0f}% {r['sharpe']:>5.2f} "
              f"${r['max_dd']:>6,.0f} ${r['avg_cost']:>6,.0f} ${r['max_cost']:>6,.0f}")

    # ── Risk-adjusted best (Sharpe) ──────────────────────────────────────

    print(f"\n{'=' * 130}")
    print("TOP 10 SCENARIOS BY SHARPE RATIO (risk-adjusted)")
    print(f"{'=' * 130}")
    top_sharpe = df_all[df_all["trades"] > 50].sort_values("sharpe", ascending=False).head(10)
    print(f"{'Rank':>4} {'Scenario':<35} {'Trades':>6} {'Total P&L':>11} "
          f"{'Win%':>5} {'Sharpe':>6} {'MaxDD':>8}")
    print("-" * 85)
    for rank, (_, r) in enumerate(top_sharpe.iterrows(), 1):
        print(f"{rank:>4} {r['name']:<35} {r['trades']:>6} ${r['total_pnl']:>9,.2f} "
              f"{r['win_rate']:>4.0f}% {r['sharpe']:>5.2f} ${r['max_dd']:>6,.0f}")

    # ── Recommendation ───────────────────────────────────────────────────

    print(f"\n{'=' * 130}")
    print("RECOMMENDATION")
    print(f"{'=' * 130}")

    # Find best in each dimension
    for group_name in ["score_floor", "time_of_day", "cost_cap", "max_concurrent", "max_position"]:
        group = df_all[df_all["group"] == group_name]
        best = group.loc[group["total_pnl"].idxmax()]
        print(f"  Best {group_name:<15}: {best['name']:<30} ${best['total_pnl']:>9,.2f} "
              f"({best['win_rate']:.0f}% WR, Sharpe {best['sharpe']:.2f})")

    best_combo = df_all[df_all["group"] == "combos"]
    if len(best_combo) > 0:
        best_c = best_combo.loc[best_combo["total_pnl"].idxmax()]
        print(f"\n  Best combo:          {best_c['name']:<30} ${best_c['total_pnl']:>9,.2f} "
              f"({best_c['win_rate']:.0f}% WR, Sharpe {best_c['sharpe']:.2f})")

    overall_best = df_all.loc[df_all["total_pnl"].idxmax()]
    print(f"\n  Overall best:        {overall_best['name']:<30} ${overall_best['total_pnl']:>9,.2f} "
          f"({overall_best['win_rate']:.0f}% WR, Sharpe {overall_best['sharpe']:.2f})")


if __name__ == "__main__":
    main()
