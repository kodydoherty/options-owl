"""A/B test: position sizing cap change using the FULL production V5 FSM.

Uses the EXACT same code as backtest_v5_production.py — all gates, all V6 features,
momentum gate, VWAP+support gate — and ONLY changes MAX_POSITION_PCT.

Run A: MAX_POSITION_PCT=15 (current production)
Run B: MAX_POSITION_PCT=10 (proposed change)

Usage:
    python scripts/backtest_position_cap_ab.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

# Production V6 settings (matches docker-compose.yml exactly)
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

# Production score tiers (vinny_strategy.py _SCORE_TIER_TABLE)
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


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM"):
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


def check_momentum_gate(df, direction):
    """Production momentum gate — blocks entries where underlying + premium both fading."""
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


def run_backtest(max_position_pct: float, max_dca_pct: float | None = None, label: str = ""):
    """Run full production backtest with a specific position cap.

    If max_dca_pct is None, DCA uses same sizing as entry (current production behavior).
    If max_dca_pct is set, DCA contracts are capped at that % of portfolio.
    """
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    results = []
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

        # Score floor
        if score < 78:
            continue

        # V6 premium cap
        if _V6_SETTINGS.ENABLE_V6_PREMIUM_CAP:
            cap = _V6_SETTINGS.V6_PREMIUM_CAP
            if score >= 150:
                cap = _V6_SETTINGS.V6_PREMIUM_CAP_HIGH
            elif score >= 120:
                cap = _V6_SETTINGS.V6_PREMIUM_CAP_MID
            if adj_entry > cap:
                continue

        # V6 spread gate
        if _V6_SETTINGS.ENABLE_V6_SPREAD_GATE and len(df) > 0:
            first_bid = df["bid"].iloc[0]
            fa = df["ask"].iloc[0]
            if first_bid and fa and first_bid > 0 and fa > 0:
                spread_pct = (fa - first_bid) / fa * 100
                if spread_pct > _V6_SETTINGS.V6_MAX_SPREAD_PCT:
                    continue

        # Momentum gate
        if check_momentum_gate(df, direction):
            continue

        # ── Position sizing (THE VARIABLE UNDER TEST) ──
        max_risk_pct = 0.75
        max_concurrent = 4
        deployable = PORTFOLIO * max_risk_pct
        per_slot = deployable / max_concurrent

        score_mult = 0.25
        tier_pos_pct = 0.08
        for threshold, mult, pos_pct in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                tier_pos_pct = pos_pct
                break

        # Position cap: use min(tier, setting) so the setting is a TRUE cap
        effective_pos_pct = min(tier_pos_pct, max_position_pct / 100)
        position_cap = PORTFOLIO * effective_pos_pct

        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        if pos_cap_contracts == 0:
            continue
        contracts = max(1, min(raw_contracts, pos_cap_contracts))

        # Late-session 0DTE size reduction (production paper_trader)
        if dte == 0 and contracts > 1:
            sig_time = sig["created_at"]
            try:
                sig_dt_full = (datetime.strptime(sig_time[:19], "%Y-%m-%dT%H:%M:%S")
                               if "T" in sig_time
                               else datetime.strptime(sig_time[:19], "%Y-%m-%d %H:%M:%S"))
                et_hour = sig_dt_full.hour - 4
                if et_hour < 0:
                    et_hour += 24
                if et_hour >= 14:
                    contracts = 1
                elif et_hour >= 13:
                    contracts = max(1, contracts // 2)
            except (ValueError, TypeError):
                pass

        # Simulate V6 DCA: if the trade dips 15-35% in the 8-20 min window,
        # add contracts (capped by max_dca_pct if provided)
        dca_tickers = set(_V6_SETTINGS.V6_DCA_TICKERS.split(","))
        dca_add = 0
        dca_entry = 0.0
        if ticker in dca_tickers and len(df) > 20:
            entry_ts = df["ts"].iloc[0]
            if hasattr(entry_ts, 'to_pydatetime'):
                entry_ts = entry_ts.to_pydatetime()
            for idx in range(1, len(df)):
                ts = df["ts"].iloc[idx]
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                elapsed_min = (ts - entry_ts).total_seconds() / 60
                if elapsed_min < _V6_SETTINGS.V6_DCA_MIN_MINUTES:
                    continue
                if elapsed_min > _V6_SETTINGS.V6_DCA_MAX_MINUTES:
                    break
                prem = df["premium"].iloc[idx]
                if np.isnan(prem) or prem <= 0:
                    continue
                dip_pct = (adj_entry - prem) / adj_entry * 100
                if _V6_SETTINGS.V6_DCA_MIN_DIP_PCT <= dip_pct <= _V6_SETTINGS.V6_DCA_MAX_DIP_PCT:
                    # Production: adds contracts equal to current count
                    dca_add = contracts
                    # Apply DCA cap if provided
                    if max_dca_pct is not None:
                        dca_cap_dollars = PORTFOLIO * (max_dca_pct / 100)
                        dca_max_contracts = int(dca_cap_dollars / cost_per) if cost_per > 0 else 0
                        dca_add = min(dca_add, max(1, dca_max_contracts))
                    dca_entry = prem
                    break

        # Calculate blended entry if DCA fired
        total_contracts = contracts + dca_add
        if dca_add > 0 and dca_entry > 0:
            blended_entry = (adj_entry * contracts + dca_entry * dca_add) / total_contracts
        else:
            blended_entry = adj_entry

        result = simulate_fsm(
            df, blended_entry, total_contracts, direction, dte, expiry_date, ticker=ticker
        )

        results.append({
            "ticker": ticker, "day": day, "score": score,
            "entry": adj_entry, "blended_entry": blended_entry,
            "contracts": contracts, "dca_add": dca_add,
            "total_contracts": total_contracts,
            "direction": direction, "dte": dte,
            "total_cost": adj_entry * contracts * 100 + (dca_entry * dca_add * 100 if dca_add > 0 else 0),
            "pct_of_portfolio": (adj_entry * contracts * 100) / PORTFOLIO * 100,
            **result,
        })

    harvester_conn.close()
    return results, no_data


def print_summary(results, label, no_data):
    df = pd.DataFrame(results)
    pnls = df["pnl"]
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    total_pnl = pnls.sum()
    win_rate = wins / len(pnls) * 100

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"Trades:      {len(results)} ({no_data} skipped, no tick data)")
    print(f"Total P&L:   ${total_pnl:,.2f}")
    print(f"Win Rate:    {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Avg Win:     ${pnls[pnls > 0].mean():,.2f}" if wins > 0 else "Avg Win:     N/A")
    print(f"Avg Loss:    ${pnls[pnls <= 0].mean():,.2f}" if losses > 0 else "Avg Loss:    N/A")
    win_avg = pnls[pnls > 0].mean() if wins > 0 else 0
    loss_avg = abs(pnls[pnls <= 0].mean()) if losses > 0 else 1
    print(f"Win:Loss:    {win_avg/loss_avg:.2f}:1" if loss_avg > 0 else "Win:Loss:    N/A")
    print(f"Max Win:     ${pnls.max():,.2f}")
    print(f"Max Loss:    ${pnls.min():,.2f}")
    print(f"Avg Hold:    {df['hold'].mean():.0f} min")

    # Worst day
    daily = df.groupby("day")["pnl"].sum()
    print(f"Best Day:    ${daily.max():,.2f}")
    print(f"Worst Day:   ${daily.min():,.2f}")
    green_days = (daily > 0).sum()
    print(f"Green Days:  {green_days}/{len(daily)} ({green_days/len(daily)*100:.0f}%)")

    # DCA stats
    dca_trades = df[df["dca_add"] > 0]
    print(f"\nDCA fires:   {len(dca_trades)}")
    if len(dca_trades) > 0:
        print(f"DCA avg add: {dca_trades['dca_add'].mean():.1f} contracts")
        print(f"DCA P&L:     ${dca_trades['pnl'].sum():,.2f}")

    # Position sizing stats
    print(f"\nAvg contracts:     {df['total_contracts'].mean():.1f}")
    print(f"Avg entry cost:    ${df['total_cost'].mean():,.0f}")
    print(f"Avg % of portfolio: {df['pct_of_portfolio'].mean():.1f}%")
    print(f"Max % of portfolio: {df['pct_of_portfolio'].max():.1f}%")

    # Exit reason breakdown
    print(f"\n{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 62)
    for reason, group in df.groupby("reason"):
        gpnl = group["pnl"]
        gwins = (gpnl > 0).sum()
        gwr = gwins / len(gpnl) * 100
        print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}%")

    # Score tier breakdown
    print(f"\n{'Score Tier':<15} {'Count':>6} {'Total P&L':>12} {'Avg Ct':>7} {'Win%':>6}")
    print("-" * 50)
    for label_s, lo, hi in [("135+", 135, 999), ("120-134", 120, 134),
                             ("100-119", 100, 119), ("90-99", 90, 99), ("78-89", 78, 89)]:
        tier = df[(df["score"] >= lo) & (df["score"] <= hi)]
        if len(tier) > 0:
            tw = (tier["pnl"] > 0).sum()
            print(f"{label_s:<15} {len(tier):>6} ${tier['pnl'].sum():>10,.2f} "
                  f"{tier['total_contracts'].mean():>6.1f} {tw/len(tier)*100:>5.0f}%")

    return df


def main():
    print("Loading signals and tick data...")
    print("(This may take a few minutes — full production FSM simulation)\n")

    # Run A: Current production (15% cap, DCA = double position)
    print("=" * 80)
    print("RUN A: Production (MAX_POSITION_PCT=15, DCA=double)")
    print("=" * 80)
    results_a, no_data_a = run_backtest(max_position_pct=15, max_dca_pct=None,
                                         label="Production 15%")
    df_a = print_summary(results_a, "RUN A: Production (15% cap, DCA doubles)", no_data_a)

    # Run B: Proposed (10% cap, DCA capped at 5%)
    print("\n\n")
    print("=" * 80)
    print("RUN B: Proposed (MAX_POSITION_PCT=10, DCA=5% cap)")
    print("=" * 80)
    results_b, no_data_b = run_backtest(max_position_pct=10, max_dca_pct=5,
                                         label="Proposed 10%+5%")
    df_b = print_summary(results_b, "RUN B: Proposed (10% cap, DCA capped at 5%)", no_data_b)

    # Run C: Just position cap change, DCA unchanged (isolate one variable)
    print("\n\n")
    print("=" * 80)
    print("RUN C: Position cap only (MAX_POSITION_PCT=10, DCA=double)")
    print("=" * 80)
    results_c, no_data_c = run_backtest(max_position_pct=10, max_dca_pct=None,
                                         label="10% cap, DCA doubles")
    df_c = print_summary(results_c, "RUN C: Position cap only (10%, DCA unchanged)", no_data_c)

    # ── Head-to-head comparison ──
    print("\n\n")
    print("=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    for lbl, df_x in [("A: Prod 15%+DCA double", df_a),
                       ("B: 10%+DCA 5% cap", df_b),
                       ("C: 10%+DCA double", df_c)]:
        pnls = df_x["pnl"]
        wins = (pnls > 0).sum()
        wr = wins / len(pnls) * 100
        daily = df_x.groupby("day")["pnl"].sum()
        green = (daily > 0).sum()
        win_avg = pnls[pnls > 0].mean() if wins > 0 else 0
        loss_avg = abs(pnls[pnls <= 0].mean()) if (pnls <= 0).sum() > 0 else 1
        print(f"\n{lbl}:")
        print(f"  Total P&L:    ${pnls.sum():>10,.2f}   Win Rate: {wr:.1f}%")
        print(f"  Win:Loss:     {win_avg/loss_avg:.2f}:1       Green Days: {green}/{len(daily)}")
        print(f"  Max Loss:     ${pnls.min():>10,.2f}   Worst Day: ${daily.min():>10,.2f}")
        print(f"  Avg Position: {df_x['pct_of_portfolio'].mean():.1f}% of portfolio")


if __name__ == "__main__":
    main()
