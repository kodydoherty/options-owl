"""Backtest each team recommendation individually against production baseline.

Tests each recommendation from WEBULL-AGENT-IMPROVEMENT-BRIEF.md and
WEBULL-EXIT-LOGIC-OVERHAUL-SPEC.md one at a time, using the full production
V5 FSM as baseline.

Usage:
    python scripts/backtest_team_recommendations.py
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

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import get_ticker_config
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
            SELECT captured_at, midpoint, bid, ask, underlying_price
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
        "captured_at", "midpoint", "bid", "ask", "underlying_price"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM",
                 max_hold_minutes=None, commit_zone_exit=False, quick_take_pct=None):
    """Run production FSM with optional overrides for A/B testing."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "peak_prem": 0, "slippage_pct": 0}

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
    peak_prem = entry_premium

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        peak_prem = max(peak_prem, premium)

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
        elapsed_min = (now - entry_ts).total_seconds() / 60

        # ── OVERRIDE: Max hold cutoff ──
        if max_hold_minutes and elapsed_min >= max_hold_minutes:
            peak_gain = (peak_prem - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": f"max_hold_{max_hold_minutes}min",
                    "hold": elapsed_min, "exit_prem": premium,
                    "peak_gain": peak_gain, "peak_prem": peak_prem, "slippage_pct": 0}

        # ── OVERRIDE: Commit zone (exit if underwater at 15 min) ──
        if commit_zone_exit and 14.5 <= elapsed_min <= 16.0:
            pnl_pct = (premium - entry_premium) / entry_premium * 100
            if pnl_pct <= 0:
                peak_gain = (peak_prem - entry_premium) / entry_premium * 100
                pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                return {"pnl": pnl, "reason": "commit_check_underwater",
                        "hold": elapsed_min, "exit_prem": premium,
                        "peak_gain": peak_gain, "peak_prem": peak_prem, "slippage_pct": 0}

        # ── OVERRIDE: Quick take (+25% in first 15 min) ──
        if quick_take_pct and elapsed_min <= 15.0:
            gain_pct = (premium - entry_premium) / entry_premium * 100
            if gain_pct >= quick_take_pct:
                peak_gain = (peak_prem - entry_premium) / entry_premium * 100
                pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                return {"pnl": pnl, "reason": f"quick_take_{quick_take_pct}pct",
                        "hold": elapsed_min, "exit_prem": premium,
                        "peak_gain": peak_gain, "peak_prem": peak_prem, "slippage_pct": 0}

        # ── Standard V5 FSM ──
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

            peak_gain = (peak_prem - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed_min,
                    "exit_prem": premium, "peak_gain": peak_gain,
                    "peak_prem": peak_prem, "slippage_pct": 0}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain,
            "peak_prem": peak_prem, "slippage_pct": 0}


def check_momentum_gate(df, direction):
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


def prepare_trades():
    """Load all tradeable signals with tick data, apply entry gates."""
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    trades = []

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

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
            fb = df["bid"].iloc[0]
            fa = df["ask"].iloc[0]
            if fb and fa and fb > 0 and fa > 0:
                spread_pct = (fa - fb) / fa * 100
                if spread_pct > _V6_SETTINGS.V6_MAX_SPREAD_PCT:
                    continue

        # Momentum gate
        if check_momentum_gate(df, direction):
            continue

        # Sizing
        max_risk_pct = 0.75
        max_concurrent = 4
        max_position_pct = 0.10
        deployable = PORTFOLIO * max_risk_pct
        per_slot = deployable / max_concurrent

        score_mult = 0.25
        tier_pos_pct = 0.08
        for threshold, mult, pos_pct in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                tier_pos_pct = pos_pct
                break

        effective_pos_pct = min(tier_pos_pct, max_position_pct)
        position_cap = PORTFOLIO * effective_pos_pct
        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        if pos_cap_contracts == 0:
            continue
        contracts = max(1, min(raw_contracts, pos_cap_contracts))

        # Late session reduction
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

        # Compute entry slippage (signal premium vs actual entry)
        signal_prem = sig["premium"]
        slippage_pct = (adj_entry - signal_prem) / signal_prem * 100 if signal_prem > 0 else 0

        trades.append({
            "ticker": ticker, "day": day, "score": score,
            "entry": adj_entry, "signal_premium": signal_prem,
            "contracts": contracts, "direction": direction,
            "dte": dte, "expiry_date": expiry_date,
            "df": df, "slippage_pct": slippage_pct,
            "option_type": "put" if direction in ("bearish", "put") else "call",
        })

    harvester_conn.close()
    return trades


def run_scenario(trades, label, **fsm_kwargs):
    """Run a scenario across all trades with optional FSM overrides."""
    results = []
    filter_puts = fsm_kwargs.pop("filter_puts", False)
    halve_puts = fsm_kwargs.pop("halve_puts", False)

    for t in trades:
        if filter_puts and t["option_type"] == "put":
            continue

        contracts = t["contracts"]
        if halve_puts and t["option_type"] == "put":
            contracts = max(1, contracts // 2)

        result = simulate_fsm(
            t["df"], t["entry"], contracts, t["direction"],
            t["dte"], t["expiry_date"], ticker=t["ticker"],
            **fsm_kwargs,
        )
        result["slippage_pct"] = t["slippage_pct"]
        result["ticker"] = t["ticker"]
        result["day"] = t["day"]
        result["score"] = t["score"]
        result["direction"] = t["direction"]
        result["option_type"] = t["option_type"]
        result["entry"] = t["entry"]
        result["contracts"] = contracts
        results.append(result)

    return results


def summarize(results, label):
    if not results:
        print(f"\n{label}: NO TRADES")
        return {}

    df = pd.DataFrame(results)
    pnls = df["pnl"]
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    total_pnl = pnls.sum()
    wr = wins / len(pnls) * 100
    daily = df.groupby("day")["pnl"].sum()
    green = (daily > 0).sum()
    win_avg = pnls[pnls > 0].mean() if wins > 0 else 0
    loss_avg = abs(pnls[pnls <= 0].mean()) if losses > 0 else 1

    # Peak capture: what % of peak gain did we actually realize?
    capture = []
    for _, r in df.iterrows():
        if r["peak_gain"] > 0:
            exit_gain = (r["exit_prem"] - r["entry"]) / r["entry"] * 100
            cap = exit_gain / r["peak_gain"] * 100
            capture.append(cap)
    median_capture = np.median(capture) if capture else 0
    mean_capture = np.mean(capture) if capture else 0

    # Direction breakdown
    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"Trades:       {len(results)}")
    print(f"Total P&L:    ${total_pnl:>10,.2f}   Win Rate: {wr:.1f}%  ({wins}W/{losses}L)")
    print(f"Win:Loss:     {win_avg/loss_avg:.2f}:1       Green Days: {green}/{len(daily)}")
    print(f"Max Win:      ${pnls.max():>10,.2f}   Max Loss: ${pnls.min():>10,.2f}")
    print(f"Worst Day:    ${daily.min():>10,.2f}")
    print(f"Avg Hold:     {df['hold'].mean():.0f} min")
    print(f"Peak Capture: median={median_capture:.1f}% mean={mean_capture:.1f}%")

    if len(calls) > 0:
        c_wr = (calls["pnl"] > 0).sum() / len(calls) * 100
        print(f"  Calls:      {len(calls)} trades, ${calls['pnl'].sum():>8,.2f}, WR={c_wr:.0f}%")
    if len(puts) > 0:
        p_wr = (puts["pnl"] > 0).sum() / len(puts) * 100
        print(f"  Puts:       {len(puts)} trades, ${puts['pnl'].sum():>8,.2f}, WR={p_wr:.0f}%")

    # Hold time buckets (matching their analysis)
    print(f"\n  Hold Time Distribution:")
    for lo, hi, lbl in [(0, 5, "0-5min"), (5, 15, "5-15min"), (15, 30, "15-30min"),
                         (30, 45, "30-45min"), (45, 75, "45-75min"), (75, 999, "75+min")]:
        bucket = df[(df["hold"] >= lo) & (df["hold"] < hi)]
        if len(bucket) > 0:
            bwr = (bucket["pnl"] > 0).sum() / len(bucket) * 100
            print(f"    {lbl:>10}: n={len(bucket):>3}, WR={bwr:.0f}%, "
                  f"P&L=${bucket['pnl'].sum():>8,.2f}, avg=${bucket['pnl'].mean():>6,.2f}")

    return {
        "label": label, "trades": len(results), "pnl": total_pnl,
        "wr": wr, "green_days": f"{green}/{len(daily)}",
        "worst_day": daily.min(), "max_loss": pnls.min(),
        "win_loss_ratio": win_avg / loss_avg if loss_avg > 0 else 0,
        "median_capture": median_capture, "mean_capture": mean_capture,
        "calls_pnl": calls["pnl"].sum() if len(calls) > 0 else 0,
        "puts_pnl": puts["pnl"].sum() if len(puts) > 0 else 0,
    }


def main():
    print("Loading trades with full production gates...")
    trades = prepare_trades()
    print(f"Loaded {len(trades)} tradeable signals with tick data\n")

    # Compute slippage stats
    slippages = [t["slippage_pct"] for t in trades]
    print(f"ENTRY SLIPPAGE (signal premium vs actual fill):")
    print(f"  Mean:   {np.mean(slippages):+.2f}%")
    print(f"  Median: {np.median(slippages):+.2f}%")
    print(f"  Std:    {np.std(slippages):.2f}%")
    puts_slip = [t["slippage_pct"] for t in trades if t["option_type"] == "put"]
    calls_slip = [t["slippage_pct"] for t in trades if t["option_type"] == "call"]
    if puts_slip:
        print(f"  Puts:   {np.mean(puts_slip):+.2f}% mean")
    if calls_slip:
        print(f"  Calls:  {np.mean(calls_slip):+.2f}% mean")

    all_summaries = []

    # ── BASELINE: Production V5 FSM (no changes) ──
    baseline = run_scenario(trades, "BASELINE")
    s = summarize(baseline, "BASELINE: Production V5 FSM (no changes)")
    all_summaries.append(s)

    # ── TEST 1: 45-min max hold (their core recommendation) ──
    test1 = run_scenario(trades, "45MIN_MAX_HOLD", max_hold_minutes=45)
    s = summarize(test1, "TEST 1: 45-min max hold (team recommendation)")
    all_summaries.append(s)

    # ── TEST 2: 30-min max hold (bearish-specific in their spec) ──
    test2 = run_scenario(trades, "30MIN_MAX_HOLD", max_hold_minutes=30)
    s = summarize(test2, "TEST 2: 30-min max hold")
    all_summaries.append(s)

    # ── TEST 3: Commit zone (exit if underwater at T+15) ──
    test3 = run_scenario(trades, "COMMIT_ZONE", commit_zone_exit=True)
    s = summarize(test3, "TEST 3: Commit zone — exit if underwater at T+15")
    all_summaries.append(s)

    # ── TEST 4: Quick take +25% in first 15 min ──
    test4 = run_scenario(trades, "QUICK_TAKE_25", quick_take_pct=25.0)
    s = summarize(test4, "TEST 4: Quick take — close at +25% in first 15min")
    all_summaries.append(s)

    # ── TEST 5: Block all puts ──
    test5 = run_scenario(trades, "NO_PUTS", filter_puts=True)
    s = summarize(test5, "TEST 5: Block all puts (bearish underperformance)")
    all_summaries.append(s)

    # ── TEST 6: Halve put sizing ──
    test6 = run_scenario(trades, "HALVE_PUTS", halve_puts=True)
    s = summarize(test6, "TEST 6: Halve put contract count")
    all_summaries.append(s)

    # ── TEST 7: Commit zone + 45-min max hold (combined) ──
    test7 = run_scenario(trades, "COMMIT+45MIN",
                          commit_zone_exit=True, max_hold_minutes=45)
    s = summarize(test7, "TEST 7: Commit zone + 45-min max hold (combined)")
    all_summaries.append(s)

    # ── HEAD-TO-HEAD COMPARISON TABLE ──
    print("\n\n")
    print("=" * 100)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 100)
    print(f"{'Scenario':<45} {'Trades':>6} {'P&L':>11} {'WR':>6} {'W:L':>5} "
          f"{'Cap%':>5} {'WrstDay':>9} {'Puts$':>9}")
    print("-" * 100)
    for s in all_summaries:
        if not s:
            continue
        print(f"{s['label']:<45} {s['trades']:>6} ${s['pnl']:>9,.0f} "
              f"{s['wr']:>5.1f}% {s['win_loss_ratio']:>4.1f}x "
              f"{s['median_capture']:>4.0f}% ${s['worst_day']:>8,.0f} "
              f"${s['puts_pnl']:>8,.0f}")

    # ── Delta from baseline ──
    if all_summaries and all_summaries[0]:
        base_pnl = all_summaries[0]["pnl"]
        base_wr = all_summaries[0]["wr"]
        print(f"\n{'Scenario':<45} {'P&L Delta':>11} {'WR Delta':>9}")
        print("-" * 70)
        for s in all_summaries[1:]:
            if not s:
                continue
            dpnl = s["pnl"] - base_pnl
            dwr = s["wr"] - base_wr
            print(f"{s['label']:<45} ${dpnl:>+9,.0f} {dwr:>+8.1f}%")


if __name__ == "__main__":
    main()
