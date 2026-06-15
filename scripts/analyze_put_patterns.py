"""Analyze PUT scalp patterns across time-of-day, tickers, and market conditions.

Scans ThetaData PUT options at multiple entry times throughout the day,
simulates V5 FSM exits, and finds profitable patterns.

Schema:
  option_ohlc: ticker, expiration, strike, right, timestamp, open, high, low, close, volume, vwap
  stock_ohlc: ticker, timestamp, open, high, low, close, volume, vwap
  option_quotes: ticker, expiration, strike, right, timestamp, bid, ask, bid_size, ask_size

Usage:
    python scripts/analyze_put_patterns.py
    python scripts/analyze_put_patterns.py --days 60
    python scripts/analyze_put_patterns.py --ticker NVDA,TSLA
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ET = ZoneInfo("America/New_York")
THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "AMD", "MSTR", "PLTR", "AVGO",
]

# Entry time slots (minutes after 9:30 open)
ENTRY_SLOTS = [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]

V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True, V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333, V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True, V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15, ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0, V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True, V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True, ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True, SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)


def slot_label(minutes_after_open: int) -> str:
    h = 9 + (30 + minutes_after_open) // 60
    m = (30 + minutes_after_open) % 60
    ampm = "AM" if h < 12 else "PM"
    h12 = h if h <= 12 else h - 12
    return f"{h12}:{m:02d}{ampm}"


def get_trading_days(conn, start_date: str, end_date: str) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) as d
        FROM stock_ohlc WHERE ticker = 'SPY' AND d >= ? AND d <= ?
        ORDER BY d
    """, (start_date, end_date)).fetchall()
    return [r[0] for r in rows]


def get_underlying_price(conn, ticker: str, date_str: str, entry_time_prefix: str) -> float | None:
    """Get underlying stock price at a given time."""
    row = conn.execute("""
        SELECT close FROM stock_ohlc
        WHERE ticker = ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp LIMIT 1
    """, (ticker, entry_time_prefix + ":00", entry_time_prefix + ":59")).fetchone()
    return float(row[0]) if row else None


def simulate_put_trade(conn, ticker: str, date_str: str, slot_min: int) -> dict | None:
    """Find ATM PUT at given time, simulate V5 exit, return trade result."""
    entry_hour = 9 + (30 + slot_min) // 60
    entry_min_of_hour = (30 + slot_min) % 60
    entry_time_prefix = f"{date_str} {entry_hour:02d}:{entry_min_of_hour:02d}"

    # Get underlying price at entry time
    underlying = get_underlying_price(conn, ticker, date_str, entry_time_prefix)
    if not underlying:
        return None

    # Find ATM PUT closest to underlying price, with valid close
    atm = conn.execute("""
        SELECT strike, close, open, high, low, volume, expiration, timestamp
        FROM option_ohlc
        WHERE ticker = ? AND right = 'PUT'
          AND timestamp >= ? AND timestamp < ?
          AND close > 0.10 AND close < 6.0
        ORDER BY ABS(strike - ?)
        LIMIT 1
    """, (ticker, entry_time_prefix + ":00", entry_time_prefix + ":59", underlying)).fetchone()

    if not atm:
        return None

    strike, entry_close, opn, high, low, vol, expiration, entry_ts = atm

    # Use close as entry (no bid/ask in ohlc — close is mid-ish)
    # Add 5% slippage for realistic fill (buying at ask)
    entry_premium = entry_close * 1.05
    if entry_premium < 0.15 or entry_premium > 6.0:
        return None

    # DTE
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        dte = max(0, (exp_date - trade_date).days)
    except (ValueError, TypeError):
        dte = 0

    # Load subsequent ticks for this contract
    ticks = conn.execute("""
        SELECT timestamp, close, volume
        FROM option_ohlc
        WHERE ticker = ? AND right = 'PUT' AND strike = ?
          AND expiration = ? AND timestamp > ?
          AND substr(timestamp, 1, 10) = ?
        ORDER BY timestamp
    """, (ticker, strike, expiration, entry_ts, date_str)).fetchall()

    if len(ticks) < 3:
        return None

    # Build tick data and run FSM
    records = []
    for ts_str, close_val, vol_val in ticks:
        try:
            ts_dt = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue
        if not close_val or close_val <= 0:
            continue
        records.append({"timestamp": ts_dt, "premium": float(close_val), "volume": int(vol_val or 0)})

    if len(records) < 3:
        return None

    tick_df = pd.DataFrame(records)
    entry_dt = datetime.fromisoformat(entry_ts)

    # V5 FSM
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    contracts = 10

    state = TradeState(
        trade_id=0, ticker=ticker, option_type="put",
        entry_premium=entry_premium, entry_time=entry_dt,
        contracts=contracts, peak_premium=entry_premium,
        dte=dte, expiry_date=date_str,
        entry_underlying_price=underlying,
        last_underlying_price=underlying,
    )

    fsm = ExitFSM(cfg, settings=V6_SETTINGS)
    exit_reason = None
    exit_premium = entry_premium
    exit_time = None
    peak_gain = 0.0

    for _, row in tick_df.iterrows():
        prem = row["premium"]
        elapsed = (row["timestamp"] - entry_dt).total_seconds()

        state.peak_premium = max(state.peak_premium, prem)

        gain_pct = (prem - entry_premium) / entry_premium * 100
        peak_gain = max(peak_gain, gain_pct)

        # Compute minutes to close (4:00 PM ET)
        try:
            now_et = row["timestamp"]
            close_et = now_et.replace(hour=16, minute=0, second=0)
            min_to_close = max(0, (close_et - now_et).total_seconds() / 60)
        except Exception:
            min_to_close = 300

        action = fsm.evaluate(
            state, current_premium=prem,
            bid=prem * 0.95, ask=prem * 1.05,
            now_et=row["timestamp"],
            minutes_to_close=min_to_close,
        )
        if action and action.should_exit:
            exit_reason = action.reason.value if hasattr(action.reason, 'value') else str(action.reason)
            # Sell at bid (5% slippage from close)
            exit_premium = prem * 0.95
            exit_time = row["timestamp"]
            break

    if not exit_reason:
        exit_reason = "eod_data_end"
        exit_premium = tick_df["premium"].iloc[-1] * 0.95
        exit_time = tick_df["timestamp"].iloc[-1]

    pnl_per_contract = (exit_premium - entry_premium) * 100
    total_pnl = pnl_per_contract * contracts
    hold_min = (exit_time - entry_dt).total_seconds() / 60 if exit_time else 0

    return {
        "ticker": ticker, "date": date_str,
        "entry_minute": slot_min,
        "entry_time_label": slot_label(slot_min),
        "entry_premium": round(entry_premium, 2),
        "exit_premium": round(exit_premium, 2),
        "pnl": round(total_pnl, 2),
        "pnl_pct": round((exit_premium - entry_premium) / entry_premium * 100, 1),
        "peak_gain": round(peak_gain, 1),
        "hold_min": round(hold_min, 1),
        "exit_reason": exit_reason,
        "strike": strike, "underlying": underlying, "dte": dte,
    }


def main():
    parser = argparse.ArgumentParser(description="PUT Scalp Pattern Analysis")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default="2026-05-21")
    args = parser.parse_args()

    conn = sqlite3.connect(THETADATA_DB)

    if args.start:
        start_date = args.start
    else:
        all_days = get_trading_days(conn, "2020-01-01", args.end)
        start_date = all_days[-args.days] if len(all_days) >= args.days else all_days[0]

    trading_days = get_trading_days(conn, start_date, args.end)
    tickers = [t.strip().upper() for t in args.ticker.split(",")] if args.ticker else TICKERS

    print(f"\n{'='*80}")
    print(f"PUT SCALP PATTERN ANALYSIS")
    print(f"{'='*80}")
    print(f"  Period: {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Entry slots: {', '.join(slot_label(s) for s in ENTRY_SLOTS)}")
    total_scans = len(trading_days) * len(tickers) * len(ENTRY_SLOTS)
    print(f"  Total scans: {total_scans:,}")
    print()

    all_results = []

    for day_idx, date_str in enumerate(trading_days):
        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            print(f"  [{day_idx+1}/{len(trading_days)}] {date_str} — {len(all_results)} trades", flush=True)

        for ticker in tickers:
            for slot_min in ENTRY_SLOTS:
                result = simulate_put_trade(conn, ticker, date_str, slot_min)
                if result:
                    all_results.append(result)

    conn.close()

    if not all_results:
        print("No trades found!")
        return

    df = pd.DataFrame(all_results)

    # ── Overall Stats ──
    wins = len(df[df["pnl"] > 0])
    losses = len(df[df["pnl"] <= 0])
    total_pnl = df["pnl"].sum()
    wr = wins / len(df) * 100
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if wins else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if losses else 0

    print(f"\n{'='*80}")
    print(f"RESULTS — {len(df)} PUT trades analyzed")
    print(f"{'='*80}")
    print(f"  Overall: {wins}W/{losses}L, {wr:.1f}% WR, ${total_pnl:+,.0f} P&L")
    print(f"  Avg win: ${avg_win:+,.0f}, Avg loss: ${avg_loss:+,.0f}")
    print(f"  Avg hold: {df['hold_min'].mean():.0f}m, Avg peak: {df['peak_gain'].mean():.0f}%")

    # ── By Time of Day ──
    print(f"\n{'─'*80}")
    print(f"BY TIME OF DAY")
    print(f"{'─'*80}")
    print(f"{'Slot':>10} | {'N':>5} | {'WR%':>5} | {'Total P&L':>12} | {'Avg P&L':>10} | {'Hold':>5} | {'PF':>6} | {'Peak':>5}")
    print("-" * 80)

    for slot_min in ENTRY_SLOTS:
        s = df[df["entry_minute"] == slot_min]
        if s.empty:
            continue
        w = len(s[s["pnl"] > 0])
        l = len(s[s["pnl"] <= 0])
        wr_s = w / len(s) * 100
        gw = s[s["pnl"] > 0]["pnl"].sum() if w else 0
        gl = abs(s[s["pnl"] <= 0]["pnl"].sum()) if l else 1
        pf = gw / gl if gl > 0 else 99.9
        marker = " <== BEST" if wr_s >= 55 and pf >= 1.5 else ""
        print(f"  {slot_label(slot_min):>8} | {len(s):>5} | {wr_s:>4.0f}% | ${s['pnl'].sum():>+10,.0f} | "
              f"${s['pnl'].mean():>+8,.0f} | {s['hold_min'].mean():>3.0f}m | {pf:>5.1f} | {s['peak_gain'].mean():>3.0f}%{marker}")

    # ── By Ticker ──
    print(f"\n{'─'*80}")
    print(f"BY TICKER")
    print(f"{'─'*80}")
    print(f"{'Ticker':>8} | {'N':>5} | {'WR%':>5} | {'Total P&L':>12} | {'Avg P&L':>10} | {'Best Slot':>10}")
    print("-" * 65)

    for ticker in sorted(df["ticker"].unique()):
        t = df[df["ticker"] == ticker]
        w = len(t[t["pnl"] > 0])
        wr_t = w / len(t) * 100
        # Find best slot
        best_slot, best_avg = "N/A", -999
        for slot_min in ENTRY_SLOTS:
            c = t[t["entry_minute"] == slot_min]
            if len(c) >= 5 and c["pnl"].mean() > best_avg:
                best_avg = c["pnl"].mean()
                best_slot = slot_label(slot_min)
        print(f"  {ticker:>6} | {len(t):>5} | {wr_t:>4.0f}% | ${t['pnl'].sum():>+10,.0f} | "
              f"${t['pnl'].mean():>+8,.0f} | {best_slot:>9}")

    # ── Heatmap: Ticker × Time ──
    print(f"\n{'─'*80}")
    print(f"WIN RATE HEATMAP (min 5 trades/cell, ++ = WR>=60%, -- = WR<40%)")
    print(f"{'─'*80}")

    slot_labels = [slot_label(s)[:5] for s in ENTRY_SLOTS]
    print(f"{'':>8} |" + "|".join(f"{l:>7}" for l in slot_labels))
    print("-" * (10 + 8 * len(ENTRY_SLOTS)))

    for ticker in sorted(df["ticker"].unique()):
        row = f"  {ticker:>6} |"
        for slot_min in ENTRY_SLOTS:
            c = df[(df["ticker"] == ticker) & (df["entry_minute"] == slot_min)]
            if len(c) >= 5:
                wr_c = len(c[c["pnl"] > 0]) / len(c) * 100
                if wr_c >= 60:
                    row += f" {wr_c:>4.0f}++|"
                elif wr_c >= 50:
                    row += f" {wr_c:>4.0f}  |"
                elif wr_c >= 40:
                    row += f" {wr_c:>4.0f}  |"
                else:
                    row += f" {wr_c:>4.0f}--|"
            else:
                row += f"    -- |"
        print(row)

    # ── P&L Heatmap ──
    print(f"\n{'─'*80}")
    print(f"AVG P&L HEATMAP (per trade, min 5 trades/cell)")
    print(f"{'─'*80}")

    print(f"{'':>8} |" + "|".join(f"{l:>7}" for l in slot_labels))
    print("-" * (10 + 8 * len(ENTRY_SLOTS)))

    for ticker in sorted(df["ticker"].unique()):
        row = f"  {ticker:>6} |"
        for slot_min in ENTRY_SLOTS:
            c = df[(df["ticker"] == ticker) & (df["entry_minute"] == slot_min)]
            if len(c) >= 5:
                avg = c["pnl"].mean()
                row += f" ${avg:>+5.0f}|"
            else:
                row += f"    -- |"
        print(row)

    # ── Top 15 Best PUT Setups ──
    print(f"\n{'─'*80}")
    print(f"TOP 15 PROFITABLE PUT SETUPS (min 10 trades)")
    print(f"{'─'*80}")

    combos = []
    for ticker in df["ticker"].unique():
        for slot_min in ENTRY_SLOTS:
            c = df[(df["ticker"] == ticker) & (df["entry_minute"] == slot_min)]
            if len(c) >= 10:
                w = len(c[c["pnl"] > 0])
                gw = c[c["pnl"] > 0]["pnl"].sum() if w else 0
                gl = abs(c[c["pnl"] <= 0]["pnl"].sum()) if (len(c) - w) else 1
                combos.append({
                    "ticker": ticker, "slot": slot_label(slot_min),
                    "trades": len(c), "wr": w / len(c) * 100,
                    "total_pnl": c["pnl"].sum(), "avg_pnl": c["pnl"].mean(),
                    "pf": gw / gl if gl > 0 else 99.9,
                    "avg_hold": c["hold_min"].mean(), "avg_peak": c["peak_gain"].mean(),
                })

    combos.sort(key=lambda x: x["avg_pnl"], reverse=True)

    print(f"{'Ticker':>8} {'Slot':>8} | {'N':>4} | {'WR%':>5} | {'PF':>5} | {'Avg P&L':>9} | {'Total':>10} | {'Hold':>4} | {'Peak':>4}")
    print("-" * 75)
    for c in combos[:15]:
        print(f"  {c['ticker']:>6} {c['slot']:>8} | {c['trades']:>4} | {c['wr']:>4.0f}% | {c['pf']:>4.1f} | "
              f"${c['avg_pnl']:>+7,.0f} | ${c['total_pnl']:>+8,.0f} | {c['avg_hold']:>3.0f}m | {c['avg_peak']:>3.0f}%")

    print(f"\n{'─'*80}")
    print(f"BOTTOM 10 WORST PUT SETUPS")
    print(f"{'─'*80}")
    for c in combos[-10:]:
        print(f"  {c['ticker']:>6} {c['slot']:>8} | {c['trades']:>4} | {c['wr']:>4.0f}% | {c['pf']:>4.1f} | "
              f"${c['avg_pnl']:>+7,.0f} | ${c['total_pnl']:>+8,.0f} | {c['avg_hold']:>3.0f}m | {c['avg_peak']:>3.0f}%")

    # ── Exit Reasons ──
    print(f"\n{'─'*80}")
    print(f"EXIT REASONS")
    print(f"{'─'*80}")
    for reason in df["exit_reason"].value_counts().index:
        r = df[df["exit_reason"] == reason]
        w = len(r[r["pnl"] > 0])
        print(f"  {reason:<25} {len(r):>5} | {w/len(r)*100:>4.0f}% WR | ${r['pnl'].sum():>+10,.0f}")

    # ── DTE analysis ──
    print(f"\n{'─'*80}")
    print(f"BY DTE")
    print(f"{'─'*80}")
    for dte_val in sorted(df["dte"].unique()):
        d = df[df["dte"] == dte_val]
        w = len(d[d["pnl"] > 0])
        print(f"  DTE={dte_val}: {len(d):>5} trades, {w/len(d)*100:>4.0f}% WR, ${d['pnl'].sum():>+10,.0f}, "
              f"avg hold {d['hold_min'].mean():.0f}m")


if __name__ == "__main__":
    main()
