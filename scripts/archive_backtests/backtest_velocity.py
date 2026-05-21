#!/usr/bin/env python3
"""Backtest velocity exit thresholds using actual harvester minute-bar data.

Simulates different velocity exit configs on real trades to find optimal
thresholds for the scaled velocity approach.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


# All unique trades from Kody's bot (deduplicated)
TRADES = [
    {"ticker": "IWM", "strike": 263.0, "expiry": "2026-04-13", "type": "call",
     "entry_prem": 0.3015, "entry_time": "2026-04-13T14:36:12", "contracts": 16},
    {"ticker": "SPY", "strike": 680.0, "expiry": "2026-04-13", "type": "call",
     "entry_prem": 0.804, "entry_time": "2026-04-13T14:51:13", "contracts": 6},
    {"ticker": "QQQ", "strike": 613.0, "expiry": "2026-04-13", "type": "call",
     "entry_prem": 0.6432, "entry_time": "2026-04-13T15:00:31", "contracts": 8},
    {"ticker": "AMZN", "strike": 237.5, "expiry": "2026-04-13", "type": "call",
     "entry_prem": 0.7537, "entry_time": "2026-04-13T15:24:19", "contracts": 7},
    {"ticker": "META", "strike": 627.5, "expiry": "2026-04-13", "type": "call",
     "entry_prem": 1.7085, "entry_time": "2026-04-13T15:39:19", "contracts": 3},
    {"ticker": "MSTR", "strike": 139.0, "expiry": "2026-04-15", "type": "call",
     "entry_prem": 3.6683, "entry_time": "2026-04-15T14:51:54", "contracts": 1},
    {"ticker": "GOOGL", "strike": 335.0, "expiry": "2026-04-15", "type": "call",
     "entry_prem": 0.8643, "entry_time": "2026-04-15T14:54:53", "contracts": 4},
    {"ticker": "QQQ", "strike": 632.0, "expiry": "2026-04-15", "type": "call",
     "entry_prem": 1.2663, "entry_time": "2026-04-15T15:01:10", "contracts": 2},
    {"ticker": "AAPL", "strike": 262.5, "expiry": "2026-04-15", "type": "call",
     "entry_prem": 0.4221, "entry_time": "2026-04-15T15:15:55", "contracts": 8},
    {"ticker": "SPY", "strike": 698.0, "expiry": "2026-04-15", "type": "call",
     "entry_prem": 1.1256, "entry_time": "2026-04-15T18:33:48", "contracts": 2},
]


def build_contract_ticker(ticker, strike, expiry, opt_type):
    opt_char = "C" if opt_type == "call" else "P"
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    expiry_str = expiry_dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"O:{ticker.upper()}{expiry_str}{opt_char}{strike_int:08d}"


def load_premium_series(conn, trade):
    contract = build_contract_ticker(
        trade["ticker"], trade["strike"], trade["expiry"], trade["type"]
    )
    cursor = conn.execute(
        "SELECT captured_at, midpoint, bid, ask FROM harvest_snapshots "
        "WHERE contract_ticker = ? AND captured_at >= ? ORDER BY captured_at LIMIT 120",
        (contract, trade["entry_time"]),
    )
    rows = cursor.fetchall()
    premiums = []
    for r in rows:
        mid = r[1]
        bid = r[2] or 0
        ask = r[3] or 0
        if mid and mid > 0:
            premiums.append(mid)
        elif bid > 0 and ask > 0:
            premiums.append((bid + ask) / 2)
    return premiums


def simulate_flat_velocity(premiums, entry_prem, drop_pct, grace_bars=2):
    """Simulate flat velocity exit: exit when premium drops X% from MFE."""
    mfe = entry_prem
    exit_prem = premiums[-1] if premiums else entry_prem
    exit_bar = len(premiums)
    exit_reason = "held_to_end"

    for i, p in enumerate(premiums):
        mfe = max(mfe, p)
        if i < grace_bars:
            continue
        was_profitable = mfe > entry_prem * 1.05
        if mfe > 0:
            current_drop = (mfe - p) / mfe * 100
        else:
            current_drop = 0
        if was_profitable and current_drop >= drop_pct:
            exit_prem = p
            exit_bar = i
            exit_reason = "velocity_exit"
            break

    pnl_pct = (exit_prem - entry_prem) / entry_prem * 100
    mfe_pnl = (mfe - entry_prem) / entry_prem * 100
    capture = (pnl_pct / mfe_pnl * 100) if mfe_pnl > 0 else 0
    return {
        "exit_prem": exit_prem,
        "exit_bar": exit_bar,
        "exit_reason": exit_reason,
        "pnl_pct": pnl_pct,
        "mfe_pnl_pct": mfe_pnl,
        "mfe_capture_pct": capture,
    }


def simulate_scaled_velocity(premiums, entry_prem, tiers, grace_bars=2):
    """Simulate scaled velocity exit: threshold varies by how profitable the trade is.

    tiers: list of (gain_pct_threshold, allowed_drop_pct)
    e.g., [(0, 30), (25, 25), (50, 20), (100, 15)]
    Sorted ascending by gain_pct_threshold. The highest matching tier applies.
    """
    mfe = entry_prem
    exit_prem = premiums[-1] if premiums else entry_prem
    exit_bar = len(premiums)
    exit_reason = "held_to_end"

    for i, p in enumerate(premiums):
        mfe = max(mfe, p)
        if i < grace_bars:
            continue
        was_profitable = mfe > entry_prem * 1.05
        if not was_profitable:
            continue

        mfe_gain_pct = (mfe - entry_prem) / entry_prem * 100
        current_drop = (mfe - p) / mfe * 100 if mfe > 0 else 0

        # Find the applicable tier
        allowed_drop = tiers[0][1]  # default to first tier
        for gain_thresh, drop_thresh in tiers:
            if mfe_gain_pct >= gain_thresh:
                allowed_drop = drop_thresh

        if current_drop >= allowed_drop:
            exit_prem = p
            exit_bar = i
            exit_reason = f"velocity_exit(drop={current_drop:.0f}%>={allowed_drop:.0f}%)"
            break

    pnl_pct = (exit_prem - entry_prem) / entry_prem * 100
    mfe_pnl = (mfe - entry_prem) / entry_prem * 100
    capture = (pnl_pct / mfe_pnl * 100) if mfe_pnl > 0 else 0
    return {
        "exit_prem": exit_prem,
        "exit_bar": exit_bar,
        "exit_reason": exit_reason,
        "pnl_pct": pnl_pct,
        "mfe_pnl_pct": mfe_pnl,
        "mfe_capture_pct": capture,
    }


def run_backtest(db_path):
    conn = sqlite3.connect(db_path)

    # Load premium series for each trade
    trade_data = []
    for t in TRADES:
        series = load_premium_series(conn, t)
        if not series:
            print(f"  SKIP {t['ticker']} ${t['strike']} - no harvester data")
            continue
        t["series"] = series
        trade_data.append(t)
        mfe = max(series)
        mfe_gain = (mfe - t["entry_prem"]) / t["entry_prem"] * 100
        print(f"  {t['ticker']} ${t['strike']}: {len(series)} bars, "
              f"entry=${t['entry_prem']:.2f}, peak=${mfe:.2f} (+{mfe_gain:.0f}%), "
              f"final=${series[-1]:.2f}")

    conn.close()

    if not trade_data:
        print("\nNo trade data found in harvester!")
        return

    # =====================================================================
    # CONFIG A: Current flat velocity (12% drop from MFE)
    # =====================================================================
    flat_configs = [
        ("Flat 12% (current)", 12),
        ("Flat 15%", 15),
        ("Flat 20%", 20),
        ("Flat 25%", 25),
        ("Flat 30%", 30),
    ]

    # =====================================================================
    # CONFIG B: Scaled velocity tiers
    # =====================================================================
    scaled_configs = [
        ("Scaled A: conservative", [(0, 25), (25, 20), (50, 15), (100, 12)]),
        ("Scaled B: balanced", [(0, 30), (25, 25), (50, 20), (100, 15)]),
        ("Scaled C: aggressive", [(0, 35), (25, 30), (50, 25), (100, 18)]),
        ("Scaled D: very aggressive", [(0, 40), (30, 30), (60, 22), (150, 15)]),
        ("Scaled E: let winners run", [(0, 35), (20, 30), (50, 25), (100, 20), (200, 15)]),
        ("No velocity (ML only)", [(0, 999)]),  # effectively disabled
    ]

    print("\n" + "=" * 90)
    print("BACKTEST RESULTS — Velocity Exit Configurations")
    print("=" * 90)

    all_results = []

    # Test flat configs
    for name, drop_pct in flat_configs:
        results = []
        for t in trade_data:
            r = simulate_flat_velocity(t["series"], t["entry_prem"], drop_pct)
            r["ticker"] = t["ticker"]
            r["contracts"] = t["contracts"]
            r["dollar_pnl"] = r["pnl_pct"] / 100 * t["entry_prem"] * t["contracts"] * 100
            results.append(r)

        total_pnl = sum(r["dollar_pnl"] for r in results)
        avg_pnl_pct = sum(r["pnl_pct"] for r in results) / len(results)
        avg_capture = sum(r["mfe_capture_pct"] for r in results) / len(results)
        wins = sum(1 for r in results if r["pnl_pct"] > 0)
        avg_bars = sum(r["exit_bar"] for r in results) / len(results)

        all_results.append({
            "name": name,
            "total_pnl": total_pnl,
            "avg_pnl_pct": avg_pnl_pct,
            "avg_capture": avg_capture,
            "win_rate": wins / len(results) * 100,
            "avg_bars": avg_bars,
            "results": results,
        })

    # Test scaled configs
    for name, tiers in scaled_configs:
        results = []
        for t in trade_data:
            r = simulate_scaled_velocity(t["series"], t["entry_prem"], tiers)
            r["ticker"] = t["ticker"]
            r["contracts"] = t["contracts"]
            r["dollar_pnl"] = r["pnl_pct"] / 100 * t["entry_prem"] * t["contracts"] * 100
            results.append(r)

        total_pnl = sum(r["dollar_pnl"] for r in results)
        avg_pnl_pct = sum(r["pnl_pct"] for r in results) / len(results)
        avg_capture = sum(r["mfe_capture_pct"] for r in results) / len(results)
        wins = sum(1 for r in results if r["pnl_pct"] > 0)
        avg_bars = sum(r["exit_bar"] for r in results) / len(results)

        all_results.append({
            "name": name,
            "total_pnl": total_pnl,
            "avg_pnl_pct": avg_pnl_pct,
            "avg_capture": avg_capture,
            "win_rate": wins / len(results) * 100,
            "avg_bars": avg_bars,
            "results": results,
        })

    # Sort by total P&L
    all_results.sort(key=lambda x: x["total_pnl"], reverse=True)

    print(f"\n{'Config':<35} {'Total $':>10} {'Avg %':>8} {'Capture':>9} {'WinRate':>9} {'AvgBars':>8}")
    print("-" * 90)
    for r in all_results:
        marker = " <-- CURRENT" if "current" in r["name"] else ""
        print(f"{r['name']:<35} ${r['total_pnl']:>8.2f} {r['avg_pnl_pct']:>7.1f}% "
              f"{r['avg_capture']:>7.1f}% {r['win_rate']:>7.0f}% {r['avg_bars']:>7.1f}{marker}")

    # Show per-trade detail for top 3
    print("\n" + "=" * 90)
    print("TOP 3 CONFIGS — Per-Trade Detail")
    print("=" * 90)
    for cfg in all_results[:3]:
        print(f"\n--- {cfg['name']} ---")
        print(f"  {'Ticker':<8} {'PnL%':>8} {'MFE%':>8} {'Capture':>9} {'Exit Bar':>9} {'$PnL':>10}  Reason")
        for r in cfg["results"]:
            print(f"  {r['ticker']:<8} {r['pnl_pct']:>7.1f}% {r['mfe_pnl_pct']:>7.1f}% "
                  f"{r['mfe_capture_pct']:>7.1f}% {r['exit_bar']:>8}  ${r['dollar_pnl']:>8.2f}  {r['exit_reason']}")


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/app/journal/options_data.db"
    print(f"Loading data from {db_path}...")
    run_backtest(db_path)
