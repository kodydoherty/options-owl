"""Backtest Vinny's strategy on REAL signals from our Discord collector.

For each signal in trade_signals:
1. Construct the Polygon option ticker from signal strike/direction/date
2. Download 1-min option bars if not already cached
3. Find the entry bar matching the signal timestamp
4. Run Vinny's strategy bar-by-bar
5. Report results

Usage:
    python scripts/backtest_signals.py
    python scripts/backtest_signals.py --download  # download missing option bars first
"""

import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options_owl.risk.vinny_strategy import (
    evaluate_phase_trail,
    is_time_decay_zone,
    score_to_contracts,
)

SIGNALS_DB = os.path.join(os.path.dirname(__file__), "..", "journal", "raw_messages.db")
HIST_DB = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
API_KEY = os.getenv("POLYGON_API_KEY", "Zi2nVXh9YJdPtfmuQRScmecxj3IlSpET")
REQUEST_DELAY = 13.0
STARTING_BALANCE = 3000.0


def load_signals():
    """Load all signals from the collector DB."""
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, score, entry_price, strike, expiry,
               target_1, target_2, stop_price, atm_premium, bot_source,
               created_at
        FROM trade_signals
        ORDER BY created_at
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def signal_to_date(signal):
    """Extract the trading date from a signal's created_at timestamp."""
    # created_at is in UTC; convert to ET (UTC-4 for EDT)
    utc_dt = datetime.fromisoformat(signal["created_at"])
    et_dt = utc_dt - timedelta(hours=4)
    return et_dt.strftime("%Y-%m-%d")


def signal_to_entry_time(signal):
    """Get the ET hour:minute from signal's created_at."""
    utc_dt = datetime.fromisoformat(signal["created_at"])
    et_dt = utc_dt - timedelta(hours=4)
    return et_dt.hour, et_dt.minute


def build_option_ticker(ticker, date_str, direction, strike):
    """Build Polygon option ticker: O:SPY260327C00642000"""
    yy = date_str[2:4]
    mm = date_str[5:7]
    dd = date_str[8:10]
    cp = "C" if direction == "call" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"O:{ticker}{yy}{mm}{dd}{cp}{strike_str}"


def have_option_bars(contract_ticker):
    """Check if we already have bars for this contract."""
    conn = sqlite3.connect(HIST_DB, timeout=30)
    count = conn.execute(
        "SELECT COUNT(*) FROM option_bars WHERE contract_ticker = ?",
        (contract_ticker,)
    ).fetchone()[0]
    conn.close()
    return count > 0


def download_option_bars(contract_ticker, date_str):
    """Download 1-min bars for an option contract from Polygon."""
    import httpx

    url = (f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/1/minute/"
           f"{date_str}/{date_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")

    time.sleep(REQUEST_DELAY)
    r = httpx.get(url, timeout=30)
    data = r.json()

    if data.get("status") == "NOT_AUTHORIZED":
        print(f"    NOT_AUTHORIZED for {contract_ticker}")
        return 0

    results = data.get("results", [])
    if not results:
        print(f"    No bars for {contract_ticker}")
        return 0

    conn = sqlite3.connect(HIST_DB, timeout=30)
    conn.executemany(
        "INSERT OR IGNORE INTO option_bars VALUES (?,?,?,?,?,?,?,?,?)",
        [(contract_ticker, b["t"], b["o"], b["h"], b["l"], b["c"],
          b.get("v", 0), b.get("vw", 0), b.get("n", 0)) for b in results]
    )
    conn.commit()
    conn.close()
    return len(results)


def load_option_bars(contract_ticker):
    """Load 1-min option bars from the historical DB."""
    conn = sqlite3.connect(HIST_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def et_time_str(timestamp_ms):
    return ts_to_et(timestamp_ms).strftime("%H:%M")


def find_entry_bar(bars, entry_hour, entry_minute):
    """Find the bar at or just after the entry time."""
    target_min = entry_hour * 60 + entry_minute
    best_idx = None
    best_diff = float("inf")
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_min = et.hour * 60 + et.minute
        diff = bar_min - target_min
        if diff >= 0 and diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


@dataclass
class SignalTradeResult:
    signal_id: int
    ticker: str
    direction: str
    score: int
    strike: float
    bot_source: str
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    pnl_dollars: float
    contracts: int
    total_cost: float
    exit_reason: str
    duration_min: int
    targets_hit: int
    entry_time: str
    exit_time: str
    date: str


def simulate_signal(signal, option_bars):
    """Simulate Vinny's strategy on a real signal using real option bars."""
    entry_hour, entry_minute = signal_to_entry_time(signal)
    entry_idx = find_entry_bar(option_bars, entry_hour, entry_minute)

    if entry_idx is None or entry_idx >= len(option_bars) - 5:
        return None

    entry_bar = option_bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry <= 0:
        return None

    # Use signal's ATM premium if available, otherwise use bar price
    signal_premium = signal.get("atm_premium")
    if signal_premium and signal_premium > 0:
        entry_premium = signal_premium * 1.005  # 0.5% slippage
    else:
        entry_premium = raw_entry * 1.005

    # Score-based sizing
    contracts = score_to_contracts(signal["score"])
    if contracts <= 0:
        return None

    cost_per_contract = entry_premium * 100
    total_cost = contracts * cost_per_contract

    # Premium-based targets (typical 0DTE targets)
    target_pcts = [20.0, 40.0, 70.0, 100.0, 150.0]
    targets = [entry_premium * (1 + p / 100) for p in target_pcts]

    # Walk bars
    peak_premium = entry_premium
    last_target_hit = 0
    remaining_pct = 100.0
    weighted_pnl = 0.0
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    last_new_high_bar = entry_idx

    entry_et = ts_to_et(entry_bar["timestamp"])
    opened_at_str = entry_et.strftime("%Y-%m-%dT%H:%M:%S")

    for bar_idx in range(entry_idx + 1, len(option_bars)):
        bar = option_bars[bar_idx]
        minutes_elapsed = bar_idx - entry_idx
        current = bar["close"]
        if current <= 0:
            continue

        bar_et = ts_to_et(bar["timestamp"])

        if current > peak_premium:
            peak_premium = current
            last_new_high_bar = bar_idx

        gain_pct = (current - entry_premium) / entry_premium * 100

        # EOD 3:45 PM
        if bar_et.hour > 15 or (bar_et.hour == 15 and bar_et.minute >= 45):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "eod_close"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Scale-out at targets (20% each)
        for t_idx, t_price in enumerate(targets):
            t_num = t_idx + 1
            if t_num > last_target_hit and current >= t_price and remaining_pct > 0:
                sell_pct = remaining_pct * 0.20
                weighted_pnl += sell_pct * gain_pct
                remaining_pct -= sell_pct
                last_target_hit = t_num

        # Grace period (5 min)
        if minutes_elapsed < 5:
            continue

        # Premium stop (-60%)
        drop_pct = (entry_premium - current) / entry_premium * 100
        if drop_pct >= 60.0:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "premium_stop"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Setup failed (no 10% gain by minute 10)
        if minutes_elapsed >= 10 and last_target_hit == 0 and gain_pct < 10.0:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "setup_failed"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Phase-based trailing stop
        if peak_premium > entry_premium:
            in_decay = is_time_decay_zone(opened_at_str, bar_et)
            trail_result = evaluate_phase_trail(
                entry_premium=entry_premium,
                current_premium=current,
                peak_premium=peak_premium,
                last_target_hit=last_target_hit,
                in_time_decay_zone=in_decay,
            )
            if trail_result.should_exit:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = f"phase_trail_p{trail_result.phase}"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Theta bleed (held >45 min + down >30%)
        if minutes_elapsed >= 45:
            loss_pct = (entry_premium - current) / entry_premium * 100
            if loss_pct >= 30.0:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "theta_bleed"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Time decay stale (no new high in 5 min)
        in_decay = is_time_decay_zone(opened_at_str, bar_et)
        if in_decay and current < peak_premium * 0.99:
            if bar_idx - last_new_high_bar >= 5:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "time_decay_stale"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # No momentum (30 min, no +5%)
        if minutes_elapsed >= 30 and last_target_hit == 0 and gain_pct < 5.0:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "no_momentum"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

    # Never exited
    if exit_reason is None:
        last_bar = option_bars[-1]
        exit_premium = last_bar["close"]
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar_idx = len(option_bars) - 1

    # Slippage on exit
    total_pnl_pct = weighted_pnl / 100 - 0.5  # 0.5% exit slippage
    pnl_dollars = total_cost * (total_pnl_pct / 100)

    return SignalTradeResult(
        signal_id=signal["id"],
        ticker=signal["ticker"],
        direction=signal["direction"],
        score=signal["score"],
        strike=signal["strike"],
        bot_source=signal["bot_source"],
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        peak_premium=peak_premium,
        pnl_pct=total_pnl_pct,
        pnl_dollars=pnl_dollars,
        contracts=contracts,
        total_cost=total_cost,
        exit_reason=exit_reason,
        duration_min=exit_bar_idx - entry_idx,
        targets_hit=last_target_hit,
        entry_time=et_time_str(option_bars[entry_idx]["timestamp"]),
        exit_time=et_time_str(option_bars[exit_bar_idx]["timestamp"]) if exit_bar_idx < len(option_bars) else "",
        date=signal_to_date(signal),
    )


def main(do_download=False):
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from Discord collector")
    print()

    # Build option tickers for each signal and check availability
    signal_contracts = []
    for sig in signals:
        date = signal_to_date(sig)
        contract = build_option_ticker(sig["ticker"], date, sig["direction"], sig["strike"])
        has_bars = have_option_bars(contract)
        signal_contracts.append({
            "signal": sig,
            "date": date,
            "contract": contract,
            "has_bars": has_bars,
        })

    available = sum(1 for s in signal_contracts if s["has_bars"])
    missing = sum(1 for s in signal_contracts if not s["has_bars"])
    print(f"Option bars: {available} available, {missing} missing")

    if missing > 0 and do_download:
        print(f"\nDownloading {missing} missing contracts (~{missing * 13}s)...")
        for sc in signal_contracts:
            if not sc["has_bars"]:
                print(f"  Downloading {sc['contract']} ({sc['date']})...", end=" ")
                count = download_option_bars(sc["contract"], sc["date"])
                print(f"{count} bars")
                sc["has_bars"] = count > 0
        print()
        available = sum(1 for s in signal_contracts if s["has_bars"])
        missing = sum(1 for s in signal_contracts if not s["has_bars"])
        print(f"After download: {available} available, {missing} still missing")
    elif missing > 0 and not do_download:
        print(f"  Run with --download to fetch missing bars ({missing * 13}s estimated)")
    print()

    # Run backtest on available signals
    results = []
    skipped = 0
    balance = STARTING_BALANCE

    for sc in signal_contracts:
        if not sc["has_bars"]:
            skipped += 1
            continue

        bars = load_option_bars(sc["contract"])
        if not bars or len(bars) < 10:
            skipped += 1
            continue

        result = simulate_signal(sc["signal"], bars)
        if result is None:
            skipped += 1
            continue

        # Cap position cost to available balance
        if result.total_cost > balance:
            result.contracts = max(1, int(balance / (result.entry_premium * 100)))
            result.total_cost = result.contracts * result.entry_premium * 100
            result.pnl_dollars = result.total_cost * (result.pnl_pct / 100)

        balance += result.pnl_dollars
        results.append(result)

    if not results:
        print("No trades could be simulated! Run with --download to get option bars.")
        return

    # Report
    wins = [r for r in results if r.pnl_dollars >= 0]
    losses = [r for r in results if r.pnl_dollars < 0]
    total_pnl = balance - STARTING_BALANCE
    win_rate = len(wins) / len(results) * 100 if results else 0
    gross_wins = sum(r.pnl_dollars for r in wins)
    gross_losses = abs(sum(r.pnl_dollars for r in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_win = sum(r.pnl_pct for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0

    print("=" * 110)
    print("  VINNY STRATEGY — REAL SIGNALS BACKTEST")
    print("=" * 110)
    print()
    print(f"  Signals:             {len(signals)} total, {len(results)} traded, {skipped} skipped (no data)")
    print(f"  Starting Balance:    ${STARTING_BALANCE:,.2f}")
    print(f"  Final Balance:       ${balance:,.2f}")
    print(f"  Total PnL:           ${total_pnl:+,.2f} ({total_pnl/STARTING_BALANCE*100:+.1f}%)")
    print()
    print(f"  Wins / Losses:       {len(wins)} / {len(losses)}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Avg Win:             {avg_win:+.1f}%")
    print(f"  Avg Loss:            {avg_loss:+.1f}%")
    pf_str = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"  Profit Factor:       {pf_str}")
    print()

    # Exit reasons
    reasons = {}
    for r in results:
        reasons[r.exit_reason] = reasons.get(r.exit_reason, 0) + 1
    print("  Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl_for = sum(r.pnl_dollars for r in results if r.exit_reason == reason)
        wr = sum(1 for r in results if r.exit_reason == reason and r.pnl_dollars >= 0) / count * 100
        print(f"    {reason:25s} {count:3d} trades  ${pnl_for:+,.2f}  WR:{wr:.0f}%")

    # By bot source
    print()
    print("  By Bot Source:")
    bots = {}
    for r in results:
        if r.bot_source not in bots:
            bots[r.bot_source] = {"w": 0, "l": 0, "pnl": 0.0}
        if r.pnl_dollars >= 0:
            bots[r.bot_source]["w"] += 1
        else:
            bots[r.bot_source]["l"] += 1
        bots[r.bot_source]["pnl"] += r.pnl_dollars
    for bot, stats in sorted(bots.items(), key=lambda x: -x[1]["pnl"]):
        total = stats["w"] + stats["l"]
        wr = stats["w"] / total * 100 if total > 0 else 0
        print(f"    {bot:20s} {total:2d} trades  {stats['w']}W/{stats['l']}L  "
              f"WR:{wr:.0f}%  ${stats['pnl']:+,.2f}")

    # Trade details
    print()
    print(f"  {'#':>3} {'Date':>10} {'Ticker':>6} {'Dir':>4} {'Score':>5} {'$Strike':>7} "
          f"{'Entry':>6} {'Exit':>6} {'Peak':>6} {'PnL%':>7} {'PnL$':>9} "
          f"{'Exit':>20} {'Tgt':>3} {'Min':>4} {'Bot':>15}")
    print("  " + "-" * 130)
    for i, r in enumerate(results, 1):
        print(f"  {i:3d} {r.date:>10} {r.ticker:>6} {r.direction:>4} {r.score:>5} "
              f"${r.strike:>6.1f} ${r.entry_premium:>5.2f} ${r.exit_premium:>5.2f} "
              f"${r.peak_premium:>5.2f} {r.pnl_pct:>+6.1f}% ${r.pnl_dollars:>+8.2f} "
              f"{r.exit_reason:>20} T{r.targets_hit:>1} {r.duration_min:>4} "
              f"{r.bot_source:>15}")

    print()
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download missing option bars from Polygon")
    args = parser.parse_args()
    main(do_download=args.download)
