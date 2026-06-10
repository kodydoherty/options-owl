"""Backtest dip-confirm entry logic against harvester tick data.

For each historical trade, replays the premium ticks around entry time and
simulates the dip-confirm algorithm with various parameter combinations.

Reports: how many trades would have been skipped, how much cheaper the entry
would have been, and the net P&L impact.

Usage:
    python scripts/backtest_dip_confirm.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HARVESTER_DB = Path("journal/owlet-harvester/options_data.db")
BOT_DBS = {
    "kody": Path("journal/owlet-kody/raw_messages.db"),
    "adam": Path("journal/owlet-adam/raw_messages.db"),
    "vinny": Path("journal/owlet-vinny/raw_messages.db"),
    "yank": Path("journal/owlet-yank/raw_messages.db"),
}

# Parameter grid to test
CONFIGS = [
    # (label, poll_sec, max_polls, fade_pct)
    ("CURRENT: 5s×3, 1% fade", 5, 3, 1.0),
    ("3s×3, 1% fade", 3, 3, 1.0),
    ("3s×5, 1% fade", 3, 5, 1.0),
    ("5s×5, 1% fade", 5, 5, 1.0),
    ("5s×3, 2% fade", 5, 3, 2.0),
    ("5s×3, 0.5% fade", 5, 3, 0.5),
    ("3s×3, 0.5% fade", 3, 3, 0.5),
    ("10s×3, 1% fade", 10, 3, 1.0),
    ("5s×6, 1% fade", 5, 6, 1.0),
    ("NO DIP CONFIRM (baseline)", 0, 0, 0),
]


@dataclass
class Trade:
    trade_id: int
    bot: str
    ticker: str
    option_type: str
    strike: float
    signal_premium: float
    entry_premium: float
    opened_at: str
    closed_at: str
    exit_reason: str
    pnl_dollars: float
    pnl_pct: float
    contracts: int
    score: int


@dataclass
class DipResult:
    """Result of simulating dip-confirm for one trade."""
    entered: bool
    entry_price: float | None  # None if skipped
    delay_sec: float  # how long we waited
    savings_pct: float  # % cheaper vs original entry


@dataclass
class ConfigResult:
    label: str
    total_trades: int = 0
    trades_entered: int = 0
    trades_skipped: int = 0
    total_pnl: float = 0.0
    baseline_pnl: float = 0.0
    total_savings_pct: float = 0.0
    savings_count: int = 0
    avg_delay_sec: float = 0.0
    delays: list[float] = field(default_factory=list)
    skipped_would_have_lost: float = 0.0
    skipped_would_have_won: float = 0.0
    skipped_tickers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades() -> list[Trade]:
    """Load all closed Webull trades from all bots."""
    trades = []
    for bot, db_path in BOT_DBS.items():
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, ticker, option_type, strike,
                   premium_per_contract, webull_entry_fill_price, signal_premium,
                   opened_at, closed_at, exit_reason,
                   pnl_dollars, pnl_pct, contracts, score
            FROM paper_trades
            WHERE status='closed' AND webull_order_id IS NOT NULL
            ORDER BY id
        """).fetchall()
        conn.close()

        for r in rows:
            trades.append(Trade(
                trade_id=r["id"],
                bot=bot,
                ticker=r["ticker"],
                option_type=r["option_type"],
                strike=r["strike"],
                signal_premium=r["signal_premium"] or r["premium_per_contract"],
                entry_premium=r["webull_entry_fill_price"] or r["premium_per_contract"],
                opened_at=r["opened_at"],
                closed_at=r["closed_at"],
                exit_reason=r["exit_reason"] or "unknown",
                pnl_dollars=r["pnl_dollars"] or 0,
                pnl_pct=r["pnl_pct"] or 0,
                contracts=r["contracts"] or 1,
                score=r["score"] or 80,
            ))

    print(f"Loaded {len(trades)} closed Webull trades across {len(BOT_DBS)} bots")
    return trades


def get_premium_ticks(
    harvester_conn: sqlite3.Connection,
    ticker: str,
    strike: float,
    option_type: str,
    entry_time: datetime,
    window_before_sec: int = 10,
    window_after_sec: int = 120,
) -> list[tuple[float, float]]:
    """Get (seconds_from_entry, ask_price) ticks from harvester around entry time.

    Returns sorted list of (offset_seconds, premium) where offset 0 = entry time.
    Uses ask price (what we'd pay to buy).
    """
    # Find the contract ticker
    entry_date = entry_time.strftime("%Y-%m-%d")

    # Try exact date first, then nearby dates
    contract = harvester_conn.execute(
        "SELECT contract_ticker FROM harvest_contracts "
        "WHERE underlying=? AND strike=? AND option_type=? AND expiry_date>=? "
        "ORDER BY expiry_date LIMIT 1",
        (ticker, strike, option_type, entry_date),
    ).fetchone()

    if not contract:
        return []

    contract_ticker = contract[0]

    # Query snapshots around entry time
    t_start = (entry_time - timedelta(seconds=window_before_sec)).isoformat()
    t_end = (entry_time + timedelta(seconds=window_after_sec)).isoformat()

    rows = harvester_conn.execute(
        "SELECT captured_at, ask, bid, midpoint FROM harvest_snapshots "
        "WHERE contract_ticker=? AND captured_at BETWEEN ? AND ? "
        "ORDER BY captured_at",
        (contract_ticker, t_start, t_end),
    ).fetchall()

    ticks = []
    for row in rows:
        cap_time = datetime.fromisoformat(row[0])
        if cap_time.tzinfo is None:
            cap_time = cap_time.replace(tzinfo=timezone.utc)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        offset = (cap_time - entry_time).total_seconds()
        # Use ask (what we'd pay), fall back to midpoint
        price = row[1] if row[1] and row[1] > 0 else row[3]
        if price and price > 0:
            ticks.append((offset, price))

    return ticks


# ---------------------------------------------------------------------------
# Dip-confirm simulation
# ---------------------------------------------------------------------------

def simulate_dip_confirm(
    ticks: list[tuple[float, float]],
    signal_premium: float,
    poll_sec: float,
    max_polls: int,
    fade_pct: float,
) -> DipResult:
    """Simulate dip-confirm algorithm against tick data.

    ticks: [(offset_seconds, premium)] sorted by time, offset 0 = original entry
    """
    if not ticks or poll_sec == 0:
        # Baseline: enter immediately
        return DipResult(entered=True, entry_price=signal_premium, delay_sec=0, savings_pct=0)

    t0 = signal_premium

    def get_price_at(target_sec: float) -> float | None:
        """Get the closest tick price at or just after target_sec."""
        # Find tick closest to target time (within ±30s since harvester polls ~60s)
        best = None
        best_dist = float("inf")
        for offset, price in ticks:
            dist = abs(offset - target_sec)
            if dist < best_dist:
                best = price
                best_dist = dist
        # Only accept if within 45 seconds of target
        return best if best_dist <= 45 else None

    # Step 1: wait poll_sec, check t1
    t1 = get_price_at(poll_sec)
    if t1 is None:
        # No data — enter immediately
        return DipResult(entered=True, entry_price=t0, delay_sec=0, savings_pct=0)

    fade = (t0 - t1) / t0 * 100 if t0 > 0 else 0

    if fade < fade_pct:
        # Premium stable/rising — enter now at t1 if cheaper
        entry = min(t0, t1)
        savings = (t0 - entry) / t0 * 100 if t0 > 0 else 0
        return DipResult(entered=True, entry_price=entry, delay_sec=poll_sec, savings_pct=savings)

    # Premium IS fading — poll for uptick
    prev = t1
    for poll in range(max_polls):
        target_sec = poll_sec + (poll + 1) * poll_sec
        current = get_price_at(target_sec)
        if current is None:
            continue

        if current > prev:
            # Uptick — enter here
            savings = (t0 - current) / t0 * 100 if t0 > 0 else 0
            return DipResult(
                entered=True, entry_price=current,
                delay_sec=target_sec, savings_pct=savings,
            )
        prev = current

    # No uptick — skip trade
    total_wait = poll_sec + max_polls * poll_sec
    return DipResult(entered=False, entry_price=None, delay_sec=total_wait, savings_pct=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HARVESTER_DB.exists():
        print(f"ERROR: {HARVESTER_DB} not found")
        sys.exit(1)

    trades = load_trades()
    if not trades:
        print("No trades found.")
        return

    harvester_conn = sqlite3.connect(str(HARVESTER_DB))

    # Pre-load ticks for each trade (reuse across configs)
    print("Loading premium ticks from harvester...")
    trade_ticks: list[tuple[Trade, list[tuple[float, float]]]] = []
    no_ticks = 0
    for t in trades:
        entry_time = datetime.fromisoformat(t.opened_at)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        ticks = get_premium_ticks(
            harvester_conn, t.ticker, t.strike, t.option_type,
            entry_time, window_before_sec=10, window_after_sec=120,
        )
        trade_ticks.append((t, ticks))
        if not ticks:
            no_ticks += 1

    harvester_conn.close()
    print(f"  {len(trade_ticks)} trades, {no_ticks} with no harvester ticks (will pass through)\n")

    # Run each config
    results: list[ConfigResult] = []
    for label, poll_sec, max_polls, fade_pct in CONFIGS:
        cr = ConfigResult(label=label)

        for trade, ticks in trade_ticks:
            cr.total_trades += 1
            cr.baseline_pnl += trade.pnl_dollars

            result = simulate_dip_confirm(
                ticks, trade.signal_premium, poll_sec, max_polls, fade_pct,
            )

            if result.entered:
                cr.trades_entered += 1
                cr.delays.append(result.delay_sec)

                # Calculate adjusted P&L based on cheaper entry
                if result.entry_price and result.entry_price != trade.entry_premium:
                    # We got a cheaper entry — adjust P&L
                    saved_per_contract = trade.entry_premium - result.entry_price
                    pnl_boost = saved_per_contract * trade.contracts * 100
                    cr.total_pnl += trade.pnl_dollars + pnl_boost
                    if saved_per_contract > 0:
                        cr.total_savings_pct += result.savings_pct
                        cr.savings_count += 1
                else:
                    cr.total_pnl += trade.pnl_dollars
            else:
                cr.trades_skipped += 1
                cr.skipped_tickers.append(f"{trade.ticker}({trade.pnl_dollars:+.0f})")
                if trade.pnl_dollars < 0:
                    cr.skipped_would_have_lost += trade.pnl_dollars
                else:
                    cr.skipped_would_have_won += trade.pnl_dollars

        cr.avg_delay_sec = sum(cr.delays) / len(cr.delays) if cr.delays else 0
        results.append(cr)

    # Print results
    print(f"{'='*100}")
    print(f"DIP-CONFIRM BACKTEST — {len(trade_ticks)} trades (all bots)")
    print(f"{'='*100}")
    print()

    # Header
    print(f"{'Config':<30} {'Entered':>8} {'Skipped':>8} {'PnL':>10} {'vs Base':>10} "
          f"{'Avg Delay':>10} {'Avg Save':>10} {'Skip$Lost':>10} {'Skip$Won':>10}")
    print("-" * 116)

    baseline_pnl = results[-1].total_pnl if results else 0  # "NO DIP CONFIRM" is baseline

    for cr in results:
        avg_save = cr.total_savings_pct / cr.savings_count if cr.savings_count > 0 else 0
        delta = cr.total_pnl - baseline_pnl
        print(
            f"{cr.label:<30} {cr.trades_entered:>8} {cr.trades_skipped:>8} "
            f"${cr.total_pnl:>+9,.0f} ${delta:>+9,.0f} "
            f"{cr.avg_delay_sec:>8.1f}s {avg_save:>9.1f}% "
            f"${cr.skipped_would_have_lost:>+9,.0f} ${cr.skipped_would_have_won:>+9,.0f}"
        )

    print()
    print("Legend:")
    print("  Entered   = trades that passed dip-confirm and we'd still take")
    print("  Skipped   = trades blocked (no uptick — premium kept falling)")
    print("  PnL       = total P&L with dip-confirm adjustments (cheaper entries + skips)")
    print("  vs Base   = improvement over no dip-confirm")
    print("  Avg Delay = avg seconds waited before entering (for entered trades)")
    print("  Avg Save  = avg % cheaper entry (for trades that got a discount)")
    print("  Skip$Lost = total $ of skipped trades that WERE losers (good skips)")
    print("  Skip$Won  = total $ of skipped trades that WERE winners (missed opportunities)")
    print()

    # Detail on best config's skipped trades
    best = max(results[:-1], key=lambda r: r.total_pnl)
    print(f"Best config: {best.label}")
    print(f"  Net improvement: ${best.total_pnl - baseline_pnl:+,.0f}")
    print(f"  Trades skipped ({best.trades_skipped}):")
    for s in best.skipped_tickers:
        print(f"    {s}")


if __name__ == "__main__":
    main()
