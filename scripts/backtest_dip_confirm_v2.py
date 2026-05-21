"""Backtest dip-confirm v2 — uses harvester 60s resolution properly.

Instead of simulating exact 5s polls (impossible with 60s ticks), this:
1. Checks if premium was falling at entry (t0 vs t+60s)
2. For fading trades, checks if premium recovered within N minutes
3. For non-recovering trades, simulates skipping them
4. For recovering trades, uses the recovery price as entry

This answers the REAL question: "If the stock is going against us right
away, does waiting for it to tick back in our favor help?"

Usage:
    python scripts/backtest_dip_confirm_v2.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

HARVESTER_DB = Path("journal/owlet-harvester/options_data.db")
BOT_DBS = {
    "kody": Path("journal/owlet-kody/raw_messages.db"),
    "adam": Path("journal/owlet-adam/raw_messages.db"),
    "vinny": Path("journal/owlet-vinny/raw_messages.db"),
    "yank": Path("journal/owlet-yank/raw_messages.db"),
}


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
class PremiumPath:
    """Premium trajectory from entry through entry + N minutes."""
    t0_price: float  # signal premium
    # List of (seconds_offset, ask_price) after entry
    ticks: list[tuple[float, float]]

    @property
    def t60_price(self) -> float | None:
        """Price ~60s after entry."""
        for offset, price in self.ticks:
            if 30 < offset < 90:
                return price
        return None

    @property
    def t120_price(self) -> float | None:
        """Price ~120s after entry."""
        for offset, price in self.ticks:
            if 90 < offset < 180:
                return price
        return None

    @property
    def is_fading(self) -> bool:
        """Was premium fading in the first 60s?"""
        t60 = self.t60_price
        if t60 is None:
            return False
        return t60 < self.t0_price * 0.99  # down > 1%

    @property
    def fade_pct(self) -> float | None:
        """% fade from t0 to t+60s."""
        t60 = self.t60_price
        if t60 is None or self.t0_price <= 0:
            return None
        return (self.t0_price - t60) / self.t0_price * 100

    def first_uptick_after_fade(self) -> tuple[float, float] | None:
        """Find first uptick after initial fade. Returns (offset_sec, price) or None."""
        if not self.is_fading:
            return None
        # Find the local minimum, then first tick higher than it
        prev_price = None
        for offset, price in self.ticks:
            if offset < 30:
                continue  # skip first few seconds
            if prev_price is not None and price > prev_price:
                return (offset, price)
            prev_price = price
        return None

    def min_price_in_window(self, max_sec: float) -> tuple[float, float] | None:
        """Find (offset, price) of the minimum ask in window."""
        best = None
        for offset, price in self.ticks:
            if offset > max_sec:
                break
            if best is None or price < best[1]:
                best = (offset, price)
        return best

    def price_recovered_by(self, max_sec: float, threshold_pct: float = 50) -> bool:
        """Did premium recover at least threshold_pct of the initial fade within max_sec?"""
        t60 = self.t60_price
        if t60 is None or not self.is_fading:
            return True  # not fading, treat as "recovered"
        fade_amount = self.t0_price - t60
        recovery_target = t60 + fade_amount * (threshold_pct / 100)
        for offset, price in self.ticks:
            if offset > max_sec:
                break
            if price >= recovery_target:
                return True
        return False


def load_trades() -> list[Trade]:
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
    return trades


def get_premium_path(
    conn: sqlite3.Connection, trade: Trade, window_sec: int = 300,
) -> PremiumPath | None:
    entry_time = datetime.fromisoformat(trade.opened_at)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    entry_date = entry_time.strftime("%Y-%m-%d")
    contract = conn.execute(
        "SELECT contract_ticker FROM harvest_contracts "
        "WHERE underlying=? AND strike=? AND option_type=? AND expiry_date>=? "
        "ORDER BY expiry_date LIMIT 1",
        (trade.ticker, trade.strike, trade.option_type, entry_date),
    ).fetchone()
    if not contract:
        return None

    t_start = (entry_time - timedelta(seconds=30)).isoformat()
    t_end = (entry_time + timedelta(seconds=window_sec)).isoformat()

    rows = conn.execute(
        "SELECT captured_at, ask, bid, midpoint FROM harvest_snapshots "
        "WHERE contract_ticker=? AND captured_at BETWEEN ? AND ? "
        "ORDER BY captured_at",
        (contract[0], t_start, t_end),
    ).fetchall()

    ticks = []
    for row in rows:
        cap_time = datetime.fromisoformat(row[0])
        if cap_time.tzinfo is None:
            cap_time = cap_time.replace(tzinfo=timezone.utc)
        offset = (cap_time - entry_time).total_seconds()
        price = row[1] if row[1] and row[1] > 0 else row[3]
        if price and price > 0:
            ticks.append((offset, price))

    if not ticks:
        return None

    return PremiumPath(t0_price=trade.signal_premium, ticks=ticks)


def main() -> None:
    if not HARVESTER_DB.exists():
        print(f"ERROR: {HARVESTER_DB} not found")
        sys.exit(1)

    trades = load_trades()
    print(f"Loaded {len(trades)} closed Webull trades\n")

    conn = sqlite3.connect(str(HARVESTER_DB))

    # Categorize trades by premium behavior at entry
    fading = []     # premium was falling at entry
    stable = []     # premium was stable/rising
    no_data = []    # no harvester data

    for t in trades:
        path = get_premium_path(conn, t, window_sec=300)
        if path is None:
            no_data.append(t)
            continue
        if path.is_fading:
            fading.append((t, path))
        else:
            stable.append((t, path))

    conn.close()

    print(f"Trade categories:")
    print(f"  Stable/rising at entry: {len(stable)} (premium NOT fading in first 60s)")
    print(f"  Fading at entry:        {len(fading)} (premium dropping > 1% in first 60s)")
    print(f"  No harvester data:      {len(no_data)}")
    print()

    # Analyze fading trades
    print(f"{'='*100}")
    print(f"FADING TRADES ANALYSIS — {len(fading)} trades where premium dropped > 1% in first 60s")
    print(f"{'='*100}")
    print()

    total_fade_pnl = sum(t.pnl_dollars for t, _ in fading)
    total_stable_pnl = sum(t.pnl_dollars for t, _ in stable)

    print(f"  Fading trades total P&L:  ${total_fade_pnl:>+10,.0f} ({len(fading)} trades)")
    print(f"  Stable trades total P&L:  ${total_stable_pnl:>+10,.0f} ({len(stable)} trades)")
    print()

    # For fading trades: did the premium recover?
    recovered_60 = []   # recovered within 60s
    recovered_120 = []  # recovered within 120s
    recovered_180 = []  # recovered within 180s
    never_recovered = []

    for t, path in fading:
        uptick = path.first_uptick_after_fade()
        if uptick and uptick[0] <= 90:
            recovered_60.append((t, path, uptick))
        elif uptick and uptick[0] <= 150:
            recovered_120.append((t, path, uptick))
        elif uptick and uptick[0] <= 240:
            recovered_180.append((t, path, uptick))
        else:
            never_recovered.append((t, path))

    print(f"  Fading trade recovery:")
    print(f"    Uptick within ~60s:   {len(recovered_60)} trades, "
          f"P&L ${sum(t.pnl_dollars for t, _, _ in recovered_60):>+10,.0f}")
    print(f"    Uptick within ~120s:  {len(recovered_120)} trades, "
          f"P&L ${sum(t.pnl_dollars for t, _, _ in recovered_120):>+10,.0f}")
    print(f"    Uptick within ~180s:  {len(recovered_180)} trades, "
          f"P&L ${sum(t.pnl_dollars for t, _, _ in recovered_180):>+10,.0f}")
    print(f"    Never recovered:      {len(never_recovered)} trades, "
          f"P&L ${sum(t.pnl_dollars for t, _ in never_recovered):>+10,.0f}")
    print()

    # Simulate strategies
    print(f"{'='*100}")
    print(f"STRATEGY SIMULATION")
    print(f"{'='*100}")
    print()

    baseline_pnl = sum(t.pnl_dollars for t in trades)

    strategies = [
        ("Baseline (no dip confirm)", None),
        ("Skip ALL fading trades", "skip_all"),
        ("Wait 60s for uptick, skip if none", 60),
        ("Wait 120s for uptick, skip if none", 120),
        ("Wait 180s for uptick, skip if none", 180),
        ("Skip fading + buy at dip (min price in 60s)", "buy_dip_60"),
        ("Skip fading + buy at dip (min price in 120s)", "buy_dip_120"),
    ]

    print(f"{'Strategy':<50} {'Trades':>7} {'Skipped':>8} {'PnL':>10} {'vs Base':>10} "
          f"{'Skip$Bad':>10} {'Skip$Good':>10}")
    print("-" * 105)

    for label, strategy in strategies:
        total_pnl = 0
        entered = 0
        skipped = 0
        skip_bad = 0.0  # avoided losses
        skip_good = 0.0  # missed gains

        for t in trades:
            # Find this trade in our categorized lists
            is_fading_trade = False
            trade_path = None
            trade_uptick = None

            for ft, fp in fading:
                if ft.trade_id == t.trade_id and ft.bot == t.bot:
                    is_fading_trade = True
                    trade_path = fp
                    break

            if strategy is None:
                # Baseline
                total_pnl += t.pnl_dollars
                entered += 1
                continue

            if not is_fading_trade:
                # Not fading — always enter
                total_pnl += t.pnl_dollars
                entered += 1
                continue

            # Fading trade — apply strategy
            if strategy == "skip_all":
                skipped += 1
                if t.pnl_dollars < 0:
                    skip_bad += t.pnl_dollars
                else:
                    skip_good += t.pnl_dollars
                continue

            if strategy in ("buy_dip_60", "buy_dip_120"):
                window = 60 if strategy == "buy_dip_60" else 120
                if trade_path:
                    dip = trade_path.min_price_in_window(window)
                    if dip:
                        savings = t.entry_premium - dip[1]
                        adjusted_pnl = t.pnl_dollars + savings * t.contracts * 100
                        total_pnl += adjusted_pnl
                        entered += 1
                        continue
                # No dip data — enter normally
                total_pnl += t.pnl_dollars
                entered += 1
                continue

            # Wait N seconds for uptick
            max_wait = strategy  # type: int
            if trade_path:
                uptick = trade_path.first_uptick_after_fade()
                if uptick and uptick[0] <= max_wait:
                    # Uptick found — enter at uptick price
                    savings = t.entry_premium - uptick[1]
                    adjusted_pnl = t.pnl_dollars + savings * t.contracts * 100
                    total_pnl += adjusted_pnl
                    entered += 1
                else:
                    # No uptick — skip
                    skipped += 1
                    if t.pnl_dollars < 0:
                        skip_bad += t.pnl_dollars
                    else:
                        skip_good += t.pnl_dollars
            else:
                total_pnl += t.pnl_dollars
                entered += 1

        delta = total_pnl - baseline_pnl
        print(f"{label:<50} {entered:>7} {skipped:>8} ${total_pnl:>+9,.0f} ${delta:>+9,.0f} "
              f"${skip_bad:>+9,.0f} ${skip_good:>+9,.0f}")

    print()

    # Detailed view of fading-then-losing trades (the ones dip-confirm should catch)
    print(f"{'='*100}")
    print(f"FADING + LOSING TRADES — these are the trades dip-confirm should protect against")
    print(f"{'='*100}")
    print()

    fading_losers = [(t, p) for t, p in fading if t.pnl_dollars < 0]
    fading_winners = [(t, p) for t, p in fading if t.pnl_dollars >= 0]

    print(f"  Fading losers:  {len(fading_losers)} trades, ${sum(t.pnl_dollars for t,_ in fading_losers):>+10,.0f}")
    print(f"  Fading winners: {len(fading_winners)} trades, ${sum(t.pnl_dollars for t,_ in fading_winners):>+10,.0f}")
    print()

    print(f"  {'Ticker':<8} {'Bot':<8} {'Score':>5} {'Entry$':>8} {'Fade%':>7} "
          f"{'PnL':>10} {'Exit Reason':<20} {'Recovered?':<15}")
    print("  " + "-" * 95)
    for t, path in sorted(fading_losers, key=lambda x: x[0].pnl_dollars):
        fade = path.fade_pct
        uptick = path.first_uptick_after_fade()
        recovered = f"yes @{uptick[0]:.0f}s" if uptick else "no"
        print(f"  {t.ticker:<8} {t.bot:<8} {t.score:>5} ${t.entry_premium:>7.2f} "
              f"{fade:>+6.1f}% ${t.pnl_dollars:>+9,.0f} {t.exit_reason:<20} {recovered:<15}")

    print()
    print(f"  Top fading WINNERS we'd miss:")
    print(f"  {'Ticker':<8} {'Bot':<8} {'Score':>5} {'Entry$':>8} {'Fade%':>7} "
          f"{'PnL':>10} {'Exit Reason':<20}")
    print("  " + "-" * 80)
    for t, path in sorted(fading_winners, key=lambda x: -x[0].pnl_dollars)[:15]:
        fade = path.fade_pct
        print(f"  {t.ticker:<8} {t.bot:<8} {t.score:>5} ${t.entry_premium:>7.2f} "
              f"{fade:>+6.1f}% ${t.pnl_dollars:>+9,.0f} {t.exit_reason:<20}")


if __name__ == "__main__":
    main()
