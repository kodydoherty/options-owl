#!/usr/bin/env python3
"""Backtest Exit v3 strategy on real premium time-series from live trades.

Parses position monitor logs to reconstruct tick-by-tick premium paths,
then simulates four exit strategies:
  1. Current (v2.1): grace period + adaptive trail + profit_retrace + catastrophic stop
  2. V3: thesis revalidation + ratcheting profit floor + bounce-fade + time urgency

Usage:
  python scripts/backtest_exit_v3.py
"""
from __future__ import annotations

import re
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PremiumTick:
    timestamp: datetime
    premium: float

@dataclass
class TradeInfo:
    trade_id: int
    bot: str
    ticker: str
    option_type: str
    strike: float
    contracts: int
    entry_premium: float
    entry_time: datetime
    expiry_date: str  # YYYY-MM-DD
    actual_exit_premium: float
    actual_pnl: float
    actual_exit_reason: str
    premium_path: list[PremiumTick] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse premium paths from logs
# ---------------------------------------------------------------------------

PREMIUM_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \|.*?"
    r"(\w+) — stream premium \$([0-9.]+) \(strike=([0-9.]+)"
)

def parse_premium_logs(log_path: str) -> dict[tuple[str, str], list[PremiumTick]]:
    """Parse premium ticks from log file, keyed by (ticker, strike)."""
    ticks: dict[tuple[str, str], list[PremiumTick]] = defaultdict(list)
    try:
        with open(log_path) as f:
            for line in f:
                m = PREMIUM_RE.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    ticker = m.group(2)
                    premium = float(m.group(3))
                    strike = m.group(4)
                    ticks[(ticker, strike)].append(PremiumTick(ts, premium))
    except FileNotFoundError:
        pass
    return ticks


def load_all_premiums(bot: str) -> dict[tuple[str, str], list[PremiumTick]]:
    """Load all premium ticks for a bot across all log dates."""
    all_ticks: dict[tuple[str, str], list[PremiumTick]] = defaultdict(list)
    log_dir = Path(f"/tmp/premiums_{bot}.txt")
    if not log_dir.exists():
        return all_ticks

    with open(log_dir) as f:
        for line in f:
            m = PREMIUM_RE.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                ticker = m.group(2)
                premium = float(m.group(3))
                strike = m.group(4)
                all_ticks[(ticker, strike)].append(PremiumTick(ts, premium))

    # Sort each by timestamp
    for key in all_ticks:
        all_ticks[key].sort(key=lambda t: t.timestamp)
    return all_ticks


# ---------------------------------------------------------------------------
# V2.1 Exit Simulation (current system)
# ---------------------------------------------------------------------------

def simulate_v2_exit(trade: TradeInfo) -> tuple[float, float, str]:
    """Simulate current v2.1 exit logic. Returns (exit_premium, pnl, reason)."""
    if not trade.premium_path:
        return trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason

    entry = trade.entry_premium
    contracts = trade.contracts
    grace_minutes = 20
    catastrophic_pct = 45.0
    stop_pct = 30.0
    adaptive_activation = 35.0
    adaptive_width = 35.0
    retrace_pct = 50.0
    retrace_min_gain = 25.0

    peak = entry
    grace_end = trade.entry_time + timedelta(minutes=grace_minutes)

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        elapsed = (tick.timestamp - trade.entry_time).total_seconds() / 60

        # Catastrophic stop (bypasses grace)
        drop_pct = (entry - p) / entry * 100
        if drop_pct >= catastrophic_pct:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "catastrophic_stop"

        # Grace period blocks normal stops
        in_grace = tick.timestamp < grace_end

        # Normal stop (after grace)
        if not in_grace and drop_pct >= stop_pct:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "stop_loss"

        # Profit retrace (all levels now, post our fix)
        peak_gain_pct = (peak - entry) / entry * 100
        if peak_gain_pct >= retrace_min_gain:
            profit_at_peak = peak - entry
            profit_now = p - entry
            if profit_at_peak > 0:
                given_back = (profit_at_peak - profit_now) / profit_at_peak * 100
                if given_back >= retrace_pct:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "profit_retrace"

        # Adaptive trail
        if peak_gain_pct >= adaptive_activation:
            trail_drop = (peak - p) / peak * 100
            if trail_drop >= adaptive_width:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "adaptive_trail"

    # EOD — exit at last tick
    last = trade.premium_path[-1]
    pnl = (last.premium - entry) * contracts * 100
    return last.premium, pnl, "eod_cutoff"


# ---------------------------------------------------------------------------
# V3 Exit Simulation
# ---------------------------------------------------------------------------

@dataclass
class ThesisState:
    """Track thesis revalidation state across ticks."""
    consecutive_negative_ticks: int = 0
    last_revalidation: datetime | None = None
    revalidation_interval_s: float = 30.0  # check every 30s when negative


@dataclass
class BounceState:
    """Track bounce-and-fade detection."""
    in_bounce_watch: bool = False
    bounce_low: float = 0.0
    bounce_detected: bool = False
    bounce_high: float = 0.0


def simulate_thesis_revalidation(
    tick: PremiumTick, entry: float, peak: float, trade: TradeInfo,
    state: ThesisState,
) -> str | None:
    """Continuous thesis revalidation when negative.

    Without real candle data in the backtest, we approximate:
    - Rapid decline (3+ consecutive drops in 30s) = thesis weakening
    - Slow bleed over time = thesis invalid
    - Sharp V recovery = thesis still valid

    Returns: 'exit' | 'hold' | None (not applicable)
    """
    p = tick.premium
    if p >= entry:
        state.consecutive_negative_ticks = 0
        return None  # Not negative, skip

    drop_pct = (entry - p) / entry * 100

    # Don't revalidate too frequently
    if state.last_revalidation and (
        (tick.timestamp - state.last_revalidation).total_seconds() < state.revalidation_interval_s
    ):
        return "hold"

    state.last_revalidation = tick.timestamp

    # Simulate thesis check using price action as proxy for candle analysis:
    # - If we're down and ACCELERATING (each tick lower), thesis is failing
    # - If we're down but STABILIZING (bouncing in range), thesis may be intact

    # Check momentum: is the decline accelerating?
    # We use the ratio of current drop to time elapsed
    elapsed_min = max(1, (tick.timestamp - trade.entry_time).total_seconds() / 60)
    decline_rate = drop_pct / elapsed_min  # % per minute

    # Time remaining on option
    try:
        expiry_dt = datetime.strptime(trade.expiry_date, "%Y-%m-%d").replace(
            hour=16, minute=0
        )
        time_remaining_min = max(0, (expiry_dt - tick.timestamp).total_seconds() / 60)
    except (ValueError, TypeError):
        time_remaining_min = 300  # assume 5 hours if unknown

    # Thesis invalidation heuristics:
    # 1. Deep + accelerating decline = invalid
    if drop_pct > 25 and decline_rate > 3.0:
        return "exit"

    # 2. Moderate decline + most of time burned = invalid (theta death)
    if drop_pct > 15 and time_remaining_min < 60:
        return "exit"

    # 3. Shallow decline + plenty of time = hold
    if drop_pct < 20 and time_remaining_min > 120:
        return "hold"

    # 4. Deep decline but decelerating = hold (may be finding support)
    if decline_rate < 1.0 and drop_pct < 35:
        return "hold"

    # 5. Very deep decline, accelerating, low time = exit
    if drop_pct > 30 and time_remaining_min < 120:
        return "exit"

    return "hold"  # default: give benefit of doubt


def simulate_v3_exit(trade: TradeInfo) -> tuple[float, float, str]:
    """Simulate v3 exit logic. Returns (exit_premium, pnl, reason)."""
    if not trade.premium_path:
        return trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason

    entry = trade.entry_premium
    contracts = trade.contracts

    # V3 settings
    FLOOR_ACTIVATION_PCT = 15.0
    FLOOR_RATCHET_PCT = 60.0  # keep 60% of gains
    BOUNCE_WATCH_THRESHOLD = 50.0  # enter bounce watch at -50%
    BOUNCE_MIN_RECOVERY_PCT = 10.0  # need 10% bounce from low
    BOUNCE_FADE_PCT = 15.0  # sell when fading 15% from bounce high

    # Time urgency adjustments
    try:
        expiry_dt = datetime.strptime(trade.expiry_date, "%Y-%m-%d").replace(
            hour=16, minute=0
        )
    except (ValueError, TypeError):
        expiry_dt = trade.entry_time + timedelta(hours=6)

    peak = entry
    profit_floor: float | None = None
    thesis_state = ThesisState()
    bounce_state = BounceState()

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        gain_pct = (p - entry) / entry * 100
        peak_gain_pct = (peak - entry) / entry * 100
        drop_from_entry_pct = (entry - p) / entry * 100 if p < entry else 0
        time_remaining_min = max(0, (expiry_dt - tick.timestamp).total_seconds() / 60)

        # --- Time urgency multipliers ---
        if time_remaining_min < 15:
            urgency = "EMERGENCY"
            floor_ratchet = 95.0
            fade_pct = 5.0
        elif time_remaining_min < 30:
            urgency = "CRITICAL"
            floor_ratchet = 90.0
            fade_pct = 8.0
        elif time_remaining_min < 60:
            urgency = "HIGH"
            floor_ratchet = 80.0
            fade_pct = 10.0
        elif time_remaining_min < 120:
            urgency = "ELEVATED"
            floor_ratchet = 70.0
            fade_pct = 12.0
        else:
            urgency = "NORMAL"
            floor_ratchet = FLOOR_RATCHET_PCT
            fade_pct = BOUNCE_FADE_PCT

        # === GATE 1: Continuous Thesis Revalidation (when negative) ===
        if p < entry:
            thesis_result = simulate_thesis_revalidation(
                tick, entry, peak, trade, thesis_state
            )
            if thesis_result == "exit":
                pnl = (p - entry) * contracts * 100
                return p, pnl, "thesis_invalid"

        # === GATE 2: Ratcheting Profit Floor ===
        if peak_gain_pct >= FLOOR_ACTIVATION_PCT:
            new_floor = entry + (peak - entry) * (floor_ratchet / 100)
            if profit_floor is None or new_floor > profit_floor:
                profit_floor = new_floor

        if profit_floor is not None and p <= profit_floor:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "profit_floor"

        # === GATE 3: Bounce-and-Fade Detection ===
        if drop_from_entry_pct >= BOUNCE_WATCH_THRESHOLD and not bounce_state.in_bounce_watch:
            bounce_state.in_bounce_watch = True
            bounce_state.bounce_low = p
            bounce_state.bounce_detected = False

        if bounce_state.in_bounce_watch:
            bounce_state.bounce_low = min(bounce_state.bounce_low, p)

            # Check for bounce
            if not bounce_state.bounce_detected and bounce_state.bounce_low > 0:
                recovery_pct = (p - bounce_state.bounce_low) / bounce_state.bounce_low * 100
                if recovery_pct >= BOUNCE_MIN_RECOVERY_PCT:
                    bounce_state.bounce_detected = True
                    bounce_state.bounce_high = p

            if bounce_state.bounce_detected:
                bounce_state.bounce_high = max(bounce_state.bounce_high, p)

                # Check for fade from bounce
                if bounce_state.bounce_high > 0:
                    fade_from_bounce = (
                        (bounce_state.bounce_high - p) / bounce_state.bounce_high * 100
                    )
                    if fade_from_bounce >= fade_pct:
                        pnl = (p - entry) * contracts * 100
                        return p, pnl, "bounce_fade"

            # Emergency time: just sell on any bounce
            if urgency in ("CRITICAL", "EMERGENCY") and p > bounce_state.bounce_low * 1.03:
                pnl = (p - entry) * contracts * 100
                return p, pnl, f"bounce_fade_{urgency.lower()}"

        # === GATE 4: Time urgency emergency exit ===
        if urgency == "EMERGENCY" and gain_pct < -10:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "emergency_exit"

    # EOD — exit at last tick
    last = trade.premium_path[-1]
    pnl = (last.premium - entry) * contracts * 100
    return last.premium, pnl, "eod_cutoff"


# ---------------------------------------------------------------------------
# Load trades and run backtest
# ---------------------------------------------------------------------------

def load_trades_from_db() -> list[TradeInfo]:
    """Load all live trades from all bot DBs on the droplet (via local dump)."""
    import sqlite3

    trades = []
    for bot in ("kody", "adam", "vinny", "yank"):
        db_path = f"/tmp/db_{bot}.sqlite"
        if not os.path.exists(db_path):
            continue

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT t.id, t.ticker, t.option_type, t.strike, t.contracts,
                   t.premium_per_contract, t.exit_premium, t.pnl_dollars,
                   t.exit_reason, t.opened_at, t.closed_at, t.expiry_date,
                   t.parent_trade_id, t.webull_order_id
            FROM paper_trades t
            WHERE t.status = 'closed'
              AND t.parent_trade_id IS NULL
              AND (length(t.webull_order_id) > 0)
            ORDER BY t.id
        """).fetchall()
        conn.close()

        for row in rows:
            try:
                entry_time = datetime.fromisoformat(row["opened_at"])
            except (ValueError, TypeError):
                continue

            trades.append(TradeInfo(
                trade_id=row["id"],
                bot=bot,
                ticker=row["ticker"],
                option_type=row["option_type"],
                strike=row["strike"],
                contracts=row["contracts"],
                entry_premium=row["premium_per_contract"],
                entry_time=entry_time,
                expiry_date=row["expiry_date"] or "",
                actual_exit_premium=row["exit_premium"] or 0,
                actual_pnl=row["pnl_dollars"] or 0,
                actual_exit_reason=row["exit_reason"] or "unknown",
            ))

    return trades


def main():
    # Step 1: Download DBs from droplet
    print("Downloading trade databases from droplet...")
    for bot in ("kody", "adam", "vinny", "yank"):
        db_src = f"root@129.212.138.145:/root/options-owl/journal/owlet-{bot}/raw_messages.db"
        db_dst = f"/tmp/db_{bot}.sqlite"
        os.system(
            f'scp -i "$HOME/.ssh/id_ed25519_do" {db_src} {db_dst} 2>/dev/null'
        )

    # Step 2: Load trades
    trades = load_trades_from_db()
    print(f"Loaded {len(trades)} parent LIVE trades")

    # Step 3: Load premium paths
    print("Loading premium time-series from logs...")
    all_premiums: dict[str, dict[tuple[str, str], list[PremiumTick]]] = {}
    for bot in ("kody", "adam", "vinny", "yank"):
        all_premiums[bot] = load_all_premiums(bot)
        total = sum(len(v) for v in all_premiums[bot].values())
        print(f"  {bot}: {total} premium ticks across {len(all_premiums[bot])} ticker/strikes")

    # Step 4: Match premium paths to trades
    matched = 0
    for trade in trades:
        bot_premiums = all_premiums.get(trade.bot, {})

        # Try multiple strike format variations
        path = []
        for fmt in (f"{trade.strike}", f"{trade.strike:.1f}", f"{trade.strike:.0f}",
                     f"{int(trade.strike)}.0" if trade.strike == int(trade.strike) else ""):
            key = (trade.ticker, fmt)
            if key in bot_premiums:
                path = bot_premiums[key]
                break

        if path:
            # Filter to ticks during this trade's lifetime
            trade_end = trade.entry_time + timedelta(hours=8)  # generous window
            trade.premium_path = [
                t for t in path
                if t.timestamp >= trade.entry_time and t.timestamp <= trade_end
            ]
            if trade.premium_path:
                matched += 1

    print(f"Matched premium paths for {matched}/{len(trades)} trades")
    print()

    # Step 5: Run both strategies
    print("=" * 100)
    print(f"{'Bot':<6} {'#':>3} {'Ticker':<6} {'Qty':>3} {'Entry':>6} {'Ticks':>5} "
          f"{'V2.1 Exit':>8} {'V2.1 PnL':>9} {'V2.1 Reason':<18} "
          f"{'V3 Exit':>7} {'V3 PnL':>9} {'V3 Reason':<18} {'Diff':>8}")
    print("-" * 100)

    v2_total = 0.0
    v3_total = 0.0
    v2_wins = 0
    v3_wins = 0
    changes = []

    for trade in trades:
        if not trade.premium_path:
            # No premium data — use actual results for both
            v2_total += trade.actual_pnl
            v3_total += trade.actual_pnl
            if trade.actual_pnl > 0:
                v2_wins += 1
                v3_wins += 1
            continue

        v2_exit, v2_pnl, v2_reason = simulate_v2_exit(trade)
        v3_exit, v3_pnl, v3_reason = simulate_v3_exit(trade)

        # Include child trade P&L from actual results (partials we can't simulate)
        # For a fair comparison, we compare parent-only
        v2_total += v2_pnl
        v3_total += v3_pnl
        if v2_pnl > 0:
            v2_wins += 1
        if v3_pnl > 0:
            v3_wins += 1

        diff = v3_pnl - v2_pnl
        marker = ""
        if abs(diff) > 1:
            marker = "**" if diff > 0 else "!!"
            changes.append((trade, v2_pnl, v2_reason, v3_pnl, v3_reason, diff))

        print(
            f"{trade.bot:<6} {trade.trade_id:>3} {trade.ticker:<6} {trade.contracts:>3} "
            f"${trade.entry_premium:.2f} {len(trade.premium_path):>5} "
            f"${v2_exit:>6.2f} ${v2_pnl:>8.2f} {v2_reason:<18} "
            f"${v3_exit:>6.2f} ${v3_pnl:>8.2f} {v3_reason:<18} "
            f"${diff:>+7.2f} {marker}"
        )

    print("=" * 100)
    print()

    # Summary
    total_trades = len(trades)
    print(f"{'METRIC':<30} {'V2.1 (Current)':>15} {'V3 (New)':>15} {'Diff':>10}")
    print("-" * 70)
    print(f"{'Total P&L':<30} ${v2_total:>13.2f} ${v3_total:>13.2f} ${v3_total-v2_total:>+9.2f}")
    print(f"{'Wins':<30} {v2_wins:>15} {v3_wins:>15} {v3_wins-v2_wins:>+10}")
    print(f"{'Losses':<30} {total_trades-v2_wins:>15} {total_trades-v3_wins:>15} {(total_trades-v3_wins)-(total_trades-v2_wins):>+10}")
    print(f"{'Win Rate':<30} {v2_wins/total_trades*100:>14.1f}% {v3_wins/total_trades*100:>14.1f}% {(v3_wins-v2_wins)/total_trades*100:>+9.1f}%")
    print()

    if changes:
        print(f"Trades that CHANGED ({len(changes)}):")
        print("-" * 80)
        improved = [c for c in changes if c[5] > 0]
        worsened = [c for c in changes if c[5] < 0]

        if improved:
            print(f"\n  IMPROVED ({len(improved)}, total ${sum(c[5] for c in improved):+.2f}):")
            for trade, v2p, v2r, v3p, v3r, diff in sorted(improved, key=lambda x: -x[5]):
                print(f"    {trade.bot} #{trade.trade_id} {trade.ticker}: "
                      f"${v2p:.2f} ({v2r}) → ${v3p:.2f} ({v3r}) = ${diff:+.2f}")

        if worsened:
            print(f"\n  WORSENED ({len(worsened)}, total ${sum(c[5] for c in worsened):+.2f}):")
            for trade, v2p, v2r, v3p, v3r, diff in sorted(worsened, key=lambda x: x[5]):
                print(f"    {trade.bot} #{trade.trade_id} {trade.ticker}: "
                      f"${v2p:.2f} ({v2r}) → ${v3p:.2f} ({v3r}) = ${diff:+.2f}")


if __name__ == "__main__":
    main()
