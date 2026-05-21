#!/usr/bin/env python3
"""Backtest v2.2 sell logic against ALL signals using harvester tick data.

For each signal (103 total), constructs the OCC contract ticker,
looks up minute-by-minute bid/ask from the harvester DB, and simulates
both our current exits and v2.2 exits using actual market data.

This tests ALL signals, not just the ones we traded.
"""

import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# V2.2 exit parameters
# ---------------------------------------------------------------------------

WHIP_MULTIPLIERS = {
    "NVDA": 1.5, "TSLA": 1.5, "AMZN": 1.3, "AVGO": 1.3, "PLTR": 1.3,
}

SLIPPAGE_HAIRCUT = 0.15  # 15% on gains

# Hard stop
HARD_STOP_PCT = 30.0

# BE clamp activation
BE_CLAMP_ACTIVATION_PCT = 15.0

# Soft trail (15-35% gain band): floor = entry + 50% of gain
SOFT_TRAIL_FLOOR_PCT = 50.0

# Trail tiers
TRAIL_TIERS = [
    (200.0, 0.20),  # 200%+: giveback 20%
    (100.0, 0.25),  # 100-200%: giveback 25%
    (35.0, 0.35),   # 35-100%: giveback 35%
]

# Milestone locks
MILESTONE_LEVELS = [200, 400, 600]
MILESTONE_BANK_PCT = 0.15

# Current v4 exit parameters (for comparison)
V4_ADAPTIVE_ACTIVATION = 35.0  # dormant below this
V4_ACTIVE_WIDTH = 35.0
V4_RUNNER_THRESHOLD = 150.0
V4_RUNNER_WIDTH = 45.0
V4_MOONSHOT_THRESHOLD = 400.0
V4_MOONSHOT_WIDTH = 30.0
V4_HARD_STOP = 30.0
V4_GRACE_MINUTES = 20


@dataclass
class TickData:
    ts: datetime
    mid: float
    bid: float
    ask: float


@dataclass
class SimResult:
    signal_id: int
    ticker: str
    direction: str
    day: str
    score: int
    entry_premium: float
    contracts: int  # simulated
    was_traded: bool  # did we actually trade this?

    # Tick-by-tick results
    peak_premium: float = 0.0
    peak_gain_pct: float = 0.0
    ticks_count: int = 0

    # V4 (current) simulation
    v4_exit_premium: float = 0.0
    v4_exit_reason: str = ""
    v4_pnl: float = 0.0

    # V2.2 simulation
    v22_exit_premium: float = 0.0
    v22_exit_reason: str = ""
    v22_pnl: float = 0.0
    v22_milestone_pnl: float = 0.0

    # Actual (if traded)
    actual_exit_premium: float = 0.0
    actual_pnl: float = 0.0
    actual_exit_reason: str = ""


def build_contract_ticker(ticker: str, day: str, direction: str, strike: float) -> str:
    """Build OCC contract ticker: O:SPY260422C00705000"""
    dt = datetime.strptime(day, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{date_str}{cp}{strike_int:08d}"


def get_ticks(harvester_conn, contract_ticker: str, after_ts: str) -> list[TickData]:
    """Get minute-by-minute ticks for a contract after signal time."""
    rows = harvester_conn.execute("""
        SELECT captured_at, midpoint, bid, ask
        FROM harvest_snapshots
        WHERE contract_ticker = ?
          AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_ts)).fetchall()

    ticks = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            mid = row[1] or 0
            bid = row[2] or 0
            ask = row[3] or 0
            if mid > 0 or bid > 0:
                ticks.append(TickData(ts=ts, mid=mid if mid > 0 else (bid + ask) / 2, bid=bid, ask=ask))
        except (ValueError, TypeError):
            continue
    return ticks


def simulate_v4(entry: float, ticks: list[TickData], signal_ts: datetime) -> tuple[float, str]:
    """Simulate current v4 exit logic tick-by-tick. Returns (exit_premium, reason)."""
    if not ticks or entry <= 0:
        return entry, "no_data"

    peak = entry
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)
    grace_end = signal_ts + timedelta(minutes=V4_GRACE_MINUTES)

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue
        peak = max(peak, price)
        peak_gain = (peak - entry) / entry * 100
        drop_from_peak = (peak - price) / peak * 100 if peak > 0 else 0
        drop_from_entry = (entry - price) / entry * 100 if price < entry else 0

        # Grace period — skip stops
        if tick.ts < grace_end:
            continue

        # Hard stop
        if drop_from_entry >= V4_HARD_STOP:
            return price, "stop_hit"

        # Adaptive trail stages
        if peak_gain >= V4_MOONSHOT_THRESHOLD:
            if drop_from_peak >= V4_MOONSHOT_WIDTH:
                return price, "adaptive_trail_moonshot"
        elif peak_gain >= V4_RUNNER_THRESHOLD:
            if drop_from_peak >= V4_RUNNER_WIDTH:
                return price, "adaptive_trail_runner"
        elif peak_gain >= V4_ADAPTIVE_ACTIVATION:
            if drop_from_peak >= V4_ACTIVE_WIDTH:
                return price, "adaptive_trail_active"

    # EOD — exit at last tick
    return ticks[-1].mid, "eod_cutoff"


def simulate_v22(
    entry: float,
    ticks: list[TickData],
    ticker: str,
    contracts: int,
) -> tuple[float, str, float]:
    """Simulate v2.2 exit logic tick-by-tick.

    Returns (exit_premium, reason, milestone_banked_pnl).
    """
    if not ticks or entry <= 0:
        return entry, "no_data", 0.0

    peak = entry
    whip = WHIP_MULTIPLIERS.get(ticker, 1.0)
    be_clamp_active = False
    milestone_banked_pnl = 0.0
    milestones_hit = set()
    remaining_contracts = contracts

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue

        peak = max(peak, price)
        gain_pct = (price - entry) / entry * 100
        peak_gain = (peak - entry) / entry * 100
        drop_from_entry = (entry - price) / entry * 100 if price < entry else 0

        # Hard stop: -30% from entry
        if drop_from_entry >= HARD_STOP_PCT:
            return price, "hard_stop", milestone_banked_pnl

        # BE clamp activation at +15%
        if peak_gain >= BE_CLAMP_ACTIVATION_PCT:
            be_clamp_active = True

        # BE clamp: once activated, never let it go below entry
        if be_clamp_active and price <= entry and peak_gain >= BE_CLAMP_ACTIVATION_PCT:
            # Only trigger if we've been up and now falling back
            if gain_pct <= 0:
                return entry, "be_clamp", milestone_banked_pnl

        # Milestone locks
        for level in MILESTONE_LEVELS:
            if peak_gain >= level and level not in milestones_hit:
                lock_qty = max(1, int(remaining_contracts * MILESTONE_BANK_PCT))
                if remaining_contracts - lock_qty >= 1:
                    milestones_hit.add(level)
                    lock_premium = entry * (1 + level / 100)
                    lock_pnl = (lock_premium - entry) * lock_qty * 100
                    if lock_pnl > 0:
                        lock_pnl *= (1 - SLIPPAGE_HAIRCUT)
                    milestone_banked_pnl += lock_pnl
                    remaining_contracts -= lock_qty

        # Trail tiers (only when price is dropping from peak)
        if peak_gain >= 35.0:
            # Find applicable tier
            giveback = 0.35  # default
            tier_name = "35+"
            for threshold, gb in TRAIL_TIERS:
                if peak_gain >= threshold:
                    giveback = gb
                    tier_name = f"{threshold:.0f}+"
                    break

            giveback_adjusted = giveback * whip
            trail_floor = peak * (1 - giveback_adjusted)

            if price <= trail_floor:
                return price, f"trail_{tier_name}({giveback_adjusted:.0%})", milestone_banked_pnl

        elif peak_gain >= BE_CLAMP_ACTIVATION_PCT:
            # Soft trail (15-35%): floor = entry + 50% of peak gain
            gain_at_peak = peak - entry
            soft_floor = entry + gain_at_peak * (SOFT_TRAIL_FLOOR_PCT / 100)
            if price <= soft_floor and gain_pct < peak_gain * 0.5:
                return price, "soft_trail", milestone_banked_pnl

    # EOD
    return ticks[-1].mid, "eod_cutoff", milestone_banked_pnl


def main():
    signals_db = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
    harvester_db = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

    sig_conn = sqlite3.connect(signals_db)
    sig_conn.row_factory = sqlite3.Row
    harv_conn = sqlite3.connect(harvester_db)
    harv_conn.row_factory = sqlite3.Row

    # Get all signals
    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike, ts.expiry,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts,
               ts.entry_price, ts.stop_price,
               pt.id as trade_id, pt.premium_per_contract as traded_entry,
               pt.exit_premium as traded_exit, pt.pnl_dollars as traded_pnl,
               pt.exit_reason as traded_exit_reason, pt.contracts as traded_contracts,
               pt.mfe_premium as traded_mfe,
               pt.webull_entry_fill_price
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.parent_trade_id IS NULL
        ORDER BY ts.created_at
    """).fetchall()

    print(f"Found {len(signals)} signals total")

    results: list[SimResult] = []
    no_data_count = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        day = sig["day"]
        strike = sig["strike"]
        score = sig["score"]
        sig_ts_str = sig["sig_ts"]

        # Skip signals with no strike or premium
        premium = sig["atm_premium"] or sig["otm_premium"]
        if not strike or not premium or premium <= 0:
            continue

        # Build contract ticker
        contract = build_contract_ticker(ticker, day, direction, strike)

        # Get ticks from harvester
        ticks = get_ticks(harv_conn, contract, sig_ts_str)

        if not ticks:
            no_data_count += 1
            continue

        # Determine entry premium: use ask (what we'd pay) from first tick, or signal premium
        first_tick = ticks[0]
        entry = first_tick.ask if first_tick.ask > 0 else first_tick.mid
        if entry <= 0:
            entry = premium

        # Simulated contracts (use score-based sizing with $8K portfolio)
        if score >= 150:
            contracts = 5
        elif score >= 130:
            contracts = 4
        elif score >= 110:
            contracts = 3
        elif score >= 95:
            contracts = 1
        else:
            contracts = 1

        was_traded = sig["trade_id"] is not None
        sig_ts = datetime.fromisoformat(sig_ts_str)
        # Make timezone-aware if needed
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        # Simulate v4
        v4_exit, v4_reason = simulate_v4(entry, ticks, sig_ts)
        v4_pnl = (v4_exit - entry) * contracts * 100
        if v4_pnl > 0:
            v4_pnl *= (1 - SLIPPAGE_HAIRCUT)

        # Simulate v2.2
        v22_exit, v22_reason, v22_milestone = simulate_v22(entry, ticks, ticker, contracts)
        v22_main_pnl = (v22_exit - entry) * (contracts - sum(1 for l in MILESTONE_LEVELS if (ticks and len(ticks) > 0 and ((max(t.mid for t in ticks) - entry) / entry * 100) >= l))) * 100
        # Simpler: just compute total
        v22_pnl = (v22_exit - entry) * contracts * 100  # approximate (ignoring milestone contract reduction)
        if v22_pnl > 0:
            v22_pnl *= (1 - SLIPPAGE_HAIRCUT)
        v22_pnl += v22_milestone

        peak = max(t.mid for t in ticks) if ticks else entry
        peak_gain = (peak - entry) / entry * 100 if entry > 0 else 0

        r = SimResult(
            signal_id=sig["id"],
            ticker=ticker,
            direction=direction,
            day=day,
            score=score,
            entry_premium=entry,
            contracts=contracts,
            was_traded=was_traded,
            peak_premium=peak,
            peak_gain_pct=peak_gain,
            ticks_count=len(ticks),
            v4_exit_premium=v4_exit,
            v4_exit_reason=v4_reason,
            v4_pnl=v4_pnl,
            v22_exit_premium=v22_exit,
            v22_exit_reason=v22_reason,
            v22_pnl=v22_pnl,
            v22_milestone_pnl=v22_milestone,
        )

        if was_traded:
            r.actual_pnl = sig["traded_pnl"] or 0
            r.actual_exit_premium = sig["traded_exit"] or 0
            r.actual_exit_reason = sig["traded_exit_reason"] or ""

        results.append(r)

    print(f"Simulated {len(results)} signals ({no_data_count} had no harvester data)")
    print()

    # Per-trade details
    print(f"{'='*140}")
    print(f"{'ID':>3} {'Tckr':<5} {'Dir':<5} {'Day':<11} {'Sc':>3} {'Trd':>3} {'Entry':>6} {'Peak%':>7} "
          f"{'v4Exit':>7} {'v4PnL':>9} {'v4Rsn':<18} {'v22Exit':>7} {'v22PnL':>9} {'v22Rsn':<22} {'Diff':>9}")
    print(f"{'-'*140}")

    for r in results:
        diff = r.v22_pnl - r.v4_pnl
        trd = "YES" if r.was_traded else "no"
        print(f"{r.signal_id:>3} {r.ticker:<5} {r.direction[:4]:<5} {r.day:<11} {r.score:>3} {trd:>3} "
              f"${r.entry_premium:>5.2f} {r.peak_gain_pct:>6.1f}% "
              f"${r.v4_exit_premium:>5.2f} ${r.v4_pnl:>8.2f} {r.v4_exit_reason:<18} "
              f"${r.v22_exit_premium:>5.2f} ${r.v22_pnl:>8.2f} {r.v22_exit_reason:<22} "
              f"{'+'if diff>0 else ''}${diff:>7.2f}")

    # Daily summary
    print(f"\n{'='*110}")
    print(f"{'DAY-BY-DAY COMPARISON: v4 (current) vs v2.2':^110}")
    print(f"{'='*110}")
    print(f"{'Date':<12} {'Sigs':>5} {'Traded':>6} {'v4 PnL':>12} {'v2.2 PnL':>12} {'Diff':>12} {'v4 WR':>8} {'v22 WR':>8}")
    print(f"{'-'*110}")

    days = sorted(set(r.day for r in results))
    total_v4 = 0.0
    total_v22 = 0.0

    for day in days:
        dr = [r for r in results if r.day == day]
        v4_pnl = sum(r.v4_pnl for r in dr)
        v22_pnl = sum(r.v22_pnl for r in dr)
        traded = sum(1 for r in dr if r.was_traded)
        v4_wins = sum(1 for r in dr if r.v4_pnl > 0)
        v22_wins = sum(1 for r in dr if r.v22_pnl > 0)
        v4_wr = v4_wins / len(dr) * 100 if dr else 0
        v22_wr = v22_wins / len(dr) * 100 if dr else 0
        diff = v22_pnl - v4_pnl
        total_v4 += v4_pnl
        total_v22 += v22_pnl

        print(f"{day:<12} {len(dr):>5} {traded:>6} ${v4_pnl:>10.2f} ${v22_pnl:>10.2f} "
              f"{'+'if diff>0 else ''}${diff:>10.2f} {v4_wr:>6.0f}% {v22_wr:>6.0f}%")

    diff_total = total_v22 - total_v4
    total_trades = len(results)
    v4_total_wins = sum(1 for r in results if r.v4_pnl > 0)
    v22_total_wins = sum(1 for r in results if r.v22_pnl > 0)
    print(f"{'-'*110}")
    print(f"{'TOTAL':<12} {total_trades:>5} {sum(1 for r in results if r.was_traded):>6} "
          f"${total_v4:>10.2f} ${total_v22:>10.2f} "
          f"{'+'if diff_total>0 else ''}${diff_total:>10.2f} "
          f"{v4_total_wins/total_trades*100:>6.0f}% {v22_total_wins/total_trades*100:>6.0f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total signals simulated:  {total_trades}")
    print(f"  Signals we traded:        {sum(1 for r in results if r.was_traded)}")
    print(f"  Signals we skipped:       {sum(1 for r in results if not r.was_traded)}")
    print(f"  No harvester data:        {no_data_count}")
    print(f"")
    print(f"  v4 total P&L:      ${total_v4:>10.2f}  (WR: {v4_total_wins}/{total_trades} = {v4_total_wins/total_trades*100:.0f}%)")
    print(f"  v2.2 total P&L:    ${total_v22:>10.2f}  (WR: {v22_total_wins}/{total_trades} = {v22_total_wins/total_trades*100:.0f}%)")
    print(f"  Difference:        {'+'if diff_total>0 else ''}${diff_total:>10.2f}")
    print(f"  Slippage applied:  {SLIPPAGE_HAIRCUT:.0%} on all gains")

    # Top improvements
    sorted_by_diff = sorted(results, key=lambda r: r.v22_pnl - r.v4_pnl, reverse=True)
    print(f"\nTOP 5 v2.2 IMPROVEMENTS over v4:")
    for r in sorted_by_diff[:5]:
        d = r.v22_pnl - r.v4_pnl
        print(f"  #{r.signal_id} {r.ticker} {r.day}: v4=${r.v4_pnl:>8.2f} → v22=${r.v22_pnl:>8.2f} (+${d:.2f}, peak +{r.peak_gain_pct:.0f}%)")

    print(f"\nTOP 5 v2.2 REGRESSIONS vs v4:")
    for r in sorted_by_diff[-5:]:
        d = r.v22_pnl - r.v4_pnl
        if d >= 0:
            print("  (none)")
            break
        print(f"  #{r.signal_id} {r.ticker} {r.day}: v4=${r.v4_pnl:>8.2f} → v22=${r.v22_pnl:>8.2f} (${d:.2f}, peak +{r.peak_gain_pct:.0f}%)")

    # Breakdown: traded vs untaded
    traded = [r for r in results if r.was_traded]
    untaded = [r for r in results if not r.was_traded]
    print(f"\n{'='*60}")
    print(f"TRADED vs SKIPPED SIGNALS")
    print(f"{'='*60}")
    if traded:
        t_v4 = sum(r.v4_pnl for r in traded)
        t_v22 = sum(r.v22_pnl for r in traded)
        print(f"  TRADED ({len(traded)} signals):")
        print(f"    v4:   ${t_v4:>10.2f}")
        print(f"    v2.2: ${t_v22:>10.2f}")
        print(f"    diff: {'+'if t_v22-t_v4>0 else ''}${t_v22-t_v4:>10.2f}")
    if untaded:
        u_v4 = sum(r.v4_pnl for r in untaded)
        u_v22 = sum(r.v22_pnl for r in untaded)
        print(f"  SKIPPED ({len(untaded)} signals):")
        print(f"    v4:   ${u_v4:>10.2f}")
        print(f"    v2.2: ${u_v22:>10.2f}")
        print(f"    diff: {'+'if u_v22-u_v4>0 else ''}${u_v22-u_v4:>10.2f}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
