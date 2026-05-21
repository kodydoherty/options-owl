#!/usr/bin/env python3
"""Backtest v2.2 sell logic against ALL historical trades in the DB.

Shows daily P&L comparison: actual exits vs v2.2 exits.
Uses actual fill prices where available, MFE from DB as peak,
and 15% slippage haircut on all gains.

v2.2 exit logic (from Vinny's spec):
  - Hard stop: -30% from entry → full exit
  - BE clamp: at +15% gain, floor = entry (never go negative after seeing green)
  - Soft trail (15-35% band): floor = 50% of peak gain (keep half the move)
  - Trail tiers by peak gain:
      35-100%: giveback 0.35 (exit = peak * 0.65)
      100-200%: giveback 0.25 (exit = peak * 0.75)
      200%+:    giveback 0.20 (exit = peak * 0.80)
  - Milestone locks at +200/400/600%: bank 15% of contracts at each level
  - Ticker whip resistance: NVDA/TSLA/AMZN/AVGO/PLTR get 1.3-1.5x wider trails
"""

import json
import sqlite3
import sys
from dataclasses import dataclass


# Whip resistance multipliers (wider trails for volatile tickers)
WHIP_MULTIPLIERS = {
    "NVDA": 1.5,
    "TSLA": 1.5,
    "AMZN": 1.3,
    "AVGO": 1.3,
    "PLTR": 1.3,
}

# Milestone lock levels: at each peak %, bank 15% of remaining contracts
MILESTONE_LEVELS = [200, 400, 600]
MILESTONE_BANK_PCT = 0.15

SLIPPAGE_HAIRCUT = 0.15  # 15% haircut on all gains


@dataclass
class V22Result:
    """Result of applying v2.2 logic to one trade."""
    trade_id: int
    ticker: str
    day: str
    contracts: int
    entry_premium: float
    actual_exit: float
    actual_pnl: float
    peak_gain_pct: float
    v22_exit_premium: float
    v22_exit_reason: str
    v22_pnl_raw: float
    v22_pnl_slipped: float
    milestone_banked_pnl: float  # P&L from milestone-locked contracts


def apply_v22_logic(
    entry: float,
    mfe: float,
    contracts: int,
    ticker: str,
) -> tuple[float, str, float]:
    """Apply v2.2 exit logic based on entry and MFE.

    Returns (v22_exit_premium, exit_reason, milestone_banked_pnl).

    Since we only have entry + MFE (not tick-by-tick data), we simulate:
    - If MFE < entry (never profitable): hard stop at -30% or actual exit
    - Otherwise: calculate the trail exit based on peak gain tier
    """
    if entry <= 0:
        return entry, "no_data", 0.0

    peak_gain_pct = (mfe - entry) / entry * 100 if mfe > entry else 0.0
    whip = WHIP_MULTIPLIERS.get(ticker, 1.0)

    # Never profitable → hard stop
    if mfe <= entry:
        stop_exit = entry * 0.70  # -30% hard stop
        return stop_exit, "hard_stop", 0.0

    # Milestone locks: bank 15% of contracts at +200%, +400%, +600%
    milestone_banked_pnl = 0.0
    remaining_contracts = contracts
    for level in MILESTONE_LEVELS:
        if peak_gain_pct >= level:
            lock_qty = max(1, int(remaining_contracts * MILESTONE_BANK_PCT))
            if remaining_contracts - lock_qty < 1:
                break  # always keep at least 1 contract
            lock_premium = entry * (1 + level / 100)  # exit at the milestone level
            lock_pnl = (lock_premium - entry) * lock_qty * 100
            # Apply slippage
            lock_pnl *= (1 - SLIPPAGE_HAIRCUT)
            milestone_banked_pnl += lock_pnl
            remaining_contracts -= lock_qty

    # Determine trail exit for remaining contracts based on peak gain tier
    if peak_gain_pct >= 200:
        giveback = 0.20 * whip
        exit_prem = mfe * (1 - giveback)
        reason = f"runner_trail_200+({giveback:.0%})"
    elif peak_gain_pct >= 100:
        giveback = 0.25 * whip
        exit_prem = mfe * (1 - giveback)
        reason = f"runner_trail_100+({giveback:.0%})"
    elif peak_gain_pct >= 35:
        giveback = 0.35 * whip
        exit_prem = mfe * (1 - giveback)
        reason = f"trail_35+({giveback:.0%})"
    elif peak_gain_pct >= 15:
        # Soft trail: floor = entry + 50% of gain
        gain = mfe - entry
        exit_prem = entry + gain * 0.50
        reason = "soft_trail_15+"
    else:
        # BE clamp: if it saw +15%, floor = entry
        # If never reached +15%, let it play out (use actual exit)
        exit_prem = entry  # BE clamp
        reason = "be_clamp"

    # Floor: never exit below the hard stop
    hard_stop = entry * 0.70
    if exit_prem < hard_stop:
        exit_prem = hard_stop
        reason = "hard_stop"

    return exit_prem, reason, milestone_banked_pnl


def backtest_trade(row: dict) -> V22Result:
    """Run v2.2 backtest on a single trade."""
    trade_id = row["id"]
    ticker = row["ticker"]
    day = row["day"]
    contracts = row["contracts"]

    # Use Webull fill price if available, else DB entry premium
    fill_price = row.get("webull_entry_fill_price")
    entry = fill_price if fill_price and fill_price > 0 else row["entry"]

    exit_p = row["exit_p"] or entry
    mfe = row["mfe"] or entry
    actual_pnl = row["actual_pnl"] or 0.0

    peak_gain_pct = (mfe - entry) / entry * 100 if entry > 0 and mfe > entry else 0.0

    v22_exit, v22_reason, milestone_pnl = apply_v22_logic(entry, mfe, contracts, ticker)

    # Calculate v2.2 P&L for remaining contracts (after milestones)
    milestone_contracts = 0
    for level in MILESTONE_LEVELS:
        if peak_gain_pct >= level:
            lock_qty = max(1, int((contracts - milestone_contracts) * MILESTONE_BANK_PCT))
            if contracts - milestone_contracts - lock_qty < 1:
                break
            milestone_contracts += lock_qty

    remaining = contracts - milestone_contracts
    v22_pnl_raw = (v22_exit - entry) * remaining * 100 + milestone_pnl / (1 - SLIPPAGE_HAIRCUT)  # raw before slippage on main
    v22_main_pnl = (v22_exit - entry) * remaining * 100

    # Apply slippage to main position gains (not losses)
    if v22_main_pnl > 0:
        v22_main_slipped = v22_main_pnl * (1 - SLIPPAGE_HAIRCUT)
    else:
        v22_main_slipped = v22_main_pnl  # no slippage on losses

    v22_pnl_slipped = v22_main_slipped + milestone_pnl  # milestone already slipped

    return V22Result(
        trade_id=trade_id,
        ticker=ticker,
        day=day,
        contracts=contracts,
        entry_premium=entry,
        actual_exit=exit_p,
        actual_pnl=actual_pnl,
        peak_gain_pct=peak_gain_pct,
        v22_exit_premium=v22_exit,
        v22_exit_reason=v22_reason,
        v22_pnl_raw=v22_pnl_raw,
        v22_pnl_slipped=v22_pnl_slipped,
        milestone_banked_pnl=milestone_pnl,
    )


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, ticker, option_type, date(opened_at) as day, contracts,
               premium_per_contract as entry, exit_premium as exit_p,
               mfe_premium as mfe, pnl_dollars as actual_pnl, exit_reason,
               webull_order_id, webull_entry_fill_price
        FROM paper_trades
        WHERE status='closed' AND parent_trade_id IS NULL
        ORDER BY opened_at
    """).fetchall()

    results: list[V22Result] = []
    for row in rows:
        r = backtest_trade(dict(row))
        results.append(r)

    # Print per-trade details
    print(f"\n{'='*120}")
    print(f"{'ID':>3} {'Ticker':<6} {'Day':<12} {'Ct':>3} {'Entry':>7} {'Peak%':>7} "
          f"{'ActExit':>8} {'ActPnL':>9} {'v22Exit':>8} {'v22PnL':>9} {'Diff':>9} {'v22 Reason':<25}")
    print(f"{'-'*120}")

    for r in results:
        diff = r.v22_pnl_slipped - r.actual_pnl
        diff_color = "+" if diff > 0 else ""
        print(f"{r.trade_id:>3} {r.ticker:<6} {r.day:<12} {r.contracts:>3} "
              f"${r.entry_premium:>5.2f} {r.peak_gain_pct:>6.1f}% "
              f"${r.actual_exit:>6.2f} ${r.actual_pnl:>8.2f} "
              f"${r.v22_exit_premium:>6.2f} ${r.v22_pnl_slipped:>8.2f} "
              f"{diff_color}${diff:>7.2f} {r.v22_exit_reason:<25}")

    # Daily summary
    print(f"\n{'='*100}")
    print(f"\n{'DAY-BY-DAY COMPARISON':^100}")
    print(f"{'='*100}")
    print(f"{'Date':<12} {'Trades':>6} {'ActualPnL':>12} {'v2.2 PnL':>12} {'Diff':>12} {'Act WR':>8} {'v22 WR':>8}")
    print(f"{'-'*100}")

    days = sorted(set(r.day for r in results))
    total_actual = 0.0
    total_v22 = 0.0

    for day in days:
        day_results = [r for r in results if r.day == day]
        act_pnl = sum(r.actual_pnl for r in day_results)
        v22_pnl = sum(r.v22_pnl_slipped for r in day_results)
        act_wins = sum(1 for r in day_results if r.actual_pnl > 0)
        v22_wins = sum(1 for r in day_results if r.v22_pnl_slipped > 0)
        act_wr = act_wins / len(day_results) * 100
        v22_wr = v22_wins / len(day_results) * 100
        diff = v22_pnl - act_pnl
        total_actual += act_pnl
        total_v22 += v22_pnl

        diff_sign = "+" if diff > 0 else ""
        print(f"{day:<12} {len(day_results):>6} ${act_pnl:>10.2f} ${v22_pnl:>10.2f} "
              f"{diff_sign}${diff:>10.2f} {act_wr:>6.0f}% {v22_wr:>6.0f}%")

    diff_total = total_v22 - total_actual
    diff_sign = "+" if diff_total > 0 else ""
    print(f"{'-'*100}")
    total_trades = len(results)
    act_total_wins = sum(1 for r in results if r.actual_pnl > 0)
    v22_total_wins = sum(1 for r in results if r.v22_pnl_slipped > 0)
    act_total_wr = act_total_wins / total_trades * 100
    v22_total_wr = v22_total_wins / total_trades * 100
    print(f"{'TOTAL':<12} {total_trades:>6} ${total_actual:>10.2f} ${total_v22:>10.2f} "
          f"{diff_sign}${diff_total:>10.2f} {act_total_wr:>6.0f}% {v22_total_wr:>6.0f}%")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total trades:      {total_trades}")
    print(f"  Actual total P&L:  ${total_actual:>10.2f}")
    print(f"  v2.2 total P&L:    ${total_v22:>10.2f}")
    print(f"  Improvement:       {diff_sign}${diff_total:>10.2f}")
    if total_actual != 0:
        mult = total_v22 / total_actual if total_actual > 0 else float('inf')
        print(f"  Multiple:          {mult:.1f}x")
    print(f"  Actual win rate:   {act_total_wr:.0f}%")
    print(f"  v2.2 win rate:     {v22_total_wr:.0f}%")
    print(f"  Slippage applied:  {SLIPPAGE_HAIRCUT:.0%} on all gains")

    # Biggest improvements
    print(f"\n{'='*60}")
    print(f"TOP 5 IMPROVEMENTS (v2.2 vs actual)")
    print(f"{'='*60}")
    sorted_by_diff = sorted(results, key=lambda r: r.v22_pnl_slipped - r.actual_pnl, reverse=True)
    for r in sorted_by_diff[:5]:
        diff = r.v22_pnl_slipped - r.actual_pnl
        print(f"  #{r.trade_id} {r.ticker} {r.day}: actual ${r.actual_pnl:>8.2f} → v2.2 ${r.v22_pnl_slipped:>8.2f} "
              f"(+${diff:.2f}, peak +{r.peak_gain_pct:.0f}%)")

    # Biggest regressions
    print(f"\nBOTTOM 5 (v2.2 worse than actual)")
    for r in sorted_by_diff[-5:]:
        diff = r.v22_pnl_slipped - r.actual_pnl
        if diff >= 0:
            print(f"  (none — v2.2 improved all trades)")
            break
        print(f"  #{r.trade_id} {r.ticker} {r.day}: actual ${r.actual_pnl:>8.2f} → v2.2 ${r.v22_pnl_slipped:>8.2f} "
              f"(${diff:.2f}, peak +{r.peak_gain_pct:.0f}%)")

    conn.close()


if __name__ == "__main__":
    main()
