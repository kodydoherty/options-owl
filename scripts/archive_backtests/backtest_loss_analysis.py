#!/usr/bin/env python3
"""Analyze losing trades: how deep do they go, do any recover?

For every trade that went negative, shows:
  - Max drawdown (MAE = max adverse excursion)
  - Did it recover after hitting the low?
  - Final outcome (EOD price)
  - What would have happened at various stop levels

This tells us where to set the loss cut to avoid zero-outs
without killing recoveries.

Usage:
  python scripts/backtest_loss_analysis.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backtest_exit_v3 import (
    PremiumTick, TradeInfo, load_all_premiums, load_trades_from_db,
)


def analyze_losses(trades: list[TradeInfo]):
    """For each trade with premium data, track the full drawdown path."""

    print(f"\n{'Bot':<6} {'#':>3} {'Ticker':<6} {'Entry':>6} {'Low':>6} {'MAE%':>6} "
          f"{'Recovered?':>10} {'Best After Low':>14} {'Final':>6} {'Final%':>7} "
          f"{'Verdict':<20}")
    print("-" * 120)

    # Track stats for stop-level analysis
    stop_analysis: dict[int, dict] = {}  # stop_pct -> {saved, killed, total_diff}

    for stop_pct in range(20, 65, 5):
        stop_analysis[stop_pct] = {"saved_count": 0, "saved_dollars": 0.0,
                                    "killed_count": 0, "killed_dollars": 0.0}

    loss_trades = []

    for trade in trades:
        if not trade.premium_path:
            continue

        entry = trade.entry_premium
        contracts = trade.contracts

        # Track the full path
        low = entry
        low_time = trade.premium_path[0].timestamp
        high_after_low = entry
        final = trade.premium_path[-1].premium

        for tick in trade.premium_path:
            if tick.premium < low:
                low = tick.premium
                low_time = tick.timestamp
                high_after_low = tick.premium  # reset
            elif tick.timestamp > low_time:
                high_after_low = max(high_after_low, tick.premium)

        mae_pct = (entry - low) / entry * 100
        final_pct = (final - entry) / entry * 100
        recovery_pct = (high_after_low - low) / low * 100 if low > 0 else 0
        recovered = high_after_low > entry  # did it get back above entry after hitting low?

        # Only show trades that went meaningfully negative
        if mae_pct < 10:
            continue

        loss_trades.append({
            "trade": trade,
            "mae_pct": mae_pct,
            "low": low,
            "recovered": recovered,
            "high_after_low": high_after_low,
            "final": final,
            "final_pct": final_pct,
            "recovery_pct": recovery_pct,
        })

        # What happens at each stop level?
        for stop_pct in stop_analysis:
            # Would this stop have been hit?
            if mae_pct >= stop_pct:
                # Stop fires at entry * (1 - stop_pct/100)
                stop_price = entry * (1 - stop_pct / 100)
                stop_pnl = (stop_price - entry) * contracts * 100
                eod_pnl = (final - entry) * contracts * 100

                diff = stop_pnl - eod_pnl  # positive = stop saved money

                if diff > 0:
                    stop_analysis[stop_pct]["saved_count"] += 1
                    stop_analysis[stop_pct]["saved_dollars"] += diff
                else:
                    stop_analysis[stop_pct]["killed_count"] += 1
                    stop_analysis[stop_pct]["killed_dollars"] += abs(diff)

        # Determine verdict
        if mae_pct >= 50 and not recovered:
            verdict = "DEAD (never recovered)"
        elif mae_pct >= 50 and recovered:
            verdict = "PHOENIX (recovered!)"
        elif mae_pct >= 30 and not recovered:
            verdict = "Slow bleed"
        elif mae_pct >= 30 and recovered:
            verdict = "V-shaped recovery"
        elif recovered:
            verdict = "Minor dip, recovered"
        else:
            verdict = "Dip, no recovery"

        rec_str = f"Yes→${high_after_low:.2f}" if recovered else "No"

        print(f"{trade.bot:<6} {trade.trade_id:>3} {trade.ticker:<6} ${entry:.2f} "
              f"${low:.2f} {mae_pct:>5.1f}% {rec_str:>10} "
              f"${high_after_low:>6.2f} ({recovery_pct:>+.0f}%) "
              f"${final:.2f} {final_pct:>+6.1f}% {verdict:<20}")

    # Summary: optimal stop level
    print("\n" + "=" * 100)
    print("STOP LEVEL ANALYSIS: What happens if we cut at X% loss?")
    print("=" * 100)
    print(f"\n{'Stop %':>7} {'Trades Hit':>10} {'Saved $':>10} {'Killed $':>10} {'Net':>10} {'Verdict':<20}")
    print("-" * 80)

    for stop_pct in sorted(stop_analysis.keys()):
        s = stop_analysis[stop_pct]
        total_hit = s["saved_count"] + s["killed_count"]
        net = s["saved_dollars"] - s["killed_dollars"]
        verdict = "BETTER" if net > 0 else "WORSE"
        print(f"  -{stop_pct}%   {total_hit:>10} ${s['saved_dollars']:>9.2f} "
              f"${s['killed_dollars']:>9.2f} ${net:>+9.2f} {verdict}")

    # Show the deep losers: trades that went below -50%
    deep_losers = [t for t in loss_trades if t["mae_pct"] >= 40]
    if deep_losers:
        print(f"\n{'='*100}")
        print("DEEP LOSERS (MAE >= 40%): Did any recover?")
        print(f"{'='*100}")
        for t in sorted(deep_losers, key=lambda x: -x["mae_pct"]):
            trade = t["trade"]
            print(f"  {trade.bot} #{trade.trade_id} {trade.ticker}: "
                  f"entry ${trade.entry_premium:.2f} → low ${t['low']:.2f} "
                  f"(-{t['mae_pct']:.0f}%) → "
                  f"{'RECOVERED to $' + str(round(t['high_after_low'], 2)) if t['recovered'] else 'never recovered'} → "
                  f"final ${t['final']:.2f} ({t['final_pct']:+.0f}%)")

    # Recovery rate by depth
    print(f"\n{'='*100}")
    print("RECOVERY RATE BY DRAWDOWN DEPTH")
    print(f"{'='*100}")
    buckets = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 100)]
    for lo, hi in buckets:
        in_bucket = [t for t in loss_trades if lo <= t["mae_pct"] < hi]
        if not in_bucket:
            continue
        recovered = sum(1 for t in in_bucket if t["recovered"])
        profitable_eod = sum(1 for t in in_bucket if t["final_pct"] > 0)
        print(f"  -{lo}% to -{hi}%: {len(in_bucket)} trades, "
              f"{recovered} recovered ({recovered/len(in_bucket)*100:.0f}%), "
              f"{profitable_eod} profitable at EOD ({profitable_eod/len(in_bucket)*100:.0f}%)")


def main():
    for bot in ("kody", "adam", "vinny", "yank"):
        db_dst = f"/tmp/db_{bot}.sqlite"
        if not os.path.exists(db_dst):
            print(f"Downloading {bot} DB from droplet...")
            db_src = f"root@129.212.138.145:/root/options-owl/journal/owlet-{bot}/raw_messages.db"
            os.system(f'scp -i "$HOME/.ssh/id_ed25519_do" {db_src} {db_dst} 2>/dev/null')

    trades = load_trades_from_db()
    print(f"Loaded {len(trades)} parent LIVE trades")

    all_premiums = {}
    for bot in ("kody", "adam", "vinny", "yank"):
        all_premiums[bot] = load_all_premiums(bot)

    matched = 0
    for trade in trades:
        bot_premiums = all_premiums.get(trade.bot, {})
        path = []
        for fmt in (f"{trade.strike}", f"{trade.strike:.1f}", f"{trade.strike:.0f}",
                     f"{int(trade.strike)}.0" if trade.strike == int(trade.strike) else ""):
            key = (trade.ticker, fmt)
            if key in bot_premiums:
                path = bot_premiums[key]
                break
        if path:
            trade_end = trade.entry_time + timedelta(hours=8)
            trade.premium_path = [
                t for t in path
                if t.timestamp >= trade.entry_time and t.timestamp <= trade_end
            ]
            if trade.premium_path:
                matched += 1

    print(f"Matched premium paths for {matched}/{len(trades)} trades")
    analyze_losses(trades)


if __name__ == "__main__":
    main()
