#!/usr/bin/env python3
"""Backtest: continuous thesis revalidation as the loss-cutting mechanism.

Instead of a hard % stop, when a trade drops below a threshold (-30%),
start evaluating trend health every cycle using price action as a proxy
for multi-TF candle analysis:

  - Is the decline accelerating or decelerating?
  - Is price making new lows or finding support?
  - Has there been any bounce (sign of buying pressure)?
  - How much time is left on the option?

If the trend is confirmed dead → cut losses.
If showing support/recovery signs → hold and let it play out.

Sweeps multiple configurations to find the optimal combo.

Usage:
  python scripts/backtest_thesis_cut.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backtest_exit_v3 import (
    PremiumTick, TradeInfo, load_all_premiums, load_trades_from_db,
    BounceState,
)


@dataclass
class ThesisCutConfig:
    label: str

    # When to start checking thesis
    check_threshold_pct: float = 30.0  # start thesis checks below this % loss

    # Trend health evaluation
    lookback_ticks: int = 6            # how many recent ticks to analyze
    new_low_count_exit: int = 4        # exit if N of last lookback ticks made new lows
    decel_hold: bool = True            # hold if decline is decelerating (finding support)
    bounce_hold_pct: float = 5.0       # hold if bounced this % from recent low
    min_ticks_before_cut: int = 6      # need at least N ticks in the zone before cutting

    # Time urgency
    time_urgency_minutes: float = 60.0  # tighten criteria with < N min left
    time_urgency_new_low_exit: int = 2  # fewer new lows needed near expiry

    # Profit floor (always on — proven winner)
    floor_activation_pct: float = 15.0
    floor_ratchet_pct: float = 60.0

    # Adaptive trail (always on)
    adaptive_activation: float = 35.0
    adaptive_width: float = 35.0


def simulate_thesis_cut(trade: TradeInfo, cfg: ThesisCutConfig) -> tuple[float, float, str]:
    if not trade.premium_path:
        return trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason

    entry = trade.entry_premium
    contracts = trade.contracts

    try:
        expiry_dt = datetime.strptime(trade.expiry_date, "%Y-%m-%d").replace(hour=16, minute=0)
    except (ValueError, TypeError):
        expiry_dt = trade.entry_time + timedelta(hours=6)

    peak = entry
    profit_floor: float | None = None
    recent_premiums: list[float] = []  # rolling window of recent ticks
    ticks_in_danger_zone = 0
    running_low = entry  # lowest premium seen since entering danger zone

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        peak_gain_pct = (peak - entry) / entry * 100
        drop_from_entry_pct = (entry - p) / entry * 100 if p < entry else 0
        time_remaining_min = max(0, (expiry_dt - tick.timestamp).total_seconds() / 60)

        # Track recent premiums for trend analysis
        recent_premiums.append(p)
        if len(recent_premiums) > cfg.lookback_ticks * 2:
            recent_premiums = recent_premiums[-(cfg.lookback_ticks * 2):]

        # === Profit floor (time-aware ratchet) ===
        if peak_gain_pct >= cfg.floor_activation_pct:
            if time_remaining_min < 15:
                ratchet = 95.0
            elif time_remaining_min < 30:
                ratchet = 90.0
            elif time_remaining_min < 60:
                ratchet = 80.0
            elif time_remaining_min < 120:
                ratchet = 70.0
            else:
                ratchet = cfg.floor_ratchet_pct

            new_floor = entry + (peak - entry) * (ratchet / 100)
            if profit_floor is None or new_floor > profit_floor:
                profit_floor = new_floor

        if profit_floor is not None and p <= profit_floor:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "profit_floor"

        # === Adaptive trail ===
        if peak_gain_pct >= cfg.adaptive_activation:
            trail_drop = (peak - p) / peak * 100
            if trail_drop >= cfg.adaptive_width:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "adaptive_trail"

        # === Continuous thesis revalidation (when sufficiently negative) ===
        if drop_from_entry_pct >= cfg.check_threshold_pct:
            ticks_in_danger_zone += 1
            running_low = min(running_low, p)

            # Need minimum ticks before making a cut decision
            if ticks_in_danger_zone < cfg.min_ticks_before_cut:
                continue

            if len(recent_premiums) < cfg.lookback_ticks:
                continue

            window = recent_premiums[-cfg.lookback_ticks:]

            # --- Trend health checks ---

            # 1. New low count: how many of the last N ticks made a new running low?
            new_low_count = 0
            window_low = window[0]
            for wp in window[1:]:
                if wp < window_low:
                    new_low_count += 1
                    window_low = wp

            # 2. Deceleration: is the decline slowing down?
            # Compare first half velocity vs second half velocity
            half = len(window) // 2
            if half >= 2:
                first_half_change = (window[half - 1] - window[0]) / entry * 100
                second_half_change = (window[-1] - window[half]) / entry * 100
                decelerating = second_half_change > first_half_change  # less negative = decelerating
            else:
                decelerating = False

            # 3. Bounce from low: has premium recovered from the recent low?
            recent_low = min(window)
            if recent_low > 0:
                bounce_from_low = (p - recent_low) / recent_low * 100
            else:
                bounce_from_low = 0

            # --- Time urgency adjustments ---
            if time_remaining_min < cfg.time_urgency_minutes:
                new_low_threshold = cfg.time_urgency_new_low_exit
            else:
                new_low_threshold = cfg.new_low_count_exit

            # --- Decision ---
            # HOLD if showing signs of life
            if cfg.decel_hold and decelerating and bounce_from_low > 2:
                continue  # support forming, hold

            if bounce_from_low >= cfg.bounce_hold_pct:
                continue  # meaningful bounce, hold

            # EXIT if trend confirmed dead
            if new_low_count >= new_low_threshold:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "thesis_dead"

            # EXIT if near expiry and still deeply negative (theta death)
            if time_remaining_min < 30 and drop_from_entry_pct >= 40:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "thesis_time_cut"

        else:
            # Reset danger zone state if we recover above threshold
            ticks_in_danger_zone = 0
            running_low = entry

    # EOD
    last = trade.premium_path[-1]
    pnl = (last.premium - entry) * contracts * 100
    return last.premium, pnl, "eod_cutoff"


def simulate_v2_current(trade: TradeInfo) -> tuple[float, float, str]:
    """V2.1 baseline for comparison."""
    if not trade.premium_path:
        return trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason

    entry = trade.entry_premium
    contracts = trade.contracts
    peak = entry
    grace_end = trade.entry_time + timedelta(minutes=20)

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        drop_pct = (entry - p) / entry * 100

        # Catastrophic stop
        if drop_pct >= 45:
            return p, (p - entry) * contracts * 100, "catastrophic_stop"

        # Grace period
        if tick.timestamp < grace_end:
            continue

        # Normal stop
        if drop_pct >= 30:
            return p, (p - entry) * contracts * 100, "stop_loss"

        # Profit retrace
        peak_gain_pct = (peak - entry) / entry * 100
        if peak_gain_pct >= 25:
            profit_at_peak = peak - entry
            given_back = (profit_at_peak - (p - entry)) / profit_at_peak * 100
            if given_back >= 50:
                return p, (p - entry) * contracts * 100, "profit_retrace"

        # Adaptive trail
        if peak_gain_pct >= 35:
            trail_drop = (peak - p) / peak * 100
            if trail_drop >= 35:
                return p, (p - entry) * contracts * 100, "adaptive_trail"

    last = trade.premium_path[-1]
    return last.premium, (last.premium - entry) * contracts * 100, "eod_cutoff"


def run_sweep(trades: list[TradeInfo]):
    # V2.1 baseline
    v2_total = 0.0
    v2_wins = 0
    v2_results = {}
    for trade in trades:
        ep, pnl, reason = simulate_v2_current(trade) if trade.premium_path else (
            trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason)
        v2_total += pnl
        if pnl > 0:
            v2_wins += 1
        v2_results[trade.trade_id] = (pnl, reason)

    total = len(trades)
    print(f"V2.1 BASELINE: P&L=${v2_total:+.2f}, Wins={v2_wins}/{total} ({v2_wins/total*100:.1f}%)")

    # Sweep configs
    configs: list[ThesisCutConfig] = []

    # Vary: check threshold, new low count, bounce hold, lookback
    for threshold in [25, 30, 35, 40]:
        for new_low_exit in [3, 4, 5]:
            for bounce_hold in [3.0, 5.0, 8.0]:
                for lookback in [4, 6, 8]:
                    for min_ticks in [4, 6, 8]:
                        configs.append(ThesisCutConfig(
                            label=f"t>{threshold}% lows>{new_low_exit}/{lookback} bounce>{bounce_hold}% wait>{min_ticks}",
                            check_threshold_pct=threshold,
                            new_low_count_exit=new_low_exit,
                            bounce_hold_pct=bounce_hold,
                            lookback_ticks=lookback,
                            min_ticks_before_cut=min_ticks,
                        ))

    # Also test: floor only (no thesis cut at all) as control
    configs.append(ThesisCutConfig(
        label="CONTROL: floor only, no cut",
        check_threshold_pct=999,  # never fires
    ))

    print(f"\nTesting {len(configs)} configurations...\n")

    results: list[tuple[ThesisCutConfig, float, int, int]] = []

    for cfg in configs:
        pnl_total = 0.0
        wins = 0
        thesis_cuts = 0

        for trade in trades:
            if not trade.premium_path:
                pnl_total += trade.actual_pnl
                if trade.actual_pnl > 0:
                    wins += 1
                continue

            _, pnl, reason = simulate_thesis_cut(trade, cfg)
            pnl_total += pnl
            if pnl > 0:
                wins += 1
            if reason in ("thesis_dead", "thesis_time_cut"):
                thesis_cuts += 1

        results.append((cfg, pnl_total, wins, thesis_cuts))

    results.sort(key=lambda x: -x[1])

    print(f"{'Rank':>4} {'P&L':>10} {'Diff':>10} {'Wins':>5} {'WR%':>6} {'Cuts':>5}  Config")
    print("-" * 130)
    for i, (cfg, pnl, wins, cuts) in enumerate(results[:25]):
        wr = wins / total * 100
        diff = pnl - v2_total
        print(f"{i+1:>4} ${pnl:>+9.2f} ${diff:>+9.2f} {wins:>5} {wr:>5.1f}% {cuts:>5}  {cfg.label}")

    print(f"\n{'...'}")
    print(f"\nCONTROL (floor only, no thesis cut):")
    for cfg, pnl, wins, cuts in results:
        if "CONTROL" in cfg.label:
            wr = wins / total * 100
            diff = pnl - v2_total
            print(f"     ${pnl:>+9.2f} ${diff:>+9.2f} {wins:>5} {wr:>5.1f}% {cuts:>5}  {cfg.label}")
            break

    # Detailed breakdown for top 3
    print("\n" + "=" * 130)
    print("DETAILED BREAKDOWN — TOP 3 CONFIGS")
    print("=" * 130)

    for rank, (cfg, total_pnl, total_wins, total_cuts) in enumerate(results[:3], 1):
        diff = total_pnl - v2_total
        print(f"\n--- RANK #{rank}: {cfg.label} ---")
        print(f"    P&L: ${total_pnl:+.2f} (diff: ${diff:+.2f}), "
              f"Wins: {total_wins}/{total} ({total_wins/total*100:.1f}%), "
              f"Thesis cuts: {total_cuts}")

        improved = []
        worsened = []

        for trade in trades:
            if not trade.premium_path:
                continue
            _, v3_pnl, v3_reason = simulate_thesis_cut(trade, cfg)
            v2_pnl, v2_reason = v2_results[trade.trade_id]
            d = v3_pnl - v2_pnl
            if d > 1:
                improved.append((trade, v2_pnl, v2_reason, v3_pnl, v3_reason, d))
            elif d < -1:
                worsened.append((trade, v2_pnl, v2_reason, v3_pnl, v3_reason, d))

        if improved:
            print(f"\n    IMPROVED ({len(improved)}, ${sum(d[5] for d in improved):+.2f}):")
            for t, v2p, v2r, v3p, v3r, d in sorted(improved, key=lambda x: -x[5]):
                print(f"      {t.bot} #{t.trade_id} {t.ticker}: "
                      f"${v2p:.2f} ({v2r}) → ${v3p:.2f} ({v3r}) = ${d:+.2f}")

        if worsened:
            print(f"\n    WORSENED ({len(worsened)}, ${sum(d[5] for d in worsened):+.2f}):")
            for t, v2p, v2r, v3p, v3r, d in sorted(worsened, key=lambda x: x[5]):
                print(f"      {t.bot} #{t.trade_id} {t.ticker}: "
                      f"${v2p:.2f} ({v2r}) → ${v3p:.2f} ({v3r}) = ${d:+.2f}")


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
    run_sweep(trades)


if __name__ == "__main__":
    main()
