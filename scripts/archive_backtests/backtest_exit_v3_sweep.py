#!/usr/bin/env python3
"""Sweep thesis revalidation + emergency exit thresholds for Exit v3.

Keeps profit floor and bounce-fade (proven winners), tests many combinations
of thesis revalidation sensitivity and emergency exit triggers.

Usage:
  python scripts/backtest_exit_v3_sweep.py
"""
from __future__ import annotations

import os
import sys
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Re-use data structures and loaders from the main backtest
sys.path.insert(0, str(Path(__file__).parent))
from backtest_exit_v3 import (
    PremiumTick, TradeInfo, load_all_premiums, load_trades_from_db,
    simulate_v2_exit, ThesisState, BounceState,
)


# ---------------------------------------------------------------------------
# Parameterized V3 simulation
# ---------------------------------------------------------------------------

@dataclass
class V3Config:
    """All tunable v3 parameters."""
    # Thesis revalidation
    thesis_min_drop_pct: float = 25.0       # only fire thesis check below this drop %
    thesis_decline_rate: float = 3.0        # % per minute to trigger exit
    thesis_time_drop_pct: float = 15.0      # drop % threshold when < 60min left
    thesis_deep_drop_pct: float = 30.0      # deep decline threshold (< 120min)

    # Emergency exit
    emergency_min_loss_pct: float = 10.0    # only emergency exit if down this %
    emergency_time_min: float = 15.0        # minutes remaining to trigger emergency

    # Profit floor (keep fixed — proven good)
    floor_activation_pct: float = 15.0
    floor_ratchet_pct: float = 60.0

    # Bounce-fade (keep fixed — proven good)
    bounce_watch_pct: float = 50.0
    bounce_min_recovery_pct: float = 10.0
    bounce_fade_pct: float = 15.0

    # Time urgency (keep fixed)
    enable_time_urgency_floor: bool = True  # tighten floor with time
    enable_time_urgency_emergency: bool = True  # emergency exit with time

    def label(self) -> str:
        return (f"thesis>={self.thesis_min_drop_pct:.0f}%,rate>{self.thesis_decline_rate:.1f},"
                f"emerg>={self.emergency_min_loss_pct:.0f}%@{self.emergency_time_min:.0f}m")


def simulate_v3_parameterized(trade: TradeInfo, cfg: V3Config) -> tuple[float, float, str]:
    """V3 exit sim with configurable thresholds."""
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
    bounce_state = BounceState()
    last_revalidation: datetime | None = None

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        gain_pct = (p - entry) / entry * 100
        peak_gain_pct = (peak - entry) / entry * 100
        drop_from_entry_pct = (entry - p) / entry * 100 if p < entry else 0
        time_remaining_min = max(0, (expiry_dt - tick.timestamp).total_seconds() / 60)
        elapsed_min = max(1, (tick.timestamp - trade.entry_time).total_seconds() / 60)

        # --- Time urgency (floor tightening only) ---
        if cfg.enable_time_urgency_floor:
            if time_remaining_min < 15:
                floor_ratchet = 95.0
            elif time_remaining_min < 30:
                floor_ratchet = 90.0
            elif time_remaining_min < 60:
                floor_ratchet = 80.0
            elif time_remaining_min < 120:
                floor_ratchet = 70.0
            else:
                floor_ratchet = cfg.floor_ratchet_pct
        else:
            floor_ratchet = cfg.floor_ratchet_pct

        # Fade % for bounce detection (tighten with time)
        if time_remaining_min < 15:
            fade_pct = 5.0
        elif time_remaining_min < 30:
            fade_pct = 8.0
        elif time_remaining_min < 60:
            fade_pct = 10.0
        else:
            fade_pct = cfg.bounce_fade_pct

        # === GATE 1: Thesis Revalidation (only when sufficiently negative) ===
        if p < entry and drop_from_entry_pct >= cfg.thesis_min_drop_pct:
            # Rate-limit checks to every 30s
            should_check = (
                last_revalidation is None or
                (tick.timestamp - last_revalidation).total_seconds() >= 30
            )
            if should_check:
                last_revalidation = tick.timestamp
                decline_rate = drop_from_entry_pct / elapsed_min

                # Check 1: Deep + accelerating = exit
                if drop_from_entry_pct > cfg.thesis_min_drop_pct and decline_rate > cfg.thesis_decline_rate:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "thesis_invalid"

                # Check 2: Moderate decline + low time = exit
                if drop_from_entry_pct > cfg.thesis_time_drop_pct and time_remaining_min < 60:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "thesis_invalid"

                # Check 3: Deep decline + moderate time = exit
                if drop_from_entry_pct > cfg.thesis_deep_drop_pct and time_remaining_min < 120:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "thesis_invalid"

        # === GATE 2: Ratcheting Profit Floor (proven — keep as-is) ===
        if peak_gain_pct >= cfg.floor_activation_pct:
            new_floor = entry + (peak - entry) * (floor_ratchet / 100)
            if profit_floor is None or new_floor > profit_floor:
                profit_floor = new_floor

        if profit_floor is not None and p <= profit_floor:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "profit_floor"

        # === GATE 3: Bounce-and-Fade (proven — keep as-is) ===
        if drop_from_entry_pct >= cfg.bounce_watch_pct and not bounce_state.in_bounce_watch:
            bounce_state.in_bounce_watch = True
            bounce_state.bounce_low = p
            bounce_state.bounce_detected = False

        if bounce_state.in_bounce_watch:
            bounce_state.bounce_low = min(bounce_state.bounce_low, p)

            if not bounce_state.bounce_detected and bounce_state.bounce_low > 0:
                recovery_pct = (p - bounce_state.bounce_low) / bounce_state.bounce_low * 100
                if recovery_pct >= cfg.bounce_min_recovery_pct:
                    bounce_state.bounce_detected = True
                    bounce_state.bounce_high = p

            if bounce_state.bounce_detected:
                bounce_state.bounce_high = max(bounce_state.bounce_high, p)
                if bounce_state.bounce_high > 0:
                    fade_from_bounce = (bounce_state.bounce_high - p) / bounce_state.bounce_high * 100
                    if fade_from_bounce >= fade_pct:
                        pnl = (p - entry) * contracts * 100
                        return p, pnl, "bounce_fade"

            # Time-critical bounce sell
            if time_remaining_min < 30 and p > bounce_state.bounce_low * 1.03:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "bounce_fade_critical"

        # === GATE 4: Emergency Exit (only deeply negative + nearly expired) ===
        if cfg.enable_time_urgency_emergency:
            if time_remaining_min < cfg.emergency_time_min and gain_pct < -cfg.emergency_min_loss_pct:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "emergency_exit"

    # EOD
    last = trade.premium_path[-1]
    pnl = (last.premium - entry) * contracts * 100
    return last.premium, pnl, "eod_cutoff"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(trades: list[TradeInfo]):
    """Test many threshold combos and rank by total P&L improvement."""

    # Compute v2.1 baseline
    v2_results = {}
    v2_total = 0.0
    v2_wins = 0
    for trade in trades:
        if not trade.premium_path:
            v2_results[trade.trade_id] = (trade.actual_exit_premium, trade.actual_pnl, trade.actual_exit_reason)
            v2_total += trade.actual_pnl
            if trade.actual_pnl > 0:
                v2_wins += 1
        else:
            ep, pnl, reason = simulate_v2_exit(trade)
            v2_results[trade.trade_id] = (ep, pnl, reason)
            v2_total += pnl
            if pnl > 0:
                v2_wins += 1

    total_trades = len(trades)
    print(f"V2.1 BASELINE: P&L=${v2_total:+.2f}, Wins={v2_wins}/{total_trades} ({v2_wins/total_trades*100:.1f}%)")
    print()

    # Define sweep ranges
    thesis_min_drops = [20, 25, 30, 35, 40, 50]       # % drop before thesis fires
    thesis_decline_rates = [2.0, 3.0, 5.0, 8.0]       # %/min acceleration threshold
    thesis_time_drops = [15, 20, 25, 30]               # % drop when < 60min left
    thesis_deep_drops = [30, 35, 40, 45]               # % drop when < 120min left
    emergency_losses = [15, 25, 35, 50]                # min loss % for emergency exit
    emergency_times = [10, 15, 20, 30]                 # minutes remaining for emergency

    # Also test disabling thesis / emergency entirely
    configs: list[V3Config] = []

    # Config 0: floor + bounce only (no thesis, no emergency)
    configs.append(V3Config(
        thesis_min_drop_pct=999,  # effectively disabled
        enable_time_urgency_emergency=False,
    ))

    # Config 1: floor + bounce + emergency only (no thesis)
    for em_loss, em_time in itertools.product(emergency_losses, emergency_times):
        configs.append(V3Config(
            thesis_min_drop_pct=999,
            emergency_min_loss_pct=em_loss,
            emergency_time_min=em_time,
        ))

    # Config 2: floor + bounce + thesis only (no emergency)
    for t_drop, t_rate in itertools.product(thesis_min_drops, thesis_decline_rates):
        configs.append(V3Config(
            thesis_min_drop_pct=t_drop,
            thesis_decline_rate=t_rate,
            thesis_time_drop_pct=t_drop,      # same as min drop
            thesis_deep_drop_pct=t_drop + 10,  # 10% deeper for longer timeframe
            enable_time_urgency_emergency=False,
        ))

    # Config 3: full v3 with best-looking combos
    for t_drop, t_rate, em_loss, em_time in itertools.product(
        [30, 35, 40, 50], [3.0, 5.0, 8.0], [25, 35, 50], [10, 15]
    ):
        configs.append(V3Config(
            thesis_min_drop_pct=t_drop,
            thesis_decline_rate=t_rate,
            thesis_time_drop_pct=t_drop,
            thesis_deep_drop_pct=t_drop + 10,
            emergency_min_loss_pct=em_loss,
            emergency_time_min=em_time,
        ))

    print(f"Testing {len(configs)} configurations...")
    print()

    # Run all configs
    results: list[tuple[V3Config, float, int, float, int]] = []  # (cfg, pnl, wins, diff, changed_count)

    for cfg in configs:
        v3_total = 0.0
        v3_wins = 0
        changed = 0

        for trade in trades:
            if not trade.premium_path:
                v3_total += trade.actual_pnl
                if trade.actual_pnl > 0:
                    v3_wins += 1
                continue

            _, v3_pnl, v3_reason = simulate_v3_parameterized(trade, cfg)
            v3_total += v3_pnl
            if v3_pnl > 0:
                v3_wins += 1

            _, v2_pnl, _ = v2_results.get(trade.trade_id, (0, 0, ""))
            if abs(v3_pnl - v2_pnl) > 1:
                changed += 1

        results.append((cfg, v3_total, v3_wins, v3_total - v2_total, changed))

    # Sort by P&L diff (best first)
    results.sort(key=lambda x: -x[3])

    # Print top 20
    print(f"{'Rank':>4} {'P&L Diff':>10} {'Total P&L':>10} {'Wins':>5} {'WR%':>6} {'Changed':>7}  Config")
    print("-" * 120)
    for i, (cfg, pnl, wins, diff, changed) in enumerate(results[:30]):
        wr = wins / total_trades * 100
        thesis_label = f"thesis>={cfg.thesis_min_drop_pct:.0f}%,rate>{cfg.thesis_decline_rate:.1f}"
        if cfg.thesis_min_drop_pct >= 999:
            thesis_label = "thesis=OFF"

        emerg_label = f"emerg>={cfg.emergency_min_loss_pct:.0f}%@{cfg.emergency_time_min:.0f}m"
        if not cfg.enable_time_urgency_emergency:
            emerg_label = "emerg=OFF"

        print(f"{i+1:>4} ${diff:>+9.2f} ${pnl:>9.2f} {wins:>5} {wr:>5.1f}% {changed:>7}  {thesis_label}  {emerg_label}")

    # Print bottom 5 for contrast
    print()
    print("WORST 5:")
    print("-" * 120)
    for i, (cfg, pnl, wins, diff, changed) in enumerate(results[-5:]):
        wr = wins / total_trades * 100
        thesis_label = f"thesis>={cfg.thesis_min_drop_pct:.0f}%,rate>{cfg.thesis_decline_rate:.1f}"
        if cfg.thesis_min_drop_pct >= 999:
            thesis_label = "thesis=OFF"
        emerg_label = f"emerg>={cfg.emergency_min_loss_pct:.0f}%@{cfg.emergency_time_min:.0f}m"
        if not cfg.enable_time_urgency_emergency:
            emerg_label = "emerg=OFF"
        print(f"     ${diff:>+9.2f} ${pnl:>9.2f} {wins:>5} {wr:>5.1f}% {changed:>7}  {thesis_label}  {emerg_label}")

    # Show detailed trade-by-trade for top 3 configs
    print()
    print("=" * 120)
    print("DETAILED BREAKDOWN — TOP 3 CONFIGS")
    print("=" * 120)

    for rank, (cfg, total_pnl, total_wins, total_diff, _) in enumerate(results[:3], 1):
        thesis_label = f"thesis>={cfg.thesis_min_drop_pct:.0f}%,rate>{cfg.thesis_decline_rate:.1f}"
        if cfg.thesis_min_drop_pct >= 999:
            thesis_label = "thesis=OFF"
        emerg_label = f"emerg>={cfg.emergency_min_loss_pct:.0f}%@{cfg.emergency_time_min:.0f}m"
        if not cfg.enable_time_urgency_emergency:
            emerg_label = "emerg=OFF"

        print(f"\n--- RANK #{rank}: {thesis_label}  {emerg_label} ---")
        print(f"    Total P&L: ${total_pnl:+.2f} (diff: ${total_diff:+.2f}), "
              f"Wins: {total_wins}/{total_trades} ({total_wins/total_trades*100:.1f}%)")
        print()

        improved = []
        worsened = []

        for trade in trades:
            if not trade.premium_path:
                continue

            _, v2_pnl, v2_reason = v2_results[trade.trade_id]
            v3_exit, v3_pnl, v3_reason = simulate_v3_parameterized(trade, cfg)
            diff = v3_pnl - v2_pnl

            if diff > 1:
                improved.append((trade, v2_pnl, v2_reason, v3_pnl, v3_reason, diff))
            elif diff < -1:
                worsened.append((trade, v2_pnl, v2_reason, v3_pnl, v3_reason, diff))

        if improved:
            print(f"    IMPROVED ({len(improved)}, total ${sum(c[5] for c in improved):+.2f}):")
            for trade, v2p, v2r, v3p, v3r, d in sorted(improved, key=lambda x: -x[5]):
                print(f"      {trade.bot} #{trade.trade_id} {trade.ticker}: "
                      f"${v2p:.2f} ({v2r}) -> ${v3p:.2f} ({v3r}) = ${d:+.2f}")

        if worsened:
            print(f"    WORSENED ({len(worsened)}, total ${sum(c[5] for c in worsened):+.2f}):")
            for trade, v2p, v2r, v3p, v3r, d in sorted(worsened, key=lambda x: x[5]):
                print(f"      {trade.bot} #{trade.trade_id} {trade.ticker}: "
                      f"${v2p:.2f} ({v2r}) -> ${v3p:.2f} ({v3r}) = ${d:+.2f}")


def main():
    # Download DBs (reuse if already cached)
    for bot in ("kody", "adam", "vinny", "yank"):
        db_dst = f"/tmp/db_{bot}.sqlite"
        if not os.path.exists(db_dst):
            print(f"Downloading {bot} DB from droplet...")
            db_src = f"root@129.212.138.145:/root/options-owl/journal/owlet-{bot}/raw_messages.db"
            os.system(f'scp -i "$HOME/.ssh/id_ed25519_do" {db_src} {db_dst} 2>/dev/null')

    trades = load_trades_from_db()
    print(f"Loaded {len(trades)} parent LIVE trades")

    # Load premium paths
    all_premiums = {}
    for bot in ("kody", "adam", "vinny", "yank"):
        all_premiums[bot] = load_all_premiums(bot)

    # Match
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
    print()

    run_sweep(trades)


if __name__ == "__main__":
    main()
