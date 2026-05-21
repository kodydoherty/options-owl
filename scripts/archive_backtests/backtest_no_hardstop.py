#!/usr/bin/env python3
"""Backtest: what happens if we remove the 30% hard stop and grace period?

Compares 4 configurations against real trade premium data:
  A) V2.1 current   — 20min grace + 30% stop + 45% catastrophic + profit_retrace + adaptive trail
  B) V3 with stop   — profit floor + bounce-fade + 30% stop + grace (what we just built)
  C) V3 no stop     — profit floor + bounce-fade, NO hard stop, NO grace period
  D) V3 no stop+    — same as C but with lower bounce-fade watch threshold (40% instead of 50%)

Usage:
  python scripts/backtest_no_hardstop.py
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
    simulate_v2_exit, BounceState,
)


# ---------------------------------------------------------------------------
# V3 sim with configurable hard stop / grace
# ---------------------------------------------------------------------------

@dataclass
class ExitConfig:
    label: str
    # Hard stop
    enable_hard_stop: bool = True
    hard_stop_pct: float = 30.0
    grace_minutes: float = 20.0
    enable_catastrophic: bool = True
    catastrophic_pct: float = 45.0
    # Profit floor
    enable_profit_floor: bool = True
    floor_activation_pct: float = 15.0
    floor_ratchet_pct: float = 60.0
    # Bounce-fade
    enable_bounce_fade: bool = True
    bounce_watch_pct: float = 50.0
    bounce_min_recovery_pct: float = 10.0
    bounce_fade_pct: float = 15.0
    # Profit retrace (legacy)
    enable_profit_retrace: bool = False
    retrace_pct: float = 50.0
    retrace_min_gain: float = 25.0
    # Adaptive trail
    enable_adaptive_trail: bool = True
    adaptive_activation: float = 35.0
    adaptive_width: float = 35.0


def simulate_configurable(trade: TradeInfo, cfg: ExitConfig) -> tuple[float, float, str]:
    """Simulate exit with fully configurable gates."""
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
    grace_end = trade.entry_time + timedelta(minutes=cfg.grace_minutes) if cfg.enable_hard_stop else None

    for tick in trade.premium_path:
        p = tick.premium
        peak = max(peak, p)
        gain_pct = (p - entry) / entry * 100
        peak_gain_pct = (peak - entry) / entry * 100
        drop_from_entry_pct = (entry - p) / entry * 100 if p < entry else 0
        time_remaining_min = max(0, (expiry_dt - tick.timestamp).total_seconds() / 60)

        in_grace = grace_end is not None and tick.timestamp < grace_end

        # === Catastrophic stop (bypasses grace) ===
        if cfg.enable_catastrophic and drop_from_entry_pct >= cfg.catastrophic_pct:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "catastrophic_stop"

        # === Hard stop (after grace) ===
        if cfg.enable_hard_stop and not in_grace and drop_from_entry_pct >= cfg.hard_stop_pct:
            pnl = (p - entry) * contracts * 100
            return p, pnl, "stop_loss"

        # === Profit floor (v3) ===
        if cfg.enable_profit_floor and peak_gain_pct >= cfg.floor_activation_pct:
            # Time urgency ratchet
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

        # === Bounce-fade (v3) ===
        if cfg.enable_bounce_fade:
            if drop_from_entry_pct >= cfg.bounce_watch_pct and not bounce_state.in_bounce_watch:
                bounce_state.in_bounce_watch = True
                bounce_state.bounce_low = p
                bounce_state.bounce_detected = False

            if bounce_state.in_bounce_watch:
                bounce_state.bounce_low = min(bounce_state.bounce_low, p)

                # Time-adjusted thresholds
                if time_remaining_min < 15:
                    min_recovery = 3.0
                    fade_threshold = 5.0
                elif time_remaining_min < 30:
                    min_recovery = 5.0
                    fade_threshold = 8.0
                elif time_remaining_min < 60:
                    min_recovery = 7.0
                    fade_threshold = 10.0
                else:
                    min_recovery = cfg.bounce_min_recovery_pct
                    fade_threshold = cfg.bounce_fade_pct

                if not bounce_state.bounce_detected and bounce_state.bounce_low > 0:
                    recovery_pct = (p - bounce_state.bounce_low) / bounce_state.bounce_low * 100
                    if recovery_pct >= min_recovery:
                        bounce_state.bounce_detected = True
                        bounce_state.bounce_high = p

                if bounce_state.bounce_detected:
                    bounce_state.bounce_high = max(bounce_state.bounce_high, p)
                    if bounce_state.bounce_high > 0:
                        fade = (bounce_state.bounce_high - p) / bounce_state.bounce_high * 100
                        if fade >= fade_threshold:
                            pnl = (p - entry) * contracts * 100
                            return p, pnl, "bounce_fade"

                # Time-critical: sell on any bounce with < 30min left
                if time_remaining_min < 30 and p > bounce_state.bounce_low * 1.03:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "bounce_fade_critical"

        # === Profit retrace (legacy fallback) ===
        if cfg.enable_profit_retrace and peak_gain_pct >= cfg.retrace_min_gain:
            profit_at_peak = peak - entry
            profit_now = p - entry
            if profit_at_peak > 0:
                given_back = (profit_at_peak - profit_now) / profit_at_peak * 100
                if given_back >= cfg.retrace_pct:
                    pnl = (p - entry) * contracts * 100
                    return p, pnl, "profit_retrace"

        # === Adaptive trail ===
        if cfg.enable_adaptive_trail and peak_gain_pct >= cfg.adaptive_activation:
            trail_drop = (peak - p) / peak * 100
            if trail_drop >= cfg.adaptive_width:
                pnl = (p - entry) * contracts * 100
                return p, pnl, "adaptive_trail"

    # EOD
    last = trade.premium_path[-1]
    pnl = (last.premium - entry) * contracts * 100
    return last.premium, pnl, "eod_cutoff"


def run_comparison(trades: list[TradeInfo]):
    configs = [
        ExitConfig(
            label="A) V2.1 current",
            enable_hard_stop=True, hard_stop_pct=30.0, grace_minutes=20.0,
            enable_catastrophic=True, catastrophic_pct=45.0,
            enable_profit_floor=False, enable_bounce_fade=False,
            enable_profit_retrace=True, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="B) V3 + stop + grace",
            enable_hard_stop=True, hard_stop_pct=30.0, grace_minutes=20.0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=True,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="C) V3 no stop, no grace",
            enable_hard_stop=False, grace_minutes=0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=True, bounce_watch_pct=50.0,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="D) V3 no stop, bounce@40%",
            enable_hard_stop=False, grace_minutes=0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=True, bounce_watch_pct=40.0,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="E) V3 no stop, bounce@35%",
            enable_hard_stop=False, grace_minutes=0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=True, bounce_watch_pct=35.0,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="F) V3 no stop, bounce@30%",
            enable_hard_stop=False, grace_minutes=0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=True, bounce_watch_pct=30.0,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
        ExitConfig(
            label="G) V3 no stop, no bounce",
            enable_hard_stop=False, grace_minutes=0,
            enable_catastrophic=False,
            enable_profit_floor=True, enable_bounce_fade=False,
            enable_profit_retrace=False, enable_adaptive_trail=True,
        ),
    ]

    total_trades = len(trades)

    # Run all configs
    all_results: list[tuple[ExitConfig, float, int, list[tuple]]] = []

    for cfg in configs:
        pnl_total = 0.0
        wins = 0
        trade_results = []

        for trade in trades:
            if not trade.premium_path:
                pnl_total += trade.actual_pnl
                if trade.actual_pnl > 0:
                    wins += 1
                trade_results.append((trade, trade.actual_pnl, trade.actual_exit_reason))
                continue

            ep, pnl, reason = simulate_configurable(trade, cfg)
            pnl_total += pnl
            if pnl > 0:
                wins += 1
            trade_results.append((trade, pnl, reason))

        all_results.append((cfg, pnl_total, wins, trade_results))

    # Print comparison table
    baseline_pnl = all_results[0][1]
    baseline_wins = all_results[0][2]

    print(f"\n{'Config':<30} {'Total P&L':>10} {'Diff':>10} {'Wins':>5} {'WR%':>6} {'WR Diff':>8}")
    print("-" * 80)
    for cfg, pnl, wins, _ in all_results:
        wr = wins / total_trades * 100
        base_wr = baseline_wins / total_trades * 100
        print(f"{cfg.label:<30} ${pnl:>+9.2f} ${pnl - baseline_pnl:>+9.2f} "
              f"{wins:>5} {wr:>5.1f}% {wr - base_wr:>+7.1f}%")

    # Detailed per-trade comparison: A vs C (current vs no-stop)
    print("\n" + "=" * 120)
    print("DETAILED: V2.1 current (A) vs V3 no stop/grace (C)")
    print("=" * 120)

    a_results = all_results[0][3]  # V2.1
    c_results = all_results[2][3]  # V3 no stop

    improved = []
    worsened = []

    print(f"\n{'Bot':<6} {'#':>3} {'Ticker':<6} {'Qty':>3} {'Entry':>6} "
          f"{'A Exit':>7} {'A P&L':>9} {'A Reason':<20} "
          f"{'C Exit':>7} {'C P&L':>9} {'C Reason':<20} {'Diff':>8}")
    print("-" * 120)

    for (trade_a, pnl_a, reason_a), (trade_c, pnl_c, reason_c) in zip(a_results, c_results):
        trade = trade_a
        diff = pnl_c - pnl_a

        if abs(diff) > 1:
            # Find exit premiums
            ep_a = pnl_a / (trade.contracts * 100) + trade.entry_premium if trade.contracts else 0
            ep_c = pnl_c / (trade.contracts * 100) + trade.entry_premium if trade.contracts else 0

            marker = "**" if diff > 0 else "!!"
            print(f"{trade.bot:<6} {trade.trade_id:>3} {trade.ticker:<6} {trade.contracts:>3} "
                  f"${trade.entry_premium:.2f} "
                  f"${ep_a:>6.2f} ${pnl_a:>8.2f} {reason_a:<20} "
                  f"${ep_c:>6.2f} ${pnl_c:>8.2f} {reason_c:<20} "
                  f"${diff:>+7.2f} {marker}")

            if diff > 0:
                improved.append((trade, pnl_a, reason_a, pnl_c, reason_c, diff))
            else:
                worsened.append((trade, pnl_a, reason_a, pnl_c, reason_c, diff))

    print()
    if improved:
        print(f"IMPROVED: {len(improved)} trades, total ${sum(d[5] for d in improved):+.2f}")
    if worsened:
        print(f"WORSENED: {len(worsened)} trades, total ${sum(d[5] for d in worsened):+.2f}")

    # Show exit reason distribution for each config
    print("\n" + "=" * 80)
    print("EXIT REASON DISTRIBUTION")
    print("=" * 80)

    for cfg, _, _, results in all_results:
        reasons = defaultdict(int)
        for _, _, reason in results:
            reasons[reason] += 1
        sorted_reasons = sorted(reasons.items(), key=lambda x: -x[1])
        print(f"\n{cfg.label}:")
        for reason, count in sorted_reasons:
            print(f"  {reason:<25} {count:>3} ({count/total_trades*100:.0f}%)")


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

    run_comparison(trades)


if __name__ == "__main__":
    main()
