"""Targeted backtest: test specific exit gate configurations.

Fixes the grid search methodology:
1. Tests call and put SEPARATELY (not both simultaneously)
2. Includes phase trail percentages as tunable parameters
3. Tests disabling problematic gates
4. Reports MFE gap (money left on the table)

Usage:
    python scripts/backtest_targeted.py
"""

import os
import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest_vinny import (
    DB_PATH,
    STARTING_BALANCE,
    VinnyParams,
    load_option_bars,
    load_trading_days,
    find_entry_bar,
    ts_to_et,
    et_time_str,
    TradeResult,
)
from options_owl.risk.vinny_strategy import (
    PHASE_TRAILS,
    is_time_decay_zone,
)


def simulate_trade_custom(option_bars, params, entry_idx, direction, strike, balance,
                          phase_trails=None, profit_lock_tiers=None):
    """Simulate trade with custom phase trail percentages and profit lock."""
    if phase_trails is None:
        phase_trails = dict(PHASE_TRAILS)

    if entry_idx >= len(option_bars) - 10:
        return None

    entry_bar = option_bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry <= 0 or raw_entry < params.min_premium:
        return None

    entry_premium = raw_entry * (1 + params.entry_slippage_bps / 10000)

    # Fixed 3 contracts for simplicity
    contracts = 3
    cost_per_contract = entry_premium * 100
    total_cost = contracts * cost_per_contract
    if total_cost > balance:
        contracts = max(1, int(balance / cost_per_contract))
        total_cost = contracts * cost_per_contract
    if total_cost > balance:
        return None

    targets = [
        entry_premium * (1 + params.t1_pct / 100),
        entry_premium * (1 + params.t2_pct / 100),
        entry_premium * (1 + params.t3_pct / 100),
        entry_premium * (1 + params.t4_pct / 100),
        entry_premium * (1 + params.t5_pct / 100),
    ]

    peak_premium = entry_premium
    last_target_hit = 0
    remaining_pct = 100.0
    weighted_pnl = 0.0
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    last_new_high_bar = entry_idx

    entry_ts = entry_bar["timestamp"]
    entry_et = ts_to_et(entry_ts)
    opened_at_str = entry_et.strftime("%Y-%m-%dT%H:%M:%S")

    # Profit lock state
    locked_floor_pct = None
    if profit_lock_tiers:
        # Parse "80:30,150:70,250:150"
        lock_tiers = []
        for tier in profit_lock_tiers.split(","):
            parts = tier.strip().split(":")
            lock_tiers.append((float(parts[0]), float(parts[1])))
        lock_tiers.sort(key=lambda x: x[0])
    else:
        lock_tiers = []

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

        # EOD hard exit
        if (bar_et.hour > params.eod_hour or
            (bar_et.hour == params.eod_hour and bar_et.minute >= params.eod_minute)):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "eod_close"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Scale-out at targets
        for t_idx, t_price in enumerate(targets):
            t_num = t_idx + 1
            if t_num > last_target_hit and current >= t_price and remaining_pct > 0:
                sell_pct = remaining_pct * (params.scale_out_pct / 100)
                weighted_pnl += sell_pct * gain_pct
                remaining_pct -= sell_pct
                last_target_hit = t_num

        # Grace period
        if minutes_elapsed < params.grace_period_min:
            continue

        # Premium stop
        drop_from_entry = (entry_premium - current) / entry_premium * 100 if entry_premium > 0 else 0
        if drop_from_entry >= params.premium_stop_pct:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "premium_stop"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Profit lock ratchet
        if lock_tiers and gain_pct > 0:
            for threshold, floor in lock_tiers:
                peak_gain = (peak_premium - entry_premium) / entry_premium * 100
                if peak_gain >= threshold:
                    if locked_floor_pct is None or floor > locked_floor_pct:
                        locked_floor_pct = floor

            if locked_floor_pct is not None and gain_pct < locked_floor_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = f"profit_lock_{locked_floor_pct:.0f}"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Setup failed
        if (minutes_elapsed >= params.setup_failed_min
                and last_target_hit == 0
                and gain_pct < params.setup_failed_gain_pct):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "setup_failed"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Phase-based trailing stop with custom trail %
        if params.enable_phase_trail and peak_premium > entry_premium:
            in_decay = is_time_decay_zone(
                opened_at_str, bar_et,
                max_hold_minutes=params.time_decay_hold_min,
                afternoon_hour=params.time_decay_afternoon_hour,
            )

            phase = min(last_target_hit, max(phase_trails.keys()))
            trail_pct = phase_trails.get(phase, phase_trails[0])

            if in_decay:
                trail_pct = min(trail_pct, 10.0)

            drop_from_peak = (peak_premium - current) / peak_premium * 100
            if drop_from_peak >= trail_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = f"phase_trail_p{phase}"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Theta bleed
        if minutes_elapsed >= params.theta_bleed_hold_min:
            loss_pct = (entry_premium - current) / entry_premium * 100
            if loss_pct >= params.theta_bleed_max_loss_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "theta_bleed"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Time decay stale
        in_decay = is_time_decay_zone(
            opened_at_str, bar_et,
            max_hold_minutes=params.time_decay_hold_min,
            afternoon_hour=params.time_decay_afternoon_hour,
        )
        if in_decay and current < peak_premium * 0.99:
            bars_since_high = bar_idx - last_new_high_bar
            if bars_since_high >= params.time_decay_stale_min:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "time_decay_stale"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # No momentum / time stop
        if (minutes_elapsed >= params.time_stop_min
                and last_target_hit == 0
                and gain_pct < params.time_stop_gain_pct):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "time_stop"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

    if exit_reason is None:
        last_bar = option_bars[-1]
        exit_premium = last_bar["close"]
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar_idx = len(option_bars) - 1

    slippage_pct = params.exit_slippage_bps / 100
    total_pnl_pct = weighted_pnl / 100 - slippage_pct
    pnl_dollars = total_cost * (total_pnl_pct / 100)

    return TradeResult(
        date="",
        ticker="",
        direction=direction,
        strike=strike,
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
        phase_at_exit=last_target_hit,
    )


def run_scenario(name, tickers, params, phase_trails=None, profit_lock_tiers=None,
                 direction="call"):
    """Run a full scenario and return summary stats."""
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    mfe_gaps = []

    for ticker in tickers:
        trading_days = load_trading_days(ticker)
        for day in trading_days:
            date_str = day["date"]
            strike = day["atm_strike"]

            if direction == "call":
                contract_ticker = day["atm_call_ticker"]
            else:
                contract_ticker = day["atm_put_ticker"]

            bars = load_option_bars(contract_ticker)
            if not bars or len(bars) < 30:
                continue

            entry_idx = find_entry_bar(bars, params.entry_hour, params.entry_minute)
            if entry_idx is None or entry_idx >= len(bars) - 10:
                continue

            result = simulate_trade_custom(
                bars, params, entry_idx, direction, strike, balance,
                phase_trails=phase_trails, profit_lock_tiers=profit_lock_tiers,
            )
            if result is None:
                continue

            result.date = date_str
            result.ticker = ticker
            balance += result.pnl_dollars
            result.balance_after = balance
            trades.append(result)

            # MFE gap
            peak_pnl = (result.peak_premium - result.entry_premium) / result.entry_premium * 100
            mfe_gaps.append(peak_pnl - result.pnl_pct)

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

    if not trades:
        return None

    wins = [t for t in trades if t.pnl_dollars >= 0]
    losses = [t for t in trades if t.pnl_dollars < 0]
    total_pnl = balance - STARTING_BALANCE
    wr = len(wins) / len(trades) * 100

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    avg_dur = sum(t.duration_min for t in trades) / len(trades)
    avg_mfe = sum(mfe_gaps) / len(mfe_gaps) if mfe_gaps else 0

    return {
        "name": name,
        "total_pnl": total_pnl,
        "pnl_pct": total_pnl / STARTING_BALANCE * 100,
        "win_rate": wr,
        "profit_factor": pf,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_duration": avg_dur,
        "avg_mfe_gap": avg_mfe,
        "reasons": reasons,
    }


def main():
    # Use top tickers by data volume
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT ticker, COUNT(*) as days
        FROM trading_days WHERE call_bars > 0
        GROUP BY ticker ORDER BY days DESC LIMIT 5
    """).fetchall()
    conn.close()
    tickers = [r[0] for r in rows]
    print(f"Testing on: {', '.join(tickers)}")
    print(f"Starting balance: ${STARTING_BALANCE:,.2f}")
    print()

    scenarios = []

    # ============================================================
    # SCENARIO 1: Current settings (baseline) — call only
    # ============================================================
    base = VinnyParams()
    scenarios.append(("1. BASELINE (current)", base, dict(PHASE_TRAILS), None))

    # ============================================================
    # SCENARIO 2: Wider phase trails (less aggressive trailing)
    # ============================================================
    wide_trails = {0: 35.0, 1: 30.0, 2: 25.0, 3: 20.0, 4: 18.0, 5: 15.0, 6: 12.0}
    scenarios.append(("2. WIDE TRAILS (35/30/25/20/18/15/12)", base, wide_trails, None))

    # ============================================================
    # SCENARIO 3: Very wide phase trails
    # ============================================================
    vwide_trails = {0: 45.0, 1: 35.0, 2: 30.0, 3: 25.0, 4: 20.0, 5: 18.0, 6: 15.0}
    scenarios.append(("3. VERY WIDE TRAILS (45/35/30/25/20/18/15)", base, vwide_trails, None))

    # ============================================================
    # SCENARIO 4: Wide trails + relaxed setup_failed
    # ============================================================
    relaxed = VinnyParams(setup_failed_min=20, setup_failed_gain_pct=5)
    scenarios.append(("4. WIDE TRAILS + RELAXED SETUP (20m/5%)", relaxed, wide_trails, None))

    # ============================================================
    # SCENARIO 5: Wide trails + NO setup_failed (disable it)
    # ============================================================
    no_setup = VinnyParams(setup_failed_min=999, setup_failed_gain_pct=0)
    scenarios.append(("5. WIDE TRAILS + NO SETUP_FAILED", no_setup, wide_trails, None))

    # ============================================================
    # SCENARIO 6: Very wide trails + no setup_failed + profit lock
    # ============================================================
    scenarios.append(("6. VWIDE + NO SETUP + PROFIT_LOCK", no_setup, vwide_trails, "80:30,150:70,250:150"))

    # ============================================================
    # SCENARIO 7: Wide trails + wider premium stop (70%)
    # ============================================================
    wide_stop = VinnyParams(premium_stop_pct=70, setup_failed_min=20, setup_failed_gain_pct=5)
    scenarios.append(("7. WIDE TRAILS + 70% STOP + RELAXED SETUP", wide_stop, wide_trails, None))

    # ============================================================
    # SCENARIO 8: Everything relaxed — max hold time
    # ============================================================
    max_hold = VinnyParams(
        setup_failed_min=30, setup_failed_gain_pct=3,
        premium_stop_pct=70,
        time_decay_hold_min=90,
        time_decay_stale_min=15,
        theta_bleed_hold_min=90,
        theta_bleed_max_loss_pct=50,
        time_stop_min=60,
        time_stop_gain_pct=3,
    )
    scenarios.append(("8. ALL RELAXED — MAX HOLD", max_hold, wide_trails, "80:30,150:70,250:150"))

    # ============================================================
    # SCENARIO 9: Tight initial, widen after T1
    # ============================================================
    progressive = {0: 30.0, 1: 35.0, 2: 30.0, 3: 25.0, 4: 20.0, 5: 15.0, 6: 12.0}
    prog_params = VinnyParams(setup_failed_min=15, setup_failed_gain_pct=8)
    scenarios.append(("9. PROGRESSIVE (30→35→30→25 after T1)", prog_params, progressive, None))

    # ============================================================
    # SCENARIO 10: No trailing stop at all — only targets + hard stop
    # ============================================================
    no_trail = VinnyParams(enable_phase_trail=False, premium_stop_pct=60,
                           setup_failed_min=999)
    scenarios.append(("10. NO TRAIL — targets + 60% stop only", no_trail, None, None))

    # ============================================================
    # SCENARIO 11: Wider trails + tighter profit lock (lock gains earlier)
    # ============================================================
    scenarios.append(("11. WIDE TRAILS + TIGHT LOCK (50:15,100:50,200:120)",
                      no_setup, wide_trails, "50:15,100:50,200:120"))

    # ============================================================
    # SCENARIO 12: Wide phase 0, aggressive tighten after targets
    # ============================================================
    wide_then_tight = {0: 40.0, 1: 25.0, 2: 20.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}
    wt_params = VinnyParams(setup_failed_min=25, setup_failed_gain_pct=5)
    scenarios.append(("12. WIDE P0 (40%) → TIGHT AFTER T1", wt_params, wide_then_tight, "80:30,150:70"))

    # Run all scenarios
    print("=" * 120)
    print(f"  {'Scenario':<52} {'PnL$':>10} {'PnL%':>7} {'WR':>5} {'PF':>5} {'Trades':>6} "
          f"{'AvgW':>6} {'AvgL':>6} {'MFEgap':>7} {'MaxDD':>6} {'AvgDur':>6}")
    print("=" * 120)

    all_results = []
    for name, params, trails, lock in scenarios:
        r = run_scenario(name, tickers, params, phase_trails=trails,
                        profit_lock_tiers=lock, direction="call")
        if r is None:
            print(f"  {name:<52} NO TRADES")
            continue
        all_results.append(r)
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {name:<52} ${r['total_pnl']:>+9.2f} {r['pnl_pct']:>+6.1f}% "
              f"{r['win_rate']:>4.0f}% {pf_str:>5} {r['num_trades']:>6} "
              f"{r['avg_win']:>+5.0f}% {r['avg_loss']:>+5.0f}% "
              f"{r['avg_mfe_gap']:>6.1f}% {r['max_drawdown']:>5.1f}% "
              f"{r['avg_duration']:>5.0f}m")

    # Now show exit reason breakdown for top 3
    print()
    print("=" * 120)
    print("  EXIT REASON BREAKDOWN (top scenarios)")
    print("=" * 120)

    sorted_results = sorted(all_results, key=lambda x: x['total_pnl'], reverse=True)
    for r in sorted_results[:5]:
        print(f"\n  {r['name']}: ${r['total_pnl']:+,.2f}")
        for reason, count in sorted(r['reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason:30s} {count:4d} trades")

    # Also run puts for the best scenario
    print()
    print("=" * 120)
    print("  BEST SCENARIO — PUT DIRECTION")
    print("=" * 120)
    best = sorted_results[0]
    # Find matching scenario
    for name, params, trails, lock in scenarios:
        if name == best["name"]:
            r_put = run_scenario(name + " [PUT]", tickers, params, phase_trails=trails,
                                profit_lock_tiers=lock, direction="put")
            if r_put:
                pf_str = f"{r_put['profit_factor']:.2f}" if r_put['profit_factor'] < 100 else "inf"
                print(f"  PUT:  PnL=${r_put['total_pnl']:>+9.2f} WR={r_put['win_rate']:.0f}% "
                      f"PF={pf_str} Trades={r_put['num_trades']} "
                      f"MFEgap={r_put['avg_mfe_gap']:.1f}%")
                print(f"  CALL: PnL=${best['total_pnl']:>+9.2f} WR={best['win_rate']:.0f}% "
                      f"PF={best['profit_factor']:.2f} Trades={best['num_trades']} "
                      f"MFEgap={best['avg_mfe_gap']:.1f}%")
            break

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
