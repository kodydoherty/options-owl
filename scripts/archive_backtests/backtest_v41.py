#!/usr/bin/env python3
"""Backtest v4.1 vs v4 production — daily P&L breakdown across all signals.

v4.1 changes from v4:
  - DISABLE decel_exit (was 37% of exits, cutting runners)
  - DISABLE no_momentum (never fired)
  - DISABLE time_tighten (never fired)
  - DISABLE time_decay_zone (fired 1x, lost $398)
  - KEEP: smart grace, BE clamp, soft trail, dollar trail, profit floor,
          adaptive trail, underlying trail, thesis cut, stop loss, EOD

Usage:
  python scripts/backtest_v41.py [signals_db] [harvester_db]
"""

import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SLIPPAGE_HAIRCUT = 0.15


@dataclass
class Tick:
    ts: datetime
    mid: float
    bid: float
    ask: float
    underlying: float = 0.0


@dataclass
class ExitResult:
    premium: float
    reason: str
    hold_minutes: float = 0.0
    peak_gain_pct: float = 0.0
    exit_gain_pct: float = 0.0


def _apply_slippage(pnl: float) -> float:
    return pnl * (1 - SLIPPAGE_HAIRCUT) if pnl > 0 else pnl


# ---------------------------------------------------------------------------
# v4 production settings
# ---------------------------------------------------------------------------
V4_SETTINGS = {
    "grace_minutes": 20, "smart_grace": True, "smart_grace_confirm_pct": 0.1,
    "premium_stop_pct": 30.0,
    "be_clamp_activation_pct": 15.0,
    "soft_trail_min_pct": 15.0, "soft_trail_max_pct": 35.0, "soft_trail_floor_pct": 50.0,
    "adaptive_activation_pct": 35.0, "adaptive_active_width": 35.0,
    "adaptive_runner_threshold": 150.0, "adaptive_runner_width": 45.0,
    "adaptive_moonshot_threshold": 400.0, "adaptive_moonshot_width": 30.0,
    "dollar_trail_activation_pct": 40.0, "dollar_trail_small_step_pct": 20.0,
    "dollar_trail_step_threshold_pct": 25.0, "dollar_trail_large_step_pct": 10.0,
    "profit_floor_activation_pct": 15.0, "profit_floor_ratchet_pct": 60.0,
    "theta_bleed_hold_min": 45.0, "theta_bleed_max_loss_pct": 30.0,
    "no_momentum_minutes": 45.0, "no_momentum_min_gain_pct": 5.0,
    "time_decay_hold_min": 45.0, "time_decay_afternoon_hour": 15,
    "time_decay_afternoon_min": 30, "time_decay_stale_min": 10.0,
    "time_tighten_after_min": 60.0, "time_tighten_factor": 0.7,
    "decel_min_hold_sec": 480, "decel_min_gain_pct": 5.0,
    "decel_short_window": 5, "decel_long_window": 15, "decel_threshold": -3.0,
    "thesis_cut_threshold_pct": 40.0, "thesis_cut_lookback": 8,
    "thesis_cut_new_low_exit": 3, "thesis_cut_bounce_hold_pct": 5.0,
    "thesis_cut_min_ticks": 4,
    "underlying_trail_tiers": [(100, 0.50), (50, 0.40), (15, 0.30), (0, 0.20)],
}

V4_GATES = {
    "stop_loss", "be_clamp", "soft_trail", "adaptive_trail",
    "underlying_trail", "time_tighten", "profit_floor",
    "thesis_cut", "decel_exit", "theta_bleed", "time_decay",
    "no_momentum", "dollar_trail", "eod",
}

# v4.1: remove the 4 legacy hold-time gates
V41_GATES = {
    "stop_loss", "be_clamp", "soft_trail", "adaptive_trail",
    "underlying_trail", "profit_floor", "thesis_cut",
    "dollar_trail", "eod",
}


def simulate(entry, ticks, signal_ts, contracts, direction, enabled_gates, settings):
    if not ticks or entry <= 0:
        return ExitResult(entry, "no_data")

    s = settings
    grace_minutes = s["grace_minutes"]
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)

    peak_premium = entry
    peak_underlying = ticks[0].underlying if ticks[0].underlying > 0 else 0
    entry_underlying = peak_underlying
    last_new_high_ts = signal_ts
    be_clamp_active = False
    thesis_ticks_in_zone = 0
    premium_history = []

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue

        if price > peak_premium:
            peak_premium = price
            last_new_high_ts = tick.ts
        if tick.underlying > 0:
            if direction.lower() in ("call", "bullish", "long"):
                if tick.underlying > peak_underlying:
                    peak_underlying = tick.underlying
            else:
                if peak_underlying == 0 or tick.underlying < peak_underlying:
                    peak_underlying = tick.underlying

        gain_pct = (price - entry) / entry * 100
        peak_gain_pct = (peak_premium - entry) / entry * 100
        drop_from_entry = max(0, (entry - price) / entry * 100)
        drop_from_peak = (peak_premium - price) / peak_premium * 100 if peak_premium > 0 else 0
        elapsed_min = (tick.ts - signal_ts).total_seconds() / 60
        premium_history.append((tick.ts, price))
        et_hour = (tick.ts.hour - 4) % 24
        et_min = tick.ts.minute

        # Grace period
        in_grace = False
        if elapsed_min < grace_minutes:
            if s.get("smart_grace") and entry_underlying > 0 and tick.underlying > 0:
                move_pct = (tick.underlying - entry_underlying) / entry_underlying * 100
                is_call = direction.lower() in ("call", "bullish", "long")
                confirmed = (is_call and move_pct > s["smart_grace_confirm_pct"]) or \
                            (not is_call and move_pct < -s["smart_grace_confirm_pct"])
                in_grace = not confirmed
            else:
                in_grace = True

        def _exit(reason):
            return ExitResult(price, reason, elapsed_min, peak_gain_pct, gain_pct)

        # === EXIT GATES IN PRODUCTION PRIORITY ORDER ===

        if "stop_loss" in enabled_gates and not in_grace:
            if drop_from_entry >= s["premium_stop_pct"]:
                return _exit("stop_loss")

        if "be_clamp" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["be_clamp_activation_pct"]:
                be_clamp_active = True
            if be_clamp_active and price <= entry:
                return _exit("be_clamp")

        if "soft_trail" in enabled_gates and not in_grace:
            if s["soft_trail_min_pct"] <= peak_gain_pct < s["soft_trail_max_pct"]:
                gain_at_peak = peak_premium - entry
                floor = entry + gain_at_peak * (s["soft_trail_floor_pct"] / 100)
                if price <= floor:
                    return _exit("soft_trail")

        if "dollar_trail" in enabled_gates and not in_grace:
            profit_pct = (peak_premium - entry) / entry * 100
            if profit_pct >= s["dollar_trail_activation_pct"]:
                profit_dollars = (peak_premium - entry) * 100
                threshold = s["dollar_trail_step_threshold_pct"] / 100 * entry * 100
                step = (s["dollar_trail_large_step_pct"] if profit_dollars >= threshold
                        else s["dollar_trail_small_step_pct"]) / 100 * entry * 100
                if step > 0:
                    steps_hit = int(profit_dollars / step)
                    trail_floor_dollars = steps_hit * step
                    trail_floor_premium = entry + trail_floor_dollars / 100
                    if price <= trail_floor_premium and trail_floor_dollars > 0:
                        return _exit("dollar_trail")

        if "decel_exit" in enabled_gates and not in_grace:
            if elapsed_min * 60 >= s["decel_min_hold_sec"] and peak_gain_pct >= s["decel_min_gain_pct"]:
                if len(premium_history) >= s["decel_long_window"]:
                    def _vel(hist, w):
                        if len(hist) < w or w < 2:
                            return 0.0
                        sp = hist[-w][1]
                        return (hist[-1][1] - sp) / sp * 100 if sp > 0 else 0.0
                    v_short = _vel(premium_history, s["decel_short_window"])
                    v_long = _vel(premium_history, s["decel_long_window"])
                    if v_short - v_long < s["decel_threshold"]:
                        return _exit("decel_exit")

        if "underlying_trail" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["adaptive_activation_pct"] and peak_underlying > 0 and tick.underlying > 0:
                trail_pct = 0.20
                for tier_gain, tier_trail in s["underlying_trail_tiers"]:
                    if peak_gain_pct >= tier_gain:
                        trail_pct = tier_trail
                        break
                is_call = direction.lower() in ("call", "bullish", "long")
                if is_call:
                    if tick.underlying <= peak_underlying * (1 - trail_pct):
                        return _exit("underlying_trail")
                else:
                    if tick.underlying >= peak_underlying * (1 + trail_pct):
                        return _exit("underlying_trail")

        if "adaptive_trail" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["adaptive_moonshot_threshold"]:
                if drop_from_peak >= s["adaptive_moonshot_width"]:
                    return _exit("adaptive_trail")
            elif peak_gain_pct >= s["adaptive_runner_threshold"]:
                if drop_from_peak >= s["adaptive_runner_width"]:
                    return _exit("adaptive_trail")
            elif peak_gain_pct >= s["adaptive_activation_pct"]:
                if drop_from_peak >= s["adaptive_active_width"]:
                    return _exit("adaptive_trail")

        if "time_tighten" in enabled_gates and not in_grace:
            if elapsed_min >= s["time_tighten_after_min"]:
                tightened = 25.0 * s["time_tighten_factor"]
                if drop_from_peak >= tightened:
                    return _exit("time_tighten")

        if "profit_floor" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["profit_floor_activation_pct"]:
                peak_gain_dollars = peak_premium - entry
                floor = entry + peak_gain_dollars * (s["profit_floor_ratchet_pct"] / 100)
                if price <= floor:
                    return _exit("profit_floor")

        if "thesis_cut" in enabled_gates and not in_grace:
            if drop_from_entry >= s["thesis_cut_threshold_pct"]:
                thesis_ticks_in_zone += 1
                if thesis_ticks_in_zone >= s["thesis_cut_min_ticks"]:
                    window = [p for _, p in premium_history[-s["thesis_cut_lookback"]:]]
                    if len(window) >= s["thesis_cut_lookback"]:
                        new_lows = 0
                        running_low = window[0]
                        for wp in window[1:]:
                            if wp < running_low:
                                new_lows += 1
                                running_low = wp
                        recent_low = min(window)
                        bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
                        if bounce < s["thesis_cut_bounce_hold_pct"] and new_lows >= s["thesis_cut_new_low_exit"]:
                            return _exit("thesis_cut")
            else:
                thesis_ticks_in_zone = 0

        if "theta_bleed" in enabled_gates:
            if elapsed_min >= s["theta_bleed_hold_min"] and drop_from_entry >= s["theta_bleed_max_loss_pct"]:
                return _exit("theta_bleed")

        if "time_decay" in enabled_gates:
            afternoon = (et_hour > s["time_decay_afternoon_hour"] or
                         (et_hour == s["time_decay_afternoon_hour"] and et_min >= s["time_decay_afternoon_min"]))
            if afternoon or elapsed_min >= s["time_decay_hold_min"]:
                if last_new_high_ts:
                    stale_min = (tick.ts - last_new_high_ts).total_seconds() / 60
                    if stale_min >= s["time_decay_stale_min"]:
                        return _exit("time_decay")

        if "no_momentum" in enabled_gates:
            if elapsed_min >= s["no_momentum_minutes"] and gain_pct < s["no_momentum_min_gain_pct"]:
                return _exit("no_momentum")

        if "eod" in enabled_gates:
            if et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")

    last = ticks[-1]
    final_gain = (last.mid - entry) / entry * 100 if entry > 0 else 0
    elapsed = (ticks[-1].ts - signal_ts).total_seconds() / 60
    return ExitResult(last.mid, "eod_data_end", elapsed, peak_gain_pct, final_gain)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_contract_ticker(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{date_str}{cp}{strike_int:08d}"


def get_ticks(conn, contract_ticker, after_ts):
    rows = conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
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
            underlying = row[4] or 0
            if mid > 0 or bid > 0:
                ticks.append(Tick(
                    ts=ts, mid=mid if mid > 0 else (bid + ask) / 2,
                    bid=bid, ask=ask, underlying=underlying,
                ))
        except (ValueError, TypeError):
            continue
    return ticks


def main():
    signals_db = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
    harvester_db = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

    sig_conn = sqlite3.connect(signals_db)
    sig_conn.row_factory = sqlite3.Row
    harv_conn = sqlite3.connect(harvester_db)

    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike, ts.expiry,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts,
               pt.id as trade_id, pt.premium_per_contract as traded_entry,
               pt.exit_premium as traded_exit, pt.pnl_dollars as traded_pnl,
               pt.exit_reason as traded_exit_reason, pt.contracts as traded_contracts
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.parent_trade_id IS NULL
        ORDER BY ts.created_at
    """).fetchall()

    sim_data = []
    no_data = no_strike = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        day = sig["day"]
        strike = sig["strike"]
        score = sig["score"] or 0
        premium = sig["atm_premium"] or sig["otm_premium"]

        if not strike or not premium or premium <= 0:
            no_strike += 1
            continue

        contract = build_contract_ticker(ticker, day, direction, strike)
        ticks = get_ticks(harv_conn, contract, sig["sig_ts"])
        if not ticks:
            no_data += 1
            continue

        first_tick = ticks[0]
        entry_price = first_tick.ask if first_tick.ask > 0 else first_tick.mid
        if entry_price <= 0:
            entry_price = premium

        if score >= 95: contracts = 5
        elif score >= 90: contracts = 4
        elif score >= 85: contracts = 3
        else: contracts = 1

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        peak = max(t.mid for t in ticks)
        peak_gain = (peak - entry_price) / entry_price * 100 if entry_price > 0 else 0

        sim_data.append({
            "id": sig["id"], "ticker": ticker, "direction": direction,
            "day": day, "score": score, "entry": entry_price,
            "contracts": contracts, "ticks": ticks, "sig_ts": sig_ts,
            "peak": peak, "peak_gain": peak_gain,
            "was_traded": sig["trade_id"] is not None,
            "actual_pnl": sig["traded_pnl"] or 0,
            "actual_exit_reason": sig["traded_exit_reason"] or "",
        })

    print(f"Signals: {len(signals)} total, {len(sim_data)} with tick data, "
          f"{no_data} no harvester, {no_strike} no strike/premium\n")

    # Run both scenarios
    scenarios = {
        "v4_production": (V4_GATES, V4_SETTINGS),
        "v4.1": (V41_GATES, V4_SETTINGS),
    }

    all_results = {}
    for sname, (gates, settings) in scenarios.items():
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], gates, settings)
            pnl = (er.premium - sd["entry"]) * sd["contracts"] * 100
            pnl = _apply_slippage(pnl)
            results.append((sd, er, pnl))
        all_results[sname] = results

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("v4.1 vs v4 PRODUCTION — SUMMARY")
    print("=" * 80)
    for sname in scenarios:
        results = all_results[sname]
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = len(results) - wins
        wr = wins / len(results) * 100
        avg_hold = sum(er.hold_minutes for _, er, _ in results) / len(results)
        avg_win = sum(pnl for _, _, pnl in results if pnl > 0) / wins if wins else 0
        avg_loss = sum(pnl for _, _, pnl in results if pnl <= 0) / losses if losses else 0
        print(f"  {sname:<15} P&L: ${total:>8.0f}  {wins}W/{losses}L ({wr:.0f}% WR)  "
              f"avg hold: {avg_hold:.0f}m  avg win: ${avg_win:.0f}  avg loss: ${avg_loss:.0f}")

    # -----------------------------------------------------------------------
    # Daily P&L breakdown
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("DAILY P&L BREAKDOWN")
    print(f"{'=' * 100}")

    days = sorted(set(sd["day"] for sd in sim_data))
    print(f"\n{'Day':<12} | {'v4 Prod P&L':>12} {'W/L':>6} {'Trades':>7} | "
          f"{'v4.1 P&L':>12} {'W/L':>6} {'Trades':>7} | {'Diff':>8} | Gate changes")
    print("-" * 110)

    cumul_v4 = 0
    cumul_v41 = 0

    for day in days:
        v4_day = [(sd, er, pnl) for sd, er, pnl in all_results["v4_production"] if sd["day"] == day]
        v41_day = [(sd, er, pnl) for sd, er, pnl in all_results["v4.1"] if sd["day"] == day]

        v4_pnl = sum(pnl for _, _, pnl in v4_day)
        v41_pnl = sum(pnl for _, _, pnl in v41_day)
        v4_wins = sum(1 for _, _, pnl in v4_day if pnl > 0)
        v41_wins = sum(1 for _, _, pnl in v41_day if pnl > 0)
        v4_losses = len(v4_day) - v4_wins
        v41_losses = len(v41_day) - v41_wins
        diff = v41_pnl - v4_pnl
        cumul_v4 += v4_pnl
        cumul_v41 += v41_pnl

        # Find gate changes for this day
        changes = []
        for i in range(len(sim_data)):
            if sim_data[i]["day"] != day:
                continue
            _, er_v4, pnl_v4 = all_results["v4_production"][i]
            _, er_v41, pnl_v41 = all_results["v4.1"][i]
            if er_v4.reason != er_v41.reason:
                changes.append(f"{sim_data[i]['ticker']}:{er_v4.reason}->{er_v41.reason}"
                               f"({'+' if pnl_v41-pnl_v4>0 else ''}{pnl_v41-pnl_v4:.0f})")

        changes_str = ", ".join(changes[:3])
        if len(changes) > 3:
            changes_str += f" +{len(changes)-3} more"

        print(f"{day:<12} | ${v4_pnl:>11.0f} {v4_wins}W/{v4_losses}L {len(v4_day):>5}  | "
              f"${v41_pnl:>11.0f} {v41_wins}W/{v41_losses}L {len(v41_day):>5}  | "
              f"${diff:>+7.0f} | {changes_str}")

    print("-" * 110)
    print(f"{'CUMULATIVE':<12} | ${cumul_v4:>11.0f} {'':>6} {'':>7} | "
          f"${cumul_v41:>11.0f} {'':>6} {'':>7} | ${cumul_v41 - cumul_v4:>+7.0f} |")

    # -----------------------------------------------------------------------
    # Gate fire comparison
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("GATE FIRE COMPARISON")
    print(f"{'=' * 80}")

    for sname in scenarios:
        results = all_results[sname]
        gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "hold_sum": 0})
        for sd, er, pnl in results:
            g = gate_stats[er.reason]
            g["count"] += 1
            g["pnl"] += pnl
            if pnl > 0: g["wins"] += 1
            g["hold_sum"] += er.hold_minutes

        sorted_gates = sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        print(f"\n{sname}:")
        print(f"  {'Gate':<22} {'Fires':>6} {'%':>6} {'P&L':>9} {'WR':>6} {'AvgHold':>8}")
        for gate, stats in sorted_gates:
            ct = stats["count"]
            pct = ct / len(results) * 100
            wr = stats["wins"] / ct * 100 if ct else 0
            avg_hold = stats["hold_sum"] / ct if ct else 0
            print(f"  {gate:<22} {ct:>6} {pct:>5.1f}% ${stats['pnl']:>8.0f} {wr:>5.0f}% {avg_hold:>6.0f}m")

    # -----------------------------------------------------------------------
    # Per-signal detail (trades that changed)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("TRADES THAT CHANGED BETWEEN v4 AND v4.1")
    print(f"{'=' * 100}")
    print(f"{'#':<4} {'Ticker':<7} {'Day':<12} {'Peak%':>7} "
          f"| {'v4 Gate':<18} {'v4 P&L':>8} {'Hold':>6} "
          f"| {'v4.1 Gate':<18} {'v4.1 P&L':>8} {'Hold':>6} | {'Diff':>8}")
    print("-" * 105)

    total_improved = 0
    total_regressed = 0
    n_improved = 0
    n_regressed = 0

    for i in range(len(sim_data)):
        sd = sim_data[i]
        _, er_v4, pnl_v4 = all_results["v4_production"][i]
        _, er_v41, pnl_v41 = all_results["v4.1"][i]

        if er_v4.reason != er_v41.reason:
            diff = pnl_v41 - pnl_v4
            marker = "+" if diff > 0 else "-" if diff < 0 else "="
            print(f"{i+1:<4} {sd['ticker']:<7} {sd['day']:<12} {sd['peak_gain']:>+6.0f}% "
                  f"| {er_v4.reason:<18} ${pnl_v4:>7.0f} {er_v4.hold_minutes:>5.0f}m "
                  f"| {er_v41.reason:<18} ${pnl_v41:>7.0f} {er_v41.hold_minutes:>5.0f}m "
                  f"| ${diff:>+7.0f} {marker}")
            if diff > 0:
                total_improved += diff
                n_improved += 1
            elif diff < 0:
                total_regressed += diff
                n_regressed += 1

    print(f"\n  Improved: {n_improved} trades, +${total_improved:.0f}")
    print(f"  Regressed: {n_regressed} trades, ${total_regressed:.0f}")
    print(f"  Net: ${total_improved + total_regressed:>+.0f}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
