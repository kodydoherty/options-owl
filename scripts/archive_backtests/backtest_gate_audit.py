#!/usr/bin/env python3
"""Gate audit: test each v4.1 exit gate individually against ALL trade data.

For each gate, shows:
  1. How many times it fires as the PRIMARY exit
  2. P&L when it fires vs what would happen if we skipped it
  3. "What-if" analysis: if this gate didn't exist, which gate would fire instead?

Also simulates today's (most recent day's) trades under v4.1 to show how
the Kody agent would have performed.

Usage:
  python scripts/backtest_gate_audit.py [signals_db] [harvester_db]
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

def _apply_slippage(pnl):
    return pnl * (1 - SLIPPAGE_HAIRCUT) if pnl > 0 else pnl


SETTINGS = {
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
    "thesis_cut_threshold_pct": 40.0, "thesis_cut_lookback": 8,
    "thesis_cut_new_low_exit": 3, "thesis_cut_bounce_hold_pct": 5.0,
    "thesis_cut_min_ticks": 4,
    "underlying_trail_tiers": [(100, 0.50), (50, 0.40), (15, 0.30), (0, 0.20)],
}

V41_GATES = {"stop_loss", "be_clamp", "soft_trail", "adaptive_trail",
             "underlying_trail", "profit_floor", "thesis_cut", "dollar_trail", "eod"}


def simulate(entry, ticks, signal_ts, contracts, direction, enabled_gates, settings):
    if not ticks or entry <= 0:
        return ExitResult(entry, "no_data")
    s = settings
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

        in_grace = False
        if elapsed_min < s["grace_minutes"]:
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

        if "eod" in enabled_gates:
            if et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")

    last = ticks[-1]
    final_gain = (last.mid - entry) / entry * 100 if entry > 0 else 0
    elapsed = (ticks[-1].ts - signal_ts).total_seconds() / 60
    return ExitResult(last.mid, "eod_data_end", elapsed, peak_gain_pct, final_gain)


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
                ticks.append(Tick(ts=ts, mid=mid if mid > 0 else (bid + ask) / 2,
                                  bid=bid, ask=ask, underlying=underlying))
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
    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        day = sig["day"]
        strike = sig["strike"]
        score = sig["score"] or 0
        premium = sig["atm_premium"] or sig["otm_premium"]
        if not strike or not premium or premium <= 0:
            continue

        contract = build_contract_ticker(ticker, day, direction, strike)
        ticks = get_ticks(harv_conn, contract, sig["sig_ts"])
        if not ticks:
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

    print(f"Loaded {len(sim_data)} signals with tick data\n")

    # -----------------------------------------------------------------------
    # 1. Gate-by-gate "what-if" audit
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("GATE-BY-GATE AUDIT: what happens if each gate is individually removed?")
    print("=" * 90)

    # First run v4.1 baseline
    v41_results = []
    for sd in sim_data:
        er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                      sd["direction"], V41_GATES, SETTINGS)
        pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
        v41_results.append((sd, er, pnl))

    v41_total = sum(pnl for _, _, pnl in v41_results)
    print(f"\nv4.1 baseline: ${v41_total:.0f}")

    # Test removing each gate one at a time
    print(f"\n{'Gate removed':<22} {'New P&L':>9} {'Diff':>8} {'Impact':>10}")
    print("-" * 55)

    for gate_to_remove in sorted(V41_GATES):
        reduced_gates = V41_GATES - {gate_to_remove}
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], reduced_gates, SETTINGS)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        diff = total - v41_total
        impact = "HELPFUL" if diff < -50 else "HARMFUL" if diff > 50 else "neutral"
        print(f"  -{gate_to_remove:<20} ${total:>8.0f} ${diff:>+7.0f}   {impact}")

    # -----------------------------------------------------------------------
    # 2. Conflict analysis: gates that overlap
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 90}")
    print("OVERLAP ANALYSIS: soft_trail vs profit_floor vs dollar_trail")
    print("=" * 90)

    # These 3 gates all protect gains in similar ranges. Which one actually matters?
    for combo_name, gates in [
        ("no soft_trail", V41_GATES - {"soft_trail"}),
        ("no profit_floor", V41_GATES - {"profit_floor"}),
        ("no dollar_trail", V41_GATES - {"dollar_trail"}),
        ("no soft+floor", V41_GATES - {"soft_trail", "profit_floor"}),
        ("only dollar+adaptive", {"stop_loss", "adaptive_trail", "dollar_trail", "eod"}),
    ]:
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], gates, SETTINGS)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = len(results) - wins
        diff = total - v41_total
        print(f"  {combo_name:<25} ${total:>8.0f} ({diff:>+6.0f}) {wins}W/{losses}L")

    # -----------------------------------------------------------------------
    # 3. Sensitivity analysis: key parameter tuning
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 90}")
    print("SENSITIVITY: key parameter sweep")
    print("=" * 90)

    # Test different stop loss %
    print(f"\n  Premium stop loss %:")
    for stop_pct in [20, 25, 30, 35, 40, 50]:
        s = {**SETTINGS, "premium_stop_pct": stop_pct}
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], V41_GATES, s)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        stops = sum(1 for _, er, _ in results if er.reason == "stop_loss")
        marker = " <-- current" if stop_pct == 30 else ""
        print(f"    {stop_pct}%: ${total:>8.0f}  {wins}W/{len(results)-wins}L  "
              f"{stops} stops{marker}")

    # Test different dollar trail activation
    print(f"\n  Dollar trail activation %:")
    for act_pct in [20, 30, 40, 50, 60, 80]:
        s = {**SETTINGS, "dollar_trail_activation_pct": act_pct}
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], V41_GATES, s)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        dt_fires = sum(1 for _, er, _ in results if er.reason == "dollar_trail")
        marker = " <-- current" if act_pct == 40 else ""
        print(f"    {act_pct}%: ${total:>8.0f}  {wins}W/{len(results)-wins}L  "
              f"{dt_fires} dollar_trail fires{marker}")

    # Test different grace periods
    print(f"\n  Grace period (minutes):")
    for grace in [5, 10, 15, 20, 25, 30]:
        s = {**SETTINGS, "grace_minutes": grace}
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], V41_GATES, s)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        marker = " <-- current" if grace == 20 else ""
        print(f"    {grace}m: ${total:>8.0f}  {wins}W/{len(results)-wins}L{marker}")

    # Test soft trail floor %
    print(f"\n  Soft trail floor % (keep this % of peak gain):")
    for floor_pct in [30, 40, 50, 60, 70, 80]:
        s = {**SETTINGS, "soft_trail_floor_pct": floor_pct}
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                          sd["direction"], V41_GATES, s)
            pnl = _apply_slippage((er.premium - sd["entry"]) * sd["contracts"] * 100)
            results.append((sd, er, pnl))
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        st_fires = sum(1 for _, er, _ in results if er.reason == "soft_trail")
        marker = " <-- current" if floor_pct == 50 else ""
        print(f"    {floor_pct}%: ${total:>8.0f}  {wins}W/{len(results)-wins}L  "
              f"{st_fires} soft_trail fires{marker}")

    # -----------------------------------------------------------------------
    # 4. Most recent trading day detail
    # -----------------------------------------------------------------------
    last_day = max(sd["day"] for sd in sim_data)
    print(f"\n\n{'=' * 90}")
    print(f"MOST RECENT DAY DETAIL: {last_day} (under v4.1)")
    print(f"{'=' * 90}")

    day_trades = [(sd, er, pnl) for sd, er, pnl in v41_results if sd["day"] == last_day]
    print(f"\n{'#':<4} {'Ticker':<7} {'Dir':<5} {'Score':>5} {'Entry':>6} {'Peak%':>7} "
          f"{'Gate':<18} {'P&L':>8} {'Hold':>6} {'Actual':>8} {'ActGate':<15}")
    print("-" * 100)

    day_pnl = 0
    for sd, er, pnl in day_trades:
        day_pnl += pnl
        print(f"{sd['id']:<4} {sd['ticker']:<7} {sd['direction'][:4]:<5} {sd['score']:>5} "
              f"${sd['entry']:>5.2f} {sd['peak_gain']:>+6.0f}% "
              f"{er.reason:<18} ${pnl:>7.0f} {er.hold_minutes:>5.0f}m "
              f"${sd['actual_pnl']:>7.0f} {sd['actual_exit_reason']:<15}")

    print(f"\n  v4.1 day P&L: ${day_pnl:.0f}")
    actual_day_pnl = sum(sd["actual_pnl"] for sd, _, _ in day_trades if sd["was_traded"])
    traded_count = sum(1 for sd, _, _ in day_trades if sd["was_traded"])
    print(f"  Actual day P&L: ${actual_day_pnl:.0f} ({traded_count} trades actually executed)")

    # -----------------------------------------------------------------------
    # 5. Critical weakness analysis
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 90}")
    print("CRITICAL WEAKNESSES IN v4.1")
    print(f"{'=' * 90}")

    # Count stop_loss fires by peak gain (were these really losers or just premature?)
    print(f"\n1. STOP LOSS ANALYSIS: are stops firing on trades that eventually recover?")
    stop_trades = [(sd, er, pnl) for sd, er, pnl in v41_results if er.reason == "stop_loss"]
    for sd, er, pnl in stop_trades:
        # Check if price recovered after stop fired
        stop_tick_idx = None
        for j, t in enumerate(sd["ticks"]):
            if t.ts.tzinfo is None:
                t_ts = t.ts.replace(tzinfo=timezone.utc)
            else:
                t_ts = t.ts
            elapsed = (t_ts - sd["sig_ts"]).total_seconds() / 60
            if elapsed >= er.hold_minutes - 0.5:
                stop_tick_idx = j
                break
        if stop_tick_idx is not None:
            remaining = sd["ticks"][stop_tick_idx:]
            if remaining:
                post_peak = max(t.mid for t in remaining)
                post_peak_gain = (post_peak - sd["entry"]) / sd["entry"] * 100
                if post_peak_gain > 10:
                    print(f"  {sd['ticker']} {sd['day']}: stopped at {er.exit_gain_pct:+.0f}%, "
                          f"but later peaked at +{post_peak_gain:.0f}% (missed ${(post_peak - sd['entry']) * sd['contracts'] * 100 * 0.85:.0f})")

    # BE clamp analysis
    print(f"\n2. BE CLAMP: is it protecting or hurting?")
    be_trades = [(sd, er, pnl) for sd, er, pnl in v41_results if er.reason == "be_clamp"]
    if be_trades:
        for sd, er, pnl in be_trades:
            # What would have happened without it?
            no_be = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                             sd["direction"], V41_GATES - {"be_clamp"}, SETTINGS)
            no_be_pnl = _apply_slippage((no_be.premium - sd["entry"]) * sd["contracts"] * 100)
            print(f"  {sd['ticker']} {sd['day']}: BE exit at ${pnl:.0f}, without BE: "
                  f"${no_be_pnl:.0f} via {no_be.reason} ({'+' if no_be_pnl > pnl else ''}"
                  f"${no_be_pnl - pnl:.0f})")
    else:
        print(f"  BE clamp never fires in v4.1 — consider disabling")

    # Dollar trail: is 40% activation too tight?
    print(f"\n3. DOLLAR TRAIL: biggest winners left on the table")
    dt_trades = [(sd, er, pnl) for sd, er, pnl in v41_results if er.reason == "dollar_trail"]
    dt_trades.sort(key=lambda x: x[0]["peak_gain"], reverse=True)
    for sd, er, pnl in dt_trades[:5]:
        captured_pct = er.exit_gain_pct / sd["peak_gain"] * 100 if sd["peak_gain"] > 0 else 0
        print(f"  {sd['ticker']} {sd['day']}: peak +{sd['peak_gain']:.0f}%, "
              f"exited at +{er.exit_gain_pct:.0f}% (captured {captured_pct:.0f}% of move), "
              f"P&L ${pnl:.0f} in {er.hold_minutes:.0f}m")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
