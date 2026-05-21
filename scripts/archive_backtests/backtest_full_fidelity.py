#!/usr/bin/env python3
"""Full-fidelity backtest: simulates ALL 18+ enabled production exit gates.

Answers Vince's question: "which gates actually fire and kill trades?"

Tests 3 scenarios:
  1. v4_production — all currently enabled gates, production settings
  2. v4_trimmed   — disable/loosen legacy hold-time gates that kill winners
  3. v4_minimal   — only hard stop + adaptive trail + BE clamp + soft trail + EOD

For each trade, tracks which gate fires first. Shows:
  - Gate fire frequency (which gates are actually deciding exits)
  - Per-gate P&L impact (are gates that fire producing wins or losses?)
  - Hold time distribution per scenario
  - Conflicting gate analysis (when would a different gate have been better?)

Usage:
  python scripts/backtest_full_fidelity.py [signals_db] [harvester_db]
"""

import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

SLIPPAGE_HAIRCUT = 0.15  # 15% on gains


# ---------------------------------------------------------------------------
# Production settings (all from settings.py defaults)
# ---------------------------------------------------------------------------

SETTINGS = {
    # Grace / smart grace
    "grace_minutes": 20,
    "smart_grace": True,
    "smart_grace_confirm_pct": 0.1,

    # Hard stop
    "premium_stop_pct": 30.0,

    # BE clamp
    "be_clamp_activation_pct": 15.0,

    # Soft trail
    "soft_trail_min_pct": 15.0,
    "soft_trail_max_pct": 35.0,
    "soft_trail_floor_pct": 50.0,

    # Adaptive trail
    "adaptive_activation_pct": 35.0,
    "adaptive_active_width": 35.0,
    "adaptive_runner_threshold": 150.0,
    "adaptive_runner_width": 45.0,
    "adaptive_moonshot_threshold": 400.0,
    "adaptive_moonshot_width": 30.0,

    # Dollar trail
    "dollar_trail_activation_pct": 40.0,
    "dollar_trail_small_step_pct": 20.0,
    "dollar_trail_step_threshold_pct": 25.0,
    "dollar_trail_large_step_pct": 10.0,

    # Profit floor
    "profit_floor_activation_pct": 15.0,
    "profit_floor_ratchet_pct": 60.0,

    # Theta bleed
    "theta_bleed_hold_min": 45.0,
    "theta_bleed_max_loss_pct": 30.0,

    # No momentum
    "no_momentum_minutes": 45.0,
    "no_momentum_min_gain_pct": 5.0,

    # Time decay zone
    "time_decay_hold_min": 45.0,
    "time_decay_afternoon_hour": 15,
    "time_decay_afternoon_min": 30,
    "time_decay_stale_min": 10.0,

    # Time tighten
    "time_tighten_after_min": 60.0,
    "time_tighten_factor": 0.7,

    # Decel exit
    "decel_min_hold_sec": 480,
    "decel_min_gain_pct": 5.0,
    "decel_short_window": 5,
    "decel_long_window": 15,
    "decel_threshold": -3.0,

    # Thesis cut
    "thesis_cut_threshold_pct": 40.0,
    "thesis_cut_lookback": 8,
    "thesis_cut_new_low_exit": 3,
    "thesis_cut_bounce_hold_pct": 5.0,
    "thesis_cut_min_ticks": 4,

    # Tranche scaleout
    "tranche_lock_gain_pct": 25.0,
    "tranche_min_contracts": 3,

    # Underlying trail tiers: gain% : trail%
    "underlying_trail_tiers": [(100, 0.50), (50, 0.40), (15, 0.30), (0, 0.20)],
}


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
# Full-fidelity tick-by-tick simulator with ALL production gates
# ---------------------------------------------------------------------------

def simulate_full(entry, ticks, signal_ts, contracts, direction, enabled_gates, settings):
    """Simulate all enabled exit gates in production priority order.

    enabled_gates: set of gate names to enable.
    Returns ExitResult with the gate that fired.
    """
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
    tranche_done = False
    thesis_ticks_in_zone = 0
    premium_history = []  # for decel
    grace_ended_by_smart = False

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue

        # Track peaks
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

        # Derived metrics
        gain_pct = (price - entry) / entry * 100
        peak_gain_pct = (peak_premium - entry) / entry * 100
        drop_from_entry = max(0, (entry - price) / entry * 100)
        drop_from_peak = (peak_premium - price) / peak_premium * 100 if peak_premium > 0 else 0
        elapsed_min = (tick.ts - signal_ts).total_seconds() / 60

        # Premium history for decel
        premium_history.append((tick.ts, price))

        # ET approximation (UTC - 4 for EDT)
        et_hour = (tick.ts.hour - 4) % 24
        et_min = tick.ts.minute

        # --- Grace period ---
        in_grace = False
        if elapsed_min < grace_minutes:
            if s.get("smart_grace") and entry_underlying > 0 and tick.underlying > 0:
                move_pct = (tick.underlying - entry_underlying) / entry_underlying * 100
                is_call = direction.lower() in ("call", "bullish", "long")
                confirmed = (is_call and move_pct > s["smart_grace_confirm_pct"]) or \
                            (not is_call and move_pct < -s["smart_grace_confirm_pct"])
                if confirmed:
                    grace_ended_by_smart = True
                    in_grace = False
                else:
                    in_grace = True
            else:
                in_grace = True

        def _exit(reason):
            return ExitResult(price, reason, elapsed_min, peak_gain_pct,
                              gain_pct)

        # =================================================================
        # EXIT GATES IN PRODUCTION PRIORITY ORDER
        # =================================================================

        # --- Gate 1: Hard stop (premium-based) ---
        if "stop_loss" in enabled_gates and not in_grace:
            if drop_from_entry >= s["premium_stop_pct"]:
                return _exit("stop_loss")

        # --- Gate 2: BE clamp ---
        if "be_clamp" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["be_clamp_activation_pct"]:
                be_clamp_active = True
            if be_clamp_active and price <= entry:
                return _exit("be_clamp")

        # --- Gate 3: Tranche scaleout (partial — we simulate as hold) ---
        # Tranche doesn't close the full position, just locks 1/3.
        # Skip in backtest — it's a partial, not an exit.

        # --- Gate 4: Volume peak (flag only, doesn't exit) ---
        # We don't have real candle data, skip the tighten modifier.

        # --- Gate 5: Soft trail (15-35% band) ---
        if "soft_trail" in enabled_gates and not in_grace:
            if s["soft_trail_min_pct"] <= peak_gain_pct < s["soft_trail_max_pct"]:
                gain_at_peak = peak_premium - entry
                floor = entry + gain_at_peak * (s["soft_trail_floor_pct"] / 100)
                if price <= floor:
                    return _exit("soft_trail")

        # --- Gate 6: Dollar trail ---
        if "dollar_trail" in enabled_gates and not in_grace:
            profit_pct = (peak_premium - entry) / entry * 100
            if profit_pct >= s["dollar_trail_activation_pct"]:
                profit_dollars = (peak_premium - entry) * 100  # per contract
                if profit_dollars >= s["dollar_trail_step_threshold_pct"] / 100 * entry * 100:
                    step = s["dollar_trail_large_step_pct"] / 100 * entry * 100
                else:
                    step = s["dollar_trail_small_step_pct"] / 100 * entry * 100
                if step > 0:
                    steps_hit = int(profit_dollars / step)
                    trail_floor_dollars = steps_hit * step
                    trail_floor_premium = entry + trail_floor_dollars / 100
                    if price <= trail_floor_premium and trail_floor_dollars > 0:
                        return _exit("dollar_trail")

        # --- Gate 7: Profit retrace (disabled in prod) ---

        # --- Gate 8: Decel exit ---
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

        # --- Gate 9: Profit lock (disabled in prod) ---

        # --- Gate 10: Underlying trail ---
        if "underlying_trail" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["adaptive_activation_pct"] and peak_underlying > 0 and tick.underlying > 0:
                # Find applicable tier
                trail_pct = 0.20
                for tier_gain, tier_trail in s["underlying_trail_tiers"]:
                    if peak_gain_pct >= tier_gain:
                        trail_pct = tier_trail
                        break
                is_call = direction.lower() in ("call", "bullish", "long")
                if is_call:
                    trail_floor = peak_underlying * (1 - trail_pct)
                    if tick.underlying <= trail_floor:
                        return _exit("underlying_trail")
                else:
                    trail_ceil = peak_underlying * (1 + trail_pct)
                    if tick.underlying >= trail_ceil:
                        return _exit("underlying_trail")

        # --- Gate 11: Adaptive trailing stop ---
        if "adaptive_trail" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["adaptive_moonshot_threshold"]:
                if drop_from_peak >= s["adaptive_moonshot_width"]:
                    return _exit("adaptive_trail_moonshot")
            elif peak_gain_pct >= s["adaptive_runner_threshold"]:
                if drop_from_peak >= s["adaptive_runner_width"]:
                    return _exit("adaptive_trail_runner")
            elif peak_gain_pct >= s["adaptive_activation_pct"]:
                if drop_from_peak >= s["adaptive_active_width"]:
                    return _exit("adaptive_trail_active")

        # --- Gate 12: Adaptive time tighten ---
        if "time_tighten" in enabled_gates and not in_grace:
            if elapsed_min >= s["time_tighten_after_min"]:
                # Uses phase trails — approximate with 25% base trail
                base_trail = 25.0
                tightened = base_trail * s["time_tighten_factor"]
                if drop_from_peak >= tightened:
                    return _exit("time_tighten")

        # --- Gate 13: Profit floor ---
        if "profit_floor" in enabled_gates and not in_grace:
            if peak_gain_pct >= s["profit_floor_activation_pct"]:
                peak_gain_dollars = peak_premium - entry
                floor = entry + peak_gain_dollars * (s["profit_floor_ratchet_pct"] / 100)
                if price <= floor:
                    return _exit("profit_floor")

        # --- Gate 14: Thesis cut ---
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

        # --- Gate 15: Targets (T5-T1) ---
        # We don't have target prices from signals in the backtest.
        # These are scale-out exits, skip for now.

        # --- Gate 16: Theta bleed ---
        if "theta_bleed" in enabled_gates:
            if elapsed_min >= s["theta_bleed_hold_min"] and drop_from_entry >= s["theta_bleed_max_loss_pct"]:
                return _exit("theta_bleed")

        # --- Gate 17: Time decay zone ---
        if "time_decay" in enabled_gates:
            afternoon = (et_hour > s["time_decay_afternoon_hour"] or
                         (et_hour == s["time_decay_afternoon_hour"] and et_min >= s["time_decay_afternoon_min"]))
            if afternoon or elapsed_min >= s["time_decay_hold_min"]:
                if last_new_high_ts:
                    stale_min = (tick.ts - last_new_high_ts).total_seconds() / 60
                    if stale_min >= s["time_decay_stale_min"]:
                        return _exit("time_decay")

        # --- Gate 18: No momentum ---
        if "no_momentum" in enabled_gates:
            if elapsed_min >= s["no_momentum_minutes"] and gain_pct < s["no_momentum_min_gain_pct"]:
                return _exit("no_momentum")

        # --- Gate 19: EOD cutoff (3:45 PM ET) ---
        if "eod" in enabled_gates:
            if et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")

    # End of data
    last = ticks[-1]
    final_gain = (last.mid - entry) / entry * 100 if entry > 0 else 0
    elapsed = (ticks[-1].ts - signal_ts).total_seconds() / 60
    return ExitResult(last.mid, "eod_data_end", elapsed, peak_gain_pct, final_gain)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# All production-enabled gates
PRODUCTION_GATES = {
    "stop_loss", "be_clamp", "soft_trail", "adaptive_trail",
    "underlying_trail", "time_tighten", "profit_floor",
    "thesis_cut", "decel_exit", "theta_bleed", "time_decay",
    "no_momentum", "dollar_trail", "eod",
}

# Trimmed: disable legacy hold-time gates that Vince flagged
TRIMMED_GATES = {
    "stop_loss", "be_clamp", "soft_trail", "adaptive_trail",
    "underlying_trail", "profit_floor", "thesis_cut",
    "decel_exit", "dollar_trail", "eod",
    # REMOVED: no_momentum, time_decay, time_tighten, theta_bleed
}

# Minimal: only the core exit mechanisms
MINIMAL_GATES = {
    "stop_loss", "be_clamp", "soft_trail", "adaptive_trail", "eod",
}

# No decel: decel_exit is the #1 offender (37% of exits, cuts runners)
NO_DECEL_GATES = PRODUCTION_GATES - {"decel_exit"}

# No decel + no dollar trail (both cut winners early)
NO_DECEL_NO_DOLLAR = PRODUCTION_GATES - {"decel_exit", "dollar_trail"}

# Smart trim: remove decel + loosen time gates, keep dollar trail for protection
SMART_TRIM_GATES = PRODUCTION_GATES - {"decel_exit", "no_momentum", "time_decay", "time_tighten"}

# Smart trim + no dollar trail
SMART_NO_DOLLAR = SMART_TRIM_GATES - {"dollar_trail"}

SCENARIOS = {
    "v4_production": {
        "desc": "All 14 enabled exit gates (current production)",
        "gates": PRODUCTION_GATES,
        "settings": SETTINGS,
    },
    "no_decel": {
        "desc": "Production minus decel_exit (the #1 early-exit gate)",
        "gates": NO_DECEL_GATES,
        "settings": SETTINGS,
    },
    "no_decel_dollar": {
        "desc": "Production minus decel_exit and dollar_trail",
        "gates": NO_DECEL_NO_DOLLAR,
        "settings": SETTINGS,
    },
    "smart_trim": {
        "desc": "Remove decel + no_momentum + time_decay + time_tighten",
        "gates": SMART_TRIM_GATES,
        "settings": SETTINGS,
    },
    "smart_no_dollar": {
        "desc": "Smart trim + remove dollar_trail",
        "gates": SMART_NO_DOLLAR,
        "settings": SETTINGS,
    },
    "v4_minimal": {
        "desc": "Only hard stop + BE clamp + soft trail + adaptive trail + EOD",
        "gates": MINIMAL_GATES,
        "settings": SETTINGS,
    },
}


# ---------------------------------------------------------------------------
# Data loading (same as backtest_v22_features.py)
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
                    ts=ts,
                    mid=mid if mid > 0 else (bid + ask) / 2,
                    bid=bid, ask=ask,
                    underlying=underlying,
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
               pt.exit_reason as traded_exit_reason, pt.contracts as traded_contracts,
               pt.mfe_premium as traded_mfe
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.parent_trade_id IS NULL
        ORDER BY ts.created_at
    """).fetchall()

    print(f"Total signals in DB: {len(signals)}")

    # Load tick data
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

        # Score-based sizing ($8K portfolio)
        if score >= 95:
            contracts = 5
        elif score >= 90:
            contracts = 4
        elif score >= 85:
            contracts = 3
        elif score >= 78:
            contracts = 1
        else:
            contracts = 1

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        peak = max(t.mid for t in ticks)
        peak_gain = (peak - entry_price) / entry_price * 100 if entry_price > 0 else 0

        sim_data.append({
            "id": sig["id"],
            "ticker": ticker,
            "direction": direction,
            "day": day,
            "score": score,
            "entry": entry_price,
            "contracts": contracts,
            "ticks": ticks,
            "sig_ts": sig_ts,
            "peak": peak,
            "peak_gain": peak_gain,
            "was_traded": sig["trade_id"] is not None,
            "actual_pnl": sig["traded_pnl"] or 0,
            "actual_exit_reason": sig["traded_exit_reason"] or "",
        })

    print(f"Signals with tick data: {len(sim_data)}")
    print(f"No harvester data: {no_data}, No strike/premium: {no_strike}")
    print()

    # ---------------------------------------------------------------------------
    # Run all scenarios
    # ---------------------------------------------------------------------------

    all_results = {}  # scenario -> list of (signal_dict, ExitResult, pnl)

    for sname, scenario in SCENARIOS.items():
        results = []
        for sd in sim_data:
            er = simulate_full(
                sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"],
                sd["direction"], scenario["gates"], scenario["settings"],
            )
            pnl = (er.premium - sd["entry"]) * sd["contracts"] * 100
            pnl = _apply_slippage(pnl)
            results.append((sd, er, pnl))
        all_results[sname] = results

    # ---------------------------------------------------------------------------
    # Report: Scenario comparison
    # ---------------------------------------------------------------------------

    print("=" * 90)
    print("SCENARIO COMPARISON")
    print("=" * 90)
    print(f"{'Scenario':<20} {'Desc':<50} {'P&L':>8} {'W/L':>8} {'WR':>6} {'AvgHold':>8}")
    print("-" * 90)

    for sname in SCENARIOS:
        results = all_results[sname]
        total_pnl = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = sum(1 for _, _, pnl in results if pnl <= 0)
        wr = wins / len(results) * 100 if results else 0
        avg_hold = sum(er.hold_minutes for _, er, _ in results) / len(results) if results else 0
        desc = SCENARIOS[sname]["desc"][:48]
        print(f"{sname:<20} {desc:<50} ${total_pnl:>7.0f} {wins}W/{losses}L {wr:>5.1f}% {avg_hold:>6.0f}m")
    print()

    # ---------------------------------------------------------------------------
    # Report: Gate fire frequency for each scenario
    # ---------------------------------------------------------------------------

    for sname in SCENARIOS:
        results = all_results[sname]
        print(f"\n{'=' * 80}")
        print(f"GATE FIRE ANALYSIS: {sname}")
        print(f"{'=' * 80}")

        gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0,
                                          "hold_mins": [], "peak_gains": []})

        for sd, er, pnl in results:
            reason = er.reason
            # Normalize adaptive_trail variants
            display_reason = reason
            if reason.startswith("adaptive_trail_"):
                display_reason = "adaptive_trail"

            g = gate_stats[display_reason]
            g["count"] += 1
            g["pnl"] += pnl
            if pnl > 0:
                g["wins"] += 1
            g["hold_mins"].append(er.hold_minutes)
            g["peak_gains"].append(er.peak_gain_pct)

        # Sort by count descending
        sorted_gates = sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True)

        print(f"\n{'Gate':<25} {'Fires':>6} {'%':>6} {'P&L':>9} {'W/L':>8} {'WR':>6} "
              f"{'AvgHold':>8} {'AvgPeak':>8}")
        print("-" * 80)

        total = len(results)
        for gate, stats in sorted_gates:
            ct = stats["count"]
            pct = ct / total * 100
            pnl_sum = stats["pnl"]
            wins = stats["wins"]
            losses = ct - wins
            wr = wins / ct * 100 if ct > 0 else 0
            avg_hold = sum(stats["hold_mins"]) / ct if ct > 0 else 0
            avg_peak = sum(stats["peak_gains"]) / ct if ct > 0 else 0
            print(f"{gate:<25} {ct:>6} {pct:>5.1f}% ${pnl_sum:>8.0f} "
                  f"{wins}W/{losses}L {wr:>5.1f}% {avg_hold:>6.0f}m {avg_peak:>+7.0f}%")

    # ---------------------------------------------------------------------------
    # Report: Per-signal detail comparison
    # ---------------------------------------------------------------------------

    print(f"\n\n{'=' * 120}")
    print("PER-SIGNAL DETAIL: which gate fires for each signal in each scenario")
    print(f"{'=' * 120}")

    prod_results = all_results["v4_production"]
    smart_results = all_results["smart_trim"]
    min_results = all_results["v4_minimal"]

    print(f"\n{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Entry':>6} {'Peak%':>7} "
          f"| {'Prod Gate':<20} {'P&L':>8} {'Hold':>6} "
          f"| {'Smart Gate':<20} {'P&L':>8} {'Hold':>6} "
          f"| {'Min Gate':<20} {'P&L':>8} {'Hold':>6}")
    print("-" * 160)

    for i in range(len(sim_data)):
        sd = sim_data[i]
        _, er_p, pnl_p = prod_results[i]
        _, er_t, pnl_t = smart_results[i]
        _, er_m, pnl_m = min_results[i]

        # Highlight rows where smart/minimal significantly outperforms production
        marker = ""
        if pnl_t - pnl_p > 50:
            marker = " <<<"
        elif pnl_p - pnl_t > 50:
            marker = " >>>"

        print(f"{i+1:<4} {sd['ticker']:<7} {sd['direction'][:4]:<5} {sd['day']:<12} "
              f"${sd['entry']:>5.2f} {sd['peak_gain']:>+6.0f}% "
              f"| {er_p.reason:<20} ${pnl_p:>7.0f} {er_p.hold_minutes:>5.0f}m "
              f"| {er_t.reason:<20} ${pnl_t:>7.0f} {er_t.hold_minutes:>5.0f}m "
              f"| {er_m.reason:<20} ${pnl_m:>7.0f} {er_m.hold_minutes:>5.0f}m"
              f"{marker}")

    # ---------------------------------------------------------------------------
    # Report: Conflict analysis — trades where different scenarios disagree
    # ---------------------------------------------------------------------------

    print(f"\n\n{'=' * 100}")
    print("CONFLICT ANALYSIS: trades where trimming gates changes the outcome")
    print(f"{'=' * 100}")

    improvements = []
    regressions = []

    for i in range(len(sim_data)):
        sd = sim_data[i]
        _, er_p, pnl_p = prod_results[i]
        _, er_t, pnl_t = smart_results[i]
        diff = pnl_t - pnl_p

        if abs(diff) > 10:  # $10 threshold for significance
            row = {
                "idx": i, "ticker": sd["ticker"], "direction": sd["direction"],
                "day": sd["day"], "peak_gain": sd["peak_gain"],
                "prod_gate": er_p.reason, "prod_pnl": pnl_p, "prod_hold": er_p.hold_minutes,
                "trim_gate": er_t.reason, "trim_pnl": pnl_t, "trim_hold": er_t.hold_minutes,
                "diff": diff,
            }
            if diff > 0:
                improvements.append(row)
            else:
                regressions.append(row)

    improvements.sort(key=lambda x: x["diff"], reverse=True)
    regressions.sort(key=lambda x: x["diff"])

    print(f"\nIMPROVEMENTS (trimmed > production): {len(improvements)} trades, "
          f"total +${sum(r['diff'] for r in improvements):.0f}")
    if improvements:
        print(f"  {'Ticker':<7} {'Day':<12} {'Prod Gate':<20} {'Prod P&L':>9} {'Hold':>6} "
              f"{'Trim Gate':<20} {'Trim P&L':>9} {'Hold':>6} {'Diff':>8}")
        for r in improvements[:15]:
            print(f"  {r['ticker']:<7} {r['day']:<12} {r['prod_gate']:<20} ${r['prod_pnl']:>8.0f} "
                  f"{r['prod_hold']:>5.0f}m {r['trim_gate']:<20} ${r['trim_pnl']:>8.0f} "
                  f"{r['trim_hold']:>5.0f}m ${r['diff']:>+7.0f}")

    print(f"\nREGRESSIONS (trimmed < production): {len(regressions)} trades, "
          f"total ${sum(r['diff'] for r in regressions):.0f}")
    if regressions:
        print(f"  {'Ticker':<7} {'Day':<12} {'Prod Gate':<20} {'Prod P&L':>9} {'Hold':>6} "
              f"{'Trim Gate':<20} {'Trim P&L':>9} {'Hold':>6} {'Diff':>8}")
        for r in regressions[:15]:
            print(f"  {r['ticker']:<7} {r['day']:<12} {r['prod_gate']:<20} ${r['prod_pnl']:>8.0f} "
                  f"{r['prod_hold']:>5.0f}m {r['trim_gate']:<20} ${r['trim_pnl']:>8.0f} "
                  f"{r['trim_hold']:>5.0f}m ${r['diff']:>+7.0f}")

    # ---------------------------------------------------------------------------
    # Report: Hold time distribution
    # ---------------------------------------------------------------------------

    print(f"\n\n{'=' * 80}")
    print("HOLD TIME DISTRIBUTION")
    print(f"{'=' * 80}")

    buckets = [(0, 5, "<5m"), (5, 15, "5-15m"), (15, 30, "15-30m"),
               (30, 60, "30-60m"), (60, 120, "1-2hr"), (120, 240, "2-4hr"),
               (240, 999, "4hr+")]

    for sname in SCENARIOS:
        results = all_results[sname]
        print(f"\n{sname}:")
        for lo, hi, label in buckets:
            trades = [(sd, er, pnl) for sd, er, pnl in results if lo <= er.hold_minutes < hi]
            if trades:
                ct = len(trades)
                total_pnl = sum(pnl for _, _, pnl in trades)
                wins = sum(1 for _, _, pnl in trades if pnl > 0)
                wr = wins / ct * 100
                print(f"  {label:>8}: {ct:>3} trades, ${total_pnl:>8.0f} P&L, "
                      f"{wins}W/{ct-wins}L ({wr:.0f}% WR)")

    # ---------------------------------------------------------------------------
    # Report: Summary recommendation
    # ---------------------------------------------------------------------------

    print(f"\n\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")

    for sname in SCENARIOS:
        results = all_results[sname]
        total_pnl = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = len(results) - wins
        wr = wins / len(results) * 100
        avg_hold = sum(er.hold_minutes for _, er, _ in results) / len(results)
        n_gates = len(SCENARIOS[sname]["gates"])
        prod_pnl = sum(pnl for _, _, pnl in all_results["v4_production"])
        diff = total_pnl - prod_pnl
        diff_str = f"  ({'+' if diff > 0 else ''}{diff:.0f})" if sname != "v4_production" else ""
        print(f"  {sname:<20} ({n_gates:>2} gates): ${total_pnl:>8.0f}  "
              f"{wins}W/{losses}L  {wr:.0f}% WR  avg {avg_hold:.0f}m{diff_str}")

    # Identify which removed gates helped vs hurt
    prod_results_local = all_results["v4_production"]
    print(f"\n  Gates REMOVED in smart_trim scenario:")
    removed = PRODUCTION_GATES - SMART_TRIM_GATES
    for gate in sorted(removed):
        fired = sum(1 for _, er, _ in prod_results_local if er.reason == gate or
                    er.reason.startswith(gate))
        pnl_when_fired = sum(pnl for _, er, pnl in prod_results_local if er.reason == gate or
                             er.reason.startswith(gate))
        if fired > 0:
            print(f"    {gate:<20} fired {fired}x, produced ${pnl_when_fired:>+8.0f} "
                  f"({'harmful' if pnl_when_fired < 0 else 'helpful'})")
        else:
            print(f"    {gate:<20} never fired")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
