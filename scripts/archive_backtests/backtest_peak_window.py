#!/usr/bin/env python3
"""Peak-window exit strategy backtest.

Calibrated for Adam's observation: option peaks happen at 1-2.5 hours.
Tests strategies designed to HOLD until the peak window, then protect gains.

Key innovation: separates "traded" signals (passed entry pipeline) from
"all" signals to match production reality.

Usage:
  python scripts/backtest_peak_window.py [signals_db] [harvester_db]
"""

import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SLIPPAGE_HAIRCUT = 0.15  # 15% on gains


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
# Tick-by-tick simulator with configurable strategy
# ---------------------------------------------------------------------------

def simulate(entry, ticks, signal_ts, contracts, direction, strategy):
    """Simulate exit strategy tick-by-tick.

    strategy dict keys:
      grace_minutes: int — no exits during this period (except hard stop override)
      hard_stop_pct: float — max loss from entry (fires even in grace after grace_stop_delay)
      grace_stop_delay: float — minutes before hard stop can fire during grace
      trail_activation_pct: float — peak gain % to activate trailing stop
      trail_width_pct: float — % drop from peak to trigger exit
      runner_threshold_pct: float — peak gain % to enter runner mode
      runner_width_pct: float — wider trail for runners
      moonshot_threshold_pct: float — peak gain % for moonshot
      moonshot_width_pct: float — tighter trail to lock huge gains
      be_clamp_pct: float — activate break-even clamp at this peak gain %
      soft_trail_min_pct: float — soft trail activates at this gain %
      soft_trail_max_pct: float — soft trail deactivates at this gain %
      soft_trail_floor_pct: float — % of peak gain to protect
      dollar_trail: bool — enable dollar stair-step trail
      dollar_trail_activation_pct: float
      dollar_trail_small_step_pct: float
      dollar_trail_step_threshold_pct: float
      dollar_trail_large_step_pct: float
      theta_bleed_min: float — exit if held X min and down Y%
      theta_bleed_loss_pct: float
      min_hold_minutes: float — absolute minimum hold (no trail exits before this)
      eod_exit: bool — exit at 3:45 PM ET
    """
    if not ticks or entry <= 0:
        return ExitResult(entry, "no_data")

    s = strategy
    grace_minutes = s.get("grace_minutes", 20)
    grace_stop_delay = s.get("grace_stop_delay", 0)
    min_hold = s.get("min_hold_minutes", 0)

    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)

    peak_premium = entry
    be_clamp_active = False

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue

        if price > peak_premium:
            peak_premium = price

        gain_pct = (price - entry) / entry * 100
        peak_gain_pct = (peak_premium - entry) / entry * 100
        drop_from_entry = max(0, (entry - price) / entry * 100)
        drop_from_peak = (peak_premium - price) / peak_premium * 100 if peak_premium > 0 else 0
        elapsed_min = (tick.ts - signal_ts).total_seconds() / 60

        et_hour = (tick.ts.hour - 4) % 24
        et_min = tick.ts.minute

        in_grace = elapsed_min < grace_minutes

        def _exit(reason):
            return ExitResult(price, reason, elapsed_min, peak_gain_pct, gain_pct)

        # === HARD STOP (always active after grace_stop_delay) ===
        if elapsed_min >= grace_stop_delay:
            # Graduated stops: tighten over time
            stop_pct = s.get("hard_stop_pct", 30)
            grad = s.get("graduated_stops")
            if grad:
                for threshold_min, grad_pct in grad:
                    if elapsed_min < threshold_min:
                        stop_pct = grad_pct
                        break
            if drop_from_entry >= stop_pct:
                return _exit("stop_loss")

        # === MINIMUM HOLD ENFORCEMENT ===
        # During min_hold period, only hard stop can fire
        if elapsed_min < min_hold:
            # EOD override
            if s.get("eod_exit", True) and et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")
            continue

        # === BE CLAMP ===
        be_threshold = s.get("be_clamp_pct", 15)
        if be_threshold > 0:
            if peak_gain_pct >= be_threshold:
                be_clamp_active = True
            if be_clamp_active and price <= entry and not in_grace:
                return _exit("be_clamp")

        # === SOFT TRAIL (early protection) ===
        soft_min = s.get("soft_trail_min_pct", 15)
        soft_max = s.get("soft_trail_max_pct", 35)
        soft_floor = s.get("soft_trail_floor_pct", 50)
        if soft_min > 0 and not in_grace:
            if soft_min <= peak_gain_pct < soft_max:
                gain_at_peak = peak_premium - entry
                floor = entry + gain_at_peak * (soft_floor / 100)
                if price <= floor:
                    return _exit("soft_trail")

        # === DOLLAR TRAIL ===
        if s.get("dollar_trail", False) and not in_grace:
            profit_pct = (peak_premium - entry) / entry * 100
            da = s.get("dollar_trail_activation_pct", 40)
            if profit_pct >= da:
                profit_dollars = (peak_premium - entry) * 100
                threshold = s.get("dollar_trail_step_threshold_pct", 25) / 100 * entry * 100
                if profit_dollars >= threshold:
                    step = s.get("dollar_trail_large_step_pct", 10) / 100 * entry * 100
                else:
                    step = s.get("dollar_trail_small_step_pct", 20) / 100 * entry * 100
                if step > 0:
                    steps_hit = int(profit_dollars / step)
                    trail_floor_dollars = steps_hit * step
                    trail_floor_premium = entry + trail_floor_dollars / 100
                    if price <= trail_floor_premium and trail_floor_dollars > 0:
                        return _exit("dollar_trail")

        # === ADAPTIVE TRAILING STOP ===
        if not in_grace:
            act = s.get("trail_activation_pct", 35)
            moon_t = s.get("moonshot_threshold_pct", 400)
            runner_t = s.get("runner_threshold_pct", 150)

            if peak_gain_pct >= moon_t:
                if drop_from_peak >= s.get("moonshot_width_pct", 30):
                    return _exit("adaptive_trail_moonshot")
            elif peak_gain_pct >= runner_t:
                if drop_from_peak >= s.get("runner_width_pct", 45):
                    return _exit("adaptive_trail_runner")
            elif peak_gain_pct >= act:
                if drop_from_peak >= s.get("trail_width_pct", 35):
                    return _exit("adaptive_trail_active")

        # === THETA BLEED ===
        tb_min = s.get("theta_bleed_min", 45)
        tb_loss = s.get("theta_bleed_loss_pct", 30)
        if tb_min > 0:
            if elapsed_min >= tb_min and drop_from_entry >= tb_loss:
                return _exit("theta_bleed")

        # === EOD CUTOFF ===
        if s.get("eod_exit", True):
            if et_hour >= 15 and et_min >= 45:
                return _exit("eod_cutoff")

    # End of data
    last = ticks[-1]
    final_gain = (last.mid - entry) / entry * 100 if entry > 0 else 0
    elapsed = (ticks[-1].ts - signal_ts).total_seconds() / 60
    peak_g = (peak_premium - entry) / entry * 100 if entry > 0 else 0
    return ExitResult(last.mid, "eod_data_end", elapsed, peak_g, final_gain)


# ---------------------------------------------------------------------------
# Strategy definitions calibrated for 1.5-2hr peak window
# ---------------------------------------------------------------------------

STRATEGIES = {
    # Current v4.1 production (baseline)
    "v4.1_prod": {
        "desc": "v4.1 production (decel/no_momentum/time_decay disabled)",
        "grace_minutes": 20,
        "grace_stop_delay": 0,
        "hard_stop_pct": 30,
        "be_clamp_pct": 15,
        "soft_trail_min_pct": 15,
        "soft_trail_max_pct": 35,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 150,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 30,
        "dollar_trail": True,
        "dollar_trail_activation_pct": 40,
        "dollar_trail_small_step_pct": 20,
        "dollar_trail_step_threshold_pct": 25,
        "dollar_trail_large_step_pct": 10,
        "theta_bleed_min": 45,
        "theta_bleed_loss_pct": 30,
        "min_hold_minutes": 0,
        "eod_exit": True,
    },

    # Peak window: 30min grace, delayed trail activation
    "peak_30m": {
        "desc": "30min grace, hold for peak development",
        "grace_minutes": 30,
        "grace_stop_delay": 0,
        "hard_stop_pct": 30,
        "be_clamp_pct": 20,
        "soft_trail_min_pct": 20,
        "soft_trail_max_pct": 40,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 150,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 30,
        "dollar_trail": True,
        "dollar_trail_activation_pct": 50,
        "dollar_trail_small_step_pct": 20,
        "dollar_trail_step_threshold_pct": 25,
        "dollar_trail_large_step_pct": 10,
        "theta_bleed_min": 60,
        "theta_bleed_loss_pct": 30,
        "min_hold_minutes": 0,
        "eod_exit": True,
    },

    # Peak window: 45min min hold, nothing exits before 45min except stop
    "hold_45m": {
        "desc": "45min minimum hold, only hard stop before that",
        "grace_minutes": 45,
        "grace_stop_delay": 0,
        "hard_stop_pct": 30,
        "be_clamp_pct": 20,
        "soft_trail_min_pct": 20,
        "soft_trail_max_pct": 50,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 150,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 30,
        "dollar_trail": True,
        "dollar_trail_activation_pct": 50,
        "dollar_trail_small_step_pct": 20,
        "dollar_trail_step_threshold_pct": 25,
        "dollar_trail_large_step_pct": 10,
        "theta_bleed_min": 90,
        "theta_bleed_loss_pct": 30,
        "min_hold_minutes": 45,
        "eod_exit": True,
    },

    # Peak window: 60min min hold with wider stop
    "hold_60m_wide": {
        "desc": "60min min hold, 35% stop, wider trail",
        "grace_minutes": 60,
        "grace_stop_delay": 20,
        "hard_stop_pct": 35,
        "be_clamp_pct": 25,
        "soft_trail_min_pct": 25,
        "soft_trail_max_pct": 60,
        "soft_trail_floor_pct": 45,
        "trail_activation_pct": 40,
        "trail_width_pct": 40,
        "runner_threshold_pct": 150,
        "runner_width_pct": 50,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 35,
        "dollar_trail": False,
        "theta_bleed_min": 120,
        "theta_bleed_loss_pct": 35,
        "min_hold_minutes": 60,
        "eod_exit": True,
    },

    # Peak window: 90min min hold — force hold until near peak
    "hold_90m": {
        "desc": "90min min hold, then tight trail to capture peak",
        "grace_minutes": 90,
        "grace_stop_delay": 20,
        "hard_stop_pct": 35,
        "be_clamp_pct": 20,
        "soft_trail_min_pct": 15,
        "soft_trail_max_pct": 50,
        "soft_trail_floor_pct": 55,
        "trail_activation_pct": 30,
        "trail_width_pct": 30,
        "runner_threshold_pct": 100,
        "runner_width_pct": 40,
        "moonshot_threshold_pct": 300,
        "moonshot_width_pct": 25,
        "dollar_trail": False,
        "theta_bleed_min": 120,
        "theta_bleed_loss_pct": 35,
        "min_hold_minutes": 90,
        "eod_exit": True,
    },

    # Two-phase: loose first 60min, then tighten
    "two_phase": {
        "desc": "Loose 60min, then tighter trail with soft_trail at 15%+",
        "grace_minutes": 60,
        "grace_stop_delay": 20,
        "hard_stop_pct": 30,
        "be_clamp_pct": 15,
        "soft_trail_min_pct": 15,
        "soft_trail_max_pct": 40,
        "soft_trail_floor_pct": 55,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 100,
        "runner_width_pct": 40,
        "moonshot_threshold_pct": 300,
        "moonshot_width_pct": 30,
        "dollar_trail": True,
        "dollar_trail_activation_pct": 50,
        "dollar_trail_small_step_pct": 25,
        "dollar_trail_step_threshold_pct": 30,
        "dollar_trail_large_step_pct": 15,
        "theta_bleed_min": 90,
        "theta_bleed_loss_pct": 30,
        "min_hold_minutes": 0,
        "eod_exit": True,
    },

    # Conservative peak: 25% stop (tighter loss control) + 60min hold
    "tight_stop_60m": {
        "desc": "25% stop + 60min hold, balance loss control with peak capture",
        "grace_minutes": 60,
        "grace_stop_delay": 0,
        "hard_stop_pct": 25,
        "be_clamp_pct": 15,
        "soft_trail_min_pct": 15,
        "soft_trail_max_pct": 40,
        "soft_trail_floor_pct": 55,
        "trail_activation_pct": 30,
        "trail_width_pct": 30,
        "runner_threshold_pct": 100,
        "runner_width_pct": 40,
        "moonshot_threshold_pct": 300,
        "moonshot_width_pct": 25,
        "dollar_trail": False,
        "theta_bleed_min": 90,
        "theta_bleed_loss_pct": 25,
        "min_hold_minutes": 60,
        "eod_exit": True,
    },

    # Graduated stop: wider early, tighter late — let trades develop
    "grad_stop": {
        "desc": "Graduated stop: 40% early, 30% mid, 25% late + 45m grace",
        "grace_minutes": 45,
        "grace_stop_delay": 0,
        "hard_stop_pct": 40,  # wide early
        "be_clamp_pct": 20,
        "soft_trail_min_pct": 20,
        "soft_trail_max_pct": 50,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 150,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 30,
        "dollar_trail": False,
        "theta_bleed_min": 90,
        "theta_bleed_loss_pct": 30,
        "min_hold_minutes": 0,
        "eod_exit": True,
        # Custom: graduated stops tighten over time
        "graduated_stops": [(45, 40), (90, 30), (999, 25)],
    },

    # v5 candidate: 60min grace, wider stop, no dollar trail (let runners run)
    "v5_candidate": {
        "desc": "v5: 60m grace, 35% stop, adaptive only (no dollar trail)",
        "grace_minutes": 60,
        "grace_stop_delay": 20,
        "hard_stop_pct": 35,
        "be_clamp_pct": 25,
        "soft_trail_min_pct": 25,
        "soft_trail_max_pct": 60,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 40,
        "trail_width_pct": 40,
        "runner_threshold_pct": 150,
        "runner_width_pct": 50,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 35,
        "dollar_trail": False,
        "theta_bleed_min": 120,
        "theta_bleed_loss_pct": 35,
        "min_hold_minutes": 45,
        "eod_exit": True,
    },

    # v5 with dollar trail for safety
    "v5_dollar": {
        "desc": "v5 + dollar trail at 60% activation (higher threshold)",
        "grace_minutes": 60,
        "grace_stop_delay": 20,
        "hard_stop_pct": 35,
        "be_clamp_pct": 25,
        "soft_trail_min_pct": 25,
        "soft_trail_max_pct": 60,
        "soft_trail_floor_pct": 50,
        "trail_activation_pct": 40,
        "trail_width_pct": 40,
        "runner_threshold_pct": 150,
        "runner_width_pct": 50,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 35,
        "dollar_trail": True,
        "dollar_trail_activation_pct": 60,
        "dollar_trail_small_step_pct": 25,
        "dollar_trail_step_threshold_pct": 30,
        "dollar_trail_large_step_pct": 15,
        "theta_bleed_min": 120,
        "theta_bleed_loss_pct": 35,
        "min_hold_minutes": 45,
        "eod_exit": True,
    },

    # Fixed-time 2hr exit (theoretical ceiling)
    "fixed_2hr": {
        "desc": "Exit at exactly 2 hours (theoretical best from data)",
        "grace_minutes": 120,
        "grace_stop_delay": 20,
        "hard_stop_pct": 35,
        "be_clamp_pct": 0,
        "soft_trail_min_pct": 0,
        "soft_trail_max_pct": 0,
        "soft_trail_floor_pct": 0,
        "trail_activation_pct": 999,  # never activates
        "trail_width_pct": 35,
        "runner_threshold_pct": 999,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 999,
        "moonshot_width_pct": 30,
        "dollar_trail": False,
        "theta_bleed_min": 120,
        "theta_bleed_loss_pct": 0.01,  # force exit at 120min
        "min_hold_minutes": 120,
        "eod_exit": True,
    },

    # Minimal + long hold: just stop + adaptive + EOD, 45min grace
    "minimal_45m": {
        "desc": "Only stop + adaptive trail + EOD, 45min grace",
        "grace_minutes": 45,
        "grace_stop_delay": 0,
        "hard_stop_pct": 30,
        "be_clamp_pct": 0,
        "soft_trail_min_pct": 0,
        "soft_trail_max_pct": 0,
        "soft_trail_floor_pct": 0,
        "trail_activation_pct": 35,
        "trail_width_pct": 35,
        "runner_threshold_pct": 150,
        "runner_width_pct": 45,
        "moonshot_threshold_pct": 400,
        "moonshot_width_pct": 30,
        "dollar_trail": False,
        "theta_bleed_min": 0,
        "theta_bleed_loss_pct": 0,
        "min_hold_minutes": 0,
        "eod_exit": True,
    },
}


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

    # Load signals with trade info
    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike, ts.expiry,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts,
               pt.id as trade_id, pt.premium_per_contract as traded_entry,
               pt.exit_premium as traded_exit, pt.pnl_dollars as traded_pnl,
               pt.exit_reason as traded_exit_reason, pt.contracts as traded_contracts,
               pt.mfe_premium as traded_mfe, pt.duration_minutes as traded_duration
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.parent_trade_id IS NULL
        ORDER BY ts.created_at
    """).fetchall()

    print(f"Total signals: {len(signals)}")

    # Build tick data for all signals
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

        # Find time to peak
        peak_idx = max(range(len(ticks)), key=lambda i: ticks[i].mid)
        peak_time_min = (ticks[peak_idx].ts - sig_ts).total_seconds() / 60

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
            "peak_time_min": peak_time_min,
            "was_traded": sig["trade_id"] is not None,
            "actual_pnl": sig["traded_pnl"] or 0,
            "actual_exit_reason": sig["traded_exit_reason"] or "",
            "actual_contracts": sig["traded_contracts"] or contracts,
            "actual_duration": sig["traded_duration"] or 0,
        })

    print(f"Signals with tick data: {len(sim_data)}")
    print(f"  No harvester data: {no_data}, No strike/premium: {no_strike}")
    # All signals are tradeable — previous "untradeable" ones were bugs
    traded_data = sim_data

    # =========================================================================
    # Peak time analysis
    # =========================================================================
    print(f"\n{'='*80}")
    print("TIME-TO-PEAK ANALYSIS (all signals with tick data)")
    print(f"{'='*80}")

    peak_times = [s["peak_time_min"] for s in sim_data if s["peak_gain"] > 5]
    if peak_times:
        peak_times_sorted = sorted(peak_times)
        print(f"  Signals with >5% peak gain: {len(peak_times)}")
        print(f"  Median time to peak: {peak_times_sorted[len(peak_times_sorted)//2]:.0f} min "
              f"({peak_times_sorted[len(peak_times_sorted)//2]/60:.1f} hr)")
        print(f"  Average time to peak: {sum(peak_times)/len(peak_times):.0f} min "
              f"({sum(peak_times)/len(peak_times)/60:.1f} hr)")
        print(f"  25th percentile: {peak_times_sorted[len(peak_times_sorted)//4]:.0f} min")
        print(f"  75th percentile: {peak_times_sorted[3*len(peak_times_sorted)//4]:.0f} min")

        # Distribution
        buckets = [(0, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 300)]
        print(f"\n  Peak time distribution:")
        for lo, hi in buckets:
            ct = sum(1 for t in peak_times if lo <= t < hi)
            pct = ct / len(peak_times) * 100
            bar = "#" * int(pct / 2)
            print(f"    {lo:>3}-{hi:<3}min: {ct:>3} ({pct:>5.1f}%) {bar}")

    # =========================================================================
    # Run all strategies on TRADED signals only
    # =========================================================================
    print(f"\n\n{'='*80}")
    print("STRATEGY COMPARISON — TRADED SIGNALS ONLY (matches production)")
    print(f"{'='*80}")
    print(f"Using {len(traded_data)} signals that passed entry pipeline\n")

    all_results_traded = {}
    for sname, strat in STRATEGIES.items():
        results = []
        for sd in traded_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"],
                          sd["actual_contracts"], sd["direction"], strat)
            pnl = (er.premium - sd["entry"]) * sd["actual_contracts"] * 100
            pnl = _apply_slippage(pnl)
            results.append((sd, er, pnl))
        all_results_traded[sname] = results

    print(f"{'Strategy':<18} {'Desc':<55} {'P&L':>8} {'W/L':>8} {'WR':>6} {'AvgHold':>8} {'PkCapt':>7}")
    print("-" * 120)

    for sname in STRATEGIES:
        results = all_results_traded[sname]
        total_pnl = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = sum(1 for _, _, pnl in results if pnl <= 0)
        wr = wins / len(results) * 100 if results else 0
        avg_hold = sum(er.hold_minutes for _, er, _ in results) / len(results) if results else 0
        # Peak capture: what % of theoretical peak gain did we capture?
        peak_captures = []
        for sd, er, _ in results:
            if sd["peak_gain"] > 5:
                capture = er.exit_gain_pct / sd["peak_gain"] * 100 if sd["peak_gain"] > 0 else 0
                peak_captures.append(capture)
        avg_capture = sum(peak_captures) / len(peak_captures) if peak_captures else 0
        desc = STRATEGIES[sname]["desc"][:53]
        print(f"{sname:<18} {desc:<55} ${total_pnl:>7.0f} {wins}W/{losses}L {wr:>5.1f}% "
              f"{avg_hold:>6.0f}m {avg_capture:>5.0f}%")

    # =========================================================================
    # Daily P&L breakdown for top strategies
    # =========================================================================
    # Find best strategy
    best_strat = max(all_results_traded.keys(),
                     key=lambda s: sum(p for _, _, p in all_results_traded[s]))
    baseline = "v4.1_prod"

    for strat_name in [baseline, best_strat]:
        if strat_name == baseline and strat_name == best_strat:
            continue
        results = all_results_traded[strat_name]

        print(f"\n{'='*100}")
        print(f"DAILY P&L BREAKDOWN: {strat_name} — {STRATEGIES[strat_name]['desc']}")
        print(f"{'='*100}")

        daily_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for sd, er, pnl in results:
            day = sd["day"]
            daily_pnl[day]["pnl"] += pnl
            daily_pnl[day]["trades"] += 1
            if pnl > 0:
                daily_pnl[day]["wins"] += 1

        print(f"\n{'Date':<12} {'Trades':>7} {'Wins':>5} {'P&L':>10} {'Cum P&L':>10}")
        print("-" * 50)
        cum = 0
        for day in sorted(daily_pnl.keys()):
            d = daily_pnl[day]
            cum += d["pnl"]
            marker = " ***" if d["pnl"] > 100 else (" !!!" if d["pnl"] < -200 else "")
            print(f"{day:<12} {d['trades']:>7} {d['wins']:>5} ${d['pnl']:>9.0f} ${cum:>9.0f}{marker}")
        print(f"{'TOTAL':<12} {sum(d['trades'] for d in daily_pnl.values()):>7} "
              f"{sum(d['wins'] for d in daily_pnl.values()):>5} "
              f"${sum(d['pnl'] for d in daily_pnl.values()):>9.0f}")

    # Also show baseline daily
    results = all_results_traded[baseline]
    print(f"\n{'='*100}")
    print(f"DAILY P&L BREAKDOWN: {baseline} — {STRATEGIES[baseline]['desc']}")
    print(f"{'='*100}")

    daily_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for sd, er, pnl in results:
        day = sd["day"]
        daily_pnl[day]["pnl"] += pnl
        daily_pnl[day]["trades"] += 1
        if pnl > 0:
            daily_pnl[day]["wins"] += 1

    print(f"\n{'Date':<12} {'Trades':>7} {'Wins':>5} {'P&L':>10} {'Cum P&L':>10}")
    print("-" * 50)
    cum = 0
    for day in sorted(daily_pnl.keys()):
        d = daily_pnl[day]
        cum += d["pnl"]
        marker = " ***" if d["pnl"] > 100 else (" !!!" if d["pnl"] < -200 else "")
        print(f"{day:<12} {d['trades']:>7} {d['wins']:>5} ${d['pnl']:>9.0f} ${cum:>9.0f}{marker}")
    print(f"{'TOTAL':<12} {sum(d['trades'] for d in daily_pnl.values()):>7} "
          f"{sum(d['wins'] for d in daily_pnl.values()):>5} "
          f"${sum(d['pnl'] for d in daily_pnl.values()):>9.0f}")

    # =========================================================================
    # Per-trade detail for best strategy vs baseline
    # =========================================================================
    print(f"\n\n{'='*140}")
    print(f"PER-TRADE: {baseline} vs {best_strat} (traded signals only)")
    print(f"{'='*140}")

    base_results = all_results_traded[baseline]
    best_results = all_results_traded[best_strat]

    print(f"\n{'#':<3} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'$Entry':>6} {'Peak%':>6} {'PkMin':>6} "
          f"| {'Base Gate':<22} {'P&L':>8} {'Hold':>6} "
          f"| {'Best Gate':<22} {'P&L':>8} {'Hold':>6} {'Diff':>8}")
    print("-" * 140)

    total_diff = 0
    for i in range(len(traded_data)):
        sd = traded_data[i]
        _, er_b, pnl_b = base_results[i]
        _, er_best, pnl_best = best_results[i]
        diff = pnl_best - pnl_b

        marker = ""
        if diff > 50:
            marker = " <<<"
        elif diff < -50:
            marker = " >>>"
        total_diff += diff

        print(f"{i+1:<3} {sd['ticker']:<7} {sd['direction'][:4]:<5} {sd['day']:<12} "
              f"${sd['entry']:>5.2f} {sd['peak_gain']:>+5.0f}% {sd['peak_time_min']:>5.0f}m "
              f"| {er_b.reason:<22} ${pnl_b:>7.0f} {er_b.hold_minutes:>5.0f}m "
              f"| {er_best.reason:<22} ${pnl_best:>7.0f} {er_best.hold_minutes:>5.0f}m "
              f"${diff:>+7.0f}{marker}")

    print(f"\nNet difference: ${total_diff:>+.0f}")

    # =========================================================================
    # Gate fire analysis for best strategy
    # =========================================================================
    for sname in [baseline, best_strat]:
        results = all_results_traded[sname]
        print(f"\n{'='*80}")
        print(f"GATE FIRE ANALYSIS: {sname}")
        print(f"{'='*80}")

        gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0,
                                          "hold_mins": [], "captures": []})
        for sd, er, pnl in results:
            reason = er.reason
            if reason.startswith("adaptive_trail_"):
                reason = "adaptive_trail"
            g = gate_stats[reason]
            g["count"] += 1
            g["pnl"] += pnl
            if pnl > 0:
                g["wins"] += 1
            g["hold_mins"].append(er.hold_minutes)
            if sd["peak_gain"] > 5:
                cap = er.exit_gain_pct / sd["peak_gain"] * 100 if sd["peak_gain"] > 0 else 0
                g["captures"].append(cap)

        sorted_gates = sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        total = len(results)

        print(f"\n{'Gate':<25} {'Fires':>6} {'%':>6} {'P&L':>9} {'W/L':>8} {'WR':>6} "
              f"{'AvgHold':>8} {'PkCapt':>7}")
        print("-" * 80)

        for gate, stats in sorted_gates:
            ct = stats["count"]
            pct = ct / total * 100
            pnl_sum = stats["pnl"]
            wins = stats["wins"]
            losses = ct - wins
            wr = wins / ct * 100 if ct > 0 else 0
            avg_hold = sum(stats["hold_mins"]) / ct if ct > 0 else 0
            avg_cap = sum(stats["captures"]) / len(stats["captures"]) if stats["captures"] else 0
            print(f"{gate:<25} {ct:>6} {pct:>5.1f}% ${pnl_sum:>8.0f} "
                  f"{wins}W/{losses}L {wr:>5.1f}% {avg_hold:>6.0f}m {avg_cap:>+6.0f}%")

    # =========================================================================
    # Actual production P&L comparison
    # =========================================================================
    print(f"\n\n{'='*80}")
    print("ACTUAL PRODUCTION P&L vs BACKTEST (traded signals)")
    print(f"{'='*80}")

    actual_total = sum(sd["actual_pnl"] for sd in traded_data)
    base_total = sum(pnl for _, _, pnl in all_results_traded[baseline])
    best_total = sum(pnl for _, _, pnl in all_results_traded[best_strat])

    print(f"\n  Actual production P&L:   ${actual_total:>+,.0f}")
    print(f"  v4.1 backtest P&L:       ${base_total:>+,.0f}")
    print(f"  {best_strat} backtest P&L: ${best_total:>+,.0f}")
    print(f"  Improvement over prod:   ${best_total - actual_total:>+,.0f}")

    # =========================================================================
    # Actual production comparison (for signals that were actually traded)
    # =========================================================================
    actually_traded = [i for i, sd in enumerate(traded_data) if sd["was_traded"]]
    if actually_traded:
        actual_pnl = sum(traded_data[i]["actual_pnl"] for i in actually_traded)
        best_traded_pnl = sum(all_results_traded[best_strat][i][2] for i in actually_traded)
        base_traded_pnl = sum(all_results_traded[baseline][i][2] for i in actually_traded)
        print(f"\n\n{'='*80}")
        print(f"COMPARISON ON {len(actually_traded)} ACTUALLY-TRADED SIGNALS")
        print(f"{'='*80}")
        print(f"  Actual production P&L: ${actual_pnl:>+,.0f}")
        print(f"  v4.1 backtest P&L:     ${base_traded_pnl:>+,.0f}")
        print(f"  {best_strat} P&L:  ${best_traded_pnl:>+,.0f}")


if __name__ == "__main__":
    main()
