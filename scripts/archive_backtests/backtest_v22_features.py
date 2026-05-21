#!/usr/bin/env python3
"""Backtest v2.2 features INDIVIDUALLY against all signals with harvester tick data.

Tests each feature in isolation and in combination to determine what helps vs hurts.

Strategies simulated:
  1. baseline     — v4 production: hard stop -30% (20min grace) + adaptive trail
  2. +be_clamp    — baseline + BE clamp (floor=entry once peak reaches +15%)
  3. +soft_trail  — baseline + soft trail in 15-35% band (floor=entry+50% of peak gain)
  4. +short_grace — baseline with 5min grace (v2.2) instead of 20min
  5. +time_exits  — baseline + theta bleed/no momentum/time decay (gates that preempt trail)
  6. reordered    — +time_exits but trails fire BEFORE time gates (Vince's reorder)
  7. phase1       — short grace + be_clamp + soft_trail (current deployed)
  8. full_v22     — phase1 + tighter trail tiers + milestone locks

Usage:
  python scripts/backtest_v22_features.py
"""

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SLIPPAGE_HAIRCUT = 0.15  # 15% on gains

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

# Shared
HARD_STOP_PCT = 30.0

# Adaptive trail (v4 production)
ADAPTIVE_ACTIVATION = 35.0
ACTIVE_WIDTH = 35.0
RUNNER_THRESHOLD = 150.0
RUNNER_WIDTH = 45.0
MOONSHOT_THRESHOLD = 400.0
MOONSHOT_WIDTH = 30.0

# BE clamp
BE_CLAMP_ACTIVATION_PCT = 15.0

# Soft trail
SOFT_TRAIL_MIN = 15.0
SOFT_TRAIL_MAX = 35.0
SOFT_TRAIL_FLOOR_PCT = 50.0

# v2.2 tighter trail tiers (narrower giveback at higher gains)
V22_TRAIL_TIERS = [
    (200.0, 0.20),  # 200%+: giveback 20%
    (100.0, 0.25),  # 100-200%: giveback 25%
    (35.0, 0.35),   # 35-100%: giveback 35%
]

# Milestone locks
MILESTONE_LEVELS = [200, 400, 600]
MILESTONE_BANK_PCT = 0.15

# Time-based exit parameters
THETA_BLEED_HOLD_MINUTES = 45.0
THETA_BLEED_MAX_LOSS_PCT = 30.0
NO_MOMENTUM_MINUTES = 45.0
TIME_DECAY_STALE_MINUTES = 10.0
TIME_DECAY_AFTERNOON_HOUR = 15
TIME_DECAY_AFTERNOON_MINUTE = 30


@dataclass
class Tick:
    ts: datetime
    mid: float
    bid: float
    ask: float


@dataclass
class ExitResult:
    premium: float
    reason: str
    banked_pnl: float = 0.0  # from milestone locks


def _apply_slippage(pnl: float) -> float:
    return pnl * (1 - SLIPPAGE_HAIRCUT) if pnl > 0 else pnl


# ---------------------------------------------------------------------------
# Individual exit checks — composable building blocks
# ---------------------------------------------------------------------------

def check_hard_stop(price, entry, drop_from_entry):
    """Returns True if hard stop triggered."""
    return drop_from_entry >= HARD_STOP_PCT


def check_adaptive_trail(price, entry, peak, peak_gain, drop_from_peak):
    """Returns (triggered, reason) for v4 adaptive trail."""
    if peak_gain >= MOONSHOT_THRESHOLD:
        if drop_from_peak >= MOONSHOT_WIDTH:
            return True, "adaptive_moonshot"
    elif peak_gain >= RUNNER_THRESHOLD:
        if drop_from_peak >= RUNNER_WIDTH:
            return True, "adaptive_runner"
    elif peak_gain >= ADAPTIVE_ACTIVATION:
        if drop_from_peak >= ACTIVE_WIDTH:
            return True, "adaptive_active"
    return False, ""


def check_v22_trail(price, entry, peak, peak_gain, drop_from_peak):
    """Returns (triggered, reason) for v2.2 tighter trail tiers."""
    if peak_gain < 35.0:
        return False, ""
    giveback = 0.35
    tier = "35+"
    for threshold, gb in V22_TRAIL_TIERS:
        if peak_gain >= threshold:
            giveback = gb
            tier = f"{threshold:.0f}+"
            break
    trail_floor = peak * (1 - giveback)
    if price <= trail_floor:
        return True, f"v22_trail_{tier}"
    return False, ""


def check_be_clamp(price, entry, peak_gain, be_clamp_active):
    """Returns (triggered, new_be_clamp_active)."""
    if peak_gain >= BE_CLAMP_ACTIVATION_PCT:
        be_clamp_active = True
    if be_clamp_active and price <= entry:
        return True, be_clamp_active
    return False, be_clamp_active


def check_soft_trail(price, entry, peak, peak_gain):
    """Returns True if soft trail triggered."""
    if peak_gain < SOFT_TRAIL_MIN or peak_gain >= SOFT_TRAIL_MAX:
        return False
    gain_at_peak = peak - entry
    floor = entry + gain_at_peak * (SOFT_TRAIL_FLOOR_PCT / 100)
    return price <= floor


def check_theta_bleed(elapsed_minutes, drop_from_entry):
    """Time-based: held 45min+ and down 30%+."""
    return elapsed_minutes >= THETA_BLEED_HOLD_MINUTES and drop_from_entry >= THETA_BLEED_MAX_LOSS_PCT


def check_no_momentum(elapsed_minutes, gain_pct):
    """Time-based: no gain after 45 minutes."""
    return elapsed_minutes >= NO_MOMENTUM_MINUTES and gain_pct <= 0


def check_time_decay(tick_ts, last_new_high_ts, elapsed_minutes):
    """Time-based: no new high in 10min (after 3:30 PM or 45min hold)."""
    # Convert to ET approximation (UTC-4 for EDT)
    et_hour = (tick_ts.hour - 4) % 24
    afternoon = (et_hour > TIME_DECAY_AFTERNOON_HOUR or
                 (et_hour == TIME_DECAY_AFTERNOON_HOUR and tick_ts.minute >= TIME_DECAY_AFTERNOON_MINUTE))
    if not afternoon and elapsed_minutes < 45:
        return False
    if last_new_high_ts is None:
        return False
    stale = (tick_ts - last_new_high_ts).total_seconds() / 60
    return stale >= TIME_DECAY_STALE_MINUTES


def check_milestones(peak_gain, entry, contracts, milestones_hit):
    """Check milestone locks, return (banked_pnl_delta, remaining_contracts, updated_set)."""
    banked = 0.0
    remaining = contracts
    for level in MILESTONE_LEVELS:
        if peak_gain >= level and level not in milestones_hit:
            lock_qty = max(1, int(remaining * MILESTONE_BANK_PCT))
            if remaining - lock_qty >= 1:
                milestones_hit.add(level)
                lock_premium = entry * (1 + level / 100)
                lock_pnl = (lock_premium - entry) * lock_qty * 100
                if lock_pnl > 0:
                    lock_pnl *= (1 - SLIPPAGE_HAIRCUT)
                banked += lock_pnl
                remaining -= lock_qty
    return banked, remaining, milestones_hit


# ---------------------------------------------------------------------------
# Strategy simulators
# ---------------------------------------------------------------------------

def simulate(entry, ticks, signal_ts, contracts, features):
    """Generic tick-by-tick simulator with composable features.

    features dict keys:
      grace_minutes: int (20 for v4, 5 for v2.2)
      be_clamp: bool
      soft_trail: bool
      v22_trail: bool (use tighter trail tiers instead of v4 adaptive)
      time_exits: bool (theta bleed, no momentum, time decay)
      time_exits_before_trail: bool (old ordering: time gates preempt trail)
      milestones: bool
    """
    if not ticks or entry <= 0:
        return ExitResult(entry, "no_data")

    grace = features.get("grace_minutes", 20)
    use_be_clamp = features.get("be_clamp", False)
    use_soft_trail = features.get("soft_trail", False)
    use_v22_trail = features.get("v22_trail", False)
    use_time_exits = features.get("time_exits", False)
    time_before_trail = features.get("time_exits_before_trail", True)
    use_milestones = features.get("milestones", False)

    peak = entry
    be_clamp_active = False
    last_new_high_ts = signal_ts
    milestones_hit = set()
    total_banked = 0.0
    remaining_contracts = contracts

    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)
    grace_end = signal_ts + timedelta(minutes=grace)

    for tick in ticks:
        price = tick.mid
        if price <= 0:
            continue

        # Track peak and new high
        if price > peak:
            peak = price
            last_new_high_ts = tick.ts

        gain_pct = (price - entry) / entry * 100
        peak_gain = (peak - entry) / entry * 100
        drop_from_peak = (peak - price) / peak * 100 if peak > 0 else 0
        drop_from_entry = (entry - price) / entry * 100 if price < entry else 0
        elapsed = (tick.ts - signal_ts).total_seconds() / 60

        # --- Grace period: skip exits ---
        if tick.ts < grace_end:
            continue

        # --- Hard stop (always first) ---
        if check_hard_stop(price, entry, drop_from_entry):
            return ExitResult(price, "stop_hit", total_banked)

        # --- BE clamp (early defensive) ---
        if use_be_clamp:
            triggered, be_clamp_active = check_be_clamp(price, entry, peak_gain, be_clamp_active)
            if triggered:
                return ExitResult(entry, "be_clamp", total_banked)

        # --- Milestones ---
        if use_milestones:
            banked, remaining_contracts, milestones_hit = check_milestones(
                peak_gain, entry, remaining_contracts, milestones_hit)
            total_banked += banked

        # --- Time exits BEFORE trail (old v4 ordering) ---
        if use_time_exits and time_before_trail:
            if check_theta_bleed(elapsed, drop_from_entry):
                return ExitResult(price, "theta_bleed", total_banked)
            if check_no_momentum(elapsed, gain_pct):
                return ExitResult(price, "no_momentum", total_banked)
            if check_time_decay(tick.ts, last_new_high_ts, elapsed):
                return ExitResult(price, "time_decay", total_banked)

        # --- Soft trail (15-35% band) ---
        if use_soft_trail:
            if check_soft_trail(price, entry, peak, peak_gain):
                return ExitResult(price, "soft_trail", total_banked)

        # --- Trail (v4 adaptive or v2.2 tighter tiers) ---
        if use_v22_trail:
            triggered, reason = check_v22_trail(price, entry, peak, peak_gain, drop_from_peak)
        else:
            triggered, reason = check_adaptive_trail(price, entry, peak, peak_gain, drop_from_peak)
        if triggered:
            return ExitResult(price, reason, total_banked)

        # --- Time exits AFTER trail (v2.2 reorder) ---
        if use_time_exits and not time_before_trail:
            if check_theta_bleed(elapsed, drop_from_entry):
                return ExitResult(price, "theta_bleed", total_banked)
            if check_no_momentum(elapsed, gain_pct):
                return ExitResult(price, "no_momentum", total_banked)
            if check_time_decay(tick.ts, last_new_high_ts, elapsed):
                return ExitResult(price, "time_decay", total_banked)

    # EOD
    return ExitResult(ticks[-1].mid, "eod_cutoff", total_banked)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = {
    "baseline": {
        "desc": "v4 production: hard stop + adaptive trail (20min grace)",
        "grace_minutes": 20,
    },
    "+be_clamp": {
        "desc": "baseline + BE clamp (+15% peak → floor=entry)",
        "grace_minutes": 20,
        "be_clamp": True,
    },
    "+soft_trail": {
        "desc": "baseline + soft trail (15-35% band, 50% floor)",
        "grace_minutes": 20,
        "soft_trail": True,
    },
    "+short_grace": {
        "desc": "baseline with 5min grace instead of 20min",
        "grace_minutes": 5,
    },
    "+time_exits": {
        "desc": "baseline + theta/momentum/decay (old order: before trail)",
        "grace_minutes": 20,
        "time_exits": True,
        "time_exits_before_trail": True,
    },
    "reordered": {
        "desc": "+time_exits but trails fire FIRST (Vince reorder)",
        "grace_minutes": 20,
        "time_exits": True,
        "time_exits_before_trail": False,
    },
    "phase1": {
        "desc": "v2.2 Phase1: short grace + BE clamp + soft trail",
        "grace_minutes": 5,
        "be_clamp": True,
        "soft_trail": True,
    },
    "full_v22": {
        "desc": "full v2.2: phase1 + tighter trails + milestones",
        "grace_minutes": 5,
        "be_clamp": True,
        "soft_trail": True,
        "v22_trail": True,
        "milestones": True,
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
        SELECT captured_at, midpoint, bid, ask
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
            if mid > 0 or bid > 0:
                ticks.append(Tick(ts=ts, mid=mid if mid > 0 else (bid + ask) / 2, bid=bid, ask=ask))
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

    # Load tick data for each signal
    sim_data = []
    no_data = 0
    no_strike = 0

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

        # Entry price: first tick ask (what we'd pay)
        first_tick = ticks[0]
        entry = first_tick.ask if first_tick.ask > 0 else first_tick.mid
        if entry <= 0:
            entry = premium

        # Simulated contracts (score-based sizing with $8K portfolio)
        contracts = 1
        if score >= 95:
            contracts = 5
        elif score >= 90:
            contracts = 4
        elif score >= 85:
            contracts = 3
        elif score >= 78:
            contracts = 1

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        peak = max(t.mid for t in ticks)
        peak_gain = (peak - entry) / entry * 100 if entry > 0 else 0

        sim_data.append({
            "id": sig["id"],
            "ticker": ticker,
            "direction": direction,
            "day": day,
            "score": score,
            "entry": entry,
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
    print(f"No harvester data: {no_data}")
    print(f"No strike/premium: {no_strike}")
    print()

    # ---------------------------------------------------------------------------
    # Run all strategies
    # ---------------------------------------------------------------------------

    strat_names = list(STRATEGIES.keys())
    # results[strat_name] = list of (signal_dict, ExitResult, pnl)
    all_results = {}

    for sname in strat_names:
        features = STRATEGIES[sname]
        results = []
        for sd in sim_data:
            er = simulate(sd["entry"], sd["ticks"], sd["sig_ts"], sd["contracts"], features)
            pnl = (er.premium - sd["entry"]) * sd["contracts"] * 100
            pnl = _apply_slippage(pnl)
            pnl += er.banked_pnl
            results.append((sd, er, pnl))
        all_results[sname] = results

    # ---------------------------------------------------------------------------
    # Per-signal detail table (baseline vs phase1 vs full_v22)
    # ---------------------------------------------------------------------------
    detail_strats = ["baseline", "phase1", "full_v22"]
    print("=" * 160)
    print(f"{'PER-SIGNAL DETAIL: baseline vs phase1 vs full_v22':^160}")
    print("=" * 160)
    hdr = (f"{'ID':>3} {'Tckr':<5} {'Dir':<5} {'Day':<11} {'Sc':>3} {'Trd':>3} "
           f"{'Entry':>6} {'Peak%':>7} "
           f"{'basePnL':>9} {'baseRsn':<20} "
           f"{'ph1PnL':>9} {'ph1Rsn':<20} "
           f"{'v22PnL':>9} {'v22Rsn':<20}")
    print(hdr)
    print("-" * 160)

    for i, sd in enumerate(sim_data):
        base_sd, base_er, base_pnl = all_results["baseline"][i]
        ph1_sd, ph1_er, ph1_pnl = all_results["phase1"][i]
        v22_sd, v22_er, v22_pnl = all_results["full_v22"][i]
        trd = "YES" if sd["was_traded"] else "no"
        print(f"{sd['id']:>3} {sd['ticker']:<5} {sd['direction'][:4]:<5} {sd['day']:<11} "
              f"{sd['score']:>3} {trd:>3} "
              f"${sd['entry']:>5.2f} {sd['peak_gain']:>6.1f}% "
              f"${base_pnl:>8.2f} {base_er.reason:<20} "
              f"${ph1_pnl:>8.2f} {ph1_er.reason:<20} "
              f"${v22_pnl:>8.2f} {v22_er.reason:<20}")

    # ---------------------------------------------------------------------------
    # Strategy comparison summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 130}")
    print(f"{'STRATEGY COMPARISON — ALL SIGNALS':^130}")
    print(f"{'=' * 130}")
    print(f"{'Strategy':<16} {'Desc':<55} {'PnL':>10} {'Wins':>5} {'Loss':>5} "
          f"{'WR':>6} {'AvgW':>8} {'AvgL':>8} {'vs base':>10}")
    print("-" * 130)

    base_total = sum(pnl for _, _, pnl in all_results["baseline"])
    for sname in strat_names:
        results = all_results[sname]
        total = sum(pnl for _, _, pnl in results)
        wins = sum(1 for _, _, pnl in results if pnl > 0)
        losses = sum(1 for _, _, pnl in results if pnl <= 0)
        wr = wins / len(results) * 100 if results else 0
        win_pnls = [pnl for _, _, pnl in results if pnl > 0]
        loss_pnls = [pnl for _, _, pnl in results if pnl <= 0]
        avg_w = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_l = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        diff = total - base_total
        diff_str = f"+${diff:.2f}" if diff > 0 else f"-${abs(diff):.2f}" if diff < 0 else "$0.00"
        desc = STRATEGIES[sname]["desc"][:53]
        print(f"{sname:<16} {desc:<55} ${total:>9.2f} {wins:>5} {losses:>5} "
              f"{wr:>5.0f}% ${avg_w:>7.2f} ${avg_l:>7.2f} {diff_str:>10}")

    # ---------------------------------------------------------------------------
    # Daily breakdown (baseline vs phase1)
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 110}")
    print(f"{'DAY-BY-DAY: baseline vs phase1 vs full_v22':^110}")
    print(f"{'=' * 110}")
    print(f"{'Date':<12} {'Sigs':>5} {'base PnL':>12} {'ph1 PnL':>12} {'v22 PnL':>12} "
          f"{'ph1 diff':>12} {'v22 diff':>12}")
    print("-" * 110)

    days = sorted(set(sd["day"] for sd in sim_data))
    for day in days:
        day_idxs = [i for i, sd in enumerate(sim_data) if sd["day"] == day]
        base_day = sum(all_results["baseline"][i][2] for i in day_idxs)
        ph1_day = sum(all_results["phase1"][i][2] for i in day_idxs)
        v22_day = sum(all_results["full_v22"][i][2] for i in day_idxs)
        ph1_d = ph1_day - base_day
        v22_d = v22_day - base_day
        print(f"{day:<12} {len(day_idxs):>5} ${base_day:>10.2f} ${ph1_day:>10.2f} ${v22_day:>10.2f} "
              f"{'+'if ph1_d>=0 else ''}${ph1_d:>10.2f} {'+'if v22_d>=0 else ''}${v22_d:>10.2f}")

    # ---------------------------------------------------------------------------
    # Feature isolation: what each feature adds/removes vs baseline
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print(f"{'FEATURE ISOLATION: delta vs baseline':^90}")
    print(f"{'=' * 90}")

    feature_strats = ["+be_clamp", "+soft_trail", "+short_grace", "+time_exits", "reordered"]
    for sname in feature_strats:
        results = all_results[sname]
        total = sum(pnl for _, _, pnl in results)
        diff = total - base_total

        # Find signals where this strategy differs from baseline
        better = []
        worse = []
        for i, (sd, er, pnl) in enumerate(results):
            _, base_er, base_pnl = all_results["baseline"][i]
            d = pnl - base_pnl
            if abs(d) > 0.50:  # only count meaningful differences
                if d > 0:
                    better.append((sd, d, er.reason, base_er.reason))
                else:
                    worse.append((sd, d, er.reason, base_er.reason))

        print(f"\n  {sname}: {'+'if diff>=0 else ''}${diff:.2f} vs baseline")
        print(f"    {STRATEGIES[sname]['desc']}")
        print(f"    Improved {len(better)} signals, worsened {len(worse)} signals")

        if better:
            better.sort(key=lambda x: x[1], reverse=True)
            print(f"    TOP IMPROVEMENTS:")
            for sd, d, new_rsn, old_rsn in better[:5]:
                print(f"      #{sd['id']} {sd['ticker']} {sd['day']}: +${d:.2f} "
                      f"({old_rsn} → {new_rsn})")

        if worse:
            worse.sort(key=lambda x: x[1])
            print(f"    TOP REGRESSIONS:")
            for sd, d, new_rsn, old_rsn in worse[:5]:
                print(f"      #{sd['id']} {sd['ticker']} {sd['day']}: ${d:.2f} "
                      f"({old_rsn} → {new_rsn})")

    # ---------------------------------------------------------------------------
    # Exit reason breakdown per strategy
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"{'EXIT REASON BREAKDOWN':^100}")
    print(f"{'=' * 100}")

    for sname in ["baseline", "phase1", "full_v22"]:
        print(f"\n  {sname}:")
        reasons = {}
        for sd, er, pnl in all_results[sname]:
            r = er.reason
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0.0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += pnl
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            avg = d["pnl"] / d["count"] if d["count"] > 0 else 0
            print(f"    {r:<25} {d['count']:>3}x  total ${d['pnl']:>10.2f}  avg ${avg:>8.2f}")

    # ---------------------------------------------------------------------------
    # Traded vs skipped signals
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"{'TRADED vs SKIPPED SIGNALS':^100}")
    print(f"{'=' * 100}")

    for label, filt in [("TRADED", True), ("SKIPPED", False)]:
        idxs = [i for i, sd in enumerate(sim_data) if sd["was_traded"] == filt]
        if not idxs:
            continue
        print(f"\n  {label} ({len(idxs)} signals):")
        print(f"    {'Strategy':<16} {'PnL':>10} {'Wins':>5} {'WR':>6}")
        for sname in ["baseline", "phase1", "full_v22"]:
            total = sum(all_results[sname][i][2] for i in idxs)
            wins = sum(1 for i in idxs if all_results[sname][i][2] > 0)
            wr = wins / len(idxs) * 100
            print(f"    {sname:<16} ${total:>9.2f} {wins:>5} {wr:>5.0f}%")

    # ---------------------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"{'VERDICT: WHAT TO SHIP':^80}")
    print(f"{'=' * 80}")

    verdicts = []
    for sname in feature_strats + ["phase1", "full_v22"]:
        total = sum(pnl for _, _, pnl in all_results[sname])
        diff = total - base_total
        wins = sum(1 for _, _, pnl in all_results[sname] if pnl > 0)
        base_wins = sum(1 for _, _, pnl in all_results["baseline"] if pnl > 0)
        # Count regressions
        regressions = sum(1 for i in range(len(sim_data))
                          if all_results[sname][i][2] < all_results["baseline"][i][2] - 0.50)
        improvements = sum(1 for i in range(len(sim_data))
                           if all_results[sname][i][2] > all_results["baseline"][i][2] + 0.50)
        verdicts.append((sname, diff, wins - base_wins, improvements, regressions))

    print(f"\n  {'Feature':<16} {'$ Impact':>10} {'Win chg':>8} {'Improved':>9} {'Regressed':>10} {'Recommendation'}")
    print(f"  {'-'*85}")
    for sname, diff, win_chg, impr, regr in verdicts:
        if diff > 50 and regr <= impr:
            rec = "SHIP"
        elif diff > 0 and regr <= impr:
            rec = "SHIP (marginal)"
        elif diff >= -20 and regr <= 2:
            rec = "NEUTRAL (safe)"
        elif diff < -50:
            rec = "DO NOT SHIP"
        else:
            rec = "PAPER-TRADE FIRST"
        diff_str = f"+${diff:.0f}" if diff >= 0 else f"-${abs(diff):.0f}"
        win_str = f"+{win_chg}" if win_chg >= 0 else f"{win_chg}"
        print(f"  {sname:<16} {diff_str:>10} {win_str:>8} {impr:>9} {regr:>10} {rec}")

    print()
    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
