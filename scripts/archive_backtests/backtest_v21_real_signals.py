"""Backtest v2.1 remaining features against REAL Discord signals + harvester data.

Replays 103 real signals through each feature combination using actual option
premium snapshots from the harvester. This is the definitive test — same signals
the bot would have traded, with real market data.

Features tested:
  1. Three-tranche scale-out (§4)
  2. Underlying-anchored trail (§5)
  3. Volume-peak modifier (§6)
  4. Early Negative Thesis Revalidation (ENRG spec)

Usage:
    python scripts/backtest_v21_real_signals.py
    python scripts/backtest_v21_real_signals.py --feature tranches
"""

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SIGNALS_DB = os.environ.get("SIGNALS_DB", "journal/owlet-kody/raw_messages.db")
HARVESTER_DB = os.environ.get("HARVESTER_DB", "journal/owlet-harvester/options_data.db")
BALANCE = 5000.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_contract_ticker(underlying, expiry_date, strike, option_type):
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type == "call" else "P"
    strike_int = int(strike * 1000)
    return f"O:{underlying}{date_str}{opt_char}{strike_int:08d}"


def resolve_expiry(signal_time_str):
    dt = datetime.fromisoformat(signal_time_str)
    return dt.strftime("%Y-%m-%d")


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    rows = conn.execute("""
        SELECT ticker, direction, strike, expiry, atm_premium, otm_premium,
               score, created_at
        FROM trade_signals ORDER BY created_at
    """).fetchall()
    conn.close()
    return rows


def get_snapshots(conn, contract_ticker, after_time):
    rows = conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
        FROM harvest_snapshots
        WHERE contract_ticker = ?
          AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_time)).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Tranche logic (§4)
# ---------------------------------------------------------------------------

def plan_tranches(qty):
    if qty <= 1:
        return [("T1", qty, "trailing")]
    if qty == 2:
        return [("T1", 1, "lock_at_25"), ("T2", 1, "trailing")]
    n1 = qty // 3
    n2 = qty // 3
    n3 = qty - n1 - n2
    return [
        ("T1", n1, "lock_at_25"),
        ("T2", n2, "trailing"),
        ("T3", n3, "runner"),
    ]


# §5 underlying trail tiers
UNDERLYING_TRAIL_TIERS = [
    (100.0, 0.0050),
    (50.0, 0.0040),
    (15.0, 0.0030),
    (0.0, 0.0020),
]


def underlying_trail_pct_for_gain(premium_gain_pct):
    for min_gain, trail_pct in UNDERLYING_TRAIL_TIERS:
        if premium_gain_pct >= min_gain:
            return trail_pct
    return UNDERLYING_TRAIL_TIERS[-1][1]


# §6 volume peak (using underlying price from snapshots)
def volume_peak_from_underlying(underlying_prices, direction):
    """Simplified volume peak using underlying price momentum from snapshots.

    Since harvester snapshots include underlying_price but not volume,
    we detect divergence: option premium rising but underlying stalling.
    """
    if len(underlying_prices) < 6:
        return None
    recent = underlying_prices[-6:]
    first_half = recent[:3]
    second_half = recent[3:]

    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)

    if direction in ("call", "bullish", "long"):
        # Underlying stalling or reversing while option was rising → exhaustion
        if second_avg < first_avg * 0.999:
            return "tighten"
    else:
        if second_avg > first_avg * 1.001:
            return "tighten"
    return None


# ENRG simplified
def early_neg_reval(direction, gain_pct, underlying_prices):
    """Simplified early negative thesis check."""
    if gain_pct >= 0 or len(underlying_prices) < 3:
        return "PROCEED"

    recent = underlying_prices[-3:]
    change = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0

    if direction in ("call", "bullish", "long"):
        if change < -0.003:
            return "IMMEDIATE_EXIT"
        if change > -0.001:
            return "HOLD"
    else:
        if change > 0.003:
            return "IMMEDIATE_EXIT"
        if change < 0.001:
            return "HOLD"
    return "PROCEED"


# Runner trail tiers (§4.4)
RUNNER_TRAIL_TIERS = [
    (400.0, 20.0),
    (200.0, 30.0),
    (100.0, 35.0),
    (50.0, 40.0),
]


def get_runner_trail_width(gain_pct):
    for min_gain, width in RUNNER_TRAIL_TIERS:
        if gain_pct >= min_gain:
            return width
    return 40.0


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    ticker: str
    direction: str
    strike: float
    signal_time: str
    score: int
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    mfe_pct: float
    mfe_gap: float
    exit_reason: str
    duration_min: float
    contracts: int
    pnl_dollars: float


def simulate_signal(snapshots, entry_premium, signal_time, ticker, direction, score,
                    enable_tranches=False, enable_underlying_trail=False,
                    enable_volume_peak=False, enable_early_neg_reval=False):
    """Simulate a single real signal through the exit pipeline."""
    if not snapshots or entry_premium <= 0:
        return None

    # Score-tiered sizing
    if score >= 95:
        contracts = 5
    elif score >= 90:
        contracts = 3
    elif score >= 85:
        contracts = 2
    elif score >= 78:
        contracts = 1
    else:
        return None

    cost = entry_premium * 100
    max_by_budget = max(1, int((BALANCE * 0.75 / 5) / cost)) if cost > 0 else 1
    contracts = min(contracts, max_by_budget, 20)

    # Set up tranches
    if enable_tranches and contracts >= 2:
        tranches = plan_tranches(contracts)
    else:
        tranches = [("ALL", contracts, "trailing")]

    tranche_states = []
    for label, qty, exit_mode in tranches:
        tranche_states.append({
            "label": label, "qty": qty, "exit_mode": exit_mode,
            "active": True, "pnl_pct": 0.0, "exit_reason": None,
        })

    # Config (current v2.1 retunes)
    grace_min = 20
    premium_stop_pct = 30.0
    adaptive_activation = 35.0
    trail_active_width = 35.0
    trail_runner_width = 45.0
    trail_moonshot_width = 30.0
    runner_threshold = 150.0
    moonshot_threshold = 400.0
    profit_lock_tiers = [(250, 150), (150, 70), (80, 25)]
    theta_bleed_min = 45
    theta_bleed_loss = 30.0
    no_momentum_min = 45
    no_momentum_gain = 5.0

    peak = entry_premium
    peak_underlying = None
    locked_floor = None
    reval_done = False
    effective_stop = premium_stop_pct
    vol_tighten = False
    underlying_prices = []
    last_new_high_at = signal_time
    exit_premium = entry_premium
    exit_dur = 0.0

    for snap in snapshots:
        captured_str, midpoint, bid, ask, underlying = snap
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        captured = datetime.fromisoformat(captured_str)
        if captured.tzinfo is not None and signal_time.tzinfo is None:
            captured = captured.replace(tzinfo=None)
        elapsed_min = (captured - signal_time).total_seconds() / 60

        # Track peaks
        if price > peak:
            peak = price
            last_new_high_at = captured

        # Track underlying
        if underlying and underlying > 0:
            underlying_prices.append(underlying)
            if peak_underlying is None or underlying > peak_underlying:
                peak_underlying = underlying

        gain_pct = (price - entry_premium) / entry_premium * 100
        peak_gain_pct = (peak - entry_premium) / entry_premium * 100

        # Update profit lock
        for threshold, lock in sorted(profit_lock_tiers, key=lambda x: -x[0]):
            if peak_gain_pct >= threshold:
                locked_floor = lock
                break

        # ET time for EOD
        et_time = captured - timedelta(hours=4)

        # Expiry safety
        market_close_et = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        min_to_close = (market_close_et - et_time).total_seconds() / 60
        if 0 < min_to_close <= 10:
            for tr in tranche_states:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "expiry_safety"
            exit_premium = price
            exit_dur = elapsed_min
            break

        # Grace period
        if elapsed_min < grace_min:
            continue

        # Early negative reval (one-shot, within 20 min)
        if enable_early_neg_reval and not reval_done and elapsed_min <= 20 and gain_pct < 0:
            reval = early_neg_reval(direction, gain_pct, underlying_prices)
            if reval == "IMMEDIATE_EXIT":
                reval_done = True
                for tr in tranche_states:
                    if tr["active"]:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "early_neg_reval"
                exit_premium = price
                exit_dur = elapsed_min
                break
            elif reval == "HOLD":
                reval_done = True
                effective_stop *= 1.15
            else:
                reval_done = True

        # Premium stop
        loss_pct = (entry_premium - price) / entry_premium * 100
        if loss_pct >= effective_stop:
            for tr in tranche_states:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "premium_stop"
            exit_premium = price
            exit_dur = elapsed_min
            break

        # Profit lock
        if locked_floor is not None and gain_pct <= locked_floor:
            for tr in tranche_states:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = f"profit_lock_{int(locked_floor)}%"
            exit_premium = price
            exit_dur = elapsed_min
            break

        # Volume peak modifier
        if enable_volume_peak and gain_pct >= 35 and underlying_prices:
            vp = volume_peak_from_underlying(underlying_prices, direction)
            if vp == "tighten" and not vol_tighten:
                vol_tighten = True

        # Per-tranche exits
        any_active = False
        for tr in tranche_states:
            if not tr["active"]:
                continue
            any_active = True

            # T1 lock at 25%
            if tr["exit_mode"] == "lock_at_25" and gain_pct >= 25.0:
                tr["active"] = False
                tr["pnl_pct"] = gain_pct
                tr["exit_reason"] = "tranche1_lock"
                continue

            # Adaptive trail
            if peak_gain_pct < adaptive_activation:
                trail_stage = "DORMANT"
                trail_width = 100.0
            elif peak_gain_pct < runner_threshold:
                trail_stage = "ACTIVE"
                trail_width = trail_active_width
            elif peak_gain_pct < moonshot_threshold:
                trail_stage = "RUNNER"
                trail_width = trail_runner_width
            else:
                trail_stage = "MOONSHOT"
                trail_width = trail_moonshot_width

            # Runner tranche gets wider trail
            if tr["exit_mode"] == "runner" and trail_stage != "DORMANT":
                trail_width = get_runner_trail_width(peak_gain_pct)

            # Volume peak tighten
            if vol_tighten and trail_stage != "DORMANT":
                trail_width *= 0.7

            if trail_stage != "DORMANT" and peak > 0:
                drop = (peak - price) / peak * 100
                if drop >= trail_width:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = f"trail_{trail_stage}"
                    continue

            # Underlying-anchored trail
            if (enable_underlying_trail and trail_stage != "DORMANT"
                    and underlying and underlying > 0 and peak_underlying):
                u_trail = underlying_trail_pct_for_gain(peak_gain_pct)
                if direction in ("call", "bullish", "long"):
                    trigger = peak_underlying * (1.0 - u_trail)
                    if underlying < trigger:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "underlying_trail"
                        continue
                else:
                    trigger = peak_underlying * (1.0 + u_trail)
                    if underlying > trigger:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "underlying_trail"
                        continue

            # Theta bleed
            if elapsed_min >= theta_bleed_min and loss_pct >= theta_bleed_loss:
                tr["active"] = False
                tr["pnl_pct"] = gain_pct
                tr["exit_reason"] = "theta_bleed"
                continue

            # No momentum
            if elapsed_min >= no_momentum_min and gain_pct < no_momentum_gain:
                tr["active"] = False
                tr["pnl_pct"] = gain_pct
                tr["exit_reason"] = "no_momentum"
                continue

            # Time decay (stale after 3 PM ET)
            if et_time.hour >= 15 and last_new_high_at:
                lnh = last_new_high_at
                if captured.tzinfo is not None and lnh.tzinfo is None:
                    lnh = lnh.replace(tzinfo=captured.tzinfo)
                elif captured.tzinfo is None and lnh.tzinfo is not None:
                    lnh = lnh.replace(tzinfo=None)
                since_high = (captured - lnh).total_seconds() / 60
                if since_high >= 10:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "time_decay"
                    continue

        exit_premium = price
        exit_dur = elapsed_min

        if not any_active:
            break

    # Close remaining at last snapshot
    if snapshots:
        last = snapshots[-1]
        lp = last[1] or ((last[2] or 0) + (last[3] or 0)) / 2
        if lp and lp > 0:
            final_gain = (lp - entry_premium) / entry_premium * 100
            for tr in tranche_states:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = final_gain
                    tr["exit_reason"] = "market_close"
            if not exit_premium or exit_premium == entry_premium:
                exit_premium = lp
                lt = datetime.fromisoformat(last[0])
                if lt.tzinfo and not signal_time.tzinfo:
                    lt = lt.replace(tzinfo=None)
                exit_dur = (lt - signal_time).total_seconds() / 60

    # Compute weighted P&L
    total_qty = sum(tr["qty"] for tr in tranche_states)
    if total_qty == 0:
        return None
    weighted_pnl = sum(tr["pnl_pct"] * tr["qty"] for tr in tranche_states) / total_qty
    pnl_dollars = (weighted_pnl / 100) * contracts * entry_premium * 100

    mfe_pct = (peak - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
    mfe_gap = mfe_pct - weighted_pnl

    # Build exit reason string
    if enable_tranches and len(tranche_states) > 1:
        reason_parts = []
        for tr in tranche_states:
            reason_parts.append(f"{tr['label']}:{tr['exit_reason']}({tr['pnl_pct']:+.0f}%)")
        exit_reason = " | ".join(reason_parts)
    else:
        exit_reason = tranche_states[0]["exit_reason"] or "unknown"

    return TradeResult(
        ticker=ticker, direction=direction, strike=0.0,
        signal_time="", score=score,
        entry_premium=entry_premium, exit_premium=exit_premium,
        peak_premium=peak,
        pnl_pct=weighted_pnl, mfe_pct=mfe_pct, mfe_gap=mfe_gap,
        exit_reason=exit_reason, duration_min=exit_dur,
        contracts=contracts, pnl_dollars=pnl_dollars,
    )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

SCENARIOS = {
    "baseline": ("Current v2.1 (baseline)", {}),
    "tranches": ("§4 Three-tranche scale-out", {"enable_tranches": True}),
    "underlying_trail": ("§5 Underlying-anchored trail", {"enable_underlying_trail": True}),
    "volume_peak": ("§6 Volume-peak modifier", {"enable_volume_peak": True}),
    "early_neg_reval": ("ENRG Early neg thesis reval", {"enable_early_neg_reval": True}),
    "tranches+underlying": ("§4+§5 Tranches + Underlying", {
        "enable_tranches": True, "enable_underlying_trail": True,
    }),
    "tranches+vol": ("§4+§6 Tranches + Vol-peak", {
        "enable_tranches": True, "enable_volume_peak": True,
    }),
    "all_features": ("ALL v2.1 features", {
        "enable_tranches": True, "enable_underlying_trail": True,
        "enable_volume_peak": True, "enable_early_neg_reval": True,
    }),
    "all_no_enrg": ("ALL minus ENRG", {
        "enable_tranches": True, "enable_underlying_trail": True,
        "enable_volume_peak": True,
    }),
}


def run_scenario(signals, harvester_conn, flags):
    results = []
    skipped = 0

    for sig in signals:
        ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at = sig

        if expiry == "0DTE":
            expiry_date = resolve_expiry(created_at)
        else:
            expiry_date = expiry

        entry_premium = atm_premium
        if entry_premium is None or entry_premium <= 0:
            entry_premium = otm_premium
        if entry_premium is None or entry_premium <= 0:
            skipped += 1
            continue

        option_type = "call" if direction in ("call", "bullish", "long") else "put"
        signal_time = datetime.fromisoformat(created_at)

        # Try signal date expiry first, then next 4 business days
        base_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        snapshots = None
        for delta in range(0, 5):
            try_date = base_date + timedelta(days=delta)
            if try_date.weekday() >= 5:
                continue
            ct = build_contract_ticker(ticker, try_date.strftime("%Y-%m-%d"), strike, option_type)
            earlier = (signal_time - timedelta(minutes=2)).isoformat()
            snapshots = get_snapshots(harvester_conn, ct, earlier)
            if not snapshots:
                snapshots = get_snapshots(harvester_conn, ct, created_at)
            if snapshots:
                break

        if not snapshots:
            skipped += 1
            continue

        # Use first snapshot as actual entry price
        first_snap = snapshots[0]
        actual_entry = first_snap[1]
        if actual_entry and actual_entry > 0:
            entry_premium = actual_entry

        result = simulate_signal(
            snapshots, entry_premium, signal_time, ticker, direction, score,
            **flags,
        )
        if result is None:
            skipped += 1
            continue

        result.signal_time = created_at[:16]
        result.strike = strike
        results.append(result)

    return results, skipped


def print_summary_table(all_results, scenarios_to_run):
    print()
    print("=" * 140)
    print("  V2.1 FEATURE BACKTEST — REAL DISCORD SIGNALS + HARVESTER DATA")
    print("=" * 140)
    print()

    baseline_results = all_results.get("baseline", [])
    baseline_pnl = sum(r.pnl_dollars for r in baseline_results)

    print(f"  {'Scenario':<35} {'N':>3} {'WR':>5} {'Total P&L':>12} {'vs Base':>10} "
          f"{'Avg P&L%':>8} {'Avg MFE':>8} {'MFE Gap':>8} {'Dur':>5}")
    print(f"  {'-'*35} {'-'*3} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    for key, (label, _) in scenarios_to_run.items():
        results = all_results.get(key, [])
        if not results:
            print(f"  {label:<35} {'—':>3}")
            continue

        wins = [r for r in results if r.pnl_pct >= 0]
        total_pnl = sum(r.pnl_dollars for r in results)
        wr = len(wins) / len(results) * 100
        avg_pnl = sum(r.pnl_pct for r in results) / len(results)
        avg_mfe = sum(r.mfe_pct for r in results) / len(results)
        avg_gap = sum(r.mfe_gap for r in results) / len(results)
        avg_dur = sum(r.duration_min for r in results) / len(results)
        delta = total_pnl - baseline_pnl
        delta_str = f"${delta:+,.0f}" if key != "baseline" else "—"

        print(f"  {label:<35} {len(results):>3} {wr:>4.0f}% ${total_pnl:>+10,.2f} {delta_str:>10} "
              f"{avg_pnl:>+7.1f}% {avg_mfe:>+7.0f}% {avg_gap:>7.1f}% {avg_dur:>4.0f}m")

    # Exit reason breakdown
    print()
    print("  EXIT REASONS:")
    for key, (label, _) in scenarios_to_run.items():
        results = all_results.get(key, [])
        if not results:
            continue
        reasons = {}
        for r in results:
            reason = r.exit_reason
            if " | " in reason:
                # For multi-tranche, show simplified
                parts = reason.split(" | ")
                for p in parts:
                    sub = p.split(":")[1].split("(")[0] if ":" in p else p
                    reasons[sub] = reasons.get(sub, 0) + 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        top = sorted(reasons.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{k}:{v}" for k, v in top)
        print(f"  {label:<35} {top_str}")

    # Trade-by-trade comparison: baseline vs best
    print()
    print("=" * 140)
    print("  TRADE-BY-TRADE: Baseline vs ALL (minus ENRG)")
    print("=" * 140)

    best_key = "all_no_enrg"
    best_results = all_results.get(best_key, [])

    if baseline_results and best_results and len(baseline_results) == len(best_results):
        print()
        print(f"  {'Ticker':<6} {'Dir':>4} {'Score':>5} {'Time':>6} | "
              f"{'Base P&L':>10} {'Base Exit':>20} | "
              f"{'New P&L':>10} {'New Exit':>20} | {'Delta':>8}")
        print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*6} | {'-'*10} {'-'*20} | "
              f"{'-'*10} {'-'*20} | {'-'*8}")

        improved = 0
        degraded = 0
        total_delta = 0

        for base, new in zip(baseline_results, best_results):
            delta = new.pnl_dollars - base.pnl_dollars
            total_delta += delta
            marker = ""
            if delta > 10:
                improved += 1
                marker = " ++"
            elif delta < -10:
                degraded += 1
                marker = " --"

            # Shorten exit reasons for display
            base_exit = base.exit_reason[:20]
            new_exit = new.exit_reason
            if " | " in new_exit:
                parts = new_exit.split(" | ")
                new_exit = "; ".join(p.split(":")[1][:12] if ":" in p else p[:12] for p in parts)
            new_exit = new_exit[:20]

            print(f"  {base.ticker:<6} {base.direction:>4} {base.score:>5} "
                  f"{base.signal_time[11:16]:>5} | "
                  f"${base.pnl_dollars:>+9.2f} {base_exit:>20} | "
                  f"${new.pnl_dollars:>+9.2f} {new_exit:>20} | "
                  f"${delta:>+7.2f}{marker}")

        base_total = sum(r.pnl_dollars for r in baseline_results)
        new_total = sum(r.pnl_dollars for r in best_results)
        print()
        print(f"  Total:  Baseline=${base_total:+,.2f}  |  "
              f"New=${new_total:+,.2f}  |  Delta=${total_delta:+,.2f}")
        print(f"  Improved: {improved}  |  Degraded: {degraded}  |  "
              f"Unchanged: {len(baseline_results) - improved - degraded}")

    print()
    print("=" * 140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Test specific feature")
    args = parser.parse_args()

    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    print(f"\n  Loading {len(signals)} real Discord signals...")

    if args.feature:
        scenarios_to_run = {
            "baseline": SCENARIOS["baseline"],
            args.feature: SCENARIOS[args.feature],
        }
    else:
        scenarios_to_run = SCENARIOS

    all_results = {}
    for key, (label, flags) in scenarios_to_run.items():
        results, skipped = run_scenario(signals, harvester_conn, flags)
        all_results[key] = results
        matched = len(results)
        print(f"  {label:<35} → {matched} trades ({skipped} skipped)")

    print_summary_table(all_results, scenarios_to_run)

    harvester_conn.close()


if __name__ == "__main__":
    main()
