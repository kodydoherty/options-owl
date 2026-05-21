"""Backtest v2.1 features against the ACTUAL 37 trades the bot executed.

Uses paper_trades from the droplet (real entry prices, real contracts, real sizing)
matched to harvester snapshots for premium curves. This is the definitive test.

Usage:
    python scripts/backtest_v21_real_trades.py
"""

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRADES_DB = os.environ.get("TRADES_DB", "journal/owlet-kody/raw_messages.db")
HARVESTER_DB = os.environ.get("HARVESTER_DB", "journal/owlet-harvester/options_data.db")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_real_trades():
    conn = sqlite3.connect(TRADES_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, option_type, strike, contracts, score,
               premium_per_contract, exit_premium, mfe_premium,
               pnl_dollars, pnl_pct, mfe_pnl_pct, exit_reason,
               expiry_date, opened_at, closed_at,
               webull_order_id, duration_minutes
        FROM paper_trades WHERE status='closed' ORDER BY id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_contract_ticker(ticker, expiry_date, strike, option_type):
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type == "call" else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{date_str}{opt_char}{strike_int:08d}"


def get_snapshots(conn, contract_ticker, after_time):
    return conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
        FROM harvest_snapshots
        WHERE contract_ticker = ? AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_time)).fetchall()


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------

def plan_tranches(qty):
    if qty <= 1:
        return [("T1", qty, "trailing")]
    if qty == 2:
        return [("T1", 1, "lock_at_25"), ("T2", 1, "trailing")]
    n1 = qty // 3
    n2 = qty // 3
    n3 = qty - n1 - n2
    return [("T1", n1, "lock_at_25"), ("T2", n2, "trailing"), ("T3", n3, "runner")]


UNDERLYING_TRAIL_TIERS = [
    (100.0, 0.0050), (50.0, 0.0040), (15.0, 0.0030), (0.0, 0.0020),
]

RUNNER_TRAIL_TIERS = [
    (400.0, 20.0), (200.0, 30.0), (100.0, 35.0), (50.0, 40.0),
]


def underlying_trail_pct(gain_pct):
    for min_gain, pct in UNDERLYING_TRAIL_TIERS:
        if gain_pct >= min_gain:
            return pct
    return 0.0020


def runner_trail_width(gain_pct):
    for min_gain, w in RUNNER_TRAIL_TIERS:
        if gain_pct >= min_gain:
            return w
    return 40.0


def volume_peak_check(underlying_prices, direction):
    if len(underlying_prices) < 6:
        return None
    recent = underlying_prices[-6:]
    first_avg = sum(recent[:3]) / 3
    second_avg = sum(recent[3:]) / 3
    if direction in ("call", "bullish"):
        if second_avg < first_avg * 0.999:
            return "tighten"
    else:
        if second_avg > first_avg * 1.001:
            return "tighten"
    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    trade_id: int
    ticker: str
    direction: str
    contracts: int
    score: int
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    pnl_dollars: float
    mfe_pct: float
    mfe_gap: float
    exit_reason: str
    duration_min: float
    day: str


def simulate_exit(snapshots, entry_premium, signal_time, direction, contracts,
                  enable_tranches=False, enable_underlying_trail=False,
                  enable_volume_peak=False):
    """Simulate exit pipeline on real snapshots.

    Returns (weighted_pnl_dollars, peak_premium, exit_reason, duration_min, debug).
    weighted_pnl_dollars accounts for tranches exiting at different prices.
    """
    if not snapshots or entry_premium <= 0:
        return 0.0, entry_premium, "no_data", 0.0, ""

    # Config (current v2.1 retunes)
    grace_min = 20
    premium_stop_pct = 30.0
    adaptive_activation = 35.0
    trail_active_width = 35.0
    trail_runner_width = 45.0
    trail_moonshot_width = 30.0
    profit_lock_tiers = [(250, 150), (150, 70), (80, 25)]
    theta_bleed_min = 45
    theta_bleed_loss = 30.0
    no_momentum_min = 45

    # Tranches
    if enable_tranches and contracts >= 2:
        tranches = plan_tranches(contracts)
    else:
        tranches = [("ALL", contracts, "trailing")]

    states = []
    for label, qty, mode in tranches:
        states.append({"label": label, "qty": qty, "mode": mode,
                       "active": True, "pnl_pct": 0.0, "exit_price": None,
                       "exit_reason": None})

    peak = entry_premium
    peak_underlying = None
    locked_floor = None
    vol_tighten = False
    underlying_prices = []
    last_new_high = signal_time
    last_elapsed = 0.0

    def _close_tranche(s, price, gain, reason):
        s["active"] = False
        s["pnl_pct"] = gain
        s["exit_price"] = price
        s["exit_reason"] = reason

    def _close_all_active(price, gain, reason):
        for s in states:
            if s["active"]:
                _close_tranche(s, price, gain, reason)

    def _all_done():
        return all(not s["active"] for s in states)

    def _weighted_pnl():
        total = 0.0
        for s in states:
            ep = s["exit_price"] if s["exit_price"] is not None else entry_premium
            total += (ep - entry_premium) * s["qty"] * 100
        return total

    for snap in snapshots:
        cap_str, midpoint, bid, ask, underlying = snap
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        captured = datetime.fromisoformat(cap_str)
        if captured.tzinfo and not signal_time.tzinfo:
            captured = captured.replace(tzinfo=None)
        elapsed_min = (captured - signal_time).total_seconds() / 60
        last_elapsed = elapsed_min

        if price > peak:
            peak = price
            last_new_high = captured

        if underlying and underlying > 0:
            underlying_prices.append(underlying)
            if peak_underlying is None or underlying > peak_underlying:
                peak_underlying = underlying

        gain_pct = (price - entry_premium) / entry_premium * 100
        peak_gain = (peak - entry_premium) / entry_premium * 100

        # Profit lock
        for thresh, lock in sorted(profit_lock_tiers, key=lambda x: -x[0]):
            if peak_gain >= thresh:
                locked_floor = lock
                break

        et = captured - timedelta(hours=4)

        # Expiry safety
        mkt_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
        to_close = (mkt_close - et).total_seconds() / 60
        if 0 < to_close <= 10:
            _close_all_active(price, gain_pct, "expiry_safety")
            return _weighted_pnl(), peak, _build_reason(states, enable_tranches), elapsed_min, ""

        if elapsed_min < grace_min:
            continue

        # Hard stop
        loss = (entry_premium - price) / entry_premium * 100
        if loss >= premium_stop_pct:
            _close_all_active(price, gain_pct, "premium_stop")
            return _weighted_pnl(), peak, _build_reason(states, enable_tranches), elapsed_min, ""

        # Profit lock
        if locked_floor is not None and gain_pct <= locked_floor:
            _close_all_active(price, gain_pct, f"profit_lock_{int(locked_floor)}%")
            return _weighted_pnl(), peak, _build_reason(states, enable_tranches), elapsed_min, ""

        # Volume peak
        if enable_volume_peak and gain_pct >= 35 and underlying_prices:
            vp = volume_peak_check(underlying_prices, direction)
            if vp == "tighten" and not vol_tighten:
                vol_tighten = True

        # Per-tranche
        for s in states:
            if not s["active"]:
                continue

            # T1 lock
            if s["mode"] == "lock_at_25" and gain_pct >= 25.0:
                _close_tranche(s, price, gain_pct, "T1_lock_25%")
                continue

            # Trail
            if peak_gain < adaptive_activation:
                stage = "DORMANT"
                width = 100.0
            elif peak_gain < 150:
                stage = "ACTIVE"
                width = trail_active_width
            elif peak_gain < 400:
                stage = "RUNNER"
                width = trail_runner_width
            else:
                stage = "MOONSHOT"
                width = trail_moonshot_width

            if s["mode"] == "runner" and stage != "DORMANT":
                width = runner_trail_width(peak_gain)

            if vol_tighten and stage != "DORMANT":
                width *= 0.7

            if stage != "DORMANT" and peak > 0:
                drop = (peak - price) / peak * 100
                if drop >= width:
                    _close_tranche(s, price, gain_pct, f"trail_{stage}")
                    continue

            # Underlying trail
            if (enable_underlying_trail and stage != "DORMANT"
                    and underlying and underlying > 0 and peak_underlying):
                u_trail = underlying_trail_pct(peak_gain)
                if direction in ("call", "bullish"):
                    if underlying < peak_underlying * (1.0 - u_trail):
                        _close_tranche(s, price, gain_pct, "underlying_trail")
                        continue
                else:
                    if underlying > peak_underlying * (1.0 + u_trail):
                        _close_tranche(s, price, gain_pct, "underlying_trail")
                        continue

            # Theta bleed
            if elapsed_min >= theta_bleed_min and loss >= theta_bleed_loss:
                _close_tranche(s, price, gain_pct, "theta_bleed")
                continue

            # No momentum
            if elapsed_min >= no_momentum_min and gain_pct < 5:
                _close_tranche(s, price, gain_pct, "no_momentum")
                continue

            # Time decay
            if et.hour >= 15 and last_new_high:
                lnh = last_new_high
                if captured.tzinfo and not lnh.tzinfo:
                    lnh = lnh.replace(tzinfo=captured.tzinfo)
                elif not captured.tzinfo and lnh.tzinfo:
                    lnh = lnh.replace(tzinfo=None)
                if (captured - lnh).total_seconds() / 60 >= 10:
                    _close_tranche(s, price, gain_pct, "time_decay")
                    continue

        if _all_done():
            break

    # Close remaining at last snapshot
    if snapshots:
        last = snapshots[-1]
        lp = last[1] or ((last[2] or 0) + (last[3] or 0)) / 2
        if lp and lp > 0:
            fg = (lp - entry_premium) / entry_premium * 100
            for s in states:
                if s["active"]:
                    _close_tranche(s, lp, fg, "market_close")

    return _weighted_pnl(), peak, _build_reason(states, enable_tranches), last_elapsed, ""


def _build_reason(states, is_tranched):
    if is_tranched and len(states) > 1:
        parts = []
        for s in states:
            parts.append(f"{s['label']}:{s['exit_reason']}({s['pnl_pct']:+.0f}%)")
        return " | ".join(parts)
    return states[0]["exit_reason"] or "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trades = load_real_trades()
    harv_conn = sqlite3.connect(HARVESTER_DB)

    scenarios = {
        "actual": "ACTUAL BOT RESULT",
        "baseline": "Backtest baseline (current config)",
        "with_features": "Backtest + Tranches + UndTrail + VolPeak",
    }

    results = {k: [] for k in scenarios}

    matched = 0
    skipped = 0

    for trade in trades:
        signal_time = datetime.fromisoformat(trade["opened_at"])
        earlier = (signal_time - timedelta(minutes=2)).isoformat()

        # Try exact expiry first, then next 4 business days
        base_date = datetime.strptime(trade["expiry_date"], "%Y-%m-%d").date()
        snapshots = None
        for delta in range(0, 5):
            try_date = base_date + timedelta(days=delta)
            if try_date.weekday() >= 5:
                continue
            ct = build_contract_ticker(
                trade["ticker"], try_date.strftime("%Y-%m-%d"),
                trade["strike"], trade["option_type"],
            )
            snapshots = get_snapshots(harv_conn, ct, earlier)
            if snapshots:
                break
        if not snapshots:
            snapshots = []

        entry = trade["premium_per_contract"]
        contracts = trade["contracts"]
        direction = trade["direction"]
        day = trade["opened_at"][:10]

        # Actual bot result
        results["actual"].append(SimResult(
            trade_id=trade["id"], ticker=trade["ticker"],
            direction=direction, contracts=contracts, score=trade["score"],
            entry_premium=entry,
            exit_premium=trade["exit_premium"] or entry,
            peak_premium=trade["mfe_premium"] or entry,
            pnl_pct=trade["pnl_pct"] or 0,
            pnl_dollars=trade["pnl_dollars"] or 0,
            mfe_pct=trade["mfe_pnl_pct"] or 0,
            mfe_gap=(trade["mfe_pnl_pct"] or 0) - (trade["pnl_pct"] or 0),
            exit_reason=trade["exit_reason"] or "unknown",
            duration_min=trade["duration_minutes"] or 0,
            day=day,
        ))

        if not snapshots:
            skipped += 1
            # Copy actual result as fallback for sim scenarios
            for key in ("baseline", "with_features"):
                results[key].append(results["actual"][-1])
            continue

        matched += 1

        # Baseline: current config, no new features
        pnl_d, peak_p, reason, dur, _ = simulate_exit(
            snapshots, entry, signal_time, direction, contracts,
        )
        mfe = (peak_p - entry) / entry * 100 if entry > 0 else 0
        total_cost = entry * contracts * 100
        pnl_pct = pnl_d / total_cost * 100 if total_cost > 0 else 0
        results["baseline"].append(SimResult(
            trade_id=trade["id"], ticker=trade["ticker"],
            direction=direction, contracts=contracts, score=trade["score"],
            entry_premium=entry, exit_premium=0, peak_premium=peak_p,
            pnl_pct=pnl_pct, pnl_dollars=pnl_d,
            mfe_pct=mfe, mfe_gap=mfe - pnl_pct,
            exit_reason=reason, duration_min=dur, day=day,
        ))

        # With all 3 features
        pnl_d2, peak_p2, reason2, dur2, _ = simulate_exit(
            snapshots, entry, signal_time, direction, contracts,
            enable_tranches=True,
            enable_underlying_trail=True,
            enable_volume_peak=True,
        )
        mfe2 = (peak_p2 - entry) / entry * 100 if entry > 0 else 0
        pnl_pct2 = pnl_d2 / total_cost * 100 if total_cost > 0 else 0

        results["with_features"].append(SimResult(
            trade_id=trade["id"], ticker=trade["ticker"],
            direction=direction, contracts=contracts, score=trade["score"],
            entry_premium=entry, exit_premium=0, peak_premium=peak_p2,
            pnl_pct=pnl_pct2, pnl_dollars=pnl_d2,
            mfe_pct=mfe2, mfe_gap=mfe2 - pnl_pct2,
            exit_reason=reason2, duration_min=dur2, day=day,
        ))

    harv_conn.close()

    # Print results
    print()
    print("=" * 140)
    print(f"  V2.1 FEATURE BACKTEST — {len(trades)} REAL BOT TRADES ({matched} with harvester data, {skipped} using actual result)")
    print("=" * 140)

    # Summary table
    print()
    print(f"  {'Scenario':<45} {'Trades':>6} {'WR':>5} {'Total P&L':>12} "
          f"{'Avg P&L%':>9} {'MFE Gap':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*5} {'-'*12} {'-'*9} {'-'*8}")

    for key, label in scenarios.items():
        res = results[key]
        wins = sum(1 for r in res if r.pnl_dollars >= 0)
        total = sum(r.pnl_dollars for r in res)
        wr = wins / len(res) * 100 if res else 0
        avg_pnl = sum(r.pnl_pct for r in res) / len(res) if res else 0
        avg_gap = sum(r.mfe_gap for r in res) / len(res) if res else 0
        print(f"  {label:<45} {len(res):>6} {wr:>4.0f}% ${total:>+10,.2f} "
              f"{avg_pnl:>+8.1f}% {avg_gap:>7.1f}%")

    # Per-day comparison
    print()
    print("  PER-DAY BREAKDOWN:")
    print(f"  {'Day':<12} {'Trades':>6} | {'Actual':>10} | {'Baseline':>10} | {'+ Features':>10} | {'Feat Δ':>8}")
    print(f"  {'-'*12} {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

    days = sorted(set(r.day for r in results["actual"]))
    for day in days:
        actual_day = [r for r in results["actual"] if r.day == day]
        base_day = [r for r in results["baseline"] if r.day == day]
        feat_day = [r for r in results["with_features"] if r.day == day]

        a_pnl = sum(r.pnl_dollars for r in actual_day)
        b_pnl = sum(r.pnl_dollars for r in base_day)
        f_pnl = sum(r.pnl_dollars for r in feat_day)
        delta = f_pnl - b_pnl

        print(f"  {day:<12} {len(actual_day):>6} | ${a_pnl:>+9,.2f} | ${b_pnl:>+9,.2f} | ${f_pnl:>+9,.2f} | ${delta:>+7,.2f}")

    a_total = sum(r.pnl_dollars for r in results["actual"])
    b_total = sum(r.pnl_dollars for r in results["baseline"])
    f_total = sum(r.pnl_dollars for r in results["with_features"])
    print(f"  {'TOTAL':<12} {len(trades):>6} | ${a_total:>+9,.2f} | ${b_total:>+9,.2f} | ${f_total:>+9,.2f} | ${f_total - b_total:>+7,.2f}")

    # Trade-by-trade
    print()
    print("=" * 140)
    print("  TRADE-BY-TRADE COMPARISON")
    print("=" * 140)
    print()
    print(f"  {'#':>2} {'Ticker':<6} {'Dir':>4} {'Ct':>3} {'Day':>10} | "
          f"{'ACTUAL':>10} {'Reason':>18} | "
          f"{'BASELINE':>10} {'Reason':>18} | "
          f"{'+ FEATURES':>10} {'Reason':>18} | {'Δ':>8}")
    print(f"  {'-'*2} {'-'*6} {'-'*4} {'-'*3} {'-'*10} | "
          f"{'-'*10} {'-'*18} | "
          f"{'-'*10} {'-'*18} | "
          f"{'-'*10} {'-'*18} | {'-'*8}")

    improved = degraded = 0
    for i in range(len(trades)):
        a = results["actual"][i]
        b = results["baseline"][i]
        f = results["with_features"][i]

        delta = f.pnl_dollars - b.pnl_dollars
        marker = ""
        if delta > 5:
            improved += 1
            marker = " ++"
        elif delta < -5:
            degraded += 1
            marker = " --"

        a_reason = a.exit_reason[:18]
        b_reason = b.exit_reason[:18] if " | " not in b.exit_reason else b.exit_reason.split("|")[0][:18]
        f_reason = f.exit_reason[:18] if " | " not in f.exit_reason else f.exit_reason.split("|")[0][:18]

        print(f"  {a.trade_id:>2} {a.ticker:<6} {a.direction:>4} {a.contracts:>3} {a.day:>10} | "
              f"${a.pnl_dollars:>+9,.2f} {a_reason:>18} | "
              f"${b.pnl_dollars:>+9,.2f} {b_reason:>18} | "
              f"${f.pnl_dollars:>+9,.2f} {f_reason:>18} | "
              f"${delta:>+7,.2f}{marker}")

    print()
    print(f"  Feature improvement: {improved} trades better, {degraded} worse, "
          f"{len(trades) - improved - degraded} unchanged")
    print(f"  Net delta vs baseline: ${f_total - b_total:+,.2f}")
    print()
    print("=" * 140)


if __name__ == "__main__":
    main()
