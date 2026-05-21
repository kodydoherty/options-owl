#!/usr/bin/env python3
"""Backtest the FINAL v2.2 plan against all signals with harvester tick data.

Final plan:
  - Hard stop -30% (unchanged)
  - BE clamp: once peak +15%, floor = entry
  - Soft trail: 15-35% band, floor = entry + 50% of peak gain
  - Smart grace (und_confirm): grace ends when underlying confirms direction, 20min cap
  - Adaptive trail: unchanged (35/150/400 thresholds)
  - Gate reorder: trails before time exits (no effect in sim, but theoretically sound)

Compares:
  1. v4_production  — what we're running now (hard stop + 20min grace + adaptive trail ONLY)
  2. final_plan     — v4 + BE clamp + soft trail + smart grace (und_confirm)

Usage:
  python scripts/backtest_final_plan.py
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone

SLIPPAGE = 0.15
HARD_STOP_PCT = 30.0
ADAPTIVE_ACTIVATION = 35.0
ACTIVE_WIDTH = 35.0
RUNNER_THRESHOLD = 150.0
RUNNER_WIDTH = 45.0
MOONSHOT_THRESHOLD = 400.0
MOONSHOT_WIDTH = 30.0
BE_CLAMP_ACTIVATION = 15.0
SOFT_TRAIL_MIN = 15.0
SOFT_TRAIL_MAX = 35.0
SOFT_TRAIL_FLOOR = 50.0
GRACE_CAP_MINUTES = 20


def build_ct(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    return f"O:{ticker}{dt.strftime('%y%m%d')}{cp}{int(strike * 1000):08d}"


def get_ticks(conn, ct, after_ts):
    rows = conn.execute("""
        SELECT captured_at, underlying_price, midpoint, bid, ask
        FROM harvest_snapshots
        WHERE contract_ticker = ? AND captured_at >= ?
        ORDER BY captured_at
    """, (ct, after_ts)).fetchall()
    ticks = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            und = r[1] or 0
            mid = r[2] or 0
            bid = r[3] or 0
            ask = r[4] or 0
            if (mid > 0 or bid > 0) and und > 0:
                ticks.append({
                    "ts": ts, "und": und,
                    "mid": mid if mid > 0 else (bid + ask) / 2,
                })
        except (ValueError, TypeError):
            continue
    return ticks


def simulate_v4(entry, entry_und, ticks, sig_ts, direction):
    """V4 production: hard stop + fixed 20min grace + adaptive trail."""
    if not ticks or entry <= 0:
        return entry, "no_data"
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.replace(tzinfo=timezone.utc)
    grace_end = sig_ts + timedelta(minutes=GRACE_CAP_MINUTES)
    peak = entry

    for tick in ticks:
        price = tick["mid"]
        if price <= 0:
            continue
        peak = max(peak, price)
        if tick["ts"] < grace_end:
            continue
        drop_entry = (entry - price) / entry * 100 if price < entry else 0
        peak_gain = (peak - entry) / entry * 100
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0

        if drop_entry >= HARD_STOP_PCT:
            return price, "stop_hit"
        if peak_gain >= MOONSHOT_THRESHOLD and drop_peak >= MOONSHOT_WIDTH:
            return price, "adaptive_moonshot"
        elif peak_gain >= RUNNER_THRESHOLD and drop_peak >= RUNNER_WIDTH:
            return price, "adaptive_runner"
        elif peak_gain >= ADAPTIVE_ACTIVATION and drop_peak >= ACTIVE_WIDTH:
            return price, "adaptive_active"

    return ticks[-1]["mid"], "eod_cutoff"


def simulate_final(entry, entry_und, ticks, sig_ts, direction):
    """Final plan: v4 + BE clamp + soft trail + smart grace (und_confirm)."""
    if not ticks or entry <= 0:
        return entry, "no_data"
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.replace(tzinfo=timezone.utc)
    grace_end = sig_ts + timedelta(minutes=GRACE_CAP_MINUTES)

    peak = entry
    be_active = False
    grace_over = False

    for tick in ticks:
        price = tick["mid"]
        und = tick["und"]
        if price <= 0:
            continue
        peak = max(peak, price)
        elapsed = (tick["ts"] - sig_ts).total_seconds() / 60
        peak_gain = (peak - entry) / entry * 100
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0
        drop_entry = (entry - price) / entry * 100 if price < entry else 0

        # Smart grace: end when underlying confirms (>0.1% move) OR 20min cap
        if not grace_over:
            und_move_pct = (und - entry_und) / entry_und * 100
            if elapsed >= GRACE_CAP_MINUTES:
                grace_over = True
            elif direction in ("call", "bullish", "long") and und_move_pct > 0.1:
                grace_over = True
            elif direction in ("put", "bearish", "short") and und_move_pct < -0.1:
                grace_over = True

        if not grace_over:
            continue

        # Hard stop
        if drop_entry >= HARD_STOP_PCT:
            return price, "stop_hit"

        # BE clamp
        if peak_gain >= BE_CLAMP_ACTIVATION:
            be_active = True
        if be_active and price <= entry:
            return entry, "be_clamp"

        # Soft trail (15-35%)
        if SOFT_TRAIL_MIN <= peak_gain < SOFT_TRAIL_MAX:
            floor = entry + (peak - entry) * (SOFT_TRAIL_FLOOR / 100)
            if price <= floor:
                return price, "soft_trail"

        # Adaptive trail
        if peak_gain >= MOONSHOT_THRESHOLD and drop_peak >= MOONSHOT_WIDTH:
            return price, "adaptive_moonshot"
        elif peak_gain >= RUNNER_THRESHOLD and drop_peak >= RUNNER_WIDTH:
            return price, "adaptive_runner"
        elif peak_gain >= ADAPTIVE_ACTIVATION and drop_peak >= ACTIVE_WIDTH:
            return price, "adaptive_active"

    return ticks[-1]["mid"], "eod_cutoff"


def apply_slippage(pnl):
    return pnl * (1 - SLIPPAGE) if pnl > 0 else pnl


def main():
    signals_db = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
    harvester_db = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

    sig_conn = sqlite3.connect(signals_db)
    sig_conn.row_factory = sqlite3.Row
    harv_conn = sqlite3.connect(harvester_db)

    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts,
               pt.id as trade_id, pt.pnl_dollars as traded_pnl,
               pt.exit_reason as traded_exit_reason
        FROM trade_signals ts
        LEFT JOIN paper_trades pt ON pt.signal_id = ts.id AND pt.parent_trade_id IS NULL
        ORDER BY ts.created_at
    """).fetchall()

    sim_data = []
    no_data = 0
    for sig in signals:
        strike = sig["strike"]
        premium = sig["atm_premium"] or sig["otm_premium"]
        if not strike or not premium or premium <= 0:
            continue
        ct = build_ct(sig["ticker"], sig["day"], sig["direction"], strike)
        ticks = get_ticks(harv_conn, ct, sig["sig_ts"])
        if not ticks:
            no_data += 1
            continue
        entry = ticks[0]["mid"]
        entry_und = ticks[0]["und"]
        if entry <= 0 or entry_und <= 0:
            continue
        score = sig["score"] or 0
        contracts = 5 if score >= 95 else (4 if score >= 90 else (3 if score >= 85 else 1))
        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)
        peak = max(t["mid"] for t in ticks)
        sim_data.append({
            "id": sig["id"], "ticker": sig["ticker"], "direction": sig["direction"],
            "day": sig["day"], "score": score, "entry": entry, "entry_und": entry_und,
            "contracts": contracts, "ticks": ticks, "sig_ts": sig_ts,
            "peak": peak, "peak_gain": (peak - entry) / entry * 100,
            "was_traded": sig["trade_id"] is not None,
            "actual_pnl": sig["traded_pnl"] or 0,
            "actual_reason": sig["traded_exit_reason"] or "",
        })

    print(f"Signals: {len(signals)} total, {len(sim_data)} with tick data, {no_data} no data")
    print()

    # Run both strategies
    v4_results = []
    final_results = []
    for sd in sim_data:
        v4_exit, v4_rsn = simulate_v4(sd["entry"], sd["entry_und"], sd["ticks"], sd["sig_ts"], sd["direction"])
        v4_pnl = apply_slippage((v4_exit - sd["entry"]) * sd["contracts"] * 100)
        v4_results.append((v4_exit, v4_rsn, v4_pnl))

        f_exit, f_rsn = simulate_final(sd["entry"], sd["entry_und"], sd["ticks"], sd["sig_ts"], sd["direction"])
        f_pnl = apply_slippage((f_exit - sd["entry"]) * sd["contracts"] * 100)
        final_results.append((f_exit, f_rsn, f_pnl))

    # Per-signal detail
    print("=" * 155)
    print(f"{'PER-SIGNAL: v4 production vs FINAL PLAN':^155}")
    print("=" * 155)
    print(f"{'ID':>3} {'Tckr':<5} {'Dir':<5} {'Day':<11} {'Sc':>3} {'Trd':>3} "
          f"{'Entry':>6} {'Peak%':>7} "
          f"{'v4PnL':>10} {'v4Rsn':<20} "
          f"{'FinalPnL':>10} {'FinalRsn':<20} {'Diff':>10}")
    print("-" * 155)

    for i, sd in enumerate(sim_data):
        v4_e, v4_r, v4_p = v4_results[i]
        f_e, f_r, f_p = final_results[i]
        d = f_p - v4_p
        trd = "YES" if sd["was_traded"] else "no"
        ds = f"+${d:.2f}" if d > 0 else f"${d:.2f}" if d < 0 else "$0.00"
        print(f"{sd['id']:>3} {sd['ticker']:<5} {sd['direction'][:4]:<5} {sd['day']:<11} "
              f"{sd['score']:>3} {trd:>3} ${sd['entry']:>5.2f} {sd['peak_gain']:>6.1f}% "
              f"${v4_p:>9.2f} {v4_r:<20} ${f_p:>9.2f} {f_r:<20} {ds:>10}")

    # Summary
    v4_total = sum(r[2] for r in v4_results)
    f_total = sum(r[2] for r in final_results)
    v4_wins = sum(1 for r in v4_results if r[2] > 0)
    f_wins = sum(1 for r in final_results if r[2] > 0)
    diff = f_total - v4_total

    print(f"\n{'=' * 80}")
    print(f"{'SUMMARY':^80}")
    print("=" * 80)
    print(f"  Signals simulated:   {len(sim_data)}")
    print(f"  Slippage:            {SLIPPAGE:.0%} on all gains")
    print()
    print(f"  v4 production P&L:   ${v4_total:>10.2f}  ({v4_wins}W / {len(sim_data)-v4_wins}L = {v4_wins/len(sim_data)*100:.0f}% WR)")
    print(f"  FINAL PLAN P&L:      ${f_total:>10.2f}  ({f_wins}W / {len(sim_data)-f_wins}L = {f_wins/len(sim_data)*100:.0f}% WR)")
    print(f"  Improvement:        +${diff:>10.2f}  ({'+'if diff>0 else ''}{diff/abs(v4_total)*100:.1f}%)" if v4_total else "")
    print()

    # Changed signals
    improved = [(i, final_results[i][2] - v4_results[i][2]) for i in range(len(sim_data))
                if final_results[i][2] > v4_results[i][2] + 0.50]
    regressed = [(i, final_results[i][2] - v4_results[i][2]) for i in range(len(sim_data))
                 if final_results[i][2] < v4_results[i][2] - 0.50]

    print(f"  Signals improved:    {len(improved)}")
    print(f"  Signals regressed:   {len(regressed)}")
    print(f"  Signals unchanged:   {len(sim_data) - len(improved) - len(regressed)}")

    # Top improvements
    improved.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  TOP IMPROVEMENTS:")
    for idx, d in improved[:8]:
        sd = sim_data[idx]
        v4_r = v4_results[idx][1]
        f_r = final_results[idx][1]
        print(f"    #{sd['id']} {sd['ticker']} {sd['day']}: +${d:.2f} "
              f"(v4:{v4_r} → final:{f_r}, peak +{sd['peak_gain']:.0f}%)")

    # Top regressions
    regressed.sort(key=lambda x: x[1])
    if regressed:
        print(f"\n  TOP REGRESSIONS:")
        for idx, d in regressed[:5]:
            sd = sim_data[idx]
            v4_r = v4_results[idx][1]
            f_r = final_results[idx][1]
            print(f"    #{sd['id']} {sd['ticker']} {sd['day']}: ${d:.2f} "
                  f"(v4:{v4_r} → final:{f_r}, peak +{sd['peak_gain']:.0f}%)")

    # Daily breakdown
    print(f"\n{'=' * 90}")
    print(f"{'DAY-BY-DAY':^90}")
    print("=" * 90)
    print(f"{'Date':<12} {'Sigs':>5} {'v4 PnL':>12} {'Final PnL':>12} {'Diff':>12} {'v4WR':>6} {'FinWR':>6}")
    print("-" * 90)
    days = sorted(set(sd["day"] for sd in sim_data))
    for day in days:
        idxs = [i for i, sd in enumerate(sim_data) if sd["day"] == day]
        v4d = sum(v4_results[i][2] for i in idxs)
        fd = sum(final_results[i][2] for i in idxs)
        dd = fd - v4d
        v4w = sum(1 for i in idxs if v4_results[i][2] > 0)
        fw = sum(1 for i in idxs if final_results[i][2] > 0)
        v4wr = v4w / len(idxs) * 100
        fwr = fw / len(idxs) * 100
        print(f"{day:<12} {len(idxs):>5} ${v4d:>10.2f} ${fd:>10.2f} "
              f"{'+'if dd>=0 else ''}${dd:>10.2f} {v4wr:>5.0f}% {fwr:>5.0f}%")
    print("-" * 90)
    dd = f_total - v4_total
    print(f"{'TOTAL':<12} {len(sim_data):>5} ${v4_total:>10.2f} ${f_total:>10.2f} "
          f"{'+'if dd>=0 else ''}${dd:>10.2f} "
          f"{v4_wins/len(sim_data)*100:>5.0f}% {f_wins/len(sim_data)*100:>5.0f}%")

    # Exit reason breakdown
    print(f"\n{'=' * 80}")
    print(f"{'EXIT REASON BREAKDOWN':^80}")
    print("=" * 80)
    for label, results in [("v4 production", v4_results), ("FINAL PLAN", final_results)]:
        reasons = {}
        for _, rsn, pnl in results:
            if rsn not in reasons:
                reasons[rsn] = {"n": 0, "pnl": 0.0}
            reasons[rsn]["n"] += 1
            reasons[rsn]["pnl"] += pnl
        print(f"\n  {label}:")
        for rsn, d in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            avg = d["pnl"] / d["n"] if d["n"] else 0
            print(f"    {rsn:<22} {d['n']:>3}x  total ${d['pnl']:>10.2f}  avg ${avg:>8.2f}")

    # Traded-only
    print(f"\n{'=' * 80}")
    print(f"{'TRADED SIGNALS ONLY':^80}")
    print("=" * 80)
    traded_idxs = [i for i, sd in enumerate(sim_data) if sd["was_traded"]]
    if traded_idxs:
        tv4 = sum(v4_results[i][2] for i in traded_idxs)
        tf = sum(final_results[i][2] for i in traded_idxs)
        tv4w = sum(1 for i in traded_idxs if v4_results[i][2] > 0)
        tfw = sum(1 for i in traded_idxs if final_results[i][2] > 0)
        print(f"  v4 production: ${tv4:>10.2f}  ({tv4w}W/{len(traded_idxs)-tv4w}L = {tv4w/len(traded_idxs)*100:.0f}% WR)")
        print(f"  FINAL PLAN:    ${tf:>10.2f}  ({tfw}W/{len(traded_idxs)-tfw}L = {tfw/len(traded_idxs)*100:.0f}% WR)")
        print(f"  Improvement:  +${tf-tv4:>10.2f}")

    print()
    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
