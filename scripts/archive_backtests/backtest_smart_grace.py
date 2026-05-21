#!/usr/bin/env python3
"""Backtest smart grace strategies that replace the fixed 20-minute grace period.

Instead of blindly ignoring stops for 20 minutes, use underlying price trend
data to decide whether to hold or cut early during the grace window.

Smart grace approaches tested:
  1. fixed_20m      — current production: blind 20min grace
  2. fixed_5m       — v2.2 proposal: blind 5min grace
  3. trend_slope    — cut if underlying moved >0.2% against thesis over last 5 ticks
  4. consecutive    — cut if underlying makes 5 consecutive ticks against thesis
  5. recovery       — 5min blind, then check if underlying is recovering toward entry
  6. prem_confirm   — grace until premium makes a new high OR 20min max
  7. und_confirm    — grace until underlying confirms direction OR 20min max

All tested WITH BE clamp + soft trail (the proven winners from feature backtest).
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
                    "ts": ts,
                    "und": und,
                    "mid": mid if mid > 0 else (bid + ask) / 2,
                })
        except (ValueError, TypeError):
            continue
    return ticks


def is_against_thesis(direction, current_und, entry_und):
    """Returns True if underlying is moving against our trade thesis."""
    if direction in ("call", "bullish", "long"):
        return current_und < entry_und
    else:
        return current_und > entry_und


def should_exit_stops(price, entry, peak, peak_gain, drop_from_peak, be_active, direction):
    """Check all exit conditions (hard stop, BE clamp, soft trail, adaptive trail).
    Returns (should_exit, reason, new_be_active)."""
    drop_from_entry = (entry - price) / entry * 100 if price < entry else 0

    # Hard stop
    if drop_from_entry >= HARD_STOP_PCT:
        return True, "stop_hit", be_active

    # BE clamp
    if peak_gain >= BE_CLAMP_ACTIVATION:
        be_active = True
    if be_active and price <= entry:
        return True, "be_clamp", be_active

    # Soft trail (15-35% band)
    if SOFT_TRAIL_MIN <= peak_gain < SOFT_TRAIL_MAX:
        floor = entry + (peak - entry) * (SOFT_TRAIL_FLOOR / 100)
        if price <= floor:
            return True, "soft_trail", be_active

    # Adaptive trail
    if peak_gain >= MOONSHOT_THRESHOLD:
        if drop_from_peak >= MOONSHOT_WIDTH:
            return True, "adaptive_moonshot", be_active
    elif peak_gain >= RUNNER_THRESHOLD:
        if drop_from_peak >= RUNNER_WIDTH:
            return True, "adaptive_runner", be_active
    elif peak_gain >= ADAPTIVE_ACTIVATION:
        if drop_from_peak >= ACTIVE_WIDTH:
            return True, "adaptive_active", be_active

    return False, "", be_active


def simulate_grace(entry_prem, entry_und, ticks, sig_ts, contracts, direction, strategy):
    """Run simulation with a specific grace strategy."""
    if not ticks or entry_prem <= 0:
        return entry_prem, "no_data", 0.0

    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.replace(tzinfo=timezone.utc)

    peak = entry_prem
    be_active = False
    grace_over = False
    prem_high = entry_prem  # highest premium seen
    und_prices = []  # track underlying prices during grace

    for tick in ticks:
        price = tick["mid"]
        und = tick["und"]
        if price <= 0:
            continue

        peak = max(peak, price)
        peak_gain = (peak - entry_prem) / entry_prem * 100
        drop_from_peak = (peak - price) / peak * 100 if peak > 0 else 0
        elapsed = (tick["ts"] - sig_ts).total_seconds() / 60
        prem_high = max(prem_high, price)
        und_prices.append(und)

        # --- Determine if grace period is over ---
        if not grace_over:
            if strategy == "fixed_20m":
                grace_over = elapsed >= 20

            elif strategy == "fixed_5m":
                grace_over = elapsed >= 5

            elif strategy == "trend_slope":
                # Grace over if: (a) 5+ ticks collected AND underlying slope is against thesis
                # OR (b) 20min hard cap
                if elapsed >= 20:
                    grace_over = True
                elif len(und_prices) >= 5:
                    recent = und_prices[-5:]
                    slope_pct = (recent[-1] - recent[0]) / entry_und * 100
                    if direction in ("call", "bullish", "long"):
                        # For calls, if underlying dropped >0.2% over last 5 ticks, grace ends
                        grace_over = slope_pct < -0.15
                    else:
                        # For puts, if underlying rose >0.2%
                        grace_over = slope_pct > 0.15

            elif strategy == "consecutive":
                # Grace over if 5 consecutive ticks against thesis, or 20min cap
                if elapsed >= 20:
                    grace_over = True
                elif len(und_prices) >= 5:
                    last_5 = und_prices[-5:]
                    all_against = all(
                        is_against_thesis(direction, p, entry_und) for p in last_5
                    )
                    grace_over = all_against

            elif strategy == "recovery":
                # 5min blind, then check if underlying is recovering
                # Grace stays ON if underlying is trending back toward entry
                if elapsed < 5:
                    grace_over = False
                elif elapsed >= 20:
                    grace_over = True
                elif len(und_prices) >= 3:
                    # Is underlying moving back toward entry?
                    recent_3 = und_prices[-3:]
                    if direction in ("call", "bullish", "long"):
                        recovering = recent_3[-1] > recent_3[0]  # moving up
                    else:
                        recovering = recent_3[-1] < recent_3[0]  # moving down
                    grace_over = not recovering

            elif strategy == "prem_confirm":
                # Grace until premium makes a new high above entry, or 20min max
                if elapsed >= 20:
                    grace_over = True
                elif prem_high > entry_prem * 1.02:  # premium confirmed +2%
                    grace_over = True  # confirmed — now protect it

            elif strategy == "und_confirm":
                # Grace until underlying confirms direction OR 20min max
                if elapsed >= 20:
                    grace_over = True
                elif len(und_prices) >= 3:
                    if direction in ("call", "bullish", "long"):
                        # Underlying needs to be above entry
                        grace_over = und > entry_und * 1.001
                    else:
                        grace_over = und < entry_und * 0.999

            # During grace, skip all exits
            if not grace_over:
                continue

        # --- Grace is over, check exits ---
        exit_now, reason, be_active = should_exit_stops(
            price, entry_prem, peak, peak_gain, drop_from_peak, be_active, direction
        )
        if exit_now:
            return price if reason != "be_clamp" else entry_prem, reason, 0.0

    # EOD
    return ticks[-1]["mid"], "eod_cutoff", 0.0


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

    # Load data
    sim_data = []
    for sig in signals:
        ticker = sig["ticker"]
        strike = sig["strike"]
        premium = sig["atm_premium"] or sig["otm_premium"]
        if not strike or not premium or premium <= 0:
            continue

        ct = build_ct(ticker, sig["day"], sig["direction"], strike)
        ticks = get_ticks(harv_conn, ct, sig["sig_ts"])
        if not ticks:
            continue

        entry_prem = ticks[0]["mid"]
        entry_und = ticks[0]["und"]
        if entry_prem <= 0 or entry_und <= 0:
            continue

        score = sig["score"] or 0
        contracts = 5 if score >= 95 else (4 if score >= 90 else (3 if score >= 85 else 1))

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        peak = max(t["mid"] for t in ticks)

        sim_data.append({
            "id": sig["id"],
            "ticker": ticker,
            "direction": sig["direction"],
            "day": sig["day"],
            "score": score,
            "entry_prem": entry_prem,
            "entry_und": entry_und,
            "contracts": contracts,
            "ticks": ticks,
            "sig_ts": sig_ts,
            "peak": peak,
            "peak_gain": (peak - entry_prem) / entry_prem * 100,
            "was_traded": sig["trade_id"] is not None,
        })

    print(f"Loaded {len(sim_data)} signals with tick data\n")

    # Strategies to test
    strategies = [
        "fixed_20m",
        "fixed_5m",
        "trend_slope",
        "consecutive",
        "recovery",
        "prem_confirm",
        "und_confirm",
    ]

    strategy_descs = {
        "fixed_20m":    "Current production: blind 20min grace",
        "fixed_5m":     "v2.2 proposal: blind 5min grace",
        "trend_slope":  "Cut if 5-tick underlying slope >0.15% against thesis (20m cap)",
        "consecutive":  "Cut if 5 consecutive ticks against thesis (20m cap)",
        "recovery":     "5min blind, then hold if underlying recovering toward entry",
        "prem_confirm": "Grace until premium hits +2%, then protect (20m cap)",
        "und_confirm":  "Grace until underlying confirms direction (20m cap)",
    }

    # Run all strategies
    all_results = {}
    for strat in strategies:
        results = []
        for sd in sim_data:
            exit_prem, reason, banked = simulate_grace(
                sd["entry_prem"], sd["entry_und"], sd["ticks"],
                sd["sig_ts"], sd["contracts"], sd["direction"], strat
            )
            pnl = (exit_prem - sd["entry_prem"]) * sd["contracts"] * 100
            if pnl > 0:
                pnl *= (1 - SLIPPAGE)
            results.append((sd, exit_prem, reason, pnl))
        all_results[strat] = results

    # Summary table
    base_total = sum(r[3] for r in all_results["fixed_20m"])

    print("=" * 120)
    print(f"{'SMART GRACE COMPARISON (all with BE clamp + soft trail)':^120}")
    print("=" * 120)
    print(f"{'Strategy':<16} {'Description':<56} {'PnL':>10} {'W':>4} {'L':>4} "
          f"{'WR':>5} {'AvgW':>8} {'AvgL':>8} {'vs 20m':>10}")
    print("-" * 120)

    for strat in strategies:
        results = all_results[strat]
        total = sum(r[3] for r in results)
        wins = sum(1 for r in results if r[3] > 0)
        losses = sum(1 for r in results if r[3] <= 0)
        wr = wins / len(results) * 100 if results else 0
        win_pnls = [r[3] for r in results if r[3] > 0]
        loss_pnls = [r[3] for r in results if r[3] <= 0]
        avg_w = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_l = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        diff = total - base_total
        diff_str = f"+${diff:.0f}" if diff >= 0 else f"-${abs(diff):.0f}"
        print(f"{strat:<16} {strategy_descs[strat][:54]:<56} ${total:>9.2f} {wins:>4} {losses:>4} "
              f"{wr:>4.0f}% ${avg_w:>7.2f} ${avg_l:>7.2f} {diff_str:>10}")

    # Per-signal comparison for key strategies
    print(f"\n{'=' * 150}")
    print(f"{'PER-SIGNAL: fixed_20m vs best smart grace strategies':^150}")
    print("=" * 150)

    compare_strats = ["fixed_20m", "recovery", "prem_confirm", "und_confirm"]
    hdr = (f"{'ID':>3} {'Tckr':<5} {'Dir':<5} {'Day':<11} {'Peak%':>7} ")
    for s in compare_strats:
        hdr += f"{'PnL_'+s[:8]:>12} {'Rsn':>10} "
    print(hdr)
    print("-" * 150)

    for i, sd in enumerate(sim_data):
        line = f"{sd['id']:>3} {sd['ticker']:<5} {sd['direction'][:4]:<5} {sd['day']:<11} {sd['peak_gain']:>6.1f}% "
        for strat in compare_strats:
            _, _, reason, pnl = all_results[strat][i]
            line += f"${pnl:>10.2f} {reason[:10]:>10} "
        print(line)

    # Feature isolation: where each smart grace DIFFERS from fixed_20m
    print(f"\n{'=' * 100}")
    print(f"{'WHERE SMART GRACE DIFFERS FROM FIXED 20M':^100}")
    print("=" * 100)

    for strat in ["trend_slope", "consecutive", "recovery", "prem_confirm", "und_confirm"]:
        results = all_results[strat]
        base = all_results["fixed_20m"]
        total = sum(r[3] for r in results)
        diff = total - base_total

        better = []
        worse = []
        for i in range(len(sim_data)):
            d = results[i][3] - base[i][3]
            if abs(d) > 0.50:
                sd = sim_data[i]
                if d > 0:
                    better.append((sd, d, results[i][2], base[i][2]))
                else:
                    worse.append((sd, d, results[i][2], base[i][2]))

        print(f"\n  {strat}: {'+'if diff>=0 else ''}${diff:.2f} vs fixed_20m")
        print(f"    {strategy_descs[strat]}")
        print(f"    Changed {len(better)+len(worse)} signals: {len(better)} improved, {len(worse)} regressed")

        if better:
            better.sort(key=lambda x: x[1], reverse=True)
            print(f"    IMPROVEMENTS:")
            for sd, d, new_r, old_r in better[:5]:
                print(f"      #{sd['id']} {sd['ticker']} {sd['day']} peak+{sd['peak_gain']:.0f}%: "
                      f"+${d:.2f} ({old_r}→{new_r})")
        if worse:
            worse.sort(key=lambda x: x[1])
            print(f"    REGRESSIONS:")
            for sd, d, new_r, old_r in worse[:5]:
                print(f"      #{sd['id']} {sd['ticker']} {sd['day']} peak+{sd['peak_gain']:.0f}%: "
                      f"${d:.2f} ({old_r}→{new_r})")

    # Traded-only comparison
    print(f"\n{'=' * 100}")
    print(f"{'TRADED SIGNALS ONLY':^100}")
    print("=" * 100)
    traded_idxs = [i for i, sd in enumerate(sim_data) if sd["was_traded"]]
    print(f"  {'Strategy':<16} {'PnL':>10} {'Wins':>5} {'WR':>6}")
    for strat in strategies:
        total = sum(all_results[strat][i][3] for i in traded_idxs)
        wins = sum(1 for i in traded_idxs if all_results[strat][i][3] > 0)
        wr = wins / len(traded_idxs) * 100 if traded_idxs else 0
        print(f"  {strat:<16} ${total:>9.2f} {wins:>5} {wr:>5.0f}%")

    # VERDICT
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print("=" * 80)

    verdicts = []
    for strat in strategies:
        total = sum(r[3] for r in all_results[strat])
        diff = total - base_total
        wins = sum(1 for r in all_results[strat] if r[3] > 0)
        base_wins = sum(1 for r in all_results["fixed_20m"] if r[3] > 0)
        n_changed = sum(1 for i in range(len(sim_data))
                        if abs(all_results[strat][i][3] - all_results["fixed_20m"][i][3]) > 0.50)
        n_regress = sum(1 for i in range(len(sim_data))
                        if all_results[strat][i][3] < all_results["fixed_20m"][i][3] - 0.50)
        verdicts.append((strat, total, diff, wins, n_changed, n_regress))

    print(f"\n  {'Strategy':<16} {'Total PnL':>10} {'vs 20m':>10} {'Wins':>5} {'Changed':>8} {'Regressed':>10}")
    print(f"  {'-'*65}")
    for strat, total, diff, wins, changed, regressed in verdicts:
        ds = f"+${diff:.0f}" if diff >= 0 else f"-${abs(diff):.0f}"
        print(f"  {strat:<16} ${total:>9.2f} {ds:>10} {wins:>5} {changed:>8} {regressed:>10}")

    print()
    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
