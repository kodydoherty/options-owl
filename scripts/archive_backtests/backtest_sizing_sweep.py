#!/usr/bin/env python3
"""Backtest sizing parameters: sweep MAX_CONCURRENT (sizing slots) and score tiers.

Tests different slot divisors and tier multipliers against all historical signals
using v5 exit simulation with real harvester premium data.

Usage:
  python scripts/backtest_sizing_sweep.py [signals_db] [harvester_db]
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

SLIPPAGE = 0.15
PORTFOLIO = 8000
MAX_PORTFOLIO_RISK_PCT = 75
MAX_POSITION_PCT = 15

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"


# ============================================================================
# Sizing configurations to test
# ============================================================================

TIER_CONFIGS = {
    "old_tiers": {
        "label": "Old (v3 tiers)",
        "tiers": [(150, 1.00), (130, 0.75), (110, 0.50), (95, 0.25), (78, 0.15)],
        "floor": 95,  # old floor
    },
    "new_tiers": {
        "label": "New (v4 tiers)",
        "tiers": [(150, 1.00), (135, 0.85), (120, 0.65), (105, 0.50), (78, 0.35)],
        "floor": 78,
    },
}

SLOT_CONFIGS = [3, 4, 5]
POS_PCT_CONFIGS = [15, 20, 25]


def size_contracts(score, cost_per_contract, balance, tiers, floor, max_concurrent,
                   max_position_pct=MAX_POSITION_PCT):
    """Replicate score_to_contracts with configurable tiers and slots."""
    if score < floor:
        return 0

    score_mult = tiers[-1][1]  # default to lowest tier
    for threshold, mult in tiers:
        if score >= threshold:
            score_mult = mult
            break

    if cost_per_contract <= 0:
        return max(1, int(5 * score_mult))

    total_deployable = balance * (MAX_PORTFOLIO_RISK_PCT / 100)
    target_per_trade = total_deployable / max(1, max_concurrent)
    scaled_target = target_per_trade * score_mult
    raw_contracts = int(scaled_target / cost_per_contract)

    max_spend = balance * (max_position_pct / 100)
    max_by_position = int(max_spend / cost_per_contract)

    final = min(raw_contracts, max_by_position)
    return max(1, final) if score >= floor else 0


# ============================================================================
# v5 exit simulation (from backtest_compare.py)
# ============================================================================

def _parse_tick(tick, sig_ts, entry):
    ts, mid, bid, ask, underlying = tick
    ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
    price = mid if mid and mid > 0 else ((bid + ask) / 2 if bid and ask else 0)
    if price <= 0:
        return None
    elapsed = (ts_dt - sig_ts).total_seconds() / 60
    gain_pct = (price - entry) / entry * 100
    et_hour = (ts_dt.hour - 4) % 24
    et_min = ts_dt.minute
    return price, elapsed, gain_pct, et_hour, et_min, underlying, ts_dt


def _make_exit(price, entry, contracts, elapsed, reason):
    pnl = (price - entry) * contracts * 100
    if pnl > 0:
        pnl *= (1 - SLIPPAGE)
    return pnl, reason, elapsed


def _eod_check(et_hour, et_min):
    return et_hour >= 15 and et_min >= 45


def _end_of_data(ticks, entry, contracts, sig_ts, peak):
    for t in reversed(ticks):
        price = t[1] if t[1] and t[1] > 0 else 0
        if price > 0:
            pnl = (price - entry) * contracts * 100
            if pnl > 0:
                pnl *= (1 - SLIPPAGE)
            ts_dt = datetime.fromisoformat(t[0]) if isinstance(t[0], str) else t[0]
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            elapsed = (ts_dt - sig_ts).total_seconds() / 60
            return pnl, "eod_data_end", elapsed
    return 0, "no_data", 0


def v5_simulate(entry, ticks, sig_ts, contracts, direction):
    """v5 exit: scalp trail, checkpoint cut, confirmed stop, soft trail, adaptive trail."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    peak = entry
    entry_underlying = None

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue
        price, elapsed, gain_pct, et_hour, et_min, underlying, ts_dt = parsed

        if price > peak:
            peak = price
        if entry_underlying is None and underlying and underlying > 0:
            entry_underlying = underlying

        peak_gain = (peak - entry) / entry * 100
        drop_entry = max(0, (entry - price) / entry * 100)
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0

        u_move = 0.0
        has_underlying = False
        underlying_against = False
        underlying_confirms = False
        if entry_underlying and underlying and underlying > 0:
            has_underlying = True
            u_move = (underlying - entry_underlying) / entry_underlying * 100
            if is_call:
                underlying_against = u_move < -0.3
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > 0.3
                underlying_confirms = u_move < -0.2

        if _eod_check(et_hour, et_min):
            return _make_exit(price, entry, contracts, elapsed, "eod_cutoff")

        if elapsed < 5:
            continue

        # SCALP TRAIL
        if peak_gain >= 20 and gain_pct > 0 and gain_pct < peak_gain * 0.6:
            if not underlying_confirms:
                return _make_exit(price, entry, contracts, elapsed, "scalp_trail")

        # CHECKPOINT CUT
        if drop_entry >= 15:
            if has_underlying and underlying_against:
                return _make_exit(price, entry, contracts, elapsed, "checkpoint_cut")

        # CONFIRMED STOP
        if underlying_against:
            if drop_entry >= 25:
                return _make_exit(price, entry, contracts, elapsed, "confirmed_stop")
        else:
            if drop_entry >= 55:
                return _make_exit(price, entry, contracts, elapsed, "hard_stop")

        # SOFT TRAIL
        if 15 <= peak_gain < 50:
            floor = entry + (peak - entry) * 0.50
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "soft_trail")

        # ADAPTIVE TRAIL
        if peak_gain >= 400:
            if drop_peak >= 30:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_moonshot")
        elif peak_gain >= 150:
            if drop_peak >= 45:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_runner")
        elif peak_gain >= 40:
            if drop_peak >= 40:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_active")

        # THETA BLEED
        if elapsed >= 120 and drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "theta_bleed")

    return _end_of_data(ticks, entry, contracts, sig_ts, peak)


def build_ct(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    ds = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    si = int(strike * 1000)
    return f"O:{ticker}{ds}{cp}{si:08d}"


def main():
    sig_conn = sqlite3.connect(SIGNALS_DB)
    sig_conn.row_factory = sqlite3.Row
    harv_conn = sqlite3.connect(HARVESTER_DB)

    signals = sig_conn.execute("""
        SELECT ts.id, ts.ticker, ts.direction, ts.score, ts.strike,
               ts.atm_premium, ts.otm_premium, date(ts.created_at) as day,
               ts.created_at as sig_ts
        FROM trade_signals ts ORDER BY ts.created_at
    """).fetchall()

    print(f"Loaded {len(signals)} signals from {SIGNALS_DB}")

    # Pre-fetch tick data for all signals
    tick_cache = {}
    no_data = no_strike = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        day = sig["day"]
        strike = sig["strike"]
        score_raw = sig["score"] or 0
        # Display scores (capped at 100) need to be converted to raw scale
        # display = min(100, round(raw * 100 / 180)), so raw ≈ display * 1.8
        # Only convert if score <= 100 (likely display score, not raw)
        score = int(score_raw * 1.8) if score_raw <= 100 else score_raw
        premium = sig["atm_premium"] or sig["otm_premium"]

        if not strike or not premium or premium <= 0:
            no_strike += 1
            continue

        contract = build_ct(ticker, day, direction, strike)
        rows = harv_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price
            FROM harvest_snapshots WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (contract, sig["sig_ts"])).fetchall()

        if not rows:
            no_data += 1
            continue

        first = rows[0]
        entry = (first[3] if first[3] and first[3] > 0 else first[1]) or premium
        if entry <= 0:
            entry = premium

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        tick_cache[sig["id"]] = {
            "ticker": ticker, "direction": direction, "day": day,
            "score": score, "entry": entry, "premium": premium,
            "rows": rows, "sig_ts": sig_ts,
            "cost_per_contract": entry * 100,
        }

    print(f"Usable signals: {len(tick_cache)}  |  No data: {no_data}  |  No strike: {no_strike}")

    # ===================================================================
    # Run sweep: for each (tier_config, slot_count), simulate all trades
    # ===================================================================

    results = {}  # (tier_name, slots, pos_pct) -> list of trade results

    for tier_name, tier_cfg in TIER_CONFIGS.items():
        for slots in SLOT_CONFIGS:
            for pos_pct in POS_PCT_CONFIGS:
                key = (tier_name, slots, pos_pct)
                trades = []

                for sig_id, data in tick_cache.items():
                    contracts = size_contracts(
                        data["score"], data["cost_per_contract"],
                        PORTFOLIO, tier_cfg["tiers"], tier_cfg["floor"], slots,
                        max_position_pct=pos_pct,
                    )
                    if contracts == 0:
                        continue

                    pnl, reason, hold = v5_simulate(
                        data["entry"], data["rows"], data["sig_ts"],
                        contracts, data["direction"],
                    )
                    trades.append({
                        "ticker": data["ticker"], "dir": data["direction"],
                        "day": data["day"], "score": data["score"],
                        "entry": data["entry"], "contracts": contracts,
                        "pnl": pnl, "reason": reason, "hold": hold,
                        "cost": contracts * data["cost_per_contract"],
                    })

                results[key] = trades

    # ===================================================================
    # REPORT
    # ===================================================================

    print(f"\n{'=' * 130}")
    print(f"SIZING SWEEP: {len(tick_cache)} signals, ${PORTFOLIO:,} portfolio, "
          f"{SLIPPAGE*100:.0f}% slippage, {MAX_PORTFOLIO_RISK_PCT}% risk cap")
    print(f"{'=' * 130}")

    # Header
    print(f"\n{'Config':<20} {'Slots':>5} {'Pos%':>5} {'Trades':>6} {'Total P&L':>12} "
          f"{'Return':>8} {'Win%':>6} {'W/L':>7} {'AvgWin':>9} {'AvgLoss':>9} "
          f"{'W:L':>6} {'AvgContr':>8} {'AvgCost':>9} {'AvgHold':>7}")
    print("-" * 130)

    best_pnl = -999999
    best_key = None

    for tier_name, tier_cfg in TIER_CONFIGS.items():
        for slots in SLOT_CONFIGS:
            for pos_pct in POS_PCT_CONFIGS:
                key = (tier_name, slots, pos_pct)
                trades = results[key]
                if not trades:
                    continue

                wins = [t for t in trades if t["pnl"] > 0]
                losses = [t for t in trades if t["pnl"] <= 0]
                total_pnl = sum(t["pnl"] for t in trades)
                wr = len(wins) / len(trades) * 100
                avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
                avg_l = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
                wl_ratio = abs(avg_w / avg_l) if avg_l else 0
                avg_contracts = sum(t["contracts"] for t in trades) / len(trades)
                avg_cost = sum(t["cost"] for t in trades) / len(trades)
                avg_hold = sum(t["hold"] for t in trades) / len(trades)

                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_key = key

                print(f"{tier_cfg['label']:<20} {slots:>5} {pos_pct:>4}% {len(trades):>6} "
                      f"${total_pnl:>+10,.0f} {total_pnl/PORTFOLIO*100:>+7.1f}% "
                      f"{wr:>5.1f}% {len(wins):>3}/{len(losses):<3} "
                      f"${avg_w:>+7,.0f} ${avg_l:>+7,.0f} "
                      f"{wl_ratio:>5.2f} {avg_contracts:>7.1f} "
                      f"${avg_cost:>7,.0f} {avg_hold:>5.0f}m")

        print()  # blank line between tier configs

    print(f"{'=' * 130}")
    print(f"BEST CONFIG: {TIER_CONFIGS[best_key[0]]['label']} with {best_key[1]} slots, "
          f"{best_key[2]}% max position "
          f"→ ${best_pnl:>+,.0f} ({best_pnl/PORTFOLIO*100:>+.1f}%)")
    print(f"{'=' * 130}")

    # ===================================================================
    # Detailed comparison: best config vs current production (new_tiers, 5 slots)
    # ===================================================================

    prod_key = ("new_tiers", 5, 15)
    if best_key != prod_key:
        print(f"\n{'=' * 130}")
        print(f"DETAILED: Best ({TIER_CONFIGS[best_key[0]]['label']}, {best_key[1]} slots, {best_key[2]}% pos) "
              f"vs Production (New tiers, 5 slots, 15% pos)")
        print(f"{'=' * 130}")

        best_trades = {(t["ticker"], t["day"]): t for t in results[best_key]}
        prod_trades = {(t["ticker"], t["day"]): t for t in results[prod_key]}

        all_keys = sorted(set(best_trades.keys()) | set(prod_trades.keys()),
                         key=lambda k: k[1])

        print(f"\n{'Day':<12} {'Ticker':<6} {'Score':>5} {'Entry':>7} "
              f"{'Best#':>5} {'BestP&L':>9} {'Prod#':>5} {'ProdP&L':>9} {'Delta':>9}")
        print("-" * 80)

        for k in all_keys:
            bt = best_trades.get(k)
            pt = prod_trades.get(k)
            day = k[1]
            ticker = k[0]

            b_c = bt["contracts"] if bt else 0
            b_pnl = bt["pnl"] if bt else 0
            p_c = pt["contracts"] if pt else 0
            p_pnl = pt["pnl"] if pt else 0
            delta = b_pnl - p_pnl

            score = (bt or pt)["score"]
            entry = (bt or pt)["entry"]

            if abs(delta) > 1:
                print(f"{day:<12} {ticker:<6} {score:>5} ${entry:>5.2f} "
                      f"{b_c:>5} ${b_pnl:>+7,.0f} {p_c:>5} ${p_pnl:>+7,.0f} "
                      f"${delta:>+7,.0f}")

    # ===================================================================
    # Score distribution analysis
    # ===================================================================

    print(f"\n{'=' * 120}")
    print("SCORE DISTRIBUTION (all usable signals)")
    print(f"{'=' * 120}")

    score_buckets = defaultdict(int)
    for data in tick_cache.values():
        s = data["score"]
        if s >= 150:
            score_buckets["150+ (elite)"] += 1
        elif s >= 135:
            score_buckets["135-149 (strong)"] += 1
        elif s >= 120:
            score_buckets["120-134 (solid)"] += 1
        elif s >= 105:
            score_buckets["105-119 (moderate)"] += 1
        elif s >= 95:
            score_buckets["95-104 (marginal+)"] += 1
        elif s >= 78:
            score_buckets["78-94 (marginal)"] += 1
        else:
            score_buckets["<78 (rejected)"] += 1

    for bucket in sorted(score_buckets.keys()):
        count = score_buckets[bucket]
        pct = count / len(tick_cache) * 100
        bar = "#" * int(pct)
        print(f"  {bucket:<25} {count:>4} ({pct:>5.1f}%) {bar}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
