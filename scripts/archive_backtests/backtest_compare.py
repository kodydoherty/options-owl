#!/usr/bin/env python3
"""Compare v3, v4.1, and v5 exit strategies side-by-side on ALL signals.

Usage:
  python scripts/backtest_compare.py [signals_db] [harvester_db]

Defaults:
  signals_db   = journal/owlet-kody/raw_messages.db
  harvester_db = journal/owlet-harvester/options_data.db
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

SLIPPAGE = 0.15
PORTFOLIO = 8000

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"


# ============================================================================
# Shared helpers
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


# ============================================================================
# v3 — Current production (v4.1 settings but EXIT_ENGINE=v3)
# Hard stop -30%, 20min grace, BE clamp +15%, soft trail 15-35%,
# adaptive trail (35/150/400), dollar trail 40% activation,
# profit floor ratchet, thesis cut -40%, theta bleed 45min
# ============================================================================

def v3_simulate(entry, ticks, sig_ts, contracts, direction):
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    peak = entry
    grace_minutes = 20

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue
        price, elapsed, gain_pct, et_hour, et_min, underlying, ts_dt = parsed

        if price > peak:
            peak = price

        peak_gain = (peak - entry) / entry * 100
        drop_entry = max(0, (entry - price) / entry * 100)
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0

        # EOD cutoff
        if _eod_check(et_hour, et_min):
            return _make_exit(price, entry, contracts, elapsed, "eod_cutoff")

        # Grace period — no stops for first 20min
        if elapsed < grace_minutes:
            # Catastrophic stop only (-45%)
            if drop_entry >= 45:
                return _make_exit(price, entry, contracts, elapsed, "catastrophic_stop")
            continue

        # Hard stop -30%
        if drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "stop_loss")

        # BE clamp: peaked +15%, now back at/below entry
        if peak_gain >= 15 and price <= entry:
            return _make_exit(price, entry, contracts, elapsed, "be_clamp")

        # Soft trail (15-35% peak gain band): floor = entry + 50% of peak gain
        if 15 <= peak_gain < 35:
            floor = entry + (peak - entry) * 0.50
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "soft_trail")

        # Dollar trail (activation 40% profit from entry)
        if peak_gain >= 40:
            entry_cost = entry * 100  # per contract
            peak_profit = (peak - entry) * 100
            current_profit = (price - entry) * 100
            # Step sizing
            if peak_profit > entry_cost * 0.25:
                step = entry_cost * 0.10  # tighter steps
            else:
                step = entry_cost * 0.20  # wider steps
            # Ratchet floor
            steps_up = int(peak_profit / step) if step > 0 else 0
            dollar_floor = steps_up * step - step  # one step below peak
            if current_profit <= dollar_floor and dollar_floor > 0:
                return _make_exit(price, entry, contracts, elapsed, "dollar_trail")

        # Adaptive trail (3 stages)
        if peak_gain >= 400:
            if drop_peak >= 30:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_moonshot")
        elif peak_gain >= 150:
            if drop_peak >= 45:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_runner")
        elif peak_gain >= 35:
            if drop_peak >= 35:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_active")

        # Profit floor ratchet (activate +15%, keep 60% of peak gain)
        if peak_gain >= 15:
            floor = entry + (peak - entry) * 0.60
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "profit_floor")

        # Thesis cut: down -40% with trend analysis (simplified: cut after 4+ ticks in zone)
        # In backtest we simplify — just cut at -40% after grace
        if drop_entry >= 40:
            return _make_exit(price, entry, contracts, elapsed, "thesis_cut")

        # Theta bleed: 45min+ and down 30%+
        if elapsed >= 45 and drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "theta_bleed")

    return _end_of_data(ticks, entry, contracts, sig_ts, peak)


# ============================================================================
# v4.1 — "Tuned production" (what was deployed before v5)
# Same as v3 but: dollar trail activation 40%, adaptive trail activation 35%,
# ENRG disabled, thesis_cut enabled, profit_floor enabled, soft trail 15-35%
# This is essentially current production with EXIT_ENGINE=v3.
# The differences from v3 are subtle — v4.1 is v3 with tuned params.
# For this comparison, v4 = v3 + wider dollar trail + thesis cut patience.
# ============================================================================

def v4_simulate(entry, ticks, sig_ts, contracts, direction):
    """v4.1 — same as v3 but thesis_cut is patient (waits for trend confirmation)."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    peak = entry
    entry_underlying = None
    grace_minutes = 20
    thesis_ticks_in_zone = 0

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

        # EOD cutoff
        if _eod_check(et_hour, et_min):
            return _make_exit(price, entry, contracts, elapsed, "eod_cutoff")

        # Grace period — no stops for first 20min (smart grace: end early if underlying confirms)
        if elapsed < grace_minutes:
            if entry_underlying and underlying and underlying > 0:
                u_move = (underlying - entry_underlying) / entry_underlying * 100
                if is_call and u_move > 0.1:
                    pass  # grace ended early — fall through to stops
                elif not is_call and u_move < -0.1:
                    pass  # grace ended early
                else:
                    continue  # still in grace
            else:
                continue

        # Hard stop -30%
        if drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "stop_loss")

        # BE clamp: peaked +15%, now back at/below entry
        if peak_gain >= 15 and price <= entry:
            return _make_exit(price, entry, contracts, elapsed, "be_clamp")

        # Soft trail (15-35% peak gain band)
        if 15 <= peak_gain < 35:
            floor = entry + (peak - entry) * 0.50
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "soft_trail")

        # Dollar trail (activation 40%)
        if peak_gain >= 40:
            entry_cost = entry * 100
            peak_profit = (peak - entry) * 100
            current_profit = (price - entry) * 100
            if peak_profit > entry_cost * 0.25:
                step = entry_cost * 0.10
            else:
                step = entry_cost * 0.20
            steps_up = int(peak_profit / step) if step > 0 else 0
            dollar_floor = steps_up * step - step
            if current_profit <= dollar_floor and dollar_floor > 0:
                return _make_exit(price, entry, contracts, elapsed, "dollar_trail")

        # Adaptive trail (3 stages)
        if peak_gain >= 400:
            if drop_peak >= 30:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_moonshot")
        elif peak_gain >= 150:
            if drop_peak >= 45:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_runner")
        elif peak_gain >= 35:
            if drop_peak >= 35:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_active")

        # Profit floor ratchet
        if peak_gain >= 15:
            floor = entry + (peak - entry) * 0.60
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "profit_floor")

        # Thesis cut (v4.1: patient — wait 4 ticks in danger zone, check bounce)
        if drop_entry >= 40:
            thesis_ticks_in_zone += 1
            if thesis_ticks_in_zone >= 4:
                # Check for bounce from recent behavior (simplified)
                return _make_exit(price, entry, contracts, elapsed, "thesis_cut")
        else:
            thesis_ticks_in_zone = max(0, thesis_ticks_in_zone - 1)

        # Theta bleed: 45min+ and down 30%+
        if elapsed >= 45 and drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "theta_bleed")

    return _end_of_data(ticks, entry, contracts, sig_ts, peak)


# ============================================================================
# v5 — dynamic signal-driven exits (no fixed time windows)
# Scalp: underlying-confirmed profit taking
# Stop: tight when underlying against, wide when not
# Checkpoint: cut when BOTH premium AND underlying against
# Trails: soft (15-50%), adaptive (40/150/400)
# ============================================================================

def v5_simulate(entry, ticks, sig_ts, contracts, direction):
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

        # Underlying movement (available at every tick)
        u_move = 0.0
        has_underlying = False
        underlying_against = False
        underlying_confirms = False
        if entry_underlying and underlying and underlying > 0:
            has_underlying = True
            u_move = (underlying - entry_underlying) / entry_underlying * 100
            if is_call:
                underlying_against = u_move < -0.4
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > 0.4
                underlying_confirms = u_move < -0.2

        # EOD cutoff (always checked)
        if _eod_check(et_hour, et_min):
            return _make_exit(price, entry, contracts, elapsed, "eod_cutoff")

        # 5min minimum grace to avoid open noise
        if elapsed < 5:
            continue

        # ============================================================
        # DYNAMIC SCALP — no time window, uses underlying confirmation
        # ============================================================
        if peak_gain >= 20 and gain_pct > 0 and gain_pct < peak_gain * 0.6:
            # Premium peaked and faded — should we scalp or hold?
            if not underlying_confirms:
                # Underlying NOT confirming → IV-driven spike, take profit
                return _make_exit(price, entry, contracts, elapsed, "scalp_trail")
            # else: underlying confirms direction → HOLD, it's a real move

        # ============================================================
        # DYNAMIC CHECKPOINT — cut when BOTH signals agree trade is dead
        # ============================================================
        if drop_entry >= 15:
            if has_underlying and underlying_against:
                # Both premium AND underlying against → trade is dead
                return _make_exit(price, entry, contracts, elapsed, "checkpoint_cut")
            # Only premium down, underlying OK → could be IV crush, hold

        # ============================================================
        # DYNAMIC GRADUATED STOP — tight when underlying against, wide when not
        # ============================================================
        if underlying_against:
            # Underlying confirms against → tight stop (25%)
            if drop_entry >= 25:
                return _make_exit(price, entry, contracts, elapsed, "confirmed_stop")
        else:
            # Underlying not against → wide stop (40%) + absolute backstop (55%)
            if drop_entry >= 55:
                return _make_exit(price, entry, contracts, elapsed, "hard_stop")

        # ============================================================
        # SOFT TRAIL (15-50% peak gain band)
        # ============================================================
        if 15 <= peak_gain < 50:
            floor = entry + (peak - entry) * 0.50
            if price <= floor:
                return _make_exit(price, entry, contracts, elapsed, "soft_trail")

        # ============================================================
        # ADAPTIVE TRAIL (3 stages)
        # ============================================================
        if peak_gain >= 400:
            if drop_peak >= 30:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_moonshot")
        elif peak_gain >= 150:
            if drop_peak >= 45:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_runner")
        elif peak_gain >= 40:
            if drop_peak >= 40:
                return _make_exit(price, entry, contracts, elapsed, "adaptive_active")

        # ============================================================
        # THETA BLEED (120min+ and down 30%+)
        # ============================================================
        if elapsed >= 120 and drop_entry >= 30:
            return _make_exit(price, entry, contracts, elapsed, "theta_bleed")

    return _end_of_data(ticks, entry, contracts, sig_ts, peak)


# ============================================================================
# v5 parametric — test different thresholds
# ============================================================================

def v5_parametric(entry, ticks, sig_ts, contracts, direction,
                  checkpoint_drop=15, checkpoint_u=0.3,
                  tight_stop=25, wide_stop=40, backstop=55,
                  enable_recovery_check=False):
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    peak = entry
    entry_underlying = None
    recent_low = entry  # track for recovery check

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue
        price, elapsed, gain_pct, et_hour, et_min, underlying, ts_dt = parsed

        if price > peak:
            peak = price
        if price < recent_low:
            recent_low = price
        if entry_underlying is None and underlying and underlying > 0:
            entry_underlying = underlying

        peak_gain = (peak - entry) / entry * 100
        drop_entry = max(0, (entry - price) / entry * 100)
        drop_peak = (peak - price) / peak * 100 if peak > 0 else 0

        # Underlying state
        u_move = 0.0
        has_underlying = False
        underlying_against = False
        underlying_confirms = False
        if entry_underlying and underlying and underlying > 0:
            has_underlying = True
            u_move = (underlying - entry_underlying) / entry_underlying * 100
            if is_call:
                underlying_against = u_move < -checkpoint_u
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > checkpoint_u
                underlying_confirms = u_move < -0.2

        # Recovery check: is premium bouncing from recent low?
        recovering = False
        if enable_recovery_check and recent_low < entry and recent_low > 0:
            bounce_from_low = (price - recent_low) / recent_low * 100
            recovering = bounce_from_low > 5  # bounced 5%+ from low

        if _eod_check(et_hour, et_min):
            return _make_exit(price, entry, contracts, elapsed, "eod_cutoff")

        if elapsed < 5:
            continue

        # DYNAMIC SCALP
        if peak_gain >= 20 and gain_pct > 0 and gain_pct < peak_gain * 0.6:
            if not underlying_confirms:
                return _make_exit(price, entry, contracts, elapsed, "scalp_trail")

        # DYNAMIC CHECKPOINT — BOTH premium AND underlying against
        if drop_entry >= checkpoint_drop:
            if has_underlying and underlying_against:
                if enable_recovery_check and recovering:
                    pass  # bouncing — hold
                else:
                    return _make_exit(price, entry, contracts, elapsed, "checkpoint_cut")

        # DYNAMIC GRADUATED STOP
        if underlying_against:
            if drop_entry >= tight_stop:
                if enable_recovery_check and recovering:
                    pass  # bouncing — hold
                else:
                    return _make_exit(price, entry, contracts, elapsed, "confirmed_stop")
        else:
            if drop_entry >= backstop:
                return _make_exit(price, entry, contracts, elapsed, "hard_stop")

        # SOFT TRAIL (15-50%)
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


# ============================================================================
# Data loading + comparison
# ============================================================================

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

    # v5 variants with different patience/threshold configs
    def v5a(entry, ticks, sig_ts, contracts, direction):
        """v5a: Patient — raise thresholds (checkpoint -25%, tight stop -30%)"""
        return v5_parametric(entry, ticks, sig_ts, contracts, direction,
                             checkpoint_drop=25, checkpoint_u=0.4,
                             tight_stop=30, wide_stop=45, backstop=60)

    def v5b(entry, ticks, sig_ts, contracts, direction):
        """v5b: Very patient — only cut when truly dead (-30% AND underlying -0.5%)"""
        return v5_parametric(entry, ticks, sig_ts, contracts, direction,
                             checkpoint_drop=30, checkpoint_u=0.5,
                             tight_stop=35, wide_stop=50, backstop=65)

    def v5c(entry, ticks, sig_ts, contracts, direction):
        """v5c: Add recovery check — don't cut if premium bouncing from low"""
        return v5_parametric(entry, ticks, sig_ts, contracts, direction,
                             checkpoint_drop=20, checkpoint_u=0.3,
                             tight_stop=25, wide_stop=40, backstop=55,
                             enable_recovery_check=True)

    def v5d(entry, ticks, sig_ts, contracts, direction):
        """v5d: Patient + recovery check — best of both"""
        return v5_parametric(entry, ticks, sig_ts, contracts, direction,
                             checkpoint_drop=25, checkpoint_u=0.4,
                             tight_stop=30, wide_stop=45, backstop=60,
                             enable_recovery_check=True)

    strategies = {
        "v3": v3_simulate,
        "v5": v5_simulate,
        "v5a": v5a,
        "v5b": v5b,
        "v5c": v5c,
        "v5d": v5d,
    }

    # results[strat_name] = list of trade dicts
    all_results = {name: [] for name in strategies}
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

        if score >= 95:
            contracts = 5
        elif score >= 90:
            contracts = 4
        elif score >= 85:
            contracts = 3
        else:
            contracts = 1

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        # Overall peak for reference
        all_mids = [r[1] for r in rows if r[1] and r[1] > 0]
        overall_peak = max(all_mids) if all_mids else entry
        overall_peak_gain = (overall_peak - entry) / entry * 100

        et_hour = (sig_ts.hour - 4) % 24

        for name, sim_fn in strategies.items():
            pnl, reason, hold = sim_fn(entry, rows, sig_ts, contracts, direction)
            all_results[name].append({
                "ticker": ticker, "dir": direction, "day": day, "score": score,
                "entry": entry, "contracts": contracts, "pnl": pnl,
                "reason": reason, "hold": hold,
                "peak_gain": overall_peak_gain, "et_hour": et_hour,
            })

    # ===================================================================
    # REPORT
    # ===================================================================
    total_signals = len(all_results["v3"])
    print(f"{'=' * 100}")
    print(f"STRATEGY COMPARISON: v3 vs v4.1 vs v5 — {total_signals} signals")
    print(f"{'=' * 100}")
    print(f"Portfolio: ${PORTFOLIO:,}  |  Slippage: {SLIPPAGE*100:.0f}%  |  "
          f"No data: {no_data}  |  No strike: {no_strike}")

    # --- Summary table ---
    print(f"\n{'=' * 100}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 100}")
    print(f"\n{'Metric':<25} {'v3':>15} {'v4.1':>15} {'v5':>15} {'Best':>10}")
    print("-" * 85)

    summaries = {}
    for name in strategies:
        res = all_results[name]
        wins = [r for r in res if r["pnl"] > 0]
        losses = [r for r in res if r["pnl"] <= 0]
        total = sum(r["pnl"] for r in res)
        wr = len(wins) / len(res) * 100 if res else 0
        avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
        avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
        wl = abs(avg_w / avg_l) if avg_l else 0
        avg_hold = sum(r["hold"] for r in res) / len(res) if res else 0
        summaries[name] = {
            "total": total, "wr": wr, "wins": len(wins), "losses": len(losses),
            "avg_w": avg_w, "avg_l": avg_l, "wl": wl, "avg_hold": avg_hold,
        }

    def _best(metric, higher_better=True):
        vals = {n: summaries[n][metric] for n in strategies}
        return max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)

    metrics = [
        ("Total P&L", "total", True, "${:>+,.0f}"),
        ("Return on $8K", "total", True, None),  # special
        ("Win Rate", "wr", True, "{:.1f}%"),
        ("Wins / Losses", None, None, None),  # special
        ("Avg Win", "avg_w", True, "${:>+,.0f}"),
        ("Avg Loss", "avg_l", False, "${:>+,.0f}"),  # less negative is better
        ("Win:Loss Ratio", "wl", True, "{:.2f}:1"),
        ("Avg Hold (min)", "avg_hold", False, "{:.0f}m"),
    ]

    for label, key, higher, fmt in metrics:
        vals = []
        for name in strategies:
            s = summaries[name]
            if label == "Return on $8K":
                vals.append(f"{s['total']/PORTFOLIO*100:>+.1f}%")
            elif label == "Wins / Losses":
                vals.append(f"{s['wins']}W/{s['losses']}L")
            else:
                vals.append(fmt.format(s[key]))
        if key:
            best = _best(key, higher)
        elif label == "Wins / Losses":
            best = _best("wins", True)
        else:
            best = ""
        print(f"{label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15} {'<-- '+best:>10}")

    # --- Daily P&L comparison ---
    print(f"\n{'=' * 100}")
    print("DAILY P&L COMPARISON")
    print(f"{'=' * 100}")

    # Gather all days
    all_days = sorted(set(r["day"] for r in all_results["v3"]))

    print(f"\n{'Date':<12} {'Trades':>6}  "
          f"{'v3 P&L':>9} {'v3 WR':>6}  "
          f"{'v4.1 P&L':>9} {'v4.1 WR':>6}  "
          f"{'v5 P&L':>9} {'v5 WR':>6}  "
          f"{'Winner':>8}")
    print("-" * 100)

    cum = {n: 0 for n in strategies}
    day_wins = {n: 0 for n in strategies}

    for day in all_days:
        day_data = {}
        n_trades = 0
        for name in strategies:
            trades = [r for r in all_results[name] if r["day"] == day]
            n_trades = len(trades)
            pnl = sum(r["pnl"] for r in trades)
            wins = sum(1 for r in trades if r["pnl"] > 0)
            wr = wins / len(trades) * 100 if trades else 0
            day_data[name] = {"pnl": pnl, "wr": wr, "trades": len(trades)}
            cum[name] += pnl

        # Find daily winner
        best_day = max(strategies, key=lambda n: day_data[n]["pnl"])
        day_wins[best_day] += 1

        pnl_strs = []
        for name in strategies:
            d = day_data[name]
            marker = " *" if name == best_day else "  "
            pnl_strs.append(f"${d['pnl']:>+8.0f} {d['wr']:>4.0f}%{marker}")

        print(f"{day:<12} {n_trades:>6}  {'  '.join(pnl_strs)}")

    print("-" * 100)
    # Cumulative row
    cum_strs = []
    for name in strategies:
        cum_strs.append(f"${cum[name]:>+8.0f}       ")
    print(f"{'CUMULATIVE':<12} {'':>6}  {'  '.join(cum_strs)}")

    print(f"\nDays won:  ", end="")
    for name in strategies:
        print(f"  {name}: {day_wins[name]}/{len(all_days)}", end="")
    print()

    # --- Gate fire breakdown per strategy ---
    for name in strategies:
        print(f"\n{'=' * 100}")
        print(f"GATE FIRE BREAKDOWN — {name}")
        print(f"{'=' * 100}")

        gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "holds": []})
        for r in all_results[name]:
            g = gate_stats[r["reason"]]
            g["count"] += 1
            g["pnl"] += r["pnl"]
            if r["pnl"] > 0:
                g["wins"] += 1
            g["holds"].append(r["hold"])

        print(f"\n{'Gate':<22} {'Fires':>5} {'%':>5} {'P&L':>9} {'W/L':>8} {'WR':>5} "
              f"{'AvgHold':>7}")
        print("-" * 70)

        for gate, s in sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            ct = s["count"]
            pct = ct / len(all_results[name]) * 100
            wr = s["wins"] / ct * 100 if ct else 0
            ah = sum(s["holds"]) / ct if ct else 0
            print(f"{gate:<22} {ct:>5} {pct:>4.0f}% ${s['pnl']:>8.0f} "
                  f"{s['wins']}W/{ct - s['wins']}L {wr:>4.0f}% {ah:>5.0f}m")

    # --- Per-trade comparison (where strategies disagree) ---
    print(f"\n{'=' * 100}")
    print("TRADES WHERE STRATEGIES DISAGREE THE MOST")
    print(f"{'=' * 100}")
    print(f"\n{'#':<3} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} "
          f"{'v3 P&L':>8} {'v3 Gate':<18} "
          f"{'v4 P&L':>8} {'v4 Gate':<18} "
          f"{'v5 P&L':>8} {'v5 Gate':<18}")
    print("-" * 145)

    # Sort by max spread between strategies
    indices = list(range(total_signals))
    def _spread(i):
        pnls = [all_results[n][i]["pnl"] for n in strategies]
        return max(pnls) - min(pnls)
    indices.sort(key=_spread, reverse=True)

    for rank, i in enumerate(indices[:30]):
        r3 = all_results["v3"][i]
        r5 = all_results["v5"][i]
        r5b = all_results["v5b"][i]
        print(f"{rank+1:<3} {r3['ticker']:<7} {r3['dir'][:4]:<5} {r3['day']:<12} {r3['score']:>5} "
              f"${r3['entry']:>5.2f} "
              f"${r3['pnl']:>+7.0f} {r3['reason']:<18} "
              f"${r5['pnl']:>+7.0f} {r5['reason']:<18} "
              f"${r5b['pnl']:>+7.0f} {r5b['reason']:<18}")

    # --- All trades detail ---
    print(f"\n{'=' * 100}")
    print("ALL TRADES — v5 Detail")
    print(f"{'=' * 100}")
    print(f"\n{'#':<3} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} {'Ct':>3} "
          f"{'P&L':>8} {'Gate':<20} {'Hold':>5} {'Peak%':>6} "
          f"{'v3':>8} {'v4':>8}")
    print("-" * 120)

    for i, r5b in enumerate(all_results["v5b"]):
        r3 = all_results["v3"][i]
        r5 = all_results["v5"][i]
        diff3 = r5b["pnl"] - r3["pnl"]
        diff5 = r5b["pnl"] - r5["pnl"]
        marker = ""
        if r5b["pnl"] > 500:
            marker = " ***"
        elif r5b["pnl"] < -300:
            marker = " !!!"
        print(f"{i+1:<3} {r5b['ticker']:<7} {r5b['dir'][:4]:<5} {r5b['day']:<12} {r5b['score']:>5} "
              f"${r5b['entry']:>5.2f} {r5b['contracts']:>3} ${r5b['pnl']:>+7.0f} "
              f"{r5b['reason']:<20} {r5b['hold']:>4.0f}m {r5b['peak_gain']:>+5.0f}% "
              f"${diff3:>+7.0f} ${diff5:>+7.0f}{marker}")


if __name__ == "__main__":
    main()
