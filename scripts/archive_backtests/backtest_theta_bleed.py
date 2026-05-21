#!/usr/bin/env python3
"""Backtest theta-bleed gate variants for 0DTE options.

The problem: 0DTE options lose premium to theta decay even when the underlying
hasn't moved. The current system doesn't catch this because:
1. Hard stop is at -50% (mid-range, when underlying is flat)
2. Theta timer is score-immune (score >= 92 bypasses it)

This script tests a new "theta_bleed_0dte" gate:
  IF 0DTE AND premium down X% AND underlying moved < Y% AND held > Z minutes
  THEN exit — theta is killing the trade, not price action.

Variants tested:
  A: -20% drop, ±0.3% underlying, 15min hold
  B: -20% drop, ±0.5% underlying, 20min hold
  C: -25% drop, ±0.3% underlying, 15min hold
  D: -15% drop, ±0.3% underlying, 20min hold (aggressive)
  E: -20% drop, ±0.3% underlying, 15min hold, tighter after 2PM (-15%)
  F: Time-scaled: -25% before noon, -20% noon-2PM, -15% after 2PM
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SLIPPAGE = 0.15
PORTFOLIO = 8000

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

# Import the production simulator
sys.path.insert(0, ".")

# ── Production constants (from backtest_prod.py) ─────────────────────────
HIGH_VOL_TICKERS = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
GRACE_PERIOD_SEC = 300
TRAIL_ACTIVATE_GAIN_PCT = 35.0
SOFT_TRAIL_BAND_LOW = 10.0
SOFT_TRAIL_BAND_HIGH = 35.0
SOFT_TRAIL_FLOOR_FRACTION = 0.60
INDEX_PROFIT_TARGET_PCT = 30.0
EOD_CUTOFF_MINUTES = 15.0
BAR1_MIN_SEC = 90
BAR1_WINDOW_SEC = 150
BAR1_THRESHOLD_PCT = -5.0
HARD_STOP_PCT = 0.30

TRAIL_TIERS = [
    (400.0, 0.20), (200.0, 0.25), (100.0, 0.30), (50.0, 0.35),
]
TRAIL_MULT_TICKERS = {"NVDA": 1.5, "TSLA": 1.5, "AMZN": 1.4, "AVGO": 1.4, "PLTR": 1.3}
HOUSE_MONEY_FLOORS = [(5.00, 2.00), (2.00, 0.80), (1.00, 0.30)]
THETA_TIMER_SCORE_IMMUNE = 92.0
THETA_TIMER_TICKER_IMMUNE = {"NVDA", "TSLA", "AMZN", "AVGO", "PLTR"}


def _parse_tick(tick, sig_ts, entry):
    ts, mid, bid, ask, underlying = tick
    ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
    price = mid if mid and mid > 0 else ((bid + ask) / 2 if bid and ask else 0)
    if price <= 0:
        return None
    elapsed_sec = (ts_dt - sig_ts).total_seconds()
    elapsed_min = elapsed_sec / 60
    gain_pct = (price - entry) / entry * 100
    et_dt = ts_dt - timedelta(hours=4)
    return {
        "price": price, "bid": bid if bid else 0, "ask": ask if ask else 0,
        "underlying": underlying if underlying else 0,
        "elapsed_sec": elapsed_sec, "elapsed_min": elapsed_min,
        "gain_pct": gain_pct,
        "et_hour": et_dt.hour, "et_min": et_dt.minute,
        "minutes_to_close": max(0, 960 - (et_dt.hour * 60 + et_dt.minute)),
        "ts_dt": ts_dt, "et_dt": et_dt,
    }


def _make_exit(price, entry, contracts, elapsed_min, reason):
    pnl = (price - entry) * contracts * 100
    if pnl > 0:
        pnl *= (1 - SLIPPAGE)
    return pnl, reason, elapsed_min


def _end_of_data(ticks, entry, contracts, sig_ts):
    for t in reversed(ticks):
        price = t[1] if t[1] and t[1] > 0 else 0
        if price > 0:
            pnl = (price - entry) * contracts * 100
            if pnl > 0:
                pnl *= (1 - SLIPPAGE)
            ts_dt = datetime.fromisoformat(t[0]) if isinstance(t[0], str) else t[0]
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            return pnl, "eod_data_end", (ts_dt - sig_ts).total_seconds() / 60
    return 0, "no_data", 0


def _get_trail_pct(gain_pct):
    for min_gain, trail in TRAIL_TIERS:
        if gain_pct >= min_gain:
            return trail
    return TRAIL_TIERS[-1][1]


def _theta_curve_mult(et_hour, et_min):
    current_hour = et_hour + et_min / 60.0
    hours_remaining = max(0.5, 16.0 - current_hour)
    raw = (hours_remaining / 6.5) ** 0.4
    return max(0.40, min(raw, 1.0))


def _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min):
    peak_gain_pct = (peak - entry) / entry * 100 if entry > 0 else 0
    tier_trail = _get_trail_pct(peak_gain_pct)
    mult = 1.0
    if ticker in TRAIL_MULT_TICKERS:
        mult *= TRAIL_MULT_TICKERS[ticker]
    if is_morning:
        mult *= 1.5
    if score is not None and score >= 90:
        mult *= 1.35
    mult = min(mult, 2.0)
    giveback_mult = min(mult, 1.20)
    multiplied = min(tier_trail * giveback_mult, 0.45)
    theta_mult = _theta_curve_mult(et_hour, et_min)
    effective = multiplied * theta_mult
    return peak * (1.0 - effective)


def _compute_house_money_floor(peak_gain_pct, entry, current_floor):
    peak_gain_frac = peak_gain_pct / 100.0
    new_floor = current_floor
    for trigger, floor_gain in HOUSE_MONEY_FLOORS:
        if peak_gain_frac >= trigger:
            candidate = entry * (1.0 + floor_gain)
            if candidate > new_floor:
                new_floor = candidate
            break
    return new_floor


# ── Theta bleed checker ──────────────────────────────────────────────────

def _check_theta_bleed_0dte(gain_pct, u_move_abs, elapsed_min, et_hour,
                             drop_threshold, underlying_threshold, min_hold,
                             time_scaled=False):
    """Check if theta is silently killing a 0DTE trade.

    Returns True if we should exit.
    """
    if elapsed_min < min_hold:
        return False

    actual_threshold = drop_threshold
    if time_scaled:
        if et_hour >= 14:
            actual_threshold = -15.0  # aggressive after 2 PM
        elif et_hour >= 12:
            actual_threshold = -20.0  # moderate noon-2 PM
        else:
            actual_threshold = -25.0  # patient before noon

    if gain_pct <= actual_threshold and u_move_abs <= underlying_threshold:
        return True
    return False


# ── Main simulator with theta bleed variant ──────────────────────────────

def simulate_with_theta_bleed(entry, ticks, sig_ts, contracts, direction, ticker,
                               score, expiry_date, variant):
    """Production v4 FSM + theta bleed 0DTE gate."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    is_index = ticker in INDEX_TICKERS
    is_high_vol = ticker in HIGH_VOL_TICKERS

    sig_date = sig_ts.date()
    dte = 0
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            pass
    is_multiday = dte > 0

    et_entry_hour = (sig_ts.hour - 4) % 24
    is_morning = et_entry_hour < 12

    peak = entry
    entry_underlying = None
    last_underlying = 0.0
    house_money_floor_price = 0.0
    locked_milestones = set()
    seconds_at_zero_bid = 0.0

    # Variant params
    variants = {
        "baseline": None,  # no theta bleed gate
        "A": {"drop": -20, "u_thresh": 0.3, "min_hold": 15, "time_scaled": False},
        "B": {"drop": -20, "u_thresh": 0.5, "min_hold": 20, "time_scaled": False},
        "C": {"drop": -25, "u_thresh": 0.3, "min_hold": 15, "time_scaled": False},
        "D": {"drop": -15, "u_thresh": 0.3, "min_hold": 20, "time_scaled": False},
        "E": {"drop": -20, "u_thresh": 0.3, "min_hold": 15, "time_scaled": False,
              "afternoon_drop": -15},
        "F": {"drop": -25, "u_thresh": 0.3, "min_hold": 15, "time_scaled": True},
    }
    v = variants.get(variant)

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue

        price = parsed["price"]
        bid = parsed["bid"]
        ask = parsed["ask"]
        underlying = parsed["underlying"]
        elapsed_sec = parsed["elapsed_sec"]
        elapsed_min = parsed["elapsed_min"]
        gain_pct = parsed["gain_pct"]
        et_hour = parsed["et_hour"]
        et_min = parsed["et_min"]
        minutes_to_close = parsed["minutes_to_close"]

        if price > peak:
            peak = price
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying
        if underlying > 0:
            last_underlying = underlying
        effective_underlying = underlying if underlying > 0 else last_underlying

        peak_gain_pct = (peak - entry) / entry * 100
        drop_entry_pct = max(0, (entry - price) / entry * 100)

        u_move = 0.0
        has_underlying = False
        underlying_against = False
        underlying_confirms = False
        u_move_abs = 999.0  # large default = don't trigger theta bleed if no data
        if entry_underlying and entry_underlying > 0 and effective_underlying > 0:
            has_underlying = True
            u_move = (effective_underlying - entry_underlying) / entry_underlying * 100
            u_move_abs = abs(u_move)
            if is_call:
                underlying_against = u_move < -0.5
                underlying_confirms = u_move > 0.2
            else:
                underlying_against = u_move > 0.5
                underlying_confirms = u_move < -0.2

        if bid <= 0:
            seconds_at_zero_bid += 60
        else:
            seconds_at_zero_bid = 0

        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = (parsed["ts_dt"] - timedelta(hours=4)).date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass
        current_multiday = current_dte > 0

        # FSM State
        if elapsed_sec < GRACE_PERIOD_SEC:
            fsm_state = "GRACE"
        elif peak_gain_pct >= TRAIL_ACTIVATE_GAIN_PCT:
            fsm_state = "TRAILING"
        else:
            fsm_state = "DEVELOPING"

        # EOD cutoff (0DTE only)
        if not current_multiday and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # GRACE
        if fsm_state == "GRACE":
            if not is_multiday:
                if BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                    bar1_change = (price - entry) / entry * 100
                    if bar1_change <= BAR1_THRESHOLD_PCT:
                        return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")
            continue

        # Bid disappearance
        if bid <= 0 and seconds_at_zero_bid >= 30:
            return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

        # Index profit target
        if is_index and INDEX_PROFIT_TARGET_PCT > 0 and gain_pct >= INDEX_PROFIT_TARGET_PCT:
            return _make_exit(price, entry, contracts, elapsed_min, "profit_target")

        # Scalp trail
        if peak_gain_pct >= 20 and gain_pct > 0 and gain_pct < peak_gain_pct * 0.6:
            should_scalp = False
            if not current_multiday and has_underlying and not underlying_confirms:
                should_scalp = True
            elif current_multiday and has_underlying and underlying_against:
                should_scalp = True
            if should_scalp:
                return _make_exit(price, entry, contracts, elapsed_min, "scalp_trail")

        # Checkpoint cut (0DTE only)
        if not current_multiday and drop_entry_pct >= 30 and has_underlying and underlying_against:
            return _make_exit(price, entry, contracts, elapsed_min, "checkpoint_cut")

        # ── NEW: Theta bleed 0DTE gate ──
        if v is not None and not current_multiday and has_underlying:
            drop_thresh = v["drop"]
            if variant == "E" and et_hour >= 14:
                drop_thresh = v.get("afternoon_drop", drop_thresh)

            if _check_theta_bleed_0dte(
                gain_pct, u_move_abs, elapsed_min, et_hour,
                drop_thresh, v["u_thresh"], v["min_hold"],
                time_scaled=v.get("time_scaled", False),
            ):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_bleed_0dte")

        # Graduated stops
        if has_underlying:
            if is_high_vol:
                tight_stop = 0.45 if not current_multiday else 0.60
                backstop = 0.75 if not current_multiday else 0.85
            else:
                tight_stop = 0.35 if not current_multiday else 0.52
                backstop = 0.65 if not current_multiday else 0.75

            if underlying_against:
                if drop_entry_pct >= tight_stop * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "confirmed_stop")
            else:
                mid_stop = (tight_stop + backstop) / 2
                if drop_entry_pct >= mid_stop * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "mid_range_stop")
                if drop_entry_pct >= backstop * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "backstop")
        else:
            stop_price = entry * (1.0 - HARD_STOP_PCT)
            if minutes_to_close > 30 and ask > 0 and bid >= 0:
                compare = (bid + ask) / 2.0
            else:
                compare = bid if bid > 0 else price
            if compare <= stop_price and compare >= 0:
                return _make_exit(price, entry, contracts, elapsed_min, "hard_stop")

        # DEVELOPING
        if fsm_state == "DEVELOPING":
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")

            # Theta timer (not score-immune in this variant — handled by theta_bleed_0dte)
            if current_dte == 0:
                if score is not None and score >= THETA_TIMER_SCORE_IMMUNE:
                    pass  # score immune — but theta_bleed_0dte above handles this
                elif ticker not in THETA_TIMER_TICKER_IMMUNE:
                    current_hour = et_hour + et_min / 60.0
                    if is_morning:
                        timer_sec = 10800
                    elif current_hour >= 14.0:
                        timer_sec = 2400
                    else:
                        timer_sec = 7200
                    if elapsed_sec >= timer_sec and gain_pct <= 5.0:
                        return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        # TRAILING
        house_money_floor_price = _compute_house_money_floor(peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        trail_stop = _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        for ms_gain, ms_frac in [(200.0, 0.15), (400.0, 0.15), (600.0, 0.15)]:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    contracts -= n_close
                    if contracts <= 0:
                        partial_pnl = (price - entry) * n_close * 100
                        if partial_pnl > 0:
                            partial_pnl *= (1 - SLIPPAGE)
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                else:
                    locked_milestones.add(ms_gain)

        if current_dte == 0:
            if score is not None and score >= THETA_TIMER_SCORE_IMMUNE:
                pass
            elif ticker not in THETA_TIMER_TICKER_IMMUNE:
                current_hour = et_hour + et_min / 60.0
                if is_morning:
                    timer_sec = 10800
                elif current_hour >= 14.0:
                    timer_sec = 2400
                else:
                    timer_sec = 7200
                if elapsed_sec >= timer_sec and gain_pct <= 5.0:
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


def build_ct(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    ds = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    si = int(strike * 1000)
    return f"O:{ticker}{ds}{cp}{si:08d}"


def score_to_contracts(score, entry_premium):
    if score >= 95:
        budget_mult = 1.0
    elif score >= 90:
        budget_mult = 0.75
    elif score >= 85:
        budget_mult = 0.50
    elif score >= 78:
        budget_mult = 0.25
    else:
        return 0
    deployable = PORTFOLIO * 0.75
    target = deployable / 5 * budget_mult
    cost = entry_premium * 100
    if cost <= 0:
        return 1
    pos_cap = int((PORTFOLIO * 0.15) / cost)
    raw = int(target / cost)
    return max(1, min(raw, max(1, pos_cap)))


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

    variant_names = ["baseline", "A", "B", "C", "D", "E", "F"]
    all_results = {v: [] for v in variant_names}
    no_data = no_strike = skipped = 0

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
        if score < 78:
            skipped += 1
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

        contracts = score_to_contracts(score, entry)
        if contracts <= 0:
            skipped += 1
            continue

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        all_mids = [r[1] for r in rows if r[1] and r[1] > 0]
        overall_peak = max(all_mids) if all_mids else entry
        overall_peak_gain = (overall_peak - entry) / entry * 100

        for variant in variant_names:
            pnl, reason, hold = simulate_with_theta_bleed(
                entry, rows, sig_ts, contracts, direction, ticker, score,
                expiry_date=day, variant=variant,
            )
            all_results[variant].append({
                "id": sig["id"], "ticker": ticker, "dir": direction,
                "day": day, "score": score, "entry": entry,
                "contracts": contracts, "pnl": pnl, "reason": reason,
                "hold": hold, "peak_gain": overall_peak_gain,
            })

    # ===================================================================
    # REPORT
    # ===================================================================
    total_signals = len(all_results["baseline"])

    print(f"{'=' * 120}")
    print(f"THETA BLEED 0DTE GATE BACKTEST — {total_signals} signals")
    print(f"{'=' * 120}")
    print(f"No data: {no_data}  |  No strike: {no_strike}  |  Score<78: {skipped}")
    print()
    print("Variants:")
    print("  baseline: No theta bleed gate (current production)")
    print("  A: -20% drop, underlying ±0.3%, 15min hold")
    print("  B: -20% drop, underlying ±0.5%, 20min hold")
    print("  C: -25% drop, underlying ±0.3%, 15min hold")
    print("  D: -15% drop, underlying ±0.3%, 20min hold (aggressive)")
    print("  E: -20% drop (±0.3%, 15min), tighter -15% after 2PM")
    print("  F: Time-scaled: -25% before noon, -20% noon-2PM, -15% after 2PM")

    # Summary table
    print(f"\n{'=' * 120}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 120}")
    header = f"{'Metric':<20}"
    for v in variant_names:
        header += f" {v:>12}"
    print(header)
    print("-" * (20 + 13 * len(variant_names)))

    summaries = {}
    for v in variant_names:
        res = all_results[v]
        wins = [r for r in res if r["pnl"] > 0]
        losses = [r for r in res if r["pnl"] <= 0]
        total = sum(r["pnl"] for r in res)
        wr = len(wins) / len(res) * 100 if res else 0
        avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
        avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
        wl = abs(avg_w / avg_l) if avg_l else 0
        tb_fires = sum(1 for r in res if r["reason"] == "theta_bleed_0dte")
        tb_pnl = sum(r["pnl"] for r in res if r["reason"] == "theta_bleed_0dte")
        summaries[v] = {
            "total": total, "wr": wr, "wins": len(wins), "losses": len(losses),
            "avg_w": avg_w, "avg_l": avg_l, "wl": wl, "tb_fires": tb_fires,
            "tb_pnl": tb_pnl,
        }

    def _row(label, key, fmt):
        line = f"{label:<20}"
        for v in variant_names:
            line += f" {fmt.format(summaries[v][key]):>12}"
        print(line)

    _row("Total P&L", "total", "${:>+,.0f}")

    line = f"{'Return %':<20}"
    for v in variant_names:
        line += f" {summaries[v]['total']/PORTFOLIO*100:>+11.1f}%"
    print(line)

    line = f"{'Win Rate':<20}"
    for v in variant_names:
        line += f" {summaries[v]['wr']:>11.1f}%"
    print(line)

    line = f"{'W / L':<20}"
    for v in variant_names:
        s = summaries[v]
        line += f" {s['wins']}W/{s['losses']}L".rjust(13)
    print(line)

    _row("Avg Win", "avg_w", "${:>+,.0f}")
    _row("Avg Loss", "avg_l", "${:>+,.0f}")

    line = f"{'Win:Loss Ratio':<20}"
    for v in variant_names:
        line += f" {summaries[v]['wl']:>10.2f}:1"
    print(line)

    print()
    _row("TB Fires", "tb_fires", "{}")
    _row("TB P&L Impact", "tb_pnl", "${:>+,.0f}")

    line = f"{'vs Baseline':<20}"
    base_pnl = summaries["baseline"]["total"]
    for v in variant_names:
        diff = summaries[v]["total"] - base_pnl
        line += f" {f'${diff:>+,.0f}':>12}"
    print(line)

    # ── Trades where theta_bleed_0dte fires (per variant) ──
    for v in variant_names:
        if v == "baseline":
            continue
        tb_trades = [(i, r) for i, r in enumerate(all_results[v])
                     if r["reason"] == "theta_bleed_0dte"]
        if not tb_trades:
            continue

        print(f"\n{'=' * 120}")
        print(f"THETA BLEED FIRES — Variant {v}")
        print(f"{'=' * 120}")
        print(f"{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} {'Ct':>3} "
              f"{'TB P&L':>9} {'Hold':>6} {'Peak%':>7} {'Base P&L':>9} {'Base Gate':<20} {'Saved':>8}")
        print("-" * 120)

        total_saved = 0
        for i, r in tb_trades:
            base_r = all_results["baseline"][i]
            saved = r["pnl"] - base_r["pnl"]
            total_saved += saved
            print(f"{i+1:<4} {r['ticker']:<7} {r['dir'][:4]:<5} {r['day']:<12} {r['score']:>5} "
                  f"${r['entry']:>5.2f} {r['contracts']:>3} ${r['pnl']:>+8.0f} "
                  f"{r['hold']:>5.0f}m {r['peak_gain']:>+6.0f}% "
                  f"${base_r['pnl']:>+8.0f} {base_r['reason']:<20} ${saved:>+7.0f}")

        print(f"\nTotal theta_bleed fires: {len(tb_trades)}, "
              f"Net saved vs baseline: ${total_saved:>+,.0f}")

    # ── Full trade comparison for best variant ──
    best = max(variant_names, key=lambda v: summaries[v]["total"])
    print(f"\n{'=' * 120}")
    print(f"BEST VARIANT: {best} (${summaries[best]['total']:>+,.0f})")
    print(f"{'=' * 120}")

    # Show trades where best differs from baseline
    diffs = []
    for i in range(total_signals):
        r_best = all_results[best][i]
        r_base = all_results["baseline"][i]
        if r_best["reason"] != r_base["reason"]:
            diffs.append((i, r_best, r_base))

    if diffs:
        print(f"\nTrades where {best} differs from baseline:")
        print(f"{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} "
              f"{best+' P&L':>9} {best+' Gate':<20} "
              f"{'Base P&L':>9} {'Base Gate':<20} {'Delta':>8}")
        print("-" * 110)

        for i, r_best, r_base in diffs:
            delta = r_best["pnl"] - r_base["pnl"]
            print(f"{i+1:<4} {r_best['ticker']:<7} {r_best['dir'][:4]:<5} {r_best['day']:<12} "
                  f"${r_best['pnl']:>+8.0f} {r_best['reason']:<20} "
                  f"${r_base['pnl']:>+8.0f} {r_base['reason']:<20} ${delta:>+7.0f}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
