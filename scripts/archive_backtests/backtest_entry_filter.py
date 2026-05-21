#!/usr/bin/env python3
"""Backtest entry filtering for late-session 0DTE trades.

The exits are fine — the problem is entering bad trades. These variants
test smarter entry filtering to avoid late-session 0DTE losers.

Variants:
  baseline:      Current production (no entry filter)
  no_0dte_2pm:   Reject 0DTE entries after 2PM ET
  no_0dte_1pm:   Reject 0DTE entries after 1PM ET
  no_0dte_noon:  Reject 0DTE entries after 12PM ET
  score_gate:    Late 0DTE needs higher score: after 1PM require 95+, after 2PM reject
  half_late:     Half position size after 1PM, 1 contract after 2PM (0DTE only)
  index_only:    After 1PM, only allow index tickers (SPY/QQQ/IWM) for 0DTE
  no_late_puts:  Reject 0DTE puts after 1PM (puts bleed faster from theta)
  score_penalty: Subtract 10 from score after 1PM, 20 after 2PM (0DTE only)
  bar1_fix:      Just the bar1 multi-day fix (no other changes) — true baseline
  bar1_no2pm:    bar1 fix + reject 0DTE after 2PM
  bar1_score:    bar1 fix + score gate for late 0DTE
  bar1_half:     bar1 fix + half size late 0DTE
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SLIPPAGE = 0.15
PORTFOLIO = 8000

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

# Production constants
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
TRAIL_TIERS = [(400.0, 0.20), (200.0, 0.25), (100.0, 0.30), (50.0, 0.35)]
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
    et_dt = ts_dt - timedelta(hours=4)
    return {
        "price": price, "bid": bid if bid else 0, "ask": ask if ask else 0,
        "underlying": underlying if underlying else 0,
        "elapsed_sec": elapsed_sec, "elapsed_min": elapsed_sec / 60,
        "gain_pct": (price - entry) / entry * 100,
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
    return max(0.40, min((hours_remaining / 6.5) ** 0.4, 1.0))


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
    multiplied = min(tier_trail * min(mult, 1.20), 0.45)
    effective = multiplied * _theta_curve_mult(et_hour, et_min)
    return peak * (1.0 - effective)


def _compute_house_money_floor(peak_gain_pct, entry, current_floor):
    peak_gain_frac = peak_gain_pct / 100.0
    for trigger, floor_gain in HOUSE_MONEY_FLOORS:
        if peak_gain_frac >= trigger:
            candidate = entry * (1.0 + floor_gain)
            if candidate > current_floor:
                return candidate
            break
    return current_floor


def simulate(entry, ticks, sig_ts, contracts, direction, ticker, score,
             expiry_date, disable_bar1_multiday=False):
    """Production v4 FSM simulator with optional bar1 multi-day fix."""
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
        if entry_underlying and entry_underlying > 0 and effective_underlying > 0:
            has_underlying = True
            u_move = (effective_underlying - entry_underlying) / entry_underlying * 100
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
                exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = (parsed["ts_dt"] - timedelta(hours=4)).date()
                current_dte = max(0, (exp_dt - tick_date).days)
            except (ValueError, TypeError):
                pass
        current_multiday = current_dte > 0

        if elapsed_sec < GRACE_PERIOD_SEC:
            fsm_state = "GRACE"
        elif peak_gain_pct >= TRAIL_ACTIVATE_GAIN_PCT:
            fsm_state = "TRAILING"
        else:
            fsm_state = "DEVELOPING"

        if not current_multiday and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        if fsm_state == "GRACE":
            # Bar1: skip for multi-day if fix enabled
            skip_bar1 = disable_bar1_multiday and is_multiday
            if not skip_bar1:
                if BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                    bar1_change = (price - entry) / entry * 100
                    if bar1_change <= BAR1_THRESHOLD_PCT:
                        return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")
            continue

        if bid <= 0 and seconds_at_zero_bid >= 30:
            return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

        if is_index and INDEX_PROFIT_TARGET_PCT > 0 and gain_pct >= INDEX_PROFIT_TARGET_PCT:
            return _make_exit(price, entry, contracts, elapsed_min, "profit_target")

        if peak_gain_pct >= 20 and gain_pct > 0 and gain_pct < peak_gain_pct * 0.6:
            should_scalp = False
            if not current_multiday and has_underlying and not underlying_confirms:
                should_scalp = True
            elif current_multiday and has_underlying and underlying_against:
                should_scalp = True
            if should_scalp:
                return _make_exit(price, entry, contracts, elapsed_min, "scalp_trail")

        if not current_multiday and drop_entry_pct >= 30 and has_underlying and underlying_against:
            return _make_exit(price, entry, contracts, elapsed_min, "checkpoint_cut")

        if has_underlying:
            if is_high_vol:
                tight_stop = 0.45 if not current_multiday else 0.60
                backstop = 0.75 if not current_multiday else 0.85
            else:
                tight_stop = 0.35 if not current_multiday else 0.52
                backstop = 0.65 if not current_multiday else 0.75
            mid_stop = (tight_stop + backstop) / 2

            if underlying_against:
                if drop_entry_pct >= tight_stop * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "confirmed_stop")
            else:
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

        if fsm_state == "DEVELOPING":
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")
            if current_dte == 0:
                if score is not None and score >= THETA_TIMER_SCORE_IMMUNE:
                    pass
                elif ticker not in THETA_TIMER_TICKER_IMMUNE:
                    current_hour_f = et_hour + et_min / 60.0
                    if is_morning:
                        timer_sec = 10800
                    elif current_hour_f >= 14.0:
                        timer_sec = 2400
                    else:
                        timer_sec = 7200
                    if elapsed_sec >= timer_sec and gain_pct <= 5.0:
                        return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

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
                current_hour_f = et_hour + et_min / 60.0
                if is_morning:
                    timer_sec = 10800
                elif current_hour_f >= 14.0:
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
        bm = 1.0
    elif score >= 90:
        bm = 0.75
    elif score >= 85:
        bm = 0.50
    elif score >= 78:
        bm = 0.25
    else:
        return 0
    deployable = PORTFOLIO * 0.75
    target = deployable / 5 * bm
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

    variant_names = [
        "baseline", "bar1_fix",
        "no_0dte_2pm", "no_0dte_1pm", "no_0dte_noon",
        "score_gate", "half_late", "index_only", "no_late_puts",
        "bar1_no2pm", "bar1_score", "bar1_half",
    ]
    variant_labels = {
        "baseline":     "Baseline",
        "bar1_fix":     "Bar1 fix",
        "no_0dte_2pm":  "No 0DTE>2PM",
        "no_0dte_1pm":  "No 0DTE>1PM",
        "no_0dte_noon": "No 0DTE>noon",
        "score_gate":   "Score gate",
        "half_late":    "Half late",
        "index_only":   "Index >1PM",
        "no_late_puts": "No puts>1PM",
        "bar1_no2pm":   "B1+No>2PM",
        "bar1_score":   "B1+ScoreGate",
        "bar1_half":    "B1+HalfLate",
    }

    all_results = {v: [] for v in variant_names}
    no_data = no_strike = skipped_score = 0

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
            skipped_score += 1
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

        base_contracts = score_to_contracts(score, entry)
        if base_contracts <= 0:
            skipped_score += 1
            continue

        sig_ts = datetime.fromisoformat(sig["sig_ts"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        et_entry_hour = (sig_ts.hour - 4) % 24
        is_0dte = True  # all signals use same-day expiry in build_ct
        is_call = direction.lower() in ("call", "bullish", "long")
        is_index = ticker in INDEX_TICKERS

        all_mids = [r[1] for r in rows if r[1] and r[1] > 0]
        overall_peak = max(all_mids) if all_mids else entry
        overall_peak_gain = (overall_peak - entry) / entry * 100

        for variant in variant_names:
            # ── Entry filtering ──
            rejected = False
            contracts = base_contracts
            disable_bar1 = variant in ("bar1_fix", "bar1_no2pm", "bar1_score", "bar1_half")

            if variant == "no_0dte_2pm" and is_0dte and et_entry_hour >= 14:
                rejected = True
            elif variant == "no_0dte_1pm" and is_0dte and et_entry_hour >= 13:
                rejected = True
            elif variant == "no_0dte_noon" and is_0dte and et_entry_hour >= 12:
                rejected = True
            elif variant == "score_gate" and is_0dte:
                if et_entry_hour >= 14:
                    rejected = True  # reject all 0DTE after 2PM
                elif et_entry_hour >= 13 and score < 95:
                    rejected = True  # require 95+ after 1PM
            elif variant == "half_late" and is_0dte:
                if et_entry_hour >= 14:
                    contracts = 1  # minimum size after 2PM
                elif et_entry_hour >= 13:
                    contracts = max(1, base_contracts // 2)  # half after 1PM
            elif variant == "index_only" and is_0dte and et_entry_hour >= 13:
                if not is_index:
                    rejected = True  # only index tickers after 1PM
            elif variant == "no_late_puts" and is_0dte and et_entry_hour >= 13:
                if not is_call:
                    rejected = True  # no puts after 1PM
            elif variant == "bar1_no2pm" and is_0dte and et_entry_hour >= 14:
                rejected = True
            elif variant == "bar1_score" and is_0dte:
                if et_entry_hour >= 14:
                    rejected = True
                elif et_entry_hour >= 13 and score < 95:
                    rejected = True
            elif variant == "bar1_half" and is_0dte:
                if et_entry_hour >= 14:
                    contracts = 1
                elif et_entry_hour >= 13:
                    contracts = max(1, base_contracts // 2)

            if rejected:
                all_results[variant].append({
                    "id": sig["id"], "ticker": ticker, "dir": direction,
                    "day": day, "score": score, "entry": entry,
                    "contracts": 0, "pnl": 0, "reason": "rejected",
                    "hold": 0, "peak_gain": overall_peak_gain,
                    "et_entry_hour": et_entry_hour,
                })
                continue

            pnl, reason, hold = simulate(
                entry, rows, sig_ts, contracts, direction, ticker, score,
                expiry_date=day, disable_bar1_multiday=disable_bar1,
            )

            all_results[variant].append({
                "id": sig["id"], "ticker": ticker, "dir": direction,
                "day": day, "score": score, "entry": entry,
                "contracts": contracts, "pnl": pnl, "reason": reason,
                "hold": hold, "peak_gain": overall_peak_gain,
                "et_entry_hour": et_entry_hour,
            })

    # ===================================================================
    # REPORT
    # ===================================================================
    total_signals = len(all_results["baseline"])

    print(f"{'=' * 160}")
    print(f"ENTRY FILTERING BACKTEST — {total_signals} signals")
    print(f"{'=' * 160}")
    print(f"No data: {no_data}  |  No strike: {no_strike}  |  Score<78: {skipped_score}")
    print()
    print("Entry filter variants:")
    print("  baseline:     Current production (bar1 fires on all, no entry filter)")
    print("  bar1_fix:     Bar1 disabled for multi-day only (deployed today)")
    print("  no_0dte_2pm:  Reject 0DTE after 2PM")
    print("  no_0dte_1pm:  Reject 0DTE after 1PM")
    print("  no_0dte_noon: Reject 0DTE after noon")
    print("  score_gate:   After 1PM require 95+, after 2PM reject (0DTE only)")
    print("  half_late:    Half contracts after 1PM, 1 contract after 2PM (0DTE)")
    print("  index_only:   Only index tickers (SPY/QQQ/IWM) after 1PM for 0DTE")
    print("  no_late_puts: No 0DTE puts after 1PM")
    print("  bar1_no2pm:   bar1 fix + reject 0DTE after 2PM")
    print("  bar1_score:   bar1 fix + score gate")
    print("  bar1_half:    bar1 fix + half size late")

    # Summary
    print(f"\n{'=' * 160}")
    print("SUMMARY")
    print(f"{'=' * 160}")

    summaries = {}
    for v in variant_names:
        res = all_results[v]
        traded = [r for r in res if r["reason"] != "rejected"]
        wins = [r for r in traded if r["pnl"] > 0]
        losses = [r for r in traded if r["pnl"] <= 0]
        total = sum(r["pnl"] for r in traded)
        wr = len(wins) / len(traded) * 100 if traded else 0
        avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
        avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
        wl = abs(avg_w / avg_l) if avg_l else 0
        n_rejected = sum(1 for r in res if r["reason"] == "rejected")
        summaries[v] = {
            "total": total, "wr": wr, "wins": len(wins), "losses": len(losses),
            "avg_w": avg_w, "avg_l": avg_l, "wl": wl,
            "traded": len(traded), "rejected": n_rejected,
        }

    # Print in groups to fit terminal
    groups = [
        ["baseline", "bar1_fix", "no_0dte_2pm", "no_0dte_1pm", "no_0dte_noon", "score_gate"],
        ["baseline", "half_late", "index_only", "no_late_puts", "bar1_no2pm", "bar1_score", "bar1_half"],
    ]

    for gi, group in enumerate(groups):
        if gi > 0:
            print()

        header = f"{'Metric':<18}"
        for v in group:
            header += f" {variant_labels[v]:>14}"
        print(header)
        print("-" * (18 + 15 * len(group)))

        line = f"{'Total P&L':<18}"
        for v in group:
            line += f" ${summaries[v]['total']:>+12,.0f}"
        print(line)

        line = f"{'Return %':<18}"
        for v in group:
            line += f" {summaries[v]['total']/PORTFOLIO*100:>+13.1f}%"
        print(line)

        line = f"{'Win Rate':<18}"
        for v in group:
            line += f" {summaries[v]['wr']:>13.1f}%"
        print(line)

        line = f"{'W / L':<18}"
        for v in group:
            s = summaries[v]
            wl_str = f"{s['wins']}W/{s['losses']}L"
            line += f" {wl_str:>14}"
        print(line)

        line = f"{'Avg Win':<18}"
        for v in group:
            line += f" ${summaries[v]['avg_w']:>+12,.0f}"
        print(line)

        line = f"{'Avg Loss':<18}"
        for v in group:
            line += f" ${summaries[v]['avg_l']:>+12,.0f}"
        print(line)

        line = f"{'Win:Loss':<18}"
        for v in group:
            line += f" {summaries[v]['wl']:>12.2f}:1"
        print(line)

        line = f"{'Traded':<18}"
        for v in group:
            line += f" {summaries[v]['traded']:>14}"
        print(line)

        line = f"{'Rejected':<18}"
        for v in group:
            line += f" {summaries[v]['rejected']:>14}"
        print(line)

        line = f"{'vs Baseline':<18}"
        base_pnl = summaries["baseline"]["total"]
        for v in group:
            diff = summaries[v]["total"] - base_pnl
            line += f" ${diff:>+12,.0f}"
        print(line)

        line = f"{'vs Bar1Fix':<18}"
        bar1_pnl = summaries["bar1_fix"]["total"]
        for v in group:
            diff = summaries[v]["total"] - bar1_pnl
            line += f" ${diff:>+12,.0f}"
        print(line)

    # ── Rank variants ──
    print(f"\n{'=' * 80}")
    print("RANKED BY TOTAL P&L")
    print(f"{'=' * 80}")
    ranked = sorted(variant_names, key=lambda v: summaries[v]["total"], reverse=True)
    for i, v in enumerate(ranked):
        s = summaries[v]
        diff = s["total"] - summaries["baseline"]["total"]
        print(f"  {i+1:>2}. {variant_labels[v]:<16} ${s['total']:>+8,.0f}  "
              f"WR={s['wr']:.0f}%  W:L={s['wl']:.2f}:1  "
              f"Traded={s['traded']}  vs_base=${diff:>+,.0f}")

    # ── Show what the best combo variant does ──
    best = ranked[0]
    print(f"\n{'=' * 140}")
    print(f"BEST: {variant_labels[best]} — trades that differ from baseline")
    print(f"{'=' * 140}")

    diffs = []
    for i in range(total_signals):
        rv = all_results[best][i]
        rb = all_results["baseline"][i]
        if rv["reason"] != rb["reason"] or abs(rv["pnl"] - rb["pnl"]) > 1:
            diffs.append((i, rv, rb))

    if diffs:
        print(f"{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} "
              f"{'V P&L':>9} {'V Gate':<20} "
              f"{'B P&L':>9} {'B Gate':<20} {'Delta':>8} {'Peak%':>7} {'Hour':>5}")
        print("-" * 140)

        total_delta = 0
        for i, rv, rb in diffs:
            delta = rv["pnl"] - rb["pnl"]
            total_delta += delta
            print(f"{i+1:<4} {rv['ticker']:<7} {rv['dir'][:4]:<5} {rv['day']:<12} "
                  f"{rv['score']:>5} ${rv['entry']:>5.2f} "
                  f"${rv['pnl']:>+8.0f} {rv['reason']:<20} "
                  f"${rb['pnl']:>+8.0f} {rb['reason']:<20} ${delta:>+7.0f} "
                  f"{rv['peak_gain']:>+6.0f}% {rv['et_entry_hour']:>4}h")

        better = sum(1 for _, rv, rb in diffs if rv["pnl"] > rb["pnl"])
        worse = sum(1 for _, rv, rb in diffs if rv["pnl"] < rb["pnl"])
        same = len(diffs) - better - worse
        print(f"\n  Net: ${total_delta:>+,.0f}  |  Better: {better}  Worse: {worse}  Same: {same}")

    # ── Daily breakdown for top 3 ──
    top3 = ranked[:3]
    print(f"\n{'=' * 140}")
    print(f"DAILY P&L — Top 3 vs Baseline")
    print(f"{'=' * 140}")

    all_days = sorted(set(r["day"] for r in all_results["baseline"]))
    header = f"{'Date':<12} {'#':>3}"
    for v in ["baseline"] + [v for v in top3 if v != "baseline"]:
        header += f" {variant_labels[v]:>14}"
    print(header)
    print("-" * 80)

    cum = {v: 0 for v in variant_names}
    for day in all_days:
        line = f"{day:<12}"
        n = sum(1 for r in all_results["baseline"] if r["day"] == day and r["reason"] != "rejected")
        line += f" {n:>3}"
        for v in ["baseline"] + [v for v in top3 if v != "baseline"]:
            day_pnl = sum(r["pnl"] for r in all_results[v] if r["day"] == day)
            cum[v] += day_pnl
            line += f" ${day_pnl:>+13,.0f}"
        print(line)

    line = f"{'TOTAL':<12} {'':>3}"
    for v in ["baseline"] + [v for v in top3 if v != "baseline"]:
        line += f" ${cum[v]:>+13,.0f}"
    print(line)

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
