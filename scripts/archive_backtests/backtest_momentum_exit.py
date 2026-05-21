#!/usr/bin/env python3
"""Backtest momentum-confirmed exits for 0DTE trades.

Instead of blanket stop tightening or entry rejection, use underlying
price trend and premium momentum to confirm when a losing trade is dead.

The key insight: a 0DTE trade down 20% with flat underlying could be:
  A) Theta decay on a still-valid thesis → HOLD (underlying will move)
  B) Dead trade, underlying drifting against → EXIT

We can tell the difference by looking at the underlying's TREND over the
last few minutes, not just the absolute move from entry.

Variants:
  baseline:     Current production (no momentum exit)
  trend5:       Down 15%+, underlying trending against over last 5 ticks → exit
  trend10:      Down 15%+, underlying trending against over last 10 ticks → exit
  trend5_late:  Same as trend5 but only after 1PM ET
  dual:         Down 15%+, underlying trending against AND premium making lower highs → exit
  momentum:     Down 15%+, underlying moved against by 0.1%+ in last 10min → exit
  mom_prem:     Down 15%+, premium down 5%+ in last 10min AND underlying not confirming → exit
  adaptive:     Before noon: 20% drop + trend10. Noon-2PM: 15% drop + trend5. After 2PM: 10% drop + trend5.
"""

import sqlite3
import sys
from collections import defaultdict, deque
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


def _underlying_trending_against(u_history, is_call, n_ticks):
    """Check if underlying is trending against the trade direction.

    Uses simple linear regression slope over last n_ticks.
    Returns True if underlying is trending against.
    """
    if len(u_history) < n_ticks:
        return False
    recent = list(u_history)[-n_ticks:]
    # Simple: compare first half avg vs second half avg
    half = n_ticks // 2
    first_half = sum(recent[:half]) / half
    second_half = sum(recent[half:]) / (n_ticks - half)

    if is_call:
        # For calls, underlying trending DOWN is against
        return second_half < first_half - 0.01  # needs at least $0.01 trend
    else:
        # For puts, underlying trending UP is against
        return second_half > first_half + 0.01


def _premium_making_lower_highs(prem_history, n_ticks):
    """Check if premium is making lower highs (bearish pattern)."""
    if len(prem_history) < n_ticks:
        return False
    recent = list(prem_history)[-n_ticks:]
    half = n_ticks // 2
    first_high = max(recent[:half])
    second_high = max(recent[half:])
    return second_high < first_high * 0.98  # second high is 2%+ lower


def _underlying_moved_against_recent(u_history, is_call, lookback):
    """Check if underlying moved against by 0.1%+ in last `lookback` ticks."""
    if len(u_history) < lookback:
        return False
    recent_start = list(u_history)[-lookback]
    recent_end = list(u_history)[-1]
    if recent_start <= 0:
        return False
    move_pct = (recent_end - recent_start) / recent_start * 100
    if is_call:
        return move_pct < -0.1  # underlying dropped 0.1%+
    else:
        return move_pct > 0.1   # underlying rose 0.1%+


def _premium_dropping_recent(prem_history, lookback, threshold_pct):
    """Check if premium dropped by threshold_pct in last lookback ticks."""
    if len(prem_history) < lookback:
        return False
    recent_start = list(prem_history)[-lookback]
    recent_end = list(prem_history)[-1]
    if recent_start <= 0:
        return False
    change_pct = (recent_end - recent_start) / recent_start * 100
    return change_pct <= -threshold_pct


def simulate(entry, ticks, sig_ts, contracts, direction, ticker, score,
             expiry_date, variant):
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

    # Rolling history for momentum checks
    u_history = deque(maxlen=15)      # underlying prices (last ~15 min)
    prem_history = deque(maxlen=15)   # premium prices (last ~15 min)

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

        # Build rolling history
        if effective_underlying > 0:
            u_history.append(effective_underlying)
        prem_history.append(price)

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

        # FSM state
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

        # ── MOMENTUM-CONFIRMED EXIT (0DTE only, post-grace) ──
        if variant != "baseline" and not current_multiday and has_underlying:
            momentum_exit = _check_momentum_exit(
                variant, gain_pct, drop_entry_pct, elapsed_min, et_hour,
                is_call, u_history, prem_history, underlying_confirms,
            )
            if momentum_exit:
                return _make_exit(price, entry, contracts, elapsed_min, "momentum_exit")

        # Graduated stops
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

        # DEVELOPING
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


def _check_momentum_exit(variant, gain_pct, drop_entry_pct, elapsed_min, et_hour,
                          is_call, u_history, prem_history, underlying_confirms):
    """Check momentum-based exit conditions."""

    if variant == "trend5":
        # Down 15%+ AND underlying trending against over 5 ticks
        if gain_pct <= -15 and elapsed_min >= 10:
            return _underlying_trending_against(u_history, is_call, 5)

    elif variant == "trend10":
        # Down 15%+ AND underlying trending against over 10 ticks
        if gain_pct <= -15 and elapsed_min >= 15:
            return _underlying_trending_against(u_history, is_call, 10)

    elif variant == "trend5_late":
        # Same as trend5 but only after 1PM ET
        if et_hour >= 13 and gain_pct <= -15 and elapsed_min >= 10:
            return _underlying_trending_against(u_history, is_call, 5)

    elif variant == "dual":
        # Down 15%+ AND underlying trending against AND premium making lower highs
        if gain_pct <= -15 and elapsed_min >= 10:
            u_against = _underlying_trending_against(u_history, is_call, 5)
            p_weak = _premium_making_lower_highs(prem_history, 6)
            return u_against and p_weak

    elif variant == "momentum":
        # Down 15%+ AND underlying moved against by 0.1%+ in last 10 ticks
        if gain_pct <= -15 and elapsed_min >= 10:
            return _underlying_moved_against_recent(u_history, is_call, 10)

    elif variant == "mom_prem":
        # Down 15%+ AND premium dropping 5%+ in last 10 ticks AND underlying not confirming
        if gain_pct <= -15 and elapsed_min >= 10:
            prem_dropping = _premium_dropping_recent(prem_history, 10, 5.0)
            return prem_dropping and not underlying_confirms

    elif variant == "adaptive":
        # Time-adaptive thresholds with momentum confirmation
        if et_hour >= 14:
            # After 2PM: aggressive — down 10%+, 5-tick trend
            if gain_pct <= -10 and elapsed_min >= 8:
                return _underlying_trending_against(u_history, is_call, 5)
        elif et_hour >= 12:
            # Noon-2PM: moderate — down 15%+, 5-tick trend
            if gain_pct <= -15 and elapsed_min >= 10:
                return _underlying_trending_against(u_history, is_call, 5)
        else:
            # Before noon: patient — down 20%+, 10-tick trend
            if gain_pct <= -20 and elapsed_min >= 15:
                return _underlying_trending_against(u_history, is_call, 10)

    return False


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

    variant_names = ["baseline", "trend5", "trend10", "trend5_late",
                     "dual", "momentum", "mom_prem", "adaptive"]
    variant_labels = {
        "baseline":    "Baseline",
        "trend5":      "Trend-5",
        "trend10":     "Trend-10",
        "trend5_late": "Trend5 >1PM",
        "dual":        "Dual confirm",
        "momentum":    "U-momentum",
        "mom_prem":    "Prem+U drop",
        "adaptive":    "Time-adaptive",
    }

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
            pnl, reason, hold = simulate(
                entry, rows, sig_ts, contracts, direction, ticker, score,
                expiry_date=day, variant=variant,
            )
            all_results[variant].append({
                "id": sig["id"], "ticker": ticker, "dir": direction,
                "day": day, "score": score, "entry": entry,
                "contracts": contracts, "pnl": pnl, "reason": reason,
                "hold": hold, "peak_gain": overall_peak_gain,
                "et_entry_hour": (sig_ts.hour - 4) % 24,
            })

    # ===================================================================
    # REPORT
    # ===================================================================
    total_signals = len(all_results["baseline"])

    print(f"{'=' * 140}")
    print(f"MOMENTUM-CONFIRMED EXIT BACKTEST — {total_signals} signals")
    print(f"{'=' * 140}")
    print()
    print("Variants:")
    print("  baseline:    Current production (no momentum exit)")
    print("  trend5:      Down 15%+, underlying trending against over 5 ticks (10min hold)")
    print("  trend10:     Down 15%+, underlying trending against over 10 ticks (15min hold)")
    print("  trend5_late: Same as trend5 but only fires after 1PM ET")
    print("  dual:        Down 15%+, underlying against AND premium making lower highs")
    print("  momentum:    Down 15%+, underlying moved -0.1%+ in last 10 ticks")
    print("  mom_prem:    Down 15%+, premium dropping 5%+ recently AND underlying not confirming")
    print("  adaptive:    Time-scaled: before noon -20%/10t, noon-2PM -15%/5t, after 2PM -10%/5t")

    # Summary
    print(f"\n{'=' * 140}")
    print("SUMMARY")
    print(f"{'=' * 140}")

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
        m_fires = sum(1 for r in res if r["reason"] == "momentum_exit")
        summaries[v] = {
            "total": total, "wr": wr, "wins": len(wins), "losses": len(losses),
            "avg_w": avg_w, "avg_l": avg_l, "wl": wl, "m_fires": m_fires,
        }

    header = f"{'Metric':<18}"
    for v in variant_names:
        header += f" {variant_labels[v]:>13}"
    print(header)
    print("-" * (18 + 14 * len(variant_names)))

    def _row(label, key, fmt):
        line = f"{label:<18}"
        for v in variant_names:
            line += f" {fmt.format(summaries[v][key]):>13}"
        print(line)

    _row("Total P&L", "total", "${:>+,.0f}")

    line = f"{'Return %':<18}"
    for v in variant_names:
        line += f" {summaries[v]['total']/PORTFOLIO*100:>+12.1f}%"
    print(line)

    line = f"{'Win Rate':<18}"
    for v in variant_names:
        line += f" {summaries[v]['wr']:>12.1f}%"
    print(line)

    line = f"{'W / L':<18}"
    for v in variant_names:
        s = summaries[v]
        wl_str = f"{s['wins']}W/{s['losses']}L"
        line += f" {wl_str:>13}"
    print(line)

    _row("Avg Win", "avg_w", "${:>+,.0f}")
    _row("Avg Loss", "avg_l", "${:>+,.0f}")

    line = f"{'Win:Loss':<18}"
    for v in variant_names:
        line += f" {summaries[v]['wl']:>11.2f}:1"
    print(line)

    _row("Mom Fires", "m_fires", "{}")

    line = f"{'vs Baseline':<18}"
    base_pnl = summaries["baseline"]["total"]
    for v in variant_names:
        diff = summaries[v]["total"] - base_pnl
        line += f" ${diff:>+12,.0f}"
    print(line)

    # Show trades where each variant differs
    for v in variant_names:
        if v == "baseline":
            continue

        diffs = []
        for i in range(total_signals):
            rv = all_results[v][i]
            rb = all_results["baseline"][i]
            if rv["reason"] != rb["reason"]:
                diffs.append((i, rv, rb))

        if not diffs:
            print(f"\n{variant_labels[v]}: No trades differ from baseline")
            continue

        diff_pnl = summaries[v]["total"] - base_pnl
        print(f"\n{'=' * 140}")
        print(f"{variant_labels[v]} — {len(diffs)} trades differ (net ${diff_pnl:>+,.0f})")
        print(f"{'=' * 140}")
        print(f"{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'$In':>5} {'Ct':>3} "
              f"{'V P&L':>9} {'V Gate':<20} {'V Hold':>6} "
              f"{'B P&L':>9} {'B Gate':<20} {'Delta':>8} {'Peak%':>7}")
        print("-" * 130)

        total_delta = 0
        for i, rv, rb in diffs:
            delta = rv["pnl"] - rb["pnl"]
            total_delta += delta
            print(f"{i+1:<4} {rv['ticker']:<7} {rv['dir'][:4]:<5} {rv['day']:<12} "
                  f"${rv['entry']:>4.2f} {rv['contracts']:>3} "
                  f"${rv['pnl']:>+8.0f} {rv['reason']:<20} {rv['hold']:>5.0f}m "
                  f"${rb['pnl']:>+8.0f} {rb['reason']:<20} ${delta:>+7.0f} "
                  f"{rv['peak_gain']:>+6.0f}%")

        better = sum(1 for _, rv, rb in diffs if rv["pnl"] > rb["pnl"])
        worse = sum(1 for _, rv, rb in diffs if rv["pnl"] < rb["pnl"])
        print(f"\n  Net: ${total_delta:>+,.0f}  |  Better: {better}  Worse: {worse}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
