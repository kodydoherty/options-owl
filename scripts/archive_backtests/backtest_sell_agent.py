#!/usr/bin/env python3
"""Backtest SellAgentV2 exit concepts against real signal data.

Tests the interesting exit ideas from sell_agent_exit_logic_1.py:
  A. profit_lock     — 85% ratchet trail on contract price, tier-based activation
  B. breakeven_guard — once peak hits +10%, floor at entry (never lose)
  C. stock_trail     — underlying-based trailing stop (0.25-0.40% by ticker tier)
  D. max_loss_30     — hard 30% contract loss cap (no graduated stops needed)
  E. combined        — profit_lock + breakeven_guard + stock_trail
  F. ratchet_only    — simple 85% ratchet from first tick (no activation threshold)

All compared against production v4 FSM baseline.

Usage:
  python scripts/backtest_sell_agent.py [signals_db] [harvester_db]
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SLIPPAGE = 0.15
PORTFOLIO = 8000

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

# ── Production V4 FSM parameters (baseline) ─────────────────────────────
HARD_STOP_PCT = 0.30
GRACE_PERIOD_SEC = 300
BAR1_MIN_SEC = 90
BAR1_WINDOW_SEC = 150
BAR1_THRESHOLD_PCT = -5.0
SOFT_TRAIL_BAND_LOW = 10.0
SOFT_TRAIL_BAND_HIGH = 35.0
SOFT_TRAIL_FLOOR_FRACTION = 0.60
TRAIL_ACTIVATE_GAIN_PCT = 35.0
TRAIL_TIERS = [(400.0, 0.20), (200.0, 0.25), (100.0, 0.30), (50.0, 0.35)]
THETA_CURVE_FLOOR = 0.40
THETA_CURVE_FULL_SESSION = 6.5
THETA_CURVE_EXPONENT = 0.4
TRAIL_MULT_GIVEBACK_CAP = 1.20
TRAIL_MULT_MAX = 2.0
TRAIL_MULT_MAX_TRAIL = 0.45
TRAIL_MULT_MORNING = 1.5
TRAIL_MULT_SCORE_90 = 1.35
TRAIL_MULT_TICKERS = {"NVDA": 1.5, "TSLA": 1.5, "AMZN": 1.4, "AVGO": 1.4, "PLTR": 1.3}
HOUSE_MONEY_FLOORS = [(5.00, 2.00), (2.00, 0.80), (1.00, 0.30)]
ATM_MILESTONES = [(200.0, 0.15), (400.0, 0.15), (600.0, 0.15)]
THETA_TIMER_BASE_SEC = 7200
THETA_TIMER_MORNING_SEC = 10800
THETA_TIMER_LATE_SEC = 2400
THETA_TIMER_SCORE_IMMUNE = 92.0
THETA_TIMER_TICKER_IMMUNE = {"NVDA", "TSLA", "AMZN", "AVGO", "PLTR"}
HIGH_VOL_TICKERS = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
INDEX_PROFIT_TARGET_PCT = 30.0
EOD_CUTOFF_MINUTES = 15.0

# ── SellAgentV2 per-ticker profiles ──────────────────────────────────────
TICKER_PROFILES = {
    "SPY":  {"tier": "LOW_VOL",  "lock": 20, "trail": 0.0025, "hard_mult": 1.6, "range_w": 3.52},
    "QQQ":  {"tier": "LOW_VOL",  "lock": 20, "trail": 0.0025, "hard_mult": 1.6, "range_w": 5.41},
    "META": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 10.54},
    "AAPL": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 3.45},
    "AMZN": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 3.82},
    "MSFT": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 6.94},
    "TSLA": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 7.48},
    "NVDA": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 2.65},
    "AMD":  {"tier": "HIGH_VOL", "lock": 30, "trail": 0.0035, "hard_mult": 1.6, "range_w": 6.46},
    "MSTR": {"tier": "HIGH_VOL", "lock": 30, "trail": 0.0035, "hard_mult": 1.6, "range_w": 6.19},
    "PLTR": {"tier": "HIGH_VOL", "lock": 30, "trail": 0.0035, "hard_mult": 1.6, "range_w": 3.46},
    "GOOGL":{"tier": "LOW_VOL",  "lock": 20, "trail": 0.0025, "hard_mult": 1.6, "range_w": 5.57},
    "AVGO": {"tier": "MED_VOL",  "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 10.16},
}
DEFAULT_PROFILE = {"tier": "MED_VOL", "lock": 25, "trail": 0.0030, "hard_mult": 1.6, "range_w": 4.0}


# ── Shared helpers ───────────────────────────────────────────────────────

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
        "price": price, "bid": bid or 0, "ask": ask or 0,
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
            elapsed = (ts_dt - sig_ts).total_seconds() / 60
            return pnl, "eod_data_end", elapsed
    return 0, "no_data", 0


# ── V4 production baseline helpers ──────────────────────────────────────

def _get_trail_pct(gain_pct):
    for min_gain, trail in TRAIL_TIERS:
        if gain_pct >= min_gain:
            return trail
    return TRAIL_TIERS[-1][1] if TRAIL_TIERS else 0.35


def _theta_curve_mult(et_hour, et_min):
    hours_remaining = max(0.5, 16.0 - et_hour - et_min / 60.0)
    raw = (hours_remaining / THETA_CURVE_FULL_SESSION) ** THETA_CURVE_EXPONENT
    return max(THETA_CURVE_FLOOR, min(raw, 1.0))


def _apply_trail_multipliers(base_trail, ticker, is_morning, score):
    mult = 1.0
    if ticker in TRAIL_MULT_TICKERS:
        mult *= TRAIL_MULT_TICKERS[ticker]
    if is_morning:
        mult *= TRAIL_MULT_MORNING
    if score is not None and score >= 90:
        mult *= TRAIL_MULT_SCORE_90
    mult = min(mult, TRAIL_MULT_MAX)
    giveback_mult = min(mult, TRAIL_MULT_GIVEBACK_CAP)
    return min(base_trail * giveback_mult, TRAIL_MULT_MAX_TRAIL)


def _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min):
    peak_gain_pct = (peak - entry) / entry * 100 if entry > 0 else 0
    tier_trail = _get_trail_pct(peak_gain_pct)
    multiplied = _apply_trail_multipliers(tier_trail, ticker, is_morning, score)
    theta_mult = _theta_curve_mult(et_hour, et_min)
    effective = multiplied * theta_mult
    return peak * (1.0 - effective), effective


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


def _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
    if score is not None and score >= THETA_TIMER_SCORE_IMMUNE:
        return False
    if ticker in THETA_TIMER_TICKER_IMMUNE:
        return False
    current_hour = et_hour + et_min / 60.0
    if is_morning:
        timer_sec = THETA_TIMER_MORNING_SEC
    elif current_hour >= 14.0:
        timer_sec = THETA_TIMER_LATE_SEC
    else:
        timer_sec = THETA_TIMER_BASE_SEC
    return elapsed_sec >= timer_sec and gain_pct <= 5.0


# ── BASELINE: Production V4 FSM ─────────────────────────────────────────

def sim_baseline(entry, ticks, sig_ts, contracts, direction, ticker, score,
                 expiry_date=None):
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    is_index = ticker in INDEX_TICKERS
    is_high_vol = ticker in HIGH_VOL_TICKERS

    sig_date = sig_ts.date()
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            dte = 0
    else:
        dte = 0
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
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
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
            if not is_multiday and BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
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

        if fsm_state == "DEVELOPING":
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")
            if current_dte == 0:
                if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        # TRAILING
        house_money_floor_price = _compute_house_money_floor(peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        trail_stop, _ = _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        for ms_gain, ms_frac in ATM_MILESTONES:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    partial_pnl = (price - entry) * n_close * 100
                    if partial_pnl > 0:
                        partial_pnl *= (1 - SLIPPAGE)
                    contracts -= n_close
                    if contracts <= 0:
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                else:
                    locked_milestones.add(ms_gain)

        if current_dte == 0:
            if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT A: ProfitLock ratchet trail ──────────────────────────────────
# From SellAgentV2: tier-based activation %, then 85% ratchet trail on
# contract price. Replaces the multi-stage v4 trail.

def sim_profit_lock(entry, ticks, sig_ts, contracts, direction, ticker, score,
                    expiry_date=None):
    """ProfitLock: activate at tier % gain, then trail at 85% of peak."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    profile = TICKER_PROFILES.get(ticker, DEFAULT_PROFILE)
    activation_pct = profile["lock"] / 100.0  # e.g., 25 -> 0.25

    sig_date = sig_ts.date()
    dte = 0
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            pass

    peak = entry
    lock_activated = False
    lock_floor = 0.0
    ratchet_pct = 0.85

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue

        price = parsed["price"]
        elapsed_min = parsed["elapsed_min"]
        elapsed_sec = parsed["elapsed_sec"]
        minutes_to_close = parsed["minutes_to_close"]

        if price > peak:
            peak = price

        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass

        # EOD cutoff (0DTE)
        if current_dte == 0 and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # Grace period: no exits
        if elapsed_sec < GRACE_PERIOD_SEC:
            continue

        # Hard stop: 30% loss cap
        if price <= entry * 0.70:
            return _make_exit(price, entry, contracts, elapsed_min, "hard_stop_30")

        # Profit lock activation
        gain_pct_frac = (price - entry) / entry
        if not lock_activated:
            if gain_pct_frac >= activation_pct:
                lock_activated = True
                gain = price - entry
                lock_floor = entry + gain * 0.75  # initial floor at 75% of gain
        else:
            # Ratchet floor up
            new_floor = peak * ratchet_pct
            if new_floor > lock_floor:
                lock_floor = new_floor

            # Exit if below floor
            if price <= lock_floor:
                return _make_exit(price, entry, contracts, elapsed_min, "profit_lock")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT B: Breakeven guardian ────────────────────────────────────────
# v4 baseline + once peak hits +10%, floor at entry. Prevents winners
# from becoming losers.

def sim_breakeven_guard(entry, ticks, sig_ts, contracts, direction, ticker, score,
                        expiry_date=None):
    """v4 baseline + breakeven guardian: once +10% peak, floor at entry."""
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
    breakeven_armed = False  # NEW: breakeven guardian state

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

        # Breakeven guardian: arm when peak hits +10%
        peak_gain_from_entry = (peak - entry) / entry
        if not breakeven_armed and peak_gain_from_entry >= 0.10:
            breakeven_armed = True

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
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
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
            if not is_multiday and BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                bar1_change = (price - entry) / entry * 100
                if bar1_change <= BAR1_THRESHOLD_PCT:
                    return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")
            continue

        # NEW: Breakeven guardian — exit if armed and price at/below entry
        if breakeven_armed and price <= entry:
            return _make_exit(price, entry, contracts, elapsed_min, "breakeven_guard")

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

        if fsm_state == "DEVELOPING":
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")
            if current_dte == 0:
                if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        house_money_floor_price = _compute_house_money_floor(peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        trail_stop, _ = _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        for ms_gain, ms_frac in ATM_MILESTONES:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    partial_pnl = (price - entry) * n_close * 100
                    if partial_pnl > 0:
                        partial_pnl *= (1 - SLIPPAGE)
                    contracts -= n_close
                    if contracts <= 0:
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                else:
                    locked_milestones.add(ms_gain)

        if current_dte == 0:
            if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT C: Stock trail ───────────────────────────────────────────────
# Uses underlying price trailing stop from SellAgentV2 as an ADDITIONAL
# exit layer on top of v4 baseline.

def sim_stock_trail(entry, ticks, sig_ts, contracts, direction, ticker, score,
                    expiry_date=None):
    """v4 baseline + underlying-price trailing stop from SellAgentV2."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    is_index = ticker in INDEX_TICKERS
    is_high_vol = ticker in HIGH_VOL_TICKERS
    profile = TICKER_PROFILES.get(ticker, DEFAULT_PROFILE)
    trail_pct = profile["trail"]  # e.g., 0.0025 = 0.25%
    hard_pct = trail_pct * profile["hard_mult"]

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
    # Stock trail state
    stock_running_high = 0.0
    stock_running_low = float("inf")
    stock_trail_stop = 0.0

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
            stock_running_high = underlying
            stock_running_low = underlying
            if is_call:
                stock_trail_stop = underlying * (1 - trail_pct)
            else:
                stock_trail_stop = underlying * (1 + trail_pct)
        if underlying > 0:
            last_underlying = underlying
        effective_underlying = underlying if underlying > 0 else last_underlying

        # Update stock trail
        if underlying > 0 and entry_underlying:
            if is_call:
                if underlying > stock_running_high:
                    stock_running_high = underlying
                    stock_trail_stop = underlying * (1 - trail_pct)
            else:
                if underlying < stock_running_low:
                    stock_running_low = underlying
                    stock_trail_stop = underlying * (1 + trail_pct)

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
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
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
            if not is_multiday and BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                bar1_change = (price - entry) / entry * 100
                if bar1_change <= BAR1_THRESHOLD_PCT:
                    return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")
            continue

        if bid <= 0 and seconds_at_zero_bid >= 30:
            return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

        # NEW: Stock price hard stop (from SellAgentV2)
        if underlying > 0 and entry_underlying and elapsed_sec >= GRACE_PERIOD_SEC:
            if is_call:
                hard_stop_price = stock_running_high * (1 - hard_pct)
                if underlying <= hard_stop_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_hard_stop")
                if underlying <= stock_trail_stop:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_trail_stop")
            else:
                hard_stop_price = stock_running_low * (1 + hard_pct)
                if underlying >= hard_stop_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_hard_stop")
                if underlying >= stock_trail_stop:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_trail_stop")

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

        if fsm_state == "DEVELOPING":
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")
            if current_dte == 0:
                if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        house_money_floor_price = _compute_house_money_floor(peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        trail_stop, _ = _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        for ms_gain, ms_frac in ATM_MILESTONES:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    partial_pnl = (price - entry) * n_close * 100
                    if partial_pnl > 0:
                        partial_pnl *= (1 - SLIPPAGE)
                    contracts -= n_close
                    if contracts <= 0:
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                else:
                    locked_milestones.add(ms_gain)

        if current_dte == 0:
            if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT D: Max loss 30% hard cut ────────────────────────────────────
# Simple: 30% contract loss = exit. No graduated stops, no underlying
# needed. Tests if the SellAgentV2 max_loss_pct concept is better.

def sim_max_loss_30(entry, ticks, sig_ts, contracts, direction, ticker, score,
                    expiry_date=None):
    """Simple 30% contract loss hard cut + v4 trailing for winners."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    is_index = ticker in INDEX_TICKERS

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
    house_money_floor_price = 0.0
    locked_milestones = set()

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue

        price = parsed["price"]
        elapsed_sec = parsed["elapsed_sec"]
        elapsed_min = parsed["elapsed_min"]
        gain_pct = parsed["gain_pct"]
        et_hour = parsed["et_hour"]
        et_min = parsed["et_min"]
        minutes_to_close = parsed["minutes_to_close"]

        if price > peak:
            peak = price

        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass
        current_multiday = current_dte > 0

        peak_gain_pct = (peak - entry) / entry * 100

        if not current_multiday and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # Grace period
        if elapsed_sec < GRACE_PERIOD_SEC:
            continue

        # Simple 30% max loss cut (replaces all graduated stops)
        if gain_pct <= -30.0:
            return _make_exit(price, entry, contracts, elapsed_min, "max_loss_30")

        # Index profit target
        if is_index and gain_pct >= INDEX_PROFIT_TARGET_PCT:
            return _make_exit(price, entry, contracts, elapsed_min, "profit_target")

        # Soft trail in developing zone
        if peak_gain_pct < TRAIL_ACTIVATE_GAIN_PCT:
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")
            if current_dte == 0:
                if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        # TRAILING
        house_money_floor_price = _compute_house_money_floor(peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        trail_stop, _ = _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        for ms_gain, ms_frac in ATM_MILESTONES:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    partial_pnl = (price - entry) * n_close * 100
                    if partial_pnl > 0:
                        partial_pnl *= (1 - SLIPPAGE)
                    contracts -= n_close
                    if contracts <= 0:
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                else:
                    locked_milestones.add(ms_gain)

        if current_dte == 0:
            if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT E: Combined best ────────────────────────────────────────────
# Profit lock ratchet + breakeven guardian + stock trail, all layered.

def sim_combined(entry, ticks, sig_ts, contracts, direction, ticker, score,
                 expiry_date=None):
    """Profit lock + breakeven guard + stock trail combined."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    profile = TICKER_PROFILES.get(ticker, DEFAULT_PROFILE)
    activation_pct = profile["lock"] / 100.0
    trail_pct = profile["trail"]
    hard_pct = trail_pct * profile["hard_mult"]

    sig_date = sig_ts.date()
    dte = 0
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            pass

    peak = entry
    lock_activated = False
    lock_floor = 0.0
    ratchet_pct = 0.85
    breakeven_armed = False
    entry_underlying = None
    stock_running_high = 0.0
    stock_running_low = float("inf")
    stock_trail_stop = 0.0

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue

        price = parsed["price"]
        underlying = parsed["underlying"]
        elapsed_sec = parsed["elapsed_sec"]
        elapsed_min = parsed["elapsed_min"]
        minutes_to_close = parsed["minutes_to_close"]

        if price > peak:
            peak = price

        # Breakeven guardian
        if not breakeven_armed and (peak - entry) / entry >= 0.10:
            breakeven_armed = True

        # Stock trail tracking
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying
            stock_running_high = underlying
            stock_running_low = underlying
            if is_call:
                stock_trail_stop = underlying * (1 - trail_pct)
            else:
                stock_trail_stop = underlying * (1 + trail_pct)
        if underlying > 0:
            if is_call and underlying > stock_running_high:
                stock_running_high = underlying
                stock_trail_stop = underlying * (1 - trail_pct)
            elif not is_call and underlying < stock_running_low:
                stock_running_low = underlying
                stock_trail_stop = underlying * (1 + trail_pct)

        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass

        # EOD
        if current_dte == 0 and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # Grace
        if elapsed_sec < GRACE_PERIOD_SEC:
            continue

        # Hard stop 30%
        if price <= entry * 0.70:
            return _make_exit(price, entry, contracts, elapsed_min, "hard_stop_30")

        # Breakeven guardian
        if breakeven_armed and price <= entry:
            return _make_exit(price, entry, contracts, elapsed_min, "breakeven_guard")

        # Stock hard stop
        if underlying > 0 and entry_underlying:
            if is_call:
                if underlying <= stock_running_high * (1 - hard_pct):
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_hard_stop")
                if underlying <= stock_trail_stop:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_trail_stop")
            else:
                if underlying >= stock_running_low * (1 + hard_pct):
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_hard_stop")
                if underlying >= stock_trail_stop:
                    return _make_exit(price, entry, contracts, elapsed_min, "stock_trail_stop")

        # Profit lock
        gain_pct_frac = (price - entry) / entry
        if not lock_activated:
            if gain_pct_frac >= activation_pct:
                lock_activated = True
                gain = price - entry
                lock_floor = entry + gain * 0.75
        else:
            new_floor = peak * ratchet_pct
            if new_floor > lock_floor:
                lock_floor = new_floor
            if price <= lock_floor:
                return _make_exit(price, entry, contracts, elapsed_min, "profit_lock")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── VARIANT F: Simple ratchet (no activation threshold) ──────────────────

def sim_ratchet_only(entry, ticks, sig_ts, contracts, direction, ticker, score,
                     expiry_date=None):
    """85% ratchet from first tick — no activation threshold needed."""
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    sig_date = sig_ts.date()
    dte = 0
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            pass

    peak = entry
    ratchet_pct = 0.85

    for tick in ticks:
        parsed = _parse_tick(tick, sig_ts, entry)
        if parsed is None:
            continue

        price = parsed["price"]
        elapsed_sec = parsed["elapsed_sec"]
        elapsed_min = parsed["elapsed_min"]
        minutes_to_close = parsed["minutes_to_close"]

        if price > peak:
            peak = price

        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = parsed["et_dt"].date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass

        if current_dte == 0 and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        if elapsed_sec < GRACE_PERIOD_SEC:
            continue

        # Hard stop 30%
        if price <= entry * 0.70:
            return _make_exit(price, entry, contracts, elapsed_min, "hard_stop_30")

        # Ratchet: always trail at 85% of peak
        ratchet_floor = peak * ratchet_pct
        if ratchet_floor > entry and price <= ratchet_floor:
            return _make_exit(price, entry, contracts, elapsed_min, "ratchet_trail")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── Data loading + scoring ───────────────────────────────────────────────

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
    target_per_trade = deployable / 5
    scaled_target = target_per_trade * budget_mult
    cost_per_contract = entry_premium * 100
    if cost_per_contract <= 0:
        return 1
    position_cap = int((PORTFOLIO * 0.15) / cost_per_contract)
    raw_contracts = int(scaled_target / cost_per_contract)
    return max(1, min(raw_contracts, max(1, position_cap)))


# ── Main ─────────────────────────────────────────────────────────────────

VARIANTS = {
    "baseline":       sim_baseline,
    "A_profit_lock":  sim_profit_lock,
    "B_breakeven":    sim_breakeven_guard,
    "C_stock_trail":  sim_stock_trail,
    "D_max_loss_30":  sim_max_loss_30,
    "E_combined":     sim_combined,
    "F_ratchet_only": sim_ratchet_only,
}


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

    # Pre-load all data
    trade_inputs = []
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

        trade_inputs.append({
            "id": sig["id"], "ticker": ticker, "dir": direction,
            "day": day, "score": score, "entry": entry,
            "contracts": contracts, "rows": rows, "sig_ts": sig_ts,
            "peak_gain": overall_peak_gain, "expiry": day,
        })

    # Run all variants
    all_results = {}
    for vname, sim_fn in VARIANTS.items():
        results = []
        for inp in trade_inputs:
            pnl, reason, hold = sim_fn(
                inp["entry"], inp["rows"], inp["sig_ts"], inp["contracts"],
                inp["dir"], inp["ticker"], inp["score"],
                expiry_date=inp["expiry"],
            )
            results.append({
                "id": inp["id"], "ticker": inp["ticker"], "dir": inp["dir"],
                "day": inp["day"], "score": inp["score"], "entry": inp["entry"],
                "contracts": inp["contracts"], "pnl": pnl, "reason": reason,
                "hold": hold, "peak_gain": inp["peak_gain"],
            })
        all_results[vname] = results

    # ═══════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════
    total_signals = len(trade_inputs)
    print(f"\n{'=' * 120}")
    print(f"SELL AGENT V2 EXIT CONCEPTS BACKTEST — {total_signals} signals")
    print(f"{'=' * 120}")
    print(f"Portfolio: ${PORTFOLIO:,}  |  Slippage: {SLIPPAGE*100:.0f}%  |  "
          f"No data: {no_data}  |  No strike: {no_strike}  |  Score<78: {skipped}")

    # ── Summary comparison ──
    print(f"\n{'=' * 120}")
    print("VARIANT COMPARISON")
    print(f"{'=' * 120}")
    print(f"\n{'Variant':<18} {'P&L':>10} {'Return':>8} {'WR':>6} {'W':>4} {'L':>4} "
          f"{'AvgW':>8} {'AvgL':>8} {'W:L':>6} {'vs Base':>10}")
    print("-" * 100)

    baseline_pnl = sum(r["pnl"] for r in all_results["baseline"])

    for vname in VARIANTS:
        res = all_results[vname]
        wins = [r for r in res if r["pnl"] > 0]
        losses = [r for r in res if r["pnl"] <= 0]
        total_pnl = sum(r["pnl"] for r in res)
        wr = len(wins) / len(res) * 100 if res else 0
        avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
        avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
        wl = abs(avg_w / avg_l) if avg_l else 0
        delta = total_pnl - baseline_pnl
        delta_str = f"${delta:>+,.0f}" if vname != "baseline" else "---"

        print(f"{vname:<18} ${total_pnl:>+9,.0f} {total_pnl/PORTFOLIO*100:>+6.1f}% "
              f"{wr:>5.1f}% {len(wins):>4} {len(losses):>4} "
              f"${avg_w:>+7,.0f} ${avg_l:>+7,.0f} {wl:>5.2f} {delta_str:>10}")

    # ── Per-variant gate breakdown ──
    for vname in VARIANTS:
        res = all_results[vname]
        total_pnl = sum(r["pnl"] for r in res)

        print(f"\n{'=' * 120}")
        print(f"EXIT GATE BREAKDOWN: {vname} (P&L: ${total_pnl:>+,.0f})")
        print(f"{'=' * 120}")

        gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "holds": []})
        for r in res:
            g = gate_stats[r["reason"]]
            g["count"] += 1
            g["pnl"] += r["pnl"]
            if r["pnl"] > 0:
                g["wins"] += 1
            g["holds"].append(r["hold"])

        print(f"\n{'Gate':<22} {'Fires':>5} {'%':>5} {'P&L':>10} {'W/L':>8} {'WR':>5} {'AvgHold':>7}")
        print("-" * 75)

        for gate, s in sorted(gate_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            ct = s["count"]
            pct = ct / len(res) * 100 if res else 0
            g_wr = s["wins"] / ct * 100 if ct else 0
            ah = sum(s["holds"]) / ct if ct else 0
            print(f"{gate:<22} {ct:>5} {pct:>4.0f}% ${s['pnl']:>+9.0f} "
                  f"{s['wins']}W/{ct - s['wins']}L {g_wr:>4.0f}% {ah:>5.0f}m")

    # ── Head-to-head trade comparison (baseline vs best non-baseline) ──
    best_variant = max(
        (v for v in VARIANTS if v != "baseline"),
        key=lambda v: sum(r["pnl"] for r in all_results[v])
    )
    best_res = all_results[best_variant]
    base_res = all_results["baseline"]

    print(f"\n{'=' * 120}")
    print(f"TRADE-BY-TRADE: baseline vs {best_variant}")
    print(f"{'=' * 120}")
    print(f"\n{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'$In':>6} {'Ct':>3} "
          f"{'Base P&L':>10} {'Base Gate':<20} "
          f"{'Best P&L':>10} {'Best Gate':<20} {'Delta':>8}")
    print("-" * 120)

    for i, (b, best) in enumerate(zip(base_res, best_res)):
        delta = best["pnl"] - b["pnl"]
        marker = ""
        if abs(delta) > 100:
            marker = " <--" if delta > 0 else " !!!"
        print(f"{i+1:<4} {b['ticker']:<7} {b['dir'][:4]:<5} {b['day']:<12} "
              f"${b['entry']:>5.2f} {b['contracts']:>3} "
              f"${b['pnl']:>+9.0f} {b['reason']:<20} "
              f"${best['pnl']:>+9.0f} {best['reason']:<20} "
              f"${delta:>+7.0f}{marker}")

    # ── Trades where variants diverge most ──
    print(f"\n{'=' * 120}")
    print("BIGGEST DIVERGENCES (where variant choice matters most)")
    print(f"{'=' * 120}")

    divergences = []
    for i, inp in enumerate(trade_inputs):
        pnls = {v: all_results[v][i]["pnl"] for v in VARIANTS}
        spread = max(pnls.values()) - min(pnls.values())
        best_v = max(pnls, key=pnls.get)
        worst_v = min(pnls, key=pnls.get)
        divergences.append((spread, i, inp, pnls, best_v, worst_v))

    divergences.sort(reverse=True)
    print(f"\n{'#':<4} {'Ticker':<7} {'Day':<12} {'Spread':>8} {'Best':>18} {'Worst':>18} {'Base':>10}")
    print("-" * 90)

    for spread, i, inp, pnls, best_v, worst_v in divergences[:15]:
        print(f"{i+1:<4} {inp['ticker']:<7} {inp['day']:<12} ${spread:>7.0f} "
              f"{best_v}: ${pnls[best_v]:>+7.0f}  "
              f"{worst_v}: ${pnls[worst_v]:>+7.0f}  "
              f"${pnls['baseline']:>+7.0f}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
