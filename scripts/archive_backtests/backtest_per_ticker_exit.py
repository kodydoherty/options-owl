#!/usr/bin/env python3
"""Backtest per-ticker exit tuning on top of v4 FSM baseline.

Inspired by SellAgentV2's TICKER_PROFILES tier system. Tests adapting
trail widths, activation thresholds, soft trail floors, and graduated
stops per ticker volatility tier — while keeping v4's strong loss
protection intact.

Tiers (from SellAgentV2's real-data profiles):
  LOW_VOL  — SPY, QQQ, GOOGL, IWM       (ETFs, <1.5% daily moves)
  MED_VOL  — META, AAPL, AMZN, MSFT,    (2-4% daily moves)
             TSLA, NVDA, AVGO
  HIGH_VOL — AMD, MSTR, PLTR, COIN,     (>5% daily moves)
             SMCI

Variants tested:
  G. trail_activation  — LOW_VOL activates trail earlier (25%), HIGH later (45%)
  H. trail_width       — LOW_VOL tighter trail, HIGH wider trail
  I. soft_trail_floor  — LOW_VOL higher floor (lock gains sooner), HIGH lower
  J. grad_stops        — Per-tier graduated stop thresholds
  K. combined_mild     — G+H+I together (conservative per-ticker tuning)
  L. combined_full     — G+H+I+J together (aggressive per-ticker tuning)
  M. index_special     — Index-specific: faster activation, tighter trail, lower target

Usage:
  python scripts/backtest_per_ticker_exit.py [signals_db] [harvester_db]
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

# ── Per-ticker tier classification ──────────────────────────────────────
# LOW_VOL: ETFs/indexes — smaller % moves, tighter ranges
# MED_VOL: Large-cap tech — moderate swings
# HIGH_VOL: Meme/momentum stocks — wild swings

TIER_MAP = {
    # LOW_VOL
    "SPY": "LOW", "QQQ": "LOW", "IWM": "LOW", "DIA": "LOW",
    "XLF": "LOW", "XLK": "LOW", "GOOGL": "LOW",
    # HIGH_VOL
    "MSTR": "HIGH", "AMD": "HIGH", "PLTR": "HIGH",
    "COIN": "HIGH", "SMCI": "HIGH",
    # MED_VOL (everything else, including TSLA/NVDA which are volatile
    # but have enough liquidity for standard trail)
    "META": "MED", "AAPL": "MED", "AMZN": "MED", "MSFT": "MED",
    "TSLA": "MED", "NVDA": "MED", "AVGO": "MED",
}

def _get_tier(ticker):
    return TIER_MAP.get(ticker, "MED")


# ── Per-tier parameter tables ────────────────────────────────────────────

# G: Trail activation — when does the trailing stop turn on?
# LOW_VOL moves less, so activate sooner to lock gains
# HIGH_VOL swings more, need more room before trailing
TIER_TRAIL_ACTIVATE = {"LOW": 25.0, "MED": 35.0, "HIGH": 45.0}

# H: Trail width multiplier applied to base tier trail
# LOW_VOL: tighter trail (less noise), HIGH_VOL: wider (more noise)
TIER_TRAIL_WIDTH_MULT = {"LOW": 0.80, "MED": 1.00, "HIGH": 1.25}

# I: Soft trail floor fraction — what % of peak gain to protect
# LOW_VOL: higher floor (protect more), HIGH_VOL: lower (give room)
TIER_SOFT_TRAIL_FLOOR = {"LOW": 0.70, "MED": 0.60, "HIGH": 0.50}

# J: Graduated stop thresholds (tight_stop, backstop) for 0DTE
# These override the HIGH_VOL_TICKERS binary split in v4
TIER_GRAD_STOPS_0DTE = {
    "LOW":  (0.30, 0.55),   # Tighter: LOW_VOL shouldn't drop 50%+ and recover
    "MED":  (0.35, 0.65),   # Same as current v4 non-high-vol
    "HIGH": (0.50, 0.80),   # Wider: HIGH_VOL needs room, normal for MSTR to dip 40%
}
TIER_GRAD_STOPS_MULTI = {
    "LOW":  (0.45, 0.65),
    "MED":  (0.52, 0.75),
    "HIGH": (0.65, 0.90),
}

# M: Index-specific overrides
INDEX_TRAIL_ACTIVATE = 20.0       # SPY/QQQ don't need +35% before trailing
INDEX_TRAIL_WIDTH_MULT = 0.75     # Tighter trail — indexes trend smoother
INDEX_PROFIT_TARGET_CUSTOM = 25.0 # Take profit earlier on indexes


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


def _get_trail_pct(gain_pct, width_mult=1.0):
    for min_gain, trail in TRAIL_TIERS:
        if gain_pct >= min_gain:
            return trail * width_mult
    base = TRAIL_TIERS[-1][1] if TRAIL_TIERS else 0.35
    return base * width_mult


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


def _compute_trail_stop(peak, entry, ticker, is_morning, score, et_hour, et_min,
                        width_mult=1.0):
    peak_gain_pct = (peak - entry) / entry * 100 if entry > 0 else 0
    tier_trail = _get_trail_pct(peak_gain_pct, width_mult)
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


# ── Configurable v4 simulator ───────────────────────────────────────────
# Single function with per-ticker overrides via config dict.

def sim_v4_configurable(entry, ticks, sig_ts, contracts, direction, ticker, score,
                        expiry_date=None, cfg=None):
    """V4 FSM with configurable per-ticker parameters.

    cfg dict keys (all optional, defaults to v4 production values):
      trail_activate   — peak gain % to enter TRAILING state
      trail_width_mult — multiplier on trail tier widths
      soft_floor_frac  — soft trail floor fraction (0.0-1.0)
      tight_stop       — graduated tight stop (0DTE)
      backstop         — graduated backstop (0DTE)
      tight_stop_multi — graduated tight stop (multi-day)
      backstop_multi   — graduated backstop (multi-day)
      profit_target    — index profit target % (None to use default)
    """
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    cfg = cfg or {}

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

    # Per-ticker config
    trail_activate = cfg.get("trail_activate", TRAIL_ACTIVATE_GAIN_PCT)
    trail_width_mult = cfg.get("trail_width_mult", 1.0)
    soft_floor_frac = cfg.get("soft_floor_frac", SOFT_TRAIL_FLOOR_FRACTION)
    profit_target = cfg.get("profit_target", INDEX_PROFIT_TARGET_PCT)

    # Graduated stops
    if not is_multiday:
        cfg_tight = cfg.get("tight_stop")
        cfg_back = cfg.get("backstop")
        if cfg_tight is not None:
            tight_stop_pct = cfg_tight
            backstop_pct = cfg_back if cfg_back is not None else cfg_tight + 0.30
        elif is_high_vol:
            tight_stop_pct = 0.45
            backstop_pct = 0.75
        else:
            tight_stop_pct = 0.35
            backstop_pct = 0.65
    else:
        cfg_tight = cfg.get("tight_stop_multi")
        cfg_back = cfg.get("backstop_multi")
        if cfg_tight is not None:
            tight_stop_pct = cfg_tight
            backstop_pct = cfg_back if cfg_back is not None else cfg_tight + 0.20
        elif is_high_vol:
            tight_stop_pct = 0.60
            backstop_pct = 0.85
        else:
            tight_stop_pct = 0.52
            backstop_pct = 0.75

    # Soft trail band high = trail activation (they're the same boundary)
    soft_band_high = trail_activate

    # Running state
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

        # FSM state (using per-ticker activation)
        if elapsed_sec < GRACE_PERIOD_SEC:
            fsm_state = "GRACE"
        elif peak_gain_pct >= trail_activate:
            fsm_state = "TRAILING"
        else:
            fsm_state = "DEVELOPING"

        # ── EOD cutoff (0DTE only) ──
        if not current_multiday and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # ── GRACE ──
        if fsm_state == "GRACE":
            if not is_multiday and BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                bar1_change = (price - entry) / entry * 100
                if bar1_change <= BAR1_THRESHOLD_PCT:
                    return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")
            continue

        # ── Bid disappearance (post-grace) ──
        if bid <= 0 and seconds_at_zero_bid >= 30:
            return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

        # ── Index profit target ──
        if is_index and profit_target > 0 and gain_pct >= profit_target:
            return _make_exit(price, entry, contracts, elapsed_min, "profit_target")

        # ── Scalp trail ──
        if peak_gain_pct >= 20 and gain_pct > 0 and gain_pct < peak_gain_pct * 0.6:
            should_scalp = False
            if not current_multiday and has_underlying and not underlying_confirms:
                should_scalp = True
            elif current_multiday and has_underlying and underlying_against:
                should_scalp = True
            if should_scalp:
                return _make_exit(price, entry, contracts, elapsed_min, "scalp_trail")

        # ── Checkpoint cut ──
        if not current_multiday and drop_entry_pct >= 30 and has_underlying and underlying_against:
            return _make_exit(price, entry, contracts, elapsed_min, "checkpoint_cut")

        # ── Graduated stops (per-ticker thresholds) ──
        if has_underlying:
            if underlying_against:
                if drop_entry_pct >= tight_stop_pct * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "confirmed_stop")
            else:
                mid_stop = (tight_stop_pct + backstop_pct) / 2
                if drop_entry_pct >= mid_stop * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "mid_range_stop")
                if drop_entry_pct >= backstop_pct * 100:
                    return _make_exit(price, entry, contracts, elapsed_min, "backstop")
        else:
            stop_price = entry * (1.0 - HARD_STOP_PCT)
            if minutes_to_close > 30 and ask > 0 and bid >= 0:
                compare = (bid + ask) / 2.0
            else:
                compare = bid if bid > 0 else price
            if compare <= stop_price and compare >= 0:
                return _make_exit(price, entry, contracts, elapsed_min, "hard_stop")

        # ── DEVELOPING ──
        if fsm_state == "DEVELOPING":
            # Soft trail (per-ticker floor fraction)
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < soft_band_high:
                floor_gain_frac = (peak_gain_pct / 100.0) * soft_floor_frac
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")

            if current_dte == 0:
                if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")
            continue

        # ── TRAILING ──

        # House-money floor
        house_money_floor_price = _compute_house_money_floor(
            peak_gain_pct, entry, house_money_floor_price)
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        # Trail stop (per-ticker width)
        trail_stop, _ = _compute_trail_stop(
            peak, entry, ticker, is_morning, score, et_hour, et_min,
            width_mult=trail_width_mult)
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        # Milestones
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

        # Theta timer
        if current_dte == 0:
            if _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning, et_hour, et_min):
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


# ── Variant builders ─────────────────────────────────────────────────────
# Each returns a config dict for the given ticker.

def cfg_baseline(ticker):
    """Production v4 — no per-ticker overrides."""
    return {}

def cfg_G_trail_activation(ticker):
    """Per-tier trail activation threshold."""
    tier = _get_tier(ticker)
    return {"trail_activate": TIER_TRAIL_ACTIVATE[tier]}

def cfg_H_trail_width(ticker):
    """Per-tier trail width multiplier."""
    tier = _get_tier(ticker)
    return {"trail_width_mult": TIER_TRAIL_WIDTH_MULT[tier]}

def cfg_I_soft_floor(ticker):
    """Per-tier soft trail floor fraction."""
    tier = _get_tier(ticker)
    return {"soft_floor_frac": TIER_SOFT_TRAIL_FLOOR[tier]}

def cfg_J_grad_stops(ticker):
    """Per-tier graduated stop thresholds."""
    tier = _get_tier(ticker)
    t0, b0 = TIER_GRAD_STOPS_0DTE[tier]
    tm, bm = TIER_GRAD_STOPS_MULTI[tier]
    return {
        "tight_stop": t0, "backstop": b0,
        "tight_stop_multi": tm, "backstop_multi": bm,
    }

def cfg_K_combined_mild(ticker):
    """G+H+I together — conservative per-ticker tuning."""
    tier = _get_tier(ticker)
    return {
        "trail_activate": TIER_TRAIL_ACTIVATE[tier],
        "trail_width_mult": TIER_TRAIL_WIDTH_MULT[tier],
        "soft_floor_frac": TIER_SOFT_TRAIL_FLOOR[tier],
    }

def cfg_L_combined_full(ticker):
    """G+H+I+J together — aggressive per-ticker tuning."""
    tier = _get_tier(ticker)
    t0, b0 = TIER_GRAD_STOPS_0DTE[tier]
    tm, bm = TIER_GRAD_STOPS_MULTI[tier]
    return {
        "trail_activate": TIER_TRAIL_ACTIVATE[tier],
        "trail_width_mult": TIER_TRAIL_WIDTH_MULT[tier],
        "soft_floor_frac": TIER_SOFT_TRAIL_FLOOR[tier],
        "tight_stop": t0, "backstop": b0,
        "tight_stop_multi": tm, "backstop_multi": bm,
    }

def cfg_M_index_special(ticker):
    """Index-specific tuning: earlier activation, tighter trail, lower target."""
    if ticker in INDEX_TICKERS:
        return {
            "trail_activate": INDEX_TRAIL_ACTIVATE,
            "trail_width_mult": INDEX_TRAIL_WIDTH_MULT,
            "profit_target": INDEX_PROFIT_TARGET_CUSTOM,
        }
    return {}  # Non-index: same as baseline


VARIANTS = {
    "baseline":          cfg_baseline,
    "G_trail_activate":  cfg_G_trail_activation,
    "H_trail_width":     cfg_H_trail_width,
    "I_soft_floor":      cfg_I_soft_floor,
    "J_grad_stops":      cfg_J_grad_stops,
    "K_mild_combined":   cfg_K_combined_mild,
    "L_full_combined":   cfg_L_combined_full,
    "M_index_special":   cfg_M_index_special,
}


# ── Data loading ─────────────────────────────────────────────────────────

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

        trade_inputs.append({
            "id": sig["id"], "ticker": ticker, "dir": direction,
            "day": day, "score": score, "entry": entry,
            "contracts": contracts, "rows": rows, "sig_ts": sig_ts,
            "peak_gain": (overall_peak - entry) / entry * 100,
            "expiry": day,
        })

    # ── Run all variants ──
    all_results = {}
    for vname, cfg_fn in VARIANTS.items():
        results = []
        for inp in trade_inputs:
            cfg = cfg_fn(inp["ticker"])
            pnl, reason, hold = sim_v4_configurable(
                inp["entry"], inp["rows"], inp["sig_ts"], inp["contracts"],
                inp["dir"], inp["ticker"], inp["score"],
                expiry_date=inp["expiry"], cfg=cfg,
            )
            results.append({
                "id": inp["id"], "ticker": inp["ticker"], "dir": inp["dir"],
                "day": inp["day"], "score": inp["score"], "entry": inp["entry"],
                "contracts": inp["contracts"], "pnl": pnl, "reason": reason,
                "hold": hold, "peak_gain": inp["peak_gain"],
                "tier": _get_tier(inp["ticker"]),
            })
        all_results[vname] = results

    total_signals = len(trade_inputs)

    # ═══════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"PER-TICKER EXIT TUNING BACKTEST — {total_signals} signals")
    print(f"{'=' * 120}")
    print(f"Portfolio: ${PORTFOLIO:,}  |  Slippage: {SLIPPAGE*100:.0f}%  |  "
          f"No data: {no_data}  |  No strike: {no_strike}  |  Score<78: {skipped}")

    # Tier distribution
    tier_counts = defaultdict(int)
    for inp in trade_inputs:
        tier_counts[_get_tier(inp["ticker"])] += 1
    print(f"Tier distribution: LOW={tier_counts['LOW']}  MED={tier_counts['MED']}  HIGH={tier_counts['HIGH']}")

    # ── Summary comparison ──
    print(f"\n{'=' * 120}")
    print("VARIANT COMPARISON (all tickers)")
    print(f"{'=' * 120}")
    print(f"\n{'Variant':<20} {'P&L':>10} {'Return':>8} {'WR':>6} {'W':>4} {'L':>4} "
          f"{'AvgW':>8} {'AvgL':>8} {'W:L':>6} {'vs Base':>10}")
    print("-" * 105)

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

        print(f"{vname:<20} ${total_pnl:>+9,.0f} {total_pnl/PORTFOLIO*100:>+6.1f}% "
              f"{wr:>5.1f}% {len(wins):>4} {len(losses):>4} "
              f"${avg_w:>+7,.0f} ${avg_l:>+7,.0f} {wl:>5.2f} {delta_str:>10}")

    # ── Per-tier breakdown for each variant ──
    print(f"\n{'=' * 120}")
    print("PER-TIER BREAKDOWN (how each variant affects each tier)")
    print(f"{'=' * 120}")

    for tier in ["LOW", "MED", "HIGH"]:
        tier_trades = [i for i, inp in enumerate(trade_inputs) if _get_tier(inp["ticker"]) == tier]
        if not tier_trades:
            continue
        print(f"\n--- {tier}_VOL ({len(tier_trades)} trades) ---")
        print(f"{'Variant':<20} {'P&L':>10} {'WR':>6} {'W':>4} {'L':>4} {'AvgW':>8} {'AvgL':>8} {'vs Base':>10}")
        print("-" * 85)

        base_tier_pnl = sum(all_results["baseline"][i]["pnl"] for i in tier_trades)

        for vname in VARIANTS:
            res = all_results[vname]
            tier_res = [res[i] for i in tier_trades]
            wins = [r for r in tier_res if r["pnl"] > 0]
            losses = [r for r in tier_res if r["pnl"] <= 0]
            total_pnl = sum(r["pnl"] for r in tier_res)
            wr = len(wins) / len(tier_res) * 100 if tier_res else 0
            avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
            avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
            delta = total_pnl - base_tier_pnl
            delta_str = f"${delta:>+,.0f}" if vname != "baseline" else "---"

            print(f"{vname:<20} ${total_pnl:>+9,.0f} {wr:>5.1f}% {len(wins):>4} {len(losses):>4} "
                  f"${avg_w:>+7,.0f} ${avg_l:>+7,.0f} {delta_str:>10}")

    # ── Per-ticker breakdown for best variant ──
    best_variant = max(
        (v for v in VARIANTS if v != "baseline"),
        key=lambda v: sum(r["pnl"] for r in all_results[v])
    )

    print(f"\n{'=' * 120}")
    print(f"PER-TICKER BREAKDOWN: baseline vs {best_variant}")
    print(f"{'=' * 120}")

    tickers_seen = sorted(set(inp["ticker"] for inp in trade_inputs))
    print(f"\n{'Ticker':<8} {'Tier':<5} {'#':>3} {'Base P&L':>10} {'Best P&L':>10} {'Delta':>8} "
          f"{'Base WR':>7} {'Best WR':>7}")
    print("-" * 70)

    for tk in tickers_seen:
        tk_indices = [i for i, inp in enumerate(trade_inputs) if inp["ticker"] == tk]
        base_tk = [all_results["baseline"][i] for i in tk_indices]
        best_tk = [all_results[best_variant][i] for i in tk_indices]
        b_pnl = sum(r["pnl"] for r in base_tk)
        v_pnl = sum(r["pnl"] for r in best_tk)
        b_wr = sum(1 for r in base_tk if r["pnl"] > 0) / len(base_tk) * 100
        v_wr = sum(1 for r in best_tk if r["pnl"] > 0) / len(best_tk) * 100
        delta = v_pnl - b_pnl
        marker = " <--" if delta > 100 else (" !!!" if delta < -100 else "")
        print(f"{tk:<8} {_get_tier(tk):<5} {len(tk_indices):>3} ${b_pnl:>+9,.0f} ${v_pnl:>+9,.0f} "
              f"${delta:>+7,.0f} {b_wr:>6.0f}% {v_wr:>6.0f}%{marker}")

    # ── Gate breakdown for baseline and best variant ──
    for vname in ["baseline", best_variant]:
        if vname == "baseline" and best_variant == "baseline":
            continue
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

    # ── Head-to-head trades where they diverge ──
    if best_variant != "baseline":
        base_res = all_results["baseline"]
        best_res = all_results[best_variant]

        print(f"\n{'=' * 120}")
        print(f"DIVERGENT TRADES: baseline vs {best_variant}")
        print(f"{'=' * 120}")
        print(f"\n{'#':<4} {'Ticker':<7} {'Tier':<5} {'Day':<12} "
              f"{'Base P&L':>10} {'Base Gate':<20} "
              f"{'Best P&L':>10} {'Best Gate':<20} {'Delta':>8}")
        print("-" * 115)

        divergent = []
        for i, (b, v) in enumerate(zip(base_res, best_res)):
            delta = v["pnl"] - b["pnl"]
            if abs(delta) > 1:  # Only show trades that actually differ
                divergent.append((abs(delta), i, b, v, delta))

        divergent.sort(reverse=True)
        net_better = sum(1 for _, _, _, _, d in divergent if d > 0)
        net_worse = sum(1 for _, _, _, _, d in divergent if d < 0)
        print(f"({len(divergent)} trades differ: {net_better} improved, {net_worse} worsened)\n")

        for _, i, b, v, delta in divergent[:25]:
            marker = " <--" if delta > 100 else (" !!!" if delta < -100 else "")
            print(f"{i+1:<4} {b['ticker']:<7} {b['tier']:<5} {b['day']:<12} "
                  f"${b['pnl']:>+9.0f} {b['reason']:<20} "
                  f"${v['pnl']:>+9.0f} {v['reason']:<20} "
                  f"${delta:>+7.0f}{marker}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
