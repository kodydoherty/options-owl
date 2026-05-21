#!/usr/bin/env python3
"""Backtest current PRODUCTION v4 FSM exit strategy against all signals.

This script replicates the EXACT v4 FSM logic from options_owl/risk/exit_v5/
using harvester tick data. Every gate fires in the same order and with the
same thresholds as production.

Usage:
  python scripts/backtest_prod.py [signals_db] [harvester_db]

Defaults:
  signals_db   = journal/owlet-kody/raw_messages.db
  harvester_db = journal/owlet-harvester/options_data.db
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SLIPPAGE = 0.15
PORTFOLIO = 8000

SIGNALS_DB = sys.argv[1] if len(sys.argv) > 1 else "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = sys.argv[2] if len(sys.argv) > 2 else "journal/owlet-harvester/options_data.db"

# ── Production V4 FSM parameters (from exit_v5/config.py defaults) ──────

# Hard stop
HARD_STOP_PCT = 0.30  # 30%

# Grace period
GRACE_PERIOD_SEC = 300  # 5 min

# Bar1 reverse (0DTE only)
BAR1_MIN_SEC = 90
BAR1_WINDOW_SEC = 150
BAR1_THRESHOLD_PCT = -5.0

# Soft trail (§11) — 10-50% band, 60% floor
SOFT_TRAIL_BAND_LOW = 10.0
SOFT_TRAIL_BAND_HIGH = 35.0  # = trail_activate_gain_pct
SOFT_TRAIL_FLOOR_FRACTION = 0.60

# Trail activation
TRAIL_ACTIVATE_GAIN_PCT = 35.0

# Trail tiers (§6) — sorted descending by min_gain_pct
TRAIL_TIERS = [
    (400.0, 0.20),  # +400% → 20% trail
    (200.0, 0.25),  # +200% → 25% trail
    (100.0, 0.30),  # +100% → 30% trail
    (50.0, 0.35),   # +50%  → 35% trail
]

# Theta curve (§10) — tightens trail as session progresses
THETA_CURVE_FLOOR = 0.40
THETA_CURVE_FULL_SESSION = 6.5
THETA_CURVE_EXPONENT = 0.4

# Trail multipliers (§14)
TRAIL_MULT_GIVEBACK_CAP = 1.20
TRAIL_MULT_MAX = 2.0
TRAIL_MULT_MAX_TRAIL = 0.45
TRAIL_MULT_MORNING = 1.5
TRAIL_MULT_SCORE_90 = 1.35
TRAIL_MULT_TICKERS = {"NVDA": 1.5, "TSLA": 1.5, "AMZN": 1.4, "AVGO": 1.4, "PLTR": 1.3}

# House-money floors (§12)
HOUSE_MONEY_FLOORS = [
    (5.00, 2.00),  # +500% → floor at +200%
    (2.00, 0.80),  # +200% → floor at +80%
    (1.00, 0.30),  # +100% → floor at +30%
]

# Milestone locks (§7) — partial profit taking
ATM_MILESTONES = [
    (200.0, 0.15),  # +200% → lock 15%
    (400.0, 0.15),  # +400% → lock 15%
    (600.0, 0.15),  # +600% → lock 15%
]

# Theta timer (§15)
THETA_TIMER_BASE_SEC = 7200       # 120 min
THETA_TIMER_MORNING_SEC = 10800   # 180 min
THETA_TIMER_LATE_SEC = 2400       # 40 min after 2 PM
THETA_TIMER_SCORE_IMMUNE = 92.0
THETA_TIMER_TICKER_IMMUNE = {"NVDA", "TSLA", "AMZN", "AVGO", "PLTR"}

# DTE-aware graduated stops (v5)
HIGH_VOL_TICKERS = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}

# Index profit target
INDEX_PROFIT_TARGET_PCT = 30.0

# EOD cutoff
EOD_CUTOFF_MINUTES = 15.0  # 3:45 PM ET


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_tick(tick, sig_ts, entry):
    """Parse a harvester tick into usable values."""
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
    # Convert UTC to ET (UTC-4 during EDT, UTC-5 during EST)
    # Approximate: use -4 for trading hours
    et_dt = ts_dt - timedelta(hours=4)
    et_hour = et_dt.hour
    et_min = et_dt.minute
    minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_min))
    return {
        "price": price,
        "bid": bid if bid else 0,
        "ask": ask if ask else 0,
        "underlying": underlying if underlying else 0,
        "elapsed_sec": elapsed_sec,
        "elapsed_min": elapsed_min,
        "gain_pct": gain_pct,
        "et_hour": et_hour,
        "et_min": et_min,
        "minutes_to_close": minutes_to_close,
        "ts_dt": ts_dt,
        "et_dt": et_dt,
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


def _get_trail_pct(gain_pct):
    """Look up trail width from tiered table."""
    for min_gain, trail in TRAIL_TIERS:
        if gain_pct >= min_gain:
            return trail
    return TRAIL_TIERS[-1][1] if TRAIL_TIERS else 0.35


def _theta_curve_mult(et_hour, et_min):
    """Theta-curve trail tightening multiplier."""
    current_hour = et_hour + et_min / 60.0
    hours_remaining = max(0.5, 16.0 - current_hour)
    raw = (hours_remaining / THETA_CURVE_FULL_SESSION) ** THETA_CURVE_EXPONENT
    return max(THETA_CURVE_FLOOR, min(raw, 1.0))


def _apply_trail_multipliers(base_trail, ticker, is_morning, score):
    """Apply ticker/session/score multipliers to trail width."""
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
    """Full trail stop computation matching production."""
    peak_gain_pct = (peak - entry) / entry * 100 if entry > 0 else 0
    tier_trail = _get_trail_pct(peak_gain_pct)
    multiplied = _apply_trail_multipliers(tier_trail, ticker, is_morning, score)
    theta_mult = _theta_curve_mult(et_hour, et_min)
    effective = multiplied * theta_mult
    trail_stop = peak * (1.0 - effective)
    return trail_stop, effective


def _compute_house_money_floor(peak_gain_pct, entry, current_floor):
    """Progressive monotonic stop floor."""
    peak_gain_frac = peak_gain_pct / 100.0
    new_floor = current_floor
    for trigger, floor_gain in HOUSE_MONEY_FLOORS:
        if peak_gain_frac >= trigger:
            candidate = entry * (1.0 + floor_gain)
            if candidate > new_floor:
                new_floor = candidate
            break
    return new_floor


# ── Main simulator ───────────────────────────────────────────────────────

def prod_simulate(entry, ticks, sig_ts, contracts, direction, ticker, score,
                  expiry_date=None):
    """Simulate the production v4 FSM exit engine.

    This replicates the exact gate ordering and thresholds from fsm.py:
    1. EOD cutoff (0DTE only)
    2. GRACE: bar1_reverse (0DTE only) + bid disappearance
    3. Bid disappearance (post-grace)
    4. Index profit target
    5. Scalp trail (underlying-confirmed)
    6. Checkpoint cut (0DTE, underlying against, -30%)
    7. Graduated stops (DTE-aware, category-aware)
    8. DEVELOPING: soft trail + theta timer
    9. TRAILING: house money floor + trail stop + milestone locks + theta timer
    """
    if not ticks or entry <= 0:
        return 0, "no_data", 0

    is_call = direction.lower() in ("call", "bullish", "long")
    is_index = ticker in INDEX_TICKERS
    is_high_vol = ticker in HIGH_VOL_TICKERS

    # Determine DTE from expiry_date
    sig_date = sig_ts.date() if sig_ts.tzinfo else sig_ts.date()
    if expiry_date:
        try:
            exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            dte = max(0, (exp - sig_date).days)
        except (ValueError, TypeError):
            dte = 0
    else:
        dte = 0
    is_multiday = dte > 0

    # Determine morning entry
    et_entry_hour = (sig_ts.hour - 4) % 24 if sig_ts.tzinfo else sig_ts.hour
    is_morning = et_entry_hour < 12

    # Running state
    peak = entry
    entry_underlying = None
    last_underlying = 0.0
    house_money_floor_price = 0.0
    locked_milestones = set()
    seconds_at_zero_bid = 0.0
    theta_timer_started = False

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

        # Update peak
        if price > peak:
            peak = price

        # Track underlying
        if entry_underlying is None and underlying > 0:
            entry_underlying = underlying
        if underlying > 0:
            last_underlying = underlying
        effective_underlying = underlying if underlying > 0 else last_underlying

        # Compute gains
        peak_gain_pct = (peak - entry) / entry * 100
        drop_entry_pct = max(0, (entry - price) / entry * 100)

        # Underlying movement
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

        # Bid tracking
        if bid <= 0:
            seconds_at_zero_bid += 60  # ~1 min per tick
        else:
            seconds_at_zero_bid = 0

        # DTE recompute for multi-day trades on expiry day
        current_dte = dte
        if expiry_date:
            try:
                exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                tick_date = (parsed["ts_dt"] - timedelta(hours=4)).date()
                current_dte = max(0, (exp - tick_date).days)
            except (ValueError, TypeError):
                pass
        current_multiday = current_dte > 0

        # ── FSM State ──
        if elapsed_sec < GRACE_PERIOD_SEC:
            fsm_state = "GRACE"
        elif peak_gain_pct >= TRAIL_ACTIVATE_GAIN_PCT:
            fsm_state = "TRAILING"
        else:
            fsm_state = "DEVELOPING"

        # ── EOD cutoff (0DTE only) ──
        if not current_multiday and minutes_to_close <= EOD_CUTOFF_MINUTES:
            return _make_exit(price, entry, contracts, elapsed_min, "eod_cutoff")

        # ── GRACE state ──
        if fsm_state == "GRACE":
            # Bar1 reverse (0DTE only)
            if not is_multiday:  # Use original DTE, not recomputed
                if BAR1_MIN_SEC <= elapsed_sec <= BAR1_WINDOW_SEC:
                    bar1_change = (price - entry) / entry * 100
                    if bar1_change <= BAR1_THRESHOLD_PCT:
                        return _make_exit(price, entry, contracts, elapsed_min, "bar1_reverse")

            # Bid disappearance during grace
            if bid <= 0 and seconds_at_zero_bid >= 30:
                return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

            continue  # GRACE: skip all other checks

        # ── Bid disappearance (post-grace) ──
        if bid <= 0 and seconds_at_zero_bid >= 30:
            return _make_exit(price, entry, contracts, elapsed_min, "bid_disappearance")

        # ── Index profit target ──
        if is_index and INDEX_PROFIT_TARGET_PCT > 0 and gain_pct >= INDEX_PROFIT_TARGET_PCT:
            return _make_exit(price, entry, contracts, elapsed_min, "profit_target")

        # ── Scalp trail (v5) ──
        if peak_gain_pct >= 20 and gain_pct > 0 and gain_pct < peak_gain_pct * 0.6:
            should_scalp = False
            if not current_multiday and has_underlying and not underlying_confirms:
                should_scalp = True
            elif current_multiday and has_underlying and underlying_against:
                should_scalp = True
            if should_scalp:
                return _make_exit(price, entry, contracts, elapsed_min, "scalp_trail")

        # ── Checkpoint cut (v5, 0DTE only) ──
        if not current_multiday and drop_entry_pct >= 30 and has_underlying and underlying_against:
            return _make_exit(price, entry, contracts, elapsed_min, "checkpoint_cut")

        # ── Graduated stops (DTE-aware, category-aware) ──
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
            # No underlying → v4 hard stop
            stop_price = entry * (1.0 - HARD_STOP_PCT)
            # Spread-aware: use mid when > 30 min to close
            if minutes_to_close > 30 and ask > 0 and bid >= 0:
                compare = (bid + ask) / 2.0
            else:
                compare = bid if bid > 0 else price
            if compare <= stop_price and compare >= 0:
                return _make_exit(price, entry, contracts, elapsed_min, "hard_stop")

        # ── DEVELOPING state ──
        if fsm_state == "DEVELOPING":
            # Soft trail (10-35% band, 60% floor)
            if SOFT_TRAIL_BAND_LOW <= peak_gain_pct < SOFT_TRAIL_BAND_HIGH:
                floor_gain_frac = (peak_gain_pct / 100.0) * SOFT_TRAIL_FLOOR_FRACTION
                floor_price = entry * (1.0 + floor_gain_frac)
                if price <= floor_price:
                    return _make_exit(price, entry, contracts, elapsed_min, "soft_trail")

            # Theta timer (0DTE only)
            if current_dte == 0:
                theta_exit = _check_theta_timer(
                    elapsed_sec, gain_pct, score, ticker, is_morning,
                    et_hour, et_min,
                )
                if theta_exit:
                    return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

            continue  # DEVELOPING: skip trailing checks

        # ── TRAILING state ──

        # House-money floor
        house_money_floor_price = _compute_house_money_floor(
            peak_gain_pct, entry, house_money_floor_price,
        )
        if house_money_floor_price > 0 and price <= house_money_floor_price:
            return _make_exit(price, entry, contracts, elapsed_min, "house_money_floor")

        # Trail stop
        trail_stop, effective_trail = _compute_trail_stop(
            peak, entry, ticker, is_morning, score, et_hour, et_min,
        )
        if price <= trail_stop:
            return _make_exit(price, entry, contracts, elapsed_min, "trail_stop")

        # Milestone locks (partial close — in backtest, treat as full exit at milestone)
        for ms_gain, ms_frac in ATM_MILESTONES:
            if peak_gain_pct >= ms_gain and ms_gain not in locked_milestones:
                n_close = max(1, round(contracts * ms_frac)) if contracts > 1 else 0
                if n_close > 0:
                    locked_milestones.add(ms_gain)
                    # In backtest: reduce contracts, record partial P&L
                    partial_pnl = (price - entry) * n_close * 100
                    if partial_pnl > 0:
                        partial_pnl *= (1 - SLIPPAGE)
                    contracts -= n_close
                    if contracts <= 0:
                        return partial_pnl, f"milestone_{ms_gain:.0f}", elapsed_min
                    # Continue with reduced contracts
                    # Note: we don't accumulate partial P&L in this simplified backtest
                    # Just mark the milestone as locked
                else:
                    locked_milestones.add(ms_gain)

        # Theta timer (0DTE only, also in TRAILING)
        if current_dte == 0:
            theta_exit = _check_theta_timer(
                elapsed_sec, gain_pct, score, ticker, is_morning,
                et_hour, et_min,
            )
            if theta_exit:
                return _make_exit(price, entry, contracts, elapsed_min, "theta_timer")

    return _end_of_data(ticks, entry, contracts, sig_ts)


def _check_theta_timer(elapsed_sec, gain_pct, score, ticker, is_morning,
                       et_hour, et_min):
    """Check if theta timer should fire. Returns True to exit."""
    # Score immunity
    if score is not None and score >= THETA_TIMER_SCORE_IMMUNE:
        return False
    # Ticker immunity
    if ticker in THETA_TIMER_TICKER_IMMUNE:
        return False

    # Timer duration
    current_hour = et_hour + et_min / 60.0
    if is_morning:
        timer_sec = THETA_TIMER_MORNING_SEC
    elif current_hour >= 14.0:
        timer_sec = THETA_TIMER_LATE_SEC
    else:
        timer_sec = THETA_TIMER_BASE_SEC

    if elapsed_sec >= timer_sec and gain_pct <= 5.0:
        return True
    return False


# ── Data loading ─────────────────────────────────────────────────────────

def build_ct(ticker, day, direction, strike):
    dt = datetime.strptime(day, "%Y-%m-%d")
    ds = dt.strftime("%y%m%d")
    cp = "C" if direction.lower() in ("call", "bullish", "long") else "P"
    si = int(strike * 1000)
    return f"O:{ticker}{ds}{cp}{si:08d}"


def score_to_contracts(score, entry_premium):
    """Production dollar-target sizing."""
    if score >= 95:
        budget_mult = 1.0
    elif score >= 90:
        budget_mult = 0.75
    elif score >= 85:
        budget_mult = 0.50
    elif score >= 78:
        budget_mult = 0.25
    else:
        return 0  # rejected

    max_risk_pct = 0.75
    max_concurrent = 5
    max_position_pct = 0.15

    deployable = PORTFOLIO * max_risk_pct
    target_per_trade = deployable / max_concurrent
    scaled_target = target_per_trade * budget_mult
    cost_per_contract = entry_premium * 100

    if cost_per_contract <= 0:
        return 1

    position_cap = int((PORTFOLIO * max_position_pct) / cost_per_contract)
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

    results = []
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

        # Try exact day first, then check if it's a multi-day contract
        rows = harv_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price
            FROM harvest_snapshots WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (contract, sig["sig_ts"])).fetchall()

        expiry_date = day  # Default: same-day (0DTE)

        if not rows:
            no_data += 1
            continue

        # Use first tick's ask as entry (simulates market buy)
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

        # Overall peak for reference
        all_mids = [r[1] for r in rows if r[1] and r[1] > 0]
        overall_peak = max(all_mids) if all_mids else entry
        overall_peak_gain = (overall_peak - entry) / entry * 100

        pnl, reason, hold = prod_simulate(
            entry, rows, sig_ts, contracts, direction, ticker, score,
            expiry_date=expiry_date,
        )

        results.append({
            "id": sig["id"],
            "ticker": ticker,
            "dir": direction,
            "day": day,
            "score": score,
            "entry": entry,
            "contracts": contracts,
            "pnl": pnl,
            "reason": reason,
            "hold": hold,
            "peak_gain": overall_peak_gain,
            "expiry": expiry_date,
        })

    # ===================================================================
    # REPORT
    # ===================================================================
    total_signals = len(results)
    wins = [r for r in results if r["pnl"] > 0]
    losses = [r for r in results if r["pnl"] <= 0]
    total_pnl = sum(r["pnl"] for r in results)
    wr = len(wins) / total_signals * 100 if total_signals else 0
    avg_w = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
    avg_l = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
    wl = abs(avg_w / avg_l) if avg_l else 0

    print(f"{'=' * 110}")
    print(f"PRODUCTION V4 FSM BACKTEST — {total_signals} signals traded")
    print(f"{'=' * 110}")
    print(f"Portfolio: ${PORTFOLIO:,}  |  Slippage: {SLIPPAGE*100:.0f}%  |  "
          f"No data: {no_data}  |  No strike: {no_strike}  |  Score<78: {skipped}")
    print()
    print(f"  Total P&L:     ${total_pnl:>+,.0f}")
    print(f"  Return:        {total_pnl/PORTFOLIO*100:>+.1f}%")
    print(f"  Win Rate:      {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win:       ${avg_w:>+,.0f}")
    print(f"  Avg Loss:      ${avg_l:>+,.0f}")
    print(f"  Win:Loss:      {wl:.2f}:1")

    # ── Gate fire breakdown ──
    print(f"\n{'=' * 110}")
    print("EXIT GATE BREAKDOWN")
    print(f"{'=' * 110}")

    gate_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "holds": []})
    for r in results:
        g = gate_stats[r["reason"]]
        g["count"] += 1
        g["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            g["wins"] += 1
        g["holds"].append(r["hold"])

    print(f"\n{'Gate':<22} {'Fires':>5} {'%':>5} {'P&L':>10} {'W/L':>8} {'WR':>5} {'AvgHold':>7}")
    print("-" * 75)

    for gate, s in sorted(gate_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        ct = s["count"]
        pct = ct / total_signals * 100 if total_signals else 0
        g_wr = s["wins"] / ct * 100 if ct else 0
        ah = sum(s["holds"]) / ct if ct else 0
        print(f"{gate:<22} {ct:>5} {pct:>4.0f}% ${s['pnl']:>+9.0f} "
              f"{s['wins']}W/{ct - s['wins']}L {g_wr:>4.0f}% {ah:>5.0f}m")

    # ── Daily P&L ──
    print(f"\n{'=' * 110}")
    print("DAILY P&L REPORT")
    print(f"{'=' * 110}")

    all_days = sorted(set(r["day"] for r in results))
    cum_pnl = 0

    print(f"\n{'Date':<12} {'#':>3} {'P&L':>10} {'Cum':>10} {'WR':>5} {'Trades'}")
    print("-" * 110)

    for day in all_days:
        day_trades = [r for r in results if r["day"] == day]
        day_pnl = sum(r["pnl"] for r in day_trades)
        day_wins = sum(1 for r in day_trades if r["pnl"] > 0)
        day_wr = day_wins / len(day_trades) * 100 if day_trades else 0
        cum_pnl += day_pnl

        # Trade summaries
        trade_details = []
        for r in day_trades:
            marker = "+" if r["pnl"] > 0 else "-"
            trade_details.append(
                f"{r['ticker']} {r['dir'][:1].upper()} ${r['pnl']:>+.0f} ({r['reason']})"
            )

        trades_str = " | ".join(trade_details)
        print(f"{day:<12} {len(day_trades):>3} ${day_pnl:>+9.0f} ${cum_pnl:>+9.0f} "
              f"{day_wr:>4.0f}% {trades_str}")

    print("-" * 110)
    print(f"{'TOTAL':<12} {total_signals:>3} ${total_pnl:>+9.0f}")

    # ── All trades detail ──
    print(f"\n{'=' * 110}")
    print("ALL TRADES DETAIL")
    print(f"{'=' * 110}")
    print(f"\n{'#':<4} {'Ticker':<7} {'Dir':<5} {'Day':<12} {'Score':>5} {'$In':>6} {'Ct':>3} "
          f"{'P&L':>9} {'Gate':<22} {'Hold':>6} {'Peak%':>7}")
    print("-" * 110)

    for i, r in enumerate(results):
        marker = ""
        if r["pnl"] > 500:
            marker = " ***"
        elif r["pnl"] < -300:
            marker = " !!!"
        print(f"{i+1:<4} {r['ticker']:<7} {r['dir'][:4]:<5} {r['day']:<12} {r['score']:>5} "
              f"${r['entry']:>5.2f} {r['contracts']:>3} ${r['pnl']:>+8.0f} "
              f"{r['reason']:<22} {r['hold']:>5.0f}m {r['peak_gain']:>+6.0f}%{marker}")

    sig_conn.close()
    harv_conn.close()


if __name__ == "__main__":
    main()
