"""Replay actual Discord signals against harvester price data.

Uses the real trade_signals from Discord and minute-by-minute option snapshots
from the harvester DB to simulate what would have happened with the current
exit pipeline config.

Usage:
    python scripts/replay_signals.py
"""

import os

# Suppress noisy debug logs before any imports
os.environ["LOGURU_LEVEL"] = "WARNING"

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options_owl.config.settings import Settings
from options_owl.risk.ml_exit import predict_sell, MLSellSignal
from options_owl.risk.ml_v2 import predict_entry, predict_peak, predict_regime, EntrySignal, PeakSignal, RegimeSignal
from options_owl.risk.vinny_strategy import (
    check_theta_bleed,
    check_time_decay_no_new_high,
    evaluate_adaptive_trail,
    evaluate_dollar_trail,
    is_time_decay_zone,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNALS_DB = os.environ.get(
    "SIGNALS_DB", "journal/owlet-kody/raw_messages.db"
)
# Legacy DB from before per-owlet journal directories (has Mar 27 – Apr 9 data)
LEGACY_SIGNALS_DB = os.environ.get(
    "LEGACY_SIGNALS_DB", "journal/raw_messages.db"
)
HARVESTER_DB = os.environ.get(
    "HARVESTER_DB", "journal/owlet-harvester/options_data.db"
)

ET_TZ = ZoneInfo("America/New_York")  # auto-handles EDT/EST
BALANCE = 5000.0  # Kody's portfolio size


def _to_et(utc_dt: datetime) -> datetime:
    """Convert a UTC datetime to Eastern Time (handles EDT/EST automatically)."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(ET_TZ)


@dataclass
class ReplayResult:
    ticker: str
    direction: str
    strike: float
    signal_time: str
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    mfe_pct: float  # max favorable excursion
    mfe_gap: float  # money left on table (mfe - actual)
    exit_reason: str
    duration_min: float
    contracts: int
    pnl_dollars: float


def build_contract_ticker(underlying: str, expiry_date: str, strike: float, option_type: str) -> str:
    """Build Polygon-style contract ticker like O:SPY260417C00709000."""
    # Parse expiry date
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type == "call" else "P"
    # Strike is in units of 1/1000 dollar, 8 digits
    strike_int = int(strike * 1000)
    return f"O:{underlying}{date_str}{opt_char}{strike_int:08d}"


def resolve_expiry(signal_time_str: str) -> str:
    """0DTE = same day as signal."""
    dt = datetime.fromisoformat(signal_time_str)
    return dt.strftime("%Y-%m-%d")


def get_snapshots(harvester_conn, contract_ticker: str, after_time: str):
    """Get minute-by-minute snapshots for a contract after signal time."""
    rows = harvester_conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
        FROM harvest_snapshots
        WHERE contract_ticker = ?
          AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_time)).fetchall()
    return rows


def simulate_exit_pipeline(
    snapshots: list,
    entry_premium: float,
    signal_time: datetime,
    settings: Settings,
    targets: dict | None = None,
    ticker: str = "",
    is_call: bool = True,
    enable_ml: bool = False,
    ml_override_trails: bool = False,
    ml_sell_threshold: float = 0.4,
    ml_min_minutes: float = 0.0,
    peak_signal: PeakSignal | None = None,
    regime_signal: RegimeSignal | None = None,
) -> tuple[float, float, str, float, int]:
    """Run snapshots through the exit pipeline.

    Args:
        enable_ml: If True, run ML predictions at each snapshot.
        ml_override_trails: If True, ML hold suppresses dollar/adaptive trail exits.
        peak_signal: ML v2 peak prediction — adjusts trail widths dynamically.
        regime_signal: ML v2 regime classification — multiplies trail widths.

    Returns: (exit_premium, peak_premium, exit_reason, duration_min, last_target_hit)
    """
    if not snapshots:
        return entry_premium, entry_premium, "no_data", 0.0, 0


    peak_premium = entry_premium
    last_new_high_at = signal_time
    last_target_hit = 0
    premium_history: list[float] = []

    # Default targets as % gain from entry
    if targets is None:
        targets = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

    # Profit lock tiers: parse from settings
    profit_lock_tiers = []
    if settings.ENABLE_PROFIT_LOCK:
        for tier_str in settings.PROFIT_LOCK_TIERS.split(","):
            parts = tier_str.strip().split(":")
            if len(parts) == 2:
                profit_lock_tiers.append((float(parts[0]), float(parts[1])))
    profit_lock_tiers.sort(key=lambda x: x[0], reverse=True)

    locked_floor_pct = None  # minimum P&L % we'll accept once locked

    for snap in snapshots:
        captured_at_str, midpoint, bid, ask, underlying_price = snap

        # Use midpoint, fall back to bid/ask avg
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        premium_history.append(price)

        captured_at = datetime.fromisoformat(captured_at_str)
        # Normalize timezone awareness
        if captured_at.tzinfo is not None and signal_time.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=None)
        elif captured_at.tzinfo is None and signal_time.tzinfo is not None:
            captured_at = captured_at.replace(tzinfo=signal_time.tzinfo)
        elapsed_min = (captured_at - signal_time).total_seconds() / 60

        # Update peak
        if price > peak_premium:
            peak_premium = price
            last_new_high_at = captured_at

        # Update targets hit
        gain_pct = (price - entry_premium) / entry_premium * 100
        for t_num in range(1, 6):
            if gain_pct >= targets[t_num] and t_num > last_target_hit:
                last_target_hit = t_num

        # Update profit lock
        if profit_lock_tiers:
            for threshold, lock in profit_lock_tiers:
                peak_gain = (peak_premium - entry_premium) / entry_premium * 100
                if peak_gain >= threshold:
                    locked_floor_pct = lock
                    break

        # --- EXIT GATES (in order) ---

        # 1. Grace period — no exits for first N minutes
        if elapsed_min < settings.STOP_GRACE_PERIOD_MINUTES:
            continue

        # Convert UTC to ET for time-of-day checks
        captured_et = _to_et(captured_at)

        # 2. Expiry safety — force close N min before market close (4 PM ET)
        market_close_et = captured_et.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=captured_et.tzinfo)
        minutes_to_close = (market_close_et - captured_et).total_seconds() / 60
        if 0 < minutes_to_close <= settings.EXPIRY_SAFETY_MINUTES:
            return price, peak_premium, "expiry_safety", elapsed_min, last_target_hit

        # 3. Premium stop (hard stop loss) — peak-aware version
        #    If trade never peaked (MFE ~0%), use normal stop from entry.
        #    If trade had a peak, tighten: stop at max(entry-based stop, peak-based stop).
        if settings.PREMIUM_STOP_ENABLED:
            loss_from_entry = (entry_premium - price) / entry_premium * 100
            peak_gain_pct = (peak_premium - entry_premium) / entry_premium * 100
            # Peak-aware: if trade peaked, add a stop relative to peak
            # e.g. peaked at +20%, peak_stop = peak * (1 - 0.40) = peak * 0.60
            # This means: once you've seen gains, don't give it ALL back
            peak_stop_pct = getattr(settings, 'PEAK_STOP_DROP_PCT', 40.0)
            peak_stop_price = peak_premium * (1 - peak_stop_pct / 100) if peak_gain_pct > 5.0 else 0
            entry_stop_price = entry_premium * (1 - settings.PREMIUM_STOP_PCT / 100)
            effective_stop = max(entry_stop_price, peak_stop_price)
            if price <= effective_stop:
                if peak_stop_price > entry_stop_price and peak_gain_pct > 5.0:
                    return price, peak_premium, f"peak_stop(peak+{peak_gain_pct:.0f}%)", elapsed_min, last_target_hit
                else:
                    return price, peak_premium, "premium_stop", elapsed_min, last_target_hit

        # 4. ML sell — runs BEFORE trails so it can cut losers early
        #    We re-evaluate the raw predictions here with custom thresholds
        #    instead of using predict_sell()'s hardcoded SELL_THRESHOLD=0.4
        ml_holding = False
        if enable_ml and ticker and elapsed_min >= ml_min_minutes:
            ml_signal = predict_sell(
                ticker=ticker,
                entry_premium=entry_premium,
                current_premium=price,
                peak_premium=peak_premium,
                minutes_since_entry=elapsed_min,
                now_hour=captured_et.hour,
                now_minute=captured_et.minute,
                is_call=is_call,
                premium_history=premium_history[-20:] if len(premium_history) > 1 else None,
                underlying_entry=None,
                underlying_current=underlying_price,
            )
            # Apply custom threshold (override predict_sell's built-in decision)
            custom_sell = (
                (ml_signal.sell_probability > ml_sell_threshold and ml_signal.expected_future_pnl < 2.0)
                or (elapsed_min >= 10 and ml_signal.expected_future_pnl < -10.0)
            )
            if custom_sell:
                return price, peak_premium, f"ml_sell(P={ml_signal.sell_probability:.2f},E={ml_signal.expected_future_pnl:+.0f}%)", elapsed_min, last_target_hit
            # ML says hold — check if expected upside is strong enough to override trails
            if ml_override_trails and ml_signal.expected_future_pnl >= 5.0:
                ml_holding = True

        # 5. Profit lock ratchet
        if locked_floor_pct is not None:
            current_gain = (price - entry_premium) / entry_premium * 100
            if current_gain <= locked_floor_pct:
                return price, peak_premium, f"profit_lock(floor={locked_floor_pct:.0f}%)", elapsed_min, last_target_hit

        # ML v2 trail adjustments: peak predictor suggests trail width, regime multiplies it
        regime_mult = regime_signal.suggested_trail_multiplier if regime_signal else 1.0

        # 6. Dollar trail (stair-step trailing stop) — ML can override
        if getattr(settings, 'ENABLE_DOLLAR_TRAIL', True):
            # Peak predictor can widen/tighten the dollar trail steps
            dt_small_step = getattr(settings, 'DOLLAR_TRAIL_SMALL_STEP_PCT', 10.0)
            dt_large_step = getattr(settings, 'DOLLAR_TRAIL_LARGE_STEP_PCT', 5.0)
            dt_activation = getattr(settings, 'DOLLAR_TRAIL_ACTIVATION_PCT', 10.0)
            if peak_signal and peak_signal.predicted_mfe_pct >= 100:
                # Big winner predicted — widen dollar trail steps by 50%
                dt_small_step *= 1.5
                dt_large_step *= 1.5
                dt_activation *= 1.5
            elif peak_signal and peak_signal.predicted_mfe_pct < 20:
                # Low MFE — tighten steps to take profits faster
                dt_small_step *= 0.7
                dt_large_step *= 0.7
                dt_activation *= 0.7
            # Apply regime multiplier
            dt_small_step *= regime_mult
            dt_large_step *= regime_mult

            dt_result = evaluate_dollar_trail(
                entry_premium, price, peak_premium,
                activation_pct=dt_activation,
                small_step_pct=dt_small_step,
                step_threshold_pct=getattr(settings, 'DOLLAR_TRAIL_STEP_THRESHOLD_PCT', 25.0),
                large_step_pct=dt_large_step,
            )
            if dt_result.should_exit:
                if ml_holding:
                    pass  # ML overrides — keep holding
                else:
                    return price, peak_premium, "dollar_trail", elapsed_min, last_target_hit

        # 7. Adaptive 3-stage trailing stop — ML can override
        if getattr(settings, 'ENABLE_ADAPTIVE_TRAIL', True):
            # Peak predictor adjusts adaptive trail widths
            active_w = getattr(settings, 'ADAPTIVE_TRAIL_ACTIVE_WIDTH', 35.0)
            runner_w = getattr(settings, 'ADAPTIVE_TRAIL_RUNNER_WIDTH', 45.0)
            moonshot_w = getattr(settings, 'ADAPTIVE_TRAIL_MOONSHOT_WIDTH', 30.0)
            if peak_signal:
                # Use peak predictor's suggested trail width as the base active width
                active_w = peak_signal.suggested_trail_width
                runner_w = max(active_w + 10, runner_w)  # runner always wider than active
                moonshot_w = max(active_w - 5, moonshot_w)  # moonshot a bit tighter
            # Apply regime multiplier to all widths
            active_w *= regime_mult
            runner_w *= regime_mult
            moonshot_w *= regime_mult

            at_result = evaluate_adaptive_trail(
                entry_premium, price, peak_premium,
                activation_pct=getattr(settings, 'ADAPTIVE_TRAIL_ACTIVATION_PCT', 40.0),
                active_width=active_w,
                runner_threshold=getattr(settings, 'ADAPTIVE_TRAIL_RUNNER_THRESHOLD', 150.0),
                runner_width=runner_w,
                moonshot_threshold=getattr(settings, 'ADAPTIVE_TRAIL_MOONSHOT_THRESHOLD', 400.0),
                moonshot_width=moonshot_w,
            )
            if at_result.should_exit:
                if ml_holding:
                    pass  # ML overrides — keep holding
                else:
                    return price, peak_premium, f"adaptive_trail({at_result.stage})", elapsed_min, last_target_hit

        # 8. Time decay zone checks (pass ET-converted times for afternoon check)
        signal_time_et = _to_et(signal_time)
        in_decay = is_time_decay_zone(
            signal_time_et, captured_et,
            max_hold_minutes=settings.TIME_DECAY_HOLD_MINUTES,
            afternoon_hour=settings.TIME_DECAY_AFTERNOON_HOUR,
            afternoon_minute=settings.TIME_DECAY_AFTERNOON_MINUTE,
        )
        if in_decay:
            stale_exit, _ = check_time_decay_no_new_high(
                price, peak_premium, last_new_high_at, captured_at,
                stale_minutes=settings.TIME_DECAY_STALE_MINUTES,
            )
            if stale_exit:
                return price, peak_premium, "time_decay_stale", elapsed_min, last_target_hit

        # 9. Theta bleed
        should_exit, _ = check_theta_bleed(
            entry_premium, price, signal_time, captured_at,
            max_hold_minutes=settings.THETA_BLEED_HOLD_MINUTES,
            max_loss_pct=settings.THETA_BLEED_MAX_LOSS_PCT,
        )
        if should_exit:
            return price, peak_premium, "theta_bleed", elapsed_min, last_target_hit

        # 10. No momentum exit
        if settings.ENABLE_NO_MOMENTUM_EXIT and elapsed_min >= settings.NO_MOMENTUM_MINUTES:
            momentum_gain = (price - entry_premium) / entry_premium * 100
            if momentum_gain < settings.NO_MOMENTUM_MIN_GAIN_PCT:
                return price, peak_premium, "no_momentum", elapsed_min, last_target_hit

    # If we get through all snapshots without exit, use last price (market close)
    if snapshots:
        last_snap = snapshots[-1]
        last_price = last_snap[1] or ((last_snap[2] or 0) + (last_snap[3] or 0)) / 2
        last_time = datetime.fromisoformat(last_snap[0])
        # Normalize timezone awareness
        if last_time.tzinfo is not None and signal_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=None)
        elif last_time.tzinfo is None and signal_time.tzinfo is not None:
            last_time = last_time.replace(tzinfo=signal_time.tzinfo)
        elapsed = (last_time - signal_time).total_seconds() / 60
        return last_price, peak_premium, "market_close", elapsed, last_target_hit

    return entry_premium, entry_premium, "no_data", 0.0, 0


def run_scenario(name: str, overrides: dict, all_signals: list, harvester_conn,
                 enable_ml: bool = False, ml_override_trails: bool = False,
                 ml_sell_threshold: float = 0.4, ml_min_minutes: float = 0.0,
                 entry_filter_threshold: float = 0.0,
                 entry_filter_afternoon: bool = False,
                 use_peak_predictor: bool = False,
                 use_regime_classifier: bool = False):
    """Run a named scenario with config overrides.

    entry_filter_threshold: if > 0, use ML v2 entry filter to reject signals
    below this confidence threshold.
    entry_filter_afternoon: if True, use the afternoon-calibrated entry model.
    use_peak_predictor: if True, use ML v2 peak predictor to adjust trail widths.
    use_regime_classifier: if True, use ML v2 regime classifier to adjust trail multiplier.
    """
    settings = Settings()
    for k, v in overrides.items():
        setattr(settings, k, v)

    signals = all_signals

    from options_owl.risk.vinny_strategy import score_to_contracts

    results = []
    skipped = 0
    ml_rejected = 0
    max_concurrent = settings.MAX_CONCURRENT
    open_trades: list[tuple[str, datetime, datetime]] = []
    open_tickers: set[str] = set()
    running_balance = BALANCE  # compound wins/losses

    for sig in signals:
        ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at, t1_pct, t2_pct = sig
        if expiry == "0DTE":
            expiry_date = resolve_expiry(created_at)
        else:
            expiry_date = expiry

        entry_premium = atm_premium
        if entry_premium is None or entry_premium <= 0:
            entry_premium = otm_premium
        if entry_premium is None or entry_premium <= 0:
            skipped += 1
            continue

        signal_time = datetime.fromisoformat(created_at)

        # Entry gates: score, max concurrent, duplicate ticker, min premium
        if score < settings.MIN_SCORE:
            continue
        open_trades = [(t, st, ct) for t, st, ct in open_trades if ct > signal_time]
        open_tickers = {t for t, _, _ in open_trades}
        if len(open_trades) >= max_concurrent:
            continue
        if ticker in open_tickers:
            continue
        if entry_premium < settings.MIN_OPTION_PREMIUM:
            continue

        # ML v2 entry filter — needs underlying context from harvester snapshots
        if entry_filter_threshold > 0:
            signal_et = _to_et(signal_time)
            is_call = direction in ("call", "bullish", "long")

            # Pull underlying price + option context from harvester snapshots
            option_type = "call" if is_call else "put"
            _expiry_date = resolve_expiry(created_at) if expiry == "0DTE" else expiry
            _ct = build_contract_ticker(ticker, _expiry_date, strike, option_type)
            _snaps = get_snapshots(harvester_conn, _ct, (signal_time - timedelta(minutes=30)).isoformat())
            u_price = 0.0
            u_mom_5 = 0.0
            u_mom_10 = 0.0
            prem_mom_5 = 0.0
            prem_mom_10 = 0.0
            if _snaps and len(_snaps) >= 5:
                # Get underlying prices from snapshots
                u_prices = [s[4] for s in _snaps if s[4] and s[4] > 0]
                o_prices = [s[1] if s[1] and s[1] > 0 else ((s[2] or 0) + (s[3] or 0)) / 2 for s in _snaps]
                o_prices = [p for p in o_prices if p > 0]
                if u_prices:
                    u_price = u_prices[-1]
                    if len(u_prices) >= 6 and u_prices[-6] > 0:
                        u_mom_5 = (u_prices[-1] - u_prices[-6]) / u_prices[-6] * 100
                    if len(u_prices) >= 11 and u_prices[-11] > 0:
                        u_mom_10 = (u_prices[-1] - u_prices[-11]) / u_prices[-11] * 100
                if o_prices and len(o_prices) >= 6 and o_prices[-6] > 0:
                    prem_mom_5 = (o_prices[-1] - o_prices[-6]) / o_prices[-6] * 100
                if o_prices and len(o_prices) >= 11 and o_prices[-11] > 0:
                    prem_mom_10 = (o_prices[-1] - o_prices[-11]) / o_prices[-11] * 100

            entry_signal = predict_entry(
                ticker=ticker, entry_premium=entry_premium,
                underlying_price=u_price,
                hour=signal_et.hour, minute=signal_et.minute,
                day_of_week=signal_et.weekday(),
                is_call=is_call,
                underlying_momentum_5m=u_mom_5,
                underlying_momentum_10m=u_mom_10,
                premium_momentum_5m=prem_mom_5,
                premium_momentum_10m=prem_mom_10,
                threshold=entry_filter_threshold,
                afternoon=entry_filter_afternoon,
            )
            if not entry_signal.should_enter:
                ml_rejected += 1
                continue

        option_type = "call" if direction in ("call", "bullish", "long") else "put"
        contract_ticker = build_contract_ticker(ticker, expiry_date, strike, option_type)
        snapshots = get_snapshots(harvester_conn, contract_ticker, created_at)
        if not snapshots:
            earlier = (signal_time - timedelta(minutes=2)).isoformat()
            snapshots = get_snapshots(harvester_conn, contract_ticker, earlier)
        if not snapshots:
            skipped += 1
            continue

        first_snap = snapshots[0]
        actual_entry = first_snap[1]
        if actual_entry and actual_entry > 0:
            entry_premium = actual_entry

        is_call = option_type == "call"

        # ML v2: get peak prediction and regime classification if enabled
        _peak_sig = None
        _regime_sig = None
        if use_peak_predictor or use_regime_classifier:
            signal_et = _to_et(signal_time)
            # Get underlying context for peak/regime predictions
            _pre_snaps = get_snapshots(harvester_conn, contract_ticker, (signal_time - timedelta(minutes=30)).isoformat())
            _u_price = 0.0
            _u_mom_5 = 0.0
            _prem_mom_5 = 0.0
            if _pre_snaps and len(_pre_snaps) >= 5:
                _u_prices = [s[4] for s in _pre_snaps if s[4] and s[4] > 0]
                _o_prices = [s[1] if s[1] and s[1] > 0 else ((s[2] or 0) + (s[3] or 0)) / 2 for s in _pre_snaps]
                _o_prices = [p for p in _o_prices if p > 0]
                if _u_prices:
                    _u_price = _u_prices[-1]
                    if len(_u_prices) >= 6 and _u_prices[-6] > 0:
                        _u_mom_5 = (_u_prices[-1] - _u_prices[-6]) / _u_prices[-6] * 100
                if _o_prices and len(_o_prices) >= 6 and _o_prices[-6] > 0:
                    _prem_mom_5 = (_o_prices[-1] - _o_prices[-6]) / _o_prices[-6] * 100

            if use_peak_predictor:
                _peak_sig = predict_peak(
                    ticker=ticker, entry_premium=entry_premium,
                    underlying_price=_u_price,
                    hour=signal_et.hour, minute=signal_et.minute,
                    day_of_week=signal_et.weekday(),
                    is_call=is_call,
                    underlying_momentum_5m=_u_mom_5,
                    premium_momentum_5m=_prem_mom_5,
                )
            if use_regime_classifier:
                _regime_sig = predict_regime(
                    ticker=ticker,
                    day_of_week=signal_et.weekday(),
                )

        exit_premium, peak_premium, exit_reason, duration_min, last_target = simulate_exit_pipeline(
            snapshots, entry_premium, signal_time, settings,
            ticker=ticker, is_call=is_call,
            enable_ml=enable_ml, ml_override_trails=ml_override_trails,
            ml_sell_threshold=ml_sell_threshold, ml_min_minutes=ml_min_minutes,
            peak_signal=_peak_sig, regime_signal=_regime_sig,
        )

        pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_pct = (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_gap = mfe_pct - pnl_pct

        contracts = score_to_contracts(
            score, cost_per_contract=entry_premium * 100,
            balance=running_balance, max_position_pct=20.0, max_concurrent=max_concurrent,
        )
        pnl_dollars = (exit_premium - entry_premium) * contracts * 100
        running_balance += pnl_dollars  # compound gains/losses

        est_close = signal_time + timedelta(minutes=max(duration_min, 1))
        open_trades.append((ticker, signal_time, est_close))
        open_tickers.add(ticker)

        results.append(ReplayResult(
            ticker=ticker, direction=direction, strike=strike,
            signal_time=created_at[:16], entry_premium=entry_premium,
            exit_premium=exit_premium, peak_premium=peak_premium,
            pnl_pct=pnl_pct, mfe_pct=mfe_pct, mfe_gap=mfe_gap,
            exit_reason=exit_reason, duration_min=duration_min,
            contracts=contracts, pnl_dollars=pnl_dollars,
        ))

    if not results:
        return name, 0, 0, 0, 0, 0, 0, {}, BALANCE

    wins = [r for r in results if r.pnl_pct >= 0]
    losses = [r for r in results if r.pnl_pct < 0]
    total_pnl = sum(r.pnl_dollars for r in results)
    wr = len(wins) / len(results) * 100
    avg_gap = sum(r.mfe_gap for r in results) / len(results)
    avg_dur = sum(r.duration_min for r in results) / len(results)

    reason_counts = {}
    for r in results:
        reason = r.exit_reason.split("(")[0]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return name, len(results), wr, total_pnl, avg_gap, avg_dur, sum(r.pnl_pct for r in results)/len(results), reason_counts, running_balance


def load_all_signals(*db_paths: str) -> list:
    """Load and merge signals from multiple DBs, deduplicated by (ticker, created_at)."""
    seen = set()
    all_signals = []
    query = """
        SELECT ticker, direction, strike, expiry, atm_premium, otm_premium,
               score, created_at, target_1_pct, target_2_pct
        FROM trade_signals
        ORDER BY created_at
    """
    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        try:
            for row in conn.execute(query).fetchall():
                key = (row[0], row[7])  # (ticker, created_at)
                if key not in seen:
                    seen.add(key)
                    all_signals.append(row)
        except sqlite3.OperationalError:
            pass  # table doesn't exist in this DB
        conn.close()
    all_signals.sort(key=lambda r: r[7])  # sort by created_at
    return all_signals


def main():
    settings = Settings()

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Load signals from both current and legacy DBs
    signals = load_all_signals(SIGNALS_DB, LEGACY_SIGNALS_DB)

    print(f"{'='*120}")
    print(f"  REPLAY: {len(signals)} Discord signals through exit pipeline")
    print(f"  Config: premium_stop={settings.PREMIUM_STOP_PCT}% | "
          f"profit_lock={settings.PROFIT_LOCK_TIERS}")
    print(f"  Balance: ${BALANCE:,.0f}")
    print(f"{'='*120}")
    print()

    results: list[ReplayResult] = []
    skipped = 0
    rejected = 0

    # --- Entry pipeline state (mimics live system) ---
    max_concurrent = settings.MAX_CONCURRENT
    open_tickers: set[str] = set()  # currently open positions (by ticker)
    open_trades: list[tuple[str, datetime, float]] = []  # (ticker, signal_time, est_close_time)
    running_balance = BALANCE  # compound wins/losses

    from options_owl.risk.vinny_strategy import score_to_contracts

    for sig in signals:
        ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at, t1_pct, t2_pct = sig

        # Resolve expiry for 0DTE
        if expiry == "0DTE":
            expiry_date = resolve_expiry(created_at)
        else:
            expiry_date = expiry

        # Use ATM premium as entry price (that's what we'd trade)
        entry_premium = atm_premium
        if entry_premium is None or entry_premium <= 0:
            # Try OTM
            entry_premium = otm_premium
        if entry_premium is None or entry_premium <= 0:
            skipped += 1
            continue

        signal_time = datetime.fromisoformat(created_at)

        # --- ENTRY PIPELINE GATES ---

        # 1. Score filter
        if score < settings.MIN_SCORE:
            print(f"  REJECT {ticker} @ {created_at[11:16]} — score {score} < {settings.MIN_SCORE}")
            rejected += 1
            continue

        # 2. Clean up closed trades (estimate: trades close within their duration)
        open_trades = [(t, st, ct) for t, st, ct in open_trades if ct > signal_time]
        open_tickers = {t for t, _, _ in open_trades}

        # 3. MAX_CONCURRENT check
        if len(open_trades) >= max_concurrent:
            print(f"  REJECT {ticker} @ {created_at[11:16]} — max concurrent ({max_concurrent}) reached [{', '.join(open_tickers)}]")
            rejected += 1
            continue

        # 4. Duplicate ticker check (no two positions in same underlying)
        if ticker in open_tickers:
            print(f"  REJECT {ticker} @ {created_at[11:16]} — duplicate ticker (already open)")
            rejected += 1
            continue

        # 5. Minimum premium check
        if entry_premium < settings.MIN_OPTION_PREMIUM:
            print(f"  REJECT {ticker} @ {created_at[11:16]} — premium ${entry_premium:.2f} < min ${settings.MIN_OPTION_PREMIUM:.2f}")
            rejected += 1
            continue

        # Build contract ticker
        option_type = "call" if direction in ("call", "bullish", "long") else "put"
        contract_ticker = build_contract_ticker(ticker, expiry_date, strike, option_type)

        # Get snapshots from harvester
        snapshots = get_snapshots(harvester_conn, contract_ticker, created_at)

        if not snapshots:
            # Try with slight time offset (harvester might be a minute behind)
            earlier = (signal_time - timedelta(minutes=2)).isoformat()
            snapshots = get_snapshots(harvester_conn, contract_ticker, earlier)

        if not snapshots:
            print(f"  SKIP {ticker} {option_type} ${strike} @ {created_at[:16]} — no harvester data for {contract_ticker}")
            skipped += 1
            continue

        # Use first snapshot's midpoint as actual entry if available
        first_snap = snapshots[0]
        actual_entry = first_snap[1]  # midpoint
        if actual_entry and actual_entry > 0:
            entry_premium = actual_entry

        # ML v2: get entry confidence and peak prediction
        signal_et = _to_et(signal_time)
        is_call = option_type == "call"

        # Get underlying context from harvester snapshots
        pre_snaps = get_snapshots(harvester_conn, contract_ticker, (signal_time - timedelta(minutes=30)).isoformat())
        u_price_at_entry = 0.0
        u_mom_5 = 0.0
        prem_mom_5 = 0.0
        if pre_snaps and len(pre_snaps) >= 5:
            u_prices = [s[4] for s in pre_snaps if s[4] and s[4] > 0]
            o_prices = [s[1] if s[1] and s[1] > 0 else ((s[2] or 0) + (s[3] or 0)) / 2 for s in pre_snaps]
            o_prices = [p for p in o_prices if p > 0]
            if u_prices:
                u_price_at_entry = u_prices[-1]
                if len(u_prices) >= 6 and u_prices[-6] > 0:
                    u_mom_5 = (u_prices[-1] - u_prices[-6]) / u_prices[-6] * 100
            if o_prices and len(o_prices) >= 6 and o_prices[-6] > 0:
                prem_mom_5 = (o_prices[-1] - o_prices[-6]) / o_prices[-6] * 100

        ml_entry = predict_entry(
            ticker=ticker, entry_premium=entry_premium,
            underlying_price=u_price_at_entry,
            hour=signal_et.hour, minute=signal_et.minute,
            day_of_week=signal_et.weekday(),
            is_call=is_call,
            underlying_momentum_5m=u_mom_5,
            premium_momentum_5m=prem_mom_5,
        )
        ml_peak = predict_peak(
            ticker=ticker, entry_premium=entry_premium,
            underlying_price=u_price_at_entry,
            hour=signal_et.hour, minute=signal_et.minute,
            day_of_week=signal_et.weekday(),
            is_call=is_call,
            underlying_momentum_5m=u_mom_5,
            premium_momentum_5m=prem_mom_5,
        )

        # Simulate through exit pipeline
        exit_premium, peak_premium, exit_reason, duration_min, last_target = simulate_exit_pipeline(
            snapshots, entry_premium, signal_time, settings,
            ticker=ticker, is_call=is_call,
        )

        pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_pct = (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_gap = mfe_pct - pnl_pct

        contracts = score_to_contracts(
            score, cost_per_contract=entry_premium * 100,
            balance=running_balance, max_position_pct=20.0, max_concurrent=max_concurrent,
        )
        pnl_dollars = (exit_premium - entry_premium) * contracts * 100
        running_balance += pnl_dollars  # compound gains/losses

        # Track this trade as open (estimate close time from duration)
        est_close = signal_time + timedelta(minutes=max(duration_min, 1))
        open_trades.append((ticker, signal_time, est_close))
        open_tickers.add(ticker)

        result = ReplayResult(
            ticker=ticker, direction=direction, strike=strike,
            signal_time=created_at[:16], entry_premium=entry_premium,
            exit_premium=exit_premium, peak_premium=peak_premium,
            pnl_pct=pnl_pct, mfe_pct=mfe_pct, mfe_gap=mfe_gap,
            exit_reason=exit_reason, duration_min=duration_min,
            contracts=contracts, pnl_dollars=pnl_dollars,
        )
        results.append(result)

        # Print each trade with ML v2 predictions
        win = "W" if pnl_pct >= 0 else "L"
        left = f" (left {mfe_gap:.0f}% on table)" if mfe_gap > 5 else ""
        ml_tag = f"ML:conf={ml_entry.confidence:.0%},predMFE={ml_peak.predicted_mfe_pct:+.0f}%"
        ml_correct = ""
        if ml_entry.confidence >= 0.5 and pnl_pct >= 0:
            ml_correct = " ✓ML"
        elif ml_entry.confidence < 0.5 and pnl_pct < 0:
            ml_correct = " ✓ML-reject"
        elif ml_entry.confidence >= 0.5 and pnl_pct < 0:
            ml_correct = " ✗ML-miss"
        elif ml_entry.confidence < 0.5 and pnl_pct >= 0:
            ml_correct = " ✗ML-skip"
        print(f"  [{win}] {ticker:5} {option_type:4} ${strike:<8} @ {created_at[11:16]} | "
              f"entry=${entry_premium:.2f} exit=${exit_premium:.2f} peak=${peak_premium:.2f} | "
              f"PnL={pnl_pct:+6.1f}% MFE={mfe_pct:+5.0f}% | "
              f"{contracts}ct ${pnl_dollars:+8.2f} bal=${running_balance:,.0f} | {duration_min:5.0f}m | {exit_reason}{left} | {ml_tag}{ml_correct}")

    # Summary
    print()
    print(f"{'='*120}")
    print(f"  SUMMARY")
    print(f"{'='*120}")

    if not results:
        print("  No trades to summarize.")
        return

    wins = [r for r in results if r.pnl_pct >= 0]
    losses = [r for r in results if r.pnl_pct < 0]
    total_pnl = sum(r.pnl_dollars for r in results)
    avg_mfe_gap = sum(r.mfe_gap for r in results) / len(results)

    print(f"  Trades: {len(results)} ({len(wins)}W / {len(losses)}L) | Rejected: {rejected} | Skipped (no data): {skipped}")
    print(f"  Win Rate: {len(wins)/len(results)*100:.0f}%")
    print(f"  Starting Balance: ${BALANCE:,.0f}")
    print(f"  Final Balance: ${running_balance:,.0f} ({(running_balance/BALANCE - 1)*100:+.1f}%)")
    print(f"  Total P&L: ${total_pnl:+,.2f} (compounded)")
    print(f"  Avg P&L%: {sum(r.pnl_pct for r in results)/len(results):+.1f}%")
    if wins:
        print(f"  Avg Win: {sum(r.pnl_pct for r in wins)/len(wins):+.1f}% (${sum(r.pnl_dollars for r in wins)/len(wins):+.2f})")
    if losses:
        print(f"  Avg Loss: {sum(r.pnl_pct for r in losses)/len(losses):+.1f}% (${sum(r.pnl_dollars for r in losses)/len(losses):+.2f})")
    print(f"  Avg MFE Gap: {avg_mfe_gap:.1f}% (money left on table)")
    print(f"  Avg Duration: {sum(r.duration_min for r in results)/len(results):.0f} min")
    print()

    # Exit reason breakdown
    reason_counts: dict[str, list] = {}
    for r in results:
        reason = r.exit_reason.split("(")[0]  # normalize
        if reason not in reason_counts:
            reason_counts[reason] = []
        reason_counts[reason].append(r)

    print(f"  EXIT REASON BREAKDOWN:")
    print(f"  {'Reason':<30} {'Count':>5} {'WR':>5} {'Avg P&L':>8} {'Avg MFE Gap':>10} {'Avg $':>10}")
    for reason, trades in sorted(reason_counts.items(), key=lambda x: -len(x[1])):
        wr = len([t for t in trades if t.pnl_pct >= 0]) / len(trades) * 100
        avg_pnl = sum(t.pnl_pct for t in trades) / len(trades)
        avg_gap = sum(t.mfe_gap for t in trades) / len(trades)
        avg_dollars = sum(t.pnl_dollars for t in trades) / len(trades)
        print(f"  {reason:<30} {len(trades):>5} {wr:>4.0f}% {avg_pnl:>+7.1f}% {avg_gap:>9.1f}% ${avg_dollars:>+9.2f}")

    # Daily P&L
    print()
    print(f"  DAILY P&L:")
    daily: dict[str, list] = {}
    for r in results:
        day = r.signal_time[:10]
        if day not in daily:
            daily[day] = []
        daily[day].append(r)

    for day, trades in sorted(daily.items()):
        day_pnl = sum(t.pnl_dollars for t in trades)
        day_wins = len([t for t in trades if t.pnl_pct >= 0])
        print(f"  {day}: {len(trades)} trades ({day_wins}W/{len(trades)-day_wins}L) → ${day_pnl:+,.2f}")

    print()
    print(f"{'='*120}")

    # ============================================================
    # SCENARIO COMPARISON
    # ============================================================
    print()
    print(f"{'='*120}")
    print(f"  SCENARIO COMPARISON — testing config variations on real signals")
    print(f"{'='*120}")

    # Current deployed config (baseline)
    deployed_base = {
        "PREMIUM_STOP_PCT": 60.0,
        "NO_MOMENTUM_MINUTES": 60,
        "TIME_DECAY_HOLD_MINUTES": 90.0,
        "THETA_BLEED_HOLD_MINUTES": 90.0,
    }

    scenarios = {
        "Current deployed": deployed_base,

        # --- DOLLAR TRAIL: the #1 exit (86% WR) but triggers in ~5-15 min ---
        # Problem: 10% activation on a $0.30 option = $3 profit triggers trail.
        # One tiny pullback = exit. Need to let trades breathe to hit 1-2.5hr peak.
        "DT activate 20%": {**deployed_base, "DOLLAR_TRAIL_ACTIVATION_PCT": 20.0},
        "DT activate 30%": {**deployed_base, "DOLLAR_TRAIL_ACTIVATION_PCT": 30.0},
        "DT activate 40%": {**deployed_base, "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0},
        # Wider steps = fewer exits on small pullbacks
        "DT steps 15/8": {**deployed_base,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 15.0, "DOLLAR_TRAIL_LARGE_STEP_PCT": 8.0},
        "DT steps 20/10": {**deployed_base,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0, "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0},
        # Higher threshold before switching to tight steps
        "DT threshold 50%": {**deployed_base, "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 50.0},
        # Combo: higher activation + wider steps (let trades run to 1-2.5hr sweet spot)
        "DT 30%/20/10/50": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 30.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 50.0},
        "DT 40%/20/10/50": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 50.0},

        # --- ADAPTIVE TRAIL: widen to let runners run longer ---
        "AT wider active 45%": {**deployed_base, "ADAPTIVE_TRAIL_ACTIVE_WIDTH": 45.0},
        "AT wider runner 55%": {**deployed_base, "ADAPTIVE_TRAIL_RUNNER_WIDTH": 55.0},
        "AT later runner 200%": {**deployed_base, "ADAPTIVE_TRAIL_RUNNER_THRESHOLD": 200.0},

        # --- DISABLE DOLLAR TRAIL: let adaptive trail be primary exit ---
        # If dollar trail is cutting winners at 15 min, what happens without it?
        "No dollar trail": {**deployed_base, "ENABLE_DOLLAR_TRAIL": False},
        "No DT + wider AT": {**deployed_base,
            "ENABLE_DOLLAR_TRAIL": False,
            "ADAPTIVE_TRAIL_ACTIVE_WIDTH": 45.0,
            "ADAPTIVE_TRAIL_RUNNER_WIDTH": 55.0},

        # --- MINIMUM HOLD TIME: force trades to stay open longer ---
        # The real insight: optimal is 1-2.5 hours. What if we enforce that?
        "Grace 15min": {**deployed_base, "STOP_GRACE_PERIOD_MINUTES": 15},
        "Grace 20min": {**deployed_base, "STOP_GRACE_PERIOD_MINUTES": 20},
        "Grace 30min": {**deployed_base, "STOP_GRACE_PERIOD_MINUTES": 30},

        # --- COMBINED: wider trail + longer hold ---
        "DT30+grace15": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 30.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
            "STOP_GRACE_PERIOD_MINUTES": 15},
        "DT40+grace20": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
            "STOP_GRACE_PERIOD_MINUTES": 20},
        "NoDT+grace20": {**deployed_base,
            "ENABLE_DOLLAR_TRAIL": False,
            "STOP_GRACE_PERIOD_MINUTES": 20},
        "NoDT+grace30": {**deployed_base,
            "ENABLE_DOLLAR_TRAIL": False,
            "STOP_GRACE_PERIOD_MINUTES": 30},

        # --- TRAILING STOP activation ---
        "Trail activate 50%": {**deployed_base, "TRAILING_STOP_ACTIVATION_PCT": 50.0},
        "Trail drop 40%": {**deployed_base, "TRAILING_STOP_DROP_PCT": 40.0},

        # --- PREMIUM STOP variations ---
        "Stop 65%": {**deployed_base, "PREMIUM_STOP_PCT": 65.0},
        "Stop 70%": {**deployed_base, "PREMIUM_STOP_PCT": 70.0},

        # --- FULL PATIENT CONFIG: everything dialed to let trades run ---
        "Patient": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 30.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 50.0,
            "STOP_GRACE_PERIOD_MINUTES": 15,
            "PREMIUM_STOP_PCT": 65.0,
            "ADAPTIVE_TRAIL_ACTIVE_WIDTH": 45.0,
            "TRAILING_STOP_ACTIVATION_PCT": 50.0,
        },
        "Ultra patient": {**deployed_base,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 25.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 12.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 75.0,
            "STOP_GRACE_PERIOD_MINUTES": 20,
            "PREMIUM_STOP_PCT": 70.0,
            "ADAPTIVE_TRAIL_ACTIVE_WIDTH": 50.0,
            "ADAPTIVE_TRAIL_RUNNER_WIDTH": 55.0,
            "TRAILING_STOP_ACTIVATION_PCT": 50.0,
        },
    }

    # ML-enabled scenarios — (overrides, enable_ml, ml_override_trails, ml_sell_threshold, ml_min_minutes)
    ml_scenarios = {
        # Baseline: default threshold 0.4
        "ML t=0.4 (default)":          ({}, True, False, 0.4, 0),
        "ML t=0.5":                    ({}, True, False, 0.5, 0),
        "ML t=0.6":                    ({}, True, False, 0.6, 0),
        "ML t=0.7":                    ({}, True, False, 0.7, 0),
        # Don't let ML sell in first 10 min (let trade breathe)
        "ML t=0.4 min10m":            ({}, True, False, 0.4, 10),
        "ML t=0.5 min10m":            ({}, True, False, 0.5, 10),
        "ML t=0.6 min10m":            ({}, True, False, 0.6, 10),
        # ML with trail override (let ML keep winners running)
        "ML t=0.5 + override":        ({}, True, True, 0.5, 0),
        "ML t=0.6 + override":        ({}, True, True, 0.6, 0),
        "ML t=0.5 min10m + override": ({}, True, True, 0.5, 10),
        # ML + no trails (ML is sole exit besides stops)
        "ML t=0.5 no trails":         ({"ENABLE_DOLLAR_TRAIL": False, "ENABLE_ADAPTIVE_TRAIL": False}, True, False, 0.5, 0),
        "ML t=0.6 no trails":         ({"ENABLE_DOLLAR_TRAIL": False, "ENABLE_ADAPTIVE_TRAIL": False}, True, False, 0.6, 0),
        # ML + peak-aware stop + looser config
        "ML t=0.5 + stop60 + peak40": ({"PREMIUM_STOP_PCT": 60.0, "PEAK_STOP_DROP_PCT": 40.0}, True, True, 0.5, 0),
        "ML t=0.6 + stop60 + peak40": ({"PREMIUM_STOP_PCT": 60.0, "PEAK_STOP_DROP_PCT": 40.0}, True, True, 0.6, 0),
        # Best non-ML config + ML on top
        "ML t=0.5 + ultra loose":     ({
            "TIME_DECAY_STALE_MINUTES": 15.0,
            "TIME_DECAY_AFTERNOON_HOUR": 15,
            "TIME_DECAY_AFTERNOON_MINUTE": 45,
            "PREMIUM_STOP_PCT": 60.0,
            "PROFIT_LOCK_TIERS": "100:30,200:100",
            "STOP_GRACE_PERIOD_MINUTES": 8,
            "NO_MOMENTUM_MINUTES": 60,
            "TIME_DECAY_HOLD_MINUTES": 90.0,
            "THETA_BLEED_HOLD_MINUTES": 90.0,
        }, True, True, 0.5, 0),
        "ML t=0.6 + ultra loose":     ({
            "TIME_DECAY_STALE_MINUTES": 15.0,
            "TIME_DECAY_AFTERNOON_HOUR": 15,
            "TIME_DECAY_AFTERNOON_MINUTE": 45,
            "PREMIUM_STOP_PCT": 60.0,
            "PROFIT_LOCK_TIERS": "100:30,200:100",
            "STOP_GRACE_PERIOD_MINUTES": 8,
            "NO_MOMENTUM_MINUTES": 60,
            "TIME_DECAY_HOLD_MINUTES": 90.0,
            "THETA_BLEED_HOLD_MINUTES": 90.0,
        }, True, True, 0.6, 0),
    }

    print(f"\n  {'Scenario':<35} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg P&L':>8} {'MFE Gap':>8} {'Avg Dur':>7} {'Top Exit Reasons'}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*40}")

    for name, overrides in scenarios.items():
        result = run_scenario(name, overrides, signals, harvester_conn)
        sname, trades, wr, total_pnl, avg_gap, avg_dur, avg_pnl, reasons, final_bal = result
        top_reasons = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:3])
        print(f"  {sname:<35} {trades:>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_pnl:>+7.1f}% {avg_gap:>7.1f}% {avg_dur:>6.0f}m {top_reasons}")

    print(f"\n  --- ML-ENABLED SCENARIOS ---")
    print(f"  {'Scenario':<35} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg P&L':>8} {'MFE Gap':>8} {'Avg Dur':>7} {'Top Exit Reasons'}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*40}")

    for name, (overrides, enable_ml, ml_override, ml_thresh, ml_min) in ml_scenarios.items():
        result = run_scenario(name, overrides, signals, harvester_conn,
                              enable_ml=enable_ml, ml_override_trails=ml_override,
                              ml_sell_threshold=ml_thresh, ml_min_minutes=ml_min)
        sname, trades, wr, total_pnl, avg_gap, avg_dur, avg_pnl, reasons, final_bal = result
        top_reasons = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:3])
        print(f"  {sname:<35} {trades:>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_pnl:>+7.1f}% {avg_gap:>7.1f}% {avg_dur:>6.0f}m {top_reasons}")

    # ============================================================
    # ML V2 ENTRY FILTER SCENARIOS
    # Test entry filter at various thresholds combined with best exit config
    # ============================================================
    print(f"\n  --- ML V2 ENTRY FILTER (DT40+grace20 exit) ---")
    print(f"  {'Scenario':<35} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg P&L':>8} {'MFE Gap':>8} {'Avg Dur':>7} {'Rejected'}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")

    best_exit = {
        **deployed_base,
        "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0,
        "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
        "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
        "STOP_GRACE_PERIOD_MINUTES": 20,
    }

    for thresh_name, thresh in [
        ("No filter (baseline)", 0.0),
        ("All-day model t=0.30", 0.30),
        ("All-day model t=0.40", 0.40),
        ("All-day model t=0.50", 0.50),
        ("All-day model t=0.60", 0.60),
    ]:
        result = run_scenario(thresh_name, best_exit, signals, harvester_conn,
                              entry_filter_threshold=thresh)
        sname, trades, wr, total_pnl, avg_gap, avg_dur, avg_pnl, reasons, final_bal = result
        print(f"  {sname:<35} {trades:>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_pnl:>+7.1f}% {avg_gap:>7.1f}% {avg_dur:>6.0f}m")

    print(f"\n  --- ML V2 AFTERNOON-CALIBRATED ENTRY FILTER ---")
    print(f"  {'Scenario':<35} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg P&L':>8} {'MFE Gap':>8} {'Avg Dur':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")

    for thresh_name, thresh in [
        ("Afternoon t=0.30", 0.30),
        ("Afternoon t=0.35", 0.35),
        ("Afternoon t=0.40", 0.40),
        ("Afternoon t=0.45", 0.45),
        ("Afternoon t=0.50", 0.50),
        ("Afternoon t=0.55", 0.55),
        ("Afternoon t=0.60", 0.60),
    ]:
        result = run_scenario(thresh_name, best_exit, signals, harvester_conn,
                              entry_filter_threshold=thresh,
                              entry_filter_afternoon=True)
        sname, trades, wr, total_pnl, avg_gap, avg_dur, avg_pnl, reasons, final_bal = result
        print(f"  {sname:<35} {trades:>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_pnl:>+7.1f}% {avg_gap:>7.1f}% {avg_dur:>6.0f}m")

    # ============================================================
    # PREMIUM FILTER + AFTERNOON ENTRY FILTER COMBINATIONS
    # Test MIN_OPTION_PREMIUM × afternoon entry filter thresholds
    # ============================================================
    print(f"\n  --- PREMIUM FILTER + AFTERNOON ENTRY FILTER COMBINATIONS ---")
    print(f"  {'Scenario':<50} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg P&L':>8} {'MFE Gap':>8} {'Avg Dur':>7}")
    print(f"  {'-'*50} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")

    for label, min_prem, thresh, afternoon in [
        # Baselines
        ("Baseline (min$0.25, no filter)", 0.25, 0.0, False),
        ("Min$0.50 only", 0.50, 0.0, False),
        ("Min$0.60 only", 0.60, 0.0, False),
        ("Min$0.75 only", 0.75, 0.0, False),
        ("Afternoon t=0.45 only (min$0.25)", 0.25, 0.45, True),
        # Combinations
        ("Min$0.50 + Afternoon t=0.40", 0.50, 0.40, True),
        ("Min$0.50 + Afternoon t=0.45", 0.50, 0.45, True),
        ("Min$0.50 + Afternoon t=0.50", 0.50, 0.50, True),
        ("Min$0.60 + Afternoon t=0.40", 0.60, 0.40, True),
        ("Min$0.60 + Afternoon t=0.45", 0.60, 0.45, True),
        ("Min$0.75 + Afternoon t=0.40", 0.75, 0.40, True),
        ("Min$0.75 + Afternoon t=0.45", 0.75, 0.45, True),
    ]:
        overrides = {
            **best_exit,
            "MIN_OPTION_PREMIUM": min_prem,
        }
        result = run_scenario(label, overrides, signals, harvester_conn,
                              entry_filter_threshold=thresh,
                              entry_filter_afternoon=afternoon)
        sname, trades, wr, total_pnl, avg_gap, avg_dur, avg_pnl, reasons, final_bal = result
        print(f"  {sname:<50} {trades:>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_pnl:>+7.1f}% {avg_gap:>7.1f}% {avg_dur:>6.0f}m")

    print()
    print(f"{'='*120}")

    # ============================================================
    # CONTRACT SIZING COMPARISON
    # Uses the winning exit config (DT40+grace20) with different sizing strategies
    # ============================================================
    print()
    print(f"{'='*120}")
    print(f"  CONTRACT SIZING COMPARISON — DT40+grace20 exit config × sizing strategies")
    print(f"{'='*120}")

    best_exit = {
        **deployed_base,
        "DOLLAR_TRAIL_ACTIVATION_PCT": 40.0,
        "DOLLAR_TRAIL_SMALL_STEP_PCT": 20.0,
        "DOLLAR_TRAIL_LARGE_STEP_PCT": 10.0,
        "STOP_GRACE_PERIOD_MINUTES": 20,
    }

    def run_sizing_scenario(label, balance, max_concurrent, max_pct, liq_cap, risk_pct, signals, harvester_conn):
        """Run a scenario with specific sizing params, compounding, and concurrency tracking."""
        settings = Settings()
        for k, v in best_exit.items():
            setattr(settings, k, v)

        from options_owl.risk.vinny_strategy import score_to_contracts

        results = []
        total_contracts = 0
        running_bal = balance
        open_trades = []
        open_tickers = set()

        for sig in signals:
            ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at, t1_pct, t2_pct = sig
            if expiry == "0DTE":
                expiry_date = resolve_expiry(created_at)
            else:
                expiry_date = expiry
            entry_premium = atm_premium
            if entry_premium is None or entry_premium <= 0:
                entry_premium = otm_premium
            if entry_premium is None or entry_premium <= 0:
                continue
            if score < settings.MIN_SCORE:
                continue
            if entry_premium < settings.MIN_OPTION_PREMIUM:
                continue

            signal_time = datetime.fromisoformat(created_at)
            open_trades = [(t, st, ct) for t, st, ct in open_trades if ct > signal_time]
            open_tickers = {t for t, _, _ in open_trades}
            if len(open_trades) >= max_concurrent:
                continue
            if ticker in open_tickers:
                continue

            option_type = "call" if direction in ("call", "bullish", "long") else "put"
            contract_ticker = build_contract_ticker(ticker, expiry_date, strike, option_type)
            snapshots = get_snapshots(harvester_conn, contract_ticker, created_at)
            if not snapshots:
                earlier = (signal_time - timedelta(minutes=2)).isoformat()
                snapshots = get_snapshots(harvester_conn, contract_ticker, earlier)
            if not snapshots:
                continue
            first_snap = snapshots[0]
            actual_entry = first_snap[1]
            if actual_entry and actual_entry > 0:
                entry_premium = actual_entry

            is_call = option_type == "call"
            exit_premium, peak_premium, exit_reason, duration_min, last_target = simulate_exit_pipeline(
                snapshots, entry_premium, signal_time, settings,
                ticker=ticker, is_call=is_call,
            )
            pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

            contracts = score_to_contracts(
                score, cost_per_contract=entry_premium * 100,
                balance=running_bal, max_position_pct=max_pct,
                max_concurrent=max_concurrent, max_portfolio_risk_pct=risk_pct,
            )
            contracts = min(contracts, liq_cap)
            contracts = max(contracts, 1)
            pnl_dollars = (exit_premium - entry_premium) * contracts * 100
            running_bal += pnl_dollars
            total_contracts += contracts

            est_close = signal_time + timedelta(minutes=max(duration_min, 1))
            open_trades.append((ticker, signal_time, est_close))
            open_tickers.add(ticker)

            results.append((pnl_pct, pnl_dollars, contracts, entry_premium))

        return results, running_bal, total_contracts

    # Sizing strategies to test
    sizing_configs = [
        # (label, balance, max_concurrent, max_position_pct, liquidity_cap, risk_pct)
        # --- CURRENT PRODUCTION ---
        ("CURRENT: $5K/3slot/20%/20liq/80%", 5000, 3, 20.0, 20, 80.0),
        # --- CONTRACT CAP TESTS (does limiting cheap option exposure help?) ---
        ("CAP 8ct:  $5K/3slot/20%/8liq/80%", 5000, 3, 20.0, 8, 80.0),
        ("CAP 10ct: $5K/3slot/20%/10liq/80%", 5000, 3, 20.0, 10, 80.0),
        ("CAP 12ct: $5K/3slot/20%/12liq/80%", 5000, 3, 20.0, 12, 80.0),
        ("CAP 15ct: $5K/3slot/20%/15liq/80%", 5000, 3, 20.0, 15, 80.0),
        # --- POSITION SIZE TESTS ---
        ("POS 25%:  $5K/3slot/25%/20liq/80%", 5000, 3, 25.0, 20, 80.0),
        ("POS 30%:  $5K/3slot/30%/20liq/80%", 5000, 3, 30.0, 20, 80.0),
        ("POS 15%:  $5K/3slot/15%/20liq/80%", 5000, 3, 15.0, 20, 80.0),
        # --- SLOT COUNT TESTS ---
        ("SLOT 2:   $5K/2slot/30%/20liq/80%", 5000, 2, 30.0, 20, 80.0),
        ("SLOT 4:   $5K/4slot/20%/20liq/80%", 5000, 4, 20.0, 20, 80.0),
        # --- BEST COMBOS ---
        ("COMBO: 10ct + 25%pos", 5000, 3, 25.0, 10, 80.0),
        ("COMBO: 12ct + 25%pos", 5000, 3, 25.0, 12, 80.0),
        ("COMBO: 10ct + 2slot", 5000, 2, 30.0, 10, 80.0),
        # --- SMALLER PORTFOLIOS ---
        ("$2.5K / 3slot / 25%pos / 20liq", 2500, 3, 25.0, 20, 80.0),
        ("$2.5K / 3slot / 25%pos / 10liq", 2500, 3, 25.0, 10, 80.0),
        ("$500 / 3slot / 30%pos / 10liq", 500, 3, 30.0, 10, 90.0),
        ("$500 / 2slot / 40%pos / 10liq", 500, 2, 40.0, 10, 90.0),
    ]

    print(f"\n  {'Scenario':<45} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'Final Bal':>10} {'Avg Ct':>7} {'Avg $Win':>10} {'Avg $Loss':>10}")
    print(f"  {'-'*45} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")

    for label, bal, mc, mp, lc, rp in sizing_configs:
        results, final_bal, total_ct = run_sizing_scenario(
            label, bal, mc, mp, lc, rp, signals, harvester_conn)
        if results:
            wins = [r for r in results if r[0] >= 0]
            losses = [r for r in results if r[0] < 0]
            wr = len(wins) / len(results) * 100
            total_pnl = sum(r[1] for r in results)
            avg_ct = total_ct / len(results)
            avg_win_d = sum(r[1] for r in wins) / len(wins) if wins else 0
            avg_loss_d = sum(r[1] for r in losses) / len(losses) if losses else 0
            print(f"  {label:<45} {len(results):>6} {wr:>4.0f}% ${total_pnl:>+10,.2f} ${final_bal:>9,.0f} {avg_ct:>6.1f} ${avg_win_d:>+9.2f} ${avg_loss_d:>+9.2f}")

    # Premium equalization analysis
    print(f"\n  --- PREMIUM EQUALIZATION ANALYSIS (DT40+grace20, $5K, 3 slots) ---")
    print(f"  How much do cheap vs expensive options contribute?")
    results_eq, _, _ = run_sizing_scenario(
        "eq", 5000, 3, 20.0, 20, 80.0, signals, harvester_conn)
    if results_eq:
        cheap = [r for r in results_eq if r[3] < 0.50]  # entry_premium < $0.50
        mid = [r for r in results_eq if 0.50 <= r[3] < 1.50]
        expensive = [r for r in results_eq if r[3] >= 1.50]
        for label, group in [("Cheap (<$0.50)", cheap), ("Mid ($0.50-1.50)", mid), ("Expensive (>$1.50)", expensive)]:
            if group:
                avg_ct = sum(r[2] for r in group) / len(group)
                avg_pnl = sum(r[0] for r in group) / len(group)
                total_d = sum(r[1] for r in group)
                avg_d = total_d / len(group)
                wr = len([r for r in group if r[0] >= 0]) / len(group) * 100
                print(f"  {label:<25} {len(group):>3} trades | WR={wr:>4.0f}% | Avg Ct={avg_ct:>5.1f} | Avg P&L={avg_pnl:>+6.1f}% | Avg $={avg_d:>+8.2f} | Total $={total_d:>+10.2f}")

        # Biggest winners and losers analysis
        print(f"\n  --- BIGGEST WINNERS vs BIGGEST LOSERS (shows if one big loss wipes gains) ---")
        sorted_by_dollars = sorted(results_eq, key=lambda r: r[1])
        print(f"  Top 5 losers (dollar):")
        for pnl_pct, pnl_d, cts, entry_p in sorted_by_dollars[:5]:
            print(f"    ${pnl_d:>+8.2f} ({pnl_pct:>+5.1f}%) | {cts:>2}ct x ${entry_p:.2f} = ${entry_p*cts*100:.0f} deployed")
        print(f"  Top 5 winners (dollar):")
        for pnl_pct, pnl_d, cts, entry_p in sorted_by_dollars[-5:]:
            print(f"    ${pnl_d:>+8.2f} ({pnl_pct:>+5.1f}%) | {cts:>2}ct x ${entry_p:.2f} = ${entry_p*cts*100:.0f} deployed")

        # Win/loss asymmetry
        all_wins_d = [r[1] for r in results_eq if r[1] >= 0]
        all_losses_d = [r[1] for r in results_eq if r[1] < 0]
        if all_wins_d and all_losses_d:
            print(f"\n  Asymmetry check:")
            print(f"    Avg win:  ${sum(all_wins_d)/len(all_wins_d):+.2f} ({len(all_wins_d)} trades)")
            print(f"    Avg loss: ${sum(all_losses_d)/len(all_losses_d):+.2f} ({len(all_losses_d)} trades)")
            print(f"    Max win:  ${max(all_wins_d):+.2f}")
            print(f"    Max loss: ${min(all_losses_d):+.2f}")
            print(f"    Win/loss ratio: {abs(sum(all_wins_d)/len(all_wins_d) / (sum(all_losses_d)/len(all_losses_d))):.2f}x")

    # Granular premium tier breakdown
    if results_eq:
        print(f"\n  --- GRANULAR PREMIUM BREAKDOWN (which price range is actually bad?) ---")
        tiers = [
            ("$0.00-$0.25", 0.00, 0.25),
            ("$0.25-$0.50", 0.25, 0.50),
            ("$0.50-$0.75", 0.50, 0.75),
            ("$0.75-$1.00", 0.75, 1.00),
            ("$1.00-$1.50", 1.00, 1.50),
            ("$1.50-$2.50", 1.50, 2.50),
            ("$2.50+     ", 2.50, 999.0),
        ]
        for label, lo, hi in tiers:
            group = [r for r in results_eq if lo <= r[3] < hi]
            if group:
                ws = len([r for r in group if r[0] >= 0])
                wr = ws / len(group) * 100
                total_d = sum(r[1] for r in group)
                avg_ct = sum(r[2] for r in group) / len(group)
                avg_d = total_d / len(group)
                print(f"  {label}  {len(group):>3} trades | WR={wr:>4.0f}% | AvgCt={avg_ct:>5.1f} | Avg$={avg_d:>+9.2f} | Total$={total_d:>+10.2f}")

    # Contract cap impact: what happens when we limit to 10ct?
    print(f"\n  --- CONTRACT CAP COMPARISON: 20 vs 10 (shows how capping changes risk) ---")
    for cap, cap_label in [(20, "20ct (current)"), (10, "10ct (proposed)")]:
        r, fb, tc = run_sizing_scenario(
            cap_label, 5000, 3, 20.0, cap, 80.0, signals, harvester_conn)
        if r:
            ws = [x for x in r if x[0] >= 0]
            ls = [x for x in r if x[0] < 0]
            total = sum(x[1] for x in r)
            max_loss = min(x[1] for x in r) if ls else 0
            max_win = max(x[1] for x in r) if ws else 0
            wr = len(ws)/len(r)*100
            avg_ct = tc/len(r)
            print(f"  {cap_label:<20} {len(r):>3} trades | WR={wr:.0f}% | P&L=${total:>+8.0f} | MaxWin=${max_win:>+6.0f} | MaxLoss=${max_loss:>+6.0f} | AvgCt={avg_ct:.1f}")

    # Test minimum premium filters
    print(f"\n  --- MINIMUM PREMIUM FILTER TEST (DT40+grace20, $5K, 3 slots, 25%pos) ---")
    for min_prem in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75]:
        # Temporarily override MIN_OPTION_PREMIUM in the exit config
        best_exit_with_minprem = {**best_exit, "MIN_OPTION_PREMIUM": min_prem}
        settings_tmp = Settings()
        for k, v in best_exit_with_minprem.items():
            setattr(settings_tmp, k, v)

        from options_owl.risk.vinny_strategy import score_to_contracts
        tmp_results = []
        tmp_bal = 5000.0
        tmp_open_trades = []
        tmp_open_tickers = set()
        tmp_total_ct = 0

        for sig in signals:
            ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at, t1_pct, t2_pct = sig
            if expiry == "0DTE":
                expiry_date = resolve_expiry(created_at)
            else:
                expiry_date = expiry
            entry_premium = atm_premium
            if entry_premium is None or entry_premium <= 0:
                entry_premium = otm_premium
            if entry_premium is None or entry_premium <= 0:
                continue
            if score < settings_tmp.MIN_SCORE:
                continue
            if entry_premium < min_prem:
                continue

            signal_time = datetime.fromisoformat(created_at)
            tmp_open_trades = [(t, st, ct) for t, st, ct in tmp_open_trades if ct > signal_time]
            tmp_open_tickers = {t for t, _, _ in tmp_open_trades}
            if len(tmp_open_trades) >= 3:
                continue
            if ticker in tmp_open_tickers:
                continue

            option_type = "call" if direction in ("call", "bullish", "long") else "put"
            contract_ticker = build_contract_ticker(ticker, expiry_date, strike, option_type)
            snapshots = get_snapshots(harvester_conn, contract_ticker, created_at)
            if not snapshots:
                earlier = (signal_time - timedelta(minutes=2)).isoformat()
                snapshots = get_snapshots(harvester_conn, contract_ticker, earlier)
            if not snapshots:
                continue
            first_snap = snapshots[0]
            actual_entry = first_snap[1]
            if actual_entry and actual_entry > 0:
                entry_premium = actual_entry
            if entry_premium < min_prem:
                continue

            is_call = option_type == "call"
            exit_premium, peak_premium, exit_reason, duration_min, last_target = simulate_exit_pipeline(
                snapshots, entry_premium, signal_time, settings_tmp,
                ticker=ticker, is_call=is_call,
            )
            pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
            contracts = score_to_contracts(
                score, cost_per_contract=entry_premium * 100,
                balance=tmp_bal, max_position_pct=25.0, max_concurrent=3,
            )
            contracts = min(contracts, 20)
            contracts = max(contracts, 1)
            pnl_dollars = (exit_premium - entry_premium) * contracts * 100
            tmp_bal += pnl_dollars
            tmp_total_ct += contracts

            est_close = signal_time + timedelta(minutes=max(duration_min, 1))
            tmp_open_trades.append((ticker, signal_time, est_close))
            tmp_results.append((pnl_pct, pnl_dollars, contracts, entry_premium))

        if tmp_results:
            wins = [r for r in tmp_results if r[0] >= 0]
            losses = [r for r in tmp_results if r[0] < 0]
            wr = len(wins) / len(tmp_results) * 100
            total_pnl = sum(r[1] for r in tmp_results)
            avg_ct = tmp_total_ct / len(tmp_results)
            print(f"  Min ${min_prem:.2f}: {len(tmp_results):>3} trades | WR={wr:>4.0f}% | Final=${tmp_bal:>10,.0f} | P&L=${total_pnl:>+10,.2f} | Avg Ct={avg_ct:>5.1f}")

    print()
    print(f"{'='*120}")

    harvester_conn.close()


if __name__ == "__main__":
    main()
