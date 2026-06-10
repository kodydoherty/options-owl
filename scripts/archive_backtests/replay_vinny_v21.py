"""Replay actual Discord signals through Vinny's v2.1 sell logic spec.

Tests each v2.1 recommendation individually and combined against real signals
to measure improvement vs our current config ("kitchen sink").

Usage:
    python scripts/replay_vinny_v21.py
"""

import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNALS_DB = os.environ.get("SIGNALS_DB", "journal/owlet-kody/raw_messages.db")
HARVESTER_DB = os.environ.get("HARVESTER_DB", "journal/owlet-harvester/options_data.db")

ET = timezone(timedelta(hours=-4))


# ---------------------------------------------------------------------------
# Vinny v2.1 static tables
# ---------------------------------------------------------------------------

# §6.1 Chrono-zones (ET times → multipliers for trail and stop)
CHRONO_ZONES = {
    "OPEN":          {"start": (9, 30),  "end": (10, 0),  "trail_mult": 1.20, "stop_mult": 1.15},
    "MORNING_POWER": {"start": (10, 0),  "end": (12, 0),  "trail_mult": 1.10, "stop_mult": 1.05},
    "MIDDAY":        {"start": (12, 0),  "end": (14, 0),  "trail_mult": 0.70, "stop_mult": 0.85},
    "AFTERNOON":     {"start": (14, 0),  "end": (15, 30), "trail_mult": 0.75, "stop_mult": 0.90},
    "LAST_30":       {"start": (15, 30), "end": (15, 55), "trail_mult": 0.50, "stop_mult": 0.70},
}

# §6.10 Ticker tiers
TICKER_TIERS = {
    "PREMIUM":  {"tickers": {"AMZN", "META", "AMD", "TSLA", "NVDA", "AAPL"}, "trail_mult": 1.15, "stop_mult": 1.05},
    "STANDARD": {"tickers": {"SPY", "QQQ", "IWM", "MSFT"}, "trail_mult": 1.00, "stop_mult": 1.00},
    "PENALTY":  {"tickers": {"GOOGL", "MU", "MSTR"}, "trail_mult": 0.80, "stop_mult": 0.90},
}

# §6.11 Direction multiplier
DIRECTION_MULT = {
    "bullish": {"trail_mult": 1.00, "stop_mult": 1.00},
    "bearish": {"trail_mult": 0.70, "stop_mult": 0.85},
}

# §7 Per-account parameters
ACCOUNT_PARAMS = {
    "5000": {
        "stop_loss_base": 0.55,  # -55%
        "grace_period_sec": 600,  # 10 min
        "trail_activation": 0.40,  # +40%
        "trail_active_width": 0.35,
        "trail_runner_width": 0.45,
        "trail_moonshot_width": 0.30,
        "profit_lock_tiers": [(300, 200), (180, 100), (100, 50), (50, 15)],  # (peak%, floor%)
        "setup_failed_sec": 1080,  # 18 min
        "setup_failed_gain": 0.05,  # +5%
        "theta_bleed_sec": 2700,  # 45 min
        "theta_bleed_loss": 0.30,  # -30%
        "eod_cutoff": (15, 40),
        "daily_loss_cap": 400,
        "scale_out": {5: 1.00, 4: 0.40, 3: 0.30, 2: 0.20, 1: 0.15},
    },
    "2500": {
        "stop_loss_base": 0.60,  # -60%
        "grace_period_sec": 600,
        "trail_activation": 0.50,  # +50%
        "trail_active_width": 0.35,
        "trail_runner_width": 0.45,
        "trail_moonshot_width": 0.30,
        "profit_lock_tiers": [(350, 220), (200, 120), (120, 60), (60, 20)],
        "setup_failed_sec": 1320,  # 22 min
        "setup_failed_gain": 0.03,  # +3%
        "theta_bleed_sec": 3000,  # 50 min
        "theta_bleed_loss": 0.35,  # -35%
        "eod_cutoff": (15, 45),
        "daily_loss_cap": 250,
        "scale_out": {5: 1.00, 3: 0.40},  # T3 + T5 only
    },
    "500": {
        "stop_loss_base": 0.70,  # -70%
        "grace_period_sec": 900,  # 15 min
        "trail_activation": 0.60,  # +60%
        "trail_active_width": 0.35,
        "trail_runner_width": 0.45,
        "trail_moonshot_width": 0.30,
        "profit_lock_tiers": [(400, 250), (200, 100), (100, 30)],
        "setup_failed_sec": 1800,  # 30 min
        "setup_failed_gain": 0.0,  # 0%
        "theta_bleed_sec": 3600,  # 60 min
        "theta_bleed_loss": 0.40,  # -40%
        "eod_cutoff": (15, 50),
        "daily_loss_cap": 75,
        "scale_out": {5: 1.00},  # T5 only
    },
}


def get_chrono_zone(et_hour: int, et_minute: int) -> tuple[str, dict]:
    """Determine chrono zone from ET time."""
    t = et_hour * 60 + et_minute
    for name, zone in CHRONO_ZONES.items():
        start = zone["start"][0] * 60 + zone["start"][1]
        end = zone["end"][0] * 60 + zone["end"][1]
        if start <= t < end:
            return name, zone
    # Default to AFTERNOON if outside defined zones
    return "AFTERNOON", CHRONO_ZONES["AFTERNOON"]


def get_ticker_tier(ticker: str) -> tuple[str, dict]:
    """Get ticker tier multipliers."""
    for tier_name, tier in TICKER_TIERS.items():
        if ticker in tier["tickers"]:
            return tier_name, tier
    return "STANDARD", TICKER_TIERS["STANDARD"]


def get_direction_mult(direction: str) -> dict:
    """Get direction multipliers."""
    if direction in ("put", "bearish"):
        return DIRECTION_MULT["bearish"]
    return DIRECTION_MULT["bullish"]


# ---------------------------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    ticker: str
    direction: str
    strike: float
    signal_time: str
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    mfe_pct: float
    mfe_gap: float
    exit_reason: str
    duration_min: float
    contracts: int
    pnl_dollars: float


def build_contract_ticker(underlying: str, expiry_date: str, strike: float, option_type: str) -> str:
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type == "call" else "P"
    strike_int = int(strike * 1000)
    return f"O:{underlying}{date_str}{opt_char}{strike_int:08d}"


def resolve_expiry(signal_time_str: str) -> str:
    dt = datetime.fromisoformat(signal_time_str)
    return dt.strftime("%Y-%m-%d")


def get_snapshots(harvester_conn, contract_ticker: str, after_time: str):
    rows = harvester_conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
        FROM harvest_snapshots
        WHERE contract_ticker = ?
          AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_time)).fetchall()
    return rows


def simulate_current_config(snapshots, entry_premium, signal_time, balance=5000):
    """Our current 'kitchen sink' config from last session."""
    if not snapshots:
        return entry_premium, entry_premium, "no_data", 0.0

    peak = entry_premium
    last_new_high_at = signal_time
    # Current config values
    grace_min = 8
    premium_stop_pct = 50.0
    profit_lock_tiers = [(80, 25), (150, 70), (250, 150)]  # sorted desc by threshold
    phase_trails = {0: 40.0, 1: 20.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}
    no_momentum_min = 45
    no_momentum_gain = 5.0
    time_decay_stale_min = 10.0
    time_decay_afternoon = (15, 30)
    time_decay_hold_min = 45.0
    theta_bleed_hold_min = 45.0
    theta_bleed_loss_pct = 30.0
    time_tighten_after = 60.0
    time_tighten_factor = 0.7
    expiry_safety_min = 10

    locked_floor = None
    last_target_hit = 0
    targets = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

    for snap in snapshots:
        captured_str, midpoint, bid, ask, underlying = snap
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        captured = datetime.fromisoformat(captured_str)
        if captured.tzinfo is not None and signal_time.tzinfo is None:
            captured = captured.replace(tzinfo=None)
        elapsed_min = (captured - signal_time).total_seconds() / 60

        if price > peak:
            peak = price
            last_new_high_at = captured

        gain_pct = (price - entry_premium) / entry_premium * 100
        for t in range(1, 6):
            if gain_pct >= targets[t] and t > last_target_hit:
                last_target_hit = t

        # Update profit lock
        peak_gain = (peak - entry_premium) / entry_premium * 100
        for threshold, lock in sorted(profit_lock_tiers, key=lambda x: -x[0]):
            if peak_gain >= threshold:
                locked_floor = lock
                break

        # Grace period
        if elapsed_min < grace_min:
            continue

        # Convert UTC to ET for time-of-day checks
        et_time = captured - timedelta(hours=4)  # UTC → EDT
        et_hour = et_time.hour
        et_minute = et_time.minute

        # Expiry safety — 4 PM ET = 20:00 UTC
        market_close_et = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        min_to_close = (market_close_et - et_time).total_seconds() / 60
        if 0 < min_to_close <= expiry_safety_min:
            return price, peak, "expiry_safety", elapsed_min

        # Premium stop
        loss_pct = (entry_premium - price) / entry_premium * 100
        if loss_pct >= premium_stop_pct:
            return price, peak, "premium_stop", elapsed_min

        # Profit lock
        if locked_floor is not None and gain_pct <= locked_floor:
            return price, peak, f"profit_lock({locked_floor:.0f}%)", elapsed_min

        # Time decay zone (using ET times)
        afternoon_time = time_decay_afternoon[0] * 60 + time_decay_afternoon[1]
        current_time_min = et_hour * 60 + et_minute
        in_decay = (elapsed_min > time_decay_hold_min) or (current_time_min >= afternoon_time)

        if in_decay:
            if last_new_high_at is not None:
                if captured.tzinfo is not None and last_new_high_at.tzinfo is None:
                    last_new_high_at = last_new_high_at.replace(tzinfo=captured.tzinfo)
                elif captured.tzinfo is None and last_new_high_at.tzinfo is not None:
                    last_new_high_at = last_new_high_at.replace(tzinfo=None)
                since_high = (captured - last_new_high_at).total_seconds() / 60
                if since_high >= time_decay_stale_min:
                    return price, peak, "time_decay_stale", elapsed_min

        # Phase trail
        phase = min(last_target_hit, 6)
        trail_pct = phase_trails.get(phase, phase_trails[0])
        if in_decay:
            trail_pct = min(trail_pct, 10.0)
        # Time tighten
        if elapsed_min > time_tighten_after:
            trail_pct *= time_tighten_factor
        if peak > 0:
            drop = (peak - price) / peak * 100
            if drop >= trail_pct:
                return price, peak, f"phase_trail(p{phase})", elapsed_min

        # Theta bleed
        if elapsed_min > theta_bleed_hold_min and loss_pct >= theta_bleed_loss_pct:
            return price, peak, "theta_bleed", elapsed_min

        # No momentum
        if elapsed_min >= no_momentum_min and gain_pct < no_momentum_gain:
            return price, peak, "no_momentum", elapsed_min

    # Market close
    if snapshots:
        last = snapshots[-1]
        lp = last[1] or ((last[2] or 0) + (last[3] or 0)) / 2
        lt = datetime.fromisoformat(last[0])
        if lt.tzinfo and not signal_time.tzinfo:
            lt = lt.replace(tzinfo=None)
        return lp, peak, "market_close", (lt - signal_time).total_seconds() / 60
    return entry_premium, entry_premium, "no_data", 0.0


def simulate_vinny_v21(snapshots, entry_premium, signal_time, ticker, direction,
                        account="5000", enable_chrono=True, enable_ticker_tier=True,
                        enable_direction=True, enable_adaptive_trail=True,
                        enable_per_account=True):
    """Simulate Vinny's v2.1 exit pipeline."""
    if not snapshots:
        return entry_premium, entry_premium, "no_data", 0.0

    params = ACCOUNT_PARAMS[account]
    peak = entry_premium
    trail_activated = False
    trail_peak = entry_premium
    last_new_high_at = signal_time
    scale_outs_done = set()

    # Resolve direction for multiplier
    dir_key = "bearish" if direction in ("put", "bearish") else "bullish"

    for snap in snapshots:
        captured_str, midpoint, bid, ask, underlying = snap
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        captured = datetime.fromisoformat(captured_str)
        if captured.tzinfo is not None and signal_time.tzinfo is None:
            captured = captured.replace(tzinfo=None)
        elapsed_sec = (captured - signal_time).total_seconds()
        elapsed_min = elapsed_sec / 60

        if price > peak:
            peak = price
            last_new_high_at = captured

        current_gain_pct = (price - entry_premium) / entry_premium
        peak_gain_pct = (peak - entry_premium) / entry_premium

        # Update trail peak
        if trail_activated and price > trail_peak:
            trail_peak = price

        # Convert UTC to ET for time-of-day checks
        et_time = captured - timedelta(hours=4)  # UTC → EDT
        et_hour = et_time.hour
        et_minute = et_time.minute

        # Compute multipliers
        chrono_name, chrono = get_chrono_zone(et_hour, et_minute)
        tier_name, tier = get_ticker_tier(ticker)
        dir_mult = get_direction_mult(direction)

        trail_mult = 1.0
        stop_mult = 1.0
        if enable_chrono:
            trail_mult *= chrono["trail_mult"]
            stop_mult *= chrono["stop_mult"]
        if enable_ticker_tier:
            trail_mult *= tier["trail_mult"]
            stop_mult *= tier["stop_mult"]
        if enable_direction:
            trail_mult *= dir_mult["trail_mult"]
            stop_mult *= dir_mult["stop_mult"]

        # §5.0 Expiry safety — 10 min before 4 PM ET
        market_close_et = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        min_to_close = (market_close_et - et_time).total_seconds() / 60
        if 0 < min_to_close <= 10:
            return price, peak, "EXPIRY_SAFETY", elapsed_min

        # Grace period
        if enable_per_account:
            grace = params["grace_period_sec"]
        else:
            grace = 480  # 8 min default
        if elapsed_sec < grace:
            continue

        # §5.4 Hard stop loss (with multipliers)
        if enable_per_account:
            effective_stop = params["stop_loss_base"] * stop_mult
        else:
            effective_stop = 0.50 * stop_mult
        if current_gain_pct <= -effective_stop:
            return price, peak, "STOP_LOSS", elapsed_min

        # §5.7 Profit lock floor
        if enable_per_account:
            pl_tiers = params["profit_lock_tiers"]
        else:
            pl_tiers = [(80, 25), (150, 70), (250, 150)]
        locked_floor = None
        for threshold, floor in sorted(pl_tiers, key=lambda x: -x[0]):
            if peak_gain_pct * 100 >= threshold:
                locked_floor = floor / 100.0
                break
        if locked_floor is not None and current_gain_pct < locked_floor:
            return price, peak, f"PROFIT_LOCK_{int(locked_floor*100)}%", elapsed_min

        # §5.8 Trailing stop (3-stage adaptive)
        if enable_adaptive_trail:
            activation = params["trail_activation"] if enable_per_account else 0.40
            if peak_gain_pct < activation:
                trail_stage = "DORMANT"
            elif peak_gain_pct < 1.50:
                trail_stage = "ACTIVE"
                base_width = params["trail_active_width"] if enable_per_account else 0.35
            elif peak_gain_pct < 4.00:
                trail_stage = "RUNNER"
                base_width = params["trail_runner_width"] if enable_per_account else 0.45
            else:
                trail_stage = "MOONSHOT"
                base_width = params["trail_moonshot_width"] if enable_per_account else 0.30

            if trail_stage != "DORMANT":
                if not trail_activated:
                    trail_activated = True
                    trail_peak = price
                trail_peak = max(trail_peak, price)

                effective_width = base_width * trail_mult
                drawdown_from_trail = (trail_peak - price) / trail_peak if trail_peak > 0 else 0
                if drawdown_from_trail >= effective_width:
                    return price, peak, f"TRAILING_STOP_{trail_stage}", elapsed_min
        else:
            # Fall back to our phase-based trail
            phase_trails = {0: 40.0, 1: 20.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}
            last_target_hit = 0
            targets = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80, 5: 1.00}
            for t in range(1, 6):
                if current_gain_pct >= targets[t]:
                    last_target_hit = t
            phase = min(last_target_hit, 6)
            trail_pct = phase_trails.get(phase, 40.0) / 100.0
            trail_pct *= trail_mult
            if peak > 0:
                drop = (peak - price) / peak
                if drop >= trail_pct:
                    return price, peak, f"phase_trail(p{phase})", elapsed_min

        # §5.10 Setup failed
        if enable_per_account:
            sf_time = params["setup_failed_sec"]
            sf_gain = params["setup_failed_gain"]
        else:
            sf_time = 999 * 60  # disabled
            sf_gain = 0.0
        if elapsed_sec > sf_time and peak_gain_pct < sf_gain and len(scale_outs_done) == 0:
            return price, peak, "SETUP_FAILED", elapsed_min

        # §5.11 Theta bleed
        if enable_per_account:
            tb_time = params["theta_bleed_sec"]
            tb_loss = params["theta_bleed_loss"]
        else:
            tb_time = 45 * 60
            tb_loss = 0.30
        if elapsed_sec > tb_time and current_gain_pct < -tb_loss:
            return price, peak, "THETA_BLEED", elapsed_min

        # §5.12 Time decay zone (after 15:00, 12 min stale)
        if et_hour >= 15:
            if last_new_high_at is not None:
                lnh = last_new_high_at
                if captured.tzinfo is not None and lnh.tzinfo is None:
                    lnh = lnh.replace(tzinfo=captured.tzinfo)
                elif captured.tzinfo is None and lnh.tzinfo is not None:
                    lnh = lnh.replace(tzinfo=None)
                since_high = (captured - lnh).total_seconds() / 60
                if since_high >= 12:
                    return price, peak, "TIME_DECAY_ZONE", elapsed_min

        # §5.14 EOD cutoff
        if enable_per_account:
            eod = params["eod_cutoff"]
        else:
            eod = (15, 40)
        eod_min = eod[0] * 60 + eod[1]
        current_min = et_hour * 60 + et_minute
        if current_min >= eod_min:
            return price, peak, "EOD_CUTOFF", elapsed_min

    # Market close
    if snapshots:
        last = snapshots[-1]
        lp = last[1] or ((last[2] or 0) + (last[3] or 0)) / 2
        lt = datetime.fromisoformat(last[0])
        if lt.tzinfo and not signal_time.tzinfo:
            lt = lt.replace(tzinfo=None)
        return lp, peak, "market_close", (lt - signal_time).total_seconds() / 60
    return entry_premium, entry_premium, "no_data", 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_signals(signals_conn):
    return signals_conn.execute("""
        SELECT ticker, direction, strike, expiry, atm_premium, otm_premium,
               score, created_at
        FROM trade_signals ORDER BY created_at
    """).fetchall()


def run_scenario(name, sim_func, sim_kwargs, signals, harvester_conn, balance=5000):
    """Run a scenario and return summary stats."""
    results = []
    skipped = 0

    for sig in signals:
        ticker, direction, strike, expiry, atm_premium, otm_premium, score, created_at = sig

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

        option_type = "call" if direction in ("call", "bullish", "long") else "put"
        contract_ticker = build_contract_ticker(ticker, expiry_date, strike, option_type)
        signal_time = datetime.fromisoformat(created_at)
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

        exit_premium, peak_premium, exit_reason, duration_min = sim_func(
            snapshots, entry_premium, signal_time,
            ticker=ticker, direction=direction,
            **sim_kwargs,
        )

        pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_pct = (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        mfe_gap = mfe_pct - pnl_pct

        # Simplified contract sizing
        cost = entry_premium * 100
        if cost > 0:
            total_deployable = balance * 0.80
            target = total_deployable / 3
            contracts = max(1, min(20, int(target / cost)))
        else:
            contracts = 1
        pnl_dollars = (exit_premium - entry_premium) * contracts * 100

        results.append(TradeResult(
            ticker=ticker, direction=direction, strike=strike,
            signal_time=created_at[:16], entry_premium=entry_premium,
            exit_premium=exit_premium, peak_premium=peak_premium,
            pnl_pct=pnl_pct, mfe_pct=mfe_pct, mfe_gap=mfe_gap,
            exit_reason=exit_reason, duration_min=duration_min,
            contracts=contracts, pnl_dollars=pnl_dollars,
        ))

    return results, skipped


def print_summary(name, results):
    """Print one-line summary for scenario comparison."""
    if not results:
        return
    wins = [r for r in results if r.pnl_pct >= 0]
    losses = [r for r in results if r.pnl_pct < 0]
    total_pnl = sum(r.pnl_dollars for r in results)
    wr = len(wins) / len(results) * 100
    avg_pnl = sum(r.pnl_pct for r in results) / len(results)
    avg_gap = sum(r.mfe_gap for r in results) / len(results)
    avg_dur = sum(r.duration_min for r in results) / len(results)

    reason_counts = {}
    for r in results:
        reason = r.exit_reason.split("(")[0].split("_")[0] if "(" in r.exit_reason else r.exit_reason
        # Simplify reason names
        reason = r.exit_reason
        if reason.startswith("TRAILING_STOP"):
            reason = "trail_" + reason.split("_")[-1]
        elif reason.startswith("profit_lock") or reason.startswith("PROFIT_LOCK"):
            reason = "profit_lock"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top3 = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])[:3])
    return (name, len(results), wr, total_pnl, avg_pnl, avg_gap, avg_dur, top3)


def print_trades(results, label=""):
    """Print per-trade detail."""
    print(f"\n  --- {label} ---")
    for r in results:
        win = "W" if r.pnl_pct >= 0 else "L"
        left = f" (left {r.mfe_gap:.0f}%)" if r.mfe_gap > 5 else ""
        print(f"  [{win}] {r.ticker:5} {r.direction:4} ${r.strike:<8} @ {r.signal_time[11:16]} | "
              f"PnL={r.pnl_pct:+6.1f}% MFE={r.mfe_pct:+5.0f}% | "
              f"{r.contracts}ct ${r.pnl_dollars:+8.2f} | {r.duration_min:5.0f}m | {r.exit_reason}{left}")


def main():
    signals_conn = sqlite3.connect(SIGNALS_DB)
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    signals = load_signals(signals_conn)

    print(f"{'='*130}")
    print(f"  VINNY v2.1 SPEC — BACKTEST AGAINST {len(signals)} REAL DISCORD SIGNALS")
    print(f"{'='*130}")

    # Current config as wrapper (doesn't need ticker/direction args)
    def current_sim(snapshots, entry_premium, signal_time, ticker=None, direction=None, **kw):
        return simulate_current_config(snapshots, entry_premium, signal_time)

    # Define all scenarios
    scenarios = [
        ("Current config (kitchen sink)", current_sim, {}),
        ("V2.1 FULL ($5K)", simulate_vinny_v21, {"account": "5000"}),
        ("V2.1 FULL ($2.5K)", simulate_vinny_v21, {"account": "2500"}),
        ("V2.1 FULL ($500)", simulate_vinny_v21, {"account": "500"}),
        ("V2.1 chrono-zones ONLY", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": True, "enable_ticker_tier": False,
            "enable_direction": False, "enable_adaptive_trail": False, "enable_per_account": False,
        }),
        ("V2.1 ticker-tiers ONLY", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": False, "enable_ticker_tier": True,
            "enable_direction": False, "enable_adaptive_trail": False, "enable_per_account": False,
        }),
        ("V2.1 direction-asymmetry ONLY", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": False, "enable_ticker_tier": False,
            "enable_direction": True, "enable_adaptive_trail": False, "enable_per_account": False,
        }),
        ("V2.1 adaptive-trail ONLY", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": False, "enable_ticker_tier": False,
            "enable_direction": False, "enable_adaptive_trail": True, "enable_per_account": False,
        }),
        ("V2.1 per-account $5K ONLY", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": False, "enable_ticker_tier": False,
            "enable_direction": False, "enable_adaptive_trail": False, "enable_per_account": True,
        }),
        ("V2.1 chrono + ticker-tier", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": True, "enable_ticker_tier": True,
            "enable_direction": False, "enable_adaptive_trail": False, "enable_per_account": False,
        }),
        ("V2.1 chrono + adaptive-trail", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": True, "enable_ticker_tier": False,
            "enable_direction": False, "enable_adaptive_trail": True, "enable_per_account": False,
        }),
        ("V2.1 chrono + ticker + direction", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": True, "enable_ticker_tier": True,
            "enable_direction": True, "enable_adaptive_trail": False, "enable_per_account": False,
        }),
        ("V2.1 ALL multipliers + phase trail", simulate_vinny_v21, {
            "account": "5000", "enable_chrono": True, "enable_ticker_tier": True,
            "enable_direction": True, "enable_adaptive_trail": False, "enable_per_account": True,
        }),
    ]

    # Run all scenarios
    all_summaries = []
    all_results = {}

    for name, sim_func, sim_kwargs in scenarios:
        results, skipped = run_scenario(name, sim_func, sim_kwargs, signals, harvester_conn)
        all_results[name] = results
        summary = print_summary(name, results)
        if summary:
            all_summaries.append(summary)

    # Print comparison table
    print(f"\n{'='*130}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'='*130}")
    print(f"  {'Scenario':<42} {'N':>3} {'WR':>5} {'Total P&L':>12} {'Avg P&L':>8} {'MFE Gap':>8} {'Dur':>5} {'Top Exit Reasons'}")
    print(f"  {'-'*42} {'-'*3} {'-'*5} {'-'*12} {'-'*8} {'-'*8} {'-'*5} {'-'*40}")

    for s in all_summaries:
        name, n, wr, pnl, avg, gap, dur, reasons = s
        print(f"  {name:<42} {n:>3} {wr:>4.0f}% ${pnl:>+10,.2f} {avg:>+7.1f}% {gap:>7.1f}% {dur:>4.0f}m {reasons}")

    # Detailed trade-by-trade for the best v2.1 scenario vs current
    print(f"\n{'='*130}")
    print(f"  TRADE-BY-TRADE: Current vs V2.1 FULL ($5K)")
    print(f"{'='*130}")

    current_results = all_results.get("Current config (kitchen sink)", [])
    v21_results = all_results.get("V2.1 FULL ($5K)", [])

    if current_results and v21_results:
        print(f"\n  {'Ticker':<6} {'Dir':>4} {'Strike':>8} {'Time':>6} | "
              f"{'CURRENT P&L':>11} {'CURRENT Exit':>25} | "
              f"{'V2.1 P&L':>11} {'V2.1 Exit':>25} | {'Delta':>7}")
        print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*6} | {'-'*11} {'-'*25} | {'-'*11} {'-'*25} | {'-'*7}")

        total_delta = 0
        for curr, v21 in zip(current_results, v21_results):
            delta = v21.pnl_pct - curr.pnl_pct
            total_delta += (v21.pnl_dollars - curr.pnl_dollars)
            marker = " <<<" if abs(delta) > 10 else ""
            print(f"  {curr.ticker:<6} {curr.direction:>4} ${curr.strike:<8} {curr.signal_time[11:16]:>5} | "
                  f"{curr.pnl_pct:>+10.1f}% {curr.exit_reason:>25} | "
                  f"{v21.pnl_pct:>+10.1f}% {v21.exit_reason:>25} | {delta:>+6.1f}%{marker}")

        curr_total = sum(r.pnl_dollars for r in current_results)
        v21_total = sum(r.pnl_dollars for r in v21_results)
        print(f"\n  Total P&L:  Current=${curr_total:+,.2f}  |  V2.1=${v21_total:+,.2f}  |  Delta=${total_delta:+,.2f}")

    # Per-day comparison
    print(f"\n{'='*130}")
    print(f"  DAILY P&L COMPARISON")
    print(f"{'='*130}")

    days = {}
    for r in current_results:
        day = r.signal_time[:10]
        if day not in days:
            days[day] = {"current": [], "v21": []}
        days[day]["current"].append(r)
    for r in v21_results:
        day = r.signal_time[:10]
        if day in days:
            days[day]["v21"].append(r)

    for day in sorted(days):
        curr_pnl = sum(r.pnl_dollars for r in days[day]["current"])
        v21_pnl = sum(r.pnl_dollars for r in days[day]["v21"])
        curr_n = len(days[day]["current"])
        v21_n = len(days[day]["v21"])
        print(f"  {day}: Current={curr_n} trades → ${curr_pnl:+,.2f}  |  V2.1={v21_n} trades → ${v21_pnl:+,.2f}  |  Delta=${v21_pnl-curr_pnl:+,.2f}")

    print(f"\n{'='*130}")

    signals_conn.close()
    harvester_conn.close()


if __name__ == "__main__":
    main()
