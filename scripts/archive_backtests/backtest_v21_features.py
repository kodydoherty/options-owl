"""Backtest the 4 remaining Vinny v2.1 features against historical 0DTE data.

Features tested:
  1. Three-tranche scale-out (§4) — split contracts into T1 lock, T2 default, T3 runner
  2. Underlying-anchored trail (§5) — trail on underlying price instead of premium
  3. Volume-peak modifier (§6) — detect exhaustion via 5m volume + candle structure
  4. Early Negative Thesis Revalidation Gate (ENRG spec) — simplified for backtest

Each feature is tested individually and in combination against the CURRENT config
(adaptive trail with v2.1 config retunes already applied) to measure incremental value.

Usage:
    python scripts/backtest_v21_features.py
    python scripts/backtest_v21_features.py --ticker SPY
    python scripts/backtest_v21_features.py --all
    python scripts/backtest_v21_features.py --feature tranches
    python scripts/backtest_v21_features.py --feature underlying_trail
    python scripts/backtest_v21_features.py --feature volume_peak
    python scripts/backtest_v21_features.py --feature early_neg_reval
"""

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 5000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trading_days(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT date, open_price, close_price, high_price, low_price,
               atm_call_ticker, atm_put_ticker, atm_strike,
               call_bars, put_bars, underlying_bars
        FROM trading_days
        WHERE ticker = ? AND call_bars > 0 AND put_bars > 0
              AND underlying_bars > 0
        ORDER BY date
    """, (ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_option_bars(contract_ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_underlying_bars(ticker, date):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM underlying_bars
        WHERE ticker = ? AND date = ?
        ORDER BY timestamp
    """, (ticker, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)  # EDT


def et_time_str(timestamp_ms):
    return ts_to_et(timestamp_ms).strftime("%H:%M")


def find_entry_bar(bars, entry_hour, entry_minute):
    target_minutes = entry_hour * 60 + entry_minute
    best_idx = None
    best_diff = float("inf")
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_minutes = et.hour * 60 + et.minute
        diff = abs(bar_minutes - target_minutes)
        if bar_minutes >= target_minutes and diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Feature implementations for backtesting
# ---------------------------------------------------------------------------

# §4 — Three-tranche scale-out
def plan_tranches(qty):
    """Split qty into tranches: T1 (lock at +25%), T2 (default trail), T3 (runner/wider trail)."""
    if qty <= 1:
        return [("T1", qty, "trailing")]
    if qty == 2:
        return [("T1", 1, "lock_at_25"), ("T2", 1, "trailing")]
    # 3+: split into thirds
    n1 = qty // 3
    n2 = qty // 3
    n3 = qty - n1 - n2
    return [
        ("T1", n1, "lock_at_25"),
        ("T2", n2, "trailing"),
        ("T3", n3, "runner"),
    ]


# §5 — Underlying-anchored trail tiers
UNDERLYING_TRAIL_TIERS = [
    (100.0, 0.0050),  # +100% premium gain → 0.50% underlying trail
    (50.0, 0.0040),   # +50% → 0.40%
    (15.0, 0.0030),   # +15% → 0.30%
    (0.0, 0.0020),    # +0% → 0.20%
]


def underlying_trail_pct_for_gain(premium_gain_pct):
    for min_gain, trail_pct in UNDERLYING_TRAIL_TIERS:
        if premium_gain_pct >= min_gain:
            return trail_pct
    return UNDERLYING_TRAIL_TIERS[-1][1]


# §6 — Volume-peak signal from underlying 5m bars
def compute_5m_bars(underlying_bars_1m, up_to_idx):
    """Aggregate 1-min underlying bars into 5-min bars ending at up_to_idx."""
    if up_to_idx < 4:
        return []
    bars_5m = []
    # Work backwards in chunks of 5
    start = max(0, up_to_idx - 14)  # last 3 five-min bars
    for i in range(start, up_to_idx + 1, 5):
        chunk = underlying_bars_1m[i:i + 5]
        if not chunk:
            continue
        bar = {
            "open": chunk[0]["open"],
            "high": max(b["high"] for b in chunk),
            "low": min(b["low"] for b in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(b["volume"] or 0 for b in chunk),
        }
        bars_5m.append(bar)
    return bars_5m


def volume_peak_signal(bars_5m, direction):
    """Returns None, 'tighten', or 'exit_half' based on last 3 completed 5m bars."""
    if len(bars_5m) < 3:
        return None

    last3 = bars_5m[-3:]
    last = last3[-1]

    # Price rising (in the alert's direction) but volume falling
    if direction == "call":
        price_rising = last3[-1]["close"] > last3[0]["close"]
    else:
        price_rising = last3[-1]["close"] < last3[0]["close"]
    vol_falling = last["volume"] < last3[0]["volume"] * 0.7

    rng = last["high"] - last["low"]
    body = abs(last["close"] - last["open"])
    long_wick = rng > 0 and (body / rng < 0.4)
    avg_vol3 = sum(b["volume"] for b in last3) / 3.0
    big_vol = last["volume"] > avg_vol3 * 2.0

    if price_rising and vol_falling:
        return "tighten"
    if long_wick and big_vol:
        return "exit_half"
    return None


# Early Negative Thesis Revalidation (simplified for backtest)
# Without multi-TF candle data from Polygon, we approximate using underlying price action
def early_neg_reval_check(direction, entry_premium, current_premium, entry_underlying,
                          current_underlying, underlying_bars_slice):
    """Simplified early negative thesis check using underlying momentum.

    Returns: 'HOLD', 'IMMEDIATE_EXIT', or 'PROCEED' (fall through to normal stops).
    """
    gain_pct = (current_premium - entry_premium) / entry_premium if entry_premium > 0 else 0

    # Only triggers when position is negative
    if gain_pct >= 0:
        return "PROCEED"

    # Check underlying direction alignment
    if len(underlying_bars_slice) < 5:
        return "PROCEED"

    # Simple momentum check: is underlying moving in our direction?
    recent = underlying_bars_slice[-5:]
    underlying_change = (recent[-1]["close"] - recent[0]["close"]) / recent[0]["close"]

    if direction == "call":
        thesis_holds = underlying_change > -0.001  # underlying not falling hard
    else:
        thesis_holds = underlying_change < 0.001   # underlying not rising hard

    # Check for extreme adverse move (falling knife / parabolic rise)
    extreme_move = abs(underlying_change) > 0.005  # >0.5% move in 5 min
    if direction == "call" and underlying_change < -0.005:
        return "IMMEDIATE_EXIT"
    if direction == "put" and underlying_change > 0.005:
        return "IMMEDIATE_EXIT"

    if thesis_holds:
        return "HOLD"  # widen stop by 15%
    return "PROCEED"


# ---------------------------------------------------------------------------
# Adaptive trail (current config — our baseline)
# ---------------------------------------------------------------------------

# Current v2.1 config retunes (already deployed)
CURRENT_CONFIG = {
    "premium_stop_pct": 30.0,      # was 50, retuned to 30
    "grace_period_min": 20,         # 20 min grace
    "adaptive_trail_activation": 35.0,  # was 40, retuned to 35
    "trail_active_width": 35.0,     # 35% drop from peak
    "trail_runner_width": 45.0,     # 45% — let runners breathe
    "trail_moonshot_width": 30.0,   # 30% — lock huge gains
    "runner_threshold": 150.0,      # +150% gain
    "moonshot_threshold": 400.0,    # +400% gain
    "profit_lock_tiers": [(250, 150), (150, 70), (80, 25)],
    "theta_bleed_hold_min": 45,
    "theta_bleed_loss_pct": 30.0,
    "no_momentum_min": 45,
    "no_momentum_gain_pct": 5.0,
    "eod_hour": 15,
    "eod_minute": 45,
    "entry_hour": 10,
    "entry_minute": 0,
    "entry_slippage_bps": 50.0,
    "exit_slippage_bps": 50.0,
    "simulated_score": 90,
}

# Runner trail tiers (§4.4 — wider for T3 tranche)
RUNNER_TRAIL_TIERS = [
    (400.0, 20.0),   # +400% gain → 20% trail
    (200.0, 30.0),   # +200% → 30%
    (100.0, 35.0),   # +100% → 35%
    (50.0, 40.0),    # +50% → 40%
]


def get_runner_trail_width(gain_pct):
    """Wider trail for T3 runner tranche."""
    for min_gain, trail_width in RUNNER_TRAIL_TIERS:
        if gain_pct >= min_gain:
            return trail_width
    return 40.0


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    date: str
    ticker: str
    direction: str
    strike: float
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    pnl_dollars: float
    contracts: int
    total_cost: float
    exit_reason: str
    duration_min: int
    mfe_pct: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    balance_after: float = 0.0


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate_trade(option_bars, underlying_bars_day, cfg, entry_idx, direction, strike,
                   balance, enable_tranches=False, enable_underlying_trail=False,
                   enable_volume_peak=False, enable_early_neg_reval=False):
    """Simulate a single trade with optional v2.1 features."""
    if entry_idx >= len(option_bars) - 10:
        return None

    entry_bar = option_bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] and entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry is None or raw_entry <= 0 or raw_entry < 0.25:
        return None

    entry_premium = raw_entry * (1 + cfg["entry_slippage_bps"] / 10000)

    # Score-based sizing (v2.1 tiers)
    score = cfg["simulated_score"]
    if score >= 95:
        base_contracts = 5
    elif score >= 90:
        base_contracts = 3
    elif score >= 85:
        base_contracts = 2
    elif score >= 78:
        base_contracts = 1
    else:
        return None

    cost_per = entry_premium * 100
    total_cost = base_contracts * cost_per
    contracts = base_contracts
    if total_cost > balance:
        contracts = max(1, int(balance / cost_per))
        total_cost = contracts * cost_per
    if total_cost > balance:
        return None

    # Build underlying timestamp index for quick lookup
    underlying_by_ts = {}
    for ub in underlying_bars_day:
        underlying_by_ts[ub["timestamp"]] = ub

    entry_ts = entry_bar["timestamp"]
    entry_underlying = None
    # Find closest underlying bar to entry
    for ub in underlying_bars_day:
        if ub["timestamp"] >= entry_ts - 60000:
            entry_underlying = ub["close"]
            break
    if entry_underlying is None and underlying_bars_day:
        entry_underlying = underlying_bars_day[-1]["close"]

    # --- Set up tranches ---
    if enable_tranches and contracts >= 2:
        tranches = plan_tranches(contracts)
    else:
        tranches = [("ALL", contracts, "trailing")]

    # Per-tranche state
    tranche_results = []
    for label, qty, exit_mode in tranches:
        tranche_results.append({
            "label": label,
            "qty": qty,
            "exit_mode": exit_mode,
            "active": True,
            "pnl_pct": 0.0,
            "exit_reason": None,
            "exit_bar": None,
        })

    # Simulation state
    peak_premium = entry_premium
    peak_underlying = entry_underlying or 0
    locked_floor = None
    reval_done = False
    stop_widen_applied = False
    effective_stop_pct = cfg["premium_stop_pct"]

    # Volume peak state
    vol_peak_tighten_applied = False

    for bar_idx in range(entry_idx + 1, len(option_bars)):
        bar = option_bars[bar_idx]
        minutes_elapsed = bar_idx - entry_idx
        current = bar["close"]
        if current is None or current <= 0:
            continue

        bar_et = ts_to_et(bar["timestamp"])

        # Track peaks
        if current > peak_premium:
            peak_premium = current

        # Find corresponding underlying price
        current_underlying = None
        for ub in underlying_bars_day:
            if ub["timestamp"] >= bar["timestamp"] - 60000:
                current_underlying = ub["close"]
                break
        if current_underlying is None:
            current_underlying = entry_underlying or 0
        if current_underlying > peak_underlying:
            peak_underlying = current_underlying

        gain_pct = (current - entry_premium) / entry_premium * 100
        peak_gain_pct = (peak_premium - entry_premium) / entry_premium * 100

        # Update profit lock floor
        for threshold, lock in sorted(cfg["profit_lock_tiers"], key=lambda x: -x[0]):
            if peak_gain_pct >= threshold:
                locked_floor = lock
                break

        # --- EOD hard exit ---
        if (bar_et.hour > cfg["eod_hour"] or
            (bar_et.hour == cfg["eod_hour"] and bar_et.minute >= cfg["eod_minute"])):
            for tr in tranche_results:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "eod_close"
                    tr["exit_bar"] = bar_idx
            break

        # --- Grace period ---
        if minutes_elapsed < cfg["grace_period_min"]:
            continue

        # --- Early Negative Thesis Revalidation (within first 20 min) ---
        if enable_early_neg_reval and not reval_done and minutes_elapsed <= 20 and gain_pct < 0:
            # Get recent underlying bars for momentum check
            ub_slice = [ub for ub in underlying_bars_day
                        if entry_ts <= ub["timestamp"] <= bar["timestamp"]]
            reval = early_neg_reval_check(
                direction, entry_premium, current, entry_underlying or 0,
                current_underlying, ub_slice,
            )
            if reval == "IMMEDIATE_EXIT":
                reval_done = True
                for tr in tranche_results:
                    if tr["active"]:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "early_neg_reval_exit"
                        tr["exit_bar"] = bar_idx
                break
            elif reval == "HOLD":
                reval_done = True
                stop_widen_applied = True
                effective_stop_pct *= 1.15  # widen stop by 15%
            else:
                reval_done = True

        # --- Premium stop (hard stop) ---
        loss_pct = (entry_premium - current) / entry_premium * 100 if entry_premium > 0 else 0
        if loss_pct >= effective_stop_pct:
            for tr in tranche_results:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "premium_stop"
                    tr["exit_bar"] = bar_idx
            break

        # --- Profit lock ---
        if locked_floor is not None and gain_pct <= locked_floor:
            for tr in tranche_results:
                if tr["active"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = f"profit_lock_{int(locked_floor)}%"
                    tr["exit_bar"] = bar_idx
            break

        # --- Volume peak modifier (§6) ---
        if enable_volume_peak and gain_pct >= 35:
            # Find underlying bar index
            ub_idx = None
            for i, ub in enumerate(underlying_bars_day):
                if ub["timestamp"] >= bar["timestamp"]:
                    ub_idx = i
                    break
            if ub_idx is not None and ub_idx >= 14:
                bars_5m = compute_5m_bars(underlying_bars_day, ub_idx)
                vp_signal = volume_peak_signal(bars_5m, direction)
                if vp_signal == "tighten" and not vol_peak_tighten_applied:
                    vol_peak_tighten_applied = True
                    # Tighten all active tranches' effective trail by 0.7x
                    # (applied via flag checked in trail logic below)
                elif vp_signal == "exit_half":
                    # Close half of remaining contracts
                    for tr in tranche_results:
                        if tr["active"] and tr["label"] in ("T1", "ALL"):
                            tr["active"] = False
                            tr["pnl_pct"] = gain_pct
                            tr["exit_reason"] = "volume_peak_half"
                            tr["exit_bar"] = bar_idx
                            break

        # --- Per-tranche exit logic ---
        any_active = False
        for tr in tranche_results:
            if not tr["active"]:
                continue
            any_active = True

            # T1 lock_at_25: exit immediately when gain >= 25%
            if tr["exit_mode"] == "lock_at_25" and gain_pct >= 25.0:
                tr["active"] = False
                tr["pnl_pct"] = gain_pct
                tr["exit_reason"] = "tranche1_lock_25%"
                tr["exit_bar"] = bar_idx
                continue

            # Adaptive trailing stop
            activation = cfg["adaptive_trail_activation"]
            if peak_gain_pct < activation:
                trail_stage = "DORMANT"
                trail_width = 100.0  # no trail
            elif peak_gain_pct < cfg["runner_threshold"]:
                trail_stage = "ACTIVE"
                trail_width = cfg["trail_active_width"]
            elif peak_gain_pct < cfg["moonshot_threshold"]:
                trail_stage = "RUNNER"
                trail_width = cfg["trail_runner_width"]
            else:
                trail_stage = "MOONSHOT"
                trail_width = cfg["trail_moonshot_width"]

            # T3 runner uses wider trail tiers
            if tr["exit_mode"] == "runner" and trail_stage != "DORMANT":
                trail_width = get_runner_trail_width(peak_gain_pct)

            # Volume peak tighten modifier
            if vol_peak_tighten_applied and trail_stage != "DORMANT":
                trail_width *= 0.7

            if trail_stage != "DORMANT" and peak_premium > 0:
                drop_from_peak = (peak_premium - current) / peak_premium * 100
                if drop_from_peak >= trail_width:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = f"adaptive_trail_{trail_stage}"
                    tr["exit_bar"] = bar_idx
                    continue

            # --- Underlying-anchored trail (§5) ---
            if (enable_underlying_trail and trail_stage != "DORMANT"
                    and current_underlying and peak_underlying > 0):
                u_trail_pct = underlying_trail_pct_for_gain(peak_gain_pct)
                if direction == "call":
                    trigger = peak_underlying * (1.0 - u_trail_pct)
                    if current_underlying < trigger:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "underlying_trail"
                        tr["exit_bar"] = bar_idx
                        continue
                else:  # put
                    trigger = peak_underlying * (1.0 + u_trail_pct)
                    if current_underlying > trigger:
                        tr["active"] = False
                        tr["pnl_pct"] = gain_pct
                        tr["exit_reason"] = "underlying_trail"
                        tr["exit_bar"] = bar_idx
                        continue

            # --- Theta bleed ---
            if minutes_elapsed >= cfg["theta_bleed_hold_min"]:
                if loss_pct >= cfg["theta_bleed_loss_pct"]:
                    tr["active"] = False
                    tr["pnl_pct"] = gain_pct
                    tr["exit_reason"] = "theta_bleed"
                    tr["exit_bar"] = bar_idx
                    continue

            # --- No momentum ---
            if (minutes_elapsed >= cfg["no_momentum_min"]
                    and gain_pct < cfg["no_momentum_gain_pct"]):
                tr["active"] = False
                tr["pnl_pct"] = gain_pct
                tr["exit_reason"] = "no_momentum"
                tr["exit_bar"] = bar_idx
                continue

        if not any_active:
            break

    # Close any still-active tranches at last bar
    last_bar = option_bars[-1]
    last_price = last_bar["close"]
    final_gain = (last_price - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
    for tr in tranche_results:
        if tr["active"]:
            tr["active"] = False
            tr["pnl_pct"] = final_gain
            tr["exit_reason"] = "eod_close"
            tr["exit_bar"] = len(option_bars) - 1

    # Compute weighted P&L across tranches
    total_qty = sum(tr["qty"] for tr in tranche_results)
    if total_qty == 0:
        return None

    weighted_pnl_pct = sum(tr["pnl_pct"] * tr["qty"] for tr in tranche_results) / total_qty
    exit_slippage = cfg["exit_slippage_bps"] / 100
    final_pnl_pct = weighted_pnl_pct - exit_slippage
    pnl_dollars = total_cost * (final_pnl_pct / 100)

    # Determine primary exit reason (from largest tranche or last to close)
    primary_reason = tranche_results[-1]["exit_reason"]
    if enable_tranches and len(tranche_results) > 1:
        reasons = [f"{tr['label']}:{tr['exit_reason']}({tr['pnl_pct']:+.0f}%)" for tr in tranche_results]
        primary_reason = " | ".join(reasons)

    last_exit_bar = max(tr["exit_bar"] or entry_idx for tr in tranche_results)
    mfe = (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

    return TradeResult(
        date="",
        ticker="",
        direction=direction,
        strike=strike,
        entry_premium=entry_premium,
        exit_premium=current if current else entry_premium,
        peak_premium=peak_premium,
        pnl_pct=final_pnl_pct,
        pnl_dollars=pnl_dollars,
        contracts=contracts,
        total_cost=total_cost,
        exit_reason=primary_reason,
        duration_min=last_exit_bar - entry_idx,
        mfe_pct=mfe,
        entry_time=et_time_str(option_bars[entry_idx]["timestamp"]),
        exit_time=et_time_str(option_bars[last_exit_bar]["timestamp"]) if last_exit_bar < len(option_bars) else "",
    )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(ticker, cfg, enable_tranches=False, enable_underlying_trail=False,
                 enable_volume_peak=False, enable_early_neg_reval=False, verbose=False):
    """Run a full backtest scenario on a single ticker.

    Uses FIXED position sizing per trade (no compounding) to isolate exit logic quality.
    Only trades calls (not both directions) to avoid doubling up on the same day.
    """
    trading_days = load_trading_days(ticker)
    if not trading_days:
        return None

    balance = STARTING_BALANCE  # fixed per-trade budget, not compounding
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    cumulative_pnl = 0.0

    for day in trading_days:
        date_str = day["date"]
        strike = day["atm_strike"]

        underlying_bars_day = load_underlying_bars(ticker, date_str)

        # Only trade calls (1 per day) to match realistic signal-driven usage
        direction = "call"
        contract_ticker = day["atm_call_ticker"]

        bars = load_option_bars(contract_ticker)
        if not bars or len(bars) < 30:
            continue

        entry_idx = find_entry_bar(bars, cfg["entry_hour"], cfg["entry_minute"])
        if entry_idx is None or entry_idx >= len(bars) - 10:
            continue

        result = simulate_trade(
            bars, underlying_bars_day, cfg, entry_idx, direction, strike,
            STARTING_BALANCE,  # fixed budget every trade
            enable_tranches=enable_tranches,
            enable_underlying_trail=enable_underlying_trail,
            enable_volume_peak=enable_volume_peak,
            enable_early_neg_reval=enable_early_neg_reval,
        )
        if result is None:
            continue

        result.date = date_str
        result.ticker = ticker
        cumulative_pnl += result.pnl_dollars
        result.balance_after = STARTING_BALANCE + cumulative_pnl
        trades.append(result)

        current_balance = STARTING_BALANCE + cumulative_pnl
        if current_balance > peak_balance:
            peak_balance = current_balance
        dd = (peak_balance - current_balance) / peak_balance * 100 if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    if not trades:
        return None

    wins = [t for t in trades if t.pnl_dollars >= 0]
    losses = [t for t in trades if t.pnl_dollars < 0]
    total_pnl = cumulative_pnl
    win_rate = len(wins) / len(trades) * 100
    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    avg_dur = sum(t.duration_min for t in trades) / len(trades)
    avg_mfe_gap = sum(t.mfe_pct - t.pnl_pct for t in trades) / len(trades)

    return {
        "ticker": ticker,
        "trades": trades,
        "balance": balance,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / STARTING_BALANCE * 100,
        "num_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_duration": avg_dur,
        "avg_mfe_gap": avg_mfe_gap,
    }


# ---------------------------------------------------------------------------
# Main — run all scenarios and compare
# ---------------------------------------------------------------------------

SCENARIOS = {
    "baseline": {
        "label": "Current v2.1 (baseline)",
        "flags": {},
    },
    "tranches": {
        "label": "§4 Three-tranche scale-out",
        "flags": {"enable_tranches": True},
    },
    "underlying_trail": {
        "label": "§5 Underlying-anchored trail",
        "flags": {"enable_underlying_trail": True},
    },
    "volume_peak": {
        "label": "§6 Volume-peak modifier",
        "flags": {"enable_volume_peak": True},
    },
    "early_neg_reval": {
        "label": "ENRG Early neg thesis reval",
        "flags": {"enable_early_neg_reval": True},
    },
    "tranches+underlying": {
        "label": "§4+§5 Tranches + Underlying trail",
        "flags": {"enable_tranches": True, "enable_underlying_trail": True},
    },
    "all_features": {
        "label": "ALL v2.1 features combined",
        "flags": {
            "enable_tranches": True,
            "enable_underlying_trail": True,
            "enable_volume_peak": True,
            "enable_early_neg_reval": True,
        },
    },
}


def run_backtest(ticker="SPY", feature=None, verbose=True):
    """Run backtest for a ticker across all or selected scenarios."""
    cfg = dict(CURRENT_CONFIG)

    if feature:
        scenarios_to_run = {"baseline": SCENARIOS["baseline"]}
        if feature in SCENARIOS:
            scenarios_to_run[feature] = SCENARIOS[feature]
        else:
            print(f"Unknown feature: {feature}")
            return
    else:
        scenarios_to_run = SCENARIOS

    results = {}
    for key, scenario in scenarios_to_run.items():
        result = run_scenario(ticker, cfg, **scenario["flags"])
        results[key] = result

    if verbose:
        print_comparison(ticker, results, scenarios_to_run)

    return results


def print_comparison(ticker, results, scenarios):
    """Print scenario comparison table."""
    print()
    print("=" * 130)
    print(f"  {ticker} — V2.1 FEATURE BACKTEST COMPARISON")
    print("=" * 130)
    print()
    print(f"  {'Scenario':<40} {'Trades':>6} {'WR':>5} {'Total P&L':>12} {'P&L%':>7} "
          f"{'PF':>5} {'MaxDD':>6} {'AvgW':>6} {'AvgL':>7} {'Dur':>5} {'MFE Gap':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*5} {'-'*12} {'-'*7} "
          f"{'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*8}")

    baseline = results.get("baseline")

    for key, scenario in scenarios.items():
        r = results.get(key)
        if r is None:
            print(f"  {scenario['label']:<40} {'NO DATA':>6}")
            continue

        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        delta = ""
        if baseline and key != "baseline":
            d = r["total_pnl"] - baseline["total_pnl"]
            delta = f"  ({d:+.0f})" if d != 0 else ""

        print(f"  {scenario['label']:<40} {r['num_trades']:>6} {r['win_rate']:>4.0f}% "
              f"${r['total_pnl']:>+10,.2f}{delta:>8} {r['total_pnl_pct']:>+6.1f}% "
              f"{pf_str:>5} {r['max_drawdown']:>5.1f}% "
              f"{r['avg_win']:>+5.0f}% {r['avg_loss']:>+6.0f}% "
              f"{r['avg_duration']:>4.0f}m {r['avg_mfe_gap']:>7.1f}%")

    # Print exit reason breakdown for each scenario
    print()
    print("  EXIT REASON BREAKDOWN:")
    print()
    for key, scenario in scenarios.items():
        r = results.get(key)
        if r is None:
            continue
        reasons = {}
        for t in r["trades"]:
            # Simplify tranche reasons for counting
            reason = t.exit_reason
            if " | " in reason:
                # Multi-tranche: take the last tranche's reason
                parts = reason.split(" | ")
                reason = "tranched:" + parts[-1].split(":")[1].split("(")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
        top5 = sorted(reasons.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{k}:{v}" for k, v in top5)
        print(f"  {scenario['label']:<40} {top_str}")

    print()
    print("=" * 130)


def run_all_tickers(feature=None, verbose=True):
    """Run backtest on all tickers and aggregate results."""
    conn = sqlite3.connect(DB_PATH)
    tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM trading_days WHERE call_bars > 0 AND underlying_bars > 0 "
        "ORDER BY ticker"
    ).fetchall()]
    conn.close()

    if not tickers:
        print("No data downloaded yet.")
        return

    if feature:
        scenarios_to_run = {"baseline": SCENARIOS["baseline"]}
        if feature in SCENARIOS:
            scenarios_to_run[feature] = SCENARIOS[feature]
    else:
        scenarios_to_run = SCENARIOS

    # Aggregate results per scenario
    aggregated = {key: {"total_pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                         "per_ticker": []} for key in scenarios_to_run}

    for ticker in tickers:
        cfg = dict(CURRENT_CONFIG)
        for key, scenario in scenarios_to_run.items():
            result = run_scenario(ticker, cfg, **scenario["flags"])
            if result is None:
                continue
            agg = aggregated[key]
            agg["total_pnl"] += result["total_pnl"]
            agg["trades"] += result["num_trades"]
            agg["wins"] += result["wins"]
            agg["losses"] += result["losses"]
            agg["per_ticker"].append(result)

    # Print aggregate comparison
    print()
    print("=" * 130)
    print(f"  ALL TICKERS ({len(tickers)}) — V2.1 FEATURE COMPARISON")
    print("=" * 130)
    print()
    print(f"  {'Scenario':<40} {'Tickers':>7} {'Trades':>6} {'WR':>5} {'Total P&L':>12} "
          f"{'vs Baseline':>12} {'AvgPnL/Ticker':>14}")
    print(f"  {'-'*40} {'-'*7} {'-'*6} {'-'*5} {'-'*12} {'-'*12} {'-'*14}")

    baseline_pnl = aggregated.get("baseline", {}).get("total_pnl", 0)

    for key, scenario in scenarios_to_run.items():
        agg = aggregated[key]
        n_tickers = len(agg["per_ticker"])
        wr = agg["wins"] / agg["trades"] * 100 if agg["trades"] > 0 else 0
        delta = agg["total_pnl"] - baseline_pnl
        avg_per = agg["total_pnl"] / n_tickers if n_tickers > 0 else 0

        delta_str = f"${delta:+,.0f}" if key != "baseline" else "—"
        print(f"  {scenario['label']:<40} {n_tickers:>7} {agg['trades']:>6} {wr:>4.0f}% "
              f"${agg['total_pnl']:>+10,.2f} {delta_str:>12} ${avg_per:>+12,.2f}")

    # Per-ticker breakdown
    print()
    print(f"  {'Ticker':<8}", end="")
    for key, scenario in scenarios_to_run.items():
        label = scenario["label"][:20]
        print(f" {label:>22}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in scenarios_to_run:
        print(f" {'-'*22}", end="")
    print()

    # Build per-ticker lookup
    ticker_results = {ticker: {} for ticker in tickers}
    for key in scenarios_to_run:
        for r in aggregated[key]["per_ticker"]:
            ticker_results[r["ticker"]][key] = r

    for ticker in tickers:
        print(f"  {ticker:<8}", end="")
        for key in scenarios_to_run:
            r = ticker_results[ticker].get(key)
            if r:
                print(f" ${r['total_pnl']:>+8,.0f} ({r['win_rate']:.0f}%WR)", end="")
            else:
                print(f" {'—':>22}", end="")
        print()

    print()
    print("=" * 130)

    # Show detailed results for top tickers
    top_tickers = ["SPY", "QQQ", "NVDA", "TSLA", "AMZN", "META"]
    for ticker in top_tickers:
        if ticker in [t for t in tickers]:
            print()
            run_backtest(ticker, feature=feature, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Vinny v2.1 remaining features")
    parser.add_argument("--ticker", default=None, help="Specific ticker to test")
    parser.add_argument("--all", action="store_true", help="Run on all tickers")
    parser.add_argument("--feature", default=None,
                        choices=["tranches", "underlying_trail", "volume_peak",
                                 "early_neg_reval", "all_features"],
                        help="Test a specific feature (default: all)")
    parser.add_argument("--score", type=int, default=90, help="Simulated signal score")
    parser.add_argument("--entry-hour", type=int, default=10, help="Entry hour (ET)")
    args = parser.parse_args()

    CURRENT_CONFIG["simulated_score"] = args.score
    CURRENT_CONFIG["entry_hour"] = args.entry_hour

    if args.all:
        run_all_tickers(feature=args.feature)
    elif args.ticker:
        run_backtest(ticker=args.ticker, feature=args.feature)
    else:
        # Default: run on SPY (most data)
        run_backtest(ticker="SPY", feature=args.feature)
