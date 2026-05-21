#!/usr/bin/env python3
"""Backtest: v4 FSM exit engine vs v2.2 (inline) vs PDF actuals.

Runs the v4 ExitFSM on the same 123 real trades from 0DTE-Performance-Report,
using the same synthetic premium paths as the v3-vs-v22 backtest.

Verifies that v4 FSM produces results within ~5% of inline v2.2 since v4
implements the same v2.2 spec.

Usage: python scripts/backtest_v4.py
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_owl.risk.exit_v5.config import V4Config
from options_owl.risk.exit_v5.fsm import ExitFSM, ExitReason, FSMState, TradeState


ET = ZoneInfo("America/New_York")

# ── 123 real trades from the PDF ──────────────────────────────────────────
# (date, time_et, ticker, bias, entry, peak, atm_pnl, atm_pct, peak_min, outcome)
TRADES_RAW = [
    ("2026-04-09","10:54","SPY","BULL",82,614,532,648.8,165,"WIN"),
    ("2026-04-09","11:30","IWM","BULL",70,287,217,310.0,109,"WIN"),
    ("2026-04-09","11:33","QQQ","BULL",76,376,300,394.7,29,"WIN"),
    ("2026-04-09","11:51","SPY","BULL",93,332,239,257.0,108,"WIN"),
    ("2026-04-10","13:06","MU","BEAR",325,263,-62,-19.1,4,"LOSS"),
    ("2026-04-10","13:12","AMZN","BULL",12,24,12,100.0,1,"WIN"),
    ("2026-04-10","13:27","MSTR","BULL",50,74,24,48.0,1,"PARTIAL"),
    ("2026-04-10","13:36","MU","BULL",160,460,300,187.5,19,"WIN"),
    ("2026-04-10","13:54","TSLA","BULL",68,395,327,480.9,125,"WIN"),
    ("2026-04-10","14:06","IWM","BULL",48,77,29,60.4,66,"WIN"),
    ("2026-04-10","14:15","NVDA","BULL",7,12,5,71.4,58,"WIN"),
    ("2026-04-10","14:27","AMD","BULL",119,141,22,18.5,1,"LOSS"),
    ("2026-04-10","15:06","SPY","BULL",23,50,27,117.4,8,"WIN"),
    ("2026-04-13","10:21","NVDA","BULL",126,230,104,82.5,37,"WIN"),
    ("2026-04-13","10:27","MSFT","BULL",213,950,737,346.0,332,"WIN"),
    ("2026-04-13","10:36","IWM","BULL",37,220,183,494.6,319,"WIN"),
    ("2026-04-13","10:51","SPY","BULL",97,634,537,553.6,315,"WIN"),
    ("2026-04-13","11:00","QQQ","BULL",68,472,404,594.1,313,"WIN"),
    ("2026-04-13","11:24","AMZN","BULL",75,234,159,212.0,274,"WIN"),
    ("2026-04-13","11:30","TSLA","BULL",211,477,266,126.1,132,"WIN"),
    ("2026-04-13","11:39","META","BULL",167,710,543,325.1,256,"WIN"),
    ("2026-04-13","12:09","QQQ","BULL",97,565,468,482.5,237,"WIN"),
    ("2026-04-13","12:15","IWM","BULL",14,220,206,1471.4,220,"WIN"),
    ("2026-04-14","09:27","SPY","BULL",184,856,672,365.2,365,"WIN"),
    ("2026-04-14","10:21","QQQ","BULL",80,547,467,583.8,339,"WIN"),
    ("2026-04-14","10:27","IWM","BULL",58,153,95,163.8,83,"WIN"),
    ("2026-04-14","10:51","SPY","BULL",119,357,238,200.0,280,"WIN"),
    ("2026-04-14","11:09","QQQ","BULL",159,547,388,244.0,291,"WIN"),
    ("2026-04-14","11:15","IWM","BULL",125,153,28,22.4,35,"LOSS"),
    ("2026-04-15","10:18","TSLA","BULL",57,1700,1643,2882.5,329,"WIN"),
    ("2026-04-15","10:21","AAPL","BULL",54,655,601,1113.0,336,"WIN"),
    ("2026-04-15","10:27","IWM","BULL",42,80,38,90.5,2,"WIN"),
    ("2026-04-15","10:30","SPY","BULL",68,329,261,383.8,328,"WIN"),
    ("2026-04-15","10:36","META","BULL",189,440,251,132.8,103,"WIN"),
    ("2026-04-15","10:54","GOOGL","BULL",56,254,198,353.6,239,"WIN"),
    ("2026-04-15","11:01","QQQ","BULL",126,559,433,343.6,294,"WIN"),
    ("2026-04-15","11:03","NVDA","BULL",42,94,52,123.8,45,"WIN"),
    ("2026-04-15","11:15","AAPL","BULL",123,400,277,225.2,282,"WIN"),
    ("2026-04-15","14:33","SPY","BULL",101,238,137,135.6,85,"WIN"),
    ("2026-04-15","15:24","SPY","BULL",41,53,12,29.3,10,"PARTIAL"),
    ("2026-04-16","10:01","IWM","BULL",71,140,69,97.2,96,"WIN"),
    ("2026-04-16","10:40","QQQ","BULL",139,440,301,216.6,99,"WIN"),
    ("2026-04-16","11:16","SPY","BULL",153,210,57,37.2,63,"PARTIAL"),
    ("2026-04-16","11:40","QQQ","BULL",125,185,60,48.0,39,"PARTIAL"),
    ("2026-04-16","12:40","SPY","BULL",82,88,6,7.3,1,"LOSS"),
    ("2026-04-16","12:46","QQQ","BULL",171,195,24,14.0,4,"LOSS"),
    ("2026-04-16","14:04","SPY","BULL",74,97,23,31.1,106,"PARTIAL"),
    ("2026-04-16","15:43","SPY","BULL",23,35,12,52.2,7,"WIN"),
    ("2026-04-17","10:25","SPY","BULL",152,348,196,128.9,141,"WIN"),
    ("2026-04-17","10:34","AMZN","BULL",104,162,58,55.8,12,"WIN"),
    ("2026-04-17","10:40","TSLA","BULL",305,969,664,217.7,49,"WIN"),
    ("2026-04-17","10:42","IWM","BULL",78,93,15,19.2,6,"LOSS"),
    ("2026-04-17","10:46","QQQ","BULL",143,180,37,25.9,50,"PARTIAL"),
    ("2026-04-17","10:49","AAPL","BULL",103,249,146,141.8,44,"WIN"),
    ("2026-04-17","10:55","META","BULL",236,404,168,71.2,287,"WIN"),
    ("2026-04-17","11:37","GOOGL","BULL",69,229,160,231.9,248,"WIN"),
    ("2026-04-17","11:40","NVDA","BULL",27,32,5,18.5,11,"LOSS"),
    ("2026-04-17","11:52","QQQ","BULL",131,164,33,25.2,45,"PARTIAL"),
    ("2026-04-17","12:10","IWM","BULL",54,93,39,72.2,36,"WIN"),
    ("2026-04-17","12:16","AMD","BULL",88,130,42,47.7,34,"PARTIAL"),
    ("2026-04-17","12:34","AMZN","BULL",126,189,63,50.0,74,"WIN"),
    ("2026-04-17","12:40","AAPL","BULL",30,35,5,16.7,1,"LOSS"),
    ("2026-04-17","12:58","PLTR","BULL",44,72,28,63.6,5,"WIN"),
    ("2026-04-17","13:01","NVDA","BULL",101,175,74,73.3,178,"WIN"),
    ("2026-04-17","13:25","QQQ","BULL",77,143,66,85.7,22,"WIN"),
    ("2026-04-17","13:28","TSLA","BEAR",148,179,31,20.9,45,"LOSS"),
    ("2026-04-17","13:31","IWM","BULL",56,72,16,28.6,20,"PARTIAL"),
    ("2026-04-17","14:25","NVDA","BULL",83,175,92,110.8,94,"WIN"),
    ("2026-04-20","10:48","IWM","BULL",54,85,31,57.4,52,"WIN"),
    ("2026-04-20","10:52","GOOGL","BULL",70,72,2,2.9,2,"LOSS"),
    ("2026-04-20","12:55","AMZN","BULL",41,123,82,200.0,171,"WIN"),
    ("2026-04-20","13:10","MSFT","BULL",58,103,45,77.6,13,"WIN"),
    ("2026-04-20","13:25","NVDA","BULL",29,217,188,648.3,154,"WIN"),
    ("2026-04-20","13:34","AAPL","BULL",35,124,89,254.3,53,"WIN"),
    ("2026-04-20","13:46","AMZN","BULL",33,123,90,272.7,120,"WIN"),
    ("2026-04-20","14:10","QQQ","BULL",80,105,25,31.2,45,"PARTIAL"),
    ("2026-04-20","14:25","IWM","BULL",37,73,36,97.3,30,"WIN"),
    ("2026-04-20","14:28","NVDA","BULL",78,217,139,178.2,91,"WIN"),
    ("2026-04-20","15:10","SPY","BULL",21,23,2,9.5,3,"LOSS"),
    ("2026-04-21","10:31","IWM","BULL",56,59,3,5.4,1,"LOSS"),
    ("2026-04-21","11:10","QQQ","BULL",140,186,46,32.9,10,"PARTIAL"),
    ("2026-04-21","15:30","SPY","BULL",95,112,17,17.9,1,"LOSS"),
    ("2026-04-22","11:00","AVGO","BULL",269,1085,816,303.4,294,"WIN"),
    ("2026-04-22","11:18","NVDA","BULL",27,40,13,48.1,163,"WIN"),
    ("2026-04-22","11:51","AMZN","BULL",58,335,277,477.6,241,"WIN"),
    ("2026-04-22","12:24","AVGO","BULL",188,350,162,86.2,214,"WIN"),
    ("2026-04-22","13:03","SPY","BULL",56,146,90,160.7,176,"WIN"),
    ("2026-04-22","13:06","NVDA","BULL",14,40,26,185.7,56,"WIN"),
    ("2026-04-22","13:12","QQQ","BULL",146,327,181,124.0,173,"WIN"),
    ("2026-04-22","13:51","AMZN","BULL",10,99,89,890.0,123,"WIN"),
    ("2026-04-22","14:00","NVDA","BULL",29,40,11,37.9,2,"PARTIAL"),
    ("2026-04-22","14:06","SPY","BULL",72,146,74,102.8,113,"WIN"),
    ("2026-04-22","14:15","GOOGL","BULL",116,234,118,101.7,97,"WIN"),
    ("2026-04-22","14:18","META","BULL",134,136,2,1.5,89,"LOSS"),
    ("2026-04-22","14:21","AVGO","BULL",125,350,225,180.0,98,"WIN"),
    ("2026-04-22","15:15","SPY","BULL",52,146,94,180.8,44,"WIN"),
    ("2026-04-23","10:45","SPY","BULL",118,183,65,55.1,47,"WIN"),
    ("2026-04-23","10:57","QQQ","BULL",154,240,86,55.8,36,"WIN"),
    ("2026-04-23","12:27","QQQ","BULL",140,166,26,18.6,9,"LOSS"),
    ("2026-04-23","13:54","SPY","BULL",156,359,203,130.1,42,"WIN"),
    ("2026-04-24","10:31","PLTR","BEAR",87,168,81,93.1,1,"WIN"),
    ("2026-04-24","11:02","NVDA","BULL",171,360,189,110.5,97,"WIN"),
    ("2026-04-24","11:06","MSTR","BEAR",159,178,19,11.9,3,"LOSS"),
    ("2026-04-24","11:26","AVGO","BEAR",257,450,193,75.1,12,"WIN"),
    ("2026-04-24","11:30","TSLA","BEAR",140,345,205,146.4,36,"WIN"),
    ("2026-04-24","11:47","SPY","BULL",173,178,5,2.9,4,"LOSS"),
    ("2026-04-24","11:54","AMZN","BULL",95,193,98,103.2,202,"WIN"),
    ("2026-04-24","12:24","TSLA","BEAR",177,270,93,52.5,18,"WIN"),
    ("2026-04-24","12:54","META","BULL",216,417,201,93.1,8,"WIN"),
    ("2026-04-24","13:12","AMZN","BULL",59,193,134,227.1,124,"WIN"),
    ("2026-04-24","13:46","TSLA","BULL",180,281,101,56.1,128,"WIN"),
    ("2026-04-24","14:00","QQQ","BULL",67,83,16,23.9,1,"LOSS"),
    ("2026-04-24","14:03","SPY","BULL",51,78,27,52.9,73,"WIN"),
    ("2026-04-24","14:07","GOOGL","BULL",139,266,127,91.4,106,"WIN"),
    ("2026-04-24","14:21","MSTR","BEAR",56,62,6,10.7,0,"LOSS"),
    ("2026-04-27","10:45","AAPL","BEAR",102,148,46,45.1,226,"PARTIAL"),
    ("2026-04-27","10:54","TSLA","BEAR",266,415,149,56.0,12,"WIN"),
    ("2026-04-27","12:09","NVDA","BULL",40,450,410,1025.0,227,"WIN"),
    ("2026-04-27","12:16","TSLA","BULL",245,1080,835,340.8,107,"WIN"),
    ("2026-04-27","13:15","NVDA","BULL",43,184,141,327.9,162,"WIN"),
    ("2026-04-27","13:24","QQQ","BULL",100,154,54,54.0,25,"WIN"),
    ("2026-04-27","13:33","TSLA","BULL",144,186,42,29.2,28,"PARTIAL"),
    ("2026-04-27","13:45","AVGO","BULL",65,120,55,84.6,128,"WIN"),
]

assert len(TRADES_RAW) == 123, f"Expected 123 trades, got {len(TRADES_RAW)}"


# ── Trade data structure ──────────────────────────────────────────────────

@dataclass
class Trade:
    date: str
    time_et: str
    ticker: str
    bias: str
    entry: float       # premium in dollars (per 1 contract = 100 shares)
    peak: float
    atm_pnl: float
    atm_pct: float
    peak_min: int       # minutes from entry to peak
    outcome: str        # WIN / PARTIAL / LOSS


def parse_trades() -> list[Trade]:
    trades = []
    for row in TRADES_RAW:
        date, time_et, ticker, bias, entry, peak, pnl, pct, peak_min, outcome = row
        trades.append(Trade(
            date=date, time_et=time_et, ticker=ticker, bias=bias,
            entry=float(entry), peak=float(peak), atm_pnl=float(pnl),
            atm_pct=float(pct), peak_min=max(int(peak_min), 1), outcome=outcome,
        ))
    return trades


# ── Synthetic premium path generation (identical to existing backtest) ────

def generate_premium_path(t: Trade, seed: int = 42) -> list[float]:
    """Returns a list of premiums, one per minute, from entry to EOD (4:00 PM ET).

    Path shape:
    1. Ramp from entry to peak over peak_min minutes (with noise)
    2. After peak, decay pattern depends on outcome:
       - WIN: hold near peak, gradual decay to ~80% of peak gain by EOD
       - PARTIAL: decay to ~40-60% of peak gain
       - LOSS: sharp reversal, ends below entry or near entry
    """
    rng = random.Random(seed)

    h, m = map(int, t.time_et.split(":"))
    entry_min_of_day = h * 60 + m
    eod_min = 16 * 60
    total_minutes = max(eod_min - entry_min_of_day, t.peak_min + 30)

    path = []

    for minute in range(total_minutes + 1):
        if minute <= t.peak_min:
            if t.peak >= t.entry:
                progress = minute / t.peak_min if t.peak_min > 0 else 1.0
                smooth = progress ** 0.7
                base = t.entry + (t.peak - t.entry) * smooth
            else:
                progress = minute / t.peak_min if t.peak_min > 0 else 1.0
                base = t.entry - (t.entry - t.peak) * (progress ** 0.5)

            noise = rng.gauss(0, base * 0.01)
            price = max(1.0, base + noise)

            if minute == t.peak_min:
                price = t.peak
        else:
            minutes_after_peak = minute - t.peak_min
            remaining_to_eod = total_minutes - t.peak_min
            decay_progress = minutes_after_peak / remaining_to_eod if remaining_to_eod > 0 else 1.0

            peak_gain = t.peak - t.entry

            if t.outcome == "WIN":
                retention = 1.0 - 0.2 * (decay_progress ** 1.5)
                base = t.entry + peak_gain * retention
            elif t.outcome == "PARTIAL":
                retention = 1.0 - 0.55 * (decay_progress ** 0.8)
                base = t.entry + peak_gain * retention
            else:
                if peak_gain > 0:
                    if decay_progress < 0.3:
                        retention = 1.0 - 3.0 * decay_progress
                        base = t.entry + peak_gain * retention
                    else:
                        final_price = t.entry + t.atm_pnl
                        past_peak_portion = (decay_progress - 0.3) / 0.7
                        base = (t.entry + peak_gain * (1 - 3 * 0.3)) * (1 - past_peak_portion) + final_price * past_peak_portion
                else:
                    final_price = t.entry + t.atm_pnl
                    base = t.entry + (final_price - t.entry) * (decay_progress ** 0.5)

            noise = rng.gauss(0, max(abs(base), 1) * 0.008)
            price = max(1.0, base + noise)

        path.append(round(price, 2))

    return path


# ── v2.2 Exit Strategy (inline, copied from existing backtest) ────────────

TICKER_MULTS = {"NVDA": 1.5, "TSLA": 1.5, "AMZN": 1.4, "AVGO": 1.4, "PLTR": 1.3}

V22_TRAIL_TIERS = [
    (400.0, 0.20),
    (200.0, 0.25),
    (100.0, 0.30),
    (50.0, 0.35),
]

HOUSE_MONEY_FLOORS_ATM = [
    (5.00, 2.00),
    (2.00, 0.80),
    (1.00, 0.30),
]

ATM_MILESTONE_LOCKS = [
    {"gain_pct": 200.0, "lock_fraction": 0.15},
    {"gain_pct": 400.0, "lock_fraction": 0.15},
    {"gain_pct": 600.0, "lock_fraction": 0.15},
]


@dataclass
class V22State:
    entry: float
    ticker: str = ""
    peak_premium: float = 0.0
    current_stop: float = 0.0
    trail_activated: bool = False
    trail_activate_gain_pct: float = 35.0
    milestones_locked: set = field(default_factory=set)
    house_money_floor: float = 0.0
    soft_trail_triggered: bool = False
    theta_timer_fired: bool = False
    contracts: float = 1.0
    locked_pnl: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    exited: bool = False


def theta_curve_multiplier_v22(current_hour: float) -> float:
    market_close_hour = 16.0
    hours_remaining = max(0.5, market_close_hour - current_hour)
    full_session_hours = 6.5
    raw = (hours_remaining / full_session_hours) ** 0.4
    return max(0.40, min(raw, 1.0))


def get_v22_trail_pct(gain_pct: float, tiers: list[tuple[float, float]]) -> float:
    for min_gain, trail in tiers:
        if gain_pct >= min_gain:
            return trail
    return tiers[-1][1] if tiers else 0.35


def apply_trail_multipliers_v22(
    base_trail_pct: float, ticker: str, is_morning: bool, score: float | None = None,
) -> float:
    mult = 1.0
    if ticker in TICKER_MULTS:
        mult *= TICKER_MULTS[ticker]
    if is_morning:
        mult *= 1.5
    if score is not None and score >= 90:
        mult *= 1.35
    mult = min(mult, 2.0)
    giveback_mult = min(mult, 1.20)
    effective_trail_pct = min(base_trail_pct * giveback_mult, 0.45)
    return effective_trail_pct


def run_v22(path: list[float], trade: Trade) -> V22State:
    """Simulate v2.2 exit strategy on a premium path."""
    s = V22State(entry=trade.entry, ticker=trade.ticker)
    s.peak_premium = trade.entry
    s.current_stop = trade.entry * 0.70  # hard stop at -30%

    h, m_val = map(int, trade.time_et.split(":"))
    entry_min = h * 60 + m_val
    is_morning = h < 12
    signal_score = 90.0

    for minute, price in enumerate(path):
        if s.exited:
            break

        current_min = entry_min + minute
        current_hour = current_min / 60.0
        gain = (price - s.entry) / s.entry
        gain_pct = gain * 100
        s.peak_premium = max(s.peak_premium, price)
        peak_gain = (s.peak_premium - s.entry) / s.entry
        peak_gain_pct = peak_gain * 100

        # Hard stop at -30%
        if price <= s.current_stop:
            s.exit_price = price
            s.exit_reason = "hard_stop"
            s.exited = True
            continue

        # Bar-1 reverse: first ~2 min, if down 5%+
        if minute <= 2 and gain_pct <= -5:
            s.exit_price = price
            s.exit_reason = "bar1_reverse"
            s.exited = True
            continue

        # Soft trail for 15-35% gain band
        if 15.0 <= peak_gain_pct < s.trail_activate_gain_pct:
            soft_floor_gain = 0.50 * peak_gain
            soft_floor_premium = s.entry * (1 + soft_floor_gain)
            if price <= soft_floor_premium:
                s.exit_price = price
                s.exit_reason = "soft_trail_break"
                s.exited = True
                continue

        # Main trailing stop (activated at +35%)
        if peak_gain_pct >= s.trail_activate_gain_pct:
            s.trail_activated = True

            base_trail = get_v22_trail_pct(peak_gain_pct, V22_TRAIL_TIERS)
            effective_trail = apply_trail_multipliers_v22(
                base_trail, trade.ticker, is_morning, signal_score,
            )
            theta_mult = theta_curve_multiplier_v22(current_hour)
            effective_trail *= theta_mult

            trail_stop = s.peak_premium * (1 - effective_trail)

            # House-money progressive floor
            for trigger, floor_gain in HOUSE_MONEY_FLOORS_ATM:
                if peak_gain >= trigger:
                    floor_premium = s.entry * (1 + floor_gain)
                    s.house_money_floor = max(s.house_money_floor, floor_premium)
                    break

            effective_stop = max(trail_stop, s.house_money_floor, s.current_stop)

            if price <= effective_stop:
                s.exit_price = price
                s.exit_reason = "trailing_stop"
                s.exited = True
                continue

        # ATM milestone profit locks
        for lock in ATM_MILESTONE_LOCKS:
            if gain_pct >= lock["gain_pct"] and lock["gain_pct"] not in s.milestones_locked:
                s.milestones_locked.add(lock["gain_pct"])
                qty_to_close = lock["lock_fraction"]
                s.locked_pnl += (price - s.entry) * qty_to_close
                s.contracts -= qty_to_close

        # Theta timer: 60 min default, immunity for high-score/tickers
        if not s.theta_timer_fired and minute >= 60:
            immune = False
            if signal_score >= 92:
                immune = True
            if trade.ticker in ("NVDA", "TSLA", "AMZN", "AVGO", "PLTR"):
                immune = True
            if is_morning and minute < 90:
                immune = True
            if current_hour >= 14 and minute < 80:
                immune = True

            if not immune and gain_pct <= 0:
                s.theta_timer_fired = True
                s.exit_price = price
                s.exit_reason = "theta_timer"
                s.exited = True
                continue

        # EOD cutoff (3:45 PM ET)
        if current_hour >= 15.75:
            s.exit_price = price
            s.exit_reason = "eod_cutoff"
            s.exited = True
            continue

    if not s.exited:
        s.exit_price = path[-1]
        s.exit_reason = "eod_force"
        s.exited = True

    return s


def v22_pnl(state: V22State) -> float:
    remaining_pnl = (state.exit_price - state.entry) * state.contracts
    return state.locked_pnl + remaining_pnl


# ── v4 FSM Runner ─────────────────────────────────────────────────────────

@dataclass
class V4Result:
    """Aggregated result from running v4 FSM on a premium path."""
    exit_price: float = 0.0
    exit_reason: str = ""
    contracts: int = 1
    locked_pnl: float = 0.0  # accumulated P&L from milestone partial exits
    exited: bool = False


def run_v4(path: list[float], trade: Trade, cfg: V4Config) -> V4Result:
    """Simulate v4 FSM exit engine on a premium path.

    Calls fsm.evaluate() once per minute (simulating 60s poll intervals).
    Uses 10 contracts to allow milestone lock granularity (15% of 10 = 1-2).
    """
    fsm = ExitFSM(cfg)
    result = V4Result(contracts=10)

    # Parse entry time with proper timezone
    h, m_val = map(int, trade.time_et.split(":"))
    year, month, day = map(int, trade.date.split("-"))
    entry_dt = datetime(year, month, day, h, m_val, 0, tzinfo=ET)

    is_morning = h < 12

    # Create TradeState
    ts = TradeState(
        trade_id=0,
        ticker=trade.ticker,
        option_type="call" if trade.bias == "BULL" else "put",
        is_otm=False,
        is_runner_tranche=False,
        entry_premium=trade.entry,
        entry_time=entry_dt,
        is_morning=is_morning,
        score=90.0,
        contracts=10,
    )
    ts.peak_premium = trade.entry

    entry_min = h * 60 + m_val

    for minute, price in enumerate(path):
        if result.exited:
            break

        now_et = entry_dt + timedelta(minutes=minute)

        # Minutes to market close (4:00 PM ET)
        current_min = entry_min + minute
        minutes_to_close = max(0.0, 16.0 * 60 - current_min)

        # Simulate bid/ask spread around premium
        bid = price * 0.97
        ask = price * 1.03

        action = fsm.evaluate(
            state=ts,
            current_premium=price,
            bid=bid,
            ask=ask,
            now_et=now_et,
            current_underlying=0.0,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            if action.reason == ExitReason.MILESTONE_LOCK and action.contracts_to_close > 0:
                # Partial exit: lock profits on closed contracts, keep monitoring
                close_qty = min(action.contracts_to_close, result.contracts)
                if close_qty > 0 and result.contracts > 1:
                    # P&L normalized to 1-contract equivalent (locked_qty / 10)
                    result.locked_pnl += (price - trade.entry) * (close_qty / 10.0)
                    result.contracts -= close_qty
                    ts.contracts = result.contracts
                # Do NOT mark as exited — continue monitoring remaining contracts
            else:
                # Full exit of remaining contracts
                result.exit_price = price
                result.exit_reason = action.reason.value
                result.exited = True

    # If never fully exited, force close at last price
    if not result.exited:
        result.exit_price = path[-1]
        result.exit_reason = "eod_force"
        result.exited = True

    return result


def v4_pnl(result: V4Result, entry: float) -> float:
    """Calculate total P&L including milestone lock partials.

    Remaining contracts are normalized to 1-contract equivalent (remaining / 10).
    """
    remaining_pnl = (result.exit_price - entry) * (result.contracts / 10.0)
    return result.locked_pnl + remaining_pnl


# ── Run backtest ──────────────────────────────────────────────────────────

def main():
    trades = parse_trades()
    random.seed(42)

    cfg = V4Config()

    by_date: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_date[t.date].append(t)

    total_v4 = 0.0
    total_v22 = 0.0
    total_pdf = 0.0

    all_v4_pnls: list[float] = []
    all_v22_pnls: list[float] = []
    v4_reasons: dict[str, int] = defaultdict(int)
    v22_reasons: dict[str, int] = defaultdict(int)

    print()
    print("=" * 130)
    print(f"{'BACKTEST: v4 FSM vs v2.2 (inline) vs PDF actuals':^130}")
    print(f"{'123 trades  |  13 sessions  |  2026-04-09 -> 2026-04-27':^130}")
    print("=" * 130)

    for date in sorted(by_date.keys()):
        day_trades = by_date[date]
        day_v4 = 0.0
        day_v22 = 0.0
        day_pdf = 0.0

        print()
        print(f"+-{'─' * 128}-+")
        print(f"| {date}  ({len(day_trades)} trades){'':>100}   |")
        print(f"+{'─' * 5}+{'─' * 7}+{'─' * 6}+{'─' * 7}+{'─' * 8}+{'─' * 8}+{'─' * 10}+{'─' * 24}+{'─' * 10}+{'─' * 24}+{'─' * 10}+")
        print(f"| {'#':>3} | {'TIME':>5} | {'TICK':>4} | {'ENTRY':>5} | {'PEAK':>6} | {'PDF':>6} | {'PDF P&L':>8} |"
              f" {'v4 FSM EXIT':^22} | {'v4 P&L':>8} | {'v2.2 EXIT':^22} | {'v2.2 P&L':>8} |")
        print(f"+{'─' * 5}+{'─' * 7}+{'─' * 6}+{'─' * 7}+{'─' * 8}+{'─' * 8}+{'─' * 10}+{'─' * 24}+{'─' * 10}+{'─' * 24}+{'─' * 10}+")

        for i, t in enumerate(day_trades):
            seed = hash(f"{t.date}{t.time_et}{t.ticker}") & 0xFFFFFFFF
            path = generate_premium_path(t, seed=seed)

            v4_result = run_v4(path, t, cfg)
            v22_state = run_v22(path, t)

            pnl_v4_val = v4_pnl(v4_result, t.entry)
            pnl_v22_val = v22_pnl(v22_state)
            pnl_pdf = t.atm_pnl

            all_v4_pnls.append(pnl_v4_val)
            all_v22_pnls.append(pnl_v22_val)
            v4_reasons[v4_result.exit_reason] += 1
            v22_reasons[v22_state.exit_reason] += 1

            day_v4 += pnl_v4_val
            day_v22 += pnl_v22_val
            day_pdf += pnl_pdf

            v4_exit = f"{v4_result.exit_reason} @${v4_result.exit_price:.0f}"
            v22_exit = f"{v22_state.exit_reason} @${v22_state.exit_price:.0f}"

            pnl_v4_str = f"${pnl_v4_val:+.0f}"
            pnl_v22_str = f"${pnl_v22_val:+.0f}"
            pnl_pdf_str = f"${pnl_pdf:+.0f}"

            # Flag significant divergence between v4 and v2.2
            diverge = " *" if abs(pnl_v4_val - pnl_v22_val) > max(abs(pnl_v22_val) * 0.15, 10) else ""

            print(f"| {i+1:>3} | {t.time_et:>5} | {t.ticker:>4} | ${t.entry:>4.0f} | ${t.peak:>5.0f} | {t.outcome:>6} |"
                  f" {pnl_pdf_str:>8} | {v4_exit:<22} | {pnl_v4_str:>8} | {v22_exit:<22} | {pnl_v22_str:>8} |{diverge}")

        total_v4 += day_v4
        total_v22 += day_v22
        total_pdf += day_pdf

        print(f"+{'─' * 5}+{'─' * 7}+{'─' * 6}+{'─' * 7}+{'─' * 8}+{'─' * 8}+{'─' * 10}+{'─' * 24}+{'─' * 10}+{'─' * 24}+{'─' * 10}+")
        match_marker = "  OK" if abs(day_v4 - day_v22) < max(abs(day_v22) * 0.10, 50) else "  DIVERGED"
        print(f"| {'DAY TOTALS':>42} | {f'${day_pdf:+,.0f}':>8} | {'':>22} | {f'${day_v4:+,.0f}':>8} |"
              f" {'':>22} | {f'${day_v22:+,.0f}':>8} |{match_marker}")
        print(f"+{'─' * 44}+{'─' * 10}+{'─' * 24}+{'─' * 10}+{'─' * 24}+{'─' * 10}+")

    # ── Grand Summary ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'GRAND TOTALS':^90}")
    print("=" * 90)
    print(f"  PDF (actual outcomes):     ${total_pdf:>+10,.0f}")
    print(f"  v4  (FSM engine):          ${total_v4:>+10,.0f}   ({total_v4/total_pdf*100:.1f}% of PDF)")
    print(f"  v2.2 (inline spec):        ${total_v22:>+10,.0f}   ({total_v22/total_pdf*100:.1f}% of PDF)")
    print()

    v4_vs_v22 = total_v4 - total_v22
    pct_diff = abs(v4_vs_v22) / abs(total_v22) * 100 if total_v22 != 0 else 0
    status = "PASS (<5%)" if pct_diff < 5.0 else ("CLOSE (<15%)" if pct_diff < 15.0 else "FAIL (>15%)")
    print(f"  v4 vs v2.2 difference:     ${v4_vs_v22:>+10,.0f}  ({pct_diff:.1f}% diff)  {status}")
    print()

    # ── Win rate ─────────────────────────────────────────────────────────
    v4_wins = sum(1 for p in all_v4_pnls if p > 0)
    v22_wins = sum(1 for p in all_v22_pnls if p > 0)
    pdf_wins = sum(1 for t in trades if t.atm_pnl > 0)
    n = len(trades)

    print(f"  Win rates:  PDF={pdf_wins}/{n} ({pdf_wins/n*100:.1f}%)  "
          f"v4={v4_wins}/{n} ({v4_wins/n*100:.1f}%)  "
          f"v2.2={v22_wins}/{n} ({v22_wins/n*100:.1f}%)")

    # ── Capture rate ────────────────────────────────────────────────────
    total_peak_available = sum(t.peak - t.entry for t in trades if t.peak > t.entry)
    print(f"  Capture rate (vs total peak available ${total_peak_available:,.0f}):")
    print(f"    PDF:  {total_pdf/total_peak_available*100:.1f}%")
    print(f"    v4:   {total_v4/total_peak_available*100:.1f}%")
    print(f"    v2.2: {total_v22/total_peak_available*100:.1f}%")
    print()

    # ── Average P&L per trade ────────────────────────────────────────────
    avg_v4 = total_v4 / n
    avg_v22 = total_v22 / n
    avg_pdf = total_pdf / n
    print(f"  Avg P&L per trade:  PDF=${avg_pdf:+.1f}  v4=${avg_v4:+.1f}  v2.2=${avg_v22:+.1f}")
    print()

    # ── Exit reason distribution ─────────────────────────────────────────
    print("  Exit reason distribution:")
    print(f"    {'Reason':<25} {'v4':>5}  {'v2.2':>5}")
    print(f"    {'─' * 37}")
    all_reasons = sorted(set(list(v4_reasons.keys()) + list(v22_reasons.keys())))
    for reason in all_reasons:
        print(f"    {reason:<25} {v4_reasons.get(reason, 0):>5}  {v22_reasons.get(reason, 0):>5}")
    print()

    # ── Largest per-trade divergences ────────────────────────────────────
    divergent = []
    for idx, (pv4, pv22) in enumerate(zip(all_v4_pnls, all_v22_pnls)):
        if abs(pv4 - pv22) > max(abs(pv22) * 0.20, 10):
            t = trades[idx]
            seed = hash(f"{t.date}{t.time_et}{t.ticker}") & 0xFFFFFFFF
            path = generate_premium_path(t, seed=seed)
            v4r = run_v4(path, t, cfg)
            v22s = run_v22(path, t)
            divergent.append((t, v4r, v22s, pv4, pv22))

    if divergent:
        print(f"  Largest divergences (>20% or >$10):")
        print(f"    {'TRADE':<25} {'v4 EXIT':<18} {'v4$':>7}  {'v2.2 EXIT':<18} {'v2.2$':>7}  {'DIFF':>7}")
        print(f"    {'─' * 85}")
        divergent.sort(key=lambda x: abs(x[3] - x[4]), reverse=True)
        for t, v4r, v22s, p4, p22 in divergent[:15]:
            print(f"    {t.date} {t.time_et} {t.ticker:<5}  "
                  f"{v4r.exit_reason:<18} ${p4:>+6.0f}  "
                  f"{v22s.exit_reason:<18} ${p22:>+6.0f}  "
                  f"${p4-p22:>+6.0f}")
        print()
    else:
        print("  No significant per-trade divergences found.")
        print()


if __name__ == "__main__":
    main()
