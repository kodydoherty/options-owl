"""Full parameter sweep — find profitable configs with 4-6+ trades/day.

Phase 1: Pre-compute option entries for ALL available strikes per signal (slow, once)
Phase 2: Sweep premium range × exit config × sizing combos (fast, in-memory)

Usage:
    python scripts/sweep_premium_configs.py                  # full sweep
    python scripts/sweep_premium_configs.py --days 30        # last 30 days
    python scripts/sweep_premium_configs.py --ticker NVDA     # single ticker
    python scripts/sweep_premium_configs.py --phase2-only     # skip phase 1 (use cached data)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ET = ZoneInfo("America/New_York")
THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
CACHE_FILE = PROJECT_DIR / "journal" / "sweep_cache.pkl"

# ── Tickers ─────────────────────────────────────────────────────────────
TICKERS = ["AMD", "TSLA", "NVDA", "META", "MSFT", "AAPL", "GOOGL", "AMZN",
           "AVGO", "SPY", "QQQ", "PLTR", "MSTR", "IWM"]


def find_strike_interval(price: float, ticker: str) -> float:
    if ticker in ("SPY", "QQQ", "IWM", "DIA"):
        return 1
    if price < 50:
        return 1
    if price < 100:
        return 2.5
    if price < 500:
        return 5
    return 5


# ═════════════════════════════════════════════════════════════════════════
# Phase 1: Pre-compute all possible option entries
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class OptionEntry:
    """One possible entry for a signal at a specific strike."""
    ticker: str
    date: str
    fire_time: str       # HH:MM ET
    direction: str       # CALL / PUT
    strike: float
    expiry: str
    entry_premium: float  # $/share at entry bar
    opt_bars: list        # list of (timestamp_str, open, high, low, close) for V5 FSM
    stock_price: float    # underlying at signal time
    moneyness: str        # ATM / OTM1 / OTM2 / ITM1 etc.


def load_trading_days(db_path: str, n_days: int) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Get the most recent n_days trading days (DESC then reverse)
    cur.execute("""
        SELECT DISTINCT DATE(timestamp) as d FROM stock_ohlc
        WHERE ticker = 'SPY'
        ORDER BY d DESC LIMIT ?
    """, (n_days,))
    days = sorted([r[0] for r in cur.fetchall()])
    conn.close()
    return days




def precompute_entries_for_day(db_path: str, ticker: str, date_str: str,
                                directions: list[str] = ["CALL", "PUT"],
                                entry_times: list[str] = None) -> list[OptionEntry]:
    """Find all possible option entries for a ticker on a given day.

    Batch-loads ALL option bars for the ticker-day in one query, then
    filters in-memory. Much faster than per-strike queries.
    """
    if entry_times is None:
        # Default: scan 9:35 through 11:00 every 5 minutes
        entry_times = [f"{h}:{m:02d}" for h in [9, 10, 11]
                       for m in range(0, 60, 5)
                       if (h == 9 and m >= 35) or (h == 10) or (h == 11 and m == 0)]

    conn = sqlite3.connect(db_path)

    # Batch load stock data
    stock_df = pd.read_sql_query("""
        SELECT timestamp, close FROM stock_ohlc
        WHERE ticker = ? AND DATE(timestamp) = ?
        ORDER BY timestamp
    """, conn, params=(ticker, date_str))
    if stock_df.empty:
        conn.close()
        return []
    stock_df["timestamp"] = pd.to_datetime(stock_df["timestamp"], utc=True).dt.tz_convert(ET)
    stock_df = stock_df[(stock_df["timestamp"].dt.hour >= 9) & (stock_df["timestamp"].dt.hour < 16)]
    if stock_df.empty:
        conn.close()
        return []

    interval = find_strike_interval(float(stock_df["close"].iloc[0]), ticker)

    # Batch load ALL option bars for this ticker-day (one query)
    opt_all = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume, strike, right, expiration
        FROM option_ohlc
        WHERE ticker = ? AND DATE(timestamp) = ?
        ORDER BY timestamp
    """, conn, params=(ticker, date_str))
    conn.close()

    if opt_all.empty:
        return []

    opt_all["timestamp"] = pd.to_datetime(opt_all["timestamp"], utc=True).dt.tz_convert(ET)
    opt_all = opt_all[(opt_all["timestamp"].dt.hour >= 9) & (opt_all["timestamp"].dt.hour < 16)]
    if opt_all.empty:
        return []

    # Prefer 0DTE expiry
    available_expiries = sorted(opt_all["expiration"].unique())
    expiry = date_str
    if date_str not in available_expiries:
        future = [e for e in available_expiries if e >= date_str]
        expiry = future[0] if future else (available_expiries[-1] if available_expiries else None)
    if expiry is None:
        return []

    # Filter to chosen expiry
    opt_exp = opt_all[opt_all["expiration"] == expiry]

    entries = []

    for entry_time in entry_times:
        h, m = map(int, entry_time.split(":"))
        target = stock_df["timestamp"].iloc[0].replace(hour=h, minute=m, second=0)
        nearby = stock_df[(stock_df["timestamp"] - target).abs() <= pd.Timedelta(minutes=3)]
        if nearby.empty:
            continue
        stock_price = float(nearby.iloc[0]["close"])
        atm = round(stock_price / interval) * interval

        for direction in directions:
            right = "CALL" if direction == "CALL" else "PUT"
            otm_dir = 1 if direction == "CALL" else -1
            strikes = [atm + i * interval * otm_dir for i in range(0, 6)]  # ATM + 5 OTM
            strikes += [atm - i * interval * otm_dir for i in range(1, 3)]  # 2 ITM

            for strike in strikes:
                if strike <= 0:
                    continue

                # Filter from pre-loaded data
                strike_bars = opt_exp[
                    (opt_exp["strike"] == strike) & (opt_exp["right"] == right)
                ].copy()
                if strike_bars.empty:
                    continue

                fire_dt = strike_bars["timestamp"].iloc[0].replace(hour=h, minute=m, second=0)
                after = strike_bars[strike_bars["timestamp"] >= fire_dt]
                if after.empty:
                    continue

                first_bar = after.iloc[0]
                if (first_bar["timestamp"] - fire_dt).total_seconds() > 300:
                    continue

                entry_price = float(first_bar["close"])
                if entry_price <= 0 or np.isnan(entry_price):
                    continue

                # Convert bars to cacheable tuples
                bars = []
                for _, row in after.iterrows():
                    bars.append((
                        row["timestamp"].isoformat(),
                        float(row["open"]), float(row["high"]),
                        float(row["low"]), float(row["close"]),
                    ))
                if len(bars) < 5:
                    continue

                # Classify moneyness
                otm_count = round((strike - atm) / interval) * (1 if direction == "CALL" else -1)
                if otm_count == 0:
                    moneyness = "ATM"
                elif otm_count > 0:
                    moneyness = f"OTM{otm_count}"
                else:
                    moneyness = f"ITM{abs(otm_count)}"

                entries.append(OptionEntry(
                    ticker=ticker, date=date_str, fire_time=entry_time,
                    direction=direction, strike=strike, expiry=expiry,
                    entry_premium=entry_price, opt_bars=bars,
                    stock_price=stock_price, moneyness=moneyness,
                ))

    return entries


# ═════════════════════════════════════════════════════════════════════════
# Phase 2: V5 FSM simulation on cached entries
# ═════════════════════════════════════════════════════════════════════════

def make_v6_settings(scalp_target: bool = True, scalp_pct: float = 25.0) -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_V6_BREAKEVEN_RATCHET=True,
        V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V6_SCALEOUT=True,
        V6_SCALEOUT_GAIN_PCT=20.0,
        V6_SCALEOUT_FRACTION=0.333,
        V6_SCALEOUT_MIN_CONTRACTS=3,
        ENABLE_V6_2PM_TIGHTEN=True,
        V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
        V6_2PM_SOFT_TRAIL_BOOST=0.15,
        ENABLE_V6_PER_TICKER_CONFIG=True,
        ENABLE_V6_PREMIUM_CAP=True,
        V6_PREMIUM_CAP=15.0,  # wide — we filter externally
        V6_PREMIUM_CAP_MID=15.0,
        V6_PREMIUM_CAP_HIGH=15.0,
        ENABLE_V6_SPREAD_GATE=True,
        V6_MAX_SPREAD_PCT=40.0,
        ENABLE_V6_EARLY_POP_GATE=True,
        ENABLE_V6_SIDEWAYS_SCALP=True,
        ENABLE_SCALP_TARGET=scalp_target,
        SCALP_TARGET_PCT=scalp_pct,
        SCALP_RUNNER_CONFIRM_PCT=40.0,
    )


def simulate_v5_on_bars(bars: list, entry_premium: float, contracts: int,
                         ticker: str, direction: str, fire_time_str: str,
                         date_str: str, v6_settings: SimpleNamespace) -> dict:
    """Run V5 FSM on pre-cached bars. Returns dollar P&L."""
    if entry_premium <= 0 or not bars:
        return {"pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0}

    option_type = "call" if direction == "CALL" else "put"
    tcfg = get_ticker_config(ticker, use_per_ticker=True, option_type=option_type)
    fsm = ExitFSM(tcfg, settings=v6_settings)

    fire_hour, fire_min = map(int, fire_time_str.split(":"))
    entry_ts = datetime(2026, 1, 1, fire_hour, fire_min, tzinfo=ET)

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=0, dte=0, expiry_date=date_str,
    )

    locked_pnl = 0.0
    remaining = contracts

    for i in range(1, len(bars)):
        ts_str, o, h, l, c = bars[i]
        prem = c
        if np.isnan(prem) or prem <= 0:
            continue

        bid = l if not np.isnan(l) else prem
        ask = h if not np.isnan(h) else prem

        ts = datetime.fromisoformat(ts_str)
        now = datetime(2026, 1, 1, ts.hour, ts.minute, tzinfo=ET)
        minutes_to_close = max(0, 16 * 60 - (ts.hour * 60 + ts.minute))

        action = fsm.evaluate(
            state, prem, bid, ask, now,
            current_underlying=0,
            minutes_to_close=minutes_to_close,
            candle_data={},
        )

        if action.should_exit:
            exit_price = bid if bid > 0 else prem
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (exit_price - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (exit_price - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold_min": i, "peak_gain": peak_gain,
            }

    # EOD fallback
    last_prem = entry_premium
    for ts_str, o, h, l, c in reversed(bars):
        if not np.isnan(c) and c > 0:
            last_prem = c
            break
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold_min": len(bars),
        "peak_gain": peak_gain,
    }


# ═════════════════════════════════════════════════════════════════════════
# Sweep engine
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    name: str
    prem_min: float
    prem_max: float
    scalp_on: bool
    scalp_pct: float
    capital_per_trade: float
    direction_filter: str  # "both", "call", "put"
    session_filter: str    # "all", "morning" (9:35-11:00), "kz" (9:30-10:45)
    moneyness_filter: str  # "all", "otm_only", "atm_only", "otm1_plus"
    exclude_tickers: set = field(default_factory=set)
    max_concurrent: int = 5
    max_position_dollars: float = 5000


@dataclass
class SweepResult:
    config: SweepConfig
    total_pnl: float
    trades: int
    wins: int
    losses: int
    trades_per_day: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown_pct: float
    runners_100: int
    runners_200: int
    best_trade: float
    worst_trade: float
    sharpe: float


def run_sweep_config(entries: list[OptionEntry], config: SweepConfig,
                      trading_days: list[str]) -> SweepResult:
    """Run one sweep config against pre-computed entries. Very fast (in-memory)."""
    v6 = make_v6_settings(scalp_target=config.scalp_on, scalp_pct=config.scalp_pct)

    # Group entries by day
    by_day = defaultdict(list)
    for e in entries:
        by_day[e.date].append(e)

    total_pnl = 0.0
    trade_pnls = []
    daily_pnls = defaultdict(float)
    portfolio = 23_000.0

    for day in trading_days:
        day_entries = by_day.get(day, [])
        if not day_entries:
            continue

        # Sort by fire_time
        day_entries.sort(key=lambda e: e.fire_time)

        open_tickers = set()
        open_count = 0

        for entry in day_entries:
            # Apply filters
            if entry.entry_premium < config.prem_min or entry.entry_premium > config.prem_max:
                continue
            if config.direction_filter == "call" and entry.direction != "CALL":
                continue
            if config.direction_filter == "put" and entry.direction != "PUT":
                continue
            if entry.ticker in config.exclude_tickers:
                continue

            # Session filter
            h, m = map(int, entry.fire_time.split(":"))
            entry_minutes = h * 60 + m
            if config.session_filter == "morning" and not (9 * 60 + 35 <= entry_minutes <= 11 * 60):
                continue
            if config.session_filter == "kz" and not (9 * 60 + 30 <= entry_minutes <= 10 * 60 + 45):
                continue

            # Moneyness filter
            if config.moneyness_filter == "otm_only" and not entry.moneyness.startswith("OTM"):
                continue
            if config.moneyness_filter == "atm_only" and entry.moneyness != "ATM":
                continue
            if config.moneyness_filter == "otm1_plus":
                if entry.moneyness in ("ATM", "ITM1", "ITM2"):
                    continue

            # Concurrent/duplicate limits
            if open_count >= config.max_concurrent:
                continue
            if entry.ticker in open_tickers:
                continue

            # Sizing
            cost_per = entry.entry_premium * 100
            raw_ct = int(config.capital_per_trade / cost_per) if cost_per > 0 else 1
            pos_cap = int(config.max_position_dollars / cost_per) if cost_per > 0 else 1
            contracts = max(1, min(raw_ct, pos_cap, 30))

            # Run V5 FSM
            result = simulate_v5_on_bars(
                entry.opt_bars, entry.entry_premium, contracts,
                entry.ticker, entry.direction, entry.fire_time,
                entry.date, v6
            )

            pnl = result["pnl"]
            total_pnl += pnl
            trade_pnls.append(pnl)
            daily_pnls[day] += pnl

            open_tickers.add(entry.ticker)
            open_count += 1
            # Assume 0DTE closes same day
            open_count -= 1

    # Compute metrics
    n_trades = len(trade_pnls)
    if n_trades == 0:
        return SweepResult(
            config=config, total_pnl=0, trades=0, wins=0, losses=0,
            trades_per_day=0, win_rate=0, profit_factor=0, avg_win=0,
            avg_loss=0, max_drawdown_pct=0, runners_100=0, runners_200=0,
            best_trade=0, worst_trade=0, sharpe=0,
        )

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    peak_eq = 23_000.0
    max_dd = 0.0
    running = 23_000.0
    for day in trading_days:
        running += daily_pnls.get(day, 0)
        peak_eq = max(peak_eq, running)
        dd = (peak_eq - running) / peak_eq * 100
        max_dd = max(max_dd, dd)

    # Sharpe (daily)
    daily_returns = [daily_pnls.get(d, 0) / 23_000 for d in trading_days]
    avg_ret = np.mean(daily_returns) if daily_returns else 0
    std_ret = np.std(daily_returns) if daily_returns else 1
    sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    return SweepResult(
        config=config, total_pnl=total_pnl, trades=n_trades,
        wins=len(wins), losses=len(losses),
        trades_per_day=n_trades / len(trading_days),
        win_rate=len(wins) / n_trades * 100,
        profit_factor=pf,
        avg_win=np.mean(wins) if wins else 0,
        avg_loss=np.mean(losses) if losses else 0,
        max_drawdown_pct=max_dd,
        runners_100=0,  # filled below if needed
        runners_200=0,
        best_trade=max(trade_pnls),
        worst_trade=min(trade_pnls),
        sharpe=sharpe,
    )


def generate_sweep_configs() -> list[SweepConfig]:
    """Generate all config combos to sweep."""
    configs = []

    # Premium ranges (the key dimension — user wants WIDE ranges tested)
    prem_ranges = [
        (0.05, 0.50, "$0.05-$0.50"),
        (0.10, 1.00, "$0.10-$1.00"),
        (0.20, 1.50, "$0.20-$1.50"),
        (0.30, 2.00, "$0.30-$2.00"),
        (0.50, 3.00, "$0.50-$3.00"),
        (0.75, 4.00, "$0.75-$4.00"),
        (1.00, 5.00, "$1.00-$5.00"),
        (1.50, 8.00, "$1.50-$8.00"),
        (2.00, 10.00, "$2.00-$10.00"),
        (3.00, 15.00, "$3.00-$15.00"),
        (0.10, 3.00, "$0.10-$3.00"),
        (0.10, 5.00, "$0.10-$5.00"),
        (0.10, 10.00, "$0.10-$10.00"),
        (0.10, 15.00, "$0.10-$15.00 (no cap)"),
        (0.50, 1.50, "$0.50-$1.50 (sweet)"),
        (0.30, 1.00, "$0.30-$1.00 (runners)"),
    ]

    # Scalp configs
    scalp_configs = [
        (True, 25.0, "scalp25"),
        (True, 50.0, "scalp50"),
        (False, 999.0, "no_scalp"),
    ]

    # Capital per trade
    capitals = [2000, 3500]

    # Direction
    direction_filters = ["both", "call"]

    # Session
    session_filters = ["morning", "all"]

    # Moneyness
    moneyness_filters = ["all", "otm_only"]

    for pmin, pmax, plabel in prem_ranges:
        for scalp_on, scalp_pct, slabel in scalp_configs:
            for capital in capitals:
                for direction in direction_filters:
                    for session in session_filters:
                        for money in moneyness_filters:
                            name = f"{plabel}|{slabel}|${capital}|{direction}|{session}|{money}"
                            configs.append(SweepConfig(
                                name=name,
                                prem_min=pmin, prem_max=pmax,
                                scalp_on=scalp_on, scalp_pct=scalp_pct,
                                capital_per_trade=capital,
                                direction_filter=direction,
                                session_filter=session,
                                moneyness_filter=money,
                            ))

    return configs


def print_results(results: list[SweepResult], title: str, limit: int = 30):
    print(f"\n{'=' * 120}")
    print(f"{title}")
    print(f"{'=' * 120}")
    print(f"{'Config':<55} {'Trades':>6} {'T/Day':>5} {'WR':>5} {'P&L':>12} {'PF':>6} {'MaxDD':>6} {'Sharpe':>7} {'Best':>10} {'Worst':>10}")
    print(f"{'─' * 120}")
    for r in results[:limit]:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
        print(f"{r.config.name:<55} {r.trades:>6} {r.trades_per_day:>5.1f} {r.win_rate:>4.0f}% "
              f"${r.total_pnl:>+10,.0f} {pf_str:>6} {r.max_drawdown_pct:>5.1f}% {r.sharpe:>7.2f} "
              f"${r.best_trade:>+9,.0f} ${r.worst_trade:>+9,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Premium config sweep")
    parser.add_argument("--days", type=int, default=60, help="Trading days")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    parser.add_argument("--phase2-only", action="store_true", help="Use cached phase 1 data")
    parser.add_argument("--no-cache", action="store_true", help="Skip reading/writing cache")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    # ── Phase 1: Pre-compute entries ────────────────────────────────────
    all_entries = []
    trading_days = []

    if args.phase2_only and CACHE_FILE.exists():
        print("Loading cached entries...")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            all_entries = cache["entries"]
            trading_days = cache["days"]
        print(f"  Loaded {len(all_entries)} entries across {len(trading_days)} days")
    else:
        print("Phase 1: Pre-computing all option entries...")
        print(f"  Tickers: {', '.join(tickers)}")
        trading_days = load_trading_days(THETADATA_DB, args.days)
        print(f"  Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

        t0 = time.time()
        for i, day in enumerate(trading_days):
            day_count = 0
            for ticker in tickers:
                entries = precompute_entries_for_day(
                    THETADATA_DB, ticker, day,
                    directions=["CALL", "PUT"],
                )
                all_entries.extend(entries)
                day_count += len(entries)

            if (i + 1) % 5 == 0 or i == len(trading_days) - 1:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(trading_days)}] {day} — {day_count} entries this day, "
                      f"{len(all_entries)} total [{elapsed:.0f}s]")

        elapsed = time.time() - t0
        print(f"\nPhase 1 complete: {len(all_entries)} entries in {elapsed:.0f}s")

        # Cache for reuse
        if not args.no_cache:
            print(f"Saving cache to {CACHE_FILE}...")
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"entries": all_entries, "days": trading_days}, f)
            cache_mb = CACHE_FILE.stat().st_size / 1024 / 1024
            print(f"  Cache saved ({cache_mb:.0f} MB)")

    if not all_entries:
        print("No entries found. Check ThetaData DB.")
        return

    # ── Quick stats ─────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("ENTRY UNIVERSE STATS:")
    prem_arr = np.array([e.entry_premium for e in all_entries])
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Premium range: ${prem_arr.min():.2f} - ${prem_arr.max():.2f}")
    print(f"  Median premium: ${np.median(prem_arr):.2f}")
    print(f"  < $1.00: {sum(1 for p in prem_arr if p < 1.0)} ({sum(1 for p in prem_arr if p < 1.0)/len(prem_arr)*100:.0f}%)")
    print(f"  $1-$3: {sum(1 for p in prem_arr if 1 <= p < 3)} ({sum(1 for p in prem_arr if 1 <= p < 3)/len(prem_arr)*100:.0f}%)")
    print(f"  $3-$10: {sum(1 for p in prem_arr if 3 <= p < 10)} ({sum(1 for p in prem_arr if 3 <= p < 10)/len(prem_arr)*100:.0f}%)")
    print(f"  $10+: {sum(1 for p in prem_arr if p >= 10)} ({sum(1 for p in prem_arr if p >= 10)/len(prem_arr)*100:.0f}%)")

    # By moneyness
    money_counts = defaultdict(int)
    for e in all_entries:
        money_counts[e.moneyness] += 1
    print(f"\n  By moneyness:")
    for m in sorted(money_counts.keys()):
        print(f"    {m:<8} {money_counts[m]:>6} ({money_counts[m]/len(all_entries)*100:.0f}%)")

    # By ticker
    tk_counts = defaultdict(int)
    for e in all_entries:
        tk_counts[e.ticker] += 1
    print(f"\n  By ticker:")
    for tk in sorted(tk_counts.keys()):
        print(f"    {tk:<8} {tk_counts[tk]:>6} ({tk_counts[tk]/len(all_entries)*100:.0f}%)")

    # By direction
    call_count = sum(1 for e in all_entries if e.direction == "CALL")
    put_count = sum(1 for e in all_entries if e.direction == "PUT")
    print(f"\n  CALL: {call_count} ({call_count/len(all_entries)*100:.0f}%)  PUT: {put_count} ({put_count/len(all_entries)*100:.0f}%)")

    # ── Phase 2: Sweep configs ──────────────────────────────────────────
    configs = generate_sweep_configs()
    print(f"\n{'=' * 80}")
    print(f"Phase 2: Sweeping {len(configs)} configs...")
    print(f"{'=' * 80}")

    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        result = run_sweep_config(all_entries, cfg, trading_days)
        results.append(result)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(configs)}] {elapsed:.0f}s...")

    elapsed = time.time() - t0
    print(f"\nSweep complete: {len(configs)} configs in {elapsed:.0f}s")

    # ── Results ─────────────────────────────────────────────────────────

    # Filter to configs with actual trades
    active = [r for r in results if r.trades >= 10]

    # TOP 30 BY P&L
    by_pnl = sorted(active, key=lambda r: r.total_pnl, reverse=True)
    print_results(by_pnl, "TOP 30 BY TOTAL P&L", 30)

    # TOP 30 BY PROFIT FACTOR (min 20 trades)
    by_pf = sorted([r for r in active if r.trades >= 20],
                    key=lambda r: r.profit_factor, reverse=True)
    print_results(by_pf, "TOP 30 BY PROFIT FACTOR (min 20 trades)", 30)

    # TOP 30 BY SHARPE
    by_sharpe = sorted(active, key=lambda r: r.sharpe, reverse=True)
    print_results(by_sharpe, "TOP 30 BY SHARPE RATIO", 30)

    # PROFITABLE CONFIGS WITH 4+ TRADES/DAY (user's target)
    high_vol = sorted([r for r in active if r.trades_per_day >= 4 and r.total_pnl > 0],
                       key=lambda r: r.total_pnl, reverse=True)
    print_results(high_vol, f"PROFITABLE CONFIGS WITH 4+ TRADES/DAY ({len(high_vol)} found)", 30)

    # PROFITABLE CONFIGS WITH 6+ TRADES/DAY
    very_high_vol = sorted([r for r in active if r.trades_per_day >= 6 and r.total_pnl > 0],
                            key=lambda r: r.total_pnl, reverse=True)
    print_results(very_high_vol, f"PROFITABLE CONFIGS WITH 6+ TRADES/DAY ({len(very_high_vol)} found)", 20)

    # PROFITABLE CONFIGS WITH 2+ TRADES/DAY + PF > 1.5
    quality = sorted([r for r in active if r.trades_per_day >= 2 and r.profit_factor > 1.5],
                      key=lambda r: r.total_pnl, reverse=True)
    print_results(quality, f"QUALITY CONFIGS: 2+ T/DAY + PF > 1.5 ({len(quality)} found)", 30)

    # $100K+ P&L configs
    big_pnl = sorted([r for r in active if r.total_pnl >= 100_000],
                      key=lambda r: r.total_pnl, reverse=True)
    print_results(big_pnl, f"$100K+ P&L CONFIGS ({len(big_pnl)} found)", 30)

    # Premium band analysis — what premium range is most profitable?
    print(f"\n{'=' * 120}")
    print("PREMIUM BAND ANALYSIS (best config per band):")
    print(f"{'=' * 120}")
    bands = [(0.05, 0.50), (0.10, 1.00), (0.30, 2.00), (0.50, 3.00),
             (1.00, 5.00), (2.00, 10.00), (0.10, 5.00), (0.10, 15.00)]
    for lo, hi in bands:
        band_results = [r for r in active
                        if r.config.prem_min == lo and r.config.prem_max == hi]
        if band_results:
            best = max(band_results, key=lambda r: r.total_pnl)
            pf_str = f"{best.profit_factor:.2f}" if best.profit_factor < 100 else "inf"
            print(f"  ${lo:.2f}-${hi:.2f}: Best={best.config.name}")
            print(f"    P&L=${best.total_pnl:+,.0f} | {best.trades} trades ({best.trades_per_day:.1f}/day) | "
                  f"WR={best.win_rate:.0f}% | PF={pf_str} | Sharpe={best.sharpe:.2f}")


if __name__ == "__main__":
    main()
