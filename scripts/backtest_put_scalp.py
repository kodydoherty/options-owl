"""PUT Scalp Track Backtest.

Scans ATM 0DTE PUTs at configurable time slots, uses simple fixed exit rules:
  - Target profit: e.g. +25%
  - Stop loss: e.g. -50%
  - Max hold: e.g. 30 min
  - No trailing, no FSM — hit target and bail

Data: ThetaData 1-min option OHLC (Jan 2023 – May 2026, 14 tickers)

Usage:
    python scripts/backtest_put_scalp.py                      # last 60 days, default params
    python scripts/backtest_put_scalp.py --days 120            # more history
    python scripts/backtest_put_scalp.py --sweep               # parameter sweep
    python scripts/backtest_put_scalp.py --all-days            # full 3+ year dataset
    python scripts/backtest_put_scalp.py --target 30 --stop 40 # custom params
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# ── Configuration ────────────────────────────────────────────────────────────

PORTFOLIO_START = 23_000
MAX_CONCURRENT = 4
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
MAX_POSITION_DOLLARS = 5_000
MAX_CONTRACTS = 200
GFV_BUFFER_PCT = 15.0
DAILY_LOSS_CB_PCT = 15.0


@dataclass
class PutScalpConfig:
    """Parameters for the PUT scalp strategy."""
    # Entry
    premium_floor: float = 0.05
    premium_cap: float = 0.50
    entry_slots: list[str] | None = None
    # Exit
    target_pct: float = 50.0    # take profit at +X%
    stop_pct: float = 60.0      # stop loss at -X%
    max_hold_min: int = 60      # max hold time in minutes
    # Filters
    min_volume: int = 0         # min option volume at entry bar
    max_spread_pct: float = 20.0  # max bid-ask spread %
    # DCA
    enable_dca: bool = False    # auto-double at dip
    dca_dip_pct: float = 25.0   # premium dip % to trigger DCA
    dca_window_min: int = 15    # minutes after entry to watch for dip
    # Ticker filter
    excluded_tickers: list[str] | None = None  # tickers to skip
    # Bear market acceleration
    bear_mode_threshold: float = -0.5  # stock down X% from open → aggressive mode
    bear_target_pct: float = 35.0      # lower target in bear mode (take quick profits)
    bear_max_concurrent: int = 6       # allow more concurrent in bear mode
    # Data
    use_bid_exit: bool = True

    def __post_init__(self):
        if self.entry_slots is None:
            self.entry_slots = ["09:30", "09:45", "10:00", "10:30", "11:00", "12:00", "13:00", "14:00"]
        if self.excluded_tickers is None:
            self.excluded_tickers = []


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_trading_days(conn, start_date: str | None = None, end_date: str | None = None) -> list[str]:
    """Get trading days from stock_ohlc."""
    query = "SELECT DISTINCT date(timestamp) FROM stock_ohlc WHERE ticker='SPY'"
    params = []
    if start_date:
        query += " AND date(timestamp) >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date(timestamp) <= ?"
        params.append(end_date)
    query += " ORDER BY 1"
    return [r[0] for r in conn.execute(query, params).fetchall()]


def load_put_data(conn, ticker: str, date_str: str):
    """Load ATM PUT option data for a ticker on a given day.

    Uses OHLC only (no quotes join) for speed. Approximates bid/ask from OHLC.
    Returns: (strike, bars)
    """
    # Get stock price at 9:30 to find ATM strike
    stock_open = conn.execute(
        "SELECT close FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp LIMIT 1",
        (ticker, date_str),
    ).fetchone()
    if not stock_open:
        return None, None
    stock_price = stock_open[0]

    # Find ATM PUT strike (0DTE)
    atm = conn.execute("""
        SELECT DISTINCT strike FROM option_ohlc
        WHERE ticker=? AND right='PUT' AND expiration=? AND date(timestamp)=?
        ORDER BY ABS(strike - ?) LIMIT 1
    """, (ticker, date_str, date_str, stock_price)).fetchone()
    if not atm:
        return None, None
    strike = atm[0]

    # Load all 1-min OHLC bars (no quotes join — 10x faster)
    rows = conn.execute("""
        SELECT time(timestamp), open, high, low, close, volume
        FROM option_ohlc
        WHERE ticker=? AND right='PUT' AND expiration=? AND date(timestamp)=? AND strike=?
        ORDER BY timestamp
    """, (ticker, date_str, date_str, strike)).fetchall()

    if not rows:
        return None, None

    bars = []
    for r in rows:
        time_str = r[0][:5]  # HH:MM
        close = r[4] or 0
        # Approximate bid/ask from OHLC (close ±3%)
        bars.append({
            "time": time_str,
            "open": r[1] or 0,
            "high": r[2] or 0,
            "low": r[3] or 0,
            "close": close,
            "volume": r[5] or 0,
            "bid": close * 0.97 if close > 0 else 0,
            "ask": close * 1.03 if close > 0 else 0,
        })

    return strike, bars


def load_stock_context(conn, ticker: str, date_str: str) -> dict | None:
    """Load stock context for filters/analysis."""
    rows = conn.execute(
        "SELECT time(timestamp), open, high, low, close, volume FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp",
        (ticker, date_str),
    ).fetchall()
    if not rows:
        return None

    by_time = {}
    for r in rows:
        t = r[0][:5]
        by_time[t] = {"open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}

    return {
        "open_price": rows[0][1],
        "by_time": by_time,
        "day_of_week": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
        "dow_num": datetime.strptime(date_str, "%Y-%m-%d").weekday(),
    }


# ── Trade Simulation ─────────────────────────────────────────────────────────


def simulate_put_scalp(bars: list[dict], entry_idx: int, entry_premium: float,
                       contracts: int, cfg: PutScalpConfig,
                       effective_target: float | None = None) -> dict:
    """Simulate a PUT scalp trade with fixed target/stop/maxhold + optional DCA.

    effective_target: override target_pct (used for bear mode).
    Returns trade result dict.
    """
    target = effective_target if effective_target is not None else cfg.target_pct
    peak_pct = 0.0
    trough_pct = 0.0

    # DCA tracking
    dca_done = False
    effective_entry = entry_premium
    effective_contracts = contracts
    dca_contracts = 0

    for i in range(entry_idx + 1, min(entry_idx + 1 + cfg.max_hold_min, len(bars))):
        bar = bars[i]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        if not high or high <= 0:
            continue

        # DCA: if premium dips by dca_dip_pct within window, double down
        minutes_in = i - entry_idx
        if cfg.enable_dca and not dca_done and 5 <= minutes_in <= cfg.dca_window_min:
            if close > 0:
                dip_pct = (entry_premium - close) / entry_premium * 100
                if dip_pct >= cfg.dca_dip_pct:
                    # Double position at lower price
                    dca_contracts = contracts
                    effective_entry = (entry_premium * contracts + close * dca_contracts) / (contracts + dca_contracts)
                    effective_contracts = contracts + dca_contracts
                    dca_done = True

        # Check vs effective entry (after DCA)
        pct_high = (high - effective_entry) / effective_entry * 100
        pct_low = (low - effective_entry) / effective_entry * 100 if low > 0 else 0

        if pct_high > peak_pct:
            peak_pct = pct_high
        if pct_low < trough_pct:
            trough_pct = pct_low

        # TARGET HIT
        if pct_high >= target:
            target_price = effective_entry * (1 + target / 100)
            exit_price = target_price * 0.98  # 2% slippage
            pnl = (exit_price - effective_entry) * effective_contracts * 100
            return {
                "pnl": round(pnl, 2),
                "reason": "target",
                "hold_min": i - entry_idx,
                "peak_pct": round(peak_pct, 1),
                "trough_pct": round(trough_pct, 1),
                "exit_prem": round(exit_price, 4),
                "dca": dca_done,
                "effective_contracts": effective_contracts,
            }

        # STOP HIT
        if pct_low <= -cfg.stop_pct:
            stop_price = effective_entry * (1 - cfg.stop_pct / 100)
            exit_price = stop_price * 0.97  # slippage on stop
            pnl = (exit_price - effective_entry) * effective_contracts * 100
            return {
                "pnl": round(pnl, 2),
                "reason": "stop",
                "hold_min": i - entry_idx,
                "peak_pct": round(peak_pct, 1),
                "trough_pct": round(trough_pct, 1),
                "exit_prem": round(exit_price, 4),
                "dca": dca_done,
                "effective_contracts": effective_contracts,
            }

    # MAX HOLD — exit at last bar
    last_bar = bars[min(entry_idx + cfg.max_hold_min, len(bars) - 1)]
    if cfg.use_bid_exit and last_bar["bid"] > 0:
        exit_price = last_bar["bid"]
    else:
        exit_price = last_bar["close"] if last_bar["close"] > 0 else effective_entry

    pnl = (exit_price - effective_entry) * effective_contracts * 100
    return {
        "pnl": round(pnl, 2),
        "reason": "max_hold",
        "hold_min": cfg.max_hold_min,
        "peak_pct": round(peak_pct, 1),
        "trough_pct": round(trough_pct, 1),
        "exit_prem": round(exit_price, 4),
        "dca": dca_done,
        "effective_contracts": effective_contracts,
    }


# ── Main Backtest ────────────────────────────────────────────────────────────


def run_backtest(conn, dates: list[str], cfg: PutScalpConfig, verbose: bool = True,
                 data_cache: dict | None = None) -> dict:
    """Run PUT scalp backtest over given dates.

    data_cache: optional {(ticker, date): (strike, bars)} to avoid re-querying DB.
    """
    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades = []
    daily_pnls = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    exit_reasons = defaultdict(int)
    equity_curve = [(dates[0], PORTFOLIO_START)]

    if data_cache is None:
        data_cache = {}

    for day_idx, date_str in enumerate(dates):
        day_open = 0
        day_spent = 0.0
        sod_balance = portfolio
        day_realized = 0.0

        if verbose and ((day_idx + 1) % 10 == 0 or day_idx == 0):
            print(f"  [{day_idx+1}/{len(dates)}] {date_str}  portfolio=${portfolio:,.0f}  "
                  f"trades={len(trades)}  dd={max_dd:.1f}%", flush=True)

        # Detect bear mode: check if SPY is down from open at each slot
        spy_context = load_stock_context(conn, "SPY", date_str) if cfg.bear_mode_threshold else None

        for ticker in TICKERS:
            if ticker in cfg.excluded_tickers:
                continue

            max_concurrent = MAX_CONCURRENT
            if day_open >= max_concurrent:
                break

            # Circuit breaker
            if day_realized < 0 and abs(day_realized) > sod_balance * DAILY_LOSS_CB_PCT / 100:
                break

            # Load PUT data (cached)
            cache_key = (ticker, date_str)
            if cache_key not in data_cache:
                data_cache[cache_key] = load_put_data(conn, ticker, date_str)
            strike, bars = data_cache[cache_key]
            if not bars or len(bars) < 10:
                continue

            # Index bars by time
            time_to_idx = {}
            for i, b in enumerate(bars):
                time_to_idx[b["time"]] = i

            # Try each entry slot
            for slot in cfg.entry_slots:
                if day_open >= max_concurrent:
                    break
                if slot not in time_to_idx:
                    continue

                entry_idx = time_to_idx[slot]
                bar = bars[entry_idx]

                # Need enough bars after entry
                remaining_bars = len(bars) - entry_idx - 1
                if remaining_bars < 5:
                    continue

                # Entry premium: use ask (what we'd pay)
                entry_premium = bar["ask"] if bar["ask"] > 0 else bar["close"]
                if not entry_premium or entry_premium <= 0:
                    continue

                # Premium filter
                if entry_premium < cfg.premium_floor or entry_premium > cfg.premium_cap:
                    continue

                # Spread filter
                if bar["bid"] > 0 and entry_premium > 0:
                    spread = (entry_premium - bar["bid"]) / entry_premium * 100
                    if spread > cfg.max_spread_pct:
                        continue

                # Volume filter
                if cfg.min_volume > 0 and bar["volume"] < cfg.min_volume:
                    continue

                # Bear mode detection: if SPY is down, use lower target (take profits faster)
                effective_target = None
                is_bear = False
                if spy_context and spy_context["by_time"].get(slot):
                    spy_at_slot = spy_context["by_time"][slot]["close"]
                    spy_open = spy_context["open_price"]
                    if spy_open > 0:
                        spy_move = (spy_at_slot - spy_open) / spy_open * 100
                        if spy_move <= cfg.bear_mode_threshold:
                            is_bear = True
                            effective_target = cfg.bear_target_pct
                            max_concurrent = cfg.bear_max_concurrent

                # Position sizing
                deployable = portfolio * MAX_RISK_PCT
                per_slot = deployable / max_concurrent
                position_cap = portfolio * MAX_POSITION_PCT
                cost_per = entry_premium * 100

                gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
                gfv_remaining = gfv_limit - day_spent
                if gfv_remaining < cost_per:
                    continue

                scaled = per_slot * 0.85
                raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
                cap_ct = int(position_cap / cost_per) if cost_per > 0 else 1
                gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                dollar_ct = int(MAX_POSITION_DOLLARS / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(raw_ct, cap_ct, gfv_ct, dollar_ct, MAX_CONTRACTS))

                day_spent += contracts * cost_per
                day_open += 1

                # Simulate exit
                result = simulate_put_scalp(bars, entry_idx, entry_premium, contracts, cfg,
                                           effective_target=effective_target)

                trade_pnl = result["pnl"]
                portfolio += trade_pnl
                day_realized += trade_pnl
                is_win = trade_pnl > 0

                per_ticker[ticker]["trades"] += 1
                if is_win:
                    per_ticker[ticker]["wins"] += 1
                per_ticker[ticker]["pnl"] += trade_pnl
                exit_reasons[result["reason"]] += 1

                trades.append({
                    "day": date_str, "ticker": ticker, "slot": slot,
                    "strike": strike, "entry": round(entry_premium, 4),
                    "contracts": contracts,
                    "pnl": trade_pnl, "reason": result["reason"],
                    "hold_min": result["hold_min"],
                    "peak_pct": result["peak_pct"],
                    "trough_pct": result["trough_pct"],
                    "exit_prem": result["exit_prem"],
                })

                if date_str not in daily_pnls:
                    daily_pnls[date_str] = 0
                daily_pnls[date_str] += trade_pnl

                # Only 1 entry per ticker per day for PUT scalps
                break

        # Track equity
        if portfolio > peak_portfolio:
            peak_portfolio = portfolio
        dd = (peak_portfolio - portfolio) / peak_portfolio * 100 if peak_portfolio > 0 else 0
        if dd > max_dd:
            max_dd = dd
        equity_curve.append((date_str, round(portfolio, 2)))

    return {
        "trades": trades,
        "portfolio": portfolio,
        "peak_portfolio": peak_portfolio,
        "max_dd": max_dd,
        "daily_pnls": daily_pnls,
        "per_ticker": dict(per_ticker),
        "exit_reasons": dict(exit_reasons),
        "equity_curve": equity_curve,
    }


# ── Reporting ────────────────────────────────────────────────────────────────


def print_results(results: dict, cfg: PutScalpConfig, label: str = ""):
    """Print backtest results summary."""
    trades = results["trades"]
    if not trades:
        print(f"\n  {label}NO TRADES")
        return

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    wr = len(wins) / n * 100
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    pf = abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 999

    # Sharpe (daily)
    dpnls = list(results["daily_pnls"].values())
    sharpe = 0
    if len(dpnls) >= 5:
        daily_returns = [p / PORTFOLIO_START for p in dpnls]
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        if std_r > 0:
            sharpe = mean_r / std_r * np.sqrt(252)

    print(f"\n{'='*70}")
    print(f"  {label}PUT SCALP RESULTS")
    print(f"{'='*70}")
    print(f"  Config: target={cfg.target_pct}%, stop={cfg.stop_pct}%, max_hold={cfg.max_hold_min}m")
    print(f"  Premium: ${cfg.premium_floor}-${cfg.premium_cap}, Slots: {cfg.entry_slots}")
    print(f"  Trades: {n}, W/L: {len(wins)}/{len(losses)}, WR: {wr:.1f}%")
    print(f"  Total P&L: ${total_pnl:+,.0f} ({total_pnl/PORTFOLIO_START*100:+.1f}%)")
    print(f"  Avg Win: ${avg_win:+,.0f}, Avg Loss: ${avg_loss:+,.0f}")
    print(f"  PF: {pf:.2f}, Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {results['max_dd']:.1f}%")
    print(f"  Final Portfolio: ${results['portfolio']:,.0f}")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason, count in sorted(results["exit_reasons"].items(), key=lambda x: -x[1]):
        reason_trades = [t for t in trades if t["reason"] == reason]
        reason_pnl = sum(t["pnl"] for t in reason_trades)
        reason_wr = len([t for t in reason_trades if t["pnl"] > 0]) / len(reason_trades) * 100
        print(f"    {reason:>12}: {count:>4} trades, {reason_wr:>5.1f}% WR, ${reason_pnl:>+10,.0f}")

    # Per ticker
    print(f"\n  Per Ticker:")
    print(f"    {'Ticker':>6} | {'N':>4} | {'WR':>5} | {'P&L':>10} | {'AvgP&L':>8}")
    print(f"    {'-'*6}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}")
    for tk in sorted(results["per_ticker"].keys(), key=lambda x: results["per_ticker"][x]["pnl"], reverse=True):
        d = results["per_ticker"][tk]
        wr_tk = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        avg = d["pnl"] / d["trades"] if d["trades"] else 0
        print(f"    {tk:>6} | {d['trades']:>4} | {wr_tk:>4.0f}% | ${d['pnl']:>+9,.0f} | ${avg:>+7,.0f}")

    # Daily P&L
    print(f"\n  Daily P&L (trading days only):")
    for day, pnl in sorted(results["daily_pnls"].items()):
        day_trades = [t for t in trades if t["day"] == day]
        day_w = len([t for t in day_trades if t["pnl"] > 0])
        day_l = len([t for t in day_trades if t["pnl"] <= 0])
        print(f"    {day}: ${pnl:>+8,.0f}  ({day_w}W/{day_l}L)")

    # Trade log
    if len(trades) <= 80:
        print(f"\n  Trade Log:")
        print(f"    {'Day':>10} {'Slot':>5} {'Ticker':>6} {'Entry':>6} {'Ct':>3} {'P&L':>9} {'Peak':>6} {'Hold':>5} {'Reason':>10}")
        print(f"    {'-'*10} {'-'*5} {'-'*6} {'-'*6} {'-'*3} {'-'*9} {'-'*6} {'-'*5} {'-'*10}")
        for t in trades:
            print(f"    {t['day']} {t['slot']:>5} {t['ticker']:>6} ${t['entry']:>5.2f} {t['contracts']:>3} ${t['pnl']:>+8,.0f} {t['peak_pct']:>5.1f}% {t['hold_min']:>4}m {t['reason']:>10}")

    return {
        "n": n, "wr": wr, "pnl": total_pnl, "pf": pf, "sharpe": sharpe,
        "max_dd": results["max_dd"], "avg_win": avg_win, "avg_loss": avg_loss,
    }


# ── Sweep ────────────────────────────────────────────────────────────────────


def _calc_stats(res):
    """Calculate summary stats from backtest results."""
    trades = res["trades"]
    if not trades:
        return None
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    wr = len(wins) / n * 100
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    avg_win = gross_win / len(wins) if wins else 0
    avg_loss = -gross_loss / len(losses) if losses else 0
    dpnls = list(res["daily_pnls"].values())
    sharpe = 0
    if len(dpnls) >= 5:
        dr = [p / PORTFOLIO_START for p in dpnls]
        if np.std(dr) > 0:
            sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252)
    return {
        "n": n, "wr": wr, "pnl": total_pnl, "pf": pf, "sharpe": sharpe,
        "dd": res["max_dd"], "avg_win": avg_win, "avg_loss": avg_loss,
    }


def _print_row(label, s):
    """Print a sweep result row."""
    print(f"  {label:>25} | {s['n']:>4} | {s['wr']:>4.0f}% | ${s['pnl']:>+9,.0f} | {s['pf']:>5.2f} | {s['sharpe']:>6.2f} | {s['dd']:>4.1f}%", flush=True)


def run_sweep(conn, dates: list[str]):
    """Parameter sweep: entry time FIRST, then exit params, then premium range."""

    print("\n" + "=" * 100, flush=True)
    print("PUT SCALP PARAMETER SWEEP", flush=True)
    print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} days)", flush=True)
    print("=" * 100, flush=True)

    # Pre-warm data cache — load all ticker+date combos ONCE
    print("  Pre-loading data...", end="", flush=True)
    data_cache = {}
    for i, date_str in enumerate(dates):
        for ticker in TICKERS:
            data_cache[(ticker, date_str)] = load_put_data(conn, ticker, date_str)
        if (i + 1) % 50 == 0:
            print(f" {i+1}/{len(dates)}", end="", flush=True)
    print(f" done ({len(data_cache)} entries)", flush=True)

    HDR = f"  {'Config':>25} | {'N':>4} | {'WR':>5} | {'P&L':>10} | {'PF':>5} | {'Sharpe':>6} | {'DD':>5}"
    SEP = f"  {'-'*25}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}"

    # ── Phase 1: ENTRY TIME (the primary variable) ──
    # Test each individual time slot with moderate exit params
    print("\n--- Phase 1: ENTRY TIME (individual slots, +25%/-40%/30m, $0.20-$2.00) ---", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)

    individual_slots = [
        "09:30", "09:35", "09:40", "09:45", "09:50", "09:55",
        "10:00", "10:15", "10:30", "10:45",
        "11:00", "11:30", "12:00", "12:30",
        "13:00", "13:30", "14:00", "14:30",
    ]
    slot_results = {}

    for slot in individual_slots:
        cfg = PutScalpConfig(
            target_pct=25, stop_pct=40, max_hold_min=30,
            premium_floor=0.20, premium_cap=2.00,
            entry_slots=[slot],
        )
        res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
        s = _calc_stats(res)
        if s:
            _print_row(slot, s)
            slot_results[slot] = s

    # Test time windows
    print("\n--- Phase 1b: ENTRY TIME WINDOWS ---", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)

    windows = [
        ("open 9:30-9:55", ["09:30", "09:35", "09:40", "09:45", "09:50", "09:55"]),
        ("mid-morn 10:00-10:45", ["10:00", "10:15", "10:30", "10:45"]),
        ("late-morn 11:00-12:30", ["11:00", "11:30", "12:00", "12:30"]),
        ("afternoon 13:00-14:30", ["13:00", "13:30", "14:00", "14:30"]),
        ("all-day wide", ["09:30", "09:45", "10:00", "10:30", "11:00", "12:00", "13:00", "14:00"]),
        ("best-of open+afternoon", ["09:30", "09:45", "13:00", "13:30", "14:00", "14:30"]),
    ]

    best_window = None
    best_window_pf = 0
    window_results = {}

    for label, slots in windows:
        cfg = PutScalpConfig(
            target_pct=25, stop_pct=40, max_hold_min=30,
            premium_floor=0.20, premium_cap=2.00,
            entry_slots=slots,
        )
        res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
        s = _calc_stats(res)
        if s:
            _print_row(label, s)
            window_results[label] = (s, slots)
            if s["pf"] > best_window_pf and s["n"] >= 10:
                best_window_pf = s["pf"]
                best_window = (label, slots)

    # ── Phase 2: EXIT PARAMS (using best time window) ──
    if not best_window:
        print("\n  NO PROFITABLE WINDOW FOUND", flush=True)
        return

    best_label, best_slots = best_window
    print(f"\n--- Phase 2: EXIT PARAMS (using {best_label}) ---", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)

    targets = [15, 20, 25, 30, 35, 50]
    stops = [20, 25, 30, 35, 40, 50, 60]
    max_holds = [10, 15, 20, 30, 45, 60]

    all_results = []

    for target in targets:
        for stop in stops:
            for hold in max_holds:
                cfg = PutScalpConfig(
                    target_pct=target, stop_pct=stop, max_hold_min=hold,
                    premium_floor=0.20, premium_cap=2.00,
                    entry_slots=best_slots,
                )
                res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
                s = _calc_stats(res)
                if s and s["n"] >= 5:
                    lbl = f"+{target}%/-{stop}%/{hold}m"
                    all_results.append({"label": lbl, "target": target, "stop": stop, "hold": hold, **s})

    # Top 15 by PF
    print(f"\n  TOP 15 EXIT CONFIGS BY PROFIT FACTOR (time={best_label}):", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)
    top = sorted(all_results, key=lambda x: x["pf"], reverse=True)[:15]
    for r in top:
        _print_row(r["label"], r)

    # Top 15 by Sharpe
    print(f"\n  TOP 15 EXIT CONFIGS BY SHARPE:", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)
    tops = sorted(all_results, key=lambda x: x["sharpe"], reverse=True)[:15]
    for r in tops:
        _print_row(r["label"], r)

    # Top 15 by raw P&L
    print(f"\n  TOP 15 EXIT CONFIGS BY P&L:", flush=True)
    print(HDR, flush=True)
    print(SEP, flush=True)
    topp = sorted(all_results, key=lambda x: x["pnl"], reverse=True)[:15]
    for r in topp:
        _print_row(r["label"], r)

    # ── Phase 3: PREMIUM RANGE (using best time + best exit) ──
    if top:
        best_exit = top[0]
        target, stop, hold = best_exit["target"], best_exit["stop"], best_exit["hold"]
        print(f"\n--- Phase 3: PREMIUM RANGE (time={best_label}, exit=+{target}%/-{stop}%/{hold}m) ---", flush=True)
        print(HDR, flush=True)
        print(SEP, flush=True)

        prem_ranges = [
            (0.05, 0.50), (0.05, 1.00), (0.10, 1.00), (0.10, 2.00),
            (0.20, 1.00), (0.20, 2.00), (0.20, 3.00),
            (0.50, 2.00), (0.50, 3.00), (0.50, 5.00),
            (1.00, 3.00), (1.00, 5.00), (1.00, 8.00),
        ]

        best_prem = None
        best_prem_pf = 0

        for lo, hi in prem_ranges:
            cfg = PutScalpConfig(
                target_pct=target, stop_pct=stop, max_hold_min=hold,
                premium_floor=lo, premium_cap=hi,
                entry_slots=best_slots,
            )
            res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
            s = _calc_stats(res)
            if s:
                lbl = f"${lo:.2f}-${hi:.2f}"
                _print_row(lbl, s)
                if s["pf"] > best_prem_pf and s["n"] >= 10:
                    best_prem_pf = s["pf"]
                    best_prem = (lo, hi)

    # ── Phase 4: RE-SWEEP EXIT with best premium (cross-validate) ──
    if best_prem and top:
        lo, hi = best_prem
        print(f"\n--- Phase 4: RE-SWEEP EXIT with best premium ${lo}-${hi} + best time ---", flush=True)
        print(HDR, flush=True)
        print(SEP, flush=True)

        final_results = []
        for target in targets:
            for stop in stops:
                for hold in max_holds:
                    cfg = PutScalpConfig(
                        target_pct=target, stop_pct=stop, max_hold_min=hold,
                        premium_floor=lo, premium_cap=hi,
                        entry_slots=best_slots,
                    )
                    res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
                    s = _calc_stats(res)
                    if s and s["n"] >= 5:
                        lbl = f"+{target}%/-{stop}%/{hold}m"
                        final_results.append({"label": lbl, **s})

        print(f"\n  FINAL TOP 15 BY PF:", flush=True)
        print(HDR, flush=True)
        print(SEP, flush=True)
        for r in sorted(final_results, key=lambda x: x["pf"], reverse=True)[:15]:
            _print_row(r["label"], r)

        print(f"\n  FINAL TOP 15 BY SHARPE:", flush=True)
        print(HDR, flush=True)
        print(SEP, flush=True)
        for r in sorted(final_results, key=lambda x: x["sharpe"], reverse=True)[:15]:
            _print_row(r["label"], r)

    # ── Phase 5: OPTIMAL CONFIG DEEP DIVE ──
    if top:
        best = top[0]
        plo, phi = best_prem if best_prem else (0.20, 2.00)
        print(f"\n{'='*70}", flush=True)
        print(f"  OPTIMAL CONFIG DEEP DIVE", flush=True)
        print(f"{'='*70}", flush=True)
        cfg = PutScalpConfig(
            target_pct=best["target"], stop_pct=best["stop"], max_hold_min=best["hold"],
            premium_floor=plo, premium_cap=phi,
            entry_slots=best_slots,
        )
        res = run_backtest(conn, dates, cfg, verbose=False, data_cache=data_cache)
        print_results(res, cfg, label="OPTIMAL ")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PUT Scalp Backtest")
    parser.add_argument("--days", type=int, default=60, help="Trading days to test")
    parser.add_argument("--all-days", action="store_true", help="Use full dataset")
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep mode")
    parser.add_argument("--target", type=float, default=25.0, help="Target profit %%")
    parser.add_argument("--stop", type=float, default=50.0, help="Stop loss %%")
    parser.add_argument("--max-hold", type=int, default=30, help="Max hold minutes")
    parser.add_argument("--premium-floor", type=float, default=0.20)
    parser.add_argument("--premium-cap", type=float, default=2.00)
    parser.add_argument("--slots", type=str,
                        default="09:30,09:45,10:00,10:30,11:00,12:00,13:00,14:00",
                        help="Comma-separated entry slots")
    parser.add_argument("--dca", action="store_true", help="Enable DCA on premium dip")
    parser.add_argument("--dca-dip", type=float, default=25.0, help="DCA dip threshold %%")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated tickers to exclude")
    parser.add_argument("--bear-threshold", type=float, default=-0.5, help="SPY drop %% for bear mode")
    parser.add_argument("--bear-target", type=float, default=35.0, help="Target %% in bear mode")
    args = parser.parse_args()

    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA cache_size = -200000")

    # Get dates
    if args.all_days:
        dates = load_trading_days(conn)
    else:
        all_dates = load_trading_days(conn)
        dates = all_dates[-args.days:]

    print(f"PUT Scalp Backtest")
    print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Portfolio: ${PORTFOLIO_START:,}")

    if args.sweep:
        run_sweep(conn, dates)
    else:
        slots = args.slots.split(",")
        excluded = [t.strip() for t in args.exclude.split(",") if t.strip()] if args.exclude else []
        cfg = PutScalpConfig(
            target_pct=args.target,
            stop_pct=args.stop,
            max_hold_min=args.max_hold,
            premium_floor=args.premium_floor,
            premium_cap=args.premium_cap,
            entry_slots=slots,
            enable_dca=args.dca,
            dca_dip_pct=args.dca_dip,
            excluded_tickers=excluded,
            bear_mode_threshold=args.bear_threshold,
            bear_target_pct=args.bear_target,
        )
        results = run_backtest(conn, dates, cfg)
        print_results(results, cfg)

    conn.close()


if __name__ == "__main__":
    main()
