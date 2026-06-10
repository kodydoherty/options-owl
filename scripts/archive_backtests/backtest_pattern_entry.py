"""Backtest the pattern-based entry model end-to-end.

Simulates continuous scanning (every minute 9:30-11:00) with the pattern model
deciding WHEN to enter, then V5 FSM handling the exit.

This is the full pipeline:
  1. For each trading day, scan each ticker every minute from 9:30-11:00
  2. At each minute, compute trailing features (5-10 candle window)
  3. If pattern model confidence > threshold → ENTER
  4. Run V5 FSM on subsequent ticks for exit simulation
  5. Track portfolio with position sizing, concurrent limits, etc.

Usage:
    python scripts/backtest_pattern_entry.py
    python scripts/backtest_pattern_entry.py --threshold 0.7
    python scripts/backtest_pattern_entry.py --ticker SPY
    python scripts/backtest_pattern_entry.py --sweep   # sweep thresholds
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# Exclude known losers
EXCLUDED_TICKERS = {"TSLA", "AAPL", "GOOGL", "MSFT"}

# Portfolio
PORTFOLIO_START = 23_000
MAX_CONCURRENT = 4
MAX_POSITION_PCT = 0.15
MAX_RISK_PCT = 0.75
GFV_BUFFER_PCT = 15.0
DAILY_LOSS_CB_PCT = 15.0
MAX_SAME_DIRECTION = 3
PREMIUM_CAP = 6.0
SPREAD_GATE_PCT = 15.0

# Scanning
SCAN_START_MIN = 0     # minutes after open to start scanning
SCAN_END_MIN = 90      # stop scanning at 11:00 (90 min after open)
SCAN_INTERVAL = 1      # check every minute

# V6 settings
_V6_SETTINGS = SimpleNamespace(
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
    V6_PREMIUM_CAP=PREMIUM_CAP,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=SPREAD_GATE_PCT,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_SIDEWAYS_SCALP=True,
    ENABLE_SCALP_TARGET=True,
    SCALP_TARGET_PCT=25.0,
    SCALP_RUNNER_CONFIRM_PCT=40.0,
)


# ── Model Loading ──────────────────────────────────────────────────────────


def load_pattern_model():
    """Load the pattern entry model."""
    model_path = MODEL_DIR / "pattern_entry.txt"
    meta_path = MODEL_DIR / "pattern_entry_meta.json"

    if not model_path.exists():
        print(f"ERROR: No pattern model at {model_path}")
        sys.exit(1)

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Loaded pattern_entry: AUC={meta['auc']:.4f}, threshold={meta['best_threshold']}")
    return model, meta


# ── Feature Computation (must match training exactly) ──────────────────────


def compute_trailing_features(closes, volumes, ivs, deltas, thetas, underlyings,
                               bids, asks, idx, opening_price):
    """Compute features from trailing candles at position idx."""
    if idx < 5:
        return None

    w5_start = max(0, idx - 5)
    w10_start = max(0, idx - 10)

    pre5 = closes[w5_start:idx]
    pre10 = closes[w10_start:idx]
    pre5_v = volumes[w5_start:idx]
    pre5_iv = ivs[w5_start:idx]
    pre5_u = underlyings[w5_start:idx]

    valid5 = pre5[~np.isnan(pre5)]
    valid10 = pre10[~np.isnan(pre10)]
    valid5_v = pre5_v[~np.isnan(pre5_v)]
    valid5_iv = pre5_iv[~np.isnan(pre5_iv)]
    valid5_u = pre5_u[~np.isnan(pre5_u)]

    if len(valid5) < 3 or valid5[0] <= 0:
        return None

    current = closes[idx]
    if np.isnan(current) or current <= 0:
        return None

    f = {}
    f["prem_slope_5"] = (valid5[-1] / valid5[0] - 1) * 100
    f["prem_slope_10"] = (valid10[-1] / valid10[0] - 1) * 100 if len(valid10) >= 5 and valid10[0] > 0 else f["prem_slope_5"]

    if len(valid5) >= 4:
        mid = len(valid5) // 2
        first_rate = (valid5[mid] / valid5[0] - 1) * 100 if valid5[0] > 0 else 0
        second_rate = (valid5[-1] / valid5[mid] - 1) * 100 if valid5[mid] > 0 else 0
        f["prem_accel"] = second_rate - first_rate
    else:
        f["prem_accel"] = 0

    last3 = valid5[-3:] if len(valid5) >= 3 else valid5
    f["prem_stabilizing"] = (max(last3) - min(last3)) / max(last3) * 100 if max(last3) > 0 else 0

    if len(valid5) >= 3 and all(c > 0 for c in valid5[:-1]):
        returns = np.diff(valid5) / valid5[:-1]
        f["prem_volatility"] = float(np.std(returns) * 100)
    else:
        f["prem_volatility"] = 0

    f["volume_avg_5"] = float(np.mean(valid5_v)) if len(valid5_v) > 0 else 0
    w20_start = max(0, idx - 20)
    vol20 = volumes[w20_start:idx]
    vol20_valid = vol20[~np.isnan(vol20)]
    avg20 = float(np.mean(vol20_valid)) if len(vol20_valid) > 0 else 1
    f["volume_ratio"] = f["volume_avg_5"] / max(avg20, 1)

    if len(valid5_v) >= 3:
        f["volume_trend"] = float(valid5_v[-1] / max(valid5_v[0], 1))
    else:
        f["volume_trend"] = 1.0

    if len(valid5_iv) >= 2:
        f["iv_change_5"] = float(valid5_iv[-1] - valid5_iv[0])
        f["iv_level"] = float(valid5_iv[-1])
    else:
        f["iv_change_5"] = 0
        f["iv_level"] = 0

    if len(valid5_u) >= 2 and valid5_u[0] > 0:
        f["und_slope_5"] = (valid5_u[-1] / valid5_u[0] - 1) * 100
    else:
        f["und_slope_5"] = 0

    f["drop_from_open"] = (current / opening_price - 1) * 100 if opening_price > 0 else 0

    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0
    f["minutes_since_open"] = idx
    f["premium"] = float(current)

    return f


# ── FSM Exit Simulation ───────────────────────────────────────────────────


def simulate_exit_from_arrays(closes, bids, asks, underlyings, entry_idx,
                                entry_premium, contracts, ticker, dte, expiry_date):
    """Run V5 FSM on remaining candles after entry."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0}

    tcfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(tcfg, settings=_V6_SETTINGS)

    entry_ts = datetime(2026, 1, 1, 9, 30)  # placeholder, adjusted by idx
    entry_ts = entry_ts + timedelta(minutes=entry_idx)

    underlying_0 = 0
    for i in range(entry_idx, min(entry_idx + 5, len(underlyings))):
        u = underlyings[i]
        if not np.isnan(u) and u > 0:
            underlying_0 = float(u)
            break

    option_type = "call"  # we're always doing ATM calls in this backtest
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=underlying_0,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts

    for idx in range(entry_idx + 1, len(closes)):
        prem = closes[idx]
        if np.isnan(prem) or prem <= 0:
            continue

        bid = float(bids[idx]) if idx < len(bids) and not np.isnan(bids[idx]) else prem
        ask = float(asks[idx]) if idx < len(asks) and not np.isnan(asks[idx]) else prem
        underlying = float(underlyings[idx]) if idx < len(underlyings) and not np.isnan(underlyings[idx]) else 0

        now = entry_ts + timedelta(minutes=(idx - entry_idx))
        minutes_to_close = max(0, (16 * 60) - (now.hour * 60 + now.minute))

        action = fsm.evaluate(
            state, prem, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
            candle_data={},
        )

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (prem - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            elapsed = idx - entry_idx
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (prem - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold_min": elapsed, "peak_gain": peak_gain,
                "exit_prem": prem,
            }

    # EOD
    last_valid = entry_premium
    for i in range(len(closes) - 1, entry_idx, -1):
        if not np.isnan(closes[i]) and closes[i] > 0:
            last_valid = closes[i]
            break
    elapsed = len(closes) - entry_idx
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_valid - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold_min": elapsed,
        "peak_gain": peak_gain, "exit_prem": last_valid,
    }


# ── Backtest Runner ────────────────────────────────────────────────────────


def run_backtest(model, meta, threshold: float, tickers: list[str],
                  start_date: str = "2025-09-18", end_date: str = "2026-05-21"):
    """Run full backtest with continuous scanning."""
    features = meta["features"]

    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    # Get trading days
    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' AND substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
        ORDER BY 1
    """, (start_date, end_date)).fetchall()]

    print(f"\n  Backtest: {len(dates)} trading days ({start_date} to {end_date})")
    print(f"  Threshold: {threshold}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Scan window: {SCAN_START_MIN}-{SCAN_END_MIN} min after open")

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades = []
    daily_pnls = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    signals_fired = 0
    signals_blocked = 0

    for day_idx, date_str in enumerate(dates):
        day_open_tickers = []
        day_open_dirs = []
        day_spent = 0.0
        day_realized = 0.0
        day_cb = False

        if (day_idx + 1) % 10 == 0:
            print(f"  Day {day_idx+1}/{len(dates)} ({date_str}) portfolio=${portfolio:,.0f}", flush=True)

        for ticker in tickers:
            if ticker in day_open_tickers:
                continue
            if day_cb:
                break

            # Load day data for this ticker
            atm = conn.execute("""
                SELECT oohlc.strike FROM option_ohlc oohlc
                JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL'
                    AND og.underlying_price > 0
                GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
            """, (ticker, date_str)).fetchone()
            if not atm:
                continue
            strike = atm[0]

            rows = conn.execute("""
                SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                       COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                       COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                       oohlc.volume, oohlc.expiration
                FROM option_ohlc oohlc
                LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                    AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
                LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
                ORDER BY oohlc.timestamp
            """, (ticker, date_str, strike)).fetchall()

            if len(rows) < 30:
                continue

            closes = np.array([float(r[0]) if r[0] else np.nan for r in rows])
            underlyings = np.array([float(r[1]) if r[1] else np.nan for r in rows])
            ivs = np.array([float(r[2]) if r[2] else np.nan for r in rows])
            bids_arr = np.array([float(r[3]) if r[3] else 0 for r in rows])
            asks_arr = np.array([float(r[4]) if r[4] else 0 for r in rows])
            deltas_arr = np.array([float(r[5]) if r[5] else np.nan for r in rows])
            thetas_arr = np.array([float(r[6]) if r[6] else np.nan for r in rows])
            volumes_arr = np.array([float(r[7]) if r[7] else 0 for r in rows])
            expiry_date = rows[0][8] if rows else date_str

            # Opening price
            opening_price = 0
            for c in closes[:5]:
                if not np.isnan(c) and c > 0:
                    opening_price = c
                    break
            if opening_price <= 0:
                continue

            # DTE
            try:
                exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                dte = max(0, (exp_dt - day_dt).days)
            except (ValueError, TypeError):
                dte = 0

            # Scan every minute in the window
            entered = False
            scan_end = min(SCAN_END_MIN, len(closes))

            for minute in range(max(SCAN_START_MIN, 5), scan_end, SCAN_INTERVAL):
                if entered or day_cb:
                    break

                feat = compute_trailing_features(
                    closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                    bids_arr, asks_arr, minute, opening_price,
                )
                if feat is None:
                    continue

                # Model prediction
                X = np.array([[feat.get(f, 0) for f in features]], dtype=np.float32)
                confidence = model.predict(X)[0]

                if confidence < threshold:
                    continue

                signals_fired += 1

                # Entry price = ask (realistic fill)
                entry_premium = float(asks_arr[minute]) if asks_arr[minute] > 0 else float(closes[minute])
                if entry_premium <= 0 or np.isnan(entry_premium):
                    continue

                # Premium cap
                if entry_premium > PREMIUM_CAP:
                    signals_blocked += 1
                    continue

                # Spread gate
                bid_val = float(bids_arr[minute]) if bids_arr[minute] > 0 else 0
                if bid_val > 0 and entry_premium > 0:
                    spread = (entry_premium - bid_val) / entry_premium * 100
                    if spread > SPREAD_GATE_PCT:
                        signals_blocked += 1
                        continue

                # Concurrent check
                if len(day_open_tickers) >= MAX_CONCURRENT:
                    signals_blocked += 1
                    continue

                # Direction (always CALL for ATM call scan)
                direction = "call"
                same_dir = sum(1 for d in day_open_dirs if d == direction)
                if same_dir >= MAX_SAME_DIRECTION:
                    signals_blocked += 1
                    continue

                # Position sizing
                deployable = portfolio * MAX_RISK_PCT
                per_slot = deployable / MAX_CONCURRENT
                position_cap = portfolio * MAX_POSITION_PCT
                cost_per = entry_premium * 100

                sod_balance = portfolio
                gfv_limit = sod_balance * (1 - GFV_BUFFER_PCT / 100)
                gfv_remaining = gfv_limit - day_spent
                if gfv_remaining < cost_per:
                    signals_blocked += 1
                    continue

                scaled = per_slot * 0.85  # flat budget
                raw_ct = int(scaled / cost_per) if cost_per > 0 else 1
                cap_ct = int(position_cap / cost_per) if cost_per > 0 else 1
                gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(raw_ct, cap_ct, gfv_ct))

                trade_cost = contracts * cost_per
                day_spent += trade_cost

                # Simulate exit
                result = simulate_exit_from_arrays(
                    closes, bids_arr, asks_arr, underlyings,
                    minute, entry_premium, contracts, ticker, dte, expiry_date,
                )

                trade_pnl = result["pnl"]
                portfolio += trade_pnl
                is_win = trade_pnl > 0

                day_open_tickers.append(ticker)
                day_open_dirs.append(direction)
                entered = True

                per_ticker[ticker]["trades"] += 1
                if is_win:
                    per_ticker[ticker]["wins"] += 1
                per_ticker[ticker]["pnl"] += trade_pnl

                trades.append({
                    "day": date_str, "ticker": ticker, "minute": minute,
                    "entry": entry_premium, "contracts": contracts,
                    "pnl": round(trade_pnl, 2), "reason": result["reason"],
                    "hold_min": result["hold_min"], "peak_gain": round(result.get("peak_gain", 0), 1),
                    "confidence": round(confidence, 3),
                })

                if date_str not in daily_pnls:
                    daily_pnls[date_str] = 0
                daily_pnls[date_str] += trade_pnl

                # Circuit breaker
                day_realized += trade_pnl
                if day_realized < 0:
                    loss_pct = abs(day_realized) / sod_balance * 100
                    if loss_pct >= DAILY_LOSS_CB_PCT:
                        day_cb = True

                # Drawdown
                if portfolio > peak_portfolio:
                    peak_portfolio = portfolio
                dd = (peak_portfolio - portfolio) / peak_portfolio * 100
                if dd > max_dd:
                    max_dd = dd

    conn.close()

    # Compute results
    total_pnl = portfolio - PORTFOLIO_START
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    pnl_list = [t["pnl"] for t in trades]
    wins_list = [p for p in pnl_list if p > 0]
    losses_list = [p for p in pnl_list if p <= 0]
    avg_win = np.mean(wins_list) if wins_list else 0
    avg_loss = np.mean(losses_list) if losses_list else 0
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_returns = list(daily_pnls.values())
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0

    # Entry minute distribution
    minutes = [t["minute"] for t in trades]
    avg_minute = np.mean(minutes) if minutes else 0

    return {
        "threshold": threshold,
        "trades": n_trades,
        "wins": wins,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_entry_minute": round(avg_minute, 1),
        "signals_fired": signals_fired,
        "signals_blocked": signals_blocked,
        "per_ticker": dict(per_ticker),
        "daily_pnls": daily_pnls,
        "trade_details": trades,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest pattern entry model")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Model threshold (default: from meta)")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds")
    parser.add_argument("--start", type=str, default="2025-09-18",
                        help="Start date (default: test set start)")
    parser.add_argument("--end", type=str, default="2026-05-21",
                        help="End date")
    parser.add_argument("--include-losers", action="store_true",
                        help="Include TSLA/AAPL/GOOGL/MSFT")
    args = parser.parse_args()

    print("Pattern Entry Model — End-to-End Backtest")
    print("=" * 70)

    model, meta = load_pattern_model()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.include_losers:
        tickers = TICKERS
    else:
        tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    if args.sweep:
        print("\nThreshold Sweep:")
        print(f"{'Thresh':<8} {'Trades':<7} {'WR%':<6} {'P&L':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgMin':>7}")
        print("-" * 60)

        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            r = run_backtest(model, meta, thresh, tickers, args.start, args.end)
            pnl_str = f"${r['total_pnl']:+,.0f}"
            print(f"{thresh:<8.2f} {r['trades']:<7} {r['win_rate']:<6.1f} {pnl_str:>10} "
                  f"{r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}% "
                  f"{r['avg_entry_minute']:>7.1f}")
    else:
        threshold = args.threshold or meta.get("best_threshold", 0.8)
        r = run_backtest(model, meta, threshold, tickers, args.start, args.end)

        print(f"\n{'=' * 70}")
        print(f"RESULTS (threshold={threshold})")
        print(f"{'=' * 70}")
        print(f"  Trades: {r['trades']}")
        print(f"  Win Rate: {r['win_rate']}%")
        print(f"  Total P&L: ${r['total_pnl']:+,.0f}")
        print(f"  Profit Factor: {r['profit_factor']}")
        print(f"  Sharpe: {r['sharpe']}")
        print(f"  Max Drawdown: {r['max_drawdown_pct']}%")
        print(f"  Avg Win: ${r['avg_win']:+,.0f}")
        print(f"  Avg Loss: ${r['avg_loss']:+,.0f}")
        print(f"  Avg Entry Minute: {r['avg_entry_minute']}")
        print(f"  Signals fired: {r['signals_fired']}")
        print(f"  Signals blocked: {r['signals_blocked']}")

        print(f"\n  Per-Ticker:")
        print(f"  {'Ticker':<8} {'Trades':<7} {'WR%':<6} {'P&L':>10}")
        print(f"  {'-'*35}")
        for ticker in sorted(r["per_ticker"].keys()):
            t = r["per_ticker"][ticker]
            wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
            print(f"  {ticker:<8} {t['trades']:<7} {wr:<6.0f} ${t['pnl']:>+9,.0f}")

        print(f"\n  Entry Minute Distribution:")
        minutes = [t["minute"] for t in r["trade_details"]]
        for bucket, lo, hi in [("5-15", 5, 15), ("15-30", 15, 30), ("30-45", 30, 45),
                                 ("45-60", 45, 60), ("60-75", 60, 75), ("75-90", 75, 90)]:
            n = sum(1 for m in minutes if lo <= m < hi)
            if n > 0:
                bucket_pnl = sum(t["pnl"] for t in r["trade_details"] if lo <= t["minute"] < hi)
                print(f"    {bucket}min: {n} trades, ${bucket_pnl:+,.0f}")

        # Save results
        results_path = PROJECT_DIR / "journal" / "v3_eval_results" / f"pattern_backtest_t{int(threshold*100)}.json"
        with open(results_path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
