"""Backtest midday model (11:00 AM - 1:00 PM ET) standalone + combined with morning.

Runs the midday pattern_midday model over the same ThetaData DB, using the same
V5 FSM exits and portfolio simulation as backtest_gold_standard.py.

Usage:
    python scripts/backtest_midday.py                    # midday-only, last 60 days
    python scripts/backtest_midday.py --combined         # morning + midday combined
    python scripts/backtest_midday.py --days 90           # last 90 days
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.backtest_gold_standard import (
    THETADATA_DB, MODEL_DIR, ET,
    TICKERS, EXCLUDED_TICKERS,
    PORTFOLIO_START, MAX_CONCURRENT, MAX_POSITION_PCT, MAX_RISK_PCT,
    GFV_BUFFER_PCT, DAILY_LOSS_CB_PCT, MAX_SAME_DIRECTION,
    PREMIUM_CAP, SPREAD_GATE_PCT,
    MAX_POSITION_DOLLARS, MIN_PREMIUM_FLOOR, MAX_CONTRACTS, MAX_INDEX_CONCURRENT,
    MIN_ENTRY_SPACING_MIN, BAD_DAY_THRESHOLD, LOSS_COUNT_THRESHOLD,
    ENABLE_AFTERNOON_DANGER, ENABLE_HARD_CUTOFF,
    AFTERNOON_DANGER_START, AFTERNOON_DANGER_END, HARD_CUTOFF_MIN,
    _V6_SETTINGS,
    SCAN_START_MIN, SCAN_END_MIN, SCAN_INTERVAL,
    load_models, compute_pattern_features, compute_entry_timing_features,
    simulate_exit, compute_regime_score,
)

# Midday scan window (minutes after open)
MIDDAY_SCAN_START = 90   # 11:00 AM ET
MIDDAY_SCAN_END = 210    # 1:00 PM ET
MIDDAY_SCAN_INTERVAL = 1
MIDDAY_PATTERN_THRESHOLD = 0.75  # From training best_threshold

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA"}


def compute_midday_features(closes, volumes, ivs, deltas, thetas, underlyings,
                             bids, asks, idx, reference_price):
    """Compute midday pattern features — must match train_pattern_midday.py."""
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

    f["drop_from_open"] = (current / reference_price - 1) * 100 if reference_price > 0 else 0

    bid = bids[idx] if idx < len(bids) else 0
    ask = asks[idx] if idx < len(asks) else 0
    f["spread_pct"] = (ask - bid) / ask * 100 if ask > 0 and bid >= 0 else 0
    f["delta"] = float(deltas[idx]) if idx < len(deltas) and not np.isnan(deltas[idx]) else 0
    f["theta"] = float(thetas[idx]) if idx < len(thetas) and not np.isnan(thetas[idx]) else 0
    f["minutes_since_open"] = idx
    f["premium"] = float(current)

    # Midday-specific features
    MIDDAY_START_MIN = 90
    if idx >= MIDDAY_START_MIN and closes[0] > 0 and not np.isnan(closes[0]):
        morning_slice = closes[:MIDDAY_START_MIN]
        valid_morning = morning_slice[~np.isnan(morning_slice) & (morning_slice > 0)]
        if len(valid_morning) >= 10:
            f["morning_range_pct"] = (float(np.max(valid_morning)) - float(np.min(valid_morning))) / float(np.max(valid_morning)) * 100
            f["morning_return_pct"] = (float(valid_morning[-1]) / float(valid_morning[0]) - 1) * 100
            f["prem_vs_session_low"] = (current / float(np.min(valid_morning)) - 1) * 100
        else:
            f["morning_range_pct"] = 0
            f["morning_return_pct"] = 0
            f["prem_vs_session_low"] = 0
    else:
        f["morning_range_pct"] = 0
        f["morning_return_pct"] = 0
        f["prem_vs_session_low"] = 0

    if idx >= MIDDAY_START_MIN:
        u_morning = underlyings[:MIDDAY_START_MIN]
        valid_u_morning = u_morning[~np.isnan(u_morning) & (u_morning > 0)]
        if len(valid_u_morning) >= 10:
            f["und_morning_return"] = (float(valid_u_morning[-1]) / float(valid_u_morning[0]) - 1) * 100
        else:
            f["und_morning_return"] = 0
    else:
        f["und_morning_return"] = 0

    return f


def run_midday_backtest(midday_model, midday_meta, entry_model, entry_features,
                         midday_threshold, entry_threshold,
                         tickers, start_date, end_date,
                         stop_model=None, regime_model=None, regime_threshold=0.0,
                         signal_model=None,
                         morning_model=None, morning_meta=None,
                         morning_threshold=0.74):
    """Run backtest with midday model, optionally combined with morning model."""
    combined = morning_model is not None
    mode = "COMBINED (morning + midday)" if combined else "MIDDAY ONLY"

    m_features = midday_meta["features"]
    if combined:
        p_features = morning_meta["features"]

    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' AND substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
        ORDER BY 1
    """, (start_date, end_date)).fetchall()]

    print(f"\n  Mode: {mode}")
    print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    if combined:
        print(f"  Morning: pattern={morning_threshold}, scan 5-90 min")
    print(f"  Midday: pattern={midday_threshold}, scan {MIDDAY_SCAN_START}-{MIDDAY_SCAN_END} min")
    print(f"  Entry filter: {'ON (t=' + str(entry_threshold) + ')' if entry_model else 'OFF'}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Portfolio: ${PORTFOLIO_START:,}")

    # UW DB for regime
    uw_conn = None
    UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")
    if regime_model and Path(UW_DB).exists():
        uw_conn = sqlite3.connect(UW_DB)

    stock_data_cache = {}
    prev_days_cache = {}
    regime_skipped_days = 0

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades = []
    daily_pnls = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "total_contracts": 0})
    exit_reasons = defaultdict(int)
    morning_trades = 0
    midday_trades = 0
    equity_curve = [(dates[0], PORTFOLIO_START)]

    for day_idx, date_str in enumerate(dates):
        day_open_tickers = []
        day_open_dirs = []
        day_spent = 0.0
        day_realized = 0.0
        day_cb = False
        sod_balance = portfolio
        last_entry_minute = -999
        day_losses = 0
        day_hard_stops = 0
        day_pattern_threshold_morning = morning_threshold if combined else 999
        day_pattern_threshold_midday = midday_threshold

        # Regime pre-filter
        if regime_model and regime_threshold > 0:
            regime_score = compute_regime_score(
                regime_model, "SPY", date_str, conn, uw_conn,
                stock_data_cache, prev_days_cache,
            )
            if regime_score < regime_threshold:
                regime_skipped_days += 1
                equity_curve.append((date_str, round(portfolio, 2)))
                continue

        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            print(f"  [{day_idx+1}/{len(dates)}] {date_str}  portfolio=${portfolio:,.0f}  "
                  f"trades={len(trades)}  dd={max_dd:.1f}%", flush=True)

        for ticker in tickers:
            if ticker in day_open_tickers:
                continue
            if day_cb:
                break
            if len(day_open_tickers) >= MAX_CONCURRENT:
                break
            if ticker in INDEX_TICKERS:
                index_open = sum(1 for t in day_open_tickers if t in INDEX_TICKERS)
                if index_open >= MAX_INDEX_CONCURRENT:
                    continue

            # Load day data (same query as gold standard)
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
                       oohlc.volume, oohlc.expiration,
                       COALESCE(og.vega, 0),
                       COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0)
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
            vegas_arr = np.array([float(r[9]) if r[9] else np.nan for r in rows])
            bid_sizes = np.array([float(r[10]) if r[10] else 0 for r in rows])
            ask_sizes = np.array([float(r[11]) if r[11] else 0 for r in rows])

            stock_rows = conn.execute("""
                SELECT close, high, low FROM stock_ohlc
                WHERE ticker=? AND date(timestamp)=?
                ORDER BY timestamp
            """, (ticker, date_str)).fetchall()
            stock_closes = np.array([float(r[0]) for r in stock_rows]) if stock_rows else np.array([])
            stock_highs = np.array([float(r[1]) for r in stock_rows]) if stock_rows else np.array([])
            stock_lows = np.array([float(r[2]) for r in stock_rows]) if stock_rows else np.array([])

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

            entered = False

            # Build scan ranges: morning (if combined) then midday
            scan_ranges = []
            if combined:
                scan_ranges.append(("morning", 5, min(90, len(closes)), morning_model, p_features, opening_price, day_pattern_threshold_morning))
            # Midday reference price = premium at minute 90
            midday_ref = 0
            ref_start = max(0, 87)
            ref_end = min(93, len(closes))
            for c in closes[ref_start:ref_end]:
                if not np.isnan(c) and c > 0:
                    midday_ref = c
                    break
            if midday_ref <= 0:
                midday_ref = opening_price  # fallback
            scan_ranges.append(("midday", MIDDAY_SCAN_START, min(MIDDAY_SCAN_END, len(closes)), midday_model, m_features, midday_ref, day_pattern_threshold_midday))

            for scan_label, scan_start, scan_end, model, features, ref_price, threshold in scan_ranges:
                if entered or day_cb:
                    break
                if len(day_open_tickers) >= MAX_CONCURRENT:
                    break

                for minute in range(scan_start, scan_end, SCAN_INTERVAL if scan_label == "morning" else MIDDAY_SCAN_INTERVAL):
                    if entered or day_cb:
                        break
                    if len(day_open_tickers) >= MAX_CONCURRENT:
                        break

                    if minute - last_entry_minute < MIN_ENTRY_SPACING_MIN:
                        continue

                    # Afternoon danger zone
                    if ENABLE_AFTERNOON_DANGER and AFTERNOON_DANGER_START <= minute <= AFTERNOON_DANGER_END:
                        continue
                    if ENABLE_HARD_CUTOFF and minute >= HARD_CUTOFF_MIN:
                        continue

                    # Pattern model
                    if scan_label == "morning":
                        feat = compute_pattern_features(
                            closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                            bids_arr, asks_arr, minute, ref_price,
                        )
                    else:
                        feat = compute_midday_features(
                            closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                            bids_arr, asks_arr, minute, ref_price,
                        )
                    if feat is None:
                        continue

                    X = np.array([[feat.get(f, 0) for f in features]], dtype=np.float32)
                    conf = model.predict(X)[0]

                    if conf < threshold:
                        continue

                    # Entry timing filter
                    if entry_model and entry_features:
                        et_feat = compute_entry_timing_features(
                            closes, volumes_arr, bids_arr, asks_arr, bid_sizes,
                            ask_sizes, ivs, deltas_arr, thetas_arr, vegas_arr,
                            underlyings, stock_closes, stock_highs, stock_lows,
                            minute, entry_features,
                        )
                        if et_feat is not None:
                            X_entry = np.array([[et_feat.get(f, 0) for f in entry_features]], dtype=np.float32)
                            entry_conf = entry_model.predict(X_entry)[0]
                            if entry_conf < entry_threshold:
                                continue

                    # Entry gates
                    entry_premium = float(asks_arr[minute]) if asks_arr[minute] > 0 else float(closes[minute])
                    if entry_premium <= 0 or np.isnan(entry_premium):
                        continue
                    if entry_premium < MIN_PREMIUM_FLOOR:
                        continue
                    if entry_premium > PREMIUM_CAP:
                        continue

                    bid_val = float(bids_arr[minute]) if bids_arr[minute] > 0 else 0
                    if bid_val > 0 and entry_premium > 0:
                        spread = (entry_premium - bid_val) / entry_premium * 100
                        if spread > SPREAD_GATE_PCT:
                            continue

                    direction = "call"
                    same_dir = sum(1 for d in day_open_dirs if d == direction)
                    if same_dir >= MAX_SAME_DIRECTION:
                        continue

                    # Position sizing
                    deployable = portfolio * MAX_RISK_PCT
                    per_slot = deployable / MAX_CONCURRENT
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

                    trade_cost = contracts * cost_per
                    day_spent += trade_cost

                    # V5 FSM exit
                    result = simulate_exit(
                        closes, bids_arr, asks_arr, underlyings,
                        minute, entry_premium, contracts, ticker, dte, expiry_date,
                    )

                    trade_pnl = result["pnl"]
                    portfolio += trade_pnl

                    day_open_tickers.append(ticker)
                    day_open_dirs.append(direction)
                    entered = True
                    last_entry_minute = minute

                    if scan_label == "morning":
                        morning_trades += 1
                    else:
                        midday_trades += 1

                    per_ticker[ticker]["trades"] += 1
                    if trade_pnl > 0:
                        per_ticker[ticker]["wins"] += 1
                    per_ticker[ticker]["pnl"] += trade_pnl
                    per_ticker[ticker]["total_contracts"] += contracts
                    exit_reasons[result["reason"]] += 1

                    trades.append({
                        "day": date_str, "ticker": ticker, "minute": minute,
                        "entry": entry_premium, "contracts": contracts,
                        "pnl": round(trade_pnl, 2), "reason": result["reason"],
                        "hold_min": result["hold_min"],
                        "peak_gain": round(result.get("peak_gain", 0), 1),
                        "pattern_conf": round(conf, 3),
                        "exit_prem": round(result.get("exit_prem", 0), 2),
                        "session": scan_label,
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

                    # Bad day mode
                    if result["reason"] in ("hard_stop", "confirmed_stop"):
                        day_hard_stops += 1
                    if trade_pnl <= 0:
                        day_losses += 1
                    if day_losses >= LOSS_COUNT_THRESHOLD or day_hard_stops >= 1:
                        day_pattern_threshold_morning = max(day_pattern_threshold_morning, BAD_DAY_THRESHOLD)
                        day_pattern_threshold_midday = max(day_pattern_threshold_midday, BAD_DAY_THRESHOLD)

                    # Drawdown
                    if portfolio > peak_portfolio:
                        peak_portfolio = portfolio
                    dd = (peak_portfolio - portfolio) / peak_portfolio * 100
                    if dd > max_dd:
                        max_dd = dd

        equity_curve.append((date_str, round(portfolio, 2)))

    conn.close()
    if uw_conn:
        uw_conn.close()

    # Results
    total_pnl = portfolio - PORTFOLIO_START
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses_n = sum(1 for t in trades if t["pnl"] <= 0)
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

    return {
        "mode": mode,
        "period": f"{dates[0]} to {dates[-1]}",
        "trading_days": len(dates),
        "trades": n_trades,
        "morning_trades": morning_trades,
        "midday_trades": midday_trades,
        "wins": wins,
        "losses": losses_n,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_portfolio": round(portfolio, 2),
        "return_pct": round(total_pnl / PORTFOLIO_START * 100, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_pnl_per_trade": round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
        "regime_skipped_days": regime_skipped_days,
        "per_ticker": dict(per_ticker),
        "exit_reasons": dict(exit_reasons),
        "trade_details": trades,
        "equity_curve": equity_curve,
    }


def print_results(r):
    """Print formatted results."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {r['mode']}")
    print(f"{'=' * 70}")
    print(f"Period:        {r['period']} ({r['trading_days']} days)")
    print(f"Trades:        {r['trades']} (morning={r['morning_trades']}, midday={r['midday_trades']})")
    print(f"Trades/Day:    {r['trades'] / r['trading_days']:.2f}")
    print(f"Win Rate:      {r['win_rate']}%")
    print(f"Total P&L:     ${r['total_pnl']:+,.0f}")
    print(f"Return:        {r['return_pct']:+.1f}%")
    print(f"Profit Factor: {r['profit_factor']:.2f}")
    print(f"Sharpe:        {r['sharpe']:.2f}")
    print(f"Max Drawdown:  {r['max_drawdown_pct']:.1f}%")
    print(f"Avg Win:       ${r['avg_win']:+,.0f}")
    print(f"Avg Loss:      ${r['avg_loss']:+,.0f}")
    print(f"$/Trade:       ${r['avg_pnl_per_trade']:+,.0f}")

    # Per-session breakdown
    morning_trades = [t for t in r["trade_details"] if t.get("session") == "morning"]
    midday_trades = [t for t in r["trade_details"] if t.get("session") == "midday"]

    if morning_trades:
        m_pnl = sum(t["pnl"] for t in morning_trades)
        m_wr = sum(1 for t in morning_trades if t["pnl"] > 0) / len(morning_trades) * 100
        print(f"\n  Morning:  {len(morning_trades)} trades, {m_wr:.0f}% WR, ${m_pnl:+,.0f}")
    if midday_trades:
        d_pnl = sum(t["pnl"] for t in midday_trades)
        d_wr = sum(1 for t in midday_trades if t["pnl"] > 0) / len(midday_trades) * 100
        print(f"  Midday:   {len(midday_trades)} trades, {d_wr:.0f}% WR, ${d_pnl:+,.0f}")

    # Per-ticker
    print(f"\n{'Ticker':<8} {'Trades':>6} {'WR%':>5} {'P&L':>10} {'Avg':>8}")
    print("-" * 40)
    for ticker in sorted(r["per_ticker"].keys(), key=lambda t: r["per_ticker"][t]["pnl"], reverse=True):
        t = r["per_ticker"][ticker]
        wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
        avg = t["pnl"] / t["trades"] if t["trades"] > 0 else 0
        print(f"{ticker:<8} {t['trades']:>6} {wr:>4.0f}% ${t['pnl']:>+9,.0f} ${avg:>+7,.0f}")

    # Trade log
    print(f"\n{'#':>3} {'Date':<12} {'Ticker':<7} {'Sess':<8} {'Min':>4} {'Entry':>7} {'Exit':>7} "
          f"{'P&L':>9} {'Peak':>6} {'Reason':<20} {'Conf':>5}")
    print("-" * 100)
    for i, t in enumerate(r["trade_details"], 1):
        win = "W" if t["pnl"] > 0 else "L"
        print(f"{i:>3} {t['day']:<12} {t['ticker']:<7} {t.get('session','?'):<8} {t['minute']:>4} "
              f"${t['entry']:>6.2f} ${t['exit_prem']:>6.2f} ${t['pnl']:>+8,.0f} "
              f"+{t['peak_gain']:>4.0f}% {t['reason']:<20} {t['pattern_conf']:.2f} [{win}]")


def main():
    parser = argparse.ArgumentParser(description="Midday Model Backtest")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--combined", action="store_true", help="Run morning + midday combined")
    parser.add_argument("--midday-threshold", type=float, default=MIDDAY_PATTERN_THRESHOLD)
    parser.add_argument("--morning-threshold", type=float, default=0.74)
    parser.add_argument("--entry-threshold", type=float, default=0.80)
    parser.add_argument("--no-entry-filter", action="store_true")
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--regime-threshold", type=float, default=0.02)
    args = parser.parse_args()

    print("=" * 70)
    print("MIDDAY MODEL BACKTEST")
    print("=" * 70)

    # Load midday model
    midday_path = MODEL_DIR / "pattern_midday.txt"
    midday_meta_path = MODEL_DIR / "pattern_midday_meta.json"
    if not midday_path.exists():
        print(f"ERROR: No midday model at {midday_path}")
        sys.exit(1)
    midday_model = lgb.Booster(model_file=str(midday_path))
    with open(midday_meta_path) as f:
        midday_meta = json.load(f)
    print(f"  Midday model: AUC={midday_meta['auc']:.4f}, {len(midday_meta['features'])} features")

    # Load shared models (entry timing, regime, etc)
    morning_model_obj = None
    morning_meta_obj = None
    if args.combined:
        pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model, signal_model = load_models(
            use_entry_filter=not args.no_entry_filter,
            use_regime=not args.no_regime,
        )
        morning_model_obj = pattern_model
        morning_meta_obj = pattern_meta
    else:
        _, _, entry_model, entry_features, stop_model, regime_model, signal_model = load_models(
            use_entry_filter=not args.no_entry_filter,
            use_regime=not args.no_regime,
        )

    # Date range
    conn = sqlite3.connect(THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1 DESC
    """).fetchall()]
    conn.close()

    start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]
    end_date = all_dates[0]

    tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    # Run midday-only first
    print("\n" + "=" * 70)
    print("RUN 1: MIDDAY ONLY")
    print("=" * 70)
    r_midday = run_midday_backtest(
        midday_model, midday_meta, entry_model, entry_features,
        args.midday_threshold, args.entry_threshold,
        tickers, start_date, end_date,
        stop_model, regime_model, args.regime_threshold,
        signal_model,
    )
    print_results(r_midday)

    if args.combined:
        print("\n" + "=" * 70)
        print("RUN 2: COMBINED (morning + midday)")
        print("=" * 70)
        r_combined = run_midday_backtest(
            midday_model, midday_meta, entry_model, entry_features,
            args.midday_threshold, args.entry_threshold,
            tickers, start_date, end_date,
            stop_model, regime_model, args.regime_threshold,
            signal_model,
            morning_model=morning_model_obj, morning_meta=morning_meta_obj,
            morning_threshold=args.morning_threshold,
        )
        print_results(r_combined)

        # Comparison
        print(f"\n{'=' * 70}")
        print("COMPARISON: MIDDAY-ONLY vs COMBINED")
        print(f"{'=' * 70}")
        print(f"{'Metric':<20} {'Midday Only':>18} {'Combined':>18}")
        print("-" * 58)
        print(f"{'Trades':<20} {r_midday['trades']:>18} {r_combined['trades']:>18}")
        print(f"{'Trades/Day':<20} {r_midday['trades']/r_midday['trading_days']:>17.2f} {r_combined['trades']/r_combined['trading_days']:>17.2f}")
        print(f"{'Win Rate':<20} {r_midday['win_rate']:>17.1f}% {r_combined['win_rate']:>17.1f}%")
        print(f"{'Total P&L':<20} ${r_midday['total_pnl']:>+16,.0f} ${r_combined['total_pnl']:>+16,.0f}")
        print(f"{'Profit Factor':<20} {r_midday['profit_factor']:>18.2f} {r_combined['profit_factor']:>18.2f}")
        print(f"{'Sharpe':<20} {r_midday['sharpe']:>18.2f} {r_combined['sharpe']:>18.2f}")
        print(f"{'Max DD':<20} {r_midday['max_drawdown_pct']:>17.1f}% {r_combined['max_drawdown_pct']:>17.1f}%")
        print(f"{'$/Trade':<20} ${r_midday['avg_pnl_per_trade']:>+16,.0f} ${r_combined['avg_pnl_per_trade']:>+16,.0f}")


if __name__ == "__main__":
    main()
