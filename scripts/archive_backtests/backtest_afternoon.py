"""Backtest afternoon models (1:00-3:00 PM ET) — CALLs + PUTs.

Runs afternoon_call and afternoon_put models, then combined with morning.

Usage:
    python scripts/backtest_afternoon.py                     # afternoon only
    python scripts/backtest_afternoon.py --combined          # morning + afternoon
    python scripts/backtest_afternoon.py --calls-only        # afternoon calls only
    python scripts/backtest_afternoon.py --puts-only         # afternoon puts only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
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
    GFV_BUFFER_PCT, DAILY_LOSS_CB_PCT,
    PREMIUM_CAP, SPREAD_GATE_PCT,
    MAX_POSITION_DOLLARS, MIN_PREMIUM_FLOOR, MAX_CONTRACTS, MAX_INDEX_CONCURRENT,
    MIN_ENTRY_SPACING_MIN, BAD_DAY_THRESHOLD, LOSS_COUNT_THRESHOLD,
    SCAN_START_MIN, SCAN_END_MIN, SCAN_INTERVAL,
    _V6_SETTINGS,
    load_models, compute_pattern_features, compute_entry_timing_features,
    simulate_exit, compute_regime_score,
)
from scripts.backtest_midday import compute_midday_features
from scripts.train_pattern_afternoon import compute_afternoon_features

# Afternoon scan window
AFT_SCAN_START = 210  # 1:00 PM ET
AFT_SCAN_END = 330    # 3:00 PM ET
AFT_SCAN_INTERVAL = 1

# Max same direction — limit correlated blowups
MAX_SAME_DIRECTION = 2
# Afternoon-specific: limit 1 PUT + 1 CALL at a time in afternoon
AFT_MAX_PER_DIRECTION = 1

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA"}


def load_afternoon_models():
    """Load afternoon CALL and PUT models."""
    models = {}
    for otype in ["call", "put"]:
        path = MODEL_DIR / f"pattern_afternoon_{otype}.txt"
        meta_path = MODEL_DIR / f"pattern_afternoon_{otype}_meta.json"
        if not path.exists():
            print(f"  WARNING: No afternoon {otype} model at {path}")
            continue
        model = lgb.Booster(model_file=str(path))
        with open(meta_path) as f:
            meta = json.load(f)
        models[otype] = {"model": model, "meta": meta}
        print(f"  Afternoon {otype}: AUC={meta['auc']:.4f}, threshold={meta['best_threshold']}, {len(meta['features'])} features")
    return models


def load_option_data(conn, ticker, date_str, right="CALL"):
    """Load option + stock data for a ticker-day."""
    atm = conn.execute("""
        SELECT oohlc.strike FROM option_ohlc oohlc
        JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
            AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
        WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right=?
            AND og.underlying_price > 0
        GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
    """, (ticker, date_str, right)).fetchone()
    if not atm:
        return None
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
        WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right=? AND oohlc.strike=?
        ORDER BY oohlc.timestamp
    """, (ticker, date_str, right, strike)).fetchall()

    if len(rows) < 30:
        return None

    data = {
        "strike": strike,
        "closes": np.array([float(r[0]) if r[0] else np.nan for r in rows]),
        "underlyings": np.array([float(r[1]) if r[1] else np.nan for r in rows]),
        "ivs": np.array([float(r[2]) if r[2] else np.nan for r in rows]),
        "bids": np.array([float(r[3]) if r[3] else 0 for r in rows]),
        "asks": np.array([float(r[4]) if r[4] else 0 for r in rows]),
        "deltas": np.array([float(r[5]) if r[5] else np.nan for r in rows]),
        "thetas": np.array([float(r[6]) if r[6] else np.nan for r in rows]),
        "volumes": np.array([float(r[7]) if r[7] else 0 for r in rows]),
        "expiry": rows[0][8] if rows else date_str,
        "vegas": np.array([float(r[9]) if r[9] else np.nan for r in rows]),
        "bid_sizes": np.array([float(r[10]) if r[10] else 0 for r in rows]),
        "ask_sizes": np.array([float(r[11]) if r[11] else 0 for r in rows]),
    }

    stock_rows = conn.execute("""
        SELECT close, high, low FROM stock_ohlc
        WHERE ticker=? AND date(timestamp)=?
        ORDER BY timestamp
    """, (ticker, date_str)).fetchall()
    data["stock_closes"] = np.array([float(r[0]) for r in stock_rows]) if stock_rows else np.array([])
    data["stock_highs"] = np.array([float(r[1]) for r in stock_rows]) if stock_rows else np.array([])
    data["stock_lows"] = np.array([float(r[2]) for r in stock_rows]) if stock_rows else np.array([])

    return data


def run_backtest(afternoon_models, entry_model, entry_features,
                 tickers, start_date, end_date,
                 regime_model=None, regime_threshold=0.0,
                 morning_model=None, morning_meta=None, morning_threshold=0.74,
                 entry_threshold=0.80,
                 enable_calls=True, enable_puts=True):
    """Run backtest with afternoon models, optionally combined with morning."""
    has_morning = morning_model is not None
    mode_parts = []
    if has_morning:
        mode_parts.append("morning CALLs")
    if enable_calls:
        mode_parts.append("afternoon CALLs")
    if enable_puts:
        mode_parts.append("afternoon PUTs")
    mode = " + ".join(mode_parts)

    conn = sqlite3.connect(THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' AND substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
        ORDER BY 1
    """, (start_date, end_date)).fetchall()]

    print(f"\n  Mode: {mode}")
    print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Portfolio: ${PORTFOLIO_START:,}")

    uw_conn = None
    UW_DB = str(PROJECT_DIR / "journal" / "uw_historical.db")
    if regime_model and Path(UW_DB).exists():
        uw_conn = sqlite3.connect(UW_DB)

    stock_data_cache = {}
    prev_days_cache = {}

    portfolio = PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades = []
    daily_pnls = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    exit_reasons = defaultdict(int)
    session_counts = defaultdict(int)
    equity_curve = [(dates[0], PORTFOLIO_START)]

    call_model_info = afternoon_models.get("call")
    put_model_info = afternoon_models.get("put")

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

        # Regime pre-filter
        if regime_model and regime_threshold > 0:
            regime_score = compute_regime_score(
                regime_model, "SPY", date_str, conn, uw_conn,
                stock_data_cache, prev_days_cache,
            )
            if regime_score < regime_threshold:
                equity_curve.append((date_str, round(portfolio, 2)))
                continue

        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            print(f"  [{day_idx+1}/{len(dates)}] {date_str}  portfolio=${portfolio:,.0f}  "
                  f"trades={len(trades)}  dd={max_dd:.1f}%", flush=True)

        for ticker in tickers:
            if day_cb:
                break
            if len(day_open_tickers) >= MAX_CONCURRENT:
                break

            # Skip if already traded this ticker today
            if ticker in day_open_tickers:
                continue

            # Index guard
            if ticker in INDEX_TICKERS:
                index_open = sum(1 for t in day_open_tickers if t in INDEX_TICKERS)
                if index_open >= MAX_INDEX_CONCURRENT:
                    continue

            # Build scan passes for this ticker
            scan_passes = []

            # Morning CALL pass
            if has_morning:
                call_data = load_option_data(conn, ticker, date_str, "CALL")
                if call_data and len(call_data["closes"]) >= 30:
                    opening_price = 0
                    for c in call_data["closes"][:5]:
                        if not np.isnan(c) and c > 0:
                            opening_price = c
                            break
                    if opening_price > 0:
                        scan_passes.append({
                            "label": "morning_call",
                            "data": call_data,
                            "direction": "call",
                            "scan_start": SCAN_START_MIN,
                            "scan_end": min(SCAN_END_MIN, len(call_data["closes"])),
                            "model": morning_model,
                            "features": morning_meta["features"],
                            "threshold": morning_threshold,
                            "ref_price": opening_price,
                            "feature_fn": "morning",
                        })

            # Afternoon CALL pass
            if enable_calls and call_model_info:
                call_data = load_option_data(conn, ticker, date_str, "CALL")
                if call_data and len(call_data["closes"]) >= AFT_SCAN_START:
                    ref = 0
                    for c in call_data["closes"][max(0, AFT_SCAN_START - 3):AFT_SCAN_START + 3]:
                        if not np.isnan(c) and c > 0:
                            ref = c
                            break
                    if ref > 0:
                        scan_passes.append({
                            "label": "afternoon_call",
                            "data": call_data,
                            "direction": "call",
                            "scan_start": AFT_SCAN_START,
                            "scan_end": min(AFT_SCAN_END, len(call_data["closes"])),
                            "model": call_model_info["model"],
                            "features": call_model_info["meta"]["features"],
                            "threshold": call_model_info["meta"]["best_threshold"],
                            "ref_price": ref,
                            "feature_fn": "afternoon",
                        })

            # Afternoon PUT pass
            if enable_puts and put_model_info:
                put_data = load_option_data(conn, ticker, date_str, "PUT")
                if put_data and len(put_data["closes"]) >= AFT_SCAN_START:
                    ref = 0
                    for c in put_data["closes"][max(0, AFT_SCAN_START - 3):AFT_SCAN_START + 3]:
                        if not np.isnan(c) and c > 0:
                            ref = c
                            break
                    if ref > 0:
                        scan_passes.append({
                            "label": "afternoon_put",
                            "data": put_data,
                            "direction": "put",
                            "scan_start": AFT_SCAN_START,
                            "scan_end": min(AFT_SCAN_END, len(put_data["closes"])),
                            "model": put_model_info["model"],
                            "features": put_model_info["meta"]["features"],
                            "threshold": put_model_info["meta"]["best_threshold"],
                            "ref_price": ref,
                            "feature_fn": "afternoon",
                        })

            entered_this_ticker = False
            for sp in scan_passes:
                if entered_this_ticker or day_cb:
                    break
                if len(day_open_tickers) >= MAX_CONCURRENT:
                    break

                d = sp["data"]
                closes = d["closes"]
                bids_arr = d["bids"]
                asks_arr = d["asks"]
                underlyings = d["underlyings"]
                ivs = d["ivs"]
                deltas_arr = d["deltas"]
                thetas_arr = d["thetas"]
                volumes_arr = d["volumes"]
                vegas_arr = d["vegas"]
                bid_sizes = d["bid_sizes"]
                ask_sizes = d["ask_sizes"]
                stock_closes = d["stock_closes"]
                stock_highs = d["stock_highs"]
                stock_lows = d["stock_lows"]
                expiry_date = d["expiry"]

                try:
                    exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    dte = max(0, (exp_dt - day_dt).days)
                except (ValueError, TypeError):
                    dte = 0

                for minute in range(sp["scan_start"], sp["scan_end"], AFT_SCAN_INTERVAL):
                    if entered_this_ticker or day_cb:
                        break
                    if len(day_open_tickers) >= MAX_CONCURRENT:
                        break
                    if minute - last_entry_minute < MIN_ENTRY_SPACING_MIN:
                        continue

                    # Direction limit
                    same_dir = sum(1 for d_dir in day_open_dirs if d_dir == sp["direction"])
                    if same_dir >= MAX_SAME_DIRECTION:
                        continue

                    # Feature computation
                    if sp["feature_fn"] == "morning":
                        feat = compute_pattern_features(
                            closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                            bids_arr, asks_arr, minute, sp["ref_price"],
                        )
                    else:
                        feat = compute_afternoon_features(
                            closes, volumes_arr, ivs, deltas_arr, thetas_arr, underlyings,
                            bids_arr, asks_arr, minute, sp["ref_price"], sp["direction"],
                        )
                    if feat is None:
                        continue

                    X = np.array([[feat.get(f, 0) for f in sp["features"]]], dtype=np.float32)
                    conf = sp["model"].predict(X)[0]
                    if conf < sp["threshold"]:
                        continue

                    # Entry timing filter (skip for PUTs — model trained on calls)
                    if entry_model and entry_features and sp["direction"] == "call":
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
                    day_open_dirs.append(sp["direction"])
                    entered_this_ticker = True
                    last_entry_minute = minute
                    session_counts[sp["label"]] += 1

                    per_ticker[ticker]["trades"] += 1
                    if trade_pnl > 0:
                        per_ticker[ticker]["wins"] += 1
                    per_ticker[ticker]["pnl"] += trade_pnl
                    exit_reasons[result["reason"]] += 1

                    trades.append({
                        "day": date_str, "ticker": ticker, "minute": minute,
                        "entry": entry_premium, "contracts": contracts,
                        "pnl": round(trade_pnl, 2), "reason": result["reason"],
                        "hold_min": result["hold_min"],
                        "peak_gain": round(result.get("peak_gain", 0), 1),
                        "pattern_conf": round(conf, 3),
                        "exit_prem": round(result.get("exit_prem", 0), 2),
                        "session": sp["label"],
                        "direction": sp["direction"],
                    })

                    if date_str not in daily_pnls:
                        daily_pnls[date_str] = 0
                    daily_pnls[date_str] += trade_pnl

                    day_realized += trade_pnl
                    if day_realized < 0 and abs(day_realized) / sod_balance * 100 >= DAILY_LOSS_CB_PCT:
                        day_cb = True

                    if result["reason"] in ("hard_stop", "confirmed_stop"):
                        day_hard_stops += 1
                    if trade_pnl <= 0:
                        day_losses += 1

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
        "trades": n_trades,
        "session_counts": dict(session_counts),
        "wins": wins,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / PORTFOLIO_START * 100, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_pnl_per_trade": round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
        "trading_days": len(dates),
        "per_ticker": dict(per_ticker),
        "exit_reasons": dict(exit_reasons),
        "trade_details": trades,
    }


def print_results(r):
    """Print formatted results."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {r['mode']}")
    print(f"{'=' * 70}")
    print(f"Trades:        {r['trades']} ({r['trades']/r['trading_days']:.2f}/day)")
    for sess, cnt in sorted(r["session_counts"].items()):
        print(f"  {sess}: {cnt}")
    print(f"Win Rate:      {r['win_rate']}%")
    print(f"Total P&L:     ${r['total_pnl']:+,.0f}")
    print(f"Return:        {r['return_pct']:+.1f}%")
    print(f"Profit Factor: {r['profit_factor']:.2f}")
    print(f"Sharpe:        {r['sharpe']:.2f}")
    print(f"Max Drawdown:  {r['max_drawdown_pct']:.1f}%")
    print(f"Avg Win:       ${r['avg_win']:+,.0f}")
    print(f"Avg Loss:      ${r['avg_loss']:+,.0f}")
    print(f"$/Trade:       ${r['avg_pnl_per_trade']:+,.0f}")

    # Per-direction breakdown
    for direction in ["call", "put"]:
        dir_trades = [t for t in r["trade_details"] if t.get("direction") == direction]
        if dir_trades:
            d_pnl = sum(t["pnl"] for t in dir_trades)
            d_wr = sum(1 for t in dir_trades if t["pnl"] > 0) / len(dir_trades) * 100
            print(f"\n  {direction.upper()}: {len(dir_trades)} trades, {d_wr:.0f}% WR, ${d_pnl:+,.0f}")

    # Per-ticker
    print(f"\n{'Ticker':<8} {'Trades':>6} {'WR%':>5} {'P&L':>10} {'Avg':>8}")
    print("-" * 40)
    for ticker in sorted(r["per_ticker"].keys(), key=lambda t: r["per_ticker"][t]["pnl"], reverse=True):
        t = r["per_ticker"][ticker]
        wr = t["wins"] / t["trades"] * 100 if t["trades"] > 0 else 0
        avg = t["pnl"] / t["trades"] if t["trades"] > 0 else 0
        print(f"{ticker:<8} {t['trades']:>6} {wr:>4.0f}% ${t['pnl']:>+9,.0f} ${avg:>+7,.0f}")

    # Trade log
    print(f"\n{'#':>3} {'Date':<12} {'Tkr':<6} {'Dir':<5} {'Sess':<16} {'Min':>4} {'Entry':>7} "
          f"{'Exit':>7} {'P&L':>9} {'Peak':>6} {'Reason':<18} {'Conf':>5}")
    print("-" * 110)
    for i, t in enumerate(r["trade_details"], 1):
        win = "W" if t["pnl"] > 0 else "L"
        print(f"{i:>3} {t['day']:<12} {t['ticker']:<6} {t.get('direction','?'):<5} "
              f"{t.get('session','?'):<16} {t['minute']:>4} ${t['entry']:>6.2f} ${t['exit_prem']:>6.2f} "
              f"${t['pnl']:>+8,.0f} +{t['peak_gain']:>4.0f}% {t['reason']:<18} {t['pattern_conf']:.2f} [{win}]")


def main():
    parser = argparse.ArgumentParser(description="Afternoon Model Backtest")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--combined", action="store_true", help="Include morning CALL model")
    parser.add_argument("--calls-only", action="store_true")
    parser.add_argument("--puts-only", action="store_true")
    parser.add_argument("--entry-threshold", type=float, default=0.80)
    parser.add_argument("--morning-threshold", type=float, default=0.74)
    parser.add_argument("--no-entry-filter", action="store_true")
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--regime-threshold", type=float, default=0.02)
    args = parser.parse_args()

    print("=" * 70)
    print("AFTERNOON MODEL BACKTEST (1:00 PM - 3:00 PM ET)")
    print("=" * 70)

    # Load afternoon models
    afternoon_models = load_afternoon_models()
    if not afternoon_models:
        print("ERROR: No afternoon models found")
        sys.exit(1)

    # Load shared models
    morning_model = None
    morning_meta = None
    if args.combined:
        pattern_model, pattern_meta, entry_model, entry_features, _, regime_model, _ = load_models(
            use_entry_filter=not args.no_entry_filter,
            use_regime=not args.no_regime,
        )
        morning_model = pattern_model
        morning_meta = pattern_meta
    else:
        _, _, entry_model, entry_features, _, regime_model, _ = load_models(
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

    enable_calls = not args.puts_only
    enable_puts = not args.calls_only

    # Run afternoon-only
    print(f"\n{'=' * 70}")
    print("RUN: AFTERNOON" + (" CALLs + PUTs" if enable_calls and enable_puts else
                               " CALLs ONLY" if enable_calls else " PUTs ONLY"))
    print(f"{'=' * 70}")
    r_aft = run_backtest(
        afternoon_models, entry_model, entry_features,
        tickers, start_date, end_date,
        regime_model, args.regime_threshold,
        entry_threshold=args.entry_threshold,
        enable_calls=enable_calls, enable_puts=enable_puts,
    )
    print_results(r_aft)

    if args.combined:
        print(f"\n{'=' * 70}")
        print("RUN: MORNING + AFTERNOON (FULL DAY)")
        print(f"{'=' * 70}")
        r_full = run_backtest(
            afternoon_models, entry_model, entry_features,
            tickers, start_date, end_date,
            regime_model, args.regime_threshold,
            morning_model=morning_model, morning_meta=morning_meta,
            morning_threshold=args.morning_threshold,
            entry_threshold=args.entry_threshold,
            enable_calls=enable_calls, enable_puts=enable_puts,
        )
        print_results(r_full)

        # Comparison
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        print(f"{'Metric':<20} {'Afternoon Only':>18} {'Full Day':>18}")
        print("-" * 58)
        print(f"{'Trades':<20} {r_aft['trades']:>18} {r_full['trades']:>18}")
        print(f"{'Trades/Day':<20} {r_aft['trades']/r_aft['trading_days']:>17.2f} {r_full['trades']/r_full['trading_days']:>17.2f}")
        print(f"{'Win Rate':<20} {r_aft['win_rate']:>17.1f}% {r_full['win_rate']:>17.1f}%")
        print(f"{'Total P&L':<20} ${r_aft['total_pnl']:>+16,.0f} ${r_full['total_pnl']:>+16,.0f}")
        print(f"{'Profit Factor':<20} {r_aft['profit_factor']:>18.2f} {r_full['profit_factor']:>18.2f}")
        print(f"{'Sharpe':<20} {r_aft['sharpe']:>18.2f} {r_full['sharpe']:>18.2f}")
        print(f"{'Max DD':<20} {r_aft['max_drawdown_pct']:>17.1f}% {r_full['max_drawdown_pct']:>17.1f}%")
        print(f"{'$/Trade':<20} ${r_aft['avg_pnl_per_trade']:>+16,.0f} ${r_full['avg_pnl_per_trade']:>+16,.0f}")


if __name__ == "__main__":
    main()
