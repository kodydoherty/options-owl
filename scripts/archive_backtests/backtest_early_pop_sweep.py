"""Fine-grained backstop sweep for early-pop trades.

Sweep backstop from 25% to 60% in 5% increments to find optimal.
Also sweep the detection params (peak_window, fade_threshold, check_at).

Usage:
    python scripts/backtest_early_pop_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

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
    ENABLE_V6_SIDEWAYS_SCALP=True,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000
INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals WHERE score >= 70 ORDER BY created_at
    """).fetchall()
    signals = []
    for r in rows:
        sig = dict(r)
        sig["premium"] = sig["atm_premium"] or sig["otm_premium"]
        sent = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        sig["option_type"] = "put" if sent in ("bearish", "put") else "call"
        if sig["premium"] and sig["premium"] > 0 and sig["strike"]:
            signals.append(sig)
    conn.close()
    return signals


def build_contract_ticker(ticker, expiry, strike, option_type):
    if not expiry:
        return ""
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        return ""
    exp_str = exp_dt.strftime("%y%m%d")
    ot = "C" if option_type.lower() in ("call", "bullish", "c") else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"


def load_ticks(harvester_conn, signal):
    ticker = signal["ticker"]
    strike = signal["strike"]
    created_at = signal["created_at"]
    option_type = signal["option_type"]
    sig_date = created_at[:10]
    sig_dt = datetime.strptime(sig_date, "%Y-%m-%d").date()
    candidates = [sig_dt]
    for delta in range(1, 6):
        d = sig_dt + timedelta(days=delta)
        if d.weekday() < 5:
            candidates.append(d)
            if len(candidates) >= 4:
                break
    for exp_date in candidates:
        expiry = exp_date.strftime("%Y-%m-%d")
        ct = build_contract_ticker(ticker, expiry, strike, option_type)
        if not ct:
            continue
        rows = harvester_conn.execute("""
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega, day_volume
            FROM harvest_snapshots
            WHERE contract_ticker = ? AND captured_at >= ?
            ORDER BY captured_at
        """, (ct, created_at)).fetchall()
        if rows and len(rows) >= 10:
            signal["_dte"] = (exp_date - sig_dt).days
            signal["_expiry_date"] = expiry
            break
    else:
        return None
    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
        "iv", "delta", "gamma", "theta", "vega", "volume"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def momentum_blocked(df, direction):
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    underlying_prices = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))
    if len(underlying_prices) < 5:
        return False
    first_half = underlying_prices[:len(underlying_prices) // 2]
    second_half = underlying_prices[len(underlying_prices) // 2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    pct_move = (avg_second - avg_first) / avg_first * 100
    prem_start = df["premium"].iloc[0]
    prem_5 = df["premium"].iloc[min(4, len(df) - 1)]
    prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0
    neg_signals = 0
    if is_call and pct_move < -0.05:
        neg_signals += 1
    elif not is_call and pct_move > 0.05:
        neg_signals += 1
    if prem_fade < -5:
        neg_signals += 1
    against = 0
    for i in range(max(0, window - 3), window):
        if i == 0:
            continue
        prev_u = df["underlying_price"].iloc[i - 1]
        cur_u = df["underlying_price"].iloc[i]
        if prev_u and cur_u:
            if is_call and cur_u < prev_u:
                against += 1
            elif not is_call and cur_u > prev_u:
                against += 1
    if against >= 3:
        neg_signals += 1
    return neg_signals >= 2


def detect_early_pop_realtime(df, entry_premium, direction, peak_window_min=10,
                               fade_threshold_pct=15, check_at_min=8,
                               min_peak_gain=5.0):
    if len(df) < 10 or entry_premium <= 0:
        return False
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    peak_prem = entry_premium
    peak_elapsed = 0.0
    current_prem = entry_premium
    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed = (ts - entry_ts).total_seconds() / 60
        if elapsed > check_at_min:
            break
        prem = df["premium"].iloc[i]
        if np.isnan(prem) or prem <= 0:
            continue
        current_prem = prem
        if prem > peak_prem:
            peak_prem = prem
            peak_elapsed = elapsed
    if peak_elapsed > peak_window_min:
        return False
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    if peak_gain < min_peak_gain:
        return False
    if peak_prem <= 0:
        return False
    fade = (peak_prem - current_prem) / peak_prem * 100
    return fade >= fade_threshold_pct


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker="SIM", cfg_override=None):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data"}
    cfg = cfg_override or get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    first_underlying = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_underlying = float(u)
            break
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )
    locked_pnl = 0.0
    remaining = contracts
    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue
        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium
        now = df["ts"].iloc[idx]
        if hasattr(now, 'to_pydatetime'):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)
        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value}
    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end"}


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    prepared = []
    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue
        premium = sig["premium"]
        if ticker not in INDEX_TICKERS:
            if score >= 150:
                cap = 9.0
            elif score >= 120:
                cap = 7.0
            else:
                cap = 6.0
            if premium > cap:
                continue
        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = premium
        deployable = PORTFOLIO * 0.75
        per_slot = deployable / 4
        position_cap = PORTFOLIO * 0.15
        score_mult = 0.25
        for threshold, mult in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                break
        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_contracts, pos_cap_contracts))
        if momentum_blocked(df, direction):
            continue
        prepared.append({
            "ticker": ticker, "direction": direction, "score": score,
            "premium": adj_entry, "contracts": contracts, "df": df,
            "dte": dte, "expiry_date": expiry_date, "day": sig["created_at"][:10],
        })

    harvester_conn.close()
    print(f"Prepared {len(prepared)} trades")

    # Baseline
    baseline_pnls = []
    for sig in prepared:
        r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                           sig["direction"], sig["dte"], sig["expiry_date"],
                           ticker=sig["ticker"])
        baseline_pnls.append(r["pnl"])
    baseline_total = sum(baseline_pnls)
    print(f"Baseline: ${baseline_total:,.2f}\n")

    # ── Sweep 1: Backstop % with fixed detection params ──────────────────────

    print("=" * 100)
    print("SWEEP 1: BACKSTOP % (detection: peak_window=10, fade=15%, check_at=8min)")
    print("=" * 100)

    backstop_range = list(range(25, 65, 5))

    print(f"\n{'BS%':>4} {'Detected':>8} {'Affected':>8} {'P&L':>12} {'Delta':>10} "
          f"{'Saved':>10} {'Cost':>10}")
    print("-" * 75)

    for bs_pct in backstop_range:
        total_pnl = 0
        detected = 0
        saved = 0
        cost = 0

        for i, sig in enumerate(prepared):
            is_ep = detect_early_pop_realtime(
                sig["df"], sig["premium"], sig["direction"],
                peak_window_min=10, fade_threshold_pct=15, check_at_min=8,
            )

            if is_ep:
                detected += 1
                cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                              backstop_0dte_pct=float(bs_pct),
                              backstop_multiday_pct=float(bs_pct + 15))
                r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                                   sig["direction"], sig["dte"], sig["expiry_date"],
                                   ticker=sig["ticker"], cfg_override=cfg)
                diff = r["pnl"] - baseline_pnls[i]
                if diff > 0:
                    saved += diff
                elif diff < 0:
                    cost += -diff
                total_pnl += r["pnl"]
            else:
                total_pnl += baseline_pnls[i]

        delta = total_pnl - baseline_total
        marker = " ***" if delta > 500 else (" **" if delta > 200 else (" *" if delta > 50 else ""))
        affected = int(saved > 0 or cost > 0)
        print(f"{bs_pct:>3}% {detected:>8} {'':>8} ${total_pnl:>10,.2f} ${delta:>+8,.2f} "
              f"${saved:>8,.2f} ${cost:>8,.2f}{marker}")

    # ── Sweep 2: Detection params with fixed backstop ────────────────────────

    print(f"\n\n{'=' * 100}")
    print("SWEEP 2: DETECTION PARAMS (backstop=40%)")
    print("=" * 100)

    detection_params = []
    for pw in [6, 8, 10, 12, 15]:
        for ft in [10, 12, 15, 18, 20, 25]:
            for cam in [6, 7, 8, 10, 12]:
                if cam > pw:
                    continue
                for mpg in [3, 5, 7, 10]:
                    detection_params.append((pw, ft, cam, mpg))

    print(f"\nTesting {len(detection_params)} detection param combos...")
    print(f"\n{'PeakWin':>7} {'Fade%':>5} {'Check':>5} {'MinPk':>5} {'Det':>4} "
          f"{'P&L':>12} {'Delta':>10} {'Saved':>10} {'Cost':>10}")
    print("-" * 85)

    results = []
    for pw, ft, cam, mpg in detection_params:
        total_pnl = 0
        detected = 0
        saved = 0
        cost = 0

        for i, sig in enumerate(prepared):
            is_ep = detect_early_pop_realtime(
                sig["df"], sig["premium"], sig["direction"],
                peak_window_min=pw, fade_threshold_pct=ft,
                check_at_min=cam, min_peak_gain=mpg,
            )

            if is_ep:
                detected += 1
                cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                              backstop_0dte_pct=40.0,
                              backstop_multiday_pct=55.0)
                r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                                   sig["direction"], sig["dte"], sig["expiry_date"],
                                   ticker=sig["ticker"], cfg_override=cfg)
                diff = r["pnl"] - baseline_pnls[i]
                if diff > 0:
                    saved += diff
                elif diff < 0:
                    cost += -diff
                total_pnl += r["pnl"]
            else:
                total_pnl += baseline_pnls[i]

        delta = total_pnl - baseline_total
        results.append((pw, ft, cam, mpg, detected, total_pnl, delta, saved, cost))

    # Sort by delta descending
    results.sort(key=lambda x: x[6], reverse=True)

    # Show top 30
    for pw, ft, cam, mpg, det, pnl, delta, saved, cost in results[:30]:
        marker = " ***" if delta > 500 else (" **" if delta > 200 else (" *" if delta > 50 else ""))
        print(f"{pw:>7} {ft:>5} {cam:>5} {mpg:>5} {det:>4} "
              f"${pnl:>10,.2f} ${delta:>+8,.2f} ${saved:>8,.2f} ${cost:>8,.2f}{marker}")

    # ── Sweep 3: Best detection + every backstop ─────────────────────────────

    best_pw, best_ft, best_cam, best_mpg = results[0][0], results[0][1], results[0][2], results[0][3]

    print(f"\n\n{'=' * 100}")
    print(f"SWEEP 3: BACKSTOP % with BEST detection (pw={best_pw}, fade={best_ft}%, "
          f"check={best_cam}, min_peak={best_mpg}%)")
    print("=" * 100)

    print(f"\n{'BS%':>4} {'Det':>4} {'P&L':>12} {'Delta':>10} {'Saved':>10} {'Cost':>10}")
    print("-" * 60)

    best_overall_delta = -999999
    best_overall_bs = 0

    for bs_pct in range(25, 65, 5):
        total_pnl = 0
        detected = 0
        saved = 0
        cost = 0

        for i, sig in enumerate(prepared):
            is_ep = detect_early_pop_realtime(
                sig["df"], sig["premium"], sig["direction"],
                peak_window_min=best_pw, fade_threshold_pct=best_ft,
                check_at_min=best_cam, min_peak_gain=best_mpg,
            )

            if is_ep:
                detected += 1
                cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                              backstop_0dte_pct=float(bs_pct),
                              backstop_multiday_pct=float(bs_pct + 15))
                r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                                   sig["direction"], sig["dte"], sig["expiry_date"],
                                   ticker=sig["ticker"], cfg_override=cfg)
                diff = r["pnl"] - baseline_pnls[i]
                if diff > 0:
                    saved += diff
                elif diff < 0:
                    cost += -diff
                total_pnl += r["pnl"]
            else:
                total_pnl += baseline_pnls[i]

        delta = total_pnl - baseline_total
        marker = " ***" if delta > 500 else (" **" if delta > 200 else "")
        print(f"{bs_pct:>3}% {detected:>4} ${total_pnl:>10,.2f} ${delta:>+8,.2f} "
              f"${saved:>8,.2f} ${cost:>8,.2f}{marker}")

        if delta > best_overall_delta:
            best_overall_delta = delta
            best_overall_bs = bs_pct

    print(f"\n  BEST OVERALL: backstop={best_overall_bs}% with detection "
          f"(pw={best_pw}, fade={best_ft}%, check={best_cam}, min_peak={best_mpg}%) "
          f"→ ${best_overall_delta:+,.2f}")

    # Show which trades are affected at the best config
    print(f"\n\nAFFECTED TRADES at optimal config:")
    print(f"{'Day':<12} {'Ticker':<7} {'Entry':>6} {'Ct':>3} "
          f"{'Base P&L':>10} {'New P&L':>10} {'Delta':>9}")
    print("-" * 65)

    for i, sig in enumerate(prepared):
        is_ep = detect_early_pop_realtime(
            sig["df"], sig["premium"], sig["direction"],
            peak_window_min=best_pw, fade_threshold_pct=best_ft,
            check_at_min=best_cam, min_peak_gain=best_mpg,
        )
        if not is_ep:
            continue

        cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                      backstop_0dte_pct=float(best_overall_bs),
                      backstop_multiday_pct=float(best_overall_bs + 15))
        r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                           sig["direction"], sig["dte"], sig["expiry_date"],
                           ticker=sig["ticker"], cfg_override=cfg)

        diff = r["pnl"] - baseline_pnls[i]
        marker = " SAVED" if diff > 10 else (" COST" if diff < -10 else "")
        print(f"{sig['day']:<12} {sig['ticker']:<7} ${sig['premium']:>5.2f} "
              f"{sig['contracts']:>3} ${baseline_pnls[i]:>8,.2f} ${r['pnl']:>8,.2f} "
              f"${diff:>+7,.2f}{marker}")


if __name__ == "__main__":
    main()
