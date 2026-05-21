"""Backtest sideways exit strategies: heuristic vs ML-assisted.

Tests strategies to capture small wins on trades that go +5-20% then chop sideways:

  1) BASELINE — current FSM (no sideways detection)
  2) HEURISTIC SIDEWAYS — exit if gain was 5-15% and premium barely moved for N minutes
  3) ML-ASSISTED — use trained LightGBM model to predict future PnL, exit when ML says sell
  4) ML + HEURISTIC — ML confirms sideways heuristic before exiting (reduces false positives)
  5) PREMIUM CAP VARIANTS — test $5.50/$6.50/$8.50 tiered alongside each exit strategy

Usage:
    python scripts/backtest_sideways_ml.py
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

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config, TickerCategory
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState, categorize_ticker
from options_owl.risk.exit_v5.types import ExitAction, ExitReason, _exit, _hold
from options_owl.risk.ml_exit import predict_sell, compute_features

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


def _is_sideways(premium_history, entry_premium, lookback=20, band_pct=5.0, min_gain_pct=5.0):
    """Detect if premium has been chopping sideways within a band.

    Returns (is_sideways, avg_gain_pct) — True if last `lookback` premiums
    stayed within `band_pct` of each other AND avg gain is in the small-win zone.
    """
    if len(premium_history) < lookback:
        return False, 0.0

    recent = premium_history[-lookback:]
    high = max(recent)
    low = min(recent)

    if low <= 0:
        return False, 0.0

    band = (high - low) / low * 100
    avg_prem = sum(recent) / len(recent)
    avg_gain = (avg_prem - entry_premium) / entry_premium * 100

    return band <= band_pct and avg_gain >= min_gain_pct, avg_gain


def simulate_trade_with_strategy(df, entry_premium, contracts, direction, dte,
                                  expiry_date, ticker, strategy="baseline"):
    """Run FSM with optional sideways/ML exit overlay.

    Strategies:
      baseline     — current production FSM only
      heuristic    — FSM + sideways heuristic (exit if sideways 20+ bars at +5%)
      ml_only      — FSM + ML model (exit when ML says sell and we're positive)
      ml_heuristic — FSM + ML confirms sideways before exit (safest)
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type == "call"

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
    premium_history = [entry_premium]

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        premium_history.append(premium)

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
        elapsed_min = (now - entry_ts).total_seconds() / 60
        gain_pct = (premium - entry_premium) / entry_premium * 100

        # Check FSM first
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

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

        # ── Strategy overlays (only if FSM says HOLD and we're past grace) ──
        if elapsed_min < 5:
            continue

        if strategy == "heuristic" or strategy == "ml_heuristic":
            is_sw, avg_gain = _is_sideways(premium_history, entry_premium,
                                            lookback=20, band_pct=5.0, min_gain_pct=5.0)
            if is_sw and elapsed_min >= 30:
                if strategy == "heuristic":
                    # Pure heuristic: exit now
                    pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
                    return {"pnl": pnl, "reason": "sideways_heuristic",
                            "hold": elapsed_min, "exit_prem": premium, "peak_gain": peak_gain}

                elif strategy == "ml_heuristic":
                    # ML confirms sideways before exit
                    ml_signal = predict_sell(
                        ticker=ticker,
                        entry_premium=entry_premium,
                        current_premium=premium,
                        peak_premium=state.peak_premium,
                        minutes_since_entry=elapsed_min,
                        now_hour=et_hour,
                        now_minute=now.minute,
                        is_call=is_call,
                        premium_history=premium_history[-30:],
                        underlying_entry=first_underlying,
                        underlying_current=underlying,
                    )
                    if ml_signal.should_sell:
                        pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                        peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
                        return {"pnl": pnl, "reason": "ml_sideways",
                                "hold": elapsed_min, "exit_prem": premium, "peak_gain": peak_gain}

        elif strategy == "ml_only":
            # ML-only: ask the model every cycle after 15 min when positive
            if elapsed_min >= 15 and gain_pct > 3:
                ml_signal = predict_sell(
                    ticker=ticker,
                    entry_premium=entry_premium,
                    current_premium=premium,
                    peak_premium=state.peak_premium,
                    minutes_since_entry=elapsed_min,
                    now_hour=et_hour,
                    now_minute=now.minute,
                    is_call=is_call,
                    premium_history=premium_history[-30:],
                    underlying_entry=first_underlying,
                    underlying_current=underlying,
                )
                if ml_signal.should_sell:
                    pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
                    return {"pnl": pnl, "reason": "ml_exit",
                            "hold": elapsed_min, "exit_prem": premium, "peak_gain": peak_gain}

    # EOD
    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def passes_premium_cap(premium, score, ticker, base_cap, mid_cap, high_cap):
    if ticker in INDEX_TICKERS:
        return True
    if score >= 150:
        return premium <= high_cap
    elif score >= 120:
        return premium <= mid_cap
    return premium <= base_cap


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-compute all tradeable signals with tick data
    all_prepared = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

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

        is_blocked = momentum_blocked(df, direction)

        all_prepared.append({
            "ticker": ticker,
            "direction": direction,
            "score": score,
            "premium": adj_entry,
            "contracts": contracts,
            "df": df,
            "dte": dte,
            "expiry_date": expiry_date,
            "momentum_blocked": is_blocked,
            "day": sig["created_at"][:10],
        })

    harvester_conn.close()
    print(f"Prepared {len(all_prepared)} signals ({no_data} no data)")

    # Premium cap + strategy combos
    CAP_CONFIGS = [
        ("$5/$6/$8 (current)", 5.0, 6.0, 8.0),
        ("$5.50/$6.50/$8.50", 5.5, 6.5, 8.5),
        ("$6/$7/$9", 6.0, 7.0, 9.0),
    ]

    STRATEGIES = ["baseline", "heuristic", "ml_only", "ml_heuristic"]

    results_grid = {}  # (cap_label, strategy) -> {pnl, wins, losses, trades, details}

    for cap_label, base_cap, mid_cap, high_cap in CAP_CONFIGS:
        # Filter by cap
        capped = [s for s in all_prepared
                  if passes_premium_cap(s["premium"], s["score"], s["ticker"],
                                        base_cap, mid_cap, high_cap)]
        tradeable = [s for s in capped if not s["momentum_blocked"]]

        for strat in STRATEGIES:
            key = (cap_label, strat)
            totals = {"pnl": 0, "wins": 0, "losses": 0, "trades": 0, "details": []}

            for i, sig in enumerate(tradeable):
                if strat == STRATEGIES[0] and (i + 1) % 50 == 0:
                    print(f"  [{cap_label}] Processing {i+1}/{len(tradeable)}...")

                result = simulate_trade_with_strategy(
                    sig["df"], sig["premium"], sig["contracts"],
                    sig["direction"], sig["dte"], sig["expiry_date"],
                    ticker=sig["ticker"], strategy=strat,
                )

                pnl = result["pnl"]
                totals["pnl"] += pnl
                totals["trades"] += 1
                if pnl > 0:
                    totals["wins"] += 1
                else:
                    totals["losses"] += 1
                totals["details"].append({
                    "ticker": sig["ticker"], "day": sig["day"],
                    "pnl": pnl, "reason": result["reason"],
                    "peak_gain": result["peak_gain"], "hold": result["hold"],
                    "entry": sig["premium"], "contracts": sig["contracts"],
                    "dte": sig["dte"],
                })

            results_grid[key] = totals

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"SIDEWAYS EXIT + PREMIUM CAP GRID — ALL COMBINATIONS")
    print(f"{'=' * 130}")

    baseline_pnl = results_grid[("$5/$6/$8 (current)", "baseline")]["pnl"]

    print(f"\n{'Cap':<22} {'Strategy':<16} {'Trades':>6} {'P&L':>12} {'vs Base':>10} "
          f"{'Win%':>6} {'AvgWin':>9} {'AvgLoss':>9}")
    print("-" * 100)

    for cap_label, _, _, _ in CAP_CONFIGS:
        for strat in STRATEGIES:
            s = results_grid[(cap_label, strat)]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            delta = s["pnl"] - baseline_pnl
            pnls = [d["pnl"] for d in s["details"]]
            avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            avg_loss = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0
            marker = " ***" if delta > 300 else (" **" if delta > 100 else "")
            print(f"{cap_label:<22} {strat:<16} {s['trades']:>6} ${s['pnl']:>10,.2f} "
                  f"${delta:>+8,.2f} {wr:>5.1f}% ${avg_win:>7,.2f} ${avg_loss:>7,.2f}{marker}")
        print()

    # ── Show what ML/heuristic changed ────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"TRADES WHERE STRATEGIES DIVERGED (current cap)")
    print(f"{'=' * 130}")

    base_details = results_grid[("$5/$6/$8 (current)", "baseline")]["details"]

    for strat in ["heuristic", "ml_only", "ml_heuristic"]:
        alt_details = results_grid[("$5/$6/$8 (current)", strat)]["details"]
        diffs = []
        for b, a in zip(base_details, alt_details):
            diff = a["pnl"] - b["pnl"]
            if abs(diff) > 10:
                diffs.append((b, a, diff))

        if not diffs:
            print(f"\n  {strat}: no divergent trades")
            continue

        diffs.sort(key=lambda x: x[2], reverse=True)
        saved = sum(d for _, _, d in diffs if d > 0)
        cost = sum(-d for _, _, d in diffs if d < 0)

        print(f"\n  {strat}: {len(diffs)} trades diverged (saved ${saved:,.2f}, cost ${cost:,.2f}, net ${saved-cost:+,.2f})")
        print(f"  {'Day':<12} {'Ticker':<7} {'DTE':>3} {'Ct':>3} {'BasePnL':>9} {'AltPnL':>9} "
              f"{'Delta':>8} {'BaseReason':<22} {'AltReason':<22}")
        print(f"  {'-' * 110}")
        for b, a, diff in diffs:
            print(f"  {b['day']:<12} {b['ticker']:<7} {b['dte']:>3} {b['contracts']:>3} "
                  f"${b['pnl']:>7,.2f} ${a['pnl']:>7,.2f} ${diff:>+6,.2f} "
                  f"{b['reason']:<22} {a['reason']:<22}")

    # ── Premium cap trade-off for blocked high-premium trades ─────────────
    print(f"\n{'=' * 130}")
    print(f"PREMIUM CAP: WHAT THE WIDER CAP ADDS")
    print(f"{'=' * 130}")

    for cap_label, base_cap, mid_cap, high_cap in CAP_CONFIGS[1:]:
        current = results_grid[("$5/$6/$8 (current)", "baseline")]
        wider = results_grid[(cap_label, "baseline")]
        extra_trades = wider["trades"] - current["trades"]
        extra_pnl = wider["pnl"] - current["pnl"]
        print(f"\n  {cap_label}: +{extra_trades} trades, ${extra_pnl:+,.2f} P&L change")

        # Find the extra trades
        current_set = {(d["day"], d["ticker"]) for d in current["details"]}
        for d in wider["details"]:
            key = (d["day"], d["ticker"])
            if key not in current_set:
                print(f"    {d['day']} {d['ticker']:6} ${d['entry']:.2f} x{d['contracts']} "
                      f"→ ${d['pnl']:+,.2f} ({d['reason']})")

    # ── Best overall combo ────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"RANKING — TOP 5 COMBINATIONS")
    print(f"{'=' * 130}")

    ranked = sorted(results_grid.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for i, ((cap, strat), s) in enumerate(ranked[:5]):
        delta = s["pnl"] - baseline_pnl
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  #{i+1} {cap} + {strat:<16} ${s['pnl']:>10,.2f} ({delta:>+,.2f}) "
              f"{wr:.1f}% WR, {s['trades']} trades")

    print(f"\n  Current production: ${baseline_pnl:,.2f}")


if __name__ == "__main__":
    main()
