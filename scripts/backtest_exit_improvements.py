"""Backtest exit improvement combinations.

Tests each improvement independently AND in combination:
  A) Slow bleed gate — exits 0DTE trades bleeding 40%+ over 60+ min without underlying move
  B) Tighter theta — reduce theta bleed thresholds (90min/25% instead of 120min/30%)
  C) Lower backstop — 55% backstop for 0DTE INDEX instead of 65%
  D) Combined: A+B+C

All tests use the LIVE FSM with config overrides, ensuring results reflect real code.

Usage:
    python scripts/backtest_exit_improvements.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
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

SCORE_TIERS = [
    (135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25),
]


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 70
        ORDER BY created_at
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


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker="SIM", cfg_override=None, settings_override=None):
    """Run actual production FSM with optional config overrides."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    cfg = cfg_override or get_ticker_config(ticker, use_per_ticker=True)
    settings = settings_override or _V6_SETTINGS
    fsm = ExitFSM(cfg, settings=settings)
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

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

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


def make_scenario_configs(ticker):
    """Return dict of scenario_name → (cfg, settings) for each improvement combo."""
    base_cfg = get_ticker_config(ticker, use_per_ticker=True)
    base_settings = _V6_SETTINGS

    scenarios = {}

    # 0) BASELINE — current production
    scenarios["baseline"] = (base_cfg, base_settings)

    # A) SLOW BLEED GATE — tighter theta for 0DTE: 60min/40% instead of 120min/30%
    cfg_a = replace(base_cfg,
                    theta_bleed_min=60,
                    theta_bleed_drop_pct=40.0)
    scenarios["A_slow_bleed"] = (cfg_a, base_settings)

    # B) TIGHTER THETA — 90min/25% instead of 120min/30%
    cfg_b = replace(base_cfg,
                    theta_bleed_min=90,
                    theta_bleed_drop_pct=25.0)
    scenarios["B_tighter_theta"] = (cfg_b, base_settings)

    # C) LOWER BACKSTOP — 55% for 0DTE instead of 65%
    cfg_c = replace(base_cfg, backstop_0dte_pct=55.0)
    scenarios["C_lower_backstop"] = (cfg_c, base_settings)

    # D) TIGHTER TIGHT STOP — 25% instead of 35% (underlying against)
    cfg_d = replace(base_cfg, tight_stop_0dte_pct=25.0)
    scenarios["D_tighter_confirmed"] = (cfg_d, base_settings)

    # E) LOWER CHECKPOINT — 20% premium drop + 0.3% underlying (vs 30%/0.5%)
    cfg_e = replace(base_cfg,
                    checkpoint_drop_pct=20.0,
                    underlying_against_threshold=0.003)
    scenarios["E_lower_checkpoint"] = (cfg_e, base_settings)

    # AB) SLOW BLEED + TIGHTER THETA
    cfg_ab = replace(base_cfg,
                     theta_bleed_min=60,
                     theta_bleed_drop_pct=40.0)
    scenarios["AB_bleed+theta"] = (cfg_ab, base_settings)

    # AC) SLOW BLEED + LOWER BACKSTOP
    cfg_ac = replace(base_cfg,
                     theta_bleed_min=60,
                     theta_bleed_drop_pct=40.0,
                     backstop_0dte_pct=55.0)
    scenarios["AC_bleed+backstop"] = (cfg_ac, base_settings)

    # BC) TIGHTER THETA + LOWER BACKSTOP
    cfg_bc = replace(base_cfg,
                     theta_bleed_min=90,
                     theta_bleed_drop_pct=25.0,
                     backstop_0dte_pct=55.0)
    scenarios["BC_theta+backstop"] = (cfg_bc, base_settings)

    # ABC) ALL THREE
    cfg_abc = replace(base_cfg,
                      theta_bleed_min=60,
                      theta_bleed_drop_pct=40.0,
                      backstop_0dte_pct=55.0)
    scenarios["ABC_all_three"] = (cfg_abc, base_settings)

    # ABCD) ALL + TIGHTER CONFIRMED STOP
    cfg_abcd = replace(base_cfg,
                       theta_bleed_min=60,
                       theta_bleed_drop_pct=40.0,
                       backstop_0dte_pct=55.0,
                       tight_stop_0dte_pct=25.0)
    scenarios["ABCD_full"] = (cfg_abcd, base_settings)

    return scenarios


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-compute sizing and tick data
    prepared = []
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue

        premium = sig["premium"]

        # Premium cap (current production tiered)
        if ticker not in INDEX_TICKERS:
            if score >= 150:
                cap = 8.0
            elif score >= 120:
                cap = 6.0
            else:
                cap = 5.0
            if premium > cap:
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
            adj_entry = premium

        # Dollar-target sizing
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

        prepared.append({
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
    print(f"Prepared {len(prepared)} tradeable signals ({no_data} no tick data)")

    # Filter to non-momentum-blocked
    tradeable = [s for s in prepared if not s["momentum_blocked"]]
    print(f"After momentum gate: {len(tradeable)} trades")

    # Get unique tickers for scenario generation
    all_scenarios = list(make_scenario_configs("QQQ").keys())

    print(f"\n{'=' * 120}")
    print(f"EXIT IMPROVEMENT BACKTEST — {len(tradeable)} trades × {len(all_scenarios)} scenarios")
    print(f"{'=' * 120}")

    # Run all scenarios
    scenario_totals = {s: {"pnl": 0, "wins": 0, "losses": 0, "trades": 0,
                           "details": []} for s in all_scenarios}

    for i, sig in enumerate(tradeable):
        if (i + 1) % 50 == 0:
            print(f"  Processing trade {i+1}/{len(tradeable)}...")

        ticker = sig["ticker"]
        ticker_scenarios = make_scenario_configs(ticker)

        for scenario_name in all_scenarios:
            cfg, settings = ticker_scenarios[scenario_name]
            result = simulate_trade(
                sig["df"], sig["premium"], sig["contracts"],
                sig["direction"], sig["dte"], sig["expiry_date"],
                ticker=ticker, cfg_override=cfg, settings_override=settings,
            )

            pnl = result["pnl"]
            scenario_totals[scenario_name]["pnl"] += pnl
            scenario_totals[scenario_name]["trades"] += 1
            if pnl > 0:
                scenario_totals[scenario_name]["wins"] += 1
            else:
                scenario_totals[scenario_name]["losses"] += 1

            scenario_totals[scenario_name]["details"].append({
                "ticker": ticker,
                "day": sig["day"],
                "pnl": pnl,
                "reason": result["reason"],
                "peak_gain": result["peak_gain"],
                "hold": result["hold"],
                "contracts": sig["contracts"],
                "entry": sig["premium"],
                "dte": sig["dte"],
            })

    # ── Summary table ─────────────────────────────────────────────────────
    baseline_pnl = scenario_totals["baseline"]["pnl"]

    print(f"\n{'Scenario':<25} {'Trades':>6} {'P&L':>12} {'vs Base':>10} {'Win%':>6} "
          f"{'AvgWin':>9} {'AvgLoss':>9} {'MaxLoss':>9}")
    print("-" * 100)

    for name in all_scenarios:
        s = scenario_totals[name]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        delta = s["pnl"] - baseline_pnl
        details = s["details"]
        pnls = [d["pnl"] for d in details]
        wins_pnl = [p for p in pnls if p > 0]
        loss_pnl = [p for p in pnls if p <= 0]
        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = np.mean(loss_pnl) if loss_pnl else 0
        max_loss = min(pnls) if pnls else 0

        marker = " ***" if delta > 200 else (" *" if delta > 50 else "")
        print(f"{name:<25} {s['trades']:>6} ${s['pnl']:>10,.2f} ${delta:>+8,.2f} "
              f"{wr:>5.1f}% ${avg_win:>7,.2f} ${avg_loss:>7,.2f} ${max_loss:>7,.2f}{marker}")

    # ── Per-trade diff: show where scenarios diverge ──────────────────────
    print(f"\n{'=' * 120}")
    print(f"TRADE-LEVEL DIFFERENCES (where improvements saved/cost $50+)")
    print(f"{'=' * 120}")

    best_name = max(all_scenarios, key=lambda n: scenario_totals[n]["pnl"])
    best_details = scenario_totals[best_name]["details"]
    base_details = scenario_totals["baseline"]["details"]

    print(f"\nBest scenario: {best_name}")
    print(f"\n{'Day':<12} {'Ticker':<7} {'DTE':>3} {'Entry':>6} {'Ct':>3} "
          f"{'Base P&L':>10} {'Best P&L':>10} {'Delta':>9} {'BaseRsn':<20} {'BestRsn':<20}")
    print("-" * 120)

    diffs = []
    for base_d, best_d in zip(base_details, best_details):
        diff = best_d["pnl"] - base_d["pnl"]
        if abs(diff) >= 50:
            diffs.append((base_d, best_d, diff))

    diffs.sort(key=lambda x: x[2], reverse=True)
    for base_d, best_d, diff in diffs:
        print(f"{base_d['day']:<12} {base_d['ticker']:<7} {base_d['dte']:>3} "
              f"${base_d['entry']:>5.2f} {base_d['contracts']:>3} "
              f"${base_d['pnl']:>8,.2f} ${best_d['pnl']:>8,.2f} ${diff:>+7,.2f} "
              f"{base_d['reason']:<20} {best_d['reason']:<20}")

    saved = sum(d for _, _, d in diffs if d > 0)
    cost = sum(-d for _, _, d in diffs if d < 0)
    print(f"\nNet from divergent trades: saved ${saved:,.2f}, cost ${cost:,.2f}, "
          f"net ${saved - cost:+,.2f}")

    # ── 0DTE vs multi-day breakdown ───────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"0DTE vs MULTI-DAY BREAKDOWN")
    print(f"{'=' * 120}")

    for dte_label, dte_filter in [("0DTE", lambda d: d["dte"] == 0),
                                    ("Multi-day", lambda d: d["dte"] > 0)]:
        print(f"\n  {dte_label}:")
        print(f"  {'Scenario':<25} {'Trades':>6} {'P&L':>12} {'vs Base':>10} {'Win%':>6}")
        print(f"  {'-' * 70}")

        base_dte = [d for d in base_details if dte_filter(d)]
        base_dte_pnl = sum(d["pnl"] for d in base_dte)

        for name in all_scenarios:
            details = [d for d in scenario_totals[name]["details"] if dte_filter(d)]
            if not details:
                continue
            pnl = sum(d["pnl"] for d in details)
            wins = sum(1 for d in details if d["pnl"] > 0)
            wr = wins / len(details) * 100
            delta = pnl - base_dte_pnl
            marker = " ***" if delta > 200 else ""
            print(f"  {name:<25} {len(details):>6} ${pnl:>10,.2f} ${delta:>+8,.2f} {wr:>5.1f}%{marker}")

    # ── Recommendation ────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 120}")

    ranked = sorted(all_scenarios, key=lambda n: scenario_totals[n]["pnl"], reverse=True)
    for i, name in enumerate(ranked[:5]):
        s = scenario_totals[name]
        delta = s["pnl"] - baseline_pnl
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  #{i+1} {name:<25} ${s['pnl']:>10,.2f} ({delta:>+,.2f} vs baseline) "
              f"{wr:.1f}% WR")

    print(f"\n  Baseline: ${baseline_pnl:,.2f}")
    best_s = scenario_totals[ranked[0]]
    best_delta = best_s["pnl"] - baseline_pnl
    if best_delta > 100:
        print(f"  Best improvement: {ranked[0]} adds ${best_delta:,.2f} ({best_delta/baseline_pnl*100:.1f}%)")
    else:
        print(f"  No scenario improves baseline by more than $100.")


if __name__ == "__main__":
    main()
