"""Per-ticker FSM config tuning — test 12 strategies per ticker, find optimal.

Tests variations of:
  - Stop tightness (tight vs wide vs default)
  - Trail width (tight vs wide vs default)
  - Profit-taking (early profit lock vs let it ride)
  - Grace period (short vs default)
  - Theta bleed (quick vs slow)

Usage:
    python scripts/backtest_per_ticker_tuning.py
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

from options_owl.risk.exit_v5.config import V5Config, AdaptiveTier, TickerCategory
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000


# ── 12 FSM Configs ──────────────────────────────────────────────────────────

def make_configs() -> dict[str, V5Config]:
    """Create 12 named FSM configurations to test per ticker."""
    base = V5Config()

    configs = {"DEFAULT": base}

    # TIGHT_STOP: tighter graduated stops (25% / 50% vs 35% / 65%)
    configs["TIGHT_STOP"] = replace(base,
        tight_stop_0dte_pct=25.0,
        backstop_0dte_pct=50.0,
        checkpoint_drop_pct=20.0,
    )

    # WIDE_STOP: wider stops, more room to breathe (45% / 75%)
    configs["WIDE_STOP"] = replace(base,
        tight_stop_0dte_pct=45.0,
        backstop_0dte_pct=75.0,
        checkpoint_drop_pct=40.0,
    )

    # TIGHT_TRAIL: tighter adaptive trails across all categories
    configs["TIGHT_TRAIL"] = replace(base,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        adaptive_index_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
    )

    # WIDE_TRAIL: wider adaptive trails — let runners run
    configs["WIDE_TRAIL"] = replace(base,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 45), AdaptiveTier(150, 65), AdaptiveTier(40, 60),
        ),
        adaptive_index_tiers=(
            AdaptiveTier(300, 35), AdaptiveTier(100, 50), AdaptiveTier(30, 45),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 35), AdaptiveTier(100, 50), AdaptiveTier(30, 45),
        ),
    )

    # EARLY_PROFIT: lock profits earlier (profit target at 20% for all, not just index)
    configs["EARLY_PROFIT"] = replace(base,
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=8.0,
        soft_trail_keep_pct=0.70,
    )

    # QUICK_SCALP: aggressive scalp trail + early soft trail
    configs["QUICK_SCALP"] = replace(base,
        scalp_peak_threshold_pct=15.0,
        scalp_fade_ratio=0.50,
        soft_trail_band_low_pct=8.0,
        soft_trail_band_high_pct=40.0,
        soft_trail_keep_pct=0.70,
    )

    # SHORT_GRACE: 3min grace instead of 5
    configs["SHORT_GRACE"] = replace(base,
        grace_period_min=3.0,
    )

    # LONG_GRACE: 8min grace
    configs["LONG_GRACE"] = replace(base,
        grace_period_min=8.0,
    )

    # FAST_THETA: aggressive theta bleed (90min/20% instead of 120min/30%)
    configs["FAST_THETA"] = replace(base,
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    )

    # TIGHT+QUICK: tight trails + quick scalp (aggressive profit taking)
    configs["TIGHT+QUICK"] = replace(base,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        adaptive_index_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
        scalp_peak_threshold_pct=15.0,
        scalp_fade_ratio=0.50,
        soft_trail_keep_pct=0.70,
    )

    # WIDE+PATIENT: wide trails + longer theta (let trades develop)
    configs["WIDE+PATIENT"] = replace(base,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 45), AdaptiveTier(150, 65), AdaptiveTier(40, 60),
        ),
        adaptive_index_tiers=(
            AdaptiveTier(300, 35), AdaptiveTier(100, 50), AdaptiveTier(30, 45),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 35), AdaptiveTier(100, 50), AdaptiveTier(30, 45),
        ),
        theta_bleed_min=150.0,
        grace_period_min=7.0,
    )

    # DEFENSIVE: tight stops + tight trail + fast theta (minimize losses)
    configs["DEFENSIVE"] = replace(base,
        tight_stop_0dte_pct=25.0,
        backstop_0dte_pct=50.0,
        checkpoint_drop_pct=20.0,
        adaptive_highvol_tiers=(
            AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35),
        ),
        adaptive_index_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
        adaptive_standard_tiers=(
            AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25),
        ),
        theta_bleed_min=90.0,
        theta_bleed_drop_pct=20.0,
    )

    return configs


# ── Data loading ─────────────────────────────────────────────────────────────

def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 78
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


def compute_contracts(entry_premium, score):
    max_risk_pct = 0.75
    max_concurrent = 5
    max_position_pct = 0.15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent
    position_cap = PORTFOLIO * max_position_pct
    if score >= 95:
        score_mult = 1.0
    elif score >= 90:
        score_mult = 0.75
    elif score >= 85:
        score_mult = 0.50
    else:
        score_mult = 0.25
    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, pos_cap, 20))


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker, cfg):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    fsm = ExitFSM(cfg)
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

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )
        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = (premium - entry_premium) * contracts * 100
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
    pnl = (last_prem - entry_premium) * contracts * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    configs = make_configs()
    config_names = list(configs.keys())
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # results[config_name] = list of {pnl, reason, ticker, day, ...}
    all_results = {name: [] for name in config_names}
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

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

        contracts = compute_contracts(adj_entry, score)

        for name in config_names:
            result = simulate_fsm(
                df, adj_entry, contracts, direction, dte, expiry_date,
                ticker=ticker, cfg=configs[name],
            )
            result.update({"ticker": ticker, "day": day, "score": score,
                           "entry": adj_entry, "contracts": contracts,
                           "direction": direction})
            all_results[name].append(result)

    harvester_conn.close()

    n_trades = len(all_results["DEFAULT"])
    if not n_trades:
        print("No results")
        return

    # ── Overall summary ──────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"PER-TICKER FSM CONFIG TUNING — {n_trades} trades, {no_data} skipped")
    print(f"{'=' * 100}")

    print(f"\n{'Config':<18} {'Total P&L':>12} {'Win Rate':>10} {'Avg Win':>10} {'Avg Loss':>10} {'Avg Hold':>8}")
    print("-" * 72)
    for name in config_names:
        df_r = pd.DataFrame(all_results[name])
        pnls = df_r["pnl"]
        wins = (pnls > 0).sum()
        total = pnls.sum()
        wr = wins / len(pnls) * 100
        avg_w = pnls[pnls > 0].mean() if wins > 0 else 0
        avg_l = pnls[pnls <= 0].mean() if (pnls <= 0).sum() > 0 else 0
        avg_h = df_r["hold"].mean()
        marker = " <--" if name == "DEFAULT" else ""
        print(f"{name:<18} ${total:>10,.2f} {wr:>9.1f}% ${avg_w:>8,.2f} ${avg_l:>8,.2f} {avg_h:>6.0f}m{marker}")

    # ── Per-ticker: best config ──────────────────────────────────────────

    tickers = sorted(set(r["ticker"] for r in all_results["DEFAULT"]))

    print(f"\n\n{'=' * 100}")
    print("BEST CONFIG PER TICKER")
    print(f"{'=' * 100}")

    print(f"\n{'Ticker':<8} {'Trades':>6} {'Default P&L':>12} {'Best Config':<18} {'Best P&L':>12} {'Delta':>12}")
    print("-" * 72)

    total_default = 0
    total_best = 0

    for t in tickers:
        n = sum(1 for r in all_results["DEFAULT"] if r["ticker"] == t)
        best_name = "DEFAULT"
        best_pnl = -999999
        for name in config_names:
            t_pnl = sum(r["pnl"] for r in all_results[name] if r["ticker"] == t)
            if t_pnl > best_pnl:
                best_pnl = t_pnl
                best_name = name
        default_pnl = sum(r["pnl"] for r in all_results["DEFAULT"] if r["ticker"] == t)
        delta = best_pnl - default_pnl
        total_default += default_pnl
        total_best += best_pnl
        marker = "" if best_name == "DEFAULT" else f" +${delta:,.0f}"
        print(f"{t:<8} {n:>6} ${default_pnl:>10,.2f} {best_name:<18} ${best_pnl:>10,.2f} ${delta:>10,.2f}{marker}")

    print(f"\n{'TOTAL':<8} {n_trades:>6} ${total_default:>10,.2f} {'OPTIMAL MIX':<18} ${total_best:>10,.2f} "
          f"${total_best - total_default:>10,.2f}")

    # ── Full grid: every config × every ticker ───────────────────────────

    print(f"\n\n{'=' * 100}")
    print("FULL GRID: P&L by Config × Ticker")
    print(f"{'=' * 100}")

    # Header
    print(f"\n{'Config':<18}", end="")
    for t in tickers:
        print(f" {t:>8}", end="")
    print(f" {'TOTAL':>10}")
    print("-" * (18 + 9 * len(tickers) + 11))

    for name in config_names:
        print(f"{name:<18}", end="")
        row_total = 0
        for t in tickers:
            t_pnl = sum(r["pnl"] for r in all_results[name] if r["ticker"] == t)
            row_total += t_pnl
            print(f" ${t_pnl:>6,.0f}", end="")
        marker = " <--" if name == "DEFAULT" else ""
        print(f" ${row_total:>8,.0f}{marker}")

    # ── Per-signal detail for top improvement tickers ────────────────────

    print(f"\n\n{'=' * 100}")
    print("PER-SIGNAL DETAIL — Tickers with biggest improvement potential")
    print(f"{'=' * 100}")

    improvements = []
    for t in tickers:
        default_pnl = sum(r["pnl"] for r in all_results["DEFAULT"] if r["ticker"] == t)
        best_name = "DEFAULT"
        best_pnl = -999999
        for name in config_names:
            t_pnl = sum(r["pnl"] for r in all_results[name] if r["ticker"] == t)
            if t_pnl > best_pnl:
                best_pnl = t_pnl
                best_name = name
        if best_name != "DEFAULT":
            improvements.append((t, best_name, best_pnl - default_pnl))

    improvements.sort(key=lambda x: x[2], reverse=True)

    for ticker, best_cfg, delta in improvements[:6]:
        default_trades = [r for r in all_results["DEFAULT"] if r["ticker"] == ticker]
        best_trades = [r for r in all_results[best_cfg] if r["ticker"] == ticker]

        print(f"\n  {ticker} — Best: {best_cfg} (+${delta:,.0f})")
        print(f"  {'Day':<12} {'Dir':<5} {'Entry':>7} {'Ct':>3} "
              f"{'Default':>10} {'DReason':<16} {best_cfg:>10} {'BReason':<16} {'Delta':>10}")
        print(f"  {'-' * 100}")
        for d_t, b_t in zip(default_trades, best_trades):
            d_delta = b_t["pnl"] - d_t["pnl"]
            marker = " OK" if d_delta > 0 else (" --" if d_delta == 0 else " XX")
            print(f"  {d_t['day']:<12} {d_t['direction'][:4]:<5} ${d_t['entry']:>5.2f} {d_t['contracts']:>3} "
                  f"${d_t['pnl']:>8,.2f} {d_t['reason']:<16} ${b_t['pnl']:>8,.2f} {b_t['reason']:<16} "
                  f"${d_delta:>8,.2f}{marker}")


if __name__ == "__main__":
    main()
