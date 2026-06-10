"""Backtest today's signals with proposed config fixes vs current production.

Tests 3 configs:
  1. CURRENT: what ran today (V5 FSM, no V6 features)
  2. V6_DEPLOYED: V6 features as deployed (scaleout, breakeven ratchet, etc)
  3. V6_TUNED: V6 + proposed fixes (wider checkpoint for NVDA, higher soft trail band)

Usage:
    python scripts/backtest_today_fixes.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    INDEX_TICKERS,
    TICKER_CONFIGS,
    V5Config,
    AdaptiveTier,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000


def _settings(v6=False):
    return SimpleNamespace(
        ENABLE_V6_PER_TICKER_CONFIG=v6,
        ENABLE_V6_BREAKEVEN_RATCHET=v6,
        V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V6_2PM_TIGHTEN=v6,
        V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
        V6_2PM_SOFT_TRAIL_BOOST=0.15,
        ENABLE_V6_PREMIUM_CAP=v6,
        V6_PREMIUM_CAP=5.0,
        ENABLE_V6_SPREAD_GATE=v6,
        V6_MAX_SPREAD_PCT=15.0,
        ENABLE_V6_SCALEOUT=v6,
        V6_SCALEOUT_GAIN_PCT=20.0,
        V6_SCALEOUT_FRACTION=0.333,
        V6_SCALEOUT_MIN_CONTRACTS=3,
        ENABLE_V6_DCA=v6,
        V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
        V6_DCA_MIN_MINUTES=8.0,
        V6_DCA_MAX_MINUTES=20.0,
        V6_DCA_MIN_DIP_PCT=15.0,
        V6_DCA_MAX_DIP_PCT=35.0,
        V6_DCA_UNDERLYING_THRESHOLD=0.5,
    )


# ── Proposed V6 TUNED configs ──────────────────────────────────────────

TUNED_TICKER_CONFIGS = {
    **TICKER_CONFIGS,
    # NVDA: widen checkpoint from 30% to 50% — high-vol needs more room
    "NVDA": V5Config(
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=8.0,
        soft_trail_keep_pct=0.70,
        checkpoint_drop_pct=50.0,  # WAS 30% (default)
    ),
    # AVGO: raise soft trail band low so it doesn't fire on small peaks
    "AVGO": V5Config(
        profit_target_index_0dte_pct=20.0,
        soft_trail_band_low_pct=20.0,  # WAS 8% — don't trail until +20%
        soft_trail_keep_pct=0.70,
    ),
    # QQQ: raise soft trail band low + longer grace
    "QQQ": V5Config(
        grace_period_min=8.0,
        soft_trail_band_low_pct=20.0,  # WAS 10% (default) — don't trail until +20%
    ),
}


# ── Data loading ────────────────────────────────────────────────────────

def load_signals(date="2026-05-04"):
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry,
               entry_price, created_at
        FROM trade_signals
        WHERE score >= 78 AND date(created_at) = ?
        ORDER BY created_at
    """, (date,)).fetchall()
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
            SELECT captured_at, midpoint, bid, ask, underlying_price
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
    df = pd.DataFrame(rows, columns=["captured_at", "midpoint", "bid", "ask", "underlying_price"])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def compute_contracts(entry_premium, score):
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 5
    position_cap = PORTFOLIO * 0.15
    if score >= 95: sm = 1.0
    elif score >= 90: sm = 0.75
    elif score >= 85: sm = 0.50
    else: sm = 0.25
    cost_per = entry_premium * 100
    scaled = per_slot * sm
    raw = int(scaled / cost_per) if cost_per > 0 else 1
    pos_cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, pos_cap, 20))


# ── Simulation ──────────────────────────────────────────────────────────

def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker, settings, custom_configs=None):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold_min": 0, "peak_gain": 0,
                "exit_prem": 0, "final_contracts": contracts}

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

    use_per_ticker = getattr(settings, "ENABLE_V6_PER_TICKER_CONFIG", False)
    if custom_configs and use_per_ticker and ticker in custom_configs:
        cfg = custom_configs[ticker]
    elif use_per_ticker:
        cfg = get_ticker_config(ticker, use_per_ticker=True)
    else:
        cfg = V5Config()

    fsm = ExitFSM(cfg, settings=settings)

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
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

    remaining = contracts
    scaleout_pnl = 0.0
    avg_entry = entry_premium
    total_cost = entry_premium * contracts * 100

    # DCA
    enable_dca = getattr(settings, "ENABLE_V6_DCA", False)
    dca_tickers = set()
    if enable_dca:
        dca_tickers = {t.strip().upper() for t in getattr(settings, "V6_DCA_TICKERS", "").split(",") if t.strip()}
    dca_fired = False

    def _result(pnl, reason, exit_prem, now):
        elapsed = (now - entry_ts).total_seconds() / 60
        pg = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
        return {"pnl": round(pnl, 2), "reason": reason, "hold_min": round(elapsed, 1),
                "peak_gain": round(pg, 1), "exit_prem": round(exit_prem, 4),
                "final_contracts": remaining}

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue
        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium
        now = df["ts"].iloc[idx]
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        underlying = df["underlying_price"].iloc[idx] or 0.0
        elapsed_min = (now - entry_ts).total_seconds() / 60.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        # DCA
        if enable_dca and not dca_fired and ticker in dca_tickers:
            dca_min = getattr(settings, "V6_DCA_MIN_MINUTES", 8.0)
            dca_max = getattr(settings, "V6_DCA_MAX_MINUTES", 20.0)
            if dca_min <= elapsed_min <= dca_max and avg_entry > 0:
                dip_pct = (avg_entry - premium) / avg_entry * 100
                dca_min_dip = getattr(settings, "V6_DCA_MIN_DIP_PCT", 15.0)
                dca_max_dip = getattr(settings, "V6_DCA_MAX_DIP_PCT", 35.0)
                if dca_min_dip <= dip_pct <= dca_max_dip:
                    underlying_ok = True
                    if first_underlying > 0 and underlying > 0:
                        u_move = (underlying - first_underlying) / first_underlying * 100
                        if is_call and u_move < -0.5:
                            underlying_ok = False
                        elif not is_call and u_move > 0.5:
                            underlying_ok = False
                    if underlying_ok:
                        dca_fired = True
                        total_cost += premium * contracts * 100
                        remaining += contracts
                        avg_entry = total_cost / (remaining * 100)
                        state.entry_premium = avg_entry
                        state.contracts = remaining
                        state.peak_premium = max(avg_entry, premium)

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0:
                close_qty = min(action.contracts_to_close, remaining)
                scaleout_pnl += (premium - avg_entry) * close_qty * 100
                remaining -= close_qty
                state.contracts = remaining
                if remaining <= 0:
                    return _result(scaleout_pnl, action.reason.value, premium, now)
                continue
            pnl = (premium - avg_entry) * remaining * 100 + scaleout_pnl
            return _result(pnl, action.reason.value, premium, now)

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    pnl = (last_prem - avg_entry) * remaining * 100 + scaleout_pnl
    return _result(pnl, "eod_data_end", last_prem, last_ts)


def fmt(v):
    return f"${v:+,.2f}"


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    configs = {
        "V5_CURRENT": (_settings(v6=False), None),
        "V6_DEPLOYED": (_settings(v6=True), None),
        "V6_TUNED": (_settings(v6=True), TUNED_TICKER_CONFIGS),
    }

    print("=" * 110)
    print("TODAY'S SIGNALS — CONFIG COMPARISON")
    print("=" * 110)

    header = f"{'Signal':<25} {'Entry':>7} {'Ct':>3}"
    for name in configs:
        header += f" | {name:>18} {'Exit':>15}"
    print(header)
    print("-" * 110)

    totals = {name: 0.0 for name in configs}

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        sig_time = sig["created_at"][11:16] if len(sig["created_at"]) > 11 else ""

        df = load_ticks(harvester_conn, sig)
        if df is None:
            print(f"  {sig_time} {ticker:<5} {direction[:4]:<5} score={score}  — NO DATA")
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]
        contracts = compute_contracts(adj_entry, score)

        label = f"{sig_time} {ticker:<5} {direction[:4].upper():<5} s={score}"
        line = f"  {label:<23} ${adj_entry:>5.2f} {contracts:>3}"

        for name, (settings, custom_cfgs) in configs.items():
            result = simulate(df, adj_entry, contracts, direction, dte, expiry_date,
                              ticker, settings, custom_cfgs)
            totals[name] += result["pnl"]
            pnl_str = fmt(result["pnl"])
            reason = result["reason"][:15]
            ct_info = f"x{result['final_contracts']}" if result["final_contracts"] != contracts else ""
            line += f" | {pnl_str:>10} {reason:<15} {ct_info}"

        print(line)

    print("-" * 110)
    total_line = f"  {'TOTAL':<23} {'':>7} {'':>3}"
    for name in configs:
        total_line += f" | {fmt(totals[name]):>10} {'':15}"
    print(total_line)
    print()

    # Show improvement
    print("IMPROVEMENT SUMMARY:")
    base = totals["V5_CURRENT"]
    for name in configs:
        if name == "V5_CURRENT":
            continue
        delta = totals[name] - base
        print(f"  {name} vs V5_CURRENT: {fmt(delta)}")

    v6d = totals["V6_DEPLOYED"]
    v6t = totals["V6_TUNED"]
    print(f"  V6_TUNED vs V6_DEPLOYED: {fmt(v6t - v6d)}")

    harvester_conn.close()


if __name__ == "__main__":
    main()
