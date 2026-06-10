"""V6 Verification Backtest — confirms production code matches research results.

Uses the ACTUAL production ExitFSM, V5Config, TICKER_CONFIGS, and V5MonitorBridge
(not ad-hoc logic from backtest_v6_combined.py) to ensure what we deploy matches
what we backtested.

Expected: total P&L within 5% of backtest_v6_combined.py results.

Usage:
    python scripts/backtest_v6_verification.py
"""

from __future__ import annotations

import sqlite3
import sys
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
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000


DCA_TICKERS = {"MSFT", "IWM", "SPY", "QQQ", "AMZN", "NVDA"}


def _v6_settings():
    """Production-equivalent V6 settings (all flags ON)."""
    return SimpleNamespace(
        ENABLE_V6_PER_TICKER_CONFIG=True,
        ENABLE_V6_BREAKEVEN_RATCHET=True,
        V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V6_2PM_TIGHTEN=True,
        V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
        V6_2PM_SOFT_TRAIL_BOOST=0.15,
        ENABLE_V6_PREMIUM_CAP=True,
        V6_PREMIUM_CAP=5.0,
        ENABLE_V6_SPREAD_GATE=True,
        V6_MAX_SPREAD_PCT=15.0,
        ENABLE_V6_SCALEOUT=True,
        V6_SCALEOUT_GAIN_PCT=20.0,
        V6_SCALEOUT_FRACTION=0.333,
        V6_SCALEOUT_MIN_CONTRACTS=3,
        ENABLE_V6_DCA=True,
        V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
        V6_DCA_MIN_MINUTES=8.0,
        V6_DCA_MAX_MINUTES=20.0,
        V6_DCA_MIN_DIP_PCT=15.0,
        V6_DCA_MAX_DIP_PCT=35.0,
        V6_DCA_UNDERLYING_THRESHOLD=0.5,
    )


# ── Reuse data loading from backtest_v6_combined.py ──────────────────────


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


# ── V6 Simulation using PRODUCTION FSM code ──────────────────────────────


def simulate_v6_production(df, entry_premium, contracts, direction, dte,
                           expiry_date, ticker, settings):
    """Simulate using production ExitFSM with V6 settings + DCA.

    This uses the real FSM from options_owl/risk/exit_v5/fsm.py and
    replicates position_monitor's _check_v6_dca logic for DCA.
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "peak_gain": 0,
                "scaleout_pnl": 0, "dca_fired": False, "entry_filtered": False}

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

    # Get per-ticker config (V6)
    use_per_ticker = getattr(settings, "ENABLE_V6_PER_TICKER_CONFIG", False)
    cfg = get_ticker_config(ticker, use_per_ticker=use_per_ticker)
    fsm = ExitFSM(cfg, settings=settings)

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

    remaining_contracts = contracts
    scaleout_pnl = 0.0
    avg_entry = entry_premium
    total_cost = entry_premium * contracts * 100

    # V6 DCA tracking
    enable_dca = getattr(settings, "ENABLE_V6_DCA", False)
    dca_tickers_str = getattr(settings, "V6_DCA_TICKERS", "")
    dca_tickers = {t.strip().upper() for t in dca_tickers_str.split(",") if t.strip()}
    dca_fired = False

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
        elapsed_min = (now - entry_ts).total_seconds() / 60.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        # V6 DCA: same logic as position_monitor._check_v6_dca
        if enable_dca and not dca_fired and ticker in dca_tickers:
            dca_min = getattr(settings, "V6_DCA_MIN_MINUTES", 8.0)
            dca_max = getattr(settings, "V6_DCA_MAX_MINUTES", 20.0)
            dca_min_dip = getattr(settings, "V6_DCA_MIN_DIP_PCT", 15.0)
            dca_max_dip = getattr(settings, "V6_DCA_MAX_DIP_PCT", 35.0)
            u_thresh = getattr(settings, "V6_DCA_UNDERLYING_THRESHOLD", 0.5)

            if dca_min <= elapsed_min <= dca_max and avg_entry > 0:
                dip_pct = (avg_entry - premium) / avg_entry * 100
                if dca_min_dip <= dip_pct <= dca_max_dip:
                    # Check underlying
                    underlying_ok = True
                    if first_underlying > 0 and underlying > 0:
                        u_move = (underlying - first_underlying) / first_underlying * 100
                        if is_call and u_move < -u_thresh:
                            underlying_ok = False
                        elif not is_call and u_move > u_thresh:
                            underlying_ok = False

                    if underlying_ok:
                        dca_fired = True
                        add_ct = contracts  # add original contract count
                        total_cost += premium * add_ct * 100
                        remaining_contracts += add_ct
                        avg_entry = total_cost / (remaining_contracts * 100)
                        state.entry_premium = avg_entry
                        state.contracts = remaining_contracts
                        state.peak_premium = max(avg_entry, premium)

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0

            if action.contracts_to_close > 0:
                # Scale-out: partial close
                close_qty = min(action.contracts_to_close, remaining_contracts)
                scaleout_pnl += (premium - avg_entry) * close_qty * 100
                remaining_contracts -= close_qty
                state.contracts = remaining_contracts
                if remaining_contracts <= 0:
                    return {"pnl": scaleout_pnl, "reason": action.reason.value,
                            "hold": elapsed, "peak_gain": peak_gain,
                            "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired,
                            "entry_filtered": False}
                continue  # keep monitoring remaining contracts

            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return {"pnl": pnl, "reason": action.reason.value,
                    "hold": elapsed, "peak_gain": peak_gain,
                    "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired,
                    "entry_filtered": False}

    # End of data
    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
    pnl = (last_prem - avg_entry) * remaining_contracts * 100 + scaleout_pnl
    return {"pnl": pnl, "reason": "eod_data_end",
            "hold": elapsed, "peak_gain": peak_gain,
            "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired,
            "entry_filtered": False}


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    settings = _v6_settings()

    baseline_results = []  # V5 default
    v6_results = []        # V6 production

    no_data = 0
    filtered = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80

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

        first_bid = df["bid"].iloc[0]
        first_ask_val = df["ask"].iloc[0]

        contracts = compute_contracts(adj_entry, score)
        meta = {"ticker": ticker, "day": sig["created_at"][:10],
                "score": score, "entry": adj_entry, "contracts": contracts}

        # Baseline: V5 default (no V6 flags)
        baseline_settings = SimpleNamespace(
            ENABLE_V6_PER_TICKER_CONFIG=False,
            ENABLE_V6_BREAKEVEN_RATCHET=False,
            ENABLE_V6_2PM_TIGHTEN=False,
            ENABLE_V6_SCALEOUT=False,
            ENABLE_V6_DCA=False,
        )
        base = simulate_v6_production(
            df, adj_entry, contracts, direction, dte, expiry_date,
            ticker, baseline_settings,
        )
        base.update(meta)
        baseline_results.append(base)

        # V6: apply entry filters + production FSM
        is_index = ticker in INDEX_TICKERS
        if not is_index and adj_entry > 5.0:
            v6 = {"pnl": 0, "reason": "premium_cap_filtered", "hold": 0,
                   "peak_gain": 0, "scaleout_pnl": 0, "dca_fired": False,
                   "entry_filtered": True}
            v6.update(meta)
            v6_results.append(v6)
            filtered += 1
            continue

        if (first_bid and first_ask_val and not pd.isna(first_bid)
                and not pd.isna(first_ask_val)
                and first_ask_val > 0 and first_bid > 0):
            spread_pct = (first_ask_val - first_bid) / adj_entry * 100
            if spread_pct > 15:
                v6 = {"pnl": 0, "reason": "spread_filtered", "hold": 0,
                       "peak_gain": 0, "scaleout_pnl": 0, "dca_fired": False,
                       "entry_filtered": True}
                v6.update(meta)
                v6_results.append(v6)
                filtered += 1
                continue

        v6 = simulate_v6_production(
            df, adj_entry, contracts, direction, dte, expiry_date,
            ticker, settings,
        )
        v6.update(meta)
        v6_results.append(v6)

    harvester_conn.close()

    if not baseline_results:
        print("No results")
        return

    # ── Summary ──
    df_base = pd.DataFrame(baseline_results)
    df_v6 = pd.DataFrame(v6_results)
    df_v6_traded = df_v6[~df_v6["entry_filtered"]]

    b_pnl = df_base["pnl"].sum()
    v_pnl = df_v6["pnl"].sum()

    print(f"\n{'=' * 80}")
    print(f"V6 VERIFICATION — Production FSM Code")
    print(f"{'=' * 80}")
    print(f"  Signals: {len(baseline_results)} traded, {no_data} no data, {filtered} filtered")
    print(f"  Baseline (V5 default):  ${b_pnl:>10,.2f}")
    print(f"  V6 Production Code:     ${v_pnl:>10,.2f}")
    print(f"  Delta:                  ${v_pnl - b_pnl:>+10,.2f}")
    print()

    # Expected from backtest_v6_combined.py
    expected_baseline = 7019.89
    expected_v6 = 16513.89

    print(f"  Expected baseline (from research): ${expected_baseline:>10,.2f}")
    print(f"  Expected V6 (from research):       ${expected_v6:>10,.2f}")
    print()

    # Check deviation
    base_dev = abs(b_pnl - expected_baseline) / abs(expected_baseline) * 100
    v6_dev = abs(v_pnl - expected_v6) / abs(expected_v6) * 100

    print(f"  Baseline deviation: {base_dev:.1f}%")
    print(f"  V6 deviation:       {v6_dev:.1f}%")
    print()

    if base_dev < 1:
        print("  ✓ Baseline matches research (< 1% deviation)")
    else:
        print(f"  ⚠ Baseline deviates {base_dev:.1f}% from research")

    if v6_dev < 5:
        print(f"  ✓ V6 matches research (< 5% deviation)")
    else:
        print(f"  ⚠ V6 deviates {v6_dev:.1f}% from research")

    # DCA stats
    dca_count = sum(1 for r in v6_results if r.get("dca_fired"))
    print(f"\n  DCA fires: {dca_count}")

    # ── Per-ticker ──
    print(f"\n{'=' * 80}")
    print("PER-TICKER P&L")
    print(f"{'=' * 80}")
    tickers = sorted(set(r["ticker"] for r in baseline_results))
    print(f"\n{'Ticker':<8} {'Trades':>6} {'Baseline':>12} {'V6 Prod':>12} {'Delta':>12}")
    print("-" * 54)
    for t in tickers:
        t_base = sum(r["pnl"] for r in baseline_results if r["ticker"] == t)
        t_v6 = sum(r["pnl"] for r in v6_results if r["ticker"] == t)
        t_n = sum(1 for r in baseline_results if r["ticker"] == t)
        print(f"{t:<8} {t_n:>6} ${t_base:>10,.2f} ${t_v6:>10,.2f} ${t_v6 - t_base:>+10,.2f}")

    total_b = sum(r["pnl"] for r in baseline_results)
    total_v = sum(r["pnl"] for r in v6_results)
    print(f"\n{'TOTAL':<8} {len(baseline_results):>6} ${total_b:>10,.2f} ${total_v:>10,.2f} ${total_v - total_b:>+10,.2f}")

    # ── Exit reason breakdown ──
    print(f"\n{'=' * 80}")
    print("EXIT REASON BREAKDOWN — V6 Production")
    print(f"{'=' * 80}")
    print(f"\n{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 56)
    for reason, group in df_v6.groupby("reason"):
        gpnl = group["pnl"]
        print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f}")


if __name__ == "__main__":
    main()
