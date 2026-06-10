"""V6 Gap Analysis — quantify each source of difference between research and production.

Runs 4 configurations:
  1. Research V6 (all features including DCA) — should match $16,514
  2. Research V6 WITHOUT DCA — isolates DCA impact
  3. Production V6 (FSM-based, no DCA) — should match $12,217
  4. Production V6 with 0.80 cap fix — new production baseline

The delta between #2 and #3 is the "true FSM gap" (gate ordering, cap, etc.).
The delta between #1 and #2 is the "DCA gap".

Usage:
    python scripts/backtest_v6_gap_analysis.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    INDEX_TICKERS,
    V5Config,
    AdaptiveTier,
    TickerCategory,
    categorize_ticker,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.types import ExitReason

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000

# Tickers where DCA helped in research
DCA_TICKERS = {"MSFT", "IWM", "SPY", "QQQ", "AMZN", "NVDA"}

# Research per-ticker configs (duplicated from backtest_v6_combined.py)
RESEARCH_TICKER_CONFIGS = {
    "NVDA": replace(V5Config(), profit_target_index_0dte_pct=20.0, soft_trail_band_low_pct=8.0, soft_trail_keep_pct=0.70),
    "GOOGL": replace(V5Config(), tight_stop_0dte_pct=45.0, backstop_0dte_pct=75.0, checkpoint_drop_pct=40.0),
    "TSLA": replace(V5Config(), grace_period_min=8.0),
    "IWM": replace(V5Config(), tight_stop_0dte_pct=45.0, backstop_0dte_pct=75.0, checkpoint_drop_pct=40.0),
    "QQQ": replace(V5Config(), grace_period_min=8.0),
    "META": replace(V5Config(), tight_stop_0dte_pct=25.0, backstop_0dte_pct=50.0, checkpoint_drop_pct=20.0,
                    adaptive_highvol_tiers=(AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35)),
                    theta_bleed_min=90.0, theta_bleed_drop_pct=20.0),
    "AAPL": replace(V5Config(), tight_stop_0dte_pct=25.0, backstop_0dte_pct=50.0, checkpoint_drop_pct=20.0,
                    adaptive_highvol_tiers=(AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35)),
                    theta_bleed_min=90.0, theta_bleed_drop_pct=20.0),
    "AMZN": replace(V5Config(), adaptive_highvol_tiers=(AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35)),
                    adaptive_standard_tiers=(AdaptiveTier(300, 20), AdaptiveTier(100, 30), AdaptiveTier(30, 25))),
    "AVGO": replace(V5Config(), profit_target_index_0dte_pct=20.0, soft_trail_band_low_pct=8.0, soft_trail_keep_pct=0.70),
    "MSFT": replace(V5Config(), profit_target_index_0dte_pct=20.0, soft_trail_band_low_pct=8.0, soft_trail_keep_pct=0.70),
    "MSTR": replace(V5Config(), adaptive_highvol_tiers=(AdaptiveTier(400, 25), AdaptiveTier(150, 40), AdaptiveTier(40, 35)),
                    scalp_peak_threshold_pct=15.0, scalp_fade_ratio=0.50, soft_trail_keep_pct=0.70),
}


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


# ── Research-style simulation (ad-hoc V6 logic OUTSIDE FSM) ────────────────


def simulate_research(df, entry_premium, contracts, direction, dte, expiry_date,
                      ticker, cfg, enable_dca=True):
    """Replicate the research backtest logic from backtest_v6_combined.py."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "peak_gain": 0,
                "scaleout_pnl": 0, "dca_fired": False}

    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type in ("call", "bullish")

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

    remaining_contracts = contracts
    scaleout_pnl = 0.0
    scaled_out = False
    dca_fired = False
    total_cost = entry_premium * contracts * 100
    avg_entry = entry_premium
    hit_20pct = False

    fsm = ExitFSM(cfg)
    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=remaining_contracts, peak_premium=entry_premium,
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
        elapsed_min = (now - entry_ts).total_seconds() / 60.0

        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        gain = (premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0

        # DCA (research style: ad-hoc, outside FSM)
        if (enable_dca and not dca_fired and ticker in DCA_TICKERS
                and 8.0 <= elapsed_min <= 20.0):
            dip_pct = (entry_premium - premium) / entry_premium * 100
            if 15.0 <= dip_pct <= 35.0:
                underlying_ok = True
                if first_underlying > 0 and underlying > 0:
                    u_move = (underlying - first_underlying) / first_underlying * 100
                    if is_call and u_move < -0.5:
                        underlying_ok = False
                    elif not is_call and u_move > 0.5:
                        underlying_ok = False
                if underlying_ok:
                    dca_fired = True
                    add_ct = max(1, contracts)
                    total_cost += premium * add_ct * 100
                    remaining_contracts += add_ct
                    avg_entry = total_cost / (remaining_contracts * 100)
                    state.entry_premium = avg_entry
                    state.contracts = remaining_contracts
                    state.peak_premium = max(avg_entry, premium)
                    gain = (premium - avg_entry) / avg_entry * 100

        # Scale-out (research style: ad-hoc, outside FSM)
        if not scaled_out and gain >= 20 and remaining_contracts >= 3:
            scaleout_ct = remaining_contracts // 3
            scaleout_pnl = (premium - avg_entry) * scaleout_ct * 100
            remaining_contracts -= scaleout_ct
            scaled_out = True
            state.contracts = remaining_contracts

        # Break-even ratchet (research style: ad-hoc, outside FSM)
        if gain >= 20:
            hit_20pct = True
        if hit_20pct and premium < avg_entry:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return {"pnl": pnl, "reason": "breakeven_ratchet", "hold": elapsed,
                    "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired}

        # 2PM tightening (research style: replace FSM)
        if et_hour >= 14 and cfg is not fsm.cfg:
            pass
        elif et_hour >= 14:
            cat = categorize_ticker(ticker)
            base_tiers = cfg.get_adaptive_tiers(cat)
            tight_tiers = tuple(
                AdaptiveTier(t.min_peak_gain, t.trail_width * 0.7)
                for t in base_tiers
            )
            tight_cfg = replace(cfg, soft_trail_keep_pct=min(0.80, cfg.soft_trail_keep_pct + 0.15))
            if cat == TickerCategory.HIGH_VOL:
                tight_cfg = replace(tight_cfg, adaptive_highvol_tiers=tight_tiers)
            elif cat == TickerCategory.INDEX:
                tight_cfg = replace(tight_cfg, adaptive_index_tiers=tight_tiers)
            else:
                tight_cfg = replace(tight_cfg, adaptive_standard_tiers=tight_tiers)
            fsm = ExitFSM(tight_cfg)

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
    pnl = (last_prem - avg_entry) * remaining_contracts * 100 + scaleout_pnl
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl, "dca_fired": dca_fired}


# ── Production-style simulation (V6 gates INSIDE FSM) ─────────────────────


def simulate_production(df, entry_premium, contracts, direction, dte, expiry_date,
                        ticker, settings):
    """Simulate using production ExitFSM with V6 settings (gates inside FSM)."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "peak_gain": 0, "scaleout_pnl": 0}

    option_type = "put" if direction in ("bearish", "put") else "call"

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
        et_minute = now.minute
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + et_minute))

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0

            if action.contracts_to_close > 0:
                close_qty = min(action.contracts_to_close, remaining_contracts)
                scaleout_pnl += (premium - avg_entry) * close_qty * 100
                remaining_contracts -= close_qty
                state.contracts = remaining_contracts
                if remaining_contracts <= 0:
                    return {"pnl": scaleout_pnl, "reason": action.reason.value,
                            "hold": elapsed, "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl}
                continue

            pnl = (premium - avg_entry) * remaining_contracts * 100 + scaleout_pnl
            return {"pnl": pnl, "reason": action.reason.value,
                    "hold": elapsed, "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
    pnl = (last_prem - avg_entry) * remaining_contracts * 100 + scaleout_pnl
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "peak_gain": peak_gain, "scaleout_pnl": scaleout_pnl}


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    v6_settings = SimpleNamespace(
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
    )

    # 4 configs: research+DCA, research-DCA, production, production (with 0.80 cap fix)
    res_dca = []       # 1: Research with DCA
    res_no_dca = []    # 2: Research without DCA
    prod_results = []  # 3: Production FSM (with 0.80 cap fix)

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

        # Entry filters (same for all V6 configs)
        is_index = ticker in INDEX_TICKERS
        entry_filtered = False
        if not is_index and adj_entry > 5.0:
            entry_filtered = True
            filtered += 1
        elif (first_bid and first_ask_val and not pd.isna(first_bid)
                and not pd.isna(first_ask_val)
                and first_ask_val > 0 and first_bid > 0):
            spread_pct = (first_ask_val - first_bid) / adj_entry * 100
            if spread_pct > 15:
                entry_filtered = True
                filtered += 1

        if entry_filtered:
            zero = {"pnl": 0, "reason": "filtered", "hold": 0, "peak_gain": 0,
                    "scaleout_pnl": 0, "dca_fired": False}
            zero.update(meta)
            res_dca.append(zero)
            res_no_dca.append(zero.copy())
            prod_results.append(zero.copy())
            continue

        ticker_cfg = RESEARCH_TICKER_CONFIGS.get(ticker, V5Config())

        # 1: Research with DCA
        r1 = simulate_research(df, adj_entry, contracts, direction, dte, expiry_date,
                               ticker, ticker_cfg, enable_dca=True)
        r1.update(meta)
        res_dca.append(r1)

        # 2: Research without DCA
        r2 = simulate_research(df, adj_entry, contracts, direction, dte, expiry_date,
                               ticker, ticker_cfg, enable_dca=False)
        r2.update(meta)
        res_no_dca.append(r2)

        # 3: Production FSM
        r3 = simulate_production(df, adj_entry, contracts, direction, dte, expiry_date,
                                 ticker, v6_settings)
        r3.update(meta)
        prod_results.append(r3)

    harvester_conn.close()

    pnl_dca = sum(r["pnl"] for r in res_dca)
    pnl_no_dca = sum(r["pnl"] for r in res_no_dca)
    pnl_prod = sum(r["pnl"] for r in prod_results)

    dca_impact = pnl_dca - pnl_no_dca
    fsm_gap = pnl_no_dca - pnl_prod

    print(f"\n{'=' * 80}")
    print("V6 GAP ANALYSIS — Decomposing research vs production difference")
    print(f"{'=' * 80}")
    print(f"  Signals: {len(res_dca)} total, {no_data} no data, {filtered} filtered")
    print()
    print(f"  1. Research V6 (with DCA):       ${pnl_dca:>10,.2f}")
    print(f"  2. Research V6 (without DCA):     ${pnl_no_dca:>10,.2f}")
    print(f"  3. Production V6 (FSM, 0.80 cap): ${pnl_prod:>10,.2f}")
    print()
    print(f"  DCA impact (1 - 2):               ${dca_impact:>+10,.2f}")
    print(f"  FSM gap (2 - 3):                  ${fsm_gap:>+10,.2f}")
    print(f"  Total gap (1 - 3):                ${pnl_dca - pnl_prod:>+10,.2f}")
    print()

    if abs(fsm_gap) < abs(pnl_dca) * 0.05:
        print("  RESULT: FSM gap is < 5% of total — DCA is the primary difference.")
        print("  DCA lives in position_monitor (not FSM). The FSM implementation is correct.")
    else:
        print(f"  RESULT: FSM gap is significant (${fsm_gap:+,.2f}).")
        print("  Investigating per-trade differences...")

    # Per-trade comparison where research-no-DCA differs from production
    print(f"\n{'=' * 80}")
    print("TRADES WHERE RESEARCH (no DCA) != PRODUCTION (>$50 difference)")
    print(f"{'=' * 80}")
    print(f"\n{'Day':<12} {'Ticker':<6} {'Entry':>7} {'Ct':>3} "
          f"{'Res noDCA':>10} {'Prod':>10} {'Delta':>10} {'Res Reason':<22} {'Prod Reason':<22}")
    print("-" * 115)

    diff_count = 0
    for i in range(len(res_no_dca)):
        r = res_no_dca[i]
        p = prod_results[i]
        delta = r["pnl"] - p["pnl"]
        if abs(delta) > 50:
            diff_count += 1
            print(f"{r['day']:<12} {r['ticker']:<6} ${r['entry']:>5.2f} {r['contracts']:>3} "
                  f"${r['pnl']:>8,.2f} ${p['pnl']:>8,.2f} ${delta:>+8,.2f} "
                  f"{r['reason']:<22} {p['reason']:<22}")

    print(f"\n  {diff_count} trades with >$50 difference")

    # DCA impact per ticker
    print(f"\n{'=' * 80}")
    print("DCA IMPACT BY TICKER")
    print(f"{'=' * 80}")
    print(f"\n{'Ticker':<8} {'DCA Fires':>10} {'With DCA':>12} {'Without DCA':>12} {'DCA Impact':>12}")
    print("-" * 58)
    tickers = sorted(set(r["ticker"] for r in res_dca))
    for t in tickers:
        t_dca = sum(r["pnl"] for r in res_dca if r["ticker"] == t)
        t_no = sum(r["pnl"] for r in res_no_dca if r["ticker"] == t)
        t_fires = sum(1 for r in res_dca if r["ticker"] == t and r.get("dca_fired"))
        impact = t_dca - t_no
        if t_fires > 0 or abs(impact) > 10:
            print(f"{t:<8} {t_fires:>10} ${t_dca:>10,.2f} ${t_no:>10,.2f} ${impact:>+10,.2f}")

    # Exit reason comparison
    print(f"\n{'=' * 80}")
    print("EXIT REASON COMPARISON — Research (no DCA) vs Production")
    print(f"{'=' * 80}")
    df_res = pd.DataFrame(res_no_dca)
    df_prod = pd.DataFrame(prod_results)

    all_reasons = sorted(set(df_res["reason"].unique()) | set(df_prod["reason"].unique()))
    print(f"\n{'Reason':<25} {'Res Count':>10} {'Res P&L':>12} {'Prod Count':>10} {'Prod P&L':>12}")
    print("-" * 72)
    for reason in all_reasons:
        r_mask = df_res["reason"] == reason
        p_mask = df_prod["reason"] == reason
        r_cnt = r_mask.sum()
        p_cnt = p_mask.sum()
        r_pnl = df_res.loc[r_mask, "pnl"].sum()
        p_pnl = df_prod.loc[p_mask, "pnl"].sum()
        print(f"{reason:<25} {r_cnt:>10} ${r_pnl:>10,.2f} {p_cnt:>10} ${p_pnl:>10,.2f}")


if __name__ == "__main__":
    main()
