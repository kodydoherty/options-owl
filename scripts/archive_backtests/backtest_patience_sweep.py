"""Backtest A/B: current V5 config vs "patient" configs.

Tests the hypothesis that giving trades more time to develop (especially
early in the day) would improve P&L. Compares:
  A) Current production config (per-ticker, 5min grace)
  B) Patient: 15min grace, no aggressive per-ticker overrides
  C) Very patient: 30min grace, wider soft trail band
  D) Time-aware patience: 30min grace before 11AM ET, 10min grace after

Usage:
    python scripts/backtest_patience_sweep.py
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

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    V5Config,
    TICKER_CONFIGS,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000


# ── Configs to compare ───────────────────────────────────────────────────────

CONFIGS = {
    "A_current": {
        "desc": "Current production (5min grace, per-ticker configs)",
        "use_per_ticker": True,
        "default_cfg": V5Config(),
        "grace_override": None,       # use per-ticker or default (5min)
        "soft_band_low": None,        # use default (15%)
    },
    "B_patient_15m": {
        "desc": "15min grace, default backstops (no DEFENSIVE overrides)",
        "use_per_ticker": False,       # disable aggressive per-ticker
        "default_cfg": V5Config(grace_period_min=15.0),
        "grace_override": 15.0,
        "soft_band_low": None,
    },
    "C_patient_30m": {
        "desc": "30min grace, soft trail band raised to 30%",
        "use_per_ticker": False,
        "default_cfg": V5Config(grace_period_min=30.0, soft_trail_band_low_pct=30.0),
        "grace_override": 30.0,
        "soft_band_low": 30.0,
    },
    "D_time_aware": {
        "desc": "30min grace before 11AM ET, 10min after, soft band 25%",
        "use_per_ticker": True,        # keep per-ticker but widen backstops
        "default_cfg": V5Config(soft_trail_band_low_pct=25.0),
        "grace_override": "time_aware",  # special: 30min before 11AM, 10min after
        "soft_band_low": 25.0,
        # Override AAPL/META DEFENSIVE backstop to default 65%
        "backstop_override": 65.0,
    },
    "E_wider_backstop": {
        "desc": "Current + all backstops at 65% (no DEFENSIVE 50%)",
        "use_per_ticker": True,
        "default_cfg": V5Config(),
        "grace_override": None,
        "soft_band_low": None,
        "backstop_override": 65.0,
    },
}


# ── Data loading (from backtest_v5_production.py) ────────────────────────────

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

    df = pd.DataFrame(rows, columns=[
        "captured_at", "midpoint", "bid", "ask", "underlying_price"
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker, config_name, config_spec):
    """Run FSM with a specific config variant."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    # Build the config for this variant
    use_per_ticker = config_spec["use_per_ticker"]
    default_cfg = config_spec["default_cfg"]

    if use_per_ticker and ticker in TICKER_CONFIGS:
        cfg = TICKER_CONFIGS[ticker]
        # Apply grace override if specified
        grace = config_spec.get("grace_override")
        if grace and grace != "time_aware":
            cfg = replace(cfg, grace_period_min=grace)
        # Apply backstop override (remove DEFENSIVE overrides)
        backstop_ov = config_spec.get("backstop_override")
        if backstop_ov:
            cfg = replace(cfg, backstop_0dte_pct=max(cfg.backstop_0dte_pct, backstop_ov))
        # Apply soft band low
        band_low = config_spec.get("soft_band_low")
        if band_low:
            cfg = replace(cfg, soft_trail_band_low_pct=band_low)
    else:
        cfg = default_cfg

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
        trade_id=1,
        ticker=ticker,
        option_type=option_type,
        entry_premium=entry_premium,
        entry_time=entry_ts,
        contracts=contracts,
        peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte,
        expiry_date=expiry_date or "",
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

        # Time-aware grace: check ET hour and adjust grace dynamically
        grace_override = config_spec.get("grace_override")
        if grace_override == "time_aware":
            et_hour = now.hour - 4
            if et_hour < 0:
                et_hour += 24
            if et_hour < 11:  # before 11 AM ET — more patience
                dynamic_cfg = replace(cfg, grace_period_min=30.0)
            elif et_hour < 14:  # 11 AM - 2 PM
                dynamic_cfg = replace(cfg, grace_period_min=15.0)
            else:  # after 2 PM — tighter
                dynamic_cfg = replace(cfg, grace_period_min=10.0)
            fsm = ExitFSM(dynamic_cfg)

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
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = (premium - entry_premium) * contracts * 100
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
            }

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = (last_prem - entry_premium) * contracts * 100
    return {
        "pnl": pnl,
        "reason": "eod_data_end",
        "hold": elapsed,
        "exit_prem": last_prem,
        "peak_gain": peak_gain,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load tick data (shared across all configs)
    tick_cache = {}
    no_data = 0
    for sig in signals:
        key = (sig["ticker"], sig["strike"], sig["created_at"])
        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue
        tick_cache[key] = (df, sig)

    print(f"Tick data for {len(tick_cache)} signals ({no_data} skipped)\n")

    # Run each config
    all_results = {}

    for config_name, config_spec in CONFIGS.items():
        results = []

        for key, (df, sig) in tick_cache.items():
            ticker = sig["ticker"]
            direction = (sig["direction"] or "bullish").lower()
            score = sig["score"] or 80
            dte = sig.get("_dte", 0)
            expiry_date = sig.get("_expiry_date", "")

            first_ask = df["ask"].iloc[0]
            first_mid = df["premium"].iloc[0]
            adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
            if adj_entry <= 0:
                adj_entry = sig["premium"]

            # Sizing (same for all configs — isolate exit strategy impact)
            cost_per = adj_entry * 100
            deployable = PORTFOLIO * 0.75
            per_slot = deployable / 8
            position_cap = PORTFOLIO * 0.08

            SCORE_TIERS = [(135, 1.0), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]
            score_mult = 0.25
            for threshold, mult in SCORE_TIERS:
                if score >= threshold:
                    score_mult = mult
                    break

            scaled_target = per_slot * score_mult
            raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
            pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
            contracts = max(1, min(raw_contracts, pos_cap_contracts))

            result = simulate(
                df, adj_entry, contracts, direction, dte, expiry_date,
                ticker, config_name, config_spec,
            )
            result["ticker"] = ticker
            result["day"] = sig["created_at"][:10]
            result["score"] = score
            result["entry"] = adj_entry
            result["contracts"] = contracts
            result["direction"] = direction
            results.append(result)

        all_results[config_name] = pd.DataFrame(results)

    harvester_conn.close()

    # ── Summary comparison ───────────────────────────────────────────────────

    print(f"{'=' * 100}")
    print(f"PATIENCE SWEEP — A/B COMPARISON")
    print(f"{'=' * 100}")
    print(f"\n{'Config':<22} {'Desc':<55} {'Total P&L':>10} {'Win%':>6} {'AvgHold':>8} {'Trades':>6}")
    print("-" * 110)

    for config_name, df_r in all_results.items():
        desc = CONFIGS[config_name]["desc"][:54]
        pnl = df_r["pnl"].sum()
        wins = (df_r["pnl"] > 0).sum()
        wr = wins / len(df_r) * 100 if len(df_r) > 0 else 0
        avg_hold = df_r["hold"].mean()
        print(f"{config_name:<22} {desc:<55} ${pnl:>8,.0f} {wr:>5.1f}% {avg_hold:>6.0f}m {len(df_r):>6}")

    # ── Detailed per-exit-reason comparison ──────────────────────────────────

    for config_name, df_r in all_results.items():
        desc = CONFIGS[config_name]["desc"]
        print(f"\n{'─' * 80}")
        print(f"{config_name}: {desc}")
        print(f"{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6} {'AvgHold':>8}")
        print("-" * 70)
        for reason, group in df_r.groupby("reason"):
            gpnl = group["pnl"]
            gwins = (gpnl > 0).sum()
            gwr = gwins / len(gpnl) * 100
            print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}% {group['hold'].mean():>6.0f}m")

    # ── Head-to-head: trades that differ between A and best alternative ──────

    baseline = all_results["A_current"]
    best_name = max(
        [k for k in all_results if k != "A_current"],
        key=lambda k: all_results[k]["pnl"].sum()
    )
    best = all_results[best_name]

    print(f"\n{'=' * 100}")
    print(f"HEAD-TO-HEAD: A_current vs {best_name}")
    print(f"{'=' * 100}")

    # Merge on index (same signal order)
    diff = pd.DataFrame({
        "ticker": baseline["ticker"],
        "day": baseline["day"],
        "direction": baseline["direction"],
        "entry": baseline["entry"],
        "contracts": baseline["contracts"],
        "pnl_current": baseline["pnl"],
        "pnl_patient": best["pnl"],
        "reason_current": baseline["reason"],
        "reason_patient": best["reason"],
        "hold_current": baseline["hold"],
        "hold_patient": best["hold"],
    })
    diff["pnl_diff"] = diff["pnl_patient"] - diff["pnl_current"]

    # Show trades where patience made the biggest difference
    diff_sorted = diff.reindex(diff["pnl_diff"].abs().sort_values(ascending=False).index)
    changed = diff_sorted[diff_sorted["pnl_diff"].abs() > 10].head(30)

    if len(changed) > 0:
        print(f"\nTrades with biggest P&L difference (patience vs current):")
        print(f"{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Ct':>3} "
              f"{'PnL_Curr':>10} {'PnL_Pat':>10} {'Diff':>10} "
              f"{'Reason_Curr':<20} {'Reason_Pat':<20} {'Hold_C':>6} {'Hold_P':>6}")
        print("-" * 130)
        for _, r in changed.iterrows():
            print(f"{r['day']:<12} {r['ticker']:<6} {r['direction'][:4]:<5} {r['contracts']:>3} "
                  f"${r['pnl_current']:>8,.2f} ${r['pnl_patient']:>8,.2f} ${r['pnl_diff']:>8,.2f} "
                  f"{r['reason_current']:<20} {r['reason_patient']:<20} "
                  f"{r['hold_current']:>5.0f}m {r['hold_patient']:>5.0f}m")

    # ── Time-of-day analysis ─────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"TIME-OF-DAY ANALYSIS (A_current)")
    print(f"{'=' * 100}")

    # Tag each trade with entry hour (ET)
    for config_name, df_r in [("A_current", baseline), (best_name, best)]:
        # Use the day + approximate time from signals
        # (tick data starts at signal time, so first tick ~ entry time)
        pass

    print(f"\nDelta: {best_name} total P&L - A_current total P&L = "
          f"${best['pnl'].sum() - baseline['pnl'].sum():+,.2f}")
    print(f"\nRecommendation: Use '{best_name}' if delta is positive and win rate is comparable.")


if __name__ == "__main__":
    main()
