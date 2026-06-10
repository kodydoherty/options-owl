"""Backtest: Per-ticker runner ceiling — tighten trail when odds of running drop.

Strategy: Hold ALL contracts (no partial sells). Instead, dynamically adjust
the adaptive trail width based on per-ticker runner probability data.

Each ticker has gain "zones" where we know from historical data whether
it's likely to keep running:
  - Below ceiling: use normal (wide) adaptive trail — let it breathe
  - Above ceiling: switch to TIGHT trail — capture the peak, don't give back

This uses the FSM exactly as-is but with modified V5Config per tick
based on how far the gain has progressed vs the ticker's known ceiling.

Usage:
    python scripts/backtest_runner_ceiling.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import copy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    AdaptiveTier,
    TickerCategory,
    V5Config,
    categorize_ticker,
    get_ticker_config,
)
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
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000

SCORE_TIERS = [
    (135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25),
]

# --- Per-ticker runner ceiling data (from per_ticker_runner_odds.py) ---
# ceiling = gain % above which <50% of trades go 50%+ higher
# tighten_at = gain % where we start tightening (slightly before ceiling)
# trail_above_ceiling = trail width to use above ceiling (tighter = exit sooner)

TICKER_CEILINGS = {
    # Reliable runners — high ceiling, keep wide trail longer
    "AMZN":  {"ceiling": 250, "tighten_at": 200, "trail_above": 20},
    "QQQ":   {"ceiling": 300, "tighten_at": 200, "trail_above": 20},
    "GOOGL": {"ceiling": 200, "tighten_at": 150, "trail_above": 20},
    "IWM":   {"ceiling": 500, "tighten_at": 400, "trail_above": 25},
    "SPY":   {"ceiling": 200, "tighten_at": 125, "trail_above": 20},
    "META":  {"ceiling": 300, "tighten_at": 200, "trail_above": 25},

    # Moderate runners — medium ceiling
    "TSLA":  {"ceiling": 150, "tighten_at": 100, "trail_above": 20},
    "MSTR":  {"ceiling": 150, "tighten_at": 100, "trail_above": 20},
    "PLTR":  {"ceiling": 125, "tighten_at": 75,  "trail_above": 20},

    # Non-runners — low ceiling, tighten early
    "AAPL":  {"ceiling": 75,  "tighten_at": 50,  "trail_above": 15},
    "NVDA":  {"ceiling": 100, "tighten_at": 75,  "trail_above": 20},
    "MSFT":  {"ceiling": 75,  "tighten_at": 50,  "trail_above": 15},
    "AMD":   {"ceiling": 125, "tighten_at": 75,  "trail_above": 20},
    "AVGO":  {"ceiling": 100, "tighten_at": 75,  "trail_above": 15},
}

# Category defaults for unknown tickers
CATEGORY_CEILING_DEFAULTS = {
    TickerCategory.INDEX:    {"ceiling": 200, "tighten_at": 150, "trail_above": 20},
    TickerCategory.HIGH_VOL: {"ceiling": 150, "tighten_at": 100, "trail_above": 25},
    TickerCategory.STANDARD: {"ceiling": 150, "tighten_at": 100, "trail_above": 20},
}


def get_ceiling(ticker):
    if ticker in TICKER_CEILINGS:
        return TICKER_CEILINGS[ticker]
    cat = categorize_ticker(ticker)
    return CATEGORY_CEILING_DEFAULTS[cat]


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def load_signals():
    conn = sqlite3.connect(SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, sentiment, score,
               atm_premium, otm_premium, strike, expiry, created_at
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
        "captured_at", "midpoint", "bid", "ask", "underlying_price",
    ])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"], format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def score_to_contracts(score, premium):
    risk_pct = 0.75
    max_concurrent = 5
    max_pos_pct = 0.15
    deployable = PORTFOLIO * risk_pct
    per_slot = deployable / max_concurrent
    pos_cap = PORTFOLIO * max_pos_pct
    mult = 0
    for tier_score, tier_mult in SCORE_TIERS:
        if score >= tier_score:
            mult = tier_mult
            break
    if mult == 0:
        return 0
    budget = per_slot * mult
    cost = premium * 100
    if cost <= 0:
        return 0
    raw = int(budget / cost)
    cap = int(pos_cap / cost)
    return max(1, min(raw, cap))


def _strip_tz(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def simulate_baseline(df, entry_premium, contracts, direction, dte, expiry_date, ticker):
    """Production FSM — no modifications."""
    if entry_premium <= 0:
        return None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    entry_ts = _strip_tz(df["ts"].iloc[0])

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
        now = _strip_tz(df["ts"].iloc[idx])
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
                locked_pnl += (premium - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            gain_pct = (premium - entry_premium) / entry_premium * 100
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "exit_gain": gain_pct, "peak_gain": peak_gain,
            }

    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    return {
        "pnl": pnl, "reason": "last_tick",
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "peak_gain": peak_gain,
    }


def simulate_ceiling(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
                     ceiling_config):
    """FSM with dynamic trail tightening at the runner ceiling.

    Below tighten_at: normal per-ticker FSM config
    Above tighten_at: override adaptive trail tiers to use trail_above width
    """
    if entry_premium <= 0:
        return None

    tighten_at = ceiling_config["tighten_at"]
    trail_above = ceiling_config["trail_above"]
    ceiling = ceiling_config["ceiling"]

    base_cfg = get_ticker_config(ticker, use_per_ticker=True)
    category = categorize_ticker(ticker)

    # Build the "tightened" config — same as base but with aggressive trail above ceiling
    tight_tiers = (
        AdaptiveTier(400, trail_above),
        AdaptiveTier(150, trail_above),
        AdaptiveTier(tighten_at, trail_above),
    )
    if category == TickerCategory.HIGH_VOL:
        tight_cfg = replace(base_cfg, adaptive_highvol_tiers=tight_tiers)
    elif category == TickerCategory.INDEX:
        tight_cfg = replace(base_cfg, adaptive_index_tiers=tight_tiers)
    else:
        tight_cfg = replace(base_cfg, adaptive_standard_tiers=tight_tiers)

    # Also tighten soft trail above ceiling
    tight_cfg = replace(tight_cfg, soft_trail_keep_pct=0.75)

    option_type = "put" if direction in ("bearish", "put") else "call"
    entry_ts = _strip_tz(df["ts"].iloc[0])

    first_underlying = 0.0
    for i in range(min(5, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            first_underlying = float(u)
            break

    # Start with normal config FSM
    current_cfg = base_cfg
    fsm = ExitFSM(current_cfg, settings=_V6_SETTINGS)

    state = TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    locked_pnl = 0.0
    remaining = contracts
    tightened = False
    tighten_tick = None

    for idx in range(1, len(df)):
        premium = df["premium"].iloc[idx]
        if np.isnan(premium) or premium <= 0:
            continue

        raw_bid = df["bid"].iloc[idx]
        raw_ask = df["ask"].iloc[idx]
        bid = float(raw_bid) if raw_bid and not pd.isna(raw_bid) else premium
        ask = float(raw_ask) if raw_ask and not pd.isna(raw_ask) else premium
        now = _strip_tz(df["ts"].iloc[idx])
        underlying = df["underlying_price"].iloc[idx] or 0.0
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100

        # Switch to tight config when peak gain hits tighten_at threshold
        if not tightened and peak_gain >= tighten_at:
            tightened = True
            tighten_tick = idx
            # Rebuild FSM with tight config, preserving state
            fsm = ExitFSM(tight_cfg, settings=_V6_SETTINGS)

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (premium - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            gain_pct = (premium - entry_premium) / entry_premium * 100
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "exit_gain": gain_pct, "peak_gain": peak_gain,
                "tightened": tightened,
                "tighten_tick": tighten_tick,
            }

    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    return {
        "pnl": pnl, "reason": "last_tick",
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "peak_gain": peak_gain,
        "tightened": tightened,
        "tighten_tick": tighten_tick,
    }


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals")

    hconn = sqlite3.connect(HARVESTER_DB)

    results_bl = []
    results_ceil = []

    # Also test a few trail_above variants
    variants = [
        ("tight_15", 15),
        ("tight_20", 20),
        ("tight_25", 25),
        ("tight_30", 30),
    ]
    variant_results = {name: [] for name, _ in variants}

    matched = 0
    for i, sig in enumerate(signals):
        df = load_ticks(hconn, sig)
        if df is None:
            continue
        matched += 1

        ticker = sig["ticker"]
        entry_premium = float(sig["premium"])
        score = sig.get("score", 85)
        contracts = score_to_contracts(score, entry_premium)
        if contracts <= 0:
            continue

        direction = (sig.get("sentiment") or sig.get("direction") or "bullish").lower()
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        # Baseline
        bl = simulate_baseline(df, entry_premium, contracts, direction, dte, expiry_date, ticker)
        if bl is None:
            continue
        bl["ticker"] = ticker
        bl["contracts"] = contracts
        results_bl.append(bl)

        # Ceiling (per-ticker data)
        ceil_cfg = get_ceiling(ticker)
        ceil = simulate_ceiling(df, entry_premium, contracts, direction, dte, expiry_date,
                                ticker, ceil_cfg)
        if ceil:
            ceil["ticker"] = ticker
            ceil["contracts"] = contracts
            results_ceil.append(ceil)

        # Variants (override trail_above for all tickers)
        for name, trail_w in variants:
            var_cfg = {**ceil_cfg, "trail_above": trail_w}
            var = simulate_ceiling(df, entry_premium, contracts, direction, dte, expiry_date,
                                   ticker, var_cfg)
            if var:
                var["ticker"] = ticker
                var["contracts"] = contracts
                variant_results[name].append(var)

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(signals)}, matched {matched}")

    hconn.close()

    # --- Report ---
    print(f"\n{'=' * 100}")
    print(f"RESULTS: {matched} signals matched")
    print(f"{'=' * 100}")

    def summarize(results, label):
        if not results:
            return 0
        pnls = [r["pnl"] for r in results]
        wins = [r for r in results if r["pnl"] > 0]
        total = sum(pnls)
        wr = len(wins) / len(results) * 100
        print(f"\n  {label}")
        print(f"    Total P&L: ${total:>10,.0f} | Trades: {len(results)} | Win rate: {wr:.1f}%")
        print(f"    Avg: ${np.mean(pnls):>8,.0f} | Median: ${np.median(pnls):>8,.0f} | "
              f"Best: ${max(pnls):>8,.0f} | Worst: ${min(pnls):>8,.0f}")
        return total

    bl_total = summarize(results_bl, "A) Baseline FSM (production)")
    ceil_total = summarize(results_ceil, "B) Runner Ceiling (per-ticker tighten)")

    diff = ceil_total - bl_total
    print(f"\n  Delta (B - A): ${diff:>+10,.0f}")

    # Variants
    print(f"\n{'=' * 100}")
    print("TRAIL WIDTH VARIANTS (trail_above_ceiling parameter)")
    print(f"{'=' * 100}")
    for name, trail_w in variants:
        vr = variant_results[name]
        v_total = sum(r["pnl"] for r in vr)
        v_diff = v_total - bl_total
        tightened_count = sum(1 for r in vr if r.get("tightened"))
        print(f"  trail={trail_w}%: ${v_total:>10,.0f} (Δ ${v_diff:>+8,.0f}) | "
              f"{tightened_count} trades tightened")

    # --- Per-ticker breakdown ---
    print(f"\n{'=' * 100}")
    print("PER-TICKER DELTA (Ceiling vs Baseline)")
    print(f"{'=' * 100}")

    ticker_deltas = {}
    for bl_r, ceil_r in zip(results_bl, results_ceil):
        tk = bl_r["ticker"]
        d = ceil_r["pnl"] - bl_r["pnl"]
        if tk not in ticker_deltas:
            ceil_info = get_ceiling(tk)
            ticker_deltas[tk] = {
                "deltas": [], "trades": 0, "tightened": 0,
                "ceiling": ceil_info["ceiling"], "tighten_at": ceil_info["tighten_at"],
            }
        ticker_deltas[tk]["deltas"].append(d)
        ticker_deltas[tk]["trades"] += 1
        if ceil_r.get("tightened"):
            ticker_deltas[tk]["tightened"] += 1

    for tk, data in sorted(ticker_deltas.items(), key=lambda x: sum(x[1]["deltas"]), reverse=True):
        total_d = sum(data["deltas"])
        print(f"  {tk:<8} (ceil={data['ceiling']:>3}%, tight@{data['tighten_at']:>3}%): "
              f"Δ ${total_d:>+8,.0f} | {data['trades']:>2} trades, {data['tightened']} tightened")

    # --- Show trades where ceiling helped/hurt the most ---
    print(f"\n{'=' * 100}")
    print("BIGGEST IMPACTS (top 10 helped + top 10 hurt)")
    print(f"{'=' * 100}")

    impacts = []
    for bl_r, ceil_r in zip(results_bl, results_ceil):
        d = ceil_r["pnl"] - bl_r["pnl"]
        if abs(d) > 1:
            impacts.append({
                "ticker": bl_r["ticker"],
                "delta": d,
                "bl_pnl": bl_r["pnl"],
                "ceil_pnl": ceil_r["pnl"],
                "bl_reason": bl_r["reason"],
                "ceil_reason": ceil_r["reason"],
                "bl_exit_gain": bl_r["exit_gain"],
                "ceil_exit_gain": ceil_r["exit_gain"],
                "peak_gain": bl_r["peak_gain"],
                "tightened": ceil_r.get("tightened", False),
            })

    helped = sorted([i for i in impacts if i["delta"] > 0], key=lambda x: -x["delta"])[:10]
    hurt = sorted([i for i in impacts if i["delta"] < 0], key=lambda x: x["delta"])[:10]

    print(f"\n  HELPED ({len(helped)} shown):")
    for e in helped:
        tk_ceil = get_ceiling(e["ticker"])
        print(f"    {e['ticker']:<8} {e['delta']:>+8,.0f} | "
              f"peak +{e['peak_gain']:.0f}% | "
              f"BL exit +{e['bl_exit_gain']:.0f}% ({e['bl_reason']}) → "
              f"Ceil exit +{e['ceil_exit_gain']:.0f}% ({e['ceil_reason']}) "
              f"[ceil={tk_ceil['ceiling']}%]")

    print(f"\n  HURT ({len(hurt)} shown):")
    for e in hurt:
        tk_ceil = get_ceiling(e["ticker"])
        print(f"    {e['ticker']:<8} {e['delta']:>+8,.0f} | "
              f"peak +{e['peak_gain']:.0f}% | "
              f"BL exit +{e['bl_exit_gain']:.0f}% ({e['bl_reason']}) → "
              f"Ceil exit +{e['ceil_exit_gain']:.0f}% ({e['ceil_reason']}) "
              f"[ceil={tk_ceil['ceiling']}%]")


if __name__ == "__main__":
    main()
