"""Compare exit strategy tweaks side-by-side against real signals + harvester tick data.

Tests several ideas:
  A. BASELINE — current V5 config (per-ticker)
  B. TIGHTER_KEEP — soft trail keeps 70% of gains instead of 60%
  C. EARLY_NEVER_GREEN — if no +5% in 10min, tighten stop to 25%
  D. BREAKEVEN_RATCHET — once +20%, floor = entry (can't lose)
  E. COMBINED — B + C + D together

Usage:
    python scripts/backtest_tweak_compare.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    V5Config, AdaptiveTier, TICKER_CONFIGS, categorize_ticker, TickerCategory,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000


# ── Data loading ─────────────────────────────────────────────────────────────


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


def size_contracts(entry_premium, score):
    max_risk_pct = 0.75
    max_concurrent = 5
    max_position_pct = 0.15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent

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
    raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap = int(PORTFOLIO * max_position_pct / cost_per) if cost_per > 0 else 1
    return max(1, min(raw_contracts, pos_cap, 20))


# ── Simulation with custom logic ────────────────────────────────────────────


@dataclass
class TweakConfig:
    """Configuration for a backtest variant."""
    name: str
    # Soft trail keep % (default 0.60)
    soft_trail_keep_pct: float = 0.60
    # Early never-green: if not +X% by Y minutes, use tighter stop
    early_ng_enabled: bool = False
    early_ng_threshold_pct: float = 5.0   # must reach this gain
    early_ng_window_min: float = 10.0     # within this many minutes
    early_ng_stop_pct: float = 25.0       # else tighten stop to this
    # Breakeven ratchet: once +X%, floor = entry
    breakeven_enabled: bool = False
    breakeven_trigger_pct: float = 20.0
    # Use per-ticker configs
    use_per_ticker: bool = True


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker, tweak: TweakConfig):
    """Run production FSM with optional tweaks applied."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

    # Get config (per-ticker or default)
    if tweak.use_per_ticker and ticker in TICKER_CONFIGS:
        cfg = TICKER_CONFIGS[ticker]
    else:
        cfg = V5Config()

    # Apply soft trail keep override
    if tweak.soft_trail_keep_pct != 0.60:
        cfg = replace(cfg, soft_trail_keep_pct=tweak.soft_trail_keep_pct)

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

    ever_hit_threshold = False  # for early never-green
    breakeven_active = False    # for breakeven ratchet

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

        # Compute minutes held
        elapsed_min = (now - entry_ts).total_seconds() / 60.0
        gain_pct = (premium - entry_premium) / entry_premium * 100

        # Track if we ever hit the early NG threshold
        if gain_pct >= tweak.early_ng_threshold_pct:
            ever_hit_threshold = True

        # Breakeven ratchet check
        if tweak.breakeven_enabled and gain_pct >= tweak.breakeven_trigger_pct:
            breakeven_active = True

        # ── Custom tweak exits (checked BEFORE FSM) ──

        # Early never-green exit
        if (tweak.early_ng_enabled
            and not ever_hit_threshold
            and elapsed_min >= tweak.early_ng_window_min):
            # Past the window, never hit threshold — use tighter stop
            drop = (entry_premium - premium) / entry_premium * 100
            if drop >= tweak.early_ng_stop_pct:
                peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
                pnl = (premium - entry_premium) * contracts * 100
                return {
                    "pnl": pnl,
                    "reason": "early_ng_stop",
                    "hold": elapsed_min,
                    "exit_prem": premium,
                    "peak_gain": peak_gain,
                }

        # Breakeven ratchet — exit if premium drops back to entry
        if breakeven_active and premium <= entry_premium:
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = (premium - entry_premium) * contracts * 100
            return {
                "pnl": pnl,
                "reason": "breakeven_ratchet",
                "hold": elapsed_min,
                "exit_prem": premium,
                "peak_gain": peak_gain,
            }

        # ── Run production FSM ──
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

    # End of data
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


TWEAKS = [
    TweakConfig(name="A_BASELINE"),
    TweakConfig(name="B_KEEP70", soft_trail_keep_pct=0.70),
    TweakConfig(name="C_EARLY_NG",
                early_ng_enabled=True,
                early_ng_threshold_pct=5.0,
                early_ng_window_min=10.0,
                early_ng_stop_pct=25.0),
    TweakConfig(name="D_BREAKEVEN",
                breakeven_enabled=True,
                breakeven_trigger_pct=20.0),
    TweakConfig(name="E_COMBINED",
                soft_trail_keep_pct=0.70,
                early_ng_enabled=True,
                early_ng_threshold_pct=5.0,
                early_ng_window_min=10.0,
                early_ng_stop_pct=25.0,
                breakeven_enabled=True,
                breakeven_trigger_pct=20.0),
    # F: More aggressive breakeven at +15%
    TweakConfig(name="F_BE15_KEEP75",
                soft_trail_keep_pct=0.75,
                breakeven_enabled=True,
                breakeven_trigger_pct=15.0),
    # G: Early NG with wider window (15 min) and lower threshold
    TweakConfig(name="G_SLOW_NG",
                early_ng_enabled=True,
                early_ng_threshold_pct=3.0,
                early_ng_window_min=15.0,
                early_ng_stop_pct=20.0,
                soft_trail_keep_pct=0.70),
]


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load tick data for all signals (shared across tweaks)
    trade_data = []
    no_data = 0
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        contracts = size_contracts(adj_entry, score)

        trade_data.append({
            "df": df,
            "entry_premium": adj_entry,
            "contracts": contracts,
            "direction": direction,
            "dte": sig.get("_dte", 0),
            "expiry_date": sig.get("_expiry_date", ""),
            "ticker": ticker,
            "score": score,
            "day": sig["created_at"][:10],
        })

    harvester_conn.close()
    print(f"Loaded tick data for {len(trade_data)} trades ({no_data} skipped)\n")

    # Run each tweak
    all_results = {}
    for tweak in TWEAKS:
        results = []
        for td in trade_data:
            r = simulate_trade(
                td["df"], td["entry_premium"], td["contracts"],
                td["direction"], td["dte"], td["expiry_date"],
                td["ticker"], tweak,
            )
            r.update({
                "ticker": td["ticker"],
                "day": td["day"],
                "score": td["score"],
                "entry": td["entry_premium"],
                "contracts": td["contracts"],
                "direction": td["direction"],
            })
            results.append(r)
        all_results[tweak.name] = results

    # ── Summary comparison ───────────────────────────────────────────────

    print(f"{'=' * 100}")
    print(f"STRATEGY COMPARISON — {len(trade_data)} trades")
    print(f"{'=' * 100}")
    print(f"\n{'Strategy':<18} {'P&L':>10} {'Win%':>6} {'W/L':>8} {'AvgWin':>9} "
          f"{'AvgLoss':>9} {'MaxWin':>9} {'MaxLoss':>10} {'AvgHold':>8}")
    print("-" * 100)

    for name, results in all_results.items():
        pnls = pd.Series([r["pnl"] for r in results])
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        wr = wins / len(pnls) * 100
        avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
        avg_loss = pnls[pnls <= 0].mean() if losses > 0 else 0
        avg_hold = np.mean([r["hold"] for r in results])

        print(f"{name:<18} ${pnls.sum():>8,.0f} {wr:>5.1f}% "
              f"{wins:>3}/{losses:<3} ${avg_win:>7,.0f} ${avg_loss:>7,.0f} "
              f"${pnls.max():>7,.0f} ${pnls.min():>8,.0f} {avg_hold:>5.0f}m")

    # ── Exit reason breakdown for best vs baseline ───────────────────────

    baseline_pnl = sum(r["pnl"] for r in all_results["A_BASELINE"])
    best_name = max(all_results.keys(), key=lambda k: sum(r["pnl"] for r in all_results[k]))

    print(f"\n{'=' * 100}")
    print(f"EXIT REASON COMPARISON: A_BASELINE vs {best_name}")
    print(f"{'=' * 100}")

    for label, results in [("A_BASELINE", all_results["A_BASELINE"]),
                           (best_name, all_results[best_name])]:
        print(f"\n  [{label}]")
        df_r = pd.DataFrame(results)
        print(f"  {'Reason':<22} {'Count':>6} {'Total P&L':>10} {'Avg P&L':>9} {'Win%':>6}")
        print(f"  {'-' * 58}")
        for reason, group in df_r.groupby("reason"):
            gpnl = group["pnl"]
            gwins = (gpnl > 0).sum()
            gwr = gwins / len(gpnl) * 100 if len(gpnl) > 0 else 0
            print(f"  {reason:<22} {len(gpnl):>6} ${gpnl.sum():>8,.0f} "
                  f"${gpnl.mean():>7,.0f} {gwr:>5.0f}%")

    # ── Trade-by-trade delta: where did the best strategy differ? ────────

    print(f"\n{'=' * 100}")
    print(f"TRADE-BY-TRADE DIFFERENCES: A_BASELINE vs {best_name} (showing changed trades)")
    print(f"{'=' * 100}")
    print(f"\n{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Base$':>8} {'Base Reason':<20} "
          f"{'New$':>8} {'New Reason':<20} {'Delta':>8}")
    print("-" * 105)

    baseline_results = all_results["A_BASELINE"]
    best_results = all_results[best_name]
    total_delta = 0
    diffs = []
    for i in range(len(baseline_results)):
        br = baseline_results[i]
        nr = best_results[i]
        if abs(br["pnl"] - nr["pnl"]) > 1.0:  # only show meaningful differences
            delta = nr["pnl"] - br["pnl"]
            diffs.append((delta, i, br, nr))
            total_delta += delta

    # Sort by delta (biggest improvements first)
    diffs.sort(key=lambda x: x[0], reverse=True)
    for delta, i, br, nr in diffs[:30]:
        print(f"{br['day']:<12} {br['ticker']:<6} {br['direction'][:4]:<5} "
              f"${br['pnl']:>7,.0f} {br['reason']:<20} "
              f"${nr['pnl']:>7,.0f} {nr['reason']:<20} "
              f"${delta:>+7,.0f}")

    print(f"\nTotal delta: ${total_delta:+,.0f} across {len(diffs)} changed trades")


if __name__ == "__main__":
    main()
