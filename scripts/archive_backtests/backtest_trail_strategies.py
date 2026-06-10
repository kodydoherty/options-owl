"""Backtest trail width + stepped scale-out strategies.

Compares multiple exit strategies on historical trades:
  A) Current production (55% HIGH_VOL RUNNER trail, single scaleout)
  B) Tighter trails (40%, 45% RUNNER)
  C) Stepped scale-out (sell 25% at each +50% gain tier)
  D) Hybrid: tighter trail + stepped scale-out

Uses the LIVE production FSM code against harvester tick data.

Usage:
    python scripts/backtest_trail_strategies.py
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
    AdaptiveTier,
    V5Config,
    TickerCategory,
    categorize_ticker,
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


# ── Strategy configs ─────────────────────────────────────────────────────────


@dataclass
class Strategy:
    name: str
    config: V5Config
    use_per_ticker: bool
    # Stepped scale-out: list of (gain_pct_threshold, fraction_to_sell)
    # e.g. [(50, 0.25), (100, 0.25), (200, 0.25)] = sell 25% at +50%, +100%, +200%
    scaleout_steps: list[tuple[float, float]] | None = None


def make_tighter_highvol(runner_trail: float) -> tuple[AdaptiveTier, ...]:
    """Make HIGH_VOL tiers with a custom RUNNER trail width."""
    return (
        AdaptiveTier(400, 35),       # moonshot: keep at 35%
        AdaptiveTier(150, runner_trail),  # runner: tighten from 55%
        AdaptiveTier(40, 50),        # active: keep at 50%
    )


STRATEGIES = [
    # ── Baseline ──
    Strategy(
        name="A) Production (wide trail, 1 scaleout)",
        config=V5Config(),
        use_per_ticker=True,
    ),

    # ── Progressive DCA-out: sell 1 contract at each milestone ──
    # Key idea: lock in profits in small bites, always keep a runner
    Strategy(
        name="B) Ladder every 30% (1 contract each)",
        config=V5Config(),
        use_per_ticker=True,
        # sell 1 contract (~15-20% of position) at each step
        # fraction is of ORIGINAL position, capped to never sell last contract
        scaleout_steps=[(30, 0.15), (60, 0.15), (100, 0.15), (150, 0.15), (250, 0.15)],
    ),
    Strategy(
        name="C) Ladder every 50% (1 contract each)",
        config=V5Config(),
        use_per_ticker=True,
        scaleout_steps=[(50, 0.15), (100, 0.15), (150, 0.15), (250, 0.15)],
    ),
    Strategy(
        name="D) Half at +50%, rest rides",
        config=V5Config(),
        use_per_ticker=True,
        scaleout_steps=[(50, 0.50)],
    ),
    Strategy(
        name="E) 1/3 at +40%, 1/3 at +100%, rest rides",
        config=V5Config(),
        use_per_ticker=True,
        scaleout_steps=[(40, 0.33), (100, 0.33)],
    ),
    Strategy(
        name="F) 1/4 at +30%, 1/4 at +75%, 1/4 at +150%",
        config=V5Config(),
        use_per_ticker=True,
        scaleout_steps=[(30, 0.25), (75, 0.25), (150, 0.25)],
    ),

    # ── Hybrid: progressive + slightly tighter trail ──
    Strategy(
        name="G) Ladder 30% + 45% runner trail",
        config=V5Config(
            adaptive_highvol_tiers=make_tighter_highvol(45),
        ),
        use_per_ticker=False,
        scaleout_steps=[(30, 0.15), (60, 0.15), (100, 0.15), (150, 0.15), (250, 0.15)],
    ),
    Strategy(
        name="H) 1/3+1/3 + 45% runner trail",
        config=V5Config(
            adaptive_highvol_tiers=make_tighter_highvol(45),
        ),
        use_per_ticker=False,
        scaleout_steps=[(40, 0.33), (100, 0.33)],
    ),
]


# ── Simulation ───────────────────────────────────────────────────────────────


def simulate_strategy(
    df: pd.DataFrame,
    entry_premium: float,
    contracts: int,
    direction: str,
    dte: int,
    expiry_date: str,
    ticker: str,
    strategy: Strategy,
) -> dict:
    """Run a strategy against tick data. Handles stepped scale-out manually."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "contracts_at_exit": 0}

    # Get config — per-ticker if enabled, else strategy-level config
    if strategy.use_per_ticker:
        cfg = get_ticker_config(ticker, use_per_ticker=True)
    else:
        cfg = strategy.config

    fsm = ExitFSM(cfg)
    option_type = "put" if direction in ("bearish", "put") else "call"

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

    # Stepped scale-out tracking
    remaining = contracts
    realized_pnl = 0.0
    scaleout_steps = list(strategy.scaleout_steps) if strategy.scaleout_steps else []
    steps_fired = set()

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
        et_hour = now.hour - 4
        if et_hour < 0:
            et_hour += 24
        minutes_to_close = max(0, (16 * 60) - (et_hour * 60 + now.minute))

        # Check stepped scale-out before FSM
        gain_pct = (premium - entry_premium) / entry_premium * 100
        for i, (threshold, fraction) in enumerate(scaleout_steps):
            if i not in steps_fired and gain_pct >= threshold and remaining > 1:
                sell_qty = max(1, int(contracts * fraction))
                sell_qty = min(sell_qty, remaining - 1)  # keep at least 1
                realized_pnl += (premium - entry_premium) * sell_qty * 100
                remaining -= sell_qty
                steps_fired.add(i)

        # Update contracts on state for FSM
        state.contracts = remaining

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            # Remaining contracts exit at current premium
            final_pnl = realized_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": final_pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "contracts_at_exit": remaining,
                "scaleouts_fired": len(steps_fired),
            }

    # End of data
    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    final_pnl = realized_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": final_pnl,
        "reason": "eod_data_end",
        "hold": elapsed,
        "exit_prem": last_prem,
        "peak_gain": peak_gain,
        "contracts_at_exit": remaining,
        "scaleouts_fired": len(steps_fired),
    }


# ── Sizing (matches production) ─────────────────────────────────────────────


def size_contracts(score, entry_premium):
    max_risk_pct = 0.75
    max_concurrent = 4    # production: MAX_CONCURRENT=4
    max_position_pct = 0.15  # production: MAX_POSITION_PCT=15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent
    position_cap = PORTFOLIO * max_position_pct

    SCORE_TIERS = [
        (135, 1.00),
        (120, 0.85),
        (100, 0.85),
        (90, 0.50),
        (85, 0.35),
        (78, 0.20),
    ]
    mult = 0.0
    for min_score, m in SCORE_TIERS:
        if score >= min_score:
            mult = m
            break
    if mult <= 0 or score < 78:
        return 0

    scaled = per_slot * mult
    cost_per = entry_premium * 100
    if cost_per <= 0:
        return 0
    raw = int(scaled / cost_per)
    cap = int(position_cap / cost_per)
    return max(1, min(raw, cap))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load ticks once per signal
    signal_ticks = []
    no_data = 0
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue
        signal_ticks.append((sig, df))
    harvester_conn.close()

    print(f"Signals with tick data: {len(signal_ticks)} (no data: {no_data})")
    print()

    # Run each strategy
    all_results = {}
    for strat in STRATEGIES:
        results = []
        for sig, df in signal_ticks:
            ticker = sig["ticker"]
            direction = (sig["direction"] or "bullish").lower()
            score = sig["score"] or 80
            entry_premium = sig["premium"]
            dte = sig.get("_dte", 0)
            expiry_date = sig.get("_expiry_date", "")

            first_ask = df["ask"].iloc[0]
            first_mid = df["premium"].iloc[0]
            adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
            if adj_entry <= 0:
                adj_entry = entry_premium

            contracts = size_contracts(score, adj_entry)
            if contracts <= 0:
                continue

            r = simulate_strategy(
                df, adj_entry, contracts, direction, dte, expiry_date, ticker, strat
            )
            r["ticker"] = ticker
            r["score"] = score
            r["day"] = sig["created_at"][:10]
            r["entry"] = adj_entry
            r["contracts"] = contracts
            r["category"] = categorize_ticker(ticker).value
            results.append(r)

        all_results[strat.name] = results

    # ── Summary ──────────────────────────────────────────────────────────────

    print("=" * 100)
    print(f"{'Strategy':<40s} {'Trades':>6s} {'Total P&L':>10s} {'Win%':>6s} "
          f"{'Avg P&L':>8s} {'Avg Hold':>8s} {'Peak>50%':>8s}")
    print("=" * 100)

    for strat_name, results in all_results.items():
        if not results:
            continue
        total_pnl = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100 if results else 0
        avg_pnl = total_pnl / len(results) if results else 0
        avg_hold = np.mean([r["hold"] for r in results])
        big_peaks = sum(1 for r in results if r["peak_gain"] >= 50)

        print(f"{strat_name:<40s} {len(results):>6d} ${total_pnl:>9,.0f} {wr:>5.1f}% "
              f"${avg_pnl:>7,.0f} {avg_hold:>7.0f}m {big_peaks:>8d}")

    # ── HIGH_VOL breakdown ───────────────────────────────────────────────────

    print()
    print("=" * 100)
    print("HIGH_VOL tickers only (MSTR, TSLA, NVDA, AVGO, META, AMD, etc.)")
    print("=" * 100)
    print(f"{'Strategy':<40s} {'Trades':>6s} {'Total P&L':>10s} {'Win%':>6s} "
          f"{'Avg P&L':>8s} {'Big Runs':>8s}")
    print("-" * 100)

    for strat_name, results in all_results.items():
        hv = [r for r in results if r["category"] == "high_vol"]
        if not hv:
            continue
        total_pnl = sum(r["pnl"] for r in hv)
        wins = sum(1 for r in hv if r["pnl"] > 0)
        wr = wins / len(hv) * 100
        avg_pnl = total_pnl / len(hv)
        big_runs = sum(1 for r in hv if r["peak_gain"] >= 100)

        print(f"{strat_name:<40s} {len(hv):>6d} ${total_pnl:>9,.0f} {wr:>5.1f}% "
              f"${avg_pnl:>7,.0f} {big_runs:>8d}")

    # ── Trades where peak > 50% (runners that matter) ────────────────────────

    print()
    print("=" * 100)
    print("Trades that peaked above +50% (the runners where trail width matters)")
    print("=" * 100)

    for strat_name, results in all_results.items():
        runners = [r for r in results if r["peak_gain"] >= 50]
        if not runners:
            continue
        total_pnl = sum(r["pnl"] for r in runners)
        avg_exit_vs_peak = np.mean([
            r["exit_prem"] / (r["entry"] * (1 + r["peak_gain"] / 100)) * 100
            for r in runners if r["peak_gain"] > 0
        ])
        print(f"{strat_name:<40s}  {len(runners)} trades  "
              f"P&L=${total_pnl:>8,.0f}  "
              f"Avg exit at {avg_exit_vs_peak:.0f}% of peak")

    # ── Per-trade detail for big runners ─────────────────────────────────────

    print()
    print("=" * 100)
    print("Per-trade detail: trades that peaked +100%+ (comparing all strategies)")
    print("=" * 100)

    # Find trades that were big runners in any strategy
    big_runner_keys = set()
    for strat_name, results in all_results.items():
        for r in results:
            if r["peak_gain"] >= 100:
                big_runner_keys.add((r["ticker"], r["day"]))

    if big_runner_keys:
        print(f"\n{'Ticker':<6s} {'Day':<12s} {'Strategy':<40s} "
              f"{'Peak':>6s} {'Exit$':>6s} {'P&L':>8s} {'Reason':<20s}")
        print("-" * 110)

        for ticker, day in sorted(big_runner_keys):
            for strat_name, results in all_results.items():
                for r in results:
                    if r["ticker"] == ticker and r["day"] == day:
                        print(f"{ticker:<6s} {day:<12s} {strat_name:<40s} "
                              f"+{r['peak_gain']:>4.0f}% ${r['exit_prem']:>5.2f} "
                              f"${r['pnl']:>7.0f} {r['reason']:<20s}")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
