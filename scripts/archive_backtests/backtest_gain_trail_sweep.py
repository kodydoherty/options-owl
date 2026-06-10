"""Sweep gain-based trail thresholds to find optimal values.

Tests a wide grid of trail widths for each tier (ACTIVE/RUNNER/MOONSHOT)
using gain-based math (peak_gain - current_gain >= threshold).

Also tests DCA mitigation:
  - No DCA
  - DCA with "never-profitable" guard (only DCA if trade was NEVER above +15%)
  - DCA with max-loss cap (kill trade entirely if down > 25% before DCA)

Usage:
    python scripts/backtest_gain_trail_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import (
    V5Config, get_ticker_config, AdaptiveTier, TickerCategory,
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


# ── Data loading (same as production backtest) ───────────────────────────────

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


def size_contracts(score, entry_premium):
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 4
    position_cap = PORTFOLIO * 0.15
    score_mult = 0.25
    for threshold, mult in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break
    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw = int(scaled_target / cost_per) if cost_per > 0 else 1
    cap = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw, cap))


def momentum_blocked(df, direction):
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    ups = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            ups.append(float(u))
    if len(ups) < 5:
        return False
    h1, h2 = ups[:len(ups)//2], ups[len(ups)//2:]
    pct = (sum(h2)/len(h2) - sum(h1)/len(h1)) / (sum(h1)/len(h1)) * 100
    ps = df["premium"].iloc[0]
    p5 = df["premium"].iloc[min(4, len(df)-1)]
    pf = (p5 - ps) / ps * 100 if ps > 0 else 0
    neg = 0
    if is_call and pct < -0.05:
        neg += 1
    elif not is_call and pct > 0.05:
        neg += 1
    if pf < -5:
        neg += 1
    against = 0
    for i in range(max(0, window-3), window):
        if i == 0:
            continue
        pu, cu = df["underlying_price"].iloc[i-1], df["underlying_price"].iloc[i]
        if pu and cu:
            if is_call and cu < pu:
                against += 1
            elif not is_call and cu > pu:
                against += 1
    if against >= 3:
        neg += 1
    return neg >= 2


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker="SIM",
             gain_trail_active=None, gain_trail_runner=None, gain_trail_moon=None,
             dca_mode="none"):
    """Run FSM with optional gain-based trail thresholds per tier.

    gain_trail_active/runner/moon: if set (float), override adaptive trail
    for that tier to use gain-based math with this threshold.
    If all None, runs pure baseline (premium-based).

    dca_mode: "none" | "always" | "smart"
      smart = only DCA if peak gain was < 15% (trade never really worked)
              AND current drawdown is < 30% (not too deep already)
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False}

    use_gain = gain_trail_active is not None
    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
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
    dca_fired = False
    max_gain_seen = 0.0
    original_entry = entry_premium

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
        elapsed_min = (now - entry_ts).total_seconds() / 60

        gain = (premium - entry_premium) / entry_premium * 100
        max_gain_seen = max(max_gain_seen, gain)

        # ── DCA ──────────────────────────────────────────────────────────
        if not dca_fired and dca_mode != "none" and elapsed_min >= 5 and remaining >= 1:
            dip_pct = (entry_premium - premium) / entry_premium * 100
            if 15 <= dip_pct <= 35:
                should_dca = False
                if dca_mode == "always":
                    should_dca = True
                elif dca_mode == "smart":
                    # Only DCA if trade never really worked (peak < 15%)
                    # AND not already too deep (dip < 30%)
                    if max_gain_seen < 15 and dip_pct < 30:
                        should_dca = True

                if should_dca:
                    dca_fired = True
                    dca_qty = remaining
                    total_cost = entry_premium * remaining + premium * dca_qty
                    remaining += dca_qty
                    entry_premium = total_cost / remaining
                    state.entry_premium = entry_premium
                    state.contracts = remaining
                    state.peak_premium = max(state.peak_premium, entry_premium)
                    continue

        # ── Gain-based trail check (before FSM) ─────────────────────────
        if use_gain:
            peak_gain = (state.peak_premium - state.entry_premium) / state.entry_premium * 100
            current_gain = (premium - state.entry_premium) / state.entry_premium * 100
            gain_drop = peak_gain - current_gain

            # Apply 2PM tightening to gain thresholds too
            tighten = 1.0
            if et_hour >= 14:
                tighten = 0.7

            # Check tiers highest first (moonshot > runner > active)
            gt_fired = False
            gt_reason = ""
            if gain_trail_moon is not None and peak_gain >= 400:
                threshold = gain_trail_moon * tighten
                if gain_drop >= threshold:
                    gt_fired = True
                    gt_reason = f"gain_moon({peak_gain:.0f}% peak, -{gain_drop:.0f}pts >= {threshold:.0f})"
            if not gt_fired and gain_trail_runner is not None and peak_gain >= 150:
                threshold = gain_trail_runner * tighten
                if gain_drop >= threshold:
                    gt_fired = True
                    gt_reason = f"gain_runner({peak_gain:.0f}% peak, -{gain_drop:.0f}pts >= {threshold:.0f})"
            if not gt_fired and gain_trail_active is not None and peak_gain >= 40:
                threshold = gain_trail_active * tighten
                if gain_drop >= threshold:
                    gt_fired = True
                    gt_reason = f"gain_active({peak_gain:.0f}% peak, -{gain_drop:.0f}pts >= {threshold:.0f})"

            if gt_fired:
                pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                return {
                    "pnl": pnl,
                    "reason": "gain_trail",
                    "hold": elapsed_min,
                    "exit_prem": premium,
                    "peak_gain": peak_gain,
                    "dca_fired": dca_fired,
                }

        # ── Normal FSM ───────────────────────────────────────────────────
        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
        )

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            peak_gain = (state.peak_premium - state.entry_premium) / state.entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "hold": elapsed_min,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "dca_fired": dca_fired,
            }

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
        "exit_prem": last_prem, "peak_gain": peak_gain, "dca_fired": dca_fired,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Pre-load all trade data
    trades = []
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
        contracts = size_contracts(score, adj_entry)
        blocked = momentum_blocked(df, direction)
        trades.append({
            "ticker": ticker, "direction": direction, "score": score,
            "day": sig["created_at"][:10], "df": df, "entry": adj_entry,
            "contracts": contracts, "dte": dte, "expiry_date": expiry_date,
            "blocked": blocked,
        })

    harvester_conn.close()
    print(f"{len(trades)} trades loaded, {no_data} skipped (no tick data)")

    # ── Phase 1: Sweep gain trail thresholds ─────────────────────────────

    # Active tier: how many gain points to give back before selling (peak 40%+)
    active_range = [20, 25, 30, 35, 40, 50, 60, 70, 80]
    # Runner tier: peak 150%+
    runner_range = [40, 50, 60, 70, 80, 90, 100, 120]
    # Moonshot tier: peak 400%+
    moon_range = [60, 80, 100, 120, 150, 180, 200]

    print(f"\n{'=' * 110}")
    print(f"PHASE 1: GAIN TRAIL THRESHOLD SWEEP ({len(active_range)}x{len(runner_range)}x{len(moon_range)} = "
          f"{len(active_range)*len(runner_range)*len(moon_range)} combos)")
    print(f"{'=' * 110}")

    # First run baseline
    baseline_results = []
    for t in trades:
        if t["blocked"]:
            baseline_results.append({"pnl": 0, "reason": "momentum_blocked",
                                     "peak_gain": 0, "dca_fired": False,
                                     "ticker": t["ticker"], "day": t["day"]})
            continue
        r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                     t["dte"], t["expiry_date"], ticker=t["ticker"], dca_mode="none")
        r["ticker"] = t["ticker"]
        r["day"] = t["day"]
        baseline_results.append(r)

    baseline_pnl = sum(r["pnl"] for r in baseline_results)
    print(f"\nBaseline (premium trail, no DCA): ${baseline_pnl:,.0f}")

    # Sweep — but be smart: first sweep active alone, then runner, then moon
    # to narrow the search space

    # Step 1: Find best active threshold (fix runner=80, moon=150)
    print(f"\nStep 1: Sweep ACTIVE tier (runner=80, moon=150)")
    print(f"{'Active':>8} {'Total P&L':>12} {'Delta':>10} {'Win%':>6}")
    print("-" * 42)

    best_active = 40
    best_active_pnl = -999999
    for act in active_range:
        total = 0
        wins = 0
        n = 0
        for t in trades:
            if t["blocked"]:
                n += 1
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"],
                         gain_trail_active=act, gain_trail_runner=80,
                         gain_trail_moon=150, dca_mode="none")
            total += r["pnl"]
            if r["pnl"] > 0:
                wins += 1
            n += 1
        wr = wins / n * 100 if n > 0 else 0
        delta = total - baseline_pnl
        marker = " <-- best" if total > best_active_pnl else ""
        print(f"{act:>8} ${total:>10,.0f} ${delta:>+9,.0f} {wr:>5.1f}%{marker}")
        if total > best_active_pnl:
            best_active_pnl = total
            best_active = act

    # Step 2: Find best runner threshold (fix active=best, moon=150)
    print(f"\nStep 2: Sweep RUNNER tier (active={best_active}, moon=150)")
    print(f"{'Runner':>8} {'Total P&L':>12} {'Delta':>10} {'Win%':>6}")
    print("-" * 42)

    best_runner = 80
    best_runner_pnl = -999999
    for run in runner_range:
        total = 0
        wins = 0
        n = 0
        for t in trades:
            if t["blocked"]:
                n += 1
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"],
                         gain_trail_active=best_active, gain_trail_runner=run,
                         gain_trail_moon=150, dca_mode="none")
            total += r["pnl"]
            if r["pnl"] > 0:
                wins += 1
            n += 1
        wr = wins / n * 100 if n > 0 else 0
        delta = total - baseline_pnl
        marker = " <-- best" if total > best_runner_pnl else ""
        print(f"{run:>8} ${total:>10,.0f} ${delta:>+9,.0f} {wr:>5.1f}%{marker}")
        if total > best_runner_pnl:
            best_runner_pnl = total
            best_runner = run

    # Step 3: Find best moonshot threshold
    print(f"\nStep 3: Sweep MOONSHOT tier (active={best_active}, runner={best_runner})")
    print(f"{'Moon':>8} {'Total P&L':>12} {'Delta':>10} {'Win%':>6}")
    print("-" * 42)

    best_moon = 150
    best_moon_pnl = -999999
    for moon in moon_range:
        total = 0
        wins = 0
        n = 0
        for t in trades:
            if t["blocked"]:
                n += 1
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"],
                         gain_trail_active=best_active, gain_trail_runner=best_runner,
                         gain_trail_moon=moon, dca_mode="none")
            total += r["pnl"]
            if r["pnl"] > 0:
                wins += 1
            n += 1
        wr = wins / n * 100 if n > 0 else 0
        delta = total - baseline_pnl
        marker = " <-- best" if total > best_moon_pnl else ""
        print(f"{moon:>8} ${total:>10,.0f} ${delta:>+9,.0f} {wr:>5.1f}%{marker}")
        if total > best_moon_pnl:
            best_moon_pnl = total
            best_moon = moon

    print(f"\nBest gain trail: active={best_active}, runner={best_runner}, moon={best_moon}")
    print(f"Best P&L: ${best_moon_pnl:,.0f} (delta ${best_moon_pnl - baseline_pnl:+,.0f} vs baseline)")

    # ── Phase 2: Fine-tune around the best values ────────────────────────

    print(f"\n{'=' * 110}")
    print(f"PHASE 2: FINE-TUNE AROUND BEST ({best_active}/{best_runner}/{best_moon})")
    print(f"{'=' * 110}")

    fine_active = [max(10, best_active - 10), best_active - 5, best_active,
                   best_active + 5, best_active + 10]
    fine_runner = [max(20, best_runner - 15), best_runner - 10, best_runner - 5,
                   best_runner, best_runner + 5, best_runner + 10, best_runner + 15]
    fine_moon = [max(30, best_moon - 20), best_moon - 10, best_moon,
                 best_moon + 10, best_moon + 20]

    top_combos = []
    for act, run, moon in product(fine_active, fine_runner, fine_moon):
        if run <= act:
            continue  # runner should be wider than active
        if moon <= run:
            continue  # moonshot should be wider than runner
        total = 0
        wins = 0
        losses = 0
        max_loss = 0
        for t in trades:
            if t["blocked"]:
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"],
                         gain_trail_active=act, gain_trail_runner=run,
                         gain_trail_moon=moon, dca_mode="none")
            total += r["pnl"]
            if r["pnl"] > 0:
                wins += 1
            else:
                losses += 1
            max_loss = min(max_loss, r["pnl"])

        top_combos.append({
            "active": act, "runner": run, "moon": moon,
            "pnl": total, "wins": wins, "losses": losses,
            "max_loss": max_loss, "delta": total - baseline_pnl,
        })

    top_combos.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n{'Active':>7} {'Runner':>7} {'Moon':>7} {'Total P&L':>12} {'Delta':>10} "
          f"{'Win%':>6} {'MaxLoss':>10}")
    print("-" * 70)
    for c in top_combos[:20]:
        n = c["wins"] + c["losses"]
        wr = c["wins"] / n * 100 if n > 0 else 0
        print(f"{c['active']:>7} {c['runner']:>7} {c['moon']:>7} "
              f"${c['pnl']:>10,.0f} ${c['delta']:>+9,.0f} {wr:>5.1f}% ${c['max_loss']:>8,.0f}")

    # ── Phase 3: Test DCA strategies with best gain trail ────────────────

    best = top_combos[0]
    ba, br, bm = best["active"], best["runner"], best["moon"]

    print(f"\n{'=' * 110}")
    print(f"PHASE 3: DCA STRATEGIES (gain trail {ba}/{br}/{bm})")
    print(f"{'=' * 110}")

    dca_modes = {
        "no_dca": "none",
        "dca_always": "always",
        "dca_smart": "smart",
    }

    for label, mode in dca_modes.items():
        results = []
        dca_count = 0
        for t in trades:
            if t["blocked"]:
                results.append({"pnl": 0, "dca_fired": False,
                                "ticker": t["ticker"], "day": t["day"],
                                "peak_gain": 0, "reason": "blocked"})
                continue
            r = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                         t["dte"], t["expiry_date"], ticker=t["ticker"],
                         gain_trail_active=ba, gain_trail_runner=br,
                         gain_trail_moon=bm, dca_mode=mode)
            r["ticker"] = t["ticker"]
            r["day"] = t["day"]
            results.append(r)
            if r.get("dca_fired"):
                dca_count += 1

        pnls = [r["pnl"] for r in results]
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        wr = wins / len(pnls) * 100
        max_l = min(pnls)

        print(f"\n{label}: ${total:,.0f} (delta ${total - baseline_pnl:+,.0f}) | "
              f"WR {wr:.1f}% ({wins}W/{losses}L) | MaxLoss ${max_l:,.0f} | DCA events: {dca_count}")

        # Show DCA trades detail
        if dca_count > 0:
            dca_trades = [r for r in results if r.get("dca_fired")]
            dca_pnl = sum(r["pnl"] for r in dca_trades)
            dca_wins = sum(1 for r in dca_trades if r["pnl"] > 0)
            print(f"  DCA trade totals: ${dca_pnl:,.0f} ({dca_wins}/{dca_count} wins)")

            # Compare with no-DCA version
            no_dca_pnl = 0
            for t in trades:
                if t["blocked"]:
                    continue
                # Check if this trade had DCA
                r_check = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                                   t["dte"], t["expiry_date"], ticker=t["ticker"],
                                   gain_trail_active=ba, gain_trail_runner=br,
                                   gain_trail_moon=bm, dca_mode=mode)
                if r_check.get("dca_fired"):
                    r_nodca = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                                       t["dte"], t["expiry_date"], ticker=t["ticker"],
                                       gain_trail_active=ba, gain_trail_runner=br,
                                       gain_trail_moon=bm, dca_mode="none")
                    no_dca_pnl += r_nodca["pnl"]

            print(f"  Same trades without DCA: ${no_dca_pnl:,.0f}")
            print(f"  DCA net impact: ${dca_pnl - no_dca_pnl:+,.0f}")

    # ── Phase 4: Per-trade comparison (best gain trail vs baseline) ──────

    print(f"\n{'=' * 110}")
    print(f"PHASE 4: BEST GAIN TRAIL ({ba}/{br}/{bm}) vs BASELINE — trade-by-trade")
    print(f"{'=' * 110}")

    diff_trades = []
    for i, t in enumerate(trades):
        if t["blocked"]:
            continue
        b = baseline_results[i]
        g = simulate(t["df"], t["entry"], t["contracts"], t["direction"],
                     t["dte"], t["expiry_date"], ticker=t["ticker"],
                     gain_trail_active=ba, gain_trail_runner=br,
                     gain_trail_moon=bm, dca_mode="none")
        if abs(b["pnl"] - g["pnl"]) > 1:
            diff_trades.append({
                "day": t["day"], "ticker": t["ticker"],
                "peak": b["peak_gain"],
                "base_pnl": b["pnl"], "base_reason": b["reason"],
                "gain_pnl": g["pnl"], "gain_reason": g["reason"],
                "delta": g["pnl"] - b["pnl"],
            })

    if diff_trades:
        diff_trades.sort(key=lambda x: x["delta"])
        print(f"\n{'Day':<12} {'Tkr':<6} {'Peak%':>6} {'Base PnL':>10} {'Base Exit':>20} "
              f"{'Gain PnL':>10} {'Gain Exit':>20} {'Delta':>8}")
        print("-" * 105)
        for d in diff_trades:
            print(f"{d['day']:<12} {d['ticker']:<6} {d['peak']:>5.0f}% "
                  f"${d['base_pnl']:>8,.0f} {d['base_reason']:<20} "
                  f"${d['gain_pnl']:>8,.0f} {d['gain_reason']:<20} "
                  f"${d['delta']:>+7,.0f}")

        helped = sum(d["delta"] for d in diff_trades if d["delta"] > 0)
        hurt = sum(d["delta"] for d in diff_trades if d["delta"] < 0)
        print(f"\nHelped: ${helped:+,.0f} | Hurt: ${hurt:+,.0f} | Net: ${helped + hurt:+,.0f}")
    else:
        print("No trades differed!")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 110}")
    print(f"Baseline (premium trail):           ${baseline_pnl:,.0f}")
    print(f"Best gain trail ({ba}/{br}/{bm}):   ${best['pnl']:,.0f} (${best['delta']:+,.0f})")


if __name__ == "__main__":
    main()
