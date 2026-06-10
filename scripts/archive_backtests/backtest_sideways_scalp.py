"""Backtest dynamic sideways detection + early profit-taking.

Idea: detect when a trade is choppy/sideways (not trending) and scalp
small profits early instead of waiting for a trend that never comes.

Sideways detection signals:
  1. Premium range-bound: high-low range within X% of entry over last N ticks
  2. No new highs: time since last peak exceeded threshold
  3. Underlying flat: underlying moved < Y% since entry
  4. Oscillation count: premium crossed entry price Z+ times

When sideways is detected AND trade is profitable, take profit at lower
thresholds (e.g. +8-15% instead of waiting for +20% breakeven ratchet).

Usage:
    python scripts/backtest_sideways_scalp.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config, AdaptiveTier
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
    h1, h2 = ups[:len(ups) // 2], ups[len(ups) // 2:]
    pct = (sum(h2) / len(h2) - sum(h1) / len(h1)) / (sum(h1) / len(h1)) * 100
    ps = df["premium"].iloc[0]
    p5 = df["premium"].iloc[min(4, len(df) - 1)]
    pf = (p5 - ps) / ps * 100 if ps > 0 else 0
    neg = 0
    if is_call and pct < -0.05:
        neg += 1
    elif not is_call and pct > 0.05:
        neg += 1
    if pf < -5:
        neg += 1
    against = 0
    for i in range(max(0, window - 3), window):
        if i == 0:
            continue
        pu, cu = df["underlying_price"].iloc[i - 1], df["underlying_price"].iloc[i]
        if pu and cu:
            if is_call and cu < pu:
                against += 1
            elif not is_call and cu > pu:
                against += 1
    if against >= 3:
        neg += 1
    return neg >= 2


def detect_sideways(premiums, timestamps, entry_premium, underlying_prices,
                    entry_underlying,
                    lookback=20, range_pct=10.0, no_new_high_min=8.0,
                    underlying_flat_pct=0.15, cross_count_thresh=3,
                    signals_needed=2):
    """Detect if a trade is in a sideways/choppy pattern.

    Returns (is_sideways, signals_hit, details_dict).

    timestamps: list of datetime objects aligned 1:1 with premiums.
    signals_needed: how many of the 4 indicators must be true to classify as sideways.
    """
    if len(premiums) < 5:
        return False, 0, {}

    window = premiums[-lookback:] if len(premiums) > lookback else premiums
    signals = 0
    details = {}

    # 1. Range-bound premium: (max-min)/entry < range_pct%
    prem_range = (max(window) - min(window)) / entry_premium * 100 if entry_premium > 0 else 999
    if prem_range < range_pct:
        signals += 1
    details["prem_range_pct"] = round(prem_range, 1)

    # 2. No new highs recently: use actual timestamps instead of assuming fixed tick interval
    peak_val = max(premiums)
    peak_idx = premiums.index(peak_val)
    if len(timestamps) == len(premiums) and peak_idx < len(timestamps):
        min_since_peak = (timestamps[-1] - timestamps[peak_idx]).total_seconds() / 60
    else:
        # Fallback: estimate from tick count
        ticks_since_peak = len(premiums) - 1 - peak_idx
        min_since_peak = ticks_since_peak * 15 / 60
    if min_since_peak >= no_new_high_min:
        signals += 1
    details["min_since_peak"] = round(min_since_peak, 1)

    # 3. Underlying flat: moved < Y% from entry
    if underlying_prices and entry_underlying > 0:
        latest_u = underlying_prices[-1]
        u_move = abs(latest_u - entry_underlying) / entry_underlying * 100
        if u_move < underlying_flat_pct:
            signals += 1
        details["underlying_move_pct"] = round(u_move, 2)
    else:
        details["underlying_move_pct"] = None

    # 4. Entry cross count: premium crossed entry price N+ times
    crosses = 0
    above = premiums[0] >= entry_premium
    for p in premiums[1:]:
        now_above = p >= entry_premium
        if now_above != above:
            crosses += 1
            above = now_above
    if crosses >= cross_count_thresh:
        signals += 1
    details["entry_crosses"] = crosses

    is_sideways = signals >= signals_needed
    return is_sideways, signals, details


def simulate(df, entry_premium, contracts, direction, dte, expiry_date,
             ticker="SIM", dca_mode="none",
             # Sideways scalp params
             sideways_scalp=False,
             sw_lookback=20,
             sw_range_pct=10.0,
             sw_no_new_high_min=8.0,
             sw_underlying_flat_pct=0.15,
             sw_cross_thresh=3,
             sw_signals_needed=2,
             sw_min_elapsed_min=5.0,
             sw_take_profit_pct=8.0):
    """Run a single trade simulation with optional sideways scalp.

    sideways_scalp: if True, detect sideways and take profit early.
    sw_take_profit_pct: take profit when gain >= this % AND sideways detected.
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False, "sideways_exit": False}

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
    premium_history = []
    timestamp_history = []
    underlying_history = []

    # DCA tracking
    dca_fired = False
    dca_time = None
    dca_fill_price = 0.0

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

        premium_history.append(float(premium))
        timestamp_history.append(now)
        # Keep underlying_history aligned 1:1 with premium_history
        underlying_history.append(float(underlying) if underlying and underlying > 0 else (underlying_history[-1] if underlying_history else first_underlying))

        # ── DCA (always mode, with 2PM gate from production) ────────────
        if not dca_fired and dca_mode != "none" and elapsed_min >= 5 and remaining >= 1:
            if et_hour < 14:  # 2PM ET gate
                dip_pct = (entry_premium - premium) / entry_premium * 100
                if 15 <= dip_pct <= 35:
                    dca_fired = True
                    dca_time = now
                    dca_fill_price = premium
                    dca_qty = remaining
                    total_cost = entry_premium * remaining + premium * dca_qty
                    remaining += dca_qty
                    entry_premium = total_cost / remaining
                    state.entry_premium = entry_premium
                    state.contracts = remaining
                    state.peak_premium = max(state.peak_premium, entry_premium)
                    continue

        # ── Sideways scalp check ────────────────────────────────────────
        # Only scalp trades that haven't trended significantly.
        # If peak gain exceeded 30%, the trade is/was trending — let adaptive trail handle it.
        if sideways_scalp and elapsed_min >= sw_min_elapsed_min and len(premium_history) >= 10:
            peak_gain_so_far = (max(premium_history) - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
            gain_pct = (premium - entry_premium) / entry_premium * 100
            if gain_pct >= sw_take_profit_pct and peak_gain_so_far < 30:
                is_sw, sw_sig, _ = detect_sideways(
                    premium_history, timestamp_history,
                    entry_premium,
                    underlying_history, first_underlying,
                    lookback=sw_lookback,
                    range_pct=sw_range_pct,
                    no_new_high_min=sw_no_new_high_min,
                    underlying_flat_pct=sw_underlying_flat_pct,
                    cross_count_thresh=sw_cross_thresh,
                    signals_needed=sw_signals_needed,
                )
                if is_sw:
                    pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                    pk = (state.peak_premium - entry_premium) / entry_premium * 100
                    return {
                        "pnl": pnl, "reason": "sideways_scalp",
                        "hold": elapsed_min, "exit_prem": premium,
                        "peak_gain": pk, "dca_fired": dca_fired,
                        "sideways_exit": True,
                    }

        # ── Normal FSM ──────────────────────────────────────────────────
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

            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            pk = (state.peak_premium - entry_premium) / entry_premium * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed_min, "exit_prem": premium,
                "peak_gain": pk, "dca_fired": dca_fired,
                "sideways_exit": False,
            }

    # End of data — close at last price
    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    pk = (state.peak_premium - entry_premium) / entry_premium * 100
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    hold_min = (last_ts - entry_ts).total_seconds() / 60
    return {
        "pnl": pnl, "reason": "eod_data_end",
        "hold": hold_min, "exit_prem": last_prem, "peak_gain": pk,
        "dca_fired": dca_fired, "sideways_exit": False,
    }


def run_scenario(signals, harvester_conn, label, dca_mode="always", **sw_kwargs):
    """Run all signals through a scenario, return list of trade results."""
    results = []
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is None or len(df) < 10:
            continue

        first_ask = df["ask"].iloc[0]
        entry_premium = float(first_ask) if first_ask and not pd.isna(first_ask) and first_ask > 0 else float(df["premium"].iloc[0])
        if entry_premium <= 0:
            continue

        direction = sig["option_type"]
        if momentum_blocked(df, direction):
            continue

        contracts = size_contracts(sig["score"], entry_premium)
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        res = simulate(
            df, entry_premium, contracts, direction, dte, expiry_date,
            ticker=sig["ticker"], dca_mode=dca_mode, **sw_kwargs,
        )
        res["ticker"] = sig["ticker"]
        res["date"] = sig["created_at"][:10]
        res["score"] = sig["score"]
        res["contracts"] = contracts
        res["entry_premium"] = entry_premium
        results.append(res)

    return results


def summarize(results, label):
    total_pnl = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    losses = sum(1 for r in results if r["pnl"] < 0)
    wr = wins / len(results) * 100 if results else 0
    sw_exits = sum(1 for r in results if r.get("sideways_exit"))
    dca_count = sum(1 for r in results if r.get("dca_fired"))
    max_loss = min((r["pnl"] for r in results), default=0)
    return {
        "label": label,
        "total_pnl": total_pnl,
        "trades": len(results),
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "sw_exits": sw_exits,
        "dca_count": dca_count,
        "max_loss": max_loss,
        "results": results,
    }


def print_summary(summaries, baseline_pnl):
    print(f"\n{'Scenario':<45s} {'Total P&L':>12s} {'vs Base':>10s} {'Win%':>6s} "
          f"{'SW#':>4s} {'DCA#':>5s} {'Trades':>7s} {'MaxLoss':>10s}")
    print("-" * 105)
    for s in sorted(summaries, key=lambda x: x["total_pnl"], reverse=True):
        delta = s["total_pnl"] - baseline_pnl
        print(f"{s['label']:<45s} $ {s['total_pnl']:>9,.0f} $ {delta:>+8,.0f} "
              f"{s['wr']:>5.1f}% {s['sw_exits']:>4d} {s['dca_count']:>5d} "
              f"{s['trades']:>7d} $ {s['max_loss']:>8,.0f}")


def print_sideways_detail(baseline_results, scenario_results, label):
    """Show trades where sideways scalp changed the outcome."""
    print(f"\n--- {label}: trades changed by sideways scalp ---")
    print(f"  {'Date':<12s} {'Tkr':<7s} {'Base PnL':>10s} {'This PnL':>10s} "
          f"{'Delta':>10s} {'Base Reason':<20s} {'This Reason':<20s}")
    print(f"  {'-'*95}")

    base_by_key = {}
    for r in baseline_results:
        key = (r["date"], r["ticker"], r.get("entry_premium", 0))
        base_by_key[key] = r

    changed = []
    for r in scenario_results:
        key = (r["date"], r["ticker"], r.get("entry_premium", 0))
        b = base_by_key.get(key)
        if b and abs(r["pnl"] - b["pnl"]) > 1:
            changed.append((r, b))

    changed.sort(key=lambda x: x[0]["pnl"] - x[1]["pnl"], reverse=True)
    total_delta = 0
    for r, b in changed[:25]:
        d = r["pnl"] - b["pnl"]
        total_delta += d
        print(f"  {r['date']:<12s} {r['ticker']:<7s} ${b['pnl']:>9,.0f} ${r['pnl']:>9,.0f} "
              f"${d:>+9,.0f} {b['reason']:<20s} {r['reason']:<20s}")
    print(f"  Net delta from changed trades: ${total_delta:>+,.0f}")


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    harvester_conn.row_factory = sqlite3.Row

    # ── Baseline: DCA with 2PM gate (production config) ─────────────
    baseline = run_scenario(signals, harvester_conn, "baseline_dca_2pm", dca_mode="always")
    baseline_sum = summarize(baseline, "baseline (DCA+2PM gate)")
    baseline_pnl = baseline_sum["total_pnl"]
    skipped = len(signals) - baseline_sum["trades"]
    print(f"{baseline_sum['trades']} trades, {skipped} skipped")
    print(f"Baseline P&L: ${baseline_pnl:,.0f}")

    # Also run no-DCA baseline for reference
    nodca = run_scenario(signals, harvester_conn, "no_dca", dca_mode="none")
    nodca_sum = summarize(nodca, "no_dca (reference)")

    all_summaries = [baseline_sum, nodca_sum]

    # ── Phase 1: Sweep sideways scalp take-profit threshold ─────────
    print("\n" + "=" * 100)
    print("PHASE 1: Sweep take-profit threshold (with default sideways detection)")
    print("=" * 100)

    for tp_pct in [5, 8, 10, 12, 15, 18, 20]:
        label = f"sw_tp{tp_pct}%"
        results = run_scenario(
            signals, harvester_conn, label, dca_mode="always",
            sideways_scalp=True, sw_take_profit_pct=tp_pct,
        )
        s = summarize(results, label)
        all_summaries.append(s)

    print_summary([s for s in all_summaries if s["label"].startswith("sw_tp") or s["label"].startswith("baseline")], baseline_pnl)

    # ── Phase 2: Sweep signals_needed (strictness) ──────────────────
    print("\n" + "=" * 100)
    print("PHASE 2: Sweep signals_needed (how many indicators must agree)")
    print("=" * 100)

    for sn in [1, 2, 3, 4]:
        for tp in [8, 10, 12, 15]:
            label = f"sw_sn{sn}_tp{tp}%"
            results = run_scenario(
                signals, harvester_conn, label, dca_mode="always",
                sideways_scalp=True, sw_take_profit_pct=tp,
                sw_signals_needed=sn,
            )
            s = summarize(results, label)
            all_summaries.append(s)

    phase2 = [s for s in all_summaries if s["label"].startswith("sw_sn")]
    print_summary(phase2, baseline_pnl)

    # ── Phase 3: Sweep min elapsed time ─────────────────────────────
    print("\n" + "=" * 100)
    print("PHASE 3: Sweep minimum time before sideways scalp can fire")
    print("=" * 100)

    # Pick the best signals_needed + tp from phase 2
    best_p2 = max(phase2, key=lambda x: x["total_pnl"]) if phase2 else None
    if best_p2:
        # Parse params from label
        import re
        m = re.match(r"sw_sn(\d+)_tp(\d+)%", best_p2["label"])
        best_sn = int(m.group(1)) if m else 2
        best_tp = int(m.group(2)) if m else 10
        print(f"Best from Phase 2: sn={best_sn}, tp={best_tp}% (${best_p2['total_pnl']:,.0f})")

        for min_el in [3, 5, 8, 10, 15, 20]:
            label = f"sw_sn{best_sn}_tp{best_tp}%_min{min_el}m"
            results = run_scenario(
                signals, harvester_conn, label, dca_mode="always",
                sideways_scalp=True, sw_take_profit_pct=best_tp,
                sw_signals_needed=best_sn, sw_min_elapsed_min=min_el,
            )
            s = summarize(results, label)
            all_summaries.append(s)

    # ── Phase 4: Sweep sideways detection params ────────────────────
    print("\n" + "=" * 100)
    print("PHASE 4: Tune sideways detection sensitivity")
    print("=" * 100)

    for range_pct in [8, 12, 15, 20]:
        for no_high_min in [5, 8, 12]:
            for cross_thresh in [2, 3, 4]:
                label = f"sw_r{range_pct}_h{no_high_min}m_x{cross_thresh}"
                results = run_scenario(
                    signals, harvester_conn, label, dca_mode="always",
                    sideways_scalp=True,
                    sw_take_profit_pct=best_tp if best_p2 else 10,
                    sw_signals_needed=best_sn if best_p2 else 2,
                    sw_range_pct=range_pct,
                    sw_no_new_high_min=no_high_min,
                    sw_cross_thresh=cross_thresh,
                )
                s = summarize(results, label)
                all_summaries.append(s)

    phase4 = [s for s in all_summaries if s["label"].startswith("sw_r")]
    print_summary(phase4[:20], baseline_pnl)  # Top 20

    # ── Phase 5: No-DCA + sideways scalp (does it help without DCA?) ─
    print("\n" + "=" * 100)
    print("PHASE 5: Sideways scalp WITHOUT DCA")
    print("=" * 100)

    for tp in [8, 10, 12, 15]:
        for sn in [1, 2, 3]:
            label = f"nodca_sw_sn{sn}_tp{tp}%"
            results = run_scenario(
                signals, harvester_conn, label, dca_mode="none",
                sideways_scalp=True, sw_take_profit_pct=tp,
                sw_signals_needed=sn,
            )
            s = summarize(results, label)
            all_summaries.append(s)

    phase5 = [s for s in all_summaries if s["label"].startswith("nodca_sw")]
    print_summary(phase5, nodca_sum["total_pnl"])

    # ── Final ranking ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ALL SCENARIOS — TOP 20")
    print("=" * 100)
    print_summary(sorted(all_summaries, key=lambda x: x["total_pnl"], reverse=True)[:20], baseline_pnl)

    # ── Detail on best sideways scenario ────────────────────────────
    best_sw = max(
        [s for s in all_summaries if s["sw_exits"] > 0],
        key=lambda x: x["total_pnl"],
        default=None,
    )
    if best_sw:
        print(f"\n{'='*100}")
        print(f"BEST SIDEWAYS SCENARIO: {best_sw['label']} (${best_sw['total_pnl']:,.0f})")
        print(f"{'='*100}")
        print_sideways_detail(baseline, best_sw["results"], best_sw["label"])

        # Show all sideways exits
        sw_trades = [r for r in best_sw["results"] if r.get("sideways_exit")]
        if sw_trades:
            print(f"\n  Sideways scalp exits ({len(sw_trades)} trades):")
            print(f"  {'Date':<12s} {'Tkr':<7s} {'PnL':>10s} {'Hold':>7s} {'Peak%':>7s} {'Entry$':>8s} {'Exit$':>8s}")
            print(f"  {'-'*65}")
            for r in sorted(sw_trades, key=lambda x: x["pnl"], reverse=True):
                print(f"  {r['date']:<12s} {r['ticker']:<7s} ${r['pnl']:>9,.0f} "
                      f"{r['hold']:>5.0f}m {r['peak_gain']:>6.1f}% "
                      f"${r['entry_premium']:>6.2f} ${r['exit_prem']:>6.2f}")
            sw_pnl = sum(r["pnl"] for r in sw_trades)
            print(f"  Total from sideways exits: ${sw_pnl:>+,.0f}")

    harvester_conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
