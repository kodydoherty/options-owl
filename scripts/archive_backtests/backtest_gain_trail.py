"""Backtest: gain-based adaptive trail vs current premium-based trail.

Tests the hypothesis that trailing on GAIN (peak_gain - current_gain)
instead of PREMIUM drop (peak_prem - current_prem) / peak_prem preserves
more profit on runners without hurting overall performance.

Also tests DCA mitigation strategies:
  A) No DCA (baseline)
  B) DCA only if trade was previously profitable (+10%+)
  C) Current DCA (always DCA on dip)

Usage:
    python scripts/backtest_gain_trail.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
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
    (135, 1.00),
    (120, 0.85),
    (100, 0.85),
    (90, 0.50),
    (78, 0.25),
]


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


def size_contracts(score, entry_premium):
    max_risk_pct = 0.75
    max_concurrent = 4
    max_position_pct = 0.15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent
    position_cap = PORTFOLIO * max_position_pct

    score_mult = 0.25
    for threshold, mult in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break

    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw_contracts, pos_cap_contracts))


def momentum_blocked(df, direction):
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    underlying_prices = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))

    if len(underlying_prices) < 5:
        return False

    first_half = underlying_prices[:len(underlying_prices)//2]
    second_half = underlying_prices[len(underlying_prices)//2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    pct_move = (avg_second - avg_first) / avg_first * 100

    prem_start = df["premium"].iloc[0]
    prem_5 = df["premium"].iloc[min(4, len(df)-1)]
    prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0

    neg_signals = 0
    if is_call and pct_move < -0.05:
        neg_signals += 1
    elif not is_call and pct_move > 0.05:
        neg_signals += 1
    if prem_fade < -5:
        neg_signals += 1
    against = 0
    for i in range(max(0, window-3), window):
        if i == 0:
            continue
        prev_u = df["underlying_price"].iloc[i-1]
        cur_u = df["underlying_price"].iloc[i]
        if prev_u and cur_u:
            if is_call and cur_u < prev_u:
                against += 1
            elif not is_call and cur_u > prev_u:
                against += 1
    if against >= 3:
        neg_signals += 1

    return neg_signals >= 2


# ── Simulation with gain-based trail option ──────────────────────────────────


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date,
                 ticker="SIM", use_gain_trail=False, dca_mode="none"):
    """Run FSM with optional gain-based trail override.

    use_gain_trail: if True, override adaptive trail to use gain-based math
    dca_mode: "none" | "profitable_only" | "always"
    """
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "dca_fired": False, "dca_pnl_impact": 0}

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

    locked_pnl = 0.0
    remaining = contracts
    dca_fired = False
    dca_contracts = 0
    dca_premium = 0.0
    max_gain_seen = 0.0  # track if trade was ever profitable

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

        # Track max gain for DCA gating
        gain = (premium - entry_premium) / entry_premium * 100
        max_gain_seen = max(max_gain_seen, gain)

        # ── DCA simulation ──────────────────────────────────────────────
        elapsed_min = (now - entry_ts).total_seconds() / 60
        if (not dca_fired
                and dca_mode != "none"
                and elapsed_min >= 5
                and remaining >= 1):
            dip_pct = (entry_premium - premium) / entry_premium * 100
            if 15 <= dip_pct <= 35:
                should_dca = False
                if dca_mode == "always":
                    should_dca = True
                elif dca_mode == "profitable_only" and max_gain_seen >= 10:
                    should_dca = True

                if should_dca:
                    dca_fired = True
                    dca_contracts = remaining  # double position
                    dca_premium = premium
                    # Update entry to weighted average
                    total_cost = entry_premium * remaining + dca_premium * dca_contracts
                    remaining += dca_contracts
                    entry_premium = total_cost / remaining
                    # Update FSM state
                    state.entry_premium = entry_premium
                    state.contracts = remaining
                    state.peak_premium = max(state.peak_premium, entry_premium)
                    continue  # skip exit eval this cycle

        # ── Gain-based trail override ────────────────────────────────────
        if use_gain_trail:
            peak_gain = (state.peak_premium - state.entry_premium) / state.entry_premium * 100
            current_gain = (premium - state.entry_premium) / state.entry_premium * 100
            gain_drop = peak_gain - current_gain  # how many percentage points given back

            # Check if adaptive trail would fire under gain-based math
            # Get the tiers (with 2PM tightening if applicable)
            et_now_hour = now.hour - 4
            if et_now_hour < 0:
                et_now_hour += 24

            active_cfg = cfg
            if et_now_hour >= 14:
                from dataclasses import replace
                tighten = 0.7
                base_tiers = cfg.get_adaptive_tiers(state.category)
                tight_tiers = tuple(
                    AdaptiveTier(t.min_peak_gain, t.trail_width * tighten)
                    for t in base_tiers
                )
                kw = {}
                from options_owl.risk.exit_v5.config import TickerCategory
                if state.category == TickerCategory.HIGH_VOL:
                    kw["adaptive_highvol_tiers"] = tight_tiers
                elif state.category == TickerCategory.INDEX:
                    kw["adaptive_index_tiers"] = tight_tiers
                else:
                    kw["adaptive_standard_tiers"] = tight_tiers
                active_cfg = replace(cfg, **kw)

            tiers = active_cfg.get_adaptive_tiers(state.category)

            # Check gain-based trail BEFORE running FSM
            gain_trail_fired = False
            for tier in tiers:
                if peak_gain >= tier.min_peak_gain and gain_drop >= tier.trail_width:
                    gain_trail_fired = True
                    break

            if gain_trail_fired and peak_gain >= 40:  # only override for meaningful gains
                elapsed = (now - entry_ts).total_seconds() / 60
                pk = (state.peak_premium - state.entry_premium) / state.entry_premium * 100
                pnl = locked_pnl + (premium - state.entry_premium) * remaining * 100
                return {
                    "pnl": pnl,
                    "reason": "adaptive_trail_gain",
                    "hold": elapsed,
                    "exit_prem": premium,
                    "peak_gain": pk,
                    "dca_fired": dca_fired,
                    "dca_pnl_impact": 0,
                }

        # ── Normal FSM evaluation ────────────────────────────────────────
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

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - state.entry_premium) / state.entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "dca_fired": dca_fired,
                "dca_pnl_impact": 0,
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
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": pnl,
        "reason": "eod_data_end",
        "hold": elapsed,
        "exit_prem": last_prem,
        "peak_gain": peak_gain,
        "dca_fired": dca_fired,
        "dca_pnl_impact": 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Scenarios to test
    scenarios = {
        "baseline":          {"use_gain_trail": False, "dca_mode": "none"},
        "gain_trail":        {"use_gain_trail": True,  "dca_mode": "none"},
        "baseline+dca":      {"use_gain_trail": False, "dca_mode": "always"},
        "gain_trail+dca":    {"use_gain_trail": True,  "dca_mode": "always"},
        "gain_trail+dca_prof": {"use_gain_trail": True, "dca_mode": "profitable_only"},
    }

    all_results = {name: [] for name in scenarios}
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

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

        if momentum_blocked(df, direction):
            for name in scenarios:
                all_results[name].append({
                    "ticker": ticker, "day": day, "score": score,
                    "entry": adj_entry, "contracts": contracts,
                    "direction": direction, "dte": dte,
                    "pnl": 0, "reason": "momentum_blocked",
                    "hold": 0, "exit_prem": adj_entry, "peak_gain": 0,
                    "dca_fired": False,
                })
            continue

        for name, params in scenarios.items():
            result = simulate_fsm(
                df, adj_entry, contracts, direction, dte, expiry_date,
                ticker=ticker, **params,
            )
            result.update({
                "ticker": ticker, "day": day, "score": score,
                "entry": adj_entry, "contracts": contracts,
                "direction": direction, "dte": dte,
            })
            all_results[name].append(result)

    harvester_conn.close()

    # ── Summary ──────────────────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"GAIN-BASED TRAIL BACKTEST — {len(all_results['baseline'])} trades, {no_data} skipped")
    print(f"{'=' * 100}")

    print(f"\n{'Scenario':<25} {'Total P&L':>12} {'Win%':>6} {'Wins':>5} {'Losses':>7} "
          f"{'Avg Win':>10} {'Avg Loss':>10} {'Max Loss':>10} {'Delta':>10}")
    print("-" * 110)

    baseline_pnl = 0
    for name in scenarios:
        df_r = pd.DataFrame(all_results[name])
        pnls = df_r["pnl"]
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        total = pnls.sum()
        wr = wins / len(pnls) * 100 if len(pnls) > 0 else 0
        avg_w = pnls[pnls > 0].mean() if wins > 0 else 0
        avg_l = pnls[pnls <= 0].mean() if losses > 0 else 0
        max_l = pnls.min()

        if name == "baseline":
            baseline_pnl = total
            delta = ""
        else:
            d = total - baseline_pnl
            delta = f"${d:+,.0f}"

        print(f"{name:<25} ${total:>10,.0f} {wr:>5.1f}% {wins:>5} {losses:>7} "
              f"${avg_w:>8,.0f} ${avg_l:>8,.0f} ${max_l:>8,.0f} {delta:>10}")

    # ── Per-trade comparison: gain trail vs baseline (where they differ) ─────

    print(f"\n{'=' * 100}")
    print(f"TRADES WHERE GAIN TRAIL DIFFERS FROM BASELINE")
    print(f"{'=' * 100}")

    base_df = pd.DataFrame(all_results["baseline"])
    gain_df = pd.DataFrame(all_results["gain_trail"])

    diff_mask = base_df["pnl"] != gain_df["pnl"]
    if diff_mask.sum() > 0:
        print(f"\n{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Peak%':>6} "
              f"{'Base PnL':>10} {'Base Exit':>12} {'Gain PnL':>10} {'Gain Exit':>12} {'Delta':>8}")
        print("-" * 100)
        for idx in diff_mask[diff_mask].index:
            b = base_df.iloc[idx]
            g = gain_df.iloc[idx]
            d = g["pnl"] - b["pnl"]
            print(f"{b['day']:<12} {b['ticker']:<6} {b['direction']:<5} "
                  f"{b['peak_gain']:>5.0f}% "
                  f"${b['pnl']:>8,.0f} {b['reason']:<12} "
                  f"${g['pnl']:>8,.0f} {g['reason']:<12} "
                  f"${d:>+7,.0f}")

        total_base = base_df.loc[diff_mask, "pnl"].sum()
        total_gain = gain_df.loc[diff_mask, "pnl"].sum()
        print(f"\nDiffering trades only: baseline=${total_base:,.0f} vs gain_trail=${total_gain:,.0f} "
              f"(delta=${total_gain - total_base:+,.0f})")
    else:
        print("No trades differed between baseline and gain trail!")

    # ── DCA analysis ─────────────────────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"DCA IMPACT ANALYSIS")
    print(f"{'=' * 100}")

    for dca_name in ["baseline+dca", "gain_trail+dca", "gain_trail+dca_prof"]:
        dca_df = pd.DataFrame(all_results[dca_name])
        dca_trades = dca_df[dca_df["dca_fired"] == True]
        if len(dca_trades) > 0:
            print(f"\n{dca_name}: {len(dca_trades)} DCA events")
            print(f"  DCA trade P&L: ${dca_trades['pnl'].sum():,.0f} "
                  f"(avg ${dca_trades['pnl'].mean():,.0f})")
            dca_wins = (dca_trades["pnl"] > 0).sum()
            print(f"  DCA wins: {dca_wins}/{len(dca_trades)}")

            # Compare DCA vs no-DCA for same trades
            no_dca_name = "gain_trail" if "gain" in dca_name else "baseline"
            no_dca_df = pd.DataFrame(all_results[no_dca_name])
            dca_idx = dca_trades.index
            no_dca_same = no_dca_df.iloc[dca_idx]
            print(f"  Same trades WITHOUT DCA: ${no_dca_same['pnl'].sum():,.0f}")
            diff = dca_trades["pnl"].sum() - no_dca_same["pnl"].sum()
            print(f"  DCA net impact: ${diff:+,.0f} ({'helped' if diff > 0 else 'HURT'})")

            print(f"\n  {'Day':<12} {'Ticker':<6} {'Peak%':>6} "
                  f"{'No DCA':>10} {'With DCA':>10} {'Delta':>8}")
            print(f"  {'-' * 65}")
            for i in dca_idx:
                nd = no_dca_df.iloc[i]
                wd = dca_df.iloc[i]
                d = wd["pnl"] - nd["pnl"]
                print(f"  {nd['day']:<12} {nd['ticker']:<6} {nd['peak_gain']:>5.0f}% "
                      f"${nd['pnl']:>8,.0f} ${wd['pnl']:>8,.0f} ${d:>+7,.0f}")
        else:
            print(f"\n{dca_name}: 0 DCA events")

    # ── Exit reason breakdown for gain trail ─────────────────────────────────

    print(f"\n{'=' * 100}")
    print(f"EXIT REASON BREAKDOWN — gain_trail")
    print(f"{'=' * 100}")

    gain_df = pd.DataFrame(all_results["gain_trail"])
    traded = gain_df[gain_df["reason"] != "momentum_blocked"]
    print(f"\n{'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 62)
    for reason, group in traded.groupby("reason"):
        gpnl = group["pnl"]
        gwins = (gpnl > 0).sum()
        gwr = gwins / len(gpnl) * 100
        print(f"{reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}%")


if __name__ == "__main__":
    main()
