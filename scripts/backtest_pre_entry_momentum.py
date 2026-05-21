"""Analyze pre-entry momentum patterns — does the underlying moving against us
before entry predict losses? Should we wait for a rebound?

For each trade, looks at the first N ticks of harvester data AFTER the signal
arrives to measure:
  1. Pre-entry underlying direction (fading or confirming?)
  2. Pre-entry premium direction (fading or confirming?)
  3. What if we waited for a rebound before entering?
  4. What if we skipped trades where underlying was moving hard against us?

Usage:
    python scripts/backtest_pre_entry_momentum.py
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

from options_owl.risk.exit_v5.config import get_ticker_config
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
    ENABLE_V6_PREMIUM_CAP=True,
    V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0,
    V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True,
    V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True,
    V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0,
    V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0,
    V6_DCA_MAX_DIP_PCT=35.0,
    V6_DCA_UNDERLYING_THRESHOLD=0.5,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 23000
SCORE_TIERS = [
    (135, 1.00, 0.15), (120, 0.85, 0.12), (100, 0.85, 0.08),
    (90, 0.50, 0.08), (78, 0.25, 0.08),
]


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
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume
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


def simulate_fsm(df, entry_idx, entry_premium, contracts, direction, dte, expiry_date, ticker):
    """Run FSM starting from entry_idx in the dataframe."""
    if entry_premium <= 0 or entry_idx >= len(df) - 5:
        return None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"

    entry_ts = df["ts"].iloc[entry_idx]
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    first_underlying = 0.0
    for i in range(entry_idx, min(entry_idx + 5, len(df))):
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

    for idx in range(entry_idx + 1, len(df)):
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

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (premium - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value, "hold": elapsed,
                    "exit_prem": premium, "peak_gain": peak_gain}

    last_prem = df["premium"].iloc[-1]
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, 'to_pydatetime'):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end", "hold": elapsed,
            "exit_prem": last_prem, "peak_gain": peak_gain}


def size_contracts(score, adj_entry):
    score_mult = 0.25
    tier_pos_pct = 0.08
    for threshold, mult, pos_pct in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            tier_pos_pct = pos_pct
            break
    effective_pos_pct = max(tier_pos_pct, 0.15)
    position_cap = PORTFOLIO * effective_pos_pct
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 4
    cost_per = adj_entry * 100
    if cost_per <= 0:
        return 1
    scaled = per_slot * score_mult
    raw = int(scaled / cost_per)
    cap_c = int(position_cap / cost_per)
    if cap_c == 0:
        return 0
    return max(1, min(raw, cap_c))


def measure_pre_entry(df, direction, window=10):
    """Measure underlying and premium momentum in first `window` ticks.

    Returns dict with momentum metrics.
    """
    is_call = direction in ("bullish", "call")
    w = min(window, len(df))

    # Underlying momentum
    underlying = []
    for i in range(w):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying.append(float(u))

    und_move_pct = 0.0
    und_direction = "flat"
    if len(underlying) >= 3:
        und_move_pct = (underlying[-1] - underlying[0]) / underlying[0] * 100
        if is_call:
            und_direction = "confirming" if und_move_pct > 0.02 else ("against" if und_move_pct < -0.02 else "flat")
        else:
            und_direction = "confirming" if und_move_pct < -0.02 else ("against" if und_move_pct > 0.02 else "flat")

    # How hard against? Count consecutive bars moving wrong way
    consecutive_against = 0
    for i in range(1, len(underlying)):
        if is_call and underlying[i] < underlying[i-1]:
            consecutive_against += 1
        elif not is_call and underlying[i] > underlying[i-1]:
            consecutive_against += 1
        else:
            break  # streak broken

    # Premium momentum (first few ticks)
    prem_start = df["premium"].iloc[0] if len(df) > 0 else 0
    prem_at_w = df["premium"].iloc[min(w-1, len(df)-1)]
    prem_move_pct = (prem_at_w - prem_start) / prem_start * 100 if prem_start > 0 else 0

    # Did premium dip then recover? (potential rebound entry)
    premiums = [df["premium"].iloc[i] for i in range(w)]
    min_prem = min(premiums)
    min_idx = premiums.index(min_prem)
    dip_pct = (prem_start - min_prem) / prem_start * 100 if prem_start > 0 else 0
    recovered = premiums[-1] > min_prem if min_idx < w - 1 else False

    # Underlying strength score: -3 (hard against) to +3 (strong confirming)
    strength = 0
    if und_move_pct != 0:
        mag = abs(und_move_pct)
        if is_call:
            sign = 1 if und_move_pct > 0 else -1
        else:
            sign = 1 if und_move_pct < 0 else -1

        if mag > 0.15:
            strength = sign * 3
        elif mag > 0.08:
            strength = sign * 2
        elif mag > 0.02:
            strength = sign * 1

    return {
        "und_move_pct": und_move_pct,
        "und_direction": und_direction,
        "consecutive_against": consecutive_against,
        "prem_move_pct": prem_move_pct,
        "dip_pct": dip_pct,
        "recovered": recovered,
        "min_prem_idx": min_idx,
        "strength": strength,
    }


def find_rebound_entry(df, direction, max_wait_ticks=20):
    """Find the first tick where premium rebounds from a dip.

    A rebound = premium dips below tick 0, then comes back to within 3%
    of tick 0 premium. Returns the tick index, or None if no rebound.
    """
    if len(df) < 5:
        return None

    is_call = direction in ("bullish", "call")
    entry_prem = df["premium"].iloc[0]
    entry_und = None
    for i in range(min(3, len(df))):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            entry_und = float(u)
            break

    saw_dip = False
    min_prem = entry_prem

    for i in range(1, min(max_wait_ticks, len(df))):
        prem = df["premium"].iloc[i]
        if np.isnan(prem) or prem <= 0:
            continue

        if prem < min_prem:
            min_prem = prem

        # Must dip at least 3% from entry
        dip_from_entry = (entry_prem - min_prem) / entry_prem * 100
        if dip_from_entry >= 3:
            saw_dip = True

        # Rebound: saw a dip, and now premium is within 5% of entry OR
        # underlying has confirmed direction (moved in our favor)
        if saw_dip:
            prem_recovery = (prem - min_prem) / min_prem * 100 if min_prem > 0 else 0
            back_to_entry_pct = (entry_prem - prem) / entry_prem * 100

            und = df["underlying_price"].iloc[i]
            und_confirmed = False
            if und and entry_und and entry_und > 0:
                und_move = (float(und) - entry_und) / entry_und * 100
                if is_call and und_move > 0.03:
                    und_confirmed = True
                elif not is_call and und_move < -0.03:
                    und_confirmed = True

            # Rebound if: premium recovered 50%+ of dip, or back within 5% of entry
            if prem_recovery >= 50 or back_to_entry_pct <= 5 or und_confirmed:
                return i

    return None


def main():
    print("Loading signals and tick data...")
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    tick_cache = {}
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is not None:
            tick_cache[sig["id"]] = (df, sig.get("_dte", 0), sig.get("_expiry_date", ""))
    harvester_conn.close()
    print(f"  {len(tick_cache)} signals with tick data")

    # ── Analysis 1: Pre-entry momentum vs trade outcome ──────────────────

    rows = []
    for sig in signals:
        if sig["id"] not in tick_cache:
            continue
        df, dte, expiry = tick_cache[sig["id"]]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        ticker = sig["ticker"]

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = sig["premium"]

        # Premium cap
        cap = 6.0
        if score >= 150: cap = 9.0
        elif score >= 120: cap = 7.0
        if adj_entry > cap:
            continue

        # Spread gate
        fb = df["bid"].iloc[0]
        fa = df["ask"].iloc[0]
        if fb and fa and fb > 0 and fa > 0:
            if (fa - fb) / fa * 100 > 15:
                continue

        contracts = size_contracts(score, adj_entry)
        if contracts == 0:
            continue

        # Measure pre-entry momentum at different windows
        mom_5 = measure_pre_entry(df, direction, window=5)
        mom_10 = measure_pre_entry(df, direction, window=10)

        # Simulate immediate entry (tick 0)
        result = simulate_fsm(df, 0, adj_entry, contracts, direction, dte, expiry, ticker)
        if result is None:
            continue

        # Find rebound entry point
        rebound_idx = find_rebound_entry(df, direction, max_wait_ticks=20)
        rebound_result = None
        rebound_entry = None
        if rebound_idx is not None and rebound_idx < len(df) - 5:
            rebound_ask = df["ask"].iloc[rebound_idx]
            rebound_mid = df["premium"].iloc[rebound_idx]
            rebound_entry = rebound_ask if rebound_ask and rebound_ask > 0 else rebound_mid
            if rebound_entry and rebound_entry > 0:
                rebound_result = simulate_fsm(
                    df, rebound_idx, rebound_entry, contracts,
                    direction, dte, expiry, ticker
                )

        row = {
            "id": sig["id"],
            "ticker": ticker,
            "day": sig["created_at"][:10],
            "score": score,
            "direction": direction,
            "entry": adj_entry,
            "contracts": contracts,
            "dte": dte,
            # Momentum metrics
            "und_move_5": mom_5["und_move_pct"],
            "und_dir_5": mom_5["und_direction"],
            "consec_against_5": mom_5["consecutive_against"],
            "prem_move_5": mom_5["prem_move_pct"],
            "strength_5": mom_5["strength"],
            "und_move_10": mom_10["und_move_pct"],
            "und_dir_10": mom_10["und_direction"],
            "consec_against_10": mom_10["consecutive_against"],
            "prem_move_10": mom_10["prem_move_pct"],
            "strength_10": mom_10["strength"],
            "dip_pct": mom_10["dip_pct"],
            "recovered": mom_10["recovered"],
            # Immediate entry results
            "pnl": result["pnl"],
            "reason": result["reason"],
            "peak_gain": result["peak_gain"],
            # Rebound entry results
            "rebound_idx": rebound_idx,
            "rebound_entry": rebound_entry,
            "rebound_pnl": rebound_result["pnl"] if rebound_result else None,
            "rebound_reason": rebound_result["reason"] if rebound_result else None,
        }
        rows.append(row)

    df_all = pd.DataFrame(rows)
    print(f"\n{len(df_all)} trades analyzed")

    # ── Table 1: Underlying direction at entry vs outcome ────────────────

    print(f"\n{'=' * 100}")
    print("UNDERLYING DIRECTION AT ENTRY (first 10 ticks) vs TRADE OUTCOME")
    print(f"{'=' * 100}")

    for dir_label in ["confirming", "flat", "against"]:
        subset = df_all[df_all["und_dir_10"] == dir_label]
        if len(subset) == 0:
            continue
        wins = (subset["pnl"] > 0).sum()
        wr = wins / len(subset) * 100
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        avg_w = subset[subset["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_l = subset[subset["pnl"] <= 0]["pnl"].mean() if len(subset) - wins > 0 else 0
        print(f"\n  {dir_label.upper():>12}: {len(subset):>3} trades | WR: {wr:.0f}% | "
              f"Total: ${total:>9,.2f} | Avg: ${avg:>7,.2f} | "
              f"AvgWin: ${avg_w:>7,.2f} | AvgLoss: ${avg_l:>7,.2f}")

    # ── Table 2: Strength score buckets ──────────────────────────────────

    print(f"\n{'=' * 100}")
    print("UNDERLYING STRENGTH SCORE (-3=hard against, +3=strong confirming)")
    print(f"{'=' * 100}")
    print(f"{'Strength':>8} {'Trades':>6} {'Win%':>5} {'TotalPnL':>11} {'AvgPnL':>9} "
          f"{'AvgWin':>9} {'AvgLoss':>9} {'BigLoss':>9}")
    print("-" * 80)

    for s in range(-3, 4):
        subset = df_all[df_all["strength_10"] == s]
        if len(subset) == 0:
            continue
        wins = (subset["pnl"] > 0).sum()
        wr = wins / len(subset) * 100
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        avg_w = subset[subset["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_l = subset[subset["pnl"] <= 0]["pnl"].mean() if len(subset) - wins > 0 else 0
        big_l = subset["pnl"].min()
        print(f"{s:>8} {len(subset):>6} {wr:>4.0f}% ${total:>9,.2f} ${avg:>7,.2f} "
              f"${avg_w:>7,.2f} ${avg_l:>7,.2f} ${big_l:>7,.2f}")

    # ── Table 3: Consecutive bars against ────────────────────────────────

    print(f"\n{'=' * 100}")
    print("CONSECUTIVE UNDERLYING BARS AGAINST US AT ENTRY (first 5 ticks)")
    print(f"{'=' * 100}")
    print(f"{'ConsecAgainst':>13} {'Trades':>6} {'Win%':>5} {'TotalPnL':>11} {'AvgPnL':>9}")
    print("-" * 50)

    for c in range(6):
        subset = df_all[df_all["consec_against_5"] == c]
        if len(subset) == 0:
            continue
        wins = (subset["pnl"] > 0).sum()
        wr = wins / len(subset) * 100
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        print(f"{c:>13} {len(subset):>6} {wr:>4.0f}% ${total:>9,.2f} ${avg:>7,.2f}")

    # ── Table 4: Premium fade at entry ───────────────────────────────────

    print(f"\n{'=' * 100}")
    print("PREMIUM MOVEMENT IN FIRST 10 TICKS (negative = fading)")
    print(f"{'=' * 100}")

    bins = [(-999, -10), (-10, -5), (-5, -2), (-2, 2), (2, 5), (5, 10), (10, 999)]
    labels = ["< -10%", "-10 to -5%", "-5 to -2%", "-2 to +2%", "+2 to +5%", "+5 to +10%", "> +10%"]
    print(f"{'PremMove':>12} {'Trades':>6} {'Win%':>5} {'TotalPnL':>11} {'AvgPnL':>9} "
          f"{'AvgWin':>9} {'AvgLoss':>9}")
    print("-" * 70)

    for (lo, hi), label in zip(bins, labels):
        subset = df_all[(df_all["prem_move_10"] >= lo) & (df_all["prem_move_10"] < hi)]
        if len(subset) == 0:
            continue
        wins = (subset["pnl"] > 0).sum()
        losses = len(subset) - wins
        wr = wins / len(subset) * 100
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        avg_w = subset[subset["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_l = subset[subset["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
        print(f"{label:>12} {len(subset):>6} {wr:>4.0f}% ${total:>9,.2f} ${avg:>7,.2f} "
              f"${avg_w:>7,.2f} ${avg_l:>7,.2f}")

    # ── Table 5: Rebound entry vs immediate entry ────────────────────────

    print(f"\n{'=' * 100}")
    print("REBOUND ENTRY vs IMMEDIATE ENTRY")
    print("(Wait for premium to dip 3%+ then recover 50%+ before entering)")
    print(f"{'=' * 100}")

    has_rebound = df_all[df_all["rebound_pnl"].notna()]
    no_rebound = df_all[df_all["rebound_pnl"].isna()]

    print(f"\n  Trades with rebound opportunity: {len(has_rebound)}")
    print(f"  Trades without (entered smoothly): {len(no_rebound)}")

    if len(has_rebound) > 0:
        imm_pnl = has_rebound["pnl"].sum()
        reb_pnl = has_rebound["rebound_pnl"].sum()
        imm_wr = (has_rebound["pnl"] > 0).mean() * 100
        reb_wr = (has_rebound["rebound_pnl"] > 0).mean() * 100

        print(f"\n  For the {len(has_rebound)} trades where a rebound was available:")
        print(f"    Immediate entry P&L:  ${imm_pnl:>10,.2f}  WR: {imm_wr:.0f}%")
        print(f"    Rebound entry P&L:    ${reb_pnl:>10,.2f}  WR: {reb_wr:.0f}%")
        print(f"    Difference:           ${reb_pnl - imm_pnl:>+10,.2f}")

        # Break down by direction at entry
        for dir_label in ["against", "flat", "confirming"]:
            sub = has_rebound[has_rebound["und_dir_10"] == dir_label]
            if len(sub) == 0:
                continue
            imm = sub["pnl"].sum()
            reb = sub["rebound_pnl"].sum()
            print(f"\n    {dir_label.upper():>12} ({len(sub)} trades):")
            print(f"      Immediate: ${imm:>9,.2f}  |  Rebound: ${reb:>9,.2f}  |  Diff: ${reb-imm:>+9,.2f}")

    # ── Table 6: What-if scenarios ───────────────────────────────────────

    print(f"\n{'=' * 100}")
    print("WHAT-IF SCENARIOS: Skip or modify entry based on pre-entry momentum")
    print(f"{'=' * 100}")

    baseline_pnl = df_all["pnl"].sum()
    baseline_trades = len(df_all)
    print(f"\n  Baseline: {baseline_trades} trades, ${baseline_pnl:,.2f}")

    scenarios = [
        ("Skip strength <= -2 (hard against)", df_all["strength_10"] > -2, None),
        ("Skip strength <= -1 (any against)", df_all["strength_10"] > -1, None),
        ("Skip strength < 0 (not confirming)", df_all["strength_10"] >= 0, None),
        ("Only strength >= 1 (confirming)", df_all["strength_10"] >= 1, None),
        ("Skip if 3+ consec bars against", df_all["consec_against_5"] < 3, None),
        ("Skip if 4+ consec bars against", df_all["consec_against_5"] < 4, None),
        ("Skip if premium fading > 5%", df_all["prem_move_10"] >= -5, None),
        ("Skip if premium fading > 10%", df_all["prem_move_10"] >= -10, None),
        ("Skip if UND against AND prem fading > 5%",
         ~((df_all["und_dir_10"] == "against") & (df_all["prem_move_10"] < -5)), None),
        ("Skip if strength<=-1 AND prem fade>5%",
         ~((df_all["strength_10"] <= -1) & (df_all["prem_move_10"] < -5)), None),
    ]

    # Add rebound scenarios
    # "Use rebound entry when available AND direction is against"
    # This requires combining immediate and rebound P&L

    print(f"\n  {'Scenario':<45} {'Trades':>6} {'P&L':>11} {'vs Base':>9} {'Win%':>5}")
    print("  " + "-" * 85)

    for name, mask, _ in scenarios:
        subset = df_all[mask]
        skipped = df_all[~mask]
        total = subset["pnl"].sum()
        wr = (subset["pnl"] > 0).mean() * 100 if len(subset) > 0 else 0
        diff = total - baseline_pnl
        skipped_pnl = skipped["pnl"].sum()
        print(f"  {name:<45} {len(subset):>6} ${total:>9,.2f} ${diff:>+7,.0f} {wr:>4.0f}%"
              f"  (skipped {len(skipped)} trades worth ${skipped_pnl:>+,.0f})")

    # Rebound hybrid: use rebound entry when against, immediate otherwise
    print(f"\n  Hybrid: rebound entry when against, immediate otherwise:")
    hybrid_pnl = 0
    hybrid_trades = 0
    hybrid_wins = 0
    for _, r in df_all.iterrows():
        if r["und_dir_10"] == "against" and r["rebound_pnl"] is not None and not np.isnan(r["rebound_pnl"]):
            hybrid_pnl += r["rebound_pnl"]
            hybrid_trades += 1
            if r["rebound_pnl"] > 0:
                hybrid_wins += 1
        else:
            hybrid_pnl += r["pnl"]
            hybrid_trades += 1
            if r["pnl"] > 0:
                hybrid_wins += 1
    hybrid_wr = hybrid_wins / hybrid_trades * 100 if hybrid_trades > 0 else 0
    print(f"    {hybrid_trades} trades, ${hybrid_pnl:,.2f} ({hybrid_wr:.0f}% WR), "
          f"vs baseline ${hybrid_pnl - baseline_pnl:+,.2f}")

    # ── Table 7: Individual "against" trades ─────────────────────────────

    print(f"\n{'=' * 100}")
    print("TRADES WHERE UNDERLYING WAS HARD AGAINST (strength <= -2)")
    print(f"{'=' * 100}")

    hard_against = df_all[df_all["strength_10"] <= -2].sort_values("pnl")
    if len(hard_against) > 0:
        print(f"{'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} {'Entry':>6} {'Ct':>3} "
              f"{'UndMove':>8} {'PremMove':>9} {'P&L':>9} {'Reason':<20} "
              f"{'RebPnL':>9}")
        print("-" * 110)
        for _, r in hard_against.iterrows():
            reb = f"${r['rebound_pnl']:>7,.2f}" if r["rebound_pnl"] is not None and not np.isnan(r["rebound_pnl"]) else "  no reb"
            print(f"{r['day']:<12} {r['ticker']:<6} {r['direction'][:4]:<5} {r['score']:>5} "
                  f"${r['entry']:>4.2f} {r['contracts']:>3} "
                  f"{r['und_move_10']:>+7.3f}% {r['prem_move_10']:>+8.1f}% "
                  f"${r['pnl']:>7,.2f} {r['reason']:<20} {reb}")


if __name__ == "__main__":
    main()
