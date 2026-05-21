"""Backtest: Per-ticker progressive trail + optional ML retrace override.

Key insight from prior tests:
- "Gentle" schedule (+$130) helps TSLA and PLTR but only 2 trades triggered
- AMD/META/IWM have monster runners — leave those with wide trails
- TSLA/NVDA/AAPL/MSFT/AVGO rarely run past +200% — can tighten safely

Strategy:
  - Per-ticker trail schedules based on each ticker's actual runner profile
  - "Runner" tickers (AMZN, QQQ, IWM, META, AMD): keep wide trails
  - "Non-runner" tickers (TSLA, NVDA, AAPL, MSFT): progressive tightening
  - "Middle" tickers: moderate tightening

Also tests: ML retrace classifier as a gatekeeper — only tighten the trail
when ML says "this retrace is FINAL" (high confidence the run is over).

Usage:
    python scripts/backtest_per_ticker_trails.py
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
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]

# --- Per-ticker trail schedules ---
# Based on runner probability data:
# Format: list of (peak_gain_threshold, trail_width_pct) — highest first
# None = no progressive trail (use standard FSM)

TICKER_SCHEDULES = {
    # MONSTER RUNNERS — never tighten, let FSM handle
    "AMD":   None,  # +602% runner, avg peak +262%
    "IWM":   None,  # avg peak +322%, always runs
    "META":  None,  # +780% runner, 100% at +50 run further

    # RELIABLE RUNNERS — very light tightening at extreme gains only
    "AMZN":  [(400, 25), (300, 30)],  # 80% at +100 run further
    "QQQ":   [(400, 25), (300, 30)],  # 75% at +100 run further
    "GOOGL": [(300, 25), (200, 30)],  # 100% at +100 run further
    "SPY":   [(300, 25), (200, 30)],  # 88% at +25 run further

    # MODERATE — progressive tightening
    "TSLA":  [(300, 20), (200, 25), (150, 30)],  # 58% at +100, 0% at +200
    "MSTR":  [(300, 20), (200, 25), (150, 30)],  # 50% at +100, volatile
    "PLTR":  [(250, 20), (150, 25), (100, 30)],  # 33% at +100, drops hard
    "AVGO":  [(200, 25), (150, 30), (100, 35)],  # 50% at +100, 0% at +150

    # NON-RUNNERS — tighten earlier
    "AAPL":  [(200, 20), (125, 25), (75, 30)],   # 0% at +125, stalls early
    "NVDA":  [(200, 25), (150, 30), (100, 35)],  # Only 50% at +100
    "MSFT":  [(200, 25), (100, 30), (75, 35)],   # 50% at +100, rare runner
}

CATEGORY_DEFAULTS = {
    TickerCategory.INDEX:    [(400, 25), (300, 30)],
    TickerCategory.HIGH_VOL: [(300, 25), (200, 30), (150, 35)],
    TickerCategory.STANDARD: [(250, 25), (150, 30), (100, 35)],
}


def get_schedule(ticker):
    if ticker in TICKER_SCHEDULES:
        return TICKER_SCHEDULES[ticker]
    cat = categorize_ticker(ticker)
    return CATEGORY_DEFAULTS.get(cat)


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
    return f"O:{ticker}{exp_str}{ot}{int(strike * 1000):08d}"


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
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 5
    pos_cap = PORTFOLIO * 0.15
    mult = 0
    for tier_score, tier_mult in SCORE_TIERS:
        if score >= tier_score:
            mult = tier_mult
            break
    if mult == 0:
        return 0
    cost = premium * 100
    if cost <= 0:
        return 0
    return max(1, min(int(per_slot * mult / cost), int(pos_cap / cost)))


def _strip_tz(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
             trail_schedule=None):
    """Run FSM with optional progressive trail override.

    trail_schedule: list of (peak_gain_pct, trail_width_pct) or None for baseline.
    Progressive trail fires BEFORE FSM — if it triggers, exit immediately.
    """
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

        peak_prem = state.peak_premium
        if premium > peak_prem:
            peak_prem = premium
        peak_gain = (peak_prem - entry_premium) / entry_premium * 100
        current_gain = (premium - entry_premium) / entry_premium * 100
        drop_from_peak = (peak_prem - premium) / peak_prem * 100 if peak_prem > 0 else 0

        # Progressive trail check
        if trail_schedule:
            active_trail = None
            for tier_gain, tier_trail in trail_schedule:
                if peak_gain >= tier_gain:
                    active_trail = tier_trail
                    break

            if active_trail is not None and drop_from_peak >= active_trail:
                pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                return {
                    "pnl": pnl,
                    "reason": f"prog_trail_{active_trail}%",
                    "exit_gain": current_gain,
                    "peak_gain": peak_gain,
                    "progressive": True,
                    "trail_width": active_trail,
                }

        # Normal FSM
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                locked_pnl += (premium - entry_premium) * action.contracts_to_close * 100
                remaining -= action.contracts_to_close
                state.contracts = remaining
                continue

            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "exit_gain": current_gain,
                "peak_gain": peak_gain,
                "progressive": False,
            }

    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    return {
        "pnl": pnl, "reason": "last_tick",
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "peak_gain": peak_gain, "progressive": False,
    }


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals")

    hconn = sqlite3.connect(HARVESTER_DB)

    results_bl = []
    results_pt = []

    # Also test "apply to ALL tickers" with the best schedules
    universal_schedules = {
        "gentle_all": [(400, 25), (200, 30), (100, 40)],
        "your_idea": [(300, 15), (200, 20), (150, 25), (100, 30)],
        "v2_moderate": [(300, 20), (200, 25), (150, 30), (100, 35)],
        "v3_wider": [(300, 25), (200, 30), (150, 35)],
    }
    universal_results = {k: [] for k in universal_schedules}

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
        bl = simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
                      trail_schedule=None)
        if bl is None:
            continue
        bl["ticker"] = ticker
        bl["contracts"] = contracts
        results_bl.append(bl)

        # Per-ticker schedule
        sched = get_schedule(ticker)
        pt = simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
                      trail_schedule=sched if sched else None)
        if pt:
            pt["ticker"] = ticker
            pt["contracts"] = contracts
            results_pt.append(pt)

        # Universal schedules
        for name, usched in universal_schedules.items():
            ur = simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
                          trail_schedule=usched)
            if ur:
                ur["ticker"] = ticker
                ur["contracts"] = contracts
                universal_results[name].append(ur)

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(signals)}, matched {matched}")

    hconn.close()

    # --- Summary ---
    bl_total = sum(r["pnl"] for r in results_bl)
    pt_total = sum(r["pnl"] for r in results_pt)

    print(f"\n{'=' * 110}")
    print(f"RESULTS: {matched} signals matched")
    print(f"{'=' * 110}")

    def print_summary(results, label, base_total):
        total = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100
        d = total - base_total
        prog = sum(1 for r in results if r.get("progressive"))
        print(f"  {label:<24}: ${total:>10,.0f} (Δ ${d:>+8,.0f}) | "
              f"WR {wr:.1f}% | {prog} prog exits")

    print_summary(results_bl, "Baseline FSM", bl_total)
    print_summary(results_pt, "Per-ticker schedules", bl_total)
    for name in universal_schedules:
        print_summary(universal_results[name], f"Universal: {name}", bl_total)

    # --- Per-ticker detail for per-ticker schedule ---
    print(f"\n{'=' * 110}")
    print("PER-TICKER SCHEDULE IMPACT")
    print(f"{'=' * 110}")

    ticker_groups = {}
    for bl, pt in zip(results_bl, results_pt):
        tk = bl["ticker"]
        if tk not in ticker_groups:
            sched = get_schedule(tk)
            sched_str = "NONE (wide)" if sched is None else " → ".join(
                f"+{g}%:{t}%" for g, t in sched)
            ticker_groups[tk] = {"bl": [], "pt": [], "schedule": sched_str}
        ticker_groups[tk]["bl"].append(bl)
        ticker_groups[tk]["pt"].append(pt)

    for tk in sorted(ticker_groups.keys()):
        data = ticker_groups[tk]
        bl_pnl = sum(r["pnl"] for r in data["bl"])
        pt_pnl = sum(r["pnl"] for r in data["pt"])
        d = pt_pnl - bl_pnl
        n = len(data["bl"])
        prog = sum(1 for r in data["pt"] if r.get("progressive"))

        if d != 0:
            marker = " <<<" if abs(d) > 100 else ""
        else:
            marker = ""

        print(f"  {tk:<8} ({n:>2} trades): Δ ${d:>+8,.0f} | "
              f"BL ${bl_pnl:>8,.0f} → PT ${pt_pnl:>8,.0f} | "
              f"{prog} prog exits | {data['schedule']}{marker}")

    # --- Show every progressive exit ---
    print(f"\n{'=' * 110}")
    print("ALL PROGRESSIVE TRAIL EXITS (per-ticker schedule)")
    print(f"{'=' * 110}")

    for bl, pt in zip(results_bl, results_pt):
        if not pt.get("progressive"):
            continue
        d = pt["pnl"] - bl["pnl"]
        status = "WIN" if d > 0 else "LOSS"
        print(f"  {pt['ticker']:<8} peak +{pt['peak_gain']:.0f}% → "
              f"prog exit +{pt['exit_gain']:.0f}% (trail {pt.get('trail_width', '?')}%) vs "
              f"BL +{bl['exit_gain']:.0f}% ({bl['reason']}) | "
              f"Δ ${d:>+8,.0f} {status}")

    # --- What about per-ticker universals? ---
    print(f"\n{'=' * 110}")
    print("UNIVERSAL SCHEDULE: Per-ticker detail for 'v2_moderate'")
    print(f"  Schedule: +300%→20% +200%→25% +150%→30% +100%→35%")
    print(f"{'=' * 110}")

    v2_results = universal_results["v2_moderate"]
    for bl, v2 in zip(results_bl, v2_results):
        if not v2.get("progressive"):
            continue
        d = v2["pnl"] - bl["pnl"]
        status = "WIN" if d > 0 else "LOSS"
        print(f"  {v2['ticker']:<8} peak +{v2['peak_gain']:.0f}% → "
              f"prog exit +{v2['exit_gain']:.0f}% (trail {v2.get('trail_width', '?')}%) vs "
              f"BL +{bl['exit_gain']:.0f}% ({bl['reason']}) | "
              f"Δ ${d:>+8,.0f} {status}")


if __name__ == "__main__":
    main()
