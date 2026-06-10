"""Backtest: ML-gated progressive trail tightening.

Strategy: Use the retrace classifier (0.994 AUC) as a GATEKEEPER.
When the trade hits a gain milestone and starts retracing:
  - If ML says "TEMPORARY retrace" → keep wide trail, hold through it
  - If ML says "FINAL retrace" → tighten trail to lock profits

This combines:
  1. Progressive trail schedules (tighter trails at higher gains)
  2. ML retrace classifier (only tighten when ML confirms it's over)

The ML prevents the false tighten on retraces that would have recovered
(like the AVGO trade that peaked +173%, retraced 30%, then ran to +248%).

Usage:
    python scripts/backtest_ml_gated_trail.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import (
    TickerCategory,
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
MODEL_DIR = PROJECT_DIR / "journal" / "models"
PORTFOLIO = 8000
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]

FEATURE_COLS = [
    "retrace_depth", "gain_at_high", "gain_at_retrace_low", "pct_of_session_peak",
    "retrace_speed", "retrace_ticks",
    "vel_3", "vel_5", "vel_10", "prem_std",
    "delta", "gamma", "theta", "vega", "iv",
    "delta_change", "iv_change", "gamma_delta_ratio",
    "volume", "vol_ratio", "vol_trend",
    "spread_pct", "spread_change", "size_imbalance",
    "u_move", "u_vel", "u_retrace_pct", "moneyness", "divergence",
    "minutes_to_close", "session_progress", "elapsed_min",
    "is_highvol", "is_index", "is_call",
    "n_prior_retraces", "avg_prior_retrace_depth", "max_prior_retrace_depth",
]

# Progressive trail schedules
SCHEDULES = {
    "gentle": [(400, 25), (200, 30), (100, 40)],
    "moderate": [(300, 20), (200, 25), (150, 30), (100, 35)],
    "your_idea": [(300, 15), (200, 20), (150, 25), (100, 30)],
}

RETRACE_THRESHOLD = 15  # % drop from local high to trigger ML check


def safe_float(v, d=0.0):
    try:
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return d
        return float(v)
    except (ValueError, TypeError):
        return d


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
            SELECT captured_at, midpoint, bid, ask, underlying_price,
                   implied_volatility, delta, gamma, theta, vega,
                   day_volume, day_vwap, open_interest, bid_size, ask_size
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
        "iv", "delta", "gamma", "theta", "vega", "volume",
        "vwap", "open_interest", "bid_size", "ask_size",
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


def extract_retrace_features(df, idx, entry_premium, local_high, local_high_idx,
                              ticker, is_call, strike, entry_underlying,
                              prior_retraces, timestamps):
    """Extract features matching the retrace classifier training exactly."""
    premiums = df["premium"].values.astype(float)
    n = len(premiums)
    if idx < 10 or idx >= n:
        return None

    prem = premiums[idx]
    if np.isnan(prem) or prem <= 0:
        return None

    now = timestamps[idx]
    now_dt = pd.Timestamp(now)
    et_hour = now_dt.hour - 4
    if et_hour < 0:
        et_hour += 24
    et_minute = now_dt.minute
    et_decimal = et_hour + et_minute / 60
    if et_decimal < 9.5 or et_decimal > 16.0:
        return None

    category = categorize_ticker(ticker)

    retrace_depth = (local_high - prem) / local_high * 100
    gain_at_high = (local_high - entry_premium) / entry_premium * 100
    gain_at_low = (prem - entry_premium) / entry_premium * 100
    pct_of_session_peak = prem / local_high * 100

    retrace_ticks = idx - local_high_idx
    retrace_speed = retrace_depth / max(retrace_ticks, 1)

    # Velocities
    vels = {}
    for lb in [3, 5, 10]:
        if idx >= lb:
            pp = premiums[idx - lb]
            vels[f"vel_{lb}"] = (prem - pp) / pp * 100 if pp > 0 else 0
        else:
            vels[f"vel_{lb}"] = 0

    # Premium std during retrace
    retrace_window = premiums[max(0, local_high_idx):idx + 1]
    retrace_valid = retrace_window[~np.isnan(retrace_window)]
    prem_std = np.std(retrace_valid) / prem if len(retrace_valid) > 1 and prem > 0 else 0

    # Greeks
    delta_val = safe_float(df["delta"].iloc[idx])
    gamma_val = safe_float(df["gamma"].iloc[idx])
    theta_val = safe_float(df["theta"].iloc[idx])
    vega_val = safe_float(df["vega"].iloc[idx])
    iv_val = safe_float(df["iv"].iloc[idx])

    d5 = safe_float(df["delta"].iloc[idx - 5], delta_val) if idx >= 5 else delta_val
    iv5 = safe_float(df["iv"].iloc[idx - 5], iv_val) if idx >= 5 else iv_val
    delta_change = delta_val - d5
    iv_change = iv_val - iv5
    gdr = gamma_val / abs(delta_val) if abs(delta_val) > 0.01 else 0

    # Volume
    vol = safe_float(df["volume"].iloc[idx])
    if idx >= 10:
        prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
        prev_vols = [v for v in prev_vols if v > 0]
        avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    if idx >= 20:
        vr = [safe_float(df["volume"].iloc[i]) for i in range(idx - 5, idx + 1)]
        vo = [safe_float(df["volume"].iloc[i]) for i in range(idx - 20, idx - 10)]
        ar = np.mean([v for v in vr if v > 0] or [0])
        ao = np.mean([v for v in vo if v > 0] or [1])
        vol_trend = ar / ao if ao > 0 else 1.0
    else:
        vol_trend = 1.0

    # Bid-ask
    bid = safe_float(df["bid"].iloc[idx], prem * 0.95)
    ask = safe_float(df["ask"].iloc[idx], prem * 1.05)
    spread_pct = (ask - bid) / prem * 100 if prem > 0 else 0

    if idx >= 5:
        pb = safe_float(df["bid"].iloc[idx - 5], bid)
        pa = safe_float(df["ask"].iloc[idx - 5], ask)
        pp = premiums[idx - 5]
        ps = (pa - pb) / pp * 100 if pp > 0 else spread_pct
        spread_change = spread_pct - ps
    else:
        spread_change = 0

    bs = safe_float(df["bid_size"].iloc[idx] if "bid_size" in df.columns else 1, 1)
    ask_s = safe_float(df["ask_size"].iloc[idx] if "ask_size" in df.columns else 1, 1)
    size_imbalance = (bs - ask_s) / (bs + ask_s) if (bs + ask_s) > 0 else 0

    # Underlying
    u = safe_float(df["underlying_price"].iloc[idx])
    u_move = (u - entry_underlying) / entry_underlying * 100 if entry_underlying > 0 else 0

    if idx >= 5:
        pu = safe_float(df["underlying_price"].iloc[idx - 5], u)
        u_vel = (u - pu) / pu * 100 if pu > 0 else 0
    else:
        u_vel = 0

    u_prices = [safe_float(df["underlying_price"].iloc[i]) for i in range(min(idx + 1, n))]
    u_valid = [x for x in u_prices if x > 0]
    if u_valid:
        u_high = max(u_valid) if is_call else min(u_valid)
        u_retrace = abs(u - u_high) / u_high * 100 if u_high > 0 else 0
    else:
        u_retrace = 0

    moneyness = 0
    if u > 0 and strike > 0:
        moneyness = ((u - strike) / strike * 100) if is_call else ((strike - u) / strike * 100)

    prem_vel = vels.get("vel_5", 0)
    divergence = (prem_vel - u_vel * 50) if is_call else (prem_vel + u_vel * 50)

    minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + et_minute))
    session_progress = max(0, min(1, (et_decimal - 9.5) / 6.5))
    elapsed = (now - timestamps[0]).astype("timedelta64[s]").astype(float) / 60

    n_prior = len(prior_retraces)
    avg_prior = np.mean(prior_retraces) if prior_retraces else 0
    max_prior = max(prior_retraces) if prior_retraces else 0

    return np.array([
        retrace_depth, gain_at_high, gain_at_low, pct_of_session_peak,
        retrace_speed, retrace_ticks,
        vels["vel_3"], vels["vel_5"], vels["vel_10"], prem_std,
        abs(delta_val), gamma_val, theta_val, vega_val, iv_val,
        delta_change, iv_change, gdr,
        vol, vol_ratio, vol_trend,
        spread_pct, spread_change, size_imbalance,
        u_move, u_vel, u_retrace, moneyness, divergence,
        minutes_to_close, session_progress, elapsed,
        1 if category == TickerCategory.HIGH_VOL else 0,
        1 if category == TickerCategory.INDEX else 0,
        1 if is_call else 0,
        n_prior, avg_prior, max_prior,
    ], dtype=np.float64)


def simulate(df, entry_premium, contracts, direction, dte, expiry_date, ticker,
             strike, trail_schedule=None, ml_model=None, ml_threshold=0.5):
    """Run FSM with optional ML-gated progressive trail.

    When ml_model is provided:
      - Track local high and retraces
      - On a 15%+ retrace, ask ML: "is this final?"
      - If ML says YES (prob > threshold) AND peak gain is in a trail tier:
        → apply the tighter trail from that tier
      - If ML says NO (temporary) → keep normal FSM trail

    When ml_model is None and trail_schedule is set:
      - Blind progressive trail (no ML gating)
    """
    if entry_premium <= 0:
        return None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
    option_type = "put" if direction in ("bearish", "put") else "call"
    is_call = option_type == "call"
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

    premiums = df["premium"].values.astype(float)
    timestamps = pd.to_datetime(df["ts"]).values

    locked_pnl = 0.0
    remaining = contracts

    # Retrace tracking for ML
    local_high = entry_premium
    local_high_idx = 0
    in_retrace = False
    ml_said_final = False  # once ML says final, enable tight trail for rest of trade
    prior_retraces = []
    ml_decisions = []

    for idx in range(1, len(df)):
        premium = premiums[idx]
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

        # Track local high
        if premium > local_high:
            local_high = premium
            local_high_idx = idx
            in_retrace = False

        peak_gain = (local_high - entry_premium) / entry_premium * 100
        current_gain = (premium - entry_premium) / entry_premium * 100
        drop_from_peak = (local_high - premium) / local_high * 100 if local_high > 0 else 0

        # ML retrace detection
        if ml_model is not None and not ml_said_final:
            if drop_from_peak >= RETRACE_THRESHOLD and not in_retrace:
                in_retrace = True
                # Only ask ML if we're at a meaningful gain
                if peak_gain >= 50:
                    features = extract_retrace_features(
                        df, idx, entry_premium, local_high, local_high_idx,
                        ticker, is_call, strike, first_underlying,
                        prior_retraces, timestamps,
                    )
                    if features is not None:
                        prob = ml_model.predict(features.reshape(1, -1))[0]
                        is_final = prob > ml_threshold
                        ml_decisions.append({
                            "tick": idx,
                            "peak_gain": peak_gain,
                            "current_gain": current_gain,
                            "retrace_depth": drop_from_peak,
                            "prob_final": prob,
                            "decision": "FINAL" if is_final else "TEMPORARY",
                        })
                        if is_final:
                            ml_said_final = True

                prior_retraces.append(drop_from_peak)

            elif drop_from_peak < 5:
                in_retrace = False

        # Progressive trail check
        use_tight_trail = False
        if trail_schedule:
            if ml_model is not None:
                # ML-gated: only use tight trail if ML said "final"
                use_tight_trail = ml_said_final
            else:
                # Blind: always use tight trail
                use_tight_trail = True

        if use_tight_trail and trail_schedule:
            active_trail = None
            for tier_gain, tier_trail in trail_schedule:
                if peak_gain >= tier_gain:
                    active_trail = tier_trail
                    break

            if active_trail is not None and drop_from_peak >= active_trail:
                pnl = locked_pnl + (premium - entry_premium) * remaining * 100
                return {
                    "pnl": pnl,
                    "reason": f"ml_prog_trail_{active_trail}%",
                    "exit_gain": current_gain,
                    "peak_gain": peak_gain,
                    "progressive": True,
                    "trail_width": active_trail,
                    "ml_decisions": ml_decisions,
                    "ml_gated": ml_model is not None,
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
                "ml_decisions": ml_decisions,
                "ml_gated": ml_model is not None,
            }

    last_prem = premiums[-1] if len(premiums) > 0 else entry_premium
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    return {
        "pnl": pnl, "reason": "last_tick",
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "peak_gain": peak_gain, "progressive": False,
        "ml_decisions": ml_decisions, "ml_gated": ml_model is not None,
    }


def main():
    print("Loading signals and model...")
    signals = load_signals()

    ml_model = lgb.Booster(model_file=str(MODEL_DIR / "retrace_classifier.txt"))
    print(f"  {len(signals)} signals, retrace classifier loaded ({ml_model.num_feature()} features)")

    hconn = sqlite3.connect(HARVESTER_DB)

    # Test configs: schedule × ML threshold
    configs = {}

    # Baseline (no progressive trail)
    configs["baseline"] = {"schedule": None, "ml": False, "threshold": 0}

    # Blind progressive trails (no ML)
    for sname, sched in SCHEDULES.items():
        configs[f"blind_{sname}"] = {"schedule": sched, "ml": False, "threshold": 0}

    # ML-gated progressive trails at different thresholds
    for sname, sched in SCHEDULES.items():
        for th in [0.3, 0.5, 0.7, 0.9]:
            configs[f"ml_{sname}_th{th}"] = {"schedule": sched, "ml": True, "threshold": th}

    all_results = {k: [] for k in configs}

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
        strike = sig["strike"]

        for cname, cfg in configs.items():
            r = simulate(
                df, entry_premium, contracts, direction, dte, expiry_date, ticker, strike,
                trail_schedule=cfg["schedule"],
                ml_model=ml_model if cfg["ml"] else None,
                ml_threshold=cfg["threshold"],
            )
            if r:
                r["ticker"] = ticker
                r["contracts"] = contracts
                all_results[cname].append(r)

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(signals)}, matched {matched}")

    hconn.close()

    # --- Summary table ---
    bl_total = sum(r["pnl"] for r in all_results["baseline"])

    print(f"\n{'=' * 120}")
    print(f"RESULTS: {matched} signals matched")
    print(f"{'=' * 120}")
    print(f"{'Config':<30} {'Total P&L':>12} {'Delta':>10} {'WR':>6} "
          f"{'Prog':>5} {'ML Checks':>10} {'ML Final':>9}")
    print(f"{'-' * 120}")

    for cname, results in all_results.items():
        total = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        wr = wins / len(results) * 100 if results else 0
        d = total - bl_total
        prog = sum(1 for r in results if r.get("progressive"))

        # Count ML decisions
        ml_checks = sum(len(r.get("ml_decisions", [])) for r in results)
        ml_finals = sum(
            sum(1 for d in r.get("ml_decisions", []) if d["decision"] == "FINAL")
            for r in results
        )

        d_str = f"${d:>+9,.0f}" if cname != "baseline" else "—"
        print(f"  {cname:<28} ${total:>10,.0f} {d_str:>10} {wr:>5.1f}% "
              f"{prog:>5} {ml_checks:>10} {ml_finals:>9}")

    # --- Detail on best ML-gated configs ---
    best_ml = max(
        [(k, sum(r["pnl"] for r in v)) for k, v in all_results.items() if k.startswith("ml_")],
        key=lambda x: x[1],
    )
    best_name = best_ml[0]
    best_results = all_results[best_name]
    bl_results = all_results["baseline"]

    print(f"\n{'=' * 120}")
    print(f"BEST ML CONFIG: {best_name} (${best_ml[1]:,.0f})")
    print(f"{'=' * 120}")

    # Show all progressive exits
    prog_trades = [(r, bl) for r, bl in zip(best_results, bl_results) if r.get("progressive")]
    if prog_trades:
        print(f"\n  Progressive exits ({len(prog_trades)}):")
        for r, bl in prog_trades:
            d = r["pnl"] - bl["pnl"]
            status = "WIN" if d > 0 else "LOSS" if d < 0 else "SAME"
            print(f"    {r['ticker']:<8} peak +{r['peak_gain']:.0f}% → "
                  f"exit +{r['exit_gain']:.0f}% (trail {r.get('trail_width', '?')}%) vs "
                  f"BL +{bl['exit_gain']:.0f}% ({bl['reason']}) | "
                  f"Δ ${d:>+8,.0f} {status}")

    # Show all ML decisions across trades
    print(f"\n  ML decisions:")
    for r, bl in zip(best_results, bl_results):
        for dec in r.get("ml_decisions", []):
            print(f"    {r['ticker']:<8} peak +{dec['peak_gain']:.0f}%, "
                  f"retrace -{dec['retrace_depth']:.0f}% → "
                  f"prob_final={dec['prob_final']:.3f} → {dec['decision']}")

    # --- Per-ticker comparison: blind vs ML-gated ---
    print(f"\n{'=' * 120}")
    print("PER-TICKER: Blind vs ML-gated (gentle schedule)")
    print(f"{'=' * 120}")

    blind = all_results.get("blind_gentle", [])
    # Find best ML gentle
    ml_gentle_configs = {k: v for k, v in all_results.items() if k.startswith("ml_gentle")}
    if ml_gentle_configs:
        best_ml_gentle_name = max(ml_gentle_configs, key=lambda k: sum(r["pnl"] for r in ml_gentle_configs[k]))
        ml_gentle = all_results[best_ml_gentle_name]

        tickers = sorted(set(r["ticker"] for r in bl_results))
        for tk in tickers:
            n = sum(1 for r in bl_results if r["ticker"] == tk)
            if n < 3:
                continue
            bl_pnl = sum(r["pnl"] for r in bl_results if r["ticker"] == tk)
            blind_pnl = sum(r["pnl"] for r in blind if r["ticker"] == tk)
            ml_pnl = sum(r["pnl"] for r in ml_gentle if r["ticker"] == tk)
            d_blind = blind_pnl - bl_pnl
            d_ml = ml_pnl - bl_pnl

            print(f"  {tk:<8} ({n:>2}): baseline ${bl_pnl:>8,.0f} | "
                  f"blind ${blind_pnl:>8,.0f} (Δ${d_blind:>+6,.0f}) | "
                  f"ML ${ml_pnl:>8,.0f} (Δ${d_ml:>+6,.0f})")


if __name__ == "__main__":
    main()
