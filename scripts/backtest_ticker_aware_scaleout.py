"""Backtest: Per-ticker runner thresholds + ML retrace classifier for partial sells.

Strategy:
  - Each ticker has a "runner threshold" — the gain % where historically >60%
    of trades go 50%+ higher. BEFORE this threshold, hold all contracts.
  - Tickers with NO runner threshold (TSLA, NVDA, AMD, MSFT, AVGO) get
    AGGRESSIVE partial sells — sell 1/3 at each retrace since they don't reliably run.
  - Tickers WITH a runner threshold get CONSERVATIVE partial sells — only sell
    after the threshold is hit, and only when ML retrace classifier says "FINAL".

Three strategies compared:
  A) Baseline FSM (current V5/V6)
  B) Ticker-aware scaleout WITHOUT ML (fixed partial sells at milestones)
  C) Ticker-aware scaleout WITH ML retrace classifier

Usage:
    python scripts/backtest_ticker_aware_scaleout.py
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
MODEL_DIR = PROJECT_DIR / "journal" / "models"

PORTFOLIO = 8000

SCORE_TIERS = [
    (135, 1.00),
    (120, 0.85),
    (100, 0.85),
    (90, 0.50),
    (78, 0.25),
]

# --- Per-ticker runner thresholds (from per_ticker_runner_odds.py) ---
# Threshold = first gain % where >60% of trades go 50%+ higher (n>=3)
# None = never reliably runs → aggressive partial sells
# V2 thresholds — refined based on backtest v1 results
# Key change: no ticker gets "None" (full aggressive) — instead use higher thresholds
# TSLA at +100% has 58% odds, so threshold=75 (not None)
# Non-runners get threshold=50-75 (moderate) not None (aggressive)
RUNNER_THRESHOLDS = {
    "SPY":   50,    # 88% at +25 but SPY runs are choppy, wait for +50
    "AAPL":  50,    # 60% at +25 but stalls at +75 — sell early
    "MSTR":  75,    # volatile, only 33% at +50, needs bigger cushion
    "PLTR":  50,    # 60% at +25 but drops hard after
    "AMZN":  75,    # 71% at +50, 83% at +75 — very reliable runner
    "META":  75,    # 100% at +50 but small sample, be conservative
    "QQQ":  100,    # 67% at +75, 75% at +100 — needs big move to confirm
    "GOOGL": 100,   # 67% at +75, 100% at +100
    "IWM":  100,    # monster runner when it goes, wait for confirmation
    # These run less reliably — higher thresholds
    "TSLA":  75,    # 57% at +75, 58% at +100 — moderate
    "NVDA": 100,    # only 50% at +100 — needs big confirm
    "MSFT": 100,    # 50% at +100 — rare runner
    "AMD":  150,    # high peak but driven by outliers, rarely runs from signal
    "AVGO":  75,    # 50% at +50/+75
}

CATEGORY_DEFAULTS = {
    TickerCategory.INDEX: 75,
    TickerCategory.HIGH_VOL: 100,
    TickerCategory.STANDARD: 75,
}


def get_runner_threshold(ticker):
    if ticker in RUNNER_THRESHOLDS:
        return RUNNER_THRESHOLDS[ticker]
    cat = categorize_ticker(ticker)
    return CATEGORY_DEFAULTS.get(cat, 75)


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


def simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker):
    """Run production FSM, return per-tick data + final result."""
    if entry_premium <= 0:
        return None, None

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
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
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    tick_data = []
    locked_pnl = 0.0
    remaining = contracts
    exit_result = None

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

        gain_pct = (premium - entry_premium) / entry_premium * 100

        tick_data.append({
            "idx": idx,
            "premium": premium,
            "gain_pct": gain_pct,
            "bid": bid,
            "ask": ask,
            "underlying": underlying,
            "now": now,
            "minutes_to_close": minutes_to_close,
        })

        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            exit_result = {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed, "exit_prem": premium,
                "peak_gain": peak_gain, "exit_gain": gain_pct,
                "contracts_at_exit": remaining,
                "locked_pnl": locked_pnl,
            }
            break

    if exit_result is None:
        last_prem = df["premium"].iloc[-1]
        last_ts = df["ts"].iloc[-1]
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        if last_ts.tzinfo is not None:
            last_ts = last_ts.replace(tzinfo=None)
        elapsed = (last_ts - entry_ts).total_seconds() / 60 if last_ts != entry_ts else 0
        peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
        pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
        exit_result = {
            "pnl": pnl, "reason": "last_tick",
            "hold": elapsed, "exit_prem": last_prem,
            "peak_gain": peak_gain, "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
            "contracts_at_exit": remaining,
            "locked_pnl": locked_pnl,
        }

    return tick_data, exit_result


def simulate_ticker_aware(df, entry_premium, contracts, direction, dte, expiry_date,
                          ticker, use_ml=False, ml_model=None, ml_features=None):
    """Ticker-aware partial sell strategy layered on top of FSM.

    Logic:
    - Track running peak and retraces
    - When gain hits runner_threshold AND a retrace of 15%+ from local peak occurs:
      - If ticker has NO threshold (non-runner): sell 1/3 at first retrace, 1/3 at second
      - If ticker HAS threshold (runner): sell 1/3 only if ML says FINAL retrace
        (or if no ML, sell 1/3 at 2nd retrace to be conservative)
    - FSM still handles the final exit for remaining contracts
    """
    if entry_premium <= 0:
        return None

    threshold = get_runner_threshold(ticker)

    cfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(cfg, settings=_V6_SETTINGS)
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
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=first_underlying,
        dte=dte, expiry_date=expiry_date or "",
    )

    premiums = df["premium"].values.astype(float)

    locked_pnl = 0.0
    remaining = contracts
    partial_sells = 0
    partial_sell_pnl = 0.0
    local_peak = entry_premium
    retrace_count = 0
    in_retrace = False
    partial_sell_log = []

    for idx in range(1, len(df)):
        premium = premiums[idx]
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

        gain_pct = (premium - entry_premium) / entry_premium * 100

        # Track local peak and retraces
        if premium > local_peak:
            local_peak = premium
            in_retrace = False

        retrace_from_local = (local_peak - premium) / local_peak * 100 if local_peak > 0 else 0

        # Detect retrace start (15%+ from local peak)
        if not in_retrace and retrace_from_local >= 15:
            in_retrace = True
            retrace_count += 1

        # --- Partial sell decision ---
        if remaining > 1 and in_retrace and partial_sells < 2:
            peak_gain_pct = (local_peak - entry_premium) / entry_premium * 100

            should_partial = False

            # Only partial sell after gain exceeds ticker's runner threshold
            # AND we're on the 2nd+ retrace (first retrace is often temporary)
            if peak_gain_pct >= threshold and retrace_count >= 2:
                should_partial = True
            # More aggressive: if gain is 2x the threshold, sell on 1st retrace
            elif peak_gain_pct >= threshold * 2 and retrace_count >= 1:
                should_partial = True

            if should_partial:
                # Sell 1/3 of remaining (min 1)
                to_sell = max(1, remaining // 3)
                if remaining - to_sell < 1:
                    to_sell = remaining - 1  # always keep at least 1

                if to_sell > 0:
                    sell_pnl = (premium - entry_premium) * to_sell * 100
                    locked_pnl += sell_pnl
                    partial_sell_pnl += sell_pnl
                    remaining -= to_sell
                    partial_sells += 1
                    state.contracts = remaining
                    partial_sell_log.append({
                        "tick": idx,
                        "sold": to_sell,
                        "at_gain": gain_pct,
                        "peak_gain": peak_gain_pct,
                        "retrace": retrace_from_local,
                        "pnl": sell_pnl,
                    })
                    in_retrace = False  # reset after partial sell

        # --- FSM evaluation for final exit ---
        action = fsm.evaluate(state, premium, bid, ask, now,
                              current_underlying=underlying,
                              minutes_to_close=minutes_to_close)

        if action.should_exit:
            if action.contracts_to_close > 0 and action.contracts_to_close < remaining:
                closed = action.contracts_to_close
                locked_pnl += (premium - entry_premium) * closed * 100
                remaining -= closed
                state.contracts = remaining
                continue

            final_pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            return {
                "pnl": final_pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "exit_gain": gain_pct,
                "partial_sells": partial_sells,
                "partial_sell_pnl": partial_sell_pnl,
                "contracts_remaining": remaining,
                "partial_log": partial_sell_log,
            }

    # End of data — last tick exit
    last_prem = premiums[-1] if len(premiums) > 0 else entry_premium
    last_ts = df["ts"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    elapsed = (last_ts - entry_ts).total_seconds() / 60 if last_ts != entry_ts else 0
    peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
    final_pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {
        "pnl": final_pnl,
        "reason": "last_tick",
        "hold": elapsed,
        "exit_prem": last_prem,
        "peak_gain": peak_gain,
        "exit_gain": (last_prem - entry_premium) / entry_premium * 100,
        "partial_sells": partial_sells,
        "partial_sell_pnl": partial_sell_pnl,
        "contracts_remaining": remaining,
        "partial_log": partial_sell_log,
    }


def _build_retrace_features_for_tick(df, idx, entry_premium, ticker, is_call,
                                      entry_underlying, retrace_depth, n_prior_retraces):
    """Build simplified retrace features for ML classifier."""
    if idx < 10:
        return None

    premiums = df["premium"].values.astype(float)
    premium = premiums[idx]
    if np.isnan(premium) or premium <= 0:
        return None

    # Basic features the retrace classifier uses
    gain_pct = (premium - entry_premium) / entry_premium * 100

    # Velocities
    vels = {}
    for lb in [3, 5, 10]:
        if idx >= lb:
            prev = premiums[idx - lb]
            vels[f"vel_{lb}"] = (premium - prev) / prev * 100 if prev > 0 else 0
        else:
            vels[f"vel_{lb}"] = 0

    # Premium stats
    window = min(30, idx)
    recent = premiums[idx - window:idx + 1]
    recent_valid = recent[~np.isnan(recent)]
    if len(recent_valid) > 1:
        prem_std = np.std(recent_valid) / premium if premium > 0 else 0
    else:
        prem_std = 0

    # Greeks
    delta_val = safe_float(df["delta"].iloc[idx])
    gamma_val = safe_float(df["gamma"].iloc[idx])
    theta_val = safe_float(df["theta"].iloc[idx])
    vega_val = safe_float(df["vega"].iloc[idx])
    iv_val = safe_float(df["iv"].iloc[idx])

    # Time
    now = df["ts"].iloc[idx]
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    et_hour = now.hour - 4
    if et_hour < 0:
        et_hour += 24
    minutes_to_close = max(0, 16 * 60 - (et_hour * 60 + now.minute))

    # Volume
    vol = safe_float(df["volume"].iloc[idx])
    if idx >= 10:
        prev_vols = [safe_float(df["volume"].iloc[i]) for i in range(idx - 10, idx)]
        prev_vols = [v for v in prev_vols if v > 0]
        avg_vol = np.mean(prev_vols) if prev_vols else max(vol, 1)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    # Underlying
    underlying = safe_float(df["underlying_price"].iloc[idx])
    u_retrace = 0
    if entry_underlying > 0 and underlying > 0:
        running_u_max = max(safe_float(df["underlying_price"].iloc[i]) for i in range(idx + 1))
        u_retrace = (running_u_max - underlying) / running_u_max * 100 if running_u_max > 0 else 0

    # Spread
    bid = safe_float(df["bid"].iloc[idx], premium * 0.95)
    ask = safe_float(df["ask"].iloc[idx], premium * 1.05)
    spread_pct = (ask - bid) / premium * 100 if premium > 0 else 0

    category = categorize_ticker(ticker)

    # Return features in the order the retrace classifier expects
    # (simplified — we'll match the feature names from training)
    return [
        retrace_depth,                          # retrace_depth
        vels["vel_3"],                          # retrace_speed (approx)
        5,                                       # retrace_ticks (approx — not exact)
        gain_pct,                               # gain_pct
        vels["vel_3"],                          # vel_3
        vels["vel_5"],                          # vel_5
        vels["vel_10"],                         # vel_10
        prem_std,                               # prem_std
        abs(delta_val),                         # delta
        gamma_val,                              # gamma
        theta_val,                              # theta
        vega_val,                               # vega
        iv_val,                                 # iv
        vol,                                    # volume
        vol_ratio,                              # vol_ratio
        underlying,                             # underlying_price
        u_retrace,                              # u_retrace_pct
        spread_pct,                             # spread_pct
        minutes_to_close,                       # minutes_to_close
        n_prior_retraces,                       # n_prior_retraces
        0,                                       # avg_prior_retrace_depth (unknown)
        0,                                       # max_prior_retrace_depth (unknown)
        1 if category == TickerCategory.HIGH_VOL else 0,
        1 if category == TickerCategory.INDEX else 0,
        1 if is_call else 0,
    ]


def main():
    print("Loading signals...")
    signals = load_signals()
    print(f"  {len(signals)} signals")

    # Try loading ML retrace classifier
    ml_model = None
    retrace_model_path = MODEL_DIR / "retrace_classifier.txt"
    if retrace_model_path.exists():
        ml_model = lgb.Booster(model_file=str(retrace_model_path))
        print(f"  Loaded retrace classifier from {retrace_model_path}")
    else:
        print("  No retrace classifier found — ML strategy will be skipped")

    print("Connecting to harvester DB...")
    hconn = sqlite3.connect(HARVESTER_DB)

    results_baseline = []
    results_ticker_aware = []
    results_ml = []

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

        # A) Baseline FSM
        _, baseline = simulate_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker)
        if baseline is None:
            continue
        baseline["ticker"] = ticker
        baseline["contracts"] = contracts
        baseline["score"] = score
        results_baseline.append(baseline)

        # B) Ticker-aware scaleout (no ML)
        ta_result = simulate_ticker_aware(
            df, entry_premium, contracts, direction, dte, expiry_date, ticker,
            use_ml=False,
        )
        if ta_result:
            ta_result["ticker"] = ticker
            ta_result["contracts"] = contracts
            ta_result["score"] = score
            results_ticker_aware.append(ta_result)

        # C) Ticker-aware + ML retrace classifier (skip if feature mismatch)
        # ML integration deferred — feature count mismatch with retrace classifier

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(signals)}, matched {matched}")

    hconn.close()

    # --- Report ---
    print(f"\n{'=' * 100}")
    print(f"RESULTS: {matched} signals matched to harvester data")
    print(f"{'=' * 100}")

    def summarize(results, label):
        if not results:
            return
        pnls = [r["pnl"] for r in results]
        wins = [r for r in results if r["pnl"] > 0]
        total = sum(pnls)
        wr = len(wins) / len(results) * 100 if results else 0
        avg = np.mean(pnls)
        median = np.median(pnls)
        print(f"\n  {label}")
        print(f"    Total P&L: ${total:,.0f}")
        print(f"    Trades: {len(results)}, Wins: {len(wins)} ({wr:.1f}%)")
        print(f"    Avg P&L: ${avg:,.0f}, Median: ${median:,.0f}")
        print(f"    Best: ${max(pnls):,.0f}, Worst: ${min(pnls):,.0f}")
        return total

    bl_total = summarize(results_baseline, "A) Baseline FSM (current production)")
    ta_total = summarize(results_ticker_aware, "B) Ticker-aware scaleout (no ML)")
    ml_total = summarize(results_ml, "C) Ticker-aware + ML retrace classifier") if ml_model else None

    # --- Delta analysis ---
    if results_ticker_aware:
        print(f"\n{'=' * 100}")
        print("DELTA: Ticker-aware vs Baseline (per trade)")
        print(f"{'=' * 100}")

        better = 0
        worse = 0
        same = 0
        delta_by_ticker = {}

        for bl, ta in zip(results_baseline, results_ticker_aware):
            d = ta["pnl"] - bl["pnl"]
            tk = bl["ticker"]
            if d > 1:
                better += 1
            elif d < -1:
                worse += 1
            else:
                same += 1

            if tk not in delta_by_ticker:
                delta_by_ticker[tk] = {"deltas": [], "trades": 0, "partial_sells": 0}
            delta_by_ticker[tk]["deltas"].append(d)
            delta_by_ticker[tk]["trades"] += 1
            delta_by_ticker[tk]["partial_sells"] += ta.get("partial_sells", 0)

        print(f"\n  Better: {better}, Worse: {worse}, Same: {same}")
        print(f"\n  Per-ticker delta:")
        for tk, data in sorted(delta_by_ticker.items(), key=lambda x: sum(x[1]["deltas"]), reverse=True):
            total_delta = sum(data["deltas"])
            n = data["trades"]
            ps = data["partial_sells"]
            threshold = get_runner_threshold(tk)
            th_str = f"+{threshold}%" if threshold else "NONE"
            print(f"    {tk:<8} ({th_str:<8}): {total_delta:>+8,.0f} over {n:>2} trades ({ps} partial sells)")

    # --- Partial sell details ---
    if results_ticker_aware:
        print(f"\n{'=' * 100}")
        print("PARTIAL SELL DETAILS")
        print(f"{'=' * 100}")

        trades_with_partials = [r for r in results_ticker_aware if r.get("partial_sells", 0) > 0]
        print(f"\n  {len(trades_with_partials)} trades had partial sells")

        if trades_with_partials:
            total_partial_pnl = sum(r["partial_sell_pnl"] for r in trades_with_partials)
            print(f"  Total partial sell P&L: ${total_partial_pnl:,.0f}")

            for r in trades_with_partials:
                logs = r.get("partial_log", [])
                pnl_diff = r["pnl"] - next(
                    (bl["pnl"] for bl in results_baseline if bl["ticker"] == r["ticker"]),
                    r["pnl"]
                )
                for log in logs:
                    gain_str = f"+{log['at_gain']:.0f}%"
                    peak_str = f"peak +{log['peak_gain']:.0f}%"
                    print(f"    {r['ticker']:<8} sold {log['sold']} @ {gain_str} ({peak_str}, "
                          f"retrace -{log['retrace']:.0f}%) → ${log['pnl']:,.0f}")

    # --- Where did partial sells HELP vs HURT? ---
    print(f"\n{'=' * 100}")
    print("PARTIAL SELL IMPACT: Did locking profit help or hurt per trade?")
    print(f"{'=' * 100}")

    helped = []
    hurt = []
    for bl, ta in zip(results_baseline, results_ticker_aware):
        if ta.get("partial_sells", 0) == 0:
            continue
        d = ta["pnl"] - bl["pnl"]
        entry = {
            "ticker": bl["ticker"],
            "delta": d,
            "bl_pnl": bl["pnl"],
            "ta_pnl": ta["pnl"],
            "bl_exit_gain": bl.get("exit_gain", 0),
            "ta_exit_gain": ta.get("exit_gain", 0),
            "peak_gain": bl.get("peak_gain", 0),
            "partial_sell_pnl": ta.get("partial_sell_pnl", 0),
            "partial_sells": ta.get("partial_sells", 0),
            "bl_reason": bl.get("reason", "?"),
            "ta_reason": ta.get("reason", "?"),
        }
        if d > 0:
            helped.append(entry)
        else:
            hurt.append(entry)

    print(f"\n  HELPED ({len(helped)} trades, total: ${sum(e['delta'] for e in helped):+,.0f}):")
    for e in sorted(helped, key=lambda x: -x["delta"]):
        print(f"    {e['ticker']:<8} {e['delta']:>+8,.0f} | "
              f"baseline: ${e['bl_pnl']:>7,.0f} (exit {e['bl_exit_gain']:>+.0f}%, {e['bl_reason']}) → "
              f"ticker-aware: ${e['ta_pnl']:>7,.0f} (partial locked ${e['partial_sell_pnl']:>,.0f})")

    print(f"\n  HURT ({len(hurt)} trades, total: ${sum(e['delta'] for e in hurt):+,.0f}):")
    for e in sorted(hurt, key=lambda x: x["delta"]):
        print(f"    {e['ticker']:<8} {e['delta']:>+8,.0f} | "
              f"baseline: ${e['bl_pnl']:>7,.0f} (exit {e['bl_exit_gain']:>+.0f}%, peak +{e['peak_gain']:.0f}%, {e['bl_reason']}) → "
              f"ticker-aware: ${e['ta_pnl']:>7,.0f} (partial locked ${e['partial_sell_pnl']:>,.0f})")

    # --- Category breakdown ---
    print(f"\n{'=' * 100}")
    print("BY CATEGORY: Runner vs Non-Runner tickers")
    print(f"{'=' * 100}")

    for label, results in [("Baseline", results_baseline), ("Ticker-aware", results_ticker_aware)]:
        runner_pnl = sum(r["pnl"] for r in results if get_runner_threshold(r["ticker"]) is not None)
        nonrunner_pnl = sum(r["pnl"] for r in results if get_runner_threshold(r["ticker"]) is None)
        runner_n = sum(1 for r in results if get_runner_threshold(r["ticker"]) is not None)
        nonrunner_n = sum(1 for r in results if get_runner_threshold(r["ticker"]) is None)
        print(f"\n  {label}:")
        print(f"    Runners ({runner_n} trades): ${runner_pnl:,.0f}")
        print(f"    Non-runners ({nonrunner_n} trades): ${nonrunner_pnl:,.0f}")


if __name__ == "__main__":
    main()
