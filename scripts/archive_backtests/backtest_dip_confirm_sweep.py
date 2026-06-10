"""Backtest dip-confirm fixes: VWAP direction block + uptick threshold sweep.

Built on top of backtest_v5_production.py's infrastructure — uses the LIVE
production FSM, production sizing, all V6 gates, momentum gate.

Tests ONE adjustment at a time against the baseline:
  A) Baseline: current production (no VWAP block, any uptick triggers entry)
  B) VWAP direction block only (block puts above VWAP, calls below VWAP)
  C) Uptick threshold only (require N% uptick, not just any $0.02 tick)
  D) VWAP block + uptick threshold combined
  E) Per-ticker category thresholds

Usage:
    python scripts/backtest_dip_confirm_sweep.py
"""

from __future__ import annotations

import sqlite3
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests as _requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

# ---------------------------------------------------------------------------
# Production settings (must match docker-compose.yml exactly)
# ---------------------------------------------------------------------------

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

POLYGON_API_KEY = "Vk7gXTz6dbp_F69UmmqIx9BDEasHfExb"
PORTFOLIO = 23000

# Production score tiers (must match vinny_strategy.py _SCORE_TIER_TABLE)
SCORE_TIERS = [
    (135, 1.00, 0.15),
    (120, 0.85, 0.12),
    (100, 0.85, 0.08),
    (90, 0.50, 0.08),
    (78, 0.25, 0.08),
]

# Ticker categories for per-ticker thresholds
HIGH_VOL = {"MSTR", "AMD", "TSLA", "NVDA", "AVGO", "META", "COIN", "SMCI", "PLTR"}
INDEX = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}

# Uptick thresholds to sweep
UPTICK_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

# Per-category optimal thresholds to test
PER_CATEGORY_CONFIGS = [
    # (label, high_vol_pct, index_pct, standard_pct)
    ("Cat A: HV=3% IDX=1.5% STD=2%", 3.0, 1.5, 2.0),
    ("Cat B: HV=4% IDX=2% STD=2.5%", 4.0, 2.0, 2.5),
    ("Cat C: HV=5% IDX=2% STD=3%", 5.0, 2.0, 3.0),
    ("Cat D: HV=3% IDX=1% STD=1.5%", 3.0, 1.0, 1.5),
    ("Cat E: HV=5% IDX=3% STD=3%", 5.0, 3.0, 3.0),
]


# ---------------------------------------------------------------------------
# Polygon candle fetching (for VWAP computation)
# ---------------------------------------------------------------------------

_polygon_cache: dict[tuple[str, str], list[dict]] = {}


@dataclass(slots=True)
class _CandleBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0


def _fetch_polygon_5m(ticker: str, from_date: str, to_date: str) -> list[dict]:
    key = (ticker, from_date)
    if key in _polygon_cache:
        return _polygon_cache[key]
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute"
        f"/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    for attempt in range(3):
        try:
            resp = _requests.get(url, timeout=30)
            if resp.status_code == 429:
                _time.sleep(12)
                continue
            results = resp.json().get("results", [])
            _polygon_cache[key] = results
            return results
        except Exception:
            _time.sleep(2)
    _polygon_cache[key] = []
    return []


def compute_vwap_at_entry(ticker: str, date_str: str, entry_ts_ms: int) -> float | None:
    """Compute VWAP from Polygon 5m bars up to entry time."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    raw = _fetch_polygon_5m(ticker, date_str, date_str)
    if not raw:
        return None
    _time.sleep(0.15)  # rate limit

    total_vp = total_vol = 0.0
    for c in raw:
        if c["t"] > entry_ts_ms:
            break
        vol = c.get("v", 0)
        if vol <= 0:
            continue
        typical = (c["h"] + c["l"] + c["c"]) / 3
        total_vp += typical * vol
        total_vol += vol

    return total_vp / total_vol if total_vol > 0 else None


# ---------------------------------------------------------------------------
# Signal & tick loading (from backtest_v5_production.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Production FSM simulation (exact copy from backtest_v5_production.py)
# ---------------------------------------------------------------------------

def simulate_with_production_fsm(df, entry_premium, contracts, direction, dte, expiry_date, ticker="SIM"):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0, "peak_gain": 0}

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
        entry_underlying_price=first_underlying, dte=dte,
        expiry_date=expiry_date or "",
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
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {
                "pnl": pnl, "reason": action.reason.value,
                "hold": elapsed, "exit_prem": premium, "peak_gain": peak_gain,
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
        "pnl": pnl, "reason": "eod_data_end",
        "hold": elapsed, "exit_prem": last_prem, "peak_gain": peak_gain,
    }


# ---------------------------------------------------------------------------
# Dip-confirm simulation using harvester tick data
# ---------------------------------------------------------------------------

@dataclass
class DipConfirmResult:
    """Result of simulating dip-confirm on a trade."""
    entered: bool
    reason: str
    adjusted_entry: float | None  # entry premium after waiting for dip
    savings_pct: float  # % cheaper than original entry


def simulate_dip_confirm(
    df: pd.DataFrame,
    entry_premium: float,
    option_type: str,
    ticker: str,
    vwap: float | None,
    uptick_threshold_pct: float,
    enable_vwap_block: bool,
    fade_pct: float = 1.0,
) -> DipConfirmResult:
    """Simulate the dip-confirm gate using harvester tick data.

    Matches production logic in paper_trader.py _wait_for_entry_confirmation().

    Steps:
    1. Check if premium fading in first ~60s
    2. If fading, check VWAP direction (if enabled)
    3. Poll for uptick exceeding threshold
    """
    if len(df) < 3:
        return DipConfirmResult(True, "no_data", None, 0)

    t0_premium = entry_premium
    # t1 = premium ~60s after signal (harvester ticks are ~60s apart)
    t1_premium = df["premium"].iloc[1] if len(df) > 1 else t0_premium

    # Check if fading
    fade = (t0_premium - t1_premium) / t0_premium * 100 if t0_premium > 0 else 0
    if fade < fade_pct:
        # Not fading — enter immediately (current prod behavior)
        return DipConfirmResult(True, "not_fading", t1_premium, 0)

    # Premium IS fading — check VWAP direction
    if enable_vwap_block and vwap is not None:
        underlying = None
        for i in range(min(3, len(df))):
            u = df["underlying_price"].iloc[i]
            if u and u > 0:
                underlying = float(u)
                break

        if underlying is not None:
            above_vwap = underlying >= vwap

            # VWAP direction validation (the FIX for GOOGL bug):
            # Above VWAP + call = bullish structure → enter
            if above_vwap and option_type == "call":
                return DipConfirmResult(True, "above_vwap_call", t1_premium,
                                       (t0_premium - t1_premium) / t0_premium * 100 if t1_premium else 0)
            # Below VWAP + put = bearish structure → enter
            if not above_vwap and option_type == "put":
                return DipConfirmResult(True, "below_vwap_put", t1_premium,
                                       (t0_premium - t1_premium) / t0_premium * 100 if t1_premium else 0)

            # BLOCK: above VWAP + put = counter-trend (GOOGL bug)
            if above_vwap and option_type == "put":
                # Don't enter immediately — this is the wrong direction
                # Fall through to uptick polling with stricter threshold
                pass

            # BLOCK: below VWAP + call = counter-trend
            if not above_vwap and option_type == "call":
                pass

    # Poll for uptick exceeding threshold
    # Use harvester ticks (first 5 ticks after t1 = ~5 minutes)
    prev = t1_premium
    low_water = t1_premium
    max_poll_ticks = min(5, len(df) - 2)  # look at next 5 ticks (~5 min at 60s resolution)

    for i in range(2, 2 + max_poll_ticks):
        if i >= len(df):
            break
        current = df["premium"].iloc[i]
        if pd.isna(current) or current <= 0:
            continue

        if current < low_water:
            low_water = current

        if current > prev:
            # Uptick detected — check if it exceeds threshold
            uptick_pct = (current - low_water) / low_water * 100 if low_water > 0 else 0

            if uptick_threshold_pct <= 0 or uptick_pct >= uptick_threshold_pct:
                savings = (t0_premium - current) / t0_premium * 100 if t0_premium > 0 else 0
                return DipConfirmResult(True, f"uptick_{uptick_pct:.1f}pct", current, savings)

        prev = current

    # No qualifying uptick — skip trade
    return DipConfirmResult(False, "no_uptick", None, 0)


# ---------------------------------------------------------------------------
# Entry gates (production logic)
# ---------------------------------------------------------------------------

def apply_entry_gates(sig, df, score):
    """Apply all production entry gates. Returns (pass, contracts, adj_entry, reason)."""
    direction = (sig["direction"] or "bullish").lower()
    entry_premium = sig["premium"]

    # Market entry from harvester
    first_ask = df["ask"].iloc[0]
    first_mid = df["premium"].iloc[0]
    adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
    if adj_entry <= 0:
        adj_entry = entry_premium

    # Score floor
    if score < 78:
        return False, 0, adj_entry, "score_below_78"

    # V6 premium cap
    if _V6_SETTINGS.ENABLE_V6_PREMIUM_CAP:
        cap = _V6_SETTINGS.V6_PREMIUM_CAP
        if score >= 150:
            cap = _V6_SETTINGS.V6_PREMIUM_CAP_HIGH
        elif score >= 120:
            cap = _V6_SETTINGS.V6_PREMIUM_CAP_MID
        if adj_entry > cap:
            return False, 0, adj_entry, "premium_cap"

    # V6 spread gate
    if _V6_SETTINGS.ENABLE_V6_SPREAD_GATE and len(df) > 0:
        first_bid = df["bid"].iloc[0]
        first_ask_val = df["ask"].iloc[0]
        if first_bid and first_ask_val and first_bid > 0 and first_ask_val > 0:
            spread_pct = (first_ask_val - first_bid) / first_ask_val * 100
            if spread_pct > _V6_SETTINGS.V6_MAX_SPREAD_PCT:
                return False, 0, adj_entry, "spread_gate"

    # Sizing (production: dollar-target with score tiers)
    max_risk_pct = 0.75
    max_concurrent = 4
    max_position_pct = 0.15
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent

    score_mult = 0.25
    tier_pos_pct = 0.08
    for threshold, mult, pos_pct in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            tier_pos_pct = pos_pct
            break

    effective_pos_pct = min(tier_pos_pct, max_position_pct)
    position_cap = PORTFOLIO * effective_pos_pct

    cost_per = adj_entry * 100
    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
    if pos_cap_contracts == 0:
        return False, 0, adj_entry, "pos_cap_zero"
    contracts = max(1, min(raw_contracts, pos_cap_contracts))

    # Late-session 0DTE size reduction
    dte = sig.get("_dte", 0)
    if dte == 0 and contracts > 1:
        sig_time = sig["created_at"]
        try:
            sig_dt_full = (
                datetime.strptime(sig_time[:19], "%Y-%m-%dT%H:%M:%S")
                if "T" in sig_time
                else datetime.strptime(sig_time[:19], "%Y-%m-%d %H:%M:%S")
            )
            et_hour = sig_dt_full.hour - 4
            if et_hour < 0:
                et_hour += 24
            if et_hour >= 14:
                contracts = 1
            elif et_hour >= 13:
                contracts = max(1, contracts // 2)
        except (ValueError, TypeError):
            pass

    return True, contracts, adj_entry, "passed"


def check_momentum_gate(df, direction):
    """Simulate MomentumConfirmGate. Returns (blocked, reason)."""
    is_call = direction in ("bullish", "call")
    window = min(15, len(df))
    underlying_prices = []
    for i in range(window):
        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            underlying_prices.append(float(u))

    if len(underlying_prices) < 5:
        return False, ""

    first_half = underlying_prices[:len(underlying_prices)//2]
    second_half = underlying_prices[len(underlying_prices)//2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    pct_move = (avg_second - avg_first) / avg_first * 100

    prem_start = df["premium"].iloc[0]
    prem_5 = df["premium"].iloc[min(4, len(df)-1)]
    prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0

    neg_signals = 0
    reason_parts = []

    if is_call and pct_move < -0.05:
        neg_signals += 1
        reason_parts.append(f"underlying fading ({pct_move:+.2f}%)")
    elif not is_call and pct_move > 0.05:
        neg_signals += 1
        reason_parts.append(f"underlying rising ({pct_move:+.2f}%)")

    if prem_fade < -5:
        neg_signals += 1
        reason_parts.append(f"premium fading ({prem_fade:+.1f}%)")

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
        reason_parts.append("3/3 bars against")

    return neg_signals >= 2, "; ".join(reason_parts)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB\n")

    harvester_conn = sqlite3.connect(HARVESTER_DB)

    # Phase 1: Load all trades that pass entry gates + momentum gate
    # and compute their FSM results
    print("=" * 100)
    print("PHASE 1: Loading trades and computing baseline FSM results")
    print("=" * 100)

    @dataclass
    class TradeData:
        sig: dict
        df: pd.DataFrame
        contracts: int
        adj_entry: float
        direction: str
        dte: int
        expiry_date: str
        ticker: str
        score: int
        option_type: str
        vwap: float | None
        fsm_result: dict

    trades: list[TradeData] = []
    no_data = 0
    blocked_momentum = 0
    blocked_gates = 0

    for i, sig in enumerate(signals):
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]

        df = load_ticks(harvester_conn, sig)
        if df is None:
            no_data += 1
            continue

        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")

        passed, contracts, adj_entry, gate_reason = apply_entry_gates(sig, df, score)
        if not passed:
            blocked_gates += 1
            continue

        mom_blocked, mom_reason = check_momentum_gate(df, direction)
        if mom_blocked:
            blocked_momentum += 1
            continue

        # Compute VWAP (for VWAP direction block testing)
        sig_time = sig["created_at"]
        try:
            sig_dt_full = (
                datetime.strptime(sig_time[:19], "%Y-%m-%dT%H:%M:%S")
                if "T" in sig_time
                else datetime.strptime(sig_time[:19], "%Y-%m-%d %H:%M:%S")
            )
            entry_ts_ms = int(sig_dt_full.timestamp() * 1000)
        except (ValueError, TypeError):
            entry_ts_ms = 0

        vwap = compute_vwap_at_entry(ticker, day, entry_ts_ms) if entry_ts_ms > 0 else None

        option_type = sig["option_type"]

        # Run FSM with original entry
        fsm_result = simulate_with_production_fsm(
            df, adj_entry, contracts, direction, dte, expiry_date, ticker=ticker,
        )

        trades.append(TradeData(
            sig=sig, df=df, contracts=contracts, adj_entry=adj_entry,
            direction=direction, dte=dte, expiry_date=expiry_date,
            ticker=ticker, score=score, option_type=option_type,
            vwap=vwap, fsm_result=fsm_result,
        ))

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(signals)} signals, {len(trades)} trades loaded...")

    harvester_conn.close()

    print(f"\nPhase 1 complete:")
    print(f"  Total signals: {len(signals)}")
    print(f"  No tick data: {no_data}")
    print(f"  Blocked by entry gates: {blocked_gates}")
    print(f"  Blocked by momentum: {blocked_momentum}")
    print(f"  Trades for dip-confirm testing: {len(trades)}")

    baseline_pnl = sum(t.fsm_result["pnl"] for t in trades)
    baseline_wins = sum(1 for t in trades if t.fsm_result["pnl"] > 0)
    baseline_wr = baseline_wins / len(trades) * 100 if trades else 0

    print(f"\nBASELINE (no dip-confirm):")
    print(f"  P&L: ${baseline_pnl:,.2f}")
    print(f"  Win Rate: {baseline_wr:.1f}% ({baseline_wins}/{len(trades)})")

    vwap_available = sum(1 for t in trades if t.vwap is not None)
    print(f"  VWAP data available: {vwap_available}/{len(trades)}")

    # ===================================================================
    # PHASE 2: Test adjustments ONE AT A TIME
    # ===================================================================

    print(f"\n{'=' * 120}")
    print(f"PHASE 2: ADJUSTMENT SWEEP — one change at a time vs baseline")
    print(f"{'=' * 120}\n")

    # Helper to run a dip-confirm config across all trades
    def run_config(
        enable_vwap_block: bool,
        uptick_threshold_pct: float,
        per_category: dict[str, float] | None = None,
    ) -> dict:
        entered_trades = []
        skipped_trades = []

        for t in trades:
            # Determine uptick threshold for this trade
            if per_category:
                if t.ticker in HIGH_VOL:
                    thresh = per_category["high_vol"]
                elif t.ticker in INDEX:
                    thresh = per_category["index"]
                else:
                    thresh = per_category["standard"]
            else:
                thresh = uptick_threshold_pct

            result = simulate_dip_confirm(
                t.df, t.adj_entry, t.option_type, t.ticker,
                t.vwap, thresh, enable_vwap_block,
            )

            if result.entered:
                # If dip-confirm gives a better entry price, re-run FSM with it
                if result.adjusted_entry and result.adjusted_entry < t.adj_entry:
                    new_fsm = simulate_with_production_fsm(
                        t.df, result.adjusted_entry, t.contracts,
                        t.direction, t.dte, t.expiry_date, ticker=t.ticker,
                    )
                    entered_trades.append((t, new_fsm, result))
                else:
                    entered_trades.append((t, t.fsm_result, result))
            else:
                skipped_trades.append((t, result))

        total_pnl = sum(fsm["pnl"] for _, fsm, _ in entered_trades)
        wins = sum(1 for _, fsm, _ in entered_trades if fsm["pnl"] > 0)
        wr = wins / len(entered_trades) * 100 if entered_trades else 0

        skip_bad = sum(t.fsm_result["pnl"] for t, _ in skipped_trades if t.fsm_result["pnl"] < 0)
        skip_good = sum(t.fsm_result["pnl"] for t, _ in skipped_trades if t.fsm_result["pnl"] > 0)

        return {
            "entered": len(entered_trades),
            "skipped": len(skipped_trades),
            "pnl": total_pnl,
            "delta": total_pnl - baseline_pnl,
            "wr": wr,
            "skip_bad": skip_bad,
            "skip_good": skip_good,
            "entered_trades": entered_trades,
            "skipped_trades": skipped_trades,
        }

    # --- Test A: VWAP direction block ONLY (no uptick threshold change) ---
    print(f"{'=' * 120}")
    print("TEST A: VWAP Direction Block ONLY (block puts above VWAP, calls below VWAP)")
    print(f"{'=' * 120}\n")

    r = run_config(enable_vwap_block=True, uptick_threshold_pct=0.0)
    print(f"  Entered: {r['entered']}  Skipped: {r['skipped']}")
    print(f"  P&L: ${r['pnl']:,.2f}  (delta: ${r['delta']:+,.2f})")
    print(f"  Win Rate: {r['wr']:.1f}%")
    print(f"  Avoided losses: ${r['skip_bad']:,.2f}")
    print(f"  Missed gains: ${r['skip_good']:,.2f}")
    print(f"  Net benefit: ${abs(r['skip_bad']) - r['skip_good']:+,.2f}")

    if r['skipped_trades']:
        print(f"\n  Skipped trades detail:")
        print(f"  {'Ticker':<8} {'Type':<5} {'Score':>5} {'Entry':>7} {'WouldPnL':>10} {'Reason':<25}")
        print(f"  {'-'*70}")
        for t, dc_result in sorted(r['skipped_trades'], key=lambda x: x[0].fsm_result['pnl']):
            print(f"  {t.ticker:<8} {t.option_type:<5} {t.score:>5} ${t.adj_entry:>5.2f} "
                  f"${t.fsm_result['pnl']:>+9,.2f} {dc_result.reason:<25}")

    # --- Test B: Uptick threshold sweep ONLY (no VWAP block) ---
    print(f"\n{'=' * 120}")
    print("TEST B: Uptick Threshold Sweep ONLY (no VWAP block)")
    print(f"{'=' * 120}\n")

    print(f"{'Threshold':>10} {'Enter':>6} {'Skip':>5} {'PnL':>12} {'Delta':>12} "
          f"{'WR':>6} {'AvoidLoss':>12} {'MissGain':>12} {'NetBenefit':>12}")
    print("-" * 110)

    best_thresh = 0.0
    best_delta = float("-inf")
    thresh_results = {}

    for thresh in UPTICK_THRESHOLDS:
        r = run_config(enable_vwap_block=False, uptick_threshold_pct=thresh)
        thresh_results[thresh] = r
        net = abs(r['skip_bad']) - r['skip_good']
        label = f"{thresh:.1f}%"
        if thresh == 0:
            label = "0% (base)"
        print(f"{label:>10} {r['entered']:>6} {r['skipped']:>5} ${r['pnl']:>10,.2f} ${r['delta']:>+10,.2f} "
              f"{r['wr']:>5.1f}% ${abs(r['skip_bad']):>10,.2f} ${r['skip_good']:>10,.2f} ${net:>+10,.2f}")
        if r['delta'] > best_delta:
            best_delta = r['delta']
            best_thresh = thresh

    print(f"\n  BEST uptick threshold: {best_thresh:.1f}% (delta: ${best_delta:+,.2f})")

    # --- Test C: VWAP block + best uptick threshold ---
    print(f"\n{'=' * 120}")
    print(f"TEST C: VWAP Block + Uptick Threshold Combined")
    print(f"{'=' * 120}\n")

    print(f"{'Threshold':>10} {'Enter':>6} {'Skip':>5} {'PnL':>12} {'Delta':>12} "
          f"{'WR':>6} {'AvoidLoss':>12} {'MissGain':>12} {'NetBenefit':>12}")
    print("-" * 110)

    best_combined_delta = float("-inf")
    best_combined_thresh = 0.0

    for thresh in UPTICK_THRESHOLDS:
        r = run_config(enable_vwap_block=True, uptick_threshold_pct=thresh)
        net = abs(r['skip_bad']) - r['skip_good']
        label = f"{thresh:.1f}%"
        if thresh == 0:
            label = "VWAP only"
        print(f"{label:>10} {r['entered']:>6} {r['skipped']:>5} ${r['pnl']:>10,.2f} ${r['delta']:>+10,.2f} "
              f"{r['wr']:>5.1f}% ${abs(r['skip_bad']):>10,.2f} ${r['skip_good']:>10,.2f} ${net:>+10,.2f}")
        if r['delta'] > best_combined_delta:
            best_combined_delta = r['delta']
            best_combined_thresh = thresh

    print(f"\n  BEST combined: VWAP block + {best_combined_thresh:.1f}% uptick (delta: ${best_combined_delta:+,.2f})")

    # --- Test D: Per-ticker category thresholds ---
    print(f"\n{'=' * 120}")
    print("TEST D: Per-Ticker Category Thresholds (VWAP block + category-specific uptick)")
    print(f"{'=' * 120}\n")

    print(f"{'Config':<40} {'Enter':>6} {'Skip':>5} {'PnL':>12} {'Delta':>12} "
          f"{'WR':>6} {'NetBenefit':>12}")
    print("-" * 105)

    best_cat_delta = float("-inf")
    best_cat_config = ""

    for label, hv, idx, std in PER_CATEGORY_CONFIGS:
        r = run_config(
            enable_vwap_block=True,
            uptick_threshold_pct=0,
            per_category={"high_vol": hv, "index": idx, "standard": std},
        )
        net = abs(r['skip_bad']) - r['skip_good']
        print(f"{label:<40} {r['entered']:>6} {r['skipped']:>5} ${r['pnl']:>10,.2f} ${r['delta']:>+10,.2f} "
              f"{r['wr']:>5.1f}% ${net:>+10,.2f}")
        if r['delta'] > best_cat_delta:
            best_cat_delta = r['delta']
            best_cat_config = label

    print(f"\n  BEST per-category: {best_cat_config} (delta: ${best_cat_delta:+,.2f})")

    # ===================================================================
    # PHASE 3: Per-ticker analysis
    # ===================================================================

    print(f"\n{'=' * 120}")
    print("PHASE 3: Per-Ticker P&L Impact (best combined config)")
    print(f"{'=' * 120}\n")

    # Use best combined config
    best_r = run_config(enable_vwap_block=True, uptick_threshold_pct=best_combined_thresh)

    # Group by ticker
    ticker_baseline: dict[str, list[float]] = {}
    ticker_adjusted: dict[str, list[float]] = {}
    ticker_skipped: dict[str, list[tuple]] = {}

    for t in trades:
        ticker_baseline.setdefault(t.ticker, []).append(t.fsm_result["pnl"])

    for t, fsm, dc in best_r["entered_trades"]:
        ticker_adjusted.setdefault(t.ticker, []).append(fsm["pnl"])

    for t, dc in best_r["skipped_trades"]:
        ticker_skipped.setdefault(t.ticker, []).append((t.fsm_result["pnl"], dc.reason))
        # Skipped trades contribute $0 to adjusted
        ticker_adjusted.setdefault(t.ticker, [])

    all_tickers = sorted(set(ticker_baseline.keys()))
    print(f"{'Ticker':<8} {'Cat':<6} {'Base$':>10} {'New$':>10} {'Delta':>10} "
          f"{'BaseTr':>6} {'NewTr':>6} {'Skip':>4} {'SkipPnL':>10}")
    print("-" * 85)

    for ticker in all_tickers:
        cat = "HV" if ticker in HIGH_VOL else ("IDX" if ticker in INDEX else "STD")
        base_pnls = ticker_baseline.get(ticker, [])
        adj_pnls = ticker_adjusted.get(ticker, [])
        skip_data = ticker_skipped.get(ticker, [])

        base_total = sum(base_pnls)
        adj_total = sum(adj_pnls)
        skip_pnl = sum(p for p, _ in skip_data)
        delta = adj_total - base_total

        print(f"{ticker:<8} {cat:<6} ${base_total:>+9,.2f} ${adj_total:>+9,.2f} ${delta:>+9,.2f} "
              f"{len(base_pnls):>6} {len(adj_pnls):>6} {len(skip_data):>4} ${skip_pnl:>+9,.2f}")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================

    print(f"\n{'=' * 120}")
    print("FINAL SUMMARY")
    print(f"{'=' * 120}\n")

    print(f"  Baseline P&L:                           ${baseline_pnl:>10,.2f}  ({len(trades)} trades, {baseline_wr:.1f}% WR)")
    print(f"  A) VWAP block only:                     delta ${run_config(True, 0.0)['delta']:>+10,.2f}")
    print(f"  B) Best uptick threshold ({best_thresh:.1f}%):       delta ${best_delta:>+10,.2f}")
    print(f"  C) VWAP + best uptick ({best_combined_thresh:.1f}%):        delta ${best_combined_delta:>+10,.2f}")
    print(f"  D) VWAP + per-category best:            delta ${best_cat_delta:>+10,.2f}  ({best_cat_config})")
    print()

    # Recommend
    options = [
        ("A) VWAP block only", run_config(True, 0.0)['delta']),
        (f"B) Uptick {best_thresh:.1f}% only", best_delta),
        (f"C) VWAP + {best_combined_thresh:.1f}% uptick", best_combined_delta),
        (f"D) Per-category: {best_cat_config}", best_cat_delta),
    ]
    options.sort(key=lambda x: -x[1])
    print(f"  RECOMMENDATION: {options[0][0]} (${options[0][1]:+,.2f})")
    if options[0][1] <= 0:
        print(f"  WARNING: No adjustment improves P&L — dip-confirm may not help on historical data")


if __name__ == "__main__":
    main()
