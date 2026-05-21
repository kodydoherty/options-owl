"""Backtest A/B: V5 FSM with vs without ENRG smart grace backstop.

Compares:
  A) NO ENRG — blind backstop fires at backstop% during grace (old behavior)
  B) WITH ENRG — consult candle voting before firing backstop (new behavior)

We synthesize candle indicators from the harvester tick data (underlying price
+ volume) to simulate what the real ENRG system would see. This lets us test
the ENRG decision logic against actual historical trades.

Usage:
    python scripts/backtest_enrg_grace.py
"""

from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

PORTFOLIO = 8000


# ── Data loading (from backtest_v5_production.py) ────────────────────────────


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


# ── Synthesize candle indicators from tick data ─────────────────────────────


def _compute_rsi(prices: list[float], period: int = 14) -> float | None:
    """Compute RSI from a price series."""
    if len(prices) < period + 1:
        return None
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    if not losses:
        return 95.0
    if not gains:
        return 5.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 95.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _detect_pattern(opens: list[float], closes: list[float],
                    highs: list[float], lows: list[float]) -> str | None:
    """Simple candle pattern detection from OHLC data."""
    if len(closes) < 3:
        return None

    # Use last 3 candles
    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body = abs(c - o)
    total_range = h - l if h > l else 0.001

    # Doji: tiny body relative to range
    if body / total_range < 0.1:
        return "doji"

    # Hammer: small body at top, long lower shadow
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        return "hammer"

    # Shooting star: small body at bottom, long upper shadow
    if upper_shadow > body * 2 and lower_shadow < body * 0.5:
        return "shooting_star"

    # Engulfing patterns (compare to previous candle)
    if len(opens) >= 2:
        prev_o, prev_c = opens[-2], closes[-2]
        # Bullish engulfing: prev bearish, current bullish, current body engulfs prev
        if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
            return "engulfing_bullish"
        # Bearish engulfing
        if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
            return "engulfing_bearish"

    return None


def _compute_obv_trend(prices: list[float], volumes: list[float]) -> str:
    """Simple OBV trend: rising, falling, or flat."""
    if len(prices) < 5 or len(volumes) < 5:
        return "flat"
    obv = 0
    obv_vals = [0]
    for i in range(1, len(prices)):
        vol = volumes[i] if i < len(volumes) else 0
        if prices[i] > prices[i-1]:
            obv += vol
        elif prices[i] < prices[i-1]:
            obv -= vol
        obv_vals.append(obv)

    # Compare last 1/3 OBV to first 1/3
    n = len(obv_vals)
    third = max(1, n // 3)
    early = sum(obv_vals[:third]) / third
    late = sum(obv_vals[-third:]) / third
    if late > early * 1.05:
        return "rising"
    elif late < early * 0.95:
        return "falling"
    return "flat"


def synthesize_candle_data(df: pd.DataFrame, current_idx: int, ticker: str) -> dict:
    """Build candle_data dict from harvester ticks, mimicking candle_cache output.

    Uses underlying_price from ticks to compute RSI, OBV trend, and patterns
    across simulated timeframes.
    """
    if current_idx < 15:
        return {}

    # Get underlying prices up to current point
    underlying = []
    volumes = []
    for i in range(max(0, current_idx - 60), current_idx + 1):
        u = df["underlying_price"].iloc[i]
        v = df["volume"].iloc[i] if "volume" in df.columns else 0
        if u and u > 0:
            underlying.append(float(u))
            volumes.append(float(v) if v and not pd.isna(v) else 0)

    if len(underlying) < 15:
        return {}

    indicators = {}

    # Simulate different timeframes by using different lookback windows
    # Ticks are ~1min apart, so:
    #   5m  = last 5 ticks aggregated
    #   15m = last 15 ticks
    #   30m = last 30 ticks
    #   1h  = last 60 ticks
    tf_windows = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}

    for tf, window in tf_windows.items():
        prices = underlying[-min(window, len(underlying)):]
        vols = volumes[-min(window, len(volumes)):]

        if len(prices) < 5:
            continue

        rsi = _compute_rsi(prices)
        obv_trend = _compute_obv_trend(prices, vols)

        # Build OHLC from the window for pattern detection
        chunk_size = max(1, len(prices) // 3)
        opens, closes, highs, lows = [], [], [], []
        for start in range(0, len(prices), chunk_size):
            chunk = prices[start:start + chunk_size]
            if chunk:
                opens.append(chunk[0])
                closes.append(chunk[-1])
                highs.append(max(chunk))
                lows.append(min(chunk))

        pattern = _detect_pattern(opens, closes, highs, lows)

        indicators[tf] = {
            "rsi": rsi,
            "obv": obv_trend,
            "pattern": pattern,
            "volume_trend": obv_trend,  # reuse
        }

    if not indicators:
        return {}

    return {"ticker": ticker, "indicators": indicators}


# ── Simulation ───────────────────────────────────────────────────────────────


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker, use_enrg: bool):
    """Run V5 FSM against tick data, optionally with ENRG candle data."""
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data", "hold": 0, "exit_prem": 0,
                "peak_gain": 0, "enrg_fired": False, "enrg_detail": ""}

    fsm = ExitFSM(V5Config())
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

    enrg_fired = False
    enrg_detail = ""

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

        # Build candle data if ENRG is enabled
        candle_data = None
        if use_enrg:
            candle_data = synthesize_candle_data(df, idx, ticker)

        action = fsm.evaluate(
            state, premium, bid, ask, now,
            current_underlying=underlying,
            minutes_to_close=minutes_to_close,
            candle_data=candle_data,
        )

        if action.should_exit:
            elapsed = (now - entry_ts).total_seconds() / 60
            peak_gain = (state.peak_premium - entry_premium) / entry_premium * 100
            pnl = (premium - entry_premium) * contracts * 100
            detail = action.detail or ""
            if "ENRG" in detail:
                enrg_fired = True
                enrg_detail = detail
            return {
                "pnl": pnl,
                "reason": action.reason.value,
                "hold": elapsed,
                "exit_prem": premium,
                "peak_gain": peak_gain,
                "enrg_fired": enrg_fired,
                "enrg_detail": enrg_detail,
            }

    # EOD — force close at last tick
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
        "enrg_fired": enrg_fired,
        "enrg_detail": enrg_detail,
    }


# ── Sizing (matches production) ─────────────────────────────────────────────


SCORE_TIERS = [
    (135, 1.00),
    (120, 0.85),
    (100, 0.85),
    (90, 0.50),
    (78, 0.25),
]


def size_trade(score, entry_premium):
    max_risk_pct = 0.75
    max_concurrent = 8
    max_position_pct = 0.08
    deployable = PORTFOLIO * max_risk_pct
    per_slot = deployable / max_concurrent
    position_cap = PORTFOLIO * max_position_pct

    score_mult = 0.25
    for threshold, mult in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break

    if score < 78:
        return 0

    cost_per = entry_premium * 100
    scaled_target = per_slot * score_mult
    raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
    pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
    return max(1, min(raw_contracts, pos_cap_contracts))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    signals = load_signals()
    print(f"Loaded {len(signals)} signals from DB")

    harvester_conn = sqlite3.connect(HARVESTER_DB)
    results_a = []  # No ENRG (old behavior)
    results_b = []  # With ENRG (new behavior)
    no_data = 0

    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        day = sig["created_at"][:10]
        entry_premium = sig["premium"]

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
            adj_entry = entry_premium

        contracts = size_trade(score, adj_entry)
        if contracts == 0:
            continue

        base = {
            "ticker": ticker,
            "day": day,
            "score": score,
            "entry": adj_entry,
            "contracts": contracts,
            "direction": direction,
            "dte": dte,
        }

        # A: No ENRG (blind backstop)
        res_a = simulate_trade(
            df, adj_entry, contracts, direction, dte, expiry_date, ticker,
            use_enrg=False,
        )
        row_a = {**base, **res_a}
        results_a.append(row_a)

        # B: With ENRG (smart backstop)
        res_b = simulate_trade(
            df, adj_entry, contracts, direction, dte, expiry_date, ticker,
            use_enrg=True,
        )
        row_b = {**base, **res_b}
        results_b.append(row_b)

    harvester_conn.close()

    if not results_a:
        print("No results — check that signals and harvester DBs exist")
        return

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    # ── Summary comparison ───────────────────────────────────────────────

    def summarize(df_r, label):
        pnls = df_r["pnl"]
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        total = pnls.sum()
        wr = wins / len(pnls) * 100
        print(f"\n{'=' * 80}")
        print(f"  {label} — {len(df_r)} trades")
        print(f"{'=' * 80}")
        print(f"  Total P&L:   ${total:,.2f}")
        print(f"  Win Rate:    {wr:.1f}% ({wins}W / {losses}L)")
        print(f"  Avg Win:     ${pnls[pnls > 0].mean():,.2f}" if wins > 0 else "  Avg Win:     N/A")
        print(f"  Avg Loss:    ${pnls[pnls <= 0].mean():,.2f}" if losses > 0 else "  Avg Loss:    N/A")
        print(f"  Avg Hold:    {df_r['hold'].mean():.0f} min")
        print(f"  Max Win:     ${pnls.max():,.2f}")
        print(f"  Max Loss:    ${pnls.min():,.2f}")

        # Exit reason breakdown
        print(f"\n  {'Reason':<25} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>6}")
        print(f"  {'-' * 62}")
        for reason, group in df_r.groupby("reason"):
            gpnl = group["pnl"]
            gwins = (gpnl > 0).sum()
            gwr = gwins / len(gpnl) * 100
            print(f"  {reason:<25} {len(gpnl):>6} ${gpnl.sum():>10,.2f} ${gpnl.mean():>8,.2f} {gwr:>5.0f}%")
        return total, wr

    total_a, wr_a = summarize(df_a, "A) NO ENRG (blind backstop)")
    total_b, wr_b = summarize(df_b, "B) WITH ENRG (smart grace backstop)")

    # ── Head-to-head comparison ──────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 80}")
    delta_pnl = total_b - total_a
    delta_wr = wr_b - wr_a
    print(f"  P&L difference:     ${delta_pnl:+,.2f} ({'ENRG wins' if delta_pnl > 0 else 'NO-ENRG wins'})")
    print(f"  Win rate diff:      {delta_wr:+.1f}%")

    # ── Show trades where ENRG changed the outcome ──────────────────────

    print(f"\n{'=' * 80}")
    print(f"  TRADES WHERE ENRG CHANGED THE OUTCOME")
    print(f"{'=' * 80}")
    print(f"  {'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} "
          f"{'NoENRG':>9} {'NoReason':<18} "
          f"{'ENRG':>9} {'Reason':<18} {'Delta':>8} {'ENRG Detail'}")
    print(f"  {'-' * 130}")

    changed_count = 0
    enrg_saved = 0
    enrg_cost = 0

    for i in range(len(df_a)):
        a = df_a.iloc[i]
        b = df_b.iloc[i]

        # Different outcome (different reason or significantly different P&L)
        if a["reason"] != b["reason"] or abs(a["pnl"] - b["pnl"]) > 5:
            changed_count += 1
            delta = b["pnl"] - a["pnl"]
            if delta > 0:
                enrg_saved += delta
            else:
                enrg_cost += abs(delta)

            enrg_note = ""
            if b.get("enrg_fired"):
                enrg_note = str(b.get("enrg_detail", ""))[:60]

            print(f"  {a['day']:<12} {a['ticker']:<6} {a['direction'][:4]:<5} {a['score']:>5} "
                  f"${a['pnl']:>7,.2f} {a['reason']:<18} "
                  f"${b['pnl']:>7,.2f} {b['reason']:<18} ${delta:>+7,.2f} {enrg_note}")

    print(f"\n  Trades with different outcome: {changed_count}")
    print(f"  ENRG improvement (sum of better trades): ${enrg_saved:,.2f}")
    print(f"  ENRG cost (sum of worse trades):         ${enrg_cost:,.2f}")
    print(f"  NET ENRG IMPACT:                         ${enrg_saved - enrg_cost:+,.2f}")

    # ── Per-trade details ────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"  ALL TRADES (B = WITH ENRG)")
    print(f"{'=' * 80}")
    print(f"  {'Day':<12} {'Ticker':<6} {'Dir':<5} {'Score':>5} {'Entry':>7} {'Ct':>3} "
          f"{'Exit':>7} {'P&L':>9} {'Peak%':>6} {'Hold':>5} {'Reason':<20}")
    print(f"  {'-' * 105}")
    for _, r in df_b.iterrows():
        print(f"  {r['day']:<12} {r['ticker']:<6} {r['direction'][:4]:<5} {r['score']:>5} "
              f"${r['entry']:>5.2f} {r['contracts']:>3} ${r['exit_prem']:>5.2f} "
              f"${r['pnl']:>8.2f} {r['peak_gain']:>5.0f}% {r['hold']:>4.0f}m {r['reason']:<20}")


if __name__ == "__main__":
    main()
