"""Test if underlying price patterns improve early-pop detection.

Full ENRG (RSI/OBV/patterns) can't work — we only have ~12 minutes of
1-tick-per-minute underlying data, not enough for 14-period RSI.

Instead, test simpler underlying-derived signals:
  A) Underlying trend: is price making lower lows/higher highs against trade?
  B) Underlying acceleration: is the move getting worse?
  C) Premium-underlying divergence: premium fading while underlying is flat/favorable?
  D) Combined with the early-pop backstop gate

Usage:
    python scripts/backtest_early_pop_candle.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from types import SimpleNamespace

from options_owl.risk.exit_v5.config import V5Config, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

_V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True, V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333, V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True, V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15, ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_SIDEWAYS_SCALP=True,
)

SIGNALS_DB = str(PROJECT_DIR / "journal" / "owlet-kody" / "raw_messages.db")
HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")
PORTFOLIO = 8000
INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
SCORE_TIERS = [(135, 1.00), (120, 0.85), (100, 0.85), (90, 0.50), (78, 0.25)]


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
    first_half = underlying_prices[:len(underlying_prices) // 2]
    second_half = underlying_prices[len(underlying_prices) // 2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    pct_move = (avg_second - avg_first) / avg_first * 100
    prem_start = df["premium"].iloc[0]
    prem_5 = df["premium"].iloc[min(4, len(df) - 1)]
    prem_fade = (prem_5 - prem_start) / prem_start * 100 if prem_start > 0 else 0
    neg_signals = 0
    if is_call and pct_move < -0.05:
        neg_signals += 1
    elif not is_call and pct_move > 0.05:
        neg_signals += 1
    if prem_fade < -5:
        neg_signals += 1
    against = 0
    for i in range(max(0, window - 3), window):
        if i == 0:
            continue
        prev_u = df["underlying_price"].iloc[i - 1]
        cur_u = df["underlying_price"].iloc[i]
        if prev_u and cur_u:
            if is_call and cur_u < prev_u:
                against += 1
            elif not is_call and cur_u > prev_u:
                against += 1
    if against >= 3:
        neg_signals += 1
    return neg_signals >= 2


def simulate_trade(df, entry_premium, contracts, direction, dte, expiry_date,
                   ticker="SIM", cfg_override=None):
    if entry_premium <= 0:
        return {"pnl": 0, "reason": "no_data"}
    cfg = cfg_override or get_ticker_config(ticker, use_per_ticker=True)
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
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value}
    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end"}


# ── Underlying-based signals ─────────────────────────────────────────────────


def analyze_underlying(df, direction, check_at_min=12):
    """Extract underlying price signals from first N minutes.

    Returns dict of signals computed from underlying_price ticks.
    """
    if len(df) < 5:
        return None

    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)

    is_call = direction in ("bullish", "call")

    # Collect underlying prices within window
    u_prices = []
    u_times = []
    premiums = []
    prem_times = []

    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed = (ts - entry_ts).total_seconds() / 60
        if elapsed > check_at_min:
            break

        u = df["underlying_price"].iloc[i]
        if u and u > 0:
            u_prices.append(float(u))
            u_times.append(elapsed)

        p = df["premium"].iloc[i]
        if not np.isnan(p) and p > 0:
            premiums.append(float(p))
            prem_times.append(elapsed)

    if len(u_prices) < 4 or len(premiums) < 4:
        return None

    # Signal A: Underlying trend (direction-adjusted)
    # Split into thirds, compare first vs last
    n = len(u_prices)
    third = max(1, n // 3)
    u_first = np.mean(u_prices[:third])
    u_last = np.mean(u_prices[-third:])
    u_trend = (u_last - u_first) / u_first * 100
    # Direction-adjusted: negative = against the trade
    u_trend_adj = u_trend if is_call else -u_trend

    # Signal B: Underlying making lower lows (for calls) or higher highs (for puts)
    # Check last 5 ticks — are they consistently moving against?
    recent = u_prices[-min(5, n):]
    lower_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
    higher_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    if is_call:
        against_streak = lower_count >= 3  # 3+ of last 4 moves are down
    else:
        against_streak = higher_count >= 3

    # Signal C: Underlying acceleration — is the against-move getting worse?
    if len(u_prices) >= 6:
        half = len(u_prices) // 2
        move_first_half = (u_prices[half] - u_prices[0]) / u_prices[0] * 100 if u_prices[0] > 0 else 0
        move_second_half = (u_prices[-1] - u_prices[half]) / u_prices[half] * 100 if u_prices[half] > 0 else 0
        if is_call:
            accelerating_against = move_second_half < move_first_half and move_second_half < -0.02
        else:
            accelerating_against = move_second_half > -move_first_half and move_second_half > 0.02
    else:
        accelerating_against = False

    # Signal D: Premium-underlying divergence
    # Premium is dropping but underlying is flat or favorable
    prem_n = len(premiums)
    prem_third = max(1, prem_n // 3)
    prem_first = np.mean(premiums[:prem_third])
    prem_last = np.mean(premiums[-prem_third:])
    prem_trend = (prem_last - prem_first) / prem_first * 100 if prem_first > 0 else 0

    # Divergence: premium falling while underlying isn't against
    divergence = prem_trend < -5 and u_trend_adj > -0.05

    # Signal E: Mini-candle pattern (crude 5-min OHLC from underlying)
    # Build 2-3 crude candles
    candle_bearish = False
    if len(u_prices) >= 10:
        mid = len(u_prices) // 2
        candle1_o = u_prices[0]
        candle1_c = u_prices[mid-1]
        candle1_h = max(u_prices[:mid])
        candle1_l = min(u_prices[:mid])
        candle2_o = u_prices[mid]
        candle2_c = u_prices[-1]
        candle2_h = max(u_prices[mid:])
        candle2_l = min(u_prices[mid:])

        if is_call:
            # Bearish for calls: second candle closes below first candle's low
            # or second candle is red and engulfs first
            candle_bearish = (candle2_c < candle1_l or
                              (candle2_c < candle2_o and
                               candle2_o > candle1_h and candle2_c < candle1_l))
        else:
            candle_bearish = (candle2_c > candle1_h or
                              (candle2_c > candle2_o and
                               candle2_o < candle1_l and candle2_c > candle1_h))

    return {
        "u_trend_adj": round(u_trend_adj, 4),
        "against_streak": against_streak,
        "accelerating_against": accelerating_against,
        "prem_trend": round(prem_trend, 2),
        "divergence": divergence,
        "candle_bearish": candle_bearish,
        "n_u_ticks": n,
        "n_prem_ticks": prem_n,
    }


def detect_early_pop(df, entry_premium, direction, peak_window_min=12,
                     fade_threshold_pct=10, check_at_min=12, min_peak_gain=3.0):
    if len(df) < 10 or entry_premium <= 0:
        return False
    entry_ts = df["ts"].iloc[0]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_ts = entry_ts.to_pydatetime()
    if entry_ts.tzinfo is not None:
        entry_ts = entry_ts.replace(tzinfo=None)
    peak_prem = entry_premium
    peak_elapsed = 0.0
    current_prem = entry_premium
    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed = (ts - entry_ts).total_seconds() / 60
        if elapsed > check_at_min:
            break
        prem = df["premium"].iloc[i]
        if np.isnan(prem) or prem <= 0:
            continue
        current_prem = prem
        if prem > peak_prem:
            peak_prem = prem
            peak_elapsed = elapsed
    if peak_elapsed > peak_window_min:
        return False
    peak_gain = (peak_prem - entry_premium) / entry_premium * 100
    if peak_gain < min_peak_gain:
        return False
    if peak_prem <= 0:
        return False
    fade = (peak_prem - current_prem) / peak_prem * 100
    return fade >= fade_threshold_pct


def main():
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)

    prepared = []
    for sig in signals:
        ticker = sig["ticker"]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        if score < 78:
            continue
        premium = sig["premium"]
        if ticker not in INDEX_TICKERS:
            cap = 9.0 if score >= 150 else (7.0 if score >= 120 else 6.0)
            if premium > cap:
                continue
        df = load_ticks(harvester_conn, sig)
        if df is None:
            continue
        dte = sig.get("_dte", 0)
        expiry_date = sig.get("_expiry_date", "")
        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        adj_entry = first_ask if first_ask and first_ask > 0 else first_mid
        if adj_entry <= 0:
            adj_entry = premium
        deployable = PORTFOLIO * 0.75
        per_slot = deployable / 4
        position_cap = PORTFOLIO * 0.15
        score_mult = 0.25
        for threshold, mult in SCORE_TIERS:
            if score >= threshold:
                score_mult = mult
                break
        cost_per = adj_entry * 100
        scaled_target = per_slot * score_mult
        raw_contracts = int(scaled_target / cost_per) if cost_per > 0 else 1
        pos_cap_contracts = int(position_cap / cost_per) if cost_per > 0 else 1
        contracts = max(1, min(raw_contracts, pos_cap_contracts))
        if momentum_blocked(df, direction):
            continue
        prepared.append({
            "ticker": ticker, "direction": direction, "score": score,
            "premium": adj_entry, "contracts": contracts, "df": df,
            "dte": dte, "expiry_date": expiry_date, "day": sig["created_at"][:10],
        })

    harvester_conn.close()
    print(f"Prepared {len(prepared)} trades")

    # Baseline
    baseline_pnls = []
    for sig in prepared:
        r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                           sig["direction"], sig["dte"], sig["expiry_date"],
                           ticker=sig["ticker"])
        baseline_pnls.append(r["pnl"])
    baseline_total = sum(baseline_pnls)
    print(f"Baseline: ${baseline_total:,.2f}\n")

    # ── Analyze underlying signals for early-pop trades ──────────────────────

    print("=" * 130)
    print("UNDERLYING PRICE SIGNALS FOR EARLY-POP TRADES")
    print("Can underlying price patterns at minute 12 predict crashers vs recoverers?")
    print("=" * 130)

    ep_trades = []
    for i, sig in enumerate(prepared):
        is_ep = detect_early_pop(sig["df"], sig["premium"], sig["direction"])
        if not is_ep:
            continue
        u_signals = analyze_underlying(sig["df"], sig["direction"])
        if u_signals is None:
            continue
        ep_trades.append({
            "idx": i,
            "ticker": sig["ticker"],
            "day": sig["day"],
            "pnl": baseline_pnls[i],
            "is_loser": baseline_pnls[i] <= 0,
            "is_big_loser": baseline_pnls[i] < -200,
            **u_signals,
        })

    print(f"\nEarly-pop trades with underlying data: {len(ep_trades)}")
    losers = [t for t in ep_trades if t["is_loser"]]
    winners = [t for t in ep_trades if not t["is_loser"]]
    print(f"Winners: {len(winners)} | Losers: {len(losers)}")

    print(f"\n{'Day':<12} {'Ticker':<7} {'P&L':>10} {'U_Trend':>8} {'AgStk':>5} "
          f"{'Accel':>5} {'PremTr':>7} {'Div':>4} {'CndlBr':>6} {'Outcome':<10}")
    print("-" * 90)

    for t in sorted(ep_trades, key=lambda x: x["pnl"]):
        outcome = "BIG LOSS" if t["is_big_loser"] else ("LOSS" if t["is_loser"] else "WIN")
        print(f"{t['day']:<12} {t['ticker']:<7} ${t['pnl']:>8,.2f} "
              f"{t['u_trend_adj']:>7.3f}% {'Y' if t['against_streak'] else 'N':>5} "
              f"{'Y' if t['accelerating_against'] else 'N':>5} "
              f"{t['prem_trend']:>6.1f}% {'Y' if t['divergence'] else 'N':>4} "
              f"{'Y' if t['candle_bearish'] else 'N':>6} {outcome}")

    # ── Compare signal rates between losers and winners ──────────────────────

    print(f"\n\nSIGNAL RATES (losers vs winners)")
    print("-" * 60)

    for signal_name in ["against_streak", "accelerating_against", "divergence", "candle_bearish"]:
        loser_rate = sum(1 for t in losers if t[signal_name]) / max(1, len(losers)) * 100
        winner_rate = sum(1 for t in winners if t[signal_name]) / max(1, len(winners)) * 100
        diff = loser_rate - winner_rate
        useful = "USEFUL" if diff > 15 else ""
        print(f"  {signal_name:<25} Losers: {loser_rate:>5.1f}%  Winners: {winner_rate:>5.1f}%  "
              f"Diff: {diff:>+5.1f}%  {useful}")

    for signal_name in ["u_trend_adj", "prem_trend"]:
        loser_avg = np.mean([t[signal_name] for t in losers]) if losers else 0
        winner_avg = np.mean([t[signal_name] for t in winners]) if winners else 0
        diff = loser_avg - winner_avg
        print(f"  {signal_name:<25} Losers: {loser_avg:>+6.3f}  Winners: {winner_avg:>+6.3f}  "
              f"Diff: {diff:>+6.3f}")

    # ── Test combined detection rules ────────────────────────────────────────

    print(f"\n\n{'=' * 130}")
    print("COMBINED RULES: early_pop + underlying signals → tighter backstop")
    print("Testing whether adding underlying signals improves the early-pop gate")
    print("=" * 130)

    # For each rule combo, check: how many early-pop trades match?
    # Apply tighter backstop only to matches. Compare vs baseline and vs ep-only.
    rules = [
        ("ep_only (no underlying)", lambda t: True),
        ("ep + u_trend < -0.05", lambda t: t["u_trend_adj"] < -0.05),
        ("ep + u_trend < -0.03", lambda t: t["u_trend_adj"] < -0.03),
        ("ep + u_trend < -0.01", lambda t: t["u_trend_adj"] < -0.01),
        ("ep + against_streak", lambda t: t["against_streak"]),
        ("ep + accel_against", lambda t: t["accelerating_against"]),
        ("ep + divergence", lambda t: t["divergence"]),
        ("ep + candle_bearish", lambda t: t["candle_bearish"]),
        ("ep + prem_trend < -8", lambda t: t["prem_trend"] < -8),
        ("ep + prem_trend < -5", lambda t: t["prem_trend"] < -5),
        ("ep + prem_trend < -3", lambda t: t["prem_trend"] < -3),
        ("ep + streak OR accel", lambda t: t["against_streak"] or t["accelerating_against"]),
        ("ep + streak OR diverge", lambda t: t["against_streak"] or t["divergence"]),
        ("ep + (streak OR accel) AND prem<-5",
         lambda t: (t["against_streak"] or t["accelerating_against"]) and t["prem_trend"] < -5),
        ("ep + candle_bear AND prem<-5",
         lambda t: t["candle_bearish"] and t["prem_trend"] < -5),
        ("ep + u_trend<-0.03 AND prem<-5",
         lambda t: t["u_trend_adj"] < -0.03 and t["prem_trend"] < -5),
        ("ep + u_trend<-0.03 OR candle_bear",
         lambda t: t["u_trend_adj"] < -0.03 or t["candle_bearish"]),
        ("ep + ANY 2 of (streak,accel,div,candle)",
         lambda t: sum([t["against_streak"], t["accelerating_against"],
                        t["divergence"], t["candle_bearish"]]) >= 2),
        ("ep + ANY 1 of (streak,accel,div,candle)",
         lambda t: sum([t["against_streak"], t["accelerating_against"],
                        t["divergence"], t["candle_bearish"]]) >= 1),
    ]

    # Build lookup: ep_trade idx → trade info with underlying signals
    ep_lookup = {t["idx"]: t for t in ep_trades}

    print(f"\n{'Rule':<48} {'Match':>5} {'Saved':>10} {'Cost':>10} {'Net':>10} "
          f"{'CatchBigL':>9} {'HitWin':>6}")
    print("-" * 105)

    for rule_name, rule_fn in rules:
        total_pnl = 0
        saved = 0
        cost = 0
        big_losers_caught = 0
        winners_hit = 0

        for i, sig in enumerate(prepared):
            if i in ep_lookup:
                t = ep_lookup[i]
                if rule_fn(t):
                    # Apply tighter backstop
                    cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                                  backstop_0dte_pct=35.0, backstop_multiday_pct=50.0)
                    r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                                       sig["direction"], sig["dte"], sig["expiry_date"],
                                       ticker=sig["ticker"], cfg_override=cfg)
                    diff = r["pnl"] - baseline_pnls[i]
                    total_pnl += r["pnl"]
                    if diff > 0:
                        saved += diff
                    elif diff < 0:
                        cost += -diff
                    if baseline_pnls[i] < -200:
                        big_losers_caught += 1
                    if baseline_pnls[i] > 0 and diff < -10:
                        winners_hit += 1
                else:
                    total_pnl += baseline_pnls[i]
            else:
                total_pnl += baseline_pnls[i]

        delta = total_pnl - baseline_total
        marker = " ***" if delta > 1500 else (" **" if delta > 1000 else (" *" if delta > 500 else ""))
        matched = sum(1 for t in ep_trades if rule_fn(t))
        print(f"{rule_name:<48} {matched:>5} ${saved:>8,.2f} ${cost:>8,.2f} "
              f"${delta:>+8,.2f} {big_losers_caught:>9} {winners_hit:>6}{marker}")

    # ── Also test: tighter backstop (30%) for underlying-confirmed early-pop ──

    print(f"\n\n{'=' * 130}")
    print("BACKSTOP SWEEP for best underlying rule")
    print("=" * 130)

    # Use the broadest useful rule: ep + ANY 1 underlying signal
    best_rule = lambda t: sum([t["against_streak"], t["accelerating_against"],
                               t["divergence"], t["candle_bearish"]]) >= 1

    for bs_pct in range(25, 55, 5):
        total_pnl = 0
        saved = 0
        cost = 0

        for i, sig in enumerate(prepared):
            if i in ep_lookup and best_rule(ep_lookup[i]):
                cfg = replace(get_ticker_config(sig["ticker"], use_per_ticker=True),
                              backstop_0dte_pct=float(bs_pct),
                              backstop_multiday_pct=float(bs_pct + 15))
                r = simulate_trade(sig["df"], sig["premium"], sig["contracts"],
                                   sig["direction"], sig["dte"], sig["expiry_date"],
                                   ticker=sig["ticker"], cfg_override=cfg)
                diff = r["pnl"] - baseline_pnls[i]
                total_pnl += r["pnl"]
                if diff > 0:
                    saved += diff
                elif diff < 0:
                    cost += -diff
            else:
                total_pnl += baseline_pnls[i]

        delta = total_pnl - baseline_total
        matched = sum(1 for t in ep_trades if best_rule(t))
        marker = " ***" if delta > 1500 else (" **" if delta > 1000 else "")
        print(f"  BS={bs_pct}% | {matched} trades | "
              f"Saved: ${saved:>8,.2f} | Cost: ${cost:>8,.2f} | "
              f"Net: ${delta:>+8,.2f}{marker}")


if __name__ == "__main__":
    main()
