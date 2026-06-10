"""Test the specific hypothesis: when premium is dropping right after signal,
does waiting for it to tick back up before buying help?

Strategy:
  1. Signal arrives at tick 0
  2. Check tick 1: is premium LOWER than tick 0?
     - If YES (fading): wait for a tick where premium > previous tick (uptick)
       then buy at that tick's ask price
     - If NO (flat/rising): buy immediately at tick 0 (no delay)
  3. Compare P&L of "dip-then-buy" vs "always buy tick 0"

Also tests:
  - Different dip thresholds (any dip, >2%, >5%)
  - Different max wait times (3, 5, 10, 15 ticks)
  - Only applying the delay to "against" trades (underlying moving wrong way)

Usage:
    python scripts/backtest_dip_then_buy.py
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
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True, V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333, V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True, V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
    V6_2PM_SOFT_TRAIL_BOOST=0.15, ENABLE_V6_PER_TICKER_CONFIG=True,
    ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0,
    V6_PREMIUM_CAP_MID=7.0, V6_PREMIUM_CAP_HIGH=9.0,
    ENABLE_V6_SPREAD_GATE=True, V6_MAX_SPREAD_PCT=15.0,
    ENABLE_V6_EARLY_POP_GATE=True,
    ENABLE_V6_DCA=True, V6_DCA_TICKERS="MSFT,IWM,SPY,QQQ,AMZN,NVDA",
    V6_DCA_MIN_MINUTES=8.0, V6_DCA_MAX_MINUTES=20.0,
    V6_DCA_MIN_DIP_PCT=15.0, V6_DCA_MAX_DIP_PCT=35.0,
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
        FROM trade_signals WHERE score >= 78 ORDER BY created_at
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
    df = pd.DataFrame(rows, columns=["captured_at", "midpoint", "bid", "ask", "underlying_price"])
    df["premium"] = df["midpoint"].where(df["midpoint"] > 0, (df["bid"] + df["ask"]) / 2)
    df["premium"] = df["premium"].where(df["premium"] > 0, np.nan)
    df = df.dropna(subset=["premium"])
    if len(df) < 10:
        return None
    df["ts"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def simulate_fsm(df, entry_idx, entry_premium, contracts, direction, dte, expiry_date, ticker):
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
            pnl = locked_pnl + (premium - entry_premium) * remaining * 100
            return {"pnl": pnl, "reason": action.reason.value}
    last_prem = df["premium"].iloc[-1]
    pnl = locked_pnl + (last_prem - entry_premium) * remaining * 100
    return {"pnl": pnl, "reason": "eod_data_end"}


def size_contracts(score, adj_entry):
    score_mult = 0.25
    for threshold, mult, pos_pct in SCORE_TIERS:
        if score >= threshold:
            score_mult = mult
            break
    effective_pos_pct = 0.15
    position_cap = PORTFOLIO * effective_pos_pct
    deployable = PORTFOLIO * 0.75
    per_slot = deployable / 4
    cost_per = adj_entry * 100
    if cost_per <= 0:
        return 1
    raw = int((per_slot * score_mult) / cost_per)
    cap_c = int(position_cap / cost_per)
    if cap_c == 0:
        return 0
    return max(1, min(raw, cap_c))


def get_entry_price(df, idx):
    """Get entry price at tick idx (ask preferred, fallback to midpoint)."""
    a = df["ask"].iloc[idx]
    p = df["premium"].iloc[idx]
    if a and not pd.isna(a) and a > 0:
        return float(a)
    if not np.isnan(p) and p > 0:
        return float(p)
    return None


def find_uptick(df, start_idx, max_wait, min_dip_pct=0.0):
    """Find the first tick after start_idx where premium ticks UP from its low.

    Returns (entry_idx, entry_price) or (None, None).
    min_dip_pct: require premium to dip at least this much from tick 0 before
                 looking for an uptick (0 = any uptick counts).
    """
    if start_idx >= len(df):
        return None, None

    base_prem = df["premium"].iloc[start_idx]
    if np.isnan(base_prem) or base_prem <= 0:
        return None, None

    low_prem = base_prem
    low_idx = start_idx

    end = min(start_idx + max_wait + 1, len(df))
    for i in range(start_idx + 1, end):
        prem = df["premium"].iloc[i]
        if np.isnan(prem) or prem <= 0:
            continue

        # Track the low
        if prem < low_prem:
            low_prem = prem
            low_idx = i
            continue

        # Check if we've dipped enough
        dip_pct = (base_prem - low_prem) / base_prem * 100
        if dip_pct < min_dip_pct:
            # Haven't dipped enough yet, keep looking for lower
            if prem < low_prem:
                low_prem = prem
                low_idx = i
            continue

        # Premium is above the low AND we've dipped enough — this is an uptick
        if prem > low_prem:
            entry_price = get_entry_price(df, i)
            return i, entry_price

    return None, None


def is_underlying_against(df, tick_idx, direction):
    """Check if underlying at tick_idx has moved against the trade direction
    compared to tick 0."""
    if tick_idx == 0 or tick_idx >= len(df):
        return False
    u0 = df["underlying_price"].iloc[0]
    u1 = df["underlying_price"].iloc[tick_idx]
    if not u0 or not u1 or u0 <= 0 or u1 <= 0:
        return False
    is_call = direction in ("bullish", "call")
    if is_call:
        return float(u1) < float(u0)  # underlying dropped = against call
    else:
        return float(u1) > float(u0)  # underlying rose = against put


def main():
    print("Loading data...")
    signals = load_signals()
    harvester_conn = sqlite3.connect(HARVESTER_DB)
    tick_cache = {}
    for sig in signals:
        df = load_ticks(harvester_conn, sig)
        if df is not None:
            tick_cache[sig["id"]] = (df, sig.get("_dte", 0), sig.get("_expiry_date", ""))
    harvester_conn.close()

    # Build filtered trade list
    trades = []
    for sig in signals:
        if sig["id"] not in tick_cache:
            continue
        df, dte, expiry = tick_cache[sig["id"]]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        ticker = sig["ticker"]

        entry_t0 = get_entry_price(df, 0)
        if entry_t0 is None or entry_t0 <= 0:
            entry_t0 = sig["premium"]

        cap = 6.0
        if score >= 150: cap = 9.0
        elif score >= 120: cap = 7.0
        if entry_t0 > cap:
            continue
        fb = df["bid"].iloc[0]
        fa = df["ask"].iloc[0]
        if fb and fa and fb > 0 and fa > 0:
            if (fa - fb) / fa * 100 > 15:
                continue

        contracts = size_contracts(score, entry_t0)
        if contracts == 0:
            continue

        # Classify: is tick 1 fading?
        prem_t0 = df["premium"].iloc[0]
        prem_t1 = df["premium"].iloc[1] if len(df) > 1 else prem_t0
        t1_change = (prem_t1 - prem_t0) / prem_t0 * 100 if prem_t0 > 0 else 0

        trades.append({
            "id": sig["id"], "ticker": ticker, "day": sig["created_at"][:10],
            "score": score, "direction": direction, "dte": dte,
            "expiry": expiry, "contracts": contracts,
            "entry_t0": entry_t0, "t1_change": t1_change,
            "df": df,
        })

    print(f"  {len(trades)} trades to analyze\n")

    # Classify trades
    fading = [t for t in trades if t["t1_change"] < -1]
    rising = [t for t in trades if t["t1_change"] > 1]
    flat = [t for t in trades if -1 <= t["t1_change"] <= 1]
    print(f"  Tick 1 fading (>1% drop): {len(fading)}")
    print(f"  Tick 1 flat (<1% move):    {len(flat)}")
    print(f"  Tick 1 rising (>1% up):    {len(rising)}")

    # ── Strategy definitions ─────────────────────────────────────────────

    # Each strategy returns (entry_idx, entry_price) for a given trade
    # If it returns (None, None), the trade is skipped

    def strat_immediate(t):
        """Always buy at tick 0."""
        return 0, t["entry_t0"]

    def make_dip_buy(max_wait, min_dip_pct=0.0, only_when_fading=True, only_when_against=False):
        """If fading at tick 1, wait for an uptick. Otherwise buy immediately."""
        def strat(t):
            df = t["df"]
            prem_t0 = df["premium"].iloc[0]
            prem_t1 = df["premium"].iloc[1] if len(df) > 1 else prem_t0
            is_fading = prem_t1 < prem_t0 * (1 - 0.01)  # >1% drop

            if only_when_against:
                against = is_underlying_against(df, 1, t["direction"])
                if not against:
                    return 0, t["entry_t0"]  # not against, buy immediately

            if only_when_fading and not is_fading:
                return 0, t["entry_t0"]  # not fading, buy immediately

            # Wait for uptick
            idx, price = find_uptick(df, 0, max_wait, min_dip_pct)
            if idx is not None and price is not None:
                return idx, price
            # No uptick found within max_wait — skip trade
            return None, None
        return strat

    def make_dip_buy_or_enter(max_wait, min_dip_pct=0.0):
        """If fading, wait for uptick. If no uptick in max_wait, buy anyway at max_wait."""
        def strat(t):
            df = t["df"]
            prem_t0 = df["premium"].iloc[0]
            prem_t1 = df["premium"].iloc[1] if len(df) > 1 else prem_t0
            is_fading = prem_t1 < prem_t0 * (1 - 0.01)

            if not is_fading:
                return 0, t["entry_t0"]

            idx, price = find_uptick(df, 0, max_wait, min_dip_pct)
            if idx is not None and price is not None:
                return idx, price
            # Fallback: buy at max_wait tick anyway (if available)
            fallback = min(max_wait, len(df) - 6)
            if fallback > 0:
                price = get_entry_price(df, fallback)
                if price:
                    return fallback, price
            return 0, t["entry_t0"]
        return strat

    strategies = [
        ("BASELINE: buy immediately", strat_immediate),

        # Wait for uptick when fading, skip if no uptick
        ("Fading→wait uptick (max 3 ticks), skip if none", make_dip_buy(3)),
        ("Fading→wait uptick (max 5 ticks), skip if none", make_dip_buy(5)),
        ("Fading→wait uptick (max 10 ticks), skip if none", make_dip_buy(10)),
        ("Fading→wait uptick (max 15 ticks), skip if none", make_dip_buy(15)),

        # Wait for uptick when fading, but buy anyway if no uptick
        ("Fading→wait uptick (max 3), else buy at 3", make_dip_buy_or_enter(3)),
        ("Fading→wait uptick (max 5), else buy at 5", make_dip_buy_or_enter(5)),
        ("Fading→wait uptick (max 10), else buy at 10", make_dip_buy_or_enter(10)),

        # Require minimum dip before looking for uptick
        ("Fading→wait 3%+ dip then uptick (max 5)", make_dip_buy(5, min_dip_pct=3)),
        ("Fading→wait 3%+ dip then uptick (max 10)", make_dip_buy(10, min_dip_pct=3)),
        ("Fading→wait 5%+ dip then uptick (max 10)", make_dip_buy(10, min_dip_pct=5)),
        ("Fading→wait 5%+ dip then uptick (max 15)", make_dip_buy(15, min_dip_pct=5)),

        # Only apply when underlying is ALSO against us
        ("Against+fading→uptick (max 5), else immediate", make_dip_buy(5, only_when_against=True)),
        ("Against+fading→uptick (max 10), else immediate", make_dip_buy(10, only_when_against=True)),

        # Require dip + underlying against
        ("Against→wait 3% dip+uptick (max 10)", make_dip_buy(10, min_dip_pct=3, only_when_against=True, only_when_fading=False)),
    ]

    # ── Run all strategies ───────────────────────────────────────────────

    print(f"\n{'=' * 120}")
    print("STRATEGY RESULTS")
    print(f"{'=' * 120}")
    print(f"  {'Strategy':<55} {'Trades':>6} {'P&L':>11} {'vs Base':>9} "
          f"{'Win%':>5} {'Skip':>5} {'FadePnL':>10} {'FadeWR':>6}")
    print(f"  {'-' * 115}")

    baseline_pnl = None

    for name, strat_fn in strategies:
        total_pnl = 0
        total_trades = 0
        wins = 0
        skipped = 0
        fade_pnl = 0
        fade_trades = 0
        fade_wins = 0

        for t in trades:
            entry_idx, entry_price = strat_fn(t)
            if entry_idx is None or entry_price is None:
                skipped += 1
                continue

            result = simulate_fsm(
                t["df"], entry_idx, entry_price, t["contracts"],
                t["direction"], t["dte"], t["expiry"], t["ticker"]
            )
            if result is None:
                skipped += 1
                continue

            total_pnl += result["pnl"]
            total_trades += 1
            if result["pnl"] > 0:
                wins += 1

            # Track fading subset separately
            if t["t1_change"] < -1:
                fade_pnl += result["pnl"]
                fade_trades += 1
                if result["pnl"] > 0:
                    fade_wins += 1

        wr = wins / total_trades * 100 if total_trades > 0 else 0
        fade_wr = fade_wins / fade_trades * 100 if fade_trades > 0 else 0

        if baseline_pnl is None:
            baseline_pnl = total_pnl

        diff = total_pnl - baseline_pnl
        print(f"  {name:<55} {total_trades:>6} ${total_pnl:>9,.2f} "
              f"${diff:>+7,.0f} {wr:>4.0f}% {skipped:>5} "
              f"${fade_pnl:>8,.2f} {fade_wr:>5.0f}%")

    # ── Deep dive: what happens to FADING trades specifically ────────────

    print(f"\n{'=' * 120}")
    print(f"DEEP DIVE: The {len(fading)} trades where premium was fading >1% at tick 1")
    print(f"{'=' * 120}")

    # Immediate entry on fading trades
    fade_immediate_pnl = 0
    fade_immediate_wins = 0
    fade_results = []
    for t in fading:
        result = simulate_fsm(
            t["df"], 0, t["entry_t0"], t["contracts"],
            t["direction"], t["dte"], t["expiry"], t["ticker"]
        )
        if result:
            fade_immediate_pnl += result["pnl"]
            if result["pnl"] > 0:
                fade_immediate_wins += 1
            fade_results.append({**t, **result, "entry_type": "immediate",
                                 "entry_idx": 0, "actual_entry": t["entry_t0"]})

    # Uptick entry on fading trades (wait max 5)
    fade_uptick_pnl = 0
    fade_uptick_wins = 0
    fade_uptick_trades = 0
    fade_skipped = 0
    fade_uptick_results = []
    for t in fading:
        idx, price = find_uptick(t["df"], 0, 5)
        if idx is None or price is None:
            fade_skipped += 1
            continue
        result = simulate_fsm(
            t["df"], idx, price, t["contracts"],
            t["direction"], t["dte"], t["expiry"], t["ticker"]
        )
        if result:
            fade_uptick_pnl += result["pnl"]
            fade_uptick_trades += 1
            if result["pnl"] > 0:
                fade_uptick_wins += 1
            fade_uptick_results.append({**t, **result, "entry_type": "uptick",
                                        "entry_idx": idx, "actual_entry": price})

    fade_wr_imm = fade_immediate_wins / len(fading) * 100 if fading else 0
    fade_wr_upt = fade_uptick_wins / fade_uptick_trades * 100 if fade_uptick_trades else 0

    print(f"\n  Immediate entry:  {len(fading)} trades, ${fade_immediate_pnl:>10,.2f}, "
          f"WR: {fade_wr_imm:.0f}%")
    print(f"  Wait-for-uptick:  {fade_uptick_trades} trades, ${fade_uptick_pnl:>10,.2f}, "
          f"WR: {fade_wr_upt:.0f}% ({fade_skipped} skipped — no uptick in 5 ticks)")

    # Compare entry prices
    print(f"\n  Entry price comparison (fading trades that got uptick entry):")
    price_diffs = []
    pnl_diffs = []
    for t in fading:
        idx, price = find_uptick(t["df"], 0, 5)
        if idx is None or price is None:
            continue
        t0_price = t["entry_t0"]
        diff_pct = (price - t0_price) / t0_price * 100
        price_diffs.append(diff_pct)

        # Compare P&L
        r_imm = simulate_fsm(t["df"], 0, t0_price, t["contracts"],
                              t["direction"], t["dte"], t["expiry"], t["ticker"])
        r_upt = simulate_fsm(t["df"], idx, price, t["contracts"],
                              t["direction"], t["dte"], t["expiry"], t["ticker"])
        if r_imm and r_upt:
            pnl_diffs.append({
                "ticker": t["ticker"], "day": t["day"], "score": t["score"],
                "t0_price": t0_price, "uptick_price": price,
                "price_diff_pct": diff_pct, "uptick_idx": idx,
                "pnl_immediate": r_imm["pnl"], "pnl_uptick": r_upt["pnl"],
                "pnl_diff": r_upt["pnl"] - r_imm["pnl"],
                "reason_imm": r_imm["reason"], "reason_upt": r_upt["reason"],
            })

    if price_diffs:
        arr = np.array(price_diffs)
        print(f"    Median entry price change: {np.median(arr):+.2f}%")
        print(f"    Mean entry price change:   {np.mean(arr):+.2f}%")
        print(f"    Got cheaper price: {(arr < 0).sum()}/{len(arr)} "
              f"({(arr < 0).mean()*100:.0f}%)")
        print(f"    Got more expensive: {(arr > 0).sum()}/{len(arr)} "
              f"({(arr > 0).mean()*100:.0f}%)")

    if pnl_diffs:
        df_comp = pd.DataFrame(pnl_diffs)
        uptick_better = (df_comp["pnl_diff"] > 0).sum()
        imm_better = (df_comp["pnl_diff"] < 0).sum()
        total_diff = df_comp["pnl_diff"].sum()

        print(f"\n  P&L comparison (trade-by-trade):")
        print(f"    Uptick entry better: {uptick_better}/{len(df_comp)} trades")
        print(f"    Immediate better:    {imm_better}/{len(df_comp)} trades")
        print(f"    Total P&L difference: ${total_diff:+,.2f}")

        # Show individual trade comparisons
        print(f"\n  {'Day':<12} {'Ticker':<6} {'Sc':>3} {'T0':>6} {'Uptick':>6} {'At':>3} "
              f"{'PnL(imm)':>10} {'PnL(upt)':>10} {'Diff':>10} {'Better'}")
        print(f"  {'-' * 100}")
        for _, r in df_comp.sort_values("pnl_diff").iterrows():
            better = "UPTICK" if r["pnl_diff"] > 0 else "IMMEDIATE"
            if abs(r["pnl_diff"]) < 10:
                better = "~same"
            print(f"  {r['day']:<12} {r['ticker']:<6} {r['score']:>3} "
                  f"${r['t0_price']:>4.2f} ${r['uptick_price']:>4.2f} t{r['uptick_idx']:>1.0f} "
                  f"${r['pnl_immediate']:>8,.2f} ${r['pnl_uptick']:>8,.2f} "
                  f"${r['pnl_diff']:>+8,.2f} {better}")

    # ── Summary: what happens to the SKIPPED trades ──────────────────────

    print(f"\n{'=' * 120}")
    print(f"WHAT ABOUT THE {fade_skipped} SKIPPED TRADES (fading, no uptick in 5 ticks)?")
    print(f"{'=' * 120}")

    skipped_results = []
    for t in fading:
        idx, price = find_uptick(t["df"], 0, 5)
        if idx is not None:
            continue  # had an uptick — not skipped
        result = simulate_fsm(
            t["df"], 0, t["entry_t0"], t["contracts"],
            t["direction"], t["dte"], t["expiry"], t["ticker"]
        )
        if result:
            skipped_results.append({**t, **result})

    if skipped_results:
        df_skip = pd.DataFrame(skipped_results)
        skip_pnl = df_skip["pnl"].sum()
        skip_wins = (df_skip["pnl"] > 0).sum()
        skip_wr = skip_wins / len(df_skip) * 100

        print(f"  These {len(df_skip)} trades (entered immediately) had:")
        print(f"    Total P&L: ${skip_pnl:,.2f}")
        print(f"    Win Rate:  {skip_wr:.0f}% ({skip_wins}W / {len(df_skip) - skip_wins}L)")
        print(f"    Avg P&L:   ${df_skip['pnl'].mean():,.2f}")

        if skip_pnl < 0:
            print(f"\n  ** Skipping these would have SAVED ${-skip_pnl:,.2f} **")
        else:
            print(f"\n  ** Skipping these would have COST ${skip_pnl:,.2f} **")

        print(f"\n  {'Day':<12} {'Ticker':<6} {'Sc':>3} {'Entry':>6} {'T1Chg':>7} "
              f"{'P&L':>10} {'Reason':<20}")
        print(f"  {'-' * 75}")
        for _, r in df_skip.sort_values("pnl").iterrows():
            print(f"  {r['day']:<12} {r['ticker']:<6} {r['score']:>3} "
                  f"${r['entry_t0']:>4.2f} {r['t1_change']:>+6.1f}% "
                  f"${r['pnl']:>8,.2f} {r['reason']:<20}")


if __name__ == "__main__":
    main()
