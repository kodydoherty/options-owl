"""Fast pre-entry momentum check — what can we learn in 1-3 ticks (~1-3 min)
before entering, without losing good trades?

Tests:
  1. Check premium direction after just 1, 2, or 3 ticks
  2. Check underlying direction after 1-3 ticks
  3. What if we enter at tick 1/2/3 instead of tick 0? (better or worse price?)
  4. Combined: wait 2 ticks, skip if both premium AND underlying are against us

Usage:
    python scripts/backtest_fast_momentum.py
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
    raw = int((per_slot * score_mult) / cost_per)
    cap_c = int(position_cap / cost_per)
    if cap_c == 0:
        return 0
    return max(1, min(raw, cap_c))


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
    print(f"  {len(tick_cache)} signals with tick data\n")

    # Build trade list with per-tick data
    trades = []
    for sig in signals:
        if sig["id"] not in tick_cache:
            continue
        df, dte, expiry = tick_cache[sig["id"]]
        direction = (sig["direction"] or "bullish").lower()
        score = sig["score"] or 80
        ticker = sig["ticker"]
        is_call = direction in ("bullish", "call")

        first_ask = df["ask"].iloc[0]
        first_mid = df["premium"].iloc[0]
        entry_t0 = first_ask if first_ask and first_ask > 0 else first_mid
        if entry_t0 <= 0:
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

        # Collect tick-by-tick data for first 5 ticks
        tick_data = []
        for i in range(min(6, len(df))):
            p = df["premium"].iloc[i]
            a = df["ask"].iloc[i]
            b = df["bid"].iloc[i]
            u = df["underlying_price"].iloc[i]
            entry_price = a if a and a > 0 else (p if p and not np.isnan(p) else None)
            tick_data.append({
                "premium": p if not np.isnan(p) else None,
                "ask": a if a and not pd.isna(a) else None,
                "underlying": float(u) if u and u > 0 else None,
                "entry_price": entry_price,
            })

        # Premium change at each tick vs tick 0
        prem_t0 = tick_data[0]["premium"]
        und_t0 = tick_data[0]["underlying"]

        prem_changes = []
        und_changes = []
        for t in tick_data:
            if t["premium"] and prem_t0 and prem_t0 > 0:
                prem_changes.append((t["premium"] - prem_t0) / prem_t0 * 100)
            else:
                prem_changes.append(None)
            if t["underlying"] and und_t0 and und_t0 > 0:
                raw = (t["underlying"] - und_t0) / und_t0 * 100
                # For puts, underlying rising = against us
                und_changes.append(-raw if not is_call else raw)
            else:
                und_changes.append(None)

        trades.append({
            "id": sig["id"], "ticker": ticker, "day": sig["created_at"][:10],
            "score": score, "direction": direction, "dte": dte,
            "expiry": expiry, "contracts": contracts,
            "entry_t0": entry_t0,
            "prem_changes": prem_changes,
            "und_changes": und_changes,
            "tick_data": tick_data,
            "df": df,
        })

    print(f"{len(trades)} trades to analyze\n")

    # ── Strategy comparison: enter at tick N with different filters ───────

    strategies = []

    # For each wait period (0, 1, 2, 3 ticks), test:
    #   - Enter unconditionally at that tick
    #   - Skip if premium dropped > X% by that tick
    #   - Skip if underlying moved against > Y% by that tick
    #   - Skip if BOTH premium AND underlying against

    for wait in [0, 1, 2, 3]:
        for prem_threshold in [None, -3, -5, -8]:
            for und_threshold in [None, -0.03, -0.05, -0.08]:
                name_parts = [f"wait={wait}"]
                if prem_threshold is not None:
                    name_parts.append(f"prem>{prem_threshold}%")
                if und_threshold is not None:
                    name_parts.append(f"und>{und_threshold}%")
                name = " ".join(name_parts)

                total_pnl = 0
                total_trades = 0
                total_wins = 0
                skipped = 0
                missed_good = 0  # good trades we skipped

                for t in trades:
                    if wait >= len(t["prem_changes"]) or wait >= len(t["tick_data"]):
                        continue

                    pc = t["prem_changes"][wait]
                    uc = t["und_changes"][wait]
                    entry_price = t["tick_data"][wait]["entry_price"]
                    if entry_price is None or entry_price <= 0:
                        continue

                    # Apply filters
                    blocked = False
                    if prem_threshold is not None and pc is not None and pc < prem_threshold:
                        blocked = True
                    if und_threshold is not None and uc is not None and uc < und_threshold:
                        blocked = True

                    if blocked:
                        # Check what would have happened
                        hypothetical = simulate_fsm(
                            t["df"], wait, entry_price, t["contracts"],
                            t["direction"], t["dte"], t["expiry"], t["ticker"]
                        )
                        if hypothetical and hypothetical["pnl"] > 0:
                            missed_good += 1
                        skipped += 1
                        continue

                    result = simulate_fsm(
                        t["df"], wait, entry_price, t["contracts"],
                        t["direction"], t["dte"], t["expiry"], t["ticker"]
                    )
                    if result:
                        total_pnl += result["pnl"]
                        total_trades += 1
                        if result["pnl"] > 0:
                            total_wins += 1

                wr = total_wins / total_trades * 100 if total_trades > 0 else 0
                strategies.append({
                    "name": name,
                    "wait": wait,
                    "prem_thresh": prem_threshold,
                    "und_thresh": und_threshold,
                    "trades": total_trades,
                    "pnl": total_pnl,
                    "win_rate": wr,
                    "skipped": skipped,
                    "missed_good": missed_good,
                })

    df_strat = pd.DataFrame(strategies)

    # Find baseline (wait=0, no filters)
    baseline = df_strat[(df_strat["wait"] == 0) &
                        (df_strat["prem_thresh"].isna()) &
                        (df_strat["und_thresh"].isna())]
    baseline_pnl = baseline["pnl"].iloc[0] if len(baseline) > 0 else 0

    # ── Print results grouped by wait time ───────────────────────────────

    print(f"{'=' * 110}")
    print(f"STRATEGY COMPARISON — Baseline: {int(baseline['trades'].iloc[0])} trades, ${baseline_pnl:,.2f}")
    print(f"{'=' * 110}")

    for wait in [0, 1, 2, 3]:
        group = df_strat[df_strat["wait"] == wait].sort_values("pnl", ascending=False)
        print(f"\n── WAIT {wait} TICK{'S' if wait != 1 else ''} (~{wait} min) before entering ──")
        print(f"  {'Strategy':<40} {'Trades':>6} {'P&L':>11} {'vs Base':>9} "
              f"{'Win%':>5} {'Skip':>5} {'MissGood':>8}")
        print(f"  {'-' * 90}")

        for _, r in group.head(20).iterrows():
            diff = r["pnl"] - baseline_pnl
            marker = " <-- BEST" if r["pnl"] == group["pnl"].max() else ""
            print(f"  {r['name']:<40} {r['trades']:>6} ${r['pnl']:>9,.2f} "
                  f"${diff:>+7,.0f} {r['win_rate']:>4.0f}% {r['skipped']:>5} "
                  f"{r['missed_good']:>8}{marker}")

    # ── Combined filter: prem AND und both against ───────────────────────

    print(f"\n{'=' * 110}")
    print("COMBINED FILTERS: Skip only when BOTH premium AND underlying are against")
    print(f"{'=' * 110}")

    for wait in [1, 2, 3]:
        print(f"\n  Wait {wait} tick(s):")
        combos = [
            (-3, -0.03), (-3, -0.05), (-5, -0.03), (-5, -0.05),
            (-5, -0.08), (-8, -0.05), (-8, -0.08),
        ]
        print(f"    {'Prem Thresh':>11} {'Und Thresh':>10} {'Trades':>6} {'P&L':>11} "
              f"{'vs Base':>9} {'Win%':>5} {'Skip':>5} {'MissGood':>8}")
        print(f"    {'-' * 75}")

        for pt, ut in combos:
            total_pnl = 0
            total_trades = 0
            total_wins = 0
            skipped = 0
            missed_good = 0

            for t in trades:
                if wait >= len(t["prem_changes"]) or wait >= len(t["tick_data"]):
                    continue
                pc = t["prem_changes"][wait]
                uc = t["und_changes"][wait]
                entry_price = t["tick_data"][wait]["entry_price"]
                if entry_price is None or entry_price <= 0:
                    continue

                # Block only when BOTH are against
                blocked = False
                if pc is not None and uc is not None:
                    if pc < pt and uc < ut:
                        blocked = True

                if blocked:
                    hypothetical = simulate_fsm(
                        t["df"], wait, entry_price, t["contracts"],
                        t["direction"], t["dte"], t["expiry"], t["ticker"]
                    )
                    if hypothetical and hypothetical["pnl"] > 0:
                        missed_good += 1
                    skipped += 1
                    continue

                result = simulate_fsm(
                    t["df"], wait, entry_price, t["contracts"],
                    t["direction"], t["dte"], t["expiry"], t["ticker"]
                )
                if result:
                    total_pnl += result["pnl"]
                    total_trades += 1
                    if result["pnl"] > 0:
                        total_wins += 1

            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            diff = total_pnl - baseline_pnl
            print(f"    {pt:>10}% {ut:>9}% {total_trades:>6} ${total_pnl:>9,.2f} "
                  f"${diff:>+7,.0f} {wr:>4.0f}% {skipped:>5} {missed_good:>8}")

    # ── Best overall strategies ──────────────────────────────────────────

    print(f"\n{'=' * 110}")
    print("TOP 15 STRATEGIES BY P&L (across all wait times)")
    print(f"{'=' * 110}")
    top = df_strat.sort_values("pnl", ascending=False).head(15)
    print(f"  {'Rank':>4} {'Strategy':<40} {'Trades':>6} {'P&L':>11} {'vs Base':>9} "
          f"{'Win%':>5} {'Skip':>5} {'MissGood':>8}")
    print(f"  {'-' * 92}")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        diff = r["pnl"] - baseline_pnl
        print(f"  {rank:>4} {r['name']:<40} {r['trades']:>6} ${r['pnl']:>9,.2f} "
              f"${diff:>+7,.0f} {r['win_rate']:>4.0f}% {r['skipped']:>5} {r['missed_good']:>8}")

    # ── Cost of waiting (good trades we'd enter late) ────────────────────

    print(f"\n{'=' * 110}")
    print("COST OF WAITING: Entry price difference at tick 0 vs tick 1/2/3")
    print("(Positive = we paid more by waiting, Negative = we got a better price)")
    print(f"{'=' * 110}")

    for wait in [1, 2, 3]:
        prices_diff = []
        for t in trades:
            if wait >= len(t["tick_data"]):
                continue
            p0 = t["tick_data"][0]["entry_price"]
            pw = t["tick_data"][wait]["entry_price"]
            if p0 and pw and p0 > 0:
                diff_pct = (pw - p0) / p0 * 100
                prices_diff.append(diff_pct)

        if prices_diff:
            arr = np.array(prices_diff)
            better = (arr < 0).sum()
            worse = (arr > 0).sum()
            print(f"\n  Wait {wait} tick(s): median price change = {np.median(arr):+.2f}%  "
                  f"mean = {np.mean(arr):+.2f}%")
            print(f"    Better price: {better} trades ({better/len(arr)*100:.0f}%)  "
                  f"Worse price: {worse} trades ({worse/len(arr)*100:.0f}%)")
            # Bucket by premium direction
            for bucket, lo, hi in [("Fading", -999, -2), ("Flat", -2, 2), ("Rising", 2, 999)]:
                in_bucket = [d for d in prices_diff if lo <= d < hi]
                if in_bucket:
                    print(f"    {bucket:>8}: {len(in_bucket)} trades, "
                          f"median {np.median(in_bucket):+.2f}%, "
                          f"mean {np.mean(in_bucket):+.2f}%")


if __name__ == "__main__":
    main()
