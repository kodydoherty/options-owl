"""Analyze whether price-related entry gates are costing or saving money.

For each gate-blocked trade, simulates what would have happened if it went through
using the V5 FSM exit engine on actual option price data.

Usage:
    python scripts/analyze_gate_impact.py              # last 60 days
    python scripts/analyze_gate_impact.py --days 126   # all available data
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from options_owl.risk.exit_v5.config import get_ticker_config, get_max_otm_distance
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
MODEL_DIR = PROJECT_DIR / "journal" / "models" / "ml_v3"

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSLA", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    "COIN", "NFLX", "JPM", "BA", "MU", "SMCI",
]
EXCLUDED_TICKERS = {"MSFT", "COIN", "AVGO", "MU"}

# Match production settings
PORTFOLIO_START = 23_000
MAX_CONCURRENT = 8
MAX_RISK_PCT = 0.75
MAX_POSITION_PCT = 0.15
PREMIUM_CAP = 6.0
MIN_PREMIUM_FLOOR = 0.20
SPREAD_GATE_PCT = 15.0
MAX_POSITION_DOLLARS = 5_000
MAX_CONTRACTS = 200
SCAN_START_MIN = 5
SCAN_END_MIN = 90

_V6_SETTINGS = SimpleNamespace(
    ENABLE_V6_BREAKEVEN_RATCHET=True,
    V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V6_SCALEOUT=True,
    V6_SCALEOUT_GAIN_PCT=20.0,
    V6_SCALEOUT_FRACTION=0.333,
    V6_SCALEOUT_MIN_CONTRACTS=3,
    ENABLE_V6_2PM_TIGHTEN=True,
    V6_2PM_TIGHTEN_FACTOR=0.70,
)


def load_pattern_model():
    model_path = MODEL_DIR / "pattern_entry.txt"
    meta_path = MODEL_DIR / "pattern_entry_meta.json"
    if not model_path.exists():
        print(f"ERROR: Pattern model not found at {model_path}")
        sys.exit(1)
    import json
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def load_entry_model():
    model_path = MODEL_DIR / "entry_timing.txt"
    if not model_path.exists():
        return None, []
    model = lgb.Booster(model_file=str(model_path))
    features = model.feature_name()
    return model, features


def build_features(td, minute, feature_names):
    """Build feature dict for a ticker/minute from option data arrays."""
    feat = {}
    closes = td["closes"]
    bids = td["bids"]
    asks = td["asks"]
    volumes = td["volumes"]
    ois = td["ois"]
    underlyings = td["underlyings"]
    n = td["n_rows"]

    if minute >= n:
        return None

    price = float(closes[minute])
    if price <= 0 or np.isnan(price):
        return None

    bid = float(bids[minute]) if bids[minute] > 0 else 0
    ask = float(asks[minute]) if asks[minute] > 0 else price
    vol = float(volumes[minute]) if not np.isnan(volumes[minute]) else 0
    oi = float(ois[minute]) if not np.isnan(ois[minute]) else 0
    und = float(underlyings[minute]) if not np.isnan(underlyings[minute]) else 0

    # Basic features
    feat["premium"] = price
    feat["bid"] = bid
    feat["ask"] = ask
    feat["spread"] = (ask - bid) / ask * 100 if ask > 0 else 0
    feat["volume"] = vol
    feat["open_interest"] = oi
    feat["underlying_price"] = und
    feat["strike"] = td["strike"]
    feat["moneyness"] = (und - td["strike"]) / und * 100 if und > 0 else 0
    feat["minute_of_day"] = minute
    feat["dte"] = td["dte"]

    # Returns
    if minute >= 1 and closes[minute - 1] > 0 and not np.isnan(closes[minute - 1]):
        feat["return_1m"] = (price / closes[minute - 1] - 1) * 100
    else:
        feat["return_1m"] = 0

    if minute >= 5 and closes[minute - 5] > 0 and not np.isnan(closes[minute - 5]):
        feat["return_5m"] = (price / closes[minute - 5] - 1) * 100
    else:
        feat["return_5m"] = 0

    if minute >= 10 and closes[minute - 10] > 0 and not np.isnan(closes[minute - 10]):
        feat["return_10m"] = (price / closes[minute - 10] - 1) * 100
    else:
        feat["return_10m"] = 0

    # Volume features
    if minute >= 5:
        recent_vol = [float(volumes[i]) for i in range(max(0, minute - 5), minute)
                      if not np.isnan(volumes[i])]
        feat["avg_volume_5m"] = np.mean(recent_vol) if recent_vol else 0
        feat["volume_ratio"] = vol / feat["avg_volume_5m"] if feat["avg_volume_5m"] > 0 else 1
    else:
        feat["avg_volume_5m"] = vol
        feat["volume_ratio"] = 1

    # Underlying move
    if minute >= 1 and not np.isnan(underlyings[minute - 1]) and underlyings[minute - 1] > 0:
        feat["underlying_return_1m"] = (und / underlyings[minute - 1] - 1) * 100
    else:
        feat["underlying_return_1m"] = 0

    # Volatility
    if minute >= 10:
        recent_rets = []
        for i in range(minute - 10, minute):
            if i >= 1 and closes[i] > 0 and closes[i - 1] > 0:
                recent_rets.append(closes[i] / closes[i - 1] - 1)
        feat["volatility_10m"] = np.std(recent_rets) * 100 if len(recent_rets) > 2 else 0
    else:
        feat["volatility_10m"] = 0

    feat["bid_ask_ratio"] = bid / ask if ask > 0 else 0
    feat["premium_to_underlying"] = price / und * 100 if und > 0 else 0

    return feat


def load_ticker_data(conn, date_str, ticker):
    """Load option OHLC data for a ticker/date, return arrays."""
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, open_interest,
               bid, ask, underlying_price, strike, expiry_date
        FROM option_ohlc
        WHERE date(timestamp) = ? AND ticker = ? AND option_type = 'call'
        ORDER BY timestamp
    """, (date_str, ticker)).fetchall()

    if len(rows) < 20:
        return None

    n = len(rows)
    strike = float(rows[0][10]) if rows[0][10] else 0
    expiry_str = rows[0][11] if rows[0][11] else ""

    # Parse expiry for DTE
    dte = 0
    if expiry_str:
        try:
            exp_date = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
            trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            dte = (exp_date - trade_date).days
        except Exception:
            pass

    closes = np.full(n, np.nan)
    bids = np.full(n, np.nan)
    asks = np.full(n, np.nan)
    volumes = np.full(n, np.nan)
    ois = np.full(n, np.nan)
    underlyings = np.full(n, np.nan)

    for i, r in enumerate(rows):
        closes[i] = float(r[4]) if r[4] else np.nan
        bids[i] = float(r[7]) if r[7] else np.nan
        asks[i] = float(r[8]) if r[8] else np.nan
        volumes[i] = float(r[5]) if r[5] else np.nan
        ois[i] = float(r[6]) if r[6] else np.nan
        underlyings[i] = float(r[9]) if r[9] else np.nan

    return {
        "closes": closes, "bids": bids, "asks": asks,
        "volumes": volumes, "ois": ois, "underlyings": underlyings,
        "strike": strike, "n_rows": n, "dte": dte, "expiry_date": expiry_str,
    }


def simulate_trade(td, entry_minute, entry_premium, ticker, contracts):
    """Run V5 FSM on actual price data from entry_minute to EOD. Returns P&L."""
    tcfg = get_ticker_config(ticker, use_per_ticker=True)
    fsm = ExitFSM(tcfg, settings=_V6_SETTINGS)

    entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=entry_minute)

    und_0 = 0
    for i in range(entry_minute, min(entry_minute + 5, td["n_rows"])):
        u = td["underlyings"][i]
        if not np.isnan(u) and u > 0:
            und_0 = float(u)
            break

    state = TradeState(
        trade_id=0, ticker=ticker, option_type="call",
        entry_premium=entry_premium, entry_time=entry_ts,
        contracts=contracts, peak_premium=entry_premium,
        entry_underlying_price=und_0,
        dte=td["dte"], expiry_date=td["expiry_date"] or "",
    )

    locked_pnl = 0.0
    remaining = contracts
    exit_reason = "eod"
    peak_pct = 0.0

    for m in range(entry_minute + 1, td["n_rows"]):
        prem = td["closes"][m]
        if np.isnan(prem) or prem <= 0:
            continue

        und = td["underlyings"][m] if not np.isnan(td["underlyings"][m]) else und_0
        now = entry_ts + timedelta(minutes=(m - entry_minute))

        state.peak_premium = max(state.peak_premium, prem)
        gain_pct = (prem / entry_premium - 1) * 100 if entry_premium > 0 else 0
        peak_pct = max(peak_pct, gain_pct)

        action = fsm.evaluate(
            current_premium=prem,
            entry_premium=entry_premium,
            now=now,
            underlying_price=und,
            state=state,
        )

        if action and action.should_exit:
            # Check scaleout
            if action.reason and "scaleout" in action.reason.value:
                sell_ct = max(1, int(remaining * 0.333))
                partial_pnl = sell_ct * (prem - entry_premium) * 100
                locked_pnl += partial_pnl
                remaining -= sell_ct
                if remaining <= 0:
                    exit_reason = action.reason.value
                    break
                continue

            exit_prem = prem
            final_pnl = locked_pnl + remaining * (exit_prem - entry_premium) * 100
            exit_reason = action.reason.value if action.reason else "unknown"
            return final_pnl, exit_reason, peak_pct

    # EOD: close at last price
    last_prem = 0
    for i in range(td["n_rows"] - 1, entry_minute, -1):
        if not np.isnan(td["closes"][i]) and td["closes"][i] > 0:
            last_prem = td["closes"][i]
            break
    if last_prem <= 0:
        last_prem = entry_premium * 0.1  # assume near-worthless

    final_pnl = locked_pnl + remaining * (last_prem - entry_premium) * 100
    return final_pnl, exit_reason, peak_pct


def main():
    parser = argparse.ArgumentParser(description="Gate Impact Analysis")
    parser.add_argument("--days", type=int, default=60, help="Last N trading days")
    args = parser.parse_args()

    print("=" * 80)
    print("GATE IMPACT ANALYSIS: Are price gates costing or saving money?")
    print("=" * 80)

    # Load models
    pattern_model, pattern_meta = load_pattern_model()
    pattern_features = pattern_meta["features"]
    pattern_threshold = 0.74
    entry_model, entry_features = load_entry_model()
    entry_threshold = 0.80

    # Get dates
    conn = sqlite3.connect(THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1 DESC
    """).fetchall()]

    end_date = all_dates[0]
    start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]
    dates = [d for d in sorted(all_dates) if start_date <= d <= end_date]
    print(f"Analyzing {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    tickers = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

    # Track blocked trades and their simulated outcomes
    gate_results = defaultdict(lambda: {"blocked": [], "would_win": 0, "would_lose": 0,
                                         "total_pnl": 0.0, "count": 0})
    total_allowed = {"count": 0, "wins": 0, "pnl": 0.0}

    for day_idx, date_str in enumerate(dates):
        if day_idx % 10 == 0:
            print(f"  Processing day {day_idx + 1}/{len(dates)} ({date_str})...")

        for ticker in tickers:
            td = load_ticker_data(conn, date_str, ticker)
            if td is None:
                continue

            for minute in range(SCAN_START_MIN, min(SCAN_END_MIN + 1, td["n_rows"] - 20)):
                # Step 1: Pattern model
                feat = build_features(td, minute, pattern_features)
                if feat is None:
                    continue

                X = np.array([[feat.get(f, 0) for f in pattern_features]], dtype=np.float32)
                try:
                    conf = float(pattern_model.predict(X)[0])
                except Exception:
                    continue

                if conf < pattern_threshold:
                    continue

                # Step 2: Entry timing
                if entry_model and entry_features:
                    X_et = np.array([[feat.get(f, 0) for f in entry_features]], dtype=np.float32)
                    try:
                        et_score = float(entry_model.predict(X_et)[0])
                    except Exception:
                        continue
                    if et_score < entry_threshold:
                        continue

                # Step 3: Check which gates would block this trade
                entry_premium = float(td["asks"][minute]) if td["asks"][minute] > 0 else float(td["closes"][minute])
                if entry_premium <= 0 or np.isnan(entry_premium):
                    continue

                # Determine contracts for simulation
                deployable = PORTFOLIO_START * MAX_RISK_PCT
                per_slot = deployable / MAX_CONCURRENT
                cost_per = entry_premium * 100
                if cost_per <= 0:
                    continue
                scaled = per_slot * 0.85
                raw_ct = int(scaled / cost_per)
                cap_ct = int(PORTFOLIO_START * MAX_POSITION_PCT / cost_per)
                dollar_ct = int(MAX_POSITION_DOLLARS / cost_per)
                contracts = max(1, min(raw_ct, cap_ct, dollar_ct, MAX_CONTRACTS))

                # Check each price gate
                blocked_by = []

                # min_premium gate
                if entry_premium < MIN_PREMIUM_FLOOR:
                    blocked_by.append("min_premium")

                # premium_cap gate
                if entry_premium > PREMIUM_CAP:
                    blocked_by.append("premium_cap")

                # spread_gate
                bid_val = float(td["bids"][minute]) if td["bids"][minute] > 0 else 0
                if bid_val > 0 and entry_premium > 0:
                    spread = (entry_premium - bid_val) / entry_premium * 100
                    if spread > SPREAD_GATE_PCT:
                        blocked_by.append("spread_gate")

                # otm_distance gate
                und_at_entry = float(td["underlyings"][minute]) if not np.isnan(td["underlyings"][minute]) else 0
                if und_at_entry > 0 and td["strike"] > 0:
                    call_otm_dollars = td["strike"] - und_at_entry
                    max_otm_dollars = get_max_otm_distance(ticker)
                    if call_otm_dollars > max_otm_dollars:
                        blocked_by.append("otm_distance")

                if not blocked_by:
                    # Trade would be allowed — simulate for baseline
                    pnl, reason, peak = simulate_trade(td, minute, entry_premium, ticker, contracts)
                    total_allowed["count"] += 1
                    total_allowed["pnl"] += pnl
                    if pnl > 0:
                        total_allowed["wins"] += 1
                    # Skip to next minute block (don't re-enter same ticker same minute)
                    continue

                # Trade was blocked — simulate what would have happened
                pnl, reason, peak = simulate_trade(td, minute, entry_premium, ticker, contracts)

                for gate in blocked_by:
                    gate_results[gate]["count"] += 1
                    gate_results[gate]["total_pnl"] += pnl
                    if pnl > 0:
                        gate_results[gate]["would_win"] += 1
                    else:
                        gate_results[gate]["would_lose"] += 1
                    gate_results[gate]["blocked"].append({
                        "date": date_str, "ticker": ticker, "minute": minute,
                        "premium": entry_premium, "pnl": pnl, "reason": reason,
                        "peak": peak, "contracts": contracts,
                    })

    conn.close()

    # ── Results ──
    print(f"\n{'=' * 80}")
    print("RESULTS: Gate Impact on Blocked Trades")
    print(f"{'=' * 80}")

    print(f"\nBaseline (allowed trades): {total_allowed['count']} trades, "
          f"WR {total_allowed['wins']}/{total_allowed['count']} "
          f"({total_allowed['wins']/max(1,total_allowed['count'])*100:.0f}%), "
          f"P&L ${total_allowed['pnl']:+,.0f}")

    print(f"\n{'Gate':<16} {'Blocked':>8} {'Would Win':>10} {'Would Lose':>11} "
          f"{'WR%':>6} {'Sim P&L':>12} {'Avg P&L':>10} {'Verdict':>10}")
    print("-" * 90)

    for gate_name in ["premium_cap", "otm_distance", "min_premium", "spread_gate"]:
        g = gate_results[gate_name]
        if g["count"] == 0:
            print(f"{gate_name:<16} {'0':>8} {'—':>10} {'—':>11} {'—':>6} {'—':>12} {'—':>10} {'no data':>10}")
            continue

        wr = g["would_win"] / g["count"] * 100
        avg_pnl = g["total_pnl"] / g["count"]

        if g["total_pnl"] < 0:
            verdict = "KEEP"  # gate is saving money
        else:
            verdict = "REMOVE?"  # gate is blocking winners

        print(f"{gate_name:<16} {g['count']:>8} {g['would_win']:>10} {g['would_lose']:>11} "
              f"{wr:>5.0f}% ${g['total_pnl']:>+10,.0f} ${avg_pnl:>+8,.0f} {verdict:>10}")

    # Show top blocked trades by absolute P&L
    print(f"\n{'=' * 80}")
    print("TOP 15 BLOCKED TRADES BY |P&L| (what we missed or dodged)")
    print(f"{'=' * 80}")
    print(f"{'Date':<12} {'Ticker':<7} {'Gate':<16} {'Min':>4} {'Prem':>6} {'Ct':>4} "
          f"{'Peak%':>7} {'P&L':>10} {'Exit':>18}")
    print("-" * 90)

    all_blocked = []
    for gate_name, g in gate_results.items():
        for t in g["blocked"]:
            t["gate"] = gate_name
            all_blocked.append(t)

    # Deduplicate (same trade blocked by multiple gates)
    seen = set()
    unique_blocked = []
    for t in all_blocked:
        key = (t["date"], t["ticker"], t["minute"])
        if key not in seen:
            seen.add(key)
            unique_blocked.append(t)

    unique_blocked.sort(key=lambda x: abs(x["pnl"]), reverse=True)

    for t in unique_blocked[:15]:
        print(f"{t['date']:<12} {t['ticker']:<7} {t['gate']:<16} {t['minute']:>4} "
              f"${t['premium']:>5.2f} {t['contracts']:>4} {t['peak']:>+6.0f}% "
              f"${t['pnl']:>+9,.0f} {t['reason']:>18}")

    # Summary recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    total_blocked_pnl = sum(g["total_pnl"] for g in gate_results.values())
    total_blocked_count = len(unique_blocked)
    print(f"  Total unique blocked trades: {total_blocked_count}")
    print(f"  Total simulated P&L if all allowed: ${total_blocked_pnl:+,.0f}")
    if total_blocked_pnl < 0:
        print(f"  Gates are SAVING ${abs(total_blocked_pnl):,.0f} by blocking losers. KEEP THEM.")
    else:
        print(f"  Gates are COSTING ${total_blocked_pnl:,.0f} by blocking winners. Consider loosening.")

    # Per-gate detail
    for gate_name in ["premium_cap", "otm_distance", "min_premium", "spread_gate"]:
        g = gate_results[gate_name]
        if g["count"] == 0:
            continue
        print(f"\n  {gate_name}: {g['count']} blocked → sim P&L ${g['total_pnl']:+,.0f}")
        if g["total_pnl"] > 0:
            # Show what we're missing
            winners = [t for t in g["blocked"] if t["pnl"] > 0]
            if winners:
                top = sorted(winners, key=lambda x: x["pnl"], reverse=True)[:3]
                for t in top:
                    print(f"    Missed winner: {t['date']} {t['ticker']} ${t['pnl']:+,.0f} "
                          f"(peak +{t['peak']:.0f}%)")
        else:
            losers = [t for t in g["blocked"] if t["pnl"] < 0]
            if losers:
                top = sorted(losers, key=lambda x: x["pnl"])[:3]
                for t in top:
                    print(f"    Dodged loser: {t['date']} {t['ticker']} ${t['pnl']:+,.0f}")


if __name__ == "__main__":
    main()
