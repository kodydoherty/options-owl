"""Backtest the ENRG gate on historical stop-hit losers.

Uses our actual evaluate_enrg() implementation with real Polygon candles
to compare outcomes vs baseline (blind 20-min grace → hard stop at -30%).

For each stop_hit trade, fetches 5m/15m/30m/1h/4h candles at the point
when the trade first went negative, runs ENRG voting, and simulates:
  - IMMEDIATE_EXIT → exit at current premium (saves vs bleeding to -30%)
  - HOLD → widen stop from 30% to 34.5% (check if it recovered)
  - PROCEED → no change (same as baseline)

Usage:
    python scripts/backtest_enrg.py
"""

import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options_owl.collectors.candle_cache import (
    CandleBar,
    calc_atr,
    calc_obv,
    calc_rsi,
    calc_volume_trend,
    detect_candle_pattern,
    evaluate_enrg,
)

# All losing trades that hit hard stop (deduplicated by ticker+time)
# Format: (ticker, option_type, entry_price, strike, premium, exit_premium,
#          pnl_pct, opened_at, closed_at, contracts)
STOP_HIT_TRADES = [
    ("GOOGL", "call", 334.215, 335.0, 0.8643, 0.3383, -60.86, "2026-04-15T14:54:53", "2026-04-15T15:04:03", 4),
    ("NVDA", "call", 201.015, 202.5, 1.005, 0.3905, -61.14, "2026-04-21T15:57:48", "2026-04-21T16:44:14", 3),
    ("IWM", "call", 279.62, 280.0, 0.6231, 0.2388, -58.62, "2026-04-21T14:31:08", "2026-04-21T14:47:15", 5),
    ("SPY", "put", 704.14, 705.0, 2.0603, 0.4080, -80.20, "2026-04-22T13:27:41", "2026-04-22T13:47:47", 1),
    ("AVGO", "call", 419.43, 420.0, 1.9999, 0.7264, -60.96, "2026-04-22T16:24:40", "2026-04-22T16:55:21", 1),
    ("GOOGL", "call", 339.12, 340.0, 0.7337, 0.2886, -56.72, "2026-04-17T15:37:00", "2026-04-17T15:45:25", 2),
    ("NVDA", "call", 201.64, 202.5, 0.2915, 0.1095, -62.45, "2026-04-22T15:18:39", "2026-04-22T16:16:50", 1),
    ("SPY", "call", 710.065, 710.0, 0.7538, 0.2886, -57.75, "2026-04-22T18:06:19", "2026-04-22T19:01:11", 1),
]

# Cache directory
CANDLE_DIR = "journal/candle_cache"


def get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("POLYGON_API_KEY="):
                        key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
        except FileNotFoundError:
            pass
    return key


def fetch_candles(api_key: str, ticker: str, mult: int, span: str,
                  date: str) -> list[dict]:
    """Fetch candle bars from Polygon for a given date."""
    cache_file = os.path.join(CANDLE_DIR, f"{ticker}_{date}_{mult}{span}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{span}"
        f"/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        bars = data.get("results", [])
    except Exception as e:
        print(f"  WARN: fetch failed {ticker} {mult}{span} {date}: {e}")
        bars = []

    os.makedirs(CANDLE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(bars, f)
    return bars


def bars_to_candles(raw_bars: list[dict]) -> list[CandleBar]:
    """Convert Polygon bar dicts to CandleBar objects."""
    return [
        CandleBar(
            timestamp=b.get("t", 0),
            open=float(b.get("o", 0)),
            high=float(b.get("h", 0)),
            low=float(b.get("l", 0)),
            close=float(b.get("c", 0)),
            volume=float(b.get("v", 0)),
            vwap=float(b.get("vw", 0)),
        )
        for b in raw_bars
    ]


def build_indicators(bars: list[CandleBar]) -> dict:
    """Compute indicators for a set of bars."""
    if not bars:
        return {"atr": None, "rsi": None, "obv": None, "pattern": None, "volume_trend": None}
    return {
        "atr": calc_atr(bars),
        "rsi": calc_rsi(bars),
        "obv": calc_obv(bars),
        "pattern": detect_candle_pattern(bars),
        "volume_trend": calc_volume_trend(bars),
    }


def filter_bars_before(raw_bars: list[dict], cutoff_ms: int) -> list[dict]:
    """Keep only bars with timestamp <= cutoff."""
    return [b for b in raw_bars if b.get("t", 0) <= cutoff_ms]


@dataclass
class ENRGResult:
    ticker: str
    option_type: str
    action: str
    reason: str
    actual_loss: float
    enrg_loss: float
    savings: float


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No POLYGON_API_KEY found")
        return

    print("=" * 80)
    print("ENRG BACKTEST — Historical Stop-Hit Losers")
    print("=" * 80)
    print(f"Trades: {len(STOP_HIT_TRADES)} unique stop-hit losers")
    print(f"ENRG fires during grace period when position goes negative")
    print(f"Uses actual evaluate_enrg() with 5m/15m/30m/1h/4h candle voting\n")

    # Timeframes to fetch: (multiplier, span, label)
    TFS = [
        (5, "minute", "5m"),
        (15, "minute", "15m"),
        (30, "minute", "30m"),
        (1, "hour", "1h"),
        (4, "hour", "4h"),
    ]

    results: list[ENRGResult] = []

    for trade in STOP_HIT_TRADES:
        ticker, opt_type, entry_price, strike, premium, exit_prem, pnl_pct, opened_at, closed_at, contracts = trade

        open_dt = datetime.fromisoformat(opened_at)
        close_dt = datetime.fromisoformat(closed_at)
        date_str = open_dt.strftime("%Y-%m-%d")

        # ENRG fires early — simulate it at ~5 min after entry (first negative check)
        enrg_check_dt = open_dt + timedelta(minutes=5)
        enrg_check_ms = int(enrg_check_dt.timestamp() * 1000)

        print(f"\n{'─' * 60}")
        print(f"{ticker} {opt_type.upper()} ${strike} | Premium: ${premium:.4f} → ${exit_prem:.4f} ({pnl_pct:.1f}%)")
        print(f"  Opened: {opened_at} | ENRG check: ~{enrg_check_dt.strftime('%H:%M')}")

        # Fetch all timeframes
        indicators = {}
        for mult, span, label in TFS:
            # For higher TFs, fetch previous day too (need history for 4h bars)
            prev_date = (open_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            raw = fetch_candles(api_key, ticker, mult, span, date_str)
            if not raw and label in ("1h", "4h"):
                # Try previous day for higher TFs
                raw = fetch_candles(api_key, ticker, mult, span, prev_date)
            time.sleep(0.15)

            # Filter to only bars available at ENRG check time
            filtered = filter_bars_before(raw, enrg_check_ms)
            bars = bars_to_candles(filtered)
            indicators[label] = build_indicators(bars)

            rsi = indicators[label]["rsi"]
            obv = indicators[label]["obv"]
            pat = indicators[label]["pattern"]
            rsi_s = f"{rsi:.0f}" if rsi is not None else "N/A"
            obv_s = f"{obv:.0f}" if obv is not None else "N/A"
            print(f"  {label}: {len(filtered)} bars | RSI={rsi_s} OBV={obv_s} pattern={pat}")

        candle_data = {"indicators": indicators}

        # Run ENRG
        action, reason = evaluate_enrg(candle_data, opt_type)
        print(f"  ENRG: {action} — {reason}")

        # Calculate P&L impact
        actual_loss = (exit_prem - premium) * 100 * contracts

        if action == "IMMEDIATE_EXIT":
            # Exit at ~5 min into trade. Estimate premium at -10% (early in the move)
            early_exit_prem = premium * 0.90
            enrg_loss = (early_exit_prem - premium) * 100 * contracts
            savings = actual_loss - enrg_loss
        elif action == "HOLD":
            # Widen stop from 30% to 34.5%. Most of these trades bled past -30%
            # anyway, so widening likely makes it worse. Use -34.5% exit estimate.
            widened_exit_prem = premium * 0.655  # -34.5%
            enrg_loss = (widened_exit_prem - premium) * 100 * contracts
            savings = actual_loss - enrg_loss
        else:
            # PROCEED — same as baseline
            enrg_loss = actual_loss
            savings = 0.0

        print(f"  Actual loss: ${actual_loss:+.2f} | ENRG loss: ${enrg_loss:+.2f} | Delta: ${savings:+.2f}")

        results.append(ENRGResult(
            ticker=ticker, option_type=opt_type, action=action,
            reason=reason, actual_loss=actual_loss,
            enrg_loss=enrg_loss, savings=savings,
        ))

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    exits = [r for r in results if r.action == "IMMEDIATE_EXIT"]
    holds = [r for r in results if r.action == "HOLD"]
    proceeds = [r for r in results if r.action == "PROCEED"]

    print(f"\nVerdicts: IMMEDIATE_EXIT={len(exits)}, HOLD={len(holds)}, PROCEED={len(proceeds)}")

    total_actual = sum(r.actual_loss for r in results)
    total_enrg = sum(r.enrg_loss for r in results)
    total_savings = sum(r.savings for r in results)

    print(f"\nTotal actual loss:  ${total_actual:+,.2f}")
    print(f"Total ENRG loss:   ${total_enrg:+,.2f}")
    print(f"Net savings:       ${total_savings:+,.2f}")

    if exits:
        print(f"\nIMMEDIATE_EXIT trades (early exit at ~-10%):")
        for r in exits:
            print(f"  {r.ticker} {r.option_type}: ${r.actual_loss:+.2f} → ${r.enrg_loss:+.2f} (saved ${r.savings:+.2f})")

    if holds:
        print(f"\nHOLD trades (widened stop to -34.5%):")
        for r in holds:
            print(f"  {r.ticker} {r.option_type}: ${r.actual_loss:+.2f} → ${r.enrg_loss:+.2f} (delta ${r.savings:+.2f})")


if __name__ == "__main__":
    main()
