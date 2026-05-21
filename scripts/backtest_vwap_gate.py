#!/usr/bin/env python3
"""Backtest: would requiring above_vwap OR at_support have improved results?

Uses PROPER multi-timeframe support detection (wick clustering across 5m/15m/1h/4h)
instead of naive "lowest low" approach.

For each historical Webull trade:
1. Pulls 5m candles from Polygon for the trade date (+ prior days for 1h/4h)
2. Aggregates into 15m, 1h, 4h candles
3. Computes VWAP from session volume-weighted typical price
4. Finds support levels via wick clustering across all timeframes
5. Checks if trade would pass: above_vwap OR at_real_support

Usage:
    python scripts/backtest_vwap_gate.py [--db path/to/raw_messages.db]
"""

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_owl.collectors.support_levels import (
    find_support_levels,
    is_at_support,
)

POLYGON_API_KEY = "Vk7gXTz6dbp_F69UmmqIx9BDEasHfExb"
POLYGON_BASE = "https://api.polygon.io"


# ---------------------------------------------------------------------------
# Fake CandleBar (matches what candle_cache uses)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CandleBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0


# ---------------------------------------------------------------------------
# Polygon data fetching
# ---------------------------------------------------------------------------

_candle_cache: dict[tuple[str, str, str], list[dict]] = {}


def get_polygon_candles(
    ticker: str, from_date: str, to_date: str, multiplier: int = 5, span: str = "minute"
) -> list[dict]:
    """Fetch candles from Polygon REST API with caching and retry."""
    cache_key = (ticker, from_date, f"{multiplier}{span}")
    if cache_key in _candle_cache:
        return _candle_cache[cache_key]

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}"
        f"/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                print(f"  Rate limited, sleeping 12s...")
                time.sleep(12)
                continue
            data = resp.json()
            results = data.get("results", [])
            _candle_cache[cache_key] = results
            return results
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt+1}/3), retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}, retrying...")
            time.sleep(2)

    _candle_cache[cache_key] = []
    return []


def aggregate_5m_to_tf(bars_5m: list[CandleBar], tf_minutes: int) -> list[CandleBar]:
    """Aggregate 5-minute bars into a higher timeframe."""
    bucket_ms = tf_minutes * 60 * 1000
    buckets: dict[int, list[CandleBar]] = {}
    for bar in bars_5m:
        bucket = int((bar.timestamp // bucket_ms) * bucket_ms)
        buckets.setdefault(bucket, []).append(bar)

    aggregated = []
    for bucket_ts in sorted(buckets):
        group = buckets[bucket_ts]
        aggregated.append(CandleBar(
            timestamp=bucket_ts,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
            vwap=group[-1].vwap if group[-1].vwap else 0.0,
        ))
    return aggregated


def build_candle_data(ticker: str, date_str: str, entry_ts_ms: int) -> dict | None:
    """Build multi-timeframe candle data for a trade entry.

    Fetches enough history for 1h/4h candle lookback.
    Returns dict matching CandleCache.get_candle_data() format.
    """
    # Fetch 5 trading days of 5m data (enough for 4h candles)
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    from_date = (dt - timedelta(days=7)).strftime("%Y-%m-%d")

    raw = get_polygon_candles(ticker, from_date, date_str)
    if not raw:
        return None

    # Only use candles up to entry time
    all_5m = [
        CandleBar(
            timestamp=c["t"],
            open=c["o"],
            high=c["h"],
            low=c["l"],
            close=c["c"],
            volume=c.get("v", 0),
            vwap=c.get("vw", 0),
        )
        for c in raw
        if c["t"] <= entry_ts_ms
    ]

    if len(all_5m) < 6:
        return None

    # Aggregate into higher timeframes
    bars_15m = aggregate_5m_to_tf(all_5m, 15)
    bars_1h = aggregate_5m_to_tf(all_5m, 60)
    bars_4h = aggregate_5m_to_tf(all_5m, 240)

    return {
        "5m": all_5m,
        "15m": bars_15m,
        "1h": bars_1h,
        "4h": bars_4h,
    }


def compute_vwap(candle_data: dict, entry_ts_ms: int) -> float | None:
    """Compute VWAP from session's 5m candles (today only)."""
    bars_5m = candle_data.get("5m", [])
    if not bars_5m:
        return None

    # Filter to today only (session VWAP)
    # Find the market open timestamp (first bar of today)
    entry_dt = datetime.fromtimestamp(entry_ts_ms / 1000, tz=None)
    today_start = entry_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_ms = today_start.timestamp() * 1000

    total_vp = 0.0
    total_vol = 0.0
    for bar in bars_5m:
        if bar.timestamp < today_start_ms:
            continue
        if bar.volume <= 0:
            continue
        typical = (bar.high + bar.low + bar.close) / 3
        total_vp += typical * bar.volume
        total_vol += bar.volume

    if total_vol <= 0:
        return None
    return total_vp / total_vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="journal/owlet-kody/raw_messages.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    trades = conn.execute("""
        SELECT id, ticker, direction, strike, option_type,
               premium_per_contract, webull_entry_fill_price,
               pnl_dollars, opened_at, exit_reason,
               COALESCE(exit_source, 'ai') as exit_source
        FROM paper_trades
        WHERE status = 'closed'
          AND webull_order_id IS NOT NULL
          AND COALESCE(exit_source, 'ai') = 'ai'
        ORDER BY opened_at
    """).fetchall()

    print(f"Found {len(trades)} closed Webull trades (AI exits)\n")

    results = []

    for i, trade in enumerate(trades):
        ticker = trade["ticker"]
        opened_at = trade["opened_at"]
        pnl = trade["pnl_dollars"] or 0
        direction = trade["direction"]

        dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00") if "Z" in opened_at else opened_at)
        date_str = dt.strftime("%Y-%m-%d")
        entry_ts_ms = int(dt.timestamp() * 1000)

        print(f"  [{i+1}/{len(trades)}] #{trade['id']} {ticker} {direction} {date_str}...", end="", flush=True)

        candle_data = build_candle_data(ticker, date_str, entry_ts_ms)
        time.sleep(0.25)  # rate limit

        if candle_data is None:
            print(" NO_DATA")
            results.append({
                "id": trade["id"], "ticker": ticker, "pnl": pnl,
                "direction": direction, "exit_reason": trade["exit_reason"],
                "status": "NO_DATA",
            })
            continue

        # Get current price at entry
        bars_5m = candle_data["5m"]
        entry_price = bars_5m[-1].close

        # Compute VWAP
        vwap = compute_vwap(candle_data, entry_ts_ms)
        above_vwap = entry_price >= vwap if vwap else None

        # Multi-TF support detection (the new proper way)
        at_support_result, support_detail = is_at_support(
            candle_data, current_price=entry_price,
            max_distance_pct=0.3, min_strength=3, min_confluence=1,
        )

        # Also find all support levels for reporting
        support_levels = find_support_levels(candle_data, entry_price)
        best_support = support_levels[0] if support_levels else None

        # Gate: pass if above_vwap OR at_support (with proper clustering)
        would_pass = bool(above_vwap) or at_support_result

        status_str = "PASS" if would_pass else "BLOCK"
        vwap_str = f"{'above' if above_vwap else 'below'}" if above_vwap is not None else "?"
        support_str = f"YES(s={best_support.strength},c={best_support.confluence})" if at_support_result and best_support else "NO"
        print(f" vwap={vwap_str} support={support_str} → {status_str} (${pnl:+.0f})")

        results.append({
            "id": trade["id"],
            "ticker": ticker,
            "direction": direction,
            "pnl": pnl,
            "exit_reason": trade["exit_reason"],
            "entry_price": round(entry_price, 2),
            "vwap": round(vwap, 2) if vwap else None,
            "above_vwap": above_vwap,
            "at_support": at_support_result,
            "support_detail": support_detail,
            "support_strength": best_support.strength if best_support else 0,
            "support_confluence": best_support.confluence if best_support else 0,
            "support_price": best_support.price if best_support else None,
            "would_pass": would_pass,
            "status": "OK",
        })

    # --- Analysis ---
    ok = [r for r in results if r["status"] == "OK"]
    passed = [r for r in ok if r["would_pass"]]
    blocked = [r for r in ok if not r["would_pass"]]
    no_data = [r for r in results if r["status"] != "OK"]

    print(f"\n{'=' * 95}")
    print(f"BACKTEST: VWAP + Multi-TF Support Gate (wick clustering)")
    print(f"  Support = wick clusters (3+ touches within 0.15% band)")
    print(f"  Confluence = same zone on multiple timeframes (5m/15m/1h/4h)")
    print(f"  Gate: PASS if (above_vwap OR at_support[strength>=3, dist<=0.3%])")
    print(f"{'=' * 95}")

    total_pnl = sum(r["pnl"] for r in ok)
    wins = sum(1 for r in ok if r["pnl"] > 0)
    print(f"\nALL TRADES ({len(ok)} analyzed, {len(no_data)} no data):")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Win rate: {wins / len(ok) * 100:.1f}%" if ok else "  N/A")
    print(f"  Avg P&L: ${total_pnl / len(ok):,.2f}" if ok else "  N/A")

    if passed:
        passed_pnl = sum(r["pnl"] for r in passed)
        passed_wins = sum(1 for r in passed if r["pnl"] > 0)
        print(f"\nWOULD PASS gate ({len(passed)} trades):")
        print(f"  Total P&L: ${passed_pnl:,.2f}")
        print(f"  Win rate: {passed_wins / len(passed) * 100:.1f}%")
        print(f"  Avg P&L: ${passed_pnl / len(passed):,.2f}")

    if blocked:
        blocked_pnl = sum(r["pnl"] for r in blocked)
        blocked_wins = sum(1 for r in blocked if r["pnl"] > 0)
        print(f"\nWOULD BE BLOCKED ({len(blocked)} trades):")
        print(f"  Total P&L: ${blocked_pnl:,.2f}")
        print(f"  Win rate: {blocked_wins / len(blocked) * 100:.1f}%")
        print(f"  Avg P&L: ${blocked_pnl / len(blocked):,.2f}")
        if blocked_pnl < 0:
            print(f"  >>> Savings from blocking: ${-blocked_pnl:,.2f}")
        else:
            print(f"  >>> Would LOSE profitable trades worth ${blocked_pnl:,.2f}")

    # Blocked trade details
    if blocked:
        print(f"\n{'─' * 95}")
        print(f"BLOCKED TRADE DETAILS:")
        print(f"{'ID':>5} {'Ticker':>6} {'Dir':>5} {'P&L':>10} {'Exit Reason':>20} {'Price':>8} {'VWAP':>8} {'Support':>20}")
        print(f"{'─' * 95}")
        for r in sorted(blocked, key=lambda x: x["pnl"]):
            sup = r.get("support_detail", "")[:40] if r.get("support_detail") else "none"
            print(
                f"{r['id']:>5} {r['ticker']:>6} {r['direction']:>5} "
                f"${r['pnl']:>8,.2f} {r['exit_reason']:>20} "
                f"{r['entry_price']:>8.2f} {r.get('vwap', 0) or 0:>8.2f} {sup}"
            )

    # Direction-aware analysis
    print(f"\n{'=' * 95}")
    print("DIRECTION-AWARE ANALYSIS")
    print("(Calls: above VWAP = good. Puts: below VWAP = good. Both: at_support = good)")
    print("=" * 95)

    for direction in ["call", "put"]:
        dir_trades = [r for r in ok if r["direction"] == direction]
        if not dir_trades:
            continue

        if direction == "call":
            smart_pass = [r for r in dir_trades if r.get("above_vwap") or r.get("at_support")]
            smart_block = [r for r in dir_trades if not r.get("above_vwap") and not r.get("at_support")]
        else:
            smart_pass = [r for r in dir_trades if not r.get("above_vwap") or r.get("at_support")]
            smart_block = [r for r in dir_trades if r.get("above_vwap") and not r.get("at_support")]

        dir_pnl = sum(r["pnl"] for r in dir_trades)
        print(f"\n{direction.upper()}S ({len(dir_trades)} trades, P&L ${dir_pnl:,.2f}):")
        if smart_pass:
            pass_pnl = sum(r["pnl"] for r in smart_pass)
            print(f"  Pass: {len(smart_pass)} trades, P&L ${pass_pnl:,.2f}, WR {sum(1 for r in smart_pass if r['pnl'] > 0) / len(smart_pass) * 100:.1f}%")
        if smart_block:
            block_pnl = sum(r["pnl"] for r in smart_block)
            print(f"  Block: {len(smart_block)} trades, P&L ${block_pnl:,.2f}, WR {sum(1 for r in smart_block if r['pnl'] > 0) / len(smart_block) * 100:.1f}%")
            if block_pnl < 0:
                print(f"  >>> Direction-aware blocking saves ${-block_pnl:,.2f}")

    # Support quality breakdown
    print(f"\n{'=' * 95}")
    print("SUPPORT QUALITY ANALYSIS (how strength/confluence affects P&L)")
    print("=" * 95)

    for label, trades_subset in [
        ("Has support (any)", [r for r in ok if r.get("at_support")]),
        ("Multi-TF support (confluence>=2)", [r for r in ok if r.get("support_confluence", 0) >= 2]),
        ("Strong support (strength>=5)", [r for r in ok if r.get("support_strength", 0) >= 5]),
        ("No support nearby", [r for r in ok if not r.get("at_support")]),
    ]:
        if trades_subset:
            sub_pnl = sum(r["pnl"] for r in trades_subset)
            sub_wins = sum(1 for r in trades_subset if r["pnl"] > 0)
            print(f"  {label}: {len(trades_subset)} trades, P&L ${sub_pnl:,.2f}, WR {sub_wins / len(trades_subset) * 100:.1f}%, Avg ${sub_pnl / len(trades_subset):,.2f}")

    conn.close()


if __name__ == "__main__":
    main()
