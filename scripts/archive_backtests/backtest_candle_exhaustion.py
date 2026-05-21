"""Backtest candle-based exhaustion detection against real 37 bot trades.

Downloads 5m candle bars from Polygon for each trade's ticker + date,
then replays the exhaustion logic at each snapshot during the trade.

Compares:
  1. Baseline (current v2.1 exits, vol-peak uses price momentum only)
  2. With candle exhaustion (vol-peak uses RSI + OBV + patterns + volume)

Shows which trades would have been improved by catching exhaustion earlier.

Usage:
    python scripts/backtest_candle_exhaustion.py [--download-only] [--no-download]
"""

import json
import os
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options_owl.collectors.candle_cache import (
    CandleBar,
    calc_atr,
    calc_obv,
    calc_rsi,
    calc_volume_trend,
    check_exhaustion,
    detect_candle_pattern,
)

TRADES_DB = os.environ.get("TRADES_DB", "journal/owlet-kody/raw_messages.db")
HARVESTER_DB = os.environ.get("HARVESTER_DB", "journal/owlet-harvester/options_data.db")
CANDLE_DIR = Path("journal/candle_cache")
POLYGON_BASE = "https://api.polygon.io"


def get_api_key():
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("POLYGON_API_KEY", "")
    return key


# ---------------------------------------------------------------------------
# 1. Download candles from Polygon
# ---------------------------------------------------------------------------

def download_candles(ticker: str, date: str, api_key: str, timeframe: str = "5"):
    """Download 5m candle bars for ticker on date. Returns list of bar dicts."""
    cache_file = CANDLE_DIR / f"{ticker}_{date}_{timeframe}m.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{timeframe}/minute"
        f"/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARN: Polygon fetch failed for {ticker} {date}: {e}")
        return []

    bars = data.get("results", [])
    if not bars:
        print(f"  WARN: No candle data for {ticker} {date} ({timeframe}m)")
        return []

    # Cache to disk
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(bars, f)

    print(f"  Downloaded {len(bars)} bars: {ticker} {date} ({timeframe}m)")
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


def build_indicators_at_time(candles: list[CandleBar], up_to_ts: float) -> dict:
    """Compute indicators using only candles up to a given timestamp."""
    bars = [c for c in candles if c.timestamp <= up_to_ts]
    if not bars:
        return {"atr": None, "rsi": None, "obv": None, "pattern": None, "volume_trend": None}
    return {
        "atr": calc_atr(bars),
        "rsi": calc_rsi(bars),
        "obv": calc_obv(bars),
        "pattern": detect_candle_pattern(bars),
        "volume_trend": calc_volume_trend(bars),
    }


# ---------------------------------------------------------------------------
# 2. Load trades + harvester snapshots (reuse from v21 backtest)
# ---------------------------------------------------------------------------

def load_real_trades():
    conn = sqlite3.connect(TRADES_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, option_type, strike, contracts, score,
               premium_per_contract, exit_premium, mfe_premium,
               pnl_dollars, pnl_pct, mfe_pnl_pct, exit_reason,
               expiry_date, opened_at, closed_at, duration_minutes
        FROM paper_trades WHERE status='closed' AND parent_trade_id IS NULL ORDER BY id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_contract_ticker(ticker, expiry_date, strike, option_type):
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type == "call" else "P"
    strike_int = int(strike * 1000)
    return "O:{}{}{}{:08d}".format(ticker, date_str, opt_char, strike_int)


def get_snapshots(conn, contract_ticker, after_time):
    return conn.execute("""
        SELECT captured_at, midpoint, bid, ask, underlying_price
        FROM harvest_snapshots
        WHERE contract_ticker = ? AND captured_at >= ?
        ORDER BY captured_at
    """, (contract_ticker, after_time)).fetchall()


# ---------------------------------------------------------------------------
# 3. Simulation — runs exit pipeline with optional candle exhaustion
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    trade_id: int
    ticker: str
    direction: str
    contracts: int
    entry_premium: float
    pnl_dollars: float
    pnl_pct: float
    mfe_pct: float
    exit_reason: str
    duration_min: float
    exhaustion_fired: bool = False
    exhaustion_at_min: float = 0.0
    exhaustion_signals: str = ""


def simulate_with_candles(
    snapshots, entry_premium, signal_time, direction, contracts,
    candles_5m=None, candles_15m=None,
    use_candle_exhaustion=False,
):
    """Simulate exit pipeline, optionally using candle exhaustion to tighten trail.

    Returns SimResult-style tuple: (pnl_dollars, peak, reason, duration, exhaustion_info).
    """
    if not snapshots or entry_premium <= 0:
        return 0.0, entry_premium, "no_data", 0.0, {}

    # Config
    grace_min = 20
    premium_stop_pct = 30.0
    adaptive_activation = 35.0
    trail_active_width = 35.0
    trail_runner_width = 45.0
    trail_moonshot_width = 30.0
    profit_lock_tiers = [(250, 150), (150, 70), (80, 25)]
    theta_bleed_min = 45
    theta_bleed_loss = 30.0
    no_momentum_min = 45
    vol_tighten_factor = 0.7

    peak = entry_premium
    peak_underlying = None
    locked_floor = None
    vol_tighten = False
    underlying_prices = []
    last_new_high = signal_time
    last_elapsed = 0.0
    exhaustion_info = {"fired": False, "at_min": 0, "signals": ""}

    for snap in snapshots:
        cap_str, midpoint, bid, ask, underlying = snap
        price = midpoint
        if price is None or price <= 0:
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue

        captured = datetime.fromisoformat(cap_str)
        if captured.tzinfo and not signal_time.tzinfo:
            captured = captured.replace(tzinfo=None)
        elapsed_min = (captured - signal_time).total_seconds() / 60
        last_elapsed = elapsed_min

        if price > peak:
            peak = price
            last_new_high = captured

        if underlying and underlying > 0:
            underlying_prices.append(underlying)
            if peak_underlying is None or underlying > peak_underlying:
                peak_underlying = underlying

        gain_pct = (price - entry_premium) / entry_premium * 100
        peak_gain = (peak - entry_premium) / entry_premium * 100

        # Profit lock
        for thresh, lock in sorted(profit_lock_tiers, key=lambda x: -x[0]):
            if peak_gain >= thresh:
                locked_floor = lock
                break

        et = captured - timedelta(hours=4)

        # Expiry safety
        mkt_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
        to_close = (mkt_close - et).total_seconds() / 60
        if 0 < to_close <= 10:
            pnl = (price - entry_premium) * contracts * 100
            return pnl, peak, "expiry_safety", elapsed_min, exhaustion_info

        if elapsed_min < grace_min:
            continue

        # Hard stop
        loss = (entry_premium - price) / entry_premium * 100
        if loss >= premium_stop_pct:
            pnl = (price - entry_premium) * contracts * 100
            return pnl, peak, "premium_stop", elapsed_min, exhaustion_info

        # Profit lock
        if locked_floor is not None and gain_pct <= locked_floor:
            pnl = (price - entry_premium) * contracts * 100
            return pnl, peak, "profit_lock", elapsed_min, exhaustion_info

        # Volume peak / candle exhaustion check
        if peak_gain >= 35 and not vol_tighten:
            if use_candle_exhaustion and candles_5m:
                # Convert snapshot time to unix ms for candle lookup
                snap_ts_ms = captured.timestamp() * 1000
                ind_5m = build_indicators_at_time(candles_5m, snap_ts_ms)
                ind_15m = {}
                if candles_15m:
                    ind_15m = build_indicators_at_time(candles_15m, snap_ts_ms)

                candle_data = {
                    "indicators": {
                        "5m": ind_5m,
                        "15m": ind_15m,
                        "1h": {"atr": None, "rsi": None, "obv": None,
                               "pattern": None, "volume_trend": None},
                    }
                }

                is_exhausted, reason = check_exhaustion(
                    candle_data, direction, peak_gain, min_gain_pct=35.0,
                )
                if is_exhausted:
                    vol_tighten = True
                    exhaustion_info = {
                        "fired": True,
                        "at_min": elapsed_min,
                        "signals": reason,
                    }
            else:
                # Fallback: price momentum
                if len(underlying_prices) >= 6:
                    recent = underlying_prices[-6:]
                    first_avg = sum(recent[:3]) / 3
                    second_avg = sum(recent[3:]) / 3
                    if direction in ("call", "bullish"):
                        if second_avg < first_avg * 0.999:
                            vol_tighten = True
                    else:
                        if second_avg > first_avg * 1.001:
                            vol_tighten = True

        # Adaptive trail
        if peak_gain < adaptive_activation:
            width = 100.0  # dormant
        elif peak_gain < 150:
            width = trail_active_width
        elif peak_gain < 400:
            width = trail_runner_width
        else:
            width = trail_moonshot_width

        if vol_tighten and peak_gain >= adaptive_activation:
            width *= vol_tighten_factor

        if peak_gain >= adaptive_activation and peak > 0:
            drop = (peak - price) / peak * 100
            if drop >= width:
                pnl = (price - entry_premium) * contracts * 100
                reason = "trail_tightened" if vol_tighten else "adaptive_trail"
                return pnl, peak, reason, elapsed_min, exhaustion_info

        # Underlying trail
        if (peak_gain >= adaptive_activation and underlying and underlying > 0
                and peak_underlying):
            tiers = [(100.0, 0.0050), (50.0, 0.0040), (15.0, 0.0030), (0.0, 0.0020)]
            u_trail = tiers[-1][1]
            for min_g, pct in tiers:
                if peak_gain >= min_g:
                    u_trail = pct
                    break
            if direction in ("call", "bullish"):
                if underlying < peak_underlying * (1.0 - u_trail):
                    pnl = (price - entry_premium) * contracts * 100
                    return pnl, peak, "underlying_trail", elapsed_min, exhaustion_info

        # Theta bleed
        if elapsed_min >= theta_bleed_min and loss >= theta_bleed_loss:
            pnl = (price - entry_premium) * contracts * 100
            return pnl, peak, "theta_bleed", elapsed_min, exhaustion_info

        # No momentum
        if elapsed_min >= no_momentum_min and gain_pct < 5:
            pnl = (price - entry_premium) * contracts * 100
            return pnl, peak, "no_momentum", elapsed_min, exhaustion_info

        # Time decay
        if et.hour >= 15 and last_new_high:
            lnh = last_new_high
            if captured.tzinfo and not lnh.tzinfo:
                lnh = lnh.replace(tzinfo=captured.tzinfo)
            elif not captured.tzinfo and lnh.tzinfo:
                lnh = lnh.replace(tzinfo=None)
            if (captured - lnh).total_seconds() / 60 >= 10:
                pnl = (price - entry_premium) * contracts * 100
                return pnl, peak, "time_decay", elapsed_min, exhaustion_info

    # Close at last snapshot
    if snapshots:
        last = snapshots[-1]
        lp = last[1] or ((last[2] or 0) + (last[3] or 0)) / 2
        if lp and lp > 0:
            pnl = (lp - entry_premium) * contracts * 100
            return pnl, peak, "market_close", last_elapsed, exhaustion_info

    return 0.0, entry_premium, "no_data", last_elapsed, exhaustion_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true",
                        help="Only download candles, don't run backtest")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, use cached candles only")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key and not args.no_download:
        print("ERROR: No POLYGON_API_KEY found in env or .env")
        sys.exit(1)

    trades = load_real_trades()
    print(f"Loaded {len(trades)} trades")

    # Collect unique (ticker, date) pairs
    ticker_dates = set()
    for t in trades:
        day = t["opened_at"][:10]
        ticker_dates.add((t["ticker"], day))

    print(f"\nNeed candles for {len(ticker_dates)} ticker/date combos")

    # Download candles
    if not args.no_download:
        CANDLE_DIR.mkdir(parents=True, exist_ok=True)
        for i, (ticker, date) in enumerate(sorted(ticker_dates)):
            for tf in ("5", "15"):
                download_candles(ticker, date, api_key, tf)
            # Rate limit: 5 req/min on free tier
            if (i + 1) % 2 == 0:
                print(f"  ... {i+1}/{len(ticker_dates)} done, sleeping 13s for rate limit")
                time.sleep(13)

        print(f"\nDownloaded candles to {CANDLE_DIR}/")

    if args.download_only:
        return

    # Load harvester for premium curves
    harv_conn = sqlite3.connect(HARVESTER_DB)

    # Run simulation
    print("\n" + "=" * 120)
    print("  CANDLE EXHAUSTION BACKTEST — {} REAL TRADES".format(len(trades)))
    print("=" * 120)

    header = (
        f"  {'ID':>3} {'Ticker':<6} {'Dir':<5} {'Qty':>3} "
        f"{'Entry':>6} {'MFE%':>6} "
        f"{'Baseline':>10} {'Candle':>10} {'Delta':>8} "
        f"{'Exhaust?':>8} {'At min':>6} {'Exit Reason':<25} {'Signals'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_baseline = 0.0
    total_candle = 0.0
    improved = 0
    worsened = 0
    no_change = 0
    exhaustion_fired_count = 0

    for trade in trades:
        signal_time = datetime.fromisoformat(trade["opened_at"])
        earlier = (signal_time - timedelta(minutes=2)).isoformat()
        day = trade["opened_at"][:10]
        ticker = trade["ticker"]
        direction = trade["direction"]
        contracts = trade["contracts"]
        entry = trade["premium_per_contract"]

        # Get harvester snapshots
        base_date = datetime.strptime(trade["expiry_date"], "%Y-%m-%d").date()
        snapshots = None
        for delta in range(0, 5):
            try_date = base_date + timedelta(days=delta)
            if try_date.weekday() >= 5:
                continue
            ct = build_contract_ticker(
                ticker, try_date.strftime("%Y-%m-%d"),
                trade["strike"], trade["option_type"],
            )
            snapshots = get_snapshots(harv_conn, ct, earlier)
            if snapshots:
                break
        if not snapshots:
            snapshots = []

        # Load candle data
        candles_5m = []
        candles_15m = []
        f5 = CANDLE_DIR / "{}_{}_5m.json".format(ticker, day)
        f15 = CANDLE_DIR / "{}_{}_15m.json".format(ticker, day)
        if f5.exists():
            with open(f5) as f:
                candles_5m = bars_to_candles(json.load(f))
        if f15.exists():
            with open(f15) as f:
                candles_15m = bars_to_candles(json.load(f))

        mfe_pct = trade["mfe_pnl_pct"] or 0

        if not snapshots:
            # No harvester data — use actual P&L
            actual_pnl = trade["pnl_dollars"] or 0
            total_baseline += actual_pnl
            total_candle += actual_pnl
            no_change += 1
            print(
                f"  {trade['id']:>3} {ticker:<6} {direction:<5} {contracts:>3} "
                f"${entry:>5.2f} {mfe_pct:>5.0f}% "
                f"{'$'+format(actual_pnl, '.2f'):>10} {'$'+format(actual_pnl, '.2f'):>10} "
                f"{'$0.00':>8} {'N/A':>8} {'':>6} {'(no harvester data)':<25}"
            )
            continue

        # Baseline: price-momentum vol-peak only
        pnl_b, peak_b, reason_b, dur_b, _ = simulate_with_candles(
            snapshots, entry, signal_time, direction, contracts,
            candles_5m=None, candles_15m=None,
            use_candle_exhaustion=False,
        )

        # With candle exhaustion
        pnl_c, peak_c, reason_c, dur_c, exh_info = simulate_with_candles(
            snapshots, entry, signal_time, direction, contracts,
            candles_5m=candles_5m, candles_15m=candles_15m,
            use_candle_exhaustion=True,
        )

        delta_pnl = pnl_c - pnl_b
        total_baseline += pnl_b
        total_candle += pnl_c

        if delta_pnl > 1:
            improved += 1
        elif delta_pnl < -1:
            worsened += 1
        else:
            no_change += 1

        fired = exh_info.get("fired", False)
        if fired:
            exhaustion_fired_count += 1

        signals_short = exh_info.get("signals", "")[:50]

        print(
            f"  {trade['id']:>3} {ticker:<6} {direction:<5} {contracts:>3} "
            f"${entry:>5.2f} {mfe_pct:>5.0f}% "
            f"{'$'+format(pnl_b, '.2f'):>10} {'$'+format(pnl_c, '.2f'):>10} "
            f"{'$'+format(delta_pnl, '+.2f'):>8} "
            f"{'YES' if fired else '-':>8} "
            f"{exh_info.get('at_min', 0):>5.0f}m "
            f"{reason_c:<25} {signals_short}"
        )

    harv_conn.close()

    # Summary
    print("\n" + "=" * 120)
    delta_total = total_candle - total_baseline
    print(f"  SUMMARY")
    print(f"  {'Baseline total P&L:':<30} ${total_baseline:>+10.2f}")
    print(f"  {'Candle exhaustion P&L:':<30} ${total_candle:>+10.2f}")
    print(f"  {'Delta:':<30} ${delta_total:>+10.2f}")
    print()
    print(f"  Exhaustion fired:  {exhaustion_fired_count}/{len(trades)} trades")
    print(f"  Improved:          {improved} trades")
    print(f"  Worsened:          {worsened} trades")
    print(f"  No change:         {no_change} trades")

    if improved > 0:
        print(f"\n  Candle exhaustion improves exits by catching reversals via RSI/OBV/patterns")
    if worsened > 0:
        print(f"  {worsened} trades worsened — review if tighten factor (0.7) is too aggressive")

    print("=" * 120)


if __name__ == "__main__":
    main()
