"""Backtest the Thesis Revalidation Gate on historical losers.

For each stop_hit trade, fetches multi-timeframe candles from Polygon at the
time the premium was ~35% down, runs trend/support/reversal analysis, and
compares outcomes:
  - BROKEN verdict → early exit at -35% (saves money vs -50-60% hard stop)
  - INTACT verdict → widen stop to -65% (check if trade recovered or bled more)
  - DEGRADED → keep current -50% stop (no change)

Usage:
    python scripts/backtest_thesis_revalidation.py
"""

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ── All losing trades that hit hard stop (from all bots) ──────────────
# Format: (ticker, option_type, entry_price, strike, premium, exit_premium,
#          pnl_pct, opened_at_str, closed_at_str, contracts)

STOP_HIT_TRADES = [
    # Kody
    ("GOOGL", "call", 334.215, 335.0, 0.8643, 0.3383, -60.86, "2026-04-15T14:54:53", "2026-04-15T15:04:03", 4),
    ("NVDA", "call", 201.015, 202.5, 1.005, 0.3905, -61.14, "2026-04-21T15:57:48", "2026-04-21T16:44:14", 3),
    ("IWM", "call", 279.62, 280.0, 0.6231, 0.2388, -58.62, "2026-04-21T14:31:08", "2026-04-21T14:47:15", 5),
    ("SPY", "put", 704.14, 705.0, 2.0603, 0.4080, -80.20, "2026-04-22T13:27:41", "2026-04-22T13:47:47", 1),
    ("AVGO", "call", 419.43, 420.0, 1.9999, 0.7264, -60.96, "2026-04-22T16:24:40", "2026-04-22T16:55:21", 1),
    # Adam
    ("SPY", "put", 704.14, 705.0, 2.0603, 0.4179, -79.72, "2026-04-22T13:27:41", "2026-04-22T13:47:42", 1),
    ("GOOGL", "call", 339.12, 340.0, 0.7337, 0.2886, -56.72, "2026-04-17T15:37:00", "2026-04-17T15:45:25", 2),
    ("IWM", "call", 279.62, 280.0, 0.6231, 0.2289, -60.34, "2026-04-21T14:31:08", "2026-04-21T14:46:48", 2),
    ("NVDA", "call", 201.015, 202.5, 1.005, 0.3632, -63.86, "2026-04-21T15:57:48", "2026-04-21T16:44:24", 1),
    ("SPY", "call", 710.065, 710.0, 0.7538, 0.2886, -57.75, "2026-04-22T18:06:19", "2026-04-22T19:01:11", 1),
    # Vinny
    ("GOOGL", "call", 339.12, 340.0, 0.7337, 0.2886, -60.67, "2026-04-17T15:37:00", "2026-04-17T15:45:24", 1),
    ("IWM", "call", 279.62, 280.0, 0.6231, 0.2388, -61.68, "2026-04-21T14:31:08", "2026-04-21T14:46:54", 1),
    ("NVDA", "call", 201.64, 202.5, 0.2915, 0.1095, -62.45, "2026-04-22T15:18:39", "2026-04-22T16:16:50", 1),
    # Yank
    ("NVDA", "call", 201.015, 202.5, 1.005, 0.3817, -62.02, "2026-04-21T15:57:48", "2026-04-21T16:44:31", 1),
    ("GOOGL", "call", 339.12, 340.0, 0.7337, 0.2886, -60.67, "2026-04-17T15:37:00", "2026-04-17T15:45:44", 1),
    ("IWM", "call", 279.62, 280.0, 0.6231, 0.2289, -63.27, "2026-04-21T14:31:08", "2026-04-21T14:46:51", 1),
]

# Polygon API key — read from .env
def _get_api_key() -> str:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("POLYGON_API_KEY="):
                    return line.strip().split("=", 1)[1].strip('"').strip("'")
    except FileNotFoundError:
        pass
    import os
    return os.environ.get("POLYGON_API_KEY", "")


def fetch_candles(api_key: str, ticker: str, multiplier: int, timespan: str,
                  from_ts: str, to_ts: str) -> list[dict]:
    """Fetch candle bars from Polygon."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
        f"/{from_ts}/{to_ts}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return data.get("results", [])
    except Exception as e:
        print(f"  ⚠ Candle fetch failed for {ticker} {multiplier}{timespan}: {e}")
        return []


# ── Technical Analysis (pure Python) ──────────────────────────────────

def sma(values: list[float], period: int) -> list[float]:
    """Simple moving average."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(values[i - period + 1:i + 1]) / period)
    return result


def analyze_trend(candles: list[dict], option_type: str) -> float:
    """Score trend alignment from -1 (broken) to +1 (intact).

    For calls: bullish trend = intact. For puts: bearish = intact.
    """
    if len(candles) < 10:
        return 0.0  # not enough data

    closes = [c["c"] for c in candles]
    sma10 = sma(closes, 10)
    sma20 = sma(closes, min(20, len(closes)))

    score = 0.0
    current = closes[-1]

    # Price vs SMA10
    if sma10[-1] is not None:
        if option_type == "call":
            score += 0.3 if current > sma10[-1] else -0.3
        else:
            score += 0.3 if current < sma10[-1] else -0.3

    # SMA10 vs SMA20
    if sma10[-1] is not None and sma20[-1] is not None:
        if option_type == "call":
            score += 0.2 if sma10[-1] > sma20[-1] else -0.2
        else:
            score += 0.2 if sma10[-1] < sma20[-1] else -0.2

    # SMA10 slope (last 5 bars)
    recent_sma = [v for v in sma10[-5:] if v is not None]
    if len(recent_sma) >= 2:
        slope = (recent_sma[-1] - recent_sma[0]) / max(recent_sma[0], 0.01)
        if option_type == "call":
            score += 0.3 if slope > 0 else -0.3
        else:
            score += 0.3 if slope < 0 else -0.3

    # Higher highs / higher lows (last 5 bars)
    highs = [c["h"] for c in candles[-5:]]
    lows = [c["l"] for c in candles[-5:]]
    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    if option_type == "call":
        score += 0.2 if (hh >= 2 and hl >= 2) else -0.2
    else:
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        score += 0.2 if (ll >= 2 and lh >= 2) else -0.2

    return max(-1.0, min(1.0, score))


def find_support_resistance(candles: list[dict]) -> tuple[list[float], list[float]]:
    """Find pivot-based support and resistance levels."""
    supports = []
    resistances = []
    for i in range(1, len(candles) - 1):
        if candles[i]["l"] < candles[i-1]["l"] and candles[i]["l"] < candles[i+1]["l"]:
            supports.append(candles[i]["l"])
        if candles[i]["h"] > candles[i-1]["h"] and candles[i]["h"] > candles[i+1]["h"]:
            resistances.append(candles[i]["h"])
    return supports, resistances


def score_support_resistance(candles: list[dict], option_type: str) -> float:
    """Score support/resistance proximity from -1 to +1."""
    if len(candles) < 5:
        return 0.0

    supports, resistances = find_support_resistance(candles)
    current = candles[-1]["c"]

    if option_type == "call":
        # For calls: near support = good (bounce expected), broke support = bad
        if supports:
            nearest = min(supports, key=lambda s: abs(current - s))
            pct_from_support = (current - nearest) / max(nearest, 0.01) * 100
            if pct_from_support < -0.3:  # broke below support
                return -0.8
            elif pct_from_support < 0.15:  # near support
                return 0.6
            else:
                return 0.1  # above support, neutral
        return 0.0
    else:
        # For puts: near resistance = good, broke resistance = bad
        if resistances:
            nearest = min(resistances, key=lambda r: abs(current - r))
            pct_from_resist = (nearest - current) / max(current, 0.01) * 100
            if pct_from_resist < -0.3:  # broke above resistance
                return -0.8
            elif pct_from_resist < 0.15:  # near resistance
                return 0.6
            else:
                return 0.1
        return 0.0


def detect_reversals(candles: list[dict], option_type: str) -> float:
    """Detect reversal patterns. Returns -1 (strong reversal) to +1 (no reversal)."""
    if len(candles) < 3:
        return 0.0

    score = 0.0

    # Engulfing pattern (last 2 bars)
    prev = candles[-2]
    last = candles[-1]
    prev_body = prev["c"] - prev["o"]
    last_body = last["c"] - last["o"]

    if option_type == "call":
        # Bearish engulfing = bad for calls
        if prev_body > 0 and last_body < 0 and abs(last_body) > abs(prev_body):
            score -= 0.5
        # Bullish engulfing = good for calls
        elif prev_body < 0 and last_body > 0 and abs(last_body) > abs(prev_body):
            score += 0.5
    else:
        # Bullish engulfing = bad for puts
        if prev_body < 0 and last_body > 0 and abs(last_body) > abs(prev_body):
            score -= 0.5
        elif prev_body > 0 and last_body < 0 and abs(last_body) > abs(prev_body):
            score += 0.5

    # 3 consecutive bars against thesis
    bodies = [c["c"] - c["o"] for c in candles[-3:]]
    if option_type == "call":
        if all(b < 0 for b in bodies):  # 3 red bars = bad
            score -= 0.5
    else:
        if all(b > 0 for b in bodies):  # 3 green bars = bad for puts
            score -= 0.5

    # Volume spike on adverse move
    if len(candles) >= 11:
        avg_vol = sum(c["v"] for c in candles[-11:-1]) / 10
        if avg_vol > 0 and candles[-1]["v"] > avg_vol * 2:
            if option_type == "call" and last_body < 0:
                score -= 0.3  # high volume sell-off
            elif option_type == "put" and last_body > 0:
                score -= 0.3  # high volume rally

    return max(-1.0, min(1.0, score))


@dataclass
class ThesisVerdict:
    trend_1m: float
    trend_5m: float
    trend_15m: float
    support: float
    reversal: float
    total_score: float
    verdict: str  # "INTACT", "DEGRADED", "BROKEN"


def evaluate_thesis(candles_1m: list, candles_5m: list, candles_15m: list,
                    option_type: str) -> ThesisVerdict:
    """Run full thesis evaluation."""
    trend_1m = analyze_trend(candles_1m, option_type)
    trend_5m = analyze_trend(candles_5m, option_type)
    trend_15m = analyze_trend(candles_15m, option_type)

    # Use 5-min for support/resistance (most reliable timeframe)
    support = score_support_resistance(candles_5m, option_type)

    # Reversal detection on 1-min (most granular)
    reversal = detect_reversals(candles_1m, option_type)

    # Weighted score
    total = (
        trend_1m * 0.15 +
        trend_5m * 0.25 +
        trend_15m * 0.25 +
        support * 0.15 +
        reversal * 0.20
    )

    if total >= 0.3:
        verdict = "INTACT"
    elif total <= -0.3:
        verdict = "BROKEN"
    else:
        verdict = "DEGRADED"

    return ThesisVerdict(trend_1m, trend_5m, trend_15m, support, reversal, total, verdict)


def fetch_underlying_bars_after_exit(api_key: str, ticker: str, exit_time: str,
                                      minutes: int = 60) -> list[dict]:
    """Fetch 1-min bars of the underlying after the exit to see what happened."""
    exit_dt = datetime.fromisoformat(exit_time)
    from_ms = int(exit_dt.timestamp() * 1000)
    to_ms = int((exit_dt + timedelta(minutes=minutes)).timestamp() * 1000)
    return fetch_candles(api_key, ticker, 1, "minute", str(from_ms), str(to_ms))


def main():
    api_key = _get_api_key()
    if not api_key:
        print("ERROR: No POLYGON_API_KEY found in .env or environment")
        return

    print("=" * 80)
    print("THESIS REVALIDATION BACKTEST — Historical Stop-Loss Losers")
    print("=" * 80)
    print(f"\nTotal stop_hit trades to analyze: {len(STOP_HIT_TRADES)}")
    print(f"Thesis trigger: -35% premium drop (before -50% hard stop)")
    print(f"Verdicts: BROKEN=exit at -35%, DEGRADED=keep -50%, INTACT=widen to -65%\n")

    # Deduplicate by (ticker, opened_at) to avoid re-fetching same candles
    seen = set()
    unique_trades = []
    for t in STOP_HIT_TRADES:
        key = (t[0], t[7][:16])  # ticker + opened_at to minute
        if key not in seen:
            seen.add(key)
            unique_trades.append(t)

    total_actual_loss = 0.0
    total_thesis_loss = 0.0
    results = []

    for trade in unique_trades:
        ticker, opt_type, entry_price, strike, premium, exit_prem, pnl_pct, opened_at, closed_at, contracts = trade

        # Time when premium was ~35% down (approximate: midpoint between open and close)
        open_dt = datetime.fromisoformat(opened_at)
        close_dt = datetime.fromisoformat(closed_at)
        midpoint_dt = open_dt + (close_dt - open_dt) * 0.5  # ~when it crossed -35%

        print(f"\n{'─'*60}")
        print(f"Trade: {ticker} {opt_type.upper()} ${strike} | Entry premium: ${premium:.4f}")
        print(f"  Opened: {opened_at} | Closed: {closed_at}")
        print(f"  Actual exit: ${exit_prem:.4f} ({pnl_pct:.1f}%) | Contracts: {contracts}")

        # Fetch candles up to the trigger point
        from_1m = int((midpoint_dt - timedelta(minutes=30)).timestamp() * 1000)
        from_5m = int((midpoint_dt - timedelta(hours=2)).timestamp() * 1000)
        from_15m = int((midpoint_dt - timedelta(hours=4)).timestamp() * 1000)
        to_ms = int(midpoint_dt.timestamp() * 1000)

        date_str = open_dt.strftime("%Y-%m-%d")

        candles_1m = fetch_candles(api_key, ticker, 1, "minute", str(from_1m), str(to_ms))
        time.sleep(0.25)  # rate limit
        candles_5m = fetch_candles(api_key, ticker, 5, "minute", str(from_5m), str(to_ms))
        time.sleep(0.25)
        candles_15m = fetch_candles(api_key, ticker, 15, "minute", str(from_15m), str(to_ms))
        time.sleep(0.25)

        print(f"  Candles: 1m={len(candles_1m)}, 5m={len(candles_5m)}, 15m={len(candles_15m)}")

        if not candles_1m and not candles_5m:
            print(f"  ⚠ No candle data — skipping")
            continue

        verdict = evaluate_thesis(candles_1m, candles_5m, candles_15m, opt_type)
        print(f"  Thesis: trend_1m={verdict.trend_1m:+.2f} trend_5m={verdict.trend_5m:+.2f} "
              f"trend_15m={verdict.trend_15m:+.2f}")
        print(f"          support={verdict.support:+.2f} reversal={verdict.reversal:+.2f}")
        print(f"  Score: {verdict.total_score:+.3f} → {verdict.verdict}")

        # Calculate P&L under different scenarios
        actual_loss = (exit_prem - premium) * 100 * contracts
        early_exit_prem = premium * 0.65  # exit at -35%
        early_exit_loss = (early_exit_prem - premium) * 100 * contracts
        widened_loss = (premium * 0.35 - premium) * 100 * contracts  # -65% worst case

        if verdict.verdict == "BROKEN":
            thesis_loss = early_exit_loss  # exit at -35% instead of actual
            action = f"EXIT EARLY at -35% (${early_exit_prem:.4f})"
            savings = actual_loss - thesis_loss
        elif verdict.verdict == "INTACT":
            # Would have widened to -65%. Need to check: did the trade recover
            # or keep bleeding? Check underlying bars after original exit.
            post_bars = fetch_underlying_bars_after_exit(api_key, ticker, closed_at, 30)
            time.sleep(0.25)

            if post_bars and len(post_bars) > 5:
                # Check if underlying moved favorably after exit
                exit_underlying = entry_price  # approximate
                post_highs = [b["h"] for b in post_bars]
                post_lows = [b["l"] for b in post_bars]

                if opt_type == "call":
                    max_recovery = max(post_highs) if post_highs else exit_underlying
                    recovered = max_recovery > exit_underlying * 1.002  # moved up 0.2%+
                else:
                    min_recovery = min(post_lows) if post_lows else exit_underlying
                    recovered = min_recovery < exit_underlying * 0.998

                if recovered:
                    # Trade could have recovered — widening would have helped
                    # Estimate: recovered to -20% instead of -50%
                    thesis_loss = (premium * 0.80 - premium) * 100 * contracts
                    action = f"WIDEN → recovered (est. -20% vs actual {pnl_pct:.0f}%)"
                    savings = actual_loss - thesis_loss
                else:
                    # Didn't recover — widening would have cost more
                    thesis_loss = widened_loss
                    action = f"WIDEN → bled to -65% (worse than actual {pnl_pct:.0f}%)"
                    savings = actual_loss - thesis_loss
            else:
                thesis_loss = actual_loss  # can't determine, assume same
                action = "WIDEN (no post-exit data)"
                savings = 0
        else:
            thesis_loss = actual_loss  # DEGRADED = no change
            action = "NO CHANGE (keep -50% stop)"
            savings = 0

        print(f"  Action: {action}")
        print(f"  Actual loss: ${actual_loss:+.2f} | Thesis loss: ${thesis_loss:+.2f} | Savings: ${savings:+.2f}")

        total_actual_loss += actual_loss
        total_thesis_loss += thesis_loss
        results.append({
            "ticker": ticker,
            "opt_type": opt_type,
            "verdict": verdict.verdict,
            "score": verdict.total_score,
            "actual_loss": actual_loss,
            "thesis_loss": thesis_loss,
            "savings": savings,
            "contracts": contracts,
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Unique trades analyzed: {len(results)}")

    broken = [r for r in results if r["verdict"] == "BROKEN"]
    intact = [r for r in results if r["verdict"] == "INTACT"]
    degraded = [r for r in results if r["verdict"] == "DEGRADED"]

    print(f"\nVerdicts: BROKEN={len(broken)}, INTACT={len(intact)}, DEGRADED={len(degraded)}")
    print(f"\nTotal actual loss:  ${total_actual_loss:+,.2f}")
    print(f"Total thesis loss:  ${total_thesis_loss:+,.2f}")
    print(f"Net savings:        ${total_actual_loss - total_thesis_loss:+,.2f}")

    if broken:
        broken_savings = sum(r["savings"] for r in broken)
        print(f"\nBROKEN trades (early exit at -35%):")
        print(f"  Count: {len(broken)}")
        print(f"  Savings: ${broken_savings:+,.2f}")
        for r in broken:
            print(f"    {r['ticker']} {r['opt_type']}: actual ${r['actual_loss']:+.2f} → thesis ${r['thesis_loss']:+.2f} (saved ${r['savings']:+.2f})")

    if intact:
        intact_delta = sum(r["savings"] for r in intact)
        print(f"\nINTACT trades (widened stop):")
        print(f"  Count: {len(intact)}")
        print(f"  Net effect: ${intact_delta:+,.2f}")
        for r in intact:
            print(f"    {r['ticker']} {r['opt_type']}: actual ${r['actual_loss']:+.2f} → thesis ${r['thesis_loss']:+.2f} (delta ${r['savings']:+.2f})")


if __name__ == "__main__":
    main()
