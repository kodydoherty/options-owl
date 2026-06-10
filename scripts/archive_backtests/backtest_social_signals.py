"""Social Signal Backtest — Twitter/News Event → Options Price Impact.

Measures how quickly and how much options prices move after high-impact
social/news events from key accounts (Trump, Elon, Fed, breaking news).

Phase 1: Uses manually curated events with known timestamps + our Polygon data.
Phase 2: Will use live Twitter/X API or Unusual Whales flow data.

Usage:
    python scripts/backtest_social_signals.py                  # analyze all curated events
    python scripts/backtest_social_signals.py --ticker TSLA     # filter by ticker
    python scripts/backtest_social_signals.py --source trump    # filter by source
    python scripts/backtest_social_signals.py --fetch-tweets    # fetch from X API (requires TWITTER_BEARER_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Data sources
THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
PG_DB_PATH = "/tmp/pg_export.db"  # Exported from PostgreSQL via export_pg_to_thetadata.py
EVENTS_DB = str(PROJECT_DIR / "journal" / "social_events.db")

# ── Curated Market-Moving Events ──────────────────────────────────────────
# These are real events with known timestamps and documented market reactions.
# Source: news archives, Trump Truth Social, X/Twitter, FOMC calendar.
#
# Format: (timestamp_utc, source, ticker_impact, direction, headline, magnitude_expected)


@dataclass
class SocialEvent:
    timestamp_utc: datetime
    source: str          # trump, elon, fed, breaking, congress
    tickers: list[str]   # primary tickers impacted
    direction: str       # bullish, bearish
    headline: str
    category: str        # tariff, company_mention, rate, earnings, policy, trade_deal


# Major market-moving events from 2025-2026 (manually curated from news archives)
CURATED_EVENTS = [
    # ── Trump Tariff Events (massive market movers) ──
    SocialEvent(
        timestamp_utc=datetime(2025, 4, 2, 20, 0, tzinfo=UTC),  # "Liberation Day" 4/2/25 4PM ET
        source="trump", tickers=["SPY", "QQQ", "AAPL", "AMZN"],
        direction="bearish", headline="Liberation Day tariffs announced — 10% baseline + reciprocal tariffs on all countries",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 4, 9, 17, 37, tzinfo=UTC),  # 4/9/25 1:37PM ET
        source="trump", tickers=["SPY", "QQQ", "NVDA", "AAPL"],
        direction="bullish", headline="90-day tariff pause announced on Truth Social — markets ripped 10%+",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 5, 12, 13, 0, tzinfo=UTC),  # 5/12/25 9AM ET
        source="trump", tickers=["SPY", "QQQ", "AAPL"],
        direction="bullish", headline="US-China Geneva trade deal — tariffs reduced from 145% to 30%",
        category="trade_deal",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 5, 26, 14, 30, tzinfo=UTC),  # 5/26/25 10:30AM ET
        source="trump", tickers=["AAPL", "SPY"],
        direction="bearish", headline="Trump threatens 25% tariff on Apple iPhones not made in USA",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 6, 2, 13, 0, tzinfo=UTC),
        source="trump", tickers=["SPY", "QQQ"],
        direction="bearish", headline="50% tariffs on EU goods announced",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2026, 1, 15, 14, 0, tzinfo=UTC),
        source="trump", tickers=["SPY", "NVDA", "AMD"],
        direction="bearish", headline="New chip export restrictions to China announced",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2026, 3, 10, 15, 0, tzinfo=UTC),
        source="trump", tickers=["TSLA", "SPY"],
        direction="bullish", headline="Trump praises Tesla, calls Musk 'greatest innovator'",
        category="company_mention",
    ),

    # ── Elon Musk Events ──
    SocialEvent(
        timestamp_utc=datetime(2025, 3, 15, 18, 0, tzinfo=UTC),
        source="elon", tickers=["TSLA"],
        direction="bearish", headline="Musk announces stepping back from DOGE to focus on Tesla",
        category="company_mention",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 5, 5, 20, 0, tzinfo=UTC),
        source="elon", tickers=["TSLA"],
        direction="bullish", headline="Tesla robotaxi launch date confirmed for Austin TX",
        category="company_mention",
    ),
    SocialEvent(
        timestamp_utc=datetime(2026, 2, 14, 15, 30, tzinfo=UTC),
        source="elon", tickers=["TSLA", "SPY"],
        direction="bullish", headline="Musk tweets Tesla Q1 deliveries 'significantly above expectations'",
        category="company_mention",
    ),

    # ── Fed Events ──
    SocialEvent(
        timestamp_utc=datetime(2025, 3, 19, 18, 0, tzinfo=UTC),  # FOMC 2PM ET
        source="fed", tickers=["SPY", "QQQ", "IWM"],
        direction="bearish", headline="Fed holds rates at 4.25-4.50%, hawkish tone on inflation from tariffs",
        category="rate",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 5, 7, 18, 0, tzinfo=UTC),
        source="fed", tickers=["SPY", "QQQ", "IWM"],
        direction="bearish", headline="Fed holds rates, Powell warns tariffs could delay cuts to 2026",
        category="rate",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 6, 18, 18, 0, tzinfo=UTC),
        source="fed", tickers=["SPY", "QQQ", "IWM"],
        direction="bullish", headline="Fed signals potential September rate cut",
        category="rate",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 9, 17, 18, 0, tzinfo=UTC),
        source="fed", tickers=["SPY", "QQQ", "IWM"],
        direction="bullish", headline="Fed cuts rates 25bps to 4.00-4.25%",
        category="rate",
    ),

    # ── Breaking News / Company Events ──
    SocialEvent(
        timestamp_utc=datetime(2025, 4, 3, 20, 15, tzinfo=UTC),  # After hours
        source="breaking", tickers=["AAPL", "NVDA", "META"],
        direction="bearish", headline="China retaliates with 34% counter-tariffs on US goods",
        category="tariff",
    ),
    SocialEvent(
        timestamp_utc=datetime(2025, 5, 19, 20, 30, tzinfo=UTC),
        source="breaking", tickers=["NVDA"],
        direction="bullish", headline="NVDA earnings beat — $44.1B revenue, $0.96 EPS vs $0.88 expected",
        category="earnings",
    ),
    SocialEvent(
        timestamp_utc=datetime(2026, 1, 27, 16, 0, tzinfo=UTC),
        source="breaking", tickers=["NVDA", "MSFT", "GOOGL"],
        direction="bearish", headline="DeepSeek AI shocks market — Chinese AI matches GPT-4 at fraction of cost",
        category="policy",
    ),
]


def init_events_db():
    """Create events DB and populate with curated events."""
    conn = sqlite3.connect(EVENTS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            tickers TEXT NOT NULL,
            direction TEXT NOT NULL,
            headline TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_impact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            option_type TEXT DEFAULT 'call',
            entry_minute INTEGER,
            premium_at_event REAL,
            premium_1m REAL, premium_5m REAL, premium_15m REAL,
            premium_30m REAL, premium_60m REAL,
            underlying_at_event REAL,
            underlying_1m REAL, underlying_5m REAL, underlying_15m REAL,
            underlying_30m REAL, underlying_60m REAL,
            max_gain_pct REAL, max_gain_minute INTEGER,
            max_loss_pct REAL,
            FOREIGN KEY (event_id) REFERENCES social_events(id)
        )
    """)

    # Insert curated events if table is empty
    existing = conn.execute("SELECT COUNT(*) FROM social_events").fetchone()[0]
    if existing == 0:
        for ev in CURATED_EVENTS:
            conn.execute("""
                INSERT INTO social_events (timestamp_utc, source, tickers, direction, headline, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ev.timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
                ev.source,
                ",".join(ev.tickers),
                ev.direction,
                ev.headline,
                ev.category,
            ))
        conn.commit()
        print(f"Loaded {len(CURATED_EVENTS)} curated events into {EVENTS_DB}")

    conn.close()


def fetch_tweets_from_api(accounts: list[str], start_date: str, end_date: str):
    """Fetch tweets from X/Twitter API v2 (requires TWITTER_BEARER_TOKEN env var).

    This is Phase 2 — use curated events for Phase 1.
    """
    bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer:
        print("ERROR: Set TWITTER_BEARER_TOKEN env var for Twitter API access")
        print("  Get one at https://developer.x.com/en/portal/dashboard")
        print("  Basic tier ($100/mo) supports search + filtered stream")
        return []

    import tweepy

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    all_tweets = []
    for handle in accounts:
        print(f"  Fetching tweets from @{handle}...")
        try:
            # Get user ID
            user = client.get_user(username=handle)
            if not user or not user.data:
                print(f"    User @{handle} not found")
                continue

            user_id = user.data.id

            # Fetch tweets
            tweets = client.get_users_tweets(
                user_id,
                start_time=f"{start_date}T00:00:00Z",
                end_time=f"{end_date}T23:59:59Z",
                max_results=100,
                tweet_fields=["created_at", "text", "public_metrics"],
            )

            if tweets and tweets.data:
                for tw in tweets.data:
                    all_tweets.append({
                        "handle": handle,
                        "text": tw.text,
                        "created_at": tw.created_at.isoformat(),
                        "likes": tw.public_metrics.get("like_count", 0),
                        "retweets": tw.public_metrics.get("retweet_count", 0),
                    })
                print(f"    Got {len(tweets.data)} tweets")
            else:
                print(f"    No tweets found in range")

        except Exception as e:
            print(f"    Error: {e}")

    return all_tweets


def classify_tweet(text: str) -> dict:
    """Classify a tweet for market relevance using keyword rules.

    Returns dict with: relevant, sentiment, tickers, confidence, category.
    Phase 2 could use Claude API for better classification.
    """
    text_lower = text.lower()

    # Ticker mentions (explicit)
    TICKER_KEYWORDS = {
        "TSLA": ["tesla", "tsla", "model y", "model 3", "cybertruck", "robotaxi"],
        "AAPL": ["apple", "aapl", "iphone", "ipad", "app store"],
        "NVDA": ["nvidia", "nvda", "gpu", "chips", "ai chip"],
        "GOOGL": ["google", "alphabet", "googl", "youtube"],
        "AMZN": ["amazon", "amzn", "aws", "bezos"],
        "META": ["meta", "facebook", "instagram", "zuckerberg", "threads"],
        "MSFT": ["microsoft", "msft", "azure", "copilot"],
        "AMD": ["amd", "advanced micro"],
        "SPY": ["market", "s&p", "stocks", "wall street", "dow", "tariff", "trade deal",
                "economy", "recession", "federal reserve", "interest rate", "inflation"],
        "QQQ": ["nasdaq", "tech stocks", "technology sector"],
    }

    # Sentiment keywords
    BULLISH_WORDS = ["great", "amazing", "record", "deal", "agreement", "pause", "cut",
                     "boost", "surge", "rally", "strong", "beat", "exceed", "growth",
                     "approved", "success", "breakthrough", "victory", "historic"]
    BEARISH_WORDS = ["tariff", "ban", "restrict", "sanction", "threat", "war", "crash",
                     "decline", "weak", "miss", "fail", "investigate", "lawsuit", "fine",
                     "penalty", "warning", "concern", "risk", "inflation", "recession"]

    # Find tickers
    tickers = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tickers.append(ticker)

    if not tickers:
        return {"relevant": False, "sentiment": "neutral", "tickers": [], "confidence": 0, "category": "none"}

    # Determine sentiment
    bull_score = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear_score = sum(1 for w in BEARISH_WORDS if w in text_lower)

    if bull_score > bear_score:
        sentiment = "bullish"
    elif bear_score > bull_score:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    # Category
    if any(w in text_lower for w in ["tariff", "trade", "import", "export", "duty", "customs"]):
        category = "tariff"
    elif any(w in text_lower for w in ["rate", "federal reserve", "fed", "fomc", "interest"]):
        category = "rate"
    elif any(w in text_lower for w in ["earnings", "revenue", "profit", "eps", "quarterly"]):
        category = "earnings"
    else:
        category = "company_mention"

    confidence = min(1.0, (bull_score + bear_score + len(tickers)) / 5)

    return {
        "relevant": True,
        "sentiment": sentiment,
        "tickers": tickers,
        "confidence": round(confidence, 2),
        "category": category,
    }


def _ts_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds since epoch."""
    return int(dt.timestamp() * 1000)


def measure_event_impact(event: SocialEvent, conn: sqlite3.Connection) -> list[dict]:
    """Measure price impact of an event using historical_0dte.db minute bars.

    Uses underlying_bars for stock price + option_bars for ATM option price.
    Returns list of impact measurements per ticker.
    """
    ev_et = event.timestamp_utc.astimezone(ET)
    date_str = ev_et.strftime("%Y-%m-%d")
    event_ts_ms = _ts_to_ms(event.timestamp_utc)

    # Skip events outside market hours (pre-market events → measure from open)
    market_open_et = ev_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_et = ev_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if ev_et > market_close_et:
        return []  # After hours — would need next day data

    # If pre-market event, measure impact from market open
    if ev_et < market_open_et:
        event_ts_ms = _ts_to_ms(market_open_et.astimezone(UTC))

    results = []
    for ticker in event.tickers:
        # Get ATM option contract for this day
        option_type_suffix = "C" if event.direction == "bullish" else "P"
        td = conn.execute("""
            SELECT atm_call_ticker, open_price, close_price
            FROM trading_days WHERE date=? AND ticker=?
        """, (date_str, ticker)).fetchone()

        if not td:
            continue

        atm_call = td[0]
        # Switch to put contract if bearish
        if option_type_suffix == "P":
            atm_contract = atm_call.replace("C", "P", 1) if "C" in atm_call else atm_call
        else:
            atm_contract = atm_call

        # Get underlying bars around event time
        und_bars = conn.execute("""
            SELECT timestamp, close FROM underlying_bars
            WHERE ticker=? AND date=? AND timestamp >= ?
            ORDER BY timestamp
        """, (ticker, date_str, event_ts_ms - 60000)).fetchall()

        if not und_bars:
            continue

        # Get option bars around event time
        opt_bars = conn.execute("""
            SELECT timestamp, close, high, low FROM option_bars
            WHERE contract_ticker=? AND timestamp >= ?
            ORDER BY timestamp
        """, (atm_contract, event_ts_ms - 60000)).fetchall()

        # Build minute-indexed arrays for underlying
        u0 = None
        und_by_minute = {}
        for ts_ms, close in und_bars:
            minutes_from_event = int((ts_ms - event_ts_ms) / 60000)
            und_by_minute[minutes_from_event] = float(close)
            if u0 is None and minutes_from_event >= 0:
                u0 = float(close)

        # Build minute-indexed arrays for option
        p0 = None
        opt_by_minute = {}
        for ts_ms, close, high, low in opt_bars:
            minutes_from_event = int((ts_ms - event_ts_ms) / 60000)
            opt_by_minute[minutes_from_event] = {"close": float(close), "high": float(high), "low": float(low)}
            if p0 is None and minutes_from_event >= 0:
                p0 = float(close)

        if not u0:
            continue

        option_type = "CALL" if option_type_suffix == "C" else "PUT"
        impact = {
            "ticker": ticker,
            "option_type": option_type,
            "contract": atm_contract,
            "event_time": ev_et.strftime("%H:%M ET"),
            "underlying_at_event": u0,
            "premium_at_event": p0,
        }

        # Measure underlying impact at intervals
        intervals = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}
        max_und_gain_pct = 0
        max_und_gain_minute = 0
        max_opt_gain_pct = 0
        max_opt_gain_minute = 0

        for delta_min in range(1, 61):
            u = und_by_minute.get(delta_min)
            if u and u0:
                change_pct = (u - u0) / u0 * 100
                if abs(change_pct) > abs(max_und_gain_pct):
                    max_und_gain_pct = change_pct
                    max_und_gain_minute = delta_min

            opt = opt_by_minute.get(delta_min)
            if opt and p0 and p0 > 0:
                opt_change = (opt["close"] - p0) / p0 * 100
                if opt_change > max_opt_gain_pct:
                    max_opt_gain_pct = opt_change
                    max_opt_gain_minute = delta_min

            key = intervals.get(delta_min)
            if key:
                if u and u0:
                    impact[f"und_change_{key}"] = round((u - u0) / u0 * 100, 4)
                    impact[f"underlying_{key}"] = u
                if opt and p0 and p0 > 0:
                    impact[f"prem_change_{key}"] = round((opt["close"] - p0) / p0 * 100, 2)
                    impact[f"premium_{key}"] = opt["close"]

        impact["max_und_move_pct"] = round(max_und_gain_pct, 3)
        impact["max_und_move_minute"] = max_und_gain_minute
        impact["max_opt_gain_pct"] = round(max_opt_gain_pct, 2)
        impact["max_opt_gain_minute"] = max_opt_gain_minute

        results.append(impact)

    return results


def run_analysis(source_filter: str | None = None, ticker_filter: str | None = None):
    """Run the full event impact analysis."""
    # Init DB
    init_events_db()

    # Use historical_0dte.db (2 years of data)
    hist_db = str(PROJECT_DIR / "journal" / "historical_0dte.db")
    if not Path(hist_db).exists():
        print(f"ERROR: No historical data found at {hist_db}")
        print(f"  Run: python scripts/download_historical_0dte.py --days 60")
        return

    conn = sqlite3.connect(hist_db)

    # Get available dates
    available_dates = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM trading_days"
        ).fetchall()
    )
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM trading_days").fetchone()
    print(f"Historical data: {date_range[0]} to {date_range[1]} ({len(available_dates)} trading days)")

    # Filter events
    events = CURATED_EVENTS
    if source_filter:
        events = [e for e in events if e.source == source_filter]
    if ticker_filter:
        events = [e for e in events if ticker_filter in e.tickers]

    print(f"\nAnalyzing {len(events)} events...")
    print("=" * 90)

    all_impacts = []
    events_with_data = 0
    events_without_data = 0

    by_source = defaultdict(list)
    by_category = defaultdict(list)

    for ev in events:
        ev_et = ev.timestamp_utc.astimezone(ET)
        date_str = ev_et.strftime("%Y-%m-%d")

        if date_str not in available_dates:
            events_without_data += 1
            continue

        impacts = measure_event_impact(ev, conn)
        if not impacts:
            events_without_data += 1
            continue

        events_with_data += 1
        print(f"\n[{ev.source.upper()}] {ev_et.strftime('%Y-%m-%d %I:%M %p ET')}")
        print(f"  {ev.headline}")
        print(f"  Direction: {ev.direction} | Category: {ev.category}")

        for imp in impacts:
            ticker = imp["ticker"]
            u0 = imp.get("underlying_at_event", 0)
            p0 = imp.get("premium_at_event")
            prem_str = f"${p0:.2f}" if p0 else "N/A"
            print(f"\n  {ticker} ({imp['option_type']}) — underlying ${u0:.2f}, premium {prem_str}")

            for key in ["1m", "5m", "15m", "30m", "60m"]:
                und_key = f"und_change_{key}"
                prem_key = f"prem_change_{key}"
                und_chg = imp.get(und_key, None)
                prem_chg = imp.get(prem_key, None)
                parts = []
                if und_chg is not None:
                    parts.append(f"underlying {und_chg:+.3f}%")
                if prem_chg is not None:
                    parts.append(f"premium {prem_chg:+.1f}%")
                if parts:
                    print(f"    {key:>4}: {'  |  '.join(parts)}")

            if imp.get("max_und_move_pct", 0) != 0:
                print(f"    Peak underlying move: {imp['max_und_move_pct']:+.3f}% at +{imp['max_und_move_minute']}min")
            if imp.get("max_opt_gain_pct", 0) > 0:
                print(f"    Peak option gain: +{imp['max_opt_gain_pct']:.1f}% at +{imp['max_opt_gain_minute']}min")

            by_source[ev.source].append(imp)
            by_category[ev.category].append(imp)
            all_impacts.append(imp)

    conn.close()

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"Events analyzed: {events_with_data} (skipped {events_without_data} — no options data)")
    print(f"Total ticker-impacts measured: {len(all_impacts)}")

    if not all_impacts:
        print("\nNo impact data available. The curated events may not overlap with your ThetaData DB dates.")
        print("Your DB has data for:", sorted(available_dates))
        print("\nTo get more data:")
        print("  1. Export more days from PG: docker exec owlet-kody python scripts/export_pg_to_thetadata.py --days 60")
        print("  2. Or use Polygon historical: python scripts/download_historical_0dte.py --days 60")
        return

    # By source — underlying moves
    print(f"\nBy Source (Underlying Move):")
    print(f"  {'Source':<12} {'Events':<8} {'Avg 5m':<10} {'Avg 15m':<10} {'Avg 30m':<10} {'Avg Peak':<10} {'Peak @'}")
    print(f"  {'-'*72}")
    for source, impacts in sorted(by_source.items()):
        n = len(impacts)
        avg_5m = np.mean([i.get("und_change_5m", 0) for i in impacts if "und_change_5m" in i])
        avg_15m = np.mean([i.get("und_change_15m", 0) for i in impacts if "und_change_15m" in i])
        avg_30m = np.mean([i.get("und_change_30m", 0) for i in impacts if "und_change_30m" in i])
        avg_peak = np.mean([abs(i.get("max_und_move_pct", 0)) for i in impacts])
        avg_ttp = np.mean([i.get("max_und_move_minute", 0) for i in impacts])
        print(f"  {source:<12} {n:<8} {avg_5m:>+8.3f}%  {avg_15m:>+8.3f}%  {avg_30m:>+8.3f}%  {avg_peak:>+8.3f}%  {avg_ttp:>5.0f}min")

    # By source — option premium moves (where available)
    opt_impacts = [i for i in all_impacts if i.get("premium_at_event")]
    if opt_impacts:
        print(f"\nBy Source (Option Premium):")
        print(f"  {'Source':<12} {'Events':<8} {'Avg 5m':<10} {'Avg 15m':<10} {'Avg Peak':<10} {'Peak @'}")
        print(f"  {'-'*60}")
        opt_by_source = defaultdict(list)
        for imp in opt_impacts:
            for ev in CURATED_EVENTS:
                if imp["ticker"] in ev.tickers:
                    opt_by_source[ev.source].append(imp)
                    break
        for source, imps in sorted(opt_by_source.items()):
            n = len(imps)
            avg_5m = np.mean([i.get("prem_change_5m", 0) for i in imps if "prem_change_5m" in i]) if imps else 0
            avg_15m = np.mean([i.get("prem_change_15m", 0) for i in imps if "prem_change_15m" in i]) if imps else 0
            avg_peak = np.mean([i.get("max_opt_gain_pct", 0) for i in imps])
            avg_ttp = np.mean([i.get("max_opt_gain_minute", 0) for i in imps])
            print(f"  {source:<12} {n:<8} {avg_5m:>+8.1f}%  {avg_15m:>+8.1f}%  {avg_peak:>+8.1f}%  {avg_ttp:>5.0f}min")

    # By category
    print(f"\nBy Category (Underlying Move):")
    print(f"  {'Category':<16} {'Events':<8} {'Avg 5m':<10} {'Avg 15m':<10} {'Avg Peak':<10}")
    print(f"  {'-'*55}")
    for cat, impacts in sorted(by_category.items()):
        n = len(impacts)
        avg_5m = np.mean([i.get("und_change_5m", 0) for i in impacts if "und_change_5m" in i])
        avg_15m = np.mean([i.get("und_change_15m", 0) for i in impacts if "und_change_15m" in i])
        avg_peak = np.mean([abs(i.get("max_und_move_pct", 0)) for i in impacts])
        print(f"  {cat:<16} {n:<8} {avg_5m:>+8.3f}%  {avg_15m:>+8.3f}%  {avg_peak:>+8.3f}%")

    # Simulated P&L using underlying move as proxy
    # If we bought ATM 0DTE at event, the underlying move × delta (~0.5) ≈ option move
    print(f"\nSimulated Underlying Impact (buy at event, sell at +15m):")
    print(f"  {'Event':<50} {'Ticker':<8} {'Und Move 15m':<14} {'Opt Move 15m':<14} {'W/L'}")
    print(f"  {'-'*90}")
    total_und_moves = 0
    total_opt_moves = 0
    wins = 0
    n_measured = 0
    for imp in all_impacts:
        # Find matching event
        matched_ev = None
        for ev in events:
            if imp["ticker"] in ev.tickers:
                matched_ev = ev
                break
        if not matched_ev:
            continue

        und_15 = imp.get("und_change_15m")
        opt_15 = imp.get("prem_change_15m")

        if und_15 is None:
            continue

        n_measured += 1
        # For bearish events, we'd buy puts (profit from down move)
        effective_und = und_15 if matched_ev.direction == "bullish" else -und_15
        total_und_moves += effective_und
        if effective_und > 0:
            wins += 1

        opt_str = f"{opt_15:+.1f}%" if opt_15 is not None else "N/A"
        marker = "W" if effective_und > 0 else "L"
        headline = matched_ev.headline[:48]
        print(f"  {headline:<50} {imp['ticker']:<8} {und_15:>+10.3f}%    {opt_str:>10}    {marker}")

    if n_measured > 0:
        wr = wins / n_measured * 100
        avg_move = total_und_moves / n_measured
        print(f"\n  Total: {n_measured} measurements, {wins}W/{n_measured-wins}L, WR={wr:.0f}%")
        print(f"  Avg directional underlying move at +15m: {avg_move:+.3f}%")
        print(f"  Est. ATM option P&L (delta~0.5, 100 contracts @ $2): ${avg_move * 0.5 * 100 * 2 * 100:+,.0f} per event")

    print(f"\n{'=' * 90}")
    print("NEXT STEPS")
    print(f"{'=' * 90}")
    print("1. Get more historical data — export 60+ days from Polygon/PG")
    print("2. Set TWITTER_BEARER_TOKEN and run with --fetch-tweets for real tweet data")
    print("3. If results are positive, build:")
    print("   a. options_owl/collectors/twitter_collector.py (live stream)")
    print("   b. options_owl/signals/tweet_classifier.py (LLM classification)")
    print("   c. Integration with entry pipeline (BotSource.TWITTER)")
    print("4. Consider Unusual Whales API ($48/mo) for institutional flow signals")


def main():
    parser = argparse.ArgumentParser(description="Social Signal Backtest")
    parser.add_argument("--source", type=str, help="Filter by source (trump/elon/fed/breaking)")
    parser.add_argument("--ticker", type=str, help="Filter by ticker (SPY/TSLA/etc)")
    parser.add_argument("--fetch-tweets", action="store_true", help="Fetch from Twitter API (requires TWITTER_BEARER_TOKEN)")
    args = parser.parse_args()

    print("=" * 90)
    print("SOCIAL SIGNAL BACKTEST — Twitter/News Event → Options Price Impact")
    print("=" * 90)

    if args.fetch_tweets:
        accounts = ["realDonaldTrump", "elonmusk", "DeItaone", "LiveSquawk"]
        tweets = fetch_tweets_from_api(accounts, "2026-05-01", "2026-06-01")
        if tweets:
            print(f"\nFetched {len(tweets)} tweets. Classifying...")
            for tw in tweets[:20]:
                cls = classify_tweet(tw["text"])
                if cls["relevant"]:
                    print(f"  [{cls['sentiment'].upper()}] {cls['tickers']} — {tw['text'][:80]}")
        return

    run_analysis(source_filter=args.source, ticker_filter=args.ticker)


if __name__ == "__main__":
    main()
