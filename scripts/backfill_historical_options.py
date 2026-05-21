"""Backfill historical options data from Polygon for dates with Discord signals but no harvester data.

Reads trade signals from the legacy and per-owlet DBs, fetches minute-bar data
from Polygon's /v2/aggs endpoint, and inserts rows into the harvester DB
(harvest_contracts + harvest_snapshots tables).

Idempotent: skips (contract_ticker, date) pairs that already have snapshots.

Usage:
    POLYGON_API_KEY=xxx python scripts/backfill_historical_options.py
    POLYGON_API_KEY=xxx python scripts/backfill_historical_options.py --dry-run
"""

import argparse
import asyncio
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
REQUEST_DELAY = 0.22  # ~4.5 req/sec (stay under 5/sec free-tier burst)

# Source DBs containing trade signals
LEGACY_SIGNALS_DB = os.path.join(
    os.path.dirname(__file__), "..", "journal", "raw_messages.db"
)
KODY_SIGNALS_DB = os.path.join(
    os.path.dirname(__file__), "..", "journal", "owlet-kody", "raw_messages.db"
)

# Target DB (harvester)
HARVESTER_DB = os.path.join(
    os.path.dirname(__file__), "..", "journal", "owlet-harvester", "options_data.db"
)

# Auto-detect all signal dates (no more hardcoding)
def _get_all_signal_dates(db_path: str) -> set[str]:
    """Extract all unique signal dates from a DB."""
    if not os.path.exists(db_path):
        return set()
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT DISTINCT DATE(created_at) FROM trade_signals").fetchall()
        conn.close()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()

LEGACY_DATES = _get_all_signal_dates(LEGACY_SIGNALS_DB)
KODY_DATES = _get_all_signal_dates(KODY_SIGNALS_DB)
ALL_SIGNAL_DATES = LEGACY_DATES | KODY_DATES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_contract_ticker(underlying: str, expiry_date: str, strike: float, option_type: str) -> str:
    """Build Polygon-style contract ticker like O:SPY260417C00709000."""
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    opt_char = "C" if option_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{date_str}{opt_char}{strike_int:08d}"


def resolve_expiry(signal_created_at: str) -> str:
    """0DTE = same day as signal (created_at is UTC ISO string)."""
    dt = datetime.fromisoformat(signal_created_at)
    return dt.strftime("%Y-%m-%d")


def load_signals_for_dates(db_path: str, target_dates: set[str]) -> list[dict]:
    """Load trade signals from a DB, filtered to specific dates.

    Returns list of dicts with keys: ticker, direction, strike, expiry,
    option_type, created_at, date.
    """
    if not os.path.exists(db_path):
        print(f"  DB not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    signals = []
    try:
        rows = conn.execute("""
            SELECT ticker, direction, strike, expiry, created_at
            FROM trade_signals
            ORDER BY created_at
        """).fetchall()
    except sqlite3.OperationalError as e:
        print(f"  Error reading {db_path}: {e}")
        conn.close()
        return []

    for ticker, direction, strike, expiry, created_at in rows:
        # Resolve expiry
        if expiry == "0DTE":
            expiry_date = resolve_expiry(created_at)
        else:
            expiry_date = expiry

        # Extract date from created_at (UTC)
        signal_date = created_at[:10]

        if signal_date not in target_dates:
            continue

        # Direction -> option_type
        option_type = "call" if direction.lower() in ("bullish", "call", "long") else "put"

        signals.append({
            "ticker": ticker,
            "strike": float(strike),
            "expiry_date": expiry_date,
            "option_type": option_type,
            "created_at": created_at,
            "date": signal_date,
        })

    conn.close()
    return signals


def get_existing_contract_dates(harvester_db: str) -> set[tuple[str, str]]:
    """Return set of (contract_ticker, date) pairs already in harvester DB."""
    if not os.path.exists(harvester_db):
        return set()

    conn = sqlite3.connect(harvester_db)
    existing = set()
    try:
        rows = conn.execute("""
            SELECT contract_ticker, DATE(captured_at) as snap_date
            FROM harvest_snapshots
            GROUP BY contract_ticker, snap_date
        """).fetchall()
        for contract_ticker, snap_date in rows:
            existing.add((contract_ticker, snap_date))
    except sqlite3.OperationalError:
        pass
    conn.close()
    return existing


def init_harvester_db(db_path: str) -> None:
    """Ensure harvester DB and tables exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS harvest_contracts (
            contract_ticker TEXT PRIMARY KEY,
            underlying TEXT NOT NULL,
            strike REAL NOT NULL,
            expiry_date TEXT NOT NULL,
            option_type TEXT NOT NULL,
            first_seen_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_contracts_underlying ON harvest_contracts(underlying);
        CREATE INDEX IF NOT EXISTS idx_contracts_expiry ON harvest_contracts(expiry_date);

        CREATE TABLE IF NOT EXISTS harvest_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_ticker TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            underlying_price REAL,
            bid REAL,
            ask REAL,
            bid_size INTEGER,
            ask_size INTEGER,
            midpoint REAL,
            last_trade_price REAL,
            last_trade_ts_ns INTEGER,
            day_open REAL,
            day_high REAL,
            day_low REAL,
            day_close REAL,
            day_volume INTEGER,
            day_vwap REAL,
            open_interest INTEGER,
            implied_volatility REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            FOREIGN KEY (contract_ticker) REFERENCES harvest_contracts(contract_ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_captured ON harvest_snapshots(captured_at);
        CREATE INDEX IF NOT EXISTS idx_snapshots_contract_time
            ON harvest_snapshots(contract_ticker, captured_at);
    """)
    conn.close()


# ---------------------------------------------------------------------------
# Polygon API
# ---------------------------------------------------------------------------


async def fetch_option_bars(
    client: httpx.AsyncClient,
    contract_ticker: str,
    date: str,
) -> list[dict] | None:
    """Fetch 1-minute bars for an option contract on a given date.

    Returns list of bar dicts or None on error.
    """
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{contract_ticker}/range/1/minute/"
        f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    await asyncio.sleep(REQUEST_DELAY)
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 429:
            print("    Rate limited -- waiting 60s...")
            await asyncio.sleep(60)
            resp = await client.get(url, timeout=30)
        data = resp.json()
        if data.get("status") == "NOT_AUTHORIZED":
            return None
        return data.get("results") or []
    except Exception as e:
        print(f"    API error for {contract_ticker} on {date}: {e}")
        return None


async def fetch_underlying_bars(
    client: httpx.AsyncClient,
    ticker: str,
    date: str,
) -> dict[int, float]:
    """Fetch 1-minute bars for the underlying and return {timestamp_ms: close_price}.

    Used to populate underlying_price in harvest_snapshots.
    """
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    await asyncio.sleep(REQUEST_DELAY)
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 429:
            await asyncio.sleep(60)
            resp = await client.get(url, timeout=30)
        data = resp.json()
        results = data.get("results") or []
        return {bar["t"]: bar["c"] for bar in results}
    except Exception as e:
        print(f"    API error fetching underlying {ticker} on {date}: {e}")
        return {}


def bars_to_snapshots(
    bars: list[dict],
    contract_ticker: str,
    underlying_prices: dict[int, float],
) -> list[tuple]:
    """Convert Polygon minute bars to harvest_snapshot insert tuples.

    Each bar becomes one snapshot row. We approximate:
    - midpoint = VWAP if available, else (open + close) / 2
    - last_trade_price = close
    - day OHLCV from bar (note: these are per-bar, not running daily aggregates)
    - bid/ask = None (not available from agg bars)
    - underlying_price = closest underlying bar close price
    """
    rows = []
    for bar in bars:
        ts_ms = bar["t"]
        captured_at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

        vwap = bar.get("vw")
        midpoint = vwap if vwap else (bar["o"] + bar["c"]) / 2

        # Find closest underlying price (same timestamp or nearest earlier)
        underlying_price = underlying_prices.get(ts_ms)
        if underlying_price is None:
            # Find nearest earlier bar
            earlier = [t for t in underlying_prices if t <= ts_ms]
            if earlier:
                underlying_price = underlying_prices[max(earlier)]

        rows.append((
            contract_ticker,       # contract_ticker
            captured_at,           # captured_at
            underlying_price,      # underlying_price
            None,                  # bid (not available from bars)
            None,                  # ask
            None,                  # bid_size
            None,                  # ask_size
            round(midpoint, 4),    # midpoint
            bar["c"],              # last_trade_price (bar close)
            ts_ms * 1_000_000,     # last_trade_ts_ns (ms -> ns approximation)
            bar["o"],              # day_open
            bar["h"],              # day_high
            bar["l"],              # day_low
            bar["c"],              # day_close
            bar.get("v"),          # day_volume
            bar.get("vw"),         # day_vwap
            None,                  # open_interest (not in bars)
            None,                  # implied_volatility
            None,                  # delta
            None,                  # gamma
            None,                  # theta
            None,                  # vega
        ))
    return rows


def insert_snapshots(db_path: str, contract_info: dict, snapshot_rows: list[tuple]) -> int:
    """Insert contract metadata + snapshot rows into harvester DB.

    Returns number of snapshots inserted.
    """
    if not snapshot_rows:
        return 0

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    now_iso = datetime.now(timezone.utc).isoformat()

    # Upsert contract metadata
    conn.execute(
        """INSERT OR IGNORE INTO harvest_contracts
           (contract_ticker, underlying, strike, expiry_date, option_type, first_seen_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            contract_info["contract_ticker"],
            contract_info["underlying"],
            contract_info["strike"],
            contract_info["expiry_date"],
            contract_info["option_type"],
            now_iso,
        ),
    )

    # Insert snapshots
    conn.executemany(
        """INSERT INTO harvest_snapshots (
            contract_ticker, captured_at, underlying_price,
            bid, ask, bid_size, ask_size, midpoint,
            last_trade_price, last_trade_ts_ns,
            day_open, day_high, day_low, day_close, day_volume, day_vwap,
            open_interest, implied_volatility, delta, gamma, theta, vega
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        snapshot_rows,
    )
    conn.commit()
    count = len(snapshot_rows)
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_backfill(dry_run: bool = False) -> None:
    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY environment variable is required")
        sys.exit(1)

    # 1. Load signals from both DBs (auto-detect all dates)
    print("Loading trade signals...")
    legacy_signals = load_signals_for_dates(LEGACY_SIGNALS_DB, ALL_SIGNAL_DATES)
    print(f"  Legacy DB: {len(legacy_signals)} signals for dates {sorted(LEGACY_DATES)}")

    kody_signals = load_signals_for_dates(KODY_SIGNALS_DB, ALL_SIGNAL_DATES)
    print(f"  Kody DB:   {len(kody_signals)} signals for dates {sorted(KODY_DATES)}")
    print(f"  All dates: {sorted(ALL_SIGNAL_DATES)}")

    all_signals = legacy_signals + kody_signals
    if not all_signals:
        print("No signals found for the target dates. Nothing to do.")
        return

    # 2. Deduplicate by contract ticker + date
    contracts_to_fetch: dict[tuple[str, str], dict] = {}  # (contract_ticker, date) -> info
    for sig in all_signals:
        contract_ticker = build_contract_ticker(
            sig["ticker"], sig["expiry_date"], sig["strike"], sig["option_type"]
        )
        key = (contract_ticker, sig["date"])
        if key not in contracts_to_fetch:
            contracts_to_fetch[key] = {
                "contract_ticker": contract_ticker,
                "underlying": sig["ticker"],
                "strike": sig["strike"],
                "expiry_date": sig["expiry_date"],
                "option_type": sig["option_type"],
                "date": sig["date"],
            }

    print(f"\nUnique (contract, date) pairs to fetch: {len(contracts_to_fetch)}")

    # 3. Check what already exists in harvester DB
    init_harvester_db(HARVESTER_DB)
    existing = get_existing_contract_dates(HARVESTER_DB)
    to_fetch = {k: v for k, v in contracts_to_fetch.items() if k not in existing}
    skipped = len(contracts_to_fetch) - len(to_fetch)
    if skipped:
        print(f"  Skipping {skipped} already-backfilled (contract, date) pairs")
    print(f"  Remaining to fetch: {len(to_fetch)}")

    if not to_fetch:
        print("\nAll data already backfilled. Nothing to do.")
        return

    if dry_run:
        print("\n--- DRY RUN ---")
        for (ct, d), info in sorted(to_fetch.items(), key=lambda x: (x[0][1], x[0][0])):
            print(f"  Would fetch: {ct} on {d} ({info['underlying']} {info['option_type']} ${info['strike']})")
        print(f"\nTotal API calls needed: ~{len(to_fetch) * 2} "
              f"(1 option bars + 1 underlying bars per contract, with caching)")
        return

    # 4. Fetch from Polygon and insert
    # Cache underlying bars per (ticker, date) to avoid redundant API calls
    underlying_cache: dict[tuple[str, str], dict[int, float]] = {}

    total_snapshots = 0
    errors = 0
    start_time = time.time()

    items = sorted(to_fetch.items(), key=lambda x: (x[0][1], x[0][0]))

    async with httpx.AsyncClient() as client:
        for i, ((contract_ticker, date), info) in enumerate(items):
            underlying = info["underlying"]
            elapsed = time.time() - start_time
            rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
            eta_min = ((len(items) - i) / (i / elapsed * 60)) if elapsed > 0 and i > 0 else 0

            print(
                f"[{i + 1}/{len(items)}] {date} | {contract_ticker} | "
                f"{underlying} {info['option_type']} ${info['strike']:.0f} | "
                f"ETA {eta_min:.1f}m"
            )

            # Fetch underlying bars (cached per ticker+date)
            und_key = (underlying, date)
            if und_key not in underlying_cache:
                underlying_prices = await fetch_underlying_bars(client, underlying, date)
                underlying_cache[und_key] = underlying_prices
                print(f"  underlying bars: {len(underlying_prices)}")
            else:
                underlying_prices = underlying_cache[und_key]
                print(f"  underlying bars: {len(underlying_prices)} (cached)")

            # Fetch option bars
            option_bars = await fetch_option_bars(client, contract_ticker, date)
            if option_bars is None:
                print(f"  NOT_AUTHORIZED -- skipping (date too old for tier)")
                errors += 1
                continue
            if not option_bars:
                print(f"  No bars returned (contract may not have traded)")
                continue

            print(f"  option bars: {len(option_bars)}")

            # Convert to snapshot rows and insert
            snapshot_rows = bars_to_snapshots(option_bars, contract_ticker, underlying_prices)
            count = insert_snapshots(HARVESTER_DB, info, snapshot_rows)
            total_snapshots += count
            print(f"  inserted: {count} snapshots")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  BACKFILL COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Signals processed: {len(all_signals)}")
    print(f"  Unique contracts:  {len(contracts_to_fetch)}")
    print(f"  Skipped (exist):   {skipped}")
    print(f"  Fetched:           {len(to_fetch)}")
    print(f"  Snapshots inserted:{total_snapshots:,}")
    print(f"  Errors:            {errors}")
    print(f"  Time:              {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Database:          {HARVESTER_DB}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical options data from Polygon for signal dates"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without making API calls",
    )
    args = parser.parse_args()
    asyncio.run(run_backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
