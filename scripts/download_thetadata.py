"""Download historical options + stock data from ThetaData into SQLite.

Populates the historical_0dte.db with:
  - 1-min option OHLC bars (open/high/low/close/volume/vwap)
  - 1-min option quotes (bid/ask at each minute)
  - 1-min option greeks (IV, delta, theta, vega, underlying_price)
  - 1-min stock OHLC bars (for underlying)

ThetaData Options Standard ($80/month):
  - Tick-level data, 8 years history
  - Option chain snapshots, NBBO quotes

Usage:
    python scripts/download_thetadata.py                          # last 30 days, all tickers
    python scripts/download_thetadata.py --days 90                # last 90 days
    python scripts/download_thetadata.py --start 2024-01-01       # from specific date
    python scripts/download_thetadata.py --ticker SPY             # single ticker
    python scripts/download_thetadata.py --ticker SPY --ohlc-only # skip greeks (faster)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DB_PATH = str(PROJECT_DIR / "journal" / "thetadata_options.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

# ThetaData credentials — read from env / .env only. No hardcoded fallback.
# NEVER log or print the credential values.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass  # env vars must be set directly if python-dotenv is unavailable

THETADATA_EMAIL = os.getenv("THETADATA_EMAIL", "")
THETADATA_PASSWORD = os.getenv("THETADATA_PASSWORD", "")


def _require_thetadata_creds() -> None:
    """Abort with a clear message if credentials are missing (values never logged)."""
    missing = [
        name
        for name, value in (
            ("THETADATA_EMAIL", THETADATA_EMAIL),
            ("THETADATA_PASSWORD", THETADATA_PASSWORD),
        )
        if not value
    ]
    if missing:
        sys.exit(
            f"Missing ThetaData credentials: {', '.join(missing)}. "
            "Set them in the environment or in .env — they are no longer hardcoded."
        )


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS option_ohlc (
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            right TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            vwap REAL,
            PRIMARY KEY (ticker, expiration, strike, right, timestamp)
        );

        CREATE TABLE IF NOT EXISTS option_quotes (
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            right TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bid REAL,
            ask REAL,
            bid_size INTEGER,
            ask_size INTEGER,
            PRIMARY KEY (ticker, expiration, strike, right, timestamp)
        );

        CREATE TABLE IF NOT EXISTS option_greeks (
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            right TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bid REAL,
            ask REAL,
            delta REAL,
            theta REAL,
            vega REAL,
            implied_vol REAL,
            underlying_price REAL,
            PRIMARY KEY (ticker, expiration, strike, right, timestamp)
        );

        CREATE TABLE IF NOT EXISTS stock_ohlc (
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            vwap REAL,
            PRIMARY KEY (ticker, timestamp)
        );

        CREATE TABLE IF NOT EXISTS download_log (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            data_type TEXT NOT NULL,
            rows_downloaded INTEGER,
            downloaded_at TEXT NOT NULL,
            PRIMARY KEY (ticker, date, data_type)
        );

        CREATE INDEX IF NOT EXISTS idx_option_ohlc_date
            ON option_ohlc(ticker, expiration);
        CREATE INDEX IF NOT EXISTS idx_option_greeks_date
            ON option_greeks(ticker, expiration);
        CREATE INDEX IF NOT EXISTS idx_stock_ohlc_date
            ON stock_ohlc(ticker, timestamp);
    """)
    conn.commit()
    return conn


def already_downloaded(conn: sqlite3.Connection, ticker: str, dt: str, data_type: str) -> bool:
    row = conn.execute(
        "SELECT rows_downloaded FROM download_log WHERE ticker=? AND date=? AND data_type=?",
        (ticker, dt, data_type),
    ).fetchone()
    return row is not None and row[0] > 0


def log_download(conn: sqlite3.Connection, ticker: str, dt: str, data_type: str, rows: int):
    conn.execute(
        "INSERT OR REPLACE INTO download_log (ticker, date, data_type, rows_downloaded, downloaded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (ticker, dt, data_type, rows, datetime.utcnow().isoformat()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# ThetaData download functions
# ---------------------------------------------------------------------------

def get_trading_days(client, start: date, end: date) -> list[date]:
    """Get list of market-open dates in range."""
    days = []
    current = start
    while current <= end:
        try:
            result = client.calendar_on_date(current)
            if len(result) > 0:
                is_open = result["is_open"][0]
                if is_open:
                    days.append(current)
        except Exception:
            pass  # skip — probably weekend
        current += timedelta(days=1)
    return days


def get_0dte_expiry(client, ticker: str, trade_date: date) -> date | None:
    """Find the 0DTE expiry for this ticker on this date.

    Returns the expiry date if a same-day or near-term contract exists.
    """
    try:
        exps = client.option_list_expirations([ticker])
        exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in exps["expiration"].to_list()]

        # Try same-day first (true 0DTE)
        if trade_date in exp_dates:
            return trade_date

        # Try next 1-3 business days
        for delta in range(1, 6):
            candidate = trade_date + timedelta(days=delta)
            if candidate in exp_dates:
                return candidate

        return None
    except Exception as exc:
        print(f"    Error listing expirations for {ticker}: {exc}")
        return None


def find_atm_strike(client, ticker: str, expiry: date, trade_date: date) -> float | None:
    """Find ATM strike using greeks data (has underlying_price).

    Returns the strike nearest to the OPEN price (first underlying_price of the day).
    On big move days, the open and close can be far apart — using the open ensures
    we download the strikes the bot would see at market open.
    """
    try:
        strikes_df = client.option_list_strikes([ticker], expiration=expiry)
        if len(strikes_df) == 0:
            return None
        _all = sorted(strikes_df["strike"].to_list())
        # Drop adjusted/non-standard strikes (e.g. 408.78 from special divs) that
        # ThetaData lists but rejects with "Rounding necessary". Keep $0.50 multiples.
        strikes = [s for s in _all if abs(s * 2 - round(s * 2)) < 1e-6] or _all

        # Try to get underlying price from a quick greeks query on the middle strike
        mid_strike = strikes[len(strikes) // 2]
        try:
            greeks = client.option_history_greeks_first_order(
                symbol=ticker,
                expiration=expiry,
                interval="1m",
                date=trade_date,
                strike=str(int(mid_strike)) if mid_strike == int(mid_strike) else str(mid_strike),
                right="call",
            )
            if len(greeks) > 0:
                prices = [p for p in greeks["underlying_price"].to_list() if p and p > 0]
                if prices:
                    # Use FIRST price (open) so we capture ATM at market open.
                    # On big down days, close can be $6+ lower — using close
                    # causes us to miss the ATM strikes the bot trades at open.
                    open_price = prices[0]
                    best = min(strikes, key=lambda s: abs(s - open_price))
                    return best
        except Exception:
            pass

        # Fallback: use middle of strike range
        return mid_strike
    except Exception as exc:
        print(f"    Error finding ATM strike for {ticker}: {exc}")
        return None


MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Session errors that require a full reconnect (ThetaData allows only 1 session)
_SESSION_ERROR_PATTERNS = [
    "UNAUTHENTICATED",
    "Invalid session",
    "session ID",
    "more than one connection",
]


class SessionExpiredError(Exception):
    """Raised when ThetaData session is invalid and needs reconnection."""
    pass


def _is_session_error(exc_str: str) -> bool:
    """Check if the error indicates a dead/duplicate session."""
    exc_lower = exc_str.lower()
    return any(pat.lower() in exc_lower for pat in _SESSION_ERROR_PATTERNS)


def _api_call_with_retry(fn, description: str, retries: int = MAX_RETRIES) -> object | None:
    """Call a ThetaData API function with retry and backoff.

    Raises SessionExpiredError if the session is invalid (caller must reconnect).
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            exc_str = str(exc)
            if "No data found" in exc_str or "no data" in exc_str.lower():
                return None  # expected — no data for this contract
            if _is_session_error(exc_str):
                _flush(f"        {description}: SESSION ERROR — {exc_str[:150]}")
                raise SessionExpiredError(exc_str) from exc
            if attempt < retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                _flush(f"        {description} failed (attempt {attempt+1}/{retries}): {exc_str[:100]}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                _flush(f"        {description} FAILED after {retries} attempts: {exc_str[:150]}")
                return None
    return None


def download_option_data(
    client,
    conn: sqlite3.Connection,
    ticker: str,
    trade_date: date,
    expiry: date,
    strike: float,
    skip_greeks: bool = False,
) -> int:
    """Download 1-min option OHLC + quotes + greeks for a specific contract.

    Returns total rows downloaded. Retries on transient failures.
    """
    total_rows = 0
    strike_str = str(int(strike)) if strike == int(strike) else str(strike)

    for right in ["call", "put"]:
        # --- OHLC ---
        log_key = f"ohlc_{right}_{strike}"
        if not already_downloaded(conn, ticker, str(trade_date), log_key):
            df = _api_call_with_retry(
                lambda: client.option_history_ohlc(
                    symbol=ticker, expiration=expiry, interval="1m",
                    date=trade_date, strike=strike_str, right=right,
                ),
                f"OHLC {ticker} {right} {strike} {trade_date}",
            )
            if df is not None and len(df) > 0:
                rows = _insert_option_ohlc(conn, df, ticker)
                log_download(conn, ticker, str(trade_date), log_key, rows)
                total_rows += rows

        # --- Quotes (bid/ask) ---
        log_key = f"quote_{right}_{strike}"
        if not already_downloaded(conn, ticker, str(trade_date), log_key):
            df = _api_call_with_retry(
                lambda: client.option_history_quote(
                    symbol=ticker, expiration=expiry, interval="1m",
                    date=trade_date, strike=strike_str, right=right,
                ),
                f"Quote {ticker} {right} {strike} {trade_date}",
            )
            if df is not None and len(df) > 0:
                rows = _insert_option_quotes(conn, df, ticker)
                log_download(conn, ticker, str(trade_date), log_key, rows)
                total_rows += rows

        # --- Greeks (IV, delta, theta, vega + underlying_price) ---
        if not skip_greeks:
            log_key = f"greeks_{right}_{strike}"
            if not already_downloaded(conn, ticker, str(trade_date), log_key):
                df = _api_call_with_retry(
                    lambda: client.option_history_greeks_first_order(
                        symbol=ticker, expiration=expiry, interval="1m",
                        date=trade_date, strike=strike_str, right=right,
                    ),
                    f"Greeks {ticker} {right} {strike} {trade_date}",
                )
                if df is not None and len(df) > 0:
                    rows = _insert_option_greeks(conn, df, ticker)
                    log_download(conn, ticker, str(trade_date), log_key, rows)
                    total_rows += rows

        time.sleep(0.05)  # rate limiting

    return total_rows


def extract_stock_from_greeks(
    conn: sqlite3.Connection,
    ticker: str,
    trade_date: date,
) -> int:
    """Extract underlying stock prices from option_greeks table (no paid Stock subscription needed).

    The greeks endpoint includes underlying_price at each timestamp.
    We aggregate call+put greeks rows to build a stock price series.
    """
    log_key = "stock_from_greeks"
    if already_downloaded(conn, ticker, str(trade_date), log_key):
        return 0

    rows = conn.execute(
        """SELECT DISTINCT timestamp, underlying_price FROM option_greeks
           WHERE ticker = ? AND timestamp LIKE ?
           AND underlying_price IS NOT NULL AND underlying_price > 0
           ORDER BY timestamp""",
        (ticker, f"{trade_date}%"),
    ).fetchall()

    if not rows:
        return 0

    inserted = 0
    for ts, price in rows:
        conn.execute(
            "INSERT OR IGNORE INTO stock_ohlc "
            "(ticker, timestamp, open, high, low, close, volume, vwap) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (ticker, ts, price, price, price, price, price),
        )
        inserted += 1
    conn.commit()
    log_download(conn, ticker, str(trade_date), log_key, inserted)
    return inserted


# ---------------------------------------------------------------------------
# DB insertion helpers
# ---------------------------------------------------------------------------

def _insert_option_ohlc(conn: sqlite3.Connection, df, ticker: str) -> int:
    if len(df) == 0:
        return 0
    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    conn.executemany(
        "INSERT OR IGNORE INTO option_ohlc "
        "(ticker, expiration, strike, right, timestamp, open, high, low, close, volume, vwap) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, r.get("expiration", ""), r.get("strike", 0), r.get("right", ""),
          str(r.get("timestamp", "")), r.get("open"), r.get("high"),
          r.get("low"), r.get("close"), r.get("volume", 0), r.get("vwap"))
         for r in rows],
    )
    conn.commit()
    return len(rows)


def _insert_option_quotes(conn: sqlite3.Connection, df, ticker: str) -> int:
    if len(df) == 0:
        return 0
    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    conn.executemany(
        "INSERT OR IGNORE INTO option_quotes "
        "(ticker, expiration, strike, right, timestamp, bid, ask, bid_size, ask_size) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, r.get("expiration", ""), r.get("strike", 0), r.get("right", ""),
          str(r.get("timestamp", "")), r.get("bid"), r.get("ask"),
          r.get("bid_size", 0), r.get("ask_size", 0))
         for r in rows],
    )
    conn.commit()
    return len(rows)


def _insert_option_greeks(conn: sqlite3.Connection, df, ticker: str) -> int:
    if len(df) == 0:
        return 0
    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    conn.executemany(
        "INSERT OR IGNORE INTO option_greeks "
        "(ticker, expiration, strike, right, timestamp, bid, ask, delta, theta, vega, "
        "implied_vol, underlying_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, r.get("expiration", ""), r.get("strike", 0), r.get("right", ""),
          str(r.get("timestamp", "")), r.get("bid"), r.get("ask"),
          r.get("delta"), r.get("theta"), r.get("vega"),
          r.get("implied_vol"), r.get("underlying_price"))
         for r in rows],
    )
    conn.commit()
    return len(rows)


def _insert_stock_ohlc(conn: sqlite3.Connection, df, ticker: str) -> int:
    if len(df) == 0:
        return 0
    rows = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
    conn.executemany(
        "INSERT OR IGNORE INTO stock_ohlc "
        "(ticker, timestamp, open, high, low, close, volume, vwap) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, str(r.get("timestamp", "")), r.get("open"), r.get("high"),
          r.get("low"), r.get("close"), r.get("volume", 0), r.get("vwap"))
         for r in rows],
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def _flush(msg: str) -> None:
    """Print and flush immediately (for background/nohup execution)."""
    print(msg, flush=True)


def get_otm_strikes(
    client,
    ticker: str,
    expiry: date,
    atm_strike: float,
    trade_date: date,
    n_otm: int = 2,
    n_otm_below: int | None = None,
) -> list[float]:
    """Get N OTM strikes above and below ATM for both calls and puts.

    Args:
        n_otm: strikes above ATM (for CALLs)
        n_otm_below: strikes below ATM (for PUTs). Defaults to n_otm if not set.
                     Use a larger value to capture crash-day PUT opportunities.

    Returns list of strikes to download (ATM + OTM).
    """
    if n_otm_below is None:
        n_otm_below = n_otm
    try:
        strikes_df = client.option_list_strikes([ticker], expiration=expiry)
        if len(strikes_df) == 0:
            return [atm_strike]
        _all = sorted(strikes_df["strike"].to_list())
        # Keep only standard ($0.50-multiple) strikes — adjusted strikes (e.g. 408.78)
        # are rejected by ThetaData with "Rounding necessary".
        all_strikes = [s for s in _all if abs(s * 2 - round(s * 2)) < 1e-6] or _all

        # Find ATM index
        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))

        # Asymmetric: more strikes below ATM for PUT coverage on crash days
        start = max(0, atm_idx - n_otm_below)
        end = min(len(all_strikes), atm_idx + n_otm + 1)
        return all_strikes[start:end]
    except Exception:
        return [atm_strike]


def _reconnect_client(old_client) -> object | None:
    """Tear down old ThetaData client and create a fresh session.

    ThetaData only allows one concurrent session per subscription.
    If the old session is stale, we must fully disconnect before reconnecting.
    """
    from thetadata import ThetaClient

    # Drop the old client reference and wait for server to release the session
    if old_client is not None:
        del old_client
        # Give ThetaData server time to release the session
        time.sleep(5)

    for attempt in range(3):
        try:
            _flush(f"    Reconnect attempt {attempt + 1}/3...")
            client = ThetaClient(email=THETADATA_EMAIL, password=THETADATA_PASSWORD)
            _flush("    Reconnected successfully!")
            return client
        except Exception as exc:
            exc_str = str(exc)
            wait = 10 * (attempt + 1)
            _flush(f"    Reconnect failed: {exc_str[:100]}. Waiting {wait}s...")
            time.sleep(wait)

    _flush("    ALL RECONNECT ATTEMPTS FAILED.")
    _flush("    Possible causes:")
    _flush("      1. Another download_thetadata.py process is running (ThetaData allows only 1 session)")
    _flush("      2. ThetaData Terminal app is open (also consumes the session)")
    _flush("      3. A previous process didn't close cleanly (session may take ~60s to expire)")
    _flush("    Fix: kill any other ThetaData processes and wait 60s, then re-run.")
    return None


def run_download(
    tickers: list[str],
    start_date: date,
    end_date: date,
    db_path: str,
    skip_greeks: bool = False,
    otm_strikes: int = 2,
    otm_below: int | None = None,
    batch_size: int = 5,
) -> None:
    from thetadata import ThetaClient

    _require_thetadata_creds()
    otm_below_actual = otm_below if otm_below is not None else otm_strikes
    conn = init_db(db_path)

    # Get trading days using simple weekday check (skip weekends)
    all_dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            all_dates.append(current)
        current += timedelta(days=1)

    _flush(f"\n{'='*70}")
    _flush("THETADATA DOWNLOAD")
    _flush(f"{'='*70}")
    _flush(f"Period:     {start_date} → {end_date} ({len(all_dates)} weekdays)")
    _flush(f"Tickers:    {', '.join(tickers)}")
    _flush(f"OTM depth:  {otm_strikes} above / {otm_below_actual} below ATM")
    _flush(f"DB:         {db_path}")
    _flush(f"Greeks:     {'skip' if skip_greeks else 'include'}")
    _flush(f"Batch size: {batch_size} days")
    _flush(f"{'='*70}\n")

    total_rows = 0
    total_errors = 0
    client = None  # lazy connect

    # Process in batches for resilience
    for batch_start in range(0, len(all_dates), batch_size):
        batch = all_dates[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(all_dates) + batch_size - 1) // batch_size

        _flush(f"\n--- BATCH {batch_num}/{total_batches} ({batch[0]} to {batch[-1]}) ---")

        # (Re)connect at start of each batch for resilience
        if client is None:
            _flush("Connecting to ThetaData...")
            try:
                client = ThetaClient(email=THETADATA_EMAIL, password=THETADATA_PASSWORD)
                _flush("Connected!")
            except Exception as exc:
                _flush(f"CONNECTION FAILED: {exc}. Retrying in 30s...")
                time.sleep(30)
                try:
                    client = ThetaClient(email=THETADATA_EMAIL, password=THETADATA_PASSWORD)
                    _flush("Connected on retry!")
                except Exception as exc2:
                    _flush(f"CONNECTION FAILED AGAIN: {exc2}. Skipping batch.")
                    total_errors += 1
                    continue

        # Cache expirations per ticker (expensive to fetch)
        exp_cache: dict[str, list[date]] = {}
        batch_rows = 0

        for date_idx, trade_date in enumerate(batch):
            day_rows = 0
            global_idx = batch_start + date_idx + 1
            _flush(f"[{global_idx}/{len(all_dates)}] {trade_date}:")

            for ticker in tickers:
                try:
                    # Get expirations (cached per batch)
                    if ticker not in exp_cache:
                        result = _api_call_with_retry(
                            lambda: client.option_list_expirations([ticker]),
                            f"Expirations {ticker}",
                        )
                        if result is not None and len(result) > 0:
                            exp_cache[ticker] = [
                                datetime.strptime(e, "%Y-%m-%d").date()
                                for e in result["expiration"].to_list()
                            ]
                        else:
                            exp_cache[ticker] = []
                        time.sleep(0.1)

                    if not exp_cache.get(ticker):
                        continue

                    # Find 0DTE or nearest expiry
                    exp_dates = exp_cache[ticker]
                    expiry = None
                    if trade_date in exp_dates:
                        expiry = trade_date
                    else:
                        future = [e for e in exp_dates if e >= trade_date]
                        if future:
                            expiry = min(future)

                    if expiry is None:
                        continue

                    # Find ATM strike
                    atm_strike = find_atm_strike(client, ticker, expiry, trade_date)
                    if atm_strike is None:
                        continue

                    dte = (expiry - trade_date).days

                    # Get ATM + OTM strikes
                    if otm_strikes > 0:
                        strikes = get_otm_strikes(client, ticker, expiry, atm_strike, trade_date, otm_strikes, n_otm_below=otm_below_actual)
                    else:
                        strikes = [atm_strike]

                    ticker_rows = 0
                    for strike in strikes:
                        rows = download_option_data(client, conn, ticker, trade_date, expiry, strike, skip_greeks)
                        ticker_rows += rows

                    # Extract stock prices from greeks data
                    if not skip_greeks:
                        stock_rows = extract_stock_from_greeks(conn, ticker, trade_date)
                        ticker_rows += stock_rows

                    if ticker_rows > 0:
                        _flush(f"  {ticker}: {ticker_rows} rows ({len(strikes)} strikes [{strikes[0]}-{strikes[-1]}], exp={expiry}, DTE={dte})")
                    day_rows += ticker_rows

                except SessionExpiredError:
                    _flush("  SESSION EXPIRED — reconnecting...")
                    total_errors += 1
                    client = _reconnect_client(client)
                    if client is None:
                        _flush("  RECONNECT FAILED — skipping rest of batch")
                        break
                    # Clear exp cache (fetched with dead session, may be stale)
                    exp_cache.clear()
                    _flush(f"  Reconnected! Retrying {ticker}...")
                    # Retry this ticker once after reconnect
                    try:
                        result = _api_call_with_retry(
                            lambda: client.option_list_expirations([ticker]),
                            f"Expirations {ticker} (retry)",
                        )
                        if result is not None and len(result) > 0:
                            exp_cache[ticker] = [
                                datetime.strptime(e, "%Y-%m-%d").date()
                                for e in result["expiration"].to_list()
                            ]
                        _flush(f"  {ticker}: retry OK after reconnect")
                    except SessionExpiredError:
                        _flush(f"  {ticker}: STILL FAILING after reconnect — another process may have the session")
                        _flush("  CHECK: is another download_thetadata.py running? ThetaData allows only 1 session.")
                        client = None
                        break

                except Exception as exc:
                    _flush(f"  {ticker}: UNEXPECTED ERROR: {str(exc)[:150]}")
                    total_errors += 1
                    # Reconnect on next ticker if connection seems dead
                    if "connection" in str(exc).lower() or "timeout" in str(exc).lower() or _is_session_error(str(exc)):
                        _flush("  Connection may be stale — reconnecting...")
                        client = _reconnect_client(client)
                        if client is None:
                            _flush("  RECONNECT FAILED — skipping rest of batch")
                            break

                time.sleep(0.1)  # rate limiting between tickers

            total_rows += day_rows
            batch_rows += day_rows
            if day_rows > 0:
                _flush(f"  Day total: {day_rows} rows")

        _flush(f"--- Batch {batch_num} done: {batch_rows:,} rows ---")

        # Brief pause between batches
        if batch_start + batch_size < len(all_dates):
            time.sleep(2)

    conn.close()

    _flush(f"\n{'='*70}")
    _flush("DOWNLOAD COMPLETE")
    _flush(f"{'='*70}")
    _flush(f"Total rows:   {total_rows:,}")
    _flush(f"Total errors: {total_errors}")
    _flush(f"Database:     {db_path}")
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    _flush(f"DB size:      {db_size:.1f} MB")
    _flush("\nTo resume interrupted downloads, just re-run with same parameters.")
    _flush("Already-downloaded data is skipped automatically via download_log table.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download ThetaData historical options data")
    parser.add_argument("--days", type=int, default=30, help="Days to download (default: 30)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    parser.add_argument("--db", type=str, default=DB_PATH, help="SQLite DB path")
    parser.add_argument("--ohlc-only", action="store_true", help="Skip greeks (faster)")
    parser.add_argument("--otm", type=int, default=4, help="OTM strikes above ATM (default: 4)")
    parser.add_argument("--otm-below", type=int, default=None, help="OTM strikes below ATM for PUTs (default: same as --otm). Use 8-12 for crash day coverage.")
    parser.add_argument("--batch-size", type=int, default=5, help="Days per batch (default: 5)")
    args = parser.parse_args()

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_dt = date.today() - timedelta(days=args.days)

    end_dt = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    run_download(
        tickers=tickers,
        start_date=start_dt,
        end_date=end_dt,
        db_path=args.db,
        skip_greeks=args.ohlc_only,
        otm_strikes=args.otm,
        otm_below=args.otm_below,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
