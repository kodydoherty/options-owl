"""Export PostgreSQL option_ticks + stock_ticks to ThetaData-compatible SQLite.

Creates a temporary SQLite DB with the same schema as thetadata_options.db
so backtest_gold_standard.py can run against recent PG data.

Usage:
    # Run on droplet inside a container that has DATABASE_URL
    python scripts/export_pg_to_thetadata.py --days 3 --output /tmp/pg_export.db
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from datetime import date as date_type
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


async def export(days: int, output: str):
    from options_owl.db import postgres as pg

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    await pg.init_pool(db_url)
    pool = pg._pool

    conn_pg = await pool.acquire()

    # Get date range
    dates = await conn_pg.fetch(f"""
        SELECT DISTINCT captured_at::date as d
        FROM option_ticks
        ORDER BY d DESC
        LIMIT {days}
    """)
    date_objs = sorted([r[0] for r in dates])
    dates = [str(d) for d in date_objs]
    print(f"Exporting {len(dates)} days: {dates}")

    # Create SQLite DB with ThetaData schema
    if os.path.exists(output):
        os.remove(output)
    db = sqlite3.connect(output)
    db.execute("PRAGMA journal_mode = WAL")

    db.execute("""
        CREATE TABLE option_ohlc (
            ticker TEXT, expiration TEXT, strike REAL, right TEXT,
            timestamp TEXT, open REAL, high REAL, low REAL, close REAL,
            volume INTEGER
        )
    """)
    db.execute("""
        CREATE TABLE option_greeks (
            ticker TEXT, expiration TEXT, strike REAL, right TEXT,
            timestamp TEXT, implied_vol REAL, delta REAL, gamma REAL,
            theta REAL, vega REAL, underlying_price REAL
        )
    """)
    db.execute("""
        CREATE TABLE option_quotes (
            ticker TEXT, expiration TEXT, strike REAL, right TEXT,
            timestamp TEXT, bid REAL, ask REAL, bid_size INTEGER, ask_size INTEGER
        )
    """)
    db.execute("""
        CREATE TABLE stock_ohlc (
            ticker TEXT, timestamp TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER
        )
    """)

    total_options = 0
    total_stock = 0

    for date_str, date_obj in zip(dates, date_objs):
        print(f"\n  {date_str}:", end=" ", flush=True)

        # Fetch CALL option ticks during market hours, paginated per ticker
        # CRITICAL: Only export 0DTE contracts (expiry = same day) to avoid
        # interleaving multiple expiries for the same strike, which causes
        # the backtest to see fake 900%+ peaks from mixing DTE premiums.
        ticker_rows = await conn_pg.fetch("""
            SELECT DISTINCT ticker FROM option_ticks
            WHERE captured_at::date = $1 AND option_type = 'call'
        """, date_obj)
        day_tickers = [r["ticker"] for r in ticker_rows]
        day_option_count = 0

        for tk in day_tickers:
            rows = await conn_pg.fetch("""
                SELECT ticker, option_type, strike, expiry_date,
                       bid, ask, mid, last, volume, open_interest,
                       iv, delta, gamma, theta, vega,
                       underlying_price, captured_at
                FROM option_ticks
                WHERE captured_at::date = $1
                  AND option_type = 'call'
                  AND ticker = $2
                  AND expiry_date = $3
                  AND captured_at::time >= '13:30:00'
                  AND captured_at::time <= '20:00:00'
                ORDER BY strike, captured_at
            """, date_obj, tk, date_str)

            ohlc_rows = []
            greeks_rows = []
            quotes_rows = []

            for r in rows:
                right = "CALL"
                strike = r["strike"]
                expiry = r["expiry_date"]
                ts = r["captured_at"]
                ts_et = ts.astimezone(ZoneInfo("America/New_York"))
                ts_str = ts_et.strftime("%Y-%m-%d %H:%M:%S%z")
                ts_str = ts_str[:-2] + ":" + ts_str[-2:]

                mid = r["mid"] or 0
                bid = r["bid"] or 0
                ask = r["ask"] or 0
                last = r["last"] or 0
                close = mid if mid > 0 else (bid + ask) / 2 if bid and ask else last

                ohlc_rows.append((
                    tk, expiry, strike, right, ts_str,
                    close, close, close, close,
                    r["volume"] or 0,
                ))
                greeks_rows.append((
                    tk, expiry, strike, right, ts_str,
                    r["iv"] or 0, r["delta"] or 0, r["gamma"] or 0,
                    r["theta"] or 0, r["vega"] or 0, r["underlying_price"] or 0,
                ))
                quotes_rows.append((
                    tk, expiry, strike, right, ts_str,
                    bid, ask, 0, 0,
                ))

            db.executemany(
                "INSERT INTO option_ohlc VALUES (?,?,?,?,?,?,?,?,?,?)", ohlc_rows
            )
            db.executemany(
                "INSERT INTO option_greeks VALUES (?,?,?,?,?,?,?,?,?,?,?)", greeks_rows
            )
            db.executemany(
                "INSERT INTO option_quotes VALUES (?,?,?,?,?,?,?,?,?)", quotes_rows
            )
            day_option_count += len(rows)

        print(f"{day_option_count} option ticks ({len(day_tickers)} tickers)", end="", flush=True)
        total_options += day_option_count

        # Fetch stock ticks
        stock_rows = await conn_pg.fetch("""
            SELECT ticker, price, bid, ask, volume, vwap, captured_at
            FROM stock_ticks
            WHERE captured_at::date = $1
            ORDER BY ticker, captured_at
        """, date_obj)

        print(f", {len(stock_rows)} stock ticks", flush=True)

        stock_ohlc = []
        for sr in stock_rows:
            ts_et = sr["captured_at"].astimezone(ZoneInfo("America/New_York"))
            ts_str = ts_et.strftime("%Y-%m-%d %H:%M:%S%z")
            ts_str = ts_str[:-2] + ":" + ts_str[-2:]
            price = sr["price"] or 0
            stock_ohlc.append((
                sr["ticker"], ts_str,
                price, price, price, price,  # OHLC = price (tick data)
                sr["volume"] or 0,
            ))

        db.executemany(
            "INSERT INTO stock_ohlc VALUES (?,?,?,?,?,?,?)", stock_ohlc
        )
        total_stock += len(stock_rows)
        db.commit()

    # Create indexes matching ThetaData DB
    print("\nCreating indexes...", flush=True)
    db.execute("CREATE INDEX idx_ohlc_ticker_ts ON option_ohlc(ticker, timestamp)")
    db.execute("CREATE INDEX idx_ohlc_ticker_strike ON option_ohlc(ticker, right, strike, timestamp)")
    db.execute("CREATE INDEX idx_greeks_lookup ON option_greeks(ticker, expiration, strike, right, timestamp)")
    db.execute("CREATE INDEX idx_quotes_lookup ON option_quotes(ticker, expiration, strike, right, timestamp)")
    db.execute("CREATE INDEX idx_stock_ticker_ts ON stock_ohlc(ticker, timestamp)")
    db.commit()
    db.close()

    await pool.release(conn_pg)

    file_size = os.path.getsize(output) / (1024 * 1024)
    print(f"\nDone: {total_options:,} option ticks + {total_stock:,} stock ticks → {output} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--output", default="/tmp/pg_export.db")
    args = parser.parse_args()
    asyncio.run(export(args.days, args.output))


if __name__ == "__main__":
    main()
