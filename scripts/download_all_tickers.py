"""Download historical 0DTE data for all tickers — parallel version.

Runs up to WORKERS concurrent ticker downloads. Each ticker downloads
sequentially (3 API calls per day: underlying + call + put bars), but
multiple tickers run in parallel.

Polygon paid plan = unlimited API calls, so parallelism is safe.
SQLite handles concurrent writes via WAL mode + retries.

Usage:
    python scripts/download_all_tickers.py
    python scripts/download_all_tickers.py --workers 6
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_historical_0dte import run_download

# All tickers from our signal history + harvester universe, ordered by priority
TICKERS = [
    # High volume ETFs — $1 strikes, most liquid 0DTE
    ("SPY", "2024-04-01"),
    ("QQQ", "2024-04-01"),
    ("IWM", "2024-04-01"),
    # High volume stocks — most of our signals
    ("TSLA", "2024-04-01"),
    ("AMZN", "2024-04-01"),
    ("NVDA", "2024-04-01"),
    ("AAPL", "2024-04-01"),
    ("MSFT", "2024-04-01"),
    ("META", "2024-04-01"),
    ("GOOGL", "2024-04-01"),
    ("AMD", "2024-04-01"),
    ("MSTR", "2024-04-01"),
    ("MU", "2024-04-01"),
    # --- NEW: harvester universe + likely future Discord signals ---
    ("NFLX", "2024-04-01"),
    ("SMCI", "2024-04-01"),
    ("PLTR", "2024-04-01"),
    ("COIN", "2024-04-01"),
    ("BA", "2024-04-01"),
    ("JPM", "2024-04-01"),
    # Sector/index ETFs
    ("DIA", "2024-04-01"),
    ("XLK", "2024-04-01"),
    ("XLF", "2024-04-01"),
    ("GLD", "2024-04-01"),
    ("SLV", "2024-04-01"),
    ("TLT", "2024-04-01"),
]

DEFAULT_WORKERS = 4  # 4 parallel tickers, each doing ~0.5s between API calls


def download_ticker(ticker: str, start: str) -> tuple[str, float, str]:
    """Download one ticker, return (ticker, elapsed_seconds, status)."""
    t0 = time.time()
    try:
        run_download(ticker=ticker, start_date=start)
        return (ticker, time.time() - t0, "OK")
    except Exception as e:
        return (ticker, time.time() - t0, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel downloads")
    parser.add_argument("--only-new", action="store_true",
                        help="Skip tickers that already have data")
    args = parser.parse_args()

    # Filter to only new tickers if requested
    tickers = TICKERS
    if args.only_new:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
        try:
            conn = sqlite3.connect(db_path)
            existing = {r[0] for r in conn.execute(
                "SELECT DISTINCT ticker FROM trading_days"
            ).fetchall()}
            conn.close()
            tickers = [(t, s) for t, s in TICKERS if t not in existing]
            print(f"  Skipping {len(TICKERS) - len(tickers)} tickers with existing data")
        except Exception:
            pass  # DB doesn't exist yet, download all

    total_start = time.time()
    print("=" * 60)
    print("  DOWNLOADING 0DTE DATA — PARALLEL MODE")
    print("=" * 60)
    print(f"  Tickers: {', '.join(t[0] for t in tickers)}")
    print(f"  Workers: {args.workers} parallel downloads")
    print(f"  Period: April 2024 to present")
    print(f"  Resumable: will skip already-downloaded days")
    print("=" * 60)
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_ticker, ticker, start): ticker
            for ticker, start in tickers
        }
        for future in as_completed(futures):
            ticker, elapsed, status = future.result()
            results.append((ticker, elapsed, status))
            symbol = "✓" if status == "OK" else "✗"
            print(f"\n  {symbol} {ticker} done in {elapsed/60:.1f}m — {status}")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DOWNLOADS COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*60}")
    for ticker, t, status in sorted(results, key=lambda x: x[0]):
        print(f"  {ticker:<8} {t/60:>6.1f}m  {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
