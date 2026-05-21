"""Fetch intraday price data for signal validation using yfinance."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import yfinance as yf
from loguru import logger

from options_owl.models.signals import PriceBar


def fetch_intraday_sync(
    ticker: str,
    date: str | None = None,
    interval: str = "1m",
) -> list[PriceBar]:
    """Fetch intraday bars for a ticker. date format: YYYY-MM-DD. None = today."""
    try:
        tk = yf.Ticker(ticker)
        if date:
            start = date
            # yfinance end is exclusive, so add 1 day
            dt = datetime.strptime(date, "%Y-%m-%d")
            end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
            df = tk.history(start=start, end=end, interval=interval)
        else:
            df = tk.history(period="1d", interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {ticker} on {date or 'today'}")
            return []

        bars = []
        for ts, row in df.iterrows():
            bars.append(
                PriceBar(
                    timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=round(row["Open"], 4),
                    high=round(row["High"], 4),
                    low=round(row["Low"], 4),
                    close=round(row["Close"], 4),
                    volume=int(row["Volume"]),
                )
            )
        logger.info(f"Fetched {len(bars)} bars for {ticker} ({interval}) on {date or 'today'}")
        return bars

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return []


async def fetch_intraday(
    ticker: str,
    date: str | None = None,
    interval: str = "1m",
) -> list[PriceBar]:
    """Async wrapper around yfinance (which is sync)."""
    return await asyncio.to_thread(fetch_intraday_sync, ticker, date, interval)


async def fetch_bars_for_signal(
    ticker: str,
    signal_date: str,
) -> list[PriceBar]:
    """Fetch 1-minute bars for a signal's trading day.

    Falls back to 5m if 1m data is unavailable (>7 days old).
    """
    bars = await fetch_intraday(ticker, signal_date, interval="1m")
    if not bars:
        logger.info(f"1m data unavailable for {ticker} on {signal_date}, trying 5m")
        bars = await fetch_intraday(ticker, signal_date, interval="5m")
    return bars
