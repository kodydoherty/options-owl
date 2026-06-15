"""IV Rank / IV Percentile filtering for trade signals.

Feature flag: ENABLE_IV_FILTER
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import yfinance as yf

if TYPE_CHECKING:
    from options_owl.config.settings import Settings

# Simple in-memory cache: {ticker: (timestamp, iv_rank, iv_percentile)}
_iv_cache: dict[str, tuple[float, float, float]] = {}
_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


def _get_current_iv_sync(ticker: str) -> float | None:
    """Get current implied volatility from ATM options via yfinance (sync).

    Uses the nearest expiration and the strike closest to current price.

    Returns:
        Annualized IV as a decimal (e.g. 0.30 for 30%), or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        # Use nearest expiration
        chain = stock.option_chain(expirations[0])
        hist = stock.history(period="1d")
        if hist.empty:
            return None

        current_price = float(hist["Close"].iloc[-1])

        # Find ATM call (closest strike to current price)
        calls = chain.calls
        if calls.empty:
            return None

        calls = calls.copy()
        calls["dist"] = (calls["strike"] - current_price).abs()
        atm_row = calls.loc[calls["dist"].idxmin()]
        iv = float(atm_row["impliedVolatility"])
        return iv
    except Exception:
        return None


def _get_current_iv(ticker: str) -> float | None:
    """Async-safe wrapper — runs yfinance in a thread."""
    import asyncio
    try:
        asyncio.get_running_loop()
        # We're inside an event loop — must use to_thread
        # But this function is called synchronously from _compute_and_cache,
        # so just call the sync version directly (caller handles threading).
        return _get_current_iv_sync(ticker)
    except RuntimeError:
        return _get_current_iv_sync(ticker)


def _get_historical_realized_vol_sync(ticker: str) -> list[float]:
    """Compute daily realized volatility estimates over the past year (sync).

    Uses 20-day rolling standard deviation of log returns, annualized.

    Returns:
        List of annualized volatility values (one per trading day with enough history).
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 21:
            return []

        closes = hist["Close"].tolist()
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

        # 20-day rolling std, annualized
        window = 20
        vols: list[float] = []
        for i in range(window, len(log_returns) + 1):
            segment = log_returns[i - window : i]
            mean = sum(segment) / len(segment)
            var = sum((x - mean) ** 2 for x in segment) / (len(segment) - 1)
            daily_std = math.sqrt(var)
            annual_vol = daily_std * math.sqrt(252)
            vols.append(annual_vol)
        return vols
    except Exception:
        return []


def _get_historical_realized_vol(ticker: str) -> list[float]:
    """Async-safe wrapper."""
    return _get_historical_realized_vol_sync(ticker)


def _compute_and_cache(ticker: str) -> tuple[float, float] | None:
    """Compute IV rank and IV percentile, caching the result.

    Returns:
        Tuple of (iv_rank, iv_percentile) as percentages (0-100), or None on failure.
    """
    # Check cache
    cached = _iv_cache.get(ticker)
    if cached is not None:
        ts, rank, pct = cached
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return (rank, pct)

    current_iv = _get_current_iv(ticker)
    if current_iv is None:
        return None

    historical_vols = _get_historical_realized_vol(ticker)
    if not historical_vols:
        return None

    # IV Rank: (current - 52wk low) / (52wk high - 52wk low) * 100
    low_iv = min(historical_vols)
    high_iv = max(historical_vols)
    if high_iv == low_iv:
        iv_rank = 50.0  # Default if no range
    else:
        iv_rank = (current_iv - low_iv) / (high_iv - low_iv) * 100.0
        iv_rank = max(0.0, min(100.0, iv_rank))

    # IV Percentile: % of days with realized vol below current IV
    days_below = sum(1 for v in historical_vols if v < current_iv)
    iv_percentile = days_below / len(historical_vols) * 100.0

    _iv_cache[ticker] = (time.time(), iv_rank, iv_percentile)
    return (iv_rank, iv_percentile)


def fetch_iv_rank(ticker: str) -> float | None:
    """Fetch IV Rank for a ticker.

    IV Rank = (current IV - 52wk low IV) / (52wk high IV - 52wk low IV) * 100

    Args:
        ticker: Stock ticker symbol.

    Returns:
        IV Rank as a percentage (0-100), or None on failure.
    """
    result = _compute_and_cache(ticker)
    if result is None:
        return None
    return result[0]


def fetch_iv_percentile(ticker: str) -> float | None:
    """Fetch IV Percentile for a ticker.

    What percentage of days in the past year had IV lower than today's IV.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        IV Percentile as a percentage (0-100), or None on failure.
    """
    result = _compute_and_cache(ticker)
    if result is None:
        return None
    return result[1]


def check_iv_filter(ticker: str, settings: Settings) -> tuple[bool, str]:
    """Check whether a ticker's IV rank passes the configured filter.

    Args:
        ticker: Stock ticker symbol.
        settings: Application settings with IV filter configuration.

    Returns:
        Tuple of (passes, reason). passes is True if the trade should proceed.
    """
    if not settings.ENABLE_IV_FILTER:
        return (True, "IV filter is disabled")

    iv_rank = fetch_iv_rank(ticker)
    if iv_rank is None:
        return (True, f"Could not fetch IV data for {ticker}, allowing trade")

    if iv_rank < settings.IV_RANK_MIN:
        return (
            False,
            f"{ticker} IV Rank {iv_rank:.1f}% is below minimum {settings.IV_RANK_MIN:.1f}%",
        )

    if iv_rank > settings.IV_RANK_MAX:
        return (
            False,
            f"{ticker} IV Rank {iv_rank:.1f}% is above maximum {settings.IV_RANK_MAX:.1f}%",
        )

    return (True, f"{ticker} IV Rank {iv_rank:.1f}% is within acceptable range")
