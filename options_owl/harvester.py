"""OptionsOwl Harvester — standalone options-data collector.

Pulls rich per-contract snapshots from Polygon Options Advanced on a timer
and stores them in a dedicated SQLite DB for later ML training and backtesting.

Captures (per contract, per poll):
- Real-time bid, ask, bid_size, ask_size, midpoint
- Last trade price + timestamp
- Day OHLC, volume, vwap
- Open interest
- Locally computed IV + greeks (delta, gamma, theta, vega) via Black-Scholes,
  since Options Advanced tier currently returns empty greeks/IV fields.

Universe, poll interval, DTE window, and strike window are all env-driven so
the container can be reconfigured without code changes.

Entry point: `python -m options_owl.harvester`
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import aiosqlite
import httpx
import yfinance as yf
from loguru import logger

from options_owl.collectors.candle_collector import CandleCollector
from options_owl.main import LOG_DIR, configure_logging, write_heartbeat
from options_owl.risk.greeks import (
    calc_delta,
    calc_gamma,
    calc_iv_from_premium,
    calc_theta,
    calc_vega,
)

# ---------------------------------------------------------------------------
# Config — all env-driven so we can tune without rebuilds
# ---------------------------------------------------------------------------

DB_PATH = Path(os.getenv("HARVEST_DB_PATH", "journal/options_data.db"))
UNIVERSE = [
    t.strip().upper()
    for t in os.getenv(
        "HARVEST_UNIVERSE",
        "SPY,QQQ,IWM,AAPL,TSLA,NVDA,META,AMD,AMZN,GOOGL,MSFT,MU,MSTR",
    ).split(",")
    if t.strip()
]
POLL_INTERVAL = int(os.getenv("HARVEST_POLL_INTERVAL", "60"))
MAX_DTE = int(os.getenv("HARVEST_MAX_DTE", "7"))
STRIKE_WINDOW = int(os.getenv("HARVEST_STRIKE_WINDOW", "5"))
MARKET_HOURS_ONLY = os.getenv("HARVEST_MARKET_HOURS_ONLY", "true").lower() == "true"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
RISK_FREE_RATE = float(os.getenv("HARVEST_RISK_FREE_RATE", "0.045"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("HARVEST_MAX_CONCURRENT", "4"))

# Reuse position_monitor's market-hours helper to stay consistent
from options_owl.execution.position_monitor import _is_market_hours  # noqa: E402

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
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
"""


async def init_db(path: Path) -> None:
    """Create the harvester DB and tables if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as conn:
        # WAL mode survives ungraceful shutdowns better than journal mode
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.executescript(SCHEMA)
        await conn.commit()


# ---------------------------------------------------------------------------
# Underlying price fetcher (yfinance — Polygon underlying is delayed on our plan)
# ---------------------------------------------------------------------------

_price_cache: dict[str, tuple[float, float]] = {}  # ticker -> (price, unix_ts)
PRICE_CACHE_TTL = 15  # seconds


def _get_underlying_price(ticker: str) -> float | None:
    """Fetch current underlying price via yfinance, cached for PRICE_CACHE_TTL."""
    now = time.time()
    cached = _price_cache.get(ticker)
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0]
    try:
        # yfinance uses ^ prefix for indices (e.g. ^VIX, ^GSPC)
        yf_symbol = f"^{ticker}" if ticker in ("VIX", "GSPC", "DJI", "IXIC") else ticker
        tk = yf.Ticker(yf_symbol)
        price = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
        if not price or price <= 0:
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
        price = float(price)
        _price_cache[ticker] = (price, now)
        return price
    except Exception as exc:
        logger.warning(f"yfinance price fetch failed for {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Polygon chain fetcher
# ---------------------------------------------------------------------------


async def _fetch_chain(
    client: httpx.AsyncClient,
    ticker: str,
    strike_low: float,
    strike_high: float,
    expiry_lo: str,
    expiry_hi: str,
) -> list[dict]:
    """Fetch filtered option chain snapshot for a ticker.

    Uses Polygon's strike_price and expiration_date filters to keep the
    response small (ATM ± window, near-term expiries only).
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
    params = {
        "strike_price.gte": strike_low,
        "strike_price.lte": strike_high,
        "expiration_date.gte": expiry_lo,
        "expiration_date.lte": expiry_hi,
        "limit": 250,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        resp = await client.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(
                f"Polygon chain fetch failed for {ticker}: "
                f"HTTP {resp.status_code} {resp.text[:160]}"
            )
            return []
        return resp.json().get("results", []) or []
    except Exception as exc:
        logger.warning(f"Polygon chain fetch error for {ticker}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Snapshot parsing + greeks computation
# ---------------------------------------------------------------------------


def _parse_contract(result: dict, underlying_price: float | None) -> dict | None:
    """Convert a Polygon chain result into a flat row ready for DB insertion.

    Computes IV and greeks locally from the Black-Scholes model since Polygon
    Options Advanced returns empty greeks on our plan.
    """
    details = result.get("details") or {}
    contract_ticker = details.get("ticker")
    if not contract_ticker:
        return None

    strike = details.get("strike_price")
    expiry = details.get("expiration_date")
    option_type = (details.get("contract_type") or "").lower()
    if strike is None or not expiry or option_type not in ("call", "put"):
        return None

    quote = result.get("last_quote") or {}
    trade = result.get("last_trade") or {}
    day = result.get("day") or {}

    bid = quote.get("bid")
    ask = quote.get("ask")
    midpoint = quote.get("midpoint")
    if midpoint is None and bid and ask:
        midpoint = round((bid + ask) / 2.0, 4)

    # Local BS greeks only if we have underlying price + a reasonable premium
    iv = delta = gamma = theta = vega = None
    if underlying_price and midpoint and midpoint > 0:
        try:
            expiry_dt = date.fromisoformat(expiry)
            today = date.today()
            days_to_expiry = (expiry_dt - today).days
            # For 0DTE use fractional time remaining (hours / 24)
            if days_to_expiry <= 0:
                now = datetime.now()
                close_hour = 16  # 4 PM ET-ish; close enough for 0DTE greeks
                hours_left = max(close_hour - now.hour - (now.minute / 60.0), 0.1)
                T = hours_left / (24.0 * 365.0)
            else:
                T = days_to_expiry / 365.0

            if T > 0:
                iv = calc_iv_from_premium(
                    midpoint, underlying_price, strike, T, RISK_FREE_RATE, option_type
                )
                if iv and 0.001 < iv < 5.0:
                    delta = round(
                        calc_delta(underlying_price, strike, T, RISK_FREE_RATE, iv, option_type),
                        4,
                    )
                    gamma = round(
                        calc_gamma(underlying_price, strike, T, RISK_FREE_RATE, iv), 6
                    )
                    theta = round(
                        calc_theta(underlying_price, strike, T, RISK_FREE_RATE, iv, option_type),
                        4,
                    )
                    vega = round(
                        calc_vega(underlying_price, strike, T, RISK_FREE_RATE, iv), 4
                    )
                    iv = round(iv, 4)
        except Exception as exc:
            logger.debug(f"Greeks computation failed for {contract_ticker}: {exc}")

    return {
        "contract_ticker": contract_ticker,
        "underlying": contract_ticker.split(":")[1][:-15] if ":" in contract_ticker else None,
        "strike": float(strike),
        "expiry_date": expiry,
        "option_type": option_type,
        "underlying_price": underlying_price,
        "bid": bid,
        "ask": ask,
        "bid_size": quote.get("bid_size"),
        "ask_size": quote.get("ask_size"),
        "midpoint": midpoint,
        "last_trade_price": trade.get("price"),
        "last_trade_ts_ns": trade.get("sip_timestamp"),
        "day_open": day.get("open"),
        "day_high": day.get("high"),
        "day_low": day.get("low"),
        "day_close": day.get("close"),
        "day_volume": day.get("volume"),
        "day_vwap": day.get("vwap"),
        "open_interest": result.get("open_interest"),
        "implied_volatility": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


async def _persist_rows(db_path: Path, ticker: str, rows: list[dict]) -> int:
    """Insert contract metadata + snapshot rows. Returns number of snapshots written.

    Opens a fresh connection per call so each ticker's data is independently
    committed — a container kill can only lose the in-flight ticker, not the
    entire poll cycle.
    """
    if not rows:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            # Upsert contract metadata (one row per contract, first-seen only)
            await conn.executemany(
                """
                INSERT OR IGNORE INTO harvest_contracts
                    (contract_ticker, underlying, strike, expiry_date, option_type, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["contract_ticker"],
                        ticker,
                        r["strike"],
                        r["expiry_date"],
                        r["option_type"],
                        now_iso,
                    )
                    for r in rows
                ],
            )
            # Insert snapshot rows
            await conn.executemany(
                """
                INSERT INTO harvest_snapshots (
                    contract_ticker, captured_at, underlying_price,
                    bid, ask, bid_size, ask_size, midpoint,
                    last_trade_price, last_trade_ts_ns,
                    day_open, day_high, day_low, day_close, day_volume, day_vwap,
                    open_interest, implied_volatility, delta, gamma, theta, vega
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["contract_ticker"],
                        now_iso,
                        r["underlying_price"],
                        r["bid"],
                        r["ask"],
                        r["bid_size"],
                        r["ask_size"],
                        r["midpoint"],
                        r["last_trade_price"],
                        r["last_trade_ts_ns"],
                        r["day_open"],
                        r["day_high"],
                        r["day_low"],
                        r["day_close"],
                        r["day_volume"],
                        r["day_vwap"],
                        r["open_interest"],
                        r["implied_volatility"],
                        r["delta"],
                        r["gamma"],
                        r["theta"],
                        r["vega"],
                    )
                    for r in rows
                ],
            )
            await conn.commit()
        return len(rows)
    except Exception as exc:
        logger.error(f"DB persist failed for {ticker} ({len(rows)} rows): {exc}")
        return 0


# ---------------------------------------------------------------------------
# Per-ticker harvest step
# ---------------------------------------------------------------------------


async def _harvest_ticker(
    client: httpx.AsyncClient,
    db_path: Path,
    ticker: str,
    semaphore: asyncio.Semaphore,
    candle_collector: CandleCollector | None = None,
) -> int:
    """Fetch + parse + persist one ticker. Returns snapshot count."""
    async with semaphore:
        underlying_price = _get_underlying_price(ticker)
        if underlying_price is None:
            logger.warning(f"{ticker}: no underlying price, skipping")
            return 0

        # Feed price to candle collector for 5m bar building
        if candle_collector is not None and underlying_price > 0:
            candle_collector.record_price(ticker, underlying_price)

        strike_low = round(underlying_price * (1 - 0.02 * STRIKE_WINDOW), 2)
        strike_high = round(underlying_price * (1 + 0.02 * STRIKE_WINDOW), 2)
        today = date.today()
        expiry_lo = today.isoformat()
        expiry_hi = (today.fromordinal(today.toordinal() + MAX_DTE)).isoformat()

        raw = await _fetch_chain(
            client, ticker, strike_low, strike_high, expiry_lo, expiry_hi
        )
        if not raw:
            return 0

        rows = [p for p in (_parse_contract(r, underlying_price) for r in raw) if p]
        if not rows:
            return 0

        written = await _persist_rows(db_path, ticker, rows)
        logger.info(
            f"{ticker}: {written} contracts @ ${underlying_price:.2f} "
            f"(strikes {strike_low:.0f}-{strike_high:.0f}, exp≤{expiry_hi})"
        )
        return written


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_harvester() -> None:
    """Main async loop: poll every POLL_INTERVAL seconds during market hours."""
    if not POLYGON_API_KEY:
        logger.critical("POLYGON_API_KEY is not set — harvester cannot run")
        sys.exit(2)

    await init_db(DB_PATH)

    # Initialize candle collector (shares the same DB)
    # Connects to wss://socket.polygon.io/stocks for real-time minute bars.
    # This is SEPARATE from the options WS (wss://socket.polygon.io/options)
    # that trading agents use — no connection conflict.
    candle_collector = CandleCollector(DB_PATH, UNIVERSE)
    await candle_collector.init_db()
    await candle_collector.start_ws(POLYGON_API_KEY)

    logger.info(
        f"Harvester started — universe={UNIVERSE}, poll={POLL_INTERVAL}s, "
        f"DTE≤{MAX_DTE}, strikes=ATM±{STRIKE_WINDOW}, db={DB_PATH}, "
        f"candle_ws={'on' if candle_collector.ws_connected else 'starting'}"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        logger.info("Harvester received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            signal.signal(sig, _shutdown)

    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            write_heartbeat()

            if MARKET_HOURS_ONLY and not _is_market_hours():
                logger.debug("Outside market hours — sleeping 5 min")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    pass
                continue

            t0 = time.time()
            try:
                results = await asyncio.gather(
                    *[
                        _harvest_ticker(
                            client, DB_PATH, ticker, semaphore, candle_collector
                        )
                        for ticker in UNIVERSE
                    ],
                    return_exceptions=True,
                )
                total = sum(r for r in results if isinstance(r, int))
                errors = [r for r in results if isinstance(r, Exception)]

                # Flush completed 5m candles to DB
                candle_count = await candle_collector.flush()

                elapsed = time.time() - t0
                logger.info(
                    f"Poll complete: {total} snapshots across {len(UNIVERSE)} "
                    f"tickers in {elapsed:.1f}s ({len(errors)} errors"
                    f"{f', {candle_count} candle bars' if candle_count else ''})"
                )
                if errors:
                    for exc in errors[:3]:
                        logger.warning(f"  harvest error: {exc}")
            except Exception as exc:
                logger.error(f"Harvest cycle failed: {exc}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass

    await candle_collector.stop_ws()
    logger.info("Harvester shut down cleanly")


def main() -> None:
    configure_logging(verbose=False)
    # Drop a heartbeat immediately so Docker healthcheck passes during startup
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_heartbeat()
    try:
        asyncio.run(run_harvester())
    except KeyboardInterrupt:
        logger.info("Harvester interrupted")


if __name__ == "__main__":
    main()
