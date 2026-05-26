import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from options_owl.config.settings import Settings

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_DIR = Path("journal/logs")

def configure_logging(verbose: bool = False) -> None:
    """Set up loguru with console + file rotation + JSON structured logs."""
    # Remove default handler
    logger.remove()

    # Console: human-readable, colorized.
    # LOG_LEVEL env var wins over --verbose flag so we can flip owlets to DEBUG
    # via docker-compose without rebuilding.
    env_level = os.getenv("LOG_LEVEL", "").strip().upper()
    log_level = env_level or ("DEBUG" if verbose else "INFO")
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File: human-readable, rotated daily, kept 7 days
    logger.add(
        LOG_DIR / "options_owl_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
    )

    # JSON structured log for machine parsing (Docker log aggregation)
    logger.add(
        LOG_DIR / "options_owl.json",
        level="INFO",
        rotation="50 MB",
        retention="7 days",
        serialize=True,
    )


# ---------------------------------------------------------------------------
# Heartbeat file — Docker healthcheck reads this
# ---------------------------------------------------------------------------

HEARTBEAT_PATH = Path("journal/heartbeat")


def write_heartbeat() -> None:
    """Write current epoch to heartbeat file for Docker healthcheck."""
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_PATH.write_text(str(int(time.time())))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Collector with retry + backoff
# ---------------------------------------------------------------------------

MAX_RETRIES = 50  # effectively unlimited — Docker will restart if process dies
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 300  # 5 minutes cap


def check_polygon_realtime_entitlement(settings: Settings) -> None:
    """Fail-fast guard: refuse to go LIVE if Polygon can't serve real-time options data.

    Verifies the options snapshot endpoint is authorized AND the last quote is
    under 2 minutes old. If either check fails while PAPER_TRADE=false, aborts
    startup so kody can't silently degrade into trading on stale/previous-day prices.
    """
    import json
    import urllib.error
    import urllib.request

    key = getattr(settings, "POLYGON_API_KEY", "") or ""
    if not key:
        if not settings.PAPER_TRADE:
            logger.critical("LIVE mode but POLYGON_API_KEY is empty — aborting.")
            sys.exit(2)
        logger.warning("POLYGON_API_KEY not set; skipping real-time data self-test.")
        return

    # Query a near-ATM SPY option (0DTE call near ~$500-700 range) instead of
    # limit=1, which returns deep ITM contracts with no active quoting.
    # First get approximate SPY price from prev-day close, then query ATM.
    try:
        prev_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey={key}"
        with urllib.request.urlopen(prev_url, timeout=10) as r:
            prev = json.loads(r.read())
        spy_price = prev.get("results", [{}])[0].get("c", 550)
    except Exception:
        spy_price = 550  # safe fallback

    today = time.strftime("%Y-%m-%d")
    lo = int(spy_price - 5)
    hi = int(spy_price + 5)
    url = (
        f"https://api.polygon.io/v3/snapshot/options/SPY"
        f"?strike_price.gte={lo}&strike_price.lte={hi}"
        f"&expiration_date.gte={today}&contract_type=call"
        f"&limit=1&order=asc&sort=strike_price&apiKey={key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())
            msg = err.get("error") or err.get("message") or str(e)
        except Exception:
            msg = str(e)
        if not settings.PAPER_TRADE:
            logger.critical(
                f"LIVE mode but Polygon snapshot unauthorized (HTTP {e.code}: {msg}) — "
                f"aborting to prevent trading on stale/delayed quotes."
            )
            sys.exit(2)
        logger.warning(f"Polygon snapshot unauthorized (HTTP {e.code}: {msg}) — paper mode, continuing.")
        return
    except Exception as e:
        if not settings.PAPER_TRADE:
            logger.critical(f"LIVE mode but Polygon self-test failed: {e} — aborting.")
            sys.exit(2)
        logger.warning(f"Polygon self-test error: {e} — paper mode, continuing.")
        return

    results = body.get("results") or []
    if not results:
        logger.warning("Polygon snapshot returned no results; cannot verify freshness.")
        return

    lq = (results[0].get("last_quote") or {})
    last_updated_ns = lq.get("last_updated")
    contract = (results[0].get("details") or {}).get("ticker", "?")
    if not last_updated_ns:
        logger.warning("Polygon snapshot missing last_quote.last_updated; cannot verify freshness.")
        return

    age_sec = time.time() - (last_updated_ns / 1e9)
    # Pre-market (before 9:30 ET / 13:30 UTC) options quotes can be 15-30 min
    # stale since the regular session hasn't started.  Use a relaxed threshold
    # so bots can boot during pre-market without crash-looping.
    import datetime as _dt
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    market_open_utc = now_utc.replace(hour=13, minute=30, second=0, microsecond=0)
    max_age = 1800 if now_utc < market_open_utc else 120  # 30 min pre-market, 2 min regular
    if age_sec < max_age:
        logger.info(f"✅ Polygon real-time OK ({contract} quote age {age_sec:.0f}s)")
    elif not settings.PAPER_TRADE:
        logger.critical(
            f"LIVE mode but Polygon quote is {age_sec/60:.1f} min old ({contract}) — "
            f"aborting to prevent stale-quote trading."
        )
        sys.exit(2)
    else:
        logger.warning(f"Polygon quote is {age_sec/60:.1f} min old — paper mode, continuing.")


def _cleanup_connections() -> None:
    """Best-effort cleanup of PG pool and Redis on crash/exit."""
    try:
        loop = asyncio.new_event_loop()

        async def _close():
            try:
                from options_owl.db import postgres as pg
                await pg.close_pool()
            except Exception:
                pass
            try:
                from options_owl.db import redis_client
                await redis_client.close()
            except Exception:
                pass

        loop.run_until_complete(_close())
        loop.close()
    except Exception:
        pass  # best-effort — don't crash on cleanup


def run_collector_with_retry(settings: Settings) -> None:
    """Run the Discord collector with exponential backoff on failures."""
    from options_owl.collectors.discord_collector import run_collector

    backoff = INITIAL_BACKOFF
    attempt = 0

    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            logger.info(f"Starting collector (attempt {attempt})…")
            write_heartbeat()
            asyncio.run(run_collector(settings))
            # Clean exit — shouldn't happen, but if it does, restart
            logger.warning("Collector exited cleanly — restarting in {backoff}s")
        except KeyboardInterrupt:
            logger.info("Collector stopped by user")
            _cleanup_connections()
            return
        except Exception as exc:
            logger.error(f"Collector crashed: {type(exc).__name__}: {exc}")
            logger.debug("Full traceback:", exc_info=True)
            _cleanup_connections()

        logger.info(f"Retrying in {backoff}s…")
        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF)

    logger.critical(f"Collector failed after {MAX_RETRIES} attempts — giving up")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OptionsOwl — Discord signal tracker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("collect", help="Run the Discord collector bot")
    sub.add_parser("backfill", help="Fetch market data and resolve unresolved signals")

    report_cmd = sub.add_parser("report", help="Print per-bot performance reports")
    report_cmd.add_argument("--compare-smee", action="store_true", help="Include Smee comparison")

    sub.add_parser("status", help="Show paper trading portfolio status")
    sub.add_parser("webull-test", help="Test Webull API connection and show account info")

    bt_cmd = sub.add_parser("backtest", help="Run backtest on historical paper trades")
    bt_cmd.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    bt_cmd.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    bt_cmd.add_argument("--balance", type=float, default=5000, help="Starting balance")
    bt_cmd.add_argument("--min-score", type=int, default=75, help="Minimum signal score")
    bt_cmd.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent positions")

    args = parser.parse_args()

    if args.command is None:
        # Default to collect for backward compat
        args.command = "collect"

    configure_logging(verbose=getattr(args, "verbose", False))
    settings = Settings()

    if args.command == "collect":
        if not settings.DISCORD_TOKEN:
            logger.error("DISCORD_TOKEN not set. Copy .env.example to .env and fill in your token.")
            return
        check_polygon_realtime_entitlement(settings)
        run_collector_with_retry(settings)

    elif args.command == "backfill":
        logger.info("Starting backfill…")
        asyncio.run(_backfill(settings))

    elif args.command == "report":
        asyncio.run(_report(settings))

    elif args.command == "status":
        asyncio.run(_status(settings))

    elif args.command == "webull-test":
        asyncio.run(_webull_test(settings))

    elif args.command == "backtest":
        asyncio.run(_backtest(settings, args))


async def _backfill(settings: Settings) -> None:
    from options_owl.collectors.market_data import fetch_bars_for_signal
    from options_owl.journal import db
    from options_owl.signals.outcome_resolver import resolve_signal

    await db.init_db(settings.DB_PATH)
    unresolved = await db.get_unresolved_signals(settings.DB_PATH)

    if not unresolved:
        logger.info("No unresolved signals to backfill.")
        return

    logger.info(f"Found {len(unresolved)} unresolved signals")

    for sig in unresolved:
        signal_id = sig["id"]
        ticker = sig["ticker"]

        # Determine date from created_at
        created = sig.get("created_at", "")
        try:
            sig_date = datetime.fromisoformat(created).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            from zoneinfo import ZoneInfo
            sig_date = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        logger.info(f"Fetching data for {ticker} (signal {signal_id}) on {sig_date}")

        bars = await fetch_bars_for_signal(ticker, sig_date)
        if not bars:
            logger.warning(f"No price data for {ticker} on {sig_date}, skipping")
            continue

        # Save price snapshots
        bar_dicts = [
            {
                "timestamp": b.timestamp.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        await db.save_price_snapshots(
            settings.DB_PATH, signal_id=signal_id, ticker=ticker, bars=bar_dicts
        )

        # Resolve outcome
        result = resolve_signal(sig, bars, signal_id)
        await db.save_signal_outcome(
            settings.DB_PATH,
            outcome=result.model_dump(mode="json"),
        )

        # Rate limit — be nice to yfinance
        await asyncio.sleep(1)

    logger.info("Backfill complete!")


async def _report(settings: Settings) -> None:
    from options_owl.journal import db
    from options_owl.signals.performance_tracker import compute_all_bots_performance, format_report

    await db.init_db(settings.DB_PATH)
    reports = await compute_all_bots_performance(settings.DB_PATH)
    output = format_report(reports)
    print(output)


async def _status(settings: Settings) -> None:
    from options_owl.execution.paper_trader import PaperTrader

    trader = PaperTrader(settings)
    await trader.init()
    print(await trader.get_status())


async def _webull_test(settings: Settings) -> None:
    from options_owl.execution.webull_executor import WebullExecutor

    executor = WebullExecutor(settings)
    try:
        account_id = await executor.init()
        print("\nWebull connection successful!")
        print(f"Account ID: {account_id}")

        info = await executor.get_account_info()
        print("\n=== Account Summary ===")
        print(f"  Total assets:  ${info.total_asset:,.2f}")
        print(f"  Cash balance:  ${info.cash_balance:,.2f}")
        print(f"  Buying power:  ${info.buying_power:,.2f}")
        print(f"  Positions:     {len(info.positions)}")

        if info.positions:
            print("\n=== Open Positions ===")
            for pos in info.positions:
                symbol = pos.get("symbol", pos.get("ticker", "?"))
                qty = pos.get("qty", pos.get("quantity", "?"))
                pnl = pos.get("unrealized_profit_loss", pos.get("unrealizedProfitLoss", "?"))
                print(f"  {symbol}: {qty} contracts, unrealized P&L: {pnl}")

        open_orders = await executor.get_open_orders()
        print(f"\nOpen orders: {len(open_orders)}")

        print(f"\nPAPER_TRADE is {'ON' if settings.PAPER_TRADE else 'OFF'}")
        if settings.PAPER_TRADE:
            print("  (set PAPER_TRADE=false in .env to enable live trading)")

    except Exception as exc:
        logger.error(f"Webull connection failed: {exc}")
        print(f"\nFailed: {exc}")
        print("\nCheck that WEBULL_APP_KEY and WEBULL_APP_SECRET are set correctly in .env")


async def _backtest(settings: Settings, args: argparse.Namespace) -> None:
    from options_owl.backtest.engine import BacktestConfig, BacktestEngine, load_historical_signals
    from options_owl.execution.paper_trader import init_paper_db
    from options_owl.journal import db

    await db.init_db(settings.DB_PATH)
    await init_paper_db(settings.DB_PATH)

    config = BacktestConfig(
        starting_balance=args.balance,
        max_position_pct=getattr(settings, "effective_max_position_pct", None) or settings.MAX_POSITION_PCT,
        max_concurrent=args.max_concurrent,
        min_score=args.min_score,
        start_date=args.start,
        end_date=args.end,
    )

    signals = await load_historical_signals(
        settings.DB_PATH,
        start_date=args.start,
        end_date=args.end,
    )

    if not signals:
        logger.warning("No closed paper trades found for the given date range.")
        return

    logger.info(f"Loaded {len(signals)} historical trades for backtesting")

    engine = BacktestEngine(config)
    engine.run(signals)
    print(engine.format_report())


if __name__ == "__main__":
    main()
