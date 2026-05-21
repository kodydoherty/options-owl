"""Paper trading engine — simulates trades against live signals with a virtual portfolio."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
from loguru import logger

from typing import TYPE_CHECKING

from options_owl.config.settings import Settings
from options_owl.models.signals import Direction, TradeSignal
from options_owl.risk.pipeline import run_entry_pipeline

if TYPE_CHECKING:
    from options_owl.execution.webull_executor import WebullExecutor

from options_owl.collectors.supabase_brain import SupabaseBrain
from options_owl.collectors.support_levels import is_at_support
from options_owl.journal.db import connect as _connect_db


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Prevents 'Task exception was never retrieved' warnings by attaching
    an error callback. Supabase writes should never block trade execution.
    """
    task = asyncio.create_task(coro)
    task.add_done_callback(_handle_task_exception)


def _handle_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning(f"[SupabaseBrain] Background task failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# DB retry + recovery queue
# ---------------------------------------------------------------------------

_SQLITE_RETRYABLE = (
    "database is locked",
    "disk I/O error",
    "busy",
)

RECOVERY_QUEUE_DIR = Path(os.getenv("JOURNAL_DIR", "journal")) / "recovery_queue"


async def _db_execute_with_retry(
    db_path: str,
    operations: list[tuple[str, tuple]],
    max_retries: int = 3,
    base_delay: float = 0.5,
    context: str = "",
) -> bool:
    """Execute a batch of SQL operations with retry + fallback to recovery queue.

    Args:
        db_path: Path to the SQLite database.
        operations: List of (sql, params) tuples to execute in a single transaction.
        max_retries: Number of retry attempts for transient errors.
        base_delay: Initial delay between retries (doubles each attempt).
        context: Human-readable label for logging (e.g. "open trade SPY").

    Returns True if committed successfully, False if queued for recovery.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with _connect_db(db_path) as conn:
                for sql, params in operations:
                    await conn.execute(sql, params)
                await conn.commit()
            return True
        except Exception as exc:
            last_exc = exc
            err_msg = str(exc).lower()
            if any(hint in err_msg for hint in _SQLITE_RETRYABLE):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"DB retry {attempt + 1}/{max_retries} for {context}: {exc} "
                    f"(retrying in {delay:.1f}s)"
                )
                await asyncio.sleep(delay)
            else:
                # Non-retryable error — skip straight to recovery queue
                break

    # All retries exhausted — write to recovery queue file
    logger.error(
        f"DB WRITE FAILED after {max_retries} retries for {context}: {last_exc} "
        f"— saving to recovery queue"
    )
    _enqueue_for_recovery(context, operations)
    return False


def _enqueue_for_recovery(context: str, operations: list[tuple[str, tuple]]) -> None:
    """Write failed DB operations to a JSON file for later replay."""
    RECOVERY_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "operations": [{"sql": sql, "params": list(params)} for sql, params in operations],
    }
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{context.replace(' ', '_')}.json"
    path = RECOVERY_QUEUE_DIR / filename
    path.write_text(json.dumps(entry, indent=2, default=str))
    logger.error(f"Recovery queue entry written: {path}")


async def replay_recovery_queue(db_path: str) -> int:
    """Replay any queued DB operations from previous failures. Returns count replayed."""
    if not RECOVERY_QUEUE_DIR.exists():
        return 0

    queue_files = sorted(RECOVERY_QUEUE_DIR.glob("*.json"))
    if not queue_files:
        return 0

    logger.info(f"Found {len(queue_files)} recovery queue entries — replaying")
    replayed = 0

    for qf in queue_files:
        try:
            entry = json.loads(qf.read_text())
            ops = [(op["sql"], tuple(op["params"])) for op in entry["operations"]]
            async with _connect_db(db_path) as conn:
                for sql, params in ops:
                    await conn.execute(sql, params)
                await conn.commit()
            qf.unlink()
            replayed += 1
            logger.info(f"Replayed recovery entry: {entry['context']} ({qf.name})")
        except Exception as exc:
            logger.error(f"Failed to replay {qf.name}: {exc} — leaving in queue")

    return replayed


# ---------------------------------------------------------------------------
# DB tables for paper trading
# ---------------------------------------------------------------------------

_CREATE_PAPER_PORTFOLIO = """
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    starting_balance REAL NOT NULL,
    current_balance REAL NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0,
    last_trade_date TEXT,
    created_at TEXT NOT NULL
);
"""

_CREATE_PAPER_TRADES = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    score INTEGER NOT NULL,
    strength TEXT NOT NULL,
    bot_source TEXT NOT NULL,

    entry_price REAL NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    premium_per_contract REAL NOT NULL,
    total_cost REAL NOT NULL,

    target_1 REAL,
    target_2 REAL,
    target_3 REAL,
    target_4 REAL,
    target_5 REAL,
    stop_price REAL,
    exit_by TEXT,
    expiry_date TEXT,

    signal_premium REAL,
    entry_slippage REAL,
    exit_slippage REAL,

    dca_tranches_remaining INTEGER DEFAULT 0,
    dca_total_contracts INTEGER DEFAULT 0,
    dca_last_add_at TEXT,

    status TEXT NOT NULL DEFAULT 'open',
    exit_price REAL,
    exit_premium REAL,
    exit_reason TEXT,
    pnl_dollars REAL,
    pnl_pct REAL,

    opened_at TEXT NOT NULL,
    closed_at TEXT
);
"""


_CREATE_TRADE_EVENTS = """
CREATE TABLE IF NOT EXISTS trade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    ticker TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
"""


async def log_trade_event(
    db_path: str,
    ticker: str,
    event_type: str,
    detail: str,
    trade_id: int | None = None,
) -> None:
    """Persist a trade lifecycle event to the DB for post-hoc analysis."""
    try:
        async with _connect_db(db_path) as conn:
            await conn.execute(
                "INSERT INTO trade_events (trade_id, ticker, event_type, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (trade_id, ticker, event_type, detail, datetime.now().isoformat()),
            )
            await conn.commit()
    except Exception:
        pass  # never let audit logging break the trade flow


def _today_et() -> datetime:
    """Return current datetime in US/Eastern (works in Docker where TZ may be UTC)."""
    from zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("America/New_York"))


def resolve_expiry_date(expiry: str | None) -> str | None:
    """Convert an expiry label like '0DTE', '1DTE', 'today', 'tomorrow', 'friday'
    into a YYYY-MM-DD date string.

    Returns None if the expiry string is unrecognised.
    """
    if not expiry:
        return None
    expiry_upper = expiry.upper().strip()
    # Already a date?
    if len(expiry_upper) == 10 and expiry_upper[4] == "-":
        return expiry_upper
    # nDTE pattern
    if expiry_upper.endswith("DTE"):
        try:
            days = int(expiry_upper.replace("DTE", ""))
            return (_today_et() + timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Natural language dates
    today = _today_et()
    if expiry_upper in ("TODAY", "SAME DAY", "INTRADAY"):
        return today.strftime("%Y-%m-%d")
    if expiry_upper in ("TOMORROW", "NEXT DAY"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    # Day-of-week names (e.g., "Friday", "Monday")
    _DAY_NAMES = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
                  "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
    if expiry_upper in _DAY_NAMES:
        target_dow = _DAY_NAMES[expiry_upper]
        current_dow = today.weekday()
        days_ahead = (target_dow - current_dow) % 7
        # Same day = today (e.g., "FRIDAY" on a Friday = 0DTE)
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    return None


async def init_paper_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    async with _connect_db(path) as conn:
        await conn.execute(_CREATE_PAPER_PORTFOLIO)
        await conn.execute(_CREATE_PAPER_TRADES)
        await conn.execute(_CREATE_TRADE_EVENTS)
        # Migration: add expiry_date column if missing (existing DBs)
        try:
            await conn.execute("ALTER TABLE paper_trades ADD COLUMN expiry_date TEXT")
        except Exception:
            pass  # column already exists
        # Migration: add parent_trade_id column for partial profit-taking
        try:
            await conn.execute("ALTER TABLE paper_trades ADD COLUMN parent_trade_id INTEGER")
        except Exception:
            pass  # column already exists
        # Migration: add slippage tracking columns
        for col in ("signal_premium REAL", "entry_slippage REAL", "exit_slippage REAL"):
            try:
                await conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col}")
            except Exception:
                pass  # column already exists
        # Migration: add strategy column to paper_trades
        try:
            await conn.execute("ALTER TABLE paper_trades ADD COLUMN strategy TEXT DEFAULT 'B'")
        except Exception:
            pass
        # Migration: add strategy column to paper_portfolio
        try:
            await conn.execute("ALTER TABLE paper_portfolio ADD COLUMN strategy TEXT DEFAULT 'B'")
        except Exception:
            pass
        # Migration: add last_target_hit for graduated scale-out
        try:
            await conn.execute("ALTER TABLE paper_trades ADD COLUMN last_target_hit INTEGER DEFAULT 0")
        except Exception:
            pass  # column already exists
        # Migration: add duration_minutes for backtest tracking
        try:
            await conn.execute("ALTER TABLE paper_trades ADD COLUMN duration_minutes REAL")
        except Exception:
            pass
        # Migration: add MFE/MAE columns for excursion tracking
        for col in (
            "mfe_premium REAL",
            "mae_premium REAL",
            "mfe_pnl_pct REAL",
            "mae_pnl_pct REAL",
        ):
            try:
                await conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col}")
            except Exception:
                pass  # column already exists
        # Migration: add Webull order tracking columns
        for col in (
            "webull_order_id TEXT",
            "webull_client_order_id TEXT",
            "webull_entry_fill_price REAL",
            "webull_exit_fill_price REAL",
            "webull_exit_order_id TEXT",
        ):
            try:
                await conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col}")
            except Exception:
                pass  # column already exists
        # Migration: add ENRG result tracking
        try:
            await conn.execute(
                "ALTER TABLE paper_trades ADD COLUMN enrg_result TEXT"
            )
        except Exception:
            pass  # column already exists
        # Migration: add sell retry tracking
        for col in (
            "sell_retry_count INTEGER DEFAULT 0",
            "sell_last_attempted_price REAL",
        ):
            try:
                await conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col}")
            except Exception:
                pass  # column already exists
        # Migration: add exit_source tracking (ai / manual / expired)
        try:
            await conn.execute(
                "ALTER TABLE paper_trades ADD COLUMN exit_source TEXT DEFAULT 'ai'"
            )
        except Exception:
            pass  # column already exists
        # Migration: add Supabase alert_id for shared brain integration
        try:
            await conn.execute(
                "ALTER TABLE paper_trades ADD COLUMN supabase_alert_id TEXT"
            )
        except Exception:
            pass  # column already exists
        await conn.commit()


async def _get_or_create_portfolio(
    conn: aiosqlite.Connection, balance: float, strategy: str = "B",
) -> dict:
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(
        "SELECT * FROM paper_portfolio WHERE strategy = ? ORDER BY id DESC LIMIT 1",
        (strategy,),
    )
    row = await cursor.fetchone()
    if row:
        return dict(row)
    await conn.execute(
        "INSERT INTO paper_portfolio (starting_balance, current_balance, strategy, created_at) "
        "VALUES (?, ?, ?, ?)",
        (balance, balance, strategy, datetime.now().isoformat()),
    )
    await conn.commit()
    cursor = await conn.execute(
        "SELECT * FROM paper_portfolio WHERE strategy = ? ORDER BY id DESC LIMIT 1",
        (strategy,),
    )
    return dict(await cursor.fetchone())  # type: ignore[arg-type]


async def get_open_trades(path: str, strategy: str | None = None) -> list[dict]:
    async with _connect_db(path) as conn:
        conn.row_factory = aiosqlite.Row
        if strategy:
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'open' AND strategy = ?",
                (strategy,),
            )
        else:
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE status = 'open'")
        return [dict(r) for r in await cursor.fetchall()]


async def get_portfolio(path: str, balance: float, strategy: str = "B") -> dict:
    async with _connect_db(path) as conn:
        return await _get_or_create_portfolio(conn, balance, strategy)


_DAILY_0DTE = {"SPY", "QQQ"}
_MWF_0DTE = {"NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AVGO"}
_WEEKLY_ONLY = {"AMD", "PLTR", "MSTR"}


def _build_expiry_candidates(ticker: str, expiry_date: str) -> list[str]:
    """Build ordered list of expiry dates to try based on ticker's options schedule.

    SPY/QQQ: daily 0DTE.
    NVDA, TSLA etc: 0DTE Mon/Wed/Fri, next-day on Tue/Thu.
    AMD, PLTR, MSTR: weekly only (Friday expiry).
    Unknown: try next 2 business days + Friday.
    """
    candidates = [expiry_date]
    base = datetime.strptime(expiry_date, "%Y-%m-%d").date()
    ticker_upper = ticker.upper()
    weekday = base.weekday()  # 0=Mon, 4=Fri

    if ticker_upper in _DAILY_0DTE:
        pass  # daily expirations, today is always valid
    elif ticker_upper in _MWF_0DTE:
        if weekday in (1, 3):  # Tue/Thu → add next day
            next_day = base + timedelta(days=1)
            candidates.append(next_day.strftime("%Y-%m-%d"))
    elif ticker_upper in _WEEKLY_ONLY:
        if weekday < 4:  # Mon-Thu → add this Friday
            friday = base + timedelta(days=(4 - weekday))
            candidates.append(friday.strftime("%Y-%m-%d"))
    else:
        # Unknown ticker — try next 2 business days + Friday
        for delta in range(1, 4):
            candidate = base + timedelta(days=delta)
            if candidate.weekday() < 5:
                candidates.append(candidate.strftime("%Y-%m-%d"))
            if len(candidates) >= 3:
                break
        if weekday < 4:
            friday = base + timedelta(days=(4 - weekday))
            fri_str = friday.strftime("%Y-%m-%d")
            if fri_str not in candidates:
                candidates.append(fri_str)

    return candidates


async def _verify_live_premium(
    signal: TradeSignal, settings: Settings,
) -> tuple[TradeSignal, str, dict | None]:
    """Fetch a live option quote and decide whether to trade and at what price.

    Returns (updated_signal, decision_reason, nbbo_at_order_time).
    nbbo_at_order_time is a dict with bid/ask/mid if available (for NBBO logging).

    Decision matrix:
    - Live quote within SMART_ENTRY_MAX_DEVIATION_PCT of signal → use live quote
    - Live quote cheaper than signal → use live quote (better deal)
    - Live quote more expensive but within tolerance → use live quote (pay up)
    - Live quote way off (> deviation %) → reject the trade
    - Live quote unavailable → fall back to signal premium with warning
    """
    import asyncio

    from options_owl.execution.position_monitor import (
        _fetch_option_chain_for_ticker,
        _lookup_premium_from_chain,
    )

    if not getattr(settings, "ENABLE_SMART_ENTRY", True):
        return signal, "smart_entry_disabled", None

    signal_premium = signal.atm_premium
    if not signal_premium or signal_premium <= 0:
        return signal, "no_signal_premium", None

    option_type = "put" if signal.direction == Direction.PUT else "call"

    # Resolve expiry to YYYY-MM-DD
    expiry_date = resolve_expiry_date(signal.expiry)
    if not expiry_date:
        logger.debug(f"[SmartEntry] {signal.ticker}: no expiry date, using signal premium")
        return signal, "no_expiry_date", None

    from options_owl.collectors.polygon_options import polygon_option_quote

    polygon_key = getattr(settings, "POLYGON_API_KEY", "") or ""
    live_premium = None
    nbbo: dict | None = None  # bid/ask/mid at order time for NBBO logging
    expiry_candidates = _build_expiry_candidates(signal.ticker, expiry_date)

    # Fast path: Polygon option quote with bid/ask (async, ~1-2s)
    used_expiry = expiry_date
    if polygon_key:
        for try_expiry in expiry_candidates:
            try:
                quote = await polygon_option_quote(
                    polygon_key, signal.ticker, signal.strike, try_expiry, option_type,
                )
                if quote:
                    if try_expiry != expiry_date:
                        logger.info(
                            f"[SmartEntry] {signal.ticker}: no 0DTE contract for {expiry_date}, "
                            f"using {try_expiry} instead"
                        )
                    used_expiry = try_expiry
                    expiry_date = try_expiry
                    break
            except Exception as exc:
                logger.debug(f"[SmartEntry] Polygon lookup for {try_expiry} failed: {exc}")
                quote = None
        else:
            quote = None

        if quote:
            # For BUY orders: use the ask price (what we actually pay)
            # For SELL orders: use the bid price (what we actually receive)
            # Fall back to mid if bid/ask not available
            ask = quote.get("ask", 0)
            bid = quote.get("bid", 0)
            mid = quote.get("mid", 0)

            # Capture NBBO for logging (always, even if we reject later)
            nbbo = {"bid": bid, "ask": ask, "mid": mid}

            if ask > 0:
                # Use ask + 5% buffer to ensure fill on fast-moving options
                buffer_pct = getattr(settings, "SMART_ENTRY_ASK_BUFFER_PCT", 5.0)
                buffered = round(ask * (1 + buffer_pct / 100), 2)
                live_premium = buffered
                logger.info(
                    f"[SmartEntry] {signal.ticker}: ASK=${ask:.2f} "
                    f"+{buffer_pct}% buffer → ${buffered:.2f} "
                    f"(bid=${bid:.2f}, mid=${mid:.2f}, exp={used_expiry})"
                )
            elif mid > 0:
                live_premium = mid
                logger.info(
                    f"[SmartEntry] {signal.ticker}: no ask available, "
                    f"using mid=${mid:.2f} (exp={used_expiry})"
                )

    # Slow fallback: yfinance option chain (sync, can take 30s+)
    chain = None
    if not live_premium:
        for try_expiry in expiry_candidates:
            try:
                chain = await asyncio.to_thread(
                    _fetch_option_chain_for_ticker, signal.ticker, try_expiry,
                )
                if chain:
                    if try_expiry != expiry_candidates[0]:
                        logger.info(
                            f"[SmartEntry] {signal.ticker}: chain found for {try_expiry} "
                            f"(not {expiry_candidates[0]})"
                        )
                    used_expiry = try_expiry
                    expiry_date = try_expiry
                    break
            except Exception as exc:
                logger.warning(f"[SmartEntry] {signal.ticker}: chain fetch for {try_expiry} failed: {exc}")

        if not chain:
            # No option chain for any expiry candidate.
            if not getattr(settings, "PAPER_TRADE", True):
                logger.warning(
                    f"[SmartEntry] {signal.ticker}: no option chain for "
                    f"expiries {expiry_candidates} "
                    f"— BLOCKING trade (live mode, contract may not exist)"
                )
                return signal.model_copy(update={"atm_premium": 0}), "no_chain_blocked_live", nbbo
            logger.warning(
                f"[SmartEntry] {signal.ticker}: no option chain available, "
                f"using signal premium ${signal_premium:.2f}"
            )
            return signal, "no_chain_available", nbbo

        # Look up the exact strike
        live_premium = _lookup_premium_from_chain(chain, signal.strike, option_type)

    if not live_premium or live_premium <= 0:
        # Try nearby strikes (within $1) if exact match not found
        live_premium = _find_nearby_strike_premium(
            chain, signal.strike, option_type, max_distance=1.0,
        )

    if not live_premium or live_premium <= 0:
        logger.warning(
            f"[SmartEntry] {signal.ticker} ${signal.strike} {option_type}: "
            f"no live quote found, using signal premium ${signal_premium:.2f}"
        )
        return signal, "no_live_quote", nbbo

    # Minimum premium check
    min_prem = getattr(settings, "SMART_ENTRY_MIN_PREMIUM", 0.10)
    if live_premium < min_prem:
        logger.info(
            f"[SmartEntry] REJECT {signal.ticker}: live premium ${live_premium:.2f} "
            f"below minimum ${min_prem:.2f}"
        )
        return signal.model_copy(update={"atm_premium": 0.0}), "live_premium_too_low", nbbo

    # Calculate deviation from signal
    deviation_pct = (live_premium - signal_premium) / signal_premium * 100
    max_dev = getattr(settings, "SMART_ENTRY_MAX_DEVIATION_PCT", 30.0)

    # Decision logic
    # Always allow cheaper entries (negative deviation = better deal for us)
    if deviation_pct <= 0 or abs(deviation_pct) <= max_dev:
        # Within tolerance or cheaper — use live premium
        use_live = getattr(settings, "SMART_ENTRY_PREFER_LIVE", True)
        trade_premium = live_premium if use_live else signal_premium

        direction = "cheaper" if deviation_pct < 0 else "pricier"
        expiry_note = ""
        updates: dict = {"atm_premium": trade_premium}
        # If we found a different expiry than the original 0DTE, update the signal
        original_expiry = resolve_expiry_date(signal.expiry)
        if used_expiry != original_expiry:
            updates["expiry"] = used_expiry  # store as YYYY-MM-DD so resolve_expiry_date passes through
            expiry_note = f" [expiry adjusted: {original_expiry} → {used_expiry}]"

        logger.info(
            f"[SmartEntry] {signal.ticker} ${signal.strike} {option_type}: "
            f"signal=${signal_premium:.2f} → live=${live_premium:.2f} "
            f"({deviation_pct:+.1f}%, {direction}) — using ${trade_premium:.2f}"
            f"{expiry_note}"
        )
        updated = signal.model_copy(update=updates)
        reason = f"live_verified:{deviation_pct:+.1f}%"
        if expiry_note:
            reason += f" exp={used_expiry}"
        return updated, reason, nbbo
    else:
        # Live is more expensive than signal beyond tolerance — reject
        logger.warning(
            f"[SmartEntry] REJECT {signal.ticker}: signal=${signal_premium:.2f} "
            f"vs live=${live_premium:.2f} ({deviation_pct:+.1f}% deviation, "
            f"max allowed +{max_dev:.0f}%)"
        )
        return signal.model_copy(update={"atm_premium": 0.0}), f"deviation_too_large:{deviation_pct:+.1f}%", nbbo


def _find_nearby_strike_premium(
    chain: dict, target_strike: float, option_type: str, max_distance: float = 1.0,
) -> float | None:
    """Find the closest available strike within max_distance and return its premium.

    Useful when the exact strike from the signal isn't in the chain
    (e.g., signal says $560 but chain has $559.5 and $560.5).
    """
    import math

    df = chain.get("calls" if option_type == "call" else "puts")
    if df is None or df.empty:
        return None

    # Find strikes within max_distance
    nearby = df[abs(df["strike"] - target_strike) <= max_distance].copy()
    if nearby.empty:
        return None

    # Sort by distance from target
    nearby = nearby.assign(_dist=abs(nearby["strike"] - target_strike))
    nearby = nearby.sort_values("_dist")
    row = nearby.iloc[0]

    actual_strike = row["strike"]

    bid = row.get("bid")
    ask = row.get("ask")

    if (
        bid is not None and ask is not None
        and not (isinstance(bid, float) and math.isnan(bid))
        and not (isinstance(ask, float) and math.isnan(ask))
        and bid > 0 and ask > 0
    ):
        premium = round((bid + ask) / 2.0, 2)
    else:
        last = row.get("lastPrice")
        if last is not None and not (isinstance(last, float) and math.isnan(last)) and last > 0:
            premium = round(float(last), 2)
        else:
            return None

    if actual_strike != target_strike:
        logger.info(
            f"[SmartEntry] Nearby strike: wanted ${target_strike} → "
            f"found ${actual_strike} @ ${premium:.2f}"
        )

    return premium


def _select_trade_premium(signal: TradeSignal) -> TradeSignal:
    """Pick the correct premium for the option the bot will actually trade.

    Discord signals provide two options:
      * atm_premium / atm_strike — the ATM (conservative) pick
      * otm_premium / otm_strike — the OTM (primary) pick

    The parser now correctly assigns labels based on the Discord message order.
    This function acts as a safety net: it picks the premium matching the
    signal's recommended strike, and rejects penny options (<$0.05).
    """
    strike = signal.strike
    entry_price = signal.entry_price or 0
    atm_prem = signal.atm_premium or 0
    otm_prem = signal.otm_premium or 0
    atm_strike = signal.atm_strike or 0
    otm_strike = signal.otm_strike or 0

    # Case 1: Signal's recommended strike matches OTM strike → use OTM premium
    # But only if ATM premium is a penny option OR OTM is cheaper (the normal case).
    # If OTM premium is MORE expensive than ATM, it's likely an ITM option mislabeled.
    if (
        otm_prem > 0
        and otm_strike
        and strike
        and abs(otm_strike - strike) < 0.01
    ):
        atm_is_penny = atm_prem < 0.05
        otm_is_cheaper = otm_prem <= atm_prem
        if otm_prem >= 0.05 and (atm_is_penny or otm_is_cheaper):
            logger.debug(
                f"{signal.ticker}: strike ${strike} matches OTM strike, "
                f"using OTM premium ${otm_prem:.2f}"
            )
            return signal.model_copy(
                update={"atm_premium": otm_prem, "atm_strike": otm_strike}
            )

    # Case 2: Signal's recommended strike matches ATM strike → use ATM premium
    if (
        atm_prem > 0
        and atm_strike
        and strike
        and abs(atm_strike - strike) < 0.01
    ):
        if atm_prem >= 0.05:  # not a penny option
            return signal  # already using the right premium

    # Case 3: Neither matches — pick the strike closest to entry_price (true ATM)
    if entry_price > 0 and atm_strike and otm_strike:
        atm_dist = abs(atm_strike - entry_price)
        otm_dist = abs(otm_strike - entry_price)

        if otm_dist < atm_dist and otm_prem >= 0.05:
            # OTM strike is actually closer to ATM (labels are swapped)
            logger.info(
                f"{signal.ticker}: labels swapped — OTM strike ${otm_strike} "
                f"is closer to entry ${entry_price:.2f} than ATM strike ${atm_strike}. "
                f"Using OTM premium ${otm_prem:.2f} instead of ${atm_prem:.4f}"
            )
            return signal.model_copy(
                update={"atm_premium": otm_prem, "atm_strike": otm_strike}
            )
        elif atm_dist <= otm_dist and atm_prem >= 0.05:
            return signal  # ATM label is correct

    # Case 4: Only one premium is reasonable (>$0.05), use it
    if atm_prem < 0.05 and otm_prem >= 0.05:
        logger.info(
            f"{signal.ticker}: ATM premium ${atm_prem:.4f} is a penny option, "
            f"switching to OTM premium ${otm_prem:.2f}"
        )
        return signal.model_copy(
            update={"atm_premium": otm_prem, "atm_strike": otm_strike}
        )
    if otm_prem < 0.05 and atm_prem >= 0.05:
        return signal  # ATM premium is the good one

    return signal


class PaperTrader:
    """Evaluates signals and simulates paper trades."""

    def __init__(
        self, settings: Settings,
        webull_executor: WebullExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = settings.DB_PATH
        self.webull_executor = webull_executor
        self.market_stream = None  # set by discord_collector after market stream starts
        self.supabase: SupabaseBrain | None = None  # set by discord_collector
        self._signal_engine = None
        self._cached_live_balance: float | None = None
        self._balance_cache_ts: float = 0.0
        # GFV in-memory tracker: avoids DB query on every trade entry
        self._gfv_daily_spent: float = 0.0
        self._gfv_day: str = ""
        # Start-of-day balance for GFV limit (locked once per day, before any trades)
        self._gfv_start_balance: float = 0.0

    async def get_portfolio_balance(self) -> float:
        """Return the current portfolio balance from the DB.

        Lightweight read — no Webull API calls. Uses the balance maintained by
        sync_portfolio_from_webull() and trade open/close updates. Returns 0
        if the DB hasn't been initialized yet.
        """
        try:
            async with _connect_db(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT current_balance FROM paper_portfolio "
                    "WHERE strategy = 'B' LIMIT 1"
                )
                row = await cursor.fetchone()
                if row and row[0] and row[0] > 0:
                    return float(row[0])
        except Exception:
            pass
        return 0.0

    async def _get_effective_balance(self) -> float:
        """Fetch live Webull balance for position sizing.

        Uses the real Webull account balance (net liquidation value) when
        available. On API error, falls back to the last synced balance from
        the DB (from a previous sync_portfolio_from_webull call). Only uses
        PORTFOLIO_SIZE as a last resort if the DB has never been synced.
        """
        import time

        env_fallback = self.settings.PORTFOLIO_SIZE

        # Only fetch live balance when trading live with a real executor
        if self.webull_executor is None or self.settings.PAPER_TRADE:
            return env_fallback

        # Use cache if fresh (within 30s)
        now = time.time()
        if self._cached_live_balance is not None and (now - self._balance_cache_ts) < 30:
            return self._cached_live_balance

        try:
            live_balance = await self.webull_executor.get_account_balance()
            if live_balance <= 0:
                # Balance query returned 0 — fall back to buying power
                account_info = await self.webull_executor.get_account_info()
                live_balance = account_info.buying_power

            if live_balance > 0:
                logger.info(
                    f"Live Webull balance: ${live_balance:,.2f} "
                    f"(PORTFOLIO_SIZE fallback=${env_fallback:,.2f})"
                )
                self._cached_live_balance = live_balance
                self._balance_cache_ts = now
                return live_balance
            else:
                logger.warning(
                    "Webull balance is $0 — trying last synced DB balance"
                )
                return await self._get_db_fallback_balance(env_fallback)
        except Exception as exc:
            logger.warning(
                f"Failed to fetch live Webull balance: {exc} — "
                f"trying last synced DB balance"
            )
            return await self._get_db_fallback_balance(env_fallback)

    async def _get_db_fallback_balance(self, env_fallback: float) -> float:
        """Read the last synced balance from paper_portfolio DB.

        This is the balance written by sync_portfolio_from_webull() and
        reflects the real Webull account value from the last successful sync.
        Only falls back to PORTFOLIO_SIZE if the DB has never been synced.
        """
        try:
            async with _connect_db(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT current_balance FROM paper_portfolio "
                    "WHERE strategy = 'B' LIMIT 1"
                )
                row = await cursor.fetchone()
                if row and row[0] and row[0] > 0:
                    db_balance = float(row[0])
                    logger.info(
                        f"Using last synced DB balance: ${db_balance:,.2f} "
                        f"(PORTFOLIO_SIZE env=${env_fallback:,.2f})"
                    )
                    self._cached_live_balance = db_balance
                    import time
                    self._balance_cache_ts = time.time()
                    return db_balance
        except Exception as db_exc:
            logger.warning(f"Failed to read DB balance: {db_exc}")

        logger.warning(
            f"No synced balance in DB — using PORTFOLIO_SIZE=${env_fallback:,.2f}"
        )
        return env_fallback

    async def _get_unsettled_proceeds(self) -> float:
        """Calculate unsettled sale proceeds to prevent Good Faith Violations.

        Options settle T+1 in cash accounts. A GFV occurs when you buy
        with unsettled funds (proceeds from today's sales) and then sell
        that position before the original funds settle.

        Simple protection: track total $ already spent on Webull buys today.
        If that exceeds the starting settled balance, the excess was funded
        by unsettled sale proceeds — don't deploy more.

        Returns the amount by which today's buys exceed settled funds.
        Zero means we're still within settled cash.
        """
        try:
            today_str = _today_et().strftime("%Y-%m-%d")
            async with _connect_db(self.db_path) as conn:
                # Total capital deployed on buys today (includes open + closed)
                cursor = await conn.execute(
                    "SELECT COALESCE(SUM(premium_per_contract * contracts * 100), 0) "
                    "FROM paper_trades "
                    "WHERE date(opened_at) = ? AND webull_order_id IS NOT NULL",
                    (today_str,),
                )
                total_bought = float((await cursor.fetchone())[0])

                # Get starting balance (set by portfolio sync at start of day)
                cursor = await conn.execute(
                    "SELECT starting_balance FROM paper_portfolio WHERE strategy = 'B'"
                )
                row = await cursor.fetchone()
                starting_balance = float(row[0]) if row and row[0] else 0.0

            if starting_balance <= 0:
                # No starting balance recorded — can't calculate, skip protection
                return 0.0

            # If total buys exceed starting balance, we're using unsettled funds
            excess = total_bought - starting_balance
            if excess > 0:
                logger.info(
                    f"GFV: today's buys=${total_bought:.2f} exceed "
                    f"starting_balance=${starting_balance:.2f} by ${excess:.2f} "
                    f"— using unsettled proceeds"
                )
                return excess

            return 0.0

        except Exception as exc:
            logger.warning(f"GFV unsettled proceeds check failed: {exc}")
            return 0.0

    async def sync_portfolio_from_webull(self) -> float | None:
        """Fetch live Webull balance and update PORTFOLIO_SIZE + paper portfolio.

        Called once per trading day so position sizing, risk calculations,
        and loss limits always reflect the real account.
        Returns the synced balance or None if unavailable.
        """
        if not getattr(self.settings, "ENABLE_PORTFOLIO_SYNC", False):
            return None
        if self.webull_executor is None or self.settings.PAPER_TRADE:
            return None

        try:
            live_balance = await self.webull_executor.get_account_balance()
            if live_balance <= 0:
                info = await self.webull_executor.get_account_info()
                live_balance = info.buying_power

            if live_balance <= 0:
                logger.warning("Webull balance is $0 — skipping portfolio sync")
                return None

            old_size = self.settings.PORTFOLIO_SIZE
            self.settings.PORTFOLIO_SIZE = live_balance
            logger.info(
                f"Portfolio synced from Webull: ${old_size:,.2f} → ${live_balance:,.2f}"
            )

            # Update paper portfolio to match reality.
            # Set starting_balance = live (daily baseline for P&L tracking).
            # Set current_balance = live (reset to real account state).
            async with _connect_db(self.db_path) as conn:
                await conn.execute(
                    "UPDATE paper_portfolio SET starting_balance = ?, "
                    "current_balance = ? WHERE strategy = 'B'",
                    (live_balance, live_balance),
                )
                await conn.commit()
                logger.info(
                    f"Paper portfolio synced to ${live_balance:,.2f} (Webull)"
                )

            return live_balance
        except Exception as exc:
            logger.warning(f"Failed to sync portfolio from Webull: {exc}")
            return None

    async def init(self) -> None:
        await init_paper_db(self.db_path)
        # Replay any queued DB writes from previous crashes
        replayed = await replay_recovery_queue(self.db_path)
        if replayed:
            logger.info(f"Replayed {replayed} recovery queue entries on startup")

        # Lazily import to avoid circular deps
        from options_owl.signals.engine import SignalEngine

        self._signal_engine = SignalEngine(self.settings)

    async def _commit_with_retry(
        self,
        conn: aiosqlite.Connection,
        context: str = "",
        max_retries: int = 3,
    ) -> bool:
        """Retry conn.commit() on transient SQLite errors.

        Returns True on success. On persistent failure, queues the
        pending SQL to the recovery queue file and returns False.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                await conn.commit()
                return True
            except Exception as exc:
                last_exc = exc
                err_msg = str(exc).lower()
                if any(hint in err_msg for hint in _SQLITE_RETRYABLE) and attempt < max_retries - 1:
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"DB commit retry {attempt + 1}/{max_retries} for {context}: "
                        f"{exc} (retrying in {delay:.1f}s)"
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        logger.error(
            f"DB COMMIT FAILED after {max_retries} retries for {context}: {last_exc}"
        )
        return False

    def should_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """Decide whether to paper trade this signal. Returns (should_trade, reason)."""
        if signal.score < self.settings.MIN_SCORE:
            return False, f"Score {signal.score} < min {self.settings.MIN_SCORE}"

        if not signal.atm_premium or signal.atm_premium <= 0:
            return False, "No ATM premium available"

        if not signal.stop_price:
            return False, "No stop price defined"

        return True, "Signal meets criteria"

    async def _open_single_trade(
        self,
        signal: TradeSignal,
        signal_id: int,
        strategy: str,
        supabase_alert_id: str | None = None,
        nbbo_at_order: dict | None = None,
    ) -> dict | None:
        """Open a single paper trade for a given strategy. Returns trade info or None."""
        async with _connect_db(self.db_path) as conn:
            portfolio = await _get_or_create_portfolio(
                conn, self.settings.PORTFOLIO_SIZE, strategy,
            )

            # ── PDT compliance check (margin accounts only) ───────────────
            # Pattern Day Trader rule: need $25K equity at previous close.
            # If balance is below $25K, block ALL new trades to stay compliant.
            PDT_MINIMUM = 25_000.0
            if self.settings.MARGIN_ACCOUNT and not self.settings.PAPER_TRADE:
                live_bal = await self._get_effective_balance()
                if live_bal < PDT_MINIMUM:
                    logger.error(
                        f"PDT BLOCK: margin account balance ${live_bal:,.2f} < "
                        f"${PDT_MINIMUM:,.2f} minimum — blocking trade to stay PDT compliant. "
                        f"Deposit funds or switch to cash account."
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "rejected",
                        f"pdt_block: balance=${live_bal:,.2f} < ${PDT_MINIMUM:,.2f}",
                    )
                    return None

            # ── Hard capital checks ───────────────────────────────────────
            # 1. Check total cost of open positions against portfolio size
            cursor = await conn.execute(
                "SELECT COALESCE(SUM(total_cost), 0) FROM paper_trades "
                "WHERE status = 'open' AND webull_order_id IS NOT NULL"
            )
            deployed_cost = (await cursor.fetchone())[0]
            portfolio_size = self.settings.PORTFOLIO_SIZE

            if deployed_cost >= portfolio_size:
                logger.warning(
                    f"CAPITAL BLOCK: deployed=${deployed_cost:.2f} >= "
                    f"portfolio=${portfolio_size:.2f} — cannot open new trade"
                )
                await log_trade_event(
                    self.db_path, signal.ticker, "rejected",
                    f"capital_block: deployed=${deployed_cost:.2f} >= portfolio=${portfolio_size:.2f}",
                )
                return None

            # 2. For live trading, get FRESH Webull buying power (no cache)
            webull_buying_power = None
            if self.webull_executor and not self.settings.PAPER_TRADE:
                try:
                    info = await self.webull_executor.get_account_info()
                    webull_buying_power = info.buying_power
                    logger.info(
                        f"CAPITAL CHECK: Webull buying_power=${webull_buying_power:.2f}, "
                        f"deployed=${deployed_cost:.2f}, portfolio={portfolio_size:.2f}"
                    )
                    if webull_buying_power <= 0:
                        logger.warning(
                            f"CAPITAL BLOCK: Webull buying power is ${webull_buying_power:.2f} "
                            f"— cannot open new trade"
                        )
                        await log_trade_event(
                            self.db_path, signal.ticker, "rejected",
                            f"capital_block: webull_buying_power=${webull_buying_power:.2f}",
                        )
                        return None
                except Exception as exc:
                    logger.warning(f"Failed to check Webull buying power: {exc}")

            # Position sizing — use the LOWER of paper balance and Webull buying power
            live_cap = await self._get_effective_balance()
            effective_balance = min(
                portfolio["current_balance"],
                live_cap,
            )
            # Hard cap at remaining portfolio capacity
            remaining_capacity = portfolio_size - deployed_cost
            effective_balance = min(effective_balance, remaining_capacity)
            # Hard cap at Webull buying power (the real constraint for cash accounts)
            if webull_buying_power is not None:
                effective_balance = min(effective_balance, webull_buying_power)

            # GFV protection: HARD BLOCK when daily buys exceed starting balance.
            # Options settle T+1 in cash accounts. If total $ bought today exceeds
            # the settled starting balance, any new buy would use unsettled proceeds
            # from today's sales — risking a Good Faith Violation on Webull.
            # 3 GFVs in 12 months = settled-cash-only restriction.
            # SKIP for margin accounts — no unsettled fund concerns.
            # 5 GFVs = account closure.
            #
            # Uses in-memory tracker seeded from DB once per day — no DB query
            # on the hot path so we don't lose trade windows.
            # GFV limit uses the START-OF-DAY balance (locked before any trades).
            # CRITICAL: Do NOT use live Webull balance here — it includes unsettled
            # proceeds from today's sales, which inflates the limit and allows
            # buying more than settled cash (the exact thing that causes GFVs).
            if not self.settings.PAPER_TRADE and not self.settings.MARGIN_ACCOUNT:
                today_str = _today_et().strftime("%Y-%m-%d")
                # Seed from DB on first call each day (handles restarts mid-day)
                if self._gfv_day != today_str:
                    # Lock start-of-day balance BEFORE it gets inflated by
                    # today's sale proceeds. On first trade of the day this is
                    # accurate. On mid-day restart, subtract today's net P&L
                    # from current balance to approximate start-of-day.
                    try:
                        async with _connect_db(self.db_path) as gfv_conn:
                            # Total $ bought today (for the spent tracker)
                            cursor = await gfv_conn.execute(
                                "SELECT COALESCE(SUM(premium_per_contract * contracts * 100), 0) "
                                "FROM paper_trades "
                                "WHERE date(opened_at) = ? AND webull_order_id IS NOT NULL",
                                (today_str,),
                            )
                            self._gfv_daily_spent = float((await cursor.fetchone())[0])

                            # Today's realized P&L (to back out from current balance)
                            cursor = await gfv_conn.execute(
                                "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                                "WHERE status = 'closed' AND date(opened_at) = ? "
                                "AND webull_order_id IS NOT NULL",
                                (today_str,),
                            )
                            today_pnl = float((await cursor.fetchone())[0])
                    except Exception:
                        self._gfv_daily_spent = 0.0
                        today_pnl = 0.0

                    # Start-of-day balance = current balance - today's realized P&L
                    # This removes the unsettled proceeds from the limit.
                    # Apply safety buffer (default 15%) so we never get close to GFV.
                    raw_sod = max(live_cap - today_pnl, portfolio_size)
                    gfv_buffer = self.settings.GFV_BUFFER_PCT / 100
                    self._gfv_start_balance = raw_sod * (1 - gfv_buffer)
                    self._gfv_day = today_str
                    logger.info(
                        f"GFV tracker seeded: ${self._gfv_daily_spent:.0f} bought today, "
                        f"start-of-day balance=${raw_sod:.0f}, "
                        f"GFV limit=${self._gfv_start_balance:.0f} "
                        f"({100 - self.settings.GFV_BUFFER_PCT:.0f}% after {self.settings.GFV_BUFFER_PCT:.0f}% buffer) "
                        f"(live=${live_cap:.0f} - today_pnl=${today_pnl:.0f})"
                    )

                # Fast in-memory check: could ANY trade fit under the GFV limit?
                # Use the locked start-of-day balance with buffer, NOT live balance.
                gfv_limit = self._gfv_start_balance
                min_trade_cost = (signal.atm_premium or 0) * 100
                if self._gfv_daily_spent + min_trade_cost > gfv_limit:
                    logger.warning(
                        f"GFV BLOCK: today's buys ${self._gfv_daily_spent:.0f} + "
                        f"min trade ${min_trade_cost:.0f} = ${self._gfv_daily_spent + min_trade_cost:.0f} "
                        f"> start-of-day balance ${gfv_limit:.0f} — blocking to prevent GFV"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "rejected",
                        f"gfv_block: spent=${self._gfv_daily_spent:.0f} + "
                        f"min_trade=${min_trade_cost:.0f} > sod_balance=${gfv_limit:.0f}",
                    )
                    return None

            if effective_balance <= 0:
                logger.info(
                    f"Effective balance ${effective_balance:.2f} <= 0 — skipping "
                    f"(deployed=${deployed_cost:.2f}, portfolio=${portfolio_size:.2f}"
                    + (f", webull_bp=${webull_buying_power:.2f}" if webull_buying_power is not None else "")
                    + ")"
                )
                return None

            signal_premium = signal.atm_premium
            assert signal_premium is not None

            premium = signal_premium * (1 + self.settings.SIMULATED_ENTRY_SLIPPAGE_BPS / 10000)
            entry_slippage = premium - signal_premium

            cost_per_contract = premium * 100
            if cost_per_contract <= 0:
                return None

            # Vinny's strategy: score-based sizing (5/3/1 contracts)
            use_vinny = getattr(self.settings, "ENABLE_VINNY_STRATEGY", False)
            use_score_sizing = getattr(self.settings, "ENABLE_SCORE_SIZING", False)

            if use_vinny and use_score_sizing:
                from options_owl.risk.vinny_strategy import score_to_contracts
                total_contracts = score_to_contracts(
                    signal.score,
                    cost_per_contract=cost_per_contract,
                    balance=effective_balance,
                    max_position_pct=self.settings.MAX_POSITION_PCT,
                    max_concurrent=self.settings.MAX_CONCURRENT,
                    max_portfolio_risk_pct=self.settings.MAX_PORTFOLIO_RISK_PCT,
                )
                if total_contracts <= 0:
                    logger.info(f"Score {signal.score} too low for Vinny sizing — 0 contracts")
                    return None
            elif self.settings.ENABLE_KELLY_SIZING:
                from options_owl.risk.kelly import compute_dynamic_position_pct
                position_pct = await compute_dynamic_position_pct(
                    self.db_path, signal.bot_source.value, self.settings,
                )
                max_position = effective_balance * (position_pct / 100)
                total_contracts = max(1, int(max_position / cost_per_contract))
            else:
                position_pct = self.settings.MAX_POSITION_PCT
                max_position = effective_balance * (position_pct / 100)
                total_contracts = max(1, int(max_position / cost_per_contract))

            # Multi-day contract cap: expensive multi-day options cause outsized losses.
            # Backtested: capping at 2 contracts reduced max single loss from -$1,900 to -$620.
            expiry_date = resolve_expiry_date(signal.expiry)
            if expiry_date:
                try:
                    exp = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    dte = (exp - _today_et().date()).days
                except (ValueError, TypeError):
                    dte = 0
            else:
                dte = 0

            if dte > 0:
                multi_max = getattr(self.settings, "MULTI_DAY_MAX_CONTRACTS", 2)
                expensive_thresh = getattr(self.settings, "MULTI_DAY_EXPENSIVE_THRESHOLD", 5.0)
                if premium > expensive_thresh:
                    # Very expensive multi-day option (e.g., $6.90 META) — cap at 1
                    old = total_contracts
                    total_contracts = min(total_contracts, 1)
                    if old != total_contracts:
                        logger.info(
                            f"SIZING: multi-day expensive cap: ${premium:.2f} > ${expensive_thresh:.2f} "
                            f"→ {old} → {total_contracts} contracts"
                        )
                elif multi_max > 0:
                    old = total_contracts
                    total_contracts = min(total_contracts, multi_max)
                    if old != total_contracts:
                        logger.info(
                            f"SIZING: multi-day cap: DTE={dte} → {old} → {total_contracts} contracts"
                        )

            # Late-session 0DTE size reduction: half after 1PM ET, 1 contract after 2PM ET.
            # Backtested: +$677 vs baseline — reduces theta-bleed risk on late entries.
            if dte == 0 and total_contracts > 1:
                hour_et = _today_et().hour
                if hour_et >= 14:  # 2 PM ET or later
                    old = total_contracts
                    total_contracts = 1
                    logger.info(
                        f"SIZING: late-0DTE cap (after 2PM ET): {old} → {total_contracts} contracts"
                    )
                elif hour_et >= 13:  # 1 PM ET or later
                    old = total_contracts
                    total_contracts = max(1, total_contracts // 2)
                    if old != total_contracts:
                        logger.info(
                            f"SIZING: late-0DTE half (after 1PM ET): {old} → {total_contracts} contracts"
                        )

            # DCA: disabled for Vinny strategy (100% at once, no DCA for 0DTE)
            dca_tranches_remaining = 0
            dca_total_contracts = total_contracts
            if (
                not use_vinny
                and strategy == "B"
                and self.settings.ENABLE_DCA
                and total_contracts > 1
            ):
                first_tranche = max(1, int(total_contracts * self.settings.DCA_FIRST_PCT / 100))
                dca_tranches_remaining = self.settings.DCA_TRANCHES - 1
                contracts = first_tranche
            else:
                contracts = total_contracts

            total_cost = contracts * cost_per_contract

            # Hard cap: never exceed PORTFOLIO_SIZE regardless of DB balance
            if total_cost > effective_balance:
                contracts = max(1, int(effective_balance / cost_per_contract))
                total_cost = contracts * cost_per_contract
                dca_tranches_remaining = 0

            if total_cost > effective_balance:
                return None

            option_type = "put" if signal.direction == Direction.PUT else "call"
            expiry_date = resolve_expiry_date(signal.expiry)

            now = datetime.now().isoformat()
            cursor = await conn.execute(
                "INSERT INTO paper_trades "
                "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
                "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
                "signal_premium, entry_slippage, "
                "dca_tranches_remaining, dca_total_contracts, "
                "target_1, target_2, target_3, target_4, target_5, "
                "stop_price, exit_by, expiry_date, strategy, supabase_alert_id, status, opened_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (
                    signal_id,
                    signal.ticker,
                    signal.direction.value,
                    signal.sentiment.value,
                    signal.score,
                    signal.strength.value,
                    signal.bot_source.value,
                    signal.entry_price,
                    signal.strike,
                    option_type,
                    contracts,
                    premium,
                    total_cost,
                    signal_premium,
                    entry_slippage,
                    dca_tranches_remaining,
                    dca_total_contracts,
                    signal.target_1,
                    signal.target_2,
                    signal.target_3,
                    signal.target_4,
                    signal.target_5,
                    signal.stop_price,
                    signal.exit_by,
                    expiry_date,
                    strategy,
                    supabase_alert_id,
                    now,
                ),
            )

            new_balance = portfolio["current_balance"] - total_cost
            await conn.execute(
                "UPDATE paper_portfolio SET current_balance = ?, total_trades = total_trades + 1 "
                "WHERE strategy = ?",
                (new_balance, strategy),
            )
            committed = await self._commit_with_retry(
                conn, f"open trade {signal.ticker} ${signal.strike}"
            )
            if not committed:
                logger.error(
                    f"CRITICAL: Failed to commit open trade for {signal.ticker} — "
                    f"trade NOT recorded in DB"
                )
                return None

            trade_id = cursor.lastrowid
            dca_tag = ""
            if dca_tranches_remaining > 0:
                dca_tag = f" | DCA: 1/{self.settings.DCA_TRANCHES} ({contracts}/{dca_total_contracts})"
            logger.info(
                f"📝 [{strategy}] PAPER TRADE: {signal.ticker} {option_type.upper()} "
                f"${signal.strike} x{contracts} @ ${premium:.2f} = ${total_cost:.2f} "
                f"| Score: {signal.score} | Bal: ${new_balance:.2f}{dca_tag}"
            )

            trade_info = {
                "trade_id": trade_id,
                "strategy": strategy,
                "ticker": signal.ticker,
                "option_type": option_type,
                "strike": signal.strike,
                "expiry_date": expiry_date,
                "contracts": contracts,
                "premium": premium,
                "signal_premium": signal_premium,
                "entry_slippage": entry_slippage,
                "total_cost": total_cost,
                "balance": new_balance,
            }

            # Update in-memory GFV tracker (before Webull call so it's counted immediately)
            actual_cost = trade_info["contracts"] * trade_info["premium"] * 100
            self._gfv_daily_spent += actual_cost

            # Route to Webull if live trading is enabled
            if self.webull_executor is not None:
                await self._place_webull_order(trade_id, trade_info, conn, nbbo_at_order=nbbo_at_order)
            else:
                logger.warning(
                    f"[TradeLifecycle] trade#{trade_id} {signal.ticker}: "
                    f"NO WEBULL EXECUTOR — trade is PAPER ONLY "
                    f"(PAPER_TRADE={self.settings.PAPER_TRADE})"
                )

            return trade_info

    async def _place_webull_order(
        self, trade_id: int, trade_info: dict, conn: aiosqlite.Connection,
        nbbo_at_order: dict | None = None,
    ) -> None:
        """Place a real Webull order for a paper trade. Updates DB with order IDs."""
        order_value = trade_info["contracts"] * trade_info["premium"] * 100

        # ── Wait for market open if pre-market signal ────────────────────
        # Options orders submitted before 9:30 ET sit as SUBMITTED and
        # time out.  Wait until 9:30:02 so the order can fill immediately.
        from zoneinfo import ZoneInfo
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))
        market_open = now_et.replace(hour=9, minute=30, second=2, microsecond=0)
        if now_et < market_open:
            wait_secs = (market_open - now_et).total_seconds()
            if wait_secs <= 300:  # only wait up to 5 min
                logger.info(
                    f"PRE-MARKET HOLD: trade#{trade_id} {trade_info['ticker']} — "
                    f"waiting {wait_secs:.0f}s until 9:30 ET"
                )
                await asyncio.sleep(wait_secs)
            else:
                logger.warning(
                    f"PRE-MARKET TOO EARLY: trade#{trade_id} {trade_info['ticker']} — "
                    f"{wait_secs:.0f}s until open, placing anyway"
                )

        # ── GFV double-check from DB (belt-and-suspenders) ───────────────
        # The in-memory GFV tracker could drift if signals arrive simultaneously.
        # This DB query is the final safety net — runs right before every order.
        if not self.settings.PAPER_TRADE and not self.settings.MARGIN_ACCOUNT and self._gfv_start_balance > 0:
            try:
                today_str = _today_et().strftime("%Y-%m-%d")
                async with _connect_db(self.db_path) as gfv_check:
                    cursor = await gfv_check.execute(
                        "SELECT COALESCE(SUM(premium_per_contract * contracts * 100), 0) "
                        "FROM paper_trades "
                        "WHERE date(opened_at) = ? AND webull_order_id IS NOT NULL",
                        (today_str,),
                    )
                    db_spent = float((await cursor.fetchone())[0])
                if db_spent + order_value > self._gfv_start_balance:
                    logger.error(
                        f"GFV HARD BLOCK (pre-order DB check): trade#{trade_id} {trade_info['ticker']} "
                        f"spent=${db_spent:.0f} + order=${order_value:.0f} = ${db_spent + order_value:.0f} "
                        f"> start-of-day balance=${self._gfv_start_balance:.0f} — BLOCKING"
                    )
                    await log_trade_event(
                        self.db_path, trade_info["ticker"], "webull_rejected",
                        f"trade#{trade_id} gfv_hard_block: db_spent=${db_spent:.0f} + "
                        f"order=${order_value:.0f} > sod=${self._gfv_start_balance:.0f}",
                        trade_id=trade_id,
                    )
                    await self._close_orphaned_trade(
                        trade_id, trade_info, conn,
                        reason="gfv_hard_block: would exceed start-of-day balance",
                    )
                    return
            except Exception as exc:
                logger.warning(f"GFV pre-order check failed (proceeding with caution): {exc}")

        # ── Final buying power gate (cash account hard block) ────────────
        # Re-check Webull buying power immediately before order placement.
        # This catches races where multiple signals arrive simultaneously.
        if not self.settings.PAPER_TRADE:
            try:
                info = await self.webull_executor.get_account_info()
                bp = info.buying_power
                if bp < order_value:
                    logger.warning(
                        f"CAPITAL BLOCK (pre-order): trade#{trade_id} {trade_info['ticker']} "
                        f"cost=${order_value:.2f} > buying_power=${bp:.2f} — "
                        f"BLOCKING order to prevent overspend"
                    )
                    await log_trade_event(
                        self.db_path, trade_info["ticker"], "webull_rejected",
                        f"trade#{trade_id} capital_block: cost=${order_value:.2f} > bp=${bp:.2f}",
                        trade_id=trade_id,
                    )
                    # Close the orphaned paper trade so it doesn't distort portfolio
                    await self._close_orphaned_trade(
                        trade_id, trade_info, conn,
                        reason=f"capital_block: cost=${order_value:.2f} > bp=${bp:.2f}",
                    )
                    return
            except Exception as exc:
                logger.warning(
                    f"Pre-order buying power check failed: {exc} — proceeding with caution"
                )

        logger.info(
            f"WEBULL ENTRY ATTEMPT: trade#{trade_id} "
            f"{trade_info['ticker']} ${trade_info['strike']} "
            f"{trade_info['option_type'].upper()} exp={trade_info['expiry_date']} "
            f"x{trade_info['contracts']} @ ${trade_info['premium']:.2f} "
            f"(value=${order_value:.2f})"
        )
        try:
            # Smart entry: retry with fresh pricing if the first attempt doesn't fill.
            # 0DTE options move fast — by the time we place the order, the ask may
            # have moved above our limit. We retry up to MAX_ENTRY_RETRIES times,
            # fetching a fresh ask each time, capped at signal + MAX_ENTRY_CHASE_PCT.
            base_premium = trade_info["premium"]
            aggress_pct = getattr(self.settings, "WEBULL_ENTRY_AGGRESS_PCT", 2.0)
            max_retries = getattr(self.settings, "MAX_ENTRY_RETRIES", 3)
            max_chase_pct = getattr(self.settings, "MAX_ENTRY_CHASE_PCT", 15.0)
            max_price = round(base_premium * (1 + max_chase_pct / 100), 2)

            result = None
            limit_used = round(base_premium * (1 + aggress_pct / 100), 2)

            for attempt in range(max_retries):
                if attempt == 0:
                    # First attempt: signal premium + aggress %
                    if limit_used != base_premium:
                        logger.info(
                            f"WEBULL AGGRESS: ${base_premium:.2f} + {aggress_pct}% → "
                            f"${limit_used:.2f} (max willing=${max_price:.2f})"
                        )
                else:
                    # Retry: only chase if underlying is moving IN OUR FAVOR.
                    # If it's moving against us, the non-fill saved us money.
                    fresh_ask = await self._get_fresh_option_ask(trade_info)
                    if fresh_ask is None:
                        logger.warning(
                            f"ENTRY RETRY #{attempt + 1}: trade#{trade_id} "
                            f"{trade_info['ticker']} — no fresh ask, giving up"
                        )
                        break

                    # Check if underlying moved in our direction
                    is_favorable = await self._is_underlying_favorable(trade_info)
                    if not is_favorable:
                        logger.info(
                            f"ENTRY RETRY #{attempt + 1}: trade#{trade_id} "
                            f"{trade_info['ticker']} — underlying moving against us, "
                            f"not chasing (fresh ask=${fresh_ask:.2f})"
                        )
                        break

                    limit_used = round(fresh_ask * (1 + aggress_pct / 100), 2)
                    if limit_used > max_price:
                        logger.warning(
                            f"ENTRY RETRY #{attempt + 1}: trade#{trade_id} "
                            f"{trade_info['ticker']} — fresh ask ${fresh_ask:.2f} → "
                            f"limit ${limit_used:.2f} exceeds cap ${max_price:.2f}, giving up"
                        )
                        break
                    logger.info(
                        f"ENTRY RETRY #{attempt + 1}: trade#{trade_id} "
                        f"{trade_info['ticker']} — fresh ask ${fresh_ask:.2f} → "
                        f"limit ${limit_used:.2f} (cap=${max_price:.2f})"
                    )

                result = await self.webull_executor.buy_option(
                    ticker=trade_info["ticker"],
                    strike=trade_info["strike"],
                    expiry_date=trade_info["expiry_date"] or "",
                    option_type=trade_info["option_type"].upper(),
                    contracts=trade_info["contracts"],
                    limit_price=limit_used,
                )

                if result.success:
                    break  # Filled — exit retry loop

                # Not filled — log and retry
                logger.warning(
                    f"ENTRY NOT FILLED (attempt {attempt + 1}/{max_retries}): "
                    f"trade#{trade_id} {trade_info['ticker']} @ ${limit_used:.2f} — "
                    f"{result.error}"
                )

            if result is None:
                # Should not happen, but handle gracefully
                from options_owl.execution.webull_executor import OrderResult
                result = OrderResult(
                    success=False, error="No order attempts made", fill_status="FAILED",
                )

            logger.info(
                f"WEBULL ENTRY RESPONSE: trade#{trade_id} success={result.success} "
                f"order_id={result.order_id} client_id={result.client_order_id} "
                f"error={result.error}"
            )

            if result.success:
                # Fetch real fill price from Webull
                fill_price = None
                if result.client_order_id:
                    fill_price = await self.webull_executor.get_fill_price(
                        result.client_order_id
                    )
                fill_str = f", fill=${fill_price:.2f}" if fill_price else ""
                logger.info(
                    f"WEBULL ENTRY: {trade_info['ticker']} "
                    f"x{trade_info['contracts']} — order_id={result.order_id}"
                    + fill_str
                )
                nbbo_detail = ""
                if nbbo_at_order:
                    nbbo_detail = (
                        f" nbbo_bid=${nbbo_at_order.get('bid', 0):.2f}"
                        f" nbbo_ask=${nbbo_at_order.get('ask', 0):.2f}"
                        f" nbbo_mid=${nbbo_at_order.get('mid', 0):.2f}"
                    )
                await log_trade_event(
                    self.db_path, trade_info["ticker"], "webull_filled",
                    f"trade#{trade_id} order_id={result.order_id} "
                    f"x{trade_info['contracts']} @ ${trade_info['premium']:.2f}"
                    + fill_str + nbbo_detail,
                    trade_id=trade_id,
                )
                # Update Webull order IDs and fill price.
                # CRITICAL: also update premium_per_contract and total_cost to
                # match the real fill — otherwise close_trade() calculates P&L
                # from the signal premium instead of the actual entry cost.
                if fill_price and fill_price > 0:
                    real_total_cost = fill_price * trade_info["contracts"] * 100
                    await conn.execute(
                        "UPDATE paper_trades SET webull_order_id = ?, "
                        "webull_client_order_id = ?, webull_entry_fill_price = ?, "
                        "premium_per_contract = ?, total_cost = ? "
                        "WHERE id = ?",
                        (result.order_id, result.client_order_id, fill_price,
                         fill_price, real_total_cost, trade_id),
                    )
                else:
                    await conn.execute(
                        "UPDATE paper_trades SET webull_order_id = ?, "
                        "webull_client_order_id = ?, webull_entry_fill_price = ? "
                        "WHERE id = ?",
                        (result.order_id, result.client_order_id, fill_price, trade_id),
                    )
                await self._commit_with_retry(
                    conn, f"webull order ID for trade #{trade_id}"
                )
                # Supabase: record fill (fire-and-forget)
                if self.supabase:
                    # Read alert_id from the trade we just inserted
                    try:
                        cursor2 = await conn.execute(
                            "SELECT supabase_alert_id FROM paper_trades WHERE id = ?",
                            (trade_id,),
                        )
                        row = await cursor2.fetchone()
                        s_alert_id = row[0] if row else None
                    except Exception:
                        s_alert_id = None
                    if s_alert_id:
                        actual_fill = fill_price or trade_info["premium"]
                        signal_prem = trade_info.get("signal_premium") or trade_info["premium"]
                        slippage = ((actual_fill - signal_prem) / signal_prem * 100) if signal_prem > 0 else None
                        _fire_and_forget(self.supabase.record_fill(
                            alert_id=s_alert_id,
                            broker_order_id=str(result.order_id or result.client_order_id or trade_id),
                            fill_price=actual_fill,
                            fill_quantity=trade_info["contracts"],
                            strike=trade_info["strike"],
                            slippage_pct=slippage,
                            nbbo_at_order=nbbo_at_order,
                        ))
            else:
                logger.error(
                    f"WEBULL ENTRY FAILED: {trade_info['ticker']} — {result.error} "
                    f"— closing orphaned paper trade #{trade_id}"
                )
                await log_trade_event(
                    self.db_path, trade_info["ticker"], "webull_rejected",
                    f"trade#{trade_id} error={result.error}",
                    trade_id=trade_id,
                )
                await self._close_orphaned_trade(
                    trade_id, trade_info, conn,
                    reason=f"webull_rejected: {result.error}",
                )
        except Exception as exc:
            logger.error(
                f"WEBULL ENTRY ERROR: {trade_info['ticker']} — {type(exc).__name__}: {exc} "
                f"— closing orphaned paper trade #{trade_id}"
            )
            await log_trade_event(
                self.db_path, trade_info["ticker"], "webull_error",
                f"trade#{trade_id} {type(exc).__name__}: {exc}",
                trade_id=trade_id,
            )
            try:
                await self._close_orphaned_trade(
                    trade_id, trade_info, conn,
                    reason=f"webull_error: {type(exc).__name__}: {exc}",
                )
            except Exception:
                pass  # best-effort cleanup

    async def _close_orphaned_trade(
        self, trade_id: int, trade_info: dict, conn: aiosqlite.Connection,
        reason: str = "webull_failed",
    ) -> None:
        """Close a paper trade that failed to place on Webull.

        Marks the trade as closed with $0 P&L and restores the portfolio balance.
        Without this, failed Webull orders leave phantom open trades that distort
        the paper portfolio balance and get monitored by position_monitor.
        """
        total_cost = trade_info["total_cost"]
        strategy = trade_info.get("strategy", "B")
        try:
            await conn.execute(
                "UPDATE paper_trades SET status = 'closed', exit_reason = ?, "
                "pnl_dollars = 0, pnl_pct = 0, closed_at = ?, "
                "exit_premium = premium_per_contract "
                "WHERE id = ? AND status = 'open'",
                (f"orphan_closed: {reason}", datetime.now().isoformat(), trade_id),
            )
            # Restore portfolio balance
            await conn.execute(
                "UPDATE paper_portfolio SET current_balance = current_balance + ? "
                "WHERE strategy = ?",
                (total_cost, strategy),
            )
            await self._commit_with_retry(
                conn, f"close orphaned trade #{trade_id}"
            )
            logger.info(
                f"ORPHAN CLOSED: trade#{trade_id} {trade_info['ticker']} — "
                f"${total_cost:.2f} restored to portfolio ({reason})"
            )
        except Exception as exc:
            logger.error(
                f"Failed to close orphaned trade #{trade_id}: {exc}"
            )

    async def close_webull_position(
        self, trade: dict, exit_premium: float,
        child_trade_id: int | None = None,
    ) -> bool:
        """Close the corresponding Webull position when a paper trade is closed.

        Retry-aware: tracks attempts per trade and adjusts pricing on each retry.
        FAST ESCALATION — 0DTE premiums crash fast, can't afford 45s per attempt.
        - Attempt 1: limit at fresh bid (not stale pipeline price)
        - Attempt 2: limit at bid - 5%
        - Attempt 3: limit at bid - 10%
        - Attempt 4: limit at bid - 15%
        - Attempt 5+: limit at bid - 20% (near-market)

        Args:
            child_trade_id: When set, also update this child row with the real
                exit fill (for scaleout partial closes where the child row
                has the actual P&L record).

        Returns True if sell succeeded (or no executor), False if sell failed
        and the position is still open on Webull.
        """
        if self.webull_executor is None:
            return True

        ticker = trade["ticker"]
        trade_id = trade["id"]
        retry_count = trade.get("sell_retry_count") or 0

        # Always fetch fresh bid — pipeline price is stale by the time we sell
        fresh_bid = await self._get_fresh_option_bid(trade)
        base_price = fresh_bid if (fresh_bid and fresh_bid > 0) else exit_premium

        # Escalate aggressively: each retry undercuts the bid more
        if retry_count == 0:
            sell_price = base_price
            price_source = "fresh_bid" if (fresh_bid and fresh_bid > 0) else "pipeline"
        elif retry_count == 1:
            sell_price = base_price * 0.95
            price_source = "bid-5%"
        elif retry_count == 2:
            sell_price = base_price * 0.90
            price_source = "bid-10%"
        elif retry_count == 3:
            sell_price = base_price * 0.85
            price_source = "bid-15%"
        else:
            # Retry 4+: near-market — just get out
            sell_price = base_price * 0.80
            price_source = f"bid-20%(retry#{retry_count})"

        # Floor: never sell below $0.01
        sell_price = max(sell_price, 0.01)

        exit_value = trade["contracts"] * sell_price * 100

        logger.info(
            f"WEBULL EXIT ATTEMPT #{retry_count + 1}: trade#{trade_id} "
            f"{ticker} ${trade['strike']} {trade['option_type'].upper()} "
            f"exp={trade.get('expiry_date')} x{trade['contracts']} "
            f"@ ${sell_price:.2f} ({price_source}) (value=${exit_value:.2f})"
        )

        # Track the attempt
        await _db_execute_with_retry(
            self.db_path,
            [(
                "UPDATE paper_trades SET sell_retry_count = ?, "
                "sell_last_attempted_price = ? WHERE id = ?",
                (retry_count + 1, sell_price, trade_id),
            )],
            context=f"sell retry tracking for trade #{trade_id}",
        )

        # Cancel any pending orders on the same contract before selling.
        # This prevents REVERSE_OPTION errors when the entry order is still
        # in SUBMITTED state or a prior sell attempt is pending.
        if retry_count > 0:
            try:
                open_orders = await self.webull_executor.get_open_orders()
                strike_str = str(trade["strike"])
                exp_str = trade.get("expiry_date") or ""
                ot_str = trade["option_type"].upper()
                for order in open_orders:
                    # Match by ticker + strike + expiry + option_type
                    legs = order.get("legs") or []
                    for leg in legs:
                        if (leg.get("symbol") == ticker
                                and str(leg.get("strike_price")) == strike_str
                                and leg.get("option_expire_date") == exp_str
                                and leg.get("option_type") == ot_str):
                            coid = order.get("client_order_id", "")
                            logger.warning(
                                f"CANCELLING PENDING ORDER before sell: "
                                f"trade#{trade_id} {ticker} client_id={coid}"
                            )
                            await self.webull_executor.cancel_order(coid)
                            await asyncio.sleep(1)  # let Webull settle
                            break
            except Exception as exc:
                logger.warning(
                    f"Failed to cancel pending orders for {ticker} #{trade_id}: {exc}"
                )

        try:
            result = await self.webull_executor.sell_option(
                ticker=ticker,
                strike=trade["strike"],
                expiry_date=trade.get("expiry_date") or "",
                option_type=trade["option_type"].upper(),
                contracts=trade["contracts"],
                limit_price=sell_price,
                has_webull_order_id=bool(trade.get("webull_order_id")),
            )

            logger.debug(
                f"WEBULL EXIT RESPONSE: trade#{trade_id} success={result.success} "
                f"order_id={result.order_id} client_id={result.client_order_id} "
                f"error={result.error}"
            )

            if result.success:
                # Handle partial fills — some contracts sold, some didn't
                requested = trade["contracts"]
                filled = result.filled_quantity
                if result.fill_status == "PARTIAL" and filled and 0 < filled < requested:
                    remaining = requested - filled
                    logger.warning(
                        f"WEBULL PARTIAL EXIT: {ticker} #{trade_id} — "
                        f"sold {filled}/{requested}, {remaining} still open. "
                        f"Will retry remaining on next cycle."
                    )
                    # Update trade to reflect only the remaining contracts
                    await _db_execute_with_retry(
                        self.db_path,
                        [(
                            "UPDATE paper_trades SET contracts = ?, "
                            "sell_retry_count = 0 WHERE id = ?",
                            (remaining, trade_id),
                        )],
                        context=f"partial fill update for trade #{trade_id}",
                    )
                    # Return False so the monitor reopens the trade for the remainder
                    return False

                # Fetch real exit fill price from Webull (with retry for 429)
                exit_fill_price = None
                if result.client_order_id:
                    exit_fill_price = await self.webull_executor.get_fill_price(
                        result.client_order_id
                    )
                logger.info(
                    f"WEBULL EXIT FILLED: {ticker} x{trade['contracts']} "
                    f"@ ${sell_price:.2f} ({price_source}) — order_id={result.order_id}"
                    + (f", fill=${exit_fill_price:.2f}" if exit_fill_price else "")
                    + (f" [after {retry_count} retries]" if retry_count > 0 else "")
                )
                # Reset retry count on success
                await _db_execute_with_retry(
                    self.db_path,
                    [(
                        "UPDATE paper_trades SET sell_retry_count = 0 WHERE id = ?",
                        (trade_id,),
                    )],
                    context=f"reset sell retry for trade #{trade_id}",
                )
                # Store real exit fill price + recompute P&L from real fills.
                # Re-read entry fill from DB — the trade dict may be stale if
                # the entry fill was written in the same monitor cycle.
                if exit_fill_price is not None:
                    async with _connect_db(self.db_path) as rconn:
                        rconn.row_factory = aiosqlite.Row
                        row = await rconn.execute(
                            "SELECT webull_entry_fill_price, premium_per_contract, "
                            "dca_total_contracts FROM paper_trades WHERE id = ?",
                            (trade_id,),
                        )
                        fresh = await row.fetchone()
                    raw_fill = (fresh["webull_entry_fill_price"] if fresh else None) or 0
                    blended_avg = (fresh["premium_per_contract"] if fresh else None) or 0
                    dca_qty = (fresh["dca_total_contracts"] if fresh else None) or 0
                    # After DCA, webull_entry_fill_price is only the FIRST fill.
                    # Use premium_per_contract (blended avg) when DCA occurred.
                    entry_fill = blended_avg if (dca_qty and dca_qty > 0 and blended_avg > 0) else raw_fill
                    if entry_fill and entry_fill > 0:
                        # Recompute P&L from real Webull fills
                        real_pnl = (exit_fill_price - entry_fill) * trade["contracts"] * 100
                        real_pnl_pct = ((exit_fill_price - entry_fill) / entry_fill * 100) if entry_fill > 0 else 0
                        dca_note = f" (blended avg, raw_fill=${raw_fill:.2f})" if dca_qty else ""
                        logger.info(
                            f"WEBULL P&L RECONCILE: trade#{trade_id} {ticker} "
                            f"entry_fill=${entry_fill:.2f}{dca_note} exit_fill=${exit_fill_price:.2f} "
                            f"real_pnl=${real_pnl:.2f} ({real_pnl_pct:+.1f}%)"
                        )
                        ops = [(
                            "UPDATE paper_trades SET webull_exit_fill_price = ?, "
                            "webull_exit_order_id = ?, pnl_dollars = ?, pnl_pct = ? "
                            "WHERE id = ?",
                            (exit_fill_price, result.order_id, real_pnl, real_pnl_pct, trade_id),
                        )]
                        # Also update scaleout child row with real exit fill
                        if child_trade_id:
                            child_pnl = (exit_fill_price - entry_fill) * trade["contracts"] * 100
                            child_pct = real_pnl_pct  # same % gain
                            ops.append((
                                "UPDATE paper_trades SET webull_exit_fill_price = ?, "
                                "webull_exit_order_id = ?, pnl_dollars = ?, pnl_pct = ?, "
                                "webull_entry_fill_price = ? "
                                "WHERE id = ?",
                                (exit_fill_price, result.order_id, child_pnl, child_pct,
                                 entry_fill, child_trade_id),
                            ))
                            logger.info(
                                f"WEBULL P&L RECONCILE (child): trade#{child_trade_id} "
                                f"pnl=${child_pnl:.2f} ({child_pct:+.1f}%)"
                            )
                        await _db_execute_with_retry(
                            self.db_path, ops,
                            context=f"webull exit fill + P&L for trade #{trade_id}",
                        )
                    else:
                        await _db_execute_with_retry(
                            self.db_path,
                            [(
                                "UPDATE paper_trades SET webull_exit_fill_price = ?, "
                                "webull_exit_order_id = ? WHERE id = ?",
                                (exit_fill_price, result.order_id, trade_id),
                            )],
                            context=f"webull exit fill for trade #{trade_id}",
                        )
                return True
            else:
                logger.error(
                    f"WEBULL EXIT FAILED (attempt #{retry_count + 1}): "
                    f"{ticker} #{trade_id} — {result.error} "
                    f"@ ${sell_price:.2f} ({price_source})"
                )
                return False
        except Exception as exc:
            logger.error(
                f"WEBULL EXIT ERROR (attempt #{retry_count + 1}): "
                f"{ticker} #{trade_id} — {type(exc).__name__}: {exc} "
                f"@ ${sell_price:.2f} ({price_source})"
            )
            return False

    async def _is_underlying_favorable(self, trade_info: dict) -> bool:
        """Check if the underlying is moving in our trade's favor.

        For calls: underlying should be at or above entry price (ripping up).
        For puts: underlying should be at or below entry price (dropping).
        Returns True if favorable or if we can't determine (fail-open).
        """
        try:
            import httpx
            ticker = trade_info["ticker"]
            api_key = self.settings.POLYGON_API_KEY
            if not api_key:
                return True

            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"https://api.polygon.io/v2/last/trade/{ticker}",
                    params={"apiKey": api_key},
                )
                data = resp.json()
                current_price = data.get("results", {}).get("p", 0)

            if not current_price or current_price <= 0:
                return True  # can't determine — allow retry
            entry_price = trade_info.get("entry_price") or 0
            if entry_price <= 0:
                return True
            is_call = trade_info["option_type"].lower() == "call"
            favorable = current_price >= entry_price if is_call else current_price <= entry_price
            logger.debug(
                f"Underlying check: {ticker} "
                f"{'call' if is_call else 'put'} entry=${entry_price:.2f} "
                f"now=${current_price:.2f} → {'favorable' if favorable else 'against'}"
            )
            return favorable
        except Exception as exc:
            logger.debug(f"Could not check underlying direction: {exc}")
            return True  # fail-open — allow retry

    async def _get_fresh_option_ask(self, trade_info: dict) -> float | None:
        """Fetch the current ask price for an option contract from Polygon.

        Used for entry retries — ensures we're pricing at what the market is
        actually offering, not the stale signal premium.
        """
        try:
            from options_owl.collectors.polygon_options import (
                _snapshot_quote,
                build_option_contract_ticker,
            )
            import httpx

            ticker = trade_info["ticker"]
            strike = trade_info["strike"]
            option_type = trade_info["option_type"].lower()
            expiry = trade_info.get("expiry_date") or ""
            api_key = self.settings.POLYGON_API_KEY

            if not api_key:
                return None

            contract = build_option_contract_ticker(ticker, strike, expiry, option_type)

            async with httpx.AsyncClient(timeout=10) as client:
                quote = await _snapshot_quote(client, api_key, ticker, contract)

            if quote and quote.get("ask") and quote["ask"] > 0:
                logger.debug(
                    f"Fresh option ask for {ticker} ${strike} {option_type}: "
                    f"bid=${quote.get('bid', 0):.2f} ask=${quote['ask']:.2f}"
                )
                return quote["ask"]
            elif quote and quote.get("mid") and quote["mid"] > 0:
                return quote["mid"] * 1.05  # slight premium since we're buying
        except Exception as exc:
            logger.debug(f"Could not fetch fresh ask for {trade_info['ticker']}: {exc}")
        return None

    async def _get_fresh_option_bid(self, trade: dict) -> float | None:
        """Fetch the current bid price for an option contract from Polygon.

        Used to adjust sell limit price on retry — ensures we're pricing
        at what the market will actually pay, not a stale premium.
        """
        try:
            from options_owl.collectors.polygon_options import (
                _snapshot_quote,
                build_option_contract_ticker,
            )
            import httpx

            ticker = trade["ticker"]
            strike = trade["strike"]
            option_type = trade["option_type"].lower()
            expiry = trade.get("expiry_date") or ""
            api_key = self.settings.POLYGON_API_KEY

            if not api_key:
                return None

            contract = build_option_contract_ticker(ticker, strike, expiry, option_type)

            async with httpx.AsyncClient(timeout=10) as client:
                quote = await _snapshot_quote(client, api_key, ticker, contract)

            if quote and quote.get("bid") and quote["bid"] > 0:
                logger.debug(
                    f"Fresh option quote for {ticker} ${strike} {option_type}: "
                    f"bid=${quote['bid']:.2f} ask=${quote['ask']:.2f} mid=${quote['mid']:.2f}"
                )
                return quote["bid"]
            elif quote and quote.get("mid") and quote["mid"] > 0:
                # No bid available, use mid as approximation
                return quote["mid"] * 0.95  # slight discount since we're selling
        except Exception as exc:
            logger.debug(f"Could not fetch fresh bid for {trade['ticker']}: {exc}")
        return None

    async def _fill_missing_premium(self, signal: TradeSignal) -> TradeSignal:
        """Look up ATM premium from Polygon (primary) or yfinance (fallback).

        Uses the same per-ticker expiry schedule as smart entry so that weekly
        tickers (MSTR, AMD, PLTR) try Friday's contract when 0DTE doesn't exist.
        """
        import asyncio

        from options_owl.execution.position_monitor import (
            _fetch_option_chain_for_ticker,
            _lookup_premium_from_chain,
        )

        expiry_date = resolve_expiry_date(signal.expiry)
        if not expiry_date:
            logger.debug(f"Cannot look up premium for {signal.ticker}: no expiry date")
            return signal

        option_type = "put" if signal.direction == Direction.PUT else "call"

        # Build expiry candidates using the same per-ticker schedule as smart entry
        expiry_candidates = _build_expiry_candidates(signal.ticker, expiry_date)

        # --- Primary: Polygon REST snapshot (use ask price for buys) ---
        api_key = getattr(self.settings, "POLYGON_API_KEY", "")
        if api_key:
            from options_owl.collectors.polygon_options import polygon_option_quote

            for try_expiry in expiry_candidates:
                try:
                    quote = await polygon_option_quote(
                        api_key, signal.ticker, signal.strike, try_expiry, option_type,
                    )
                    if quote:
                        ask = quote.get("ask", 0)
                        mid = quote.get("mid", 0)
                        if ask > 0:
                            buffer_pct = getattr(self.settings, "SMART_ENTRY_ASK_BUFFER_PCT", 5.0)
                            premium = round(ask * (1 + buffer_pct / 100), 2)
                        else:
                            premium = mid
                        if premium and premium > 0:
                            update = {"atm_premium": premium, "atm_strike": signal.strike}
                            if try_expiry != expiry_date:
                                update["expiry"] = try_expiry
                                logger.info(
                                    f"Polygon filled premium for {signal.ticker} "
                                    f"${signal.strike} {option_type}: ${premium:.2f} "
                                    f"(ask=${ask:.2f}, mid=${mid:.2f}, "
                                    f"exp={try_expiry} — fallback from {expiry_date})"
                                )
                            else:
                                logger.info(
                                    f"Polygon filled premium for {signal.ticker} "
                                    f"${signal.strike} {option_type}: ${premium:.2f} "
                                    f"(ask=${ask:.2f}, mid=${mid:.2f})"
                                )
                            return signal.model_copy(update=update)
                except Exception as exc:
                    logger.debug(f"Polygon premium lookup failed for {signal.ticker} exp={try_expiry}: {exc}")

        # --- Fallback: yfinance chain ---
        for try_expiry in expiry_candidates:
            try:
                chain = await asyncio.to_thread(
                    _fetch_option_chain_for_ticker, signal.ticker, try_expiry,
                )
                if not chain:
                    continue

                premium = _lookup_premium_from_chain(chain, signal.strike, option_type)
                if premium and premium > 0:
                    update = {"atm_premium": premium, "atm_strike": signal.strike}
                    if try_expiry != expiry_date:
                        update["expiry"] = try_expiry
                        logger.info(
                            f"yfinance filled premium for {signal.ticker} "
                            f"${signal.strike} {option_type}: ${premium:.2f} "
                            f"(exp={try_expiry} — fallback from {expiry_date})"
                        )
                    else:
                        logger.info(
                            f"yfinance filled premium for {signal.ticker} "
                            f"${signal.strike} {option_type}: ${premium:.2f}"
                        )
                    return signal.model_copy(update=update)
            except Exception as exc:
                logger.debug(f"yfinance chain for {signal.ticker} exp={try_expiry} failed: {exc}")

        logger.warning(
            f"No premium found for {signal.ticker} ${signal.strike} {option_type} "
            f"across expiries {expiry_candidates}"
        )
        return signal

    async def evaluate_and_trade(self, signal: TradeSignal, signal_id: int) -> dict | None:
        """Evaluate a signal through the entry pipeline and open a paper trade if approved."""

        # Supabase: look up alert_id for this signal (non-blocking)
        # Single attempt + one retry after 2s if not found (scanner may lag behind Discord)
        supabase_alert_id = None
        supabase_conviction = None
        if self.supabase and self.supabase.enabled:
            direction = "bearish" if signal.direction == Direction.PUT else "bullish"
            try:
                alert = await self.supabase.lookup_alert(signal.ticker, direction)
                if not alert:
                    # Scanner may fire slightly after Discord — brief retry
                    await asyncio.sleep(2)
                    alert = await self.supabase.lookup_alert(signal.ticker, direction)
                if alert:
                    supabase_alert_id = alert.get("alert_id")
                    supabase_conviction = alert.get("conviction_0_100")
                    logger.info(
                        f"[SupabaseBrain] {signal.ticker}: "
                        f"alert_id={str(supabase_alert_id)[:8] if supabase_alert_id else '?'}... "
                        f"conviction={supabase_conviction}"
                    )
            except Exception as exc:
                logger.debug(f"[SupabaseBrain] Alert lookup failed (proceeding): {exc}")
            if not supabase_alert_id:
                logger.info(
                    f"[SupabaseBrain] {signal.ticker}: no alert_id found "
                    f"(decision logging will be skipped for this signal)"
                )

        # Daily circuit breaker: stop trading if today's realized + unrealized losses exceed threshold.
        # Includes open-trade unrealized P&L so a string of underwater positions triggers the breaker.
        cb_pct = getattr(self.settings, "DAILY_LOSS_CIRCUIT_BREAKER_PCT", 0)
        if cb_pct > 0:
            cb_limit = self.settings.PORTFOLIO_SIZE * (cb_pct / 100)
            try:
                async with _connect_db(self.db_path) as conn:
                    # Only count REAL Webull trades — paper-only losses are not real money.
                    today_et_str = _today_et().strftime("%Y-%m-%d")
                    cursor = await conn.execute(
                        "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                        "WHERE status = 'closed' AND date(closed_at) = ? "
                        "AND webull_order_id IS NOT NULL",
                        (today_et_str,),
                    )
                    realized_pnl = (await cursor.fetchone())[0]
                    # Unrealized losses from today's Webull trades only.
                    cursor = await conn.execute(
                        "SELECT COALESCE(SUM("
                        "  (COALESCE(mae_premium, premium_per_contract) - premium_per_contract)"
                        "  * contracts * 100"
                        "), 0) FROM paper_trades WHERE status = 'open'"
                        " AND date(opened_at) = ? AND webull_order_id IS NOT NULL",
                        (today_et_str,),
                    )
                    unrealized_pnl = (await cursor.fetchone())[0]
                    today_pnl = realized_pnl + min(0, unrealized_pnl)  # only count unrealized losses
                if today_pnl < -cb_limit:
                    logger.warning(
                        f"CIRCUIT BREAKER: daily P&L ${today_pnl:.0f} (realized=${realized_pnl:.0f} "
                        f"unrealized=${unrealized_pnl:.0f}) exceeds -${cb_limit:.0f} "
                        f"({cb_pct}% of ${self.settings.PORTFOLIO_SIZE:.0f}) — blocking {signal.ticker}"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "rejected",
                        f"circuit_breaker: daily_pnl=${today_pnl:.0f} (real=${realized_pnl:.0f} "
                        f"unreal=${unrealized_pnl:.0f}) < -${cb_limit:.0f}",
                    )
                    return None
            except Exception as exc:
                logger.warning(f"Circuit breaker check failed (proceeding): {exc}")

        # Use the premium for the actual trade strike, not the deep-ITM ATM premium.
        # For individual stocks (GOOGL, META, etc.), the parser's atm_premium is the
        # true ATM option ($14-$60), but the signal recommends a near-OTM strike whose
        # premium is in otm_premium ($0.50-$1.50).  The paper trader should use the
        # option it will actually trade.
        signal = _select_trade_premium(signal)

        # If ATM premium is missing OR unreasonably high for 0DTE (Discord bots
        # sometimes report multi-week prices for individual stocks), look it up.
        needs_lookup = (
            not signal.atm_premium
            or signal.atm_premium <= 0
            or (
                signal.expiry
                and "0dte" in signal.expiry.lower()
                and signal.atm_premium > 5.0
                and signal.entry_price
                and signal.entry_price < 500
            )
        )
        if needs_lookup:
            original = signal.atm_premium
            signal = await self._fill_missing_premium(signal)
            if original and original > 0 and signal.atm_premium == original:
                # Lookup didn't find anything — keep original even if expensive
                logger.debug(
                    f"{signal.ticker}: premium ${original:.2f} seems high for 0DTE "
                    f"but market lookup unavailable"
                )

        # Smart entry: verify live option premium and decide trade price
        live_price_reason = "skipped"
        nbbo_at_order: dict | None = None
        if getattr(self.settings, "ENABLE_SMART_ENTRY", True):
            signal, live_price_reason, nbbo_at_order = await _verify_live_premium(signal, self.settings)
            nbbo_str = ""
            if nbbo_at_order:
                nbbo_str = (
                    f" nbbo_bid=${nbbo_at_order.get('bid', 0):.2f}"
                    f" nbbo_ask=${nbbo_at_order.get('ask', 0):.2f}"
                    f" nbbo_mid=${nbbo_at_order.get('mid', 0):.2f}"
                )
            logger.info(
                f"[TradeLifecycle] {signal.ticker}: smart_entry={live_price_reason} "
                f"premium=${signal.atm_premium or 0:.2f}{nbbo_str}"
            )
            await log_trade_event(
                self.db_path, signal.ticker, "smart_entry",
                f"result={live_price_reason} premium=${signal.atm_premium or 0:.2f} "
                f"strike=${signal.strike} expiry={signal.expiry}{nbbo_str}",
            )
            if not signal.atm_premium or signal.atm_premium <= 0:
                logger.info(
                    f"SKIP: {signal.ticker} — smart entry rejected: {live_price_reason}"
                )
                await log_trade_event(
                    self.db_path, signal.ticker, "rejected",
                    f"smart_entry_blocked: {live_price_reason}",
                )
                return None

        # Fetch current underlying price for anti-chase gate (only if Vinny strategy enabled)
        current_underlying = None
        if getattr(self.settings, "ENABLE_VINNY_STRATEGY", False):
            current_underlying = await self._get_current_price(signal.ticker)

        async with _connect_db(self.db_path) as conn:
            portfolio = await _get_or_create_portfolio(conn, self.settings.PORTFOLIO_SIZE, "B")

            cursor = await conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
            )
            open_count = (await cursor.fetchone())[0]  # type: ignore[index]

            cursor = await conn.execute(
                "SELECT DISTINCT ticker FROM paper_trades WHERE status = 'open'"
            )
            open_tickers = {row[0] for row in await cursor.fetchall()}

            # Fetch open positions with direction for correlation cap gate
            cursor = await conn.execute(
                "SELECT ticker, option_type FROM paper_trades WHERE status = 'open'"
            )
            open_positions = [(row[0], row[1]) for row in await cursor.fetchall()]

        # Lazy-init candle cache for momentum confirmation gate
        candle_cache = None
        polygon_key = getattr(self.settings, "POLYGON_API_KEY", None)
        if polygon_key:
            try:
                from options_owl.collectors.candle_cache import CandleCache
                if not hasattr(self, "_candle_cache"):
                    shared_db = getattr(self.settings, "SHARED_CANDLE_DB", "") or ""
                    self._candle_cache = CandleCache(
                        polygon_key, shared_db_path=shared_db or None
                    )
                candle_cache = self._candle_cache
            except Exception as exc:
                logger.warning(f"Failed to init candle cache for momentum gate: {exc}")

        ctx = {
            "signal": signal,
            "settings": self.settings,
            "db_path": self.db_path,
            "portfolio": portfolio,
            "open_count": open_count,
            "open_tickers": open_tickers,
            "open_positions": open_positions,
            "current_price": current_underlying,
            "webull_executor": self.webull_executor,
            "candle_cache": candle_cache,
        }
        result = await run_entry_pipeline(ctx)

        if not result.approved:
            logger.debug(result.summary())
            logger.info(
                f"SKIP: {signal.ticker} — {'; '.join(result.failure_reasons)}"
            )
            await log_trade_event(
                self.db_path, signal.ticker, "pipeline_rejected",
                "; ".join(result.failure_reasons),
            )
            # Supabase: record skip decision (fire-and-forget)
            if self.supabase and supabase_alert_id:
                _fire_and_forget(self.supabase.record_skip(
                    alert_id=supabase_alert_id,
                    failure_reasons=result.failure_reasons,
                    signal_score=signal.score,
                    conviction=supabase_conviction,
                    intended_strike=signal.strike,
                ))
            return None

        # Handle signal flip: close opposite-direction position before opening new
        flip_ticker = ctx.get("signal_flip_ticker")
        if flip_ticker:
            old_dir = ctx.get("signal_flip_old_direction", "?")
            logger.info(
                f"[TradeLifecycle] SIGNAL FLIP: {flip_ticker} "
                f"{old_dir}→{signal.direction.value.lower()}, "
                f"closing old position"
            )
            await self._close_signal_flip(flip_ticker, old_dir)

        logger.info(
            f"[TradeLifecycle] {signal.ticker}: pipeline=APPROVED, "
            f"proceeding to open trade (score={signal.score}, "
            f"premium=${signal.atm_premium:.2f}, strike=${signal.strike})"
        )
        await log_trade_event(
            self.db_path, signal.ticker, "pipeline_approved",
            f"score={signal.score} premium=${signal.atm_premium:.2f} "
            f"strike=${signal.strike} expiry={signal.expiry}",
        )

        # Dip-confirm: when premium is fading, wait for an uptick before buying.
        # This avoids entering trades that are crashing and gets cheaper fills.
        if getattr(self.settings, "ENABLE_DIP_CONFIRM", False):
            confirmed, confirm_premium = await self._wait_for_entry_confirmation(signal)
            if not confirmed:
                logger.info(
                    f"SKIP: {signal.ticker} — dip confirm: no uptick within "
                    f"{self.settings.DIP_CONFIRM_MAX_POLLS} polls, skipping trade"
                )
                await log_trade_event(
                    self.db_path, signal.ticker, "rejected",
                    f"dip_confirm_skipped: no uptick within "
                    f"{self.settings.DIP_CONFIRM_MAX_POLLS} polls",
                )
                return None
            if confirm_premium and confirm_premium != signal.atm_premium:
                logger.info(
                    f"[TradeLifecycle] {signal.ticker}: dip confirm entry at "
                    f"${confirm_premium:.2f} (was ${signal.atm_premium:.2f}, "
                    f"saved {((signal.atm_premium - confirm_premium) / signal.atm_premium * 100):+.1f}%)"
                )
                signal = signal.model_copy(update={"atm_premium": confirm_premium})

        trade_info = await self._open_single_trade(
            signal, signal_id, strategy="B",
            supabase_alert_id=supabase_alert_id,
            nbbo_at_order=nbbo_at_order,
        )

        # Supabase: record execution decision (fire-and-forget)
        if self.supabase and supabase_alert_id and trade_info:
            _fire_and_forget(self.supabase.record_executed(
                alert_id=supabase_alert_id,
                contracts=trade_info.get("contracts", 0),
                intended_contracts=trade_info.get("contracts", 0),
                strike=trade_info.get("strike"),
                conviction=supabase_conviction,
            ))

        return trade_info

    async def _wait_for_entry_confirmation(
        self, signal: TradeSignal,
    ) -> tuple[bool, float | None]:
        """Smart dip-confirm: check underlying vs support before entering a fading trade.

        Returns (should_enter, entry_premium).
        - (True, None) — enter immediately at current price
        - (True, new_premium) — uptick detected, enter at the new (cheaper) price
        - (False, None) — no uptick + breaking support, skip the trade

        Decision tree:
        1. Premium NOT fading (< 1%) → ENTER immediately
        2. Premium fading, underlying ABOVE recent 5m VWAP → ENTER (premium decay, not trend)
        3. Premium fading, underlying NEAR support → WAIT for bounce → enter at dip
        4. Premium fading, underlying BREAKING support → SKIP (no floor)
        5. Premium fading, no candle data → fall back to timer-based polling
        """
        stream = self.market_stream
        if not stream:
            return True, None

        option_type = "put" if signal.direction == Direction.PUT else "call"
        expiry_date = resolve_expiry_date(signal.expiry)
        if not expiry_date:
            return True, None

        t0_premium = signal.atm_premium
        if not t0_premium or t0_premium <= 0:
            return True, None

        max_polls = getattr(self.settings, "DIP_CONFIRM_MAX_POLLS", 6)
        poll_sec = getattr(self.settings, "DIP_CONFIRM_POLL_SEC", 5.0)
        fade_pct = getattr(self.settings, "DIP_CONFIRM_FADE_PCT", 1.0)

        # Subscribe to WS for this contract
        try:
            await stream.subscribe_option(
                signal.ticker, signal.strike, expiry_date, option_type,
            )
        except Exception as exc:
            logger.warning(f"[DipConfirm] {signal.ticker}: WS subscribe failed: {exc}")
            return True, None

        try:
            # Step 1: wait one interval, check if premium is fading
            await asyncio.sleep(poll_sec)

            t1 = await stream.get_option_premium(
                signal.ticker, signal.strike, expiry_date, option_type,
            )
            if t1 is None:
                logger.debug(f"[DipConfirm] {signal.ticker}: no WS premium — entering immediately")
                await log_trade_event(
                    self.db_path, signal.ticker, "dip_confirm_wait",
                    "no_ws_premium — entering immediately",
                )
                return True, None

            fade = (t0_premium - t1) / t0_premium * 100
            if fade < fade_pct:
                logger.info(
                    f"[DipConfirm] {signal.ticker}: premium stable/rising "
                    f"(t0=${t0_premium:.2f} → t1=${t1:.2f}, fade={fade:+.1f}%) — entering now"
                )
                await log_trade_event(
                    self.db_path, signal.ticker, "dip_confirm_wait",
                    f"premium_stable: t0=${t0_premium:.2f} t1=${t1:.2f} fade={fade:+.1f}%",
                )
                return True, t1 if t1 < t0_premium else None

            # Step 2: premium IS fading — check underlying vs support
            support_info = await self._check_support_level(signal.ticker, option_type)

            if support_info:
                at_support, above_vwap, details = support_info
                logger.info(
                    f"[DipConfirm] {signal.ticker}: fading {fade:+.1f}%, "
                    f"support_check: at_support={at_support}, above_vwap={above_vwap}, {details}"
                )

                # Above VWAP = underlying in bullish structure, premium fade is temporary
                if above_vwap and option_type == "call":
                    logger.info(
                        f"[DipConfirm] {signal.ticker}: above VWAP despite fade — entering now"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "dip_confirm_entered",
                        f"above_vwap: fade={fade:+.1f}% but underlying above VWAP — {details}",
                    )
                    return True, t1

                if not above_vwap and option_type == "put":
                    logger.info(
                        f"[DipConfirm] {signal.ticker}: below VWAP for put — entering now"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "dip_confirm_entered",
                        f"below_vwap_put: fade={fade:+.1f}% but underlying below VWAP — {details}",
                    )
                    return True, t1

                # VWAP direction block: counter-trend trades are high risk
                # Put above VWAP = buying puts in bullish structure (e.g. GOOGL bug)
                # Call below VWAP = buying calls in bearish structure
                if above_vwap and option_type == "put":
                    logger.info(
                        f"[DipConfirm] {signal.ticker}: VWAP BLOCK — put above VWAP "
                        f"(counter-trend, fade={fade:+.1f}%) — {details}"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "dip_confirm_vwap_blocked",
                        f"put_above_vwap: fade={fade:+.1f}% underlying above VWAP "
                        f"= counter-trend — SKIPPING — {details}",
                    )
                    return False, None

                if not above_vwap and option_type == "call":
                    logger.info(
                        f"[DipConfirm] {signal.ticker}: VWAP BLOCK — call below VWAP "
                        f"(counter-trend, fade={fade:+.1f}%) — {details}"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "dip_confirm_vwap_blocked",
                        f"call_below_vwap: fade={fade:+.1f}% underlying below VWAP "
                        f"= counter-trend — SKIPPING — {details}",
                    )
                    return False, None

            await log_trade_event(
                self.db_path, signal.ticker, "dip_confirm_wait",
                f"fading: t0=${t0_premium:.2f} t1=${t1:.2f} fade={fade:+.1f}% — polling for uptick",
            )

            # Step 3: poll for uptick (underlying near or below support)
            prev = t1
            low_water = t1  # track the lowest premium seen
            for poll in range(max_polls):
                await asyncio.sleep(poll_sec)
                current = await stream.get_option_premium(
                    signal.ticker, signal.strike, expiry_date, option_type,
                )
                if current is None:
                    continue

                if current < low_water:
                    low_water = current

                if current > prev:
                    # Uptick detected — enter at this price
                    savings_pct = (t0_premium - current) / t0_premium * 100
                    logger.info(
                        f"[DipConfirm] {signal.ticker}: UPTICK on poll {poll + 1}/{max_polls} "
                        f"(${prev:.2f} → ${current:.2f}, low=${low_water:.2f}) "
                        f"— entering at ${current:.2f} (saved {savings_pct:+.1f}%)"
                    )
                    await log_trade_event(
                        self.db_path, signal.ticker, "dip_confirm_entered",
                        f"uptick_poll={poll + 1} prev=${prev:.2f} now=${current:.2f} "
                        f"low=${low_water:.2f} saved={savings_pct:+.1f}%",
                    )
                    return True, current
                prev = current

            # No uptick — skip trade
            total_fade = (t0_premium - prev) / t0_premium * 100 if prev else fade
            logger.info(
                f"[DipConfirm] {signal.ticker}: NO uptick after {max_polls} polls "
                f"(t0=${t0_premium:.2f} → low=${low_water:.2f} → last=${prev:.2f}, "
                f"total_fade={total_fade:+.1f}%) — SKIPPING"
            )
            return False, None

        except Exception as exc:
            logger.warning(f"[DipConfirm] {signal.ticker}: error during poll: {exc}")
            return True, None
        finally:
            try:
                await stream.unsubscribe_option(
                    signal.ticker, signal.strike, expiry_date, option_type,
                )
            except Exception:
                pass

    async def _check_support_level(
        self, ticker: str, option_type: str,
    ) -> tuple[bool, bool, str] | None:
        """Check if the underlying is near multi-TF support and compute VWAP.

        Returns (at_support, above_vwap, detail_string) or None if no data.

        Support uses multi-timeframe wick clustering (5m/15m/1h/4h):
        - Clusters candle lows within 0.15% tolerance bands
        - Requires 3+ touches and at least 1 timeframe confluence
        - "Near" = within 0.3% of a qualifying support level

        VWAP is computed from today's 5m candle session data using
        volume-weighted typical price = (high + low + close) / 3.
        """
        if not hasattr(self, "_candle_cache") or self._candle_cache is None:
            return None

        try:
            candle_data = await asyncio.wait_for(
                self._candle_cache.get_candle_data(ticker), timeout=15,
            )
        except (asyncio.TimeoutError, Exception):
            return None

        bars_5m = candle_data.get("5m", [])
        if len(bars_5m) < 4:
            return None

        # Current price = last bar's close
        current_price = bars_5m[-1].close

        # Multi-TF support via wick clustering (5m, 15m, 1h, 4h)
        at_support_result, support_detail = is_at_support(
            candle_data, current_price=current_price,
            max_distance_pct=0.3, min_strength=3, min_confluence=1,
        )

        # VWAP from today's 5m session bars (volume-weighted typical price)
        total_vw = 0.0
        total_vol = 0.0
        for b in bars_5m:
            if b.volume > 0:
                typical_price = (b.high + b.low + b.close) / 3
                total_vw += typical_price * b.volume
                total_vol += b.volume
        vwap = total_vw / total_vol if total_vol > 0 else current_price
        above_vwap = current_price >= vwap

        detail = (
            f"price=${current_price:.2f} vwap=${vwap:.2f} "
            f"support={at_support_result} ({support_detail[:80]})"
        )
        return at_support_result, above_vwap, detail

    async def _close_signal_flip(self, ticker: str, old_direction: str) -> None:
        """Close open positions on ticker with the given direction (signal flip).

        Called when a new signal arrives for the same ticker but opposite direction.
        Closes at current premium via the normal close flow.
        """
        async with _connect_db(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'open' "
                "AND ticker = ? AND LOWER(option_type) = ?",
                (ticker, old_direction.lower()),
            )
            trades = [dict(row) for row in await cursor.fetchall()]

        for trade in trades:
            # Use entry premium as fallback — position_monitor will handle
            # the actual close with real premium on its next cycle
            exit_premium = trade.get("premium_per_contract", 0)
            exit_price = trade.get("entry_price", 0)
            logger.info(
                f"[TradeLifecycle] SIGNAL FLIP CLOSE: #{trade['id']} "
                f"{ticker} {old_direction} x{trade['contracts']} "
                f"(closing for direction reversal)"
            )
            await log_trade_event(
                self.db_path, ticker, "signal_flip_close",
                f"Closed #{trade['id']} {old_direction} for direction reversal",
            )
            await self.close_trade(
                trade["id"], exit_price, exit_premium, "signal_flip"
            )
            # Also close on Webull if live
            await self.close_webull_position(trade, exit_premium)

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_premium: float,
        reason: str,
    ) -> dict:
        """Close a paper trade and update the correct strategy's portfolio."""
        async with _connect_db(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
            trade = dict(await cursor.fetchone())  # type: ignore[arg-type]

            strategy = trade.get("strategy") or "B"
            contracts = trade["contracts"]
            total_cost = trade["total_cost"]

            actual_exit = exit_premium * (1 - self.settings.SIMULATED_EXIT_SLIPPAGE_BPS / 10000)
            exit_slippage = exit_premium - actual_exit

            proceeds = actual_exit * contracts * 100
            pnl = proceeds - total_cost
            pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0

            now = datetime.now()

            # Calculate trade duration
            duration_minutes = None
            try:
                opened_dt = datetime.fromisoformat(trade["opened_at"])
                duration_minutes = round((now - opened_dt).total_seconds() / 60, 1)
            except (ValueError, TypeError):
                pass

            await conn.execute(
                "UPDATE paper_trades SET status = 'closed', exit_price = ?, exit_premium = ?, "
                "exit_slippage = ?, exit_reason = ?, pnl_dollars = ?, pnl_pct = ?, "
                "duration_minutes = ?, closed_at = ? "
                "WHERE id = ?",
                (exit_price, actual_exit, exit_slippage, reason, pnl, pnl_pct,
                 duration_minutes, now.isoformat(), trade_id),
            )

            portfolio = await _get_or_create_portfolio(conn, self.settings.PORTFOLIO_SIZE, strategy)
            new_balance = portfolio["current_balance"] + proceeds
            today_str = _today_et().strftime("%Y-%m-%d")

            daily_pnl = portfolio["daily_pnl"]
            if portfolio["last_trade_date"] != today_str:
                daily_pnl = 0
            daily_pnl += pnl

            win_col = "wins" if pnl >= 0 else "losses"
            await conn.execute(
                f"UPDATE paper_portfolio SET current_balance = ?, daily_pnl = ?, "
                f"last_trade_date = ?, {win_col} = {win_col} + 1 "
                f"WHERE strategy = ?",
                (new_balance, daily_pnl, today_str, strategy),
            )
            committed = await self._commit_with_retry(
                conn, f"close trade #{trade_id} {trade['ticker']} {reason}"
            )
            if not committed:
                logger.error(
                    f"CRITICAL: Failed to commit close for trade #{trade_id} "
                    f"{trade['ticker']} — DB may show trade still open"
                )

            emoji = "✅" if pnl >= 0 else "❌"
            logger.info(
                f"{emoji} [{strategy}] CLOSED: {trade['ticker']} {trade['option_type'].upper()} "
                f"${trade['strike']} x{contracts} | {reason} "
                f"| PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Bal: ${new_balance:.2f}"
            )

            # Supabase: record close (fire-and-forget)
            s_alert_id = trade.get("supabase_alert_id")
            if self.supabase and s_alert_id and trade.get("webull_order_id"):
                peak_prem = trade.get("mfe_premium")
                _fire_and_forget(self.supabase.record_close(
                    alert_id=s_alert_id,
                    close_price=actual_exit,
                    exit_reason=reason,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl,
                    hold_minutes=duration_minutes,
                    peak_premium=peak_prem,
                ))

            return {
                "trade_id": trade_id,
                "ticker": trade["ticker"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "balance": new_balance,
            }

    async def dca_add_contracts(
        self,
        trade_id: int,
        current_premium: float,
    ) -> dict | None:
        """Add a DCA tranche to an existing open trade.

        Buys additional contracts at the current (lower) premium, updates the
        trade's average entry premium, total cost, and contract count.
        Returns info about the addition, or None if DCA is not possible.
        """
        async with _connect_db(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            trade = await conn.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            )
            trade = await trade.fetchone()
            if not trade or trade["status"] != "open":
                return None

            tranches_left = trade["dca_tranches_remaining"] or 0
            total_target = trade["dca_total_contracts"] or trade["contracts"]
            if tranches_left <= 0:
                return None

            strategy = trade.get("strategy") or "B"
            portfolio = await _get_or_create_portfolio(conn, self.settings.PORTFOLIO_SIZE, strategy)

            # Calculate how many contracts for this tranche
            already_bought = trade["contracts"]
            remaining_to_buy = total_target - already_bought
            if remaining_to_buy <= 0:
                return None

            # Split remaining evenly across remaining tranches
            tranche_contracts = max(1, remaining_to_buy // tranches_left)

            # Apply slippage to current premium
            buy_premium = current_premium * (
                1 + self.settings.SIMULATED_ENTRY_SLIPPAGE_BPS / 10000
            )
            tranche_cost = tranche_contracts * buy_premium * 100

            live_cap = await self._get_effective_balance()
            dca_effective = min(portfolio["current_balance"], live_cap)
            if tranche_cost > dca_effective:
                tranche_contracts = max(
                    1, int(dca_effective / (buy_premium * 100))
                )
                tranche_cost = tranche_contracts * buy_premium * 100
                if tranche_cost > dca_effective:
                    return None

            # Calculate new weighted average premium
            old_cost = trade["contracts"] * trade["premium_per_contract"] * 100
            new_total_contracts = trade["contracts"] + tranche_contracts
            new_total_cost = old_cost + tranche_cost
            new_avg_premium = new_total_cost / (new_total_contracts * 100)

            now = datetime.now().isoformat()
            await conn.execute(
                "UPDATE paper_trades SET contracts = ?, premium_per_contract = ?, "
                "total_cost = ?, dca_tranches_remaining = ?, dca_last_add_at = ? "
                "WHERE id = ?",
                (
                    new_total_contracts,
                    round(new_avg_premium, 4),
                    round(new_total_cost, 2),
                    tranches_left - 1,
                    now,
                    trade_id,
                ),
            )

            new_balance = portfolio["current_balance"] - tranche_cost
            await conn.execute(
                "UPDATE paper_portfolio SET current_balance = ? WHERE strategy = ?",
                (new_balance, strategy),
            )
            await self._commit_with_retry(
                conn, f"DCA add trade #{trade_id} {trade['ticker']}"
            )

            tranche_num = self.settings.DCA_TRANCHES - tranches_left + 1
            logger.info(
                f"📈 [{strategy}] DCA {tranche_num}/{self.settings.DCA_TRANCHES}: "
                f"{trade['ticker']} {trade['option_type'].upper()} "
                f"+{tranche_contracts}x @ ${buy_premium:.2f} "
                f"(avg now ${new_avg_premium:.2f}, total {new_total_contracts}x) "
                f"| Bal: ${new_balance:.2f}"
            )

            return {
                "trade_id": trade_id,
                "tranche": tranche_num,
                "contracts_added": tranche_contracts,
                "buy_premium": buy_premium,
                "new_avg_premium": new_avg_premium,
                "total_contracts": new_total_contracts,
                "balance": new_balance,
            }

    async def partial_close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_premium: float,
        reason: str,
        close_pct: float,
    ) -> dict:
        """Close a percentage of a trade's contracts.

        Splits the original trade into:
        - A closed child row for the portion being sold
        - The original row updated with reduced contracts (still open)

        If close_pct would close all contracts, delegates to close_trade().
        """
        async with _connect_db(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
            trade = dict(await cursor.fetchone())  # type: ignore[arg-type]

            total_contracts = trade["contracts"]
            contracts_to_close = round(total_contracts * close_pct / 100)

            # If rounding means we'd close everything, do a full close
            if contracts_to_close >= total_contracts or contracts_to_close <= 0:
                return await self.close_trade(trade_id, exit_price, exit_premium, reason)

            strategy = trade.get("strategy") or "B"
            remaining_contracts = total_contracts - contracts_to_close

            actual_exit = exit_premium * (1 - self.settings.SIMULATED_EXIT_SLIPPAGE_BPS / 10000)
            exit_slippage = exit_premium - actual_exit

            # Calculate P&L for the closed portion
            cost_per_contract = trade["premium_per_contract"] * 100
            closed_cost = contracts_to_close * cost_per_contract
            proceeds = actual_exit * contracts_to_close * 100
            pnl = proceeds - closed_cost
            pnl_pct = (pnl / closed_cost * 100) if closed_cost > 0 else 0

            now = datetime.now()

            # Calculate duration
            duration_minutes = None
            try:
                opened_dt = datetime.fromisoformat(trade["opened_at"])
                duration_minutes = round((now - opened_dt).total_seconds() / 60, 1)
            except (ValueError, TypeError):
                pass

            # Insert a closed child row for the partial close
            # Copy Webull IDs so the child can be matched in reconciliation
            cursor = await conn.execute(
                "INSERT INTO paper_trades "
                "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
                "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
                "signal_premium, entry_slippage, "
                "target_1, target_2, target_3, target_4, target_5, "
                "stop_price, exit_by, expiry_date, strategy, "
                "webull_order_id, webull_client_order_id, webull_entry_fill_price, "
                "status, exit_price, exit_premium, exit_slippage, exit_reason, pnl_dollars, pnl_pct, "
                "duration_minutes, opened_at, closed_at, parent_trade_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, "
                "'closed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade["signal_id"],
                    trade["ticker"],
                    trade["direction"],
                    trade["sentiment"],
                    trade["score"],
                    trade["strength"],
                    trade["bot_source"],
                    trade["entry_price"],
                    trade["strike"],
                    trade["option_type"],
                    contracts_to_close,
                    trade["premium_per_contract"],
                    closed_cost,
                    trade.get("signal_premium"),
                    trade.get("entry_slippage"),
                    trade["target_1"],
                    trade["target_2"],
                    trade.get("target_3"),
                    trade.get("target_4"),
                    trade.get("target_5"),
                    trade["stop_price"],
                    trade["exit_by"],
                    trade.get("expiry_date"),
                    strategy,
                    trade.get("webull_order_id"),
                    trade.get("webull_client_order_id"),
                    trade.get("webull_entry_fill_price"),
                    exit_price,
                    actual_exit,
                    exit_slippage,
                    reason,
                    pnl,
                    pnl_pct,
                    duration_minutes,
                    trade["opened_at"],
                    now.isoformat(),
                    trade_id,
                ),
            )
            child_trade_id = cursor.lastrowid

            # Update the original row: reduce contracts and total_cost
            remaining_cost = remaining_contracts * cost_per_contract
            await conn.execute(
                "UPDATE paper_trades SET contracts = ?, total_cost = ? WHERE id = ?",
                (remaining_contracts, remaining_cost, trade_id),
            )

            # Update portfolio: add proceeds, record win/loss
            portfolio = await _get_or_create_portfolio(conn, self.settings.PORTFOLIO_SIZE, strategy)
            new_balance = portfolio["current_balance"] + proceeds
            today_str = _today_et().strftime("%Y-%m-%d")

            daily_pnl = portfolio["daily_pnl"]
            if portfolio["last_trade_date"] != today_str:
                daily_pnl = 0
            daily_pnl += pnl

            win_col = "wins" if pnl >= 0 else "losses"
            await conn.execute(
                f"UPDATE paper_portfolio SET current_balance = ?, daily_pnl = ?, "
                f"last_trade_date = ?, {win_col} = {win_col} + 1 "
                f"WHERE strategy = ?",
                (new_balance, daily_pnl, today_str, strategy),
            )
            await self._commit_with_retry(
                conn, f"partial close trade #{trade_id} {trade['ticker']}"
            )

            logger.info(
                f"🔀 [{strategy}] PARTIAL: {trade['ticker']} {trade['option_type'].upper()} "
                f"${trade['strike']} — {contracts_to_close}/{total_contracts} | {reason} "
                f"| PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Left: {remaining_contracts} "
                f"| Bal: ${new_balance:.2f}"
            )

            return {
                "trade_id": trade_id,
                "child_trade_id": child_trade_id,
                "ticker": trade["ticker"],
                "contracts_closed": contracts_to_close,
                "contracts_remaining": remaining_contracts,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "balance": new_balance,
            }

    async def _get_current_price(self, ticker: str) -> float | None:
        """Fetch the current underlying price for a ticker.

        Uses yfinance as a quick lookup. Returns None if unavailable.
        """
        import asyncio

        try:
            import yfinance as yf

            def _fetch():
                tk = yf.Ticker(ticker)
                info = tk.fast_info
                return float(info.get("lastPrice", 0) or info.get("last_price", 0))

            price = await asyncio.to_thread(_fetch)
            return price if price > 0 else None
        except Exception as exc:
            logger.debug(f"Could not fetch current price for {ticker}: {exc}")
            return None

    async def get_status(self) -> str:
        """Get current portfolio status."""
        async with _connect_db(self.db_path) as conn:
            live_cap = await self._get_effective_balance()
            portfolio = await _get_or_create_portfolio(conn, live_cap, "B")

            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE status = 'open'")
            open_trades = [dict(r) for r in await cursor.fetchall()]

        total_pnl = portfolio["current_balance"] - portfolio["starting_balance"]
        total_pnl_pct = total_pnl / portfolio["starting_balance"] * 100 if portfolio["starting_balance"] else 0

        lines = [
            "=" * 50,
            "  OPTIONS OWL — PAPER PORTFOLIO",
            "=" * 50,
            f"  Starting:  ${portfolio['starting_balance']:.2f}",
            f"  Current:   ${portfolio['current_balance']:.2f}",
            f"  Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.1f}%)",
            f"  Daily PnL: ${portfolio['daily_pnl']:+.2f}",
            f"  Trades:    {portfolio['total_trades']} ({portfolio['wins']}W / {portfolio['losses']}L)",
            "",
        ]

        if open_trades:
            lines.append(f"  Open Positions ({len(open_trades)}):")
            for t in open_trades:
                lines.append(
                    f"    {t['ticker']} {t['option_type'].upper()} ${t['strike']} "
                    f"x{t['contracts']} @ ${t['premium_per_contract']:.2f} "
                    f"(cost: ${t['total_cost']:.2f})"
                )
        else:
            lines.append("  No open positions")

        lines.append("=" * 50)
        return "\n".join(lines)
