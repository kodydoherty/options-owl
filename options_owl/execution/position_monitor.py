"""Background position monitor — checks open paper trades and closes them on target/stop/expiry."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

import discord
import yfinance as yf
from loguru import logger

from options_owl.journal.db import connect as _connect_db

from options_owl.execution.alerts import (
    alert_expiry_danger,
    alert_exit_error,
    alert_force_closed,
    alert_position_mismatch,
    alert_premium_blackout,
    clear_alerts_for_trade,
)
from options_owl.execution.paper_trader import PaperTrader, get_open_trades, log_trade_event
from options_owl.risk.pipeline import run_exit_pipeline

if TYPE_CHECKING:
    from options_owl.collectors.market_data_stream import MarketDataStream

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US Eastern timezone offset (EST = UTC-5, EDT = UTC-4).
# We use a fixed UTC-5 and account for DST via Python zoneinfo when available.
try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone as _tz

    ET = _tz(timedelta(hours=-5))  # type: ignore[assignment]

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Delta approximation for ATM options (used as fallback)
ATM_DELTA = 0.50

POLL_INTERVAL_SECONDS = 5

# Track consecutive premium failures per trade for alerting
_premium_fail_count: dict[int, int] = {}
PREMIUM_FAIL_ALERT_THRESHOLD = 3  # alert after 3 consecutive failures

# Per-trade premium history for velocity/deceleration calculations.
# Keys are trade IDs, values are lists of (timestamp, premium) tuples.
_premium_histories: dict[int, list[tuple[float, float]]] = {}

# Per-trade underlying price history for volume-peak detection (v2.1 §6).
_underlying_price_histories: dict[int, list[float]] = {}

# Per-trade peak underlying price for underlying-anchored trail (v2.1 §5).
_peak_underlying_prices: dict[int, float] = {}

# Per-trade bounce-fade state for v3 exit (persists across poll cycles).
# Keys are trade IDs, values are {"low": float, "detected": bool, "high": float}.
_bounce_states: dict[int, dict] = {}

# Per-trade thesis-cut state for v3 exit (persists across poll cycles).
# Keys are trade IDs, values are {"ticks_in_zone": int}.
_thesis_cut_states: dict[int, dict] = {}

# Track last date we synced portfolio from Webull (once per trading day)
_last_portfolio_sync_date: str = ""


def _cleanup_trade_state(trade_id: int) -> None:
    """Clean up all per-trade state dicts when a trade is fully closed."""
    clear_alerts_for_trade(trade_id)
    _premium_histories.pop(trade_id, None)
    _underlying_price_histories.pop(trade_id, None)
    _peak_underlying_prices.pop(trade_id, None)
    _bounce_states.pop(trade_id, None)
    _thesis_cut_states.pop(trade_id, None)
    _premium_fail_count.pop(trade_id, None)
    _v6_dca_fired.discard(trade_id)
    if _v5_bridge is not None:
        _v5_bridge.cleanup_trade(trade_id)

# Shared candle cache instance (initialized on first use)
_candle_cache: object | None = None  # CandleCache instance, lazy-initialized

# Track last candle fetch time per ticker (fetch once per minute, not every 5s)
_last_candle_fetch: dict[str, float] = {}
CANDLE_FETCH_INTERVAL = 60  # seconds between candle refreshes

# Track last position reconciliation time (every 5 minutes)
_last_reconciliation_time: float = 0.0
RECONCILIATION_INTERVAL = 300  # seconds between reconciliation checks

# Track last Supabase account state push time
_last_supabase_account_push: float = 0.0

# V5 FSM bridge (lazy-initialized when EXIT_ENGINE=v5)
_v5_bridge: object | None = None  # V5MonitorBridge instance

# Phantom trade detection: consecutive miss counts per trade_id
_phantom_miss_counts: dict[int, int] = {}

# V6 DCA: tracks which trade IDs have already fired DCA (one-shot per trade)
_v6_dca_fired: set[int] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_et() -> datetime:
    """Return current time in US/Eastern."""
    return datetime.now(tz=ET)


def _is_market_hours() -> bool:
    """True if the current time is within regular US equity market hours (Mon-Fri 9:30-16:00 ET)."""
    now = _now_et()
    # Weekend check (Mon=0, Sun=6)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def _fetch_current_price(ticker: str) -> float | None:
    """Fetch the latest price for *ticker* via yfinance (synchronous)."""
    try:
        tk = yf.Ticker(ticker)
        # fast_info.last_price is the quickest path in yfinance
        price = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
        if price and price > 0:
            return float(price)
        # Fallback: most recent close from 1-day history
        hist = tk.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch price for {ticker}: {e}")
        return None


async def _fetch_price_async(ticker: str) -> float | None:
    return await asyncio.to_thread(_fetch_current_price, ticker)


# ---------------------------------------------------------------------------
# Option chain lookup
# ---------------------------------------------------------------------------


def _fetch_option_chain_for_ticker(ticker: str, expiry_date: str) -> dict | None:
    """Fetch the full option chain for *ticker* at *expiry_date* (YYYY-MM-DD).

    Returns a dict with keys 'calls' and 'puts', each a pandas DataFrame,
    or None if the chain could not be retrieved.
    """
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry_date)
        return {"calls": chain.calls, "puts": chain.puts}
    except Exception as e:
        logger.debug(f"Could not fetch option chain for {ticker} exp={expiry_date}: {e}")
        return None


def _lookup_premium_from_chain(
    chain: dict,
    strike: float,
    option_type: str,
) -> float | None:
    """Look up the bid/ask midpoint for a specific strike in a cached option chain.

    *chain* is the dict returned by ``_fetch_option_chain_for_ticker``.
    *option_type* is ``"call"`` or ``"put"``.

    Returns the midpoint premium, or ``lastPrice`` as fallback, or None if
    the contract is not found or data is unusable.
    """
    df = chain.get("calls" if option_type == "call" else "puts")
    if df is None or df.empty:
        return None

    # Filter by strike — allow a tiny tolerance for float comparison
    matches = df[abs(df["strike"] - strike) < 0.01]
    if matches.empty:
        return None

    row = matches.iloc[0]

    # Try bid/ask midpoint first
    bid = row.get("bid")
    ask = row.get("ask")

    if (
        bid is not None
        and ask is not None
        and not (isinstance(bid, float) and math.isnan(bid))
        and not (isinstance(ask, float) and math.isnan(ask))
        and bid > 0
        and ask > 0
    ):
        midpoint = (bid + ask) / 2.0
        return round(midpoint, 2)

    # Fallback: lastPrice
    last = row.get("lastPrice")
    if (
        last is not None
        and not (isinstance(last, float) and math.isnan(last))
        and last > 0
    ):
        return round(float(last), 2)

    return None


def _resolve_expiry_for_lookup(trade: dict) -> str | None:
    """Determine the option expiry date string (YYYY-MM-DD) for a trade.

    Uses the stored ``expiry_date`` column first.  Falls back to today's
    date if the trade was opened today (likely 0DTE).
    """
    expiry_date = trade.get("expiry_date")
    if expiry_date:
        return expiry_date

    # Legacy trades without expiry_date — assume 0DTE (today) if opened today
    opened_at = trade.get("opened_at", "")
    today = _now_et().strftime("%Y-%m-%d")
    if opened_at.startswith(today):
        return today

    return None


async def _fetch_option_chain_async(ticker: str, expiry_date: str) -> dict | None:
    return await asyncio.to_thread(_fetch_option_chain_for_ticker, ticker, expiry_date)


# ---------------------------------------------------------------------------
# Price evaluation
# ---------------------------------------------------------------------------


def _estimate_exit_premium(
    entry_premium: float,
    entry_price: float,
    current_price: float,
    option_type: str,
    delta: float = ATM_DELTA,
) -> float:
    """Estimate current option premium using a simple delta approximation.

    For a call: premium changes by +delta per $1 increase in underlying.
    For a put:  premium changes by +delta per $1 decrease in underlying.
    Clamp at zero — premium cannot go negative.
    """
    price_move = current_price - entry_price
    if option_type == "put":
        # Put gains value when underlying drops
        premium_change = -price_move * delta
    else:
        # Call gains value when underlying rises
        premium_change = price_move * delta

    estimated = entry_premium + premium_change
    return max(estimated, 0.01)  # floor at $0.01


def _check_exit_condition(
    trade: dict,
    current_price: float,
) -> tuple[str | None, str]:
    """Determine if *trade* should be closed given *current_price*.

    Returns (exit_reason, description) or (None, "") if the trade should stay open.
    Priority: stop > T2 > T1 > time expiry.
    """
    option_type = trade["option_type"]  # "call" or "put"
    t1 = trade["target_1"]
    t2 = trade["target_2"]
    # NOTE: Underlying price stops removed — premium-based stops handle risk.
    # Only check target hits here (stop logic lives in the exit pipeline).
    if option_type == "put":
        if t2 is not None and current_price <= t2:
            return "t2_hit", f"T2 hit (price ${current_price:.2f} <= T2 ${t2:.2f})"
        if t1 is not None and current_price <= t1:
            return "t1_hit", f"T1 hit (price ${current_price:.2f} <= T1 ${t1:.2f})"
    else:
        if t2 is not None and current_price >= t2:
            return "t2_hit", f"T2 hit (price ${current_price:.2f} >= T2 ${t2:.2f})"
        if t1 is not None and current_price >= t1:
            return "t1_hit", f"T1 hit (price ${current_price:.2f} >= T1 ${t1:.2f})"

    # Time-based exit
    now = _now_et()
    exit_by = trade.get("exit_by")
    if exit_by:
        try:
            # exit_by is stored as "HH:MM"
            exit_h, exit_m = (int(x) for x in exit_by.split(":"))
            exit_time = now.replace(hour=exit_h, minute=exit_m, second=0, microsecond=0)
            if now >= exit_time:
                return "time_expiry", f"Exit-by time reached ({exit_by} ET)"
        except (ValueError, TypeError):
            pass

    # End-of-day close for 0DTE (15 minutes before market close to avoid pinning risk)
    eod_cutoff = now.replace(hour=15, minute=45, second=0, microsecond=0)
    if now >= eod_cutoff:
        return "eod_expiry", "End-of-day close (15:45 ET cutoff)"

    return None, ""


# ---------------------------------------------------------------------------
# MFE / MAE tracking
# ---------------------------------------------------------------------------


async def _check_v6_dca(
    trade: dict,
    exit_premium: float,
    current_price: float,
    settings,
    paper_trader: PaperTrader,
    db_path: str,
) -> None:
    """V6 DCA: add contracts on a developing-phase dip for whitelisted tickers.

    Backtested: +$4,120 improvement (23 fires across 6 tickers).
    One-shot per trade — once fired, won't re-fire for the same trade_id.

    Conditions (all must be true):
      1. Ticker is in V6_DCA_TICKERS whitelist
      2. 8-20 minutes after entry (developing phase, not too early/late)
      3. Premium dipped 15-35% from entry (meaningful dip, not broken thesis)
      4. Underlying price isn't moving against the position by > 0.5%
    """
    trade_id = trade["id"]
    ticker = trade["ticker"]

    # Time-of-day gate: no DCA after 2PM ET (late-day DCAs amplify theta decay losses)
    if _now_et().hour >= 14:
        return

    # Parse ticker whitelist
    dca_tickers_str = getattr(settings, "V6_DCA_TICKERS", "")
    dca_tickers = {t.strip().upper() for t in dca_tickers_str.split(",") if t.strip()}
    if ticker not in dca_tickers:
        return

    # Check time window
    opened_at = trade.get("opened_at", "")
    if not opened_at:
        return
    try:
        opened_dt = datetime.fromisoformat(opened_at)
        elapsed_min = (datetime.now() - opened_dt).total_seconds() / 60
    except (ValueError, TypeError):
        return

    min_min = getattr(settings, "V6_DCA_MIN_MINUTES", 8.0)
    max_min = getattr(settings, "V6_DCA_MAX_MINUTES", 20.0)
    if not (min_min <= elapsed_min <= max_min):
        return

    # Check dip range
    entry_prem = trade["premium_per_contract"]
    if entry_prem <= 0:
        return
    dip_pct = (entry_prem - exit_premium) / entry_prem * 100

    min_dip = getattr(settings, "V6_DCA_MIN_DIP_PCT", 15.0)
    max_dip = getattr(settings, "V6_DCA_MAX_DIP_PCT", 35.0)
    if not (min_dip <= dip_pct <= max_dip):
        return

    # Check underlying isn't against us
    u_threshold = getattr(settings, "V6_DCA_UNDERLYING_THRESHOLD", 0.5)
    entry_price = trade.get("entry_price", 0.0) or 0.0
    if entry_price > 0 and current_price and current_price > 0:
        u_move = (current_price - entry_price) / entry_price * 100
        option_type = trade.get("option_type", "call")
        if option_type == "call" and u_move < -u_threshold:
            logger.debug(
                f"  #{trade_id} {ticker} V6 DCA blocked: underlying {u_move:+.2f}% against call"
            )
            return
        if option_type == "put" and u_move > u_threshold:
            logger.debug(
                f"  #{trade_id} {ticker} V6 DCA blocked: underlying {u_move:+.2f}% against put"
            )
            return

    # Fire DCA: add contracts, capped by MAX_DCA_POSITION_PCT
    add_contracts = trade["contracts"]
    dca_cap_pct = getattr(settings, "MAX_DCA_POSITION_PCT", 15.0)
    cost_per_contract = exit_premium * 100
    if cost_per_contract > 0:
        balance = getattr(settings, "PORTFOLIO_SIZE", 2000.0)
        try:
            live_bal = await paper_trader.get_portfolio_balance()
            if live_bal and live_bal > 0:
                balance = live_bal
        except Exception:
            pass
        dca_max_spend = balance * (dca_cap_pct / 100)
        dca_max_contracts = int(dca_max_spend / cost_per_contract)
        if dca_max_contracts < 1:
            logger.info(
                f"  #{trade_id} {ticker} V6 DCA blocked: cap {dca_cap_pct}% of "
                f"${balance:.0f} = ${dca_max_spend:.0f} < 1 contract @ ${exit_premium:.2f}"
            )
            return
        if add_contracts > dca_max_contracts:
            logger.info(
                f"  #{trade_id} {ticker} V6 DCA capped: {add_contracts} → {dca_max_contracts} "
                f"contracts (MAX_DCA_POSITION_PCT={dca_cap_pct}%)"
            )
            add_contracts = dca_max_contracts
    _v6_dca_fired.add(trade_id)

    logger.info(
        f"  #{trade_id} {ticker} V6 DCA TRIGGERED: dip {dip_pct:.1f}% "
        f"({min_dip}-{max_dip}% window), elapsed {elapsed_min:.1f}min, "
        f"adding {add_contracts} contracts @ ${exit_premium:.2f}"
    )

    # Determine DCA fill price — Webull fill if available, else paper premium
    dca_fill_price = exit_premium  # default: paper quote
    new_total_contracts = trade["contracts"] + add_contracts

    # Place Webull order FIRST to get real fill price
    if paper_trader.webull_executor is not None:
        logger.info(
            f"  #{trade_id} {ticker} V6 DCA Webull order: "
            f"x{add_contracts} @ ${exit_premium:.2f}"
        )
        try:
            aggress_pct = getattr(settings, "WEBULL_ENTRY_AGGRESS_PCT", 2.0)
            aggressive_limit = round(exit_premium * (1 + aggress_pct / 100), 2)
            result = await paper_trader.webull_executor.buy_option(
                ticker=ticker,
                strike=trade["strike"],
                expiry_date=trade.get("expiry_date", "") or "",
                option_type=trade.get("option_type", "call").upper(),
                contracts=add_contracts,
                limit_price=aggressive_limit,
            )
            if result.success:
                # Get actual fill price from Webull
                wb_fill = None
                if result.client_order_id:
                    wb_fill = await paper_trader.webull_executor.get_fill_price(
                        result.client_order_id
                    )
                if wb_fill and wb_fill > 0:
                    dca_fill_price = wb_fill
                    logger.info(
                        f"  #{trade_id} {ticker} V6 DCA Webull FILLED: "
                        f"order_id={result.order_id} fill=${wb_fill:.2f}"
                    )
                else:
                    logger.info(
                        f"  #{trade_id} {ticker} V6 DCA Webull FILLED: "
                        f"order_id={result.order_id} (fill price unavailable, using ${exit_premium:.2f})"
                    )
                # Log trade event for DCA fill
                await log_trade_event(
                    db_path, ticker, "webull_dca_filled",
                    f"trade#{trade_id} order_id={result.order_id} "
                    f"x{add_contracts} @ ${dca_fill_price:.2f} "
                    f"(limit=${aggressive_limit:.2f})",
                    trade_id=trade_id,
                )
            else:
                logger.error(
                    f"  #{trade_id} {ticker} V6 DCA Webull FAILED: {result.error}"
                )
                await log_trade_event(
                    db_path, ticker, "webull_dca_failed",
                    f"trade#{trade_id} error={result.error}",
                    trade_id=trade_id,
                )
        except Exception as exc:
            logger.error(
                f"  #{trade_id} {ticker} V6 DCA Webull ERROR: {exc}"
            )

    # Compute blended entry using the REAL fill price (Webull or paper)
    # Use webull_entry_fill_price for the original entry if available
    orig_entry = trade.get("webull_entry_fill_price") or entry_prem
    old_cost = trade["contracts"] * orig_entry * 100
    new_cost = add_contracts * dca_fill_price * 100
    new_avg_premium = (old_cost + new_cost) / (new_total_contracts * 100)

    try:
        async with _connect_db(db_path) as conn:
            await conn.execute(
                "UPDATE paper_trades SET contracts = ?, premium_per_contract = ?, "
                "total_cost = ? WHERE id = ?",
                (
                    new_total_contracts,
                    round(new_avg_premium, 4),
                    round(old_cost + new_cost, 2),
                    trade_id,
                ),
            )
            await conn.commit()

        logger.info(
            f"  #{trade_id} {ticker} V6 DCA complete: "
            f"{trade['contracts']}+{add_contracts}={new_total_contracts} contracts, "
            f"avg entry ${orig_entry:.2f}+${dca_fill_price:.2f} → ${new_avg_premium:.2f}"
        )

        # Update V5 FSM TradeState if active
        if _v5_bridge is not None:
            bridge_states = getattr(_v5_bridge, "_states", {})
            if trade_id in bridge_states:
                fs = bridge_states[trade_id]
                old_peak = fs.peak_premium
                fs.entry_premium = new_avg_premium
                fs.contracts = new_total_contracts
                # CRITICAL: Reset peak to current premium after DCA.
                # The old peak was relative to the old entry price. Keeping it
                # causes soft_trail/scalp_trail to see a fake "peak gain" and
                # immediately exit on the next cycle. Reset to current premium
                # so exit gates evaluate fresh from the new cost basis.
                fs.peak_premium = exit_premium
                # Also reset the FSM state to GRACE so the trade gets a fresh
                # grace period after averaging in.
                fs.state = "GRACE"
                fs.entry_time = datetime.now()
                logger.info(
                    f"  #{trade_id} V6 DCA: reset FSM — "
                    f"avg=${new_avg_premium:.2f}, contracts={new_total_contracts}, "
                    f"peak ${old_peak:.2f}→${exit_premium:.2f}, state→GRACE"
                )

    except Exception as exc:
        logger.error(f"  #{trade_id} {ticker} V6 DCA DB update failed: {exc}")


async def _update_mfe_mae(
    db_path: str,
    trade_id: int,
    exit_premium: float,
    entry_premium: float,
) -> None:
    """Update Max Favorable Excursion and Max Adverse Excursion for a trade.

    MFE tracks the highest premium seen while the trade is open.
    MAE tracks the lowest premium seen while the trade is open.
    """
    async with _connect_db(db_path) as conn:
        cursor = await conn.execute(
            "SELECT mfe_premium, mae_premium FROM paper_trades WHERE id = ?",
            (trade_id,),
        )
        row = await cursor.fetchone()
        current_mfe = row[0] if row and row[0] is not None else entry_premium
        current_mae = row[1] if row and row[1] is not None else entry_premium

        new_mfe = max(current_mfe, exit_premium)
        new_mae = min(current_mae, exit_premium)

        if new_mfe != current_mfe or new_mae != current_mae:
            mfe_pnl = (
                (new_mfe - entry_premium) / entry_premium * 100
                if entry_premium > 0
                else 0
            )
            mae_pnl = (
                (new_mae - entry_premium) / entry_premium * 100
                if entry_premium > 0
                else 0
            )
            await conn.execute(
                "UPDATE paper_trades SET mfe_premium=?, mae_premium=?, "
                "mfe_pnl_pct=?, mae_pnl_pct=? WHERE id=?",
                (new_mfe, new_mae, mfe_pnl, mae_pnl, trade_id),
            )
            await conn.commit()


# ---------------------------------------------------------------------------
# Position reconciliation — auto-recover orphaned Webull positions
# ---------------------------------------------------------------------------


async def _reconcile_positions(
    paper_trader: PaperTrader,
    discord_client: discord.Client | None,
) -> None:
    """Compare open Webull positions against paper DB and auto-recover mismatches.

    1. Webull has position but paper DB doesn't → create a paper trade so the
       monitor picks it up and manages exit like any other trade.
    2. Paper DB has open trade with webull_order_id but Webull doesn't have it →
       position was already closed externally (expired, manual close); close it in DB.
    """
    global _last_reconciliation_time
    import time

    now = time.time()
    if now - _last_reconciliation_time < RECONCILIATION_INTERVAL:
        return
    _last_reconciliation_time = now

    if paper_trader.webull_executor is None or paper_trader.settings.PAPER_TRADE:
        return

    try:
        webull_positions = await paper_trader.webull_executor.get_open_option_positions()
    except Exception as exc:
        logger.debug(f"Position reconciliation: failed to fetch Webull positions: {exc}")
        return

    db_path = paper_trader.db_path
    open_trades = await get_open_trades(db_path)

    # Build a set of (ticker, strike, option_type, expiry) for paper DB
    db_keys: dict[tuple, dict] = {}
    for trade in open_trades:
        key = (
            trade["ticker"].upper(),
            round(trade["strike"], 2),
            trade["option_type"].lower(),
            trade.get("expiry_date", ""),
        )
        db_keys[key] = trade

    # Build a set for Webull
    webull_keys: dict[tuple, dict] = {}
    for pos in webull_positions:
        key = (
            pos["ticker"].upper(),
            round(pos["strike"], 2),
            pos["option_type"].lower(),
            pos["expiry_date"],
        )
        webull_keys[key] = pos

    # --- Auto-recover orphaned Webull positions ---
    for key, pos in webull_keys.items():
        if key in db_keys:
            continue

        # Webull has this position but our DB doesn't — create a paper trade
        ticker, strike, option_type, expiry_date = key
        contracts = pos["quantity"]
        direction = "put" if option_type == "put" else "call"

        logger.warning(
            f"RECONCILE: Orphaned Webull position found — "
            f"{ticker} ${strike} {option_type.upper()} exp={expiry_date} "
            f"x{contracts}. Creating paper trade to resume monitoring."
        )

        try:
            import aiosqlite

            # Try to fetch live premium so exit logic works correctly
            live_premium = None
            try:
                from options_owl.collectors.polygon_options import polygon_option_premium
                live_premium = await polygon_option_premium(
                    getattr(paper_trader.settings, "POLYGON_API_KEY", ""),
                    ticker, strike, expiry_date, option_type,
                )
            except Exception:
                pass

            if not live_premium or live_premium <= 0:
                # Fallback: try yfinance chain
                try:
                    chain = await asyncio.to_thread(
                        _fetch_option_chain_for_ticker, ticker, expiry_date
                    )
                    if chain:
                        live_premium = _lookup_premium_from_chain(chain, strike, option_type)
                except Exception:
                    pass

            entry_premium = live_premium if live_premium and live_premium > 0 else 0.01
            underlying_price = _fetch_current_price(ticker) or 0.0
            total_cost = contracts * entry_premium * 100

            now_iso = _now_et().isoformat()
            async with _connect_db(db_path) as conn:
                await conn.execute(
                    "INSERT INTO paper_trades "
                    "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
                    "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
                    "signal_premium, entry_slippage, "
                    "target_1, target_2, target_3, target_4, target_5, "
                    "stop_price, exit_by, expiry_date, strategy, status, opened_at, "
                    "webull_order_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                    (
                        0,              # signal_id (no signal — recovered)
                        ticker,
                        direction,
                        "neutral",      # sentiment unknown
                        80,             # default score (mid-range)
                        "moderate",     # strength unknown
                        "reconcile",    # bot_source: marks this as recovered
                        underlying_price,
                        strike,
                        option_type,
                        contracts,
                        entry_premium,
                        total_cost,
                        None,           # signal_premium
                        None,           # entry_slippage
                        None, None, None, None, None,  # targets
                        None,           # stop_price
                        None,           # exit_by
                        expiry_date,
                        "B",            # strategy
                        now_iso,
                        "reconciled",   # webull_order_id: flag as recovered
                    ),
                )
                await conn.commit()

            logger.info(
                f"RECONCILE: Created paper trade for {ticker} ${strike} "
                f"{option_type.upper()} x{contracts} @ ${entry_premium:.2f} "
                f"— now being monitored"
            )

            if discord_client:
                await alert_position_mismatch(
                    discord_client, paper_trader.settings,
                    f"**Auto-recovered orphaned position**\n"
                    f"{ticker} ${strike} {option_type.upper()} exp={expiry_date} "
                    f"x{contracts} @ ${entry_premium:.2f}\n"
                    f"Created paper trade and resumed monitoring.",
                )
        except Exception as exc:
            logger.error(f"RECONCILE: Failed to create paper trade for {ticker}: {exc}")
            if discord_client:
                await alert_position_mismatch(
                    discord_client, paper_trader.settings,
                    f"**Failed to recover orphaned position!**\n"
                    f"{ticker} ${strike} {option_type.upper()} exp={expiry_date} "
                    f"x{contracts}\nError: {exc}\n**Manual intervention required.**",
                )

    # --- Close phantom DB trades (in DB but gone from Webull) ---
    # Safety: if Webull returned zero positions, skip phantom detection entirely.
    # An empty response likely means API failure, not "all positions closed".
    if not webull_keys and db_keys:
        logger.warning(
            f"RECONCILE: Webull returned 0 positions but DB has "
            f"{len(db_keys)} open trades — skipping phantom detection "
            f"(likely API issue, not real closure)"
        )
    else:
        for key, trade in db_keys.items():
            if key in webull_keys:
                continue
            if not trade.get("webull_order_id"):
                continue  # paper-only trade, not a Webull mismatch

            trade_id = trade["id"]
            ticker = trade["ticker"]

            # Safety: don't phantom-close trades opened less than 30 min ago.
            # Webull position API has propagation delay after fills.
            opened_at = trade.get("opened_at", "")
            if opened_at:
                try:
                    open_time = datetime.fromisoformat(opened_at)
                    # opened_at is stored as naive UTC — compare in UTC
                    if open_time.tzinfo is not None:
                        open_time = open_time.replace(tzinfo=None)
                    now_utc = datetime.utcnow()
                    age_minutes = (now_utc - open_time).total_seconds() / 60
                    if age_minutes < 30:
                        logger.info(
                            f"RECONCILE: #{trade_id} {ticker} not on Webull but only "
                            f"{age_minutes:.0f}min old — skipping (propagation delay)"
                        )
                        continue
                except (ValueError, TypeError):
                    pass

            # Safety: require consecutive misses before phantom-closing.
            # Track miss count per trade_id in module-level dict.
            _phantom_miss_counts[trade_id] = _phantom_miss_counts.get(trade_id, 0) + 1
            miss_count = _phantom_miss_counts[trade_id]
            REQUIRED_MISSES = 3  # must be missing for 3 consecutive reconciliation cycles

            if miss_count < REQUIRED_MISSES:
                logger.warning(
                    f"RECONCILE: #{trade_id} {ticker} not on Webull "
                    f"(miss {miss_count}/{REQUIRED_MISSES}) — waiting for confirmation"
                )
                continue

            logger.warning(
                f"RECONCILE: Phantom DB trade #{trade_id} {ticker} — "
                f"has webull_order_id but no Webull position for "
                f"{REQUIRED_MISSES} consecutive checks. "
                f"Closing in DB (position likely expired or was closed externally)."
            )

            try:
                now_iso = _now_et().isoformat()
                entry = trade.get("premium_per_contract", 0) or 0
                contracts = trade.get("contracts", 1) or 1
                # Use MAE (worst seen) as conservative exit estimate rather than
                # assuming -100% total loss. Falls back to 0 if no MAE tracked yet.
                exit_prem = trade.get("mae_premium") or 0.0
                pnl = (exit_prem - entry) * contracts * 100
                pnl_pct = ((exit_prem - entry) / entry * 100) if entry > 0 else -100.0
                async with _connect_db(db_path) as conn:
                    await conn.execute(
                        "UPDATE paper_trades SET status = 'closed', "
                        "exit_reason = 'reconcile_phantom', closed_at = ?, "
                        "exit_premium = ?, pnl_dollars = ?, pnl_pct = ? "
                        "WHERE id = ?",
                        (now_iso, exit_prem, pnl, pnl_pct, trade_id),
                    )
                    await conn.commit()

                _phantom_miss_counts.pop(trade_id, None)

                logger.info(
                    f"RECONCILE: Closed phantom trade #{trade_id} {ticker} in DB "
                    f"(exit_prem=${exit_prem:.2f}, pnl=${pnl:.2f})"
                )

                if discord_client:
                    await alert_position_mismatch(
                        discord_client, paper_trader.settings,
                        f"**Closed phantom trade** #{trade_id}\n"
                        f"{ticker} ${trade['strike']} {trade['option_type'].upper()} "
                        f"x{trade['contracts']}\n"
                        f"Position no longer on Webull for {REQUIRED_MISSES} checks.",
                    )
            except Exception as exc:
                logger.error(
                    f"RECONCILE: Failed to close phantom trade #{trade_id}: {exc}"
                )

    # Clear miss counts for trades that ARE on Webull (reset if position reappears)
    for key, trade in db_keys.items():
        if key in webull_keys:
            _phantom_miss_counts.pop(trade["id"], None)


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------


async def run_position_monitor(
    paper_trader: PaperTrader,
    market_stream: MarketDataStream | None = None,
    discord_client: discord.Client | None = None,
) -> None:
    """Long-running coroutine that monitors open paper trades.

    Designed to be launched as an ``asyncio.create_task`` from the Discord
    collector's ``on_ready`` hook.

    Parameters
    ----------
    paper_trader : PaperTrader
        The paper trading engine for opening/closing trades.
    market_stream : MarketDataStream | None
        Optional real-time data stream (Polygon / yfinance). When provided,
        prices and option premiums are fetched via the stream instead of
        direct yfinance calls.
    discord_client : discord.Client | None
        Discord client for sending emergency DM alerts.
    """
    db_path = paper_trader.db_path
    polygon_api_key = getattr(paper_trader.settings, "POLYGON_API_KEY", "") or ""
    _subscribed_options: set[int] = set()  # trade IDs with active option WS subscriptions
    poll_interval = POLL_INTERVAL_SECONDS
    if market_stream is not None:
        poll_interval = market_stream.poll_interval
        logger.info(
            f"Position monitor started — using {market_stream.provider.value} stream, "
            f"polling every {poll_interval}s during market hours"
        )
    else:
        logger.info(f"Position monitor started — polling every {poll_interval}s during market hours")

    while True:
        try:
            if not _is_market_hours():
                await asyncio.sleep(300)
                continue

            # Daily portfolio sync from Webull (once per trading day at market open).
            # Only sync when no trades are open to avoid resetting balance mid-trade.
            global _last_portfolio_sync_date
            today_str = _now_et().strftime("%Y-%m-%d")
            if today_str != _last_portfolio_sync_date:
                open_trades = await get_open_trades(db_path)
                if not open_trades:
                    synced = await paper_trader.sync_portfolio_from_webull()
                    if synced is not None:
                        logger.info(f"Daily portfolio sync complete: ${synced:,.2f}")
                        _last_portfolio_sync_date = today_str
                    else:
                        logger.warning("Daily portfolio sync failed — will retry next cycle")
                else:
                    logger.debug(
                        f"Skipping portfolio sync — {len(open_trades)} open trade(s)"
                    )

            trades = await get_open_trades(db_path)
            if not trades:
                await asyncio.sleep(poll_interval)
                continue

            logger.info(f"Position monitor: checking {len(trades)} open trade(s)")

            # Ensure option WS subscriptions for all open trades
            if market_stream is not None:
                open_ids = {t["id"] for t in trades}
                # Subscribe to new trades
                for t in trades:
                    if t["id"] not in _subscribed_options:
                        exp = _resolve_expiry_for_lookup(t)
                        if exp:
                            await market_stream.subscribe_option(
                                t["ticker"], t["strike"], exp, t["option_type"],
                            )
                            _subscribed_options.add(t["id"])
                # Unsubscribe closed trades
                closed_ids = _subscribed_options - open_ids
                if closed_ids:
                    _subscribed_options -= closed_ids

            # Cache option chains per (ticker, expiry_date) for this poll cycle
            # when using legacy yfinance path.
            chain_cache: dict[tuple[str, str], dict | None] = {}

            for trade in trades:
                ticker = trade["ticker"]

                # --- Fetch current underlying price ---
                if market_stream is not None:
                    current_price = await market_stream.get_price(ticker)
                else:
                    current_price = await _fetch_price_async(ticker)

                if current_price is None:
                    logger.warning(f"Position monitor: could not get price for {ticker}, skipping")
                    continue

                # --- Determine current option premium ---
                exit_premium: float | None = None
                expiry_date = _resolve_expiry_for_lookup(trade)

                # Source 0: Webull quote API — disabled.
                # The /trade/security endpoint returns 404 for options;
                # Webull's OpenAPI SDK doesn't support option quote lookups.
                # Polygon WS stream is reliable and lower-latency anyway.

                # Source 1: market stream (Polygon WS or yfinance chain)
                if exit_premium is None and market_stream is not None and expiry_date:
                    exit_premium = await market_stream.get_option_premium(
                        ticker,
                        strike=trade["strike"],
                        expiry=expiry_date,
                        option_type=trade["option_type"],
                    )
                    if exit_premium is not None:
                        logger.debug(
                            f"  {ticker} — stream premium ${exit_premium:.2f} "
                            f"(strike={trade['strike']}, exp={expiry_date})"
                        )

                # Polygon REST direct lookup (when no stream or stream returned None)
                if exit_premium is None and expiry_date and polygon_api_key:
                    from options_owl.collectors.polygon_options import (
                        polygon_option_premium,
                    )

                    exit_premium = await polygon_option_premium(
                        polygon_api_key,
                        ticker,
                        strike=trade["strike"],
                        expiry=expiry_date,
                        option_type=trade["option_type"],
                    )
                    if exit_premium is not None:
                        logger.debug(
                            f"  {ticker} — Polygon premium ${exit_premium:.2f} "
                            f"(strike={trade['strike']}, exp={expiry_date})"
                        )

                # Fallback: yfinance chain lookup
                if exit_premium is None and expiry_date:
                    cache_key = (ticker, expiry_date)
                    if cache_key not in chain_cache:
                        chain_cache[cache_key] = await _fetch_option_chain_async(ticker, expiry_date)

                    chain = chain_cache[cache_key]
                    if chain is not None:
                        exit_premium = _lookup_premium_from_chain(
                            chain,
                            strike=trade["strike"],
                            option_type=trade["option_type"],
                        )
                        if exit_premium is not None:
                            logger.debug(
                                f"  {ticker} — yfinance chain premium ${exit_premium:.2f} "
                                f"(strike={trade['strike']}, exp={expiry_date})"
                            )

                # Final fallback: delta approximation
                if exit_premium is None:
                    exit_premium = _estimate_exit_premium(
                        entry_premium=trade["premium_per_contract"],
                        entry_price=trade["entry_price"],
                        current_price=current_price,
                        option_type=trade["option_type"],
                    )
                    logger.warning(
                        f"  #{trade['id']} {ticker} — using delta-estimated premium "
                        f"${exit_premium:.2f} (stream + chain both failed)"
                    )
                    # Track premium failures and alert
                    tid = trade["id"]
                    _premium_fail_count[tid] = _premium_fail_count.get(tid, 0) + 1
                    if _premium_fail_count[tid] >= PREMIUM_FAIL_ALERT_THRESHOLD and discord_client:
                        await alert_premium_blackout(
                            discord_client, paper_trader.settings, trade, _premium_fail_count[tid],
                        )
                else:
                    # Reset failure count on successful premium fetch
                    _premium_fail_count.pop(trade["id"], None)

                # --- Expiry safety net: force-close near market close ---
                now_et = _now_et()
                expiry_safety_min = getattr(paper_trader.settings, "EXPIRY_SAFETY_MINUTES", 10)
                close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                minutes_to_close = (close_time - now_et).total_seconds() / 60

                # Only force-close trades expiring today (not multi-day options)
                trade_expiry = trade.get("expiry_date", "")
                expires_today = (not trade_expiry) or trade_expiry == now_et.strftime("%Y-%m-%d")
                if 0 < minutes_to_close <= expiry_safety_min and trade["status"] == "open" and expires_today:
                    logger.warning(
                        f"  #{trade['id']} {ticker} — {minutes_to_close:.0f}m to close, "
                        f"FORCE CLOSING (expiry safety)"
                    )
                    if discord_client:
                        await alert_expiry_danger(
                            discord_client, paper_trader.settings, trade, int(minutes_to_close),
                        )
                    try:
                        await paper_trader.close_trade(
                            trade_id=trade["id"],
                            exit_price=current_price,
                            exit_premium=exit_premium,
                            reason="expiry_safety",
                        )
                        await paper_trader.close_webull_position(trade, exit_premium)
                        if discord_client:
                            await alert_force_closed(
                                discord_client, paper_trader.settings, trade, exit_premium,
                                "Expiry safety — force-closed before market close",
                            )
                    except Exception as exc:
                        logger.error(f"  #{trade['id']} expiry safety close failed: {exc}")
                        if discord_client:
                            await alert_exit_error(
                                discord_client, paper_trader.settings, trade, str(exc),
                            )
                    continue

                # Update MFE/MAE before checking exit conditions
                await _update_mfe_mae(
                    db_path, trade["id"], exit_premium, trade["premium_per_contract"],
                )

                # DCA: check if we should add contracts on a dip
                settings = paper_trader.settings
                if (
                    settings.ENABLE_DCA
                    and (trade.get("dca_tranches_remaining") or 0) > 0
                    and exit_premium is not None
                ):
                    entry_prem = trade["premium_per_contract"]
                    dip_pct = (entry_prem - exit_premium) / entry_prem * 100 if entry_prem > 0 else 0

                    # Check time limit
                    opened_at = trade.get("opened_at", "")
                    time_ok = True
                    try:
                        opened_dt = datetime.fromisoformat(opened_at)
                        elapsed_min = (datetime.now() - opened_dt).total_seconds() / 60
                        if elapsed_min > settings.DCA_TIME_LIMIT_MINUTES:
                            time_ok = False
                    except (ValueError, TypeError):
                        pass

                    if not time_ok:
                        logger.debug(
                            f"  #{trade['id']} {ticker} DCA skipped: past time limit"
                        )
                    elif dip_pct < settings.DCA_DIP_PCT:
                        logger.debug(
                            f"  #{trade['id']} {ticker} DCA skipped: dip {dip_pct:.1f}% "
                            f"< threshold {settings.DCA_DIP_PCT:.1f}%"
                        )

                    if time_ok and dip_pct >= settings.DCA_DIP_PCT:
                        logger.info(
                            f"  #{trade['id']} {ticker} DCA triggered: dip {dip_pct:.1f}%"
                        )
                        await paper_trader.dca_add_contracts(
                            trade_id=trade["id"],
                            current_premium=exit_premium,
                        )
                        # Re-fetch trade after DCA update
                        trades_refreshed = await get_open_trades(db_path)
                        for t in trades_refreshed:
                            if t["id"] == trade["id"]:
                                trade = t
                                break

                # V6 DCA: add contracts on a dip during the developing phase
                # Separate from V3 DCA — fires once per trade for whitelisted tickers
                if (
                    getattr(settings, "ENABLE_V6_DCA", False)
                    and exit_premium is not None
                    and trade["id"] not in _v6_dca_fired
                ):
                    await _check_v6_dca(
                        trade, exit_premium, current_price, settings,
                        paper_trader, db_path,
                    )
                    # Re-fetch trade after V6 DCA — contract count may have changed.
                    # CRITICAL: skip exit evaluation this cycle. DCA lowers entry,
                    # which inflates peak_gain and can immediately trigger soft_trail
                    # on the same tick (killed AMZN #126 and MSFT #132 on 2026-05-08).
                    if trade["id"] in _v6_dca_fired:
                        trades_refreshed = await get_open_trades(db_path)
                        for t in trades_refreshed:
                            if t["id"] == trade["id"]:
                                trade = t
                                break
                        logger.info(
                            f"  #{trade['id']} {ticker} V6 DCA just fired — "
                            f"skipping exit eval this cycle to let trade develop"
                        )
                        continue

                # Record premium history for velocity/decel calculations
                trade_id = trade["id"]
                if exit_premium is not None:
                    import time as _time
                    _premium_histories.setdefault(trade_id, []).append(
                        (_time.time(), exit_premium)
                    )
                    # Cap at 200 entries (~16 min at 5s polls) to bound memory
                    if len(_premium_histories[trade_id]) > 200:
                        _premium_histories[trade_id] = _premium_histories[trade_id][-200:]

                # Track underlying price for volume-peak + underlying trail (v2.1)
                if current_price and current_price > 0:
                    _underlying_price_histories.setdefault(trade_id, []).append(current_price)
                    if len(_underlying_price_histories[trade_id]) > 200:
                        _underlying_price_histories[trade_id] = _underlying_price_histories[trade_id][-200:]
                    # Update peak underlying
                    prev_peak = _peak_underlying_prices.get(trade_id, 0.0)
                    opt_type = trade.get("option_type", "call")
                    if opt_type == "call":
                        _peak_underlying_prices[trade_id] = max(prev_peak, current_price)
                    else:
                        # For puts, "peak" means lowest underlying (best for puts)
                        _peak_underlying_prices[trade_id] = (
                            min(prev_peak, current_price) if prev_peak > 0 else current_price
                        )

                # Inject peak_underlying into trade dict for the exit gate
                trade["peak_underlying_price"] = _peak_underlying_prices.get(trade_id, 0.0)

                # Fetch multi-timeframe candle data
                # When market_stream is available, candles are built from WS
                # minute bars (no extra REST calls, always fresh).
                candle_data: dict = {}
                global _candle_cache
                polygon_key = getattr(paper_trader.settings, "POLYGON_API_KEY", "") or ""
                if polygon_key or market_stream is not None:
                    if _candle_cache is None:
                        from options_owl.collectors.candle_cache import CandleCache
                        shared_db = getattr(paper_trader.settings, "SHARED_CANDLE_DB", "") or ""
                        _candle_cache = CandleCache(
                            polygon_key,
                            market_stream=market_stream,
                            shared_db_path=shared_db or None,
                        )

                    try:
                        candle_data = await asyncio.wait_for(
                            _candle_cache.get_candle_data(ticker), timeout=15,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"  {ticker} candle fetch timed out (15s)")
                    except Exception as exc:
                        logger.debug(f"  {ticker} candle fetch failed: {exc}")

                # Max trade loss cap: force-exit if unrealized loss exceeds % of total portfolio
                reason = None
                description = ""
                max_loss_pct = getattr(paper_trader.settings, "MAX_TRADE_LOSS_EXIT_PCT", 0)
                if max_loss_pct > 0 and exit_premium > 0:
                    # Daily combined P&L cap: sum today's realized losses + this
                    # trade's unrealized loss. If total exceeds N% of portfolio,
                    # force-exit to stop the bleeding.
                    entry_prem = trade["premium_per_contract"]
                    contracts = trade["contracts"]
                    unrealized_pnl = (exit_premium - entry_prem) * contracts * 100

                    live_balance = await paper_trader.get_portfolio_balance()
                    portfolio_base = live_balance if live_balance > 0 else paper_trader.settings.PORTFOLIO_SIZE
                    max_loss_dollars = portfolio_base * (max_loss_pct / 100)

                    # Get today's realized P&L from DB
                    today_realized = 0.0
                    try:
                        today_str = _now_et().strftime("%Y-%m-%d")
                        async with _connect_db(db_path) as pconn:
                            cursor = await pconn.execute(
                                "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                                "WHERE status = 'closed' AND date(closed_at) = ? "
                                "AND webull_order_id IS NOT NULL",
                                (today_str,),
                            )
                            row = await cursor.fetchone()
                            today_realized = row[0] if row else 0.0
                    except Exception:
                        pass

                    daily_total = today_realized + unrealized_pnl
                    if daily_total < -max_loss_dollars:
                        reason = "max_loss_cap"
                        description = (
                            f"Daily P&L ${daily_total:.0f} "
                            f"(realized=${today_realized:.0f} + this=${unrealized_pnl:.0f}) "
                            f"exceeds {max_loss_pct}% cap (${-max_loss_dollars:.0f})"
                        )
                        logger.warning(
                            f"  #{trade_id} {ticker} DAILY LOSS CAP: "
                            f"daily=${daily_total:.0f} > cap=${-max_loss_dollars:.0f} "
                            f"({max_loss_pct}% of ${portfolio_base:.0f})"
                        )

                # Run exit engine (v3 pipeline or v5 FSM based on EXIT_ENGINE setting)
                now_et = _now_et()
                exit_engine = getattr(paper_trader.settings, "EXIT_ENGINE", "v3")
                use_v5 = exit_engine in ("v4", "v5")  # v4 accepted for backward compat

                if reason is not None:
                    pass  # max_loss_cap already triggered above, skip FSM
                elif use_v5:
                    # V5 FSM exit engine
                    global _v5_bridge
                    if _v5_bridge is None:
                        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
                        _v5_bridge = V5MonitorBridge(paper_trader.settings)
                        logger.info("EXIT_FSM: Initialized v5 exit engine")
                    reason, description = _v5_bridge.evaluate(
                        trade, exit_premium, current_price, now_et,
                        candle_data=candle_data,
                    )
                else:
                    # V3 legacy exit pipeline
                    exit_ctx = {
                        "trade": trade,
                        "current_price": current_price,
                        "exit_premium": exit_premium,
                        "now_et": now_et,
                        "settings": paper_trader.settings,
                        "premium_history": _premium_histories.get(trade_id, []),
                        "underlying_price_history": _underlying_price_histories.get(trade_id, []),
                        "candle_data": candle_data,
                        "bounce_state": _bounce_states.get(trade_id, {}),
                        "thesis_cut_state": _thesis_cut_states.get(trade_id, {}),
                    }
                    reason, description = await run_exit_pipeline(exit_ctx)

                    # Persist bounce-fade state across poll cycles (v3)
                    if exit_ctx.get("bounce_state"):
                        _bounce_states[trade_id] = exit_ctx["bounce_state"]

                    # Persist thesis-cut state across poll cycles (v3)
                    if exit_ctx.get("thesis_cut_state"):
                        _thesis_cut_states[trade_id] = exit_ctx["thesis_cut_state"]

                # Persist ENRG result (one-shot: prevents re-evaluation) — v3 only
                if not use_v5:
                    enrg_result = exit_ctx.get("enrg_result")
                    if enrg_result and not trade.get("enrg_result"):
                        try:
                            async with _connect_db(db_path) as conn:
                                await conn.execute(
                                    "UPDATE paper_trades SET enrg_result = ? WHERE id = ?",
                                    (enrg_result, trade_id),
                                )
                                await conn.commit()
                            enrg_reason = exit_ctx.get("enrg_reason", "")
                            logger.info(
                                f"  #{trade_id} {ticker} ENRG result persisted: "
                                f"{enrg_result} — {enrg_reason}"
                            )
                        except Exception as exc:
                            logger.debug(f"  #{trade_id} ENRG persist failed: {exc}")

                if reason is None:
                    entry_prem = trade["premium_per_contract"]
                    pnl_pct = (
                        (exit_premium - entry_prem) / entry_prem * 100
                        if entry_prem > 0 else 0
                    )
                    mfe = trade.get("mfe_premium") or entry_prem
                    logger.info(
                        f"  #{trade['id']} {ticker} {trade['option_type'].upper()} "
                        f"${trade['strike']} — price=${current_price:.2f} "
                        f"prem=${exit_premium:.2f} ({pnl_pct:+.1f}%) "
                        f"peak=${mfe:.2f} — HOLD"
                    )
                    continue

                # ML override: when description starts with [ML_HOLD], ML wants
                # to keep the remaining position — only do scale-out partial
                ml_holding = description.startswith("[ML_HOLD]")
                if ml_holding:
                    logger.info(
                        f"Position monitor ML-override scale-out #{trade['id']}: {description}"
                    )
                else:
                    logger.info(f"Position monitor closing trade #{trade['id']}: {description}")

                settings = paper_trader.settings

                # Graduated scale-out at targets, close all on stop/trailing/time/EOD
                # Vinny's strategy: 20% at each target T1-T5
                use_vinny = getattr(settings, "ENABLE_VINNY_STRATEGY", False)
                _SCALE_OUT_TARGETS = {
                    "t1_hit": (1, "SCALE_OUT_T1_PCT"),
                    "t2_hit": (2, "SCALE_OUT_T2_PCT"),
                    "t3_hit": (3, "SCALE_OUT_T3_PCT"),
                    "t4_hit": (4, "SCALE_OUT_T4_PCT"),
                }

                try:
                    # V4 milestone lock: partial close at gain milestones
                    milestone_match = (
                        reason == "milestone_lock"
                        and description
                        and "[MILESTONE_LOCK:" in description
                    )
                    # V6 scale-out at +20%: sell fraction of contracts
                    v6_scaleout_match = (
                        reason == "scaleout_20"
                        and description
                        and "[V6_SCALEOUT:" in description
                    )
                    # Tranche scale-out: close 1/3 of contracts at +25% (v2.1 §4)
                    tranche_match = (
                        reason == "tranche_lock"
                        and description
                        and "[TRANCHE_SCALEOUT:" in description
                    )
                    if v6_scaleout_match:
                        import re as _re
                        m = _re.search(r"\[V6_SCALEOUT:(\d+)\]", description)
                        close_qty = int(m.group(1)) if m else max(1, trade["contracts"] // 3)
                        close_pct = close_qty / trade["contracts"] * 100
                        logger.info(
                            f"  #{trade['id']} {ticker} V6 SCALEOUT: "
                            f"closing {close_qty}/{trade['contracts']} contracts "
                            f"at +{((exit_premium - trade['premium_per_contract']) / trade['premium_per_contract'] * 100):.1f}%"
                        )
                        result = await paper_trader.partial_close_trade(
                            trade_id=trade["id"],
                            exit_price=current_price,
                            exit_premium=exit_premium,
                            reason=reason,
                            close_pct=close_pct,
                        )
                        if result and "contracts_closed" in result:
                            partial_trade = {**trade, "contracts": result["contracts_closed"]}
                            await paper_trader.close_webull_position(
                                partial_trade, exit_premium,
                                child_trade_id=result.get("child_trade_id"),
                            )
                        else:
                            await paper_trader.close_webull_position(trade, exit_premium)
                            _cleanup_trade_state(trade["id"])

                    elif milestone_match:
                        import re as _re
                        m = _re.search(r"\[MILESTONE_LOCK:(\d+)\]", description)
                        close_qty = int(m.group(1)) if m else max(1, trade["contracts"] // 5)
                        close_pct = close_qty / trade["contracts"] * 100
                        logger.info(
                            f"  #{trade['id']} {ticker} V4 MILESTONE LOCK: "
                            f"closing {close_qty}/{trade['contracts']} contracts "
                            f"at +{((exit_premium - trade['premium_per_contract']) / trade['premium_per_contract'] * 100):.1f}%"
                        )
                        result = await paper_trader.partial_close_trade(
                            trade_id=trade["id"],
                            exit_price=current_price,
                            exit_premium=exit_premium,
                            reason=reason,
                            close_pct=close_pct,
                        )
                        if result and "contracts_closed" in result:
                            partial_trade = {**trade, "contracts": result["contracts_closed"]}
                            await paper_trader.close_webull_position(
                                partial_trade, exit_premium,
                                child_trade_id=result.get("child_trade_id"),
                            )
                        else:
                            # Fell back to full close — clean up state
                            await paper_trader.close_webull_position(trade, exit_premium)
                            _cleanup_trade_state(trade["id"])

                    elif tranche_match:
                        import re as _re
                        m = _re.search(r"\[TRANCHE_SCALEOUT:(\d+)\]", description)
                        close_qty = int(m.group(1)) if m else max(1, trade["contracts"] // 3)
                        close_pct = close_qty / trade["contracts"] * 100
                        logger.info(
                            f"  #{trade['id']} {ticker} TRANCHE LOCK: "
                            f"closing {close_qty}/{trade['contracts']} contracts "
                            f"at +{((exit_premium - trade['premium_per_contract']) / trade['premium_per_contract'] * 100):.1f}%"
                        )
                        result = await paper_trader.partial_close_trade(
                            trade_id=trade["id"],
                            exit_price=current_price,
                            exit_premium=exit_premium,
                            reason=reason,
                            close_pct=close_pct,
                        )
                        if result and "contracts_closed" in result:
                            partial_trade = {**trade, "contracts": result["contracts_closed"]}
                            await paper_trader.close_webull_position(
                                partial_trade, exit_premium,
                                child_trade_id=result.get("child_trade_id"),
                            )
                        else:
                            # Fell back to full close — sell ALL on Webull
                            await paper_trader.close_webull_position(trade, exit_premium)
                            _cleanup_trade_state(trade["id"])

                    elif (
                        reason in _SCALE_OUT_TARGETS
                        and (settings.ENABLE_PARTIAL_PROFITS or ml_holding)
                        and (settings.ENABLE_SCALE_OUT or ml_holding)
                        and trade["contracts"] > 1
                    ):
                        target_num, pct_attr = _SCALE_OUT_TARGETS[reason]
                        close_pct = 20.0 if use_vinny else getattr(settings, pct_attr, 50.0)

                        is_highest = True
                        for higher in range(target_num + 1, 6):
                            if trade.get(f"target_{higher}") is not None:
                                is_highest = False
                                break

                        if is_highest and not ml_holding:
                            await paper_trader.close_trade(
                                trade_id=trade["id"],
                                exit_price=current_price,
                                exit_premium=exit_premium,
                                reason=reason,
                            )
                            await paper_trader.close_webull_position(trade, exit_premium)
                            _cleanup_trade_state(trade["id"])
                        else:
                            result = await paper_trader.partial_close_trade(
                                trade_id=trade["id"],
                                exit_price=current_price,
                                exit_premium=exit_premium,
                                reason=reason,
                                close_pct=close_pct,
                            )
                            async with _connect_db(db_path) as conn:
                                await conn.execute(
                                    "UPDATE paper_trades SET last_target_hit = ? WHERE id = ?",
                                    (target_num, trade["id"]),
                                )
                                await conn.commit()
                            # partial_close_trade may fall back to full close when
                            # rounding gives 0 contracts. Detect via "contracts_closed"
                            # key which only exists on real partials.
                            if result and "contracts_closed" in result:
                                webull_trade = {**trade, "contracts": result["contracts_closed"]}
                                await paper_trader.close_webull_position(
                                    webull_trade, exit_premium,
                                    child_trade_id=result.get("child_trade_id"),
                                )
                            else:
                                # Fell back to full close — sell ALL on Webull
                                await paper_trader.close_webull_position(trade, exit_premium)
                                _cleanup_trade_state(trade["id"])
                    else:
                        # T5, stop, trailing_stop, phase_trail,
                        # theta_bleed, time_decay_zone, time, EOD, theta → close all
                        await paper_trader.close_trade(
                            trade_id=trade["id"],
                            exit_price=current_price,
                            exit_premium=exit_premium,
                            reason=reason,
                        )
                        # Re-read contract count from DB — DCA may have added
                        # contracts since we loaded the trade dict this cycle
                        try:
                            async with _connect_db(db_path) as conn:
                                cur = await conn.execute(
                                    "SELECT contracts FROM paper_trades WHERE id = ?",
                                    (trade["id"],),
                                )
                                row = await cur.fetchone()
                                if row and row[0] != trade["contracts"]:
                                    logger.info(
                                        f"  #{trade['id']} {ticker} contract count updated: "
                                        f"{trade['contracts']} → {row[0]} (DCA)"
                                    )
                                    trade = {**trade, "contracts": row[0]}
                        except Exception as exc:
                            logger.warning(f"  #{trade['id']} contract count re-read failed: {exc}")
                        webull_sold = await paper_trader.close_webull_position(trade, exit_premium)

                        # If Webull sell failed, reopen trade so monitor retries
                        # on next cycle with adjusted pricing
                        if not webull_sold and trade.get("webull_order_id"):
                            retry_count = (trade.get("sell_retry_count") or 0)

                            # After 7 failed attempts, the position likely doesn't
                            # exist on Webull (already sold, expired, or exercised).
                            # Force-close in DB to stop the infinite retry loop.
                            MAX_SELL_RETRIES = 7
                            if retry_count >= MAX_SELL_RETRIES:
                                logger.error(
                                    f"  #{trade['id']} {ticker} WEBULL SELL ABANDONED "
                                    f"after {retry_count} attempts — force-closing in DB. "
                                    f"Position may have been sold/expired on Webull already."
                                )
                                # Mark as manual close — position was gone from Webull,
                                # meaning user sold manually or it expired/exercised.
                                # This separates manual exits from AI exits for backtesting.
                                async with _connect_db(db_path) as conn:
                                    await conn.execute(
                                        "UPDATE paper_trades SET exit_source = 'manual' "
                                        "WHERE id = ?",
                                        (trade["id"],),
                                    )
                                    await conn.commit()
                                await log_trade_event(
                                    db_path, ticker, "manual_close_detected",
                                    f"trade#{trade['id']} — Webull sell abandoned after "
                                    f"{retry_count} attempts. Position not found on Webull. "
                                    f"Likely sold manually by user. "
                                    f"exit_premium in DB is approximate market price, "
                                    f"not actual fill.",
                                    trade_id=trade["id"],
                                )
                                if discord_client:
                                    from options_owl.execution.alerts import alert_critical
                                    await alert_critical(
                                        discord_client, paper_trader.settings,
                                        f"SELL ABANDONED: {ticker} ${trade['strike']} "
                                        f"{trade['option_type'].upper()} x{trade['contracts']} "
                                        f"— {retry_count} failed attempts. "
                                        f"Force-closed in DB (exit_source=manual). "
                                        f"CHECK WEBULL MANUALLY for actual fill price.",
                                    )
                                _cleanup_trade_state(trade["id"])
                                continue

                            logger.warning(
                                f"  #{trade['id']} {ticker} WEBULL SELL FAILED "
                                f"(attempt #{retry_count}) — "
                                f"reopening trade for retry with adjusted price"
                            )
                            async with _connect_db(db_path) as conn:
                                await conn.execute(
                                    "UPDATE paper_trades SET status = 'open', "
                                    "exit_reason = NULL, closed_at = NULL, "
                                    "exit_premium = NULL, pnl_dollars = NULL, "
                                    "pnl_pct = NULL WHERE id = ?",
                                    (trade["id"],),
                                )
                                await conn.commit()

                            # Alert on persistent sell failures
                            if retry_count >= 3 and discord_client:
                                from options_owl.execution.alerts import alert_critical
                                await alert_critical(
                                    discord_client, paper_trader.settings,
                                    f"SELL STUCK: {ticker} ${trade['strike']} "
                                    f"{trade['option_type'].upper()} x{trade['contracts']} "
                                    f"— {retry_count} failed sell attempts. "
                                    f"Position still open on Webull, chasing bid.",
                                )
                            # Don't cleanup state — monitor will re-evaluate
                            continue

                        _cleanup_trade_state(trade["id"])

                except Exception as exc:
                    logger.error(f"  #{trade['id']} {ticker} EXIT FAILED: {exc}")
                    if discord_client:
                        await alert_exit_error(
                            discord_client, paper_trader.settings, trade, str(exc),
                        )

        except asyncio.CancelledError:
            logger.info("Position monitor cancelled — shutting down")
            raise
        except Exception:
            logger.exception("Position monitor encountered an error")

        # Periodic position reconciliation (every 5 min)
        try:
            await _reconcile_positions(paper_trader, discord_client)
        except Exception:
            logger.debug("Position reconciliation error", exc_info=True)

        # Supabase account state push (every 5 min during market hours)
        global _last_supabase_account_push
        supabase = getattr(paper_trader, "supabase", None)
        push_interval = getattr(paper_trader.settings, "SUPABASE_ACCOUNT_STATE_INTERVAL_SEC", 300)
        _now_mono = asyncio.get_event_loop().time()
        if supabase and supabase.enabled and (_now_mono - _last_supabase_account_push) >= push_interval:
            try:
                open_trades = await get_open_trades(db_path)
                balance = await paper_trader.get_portfolio_balance()
                # Get daily P&L from DB
                today_str = _now_et().strftime("%Y-%m-%d")
                async with _connect_db(db_path) as conn:
                    cursor = await conn.execute(
                        "SELECT COALESCE(SUM(pnl_dollars), 0) FROM paper_trades "
                        "WHERE status = 'closed' AND date(closed_at) = ? "
                        "AND webull_order_id IS NOT NULL",
                        (today_str,),
                    )
                    daily_pnl = (await cursor.fetchone())[0]
                buying_power = None
                if paper_trader.webull_executor:
                    try:
                        info = await paper_trader.webull_executor.get_account_info()
                        buying_power = info.buying_power
                    except Exception:
                        pass
                await supabase.push_account_state(
                    equity_usd=balance,
                    cash_usd=balance,
                    daily_pnl_usd=daily_pnl,
                    open_positions=len(open_trades),
                    buying_power=buying_power,
                )
                _last_supabase_account_push = _now_mono
            except Exception as exc:
                logger.debug(f"Supabase account state push error: {exc}")

        await asyncio.sleep(poll_interval)
