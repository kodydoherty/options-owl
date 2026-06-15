"""Circuit breakers — halt trading when risk thresholds are breached."""

from __future__ import annotations

from datetime import datetime, timedelta, time

from loguru import logger

from options_owl.journal.db import connect as _connect_db

try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone as _tz

    ET = _tz(timedelta(hours=-5))  # type: ignore[assignment]

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def _now_et() -> datetime:
    """Return current time in US/Eastern."""
    return datetime.now(tz=ET)


class CircuitBreaker:
    """Collection of circuit-breaker checks that can block new trades or
    force-close existing positions when risk limits are hit."""

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    async def check_consecutive_losses(
        db_path: str,
        max_consecutive: int = 3,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) if the last *max_consecutive* closed trades were all losses.

        A win (pnl_dollars >= 0) anywhere in the tail resets the streak.
        """
        try:
            # Only count today's losses — consecutive loser resets daily.
            # closed_at is stored as naive UTC; convert today's ET boundaries.
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            now_et = datetime.now(tz=ZoneInfo("America/New_York"))
            today_start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end_et = today_start_et + timedelta(days=1)
            utc = ZoneInfo("UTC")
            start_utc = today_start_et.astimezone(utc).strftime("%Y-%m-%d %H:%M:%S")
            end_utc = today_end_et.astimezone(utc).strftime("%Y-%m-%d %H:%M:%S")

            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT pnl_dollars FROM paper_trades "
                    "WHERE status = 'closed' "
                    "AND parent_trade_id IS NULL "
                    "AND COALESCE(strategy, 'B') = 'B' "
                    "AND closed_at >= ? AND closed_at < ? "
                    "ORDER BY closed_at DESC LIMIT ?",
                    (start_utc, end_utc, max_consecutive),
                )
                rows = await cursor.fetchall()

            if len(rows) < max_consecutive:
                return False, ""

            # All must be losses (< 0) to trigger the breaker
            if all(row[0] < 0 for row in rows):
                return True, (
                    f"Consecutive-loss breaker: last {max_consecutive} trades were all losses"
                )
            return False, ""
        except Exception as exc:
            logger.warning(f"Circuit breaker consecutive-loss check failed: {exc}")
            return False, ""

    @staticmethod
    async def check_drawdown_from_peak(
        db_path: str,
        portfolio_size: float,
        max_drawdown_pct: float = 15.0,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) if portfolio has dropped > *max_drawdown_pct* %
        from its high-water mark (peak balance recorded in paper_portfolio).
        """
        try:
            async with _connect_db(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT current_balance, starting_balance "
                    "FROM paper_portfolio ORDER BY id DESC LIMIT 1"
                )
                row = await cursor.fetchone()

            if row is None:
                return False, ""

            current_balance = float(row[0])
            starting_balance = float(row[1])
            # Peak is the maximum of starting balance and current balance seen
            # (starting_balance serves as the initial high-water mark)
            peak = max(starting_balance, portfolio_size)

            if peak <= 0:
                return False, ""

            drawdown_pct = (peak - current_balance) / peak * 100.0

            if drawdown_pct >= max_drawdown_pct:
                return True, (
                    f"Drawdown breaker: portfolio down {drawdown_pct:.1f}% from peak "
                    f"(${current_balance:.2f} vs peak ${peak:.2f}, limit {max_drawdown_pct:.0f}%)"
                )
            return False, ""
        except Exception as exc:
            logger.warning(f"Circuit breaker drawdown check failed: {exc}")
            return False, ""

    @staticmethod
    def check_opening_buffer(minutes: int = 10) -> tuple[bool, str]:
        """Return (blocked, reason) if we are within the first *minutes* of market open.

        Market opens at 9:30 ET, so with the default 10 minutes this blocks
        trades between 9:30 and 9:40 ET.
        """
        now = _now_et()
        # Only applies on weekdays
        if now.weekday() >= 5:
            return False, ""

        open_dt = now.replace(
            hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0,
        )
        buffer_end = open_dt + timedelta(minutes=minutes)

        if open_dt <= now.replace(tzinfo=now.tzinfo) < buffer_end:
            return True, (
                f"Opening buffer: within first {minutes} min of market open "
                f"({now.strftime('%H:%M')} ET)"
            )
        return False, ""

    @staticmethod
    def check_closing_buffer(minutes: int = 15) -> tuple[bool, str]:
        """Return (blocked, reason) if we are within the last *minutes* before market close.

        Market closes at 16:00 ET, so with the default 15 minutes this blocks
        trades from 15:45 ET onward.
        """
        now = _now_et()
        if now.weekday() >= 5:
            return False, ""

        close_dt = now.replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0,
        )
        buffer_start = close_dt - timedelta(minutes=minutes)

        if buffer_start <= now.replace(tzinfo=now.tzinfo) < close_dt:
            return True, (
                f"Closing buffer: within last {minutes} min before close "
                f"({now.strftime('%H:%M')} ET)"
            )
        return False, ""

    # ------------------------------------------------------------------
    # Aggregate check
    # ------------------------------------------------------------------

    @classmethod
    async def check_all(
        cls,
        db_path: str,
        settings: object,
    ) -> tuple[bool, list[str]]:
        """Run all enabled circuit-breaker checks and return (approved, reasons).

        *approved* is True when **no** breakers fired.
        """
        reasons: list[str] = []

        max_consecutive = getattr(settings, "CB_MAX_CONSECUTIVE_LOSSES", 3)
        max_drawdown_pct = getattr(settings, "CB_MAX_DRAWDOWN_PCT", 15.0)
        portfolio_size = getattr(settings, "PORTFOLIO_SIZE", 0.0)
        opening_minutes = getattr(settings, "CB_OPENING_BUFFER_MINUTES", 10)
        closing_minutes = getattr(settings, "CB_CLOSING_BUFFER_MINUTES", 15)

        # Consecutive losses
        blocked, reason = await cls.check_consecutive_losses(db_path, max_consecutive)
        if blocked:
            reasons.append(reason)

        # Drawdown from peak
        blocked, reason = await cls.check_drawdown_from_peak(
            db_path, portfolio_size, max_drawdown_pct,
        )
        if blocked:
            reasons.append(reason)

        # Opening buffer
        blocked, reason = cls.check_opening_buffer(opening_minutes)
        if blocked:
            reasons.append(reason)

        # Closing buffer
        blocked, reason = cls.check_closing_buffer(closing_minutes)
        if blocked:
            reasons.append(reason)

        approved = len(reasons) == 0
        return approved, reasons

    # ------------------------------------------------------------------
    # Emergency close
    # ------------------------------------------------------------------

    @staticmethod
    async def emergency_close_all(paper_trader: object) -> int:
        """Force-close every open position. Returns the number of trades closed.

        *paper_trader* is expected to be a ``PaperTrader`` instance (typed as
        ``object`` to avoid circular imports at module level).
        """
        from options_owl.execution.paper_trader import get_open_trades

        db_path = getattr(paper_trader, "db_path", "")
        trades = await get_open_trades(db_path)

        if not trades:
            logger.info("Emergency close: no open positions to close")
            return 0

        closed = 0
        for trade in trades:
            try:
                # Use entry premium as exit premium (worst-case approximation)
                await paper_trader.close_trade(  # type: ignore[attr-defined]
                    trade_id=trade["id"],
                    exit_price=trade["entry_price"],
                    exit_premium=0.01,  # near-zero to represent emergency liquidation
                    reason="emergency_circuit_breaker",
                )
                closed += 1
            except Exception as exc:
                logger.error(f"Emergency close failed for trade #{trade['id']}: {exc}")

        logger.warning(f"Emergency close: liquidated {closed}/{len(trades)} positions")
        return closed
