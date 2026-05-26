"""PostgreSQL-backed state: cooldowns, counters, circuit breaker."""

from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from options_owl.sourcing import db


class StateManager:
    """Manages all persistent scanner state in PostgreSQL."""

    async def record_cooldown(self, ticker: str, direction: str, minutes: int = 90) -> None:
        """Record a cooldown after emitting a signal."""
        await db.execute(
            """
            INSERT INTO cooldowns (ticker, direction, last_alert_at, cooldown_minutes)
            VALUES ($1, $2, NOW(), $3)
            ON CONFLICT (ticker, direction)
            DO UPDATE SET last_alert_at = NOW(), cooldown_minutes = $3
            """,
            ticker, direction, minutes,
        )

    async def is_on_cooldown(self, ticker: str, direction: str) -> bool:
        """Check if ticker+direction is still in cooldown."""
        row = await db.fetchrow(
            """
            SELECT last_alert_at, cooldown_minutes FROM cooldowns
            WHERE ticker = $1 AND direction = $2
            """,
            ticker, direction,
        )
        if row is None:
            return False

        expires_at = row["last_alert_at"] + timedelta(minutes=row["cooldown_minutes"])
        return datetime.now(tz=row["last_alert_at"].tzinfo) < expires_at

    async def is_opposite_cooldown(self, ticker: str, direction: str) -> bool:
        """Check 30-min opposite-direction cooldown."""
        opposite = "BEARISH" if direction == "BULLISH" else "BULLISH"
        row = await db.fetchrow(
            "SELECT last_alert_at FROM cooldowns WHERE ticker = $1 AND direction = $2",
            ticker, opposite,
        )
        if row is None:
            return False
        expires_at = row["last_alert_at"] + timedelta(minutes=30)
        return datetime.now(tz=row["last_alert_at"].tzinfo) < expires_at

    async def increment_alert_counter(self, ticker: str, direction: str) -> None:
        """Increment daily alert counter."""
        await db.execute(
            """
            INSERT INTO alert_counters (date, ticker, direction, count)
            VALUES (CURRENT_DATE, $1, $2, 1)
            ON CONFLICT (date, ticker, direction)
            DO UPDATE SET count = alert_counters.count + 1
            """,
            ticker, direction,
        )

    async def get_daily_alert_count(self, ticker: str | None = None) -> int:
        """Get total alerts emitted today (optionally filtered by ticker)."""
        if ticker:
            return await db.fetchval(
                "SELECT COALESCE(SUM(count), 0) FROM alert_counters WHERE date = CURRENT_DATE AND ticker = $1",
                ticker,
            )
        return await db.fetchval(
            "SELECT COALESCE(SUM(count), 0) FROM alert_counters WHERE date = CURRENT_DATE",
        )

    async def check_circuit_breaker(self) -> bool:
        """Returns True if circuit breaker is tripped."""
        row = await db.fetchrow("SELECT is_tripped FROM circuit_breaker WHERE id = 1")
        if row is None:
            return False
        return row["is_tripped"]

    async def trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        await db.execute(
            """
            INSERT INTO circuit_breaker (id, is_tripped, trip_count, tripped_at, reason)
            VALUES (1, TRUE, 1, NOW(), $1)
            ON CONFLICT (id)
            DO UPDATE SET is_tripped = TRUE,
                         trip_count = circuit_breaker.trip_count + 1,
                         tripped_at = NOW(),
                         reason = $1
            """,
            reason,
        )
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")

    async def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        await db.execute(
            """
            UPDATE circuit_breaker SET is_tripped = FALSE, reset_at = NOW()
            WHERE id = 1
            """,
        )
