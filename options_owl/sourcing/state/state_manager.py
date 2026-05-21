"""SQLite-backed state: cooldowns, counters, circuit breaker."""

from __future__ import annotations


class StateManager:
    """Manages all persistent scanner state in state.db."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Create tables if they don't exist."""
        raise NotImplementedError("Phase 3: implement state manager")

    async def refresh(self) -> None:
        """Load current state at start of scan cycle."""
        raise NotImplementedError

    async def record_cooldown(self, ticker: str, direction: str) -> None:
        raise NotImplementedError

    async def is_on_cooldown(self, ticker: str, direction: str) -> bool:
        raise NotImplementedError

    async def increment_alert_counter(self, ticker: str, direction: str) -> None:
        raise NotImplementedError

    async def check_circuit_breaker(self) -> bool:
        raise NotImplementedError
