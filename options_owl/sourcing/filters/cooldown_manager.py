"""Per-ticker, per-direction cooldown enforcement.

Uses the PostgreSQL-backed StateManager for persistent cooldown tracking.
"""

from __future__ import annotations

from loguru import logger

from options_owl.sourcing.state.state_manager import StateManager

# Module-level singleton (initialized lazily)
_state: StateManager | None = None


def _get_state() -> StateManager:
    global _state
    if _state is None:
        _state = StateManager()
    return _state


async def is_on_cooldown(ticker: str, direction: str) -> bool:
    """Check if ticker+direction is still in cooldown.

    Same direction: 90 min. Opposite direction: 30 min.
    Returns True if signal should be blocked.
    """
    state = _get_state()

    # Same-direction cooldown (90 min)
    if await state.is_on_cooldown(ticker, direction):
        logger.debug(f"COOLDOWN: {ticker} {direction} still in same-direction cooldown")
        return True

    # Opposite-direction cooldown (30 min)
    if await state.is_opposite_cooldown(ticker, direction):
        logger.debug(f"COOLDOWN: {ticker} {direction} blocked by opposite-direction cooldown")
        return True

    return False


async def record_signal_emitted(ticker: str, direction: str, cooldown_minutes: int = 90) -> None:
    """Record that a signal was emitted, starting the cooldown timer."""
    state = _get_state()
    await state.record_cooldown(ticker, direction, cooldown_minutes)
    await state.increment_alert_counter(ticker, direction)
    logger.info(f"COOLDOWN: recorded {ticker} {direction}, {cooldown_minutes}min cooldown started")


async def check_daily_cap(ticker: str | None = None, max_daily: int = 50) -> bool:
    """Check if daily alert cap has been reached.

    Returns True if cap is exceeded (should block).
    """
    state = _get_state()
    count = await state.get_daily_alert_count(ticker)
    if count >= max_daily:
        logger.warning(f"DAILY CAP: {count}/{max_daily} alerts today (ticker={ticker or 'ALL'})")
        return True
    return False


async def check_circuit_breaker() -> bool:
    """Check if circuit breaker is tripped.

    Returns True if circuit breaker is active (should block ALL signals).
    """
    state = _get_state()
    return await state.check_circuit_breaker()
