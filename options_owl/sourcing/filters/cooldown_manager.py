"""Per-ticker, per-direction cooldown enforcement."""

from __future__ import annotations


async def is_on_cooldown(ticker: str, direction: str, db_path: str) -> bool:
    """Check if ticker+direction is still in cooldown.

    Same direction: 90 min. Opposite direction: 30 min.
    """
    raise NotImplementedError("Phase 3: implement cooldown manager")
