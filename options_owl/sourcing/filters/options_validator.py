"""Options chain validation: spread, premium cap, volume."""

from __future__ import annotations


def validate_chain(chain: dict, score: int) -> tuple[bool, str]:
    """Validate options chain data.

    Returns (is_valid, reason_if_invalid).
    """
    raise NotImplementedError("Phase 3: implement options chain validation")
