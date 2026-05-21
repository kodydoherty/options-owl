"""Pydantic models for scan results, scores, state."""

from __future__ import annotations

from pydantic import BaseModel


class ScanResult(BaseModel):
    """Result of a single ticker evaluation in a scan cycle."""

    ticker: str
    direction: str | None = None
    score: int = 0
    state: str = "INIT"
    filter_result: str = ""
    filter_reason: str = ""
    strike: float | None = None
    premium: float | None = None
    emitted: bool = False
