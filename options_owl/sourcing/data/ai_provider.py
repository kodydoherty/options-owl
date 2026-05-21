"""Grok AI (xAI) trade analysis — chat completions."""

from __future__ import annotations


async def analyze_trade(ticker: str, direction: str, indicators: dict) -> dict | None:
    """Send trade context to Grok AI for analysis.

    Behind ENABLE_SOURCE_GROK_AI flag. Adds 2-5s latency.
    """
    raise NotImplementedError("Phase 3: wire up xAI chat completions")
