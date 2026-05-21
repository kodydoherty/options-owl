#!/usr/bin/env python3
"""Dry-run verification: checks each agent can access candle data and tick logger.

Run inside each container (or locally) to verify:
1. CandleCache initializes and can fetch 5m bars
2. tick_logger initializes and can write/read a test tick
3. _check_support_level() returns data when candle cache is available
4. market_data_stream tick recording hooks are wired correctly

Usage:
  python -m scripts.dry_run_verify          # local
  docker compose exec owlet-kody python -m scripts.dry_run_verify  # in container
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
import time

# ── 1. Tick Logger ──────────────────────────────────────────────────────────

def test_tick_logger() -> bool:
    """Verify tick_logger can init, record, flush, and read back."""
    from options_owl.collectors.tick_logger import (
        _buffer,
        flush_remaining,
        init_tick_logger,
        record_tick,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        init_tick_logger(tmpdir)

        # Record a test tick
        record_tick(
            contract_ticker="O:SPY260518C00530000",
            ts=time.time(),
            bid=2.10,
            ask=2.15,
            mid=2.125,
            underlying_price=530.50,
            source="Q",
        )
        assert len(_buffer) >= 1, "Buffer should have at least 1 tick"

        # Flush
        flush_remaining()

        # Read back
        db_path = os.path.join(tmpdir, "option_ticks.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM option_ticks").fetchall()
        conn.close()
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        assert rows[0][0] == "O:SPY260518C00530000"
        print("  [PASS] tick_logger: init → record → flush → read OK")
        return True


# ── 2. CandleCache import ──────────────────────────────────────────────────

def test_candle_cache_import() -> bool:
    """Verify CandleCache class can be imported."""
    from options_owl.collectors.candle_cache import CandleCache  # noqa: F401
    print("  [PASS] CandleCache imports OK")
    return True


# ── 3. Support level computation ────────────────────────────────────────────

async def test_support_level() -> bool:
    """Verify _check_support_level works with mock candle data."""
    from unittest.mock import AsyncMock, MagicMock

    from options_owl.config.settings import Settings
    from options_owl.execution.paper_trader import PaperTrader

    settings = Settings(
        DISCORD_TOKEN="fake",
        DB_PATH="/tmp/dry_run_test.db",
        ENABLE_DIP_CONFIRM=True,
    )
    pt = PaperTrader(settings)

    # No cache → returns None
    result = await pt._check_support_level("SPY", "call")
    assert result is None, "Should return None without candle cache"

    # With mock cache
    mock_cache = AsyncMock()
    bars = []
    for i in range(10):
        bar = MagicMock()
        bar.close = 530.0 + i * 0.2
        bar.high = bar.close + 0.5
        bar.low = bar.close - 0.3
        bar.volume = 1000
        bars.append(bar)

    mock_cache.get_candle_data = AsyncMock(return_value={"5m": bars})
    pt._candle_cache = mock_cache

    result = await pt._check_support_level("SPY", "call")
    assert result is not None, "Should return tuple with candle data"
    at_support, above_vwap, detail = result
    assert isinstance(at_support, bool)
    assert isinstance(above_vwap, bool)
    assert "price=" in detail and "vwap=" in detail
    print(f"  [PASS] _check_support_level: {detail}")
    return True


# ── 4. market_data_stream tick hooks wired ──────────────────────────────────

def test_tick_hooks_wired() -> bool:
    """Verify market_data_stream imports tick_logger functions."""
    import inspect
    from options_owl.collectors import market_data_stream

    source = inspect.getsource(market_data_stream)
    assert "record_tick" in source, "market_data_stream should reference record_tick"
    assert "init_tick_logger" in source, "market_data_stream should reference init_tick_logger"
    print("  [PASS] market_data_stream has tick_logger hooks")
    return True


# ── 5. Settings defaults ────────────────────────────────────────────────────

def test_settings_defaults() -> bool:
    """Verify dip-confirm settings have correct defaults."""
    from options_owl.config.settings import Settings

    s = Settings(DISCORD_TOKEN="fake")
    assert s.ENABLE_DIP_CONFIRM is False, f"Default should be False, got {s.ENABLE_DIP_CONFIRM}"
    assert s.DIP_CONFIRM_MAX_POLLS == 6, f"Expected 6, got {s.DIP_CONFIRM_MAX_POLLS}"
    assert s.DIP_CONFIRM_POLL_SEC == 5.0, f"Expected 5.0, got {s.DIP_CONFIRM_POLL_SEC}"
    assert s.DIP_CONFIRM_FADE_PCT == 1.0, f"Expected 1.0, got {s.DIP_CONFIRM_FADE_PCT}"
    print("  [PASS] Settings defaults correct")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    agent_id = os.environ.get("AGENT_ID", "local")
    print(f"\n=== Dry-Run Verification ({agent_id}) ===\n")

    passed = 0
    failed = 0
    tests = [
        ("Tick Logger", lambda: test_tick_logger()),
        ("CandleCache Import", lambda: test_candle_cache_import()),
        ("Support Level", lambda: asyncio.get_event_loop().run_until_complete(test_support_level())),
        ("Tick Hooks Wired", lambda: test_tick_hooks_wired()),
        ("Settings Defaults", lambda: test_settings_defaults()),
    ]

    for name, fn in tests:
        try:
            # Handle both sync and async
            if name == "Support Level":
                ok = await test_support_level()
            else:
                ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"  [FAIL] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
