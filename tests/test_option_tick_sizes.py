"""Tests for bid_size/ask_size plumbing through the harvester option_ticks path.

Verifies the harvester -> DB INSERT -> read (OptionSnapshot) chain carries
quote sizes end-to-end. No live PostgreSQL connection is required; the pool
and connection are mocked so we inspect the SQL / parameter construction.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from options_owl.db import postgres as pg
from options_owl.sourcing.data import harvester_options as ho


class _FakeConn:
    def __init__(self):
        self.execute_calls: list[tuple] = []
        self.executemany_calls: list[tuple] = []

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))

    async def executemany(self, sql, rows):
        self.executemany_calls.append((sql, rows))


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


@pytest.fixture
def fake_pool(monkeypatch):
    conn = _FakeConn()
    pool = _FakePool(conn)
    monkeypatch.setattr(pg, "_pool", pool)
    # is_connected() checks pg._pool is not None
    return conn


def _schema_has_size_columns() -> bool:
    return (
        "bid_size INTEGER" in pg.SCHEMA_SQL
        and "ask_size INTEGER" in pg.SCHEMA_SQL
    )


def test_schema_defines_size_columns_and_migrations():
    """CREATE TABLE includes bid_size/ask_size and idempotent ALTER migrations."""
    assert _schema_has_size_columns()
    assert "ALTER TABLE option_ticks ADD COLUMN IF NOT EXISTS bid_size INTEGER" in pg.SCHEMA_SQL
    assert "ALTER TABLE option_ticks ADD COLUMN IF NOT EXISTS ask_size INTEGER" in pg.SCHEMA_SQL


def test_write_option_ticks_batch_includes_sizes(fake_pool):
    """Batch insert SQL + parameter tuple carry bid_size/ask_size."""
    ticks = [
        {
            "ticker": "SPY", "option_type": "call", "strike": 550.0,
            "expiry_date": "2026-06-10", "bid": 2.45, "ask": 2.55,
            "bid_size": 120, "ask_size": 80, "mid": 2.50, "last": 2.50,
            "volume": 1000, "open_interest": 500, "iv": 0.30,
            "delta": 0.50, "gamma": 0.02, "theta": -0.05, "vega": 0.10,
            "underlying_price": 550.0,
        }
    ]
    asyncio.run(pg.write_option_ticks_batch(ticks))

    assert len(fake_pool.executemany_calls) == 1
    sql, rows = fake_pool.executemany_calls[0]
    assert "bid_size" in sql and "ask_size" in sql
    # 18 placeholders -> $18 present, $19 absent
    assert "$18" in sql and "$19" not in sql
    row = rows[0]
    assert len(row) == 18
    # column order: ..., bid($5), ask($6), bid_size($7), ask_size($8), mid($9)...
    assert row[6] == 120  # bid_size
    assert row[7] == 80   # ask_size


def test_write_option_tick_single_includes_sizes(fake_pool):
    """Single-row insert SQL + params carry bid_size/ask_size."""
    asyncio.run(
        pg.write_option_tick(
            "SPY", "call", 550.0, "2026-06-10",
            bid=2.45, ask=2.55, mid=2.50, last=2.50,
            volume=1000, open_interest=500, iv=0.30,
            delta=0.50, gamma=0.02, theta=-0.05, vega=0.10,
            underlying_price=550.0, bid_size=120, ask_size=80,
        )
    )
    assert len(fake_pool.execute_calls) == 1
    sql, args = fake_pool.execute_calls[0]
    assert "bid_size" in sql and "ask_size" in sql
    assert "$18" in sql and "$19" not in sql
    # args order matches placeholders: bid_size is 7th, ask_size 8th
    assert args[6] == 120
    assert args[7] == 80


def test_option_snapshot_roundtrips_sizes(monkeypatch):
    """fetch_atm_option_snapshot populates bid_size/ask_size from the DB row."""
    db_row = {
        "strike": 550.0, "option_type": "call", "expiry_date": "2026-06-10",
        "underlying_price": 550.0, "bid": 2.45, "ask": 2.55,
        "bid_size": 120, "ask_size": 80, "mid": 2.50, "volume": 1000,
        "iv": 0.30, "delta": 0.50, "theta": -0.05, "vega": 0.10,
        "captured_at": "2026-06-10T13:00:00Z",
    }

    monkeypatch.setattr(pg, "is_connected", lambda: True)

    async def _fake_fetch(*_args, **_kwargs):
        return [db_row]

    monkeypatch.setattr(pg, "fetch", _fake_fetch)

    snap = asyncio.run(ho.fetch_atm_option_snapshot("SPY", "CALL"))
    assert snap is not None
    assert snap.bid_size == 120
    assert snap.ask_size == 80


def test_option_snapshot_null_sizes_default_zero(monkeypatch):
    """NULL bid_size/ask_size (old rows) coerce to 0, not crash."""
    db_row = {
        "strike": 550.0, "option_type": "call", "expiry_date": "2026-06-10",
        "underlying_price": 550.0, "bid": 2.45, "ask": 2.55,
        "bid_size": None, "ask_size": None, "mid": 2.50, "volume": 1000,
        "iv": 0.30, "delta": 0.50, "theta": -0.05, "vega": 0.10,
        "captured_at": "2026-06-10T13:00:00Z",
    }
    monkeypatch.setattr(pg, "is_connected", lambda: True)

    async def _fake_fetch(*_args, **_kwargs):
        return [db_row]

    monkeypatch.setattr(pg, "fetch", _fake_fetch)

    snap = asyncio.run(ho.fetch_atm_option_snapshot("SPY", "CALL"))
    assert snap is not None
    assert snap.bid_size == 0
    assert snap.ask_size == 0
