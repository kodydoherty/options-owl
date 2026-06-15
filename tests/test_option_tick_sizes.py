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


def test_schema_defines_charm_vanna_columns_and_migrations():
    """option_ticks gains charm/vanna (CREATE TABLE + idempotent ALTERs)."""
    assert "charm REAL" in pg.SCHEMA_SQL
    assert "vanna REAL" in pg.SCHEMA_SQL
    assert "ALTER TABLE option_ticks ADD COLUMN IF NOT EXISTS charm REAL" in pg.SCHEMA_SQL
    assert "ALTER TABLE option_ticks ADD COLUMN IF NOT EXISTS vanna REAL" in pg.SCHEMA_SQL


def test_schema_defines_gex_ticks_table_and_index():
    """gex_ticks table exists per spec section 5.4 with all aggregate columns."""
    assert "CREATE TABLE IF NOT EXISTS gex_ticks" in pg.SCHEMA_SQL
    assert "CREATE INDEX IF NOT EXISTS idx_gex_ticks_time ON gex_ticks(ticker, captured_at)" in pg.SCHEMA_SQL
    for col in (
        "net_gamma", "call_gamma", "put_gamma", "net_charm",
        "net_vanna", "total_oi", "spot", "n_contracts",
    ):
        assert col in pg.SCHEMA_SQL, f"gex_ticks missing column {col}"


def _sample_tick() -> dict:
    return {
        "ticker": "SPY", "option_type": "call", "strike": 550.0,
        "expiry_date": "2026-06-10", "bid": 2.45, "ask": 2.55,
        "bid_size": 120, "ask_size": 80, "mid": 2.50, "last": 2.50,
        "volume": 1000, "open_interest": 500, "iv": 0.30,
        "delta": 0.50, "gamma": 0.02, "theta": -0.05, "vega": 0.10,
        "charm": -0.0012, "vanna": 0.0034,
        "underlying_price": 550.0,
    }


def test_write_option_ticks_batch_includes_sizes(fake_pool):
    """Batch insert SQL + parameter tuple carry bid_size/ask_size."""
    asyncio.run(pg.write_option_ticks_batch([_sample_tick()]))

    assert len(fake_pool.executemany_calls) == 1
    sql, rows = fake_pool.executemany_calls[0]
    assert "bid_size" in sql and "ask_size" in sql
    # 20 placeholders -> $20 present, $21 absent
    assert "$20" in sql and "$21" not in sql
    row = rows[0]
    assert len(row) == 20
    # column order: ..., bid($5), ask($6), bid_size($7), ask_size($8), mid($9)...
    assert row[6] == 120  # bid_size
    assert row[7] == 80   # ask_size


def test_write_option_ticks_batch_includes_charm_vanna(fake_pool):
    """Batch insert SQL + params carry charm/vanna ($18/$19, before underlying)."""
    asyncio.run(pg.write_option_ticks_batch([_sample_tick()]))

    sql, rows = fake_pool.executemany_calls[0]
    assert "charm" in sql and "vanna" in sql
    row = rows[0]
    # ..., theta($16), vega($17), charm($18), vanna($19), underlying_price($20)
    assert row[17] == -0.0012  # charm
    assert row[18] == 0.0034   # vanna
    assert row[19] == 550.0    # underlying_price


def test_write_option_ticks_batch_missing_charm_vanna_defaults_none(fake_pool):
    """Old callers without charm/vanna keys insert NULL, not crash."""
    tick = _sample_tick()
    del tick["charm"], tick["vanna"]
    asyncio.run(pg.write_option_ticks_batch([tick]))
    _, rows = fake_pool.executemany_calls[0]
    assert rows[0][17] is None
    assert rows[0][18] is None


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
    assert "$20" in sql and "$21" not in sql
    # args order matches placeholders: bid_size is 7th, ask_size 8th
    assert args[6] == 120
    assert args[7] == 80


def test_write_option_tick_single_includes_charm_vanna(fake_pool):
    """Single-row insert carries charm/vanna in the right positions."""
    asyncio.run(
        pg.write_option_tick(
            "SPY", "call", 550.0, "2026-06-10",
            bid=2.45, ask=2.55, mid=2.50, last=2.50,
            volume=1000, open_interest=500, iv=0.30,
            delta=0.50, gamma=0.02, theta=-0.05, vega=0.10,
            underlying_price=550.0, bid_size=120, ask_size=80,
            charm=-0.0012, vanna=0.0034,
        )
    )
    sql, args = fake_pool.execute_calls[0]
    assert "charm" in sql and "vanna" in sql
    # ..., theta($16), vega($17), charm($18), vanna($19), underlying_price($20)
    assert args[17] == -0.0012
    assert args[18] == 0.0034
    assert args[19] == 550.0


def test_write_gex_ticks_batch_columns_and_params(fake_pool):
    """gex_ticks INSERT uses parameterized placeholders with the spec columns."""
    rows = [
        {
            "ticker": "SPY", "net_gamma": 150_000.0, "call_gamma": 200_000.0,
            "put_gamma": 50_000.0, "net_charm": -120.0, "net_vanna": 80.0,
            "total_oi": 1500, "spot": 550.0, "n_contracts": 42,
        }
    ]
    asyncio.run(pg.write_gex_ticks_batch(rows))

    assert len(fake_pool.executemany_calls) == 1
    sql, params = fake_pool.executemany_calls[0]
    assert "INSERT INTO gex_ticks" in sql
    for col in (
        "ticker", "net_gamma", "call_gamma", "put_gamma",
        "net_charm", "net_vanna", "total_oi", "spot", "n_contracts",
    ):
        assert col in sql
    # 9 placeholders, parameterized only (no value interpolation)
    assert "$9" in sql and "$10" not in sql
    assert "SPY" not in sql.replace("INSERT INTO gex_ticks", "")
    row = params[0]
    assert row == ("SPY", 150_000.0, 200_000.0, 50_000.0, -120.0, 80.0, 1500, 550.0, 42)


def test_write_gex_ticks_batch_empty_noop(fake_pool):
    """Empty batch writes nothing."""
    asyncio.run(pg.write_gex_ticks_batch([]))
    assert fake_pool.executemany_calls == []


def test_fetch_latest_gex_returns_latest_row(monkeypatch):
    """fetch_latest_gex queries parameterized and returns a dict."""
    captured: dict = {}

    async def _fake_fetchrow(query, *args):
        captured["query"] = query
        captured["args"] = args
        return {
            "ticker": "SPY", "net_gamma": 1.0, "call_gamma": 2.0,
            "put_gamma": 1.0, "net_charm": 0.1, "net_vanna": 0.2,
            "total_oi": 10, "spot": 550.0, "n_contracts": 4,
            "captured_at": "2026-06-10T13:00:00Z",
        }

    monkeypatch.setattr(pg, "is_connected", lambda: True)
    monkeypatch.setattr(pg, "fetchrow", _fake_fetchrow)

    row = asyncio.run(pg.fetch_latest_gex("spy"))
    assert row is not None
    assert row["net_gamma"] == 1.0
    # parameterized: ticker passed as $1 (uppercased), not interpolated
    assert "$1" in captured["query"]
    assert captured["args"] == ("SPY",)
    assert "ORDER BY captured_at DESC" in captured["query"]


def test_fetch_latest_gex_empty_table_returns_none(monkeypatch):
    """Empty gex_ticks during rollout → None (caller 0-fills features)."""
    async def _fake_fetchrow(query, *args):
        return None

    monkeypatch.setattr(pg, "is_connected", lambda: True)
    monkeypatch.setattr(pg, "fetchrow", _fake_fetchrow)
    assert asyncio.run(pg.fetch_latest_gex("SPY")) is None


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
