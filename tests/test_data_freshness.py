"""Tests for the runtime market-data freshness guard.

The guard blocks NEW entries when the harvester feed is stale during market hours. The critical case
is frozen-but-fresh-timestamp data (Juneteenth): a recent bar_time with zero volume must read STALE.
And it must FAIL OPEN (allow) on any error or outside market hours, so the guard can't itself halt
trading on a transient hiccup.
"""

import datetime

import pytest

from options_owl.risk import data_freshness
from options_owl.risk.data_freshness import check_market_data_fresh

_UTC = datetime.timezone.utc


def _bar(age_sec: float, volume: int) -> dict:
    return {
        "bar_time": datetime.datetime.now(tz=_UTC) - datetime.timedelta(seconds=age_sec),
        "close": 740.0,
        "volume": volume,
    }


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr(data_freshness, "is_market_open", lambda now=None: True)


def _patch_candles(monkeypatch, rows=None, raises=None):
    async def _fake(ticker, timeframe, limit=3):
        if raises is not None:
            raise raises
        return rows

    monkeypatch.setattr(data_freshness.postgres, "read_stock_candles", _fake)


@pytest.mark.asyncio
async def test_fresh_when_recent_bar_has_volume(market_open, monkeypatch):
    _patch_candles(monkeypatch, rows=[_bar(120, 50000), _bar(60, 60000), _bar(5, 70000)])
    fresh, reason = await check_market_data_fresh(max_age_sec=180)
    assert fresh is True, reason


@pytest.mark.asyncio
async def test_stale_when_newest_bar_too_old(market_open, monkeypatch):
    _patch_candles(monkeypatch, rows=[_bar(600, 50000), _bar(540, 40000), _bar(480, 30000)])
    fresh, reason = await check_market_data_fresh(max_age_sec=180)
    assert fresh is False
    assert "stalled" in reason


@pytest.mark.asyncio
async def test_stale_when_frozen_zero_volume(market_open, monkeypatch):
    # Juneteenth case: fresh timestamps, but every bar has 0 volume (frozen price).
    _patch_candles(monkeypatch, rows=[_bar(120, 0), _bar(60, 0), _bar(5, 0)])
    fresh, reason = await check_market_data_fresh(max_age_sec=180)
    assert fresh is False
    assert "0 volume" in reason or "frozen" in reason


@pytest.mark.asyncio
async def test_no_rows_is_stale(market_open, monkeypatch):
    _patch_candles(monkeypatch, rows=[])
    fresh, reason = await check_market_data_fresh()
    assert fresh is False
    assert "feed down" in reason


@pytest.mark.asyncio
async def test_market_closed_always_fresh(monkeypatch):
    # Outside RTH / holiday — frozen data is expected, do NOT halt.
    monkeypatch.setattr(data_freshness, "is_market_open", lambda now=None: False)
    _patch_candles(monkeypatch, rows=[_bar(99999, 0)])  # very stale, but market closed
    fresh, reason = await check_market_data_fresh()
    assert fresh is True
    assert "market closed" in reason


@pytest.mark.asyncio
async def test_db_error_fails_open(market_open, monkeypatch):
    # A transient PG error must NOT halt trading — the guard fails OPEN.
    _patch_candles(monkeypatch, raises=RuntimeError("pg down"))
    fresh, reason = await check_market_data_fresh()
    assert fresh is True
    assert "failing open" in reason
