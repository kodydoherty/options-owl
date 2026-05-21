"""Lightweight 1-second option tick logger for backtesting dip-confirm.

Captures every option premium update (trade/quote) from the WebSocket stream
into a SQLite DB. Only records ticks for actively subscribed option contracts.

Data is written in batches every 5 seconds to avoid I/O overhead.
DB is stored at journal/option_ticks.db — kept per-agent, survives rebuilds.

Schema:
    option_ticks(
        contract_ticker TEXT,    -- e.g. "O:SPY260518P00740000"
        ts REAL,                 -- unix timestamp (sub-second precision)
        bid REAL,
        ask REAL,
        mid REAL,
        underlying_price REAL,
        source TEXT              -- 'T' (trade) or 'Q' (quote)
    )
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

from loguru import logger

# Batch buffer — accumulates ticks, flushed every N seconds
_FLUSH_INTERVAL = 5.0  # seconds
_MAX_BUFFER = 5000  # flush if buffer hits this size

# Module-level state (one per process)
_buffer: deque[tuple] = deque(maxlen=50000)
_lock = threading.Lock()
_db_path: str | None = None
_initialized = False
_last_flush = 0.0


def init_tick_logger(journal_dir: str = "journal") -> None:
    """Initialize the tick logger DB. Call once at startup."""
    global _db_path, _initialized, _last_flush

    _db_path = str(Path(journal_dir) / "option_ticks.db")
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)

    conn = sqlite3.connect(_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS option_ticks (
            contract_ticker TEXT NOT NULL,
            ts REAL NOT NULL,
            bid REAL,
            ask REAL,
            mid REAL,
            underlying_price REAL,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticks_contract_ts
        ON option_ticks(contract_ticker, ts)
    """)
    conn.commit()
    conn.close()

    _initialized = True
    _last_flush = time.time()
    logger.info(f"[TickLogger] Initialized at {_db_path}")


def record_tick(
    contract_ticker: str,
    ts: float,
    bid: float | None = None,
    ask: float | None = None,
    mid: float | None = None,
    underlying_price: float | None = None,
    source: str = "Q",
) -> None:
    """Record a single tick. Non-blocking — appends to buffer."""
    if not _initialized:
        return

    with _lock:
        _buffer.append((
            contract_ticker, ts,
            bid, ask, mid,
            underlying_price, source,
        ))

    # Check if we should flush (non-blocking check)
    if len(_buffer) >= _MAX_BUFFER or time.time() - _last_flush >= _FLUSH_INTERVAL:
        _flush_buffer()


def _flush_buffer() -> None:
    """Write buffered ticks to SQLite. Thread-safe."""
    global _last_flush

    if not _db_path or not _buffer:
        return

    with _lock:
        batch = list(_buffer)
        _buffer.clear()
    _last_flush = time.time()

    if not batch:
        return

    try:
        conn = sqlite3.connect(_db_path)
        conn.executemany(
            "INSERT INTO option_ticks "
            "(contract_ticker, ts, bid, ask, mid, underlying_price, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"[TickLogger] Flush failed ({len(batch)} ticks): {exc}")


def flush_remaining() -> None:
    """Flush any remaining ticks (call at shutdown)."""
    _flush_buffer()
