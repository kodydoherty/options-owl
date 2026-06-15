"""Darkpool forward-collector (shadow). UW darkpool is recent-only (no history), so to ever build
B3 (dark-pool prints as support/resistance) we must collect it going forward. Polls
/api/darkpool/{ticker} for the traded universe every few minutes during market hours and stores
NEW prints (deduped) into journal/darkpool/darkpool.db. Observe-only — zero impact on trading.

Run: python scripts/uw_darkpool_shadow.py   (key from .env / env UNUSUAL_WHALES_API_KEY)
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parent.parent
KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")
if not KEY and (ROOT / ".env").exists():
    KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
                if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
ET = ZoneInfo("America/New_York")
BASE = "https://api.unusualwhales.com/api/darkpool"
DB = ROOT / "journal" / "darkpool" / "darkpool.db"
# traded universe (flow whitelist + indices + new tickers)
TICKERS = ["SPY", "QQQ", "IWM", "META", "AMZN", "AAPL", "TSLA", "MU", "AMD", "NVDA",
           "ORCL", "INTC", "ARM", "GOOG", "LRCX", "MSFT", "GOOGL"]
POLL_S = 300


def _init():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS darkpool_prints (
        ticker TEXT, executed_at TEXT, size REAL, price REAL, premium REAL,
        nbbo_bid REAL, nbbo_ask REAL, sale_cond_codes TEXT, captured_at TEXT,
        PRIMARY KEY (ticker, executed_at, size, price))""")
    c.execute("PRAGMA journal_mode=WAL")
    c.commit()
    return c


def _market_open() -> bool:
    n = datetime.now(ET)
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return 9 * 60 + 30 <= mins <= 16 * 60


async def poll_once(con) -> None:
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    now = datetime.now(ET).isoformat()
    async with httpx.AsyncClient(timeout=20) as client:
        for tk in TICKERS:
            try:
                r = await client.get(f"{BASE}/{tk}", headers=hdr, params={"limit": 500})
                if r.status_code != 200:
                    continue
                for p in r.json().get("data", []) or []:
                    try:
                        con.execute(
                            "INSERT OR IGNORE INTO darkpool_prints VALUES (?,?,?,?,?,?,?,?,?)",
                            (tk, p.get("executed_at"), float(p.get("size") or 0),
                             float(p.get("price") or 0), float(p.get("premium") or 0),
                             float(p.get("nbbo_bid") or 0), float(p.get("nbbo_ask") or 0),
                             str(p.get("sale_cond_codes") or ""), now))
                    except Exception:
                        pass
            except Exception:
                pass
            await asyncio.sleep(0.4)
    con.commit()


async def main():
    con = _init()
    print(f"DARKPOOL_SHADOW: started, {len(TICKERS)} tickers, db={DB}", flush=True)
    while True:
        if _market_open():
            n0 = con.total_changes
            await poll_once(con)
            total = con.execute("SELECT COUNT(*) FROM darkpool_prints").fetchone()[0]
            print(f"DARKPOOL_SHADOW: +{con.total_changes - n0} new prints "
                  f"(total {total}) @ {datetime.now(ET).strftime('%H:%M')}", flush=True)
            await asyncio.sleep(POLL_S)
        else:
            await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(main())
