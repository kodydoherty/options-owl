"""UW flow WS → PERSISTING collector (observe-only, no trading).

Subscribes to the Unusual Whales flow-alerts WebSocket and stores EVERY alert (not just qualifying)
to journal/uw_flow/uw_flow_alerts.db so we have the full UW flow history in one queryable DB — for
B5 (alert-quality), flow-pattern research, and live-vs-backtest reconciliation. Also logs the
qualifying whale-sweep signals it WOULD emit.

Live WS has no type/strike/expiry fields — they're in the OCC `option_chain` symbol; we parse them
(same fix as the production collector). Run: python scripts/uw_flow_shadow.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")
if not KEY and (ROOT / ".env").exists():
    KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
                if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
URI = f"wss://api.unusualwhales.com/socket?token={KEY}"
DB = ROOT / "journal" / "uw_flow" / "uw_flow_alerts.db"
PUT_WL = {"META", "AMZN", "AAPL", "TSLA", "MU", "SPY"}
CALL_WL = {"META", "SPY", "AMZN", "TSLA", "AMD", "ORCL", "INTC", "ARM", "GOOG", "LRCX"}
MIN_PREMIUM = 250_000
MIN_ASK_FRAC = 0.60


def _occ(chain: str):
    """Parse OCC option_chain → (type, strike, expiry). Reuses the production parser."""
    try:
        from options_owl.collectors.flow_collector import parse_occ_ticker
        p = parse_occ_ticker(chain if chain.startswith("O:") else f"O:{chain}") if chain else None
        return (p["type"], p["strike"], p["expiry"]) if p else (None, 0.0, "")
    except Exception:
        return (None, 0.0, "")


def _init():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS uw_flow_alerts (
        id TEXT PRIMARY KEY, ticker TEXT, type TEXT, strike REAL, expiry TEXT,
        total_premium REAL, ask_frac REAL, has_sweep INT, rule_name TEXT,
        all_opening_trades INT, has_singleleg INT, has_multileg INT,
        underlying_price REAL, volume_oi_ratio REAL, qualifying INT, created_at TEXT, captured_at TEXT)""")
    c.execute("PRAGMA journal_mode=WAL")
    c.commit()
    return c


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _store(con, a: dict) -> bool:
    """Store one alert; return True if it qualifies as a whale-sweep signal."""
    ticker = str(a.get("ticker", "")).upper()
    otype, strike, expiry = _occ(str(a.get("option_chain", "")))
    if not otype:
        otype = str(a.get("type", "")).lower() or None
    total = _f(a.get("total_premium"))
    askf = _f(a.get("total_ask_side_prem")) / total if total > 0 else 0.0
    wl = PUT_WL if otype == "put" else CALL_WL
    qualifying = int(otype in ("put", "call") and ticker in wl and total >= MIN_PREMIUM
                     and askf >= MIN_ASK_FRAC and bool(a.get("has_sweep")))
    now = datetime.now(timezone.utc).isoformat()
    con.execute("INSERT OR IGNORE INTO uw_flow_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(a.get("id", now)), ticker, otype, strike, expiry, total, round(askf, 3),
                 int(bool(a.get("has_sweep"))), str(a.get("rule_name", "")),
                 int(bool(a.get("all_opening_trades"))), int(bool(a.get("has_singleleg"))),
                 int(bool(a.get("has_multileg"))), _f(a.get("underlying_price")),
                 _f(a.get("volume_oi_ratio")), qualifying, str(a.get("executed_at", "")), now))
    return bool(qualifying)


async def main():
    con = _init()
    print(f"UW_FLOW_SHADOW: persisting collector started → {DB}", flush=True)
    attempt = 0
    while True:
        try:
            async with websockets.connect(URI, ping_interval=20) as ws:
                await ws.send(json.dumps({"channel": "flow-alerts", "msg_type": "join"}))
                print("UW_FLOW_SHADOW: connected + joined flow-alerts", flush=True)
                attempt = 0
                n = q = 0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    payloads = msg if isinstance(msg, list) else [msg]
                    for p in payloads:
                        if not isinstance(p, dict) or "ticker" not in p:
                            continue
                        n += 1
                        if _store(con, p):
                            q += 1
                            print(f"UW_FLOW_SHADOW: QUALIFYING {p.get('ticker')} "
                                  f"{_occ(str(p.get('option_chain','')))[0]} "
                                  f"${_f(p.get('total_premium'))/1e3:.0f}k sweep", flush=True)
                    if n % 200 == 0:
                        con.commit()
                        print(f"UW_FLOW_SHADOW: {n} alerts stored, {q} qualifying", flush=True)
        except Exception as exc:
            con.commit()
            attempt += 1
            delay = min(5 * attempt, 60)
            print(f"UW_FLOW_SHADOW: WS dropped ({type(exc).__name__}); reconnect in {delay}s", flush=True)
            await asyncio.sleep(delay)


if __name__ == "__main__":
    asyncio.run(main())
