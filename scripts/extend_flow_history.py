"""Extend our flow-alert history back ~12 months (UW's retention floor for our plan) so the
long-dated whale-following test spans multiple regimes with FULLY-EXPIRED (complete) holds.

Paginates the global flow-alerts endpoint via `older_than` from now back to STOP_DATE, storing
every >$250k whale alert into journal/uw_historical.db `flow_alerts` (INSERT OR IGNORE by id →
resumable). Then download_longdated_flow.py picks up the long-dated (>30 DTE) subset automatically.

    python scripts/extend_flow_history.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "uw_historical.db"
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
HDR = {"Authorization": f"Bearer {KEY}", "Accept": "application/json", "UW-CLIENT-API-ID": "100001",
       "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120 Safari/537.36")}
BASE = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
STOP_DATE = "2025-07-01"     # UW retention floor for our plan (~12mo); older → empty/403
MIN_PREMIUM = 250_000        # whale filter


def _insert(conn, d):
    n = 0
    for row in d:
        if not isinstance(row, dict):
            continue
        aid = row.get("id") or (str(row.get("option_chain", "")) + "_" + str(row.get("created_at", "")))
        conn.execute(
            """INSERT OR IGNORE INTO flow_alerts
            (id, ticker, created_at, type, strike, expiry, price, volume, open_interest,
             total_premium, underlying_price, trade_count, iv_start, iv_end, volume_oi_ratio,
             has_sweep, has_floor, has_multileg, all_opening_trades, alert_rule,
             total_bid_side_prem, total_ask_side_prem)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (aid, row.get("ticker"), row.get("created_at", ""), row.get("type"),
             float(row.get("strike") or 0), row.get("expiry"), float(row.get("price") or 0),
             int(row.get("volume") or 0), int(row.get("open_interest") or 0),
             float(row.get("total_premium") or 0), float(row.get("underlying_price") or 0),
             int(row.get("trade_count") or 0), float(row.get("iv_start") or 0),
             float(row.get("iv_end") or 0), float(row.get("volume_oi_ratio") or 0),
             int(row.get("has_sweep", False)), int(row.get("has_floor", False)),
             int(row.get("has_multileg", False)), int(row.get("all_opening_trades", False)),
             row.get("alert_rule"), float(row.get("total_bid_side_prem") or 0),
             float(row.get("total_ask_side_prem") or 0)),
        )
        n += 1
    return n


def main():
    if not KEY:
        print("No UNUSUAL_WHALES_API_KEY"); return
    conn = sqlite3.connect(str(DB))
    sess = requests.Session()
    older = None
    total = 0
    pages = 0
    oldest = "now"
    while pages < 3000:
        params = {"limit": 200, "min_premium": MIN_PREMIUM}
        if older:
            params["older_than"] = older
        try:
            r = sess.get(BASE, headers=HDR, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  net err {exc}; retrying", flush=True); time.sleep(2); continue
        if r.status_code == 429:
            time.sleep(2); continue
        if r.status_code != 200:
            print(f"stopped: HTTP {r.status_code} {r.text[:100]}", flush=True); break
        j = r.json()
        d = j.get("data") if isinstance(j, dict) else j
        if not d:
            print("stopped: empty page (retention floor)", flush=True); break
        total += _insert(conn, d)
        pages += 1
        oldest = min(x["created_at"] for x in d if isinstance(x, dict))
        older = oldest
        if pages % 25 == 0:
            conn.commit()
            print(f"  page {pages}: {total} stored, oldest {oldest[:10]}", flush=True)
        if oldest[:10] < STOP_DATE:
            print(f"reached STOP_DATE {STOP_DATE}", flush=True); break
        time.sleep(0.12)
    conn.commit()
    conn.close()
    print(f"FLOW_HISTORY_DONE {total} alerts stored, oldest {oldest[:10]}", flush=True)


if __name__ == "__main__":
    main()
