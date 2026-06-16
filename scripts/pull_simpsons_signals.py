"""Pull + parse the Simpsons 'distribution' signals (Discord channel) for backtesting.

Paginates the full channel history via the bot token, parses each Homer 'ENTER NOW' embed into
(timestamp_utc, ticker, direction, score, stage), writes journal/simpsons/simpsons_signals.csv.
The Discord message timestamp (UTC) is the authoritative signal time. Read-only on Discord.
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TOKEN = next((l.split("=", 1)[1].strip().strip('"') for l in (ROOT / ".env").read_text().splitlines()
              if l.startswith("DISCORD_TOKEN=")), "")
CHANNELS = ["1495293929086783639", "1514355053878575155"]  # #simpsons-dev (older history) + amd-only
OUT = ROOT / "journal" / "simpsons" / "simpsons_signals.csv"
H = {"Authorization": f"Bot {TOKEN}"}

TICK_DIR = re.compile(r'"([A-Z]{1,6})"\s*[—-]\s*"(PUTS|CALLS|PUT|CALL)"')
SCORE = re.compile(r'SCORE:\s*\*\*\s*(\d+)\s*pts')


def parse_embed(desc: str):
    """Return (ticker, direction, score) for an ENTER-NOW distribution signal, else None."""
    if not desc or "ENTER NOW" not in desc.upper():
        return None
    td = TICK_DIR.search(desc)
    if not td:
        return None
    ticker = td.group(1).upper()
    direction = "put" if td.group(2).upper().startswith("PUT") else "call"
    sc = SCORE.search(desc)
    score = int(sc.group(1)) if sc else 0
    return ticker, direction, score


def main():
    if not TOKEN:
        print("no DISCORD_TOKEN"); return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for CH in CHANNELS:
      before, pages = None, 0
      while pages < 200:  # up to ~20k messages
        params = {"limit": 100}
        if before:
            params["before"] = before
        r = requests.get(f"https://discord.com/api/v10/channels/{CH}/messages",
                         headers=H, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5); continue
        if r.status_code != 200:
            print("stop:", r.status_code, r.json().get("message")); break
        msgs = r.json()
        if not msgs:
            break
        for x in msgs:
            for e in x.get("embeds", []):
                p = parse_embed(e.get("description", "") or "")
                if p:
                    rows.append({"ts_utc": x["timestamp"][:19], "ticker": p[0],
                                 "direction": p[1], "score": p[2],
                                 "author": x["author"]["username"].split()[0]})
        before = msgs[-1]["id"]
        pages += 1
        if pages % 5 == 0:
            print(f"  {pages} pages, {len(rows)} signals so far...", flush=True)
        time.sleep(0.3)

    # dedup across channels (amd-only is a filtered subset of #simpsons-dev)
    seen, deduped = set(), []
    for r in rows:
        k = (r["ts_utc"], r["ticker"], r["direction"])
        if k not in seen:
            seen.add(k); deduped.append(r)
    rows = deduped
    rows.sort(key=lambda r: r["ts_utc"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts_utc", "ticker", "direction", "score", "author"])
        w.writeheader(); w.writerows(rows)

    print(f"\n{len(rows)} ENTER-NOW signals → {OUT}")
    if rows:
        print(f"date range: {rows[0]['ts_utc'][:10]} → {rows[-1]['ts_utc'][:10]}")
        from collections import Counter
        bt = Counter(f"{r['ticker']}/{r['direction']}" for r in rows)
        print("by ticker/dir:", dict(bt.most_common(15)))
        print("by ticker:", dict(Counter(r["ticker"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
