#!/usr/bin/env python3
"""Broker-side stop-loss LIVE verification — read-only.

Runs at/after market open to confirm the broker-stop feature is working correctly on the LIVE bots:
for every open live position, a resting Webull STOP_LOSS should have been placed (and ratcheted). This
script does NOT touch Webull or any order — it reads the bots' DBs + today's persisted logs only, so it
is safe to run repeatedly (e.g. every 2 min through the open) with zero risk to trading.

Verdict:
  * PASS  — every open live position has a tracked/rested broker stop (or there are no positions yet).
  * WARN  — an open position has no broker stop yet (could be a transient placement lag).
  * FAIL  — Webull REJECTED our STOP_LOSS payload or the manager gave up (poll-only fallback engaged) —
            i.e. the venue is not accepting the order (schema / permission problem to fix before trusting it).

Exit code: 0 = PASS, 1 = WARN, 2 = FAIL (so a cron/babysitter can escalate on non-zero).

Usage:
  python3 scripts/broker_stop_check.py                 # default: kody,dennis (the live bots)
  python3 scripts/broker_stop_check.py --bots kody,dennis,yank
  python3 scripts/broker_stop_check.py --base /root/options-owl
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
from datetime import date

LIVE_DEFAULT = "kody,dennis"

# Log markers emitted by broker_stop.py / webull_executor.place_stop_loss
RE_TRACKED = re.compile(r"BROKER STOP (?:tracked|RESTED): .*?#?(\d+)?.*?stop=\$?([\d.]+)")
RE_REJECT = re.compile(r"BROKER STOP (?:REJECTED|giving up)")
RE_RATCHET = re.compile(r"BROKER STOP ratchet: trade#(\d+)")
RE_RECONCILE = re.compile(r"BROKER STOP reconcile: re-adopted (\d+) .*cancelled (\d+)")
RE_TRACKED_TID = re.compile(r"BROKER STOP tracked: trade#(\d+)")


def _open_positions(db_path: str) -> list[dict]:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id,ticker,option_type,contracts,webull_order_id FROM paper_trades "
            "WHERE status='open'"
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        return [{"_error": str(exc)}]


def _todays_log_text(base: str, bot: str) -> str:
    day = date.today().strftime("%Y-%m-%d")
    path = os.path.join(base, "journal", f"owlet-{bot}", "logs", f"options_owl_{day}.log")
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def check_bot(base: str, bot: str) -> tuple[str, list[str]]:
    lines: list[str] = []
    positions = _open_positions(os.path.join(base, "journal", f"owlet-{bot}", "raw_messages.db"))
    if positions and positions[0].get("_error"):
        return "FAIL", [f"[{bot}] DB read error: {positions[0]['_error']}"]

    log = _todays_log_text(base, bot)
    live_positions = [p for p in positions if p.get("webull_order_id")]
    tracked_tids = {int(m) for m in RE_TRACKED_TID.findall(log)}
    ratcheted_tids = {int(m) for m in RE_RATCHET.findall(log)}
    n_reject = len(RE_REJECT.findall(log))
    reconcile = RE_RECONCILE.findall(log)

    verdict = "PASS"
    lines.append(
        f"[{bot}] open={len(positions)} live(webull)={len(live_positions)} "
        f"stops_tracked={len(tracked_tids)} ratchets={len(ratcheted_tids)} "
        f"rejects/giveups={n_reject}"
    )
    if reconcile:
        adopted, cancelled = reconcile[-1]
        lines.append(f"[{bot}] startup reconcile: re-adopted {adopted}, cancelled {cancelled} orphan(s)")

    # FAIL: venue rejected our order or manager gave up → poll-only fallback engaged
    if n_reject > 0:
        verdict = "FAIL"
        lines.append(f"[{bot}] ⚠️ {n_reject} broker-stop REJECT/GIVE-UP event(s) — Webull not accepting the "
                     f"STOP_LOSS payload; feature is in poll-only fallback. Investigate the order schema.")

    # WARN: an open live position has no tracked stop yet
    missing = [p for p in live_positions if p["id"] not in tracked_tids]
    if missing and verdict != "FAIL":
        verdict = "WARN"
    for p in missing:
        lines.append(f"[{bot}] ⏳ #{p['id']} {p['ticker']} {p['option_type']} x{p['contracts']} "
                     f"— no broker stop tracked yet")

    if verdict == "PASS" and live_positions:
        lines.append(f"[{bot}] ✅ all {len(live_positions)} live position(s) have a resting broker stop")
    elif verdict == "PASS":
        lines.append(f"[{bot}] ✅ no live positions yet — nothing to protect")
    return verdict, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default=LIVE_DEFAULT, help="comma-separated bot names")
    ap.add_argument("--base", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repo root (default: parent of scripts/)")
    args = ap.parse_args()

    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    worst = "PASS"
    print(f"=== BROKER STOP LIVE CHECK {date.today()} ===")
    for bot in [b.strip() for b in args.bots.split(",") if b.strip()]:
        verdict, lines = check_bot(args.base, bot)
        for ln in lines:
            print(ln)
        if rank[verdict] > rank[worst]:
            worst = verdict
    print(f"=== VERDICT: {worst} ===")
    return rank[worst]


if __name__ == "__main__":
    raise SystemExit(main())
