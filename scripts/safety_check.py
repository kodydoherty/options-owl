"""Out-of-process safety check: does our RECORD match the BROKER, and can we still exit?

WHY THIS EXISTS SEPARATELY FROM THE BOT'S OWN MONITORS
------------------------------------------------------
On 2026-08-13 a phantom quantity (recorded 78 vs 71 held) blocked an exit for 35 minutes
and turned a +$1,313 position into a -$1,280 loss. Three layers of monitoring were live
and all three were silent:

  * in-process alerts were gated on `and discord_client:` and no client was configured,
    so 1,093 sell failures produced zero alerts;
  * the stuck-exit alert fired at 5/10/20 and then never again;
  * the external babysitter had been failing on a missing exec bit for 2,537 of its
    2,931 cron invocations (87%).

Those are now fixed, but the first two live INSIDE the bot process. A watchdog that shares
its subject's fate cannot catch a wedged event loop, a crash-loop, or a hung monitor. This
runs OUTSIDE, on cron, and deliberately depends on as little as possible: no Claude, no
API key, no Discord. It reads sqlite, reads the broker, compares, and shouts.

DESIGN RULES (learned from the monitors that failed)
----------------------------------------------------
1. A check that cannot run must be LOUD. Every exit path writes the status file, so
   "no heartbeat" is itself detectable. Silence is never treated as health.
2. Prove it ran. The status file carries a UTC timestamp; staleness is a failure.
3. Never auto-heal here. This process only observes and reports. Healing lives in the bot
   where the position lock is held -- two writers racing on the same record is worse than
   the bug being watched.

EXIT CODES
  0 = all checks passed
  1 = a breach was found (details in the status file and on stdout)
  2 = the check itself could not run (treat as a breach: we are flying blind)

Usage (droplet):
  docker exec owlet-kody python /app/scripts/safety_check.py --bot kody
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUS_PATH = "/app/journal/safety_check.status"
STUCK_SELL_THRESHOLD = 15      # consecutive failed sells on one trade before it is a breach
LOG_SILENCE_MIN = 10.0         # a live bot writing nothing for this long is suspect


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_status(payload: dict, path: str) -> None:
    """Always write, even on failure — a missing/stale file is the signal we're blind."""
    payload["checked_at"] = _now().isoformat()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"SAFETY: could not write status file {path}: {exc}", file=sys.stderr)


def db_open_trades(db_path: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, ticker, strike, option_type, expiry_date, contracts "
            "FROM paper_trades WHERE status = 'open'"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


async def broker_positions() -> list[dict]:
    from options_owl.config.settings import Settings
    from options_owl.execution.webull_executor import WebullExecutor

    ex = WebullExecutor(Settings())
    await ex.init()
    return await ex.get_open_option_positions() or []


def check_quantities(trades: list[dict], positions: list[dict]) -> list[str]:
    """THE invariant: every open record must match the broker's contract count."""
    breaches: list[str] = []
    by_key: dict[tuple, int] = {}
    for p in positions:
        key = (
            str(p.get("ticker", "")).upper(),
            round(float(p.get("strike") or 0), 2),
            str(p.get("option_type", "")).lower(),
        )
        by_key[key] = int(p.get("quantity") or 0)

    for t in trades:
        key = (
            str(t["ticker"]).upper(),
            round(float(t["strike"] or 0), 2),
            str(t["option_type"]).lower(),
        )
        db_qty = int(t["contracts"] or 0)
        broker_qty = by_key.get(key)
        if broker_qty is None:
            breaches.append(
                f"#{t['id']} {key[0]} ${key[1]} {key[2].upper()} x{db_qty} is OPEN in the "
                f"DB but ABSENT at the broker (exit may already have happened, or the "
                f"record is stale)"
            )
        elif broker_qty != db_qty:
            breaches.append(
                f"#{t['id']} {key[0]} ${key[1]} {key[2].upper()} QUANTITY MISMATCH: "
                f"recorded {db_qty}x vs broker {broker_qty}x — every exit will be rejected "
                f"as a naked short until this is corrected (the 2026-08-13 failure)"
            )
    return breaches


def check_stuck_exits(log_path: str) -> list[str]:
    """Repeated identical sell failures mean the exit path cannot execute."""
    breaches: list[str] = []
    if not os.path.exists(log_path):
        return breaches
    counts: dict[str, int] = {}
    try:
        with open(log_path, errors="ignore") as fh:
            for line in fh:
                if "EXIT ATTEMPT #" not in line or "trade#" not in line:
                    continue
                tid = line.split("trade#", 1)[1].split()[0].strip(":")
                counts[tid] = counts.get(tid, 0) + 1
    except Exception as exc:  # noqa: BLE001
        return [f"could not scan the log for stuck exits: {exc}"]

    for tid, n in counts.items():
        if n >= STUCK_SELL_THRESHOLD:
            breaches.append(
                f"trade#{tid} has {n} exit attempts today — an exit that cannot execute "
                f"(threshold {STUCK_SELL_THRESHOLD})"
            )
    return breaches


def _market_is_open() -> bool:
    """Single source of truth, per CLAUDE.md. Falls back to a conservative weekday/RTH
    window if the module cannot be imported, so a bad import can never make the check
    scream all night."""
    try:
        from options_owl.sourcing.utils.market_hours import is_market_open
        return bool(is_market_open())
    except Exception:  # noqa: BLE001
        from zoneinfo import ZoneInfo
        et = _now().astimezone(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return False
        mins = et.hour * 60 + et.minute
        return 9 * 60 + 30 <= mins <= 16 * 60


def check_log_freshness(log_path: str) -> list[str]:
    """A live bot that has stopped writing may have a wedged event loop.

    ONLY meaningful while the market is open. The bots are legitimately idle overnight and
    at weekends, and the first version of this check did not know that: it fired every 5
    minutes all evening, alerting on both ntfy and SMS. A watchdog that cries wolf nightly
    gets muted, which is precisely the failure it exists to prevent -- so silence outside
    market hours is expected, not a breach.
    """
    if not _market_is_open():
        return []
    if not os.path.exists(log_path):
        return [f"no log file at {log_path} — is the bot running?"]
    age_min = (_now().timestamp() - os.path.getmtime(log_path)) / 60.0
    if age_min > LOG_SILENCE_MIN:
        return [
            f"log has been silent for {age_min:.1f} min (>{LOG_SILENCE_MIN:.0f}) during "
            f"MARKET HOURS — the monitor loop may be blocked, which stops ALL exits"
        ]
    return []


async def main_async(args) -> int:
    status: dict = {"bot": args.bot, "breaches": [], "ok": False}
    log_path = (
        f"/app/journal/logs/options_owl_{_now().strftime('%Y-%m-%d')}.log"
        if args.log is None else args.log
    )

    try:
        trades = db_open_trades(args.db)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"cannot read the trade DB: {exc}"
        _write_status(status, args.status)
        print(f"SAFETY BLIND: {status['error']}", file=sys.stderr)
        return 2

    breaches: list[str] = []
    breaches += check_log_freshness(log_path)
    # Stuck exits matter only while we could still act on them. After the close the count
    # is a historical fact about a trade that is already resolved, and re-alerting on it
    # every 5 minutes overnight is noise.
    if _market_is_open():
        breaches += check_stuck_exits(log_path)

    # Only consult the broker when we have open records to compare, so a credentials
    # problem cannot mark a genuinely flat, healthy bot as broken.
    if trades:
        try:
            positions = await asyncio.wait_for(broker_positions(), timeout=45)
        except Exception as exc:  # noqa: BLE001
            status["error"] = f"cannot read broker positions: {exc}"
            status["open_records"] = len(trades)
            _write_status(status, args.status)
            print(f"SAFETY BLIND: {status['error']}", file=sys.stderr)
            return 2
        breaches += check_quantities(trades, positions)
        status["broker_positions"] = len(positions)

    status["open_records"] = len(trades)
    status["breaches"] = breaches
    status["ok"] = not breaches
    _write_status(status, args.status)

    if breaches:
        print(f"SAFETY BREACH ({args.bot}): {len(breaches)} issue(s)")
        for b in breaches:
            print(f"  - {b}")
        return 1
    print(f"SAFETY OK ({args.bot}): {len(trades)} open record(s), invariants hold")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="kody")
    ap.add_argument("--db", default="/app/journal/raw_messages.db")
    ap.add_argument("--log", default=None)
    ap.add_argument("--status", default=STATUS_PATH)
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(main_async(args)))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # An unexpected crash must not look like success.
        _write_status({"bot": args.bot, "ok": False, "error": f"crashed: {exc}"}, args.status)
        print(f"SAFETY BLIND: check crashed: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
