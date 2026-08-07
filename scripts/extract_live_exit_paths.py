"""Extract REAL live trades + their REAL recorded option price paths (droplet-side).

Runs ON the droplet, where the data lives:
  - live trades  -> journal/owlet-<bot>/raw_messages.db  (real Webull entry fills)
  - price paths  -> Postgres option_ticks (45GB, real recorded bid/ask, 2026-05-26 -> today)

Emits a compact pickle the exit sweep replays locally. The 45GB never leaves the droplet.

Why this beats a thetadata backtest for an EXIT question:
  * entries are the ACTUAL Webull fills (no fill modelling, no fantasy-fill contamination)
  * the path is the ACTUAL quoted bid/ask the bot could have sold into
  * coverage runs through TODAY (includes the August bleed), not thetadata's 2026-07-15 wall

CRITICAL: the path is extended past the trade's real close to END OF SESSION, so variants
that hold LONGER than prod did have real data to hold into. Without that the sweep can only
ever cut earlier, never later, and every "hold longer" variant is silently truncated.

Usage (on droplet):
  python scripts/extract_live_exit_paths.py --bots kody,dennis --days 95 \
      --out /tmp/live_exit_paths.pkl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import asyncpg

# asyncpg is what the project already depends on (psycopg2 is not in the image).
PG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://owl:owl_dev_2026@postgres:5432/options_owl"
)


def load_trades(bot: str, days: int, journal_root: str = "journal") -> list[dict]:
    """Real closed trades for one bot. Entry basis = the actual Webull fill when present.

    journal_root matters: each bot CONTAINER mounts only its own dir at /app/journal, so
    reading the whole fleet requires the host tree mounted somewhere (see --journal-root).
    """
    db = os.path.join(journal_root, f"owlet-{bot}", "raw_messages.db")
    if not os.path.exists(db):
        print(f"  (no db for {bot} at {db})", flush=True)
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT id, ticker, option_type, strike, expiry_date, contracts,
               premium_per_contract, webull_entry_fill_price, exit_premium,
               exit_reason, pnl_dollars, opened_at, closed_at, status,
               bot_source, webull_order_id, score
        FROM paper_trades
        WHERE date(opened_at) >= date('now', ?)
          AND status = 'closed'
          AND strike IS NOT NULL AND expiry_date IS NOT NULL
          AND contracts > 0
        ORDER BY id
        """,
        (f"-{days} days",),
    ).fetchall()
    con.close()

    out = []
    for r in rows:
        # Prefer the REAL broker fill as cost basis; fall back to the recorded premium.
        entry = r["webull_entry_fill_price"] or r["premium_per_contract"]
        if not entry or entry <= 0:
            continue
        out.append(
            {
                "bot": bot,
                "tid": r["id"],
                "tk": r["ticker"],
                "otype": (r["option_type"] or "call").lower(),
                "strike": float(r["strike"]),
                "expiry": r["expiry_date"],
                "contracts": int(r["contracts"]),
                "entry": float(entry),
                "opened_at": r["opened_at"],
                "closed_at": r["closed_at"],
                "actual_exit_reason": r["exit_reason"],
                "actual_pnl": r["pnl_dollars"],
                "actual_exit_premium": r["exit_premium"],
                "bot_source": r["bot_source"],
                "was_webull": bool(r["webull_order_id"]),
                "score": r["score"],
            }
        )
    return out


def _f(v):
    """asyncpg returns Decimal/None for numerics; normalise to float|None."""
    return float(v) if v is not None else None


def _parse_ts(s: str) -> datetime:
    """Parse a paper_trades timestamp to an aware UTC datetime.

    The column is NOT uniform: most rows are UTC-naive ('2026-08-07 14:10:13.123'),
    but some carry an explicit ET offset ('2026-05-12 14:01:40.279464-04:00').
    Stripping the offset (the obvious shortcut) would shift those rows by 4 hours and
    silently misalign them against the UTC option_ticks path. Honour it instead, and
    treat naive values as UTC per the CLAUDE.md convention.
    """
    raw = (s or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    try:
        dt = datetime.fromisoformat(raw.replace(" ", "T", 1))
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp: {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def attach_paths(trades: list[dict], pad_hours: float) -> list[dict]:
    """Attach the real recorded tick path for each contract, extended to end of session.

    Batched by contract so the (ticker, option_type, strike, expiry, captured_at) index
    is used — a per-trade round trip against a 163M-row table would take hours.
    """
    conn = await asyncpg.connect(PG_DSN)

    # Group trades by contract so one query serves every trade on that contract.
    by_contract: dict[tuple, list[dict]] = defaultdict(list)
    for t in trades:
        by_contract[(t["tk"], t["otype"], t["strike"], t["expiry"])].append(t)

    kept = []
    try:
        for i, (key, group) in enumerate(sorted(by_contract.items()), 1):
            tk, otype, strike, expiry = key
            opens = [_parse_ts(t["opened_at"]) for t in group]
            lo = min(opens)
            # Extend to end of the last trade's session so "hold longer" variants have data.
            hi = max(opens) + timedelta(hours=pad_hours)

            # Greeks come along for the ride: IV enables expected-move / IV-normalised
            # exit bands (Stock x IV x sqrt(DTE/365)), delta/gamma let us ask whether a
            # runner is identifiable at the dip. Sizes let us test book-depth effects.
            ticks = await conn.fetch(
                """
                SELECT captured_at, bid, ask, mid, underlying_price,
                       iv, delta, gamma, theta, vega, bid_size, ask_size
                FROM option_ticks
                WHERE ticker=$1 AND option_type=$2 AND strike=$3 AND expiry_date=$4
                  AND captured_at BETWEEN $5 AND $6
                ORDER BY captured_at
                """,
                tk, otype, strike, expiry, lo - timedelta(minutes=2), hi,
            )

            if len(ticks) < 5:
                continue

            for t in group:
                t0 = _parse_ts(t["opened_at"])
                path = [
                    {
                        "ts": tick["captured_at"],
                        "bid": float(tick["bid"]) if tick["bid"] is not None else None,
                        "ask": float(tick["ask"]) if tick["ask"] is not None else None,
                        "mid": float(tick["mid"]),
                        "up": (
                            float(tick["underlying_price"])
                            if tick["underlying_price"] is not None
                            else None
                        ),
                        "iv": _f(tick["iv"]),
                        "delta": _f(tick["delta"]),
                        "gamma": _f(tick["gamma"]),
                        "theta": _f(tick["theta"]),
                        "vega": _f(tick["vega"]),
                        "bid_sz": tick["bid_size"],
                        "ask_sz": tick["ask_size"],
                    }
                    for tick in ticks
                    if tick["captured_at"] >= t0
                    and tick["mid"] is not None
                    and tick["mid"] > 0
                ]
                if len(path) < 5:
                    continue
                t["path"] = path
                kept.append(t)

            if i % 100 == 0:
                print(
                    f"  ...{i}/{len(by_contract)} contracts, {len(kept)} trades w/ paths",
                    flush=True,
                )
    finally:
        await conn.close()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="kody,dennis")
    ap.add_argument("--days", type=int, default=95)
    ap.add_argument("--pad-hours", type=float, default=6.5,
                    help="Extend each path this far past entry (end-of-session) so "
                         "hold-longer variants are not truncated")
    ap.add_argument("--out", default="/tmp/live_exit_paths.pkl")
    ap.add_argument("--journal-root", default="journal",
                    help="Path holding owlet-<bot>/ dirs (a bot container only mounts its own)")
    args = ap.parse_args()

    trades: list[dict] = []
    for bot in args.bots.split(","):
        got = load_trades(bot.strip(), args.days, args.journal_root)
        print(f"{bot.strip()}: {len(got)} closed trades", flush=True)
        trades.extend(got)

    print(f"total {len(trades)} trades — attaching real tick paths...", flush=True)
    kept = asyncio.run(attach_paths(trades, args.pad_hours))

    n_wb = sum(1 for t in kept if t["was_webull"])
    print(f"\n{len(kept)}/{len(trades)} trades have usable paths ({n_wb} real-Webull)")
    if kept:
        ds = sorted(t["opened_at"][:10] for t in kept)
        print(f"window: {ds[0]} -> {ds[-1]}")
        avg = sum(len(t["path"]) for t in kept) / len(kept)
        print(f"avg ticks/trade: {avg:.0f}")

    with open(args.out, "wb") as f:
        pickle.dump(kept, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
