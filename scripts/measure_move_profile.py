"""Measure each ticker's intraday OPTION move profile, independent of our trades.

WHY
---
The exit stack uses ABSOLUTE percentage thresholds (profit-lock arms +25%, nevergreen
cuts -8%, hardstop -25%) that were calibrated on the tech/index book. Live fills show
the expansion tickers move roughly half as much:

    tech/index   avg MFE 23.4%   avg MAE -20.4%   (n=513)
    expansion    avg MFE 13.6%   avg MAE -14.6%   (n=26)

and the consequence is mechanical: with a 13.6% average MFE, most expansion trades can
NEVER reach a +25% profit-lock arm. Their live exit-reason mix confirms it — only
nevergreen_cut and multiday_call_hardstop appear, i.e. loss gates ONLY. They exit
through the downside gates because the upside gate is out of reach.

WHAT THIS DOES
--------------
Rebuilds the move profile from `option_ticks` directly rather than from our fills, so a
ticker with 2 trades is measured as reliably as one with 200. For each ticker/day it
picks the ATM 0DTE-ish call at a reference time and walks the rest of the session,
recording MFE and MAE from that entry. The result is a per-ticker distribution of "how
far does this thing actually move", which is what the thresholds should scale to.

Then thresholds are derived by PERCENTILE MATCHING, not by a naive ratio: find where the
live threshold sits in the tech distribution, and read the equivalent absolute value off
each other ticker's distribution. A ratio would assume the distributions have the same
shape; percentile matching does not.

HONEST LIMITS
-------------
* option_ticks starts 2026-05-26, so this is ~10 weeks, not a full cycle.
* The reference entry is a fixed clock time, not a real signal — this measures the
  ticker's move ENVELOPE, not our edge. That is the point (it must not depend on our
  entry timing), but it means the absolute MFE here is not what a real trade would get.
* Sampling is one contract per ticker/day; a ticker whose ATM strike is poorly covered
  in a given day is skipped rather than approximated.

Usage (droplet, container with postgres access):
  python scripts/measure_move_profile.py --out /data/move_profile.pkl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
from datetime import timedelta
from zoneinfo import ZoneInfo

import asyncpg

ET = ZoneInfo("America/New_York")
PG_DSN = os.environ.get("DATABASE_URL", "postgresql://owl:owl_dev_2026@postgres:5432/options_owl")

# Groups mirror how the exit stack actually treats them today.
GROUPS = {
    "index": ["SPY", "QQQ", "IWM"],
    "tech": ["NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "AMD", "MSTR",
             "PLTR", "NFLX", "SMCI", "BA", "JPM"],
    "expansion": ["ORCL", "INTC", "TSM", "ARM", "SMH", "USO", "SLV", "GDX"],
}


async def profile_ticker(conn, ticker: str, ref_hour: int, ref_min: int) -> list[dict]:
    """One sample per trading day: ATM call at the reference time, then MFE/MAE after."""
    days = await conn.fetch(
        """
        SELECT DISTINCT captured_at::date AS d
        FROM option_ticks WHERE ticker=$1 AND option_type='call'
        ORDER BY d
        """,
        ticker,
    )
    out: list[dict] = []
    for row in days:
        d = row["d"]
        # reference instant, ET -> UTC (stored tz-aware)
        ref = (
            __import__("datetime").datetime(d.year, d.month, d.day, ref_hour, ref_min, tzinfo=ET)
        )
        # nearest expiry available that day = the 0DTE-ish contract we would trade
        exp = await conn.fetchval(
            """
            SELECT MIN(expiry_date) FROM option_ticks
            WHERE ticker=$1 AND option_type='call' AND captured_at::date=$2
              AND expiry_date >= $3
            """,
            ticker, d, d.isoformat(),
        )
        if not exp:
            continue
        # the strike closest to spot at the reference instant
        atm = await conn.fetchrow(
            """
            SELECT strike, underlying_price, mid, bid
            FROM option_ticks
            WHERE ticker=$1 AND option_type='call' AND expiry_date=$2
              AND captured_at BETWEEN $3 AND $4
              AND underlying_price IS NOT NULL AND mid > 0
            ORDER BY ABS(strike - underlying_price), captured_at
            LIMIT 1
            """,
            ticker, exp, ref - timedelta(minutes=5), ref + timedelta(minutes=15),
        )
        if not atm or not atm["mid"] or atm["mid"] <= 0:
            continue
        entry = float(atm["mid"])
        path = await conn.fetch(
            """
            SELECT bid, mid FROM option_ticks
            WHERE ticker=$1 AND option_type='call' AND strike=$2 AND expiry_date=$3
              AND captured_at BETWEEN $4 AND $5
            ORDER BY captured_at
            """,
            ticker, atm["strike"], exp, ref, ref + timedelta(hours=6),
        )
        px = [float(r["bid"] or r["mid"]) for r in path if (r["bid"] or r["mid"])]
        if len(px) < 10:
            continue
        out.append({
            "date": d.isoformat(),
            "entry": entry,
            "mfe": (max(px) - entry) / entry * 100.0,
            "mae": (min(px) - entry) / entry * 100.0,
        })
    return out


async def main_async(args) -> None:
    conn = await asyncpg.connect(PG_DSN)
    prof: dict[str, list] = {}
    try:
        tickers = [t for g in GROUPS.values() for t in g]
        for i, tk in enumerate(tickers, 1):
            prof[tk] = await profile_ticker(conn, tk, args.ref_hour, args.ref_min)
            print(f"  [{i}/{len(tickers)}] {tk:<6} {len(prof[tk])} day-samples", flush=True)
    finally:
        await conn.close()
    pickle.dump({"groups": GROUPS, "profile": prof}, open(args.out, "wb"))
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-hour", type=int, default=10)
    ap.add_argument("--ref-min", type=int, default=0)
    ap.add_argument("--out", default="/data/move_profile.pkl")
    main_async_args = ap.parse_args()
    asyncio.run(main_async(main_async_args))


if __name__ == "__main__":
    main()
