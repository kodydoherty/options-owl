"""Droplet-side: rebuild the UNDERLYING context runner_v1 needs, per historical trade.

Companion to validate_runner_v1.py. The live scorer reads 1m/5m stock candles and a
5-minute option-volume figure at entry; both still exist in Postgres, so the exact
feature vector can be reconstructed for trades that pre-date P(runner) persistence.

Emits {trade_id: {session, prior_close, prior_high, prior_low, opt_vol_5}}.
A trade whose context cannot be built is OMITTED rather than zero-filled — the
validator skips those, because a defaulted feature would silently shift the score.

Usage (on droplet, via a container that can reach postgres):
  python scripts/extract_runner_context.py --trades /data/live_exit_paths_greeks.pkl \
      --out /data/runner_ctx.pkl
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


async def build(trades: list[dict]) -> dict:
    conn = await asyncpg.connect(PG_DSN)
    out: dict[int, dict] = {}
    try:
        for i, t in enumerate(trades, 1):
            entry_ts = t["path"][0]["ts"]
            et_day = entry_ts.astimezone(ET).date()

            # session 1m closes up to entry (oldest -> newest), current ET date only
            rows = await conn.fetch(
                """
                SELECT bar_time, close FROM stock_candles
                WHERE ticker=$1 AND timeframe='1m'
                  AND bar_time >= $2 AND bar_time <= $3
                ORDER BY bar_time
                """,
                t["tk"], entry_ts - timedelta(hours=10), entry_ts,
            )
            sess = [float(r["close"]) for r in rows
                    if r["close"] and r["bar_time"].astimezone(ET).date() == et_day]

            # prior trading day OHLC from 5m candles
            prior = await conn.fetch(
                """
                SELECT bar_time, high, low, close FROM stock_candles
                WHERE ticker=$1 AND timeframe='5m'
                  AND bar_time >= $2 AND bar_time < $3
                ORDER BY bar_time
                """,
                t["tk"], entry_ts - timedelta(days=8), entry_ts,
            )
            days: dict = {}
            for r in prior:
                d = r["bar_time"].astimezone(ET).date()
                if d < et_day:
                    days.setdefault(d, []).append(r)
            pc = ph = pl = 0.0
            if days:
                pb = days[max(days)]
                pc = float(pb[-1]["close"] or 0)
                ph = max(float(x["high"] or 0) for x in pb)
                pl = min(float(x["low"] or 0) for x in pb if x["low"])

            # 5-minute option volume immediately before entry
            ov = await conn.fetchval(
                """
                SELECT COALESCE(MAX(volume),0) - COALESCE(MIN(volume),0) FROM option_ticks
                WHERE ticker=$1 AND option_type=$2 AND strike=$3 AND expiry_date=$4
                  AND captured_at BETWEEN $5 AND $6
                """,
                t["tk"], t["otype"], t["strike"], t["expiry"],
                entry_ts - timedelta(minutes=5), entry_ts,
            )

            if len(sess) >= 5 and pc > 0:
                out[t["tid"]] = {"session": sess, "prior_close": pc, "prior_high": ph,
                                 "prior_low": pl, "opt_vol_5": float(ov or 0)}
            if i % 100 == 0:
                print(f"  ...{i}/{len(trades)}  built={len(out)}", flush=True)
    finally:
        await conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", default="/data/runner_ctx.pkl")
    args = ap.parse_args()
    trades = [t for t in pickle.load(open(args.trades, "rb"))
              if t["was_webull"] and t["otype"] == "call"]
    print(f"building underlying context for {len(trades)} real-fill CALLs...", flush=True)
    ctx = asyncio.run(build(trades))
    print(f"built {len(ctx)}/{len(trades)} "
          f"({100*len(ctx)/max(1,len(trades)):.0f}%) — the rest lack candle history and are omitted")
    pickle.dump(ctx, open(args.out, "wb"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
