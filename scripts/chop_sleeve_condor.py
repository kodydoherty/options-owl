"""SPY 0DTE iron-condor "chop sleeve" backtest (2026-07-13).

Our whole book is LONG premium → it structurally bleeds on choppy/range days (theta + whipsaw). A short-
premium sleeve is the STRUCTURAL inverse: it makes money when price stays in a range and loses on trend
days. This tests whether a small, defined-risk 0DTE iron condor on SPY:
  (a) is GREEN on our 10 worst long-book days (the hedge thesis), and
  (b) has NON-NEGATIVE expected value over the full sample (so it's not just insurance we pay for).

Engine (conservative on purpose — every fill is the bad side of the spread):
  - Enter at ENTRY_MIN (default 11:00 ET) — the research is emphatic that condors opened in the first 2h
    average ~-0.4% while later entries average large positives (opening range must set first).
  - Short strikes at ~SHORT_DELTA (default 0.16 ≈ 1 SD, "outside the expected move"); wings WIDTH wide.
  - Entry credit = sell shorts at BID, buy wings at ASK (worst fills).
  - Walk each minute: close cost = buy shorts at ASK, sell wings at BID. HARD STOP if mark loss >=
    STOP_MULT x credit. Otherwise close at EXIT_MIN (default 14:30 ET) — no holding into the gamma hour.
  - P&L per 1 condor (x100 multiplier). Max risk ~ (WIDTH - credit)*100.

Data: journal/thetadata_options.db (option_greeks: delta+bid+ask+underlying; option_quotes: bid/ask/min).
Pulls cached to scratchpad so re-runs are API/DB-free.

Usage:
  python scripts/chop_sleeve_condor.py                 # 2026 YTD
  python scripts/chop_sleeve_condor.py --start 2024-01-01   # full 2.5yr EV
  python scripts/chop_sleeve_condor.py --delta 0.20 --width 5 --stop-mult 2.0 --entry 90 --exit 300
"""
import argparse
import os
import pickle
import sqlite3
from collections import defaultdict

DB = "journal/thetadata_options.db"
SCRATCH = "/private/tmp/claude-501/-Users-kody-dev-options-owl/ab4377c1-219f-4050-8390-59ecad2d6e56/scratchpad"

# 10 worst long-book days (flow resim, prod-faithful) — see chop_sleeve ranking
WORST = {"2026-06-15", "2026-06-10", "2026-04-02", "2026-05-14", "2026-05-11",
         "2026-04-15", "2026-03-31", "2026-04-06", "2026-05-07", "2026-05-15"}


def _mins(ts):
    """minutes since 09:30 from 'YYYY-MM-DD HH:MM:SS-05:00'."""
    hm = ts[11:16]
    h, m = int(hm[:2]), int(hm[3:5])
    return (h - 9) * 60 + m - 30


def trading_days(con, start, end):
    q = "SELECT DISTINCT expiration FROM option_ohlc WHERE ticker='SPY' AND expiration>=? AND expiration<=? ORDER BY expiration"
    return [r[0] for r in con.execute(q, (start, end))]


def build_day(con, day, entry_min, width, short_delta):
    """Return the 4-leg condor for `day`, or None if unbuildable."""
    # strikes + greeks near entry (pull whole day, filter minutes in Python — offset-proof)
    rows = con.execute(
        "SELECT strike,right,timestamp,delta,bid,ask,underlying_price FROM option_greeks "
        "WHERE ticker='SPY' AND expiration=? ",
        (day,),
    ).fetchall()
    if not rows:
        return None
    # nearest-to-entry snapshot per (strike,right), within +/-90 min of entry
    best = {}
    for strike, right, ts, delta, bid, ask, up in rows:
        if delta is None or bid is None or ask is None:
            continue
        d = abs(_mins(ts) - entry_min)
        if d > 90:
            continue
        k = (strike, right)
        if k not in best or d < best[k][0]:
            best[k] = (d, delta, bid, ask, up)
    if not best:
        return None
    calls = sorted((s, v) for (s, r), v in best.items() if r == "CALL")
    puts = sorted((s, v) for (s, r), v in best.items() if r == "PUT")
    if not calls or not puts:
        return None
    # call short: smallest strike with delta <= short_delta (just OTM of the 1SD line)
    cs = next((s for s, v in calls if abs(v[1]) <= short_delta), None)
    # put short: largest strike with |delta| <= short_delta
    ps = next((s for s, v in reversed(puts) if abs(v[1]) <= short_delta), None)
    if cs is None or ps is None:
        return None
    cl, pl = cs + width, ps - width
    legs = {"cs": (cs, "CALL"), "cl": (cl, "CALL"), "ps": (ps, "PUT"), "pl": (pl, "PUT")}
    # entry quotes for all 4 legs (nearest to entry)
    px = {}
    for name, (strike, right) in legs.items():
        b = best.get((strike, right))
        if b is None:
            return None
        px[name] = (b[2], b[3])  # (bid, ask)
    # entry credit: sell shorts @ bid, buy wings @ ask
    credit = (px["cs"][0] - px["cl"][1]) + (px["ps"][0] - px["pl"][1])
    if credit <= 0:
        return None
    up = best[(cs, "CALL")][4]
    return {"day": day, "legs": legs, "credit": credit, "under": up,
            "cs": cs, "cl": cl, "ps": ps, "pl": pl}


def leg_series(con, day, legs, entry_min, exit_min):
    """Per-minute bid/ask for each leg between entry and exit. {name: {min: (bid,ask)}}."""
    out = {n: {} for n in legs}
    strikes = tuple({s for s, _ in legs.values()})
    rows = con.execute(
        f"SELECT strike,right,timestamp,bid,ask FROM option_quotes "
        f"WHERE ticker='SPY' AND expiration=? AND strike IN ({','.join('?' * len(strikes))}) ",
        (day, *strikes),
    ).fetchall()
    rev = {v: n for n, v in legs.items()}
    for strike, right, ts, bid, ask in rows:
        n = rev.get((strike, right))
        if n is None or bid is None or ask is None:
            continue
        mi = _mins(ts)
        if entry_min <= mi <= exit_min:
            out[n][mi] = (bid, ask)
    return out


def sim(con, cd, entry_min, exit_min, stop_mult):
    """Simulate one condor. Returns (pnl_per_contract, exit_reason, exit_min)."""
    legs = cd["legs"]
    ser = leg_series(con, cd["day"], legs, entry_min, exit_min)
    credit = cd["credit"]
    stop_loss = stop_mult * credit
    last = {}  # forward-fill quotes
    for mi in range(entry_min + 1, exit_min + 1):
        ok = True
        for n in legs:
            if mi in ser[n]:
                last[n] = ser[n][mi]
            if n not in last:
                ok = False
        if not ok:
            continue
        # cost to close now: buy shorts @ ask, sell wings @ bid
        close_cost = (last["cs"][1] - last["cl"][0]) + (last["ps"][1] - last["pl"][0])
        pnl = (credit - close_cost) * 100
        if (credit - close_cost) <= -stop_loss:  # hard stop
            return pnl, "stop", mi
    # close at exit_min (or last available)
    if last and all(n in last for n in legs):
        close_cost = (last["cs"][1] - last["cl"][0]) + (last["ps"][1] - last["pl"][0])
        return (credit - close_cost) * 100, "eod", exit_min
    return 0.0, "nodata", exit_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--delta", type=float, default=0.16, help="short-strike target delta (~1SD)")
    ap.add_argument("--width", type=float, default=5.0, help="wing width $")
    ap.add_argument("--stop-mult", type=float, default=2.0, help="hard stop = X * credit")
    ap.add_argument("--entry", type=int, default=90, help="entry minute from 09:30 (90=11:00)")
    ap.add_argument("--exit", type=int, default=300, help="exit minute from 09:30 (300=14:30)")
    ap.add_argument("--fresh", action="store_true", help="ignore cache")
    args = ap.parse_args()

    cache = f"{SCRATCH}/condor_{args.start}_{args.end}_d{args.delta}_w{args.width}_s{args.stop_mult}_{args.entry}_{args.exit}.pkl"
    con = sqlite3.connect(DB)
    con.execute("PRAGMA query_only=1")

    if os.path.exists(cache) and not args.fresh:
        results = pickle.load(open(cache, "rb"))
    else:
        days = trading_days(con, args.start, args.end)
        results = []
        for i, day in enumerate(days):
            cd = build_day(con, day, args.entry, args.width, args.delta)
            if cd is None:
                continue
            pnl, reason, xm = sim(con, cd, args.entry, args.exit, args.stop_mult)
            results.append({"day": day, "pnl": pnl, "reason": reason, "credit": cd["credit"],
                            "cs": cd["cs"], "ps": cd["ps"], "under": cd["under"]})
            if (i + 1) % 25 == 0:
                print(f"  ...{i+1}/{len(days)} days")
        pickle.dump(results, open(cache, "wb"))

    n = len(results)
    tot = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    stops = sum(1 for r in results if r["reason"] == "stop")
    avg_credit = sum(r["credit"] for r in results) / n if n else 0
    print(f"\n{'='*66}\nSPY 0DTE IRON CONDOR — {args.start}..{args.end}")
    print(f"entry min {args.entry} (={9+(args.entry+30)//60}:{(args.entry+30)%60:02d}) "
          f"exit min {args.exit}  Δ{args.delta}  ${args.width:.0f}-wide  stop {args.stop_mult}x credit")
    print(f"{'='*66}")
    print(f"  condors: {n}   avg credit ${avg_credit:.2f}  (max risk ~${(args.width-avg_credit)*100:.0f})")
    print(f"  TOTAL P&L (1 contract/day): ${tot:+,.0f}   per-day EV ${tot/n:+.1f}")
    print(f"  win rate {wins}/{n} = {100*wins/n:.0f}%   hard-stopped {stops}/{n} = {100*stops/n:.0f}%")

    # (a) worst-long-book-day overlap
    print(f"\n  === (a) CONDOR on our 10 WORST long-book days (hedge check) ===")
    wr = [r for r in results if r["day"] in WORST]
    if wr:
        wt = sum(r["pnl"] for r in wr)
        ww = sum(1 for r in wr if r["pnl"] > 0)
        print(f"  {'day':<12}{'condor P&L':>12}{'exit':>8}")
        for r in sorted(wr, key=lambda z: z["day"]):
            print("  %-12s%12s%8s" % (r["day"], f"${r['pnl']:+,.0f}", r["reason"]))
        print(f"  ---> condor on worst days: ${wt:+,.0f}  ({ww}/{len(wr)} green)")
    else:
        print("  (no worst-day overlap in this window)")

    # (b) tail: worst condor days (the trend-day cost)
    print(f"\n  === (b) worst 8 condor days (the trend-day cost we pay) ===")
    for r in sorted(results, key=lambda z: z["pnl"])[:8]:
        print("  %-12s%12s%8s  credit $%.2f" % (r["day"], f"${r['pnl']:+,.0f}", r["reason"], r["credit"]))
    print(f"\nVERDICT: ships as a hedge only if (a) net-green on worst long-book days AND (b) full-sample")
    print(f"per-day EV >= ~0. If EV<0 it's paid insurance — value = the negative correlation, priced.")


if __name__ == "__main__":
    main()
