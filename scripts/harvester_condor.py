"""SPY 0DTE iron-condor backtest on REAL harvested bid/ask (2026-07-13).

Lever #1: the thetadata archive is ATM-only, but the live harvester (options_data.db) stored the full
+/-10% chain with REAL bid/ask + delta at ~1-2min cadence. This runs the condor on that real data over the
harvested window (local copy: 2026-03-27 .. 2026-05-26 — includes 6-7 of our 10 worst long-book days).

Same engine/acceptance as chop_sleeve_condor.py, but every price is a REAL harvested NBBO (no modeling):
  - Enter ENTRY_MIN (11:00 ET); shorts at ~SHORT_DELTA (real stored delta); wings WIDTH wide.
  - Credit = sell shorts@bid, buy wings@ask. Close = buy shorts@ask, sell wings@bid.
  - HARD STOP at STOP_MULT x credit; else close at EXIT_MIN (14:30 ET). P&L per 1 condor (x100).

Timestamps are UTC; window is EDT so 09:30 ET = 13:30 UTC -> minute = (Hutc*60+Mutc) - 810.

Usage: python scripts/harvester_condor.py [--delta 0.16] [--width 5] [--stop-mult 2.0] [--min-credit 0.40]
"""
import argparse
import os
import pickle
import sqlite3

DB = "journal/owlet-harvester/options_data.db"
SCRATCH = "/private/tmp/claude-501/-Users-kody-dev-options-owl/ab4377c1-219f-4050-8390-59ecad2d6e56/scratchpad"
WORST = {"2026-06-15", "2026-06-10", "2026-04-02", "2026-05-14", "2026-05-11",
         "2026-04-15", "2026-03-31", "2026-04-06", "2026-05-07", "2026-05-15"}


def _min_utc(ts):
    """minute-from-open for a UTC ISO ts on an EDT day (09:30 ET = 13:30 UTC)."""
    h, m = int(ts[11:13]), int(ts[14:16])
    return h * 60 + m - 810


def day_rows(con, day):
    """All SPY snapshots for contracts expiring `day`, captured on `day`. Grouped by contract."""
    rows = con.execute(
        "SELECT s.contract_ticker, c.strike, c.option_type, s.captured_at, s.bid, s.ask, s.delta, s.underlying_price "
        "FROM harvest_snapshots s JOIN harvest_contracts c ON s.contract_ticker = c.contract_ticker "
        "WHERE c.underlying='SPY' AND c.expiry_date=? AND s.captured_at LIKE ? ",
        (day, f"{day}%"),
    ).fetchall()
    by = {}
    for ct, strike, otype, ts, bid, ask, delta, up in rows:
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            continue
        try:
            delta = float(delta) if delta is not None else None
        except (TypeError, ValueError):
            delta = None
        mi = _min_utc(ts)
        by.setdefault((strike, otype.upper()), []).append((mi, bid, ask, delta, up))
    return by


def nearest(series, target):
    return min(series, key=lambda r: abs(r[0] - target))


def build(by, entry_min, width, short_delta):
    # entry snapshot per strike/right (nearest to entry, within +/-60min)
    ent = {}
    for k, ser in by.items():
        cand = [r for r in ser if abs(r[0] - entry_min) <= 60]
        if cand:
            ent[k] = nearest(cand, entry_min)
    calls = sorted((s, ent[(s, "CALL")]) for s, r in ent if r == "CALL")
    puts = sorted((s, ent[(s, "PUT")]) for s, r in ent if r == "PUT")
    if not calls or not puts:
        return None
    cs = next((s for s, r in calls if r[3] is not None and abs(r[3]) <= short_delta), None)
    ps = next((s for s, r in reversed(puts) if r[3] is not None and abs(r[3]) <= short_delta), None)
    if cs is None or ps is None:
        return None
    cl, pl = cs + width, ps - width
    legs = {"cs": (cs, "CALL"), "cl": (cl, "CALL"), "ps": (ps, "PUT"), "pl": (pl, "PUT")}
    for name, k in legs.items():
        if k not in ent:
            return None
    credit = (ent[legs["cs"]][2 - 1] - ent[legs["cl"]][2]) + (ent[legs["ps"]][1] - ent[legs["pl"]][2])
    # ent tuple = (mi, bid, ask, delta, up); bid=idx1 ask=idx2
    credit = (ent[legs["cs"]][1] - ent[legs["cl"]][2]) + (ent[legs["ps"]][1] - ent[legs["pl"]][2])
    if credit <= 0:
        return None
    return {"legs": legs, "credit": credit, "under": ent[legs["cs"]][4], "cs": cs, "ps": ps}


def sim(by, cd, entry_min, exit_min, stop_mult, profit_take=0.0):
    legs = cd["legs"]
    credit = cd["credit"]
    # build per-minute (bid,ask) for each leg
    ser = {}
    for name, k in legs.items():
        d = {}
        for mi, bid, ask, delta, up in by.get(k, []):
            if entry_min < mi <= exit_min:
                d[mi] = (bid, ask)
        ser[name] = d
    last = {}
    for mi in range(entry_min + 1, exit_min + 1):
        for name in legs:
            if mi in ser[name]:
                last[name] = ser[name][mi]
        if not all(n in last for n in legs):
            continue
        close_cost = (last["cs"][1] - last["cl"][0]) + (last["ps"][1] - last["pl"][0])
        captured = credit - close_cost
        # profit-take: close once we've captured >= profit_take fraction of the credit
        if profit_take > 0 and captured >= profit_take * credit:
            return captured * 100, "take", mi
        # hard stop (skipped when stop_mult<=0 = defined-risk hold-to-close)
        if stop_mult > 0 and captured <= -stop_mult * credit:
            return captured * 100, "stop", mi
    if last and all(n in last for n in legs):
        close_cost = (last["cs"][1] - last["cl"][0]) + (last["ps"][1] - last["pl"][0])
        return (credit - close_cost) * 100, "eod", exit_min
    return 0.0, "nodata", exit_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.16)
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--stop-mult", type=float, default=2.0)
    ap.add_argument("--entry", type=int, default=90)
    ap.add_argument("--exit", type=int, default=300)
    ap.add_argument("--profit-take", type=float, default=0.0, help="close at this fraction of credit (0=off)")
    ap.add_argument("--min-credit", type=float, default=0.0, help="skip days whose credit < this")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cache = f"{SCRATCH}/hcondor_d{args.delta}_w{args.width}_s{args.stop_mult}_{args.entry}_{args.exit}_pt{args.profit_take}.pkl"
    con = sqlite3.connect(DB)
    con.execute("PRAGMA query_only=1")
    if os.path.exists(cache) and not args.fresh:
        results = pickle.load(open(cache, "rb"))
    else:
        days = [r[0] for r in con.execute(
            "SELECT DISTINCT expiry_date FROM harvest_contracts WHERE underlying='SPY' ORDER BY expiry_date")]
        results = []
        for day in days:
            by = day_rows(con, day)
            if not by:
                continue
            cd = build(by, args.entry, args.width, args.delta)
            if cd is None:
                continue
            pnl, reason, xm = sim(by, cd, args.entry, args.exit, args.stop_mult, args.profit_take)
            results.append({"day": day, "pnl": pnl, "reason": reason, "credit": cd["credit"],
                            "cs": cd["cs"], "ps": cd["ps"], "under": cd["under"]})
            print(f"  {day}  cr ${cd['credit']:.2f}  spot {cd['under']:.0f}  short {cd['ps']:.0f}/{cd['cs']:.0f}  "
                  f"P&L ${pnl:+.0f} [{reason}]")
        pickle.dump(results, open(cache, "wb"))

    results = [r for r in results if r["credit"] >= args.min_credit]
    n = len(results)
    if not n:
        print("no condors built"); return
    tot = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    stops = sum(1 for r in results if r["reason"] == "stop")
    ac = sum(r["credit"] for r in results) / n
    print(f"\n{'='*64}\nSPY 0DTE IRON CONDOR on REAL harvested bid/ask — {results[0]['day']}..{results[-1]['day']}")
    print(f"Δ{args.delta} ${args.width:.0f}-wide  stop {args.stop_mult}x  entry 11:00 exit 14:30  (min-credit ${args.min_credit})")
    print(f"{'='*64}")
    print(f"  condors: {n}   avg credit ${ac:.2f}  (max risk ~${(args.width-ac)*100:.0f}/contract)")
    print(f"  TOTAL (1 contract/day): ${tot:+,.0f}   per-day EV ${tot/n:+.1f}")
    print(f"  win rate {wins}/{n}={100*wins/n:.0f}%   hard-stopped {stops}/{n}={100*stops/n:.0f}%")

    wr = [r for r in results if r["day"] in WORST]
    print(f"\n  === (a) condor on WORST long-book days present in window ===")
    if wr:
        for r in sorted(wr, key=lambda z: z["day"]):
            print("  %-12s condor %10s  [%s]  credit $%.2f" % (r["day"], f"${r['pnl']:+,.0f}", r["reason"], r["credit"]))
        wt = sum(r["pnl"] for r in wr); ww = sum(1 for r in wr if r["pnl"] > 0)
        print(f"  ---> ${wt:+,.0f}  ({ww}/{len(wr)} green)")
    else:
        print("  (none in window)")
    print(f"\n  === (b) worst 6 condor days (trend-day cost) ===")
    for r in sorted(results, key=lambda z: z["pnl"])[:6]:
        print("  %-12s %10s  [%s]  credit $%.2f" % (r["day"], f"${r['pnl']:+,.0f}", r["reason"], r["credit"]))


if __name__ == "__main__":
    main()
