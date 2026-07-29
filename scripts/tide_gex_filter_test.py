"""UW net-premium TIDE (#4) + GEX regime (#3) as confirmation vetoes on the FLOW book (2026-07-10).

Every standalone entry signal this session refuted; flow is our real book. So instead of trading
tide/GEX directly, test them as VETOES on flow trades — can they cut the counter-trend losers
(the SPY-put-into-a-rally type) without gutting the winners? This is continuous with the live
ENABLE_V7_TIDE_GATE (puts-only today); here we test both sides + GEX on fresh intraday data.

For each flow trade (ticker, date, entry-minute) from the prod-faithful flow backtest, we look up:
  - TIDE  = cumulative net_delta (and net_call−net_put premium) from the open THROUGH entry
            (net_prem_ticks). Positive = net bullish positioning.
  - GEX   = sign of gamma_per_pct_oi at entry (spot_gex, market hours). Negative = dealers short
            gamma = moves amplify (momentum-friendly); positive = pinning (momentum fades).
Then simulate filters and report book P&L delta. A veto is real only if the dropped bucket was a
net loser AND the kept book improves — not just thinning a slice.

Usage: python scripts/tide_gex_filter_test.py
"""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_gold_standard_report as R  # noqa: E402

BASE = 750.0
DB = str(Path(__file__).resolve().parent.parent / "journal" / "uw_historical.db")


def _mi(ts):
    """UTC ISO 'YYYY-MM-DDThh:mm:...Z' -> minutes after 9:30 ET (EDT)."""
    hh = int(ts[11:13]); mm = int(ts[14:16])
    return (hh - 13) * 60 + (mm - 30)


def load_intraday():
    """Return tide[(tk,date)] = sorted [(mi, cum_delta, cum_prem)], gex[(tk,date)] = sorted [(mi, gamma)]."""
    conn = sqlite3.connect(DB)
    tide_raw = defaultdict(list)
    for tk, d, tt, ncp, npp, nd in conn.execute(
            "SELECT ticker,date,tape_time,net_call_premium,net_put_premium,net_delta FROM net_prem_ticks"):
        mi = _mi(tt)
        if 0 <= mi <= 390:
            tide_raw[(tk, d)].append((mi, nd or 0.0, (ncp or 0.0) - (npp or 0.0)))
    tide = {}
    for k, rows in tide_raw.items():
        rows.sort()
        cd = cp = 0.0; series = []
        for mi, nd, pr in rows:
            cd += nd; cp += pr; series.append((mi, cd, cp))
        tide[k] = series
    gex = defaultdict(list)
    for tk, d, tm, g in conn.execute("SELECT ticker,date,time,gamma_per_pct_oi FROM spot_gex"):
        mi = _mi(tm)
        if 0 <= mi <= 390 and g is not None:
            gex[(tk, d)].append((mi, g))
    for k in gex:
        gex[k].sort()
    conn.close()
    return tide, gex


def _at(series, mi, idx):
    """Last tuple with tuple[0] <= mi; return tuple[idx] or None."""
    val = None
    for row in series:
        if row[0] <= mi:
            val = row[idx]
        else:
            break
    return val


def pnl(recs):
    return sum(BASE * r["ret_pct"] / 100 for r in recs)


def pf(recs):
    g = sum(BASE * r["ret_pct"] / 100 for r in recs if r["ret_pct"] > 0)
    l = -sum(BASE * r["ret_pct"] / 100 for r in recs if r["ret_pct"] < 0)
    return g / l if l > 0 else float("inf")


def wr(recs):
    return (sum(1 for r in recs if r["ret_pct"] > 0) / len(recs) * 100) if recs else 0.0


def line(lbl, recs, base=None):
    d = f"  Δ${pnl(recs) - base:>+8,.0f}" if base is not None else ""
    print(f"  {lbl:<40}{len(recs):>4} tr  ${pnl(recs):>+8,.0f}  PF {pf(recs):>5.2f}  WR {wr(recs):>3.0f}%{d}")


def main():
    print("Collecting flow book (calls+puts, prod-faithful V7 exits)…")
    trades = R.collect(True, R.PUT_UNIV) + R.collect(False, R.CALL_UNIV)
    tide, gex = load_intraday()
    tdays = sorted({k[1] for k in tide})
    print(f"intraday tide covers {tdays[0]}..{tdays[-1]} ({len(tdays)} days)\n" if tdays else "no tide data\n")

    # Attach tide/GEX to each flow trade that falls in the covered window.
    matched = []
    for t in trades:
        k = (t["ticker"], t["date"])
        if k not in tide:
            continue
        t = dict(t)
        t["tide_delta"] = _at(tide[k], t["mi"], 1)
        t["tide_prem"] = _at(tide[k], t["mi"], 2)
        t["gex"] = _at(gex.get(k, []), t["mi"], 1)
        matched.append(t)

    print(f"{len(matched)} of {len(trades)} flow trades fall in the intraday-data window\n")
    if not matched:
        print("No overlap yet — intraday download still filling. Re-run when it finishes.")
        return
    base = pnl(matched)
    print(f"=== BASELINE (matched flow book) ===  ${base:+,.0f}  PF {pf(matched):.2f}  WR {wr(matched):.0f}%\n")

    calls = [t for t in matched if t["side"] == "call"]
    puts = [t for t in matched if t["side"] == "put"]

    print("=== TIDE alignment (net_delta sign at entry) ===")
    # aligned: call with bullish tide, put with bearish tide
    aligned = [t for t in matched if t["tide_delta"] is not None
               and ((t["side"] == "call" and t["tide_delta"] > 0) or (t["side"] == "put" and t["tide_delta"] < 0))]
    misaligned = [t for t in matched if t["tide_delta"] is not None
                  and ((t["side"] == "call" and t["tide_delta"] <= 0) or (t["side"] == "put" and t["tide_delta"] >= 0))]
    line("tide-ALIGNED (keep)", aligned)
    line("tide-MISALIGNED (candidate to drop)", misaligned)
    line("BOOK if drop misaligned", aligned, base)
    print("  -- calls only --")
    line("  call + bullish tide", [t for t in calls if t["tide_delta"] and t["tide_delta"] > 0])
    line("  call + bearish tide", [t for t in calls if t["tide_delta"] is not None and t["tide_delta"] <= 0])
    print("  -- puts only --")
    line("  put + bearish tide", [t for t in puts if t["tide_delta"] is not None and t["tide_delta"] < 0])
    line("  put + bullish tide", [t for t in puts if t["tide_delta"] is not None and t["tide_delta"] >= 0])

    print("\n=== GEX regime (sign of gamma_per_pct_oi at entry) ===")
    neg = [t for t in matched if t["gex"] is not None and t["gex"] < 0]
    pos = [t for t in matched if t["gex"] is not None and t["gex"] > 0]
    line("GEX<0 (amplifying / momentum-ok)", neg)
    line("GEX>0 (pinning — candidate to drop)", pos)
    line("BOOK if keep GEX<0 only", neg, base)

    print("\n=== GEX robustness: month-by-month (does GEX<0 beat GEX>0 EVERY month?) ===")
    print(f"  {'month':<9}{'GEX<0 n/PF/P&L':>26}{'GEX>0 n/PF/P&L':>26}")
    months = sorted({t["date"][:7] for t in matched if t["gex"] is not None})
    for m in months:
        mn = [t for t in matched if t["date"][:7] == m and t["gex"] is not None and t["gex"] < 0]
        mp = [t for t in matched if t["date"][:7] == m and t["gex"] is not None and t["gex"] > 0]
        print(f"  {m:<9}{f'{len(mn)}  PF{pf(mn):.2f}  ${pnl(mn):+,.0f}':>26}"
              f"{f'{len(mp)}  PF{pf(mp):.2f}  ${pnl(mp):+,.0f}':>26}")

    print("\n=== COMBINED (tide-aligned AND GEX<0) ===")
    combo = [t for t in aligned if t["gex"] is not None and t["gex"] < 0]
    line("keep tide-aligned & GEX<0", combo, base)

    print("\nNOTE: flat-$ measure. A veto ships only if the dropped bucket is a genuine net loser and")
    print("the kept book P&L rises — thin/variance slices don't count. Bounded by thetadata (≤2026-07-01).")


if __name__ == "__main__":
    main()
