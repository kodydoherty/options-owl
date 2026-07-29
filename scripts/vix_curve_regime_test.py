"""VIX-curve day-regime separation test (2026-07-14).

Question: do our chop/losing days cluster in a distinct VIX-term-structure regime that's knowable at the
OPEN (before our ~11am entries)? If yes -> a forward-looking (implied) regime tilt/detector is possible;
if the buckets don't separate, we've cheaply refuted it (same wall as the realized efficiency-ratio).

No lookahead: every curve value is the day's OPEN (^VIX1D/^VIX9D/^VIX/^VIX3M/^VVIX), available 09:30 ET.
Book P&L = flow resim (prod-faithful FSM), per day. Metrics:
  slope9 = VIX9D/VIX   (<1 contango/calm, >1 backwardation/stress)
  slope1 = VIX1D/VIX   (short-end; 0DTE-specific)
  far    = VIX/VIX3M   (near/med; <1 normal)
  vvix   = VVIX level  (vol-of-vol; rising = move coming)
  v1d    = VIX1D level  (absolute 0DTE implied vol)
  onchg  = VIX open / prior VIX close - 1  (overnight vol expansion vs crush)

Usage: python scripts/vix_curve_regime_test.py [--since 2026-03-01]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402
from exit_risk_sweep import BASE, CACHE, _mk_settings  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402

WORST = {"2026-06-15", "2026-06-10", "2026-04-02", "2026-05-14", "2026-05-11",
         "2026-04-15", "2026-03-31", "2026-04-06", "2026-05-07", "2026-05-15"}


def load_curve(start):
    tk = {"v1d": "^VIX1D", "v9d": "^VIX9D", "vix": "^VIX", "v3m": "^VIX3M", "vvix": "^VVIX"}
    o, c = defaultdict(dict), defaultdict(dict)
    for key, sym in tk.items():
        h = yf.Ticker(sym).history(start=start, end="2026-07-02")
        for ts, row in h.iterrows():
            d = str(ts.date())
            if row["Open"] > 0:
                o[d][key] = float(row["Open"])
            if row["Close"] > 0:
                c[d][key] = float(row["Close"])
    return o, c


def metrics(day, o, prev_vix_close):
    x = o.get(day, {})
    if not all(k in x for k in ("v1d", "v9d", "vix", "v3m", "vvix")):
        return None
    m = {
        "slope9": x["v9d"] / x["vix"],
        "slope1": x["v1d"] / x["vix"],
        "far": x["vix"] / x["v3m"],
        "vvix": x["vvix"],
        "v1d": x["v1d"],
    }
    m["onchg"] = (x["vix"] / prev_vix_close - 1) * 100 if prev_vix_close else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    args = ap.parse_args()

    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    S = _mk_settings()
    day_pnl = defaultdict(float)
    for t in trades:
        r, _ = sim_trade(t, S)
        day_pnl[t["date"]] += BASE * r / 100

    o, c = load_curve(args.since)
    # prior VIX close per day
    cdays = sorted(c)
    prev = {}
    for i, d in enumerate(cdays):
        prev[d] = c[cdays[i - 1]].get("vix") if i > 0 else None

    rows = []  # (day, pnl, metrics)
    for d in sorted(day_pnl):
        m = metrics(d, o, prev.get(d))
        if m:
            rows.append((d, day_pnl[d], m))
    n = len(rows)
    tot = sum(r[1] for r in rows)
    print(f"\nVIX-curve regime test — {n} trading days ({rows[0][0]}..{rows[-1][0]}), book ${tot:+,.0f}")
    print(f"(backwardation = slope9>1 = near-term stress; contango = slope9<1 = calm)\n")

    def quartile_table(key, label, invert=False):
        vals = sorted(r[2][key] for r in rows)
        q1, q2, q3 = vals[n // 4], vals[n // 2], vals[3 * n // 4]
        buckets = [("Q1 low", lambda v: v < q1), ("Q2", lambda v: q1 <= v < q2),
                   ("Q3", lambda v: q2 <= v < q3), ("Q4 high", lambda v: v >= q3)]
        print(f"  === {label} (cuts {q1:.3f}/{q2:.3f}/{q3:.3f}) ===")
        print(f"  {'bucket':<10}{'days':>5}{'book P&L':>12}{'avg/day':>10}{'win days':>10}{'worst-days in bucket':>22}")
        for bl, cond in buckets:
            bd = [r for r in rows if cond(r[2][key])]
            if not bd:
                continue
            p = sum(r[1] for r in bd)
            wd = sum(1 for r in bd if r[1] > 0)
            nw = sum(1 for r in bd if r[0] in WORST)
            print(f"  {bl:<10}{len(bd):>5}{f'${p:+,.0f}':>12}{f'${p/len(bd):+,.0f}':>10}"
                  f"{f'{wd}/{len(bd)}':>10}{nw:>22}")
        print()

    quartile_table("slope9", "slope9 = VIX9D/VIX  (LOW=contango/calm, HIGH=backwardation/stress)")
    quartile_table("slope1", "slope1 = VIX1D/VIX  (0DTE short-end)")
    quartile_table("far", "far = VIX/VIX3M  (LOW=steep contango/calm)")
    quartile_table("vvix", "vvix level  (HIGH=vol-of-vol elevated, 'move coming')")
    quartile_table("v1d", "v1d = VIX1D level  (absolute 0DTE implied vol)")
    quartile_table("onchg", "onchg = overnight VIX change %  (HIGH=vol expanding into open)")

    # backwardation split (the headline stress flag)
    bw = [r for r in rows if r[2]["slope9"] >= 1.0]
    ct = [r for r in rows if r[2]["slope9"] < 1.0]
    print(f"  === HEADLINE: backwardation vs contango ===")
    for lab, grp in (("backwardation (slope9>=1)", bw), ("contango (slope9<1)", ct)):
        if grp:
            p = sum(r[1] for r in grp)
            print(f"  {lab:<28} {len(grp):>3} days  ${p:+,.0f}  (${p/len(grp):+.0f}/day, "
                  f"{sum(1 for r in grp if r[1]>0)}/{len(grp)} green)")
    print("\nVERDICT: a metric is USABLE only if book P&L separates cleanly across its buckets AND the worst")
    print("days concentrate in one tail. Flat P&L across buckets = no forward signal (refuted, like realized chop).")


if __name__ == "__main__":
    main()
