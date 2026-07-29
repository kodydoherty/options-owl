"""Applied VVIX sizing-tilt confirmation test (2026-07-14).

The day-bucket test showed low-VVIX mornings >> high-VVIX for our book. This checks whether that's a
DEPLOYABLE trade-by-trade sizing tilt (like conf_linear) or just a day-bucket artifact. Isolates the tilt
from a FLAT base (BASE x ret_pct) so it's the tilt's value alone, and reports DRAWDOWN (the tell conf_linear
was real was LOWER DD at higher P&L, not just more P&L).

Tilt: size up on low VVIX, down on high VVIX. mult = hi - (hi-lo)*pctile(VVIX_open).
  - in-sample pctile = rank vs ALL book days (upper bound, mild lookahead — how conf_linear was validated)
  - expanding pctile = rank vs PRIOR days only (NO lookahead, deployable). Edge must survive THIS.

Books: flow (sim_trade), ML calls (gold-standard dump ret_pct). Reports flat vs tilt: P&L, PF, maxDD, P&L/DD.

Usage: python scripts/vvix_sizing_tilt.py [--dump <ml_dump.json>] [--since 2026-03-01]
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yfinance as yf  # noqa: E402
from exit_risk_sweep import BASE, CACHE, _mk_settings  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402


def vvix_open(since):
    h = yf.Ticker("^VVIX").history(start=since, end="2026-07-02")
    return {str(ts.date()): float(r["Open"]) for ts, r in h.iterrows() if r["Open"] > 0}


def pctile_maps(days_sorted, vv):
    """Return (insample_pctile, expanding_pctile) dicts keyed by day."""
    vals = sorted(vv[d] for d in days_sorted if d in vv)
    def rank(x, arr):
        lo = sum(1 for v in arr if v < x)
        return lo / max(1, len(arr) - 1) if len(arr) > 1 else 0.5
    insample = {d: rank(vv[d], vals) for d in days_sorted if d in vv}
    expanding, seen = {}, []
    for d in days_sorted:
        if d not in vv:
            continue
        expanding[d] = rank(vv[d], sorted(seen)) if len(seen) >= 5 else 0.5
        seen.append(vv[d])
    return insample, expanding


def mult(p, lo, hi):
    return hi - (hi - lo) * p


def max_dd(curve):
    peak, dd = curve[0] if curve else 0, 0.0
    eq = 0.0
    peak = 0.0
    for v in curve:
        eq += v
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return dd  # negative


def stats(rows):
    """rows: list of (day, entry_min, pnl). Return (total, PF, maxDD, pnl/dd)."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    pnls = [r[2] for r in rows]
    tot = sum(pnls)
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float("inf")
    dd = max_dd(pnls)
    ratio = tot / -dd if dd < 0 else float("inf")
    return tot, pf, dd, ratio


def apply_tilt(base_rows, pmap, lo, hi):
    """base_rows: (day, entry_min, flat_pnl, vvix_day_present_bool). Returns tilted rows."""
    out = []
    for day, mi, flat in base_rows:
        p = pmap.get(day)
        m = mult(p, lo, hi) if p is not None else 1.0
        out.append((day, mi, flat * m))
    return out


def load_flow(since):
    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= since]
    S = _mk_settings()
    rows = []
    for t in trades:
        r, _ = sim_trade(t, S)
        rows.append((t["date"], int(t["mi"]), BASE * r / 100))
    return rows


def load_ml(dump, since):
    if not dump or not Path(dump).exists():
        return []
    rows = []
    for t in json.load(open(dump)):
        d = t["day"]
        if d < since:
            continue
        rp = t.get("ret_pct")
        if rp is None:
            continue
        rows.append((d, int(t.get("minute", 0)), BASE * rp / 100))
    return rows


def report(name, base_rows, vv):
    days = sorted({r[0] for r in base_rows})
    ins, exp = pctile_maps(days, vv)
    ft = stats(base_rows)
    print(f"\n{'='*74}\n{name}: {len(base_rows)} trades, {len(days)} days")
    print(f"{'='*74}")
    print(f"  {'variant':<26}{'P&L':>11}{'PF':>7}{'maxDD':>11}{'P&L/DD':>9}")
    print(f"  {'FLAT (base)':<26}{f'${ft[0]:+,.0f}':>11}{ft[1]:>7.2f}{f'${ft[2]:+,.0f}':>11}{ft[3]:>9.2f}")
    for label, pmap in (("in-sample", ins), ("expanding(no-look)", exp)):
        for lo, hi in ((0.7, 1.3), (0.5, 1.6), (0.4, 1.8)):
            tr = apply_tilt(base_rows, pmap, lo, hi)
            t = stats(tr)
            tag = f"{label} {lo}-{hi}"
            d_pnl = t[0] - ft[0]
            print(f"  {tag:<26}{f'${t[0]:+,.0f}':>11}{t[1]:>7.2f}{f'${t[2]:+,.0f}':>11}{t[3]:>9.2f}"
                  f"   Δ${d_pnl:+,.0f}")
    return ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/private/tmp/claude-501/-Users-kody-dev-options-owl/"
                    "ab4377c1-219f-4050-8390-59ecad2d6e56/scratchpad/ml_calls_dump.json")
    ap.add_argument("--since", default="2026-03-01")
    args = ap.parse_args()

    vv = vvix_open(args.since)
    flow = load_flow(args.since)
    ml = load_ml(args.dump, args.since)

    report("FLOW BOOK", flow, vv)
    if ml:
        report("ML CALL BOOK", ml, vv)
        report("COMBINED (flow + ML)", flow + ml, vv)
    else:
        print("\n(ML dump not found — flow only)")

    print("\nVERDICT: real edge = tilt raises P&L/DD (esp. EXPANDING/no-look) AND lowers maxDD, like")
    print("conf_linear. If expanding washes to ~flat, the day-bucket was an artifact / needs live VVIX feed.")


if __name__ == "__main__":
    main()
