"""DECISIVE alpha-vs-beta test of the whale (UW flow) signal in a SLOWER, fill-tax-free
expression: buy the UNDERLYING STOCK on each qualifying long-dated whale sweep and hold N days.

The whole point is the beta subtraction the prior fast-option tests skipped:
  For each trade compute the whale-following return AND two benchmarks over the SAME horizon:
    (a) SPY-beta      — buy/short SPY over the trade's actual [entry, entry+N] window.
    (b) name-beta     — the RANDOM-timed return of owning the SAME ticker for N days
                        (mean over ALL overlapping N-day windows in the 12mo sample).
  alpha_vs_spy  = whale_ret - spy_beta   (selection over index beta)
  alpha_vs_name = whale_ret - name_beta  (TIMING selection over "these names drifted up")
Whale-following must beat BOTH to be alpha. If whale ~= name-beta it is just "these names went
up" = NOT tradeable alpha.

Directional convention: CALL/bullish = long stock; PUT/bearish = short stock. Benchmarks take the
same sign (short SPY / short random-name for bearish) so the spread is apples-to-apples.

Significance: paired t-test (whale vs name-beta, per-trade) + bootstrap CI on mean alpha.
Splits: direction, holding period N, monthly regime (up/chop/down), name-liquidity, per-ticker
concentration + leave-one-ticker-out.

Read-only, self-contained. Uses journal/longdated_flow_options.db (already downloaded).

    python scripts/whale_stock_swing_alpha.py
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "longdated_flow_options.db"
HOLDS = [1, 3, 5, 10]
RNG = np.random.default_rng(42)


def load_stock_bars(c):
    """ticker -> (dates:list[str], close:np.array, dvol:np.array)."""
    rows = c.execute(
        "SELECT ticker, substr(timestamp,1,10) d, close, volume FROM stock_ohlc "
        "WHERE close>0 ORDER BY ticker, timestamp").fetchall()
    bars = defaultdict(list)
    for tk, d, cl, vol in rows:
        bars[tk].append((d, float(cl), float(cl) * float(vol or 0)))
    out = {}
    for tk, lst in bars.items():
        dates = [x[0] for x in lst]
        close = np.array([x[1] for x in lst])
        dvol = np.array([x[2] for x in lst])
        out[tk] = (dates, close, dvol)
    return out


def name_beta_table(bars, N):
    """(ticker,N) -> mean directional-agnostic N-day fwd return over ALL windows (long convention).
    For bearish trades we negate at use-time. Returns dict ticker -> mean_ret (fraction)."""
    tbl = {}
    for tk, (dates, close, _) in bars.items():
        if len(close) <= N:
            continue
        r = close[N:] / close[:-N] - 1.0
        if len(r):
            tbl[tk] = float(r.mean())
    return tbl


def date_index(dates):
    return {d: i for i, d in enumerate(dates)}


def main():
    c = sqlite3.connect(str(DB))
    sigs = c.execute(
        "SELECT ticker,right,entry_date,total_premium,dte FROM signals").fetchall()
    bars = load_stock_bars(c)
    c.close()

    if "SPY" not in bars:
        print("No SPY bars — abort"); sys.exit(1)
    spy_dates, spy_close, _ = bars["SPY"]
    spy_idx = date_index(spy_dates)

    # avg dollar-volume per ticker (liquidity)
    dvol_avg = {tk: float(np.mean(dv)) for tk, (_, _, dv) in bars.items()}

    # precompute name-beta tables per N
    nbeta = {N: name_beta_table(bars, N) for N in HOLDS}
    # per-ticker index maps
    idxmap = {tk: date_index(d) for tk, (d, _, _) in bars.items()}

    # Build per-trade records for each N
    # record: dict(tk, right, N, entry_date, month, whale, spy, name, dvol)
    records = {N: [] for N in HOLDS}
    skipped = 0
    for tk, right, d0, prem, dte in sigs:
        if tk not in bars:
            skipped += 1; continue
        dates, close, _ = bars[tk]
        imap = idxmap[tk]
        # first bar on/after d0
        d0 = d0[:10]
        ei = next((imap[dd] for dd in dates if dd >= d0), None)
        if ei is None:
            skipped += 1; continue
        entry_date = dates[ei]
        # SPY entry index (same date or next available)
        si = spy_idx.get(entry_date)
        if si is None:
            si = next((spy_idx[dd] for dd in spy_dates if dd >= entry_date), None)
        bull = (right == "CALL")
        sign = 1.0 if bull else -1.0
        for N in HOLDS:
            if ei + N >= len(close):
                continue
            if si is None or si + N >= len(spy_close):
                continue
            whale = sign * (close[ei + N] / close[ei] - 1.0)
            spy = sign * (spy_close[si + N] / spy_close[si] - 1.0)
            nb = nbeta[N].get(tk)
            if nb is None:
                continue
            name = sign * nb
            records[N].append({
                "tk": tk, "right": right, "N": N, "entry_date": entry_date,
                "month": entry_date[:7], "whale": whale, "spy": spy, "name": name,
                "dvol": dvol_avg.get(tk, 0.0),
            })
    print(f"Skipped {skipped} signals (no stock bars / no forward window)\n")
    return records, bars, spy_dates, spy_close, spy_idx


# ---------- stats helpers ----------
def agg(rets):
    rets = np.asarray(rets, float)
    if len(rets) == 0:
        return dict(n=0)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    gp = wins.sum(); gl = -losses.sum()
    pf = gp / gl if gl > 0 else float("inf")
    return dict(n=len(rets), mean=rets.mean() * 100, med=float(np.median(rets)) * 100,
                wr=100 * len(wins) / len(rets), pf=pf)


def paired_t(a, b):
    from scipy import stats
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if len(d) < 3 or d.std() == 0:
        return float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def bootstrap_ci(diff, nboot=10000):
    diff = np.asarray(diff, float)
    if len(diff) < 3:
        return float("nan"), float("nan")
    n = len(diff)
    means = np.empty(nboot)
    for i in range(nboot):
        means[i] = diff[RNG.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)) * 100, float(np.percentile(means, 97.5)) * 100


def line(label, a):
    if a["n"] == 0:
        return f"   {label:<14} n=0"
    pf = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
    return (f"   {label:<14} n={a['n']:<5} mean={a['mean']:+6.2f}% med={a['med']:+6.2f}% "
            f"WR={a['wr']:4.1f}% PF={pf}")


def report_block(recs, title):
    """recs = list of trade dicts. Prints whale/spy/name + alpha spreads + significance."""
    if not recs:
        print(f"\n### {title}: (no trades)"); return
    whale = [r["whale"] for r in recs]
    spy = [r["spy"] for r in recs]
    name = [r["name"] for r in recs]
    a_spy = np.array(whale) - np.array(spy)
    a_name = np.array(whale) - np.array(name)
    print(f"\n### {title}  (n={len(recs)})")
    print(line("WHALE", agg(whale)))
    print(line("SPY-beta", agg(spy)))
    print(line("name-beta", agg(name)))
    t_n, p_n = paired_t(whale, name)
    t_s, p_s = paired_t(whale, spy)
    lo_n, hi_n = bootstrap_ci(a_name)
    lo_s, hi_s = bootstrap_ci(a_spy)
    beat_name = 100 * np.mean(a_name > 0)
    print(f"   alpha vs name : {a_name.mean()*100:+.2f}%/tr  (95%CI [{lo_n:+.2f},{hi_n:+.2f}]  "
          f"t={t_n:+.2f} p={p_n:.4f}  beat-name={beat_name:.0f}%)")
    print(f"   alpha vs SPY  : {a_spy.mean()*100:+.2f}%/tr  (95%CI [{lo_s:+.2f},{hi_s:+.2f}]  "
          f"t={t_s:+.2f} p={p_s:.4f})")


if __name__ == "__main__":
    records, bars, spy_dates, spy_close, spy_idx = main()

    print("=" * 78)
    print("STOCK-SWING WHALE-FOLLOWING: ALPHA vs BETA  (12mo, 2025-07 → 2026-07)")
    print("Long stock on CALL/bullish sweep, short stock on PUT/bearish sweep.")
    print("name-beta = random-timed (all-window mean) N-day return of the SAME ticker.")
    print("=" * 78)

    # ---- 1) by holding period, ALL ----
    for N in HOLDS:
        report_block(records[N], f"HOLD {N}d — ALL")

    # ---- 2) by direction x N ----
    print("\n" + "=" * 78)
    print("BY DIRECTION")
    print("=" * 78)
    for N in HOLDS:
        for side in ("CALL", "PUT"):
            recs = [r for r in records[N] if r["right"] == side]
            report_block(recs, f"HOLD {N}d — {side}")

    # ---- 3) regime (monthly SPY trend) ----
    print("\n" + "=" * 78)
    print("BY REGIME (entry-month SPY return: up>+1%, down<-1%, else chop)")
    print("=" * 78)
    # monthly SPY return
    mrows = defaultdict(list)
    for d, cl in zip(spy_dates, spy_close):
        mrows[d[:7]].append(cl)
    mreg = {}
    for m, cls in mrows.items():
        rr = cls[-1] / cls[0] - 1
        mreg[m] = "up" if rr > 0.01 else ("down" if rr < -0.01 else "chop")
    print("month regimes:", {m: mreg[m] for m in sorted(mreg)})
    for reg in ("up", "chop", "down"):
        recs = [r for r in records[5] if mreg.get(r["month"]) == reg]  # focus N=5
        report_block(recs, f"HOLD 5d — regime={reg}")

    # ---- 4) name-liquidity tertiles (N=5) ----
    print("\n" + "=" * 78)
    print("BY NAME-LIQUIDITY (avg $volume tertile, N=5d)")
    print("=" * 78)
    recs5 = records[5]
    if recs5:
        dvs = sorted(set(r["dvol"] for r in recs5))
        q1 = np.percentile([r["dvol"] for r in recs5], 33)
        q2 = np.percentile([r["dvol"] for r in recs5], 67)
        for lab, lo, hi in [("low-liq", -1, q1), ("mid-liq", q1, q2), ("high-liq", q2, 1e30)]:
            recs = [r for r in recs5 if lo < r["dvol"] <= hi]
            report_block(recs, f"HOLD 5d — {lab}")

    # ---- 5) per-ticker concentration + leave-one-out (CALLs, N=5) ----
    print("\n" + "=" * 78)
    print("CONCENTRATION — top tickers by alpha-vs-name contribution (CALLs, N=5d)")
    print("=" * 78)
    call5 = [r for r in records[5] if r["right"] == "CALL"]
    by_tk = defaultdict(list)
    for r in call5:
        by_tk[r["tk"]].append(r["whale"] - r["name"])
    contrib = [(tk, np.sum(v), len(v), np.mean(v) * 100) for tk, v in by_tk.items()]
    contrib.sort(key=lambda x: -x[1])
    print("  ticker   sum_alpha(frac)  n   mean_alpha%")
    for tk, s, n, m in contrib[:12]:
        print(f"  {tk:<7} {s:+.3f}          {n:<4} {m:+.2f}%")
    print("  ...")
    for tk, s, n, m in contrib[-5:]:
        print(f"  {tk:<7} {s:+.3f}          {n:<4} {m:+.2f}%")
    # leave-one-out: overall alpha-vs-name mean with top ticker removed
    alla = np.array([r["whale"] - r["name"] for r in call5])
    print(f"\n  CALL N=5 overall alpha-vs-name: {alla.mean()*100:+.3f}%/tr (n={len(call5)})")
    if contrib:
        top = contrib[0][0]
        rest = np.array([r["whale"] - r["name"] for r in call5 if r["tk"] != top])
        print(f"  remove top ticker ({top}): {rest.mean()*100:+.3f}%/tr (n={len(rest)})")
        # remove top 3
        top3 = set(x[0] for x in contrib[:3])
        rest3 = np.array([r["whale"] - r["name"] for r in call5 if r["tk"] not in top3])
        print(f"  remove top 3 ({sorted(top3)}): {rest3.mean()*100:+.3f}%/tr (n={len(rest3)})")
