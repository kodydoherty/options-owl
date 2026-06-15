"""A/B test: flow-following with ATM strike (gold-standard) vs cheaper OTM strikes.

Same whale ask-side sweeps + same V7 wide-trail exits as the gold standard, but selects the
OTM strike whose ENTRY premium is closest to a target $/share, instead of ATM. Reports
per-config aggregate, PER-TICKER + PER-MONTH consistency, and a per-ticker DEPLOY verdict so
we only flip OTM on names where it consistently beats ATM (MU, e.g., is a net loser OTM).

Read-only. Reuses uw_ticker_discovery (data + V7 sim). Caches fetched sweeps to /tmp for fast
iteration. Run: python scripts/backtest_flow_otm_test.py
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
OTM_TARGETS = [1.0, 2.0]          # $/share targets to test
PRIMARY = "OTM_$2"                # the config used for the deploy verdict
MIN_PREM_FLOOR = 0.10
CACHE = Path("/tmp/flow_otm_sweeps.pkl")


def _series(opts, d, strike, dte0, em, stock, spot):
    ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
    ch = ch[ch["mi"] >= em].sort_values("mi")
    if len(ch) < 5:
        return None
    pp = ch["close"].values.astype(float)
    if np.isnan(pp[0]) or pp[0] <= 0:
        return None
    mp = ch["mi"].values.astype(int)
    up = [stock[d].get(int(m), spot) for m in mp]
    ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
    return pp, mp, up, ets


def _pick_atm(same, spot):
    av = same.assign(dist=(same["strike"] - spot).abs()).sort_values("dist")
    return av.iloc[0]["strike"] if not av.empty else None


def _pick_otm(same, spot, is_put, target):
    side = same[same["strike"] < spot] if is_put else same[same["strike"] > spot]
    side = side[side["close"] >= MIN_PREM_FLOOR]
    if side.empty:
        return None
    side = side.assign(pd=(side["close"] - target).abs()).sort_values("pd")
    return side.iloc[0]["strike"]


def _fetch_cached():
    if CACHE.exists():
        print(f"  (using cached sweeps {CACHE})", flush=True)
        return pickle.loads(CACHE.read_bytes())
    out = {}
    for is_put in (True, False):
        D.UNIVERSE = list(PUT_UNIV | CALL_UNIV)
        out[is_put] = D.fetch_sweeps(is_put)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def run_side(is_put, sig):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    wl = PUT_UNIV if is_put else CALL_UNIV
    sig = sig[sig["ticker"].isin(wl)]
    configs = ["ATM"] + [f"OTM_${t:.0f}" for t in OTM_TARGETS]
    # rows[config] = list of (ret, entry_prem, ticker, month)
    rows = {c: [] for c in configs}
    for tk in sorted(sig["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em, mo = ev["date"], int(ev["mb"]), ev["month"]
            if d not in stock or em not in stock[d]:
                continue
            spot = stock[d][em]
            oday = opts[(opts["date"] == d) & (opts["mi"] == em)]
            if oday.empty:
                continue
            dte0 = oday["dte"].min()
            same = oday[oday["dte"] == dte0]
            picks = {"ATM": _pick_atm(same, spot)}
            for t in OTM_TARGETS:
                picks[f"OTM_${t:.0f}"] = _pick_otm(same, spot, is_put, t)
            for cname, strike in picks.items():
                if strike is None:
                    continue
                ser = _series(opts, d, strike, dte0, em, stock, spot)
                if ser is None:
                    continue
                pp, mp, up, ets = ser
                ret = D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
                rows[cname].append((ret, pp[0], tk, mo))
    return rows


def _stats(arr):
    a = np.array([r[0] for r in arr])
    if len(a) == 0:
        return None
    g = a[a > 0].sum(); l = -a[a < 0].sum()
    pf = g / l if l > 0 else float("inf")
    return {"n": len(a), "mean": a.mean(), "win": np.mean(a > 0) * 100,
            "total": a.sum(), "pf": pf}


def _consistency(arr):
    """Fraction of months (>=2 trades) that were net-positive."""
    by_mo = defaultdict(list)
    for r in arr:
        by_mo[r[3]].append(r[0])
    mbk = [np.mean(v) for v in by_mo.values() if len(v) >= 2]
    return (np.mean([m > 0 for m in mbk]) * 100 if mbk else 0.0), len(mbk)


def _pf(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def _report(label, rows):
    print(f"\n=== {label} — aggregate ===")
    print(f"{'config':<9}{'n':>5}{'mean%':>8}{'win%':>7}{'total%':>9}{'PF':>7}{'avg_$':>8}")
    for c, arr in rows.items():
        s = _stats(arr)
        if not s:
            print(f"{c:<9}  (none)"); continue
        avg = np.mean([r[1] for r in arr]) * 100
        print(f"{c:<9}{s['n']:>5}{s['mean']:>+8.1f}{s['win']:>7.0f}{s['total']:>+9.0f}"
              f"{_pf(s['pf']):>7}{avg:>8.0f}")

    print(f"\n--- per-ticker: ATM vs {PRIMARY} (verdict: flip OTM only if it wins + is consistent) ---")
    print(f"{'tk':<6}{'ATMn':>5}{'ATMpf':>7}{'ATMwin':>7}  {'OTMn':>5}{'OTMpf':>7}{'OTMwin':>7}{'OTMmo+':>7}  verdict")
    tickers = sorted({r[2] for r in rows['ATM']})
    deploy = []
    for tk in tickers:
        atm = _stats([r for r in rows['ATM'] if r[2] == tk])
        otm_arr = [r for r in rows[PRIMARY] if r[2] == tk]
        otm = _stats(otm_arr)
        if not atm or not otm:
            continue
        cons, nmo = _consistency(otm_arr)
        # deploy OTM if: beats ATM PF, PF>=1.3, >=60% months positive, enough sample
        ok = (otm['pf'] > atm['pf'] and otm['pf'] >= 1.3 and cons >= 60 and otm['n'] >= 15 and nmo >= 2)
        verdict = "OTM ✓" if ok else "keep ATM"
        if ok:
            deploy.append(tk)
        print(f"{tk:<6}{atm['n']:>5}{_pf(atm['pf']):>7}{atm['win']:>7.0f}  "
              f"{otm['n']:>5}{_pf(otm['pf']):>7}{otm['win']:>7.0f}{cons:>6.0f}%  {verdict}")
    return deploy


def main():
    print("Flow OTM A/B (per-ticker + per-month) — loading sweeps...", flush=True)
    sweeps = _fetch_cached()
    put_rows = run_side(True, sweeps[True])
    call_rows = run_side(False, sweeps[False])
    put_deploy = _report("PUT sweeps", put_rows)
    call_deploy = _report("CALL sweeps", call_rows)
    print("\n================ DEPLOY VERDICT ================")
    print(f"PUT  tickers where {PRIMARY} beats ATM (consistent): {', '.join(put_deploy) or '(none)'}")
    print(f"CALL tickers where {PRIMARY} beats ATM (consistent): {', '.join(call_deploy) or '(none)'}")
    print("\nNOTE: backtest prices use option `close` + flat 3% haircut — it does NOT model the "
          "wider far-OTM bid/ask spread or thinner OTM liquidity, so real OTM fills will be worse "
          "than shown. Treat per-ticker OTM✓ as a candidate set to PAPER-validate, sized down.")


if __name__ == "__main__":
    main()
