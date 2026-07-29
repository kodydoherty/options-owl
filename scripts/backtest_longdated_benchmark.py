"""Alpha-vs-beta test for the long-dated whale sleeve (2026-07-18): compare the whale's actual CALL
contract against two neutral benchmarks over the SAME signals + same exit rules:
  • ATM = same-ticker ATM call (isolates the whale's strike/entry selection)
  • SPY = SPY ATM call (the beta reference — "just be long the index")

Only signals where ALL THREE have usable daily paths are scored (apples-to-apples). The gap
(whale − SPY) is the selection alpha over index beta. Reads longdated_flow_options.db. Read-only.

    python scripts/backtest_longdated_benchmark.py
"""
from __future__ import annotations

import sqlite3
import statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "longdated_flow_options.db"


def _path_whale(c, tk, right, strike, expiry):
    return c.execute(
        "SELECT substr(timestamp,1,10) d,open,high,low,close FROM option_ohlc "
        "WHERE ticker=? AND right=? AND strike=? AND expiration=? AND close>0 ORDER BY timestamp",
        (tk, right, strike, expiry)).fetchall()


def _path_bench(c, kind, tk, expiry, strike):
    return c.execute(
        "SELECT substr(timestamp,1,10) d,open,high,low,close FROM benchmark_ohlc "
        "WHERE kind=? AND ticker=? AND expiration=? AND strike=? AND close>0 ORDER BY timestamp",
        (kind, tk, expiry, strike)).fetchall()


def _entry_and_path(bars, d0):
    if len(bars) < 2:
        return None, None
    eb = next((b for b in bars if b[0] >= d0[:10]), bars[0])
    e = eb[1] or eb[4]
    if not e or e <= 0:
        return None, None
    path = [b for b in bars if b[0] >= eb[0]]
    return (e, path) if len(path) >= 2 else (None, None)


def _sim(e, path, rule):
    peak = e
    for i, (_, o, hi, lo, cl) in enumerate(path[1:], start=1):
        hi = hi or cl; lo = lo or cl; cl = cl or e
        peak = max(peak, hi)
        up = (hi / e - 1) * 100; dn = (lo / e - 1) * 100; ddp = (cl / peak - 1) * 100
        if rule == "trail_40":
            if ddp <= -40:
                return (cl / e - 1) * 100
        elif rule == "target_50_stop_50":
            if dn <= -50:
                return -50.0
            if up >= 50:
                return 50.0
        elif rule == "time_21d":
            if i >= 21:
                return (cl / e - 1) * 100
    return (path[-1][4] or e) / e * 100 - 100


def _agg(rets):
    if not rets:
        return "n=0"
    w = [r for r in rets if r > 0]; gl = abs(sum(r for r in rets if r <= 0))
    pf = sum(w) / gl if gl > 0 else float("inf")
    return f"n={len(rets):<4} mean={S.mean(rets):+6.1f}% WR={100*len(w)/len(rets):3.0f}% PF={pf:4.2f}"


def main():
    c = sqlite3.connect(str(DB))
    rows = c.execute(
        "SELECT sig_ticker,sig_expiry,sig_strike,entry_date,atm_strike,spy_expiry,spy_strike "
        "FROM benchmark_map").fetchall()
    triples = []
    for tk, exp, wstrike, d0, atm, spyexp, spystr in rows:
        we, wp = _entry_and_path(_path_whale(c, tk, "CALL", wstrike, exp), d0)
        ae, ap = _entry_and_path(_path_bench(c, "ATM", tk, exp, atm), d0) if atm else (None, None)
        se, sp = _entry_and_path(_path_bench(c, "SPY", "SPY", spyexp, spystr), d0) if spystr else (None, None)
        if we and ae and se:   # apples-to-apples: all three present
            triples.append(((we, wp), (ae, ap), (se, sp)))
    c.close()
    print(f"{len(triples)} signals with whale + ATM + SPY paths all present (apples-to-apples)\n")
    if not triples:
        print("No complete triples (benchmark download may still be running)."); return
    for rule in ["trail_40", "target_50_stop_50", "time_21d"]:
        wr = [_sim(w[0], w[1], rule) for w, a, s in triples]
        ar = [_sim(a[0], a[1], rule) for w, a, s in triples]
        sr = [_sim(s[0], s[1], rule) for w, a, s in triples]
        print(f"── {rule} ──")
        print(f"   WHALE       {_agg(wr)}")
        print(f"   ATM (same)  {_agg(ar)}")
        print(f"   SPY (beta)  {_agg(sr)}")
        print(f"   → alpha vs SPY: {S.mean(wr)-S.mean(sr):+.1f}%/trade | vs ATM: {S.mean(wr)-S.mean(ar):+.1f}%/trade")
        print()


if __name__ == "__main__":
    main()
