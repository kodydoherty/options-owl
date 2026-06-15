"""Backtest: whale's ACTUAL (multi-day) contract vs nearest-DTE, same alerts, same V7 exits.

Answers "does following the whale's real far-dated expiry have edge, or should flow normalize to
nearest-DTE?" For each historical whale sweep it runs the V7 ExitFSM on BOTH:
  • MULTI-DAY: the whale's actual ticker/expiry/strike (journal/multiday_flow_options.db, from
    scripts/download_multiday_flow.py) with the real DTE.
  • NEAREST-DTE: ATM nearest-expiry (journal/thetadata_options.db) — the deployed/backtested path.
Reports per-side + per-ticker mean%/win%/PF for each, and a DTE-bucket breakdown. Read-only.

Run AFTER the downloader finishes: python scripts/backtest_multiday_flow.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MDB = ROOT / "journal" / "multiday_flow_options.db"
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL


def _md_stock(tk):
    con = sqlite3.connect(str(MDB))
    s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=? ORDER BY timestamp",
                          con, params=(tk,))
    con.close()
    if s.empty:
        return s
    s["et"] = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(D.ET)
    return s


def _md_opt(tk, exp, strike, right):
    con = sqlite3.connect(str(MDB))
    p = pd.read_sql_query(
        "SELECT timestamp, close FROM option_ohlc WHERE ticker=? AND expiration=? AND strike=? "
        "AND right=? ORDER BY timestamp", con, params=(tk, exp, float(strike), right))
    con.close()
    if p.empty:
        return p
    p["et"] = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(D.ET)
    return p


def _sim_multiday(opt, stk, entry_et, cfg, dte, otype):
    """V7 sim on the whale's real contract. mp = absolute epoch-minutes so _sim's `now` tracks
    real wall-clock across days; mtc is minutes-to-4pm of the current day."""
    o = opt[opt["et"] >= entry_et].copy()
    o = o[(o["close"].notna()) & (o["close"] > 0)]
    if len(o) < 5:
        return None
    # underlying aligned to each option bar (nearest prior stock minute)
    merged = pd.merge_asof(o[["et", "close"]].sort_values("et"),
                           stk[["et", "close"]].rename(columns={"close": "ucl"}).sort_values("et"),
                           on="et", direction="nearest")
    pp = merged["close"].to_numpy(dtype=float)
    up = merged["ucl"].to_numpy(dtype=float)
    if np.isnan(pp[0]) or pp[0] <= 0 or np.isnan(up[0]):
        return None
    mp = (merged["et"].astype("int64") // 60_000_000_000).to_numpy()  # epoch-minutes
    ets = merged["et"].iloc[0].to_pydatetime()
    return D._sim(pp, list(mp), list(up), pp[0], ets, cfg, int(dte), otype)


def _nearest_dte(tk, right, otype, is_put, d, em, stock, opts, cfg):
    """Gold-standard nearest-DTE ATM sim from thetadata (the deployed path)."""
    if d not in stock or em not in stock[d]:
        return None
    spot = stock[d][em]
    oday = opts[(opts["date"] == d) & (opts["mi"] == em)]
    if oday.empty:
        return None
    dte0 = oday["dte"].min()
    av = oday[oday["dte"] == dte0].assign(dist=(oday["strike"] - spot).abs()).sort_values("dist")
    strike = av.iloc[0]["strike"]
    ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
    ch = ch[ch["mi"] >= em].sort_values("mi")
    if len(ch) < 5:
        return None
    pp = ch["close"].values.astype(float)
    mp = ch["mi"].values.astype(int)
    up = [stock[d].get(int(m), spot) for m in mp]
    if np.isnan(pp[0]) or pp[0] <= 0:
        return None
    ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
    return D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype), int(dte0)


def _agg(rows, label):
    a = np.array([r for r in rows if r is not None])
    if len(a) == 0:
        return f"{label:<14} n=0"
    g = a[a > 0].sum(); l = -a[a < 0].sum()
    pf = "inf" if l == 0 else f"{g / l:.2f}"
    return (f"{label:<14} n={len(a):>4}  mean={a.mean():>+6.1f}%  win={np.mean(a > 0) * 100:>3.0f}%  "
            f"total={a.sum():>+7.0f}%  PF={pf}")


def run_side(is_put):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    wl = PUT_UNIV if is_put else CALL_UNIV
    D.UNIVERSE = list(wl)
    sig = D.fetch_sweeps(is_put)
    if sig.empty:
        return
    sig = sig[sig["ticker"].isin(wl)]
    md_all, nd_all = [], []
    by_tk: dict = {}
    by_dte: dict = {}
    for tk in sorted(sig["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg_nd = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        mstk = _md_stock(tk)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em, exp, strike = ev["date"], int(ev["mb"]), str(ev["expiry"]), float(ev["strike"])
            dte = (datetime.strptime(exp[:10], "%Y-%m-%d") - datetime.strptime(d, "%Y-%m-%d")).days
            # nearest-DTE
            nd = _nearest_dte(tk, right, otype, is_put, d, em, stock, opts, cfg_nd)
            nd_ret = nd[0] if nd else None
            # multi-day (whale's real contract) — its own DTE drives the exit config
            md_ret = None
            if not mstk.empty:
                opt = _md_opt(tk, exp, strike, right)
                if not opt.empty:
                    entry_et = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
                    cfg_md = D.apply_v7_wide_trail_exits(
                        D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
                    md_ret = _sim_multiday(opt, mstk, entry_et, cfg_md, dte, otype)
            if nd_ret is not None:
                nd_all.append(nd_ret)
            if md_ret is not None:
                md_all.append(md_ret)
                by_tk.setdefault(tk, {"md": [], "nd": []})["md"].append(md_ret)
                if nd_ret is not None:
                    by_tk[tk]["nd"].append(nd_ret)
                bucket = "0-1" if dte <= 1 else "2-7" if dte <= 7 else "8-30" if dte <= 30 else "31+"
                by_dte.setdefault(bucket, []).append(md_ret)

    print(f"\n================ {otype.upper()} sweeps ================")
    print(_agg(nd_all, "NEAREST-DTE"))
    print(_agg(md_all, "MULTI-DAY"))
    print("  -- multi-day by whale DTE bucket --")
    for b in ("0-1", "2-7", "8-30", "31+"):
        if b in by_dte:
            print("   " + _agg(by_dte[b], f"DTE {b}"))
    print("  -- per-ticker (multi-day vs nearest, n>=8) --")
    for tk, v in sorted(by_tk.items()):
        if len(v["md"]) >= 8:
            print(f"   {tk:<6} " + _agg(v["md"], "MD").replace("MD", "") + "  |  vs ND " + _agg(v["nd"], "").strip())


def main():
    if not MDB.exists():
        print(f"{MDB} not found — run scripts/download_multiday_flow.py first."); return
    run_side(True)
    run_side(False)
    print("\nVerdict: if MULTI-DAY PF/mean clearly beats NEAREST-DTE (esp. in 8-30/31+ buckets), "
          "multi-day flow is worth a tuned exit path; if it's worse/flat, normalize-to-nearest "
          "(already deployed) is correct.")


if __name__ == "__main__":
    main()
