"""B1: flow CLUSTERING test — do names getting MULTIPLE qualifying whale sweeps in a short window
run MORE than single-sweep names? If yes, cluster size is a bet-bigger signal (Track E).

Method: fetch raw qualifying sweeps (NO 5m dedup), per ticker+date+direction count how many
qualifying sweeps land within CLUSTER_WIN minutes around each 5m entry bar. Bucket entries by
cluster size, simulate each with the real V7 ExitFSM, compare runner-rate / mean / PF / P90.
Read-only. Reuses uw_ticker_discovery internals.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402  (_stock,_opts,_sim,KEY,BASE,ET,UNIVERSE,MIN_PREM)

CLUSTER_WIN = 30  # minutes


def fetch_raw(is_put):
    hdr = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(260):
        p = {"limit": 200, "is_put": "true" if is_put else "false", "min_premium": D.MIN_PREM}
        if older:
            p["older_than"] = older
        r = None
        for a in range(5):
            try:
                r = requests.get(D.BASE, headers=hdr, params=p, timeout=30); break
            except requests.exceptions.RequestException:
                time.sleep(2 * (a + 1))
        if r is None or r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < D.START:
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    want = "put" if is_put else "call"
    df = df[(df["type"] == want) & df["ticker"].isin(D.UNIVERSE)].copy()
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["total_premium"].astype(float).clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(D.ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"].between(0, 375)]
    return df


def run_side(is_put):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    raw = fetch_raw(is_put)
    if raw.empty:
        return []
    # cluster size per (ticker,date): for each event, # qualifying sweeps within +/- CLUSTER_WIN
    raw = raw.sort_values(["ticker", "date", "mi"])
    rows = []
    for (tk, d), g in raw.groupby(["ticker", "date"]):
        mis = g["mi"].to_numpy()
        # one entry per 5m bar (like discovery), tagged with cluster size in the window
        seen = set()
        for mi in mis:
            mb = (mi // 5) * 5
            if mb in seen:
                continue
            seen.add(mb)
            csize = int(np.sum(np.abs(mis - mi) <= CLUSTER_WIN))
            rows.append({"tk": tk, "date": d, "mb": int(mb), "csize": csize})
    ev = pd.DataFrame(rows)

    out = []
    for tk in sorted(ev["tk"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        for _, e in ev[ev["tk"] == tk].iterrows():
            d, em = e["date"], int(e["mb"])
            if d not in stock or em not in stock[d]:
                continue
            spot = stock[d][em]
            oday = opts[(opts["date"] == d) & (opts["mi"] == em)]
            if oday.empty:
                continue
            dte0 = oday["dte"].min()
            av = oday[oday["dte"] == dte0].assign(dist=(oday["strike"] - spot).abs()).sort_values("dist")
            strike = av.iloc[0]["strike"]
            ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
            ch = ch[ch["mi"] >= em].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].values.astype(float)
            mp = ch["mi"].values.astype(int)
            up = [stock[d].get(int(m), spot) for m in mp]
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
            ret = D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
            out.append({"side": otype, "csize": int(e["csize"]), "ret": ret})
    return out


def main():
    allrows = run_side(True) + run_side(False)
    df = pd.DataFrame(allrows)
    if df.empty:
        print("no results"); return
    df["bucket"] = np.where(df["csize"] >= 2, "clustered(>=2)", "single(1)")
    print(f"=== B1 FLOW CLUSTERING (window {CLUSTER_WIN}min) — does cluster size predict runners? ===")
    print(f"{'bucket':<16}{'n':>5}{'mean%':>8}{'runner%':>9}{'win%':>7}{'PF':>7}{'P90%':>7}")
    for b in ("single(1)", "clustered(>=2)"):
        s = df[df["bucket"] == b]["ret"].to_numpy()
        if len(s) == 0:
            continue
        g = s[s > 0].sum(); l = -s[s < 0].sum()
        pf = g / l if l > 0 else float("inf")
        run = np.mean(s >= 100) * 100
        p90 = np.percentile(s, 90)
        print(f"{b:<16}{len(s):>5}{s.mean():>+8.1f}{run:>9.1f}{np.mean(s>0)*100:>7.0f}{pf:>7.2f}{p90:>+7.0f}")
    # finer: by exact cluster size
    print("\n  by exact cluster size:")
    for c in sorted(df["csize"].unique()):
        s = df[df["csize"] == c]["ret"].to_numpy()
        if len(s) < 5:
            continue
        print(f"    size {c}: n={len(s):<4} mean={s.mean():+6.1f}%  runner={np.mean(s>=100)*100:4.1f}%  "
              f"PF={(s[s>0].sum()/max(1e-9,-s[s<0].sum())):.2f}")
    print("\nIf clustered >> single on runner% AND PF → size up clustered flow (Track E).")


if __name__ == "__main__":
    main()
