"""B6/C2/C3: conviction TIERING — within the whitelist, do bigger sweeps ($ premium) and more
ask-dominant sweeps (ask_frac) run more? If yes, tier bet size by sweep conviction (Track E).
Restricted to the deployed flow whitelist (CUR_PUT/CUR_CALL) so buckets reflect tradeable signals.
Reuses uw_ticker_discovery internals. Read-only.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402


def fetch(is_put, wl):
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
    df = df[(df["type"] == want) & df["ticker"].isin(wl)].copy()
    df["prem"] = df["total_premium"].astype(float)
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["prem"].clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(D.ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"].between(0, 375)]
    df["mb"] = (df["mi"] // 5) * 5
    return df.drop_duplicates(subset=["ticker", "date", "mb"])


def sim_all(is_put, wl):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    sig = fetch(is_put, wl)
    out = []
    for tk in sorted(sig["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em = ev["date"], int(ev["mb"])
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
            out.append({"prem": ev["prem"], "ask_frac": ev["ask_frac"],
                        "ret": D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)})
    return out


def _report(df, col, edges, labels):
    print(f"\n  by {col}:")
    print(f"    {'bucket':<14}{'n':>5}{'mean%':>8}{'runner%':>9}{'PF':>7}")
    df["_b"] = pd.cut(df[col], bins=edges, labels=labels)
    for b in labels:
        s = df[df["_b"] == b]["ret"].to_numpy()
        if len(s) < 5:
            continue
        pf = s[s > 0].sum() / max(1e-9, -s[s < 0].sum())
        print(f"    {b:<14}{len(s):>5}{s.mean():>+8.1f}{np.mean(s >= 100) * 100:>9.1f}{pf:>7.2f}")


def main():
    rows = sim_all(True, D.CUR_PUT) + sim_all(False, D.CUR_CALL)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results"); return
    print(f"=== CONVICTION TIERING (whitelist only, n={len(df)}) ===")
    _report(df.copy(), "prem", [0, 5e5, 1e6, 1e12], ["$250k-500k", "$500k-1M", "$1M+"])
    _report(df.copy(), "ask_frac", [0.6, 0.7, 0.85, 1.01], ["0.60-0.70", "0.70-0.85", "0.85+"])
    print("\nIf PF/runner rise with premium AND ask_frac → tier bet size by sweep conviction (Track E).")


if __name__ == "__main__":
    main()
