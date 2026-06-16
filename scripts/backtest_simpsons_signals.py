"""Backtest the Simpsons 'distribution' signals through OUR buy/sell strat (V7 exits).

For each ENTER-NOW signal (journal/simpsons/simpsons_signals.csv): download the underlying +
the ATM nearest-DTE option (from the signal minute, Polygon aggs), enter at the signal time, run
the real V7 ExitFSM, measure return. SMALL SAMPLE — sanity check only. Run after pull_simpsons_signals.py.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PKEY = next((l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines()
             if l.startswith("POLYGON_API_KEY=")), "")
SIG = ROOT / "journal" / "simpsons" / "simpsons_signals.csv"
BASE = "https://api.polygon.io"
ET = D.ET


def aggs(client, tick, frm, to, tf="minute"):
    url = f"{BASE}/v2/aggs/ticker/{tick}/range/1/{tf}/{frm}/{to}"
    for a in range(4):
        try:
            r = client.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": PKEY}, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (a + 1)); continue
            return r.json().get("results", []) if r.status_code == 200 else []
        except httpx.HTTPError:
            time.sleep(2 * (a + 1))
    return []


def occ(tk, expiry, strike, is_put):
    return f"O:{tk}{expiry[2:4]}{expiry[5:7]}{expiry[8:10]}{'P' if is_put else 'C'}{int(round(strike*1000)):08d}"


def bars_df(res):
    if not res:
        return pd.DataFrame(columns=["et", "close"])
    df = pd.DataFrame(res)
    df["et"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    return df[["et", "c"]].rename(columns={"c": "close"})


def main():
    client = httpx.Client()
    rows = list(csv.DictReader(open(SIG)))
    out = []
    for s in rows:
        tk, is_put = s["ticker"], s["direction"] == "put"
        otype = "put" if is_put else "call"
        sig_et = pd.to_datetime(s["ts_utc"] + "+00:00").tz_convert(ET)
        d0 = sig_et.strftime("%Y-%m-%d")
        # stock bars for the day + 1 (short-DTE hold)
        end = (sig_et + timedelta(days=2)).strftime("%Y-%m-%d")
        stk = bars_df(aggs(client, tk, d0, end))
        if stk.empty:
            out.append({**s, "ret": None, "note": "no stock data"}); continue
        spot = float(stk.iloc[(stk["et"] - sig_et).abs().argmin()]["close"])
        # resolve ATM nearest-DTE: try signal date then next business days; pick a strike near spot
        atm = round(spot)
        chosen = None
        for dexp in range(0, 6):
            exp = (sig_et + timedelta(days=dexp)).strftime("%Y-%m-%d")
            if datetime.strptime(exp, "%Y-%m-%d").weekday() >= 5:
                continue
            for st in [atm, atm + 1, atm - 1, round(spot / 2.5) * 2.5, round(spot / 5) * 5]:
                ob = bars_df(aggs(client, occ(tk, exp, st, is_put), d0, end))
                ob = ob[ob["et"] >= sig_et]
                if len(ob) >= 5 and ob.iloc[0]["close"] > 0:
                    chosen = (exp, st, ob); break
            if chosen:
                break
            time.sleep(0.05)
        if not chosen:
            out.append({**s, "ret": None, "note": "no option data"}); continue
        exp, st, ob = chosen
        dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.strptime(d0, "%Y-%m-%d")).days
        m = pd.merge_asof(ob.sort_values("et"), stk.rename(columns={"close": "u"}).sort_values("et"),
                          on="et", direction="nearest")
        pp = m["close"].to_numpy(float)
        up = m["u"].to_numpy(float)
        mp = (m["et"].astype("int64") // 60_000_000_000).to_numpy()
        ets = m["et"].iloc[0].to_pydatetime()
        cfg = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        ret = D._sim(pp, list(mp), list(up), pp[0], ets, cfg, int(dte), otype)
        out.append({**s, "ret": round(ret, 1), "strike": st, "exp": exp, "dte": dte, "entry": round(pp[0], 2), "note": "ok"})
        time.sleep(0.1)

    print(f"\n{'date':<11}{'tk':<6}{'dir':<5}{'score':>5}  {'strike':>7}{'dte':>4}{'entry':>7}{'ret%':>8}")
    sims = []
    for r in out:
        if r.get("ret") is not None:
            sims.append(r["ret"])
            print(f"{r['ts_utc'][:10]:<11}{r['ticker']:<6}{r['direction']:<5}{r['score']:>5}  "
                  f"{r.get('strike',0):>7}{r.get('dte',0):>4}{r.get('entry',0):>7.2f}{r['ret']:>+8.1f}")
        else:
            print(f"{r['ts_utc'][:10]:<11}{r['ticker']:<6}{r['direction']:<5}{r['score']:>5}  -- {r['note']}")
    if sims:
        a = np.array(sims)
        g = a[a > 0].sum(); l = -a[a < 0].sum()
        print(f"\nN={len(a)}  mean={a.mean():+.1f}%  win={np.mean(a>0)*100:.0f}%  "
              f"PF={'inf' if l==0 else round(g/l,2)}  total={a.sum():+.0f}%")
    print("\n** 13 signals / 5 days = anecdotal, NOT a validation. Real read needs forward shadow-collection. **")


if __name__ == "__main__":
    main()
