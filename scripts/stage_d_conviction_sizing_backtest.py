"""Stage D validation: does CONVICTION SIZING beat FLAT on the flow book?
Applies a conviction multiplier (cluster size + ticker-type-aware premium + ask_frac) to each
historical flow trade and compares vs flat $750 sleeves. Reports BOTH:
  (A) normalized (mean mult = 1.0)  -> pure ALLOCATION skill (same total capital, reallocated)
  (B) raw                           -> actual bet-bigger effect (more capital on high conviction)
Includes SPY puts to test whether conviction GATING makes them viable. Read-only.
Multipliers come from the validated tests (uw_flow_clustering, uw_conviction_tiering, spy_put).
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
# MIRROR: use the EXACT production conviction multiplier so the gold-standard test == live behavior.
from options_owl.risk.exit_v5.config import INDEX_TICKERS  # noqa: E402
from options_owl.risk.vinny_strategy import flow_conviction_mult  # noqa: E402

CLUSTER_WIN = 30
SLEEVE = 750.0
INDEX = set(INDEX_TICKERS)
PUT_UNIV = D.CUR_PUT | {"SPY"}   # test SPY puts under gating
CALL_UNIV = D.CUR_CALL


def conviction_mult(csize, prem, askf, is_index):
    # delegate to the production helper (single source of truth); P(runner) None until serve-wired
    return flow_conviction_mult(csize, prem, askf, is_index, None)[0]


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
    return df[df["mi"].between(0, 375)].sort_values(["ticker", "date", "mi"])


def trades_for(is_put, wl):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    raw = fetch(is_put, wl)
    out = []
    for tk in sorted(raw["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        g_tk = raw[raw["ticker"] == tk]
        for d, g in g_tk.groupby("date"):
            mis = g["mi"].to_numpy()
            seen = set()
            for _, ev in g.iterrows():
                mb = (int(ev["mi"]) // 5) * 5
                if mb in seen or d not in stock or mb not in stock[d]:
                    continue
                seen.add(mb)
                csize = int(np.sum(np.abs(mis - ev["mi"]) <= CLUSTER_WIN))
                spot = stock[d][mb]
                oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
                if oday.empty:
                    continue
                dte0 = oday["dte"].min()
                av = oday[oday["dte"] == dte0].assign(dist=(oday["strike"] - spot).abs()).sort_values("dist")
                strike = av.iloc[0]["strike"]
                ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
                ch = ch[ch["mi"] >= mb].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].values.astype(float)
                mp = ch["mi"].values.astype(int)
                up = [stock[d].get(int(m), spot) for m in mp]
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=mb)
                ret = D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
                m = conviction_mult(csize, ev["prem"], ev["ask_frac"], tk in INDEX)
                out.append({"date": d, "tk": tk, "ret_pct": ret, "mult": m})
    return out


def equity_dd(pnls_by_day):
    eq, peak, dd = 0.0, 0.0, 0.0
    for d in sorted(pnls_by_day):
        eq += pnls_by_day[d]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return dd


def summarize(df, size_col, label):
    pnl = (df["ret_pct"] / 100.0 * df[size_col])
    gains = pnl[pnl > 0].sum(); losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    byday = pnl.groupby(df["date"]).sum().to_dict()
    dd = equity_dd(byday)
    print(f"  {label:<22} P&L ${pnl.sum():>+9,.0f}  PF {pf:>5.2f}  maxDD ${dd:>+8,.0f}  "
          f"avg ${df[size_col].mean():>5.0f}/trade  n={len(df)}")
    return pnl.sum(), pf, dd


def main():
    rows = trades_for(True, PUT_UNIV) + trades_for(False, CALL_UNIV)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no trades"); return
    df["flat"] = SLEEVE
    df["conv_raw"] = SLEEVE * df["mult"]
    df["conv_norm"] = SLEEVE * df["mult"] / df["mult"].mean()  # same total capital, reallocated
    print(f"=== STAGE D — conviction sizing vs flat (n={len(df)}, incl SPY puts under gating) ===")
    f_pnl, f_pf, f_dd = summarize(df, "flat", "FLAT $750")
    n_pnl, n_pf, n_dd = summarize(df, "conv_norm", "CONVICTION (norm)")
    r_pnl, r_pf, r_dd = summarize(df, "conv_raw", "CONVICTION (raw)")
    print("\n  --- verdict (normalized = pure allocation skill, same capital) ---")
    print(f"  PF: flat {f_pf:.2f} -> norm {n_pf:.2f}  ({'BETTER' if n_pf>f_pf else 'worse'})")
    print(f"  P&L: flat ${f_pnl:,.0f} -> norm ${n_pnl:,.0f}  (DD flat ${f_dd:,.0f} -> norm ${n_dd:,.0f})")
    print(f"  raw (bet-bigger): ${r_pnl:,.0f} P&L at PF {r_pf:.2f}, maxDD ${r_dd:,.0f}")
    # SPY-put contribution under gating
    spy = df[(df.tk == "SPY")]
    if len(spy):
        sp = (spy["ret_pct"]/100.0*spy["conv_norm"]).sum()
        print(f"\n  SPY (gated) contribution under conviction sizing: ${sp:+,.0f} over n={len(spy)} "
              f"(avg mult {spy['mult'].mean():.2f})")


if __name__ == "__main__":
    main()
