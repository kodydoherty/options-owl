"""Phase 5: full premium backtest — follow whale ask-side put SWEEPS on the whitelisted
names (TSLA/AMZN/AAPL/META/NVDA, exclude MSTR/AMD), enter the name's near-ATM put, manage
with the real V5/V7 ExitFSM on REAL premiums. Does following the whales net positive?

Signal: Unusual Whales flow (ask-side >=60% + sweep). Premiums: thetadata option_ohlc.
Overlap window ~2026-03-14..06-09. Read-only.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
ET = ZoneInfo("America/New_York")
BASE = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
WHITELIST = ["TSLA", "AMZN", "AAPL", "META", "NVDA"]
MIN_PREM = 250_000
EXIT_HAIRCUT = 0.03


class _S:
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0


def fetch_sweeps():
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(220):   # deep enough to cover the full 90d window at $250k density
        p = {"limit": 200, "is_put": "true", "min_premium": MIN_PREM}
        if older:
            p["older_than"] = older
        r = requests.get(BASE, headers=hdr, params=p, timeout=20)
        if r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < "2026-03-14":
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df = df[(df["type"] == "put") & df["ticker"].isin(WHITELIST)].copy()
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["total_premium"].astype(float).clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"].between(0, 375)]
    df["mb"] = (df["mi"] // 5) * 5
    return df.drop_duplicates(subset=["ticker", "date", "mb"])  # one entry per name/5m


def _stock(tk):
    con = sqlite3.connect(DB)
    s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=?", con, params=(tk,))
    con.close()
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    s["date"] = ts.dt.strftime("%Y-%m-%d")
    s["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return {d: g.set_index("mi")["close"].to_dict() for d, g in s.groupby("date")}


def _puts(tk):
    con = sqlite3.connect(DB)
    p = pd.read_sql_query("SELECT timestamp, expiration, strike, close FROM option_ohlc "
                          "WHERE ticker=? AND right='PUT' ORDER BY timestamp", con, params=(tk,))
    con.close()
    ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(ET)
    p["date"] = ts.dt.strftime("%Y-%m-%d")
    p["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    p["dte"] = (pd.to_datetime(p["expiration"]) - pd.to_datetime(p["date"])).dt.days
    return p


def _sim(pp, mp, up, ep, ets, cfg, dte):
    fsm = ExitFSM(cfg, settings=_S())
    st = TradeState(trade_id=1, ticker="X", option_type="put", entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        act = fsm.evaluate(st, prem, prem * (1 - EXIT_HAIRCUT), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            return (prem * (1 - EXIT_HAIRCUT) - ep) / ep * 100
    return (last * (1 - EXIT_HAIRCUT) - ep) / ep * 100


def main():
    print("Fetching whale put sweeps (whitelist)...", flush=True)
    sig = fetch_sweeps()
    print(f"Sweep entry events: {len(sig)} across {sig['ticker'].nunique()} names "
          f"({sig['date'].min()}..{sig['date'].max()})\n")
    out = {"V6": [], "V7": []}
    for tk in WHITELIST:
        stock, puts = _stock(tk), _puts(tk)
        cfgs = {"V6": get_ticker_config(tk, use_per_ticker=True, option_type="put")}
        cfgs["V7"] = apply_v7_wide_trail_exits(cfgs["V6"], is_put=True)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em = ev["date"], int(ev["mb"])
            if d not in stock or em not in stock[d]:
                continue
            spot = stock[d][em]
            pday = puts[(puts["date"] == d) & (puts["mi"] == em)]
            if pday.empty:
                continue
            dte0 = pday["dte"].min()
            av = pday[pday["dte"] == dte0].assign(dist=(pday["strike"] - spot).abs()).sort_values("dist")
            strike = av.iloc[0]["strike"]
            ch = puts[(puts["date"] == d) & (puts["strike"] == strike) & (puts["dte"] == dte0)]
            ch = ch[ch["mi"] >= em].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].values.astype(float)
            mp = ch["mi"].values.astype(int)
            up = [stock[d].get(int(m), spot) for m in mp]
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
            for v in ("V6", "V7"):
                out[v].append({"tk": tk, "date": d, "ret": _sim(pp, mp, up, pp[0], ets, cfgs[v], int(dte0))})

    # dump V7 flow trades with $ P&L at a $750/trade sleeve (for the unified per-day report)
    ALLOC = 750.0
    fdf = pd.DataFrame(out["V7"])
    if not fdf.empty:
        fdf["pnl"] = (fdf["ret"] / 100.0 * ALLOC).round(2)
        fdf.to_csv(ROOT / "journal" / "v3_eval_results" / "uw_flow_trades.csv", index=False)
        print(f"  flow trades -> uw_flow_trades.csv ({len(fdf)} trades, ${ALLOC:.0f}/trade)")

    for v in ("V6", "V7"):
        df = pd.DataFrame(out[v])
        if df.empty:
            print(f"{v}: no trades"); continue
        r = df["ret"].values
        print(f"=== {v} exits === trades={len(r)}  mean={r.mean():+.1f}%  median={np.median(r):+.1f}%  "
              f"win={np.mean(r>0)*100:.0f}%  total={r.sum():+.0f}%  p10={np.percentile(r,10):+.0f}%  p90={np.percentile(r,90):+.0f}%")
        if v == "V7":
            for tk, g in df.groupby("tk"):
                rr = g["ret"].values
                print(f"    {tk:<6} n={len(rr):<3} mean={rr.mean():+6.1f}%  win={np.mean(rr>0)*100:3.0f}%  total={rr.sum():+6.0f}%")


if __name__ == "__main__":
    main()
