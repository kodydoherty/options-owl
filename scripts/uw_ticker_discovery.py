"""UW ticker DISCOVERY sweep: backtest whale ask-side SWEEPS across ALL 14 thetadata names,
both sides (put sweeps -> expect drop, call sweeps -> expect rise), managed by the real V7
ExitFSM on real premiums. Goal: find which tickers BEYOND the current whitelist are
*consistently* profitable (profitable in >=2 of 3 monthly buckets with enough sample), so we
can widen UW_FLOW_PUT_TICKERS / UW_FLOW_CALL_TICKERS on evidence, not hunch.

Current whitelist:  PUT = META,AMZN,AAPL,TSLA   CALL = TSLA,AAPL,AMD,AVGO,PLTR
Read-only. Signal = UW flow-alerts (ask-side >=60% + sweep, >=$250k). Premiums = thetadata.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict
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
_DEFAULT_UNIVERSE = ["AAPL", "SPY", "GOOGL", "IWM", "MSTR", "AMZN", "AMD", "PLTR",
                     "QQQ", "MSFT", "AVGO", "TSLA", "META", "NVDA",
                     "MU", "SMH", "MRVL", "TSM", "INTC", "ORCL", "QCOM"]  # +new-flow tickers
# Override with UW_DISCOVERY_TICKERS="MU,SMH,..." to test newly-downloaded names.
UNIVERSE = ([t.strip().upper() for t in os.environ["UW_DISCOVERY_TICKERS"].split(",") if t.strip()]
            if os.environ.get("UW_DISCOVERY_TICKERS") else _DEFAULT_UNIVERSE)
# V7 gold-standard flow whitelist (mirrors settings.py, updated 2026-06-13: +MU put; +ORCL/INTC call, -AVGO call)
CUR_PUT = {"META", "AMZN", "AAPL", "TSLA", "MU"}
CUR_CALL = {"META", "SPY", "AMZN", "TSLA", "AMD", "ORCL", "INTC", "ARM", "GOOG", "LRCX"}  # -AAPL/PLTR; +ARM/GOOG/LRCX
MIN_PREM = 250_000
EXIT_HAIRCUT = 0.03
START = "2026-03-14"


class _S:
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0


def fetch_sweeps(is_put: bool):
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(260):
        p = {"limit": 200, "is_put": "true" if is_put else "false", "min_premium": MIN_PREM}
        if older:
            p["older_than"] = older
        r = None
        for attempt in range(5):  # retry on transient timeouts/connection errors
            try:
                r = requests.get(BASE, headers=hdr, params=p, timeout=30)
                break
            except requests.exceptions.RequestException as exc:
                print(f"  fetch retry {attempt + 1}/5 (is_put={is_put}): {type(exc).__name__}")
                time.sleep(2 * (attempt + 1))
        if r is None:
            print(f"  giving up after 5 timeouts (is_put={is_put}) — partial data")
            break
        if r.status_code != 200:
            print(f"  API {r.status_code} (is_put={is_put}) — stopping pagination")
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < START:
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    want = "put" if is_put else "call"
    df = df[(df["type"] == want) & df["ticker"].isin(UNIVERSE)].copy()
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["total_premium"].astype(float).clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["month"] = ts.dt.strftime("%Y-%m")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"].between(0, 375)]
    df["mb"] = (df["mi"] // 5) * 5
    return df.drop_duplicates(subset=["ticker", "date", "mb"])


def _stock(tk):
    con = sqlite3.connect(DB)
    s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=?", con, params=(tk,))
    con.close()
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    s["date"] = ts.dt.strftime("%Y-%m-%d")
    s["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return {d: g.set_index("mi")["close"].to_dict() for d, g in s.groupby("date")}


def _opts(tk, right):
    con = sqlite3.connect(DB)
    p = pd.read_sql_query("SELECT timestamp, expiration, strike, close FROM option_ohlc "
                          "WHERE ticker=? AND right=? ORDER BY timestamp", con, params=(tk, right))
    con.close()
    ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(ET)
    p["date"] = ts.dt.strftime("%Y-%m-%d")
    p["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    p["dte"] = (pd.to_datetime(p["expiration"]) - pd.to_datetime(p["date"])).dt.days
    return p


def _sim(pp, mp, up, ep, ets, cfg, dte, otype):
    fsm = ExitFSM(cfg, settings=_S())
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
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


def run_side(is_put: bool):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    cur = CUR_PUT if is_put else CUR_CALL
    print(f"\nFetching whale {otype} sweeps (all 14 names)...", flush=True)
    sig = fetch_sweeps(is_put)
    if sig.empty:
        print(f"  no {otype} sweeps returned"); return []
    print(f"  {len(sig)} sweep events, {sig['ticker'].nunique()} names, "
          f"{sig['date'].min()}..{sig['date'].max()}")
    results = []
    for tk in sorted(sig["ticker"].unique()):
        stock, opts = _stock(tk), _opts(tk, right)
        cfg = apply_v7_wide_trail_exits(
            get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        rets_by_month = defaultdict(list)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em, mo = ev["date"], int(ev["mb"]), ev["month"]
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
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
            rets_by_month[mo].append(_sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype))
        allr = [r for rs in rets_by_month.values() for r in rs]
        if len(allr) < 3:
            continue
        a = np.array(allr)
        gains = a[a > 0].sum(); losses = -a[a < 0].sum()
        pf = gains / losses if losses > 0 else float("inf")
        # consistency: fraction of monthly buckets (>=2 samples) that were net positive
        mbk = [np.mean(v) for v in rets_by_month.values() if len(v) >= 2]
        consist = np.mean([m > 0 for m in mbk]) if mbk else 0.0
        results.append({"side": otype, "tk": tk, "n": len(a), "mean": a.mean(),
                        "win": np.mean(a > 0) * 100, "total": a.sum(), "pf": pf,
                        "months": len(mbk), "consist": consist * 100,
                        "in_wl": tk in cur})
        for mo, v in rets_by_month.items():
            _MONTHLY.append({"side": otype, "tk": tk, "month": mo, "n": len(v),
                             "mean": float(np.mean(v)), "total": float(np.sum(v))})
    return results


_MONTHLY = []


def main():
    rows = run_side(True) + run_side(False)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results"); return
    df = df.sort_values(["side", "total"], ascending=[True, False])
    out = ROOT / "journal" / "v3_eval_results" / "uw_ticker_discovery.csv"
    df.to_csv(out, index=False)
    if _MONTHLY:
        pd.DataFrame(_MONTHLY).to_csv(
            ROOT / "journal" / "v3_eval_results" / "uw_ticker_discovery_monthly.csv", index=False)
    for side in ("put", "call"):
        s = df[df["side"] == side]
        if s.empty:
            continue
        print(f"\n=== {side.upper()} sweeps — per ticker (sorted by total return %) ===")
        print(f"{'tk':<6}{'WL':<4}{'n':>4}{'mean%':>8}{'win%':>7}{'total%':>9}{'PF':>7}{'mo+%':>7}")
        for _, r in s.iterrows():
            wl = "✓" if r["in_wl"] else " "
            pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
            print(f"{r['tk']:<6}{wl:<4}{r['n']:>4}{r['mean']:>+8.1f}{r['win']:>7.0f}"
                  f"{r['total']:>+9.0f}{pf:>7}{r['consist']:>7.0f}")
        # NEW candidates: not in whitelist, positive total, consistent across months
        nc = s[(~s["in_wl"]) & (s["total"] > 0) & (s["consist"] >= 60) & (s["months"] >= 2)]
        if not nc.empty:
            print(f"  ** NEW {side} candidates (off-whitelist, +total, >=60% months+): "
                  f"{', '.join(nc['tk'])} **")
        # whitelist names that are NOT pulling weight
        drop = s[(s["in_wl"]) & ((s["total"] <= 0) | (s["consist"] < 40))]
        if not drop.empty:
            print(f"  ** REVIEW {side} whitelist (weak/inconsistent): {', '.join(drop['tk'])} **")
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
