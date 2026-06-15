"""Phase 3b: high-beta + multi-DTE PUT backtest (the 0DTE left-tail fix).

SPY 0DTE puts had a fat LEFT tail (wipe out to ~-100% on rebounds). This tests whether
HIGH-BETA names (move more on down days) at LONGER DTE (retain value on a rebound instead
of going to zero) shrink that tail. Entry timing still from the SPY down-day classifier
(down_day_classifier_oos.csv); the instrument is the name's near-ATM put. Real premiums +
real V5/V7 ExitFSM. Reports by ticker and DTE bucket.

Offline research. Read-only on journal/thetadata_options.db.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
SIG = ROOT / "journal" / "v3_eval_results" / "down_day_classifier_oos.csv"
ET = ZoneInfo("America/New_York")
TICKERS = ["MSTR", "PLTR", "AMD", "NVDA", "TSLA"]
THRESHOLD = 0.6
EXIT_HAIRCUT = 0.03


class _S:
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0


def _mi(ts):
    return (ts.dt.hour - 9) * 60 + ts.dt.minute - 30


def _load_stock(tk):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, close FROM stock_ohlc WHERE ticker=? ORDER BY timestamp", con, params=(tk,))
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = _mi(ts)
    return {d: g.set_index("mi")["close"].to_dict() for d, g in df.groupby("date")}


def _load_puts(tk):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, expiration, strike, close FROM option_ohlc "
        "WHERE ticker=? AND right='PUT' ORDER BY timestamp", con, params=(tk,))
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = _mi(ts)
    df["dte"] = (pd.to_datetime(df["expiration"]) - pd.to_datetime(df["date"])).dt.days
    return df


def _run(prem_path, mi_path, und_path, ep, ets, cfg, dte):
    fsm = ExitFSM(cfg, settings=_S())
    state = TradeState(trade_id=1, ticker="X", option_type="put", entry_premium=ep,
                       entry_time=ets, contracts=1, peak_premium=ep,
                       entry_underlying_price=und_path[0], dte=dte,
                       expiry_date=ets.strftime("%Y-%m-%d"))
    last_valid = ep
    for k in range(1, len(prem_path)):
        prem = prem_path[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last_valid = prem
        now = ets + timedelta(minutes=int(mi_path[k] - mi_path[0]))
        mtc = max(0, (16 * 60) - (now.hour * 60 + now.minute))
        action = fsm.evaluate(state, prem, prem * (1 - EXIT_HAIRCUT), prem, now,
                              current_underlying=und_path[k], minutes_to_close=mtc, candle_data={})
        if action.should_exit:
            return (prem * (1 - EXIT_HAIRCUT) - ep) / ep * 100
    return (last_valid * (1 - EXIT_HAIRCUT) - ep) / ep * 100


def main():
    sig = pd.read_csv(SIG)[["date", "entry_min", "p"]]
    fires = {d: int(g.sort_values("entry_min").iloc[0]["entry_min"])
             for d, g in sig[sig["p"] >= THRESHOLD].groupby("date")}
    print(f"SPY signal fire-days (p>={THRESHOLD}): {len(fires)}\n")
    print(f"{'ticker':<7}{'dte_bucket':<11}{'trades':>7}{'mean%':>8}{'med%':>7}{'win%':>6}{'p10%':>7}{'total%':>8}")
    allrows = []
    for tk in TICKERS:
        stock = _load_stock(tk)
        puts = _load_puts(tk)
        cfg = apply_v7_wide_trail_exits(get_ticker_config(tk, use_per_ticker=True, option_type="put"), is_put=True)
        for d, em in fires.items():
            if d not in stock or em not in stock[d]:
                continue
            spot = stock[d][em]
            pday = puts[(puts["date"] == d) & (puts["mi"] == em)]
            if pday.empty:
                continue
            # nearest-expiry, then ATM
            dte0 = pday["dte"].min()
            avail = pday[pday["dte"] == dte0].assign(dist=(pday["strike"] - spot).abs()).sort_values("dist")
            strike = avail.iloc[0]["strike"]
            chain = puts[(puts["date"] == d) & (puts["strike"] == strike) & (puts["dte"] == dte0)]
            chain = chain[chain["mi"] >= em].sort_values("mi")
            if len(chain) < 5:
                continue
            pp = chain["close"].values.astype(float)
            mp = chain["mi"].values.astype(int)
            up = [stock[d].get(int(m), spot) for m in mp]
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
            r = _run(pp, mp, up, pp[0], ets, cfg, int(dte0))
            allrows.append({"tk": tk, "dte": int(dte0), "ret": r})
    df = pd.DataFrame(allrows)
    if df.empty:
        print("no trades")
        return
    df["bucket"] = pd.cut(df["dte"], [-1, 0, 2, 99], labels=["0DTE", "1-2DTE", "3+DTE"])
    for tk in TICKERS:
        for b in ["0DTE", "1-2DTE", "3+DTE"]:
            g = df[(df["tk"] == tk) & (df["bucket"] == b)]
            if len(g) >= 5:
                r = g["ret"].values
                print(f"{tk:<7}{b:<11}{len(r):>7}{r.mean():>7.1f}{np.median(r):>7.1f}"
                      f"{np.mean(r>0)*100:>5.0f}{np.percentile(r,10):>7.0f}{r.sum():>8.0f}")
    print("\n=== ALL high-beta combined, by DTE bucket ===")
    for b in ["0DTE", "1-2DTE", "3+DTE"]:
        g = df[df["bucket"] == b]
        if len(g):
            r = g["ret"].values
            print(f"  {b:<8} n={len(r):<4} mean={r.mean():+5.1f}%  med={np.median(r):+5.1f}%  "
                  f"win={np.mean(r>0)*100:3.0f}%  p10(left tail)={np.percentile(r,10):+5.0f}%  total={r.sum():+6.0f}%")


if __name__ == "__main__":
    main()
