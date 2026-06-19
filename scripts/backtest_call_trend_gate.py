"""CALL trend/regime ENTRY GATE backtest (no-lookahead).

Thesis (from live MFE analysis): losing 0DTE CALLs have ~0 MFE because we buy calls into
chop/weakness. Test pre-entry trend filters that use ONLY stock_ohlc up to the entry minute,
and measure whether they cut the dead-on-arrival losers while retaining runners.

Baseline: ATM 0DTE CALL entry at 10:00 ET (mi=30) on the liquid call names, exited by the
REAL V7 + profit-lock ExitFSM (same as backtest_2yr_regime.py). MFE = peak premium % the trade
reached before exit (tracked here, parallels the live mfe_pnl_pct column).

Gates (all computed from underlying minute bars with mi <= ENTRY_MI only — NO LOOKAHEAD):
  (a) vwap   : underlying close at entry > session VWAP (mi 0..entry, volume-weighted)
  (b) hh     : last 15m momentum — close[entry] > close[entry-15] AND > the prior-15m high
  (c) spy    : SPY itself green-on-day at entry (close[entry] > open of day, mi=0)
  (d) combo  : vwap AND spy (underlying trend + market trend)

Per-YEAR (2024/2025/2026) and per-regime. Reports PF / WR / avg-loss / n, plus how many
winners vs losers each gate removes and the avg MFE of kept vs dropped trades.

Run: cd /Users/kody/dev/options-owl && python scripts/backtest_call_trend_gate.py
Validate first: env VALIDATE=1 (3 tickers, 1 month, prints sanity rows).
Read-only.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
TICKERS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AMD", "AMZN"]
ENTRY_MI = 30  # 10:00 ET
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)

VALIDATE = os.environ.get("VALIDATE") == "1"


def load_0dte_call(tk):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT strike, right, timestamp, close FROM option_ohlc "
        "WHERE ticker=? AND right='CALL' AND expiration = substr(timestamp,1,10) ORDER BY timestamp",
        con, params=(tk,))
    con.close()
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


def load_stock_full(tk):
    """Per-minute OHLCV for the underlying, keyed by date -> DataFrame indexed by mi.

    Returns enough to compute no-lookahead session VWAP and momentum at the entry minute.
    """
    con = sqlite3.connect(DB)
    s = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM stock_ohlc WHERE ticker=?",
        con, params=(tk,))
    con.close()
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    s["date"] = ts.dt.strftime("%Y-%m-%d")
    s["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    out = {}
    for d, g in s.groupby("date"):
        out[d] = g.set_index("mi").sort_index()
    return out


def sim(pp, mp, up, cfg, ets):
    """Run the real V7 FSM. Returns (exit_ret_pct, mfe_pct) where mfe = peak premium % reached."""
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type="call", entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=0,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    peak = ep
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        if pp[k] > peak:
            peak = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k],
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            mfe = (peak * (1 - H) - ep) / ep * 100
            return (pp[k] * (1 - H) - ep) / ep * 100, mfe
    mfe = (peak * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100, mfe


def compute_gates(sf_day, spy_day):
    """No-lookahead gate features at the entry minute. sf_day = underlying full bars for the day,
    spy_day = SPY full bars. Uses ONLY mi <= ENTRY_MI. Returns dict of bool gates or None if data missing."""
    if sf_day is None or ENTRY_MI not in sf_day.index:
        return None
    pre = sf_day[sf_day.index.to_series().between(0, ENTRY_MI)]  # regular-session bars up to entry
    if pre.empty or 0 not in pre.index:
        return None
    entry_close = float(pre.loc[ENTRY_MI, "close"]) if ENTRY_MI in pre.index else None
    if entry_close is None:
        return None
    # (a) session VWAP from open (mi=0) to entry, volume-weighted (typical price)
    tp = (pre["high"] + pre["low"] + pre["close"]) / 3.0
    vol = pre["volume"].astype(float)
    vol_sum = vol.sum()
    if vol_sum <= 0:
        vwap = float(pre["close"].mean())
    else:
        vwap = float((tp * vol).sum() / vol_sum)
    g_vwap = entry_close > vwap

    # (b) higher-highs / rising-closes over last 15m
    g_hh = False
    if (ENTRY_MI - 15) in pre.index:
        c15 = float(pre.loc[ENTRY_MI - 15, "close"])
        prior_window = pre[pre.index.to_series().between(ENTRY_MI - 15, ENTRY_MI - 1)]
        prior_high = float(prior_window["high"].max()) if not prior_window.empty else float("inf")
        g_hh = (entry_close > c15) and (entry_close > prior_high)

    # (c) SPY green-on-day at entry (market trend filter)
    g_spy = False
    if spy_day is not None and 0 in spy_day.index and ENTRY_MI in spy_day.index:
        spy_open = float(spy_day.loc[0, "open"])
        spy_now = float(spy_day.loc[ENTRY_MI, "close"])
        g_spy = spy_now > spy_open

    # (d) combo
    g_combo = g_vwap and g_spy
    return {"vwap": g_vwap, "hh": g_hh, "spy": g_spy, "combo": g_combo}


def regime_of(spy_day):
    """SPY % move open(mi0)->entry, for regime bucketing (same as backtest_2yr_regime)."""
    if spy_day is None or 0 not in spy_day.index or ENTRY_MI not in spy_day.index:
        return None
    o = float(spy_day.loc[0, "open"])
    n = float(spy_day.loc[ENTRY_MI, "close"])
    return (n - o) / o * 100 if o else None


def pf(x):
    x = np.array(x, float)
    if x.size == 0:
        return float("nan")
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else float("inf")


def stats(rets):
    a = np.array(rets, float)
    n = len(a)
    if n == 0:
        return dict(n=0, pf=float("nan"), wr=float("nan"), avgloss=float("nan"), total=0.0)
    losers = a[a < 0]
    return dict(n=n, pf=pf(a), wr=float(np.mean(a > 0)),
                avgloss=float(losers.mean()) if losers.size else 0.0, total=float(a.sum()))


def main():
    tickers = TICKERS[:3] if VALIDATE else TICKERS
    print(f"loading SPY full bars (regime + market gate)...{' [VALIDATE]' if VALIDATE else ''}", flush=True)
    spy_full = load_stock_full("SPY")

    rows = []  # one per trade
    for tk in tickers:
        df = load_0dte_call(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE call data", flush=True)
            continue
        sf = load_stock_full(tk)
        cfg_c = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        nd = 0
        sane_printed = 0
        for date, g in df.groupby("date"):
            yr = date[:4]
            if VALIDATE and not date.startswith("2026-04"):
                continue
            sf_day = sf.get(date)
            spy_day = spy_full.get(date)
            if sf_day is None or ENTRY_MI not in sf_day.index:
                continue
            spot = float(sf_day.loc[ENTRY_MI, "close"])
            reg = regime_of(spy_day)
            if reg is None:
                continue
            gates = compute_gates(sf_day, spy_day)
            if gates is None:
                continue
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            ch = g[(g["strike"] == atm) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].to_numpy(float)
            mp = ch["mi"].to_numpy(int)
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            up = [float(sf_day.loc[int(m), "close"]) if int(m) in sf_day.index else spot for m in mp]
            ets = D.datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
            ret, mfe = sim(pp, list(mp), up, cfg_c, ets)
            rows.append(dict(date=date, yr=yr, tk=tk, ret=ret, mfe=mfe, reg=reg, **gates))
            nd += 1
            if VALIDATE and sane_printed < 5:
                print(f"    SANITY {tk} {date} spot={spot:.2f} atm={atm} entryP={pp[0]:.2f} "
                      f"ret={ret:+.1f}% mfe={mfe:+.1f}% reg={reg:+.2f} "
                      f"vwap={gates['vwap']} hh={gates['hh']} spy={gates['spy']}", flush=True)
        print(f"  {tk}: {nd} call trades", flush=True)

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        print("no trades"); return
    print(f"\ntotal {len(tdf)} call trades, {tdf.date.nunique()} days, "
          f"{tdf.date.min()}..{tdf.date.max()}")

    gate_cols = ["vwap", "hh", "spy", "combo"]

    # ---- Per-year baseline vs each gate ----
    print("\n" + "=" * 78)
    print("PER-YEAR: BASELINE (all calls) vs GATED (keep only when gate True)")
    print("=" * 78)
    for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        b = stats(sub["ret"])
        print(f"\n[{yr}]  n={b['n']}")
        print(f"  {'gate':<10}{'n':>5}{'PF':>7}{'WR':>7}{'avgLoss%':>10}{'total%':>10}"
              f"{'kept%':>7}")
        pf_s = "inf" if b["pf"] == float("inf") else f"{b['pf']:.2f}"
        print(f"  {'BASELINE':<10}{b['n']:>5}{pf_s:>7}{b['wr']*100:>6.0f}%"
              f"{b['avgloss']:>10.1f}{b['total']:>+10.0f}{'100':>7}")
        for gc in gate_cols:
            kept = sub[sub[gc]]
            s = stats(kept["ret"])
            if s["n"] == 0:
                print(f"  {gc:<10}{0:>5}{'--':>7}")
                continue
            pf_s = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
            keptpct = s["n"] / b["n"] * 100 if b["n"] else 0
            flag = " (n<30)" if s["n"] < 30 else ""
            print(f"  {gc:<10}{s['n']:>5}{pf_s:>7}{s['wr']*100:>6.0f}%"
                  f"{s['avgloss']:>10.1f}{s['total']:>+10.0f}{keptpct:>6.0f}%{flag}")

    # ---- Regime split (SPY down vs flat vs up at entry) ----
    print("\n" + "=" * 78)
    print("PER-REGIME (SPY open->entry move): DOWN<-0.1% | FLAT | UP>+0.1%")
    print("=" * 78)
    tdf["regime"] = np.where(tdf["reg"] < -0.1, "DOWN",
                             np.where(tdf["reg"] > 0.1, "UP", "FLAT"))
    for rg in ["DOWN", "FLAT", "UP"]:
        sub = tdf[tdf["regime"] == rg]
        b = stats(sub["ret"])
        print(f"\n[{rg}]  n={b['n']}")
        pf_s = "inf" if b["pf"] == float("inf") else f"{b['pf']:.2f}"
        print(f"  {'gate':<10}{'n':>5}{'PF':>7}{'WR':>7}{'avgLoss%':>10}{'total%':>10}")
        print(f"  {'BASELINE':<10}{b['n']:>5}{pf_s:>7}{b['wr']*100:>6.0f}%"
              f"{b['avgloss']:>10.1f}{b['total']:>+10.0f}")
        for gc in gate_cols:
            s = stats(sub[sub[gc]]["ret"])
            if s["n"] == 0:
                continue
            pf_s = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
            flag = " (n<30)" if s["n"] < 30 else ""
            print(f"  {gc:<10}{s['n']:>5}{pf_s:>7}{s['wr']*100:>6.0f}%"
                  f"{s['avgloss']:>10.1f}{s['total']:>+10.0f}{flag}")

    # ---- KEY: what does each gate DROP? winners vs losers, MFE of kept vs dropped ----
    print("\n" + "=" * 78)
    print("GATE SELECTIVITY: of the trades each gate DROPS, how many were winners vs losers,")
    print("and the avg MFE (peak %) of kept vs dropped. A good gate drops mostly dead losers.")
    print("=" * 78)
    print(f"{'gate':<10}{'dropN':>7}{'dropWin':>9}{'dropLose':>10}{'dropMFE':>9}"
          f"{'keepMFE':>9}{'dropAvgRet':>11}{'keepAvgRet':>11}")
    for gc in gate_cols:
        dropped = tdf[~tdf[gc]]
        kept = tdf[tdf[gc]]
        dn = len(dropped)
        dwin = int((dropped["ret"] > 0).sum())
        dlose = int((dropped["ret"] < 0).sum())
        dmfe = dropped["mfe"].mean() if dn else float("nan")
        kmfe = kept["mfe"].mean() if len(kept) else float("nan")
        dret = dropped["ret"].mean() if dn else float("nan")
        kret = kept["ret"].mean() if len(kept) else float("nan")
        print(f"{gc:<10}{dn:>7}{dwin:>9}{dlose:>10}{dmfe:>+9.1f}{kmfe:>+9.1f}"
              f"{dret:>+11.1f}{kret:>+11.1f}")

    # ---- Dead-on-arrival loser analysis: trades that never reached +10% MFE ----
    print("\n" + "=" * 78)
    print("DEAD-ON-ARRIVAL focus: losers with MFE < +10% (the 'never ran' deadweight).")
    print("How many does each gate remove vs how many real runners (MFE>=40%) it also kills.")
    print("=" * 78)
    dead = tdf[(tdf["ret"] < 0) & (tdf["mfe"] < 10)]
    runners = tdf[tdf["mfe"] >= 40]
    print(f"baseline: {len(dead)} dead losers, {len(runners)} runners (MFE>=40%)")
    print(f"{'gate':<10}{'deadKept':>10}{'deadKilled':>12}{'%deadKill':>11}"
          f"{'runKept':>9}{'runKilled':>11}{'%runKill':>10}")
    for gc in gate_cols:
        dead_killed = int((~dead[gc]).sum())
        dead_kept = int(dead[gc].sum())
        run_killed = int((~runners[gc]).sum())
        run_kept = int(runners[gc].sum())
        pdk = dead_killed / len(dead) * 100 if len(dead) else 0
        prk = run_killed / len(runners) * 100 if len(runners) else 0
        print(f"{gc:<10}{dead_kept:>10}{dead_killed:>12}{pdk:>10.0f}%"
              f"{run_kept:>9}{run_killed:>11}{prk:>9.0f}%")

    out = Path(D.__file__).resolve().parent.parent / "journal" / "v3_eval_results" / "call_trend_gate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out, index=False)
    print(f"\nper-trade results -> {out}")


if __name__ == "__main__":
    main()
