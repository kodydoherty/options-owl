"""2.5-YEAR regime call/put-mix test (the fix for the June 'caught long' P&L drop).

Real thetadata (2024-01..2026-06). For each ticker/day, enter the ATM 0DTE call AND put at 10:00 ET,
run the V7+profit-lock exit. Then compare the deployed FIXED budget (call 1.0 / put 0.5) vs a
REGIME-AWARE budget that shifts toward puts when SPY is down on the day and toward calls when SPY is
up. Reports per-YEAR PF + P&L so we see it across regimes (2024/25/26), not 60 cherry-picked days.
Read-only.
"""
from __future__ import annotations

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
ENTRY_MI = 30   # 10:00 ET
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def load_0dte(tk):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT strike, right, timestamp, close FROM option_ohlc "
        "WHERE ticker=? AND expiration = substr(timestamp,1,10) ORDER BY timestamp",
        con, params=(tk,))
    con.close()
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


def sim(pp, mp, up, cfg, otype, ets):
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=0,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k], minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def main():
    print("loading SPY stock for regime + entries...", flush=True)
    spy_stock = D._stock("SPY")
    trades = []   # (date, year, tk, side, ret, regime)
    for tk in TICKERS:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        cfg_c = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        cfg_p = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="put"), is_put=True)
        nd = 0
        for date, g in df.groupby("date"):
            if date not in stock or ENTRY_MI not in stock[date]:
                continue
            spot = stock[date][ENTRY_MI]
            # SPY regime: % move open->entry
            sp = spy_stock.get(date, {})
            if 0 not in sp or ENTRY_MI not in sp:
                continue
            reg = (sp[ENTRY_MI] - sp[0]) / sp[0] * 100
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            yr = date[:4]
            for side, right, cfg in (("call", "CALL", cfg_c), ("put", "PUT", cfg_p)):
                ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].to_numpy(float)
                mp = ch["mi"].to_numpy(int)
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                up = [stock[date].get(int(m), spot) for m in mp]
                ets = D.datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
                ret = sim(pp, list(mp), list(up), cfg, side, ets)
                trades.append((date, yr, tk, side, ret, reg))
                nd += 1
        print(f"  {tk}: {nd} trades", flush=True)

    tdf = pd.DataFrame(trades, columns=["date", "yr", "tk", "side", "ret", "reg"])
    print(f"\ntotal {len(tdf)} trades  ({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)")

    def budgets(mode, side, reg):
        if mode == "fixed":
            return 1.0 if side == "call" else 0.5
        # regime-aware: SPY down -> favor puts; up -> favor calls
        if reg < -0.1:
            return 0.25 if side == "call" else 1.0
        if reg > 0.1:
            return 1.0 if side == "call" else 0.25
        return 0.6

    def pf(x):
        x = np.array(x, float)
        g, l = x[x > 0].sum(), -x[x < 0].sum()
        return (g / l if l > 0 else float("inf"))

    print(f"\n{'year':<8}{'n':>6}   FIXED (call1/put.5)        REGIME-AWARE")
    print(f"{'':8}{'':6}   {'PF':>6}{'P&L(units)':>12}   {'PF':>6}{'P&L(units)':>12}")
    print("-" * 60)
    for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        fx = [r * budgets("fixed", s, g) for r, s, g in zip(sub.ret, sub.side, sub.reg)]
        rg = [r * budgets("regime", s, g) for r, s, g in zip(sub.ret, sub.side, sub.reg)]
        print(f"{yr:<8}{len(sub):>6}   {pf(fx):>6.2f}{sum(fx):>+12.0f}   {pf(rg):>6.2f}{sum(rg):>+12.0f}")
    print("\nREGIME-AWARE WINS if it lifts PF/P&L in the DOWN periods (e.g. a bearish stretch) without")
    print("giving up much in up periods — i.e. it stops the book getting caught long. Per-year = OOS-ish.")


if __name__ == "__main__":
    main()
