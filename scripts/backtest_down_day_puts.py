"""Phase 3: REAL PUT backtest of confirm-then-commit on the down-day signal.

Fires SPY 0DTE PUTs when the mid-day classifier (down_day_classifier_oos.csv, leak-free
walk-forward) crosses a threshold, using REAL SPY PUT premiums (thetadata option_ohlc) and
the REAL V5/V7 ExitFSM (cut losers hard, ride winners with the no-ceiling trail). Answers:
does the convex asymmetry actually net positive on real premiums, not just the 6x proxy?

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
EXIT_HAIRCUT = 0.03   # sell at close*(1-haircut) — no bid/ask in option_ohlc


class _S:  # minimal settings for the FSM (V7 PUT: scaleout/2pm off, ratchet on)
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0


def _load_spy_underlying():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, close FROM stock_ohlc WHERE ticker='SPY' ORDER BY timestamp", con)
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


def _load_spy_0dte_puts():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, expiration, strike, close FROM option_ohlc "
        "WHERE ticker='SPY' AND right='PUT' ORDER BY timestamp", con)
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["expiration"] == df["date"]]   # 0DTE only (max gamma on crash days)
    return df


def _run_one(prem_path, mi_path, und_path, entry_prem, entry_ts, cfg):
    fsm = ExitFSM(cfg, settings=_S())
    state = TradeState(trade_id=1, ticker="SPY", option_type="put",
                       entry_premium=entry_prem, entry_time=entry_ts, contracts=1,
                       peak_premium=entry_prem, entry_underlying_price=und_path[0],
                       dte=0, expiry_date=entry_ts.strftime("%Y-%m-%d"))
    for k in range(1, len(prem_path)):
        prem = prem_path[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        now = entry_ts + timedelta(minutes=int(mi_path[k] - mi_path[0]))
        mtc = max(0, (16 * 60) - (now.hour * 60 + now.minute))
        bid = prem * (1 - EXIT_HAIRCUT)
        action = fsm.evaluate(state, prem, bid, prem, now,
                              current_underlying=und_path[k], minutes_to_close=mtc,
                              candle_data={})
        if action.should_exit:
            return (bid - entry_prem) / entry_prem * 100  # % return on premium
    # EOD at bid
    last = prem_path[-1] * (1 - EXIT_HAIRCUT)
    return (last - entry_prem) / entry_prem * 100


def main():
    sig = pd.read_csv(SIG)[["date", "entry_min", "p"]]
    und = _load_spy_underlying()
    puts = _load_spy_0dte_puts()
    und_by_day = {d: g.set_index("mi")["close"].to_dict() for d, g in und.groupby("date")}

    cfg_v6 = get_ticker_config("SPY", use_per_ticker=True, option_type="put")
    cfg_v7 = apply_v7_wide_trail_exits(cfg_v6, is_put=True)

    print(f"Signal days: {sig['date'].nunique()} | SPY 0DTE PUT days: {puts['date'].nunique()}")
    for thr in [0.5, 0.6, 0.7]:
        for name, cfg in [("V6put", cfg_v6), ("V7put", cfg_v7)]:
            rets = []
            for d, g in sig[sig["p"] >= thr].groupby("date"):
                em = int(g.sort_values("entry_min").iloc[0]["entry_min"])  # first fire
                pday = puts[puts["date"] == d]
                if pday.empty or d not in und_by_day or em not in und_by_day[d]:
                    continue
                spx = und_by_day[d][em]
                # ATM 0DTE put: strike nearest SPY, with a premium bar at entry minute
                avail = pday[pday["mi"] == em]
                if avail.empty:
                    continue
                avail = avail.assign(dist=(avail["strike"] - spx).abs()).sort_values("dist")
                strike = avail.iloc[0]["strike"]
                chain = pday[pday["strike"] == strike].sort_values("mi")
                chain = chain[chain["mi"] >= em]
                if len(chain) < 5:
                    continue
                prem_path = chain["close"].values.astype(float)
                mi_path = chain["mi"].values.astype(int)
                und_path = [und_by_day[d].get(int(m), spx) for m in mi_path]
                ep = prem_path[0]
                if ep <= 0:
                    continue
                ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
                rets.append(_run_one(prem_path, mi_path, und_path, ep, ets, cfg))
            if rets:
                r = np.array(rets)
                print(f"  thr>={thr} {name:<6} trades={len(r):<4} "
                      f"mean={r.mean():+6.1f}%  median={np.median(r):+6.1f}%  "
                      f"win={np.mean(r>0)*100:4.0f}%  total={r.sum():+7.0f}%  "
                      f"p90={np.percentile(r,90):+6.0f}%")


if __name__ == "__main__":
    main()
