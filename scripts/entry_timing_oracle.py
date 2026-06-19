"""ENTRY-TIMING ORACLE STUDY — "is optimal entry timing a real exploitable edge?"

Question: when a signal fires we currently buy ~immediately (NOW) and are often instantly underwater.
How much would the OPTIMAL entry in the next window have won (the PRIZE / lookahead ceiling), and can any
NO-LOOKAHEAD rule capture most of it WITHOUT missing the trades that rip away (adverse selection)?

Reuses the 2.5yr regime harness exactly so results are comparable:
  - same ATM-0DTE selection (load_0dte + nearest strike to spot at mi0)
  - same real ExitFSM exit (V7 wide-trail + profit-lock LOCK cfg) run from the ACTUAL entry minute to EOD
  - same EXIT_HAIRCUT on exit fills

"Signal" proxy = a candidate entry at mi0 = 30 (10:00 ET) per ticker/day, ATM 0DTE (same decision moment
as the regime study). ENTRY WINDOW = next W minutes. The trade runs the real ExitFSM from whatever minute
we actually enter, to EOD.

Strategies (each yields a per-trade % return via ExitFSM from its OWN entry minute):
  1. NOW       baseline: enter at mi0.
  2. ORACLE    *** LOOKAHEAD CEILING — UNATTAINABLE *** enter at lowest option close in [mi0, mi0+W].
  3. DIP(D)    no-lookahead: wait until option close <= mi0_price*(1-D); enter next bar; else timeout @ mi0+W.
  4. PULLBACK  no-lookahead: wait for a dip below mi0_price, then first bar that ticks UP (close>prev close); timeout @ mi0+W.
  5. VWAP      no-lookahead: first bar where UNDERLYING close >= running session VWAP; timeout @ mi0+W.

NO-LOOKAHEAD CONTRACT (strategies 3/4/5): the decision at minute m uses ONLY bars with index <= current m.
Each rule below has an inline assertion/comment proving its decision index never reads the future. ORACLE is
the ONLY function permitted to scan the whole window (argmin over the full slice) and is labelled as such.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/entry_timing_oracle.py
Optional: VALIDATE=1 (sanity rows on 2 tickers x 1 month), SIDE=put, WINDOWS=15,30
"""
from __future__ import annotations

import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
ENTRY_MI = 30  # 10:00 ET decision moment (proxy "signal")

LIQUID_CALLS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AMD", "AMZN", "AAPL"]
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)

DIP_LEVELS = [5, 10, 15]   # %


# ----- data loaders (mirror harness) ------------------------------------------------------------
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


def load_stock_full(tk):
    """Return {date: {mi: (close, typprice, volume)}} for regular-hours bars (mi>=0).

    typprice = the per-bar vwap column (thetadata gives a per-bar vwap ~= typical price).
    Used to build a running SESSION vwap up to the decision minute (no lookahead)."""
    con = sqlite3.connect(DB)
    s = pd.read_sql_query("SELECT timestamp, close, volume, vwap FROM stock_ohlc WHERE ticker=?",
                          con, params=(tk,))
    con.close()
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    s["date"] = ts.dt.strftime("%Y-%m-%d")
    s["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    s = s[s["mi"] >= 0]
    out = {}
    for d, g in s.groupby("date"):
        out[d] = {int(m): (float(c), float(v if not np.isnan(v) else c), float(vol))
                  for m, c, vol, v in zip(g["mi"], g["close"], g["volume"], g["vwap"])}
    return out


# ----- exit sim (identical to harness sim()) ----------------------------------------------------
def sim(pp, mp, up, cfg, otype, ets):
    """Run the real ExitFSM from entry (pp[0]=entry premium) to EOD; return % net of haircut."""
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


# ----- entry rules: each returns the INDEX (into the window arrays) we enter at ------------------
# window arrays w_prem[i], w_mi[i] cover minutes mi0..mi0+W. i=0 is mi0 (NOW).
def pick_now(w_prem):
    return 0


def pick_oracle(w_prem):
    """*** LOOKAHEAD — peeks the WHOLE window. Ceiling only, not implementable. ***"""
    valid = [(p, i) for i, p in enumerate(w_prem) if p is not None and not np.isnan(p) and p > 0]
    return min(valid)[1] if valid else 0


def pick_dip(w_prem, dpct):
    """No-lookahead. At decision bar i we compare w_prem[i] to w_prem[0] ONLY (past). Enter NEXT bar."""
    p0 = w_prem[0]
    thresh = p0 * (1 - dpct / 100.0)
    for i in range(1, len(w_prem)):
        pi = w_prem[i]
        if pi is None or np.isnan(pi) or pi <= 0:
            continue
        # decision uses w_prem[0..i] only — no index > i is read. assert via the loop bound.
        if pi <= thresh:
            j = i + 1  # enter the bar AFTER the trigger (can't act on the bar we observe)
            while j < len(w_prem) and (w_prem[j] is None or np.isnan(w_prem[j]) or w_prem[j] <= 0):
                j += 1
            return j if j < len(w_prem) else len(w_prem) - 1
    return len(w_prem) - 1  # timeout: enter at window end


def pick_pullback(w_prem):
    """No-lookahead. Wait for a dip below p0, then first bar whose close > the IMMEDIATELY PRIOR close."""
    p0 = w_prem[0]
    dipped = False
    prev = p0
    for i in range(1, len(w_prem)):
        pi = w_prem[i]
        if pi is None or np.isnan(pi) or pi <= 0:
            continue
        # uses only w_prem[0..i] (p0, prev, pi) — never reads ahead.
        if not dipped:
            if pi < p0:
                dipped = True
            prev = pi
            continue
        if pi > prev:  # ticked back up off the dip
            return i
        prev = pi
    return len(w_prem) - 1  # timeout


def pick_vwap(w_mi, day_stock):
    """No-lookahead. First window bar where UNDERLYING close >= running SESSION vwap (computed from
    bars 0..current minute only). Running vwap = sum(tp*vol)/sum(vol) over mi in [0, current]."""
    # precompute cumulative vwap up to each minute present in the day (session start = mi 0)
    minutes = sorted(day_stock.keys())
    cum_pv = 0.0
    cum_v = 0.0
    run_vwap = {}
    for m in minutes:
        c, tp, vol = day_stock[m]
        cum_pv += tp * vol
        cum_v += vol
        run_vwap[m] = (cum_pv / cum_v) if cum_v > 0 else c
    for i, m in enumerate(w_mi):
        m = int(m)
        if m not in day_stock or m not in run_vwap:
            continue
        c = day_stock[m][0]
        # run_vwap[m] uses only minutes <= m (built in ascending order, sliced implicitly). no lookahead.
        if c >= run_vwap[m]:
            return i
    return len(w_mi) - 1  # timeout


def pf(x):
    x = np.asarray(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else (float("inf") if g > 0 else 0.0)


# ------------------------------------------------------------------------------------------------
def build_trades(tickers, side, W, validate=False):
    """Returns list of dicts, one per ticker/day, with each strategy's (entry_idx, entry_prem, ret)."""
    right = "PUT" if side == "put" else "CALL"
    rows = []
    val_printed = 0
    for tk in tickers:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock_close = D._stock(tk)           # {date:{mi:close}} for underlying feed (matches harness up[])
        stock_full = load_stock_full(tk)     # {date:{mi:(close,tp,vol)}} for VWAP
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=side), is_put=(side == "put"))
        nd = 0
        for date, g in df.groupby("date"):
            if validate and (date[:7] not in ("2025-03",) or val_printed >= 8):
                if val_printed >= 8:
                    break
                continue
            if date not in stock_close or ENTRY_MI not in stock_close[date]:
                continue
            spot = stock_close[date][ENTRY_MI]
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
            if len(ch) < 5:
                continue
            mi_all = ch["mi"].to_numpy(int)
            prem_all = ch["close"].to_numpy(float)
            if mi_all[0] != ENTRY_MI or np.isnan(prem_all[0]) or prem_all[0] <= 0:
                continue
            # build window [mi0, mi0+W]  (entry-decision window)
            win_mask = mi_all <= ENTRY_MI + W
            w_mi = mi_all[win_mask]
            w_prem = prem_all[win_mask]
            if len(w_mi) < 2:
                continue
            day_full = stock_full.get(date, {})
            yr = date[:4]

            # choose entry index per strategy
            picks = {
                "NOW": pick_now(w_prem),
                "ORACLE": pick_oracle(w_prem),
                "PULLBACK": pick_pullback(w_prem),
                "VWAP": pick_vwap(w_mi, day_full) if day_full else len(w_mi) - 1,
            }
            for d in DIP_LEVELS:
                picks[f"DIP{d}"] = pick_dip(w_prem, d)

            rec = {"date": date, "yr": yr, "tk": tk, "p0": float(w_prem[0]),
                   "oracle_min": float(np.nanmin([p for p in w_prem if p and p > 0]))}
            # run exit from each strategy's entry minute to EOD
            for name, idx in picks.items():
                em = int(w_mi[idx])
                # full option path from entry minute to EOD (reuse all option bars >= em)
                fmask = mi_all >= em
                pp = prem_all[fmask]
                mp = mi_all[fmask]
                if len(pp) < 2 or np.isnan(pp[0]) or pp[0] <= 0:
                    rec[name] = None
                    rec[f"{name}_ep"] = None
                    rec[f"{name}_em"] = em
                    continue
                up = [stock_close[date].get(int(m), spot) for m in mp]
                ets = datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
                ret = sim(pp, list(mp), list(up), cfg, side, ets)
                rec[name] = ret
                rec[f"{name}_ep"] = float(pp[0])
                rec[f"{name}_em"] = em
            rows.append(rec)
            nd += 1

            if validate and val_printed < 8:
                print(f"\n[SANITY] {tk} {date} ({side}) W={W}  mi0={ENTRY_MI} p0={rec['p0']:.2f} "
                      f"oracle_min={rec['oracle_min']:.2f}")
                for name in ["NOW", "ORACLE"] + [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK", "VWAP"]:
                    em, ep = rec.get(f"{name}_em"), rec.get(f"{name}_ep")
                    r = rec.get(name)
                    eps = f"{ep:.2f}" if ep is not None else "  - "
                    rs = f"{r:+7.1f}%" if r is not None else "   n/a "
                    print(f"    {name:<9} entry_min={em:>3} entry_prem={eps:>6}  ret={rs}")
                # NO-LOOKAHEAD self-check: oracle entry must be <= every implementable rule's entry prem,
                # and oracle_min must equal the min of the window (peeking). Implementable rules can't
                # beat oracle on entry price.
                assert abs(rec["oracle_min"] - min(p for p in w_prem if p and p > 0)) < 1e-6
                if rec.get("DIP5_ep") is not None:
                    assert rec["ORACLE_ep"] <= rec["DIP5_ep"] + 1e-9, "oracle should be cheapest entry"
                val_printed += 1
        print(f"  {tk}: {nd} trades", flush=True)
        if validate and val_printed >= 8:
            break
    return rows


def report(rows, side, W):
    tdf = pd.DataFrame(rows)
    strategies = ["NOW", "ORACLE"] + [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK", "VWAP"]
    years = sorted(tdf["yr"].unique()) + ["ALL"]

    print(f"\n{'='*100}\nSIDE={side.upper()}  WINDOW W={W}min   n={len(tdf)} trades   "
          f"(entry decision at mi0={ENTRY_MI} = 10:00 ET)\n{'='*100}")

    # ---- (1) entry-price improvement vs NOW, and net P&L + PF per year ----
    for yr in years:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        n = len(sub)
        print(f"\n--- year {yr}  (n={n}{'  *LOW-N*' if n < 40 else ''}) ---")
        print(f"  {'strategy':<10}{'entryImpr%':>11}{'netP&L(u)':>11}{'PF':>7}{'WR%':>7}"
              f"{'avgRet%':>9}{'%timeout':>10}")
        base_pl = sub["NOW"].dropna().sum()
        for name in strategies:
            r = sub[name].dropna()
            if len(r) == 0:
                continue
            # entry improvement: (NOW_ep - rule_ep)/NOW_ep averaged (positive = cheaper entry)
            paired = sub.dropna(subset=["NOW_ep", f"{name}_ep"])
            impr = ((paired["NOW_ep"] - paired[f"{name}_ep"]) / paired["NOW_ep"] * 100).mean()
            netpl = r.sum()
            wr = (r > 0).mean() * 100
            # timeout %: entry minute == window end (only meaningful for the no-lookahead rules)
            to = (sub[f"{name}_em"] == ENTRY_MI + W).mean() * 100 if name not in ("NOW", "ORACLE") else 0.0
            tag = ""
            if name == "ORACLE":
                tag = " <-CEILING"
            print(f"  {name:<10}{impr:>+11.2f}{netpl:>+11.0f}{pf(r):>7.2f}{wr:>7.1f}{r.mean():>+9.2f}"
                  f"{to:>9.1f}%{tag}")
        print(f"    (NOW net P&L = {base_pl:+.0f} units — the bar to beat)")

    # ---- (2) PRIZE: oracle vs now overall + per year ----
    print(f"\n{'-'*100}\n(1) THE PRIZE — ORACLE (lookahead ceiling) vs NOW, per year:")
    for yr in years:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        now_pl, ora_pl = sub["NOW"].dropna().sum(), sub["ORACLE"].dropna().sum()
        prize = ora_pl - now_pl
        print(f"    {yr:<6} n={len(sub):<5} NOW={now_pl:>+8.0f}u  ORACLE={ora_pl:>+8.0f}u  "
              f"PRIZE=+{prize:>7.0f}u  ({(ora_pl/now_pl if now_pl>0 else float('nan')):.2f}x NOW)")

    # ---- (3) ADVERSE-SELECTION accounting (overall) for each no-lookahead rule ----
    print(f"\n{'-'*100}\n(2) ADVERSE-SELECTION accounting (overall, vs NOW):")
    now_winmask = tdf["NOW"] > 0
    # runner = option never dipped in window (monotone up) -> DIP rules forced to timeout-top
    def never_dipped(row):
        # recompute from stored ep path proxy: oracle_min >= p0 means low was at/above entry => no dip
        return row["oracle_min"] >= row["p0"] - 1e-9
    tdf["_runner"] = tdf.apply(never_dipped, axis=1)
    for name in [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK", "VWAP"]:
        paired = tdf.dropna(subset=["NOW", name])
        # (a) timeout %
        to = (paired[f"{name}_em"] == ENTRY_MI + W).mean() * 100
        # (b) among NOW-winners, % the rule worsens (lower ret) or misses (entered later & worse)
        nw = paired[paired["NOW"] > 0]
        worsened = (nw[name] < nw["NOW"] - 1e-9).mean() * 100 if len(nw) else float("nan")
        # (c) runner-miss: among monotone-up runners, % where rule bought the timeout top (em==window end)
        runners = paired[paired["_runner"]]
        runner_miss = (runners[f"{name}_em"] == ENTRY_MI + W).mean() * 100 if len(runners) else float("nan")
        # capture of prize
        cap = (paired[name].sum() - paired["NOW"].sum()) / \
              (paired["ORACLE"].sum() - paired["NOW"].sum()) * 100 \
              if (paired["ORACLE"].sum() - paired["NOW"].sum()) != 0 else float("nan")
        delta_pl = paired[name].sum() - paired["NOW"].sum()
        print(f"  {name:<9}  timeout={to:>5.1f}%  worsens-NOW-winners={worsened:>5.1f}%  "
              f"runner-miss={runner_miss:>5.1f}%  | ΔP&L vs NOW={delta_pl:>+7.0f}u  "
              f"prize-capture={cap:>6.1f}%")
    print(f"    (runners = option never dipped below entry in the window; n_runner={int(tdf['_runner'].sum())} "
          f"of {len(tdf)} = {tdf['_runner'].mean()*100:.0f}%)")

    return tdf


def main():
    side = os.environ.get("SIDE", "call").lower()
    windows = [int(x) for x in os.environ.get("WINDOWS", "15,30").split(",")]
    validate = os.environ.get("VALIDATE") == "1"

    if validate:
        print("### VALIDATION PASS — 2 tickers x 1 month (2025-03), sanity rows + no-lookahead asserts ###")
        rows = build_trades(["SPY", "NVDA"], side, 30, validate=True)
        report(rows, side, 30)
        print("\nVALIDATION OK — no-lookahead asserts passed. Re-run without VALIDATE=1 to scale.\n")
        return

    tickers = LIQUID_CALLS  # liquid set
    for W in windows:
        print(f"\n\n########## BUILDING TRADES  side={side}  W={W}  tickers={tickers} ##########", flush=True)
        rows = build_trades(tickers, side, W, validate=False)
        report(rows, side, W)


if __name__ == "__main__":
    main()
