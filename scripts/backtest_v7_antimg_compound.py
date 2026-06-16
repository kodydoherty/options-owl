"""Gold-standard V7 (ML + flow) compounding off $18k, LAST 60 DAYS — updated 2026-06-16 with the
deployed exit/add changes:

  • CALL profit-lock (ENABLE_V7_PROFIT_LOCK): keep 60% of peak gain once a call peaks +30%
    (puts exempt — they keep the V7 wide trail). Applied in-sim to FLOW trades (path available).
  • Separate-leg multi-level adds via M1 OWN-TRAIL: each add runs its own fresh V7 FSM from the
    add minute (calls +30/+80/+150, puts +30/+100). This REPLACES the old M2 "add exits with the
    parent" approximation, which the 2026-06-16 add-handling backtest proved is a big loser
    (the wide parent trail drags the add to break-even). Adds are sized 1x contracts at the +L fill.

ML trades come from v7_core_trades.csv (peak + pnl_pct) — no per-minute path, so the profit-lock and
M1 adds can NOT be re-simulated for ML. ML is carried at its base CSV return with NO adds
(CONSERVATIVE: understates, since flow is ~4x the ML edge and carries the adds). Read-only, cached.

Rows: BASE V7 (no lock, no adds) → +CALL profit-lock → +multi-level M1 adds (FULL deployed strategy).
Sizing = deployed (RISK 75% / 8 slots / 15% cap / PUT budget 0.50 / $50k liquidity cap), conviction-scaled.
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.bot_runner import select_flow_strike  # noqa: E402
from options_owl.risk.exit_v5.config import INDEX_TICKERS  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402
from options_owl.risk.vinny_strategy import flow_conviction_mult  # noqa: E402

HAIRCUT = D.EXIT_HAIRCUT
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
OTM_CALL, OTM_PUT, OTM_TARGET = {"AMD", "INTC", "META", "SPY"}, {"TSLA"}, 2.0
CLUSTER_WIN = 30
START_BAL, RISK_PCT, MAX_CONC, POS_CAP, PUT_BUDGET = 18000.0, 0.75, 8, 0.15, 0.50
DAYS = 60
LIQ_CAP = 50000.0
CALL_LEVELS, PUT_LEVELS = [30, 80, 150], [30, 100]


class _SL:  # settings with the deployed CALL profit-lock ON (gate is call-only inside the FSM)
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0
    ENABLE_V7_PROFIT_LOCK = True
    V7_PROFIT_LOCK_KEEP_FRAC = 0.6
    V7_PROFIT_LOCK_ACTIVATE_PCT = 30.0


def run_sim(pp, mp, up, ets, cfg, dte, otype, start_idx=0, settings=None):
    """FSM exit sim from start_idx. Returns (ret%, peak%) relative to entry at start_idx."""
    ep = pp[start_idx]
    fsm = ExitFSM(cfg, settings=settings or D._S())
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep,
                    entry_time=ets + timedelta(minutes=int(mp[start_idx] - mp[0])),
                    contracts=1, peak_premium=ep, entry_underlying_price=up[start_idx], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last, peak = ep, ep
    for k in range(start_idx + 1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last, peak = prem, max(peak, prem)
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        a = fsm.evaluate(st, prem, prem * (1 - HAIRCUT), prem, now,
                         current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if a.should_exit:
            return (prem * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100
    return (last * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100


def add_idx(pp, mp, ep, L):
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        mins = mp[k] - mp[0]
        if mins < 3:
            continue
        if mins > 60:
            return None
        if (prem - ep) / ep * 100 >= L:
            return k
    return None


def flow_paths():
    sweeps = pickle.loads(Path("/tmp/flow_otm_sweeps.pkl").read_bytes())
    out = []
    for is_put, wl in ((True, PUT_UNIV), (False, CALL_UNIV)):
        sig = sweeps[is_put]
        sig = sig[sig["ticker"].isin(wl)]
        otype, right = ("put", "PUT") if is_put else ("call", "CALL")
        levels = PUT_LEVELS if is_put else CALL_LEVELS
        for tk in sorted(sig["ticker"].unique()):
            stock, opts = D._stock(tk), D._opts(tk, right)
            cfg = D.apply_v7_wide_trail_exits(
                D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
            is_idx = tk in INDEX_TICKERS
            g_tk = sig[sig["ticker"] == tk]
            for d, gg in g_tk.groupby("date"):
                mis = gg["mi"].to_numpy()
                seen = set()
                for _, ev in gg.iterrows():
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
                    same = oday[oday["dte"] == dte0]
                    pchain = [{"strike": float(r.strike), "mid": float(r.close)} for r in same.itertuples()]
                    strike, _ = select_flow_strike(pchain, spot, is_put, tk in (OTM_PUT if is_put else OTM_CALL), OTM_TARGET)
                    if not strike:
                        continue
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
                    ret_nl, peak = run_sim(pp, list(mp), list(up), ets, cfg, int(dte0), otype, settings=D._S())
                    ret_lk, _ = run_sim(pp, list(mp), list(up), ets, cfg, int(dte0), otype, settings=_SL())
                    adds = []
                    for L in levels:
                        ai = add_idx(pp, mp, pp[0], L)
                        if ai is None:
                            continue
                        ar, _ = run_sim(pp, list(mp), list(up), ets, cfg, int(dte0), otype, start_idx=ai, settings=_SL())
                        adds.append((L, ar))
                    mult = flow_conviction_mult(csize, float(ev["total_premium"]), float(ev["ask_frac"]), is_idx, None)[0]
                    out.append({"date": d, "src": "flow", "tk": tk, "ret_nl": ret_nl, "ret_lk": ret_lk,
                                "peak": peak, "mult": mult, "is_put": is_put, "adds": adds})
    return pd.DataFrame(out)


def compound(df, lock, antimg, max_level=999):
    """lock=use profit-locked base ret (flow only); antimg=fold M1 add legs (L<=max_level, flow only)."""
    bal = START_BAL
    daily, legs = {}, []
    for d, g in df.groupby("date", sort=True):
        per_slot = bal * RISK_PCT / MAX_CONC
        cap = bal * POS_CAP
        day = 0.0
        for t in g.itertuples():
            ret = (t.ret_lk if (lock and t.src == "flow") else t.ret_nl)
            size = min(per_slot * t.mult * (PUT_BUDGET if t.is_put else 1.0), cap, LIQ_CAP)
            day += size * ret / 100.0
            legs.append(ret)
            if antimg and t.src == "flow":
                for L, ar in t.adds:
                    if L > max_level:
                        continue
                    add_size = min(size * (1 + L / 100.0), cap, LIQ_CAP)
                    day += add_size * ar / 100.0
                    legs.append(ar)
        daily[d] = day
        bal += day
    return bal, daily, np.array(legs)


def _dd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def main():
    print("building 60d trade set (flow paths: base + profit-lock + M1 adds; ML from CSV)...", flush=True)
    fl = flow_paths()
    ml = pd.read_csv("journal/v3_eval_results/v7_core_trades.csv")
    mlt = pd.DataFrame({"date": ml["day"].astype(str), "src": "ML",
                        "ret_nl": ml["pnl_pct"], "ret_lk": ml["pnl_pct"],
                        "peak": ml["peak_gain"], "mult": ml["size_mult"].fillna(1.0),
                        "is_put": ml["direction"].str.lower() == "put"})
    mlt["adds"] = [[] for _ in range(len(mlt))]
    allt = pd.concat([fl, mlt], ignore_index=True)
    maxd = pd.to_datetime(allt["date"]).max()
    cut = (maxd - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%d")
    win = allt[allt["date"] >= cut].copy()
    print(f"window: {cut} → {maxd.strftime('%Y-%m-%d')} (last {DAYS}d) | {len(win)} trades "
          f"({(win.src=='flow').sum()} flow, {(win.src=='ML').sum()} ML)\n")

    print(f"{'config':<28}{'start':>9}{'end':>12}{'P&L':>12}{'PF':>7}{'maxDD':>11}{'WR':>6}{'legs':>6}")
    rows = [("BASE V7 (no lock/add)", False, False, 0),
            ("+ CALL profit-lock", True, False, 0),
            ("+ profit-lock + adds @30 only", True, True, 30),
            ("+ profit-lock + adds 30/80/150", True, True, 999)]
    for label, lock, am, mx in rows:
        bal, daily, legs = compound(win, lock, am, max_level=mx)
        g = legs[legs > 0].sum(); ll = -legs[legs < 0].sum()
        pf = g / ll if ll > 0 else float("inf")
        print(f"{label:<28}${START_BAL:>8,.0f}${bal:>11,.0f}${bal-START_BAL:>+11,.0f}"
              f"{pf:>7.2f}${_dd(daily):>+10,.0f}{np.mean(legs > 0) * 100:>5.0f}%{len(legs):>6}")
    print("\nNOTE: profit-lock + M1 adds applied to FLOW (path available); ML carried at base CSV "
          "return, no adds (conservative). Multi-level adds are higher-variance (amplify trends, "
          "drag chop). M1 add model validated 2026-06-16 (M2 ride-parent is a loser).")


if __name__ == "__main__":
    main()
