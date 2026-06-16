"""Gold-standard V7 (ML + flow) WITH the anti-martingale add, compounding off $18k, last 30 days.

Re-runs the flow book (cached sweeps, prod select_flow_strike + conviction sizing) recording peak,
plus the ML calls, then compounds with the deployed sizing (RISK 75% / 8 slots / 15% cap / PUT
budget 0.50). Anti-martingale add legs (CALL +30, PUT +30 & +100) fold in when a trade's peak
crosses the level. Reports BASE vs WITH-antimg so the add's contribution is explicit. Read-only.
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
DAYS = 30
LIQ_CAP = 50000.0   # prod MAX_POSITION_DOLLARS — realism brake (uncapped = unfillable fantasy)


def sim_peak(pp, mp, up, ep, ets, cfg, dte, otype):
    fsm = ExitFSM(cfg, settings=D._S())
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last, peak = ep, ep
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        peak = max(peak, prem)
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        act = fsm.evaluate(st, prem, prem * (1 - HAIRCUT), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            return (prem * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100
    return (last * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100


def flow_trades():
    sweeps = pickle.loads(Path("/tmp/flow_otm_sweeps.pkl").read_bytes())
    out = []
    for is_put, wl in ((True, PUT_UNIV), (False, CALL_UNIV)):
        sig = sweeps[is_put]
        sig = sig[sig["ticker"].isin(wl)]
        otype, right = ("put", "PUT") if is_put else ("call", "CALL")
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
                    ret, peak = sim_peak(pp, list(mp), list(up), pp[0], ets, cfg, int(dte0), otype)
                    mult = flow_conviction_mult(csize, float(ev["total_premium"]), float(ev["ask_frac"]), is_idx, None)[0]
                    out.append({"date": d, "src": "flow", "ret": ret, "peak": peak,
                                "mult": mult, "is_put": is_put})
    return pd.DataFrame(out)


def antimg_levels(is_put):
    return [100, 30] if is_put else [30]


def compound(df, use_antimg):
    bal = START_BAL
    daily, legs = {}, []
    for d, g in df.groupby("date", sort=True):
        per_slot = bal * RISK_PCT / MAX_CONC
        cap = bal * POS_CAP
        day = 0.0
        for t in g.itertuples():
            size = min(per_slot * t.mult * (PUT_BUDGET if t.is_put else 1.0), cap, LIQ_CAP)
            day += size * t.ret / 100.0
            legs.append(t.ret)
            if use_antimg and t.peak is not None and not np.isnan(t.peak):
                for L in antimg_levels(t.is_put):
                    if t.peak >= L:
                        add_size = min(size * (1 + L / 100.0), cap, LIQ_CAP)  # 1x contracts at +L
                        add_ret = (1 + t.ret / 100.0) / (1 + L / 100.0) - 1
                        day += add_size * add_ret
                        legs.append(add_ret * 100)
        daily[d] = day
        bal += day
    return bal, daily, np.array(legs)


def _dd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def main():
    print("building trade set (flow cached + ML calls, peak-recorded)...", flush=True)
    fl = flow_trades()
    ml = pd.read_csv("journal/v3_eval_results/v7_core_trades.csv")
    mlt = pd.DataFrame({"date": ml["day"].astype(str), "src": "ML", "ret": ml["pnl_pct"],
                        "peak": ml["peak_gain"], "mult": ml["size_mult"].fillna(1.0),
                        "is_put": ml["direction"].str.lower() == "put"})
    allt = pd.concat([fl, mlt], ignore_index=True)
    maxd = pd.to_datetime(allt["date"]).max()
    cut = (maxd - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%d")
    win = allt[allt["date"] >= cut].copy()
    print(f"window: {cut} → {maxd.strftime('%Y-%m-%d')} (last {DAYS}d) | {len(win)} trades "
          f"({(win.src=='flow').sum()} flow, {(win.src=='ML').sum()} ML)\n")

    print(f"{'config':<20}{'start':>9}{'end':>12}{'P&L':>12}{'PF':>7}{'maxDD':>11}{'WR':>6}{'legs':>6}")
    for label, ua in [("BASE (no antimg)", False), ("WITH antimg", True)]:
        bal, daily, legs = compound(win, ua)
        g = legs[legs > 0].sum(); l = -legs[legs < 0].sum()
        pf = g / l if l > 0 else float("inf")
        print(f"{label:<20}${START_BAL:>8,.0f}${bal:>11,.0f}${bal-START_BAL:>+11,.0f}"
              f"{pf:>7.2f}${_dd(daily):>+10,.0f}{np.mean(legs>0)*100:>5.0f}%{len(legs):>6}")


if __name__ == "__main__":
    main()
