"""Pyramid-ladder backtest: add to CONFIRMED runners at one or more levels, calls AND puts.

Re-runs the flow sim (cached UW sweeps, no API) + ML calls, recording peak_gain per trade. Then,
for add-levels L (a trade adds at L if its peak >= L), models the add leg as bought at (1+L/100)x
entry and exiting with the base (return = (1+exit/100)/(1+L/100) - 1). Reports (1) each add level's
STANDALONE economics per direction — do +30/+50/+100 adds pay? — and (2) blended config PF/mean.
Approximation: ignores add-fill slippage; the % exit is unchanged by size. Read-only.
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
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

HAIRCUT = D.EXIT_HAIRCUT
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
CACHE = Path("/tmp/flow_otm_sweeps.pkl")


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
    sweeps = pickle.loads(CACHE.read_bytes())
    out = []
    for is_put, wl in ((True, PUT_UNIV), (False, CALL_UNIV)):
        sig = sweeps[is_put]
        sig = sig[sig["ticker"].isin(wl)]
        otype = "put" if is_put else "call"
        right = "PUT" if is_put else "CALL"
        for tk in sorted(sig["ticker"].unique()):
            stock, opts = D._stock(tk), D._opts(tk, right)
            cfg = D.apply_v7_wide_trail_exits(
                D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
            for _, ev in sig[sig["ticker"] == tk].iterrows():
                d, em = ev["date"], int(ev["mb"])
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
                ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
                ret, peak = sim_peak(pp, list(mp), list(up), pp[0], ets, cfg, int(dte0), otype)
                out.append({"dir": otype, "peak": peak, "ret": ret})
    return pd.DataFrame(out)


def add_leg_returns(df, L):
    """Return array of add-leg %returns for trades that crossed +L% (bought at 1+L)."""
    elig = df[df.peak >= L]
    return ((1 + elig.ret / 100) / (1 + L / 100) - 1).to_numpy() * 100


def stats(a):
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    g = a[a > 0].sum(); l = -a[a < 0].sum()
    return (len(a), a.mean(), (g / l if l > 0 else float("inf")), a.sum(), np.mean(a > 0) * 100)


def main():
    print("building peak-recorded trade set (flow cached + ML calls)...", flush=True)
    fl = flow_trades()
    ml = pd.read_csv("journal/v3_eval_results/v7_core_trades.csv")[["direction", "peak_gain", "pnl_pct"]]
    ml = ml.rename(columns={"direction": "dir", "peak_gain": "peak", "pnl_pct": "ret"})
    allt = pd.concat([fl, ml], ignore_index=True)
    print(f"trades: {len(allt)} ({(allt.dir=='call').sum()} call, {(allt.dir=='put').sum()} put)\n")

    LEVELS = [30, 50, 100, 150]
    for dname, dfd in [("CALL", allt[allt.dir == "call"]), ("PUT", allt[allt.dir == "put"])]:
        print(f"================ {dname}  (n={len(dfd)}, base PF {stats(dfd.ret.to_numpy())[2]:.2f}) ================")
        print(f"-- standalone ADD-LEG economics (does adding AT each level pay?) --")
        print(f"{'level':<8}{'n_elig':>7}{'%reach':>7}{'add_mean%':>10}{'add_win%':>9}{'add_PF':>8}")
        for L in LEVELS:
            a = add_leg_returns(dfd, L)
            n, m, pf, tot, win = stats(a)
            print(f"+{L:<7}{n:>7}{n/len(dfd)*100:>6.0f}%{m:>+10.1f}{win:>8.0f}%{pf:>8.2f}")
        # blended configs: base + adds at the listed levels (each add = 1 sleeve)
        print(f"-- blended configs (base + adds; mean=per-sleeve, total=sum of all sleeves) --")
        print(f"{'config':<12}{'sleeves':>8}{'mean%':>8}{'PF':>7}{'total%':>9}")
        cfgs = {"base": [], "+30": [30], "+30+100": [30, 100], "+50": [50],
                "+50+150": [50, 150], "+100": [100]}
        for cn, lv in cfgs.items():
            legs = list(dfd.ret.to_numpy())
            for L in lv:
                legs += list(add_leg_returns(dfd, L))
            n, m, pf, tot, win = stats(np.array(legs))
            print(f"{cn:<12}{n:>8}{m:>+8.1f}{pf:>7.2f}{tot:>+9.0f}")
        print()
    print("NOTE: add-leg return approximated (bought at +L, exits with base; no add-fill slippage). "
          "A level 'pays' if add_PF > 1 AND beats the base PF. Pyramid worth it if higher levels still pay.")


if __name__ == "__main__":
    main()
