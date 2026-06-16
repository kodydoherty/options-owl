"""Add-handling backtest: how should the anti-martingale ADD leg exit, how many levels, and
does a profit-lock ratchet beat the V7 wide trail's round-trip-to-breakeven? Read-only, cached.

Re-runs the flow sim (cached UW sweeps) keeping the FULL per-minute premium path per trade, so we
can simulate the ADD leg three ways at each level L (add when peak first crosses +L in a 3-60min
window, mirroring ANTIMG_MIN/MAX_MINUTES):

  M1 own-trail   — the add runs its OWN fresh V7 FSM from the add minute (CURRENT live behavior).
                   Risk seen 2026-06-16: TSLA add shaken out 8min later, missed the run to peak.
  M2 ride-parent — the add exits at the PARENT's exit price (what backtest_pyramid_ladder validated).
  M3 blend       — base+add collapse to one sleeve on a blended entry; the FSM fires on the diluted
                   gain%, so the whole position exits earlier (the old SPY "+10%" complaint).

Then: multi-level ladders (calls 30/80/150, puts 30/100) under the winning model, and a profit-lock
ratchet sweep on the BASE (keep K% of peak gain once peak>=ACT) vs the V7 wide trail — to see if a
tighter give-back keeps more of a +111% TSLA-style runner without killing the moonshots.

Approximation: 1 contract per leg, no add-fill slippage, underlying-confirm gate omitted (minor).
Uses ONLY flow trades (ML CSV has no path, so M1/M3 can't be path-simulated for it).
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

HAIRCUT = D.EXIT_HAIRCUT
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
CACHE = Path("/tmp/flow_otm_sweeps.pkl")
ADD_MIN, ADD_MAX = 3, 60  # ANTIMG_MIN/MAX_MINUTES window for the add
CALL_LEVELS = [30, 80, 150]
PUT_LEVELS = [30, 100]


def run_fsm(pp, mp, up, ets, cfg, dte, otype, start_idx=0, entry=None):
    """Run the V7 FSM from start_idx. Returns (exit_idx, exit_prem_raw, ret_pct, peak_pct)."""
    ep = entry if entry is not None else pp[start_idx]
    fsm = ExitFSM(cfg, settings=D._S())
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
        act = fsm.evaluate(st, prem, prem * (1 - HAIRCUT), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            return k, prem, (prem * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100
    return len(pp) - 1, last, (last * (1 - HAIRCUT) - ep) / ep * 100, (peak - ep) / ep * 100


def sim_ratchet(pp, mp, ep, keep, activate):
    """Profit-lock ratchet: once peak gain >= activate%, exit when current gain < keep*peak_gain.
    Returns ret_pct. A pure give-back rule on peak GAIN (not the FSM)."""
    peak_g = 0.0
    last = ep
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        g = (prem - ep) / ep * 100
        peak_g = max(peak_g, g)
        if peak_g >= activate and g < keep * peak_g:
            return (prem * (1 - HAIRCUT) - ep) / ep * 100
    return (last * (1 - HAIRCUT) - ep) / ep * 100


def add_index(pp, mp, ep, L):
    """First minute index where gain crosses +L within the [ADD_MIN, ADD_MAX] window, else None."""
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        mins = mp[k] - mp[0]
        if mins < ADD_MIN:
            continue
        if mins > ADD_MAX:
            return None
        if (prem - ep) / ep * 100 >= L:
            return k
    return None


def load_paths():
    """Yield dicts with the full path per flow trade."""
    sweeps = pickle.loads(CACHE.read_bytes())
    trades = []
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
                trades.append(dict(otype=otype, pp=pp, mp=mp, up=up, ets=ets, cfg=cfg, dte=int(dte0),
                                   date=d, tk=tk, month=d[:7]))
    return trades


def stats(a):
    a = np.array(a, dtype=float)
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    g, l = a[a > 0].sum(), -a[a < 0].sum()
    return (len(a), a.mean(), (g / l if l > 0 else float("inf")), a.sum(), np.mean(a > 0) * 100)


def fmt(name, s):
    return f"{name:<16}{s[0]:>7}{s[1]:>+9.1f}{s[2]:>8.2f}{s[4]:>8.0f}%{s[3]:>+11.0f}"


def main():
    print("loading flow paths (cached)...", flush=True)
    trades = load_paths()
    calls = [t for t in trades if t["otype"] == "call"]
    puts = [t for t in trades if t["otype"] == "put"]
    print(f"flow trades: {len(trades)}  ({len(calls)} call, {len(puts)} put)\n")

    for dname, ts, LEVELS in (("CALL", calls, CALL_LEVELS), ("PUT", puts, PUT_LEVELS)):
        # base FSM per trade (cache exit so M2 can reuse it)
        base = []
        for t in ts:
            ei, ep_x, ret, peak = run_fsm(t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"])
            base.append(dict(ret=ret, peak=peak, exit_prem=ep_x, t=t))
        bs = stats([b["ret"] for b in base])
        print(f"================ {dname}  (n={len(ts)}) ================")
        print(f"{'':<16}{'n':>7}{'mean%':>9}{'PF':>8}{'win%':>8}{'total%':>11}")
        print(fmt("BASE (V7)", bs))

        # ---- add-leg economics by model & level ----
        print(f"\n-- ADD-LEG return by exit model (add when peak crosses +L, {ADD_MIN}-{ADD_MAX}min) --")
        print(f"{'level':<8}{'%reach':>7}   {'M1 own-trail PF/mean':>22}   {'M2 ride-parent PF/mean':>24}")
        m1_all, m2_all = {L: [] for L in LEVELS}, {L: [] for L in LEVELS}
        for L in LEVELS:
            for b in base:
                t = b["t"]
                ai = add_index(t["pp"], t["mp"], t["pp"][0], L)
                if ai is None:
                    continue
                add_ep = t["pp"][ai]
                # M1: own fresh FSM from the add minute
                _, _, r1, _ = run_fsm(t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"], start_idx=ai)
                m1_all[L].append(r1)
                # M2: exit at the parent's exit price
                r2 = (b["exit_prem"] * (1 - HAIRCUT) - add_ep) / add_ep * 100
                m2_all[L].append(r2)
            s1, s2 = stats(m1_all[L]), stats(m2_all[L])
            reach = len(m1_all[L]) / len(ts) * 100
            print(f"+{L:<7}{reach:>6.0f}%   PF {s1[2]:>5.2f}  mean {s1[1]:>+6.1f} (n={s1[0]:<3})   "
                  f"PF {s2[2]:>5.2f}  mean {s2[1]:>+6.1f} (n={s2[0]})")

        # ---- combined book: base + ladder, per model ----
        print(f"\n-- COMBINED book (base + add ladder), per model --")
        print(f"{'config':<16}{'sleeves':>8}{'mean%':>9}{'PF':>8}{'win%':>8}{'total%':>11}")
        ladders = {"base only": [], "+30": [30], "+30+80": [30, 80],
                   "+30+80+150": [30, 80, 150]} if dname == "CALL" else \
                  {"base only": [], "+30": [30], "+30+100": [30, 100]}
        for cn, levs in ladders.items():
            for model, store in (("M1", m1_all), ("M2", m2_all)):
                legs = [b["ret"] for b in base]
                for L in levs:
                    legs += store[L]
                print(fmt(f"{cn} [{model}]", stats(legs)))

        # ---- profit-lock ratchet sweep on the BASE ----
        print(f"\n-- PROFIT-LOCK ratchet on BASE (keep K% of peak gain once peak>=ACT) vs V7 wide trail --")
        print(f"{'rule':<16}{'n':>7}{'mean%':>9}{'PF':>8}{'win%':>8}{'total%':>11}")
        print(fmt("V7 wide (now)", bs))
        for act in (30, 50):
            for keep in (0.5, 0.6, 0.7):
                rets = [sim_ratchet(t["pp"], t["mp"], t["pp"][0], keep, act) for t in ts]
                print(fmt(f"keep{int(keep*100)}@+{act}", stats(rets)))
        print()


if __name__ == "__main__":
    main()
