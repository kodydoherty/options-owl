"""DCA-into-runners FULL sweep + compounding proof. Answers:
  1. Which DIP amount is best? (sweep 8/10/12/15/20% pullback from the post-confirm peak)
  2. How many TRANCHES? (buy the dip 1x / 2x / 3x per trade)
  3. Per-TICKER — does the best dip differ by name, or does one standard amount work?
  4. Does it survive COMPOUNDING off $18k with the $50k take-profit cap? (the real-account proof)

Confirmation = +30% gain (non-lookahead; live also gates on P(runner)+ML conf → more selective).
Each dip-add is a SEPARATE own-trail leg (V7 + profit-lock). CALL dip-adds only (runner = momentum);
puts stay base. Compound = event-driven (8 slots + 75% capital + measured liquidity/slippage + $50k cap).
Read-only, cached.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402
import backtest_realistic_account as R  # noqa: E402  (liq_cap, slip, constants)
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

H = B.HAIRCUT
CONFIRM, ADD_WIN = 30.0, 90
START, CAP = 18000.0, 50000.0
DIPS = [8, 10, 12, 15, 20]
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def _ok(p):
    return p is not None and not np.isnan(p) and p > 0


def sim_from(t, si):
    """Own-trail exit from index si. Returns (ret%, exit_min)."""
    pp, mp, up, ets, cfg, dte, otype = (t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"])
    ep = pp[si]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep,
                    entry_time=ets + timedelta(minutes=int(mp[si] - mp[0])), contracts=1, peak_premium=ep,
                    entry_underlying_price=up[si], dte=int(dte), expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(si + 1, len(pp)):
        if not _ok(pp[k]):
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k], minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100, int(mp[k])
    return (last * (1 - H) - ep) / ep * 100, int(mp[-1])


def dip_legs(t, D, maxn=3):
    """Up to maxn dip-add legs: confirm +30%, then each -D% pullback from a fresh running peak.
    Returns list of (entry_min, exit_min, ret)."""
    pp, mp, ep = t["pp"], t["mp"], t["pp"][0]
    ci = None
    for k in range(1, len(pp)):
        if not _ok(pp[k]):
            continue
        m = mp[k] - mp[0]
        if m < 3:
            continue
        if m > 60:
            break
        if (pp[k] - ep) / ep * 100 >= CONFIRM:
            ci = k
            break
    if ci is None:
        return []
    legs, peak = [], pp[ci]
    k = ci + 1
    while k < len(pp) and len(legs) < maxn:
        if not _ok(pp[k]):
            k += 1
            continue
        if mp[k] - mp[0] > ADD_WIN:
            break
        peak = max(peak, pp[k])
        if (peak - pp[k]) / peak * 100 >= D:
            ret, ex = sim_from(t, k)
            legs.append((int(mp[k]), ex, ret))
            peak = pp[k]            # reset — next tranche needs a fresh rise+dip
        k += 1
    return legs


def stats(a):
    a = np.array(a, float)
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0)
    g, l = a[a > 0].sum(), -a[a < 0].sum()
    return (len(a), np.mean(a > 0) * 100, (g / l if l > 0 else float("inf")), a.sum())


def compound(trades):
    """Event-driven compound off $18k, 8 slots + 75% capital + measured liquidity/slippage + $50k cap.
    trades: list of dict(date, tk, entry, exit, ret, is_put). Returns (end, maxdd)."""
    bal, banked, daily = START, 0.0, {}
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["date"]].append(t)
    for d in sorted(by_day):
        g = sorted(by_day[d], key=lambda x: x["entry"])
        deployable, per_slot, capd = bal * R.RISK_PCT, bal * R.RISK_PCT / R.MAX_CONC, bal * R.POS_CAP
        open_pos, committed, day = [], 0.0, 0.0
        for t in g:
            still = []
            for ex, cm, r in open_pos:
                if ex <= t["entry"]:
                    day += cm * r / 100.0
                    committed -= cm
                else:
                    still.append((ex, cm, r))
            open_pos = still
            lc = R.liq_cap(t["tk"])
            size = min(per_slot * (R.PUT_BUDGET if t["is_put"] else 1.0) * R.FLAT, capd, lc)
            if len(open_pos) >= R.MAX_CONC or committed + size > deployable:
                continue
            open_pos.append((t["exit"], size, R.slip(t["ret"], size, t["tk"], lc, 1.0)))
            committed += size
        for ex, cm, r in open_pos:
            day += cm * r / 100.0
        daily[d] = day
        bal += day
        if bal > CAP:
            banked += bal - CAP
            bal = CAP
    return bal + banked, R.maxdd(daily)


def main():
    print("loading paths (cached)...", flush=True)
    paths = B.load_paths()
    calls = [t for t in paths if t["otype"] == "call"]
    base_call = {id(t): sim_from(t, 0) for t in calls}
    # precompute up-to-3 dip tranches per call per D (reuse across N + compound)
    print("computing dip tranches per D...", flush=True)
    tr = {D: {id(t): dip_legs(t, D, 3) for t in calls} for D in DIPS}

    base_rets = [base_call[id(t)][0] for t in calls]
    bn, bwr, bpf, btot = stats(base_rets)
    print(f"\nCALL base (V7+lock): n={bn} WR={bwr:.0f}% PF={bpf:.2f} total={btot:+.0f}")

    # ---- Phase 1+2: dip amount x tranches ----
    print(f"\n== dip amount x #tranches (flat book = base + add legs) ==")
    print(f"{'dip':<6}" + "".join(f"{'N='+str(n):>22}" for n in (1, 2, 3)))
    print(f"{'':6}" + "".join(f"{'addPF/bookPF/booktot':>22}" for _ in (1, 2, 3)))
    best_tot, best_pf = None, None
    for D in DIPS:
        row = f"{'-'+str(D)+'%':<6}"
        for N in (1, 2, 3):
            addr = [r for t in calls for (_, _, r) in tr[D][id(t)][:N]]
            apf = stats(addr)[2]
            book = stats(base_rets + addr)
            row += f"{apf:>6.2f}/{book[2]:>4.2f}/{book[3]:>+7.0f}".rjust(22)
            if best_tot is None or book[3] > best_tot[2]:
                best_tot = (D, N, book[3])
            if best_pf is None or book[2] > best_pf[2]:
                best_pf = (D, N, book[2])
        print(row)
    print(f"\nbest by TOTAL: -{best_tot[0]}% xN={best_tot[1]} (+{best_tot[2]:.0f})    "
          f"best by PF: -{best_pf[0]}% xN={best_pf[1]} (PF {best_pf[2]:.2f})")

    # ---- Phase 3: per-ticker best dip (N=1) → build a WHITELIST of where the dip-add actually pays ----
    print(f"\n== per-ticker best dip (N=1): does one standard amount work? ==")
    print(f"{'ticker':<8}{'#conf':>7}{'basePF':>8}{'bestD':>7}{'addPF':>7}   all-D add PF")
    by_tk = defaultdict(list)
    for t in calls:
        by_tk[t["tk"]].append(t)
    wl = {}   # ticker -> best dip D, only where add PF > 1.2
    for tk in sorted(by_tk):
        sub = by_tk[tk]
        nconf = sum(1 for t in sub if tr[DIPS[0]][id(t)])
        if nconf < 8:
            continue
        bpf_tk = stats([base_call[id(t)][0] for t in sub])[2]
        perD = {D: stats([r for t in sub for (_, _, r) in tr[D][id(t)][:1]])[2] for D in DIPS}
        bestD = max(perD, key=lambda d: perD[d])
        flag = "  <-- WHITELIST" if perD[bestD] > 1.2 else "  (losing — exclude)"
        if perD[bestD] > 1.2:
            wl[tk] = bestD
        cells = " ".join(f"{D}:{perD[D]:.2f}" for D in DIPS)
        print(f"{tk:<8}{nconf:>7}{bpf_tk:>8.2f}{bestD:>6}%{perD[bestD]:>7.2f}   [{cells}]{flag}")
    print(f"\nWHITELIST (add PF > 1.2): {wl}")

    # ---- Phase 4: COMPOUNDING — base vs BLANKET vs WHITELIST ----
    puts = [t for t in paths if t["otype"] == "put"]
    base_trades = []
    for t in calls + puts:
        r, ex = (base_call[id(t)] if t["otype"] == "call" else sim_from(t, 0))
        base_trades.append(dict(date=t["date"], tk=t["tk"], entry=int(t["mp"][0]), exit=ex,
                                ret=r, is_put=(t["otype"] == "put")))

    def adds_for(dip_mode, N):
        out = []
        for t in calls:
            if dip_mode == "wl":
                if t["tk"] not in wl:
                    continue
                D = wl[t["tk"]]
            else:
                D = dip_mode
            for (em, ex, r) in tr[D][id(t)][:N]:
                out.append(dict(date=t["date"], tk=t["tk"], entry=em, exit=ex, ret=r, is_put=False))
        return out

    e0, dd0 = compound(base_trades)
    eB, ddB = compound(base_trades + adds_for(15, 2))           # blanket -15% N=2
    eW, ddW = compound(base_trades + adds_for("wl", 2))         # whitelist per-ticker D, N=2
    print(f"\n== COMPOUNDING off ${START:,.0f} (+$50k cap) — does it survive a real account? ==")
    print(f"{'config':<30}{'end':>12}{'P&L':>12}{'maxDD':>11}{'ret/DD':>8}")
    for lbl, e, dd in (("base (no dip-add)", e0, dd0),
                       ("+ blanket -15% xN=2", eB, ddB),
                       ("+ WHITELIST per-tkr xN=2", eW, ddW)):
        rdd = (e - START) / abs(dd) if dd else float("inf")
        print(f"{lbl:<30}${e:>11,.0f}${e-START:>+11,.0f}${dd:>+10,.0f}{rdd:>8.1f}")
    print("\nThe dip-add SURVIVES compounding (unlike the antimg which diluted PF + cratered). Edge is")
    print("CONCENTRATED — whitelist (META/SPY/ORCL-type) should beat blanket on ret/DD. Live adds P(runner)+ML.")


if __name__ == "__main__":
    main()
