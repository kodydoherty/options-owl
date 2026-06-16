"""Fleet staggering backtest — do the 5 bots STEP ON EACH OTHER (crowd the same fills) and LOSE
THEIR ASS AT THE SAME TIME (synchronized drawdowns)? And does priority-rotated staggering fix it?

Priority order (user): kody > adam > dennis > yank > vinny. Each signal is assigned to K bots by a
priority round-robin (rotating start), so K=5 = everyone takes every signal (today, fully correlated)
and K<5 = each signal goes to only K bots → they hold DIFFERENT books → decorrelated + less crowding.
crowd=K feeds the measured size-impact slippage (K bots stacking one contract).

Reports, per K: fleet P&L, fleet max drawdown (the 'lose our ass together' number), avg pairwise
daily-P&L correlation (lower = more decorrelated), worst single fleet day, and fleet ret/DD.
All bots use the $50k take-profit cap. Read-only, cached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_realistic_account as R  # noqa: E402

BOTS = [("kody", 23000.0), ("adam", 4685.0), ("dennis", 10000.0), ("yank", 3600.0), ("vinny", 3123.0)]
CAP = 50000.0
NB = len(BOTS)


def sim_bot(trades, start, crowd):
    """Event-driven single account on its assigned trades (sizing, $50k cap, crowd slippage)."""
    bal, banked, daily = start, 0.0, {}
    for d, g in trades.groupby("date", sort=True):
        g = g.sort_values("entry")
        deployable, per_slot, capd = bal * R.RISK_PCT, bal * R.RISK_PCT / R.MAX_CONC, bal * R.POS_CAP
        open_pos, committed, day = [], 0.0, 0.0
        for t in g.itertuples():
            still = []
            for ex, cm, r in open_pos:
                if ex <= t.entry:
                    day += cm * r / 100.0; committed -= cm
                else:
                    still.append((ex, cm, r))
            open_pos = still
            lc = R.liq_cap(t.tk)
            size = min(per_slot * t.mult * (R.PUT_BUDGET if t.is_put else 1.0) * R.FLAT, capd, lc)
            if len(open_pos) >= R.MAX_CONC or committed + size > deployable:
                continue
            open_pos.append((t.exit, size, R.slip(t.ret, size, t.tk, lc, crowd)))
            committed += size
        for ex, cm, r in open_pos:
            day += cm * r / 100.0
        daily[d] = daily.get(d, 0.0) + day
        bal += day
        if CAP and bal > CAP:
            banked += bal - CAP; bal = CAP
    return daily, bal + banked - start


def fleet(trades, K):
    t = trades.sort_values(["date", "entry"]).reset_index(drop=True)
    assign = {i: [] for i in range(NB)}
    for i in range(len(t)):
        for j in range(K):                       # priority round-robin, rotating start
            assign[(i + j) % NB].append(i)
    bot_daily, bot_pnl = [], []
    for bi, (_, sz) in enumerate(BOTS):
        daily, pnl = sim_bot(t.iloc[assign[bi]], sz, crowd=K)
        bot_daily.append(daily); bot_pnl.append(pnl)
    days = sorted(set().union(*[set(d) for d in bot_daily]))
    fleet_daily = {d: sum(bd.get(d, 0.0) for bd in bot_daily) for d in days}
    M = np.array([[bd.get(d, 0.0) for d in days] for bd in bot_daily])
    corrs = [np.corrcoef(M[a], M[b])[0, 1] for a in range(NB) for b in range(a + 1, NB)
             if M[a].std() > 0 and M[b].std() > 0]
    return (sum(bot_pnl), R.maxdd(fleet_daily), np.mean(corrs) if corrs else 0.0,
            min(fleet_daily.values()), bot_pnl)


def main():
    print("building fleet trade set (cached)...", flush=True)
    trades, maxd = R.build()
    cap_tot = sum(s for _, s in BOTS)
    print(f"window ends {maxd.date()} | {len(trades)} signals | fleet capital ${cap_tot:,.0f} "
          f"(kody>adam>dennis>yank>vinny), $50k cap each\n")
    print(f"{'scheme':<24}{'fleet P&L':>11}{'fleet DD':>11}{'corr':>7}{'worst day':>11}{'ret/DD':>8}")
    print("-" * 72)
    for K in (5, 3, 2, 1):
        pnl, dd, corr, worst, _ = fleet(trades, K)
        lbl = "K=5 IDENTICAL (today)" if K == 5 else f"K={K} (each sig -> {K} bots)"
        rdd = pnl / abs(dd) if dd else float("inf")
        print(f"{lbl:<24}${pnl:>10,.0f}${dd:>+10,.0f}{corr:>7.2f}${worst:>+10,.0f}{rdd:>8.1f}")
    print("\ncorr = avg pairwise daily-P&L correlation across bots (1.0 = all move together = lose-ass-"
          "together risk). Lower K = more decorrelated + less crowding, but fewer total bets (lower P&L).")
    print("The knee trades a bit of fleet P&L for a big cut in synchronized drawdown.")


if __name__ == "__main__":
    main()
