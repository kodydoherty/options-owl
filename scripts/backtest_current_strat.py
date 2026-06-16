"""Clean backtest of the CURRENTLY DEPLOYED strategy — no bugs, fixed bet size (edge measure, not
fantasy compounding). The live last-2-weeks P&L is bug-polluted (flow didn't trade until 2026-06-15,
webull rejects, GEX error, scan timeouts — all fixed this session), so the only honest read of the
current strat is the backtest.

Deployed config: flow (conviction-sized, OTM strike) + ML, V7 wide-trail exits + CALL profit-lock,
anti-martingale adds OFF, tide gate on (already baked into the cached flow returns). Fixed $750/trade
so weeks are comparable. Reports per ISO-week PF/WR/net, and SPY broken out (the missed-mover).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v7_antimg_compound as C  # noqa: E402

BET = 750.0  # fixed $ per trade — edge measure, not compounding


def stats(rets):
    a = np.array(rets, dtype=float)
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0)
    g, l = a[a > 0].sum(), -a[a < 0].sum()
    pf = g / l if l > 0 else float("inf")
    net = BET * a.sum() / 100.0
    return (len(a), np.mean(a > 0) * 100, pf, net)


def line(name, rets):
    n, wr, pf, net = stats(rets)
    return f"{name:<14}{n:>6}{wr:>6.0f}%{pf:>8.2f}{net:>+12,.0f}"


def main():
    print("building CURRENT-STRAT trade set (profit-lock ON, adds OFF, fixed $750/trade)...", flush=True)
    fl = C.flow_paths()
    fl["ret"] = fl["ret_lk"]  # current strat uses the profit-locked exit for flow
    ml = pd.read_csv("journal/v3_eval_results/v7_core_trades.csv")
    mlt = pd.DataFrame({"date": ml["day"].astype(str), "src": "ML", "tk": ml.get("ticker", "?"),
                        "ret": ml["pnl_pct"], "is_put": ml["direction"].str.lower() == "put"})
    allt = pd.concat([fl[["date", "src", "tk", "ret", "is_put"]], mlt], ignore_index=True)
    allt["dt"] = pd.to_datetime(allt["date"])
    allt["wk"] = allt["dt"].dt.strftime("%G-W%V")
    maxd = allt["dt"].max()
    win = allt[allt["dt"] >= maxd - pd.Timedelta(days=60)].copy()
    print(f"window {(maxd - pd.Timedelta(days=60)).date()} → {maxd.date()} | {len(win)} trades\n")

    print(f"{'period':<14}{'n':>6}{'WR':>8}{'PF':>8}{'net@$750':>12}")
    print("-" * 48)
    for wk in sorted(win["wk"].unique()):
        print(line(wk, win[win["wk"] == wk]["ret"]))
    print("-" * 48)
    print(line("ALL 60d", win["ret"]))
    print(line("  CALLs", win[~win["is_put"]]["ret"]))
    print(line("  PUTs", win[win["is_put"]]["ret"]))
    print(line("  SPY only", win[win["tk"] == "SPY"]["ret"]))

    print("\nlast 14d (the bug-polluted live window, here CLEAN in backtest):")
    l14 = win[win["dt"] >= maxd - pd.Timedelta(days=14)]
    print(line("  clean 14d", l14["ret"]))
    print("\nNOTE: backtest is bug-free + fixed $750/trade. Live last-14d (kody) was -$926 call / "
          "-$921 put — the GAP vs this is the bugs (now fixed). Cached flow ends ~06-12.")


if __name__ == "__main__":
    main()
