"""Why did P&L drop April->June? Per-month diagnostic of the BASE strategy (clean backtest, no bugs).
Decomposes each month into the drivers: win rate, profit factor, avg winner vs avg loser, and — the
key one for a runner-hunting strategy — RUNNER FREQUENCY (% of trades that peaked >50% / >100%).
If June dropped because fewer big runners showed up, that's a market regime, not a broken strategy.
Read-only, cached.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402


def main():
    print("loading paths (cached)...", flush=True)
    trades = B.load_paths()
    rows = []
    for t in trades:
        res = B.run_fsm(t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"])
        rows.append((t["date"][:7], t["otype"], res[2], res[3]))   # [2]=ret% [3]=peak%

    by = defaultdict(list)
    for mo, ot, r, peak in rows:
        by[mo].append((ot, r, peak))

    print(f"\n{'month':<9}{'n':>5}{'WR':>6}{'PF':>7}{'mean%':>8}{'avgWin':>8}{'avgLoss':>9}"
          f"{'run>50%':>9}{'run>100%':>10}")
    print("-" * 71)
    for mo in sorted(by):
        a = by[mo]
        rets = np.array([r for _, r, _ in a])
        peaks = np.array([p for _, _, p in a])
        g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
        pf = g / l if l > 0 else float("inf")
        win = rets[rets > 0].mean() if (rets > 0).any() else 0
        loss = rets[rets < 0].mean() if (rets < 0).any() else 0
        print(f"{mo:<9}{len(a):>5}{np.mean(rets > 0)*100:>5.0f}%{pf:>7.2f}{rets.mean():>+8.1f}"
              f"{win:>+8.0f}{loss:>+9.0f}{np.mean(peaks >= 50)*100:>8.0f}%{np.mean(peaks >= 100)*100:>9.0f}%")

    # call vs put per month (which side decayed?)
    print(f"\n{'month':<9}{'CALL PF':>9}{'CALL n':>8}{'PUT PF':>9}{'PUT n':>8}")
    print("-" * 43)
    for mo in sorted(by):
        for side in ("call", "put"):
            pass
        c = np.array([r for ot, r, _ in by[mo] if ot == "call"])
        p = np.array([r for ot, r, _ in by[mo] if ot == "put"])
        def _pf(x):
            return (x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else float("inf")
        print(f"{mo:<9}{(_pf(c) if len(c) else 0):>9.2f}{len(c):>8}{(_pf(p) if len(p) else 0):>9.2f}{len(p):>8}")

    print("\nRUNNER FREQ is the tell: a runner-hunting book lives on the >50%/>100% trades. If those dried")
    print("up in June, the edge didn't break — the MARKET stopped handing out the big moves (regime).")


if __name__ == "__main__":
    main()
