"""Per-ticker OOS scorecard — did we whitelist/block any ticker WRONGLY?

For every CALL ticker in the cache: base PF and dip-add book PF, on TRAIN (first half) vs TEST
(unseen second half). A ticker only deserves a verdict if it holds OOS:
  - TRADE-worthy: TEST base PF > 1.0 (the base trade has edge on unseen data).
  - DIP-worthy:   TEST dip book PF > TEST base PF (adding helps OOS, not just in-sample).
  - OVERFIT:      strong on TRAIN, collapses on TEST.
This catches both false-positives (whitelisted on noise) and false-negatives (blocked something real).
NOTE: only tickers already in the traded universe are in the cache — truly-blocked names need a fresh
flow fetch to test. Read-only, cached.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402
import backtest_dca_dip_sweep as S  # noqa: E402
from backtest_dca_confirm_oos import dip_legs_c  # noqa: E402


def main():
    print("loading + splitting...", flush=True)
    calls = [t for t in B.load_paths() if t["otype"] == "call"]
    dates = sorted({t["date"] for t in calls})
    cut = dates[len(dates) // 2]
    train = defaultdict(list)
    test = defaultdict(list)
    for t in calls:
        (train if t["date"] < cut else test)[t["tk"]].append(t)

    def pf(sub, dip=False):
        if not sub:
            return (0, 0.0)
        base = [S.sim_from(t, 0)[0] for t in sub]
        if not dip:
            return (len(base), S.stats(base)[2])
        adds = [r for t in sub for r in dip_legs_c(t, 15, 30, 2)]
        return (len(base), S.stats(base + adds)[2])

    print(f"\nCALL tickers — TRAIN ({dates[0]}..{cut}) vs TEST ({cut}..{dates[-1]}), -15%/+30/N=2 dip\n")
    print(f"{'ticker':<7}{'nTr':>5}{'nTe':>5}{'baseTr':>8}{'baseTe':>8}{'dipTr':>8}{'dipTe':>8}   verdict")
    print("-" * 76)
    rows = []
    for tk in sorted(set(train) | set(test)):
        nTr, bTr = pf(train[tk])
        nTe, bTe = pf(test[tk])
        _, dTr = pf(train[tk], dip=True)
        _, dTe = pf(test[tk], dip=True)
        rows.append((tk, nTr, nTe, bTr, bTe, dTr, dTe))
    for tk, nTr, nTe, bTr, bTe, dTr, dTe in sorted(rows, key=lambda r: -r[4]):
        if nTe < 5:
            verdict = "thin (need data)"
        elif bTe < 1.0:
            verdict = "BASE loses OOS — weak ticker"
        elif dTe > bTe + 0.05:
            verdict = "DIP HOLDS OOS *"
        elif dTr > bTr + 0.2 and dTe <= bTe:
            verdict = "dip OVERFIT (train-only)"
        else:
            verdict = "trade base, no dip edge"
        print(f"{tk:<7}{nTr:>5}{nTe:>5}{bTr:>8.2f}{bTe:>8.2f}{dTr:>8.2f}{dTe:>8.2f}   {verdict}")
    print("\n* = the only tickers where ADDING on the dip survives out-of-sample. Everything else: trade")
    print("the base, skip the dip-add. Blocked-from-trading names aren't in the cache — fetch to test them.")


if __name__ == "__main__":
    main()
