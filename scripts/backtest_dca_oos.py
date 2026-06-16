"""OUT-OF-SAMPLE test for the dip-add — is the whitelist real or overfit to our data?

Split the 60d into TRAIN (first half) and TEST (second half, never seen). Derive the ticker whitelist
+ per-ticker dip ON TRAIN ONLY, then evaluate on TEST. If the edge holds OOS it's real; if it
collapses it was overfit to small per-ticker samples. Compares vs a BLANKET dip (-15% N=2, one
parameter, hard to overfit) and base. The blanket is the robust / less-limiting alternative.
Read-only, cached.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402
import backtest_dca_dip_sweep as S  # noqa: E402  (sim_from, dip_legs, stats, DIPS)


def base_ret(t):
    return S.sim_from(t, 0)[0]


def add_rets(trset, dlegs, dip_for, N):
    """dip_for: callable tk->D (or None to skip). Returns list of add-leg returns."""
    out = []
    for t in trset:
        D = dip_for(t["tk"])
        if D is None:
            continue
        out.extend(r for (_, _, r) in dlegs[D][id(t)][:N])
    return out


def main():
    print("loading + splitting (train=first half, test=second half)...", flush=True)
    calls = [t for t in B.load_paths() if t["otype"] == "call"]
    dates = sorted({t["date"] for t in calls})
    cut = dates[len(dates) // 2]
    train = [t for t in calls if t["date"] < cut]
    test = [t for t in calls if t["date"] >= cut]
    print(f"train {dates[0]}..{cut} (n={len(train)}) | test {cut}..{dates[-1]} (n={len(test)})\n")

    # precompute dip legs (up to 2 tranches) per D for both sets
    dl_tr = {D: {id(t): S.dip_legs(t, D, 2) for t in train} for D in S.DIPS}
    dl_te = {D: {id(t): S.dip_legs(t, D, 2) for t in test} for D in S.DIPS}

    # --- derive whitelist + per-ticker dip ON TRAIN ONLY ---
    by_tk = defaultdict(list)
    for t in train:
        by_tk[t["tk"]].append(t)
    wl = {}
    for tk, sub in by_tk.items():
        nconf = sum(1 for t in sub if dl_tr[S.DIPS[0]][id(t)])
        if nconf < 6:
            continue
        perD = {D: S.stats([r for t in sub for (_, _, r) in dl_tr[D][id(t)][:1]])[2] for D in S.DIPS}
        bestD = max(perD, key=lambda d: perD[d])
        if perD[bestD] > 1.2:
            wl[tk] = bestD
    print(f"WHITELIST learned on TRAIN: {wl}\n")

    def report(name, trset, dlegs):
        b = S.stats([base_ret(t) for t in trset])
        blanket = S.stats([base_ret(t) for t in trset]
                          + add_rets(trset, dlegs, lambda tk: 15, 2))
        whitel = S.stats([base_ret(t) for t in trset]
                         + add_rets(trset, dlegs, lambda tk: wl.get(tk), 2))
        print(f"  {name:<8}base PF {b[2]:.2f} (+{b[3]:.0f})   "
              f"blanket-15 PF {blanket[2]:.2f} (+{blanket[3]:.0f})   "
              f"whitelist PF {whitel[2]:.2f} (+{whitel[3]:.0f})")

    print("book PF (+total) — base vs blanket vs train-derived whitelist:")
    report("TRAIN", train, dl_tr)
    report("TEST", test, dl_te)

    # --- per-MONTH (different time periods) — is the edge consistent or regime-dependent? ---
    print("\nper-MONTH (does the edge hold across different periods?):")
    dl_all = {D: {**dl_tr[D], **dl_te[D]} for D in S.DIPS}
    by_mo = defaultdict(list)
    for t in calls:
        by_mo[t["date"][:7]].append(t)
    for mo in sorted(by_mo):
        report(mo, by_mo[mo], dl_all)

    print("\nREAD: if whitelist TEST/each-month PF ~ TRAIN PF and still beats base -> real edge. If it")
    print("collapses toward/below base while blanket holds -> OVERFIT; prefer the BLANKET (one param) OR")
    print("gate live by P(runner)+ML (a validated model, not in-sample tickers). Cache is only 60d — a")
    print("truly long test needs fetching more flow history (uw_ticker_discovery).")


if __name__ == "__main__":
    main()
