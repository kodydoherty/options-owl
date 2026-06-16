"""OUT-OF-SAMPLE test of Kody's refinement: wait for STRONGER runner confirmation (+30/+50/+80%)
before DCAing, and focus on SPY / TSLA (big liquid names). Tested on the TEST half only (data the
choice never saw). A higher confirm bar = only add to trades that REALLY proved they run.

If a higher confirm lifts the OOS book PF above base, there's a real, non-overfit signal. If not,
the dip-add doesn't survive on this 60d sample and needs more data + the P(runner)+ML model gate.
Read-only, cached.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402
import backtest_dca_dip_sweep as S  # noqa: E402


def dip_legs_c(t, D, confirm, maxn=2):
    pp, mp, ep = t["pp"], t["mp"], t["pp"][0]
    ci = None
    for k in range(1, len(pp)):
        if not S._ok(pp[k]):
            continue
        m = mp[k] - mp[0]
        if m < 3:
            continue
        if m > 60:
            break
        if (pp[k] - ep) / ep * 100 >= confirm:
            ci = k
            break
    if ci is None:
        return []
    legs, peak, k = [], pp[ci], ci + 1
    while k < len(pp) and len(legs) < maxn:
        if not S._ok(pp[k]):
            k += 1
            continue
        if mp[k] - mp[0] > S.ADD_WIN:
            break
        peak = max(peak, pp[k])
        if (peak - pp[k]) / peak * 100 >= D:
            legs.append(S.sim_from(t, k)[0])
            peak = pp[k]
        k += 1
    return legs


def main():
    print("loading + splitting (test = second half, unseen)...", flush=True)
    calls = [t for t in B.load_paths() if t["otype"] == "call"]
    dates = sorted({t["date"] for t in calls})
    cut = dates[len(dates) // 2]
    test = [t for t in calls if t["date"] >= cut]
    print(f"TEST {cut}..{dates[-1]} (n={len(test)})\n")

    groups = (("ALL", test), ("SPY", [t for t in test if t["tk"] == "SPY"]),
              ("TSLA", [t for t in test if t["tk"] == "TSLA"]),
              ("META", [t for t in test if t["tk"] == "META"]))
    print(f"{'group':<7}{'base PF':>9}   confirm x dip → book PF (#adds)  [OOS]")
    print("-" * 74)
    for name, sub in groups:
        if len(sub) < 8:
            continue
        base = S.stats([S.sim_from(t, 0)[0] for t in sub])
        base_rets = [S.sim_from(t, 0)[0] for t in sub]
        line = f"{name:<7}{base[2]:>9.2f}   "
        cells = []
        for confirm in (30, 50, 80):
            for D in (15, 20):
                adds = [r for t in sub for r in dip_legs_c(t, D, confirm, 2)]
                bk = S.stats(base_rets + adds)
                cells.append(f"+{confirm}/-{D}:{bk[2]:.2f}({len(adds)})")
        print(line + "  ".join(cells))
    print("\nREAD: book PF must beat the group's base PF on this UNSEEN half to be a real edge. Higher")
    print("confirm = fewer, surer adds. If nothing beats base OOS, the idea needs MORE flow history +")
    print("the P(runner)+ML gate (validated on 3.3M snapshots, not 60d) — don't deploy on 60d tuning.")


if __name__ == "__main__":
    main()
