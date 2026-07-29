"""Early small-win floor sweep (2026-07-14).

Kody's pain: "goes up a little, dives negative, never recovers." Our protection gates arm too HIGH to
catch it — profit-lock +25%, breakeven ratchet +20%, scaleout +20%. A trade that peaks at +10-15% has NO
floor and round-trips through breakeven to the -25% hardstop. This tests adding a LOW early floor: once a
trade peaks >= ARM%, if it falls back to <= FLOOR%, exit there (lock a small win / scratch) instead of
letting it die.

The tension: a tight early floor ALSO clips dip-then-rip winners. So we measure BOTH:
  - SAVED: baseline trades that peaked >= ARM but closed <= 0 (the "green then dead" losers) — how many
    become small wins, and the P&L recovered.
  - CLIPPED: winners that dipped to FLOOR after ARM then would have run — the runner cost (best-days delta).

No lookahead: floor uses only the premium path up to each minute. Baseline = prod FSM (sim_trade, incl the
+25% profit-lock + -25% hardstop). Flow book. Reports net P&L, WR, and the conversion breakdown.

Usage: python scripts/early_lock_sweep.py [--since 2026-03-01]
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from exit_risk_sweep import BASE, CACHE, HC, _mk_settings  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402

WORST = {"2026-06-15", "2026-06-10", "2026-04-02", "2026-05-14", "2026-05-11",
         "2026-04-15", "2026-03-31", "2026-04-06", "2026-05-07", "2026-05-15"}
BEST = {"2026-05-26", "2026-05-08", "2026-06-25", "2026-05-06", "2026-04-21",
        "2026-03-26", "2026-04-23", "2026-05-28", "2026-06-09", "2026-06-11"}


def early_floor(t, base_ret, base_xm, arm, floor):
    """Return (ret, fired) applying an early small-win floor before the baseline exit."""
    pp, mp, ep = t["pp"], t["mp"], t["ep"]
    peak = 0.0
    armed = False
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        if int(mp[k]) >= base_xm:
            break
        gain = (prem / ep - 1) * 100
        peak = max(peak, gain)
        if peak >= arm:
            armed = True
        if armed and gain <= floor:
            return (prem * (1 - HC) - ep) / ep * 100, True
    return base_ret, False


def peak_gain(t, xm):
    pp, mp, ep = t["pp"], t["mp"], t["ep"]
    pk = 0.0
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        if int(mp[k]) >= xm:
            break
        pk = max(pk, (prem / ep - 1) * 100)
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    args = ap.parse_args()
    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    S = _mk_settings()

    base = []  # (t, ret, xm, peak)
    for t in trades:
        r, xm = sim_trade(t, S)
        base.append((t, r, xm, peak_gain(t, xm)))

    def pnl(rets):
        return sum(BASE * x / 100 for x in rets)

    b_all = pnl([r for _, r, _, _ in base])
    b_win = sum(1 for _, r, _, _ in base if r > 0)
    n = len(base)
    # the target population: "green then dead" — peaked >=10% but closed <=0
    grn_dead = [(t, r) for t, r, _, pk in base if pk >= 10 and r <= 0]
    gd_loss = pnl([r for _, r in grn_dead])
    print(f"\nEarly small-win floor sweep — {n} flow trades since {args.since}")
    print(f"BASELINE: ${b_all:+,.0f}  WR {b_win}/{n}={100*b_win/n:.0f}%")
    print(f"  'GREEN THEN DEAD' (peaked >=+10% but closed <=0): {len(grn_dead)} trades, "
          f"${gd_loss:+,.0f} — the population we're trying to rescue\n")

    print(f"  {'arm%':>5}{'floor%':>7}{'P&L':>11}{'Δ P&L':>9}{'WR':>7}{'fired':>7}"
          f"{'saved$':>9}{'best-days Δ':>13}")
    b_best = pnl([r for t, r, _, _ in base if t["date"] in BEST])
    for arm in (8, 10, 12, 15, 20):
        for floor in (0.0, 3.0, 5.0):
            rows, fired = [], 0
            for t, r, xm, pk in base:
                nr, f = early_floor(t, r, xm, arm, floor)
                rows.append((t, nr))
                if f:
                    fired += 1
            a_all = pnl([r for _, r in rows])
            a_win = sum(1 for _, r in rows if r > 0)
            a_best = pnl([r for t, r in rows if t["date"] in BEST])
            # $ recovered on the green-then-dead population specifically
            saved = 0.0
            for (t, br) in grn_dead:
                b_ret = next(rr for tt, rr, _, _ in base if tt is t)
                n_ret, _ = early_floor(t, b_ret, next(x for tt, _, x, _ in base if tt is t), arm, floor)
                saved += BASE * (n_ret - b_ret) / 100
            print(f"  {arm:>5}{floor:>7.0f}{f'${a_all:+,.0f}':>11}{f'{a_all-b_all:+,.0f}':>9}"
                  f"{f'{100*a_win/n:.0f}%':>7}{fired:>7}{f'${saved:+,.0f}':>9}"
                  f"{f'${a_best-b_best:+,.0f}':>13}")

    print("\nREAD: Δ P&L>0 with WR up = converts round-trip losers to small wins net-positive. best-days Δ<0")
    print("= runner clip cost (the winners we stopped early). Ships if Δ P&L>0 AND runner clip acceptable.")


if __name__ == "__main__":
    main()
