"""DCA-into-confirmed-runners backtest (Kody's idea): once a trade CONFIRMS as a runner (reached
+CONFIRM% — in live this also requires high P(runner) + ML conf), wait for a SMALL DIP and add there,
instead of chasing at the high. Compares vs the (removed) anti-martingale add that bought at +30%.

The thesis: the anti-martingale add lost because it bought HIGH (+30%). Buying the DIP after
confirmation gives the add leg a better basis → better economics. Each add is a SEPARATE own-trail
leg (V7 + profit-lock), like the deployed adds. Reports the ADD-LEG economics (does the dip-buy beat
the +30%-chase?) and the combined book. CALLS only (the runner thesis is a call/momentum play).

Confirmation here is gain-based (+30%, non-lookahead — by the add moment it HAS hit +30%). Live would
ALSO gate on P(runner)+ML conf, making it MORE selective (fewer, higher-quality adds). Read-only.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_add_handling as B  # noqa: E402
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

H = B.HAIRCUT
CONFIRM = 30.0          # runner confirmation gain
ADD_WIN = 90            # don't add later than 90min after open
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def _ok(p):
    return p is not None and not np.isnan(p) and p > 0


def confirm_idx(pp, mp, ep):
    for k in range(1, len(pp)):
        if not _ok(pp[k]):
            continue
        m = mp[k] - mp[0]
        if m < 3:
            continue
        if m > 60:
            return None
        if (pp[k] - ep) / ep * 100 >= CONFIRM:
            return k
    return None


def dip_add_idx(pp, mp, ci, dip_pct):
    """First index after confirm where premium dips dip_pct below the post-confirm running peak."""
    peak = pp[ci]
    for k in range(ci + 1, len(pp)):
        if not _ok(pp[k]):
            continue
        if mp[k] - mp[0] > ADD_WIN:
            return None
        peak = max(peak, pp[k])
        if (peak - pp[k]) / peak * 100 >= dip_pct:
            return k
    return None


def sim_from(t, start_idx):
    pp, mp, up, ets, cfg, dte, otype = (t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"])
    ep = pp[start_idx]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep,
                    entry_time=ets + timedelta(minutes=int(mp[start_idx] - mp[0])), contracts=1,
                    peak_premium=ep, entry_underlying_price=up[start_idx], dte=int(dte),
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(start_idx + 1, len(pp)):
        if not _ok(pp[k]):
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k], minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def stats(a):
    a = np.array(a, float)
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0)
    g, l = a[a > 0].sum(), -a[a < 0].sum()
    return (len(a), np.mean(a > 0) * 100, (g / l if l > 0 else float("inf")), a.sum())


def main():
    print("loading call paths (cached)...", flush=True)
    calls = [t for t in B.load_paths() if t["otype"] == "call"]
    base = [sim_from(t, 0) for t in calls]
    bn, bwr, bpf, btot = stats(base)
    print(f"\nCALL base (V7+lock): n={bn} WR={bwr:.0f}% PF={bpf:.2f} total={btot:+.0f}\n")

    print(f"{'add config':<26}{'#adds':>6}{'add WR':>8}{'add PF':>8}{'add tot':>9}   {'BOOK PF':>8}{'BOOK tot':>10}")
    print("-" * 84)

    # anti-martingale: add AT confirm (+30%, the chase) — the removed one
    amg = [sim_from(t, ci) for t in calls if (ci := confirm_idx(t["pp"], t["mp"], t["pp"][0])) is not None]
    an, awr, apf, atot = stats(amg)
    bk = stats(base + amg)
    print(f"{'antimg (+30 chase)':<26}{an:>6}{awr:>7.0f}%{apf:>8.2f}{atot:>+9.0f}   {bk[2]:>8.2f}{bk[3]:>+10.0f}")

    # DCA-on-dip: confirm +30, then add on a -DIP% pullback from the post-confirm peak
    for dip in (8, 12, 15, 20):
        adds = []
        for t in calls:
            ci = confirm_idx(t["pp"], t["mp"], t["pp"][0])
            if ci is None:
                continue
            ai = dip_add_idx(t["pp"], t["mp"], ci, dip)
            if ai is None:
                continue
            adds.append(sim_from(t, ai))
        n, wr, pf, tot = stats(adds)
        bk = stats(base + adds)
        print(f"{'dca-dip (+30, -'+str(dip)+'% pull)':<26}{n:>6}{wr:>7.0f}%{pf:>8.2f}{tot:>+9.0f}   {bk[2]:>8.2f}{bk[3]:>+10.0f}")

    print("\nadd PF = the ADD LEG's own profit factor (buying the dip should beat the +30% chase).")
    print("BOOK = base + adds, FLAT. NOTE: flat-+EV adds still hurt COMPOUNDING (over-betting) — a")
    print("winning dip-add here earns a compounding + p_runner-gated follow-up before any deploy.")


if __name__ == "__main__":
    main()
