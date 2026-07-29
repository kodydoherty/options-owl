"""V6 DCA (average-down) sweep on the flow book (2026-07-13).

Kody flagged that DCA — doubling a position when premium dips 15-35% — keeps producing the single
biggest losses (AMZN -$936: 24 contracts averaged into a falling call). This quantifies whether
averaging-down actually pays or just doubles losers, per the LIVE V6 DCA rules:
  - premium dips into [MIN_DIP, MAX_DIP]% of entry (default 15-35)
  - in the [8, 20]-min window after entry
  - only if the underlying HASN'T moved against > 0.5% (a "premium dip without a thesis break")
  - then DOUBLES the position, blending the basis.

Model (per trade, from the cached raw paths): run the prod FSM twice — once no-DCA (baseline, 1 unit
from entry) and once WITH the add (2 units, blended basis, FSM re-based at the dip). Compare absolute
P&L AND capital-normalized ROI (DCA deploys 2× on a triggered trade — the honest denominator). Split
by call/put, 0DTE/multi-day, and whitelist vs all tickers. Flat-$750/unit, same trades. Bounded by
thetadata (≤2026-07-01).

Usage: python scripts/dca_sweep.py [--min-dip 15 --max-dip 35 --all-tickers]
"""
import argparse
import pickle
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import flow_gold_standard_report as R  # noqa: E402
from exit_risk_sweep import CACHE, _base_cfg, _mk_settings, HC, BASE  # noqa: E402

WHITELIST = {"IWM", "SPY", "QQQ", "AMZN", "NVDA"}  # V6_DCA_TICKERS
WIN_MIN, WIN_MAX = 8.0, 20.0
U_THRESH = 0.5


def _fsm_exit(t, start_k, basis, settings):
    """Run the FSM from index start_k with entry_premium=basis; return exit PRICE (pre-haircut)."""
    cfg = _base_cfg(t["tk"], t["otype"])
    pp, mp, up = t["pp"], t["mp"], t["up"]
    ets0 = t["ets"] + timedelta(minutes=int(mp[start_k] - mp[0]))
    fsm = R.ExitFSM(cfg, settings=settings)
    st = R.TradeState(trade_id=1, ticker=t["tk"], option_type=t["otype"], entry_premium=basis,
                      entry_time=ets0, contracts=1, peak_premium=basis,
                      entry_underlying_price=up[start_k], dte=t["dte0"],
                      expiry_date=t["ets"].strftime("%Y-%m-%d"))
    last = pp[start_k]
    for k in range(start_k + 1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        now = ets0 + timedelta(minutes=int(mp[k] - mp[start_k]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        act = fsm.evaluate(st, prem, prem * (1 - HC), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            return prem
    return last


def dca_trigger(t, min_dip, max_dip, whitelist_only):
    """Return the index of the first V6-DCA trigger, or None."""
    if whitelist_only and t["tk"] not in WHITELIST:
        return None
    pp, mp, up, ep = t["pp"], t["mp"], t["up"], t["ep"]
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        elapsed = mp[k] - mp[0]
        if elapsed < WIN_MIN:
            continue
        if elapsed > WIN_MAX:
            return None  # past the window
        dip = (ep - prem) / ep * 100
        if not (min_dip <= dip <= max_dip):
            continue
        u_move = (up[k] - up[0]) / up[0] * 100 if up[0] else 0.0
        if t["otype"] == "call" and u_move < -U_THRESH:
            continue  # underlying against the call — thesis broken, blocked
        if t["otype"] == "put" and u_move > U_THRESH:
            continue
        return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dip", type=float, default=15.0)
    ap.add_argument("--max-dip", type=float, default=35.0)
    ap.add_argument("--all-tickers", action="store_true", help="ignore the whitelist")
    args = ap.parse_args()

    trades = pickle.load(open(CACHE, "rb"))
    S = _mk_settings()
    wl_only = not args.all_tickers

    rows = []  # (t, triggered, pnl_base, pnl_dca, cap_base, cap_dca)
    for t in trades:
        x_a = _fsm_exit(t, 0, t["ep"], S)
        pnl_base = BASE * (x_a * (1 - HC) - t["ep"]) / t["ep"]
        k = dca_trigger(t, args.min_dip, args.max_dip, wl_only)
        if k is None:
            rows.append((t, False, pnl_base, pnl_base, BASE, BASE))
            continue
        p_dip = float(t["pp"][k])
        blend = (t["ep"] + p_dip) / 2
        x_b = _fsm_exit(t, k, blend, S)
        # 2 units: one bought at ep, one at the dip; both exit at x_b (haircut on exit)
        exitp = x_b * (1 - HC)
        pnl_dca = BASE * (exitp - t["ep"]) / t["ep"] + BASE * (exitp - p_dip) / p_dip
        rows.append((t, True, pnl_base, pnl_dca, BASE, 2 * BASE))

    def rep(label, rs):
        if not rs:
            print(f"  {label:<30} (none)"); return
        pb = sum(r[2] for r in rs); pd = sum(r[3] for r in rs)
        cb = sum(r[4] for r in rs); cd = sum(r[5] for r in rs)
        trig = sum(1 for r in rs if r[1])
        print(f"  {label:<30}{len(rs):>4} tr ({trig:>3} DCA'd)  "
              f"noDCA ${pb:>+8,.0f} (ROI {pb/cb*100:>+5.1f}%)  "
              f"DCA ${pd:>+8,.0f} (ROI {pd/cd*100:>+5.1f}%)  Δ${pd-pb:>+8,.0f}")

    print(f"\nV6 DCA sweep — dip [{args.min_dip:.0f},{args.max_dip:.0f}]%, "
          f"{'ALL tickers' if args.all_tickers else 'whitelist only'}, "
          f"{len(trades)} trades\n")
    print("(noDCA = current 1-unit book; DCA = double on trigger. ROI normalizes for the 2× capital"
          " DCA deploys — the honest comparison, since P&L alone rewards just betting more.)\n")
    trig_rows = [r for r in rows if r[1]]
    rep("ALL trades", rows)
    rep("  DCA-TRIGGERED only", trig_rows)
    rep("    triggered CALLS", [r for r in trig_rows if r[0]["otype"] == "call"])
    rep("    triggered PUTS", [r for r in trig_rows if r[0]["otype"] == "put"])
    rep("    triggered 0DTE", [r for r in trig_rows if r[0]["dte0"] == 0])
    rep("    triggered multi-day", [r for r in trig_rows if r[0]["dte0"] > 0])
    print("\nVERDICT: on the TRIGGERED subset, if DCA ROI < noDCA ROI, averaging-down is destroying")
    print("capital efficiency (doubling into losers). If DCA absolute P&L < noDCA too, it's pure loss.")


if __name__ == "__main__":
    main()
