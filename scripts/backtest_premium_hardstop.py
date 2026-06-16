"""0DTE premium HARD-STOP backtest — fix for the NVDA #34 case (premium melted -42% while the stock
barely moved; the underlying-aware FSM held it on the wide 65% backstop).

Adds a premium-based hard floor: for 0DTE trades only, if the contract is down >= X% FROM ENTRY,
exit immediately regardless of the underlying. (From entry, not peak — trades that worked are already
protected by the breakeven ratchet / trail; this only catches the ones that never worked.) Layered
ON TOP of the V7 wide trail + CALL profit-lock (current deployed config).

Sweeps X = off / 20 / 25 / 30%. The whole risk is WHIPSAW — cutting trades that dip then rip — so we
report P&L, PF, WR AND the tail (avg of the worst 5% of trades) to see if it caps disasters without
killing winners. CALL and PUT shown separately; only 0DTE trades are affected. Read-only, cached.
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
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def sim_hs(t, hs):
    """Exit return with an optional 0DTE premium hard-stop at -hs% from entry (0 = off)."""
    pp, mp, up, ets, cfg, dte, otype = (t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"])
    ep = pp[0]
    is_0dte = int(dte) == 0
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=int(dte),
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        if hs and is_0dte and (prem - ep) / ep * 100 <= -hs:        # premium hard-stop
            return (prem * (1 - H) - ep) / ep * 100, True
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        a = fsm.evaluate(st, prem, prem * (1 - H), prem, now,
                         current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if a.should_exit:
            return (prem * (1 - H) - ep) / ep * 100, False
    return (last * (1 - H) - ep) / ep * 100, False


def stats(rets):
    a = np.array(rets, float)
    if len(a) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    g, l = a[a > 0].sum(), -a[a < 0].sum()
    pf = g / l if l > 0 else float("inf")
    tail = np.mean(np.sort(a)[:max(1, len(a) // 20)])   # avg of worst 5%
    return (len(a), np.mean(a > 0) * 100, pf, a.sum(), a.mean(), tail)


def main():
    print("loading paths (cached)...", flush=True)
    trades = B.load_paths()
    for dname in ("call", "put"):
        ts = [t for t in trades if t["otype"] == dname]
        n0 = sum(1 for t in ts if int(t["dte"]) == 0)
        print(f"\n================ {dname.upper()}  (n={len(ts)}, 0DTE={n0}) ================")
        print(f"{'hard-stop':<12}{'n':>6}{'WR':>7}{'PF':>7}{'total%':>9}{'mean%':>8}{'worst5%':>9}{'#cut':>6}")
        print("-" * 64)
        for hs in (0, 20, 25, 30):
            rets, cut = [], 0
            for t in ts:
                r, was_cut = sim_hs(t, hs)
                rets.append(r); cut += was_cut
            n, wr, pf, tot, mean, tail = stats(rets)
            name = "baseline" if hs == 0 else f"-{hs}% 0DTE"
            print(f"{name:<12}{n:>6}{wr:>6.0f}%{pf:>7.2f}{tot:>+9.0f}{mean:>+8.1f}{tail:>+9.0f}{cut:>6}")
    print("\nworst5% = avg return of the worst 5% of trades (the disaster tail — NVDA #34 lived here).")
    print("A hard-stop WINS if it lifts PF + total AND shrinks the worst5% tail. It LOSES (whipsaw) if")
    print("PF/total drop (cutting dips-that-rip). 0DTE-only; multi-day untouched.")


if __name__ == "__main__":
    main()
