"""REALISTIC account projection — what could kody's actual $23k become in 60 days, with the things
the fantasy compound ($2.1M) and the flat-$750 ($52k) both ignore:

  • REAL capacity: event-driven 8 concurrent slots + 75% deployable CAPITAL. A trade holds its
    capital from entry minute to exit minute; a new signal is SKIPPED if all 8 slots are full OR
    there isn't enough free capital. So you do NOT take all ~33 signals/day — you take what fits.
  • Account-scaled sizing: per_slot = bal×75%/8, ×0.85 flat budget, ×conviction mult, ×0.50 for puts,
    capped at 15% of balance and $50k. Compounds off $23k.
  • Slippage BOTH sides: entry pays up ENTRY_SLIP (you buy the ask, not mid); exit already haircut 3%.
  • Liquidity: a per-trade contract cap. At $23k the 15% cap ($3.5k) is the binding constraint, NOT
    chain liquidity — liquidity only bites at much larger accounts (that's the crowding/fantasy point).

Deployed strat: flow conviction + V7 wide trail + CALL profit-lock, adds OFF. Reports the $23k end
balance + the capacity stats (how many signals were skipped for slots/capital). Read-only, cached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v7_antimg_compound as C  # noqa: E402

START = 23000.0
RISK_PCT, MAX_CONC, POS_CAP, PUT_BUDGET, FLAT = 0.75, 8, 0.15, 0.50, 0.85
ENTRY_SLIP = 0.03   # base: buy the ask, not mid (~half a 6% 0DTE spread). exit already haircut 3%.
DAYS = 60

# Per-trade LIQUIDITY ceiling — the most $ you can push into ONE 0DTE contract without serious
# market impact. THE realism lever: once the account is big enough that 15% > this, per-trade size
# plateaus here and compounding goes from geometric to ~linear. Conservative for a mixed universe.
LIQ = {"SPY": 60000.0, "QQQ": 60000.0}
LIQ_BIG = {"AMZN", "META", "TSLA", "NVDA", "GOOG", "AMD", "AVGO"}  # $20k
LIQ_DEFAULT = 8000.0  # thin names (MU/ARM/LRCX/INTC/ORCL/AAPL/...) — small fillable size


def liq_cap(tk):
    if tk in LIQ:
        return LIQ[tk]
    return 20000.0 if tk in LIQ_BIG else LIQ_DEFAULT


def slip(ret_pct, size, cap):
    """Entry slippage that GROWS with order size vs the contract's liquidity (impact), on a return
    already net of the 3% exit haircut. A trade at the liquidity cap eats ~+5% extra."""
    s = ENTRY_SLIP + 0.05 * min(1.0, size / cap)
    return ((1 + ret_pct / 100.0) / (1 + s) - 1) * 100.0


def build():
    fl = C.flow_paths()
    flt = pd.DataFrame({
        "date": fl["date"], "tk": fl["tk"], "entry": fl["entry_min"], "exit": fl["exit_min"],
        "ret": fl["ret_lk"], "mult": fl["mult"], "is_put": fl["is_put"]})
    ml = pd.read_csv("journal/v3_eval_results/v7_core_trades.csv")
    mlt = pd.DataFrame({
        "date": ml["day"].astype(str), "tk": ml.get("ticker", "?"), "entry": ml["minute"].astype(int),
        "exit": (ml["minute"] + ml["hold_min"].fillna(30)).astype(int),
        "ret": ml["pnl_pct"], "mult": ml["size_mult"].fillna(1.0),
        "is_put": ml["direction"].str.lower() == "put"})
    allt = pd.concat([flt, mlt], ignore_index=True)
    allt["dt"] = pd.to_datetime(allt["date"])
    maxd = allt["dt"].max()
    return allt[allt["dt"] >= maxd - pd.Timedelta(days=DAYS)].copy(), maxd


def run(trades):
    bal = START
    taken = skipped_slot = skipped_cap = 0
    daily = {}
    for d, g in trades.groupby("date", sort=True):
        g = g.sort_values("entry")
        deployable = bal * RISK_PCT
        per_slot = bal * RISK_PCT / MAX_CONC
        cap_dollars = bal * POS_CAP
        open_pos = []          # (exit_min, committed_dollars, ret_pct)
        committed = 0.0
        day_pnl = 0.0
        for t in g.itertuples():
            # free capital + slots from positions that have closed by this entry minute
            still = []
            for ex, cm, r in open_pos:
                if ex <= t.entry:
                    day_pnl += cm * r / 100.0          # realize on close
                    committed -= cm
                else:
                    still.append((ex, cm, r))
            open_pos = still
            lc = liq_cap(t.tk)
            size = min(per_slot * t.mult * (PUT_BUDGET if t.is_put else 1.0) * FLAT, cap_dollars, lc)
            if len(open_pos) >= MAX_CONC:
                skipped_slot += 1
                continue
            if committed + size > deployable:
                skipped_cap += 1
                continue
            ret = slip(t.ret, size, lc)            # size-dependent entry slippage
            open_pos.append((t.exit, size, ret))
            committed += size
            taken += 1
        for ex, cm, r in open_pos:                     # close any still open at EOD
            day_pnl += cm * r / 100.0
        daily[d] = day_pnl
        bal += day_pnl
    return bal, daily, taken, skipped_slot, skipped_cap


def maxdd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def main():
    print(f"building realistic ${START:,.0f} projection (8 slots + capital cap + slippage)...", flush=True)
    trades, maxd = build()
    sig = len(trades)
    bal, daily, taken, sk_slot, sk_cap = run(trades)
    ndays = len(daily)
    print(f"window ends {maxd.date()} | {sig} signals sourced over {ndays} trading days "
          f"(~{sig/ndays:.0f}/day)\n")
    print(f"  start balance        ${START:>12,.0f}")
    print(f"  end balance          ${bal:>12,.0f}")
    print(f"  P&L                  ${bal-START:>+12,.0f}   ({(bal/START-1)*100:+.0f}%, "
          f"{(bal/START)**(1/ (ndays/21)) -1:+.1%}/mo)")
    print(f"  max drawdown         ${maxdd(daily):>+12,.0f}")
    print(f"  trades TAKEN         {taken:>12,}   ({taken/ndays:.0f}/day)")
    print(f"  skipped — slots full {sk_slot:>12,}")
    print(f"  skipped — no capital {sk_cap:>12,}")
    print(f"  capture rate         {100*taken/sig:>11.0f}%   (of {sig} sourced signals)")
    print(f"\nvs fantasy compound ($2.1M off $18k, infinite liquidity) and flat-$750 edge (+$52.6k).")
    print("Binding constraint at $23k = the 15% position cap + 8 slots/capital, NOT chain liquidity. "
          "Liquidity/crowding only bites at much larger balances.")


if __name__ == "__main__":
    main()
