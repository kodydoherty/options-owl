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
DAYS = 60

# Per-trade LIQUIDITY ceiling + per-ticker SPREAD — MEASURED from PG option_ticks (14d, near-ATM,
# 0-2 DTE, 2026-06-16). LIQ = ~3% of median daily $ volume (day_vol x mid x 100) = realistic max
# without serious impact. SPR = measured median spread (the real per-trade cost). SPY is effectively
# uncapped at our size; the thin names (ARM-type) are gated by SPREAD more than size.
LIQ = {"SPY": 900000.0, "QQQ": 600000.0, "TSLA": 280000.0, "MU": 140000.0, "AMD": 53000.0,
       "META": 35000.0, "NVDA": 60000.0, "AMZN": 40000.0, "GOOG": 35000.0, "AVGO": 25000.0,
       "ARM": 8000.0, "LRCX": 8000.0, "ORCL": 12000.0, "INTC": 15000.0, "AAPL": 40000.0}
SPR = {"SPY": 0.010, "TSLA": 0.020, "MU": 0.030, "AMD": 0.055, "META": 0.070, "NVDA": 0.040,
       "AMZN": 0.045, "GOOG": 0.050, "AVGO": 0.060, "ARM": 0.140, "LRCX": 0.120, "ORCL": 0.080,
       "INTC": 0.070, "AAPL": 0.035}
LIQ_DEFAULT, SPR_DEFAULT = 15000.0, 0.06


def liq_cap(tk):
    return LIQ.get(tk, LIQ_DEFAULT)


def slip(ret_pct, size, tk, cap, crowd=1.0):
    """Real per-trade cost: cross half the MEASURED spread on entry + a size-impact term. CROWD = how
    many bots stack the SAME contract at once (5 bots all at the cap → their sizes add, so impact
    scales with crowd*size vs the contract's liquidity). Mainly bites thin names. Exit haircut in ret."""
    s = SPR.get(tk, SPR_DEFAULT) / 2.0 + 0.10 * min(1.0, crowd * size / cap)
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


def run(trades, cap_at=None, crowd=1.0):
    """cap_at=None: compound endlessly. cap_at=$X: once working capital hits $X, FREEZE sizing at $X
    and sweep every dollar above it to 'banked' (withdrawn, never risked). Bounds absolute drawdown.
    crowd: # of bots stacking the same contract (5 = all bots live at the cap)."""
    bal = START
    banked = 0.0
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
            ret = slip(t.ret, size, t.tk, lc, crowd)   # measured spread + size impact (× crowd)
            open_pos.append((t.exit, size, ret))
            committed += size
            taken += 1
        for ex, cm, r in open_pos:                     # close any still open at EOD
            day_pnl += cm * r / 100.0
        daily[d] = day_pnl
        bal += day_pnl
        if cap_at and bal > cap_at:                    # take profits: sweep excess off the table
            banked += bal - cap_at
            bal = cap_at
    return bal + banked, banked, daily, taken, skipped_slot, skipped_cap


def maxdd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


CAPS = [None, 150000.0, 100000.0, 75000.0, 50000.0, 35000.0, 25000.0]


def ddpct_of(daily):
    eq = pk = 0.0
    for d in sorted(daily):
        eq += daily[d]; pk = max(pk, eq)
    dd = maxdd(daily)
    return dd, (100 * dd / (START + pk) if (START + pk) else 0)


def sweep(trades, crowd, label):
    print(f"\n=== {label} (crowd={crowd:g}) ===")
    print(f"{'take-profit cap':<18}{'end wealth':>12}{'banked':>11}{'maxDD':>11}{'DD%':>6}{'ret/DD':>8}")
    print("-" * 66)
    for cap in CAPS:
        total, banked, daily, taken, _, _ = run(trades, cap_at=cap, crowd=crowd)
        dd, dp = ddpct_of(daily)
        name = "endless" if cap is None else f"${cap/1000:.0f}k"
        rdd = (total - START) / abs(dd) if dd else float("inf")
        print(f"{name:<18}${total:>11,.0f}${banked:>10,.0f}${dd:>+10,.0f}{dp:>5.0f}%{rdd:>8.1f}")


def main():
    print(f"building realistic ${START:,.0f} projection (measured liquidity + spreads)...", flush=True)
    trades, maxd = build()
    print(f"window ends {maxd.date()} | {len(trades)} signals over {trades['date'].nunique()} days")
    print("ret/DD = P&L per $1 of max drawdown (higher = better risk-adjusted). flat-$750 edge = +$52.6k.")
    sweep(trades, 1, "1 bot / isolated (paper bots don't compete for fills)")
    sweep(trades, 5, "all 5 bots LIVE at the cap (sizes stack on the same contract)")
    print("\nNOTE: crowd hits THIN names (ARM/LRCX ~14% spread, small books) hardest; SPY/liquid names "
          "barely care. Lower caps = lower DD + less crowding cost but smaller banked income. The knee "
          "(best ret/DD) is the take-profit target. Each bot is a SEPARATE account — cap is per-account.")


if __name__ == "__main__":
    main()
