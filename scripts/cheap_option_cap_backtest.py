"""CHEAP-OPTION PROTECTION backtest — are we sizing biggest into the worst trades?

URGENT CONTEXT: owlet-kody bought 48 contracts of a $0.36 OTM 0DTE SMCI call (score 62 / pattern
0.630, barely qualifying) and lost -$383 in 7 min. Root cause = FLAT-BUDGET sizing:
    contracts = slot_budget / (premium * 100)
so the CHEAPER the option, the BIGGER the contract count. The premium CAP only blocks EXPENSIVE
options (>$6); nothing caps CHEAP lottery tickets. We test whether a per-trade CONTRACT CAP and/or a
MIN-PREMIUM FLOOR protect the live books without cutting real winners.

REUSES the 2.5yr regime harness exactly:
  - load_0dte + nearest-strike-to-spot ATM 0DTE selection at 10:00 ET (mi0=30)
  - real ExitFSM exit (V7 wide-trail + profit-lock LOCK cfg; PUT cfg for puts)
  - EXIT_HAIRCUT on exit fills
Each ticker/day yields a CALL trade and a PUT trade (same as backtest_2yr_regime).

SIZING MODEL (the live flat-budget model, simplified to a single $1700 slot so cheap-vs-expensive
contract counts are comparable across trades):
    SLOT = $1700.   cost_per_contract = entry_premium * 100.
    contracts_flat = max(1, floor(SLOT / cost_per_contract))
    pnl_dollars    = contracts * cost_per_contract * (ret_pct / 100)
                   = contracts * entry_premium * 100 * ret/100
A CONTRACT CAP caps `contracts`. A MIN-PREMIUM FLOOR skips the trade entirely.

Three questions (CALLS and PUTS split, per YEAR + overall, n on every cell):
  Q1 premium-bucket expectancy: are cheap options structurally worse (PF<1, neg avg ret)?
  Q2 contract cap (15/20/25/30/40): P&L / PF / maxDD / P&L stdev / bind-rate.
  Q3 min-premium floor (0.30/0.40/0.50/0.75): P&L kept vs dropped, PF of dropped trades.

Read-only. Validate: VALIDATE=1 (1-2 tickers x 1 month, bucket counts + sample sizing calc).
Run: cd /Users/kody/dev/options-owl && python scripts/cheap_option_cap_backtest.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

from backtest_2yr_regime import load_0dte  # noqa: E402

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
ENTRY_MI = 30  # 10:00 ET
SLOT = 1700.0  # flat slot budget ($); cheap-vs-expensive contract counts comparable

# Full liquid universe present in the DB (single-name + index ETFs that actually have 0DTE).
TICKERS = ["SPY", "QQQ", "IWM", "TSLA", "NVDA", "META", "AMD", "AMZN", "AAPL",
           "MSFT", "GOOGL", "GOOG", "AVGO", "MU", "INTC", "ORCL", "QCOM", "MRVL",
           "PLTR", "MSTR", "ARM", "SMH", "TSM", "LRCX"]

# Premium buckets (entry premium $).  edges -> (lo, hi]; first bucket includes the $0.30 floor.
BUCKETS = [(0.30, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
BUCKET_LABELS = ["$0.30-0.50", "$0.50-1.0", "$1.0-2.0", "$2.0-4.0", "$4.0+"]
CAPS = [15, 20, 25, 30, 40]
FLOORS = [0.30, 0.40, 0.50, 0.75]
MIN_PREMIUM_INCLUDE = 0.30  # we only study tradeable cheap lottery tickets >= $0.30 (matches prod floor)

LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def sim(pp, mp, up, cfg, otype, ets):
    """Real ExitFSM from entry (pp[0]) to EOD; return % net of haircut. Mirrors harness sim()."""
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=0,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k], minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def pf(x):
    x = np.asarray(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else (float("inf") if g > 0 else 0.0)


def contracts_flat(prem, cap=None):
    cpc = prem * 100.0
    c = max(1, int(SLOT // cpc))
    if cap is not None:
        c = min(c, cap)
    return c


def max_drawdown(pnls):
    """Sequential max drawdown of cumulative $ P&L (trades in chronological order)."""
    if len(pnls) == 0:
        return 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def build_trades(tickers, validate=False):
    rows = []
    val_n = 0
    for tk in tickers:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        cfg_c = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        cfg_p = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="put"),
                                            is_put=True)
        nd = 0
        for date, g in df.groupby("date"):
            if validate and date[:7] != "2025-03":
                continue
            if date not in stock or ENTRY_MI not in stock[date]:
                continue
            spot = stock[date][ENTRY_MI]
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            yr = date[:4]
            for side, right, cfg in (("call", "CALL", cfg_c), ("put", "PUT", cfg_p)):
                ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].to_numpy(float)
                mp = ch["mi"].to_numpy(int)
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                up = [stock[date].get(int(m), spot) for m in mp]
                ets = D.datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
                ret = sim(pp, list(mp), list(up), cfg, side, ets)
                rows.append({"date": date, "yr": yr, "tk": tk, "side": side,
                             "prem": float(pp[0]), "ret": ret})
                nd += 1
        print(f"  {tk}: {nd} trades", flush=True)
        val_n += nd
        if validate and val_n >= 1:
            # one ticker of one month is enough for the sanity pass
            pass
    tdf = pd.DataFrame(rows)
    # restrict to tradeable cheap-and-up options (>= prod min premium); ATM means few sub-$0.30
    tdf = tdf[tdf["prem"] >= MIN_PREMIUM_INCLUDE].reset_index(drop=True)
    return tdf


def bucket_of(prem):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= prem < hi or (i == 0 and prem == lo):
            return i
    return len(BUCKETS) - 1


# ============================ Q1: premium-bucket expectancy =====================================
def q1(tdf):
    print("\n" + "=" * 100)
    print("Q1 — ARE CHEAP OPTIONS STRUCTURALLY WORSE?  (bucket by entry premium)")
    print("  implied_contracts = flat $1700 slot / (premium*100), UNCAPPED (the sizing the bug uses)")
    print("=" * 100)
    tdf = tdf.copy()
    tdf["bk"] = tdf["prem"].apply(bucket_of)
    for side in ("call", "put"):
        sub_all = tdf[tdf["side"] == side]
        print(f"\n### {side.upper()}S ###")
        for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
            sub = sub_all if yr == "ALL" else sub_all[sub_all["yr"] == yr]
            print(f"\n  -- {side} {yr}  (n={len(sub)}) --")
            print(f"    {'bucket':<12}{'n':>6}{'WR%':>7}{'PF':>7}{'avgRet%':>9}"
                  f"{'implContracts':>14}{'$P&L(flat)':>12}")
            for bi, lbl in enumerate(BUCKET_LABELS):
                b = sub[sub["bk"] == bi]
                if len(b) == 0:
                    print(f"    {lbl:<12}{0:>6}{'--':>7}{'--':>7}{'--':>9}{'--':>14}{'--':>12}")
                    continue
                wr = (b["ret"] > 0).mean() * 100
                p = pf(b["ret"].values)
                avg = b["ret"].mean()
                # implied contract count midpoint of bucket (representative)
                ic = b["prem"].apply(lambda x: contracts_flat(x)).mean()
                pnl = (b["prem"] * 100 * b["prem"].apply(lambda x: contracts_flat(x)) * b["ret"] / 100).sum()
                pfs = f"{p:.2f}" if np.isfinite(p) else "inf"
                print(f"    {lbl:<12}{len(b):>6}{wr:>7.1f}{pfs:>7}{avg:>+9.1f}{ic:>14.1f}{pnl:>+12.0f}")


# ============================ Q2: per-trade contract cap ========================================
def q2(tdf):
    print("\n" + "=" * 100)
    print("Q2 — PER-TRADE CONTRACT CAP  (flat $1700 slot, then hard-cap contracts)")
    print("  metrics: total $P&L, PF, maxDD($), P&L stdev($/trade), %trades cap BINDS")
    print("=" * 100)
    tdf = tdf.sort_values("date").reset_index(drop=True)
    for side in ("call", "put"):
        sub_all = tdf[tdf["side"] == side]
        print(f"\n### {side.upper()}S ###")
        for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
            sub = (sub_all if yr == "ALL" else sub_all[sub_all["yr"] == yr]).sort_values("date")
            if len(sub) == 0:
                continue
            print(f"\n  -- {side} {yr}  (n={len(sub)}) --")
            print(f"    {'cap':<10}{'$P&L':>11}{'PF':>7}{'maxDD$':>11}{'stdev$':>10}{'bind%':>8}"
                  f"{'%P&Lkept':>10}")
            # baseline = uncapped
            base_pnls = []
            for _, r in sub.iterrows():
                c = contracts_flat(r["prem"])
                base_pnls.append(c * r["prem"] * 100 * r["ret"] / 100)
            base_pnls = np.array(base_pnls)
            base_total = base_pnls.sum()
            # uncapped row
            print(f"    {'NONE':<10}{base_total:>+11.0f}{pf(base_pnls):>7.2f}"
                  f"{max_drawdown(base_pnls):>11.0f}{base_pnls.std():>10.0f}{0.0:>7.1f}%{100.0:>9.0f}%")
            for cap in CAPS:
                pnls, binds = [], 0
                for _, r in sub.iterrows():
                    c_un = contracts_flat(r["prem"])
                    c = min(c_un, cap)
                    if c_un > cap:
                        binds += 1
                    pnls.append(c * r["prem"] * 100 * r["ret"] / 100)
                pnls = np.array(pnls)
                total = pnls.sum()
                kept = (total / base_total * 100) if base_total != 0 else float("nan")
                bind = binds / len(sub) * 100
                print(f"    {cap:<10}{total:>+11.0f}{pf(pnls):>7.2f}"
                      f"{max_drawdown(pnls):>11.0f}{pnls.std():>10.0f}{bind:>7.1f}%{kept:>9.0f}%")


# ============================ Q3: min-premium floor =============================================
def q3(tdf):
    print("\n" + "=" * 100)
    print("Q3 — MIN-PREMIUM FLOOR  (skip trades with entry premium < floor)")
    print("  PF-of-DROPPED < 1.0 => the floor is pure benefit (it removes net-losing trades)")
    print("  $ uses flat $1700 slot sizing (uncapped) so dropped-$ reflects the over-sized cheap tail")
    print("=" * 100)
    for side in ("call", "put"):
        sub_all = tdf[tdf["side"] == side]
        print(f"\n### {side.upper()}S ###")
        for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
            sub = sub_all if yr == "ALL" else sub_all[sub_all["yr"] == yr]
            if len(sub) == 0:
                continue
            # precompute flat $ per trade (uncapped)
            sub = sub.copy()
            sub["pnl$"] = sub.apply(lambda r: contracts_flat(r["prem"]) * r["prem"] * 100 * r["ret"] / 100,
                                    axis=1)
            tot = sub["pnl$"].sum()
            print(f"\n  -- {side} {yr}  (n={len(sub)}, total $P&L={tot:+.0f}) --")
            print(f"    {'floor':<8}{'nDropped':>10}{'nKept':>7}{'$kept':>11}{'$dropped':>11}"
                  f"{'PFdropped':>11}{'PFkept':>9}")
            for fl in FLOORS:
                drop = sub[sub["prem"] < fl]
                keep = sub[sub["prem"] >= fl]
                pfd = pf(drop["ret"].values) if len(drop) else float("nan")
                pfk = pf(keep["ret"].values) if len(keep) else float("nan")
                pfds = f"{pfd:.2f}" if (len(drop) and np.isfinite(pfd)) else ("inf" if len(drop) else "--")
                pfks = f"{pfk:.2f}" if (len(keep) and np.isfinite(pfk)) else ("inf" if len(keep) else "--")
                print(f"    {fl:<8.2f}{len(drop):>10}{len(keep):>7}{keep['pnl$'].sum():>+11.0f}"
                      f"{drop['pnl$'].sum():>+11.0f}{pfds:>11}{pfks:>9}")


def validate(tdf):
    print("\n" + "#" * 80)
    print("# VALIDATION — bucket counts + sample sizing calc")
    print("#" * 80)
    print(f"\ntotal trades (>= ${MIN_PREMIUM_INCLUDE} entry): {len(tdf)}  "
          f"({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)")
    tdf = tdf.copy()
    tdf["bk"] = tdf["prem"].apply(bucket_of)
    print("\nbucket counts:")
    for bi, lbl in enumerate(BUCKET_LABELS):
        b = tdf[tdf["bk"] == bi]
        cmin = contracts_flat(BUCKETS[bi][0])
        chi = contracts_flat(min(BUCKETS[bi][1], 50.0))
        print(f"  {lbl:<12} n={len(b):>5}   implied contracts @bucket edges: "
              f"{chi}..{cmin} (flat ${SLOT:.0f}/slot)")
    print("\nsample sizing calc (the SMCI-style cheap lottery zone):")
    for prem in (0.36, 0.45, 0.80, 1.50, 3.0, 5.0):
        c = contracts_flat(prem)
        cpc = prem * 100
        print(f"  prem=${prem:<5} cost/contract=${cpc:>6.0f}  -> flat contracts = floor(1700/{cpc:.0f}) "
              f"= {c}   (capped@20 -> {min(c,20)})")
    # spot-check one cheap-bucket cell PF
    cb = tdf[(tdf["bk"] == 0)]
    print(f"\ncheapest bucket ($0.30-0.50): n={len(cb)}  WR={ (cb['ret']>0).mean()*100:.1f}%  "
          f"PF={pf(cb['ret'].values):.2f}  avgRet={cb['ret'].mean():+.1f}%")


def main():
    is_val = os.environ.get("VALIDATE") == "1"
    if is_val:
        print("### VALIDATION PASS — SPY + NVDA, 2025-03 only ###")
        tdf = build_trades(["SPY", "NVDA"], validate=True)
        validate(tdf)
        print("\nVALIDATION OK — re-run without VALIDATE=1 for the full universe.\n")
        return

    print(f"########## FULL RUN — {len(TICKERS)} tickers, 2024-01..2026-06 ##########", flush=True)
    tdf = build_trades(TICKERS, validate=False)
    print(f"\ntotal trades (>= ${MIN_PREMIUM_INCLUDE} entry): {len(tdf)}  "
          f"({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)", flush=True)
    q1(tdf)
    q2(tdf)
    q3(tdf)
    print("\nDONE.")


if __name__ == "__main__":
    main()
