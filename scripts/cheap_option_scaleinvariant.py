"""SCALE-INVARIANT cheap-option protection backtest.

CONTEXT: owlet-kody flat-sized 48 contracts of a $0.36 OTM 0DTE SMCI call and lost -$383.
Root cause = flat-budget sizing: contracts = slot_budget / (premium*100). Cheaper option => more
contracts. The earlier fix (scripts/cheap_option_cap_backtest.py) used a FIXED 25-contract cap, but a
fixed contract count does NOT scale with portfolio (strangles big accounts, never binds on small ones).

We need a SCALE-INVARIANT lever. Earlier key finding (cheap_option_cap_backtest + memory): cheap options
are NOT worse expectancy — cheapest CALL bucket PF~0.99, cheapest PUT bucket PF~1.66 (cheap puts are the
BEST cohort). They're higher VARIANCE. So the lever must cut CALL-tail variance WITHOUT clipping the
cheap-PUT edge.

TWO SCALE-INVARIANT LEVERS (both replace the fixed contract cap; both applied to slot_budget which scales
with balance, so they bind on the cheap tail at ALL account sizes):

  LEVER 1 — MIN_SIZING_PREMIUM (effective cost-per-contract floor):
      contracts = slot_budget / (max(premium, FLOOR) * 100)
      A $0.36 option is sized as if it cost FLOOR. Fewer contracts for cheap options, proportional.
      FLOOR in {0.50, 0.68, 0.85, 1.00}.  (fixed-25-on-$1700-slot ~= FLOOR $0.68.)

  LEVER 2 — DELTA-SCALED budget (targets the CAUSE: cheap OTM 0DTE = high gamma / low delta):
      contracts = slot_budget * min(1, |delta|/DELTA_REF) / (premium*100)
      Low-delta lottery tickets get a budget haircut; ATM (|delta|~0.5) ~full budget.
      DELTA_REF in {0.35, 0.45, 0.55}.  delta from option_greeks table (entry snapshot, abs value).

HARNESS REUSE: load_0dte + ATM 0DTE selection at 10:00 ET (ENTRY_MI=30), real ExitFSM (V7 wide-trail +
profit-lock LOCK cfg; PUT cfg for puts), EXIT_HAIRCUT. Same as backtest_2yr_regime / cheap_option_cap.

SIZING uses REAL per-bot slot budgets: slot = balance * 0.75 / MAX_CONCURRENT(=5).
  kody $23k, dennis $10k, paper ~$3-4.7k, plus $100k and $300k to PROVE scale-invariance.

OUTPUT per YEAR + per SIDE (call/put) with n: total P&L, PF, maxDD, per-trade P&L stdev (the variance
we're cutting), %trades the lever BINDS on, %P&L kept vs flat. CALLS-ONLY vs BOTH-SIDES (does the
haircut clip the cheap-put edge?). Scale-invariance proof + SMCI reproduction.

Read-only. Validate small: VALIDATE=1 (SPY+NVDA, 2025-03, sizing math for cheap vs normal @ 2 slots).
Run full: cd /Users/kody/dev/options-owl && python scripts/cheap_option_scaleinvariant.py
"""
from __future__ import annotations

import os
import sqlite3
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

# Real per-bot slot budgets: balance * 0.75 / MAX_CONCURRENT(=5).  Plus scale-invariance probes.
MAX_CONCURRENT = 5
RISK_PCT = 0.75
ACCOUNTS = {
    "paper$3.1k": 3123.0,
    "paper$4.7k": 4685.0,
    "dennis$10k": 10000.0,
    "kody$23k": 23000.0,
    "$100k": 100000.0,
    "$300k": 300000.0,
}
MAX_POSITION_PCT = 15.0  # prod position cap (% of balance per trade) — applied downstream of the lever

# Levers
FLOORS = [0.50, 0.68, 0.85, 1.00]      # LEVER 1 MIN_SIZING_PREMIUM
DELTA_REFS = [0.35, 0.45, 0.55]        # LEVER 2 DELTA_REF

TICKERS = ["SPY", "QQQ", "IWM", "TSLA", "NVDA", "META", "AMD", "AMZN", "AAPL",
           "MSFT", "GOOGL", "GOOG", "AVGO", "MU", "INTC", "ORCL", "QCOM", "MRVL",
           "PLTR", "MSTR", "ARM", "SMH", "TSM", "LRCX"]

MIN_PREMIUM_INCLUDE = 0.30  # tradeable cheap options >= prod floor

LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def sim(pp, mp, up, cfg, otype, ets):
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


def max_drawdown(pnls):
    if len(pnls) == 0:
        return 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


# ---------------- delta lookup (entry snapshot from option_greeks) ----------------
def _delta_map(tk):
    """date -> {(strike,right): |delta| at first snapshot mi>=ENTRY_MI on the 0DTE expiry}.
    Mirrors load_0dte's 0DTE filter (expiration == same calendar date as timestamp)."""
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT strike, right, timestamp, delta FROM option_greeks "
        "WHERE ticker=? AND expiration = substr(timestamp,1,10) ORDER BY timestamp",
        con, params=(tk,))
    con.close()
    if df.empty:
        return {}
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"] >= ENTRY_MI]
    out: dict = {}
    for (date, strike, right), g in df.groupby(["date", "strike", "right"]):
        g = g.sort_values("mi")
        d = g["delta"].iloc[0]
        if d is None or (isinstance(d, float) and np.isnan(d)):
            continue
        out.setdefault(date, {})[(float(strike), right)] = abs(float(d))
    return out


# ---------------------------- build trade set --------------------------------
def build_trades(tickers, validate=False):
    rows = []
    for tk in tickers:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        dmap = _delta_map(tk)
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
                dlt = dmap.get(date, {}).get((float(atm), right), np.nan)
                rows.append({"date": date, "yr": yr, "tk": tk, "side": side,
                             "prem": float(pp[0]), "ret": ret, "delta": dlt})
                nd += 1
        print(f"  {tk}: {nd} trades  (delta cov {sum(1 for r in rows if r['tk']==tk and not np.isnan(r['delta']))}/{nd})",
              flush=True)
    tdf = pd.DataFrame(rows)
    tdf = tdf[tdf["prem"] >= MIN_PREMIUM_INCLUDE].sort_values("date").reset_index(drop=True)
    return tdf


# ---------------------------- sizing models ----------------------------------
def contracts_flat(prem, slot):
    return max(1, int(slot // (prem * 100.0)))


def contracts_floor(prem, slot, floor):
    eff = max(prem, floor)
    return max(1, int(slot // (eff * 100.0)))


def contracts_delta(prem, slot, delta, ref):
    """Delta-scaled budget. If delta missing -> no haircut (fail-open), flagged separately."""
    if delta is None or np.isnan(delta):
        hair = 1.0
    else:
        hair = min(1.0, delta / ref)
    eff_budget = slot * hair
    return max(1, int(eff_budget // (prem * 100.0)))


def position_cap(prem, balance):
    """MAX_POSITION_PCT cap (contracts). Applied downstream of the lever, like prod."""
    max_spend = balance * (MAX_POSITION_PCT / 100.0)
    return max(1, int(max_spend // (prem * 100.0)))


def pnl_of(contracts, prem, ret):
    return contracts * prem * 100.0 * (ret / 100.0)


# ---------------------------- evaluate one config ----------------------------
def eval_config(sub, balance, sizer):
    """sizer(prem, slot, delta) -> raw contracts (pre position-cap). Returns metrics dict."""
    slot = balance * RISK_PCT / MAX_CONCURRENT
    base_pnls, lev_pnls, binds = [], [], 0
    for _, r in sub.iterrows():
        c_flat = contracts_flat(r["prem"], slot)
        c_lev = sizer(r["prem"], slot, r["delta"])
        cap = position_cap(r["prem"], balance)
        c_flat_capped = min(c_flat, cap)
        c_lev_capped = min(c_lev, cap)
        if c_lev_capped < c_flat_capped:
            binds += 1
        base_pnls.append(pnl_of(c_flat_capped, r["prem"], r["ret"]))
        lev_pnls.append(pnl_of(c_lev_capped, r["prem"], r["ret"]))
    base_pnls = np.array(base_pnls)
    lev_pnls = np.array(lev_pnls)
    return {
        "n": len(sub),
        "base_total": base_pnls.sum(),
        "total": lev_pnls.sum(),
        "pf": pf(lev_pnls if False else sub["ret"].values) if False else pf(lev_pnls),
        "pf_ret": pf(sub["ret"].values),
        "maxdd": max_drawdown(lev_pnls),
        "base_maxdd": max_drawdown(base_pnls),
        "stdev": lev_pnls.std(),
        "base_stdev": base_pnls.std(),
        "bind": binds / max(1, len(sub)) * 100,
        "kept": (lev_pnls.sum() / base_pnls.sum() * 100) if base_pnls.sum() != 0 else float("nan"),
    }


def fmt_pf(p):
    return f"{p:.2f}" if np.isfinite(p) else "inf"


def lever_table(tdf, sizers, title, account="kody$23k"):
    """Print per-year, per-side table for one account. sizers = [(label, sizer_fn), ...]."""
    bal = ACCOUNTS[account]
    print("\n" + "=" * 118)
    print(f"{title}   [account={account}  slot=${bal*RISK_PCT/MAX_CONCURRENT:,.0f}]")
    print("=" * 118)
    for side in ("call", "put"):
        sub_all = tdf[tdf["side"] == side]
        print(f"\n### {side.upper()}S ###")
        for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
            sub = sub_all if yr == "ALL" else sub_all[sub_all["yr"] == yr]
            if len(sub) == 0:
                continue
            print(f"\n  -- {side} {yr}  (n={len(sub)}) --")
            print(f"    {'config':<14}{'$P&L':>11}{'PF':>7}{'maxDD$':>11}{'stdev$':>9}"
                  f"{'bind%':>8}{'%kept':>8}")
            for lbl, fn in sizers:
                m = eval_config(sub, bal, fn)
                print(f"    {lbl:<14}{m['total']:>+11.0f}{fmt_pf(m['pf']):>7}"
                      f"{m['maxdd']:>11.0f}{m['stdev']:>9.0f}{m['bind']:>7.1f}%{m['kept']:>7.0f}%")


# ---------------------------- main reports -----------------------------------
def report_lever1(tdf):
    sizers = [("NONE(flat)", lambda p, s, d: contracts_flat(p, s))]
    for fl in FLOORS:
        sizers.append((f"floor${fl:.2f}", (lambda p, s, d, fl=fl: contracts_floor(p, s, fl))))
    lever_table(tdf, sizers, "LEVER 1 — MIN_SIZING_PREMIUM floor  (BOTH SIDES shown; see side split)")


def report_lever2(tdf):
    sizers = [("NONE(flat)", lambda p, s, d: contracts_flat(p, s))]
    for rf in DELTA_REFS:
        sizers.append((f"dref{rf:.2f}", (lambda p, s, d, rf=rf: contracts_delta(p, s, d, rf))))
    lever_table(tdf, sizers, "LEVER 2 — DELTA-SCALED budget  (BOTH SIDES shown; see side split)")


def report_callsonly_vs_both(tdf, floor=0.68, dref=0.45):
    """The crux: does applying the haircut to PUTS clip the cheap-put edge?"""
    print("\n" + "=" * 118)
    print("CALLS-ONLY vs BOTH-SIDES — does the haircut clip the cheap-PUT edge?  [account=kody$23k]")
    print(f"  using LEVER1 floor=${floor:.2f} and LEVER2 dref={dref:.2f} as representatives")
    print("=" * 118)
    bal = ACCOUNTS["kody$23k"]

    def flat(p, s, d):
        return contracts_flat(p, s)

    def f1(p, s, d):
        return contracts_floor(p, s, floor)

    def f2(p, s, d):
        return contracts_delta(p, s, d, dref)

    for lev_name, fn in (("FLOOR", f1), ("DELTA", f2)):
        print(f"\n### {lev_name} lever ###")
        # BOTH = haircut on both; CALLS-ONLY = haircut on calls, flat on puts.
        for scope in ("BOTH-SIDES", "CALLS-ONLY"):
            print(f"\n  scope = {scope}")
            print(f"    {'side':<8}{'n':>6}{'flat$P&L':>11}{'lever$P&L':>11}{'flatPF':>8}"
                  f"{'leverPF':>8}{'flatStd':>9}{'levStd':>9}{'%kept':>7}")
            for side in ("call", "put"):
                sub = tdf[tdf["side"] == side]
                use = fn if (scope == "BOTH-SIDES" or side == "call") else flat
                m = eval_config(sub, bal, use)
                # flat baseline metrics
                mb = eval_config(sub, bal, flat)
                print(f"    {side:<8}{m['n']:>6}{mb['total']:>+11.0f}{m['total']:>+11.0f}"
                      f"{fmt_pf(mb['pf']):>8}{fmt_pf(m['pf']):>8}{mb['stdev']:>9.0f}{m['stdev']:>9.0f}"
                      f"{m['kept']:>6.0f}%")
            # combined book (calls + puts) totals for each scope
            comb_flat, comb_lev, all_pnl_flat, all_pnl_lev = 0.0, 0.0, [], []
            for side in ("call", "put"):
                sub = tdf[tdf["side"] == side]
                slot = bal * RISK_PCT / MAX_CONCURRENT
                use = fn if (scope == "BOTH-SIDES" or side == "call") else flat
                for _, r in sub.iterrows():
                    cap = position_cap(r["prem"], bal)
                    cf = min(contracts_flat(r["prem"], slot), cap)
                    cl = min(use(r["prem"], slot, r["delta"]), cap)
                    all_pnl_flat.append(pnl_of(cf, r["prem"], r["ret"]))
                    all_pnl_lev.append(pnl_of(cl, r["prem"], r["ret"]))
            all_pnl_flat = np.array(all_pnl_flat)
            all_pnl_lev = np.array(all_pnl_lev)
            print(f"    {'BOOK':<8}{len(all_pnl_lev):>6}{all_pnl_flat.sum():>+11.0f}{all_pnl_lev.sum():>+11.0f}"
                  f"{fmt_pf(pf(all_pnl_flat)):>8}{fmt_pf(pf(all_pnl_lev)):>8}"
                  f"{all_pnl_flat.std():>9.0f}{all_pnl_lev.std():>9.0f}"
                  f"{all_pnl_lev.sum()/all_pnl_flat.sum()*100:>6.0f}%")


def report_scale_invariance(tdf, floor=0.68, dref=0.45):
    """Prove the lever binds on the cheap tail at $3k AND $300k, and does NOT strangle
    normally-priced options at $300k."""
    print("\n" + "=" * 118)
    print("SCALE-INVARIANCE PROOF — bind-rate on CHEAP-CALL tail across account sizes")
    print(f"  (LEVER1 floor=${floor:.2f}, LEVER2 dref={dref:.2f}; CALLS only)")
    print("=" * 118)
    calls = tdf[tdf["side"] == "call"].copy()
    cheap = calls[calls["prem"] < 0.50]    # the SMCI tail
    normal = calls[calls["prem"] >= 2.0]   # normally-priced options (should NOT be strangled)
    print(f"\n  cheap-call tail (prem<$0.50): n={len(cheap)}   normal-call (prem>=$2.0): n={len(normal)}")
    print(f"\n  {'account':<12}{'slot$':>10}"
          f"{'FLOOR bind% cheap':>19}{'FLOOR bind% normal':>20}"
          f"{'DELTA bind% cheap':>19}{'DELTA bind% normal':>20}")
    for acct, bal in ACCOUNTS.items():
        slot = bal * RISK_PCT / MAX_CONCURRENT

        def bindrate(df_sub, fn):
            b = 0
            for _, r in df_sub.iterrows():
                cap = position_cap(r["prem"], bal)
                cf = min(contracts_flat(r["prem"], slot), cap)
                cl = min(fn(r["prem"], slot, r["delta"]), cap)
                if cl < cf:
                    b += 1
            return b / max(1, len(df_sub)) * 100

        f1 = lambda p, s, d: contracts_floor(p, s, floor)
        f2 = lambda p, s, d: contracts_delta(p, s, d, dref)
        print(f"  {acct:<12}{slot:>10,.0f}"
              f"{bindrate(cheap, f1):>18.0f}%{bindrate(normal, f1):>19.0f}%"
              f"{bindrate(cheap, f2):>18.0f}%{bindrate(normal, f2):>19.0f}%")
    print("\n  (scale-invariant = cheap-tail bind% stays HIGH at every size; normal bind% stays ~0)")


def report_smci(tdf, floor=0.68, dref=0.45):
    print("\n" + "=" * 118)
    print("SMCI REPRODUCTION — $0.36 0DTE call, kody slot")
    print("=" * 118)
    bal = ACCOUNTS["kody$23k"]
    slot = bal * RISK_PCT / MAX_CONCURRENT
    prem = 0.36
    cap = position_cap(prem, bal)
    # representative ATM-call delta for the delta lever illustration
    cd = tdf[(tdf["side"] == "call") & (tdf["delta"].notna())]["delta"]
    atm_delta = 0.5  # ATM 0DTE call ~0.5; an OTM $0.36 lottery call is much lower
    otm_delta = 0.18  # typical low-delta cheap OTM
    print(f"  slot=${slot:,.0f}  pos_cap(15%)= {cap} contracts  cost/contract=${prem*100:.0f}")
    print(f"  FLAT (uncapped-by-lever): {contracts_flat(prem, slot)} contracts "
          f"-> after pos_cap: {min(contracts_flat(prem, slot), cap)}")
    print(f"  LEVER1 floor=${floor:.2f}: sized as ${max(prem,floor)*100:.0f}/contract "
          f"-> {contracts_floor(prem, slot, floor)} contracts (pos_cap {min(contracts_floor(prem,slot,floor),cap)})")
    print(f"  LEVER2 dref={dref:.2f} @ OTM delta~{otm_delta}: haircut={min(1,otm_delta/dref):.2f} "
          f"-> {contracts_delta(prem, slot, otm_delta, dref)} contracts "
          f"(pos_cap {min(contracts_delta(prem,slot,otm_delta,dref),cap)})")
    print(f"  (the live trade was 48 contracts on a SMALLER-balance day; on kody's $23k slot flat sizing "
          f"alone gives {contracts_flat(prem, slot)} pre-cap, {min(contracts_flat(prem, slot), cap)} post-cap)")


def validate(tdf):
    print("\n" + "#" * 80)
    print("# VALIDATION — sizing math, cheap vs normal, two slot sizes")
    print("#" * 80)
    print(f"\ntrades: {len(tdf)}  ({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)")
    print(f"delta coverage: {tdf['delta'].notna().mean()*100:.0f}%")
    for slot_bal, label in ((ACCOUNTS["paper$3.1k"], "paper$3.1k"), (ACCOUNTS["$300k"], "$300k")):
        slot = slot_bal * RISK_PCT / MAX_CONCURRENT
        print(f"\n  -- slot for {label}: ${slot:,.0f} (bal ${slot_bal:,.0f}) --")
        print(f"    {'prem':>6}{'delta':>7}{'flat':>6}{'floor.68':>10}{'dref.45':>9}{'posCap':>8}")
        for prem, dlt in ((0.36, 0.18), (0.45, 0.30), (0.80, 0.42), (1.50, 0.50), (3.00, 0.52)):
            print(f"    {prem:>6.2f}{dlt:>7.2f}{contracts_flat(prem, slot):>6}"
                  f"{contracts_floor(prem, slot, 0.68):>10}{contracts_delta(prem, slot, dlt, 0.45):>9}"
                  f"{position_cap(prem, slot_bal):>8}")
    cb = tdf[(tdf["side"] == "call") & (tdf["prem"] < 0.50)]
    pb = tdf[(tdf["side"] == "put") & (tdf["prem"] < 0.50)]
    print(f"\n  cheap-call ($0.30-0.50): n={len(cb)} PF={pf(cb['ret'].values):.2f} "
          f"avgRet={cb['ret'].mean():+.0f}%  | cheap-put: n={len(pb)} PF={pf(pb['ret'].values):.2f} "
          f"avgRet={pb['ret'].mean():+.0f}%")
    print("\nVALIDATION OK — re-run without VALIDATE=1 for full universe.\n")


def main():
    if os.environ.get("VALIDATE") == "1":
        print("### VALIDATION — SPY + NVDA, 2025-03 only ###", flush=True)
        tdf = build_trades(["SPY", "NVDA"], validate=True)
        validate(tdf)
        return

    print(f"########## FULL RUN — {len(TICKERS)} tickers, 2024-01..2026-06 ##########", flush=True)
    tdf = build_trades(TICKERS, validate=False)
    print(f"\ntotal {len(tdf)} trades  ({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)  "
          f"delta cov {tdf['delta'].notna().mean()*100:.0f}%", flush=True)
    report_lever1(tdf)
    report_lever2(tdf)
    report_callsonly_vs_both(tdf)
    report_scale_invariance(tdf)
    report_smci(tdf)
    print("\nDONE.")


if __name__ == "__main__":
    main()
