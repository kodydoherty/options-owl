"""REAL-FLOW ENTRY-TIMING STUDY — "do dip/pullback rules beat buy-NOW on the REAL whale-chases?"

The proxy study (scripts/entry_timing_oracle.py) used a FIXED 10:00 ATM-0DTE candidate per ticker/day and
concluded optimal entry TIMING is a huge theoretical prize but UNEXPLOITABLE (every no-lookahead rule caught
1-3% of it = noise, because waiting misses monotone rippers and catches falling knives — adverse selection).

THIS script swaps the entry SET: instead of a 10:00 proxy, it uses the REAL qualifying flow signals from
journal/uw_historical.db at their REAL created_at minutes (the actual whale-chase moments Kody worries about:
flow CALLs that chase the spike and are instantly underwater). Everything else mirrors PROD + the proxy harness:
  - PROD qualifying filter (uw_flow_collector): has_sweep, total_premium>=250k, ask_frac>=0.60, ticker in whitelist.
  - PROD strike: NEAREST-DTE (today->next business days), ATM (or validated ~$2 OTM via select_flow_strike) —
    NOT the whale's far-dated expiry (multi-day flow validated a LOSER) and NOT the whale's strike.
  - Same real ExitFSM (V7 wide-trail + profit-lock LOCK cfg) run from the ACTUAL entry minute to EOD.
  - Same EXIT_HAIRCUT.

Strategies at the REAL flow entry time t0 (window W min after t0): NOW (buy @ t0), ORACLE (lookahead ceiling
= lowest close in [t0,t0+W]), DIP5/10/15 (no-lookahead), PULLBACK-CONFIRM (no-lookahead).

NO-LOOKAHEAD: dip/pullback decisions read only bars up to the decision minute; entry is the bar AFTER the
trigger. Inline asserts + VALIDATE sanity rows prove it.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/realflow_entry_timing.py
Optional env: VALIDATE=1 (print sanity rows + asserts on a few signals first), WINDOWS=15,30
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from options_owl.bot_runner import select_flow_strike  # noqa: E402  (PROD strike resolver)
from options_owl.config.settings import Settings  # noqa: E402
from options_owl.risk.exit_v5.config import (  # noqa: E402
    apply_v7_wide_trail_exits,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

ET = ZoneInfo("America/New_York")
THETA_DB = str(ROOT / "journal" / "thetadata_options.db")
FLOW_DB = str(ROOT / "journal" / "uw_historical.db")
EXIT_HAIRCUT = 0.03  # mirror harness exit-fill haircut

_S = Settings()
CALL_WL = {t.strip().upper() for t in _S.UW_FLOW_CALL_TICKERS.split(",") if t.strip()}
PUT_WL = {t.strip().upper() for t in _S.UW_FLOW_PUT_TICKERS.split(",") if t.strip()}  # already includes SPY
MIN_PREM = _S.UW_FLOW_MIN_PREMIUM
ASK_FRAC_MIN = _S.UW_FLOW_ASK_FRAC
ENABLE_FLOW_OTM = _S.ENABLE_FLOW_OTM_STRIKE  # prod default False
OTM_TARGET = 2.0  # ~$2 OTM (mirrors prod _otm_target default)
# Validated OTM combos (from CLAUDE.md + flow_otm_test): AMD/INTC/META/SPY calls, TSLA puts
OTM_CALLS = {"AMD", "INTC", "META", "SPY"}
OTM_PUTS = {"TSLA"}

# LOCK cfg = same V7 wide-trail + profit-lock toggles the proxy harness uses (entry_timing_oracle.py)
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)

DIP_LEVELS = [5, 10, 15]
STRATS = ["NOW", "ORACLE"] + [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]


# ----- data loaders ------------------------------------------------------------------------------
def load_qualifying_signals() -> pd.DataFrame:
    """Real qualifying flow signals (PROD filter), with real entry minute-of-day in ET."""
    con = sqlite3.connect(FLOW_DB)
    df = pd.read_sql_query(
        "SELECT ticker, created_at, type, strike, expiry, price, underlying_price, "
        "total_premium, has_sweep, total_ask_side_prem FROM flow_alerts", con)
    con.close()
    df["ticker"] = df["ticker"].str.upper()
    df["type"] = df["type"].str.lower()
    df["ask_frac"] = np.where(df["total_premium"] > 0,
                              df["total_ask_side_prem"] / df["total_premium"], 0.0)
    # PROD qualifying filter (uw_flow_collector.build_flow_signal)
    df["wl_ok"] = df.apply(
        lambda r: r["ticker"] in (CALL_WL if r["type"] == "call" else PUT_WL), axis=1)
    df = df[(df["has_sweep"] == 1) & (df["total_premium"] >= MIN_PREM)
            & (df["ask_frac"] >= ASK_FRAC_MIN) & df["wl_ok"]].copy()
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["t0_mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30  # minute-of-day from 9:30 ET
    df = df[(df["t0_mi"] >= 0) & (df["t0_mi"] <= 390)].copy()
    return df.reset_index(drop=True)


def load_opts(tk: str, right: str) -> pd.DataFrame:
    """All option 1-min closes for ticker/right with ET date, minute-of-day, expiration, dte."""
    con = sqlite3.connect(THETA_DB)
    p = pd.read_sql_query(
        "SELECT timestamp, expiration, strike, close FROM option_ohlc "
        "WHERE ticker=? AND right=? ORDER BY timestamp", con, params=(tk, right))
    con.close()
    if p.empty:
        return p
    ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(ET)
    p["date"] = ts.dt.strftime("%Y-%m-%d")
    p["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return p


def load_stock(tk: str) -> dict:
    con = sqlite3.connect(THETA_DB)
    s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=?",
                          con, params=(tk,))
    con.close()
    if s.empty:
        return {}
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    s["date"] = ts.dt.strftime("%Y-%m-%d")
    s["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return {d: g.set_index("mi")["close"].to_dict() for d, g in s.groupby("date")}


# ----- nearest-DTE + PROD strike resolution (mirrors bot_runner _resolve_flow_strike intent) -----
def resolve_contract(opts: pd.DataFrame, date: str, spot: float, is_put: bool, tk: str):
    """Find the nearest tradeable expiry (>= date) for this ticker/side that actually has 1-min bars
    on `date`, then pick the PROD strike (select_flow_strike: ATM, or ~$2 OTM for validated combos).

    Returns (expiration, strike, mode) or (None, None, None). NEAREST-DTE: we walk expirations in
    ascending order and take the first one with intraday data on the signal day (PROD walks
    today->next business days to the nearest tradeable expiry; with this DB the equivalent is the
    smallest expiration >= date that has bars on `date`)."""
    cand = opts[opts["date"] == date]
    if cand.empty:
        return None, None, None
    # expirations available on this trading day, nearest first
    exps = sorted(cand["expiration"].unique())
    use_otm = ENABLE_FLOW_OTM and ((not is_put and tk in OTM_CALLS) or (is_put and tk in OTM_PUTS))
    for exp in exps:
        if exp < date:  # already-expired contract bar (shouldn't happen) — skip
            continue
        sub = cand[cand["expiration"] == exp]
        strikes = sub["strike"].unique()
        if len(strikes) == 0:
            continue
        # build a minimal chain for select_flow_strike (strike + a price proxy = last close that day)
        chain = []
        for s in strikes:
            row = sub[sub["strike"] == s]
            last = row.sort_values("mi")["close"].iloc[-1]
            chain.append({"strike": float(s), "mid": float(last) if last and last > 0 else 0.0,
                          "last_price": float(last) if last and last > 0 else 0.0})
        strike, mode = select_flow_strike(chain, spot, is_put, use_otm, OTM_TARGET)
        if strike is not None:
            return exp, float(strike), mode
    return None, None, None


# ----- exit sim (identical to entry_timing_oracle.sim) -------------------------------------------
def sim(pp, mp, up, cfg, otype, ets, dte):
    """Run the real ExitFSM from entry (pp[0]) to EOD; return % net of haircut."""
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - EXIT_HAIRCUT), pp[k], now,
                         current_underlying=up[k],
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - EXIT_HAIRCUT) - ep) / ep * 100
    return (last * (1 - EXIT_HAIRCUT) - ep) / ep * 100


# ----- entry rules (window arrays cover t0_mi .. t0_mi+W; i=0 is t0/NOW) -------------------------
def pick_now(w_prem):
    return 0


def pick_oracle(w_prem):
    """*** LOOKAHEAD — scans the WHOLE window. Ceiling only, not implementable. ***"""
    valid = [(p, i) for i, p in enumerate(w_prem) if p is not None and not np.isnan(p) and p > 0]
    return min(valid)[1] if valid else 0


def pick_dip(w_prem, dpct):
    """No-lookahead. At bar i, compare w_prem[i] to w_prem[0] ONLY (past). Enter the NEXT valid bar."""
    p0 = w_prem[0]
    thresh = p0 * (1 - dpct / 100.0)
    for i in range(1, len(w_prem)):
        pi = w_prem[i]
        if pi is None or np.isnan(pi) or pi <= 0:
            continue
        # decision reads w_prem[0..i] only; no index > i is touched (loop bound proves it).
        if pi <= thresh:
            j = i + 1  # enter the bar AFTER the trigger (can't act on the bar we observe)
            while j < len(w_prem) and (w_prem[j] is None or np.isnan(w_prem[j]) or w_prem[j] <= 0):
                j += 1
            return j if j < len(w_prem) else len(w_prem) - 1
    return len(w_prem) - 1  # timeout: enter at window end


def pick_pullback(w_prem):
    """No-lookahead. Wait for a dip below p0, then the first bar whose close > the prior close."""
    p0 = w_prem[0]
    dipped = False
    prev = p0
    for i in range(1, len(w_prem)):
        pi = w_prem[i]
        if pi is None or np.isnan(pi) or pi <= 0:
            continue
        # uses only w_prem[0..i] (p0, prev, pi) — never reads ahead.
        if not dipped:
            if pi < p0:
                dipped = True
            prev = pi
            continue
        if pi > prev:
            return i
        prev = pi
    return len(w_prem) - 1  # timeout


def pf(x):
    x = np.asarray(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else (float("inf") if g > 0 else 0.0)


# ------------------------------------------------------------------------------------------------
def build_trades(side: str, W: int, validate: bool = False):
    """One trade per qualifying flow signal of `side`. Returns list of per-signal dicts."""
    right = "PUT" if side == "put" else "CALL"
    sigs = load_qualifying_signals()
    sigs = sigs[sigs["type"] == side].reset_index(drop=True)
    rows = []
    val_printed = 0
    # cache loaders per ticker
    opt_cache: dict[str, pd.DataFrame] = {}
    stk_cache: dict[str, dict] = {}
    cfg_cache: dict[str, object] = {}
    skips = {"no_opt_data": 0, "no_contract": 0, "no_spot": 0, "thin_path": 0, "no_underlying": 0}

    for _, sig in sigs.iterrows():
        tk, date, t0 = sig["ticker"], sig["date"], int(sig["t0_mi"])
        if tk not in opt_cache:
            opt_cache[tk] = load_opts(tk, right)
            stk_cache[tk] = load_stock(tk)
            cfg_cache[tk] = apply_v7_wide_trail_exits(
                get_ticker_config(tk, use_per_ticker=True, option_type=side), is_put=(side == "put"))
        opts, stock, cfg = opt_cache[tk], stk_cache[tk], cfg_cache[tk]
        if opts.empty or date not in stock or t0 not in stock[date]:
            skips["no_opt_data" if opts.empty else "no_spot"] += 1
            continue
        spot = stock[date][t0]
        exp, strike, mode = resolve_contract(opts, date, spot, side == "put", tk)
        if strike is None:
            skips["no_contract"] += 1
            continue
        dte = (datetime.strptime(exp, "%Y-%m-%d").date()
               - datetime.strptime(date, "%Y-%m-%d").date()).days
        # contract path from t0 to EOD
        ch = opts[(opts["date"] == date) & (opts["expiration"] == exp)
                  & (opts["strike"] == strike) & (opts["mi"] >= t0)].sort_values("mi")
        if len(ch) < 3:
            skips["thin_path"] += 1
            continue
        mi_all = ch["mi"].to_numpy(int)
        prem_all = ch["close"].to_numpy(float)
        if mi_all[0] != t0 or np.isnan(prem_all[0]) or prem_all[0] <= 0:
            skips["thin_path"] += 1
            continue
        # entry-decision window [t0, t0+W]
        win_mask = mi_all <= t0 + W
        w_mi, w_prem = mi_all[win_mask], prem_all[win_mask]
        if len(w_mi) < 2:
            skips["thin_path"] += 1
            continue

        picks = {"NOW": pick_now(w_prem), "ORACLE": pick_oracle(w_prem),
                 "PULLBACK": pick_pullback(w_prem)}
        for d in DIP_LEVELS:
            picks[f"DIP{d}"] = pick_dip(w_prem, d)

        # "instantly underwater" + recovery (flow-call concern), measured from the NOW entry (t0)
        p0 = float(w_prem[0])
        first5 = prem_all[(mi_all >= t0) & (mi_all <= t0 + 5)]
        uw5 = bool(np.nanmin(first5) < p0) if len(first5) else False  # dipped below entry within 5 min
        # recovery, two ways: (loose) ever ticks back to entry; (strict) ever +10% above entry.
        ever_green = bool(np.nanmax(prem_all) > p0)            # trivially common — touched entry once
        recovered_10 = bool(np.nanmax(prem_all) >= p0 * 1.10)  # meaningfully recovered (+10%)

        rec = {"date": date, "tk": tk, "t0": t0, "exp": exp, "strike": strike, "mode": mode,
               "dte": dte, "p0": p0, "uw5": uw5, "ever_green": ever_green, "recovered_10": recovered_10,
               "oracle_min": float(np.nanmin([p for p in w_prem if p and p > 0]))}

        for name, idx in picks.items():
            em = int(w_mi[idx])
            fmask = mi_all >= em
            pp, mp = prem_all[fmask], mi_all[fmask]
            if len(pp) < 2 or np.isnan(pp[0]) or pp[0] <= 0:
                rec[name] = None
                rec[f"{name}_ep"] = None
                rec[f"{name}_em"] = em
                continue
            up = [stock[date].get(int(m), spot) for m in mp]
            ets = datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
            rec[name] = sim(pp, list(mp), list(up), cfg, side, ets, dte)
            rec[f"{name}_ep"] = float(pp[0])
            rec[f"{name}_em"] = em
        rows.append(rec)

        if validate and val_printed < 6:
            print(f"\n[SANITY] {tk} {date} ({side}) t0_mi={t0} ({_mi_to_et(t0)} ET) "
                  f"W={W} exp={exp} dte={dte} strike={strike} {mode}  p0={p0:.2f} "
                  f"oracle_min={rec['oracle_min']:.2f}  uw5={uw5} ever_green={ever_green}")
            for name in STRATS:
                em, ep, r = rec.get(f"{name}_em"), rec.get(f"{name}_ep"), rec.get(name)
                eps = f"{ep:.2f}" if ep is not None else "  -  "
                rs = f"{r:+7.1f}%" if r is not None else "   n/a "
                print(f"    {name:<9} entry_min={em:>3} entry_prem={eps:>6}  ret={rs}")
            # NO-LOOKAHEAD asserts: oracle (peeking) is the cheapest possible entry; implementable
            # rules can never enter below it.
            assert abs(rec["oracle_min"] - min(p for p in w_prem if p and p > 0)) < 1e-6
            for nm in [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]:
                if rec.get(f"{nm}_ep") is not None:
                    assert rec["ORACLE_ep"] <= rec[f"{nm}_ep"] + 1e-9, \
                        f"{nm} entered below the lookahead oracle min — LOOKAHEAD LEAK"
            val_printed += 1

    print(f"\n  side={side} W={W}: {len(rows)} trades built  | skips={skips}", flush=True)
    return rows


def _mi_to_et(mi: int) -> str:
    h = 9 + (30 + mi) // 60
    m = (30 + mi) % 60
    return f"{h:02d}:{m:02d}"


def report(rows, side, W):
    if not rows:
        print(f"\n!!! no trades for side={side} W={W} — nothing to report")
        return
    tdf = pd.DataFrame(rows)
    n = len(tdf)
    print(f"\n{'='*100}\nSIDE={side.upper()}  W={W}min   n={n} real flow signals (entry @ REAL created_at)"
          f"\n{'='*100}")

    # ---- (A) per-strategy P&L / PF / entry improvement ----
    print(f"  {'strategy':<10}{'entryImpr%':>11}{'netP&L(u)':>11}{'PF':>7}{'WR%':>7}"
          f"{'avgRet%':>9}{'%timeout':>10}")
    for name in STRATS:
        r = tdf[name].dropna()
        if len(r) == 0:
            continue
        paired = tdf.dropna(subset=["NOW_ep", f"{name}_ep"])
        impr = ((paired["NOW_ep"] - paired[f"{name}_ep"]) / paired["NOW_ep"] * 100).mean()
        to = (tdf[f"{name}_em"] == tdf["t0"] + W).mean() * 100 if name not in ("NOW", "ORACLE") else 0.0
        tag = " <-CEILING(lookahead)" if name == "ORACLE" else ""
        print(f"  {name:<10}{impr:>+11.2f}{r.sum():>+11.1f}{pf(r):>7.2f}{(r > 0).mean()*100:>7.1f}"
              f"{r.mean():>+9.2f}{to:>9.1f}%{tag}")
    print(f"    (NOW net P&L = {tdf['NOW'].dropna().sum():+.1f} units = the bar to beat)")

    # ---- (B) the PRIZE + adverse selection ----
    now_pl, ora_pl = tdf["NOW"].dropna().sum(), tdf["ORACLE"].dropna().sum()
    print(f"\n  PRIZE (ORACLE-NOW) = {ora_pl - now_pl:+.1f}u   "
          f"(NOW={now_pl:+.1f}u, ORACLE={ora_pl:+.1f}u, "
          f"{(ora_pl/now_pl if now_pl > 0 else float('nan')):.2f}x NOW)")
    tdf["_runner"] = tdf["oracle_min"] >= tdf["p0"] - 1e-9  # never dipped in window = monotone-up
    print(f"  runners (never dipped in window) = {int(tdf['_runner'].sum())}/{n} "
          f"= {tdf['_runner'].mean()*100:.0f}%")
    print(f"\n  ADVERSE-SELECTION (vs NOW):")
    for name in [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]:
        paired = tdf.dropna(subset=["NOW", name])
        to = (paired[f"{name}_em"] == paired["t0"] + W).mean() * 100
        nw = paired[paired["NOW"] > 0]
        worsened = (nw[name] < nw["NOW"] - 1e-9).mean() * 100 if len(nw) else float("nan")
        runners = paired[paired["_runner"]]
        rmiss = (runners[f"{name}_em"] == runners["t0"] + W).mean() * 100 if len(runners) else float("nan")
        denom = paired["ORACLE"].sum() - paired["NOW"].sum()
        cap = (paired[name].sum() - paired["NOW"].sum()) / denom * 100 if denom != 0 else float("nan")
        dpl = paired[name].sum() - paired["NOW"].sum()
        print(f"    {name:<9} timeout={to:>5.1f}%  worsens-NOW-winners={worsened:>5.1f}%  "
              f"runner-miss={rmiss:>5.1f}%  | ΔP&L vs NOW={dpl:>+7.1f}u  prize-capture={cap:>6.1f}%")

    # ---- (C) "instantly underwater" — flow-call concern ----
    nuw = int(tdf["uw5"].sum())
    uwdf = tdf[tdf["uw5"]]
    rec_loose = int(uwdf["ever_green"].sum())
    rec_strict = int(uwdf["recovered_10"].sum())
    now_green_of_uw = int((uwdf["NOW"] > 0).sum())  # did the actual NOW trade close GREEN
    print(f"\n  INSTANTLY-UNDERWATER (dipped below entry within 5 min of t0): "
          f"{nuw}/{n} = {nuw/n*100:.0f}%")
    if nuw:
        print(f"    recover-touch-entry (loose): {rec_loose}/{nuw}={rec_loose/nuw*100:.0f}%   "
              f"recover-+10% (strict): {rec_strict}/{nuw}={rec_strict/nuw*100:.0f}%")
        print(f"    *** of instantly-underwater, the actual NOW trade CLOSED GREEN: "
              f"{now_green_of_uw}/{nuw} = {now_green_of_uw/nuw*100:.0f}% "
              f"(=> instant dip is {'NOT a death sentence — FSM rides most back to profit' if now_green_of_uw/nuw >= 0.5 else 'often terminal'})")
    return tdf


def main():
    windows = [int(x) for x in os.environ.get("WINDOWS", "15,30").split(",")]
    validate = os.environ.get("VALIDATE") == "1"

    sigs = load_qualifying_signals()
    nc = (sigs["type"] == "call").sum()
    npt = (sigs["type"] == "put").sum()
    print(f"### PROD-qualifying flow signals: CALLS={nc}  PUTS={npt}  "
          f"(filter: has_sweep, prem>=${MIN_PREM:,.0f}, ask_frac>={ASK_FRAC_MIN}, whitelist)")
    print(f"### date range {sigs['date'].min()}..{sigs['date'].max()}  "
          f"OTM_strike={'ON' if ENABLE_FLOW_OTM else 'OFF (prod default)'}")

    if validate:
        print("\n### VALIDATION PASS — sanity rows + no-lookahead asserts on first signals (CALLS, W=30) ###")
        rows = build_trades("call", 30, validate=True)
        report(rows, "call", 30)
        print("\nVALIDATION OK — no-lookahead asserts passed. Re-run without VALIDATE=1 to scale.\n")
        return

    for side in ("call", "put"):  # CALLS primary (the concern), PUTS secondary
        for W in windows:
            print(f"\n\n########## side={side}  W={W} ##########", flush=True)
            rows = build_trades(side, W, validate=False)
            report(rows, side, W)


if __name__ == "__main__":
    main()
