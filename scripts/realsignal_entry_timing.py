"""REAL-SIGNAL ENTRY-TIMING STUDY — does "optimal entry timing" beat buy-now on the bot's ACTUAL entries?

This re-checks the entry-timing question on the bot's REAL placed trades (NO 10:00 proxy). A prior PROXY
study (scripts/entry_timing_oracle.py, ATM-0DTE candidate at mi0=30 per ticker/day) found timing
UNEXPLOITABLE — waiting catches falling knives / misses rippers (adverse selection). Here we swap the entry
SET to the EXACT contracts the bot bought (journal/real_signals_kody.csv), at their REAL entry minute, and
re-run the SAME no-lookahead rules + the SAME real ExitFSM (V7 wide-trail + profit-lock LOCK cfg).

REUSES entry_timing_oracle.py machinery verbatim where possible:
  - sim()  (real ExitFSM from entry minute to EOD, EXIT_HAIRCUT on fills)
  - pick_now / pick_oracle / pick_dip / pick_pullback  (no-lookahead entry rules, same code)
  - LOCK cfg (V7 wide-trail + profit-lock)
  - D.apply_v7_wide_trail_exits / D.get_ticker_config  (same per-ticker exit cfg as the harness)

KEY DIFFERENCE vs the proxy study: we do NOT re-resolve an ATM strike. Each CSV row IS the contract the bot
bought (ticker, expiration=expiry_date, strike, right). We pull THAT contract's 1-min closes from the REAL
entry minute (opened_at, stored UTC -> converted to ET) to EOD. Rows whose exact contract is absent from the
thetadata DB are SKIPPED and reported (the DB only stocks a thin ~5-strike ATM band per expiry, so the bot's
exact strike is sometimes just outside it — a data-coverage gap, not a logic gap).

Arms reported SEPARATELY (REAL_ML primary, REAL_DISCORD larger-n confirmation), calls/puts SEPARATELY, n on
every cell. No-lookahead discipline is inherited from entry_timing_oracle.py's rule functions (each only reads
window bars up to the decision minute; entry on the bar AFTER the trigger). ORACLE is the lookahead ceiling.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/realsignal_entry_timing.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402  (provides DB, ET, EXIT_HAIRCUT, cfg helpers)
import entry_timing_oracle as O  # noqa: E402  (REUSE sim, pick_now/oracle/dip/pullback, LOCK, pf)

DB, ET = D.DB, D.ET
CSV = str(Path(__file__).resolve().parent.parent / "journal" / "real_signals_kody.csv")
WINDOWS = [15, 30]
DIP_LEVELS = O.DIP_LEVELS  # [5, 10, 15]


def load_contract(con, ticker, expiry, strike, right):
    """1-min closes for the EXACT contract the bot bought. Returns df[mi, close] (mi from 9:30 ET) or None."""
    q = pd.read_sql_query(
        "SELECT timestamp, close FROM option_ohlc "
        "WHERE ticker=? AND expiration=? AND strike=? AND right=? ORDER BY timestamp",
        con, params=(ticker, expiry, float(strike), right))
    if q.empty:
        return None
    ts = pd.to_datetime(q["timestamp"], utc=True).dt.tz_convert(ET)
    q = q.assign(mi=(ts.dt.hour - 9) * 60 + ts.dt.minute - 30, date=ts.dt.strftime("%Y-%m-%d"))
    return q


def build_real_trades(df, W, validate_rows=0):
    """One record per matched CSV row, with each strategy's (entry_idx, entry_prem, ret)."""
    con = sqlite3.connect(DB)
    rows, skip_nocontract, skip_noentry = [], [], []
    stock_cache = {}
    val_printed = 0

    for _, r in df.iterrows():
        side = "put" if r["option_type"] == "put" else "call"
        right = "PUT" if side == "put" else "CALL"
        ch = load_contract(con, r["ticker"], r["expiry_date"], r["strike"], right)
        if ch is None or ch.empty:
            skip_nocontract.append(r)
            continue

        # REAL entry minute: opened_at is stored UTC -> convert to ET -> minutes from 9:30
        oa = pd.Timestamp(r["opened_at"], tz="UTC").tz_convert(ET)
        date = oa.strftime("%Y-%m-%d")
        emi0 = (oa.hour - 9) * 60 + oa.minute - 30

        ch = ch[ch["date"] == date]
        if ch.empty:
            skip_noentry.append(r)
            continue

        # underlying feed for this ticker/day (matches harness up[])
        if r["ticker"] not in stock_cache:
            stock_cache[r["ticker"]] = D._stock(r["ticker"])
        day_stock = stock_cache[r["ticker"]].get(date, {})
        spot0 = day_stock.get(emi0, None)

        mi_all = ch["mi"].to_numpy(int)
        prem_all = ch["close"].to_numpy(float)

        # path from the REAL entry minute onward (find first bar at/after entry minute)
        start_mask = mi_all >= emi0
        if start_mask.sum() < 2:
            skip_noentry.append(r)
            continue
        mi_all = mi_all[start_mask]
        prem_all = prem_all[start_mask]
        # t0 is the first available bar at/after the real entry minute
        mi0 = int(mi_all[0])
        if np.isnan(prem_all[0]) or prem_all[0] <= 0:
            skip_noentry.append(r)
            continue

        # entry-decision window [t0, t0+W]
        win_mask = mi_all <= mi0 + W
        w_mi = mi_all[win_mask]
        w_prem = prem_all[win_mask]
        if len(w_mi) < 2:
            skip_noentry.append(r)
            continue

        # per-ticker V7 wide-trail exit cfg (same as harness/oracle)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(r["ticker"], use_per_ticker=True, option_type=side), is_put=(side == "put"))

        # ---- choose entry index per strategy (REUSE oracle's no-lookahead rule fns) ----
        picks = {
            "NOW": O.pick_now(w_prem),
            "ORACLE": O.pick_oracle(w_prem),
            "PULLBACK": O.pick_pullback(w_prem),
        }
        for d in DIP_LEVELS:
            picks[f"DIP{d}"] = O.pick_dip(w_prem, d)

        rec = {
            "id": r["id"], "tk": r["ticker"], "side": side, "arm": r["arm"], "date": date,
            "p0": float(w_prem[0]),
            "oracle_min": float(np.nanmin([p for p in w_prem if p and p > 0])),
        }
        # instant-underwater diagnostics off the contract path (independent of which rule we pick)
        rec["_iu5"], rec["_iu5_recovers"] = _instant_underwater(mi_all, prem_all, mi0)

        for name, idx in picks.items():
            em = int(w_mi[idx])
            fmask = mi_all >= em
            pp = prem_all[fmask]
            mp = mi_all[fmask]
            if len(pp) < 2 or np.isnan(pp[0]) or pp[0] <= 0:
                rec[name] = None
                rec[f"{name}_ep"] = None
                rec[f"{name}_em"] = em
                continue
            up = [day_stock.get(int(m), spot0 if spot0 else pp[0]) for m in mp]
            ets = datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=em)
            ret = O.sim(pp, list(mp), list(up), cfg, side, ets)  # REUSE oracle sim()
            rec[name] = ret
            rec[f"{name}_ep"] = float(pp[0])
            rec[f"{name}_em"] = em
        rows.append(rec)

        if validate_rows and val_printed < validate_rows:
            print(f"\n[PROOF] id={rec['id']} {rec['tk']} {date} ({side}) arm={rec['arm']} W={W}  "
                  f"t0(mi)={mi0}  t0_price={rec['p0']:.3f}  oracle_min(window)={rec['oracle_min']:.3f}")
            for name in ["NOW", "ORACLE"] + [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]:
                em, ep, ret = rec.get(f"{name}_em"), rec.get(f"{name}_ep"), rec.get(name)
                eps = f"{ep:.3f}" if ep is not None else "  -  "
                rs = f"{ret:+7.1f}%" if ret is not None else "   n/a "
                tag = " <-LOOKAHEAD CEILING" if name == "ORACLE" else ""
                print(f"    {name:<9} entry_min={em:>3} entry_prem={eps:>7}  ret={rs}{tag}")
            # no-lookahead self-check: ORACLE entry price is the cheapest of any rule (it peeks)
            assert abs(rec["oracle_min"] - min(p for p in w_prem if p and p > 0)) < 1e-6
            if rec.get("DIP5_ep") is not None:
                assert rec["ORACLE_ep"] <= rec["DIP5_ep"] + 1e-9, "oracle must be cheapest entry"
            if rec.get("PULLBACK_ep") is not None:
                assert rec["ORACLE_ep"] <= rec["PULLBACK_ep"] + 1e-9, "oracle must be cheapest entry"
            val_printed += 1

    con.close()
    return rows, skip_nocontract, skip_noentry


def _instant_underwater(mi_all, prem_all, mi0):
    """Off the contract path: did the entry go underwater within 5 min of t0, and if so did it ever
    recover to green (close > p0) afterward? Returns (is_underwater_5min, recovers_to_green | None)."""
    p0 = prem_all[0]
    win5 = (mi_all >= mi0) & (mi_all <= mi0 + 5)
    underwater = bool(np.any(prem_all[win5] < p0))
    if not underwater:
        return False, None
    # of those that went red within 5 min: did the contract EVER trade above p0 again (rest of day)?
    recovers = bool(np.any(prem_all[mi_all > mi0] > p0))
    return True, recovers


def pf(x):
    return O.pf(x)


def _cell(sub, strategies):
    """Print the per-strategy table for a slice (already filtered to arm/side)."""
    n = len(sub)
    print(f"  n={n}{'   *LOW-N — directional only*' if n < 30 else ''}")
    print(f"  {'strategy':<10}{'entryImpr%':>11}{'netP&L(u)':>11}{'PF':>7}{'WR%':>7}"
          f"{'avgRet%':>9}{'%timeout':>10}")
    for name in strategies:
        r = sub[name].dropna()
        if len(r) == 0:
            continue
        paired = sub.dropna(subset=["NOW_ep", f"{name}_ep"])
        impr = ((paired["NOW_ep"] - paired[f"{name}_ep"]) / paired["NOW_ep"] * 100).mean() if len(paired) else float("nan")
        wr = (r > 0).mean() * 100
        # timeout %: the rule entered at the window end relative to its own t0 (t0 = NOW_em)
        to = ((sub[f"{name}_em"] - sub["NOW_em"]) >= WIN_GLOBAL).mean() * 100 if name not in ("NOW", "ORACLE") else 0.0
        tag = " <-CEILING" if name == "ORACLE" else ""
        print(f"  {name:<10}{impr:>+11.2f}{r.sum():>+11.0f}{pf(r):>7.2f}{wr:>7.1f}{r.mean():>+9.2f}{to:>9.1f}%{tag}")


def _adverse(sub, strategies_nl):
    """Adverse-selection accounting for a slice."""
    print(f"  {'rule':<9}{'timeout%':>9}{'worsensNOWwin%':>16}{'runnerMiss%':>13}"
          f"{'ΔP&L vs NOW':>13}{'prizeCap%':>11}")
    now_pl = sub["NOW"].dropna().sum()
    ora_pl = sub["ORACLE"].dropna().sum()
    prize = ora_pl - now_pl
    sub = sub.copy()
    sub["_runner"] = sub["oracle_min"] >= sub["p0"] - 1e-9  # never dipped below entry in window
    for name in strategies_nl:
        paired = sub.dropna(subset=["NOW", name])
        if len(paired) == 0:
            continue
        to = ((paired[f"{name}_em"] - paired["NOW_em"]) >= WIN_GLOBAL).mean() * 100
        nw = paired[paired["NOW"] > 0]
        worsened = (nw[name] < nw["NOW"] - 1e-9).mean() * 100 if len(nw) else float("nan")
        runners = paired[paired["_runner"]]
        runner_miss = ((runners[f"{name}_em"] - runners["NOW_em"]) >= WIN_GLOBAL).mean() * 100 if len(runners) else float("nan")
        delta = paired[name].sum() - paired["NOW"].sum()
        cap = (delta / prize * 100) if prize != 0 else float("nan")
        print(f"  {name:<9}{to:>8.1f}%{worsened:>15.1f}%{runner_miss:>12.1f}%"
              f"{delta:>+13.0f}{cap:>10.1f}%")
    nr = int(sub["_runner"].sum())
    print(f"    runners (never dipped below entry in window) = {nr}/{len(sub)} = {sub['_runner'].mean()*100:.0f}%"
          f"   |  PRIZE (ORACLE-NOW) = {prize:+.0f}u")


def report(rows, W):
    global WIN_GLOBAL
    WIN_GLOBAL = W
    tdf = pd.DataFrame(rows)
    strategies = ["NOW", "ORACLE"] + [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]
    strategies_nl = [f"DIP{d}" for d in DIP_LEVELS] + ["PULLBACK"]

    print(f"\n\n{'#'*100}\n# WINDOW W={W}min   (entry decision at the REAL opened_at minute)\n{'#'*100}")

    for arm in ["REAL_ML", "REAL_DISCORD"]:
        for side in ["call", "put"]:
            sub = tdf[(tdf["arm"] == arm) & (tdf["side"] == side)]
            if len(sub) == 0:
                continue
            print(f"\n{'='*100}\n{arm}  /  {side.upper()}   W={W}\n{'='*100}")
            _cell(sub, strategies)
            now_pl, ora_pl = sub["NOW"].dropna().sum(), sub["ORACLE"].dropna().sum()
            print(f"  PRIZE: NOW={now_pl:+.0f}u  ORACLE={ora_pl:+.0f}u (ceiling)  "
                  f"prize=+{ora_pl-now_pl:.0f}u  ({(ora_pl/now_pl if now_pl>0 else float('nan')):.2f}x NOW)")
            print("  -- adverse selection --")
            _adverse(sub, strategies_nl)

    # arm-level combined (calls+puts) — the headline read
    for arm in ["REAL_ML", "REAL_DISCORD"]:
        sub = tdf[tdf["arm"] == arm]
        if len(sub) == 0:
            continue
        print(f"\n{'-'*100}\n{arm} — COMBINED calls+puts   W={W}   (n={len(sub)})\n{'-'*100}")
        _cell(sub, strategies)
        print("  -- adverse selection --")
        _adverse(sub, strategies_nl)

    # instant-underwater (off contract path, independent of W within the day)
    print(f"\n{'-'*100}\nINSTANT-UNDERWATER (within 5 min of t0)  W={W}\n{'-'*100}")
    for arm in ["REAL_ML", "REAL_DISCORD"]:
        for side in ["call", "put", "all"]:
            s = tdf[tdf["arm"] == arm] if side == "all" else tdf[(tdf["arm"] == arm) & (tdf["side"] == side)]
            if len(s) == 0:
                continue
            iu = s["_iu5"]
            iu_n = int(iu.sum())
            iu_pct = iu.mean() * 100
            among = s[s["_iu5"]]
            rec_pct = among["_iu5_recovers"].mean() * 100 if len(among) else float("nan")
            print(f"  {arm:<13} {side:<5} n={len(s):<4} underwater<5min={iu_pct:>5.1f}% ({iu_n})  "
                  f"| of those, EVER recover to green={rec_pct:>5.1f}%  (else terminal red)")

    return tdf


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} real signals from {CSV}")
    print(f"  by arm: {df['arm'].value_counts().to_dict()}")
    print(f"  by side: {df['option_type'].value_counts().to_dict()}")

    # ---------- VALIDATION PASS: prove matching + no-lookahead on a few rows ----------
    print(f"\n{'='*100}\nVALIDATION PASS — proof rows (contract match + no-lookahead) on first matched trades\n{'='*100}")
    _rows, snc, sne = build_real_trades(df.head(40), 30, validate_rows=3)
    print("\nVALIDATION OK — contract paths matched, no-lookahead asserts passed (ORACLE is cheapest entry).")

    # ---------- FULL RUN ----------
    for W in WINDOWS:
        rows, skip_nocontract, skip_noentry = build_real_trades(df, W)
        tdf = pd.DataFrame(rows)
        snc_df = pd.DataFrame(skip_nocontract)
        sne_df = pd.DataFrame(skip_noentry)
        print(f"\n\n{'='*100}\nMATCH SUMMARY (W={W})\n{'='*100}")
        print(f"  matched: {len(tdf)}   skipped(no contract in DB): {len(snc_df)}   "
              f"skipped(no entry path): {len(sne_df)}")
        for arm in ["REAL_ML", "REAL_DISCORD"]:
            m = (tdf["arm"] == arm).sum() if len(tdf) else 0
            nc = (snc_df["arm"] == arm).sum() if len(snc_df) else 0
            ne = (sne_df["arm"] == arm).sum() if len(sne_df) else 0
            mc = ((tdf["arm"] == arm) & (tdf["side"] == "call")).sum() if len(tdf) else 0
            mp = ((tdf["arm"] == arm) & (tdf["side"] == "put")).sum() if len(tdf) else 0
            print(f"    {arm:<13} matched={m} ({mc}c/{mp}p)  skipped_nocontract={nc}  skipped_noentry={ne}")
        report(rows, W)


if __name__ == "__main__":
    main()
