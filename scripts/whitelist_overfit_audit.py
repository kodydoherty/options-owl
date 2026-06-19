"""WHITELIST / TICKER-UNIVERSE OVERFIT AUDIT.

For EACH of the 31 thetadata tickers, run the ATM 0DTE CALL proxy AND the PUT proxy
(10:00 ET entry, real V7+profit-lock ExitFSM, same exit haircut as backtest_2yr_regime.py),
computing PF / WR / total-return / n SEPARATELY for 2024 / 2025 / 2026.

Goal: is each ticker actually good for calls / for puts across the FULL 2.5yr, or only good
in 2026? Flag CURVE-FIT names (whitelisted but PF<1 in 2024 or 2025) and MISSED names (not
whitelisted but PF>1 in all 3 years). Produce a 2.5yr-validated recommended call & put list.

This is a GENERALIZATION test. ATM-0DTE proxy != flow-sweep edge, but flow signals only span
~3mo so this is the best 2.5yr ticker-universe test we have. Read-only.

Usage: cd /Users/kody/dev/options-owl && python scripts/whitelist_overfit_audit.py
       (add --quick TK1,TK2 MONTH-PREFIX for a sanity-check subset, e.g. --quick SPY,TSLA 2024-01)
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
ENTRY_MI = 30  # 10:00 ET

ALL_TICKERS = ["AAPL", "AMD", "AMZN", "ARM", "ASML", "AVGO", "CRWD", "DELL", "GLD",
               "GOOG", "GOOGL", "IBM", "INTC", "IWM", "LRCX", "META", "MRVL", "MSFT",
               "MSTR", "MU", "NVDA", "ORCL", "PLTR", "QCOM", "QQQ", "SLV", "SMH",
               "SOXX", "SPY", "TSLA", "TSM"]

# Current production lists (from CLAUDE.md / settings.py)
CALL_WL = {"TSLA", "AAPL", "AMD", "PLTR", "META", "SPY", "AMZN"}
PUT_WL = {"META", "AMZN", "AAPL", "TSLA", "MU", "SPY"}
CALL_BLACK = {"PLTR", "AMD", "MSTR", "GOOGL"}  # calls historically bad

LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)


def load_0dte(tk, month_filter=None):
    con = sqlite3.connect(DB)
    q = ("SELECT strike, right, timestamp, close FROM option_ohlc "
         "WHERE ticker=? AND expiration = substr(timestamp,1,10)")
    params = [tk]
    if month_filter:
        q += " AND substr(timestamp,1,7)=?"
        params.append(month_filter)
    q += " ORDER BY timestamp"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


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
                         current_underlying=up[k],
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def run_ticker(tk, month_filter=None):
    """Return list of (year, side, ret)."""
    df = load_0dte(tk, month_filter)
    if df.empty:
        return []
    stock = D._stock(tk)
    cfg_c = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
    cfg_p = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="put"), is_put=True)
    out = []
    for date, g in df.groupby("date"):
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
            out.append((yr, side, ret))
    return out


def pf(x):
    x = np.array(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    if l <= 0:
        return float("inf") if g > 0 else 0.0
    return g / l


def stats(rets):
    a = np.array(rets, float)
    return dict(n=len(a), pf=pf(a), wr=float(np.mean(a > 0)), total=float(a.sum()))


def main():
    quick = None
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick = (sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else None)
        tickers = quick[0]
        month_filter = quick[1]
        print(f"QUICK SANITY: tickers={tickers} month={month_filter}\n", flush=True)
    else:
        tickers = ALL_TICKERS
        month_filter = None

    # rows[(tk, side, year)] = list of returns
    rows = defaultdict(list)
    for tk in tickers:
        tr = run_ticker(tk, month_filter)
        for yr, side, ret in tr:
            rows[(tk, side, yr)].append(ret)
        nc = sum(1 for _, s, _ in tr if s == "call")
        npp = sum(1 for _, s, _ in tr if s == "put")
        print(f"  {tk:<6} call={nc:>4} put={npp:>4}", flush=True)

    if quick:
        print("\n--- QUICK sanity rows ---")
        for (tk, side, yr), r in sorted(rows.items()):
            st = stats(r)
            print(f"{tk:<6}{side:<5}{yr}  n={st['n']:>3} PF={st['pf']:>5.2f} "
                  f"WR={st['wr']*100:>4.0f}% tot={st['total']:>+8.0f}")
        return

    years = ["2024", "2025", "2026"]

    # ---- Build per-side per-ticker per-year table ----
    def side_table(side, wl, black=None):
        print(f"\n{'='*92}")
        print(f"{side.upper()} ATM-0DTE proxy — per ticker per year   (WL=in current whitelist"
              + (", BL=blacklist" if black else "") + ")")
        print(f"{'='*92}")
        hdr = f"{'tk':<6}{'flag':<5}"
        for y in years:
            hdr += f"|{'n':>4}{'PF':>6}{'WR':>5}{'tot%':>8} "
        print(hdr)
        print("-" * len(hdr))
        table = {}
        for tk in tickers:
            table[tk] = {y: stats(rows.get((tk, side, y), [])) for y in years}
        # sort by all-years total
        order = sorted(tickers, key=lambda t: -sum(table[t][y]["total"] for y in years))
        for tk in order:
            flag = ""
            if tk in wl:
                flag = "WL"
            if black and tk in black:
                flag = "BL" if not flag else flag + "/BL"
            line = f"{tk:<6}{flag:<5}"
            for y in years:
                s = table[tk][y]
                pfs = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
                line += f"|{s['n']:>4}{pfs:>6}{s['wr']*100:>4.0f}%{s['total']:>+8.0f} "
            print(line)
        return table

    call_tab = side_table("call", CALL_WL, CALL_BLACK)
    put_tab = side_table("put", PUT_WL)

    # ---- Overfit verdicts ----
    def verdict(tab, wl, side_name):
        print(f"\n{'#'*92}")
        print(f"{side_name.upper()} VERDICTS")
        print(f"{'#'*92}")

        def allyears_ok(tk, min_n=20):
            ss = [tab[tk][y] for y in years]
            # require PF>1 in every year that has enough sample; ignore years with n<min_n
            usable = [s for s in ss if s["n"] >= min_n]
            if len(usable) < 2:  # need at least 2 years of real sample
                return None, usable
            return all(s["pf"] > 1.0 for s in usable), usable

        # CURVE-FIT: in whitelist but fails in 2024 or 2025 (PF<1 with adequate sample)
        print("\nCURVE-FIT RISK (whitelisted but PF<1 in an EARLIER year w/ n>=20):")
        any_cf = False
        for tk in wl:
            if tk not in tab:
                print(f"  {tk:<6} — NO DATA in thetadata universe")
                continue
            bad = []
            for y in ("2024", "2025"):
                s = tab[tk][y]
                if s["n"] >= 20 and s["pf"] < 1.0:
                    bad.append(f"{y}(PF{s['pf']:.2f},n{s['n']})")
            cur = tab[tk]["2026"]
            cur_str = f"2026 PF{cur['pf']:.2f} n{cur['n']}"
            if bad:
                any_cf = True
                print(f"  {tk:<6} FIT-RISK: weak {', '.join(bad)}  vs  {cur_str}")
        if not any_cf:
            print("  (none — all whitelisted names hold up in earlier years)")

        # whitelisted and solid all 3 years
        print("\nWHITELIST CONFIRMED (PF>1 every usable year, n>=20):")
        for tk in sorted(wl):
            if tk not in tab:
                continue
            ok, usable = allyears_ok(tk)
            if ok:
                ys = " ".join(f"{y}:PF{tab[tk][y]['pf']:.2f}" for y in years if tab[tk][y]["n"] >= 20)
                print(f"  {tk:<6} {ys}")

        # MISSED: not whitelisted, PF>1 in all usable years (n>=30 total to be safe)
        print("\nMISSED NAMES (NOT whitelisted but PF>1 in all usable years, total n>=30):")
        missed = []
        for tk in tickers:
            if tk in wl:
                continue
            ok, usable = allyears_ok(tk, min_n=20)
            tot_n = sum(tab[tk][y]["n"] for y in years)
            if ok and tot_n >= 30:
                missed.append(tk)
                ys = " ".join(f"{y}:PF{tab[tk][y]['pf']:.2f}(n{tab[tk][y]['n']})"
                              for y in years if tab[tk][y]["n"] >= 20)
                print(f"  {tk:<6} {ys}  totN={tot_n}")
        if not missed:
            print("  (none)")

        # RECOMMENDED 2.5yr-validated list: PF>1 in all usable years AND total n>=30
        rec = []
        for tk in tickers:
            ok, usable = allyears_ok(tk, min_n=20)
            tot_n = sum(tab[tk][y]["n"] for y in years)
            if ok and tot_n >= 30:
                rec.append(tk)
        print(f"\nRECOMMENDED {side_name} list (2.5yr-validated, PF>1 all usable yrs, n>=30):")
        print(f"  {sorted(rec)}")
        cur_set = wl
        print(f"  current {side_name} WL: {sorted(cur_set)}")
        print(f"  ADD (missed): {sorted(set(rec) - cur_set)}")
        print(f"  REVIEW/DROP (in WL, not validated): {sorted(cur_set - set(rec))}")
        return rec

    call_rec = verdict(call_tab, CALL_WL, "call")
    put_rec = verdict(put_tab, PUT_WL, "put")

    # ---- Blacklist check ----
    print(f"\n{'#'*92}\nCALL BLACKLIST CHECK (always bad, or only recently?)\n{'#'*92}")
    for tk in sorted(CALL_BLACK):
        if tk not in call_tab:
            print(f"  {tk:<6} no data")
            continue
        ys = " ".join(f"{y}:PF{call_tab[tk][y]['pf']:.2f}(n{call_tab[tk][y]['n']},"
                      f"tot{call_tab[tk][y]['total']:+.0f})" for y in years)
        print(f"  {tk:<6} {ys}")


if __name__ == "__main__":
    main()
