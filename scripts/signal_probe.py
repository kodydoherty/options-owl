#!/usr/bin/env python3
"""Signal-agnostic executable-edge probe (2026-08-06) — answers Kody's question: for the volatile new
tickers, are we using the WRONG SIGNAL (dip-buy vs momentum), or is there just no CAPTURABLE edge?

For each ticker, compares three entry rules on 0DTE ATM calls, honest fill (buy the run-up ask-proxy a
bar after the trigger), over all available days:
  DIP       — first candle within 2% of the killzone (90m) low   (our current pattern)
  MOMENTUM  — first candle that breaks the opening-range (30m) high (breakout/trend)
  BASELINE  — a fixed mid-morning entry (control)

Two numbers per rule:
  best-fwd %  = the peak the option reached after entry (did the MOVE exist — upper bound, perfect exit)
  take30 %    = realistic exit: sell at first +30%, else EOD close (can we CAPTURE it)

Reading: MOMENTUM take30 > 0 where DIP < 0 → wrong signal (there's money, we're fishing wrong).
         best-fwd big but take30 ≤ 0 for all → the move exists but reverses too fast to capture (exit).
         everything ≤ 0 including best-fwd → no edge, they don't pay even with a perfect exit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

TICKERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SOXL", "TQQQ", "IBIT"]
TAKE = 30.0   # realistic take-profit %


def atm_0dte(opts, d, mb, spot):
    oday = opts[opts["date"] == d]
    if oday.empty:
        return None
    dte0 = oday["dte"].min()
    same = oday[oday["dte"] == dte0]
    strikes = same["strike"].unique()
    if len(strikes) == 0:
        return None
    strike = strikes[int(np.argmin(np.abs(strikes - spot)))]
    ch = same[(same["strike"] == strike) & (same["mi"] >= mb)].sort_values("mi")
    pp = ch["close"].values.astype(float)
    return pp if len(pp) >= 3 and pp[0] > 0 else None


def rets(pp):
    """(best_fwd_pct, take30_pct) from an honest entry = run-up (close a bar later, never below pp[0])."""
    entry = max(float(pp[0]), float(pp[1]))
    if entry <= 0:
        return None
    fwd = pp[2:]
    fwd = fwd[~np.isnan(fwd) & (fwd > 0)]
    if len(fwd) == 0:
        return None
    best = (float(np.nanmax(fwd)) / entry - 1) * 100
    take = None
    for px in fwd:
        if px >= entry * (1 + TAKE / 100):
            take = TAKE
            break
    if take is None:
        take = (float(fwd[-1]) / entry - 1) * 100   # EOD
    return best, take


def probe(tk):
    stock, opts = D._stock(tk), D._opts(tk, "CALL")
    R = {"DIP": [], "MOMENTUM": [], "BASELINE": []}
    days = 0
    for d, s in stock.items():
        mins = sorted(m for m in s if 0 <= m <= 90 and s[m] > 0)
        if len(mins) < 20:
            continue
        days += 1
        or_high = max(s[m] for m in mins if m <= 30)
        kz_low = min(s[m] for m in mins if m <= 90)
        ent = {}
        for m in mins:
            if m < 5:
                continue
            if "DIP" not in ent and s[m] <= kz_low * 1.02:
                ent["DIP"] = m
            if "MOMENTUM" not in ent and s[m] > or_high:
                ent["MOMENTUM"] = m
        ent["BASELINE"] = 30 if 30 in s else mins[len(mins) // 3]
        for rule, mb in ent.items():
            path = atm_0dte(opts, d, mb, s[mb])
            if path is None:
                continue
            rr = rets(path)
            if rr is not None:
                R[rule].append(rr)
    return R, days


print(f"{'TICKER':7s}{'RULE':10s}{'n':>5s}{'best-fwd%':>11s}{'take30%avg':>12s}{'win%':>7s}{'sum%':>9s}")
print("=" * 62)
for tk in TICKERS:
    try:
        R, days = probe(tk)
    except Exception as exc:
        print(f"{tk}: ERROR {exc}")
        continue
    print(f"-- {tk} ({days} days) --")
    for rule in ("DIP", "MOMENTUM", "BASELINE"):
        rr = R[rule]
        if not rr:
            print(f"{'':7s}{rule:10s}{'0':>5s}  (no entries)")
            continue
        best = np.array([x[0] for x in rr]); take = np.array([x[1] for x in rr])
        print(f"{'':7s}{rule:10s}{len(rr):>5d}{np.mean(best):>+10.1f}%{np.mean(take):>+11.1f}%"
              f"{np.mean(take > 0) * 100:>6.0f}%{np.sum(take):>+8.0f}%")
