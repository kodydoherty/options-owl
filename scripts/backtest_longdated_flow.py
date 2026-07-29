"""Backtest the 15-20% "follow-the-whale long-dated sleeve" (2026-07-18, Kody's idea).

Reads journal/longdated_flow_options.db (signals + DAILY option/underlying bars from
download_longdated_flow.py). For each long-dated (>30 DTE) whale ask-side sweep, simulate buying
the whale's ACTUAL contract at the alert and holding on DAILY bars under several exit rules, then
report the per-trade edge (mean/median return %, win %, profit factor, avg hold days) per rule and
per side (call/put). Also a flat-$ sleeve P&L for scale. Read-only, self-contained.

    python scripts/backtest_longdated_flow.py
"""
from __future__ import annotations

import sqlite3
import statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "longdated_flow_options.db"
SLEEVE_PER_TRADE = 2000.0   # flat $ per position for a scale estimate (sleeve spread across trades)


def _load():
    c = sqlite3.connect(str(DB))
    sigs = c.execute(
        "SELECT ticker,right,strike,expiry,entry_date,entry_premium,total_premium,dte FROM signals"
    ).fetchall()
    trades = []
    for tk, right, strike, expiry, d0, entry_prem, tot, dte in sigs:
        bars = c.execute(
            """SELECT substr(timestamp,1,10) d, open, high, low, close FROM option_ohlc
               WHERE ticker=? AND right=? AND strike=? AND expiration=? AND close>0
               ORDER BY timestamp""",
            (tk, right, strike, expiry),
        ).fetchall()
        if len(bars) < 2:
            continue
        # entry = close on/after the alert date (realistic same-day fill); fallback alert price
        entry_bar = next((b for b in bars if b[0] >= d0[:10]), bars[0])
        entry_px = entry_bar[1] or entry_bar[4] or entry_prem  # open of entry day
        if not entry_px or entry_px <= 0:
            entry_px = entry_prem
        if not entry_px or entry_px <= 0:
            continue
        path = [b for b in bars if b[0] >= entry_bar[0]]
        if len(path) < 2:
            continue
        trades.append({"tk": tk, "right": right, "dte": dte, "entry": float(entry_px), "path": path})
    c.close()
    return trades


def _sim(tr, rule):
    """Return (ret_pct, hold_days). path rows = (date, open, high, low, close)."""
    e = tr["entry"]
    path = tr["path"]
    peak = e
    for i, (_, o, hi, lo, cl) in enumerate(path[1:], start=1):
        hi = hi or cl; lo = lo or cl; cl = cl or e
        peak = max(peak, hi)
        up = (hi / e - 1) * 100
        dn = (lo / e - 1) * 100
        dd_from_peak = (cl / peak - 1) * 100
        if rule == "target_100_stop_50":
            if dn <= -50:
                return -50.0, i
            if up >= 100:
                return 100.0, i
        elif rule == "target_50_stop_50":
            if dn <= -50:
                return -50.0, i
            if up >= 50:
                return 50.0, i
        elif rule == "trail_40":
            if dd_from_peak <= -40:
                return (cl / e - 1) * 100, i
        elif rule == "time_21d":
            if i >= 21:
                return (cl / e - 1) * 100, i
        elif rule == "stop_50_only":
            if dn <= -50:
                return -50.0, i
    # ran out of bars (≈ expiry or +90d cap) → exit at last close
    last = path[-1][4] or e
    return (last / e - 1) * 100, len(path) - 1


def _agg(rets):
    if not rets:
        return "no trades"
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gp = sum(wins); gl = abs(sum(losses))
    pf = gp / gl if gl > 0 else float("inf")
    sleeve_pnl = sum(SLEEVE_PER_TRADE * r / 100 for r in rets)
    return (f"n={len(rets):<3} mean={S.mean(rets):+6.1f}% median={S.median(rets):+6.1f}% "
            f"WR={100*len(wins)/len(rets):4.0f}% PF={pf:4.2f} "
            f"sleeve@${SLEEVE_PER_TRADE:.0f}/tr=${sleeve_pnl:+,.0f}")


def main():
    trades = _load()
    print(f"Loaded {len(trades)} long-dated whale trades with usable daily paths\n")
    if not trades:
        print("No trades (download may still be running).")
        return
    calls = [t for t in trades if t["right"] == "CALL"]
    puts = [t for t in trades if t["right"] == "PUT"]
    print(f"  {len(calls)} calls, {len(puts)} puts | avg DTE {S.mean([t['dte'] for t in trades]):.0f}\n")

    rules = ["stop_50_only", "target_100_stop_50", "target_50_stop_50", "trail_40", "time_21d"]
    for rule in rules:
        allr = [_sim(t, rule) for t in trades]
        holds = [h for _, h in allr]
        rets = [r for r, _ in allr]
        cr = [_sim(t, rule)[0] for t in calls]
        pr = [_sim(t, rule)[0] for t in puts]
        print(f"── {rule}  (avg hold {S.mean(holds):.0f}d) ──")
        print(f"   ALL   {_agg(rets)}")
        print(f"   CALLS {_agg(cr)}")
        print(f"   PUTS  {_agg(pr)}")
        print()


if __name__ == "__main__":
    main()
