"""TWEET-EVENT PERSISTENCE + TRADABILITY backtest.

Question: an Elon/Trump tweet-trading strategy's edge is millisecond SPEED, which we LOSE (droplet
polling, seconds-to-minutes entry lag). The ONLY way we win is if the underlying move PERSISTS for
minutes (catchable at t0+1..t0+5) instead of a sub-second spike-and-fade. Test that empirically.

For each vetted event (journal/tweet_events.csv):
  1. PERSISTENCE: underlying % move from t0 at t0+{1,5,15,30,60} min. SPIKE-FADE vs PERSISTENT DRIFT.
  2. TRADABILITY: buy nearest-DTE ATM option in the move direction at t0+1 AND t0+5 (our realistic lag),
     run the REAL V7 exit (reuse entry_timing_oracle.sim() + LOCK cfg + ExitFSM; PUT cfg for puts).
     Report per-event realized return + aggregate PF/WR at each lag.

Anchoring: events are in ET; the thetadata DB stores tz-aware ISO with the ET offset, so mi = (hour-9)*60
+ minute - 30 maps the event minute directly. We pick the contract with expiration == t0's date if present
(true 0DTE); else the NEAREST expiration available for that ticker on that date (nearest-DTE, matching prod
flow strike resolution). ATM = strike nearest the underlying close at t0.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/tweet_event_persistence.py
Optional: VALIDATE=1 (print full join detail for the first 2 events: t0, underlying path, chosen contract, exit).
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import entry_timing_oracle as O  # noqa: E402  (reuse sim(), LOCK, pf())
import uw_ticker_discovery as D  # noqa: E402  (DB, ET, EXIT_HAIRCUT, config helpers)

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
EVENTS_CSV = ROOT / "journal" / "tweet_events.csv"
LAGS = [1, 5]                       # entry lag in minutes (our realistic reaction time)
HORIZONS = [1, 5, 15, 30, 60]      # persistence horizons


def _mi(hh, mm):
    """ET clock -> minute-index from 9:30 open."""
    return (hh - 9) * 60 + mm - 30


def load_stock_day(tk, date):
    """{mi: close} for the regular-hours session on `date`."""
    con = sqlite3.connect(DB)
    s = pd.read_sql_query(
        "SELECT timestamp, close FROM stock_ohlc WHERE ticker=? AND substr(timestamp,1,10)=?",
        con, params=(tk, date))
    con.close()
    if s.empty:
        return {}
    ts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
    mi = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return {int(m): float(c) for m, c in zip(mi, s["close"]) if m >= 0}


def load_option_day(tk, date, right):
    """All option bars for `ticker/date/right`, with expiration + dte. Returns df[mi, strike, expiration, close, dte]."""
    con = sqlite3.connect(DB)
    p = pd.read_sql_query(
        "SELECT timestamp, expiration, strike, close FROM option_ohlc "
        "WHERE ticker=? AND right=? AND substr(timestamp,1,10)=? ORDER BY timestamp",
        con, params=(tk, right, date))
    con.close()
    if p.empty:
        return p
    ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(ET)
    p["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    p["dte"] = (pd.to_datetime(p["expiration"]) - pd.to_datetime(date)).dt.days
    return p[p["mi"] >= 0]


def pick_contract(opt, spot, t0_mi):
    """Pick nearest-DTE, ATM contract that has a live bar at/after t0_mi.

    Prefer the smallest non-negative dte (0DTE if it exists, else nearest expiry). Among that expiry,
    ATM = strike nearest spot. Returns (expiration, strike, dte) or None.
    """
    live = opt[opt["mi"] >= t0_mi]
    if live.empty:
        return None
    # candidate expirations present with a live bar, smallest dte first
    exps = (live[live["dte"] >= 0].groupby("expiration")["dte"].first().sort_values())
    for exp in exps.index:
        sub = live[live["expiration"] == exp]
        strikes = sub["strike"].unique()
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        # ensure that strike actually has a bar at/after t0
        if not sub[sub["strike"] == atm].empty:
            return exp, float(atm), int(exps[exp])
    return None


def run_event(ev, validate=False):
    date = ev["date"]
    hh, mm = map(int, ev["time_et"].split(":"))
    t0 = _mi(hh, mm)
    direction = ev["direction"].strip().lower()
    side = "call" if direction == "up" else "put"
    right = "CALL" if side == "call" else "PUT"
    # pick the FIRST listed ticker that has data (index events list SPY|QQQ; we trade the most-liquid present)
    tickers = ev["tickers"].split("|")
    out = {"label": f"{date} {ev['time_et']} {ev['author']} {direction.upper()}",
           "side": side, "tickers": ev["tickers"], "conf": ev["ts_confidence"]}

    chosen_tk = None
    stock = {}
    for tk in tickers:
        s = load_stock_day(tk, date)
        if t0 in s:
            chosen_tk, stock = tk, s
            break
    if chosen_tk is None:
        out["error"] = f"no stock bar at t0 for {tickers} on {date}"
        return out
    out["ticker"] = chosen_tk
    spot0 = stock[t0]
    out["spot0"] = spot0

    # ---- PERSISTENCE: underlying move from t0 ----
    moves = {}
    for hz in HORIZONS:
        m = t0 + hz
        if m in stock:
            moves[hz] = (stock[m] - spot0) / spot0 * 100.0
        else:
            moves[hz] = None
    out["moves"] = moves

    # ---- TRADABILITY: buy ATM nearest-DTE option at each lag, run real V7 exit ----
    opt = load_option_day(chosen_tk, date, right)
    cfg = D.apply_v7_wide_trail_exits(
        D.get_ticker_config(chosen_tk, use_per_ticker=True, option_type=side), is_put=(side == "put"))
    out["lag_rets"] = {}
    out["contract"] = None
    if opt.empty:
        out["opt_error"] = f"no {right} option bars for {chosen_tk} on {date}"
        return out

    for lag in LAGS:
        em = t0 + lag
        pick = pick_contract(opt, spot0, em)
        if pick is None:
            out["lag_rets"][lag] = None
            continue
        exp, strike, dte = pick
        out["contract"] = {"exp": exp, "strike": strike, "dte": dte}
        # option premium path from entry minute to EOD for the chosen contract
        ch = opt[(opt["expiration"] == exp) & (opt["strike"] == strike) & (opt["mi"] >= em)].sort_values("mi")
        mp = ch["mi"].to_numpy(int)
        pp = ch["close"].to_numpy(float)
        if len(pp) < 2 or mp[0] != em or np.isnan(pp[0]) or pp[0] <= 0:
            # if no bar exactly at em, accept the first bar >= em as the entry
            if len(pp) >= 2 and not (np.isnan(pp[0]) or pp[0] <= 0):
                em = int(mp[0])
            else:
                out["lag_rets"][lag] = None
                continue
        up = [stock.get(int(m), spot0) for m in mp]
        ets = datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
        # reuse the oracle's exit sim EXACTLY (same LOCK cfg + ExitFSM + haircut)
        ret = O.sim(list(pp), list(mp), up, cfg, side, ets)
        out["lag_rets"][lag] = {"ret": ret, "entry_prem": float(pp[0]), "entry_mi": int(mp[0]),
                                "exp": exp, "strike": strike, "dte": dte}

    if validate:
        print(f"\n[VALIDATE] {out['label']}  tk={chosen_tk} side={side} t0_mi={t0} spot0={spot0:.2f}")
        print("  underlying path from t0:")
        for hz in HORIZONS:
            mv = moves[hz]
            print(f"    t0+{hz:<2}min: {('%+.2f%%' % mv) if mv is not None else 'n/a'}")
        if out["contract"]:
            c = out["contract"]
            print(f"  contract: exp={c['exp']} strike={c['strike']} dte={c['dte']} right={right}")
        for lag in LAGS:
            lr = out["lag_rets"].get(lag)
            if lr:
                print(f"  ENTRY t0+{lag}: prem=${lr['entry_prem']:.2f} @mi{lr['entry_mi']} -> exit ret {lr['ret']:+.1f}%")
            else:
                print(f"  ENTRY t0+{lag}: NO TRADE (no contract/bar)")
    return out


def classify(moves):
    """SPIKE-FADE vs PERSISTENT DRIFT, from the underlying move shape."""
    m1, m5, m15, m30 = moves.get(1), moves.get(5), moves.get(15), moves.get(30)
    early = next((x for x in (m1, m5) if x is not None), None)
    late = next((x for x in (m30, m15) if x is not None), None)
    if early is None or late is None or abs(early) < 1e-9:
        return "n/a"
    # persistent if the move at +15/+30 is at least as big AND same sign as the early move
    if np.sign(late) == np.sign(early) and abs(late) >= abs(early) * 0.8:
        return "DRIFT"
    if np.sign(late) != np.sign(early) or abs(late) < abs(early) * 0.5:
        return "SPIKE-FADE"
    return "MIXED"


def main():
    validate = os.environ.get("VALIDATE") == "1"
    with open(EVENTS_CSV) as f:
        events = list(csv.DictReader(f))
    print(f"Loaded {len(events)} vetted events from {EVENTS_CSV.name}\n")

    results = []
    for i, ev in enumerate(events):
        r = run_event(ev, validate=(validate and i < 2))
        results.append(r)

    if validate:
        print("\nVALIDATION done — re-run without VALIDATE=1 for the full report.\n")

    # ---- (1) PERSISTENCE table ----
    print("=" * 108)
    print("(1) PERSISTENCE — underlying % move from t0 (positive = up). Classify spike-fade vs drift.")
    print("=" * 108)
    hdr = f"{'event':<40}{'tk':<6}{'dir':<5}" + "".join(f"{'+'+str(h):>9}" for h in HORIZONS) + f"{'class':>13}"
    print(hdr)
    classes = []
    for r in results:
        if r.get("error") or "moves" not in r:
            print(f"{r['label']:<40}{'--':<6}{'--':<5}  {r.get('error','no data')}")
            continue
        cls = classify(r["moves"])
        classes.append(cls)
        row = f"{r['label']:<40}{r['ticker']:<6}{r['side']:<5}"
        for h in HORIZONS:
            mv = r["moves"][h]
            row += f"{('%+.2f' % mv) if mv is not None else '   n/a':>9}"
        row += f"{cls:>13}"
        print(row)
    print("\n  classification tally: " +
          ", ".join(f"{c}={classes.count(c)}" for c in ("DRIFT", "SPIKE-FADE", "MIXED", "n/a") if classes.count(c)))

    # ---- (2) TRADABILITY table ----
    print("\n" + "=" * 108)
    print("(2) TRADABILITY — buy ATM nearest-DTE option in move direction at t0+lag, REAL V7 exit to EOD.")
    print("=" * 108)
    for lag in LAGS:
        rets = []
        print(f"\n--- ENTRY LAG t0+{lag}min ---")
        print(f"  {'event':<40}{'tk':<6}{'contract':<22}{'entry$':>8}{'realized%':>11}")
        for r in results:
            lr = r.get("lag_rets", {}).get(lag)
            if not lr:
                print(f"  {r.get('label','?'):<40}{r.get('ticker','--'):<6}{'NO TRADE':<22}")
                continue
            con = f"{lr['strike']:g}{r['side'][0].upper()} dte{lr['dte']}"
            rets.append(lr["ret"])
            print(f"  {r['label']:<40}{r['ticker']:<6}{con:<22}{lr['entry_prem']:>8.2f}{lr['ret']:>+11.1f}")
        if rets:
            arr = np.array(rets, float)
            wr = (arr > 0).mean() * 100
            print(f"    >> n={len(rets)}  PF={O.pf(arr):.2f}  WR={wr:.0f}%  "
                  f"avgRet={arr.mean():+.1f}%  sumRet={arr.sum():+.1f}%  "
                  f"(equal-$ basket: PF measures gross profit/loss ratio)")
        else:
            print("    >> no tradable events at this lag")

    # ---- (3) lag decay ----
    print("\n" + "-" * 108)
    print("(3) LAG DECAY — same events, t0+1 vs t0+5:")
    paired = [(r["lag_rets"].get(1), r["lag_rets"].get(5)) for r in results
              if r.get("lag_rets", {}).get(1) and r.get("lag_rets", {}).get(5)]
    if paired:
        r1 = np.array([a["ret"] for a, _ in paired])
        r5 = np.array([b["ret"] for _, b in paired])
        print(f"  matched events (tradable at BOTH lags): n={len(paired)}")
        print(f"    t0+1: PF={O.pf(r1):.2f} WR={(r1>0).mean()*100:.0f}% avg={r1.mean():+.1f}% sum={r1.sum():+.1f}%")
        print(f"    t0+5: PF={O.pf(r5):.2f} WR={(r5>0).mean()*100:.0f}% avg={r5.mean():+.1f}% sum={r5.sum():+.1f}%")
        print(f"    decay (t0+5 minus t0+1 avg ret): {r5.mean()-r1.mean():+.1f} pts/trade")
    else:
        print("  not enough events tradable at both lags to measure decay")


if __name__ == "__main__":
    main()
