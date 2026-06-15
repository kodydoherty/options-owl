"""Find + classify down market days from SPY 1m data, and test whether a
sustained crash is distinguishable from a dip-and-rebound MID-DAY (the trap that
cost owlet-kody -$824 on 2026-06-11 when Trump's statement V-shaped the market).

Offline research. Read-only on journal/thetadata_options.db.

Per RTH day:
  open    = 09:30 price        close = 16:00 price       low = RTH min
  max_dd  = (low-open)/open    oc_ret = (close-open)/open
  recovery= (close-low)/(open-low)   # ~0 closed at low (sustained), ~1 full rebound

Classes (down days only, max_dd <= -0.5%):
  SUSTAINED  oc_ret <= -0.5% and recovery < 0.40   (PUTs win)
  REBOUND    recovery > 0.60                        (PUTs die — the trap)
  PARTIAL    in between

Mid-day separability: at each checkpoint (11:00/12:00/13:00 ET) measure drawdown-so-far
and whether the day ultimately SUSTAINED, to see if we can react in time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB = str(Path(__file__).resolve().parent.parent / "journal" / "thetadata_options.db")
DD_THRESH = -0.005       # -0.5% intraday drawdown = a "down day"
SUSTAINED_OC = -0.005    # close-vs-open <= -0.5%
SUSTAINED_REC = 0.40     # recovered < 40% of the drop
REBOUND_REC = 0.60       # recovered > 60% of the drop
CHECKPOINTS = ["11:00", "12:00", "13:00"]


def load_spy() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close FROM stock_ohlc WHERE ticker='SPY' ORDER BY timestamp",
        con,
    )
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["hhmm"] = ts.dt.strftime("%H:%M")
    # RTH only
    df = df[(df["hhmm"] >= "09:30") & (df["hhmm"] <= "16:00")]
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("hhmm")
        if len(g) < 60:
            continue
        op = float(g.iloc[0]["open"]) or float(g.iloc[0]["close"])
        cl = float(g.iloc[-1]["close"])
        lo = float(g["low"].min())
        lo_time = g.loc[g["low"].idxmin(), "hhmm"]
        if op <= 0:
            continue
        max_dd = (lo - op) / op
        oc_ret = (cl - op) / op
        recovery = (cl - lo) / (op - lo) if op > lo else 1.0
        cls = "up_or_flat"
        if max_dd <= DD_THRESH:
            if oc_ret <= SUSTAINED_OC and recovery < SUSTAINED_REC:
                cls = "SUSTAINED"
            elif recovery > REBOUND_REC:
                cls = "REBOUND"
            else:
                cls = "PARTIAL"
        rec = {"date": d, "open": op, "close": cl, "low": lo, "low_time": lo_time,
               "max_dd_pct": max_dd * 100, "oc_ret_pct": oc_ret * 100,
               "recovery": recovery, "cls": cls}
        # mid-day drawdown at each checkpoint
        for cp in CHECKPOINTS:
            sofar = g[g["hhmm"] <= cp]
            rec[f"dd_{cp}"] = ((float(sofar["low"].min()) - op) / op * 100) if len(sofar) else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading SPY 1m (read-only)...", flush=True)
    df = load_spy()
    res = classify(df)
    n = len(res)
    print(f"\nTrading days analyzed: {n}  ({res['date'].min()}..{res['date'].max()})\n")

    counts = res["cls"].value_counts()
    print("=== Day classes ===")
    for k in ["up_or_flat", "REBOUND", "PARTIAL", "SUSTAINED"]:
        c = int(counts.get(k, 0))
        print(f"  {k:<11} {c:>4}  ({c/n*100:.1f}%)")

    down = res[res["cls"].isin(["SUSTAINED", "REBOUND", "PARTIAL"])]
    sustained = res[res["cls"] == "SUSTAINED"]
    rebound = res[res["cls"] == "REBOUND"]
    print(f"\nDown days (any): {len(down)}  | SUSTAINED {len(sustained)}  REBOUND {len(rebound)}")

    print("\n=== Mid-day separability: avg drawdown-so-far by final class ===")
    print(f"{'checkpoint':<12}{'SUSTAINED':>12}{'REBOUND':>12}{'gap':>8}")
    for cp in CHECKPOINTS:
        s = sustained[f"dd_{cp}"].mean()
        r = rebound[f"dd_{cp}"].mean()
        print(f"{cp+' ET':<12}{s:>11.2f}%{r:>11.2f}%{(s-r):>7.2f}")

    print("\n=== Worst 12 SUSTAINED crash days (PUT goldmine) ===")
    print(sustained.sort_values("oc_ret_pct").head(12)[
        ["date", "max_dd_pct", "oc_ret_pct", "recovery", "low_time"]
    ].to_string(index=False))

    print("\n=== 2026-06-11 (the Trump rebound) ===")
    row = res[res["date"] == "2026-06-11"]
    print(row[["cls", "max_dd_pct", "oc_ret_pct", "recovery", "low_time",
               "dd_11:00", "dd_12:00", "dd_13:00"]].to_string(index=False) if len(row) else "  (not in data)")

    out = Path(DB).parent / "v3_eval_results" / "down_days.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved per-day table -> {out}")


if __name__ == "__main__":
    main()
