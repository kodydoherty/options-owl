"""Phase 2: mid-day SUSTAINED-vs-REBOUND classifier + confirm-then-commit PUT P&L proxy.

The trap (phase 1): rebound days (103) outnumber sustained crashes (81) and are
indistinguishable by drawdown mid-day. Can STRUCTURE features tell them apart in time
to fire PUTs? And does a "fire only when the model says sustained" rule actually capture
the down move while dodging the rebound trap?

Offline research. Read-only on journal/thetadata_options.db. Labels from down_days.csv.

Sample = (day, entry_minute) on a meaningfully-red decision point. Features are all
computable at that minute from the day's past bars (serve-time safe). Label = 1 if the
day ends SUSTAINED. Walk-forward (expanding monthly) keeps it leak-free.

P&L proxy: a PUT entered at minute T captures -(SPY close-vs-T return). Net edge =
mean captured move when the model fires vs firing on every red minute (the naive rule).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
LABELS = ROOT / "journal" / "v3_eval_results" / "down_days.csv"
OUT = ROOT / "journal" / "v3_eval_results"

ENTRY_MINS = list(range(60, 361, 5))   # 10:30..15:30 ET (incl. late-afternoon accel window)
RED_GATE = -0.25                         # only decide when dd-so-far <= -0.25%
FEATURES = [
    "min_idx", "dd_sofar", "ret_sofar", "vwap_dev", "frac_below_vwap",
    "mins_since_low", "ret_5m", "ret_15m", "ret_30m", "accel_15m",
    "rvol_30m", "new_lows_30m", "range_pos", "qqq_dd_sofar",
    # "dead-cat-bounce vs reversal" features the pros use (added phase 2b):
    "fib_retrace",        # % of open->low drop retraced now (38-61% = the short zone)
    "bounce_vol_ratio",   # bounce-leg volume / drop-leg volume (low = weak DCB = sustained)
    "vwap_rejects_30m",   # times in last 30m price tested VWAP from below and got rejected
]


def _load_1m(ticker: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp, high, low, close, volume FROM stock_ohlc "
        "WHERE ticker=? ORDER BY timestamp",
        con, params=(ticker,),
    )
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["min_idx"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30  # minutes after 09:30
    df = df[(df["min_idx"] >= 0) & (df["min_idx"] <= 390)]
    return df


def build_samples() -> pd.DataFrame:
    labels = pd.read_csv(LABELS)[["date", "cls"]]
    lab = dict(zip(labels["date"], labels["cls"]))
    spy = _load_1m("SPY")
    qqq = _load_1m("QQQ")
    qqq_dd = {}  # (date,min_idx) -> qqq drawdown so far
    for d, g in qqq.groupby("date"):
        g = g.sort_values("min_idx")
        op = float(g.iloc[0]["close"])
        runlow = np.minimum.accumulate(g["low"].values)
        for mi, rl in zip(g["min_idx"].values, runlow):
            qqq_dd[(d, int(mi))] = (rl - op) / op * 100 if op > 0 else 0.0

    rows = []
    for d, g in spy.groupby("date"):
        if d not in lab:
            continue
        g = g.sort_values("min_idx").reset_index(drop=True)
        if len(g) < 120:
            continue
        op = float(g.iloc[0]["close"])
        close = g["close"].values.astype(float)
        low = g["low"].values.astype(float)
        high = g["high"].values.astype(float)
        vol = g["volume"].values.astype(float)
        mi = g["min_idx"].values.astype(int)
        eod = close[-1]
        rets = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-9)
        cum_pv = np.cumsum(close * vol)
        cum_v = np.maximum(np.cumsum(vol), 1e-9)
        vwap = cum_pv / cum_v
        run_low = np.minimum.accumulate(low)
        run_high = np.maximum.accumulate(high)
        below = (close < vwap).astype(float)
        frac_below = np.cumsum(below) / (np.arange(len(close)) + 1)
        # minutes since the running low last decreased (new low recency)
        mins_since_low = np.zeros(len(close))
        last = 0
        for i in range(len(close)):
            if i > 0 and run_low[i] < run_low[i - 1] - 1e-9:
                last = i
            mins_since_low[i] = i - last
        y = 1 if lab[d] == "SUSTAINED" else 0
        pos = {m: i for i, m in enumerate(mi)}
        for em in ENTRY_MINS:
            if em not in pos:
                continue
            i = pos[em]
            dd = (run_low[i] - op) / op * 100
            if dd > RED_GATE:   # not red enough to consider PUTs
                continue

            def _ret(n):
                j = max(0, i - n)
                return (close[i] - close[j]) / max(close[j], 1e-9) * 100

            ret15 = _ret(15)
            j30 = max(0, i - 30)
            prev15 = (close[j30 + 15] - close[j30]) / max(close[j30], 1e-9) * 100 if i >= 30 else ret15
            new_lows_30 = int(np.sum(np.diff(run_low[max(0, i - 30):i + 1]) < -1e-9))
            rng = run_high[i] - run_low[i]
            # dead-cat-bounce structure
            low_idx = i - int(mins_since_low[i])
            fib_retrace = (close[i] - run_low[i]) / (op - run_low[i]) * 100 if op > run_low[i] else 0.0
            bounce_vol = vol[low_idx:i + 1].mean() if i > low_idx else vol[i]
            drop_vol = vol[max(0, low_idx - 15):low_idx + 1].mean()
            bounce_vol_ratio = bounce_vol / max(drop_vol, 1.0)
            w = slice(max(0, i - 30), i + 1)
            vwap_rejects = int(np.sum((high[w] >= vwap[w]) & (close[w] < vwap[w])))
            rows.append({
                "date": d, "ym": d[:7], "entry_min": em, "y": y,
                "spy_fwd_ret": (eod - close[i]) / max(close[i], 1e-9) * 100,
                "min_idx": em, "dd_sofar": dd, "ret_sofar": (close[i] - op) / op * 100,
                "vwap_dev": (close[i] - vwap[i]) / max(vwap[i], 1e-9) * 100,
                "frac_below_vwap": frac_below[i],
                "mins_since_low": mins_since_low[i],
                "ret_5m": _ret(5), "ret_15m": ret15, "ret_30m": _ret(30),
                "accel_15m": ret15 - prev15,
                "rvol_30m": float(np.std(rets[max(0, i - 30):i + 1]) * 100),
                "new_lows_30m": new_lows_30,
                "range_pos": (close[i] - run_low[i]) / rng if rng > 0 else 0.5,
                "qqq_dd_sofar": qqq_dd.get((d, em), 0.0),
                "fib_retrace": fib_retrace,
                "bounce_vol_ratio": bounce_vol_ratio,
                "vwap_rejects_30m": vwap_rejects,
            })
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame):
    months = sorted(df["ym"].unique())
    oos = np.full(len(df), np.nan)
    aucs = []
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "learning_rate": 0.03, "num_leaves": 24, "min_child_samples": 80,
              "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
              "max_depth": 5, "lambda_l1": 1.0, "lambda_l2": 2.0}
    for fi in range(2, len(months)):
        tr = df[df["ym"].isin(months[:fi])]
        te = df[df["ym"] == months[fi]]
        if len(tr) < 300 or len(te) < 40 or te["y"].nunique() < 2:
            continue
        params["scale_pos_weight"] = (tr["y"] == 0).sum() / max((tr["y"] == 1).sum(), 1)
        m = lgb.train(params, lgb.Dataset(tr[FEATURES], label=tr["y"].values),
                      num_boost_round=300)
        p = m.predict(te[FEATURES])
        oos[te.index] = p
        if te["y"].nunique() > 1:
            aucs.append(roc_auc_score(te["y"].values, p))
    return oos, aucs


def main():
    print("Building samples (SPY+QQQ 1m, structure features)...", flush=True)
    df = build_samples().reset_index(drop=True)
    print(f"Samples: {len(df):,}  (red decision points)  | sustained-rate {df['y'].mean()*100:.1f}%")
    oos, aucs = walk_forward(df)
    df["p"] = oos
    val = df[df["p"].notna()].copy()
    print(f"\nWalk-forward OOS AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} ({len(aucs)} folds)")

    # PUT P&L proxy: captured move = -spy_fwd_ret (PUT gains when SPY falls further)
    val["put_capture"] = -val["spy_fwd_ret"]
    naive = val["put_capture"].mean()
    print("\n=== Confirm-then-commit PUT edge (mean SPY move captured per fired entry) ===")
    print(f"  Naive (fire on EVERY red minute): {naive:+.3f}%  (n={len(val)})")
    print(f"  {'threshold':<11}{'fires':>7}{'fire%':>7}{'avg_capture':>13}{'win%':>7}")
    for thr in [0.5, 0.6, 0.7, 0.8]:
        f = val[val["p"] >= thr]
        if len(f) == 0:
            continue
        cap = f["put_capture"].mean()
        win = (f["put_capture"] > 0).mean() * 100
        print(f"  p>={thr:<8}{len(f):>7}{len(f)/len(val)*100:>6.1f}%{cap:>12.3f}%{win:>6.1f}%")

    # Edge by entry-time bucket (phase-1 said sustained crashes accelerate into the close)
    print("\n=== PUT capture by entry time (p>=0.6 fires) ===")
    fired = val[val["p"] >= 0.6].copy()
    fired["bucket"] = pd.cut(fired["min_idx"], [0, 150, 240, 300, 361],
                             labels=["10:30-12:00", "12:00-13:30", "13:30-14:30", "14:30-15:30"])
    print(f"  {'window':<14}{'fires':>7}{'avg_capture':>13}{'win%':>7}")
    for b, g in fired.groupby("bucket", observed=True):
        if len(g):
            print(f"  {str(b):<14}{len(g):>7}{g['put_capture'].mean():>12.3f}%{(g['put_capture']>0).mean()*100:>6.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    val.to_csv(OUT / "down_day_classifier_oos.csv", index=False)
    print(f"\nSaved OOS predictions -> {OUT / 'down_day_classifier_oos.csv'}")


if __name__ == "__main__":
    main()
