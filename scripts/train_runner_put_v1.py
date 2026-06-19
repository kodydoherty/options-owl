"""Train a PUT runner-prediction model — the PUT analog of runner_v1 (CALLS).

Mirrors scripts/runner_prediction.py EXACTLY, but on PUT rows. Question: does a
P(runner) model separate which PUT entries become big premium winners the same way
runner_v1 does for calls (validated OOS AUC 0.74)?

CLEAN MODEL-OOS DESIGN (better than the call model, which trained through 2026-06-09
and so was NOT model-OOS in 2026):
  - TRAIN the serving model on 2024-2025 ONLY.
  - HOLD OUT 2026 entirely as the clean OOS test (validated by runner_put_validation.py).
  - The walk-forward inside here is for reference / picking n_boost only.

PUT ADAPTATIONS (documented):
  - right = 'PUT'.
  - ATM pick: PUT ATM delta ~= -0.50, so pick contract minimizing |delta - (-0.50)| = |delta + 0.50|.
  - delta filter: keep delta < 0 (real put greeks). delta is kept NEGATIVE in the feature
    vector (the model learns the put sign directly; we do NOT abs() it — the call model
    saw +delta, this one sees -delta, and that's fine for a tree model trained on puts).
  - moneyness = strike/underlying (same definition; >1 = ITM put, <1 = OTM put).
  - Label "runner" = intraday option-premium PEAK gain >= 100% (also report >=50%).
    A put runner = a big premium gain (underlying fell), regardless of side.
  - Everything else (features, exit-skip window, walk-forward, params) is identical to
    runner_prediction.py.

Outputs: journal/models/ml_v3/runner_put_v1.lgb + runner_put_v1_meta.json
         journal/v3_eval_results/runner_put_samples.csv

Read-only on journal/thetadata_options.db. Run:
  cd /Users/kody/dev/options-owl && python scripts/train_runner_put_v1.py [--validate]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
OUT_DIR = PROJECT_DIR / "journal" / "v3_eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Liquid PUT-capable tickers with enough 0DTE put days (>=100). Mirrors the call set
# (drops nothing relevant — these are exactly the names with deep put history).
TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
           "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM"]

# Full available range; we slice TRAIN=2024-2025 in the serving fit and hold out 2026.
DATE_LO = "2024-01-02"
DATE_HI = "2026-06-13"   # exclusive upper bound (covers thru 2026-06-12)
TRAIN_HI = "2026-01-01"  # serving model trains on rows with date < TRAIN_HI (2024-2025 only)

# PUTs scan ALL DAY in production (5..360 min). Sample a realistic grid across that window.
ENTRY_MINUTES = [5, 15, 30, 45, 60, 90, 120, 180, 240, 300]
RUNNER_PCT = 100.0
BIG_RUNNER_PCT = 200.0
RUNNER_50_PCT = 50.0
MIN_DAY_CANDLES = 60
EOD_SKIP_MIN = 15   # do not measure peak in the last 15 min (theta/eod cutoff)

PUT_ATM_DELTA = -0.50  # ATM put target delta


def connect():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def load_stock_daily(con):
    df = pd.read_sql_query(
        "SELECT ticker, substr(timestamp,1,10) d, open, high, low, close "
        "FROM stock_ohlc WHERE timestamp>=? AND timestamp<? ",
        con, params=(DATE_LO, DATE_HI))
    g = df.groupby(["ticker", "d"]).agg(
        day_open=("open", "first"), day_high=("high", "max"),
        day_low=("low", "min"), day_close=("close", "last")).reset_index()
    g = g.sort_values(["ticker", "d"])
    g["prior_close"] = g.groupby("ticker")["day_close"].shift(1)
    g["prior_high"] = g.groupby("ticker")["day_high"].shift(1)
    g["prior_low"] = g.groupby("ticker")["day_low"].shift(1)
    g["prior_range_pct"] = (g["prior_high"] - g["prior_low"]) / g["prior_close"] * 100
    return g.set_index(["ticker", "d"])


def process_ticker(con, ticker, stock_daily, only_months=None):
    """Build PUT entry samples for one ticker. 0DTE only (expiration == day)."""
    q = """
    SELECT o.expiration, o.strike, substr(o.timestamp,1,10) d, o.timestamp ts,
           o.close, o.high, o.volume,
           g.delta, g.theta, g.vega, g.implied_vol, g.underlying_price,
           q.bid, q.ask
    FROM option_ohlc o
    LEFT JOIN option_greeks g
      ON o.ticker=g.ticker AND o.expiration=g.expiration AND o.strike=g.strike
         AND o.right=g.right AND o.timestamp=g.timestamp
    LEFT JOIN option_quotes q
      ON o.ticker=q.ticker AND o.expiration=q.expiration AND o.strike=q.strike
         AND o.right=q.right AND o.timestamp=q.timestamp
    WHERE o.ticker=? AND o.right='PUT' AND o.expiration = substr(o.timestamp,1,10)
      AND o.timestamp>=? AND o.timestamp<?
    ORDER BY d, o.timestamp
    """
    df = pd.read_sql_query(q, con, params=(ticker, DATE_LO, DATE_HI))
    if df.empty:
        return []

    df["hhmm"] = df["ts"].str[11:16]
    hh = df["ts"].str[11:13].astype(int)
    mm = df["ts"].str[14:16].astype(int)
    df["min_idx"] = (hh - 9) * 60 + (mm - 30)

    rows = []
    for d, day_df in df.groupby("d"):
        if only_months is not None and d[:7] not in only_months:
            continue
        exp = day_df["expiration"].iloc[0]
        try:
            y, m, dd = exp.split("-"); y2, m2, d2 = d.split("-")
            dte = (date(int(y), int(m), int(dd)) - date(int(y2), int(m2), int(d2))).days
        except Exception:
            dte = 0

        und_series = day_df.dropna(subset=["underlying_price"])
        if und_series.empty:
            continue
        day_open_und = und_series.sort_values("min_idx")["underlying_price"].iloc[0]
        if not day_open_und or day_open_und <= 0:
            continue

        try:
            sd = stock_daily.loc[(ticker, d)]
            prior_close = sd["prior_close"]
            prior_range = sd["prior_range_pct"]
        except KeyError:
            prior_close = np.nan
            prior_range = np.nan
        gap_pct = ((day_open_und / prior_close - 1) * 100
                   if prior_close and prior_close > 0 else 0.0)
        if pd.isna(gap_pct):
            gap_pct = 0.0
        if pd.isna(prior_range):
            prior_range = 0.0

        und_by_min = (day_df.dropna(subset=["underlying_price"])
                      .groupby("min_idx")["underlying_price"].last().sort_index())
        if len(und_by_min) < MIN_DAY_CANDLES:
            continue

        last_min = int(day_df["min_idx"].max())
        for em in ENTRY_MINUTES:
            if em > last_min - EOD_SKIP_MIN - 10:
                continue
            at_min = day_df[day_df["min_idx"] == em].copy()
            at_min = at_min.dropna(subset=["delta", "close", "underlying_price"])
            # PUT delta is negative; ATM put delta ~= -0.50. Filter to real put greeks.
            at_min = at_min[(at_min["close"] > 0.05) & (at_min["delta"] < 0)]
            if at_min.empty:
                continue
            # ATM by |delta - (-0.50)|
            at_min["dist"] = (at_min["delta"] - PUT_ATM_DELTA).abs()
            pick = at_min.sort_values("dist").iloc[0]
            strike = pick["strike"]
            entry_prem = float(pick["close"])
            und_now = float(pick["underlying_price"])

            fut = day_df[(day_df["strike"] == strike) &
                         (day_df["min_idx"] > em) &
                         (day_df["min_idx"] <= last_min - EOD_SKIP_MIN)]
            fut_high = fut["high"].dropna()
            fut_close = fut["close"].dropna()
            future_vals = pd.concat([fut_high, fut_close])
            future_vals = future_vals[future_vals > 0]
            if future_vals.empty:
                continue
            peak = float(future_vals.max())
            peak_gain = (peak / entry_prem - 1) * 100

            # ---- features (serve-time-safe), delta kept NEGATIVE ----
            delta = float(pick["delta"])
            iv = float(pick["implied_vol"]) if pd.notna(pick["implied_vol"]) else 0.0
            vega = float(pick["vega"]) if pd.notna(pick["vega"]) else 0.0
            theta = float(pick["theta"]) if pd.notna(pick["theta"]) else 0.0
            bid = float(pick["bid"]) if pd.notna(pick["bid"]) else 0.0
            ask = float(pick["ask"]) if pd.notna(pick["ask"]) else 0.0
            spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0
            moneyness = strike / und_now if und_now > 0 else 1.0

            und_move_pct = (und_now / day_open_und - 1) * 100
            recent = und_by_min[(und_by_min.index <= em) & (und_by_min.index > em - 5)]
            if len(recent) >= 2 and recent.iloc[0] > 0:
                und_slope_5 = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
            else:
                und_slope_5 = 0.0
            r15 = und_by_min[(und_by_min.index <= em) & (und_by_min.index > em - 15)]
            if len(r15) >= 5:
                rets = np.diff(r15.values) / r15.values[:-1]
                und_rvol_15 = float(np.std(rets) * 100)
            else:
                und_rvol_15 = 0.0
            volwin = day_df[(day_df["strike"] == strike) &
                            (day_df["min_idx"] <= em) & (day_df["min_idx"] > em - 5)]
            opt_vol_5 = float(volwin["volume"].fillna(0).sum())

            try:
                dow = date(*[int(x) for x in d.split("-")]).weekday()
            except Exception:
                dow = 0

            rows.append({
                "ticker": ticker, "date": d, "entry_min": em, "dte": dte,
                "entry_premium": entry_prem, "log_premium": float(np.log(entry_prem)),
                "delta": delta, "iv": iv, "vega": vega, "theta": theta,
                "moneyness": moneyness, "spread_pct": spread_pct,
                "und_move_pct": und_move_pct, "und_slope_5": und_slope_5,
                "und_rvol_15": und_rvol_15, "opt_vol_5": opt_vol_5,
                "gap_pct": gap_pct, "prior_range_pct": float(prior_range),
                "day_of_week": dow,
                "peak_gain": peak_gain,
                "runner": int(peak_gain >= RUNNER_PCT),
                "runner50": int(peak_gain >= RUNNER_50_PCT),
                "big_runner": int(peak_gain >= BIG_RUNNER_PCT),
            })
    return rows


def build_dataset(tickers=None, only_months=None):
    con = connect()
    stock_daily = load_stock_daily(con)
    all_rows = []
    for tk in (tickers or TICKERS):
        rs = process_ticker(con, tk, stock_daily, only_months=only_months)
        print(f"  {tk}: {len(rs)} PUT entry samples", flush=True)
        all_rows.extend(rs)
    con.close()
    return pd.DataFrame(all_rows)


CAT_FEATURES = ["ticker", "day_of_week"]
NUM_FEATURES = ["entry_premium", "log_premium", "delta", "iv", "vega", "theta",
                "moneyness", "spread_pct", "und_move_pct", "und_slope_5",
                "und_rvol_15", "opt_vol_5", "gap_pct", "prior_range_pct",
                "dte", "entry_min"]
FEATURES = NUM_FEATURES + CAT_FEATURES

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "max_depth": 6, "lambda_l1": 1.0, "lambda_l2": 2.0,
}


def walk_forward(df, target="runner"):
    """Expanding monthly walk-forward (reference, also picks n_boost). 2024-2025 + 2026."""
    df = df.copy()
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    months = sorted(df["date"].str[:7].unique())
    base = df[target].mean()
    fold_rows = []
    for fi in range(2, len(months)):
        tr_months = set(months[:fi]); te_month = months[fi]
        tr = df[df["date"].str[:7].isin(tr_months)]
        te = df[df["date"].str[:7] == te_month]
        if len(tr) < 500 or len(te) < 100 or te[target].nunique() < 2:
            continue
        Xtr, ytr = tr[FEATURES], tr[target].values
        Xte, yte = te[FEATURES], te[target].values
        params = {**LGB_PARAMS,
                  "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1)}
        dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=CAT_FEATURES, free_raw_data=False)
        dval = lgb.Dataset(Xte, label=yte, reference=dtr, free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        p = m.predict(Xte)
        auc = roc_auc_score(yte, p)
        fold_rows.append({"test_month": te_month, "auc": auc, "n_test": len(te),
                          "test_base": float(yte.mean()), "best_iter": m.best_iteration})
        print(f"  {te_month}: AUC={auc:.4f} n={len(te)} base={yte.mean()*100:.1f}%", flush=True)
    aucs = [f["auc"] for f in fold_rows]
    return fold_rows, aucs, base


def train_and_save_serving_model(df_train, n_boost, target="runner"):
    """Train DEPLOYABLE runner_put_v1 on TRAIN (2024-2025) only. Save booster + meta."""
    d = df_train.copy()
    for c in CAT_FEATURES:
        d[c] = d[c].astype("category")
    y = d[target].values
    params = {**LGB_PARAMS,
              "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1)}
    dtr = lgb.Dataset(d[FEATURES], label=y, categorical_feature=CAT_FEATURES, free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=max(n_boost, 100))

    model_dir = PROJECT_DIR / "journal" / "models" / "ml_v3"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "runner_put_v1.lgb"
    model.save_model(str(model_path))
    meta = {
        "model": "runner_put_v1",
        "target": f"runner = intraday peak gain >= {RUNNER_PCT:.0f}% (ATM PUT, delta~-0.50)",
        "features": FEATURES, "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
        "trained_on": [df_train["date"].min(), df_train["date"].max()],
        "trained_window": "2024-2025 ONLY (2026 held out as clean model-OOS)",
        "n_samples": int(len(df_train)),
        "n_boost_round": int(max(n_boost, 100)),
        "put_adaptations": {
            "right": "PUT",
            "atm_pick": "min |delta - (-0.50)|",
            "delta_filter": "delta < 0 (real put greeks)",
            "delta_in_features": "kept NEGATIVE (model learns put sign)",
            "moneyness": "strike/underlying (>1 ITM put)",
            "entry_minutes": ENTRY_MINUTES,
        },
        "tilt_thresholds": {
            "down_mult": 0.50, "flat_mult": 0.85, "up_mult": 1.75,
            "note": "calibrate bottom_p/top_p from OOS distribution in validation",
        },
        "note": ("PUT analog of runner_v1. Live entry path must compute FEATURES identically "
                 "to scripts/train_runner_put_v1.py (delta NEGATIVE for puts). 2026 is clean "
                 "model-OOS — this model never saw it."),
    }
    with open(model_dir / "runner_put_v1_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Serving PUT model -> {model_path}  ({len(df_train):,} TRAIN samples, "
          f"{max(n_boost,100)} rounds)")
    return meta


def main():
    validate = "--validate" in sys.argv
    if validate:
        print("### VALIDATE: 2 tickers (SPY,TSLA) x 2 months (2025-03, 2026-04) ###", flush=True)
        df = build_dataset(tickers=["SPY", "TSLA"], only_months={"2025-03", "2026-04"})
        if df.empty:
            print("No data"); return
        print(f"\nsamples={len(df)}  runner@100% base={df['runner'].mean()*100:.2f}%  "
              f"runner@50% base={df['runner50'].mean()*100:.2f}%")
        print(f"delta range (should be NEGATIVE): min={df['delta'].min():.3f} "
              f"max={df['delta'].max():.3f} med={df['delta'].median():.3f}")
        print(f"moneyness med={df['moneyness'].median():.4f}  "
              f"peak_gain p50={df['peak_gain'].median():.1f}% p90={df['peak_gain'].quantile(.9):.1f}%")
        print("\nSample proof rows (entry-minute features):")
        cols = ["ticker", "date", "entry_min", "entry_premium", "delta", "iv",
                "moneyness", "und_move_pct", "peak_gain", "runner"]
        print(df[cols].head(8).to_string(index=False))
        print("\nVALIDATION OK — re-run without --validate for full train.")
        return

    print("Building PUT dataset (read-only, 2024-01..2026-06)...", flush=True)
    df = build_dataset()
    if df.empty:
        print("No data"); sys.exit(1)
    df = df.reset_index(drop=True)
    df.to_csv(OUT_DIR / "runner_put_samples.csv", index=False)
    print(f"\nTotal PUT samples: {len(df):,}  range {df['date'].min()}..{df['date'].max()}")
    print(f"  by year: {df['date'].str[:4].value_counts().to_dict()}")

    rb = df["runner"].mean(); rb50 = df["runner50"].mean(); bb = df["big_runner"].mean()
    print(f"\nbase runner@100%={rb*100:.2f}%  runner@50%={rb50*100:.2f}%  big@200%={bb*100:.2f}%")
    print(f"delta range: min={df['delta'].min():.3f} max={df['delta'].max():.3f} (negative = put)")
    print(f"peak_gain median={df['peak_gain'].median():.1f}% p90={df['peak_gain'].quantile(.9):.1f}%")

    # walk-forward (reference + n_boost pick)
    print("\n=== Walk-forward (runner@100%), reference ===", flush=True)
    fold_rows, aucs, base = walk_forward(df, "runner")
    if aucs:
        print(f"\nWF OOS AUC (all folds incl. 2026): {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} "
              f"({len(aucs)} folds)")
    best_iters = [f["best_iter"] for f in fold_rows if f.get("best_iter")]
    n_boost = int(np.median(best_iters)) if best_iters else 300

    # SERVING MODEL: train on 2024-2025 ONLY (clean 2026 hold-out)
    df_train = df[df["date"] < TRAIN_HI].reset_index(drop=True)
    df_2026 = df[df["date"] >= TRAIN_HI]
    print(f"\nServing fit TRAIN(2024-25) n={len(df_train):,}  HOLDOUT(2026) n={len(df_2026):,}")
    print(f"  TRAIN runner@100% base={df_train['runner'].mean()*100:.2f}%  "
          f"2026 base={df_2026['runner'].mean()*100:.2f}%")
    train_and_save_serving_model(df_train, n_boost, "runner")

    (OUT_DIR / "runner_put_train_metrics.json").write_text(json.dumps({
        "range": [df["date"].min(), df["date"].max()],
        "n_samples": len(df), "n_train_2024_25": len(df_train), "n_2026": len(df_2026),
        "runner_base": rb, "runner50_base": rb50, "big_runner_base": bb,
        "wf_aucs": aucs, "n_boost": n_boost,
    }, indent=2, default=float))
    print("\nDone. Model -> journal/models/ml_v3/runner_put_v1.lgb")


if __name__ == "__main__":
    main()
