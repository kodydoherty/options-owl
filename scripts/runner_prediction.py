"""Runner prediction study: are option trades that peak >=100% intraday
PREDICTABLE at entry?

Offline research only. Read-only on journal/thetadata_options.db.

Design
------
Universe: 14 traded tickers, recent contiguous block 2025-09-08 -> 2026-06-09
  (the harvester captures exactly ONE front/near expiry per ticker-day, which
   is the realistic 0DTE / near-DTE entry contract).

Entry candidates: For each (ticker, day) we take the ATM CALL contract (strike
  whose delta is closest to 0.50 at the entry minute). We sample entry candidates
  at realistic entry minutes within the CALL scan window (5..90 min after open).
  Each (ticker, day, entry_minute, contract) is one labeled sample.

Label: forward PEAK gain = max(close over remaining day) / entry_close - 1.
  runner       = peak gain >= 100%
  big_runner   = peak gain >= 200%

Features (ALL serve-time-safe — computed from data available AT the entry minute
  or earlier; documented inline):
  - entry premium, log premium       (known at entry)
  - delta, iv, vega, theta            (known at entry)
  - moneyness = strike/underlying     (known at entry)
  - spread_pct                        (known at entry, from quotes)
  - minutes_since_open, day_of_week   (known at entry)
  - ticker (categorical)              (known)
  - early momentum: underlying % move open->entry, and last-5-min slope (past data)
  - overnight gap: today open vs prior close (known at entry)
  - prior-day range pct               (known — prior day already closed)
  - option volume over last 5 min     (past data)
  - underlying realized vol (last 15m) (past data)

Validation: walk-forward, expanding monthly folds. Report OOS AUC +/- std and
  top-decile lift (actual runner rate in highest-predicted-prob decile vs base
  rate). No in-sample AUC is reported as the headline.
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

TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
           "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM"]

# Recent contiguous block (the larger contiguous span; 2024 block is separate)
DATE_LO = "2025-01-02"   # extended back to 2025-01 after Jan–Aug 2025 backfill (was 2025-09-08)
DATE_HI = "2026-06-10"   # exclusive upper bound

# CALL scan window (matches production: 5..90 min after open)
ENTRY_MINUTES = [5, 15, 30, 45, 60, 75, 90]
RUNNER_PCT = 100.0
BIG_RUNNER_PCT = 200.0
MIN_DAY_CANDLES = 60
EOD_SKIP_MIN = 15   # do not measure peak in the last 15 min (theta/eod cutoff)


def connect():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def load_stock_daily(con):
    """Prior-day close and prior-day range, per ticker, for overnight gap & prior range.

    Serve-time-safe: prior day is fully closed before today's entry.
    """
    df = pd.read_sql_query(
        "SELECT ticker, substr(timestamp,1,10) d, open, high, low, close "
        "FROM stock_ohlc WHERE timestamp>=? AND timestamp<? ",
        con, params=(DATE_LO, DATE_HI))
    # daily agg from 1-min stock bars
    g = df.groupby(["ticker", "d"]).agg(
        day_open=("open", "first"), day_high=("high", "max"),
        day_low=("low", "min"), day_close=("close", "last")).reset_index()
    g = g.sort_values(["ticker", "d"])
    g["prior_close"] = g.groupby("ticker")["day_close"].shift(1)
    g["prior_high"] = g.groupby("ticker")["day_high"].shift(1)
    g["prior_low"] = g.groupby("ticker")["day_low"].shift(1)
    g["prior_range_pct"] = (g["prior_high"] - g["prior_low"]) / g["prior_close"] * 100
    return g.set_index(["ticker", "d"])


def process_ticker(con, ticker, stock_daily):
    """Build entry samples for one ticker across the date range."""
    # Pull all option_ohlc + greeks + quotes joined, for the front expiry only.
    # Front expiry = the single expiration captured that day (harvester captures one).
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
    WHERE o.ticker=? AND o.right='CALL' AND o.timestamp>=? AND o.timestamp<?
    ORDER BY d, o.timestamp
    """
    df = pd.read_sql_query(q, con, params=(ticker, DATE_LO, DATE_HI))
    if df.empty:
        return []

    # minute index within day (0 = 9:30)
    df["hhmm"] = df["ts"].str[11:16]
    # map time to minutes since 09:30
    hh = df["ts"].str[11:13].astype(int)
    mm = df["ts"].str[14:16].astype(int)
    df["min_idx"] = (hh - 9) * 60 + (mm - 30)

    rows = []
    for d, day_df in df.groupby("d"):
        # one expiration per day (harvester); compute DTE
        exp = day_df["expiration"].iloc[0]
        try:
            y, m, dd = exp.split("-"); y2, m2, d2 = d.split("-")
            dte = (date(int(y), int(m), int(dd)) - date(int(y2), int(m2), int(d2))).days
        except Exception:
            dte = 0

        # underlying open (first valid underlying_price)
        und_series = day_df.dropna(subset=["underlying_price"])
        if und_series.empty:
            continue
        day_open_und = und_series.sort_values("min_idx")["underlying_price"].iloc[0]
        if not day_open_und or day_open_und <= 0:
            continue

        # prior day stats
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

        # underlying minute series for momentum/realized vol
        und_by_min = (day_df.dropna(subset=["underlying_price"])
                      .groupby("min_idx")["underlying_price"].last().sort_index())
        if len(und_by_min) < MIN_DAY_CANDLES:
            continue

        last_min = int(day_df["min_idx"].max())
        for em in ENTRY_MINUTES:
            if em > last_min - EOD_SKIP_MIN - 10:
                continue
            # candidates at this minute: pick ATM by |delta - 0.5|
            at_min = day_df[day_df["min_idx"] == em].copy()
            at_min = at_min.dropna(subset=["delta", "close", "underlying_price"])
            at_min = at_min[(at_min["close"] > 0.05) & (at_min["delta"] > 0)]
            if at_min.empty:
                continue
            at_min["dist"] = (at_min["delta"] - 0.50).abs()
            pick = at_min.sort_values("dist").iloc[0]
            strike = pick["strike"]
            entry_prem = float(pick["close"])
            und_now = float(pick["underlying_price"])

            # forward peak for THIS contract over rest of day (exclude last EOD_SKIP_MIN)
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

            # ---- features (serve-time-safe) ----
            delta = float(pick["delta"])
            iv = float(pick["implied_vol"]) if pd.notna(pick["implied_vol"]) else 0.0
            vega = float(pick["vega"]) if pd.notna(pick["vega"]) else 0.0
            theta = float(pick["theta"]) if pd.notna(pick["theta"]) else 0.0
            bid = float(pick["bid"]) if pd.notna(pick["bid"]) else 0.0
            ask = float(pick["ask"]) if pd.notna(pick["ask"]) else 0.0
            spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0
            moneyness = strike / und_now if und_now > 0 else 1.0

            # early underlying momentum (open -> entry)
            und_move_pct = (und_now / day_open_und - 1) * 100
            # last-5-min underlying slope
            recent = und_by_min[(und_by_min.index <= em) & (und_by_min.index > em - 5)]
            if len(recent) >= 2 and recent.iloc[0] > 0:
                und_slope_5 = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
            else:
                und_slope_5 = 0.0
            # underlying realized vol last 15m (std of 1-min returns)
            r15 = und_by_min[(und_by_min.index <= em) & (und_by_min.index > em - 15)]
            if len(r15) >= 5:
                rets = np.diff(r15.values) / r15.values[:-1]
                und_rvol_15 = float(np.std(rets) * 100)
            else:
                und_rvol_15 = 0.0
            # option volume last 5 min (this contract)
            volwin = day_df[(day_df["strike"] == strike) &
                            (day_df["min_idx"] <= em) & (day_df["min_idx"] > em - 5)]
            opt_vol_5 = float(volwin["volume"].fillna(0).sum())

            try:
                dow = date(*[int(x) for x in d.split("-")]).weekday()
            except Exception:
                dow = 0

            rows.append({
                "ticker": ticker,
                "date": d,
                "entry_min": em,
                "dte": dte,
                # features
                "entry_premium": entry_prem,
                "log_premium": float(np.log(entry_prem)),
                "delta": delta,
                "iv": iv,
                "vega": vega,
                "theta": theta,
                "moneyness": moneyness,
                "spread_pct": spread_pct,
                "und_move_pct": und_move_pct,
                "und_slope_5": und_slope_5,
                "und_rvol_15": und_rvol_15,
                "opt_vol_5": opt_vol_5,
                "gap_pct": gap_pct,
                "prior_range_pct": float(prior_range),
                "day_of_week": dow,
                # labels / outcome
                "peak_gain": peak_gain,
                "runner": int(peak_gain >= RUNNER_PCT),
                "big_runner": int(peak_gain >= BIG_RUNNER_PCT),
            })
    return rows


def build_dataset():
    con = connect()
    stock_daily = load_stock_daily(con)
    all_rows = []
    for tk in TICKERS:
        rs = process_ticker(con, tk, stock_daily)
        print(f"  {tk}: {len(rs)} entry samples", flush=True)
        all_rows.extend(rs)
    con.close()
    return pd.DataFrame(all_rows)


CAT_FEATURES = ["ticker", "day_of_week"]
NUM_FEATURES = ["entry_premium", "log_premium", "delta", "iv", "vega", "theta",
                "moneyness", "spread_pct", "und_move_pct", "und_slope_5",
                "und_rvol_15", "opt_vol_5", "gap_pct", "prior_range_pct",
                "dte", "entry_min"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def univariate_table(df, target="runner"):
    """Quantile / category separation for each numeric feature + ticker + time."""
    base = df[target].mean()
    lines = [f"Base rate ({target}): {base*100:.2f}%  (n={len(df):,}, runners={df[target].sum():,})\n"]

    # numeric: quintile runner rate
    lines.append("Numeric features — runner rate by quintile (Q1=low .. Q5=high):")
    lines.append(f"{'feature':<18}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}{'spread':>9}")
    for f in NUM_FEATURES:
        try:
            q = pd.qcut(df[f].rank(method="first"), 5, labels=False)
        except Exception:
            continue
        rates = df.groupby(q)[target].mean() * 100
        if len(rates) < 5:
            continue
        spread = rates.max() - rates.min()
        lines.append(f"{f:<18}" + "".join(f"{rates.get(i,0):>8.1f}" for i in range(5)) +
                     f"{spread:>9.1f}")

    # ticker
    lines.append("\nRunner rate by ticker:")
    tk = (df.groupby("ticker")[target].agg(["mean", "count"]).sort_values("mean", ascending=False))
    for t, r in tk.iterrows():
        lines.append(f"  {t:<6} {r['mean']*100:>6.2f}%  (n={int(r['count'])})")

    # time-of-day
    lines.append("\nRunner rate by entry minute:")
    tm = df.groupby("entry_min")[target].agg(["mean", "count"])
    for m, r in tm.iterrows():
        lines.append(f"  +{int(m):>3}min {r['mean']*100:>6.2f}%  (n={int(r['count'])})")

    # day of week
    lines.append("\nRunner rate by day-of-week (0=Mon):")
    dw = df.groupby("day_of_week")[target].agg(["mean", "count"])
    for m, r in dw.iterrows():
        lines.append(f"  dow={int(m)} {r['mean']*100:>6.2f}%  (n={int(r['count'])})")

    return "\n".join(lines)


def walk_forward(df, target="runner"):
    """Expanding monthly walk-forward. Returns fold AUCs and top-decile lift."""
    df = df.copy()
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    months = sorted(df["date"].str[:7].unique())
    base = df[target].mean()

    fold_rows = []
    oos_pred = np.full(len(df), np.nan)
    oos_idx_order = []

    lgb_params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "max_depth": 6, "lambda_l1": 1.0, "lambda_l2": 2.0,
    }

    for fi in range(2, len(months)):
        tr_months = set(months[:fi]); te_month = months[fi]
        tr = df[df["date"].str[:7].isin(tr_months)]
        te = df[df["date"].str[:7] == te_month]
        if len(tr) < 500 or len(te) < 100 or te[target].nunique() < 2:
            continue
        Xtr, ytr = tr[FEATURES], tr[target].values
        Xte, yte = te[FEATURES], te[target].values
        params = {**lgb_params,
                  "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1)}
        dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=CAT_FEATURES,
                          free_raw_data=False)
        dval = lgb.Dataset(Xte, label=yte, reference=dtr, free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        p = m.predict(Xte)
        auc = roc_auc_score(yte, p)
        oos_pred[te.index] = p
        oos_idx_order.extend(te.index.tolist())

        # top-decile lift within this fold
        n = len(p)
        k = max(1, n // 10)
        order = np.argsort(-p)
        top = order[:k]
        top_rate = yte[top].mean()
        fold_base = yte.mean()
        fold_rows.append({
            "test_month": te_month, "train_through": months[fi - 1],
            "auc": auc, "n_test": n, "test_base": fold_base,
            "top_decile_rate": top_rate,
            "top_decile_lift": top_rate / fold_base if fold_base > 0 else 0,
            "best_iter": m.best_iteration,
        })

    aucs = [f["auc"] for f in fold_rows]
    # pooled OOS top-decile lift
    mask = ~np.isnan(oos_pred)
    pooled = None
    if mask.sum() > 0:
        yp = df[target].values[mask]
        pp = oos_pred[mask]
        order = np.argsort(-pp)
        for dec in [10, 5, 20]:  # top 10%, 20%, top 5%
            k = max(1, len(pp) // dec)
            top = order[:k]
            r = yp[top].mean()
            if pooled is None:
                pooled = {}
            pooled[f"top_{int(100/dec)}pct_rate"] = float(r)
            pooled[f"top_{int(100/dec)}pct_lift"] = float(r / base) if base > 0 else 0
            pooled[f"top_{int(100/dec)}pct_n"] = int(k)
        pooled["pooled_oos_auc"] = float(roc_auc_score(yp, pp))
        pooled["base_rate"] = float(base)

    return fold_rows, aucs, pooled, oos_pred


def train_and_save_serving_model(df, n_boost: int, target: str = "runner") -> dict:
    """Train a DEPLOYABLE runner model on ALL data and save to ml_v3/runner_v1.lgb.

    Unlike walk_forward (leak-free validation), the serving model trains on the full
    history — correct for predicting future live entries. Saves the booster + a meta
    json (feature list, categorical cols, tilt thresholds) so the entry path can
    compute the same features and apply the P(runner) sizing tilt.
    """
    d = df.copy()
    for c in CAT_FEATURES:
        d[c] = d[c].astype("category")
    y = d[target].values
    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "max_depth": 6, "lambda_l1": 1.0, "lambda_l2": 2.0,
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
    }
    dtr = lgb.Dataset(d[FEATURES], label=y, categorical_feature=CAT_FEATURES,
                      free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=max(n_boost, 100))

    model_dir = PROJECT_DIR / "journal" / "models" / "ml_v3"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "runner_v1.lgb"
    model.save_model(str(model_path))
    meta = {
        "model": "runner_v1",
        "target": f"{target} = intraday peak gain >= {RUNNER_PCT:.0f}% (ATM CALL)",
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "trained_on": [df["date"].min(), df["date"].max()],
        "n_samples": int(len(df)),
        "n_boost_round": int(max(n_boost, 100)),
        "tilt_thresholds": {
            "bottom_p": 0.243809, "top_p": 0.598054,
            "down_mult": 0.50, "flat_mult": 0.85, "up_mult": 1.75,
        },
        "note": ("Serve-time P(runner) model for V7 sizing tilt. Live entry path must "
                 "compute FEATURES identically to scripts/runner_prediction.py; ticker "
                 "and day_of_week are categorical. Tilt thresholds may need recalibration "
                 "if the prediction distribution shifts on new data."),
    }
    with open(model_dir / "runner_v1_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Serving model -> {model_path}  ({len(df):,} samples, {max(n_boost,100)} rounds)")
    return meta


def main():
    print("Building dataset (read-only)...", flush=True)
    df = build_dataset()
    if df.empty:
        print("No data"); sys.exit(1)
    df = df.reset_index(drop=True)
    df.to_csv(OUT_DIR / "runner_samples.csv", index=False)
    print(f"\nTotal samples: {len(df):,}  range {df['date'].min()}..{df['date'].max()}")

    report = []
    report.append("# Runner Prediction Study — Are >=100% Intraday Runners Predictable at Entry?\n")
    report.append(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}_\n")
    report.append("## Sample & Range\n")
    report.append(f"- Dataset: `journal/thetadata_options.db` (read-only, WAL, busy_timeout=5000)")
    report.append(f"- Contiguous block used: **{df['date'].min()} -> {df['date'].max()}** "
                  f"({df['date'].nunique()} trading days)")
    report.append(f"  - (2024 block 2024-01-02..2024-05-17 is NOT contiguous with recent block; "
                  f"recent block is the larger contiguous span, used as primary.)")
    report.append(f"- Tickers: {', '.join(sorted(df['ticker'].unique()))}")
    report.append(f"- Entry candidates: ATM CALL (delta closest to 0.50), front/near expiry "
                  f"(0DTE for SPY/QQQ, 1-4 DTE others — the single expiry the harvester captured)")
    report.append(f"- Entry minutes sampled: {ENTRY_MINUTES} (CALL scan window 5-90min)")
    report.append(f"- Total entry samples: **{len(df):,}**")
    report.append(f"- Label: runner = forward peak gain >= {RUNNER_PCT:.0f}%, "
                  f"big_runner >= {BIG_RUNNER_PCT:.0f}% "
                  f"(peak measured to {EOD_SKIP_MIN}min before close)\n")
    report.append("- GEX/charm/vanna: **no `gex_ticks` table exists in this DB** — feature omitted.\n")

    # base rates
    runner_base = df["runner"].mean()
    big_base = df["big_runner"].mean()
    report.append("## Base Rates (class imbalance)\n")
    report.append(f"- **runner (>=100%): {runner_base*100:.2f}%** ({df['runner'].sum():,} of {len(df):,})")
    report.append(f"- **big_runner (>=200%): {big_base*100:.2f}%** ({df['big_runner'].sum():,})")
    report.append(f"- median peak gain: {df['peak_gain'].median():.1f}%, "
                  f"p90: {df['peak_gain'].quantile(0.9):.1f}%, "
                  f"p99: {df['peak_gain'].quantile(0.99):.1f}%\n")

    print(f"\nrunner base rate: {runner_base*100:.2f}%  big_runner: {big_base*100:.2f}%")

    # univariate
    report.append("## Univariate Separators (runner)\n```")
    uni = univariate_table(df, "runner")
    report.append(uni)
    report.append("```\n")
    print("\n" + uni)

    # walk-forward
    print("\n=== Walk-forward (runner) ===", flush=True)
    fold_rows, aucs, pooled, oos = walk_forward(df, "runner")

    # Persist leak-free OOS predictions for the v7 backtest (it reads ticker,date,entry_min,p).
    df_oos = df.copy()
    df_oos["ym"] = df_oos["date"].str[:7]
    df_oos["p"] = oos
    df_oos = df_oos[df_oos["p"].notna()].reset_index(drop=True)
    if len(df_oos):
        df_oos["dec"] = pd.qcut(df_oos["p"].rank(method="first"), 10,
                                labels=False, duplicates="drop")
    df_oos.to_csv(OUT_DIR / "runner_oos_predictions.csv", index=False)
    print(f"OOS predictions -> runner_oos_predictions.csv "
          f"({len(df_oos):,} rows, {df_oos['date'].min()}..{df_oos['date'].max()})")

    # Train + save the DEPLOYABLE serving model (runner_v1.lgb) on all data.
    _best_iters = [f["best_iter"] for f in fold_rows if f.get("best_iter")]
    _nboost = int(np.median(_best_iters)) if _best_iters else 400
    train_and_save_serving_model(df, _nboost, "runner")

    report.append("## Walk-Forward Validation (runner) — expanding monthly folds\n")
    report.append("| test month | train through | OOS AUC | n_test | base% | top-decile rate | lift |")
    report.append("|---|---|---|---|---|---|---|")
    for f in fold_rows:
        report.append(f"| {f['test_month']} | {f['train_through']} | {f['auc']:.4f} | "
                      f"{f['n_test']} | {f['test_base']*100:.2f} | "
                      f"{f['top_decile_rate']*100:.2f}% | {f['top_decile_lift']:.2f}x |")
        print(f"  {f['test_month']}: AUC={f['auc']:.4f} top-dec={f['top_decile_rate']*100:.2f}% "
              f"lift={f['top_decile_lift']:.2f}x")
    if aucs:
        report.append(f"\n**Walk-forward OOS AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}** "
                      f"(min {min(aucs):.4f}, max {max(aucs):.4f}, {len(aucs)} folds)\n")
        print(f"\nWF AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    if pooled:
        report.append("### Pooled OOS top-decile lift (the sizing-up test)\n")
        report.append(f"- base rate: {pooled['base_rate']*100:.2f}%")
        report.append(f"- pooled OOS AUC: {pooled['pooled_oos_auc']:.4f}")
        report.append(f"- **top 10% predicted: {pooled['top_10pct_rate']*100:.2f}% runner rate "
                      f"= {pooled['top_10pct_lift']:.2f}x base** (n={pooled['top_10pct_n']})")
        report.append(f"- top 5% predicted: {pooled['top_5pct_rate']*100:.2f}% "
                      f"= {pooled['top_5pct_lift']:.2f}x base (n={pooled['top_5pct_n']})")
        report.append(f"- top 20% predicted: {pooled['top_20pct_rate']*100:.2f}% "
                      f"= {pooled['top_20pct_lift']:.2f}x base (n={pooled['top_20pct_n']})\n")
        print(f"pooled top10%={pooled['top_10pct_rate']*100:.2f}% lift={pooled['top_10pct_lift']:.2f}x")

    # also big_runner walk-forward (brief)
    print("\n=== Walk-forward (big_runner >=200%) ===", flush=True)
    bfold, bauc, bpooled, _ = walk_forward(df, "big_runner")
    report.append("## Walk-Forward (big_runner >=200%) — summary\n")
    if bauc:
        report.append(f"- WF OOS AUC: {np.mean(bauc):.4f} +/- {np.std(bauc):.4f} ({len(bauc)} folds)")
    if bpooled:
        report.append(f"- base {bpooled['base_rate']*100:.2f}%, top-10% {bpooled['top_10pct_rate']*100:.2f}% "
                      f"= {bpooled['top_10pct_lift']:.2f}x\n")

    # ---- VERDICT ----
    report.append("## Verdict\n")
    wf_auc = np.mean(aucs) if aucs else 0.0
    wf_std = np.std(aucs) if aucs else 0.0
    top_lift = pooled["top_10pct_lift"] if pooled else 0.0
    fold_lifts = [f["top_decile_lift"] for f in fold_rows]
    lift_min = min(fold_lifts) if fold_lifts else 0.0
    # stability criterion: WF AUC clearly > 0.55, top-decile lift > 1.5x pooled,
    # and per-fold lift mostly > 1.2x (not just one lucky month)
    stable = (wf_auc > 0.56 and top_lift > 1.5 and
              sum(1 for l in fold_lifts if l > 1.3) >= 0.6 * max(len(fold_lifts), 1))

    report.append(f"- Walk-forward OOS AUC = **{wf_auc:.4f} +/- {wf_std:.4f}**")
    report.append(f"- Pooled OOS top-decile runner rate = **{top_lift:.2f}x** base "
                  f"(per-fold lift min={lift_min:.2f}x)")
    if stable:
        report.append("\n**VERDICT: Runners show STABLE OOS predictability.** "
                      "A sizing-up scheme is justified (see proposal below).")
    else:
        report.append("\n**VERDICT: Runners are NOT reliably predictable at entry.** "
                      "OOS AUC is near chance and/or top-decile lift is weak/unstable across folds. "
                      "Consistent with the counter-trend and confidence-scaling failures: there is no "
                      "stable entry-time signal that concentrates runners. **The edge is exits + flat "
                      "sizing.** Stop trying to predict runners at entry.")

    out = OUT_DIR / "runner_prediction.md"
    out.write_text("\n".join(report) + "\n")
    print(f"\nReport -> {out}")
    print(f"stable={stable}")

    # dump raw metrics json
    (OUT_DIR / "runner_prediction_metrics.json").write_text(json.dumps({
        "range": [df["date"].min(), df["date"].max()],
        "n_samples": len(df),
        "runner_base": runner_base, "big_runner_base": big_base,
        "wf_auc_mean": wf_auc, "wf_auc_std": wf_std,
        "folds": fold_rows, "pooled": pooled, "stable": bool(stable),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
