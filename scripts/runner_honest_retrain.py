"""Runner_v1 HONEST-LABEL retrain + OOS ranking A/B (2026-07-29).

Fixes the fantasy-fill contamination in the runner_v1 training label and RETESTS
whether the honest-labeled model ranks runners better OUT-OF-SAMPLE.

THE BUG (scripts/runner_prediction.py ~L182/196):
    entry_prem = pick["close"]                      # fantasy signal-instant close
    peak_gain  = (peak/entry_prem - 1)*100
    runner     = int(peak_gain >= 100)
The denominator is the price you'd fantasy-fill at (the close AT the signal minute),
not the price a live order actually PAYS. Live kody fills at the ASK, one order-
placement bar later, after the momentum run-up (measured ~8% ML / ~20% flow). So
~1-in-6 "runner" positives are fantasy (memory: runner-v1-honest-verdict-2026-07-29).

THE FIX (env HONEST_FILL_LABELS=1, default OFF so byte-identical otherwise):
    honest_denom = max(close@min, ask@min, ask@min+1) * (1 + HONEST_FILL_SPREAD)
    peak_gain    = (peak/honest_denom - 1)*100
matching backtest_gold_standard._executable_entry_ask (ask, delay=1 bar, floor at
signal ask, never a dip discount) + the flow-harness 2% spread cross. Only the LABEL
denominator changes; the FEATURE entry_premium stays = close (serve reads snap.mid ≈
close, so keeping the feature at close preserves train/serve consistency).

RUNNER THRESHOLD: user spec 2026-07-29 — target runner = peak_gain>=50% (was 100%).
We sweep {50,75,100} to show how OOS ranking quality changes; 50% is the deliverable.

Read-only on journal/thetadata_options.db. NEVER overwrites runner_v1.lgb — honest
serving models saved to runner_v1_honest_r{pct}.lgb.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_prediction as RP  # reuse connect/load_stock_daily/walk_forward/FEATURES

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "journal" / "v3_eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_CSV = OUT_DIR / "runner_samples_enriched.csv"

HONEST_FILL_SPREAD = float(os.getenv("HONEST_FILL_SPREAD", "0.02"))
HONEST_FILL_DELAY = int(os.getenv("HONEST_FILL_DELAY", "1"))
THRESHOLDS = [50, 75, 100]


def process_ticker_enriched(con, ticker, stock_daily):
    """process_ticker copy that ALSO captures ask@min, ask@min+DELAY (for the honest denom) and peak.

    Everything else (feature computation, ATM-by-|delta-0.5| pick, EOD skip) is byte-identical to
    runner_prediction.process_ticker so the two label modes differ ONLY in the denominator."""
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
    df = pd.read_sql_query(q, con, params=(ticker, RP.DATE_LO, RP.DATE_HI))
    if df.empty:
        return []
    hh = df["ts"].str[11:13].astype(int)
    mm = df["ts"].str[14:16].astype(int)
    df["min_idx"] = (hh - 9) * 60 + (mm - 30)

    rows = []
    for d, day_df in df.groupby("d"):
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
            prior_close = sd["prior_close"]; prior_range = sd["prior_range_pct"]
        except KeyError:
            prior_close = np.nan; prior_range = np.nan
        gap_pct = ((day_open_und / prior_close - 1) * 100 if prior_close and prior_close > 0 else 0.0)
        if pd.isna(gap_pct):
            gap_pct = 0.0
        if pd.isna(prior_range):
            prior_range = 0.0
        und_by_min = (day_df.dropna(subset=["underlying_price"])
                      .groupby("min_idx")["underlying_price"].last().sort_index())
        if len(und_by_min) < RP.MIN_DAY_CANDLES:
            continue
        last_min = int(day_df["min_idx"].max())
        for em in RP.ENTRY_MINUTES:
            if em > last_min - RP.EOD_SKIP_MIN - 10:
                continue
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

            # ── honest-fill denominator inputs ──
            ask_now = float(pick["ask"]) if pd.notna(pick["ask"]) and pick["ask"] > 0 else np.nan
            # ask at em+DELAY for the SAME strike (captures the run-up between signal and fill)
            nxt = day_df[(day_df["strike"] == strike) & (day_df["min_idx"] == em + HONEST_FILL_DELAY)]
            ask_next = np.nan
            if not nxt.empty:
                av = nxt["ask"].dropna()
                if not av.empty and float(av.iloc[0]) > 0:
                    ask_next = float(av.iloc[0])

            fut = day_df[(day_df["strike"] == strike) &
                         (day_df["min_idx"] > em) &
                         (day_df["min_idx"] <= last_min - RP.EOD_SKIP_MIN)]
            future_vals = pd.concat([fut["high"].dropna(), fut["close"].dropna()])
            future_vals = future_vals[future_vals > 0]
            if future_vals.empty:
                continue
            peak = float(future_vals.max())

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
            und_slope_5 = ((recent.iloc[-1] / recent.iloc[0] - 1) * 100
                           if len(recent) >= 2 and recent.iloc[0] > 0 else 0.0)
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
                "gap_pct": gap_pct, "prior_range_pct": float(prior_range), "day_of_week": dow,
                # raw outcome + honest-denom inputs
                "peak": peak, "close_at_min": entry_prem,
                "ask_at_min": ask_now, "ask_next": ask_next,
            })
    return rows


def build_enriched():
    if ENRICHED_CSV.exists() and os.getenv("REBUILD", "0") != "1":
        print(f"Loading cached enriched dataset {ENRICHED_CSV}", flush=True)
        return pd.read_csv(ENRICHED_CSV)
    con = RP.connect()
    stock_daily = RP.load_stock_daily(con)
    all_rows = []
    for tk in RP.TICKERS:
        rs = process_ticker_enriched(con, tk, stock_daily)
        print(f"  {tk}: {len(rs)} samples", flush=True)
        all_rows.extend(rs)
    con.close()
    df = pd.DataFrame(all_rows).reset_index(drop=True)
    df.to_csv(ENRICHED_CSV, index=False)
    print(f"Enriched dataset -> {ENRICHED_CSV} ({len(df):,} rows)", flush=True)
    return df


def add_denominators(df):
    """fantasy denom = close; honest denom = max(close, ask@min, ask@min+1) * (1+spread)."""
    close = df["close_at_min"].astype(float)
    ask_now = df["ask_at_min"].astype(float)
    ask_next = df["ask_next"].astype(float)
    honest = np.maximum.reduce([
        close.values,
        np.where(np.isnan(ask_now.values), close.values, ask_now.values),
        np.where(np.isnan(ask_next.values), 0.0, ask_next.values),
    ])
    honest = honest * (1 + HONEST_FILL_SPREAD)
    df["denom_fantasy"] = close
    df["denom_honest"] = honest
    df["peak_gain_fantasy"] = (df["peak"] / df["denom_fantasy"] - 1) * 100
    df["peak_gain_honest"] = (df["peak"] / df["denom_honest"] - 1) * 100
    return df


def quartile_separation(df, oos_pred, target_col):
    """Realized runner-rate by predicted-P quartile (OOS, equal-count). Returns dict + monotone flag."""
    mask = ~np.isnan(oos_pred)
    y = df[target_col].values[mask]
    p = oos_pred[mask]
    order_q = pd.qcut(pd.Series(p).rank(method="first"), 4, labels=False)
    rates = pd.Series(y).groupby(order_q.values).mean() * 100
    counts = pd.Series(y).groupby(order_q.values).size()
    out = {f"Q{i+1}": float(rates.get(i, np.nan)) for i in range(4)}
    out_n = {f"Q{i+1}_n": int(counts.get(i, 0)) for i in range(4)}
    vals = [rates.get(i, np.nan) for i in range(4)]
    monotone = all(vals[i] <= vals[i + 1] for i in range(3))
    out["monotone_Q1<=Q2<=Q3<=Q4"] = bool(monotone)
    out["Q4_over_Q1"] = float(vals[3] / vals[0]) if vals[0] and vals[0] > 0 else float("nan")
    out.update(out_n)
    return out


def run_experiment(df, label_mode, thresh, save_model=False):
    """label_mode in {'fantasy','honest'}; thresh in %; returns metrics dict."""
    denom_col = "peak_gain_fantasy" if label_mode == "fantasy" else "peak_gain_honest"
    tcol = "runner"
    d = df.copy()
    d[tcol] = (d[denom_col] >= thresh).astype(int)
    base = d[tcol].mean()
    fold_rows, aucs, pooled, oos = RP.walk_forward(d, tcol)

    # quartile separation (all + expensive/near-ATM tier where the flip concentrates)
    qsep_all = quartile_separation(d, oos, tcol)
    exp_mask = (d["entry_premium"] >= d["entry_premium"].median()).values  # pricier half
    d_exp = d[exp_mask].reset_index(drop=True)
    oos_exp = oos[exp_mask]
    qsep_exp = quartile_separation(d_exp, oos_exp, tcol) if len(d_exp) else {}

    res = {
        "label_mode": label_mode, "thresh": thresh, "base_rate_pct": float(base * 100),
        "n_pos": int(d[tcol].sum()), "n": int(len(d)),
        "wf_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "wf_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "n_folds": len(aucs),
        "pooled_oos_auc": pooled.get("pooled_oos_auc") if pooled else None,
        "top10_lift": pooled.get("top_10pct_lift") if pooled else None,
        "top10_rate_pct": pooled.get("top_10pct_rate", 0) * 100 if pooled else None,
        "top5_lift": pooled.get("top_5pct_lift") if pooled else None,
        "qsep_all": qsep_all, "qsep_expensive": qsep_exp,
    }
    if save_model and label_mode == "honest":
        _save_honest_model(d, tcol, thresh, oos, fold_rows)
    return res


def _save_honest_model(d, tcol, thresh, oos, fold_rows):
    """Train serving model on ALL data, save to runner_v1_honest_r{thresh}.lgb. NEVER runner_v1.lgb."""
    dd = d.copy()
    for c in RP.CAT_FEATURES:
        dd[c] = dd[c].astype("category")
    y = dd[tcol].values
    best_iters = [f["best_iter"] for f in fold_rows if f.get("best_iter")]
    nboost = int(np.median(best_iters)) if best_iters else 400
    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "max_depth": 6, "lambda_l1": 1.0, "lambda_l2": 2.0,
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
    }
    dtr = lgb.Dataset(dd[RP.FEATURES], label=y, categorical_feature=RP.CAT_FEATURES, free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=max(nboost, 100))
    model_dir = PROJECT_DIR / "journal" / "models" / "ml_v3"
    path = model_dir / f"runner_v1_honest_r{thresh}.lgb"
    assert path.name != "runner_v1.lgb"
    model.save_model(str(path))
    # tilt cut points from OOS predicted-P quartiles (train percentiles used live)
    p_all = oos[~np.isnan(oos)]
    q = np.quantile(p_all, [0.25, 0.5, 0.75]) if len(p_all) else [0, 0, 0]
    meta = {
        "model": f"runner_v1_honest_r{thresh}",
        "target": f"runner = intraday peak gain >= {thresh}% (ATM CALL), HONEST-fill label",
        "label_fix": (f"denom = max(close, ask@min, ask@min+{HONEST_FILL_DELAY}bar) * "
                      f"(1+{HONEST_FILL_SPREAD}); vs deployed fantasy denom = close@min"),
        "features": RP.FEATURES, "cat_features": RP.CAT_FEATURES, "num_features": RP.NUM_FEATURES,
        "trained_on": [str(d["date"].min()), str(d["date"].max())],
        "n_samples": int(len(d)), "n_boost_round": int(max(nboost, 100)),
        "runner_threshold_pct": thresh,
        "suggested_quartile_cuts": {"q1": float(q[0]), "q2": float(q[1]), "q3": float(q[2])},
        "note": "HONEST-label retrain (2026-07-29). NOT deployed. See scripts/runner_honest_retrain.py.",
    }
    with open(model_dir / f"runner_v1_honest_r{thresh}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"  saved honest serving model -> {path.name} ({len(d):,} samples, {max(nboost,100)} rounds)",
          flush=True)


def main():
    print("=== Runner_v1 honest-label retrain + OOS A/B ===", flush=True)
    df = build_enriched()
    df = add_denominators(df)

    # label-flip diagnostics
    print(f"\nHonest denom = max(close, ask@min, ask@min+{HONEST_FILL_DELAY}) * (1+{HONEST_FILL_SPREAD})")
    med_markup = ((df["denom_honest"] / df["denom_fantasy"] - 1) * 100).median()
    print(f"median honest markup over close: {med_markup:.2f}%  "
          f"(ask_next present: {df['ask_next'].notna().mean()*100:.1f}%)")
    for t in THRESHOLDS:
        f_pos = (df["peak_gain_fantasy"] >= t)
        h_pos = (df["peak_gain_honest"] >= t)
        flipped = (f_pos & ~h_pos).sum()
        print(f"  thr {t}%: fantasy pos={f_pos.sum()} honest pos={h_pos.sum()} "
              f"flip(f→not-h)={flipped} ({flipped/max(f_pos.sum(),1)*100:.1f}% of fantasy positives)")

    results = []
    for t in THRESHOLDS:
        for mode in ["fantasy", "honest"]:
            print(f"\n--- {mode} label, runner>={t}% ---", flush=True)
            r = run_experiment(df, mode, t, save_model=(mode == "honest"))
            print(f"    base={r['base_rate_pct']:.2f}% WF_AUC={r['wf_auc_mean']:.4f}±{r['wf_auc_std']:.4f} "
                  f"pooledAUC={r['pooled_oos_auc']:.4f} top10lift={r['top10_lift']:.2f}x", flush=True)
            print(f"    Qsep(all)   Q1..Q4 rate%: "
                  f"{r['qsep_all'].get('Q1',0):.1f} {r['qsep_all'].get('Q2',0):.1f} "
                  f"{r['qsep_all'].get('Q3',0):.1f} {r['qsep_all'].get('Q4',0):.1f} "
                  f"mono={r['qsep_all'].get('monotone_Q1<=Q2<=Q3<=Q4')} "
                  f"Q4/Q1={r['qsep_all'].get('Q4_over_Q1'):.2f}", flush=True)
            print(f"    Qsep(exp)   Q1..Q4 rate%: "
                  f"{r['qsep_expensive'].get('Q1',0):.1f} {r['qsep_expensive'].get('Q2',0):.1f} "
                  f"{r['qsep_expensive'].get('Q3',0):.1f} {r['qsep_expensive'].get('Q4',0):.1f} "
                  f"mono={r['qsep_expensive'].get('monotone_Q1<=Q2<=Q3<=Q4')} "
                  f"Q4/Q1={r['qsep_expensive'].get('Q4_over_Q1'):.2f}", flush=True)
            results.append(r)

    (OUT_DIR / "runner_honest_ab_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nMetrics -> {OUT_DIR / 'runner_honest_ab_metrics.json'}")


if __name__ == "__main__":
    main()
