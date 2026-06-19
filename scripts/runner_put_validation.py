"""VALIDATE runner_put_v1 — can we FIND RUNNERS FOR PUTS?

The PUT analog of the validated CALL runner work. Clean model-OOS: runner_put_v1 was
trained on 2024-2025 ONLY (train_runner_put_v1.py), so 2026 is a true hold-out for BOTH
the model and the sizing map.

Three tests, mirroring the call analysis:
  TEST 1 — 2026 OOS SEPARATION (the headline). Score every 2026 ATM-0DTE PUT proxy entry
    (10:00 ET, nearest -0.50 delta) with runner_put_v1. Report AUC@50% / AUC@100%,
    Spearman(P vs realized), and quartile monotonicity of realized return via the REAL
    ExitFSM (V7 PUT exit cfg, no profit ceiling). Confirms P(runner) has real spread.
  TEST 2 — REAL PUT TRADES cross-check. Score the real PUT entries in real_signals_kody.csv
    (REAL_ML/REAL_DISCORD puts) with runner_put_v1; AUC/quartiles (small n, directional).
  TEST 3 — WALK-FORWARD SIZING. Fit a quartile sizing map (0.5/0.85/1.15/1.5x) on TRAIN
    (2024-2025), freeze, apply to 2026 TEST; flat vs sized per-unit-capital PF/P&L OOS.

NO-LEAKAGE: model trained 2024-25; sizing cuts fit on 2024-25; 2026 = clean OOS for both.
Features entry-minute-only.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/runner_put_validation.py [--validate]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import runner_separation_realdata as RS  # noqa: E402  (real-signal scorer + feature builders + sim)
import entry_timing_oracle as OR  # noqa: E402  (load_0dte + sim)
import uw_ticker_discovery as D  # noqa: E402  (_stock + ticker cfg + apply_v7_wide_trail_exits)

LIQUID_PUTS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AMD", "AMZN", "AAPL", "IWM", "GOOGL", "MSFT"]
QUART_MULTS = [0.5, 0.85, 1.15, 1.5]

_PUT_MODEL = None
_PUT_META = None


def load_runner_put_v1():
    global _PUT_MODEL, _PUT_META
    if _PUT_MODEL is not None:
        return _PUT_MODEL, _PUT_META
    import json
    import lightgbm as lgb
    p = PROJECT / "journal" / "models" / "ml_v3" / "runner_put_v1.lgb"
    _PUT_MODEL = lgb.Booster(model_file=str(p))
    _PUT_META = json.load(open(PROJECT / "journal" / "models" / "ml_v3" / "runner_put_v1_meta.json"))
    return _PUT_MODEL, _PUT_META


def score_put(feat):
    import pandas as pd
    model, meta = load_runner_put_v1()
    cols = meta["features"]
    df = pd.DataFrame([{c: feat.get(c) for c in cols}], columns=cols)
    for c in meta["cat_features"]:
        df[c] = df[c].astype("category")
    return float(model.predict(df)[0])


def auc(scores, labels):
    from scipy.stats import rankdata
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def quartile_table(p, rets, run_lbl):
    order = np.argsort(p)
    p = np.asarray(p, float)[order]
    rets = np.asarray(rets, float)[order]
    run_lbl = np.asarray(run_lbl, float)[order]
    n = len(p)
    lines, avgs = [], []
    for qi in range(4):
        lo = qi * n // 4
        hi = (qi + 1) * n // 4 if qi < 3 else n
        if hi <= lo:
            continue
        rr = rets[lo:hi]
        avgs.append(float(rr.mean()))
        lines.append(f"    Q{qi+1} (p={p[lo]:.3f}-{p[hi-1]:.3f}, n={hi-lo}): "
                     f"runner%={run_lbl[lo:hi].mean()*100:4.0f}  WR={(rr>0).mean()*100:4.0f}%  "
                     f"avgRet={rr.mean():+7.1f}%")
    mono = all(avgs[i] <= avgs[i+1] for i in range(len(avgs)-1)) if len(avgs) >= 2 else False
    return "\n".join(lines), mono, avgs


def per_unit(rets, mults):
    rets = np.asarray(rets, float); mults = np.asarray(mults, float)
    cap = mults.sum()
    win = float((rets[rets > 0] * mults[rets > 0]).sum())
    loss = float(-(rets[rets < 0] * mults[rets < 0]).sum())
    return {"n": len(rets), "pnl": float((rets * mults).sum()),
            "per_unit": float((rets * mults).sum()) / cap if cap else 0.0,
            "pf": win / loss if loss > 0 else float("inf"), "cap": float(cap)}


def quartile_mult(p, cuts):
    if p < cuts[0]:
        return QUART_MULTS[0]
    if p < cuts[1]:
        return QUART_MULTS[1]
    if p < cuts[2]:
        return QUART_MULTS[2]
    return QUART_MULTS[3]


# ----------------------------------------------------------- build proxy PUT rows (2024-2026)
def build_proxy_puts(tickers, validate=False):
    """ATM-0DTE PUT entries at 10:00 ET; runner_put_v1 score + realized via ExitFSM (PUT cfg)."""
    side = "put"; right = "PUT"
    recs = []
    RS.load_runner_v1  # ensure module import ok
    load_runner_put_v1()
    for tk in tickers:
        df = OR.load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True); continue
        stock_close = D._stock(tk)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=side), is_put=True)
        theta = RS.connect(RS.THETA_DB)
        nd = 0
        for day, g in df.groupby("date"):
            if validate and day[:7] not in ("2025-03", "2026-04"):
                continue
            if day not in stock_close or OR.ENTRY_MI not in stock_close[day]:
                continue
            spot = stock_close[day][OR.ENTRY_MI]
            strikes = g["strike"].unique()
            atm = float(strikes[np.argmin(np.abs(strikes - spot))])
            ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= OR.ENTRY_MI)].sort_values("mi")
            if len(ch) < 5:
                continue
            mi_all = ch["mi"].to_numpy(int)
            prem_all = ch["close"].to_numpy(float)
            if mi_all[0] != OR.ENTRY_MI or np.isnan(prem_all[0]) or prem_all[0] <= 0:
                continue

            exp = day  # 0DTE
            strike_used, rrows, why = RS.load_contract_day(theta, tk, right, exp, day, atm, tol_pct=5.0)
            if rrows is None:
                continue
            stock_day = RS.load_stock_day(theta, tk, day)
            if not stock_day:
                continue
            entry_idx = RS.resolve_entry_idx(rrows, OR.ENTRY_MI)
            if entry_idx is None or entry_idx >= len(rrows) - 2:
                continue
            # runner_v1 feature builder REQUIRES delta>0 (call assumption). For PUTs we build the
            # same 18-feature dict directly, keeping delta NEGATIVE (matches training).
            feats = build_put_features(theta, rrows, entry_idx, stock_day, tk, strike_used, day)
            if feats is None:
                continue
            p_put = score_put(feats)

            ep = float(prem_all[0])
            up = [stock_close[day].get(int(m), spot) for m in mi_all]
            ets = datetime(*map(int, day.split("-")), 9, 30, tzinfo=OR.ET) + timedelta(minutes=int(mi_all[0]))
            ret = OR.sim(prem_all, list(mi_all), up, cfg, side, ets)
            _, peak_gain = RS.realized_outcome(rrows, entry_idx, "put", day)

            recs.append({"p": p_put, "ret": ret, "day": day, "tk": tk, "peak": peak_gain,
                         "delta": feats["delta"]})
            nd += 1
        theta.close()
        print(f"  {tk}: {nd} scored proxy puts", flush=True)
    return recs


def build_put_features(theta, rrows, entry_idx, stock_day, ticker, strike_used, day):
    """The 18-feature schema for PUTs (delta kept negative). Mirrors train_runner_put_v1."""
    from datetime import date as _date
    er = rrows[entry_idx]
    if er["delta"] is None or er["delta"] >= 0 or er["close"] is None or er["close"] <= 0:
        return None
    entry_prem = er["close"]
    und_now = er["underlying_price"]
    if not und_now or und_now <= 0:
        ms = sorted(m for m in stock_day if m <= er["mi"])
        und_now = stock_day[ms[-1]] if ms else 0
    if not und_now or und_now <= 0:
        return None
    iv = er["iv"] if er["iv"] is not None else 0.0
    vega = er["vega"] if er["vega"] is not None else 0.0
    theta_v = er["theta"] if er["theta"] is not None else 0.0
    bid = er["bid"] if er["bid"] is not None else 0.0
    ask = er["ask"] if er["ask"] is not None else 0.0
    spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0
    moneyness = strike_used / und_now if und_now > 0 else 1.0
    entry_mi = er["mi"]

    day_open_und = stock_day.get(min(stock_day.keys())) if stock_day else und_now
    und_move_pct = (und_now / day_open_und - 1) * 100 if day_open_und else 0.0
    recent = [stock_day[m] for m in sorted(stock_day) if entry_mi - 5 < m <= entry_mi]
    und_slope_5 = (recent[-1] / recent[0] - 1) * 100 if len(recent) >= 2 and recent[0] > 0 else 0.0
    r15 = [stock_day[m] for m in sorted(stock_day) if entry_mi - 15 < m <= entry_mi]
    if len(r15) >= 5:
        arr = np.array(r15)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.diff(arr) / arr[:-1]
        und_rvol_15 = float(np.nanstd(d[np.isfinite(d)]) * 100) if np.isfinite(d).any() else 0.0
    else:
        und_rvol_15 = 0.0
    opt_vol_5 = float(sum(r["volume"] for r in rrows if entry_mi - 5 < r["mi"] <= entry_mi))

    prior_close, prior_range = RS.prior_day_stats(theta, ticker, day)
    gap_pct = ((day_open_und / prior_close - 1) * 100) if (prior_close and day_open_und) else 0.0
    try:
        dow = _date(*[int(x) for x in day.split("-")]).weekday()
    except Exception:
        dow = 0

    return {
        "entry_premium": entry_prem, "log_premium": float(np.log(entry_prem)),
        "delta": er["delta"], "iv": iv, "vega": vega, "theta": theta_v,
        "moneyness": moneyness, "spread_pct": spread_pct,
        "und_move_pct": und_move_pct, "und_slope_5": und_slope_5,
        "und_rvol_15": und_rvol_15, "opt_vol_5": opt_vol_5,
        "gap_pct": gap_pct if gap_pct is not None else 0.0,
        "prior_range_pct": prior_range, "dte": 0, "entry_min": entry_mi,
        "ticker": ticker, "day_of_week": dow,
    }


# ----------------------------------------------------------- TEST 1: 2026 OOS separation
def test1_separation(validate=False):
    print("\n" + "=" * 80)
    print("TEST 1 — 2026 OOS SEPARATION (clean model hold-out)  [THE HEADLINE]")
    print("=" * 80)
    tickers = ["SPY", "TSLA"] if validate else LIQUID_PUTS
    recs = build_proxy_puts(tickers, validate=validate)
    if not recs:
        print("  no rows"); return None
    by_yr = {}
    for r in recs:
        by_yr.setdefault(r["day"][:4], 0); by_yr[r["day"][:4]] += 1
    print(f"\n  total scored proxy puts: {len(recs)}  by year: {by_yr}")
    print(f"  delta sanity (should be negative): "
          f"min={min(r['delta'] for r in recs):.3f} max={max(r['delta'] for r in recs):.3f}")

    test = [r for r in recs if r["day"][:4] == "2026"]
    if not test:
        print("  no 2026 rows"); return None
    te_p = [r["p"] for r in test]
    te_ret = [r["ret"] for r in test]
    te_run50 = [1 if r["peak"] >= 50 else 0 for r in test]
    te_run100 = [1 if r["peak"] >= 100 else 0 for r in test]
    a50 = auc(te_p, te_run50); a100 = auc(te_p, te_run100)
    sp = spearman(te_p, te_ret)
    print(f"\n  2026 TEST n={len(test)}  P(runner) spread: min={min(te_p):.3f} "
          f"med={np.median(te_p):.3f} max={max(te_p):.3f} std={np.std(te_p):.3f}")
    print(f"  base runner@50%={np.mean(te_run50)*100:.0f}%  @100%={np.mean(te_run100)*100:.0f}%")
    print(f"  AUC@50%={a50 if a50 is None else round(a50,3)}  "
          f"AUC@100%={a100 if a100 is None else round(a100,3)}  "
          f"Spearman(P vs realized ExitFSM ret)={sp if sp is None else round(sp,3)}")
    print(f"  quartiles by P(runner) (realized ExitFSM ret should rise L->R):")
    qtbl, mono, avgs = quartile_table(te_p, te_ret, te_run50)
    print(qtbl)
    print(f"  monotone-increasing avgRealized across quartiles? {mono}  "
          f"(avgs={[round(a,1) for a in avgs]})")
    return {"recs": recs, "test": test, "a50": a50, "a100": a100, "mono": mono,
            "spearman": sp, "avgs": avgs}


# ----------------------------------------------------------- TEST 2: real put trades
def test2_real_puts():
    print("\n" + "=" * 80)
    print("TEST 2 — REAL PUT trades cross-check (REAL_ML + REAL_DISCORD, small n)")
    print("=" * 80)
    theta = RS.connect(RS.THETA_DB)
    real = RS.load_real_signals()
    puts = [e for e in real if e["otype"] == "put" and e["arm"] in ("REAL_ML", "REAL_DISCORD")]
    print(f"  loaded {len(puts)} real PUT signals")
    results, skips = RS.process(puts, theta)

    recs = []
    for r in results:
        # rebuild PUT features (RS.build_runner_v1_features rejects delta<0); use recorded outcome
        recs.append(r)
    theta.close()

    # score each with runner_put_v1 via our PUT feature builder by re-loading the contract day
    scored = []
    theta = RS.connect(RS.THETA_DB)
    for e in puts:
        ticker = e["ticker"]; exp = e["expiry"]; strike = e["strike"]
        day = e["entry_dt"].strftime("%Y-%m-%d")
        entry_mi = (e["entry_dt"].hour - 9) * 60 + e["entry_dt"].minute - 30
        if strike <= 0:
            continue
        strike_used, rrows, why = RS.load_contract_day(theta, ticker, "PUT", exp, day, strike)
        if rrows is None:
            continue
        entry_idx = RS.resolve_entry_idx(rrows, entry_mi)
        if entry_idx is None or entry_idx >= len(rrows) - 2:
            continue
        stock_day = RS.load_stock_day(theta, ticker, day)
        if not stock_day:
            continue
        feats = build_put_features(theta, rrows, entry_idx, stock_day, ticker, strike_used, day)
        if feats is None:
            continue
        p_put = score_put(feats)
        # outcome: prefer recorded mfe/pnl (ground truth)
        if e["rec_mfe"] is not None and e["rec_pnl"] is not None:
            peak = e["rec_mfe"]; realized = e["rec_pnl"]; src = "recorded"
        else:
            realized, peak = RS.realized_outcome(rrows, entry_idx, "put", day)
            src = "reconstructed"
        scored.append({"p": p_put, "ret": realized, "peak": peak, "tk": ticker, "src": src,
                       "run50": 1 if peak >= 50 else 0, "run30": 1 if peak >= 30 else 0})
    theta.close()

    n = len(scored)
    print(f"  matched + scored: {n} real puts")
    if n < 4:
        print("  too few to evaluate."); return None
    p = [r["p"] for r in scored]; ret = [r["ret"] for r in scored]
    run30 = [r["run30"] for r in scored]; run50 = [r["run50"] for r in scored]
    print(f"  P spread: min={min(p):.3f} med={np.median(p):.3f} max={max(p):.3f}")
    print(f"  base runner@30%={np.mean(run30)*100:.0f}%  @50%={np.mean(run50)*100:.0f}%  "
          f"WR={np.mean([1 if r['ret']>0 else 0 for r in scored])*100:.0f}%")
    a30 = auc(p, run30); a50 = auc(p, run50); sp = spearman(p, ret)
    print(f"  AUC@30%={a30 if a30 is None else round(a30,3)}  "
          f"AUC@50%={a50 if a50 is None else round(a50,3)}  "
          f"Spearman(P vs realized)={sp if sp is None else round(sp,3)}")
    if n >= 8:
        qtbl, mono, avgs = quartile_table(p, ret, run30)
        print(f"  quartiles by P(runner) (runner%=@30%):")
        print(qtbl)
        print(f"  monotone avgRet? {mono}")
    return {"n": n, "a30": a30, "a50": a50, "spearman": sp}


# ----------------------------------------------------------- TEST 3: walk-forward sizing
def test3_sizing(recs):
    print("\n" + "=" * 80)
    print("TEST 3 — WALK-FORWARD SIZING  (fit map on 2024-25, freeze, apply to 2026)")
    print("=" * 80)
    train = [r for r in recs if r["day"][:4] in ("2024", "2025")]
    test = [r for r in recs if r["day"][:4] == "2026"]
    if not train or not test:
        print("  missing window"); return None
    tp = sorted(r["p"] for r in train)
    cuts = [float(np.quantile(tp, q)) for q in (0.25, 0.50, 0.75)]
    print(f"  TRAIN(2024-25) n={len(train)}  FROZEN quartile cuts: "
          f"<{cuts[0]:.3f}->x{QUART_MULTS[0]} | <{cuts[1]:.3f}->x{QUART_MULTS[1]} | "
          f"<{cuts[2]:.3f}->x{QUART_MULTS[2]} | >=->x{QUART_MULTS[3]}")

    te_ret = [r["ret"] for r in test]
    flat = per_unit(te_ret, [1.0] * len(test))
    sized = per_unit(te_ret, [quartile_mult(r["p"], cuts) for r in test])
    print(f"  TEST(2026) n={len(test)}")
    print(f"    FLAT : per-unit={flat['per_unit']:+6.2f}%  PF={flat['pf']:.2f}  "
          f"totP&L={flat['pnl']:+.0f}u")
    print(f"    SIZED: per-unit={sized['per_unit']:+6.2f}%  PF={sized['pf']:.2f}  "
          f"totP&L={sized['pnl']:+.0f}u  (per-unit {sized['per_unit']-flat['per_unit']:+.2f}pp, "
          f"PF {sized['pf']-flat['pf']:+.2f})")
    beats = sized["per_unit"] >= flat["per_unit"]
    print(f"    -> sized {'BEATS' if beats else 'DOES NOT BEAT'} flat per-unit OOS")
    return {"flat": flat, "sized": sized, "beats": beats}


def main():
    validate = "--validate" in sys.argv
    if validate:
        print("### VALIDATE — 2 tickers (SPY,TSLA) x 2 months, sanity ###")
        t1 = test1_separation(validate=True)
        if t1:
            test3_sizing(t1["recs"])
        print("\nVALIDATION done. Re-run without --validate for full report.")
        return

    t1 = test1_separation()
    t2 = test2_real_puts()
    t3 = test3_sizing(t1["recs"]) if t1 else None

    print("\n" + "=" * 80)
    print("DECISION — can we FIND PUT RUNNERS?  (vs CALL benchmark runner_v1 AUC 0.74 OOS)")
    print("=" * 80)
    if t1:
        a = t1["a100"] if t1["a100"] is not None else (t1["a50"] or 0)
        a50 = t1["a50"] or 0
        sep_real = (a > 0.6 or a50 > 0.6) and t1["mono"]
        size_ok = t3 and t3["beats"]
        print(f"  2026 OOS: AUC@50%={t1['a50']}  AUC@100%={t1['a100']}  "
              f"monotone-quartiles={t1['mono']}  Spearman={t1['spearman']}")
        if size_ok is not None:
            print(f"  Sizing OOS beats flat per-unit? {size_ok}")
        print()
        if sep_real and size_ok:
            print("  >>> GREENLIGHT: PUT runners DO separate OOS (AUC>0.6, monotone) AND sizing")
            print("      beats flat. Build a P(runner) PUT sizing tilt analogous to the call path.")
        elif not sep_real:
            print("  >>> KILL: PUT runners do NOT separate at entry (AUC ~0.5-0.6 / non-monotone).")
            print("      Puts have runners but they're NOT predictable at entry. Keep the no-ceiling")
            print("      PUT trail (it captures runners post-hoc); do NOT size by a fake entry signal.")
        else:
            print("  >>> PARTIAL: separation present but sizing didn't beat flat OOS — INCONCLUSIVE.")
    print("\n  CAVEATS: proxy 10:00-ET ATM-0DTE entries (not live signals); thetadata minute-HIGH")
    print("    peak metric over-states runner% on illiquid 0DTE puts (single-tick spikes) — SAME")
    print("    metric as the call model, so the comparison is apples-to-apples; realized ret uses")
    print("    the real ExitFSM (no spread modeling on fills). Put delta kept NEGATIVE in features.")
    print("    Real-put n is small (directional only).")


if __name__ == "__main__":
    main()
