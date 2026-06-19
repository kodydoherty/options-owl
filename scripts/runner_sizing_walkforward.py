"""WALK-FORWARD validation of runner_v1 P(runner)->size MAPPING (NOT the model).

The prior study (runner_separation_realdata.py) fit the sizing multipliers IN-SAMPLE on the
same trades it measured. This script confirms (or KILLS) that the sizing edge survives OUT-OF-SAMPLE
by a time-split walk-forward. The runner_v1 MODEL is fixed (journal/models/ml_v3/runner_v1.lgb) —
we DO NOT retrain it. We only fit the P(runner)->size mapping (quartile cut points + linear
slope/intercept) on a TRAIN window, FREEZE it, and apply to a held-out TEST window.

ARM A — real calls, time-split (small n, the real-signal read):
  reuse runner_separation_realdata to score REAL call entries (REAL_ML+REAL_DISCORD) with runner_v1;
  outcome = recorded mfe/pnl. Sort by date, split early/late (50/50 and 60/40). Fit map on TRAIN,
  apply to TEST, compare flat vs sized per-unit-capital P&L + PF on TEST only.

ARM B — 2.5yr ATM-0DTE proxy calls, time-split (large-n robustness backbone):
  reuse entry_timing_oracle loaders + sim() to build ATM-0DTE CALL entries at 10:00 ET across the
  liquid set 2024-2026; score each with runner_v1; realized return via the real ExitFSM (LOCK cfg).
  Fit map on 2024+2025 (TRAIN), test on 2026 (the hard OOS regime). Report AUC + quartile
  monotonicity in 2026 (did separation hold?) and flat-vs-sized per-unit in 2026.

NO-LEAKAGE: cut points, slope, intercept, and normalization are computed on TRAIN ONLY and printed,
then frozen before TEST is touched. runner_v1 feature build is entry-minute-only (no lookahead),
identical to runner_separation_realdata / runner_prediction.

CAVEAT baked into the report: runner_v1_meta says trained_on 2025-01-02..2026-06-09, so 2026 is NOT
clean OOS for the MODEL — only for the sizing MAP. Flagged explicitly below.

Read-only on all DBs. Run: cd /Users/kody/dev/options-owl && python scripts/runner_sizing_walkforward.py
Optional: VALIDATE=1 (small smoke test, ~2 tickers x few months in Arm B + Arm A scored as-is).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

# Arm A reuses the real-data scoring harness wholesale (feature build + scoring + outcome).
import runner_separation_realdata as RS  # noqa: E402
# Arm B reuses the oracle loaders + sim.
import entry_timing_oracle as OR  # noqa: E402
import uw_ticker_discovery as D  # noqa: E402

# Sizing-map config (the thing we are validating generalizes)
QUART_MULTS = [0.5, 0.85, 1.15, 1.5]   # quartile multipliers (same as prior in-sample study)
LIN_LO, LIN_HI = 0.5, 1.5              # continuous P(runner)->[LIN_LO, LIN_HI] linear map range


# ============================================================ sizing-map fit/apply (no leakage)
def fit_quartile_cuts(train_p):
    """Cut points = the 25/50/75 percentiles of P(runner) on TRAIN ONLY. Returns 3 thresholds."""
    a = np.sort(np.asarray(train_p, float))
    return [float(np.quantile(a, q)) for q in (0.25, 0.50, 0.75)]


def quartile_mult(p, cuts):
    """Map a P(runner) to a quartile multiplier using FROZEN train cut points."""
    if p < cuts[0]:
        return QUART_MULTS[0]
    if p < cuts[1]:
        return QUART_MULTS[1]
    if p < cuts[2]:
        return QUART_MULTS[2]
    return QUART_MULTS[3]


def fit_linear_map(train_p):
    """Fit a linear P->mult on TRAIN ONLY: normalize by train [min,max] then scale to [LIN_LO,LIN_HI].

    Returns (p_min, p_max). Apply: mult = LIN_LO + (clip((p-p_min)/(p_max-p_min),0,1))*(LIN_HI-LIN_LO).
    The normalization bounds are TRAIN statistics -> frozen before TEST.
    """
    a = np.asarray(train_p, float)
    return float(a.min()), float(a.max())


def linear_mult(p, bounds):
    p_min, p_max = bounds
    if p_max <= p_min:
        return 1.0
    z = (p - p_min) / (p_max - p_min)
    z = min(1.0, max(0.0, z))
    return LIN_LO + z * (LIN_HI - LIN_LO)


def per_unit_pf(rets, mults):
    """Per-unit-capital P&L and PF for a set of returns(%) under per-trade multipliers.

    per-unit = sum(ret*mult)/sum(mult)  (normalizes for capital deployed -> fair vs flat).
    PF = sum(positive ret*mult)/abs(sum(negative ret*mult)).
    """
    rets = np.asarray(rets, float)
    mults = np.asarray(mults, float)
    cap = mults.sum()
    pnl = float((rets * mults).sum())
    win = float((rets[rets > 0] * mults[rets > 0]).sum())
    loss = float(-(rets[rets < 0] * mults[rets < 0]).sum())
    pf = win / loss if loss > 0 else float("inf")
    return {"n": len(rets), "pnl": pnl, "per_unit": pnl / cap if cap else 0.0, "pf": pf, "cap": float(cap)}


def auc(scores, labels):
    from scipy.stats import rankdata
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def quartile_monotone(p, rets, runner_lbl):
    """Quartile table on a single set (sorted by p): runner-rate, WR, avgRet per quartile."""
    order = np.argsort(p)
    p = np.asarray(p, float)[order]
    rets = np.asarray(rets, float)[order]
    runner_lbl = np.asarray(runner_lbl, float)[order]
    n = len(p)
    out = []
    avgs = []
    for qi in range(4):
        lo = qi * n // 4
        hi = (qi + 1) * n // 4 if qi < 3 else n
        if hi <= lo:
            continue
        rr = rets[lo:hi]
        avgs.append(float(rr.mean()))
        out.append(f"      Q{qi+1} (p={p[lo]:.3f}-{p[hi-1]:.3f}, n={hi-lo}): "
                   f"runner%={runner_lbl[lo:hi].mean()*100:4.0f}  WR={(rr>0).mean()*100:4.0f}%  "
                   f"avgRet={rr.mean():+6.1f}%")
    monotone = all(avgs[i] <= avgs[i+1] for i in range(len(avgs)-1)) if len(avgs) >= 2 else False
    return "\n".join(out), monotone, avgs


# ============================================================ TEST evaluation (frozen map)
def eval_split(train, test, label):
    """train/test = lists of dicts with 'p' (P(runner)) and 'ret' (realized %). Fit on train, eval on test."""
    tp = [r["p"] for r in train]
    cuts = fit_quartile_cuts(tp)
    lin = fit_linear_map(tp)

    print(f"\n  --- {label} ---")
    print(f"  TRAIN n={len(train)}  P(runner) min={min(tp):.3f} med={np.median(tp):.3f} max={max(tp):.3f}")
    print(f"  FROZEN quartile cuts (from TRAIN): "
          f"<{cuts[0]:.3f}->x{QUART_MULTS[0]} | <{cuts[1]:.3f}->x{QUART_MULTS[1]} | "
          f"<{cuts[2]:.3f}->x{QUART_MULTS[2]} | >=->x{QUART_MULTS[3]}")
    print(f"  FROZEN linear map (from TRAIN): p_min={lin[0]:.3f} p_max={lin[1]:.3f} -> "
          f"mult in [{LIN_LO},{LIN_HI}] (clipped)")

    if not test:
        print("  TEST n=0 — no held-out trades"); return None

    te_p = [r["p"] for r in test]
    te_ret = [r["ret"] for r in test]
    flat = per_unit_pf(te_ret, [1.0] * len(te_ret))
    q_mults = [quartile_mult(p, cuts) for p in te_p]
    lin_mults = [linear_mult(p, lin) for p in te_p]
    sized_q = per_unit_pf(te_ret, q_mults)
    sized_l = per_unit_pf(te_ret, lin_mults)

    print(f"  TEST  n={len(test)}  P(runner) min={min(te_p):.3f} med={np.median(te_p):.3f} max={max(te_p):.3f}")
    print(f"    FLAT       : per-unit={flat['per_unit']:+6.2f}%  PF={flat['pf']:.2f}  totP&L={flat['pnl']:+.0f}u  cap={flat['cap']:.1f}")
    print(f"    SIZED-quart: per-unit={sized_q['per_unit']:+6.2f}%  PF={sized_q['pf']:.2f}  totP&L={sized_q['pnl']:+.0f}u  cap={sized_q['cap']:.1f}  "
          f"(per-unit {sized_q['per_unit']-flat['per_unit']:+.2f}pp, PF {sized_q['pf']-flat['pf']:+.2f})")
    print(f"    SIZED-lin  : per-unit={sized_l['per_unit']:+6.2f}%  PF={sized_l['pf']:.2f}  totP&L={sized_l['pnl']:+.0f}u  cap={sized_l['cap']:.1f}  "
          f"(per-unit {sized_l['per_unit']-flat['per_unit']:+.2f}pp, PF {sized_l['pf']-flat['pf']:+.2f})")
    return {"flat": flat, "sized_q": sized_q, "sized_l": sized_l, "cuts": cuts, "lin": lin}


# ============================================================ ARM A — real calls
def arm_a():
    print("\n" + "=" * 80)
    print("ARM A — REAL calls (REAL_ML + REAL_DISCORD), runner_v1, time-split walk-forward")
    print("=" * 80)
    theta = RS.connect(RS.THETA_DB)
    real = RS.load_real_signals()
    calls = [e for e in real if e["otype"] == "call" and e["arm"] in ("REAL_ML", "REAL_DISCORD")]
    print(f"Loaded {len(calls)} REAL_ML+REAL_DISCORD call signals")
    results, skips = RS.process(calls, theta)
    theta.close()

    # keep rows with a runner_v1 score AND a real recorded outcome
    recs = []
    for r in results:
        if r.get("p_rv1") is None:
            continue
        recs.append({"p": r["p_rv1"], "ret": r["realized"], "day": r["day"],
                     "peak": r["peak_gain"], "runner50": r["is_runner50"]})
    recs.sort(key=lambda r: r["day"])
    n = len(recs)
    print(f"\nMatched + scored: {n} calls (skips: {len(skips)})")
    if n < 8:
        print("  TOO FEW to split — Arm A inconclusive.")
        return None
    print(f"  date range {recs[0]['day']} .. {recs[-1]['day']}  base runner@50%={np.mean([r['runner50'] for r in recs])*100:.0f}%")

    out = {}
    for frac, tag in [(0.50, "50/50 split"), (0.60, "60/40 split")]:
        k = int(round(n * frac))
        train, test = recs[:k], recs[k:]
        split_day = test[0]["day"] if test else "n/a"
        out[tag] = eval_split(train, test, f"Arm A {tag} (train<{split_day}, test>={split_day})")
    return out


# ============================================================ ARM B — 2.5yr proxy calls
def arm_b(validate=False):
    print("\n" + "=" * 80)
    print("ARM B — 2.5yr ATM-0DTE proxy CALLs (10:00 ET), runner_v1, fit 2024+25 -> test 2026")
    print("=" * 80)
    print("  NOTE: proxy entries (NOT live signals); large-n generalization check on the same runner_v1.")

    tickers = ["SPY", "NVDA"] if validate else OR.LIQUID_CALLS
    recs = build_armb_rows(tickers, validate=validate)
    n = len(recs)
    if n < 50:
        print(f"  only {n} rows — thin");
    recs.sort(key=lambda r: r["day"])
    yrs = {}
    for r in recs:
        yrs.setdefault(r["day"][:4], 0)
        yrs[r["day"][:4]] += 1
    print(f"\nTotal scored proxy calls: {n}  by year: {yrs}")
    if n == 0:
        return None

    train = [r for r in recs if r["day"][:4] in ("2024", "2025")]
    test = [r for r in recs if r["day"][:4] == "2026"]
    print(f"  TRAIN(2024+25) n={len(train)}   TEST(2026) n={len(test)}")
    if not train or not test:
        print("  missing a window — Arm B inconclusive"); return None

    # --- separation hold-up in 2026 (AUC + quartile monotonicity) BEFORE sizing ---
    te_p = [r["p"] for r in test]
    te_ret = [r["ret"] for r in test]
    te_run50 = [1 if r["peak"] >= 50 else 0 for r in test]
    te_run100 = [1 if r["peak"] >= 100 else 0 for r in test]
    a50 = auc(te_p, te_run50)
    a100 = auc(te_p, te_run100)
    sp = None
    from scipy.stats import rankdata
    if len(te_p) >= 3 and np.std(rankdata(te_p)) > 0 and np.std(rankdata(te_ret)) > 0:
        sp = float(np.corrcoef(rankdata(te_p), rankdata(te_ret))[0, 1])
    qtbl, monotone, avgs = quartile_monotone(te_p, te_ret, te_run50)
    print(f"\n  [2026 OOS separation — did P(runner) still rank outcomes?]")
    print(f"    base runner@50%={np.mean(te_run50)*100:.0f}% @100%={np.mean(te_run100)*100:.0f}%  "
          f"AUC@50%={a50 if a50 is None else round(a50,3)}  AUC@100%={a100 if a100 is None else round(a100,3)}  "
          f"Spearman(P vs ret)={sp if sp is None else round(sp,3)}")
    print(f"    quartiles by P(runner) (avgRet should rise L->R):")
    print(qtbl)
    print(f"    monotone-increasing avgRet across quartiles? {monotone}  (avgs={[round(a,1) for a in avgs]})")

    res = eval_split(train, test, "Arm B fit 2024+25 -> TEST 2026")

    # --- cheap expanding-window variant: train through year Y, test Y+1 ---
    print(f"\n  [Expanding-window variant]")
    for cut_yr, test_yr in [("2024", "2025"), ("2025", "2026")]:
        tr = [r for r in recs if r["day"][:4] <= cut_yr]
        teset = [r for r in recs if r["day"][:4] == test_yr]
        if not tr or not teset:
            continue
        eval_split(tr, teset, f"Arm B train<= {cut_yr} -> TEST {test_yr}")
    return {"res": res, "auc50": a50, "auc100": a100, "monotone": monotone, "spearman": sp,
            "n_test": len(test), "n_train": len(train)}


def build_armb_rows(tickers, validate=False):
    """Build ATM-0DTE CALL entries at 10:00 ET, score runner_v1, realized via ExitFSM. Returns list."""
    side = "call"
    right = "CALL"
    recs = []
    RS.load_runner_v1()  # warm
    for tk in tickers:
        df = OR.load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True); continue
        stock_close = D._stock(tk)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=side), is_put=False)
        # prior-day stats need a sqlite con (RS.prior_day_stats)
        theta = RS.connect(RS.THETA_DB)
        # stock_day with full bars for runner_v1 feature build (close only is enough for rv1 features)
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

            # --- runner_v1 features at the entry minute (need greeks from option_greeks join) ---
            # Build a 'rows' list compatible with RS.build_runner_v1_features for this contract/day.
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
            prior_close, prior_range = RS.prior_day_stats(theta, tk, day)
            day_open_und = stock_day.get(min(stock_day.keys()))
            gap_pct = ((day_open_und / prior_close - 1) * 100) if (prior_close and day_open_und) else 0.0
            rv1_feats = RS.build_runner_v1_features(rrows, entry_idx, stock_day, tk, strike_used,
                                                    0, gap_pct, prior_range, day)
            if rv1_feats is None:
                continue
            p_rv1 = RS.score_runner_v1(rv1_feats)

            # --- realized return via the SAME oracle sim() from the 10:00 entry to EOD ---
            ep = float(prem_all[0])
            up = [stock_close[day].get(int(m), spot) for m in mi_all]
            ets = datetime(*map(int, day.split("-")), 9, 30, tzinfo=OR.ET) + timedelta(minutes=int(mi_all[0]))
            ret = OR.sim(prem_all, list(mi_all), up, cfg, side, ets)
            # forward peak for separation labels (use option highs from rrows after entry, parity w/ RS)
            _, peak_gain = RS.realized_outcome(rrows, entry_idx, "call", day)

            recs.append({"p": p_rv1, "ret": ret, "day": day, "tk": tk, "peak": peak_gain})
            nd += 1
        theta.close()
        print(f"  {tk}: {nd} scored proxy calls", flush=True)
    return recs


# ============================================================ main
def greenlight(arm_a_out, arm_b_out):
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    a_ok = None
    if arm_a_out:
        # require BOTH the 50/50 test half to show sized >= flat per-unit (use quartile map as primary)
        r = arm_a_out.get("50/50 split")
        if r:
            a_ok = (r["sized_q"]["per_unit"] >= r["flat"]["per_unit"]) or \
                   (r["sized_l"]["per_unit"] >= r["flat"]["per_unit"])
            print(f"  Arm A 50/50 TEST: flat per-unit={r['flat']['per_unit']:+.2f}%  "
                  f"sized-q={r['sized_q']['per_unit']:+.2f}%  sized-lin={r['sized_l']['per_unit']:+.2f}%  "
                  f"-> {'sized BEATS flat' if a_ok else 'sized <= flat'}")
    b_ok = None
    if arm_b_out and arm_b_out.get("res"):
        r = arm_b_out["res"]
        sep_ok = (arm_b_out["auc50"] or 0) > 0.5 or (arm_b_out["auc100"] or 0) > 0.5
        size_ok = (r["sized_q"]["per_unit"] >= r["flat"]["per_unit"]) or \
                  (r["sized_l"]["per_unit"] >= r["flat"]["per_unit"])
        b_ok = sep_ok and size_ok
        print(f"  Arm B 2026 OOS: AUC@50%={arm_b_out['auc50']}  AUC@100%={arm_b_out['auc100']}  "
              f"monotone={arm_b_out['monotone']}  -> separation {'HOLDS' if sep_ok else 'COLLAPSES'}")
        print(f"  Arm B 2026 OOS: flat per-unit={r['flat']['per_unit']:+.2f}%  "
              f"sized-q={r['sized_q']['per_unit']:+.2f}%  sized-lin={r['sized_l']['per_unit']:+.2f}%  "
              f"-> {'sized BEATS flat' if size_ok else 'sized <= flat'}")

    print()
    if b_ok and (a_ok is not False):
        print("  >>> GREENLIGHT (conditional): sizing edge survives the large-n 2026 OOS test.")
    elif b_ok is False:
        print("  >>> KILL: 2026 large-n OOS shows separation collapse and/or sized <= flat — the")
        print("           in-sample 2x was optimistic. DO NOT build runner-proportional sizing as-is.")
    else:
        print("  >>> INCONCLUSIVE — see arm results above.")
    print("\n  CAVEAT: runner_v1_meta trained_on = 2025-01-02..2026-06-09 -> 2026 is NOT clean OOS")
    print("          for the MODEL (only for the sizing MAP). A true model-OOS would need a model")
    print("          trained only through 2025. Arm A real-n is tiny (2-month window). Arm B uses")
    print("          PROXY 10:00 entries, not live signals.")


def main():
    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        print("### VALIDATION PASS — Arm A scored as-is + Arm B on 2 tickers, few months ###")
    a_out = arm_a()
    b_out = arm_b(validate=validate)
    greenlight(a_out, b_out)


if __name__ == "__main__":
    main()
