# Runner-v1 P(runner) Call Sizing (Stage D serve path)

**Status:** active · **Created:** 2026-06-17 · **Owner:** Kody

## Hypothesis
Entry *timing* is unexploitable (settled: 3 real-data arms, adverse selection). But entry *selection/sizing*
via P(runner) IS the lever. `runner_v1` (ml_v3) separates real CALL outcomes (AUC 0.74; quartiles −8%→+9%),
and quartile-proportional sizing held out-of-sample (Arm A real calls +3.8pp; Arm B 2026 OOS +2.77pp, PF +0.18,
separation AUC 0.80 held). Bet bigger on the rippers we can't time into, shrink the round-trippers.

## Why it's a different answer from the 2026-06-15 "+0.06 PF" null
That null used the **weak serve model (signal_ml_v2, AUC 0.55)** on the already-gated flow set. The prod serve
path (`flow_runner.predict_entry_confidence`) still uses signal_ml_v2. This ships **runner_v1** (AUC 0.74) on
the broad CALL book — a different model + population.

## Build
1. **Live scorer** `compute_runner_v1_p(signal, settings)` in `risk/flow_runner.py` — reuse the existing live
   data fetch (greeks snapshot + option 1m bars + underlying 1m bars from Polygon, as `compute_flow_p_runner`),
   ADD a prior-day stock fetch for `gap_pct`/`prior_range_pct`. Build the 18 features EXACTLY as the validated
   `scripts/runner_separation_realdata.py:build_runner_v1_features` (source of truth; matches training
   `scripts/runner_prediction.py`). Score via `score_runner_v1` (lgb.Booster, pandas DataFrame in meta feature
   order, cat_features ticker/day_of_week as category). CALLS only (model abstains delta≤0). None on any
   missing data (safe no-op). Log P(runner) + key inputs prominently (observe-first).
2. **Sizing curve** `runner_v1_size_mult(p_runner, ...)` in `risk/vinny_strategy.py` — quartile map
   (walk-forward train cuts): <0.580→0.5× · <0.630→0.85× · <0.670→1.15× · ≥0.670→1.5×. (Linear map rejected —
   fragile OOS.) Env-tunable.
3. **Wiring** in `execution/paper_trader.py` — CALLS only, fold into `_conv_mult` before `score_to_contracts`
   (same pattern as regime budget), gated `ENABLE_RUNNER_V1_SIZING` (default false). Position caps still bound it.
4. **Settings** flag + 7 knobs. **Tests** for the sizing curve. **Paper canary** adam/vinny.

## Rollout (observe-first — the model meta warns thresholds may need recalibration on new live data)
Deploy paper → confirm the LIVE P(runner) distribution is sane (spread 0–1, not degenerate, ~matches backtest
mean/monotonicity) via logs for several sessions → only then trust the sizing / consider live (kody/dennis).

## Caveats
- runner_v1 trained 2025-01-02..2026-06-09 → forward paper is the true model-OOS test.
- CALLS only. Sizing is capital-efficiency (lifts losing regime to ~breakeven), not a standalone money printer.
- Live feature fidelity (Polygon live vs thetadata) is the key risk → observe-first gate above.

## Result
_(pending paper canary)_
