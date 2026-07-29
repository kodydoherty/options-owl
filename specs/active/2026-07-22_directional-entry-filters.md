# Directional Entry Filters — "Trade With the Underlying, Not Against It"

**Status:** active · **Created:** 2026-07-22 · **Owner:** Kody

## Hypothesis
On choppy tapes we bleed because we buy options *counter* to the underlying's current move — calls into
falling tickers, puts into flat/rising tickers. Requiring the underlying to already be moving in the option's
direction at entry ("move-from-open" confirmation) should cut the chop bleed. One principle fixes both sides.

## Root-cause finding (real live trades, kody+dennis)
The last 2 weeks bled ~−$3,600 (kody), ~2/3 from CALLS. Reconstructing each trade's entry context from the tape:
- **CALLS bought into a FALLING ticker (≤−0.5% from open): −$3,199, PF 0.41** (61 trades) — the single biggest
  bleed bucket. Calls on RISING tickers (≥+0.5%): **+$4,441, PF 1.35** (237 trades).
- **PUTS bought when the ticker was FLAT/UP (≥0%): −$2,013, PF ~0.3** — the mirror problem.

## What works (empirically ranked on real trades)
- **move-from-open is THE signal.** For calls: the single filter took the book −$17 → +$4,441 (6wk). It is the
  dominant predictor.
- **VWAP (price vs VWAP) is a weaker second** (calls +$2,294). EMA-trend weak. **RSI HURTS (−$1,374).**
- **Naive multi-signal combos are WORSE than move-from-open alone** — the weak/bad signals dilute the strong one
  (≥2-of-4 vote +$2,981 < move-from-open +$4,441; ≥3-of-4 +$138). "Smarter with more candle data" REFUTED;
  the *simple* directional filter wins. (Still testing targeted move-from-open+VWAP pair at optimal thresholds.)
- **Threshold matters + must not be overfit:** in the 2-wk chop, calls flip green at move-from-open ≥+0.75%
  (+$124) or VWAP ≥+0.2% (+$329) — but picking the value that maxes 2 weeks = curve-fitting. Threshold is being
  chosen on 6 months (see Open).

## PUT side (settled this session — separate but same principle)
- Stacking whippy-ticker exclusion (TSLA/NVDA/NFLX/MU/BA) + underlying-DOWN ≤−0.5% trigger flipped the live 6wk
  put book −$3,642 → **+$766** (in-sample), and validated out-of-sample on the 6-mo harness (+$3,990, PF 3.25,
  70% WR, 1.4% DD). Puts are a **regime-dependent defensive book**: bleed in chop, pay in selloffs; the filter
  makes them **positive in both regimes** (smaller but consistent). Puts are PAUSED on live money; the strict
  config is canarying on **yank** (paper). See [[put-fix-canary-2026-07-20]].

## Existing infra reviewed (don't rebuild)
`risk/pipeline.py:DirectionalRegimeGate` ALREADY does candle-based direction confirmation (RSI 5m/15m + bullish
candle count + 30-min momentum + EMA9/21 → regime_score, blocks calls if score < −1). Three reasons it misses the
bleed: (1) threshold too loose (−1), (2) uses 30-MIN momentum not move-from-OPEN (misses "down all day but flat
last 30min"), (3) flow calls bypass it except a hard −1.5% dive (3× too loose vs the −0.5% where calls bleed).
The put trigger built this session (`ENABLE_PUT_UNDERLYING_TRIGGER` in PutMarketDirectionGate) is the put mirror.

## What's deployed / canarying / testing
- **LIVE (all bots):** put underlying-trigger gate code (`ENABLE_PUT_UNDERLYING_TRIGGER`, default off) + whippy
  exclusion (yank only). Live money puts PAUSED (`ENABLE_PUT_TRADING=false`).
- **CANARY (yank, paper):** strict put filter (exclude whippy + ≤−0.5% trigger).
- **TESTING:** 6-month CALL move-from-open trigger sweep (`CALL_DIRECTION_TRIGGER` in backtest_gold_standard.py,
  values None/0/0.25/0.5/0.75/1.0) + a non-overlapping PRIOR-6mo out-of-sample sweep (`--offset 126`).

## Robustness protocol (to avoid overfitting — this is the discipline)
1. Choose the call trigger threshold on **6 months**, not 2 weeks.
2. Confirm the SAME threshold on a **non-overlapping prior 6 months** (out-of-sample).
3. Confirm it also helps the **live 2-week** loss window.
4. A robust filter is a **smooth plateau** across nearby thresholds, not a spike at one value.
5. **Canary on paper** before live money (same bar puts cleared).

## ⚠️ Harness is UNRELIABLE for the CALL test (2026-07-22) — validate via paper canary, not harness
The 6-month harness call sweep CONTRADICTED the live data: harness call book +$549k / 73% WR, and the
move-from-open filter CUT it to +$14k (harness ML model wins on dips — a "fantasy dip-buyer"). But LIVE ML calls
run 34-47% WR and LOSE on dips (−$1,450), same as flow. Split by source confirms BOTH ML and flow calls lose on
FALLING tickers (ML −$1,450, flow −$1,749) and win on RISING (flow +$3,894). → **The harness ML book is
fantasy-optimistic (73% vs live 34%), same fidelity gap that generated ~0 puts. Do NOT trust the harness for
call/put entry-filter validation.** The robust check "worked" by exposing the harness as the unreliable witness.
**The filter is supported by real trades; the only trustworthy long-window test is a LIVE PAPER CANARY** (like
the put fix on yank). `entry_und_move` was added to the harness trade dump (harmless, unused now).

## Open questions (in flight)
- Build the live CALL move-from-open gate (flag-gated, mirror of the put trigger, apply to ML + flow) and
  paper-canary it — the harness route is closed. Threshold: start ≥0% (require ticker not falling); the live
  data shows the big cut is the falling-ticker bucket, and ≥+0.5% higher thresholds thin the sample fast.
- Does move-from-open + VWAP (targeted pair) beat move-from-open alone? (untested at optimal thresholds)

## Harness / scripts (reusable)
`scripts/smart_gate_test.py` (candle-signal reconstruction on real trades), `scripts/gate_sweep.py` (threshold +
combo sweep, windowed), `scripts/call_filter_6mo.py` (harness call-trigger sweep, `--offset` for OOS window).
`backtest_gold_standard.py:CALL_DIRECTION_TRIGGER` (new harness call filter).

## Related memories
[[put-fix-canary-2026-07-20]], [[thetadata-greeks-required-for-backtest-2026-07-16]] (harness put fidelity gap:
generates ~0 puts under prod gates → test filters on REAL trades, not the harness put path).
