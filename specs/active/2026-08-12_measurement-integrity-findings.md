# Measurement integrity: what is real, what was refuted (2026-08-12)

Status: ACTIVE
Category: experiments / infrastructure

Read this before re-opening runner_v1 skew, the pattern threshold, or the execution gap.
Several confident-sounding results in this area were WRONG, and the reasons they were
wrong are reusable.

## 1. runner_v1 train/serve skew: REFUTED (do not re-derive)

**Claimed:** the live serving path emits p_runner in a narrow 0.684 to 0.828 band while a
faithful Postgres rebuild spans 0.054 to 0.940, so the live feature vector is broken, the
Q1 floor can never fire, and the tiering is inert.

**Reality:** rebuilding the SAME trades that carry a live score gives mean divergence
-0.017, max |delta| 0.095. The live path and the rebuild agree. There is no skew.

**The error:** 13 live values were compared against a rebuild over a DIFFERENT ~100-day
trade population. Two populations, not two code paths. The control that settles it is
"rebuild the same trades", and it had not been run.

**A second error inside the first:** the diff tool keyed live scores by trade id alone.
The bots keep separate sqlite databases whose ids both start at 1, so kody's #416
overwrote dennis's and every trade was scored against the other bot's value. It showed up
as a dramatic META-only divergence that vanished once keyed by (bot, id). Any join across
bot databases MUST be keyed by (bot, id).

Tool: `scripts/diff_runner_serve_vs_rebuild.py` (now correctly keyed).

## 2. The Q1 floor was never inert, but was untested

Rebuilt p_runner per August trade: every sub-0.39 case is on 08-04 and 08-06, and nothing
below 0.61 appears after the gate deployed on 08-07. Pre-gate they were ~16% of trades.
The gate has not fired because the opportunity has not recurred.

What WAS wrong: a gate that skips entries on the live-money path had never executed in
production and had no test, which is indistinguishable from broken until it matters.
Covered now by `tests/test_runner_v1_floor_gate.py` (4 tests through the real
`evaluate_and_trade`), including that a None score must NOT count as sub-floor: the
scorer abstains often, and treating abstention as sub-floor would halt trading on any
Redis or candle hiccup.

## 3. The pattern threshold is not the volume lever

Sweeping `--pattern-threshold` at 0.62 / 0.56 / 0.50 returns byte-identical results
(1127 trades, +$147,071). Two gates sit underneath it:

```python
if pattern_conf < pattern_threshold: continue
score = int(pattern_conf * 100)
if score < MIN_SCORE: continue          # MIN_SCORE=60 -> hard floor conf >= 0.60
...
if entry_conf < entry_threshold: continue   # default 0.80 — binding IN PROD
```

`MIN_SCORE=60` makes any threshold below 0.60 unreachable (`score = int(conf*100)`), which
is what caps the sweep. NOTE: those runs had the entry filter DISABLED (see 3b), so the
identical 0.62/0.56/0.50 totals are explained by MIN_SCORE plus the empirical fact that no
candidate lands in conf [0.60, 0.62) — not by the entry model. In PROD the entry-timing
gate at 0.80 is live and is the binding constraint on volume.

**The binding gate was never calibrated.** Every model publishes a `best_threshold`
(pattern 0.80, put-entry 0.85, expansion 0.75) EXCEPT `entry_timing_meta.json`, which
carries only an AUC. Prod uses `DEFAULT_ENTRY_THRESHOLD = 0.80` and the meta-override path
exists only for puts. Prod and the harness both fall back to the same 0.80 default, so the
VALUE is not a drift; it has simply never been earned. (Whether the filter runs at all was
a drift — see 3b.)

**Sweep protocol:** hold `--pattern-threshold` explicitly. It defaults to 0.74, so
omitting it while varying the entry threshold moves two variables at once. That produced a
run where "loosening" a gate LOST 621 trades, which is impossible and is the tell.

## 3b. HARNESS DRIFT: every earlier sweep ran with the entry filter OFF

`--no-entry-filter` was passed to all of `thr_0.70/0.62/0.56/0.50`, `dirreg_off` and
`dirreg_mom_off`. Production **does** load and apply the entry-timing model
(`ML_PIPELINE: Loaded entry_timing (30 features)`, threshold 0.80), so those runs were
NOT prod-faithful: they modelled a book roughly twice the size prod actually trades.

Consequences:

- **The $58 / $65 dirreg decision rule below is INVALID** and must be re-derived with the
  filter ON. Re-runs: `faith_base` + `faith_dirreg_off`.
- The explanation that "entry_timing at 0.80 is the gate that makes 0.62/0.56/0.50
  identical" is wrong FOR THOSE RUNS, since the gate was disabled in them. With the filter
  off the binding floor is `MIN_SCORE=60`, and the empirical fact is simply that no
  candidate lands in conf [0.60, 0.62). entry_timing IS the binding gate in prod.
- The first entry-threshold sweep compared a filter-OFF baseline against filter-ON
  variants, which is why "loosening" appeared to lose trades a second time.

**How it was caught:** the variants were internally consistent (0.70 -> 875, 0.60 -> 988,
0.50 -> 1060, correct direction) while the baseline sat above all of them at 1127. When a
series is monotonic but the baseline does not fit it, suspect the baseline, then diff the
run headers. `diff <(head -30 a.log) <(head -30 b.log)` showed `Entry filter: OFF` vs `ON`
immediately.

**Protocol:** diff the run headers of baseline vs variant before computing any marginal.
Config drift between runs is invisible in the totals and fatal to the comparison.

## 4. The execution gap ($84/trade) and what it gates

The gap is hold-time divergence, not fill slippage: a stale price cannot trip a trailing
gate, so trades rode into loss gates. Post-fix, hold time went 16.5 -> 4.5 min on kody and
17.2 -> 4.5 min on dennis, and `profit_lock` / `scalp_target` appear in the exit mix again.
Two bots landing on 4.5 min independently is not noise.

The DOLLAR value is not yet measurable (n=13, and kody's +$911 is +$809 from one trade).
Decision rule, from quarterly marginals:

| change | Q1 | Q2 | Q3 | Q4 | deploy if bar |
|---|---|---|---|---|---|
| dirreg off | 58 | 98 | 76 | 136 | < $58 |
| dirreg + momentum off | 65 | 91 | 117 | 108 | < $65 |

**These numbers are SUPERSEDED** (see 3b: computed with the entry filter off). The method
stands; the values must be re-derived from `faith_base` / `faith_dirreg_off`.

At $84 they are 2/4 and 3/4. Re-measure with `option_ticks` (covers the post-fix window;
thetadata core tickers stall at 2026-07-15 and a re-pull returned zero rows).

## 5. Standing rule that keeps catching things

A change that BUYS trades must pay the execution cost **on the trades it buys**, per
quarter, not on the book average. Pooled numbers hid this for dirreg ($91.9/trade pooled,
only 2/4 quarters above the bar). Same arithmetic refuted afternoon trading ($53.70) and
puts ($22).

## Open

- Entry-threshold sweep at 0.70 / 0.60 / 0.50 with pattern pinned at 0.62 (running).
- Concurrency-aware sizing backtest (`--concurrency-slots`, code committed and disabled).
- Re-measure the execution gap in ~2 weeks (~4.3 trades/session, so n=40 is ~9 sessions).
