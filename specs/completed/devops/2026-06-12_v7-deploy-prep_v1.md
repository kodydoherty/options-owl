# V7 Convex Redesign — Deployment Prep

**Guiding rule (Kody, 2026-06-11, still in force):** fix entry quality / recover win
rate BEFORE up-sizing. V7 bundles entry+sizing+exit; the sizing-up pieces (P(runner)
tilt, cap loosening) deploy LAST and only if win rate holds.
Rollback baseline frozen at `backups/v6-gold-standard-2026-06-12/`.

## ROLLOUT STATUS (updated 2026-06-12)

| Stage | Change | What ships | Status | Validation |
|---|---|---|---|---|
| **Pre** | **Regime 1m-candle fix** | collector writes 1m → PG; regime model revives | ✅ **LIVE** (all owlets, Mon open) | data-bug fix, see [[regime-1m-candle-bug-2026-06-12]] |
| **A** | **V7 wide-trail exits** (`ENABLE_V7_WIDE_TRAIL`) | no ceiling, widened trail, scaleout/2PM off, ratchet kept; CALLs theta 60, **PUTs keep no-limit** | ✅ **LIVE on kody + dennis** | exit-only ablation: BOTH OOS windows beat V6 at identical WR (+72% / +27% P&L, higher PF) |
| **B** | **Thin entry** (`ENABLE_V7_THIN_ENTRY`) | drop premium-cap + score-floor; keep spread/EOD/pos-cap/loss-CB/GFV | ⬜ **NOT BUILT** — port `risk/pipeline.py`, test, watch live WR first | full-V7 3-window win bundles this; needs isolation like exits |
| **C** | **Anti-martingale add** (`ENABLE_V7_ANTIMG`) | one capped add to a confirmed winner (+30% prem, und-confirm); replaces V6 DCA | ⬜ **NOT BUILT** — `position_monitor.py` + exit_v5; monitor-loop integration test mandatory | unvalidated in isolation |
| **D** | **P(runner)-tilt sizing + cap=4** (`ENABLE_V7_RUNNER_TILT`) | top-decile 1.75x / mid 0.85 / bottom 0.5x; loosen multi-day cap 2→4; remove the inverted 60% conf tier | ⬜ **BLOCKED** — needs serve-time runner model wiring (§2); `runner_v1.lgb` now exists but features not wired into entry path. **Up-size = deploy LAST per the rule** | full-V7 OOS: +27-72% P&L but absolute $ inflated (shared-model leak); WR dips 3-4pts |

**Next concrete steps:** (1) wire serve-time P(runner) features (§2b/2c) so Stage D is even possible; (2) isolate-ablation Stage B (thin entry) like we did exits; (3) build Stage C monitor-loop test. Stages B/C/D each need their own validation before flag-flip — do NOT bundle.

---

## 1. Change inventory — V7 convex vs V6 production

| # | Change | Production file to edit | New flag | Validation status |
|---|---|---|---|---|
| 1 | **Thin entry**: drop premium cap + score floor; keep spread/EOD/pos-cap/loss-CB/GFV; anti_chase OFF, momentum ON | `risk/pipeline.py` (premium-cap gate, score-floor gate) | `ENABLE_V7_THIN_ENTRY` | anti_chase-off already live & validated; premium-cap already `false` in prod |
| 2 | **Remove inverted 60% conf tier** → flat budget | `risk/vinny_strategy.py:562-566` (`_CONFIDENCE_TIERS`, the `(0.80, 0.60)` row) | folds into #3 | OOS-ROBUST (bug removal) — safe even standalone |
| 3 | **P(runner)-tilted sizing**: ≤0.2438→0.5×, mid→0.85×, ≥0.5981→1.75×; hard ceiling = MAX_POSITION_PCT | `risk/vinny_strategy.py` `score_to_contracts` / `_ml_confidence_to_mult` | `ENABLE_V7_RUNNER_TILT` | **BLOCKED** — needs serve-time runner model (§2) |
| 4 | **Loosen multi-day contract cap** 2 → 4 | `risk/vinny_strategy.py` (cap logic) | `ENABLE_V7_RUNNER_TILT` (same) | sizing-up → deploy LAST |
| 5 | **Anti-martingale add** (replaces V6 DCA): +30% premium, 3–60min, underlying ≥+0.10%, add 1× orig, respect 15% cap, whole stack one trail | `execution/position_monitor.py` + `risk/exit_v5/` (add path) | `ENABLE_V7_ANTIMG` (and set `ENABLE_V6_DCA=false`) | needs its own monitor-loop integration test |
| 6 | **Exit: widening adaptive trail** (active ×1.1, runner ×1.3, moonshot ×1.5), no scaleout, no CALL profit ceiling, KEEP breakeven ratchet + fast stall-stop | `risk/exit_v5/config.py` (trail tiers, scaleout, profit_target) | `ENABLE_V7_WIDE_TRAIL` (and `ENABLE_V6_SCALEOUT=false`) | lower risk — doesn't raise entry $; deploy after own OOS recheck |

**Exact V7 constants** (source: `scripts/backtest_gold_standard_v7.py:105-203`):
```
TILT_BOTTOM_P=0.243809  TILT_TOP_P=0.598054  DOWN=0.50 FLAT=0.85 UP=1.75
ANTIMG: trigger +30%, window 3–60min, und_confirm +0.10%, add 1.0×, cap 15%
TRAIL widen: active×1.1, runner×1.3, moonshot×1.5
```

---

## 2. HARD BLOCKER — no serve-time P(runner) model exists

`scripts/runner_prediction.py` is **research-only**: it does an expanding walk-forward
and writes `runner_oos_predictions.csv` + `runner_samples.csv`. **It never saves a
deployable model.** The v7 backtest reads P(runner) from that precomputed CSV — a path
that does not exist live.

To deploy changes #3/#4 we must build:
- **2a. Trainer** → train a runner model on ALL available data and `save_model()` to
  `journal/models/ml_v3/runner_v1.lgb` (+ `_meta.json` with feature list + the decile
  cut points `TILT_BOTTOM_P`/`TILT_TOP_P`). Reuse `runner_prediction.py`'s feature
  builder so live features == training features.
- **2b. Serve-time features** → compute the runner feature vector at entry inside the
  entry path (same features: moneyness, spread_pct, und_move_pct, und_slope_5, prior-day
  range, etc. — all known at entry). Must exactly match the trainer.
- **2c. Wire into sizing** → `score_to_contracts` looks up P(runner), applies the tilt
  multiplier, behind `ENABLE_V7_RUNNER_TILT`.

Until 2a–2c exist, V7 can deploy only its **entry + exit** pieces (#1, #5, #6) — which is
fine and aligns with the "entry-quality first, size-up last" rule. The sizing tilt is a
separate, later milestone.

---

## 3. Feature flags (add to `config/settings.py`, default `False`)
```
ENABLE_V7_THIN_ENTRY     = False   # drop premium-cap + score-floor gates
ENABLE_V7_WIDE_TRAIL     = False   # widening adaptive trail, no scaleout/ceiling
ENABLE_V7_ANTIMG         = False   # anti-martingale add (turn OFF ENABLE_V6_DCA)
ENABLE_V7_RUNNER_TILT    = False   # P(runner) sizing + cap=4 (needs §2)
```
Mutually-exclusive guards: V7 wide-trail implies `ENABLE_V6_SCALEOUT=false`; V7 antimg
implies `ENABLE_V6_DCA=false`. Add an assertion at startup so both aren't on at once.

---

## 4. Staged rollout (entry-quality first; size-up last)
1. **Phase A — exits** (#6): `ENABLE_V7_WIDE_TRAIL=true` on owlet-kody only, paper-mirror
   1 day → live. Lowest risk, no entry-$ increase.
2. **Phase B — thin entry** (#1): add once win rate confirmed holding.
3. **Phase C — anti-martingale** (#5): replace DCA; monitor-loop integration test first.
4. **Phase D — runner sizing** (#3/#4): only after §2 built AND win rate recovered. This
   is the up-size step — deploy to one small bot (vinny/$500) before kody.

---

## 5. Test plan (MANDATORY before any flag flips)
- `pytest tests/ -q` fully green (currently 2508 pass).
- **New monitor-loop integration tests** for anti-martingale add path (per CLAUDE.md:
  position_monitor changes are HIGH RISK — the sell path). Add to
  `tests/test_monitor_integration.py`.
- **Source-safety tests** (`TestSourceCodeSafety`) for any new conditional-only vars.
- New unit tests: tilt multiplier boundaries (P=0.2438/0.5981 edges), thin-entry gate
  bypass keeps risk gates, wide-trail tier widths, antimg trigger/confirm/cap.
- Re-run gold-standard backtest (V6 baseline) to confirm no regression in the shared harness.
- `ruff check options_owl/ tests/` clean.

## 6. Rollback
Code → `git checkout f9497c2 -- options_owl/risk/<f>` or copy from
`backups/v6-gold-standard-2026-06-12/`. Models → restore
`journal/models/ml_v3_backup_20260611_124224/`. Flags → all `ENABLE_V7_*=false`,
`docker compose up -d` (NOT restart).

## 7. Pre-deploy checklist
- [ ] v7 retrain+backtest done; V7 beats V6 risk-adjusted on **oos AND oos2**
- [ ] Serve-time runner model built + tested (§2) — for sizing pieces only
- [ ] Flags added, default false, mutual-exclusion assertion in place
- [ ] All tests green incl. new monitor-loop integration tests
- [ ] Staged per §4; verified live container env after each (`docker exec ... env`)
- [ ] V6 backup confirmed intact for rollback
