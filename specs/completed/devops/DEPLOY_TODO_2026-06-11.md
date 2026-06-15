# Deployment TODO — ordered so we FIX LOSING before we UP-SIZE

**Guiding rule (Kody, 2026-06-11):** every trade but one lost money today. Do **not** deploy any
change that increases $/trade until entry quality is fixed and win rate recovers. Bigger size on
bad entries = bigger losses. Sequence is entry-quality FIRST, sizing-up LAST.

Validation bar for anything below: must beat current config on **risk-adjusted return across
multiple historical windows (OOS)**, not just one. Deploy via `rebuild.sh` (or `.env` + staggered
`up -d` for config), verify live container env, keep `.env.bak` for rollback.

---

## ✅ PHASE 0 — LIVE NOW (stop-the-bleeding, already deployed)
- [x] **momentum_confirm ON + anti_chase OFF** — best-of-both gate config. Blocks counter-trend
      faders (today's losers), keeps the +$38K runner edge. (per-gate sweep: anti_chase off = the
      whole win; momentum off was ~neutral in backtest but net-negative live.)
- [x] **Entry-fill chase ladder** — buy-side re-price ladder (ask×1.05→1.15, double-fill-guarded).
      Stops "order not filled → cancelled" misses (the QQQ #253 + the +100% NVDA-put miss).
- [x] **GEX/charm/vanna capture + `gex_ticks`** + flow-5m timestamp fix (harvester).

## 🔴 PHASE 1 — ENTRY QUALITY (must clear before any sizing-up)
- [x] ~~**Reversal-confirmed tick entry**~~ — **REFUTED by real tick data (2026-06-11).** Counter-trend
      is NOT the fader cause: against-trend trades won MORE (66% vs 51%); half the faders faded
      *with* the micro-trend. Reversal-waiting missed winners / paid up. **Dropped.** (entry-quality
      spec marked accordingly.) Two entry-time fader filters now tried & failed: direction +
      confidence-scaling. Working hypothesis: faders can't be filtered at entry → lean on exits +
      sizing + runner-prediction instead.
- [ ] **Runner *prediction* feature-analysis** (needs backfill) — test whether *runners* (not faders)
      are predictable at entry. If yes → size UP on runner-likely setups (the asymmetric play). If
      no → accept faders as the cost; edge is purely exits + flat sizing + anti_chase-off.
- [ ] **Confirm win rate recovers** with momentum-on (watch live a few days) before up-sizing.
- [ ] **Fix regime-serving live-PG bugs** (morning-data window + `gex_ticks` DataError) — required
      before the improved regime model can deploy; currently fails-open + log-spams.
- [ ] **Investigate:** multi-day stops firing too tight (~−34% on 1-DTE; should be ~52% multi-day).
- [ ] **Investigate:** Webull entry no-fills (the ladder should fix; confirm live).

## 🟡 PHASE 2 — EXITS (lower risk; doesn't increase entry $; deploy after its own OOS recheck)
- [ ] **Exit-tuning candA** — grace 8min, scaleout +25%, soft-keep 0.5, adaptive-mult 0.8.
      OOS-validated: +18% P&L, PF 2.14→2.23, same trade count. Stage for deploy.

## 🟢 PHASE 3 — SIZING UP (LAST — only after Phase 1 win-rate recovers)
- [ ] **Remove the backwards 60% confidence tier → flat budget for all qualifying signals.**
      OOS-ROBUST (it's removing a bug, not adding an overfit edge): the 0.80–0.90 bucket is the
      model's *best* yet is sized smallest. This is why the conf-0.86 NVDA call was $442. Implement
      in `vinny_strategy.score_to_contracts`; test; deploy.
- [ ] **Loosen the multi-day contract cap** (the other reason calls are 2 contracts) — *result
      pending:* sizing-experiment OOS still computing. Adopt only the OOS-validated setting.
- [ ] **MAX_CONCURRENT / capital-utilization** test (per-slot = deployable/8 under-deploys the book).

## ❌ REJECTED (do NOT build)
- [x] **Confidence-*scaling* sizing** — looked great in-sample, **collapses OOS** (overfit). Only the
      flat 60%-tier removal (Phase 3) survives.

---
### Currently running (feeds the above)
- Multi-year ThetaData backfill (→ ~2.75 yr; at 2024-05) — enables runner analysis.
- Sizing-experiment OOS pass (multi-day-cap + scheme comparison).
- Tick-reversal entry validation (real `stock_ticks`).
