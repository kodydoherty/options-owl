---
title: "OptionsOwl — Live-Trading Fixes (2026-06-15, first live session)"
---

# OptionsOwl — V7 Live-Trading Fixes

**Date:** 2026-06-15 (first LIVE session of the full V7 + UW flow system) · **Severity:** High · **Status:** Fixed + deployed live

On the first live session, **zero trades were flowing** despite a healthy-looking system. Three independent bugs were
found mid-market and fixed. All three were deployed live (rebuild.sh, full test gate) the same session.

---

## Fix 1 — Score floor neutralized the 0.62 pattern gate (CRITICAL, ML side)

- **Symptom:** ML scan found candidates (e.g. `TSLA CALL pattern=0.684 score=68`), the 28-gate pipeline APPROVED them,
  then they were silently rejected — 51 signals generated, ~0 trades.
- **Root cause:** `score_to_contracts` had `_SCORE_FLOOR=75` (≈ pattern 0.75) and `_MIN_ML_CONFIDENCE=0.70`. The
  deployed `ML_PATTERN_THRESHOLD=0.62` let 0.62–0.75 signals through the entry gate, but the **sizing floor (75/0.70)
  rejected every one** — so the 0.62 change delivered *none* of its backtested volume.
- **Fix:** aligned `_SCORE_FLOOR = 62` and `_MIN_ML_CONFIDENCE = 0.62` (PUT floor stays 0.65 — put model threshold).
  `options_owl/risk/vinny_strategy.py`. Verified live: `FLOOR 62 / 0.62`; a score-65/conf-0.65 call now sizes 6 contracts.
- **Note:** the fix deployed ~11:35 ET, after the 90-min CALL scan window (closed ~11:00 ET), so the big payoff is the
  next open. PUT scan (5–360 min) benefits same day.

## Fix 2 — ML scan Polygon fallback used the wrong expiry (7 tickers blind)

- **Symptom:** `Polygon chain: JPM/MSTR/NFLX/PLTR/AMD/SMCI/BA exp=2026-06-15 → 0 contracts` — those names couldn't be
  scanned. (The 9 liquid 0DTE names scanned fine via Redis.)
- **Root cause:** `fetch_live_option_chain` queried only today/tomorrow. Names without a Monday 0DTE (they expire Friday)
  returned 0. The harvester captures the correct range; only the Polygon fallback was naive.
- **Fix:** bounded **near-expiry fallback** in `fetch_live_option_chain` — if the requested expiry has 0 contracts, walk
  forward over the next ~5 business days to the ticker's nearest available expiry. `options_owl/sourcing/ml_pipeline.py`.
  Verified live: `JPM → 86, PLTR → 116, MSTR → 440` contracts at 06-18; `polygon_fb=7` recovering.

## Fix 3 — UW flow collector dropped 100% of WS alerts (CRITICAL, flow side)

- **Symptom:** UW flow WS connected + streaming (19 msgs/22 s), but **zero flow signals ever** — in production or shadow.
- **Root cause:** the live **WebSocket message format has no `type` field** (nor `strike`/`expiry`) — they're encoded in
  the OCC `option_chain` symbol (`MSTR260618C00115000`). `evaluate_flow_alert` did
  `if opt_type not in ("put","call"): return None`, so **every alert was rejected.** The backtest used the *REST*
  endpoint (which HAS `type`), so this never surfaced until live.
- **Fix:** when `type` is absent, derive put/call + strike + expiry by parsing the OCC `option_chain` via
  `parse_occ_ticker` (prepending `O:`). `options_owl/collectors/uw_flow_collector.py`. Verified live: a WS-format META
  call alert now **FIRES** (CALL, strike 560) — previously dropped. 32 flow tests green (+2 regression tests).

---

## Validation & deploy

- Full suite (~2,545 tests) green on each deploy; ruff clean. All three shipped via `scripts/rebuild.sh` (test-gated),
  flags unchanged, staggered restart. No open positions during restarts.
- Tests updated to the new floors (`test_vinny_strategy`, `test_risk_gates`, `test_ml_strategy_e2e`,
  `test_code_review_fixes`, `test_conviction_sizing`) + WS-format regression tests added.

## Lessons

- **Backtest/serve format skew:** the flow backtest used the REST schema; the live feed is the WS schema (different field
  names). Always validate the serve path against the *live* data shape, not just the backtest source.
- **Config consistency:** lowering one gate (pattern 0.62) without aligning the downstream sizing floor (75) silently
  neutralized it. Thresholds that gate the same thing must move together.
- **Per-ticker expiry everywhere:** smart-entry already had per-ticker expiry; the scan's fallback did not.

## Remaining (follow-ups)

- Fix the **flow-shadow** (`uw_flow_shadow.py`) — same `type`/OCC bug; observe-only, so non-blocking.
- Move **darkpool collection from REST → WS** and persist all UW WS data into the harvester DB (architecture request).
- Validate the now-flowing flow trades reach Webull **live** (not paper) — trade #265 (pre-fix) went paper.
