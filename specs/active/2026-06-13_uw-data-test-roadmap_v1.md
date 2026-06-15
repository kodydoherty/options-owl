# UW Data — Test Roadmap & Gold-Standard V7 Value Map (2026-06-13)

**Goal:** enumerate every test we can run with UW flow data (+ newly-downloaded tickers), and for
each, measure whether it ADDS VALUE to the V7 gold standard. "Adds value" = improves the V7 flow
backtest on **total P&L AND profit-factor AND max-DD**, holding across the per-month split (UW caps
history ~3mo, so split-half/per-month consistency is our OOS proxy until history extends).

**Baseline to beat (current V7 gold standard, flow side):** `uw_ticker_discovery` full window —
PUT whitelist META/AMZN/AAPL/TSLA (PF 3.86/4.07/3.22/1.22); CALL whitelist META/SPY/AMZN/TSLA/AAPL/AMD/PLTR
(META PF 2.37, SPY 1.31, AMZN 1.22; AAPL/PLTR weak). Harness: `scripts/uw_ticker_discovery.py`,
`scripts/uw_whitelist_ab.py` (per-month A/B). Risk gates + V7 exits unchanged.

**Method for EVERY test:** run vs the baseline above, report Δ(P&L, PF, DD) + per-month consistency.
Only promote a change if it wins on PF AND doesn't blow up DD AND holds across sub-periods.

---

## TRACK A — New tickers (data downloading NOW: MU, SMH, MRVL, TSM, INTC, ORCL, QCOM)
Download: `scripts/download_new_flow_tickers.sh` (OHLC-only, 2026-03-01..06-12 → thetadata_options.db).

- [x] **A1. Flow discovery on the 7 new tickers** — DONE 2026-06-13. Winners: **MU put** (PF 1.12,
      75% mo, n=137), **ORCL call** (PF 1.95, 100% mo!), **INTC call** (PF 1.30, 67% mo). MU CALL is a
      LOSER (PF 0.76, -619%, n=274) despite top flow — MU is a PUT name. Reject MRVL/TSM/QCOM/SMH.
      (Data gotcha fixed: --ohlc-only skipped greeks→no stock_ohlc; backfill_stock_ohlc_polygon.py added.)
- [ ] **A2. Promote winners** — ADD ORCL+INTC → `UW_FLOW_CALL_TICKERS`; ADD MU → `UW_FLOW_PUT_TICKERS`.
      (Hold ORCL put: PF 7.45 but 33% mo / n=13 = one lucky month.)
- [ ] **A3. Gold-standard delta** — rerun the full flow backtest WITH the new tickers in the whitelist vs
      WITHOUT. Does portfolio P&L/PF improve at acceptable DD? This is the "adds value to V7" gate.
- [ ] **A4. Decide deploy** — if A3 wins, update settings whitelists + `rebuild.sh` (paper bots validate first).
- [ ] **A5. (later) Full-history + greeks download** for any new winner, to enable per-ticker pattern/runner ML
      (only worth it if the flow edge is large — flow bypasses ML, so low priority).

## TRACK B — New strategies on existing UW data
- [x] **B1. Flow clustering** ✅ DONE 2026-06-13 (`scripts/uw_flow_clustering_test.py`, 30min window).
      Single sweep = LOSER (PF 0.80, mean -2.6%); clustered ≥2 = edge (PF 1.13), scaling: size2 1.09,
      size4 1.42, size6 1.69, size8 2.20, size9 2.72. → SIZE UP by cluster count, size DOWN/skip singles.
      (Caveat: ran on full 21-ticker universe incl losers, so single-PF is pessimistic; trend is the signal.)
- [ ] **B2. Aggregate flow as a market-direction gate** — pull market-wide net put/call whale premium per day
      (UW market-tide / net-flow endpoint); gate ALL trades to the institutional bias. Test: does adding the
      gate improve PF on both call and put books vs ungated? Risk: fewer trades.
- [ ] **B3. Dark-pool prints as S/R** — pull UW darkpool endpoint; use large block levels as entry-pullback /
      exit-target references. Test as an exit-tightening or entry-timing overlay vs V7 baseline.
- [ ] **B4. Intraday spot-GEX / gamma-flip levels** — daily GEX was REFUTED; retest at INTRADAY granularity as
      0DTE support/resistance (pin vs breakout). Build a small intraday-GEX probe first; only backtest if the
      level has visible predictive structure.
- [ ] **B5. Repeat-hitter / smart-money** — tag flow by entity persistence (recurring profitable chains); weight
      signals by historical hit-rate. Higher effort; needs entity tracking over time.
- [x] **B6/C2/C3. Conviction tiering** ✅ DONE 2026-06-13 (`uw_conviction_tiering_test.py`, whitelist).
      Premium: $1M+ PF **2.56** (vs $250-500k 1.59, $500k-1M 0.88) → size up $1M+. ask_frac: 0.85+ PF
      **1.62** vs 0.60-0.85 PF 1.03 → size up high-ask / consider raising the 0.60 floor. Both = size levers.

## TRACK E — Bet BIGGER on validated runners ⭐ (the V7 size-up thesis)
**Hard prior:** ML-confidence/score sizing was REFUTED ([[entry-filter-refuted-2026-06-11]]) — scores
don't predict outcomes, flat 0.85 was adopted. So the rule is **validate-then-size**: prove a signal
predicts runners BEFORE sizing up on it, else we just amplify variance/DD. Mechanism for any winner:
a conviction multiplier in `vinny_strategy.score_to_contracts`, CAPPED by MAX_POSITION_PCT.
- [ ] **E0. Validation harness** — bucket trades by each candidate signal; measure runner-rate (≥100%),
      P90 return, and PF per bucket. A signal "earns size" only if top bucket >> bottom on runner-rate AND PF.
- [ ] **E1. Runner model serve-path (Stage D)** — `runner_v1.lgb` EXISTS but was never wired to serve.
      First just VALIDATE: does its OOS P(runner) actually separate runners? (`runner_prediction.py` wrote
      OOS preds.) If AUC holds + high-P(runner) trades run more → wire P(runner)→size tilt, backtest vs flat.
- [ ] **E2. Flow clustering → size** (ties to B1) — do ≥2-sweep names run more? If yes, size up clustered flow.
- [ ] **E3. Sweep conviction → size** (ties to B6) — do $1M+ / high-ask-frac sweeps run more? Tier size by it.
- [ ] **E4. Cleanup** — remove the backwards 60% conf tier in `vinny_strategy.py` (flagged in entry-filter-refuted).
- [ ] **E5. Deploy gate** — only after a tilt beats flat on PF AND keeps DD acceptable AND holds per-month;
      roll out small bot (vinny) → kody. Anti-martingale stays (add to winners, never to losers).

## TRACK C — Tuning the existing flow config (cheap, existing data)
- [ ] **C1. Trim weak call names** — drop AAPL/PLTR (PF<0.85) from CALL whitelist; measure gold-standard delta.
- [ ] **C2. ask_frac sweep** — 0.55 / 0.60 / 0.65 / 0.70: tighter conviction vs volume. Per-month consistent?
- [ ] **C3. min-premium sweep** — 250k / 500k / 1M (ties to B6).
- [ ] **C4. Entry-timing offset** — enter at the flagged 5m bar vs wait 1 bar for confirmation/dip (mirror the
      dip-confirm logic). Does waiting improve fill/PF?
- [ ] **C5. PUT vs CALL budget split** — current PUT_BUDGET_MULTIPLIER=0.50; re-fit now that the call whitelist changed.

## TRACK D — Validation & integrity (don't fool ourselves)
- [ ] **D1. True OOS** — re-run the winning config when UW history extends past the current ~3mo window.
- [ ] **D2. Optimism-bias note** — the discovery window OVERLAPS V7 exit tuning; keep flagging until a clean OOS exists.
- [ ] **D3. Live paper-vs-live reconciliation** — `scripts/monday_flow_check.sh` weekly: do live (kody/dennis)
      flow trades match paper (adam/vinny/yank) on the same signals? Divergence = gate/fill bug.
- [ ] **D4. Full-flow e2e regression** — `scripts/e2e_flow_test.py` after any pipeline change (gates + cleanup).

---

## Value-map summary (what likely moves the V7 gold standard, best guess pre-test)
| Change | Expected value | Effort | Confidence |
|---|---|---|---|
| A: add MU (if it backtests like its flow rank) | HIGH (top-3 flow both sides) | med (download running) | med — MUST verify |
| B1: flow clustering | med-high (better entries) | low | med |
| C1: trim AAPL/PLTR calls | low-med (cuts losers) | trivial | high |
| B6/C2/C3: size/conviction tiering | med | low | med |
| B2: aggregate-flow direction gate | med (helps both books) | med | low-med |
| A: other semis (SMH/MRVL/TSM/INTC/ORCL/QCOM) | unknown — backtest first | med | low |
| B3/B4/B5: darkpool / intraday-GEX / smart-money | unknown — exploratory | high | low |

**Order of execution:** A1→A3 (new tickers, data landing now) → C1 (free win) → B1 (clustering) →
B6/C2/C3 (tiering) → B2 (direction gate) → exploratory B3/B4/B5. Each promotes only on a gold-standard
win + per-month consistency; deploy via rebuild.sh, paper bots first.
