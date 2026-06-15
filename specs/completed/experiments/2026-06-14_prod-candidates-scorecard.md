# Prod-Candidates Scorecard — what we built/found (session 2026-06-13→14)

Status legend: ✅ LIVE in prod | 🟡 deployed to canary (paper) | 🔬 validated, NOT deployed | ⏳ needs work

| # | Change | Validation | Status | Prod recommendation |
|---|---|---|---|---|
| 1 | **V7 gate 0.62 / entry 0.80** | multi-value sweep, 3 OOS windows, consistent (PF 3.8, 1.8× volume) | ✅ LIVE all bots | keep; watch DD ~10% live |
| 2 | **UW flow signals (calls+puts)** | 90-day flow backtest, WS e2e verified | ✅ LIVE all bots (paper on adam/vinny/yank) | keep; Monday paper-vs-live check |
| 3 | **New tickers: MU(put), ORCL/INTC(call)** | discovery: ORCL call PF1.95/100%mo, INTC 1.30, MU put 1.12/75%mo | ✅ LIVE | keep |
| 4 | **C1 trim AAPL/PLTR calls** | both PF<0.85, 33% mo (losers) | ✅ LIVE | keep |
| 5 | **Stage D conviction sizing** | +72% on equal capital, combined PF 1.36→1.75, lower DD; SPY puts +$15.4k gated | 🟡 vinny canary (paper) | **enable on kody after canary** (highest-value lever) |
| 6 | **New-tier tickers: ARM/GOOG/LRCX (calls)** | discovery: ARM PF5.08/100%mo, GOOG 2.53/67%mo, LRCX 1.41/67%mo; e2e added +$5.2k flow leg | ✅ LIVE (deployed 2026-06-14) | keep |
| 7 | **Per-ticker ML (17 new tickers)** | EVAL DONE 2026-06-14: signal models GOOG 0.85 / ORCL 0.77 / MU 0.71 / INTC 0.70 beat generic 0.68; SMH/GLD/SLV ~eq; TSM 0.20 degenerate; 9/17 didn't train (thin data). These are ML-SOURCING-scan models; new tickers are FLOW-ONLY (not scanned) → not on live path. | 🔬 eval done, NOT wired | DON'T wire on thin ~3mo data; revisit w/ more data. Wiring = add winners to ML scan (separate decision). |
| 8 | **SPY puts (gated)** | breakeven naive (PF1.02) → heavy-cluster PF1.48; rides Stage D (excl $1M hedges) | 🟡 via Stage D | enable with Stage D on kody |
| 9 | **Serve-time P(runner)** | BUILT 2026-06-14, flag-gated OFF (`ENABLE_V7_RUNNER_TILT`). `risk/flow_runner.py` fetches Polygon greeks-snapshot + option 1m bars + underlying 1m bars → `compute_option_features_from_live` (SAME source of truth as training, no skew) → `predict_entry_confidence` runner_score → folds into conviction multiplier. Returns None on ANY missing data (greeks required) = safe no-op. 3 unit tests. | 🟡 built, OFF (pending live validation) | VALIDATE: enable on a PAPER bot Monday, watch FLOW_P_RUNNER logs (iv/delta sane, p_runner well-distributed not 0/1) to confirm serve≈training; then enable on kody. iv_history degraded (3 of 41 feats default to 0) — refine if validation shows skew. |
| 10 | **Liquidity cap (MAX_POSITION_DOLLARS=$50k)** | combined compounding realistic only with ~$25-50k/trade cap; no-op for current accounts | ✅ LIVE (deployed 2026-06-14, default $50k) | keep; revisit value as accounts grow |

## FINAL 60-day e2e (2026-03-16→06-12, all changes deployed)
- **Fixed-sleeve (edge):** V7 ML +$21.2k + flow conviction +$92.0k = **COMBINED +$113.2k, PF 1.78, WR 58%, maxDD −$4.6k** (1803 trades). New tickers added +$5.2k to the flow leg.
- **Compounding off $20k (current 0.62 config):** V7 ML = **$234,746** (PF 4.13, the "$200k" confirmed).
- **Compounding + flow, liquidity-capped:** $50k cap → +$3.25M (ML $966k / flow $2.29M, maxDD −$204k); $25k cap → +$1.79M. Uncapped → $35M (fantasy, unfillable). Flow ≈ 2.3× ML at every cap.

## Key reconciliation (don't lose this)
- At **fixed $750/trade**: V7 ML = $21.2k, flow (conviction) = $86.8k, **combined = $108k, PF 1.75** (63 days). This isolates EDGE.
- The "**$200k+**" memory = V7 ML **compounding** off $20k = $234.7k (aggressive %-compounding, not liquidity-capped).
- **Flow edge ≈ 4× ML edge at equal sizing** — flow is the stronger signal; ML's big number is mostly compounding leverage.
- Compounding + flow is realistic ONLY with a per-trade liquidity cap (~$25-50k for liquid 0DTE).

## UW-strat research (2026-06-14 night)
- **B2 — market-tide gate — BUILT, flag-gated OFF (`ENABLE_V7_TIDE_GATE`):** ⚠️ the first test (PF 2.14) had
  LOOKAHEAD (EOD tide). Point-in-time re-test (`uw_b2_pointintime_test.py`, tide AS OF entry minute): aligned PF
  1.27 / misaligned 0.73 — real but weaker. **PUTS-ONLY edge:** put aligned PF **1.66** (+$43k) vs misaligned **0.56**
  (−$39k, net loser); CALLS ~no edge (1.03 vs 0.91). Wired: `flow_runner.get_market_tide_bias` (one live call) →
  `flow_conviction_mult` sizes a PUT-against-bullish-tide ×0.30 (`V7_TIDE_MISALIGNED_PUT_MULT`); calls untouched.
  4 unit tests. DEFAULT OFF — validate on a paper bot Monday (tide bias sane, misaligned puts sized down) then enable.
- **B3 — darkpool S/R:** UW darkpool is RECENT-ONLY (no history) → can't backtest. Deployed `owlet-darkpool-shadow`
  (forward-collector → journal/darkpool/darkpool.db, ~8k prints/poll, 17 tickers). Backtest in a few weeks of data.
- **B4 — intraday GEX/gamma-flip:** `/stock/{tkr}/spot-exposures` HAS history (backtestable) — not yet tested; queued.
- **B5 — smart-money/repeat-hitter:** NOT feasible — flow-alerts carry NO trader/entity ID. Reframe to alert-quality
  (all_opening_trades / has_singleleg vs has_multileg / rule_id) as conviction proxies — testable on existing data.

## Next actions (ordered)
1. Let the fresh full v7 compounding backtest finish (running) → confirm ~$234k with current 0.62 config; re-run combined.
2. Add ARM/GOOG/LRCX to call whitelist (#6) — validate gold-standard delta, deploy.
3. Enable Stage D conviction sizing on kody (#5) after vinny canary behaves (Monday flow).
4. Eval per-ticker ML (#7) — AUC vs generic; wire winners.
5. Build serve-time P(runner) feature vector (#9) — activates the strongest size lever live.
6. Monday: `monday_flow_check.sh` paper-vs-live reconciliation (#2).
7. Rotate UW API key (passed through chat).
