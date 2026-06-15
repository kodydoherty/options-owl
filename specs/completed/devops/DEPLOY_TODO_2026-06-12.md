# Deploy TODO — Full V7 + UW Flow (2026-06-12)

**Guiding rules (still in force):**
- Entry-quality first, **size-up LAST** (Stage D deploys after win-rate holds).
- Every stage gets its **own isolated backtest validation** before the flag flips — do NOT bundle.
- New/risky paths roll out **shadow → paper (1 bot) → live (kody)**. All flag-gated, reversible.
- Rollback baseline: `backups/v6-gold-standard-2026-06-12/`.

---

## ✅ DONE (live in prod)
- [x] **1m-candle regime fix** — collector writes 1m → PG; regime model revives. (all owlets)
- [x] **V7 wide-trail exits (CALL+PUT)** — `ENABLE_V7_WIDE_TRAIL=true` on **kody + dennis**.
      Validated: both OOS windows beat V6 at identical WR. PUTs keep no-hold-limit.
- [x] **UW flow shadow collector** — `owlet-flow-shadow` live, log-only, collecting.

---

## TRACK 0 — Gate optimization (the volume fix) ⭐ NEW, highest priority
**Finding (2026-06-12 sweep):** the **pattern ML model at 0.74 blocks 95% of candidates** —
the real reason we only traded 1.6/day. It's the volume killer, not the risk gates. Sweep
across 3 windows → **pattern 0.62 / entry-timing 0.80 is the sweet spot**: ~1.8× volume
(99→180 trades/60d), PF still ~4, but DD rises 7%→12% (more aggressive). 0.50 over-loosens
(PF collapses to ~1.7); entry-timing OFF spikes variance — keep it at 0.80.
- [x] **Multi-value sweep DONE (2026-06-13)** — pattern {0.68,0.62,0.56} × entry {0.90,0.80,0.70}
      across 3 non-overlapping OOS windows. Ranking held in every window (robust). Locked
      **pattern 0.62 / entry 0.80**: avg PF 3.8, avg DD 10, ~3.1 trades/day (≈1.8× the old 1.7/day).
      `0.62/0.70` = more volume but PF→3.1, DD→12.4, higher variance (rejected). `0.56/*` PF collapses.
- [x] **Config set in docker-compose.yml** — `ML_PATTERN_THRESHOLD=0.62` on kody + dennis
      (entry already 0.80 by default). NOT yet deployed — awaiting rebuild.
- [ ] **Deploy** via `rebuild.sh` — RECOMMEND dennis-only canary first (smaller $10k), watch live
      WR/DD a few sessions, then kody. Rollback = remove the env line + `up -d`.
- [ ] **0.68/0.80 fallback** (PF 5.3, DD 6.5, 2.3/day) if live DD at 0.62 runs hotter than the 10% backtest.

## TRACK 1 — Finish the V7 rollout (stages B/C/D)

### Stage B — Thin entry  (`ENABLE_V7_THIN_ENTRY`)
- [ ] Isolated ablation: V6 entry+exit + **only** the thin-entry change (drop premium-cap +
      score-floor; KEEP spread/EOD/pos-cap/loss-CB/GFV) vs baseline, both OOS windows.
      Reuse the `--v7-exits`-style flag pattern in `backtest_gold_standard.py`.
- [ ] If it wins risk-adjusted → port into `risk/pipeline.py` behind `ENABLE_V7_THIN_ENTRY`
      (default off) + unit tests (gate bypass still keeps the risk gates).
- [ ] `rebuild.sh`, enable on **kody only**, watch live WR a few days before dennis.

### Stage C — Anti-martingale add  (`ENABLE_V7_ANTIMG`, replaces V6 DCA)
- [ ] Backtest the add rule (one capped add to a confirmed winner: prem +30%, underlying
      confirming, ≤60min, respect 15% cap) standalone vs V6 DCA.
- [ ] Port into `execution/position_monitor.py` + `exit_v5/` behind `ENABLE_V7_ANTIMG`
      (and set `ENABLE_V6_DCA=false` when on). **Monitor-loop integration test MANDATORY**
      (HIGH RISK sell-path file — per CLAUDE.md). Source-safety test for new vars.
- [ ] Full suite green → `rebuild.sh` → kody only → dennis.

### Stage D — P(runner)-tilt sizing + cap loosening  (`ENABLE_V7_RUNNER_TILT`) — **LAST, the up-size**
- [ ] **BLOCKER: build serve-time runner model path.** `runner_v1.lgb` exists; wire it in:
  - [ ] Compute the runner feature vector at entry (match `scripts/runner_prediction.py` exactly).
  - [ ] Wire P(runner) lookup into `vinny_strategy.score_to_contracts`.
- [ ] In `score_to_contracts`: tilt (≤0.2438→0.5×, mid→0.85×, ≥0.5981→1.75×, ceiling=MAX_POSITION_PCT),
      loosen multi-day cap 2→4, **remove the inverted 60% conf tier** (`vinny_strategy.py:562-566`).
- [ ] Unit tests (tilt boundaries, cap, tier removal) + full suite.
- [ ] Only after Stage B win-rate confirmed live → deploy to a **small bot (vinny/$500) first**, then kody.

---

## TRACK 2 — UW flow → live trading (down-day strategy)

### ✅ DEPLOYED 2026-06-13 (user directive — calls+puts, all bots)
- [x] `uw_flow_collector.py` production module + 15 tests green; gate-bypass for flow; wired into bot_runner.
- [x] **CALL whitelist revised + validated** (uw_ticker_discovery, per-month split): `UW_FLOW_CALL_TICKERS=META,SPY,AMZN,TSLA,AAPL,AMD,PLTR` (added META/SPY/AMZN, dropped AVGO). PUT whitelist unchanged.
- [x] `ENABLE_UW_FLOW_SIGNAL=true` on **ALL 5 bots** — LIVE kody/dennis, PAPER adam/vinny/yank.
- [ ] ⚠️ **Mon open = first LIVE UW-flow exposure** (never live before). Watch closely; compare paper bots (adam/vinny/yank) vs live (kody/dennis) for the same signals. Rollback = flag false + `up -d`.
- [ ] Validate on a true OOS window once UW history extends past the current ~3mo cap.
- [ ] Trim AAPL/PLTR from call whitelist (PF<0.85) after a few sessions of live data.

### Phase 6a — Validate live signal (superseded — now live, was shadow)
- [ ] **Mon+: watch `docker compose logs owlet-flow-shadow -f`** — confirm whale put-sweep
      signals on META/AMZN/AAPL/TSLA fire at the expected rate (~3/day) and look sane vs backtest.
- [ ] Spot-check 3-5 signals against the next-30-60min underlying move (did the name drop?).

### Phase 6b — Production collector (paper-first)
- [ ] Port `scripts/uw_flow_shadow.py` → `options_owl/collectors/uw_flow_collector.py` (proper
      module, reconnect, `asyncio.wait_for` on the WS read per the event-loop rule).
- [ ] On a qualifying signal → **emit a PUT into the sourcing signals table** (same path Discord
      signals take) → existing entry gates → V7 exits. Name-whitelist META/AMZN/AAPL/TSLA.
- [ ] Add `ENABLE_UW_FLOW_SIGNAL` (settings, default off) + `UW_FLOW_TICKERS` whitelist.
- [ ] Unit tests (signal filter, signal emission shape). Full suite green.
- [ ] Deploy `rebuild.sh`; enable on **one PAPER bot** (PAPER_TRADE=true) for a few days —
      confirm DB trades match the shadow log + backtest expectancy.

### Phase 6c — Go live
- [ ] After paper confirms: enable `ENABLE_UW_FLOW_SIGNAL` on **kody** (live), keep size modest.
- [ ] Verify live container env; watch first live flow-driven trades; confirm V7 exits manage them.
- [ ] Then dennis.

---

## Cross-cutting / housekeeping
- [ ] Fix the regime `gex_ticks` serve warning (separate from the 1m fix; log-spam, fails-open).
- [ ] Rotate the UW API key (it passed through chat) — API Dashboard.
- [ ] Optional: backfill GEX/charm/vanna into the training set so models can USE it (prod captures it now).
- [ ] Confirm OOS#2 robustness for V7 exits / flow strategy on a second non-overlapping window when more data exists.

---

### Suggested order
1. **Track 0 (gate optimization 0.62)** — the volume fix, validate + paper first
2. **6a** (free — just watch Monday) → **6b paper** → **6c live**  *(the validated new edge)*
3. In parallel: **Stage B** (thin entry) → confirm WR → **Stage C** (anti-martingale)
4. **Stage D** (runner sizing) LAST, only after B's win-rate holds + serve-time model built

---

## TRACK 4 — End-to-end build + test for LIVE deploy (the "no bugs" gate)
Each new piece below ships only behind a flag, paper-first, with the tests listed. This is a
methodical multi-step effort — do NOT rush it; the monitor/entry paths are real money.

### Production code to write
- [x] `config/settings.py`: `ENABLE_UW_FLOW_SIGNAL=False`, `UW_FLOW_PUT_TICKERS`,
      `UW_FLOW_CALL_TICKERS`, `UW_FLOW_MIN_PREMIUM`, `UW_FLOW_ASK_FRAC`, `UW_FLOW_REQUIRE_SWEEP`,
      `UNUSUAL_WHALES_API_KEY`. (`ML_PATTERN_THRESHOLD=0.62` still TODO — the volume fix.)
- [x] `collectors/uw_flow_collector.py` — production module: WS connect+reconnect, pure
      `evaluate_flow_alert` filter (ask-side sweep, whitelist, premium) + `flow_signal_to_trade_signal`
      builder (BotSource.UW_FLOW). **12 unit tests green.**
- [x] **Signal emission hook** — `on_signal` builds the UW_FLOW TradeSignal and calls
      `paper_trader.evaluate_and_trade` directly (negative synthetic id). No PG round-trip needed.
- [x] **Gate-bypass for flow** — `_is_flow_sourced` added; 4 gates SKIP for UW_FLOW source:
      put_ticker_exclusion, put_market_direction, put_bearish_confirm, directional_regime.
      Risk gates (spread/delta/premium/EOD/cap) still apply. (Pattern/entry-timing bypassed by
      virtue of flow entering post-scan.)
- [x] **Wired into `bot_runner`** — supervised `uw_flow_collector` task, started only when
      `ENABLE_UW_FLOW_SIGNAL` + key present. Off = no task (zero impact).

### Tests (the end-to-end "no bugs" suite) — `tests/test_uw_flow_collector.py` (15 tests)
- [x] **Unit:** flow filter (ask-side/sweep/whitelist/premium, call vs put, malformed) + signal
      builder shape (source/direction/sentiment). [10 + 2]
- [x] **Gate-bypass unit:** flow signal SKIPs all 4 directional gates; non-flow NOT bypassed. [2]
- [x] **WS integration:** mocked socket → parse → filter → on_signal dispatches only the
      qualifying alert. [1]
- [x] Full suite green (2529); `ruff check` clean.
- [ ] **REMAINING — full E2E:** WS → evaluate_and_trade → trade row → position_monitor → V7 exit
      (mock Webull). Note: once entered, a UW_FLOW trade is managed by the EXISTING (tested) monitor
      + V7 FSM path — Track 4 is entry-side, it adds NO new sell-path code. This e2e is the final
      confidence check before live; recommended before flipping the flag on a paper bot.

### Deploy gate
- [ ] `rebuild.sh` (tests MUST pass) → shadow/paper confirm a few days → flip flag on kody →
      verify live env + first live trades → dennis. Rollback = flags off + `up -d`.

> Recommendation: run this as a focused build (or a multi-agent workflow), not at the tail of a
> session — the live-money entry+exit paths demand careful, rested implementation + review.
