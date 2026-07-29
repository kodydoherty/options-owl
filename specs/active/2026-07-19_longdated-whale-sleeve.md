# Long-Dated Whale-Following Sleeve (15-20% allocation)

**Status:** active (spec) · **Created:** 2026-07-19 · **Owner:** Kody

## Hypothesis (VALIDATED — backtest done, build pending)
A 15-20% capital sleeve that buys the whale's **actual long-dated (>30 DTE) call contract** on ask-side
whale sweeps — the expiry/strike they actually chose, NOT the 0DTE version the current flow book substitutes —
is a real, additive strategy. Whales put ~82% of their premium in 8+ DTE contracts ($1.2B vs $261M in 0-2 DTE);
we currently ignore that expiry on most flow.

## Evidence (12-month, multi-regime, 5,036 apples-to-apples signals)
Harness: `extend_flow_history.py` (12mo UW flow, retention floor 2025-07) → `download_longdated_flow.py`
(Polygon daily bars for the whale's exact contract) → `download_longdated_benchmark.py` +
`backtest_longdated_benchmark.py` (whale vs same-ticker ATM vs SPY-ATM).

| Exit rule | WHALE | ATM (same tkr) | SPY (beta) | alpha vs SPY | alpha vs ATM |
|---|---|---|---|---|---|
| trail 40% | +22.5% (PF 2.02) | +16.6% | +6.1% | **+16.4%/tr** | +5.9% |
| time 21d | +26.9% (PF 1.83) | +19.1% | +9.4% | **+17.4%/tr** | +7.7% |
| +50%/−50% | +5.7% (PF 1.27) | +4.1% | +3.0% | +2.7% | +1.6% |

**Decomposition:** name+timing selection ≈ +10%/tr (ATM−SPY), strike+entry selection ≈ +6-8%/tr (WHALE−ATM).
Whale beats BOTH benchmarks in every rule → **real layered selection alpha, not "expensive beta."**

## Known caveats (must be respected in the build)
1. **Long-biased.** 12mo was an up year (even SPY calls won +6-9%). The *alpha* is regime-robust; the *absolute*
   return is market-dependent — a flat/down year makes less or loses (the isolated May-Jul 2026 down-window had
   calls lose, puts win). Puts do NOT hedge it (they lose too). → treat as a directional sleeve; see Risk.
2. **High-variance / convex.** The high-alpha rules (trail_40/time_21d) are ~34-41% WR — a few big winners carry
   it. Expect long stretches of small losses between moonshots. +50/-50 is steadier (55% WR) but less alpha.
3. **Fills/spread NOT modeled** — daily-open entry, no spread. Far-dated strikes are wider → realized edge below
   the backtest. A live spread gate + paper-canary are mandatory before real money.
4. **Capital lockup** — weeks-long holds tie up sleeve capital; concurrency is bounded by capital, not ideas.

## Build

### 1. Signal path (reuse the existing flow WS — do NOT open a new connection)
- The harvester is the sole UW flow-alerts WS holder → publishes to Redis `owl:flow:signals`. The 0DTE flow book
  (`bot_runner._on_flow_signal`) consumes these and RESOLVES nearest-DTE (ignoring `fs.expiry`).
- Add a **parallel consumer** `_on_longdated_flow_signal` gated by `ENABLE_LONGDATED_SLEEVE`. Filter for:
  ask-side SWEEP, `total_premium ≥ LONGDATED_MIN_PREMIUM` ($250k), `has_sweep`, and **DTE = (fs.expiry − today)
  > LONGDATED_MIN_DTE (30)**. Universe: start with the current flow whitelist ∪ the validated call names; the
  backtest used all tickers, so a broad liquid universe is defensible — but gate on liquidity (see 4).
- New `BotSource.LONGDATED_WHALE = "longdated_whale"` (models/signals.py) so these positions are tagged and
  tracked separately from 0DTE.

### 2. Contract selection — buy the whale's ACTUAL contract
- Use `fs.expiry` + `fs.strike` + `fs.option_type` verbatim (validated: whale strike beats ATM by +6-8%/tr).
  Do NOT ATM-normalize, do NOT resolve nearest-DTE (that's the 0DTE path).
- Resolve the live contract via the harvester Redis snapshot first (Polygon fallback).

### 3. Exit rule (SLOW cadence — this is the biggest departure from 0DTE)
- **Primary (recommended): trailing-stop-40** — track peak premium since entry, exit when premium falls ≥40%
  from peak; plus a hard **−50% stop** and a **time backstop** (exit N days before expiry, e.g. 5, to dodge
  terminal theta). Best risk-adjusted in the backtest (PF 2.02, +16.4% alpha) but convex.
- **Alt (steadier): +50%/−50% bracket** — 55% WR, +median, PF 1.27; lower alpha, much lower variance. Canary
  can A/B these on paper before committing.
- **Cadence:** a SEPARATE slow monitor loop (every few minutes / hourly), NOT the 5s 0DTE loop. Long-dated
  premiums move on daily timescales; 5s polling is wasteful and the FSM gates are 0DTE-tuned (wrong here).
- **CRITICAL:** sleeve positions MUST be EXEMPT from `ENABLE_EOD_CLOSE_ALL` / late-day cutoffs — they hold
  overnight for weeks by design. This is the #1 way a naive integration would silently break the strategy.

### 4. Sizing & risk controls
- Sleeve cap: `LONGDATED_SLEEVE_PCT` (15-20%) of portfolio, enforced as total open sleeve cost.
- Concurrency: `LONGDATED_MAX_CONCURRENT` (start ~5) → per-position ≈ sleeve/concurrent (~3-4% each). Do NOT
  reuse 0DTE conf_linear/runner_v1 sizing (0DTE-specific) — flat or conviction-tiered (flow cluster/premium).
- **Affordability gate:** long-dated contracts are expensive ($10-50/contract); skip if 1 contract > per-position
  budget (the flow "priced-out" finding — small accounts can't participate; kody's $23k can hold ~1-2/position).
- **Spread gate:** skip if bid-ask spread > threshold (far-dated illiquidity — the unmodeled backtest risk).
- **Delta gate:** the existing DeltaEntryGate (max 0.70) still applies — no deep-ITM (overpaying intrinsic).
- **Direction risk:** the sleeve + 0DTE book are BOTH long-call-biased → combined directional exposure. Consider
  a combined long-exposure cap OR only arm the sleeve when regime is not-bearish (the sleeve loses in down tapes).

### 5. Execution
- Reuse `webull_executor` for the BUY (entry chase applies; liquidity-aware sizing now live). Multi-day hold on a
  cash account is fine on the long side (no GFV on buys; sell-to-close later settles normally).
- Reconciliation: sleeve positions persist across restarts — the existing `_reconcile_positions` sweep + the new
  slow monitor must recognize `bot_source='longdated_whale'` and NOT apply 0DTE EOD/theta gates to them.

## Paper-canary plan (MANDATORY before live)
- Arm `ENABLE_LONGDATED_SLEEVE=true` on ONE paper bot (vinny/yank/alan) — live paper fills, real slow holds.
- Run **weeks** (holds are multi-week — the canary needs time; start ASAP). Compare paper fills vs backtest to
  quantify the spread/fill haircut (the unmodeled risk). A/B the two exit rules if capital allows.
- Watch: fill quality on far-dated strikes, spread-gate reject rate, actual hold durations vs backtest, absolute
  return vs the concurrent market direction (is the alpha showing up live?).
- Go-live criteria: paper sleeve PF > ~1.3 after fills over a multi-week window spanning at least one down stretch.

## Open questions / decisions for Kody
1. **Exit rule** to canary first: trail_40 (max alpha, convex) vs +50/−50 (steady). Recommend canary BOTH.
2. **Universe**: broad-liquid (backtest used all) vs the validated call-name whitelist. Recommend broad + a hard
   liquidity/spread gate.
3. **Sleeve %**: 15 vs 20. Recommend start 15, room to grow.
4. **Direction guard**: cap combined long exposure, or regime-arm the sleeve? (mitigates the down-year risk).

## Files (new/changed when built)
- `models/signals.py` (BotSource.LONGDATED_WHALE), `execution/bot_runner.py` (`_on_longdated_flow_signal`),
  new slow monitor (`execution/longdated_monitor.py`), `config/settings.py` (ENABLE_LONGDATED_SLEEVE +
  LONGDATED_* knobs), `docker-compose.yml` (per-bot flag, canary first).
- Research harness (DONE, reusable): `scripts/{extend_flow_history,download_longdated_flow,
  download_longdated_benchmark,backtest_longdated_flow,backtest_longdated_benchmark}.py`.

## Validation memory
[[longdated-whale-following-2026-07-18]] (full results + caveats).
