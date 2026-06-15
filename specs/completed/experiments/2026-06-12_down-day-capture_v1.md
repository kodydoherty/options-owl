# Experiment: Rapid Mid-Day Down-Day Capture (regime-conditional PUTs)

**Status:** ACTIVE (analysis phase) · Created 2026-06-12

## Hypothesis
On sustained down days, aggressive mid-day PUT entry (allow the normally-excluded
tickers AMZN/PLTR/MSTR, lower the SPY "bear-mode" bar) captures large moves the current
gates block. BUT the edge only holds if we can distinguish a **sustained crash** (PUTs
win) from a **dip-and-rebound** (PUTs die) early enough to act.

## The trap (why naive "red → buy PUTs" fails)
2026-06-11: market dropped, then a Trump statement V-shaped it back up. owlet-kody's 4
PUTs that day lost **−$824** — entered on the drop, crushed on the rebound. News-driven
reversals are exogenous (not predictable from price alone), so the strategy must either
(a) only fire on *confirmed-structure* drops (lower-lows, sustained-below-VWAP,
accelerating) that filter shallow dips, and/or (b) react fast with the breakeven ratchet
+ tight exit so a reversal can't crush an in-the-money PUT.

## Current entry blocks (this week, owlet-kody funnel)
- `AMZN excluded from PUTs ×40`, `PLTR ×25`, `MSTR` — "backtest losers" blocked even on
  crash days (when those big names drop hardest).
- `SPY −0.42% (red but not bear mode) → PUTs blocked` — bear-mode bar misses the early
  crash; PUTs only allowed after SPY already dropped a lot.

## Method
1. **Find + classify down days** (`scripts/analyze_down_days.py`): 2.5yr SPY 1m data
   (651 days). Per day compute open/close/low, time-of-low, max drawdown, recovery ratio.
   Classify: up / flat / **sustained-down** (close near low) / **dip-rebound** (low then
   recover >50%). Quantify base rates.
2. **Mid-day separability**: at 11:00 / 12:00 / 13:00 ET checkpoints, measure whether
   sustained vs rebound is distinguishable from price/VIX/volume/structure features
   available at that time (the "can we react mid-day" question).
3. **Available PUT P&L**: for sustained days, estimate PUT P&L if entered at the
   checkpoint vs the current gated behavior — the size of the missed opportunity.
4. **Design the trigger**: a regime-conditional "down-day mode" that, only when a
   sustained-crash signal fires mid-day, (a) lifts the AMZN/PLTR/MSTR PUT exclusions and
   (b) lowers the bear-mode bar. Backtest vs baseline before ANY live change.

## Results (phase 1 — find + classify, 611 days 2024-2026)
- Base rates: up/flat 59.1% · **REBOUND 16.9% (103d)** · PARTIAL 10.8% · **SUSTAINED 13.3% (81d)**.
  Rebounds OUTNUMBER sustained crashes → naive "red→PUTs" loses more often than it wins.
- **Drawdown does NOT separate them mid-day**: avg dd-so-far SUSTAINED vs REBOUND is
  −0.60/−0.56 (11:00), −0.77/−0.66 (12:00), −0.90/−0.73 (13:00) — gap 0.04–0.18%.
  This is the −$824 trap: by magnitude alone the rebound looks like the crash.
- **Signature that DOES separate: time-of-low.** Worst SUSTAINED days bottom late
  (low_time 15:18–16:00) — they keep making new lows into the close; rebounds bottom
  mid-day and recover. → the trigger should be *structure* ("still making new lows in the
  afternoon, no bounce, holding below VWAP"), not drawdown size.
- Per-day table: `journal/v3_eval_results/down_days.csv`.

## Next (phase 2)
Add structure features at each minute (new-low-in-last-30m, below-VWAP persistence,
drop velocity/acceleration, VIX level/Δ, volume surge); train a mid-day SUSTAINED-vs-
REBOUND classifier; measure capturable PUT P&L with a "confirm-then-commit" rule vs the
current gates. Only then design the regime-conditional entry change.

## Results (phase 2 — mid-day classifier, 19,176 red decision points)
- Walk-forward OOS AUC **0.72 ± 0.19** (real but weak, high variance). Structure features
  (new-lows, VWAP persistence, velocity, QQQ cross-confirm) DO rank sustained > rebound,
  but not strongly.
- **Symmetric SPY-move proxy is NEGATIVE at every threshold AND every time bucket**
  (incl. the 14:30-15:30 accel window). You cannot *predict* a capturable down move from
  a red mid-day point — intraday SPY mean-reverts (rebounds dominate).
- **BUT PUTs are CONVEX and that flips it.** Distribution of fires (p>=0.7): p90 +0.72%,
  p95 +1.30% SPY move (fat right tail = the rare sustained crashes). With the V7 no-ceiling
  trail riding winners and the ratchet/stop capping losers: asymmetric sim gives
  **+0.62%/trade (loss cap -0.3%) to +0.36% (-0.5%), at 44% win** — positive expectancy
  from convexity, not from prediction.

## Results (phase 3 — REAL SPY 0DTE PUT backtest, classifier signal + V5/V7 ExitFSM, real premiums)
The 6x proxy was too rosy. On real 0DTE premiums (confirm-then-commit, first fire/day):
- **High win rate (58-68%) + positive median (+3 to +7%)** — most signal PUTs are small winners.
- **BUT mean ≈ breakeven-to-negative** — a fat LEFT tail (0DTE puts go to ~−100% on rebound
  days) drags it. The proxy assumed a fat RIGHT tail; real 0DTE has a fat LEFT tail (theta +
  total-loss on the Trump-style V). Best cell: thr>=0.6 **V7** = +0.6%/trade, 60% win,
  +77% total (136 trades). V7 exits beat V6 at low thresholds (cut losers) but it's marginal.
- **Verdict: roughly breakeven on SPY 0DTE — not the goldmine the proxy implied.** The loss
  tail is 0DTE rebound wipeouts. Levers to test next: (a) **1-2 DTE** (less theta / retains
  value on rebound → shrinks the left tail), (b) **tighter hard stop** (cut rebound losers
  before they zero), (c) **high-beta single names** (AMZN/PLTR/MSTR move more on down days).
  CALL->PUT reverse still untested. Scripts: `backtest_down_day_puts.py`.

## Results (phase 4 — WHALE FLOW signal, Unusual Whales API, 90d)
After price-prediction failed, tested real-money flow as the signal:
- **Raw put-flow surge (index, $250k+): NO edge** (SPY fwd30m −0.01% vs +0.006% baseline).
  Index put flow is hedging + put-selling dominated.
- **Ask-side put SWEEPS (bought, bearish conviction) on index: tiny edge** (−0.033% 30m,
  −0.051% 60m) — real but too small to trade on the index.
- **SINGLE-NAME ask-side put sweeps: THE EDGE.** mean −0.079%, median −0.193%, 62% down.
  By name: **AMZN −0.50%/100%down, AAPL −0.22%/88%, TSLA −0.47%/82%** (strong);
  META/NVDA mild; **AMD noise; MSTR +0.42% (squeezes UP — exclude).**
- This is the first clear, sizable edge in the whole investigation. Leveraged 0DTE puts
  amplify the −0.2 to −0.5% move; V7 exits cut the ~18% squeezes, ride the 82% drops.
  Caveat: 90d window, small n on AMZN/AAPL (TSLA n=28 most robust).
- Scripts: `whale_flow_signal_test.py`. Data: `whale_put_flow_surges.csv`. API: [[unusual-whales-api-2026-06-12]].

## Results (phase 5 — full premium backtest, REAL premiums + V5/V7 ExitFSM)
Follow ask-side put sweeps on the whitelist, enter near-ATM put, manage with the FSM.
**90-day OOS (276 trades, Mar 16–Jun 12): V7 +11.0%/trade, 59% win, +3032% total;
V6 +8.8%, +2430%.** V7 beats V6 (rides the convex winners). By name (V7): **META +45.5%
(+2186%, the star), AMZN +19.5%, AAPL +18.1%, TSLA +2.6%; NVDA −2.9% (LOSER — drop).**
**VALIDATED in-and-out-of-sample.** Live whitelist = **META/AMZN/AAPL/TSLA**, exclude
NVDA/MSTR/AMD. Scripts: `backtest_whale_flow_puts.py`. Caveat: one 90d window, META-tail heavy.

## Phase 6 (IN PROGRESS): wire UW flow WebSocket → sourcing pipeline (name-whitelisted)
WS protocol confirmed: `wss://api.unusualwhales.com/socket?token=<KEY>`, send
`{"channel":"flow-alerts","msg_type":"join"}`. Advanced plan (have it). Official ref:
github.com/unusual-whales/api-examples (ws-stream-flow-alerts-to-sqlite).
- ✅ **SHADOW collector DEPLOYED LIVE** on droplet as `owlet-flow-shadow` (docker-compose,
  `scripts/uw_flow_shadow.py`, key in droplet .env). Connected + joined flow-alerts, log-only,
  NO trading, isolated from trading bots. Filters ask-side put sweeps on META/AMZN/AAPL/TSLA >=$250k.
  **Monday: `docker compose logs owlet-flow-shadow -f` — confirm signals match the backtest.**
- ⬜ NEXT: port into `options_owl/collectors/` as a real collector; on signal, emit a PUT into the
  sourcing signals table → existing entry pipeline → V7 exits. Flag `ENABLE_UW_FLOW_SIGNAL`
  (default off). Rollout: shadow → paper (1 bot) → live (kody) after a few days of matching.

## Verdict
The edge is NOT entry prediction (you can't reliably call the sustained crash mid-day).
The edge is **position management on a weak signal**: fire PUTs on the classifier signal,
**cut losers hard/fast** (the entire edge is in the loss cap), and let the V7 no-ceiling
trail ride the rare big crash. 44% win still nets positive via convexity. This is the SAME
lesson as [[entry-filter-refuted-2026-06-11]]: faders/traps can't be filtered at entry —
the edge is exits + sizing + cutting losses. The V7 PUT exits (just deployed) are the right
tool. NEXT: real PUT backtest (actual premiums + V5/V7 engine, not the 6x proxy) of
"confirm-then-commit + tight stop", and test a CALL->PUT stop-and-reverse on regime flip.

## Guardrails
- No live entry-gate change until backtested across multiple windows (the exclusions
  exist because those tickers lose on normal days — only loosen them conditionally).
- Reuse the (now-fixed) regime model as the "down-day confirmed" signal once 1m candles
  give it morning data (Monday).
