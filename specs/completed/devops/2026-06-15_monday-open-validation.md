# Monday-Open Validation — first LIVE session of the full V7 + UW flow system

**Context:** 2026-06-15 is the first time the full deployed stack trades live money (kody $23k +
dennis $10k live; adam/vinny/yank paper). Everything below is deployed; this is the validation that
the backtest holds live. Run after the open.

## Checklist (run ~9:45–10:30 ET)
1. **Live bots recovered:** `docker compose ps` — owlet-kody / owlet-dennis show `Up (healthy)`, NOT
   `Restarting`. (They crash-loop on weekend stale quotes by design; must recover at 9:30 open.)
2. **Flow firing:** `bash scripts/monday_flow_check.sh` on the droplet — expect ~3 qualifying
   whale-flow signals/day; sane premiums; NO $1M+ index sweeps sized up (those are hedges, mult≈0.4).
3. **Paper vs live parity:** the SAME flow tickers should hit adam/vinny/yank (paper) and
   kody/dennis (live). Divergence ⇒ gate/fill/Webull issue — investigate.
4. **Conviction sizing live:** `grep CONVICTION_SIZING journal/owlet-kody/logs/options_owl_$(date +%F).log`
   — clustered / $1M single-stock sweeps sized UP; single sweeps + index hedges sized DOWN.
5. **(Optional) P(runner) canary:** flip `ENABLE_V7_RUNNER_TILT=true` on ONE paper bot (e.g. vinny),
   `up -d`, then `grep FLOW_P_RUNNER` — confirm iv/delta look sane and p_runner is spread across 0–1
   (NOT all 0 or all 1 = skew). Only enable on kody after this looks clean.
6. **(Optional) B2 tide-gate canary:** flip `ENABLE_V7_TIDE_GATE=true` on ONE paper bot (e.g. yank),
   `up -d`, then `grep CONVICTION_SIZING` for `tide-misaligned-put×0.30` — confirm PUT flow against a
   bullish tide gets sized down and CALLs are untouched. Validated point-in-time PUTS-ONLY (put aligned
   PF 1.66 vs misaligned 0.56; calls ~no edge). Enable on kody only after the canary behaves.

## What "good" looks like
- Live bots healthy, ~3 flow trades/day at fills near backtest, conviction multipliers behaving,
  paper≈live on the same signals. Then the +$113k-edge / PF 1.78 backtest is plausibly holding.

## Rollback (if anything looks wrong)
- Flow: `ENABLE_UW_FLOW_SIGNAL=false` + `up -d`.
- Conviction sizing: `ENABLE_V7_CONVICTION_SIZING=false` + `up -d`.
- Everything reverts to the validated V7 wide-trail + 0.62 gate baseline.
