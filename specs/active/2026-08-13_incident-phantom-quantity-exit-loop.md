# INCIDENT: phantom quantity blocked an exit for 35 minutes (2026-08-13)

Status: ACTIVE (fixes committed 6ddcbe6, NOT deployed)
Category: bugfixes / incident
Cost: a +$1,313 position closed at about -$1,280. Roughly a $2,600 swing on one trade.

## Timeline (kody, IWM 305C 0DTE, trade #681)

```
13:55:05  ENTRY ATTEMPT x78
13:55:05  LIQUIDITY SIZE-DOWN 78x -> 71x      (fits the exit book)
13:55:10  FILLED DURING CANCEL                 record keeps 78; broker holds 71
14:19:54  FSM: profit_lock at price 0.48       unrealized +$1,313.50
          sell 78 vs 71 held -> HTTP 417 MUST_BE_CLOSE_THAN_SELL_SHORT
14:34     price 0.31                           +$106.50, still rejected
14:55     price 0.11                           -$1,313.50, ~760 rejections
```

The exit decision was correct and timely. It could not execute.

## Defect 1 — entry did not persist the true filled size

`_place_buy_with_escalation` sizes down for liquidity (`contracts = cap`), then the order
filled through the **FILLED-DURING-CANCEL** branch, which set:

```python
filled_qty = None if confirm_status == "FILLED" else await self._get_filled_quantity(...)
```

`None` means "matches the request, nothing to correct". But "filled" there meant the
SIZED-DOWN 71, not the requested 78, so `paper_trader` kept 78. The normal FILLED path
already guards this with `contracts != requested_contracts`; this branch never made the
comparison. Fixed by mirroring it.

Note: telemetry was added to this exact return on 2026-08-12 without noticing the
quantity semantics on the adjacent line.

## Defect 2 — a permanent rejection was classified as transient

The rejection fell through to `SellOutcome.TRANSIENT_ERROR`, which by design does NOT
consume the manual-close abandonment budget, so the monitor reopened and retried every
~4s indefinitely. A condition that can never succeed must escalate, not loop.

Added `SellOutcome.QUANTITY_MISMATCH` plus a self-heal: read the broker's true size and
shrink the record to it, so the next exit submits a fillable size.

## The trap: this error has TWO causes

`MUST_BE_CLOSE_THAN_SELL_SHORT` also fires when a pending/resting order ties up available
quantity (the META #467/#424/#376 chain, fixed 2026-07-07 with cancel-and-confirm). That
cause IS transient and recovers on retry. Classifying the error permanent outright would
have traded one blocked-exit bug for another.

The self-heal therefore acts ONLY when the broker reports a provably SMALLER position.
An unreadable size, or a size equal to the record, falls through to the existing transient
retry. A broker size of 0 routes to POSITION_NOT_FOUND.

**Shrink-only is deliberate.** Selling fewer than held cannot oversell; raising the count
could create a real naked short.

## Diagnostic notes for next time

- `get_open_option_positions()` returned an empty list once mid-incident and I reported the
  position as flat when it was not. **Verify a flat reading twice before acting on it.**
- The DB oscillates during a retry loop (`close -> sell fails -> revert_and_reopen`), so
  `status` and `pnl_dollars` read at any instant are unreliable. The monitor's CLOSED
  notifications reported +$648, +$66, -$1,330 for the same open trade. **Trust the broker
  position, not the DB, while a loop is running.**
- 429s roughly tripled during the loop (11-22 per 10-min before, peaking at 71) and were
  concentrated on `position` and `order` calls. They pre-existed the loop, so it
  contributed rather than caused.

## Follow-ups

- Deploy 6ddcbe6 (restarts live bots — needs approval).
- Reconcile #681's record; the loop persists while it reads open with 78.
- The daily loss cap fired correctly and stopped further entries.
- **Fill accounting deserves a deliberate review.** Two consecutive days produced bugs in
  that path, each surfacing somewhere else entirely (telemetry gaps 08-12, a sell loop
  08-13). Another incremental patch is not the right answer.
