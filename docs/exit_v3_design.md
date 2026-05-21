# Exit Pipeline v3 — Thesis-Driven Exit Strategy

## Overview

Replace the timed grace period and hard % stops with intelligent,
chart-confirmed exit logic. Three core components:

1. Continuous Thesis Revalidation (replaces grace period + hard stops)
2. Ratcheting Profit Floor (replaces dormant zone gap)
3. Bounce-and-Fade Detection (replaces catastrophic stop)

## 1. Continuous Thesis Revalidation

**Current:** 20-min grace period blocks all stops. ENRG fires once.
**New:** Every exit poll cycle (5s), if position is negative, revalidate
the trade thesis using multi-TF candle data.

```
every 5s poll:
  if position is NEGATIVE:
    candles = fetch(5m, 15m, 30m, 1h)

    for each timeframe:
      vote = evaluate(RSI, OBV, pattern, trend)
      # BULLISH (for calls) / BEARISH (for puts) / NEUTRAL

    weighted_score = sum(votes * weights)
    # weights: 5m=1, 15m=1, 30m=2, 1h=3

    if STRONG_INVALIDATION:
      # majority of TFs confirm thesis is dead
      # e.g. 1h engulfing_bearish on a call, RSI divergence
      → EXIT immediately (small loss, capital preserved)

    if WEAK_INVALIDATION:
      # mixed signals, thesis weakening
      → TIGHTEN floor to breakeven if possible

    if THESIS_VALID:
      # higher TFs still confirm direction
      → HOLD, no stop applied
```

Key difference from current ENRG:
- Fires continuously, not once
- No blind grace period — every negative tick is evaluated
- Extreme patterns on 30m/1h trigger immediate exit
- Result is NOT persisted (re-evaluates each cycle)

## 2. Ratcheting Profit Floor

**Current:** Adaptive trail dormant below +35%, profit_retrace at 50% give-back.
**New:** Profit floor activates at +15% and ratchets up, never down.

```
PROFIT_FLOOR_ACTIVATION = 15%   # floor starts here
PROFIT_FLOOR_RATCHET = 60%      # floor = 60% of peak gain

Example:
  entry = $1.00

  price hits $1.15 (+15%) → floor activates at $1.09 (60% of $0.15 gain)
  price hits $1.30 (+30%) → floor ratchets to $1.18 (60% of $0.30 gain)
  price hits $1.50 (+50%) → floor ratchets to $1.30 (60% of $0.50 gain)
  price hits $2.00 (+100%) → floor ratchets to $1.60 (60% of $1.00 gain)

  price reverses to $1.60 → EXIT at floor, locked in +60% gain

Floor formula:
  floor = entry + (peak_gain * RATCHET_PCT)
  if current_premium <= floor → EXIT
```

This replaces:
- profit_retrace (50% give-back threshold)
- adaptive trail dormant zone (no protection below +35%)
- The gap between profit_retrace and adaptive trail

The ratchet is TIGHTER than adaptive trail (keeps 60% of gains vs 65%),
but activates MUCH earlier (+15% vs +35%).

## 3. Bounce-and-Fade Detection (replaces catastrophic stop)

**Current:** 45% hard stop bypasses grace period.
**New:** On deep dips, wait for a bounce then sell the fade.

```
if drop_from_entry >= 50%:
  # Don't panic sell — we're likely at the bottom
  # Enter BOUNCE_WATCH mode

  track: bounce_high = current_premium  (reset on each new high)

  # Wait for any recovery (even small)
  if premium > bounce_low * 1.10:  # 10% bounce from the bottom
    # We have a bounce. Now watch for the fade.
    bounce_detected = True
    bounce_high = max(bounce_high, premium)

  if bounce_detected:
    # If it starts rolling over from the bounce high
    fade_pct = (bounce_high - premium) / bounce_high * 100

    if fade_pct >= 15%:  # dropping 15% from bounce high
      → EXIT (caught the bounce, selling before second leg down)
      # This saves capital vs selling at the absolute bottom

  # TIME PRESSURE: adjust based on time remaining
  time_remaining = expiry - now

  if time_remaining < 60min:
    # Last hour — theta is killing us, be more aggressive
    fade_threshold = 8%    # tighter fade detection
    bounce_threshold = 5%  # accept smaller bounces

  if time_remaining < 30min:
    # Last 30 min — get out on ANY bounce
    if premium > entry * 0.55:  # any premium above the low
      → EXIT immediately

  if time_remaining < 15min:
    # Emergency — just sell whatever we have
    → EXIT at market
```

### Why this is better than a hard 45% stop:

Real example — TSLA $380 CALL (trade #48 vs Adam's #49):
- Entry: $1.55
- Dropped to: $0.54 (-65%)
- 45% stop would exit at: ~$0.85 (loss = -$420)
- Bounce detection: wait for bounce to ~$0.80, fade to $0.70 → exit
  - If it recovers (like Adam's did to $1.81): HOLD via thesis revalidation
  - If it fades after bounce: exit at ~$0.70 (loss = -$510, similar to stop)
  - KEY: thesis revalidation would have caught the recovery

## 4. Time-Aware Urgency Tiers

All exit logic adjusts based on time remaining on the option:

| Time to Expiry | Urgency | Effect |
|---------------|---------|--------|
| > 2 hours | NORMAL | Standard thresholds |
| 1-2 hours | ELEVATED | Tighten profit floor ratchet to 70% |
| 30-60 min | HIGH | Tighten to 80%, bounce fade at 8% |
| 15-30 min | CRITICAL | Exit on any bounce, floor at 90% |
| < 15 min | EMERGENCY | Market exit, preserve any value |

## Gate Priority (proposed v3 order)

1. `thesis_revalidation` — continuous multi-TF check when negative
2. `profit_floor` — ratcheting floor from +15% (replaces profit_retrace)
3. `bounce_fade` — deep dip bounce detection (replaces catastrophic stop)
4. `tranche_scaleout` — lock 1/3 at +25%
5. `volume_peak` — exhaustion detection
6. `dollar_trail` — stair-step trailing
7. `adaptive_trailing_stop` — 3-stage trail (backup to profit_floor)
8. `eod_cutoff` — 3:45 PM ET hard close

## Migration Path

Phase 1: Ratcheting profit floor (low risk, replaces dormant zone gap)
Phase 2: Continuous thesis revalidation (expand ENRG to run every cycle)
Phase 3: Bounce-fade detection (replace catastrophic stop)
Phase 4: Time-aware urgency tiers
