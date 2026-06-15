# Spec 07: Extended Scan Window (Midday + Afternoon)

## Goal

Expand ML scanning beyond 9:35-11:00 AM to capture trades during midday and
afternoon sessions, gated by the regime detector (spec 06) to avoid choppy periods.

## Problem

The bot currently scans 9:35-11:00 AM only (90 minutes). This misses 4+ hours of
tradeable market time. H7 backtest showed extending to 2PM with CALLs-only had 23.7%
drawdown — but that lacked regime gating and PUT support.

## Design

### Time Windows

| Window | Time (ET) | Direction | Conditions |
|---|---|---|---|
| Morning | 9:35-11:00 | CALL primary | Current system, regime can block |
| Midday | 11:00-1:00 | Regime-dependent | CALL if bullish, PUT if bearish, skip if choppy |
| Afternoon | 1:00-2:30 | PUT primary | Current PUT scalp system + regime CALLs |
| Late | 2:30-3:00 | PUT only | Tight stops, regime-bearish required |
| Closed | 3:00-3:55 | No new entries | Theta death zone |

### Implementation

Modify `_run_ml_scan_loop` in `bot_runner.py`:

```python
def _get_allowed_directions(minute: int, regime: RegimeState) -> list[str]:
    """Return which directions the scanner may emit for this minute."""
    if minute < 5:
        return []  # opening buffer
    if minute <= 90:  # 9:35-11:00
        dirs = ["call"]
        if regime == BEARISH:
            dirs = ["put"]
        elif regime == CHOPPY:
            dirs = ["call"]  # morning momentum usually resolves
        return dirs
    if minute <= 210:  # 11:00-1:00
        if regime == BULLISH:
            return ["call"]
        if regime == BEARISH:
            return ["put"]
        return []  # choppy → sit out midday
    if minute <= 300:  # 1:00-2:30
        dirs = ["put"]
        if regime == BULLISH:
            dirs.append("call")
        return dirs
    if minute <= 330:  # 2:30-3:00
        if regime == BEARISH:
            return ["put"]
        return []
    return []  # 3:00+ no new entries
```

### Per-Window V5 Config Adjustments

| Window | Grace | Backstop | Scalp Target | Max Hold |
|---|---|---|---|---|
| Morning (9:35-11) | 5 min | -65% | 35% | 90 min |
| Midday (11-1) | 3 min | -50% | 25% | 60 min |
| Afternoon (1-2:30) | 3 min | -60% | 50% (PUT scalp) | 60 min |
| Late (2:30-3) | 2 min | -40% | 20% | 30 min |

Midday and late afternoon get tighter stops because theta decay accelerates and
liquidity drops.

### TickerScanState Reset

Currently `entry_emitted` blocks re-entry for the same ticker all day. With extended
hours, we need per-window cooldowns:

- After a CALL entry, that ticker is on cooldown for 90 minutes (current behavior)
- After a PUT entry, same 90 minute cooldown
- A CALL cooldown does NOT block PUTs (different direction = different trade)

## Testing Plan

### Backtest Scenarios

Run `backtest_combined.py` with extended windows vs morning-only:

1. **Morning only (baseline)**: 9:35-11:00 CALLs + 1:00-2:30 PUTs
2. **Morning + Midday**: Add 11:00-1:00 regime-gated
3. **Full day**: All windows active
4. **Full day + regime sizing**: Choppy = 60% size

Metrics: trades/day, WR, P&L, MaxDD, Sharpe
Target: +1-2 trades/day, MaxDD < 10%, positive P&L from added windows

### Unit Tests

1. `_get_allowed_directions()` returns correct directions for each window + regime
2. Cooldown per direction — CALL cooldown doesn't block PUTs
3. V5 config adjustments apply per window (tighter stops in afternoon)
4. No entries emitted after 3:00 PM

### Integration Tests

1. Full scan loop with mock regime detector switching mid-day
2. Verify morning CALL → midday regime flip → afternoon PUT transition
3. Verify choppy regime blocks midday entries

## Settings

```
ENABLE_EXTENDED_SCAN: bool = False  # gate behind flag
ML_SCAN_MIDDAY_START_MIN: int = 90   # 11:00 AM
ML_SCAN_MIDDAY_END_MIN: int = 210    # 1:00 PM
ML_SCAN_AFTERNOON_END_MIN: int = 300 # 2:30 PM
ML_SCAN_LATE_END_MIN: int = 330      # 3:00 PM
MIDDAY_SCALP_TARGET_PCT: float = 25.0
LATE_BACKSTOP_PCT: float = 40.0
```

## Dependencies

- Spec 06 (Intraday Regime Detector) — MUST be built first

## Acceptance Criteria

- [ ] Scanner operates across all time windows with correct direction gating
- [ ] Regime detector controls midday and late entries
- [ ] Per-window V5 configs applied (tighter afternoon stops)
- [ ] Per-direction cooldowns (CALL/PUT independent)
- [ ] Backtest shows net positive P&L from extended hours
- [ ] MaxDD stays under 10% with regime gating
- [ ] Feature gated behind ENABLE_EXTENDED_SCAN
- [ ] No entries after 3:00 PM ET
