# Spec 08: Regime-Triggered Stop Tightening

## Goal

When the regime detector identifies a market reversal, automatically tighten exit
stops on open positions that are now counter-trend. Prevents the common scenario
where a CALL trade is profitable but bleeds out during a bearish reversal.

## Problem

Today, stop tightening only happens:
- After 2PM ET (V6_2PM_TIGHTEN) — 30% tighter adaptive trails
- Grace period backstop — but only for catastrophic losses

There is NO protection for regime reversals at 10:30 AM or 12:00 PM. A CALL trade
opened at 9:45 during bullish conditions can sit open with wide stops while the
market reverses, giving back all gains.

## Design

### Trigger Conditions

When regime flips (confirmed, after hysteresis):

| Flip | Action on Open Positions |
|---|---|
| BULLISH -> BEARISH | Tighten all open CALLs: adaptive trail reduced 40% |
| BEARISH -> BULLISH | Tighten all open PUTs: adaptive trail reduced 40% |
| Any -> CHOPPY | Tighten all positions: adaptive trail reduced 20% |

### Emergency Tighten

If SPY drops 0.5%+ in 15 minutes (hard reversal from spec 06):
- All open CALLs: adaptive trail reduced to 25% (extremely tight)
- Positions already profitable: floor moves to entry price (breakeven ratchet)
- Positions underwater: backstop tightened to -40% (from -65%)

### Implementation

Modify `position_monitor.py` monitor loop:

```python
# In the 5-second monitor cycle, before evaluating exits:
if regime_detector and regime_detector.state_changed_since(last_check):
    new_regime = regime_detector.state
    for trade in open_trades:
        if _is_counter_trend(trade.direction, new_regime):
            # Inject regime tightening into the FSM config
            fsm = get_fsm(trade)
            fsm.apply_regime_tighten(factor=0.60)  # 40% tighter
            logger.info(
                f"REGIME_TIGHTEN: #{trade.id} {trade.ticker} "
                f"{trade.direction} — regime flipped to {new_regime}"
            )
```

Add to `exit_v5/fsm.py`:

```python
def apply_regime_tighten(self, factor: float = 0.60) -> None:
    """Tighten adaptive trail widths by factor (0.6 = 40% tighter)."""
    self._regime_tighten_factor = factor
    # Applied in _get_adaptive_trail_width() multiplication
```

### Interaction with Existing Stops

- Regime tightening STACKS with 2PM tightening (multiplicative)
  - At 2:15 PM during reversal: 0.70 (2PM) * 0.60 (regime) = 0.42x trail width
- Breakeven ratchet (V6) takes precedence — floor never goes below entry
- Scaleout is unaffected — regime tightening only changes trail widths

## Testing Plan

### Unit Tests

1. `apply_regime_tighten(0.60)` reduces adaptive trail by 40%
2. Regime tighten stacks with 2PM tighten multiplicatively
3. Breakeven ratchet still holds (floor >= entry even with tightening)
4. Only counter-trend positions are tightened (CALLs during bearish, PUTs during bullish)
5. CHOPPY tightens both directions at 20%

### Backtest Validation

Replay historical days with known reversals:
- May 5 (big afternoon selloff) — measure CALL P&L with vs without regime tighten
- April 15 (morning rally → afternoon fade)
- Quantify: how much P&L was saved by earlier exits?

### Integration Tests

1. Open CALL trade → flip regime to BEARISH → verify trail width reduced
2. Open PUT trade → flip regime to BULLISH → verify trail width reduced
3. Emergency tighten: SPY -0.5% in 15 min → verify 25% trail on all CALLs
4. Verify tightening doesn't affect already-closed trades

## Settings

```
ENABLE_REGIME_STOP_TIGHTEN: bool = True
REGIME_TIGHTEN_FACTOR: float = 0.60       # 40% tighter on reversal
REGIME_CHOPPY_TIGHTEN_FACTOR: float = 0.80 # 20% tighter in chop
REGIME_EMERGENCY_TRAIL_PCT: float = 25.0   # emergency tighten trail
REGIME_EMERGENCY_BACKSTOP_PCT: float = 40.0
```

## Dependencies

- Spec 06 (Intraday Regime Detector) — provides regime state + flip events
- Existing V5 FSM exit engine

## Acceptance Criteria

- [ ] Counter-trend positions get tighter trails within 5 seconds of regime flip
- [ ] Emergency tighten fires on hard reversal (0.5% SPY drop)
- [ ] Stacks correctly with 2PM tighten
- [ ] Breakeven ratchet still respected
- [ ] Backtest shows P&L improvement on reversal days
- [ ] No impact on same-trend positions
