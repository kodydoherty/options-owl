# Spec 10: Dynamic PUT Expansion

## Goal

Automatically expand PUT trading beyond the fixed 1:00-2:30 PM window when the
regime detector identifies bearish conditions. PUTs become the primary play during
selloffs, providing a natural hedge against CALL losses.

## Problem

Today PUTs are constrained:
- Fixed window: 1:00-2:30 PM only
- Bear mode (SPY down 0.5% from open) expands PUT tickers but is a static check
- No PUT trading during morning selloffs (9:35-11:00 is CALL-only)

When the market sells off at 10 AM, the bot keeps trying CALLs (losing) while
the best PUT opportunities go untaken.

## Design

### Dynamic PUT Rules

PUTs are allowed whenever:

1. **Regime is BEARISH** (from spec 06) — any time window
2. **Afternoon window** (1:00-2:30 PM) — regardless of regime (current behavior)
3. **Late window** (2:30-3:00 PM) — only if regime is BEARISH

PUTs are blocked when:
1. **Regime is BULLISH** and outside afternoon window
2. **Ticker is in PUT exclusion list** AND regime is not BEARISH
   - Current exclusions: AAPL, GOOGL, NVDA, AMZN (net losers in 3yr backtest)
   - When regime is BEARISH, exclusions are lifted (strong trend overrides)

### PUT Slot Management

| Regime | Max PUT Slots | Max CALL Slots | Total Concurrent |
|---|---|---|---|
| BULLISH | 2 | 6 | 8 |
| BEARISH | 6 | 2 | 8 |
| CHOPPY | 3 | 3 | 6 (reduced total) |

### PUT V5 Config by Time Window

| Window | Config | Scalp Target | Max Hold | Backstop |
|---|---|---|---|---|
| Morning PUT (regime) | PUT_SCALP_CONFIG | +35% | 60 min | -50% |
| Midday PUT (regime) | PUT_SCALP_CONFIG | +35% | 60 min | -50% |
| Afternoon PUT (1-2:30) | PUT_SCALP_CONFIG | +50% | 60 min | -60% |
| Late PUT (2:30-3) | TIGHT_PUT_CONFIG | +20% | 30 min | -40% |

Morning/midday PUTs use tighter targets than afternoon PUTs because they're
counter-seasonal (morning typically trends up).

### Bear Mode Integration

Current bear mode (SPY down 0.5% from open) is replaced by the regime detector:

```python
# Old: static check at scan time
if spy_change < -0.005:
    expand_put_tickers()

# New: regime detector handles this dynamically
# Bear mode becomes just one trigger for BEARISH regime
# (along with EMA crossover, VWAP break, ADX confirmation)
```

## Testing Plan

### Backtest Scenarios

Using `backtest_combined.py`:

1. **Current system**: Fixed CALL morning + fixed PUT afternoon
2. **Dynamic PUTs**: PUTs allowed during morning bearish regime
3. **Dynamic PUTs + slot management**: 6 PUT slots during bearish
4. **Full system**: Dynamic PUTs + regime + conviction sizing

Per-scenario metrics: PUT trades/day, PUT WR, PUT P&L, combined P&L, MaxDD

### Historical Replay Days

Test on known selloff days:
- Days where SPY dropped 1%+ by noon
- Days with morning rally → afternoon reversal
- Flat/choppy days (PUTs should stay minimal)

### Unit Tests

1. PUT direction allowed during BEARISH regime at any time
2. PUT excluded tickers unlocked during BEARISH regime
3. Slot allocation matches regime state
4. Late window (2:30-3) requires BEARISH to allow PUTs
5. BULLISH regime blocks PUTs outside afternoon window

### Integration Tests

1. Scanner emits PUT signal during morning when regime is BEARISH
2. Scanner blocks PUT signal during morning when regime is BULLISH
3. Slot limits enforced — 7th PUT blocked when 6 already open in BEARISH
4. PUT V5 config correct per time window

## Settings

```
ENABLE_DYNAMIC_PUTS: bool = False  # gate behind flag
PUT_SLOTS_BULLISH: int = 2
PUT_SLOTS_BEARISH: int = 6
PUT_SLOTS_CHOPPY: int = 3
CALL_SLOTS_BULLISH: int = 6
CALL_SLOTS_BEARISH: int = 2
CALL_SLOTS_CHOPPY: int = 3
MORNING_PUT_SCALP_TARGET_PCT: float = 35.0
LATE_PUT_SCALP_TARGET_PCT: float = 20.0
LATE_PUT_MAX_HOLD_MIN: int = 30
```

## Dependencies

- Spec 06 (Intraday Regime Detector) — provides regime state
- Spec 07 (Extended Scan Window) — provides midday/afternoon window structure
- Existing PUT_SCALP_CONFIG in exit_v5/config.py

## Acceptance Criteria

- [ ] PUTs trade during morning selloffs when regime is BEARISH
- [ ] PUT exclusion list lifted during confirmed BEARISH regime
- [ ] Slot allocation dynamically adjusts with regime
- [ ] Choppy regime reduces total concurrent trades
- [ ] Backtest shows positive PUT P&L from dynamic expansion
- [ ] Combined CALL+PUT drawdown stays under 10%
- [ ] Feature gated behind ENABLE_DYNAMIC_PUTS
