# Spec 09: Conviction-Based Position Sizing

## Goal

Replace flat 85% budget allocation with a tiered system that sizes up on
high-conviction setups and sizes down on marginal ones. Uses regime alignment,
ML confidence, and time-of-day as inputs.

## Problem

Today every qualifying trade (score >= 78) gets identical 85% budget. This means:
- A 0.95 confidence pattern in a strong bullish trend gets the same size as
  a 0.86 confidence pattern in choppy conditions
- Morning trades (highest WR historically) get the same size as afternoon trades
- Regime-aligned trades (CALL during bullish) get the same as counter-trend

## Design

### Conviction Score

Compute a conviction multiplier (0.4 - 1.0) from three factors:

```python
def compute_conviction_multiplier(
    ml_confidence: float,
    regime: RegimeState,
    direction: str,
    minute: int,
) -> float:
    """Returns budget multiplier based on setup quality."""

    # Base: ML model confidence
    if ml_confidence >= 0.95:
        base = 1.0   # exceptional pattern
    elif ml_confidence >= 0.90:
        base = 0.85  # strong pattern
    else:
        base = 0.70  # qualifying but not exceptional

    # Regime alignment bonus/penalty
    if regime == BULLISH and direction == "call":
        regime_mult = 1.0   # aligned
    elif regime == BEARISH and direction == "put":
        regime_mult = 1.0   # aligned
    elif regime == CHOPPY:
        regime_mult = 0.70  # reduced size in chop
    else:
        regime_mult = 0.50  # counter-trend — half size or skip

    # Time-of-day factor (morning has highest WR)
    if minute <= 60:      # 9:30-10:30
        time_mult = 1.0
    elif minute <= 120:   # 10:30-11:30
        time_mult = 0.90
    elif minute <= 210:   # 11:30-1:00
        time_mult = 0.80
    else:                 # 1:00+
        time_mult = 0.75

    return max(0.40, min(1.0, base * regime_mult * time_mult))
```

### Sizing Integration

Modify `vinny_strategy.py:score_to_contracts()`:

```python
# Current: flat 85%
scaled_target = target_per_trade * 0.85

# New: conviction-based
conviction = compute_conviction_multiplier(ml_confidence, regime, direction, minute)
scaled_target = target_per_trade * conviction
```

### Example Sizing (Kody $23K, 8 slots)

| Scenario | ML Conf | Regime | Direction | Time | Mult | Budget |
|---|---|---|---|---|---|---|
| Best case | 0.96 | BULLISH | CALL | 9:45 | 1.00 | $2,156 |
| Strong morning | 0.91 | BULLISH | CALL | 10:15 | 0.85 | $1,833 |
| Midday aligned | 0.88 | BEARISH | PUT | 12:30 | 0.56 | $1,207 |
| Choppy afternoon | 0.87 | CHOPPY | PUT | 1:30 | 0.40 | $863 |
| Counter-trend | 0.90 | BEARISH | CALL | 11:00 | 0.40 | $863 |

## Testing Plan

### Unit Tests

1. `compute_conviction_multiplier` returns correct values for all combinations:
   - High conf + aligned + morning = 1.0
   - Low conf + choppy + afternoon = 0.40 (floor)
   - Counter-trend always <= 0.50
2. `score_to_contracts` uses conviction multiplier correctly
3. Floor of 0.40 is enforced (minimum 1 contract still)
4. Maximum of 1.0 cap (no over-sizing)

### Backtest Validation

Run `backtest_combined.py` comparing:
1. Flat 85% (current)
2. Conviction-based sizing
3. Conviction-based with regime detector

Metrics: Total P&L, Sharpe ratio, max drawdown, average win size vs average loss size
Target: Higher Sharpe (better risk-adjusted returns), similar or better total P&L

### Risk Validation

- Verify max position size never exceeds MAX_POSITION_PCT
- Verify total exposure never exceeds MAX_PORTFOLIO_RISK_PCT
- Counter-trend trades should rarely hit max concurrent slots

## Settings

```
ENABLE_CONVICTION_SIZING: bool = False  # gate behind flag
CONVICTION_HIGH_CONF_THRESHOLD: float = 0.95
CONVICTION_MEDIUM_CONF_THRESHOLD: float = 0.90
CONVICTION_CHOPPY_MULT: float = 0.70
CONVICTION_COUNTER_TREND_MULT: float = 0.50
CONVICTION_FLOOR: float = 0.40
```

## Dependencies

- Spec 06 (Intraday Regime Detector) — provides regime state
- Existing sizing in vinny_strategy.py

## Acceptance Criteria

- [ ] High-conviction trades get more capital than marginal trades
- [ ] Counter-trend trades are automatically downsized
- [ ] Choppy market reduces all position sizes
- [ ] Morning trades sized larger than afternoon
- [ ] All existing risk caps (MAX_POSITION_PCT, MAX_PORTFOLIO_RISK_PCT) still enforced
- [ ] Backtest Sharpe ratio improves vs flat sizing
- [ ] Feature gated behind ENABLE_CONVICTION_SIZING
