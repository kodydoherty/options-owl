# Spec 06: Intraday Regime Detector

## Goal

Rule-based (no ML) real-time regime classification that determines market direction
every 5 minutes using existing candle cache data. Gates which direction (CALL/PUT)
the ML scanner is allowed to trade.

## Problem

Today the bot statically assigns time windows — CALLs in the morning, PUTs in the
afternoon. When the market reverses mid-day, open CALLs bleed out and the bot can't
pivot to PUTs until the fixed window arrives.

## Design

### Regime States

| State | Meaning | Trading Action |
|---|---|---|
| BULLISH | Trending up, momentum confirmed | CALLs full size, PUTs blocked |
| BEARISH | Trending down, momentum confirmed | PUTs full size, CALLs blocked |
| CHOPPY | Oscillating, no clear direction | Both directions at 60% size, scalp only |

### Detection Rules (SPY/QQQ as proxy)

Evaluate every 5 minutes using existing `candle_cache.py` data:

```
BULLISH when ALL of:
  - Price > VWAP (5m)
  - EMA9 > EMA21 (5m chart)
  - RSI(14) > 50
  - ADX(14) > 20 (trending, not ranging)

BEARISH when ALL of:
  - Price < VWAP (5m)
  - EMA9 < EMA21 (5m chart)
  - RSI(14) < 50
  - ADX(14) > 20

CHOPPY when:
  - ADX(14) < 20 (no trend)
  - OR price oscillating around VWAP (crosses 3+ times in last hour)
```

### Hysteresis (Prevent Flip-Flopping)

- Regime must hold for 2 consecutive 5-minute checks (10 min) before switching
- Once confirmed, regime persists for minimum 15 minutes before re-evaluation
- Exception: "hard reversal" override — SPY drops 0.5%+ in 15 min → immediate BEARISH

### Interface

```python
# options_owl/risk/regime_detector.py

class RegimeState(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    CHOPPY = "choppy"

class RegimeDetector:
    def __init__(self, candle_cache):
        self.state = RegimeState.CHOPPY  # start neutral
        self.state_since: datetime  # when current state was confirmed
        self.pending_state: RegimeState | None  # candidate awaiting confirmation
        self.pending_since: datetime | None

    async def update(self) -> RegimeState:
        """Called every 5 min. Returns current confirmed regime."""

    def allows_direction(self, direction: str) -> bool:
        """Can the scanner emit a CALL or PUT signal right now?"""

    def get_size_multiplier(self) -> float:
        """1.0 for trending, 0.6 for choppy."""
```

### Integration Points

1. **bot_runner.py `_run_ml_scan_loop`**: Before evaluating each ticker, check
   `regime.allows_direction(direction)`. Skip if blocked.
2. **vinny_strategy.py `score_to_contracts`**: Multiply budget by
   `regime.get_size_multiplier()` for choppy regimes.
3. **position_monitor.py**: When regime flips from BULLISH→BEARISH, tighten
   adaptive trails on open CALLs by 40% (see spec 08).

### Data Requirements

All data already available in `candle_cache.py`:
- 5m candles with VWAP, EMA9, EMA21, RSI(14)
- ADX needs to be added (uses existing candle highs/lows/closes)

## Testing Plan

### Unit Tests

1. Regime classification with synthetic candle data:
   - All bullish conditions met → BULLISH
   - All bearish conditions met → BEARISH
   - ADX < 20 → CHOPPY
   - Mixed signals (price > VWAP but EMA9 < EMA21) → CHOPPY

2. Hysteresis:
   - Single bullish reading doesn't flip state
   - Two consecutive bullish readings → confirmed BULLISH
   - State holds for 15 min minimum even if one reading is neutral
   - Hard reversal (0.5% drop) overrides hysteresis

3. Direction gating:
   - BULLISH blocks PUTs, allows CALLs
   - BEARISH blocks CALLs, allows PUTs
   - CHOPPY allows both at reduced size

### Backtest Validation

Add regime detector to `backtest_combined.py`:
- Compare trades WITH regime gating vs WITHOUT
- Measure: avoided losers (CALLs during bearish), missed winners, net P&L impact
- Target: reduce losing trades by 30%+ without losing >10% of winners

### Integration Test

- Mock candle cache with known regime data
- Verify scanner skips CALL signals during BEARISH regime
- Verify scanner allows PUT signals during BEARISH regime
- Verify size reduction during CHOPPY regime

## Settings

```
ENABLE_REGIME_DETECTOR: bool = True
REGIME_CHECK_INTERVAL_MIN: int = 5
REGIME_HYSTERESIS_CHECKS: int = 2  # consecutive readings to confirm
REGIME_MIN_HOLD_MIN: int = 15
REGIME_HARD_REVERSAL_PCT: float = 0.5  # SPY drop % for immediate flip
REGIME_CHOPPY_SIZE_MULT: float = 0.6
```

## Acceptance Criteria

- [ ] RegimeDetector produces correct state from candle data
- [ ] Hysteresis prevents flip-flopping (min 2 readings + 15 min hold)
- [ ] Hard reversal override works within 5 minutes
- [ ] Scanner respects regime gating for CALL/PUT direction
- [ ] Position sizing reduced in CHOPPY regime
- [ ] All data comes from existing candle_cache (no new API calls)
- [ ] Backtest shows net positive impact on combined CALL+PUT P&L
- [ ] Unit tests cover all regime transitions and edge cases
