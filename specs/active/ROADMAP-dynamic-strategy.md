# Dynamic Strategy Roadmap

## Overview

Five interconnected features that transform the bot from a fixed-schedule morning
scanner into a full-day adaptive trading system. Each feature is gated behind a
flag and can be enabled independently, but they compound when combined.

## Implementation Order (Dependencies)

```
Spec 06: Regime Detector (foundation — everything depends on this)
    |
    +---> Spec 07: Extended Scan Window (uses regime to gate midday entries)
    |
    +---> Spec 08: Regime Stop Tightening (uses regime flips to protect positions)
    |
    +---> Spec 09: Conviction Sizing (uses regime alignment for size multiplier)
    |
    +---> Spec 10: Dynamic PUT Expansion (uses regime to unlock PUTs in morning)
```

Build order: 06 -> 08 -> 09 -> 10 -> 07

Rationale: Regime detector first. Stop tightening (08) protects existing trades
immediately. Conviction sizing (09) improves risk-adjusted returns on existing
trades. Dynamic PUTs (10) adds new trade types. Extended window (07) is last
because it has the most moving parts and benefits from all other features.

## Expected Impact (Cumulative)

| Feature | Additional Trades/Day | P&L Impact | Risk |
|---|---|---|---|
| Regime Detector alone | 0 (filter only) | Fewer losing trades | Very low |
| + Stop Tightening | 0 | Save 5-10% on reversals | Very low |
| + Conviction Sizing | 0 | Better risk-adjusted returns | Low |
| + Dynamic PUTs | +0.5-1.0 | Natural hedge on down days | Low |
| + Extended Window | +1.0-1.5 | More opportunities | Medium |
| **Total** | **+1.5-2.5/day** | **+30-50% P&L** | **Low-Medium** |

## Backtest Validation Strategy

Each feature must pass a standalone backtest AND a combined backtest:

1. **Standalone**: Feature ON vs OFF, everything else at current production config
2. **Combined**: All features ON vs current production config
3. **Stress test**: Combined features on worst historical days (3+ losing trades)

All backtests must show:
- MaxDD < 10% (hard requirement)
- Sharpe >= current production Sharpe
- No single feature makes combined performance worse

## Rollout Plan

1. Build + unit test all features (flags OFF)
2. Backtest each feature independently
3. Backtest combined
4. Deploy with all flags OFF
5. Enable one feature per day, monitor live performance
6. Full system live after 5 trading days of validation
