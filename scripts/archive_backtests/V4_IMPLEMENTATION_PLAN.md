# Exit Engine v4 — Implementation Plan

## Guiding Principles

1. **Zero-downtime migration** — v3 keeps running on all owlets until v4 is proven
2. **Feature flag switch** — `EXIT_ENGINE=v4` per owlet, so we can A/B test
3. **Tests before code** — write tests for each module first, then implement to pass
4. **No shared mutable state** — all state is in `TradeState` dataclass, passed explicitly
5. **Pure functions** — every computation is a pure function (input → output), no globals
6. **Every decision is logged** — state transitions, trail computations, exit triggers

---

## Phase 1: Foundation (no production impact)

Write all new files. Nothing touches existing code. All owlets stay on v3.

### 1.1 Config — `options_owl/risk/exit_v4/config.py`

Typed, validated configuration. All v2.2 parameters in one place.

```
V4Config (frozen dataclass)
├── hard_stop_pct: float = 0.30
├── trail_activate_gain_pct: float = 35.0
├── trail_tiers: list[TrailTier]          # §6 default tiers
├── runner_trail_tiers: list[TrailTier]   # §6 runner (T3) tiers
├── underlying_trail_tiers: list[...]     # §5
├── atm_milestones: list[MilestoneLock]   # §7 ATM locks
├── house_money_floors: list[...]         # §12 ATM floors
├── otm_trail_tiers: list[TrailTier]      # §8 OTM tiers
├── otm_milestones: list[MilestoneLock]   # §8 OTM locks
├── otm_spike_threshold_pct: float        # §9
├── otm_spike_lock_fraction: float        # §9
├── theta_curve: ThetaCurveConfig         # §10
├── soft_trail: SoftTrailConfig           # §11
├── ticker_multipliers: dict[str, float]  # §14
├── theta_timer: ThetaTimerConfig         # §15
├── defensive: DefensiveConfig            # §16
├── post_stop_cooldown_sec: int           # §17
└── from_settings(settings) -> V4Config   # bridge from current Settings
```

**Tests: `tests/test_exit_v4/test_config.py`**
- Default config has all v2.2 values
- from_settings() correctly maps Settings → V4Config
- Trail tiers are sorted descending by min_gain_pct
- Invalid configs raise ValidationError

### 1.2 Trail Computation — `options_owl/risk/exit_v4/trail.py`

All pure functions. No side effects. The most critical module — gets the most tests.

```python
def get_trail_pct(gain_pct: float, tiers: list[TrailTier]) -> float
    """Look up trail width from tiered table. First match wins (tiers sorted desc)."""

def theta_curve_multiplier(now_et: datetime, market_close_et: datetime, cfg: ThetaCurveConfig) -> float
    """§10: Returns multiplier in [floor, 1.0] based on time remaining."""

def apply_trail_multipliers(base_trail_pct: float, ticker: str, is_morning: bool,
                            score: float | None, cfg: V4Config) -> tuple[float, float]
    """§14: Returns (effective_trail_pct, effective_whip_prob).
    Multiplier benefits whip resistance; giveback capped at 1.2x."""

def compute_trail_stop(peak_premium: float, entry_premium: float,
                       ticker: str, is_morning: bool, score: float | None,
                       now_et: datetime, market_close_et: datetime,
                       cfg: V4Config) -> float
    """Full trail stop computation: tier lookup → multipliers → theta curve → stop price."""
```

**Tests: `tests/test_exit_v4/test_trail.py`**
- get_trail_pct returns correct tier for each gain level (boundaries!)
- get_trail_pct returns widest tier when below all thresholds
- theta_curve_multiplier at 10:00 AM → ~1.0
- theta_curve_multiplier at 3:30 PM → ~0.40 (floor)
- theta_curve_multiplier at 2:00 PM → ~0.66
- apply_trail_multipliers: NVDA + morning + score 92 → mult capped at 2.0
- apply_trail_multipliers: IWM (no ticker mult) → mult = 1.0
- giveback_mult capped at 1.20
- effective_trail_pct capped at 0.45
- compute_trail_stop integrates all three correctly
- Runner tiers used for T3 tranche (wider than default)

### 1.3 Defensive Layer — `options_owl/risk/exit_v4/defensive.py`

```python
def check_bar1_reverse(gain_pct: float, seconds_since_fill: float,
                       candle_1m: dict | None) -> ExitAction | None
    """§16.3: At fill+90s, if first 1m candle is reverse direction + volume spike,
    exit at -5%. Returns None if no action."""

def check_bid_disappearance(bid: float, bid_zero_since: float | None,
                            now_epoch: float, timeout_sec: float) -> tuple[ExitAction | None, float | None]
    """§16.1: If bid <= 0.01 for >= timeout_sec, exit. Returns (action, updated_bid_zero_since)."""

def get_stop_reference_price(bid: float, ask: float, minutes_to_close: float) -> float
    """§16.2: Use mid price when > 30min to close, bid when <= 30min."""
```

**Tests: `tests/test_exit_v4/test_defensive.py`**
- bar1_reverse: returns None before 90s
- bar1_reverse: returns None after 150s (window closed)
- bar1_reverse: returns exit at -5% with reverse candle
- bar1_reverse: returns None when no volume spike
- bid_disappearance: returns None when bid > 0.01
- bid_disappearance: starts timer when bid goes to 0
- bid_disappearance: exits after timeout_sec of zero bid
- bid_disappearance: resets timer when bid recovers
- stop_reference: uses mid when > 30 min to close
- stop_reference: uses bid when <= 30 min to close

### 1.4 Milestones — `options_owl/risk/exit_v4/milestones.py`

```python
def check_atm_milestones(gain_pct: float, contracts: float,
                         milestones_locked: set[float],
                         milestones: list[MilestoneLock],
                         momentum_confirmed: bool) -> tuple[ExitAction | None, set[float]]
    """§7: Lock fraction at gain milestones. Returns (action, updated_locked_set)."""

def compute_house_money_floor(peak_gain: float, entry_premium: float,
                              current_floor: float,
                              floors: list[HouseMoneyFloor]) -> float
    """§12: Progressive monotonic stop. Only raises, never lowers."""

def check_otm_spike(gain_pct: float, contracts: float,
                    cfg: V4Config) -> ExitAction | None
    """§9: Lock 40% at +800% on OTM contracts."""
```

**Tests: `tests/test_exit_v4/test_milestones.py`**
- ATM milestone: locks 15% at +200%
- ATM milestone: doesn't re-lock same tier
- ATM milestone: requires momentum confirmation when configured
- House money: +500% → floor at +200%
- House money: monotonic — never decreases even if gain drops
- House money: highest trigger only (break after first match)
- OTM spike: locks 40% at +800%
- OTM spike: no action below threshold

### 1.5 Soft Trail — `options_owl/risk/exit_v4/soft_trail.py`

```python
def check_soft_trail(current_premium: float, entry_premium: float,
                     peak_premium: float, trail_activate_gain_pct: float,
                     cfg: SoftTrailConfig) -> ExitAction | None
    """§11: Floor at 50% of peak gain when peak in [15%, trail_activation).
    Returns exit action if premium falls below floor."""
```

**Tests: `tests/test_exit_v4/test_soft_trail.py`**
- No action when peak gain < 15%
- No action when peak gain >= trail_activate (trail handles it)
- Exit when premium falls to 50% of peak gain in the 15-35% band
- No action when premium is above floor
- Edge case: peak gain exactly 15% (boundary)
- Edge case: peak gain exactly 34.9% (just below activation)

### 1.6 FSM Engine — `options_owl/risk/exit_v4/fsm.py`

The core. Routes to the right checks based on current state.

```python
class FSMState(Enum):
    GRACE = "grace"
    DEVELOPING = "developing"
    TRAILING = "trailing"

@dataclass
class TradeState:
    """Mutable per-trade state. Created at fill, updated each poll cycle."""
    trade_id: int
    ticker: str
    score: float
    direction: str                    # "bullish" or "bearish"
    is_otm: bool
    entry_premium: float
    contracts: float
    fill_time: datetime
    # Updated each cycle
    peak_premium: float
    peak_underlying: float | None
    state: FSMState
    current_stop: float               # hard stop (raised by house-money)
    house_money_floor: float
    milestones_locked: set[float]
    locked_pnl: float
    bid_zero_since: float | None
    enrg_result: str | None
    premium_history: list[tuple[float, float]]  # (epoch, premium)

@dataclass(frozen=True)
class ExitAction:
    action: str          # "hold", "exit", "partial"
    reason: str          # e.g. "hard_stop", "trailing_stop", "sc_MILESTONE_200"
    qty_fraction: float  # 0.0 for hold/full exit, 0.15 for milestone lock
    debug: dict          # all intermediate values for logging

class ExitFSM:
    """Stateless evaluator. All state lives in TradeState."""

    def __init__(self, config: V4Config):
        self.cfg = config

    def evaluate(self, state: TradeState, current_premium: float,
                 now_et: datetime, underlying_price: float | None = None,
                 candle_data: dict | None = None,
                 quote: dict | None = None) -> ExitAction:
        """Single entry point. Returns what to do RIGHT NOW."""

        # 1. Update peak
        state.peak_premium = max(state.peak_premium, current_premium)

        # 2. Compute gains
        gain_pct = (current_premium - state.entry_premium) / state.entry_premium * 100
        peak_gain_pct = (state.peak_premium - state.entry_premium) / state.entry_premium * 100

        # 3. State transitions (before evaluation)
        self._maybe_transition(state, peak_gain_pct, now_et)

        # 4. Universal checks (all states)
        action = self._check_universal(state, current_premium, gain_pct, now_et, quote)
        if action: return action

        # 5. State-specific evaluation
        if state.state == FSMState.GRACE:
            return self._eval_grace(state, current_premium, gain_pct, now_et, candle_data)
        elif state.state == FSMState.DEVELOPING:
            return self._eval_developing(state, current_premium, gain_pct, peak_gain_pct,
                                         now_et, underlying_price, candle_data)
        elif state.state == FSMState.TRAILING:
            return self._eval_trailing(state, current_premium, gain_pct, peak_gain_pct,
                                       now_et, underlying_price, candle_data)

    def _check_universal(self, ...) -> ExitAction | None:
        """EOD cutoff, bid disappearance — checked in every state."""

    def _eval_grace(self, ...) -> ExitAction:
        """GRACE: only bar-1 reverse can exit. Otherwise HOLD."""

    def _eval_developing(self, ...) -> ExitAction:
        """DEVELOPING: hard stop, ENRG, soft trail, theta timer."""

    def _eval_trailing(self, ...) -> ExitAction:
        """TRAILING: tiered trail, theta-curve, house-money, milestones."""

    def _maybe_transition(self, state, peak_gain_pct, now_et):
        """Check and execute state transitions. Logs every transition."""
```

**Tests: `tests/test_exit_v4/test_fsm.py`** (most critical)

State transitions:
- GRACE → DEVELOPING after 90 seconds
- DEVELOPING → TRAILING when peak gain hits 35%
- No backward transitions (TRAILING never goes back to DEVELOPING)

GRACE state:
- Returns HOLD for normal price movement
- Returns EXIT on bar-1 reverse with volume spike
- Does NOT check hard stop (too early, let it settle)

DEVELOPING state:
- Hard stop at -30% triggers EXIT
- Soft trail triggers when peak was 20% and current drops to 10%
- ENRG fires when negative (delegates to candle_cache)
- Theta timer fires at 60min when no gain and no immunity
- Theta timer immune for score >= 92
- Theta timer immune for NVDA, TSLA, AMZN, AVGO, PLTR

TRAILING state:
- Trail stop computed correctly from peak
- Trail tightens via theta-curve near EOD
- Trail widened by ticker multipliers for NVDA/TSLA
- House-money floor raised at +100%/+200%/+500%
- House-money floor is monotonic (never decreases)
- Effective stop = max(trail_stop, house_money_floor, hard_stop)
- Milestone lock fires partial at +200%/+400%/+600%
- Milestone requires momentum confirmation
- No double-lock on same tier

### 1.7 Integration Tests — `tests/test_exit_v4/test_integration.py`

Full lifecycle simulations using synthetic premium paths:

- **Big winner**: Entry $1.00 → peak $5.00 → exit via trail at ~$4.00
  - Verify: GRACE → DEVELOPING → TRAILING, milestone locks at +200%/+400%, house-money floor at +200%
- **Small winner**: Entry $1.00 → peak $1.45 → reversal → exit via soft trail at ~$1.22
  - Verify: GRACE → DEVELOPING, soft trail captures 50% of peak gain
- **Loser**: Entry $1.00 → drops to $0.70 → exit via hard stop
  - Verify: GRACE → DEVELOPING, hard stop at -30%
- **Bar-1 reverse**: Entry $1.00 → immediately drops to $0.93 with reverse candle
  - Verify: exits in GRACE state at -5%
- **Theta timer**: Entry $1.00 → flat at $0.98 for 65 minutes → exit
  - Verify: theta timer fires at 60min, no gain = exit
- **Theta immune**: Same scenario but ticker=NVDA → no exit, holds
- **EOD cutoff**: Entry $1.00 at 3:30 PM → exits at 3:45 PM regardless
- **Runner with theta-curve**: Entry at 10AM, peaks at $3.00 at 3PM
  - Verify: trail tightened by theta-curve near EOD
- **MSTR ticker multiplier**: Verify no ticker multiplier (not in list)
- **Morning power TSLA**: Verify wider trail (morning + ticker mult)

### 1.8 Backtest Validation — `scripts/backtest_v4_vs_v22.py`

Run v4 FSM on the same 123 trades. Verify results match the v2.2 simulation from our earlier backtest. If they diverge, the FSM has a bug.

---

## Phase 2: Position Monitor v4 (still no production impact)

### 2.1 `options_owl/execution/position_monitor_v4.py`

Clean rewrite. Only pulls what it needs from the old monitor:
- Premium fetching (Polygon WS → REST → yfinance → delta fallback)
- Underlying price fetching
- Candle cache integration
- Reconciliation logic
- Alert system

**Does NOT carry over:**
- `_bounce_states`, `_thesis_cut_states` (v3 artifacts)
- `_premium_histories` as module-level dict (moved into TradeState)
- Gate context building (replaced by FSM.evaluate() call)

```python
# Core loop (simplified):
async def _monitor_loop(client, paper_trader, settings, ...):
    fsm = ExitFSM(V4Config.from_settings(settings))
    trade_states: dict[int, TradeState] = {}

    while True:
        open_trades = await get_open_trades(db_path)

        for trade in open_trades:
            # Get or create state
            state = trade_states.get(trade["id"]) or _init_trade_state(trade)
            trade_states[trade["id"]] = state

            # Fetch inputs
            premium = await _get_current_premium(trade, ...)
            underlying = await _get_underlying_price(trade["ticker"], ...)
            candles = _get_candle_data(trade["ticker"], ...)
            quote = _get_quote(trade, ...)

            # Evaluate
            action = fsm.evaluate(state, premium, _now_et(), underlying, candles, quote)

            # Log EVERY evaluation (DEBUG for hold, INFO for exit/partial)
            _log_evaluation(trade, state, action)

            # Act
            if action.action == "exit":
                await _close_trade(trade, action, ...)
                del trade_states[trade["id"]]
            elif action.action == "partial":
                await _partial_close(trade, action, ...)

        # Cleanup states for trades no longer open
        _cleanup_stale_states(trade_states, open_trades)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
```

### 2.2 Integration into `main.py`

```python
# In on_ready():
exit_engine = getattr(settings, "EXIT_ENGINE", "v3")
if exit_engine == "v4":
    from options_owl.execution.position_monitor_v4 import start_position_monitor
    logger.info("EXIT ENGINE: v4 (FSM)")
else:
    from options_owl.execution.position_monitor import start_position_monitor
    logger.info("EXIT ENGINE: v3 (gate pipeline)")
```

### 2.3 Settings Addition

One new setting:
```python
EXIT_ENGINE: str = "v3"  # "v3" or "v4" — which exit engine to use
```

---

## Phase 3: Staging Validation (paper trading only)

### 3.1 Deploy v4 on ONE owlet in paper mode

Pick owlet-vinny ($500 portfolio, lowest risk). Set in docker-compose:
```yaml
owlet-vinny:
  environment:
    - EXIT_ENGINE=v4
    - PAPER_TRADE=true      # force paper even if normally live
```

### 3.2 Run for 2-3 trading sessions

Monitor via persisted logs:
```bash
grep 'EXIT_FSM' journal/owlet-vinny/logs/options_owl_$(date +%Y-%m-%d).log
grep 'TRAIL\|MILESTONE\|HOUSE_MONEY' journal/owlet-vinny/logs/options_owl_$(date +%Y-%m-%d).log
```

Compare v4 decisions against v3 (owlet-kody still on v3):
```bash
# Side-by-side: same signal, different exit decisions
./scripts/compare_exit_engines.sh 2026-05-01
```

### 3.3 Acceptance Criteria (must ALL pass before Phase 4)

- [ ] All 123-trade backtest matches v2.2 simulation within 2%
- [ ] Zero crashes during 2+ trading sessions
- [ ] FSM state transitions logged correctly for every trade
- [ ] Milestone locks fire at correct thresholds
- [ ] House-money floors never decrease
- [ ] Trail computation matches v2.2 spec at 10AM, 2PM, and 3:30PM
- [ ] ENRG still fires correctly when position goes negative
- [ ] Bar-1 reverse fires within first 90 seconds only
- [ ] Theta timer respects immunity (ticker + score)
- [ ] EOD cutoff fires at 3:45 PM ET
- [ ] Position reconciliation still works
- [ ] Webull orders execute correctly (fill verification)
- [ ] Alerts fire on exit failures

---

## Phase 4: Live Rollout (gradual)

### 4.1 Single owlet live

Switch owlet-vinny to live with v4:
```yaml
owlet-vinny:
  environment:
    - EXIT_ENGINE=v4
    - PAPER_TRADE=false
```

Run for 1 week. Compare P&L against v3 owlets.

### 4.2 All owlets

Once confirmed, switch all owlets:
```yaml
# All bots
- EXIT_ENGINE=v4
```

### 4.3 Archive v3

After 2 weeks stable on v4:
```bash
# Rename old files (don't delete — rollback insurance)
git mv options_owl/risk/pipeline.py options_owl/risk/_archive_pipeline_v3.py
git mv options_owl/execution/position_monitor.py options_owl/execution/_archive_position_monitor_v3.py
```

Remove v3 settings from `settings.py` (50+ settings → clean).

---

## Phase 5: Cleanup

- Remove v3 settings from settings.py
- Remove v3 test files (test_exit_pipeline.py, test_partial_profits.py adjustments)
- Update CLAUDE.md with v4 architecture
- Update trade-log.sh if log format changes

---

## File Inventory

### New Files (Phase 1)
| File | Lines (est) | Purpose |
|---|---|---|
| `options_owl/risk/exit_v4/__init__.py` | 20 | Public API exports |
| `options_owl/risk/exit_v4/config.py` | 150 | V4Config + sub-configs |
| `options_owl/risk/exit_v4/trail.py` | 120 | Trail computation (pure functions) |
| `options_owl/risk/exit_v4/defensive.py` | 80 | Bar-1 reverse, bid disappearance |
| `options_owl/risk/exit_v4/milestones.py` | 90 | Milestone locks, house-money floors |
| `options_owl/risk/exit_v4/soft_trail.py` | 40 | 15-35% gain band floor |
| `options_owl/risk/exit_v4/fsm.py` | 300 | FSM engine |
| **Total new logic** | **~800** | |

### New Tests (Phase 1)
| File | Tests (est) | Coverage |
|---|---|---|
| `tests/test_exit_v4/test_config.py` | 8 | Config defaults, validation |
| `tests/test_exit_v4/test_trail.py` | 15 | Trail tiers, theta-curve, multipliers |
| `tests/test_exit_v4/test_defensive.py` | 10 | Bar-1 reverse, bid disappearance |
| `tests/test_exit_v4/test_milestones.py` | 8 | Milestone locks, house-money |
| `tests/test_exit_v4/test_soft_trail.py` | 6 | Soft trail band |
| `tests/test_exit_v4/test_fsm.py` | 25 | State transitions, all states |
| `tests/test_exit_v4/test_integration.py` | 10 | Full trade lifecycles |
| **Total tests** | **~82** | |

### New Files (Phase 2)
| File | Lines (est) | Purpose |
|---|---|---|
| `options_owl/execution/position_monitor_v4.py` | 400 | Clean monitor using FSM |
| `scripts/compare_exit_engines.sh` | 30 | Side-by-side comparison tool |

---

## Logging Convention

Every log line from v4 is prefixed with `EXIT_FSM` for easy grepping.

```
# State transition (INFO)
EXIT_FSM [trade_42] STATE: DEVELOPING -> TRAILING | peak_gain=+37.2% trigger=trail_activation

# Hold decision (DEBUG — only in daily log file, not JSON)
EXIT_FSM [trade_42] HOLD | state=TRAILING gain=+45.1% peak=+52.3% trail_stop=$1.84 current=$2.12

# Exit decision (INFO)
EXIT_FSM [trade_42] EXIT | reason=trailing_stop state=TRAILING | premium=$1.84 stop=$1.87 peak=$2.52 entry=$1.00 pnl=+$84

# Partial close (INFO)
EXIT_FSM [trade_42] PARTIAL | reason=sc_MILESTONE_200 | lock 15% (1 contract) at $3.00 | remaining=6

# Trail debug (DEBUG)
EXIT_FSM [trade_42] TRAIL | tier=+200%->25% base=0.250 theta=0.72 ticker_mult=1.50(NVDA) eff=0.216 stop=$1.97

# House-money floor update (INFO)
EXIT_FSM [trade_42] HOUSE_MONEY | gain=+215% -> floor raised to +80% ($1.80) | prev_floor=$1.30
```

---

## Debugging Playbook

### "Why did this trade exit?"
```bash
grep 'EXIT_FSM \[trade_42\]' journal/owlet-kody/logs/options_owl_2026-05-01.log
```
Shows full lifecycle: state transitions, every trail computation, the exact tick that triggered exit.

### "Why didn't this trade exit earlier?"
```bash
grep 'EXIT_FSM \[trade_42\] HOLD\|TRAIL' journal/owlet-kody/logs/options_owl_2026-05-01.log | tail -20
```
Shows the trail stop vs current premium on each cycle leading up to exit.

### "What's the FSM doing right now?"
```bash
grep 'EXIT_FSM.*HOLD' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log | tail -5
```

### "Are milestone locks working?"
```bash
grep 'MILESTONE\|HOUSE_MONEY' journal/owlet-kody/logs/options_owl_$(date +%Y-%m-%d).log
```

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| v4 has a bug that loses money | Feature flag: instant rollback to v3 by changing one env var |
| FSM gets stuck in wrong state | State transitions logged at INFO; monitor for state durations |
| Trail computation wrong | Trail debug logged at DEBUG; 15 unit tests on trail math |
| Milestone double-locks | milestones_locked set prevents re-locking; tested |
| House-money floor decreases | monotonic check in code + test; floor only uses max() |
| Premium fetch fails | Same 4-source fallback as v3 (WS → REST → yfinance → delta) |
| Webull order fails | Same executor + retry + alert system as v3 |
| State lost on restart | TradeState rebuilt from DB on startup (entry, peak from trade record) |

---

## Build Order

This is the exact sequence of implementation steps:

```
1. config.py + test_config.py          → run tests → green
2. trail.py + test_trail.py            → run tests → green
3. defensive.py + test_defensive.py    → run tests → green
4. milestones.py + test_milestones.py  → run tests → green
5. soft_trail.py + test_soft_trail.py  → run tests → green
6. fsm.py + test_fsm.py               → run tests → green
7. test_integration.py                 → run tests → green
8. backtest_v4_vs_v22.py              → verify matches v2.2 simulation
9. position_monitor_v4.py             → manual test locally
10. Add EXIT_ENGINE setting            → deploy to vinny (paper)
11. Monitor 2-3 sessions              → acceptance criteria
12. Go live on vinny                   → monitor 1 week
13. Go live on all owlets              → monitor 2 weeks
14. Archive v3                         → cleanup
```

Each step is a commit. Each commit passes all tests. No step depends on a future step.
