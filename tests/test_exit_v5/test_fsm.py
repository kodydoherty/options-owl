"""Tests for ExitFSM v5 — category-aware, DTE-aware exits.

Matches v5 category-aware logic:
- 5min grace (not 45min)
- Scalp requires underlying confirmation
- Checkpoint: 30% drop + underlying against (0DTE only)
- Graduated stop: underlying-based, not time-based
- Profit target: index 0DTE at 30%
- Theta exit: 0DTE bleed + multi-day timer
- Category-aware adaptive trail tiers
"""

from datetime import datetime, timedelta

from options_owl.risk.exit_v5.config import TickerCategory, V5Config
from options_owl.risk.exit_v5.fsm import ExitFSM, FSMState, TradeState
from options_owl.risk.exit_v5.types import ExitReason


def _now_et(hour: int = 10, minute: int = 30) -> datetime:
    return datetime(2026, 4, 28, hour, minute, 0)


def _make_state(
    entry_premium: float = 1.00,
    contracts: int = 5,
    ticker: str = "AAPL",
    entry_time: datetime | None = None,
    option_type: str = "call",
    **kwargs,
) -> TradeState:
    return TradeState(
        trade_id=1, ticker=ticker, option_type=option_type,
        entry_premium=entry_premium,
        entry_time=entry_time or _now_et(10, 0),
        contracts=contracts,
        peak_premium=entry_premium,
        **kwargs,
    )


# ── State transitions ───────────────────────────────────────────────────

class TestFSMStateTransitions:

    def test_starts_in_grace(self):
        fsm = ExitFSM(V5Config())
        state = _make_state()
        fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 0) + timedelta(seconds=30))
        assert state.state == FSMState.GRACE

    def test_grace_lasts_5_minutes(self):
        """v5: grace period is 5 minutes."""
        fsm = ExitFSM(V5Config())
        state = _make_state()
        result = fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 4))
        assert state.state == FSMState.GRACE
        assert result.reason == ExitReason.HOLD

    def test_grace_backstop_fires_on_catastrophic_loss(self):
        """Backstop fires even during grace — a -95% trade should NOT sit for 5min."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_premium=4.62)
        # 2min into grace, premium at $0.20 = -95.7% > 65% backstop
        result = fsm.evaluate(state, 0.20, 0.15, 0.25, _now_et(10, 2))
        assert state.state == FSMState.GRACE
        assert result.should_exit
        assert result.reason == ExitReason.HARD_STOP

    def test_grace_holds_moderate_loss(self):
        """Grace still holds on moderate losses below backstop."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_premium=2.00)
        # 2min into grace, premium at $1.60 = -20% < 30% backstop
        result = fsm.evaluate(state, 1.60, 1.55, 1.65, _now_et(10, 2))
        assert not result.should_exit

    def test_grace_to_developing(self):
        """After 5min with low gain → DEVELOPING."""
        fsm = ExitFSM(V5Config())
        state = _make_state()
        fsm.evaluate(state, 1.10, 1.10, 1.20, _now_et(10, 6))
        assert state.state == FSMState.DEVELOPING

    def test_developing_to_trailing(self):
        """After 5min with peak gain >= 40% → TRAILING."""
        fsm = ExitFSM(V5Config())
        state = _make_state()
        state.peak_premium = 1.50
        fsm.evaluate(state, 1.40, 1.40, 1.50, _now_et(10, 6))
        assert state.state == FSMState.TRAILING


# ── Category assignment ──────────────────────────────────────────────────

class TestCategoryAssignment:

    def test_highvol_category(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="TSLA")
        fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 0) + timedelta(seconds=30))
        assert state.category == TickerCategory.HIGH_VOL

    def test_index_category(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="SPY")
        fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 0) + timedelta(seconds=30))
        assert state.category == TickerCategory.INDEX

    def test_standard_category(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="AAPL")
        fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 0) + timedelta(seconds=30))
        assert state.category == TickerCategory.STANDARD


# ── Profit target (index 0DTE only) ─────────────────────────────────────

class TestProfitTarget:

    def test_index_0dte_takes_profit_at_30pct(self):
        """Index 0DTE: lock gains at 30%."""
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="SPY", entry_underlying_price=500.0)
        result = fsm.evaluate(state, 1.32, 1.30, 1.34, _now_et(10, 6),
                              current_underlying=501.0)
        assert result.should_exit
        assert result.reason == ExitReason.PROFIT_TARGET

    def test_non_index_no_profit_target(self):
        """Non-index tickers don't get profit target."""
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="AAPL", entry_underlying_price=200.0)
        result = fsm.evaluate(state, 1.35, 1.33, 1.37, _now_et(10, 6),
                              current_underlying=201.0)
        assert not result.should_exit or result.reason != ExitReason.PROFIT_TARGET

    def test_multiday_index_no_profit_target(self):
        """Multi-day index trades don't get profit target."""
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="SPY", entry_underlying_price=500.0,
                            dte=1, expiry_date="2026-04-29")
        result = fsm.evaluate(state, 1.35, 1.33, 1.37, _now_et(10, 6),
                              current_underlying=501.0)
        assert not result.should_exit or result.reason != ExitReason.PROFIT_TARGET


# ── Scalp trail (underlying-aware) ──────────────────────────────────────

class TestScalpTrail:

    def test_0dte_scalp_fires_without_confirm(self):
        """0DTE: scalp fires when underlying doesn't confirm."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        state.peak_premium = 1.25
        # underlying flat (0.1% up) → doesn't confirm (+0.2% needed)
        result = fsm.evaluate(state, 1.08, 1.08, 1.15, _now_et(10, 10),
                              current_underlying=100.1)
        assert result.should_exit
        assert result.reason == ExitReason.SCALP_TRAIL

    def test_0dte_scalp_holds_with_confirm(self):
        """0DTE: scalp holds when underlying confirms."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        state.peak_premium = 1.25
        # underlying up 0.5% → confirms the call direction
        # current=1.16 stays above soft trail floor (1.15 = 1.00 + 0.25*0.60)
        result = fsm.evaluate(state, 1.16, 1.16, 1.20, _now_et(10, 10),
                              current_underlying=100.5)
        assert not result.should_exit

    def test_multiday_scalp_needs_against(self):
        """Multi-day: scalp only fires with underlying against."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")
        state.peak_premium = 1.25
        # underlying flat → not against → hold
        # current=1.16 stays above soft trail floor (1.15)
        result = fsm.evaluate(state, 1.16, 1.16, 1.20, _now_et(10, 10),
                              current_underlying=100.1)
        assert not result.should_exit


# ── Graduated stops (underlying-based) ──────────────────────────────────

class TestGraduatedStops:

    def test_0dte_checkpoint_before_confirmed(self):
        """0DTE: checkpoint (30%) fires before confirmed stop (35%) when underlying against."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        # down 36%, underlying -0.6% (against for call)
        result = fsm.evaluate(state, 0.64, 0.64, 0.70, _now_et(10, 10),
                              current_underlying=99.4)
        assert result.should_exit
        assert result.reason == ExitReason.CHECKPOINT_CUT

    def test_0dte_backstop_at_65pct(self):
        """0DTE: backstop at 65% when underlying NOT against."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        # down 66%, underlying flat (not against)
        result = fsm.evaluate(state, 0.34, 0.34, 0.40, _now_et(10, 10),
                              current_underlying=100.1)
        assert result.should_exit
        assert result.reason == ExitReason.HARD_STOP

    def test_0dte_holds_at_moderate_drop_not_against(self):
        """0DTE: 20% drop but underlying not against → hold."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        result = fsm.evaluate(state, 0.80, 0.80, 0.85, _now_et(10, 10),
                              current_underlying=100.1)
        assert not result.should_exit

    def test_multiday_wider_stops(self):
        """Multi-day uses wider stops (30% tight, 50% backstop)."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")
        # down 25%, underlying against → 25 < 30% tight → hold
        result = fsm.evaluate(state, 0.75, 0.75, 0.80, _now_et(10, 10),
                              current_underlying=99.4)
        assert not result.should_exit


# ── Checkpoint cut (0DTE only, underlying-confirmed) ────────────────────

class TestCheckpointCut:

    def test_fires_when_down_15_and_against(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        # down 18% with underlying against → fires checkpoint (>15%)
        result = fsm.evaluate(state, 0.82, 0.82, 0.85, _now_et(10, 20),
                              current_underlying=99.4)
        assert result.should_exit
        assert result.reason == ExitReason.CHECKPOINT_CUT

    def test_holds_when_not_against(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        # down 18% but underlying not against → hold
        result = fsm.evaluate(state, 0.82, 0.82, 0.85, _now_et(10, 20),
                              current_underlying=100.1)
        assert not result.should_exit

    def test_disabled_for_multiday(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")
        # down 20%, underlying against → checkpoint disabled for multiday, but below backstop (50%)
        result = fsm.evaluate(state, 0.80, 0.80, 0.85, _now_et(10, 20),
                              current_underlying=99.0)
        assert not result.should_exit


# ── Soft trail ──────────────────────────────────────────────────────────

class TestSoftTrail:

    def test_fires_after_grace(self):
        """Soft trail fires after 5min grace. Floor = entry + (peak-entry)*0.60."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        state.peak_premium = 1.30  # 30% peak
        # floor = 1.00 + 0.30*0.60 = 1.18, current 1.10 < floor
        # underlying up 0.5% → confirms call, so scalp won't fire
        result = fsm.evaluate(state, 1.10, 1.10, 1.20, _now_et(10, 6),
                              current_underlying=100.5)
        assert result.should_exit
        assert result.reason == ExitReason.SOFT_TRAIL


# ── Adaptive trail ──────────────────────────────────────────────────────

class TestAdaptiveTrail:

    def test_fires_after_grace(self):
        """Adaptive trail fires after 5min grace."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0)
        state.peak_premium = 1.80  # 80% peak
        # Standard/Index tiers: active at 30%, trail 35%
        # 80% peak, drop 42% from peak. underlying up 1% confirms → no scalp
        result = fsm.evaluate(state, 1.05, 1.05, 1.10, _now_et(10, 6),
                              current_underlying=101.0)
        assert result.should_exit
        assert result.reason == ExitReason.ADAPTIVE_TRAIL

    def test_highvol_wider_trail(self):
        """High-vol ticker gets wider adaptive trail."""
        fsm = ExitFSM(V5Config())
        state = _make_state(ticker="TSLA", entry_underlying_price=200.0)
        state.peak_premium = 1.80  # 80% peak
        # High-vol active: peak >= 40%, trail >= 50%
        # drop = (1.80 - 1.05) / 1.80 = 41.7% < 50% → hold
        result = fsm.evaluate(state, 1.05, 1.05, 1.10, _now_et(10, 6),
                              current_underlying=202.0)
        assert not result.should_exit


# ── Theta exit ─────────────────────────────────────────────────────────

class TestThetaExit:

    def test_0dte_bleed_fires(self):
        """0DTE: theta bleed fires at 120min+ and down but below backstop.

        Note: with tight stops (30% backstop), theta_bleed can only fire
        when the loss is gradual enough to not trigger backstop earlier.
        Using a custom config with wider backstop to isolate the test.
        """
        from dataclasses import replace as dc_replace
        wide_cfg = dc_replace(V5Config(), backstop_0dte_pct=65.0)
        fsm = ExitFSM(wide_cfg)
        state = _make_state(entry_underlying_price=100.0)
        result = fsm.evaluate(state, 0.68, 0.68, 0.75, _now_et(12, 5),
                              current_underlying=100.1)
        assert result.should_exit
        assert result.reason == ExitReason.THETA_BLEED

    def test_multiday_timer_fires(self):
        """Multi-day: theta timer fires at 180min+ and down 15%."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")
        # 3+ hours later, down 20%
        result = fsm.evaluate(state, 0.80, 0.80, 0.85, _now_et(13, 5),
                              current_underlying=100.1)
        assert result.should_exit
        assert result.reason == ExitReason.THETA_TIMER

    def test_multiday_timer_holds_if_not_down(self):
        """Multi-day: theta timer doesn't fire if not down enough."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, dte=1, expiry_date="2026-04-29")
        # 3+ hours later but only down 10% (< 15%)
        result = fsm.evaluate(state, 0.90, 0.90, 0.95, _now_et(13, 5),
                              current_underlying=100.1)
        assert result.reason != ExitReason.THETA_TIMER


# ── EOD cutoff ──────────────────────────────────────────────────────────

class TestEODCutoff:

    def test_0dte_fires(self):
        fsm = ExitFSM(V5Config())
        state = _make_state()
        result = fsm.evaluate(state, 1.10, 1.10, 1.20, _now_et(15, 45),
                              minutes_to_close=15.0)
        assert result.should_exit
        assert result.reason == ExitReason.EOD_CUTOFF

    def test_multiday_no_eod(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(dte=1, expiry_date="2026-04-29")
        result = fsm.evaluate(state, 1.10, 1.10, 1.20, _now_et(15, 45),
                              minutes_to_close=15.0)
        assert result.reason != ExitReason.EOD_CUTOFF


# ── Guards ──────────────────────────────────────────────────────────────

class TestGuards:

    def test_zero_entry_premium_force_exits(self):
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_premium=0.0)
        result = fsm.evaluate(state, 1.00, 1.00, 1.10, _now_et(10, 30))
        assert result.should_exit
        assert result.reason == ExitReason.HARD_STOP


# ── Put options ─────────────────────────────────────────────────────────

class TestPutOptions:

    def test_put_underlying_up_is_against(self):
        """For puts, underlying moving UP is 'against'."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, option_type="put")
        # down 36%, underlying UP 0.6% (against for put)
        result = fsm.evaluate(state, 0.64, 0.64, 0.70, _now_et(10, 10),
                              current_underlying=100.6)
        assert result.should_exit
        assert result.reason == ExitReason.CHECKPOINT_CUT

    def test_put_underlying_down_confirms(self):
        """For puts, underlying moving DOWN confirms the thesis."""
        fsm = ExitFSM(V5Config())
        state = _make_state(entry_underlying_price=100.0, option_type="put")
        state.peak_premium = 1.25  # peaked at +25%
        # current=1.16 stays above soft trail floor (1.15 = 1.00 + 0.25*0.60)
        result = fsm.evaluate(state, 1.16, 1.16, 1.20, _now_et(10, 10),
                              current_underlying=99.5)
        assert not result.should_exit
