"""Comprehensive exit pipeline integration and E2E tests.

Validates that:
1. Every exit gate fires at the correct threshold and does NOT fire below threshold
2. The ML model is fully hooked up and invoked for every signal
3. No dead logic / empty spots in the exit pipeline
4. Gates evaluate in correct priority order
5. Phase trails, profit lock, sizing, settings, anti-chase, and consecutive loser all work
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from options_owl.risk.pipeline import (
    DEFAULT_EXIT_GATES,
    EXIT_GATE_TO_REASON,
    AdaptiveTimeTightenExitGate,
    AdaptiveTrailingStopExitGate,
    DecelExitGate,
    EODExitGate,
    GateResult,
    MLSellExitGate,
    NoMomentumExitGate,
    ProfitLockExitGate,
    ProfitRetraceExitGate,
    StopLossExitGate,
    Target1ExitGate,
    Target2ExitGate,
    Target3ExitGate,
    Target4ExitGate,
    Target5ExitGate,
    ThetaBleedExitGate,
    TimeDecayZoneExitGate,
    TimeExpiryExitGate,
    TrailingStopExitGate,
    BEClampExitGate,
    SoftTrailExitGate,
    is_grace_active,
    DollarTrailExitGate,
    ProfitFloorExitGate,
    BounceFadeExitGate,
    ThesisCutExitGate,
    run_exit_pipeline,
)
from options_owl.risk.vinny_strategy import (
    check_anti_chase,
    check_consecutive_loser_pause,
    check_theta_bleed,
    check_time_decay_no_new_high,
    compute_vix_adjusted_trail,
    evaluate_adaptive_trail,
    is_time_decay_zone,
    score_to_contracts,
)

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except ImportError:
    ET = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Fake settings — mirrors the real Settings class attributes
# ---------------------------------------------------------------------------


class _S:
    """Lightweight fake settings with all pipeline-relevant attributes."""

    def __init__(self, **kw):
        defaults = {
            # Stop loss
            "PREMIUM_STOP_ENABLED": True,
            "PREMIUM_STOP_PCT": 50.0,
            "STOP_GRACE_PERIOD_MINUTES": 0,
            "ENABLE_SMART_GRACE": False,  # off by default in tests (test explicitly)
            "ENABLE_UNDERLYING_STOP": False,
            "MIN_UNDERLYING_STOP_PCT": 0.5,
            # Trailing stop
            "ENABLE_TRAILING_STOP": True,
            "TRAILING_STOP_ACTIVATION_PCT": 30.0,
            "TRAILING_STOP_DROP_PCT": 40.0,
            # No momentum
            "ENABLE_NO_MOMENTUM_EXIT": True,
            "NO_MOMENTUM_MINUTES": 45,
            "NO_MOMENTUM_MIN_GAIN_PCT": 5.0,
            # ML
            "ENABLE_ML_EXIT": False,
            "ML_OVERRIDE_TARGETS": False,
            "ML_OVERRIDE_MIN_FUTURE_PNL": 5.0,
            # Vinny
            "ENABLE_VINNY_STRATEGY": True,
            "TIME_DECAY_HOLD_MINUTES": 45.0,
            "TIME_DECAY_AFTERNOON_HOUR": 15,
            "TIME_DECAY_AFTERNOON_MINUTE": 30,
            "TIME_DECAY_STALE_MINUTES": 10.0,
            # Theta bleed
            "THETA_BLEED_HOLD_MINUTES": 45.0,
            "THETA_BLEED_MAX_LOSS_PCT": 30.0,
            # Theta decay legacy
            "ENABLE_THETA_DECAY_EXIT": False,
            # Velocity exit (legacy)
            "ENABLE_VELOCITY_EXIT": False,
            "VELOCITY_DROP_PCT": 12.0,
            "VELOCITY_WINDOW_MINUTES": 4,
            # Dollar trail (replaces velocity exit)
            "ENABLE_DOLLAR_TRAIL": False,  # off by default in tests (test explicitly)
            "DOLLAR_TRAIL_ACTIVATION_PCT": 10.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 10.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 25.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 5.0,
            # Profit retrace
            "ENABLE_PROFIT_RETRACE": False,  # off by default in tests (test explicitly)
            "PROFIT_RETRACE_PCT": 35.0,
            "PROFIT_RETRACE_MIN_GAIN_PCT": 10.0,
            # Decel exit
            "ENABLE_DECEL_EXIT": False,  # off by default in tests (test explicitly)
            "DECEL_SHORT_WINDOW": 5,
            "DECEL_LONG_WINDOW": 15,
            "DECEL_THRESHOLD": -3.0,
            "DECEL_MIN_GAIN_PCT": 5.0,
            "DECEL_MIN_HOLD_SECONDS": 480,
            # BE clamp (v2.2 §4)
            "ENABLE_BE_CLAMP": False,  # off by default in tests (test explicitly)
            "BE_CLAMP_ACTIVATION_PCT": 15.0,
            # Soft trail (v2.2 §11)
            "ENABLE_SOFT_TRAIL": False,  # off by default in tests (test explicitly)
            "SOFT_TRAIL_MIN_PCT": 15.0,
            "SOFT_TRAIL_MAX_PCT": 35.0,
            "SOFT_TRAIL_FLOOR_PCT": 50.0,
            # Catastrophic stop (v3: disabled, replaced by bounce-fade)
            "ENABLE_CATASTROPHIC_STOP": False,
            "CATASTROPHIC_STOP_PCT": 45.0,
            # Profit floor (v3)
            "ENABLE_PROFIT_FLOOR": False,  # off by default in tests (test explicitly)
            "PROFIT_FLOOR_ACTIVATION_PCT": 15.0,
            "PROFIT_FLOOR_RATCHET_PCT": 60.0,
            # Bounce-fade (v3)
            "ENABLE_BOUNCE_FADE": False,  # off by default in tests (test explicitly)
            "BOUNCE_FADE_WATCH_PCT": 50.0,
            "BOUNCE_FADE_MIN_RECOVERY_PCT": 10.0,
            # Thesis cut (v3)
            "ENABLE_THESIS_CUT": False,  # off by default in tests (test explicitly)
            "THESIS_CUT_THRESHOLD_PCT": 40.0,
            "THESIS_CUT_LOOKBACK_TICKS": 8,
            "THESIS_CUT_NEW_LOW_EXIT": 3,
            "THESIS_CUT_BOUNCE_HOLD_PCT": 5.0,
            "THESIS_CUT_MIN_TICKS": 4,
            "THESIS_CUT_TIME_URGENCY_MIN": 30.0,
            "THESIS_CUT_TIME_CUT_DROP_PCT": 40.0,
            "BOUNCE_FADE_PCT": 15.0,
            # Profit lock
            "ENABLE_PROFIT_LOCK": True,
            "PROFIT_LOCK_TIERS": "80:25,150:70,250:150",
            # Adaptive trail (v2.1)
            "ENABLE_ADAPTIVE_TRAIL": False,  # off by default in tests (test explicitly)
            "ADAPTIVE_TRAIL_ACTIVATION_PCT": 40.0,
            "ADAPTIVE_TRAIL_ACTIVE_WIDTH": 35.0,
            "ADAPTIVE_TRAIL_RUNNER_THRESHOLD": 150.0,
            "ADAPTIVE_TRAIL_RUNNER_WIDTH": 45.0,
            "ADAPTIVE_TRAIL_MOONSHOT_THRESHOLD": 400.0,
            "ADAPTIVE_TRAIL_MOONSHOT_WIDTH": 30.0,
            # Time tighten
            "ENABLE_TIME_TIGHTEN": True,
            "TIME_TIGHTEN_AFTER_MINUTES": 60.0,
            "TIME_TIGHTEN_FACTOR": 0.7,
            # Anti-chase
            "ANTI_CHASE_MAX_MOVE_PCT": 0.3,
            # Consecutive loser
            "CONSECUTIVE_LOSER_MAX": 2,
            "CONSECUTIVE_LOSER_PAUSE_MINUTES": 15.0,
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def _base_trade(**overrides):
    """Produce a dict that looks like a paper_trades DB row."""
    # Opened 30 min ago (past grace period, before decay zone)
    opened = datetime(2026, 4, 10, 10, 0, tzinfo=ET)
    t = {
        "id": 1,
        "ticker": "SPY",
        "option_type": "call",
        "strike": 560.0,
        "entry_price": 560.0,
        "premium_per_contract": 2.50,
        "contracts": 5,
        "status": "open",
        "stop_price": 555.0,
        "target_1": 565.0,
        "target_2": 570.0,
        "target_3": None,
        "target_4": None,
        "target_5": None,
        "exit_by": None,
        "opened_at": opened.isoformat(),
        "expiry_date": "2026-04-10",
        "mfe_premium": 2.50,
        "mae_premium": 2.50,
        "last_target_hit": None,
        "last_new_high_at": None,
    }
    t.update(overrides)
    return t


def _ctx(trade=None, price=562.0, premium=3.50, settings=None, now=None, **extra):
    """Produce a standard exit pipeline context dict."""
    if now is None:
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
    return {
        "trade": trade or _base_trade(),
        "current_price": price,
        "exit_premium": premium,
        "now_et": now,
        "settings": settings or _S(),
        **extra,
    }


# ===================================================================
# 1. GATE FIRING TESTS — each gate fires at threshold, holds below
# ===================================================================


class TestPremiumStopGate:
    """premium_stop fires when premium drops >= PREMIUM_STOP_PCT from entry."""

    @pytest.mark.asyncio
    async def test_fires_at_threshold(self):
        # Entry 2.50, premium 1.20 → drop 52% >= 50% threshold
        # Also hits catastrophic stop (45%) — either way it's FAIL
        ctx = _ctx(premium=1.20)
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_below_threshold(self):
        # Entry 2.50, premium 1.50 → drop 40% < 50%
        ctx = _ctx(premium=1.50)
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_grace_period_blocks_normal_stop(self):
        # Just opened — within grace period. Drop is -30% (below catastrophic 45%).
        now = datetime(2026, 4, 10, 10, 2, tzinfo=ET)
        opened = (now - timedelta(minutes=2)).isoformat()
        trade = _base_trade(opened_at=opened)
        ctx = _ctx(trade=trade, premium=1.75, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=5))
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_catastrophic_stop_bypasses_grace_period(self):
        # Just opened — within grace period, but -60% drop triggers catastrophic
        # (only when ENABLE_CATASTROPHIC_STOP=True — disabled by default in v3).
        now = datetime(2026, 4, 10, 10, 2, tzinfo=ET)
        opened = (now - timedelta(minutes=2)).isoformat()
        trade = _base_trade(opened_at=opened)
        ctx = _ctx(trade=trade, premium=0.50, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=5,
                               ENABLE_CATASTROPHIC_STOP=True))
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "CATASTROPHIC" in r.reason

    @pytest.mark.asyncio
    async def test_catastrophic_stop_disabled_by_default(self):
        # When ENABLE_CATASTROPHIC_STOP=False (v3 default), deep drops don't
        # trigger catastrophic stop — bounce_fade handles them instead.
        now = datetime(2026, 4, 10, 10, 2, tzinfo=ET)
        opened = (now - timedelta(minutes=2)).isoformat()
        trade = _base_trade(opened_at=opened)
        ctx = _ctx(trade=trade, premium=0.50, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=5))
        r = await StopLossExitGate().evaluate(ctx)
        # Should PASS (grace period), not FAIL — catastrophic stop disabled
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()


class TestProfitLockGate:
    """profit_lock fires when peak gain reached a tier and current falls to lock floor."""

    @pytest.mark.asyncio
    async def test_fires_when_below_lock_floor(self):
        # Entry 2.00, MFE 5.60 → peak gain 180% → tier "150:70" applies → lock at +70%
        # Current must be <= 2.00 * (1 + 0.70) = 3.40 to fire
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=5.60)
        ctx = _ctx(trade=trade, premium=3.30)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Profit lock" in r.reason

    @pytest.mark.asyncio
    async def test_holds_above_lock_floor(self):
        # Same peak +180%, current 3.60 → gain +80% > lock floor +70%
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=5.60)
        ctx = _ctx(trade=trade, premium=3.60)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_no_tier_reached(self):
        # Peak gain only 50% — no tier reached
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=3.00)
        ctx = _ctx(trade=trade, premium=2.80)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "hasn't reached any tier" in r.reason


class TestTimeDecayStaleGate:
    """time_decay_zone fires when in decay zone and no new premium high for N minutes."""

    @pytest.mark.asyncio
    async def test_fires_when_stale(self):
        now = datetime(2026, 4, 10, 11, 0, tzinfo=ET)
        opened = now - timedelta(minutes=50)  # 50 min → in decay zone
        last_high = now - timedelta(minutes=12)  # 12 min > 10 min stale
        trade = _base_trade(
            opened_at=opened.isoformat(),
            mfe_premium=3.50,
            last_new_high_at=last_high.isoformat(),
        )
        ctx = _ctx(trade=trade, premium=3.00, now=now)
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_when_fresh_high(self):
        now = datetime(2026, 4, 10, 11, 0, tzinfo=ET)
        opened = now - timedelta(minutes=50)
        last_high = now - timedelta(minutes=3)  # only 3 min ago
        trade = _base_trade(
            opened_at=opened.isoformat(),
            mfe_premium=3.50,
            last_new_high_at=last_high.isoformat(),
        )
        ctx = _ctx(trade=trade, premium=3.00, now=now)
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_holds_when_not_in_decay_zone(self):
        # Only 20 min held — not in decay zone yet
        now = datetime(2026, 4, 10, 10, 20, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(opened_at=opened.isoformat(), mfe_premium=3.50)
        ctx = _ctx(trade=trade, premium=3.00, now=now)
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestThetaBleedGate:
    """theta_bleed fires when held > N min AND losing >= threshold.

    Note: ThetaBleedExitGate calls check_theta_bleed without passing now,
    so it uses datetime.now() (naive). We must use naive opened_at strings.
    """

    @pytest.mark.asyncio
    async def test_fires_when_held_long_and_losing(self):
        now_et = datetime(2026, 4, 10, 11, 0, tzinfo=ET)
        opened = now_et - timedelta(minutes=50)  # > 45 min
        # Entry 2.50, exit 1.70 → loss 32% > 30%
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
        )
        ctx = _ctx(trade=trade, premium=1.70, now=now_et)
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Theta bleed" in r.reason

    @pytest.mark.asyncio
    async def test_holds_when_profitable(self):
        now_et = datetime(2026, 4, 10, 11, 0, tzinfo=ET)
        opened = now_et - timedelta(minutes=50)
        trade = _base_trade(opened_at=opened.isoformat(), premium_per_contract=2.50)
        ctx = _ctx(trade=trade, premium=3.00, now=now_et)
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_holds_when_early(self):
        now_et = datetime(2026, 4, 10, 10, 20, tzinfo=ET)
        opened = now_et - timedelta(minutes=20)  # < 45 min
        trade = _base_trade(opened_at=opened.isoformat(), premium_per_contract=2.50)
        ctx = _ctx(trade=trade, premium=1.50, now=now_et)
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestNoMomentumGate:
    """no_momentum fires when held > N min with insufficient gain."""

    @pytest.mark.asyncio
    async def test_fires_after_time_with_no_gain(self):
        now = datetime.now()
        opened = now - timedelta(minutes=50)  # > 45 min
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
        )
        ctx = _ctx(trade=trade, premium=2.55, now=now.replace(tzinfo=ET))
        # gain = 2%, need 5%
        r = await NoMomentumExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "No momentum" in r.reason

    @pytest.mark.asyncio
    async def test_holds_when_gaining(self):
        now = datetime.now()
        opened = now - timedelta(minutes=50)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
        )
        ctx = _ctx(trade=trade, premium=2.80, now=now.replace(tzinfo=ET))
        # gain = 12% > 5%
        r = await NoMomentumExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_holds_when_early(self):
        now = datetime.now()
        opened = now - timedelta(minutes=10)  # < 45 min
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
        )
        ctx = _ctx(trade=trade, premium=2.20, now=now.replace(tzinfo=ET))
        r = await NoMomentumExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestExpirySafetyEODGate:
    """eod_cutoff fires after 15:45 ET."""

    @pytest.mark.asyncio
    async def test_fires_after_cutoff(self):
        now = datetime(2026, 4, 10, 15, 50, tzinfo=ET)
        ctx = _ctx(now=now)
        r = await EODExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "EOD" in r.reason

    @pytest.mark.asyncio
    async def test_holds_before_cutoff(self):
        now = datetime(2026, 4, 10, 14, 0, tzinfo=ET)
        ctx = _ctx(now=now)
        r = await EODExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestTrailingStopGate:
    """trailing_stop fires after activation when premium drops from peak."""

    @pytest.mark.asyncio
    async def test_fires_when_drops_from_peak(self):
        # Entry 2.50, MFE 3.50 → peak gain 40% >= activation 30%
        # Current 2.00 → drop from peak (3.50 - 2.00) / 3.50 = 42.9% >= 40%
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.50)
        ctx = _ctx(trade=trade, premium=2.00)
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Trailing stop" in r.reason

    @pytest.mark.asyncio
    async def test_holds_when_not_activated(self):
        # MFE only 10% above entry — not activated
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=2.75)
        ctx = _ctx(trade=trade, premium=2.50)
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===================================================================
# 2. ML INTEGRATION TESTS
# ===================================================================


class TestMLIntegration:
    """Verify predict_sell is called with correct args and result influences exit."""

    @pytest.mark.asyncio
    async def test_ml_sell_invoked_and_triggers_exit(self):
        """When ENABLE_ML_EXIT=True and model says sell, gate should FAIL."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=True,
            sell_probability=0.75,
            expected_future_pnl=-5.0,
            reason="ML combo: P(sell)=0.75, E[future]=-5.0%",
            model_used="generic",
        )

        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(opened_at=opened.isoformat())
        ctx = _ctx(
            trade=trade,
            now=now,
            settings=_S(ENABLE_ML_EXIT=True),
        )

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal) as mock_ps:
            r = await MLSellExitGate().evaluate(ctx)
            assert r.result == GateResult.FAIL
            assert "ML combo" in r.reason

            # Verify call contract
            mock_ps.assert_called_once()
            call_kwargs = mock_ps.call_args
            assert call_kwargs.kwargs["ticker"] == "SPY"
            assert call_kwargs.kwargs["entry_premium"] == 2.50
            assert call_kwargs.kwargs["current_premium"] == 3.50
            assert call_kwargs.kwargs["is_call"] is True

    @pytest.mark.asyncio
    async def test_ml_hold_passes_gate(self):
        """When model says hold, gate should PASS."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.20,
            expected_future_pnl=8.0,
            reason="hold",
            model_used="SPY",
        )

        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(opened_at=opened.isoformat())
        ctx = _ctx(trade=trade, now=now, settings=_S(ENABLE_ML_EXIT=True))

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            r = await MLSellExitGate().evaluate(ctx)
            assert r.result == GateResult.PASS
            assert "ML hold" in r.reason

    @pytest.mark.asyncio
    async def test_ml_disabled_skips(self):
        """When ENABLE_ML_EXIT=False, gate should SKIP without calling model."""
        ctx = _ctx(settings=_S(ENABLE_ML_EXIT=False))
        with patch("options_owl.risk.ml_exit.predict_sell") as mock_ps:
            r = await MLSellExitGate().evaluate(ctx)
            assert r.result == GateResult.SKIP
            mock_ps.assert_not_called()

    @pytest.mark.asyncio
    async def test_ml_override_converts_target_to_scale_out(self):
        """When ML holds with high confidence, target hits become [ML_HOLD] scale-outs."""
        from options_owl.risk.ml_exit import MLSellSignal

        ml_hold = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=12.0,
            reason="hold",
            model_used="SPY",
        )

        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            target_1=563.0,
        )
        settings = _S(
            ENABLE_ML_EXIT=True,
            ML_OVERRIDE_TARGETS=True,
            ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
            # Disable gates that might fire first
            PREMIUM_STOP_ENABLED=False,
            ENABLE_TRAILING_STOP=False,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,

            ENABLE_TIME_TIGHTEN=False,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_NO_MOMENTUM_EXIT=False,
            ENABLE_THETA_DECAY_EXIT=False,
        )
        # Price above T1 for a call
        ctx = _ctx(trade=trade, price=564.0, premium=3.50, now=now, settings=settings)

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=ml_hold):
            reason, desc = await run_exit_pipeline(ctx)
            assert reason == "t1_hit"
            assert desc.startswith("[ML_HOLD]")


# ===================================================================
# 3. GATE ORDERING TEST
# ===================================================================


class TestGateOrdering:
    """Verify gates are evaluated in priority order: safety > risk > profit > time."""

    def test_default_exit_gate_order(self):
        """The DEFAULT_EXIT_GATES list must follow the documented priority (v2.2 Phase 1)."""
        names = [g().name for g in DEFAULT_EXIT_GATES]
        # Tier 1: Emergency/defensive
        assert names[0] == "enrg"
        assert names[1] == "stop_loss"
        assert names[2] == "be_clamp"  # v2.2 §4
        # Tier 2: ML and scale-out
        assert names[3] == "ml_sell"
        assert names[4] == "tranche_scaleout"
        # Tier 3: Trail modifier
        assert names[5] == "volume_peak"
        # Tier 4: Trail gates (primary exit for winners)
        assert names[6] == "soft_trail"  # v2.2 §11
        assert names[7] == "adaptive_trailing_stop"
        # Trails before time-based exits (Vince's key insight)
        assert names.index("adaptive_trailing_stop") < names.index("theta_bleed")
        assert names.index("adaptive_trailing_stop") < names.index("no_momentum")
        assert names.index("adaptive_trailing_stop") < names.index("decel_exit")
        # Targets T5 > T4 > T3 > T2 > T1
        t_indices = [names.index(f"target_{i}") for i in range(5, 0, -1)]
        assert t_indices == sorted(t_indices)
        # EOD is last or near-last
        assert names.index("eod_cutoff") > names.index("target_1")
        # Theta bleed before time decay zone
        assert names.index("theta_bleed") < names.index("time_decay_zone")

    def test_every_exit_gate_has_reason_mapping(self):
        """Every exit gate name must appear in EXIT_GATE_TO_REASON."""
        for gate_cls in DEFAULT_EXIT_GATES:
            gate = gate_cls()
            assert gate.name in EXIT_GATE_TO_REASON, (
                f"Gate '{gate.name}' missing from EXIT_GATE_TO_REASON mapping"
            )

    @pytest.mark.asyncio
    async def test_stop_beats_target(self):
        """When both stop and target fire, stop (higher priority) wins."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=5.00,
            target_1=558.0,  # also hit for a call at price 556
            mfe_premium=5.00,
        )
        settings = _S(
            PREMIUM_STOP_ENABLED=True,
            PREMIUM_STOP_PCT=50.0,
            STOP_GRACE_PERIOD_MINUTES=0,
            ENABLE_ML_EXIT=False,
            ENABLE_TRAILING_STOP=False,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,

            ENABLE_TIME_TIGHTEN=False,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_NO_MOMENTUM_EXIT=False,
            ENABLE_THETA_DECAY_EXIT=False,
        )
        # Premium down 60% from entry → stop fires
        ctx = _ctx(trade=trade, price=556.0, premium=2.00, now=now, settings=settings)
        reason, _ = await run_exit_pipeline(ctx)
        assert reason == "stop_hit"


# ===================================================================
# 5. PROFIT LOCK RATCHET TEST
# ===================================================================


class TestProfitLockRatchet:
    """Verify that the floor only ratchets UP with higher tiers."""

    @pytest.mark.asyncio
    async def test_highest_tier_takes_precedence(self):
        # Peak +260% → tier "250:150" applies (not 150:70 or 80:25)
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=3.60)
        # Current at +145% → below lock floor +150% → FIRE
        ctx = _ctx(trade=trade, premium=2.45)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "250" in r.reason  # tier 250 applied

    @pytest.mark.asyncio
    async def test_lower_tier_floor_is_lower(self):
        # Peak +100% → tier "80:25" applies (lock at +25%)
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=4.00)
        # Current at +30% → above lock floor +25% → HOLD
        ctx = _ctx(trade=trade, premium=2.60)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_custom_tier_parsing(self):
        # Custom tier string "50:20,100:50"
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=4.20)
        # Peak +110% → tier "100:50" → lock at +50%
        # Current at +40% → below lock +50% → FIRE
        settings = _S(PROFIT_LOCK_TIERS="50:20,100:50")
        ctx = _ctx(trade=trade, premium=2.80, settings=settings)
        r = await ProfitLockExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


# ===================================================================
# 6. SCORE-BASED SIZING TESTS
# ===================================================================


class TestScoreBasedSizing:
    """Verify score_to_contracts returns correct values (flat sizing above 78)."""

    def test_below_78_returns_zero(self):
        assert score_to_contracts(77) == 0
        assert score_to_contracts(50) == 0
        assert score_to_contracts(0) == 0

    def test_flat_fallback_without_cost(self):
        # Flat 85% mult → fallback = max(1, int(5*0.85)) = 4 for all qualifying scores
        assert score_to_contracts(150) == 4
        assert score_to_contracts(130) == 4
        assert score_to_contracts(110) == 4
        assert score_to_contracts(95) == 4
        assert score_to_contracts(78) == 4

    def test_dollar_target_sizing(self):
        # $10k balance, 75% risk cap, 4 concurrent → $1875 per slot
        # Flat 85% mult → $1593, 15% cap=$1500 → 7
        assert score_to_contracts(130, cost_per_contract=200, balance=10000) == 7

    def test_expensive_option_skipped(self):
        # $500/contract on $1k account → 15% cap=$150 < $500 → SKIP
        assert score_to_contracts(130, cost_per_contract=500, balance=1000) == 0

    def test_cheap_option_scales(self):
        # $5/contract on $100k, 4 slots → slot=$18750, 85%=$15937, 15% cap=$15k → 3000
        assert score_to_contracts(130, cost_per_contract=5, balance=100000) == 3000
        # Same for score 150 — flat sizing
        assert score_to_contracts(150, cost_per_contract=5, balance=100000) == 3000

    def test_max_position_pct_cap(self):
        # max_position_pct=5% is a hard ceiling
        # slot=$1875, 85%=$1593, raw=7, pos_cap(5%=$500)=2 → 2
        result = score_to_contracts(
            130, cost_per_contract=200, balance=10000, max_position_pct=5.0,
        )
        assert result == 2

    def test_zero_cost_uses_fallback(self):
        assert score_to_contracts(150, cost_per_contract=0, balance=10000) == 4


# ===================================================================
# 7. SETTINGS INTEGRATION
# ===================================================================


class TestSettingsIntegration:
    """Verify Settings class loads from env vars including PROFIT_LOCK_TIERS."""

    def test_settings_defaults(self, monkeypatch):
        """Verify Settings class can be instantiated and critical fields are set."""
        from options_owl.config.settings import Settings

        # Isolate from .env file by clearing relevant env vars
        for key in ("PREMIUM_STOP_PCT", "PORTFOLIO_SIZE", "MAX_CONCURRENT",
                     "ENABLE_GREEKS", "ENABLE_VIX_FILTER",
                     "WEBULL_KILL_SWITCH", "MAX_PORTFOLIO_RISK_PCT", "ENABLE_RISK_MANAGER",
                     "DATA_FEED_PROVIDER", "POLYGON_API_KEY", "ENABLE_POLYGON_WS",
                     "MAX_POSITION_PCT",
                     "MAX_LOSS_PER_TRADE_PCT"):
            monkeypatch.delenv(key, raising=False)

        s = Settings(DISCORD_TOKEN="test", _env_file=None)
        assert s.PAPER_TRADE is True
        assert s.PREMIUM_STOP_PCT == 30.0  # v2.1: tighter hard stop
        assert s.ENABLE_ML_EXIT is False  # ML disabled — actively harmful with insufficient training data
        assert s.ENABLE_VINNY_STRATEGY is True
        assert s.ENABLE_PROFIT_LOCK is False  # v3: superseded by profit_floor
        assert s.ENABLE_PROFIT_FLOOR is True  # v3: ratcheting profit floor
        assert s.ENABLE_BOUNCE_FADE is False  # v3: disabled (thesis_cut handles losses)
        assert s.ENABLE_THESIS_CUT is True  # v3: trend-confirmed loss cutting
        assert s.ENABLE_CATASTROPHIC_STOP is False  # v3: disabled, thesis_cut handles deep dips
        assert s.PREMIUM_STOP_ENABLED is True  # v2.2 §3: hard stop -30% always enabled
        assert s.ENABLE_PROFIT_RETRACE is False  # v3: superseded by soft_trail
        assert s.STOP_GRACE_PERIOD_MINUTES == 20  # 20min cap (smart grace ends it early)
        assert s.ENABLE_SMART_GRACE is True  # smart grace: underlying confirms direction
        assert s.ENABLE_BE_CLAMP is True  # v2.2 §4: breakeven clamp
        assert s.ENABLE_SOFT_TRAIL is True  # v2.2 §11: soft trail 15-35%

    def test_profit_lock_tiers_parsing(self):
        from options_owl.config.settings import Settings

        s = Settings(DISCORD_TOKEN="test", PROFIT_LOCK_TIERS="80:25,150:70,250:150")
        assert s.PROFIT_LOCK_TIERS == "80:25,150:70,250:150"
        # Verify parsing logic matches what ProfitLockExitGate does
        tiers = []
        for pair in s.PROFIT_LOCK_TIERS.split(","):
            threshold, lock = pair.strip().split(":")
            tiers.append((float(threshold), float(lock)))
        assert tiers == [(80.0, 25.0), (150.0, 70.0), (250.0, 150.0)]

    def test_guild_ids_parsing(self):
        from options_owl.config.settings import Settings

        s = Settings(DISCORD_TOKEN="test", DISCORD_GUILD_IDS="123,456,789")
        assert s.guild_ids == [123, 456, 789]

    def test_env_override(self, monkeypatch):
        from options_owl.config.settings import Settings

        monkeypatch.setenv("PREMIUM_STOP_PCT", "35.0")
        monkeypatch.setenv("DISCORD_TOKEN", "test")
        s = Settings()
        assert s.PREMIUM_STOP_PCT == 35.0


# ===================================================================
# 8. ANTI-CHASE TEST
# ===================================================================


class TestAntiChaseDetailed:
    """Verify rejection when underlying moves too far."""

    def test_within_threshold_passes(self):
        passed, _ = check_anti_chase(550.0, 550.50, max_move_pct=0.3)
        assert passed is True

    def test_beyond_threshold_fails(self):
        # 0.55% move > 0.3%
        passed, reason = check_anti_chase(550.0, 553.0, max_move_pct=0.3)
        assert passed is False
        assert "Anti-chase" in reason

    def test_exact_threshold_passes(self):
        # 0.3% move exactly = threshold → passes (> not >=)
        price = 550.0 * 1.003
        passed, _ = check_anti_chase(550.0, price, max_move_pct=0.3)
        assert passed is True  # equal to, not exceeding

    def test_downward_move_also_triggers(self):
        # Underlying drops — abs move matters
        passed, _ = check_anti_chase(550.0, 547.0, max_move_pct=0.3)
        assert passed is False

    def test_zero_alert_price_always_passes(self):
        passed, _ = check_anti_chase(0, 999.0)
        assert passed is True


# ===================================================================
# 9. CONSECUTIVE LOSER PAUSE TEST
# ===================================================================


class TestConsecutiveLoserPauseDetailed:
    """Verify cooldown works correctly."""

    def test_below_threshold_can_trade(self):
        can, _ = check_consecutive_loser_pause(1, None, max_consecutive=2)
        assert can is True

    def test_at_threshold_in_cooldown(self):
        now = datetime(2026, 4, 10, 10, 10)
        last = datetime(2026, 4, 10, 10, 5)  # 5 min ago < 15
        can, reason = check_consecutive_loser_pause(
            2, last, now, max_consecutive=2, pause_minutes=15.0,
        )
        assert can is False
        assert "cooling down" in reason

    def test_cooldown_expired_can_trade(self):
        now = datetime(2026, 4, 10, 10, 25)
        last = datetime(2026, 4, 10, 10, 5)  # 20 min ago > 15
        can, reason = check_consecutive_loser_pause(
            2, last, now, max_consecutive=2, pause_minutes=15.0,
        )
        assert can is True
        assert "expired" in reason

    def test_three_losses_still_pauses(self):
        now = datetime(2026, 4, 10, 10, 10)
        last = datetime(2026, 4, 10, 10, 8)
        can, _ = check_consecutive_loser_pause(
            3, last, now, max_consecutive=2, pause_minutes=15.0,
        )
        assert can is False

    def test_no_timestamp_always_can_trade(self):
        can, _ = check_consecutive_loser_pause(5, None, max_consecutive=2)
        assert can is True


# ===================================================================
# 10. END-TO-END SIGNAL FLOW TEST
# ===================================================================


class TestEndToEndExitPipeline:
    """A mock signal flowing through the full pipeline from context to exit decision."""

    @pytest.mark.asyncio
    async def test_full_pipeline_hold(self):
        """Trade in profit, no gates fire → HOLD."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=15)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=3.50,
        )
        settings = _S(
            ENABLE_ML_EXIT=False,
            PREMIUM_STOP_ENABLED=True,
            PREMIUM_STOP_PCT=50.0,
            STOP_GRACE_PERIOD_MINUTES=0,
        )
        ctx = _ctx(trade=trade, price=562.0, premium=3.20, now=now, settings=settings)
        reason, desc = await run_exit_pipeline(ctx)
        assert reason is None, f"Expected HOLD but got exit: {reason} — {desc}"

    @pytest.mark.asyncio
    async def test_full_pipeline_stop_loss(self):
        """Premium crashes → stop fires first."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=15)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=5.00,
            mfe_premium=5.00,
        )
        settings = _S(
            ENABLE_ML_EXIT=False,
            PREMIUM_STOP_ENABLED=True,
            PREMIUM_STOP_PCT=50.0,
            STOP_GRACE_PERIOD_MINUTES=0,
        )
        # Premium 2.00 → down 60% > 50%
        ctx = _ctx(trade=trade, price=555.0, premium=2.00, now=now, settings=settings)
        reason, desc = await run_exit_pipeline(ctx)
        assert reason == "stop_hit"

    @pytest.mark.asyncio
    async def test_full_pipeline_t1_hit(self):
        """Price hits T1 → partial exit."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=15)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=3.80,
            target_1=565.0,
        )
        settings = _S(
            ENABLE_ML_EXIT=False,
            PREMIUM_STOP_ENABLED=True,
            PREMIUM_STOP_PCT=50.0,
            STOP_GRACE_PERIOD_MINUTES=0,
            ENABLE_TRAILING_STOP=False,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,

            ENABLE_TIME_TIGHTEN=False,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_NO_MOMENTUM_EXIT=False,
            ENABLE_THETA_DECAY_EXIT=False,
        )
        ctx = _ctx(trade=trade, price=566.0, premium=3.80, now=now, settings=settings)
        reason, desc = await run_exit_pipeline(ctx)
        assert reason == "t1_hit"

    @pytest.mark.asyncio
    async def test_full_pipeline_eod_close(self):
        """After 15:45 ET → EOD exit fires."""
        now = datetime(2026, 4, 10, 15, 50, tzinfo=ET)
        opened = now - timedelta(hours=5)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=2.60,
        )
        settings = _S(
            ENABLE_ML_EXIT=False,
            PREMIUM_STOP_ENABLED=False,
            ENABLE_TRAILING_STOP=False,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,

            ENABLE_TIME_TIGHTEN=False,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_NO_MOMENTUM_EXIT=False,
            ENABLE_THETA_DECAY_EXIT=False,
        )
        ctx = _ctx(trade=trade, price=562.0, premium=2.55, now=now, settings=settings)
        reason, desc = await run_exit_pipeline(ctx)
        assert reason == "eod_expiry"

    @pytest.mark.asyncio
    async def test_full_pipeline_ml_sell_overrides_hold(self):
        """ML says sell → ml_sell fires before any target gate."""
        from options_owl.risk.ml_exit import MLSellSignal

        ml_signal = MLSellSignal(
            should_sell=True,
            sell_probability=0.80,
            expected_future_pnl=-8.0,
            reason="ML combo: P(sell)=0.80, E[future]=-8.0%",
            model_used="SPY",
        )

        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=20)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=3.00,
            target_1=563.0,
        )
        settings = _S(
            ENABLE_ML_EXIT=True,
            PREMIUM_STOP_ENABLED=False,
            ENABLE_TRAILING_STOP=False,
        )
        ctx = _ctx(trade=trade, price=564.0, premium=3.00, now=now, settings=settings)

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=ml_signal):
            reason, desc = await run_exit_pipeline(ctx)
            # ML fires before targets (priority 2 vs 13)
            assert reason == "ml_sell"

    @pytest.mark.asyncio
    async def test_pipeline_all_gates_evaluated_on_hold(self):
        """When no gate fires, verify we don't crash (all gates evaluated cleanly)."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=5)
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=2.60,
        )
        settings = _S(
            PREMIUM_STOP_ENABLED=True,
            PREMIUM_STOP_PCT=50.0,
            STOP_GRACE_PERIOD_MINUTES=10,  # in grace period
            ENABLE_ML_EXIT=False,
            ENABLE_TRAILING_STOP=False,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,

            ENABLE_TIME_TIGHTEN=False,
            ENABLE_VINNY_STRATEGY=False,
            ENABLE_NO_MOMENTUM_EXIT=False,
            ENABLE_THETA_DECAY_EXIT=False,
        )
        ctx = _ctx(trade=trade, price=562.0, premium=2.55, now=now, settings=settings)
        reason, desc = await run_exit_pipeline(ctx)
        assert reason is None


# ===================================================================
# Profit retrace gate
# ===================================================================


class TestProfitRetraceExitGate:
    """profit_retrace fires when trade gives back X% of profit in dormant zone."""

    def _retrace_settings(self, **kw):
        defaults = dict(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0)
        defaults.update(kw)
        return _S(**defaults)

    @pytest.mark.asyncio
    async def test_disabled(self):
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=False),
            trade=_base_trade(mfe_premium=3.0),
            premium=2.0,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_fires_above_adaptive_activation(self):
        """Peak gain >= 40% — retrace still protects (covers adaptive trail zone too)."""
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.5),  # +50%
            premium=1.1,  # gave back 80% of $0.50 profit — above 35% threshold
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL
        assert "retrace" in out.reason.lower()

    @pytest.mark.asyncio
    async def test_skip_below_min_gain(self):
        """Peak gain < min gain → retrace not active."""
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.05),  # +5%
            premium=0.95,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS
        assert "min" in out.reason

    @pytest.mark.asyncio
    async def test_fires_on_retrace(self):
        """Entry $1.00, peak $1.30 (+30%), retrace 35% of $0.30 profit.
        Retrace line = $1.30 - 0.35*$0.30 = $1.195. Current $1.10 < $1.195 → FAIL.
        """
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.3),
            premium=1.10,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL
        assert "Profit retrace" in out.reason

    @pytest.mark.asyncio
    async def test_pass_retrace_not_exceeded(self):
        """Entry $1.00, peak $1.30 (+30%), current $1.25 → only 16.7% retraced → PASS."""
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.3),
            premium=1.25,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_exact_example_from_spec(self):
        """Entry $1.00, peak $1.35 (+35%), retrace 35% of $0.35 = $0.1225.
        Retrace line = $1.35 - $0.1225 = $1.2275. Current $1.20 < $1.2275 → FAIL.
        """
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.35),
            premium=1.20,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_no_profit_to_protect(self):
        """MFE <= entry → no profit → skip."""
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=self._retrace_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.0),
            premium=0.80,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_in_pipeline_order(self):
        """Profit retrace should be in DEFAULT_EXIT_GATES and mapped."""
        gate_names = [g().name for g in DEFAULT_EXIT_GATES]
        assert "profit_retrace" in gate_names
        # Should be after profit_lock (v2.2 reorder: profit protection tier)
        pl_idx = gate_names.index("profit_lock")
        pr_idx = gate_names.index("profit_retrace")
        assert pl_idx < pr_idx
        # Should be in EXIT_GATE_TO_REASON
        assert "profit_retrace" in EXIT_GATE_TO_REASON


# ===================================================================
# ===================================================================
# Deceleration exit gate
# ===================================================================


class TestDecelExitGate:
    """decel_exit fires when short-term momentum collapses vs long-term."""

    def _decel_settings(self, **kw):
        defaults = dict(ENABLE_DECEL_EXIT=True, DECEL_SHORT_WINDOW=5,
                        DECEL_LONG_WINDOW=15, DECEL_THRESHOLD=-3.0,
                        DECEL_MIN_GAIN_PCT=5.0, DECEL_MIN_HOLD_SECONDS=480)
        defaults.update(kw)
        return _S(**defaults)

    def _make_history(self, premiums, interval=5.0):
        """Build premium history from a list of premium values."""
        import time
        base = time.time() - len(premiums) * interval
        return [(base + i * interval, p) for i, p in enumerate(premiums)]

    @pytest.mark.asyncio
    async def test_disabled(self):
        gate = DecelExitGate()
        ctx = _ctx(settings=_S(ENABLE_DECEL_EXIT=False), premium=1.0)
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_not_enough_history(self):
        gate = DecelExitGate()
        ctx = _ctx(
            settings=self._decel_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.2),
            premium=1.15,
            premium_history=self._make_history([1.0, 1.05, 1.1]),  # only 3 readings
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS
        assert "history" in out.reason

    @pytest.mark.asyncio
    async def test_below_min_gain(self):
        """Peak gain < 5% → decel not active."""
        gate = DecelExitGate()
        ctx = _ctx(
            settings=self._decel_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.03),  # +3%
            premium=1.01,
            premium_history=self._make_history([1.0] * 20),
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS
        assert "Peak gain" in out.reason

    @pytest.mark.asyncio
    async def test_fires_on_momentum_collapse(self):
        """Short-term velocity drops well below long-term → FAIL."""
        gate = DecelExitGate()
        # Premium was rising over long window, then dropping over short window
        # Long window: 1.0 → 1.2 (+20% over 15 readings)
        # Short window: 1.2 → 1.1 (-8.3% over 5 readings)
        # accel = -8.3 - 20.0 = -28.3 < -3.0
        long_rise = [1.0 + 0.02 * i for i in range(10)]  # 1.0 → 1.18
        short_drop = [1.18, 1.15, 1.12, 1.10, 1.08]      # dropping
        history = self._make_history(long_rise + short_drop)

        ctx = _ctx(
            settings=self._decel_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.2),
            premium=1.08,
            premium_history=history,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL
        assert "Decel exit" in out.reason

    @pytest.mark.asyncio
    async def test_pass_when_momentum_ok(self):
        """Recent spike → short vel much higher than long vel → PASS."""
        gate = DecelExitGate()
        # Flat for 15 bars, then sharp spike in last 5: short vel >> long vel
        flat = [1.05] * 15
        spike = [1.05, 1.08, 1.12, 1.18, 1.25]
        history = self._make_history(flat + spike)

        ctx = _ctx(
            settings=self._decel_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.25),
            premium=1.25,
            premium_history=history,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_held_too_short(self):
        """Held < min_hold_seconds → PASS regardless."""
        gate = DecelExitGate()
        # Trade opened 2 minutes ago (< 8 min min hold)
        recent_open = datetime(2026, 4, 10, 10, 28, tzinfo=ET)
        ctx = _ctx(
            settings=self._decel_settings(),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.2,
                              opened_at=recent_open.isoformat()),
            premium=1.08,
            premium_history=self._make_history([1.2 - 0.01 * i for i in range(20)]),
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.PASS
        assert "Held" in out.reason

    @pytest.mark.asyncio
    async def test_in_pipeline_and_mapped(self):
        """Decel gate is in DEFAULT_EXIT_GATES and EXIT_GATE_TO_REASON."""
        gate_names = [g().name for g in DEFAULT_EXIT_GATES]
        assert "decel_exit" in gate_names
        # v2.2 reorder: decel is in loss-cutting tier (after trails, after profit protection)
        at_idx = gate_names.index("adaptive_trailing_stop")
        de_idx = gate_names.index("decel_exit")
        assert at_idx < de_idx  # trails fire before decel
        assert "decel_exit" in EXIT_GATE_TO_REASON


# ===================================================================
# Gate conflict and coverage integration tests
# ===================================================================


class TestGateConflicts:
    """Verify no gates conflict with each other and all gain ranges are covered."""

    @pytest.mark.asyncio
    async def test_profit_retrace_fires_above_adaptive_activation(self):
        """Profit retrace fires even above adaptive trail activation — catches big give-backs."""
        gate = ProfitRetraceExitGate()
        ctx = _ctx(
            settings=_S(ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
                        PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
                        ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.5),  # +50%
            premium=1.1,  # gave back 80% of profit
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_trailing_stop_defers_to_adaptive_when_enabled(self):
        """Trailing stop SKIP when adaptive trail is enabled."""
        gate = TrailingStopExitGate()
        ctx = _ctx(
            settings=_S(ENABLE_TRAILING_STOP=True, ENABLE_ADAPTIVE_TRAIL=True),
            trade=_base_trade(mfe_premium=4.0),
            premium=2.0,
        )
        out = await gate.evaluate(ctx)
        assert out.result == GateResult.SKIP
        assert "adaptive trail is primary" in out.reason

    @pytest.mark.asyncio
    async def test_full_pipeline_no_double_exit(self):
        """Run the full pipeline — only one gate should FAIL at most."""
        # Trade is losing but within stop range and past grace
        ctx = _ctx(
            settings=_S(
                ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
                PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
                ENABLE_DECEL_EXIT=True, DECEL_SHORT_WINDOW=5, DECEL_LONG_WINDOW=15,
                DECEL_THRESHOLD=-3.0, DECEL_MIN_GAIN_PCT=5.0, DECEL_MIN_HOLD_SECONDS=0,
                ENABLE_ADAPTIVE_TRAIL=True,
                ENABLE_TRAILING_STOP=True,
                ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=True, PROFIT_LOCK_TIERS="80:25",
            ),
            trade=_base_trade(premium_per_contract=2.50, mfe_premium=3.0),
            premium=2.80,  # +12% from entry, below 40% dormant
        )
        reason, desc = await run_exit_pipeline(ctx)
        # Should either hold or exit for one reason — never multiple
        # (pipeline returns first FAIL)
        if reason is not None:
            assert isinstance(reason, str)
            assert len(reason) > 0

    @pytest.mark.asyncio
    async def test_gain_coverage_0_to_10pct_below_decel_min(self):
        """0-5% gain zone: no retrace, no decel (below all min gains), no adaptive trail."""
        ctx = _ctx(
            settings=_S(
                ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
                PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
                ENABLE_DECEL_EXIT=True, DECEL_MIN_GAIN_PCT=5.0,
                DECEL_MIN_HOLD_SECONDS=0,
                ENABLE_ADAPTIVE_TRAIL=True,
                ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0,
            ),
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.03),  # +3% peak
            premium=1.02,  # +2%
            premium_history=[(0, 1.0 + 0.001 * i) for i in range(20)],
        )
        reason, desc = await run_exit_pipeline(ctx)
        # Below 5% peak: decel won't fire (min gain), retrace won't fire (min gain 10%)
        assert reason is None  # HOLD

    @pytest.mark.asyncio
    async def test_gain_coverage_10_to_40pct(self):
        """10-40% gain zone: profit retrace is active, adaptive trail is dormant."""
        retrace_gate = ProfitRetraceExitGate()
        adaptive_gate = AdaptiveTrailingStopExitGate()

        settings = _S(
            ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
            PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
            ENABLE_ADAPTIVE_TRAIL=True,
            ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0,
        )

        # Peak +25%, now giving back 50% of profit → retrace should fire
        ctx = _ctx(
            settings=settings,
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.25),
            premium=1.125,  # gave back 50% of $0.25 profit
        )
        retrace_out = await retrace_gate.evaluate(ctx)
        adaptive_out = await adaptive_gate.evaluate(ctx)
        assert retrace_out.result == GateResult.FAIL  # retrace catches this
        assert adaptive_out.result == GateResult.PASS  # adaptive is dormant (PASS)

    @pytest.mark.asyncio
    async def test_gain_coverage_above_40pct(self):
        """Above 40% gain: both retrace and adaptive trail protect gains.
        Retrace fires first (gate #6) since it runs before adaptive (gate #10).
        """
        retrace_gate = ProfitRetraceExitGate()
        adaptive_gate = AdaptiveTrailingStopExitGate()

        settings = _S(
            ENABLE_PROFIT_RETRACE=True, PROFIT_RETRACE_PCT=35.0,
            PROFIT_RETRACE_MIN_GAIN_PCT=10.0,
            ENABLE_ADAPTIVE_TRAIL=True,
            ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0,
            ADAPTIVE_TRAIL_ACTIVE_WIDTH=35.0,
        )

        # Peak +80%, now dropped 40% from peak → both should fire
        ctx = _ctx(
            settings=settings,
            trade=_base_trade(premium_per_contract=1.0, mfe_premium=1.8),
            premium=1.08,  # 40% drop from peak = 90% of profit given back
        )
        retrace_out = await retrace_gate.evaluate(ctx)
        adaptive_out = await adaptive_gate.evaluate(ctx)
        assert retrace_out.result == GateResult.FAIL  # retrace catches big give-back
        assert adaptive_out.result == GateResult.FAIL  # adaptive also fires

    @pytest.mark.asyncio
    async def test_gate_count_updated(self):
        """Gate count should be 29 after v2.2 Phase 1 (added be_clamp + soft_trail)."""
        assert len(DEFAULT_EXIT_GATES) == 29

    @pytest.mark.asyncio
    async def test_all_gates_have_reason_mapping(self):
        """Every gate in DEFAULT_EXIT_GATES must have a mapping in EXIT_GATE_TO_REASON."""
        for gate_cls in DEFAULT_EXIT_GATES:
            gate = gate_cls()
            assert gate.name in EXIT_GATE_TO_REASON, (
                f"Gate '{gate.name}' missing from EXIT_GATE_TO_REASON"
            )


# ===================================================================
# BONUS: Dollar trail and adaptive time tighten gates
# ===================================================================


class TestDollarTrailExitGate:
    """dollar_trail fires on stair-step trailing stop."""

    @pytest.mark.asyncio
    async def test_dormant_below_activation(self):
        """Below 10% profit, trail is dormant → PASS."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=10)
        # Entry 2.00 ($200/contract), MFE 2.15 (peak profit $15 < activation $20), now 2.10
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.00,
            mfe_premium=2.15,
        )
        settings = _S(ENABLE_DOLLAR_TRAIL=True)
        ctx = _ctx(trade=trade, premium=2.10, now=now, settings=settings)
        r = await DollarTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fires_when_profit_drops_below_stop(self):
        """Profit peaked at $35, stop at $20 (activation). Drops to $15 → FAIL."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=10)
        # Entry 2.00, MFE 2.35 (peak profit $35), now 2.15 (profit $15)
        # Activation at $20 (10% of $200). Peak $35: steps = int((35-20)/20) = 0 → stop at $20
        # Current profit $15 <= $20 → exit
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.00,
            mfe_premium=2.35,
        )
        settings = _S(ENABLE_DOLLAR_TRAIL=True)
        ctx = _ctx(trade=trade, premium=2.15, now=now, settings=settings)
        r = await DollarTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_above_stop_level(self):
        """Profit peaked at $45, stop at $40. Current $42 → PASS."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=10)
        # Entry 2.00, MFE 2.45 (peak $45), now 2.42 (profit $42)
        # Steps: (45-20)/20 = 1 → stop at 20+20 = $40. $42 > $40 → hold
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.00,
            mfe_premium=2.45,
        )
        settings = _S(ENABLE_DOLLAR_TRAIL=True)
        ctx = _ctx(trade=trade, premium=2.42, now=now, settings=settings)
        r = await DollarTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_10_dollar_steps_above_50(self):
        """Peak profit $65, stop at $60 ($10 steps above $50). Drop to $55 → FAIL."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        opened = now - timedelta(minutes=10)
        # Entry 2.00, MFE 2.65 (peak $65), now 2.55 (profit $55)
        # Phase 1: (50-20)/20 = 1 step → phase1_top = 20+20 = $40
        # Phase 2: (65-40)/10 = 2 steps → stop = 40+20 = $60
        # Current $55 <= $60 → exit
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.00,
            mfe_premium=2.65,
        )
        settings = _S(ENABLE_DOLLAR_TRAIL=True)
        ctx = _ctx(trade=trade, premium=2.55, now=now, settings=settings)
        r = await DollarTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_disabled_skips(self):
        """When ENABLE_DOLLAR_TRAIL=False, gate is skipped."""
        now = datetime(2026, 4, 10, 10, 30, tzinfo=ET)
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.65)
        settings = _S(ENABLE_DOLLAR_TRAIL=False)
        ctx = _ctx(trade=trade, premium=2.00, now=now, settings=settings)
        r = await DollarTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


class TestAdaptiveTimeTighten:
    """adaptive_time_tighten fires when held long and drop exceeds tightened trail."""

    @pytest.mark.asyncio
    async def test_fires_after_tighten_threshold(self):
        now = datetime(2026, 4, 10, 11, 10, tzinfo=ET)
        opened = now - timedelta(minutes=70)  # > 60 min
        # Phase 0 trail=40%, factor 0.7 → tightened to 28%
        # MFE 5.00, current 3.50 → drop 30% > 28% → FIRE
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=5.00,
            last_target_hit=None,
        )
        ctx = _ctx(trade=trade, premium=3.50, now=now)
        r = await AdaptiveTimeTightenExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_before_time_threshold(self):
        now = datetime(2026, 4, 10, 10, 40, tzinfo=ET)
        opened = now - timedelta(minutes=30)  # < 60 min
        trade = _base_trade(
            opened_at=opened.isoformat(),
            premium_per_contract=2.50,
            mfe_premium=5.00,
        )
        ctx = _ctx(trade=trade, premium=3.50, now=now)
        r = await AdaptiveTimeTightenExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===================================================================
# ML predict_sell contract test (unit-level)
# ===================================================================


class TestMLPredictSellContract:
    """Verify compute_features and predict_sell input/output contracts."""

    def test_compute_features_returns_all_keys(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.50,
            current_premium=3.00,
            peak_premium=3.20,
            minutes_since_entry=30.0,
            now_hour=10,
            now_minute=30,
            ticker="SPY",
            is_call=True,
        )
        expected_keys = {
            "pnl_pct", "mfe_pct", "drawdown_from_peak_pct",
            "minutes_since_entry", "hour_of_day", "minute_of_hour",
            "in_sweet_spot", "past_sweet_spot", "minutes_past_sweet_spot",
            "time_pressure", "premium_velocity_5m", "premium_velocity_10m",
            "premium_velocity_15m", "pnl_acceleration", "mfe_retracement_ratio",
            "bar_range_pct", "volume", "volume_vs_avg",
            "rolling_volatility_10m", "underlying_pnl_pct",
            "underlying_velocity_5m", "is_call", "entry_premium",
            "bars_since_new_high", "consecutive_down_bars",
            "risk_reward_ratio",
        }
        assert set(features.keys()) == expected_keys

    def test_compute_features_pnl_calculation(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.00,
            current_premium=3.00,
            peak_premium=3.50,
            minutes_since_entry=60.0,
            now_hour=11,
            now_minute=0,
            ticker="QQQ",
            is_call=False,
        )
        assert features["pnl_pct"] == pytest.approx(50.0)
        assert features["mfe_pct"] == pytest.approx(75.0)
        assert features["is_call"] == 0

    def test_predict_sell_returns_ml_signal(self):
        from options_owl.risk.ml_exit import MLSellSignal, predict_sell

        # predict_sell for an unknown ticker should use the generic model (if available)
        # or return model_used="none" if no model files exist.
        # Either way, the return type must be MLSellSignal.
        result = predict_sell(
            ticker="ZZZZZ",
            entry_premium=2.00,
            current_premium=3.00,
            peak_premium=3.50,
            minutes_since_entry=30.0,
            now_hour=10,
            now_minute=30,
            is_call=True,
        )
        assert isinstance(result, MLSellSignal)
        assert isinstance(result.should_sell, bool)
        assert isinstance(result.sell_probability, float)
        assert isinstance(result.expected_future_pnl, float)
        assert result.model_used in ("generic", "none")

    def test_predict_sell_no_model_returns_hold(self):
        """When model files don't exist, predict_sell should return hold."""
        from options_owl.risk.ml_exit import MLSellSignal, _model_cache, predict_sell

        # Force the "no model" path by caching None for generic
        old_cache = dict(_model_cache)
        _model_cache["generic"] = None
        try:
            result = predict_sell(
                ticker="ZZZZZ",
                entry_premium=2.00,
                current_premium=3.00,
                peak_premium=3.50,
                minutes_since_entry=30.0,
                now_hour=10,
                now_minute=30,
                is_call=True,
            )
            assert isinstance(result, MLSellSignal)
            assert result.should_sell is False
            assert result.model_used == "none"
        finally:
            _model_cache.clear()
            _model_cache.update(old_cache)


# ---------------------------------------------------------------------------
# Adaptive 3-stage trailing stop (v2.1)
# ---------------------------------------------------------------------------


class TestAdaptiveTrailFunction:
    """Test the evaluate_adaptive_trail() function directly."""

    def test_dormant_below_activation(self):
        """Below activation threshold, trail should not trigger."""
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.30, peak_premium=1.35,
            activation_pct=40.0,
        )
        assert result.should_exit is False
        assert result.stage == "DORMANT"
        assert result.trail_width_pct == 0.0

    def test_dormant_even_with_big_drop(self):
        """Below activation, even a big drop from peak shouldn't trigger."""
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.05, peak_premium=1.39,
            activation_pct=40.0,
        )
        assert result.should_exit is False
        assert result.stage == "DORMANT"

    def test_active_stage_no_exit(self):
        """In ACTIVE stage, drop less than width should not exit."""
        # Peak gain = (1.50 - 1.00) / 1.00 = 50% → ACTIVE
        # Drop from peak = (1.50 - 1.35) / 1.50 = 10% < 35%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.35, peak_premium=1.50,
            activation_pct=40.0, active_width=35.0,
        )
        assert result.should_exit is False
        assert result.stage == "ACTIVE"
        assert result.trail_width_pct == 35.0

    def test_active_stage_triggers_exit(self):
        """In ACTIVE stage, drop >= width should trigger exit."""
        # Peak gain = 50% → ACTIVE
        # Drop from peak = (1.50 - 0.90) / 1.50 = 40% > 35%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=0.90, peak_premium=1.50,
            activation_pct=40.0, active_width=35.0,
        )
        assert result.should_exit is True
        assert result.stage == "ACTIVE"
        assert "dropped" in result.reason

    def test_runner_stage_no_exit(self):
        """In RUNNER stage, drop less than wider width should hold."""
        # Peak gain = (2.80 - 1.00) / 1.00 = 180% → RUNNER
        # Drop from peak = (2.80 - 2.00) / 2.80 = 28.6% < 45%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=2.00, peak_premium=2.80,
            activation_pct=40.0, runner_threshold=150.0, runner_width=45.0,
        )
        assert result.should_exit is False
        assert result.stage == "RUNNER"
        assert result.trail_width_pct == 45.0

    def test_runner_stage_triggers_exit(self):
        """In RUNNER stage, drop >= width should exit."""
        # Peak gain = 180% → RUNNER
        # Drop from peak = (2.80 - 1.40) / 2.80 = 50% > 45%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.40, peak_premium=2.80,
            activation_pct=40.0, runner_threshold=150.0, runner_width=45.0,
        )
        assert result.should_exit is True
        assert result.stage == "RUNNER"

    def test_moonshot_stage_no_exit(self):
        """In MOONSHOT stage, drop less than tighter width should hold."""
        # Peak gain = (5.50 - 1.00) / 1.00 = 450% → MOONSHOT
        # Drop from peak = (5.50 - 4.50) / 5.50 = 18.2% < 30%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=4.50, peak_premium=5.50,
            activation_pct=40.0, moonshot_threshold=400.0, moonshot_width=30.0,
        )
        assert result.should_exit is False
        assert result.stage == "MOONSHOT"
        assert result.trail_width_pct == 30.0

    def test_moonshot_stage_triggers_exit(self):
        """In MOONSHOT stage, drop >= width should exit."""
        # Peak gain = 450% → MOONSHOT
        # Drop from peak = (5.50 - 3.50) / 5.50 = 36.4% > 30%
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=3.50, peak_premium=5.50,
            activation_pct=40.0, moonshot_threshold=400.0, moonshot_width=30.0,
        )
        assert result.should_exit is True
        assert result.stage == "MOONSHOT"

    def test_stage_boundaries(self):
        """Test exact boundary between stages."""
        # Peak gain = exactly 150% → should be RUNNER, not ACTIVE
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=2.50, peak_premium=2.50,
            activation_pct=40.0, runner_threshold=150.0,
        )
        assert result.stage == "RUNNER"

        # Peak gain = exactly 400% → should be MOONSHOT
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=5.00, peak_premium=5.00,
            activation_pct=40.0, moonshot_threshold=400.0,
        )
        assert result.stage == "MOONSHOT"

    def test_zero_entry_premium(self):
        """Edge case: zero entry premium should not crash."""
        result = evaluate_adaptive_trail(
            entry_premium=0.0, current_premium=1.00, peak_premium=1.50,
        )
        assert result.should_exit is False
        assert result.stage == "DORMANT"

    def test_peak_gain_pct_calculation(self):
        """Verify peak_gain_pct is correctly computed."""
        result = evaluate_adaptive_trail(
            entry_premium=2.00, current_premium=4.00, peak_premium=5.00,
            activation_pct=40.0,
        )
        assert result.peak_gain_pct == pytest.approx(150.0)  # (5 - 2) / 2 * 100

    def test_drop_from_peak_pct_calculation(self):
        """Verify drop_from_peak_pct is correctly computed."""
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.50, peak_premium=2.00,
            activation_pct=40.0,
        )
        assert result.drop_from_peak_pct == pytest.approx(25.0)  # (2 - 1.5) / 2 * 100


class TestAdaptiveTrailExitGate:
    """Test the AdaptiveTrailingStopExitGate in the pipeline."""

    @pytest.mark.asyncio
    async def test_gate_disabled_when_setting_off(self):
        """Gate should SKIP when ENABLE_ADAPTIVE_TRAIL=False."""
        settings = _S(ENABLE_ADAPTIVE_TRAIL=False)
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = {"trade": trade, "exit_premium": 1.20, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_gate_dormant_holds(self):
        """Gate should PASS when in DORMANT stage (below activation)."""
        settings = _S(ENABLE_ADAPTIVE_TRAIL=True, ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0)
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.30)
        ctx = {"trade": trade, "exit_premium": 1.20, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_gate_active_exit(self):
        """Gate should FAIL when ACTIVE drop exceeds width."""
        settings = _S(ENABLE_ADAPTIVE_TRAIL=True, ADAPTIVE_TRAIL_ACTIVE_WIDTH=35.0)
        # Peak at $1.50, current at $0.90 → drop = 40% > 35%
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = {"trade": trade, "exit_premium": 0.90, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "ACTIVE" in outcome.reason

    @pytest.mark.asyncio
    async def test_gate_runner_holds_with_larger_width(self):
        """Gate should PASS in RUNNER stage with wider trail width."""
        settings = _S(
            ENABLE_ADAPTIVE_TRAIL=True,
            ADAPTIVE_TRAIL_RUNNER_THRESHOLD=150.0,
            ADAPTIVE_TRAIL_RUNNER_WIDTH=45.0,
        )
        # Peak gain = 200% → RUNNER, drop = 28.6% < 45%
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=3.00)
        ctx = {"trade": trade, "exit_premium": 2.20, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_gate_moonshot_tightens(self):
        """Gate should FAIL in MOONSHOT with tighter width."""
        settings = _S(
            ENABLE_ADAPTIVE_TRAIL=True,
            ADAPTIVE_TRAIL_MOONSHOT_THRESHOLD=400.0,
            ADAPTIVE_TRAIL_MOONSHOT_WIDTH=30.0,
        )
        # Peak gain = 500% → MOONSHOT, drop = 33.3% > 30%
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=6.00)
        ctx = {"trade": trade, "exit_premium": 4.00, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "MOONSHOT" in outcome.reason

    @pytest.mark.asyncio
    async def test_gate_missing_premium_skips(self):
        """Gate should SKIP when exit_premium is None."""
        settings = _S(ENABLE_ADAPTIVE_TRAIL=True)
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = {"trade": trade, "exit_premium": None, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_gate_no_mfe_uses_entry(self):
        """When mfe_premium is None, gate should use entry as peak."""
        settings = _S(ENABLE_ADAPTIVE_TRAIL=True, ADAPTIVE_TRAIL_ACTIVATION_PCT=40.0)
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=None)
        ctx = {"trade": trade, "exit_premium": 0.50, "settings": settings}
        gate = AdaptiveTrailingStopExitGate()
        outcome = await gate.evaluate(ctx)
        # Entry=peak=1.00, gain=0% → DORMANT
        assert outcome.result == GateResult.PASS


class TestAdaptiveTrailInPipeline:
    """Test that adaptive trail integrates correctly in the full exit pipeline."""

    @pytest.mark.asyncio
    async def test_adaptive_trail_in_default_gates(self):
        """AdaptiveTrailingStopExitGate should be in DEFAULT_EXIT_GATES."""
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "adaptive_trailing_stop" in gate_names

    @pytest.mark.asyncio
    async def test_adaptive_trail_has_reason_mapping(self):
        """adaptive_trailing_stop should be mapped in EXIT_GATE_TO_REASON."""
        assert "adaptive_trailing_stop" in EXIT_GATE_TO_REASON
        assert EXIT_GATE_TO_REASON["adaptive_trailing_stop"] == "adaptive_trail"

class TestAdaptiveTrailEdgeCases:
    """Edge case testing for the adaptive trail."""

    def test_exactly_at_activation_boundary(self):
        """Peak gain just above activation_pct should enter ACTIVE."""
        # Peak gain = 40.01% → just above threshold → ACTIVE
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.41, peak_premium=1.41,
            activation_pct=40.0,
        )
        assert result.stage == "ACTIVE"
        assert result.should_exit is False  # no drop from peak

    def test_exactly_at_activation_boundary_below(self):
        """Peak gain exactly at activation_pct is DORMANT (strict < check)."""
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.40, peak_premium=1.40,
            activation_pct=40.0,
        )
        assert result.stage == "DORMANT"

    def test_drop_exactly_at_width(self):
        """Drop exactly at width boundary should trigger exit."""
        # ACTIVE width = 35%, need drop = exactly 35%
        # Peak = 2.00, drop 35% = 2.00 * 0.65 = 1.30
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=1.30, peak_premium=2.00,
            activation_pct=40.0, active_width=35.0,
        )
        assert result.should_exit is True

    def test_very_small_premium(self):
        """Should work with tiny premiums (cheap 0DTE options)."""
        result = evaluate_adaptive_trail(
            entry_premium=0.05, current_premium=0.12, peak_premium=0.15,
            activation_pct=40.0,
        )
        # Peak gain = (0.15 - 0.05) / 0.05 = 200% → RUNNER
        assert result.stage == "RUNNER"
        assert result.should_exit is False

    def test_premium_above_peak_no_exit(self):
        """Current premium above peak should never trigger exit."""
        result = evaluate_adaptive_trail(
            entry_premium=1.00, current_premium=2.10, peak_premium=2.00,
            activation_pct=40.0,
        )
        assert result.should_exit is False
        assert result.drop_from_peak_pct < 0  # negative drop = above peak


# ---------------------------------------------------------------------------
# V3 Exit Gates — Profit Floor + Bounce-Fade
# ---------------------------------------------------------------------------


class TestProfitFloorGate:
    """Ratcheting profit floor: activates at +15%, locks 60% of peak gain."""

    @pytest.mark.asyncio
    async def test_fires_when_below_floor(self):
        """Entry $1.00, peak $1.50 (+50%), floor = 1.00 + 0.50*0.60 = $1.30.
        Current $1.25 < floor $1.30 → should exit."""
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = _ctx(trade=trade, premium=1.25,
                   settings=_S(ENABLE_PROFIT_FLOOR=True))
        r = await ProfitFloorExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Profit floor" in r.reason

    @pytest.mark.asyncio
    async def test_holds_above_floor(self):
        """Entry $1.00, peak $1.50, floor = $1.30. Current $1.35 > floor → hold."""
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = _ctx(trade=trade, premium=1.35,
                   settings=_S(ENABLE_PROFIT_FLOOR=True))
        r = await ProfitFloorExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_inactive_below_activation(self):
        """Peak gain +10% < activation +15% → floor not active."""
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.10)
        ctx = _ctx(trade=trade, premium=1.05,
                   settings=_S(ENABLE_PROFIT_FLOOR=True))
        r = await ProfitFloorExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_time_urgency_tightens_ratchet(self):
        """With < 60 min to expiry, ratchet tightens to 80%.
        Entry $1.00, peak $1.50, floor = 1.00 + 0.50*0.80 = $1.40.
        Current $1.38 < floor $1.40 → should exit."""
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50,
                            expiry_date="2026-04-27")
        # 45 min before 4 PM close
        now = datetime(2026, 4, 27, 15, 15, tzinfo=ET)
        ctx = _ctx(trade=trade, premium=1.38, now=now,
                   settings=_S(ENABLE_PROFIT_FLOOR=True))
        r = await ProfitFloorExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_disabled_when_setting_off(self):
        trade = _base_trade(premium_per_contract=1.00, mfe_premium=1.50)
        ctx = _ctx(trade=trade, premium=1.20,
                   settings=_S(ENABLE_PROFIT_FLOOR=False))
        r = await ProfitFloorExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


class TestBounceFadeGate:
    """Bounce-fade: deep dip bounce detection replaces catastrophic stop."""

    @pytest.mark.asyncio
    async def test_inactive_above_watch_threshold(self):
        """Drop only 30% < watch threshold 50% → not watching."""
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.70,
                   settings=_S(ENABLE_BOUNCE_FADE=True))
        r = await BounceFadeExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_enters_bounce_watch(self):
        """Drop 55% → enters bounce watch mode but doesn't exit yet."""
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.45,
                   settings=_S(ENABLE_BOUNCE_FADE=True))
        r = await BounceFadeExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "watching" in r.reason
        # Bounce state should be saved
        assert ctx["bounce_state"]["low"] == 0.45

    @pytest.mark.asyncio
    async def test_detects_bounce_and_fade(self):
        """Simulate: drop to $0.40, bounce to $0.50, fade to $0.42 → exit."""
        trade = _base_trade(premium_per_contract=1.00)
        settings = _S(ENABLE_BOUNCE_FADE=True)

        # First: deep drop, enter watch
        ctx1 = _ctx(trade=trade, premium=0.40, settings=settings)
        await BounceFadeExitGate().evaluate(ctx1)
        bounce_state = ctx1["bounce_state"]
        assert bounce_state["low"] == 0.40

        # Second: bounce up to $0.50 (+25% from low)
        ctx2 = _ctx(trade=trade, premium=0.50, settings=settings)
        ctx2["bounce_state"] = bounce_state
        await BounceFadeExitGate().evaluate(ctx2)
        bounce_state = ctx2["bounce_state"]
        assert bounce_state["detected"] is True
        assert bounce_state["high"] == 0.50

        # Third: fade from $0.50 to $0.42 = -16% fade → should exit
        ctx3 = _ctx(trade=trade, premium=0.42, settings=settings)
        ctx3["bounce_state"] = bounce_state
        r = await BounceFadeExitGate().evaluate(ctx3)
        assert r.result == GateResult.FAIL
        assert "Bounce-fade" in r.reason

    @pytest.mark.asyncio
    async def test_disabled_when_setting_off(self):
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.40,
                   settings=_S(ENABLE_BOUNCE_FADE=False))
        r = await BounceFadeExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


class TestThesisCutGate:
    """Thesis cut: trend-confirmed loss cutting (v3). Replaces hard stop."""

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.50,
                   settings=_S(ENABLE_THESIS_CUT=False))
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_pass_when_above_threshold(self):
        """Drop < 40% should pass (not in danger zone)."""
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.70,
                   settings=_S(ENABLE_THESIS_CUT=True))
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "30.0% < thesis check threshold" in r.reason

    @pytest.mark.asyncio
    async def test_wait_for_min_ticks(self):
        """Below threshold but not enough ticks yet — hold."""
        trade = _base_trade(premium_per_contract=1.00)
        ctx = _ctx(trade=trade, premium=0.50,
                   settings=_S(ENABLE_THESIS_CUT=True, THESIS_CUT_MIN_TICKS=4))
        # Simulate first tick in zone
        ctx["thesis_cut_state"] = {"ticks_in_zone": 0}
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "waiting" in r.reason
        assert ctx["thesis_cut_state"]["ticks_in_zone"] == 1

    @pytest.mark.asyncio
    async def test_hold_on_bounce(self):
        """Down 50% but bouncing 8% from low — hold (showing support)."""
        trade = _base_trade(premium_per_contract=1.00)
        # Premium history: declining then bouncing
        history = [(i, 1.00 - i * 0.05) for i in range(6)]  # 1.00, 0.95, ..., 0.75
        history.append((6, 0.45))  # sharp drop
        history.append((7, 0.50))  # bounce
        ctx = _ctx(trade=trade, premium=0.50,
                   settings=_S(ENABLE_THESIS_CUT=True, THESIS_CUT_BOUNCE_HOLD_PCT=5.0))
        ctx["premium_history"] = history
        ctx["thesis_cut_state"] = {"ticks_in_zone": 10}  # past min_ticks
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "bouncing" in r.reason

    @pytest.mark.asyncio
    async def test_cut_on_new_lows(self):
        """Down 50% with 3+ new lows, no bounce — cut losses."""
        trade = _base_trade(premium_per_contract=1.00)
        # Premium history: steady decline making new lows
        history = [
            (0, 0.60), (1, 0.58), (2, 0.55),
            (3, 0.52), (4, 0.50), (5, 0.48),
            (6, 0.46), (7, 0.44),
        ]
        ctx = _ctx(trade=trade, premium=0.44,
                   settings=_S(ENABLE_THESIS_CUT=True, THESIS_CUT_NEW_LOW_EXIT=3))
        ctx["premium_history"] = history
        ctx["thesis_cut_state"] = {"ticks_in_zone": 10}
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Thesis dead" in r.reason

    @pytest.mark.asyncio
    async def test_time_urgency_cut(self):
        """Down 40%+ with < 30min left — cut regardless of trend."""
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        trade = _base_trade(premium_per_contract=1.00)
        trade["expiry_date"] = "2026-04-27"
        # 15:45 ET → 15min to 16:00 close
        now = datetime(2026, 4, 27, 15, 45, tzinfo=ET)
        ctx = _ctx(trade=trade, premium=0.55, now=now,
                   settings=_S(ENABLE_THESIS_CUT=True))
        ctx["premium_history"] = [(i, 0.55) for i in range(10)]  # flat (no new lows)
        ctx["thesis_cut_state"] = {"ticks_in_zone": 10}
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "time cut" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_hold_on_deceleration(self):
        """Down 45% but decline decelerating + bounce > 2% from low — hold."""
        trade = _base_trade(premium_per_contract=1.00)
        # First half drops fast, second half stabilizes with meaningful bounce
        # Only 2 new lows (below threshold of 3), plus deceleration + bounce > 2%
        history = [
            (0, 0.58), (1, 0.55), (2, 0.53), (3, 0.52),
            (4, 0.52), (5, 0.52), (6, 0.515), (7, 0.53),
        ]
        ctx = _ctx(trade=trade, premium=0.53,
                   settings=_S(ENABLE_THESIS_CUT=True))
        ctx["premium_history"] = history
        ctx["thesis_cut_state"] = {"ticks_in_zone": 10}
        r = await ThesisCutExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_state_persists_across_calls(self):
        """Thesis cut state (ticks_in_zone) persists via ctx."""
        trade = _base_trade(premium_per_contract=1.00)
        settings = _S(ENABLE_THESIS_CUT=True, THESIS_CUT_MIN_TICKS=4)

        # First call: tick 1
        ctx1 = _ctx(trade=trade, premium=0.50, settings=settings)
        ctx1["thesis_cut_state"] = {}
        await ThesisCutExitGate().evaluate(ctx1)
        state = ctx1["thesis_cut_state"]
        assert state["ticks_in_zone"] == 1

        # Second call: tick 2 (pass state forward)
        ctx2 = _ctx(trade=trade, premium=0.48, settings=settings)
        ctx2["thesis_cut_state"] = state
        await ThesisCutExitGate().evaluate(ctx2)
        assert ctx2["thesis_cut_state"]["ticks_in_zone"] == 2


class TestExitV3GateOrder:
    """Verify new v3 gates are in the pipeline in the correct position."""

    def test_profit_floor_in_default_gates(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "profit_floor" in gate_names

    def test_bounce_fade_in_default_gates(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "bounce_fade" in gate_names

    def test_thesis_cut_in_default_gates(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "thesis_cut" in gate_names

    def test_profit_floor_before_profit_retrace(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert gate_names.index("profit_floor") < gate_names.index("profit_retrace")

    def test_adaptive_trail_before_loss_cutting(self):
        """v2.2 reorder: trails fire BEFORE loss-cutting gates (Vince's key insight)."""
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert gate_names.index("adaptive_trailing_stop") < gate_names.index("bounce_fade")
        assert gate_names.index("adaptive_trailing_stop") < gate_names.index("thesis_cut")

    def test_thesis_cut_after_bounce_fade(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert gate_names.index("bounce_fade") < gate_names.index("thesis_cut")

    def test_exit_reason_map_has_new_gates(self):
        assert "profit_floor" in EXIT_GATE_TO_REASON
        assert "bounce_fade" in EXIT_GATE_TO_REASON
        assert "thesis_cut" in EXIT_GATE_TO_REASON

    def test_be_clamp_in_default_gates(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "be_clamp" in gate_names

    def test_soft_trail_in_default_gates(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert "soft_trail" in gate_names

    def test_be_clamp_before_soft_trail(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert gate_names.index("be_clamp") < gate_names.index("soft_trail")

    def test_soft_trail_before_adaptive_trail(self):
        gate_names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert gate_names.index("soft_trail") < gate_names.index("adaptive_trailing_stop")

    def test_exit_reason_map_has_v22_gates(self):
        assert "be_clamp" in EXIT_GATE_TO_REASON
        assert "soft_trail" in EXIT_GATE_TO_REASON


# ===================================================================
# v2.2 GATE TESTS — BEClampExitGate and SoftTrailExitGate
# ===================================================================


class TestBEClampExitGate:
    """BE clamp (v2.2 §4): once peak +15%, floor = entry. Never go red after green."""

    @pytest.mark.asyncio
    async def test_fires_when_peaked_and_dropped_to_entry(self):
        """Entry $2.50, peak $3.00 (+20%), current $2.50 = entry → exit."""
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.00)
        ctx = _ctx(trade=trade, premium=2.50,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "BE clamp" in r.reason
        assert "locking breakeven" in r.reason

    @pytest.mark.asyncio
    async def test_fires_when_below_entry(self):
        """Entry $2.50, peak $3.00 (+20%), current $2.30 < entry → exit."""
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.00)
        ctx = _ctx(trade=trade, premium=2.30,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_when_still_profitable(self):
        """Entry $2.50, peak $3.00 (+20%), current $2.80 → still green, hold."""
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.00)
        ctx = _ctx(trade=trade, premium=2.80,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_holds_when_peak_below_activation(self):
        """Entry $2.50, peak $2.75 (+10% < 15% activation) → not active."""
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=2.75)
        ctx = _ctx(trade=trade, premium=2.40,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "activation" in r.reason

    @pytest.mark.asyncio
    async def test_grace_period_blocks(self):
        """Within grace period — should hold even if below entry."""
        now = datetime.now(tz=ET)
        opened = (now - timedelta(minutes=2)).isoformat()
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.00,
                            opened_at=opened)
        ctx = _ctx(trade=trade, premium=2.40, now=now,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=5))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=3.00)
        ctx = _ctx(trade=trade, premium=2.40,
                   settings=_S(ENABLE_BE_CLAMP=False))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_skip_missing_mfe(self):
        """No MFE data → skip."""
        trade = _base_trade(premium_per_contract=2.50, mfe_premium=None)
        ctx = _ctx(trade=trade, premium=2.40,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_just_above_activation_threshold(self):
        """Entry $2.00, peak $2.32 (+16%) → above activation, should fire."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.32)
        ctx = _ctx(trade=trade, premium=1.90,
                   settings=_S(ENABLE_BE_CLAMP=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await BEClampExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestSoftTrailExitGate:
    """Soft trail (v2.2 §11): 15-35% band, floor = entry + 50% of peak gain."""

    @pytest.mark.asyncio
    async def test_fires_when_below_floor(self):
        """Entry $2.00, peak $2.50 (+25%), floor = 2.00 + 0.50*0.50 = $2.25.
        Current $2.20 < $2.25 → exit."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.50)
        ctx = _ctx(trade=trade, premium=2.20,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Soft trail" in r.reason

    @pytest.mark.asyncio
    async def test_holds_above_floor(self):
        """Entry $2.00, peak $2.50 (+25%), floor = $2.25. Current $2.30 > floor → hold."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.50)
        ctx = _ctx(trade=trade, premium=2.30,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_inactive_below_min_band(self):
        """Peak gain +10% < min 15% → not active."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.20)
        ctx = _ctx(trade=trade, premium=2.05,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "min" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_hands_off_above_max_band(self):
        """Peak gain +40% >= max 35% → SKIP (adaptive trail takes over)."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.80)
        ctx = _ctx(trade=trade, premium=2.50,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP
        assert "adaptive trail" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_grace_period_blocks(self):
        """Within grace period — should hold."""
        now = datetime.now(tz=ET)
        opened = (now - timedelta(minutes=2)).isoformat()
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.50,
                            opened_at=opened)
        ctx = _ctx(trade=trade, premium=2.10, now=now,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=5))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.50)
        ctx = _ctx(trade=trade, premium=2.10,
                   settings=_S(ENABLE_SOFT_TRAIL=False))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_skip_no_profit(self):
        """MFE <= entry → no profit to protect."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.00)
        ctx = _ctx(trade=trade, premium=1.90,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_exact_at_floor(self):
        """Entry $2.00, peak $2.40 (+20%), floor = 2.00 + 0.40*0.50 = $2.20.
        Current exactly $2.20 = floor → exit (<=)."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.40)
        ctx = _ctx(trade=trade, premium=2.20,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_exact_at_max_boundary(self):
        """Peak gain exactly +35% = max → SKIP (hands off to adaptive)."""
        trade = _base_trade(premium_per_contract=2.00, mfe_premium=2.70)
        ctx = _ctx(trade=trade, premium=2.40,
                   settings=_S(ENABLE_SOFT_TRAIL=True, STOP_GRACE_PERIOD_MINUTES=0))
        r = await SoftTrailExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


# ===================================================================
# SMART GRACE TESTS — underlying-based grace termination
# ===================================================================


class TestSmartGrace:
    """Smart grace: end grace early when underlying confirms trade direction."""

    def test_grace_active_within_time_cap(self):
        """Within 20min and underlying against thesis → grace active."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="call", entry_price=560.0)
        ctx = _ctx(trade=trade, price=558.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is True
        assert "grace" in reason.lower()

    def test_grace_ends_at_time_cap(self):
        """After 20min → grace ends regardless."""
        now = datetime(2026, 4, 10, 10, 25, tzinfo=ET)
        opened = (now - timedelta(minutes=25)).isoformat()
        trade = _base_trade(opened_at=opened)
        ctx = _ctx(trade=trade, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is False
        assert "cap" in reason.lower()

    def test_smart_grace_ends_call_confirmed(self):
        """Call trade: underlying >0.1% above entry → grace ends early."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="call", entry_price=560.0)
        # 561.0 = +0.18% above 560.0 → exceeds 0.1% threshold
        ctx = _ctx(trade=trade, price=561.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is False
        assert "confirmed" in reason.lower()

    def test_smart_grace_holds_call_tiny_move(self):
        """Call trade: underlying barely above entry (<0.1%) → grace stays."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="call", entry_price=560.0)
        # 560.3 = +0.05% — below 0.1% threshold
        ctx = _ctx(trade=trade, price=560.3, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is True

    def test_smart_grace_holds_call_against(self):
        """Call trade: underlying below entry → grace stays active."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="call", entry_price=560.0)
        ctx = _ctx(trade=trade, price=558.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is True

    def test_smart_grace_ends_put_confirmed(self):
        """Put trade: underlying >0.1% below entry → grace ends early."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="put", entry_price=560.0)
        # 558.0 = -0.36% below 560.0 → exceeds 0.1% threshold
        ctx = _ctx(trade=trade, price=558.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is False
        assert "confirmed" in reason.lower()

    def test_smart_grace_holds_put_against(self):
        """Put trade: underlying above entry → grace stays active."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="put", entry_price=560.0)
        ctx = _ctx(trade=trade, price=562.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=True))
        active, reason = is_grace_active(ctx)
        assert active is True

    def test_smart_grace_disabled_falls_back_to_timer(self):
        """ENABLE_SMART_GRACE=False → pure time-based grace."""
        now = datetime(2026, 4, 10, 10, 5, tzinfo=ET)
        opened = (now - timedelta(minutes=5)).isoformat()
        trade = _base_trade(opened_at=opened, option_type="call", entry_price=560.0)
        # Underlying confirmed (above entry) but smart grace disabled
        ctx = _ctx(trade=trade, price=565.0, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=20, ENABLE_SMART_GRACE=False))
        active, reason = is_grace_active(ctx)
        assert active is True  # still in grace — smart grace off, 5min < 20min cap

    def test_grace_zero_minutes_disabled(self):
        """STOP_GRACE_PERIOD_MINUTES=0 → no grace at all."""
        now = datetime(2026, 4, 10, 10, 1, tzinfo=ET)
        opened = (now - timedelta(minutes=1)).isoformat()
        trade = _base_trade(opened_at=opened)
        ctx = _ctx(trade=trade, now=now,
                   settings=_S(STOP_GRACE_PERIOD_MINUTES=0))
        active, reason = is_grace_active(ctx)
        assert active is False
