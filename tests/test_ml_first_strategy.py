"""Comprehensive tests for the ML-first exit strategy.

Covers:
- MLSellExitGate with mocked LightGBM models
- TrailingStopExitGate with loosened 50% drop threshold
- Disabled gates (phase trail, scale-out, theta decay, no momentum)
- VIX filter enabled (pause at VIX > 35, reduce at > 25)
- Premium selection (_select_trade_premium)
- Full entry→exit pipeline integration with ML-first settings
- State management: portfolio balance, concurrent positions, trade lifecycle
- Webull executor safety rails for small-account live trading
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from options_owl.risk.pipeline import (
    DEFAULT_EXIT_GATES,
    AdaptiveTimeTightenExitGate,
    GateResult,
    MLSellExitGate,
    NoMomentumExitGate,
    ProfitLockExitGate,
    StopLossExitGate,
    Target1ExitGate,
    Target2ExitGate,
    ThetaBleedExitGate,
    ThetaDecayExitGate,
    TimeDecayZoneExitGate,
    TrailingStopExitGate,
    DollarTrailExitGate,
    VIXRegimeGate,
    run_exit_pipeline,
)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    ET = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# ML-first settings: mirrors what .env has after the ML-first changes
# ---------------------------------------------------------------------------

class _MLFirstSettings:
    """Settings object reflecting ML-first .env configuration."""

    def __init__(self, **overrides):
        defaults = {
            # Core
            "MIN_SCORE": 75,
            "PORTFOLIO_SIZE": 10000.0,
            "MAX_POSITION_PCT": 20.0,
            "MAX_CONCURRENT": 3,
            "DAILY_LOSS_LIMIT_PCT": 10.0,
            "PAPER_TRADE": True,
            # ML-first: enabled
            "ENABLE_ML_EXIT": True,
            "ENABLE_VINNY_STRATEGY": True,
            "ENABLE_TRAILING_STOP": True,
            "TRAILING_STOP_ACTIVATION_PCT": 30.0,
            "TRAILING_STOP_DROP_PCT": 50.0,  # loosened from 40%
            # ML-first: disabled
            "ENABLE_SCALE_OUT": False,
            "ENABLE_PARTIAL_PROFITS": False,
            "ENABLE_THETA_DECAY_EXIT": False,
            "ENABLE_NO_MOMENTUM_EXIT": False,
            # Stop loss
            "PREMIUM_STOP_ENABLED": True,
            "PREMIUM_STOP_PCT": 60.0,
            "STOP_GRACE_PERIOD_MINUTES": 5,
            "ENABLE_UNDERLYING_STOP": False,
            "MIN_UNDERLYING_STOP_PCT": 0.5,
            # VIX filter enabled
            "ENABLE_VIX_FILTER": True,
            "VIX_MAX": 35.0,
            "VIX_HIGH_THRESHOLD": 25.0,
            "VIX_POSITION_REDUCTION_PCT": 50.0,
            # Risk
            "ENABLE_RISK_MANAGER": True,
            "MAX_PORTFOLIO_RISK_PCT": 60.0,
            "MAX_LOSS_PER_TRADE_PCT": 20.0,
            "WEEKLY_LOSS_LIMIT_PCT": 20.0,
            "ENABLE_CIRCUIT_BREAKERS": True,
            "CB_MAX_CONSECUTIVE_LOSSES": 3,
            "CB_OPENING_BUFFER_MINUTES": 10,
            "CB_CLOSING_BUFFER_MINUTES": 15,
            # Vinny extras
            "TIME_DECAY_HOLD_MINUTES": 45.0,
            "TIME_DECAY_AFTERNOON_HOUR": 15,
            "TIME_DECAY_AFTERNOON_MINUTE": 0,
            "TIME_DECAY_STALE_MINUTES": 5.0,
            "THETA_BLEED_HOLD_MINUTES": 45.0,
            "THETA_BLEED_MAX_LOSS_PCT": 30.0,
            "CONSECUTIVE_LOSER_MAX": 2,
            "CONSECUTIVE_LOSER_PAUSE_MINUTES": 15.0,
            # Filters
            "ENABLE_IV_FILTER": False,
            "ENABLE_ANALYST_FILTER": False,
            "ENABLE_LIQUIDITY_FILTER": False,
            "ENABLE_KELLY_SIZING": False,
            "ENABLE_SCORE_SIZING": True,
            # Backtested safeguards
            "ENABLE_VELOCITY_EXIT": False,  # legacy, replaced by dollar trail
            "VELOCITY_DROP_PCT": 12.0,
            "VELOCITY_WINDOW_MINUTES": 4,
            "ENABLE_DOLLAR_TRAIL": True,
            "DOLLAR_TRAIL_ACTIVATION_PCT": 10.0,
            "DOLLAR_TRAIL_SMALL_STEP_PCT": 10.0,
            "DOLLAR_TRAIL_STEP_THRESHOLD_PCT": 25.0,
            "DOLLAR_TRAIL_LARGE_STEP_PCT": 5.0,
            "ENABLE_PROFIT_LOCK": True,
            "PROFIT_LOCK_TIERS": "80:30,150:70,250:150",
            "ENABLE_TIME_TIGHTEN": True,
            "TIME_TIGHTEN_AFTER_MINUTES": 60.0,
            "TIME_TIGHTEN_FACTOR": 0.7,
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


def _ml_exit_ctx(**overrides):
    """Base exit context for ML-first strategy tests."""
    now = datetime(2026, 4, 8, 11, 30, tzinfo=ET)
    opened = (now - timedelta(minutes=60)).isoformat()
    ctx = {
        "trade": {
            "id": 1,
            "ticker": "SPY",
            "option_type": "call",
            "strike": 560.0,
            "entry_price": 560.0,
            "stop_price": 555.0,
            "target_1": 565.0,
            "target_2": 570.0,
            "target_3": None,
            "target_4": None,
            "target_5": None,
            "exit_by": None,
            "premium_per_contract": 2.50,
            "expiry_date": "2026-04-08",
            "opened_at": opened,
            "mfe_premium": 3.50,
            "last_target_hit": 0,
            "last_new_high_at": None,
        },
        "current_price": 563.0,
        "exit_premium": 3.00,
        "now_et": now,
        "settings": _MLFirstSettings(),
        "premium_history": None,
        "current_vix": None,
    }
    ctx.update(overrides)
    return ctx


# ===========================================================================
# 1. MLSellExitGate
# ===========================================================================


class TestMLSellExitGate:
    """Test the ML exit gate with mocked LightGBM models."""

    @pytest.mark.asyncio
    async def test_ml_sell_triggers_when_both_agree(self):
        """ML should sell when P(sell) > 0.4 AND E[future] < 2%."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=True,
            sell_probability=0.65,
            expected_future_pnl=-5.0,
            reason="ML combo: P(sell)=0.65, E[future]=-5.0%",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx()
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "ML combo" in r.reason

    @pytest.mark.asyncio
    async def test_ml_hold_when_upside_expected(self):
        """ML should hold when E[future] is positive."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.20,
            expected_future_pnl=15.0,
            reason="hold",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx()
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "ML hold" in r.reason

    @pytest.mark.asyncio
    async def test_ml_skip_when_disabled(self):
        """Gate should SKIP when ENABLE_ML_EXIT=False."""
        ctx = _ml_exit_ctx(settings=_MLFirstSettings(ENABLE_ML_EXIT=False))
        r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_ml_skip_when_no_premium(self):
        """Gate should SKIP when exit_premium is None."""
        ctx = _ml_exit_ctx(exit_premium=None)
        r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_ml_skip_when_no_opened_at(self):
        """Gate should SKIP when trade has no opened_at timestamp."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["opened_at"] = None
        r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_ml_regressor_bearish_triggers_sell(self):
        """ML should sell when E[future] < -10% even if P(sell) is low."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=True,
            sell_probability=0.30,
            expected_future_pnl=-15.0,
            reason="ML regressor bearish: E[future]=-15.0%",
            model_used="generic",
        )
        ctx = _ml_exit_ctx()
        ctx["trade"]["ticker"] = "AAPL"
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            r = await MLSellExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "regressor bearish" in r.reason

    @pytest.mark.asyncio
    async def test_ml_uses_model_name_in_reason(self):
        """Reason string should include the model name used."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=5.0,
            reason="hold",
            model_used="QQQ",
        )
        ctx = _ml_exit_ctx()
        ctx["trade"]["ticker"] = "QQQ"
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            r = await MLSellExitGate().evaluate(ctx)
        assert "QQQ" in r.reason

    @pytest.mark.asyncio
    async def test_ml_receives_premium_history(self):
        """Premium history from context should be passed to predict_sell."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False, sell_probability=0.1, expected_future_pnl=5.0,
            reason="hold", model_used="SPY",
        )
        history = [2.50, 2.60, 2.80, 3.00, 3.10, 2.90, 3.00]
        ctx = _ml_exit_ctx(premium_history=history)
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal) as mock_pred:
            await MLSellExitGate().evaluate(ctx)
            call_kwargs = mock_pred.call_args[1]
            assert call_kwargs["premium_history"] == history

    @pytest.mark.asyncio
    async def test_ml_put_option_is_call_false(self):
        """For put options, is_call should be False in the predict_sell call."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False, sell_probability=0.1, expected_future_pnl=5.0,
            reason="hold", model_used="SPY",
        )
        ctx = _ml_exit_ctx()
        ctx["trade"]["option_type"] = "put"
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal) as mock_pred:
            await MLSellExitGate().evaluate(ctx)
            assert mock_pred.call_args[1]["is_call"] is False


# ===========================================================================
# 2. TrailingStopExitGate (loosened to 50%)
# ===========================================================================


class TestTrailingStopLoosened:
    """Trailing stop should use the loosened 50% drop threshold."""

    @pytest.mark.asyncio
    async def test_holds_at_45pct_drop_from_peak(self):
        """45% drop from peak should HOLD with 50% threshold."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 4.00  # peak = 100% gain
        ctx["exit_premium"] = 2.20  # 45% drop from 4.00
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_triggers_at_55pct_drop_from_peak(self):
        """55% drop from peak should trigger exit with 50% threshold."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 4.00
        ctx["exit_premium"] = 1.80  # 55% drop from 4.00
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Trailing stop" in r.reason

    @pytest.mark.asyncio
    async def test_not_activated_until_30pct_gain(self):
        """Trailing stop should not activate until peak gain >= 30%."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.50  # only 25% peak gain
        ctx["exit_premium"] = 1.00  # down 60% from peak, but not activated
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "activation" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        ctx = _ml_exit_ctx(settings=_MLFirstSettings(ENABLE_TRAILING_STOP=False))
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_skip_missing_mfe(self):
        ctx = _ml_exit_ctx()
        ctx["trade"]["mfe_premium"] = None
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_exact_threshold_triggers(self):
        """Exactly 50% drop should trigger."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 4.00  # peak gain 100%
        ctx["exit_premium"] = 2.00  # exactly 50% drop
        r = await TrailingStopExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


# ===========================================================================
# 3. Disabled gates: ThetaDecay, NoMomentum
# ===========================================================================


class TestDisabledGates:
    """Gates disabled in ML-first strategy should always SKIP."""

    @pytest.mark.asyncio
    async def test_theta_decay_skips(self):
        ctx = _ml_exit_ctx()
        r = await ThetaDecayExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_no_momentum_skips(self):
        ctx = _ml_exit_ctx()
        r = await NoMomentumExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


# ===========================================================================
# 4. ThetaBleedExitGate
# ===========================================================================


class TestThetaBleedExit:
    @pytest.mark.asyncio
    async def test_holds_before_45min(self):
        now = datetime.now()
        opened = (now - timedelta(minutes=30)).isoformat()
        ctx = _ml_exit_ctx(now_et=now)
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["exit_premium"] = 1.20  # big loss, but only 30 min
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_triggers_after_45min_with_big_loss(self):
        now = datetime.now()
        opened = (now - timedelta(minutes=50)).isoformat()
        ctx = _ml_exit_ctx(now_et=now)
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["exit_premium"] = 1.30  # 35% loss, > 30% threshold
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_holds_after_45min_small_loss(self):
        now = datetime.now()
        opened = (now - timedelta(minutes=50)).isoformat()
        ctx = _ml_exit_ctx(now_et=now)
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["exit_premium"] = 1.60  # only 20% loss, < 30%
        r = await ThetaBleedExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===========================================================================
# 6. VIX Regime Gate (entry pipeline)
# ===========================================================================


class _FakeSignal:
    def __init__(self, **kwargs):
        defaults = {
            "ticker": "SPY", "score": 85, "atm_premium": 2.50,
            "stop_price": 555.0,
            "bot_source": type("BS", (), {"value": "Captain Hook"})(),
            "direction": type("D", (), {"value": "call"})(),
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestVIXRegimeGate:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        ctx = {
            "signal": _FakeSignal(),
            "settings": _MLFirstSettings(ENABLE_VIX_FILTER=False),
        }
        r = await VIXRegimeGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_blocks_when_vix_above_max(self):
        """VIX > 35 should reject trades."""
        from dataclasses import dataclass

        @dataclass
        class FakeRegime:
            can_trade: bool
            reason: str
            reduce_size: bool = False
            reduction_pct: float = 0.0

        ctx = {
            "signal": _FakeSignal(),
            "settings": _MLFirstSettings(ENABLE_VIX_FILTER=True),
        }
        with patch(
            "options_owl.risk.vix_regime.check_vix_regime",
            return_value=FakeRegime(can_trade=False, reason="VIX 38 > max 35"),
        ):
            r = await VIXRegimeGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_passes_when_vix_normal(self):
        from dataclasses import dataclass

        @dataclass
        class FakeRegime:
            can_trade: bool
            reason: str

        ctx = {
            "signal": _FakeSignal(),
            "settings": _MLFirstSettings(ENABLE_VIX_FILTER=True),
        }
        with patch(
            "options_owl.risk.vix_regime.check_vix_regime",
            return_value=FakeRegime(can_trade=True, reason="VIX 18 — normal"),
        ):
            r = await VIXRegimeGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===========================================================================
# 7. Premium selection (_select_trade_premium)
# ===========================================================================


class TestSelectTradePremium:
    def test_swaps_when_strikes_match(self):
        from options_owl.execution.paper_trader import _select_trade_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="AMD", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=85, strength=SignalStrength.STRONG, entry_price=160.0,
            target_price=165.0, expected_move_pct=2.0, strike=157.5,
            expiry="0DTE", risk_reward=2.0, target_1=162.0, target_2=165.0,
            stop_price=158.0, atm_strike=160.0, atm_premium=14.50,
            otm_strike=157.5, otm_premium=0.85,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        fixed = _select_trade_premium(sig)
        assert fixed.atm_premium == 0.85
        assert fixed.atm_strike == 157.5

    def test_no_swap_when_strikes_differ(self):
        from options_owl.execution.paper_trader import _select_trade_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="AMD", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=85, strength=SignalStrength.STRONG, entry_price=160.0,
            target_price=165.0, expected_move_pct=2.0, strike=160.0,
            expiry="0DTE", risk_reward=2.0, target_1=162.0, target_2=165.0,
            stop_price=158.0, atm_strike=160.0, atm_premium=14.50,
            otm_strike=157.5, otm_premium=0.85,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        fixed = _select_trade_premium(sig)
        # strikes differ (160 vs 157.5), no swap
        assert fixed.atm_premium == 14.50

    def test_no_swap_when_otm_premium_missing(self):
        from options_owl.execution.paper_trader import _select_trade_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=85, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=560.0, otm_premium=None,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        fixed = _select_trade_premium(sig)
        assert fixed.atm_premium == 2.50

    def test_no_swap_when_otm_more_expensive(self):
        from options_owl.execution.paper_trader import _select_trade_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=85, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=558.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=557.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=4.00,  # OTM more expensive (deep ITM)
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        fixed = _select_trade_premium(sig)
        assert fixed.atm_premium == 2.50


# ===========================================================================
# 8. Full exit pipeline integration — ML-first configuration
# ===========================================================================


class TestExitPipelineMLFirst:
    """Integration tests for the full exit pipeline with ML-first settings."""

    @pytest.mark.asyncio
    async def test_stop_loss_has_highest_priority(self):
        """Stop loss should trigger before ML even has a chance."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 0.80  # 68% drop — exceeds 60% threshold
        # Make trade old enough to pass grace period
        ctx["trade"]["opened_at"] = (
            ctx["now_et"] - timedelta(minutes=30)
        ).isoformat()

        reason, desc = await run_exit_pipeline(ctx)
        assert reason == "stop_hit"

    @pytest.mark.asyncio
    async def test_ml_triggers_before_trailing_stop(self):
        """ML gate (#2) should trigger before trailing stop (#4)."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=True, sell_probability=0.70,
            expected_future_pnl=-8.0,
            reason="ML combo: P(sell)=0.70, E[future]=-8.0%",
            model_used="SPY",
        )
        now_et = datetime(2026, 4, 8, 11, 30, tzinfo=ET)
        opened_at = (now_et - timedelta(minutes=60)).isoformat()
        ctx = _ml_exit_ctx(now_et=now_et)
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 4.00
        ctx["exit_premium"] = 1.90  # 52.5% drop from peak
        ctx["trade"]["opened_at"] = opened_at
        # Bypass grace period (uses datetime.now() internally)
        ctx["settings"] = _MLFirstSettings(STOP_GRACE_PERIOD_MINUTES=0)

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "ml_sell"

    @pytest.mark.asyncio
    async def test_disabled_gates_are_skipped_in_full_pipeline(self):
        """Phase trail, theta decay, no momentum should not trigger."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False, sell_probability=0.10,
            expected_future_pnl=10.0, reason="hold", model_used="SPY",
        )
        now = datetime.now()
        now_et = datetime(2026, 4, 8, 11, 30, tzinfo=ET)
        ctx = _ml_exit_ctx(now_et=now_et)
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 2.70  # up 8% — exceeds 5% setup threshold
        ctx["trade"]["mfe_premium"] = 2.80
        ctx["trade"]["last_target_hit"] = 0
        ctx["trade"]["opened_at"] = (now - timedelta(minutes=5)).isoformat()
        ctx["settings"] = _MLFirstSettings(
            STOP_GRACE_PERIOD_MINUTES=0, ENABLE_DOLLAR_TRAIL=False,
        )

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        # No exit reason — all gates hold
        assert reason is None

    @pytest.mark.asyncio
    async def test_eod_cutoff_still_works(self):
        """EOD gate should still trigger at 3:45 PM even when ML says hold."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False, sell_probability=0.10,
            expected_future_pnl=20.0, reason="hold", model_used="SPY",
        )
        now_et = datetime(2026, 4, 8, 15, 50, tzinfo=ET)
        ctx = _ml_exit_ctx(now_et=now_et)
        ctx["trade"]["opened_at"] = (now_et - timedelta(minutes=300)).isoformat()
        ctx["settings"] = _MLFirstSettings(
            STOP_GRACE_PERIOD_MINUTES=0,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,
            ENABLE_TIME_TIGHTEN=False,
        )

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "eod_expiry"

    @pytest.mark.asyncio
    async def test_trailing_stop_triggers_when_ml_holds(self):
        """If ML says hold but trailing stop drops 55% from peak, trailing stop wins.

        Note: velocity_exit and profit_lock are disabled here to test
        trailing_stop in isolation (they would fire first otherwise).
        """
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False, sell_probability=0.10,
            expected_future_pnl=5.0, reason="hold", model_used="SPY",
        )
        now_et = datetime(2026, 4, 8, 11, 30, tzinfo=ET)
        ctx = _ml_exit_ctx(now_et=now_et)
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 4.00  # 100% peak gain — activated
        ctx["exit_premium"] = 1.70  # 57.5% drop from peak
        ctx["trade"]["opened_at"] = (now_et - timedelta(minutes=60)).isoformat()
        ctx["settings"] = _MLFirstSettings(
            STOP_GRACE_PERIOD_MINUTES=0,
            ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
            ENABLE_PROFIT_LOCK=False,
            ENABLE_TIME_TIGHTEN=False,
        )

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "trailing_stop"


# ===========================================================================
# 9. StopLossExitGate — premium-based with grace period
# ===========================================================================


class TestStopLossMLFirst:
    @pytest.mark.asyncio
    async def test_grace_period_blocks_normal_stop(self):
        """During grace period, normal stop is blocked (but catastrophic still fires)."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 1.70  # -32% drop — below catastrophic 45%
        ctx["trade"]["opened_at"] = (ctx["now_et"] - timedelta(minutes=2)).isoformat()
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_premium_stop_at_60pct(self):
        """Premium stop should fire when premium drops 60%+ from entry (hits catastrophic first)."""
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(STOP_GRACE_PERIOD_MINUTES=0)
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 0.90  # 64% drop — hits catastrophic (45%) before normal stop
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_premium_stop_holds_at_40pct_drop(self):
        """Premium drop of 40% should NOT trigger 60% threshold (and is below catastrophic 45%)."""
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(STOP_GRACE_PERIOD_MINUTES=0)
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 1.50  # exactly 40% drop — below both 60% stop and 45% catastrophic
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===========================================================================
# 10. Score-based position sizing (Vinny)
# ===========================================================================


class TestScoreSizing:
    def test_flat_sizing_all_scores(self):
        """All scores >= 78 get same fallback: int(5*0.85) = 4."""
        from options_owl.risk.vinny_strategy import score_to_contracts
        assert score_to_contracts(150) == 4
        assert score_to_contracts(130) == 4
        assert score_to_contracts(110) == 4
        assert score_to_contracts(95) == 4
        assert score_to_contracts(78) == 4

    def test_below_threshold_gets_0(self):
        from options_owl.risk.vinny_strategy import score_to_contracts
        assert score_to_contracts(77) == 0  # below 78 = rejected

    def test_capped_by_affordability(self):
        """With $400 balance and $170 cost/contract — position cap limits it."""
        from options_owl.risk.vinny_strategy import score_to_contracts
        # 100% cap: $400 × 100% = $400 > $170 → slot=$75, 85%=$63, raw=0 → floor=1
        result = score_to_contracts(
            score=150,
            cost_per_contract=170.0,
            balance=400.0,
            max_position_pct=100.0,
        )
        assert result == 1  # slot is tiny but position cap allows it
        # $200 balance, 100% cap: slot=$37, 85%=$31, raw=0, floor=1, pos_cap=1 → 1
        result2 = score_to_contracts(
            score=150,
            cost_per_contract=170.0,
            balance=200.0,
            max_position_pct=100.0,
        )
        assert result2 == 1

    def test_capped_by_max_position_pct(self):
        """$10K balance, 20% pos cap → $2000/$170 = 11. Slot=$1875, 85%=$1593 raw=9 → 9."""
        from options_owl.risk.vinny_strategy import score_to_contracts
        result = score_to_contracts(
            score=150,
            cost_per_contract=170.0,
            balance=10000.0,
            max_position_pct=20.0,
        )
        assert result == 9  # slot is binding (raw < pos_cap)

    def test_small_account_skips_expensive(self):
        """$300/contract on $200 account with 15% cap → pos_cap=$30 < $300 → SKIP."""
        from options_owl.risk.vinny_strategy import score_to_contracts
        result = score_to_contracts(
            score=95,
            cost_per_contract=300.0,
            balance=200.0,
            max_position_pct=15.0,
        )
        assert result == 0


# ===========================================================================
# 11. Webull executor safety rails (for $400 live trading)
# ===========================================================================


class TestWebullSafetyRails:
    def test_paper_trade_blocks_real_orders(self):
        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        s = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="k",
            WEBULL_APP_SECRET="s",
            PAPER_TRADE=True,
        )
        executor = WebullExecutor(s)
        with pytest.raises(RuntimeError, match="PAPER_TRADE"):
            executor._check_safety_limits(1, 2.50, "BUY")

    def test_max_contracts_cap(self):
        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import MAX_ORDER_CONTRACTS, WebullExecutor

        s = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="k",
            WEBULL_APP_SECRET="s",
            PAPER_TRADE=False,
        )
        executor = WebullExecutor(s)
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(MAX_ORDER_CONTRACTS + 1, 1.00, "BUY")

    def test_max_value_cap(self):
        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        s = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="k",
            WEBULL_APP_SECRET="s",
            PAPER_TRADE=False,
        )
        executor = WebullExecutor(s)
        # 5 contracts * $11 * 100 = $5500 > $5000
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(5, 11.0, "BUY")

    def test_kill_switch_blocks_all(self):
        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        s = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="k",
            WEBULL_APP_SECRET="s",
            WEBULL_KILL_SWITCH=True,
        )
        executor = WebullExecutor(s)
        with pytest.raises(RuntimeError, match="KILL_SWITCH"):
            executor._check_kill_switch()

    def test_400_dollar_trade_within_limits(self):
        """$400 portfolio: 2 contracts at $1.50 = $300 total, well within caps."""
        from options_owl.config.settings import Settings
        from options_owl.execution.webull_executor import WebullExecutor

        s = Settings(
            DISCORD_TOKEN="fake",
            WEBULL_APP_KEY="k",
            WEBULL_APP_SECRET="s",
            PAPER_TRADE=False,
        )
        executor = WebullExecutor(s)
        # Should not raise
        executor._check_safety_limits(2, 1.50, "BUY")


# ===========================================================================
# 12. ML feature computation
# ===========================================================================


class TestMLFeatureComputation:
    def test_basic_feature_computation(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.00,
            current_premium=2.60,
            peak_premium=3.00,
            minutes_since_entry=60.0,
            now_hour=11,
            now_minute=30,
            ticker="SPY",
            is_call=True,
        )
        assert features["pnl_pct"] == pytest.approx(30.0)
        assert features["mfe_pct"] == pytest.approx(50.0)
        assert features["is_call"] == 1
        assert features["minutes_since_entry"] == 60.0
        assert features["hour_of_day"] == 11

    def test_in_sweet_spot(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.00, current_premium=2.50, peak_premium=2.50,
            minutes_since_entry=90.0, now_hour=12, now_minute=0,
            ticker="SPY", is_call=True,
        )
        assert features["in_sweet_spot"] == 1
        assert features["past_sweet_spot"] == 0

    def test_past_sweet_spot(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.00, current_premium=2.50, peak_premium=2.50,
            minutes_since_entry=180.0, now_hour=14, now_minute=0,
            ticker="SPY", is_call=True,
        )
        assert features["in_sweet_spot"] == 0
        assert features["past_sweet_spot"] == 1
        assert features["minutes_past_sweet_spot"] == 30.0

    def test_velocity_with_history(self):
        from options_owl.risk.ml_exit import compute_features

        history = [2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50]
        features = compute_features(
            entry_premium=2.00, current_premium=2.50, peak_premium=2.50,
            minutes_since_entry=60.0, now_hour=11, now_minute=0,
            ticker="SPY", is_call=True, premium_history=history,
        )
        assert features["premium_velocity_5m"] != 0.0
        assert features["rolling_volatility_10m"] >= 0.0

    def test_consecutive_down_bars(self):
        from options_owl.risk.ml_exit import compute_features

        history = [2.50, 2.60, 2.70, 2.65, 2.60, 2.55]  # 3 consecutive down
        features = compute_features(
            entry_premium=2.00, current_premium=2.55, peak_premium=2.70,
            minutes_since_entry=60.0, now_hour=11, now_minute=0,
            ticker="SPY", is_call=True, premium_history=history,
        )
        assert features["consecutive_down_bars"] == 3

    def test_zero_entry_premium_clamped(self):
        """Entry premium of 0 should be clamped to 0.01 to avoid division by zero."""
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=0.0, current_premium=1.00, peak_premium=1.00,
            minutes_since_entry=10.0, now_hour=10, now_minute=0,
            ticker="SPY", is_call=True,
        )
        assert features["pnl_pct"] > 0  # should not crash

    def test_put_option(self):
        from options_owl.risk.ml_exit import compute_features

        features = compute_features(
            entry_premium=2.00, current_premium=2.50, peak_premium=2.50,
            minutes_since_entry=60.0, now_hour=11, now_minute=0,
            ticker="SPY", is_call=False,
        )
        assert features["is_call"] == 0


# ===========================================================================
# 13. Target gates — last_target_hit prevents re-triggering
# ===========================================================================


class TestTargetGateStateManagement:
    @pytest.mark.asyncio
    async def test_t1_skips_when_already_hit(self):
        ctx = _ml_exit_ctx(current_price=566.0)
        ctx["trade"]["last_target_hit"] = 1
        r = await Target1ExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP
        assert "already hit" in r.reason

    @pytest.mark.asyncio
    async def test_t2_skips_when_already_hit(self):
        ctx = _ml_exit_ctx(current_price=571.0)
        ctx["trade"]["last_target_hit"] = 2
        r = await Target2ExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_t1_triggers_on_first_hit(self):
        ctx = _ml_exit_ctx(current_price=566.0)
        ctx["trade"]["last_target_hit"] = 0
        r = await Target1ExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_t2_triggers_when_t1_hit_but_not_t2(self):
        ctx = _ml_exit_ctx(current_price=571.0)
        ctx["trade"]["last_target_hit"] = 1
        r = await Target2ExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_put_target_direction(self):
        """For puts, target is hit when price goes DOWN."""
        ctx = _ml_exit_ctx(current_price=553.0)
        ctx["trade"]["option_type"] = "put"
        ctx["trade"]["target_1"] = 555.0
        ctx["trade"]["last_target_hit"] = 0
        r = await Target1ExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


# ===========================================================================
# 14. Exit gate ordering — verify DEFAULT_EXIT_GATES order
# ===========================================================================


class TestExitGateOrdering:
    def test_enrg_is_first(self):
        from options_owl.risk.pipeline import ENRGExitGate
        assert DEFAULT_EXIT_GATES[0] is ENRGExitGate

    def test_stop_loss_is_second(self):
        assert DEFAULT_EXIT_GATES[1] is StopLossExitGate

    def test_ml_after_defensive(self):
        """ML is after defensive gates (stop, BE clamp) in v2.2 reorder."""
        names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert names.index("ml_sell") > names.index("stop_loss")
        assert names.index("ml_sell") > names.index("be_clamp")

    def test_eod_is_near_end(self):
        names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert names.index("eod_cutoff") > names.index("ml_sell")

    def test_total_gate_count(self):
        assert len(DEFAULT_EXIT_GATES) == 29  # v2.2 Phase 1: +be_clamp, +soft_trail


# ===========================================================================
# 15. TimeDecayZoneExitGate
# ===========================================================================


class TestTimeDecayZoneExit:
    @pytest.mark.asyncio
    async def test_skip_when_vinny_disabled(self):
        ctx = _ml_exit_ctx(settings=_MLFirstSettings(ENABLE_VINNY_STRATEGY=False))
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_pass_when_not_in_decay_zone(self):
        """Before 45 min and before 3 PM — not in decay zone."""
        now = datetime(2026, 4, 8, 10, 30, tzinfo=ET)
        opened = (now - timedelta(minutes=20)).isoformat()
        ctx = _ml_exit_ctx(now_et=now)
        ctx["trade"]["opened_at"] = opened
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_pass_when_still_making_highs(self):
        """In decay zone but premium is at/near peak — hold."""
        now = datetime(2026, 4, 8, 15, 15, tzinfo=ET)  # 3:15 PM
        opened = (now - timedelta(minutes=60)).isoformat()
        ctx = _ml_exit_ctx(now_et=now)
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["mfe_premium"] = 3.50
        ctx["exit_premium"] = 3.48  # within 1% of peak
        r = await TimeDecayZoneExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


# ===========================================================================
# 16. Paper trader integration — full trade lifecycle
# ===========================================================================


# ===========================================================================
# 17. Smart entry — live price verification
# ===========================================================================


class TestSmartEntry:
    """Test the live premium verification and on-the-fly price adjustment."""

    @pytest.mark.asyncio
    async def test_uses_live_premium_when_close(self):
        """When live quote is within tolerance, should use live price."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        # Mock the chain lookup to return a close price
        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}
        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=True)

        with patch("options_owl.execution.position_monitor._fetch_option_chain_for_ticker", return_value=mock_chain), \
             patch("options_owl.execution.position_monitor._lookup_premium_from_chain", return_value=2.65):
            updated, reason, nbbo = await _verify_live_premium(sig, settings)

        assert updated.atm_premium == 2.65  # uses live quote
        assert "live_verified" in reason
        assert "+6.0%" in reason

    @pytest.mark.asyncio
    async def test_rejects_when_deviation_too_large(self):
        """When live quote is >30% off from signal, should reject."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}
        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=True)

        # Live price is 50% higher — way off
        with patch("options_owl.execution.position_monitor._fetch_option_chain_for_ticker", return_value=mock_chain), \
             patch("options_owl.execution.position_monitor._lookup_premium_from_chain", return_value=3.80):
            updated, reason, nbbo = await _verify_live_premium(sig, settings)

        assert updated.atm_premium == 0.0  # rejected
        assert "deviation_too_large" in reason

    @pytest.mark.asyncio
    async def test_accepts_cheaper_live_price(self):
        """When live quote is cheaper than signal, use it (better deal)."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}
        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=True)

        with patch("options_owl.execution.position_monitor._fetch_option_chain_for_ticker", return_value=mock_chain), \
             patch("options_owl.execution.position_monitor._lookup_premium_from_chain", return_value=2.10):
            updated, reason, nbbo = await _verify_live_premium(sig, settings)

        assert updated.atm_premium == 2.10  # cheaper
        assert "live_verified" in reason

    @pytest.mark.asyncio
    async def test_falls_back_to_signal_when_no_chain(self):
        """When chain lookup fails, should fall back to signal premium."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=True)

        with patch("options_owl.execution.position_monitor._fetch_option_chain_for_ticker", return_value=None):
            updated, reason, nbbo = await _verify_live_premium(sig, settings)

        assert updated.atm_premium == 2.50  # kept signal premium
        assert "no_chain_available" in reason

    @pytest.mark.asyncio
    async def test_rejects_below_minimum_premium(self):
        """Live premium below $0.10 should be rejected."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=565.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=0.50,
            otm_strike=565.0, otm_premium=0.30,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        mock_chain = {"calls": MagicMock(), "puts": MagicMock()}
        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=True)

        with patch("options_owl.execution.position_monitor._fetch_option_chain_for_ticker", return_value=mock_chain), \
             patch("options_owl.execution.position_monitor._lookup_premium_from_chain", return_value=0.05):
            updated, reason, nbbo = await _verify_live_premium(sig, settings)

        assert updated.atm_premium == 0.0
        assert "live_premium_too_low" in reason

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """With ENABLE_SMART_ENTRY=False, should pass through unchanged."""
        from options_owl.execution.paper_trader import _verify_live_premium
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=2.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )

        settings = _MLFirstSettings(ENABLE_SMART_ENTRY=False)
        updated, reason, nbbo = await _verify_live_premium(sig, settings)
        assert updated.atm_premium == 2.50
        assert reason == "smart_entry_disabled"
        assert nbbo is None  # no quote fetched when disabled

    @pytest.mark.asyncio
    async def test_finds_nearby_strike(self):
        """When exact strike not found, should find nearby within $1."""
        import pandas as pd
        from options_owl.execution.paper_trader import _find_nearby_strike_premium

        df = pd.DataFrame({
            "strike": [559.0, 559.5, 560.5, 561.0],
            "bid": [2.30, 2.50, 2.10, 1.90],
            "ask": [2.40, 2.60, 2.20, 2.00],
            "lastPrice": [2.35, 2.55, 2.15, 1.95],
        })
        chain = {"calls": df, "puts": pd.DataFrame()}

        # Looking for $560 — should find $559.5 or $560.5 (both within $1)
        result = _find_nearby_strike_premium(chain, 560.0, "call", max_distance=1.0)
        assert result is not None
        assert result == 2.55 or result == 2.15  # midpoint of closest


class TestPaperTraderLifecycle:
    @pytest.mark.asyncio
    async def test_open_and_close_trade(self, tmp_db_path):
        from options_owl.config.settings import Settings
        from options_owl.execution.paper_trader import PaperTrader, get_open_trades, get_portfolio
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        settings = Settings(
            DISCORD_TOKEN="fake", DB_PATH=tmp_db_path,
            PORTFOLIO_SIZE=400.0, MAX_POSITION_PCT=100.0,
            MIN_SCORE=75, DAILY_LOSS_LIMIT_PCT=10.0,
            ENABLE_RISK_MANAGER=False, SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=0.0, ENABLE_DCA=False,
            ENABLE_VINNY_STRATEGY=False, ENABLE_SCORE_SIZING=False,
            ENABLE_SMART_ENTRY=False,
        )
        trader = PaperTrader(settings)
        await trader.init()

        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=85, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=560.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=1.50,
            otm_strike=558.0, otm_premium=0.80,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        assert result["contracts"] >= 1

        # Check open trades
        trades = await get_open_trades(tmp_db_path)
        assert len(trades) == 1

        # Close with profit
        close_result = await trader.close_trade(
            trade_id=result["trade_id"],
            exit_price=563.0,
            exit_premium=2.10,
            reason="ml_sell",
        )
        assert close_result["pnl"] > 0
        assert close_result["reason"] == "ml_sell"

        # Verify balance increased
        portfolio = await get_portfolio(tmp_db_path, 400.0)
        assert portfolio["current_balance"] > 400.0

    @pytest.mark.asyncio
    async def test_400_dollar_portfolio_sizing(self, tmp_db_path):
        """With $400, should be able to buy at least 1 contract of cheap options."""
        from options_owl.config.settings import Settings
        from options_owl.execution.paper_trader import PaperTrader
        from options_owl.models.signals import (
            BotSource, Direction, Sentiment, SignalStrength, TradeSignal,
        )

        settings = Settings(
            DISCORD_TOKEN="fake", DB_PATH=tmp_db_path,
            PORTFOLIO_SIZE=400.0, MAX_POSITION_PCT=50.0,
            MIN_SCORE=75, DAILY_LOSS_LIMIT_PCT=10.0,
            ENABLE_RISK_MANAGER=False, SIMULATED_ENTRY_SLIPPAGE_BPS=0.0,
            SIMULATED_EXIT_SLIPPAGE_BPS=0.0, ENABLE_DCA=False,
            ENABLE_VINNY_STRATEGY=False, ENABLE_SCORE_SIZING=False,
            ENABLE_SMART_ENTRY=False,
        )
        trader = PaperTrader(settings)
        await trader.init()

        # Cheap 0DTE option: $0.80 premium = $80/contract
        sig = TradeSignal(
            ticker="SPY", sentiment=Sentiment.BULLISH, direction=Direction.CALL,
            score=90, strength=SignalStrength.STRONG, entry_price=560.0,
            target_price=565.0, expected_move_pct=1.0, strike=562.0,
            expiry="0DTE", risk_reward=2.0, target_1=562.0, target_2=565.0,
            stop_price=558.0, atm_strike=560.0, atm_premium=0.80,
            otm_strike=562.0, otm_premium=0.50,
            bot_source=BotSource.CAPTAIN_HOOK, is_elite=True,
        )
        result = await trader.evaluate_and_trade(sig, signal_id=1)
        assert result is not None
        # $400 * 50% = $200 budget, $80/contract = 2 contracts
        assert result["contracts"] >= 1
        assert result["total_cost"] <= 400.0


# ===========================================================================
# 19. DollarTrailExitGate (replaces VelocityExitGate)
# ===========================================================================


class TestDollarTrailExitGate:
    @pytest.mark.asyncio
    async def test_dollar_trail_triggers_when_profit_drops_below_stop(self):
        """Should exit when profit drops below stair-step stop level."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.35  # peak profit $35/contract
        ctx["exit_premium"] = 2.15  # profit $15, activation $20, stop $20 → exit
        ctx["settings"] = _MLFirstSettings(ENABLE_DOLLAR_TRAIL=True)
        gate = DollarTrailExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "dollar_trail" in outcome.gate_name

    @pytest.mark.asyncio
    async def test_dollar_trail_holds_above_stop(self):
        """Should hold when profit is above stop level."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.45  # peak $45, stop $40
        ctx["exit_premium"] = 2.42  # profit $42 > stop $40
        ctx["settings"] = _MLFirstSettings(ENABLE_DOLLAR_TRAIL=True)
        gate = DollarTrailExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_dollar_trail_skips_when_disabled(self):
        """Should skip when ENABLE_DOLLAR_TRAIL=False."""
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(ENABLE_DOLLAR_TRAIL=False)
        gate = DollarTrailExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_dollar_trail_dormant_below_activation(self):
        """Below 10% profit, trail is dormant → PASS."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.15  # peak $15 < activation $20
        ctx["exit_premium"] = 2.10
        ctx["settings"] = _MLFirstSettings(ENABLE_DOLLAR_TRAIL=True)
        gate = DollarTrailExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_dollar_trail_10_step_above_50(self):
        """$10 steps above $50 threshold. Peak $65, stop $60. Drop to $55 → exit."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.65  # peak $65
        ctx["exit_premium"] = 2.55  # profit $55 <= stop $60
        ctx["settings"] = _MLFirstSettings(ENABLE_DOLLAR_TRAIL=True)
        gate = DollarTrailExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL


# ===========================================================================
# 20. ProfitLockExitGate
# ===========================================================================


class TestProfitLockExitGate:
    @pytest.mark.asyncio
    async def test_profit_lock_triggers_when_below_floor(self):
        """After peak gain reached a tier, exit if current gain drops below lock floor."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 1.00
        ctx["trade"]["mfe_premium"] = 2.00  # peaked at +100%
        ctx["exit_premium"] = 1.20  # now at +20%, lock floor is +50% for tier 80:30
        ctx["settings"] = _MLFirstSettings(
            ENABLE_PROFIT_LOCK=True, PROFIT_LOCK_TIERS="80:30,150:70",
        )
        gate = ProfitLockExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "profit_lock" in outcome.gate_name

    @pytest.mark.asyncio
    async def test_profit_lock_holds_above_floor(self):
        """Should not exit if current gain is above the lock floor."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 1.00
        ctx["trade"]["mfe_premium"] = 2.00  # peaked at +100%
        ctx["exit_premium"] = 1.50  # now at +50%, lock floor for 80-tier is +30%
        ctx["settings"] = _MLFirstSettings(
            ENABLE_PROFIT_LOCK=True, PROFIT_LOCK_TIERS="80:30,150:70",
        )
        gate = ProfitLockExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_profit_lock_no_tier_reached(self):
        """Should pass if peak gain never reached any tier threshold."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 1.00
        ctx["trade"]["mfe_premium"] = 1.50  # peaked at +50%, lowest tier is 80
        ctx["exit_premium"] = 1.10
        ctx["settings"] = _MLFirstSettings(
            ENABLE_PROFIT_LOCK=True, PROFIT_LOCK_TIERS="80:30,150:70",
        )
        gate = ProfitLockExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS
        assert "hasn't reached" in outcome.reason

    @pytest.mark.asyncio
    async def test_profit_lock_highest_tier_applies(self):
        """The highest applicable tier lock should be used."""
        ctx = _ml_exit_ctx()
        ctx["trade"]["premium_per_contract"] = 1.00
        ctx["trade"]["mfe_premium"] = 2.60  # peaked at +160%, tier 150:70 applies
        ctx["exit_premium"] = 1.60  # now at +60%, below lock floor of +70%
        ctx["settings"] = _MLFirstSettings(
            ENABLE_PROFIT_LOCK=True, PROFIT_LOCK_TIERS="80:30,150:70,250:150",
        )
        gate = ProfitLockExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_profit_lock_skips_when_disabled(self):
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(ENABLE_PROFIT_LOCK=False)
        gate = ProfitLockExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP


# ===========================================================================
# 21. AdaptiveTimeTightenExitGate
# ===========================================================================


class TestAdaptiveTimeTightenExitGate:
    @pytest.mark.asyncio
    async def test_tighten_triggers_after_threshold(self):
        """After TIME_TIGHTEN_AFTER_MINUTES, trail should be tighter."""
        now = datetime(2026, 4, 8, 12, 30, tzinfo=ET)
        opened = (now - timedelta(minutes=90)).isoformat()  # 90 min ago
        ctx = _ml_exit_ctx()
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 3.00  # peaked at +50%
        ctx["trade"]["last_target_hit"] = 0  # phase 0, base trail 40%
        ctx["exit_premium"] = 2.50  # dropped 16.7% from peak
        ctx["now_et"] = now
        ctx["settings"] = _MLFirstSettings(
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_TIME_TIGHTEN=True,
            TIME_TIGHTEN_AFTER_MINUTES=60.0,
            TIME_TIGHTEN_FACTOR=0.7,  # 40% × 0.7 = 28% effective trail
        )
        gate = AdaptiveTimeTightenExitGate()
        outcome = await gate.evaluate(ctx)
        # 16.7% drop < 28% tightened trail → PASS
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_tighten_triggers_bigger_drop(self):
        """Bigger drop should trigger with tightened trail."""
        now = datetime(2026, 4, 8, 12, 30, tzinfo=ET)
        opened = (now - timedelta(minutes=90)).isoformat()
        ctx = _ml_exit_ctx()
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 3.00
        ctx["trade"]["last_target_hit"] = 0  # phase 0, base 40%
        ctx["exit_premium"] = 2.00  # dropped 33.3% from peak
        ctx["now_et"] = now
        ctx["settings"] = _MLFirstSettings(
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_TIME_TIGHTEN=True,
            TIME_TIGHTEN_AFTER_MINUTES=60.0,
            TIME_TIGHTEN_FACTOR=0.7,  # 40% × 0.7 = 28% effective trail
        )
        gate = AdaptiveTimeTightenExitGate()
        outcome = await gate.evaluate(ctx)
        # 33.3% drop >= 28% → FAIL
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_no_tighten_before_threshold(self):
        """Before TIME_TIGHTEN_AFTER_MINUTES, should not apply tightening."""
        now = datetime(2026, 4, 8, 11, 30, tzinfo=ET)
        opened = (now - timedelta(minutes=30)).isoformat()  # only 30 min
        ctx = _ml_exit_ctx()
        ctx["trade"]["opened_at"] = opened
        ctx["trade"]["mfe_premium"] = 3.00
        ctx["exit_premium"] = 2.40  # 20% drop from peak
        ctx["now_et"] = now
        ctx["settings"] = _MLFirstSettings(
            ENABLE_VINNY_STRATEGY=True,
            ENABLE_TIME_TIGHTEN=True,
            TIME_TIGHTEN_AFTER_MINUTES=60.0,
        )
        gate = AdaptiveTimeTightenExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS
        assert "30m < 60m" in outcome.reason

    @pytest.mark.asyncio
    async def test_tighten_skips_when_disabled(self):
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(ENABLE_TIME_TIGHTEN=False)
        gate = AdaptiveTimeTightenExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_tighten_skips_without_vinny(self):
        ctx = _ml_exit_ctx()
        ctx["settings"] = _MLFirstSettings(
            ENABLE_TIME_TIGHTEN=True, ENABLE_VINNY_STRATEGY=False,
        )
        gate = AdaptiveTimeTightenExitGate()
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP


# ===========================================================================
# 22. Safeguard ordering in pipeline
# ===========================================================================


class TestMLOverrideTargets:
    """ML override: when ML says HOLD with E[future] >= threshold, target gates
    should return scale-out-only exits (tagged with [ML_HOLD]) instead of full closes."""

    @pytest.mark.asyncio
    async def test_ml_hold_overrides_t1(self):
        """When ML says hold with high expected future PnL, T1 hit should be tagged [ML_HOLD]."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.15,
            expected_future_pnl=25.0,  # expects +25% more
            reason="ML hold: P(sell)=0.15 E[future]=+25.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            current_price=566.0,  # above T1=565.0
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "t1_hit"
        assert desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_ml_low_confidence_does_not_override(self):
        """When ML expects low future PnL, T1 should fire normally (no [ML_HOLD])."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.35,
            expected_future_pnl=2.0,  # only +2% expected — below threshold
            reason="ML hold: P(sell)=0.35 E[future]=+2.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            current_price=566.0,
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "t1_hit"
        assert not desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_ml_override_disabled(self):
        """When ML_OVERRIDE_TARGETS=False, T1 fires normally even if ML wants to hold."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=50.0,
            reason="ML hold: P(sell)=0.10 E[future]=+50.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            current_price=566.0,
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=False,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "t1_hit"
        assert not desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_ml_sell_still_overrides_targets(self):
        """When ML says SELL, it should fire immediately (not wait for targets)."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=True,
            sell_probability=0.70,
            expected_future_pnl=-15.0,
            reason="ML combo: P(sell)=0.70, E[future]=-15.0%",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            current_price=563.0,  # below T1=565.0
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "ml_sell"
        assert not desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_stop_loss_not_affected_by_ml_override(self):
        """Stop loss should always fire regardless of ML — it's before ML in pipeline."""
        ctx = _ml_exit_ctx(
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                PREMIUM_STOP_ENABLED=True,
                PREMIUM_STOP_PCT=60.0,
            ),
        )
        # Premium dropped 70% from entry
        ctx["exit_premium"] = 0.75  # entry was 2.50
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["trade"]["opened_at"] = (
            ctx["now_et"] - timedelta(minutes=30)
        ).isoformat()
        reason, desc = await run_exit_pipeline(ctx)
        assert reason == "stop_hit"

    @pytest.mark.asyncio
    async def test_ml_hold_overrides_t5_highest_target(self):
        """When ML holds and price hits T5 (highest defined target), the [ML_HOLD] tag
        should still be present — position_monitor does partial not full close."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=30.0,
            reason="ML hold: P(sell)=0.10 E[future]=+30.0% model=SPY",
            model_used="SPY",
        )
        # Trade has T1-T5 all defined; price hits T5
        ctx = _ml_exit_ctx(
            current_price=580.0,  # above T5=579.0
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        ctx["trade"].update({
            "target_1": 565.0,
            "target_2": 570.0,
            "target_3": 575.0,
            "target_4": 578.0,
            "target_5": 579.0,
            "last_target_hit": 4,  # T1-T4 already hit
        })
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "t5_hit"
        assert desc.startswith("[ML_HOLD]"), f"Expected [ML_HOLD] prefix but got: {desc}"

    @pytest.mark.asyncio
    async def test_ml_hold_with_scale_out_disabled(self):
        """ML_OVERRIDE=True + ENABLE_SCALE_OUT=False → ML override should STILL force
        scale-out via the `or ml_holding` logic in position_monitor.

        At the pipeline level, this means the exit pipeline should still return
        [ML_HOLD] tagged results even when ENABLE_SCALE_OUT is False, because the
        pipeline only checks ML_OVERRIDE_TARGETS, not ENABLE_SCALE_OUT."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.12,
            expected_future_pnl=20.0,
            reason="ML hold: P(sell)=0.12 E[future]=+20.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            current_price=566.0,  # above T1=565.0
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_SCALE_OUT=False,
                ENABLE_PARTIAL_PROFITS=False,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        # Pipeline returns [ML_HOLD] — position_monitor will use `or ml_holding`
        # to bypass the ENABLE_SCALE_OUT check
        assert reason == "t1_hit"
        assert desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_ml_override_does_not_affect_trailing_stop(self):
        """When ML holds but trailing stop fires, it should be a full close.

        Trailing stop is NOT a target gate — it should fire normally regardless
        of ML hold status."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=25.0,
            reason="ML hold: P(sell)=0.10 E[future]=+25.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_TRAILING_STOP=True,
                TRAILING_STOP_ACTIVATION_PCT=30.0,
                TRAILING_STOP_DROP_PCT=50.0,
                ENABLE_VELOCITY_EXIT=False, ENABLE_DOLLAR_TRAIL=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        # Trailing stop scenario: premium peaked at 5.00 (100% gain from 2.50)
        # Now dropped to 2.00 → 60% drop from peak → above 50% threshold
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["trade"]["mfe_premium"] = 5.00
        ctx["exit_premium"] = 2.00

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "trailing_stop"
        # Trailing stop should NOT have [ML_HOLD] tag — it's a full close
        assert not desc.startswith("[ML_HOLD]")

    @pytest.mark.asyncio
    async def test_ml_override_does_not_affect_dollar_trail(self):
        """Dollar trail should fire normally regardless of ML hold.

        Dollar trail is not a target gate, so ML override should not convert it."""
        from options_owl.risk.ml_exit import MLSellSignal

        mock_signal = MLSellSignal(
            should_sell=False,
            sell_probability=0.10,
            expected_future_pnl=25.0,
            reason="ML hold: P(sell)=0.10 E[future]=+25.0% model=SPY",
            model_used="SPY",
        )
        ctx = _ml_exit_ctx(
            settings=_MLFirstSettings(
                ML_OVERRIDE_TARGETS=True,
                ML_OVERRIDE_MIN_FUTURE_PNL=5.0,
                ENABLE_DOLLAR_TRAIL=True,
                ENABLE_TRAILING_STOP=False,
                ENABLE_PROFIT_LOCK=False,
                ENABLE_TIME_TIGHTEN=False,
            ),
        )
        # Dollar trail: entry $2.00, peak $2.35 ($35 profit), current $2.15 ($15)
        # Activation=$20, stop=$20. $15 <= $20 → exit
        ctx["trade"]["premium_per_contract"] = 2.00
        ctx["trade"]["mfe_premium"] = 2.35
        ctx["exit_premium"] = 2.15

        with patch("options_owl.risk.ml_exit.predict_sell", return_value=mock_signal):
            reason, desc = await run_exit_pipeline(ctx)
        assert reason == "dollar_trail"
        assert not desc.startswith("[ML_HOLD]")


class TestSafeguardOrdering:
    def test_soft_trail_before_adaptive(self):
        """v2.2: Soft trail fires before adaptive trail."""
        names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert names.index("soft_trail") < names.index("adaptive_trailing_stop")

    def test_adaptive_trail_before_dollar_trail(self):
        """v2.2: Adaptive trail fires before dollar trail (dollar moved to Tier 8)."""
        names = [g.name for g in (g() for g in DEFAULT_EXIT_GATES)]
        assert names.index("adaptive_trailing_stop") < names.index("dollar_trail")
