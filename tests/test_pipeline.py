"""Tests for the trade pipeline state machine."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from options_owl.risk.pipeline import (
    BalanceGate,
    CircuitBreakerGate,
    ConcurrentPositionsGate,
    DailyLossGate,
    DuplicateTickerGate,
    EODExitGate,
    EXIT_GATE_TO_REASON,
    GateResult,
    PipelineResult,
    PremiumGate,
    ScoreGate,
    StopLossExitGate,
    StopPriceGate,
    Target1ExitGate,
    Target2ExitGate,
    ThetaDecayExitGate,
    TimeExpiryExitGate,
    run_entry_pipeline,
    run_exit_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSignal:
    def __init__(self, **kwargs):
        defaults = {
            "ticker": "SPY",
            "score": 85,
            "atm_premium": 2.50,
            "stop_price": 555.0,
            "bot_source": type("BS", (), {"value": "Captain Hook"})(),
            "direction": type("D", (), {"value": "call"})(),
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class _FakeSettings:
    def __init__(self, **kwargs):
        defaults = {
            "MIN_SCORE": 75,
            "PORTFOLIO_SIZE": 5000.0,
            "MAX_POSITION_PCT": 20.0,
            "MAX_CONCURRENT": 3,
            "DAILY_LOSS_LIMIT_PCT": 10.0,
            "MAX_PORTFOLIO_RISK_PCT": 60.0,
            "MAX_LOSS_PER_TRADE_PCT": 20.0,
            "WEEKLY_LOSS_LIMIT_PCT": 20.0,
            "ENABLE_RISK_MANAGER": True,
            "ENABLE_IV_FILTER": False,
            "ENABLE_VIX_FILTER": False,
            "ENABLE_ANALYST_FILTER": False,
            "ENABLE_CIRCUIT_BREAKERS": False,
            "ENABLE_THETA_DECAY_EXIT": False,
            "PREMIUM_STOP_ENABLED": False,
            "MIN_UNDERLYING_STOP_PCT": 0.5,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def _base_entry_ctx(**overrides):
    ctx = {
        "signal": _FakeSignal(),
        "settings": _FakeSettings(),
        "db_path": ":memory:",
        "portfolio": {
            "current_balance": 5000.0,
            "daily_pnl": 0.0,
            "last_trade_date": None,
        },
        "open_count": 0,
        "open_tickers": set(),
    }
    ctx.update(overrides)
    return ctx


def _base_exit_ctx(**overrides):
    from datetime import datetime

    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        from datetime import timezone, timedelta
        et = timezone(timedelta(hours=-5))

    ctx = {
        "trade": {
            "option_type": "call",
            "stop_price": 550.0,
            "target_1": 565.0,
            "target_2": 570.0,
            "exit_by": None,
            "premium_per_contract": 2.50,
            "entry_price": 560.0,
            "strike": 560.0,
            "expiry_date": "2026-03-30",
            "opened_at": "2026-03-30T10:00:00",
        },
        "current_price": 562.0,
        "exit_premium": 3.50,
        "now_et": datetime(2026, 3, 30, 11, 0, tzinfo=et),
        "settings": _FakeSettings(ENABLE_THETA_DECAY_EXIT=False),
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Entry gate unit tests
# ---------------------------------------------------------------------------

class TestScoreGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        ctx = _base_entry_ctx()
        r = await ScoreGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail(self):
        ctx = _base_entry_ctx(signal=_FakeSignal(score=50))
        r = await ScoreGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestPremiumGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        r = await PremiumGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail_no_premium(self):
        ctx = _base_entry_ctx(signal=_FakeSignal(atm_premium=None))
        r = await PremiumGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestStopPriceGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        r = await StopPriceGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail(self):
        ctx = _base_entry_ctx(signal=_FakeSignal(stop_price=None))
        r = await StopPriceGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestDailyLossGate:
    @pytest.mark.asyncio
    async def test_pass_new_day(self):
        r = await DailyLossGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail_limit_hit(self):
        from options_owl.risk.pipeline import _now_et
        today = _now_et().strftime("%Y-%m-%d")
        ctx = _base_entry_ctx(portfolio={
            "current_balance": 4000.0,
            "daily_pnl": -600.0,  # exceeds 10% of 5000
            "last_trade_date": today,
        })
        r = await DailyLossGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestConcurrentPositionsGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        r = await ConcurrentPositionsGate().evaluate(_base_entry_ctx(open_count=1))
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail(self):
        r = await ConcurrentPositionsGate().evaluate(_base_entry_ctx(open_count=3))
        assert r.result == GateResult.FAIL


class TestDuplicateTickerGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        r = await DuplicateTickerGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail_same_direction(self):
        ctx = _base_entry_ctx(
            open_tickers={"SPY"},
            open_positions=[("SPY", "call")],
        )
        r = await DuplicateTickerGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_pass_signal_flip(self):
        """Opposite direction on same ticker should PASS (signal flip)."""
        ctx = _base_entry_ctx(
            open_tickers={"SPY"},
            open_positions=[("SPY", "put")],  # existing put, new signal is call
        )
        r = await DuplicateTickerGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert ctx.get("signal_flip_ticker") == "SPY"


class TestBalanceGate:
    @pytest.mark.asyncio
    async def test_pass(self):
        r = await BalanceGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail(self):
        ctx = _base_entry_ctx(portfolio={"current_balance": 100.0})
        r = await BalanceGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestCircuitBreakerGate:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        r = await CircuitBreakerGate().evaluate(_base_entry_ctx())
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_pass_when_enabled(self):
        ctx = _base_entry_ctx(settings=_FakeSettings(ENABLE_CIRCUIT_BREAKERS=True))
        with patch("options_owl.risk.circuit_breaker.CircuitBreaker.check_all",
                    new_callable=AsyncMock, return_value=(True, [])):
            r = await CircuitBreakerGate().evaluate(ctx)
            assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail_when_tripped(self):
        ctx = _base_entry_ctx(settings=_FakeSettings(ENABLE_CIRCUIT_BREAKERS=True))
        with patch("options_owl.risk.circuit_breaker.CircuitBreaker.check_all",
                    new_callable=AsyncMock, return_value=(False, ["3 consecutive losses"])):
            r = await CircuitBreakerGate().evaluate(ctx)
            assert r.result == GateResult.FAIL


# ---------------------------------------------------------------------------
# Exit gate unit tests
# ---------------------------------------------------------------------------

class TestStopLossExit:
    @pytest.mark.asyncio
    async def test_underlying_stop_disabled_by_default(self):
        """With ENABLE_UNDERLYING_STOP=False (default), underlying stops should SKIP."""
        ctx = _base_exit_ctx(current_price=549.0)
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_call_stop_hit_when_enabled(self):
        """Underlying stop triggers when explicitly enabled."""
        ctx = _base_exit_ctx(current_price=549.0)
        ctx["settings"] = _FakeSettings(ENABLE_UNDERLYING_STOP=True)
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_put_stop_hit_when_enabled(self):
        ctx = _base_exit_ctx(current_price=572.0)
        ctx["settings"] = _FakeSettings(ENABLE_UNDERLYING_STOP=True)
        ctx["trade"]["option_type"] = "put"
        ctx["trade"]["entry_price"] = 560.0
        ctx["trade"]["stop_price"] = 570.0
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_premium_stop_triggered(self):
        """Premium drops 44% from entry — should trigger with 35% threshold."""
        ctx = _base_exit_ctx(current_price=558.0)
        ctx["settings"] = _FakeSettings(
            PREMIUM_STOP_ENABLED=True, PREMIUM_STOP_PCT=35.0, STOP_GRACE_PERIOD_MINUTES=0,
        )
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 1.40  # -44% drop
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL
        assert "Premium stop" in r.reason

    @pytest.mark.asyncio
    async def test_premium_stop_not_triggered(self):
        """Premium only drops 10% — should hold."""
        ctx = _base_exit_ctx(current_price=558.0)
        ctx["settings"] = _FakeSettings(
            PREMIUM_STOP_ENABLED=True, PREMIUM_STOP_PCT=35.0, STOP_GRACE_PERIOD_MINUTES=0,
        )
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["exit_premium"] = 2.25  # -10% drop
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_grace_period_blocks_stop(self):
        """During grace period, stop should PASS even if premium is down."""
        from datetime import datetime, timedelta
        ctx = _base_exit_ctx(current_price=558.0)
        ctx["settings"] = _FakeSettings(
            PREMIUM_STOP_ENABLED=True, PREMIUM_STOP_PCT=35.0, STOP_GRACE_PERIOD_MINUTES=5,
        )
        ctx["trade"]["premium_per_contract"] = 2.50
        ctx["trade"]["opened_at"] = (ctx["now_et"] - timedelta(minutes=2)).isoformat()
        ctx["exit_premium"] = 1.70  # -32% drop — would trigger stop but not catastrophic
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS
        assert "grace" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_min_underlying_stop_distance(self):
        """Stop too close to entry should be widened by MIN_UNDERLYING_STOP_PCT."""
        ctx = _base_exit_ctx(current_price=559.5)
        ctx["settings"] = _FakeSettings(
            PREMIUM_STOP_ENABLED=False, ENABLE_UNDERLYING_STOP=True,
            MIN_UNDERLYING_STOP_PCT=0.5, STOP_GRACE_PERIOD_MINUTES=0,
        )
        ctx["trade"]["entry_price"] = 560.0
        ctx["trade"]["stop_price"] = 559.9
        r = await StopLossExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestTarget2Exit:
    @pytest.mark.asyncio
    async def test_hit(self):
        ctx = _base_exit_ctx(current_price=571.0)
        r = await Target2ExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_not_hit(self):
        ctx = _base_exit_ctx(current_price=565.0)
        r = await Target2ExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestTarget1Exit:
    @pytest.mark.asyncio
    async def test_hit(self):
        ctx = _base_exit_ctx(current_price=566.0)
        r = await Target1ExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_not_hit(self):
        ctx = _base_exit_ctx(current_price=563.0)
        r = await Target1ExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS


class TestEODExit:
    @pytest.mark.asyncio
    async def test_before_cutoff(self):
        ctx = _base_exit_ctx()  # 11:00 ET
        r = await EODExitGate().evaluate(ctx)
        assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_after_cutoff(self):
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            from datetime import timezone, timedelta
            et = timezone(timedelta(hours=-5))
        ctx = _base_exit_ctx(now_et=datetime(2026, 3, 30, 15, 50, tzinfo=et))
        r = await EODExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL


class TestTimeExpiryExit:
    @pytest.mark.asyncio
    async def test_skip_no_exit_by(self):
        ctx = _base_exit_ctx()
        r = await TimeExpiryExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_hit(self):
        ctx = _base_exit_ctx()
        ctx["trade"]["exit_by"] = "10:30"
        r = await TimeExpiryExitGate().evaluate(ctx)
        assert r.result == GateResult.FAIL  # now is 11:00, past 10:30


class TestThetaDecayExit:
    @pytest.mark.asyncio
    async def test_skip_disabled(self):
        ctx = _base_exit_ctx()
        r = await ThetaDecayExitGate().evaluate(ctx)
        assert r.result == GateResult.SKIP


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestEntryPipeline:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        ctx = _base_entry_ctx(settings=_FakeSettings(ENABLE_RISK_MANAGER=False))
        result = await run_entry_pipeline(ctx, gates=[ScoreGate, PremiumGate, StopPriceGate])
        assert result.approved is True
        assert len(result.failures) == 0

    @pytest.mark.asyncio
    async def test_one_fails(self):
        ctx = _base_entry_ctx(signal=_FakeSignal(score=10))
        result = await run_entry_pipeline(ctx, gates=[ScoreGate, PremiumGate])
        assert result.approved is False
        assert len(result.failures) == 1
        assert result.failures[0].gate_name == "score"

    @pytest.mark.asyncio
    async def test_multiple_fail(self):
        ctx = _base_entry_ctx(
            signal=_FakeSignal(score=10, atm_premium=None),
        )
        result = await run_entry_pipeline(ctx, gates=[ScoreGate, PremiumGate])
        assert result.approved is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_summary_contains_all_gates(self):
        ctx = _base_entry_ctx()
        result = await run_entry_pipeline(ctx, gates=[ScoreGate, PremiumGate])
        summary = result.summary()
        assert "score" in summary
        assert "premium" in summary


class TestExitPipeline:
    @pytest.mark.asyncio
    async def test_no_exit(self):
        ctx = _base_exit_ctx(current_price=562.0)
        reason, desc = await run_exit_pipeline(
            ctx, gates=[StopLossExitGate, Target2ExitGate, Target1ExitGate]
        )
        assert reason is None

    @pytest.mark.asyncio
    async def test_stop_priority(self):
        # Premium stop triggered AND T2 hit — stop should win (runs first in pipeline)
        ctx = _base_exit_ctx(current_price=549.0)
        ctx["trade"]["target_2"] = 540.0  # also hit for a call
        ctx["trade"]["premium_per_contract"] = 5.0
        ctx["exit_premium"] = 1.5  # down 70% from entry (> 60% threshold)
        ctx["trade"]["opened_at"] = (
            ctx["now_et"] - __import__("datetime").timedelta(minutes=10)
        ).isoformat()  # past grace period
        ctx["settings"] = type("S", (), {
            "PREMIUM_STOP_ENABLED": True,
            "PREMIUM_STOP_PCT": 60.0,
            "STOP_GRACE_PERIOD_MINUTES": 0,
            "ENABLE_UNDERLYING_STOP": False,
            "ENABLE_TRAILING_STOP": False,
        })()
        reason, desc = await run_exit_pipeline(
            ctx, gates=[StopLossExitGate, Target2ExitGate, Target1ExitGate]
        )
        assert reason == "stop_hit"

    @pytest.mark.asyncio
    async def test_t1_triggers(self):
        ctx = _base_exit_ctx(current_price=566.0)
        reason, desc = await run_exit_pipeline(
            ctx, gates=[StopLossExitGate, Target2ExitGate, Target1ExitGate]
        )
        assert reason == "t1_hit"

    @pytest.mark.asyncio
    async def test_t2_before_t1(self):
        ctx = _base_exit_ctx(current_price=571.0)
        reason, desc = await run_exit_pipeline(
            ctx, gates=[StopLossExitGate, Target2ExitGate, Target1ExitGate]
        )
        assert reason == "t2_hit"

    @pytest.mark.asyncio
    async def test_exit_gate_reason_mapping(self):
        assert EXIT_GATE_TO_REASON["stop_loss"] == "stop_hit"
        assert EXIT_GATE_TO_REASON["target_1"] == "t1_hit"
        assert EXIT_GATE_TO_REASON["theta_decay"] == "theta_decay"


class TestPipelineResult:
    def test_failure_reasons(self):
        from options_owl.risk.pipeline import GateOutcome
        result = PipelineResult(
            approved=False,
            outcomes=[
                GateOutcome("a", GateResult.PASS, "ok"),
                GateOutcome("b", GateResult.FAIL, "bad"),
                GateOutcome("c", GateResult.SKIP, "disabled"),
                GateOutcome("d", GateResult.FAIL, "also bad"),
            ],
        )
        assert result.failure_reasons == ["bad", "also bad"]
        assert len(result.failures) == 2
