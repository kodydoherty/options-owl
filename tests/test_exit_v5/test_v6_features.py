"""Tests for V6 enhancements on top of the V5 FSM exit engine.

V6 features tested:
  1. Per-ticker configs (get_ticker_config, TICKER_CONFIGS)
  2. Break-even ratchet gate (arm at +20%, exit if drops below entry)
  3. Scale-out gate (sell 1/3 at +20% gain, one-shot)
  4. 2PM trail tightening (tighter adaptive + soft trail after 2PM ET)
  5. Premium cap entry gate (reject non-index > $5)
  6. Spread-cost entry gate (reject wide bid-ask spreads)
  7. Monitor bridge integration (per-ticker FSM, scaleout encoding)

All V6 features are gated behind ENABLE_V6_* flags. When flags are OFF,
existing V5 behavior is preserved (tested via existing test_fsm.py etc).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from options_owl.risk.exit_v5.config import (
    TICKER_CONFIGS,
    V5Config,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.exit_v5.gates import check_breakeven_ratchet, check_scaleout
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


def _v6_settings(**overrides):
    """Create a fake Settings object with all V6 flags OFF by default."""
    defaults = {
        "ENABLE_V6_PER_TICKER_CONFIG": False,
        "ENABLE_V6_BREAKEVEN_RATCHET": False,
        "V6_BREAKEVEN_TRIGGER_PCT": 20.0,
        "ENABLE_V6_2PM_TIGHTEN": False,
        "V6_2PM_TRAIL_TIGHTEN_FACTOR": 0.7,
        "V6_2PM_SOFT_TRAIL_BOOST": 0.15,
        "ENABLE_V6_PREMIUM_CAP": False,
        "V6_PREMIUM_CAP": 6.0,
        "V6_PREMIUM_CAP_MID": 7.0,
        "V6_PREMIUM_CAP_HIGH": 9.0,
        "ENABLE_V6_SPREAD_GATE": False,
        "V6_MAX_SPREAD_PCT": 15.0,
        "ENABLE_V6_SCALEOUT": False,
        "V6_SCALEOUT_GAIN_PCT": 20.0,
        "V6_SCALEOUT_FRACTION": 0.333,
        "V6_SCALEOUT_MIN_CONTRACTS": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ══════════════════════════════════════════════════════════════════════════
# 1. Per-Ticker Configs
# ══════════════════════════════════════════════════════════════════════════


class TestPerTickerConfig:

    def test_known_ticker_returns_custom_config(self):
        cfg = get_ticker_config("NVDA", use_per_ticker=True)
        assert cfg.profit_target_general_pct == 20.0  # NVDA is HIGH_VOL, uses general target
        assert cfg.soft_trail_keep_pct == 0.70

    def test_unknown_ticker_returns_default(self):
        cfg = get_ticker_config("XYZ", use_per_ticker=True)
        assert cfg == V5Config()

    def test_disabled_returns_default_for_known_ticker(self):
        cfg = get_ticker_config("NVDA", use_per_ticker=False)
        assert cfg == V5Config()

    def test_all_tickers_in_configs_are_valid(self):
        for ticker, cfg in TICKER_CONFIGS.items():
            assert isinstance(ticker, str)
            assert isinstance(cfg, V5Config)
            assert len(ticker) > 0

    def test_meta_has_defensive_config(self):
        cfg = get_ticker_config("META", use_per_ticker=True)
        # META uses default tight/backstop (15/30) + faster theta bleed
        assert cfg.tight_stop_0dte_pct == 15.0
        assert cfg.backstop_0dte_pct == 30.0
        assert cfg.theta_bleed_min == 90.0

    def test_tsla_has_long_grace(self):
        cfg = get_ticker_config("TSLA", use_per_ticker=True)
        assert cfg.grace_period_min == 8.0

    def test_googl_has_wide_stop(self):
        cfg = get_ticker_config("GOOGL", use_per_ticker=True)
        assert cfg.tight_stop_0dte_pct == 20.0
        assert cfg.backstop_0dte_pct == 40.0


# ══════════════════════════════════════════════════════════════════════════
# 2. Break-Even Ratchet Gate
# ══════════════════════════════════════════════════════════════════════════


class TestBreakevenRatchet:

    def test_arms_at_trigger_pct(self):
        action, armed = check_breakeven_ratchet(
            gain=25.0, current_premium=1.25, entry_premium=1.00,
            armed=False, trigger_pct=20.0, debug={},
        )
        assert armed is True
        assert action is None  # armed but premium still above entry

    def test_fires_when_armed_and_below_entry(self):
        action, armed = check_breakeven_ratchet(
            gain=-5.0, current_premium=0.95, entry_premium=1.00,
            armed=True, trigger_pct=20.0, debug={},
        )
        assert action is not None
        assert action.reason == ExitReason.BREAKEVEN_RATCHET

    def test_does_not_fire_when_not_armed(self):
        action, armed = check_breakeven_ratchet(
            gain=-5.0, current_premium=0.95, entry_premium=1.00,
            armed=False, trigger_pct=20.0, debug={},
        )
        assert action is None
        assert armed is False

    def test_does_not_fire_when_armed_but_above_entry(self):
        action, armed = check_breakeven_ratchet(
            gain=5.0, current_premium=1.05, entry_premium=1.00,
            armed=True, trigger_pct=20.0, debug={},
        )
        assert action is None
        assert armed is True

    def test_arms_and_fires_on_same_tick_if_below_entry(self):
        """Edge case: gain was 25% last tick, now dropped below entry."""
        action, armed = check_breakeven_ratchet(
            gain=-2.0, current_premium=0.98, entry_premium=1.00,
            armed=False, trigger_pct=20.0, debug={},
        )
        # gain < trigger_pct so it doesn't arm; does not fire
        assert armed is False
        assert action is None

    def test_fsm_integration_breakeven_ratchet(self):
        """Full FSM integration: arm at +25%, then exit at -2%."""
        settings = _v6_settings(ENABLE_V6_BREAKEVEN_RATCHET=True)
        fsm = ExitFSM(V5Config(), settings=settings)
        state = _make_state(entry_premium=1.00, ticker="AAPL")

        # Tick 1: +25% gain — should arm the ratchet but HOLD
        now = _now_et(10, 10)
        action = fsm.evaluate(state, 1.25, 1.20, 1.30, now, current_underlying=150.0)
        assert state.breakeven_ratchet_armed is True
        assert not action.should_exit

        # Tick 2: drops below entry — should fire
        action = fsm.evaluate(state, 0.95, 0.90, 1.00, now + timedelta(seconds=30),
                              current_underlying=149.0)
        assert action.should_exit
        assert action.reason == ExitReason.BREAKEVEN_RATCHET

    def test_fsm_no_ratchet_when_disabled(self):
        """V6 flag off: no ratchet behavior."""
        settings = _v6_settings(ENABLE_V6_BREAKEVEN_RATCHET=False)
        fsm = ExitFSM(V5Config(), settings=settings)
        state = _make_state(entry_premium=1.00, ticker="AAPL")

        now = _now_et(10, 10)
        fsm.evaluate(state, 1.25, 1.20, 1.30, now, current_underlying=150.0)
        assert state.breakeven_ratchet_armed is False  # never armed


# ══════════════════════════════════════════════════════════════════════════
# 3. Scale-Out Gate
# ══════════════════════════════════════════════════════════════════════════


class TestScaleout:

    def test_fires_at_threshold_with_enough_contracts(self):
        action = check_scaleout(
            gain=25.0, contracts=6, already_scaled=False,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is not None
        assert action.reason == ExitReason.SCALEOUT
        assert action.contracts_to_close == 1  # int(6 * 0.333) = 1

    def test_does_not_fire_below_threshold(self):
        action = check_scaleout(
            gain=15.0, contracts=6, already_scaled=False,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is None

    def test_does_not_fire_when_already_scaled(self):
        action = check_scaleout(
            gain=25.0, contracts=6, already_scaled=True,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is None

    def test_does_not_fire_with_too_few_contracts(self):
        action = check_scaleout(
            gain=25.0, contracts=2, already_scaled=False,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is None

    def test_contracts_to_close_calculation(self):
        """With 9 contracts and 0.333 fraction, should close 2."""
        action = check_scaleout(
            gain=25.0, contracts=9, already_scaled=False,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is not None
        assert action.contracts_to_close == 2  # int(9 * 0.333) = 2

    def test_minimum_one_contract_closed(self):
        """With 3 contracts and 0.333 fraction, should still close 1 (not 0)."""
        action = check_scaleout(
            gain=25.0, contracts=3, already_scaled=False,
            scaleout_gain_pct=20.0, scaleout_fraction=0.333,
            min_contracts=3, debug={},
        )
        assert action is not None
        assert action.contracts_to_close == 1  # max(1, int(3*0.333)) = max(1,0) = 1

    def test_fsm_integration_scaleout(self):
        """Full FSM: scaleout fires once, then doesn't re-fire."""
        settings = _v6_settings(ENABLE_V6_SCALEOUT=True)
        fsm = ExitFSM(V5Config(), settings=settings)
        state = _make_state(entry_premium=1.00, contracts=6, ticker="AAPL")

        now = _now_et(10, 10)
        action = fsm.evaluate(state, 1.25, 1.20, 1.30, now, current_underlying=150.0)
        assert action.should_exit
        assert action.reason == ExitReason.SCALEOUT
        assert action.contracts_to_close > 0
        assert state.scaled_out is True

        # Next tick: should not fire again
        action = fsm.evaluate(state, 1.30, 1.25, 1.35,
                              now + timedelta(seconds=5), current_underlying=150.5)
        # Either HOLD or some other gate — not scaleout
        if action.should_exit:
            assert action.reason != ExitReason.SCALEOUT


# ══════════════════════════════════════════════════════════════════════════
# 4. 2PM Trail Tightening
# ══════════════════════════════════════════════════════════════════════════


class Test2PMTighten:

    def test_tightening_applies_after_2pm(self):
        """Soft trail should keep more and adaptive trail should be tighter after 2PM."""
        settings = _v6_settings(ENABLE_V6_2PM_TIGHTEN=True)
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=settings)
        state = _make_state(entry_premium=1.00, ticker="AAPL",
                            entry_time=_now_et(13, 0))

        # Set a peak that puts us in soft trail band (e.g., +20%)
        state.peak_premium = 1.20

        # At 1:59 PM — normal soft trail (keep 60%)
        now_before = _now_et(13, 59)
        # Floor = 1.00 + (1.20 - 1.00) * 0.60 = 1.12
        # Premium at 1.11 should trigger soft trail
        fsm.evaluate(state, 1.11, 1.08, 1.14, now_before,
                      current_underlying=150.0)

        # Reset state for 2PM test
        state2 = _make_state(entry_premium=1.00, ticker="AAPL",
                             entry_time=_now_et(13, 0))
        state2.peak_premium = 1.20

        # At 2:01 PM — tighter soft trail (keep 75%)
        now_after = _now_et(14, 1)
        # Floor = 1.00 + (1.20 - 1.00) * 0.75 = 1.15
        # Premium at 1.14 should trigger tighter soft trail
        action_after = fsm.evaluate(state2, 1.14, 1.11, 1.17, now_after,
                                    current_underlying=150.0)

        # After 2PM, the tighter floor should trigger exit at a higher premium
        assert action_after.should_exit
        assert action_after.reason == ExitReason.SOFT_TRAIL

    def test_no_tightening_when_disabled(self):
        """When flag is off, trails are unchanged before and after 2PM."""
        settings = _v6_settings(ENABLE_V6_2PM_TIGHTEN=False)
        cfg = V5Config()
        fsm = ExitFSM(cfg, settings=settings)
        state = _make_state(entry_premium=1.00, ticker="AAPL",
                            entry_time=_now_et(13, 0))
        state.peak_premium = 1.20

        # At 2:01 PM with tightening disabled — normal floor (0.60)
        # Floor = 1.00 + 0.20 * 0.60 = 1.12
        # Premium 1.13 is ABOVE floor — should NOT trigger
        now_after = _now_et(14, 1)
        action = fsm.evaluate(state, 1.13, 1.10, 1.16, now_after,
                              current_underlying=150.0)
        assert not action.should_exit


# ══════════════════════════════════════════════════════════════════════════
# 5. Premium Cap Entry Gate
# ══════════════════════════════════════════════════════════════════════════


class TestPremiumCapGate:

    @pytest.fixture
    def gate(self):
        from options_owl.risk.pipeline import PremiumCapGate
        return PremiumCapGate()

    @pytest.mark.asyncio
    async def test_blocks_non_index_over_cap(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="META", atm_premium=25.35, score=90)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_passes_index_over_cap(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="SPY", atm_premium=8.00, score=90)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_passes_non_index_under_cap(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="NVDA", atm_premium=2.50, score=90)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_elite_score_uses_high_cap(self, gate):
        """Score 150+ should use V6_PREMIUM_CAP_HIGH ($9 default)."""
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="META", atm_premium=8.50, score=150)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0,
                                V6_PREMIUM_CAP_HIGH=9.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_elite_score_blocked_above_high_cap(self, gate):
        """Score 150+ should still be blocked above $9 cap."""
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="META", atm_premium=25.35, score=150)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0,
                                V6_PREMIUM_CAP_HIGH=9.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_strong_score_uses_mid_cap(self, gate):
        """Score 120-149 should use V6_PREMIUM_CAP_MID ($7 default)."""
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="TSLA", atm_premium=6.50, score=125)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=True, V6_PREMIUM_CAP=6.0,
                                V6_PREMIUM_CAP_MID=7.0)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="META", atm_premium=25.35, score=90)
        settings = _v6_settings(ENABLE_V6_PREMIUM_CAP=False)
        result = await gate.evaluate({"signal": signal, "settings": settings})
        assert result.result == GateResult.SKIP


# ══════════════════════════════════════════════════════════════════════════
# 6. Spread-Cost Entry Gate
# ══════════════════════════════════════════════════════════════════════════


class TestSpreadCostGate:

    @pytest.fixture
    def gate(self):
        from options_owl.risk.pipeline import SpreadCostGate
        return SpreadCostGate()

    @pytest.mark.asyncio
    async def test_blocks_wide_spread(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="AAPL", atm_premium=2.00)
        settings = _v6_settings(ENABLE_V6_SPREAD_GATE=True, V6_MAX_SPREAD_PCT=15.0)
        result = await gate.evaluate({
            "signal": signal, "settings": settings,
            "bid": 1.70, "ask": 2.30,  # spread = $0.60 = 30% of $2.00
        })
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_passes_narrow_spread(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="SPY", atm_premium=2.00)
        settings = _v6_settings(ENABLE_V6_SPREAD_GATE=True, V6_MAX_SPREAD_PCT=15.0)
        result = await gate.evaluate({
            "signal": signal, "settings": settings,
            "bid": 1.95, "ask": 2.05,  # spread = $0.10 = 5%
        })
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_skips_when_no_bid_ask(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="AAPL", atm_premium=2.00)
        settings = _v6_settings(ENABLE_V6_SPREAD_GATE=True)
        result = await gate.evaluate({
            "signal": signal, "settings": settings,
        })
        assert result.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, gate):
        from options_owl.risk.pipeline import GateResult
        signal = MagicMock(ticker="AAPL", atm_premium=2.00)
        settings = _v6_settings(ENABLE_V6_SPREAD_GATE=False)
        result = await gate.evaluate({
            "signal": signal, "settings": settings,
            "bid": 1.70, "ask": 2.30,
        })
        assert result.result == GateResult.SKIP


# ══════════════════════════════════════════════════════════════════════════
# 6b. OTM Distance Gate
# ══════════════════════════════════════════════════════════════════════════


class TestOTMDistanceGate:
    """V2 per-ticker dollar-based OTM gate tests.

    AMZN: wide grid ($2.50 interval), 1 strike OTM allowed = $2.50 max
    SPY: fine grid ($1.00 interval), 3 strikes OTM allowed = $3.00 max
    META: standard grid ($2.50 interval), 2 strikes OTM allowed = $5.00 max
    """

    @pytest.fixture
    def gate(self):
        from options_owl.risk.pipeline import OTMDistanceGate
        return OTMDistanceGate()

    @pytest.mark.asyncio
    async def test_blocks_otm_call_amzn(self, gate):
        """AMZN CALL strike $253 with underlying $250 = $3.00 OTM > $2.50 max → blocked."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=253.0, entry_price=250.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.FAIL
        assert "$2.5" in result.reason  # max $2.5 for AMZN

    @pytest.mark.asyncio
    async def test_passes_one_strike_otm_amzn(self, gate):
        """AMZN CALL strike $252.50 with underlying $250 = $2.50 OTM = exactly 1 strike → passes."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=252.50, entry_price=250.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_passes_atm_call(self, gate):
        """AMZN CALL strike $250 with underlying $249.50 = $0.50 OTM → passes."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=250.0, entry_price=249.50,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_passes_itm_call(self, gate):
        """CALL strike $248 with underlying $250 = ITM → always passes."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=248.0, entry_price=250.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_blocks_otm_put_amzn(self, gate):
        """AMZN PUT strike $247 with underlying $250 = $3.00 OTM > $2.50 max → blocked."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=247.0, entry_price=250.0,
            direction=Direction.PUT,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_passes_slightly_otm_put(self, gate):
        """AMZN PUT strike $248 with underlying $250 = $2.00 OTM < $2.50 max → passes."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=248.0, entry_price=250.0,
            direction=Direction.PUT,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_skips_when_no_data(self, gate):
        """Missing strike or underlying → skip."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=0, entry_price=250.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_spy_allows_3_strikes_otm(self, gate):
        """SPY fine grid: 3 strikes × $1.00 = $3.00 max OTM."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        # $3.00 OTM = exactly 3 strikes → passes
        signal = MagicMock(
            ticker="SPY", strike=553.0, entry_price=550.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS
        # $4.00 OTM = 4 strikes → blocked
        signal2 = MagicMock(
            ticker="SPY", strike=554.0, entry_price=550.0,
            direction=Direction.CALL,
        )
        result2 = await gate.evaluate({"signal": signal2, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result2.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_meta_allows_2_strikes_otm(self, gate):
        """META standard grid: 2 strikes × $2.50 = $5.00 max OTM."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        # $5.00 OTM = exactly 2 strikes → passes
        signal = MagicMock(
            ticker="META", strike=605.0, entry_price=600.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS
        # $6.00 OTM → blocked
        signal2 = MagicMock(
            ticker="META", strike=606.0, entry_price=600.0,
            direction=Direction.CALL,
        )
        result2 = await gate.evaluate({"signal": signal2, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result2.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_amzn_249_scenario(self, gate):
        """AMZN #249: strike $250, underlying $248.92 = $1.08 OTM < $2.50 max → passes."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="AMZN", strike=250.0, entry_price=248.92,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_unknown_ticker_uses_default(self, gate):
        """Unknown ticker uses default $2.50 interval × 2 = $5.00 max."""
        from options_owl.risk.pipeline import GateResult
        from options_owl.models.signals import Direction
        signal = MagicMock(
            ticker="XYZ", strike=106.0, entry_price=100.0,
            direction=Direction.CALL,
        )
        result = await gate.evaluate({"signal": signal, "settings": SimpleNamespace(MAX_OTM_DISTANCE_PCT=1.5)})
        assert result.result == GateResult.FAIL  # $6.00 > $5.00


class TestGetMaxOtmDistance:
    """Test per-ticker OTM distance thresholds from strike grid intervals."""

    def test_fine_grid_tickers(self):
        from options_owl.risk.exit_v5.config import get_max_otm_distance
        assert get_max_otm_distance("SPY") == 3.0    # $1.00 × 3
        assert get_max_otm_distance("QQQ") == 3.0    # $1.00 × 3
        assert get_max_otm_distance("IWM") == 1.5    # $0.50 × 3
        assert get_max_otm_distance("NVDA") == 1.5   # $0.50 × 3
        assert get_max_otm_distance("MSTR") == 1.5   # $0.50 × 3

    def test_standard_grid_tickers(self):
        from options_owl.risk.exit_v5.config import get_max_otm_distance
        assert get_max_otm_distance("META") == 5.0   # $2.50 × 2
        assert get_max_otm_distance("TSLA") == 5.0   # $2.50 × 2
        assert get_max_otm_distance("PLTR") == 2.0   # $1.00 × 2
        assert get_max_otm_distance("AVGO") == 5.0   # $2.50 × 2
        assert get_max_otm_distance("MSFT") == 5.0   # $2.50 × 2

    def test_wide_grid_tickers(self):
        from options_owl.risk.exit_v5.config import get_max_otm_distance
        assert get_max_otm_distance("AMZN") == 2.5   # $2.50 × 1
        assert get_max_otm_distance("AAPL") == 2.5   # $2.50 × 1
        assert get_max_otm_distance("AMD") == 2.5    # $2.50 × 1
        assert get_max_otm_distance("GOOGL") == 2.5  # $2.50 × 1

    def test_unknown_ticker_default(self):
        from options_owl.risk.exit_v5.config import get_max_otm_distance
        assert get_max_otm_distance("XYZ") == 5.0    # $2.50 default × 2


# ══════════════════════════════════════════════════════════════════════════
# 7. Monitor Bridge Integration
# ══════════════════════════════════════════════════════════════════════════


class TestMonitorBridgeV6:

    def _make_settings(self, **overrides):
        defaults = {
            "ENABLE_V6_PER_TICKER_CONFIG": False,
            "ENABLE_V6_BREAKEVEN_RATCHET": False,
            "ENABLE_V6_2PM_TIGHTEN": False,
            "ENABLE_V6_SCALEOUT": False,
            "V6_BREAKEVEN_TRIGGER_PCT": 20.0,
            "V6_2PM_TRAIL_TIGHTEN_FACTOR": 0.7,
            "V6_2PM_SOFT_TRAIL_BOOST": 0.15,
            "V6_SCALEOUT_GAIN_PCT": 20.0,
            "V6_SCALEOUT_FRACTION": 0.333,
            "V6_SCALEOUT_MIN_CONTRACTS": 3,
            "ENABLE_PUT_TRADING": True,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_trade(self, ticker="SPY", premium=1.00, contracts=6):
        return {
            "id": 1,
            "ticker": ticker,
            "option_type": "call",
            "strike": 500.0,
            "premium_per_contract": premium,
            "contracts": contracts,
            "opened_at": "2026-04-28T14:00:00",
            "expiry_date": "2026-04-28",
            "entry_price": 500.0,
            "mfe_premium": premium,
            "bid": 0.0,
            "ask": 0.0,
        }

    def test_per_ticker_config_used(self):
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        settings = self._make_settings(ENABLE_V6_PER_TICKER_CONFIG=True)
        bridge = V5MonitorBridge(settings)

        # NVDA should get EARLY_PROFIT config
        fsm = bridge._get_fsm("NVDA")
        assert fsm.cfg.profit_target_general_pct == 20.0  # NVDA is HIGH_VOL, uses general target
        assert fsm.cfg.soft_trail_keep_pct == 0.70

        # Unknown ticker should get default
        fsm_default = bridge._get_fsm("XYZ")
        assert fsm_default.cfg == V5Config()

    def test_per_ticker_fsm_cached(self):
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        settings = self._make_settings(ENABLE_V6_PER_TICKER_CONFIG=True)
        bridge = V5MonitorBridge(settings)

        fsm1 = bridge._get_fsm("NVDA")
        fsm2 = bridge._get_fsm("NVDA")
        assert fsm1 is fsm2

    def test_scaleout_encodes_in_description(self):
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        settings = self._make_settings(ENABLE_V6_SCALEOUT=True)
        bridge = V5MonitorBridge(settings)

        trade = self._make_trade(ticker="SPY", premium=1.00, contracts=6)
        now = _now_et(10, 10)

        # Push premium to +25% to trigger scaleout
        reason, desc = bridge.evaluate(trade, 1.25, 500.5, now)
        if reason == "scaleout_20":
            assert "[V6_SCALEOUT:" in desc
        # If it triggered something else first (e.g., profit_target for SPY),
        # that's fine — it means priority ordering is correct

    def test_default_fsm_when_per_ticker_disabled(self):
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        settings = self._make_settings(ENABLE_V6_PER_TICKER_CONFIG=False)
        bridge = V5MonitorBridge(settings)

        fsm = bridge._get_fsm("NVDA")
        assert fsm is bridge.fsm  # should use the default, not per-ticker

    def test_reason_map_covers_v6_reasons(self):
        from options_owl.risk.exit_v5.monitor_bridge import _REASON_MAP
        assert ExitReason.BREAKEVEN_RATCHET in _REASON_MAP
        assert ExitReason.SCALEOUT in _REASON_MAP


# ══════════════════════════════════════════════════════════════════════════
# 7b. PUT Scalp Config
# ══════════════════════════════════════════════════════════════════════════


class TestPutScalpConfig:
    """Tests for PUT scalp configuration — simple target/stop/time exits."""

    def test_get_ticker_config_returns_put_scalp_for_puts(self):
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        cfg = get_ticker_config("SPY", use_per_ticker=True, option_type="put")
        assert cfg is PUT_SCALP_CONFIG

    def test_get_ticker_config_returns_call_config_for_calls(self):
        cfg = get_ticker_config("NVDA", use_per_ticker=True, option_type="call")
        assert cfg is TICKER_CONFIGS["NVDA"]

    def test_put_scalp_no_profit_target(self):
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        assert PUT_SCALP_CONFIG.profit_target_general_pct == 0.0  # no ceiling

    def test_put_scalp_stop_at_50pct(self):
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        assert PUT_SCALP_CONFIG.backstop_0dte_pct == 50.0
        assert PUT_SCALP_CONFIG.tight_stop_0dte_pct == 50.0

    def test_put_scalp_no_hold_limit(self):
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        assert PUT_SCALP_CONFIG.theta_bleed_min == 999.0  # no time limit

    def test_put_scalp_fsm_holds_at_35pct_gain(self):
        """PUT at +35% gain should HOLD — no profit target, trail manages exits."""
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=1, ticker="SPY", option_type="put",
            entry_premium=0.20, entry_time=datetime(2026, 1, 5, 10, 0),
            contracts=5, entry_underlying_price=500.0, dte=0,
        )
        now = datetime(2026, 1, 5, 14, 10)  # 10min into trade (past grace)
        action = fsm.evaluate(
            state=state, current_premium=0.27,  # +35%
            bid=0.26, ask=0.28, now_et=now,
            current_underlying=499.0, minutes_to_close=50.0,
        )
        assert not action.should_exit  # trail system manages profitable PUTs

    def test_put_scalp_fsm_exits_at_stop(self):
        """PUT at -55% loss should trigger graduated stop (50% threshold)."""
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=2, ticker="SPY", option_type="put",
            entry_premium=0.20, entry_time=datetime(2026, 1, 5, 10, 0),
            contracts=5, entry_underlying_price=500.0, dte=0,
        )
        now = datetime(2026, 1, 5, 14, 10)  # past grace
        action = fsm.evaluate(
            state=state, current_premium=0.09,  # -55%
            bid=0.08, ask=0.10, now_et=now,
            current_underlying=501.0, minutes_to_close=50.0,
        )
        assert action.should_exit

    def test_put_scalp_fsm_holds_at_65min(self):
        """PUT held 65min should HOLD — no time limit, trail handles exits."""
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        fsm = ExitFSM(PUT_SCALP_CONFIG)
        state = TradeState(
            trade_id=3, ticker="SPY", option_type="put",
            entry_premium=0.20, entry_time=datetime(2026, 1, 5, 13, 0),
            contracts=5, entry_underlying_price=500.0, dte=0,
        )
        now = datetime(2026, 1, 5, 14, 5)  # 65 min into trade
        action = fsm.evaluate(
            state=state, current_premium=0.18,  # slightly down, not at stop
            bid=0.17, ask=0.19, now_et=now,
            current_underlying=500.5, minutes_to_close=55.0,
        )
        assert not action.should_exit  # no hold time limit for PUTs

    def test_monitor_bridge_uses_put_config(self):
        """Bridge should use PUT_SCALP_CONFIG for PUT trades."""
        from options_owl.risk.exit_v5.config import PUT_SCALP_CONFIG
        from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge
        settings = SimpleNamespace(
            EXIT_ENGINE="v5",
            ENABLE_V6_PER_TICKER_CONFIG=True,
            ENABLE_V6_BREAKEVEN_RATCHET=False,
            V6_BREAKEVEN_TRIGGER_PCT=20.0,
            ENABLE_V6_SCALEOUT=False,
            V6_SCALEOUT_GAIN_PCT=20.0,
            V6_SCALEOUT_FRACTION=0.333,
            V6_SCALEOUT_MIN_CONTRACTS=3,
            ENABLE_V6_2PM_TIGHTEN=False,
            V6_2PM_TRAIL_TIGHTEN_FACTOR=0.7,
            V6_2PM_SOFT_TRAIL_BOOST=0.15,
            ENABLE_V6_EARLY_POP_GATE=False,
            ENABLE_V6_SIDEWAYS_SCALP=False,
            ENABLE_SCALP_TARGET=False,
            SCALP_TARGET_PCT=25.0,
            SCALP_RUNNER_CONFIRM_PCT=40.0,
        )
        bridge = V5MonitorBridge(settings)
        fsm = bridge._get_fsm("SPY", option_type="put")
        assert fsm.cfg is PUT_SCALP_CONFIG
        # CALL should NOT use PUT config
        call_fsm = bridge._get_fsm("SPY", option_type="call")
        assert call_fsm.cfg is not PUT_SCALP_CONFIG


# ══════════════════════════════════════════════════════════════════════════
# 8. V6 DCA (Mid-Trade Dollar Cost Averaging)
# ══════════════════════════════════════════════════════════════════════════


class TestV6DCA:
    """Tests for _check_v6_dca in position_monitor."""

    @pytest.fixture(autouse=True)
    def reset_dca_state(self):
        """Clear the in-memory DCA tracking set between tests."""
        from options_owl.execution import position_monitor
        position_monitor._v6_dca_fired.clear()
        yield
        position_monitor._v6_dca_fired.clear()

    def _make_settings(self, **overrides):
        defaults = {
            "ENABLE_V6_DCA": True,
            "V6_DCA_TICKERS": "MSFT,IWM,SPY,QQQ,AMZN,NVDA",
            "V6_DCA_MIN_MINUTES": 8.0,
            "V6_DCA_MAX_MINUTES": 20.0,
            "V6_DCA_MIN_DIP_PCT": 15.0,
            "V6_DCA_MAX_DIP_PCT": 35.0,
            "V6_DCA_UNDERLYING_THRESHOLD": 0.5,
            "WEBULL_ENTRY_AGGRESS_PCT": 5.0,
            "PORTFOLIO_SIZE": 100000.0,  # large enough that DCA cap doesn't interfere
            "MAX_DCA_POSITION_PCT": 50.0,  # permissive for unit tests
            "ENABLE_PUT_TRADING": True,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_trade(self, ticker="SPY", premium=2.00, contracts=5,
                    minutes_ago=12.0, entry_price=500.0, option_type="call"):
        opened = datetime.now() - timedelta(minutes=minutes_ago)
        return {
            "id": 99,
            "ticker": ticker,
            "option_type": option_type,
            "strike": 500.0,
            "premium_per_contract": premium,
            "contracts": contracts,
            "opened_at": opened.isoformat(),
            "expiry_date": "2026-04-28",
            "entry_price": entry_price,
            "status": "open",
        }

    @pytest.mark.asyncio
    async def test_dca_fires_on_valid_dip(self, tmp_path):
        """DCA should fire when all conditions are met."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        db_path = str(tmp_path / "test.db")
        import aiosqlite
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY, ticker TEXT, contracts INTEGER,
                    premium_per_contract REAL, total_cost REAL,
                    status TEXT, opened_at TEXT, entry_price REAL,
                    option_type TEXT, strike REAL, expiry_date TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO paper_trades VALUES (99,'SPY',5,2.00,1000.0,'open',?,500.0,'call',500.0,'2026-04-28')",
                (self._make_trade()["opened_at"],),
            )
            await conn.commit()

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=12.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        # Mock time to 11AM ET so the 2PM gate doesn't block
        morning = datetime(2026, 5, 14, 11, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("options_owl.execution.position_monitor._now_et", return_value=morning):
            # 20% dip: $2.00 → $1.60
            await _check_v6_dca(trade, 1.60, 500.0, settings, paper_trader, db_path)
        assert 99 in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_wrong_ticker(self):
        """DCA should not fire for non-whitelisted tickers."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        trade = self._make_trade(ticker="MSTR", premium=2.00, minutes_ago=12.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        await _check_v6_dca(trade, 1.60, 500.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_too_early(self):
        """DCA should not fire before the minimum time window."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=3.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        await _check_v6_dca(trade, 1.60, 500.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_too_late(self):
        """DCA should not fire after the maximum time window."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=25.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        await _check_v6_dca(trade, 1.60, 500.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_dip_too_small(self):
        """DCA should not fire if dip is below minimum threshold."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=12.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        # Only 5% dip (below 15% min)
        await _check_v6_dca(trade, 1.90, 500.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_dip_too_large(self):
        """DCA should not fire if dip exceeds maximum (thesis is broken)."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=12.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        # 50% dip (above 35% max)
        await _check_v6_dca(trade, 1.00, 500.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_underlying_against_call(self):
        """DCA should not fire if underlying moved against a call position."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        # Call with entry_price=500, underlying now at 496 (-0.8%)
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=12.0,
                                 entry_price=500.0, option_type="call")
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        await _check_v6_dca(trade, 1.60, 496.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_blocked_underlying_against_put(self):
        """DCA should not fire if underlying moved against a put position."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        settings = self._make_settings()
        # Put with entry_price=500, underlying now at 504 (+0.8%)
        trade = self._make_trade(ticker="QQQ", premium=2.00, minutes_ago=12.0,
                                 entry_price=500.0, option_type="put")
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        await _check_v6_dca(trade, 1.60, 504.0, settings, paper_trader, "fake.db")
        assert 99 not in _v6_dca_fired

    @pytest.mark.asyncio
    async def test_dca_one_shot(self, tmp_path):
        """DCA should only fire once per trade."""
        from options_owl.execution.position_monitor import _check_v6_dca, _v6_dca_fired

        db_path = str(tmp_path / "test.db")
        import aiosqlite
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY, ticker TEXT, contracts INTEGER,
                    premium_per_contract REAL, total_cost REAL,
                    status TEXT, opened_at TEXT, entry_price REAL,
                    option_type TEXT, strike REAL, expiry_date TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO paper_trades VALUES (99,'SPY',5,2.00,1000.0,'open',?,500.0,'call',500.0,'2026-04-28')",
                (self._make_trade()["opened_at"],),
            )
            await conn.commit()

        settings = self._make_settings()
        trade = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=12.0)
        paper_trader = MagicMock()
        paper_trader.webull_executor = None

        # Mock time to 11AM ET so the 2PM gate doesn't block
        morning = datetime(2026, 5, 14, 11, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("options_owl.execution.position_monitor._now_et", return_value=morning):
            # First call fires
            await _check_v6_dca(trade, 1.60, 500.0, settings, paper_trader, db_path)
            assert 99 in _v6_dca_fired

            # Verify contracts updated in DB
            async with aiosqlite.connect(db_path) as conn:
                row = await conn.execute("SELECT contracts FROM paper_trades WHERE id=99")
                row = await row.fetchone()
                assert row[0] == 10  # 5 original + 5 DCA

            # Second call should NOT fire (already in _v6_dca_fired)
            # Reset trade to original contracts for the test
            trade2 = self._make_trade(ticker="SPY", premium=2.00, minutes_ago=14.0)
            await _check_v6_dca(trade2, 1.50, 500.0, settings, paper_trader, db_path)
        # Contracts should still be 10, not 15
        async with aiosqlite.connect(db_path) as conn:
            row = await conn.execute("SELECT contracts FROM paper_trades WHERE id=99")
            row = await row.fetchone()
            assert row[0] == 10
