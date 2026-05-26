"""Tests for Simpsons-inspired veto gates."""

from unittest.mock import patch

import pytest

from options_owl.sourcing.data.indicator_engine import IndicatorSet
from options_owl.sourcing.filters.veto_gates import (
    _veto_afternoon_danger_zone,
    _veto_low_atr_chop,
    _veto_ml_warmup,
    _veto_no_institutional_sweep,
    _veto_no_momentum_confirm,
    _veto_sweep_no_volume,
    _veto_wide_spread,
    run_veto_gates,
)
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState, TierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx_with_indicators(**overrides) -> SignalContext:
    """Build a SignalContext with indicators. Sweep + volume default to passing."""
    ind_defaults = {
        "sweep_pdl": True,  # has a sweep by default (CALL direction)
        "volume_ratio": 2.0,  # above 1.5x threshold
        "atr14": 2.5,
        "adx": 25.0,
        "last_close": 100.0,
    }
    ind_defaults.update(overrides)
    ind = IndicatorSet(**ind_defaults)

    ctx = SignalContext(
        ticker="NVDA",
        scan_time="2026-05-21T10:30:00-04:00",
        state=SignalState.SCORED,
        direction=Direction.CALL,
        score_total=72,
        indicators=ind,
        candles_5m=[
            {"close": 99.5, "high": 100.0, "low": 99.0, "volume": 1000},
            {"close": 99.8, "high": 100.2, "low": 99.3, "volume": 1100},
            {"close": 100.1, "high": 100.5, "low": 99.5, "volume": 1200},
        ],
    )
    return ctx


# ---------------------------------------------------------------------------
# Afternoon Danger Zone
# ---------------------------------------------------------------------------

class TestAfternoonVeto:
    def test_blocks_at_2pm(self):
        """1:30-3:00 PM ET should be blocked."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_time = datetime(2026, 5, 21, 14, 0, 0, tzinfo=ET)  # 2:00 PM

        ctx = _ctx_with_indicators()
        with patch("options_owl.sourcing.filters.veto_gates.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            blocked, reason = _veto_afternoon_danger_zone(ctx)
        assert blocked is True
        assert "afternoon_danger_zone" in reason

    def test_passes_at_10am(self):
        """10:00 AM ET should pass."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_time = datetime(2026, 5, 21, 10, 0, 0, tzinfo=ET)

        ctx = _ctx_with_indicators()
        with patch("options_owl.sourcing.filters.veto_gates.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            blocked, reason = _veto_afternoon_danger_zone(ctx)
        assert blocked is False

    def test_passes_at_3_15pm(self):
        """3:15 PM ET (power hour) should pass."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_time = datetime(2026, 5, 21, 15, 15, 0, tzinfo=ET)

        ctx = _ctx_with_indicators()
        with patch("options_owl.sourcing.filters.veto_gates.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            blocked, reason = _veto_afternoon_danger_zone(ctx)
        assert blocked is False


# ---------------------------------------------------------------------------
# ML Warmup
# ---------------------------------------------------------------------------

class TestMLWarmup:
    def test_blocks_with_few_observations(self):
        """ML with < 10 premium observations should be blocked."""
        ctx = _ctx_with_indicators()
        ctx.ml_confidence = 0.85
        ctx.ml_model_source = "signal_NVDA"

        # Mock option history with only 3 snapshots
        class MockHistory:
            snapshots = [1, 2, 3]
        ctx._option_history = MockHistory()  # type: ignore

        blocked, reason = _veto_ml_warmup(ctx)
        assert blocked is True
        assert "3 premium observations" in reason

    def test_passes_with_enough_observations(self):
        """ML with >= 10 observations should pass."""
        ctx = _ctx_with_indicators()
        ctx.ml_confidence = 0.85
        ctx.ml_model_source = "signal_NVDA"

        class MockHistory:
            snapshots = list(range(15))
        ctx._option_history = MockHistory()  # type: ignore

        blocked, reason = _veto_ml_warmup(ctx)
        assert blocked is False

    def test_passes_when_no_ml(self):
        """Non-ML signals (tech-only) should not be blocked."""
        ctx = _ctx_with_indicators()
        ctx.ml_confidence = None
        ctx.ml_model_source = ""

        blocked, reason = _veto_ml_warmup(ctx)
        assert blocked is False

    def test_blocks_when_no_history(self):
        """ML with no option history at all should be blocked."""
        ctx = _ctx_with_indicators()
        ctx.ml_confidence = 0.85
        ctx.ml_model_source = "signal_NVDA"
        # No _option_history attribute

        blocked, reason = _veto_ml_warmup(ctx)
        assert blocked is True
        assert "0 premium observations" in reason


# ---------------------------------------------------------------------------
# Institutional Sweep
# ---------------------------------------------------------------------------

class TestInstitutionalSweep:
    def test_call_passes_with_pdl_sweep(self):
        """CALL with PDL sweep (bear trap → bullish) should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=True)
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is False

    def test_call_passes_with_pwl_sweep(self):
        """CALL with PWL sweep should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=False, sweep_pwl=True)
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is False

    def test_call_passes_with_session_low_sweep(self):
        """CALL with session low sweep should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=False, sweep_session_low=True)
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is False

    def test_call_passes_with_strong_volume_no_sweep(self):
        """CALL without sweep but with 2x+ volume should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=False, sweep_pwl=False, sweep_session_low=False, volume_ratio=2.5)
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is False

    def test_call_blocks_without_sweep_or_volume(self):
        """CALL without sweep AND without strong volume should be blocked."""
        ctx = _ctx_with_indicators(sweep_pdl=False, sweep_pwl=False, sweep_session_low=False, volume_ratio=1.2)
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is True
        assert "no_institutional_sweep_or_volume" in reason

    def test_put_passes_with_pdh_sweep(self):
        """PUT with PDH sweep (bull trap → bearish) should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=False, sweep_pdh=True)
        ctx.direction = Direction.PUT
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is False

    def test_put_blocks_without_sweep_or_volume(self):
        """PUT without any bullish sweep and low volume should be blocked."""
        ctx = _ctx_with_indicators(
            sweep_pdl=True, sweep_pdh=False, sweep_pwh=False, sweep_session_high=False,
            volume_ratio=1.0,
        )
        ctx.direction = Direction.PUT
        blocked, reason = _veto_no_institutional_sweep(ctx)
        assert blocked is True


# ---------------------------------------------------------------------------
# Low ATR Chop
# ---------------------------------------------------------------------------

class TestLowATRChop:
    def test_blocks_very_low_atr_low_adx(self):
        """Very low ATR + very low ADX = dead market, should block."""
        ctx = _ctx_with_indicators(atr14=0.1, adx=10.0)
        blocked, reason = _veto_low_atr_chop(ctx)
        assert blocked is True
        assert "low_atr_chop" in reason

    def test_passes_normal_atr(self):
        """Normal ATR should pass even with low ADX."""
        ctx = _ctx_with_indicators(atr14=2.5, adx=10.0)
        blocked, reason = _veto_low_atr_chop(ctx)
        assert blocked is False

    def test_passes_low_atr_moderate_adx(self):
        """Low ATR but moderate ADX (some trend) should pass."""
        ctx = _ctx_with_indicators(atr14=0.1, adx=15.0)
        blocked, reason = _veto_low_atr_chop(ctx)
        assert blocked is False


# ---------------------------------------------------------------------------
# Wide Spread
# ---------------------------------------------------------------------------

class TestWideSpread:
    def test_blocks_wide_spread_from_context(self):
        ctx = _ctx_with_indicators()
        ctx.spread_pct = 45.0
        blocked, reason = _veto_wide_spread(ctx)
        assert blocked is True
        assert "spread_too_wide" in reason

    def test_passes_narrow_spread(self):
        ctx = _ctx_with_indicators()
        ctx.spread_pct = 8.0
        blocked, reason = _veto_wide_spread(ctx)
        assert blocked is False

    def test_passes_no_spread_data(self):
        ctx = _ctx_with_indicators()
        ctx.spread_pct = None
        blocked, reason = _veto_wide_spread(ctx)
        assert blocked is False


# ---------------------------------------------------------------------------
# Sweep Volume Confirmation
# ---------------------------------------------------------------------------

class TestSweepVolume:
    def test_blocks_sweep_without_volume(self):
        """Sweep detected but volume < 1.5x should block."""
        ctx = _ctx_with_indicators(sweep_pdl=True, volume_ratio=1.0)
        blocked, reason = _veto_sweep_no_volume(ctx)
        assert blocked is True
        assert "sweep_no_volume_confirm" in reason

    def test_passes_sweep_with_volume(self):
        """Sweep with >= 1.5x volume should pass."""
        ctx = _ctx_with_indicators(sweep_pdl=True, volume_ratio=2.0)
        blocked, reason = _veto_sweep_no_volume(ctx)
        assert blocked is False

    def test_skips_when_no_sweep(self):
        """No sweep detected — this gate doesn't apply."""
        ctx = _ctx_with_indicators(
            sweep_pdl=False, sweep_pdh=False, sweep_pwl=False,
            sweep_pwh=False, sweep_session_high=False, sweep_session_low=False,
            volume_ratio=0.5,
        )
        blocked, reason = _veto_sweep_no_volume(ctx)
        assert blocked is False


# ---------------------------------------------------------------------------
# Momentum Confirmation
# ---------------------------------------------------------------------------

class TestMomentumConfirm:
    def test_call_blocks_negative_momentum(self):
        """CALL with falling price should be blocked."""
        ctx = _ctx_with_indicators()
        ctx.direction = Direction.CALL
        ctx.candles_5m = [
            {"close": 100.5}, {"close": 100.0}, {"close": 99.0},
        ]
        blocked, reason = _veto_no_momentum_confirm(ctx)
        assert blocked is True
        assert "CALL but 5m trend" in reason

    def test_call_passes_positive_momentum(self):
        """CALL with rising price should pass."""
        ctx = _ctx_with_indicators()
        ctx.direction = Direction.CALL
        ctx.candles_5m = [
            {"close": 99.0}, {"close": 99.5}, {"close": 100.5},
        ]
        blocked, reason = _veto_no_momentum_confirm(ctx)
        assert blocked is False

    def test_put_blocks_positive_momentum(self):
        """PUT with rising price should be blocked."""
        ctx = _ctx_with_indicators()
        ctx.direction = Direction.PUT
        ctx.candles_5m = [
            {"close": 99.0}, {"close": 99.5}, {"close": 100.5},
        ]
        blocked, reason = _veto_no_momentum_confirm(ctx)
        assert blocked is True
        assert "PUT but 5m trend" in reason

    def test_put_passes_negative_momentum(self):
        """PUT with falling price should pass."""
        ctx = _ctx_with_indicators()
        ctx.direction = Direction.PUT
        ctx.candles_5m = [
            {"close": 100.5}, {"close": 100.0}, {"close": 99.0},
        ]
        blocked, reason = _veto_no_momentum_confirm(ctx)
        assert blocked is False


# ---------------------------------------------------------------------------
# Integration: run_veto_gates
# ---------------------------------------------------------------------------

class TestRunVetoGates:
    def test_clean_signal_passes_all_gates(self):
        """Signal with all good data should pass all veto gates."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_time = datetime(2026, 5, 21, 10, 0, 0, tzinfo=ET)  # 10 AM

        ctx = _ctx_with_indicators(
            sweep_pdl=True, volume_ratio=2.0, atr14=2.5, adx=25.0,
        )
        ctx.spread_pct = 8.0

        with patch("options_owl.sourcing.filters.veto_gates.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            blocked, reason = run_veto_gates(ctx)
        assert blocked is False
        assert reason == ""

    def test_first_veto_wins(self):
        """When multiple vetos would fire, the first one in order wins."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        # 2 PM ET — afternoon veto should fire first
        mock_time = datetime(2026, 5, 21, 14, 0, 0, tzinfo=ET)

        ctx = _ctx_with_indicators(
            sweep_pdl=False,  # would also fail sweep gate
            volume_ratio=0.5,  # would also fail volume gate
        )
        ctx.spread_pct = 50.0  # would also fail spread gate

        with patch("options_owl.sourcing.filters.veto_gates.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            blocked, reason = run_veto_gates(ctx)
        assert blocked is True
        assert "afternoon_danger_zone" in reason  # first gate
