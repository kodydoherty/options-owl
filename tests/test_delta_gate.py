"""Tests for the DeltaEntryGate — delta-based entry filtering.

Tests cover:
- Gate disabled → SKIP
- No delta data → SKIP
- Zero delta → SKIP
- Delta out of valid range (> 1.0) → SKIP
- Delta below min (far OTM) → FAIL
- Delta above max (deep ITM) → FAIL
- Delta in sweet spot → PASS
- Custom thresholds work
- PUT delta (negative) uses abs value
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from options_owl.risk.pipeline import DeltaEntryGate, GateResult


@pytest.fixture
def gate():
    return DeltaEntryGate()


def _make_ctx(
    delta=None,
    enable=True,
    min_delta=0.15,
    max_delta=0.70,
):
    """Build a minimal pipeline context for testing the delta gate."""
    settings = MagicMock()
    settings.ENABLE_DELTA_GATE = enable
    settings.DELTA_ENTRY_MIN = min_delta
    settings.DELTA_ENTRY_MAX = max_delta
    return {
        "settings": settings,
        "entry_delta": delta,
    }


class TestDeltaEntryGate:
    """Unit tests for DeltaEntryGate."""

    @pytest.mark.asyncio
    async def test_gate_disabled_returns_skip(self, gate):
        ctx = _make_ctx(delta=0.40, enable=False)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP
        assert "disabled" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_delta_returns_skip(self, gate):
        ctx = _make_ctx(delta=None)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP
        assert "No delta" in outcome.reason

    @pytest.mark.asyncio
    async def test_zero_delta_returns_skip(self, gate):
        ctx = _make_ctx(delta=0)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP
        assert "No delta" in outcome.reason

    @pytest.mark.asyncio
    async def test_delta_out_of_range_returns_skip(self, gate):
        """Delta > 1.0 is invalid data — skip, don't block."""
        ctx = _make_ctx(delta=1.5)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP
        assert "valid range" in outcome.reason

    @pytest.mark.asyncio
    async def test_far_otm_fails(self, gate):
        """Delta 0.10 < 0.15 min → far OTM lottery ticket."""
        ctx = _make_ctx(delta=0.10)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "far OTM" in outcome.reason

    @pytest.mark.asyncio
    async def test_deep_itm_fails(self, gate):
        """Delta 0.80 > 0.70 max → deep ITM, limited leverage."""
        ctx = _make_ctx(delta=0.80)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL
        assert "deep ITM" in outcome.reason

    @pytest.mark.asyncio
    async def test_sweet_spot_passes(self, gate):
        """Delta 0.40 in [0.15, 0.70] → PASS."""
        ctx = _make_ctx(delta=0.40)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS
        assert "0.400" in outcome.reason
        assert "in range" in outcome.reason

    @pytest.mark.asyncio
    async def test_boundary_min_passes(self, gate):
        """Delta exactly at min threshold → PASS."""
        ctx = _make_ctx(delta=0.15)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_boundary_max_passes(self, gate):
        """Delta exactly at max threshold → PASS."""
        ctx = _make_ctx(delta=0.70)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_put_delta_uses_abs(self, gate):
        """PUT delta is negative — gate uses abs value."""
        ctx = _make_ctx(delta=-0.40)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS
        assert "0.400" in outcome.reason

    @pytest.mark.asyncio
    async def test_put_far_otm_fails(self, gate):
        """PUT with delta -0.05 → abs 0.05 < 0.15 → FAIL."""
        ctx = _make_ctx(delta=-0.05)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_put_deep_itm_fails(self, gate):
        """PUT with delta -0.85 → abs 0.85 > 0.70 → FAIL."""
        ctx = _make_ctx(delta=-0.85)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, gate):
        """Custom min/max thresholds are respected."""
        ctx = _make_ctx(delta=0.25, min_delta=0.20, max_delta=0.60)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.PASS

        ctx = _make_ctx(delta=0.65, min_delta=0.20, max_delta=0.60)
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_no_settings_returns_skip(self, gate):
        """Missing settings in ctx → gate disabled path."""
        ctx = {"entry_delta": 0.40}
        outcome = await gate.evaluate(ctx)
        assert outcome.result == GateResult.SKIP

    @pytest.mark.asyncio
    async def test_gate_name(self, gate):
        assert gate.name == "delta_entry"


class TestDeltaGateInPipeline:
    """Verify DeltaEntryGate is wired into the entry pipeline correctly."""

    def test_delta_gate_in_default_gates(self):
        """DeltaEntryGate is in the DEFAULT_ENTRY_GATES list."""
        from options_owl.risk.pipeline import DEFAULT_ENTRY_GATES, DeltaEntryGate
        assert DeltaEntryGate in DEFAULT_ENTRY_GATES

    def test_delta_gate_after_spread_gate(self):
        """DeltaEntryGate comes after SpreadCostGate in the pipeline."""
        from options_owl.risk.pipeline import (
            DEFAULT_ENTRY_GATES,
            DeltaEntryGate,
            SpreadCostGate,
        )
        spread_idx = DEFAULT_ENTRY_GATES.index(SpreadCostGate)
        delta_idx = DEFAULT_ENTRY_GATES.index(DeltaEntryGate)
        assert delta_idx > spread_idx

    def test_settings_has_delta_fields(self):
        """Settings class has ENABLE_DELTA_GATE, DELTA_ENTRY_MIN, DELTA_ENTRY_MAX."""
        from options_owl.config.settings import Settings
        s = Settings(DISCORD_TOKEN="test")
        assert hasattr(s, "ENABLE_DELTA_GATE")
        assert hasattr(s, "DELTA_ENTRY_MIN")
        assert hasattr(s, "DELTA_ENTRY_MAX")
        assert s.ENABLE_DELTA_GATE is False  # disabled by default
        assert s.DELTA_ENTRY_MIN == 0.15
        assert s.DELTA_ENTRY_MAX == 0.70
