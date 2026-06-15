"""Tests for concurrent position handling and ticker blocking.

Validates that:
1. Blocked tickers (COIN, AVGO, MU, MSFT) are rejected by the pipeline
2. ConcurrentPositionsGate respects MAX_CONCURRENT=8
3. DuplicateTickerGate allows different tickers but blocks same ticker+direction
4. Multiple signals can enter simultaneously (no artificial spacing)
5. The backtest concurrent architecture matches production behavior
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from options_owl.config.settings import Settings
from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)
from options_owl.risk.exit_v5.config import (
    HIGH_VOL_TICKERS,
    categorize_ticker,
    TickerCategory,
)
from options_owl.risk.pipeline import (
    BlockedTickerGate,
    ConcurrentPositionsGate,
    DuplicateTickerGate,
    GateResult,
)


def _make_signal(
    ticker: str = "SPY",
    direction: str = "call",
    score: int = 95,
    strike: float = 500.0,
    premium: float = 2.50,
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        direction=Direction(direction),
        score=score,
        strength=SignalStrength.STRONG,
        entry_price=strike,
        target_price=strike * 1.01,
        expected_move_pct=1.0,
        risk_reward=3.0,
        strike=strike,
        atm_premium=premium,
        expiry="0DTE",
        bot_source=BotSource.CAPTAIN_HOOK,
        sentiment=Sentiment.BULLISH,
    )


def _make_settings(**overrides) -> SimpleNamespace:
    defaults = {
        "BLOCKED_TICKERS": "MSFT,COIN,AVGO,MU",
        "PUT_EXCLUDED_TICKERS": "PLTR,AMD,MSTR,AVGO",
        "MAX_CONCURRENT": 8,
        "PORTFOLIO_SIZE": 23000,
        "ENABLE_CORRELATION_CAP": True,
        "CORRELATION_CAP_MAX_PER_GROUP": 3,
        "ENABLE_PUT_MARKET_DIRECTION_GATE": False,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── BlockedTickerGate Tests ──────────────────────────────────────────────


class TestBlockedTickerGate:
    """Verify blocked tickers are rejected at the gate level."""

    @pytest.mark.asyncio
    async def test_coin_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="COIN"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL
        assert "blocklist" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_avgo_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="AVGO"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_mu_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="MU"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_msft_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="MSFT"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_spy_not_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="SPY"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_nvda_not_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="NVDA"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_tsla_not_blocked(self):
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="TSLA"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Blocked ticker check should be case-insensitive."""
        gate = BlockedTickerGate()
        ctx = {"signal": _make_signal(ticker="coin"), "settings": _make_settings()}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL

    @pytest.mark.asyncio
    async def test_empty_blocklist_allows_all(self):
        gate = BlockedTickerGate()
        ctx = {
            "signal": _make_signal(ticker="COIN"),
            "settings": _make_settings(BLOCKED_TICKERS=""),
        }
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS


# ── ConcurrentPositionsGate Tests ────────────────────────────────────────


class TestConcurrentPositionsGate:
    """Verify MAX_CONCURRENT=8 is properly enforced."""

    @pytest.mark.asyncio
    async def test_allows_up_to_max_concurrent(self):
        gate = ConcurrentPositionsGate()
        settings = _make_settings(MAX_CONCURRENT=8)
        # 7 open, should allow 8th
        ctx = {"signal": _make_signal(), "settings": settings, "open_count": 7}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_blocks_at_max_concurrent(self):
        gate = ConcurrentPositionsGate()
        settings = _make_settings(MAX_CONCURRENT=8)
        # Already at 8, should block 9th
        ctx = {"signal": _make_signal(), "settings": settings, "open_count": 8}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL
        assert "8 open >= max 8" in result.reason

    @pytest.mark.asyncio
    async def test_zero_open_always_passes(self):
        gate = ConcurrentPositionsGate()
        settings = _make_settings(MAX_CONCURRENT=8)
        ctx = {"signal": _make_signal(), "settings": settings, "open_count": 0}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_no_same_direction_cap(self):
        """Production has no MAX_SAME_DIRECTION — all 8 can be CALLs."""
        gate = ConcurrentPositionsGate()
        settings = _make_settings(MAX_CONCURRENT=8)
        # 7 open (all calls), adding 8th call should pass
        ctx = {"signal": _make_signal(), "settings": settings, "open_count": 7}
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS


# ── DuplicateTickerGate Tests ────────────────────────────────────────────


class TestDuplicateTickerGate:
    """Verify same ticker+direction blocked but different tickers allowed."""

    @pytest.mark.asyncio
    async def test_different_tickers_pass(self):
        gate = DuplicateTickerGate()
        ctx = {
            "signal": _make_signal(ticker="NVDA"),
            "open_tickers": {"SPY", "TSLA", "AMD"},
            "open_positions": [("SPY", "call"), ("TSLA", "call"), ("AMD", "call")],
        }
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_same_ticker_same_direction_blocked(self):
        gate = DuplicateTickerGate()
        ctx = {
            "signal": _make_signal(ticker="SPY", direction="call"),
            "open_tickers": {"SPY"},
            "open_positions": [("SPY", "call")],
        }
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.FAIL
        assert "Already have open SPY call" in result.reason

    @pytest.mark.asyncio
    async def test_same_ticker_opposite_direction_passes(self):
        """Signal flip: SPY call open, SPY put signal → PASS + flag for close."""
        gate = DuplicateTickerGate()
        ctx = {
            "signal": _make_signal(ticker="SPY", direction="put"),
            "open_tickers": {"SPY"},
            "open_positions": [("SPY", "call")],
        }
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS
        assert ctx.get("signal_flip_ticker") == "SPY"

    @pytest.mark.asyncio
    async def test_no_open_positions_passes(self):
        gate = DuplicateTickerGate()
        ctx = {
            "signal": _make_signal(ticker="SPY"),
            "open_tickers": set(),
            "open_positions": [],
        }
        result = await gate.evaluate(ctx)
        assert result.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_multiple_different_tickers_all_pass(self):
        """Simulates multiple signals arriving at the same time for different tickers."""
        gate = DuplicateTickerGate()
        tickers = ["SPY", "NVDA", "TSLA", "AMD", "GOOGL", "AMZN"]
        open_tickers: set[str] = set()
        open_positions: list[tuple[str, str]] = []

        for tk in tickers:
            ctx = {
                "signal": _make_signal(ticker=tk),
                "open_tickers": open_tickers.copy(),
                "open_positions": open_positions.copy(),
            }
            result = await gate.evaluate(ctx)
            assert result.result == GateResult.PASS, f"{tk} should pass but was blocked"
            # Simulate adding to open set
            open_tickers.add(tk)
            open_positions.append((tk, "call"))


# ── Ticker Category Tests ────────────────────────────────────────────────


class TestTickerCategories:
    """Verify ticker categorization after COIN/AVGO removal."""

    def test_coin_not_in_high_vol(self):
        assert "COIN" not in HIGH_VOL_TICKERS

    def test_avgo_not_in_high_vol(self):
        assert "AVGO" not in HIGH_VOL_TICKERS

    def test_tsla_still_high_vol(self):
        assert "TSLA" in HIGH_VOL_TICKERS
        assert categorize_ticker("TSLA") == TickerCategory.HIGH_VOL

    def test_mstr_still_high_vol(self):
        assert "MSTR" in HIGH_VOL_TICKERS

    def test_spy_is_index(self):
        assert categorize_ticker("SPY") == TickerCategory.INDEX

    def test_unknown_ticker_is_standard(self):
        assert categorize_ticker("XYZ") == TickerCategory.STANDARD


# ── ML Pipeline Exclusion Tests ──────────────────────────────────────────


class TestMLPipelineExclusions:
    """Verify ML pipeline excludes the right tickers."""

    def test_excluded_tickers_match_blocked(self):
        from options_owl.sourcing.ml_pipeline import EXCLUDED_TICKERS
        assert "MSFT" in EXCLUDED_TICKERS
        assert "COIN" in EXCLUDED_TICKERS
        assert "AVGO" in EXCLUDED_TICKERS
        assert "MU" in EXCLUDED_TICKERS

    def test_profitable_tickers_not_excluded(self):
        from options_owl.sourcing.ml_pipeline import EXCLUDED_TICKERS
        for tk in ["SPY", "QQQ", "NVDA", "TSLA", "PLTR", "AMD", "GOOGL"]:
            assert tk not in EXCLUDED_TICKERS, f"{tk} should NOT be excluded"

    def test_active_tickers_after_exclusion(self):
        from options_owl.sourcing.ml_pipeline import EXCLUDED_TICKERS, TICKERS
        active = [t for t in TICKERS if t not in EXCLUDED_TICKERS]
        # Should have 16 active tickers (20 total - 4 excluded)
        assert len(active) == 16
        assert "SPY" in active
        assert "COIN" not in active


# ── Settings Consistency Tests ───────────────────────────────────────────


class TestSettingsConsistency:
    """Verify settings are internally consistent."""

    def test_blocked_tickers_includes_coin_avgo_mu(self):
        s = Settings()
        blocked = {t.strip().upper() for t in s.BLOCKED_TICKERS.split(",") if t.strip()}
        assert "COIN" in blocked
        assert "AVGO" in blocked
        assert "MU" in blocked
        assert "MSFT" in blocked

    def test_dca_tickers_excludes_blocked(self):
        """DCA tickers should not include blocked tickers."""
        s = Settings()
        blocked = {t.strip().upper() for t in s.BLOCKED_TICKERS.split(",") if t.strip()}
        dca = {t.strip().upper() for t in s.V6_DCA_TICKERS.split(",") if t.strip()}
        overlap = blocked & dca
        assert not overlap, f"DCA tickers overlap with blocked: {overlap}"


# ── Backtest Config Match Tests ──────────────────────────────────────────


class TestBacktestMatchesProduction:
    """Verify backtest constants match production docker-compose.yml config."""

    def test_max_concurrent_matches_production(self):
        """Backtest MAX_CONCURRENT should match docker-compose (8)."""
        import scripts.backtest_gold_standard as gs
        assert gs.MAX_CONCURRENT == 8, (
            f"Backtest MAX_CONCURRENT={gs.MAX_CONCURRENT}, production=8"
        )

    def test_max_same_direction_matches_concurrent(self):
        """No same-direction cap in production — backtest should match."""
        import scripts.backtest_gold_standard as gs
        assert gs.MAX_SAME_DIRECTION >= gs.MAX_CONCURRENT, (
            f"MAX_SAME_DIRECTION={gs.MAX_SAME_DIRECTION} should be >= "
            f"MAX_CONCURRENT={gs.MAX_CONCURRENT}"
        )

    def test_excluded_tickers_match_blocked(self):
        """Backtest exclusions should match production BLOCKED_TICKERS."""
        import scripts.backtest_gold_standard as gs
        s = Settings()
        blocked = {t.strip().upper() for t in s.BLOCKED_TICKERS.split(",") if t.strip()}
        assert gs.EXCLUDED_TICKERS == blocked, (
            f"Backtest={gs.EXCLUDED_TICKERS}, production={blocked}"
        )

    def test_no_entry_spacing_constraint(self):
        """Production has no MIN_ENTRY_SPACING — backtest should not artificially limit."""
        import scripts.backtest_gold_standard as gs
        # The new concurrent architecture removed the spacing check from the main loop.
        # Verify MIN_ENTRY_SPACING_MIN still exists as a constant but is NOT used
        # in the scan loop (it was removed from the minute-first architecture).
        import inspect
        source = inspect.getsource(gs.run_backtest)
        assert "last_entry_minute" not in source or "MIN_ENTRY_SPACING" not in source, (
            "run_backtest should not use MIN_ENTRY_SPACING in the scan loop"
        )


# ── Concurrent Position Simulation Tests ─────────────────────────────────


class TestConcurrentPositionArchitecture:
    """Test that the backtest minute-first architecture correctly handles
    concurrent positions — multiple trades can be open and close independently."""

    def test_backtest_imports_cleanly(self):
        """Verify the refactored backtest module loads without errors."""
        import scripts.backtest_gold_standard as gs
        assert hasattr(gs, "run_backtest")
        assert hasattr(gs, "simulate_exit")

    def test_simulate_exit_returns_expected_keys(self):
        """simulate_exit should return pnl, reason, hold_min, peak_gain, exit_prem."""
        import numpy as np
        import scripts.backtest_gold_standard as gs

        # Create synthetic data: premium goes up 30% then comes back
        n = 100
        closes = np.array([1.0 + 0.3 * (i / 30) if i < 30 else 1.3 - 0.01 * (i - 30)
                           for i in range(n)])
        bids = closes * 0.95
        asks = closes * 1.05
        underlyings = np.full(n, 500.0)

        result = gs.simulate_exit(
            closes, bids, asks, underlyings,
            entry_idx=0, entry_premium=1.0, contracts=5,
            ticker="SPY", dte=0, expiry_date="2026-01-01",
        )

        assert "pnl" in result
        assert "reason" in result
        assert "hold_min" in result
        assert "peak_gain" in result
        assert "exit_prem" in result
        # Should have exited profitably (30% gain hit triggers)
        assert result["peak_gain"] > 0
