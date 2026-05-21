"""Tests for the liquidity filter (open interest / volume / bid-ask spread)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.risk.liquidity_filter import (
    OptionLiquidity,
    _calc_spread_pct,
    check_liquidity,
)
from options_owl.risk.pipeline import GateResult, LiquidityGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __init__(self, **kwargs):
        defaults = {
            "ENABLE_LIQUIDITY_FILTER": True,
            "MIN_OPEN_INTEREST": 100,
            "MIN_VOLUME": 50,
            "MAX_BID_ASK_SPREAD_PCT": 15.0,
            "POLYGON_API_KEY": "",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def _make_liquidity(**kwargs) -> OptionLiquidity:
    defaults = {
        "ticker": "SPY",
        "strike": 560.0,
        "expiry": "2026-03-30",
        "option_type": "call",
    }
    defaults.update(kwargs)
    return OptionLiquidity(**defaults)


def _gate_ctx(**overrides):
    signal = MagicMock()
    signal.ticker = "SPY"
    signal.strike = 560.0
    signal.expiry = "2026-03-30"
    signal.direction.value = "call"

    ctx = {
        "signal": signal,
        "settings": _FakeSettings(),
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# check_liquidity tests
# ---------------------------------------------------------------------------


class TestCheckLiquidity:
    def test_disabled_filter_passes(self):
        """When filter is called, but data has good values, it passes."""
        settings = _FakeSettings(ENABLE_LIQUIDITY_FILTER=False)
        liq = _make_liquidity(open_interest=500, volume=200, bid_ask_spread_pct=5.0)
        passes, reason = check_liquidity(liq, settings)
        assert passes is True

    def test_low_open_interest_blocks(self):
        settings = _FakeSettings()
        liq = _make_liquidity(open_interest=10, volume=200, bid_ask_spread_pct=5.0)
        passes, reason = check_liquidity(liq, settings)
        assert passes is False
        assert "Open interest" in reason
        assert "10" in reason

    def test_low_volume_blocks(self):
        settings = _FakeSettings()
        liq = _make_liquidity(open_interest=500, volume=10, bid_ask_spread_pct=5.0)
        passes, reason = check_liquidity(liq, settings)
        assert passes is False
        assert "Volume" in reason
        assert "10" in reason

    def test_wide_spread_blocks(self):
        settings = _FakeSettings()
        liq = _make_liquidity(open_interest=500, volume=200, bid_ask_spread_pct=25.0)
        passes, reason = check_liquidity(liq, settings)
        assert passes is False
        assert "spread" in reason.lower()

    def test_all_checks_pass_with_good_liquidity(self):
        settings = _FakeSettings()
        liq = _make_liquidity(
            open_interest=500,
            volume=200,
            bid=2.50,
            ask=2.60,
            bid_ask_spread_pct=3.92,
        )
        passes, reason = check_liquidity(liq, settings)
        assert passes is True
        assert "OI=500" in reason
        assert "Vol=200" in reason

    def test_missing_data_passes_by_default(self):
        settings = _FakeSettings()
        liq = _make_liquidity()  # all None
        passes, reason = check_liquidity(liq, settings)
        assert passes is True
        assert "passed by default" in reason.lower()

    def test_partial_data_checks_available(self):
        """If only OI is available and passes, it should pass."""
        settings = _FakeSettings()
        liq = _make_liquidity(open_interest=500)
        passes, reason = check_liquidity(liq, settings)
        assert passes is True
        assert "OI=500" in reason

    def test_partial_data_fails_on_available(self):
        """If only OI is available and it fails, block the trade."""
        settings = _FakeSettings()
        liq = _make_liquidity(open_interest=10)
        passes, reason = check_liquidity(liq, settings)
        assert passes is False


# ---------------------------------------------------------------------------
# _calc_spread_pct tests
# ---------------------------------------------------------------------------


class TestCalcSpreadPct:
    def test_normal_spread(self):
        result = _calc_spread_pct(2.50, 2.60)
        # spread = 0.10, midpoint = 2.55, pct = 3.92%
        assert result is not None
        assert abs(result - 3.92) < 0.1

    def test_none_bid(self):
        assert _calc_spread_pct(None, 2.60) is None

    def test_none_ask(self):
        assert _calc_spread_pct(2.50, None) is None

    def test_zero_bid(self):
        assert _calc_spread_pct(0, 2.60) is None

    def test_zero_ask(self):
        assert _calc_spread_pct(2.50, 0) is None


# ---------------------------------------------------------------------------
# Polygon REST parsing (mocked)
# ---------------------------------------------------------------------------


class TestPolygonFetch:
    @pytest.mark.asyncio
    async def test_polygon_rest_parsing(self):
        """Test that Polygon REST response is correctly parsed into OptionLiquidity."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "open_interest": 1500,
                "day": {"volume": 320},
                "last_quote": {"bid": 3.10, "ask": 3.30},
            }
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("options_owl.risk.liquidity_filter.httpx", create=True):
            with patch.dict("sys.modules", {"httpx": MagicMock()}):
                import sys
                mock_httpx_mod = sys.modules["httpx"]
                mock_httpx_mod.AsyncClient = MagicMock(return_value=mock_client_instance)

                with patch(
                    "options_owl.risk.liquidity_filter.httpx",
                    mock_httpx_mod,
                    create=True,
                ):
                    # Directly test the internal function
                    from options_owl.risk.liquidity_filter import _fetch_polygon_liquidity

                    settings = _FakeSettings(POLYGON_API_KEY="test_key")
                    result = await _fetch_polygon_liquidity(
                        "SPY", 560.0, "2026-03-30", "call", settings,
                    )

        assert result is not None
        assert result.open_interest == 1500
        assert result.volume == 320
        assert result.bid == 3.10
        assert result.ask == 3.30
        assert result.bid_ask_spread_pct is not None

    @pytest.mark.asyncio
    async def test_polygon_non_200_returns_none(self):
        """Polygon returning non-200 should gracefully return None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", {"httpx": MagicMock()}) as _:
            import sys
            mock_httpx_mod = sys.modules["httpx"]
            mock_httpx_mod.AsyncClient = MagicMock(return_value=mock_client_instance)

            with patch(
                "options_owl.risk.liquidity_filter.httpx",
                mock_httpx_mod,
                create=True,
            ):
                from options_owl.risk.liquidity_filter import _fetch_polygon_liquidity

                settings = _FakeSettings(POLYGON_API_KEY="test_key")
                result = await _fetch_polygon_liquidity(
                    "SPY", 560.0, "2026-03-30", "call", settings,
                )

        assert result is None


# ---------------------------------------------------------------------------
# yfinance fallback (mocked)
# ---------------------------------------------------------------------------


class TestYfinanceFallback:
    @pytest.mark.asyncio
    async def test_yfinance_fallback(self):
        """Test yfinance option chain data is correctly parsed."""
        import pandas as pd

        mock_df = pd.DataFrame([{
            "strike": 560.0,
            "openInterest": 800,
            "volume": 150,
            "bid": 2.40,
            "ask": 2.55,
        }])

        mock_chain = MagicMock()
        mock_chain.calls = mock_df
        mock_chain.puts = pd.DataFrame()

        mock_ticker = MagicMock()
        mock_ticker.option_chain.return_value = mock_chain

        with patch("options_owl.risk.liquidity_filter.yf", create=True) as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker

            # The function uses asyncio.to_thread, so we need to patch at import level
            from options_owl.risk.liquidity_filter import _fetch_yfinance_liquidity

            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = await _fetch_yfinance_liquidity(
                    "SPY", 560.0, "2026-03-30", "call",
                )

        assert result is not None
        assert result.open_interest == 800
        assert result.volume == 150
        assert result.bid == 2.40
        assert result.ask == 2.55
        assert result.bid_ask_spread_pct is not None

    @pytest.mark.asyncio
    async def test_yfinance_no_matching_strike(self):
        """yfinance chain with no matching strike should return None."""
        import pandas as pd

        mock_df = pd.DataFrame([{
            "strike": 570.0,  # wrong strike
            "openInterest": 800,
            "volume": 150,
            "bid": 2.40,
            "ask": 2.55,
        }])

        mock_chain = MagicMock()
        mock_chain.calls = mock_df

        mock_ticker = MagicMock()
        mock_ticker.option_chain.return_value = mock_chain

        with patch("yfinance.Ticker", return_value=mock_ticker):
            from options_owl.risk.liquidity_filter import _fetch_yfinance_liquidity

            result = await _fetch_yfinance_liquidity(
                "SPY", 560.0, "2026-03-30", "call",
            )

        assert result is None


# ---------------------------------------------------------------------------
# LiquidityGate pipeline tests
# ---------------------------------------------------------------------------


class TestLiquidityGate:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        ctx = _gate_ctx(settings=_FakeSettings(ENABLE_LIQUIDITY_FILTER=False))
        r = await LiquidityGate().evaluate(ctx)
        assert r.result == GateResult.SKIP
        assert "disabled" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_pass_when_liquid(self):
        good_liq = _make_liquidity(
            open_interest=500, volume=200, bid_ask_spread_pct=5.0,
        )
        with patch(
            "options_owl.risk.liquidity_filter.fetch_option_liquidity",
            new_callable=AsyncMock,
            return_value=good_liq,
        ):
            ctx = _gate_ctx()
            r = await LiquidityGate().evaluate(ctx)
            assert r.result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_fail_when_illiquid(self):
        bad_liq = _make_liquidity(
            open_interest=5, volume=200, bid_ask_spread_pct=5.0,
        )
        with patch(
            "options_owl.risk.liquidity_filter.fetch_option_liquidity",
            new_callable=AsyncMock,
            return_value=bad_liq,
        ):
            ctx = _gate_ctx()
            r = await LiquidityGate().evaluate(ctx)
            assert r.result == GateResult.FAIL
            assert "Open interest" in r.reason

    @pytest.mark.asyncio
    async def test_error_skips(self):
        with patch(
            "options_owl.risk.liquidity_filter.fetch_option_liquidity",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            ctx = _gate_ctx()
            r = await LiquidityGate().evaluate(ctx)
            assert r.result == GateResult.SKIP
            assert "error" in r.reason.lower()
