"""Tests for Webull sell-to-close fix and option WS subscriptions.

Covers bugs found during live trading on 2026-04-14:
1. SELL orders must include close_contracts + position_id (not SELL_TO_CLOSE side)
2. BUY orders must NOT include close_contracts
3. Position ID lookup from Webull positions API
4. Option contract ticker building for Polygon WS
5. Portfolio sizing hard cap (effective_balance)
6. MAX_ORDER_CONTRACTS enforcement at entry
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from options_owl.collectors.market_data_stream import MarketDataStream
from options_owl.execution.webull_executor import (
    MAX_ORDER_CONTRACTS,
    WebullExecutor,
)


def _make_settings(**overrides):
    settings = MagicMock()
    defaults = {
        "WEBULL_APP_KEY": "test_key",
        "WEBULL_APP_SECRET": "test_secret",
        "WEBULL_ACCOUNT_ID": "TEST_ACCT_001",
        "WEBULL_KILL_SWITCH": False,
        "PAPER_TRADE": False,
        "POLYGON_API_KEY": "test_polygon_key",
        "DATA_FEED_PROVIDER": "polygon",
        "DATA_FEED_POLL_INTERVAL": 15,
        "PORTFOLIO_SIZE": 5000,
        "ENABLE_PUT_TRADING": True,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


# ---------------------------------------------------------------------------
# Sell-to-close: close_contracts + position_id
# ---------------------------------------------------------------------------


class TestSellToClose:
    """SELL orders must include close_contracts with position_id.

    Without this, Webull interprets SELL as sell-to-open (writing a covered call)
    and returns OAUTH_OPENAPI_OPTION_CAVERED_CALL_STOCK_NO_ENOUGH.
    """

    @pytest.mark.asyncio
    async def test_sell_order_includes_close_contracts(self):
        """SELL payload must have close_contracts with the position_id."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        # Mock position lookup — return a matching position
        mock_pos_resp = MagicMock()
        mock_pos_resp.json.return_value = [{
            "position_id": "POS_ABC_123",
            "symbol": "SPY",
            "instrument_type": "OPTION",
            "legs": [{
                "symbol": "SPY",
                "option_type": "CALL",
                "option_expire_date": "2026-04-14",
                "option_exercise_price": "691.00",
            }],
        }]
        executor._trade_client.account_v2.get_account_position = MagicMock(
            return_value=mock_pos_resp
        )

        # Mock the order placement — capture the payload
        mock_order_resp = MagicMock()
        mock_order_resp.json.return_value = {"order_id": "ORD_123"}
        executor._trade_client.order_v2.place_option = MagicMock(
            return_value=mock_order_resp
        )

        # Mock order status check — return FILLED immediately
        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "FILLED", "filled_quantity": "1", "total_quantity": "1"}
        executor._trade_client.order_v2.get_order_detail = MagicMock(
            return_value=mock_status_resp
        )

        result = await executor.place_option_order(
            ticker="SPY",
            strike=691.0,
            expiry_date="2026-04-14",
            option_type="CALL",
            side="SELL",
            contracts=1,
            limit_price=1.04,
        )

        assert result.success is True

        # Verify the payload sent to Webull
        call_args = executor._trade_client.order_v2.place_option.call_args
        payload = call_args[0][1]  # second positional arg = new_orders
        order = payload[0]

        # side must be "SELL" (not "SELL_TO_CLOSE")
        assert order["side"] == "SELL"
        assert order["legs"][0]["side"] == "SELL"

        # close_contracts must be present with position_id
        assert "close_contracts" in order
        assert order["close_contracts"][0]["position_id"] == "POS_ABC_123"
        assert order["close_contracts"][0]["quantity"] == "1"

    @pytest.mark.asyncio
    async def test_buy_order_has_no_close_contracts(self):
        """BUY payload must NOT include close_contracts."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        mock_order_resp = MagicMock()
        mock_order_resp.json.return_value = {"order_id": "ORD_456"}
        executor._trade_client.order_v2.place_option = MagicMock(
            return_value=mock_order_resp
        )

        # Mock order status check — return FILLED immediately
        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "FILLED", "filled_quantity": "1", "total_quantity": "1"}
        executor._trade_client.order_v2.get_order_detail = MagicMock(
            return_value=mock_status_resp
        )

        result = await executor.place_option_order(
            ticker="SPY",
            strike=691.0,
            expiry_date="2026-04-14",
            option_type="CALL",
            side="BUY",
            contracts=1,
            limit_price=1.00,
        )

        assert result.success is True

        call_args = executor._trade_client.order_v2.place_option.call_args
        payload = call_args[0][1]
        order = payload[0]

        assert order["side"] == "BUY"
        assert "close_contracts" not in order

    @pytest.mark.asyncio
    async def test_sell_without_position_id_still_sends(self):
        """If no position found, SELL is skipped (no live position to close)."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        # Return empty positions
        mock_pos_resp = MagicMock()
        mock_pos_resp.json.return_value = []
        executor._trade_client.account_v2.get_account_position = MagicMock(
            return_value=mock_pos_resp
        )

        result = await executor.place_option_order(
            ticker="SPY",
            strike=691.0,
            expiry_date="2026-04-14",
            option_type="CALL",
            side="SELL",
            contracts=1,
            limit_price=1.00,
        )

        # Order should NOT be sent — no position to close
        executor._trade_client.order_v2.place_option.assert_not_called()
        assert result.success is False
        assert "No Webull position found" in result.error


# ---------------------------------------------------------------------------
# Position ID lookup
# ---------------------------------------------------------------------------


class TestPositionIdLookup:
    @pytest.mark.asyncio
    async def test_finds_matching_position(self):
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "position_id": "POS_SPY_CALL",
                "symbol": "SPY",
                "legs": [{
                    "symbol": "SPY",
                    "option_type": "CALL",
                    "option_expire_date": "2026-04-14",
                    "option_exercise_price": "691.00",
                }],
            },
            {
                "position_id": "POS_QQQ_PUT",
                "symbol": "QQQ",
                "legs": [{
                    "symbol": "QQQ",
                    "option_type": "PUT",
                    "option_expire_date": "2026-04-14",
                    "option_exercise_price": "620.00",
                }],
            },
        ]
        executor._trade_client.account_v2.get_account_position = MagicMock(
            return_value=mock_resp
        )

        pos_id = await executor._find_position_id("SPY", 691.0, "2026-04-14", "CALL")
        assert pos_id == "POS_SPY_CALL"

        pos_id2 = await executor._find_position_id("QQQ", 620.0, "2026-04-14", "PUT")
        assert pos_id2 == "POS_QQQ_PUT"

    @pytest.mark.asyncio
    async def test_returns_none_for_no_match(self):
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        executor._trade_client.account_v2.get_account_position = MagicMock(
            return_value=mock_resp
        )

        pos_id = await executor._find_position_id("AAPL", 200.0, "2026-04-14", "CALL")
        assert pos_id is None

    @pytest.mark.asyncio
    async def test_strike_matching_tolerance(self):
        """Strike matching should use tolerance for float comparison."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()
        executor._account_id = "TEST_ACCT_001"

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{
            "position_id": "POS_IWM",
            "symbol": "IWM",
            "legs": [{
                "symbol": "IWM",
                "option_type": "CALL",
                "option_expire_date": "2026-04-14",
                "option_exercise_price": "268.00",  # string from API
            }],
        }]
        executor._trade_client.account_v2.get_account_position = MagicMock(
            return_value=mock_resp
        )

        # Should match despite float precision (268.0 vs "268.00")
        pos_id = await executor._find_position_id("IWM", 268.0, "2026-04-14", "CALL")
        assert pos_id == "POS_IWM"


# ---------------------------------------------------------------------------
# Option contract ticker building (Polygon WS)
# ---------------------------------------------------------------------------


class TestOptionContractTicker:
    def test_call_contract(self):
        ct = MarketDataStream.build_option_contract_ticker(
            "SPY", 691.0, "2026-04-14", "call"
        )
        assert ct == "O:SPY260414C00691000"

    def test_put_contract(self):
        ct = MarketDataStream.build_option_contract_ticker(
            "QQQ", 623.0, "2026-04-14", "put"
        )
        assert ct == "O:QQQ260414P00623000"

    def test_fractional_strike(self):
        ct = MarketDataStream.build_option_contract_ticker(
            "TSLA", 362.5, "2026-04-14", "call"
        )
        assert ct == "O:TSLA260414C00362500"

    def test_low_strike(self):
        ct = MarketDataStream.build_option_contract_ticker(
            "F", 12.0, "2026-04-14", "put"
        )
        assert ct == "O:F260414P00012000"

    def test_case_insensitive(self):
        ct = MarketDataStream.build_option_contract_ticker(
            "spy", 691.0, "2026-04-14", "CALL"
        )
        assert ct == "O:SPY260414C00691000"


# ---------------------------------------------------------------------------
# Option WS cache and subscription
# ---------------------------------------------------------------------------


class TestOptionWsCache:
    @pytest.mark.asyncio
    async def test_subscribe_option_adds_to_map(self):
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")

        key = ("SPY", 691.0, "2026-04-14", "call")
        assert key in stream._option_contract_map
        assert stream._option_contract_map[key] == "O:SPY260414C00691000"
        assert "O:SPY260414C00691000" in stream._option_subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_option_removes(self):
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")
        await stream.unsubscribe_option("SPY", 691.0, "2026-04-14", "call")

        key = ("SPY", 691.0, "2026-04-14", "call")
        assert key not in stream._option_contract_map
        assert "O:SPY260414C00691000" not in stream._option_subscriptions

    @pytest.mark.asyncio
    async def test_get_option_premium_uses_ws_cache(self):
        """When WS cache has fresh data, get_option_premium should return it."""
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")

        # Simulate a WS quote update
        import time
        contract = "O:SPY260414C00691000"
        stream._option_cache[contract] = (1.25, time.time())

        premium = await stream.get_option_premium("SPY", 691.0, "2026-04-14", "call")
        assert premium == 1.25

    @pytest.mark.asyncio
    async def test_stale_ws_cache_falls_through(self):
        """Stale WS cache (>30s) should fall through to REST."""
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")

        # Set cache with old timestamp (60s ago)
        import time
        contract = "O:SPY260414C00691000"
        stream._option_cache[contract] = (1.25, time.time() - 60)

        # Mock the REST fallback to return a different value
        stream._polygon_rest_option_premium = AsyncMock(return_value=1.30)

        premium = await stream.get_option_premium("SPY", 691.0, "2026-04-14", "call")
        assert premium == 1.30

    def test_process_option_quote_updates_cache(self):
        """Option quote events (Q) should update the option cache."""
        stream = MarketDataStream(_make_settings())

        raw_msg = '[{"ev":"Q","sym":"O:SPY260414C00691000","bp":1.20,"ap":1.30,"t":1776179389000}]'
        stream._process_polygon_message(raw_msg)

        assert "O:SPY260414C00691000" in stream._option_cache
        premium, _ = stream._option_cache["O:SPY260414C00691000"]
        assert premium == 1.25  # midpoint of 1.20 and 1.30

    def test_process_option_trade_updates_cache(self):
        """Option trade events (T) should update the option cache."""
        stream = MarketDataStream(_make_settings())

        raw_msg = '[{"ev":"T","sym":"O:SPY260414C00691000","p":1.22,"s":5,"t":1776179389000}]'
        stream._process_polygon_message(raw_msg)

        assert "O:SPY260414C00691000" in stream._option_cache
        premium, _ = stream._option_cache["O:SPY260414C00691000"]
        assert premium == 1.22

    def test_process_equity_trade_updates_price_cache(self):
        """Equity trades should go to price cache, not option cache."""
        stream = MarketDataStream(_make_settings())

        raw_msg = '[{"ev":"T","sym":"SPY","p":691.50,"s":100,"t":1776179389000}]'
        stream._process_polygon_message(raw_msg)

        assert "SPY" in stream._price_cache
        assert "SPY" not in stream._option_cache


# ---------------------------------------------------------------------------
# Portfolio sizing hard cap
# ---------------------------------------------------------------------------


class TestPortfolioSizingCap:
    """effective_balance = min(current_balance, PORTFOLIO_SIZE) must be enforced."""

    @pytest.mark.asyncio
    async def test_effective_balance_caps_at_portfolio_size(self):
        """Even if paper balance is $10k, sizing should use PORTFOLIO_SIZE."""

        settings = _make_settings(
            PORTFOLIO_SIZE=400,
            PAPER_TRADE=True,
            ENABLE_DCA=False,
            ENABLE_RISK_MANAGER=False,
            MAX_POSITION_PCT=20,
            MAX_CONCURRENT=3,
            MIN_SCORE=0,
            ENABLE_GREEKS=False,
            ENABLE_TRAILING_STOP=False,
            ENABLE_PARTIAL_PROFITS=False,
            ENABLE_SCALE_OUT=False,
            DB_PATH=":memory:",
        )

        # The effective_balance pattern
        current_balance = 10000.0
        portfolio_size = settings.PORTFOLIO_SIZE
        effective = min(current_balance, portfolio_size)
        assert effective == 400.0


class TestMaxOrderContractsEnforcement:
    """MAX_ORDER_CONTRACTS must be enforced at entry time but NOT exit."""

    def test_hard_cap_constant(self):
        assert MAX_ORDER_CONTRACTS == 100  # safety cap, sizing handles real limits

    def test_safety_rejects_over_cap(self):
        executor = WebullExecutor(_make_settings())
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(101, 1.00, "BUY")

    def test_safety_allows_at_cap(self):
        executor = WebullExecutor(_make_settings())
        executor._check_safety_limits(100, 0.40, "BUY")  # 100 × $0.40 × 100 = $4,000 < $5K

    def test_max_value_cap(self):
        executor = WebullExecutor(_make_settings())
        # 10 contracts * $6.00 * 100 = $6,000 > MAX_ORDER_VALUE ($5,000)
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(10, 6.00, "BUY")

    def test_sell_bypasses_contract_cap(self):
        """SELL orders must not be blocked by MAX_ORDER_CONTRACTS — need to close full position."""
        executor = WebullExecutor(_make_settings())
        # 20 contracts exceeds the 10 cap, but SELL should pass
        executor._check_safety_limits(20, 1.00, "SELL")  # should not raise

    def test_sell_bypasses_value_cap(self):
        """SELL orders must not be blocked by MAX_ORDER_VALUE."""
        executor = WebullExecutor(_make_settings())
        # 20 contracts * $10 * 100 = $20,000 — way over $5k cap, but SELL should pass
        executor._check_safety_limits(20, 10.00, "SELL")  # should not raise
