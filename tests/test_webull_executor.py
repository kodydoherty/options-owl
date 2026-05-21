"""Tests for Webull executor — safety rails, order validation, kill switch, quotes."""

from unittest.mock import MagicMock

import pytest

from options_owl.execution.webull_executor import (
    MAX_ORDER_CONTRACTS,
    OrderResult,
    WebullExecutor,
    _round_option_price,
)


def _make_settings(**overrides):
    settings = MagicMock()
    defaults = {
        "WEBULL_APP_KEY": "test_key",
        "WEBULL_APP_SECRET": "test_secret",
        "WEBULL_ACCOUNT_ID": "12345",
        "WEBULL_KILL_SWITCH": False,
        "PAPER_TRADE": True,
        "MARGIN_ACCOUNT": False,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


class TestSafetyRails:
    def test_paper_trade_blocks_orders(self):
        """Orders must be blocked when PAPER_TRADE=True."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=True))
        with pytest.raises(RuntimeError, match="PAPER_TRADE=true"):
            executor._check_safety_limits(1, 2.00, "BUY")

    def test_kill_switch_blocks_orders(self):
        """Kill switch must block all orders."""
        executor = WebullExecutor(_make_settings(WEBULL_KILL_SWITCH=True))
        with pytest.raises(RuntimeError, match="KILL_SWITCH"):
            executor._check_kill_switch()

    def test_max_contracts_enforced(self):
        """Cannot exceed MAX_ORDER_CONTRACTS."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(MAX_ORDER_CONTRACTS + 1, 1.00, "BUY")

    def test_max_value_enforced(self):
        """Cannot exceed MAX_ORDER_VALUE."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        # 5 contracts * $20.00 * 100 = $10,000 > $5,000 cap
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(5, 20.00, "BUY")

    def test_valid_order_passes(self):
        """Valid order within limits should not raise."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        # 3 contracts * $2.00 * 100 = $600 < $5,000 cap
        executor._check_safety_limits(3, 2.00, "BUY")  # should not raise

    def test_kill_switch_off_passes(self):
        executor = WebullExecutor(_make_settings(WEBULL_KILL_SWITCH=False))
        executor._check_kill_switch()  # should not raise


class TestMissingCredentials:
    def test_no_app_key_raises(self):
        executor = WebullExecutor(_make_settings(WEBULL_APP_KEY="", WEBULL_APP_SECRET="secret"))
        with pytest.raises(RuntimeError, match="WEBULL_APP_KEY"):
            executor._ensure_clients()

    def test_no_app_secret_raises(self):
        executor = WebullExecutor(_make_settings(WEBULL_APP_KEY="key", WEBULL_APP_SECRET=""))
        with pytest.raises(RuntimeError, match="WEBULL_APP_KEY"):
            executor._ensure_clients()


class TestOrderResult:
    def test_success_result(self):
        r = OrderResult(success=True, order_id="123", client_order_id="abc")
        assert r.success is True
        assert r.order_id == "123"

    def test_failure_result(self):
        r = OrderResult(success=False, error="insufficient funds")
        assert r.success is False
        assert r.error == "insufficient funds"


# ---------------------------------------------------------------------------
# Cash account enforcement
# ---------------------------------------------------------------------------

# Simulated Webull account list (matches real API response structure)
_MOCK_ACCOUNTS = [
    {"account_id": "MARGIN_ID_001", "account_type": "MARGIN",
     "account_class": "INDIVIDUAL_MARGIN", "account_label": "Individual Margin"},
    {"account_id": "CASH_ID_002", "account_type": "CASH",
     "account_class": "INDIVIDUAL_CASH", "account_label": "Individual Cash"},
    {"account_id": "FUTURES_ID_003", "account_type": "MARGIN",
     "account_class": "FUTURES", "account_label": "Futures"},
    {"account_id": "CRYPTO_ID_004", "account_type": "CASH",
     "account_class": "CRYPTO", "account_label": "Crypto"},
]


class TestCashAccountEnforcement:
    """OptionsOwl must ONLY trade on cash accounts, never margin."""

    @pytest.mark.asyncio
    async def test_auto_detect_selects_cash_account(self):
        """Auto-detect should pick the Individual Cash account when MARGIN_ACCOUNT=false."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="", MARGIN_ACCOUNT=False))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        account_id = await executor._detect_account_id()
        assert account_id == "CASH_ID_002"

    @pytest.mark.asyncio
    async def test_auto_detect_selects_margin_account(self):
        """Auto-detect should pick the Individual Margin account when MARGIN_ACCOUNT=true."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="", MARGIN_ACCOUNT=True))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        account_id = await executor._detect_account_id()
        assert account_id == "MARGIN_ID_001"

    @pytest.mark.asyncio
    async def test_auto_detect_skips_margin_when_cash_mode(self):
        """In cash mode, auto-detect must not select a MARGIN account."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="", MARGIN_ACCOUNT=False))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        account_id = await executor._detect_account_id()
        assert account_id != "MARGIN_ID_001"
        assert account_id != "FUTURES_ID_003"

    @pytest.mark.asyncio
    async def test_auto_detect_skips_crypto(self):
        """Auto-detect must not select CRYPTO accounts."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID=""))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        account_id = await executor._detect_account_id()
        assert account_id != "CRYPTO_ID_004"

    @pytest.mark.asyncio
    async def test_auto_detect_raises_if_no_matching_account(self):
        """If no account of the requested type exists, should raise."""
        # Cash mode but only margin accounts available
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="", MARGIN_ACCOUNT=False))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        margin_only = [
            {"account_id": "MARGIN_ONLY", "account_type": "MARGIN",
             "account_class": "INDIVIDUAL_MARGIN"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = margin_only
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="CASH"):
            await executor._detect_account_id()

    @pytest.mark.asyncio
    async def test_auto_detect_raises_if_no_margin_account(self):
        """Margin mode but only cash accounts available should raise."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="", MARGIN_ACCOUNT=True))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        cash_only = [
            {"account_id": "CASH_ONLY", "account_type": "CASH",
             "account_class": "INDIVIDUAL_CASH"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = cash_only
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="MARGIN"):
            await executor._detect_account_id()

    @pytest.mark.asyncio
    async def test_verify_rejects_margin_account_id(self):
        """If WEBULL_ACCOUNT_ID points to a margin account, init must fail."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="MARGIN_ID_001"))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="never margin"):
            await executor._verify_cash_account("MARGIN_ID_001")

    @pytest.mark.asyncio
    async def test_verify_accepts_cash_account_id(self):
        """Configured CASH account ID should pass verification."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="CASH_ID_002"))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        # Should not raise
        await executor._verify_cash_account("CASH_ID_002")

    @pytest.mark.asyncio
    async def test_verify_rejects_futures_account(self):
        """Futures account should be rejected."""
        executor = WebullExecutor(_make_settings(WEBULL_ACCOUNT_ID="FUTURES_ID_003"))
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_ACCOUNTS
        executor._trade_client.account_v2.get_account_list = MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="never margin"):
            await executor._verify_cash_account("FUTURES_ID_003")


# ---------------------------------------------------------------------------
# Option quote parsing
# ---------------------------------------------------------------------------


class TestOptionSnapshotParsing:
    """Test _parse_option_snapshot handles various Webull response formats."""

    def test_basic_bid_ask(self):
        data = {"bid": 1.50, "ask": 1.70, "last": 1.60}
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["bid"] == 1.50
        assert result["ask"] == 1.70
        assert result["mid"] == 1.60

    def test_camel_case_fields(self):
        data = {"bidPrice": 2.00, "askPrice": 2.20, "lastPrice": 2.10}
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["bid"] == 2.00
        assert result["ask"] == 2.20
        assert result["mid"] == 2.10

    def test_underscore_fields(self):
        data = {"bid_price": 0.50, "ask_price": 0.60}
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["mid"] == 0.55

    def test_nested_quote_structure(self):
        data = {"quote": {"bid": 3.00, "ask": 3.40}}
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["mid"] == 3.20

    def test_last_price_only_fallback(self):
        data = {"close": 1.25}
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["mid"] == 1.25
        assert result["bid"] == 0.0

    def test_empty_data_returns_none(self):
        assert WebullExecutor._parse_option_snapshot({}) is None
        assert WebullExecutor._parse_option_snapshot([]) is None

    def test_list_wrapper(self):
        data = [{"bid": 1.00, "ask": 1.10}]
        result = WebullExecutor._parse_option_snapshot(data)
        assert result is not None
        assert result["mid"] == 1.05

    def test_zero_bid_ask_returns_none(self):
        data = {"bid": 0, "ask": 0}
        assert WebullExecutor._parse_option_snapshot(data) is None


class TestOptionQuotesParsing:
    """Test _parse_option_quotes handles depth quote formats."""

    def test_flat_bid_ask(self):
        data = {"bid": 2.00, "ask": 2.30}
        result = WebullExecutor._parse_option_quotes(data)
        assert result is not None
        assert result["mid"] == 2.15

    def test_last_price_fallback(self):
        data = {"lastPrice": 1.50}
        result = WebullExecutor._parse_option_quotes(data)
        assert result is not None
        assert result["mid"] == 1.50

    def test_empty_returns_none(self):
        assert WebullExecutor._parse_option_quotes({}) is None


class TestInstrumentCache:
    """Test instrument_id caching for option lookups."""

    @pytest.mark.asyncio
    async def test_instrument_id_cached(self):
        """Second lookup should use cache, not call API again."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"instrument_id": "INS_123"}
        executor._trade_client.trade_instrument.get_trade_security_detail = MagicMock(
            return_value=mock_resp
        )

        # First call — hits API
        result1 = await executor._lookup_instrument_id("SPY", 550.0, "2026-04-22", "call")
        assert result1 == "INS_123"

        # Second call — should use cache
        result2 = await executor._lookup_instrument_id("SPY", 550.0, "2026-04-22", "call")
        assert result2 == "INS_123"

        # API should only be called once
        assert executor._trade_client.trade_instrument.get_trade_security_detail.call_count == 1

    @pytest.mark.asyncio
    async def test_instrument_lookup_failure_returns_none(self):
        """Failed instrument lookup should return None, not crash."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        executor._trade_client.trade_instrument.get_trade_security_detail = MagicMock(
            side_effect=Exception("API error")
        )

        result = await executor._lookup_instrument_id("SPY", 550.0, "2026-04-22", "call")
        assert result is None


class TestGetOptionQuote:
    """Test the full get_option_quote flow."""

    @pytest.mark.asyncio
    async def test_returns_quote_on_success(self):
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        # Mock instrument lookup
        inst_resp = MagicMock()
        inst_resp.json.return_value = {"instrument_id": "INS_SPY_550C"}
        executor._trade_client.trade_instrument.get_trade_security_detail = MagicMock(
            return_value=inst_resp
        )

        # Mock data client snapshot
        executor._data_client = MagicMock()
        snap_resp = MagicMock()
        snap_resp.json.return_value = {"bid": 1.50, "ask": 1.70, "last": 1.60}
        executor._data_client.market_data.get_snapshot = MagicMock(return_value=snap_resp)

        result = await executor.get_option_quote("SPY", 550.0, "2026-04-22", "call")
        assert result is not None
        assert result["bid"] == 1.50
        assert result["ask"] == 1.70
        assert result["mid"] == 1.60
        assert result["instrument_id"] == "INS_SPY_550C"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_instrument(self):
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        inst_resp = MagicMock()
        inst_resp.json.return_value = {}
        executor._trade_client.trade_instrument.get_trade_security_detail = MagicMock(
            return_value=inst_resp
        )

        result = await executor.get_option_quote("SPY", 550.0, "2026-04-22", "call")
        assert result is None

    @pytest.mark.asyncio
    async def test_falls_back_to_get_quotes(self):
        """If snapshot fails, should try get_quotes."""
        executor = WebullExecutor(_make_settings())
        executor._api_client = MagicMock()
        executor._trade_client = MagicMock()

        inst_resp = MagicMock()
        inst_resp.json.return_value = {"instrument_id": "INS_123"}
        executor._trade_client.trade_instrument.get_trade_security_detail = MagicMock(
            return_value=inst_resp
        )

        executor._data_client = MagicMock()
        # Snapshot returns no usable data
        snap_resp = MagicMock()
        snap_resp.json.return_value = {}
        executor._data_client.market_data.get_snapshot = MagicMock(return_value=snap_resp)

        # get_quotes returns valid data
        quotes_resp = MagicMock()
        quotes_resp.json.return_value = {"bid": 2.00, "ask": 2.20}
        executor._data_client.market_data.get_quotes = MagicMock(return_value=quotes_resp)

        result = await executor.get_option_quote("SPY", 550.0, "2026-04-22", "call")
        assert result is not None
        assert result["mid"] == 2.10

    @pytest.mark.asyncio
    async def test_quote_cache_works(self):
        """Cached quotes should be returned without API call."""
        import time

        executor = WebullExecutor(_make_settings())
        executor._instrument_cache[("SPY", 550.0, "2026-04-22", "call")] = "INS_123"
        executor._quote_cache["INS_123"] = (1.50, 1.70, 1.60, time.time())

        # No data client needed — cache should serve the result
        result = await executor.get_option_quote("SPY", 550.0, "2026-04-22", "call")
        assert result is not None
        assert result["mid"] == 1.60


# ---------------------------------------------------------------------------
# Price step rounding (Webull requires $0.05 increments for premium >= $3.00)
# ---------------------------------------------------------------------------


class TestOptionPriceRounding:
    """Webull rejects orders with premium >= $3.00 not in $0.05 steps."""

    def test_below_3_no_rounding(self):
        assert _round_option_price(2.99, "BUY") == 2.99
        assert _round_option_price(1.48, "SELL") == 1.48
        assert _round_option_price(0.01, "BUY") == 0.01

    def test_buy_rounds_up_to_nickel(self):
        assert _round_option_price(3.22, "BUY") == 3.25
        assert _round_option_price(6.59, "BUY") == 6.60
        assert _round_option_price(4.78, "BUY") == 4.80

    def test_sell_rounds_down_to_nickel(self):
        assert _round_option_price(3.22, "SELL") == 3.20
        assert _round_option_price(6.59, "SELL") == 6.55
        assert _round_option_price(4.78, "SELL") == 4.75

    def test_already_on_nickel_no_change(self):
        assert _round_option_price(3.00, "BUY") == 3.00
        assert _round_option_price(5.25, "SELL") == 5.25
        assert _round_option_price(10.00, "BUY") == 10.00

    def test_exact_boundary_3(self):
        assert _round_option_price(3.00, "SELL") == 3.00
        assert _round_option_price(3.01, "SELL") == 3.00
        assert _round_option_price(3.01, "BUY") == 3.05

    def test_case_insensitive_side(self):
        assert _round_option_price(3.22, "buy") == 3.25
        assert _round_option_price(3.22, "sell") == 3.20
