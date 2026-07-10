"""Tests for Webull executor — safety rails, order validation, kill switch, quotes."""

from unittest.mock import AsyncMock, MagicMock

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
        "ENABLE_PUT_TRADING": True,
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

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_orders(self):
        """Kill switch must block all orders."""
        executor = WebullExecutor(_make_settings(WEBULL_KILL_SWITCH=True))
        with pytest.raises(RuntimeError, match="KILL_SWITCH"):
            await executor._check_kill_switch()

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

    @pytest.mark.asyncio
    async def test_kill_switch_off_passes(self):
        executor = WebullExecutor(_make_settings(WEBULL_KILL_SWITCH=False))
        await executor._check_kill_switch()  # should not raise


class TestBuyOrderClamp:
    """A BUY over the per-order cap must be CLAMPED to the cap (capture the position),
    not hard-rejected (which orphaned cheap high-conviction entries to $0 — SMCI 2026-07-10)."""

    @pytest.mark.asyncio
    async def test_oversized_buy_is_clamped_not_rejected(self):
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        executor._ensure_clients = MagicMock()
        executor._check_kill_switch = AsyncMock()
        captured = {}

        async def _fake_escalation(*, contracts, **kw):
            captured["contracts"] = contracts
            return OrderResult(success=True, order_id="x")

        executor._place_buy_with_escalation = _fake_escalation
        await executor.place_option_order(
            ticker="SMCI", strike=50.0, expiry_date="2026-07-10", option_type="CALL",
            side="BUY", contracts=MAX_ORDER_CONTRACTS + 20, limit_price=0.25,
        )
        assert captured["contracts"] == MAX_ORDER_CONTRACTS

    @pytest.mark.asyncio
    async def test_within_cap_buy_unchanged(self):
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        executor._ensure_clients = MagicMock()
        executor._check_kill_switch = AsyncMock()
        captured = {}

        async def _fake_escalation(*, contracts, **kw):
            captured["contracts"] = contracts
            return OrderResult(success=True, order_id="x")

        executor._place_buy_with_escalation = _fake_escalation
        await executor.place_option_order(
            ticker="SPY", strike=500.0, expiry_date="2026-07-10", option_type="CALL",
            side="BUY", contracts=10, limit_price=1.00,
        )
        assert captured["contracts"] == 10

    def test_safety_backstop_still_rejects_oversized(self):
        """The _check_safety_limits backstop still raises if an unclamped size ever reaches it."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        with pytest.raises(ValueError, match="hard cap"):
            executor._check_safety_limits(MAX_ORDER_CONTRACTS + 1, 0.25, "BUY")


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

    def test_build_order_payload_rounds_illegal_step(self):
        """Regression: the chase fallback could hand $3.07 (ceiling 3.12 - 0.05)
        to the payload builder, which Webull rejects (OPTION_PRICE_STEP_GTE).
        The builder must defensively snap every limit price to a legal increment."""
        executor = WebullExecutor(_make_settings(PAPER_TRADE=False))
        for side, illegal, expected in [
            ("BUY", 3.07, "3.10"),   # >= $3 buy rounds UP to nickel
            ("SELL", 3.07, "3.05"),  # >= $3 sell rounds DOWN to nickel
            ("BUY", 3.12, "3.15"),
            ("SELL", 3.12, "3.10"),
            ("BUY", 2.97, "2.97"),   # < $3 keeps penny increment
        ]:
            payload = executor._build_order_payload(
                client_order_id="x", ticker="TSLA", strike=400.0,
                expiry_date="2026-06-22", option_type="call", side=side,
                contracts=1, limit_price=illegal,
            )
            got = payload[0]["limit_price"]
            assert got == expected, f"{side} {illegal} -> {got}, want {expected}"
            # And it is always a legal step: penny < $3, nickel >= $3.
            cents = round(float(got) * 100)
            assert cents % 5 == 0 or float(got) < 3.0

    def test_case_insensitive_side(self):
        assert _round_option_price(3.22, "buy") == 3.25
        assert _round_option_price(3.22, "sell") == 3.20


def _chase_settings(**overrides):
    """Settings with the entry-chase knobs set to real numbers (MagicMock would
    return mocks that break float())."""
    defaults = {
        "PAPER_TRADE": False,
        "WEBULL_KILL_SWITCH": False,
        "WEBULL_ENTRY_AGGRESS_PCT": 5.0,
        "WEBULL_ENTRY_INDEX_AGGRESS_PCT": 10.0,
        "WEBULL_ENTRY_FILL_ATTEMPTS": 4,
        "WEBULL_ENTRY_MAX_CHASE_PCT": 15.0,
        "WEBULL_ENTRY_PER_ATTEMPT_SEC": 4.0,
        "WEBULL_ENTRY_POLL_SEC": 1.0,
        "WEBULL_ENTRY_USE_LIVE_QUOTE": True,
        "WEBULL_ENTRY_QUOTE_MAX_AGE_SEC": 20.0,
    }
    defaults.update(overrides)
    return _make_settings(**defaults)


class TestFetchAskLiveQuote:
    """#3 — chase prices off the harvester's live Redis ask, freshness-guarded, HTTP fallback."""

    @pytest.mark.asyncio
    async def test_prefers_fresh_redis_over_http(self, monkeypatch):
        import time

        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings())

        async def fresh(ck):
            assert ck == "SPY:put:733.0:2026-06-25"  # key format the harvester uses
            return {"ask": 2.50, "t": time.time()}

        monkeypatch.setattr(redis_client, "get_option_snapshot", fresh)
        executor.get_option_quote = AsyncMock(return_value={"ask": 9.99})  # must NOT be used
        ask = await executor._fetch_ask("SPY", 733.0, "2026-06-25", "put")
        assert ask == 2.50
        executor.get_option_quote.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_redis_falls_back_to_http(self, monkeypatch):
        import time

        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings(WEBULL_ENTRY_QUOTE_MAX_AGE_SEC=20.0))

        async def stale(ck):
            return {"ask": 2.50, "t": time.time() - 100}  # 100s old > 20s guard

        monkeypatch.setattr(redis_client, "get_option_snapshot", stale)
        executor.get_option_quote = AsyncMock(return_value={"ask": 3.00})
        ask = await executor._fetch_ask("SPY", 733.0, "2026-06-25", "put")
        assert ask == 3.00  # stale snapshot rejected, HTTP used

    @pytest.mark.asyncio
    async def test_redis_miss_falls_back_to_http(self, monkeypatch):
        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings())

        async def miss(ck):
            return None

        monkeypatch.setattr(redis_client, "get_option_snapshot", miss)
        executor.get_option_quote = AsyncMock(return_value={"ask": 3.00})
        ask = await executor._fetch_ask("NVDA", 195.0, "2026-06-26", "put")
        assert ask == 3.00

    @pytest.mark.asyncio
    async def test_live_quote_disabled_skips_redis(self, monkeypatch):
        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings(WEBULL_ENTRY_USE_LIVE_QUOTE=False))
        called = {"redis": False}

        async def snap(ck):
            called["redis"] = True
            return {"ask": 2.50}

        monkeypatch.setattr(redis_client, "get_option_snapshot", snap)
        executor.get_option_quote = AsyncMock(return_value={"ask": 3.00})
        ask = await executor._fetch_ask("SPY", 733.0, "2026-06-25", "put")
        assert ask == 3.00
        assert called["redis"] is False  # Redis path skipped when flag off


class TestEntryChaseAggression:
    """#1 + #2 — faster cadence + index 0DTE crosses harder on rung 1."""

    async def _capture_first_rung_limit(self, executor, ticker):
        captured = {}

        async def fake_submit(payload):
            captured["limit"] = float(payload[0]["limit_price"])
            return ("ORDER1", {}, None)

        executor._fetch_ask = AsyncMock(return_value=1.00)  # clean ask
        executor._submit_order_payload = fake_submit
        executor._wait_for_fill = AsyncMock(return_value="FILLED")  # fill on rung 1
        res = await executor._place_buy_with_escalation(
            ticker=ticker, strike=733.0, expiry_date="2026-06-25",
            option_type="put", contracts=1, initial_limit=1.05,
        )
        assert res.fill_status == "FILLED"
        return captured["limit"]

    @pytest.mark.asyncio
    async def test_index_crosses_harder_on_rung_1(self):
        executor = WebullExecutor(_chase_settings())
        limit = await self._capture_first_rung_limit(executor, "SPY")
        assert limit == 1.10  # ask 1.00 × (1 + 10% index aggress)

    @pytest.mark.asyncio
    async def test_standard_ticker_uses_base_aggress(self):
        executor = WebullExecutor(_chase_settings())
        limit = await self._capture_first_rung_limit(executor, "NVDA")
        assert limit == 1.05  # ask 1.00 × (1 + 5% base aggress)

    @pytest.mark.asyncio
    async def test_cadence_read_from_settings(self):
        """Per-rung timeout + poll come from settings (faster than the old 12s/3s)."""
        executor = WebullExecutor(_chase_settings(WEBULL_ENTRY_PER_ATTEMPT_SEC=4.0,
                                                  WEBULL_ENTRY_POLL_SEC=1.0))
        seen = {}

        async def fake_wait(coid, timeout_seconds, poll_interval):
            seen["timeout"] = timeout_seconds
            seen["poll"] = poll_interval
            return "FILLED"

        executor._fetch_ask = AsyncMock(return_value=1.00)
        executor._submit_order_payload = AsyncMock(return_value=("ORDER1", {}, None))
        executor._wait_for_fill = fake_wait
        await executor._place_buy_with_escalation(
            ticker="SPY", strike=733.0, expiry_date="2026-06-25",
            option_type="put", contracts=1, initial_limit=1.05,
        )
        assert seen["timeout"] == 4.0
        assert seen["poll"] == 1.0


class TestReconnectResetsDataClient:
    """B4 — _reconnect() must also drop the market-data client (it wraps the api client).

    If it doesn't, _ensure_data_client() (which early-returns when non-None) keeps serving a
    client built on the torn-down api client, silently breaking the live-quote entry chase.
    """

    def test_reconnect_nulls_data_client(self):
        executor = WebullExecutor(_chase_settings())
        executor._data_client = object()  # simulate a live market-data client
        executor._ensure_clients = lambda: None  # don't actually rebuild SDK clients
        executor._reconnect()
        assert executor._data_client is None


class TestFetchAskFreshnessGuard:
    """B5 — an unverifiable-age Redis snapshot (missing/zero timestamp) must NOT be trusted."""

    @pytest.mark.asyncio
    async def test_missing_timestamp_falls_back_to_http(self, monkeypatch):
        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings())

        async def no_ts(ck):
            return {"ask": 2.50}  # no 't' field -> age can't be verified

        monkeypatch.setattr(redis_client, "get_option_snapshot", no_ts)
        executor.get_option_quote = AsyncMock(return_value={"ask": 3.00})
        ask = await executor._fetch_ask("SPY", 733.0, "2026-06-25", "put")
        assert ask == 3.00  # freshness guard rejected the un-timestamped snapshot

    @pytest.mark.asyncio
    async def test_zero_timestamp_falls_back_to_http(self, monkeypatch):
        from options_owl.db import redis_client

        executor = WebullExecutor(_chase_settings())

        async def zero_ts(ck):
            return {"ask": 2.50, "t": 0}

        monkeypatch.setattr(redis_client, "get_option_snapshot", zero_ts)
        executor.get_option_quote = AsyncMock(return_value={"ask": 3.00})
        ask = await executor._fetch_ask("SPY", 733.0, "2026-06-25", "put")
        assert ask == 3.00


class TestEntryChaseValueCap:
    """L1 — the hard $5k order-value rail is re-applied to each escalated (repriced) rung."""

    @pytest.mark.asyncio
    async def test_value_cap_clamps_escalated_limit(self):
        from options_owl.execution.webull_executor import MAX_ORDER_VALUE

        executor = WebullExecutor(_chase_settings())
        captured = {}

        async def fake_submit(payload):
            captured["limit"] = float(payload[0]["limit_price"])
            return ("ORDER1", {}, None)

        executor._fetch_ask = AsyncMock(return_value=1.00)
        executor._submit_order_payload = fake_submit
        executor._wait_for_fill = AsyncMock(return_value="FILLED")
        # 60 contracts × ~$1.05 × 100 = ~$6,300 > $5,000 hard cap → limit must be clamped down
        res = await executor._place_buy_with_escalation(
            ticker="NVDA", strike=733.0, expiry_date="2026-06-25",
            option_type="put", contracts=60, initial_limit=1.05,
        )
        assert res.fill_status == "FILLED"
        assert captured["limit"] * 60 * 100 <= MAX_ORDER_VALUE
        assert captured["limit"] == 0.83  # 5000/(60*100)=0.833 rounded DOWN to a legal penny step


def _exit_settings(**overrides):
    """Settings with the fast-exit-chase knobs set to real numbers."""
    defaults = {
        "PAPER_TRADE": False,
        "WEBULL_KILL_SWITCH": False,
        "ENABLE_FAST_EXIT_CHASE": True,
        "WEBULL_ENTRY_USE_LIVE_QUOTE": True,
        "WEBULL_ENTRY_QUOTE_MAX_AGE_SEC": 20.0,
        "WEBULL_EXIT_FILL_ATTEMPTS": 4,
        "WEBULL_EXIT_PER_ATTEMPT_SEC": 2.5,
        "WEBULL_EXIT_POLL_SEC": 1.0,
        "WEBULL_EXIT_AGGRESS_PCT": 2.0,
        "WEBULL_EXIT_STEP_PCT": 6.0,
        "WEBULL_EXIT_MAX_DISCOUNT_PCT": 25.0,
    }
    defaults.update(overrides)
    return _make_settings(**defaults)


class TestExitChase:
    """Fast, tiered EXIT chase (ENABLE_FAST_EXIT_CHASE) — sell-side mirror of the entry
    chase, crossing DOWN toward the bid, with the SAME double-fill (naked-short) safety."""

    async def _run(self, executor, wait_returns, confirm_return="CANCELLED", contracts=2):
        limits = []

        async def fake_submit(payload):
            limits.append(float(payload[0]["limit_price"]))
            return (f"OID{len(limits)}", {}, None)

        state = {"i": 0}

        async def fake_wait(coid, timeout_seconds, poll_interval):
            i = state["i"]
            state["i"] += 1
            return wait_returns[i] if i < len(wait_returns) else "SUBMITTED"

        executor._fetch_bid = AsyncMock(return_value=1.00)  # clean, fresh bid
        executor._submit_order_payload = fake_submit
        executor._wait_for_fill = fake_wait
        executor._confirm_cancelled = AsyncMock(return_value=confirm_return)
        executor.cancel_order = AsyncMock(return_value=True)
        executor._get_filled_quantity = AsyncMock(return_value=1)
        res = await executor._place_sell_with_escalation(
            ticker="SPY", strike=733.0, expiry_date="2026-07-02",
            option_type="put", contracts=contracts, initial_limit=1.00,
        )
        return res, limits

    @pytest.mark.asyncio
    async def test_fills_rung1_just_below_bid(self):
        executor = WebullExecutor(_exit_settings())
        res, limits = await self._run(executor, ["FILLED"])
        assert res.fill_status == "FILLED"
        assert limits == [0.98]  # bid 1.00 × (1 - 2% aggress)

    @pytest.mark.asyncio
    async def test_crosses_harder_when_not_filled(self):
        executor = WebullExecutor(_exit_settings())
        res, limits = await self._run(executor, ["SUBMITTED", "SUBMITTED", "FILLED"])
        assert res.fill_status == "FILLED"
        assert limits == [0.98, 0.92, 0.86]  # 2%, 8%, 14% below bid — progressively marketable

    @pytest.mark.asyncio
    async def test_floor_caps_the_discount(self):
        executor = WebullExecutor(_exit_settings(
            WEBULL_EXIT_MAX_DISCOUNT_PCT=10.0, WEBULL_EXIT_FILL_ATTEMPTS=6))
        res, limits = await self._run(executor, ["SUBMITTED"] * 6)
        assert min(limits) >= 0.90 - 1e-9   # never more than 10% below the bid
        assert 0.90 in limits

    @pytest.mark.asyncio
    async def test_honors_fill_during_cancel_and_stops(self):
        """SAFETY: if a rung fills in the race with our cancel, honor it and DO NOT submit
        another sell on top (that would oversell / go naked short)."""
        executor = WebullExecutor(_exit_settings())
        res, limits = await self._run(executor, ["SUBMITTED"], confirm_return="FILLED")
        assert res.fill_status == "FILLED"
        assert len(limits) == 1  # exactly one submit — no second sell after the surprise fill

    @pytest.mark.asyncio
    async def test_aborts_when_cancel_unconfirmed(self):
        """SAFETY: if we can't confirm the prior sell is dead, ABORT — never leave two live
        sell orders working."""
        executor = WebullExecutor(_exit_settings())
        res, limits = await self._run(executor, ["SUBMITTED"], confirm_return="WORKING")
        assert res.success is False
        assert "oversell" in res.error.lower()
        assert len(limits) == 1  # never re-submitted

    @pytest.mark.asyncio
    async def test_partial_fill_returns_partial_and_stops(self):
        executor = WebullExecutor(_exit_settings())
        res, limits = await self._run(executor, ["PARTIAL"])
        assert res.success is True
        assert res.fill_status == "PARTIAL"
        assert len(limits) == 1  # remainder left for the monitor's next cycle
        executor.cancel_order.assert_awaited()  # unfilled remainder cancelled

    @pytest.mark.asyncio
    async def test_limits_are_legal_steps_above_3_dollars(self):
        executor = WebullExecutor(_exit_settings())
        executor._fetch_bid = AsyncMock(return_value=4.00)  # >$3 → nickel increments
        limits = []

        async def fake_submit(payload):
            limits.append(float(payload[0]["limit_price"]))
            return (f"OID{len(limits)}", {}, None)

        state = {"i": 0}

        async def fake_wait(coid, timeout_seconds, poll_interval):
            state["i"] += 1
            return "SUBMITTED"

        executor._submit_order_payload = fake_submit
        executor._wait_for_fill = fake_wait
        executor._confirm_cancelled = AsyncMock(return_value="CANCELLED")
        await executor._place_sell_with_escalation(
            ticker="SPY", strike=733.0, expiry_date="2026-07-02",
            option_type="put", contracts=2, initial_limit=4.00,
        )
        for lim in limits:
            assert abs((lim / 0.05) - round(lim / 0.05)) < 1e-9, f"{lim} not a nickel step"

    def test_place_option_order_routes_sell_through_chase(self):
        """The SELL routing is gated on ENABLE_FAST_EXIT_CHASE and calls the chase."""
        import inspect

        src = inspect.getsource(WebullExecutor.place_option_order)
        assert "ENABLE_FAST_EXIT_CHASE" in src
        assert "_place_sell_with_escalation" in src


class TestFetchBidVenueFirst:
    """_fetch_bid (fast-exit chase) must prefer Webull's own venue quote — the MU #399
    lesson: Polygon/Redis quotes can be stale/wide for thin contracts."""

    @pytest.mark.asyncio
    async def test_prefers_webull_venue_bid(self):
        executor = WebullExecutor(_exit_settings())
        executor.get_option_quote = AsyncMock(return_value={"bid": 5.25, "ask": 5.60, "mid": 5.42})
        bid = await executor._fetch_bid("MU", 985.0, "2026-07-02", "put")
        assert bid == 5.25
        executor.get_option_quote.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_venue_failure_returns_none_gracefully(self):
        executor = WebullExecutor(_exit_settings(WEBULL_ENTRY_USE_LIVE_QUOTE=False))
        executor.get_option_quote = AsyncMock(side_effect=RuntimeError("socket dead"))
        bid = await executor._fetch_bid("MU", 985.0, "2026-07-02", "put")
        assert bid is None  # no venue, redis disabled → None, never raises
