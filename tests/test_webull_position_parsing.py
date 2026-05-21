"""Tests for Webull position parsing — ensures we never lose track of positions.

The Webull API returns positions with option details nested in 'legs' arrays,
not at the top level. This test ensures our parser handles this correctly.

Bug discovered 2026-05-12: get_open_option_positions() returned empty list
because it looked for option_type/expiry at top level, but they're in legs[0].
This caused the reconciler to never detect orphaned Webull positions.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from options_owl.execution.webull_executor import WebullExecutor


# Real Webull API response format (captured from production logs)
WEBULL_POSITIONS_RESPONSE = [
    {
        "currency": "USD",
        "quantity": "3",
        "cost": "531.00",
        "proportion": "0.6546",
        "legs": [
            {
                "symbol": "SPY",
                "cost": "1.77",
                "proportion": "0.6546",
                "leg_id": "6UI8SMND307JDBTMKF80UFAIV9",
                "instrument_type": "OPTION",
                "last_price": "1.965",
                "unrealized_profit_loss": "58.50",
                "day_profit_loss": "58.35",
                "day_realized_profit_loss": "-0.15",
                "option_type": "PUT",
                "option_expire_date": "2026-05-12",
                "option_exercise_price": "735.00",
                "option_contract_multiplier": "100",
                "option_contract_deliverable": "100",
                "expiration_type": "PM",
            }
        ],
        "position_id": "6UI8SMND307JDBTMKF80UFAIV9",
        "symbol": "SPY",
        "option_strategy": "SINGLE",
        "instrument_type": "OPTION",
        "cost_price": "1.77",
        "last_price": "1.965",
        "market_value": "589.50",
        "unrealized_profit_loss": "58.50",
        "unrealized_profit_loss_rate": "0.1102",
        "day_profit_loss": "58.35",
        "day_realized_profit_loss": "-0.15",
    },
    {
        "currency": "USD",
        "quantity": "2",
        "cost": "734.00",
        "proportion": "0.7479",
        "legs": [
            {
                "symbol": "GOOGL",
                "cost": "3.67",
                "proportion": "0.7479",
                "leg_id": "OQJAU0DGKNSP7Q106DUEL7LK69",
                "instrument_type": "OPTION",
                "last_price": "4.925",
                "unrealized_profit_loss": "251.00",
                "day_profit_loss": "250.90",
                "day_realized_profit_loss": "-0.10",
                "option_type": "PUT",
                "option_expire_date": "2026-05-13",
                "option_exercise_price": "387.50",
                "option_contract_multiplier": "100",
                "option_contract_deliverable": "100",
                "expiration_type": "PM",
            }
        ],
        "position_id": "OQJAU0DGKNSP7Q106DUEL7LK69",
        "symbol": "GOOGL",
        "option_strategy": "SINGLE",
        "instrument_type": "OPTION",
        "cost_price": "3.67",
        "last_price": "4.93",
        "market_value": "985.00",
        "unrealized_profit_loss": "251.00",
        "unrealized_profit_loss_rate": "0.3420",
        "day_profit_loss": "250.90",
        "day_realized_profit_loss": "-0.10",
    },
    {
        "currency": "USD",
        "quantity": "2",
        "cost": "360.00",
        "proportion": "0.2521",
        "legs": [
            {
                "symbol": "AAPL",
                "cost": "1.80",
                "proportion": "0.2521",
                "leg_id": "GJH0I4SV2LMJDFAL1572CHU668",
                "instrument_type": "OPTION",
                "last_price": "1.66",
                "unrealized_profit_loss": "-28.00",
                "day_profit_loss": "-28.10",
                "day_realized_profit_loss": "-0.10",
                "option_type": "CALL",
                "option_expire_date": "2026-05-13",
                "option_exercise_price": "295.00",
                "option_contract_multiplier": "100",
                "option_contract_deliverable": "100",
                "expiration_type": "PM",
            }
        ],
        "position_id": "GJH0I4SV2LMJDFAL1572CHU668",
        "symbol": "AAPL",
        "option_strategy": "SINGLE",
        "instrument_type": "OPTION",
        "cost_price": "1.80",
        "last_price": "1.66",
        "market_value": "332.00",
        "unrealized_profit_loss": "-28.00",
        "unrealized_profit_loss_rate": "-0.0778",
        "day_profit_loss": "-28.10",
        "day_realized_profit_loss": "-0.10",
    },
]


@pytest.fixture
def executor():
    settings = MagicMock()
    settings.WEBULL_APP_KEY = "test"
    settings.WEBULL_APP_SECRET = "test"
    settings.WEBULL_ACCOUNT_ID = "test_account"
    settings.PAPER_TRADE = False
    w = WebullExecutor(settings)
    w._api_client = MagicMock()
    w._trade_client = MagicMock()
    w._account_id = "test_account"
    return w


@pytest.mark.asyncio
async def test_parse_nested_legs_positions(executor):
    """Positions with option details in nested legs must be parsed correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = WEBULL_POSITIONS_RESPONSE
    executor._trade_client.account_v2.get_account_position = MagicMock(
        return_value=mock_response
    )

    positions = await executor.get_open_option_positions()

    assert len(positions) == 3, f"Expected 3 positions, got {len(positions)}"

    spy = next(p for p in positions if p["ticker"] == "SPY")
    assert spy["strike"] == 735.0
    assert spy["option_type"] == "put"
    assert spy["expiry_date"] == "2026-05-12"
    assert spy["quantity"] == 3
    assert spy["position_id"] == "6UI8SMND307JDBTMKF80UFAIV9"

    googl = next(p for p in positions if p["ticker"] == "GOOGL")
    assert googl["strike"] == 387.5
    assert googl["option_type"] == "put"
    assert googl["expiry_date"] == "2026-05-13"
    assert googl["quantity"] == 2

    aapl = next(p for p in positions if p["ticker"] == "AAPL")
    assert aapl["strike"] == 295.0
    assert aapl["option_type"] == "call"
    assert aapl["expiry_date"] == "2026-05-13"
    assert aapl["quantity"] == 2


@pytest.mark.asyncio
async def test_empty_positions_returns_empty(executor):
    """Empty position list should return empty, not crash."""
    mock_response = MagicMock()
    mock_response.json.return_value = []
    executor._trade_client.account_v2.get_account_position = MagicMock(
        return_value=mock_response
    )

    positions = await executor.get_open_option_positions()
    assert positions == []


@pytest.mark.asyncio
async def test_position_without_legs_still_parsed(executor):
    """Position with option details at top level (hypothetical flat format)."""
    flat_response = [
        {
            "quantity": "5",
            "ticker": "NVDA",
            "option_type": "CALL",
            "option_expire_date": "2026-05-14",
            "option_exercise_price": "220.00",
            "position_id": "abc123",
        }
    ]
    mock_response = MagicMock()
    mock_response.json.return_value = flat_response
    executor._trade_client.account_v2.get_account_position = MagicMock(
        return_value=mock_response
    )

    positions = await executor.get_open_option_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "NVDA"
    assert positions[0]["strike"] == 220.0
    assert positions[0]["quantity"] == 5


@pytest.mark.asyncio
async def test_find_position_id_with_nested_legs(executor):
    """_find_position_id must find positions even when data is in nested legs."""
    mock_response = MagicMock()
    mock_response.json.return_value = WEBULL_POSITIONS_RESPONSE
    executor._trade_client.account_v2.get_account_position = MagicMock(
        return_value=mock_response
    )

    # Should find GOOGL position
    pid = await executor._find_position_id("GOOGL", 387.5, "2026-05-13", "PUT")
    assert pid == "OQJAU0DGKNSP7Q106DUEL7LK69"

    # Should find SPY position
    pid = await executor._find_position_id("SPY", 735.0, "2026-05-12", "PUT")
    assert pid == "6UI8SMND307JDBTMKF80UFAIV9"

    # Should find AAPL position
    pid = await executor._find_position_id("AAPL", 295.0, "2026-05-13", "CALL")
    assert pid == "GJH0I4SV2LMJDFAL1572CHU668"

    # Should NOT find non-existent position
    pid = await executor._find_position_id("MSFT", 400.0, "2026-05-12", "PUT")
    assert pid is None


@pytest.mark.asyncio
async def test_zero_quantity_positions_skipped(executor):
    """Positions with quantity 0 should be excluded."""
    response = [
        {
            "quantity": "0",
            "symbol": "EXPIRED",
            "legs": [
                {
                    "symbol": "EXPIRED",
                    "option_type": "CALL",
                    "option_expire_date": "2026-05-11",
                    "option_exercise_price": "100.00",
                }
            ],
            "position_id": "expired123",
        }
    ]
    mock_response = MagicMock()
    mock_response.json.return_value = response
    executor._trade_client.account_v2.get_account_position = MagicMock(
        return_value=mock_response
    )

    positions = await executor.get_open_option_positions()
    assert len(positions) == 0
