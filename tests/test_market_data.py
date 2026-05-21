from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from options_owl.collectors.market_data import fetch_intraday_sync


class TestFetchIntraday:
    @patch("options_owl.collectors.market_data.yf.Ticker")
    def test_returns_price_bars(self, mock_ticker_cls):
        """Test that yfinance DataFrame is converted to PriceBar list."""
        index = pd.date_range("2026-03-27 09:30", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "Open": [170.0, 170.5, 170.2],
                "High": [170.8, 171.0, 170.5],
                "Low": [169.5, 170.0, 169.8],
                "Close": [170.5, 170.2, 170.0],
                "Volume": [10000, 12000, 8000],
            },
            index=index,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_cls.return_value = mock_ticker

        bars = fetch_intraday_sync("NVDA", "2026-03-27")
        assert len(bars) == 3
        assert bars[0].open == 170.0
        assert bars[0].high == 170.8
        assert bars[0].volume == 10000
        assert bars[2].close == 170.0

    @patch("options_owl.collectors.market_data.yf.Ticker")
    def test_empty_dataframe_returns_empty(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        bars = fetch_intraday_sync("FAKE", "2026-03-27")
        assert bars == []

    @patch("options_owl.collectors.market_data.yf.Ticker")
    def test_exception_returns_empty(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("API error")

        bars = fetch_intraday_sync("FAIL", "2026-03-27")
        assert bars == []


class TestFetchIntradayAsync:
    @pytest.mark.asyncio
    @patch("options_owl.collectors.market_data.yf.Ticker")
    async def test_async_wrapper(self, mock_ticker_cls):
        from options_owl.collectors.market_data import fetch_intraday

        index = pd.date_range("2026-03-27 09:30", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                "Open": [170.0, 170.5],
                "High": [170.8, 171.0],
                "Low": [169.5, 170.0],
                "Close": [170.5, 170.2],
                "Volume": [10000, 12000],
            },
            index=index,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_cls.return_value = mock_ticker

        bars = await fetch_intraday("NVDA", "2026-03-27")
        assert len(bars) == 2
