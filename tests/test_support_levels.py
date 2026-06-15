"""Tests for multi-timeframe support level detection."""

from dataclasses import dataclass


@dataclass(slots=True)
class FakeBar:
    timestamp: float = 0
    open: float = 100.0
    high: float = 101.0
    low: float = 99.0
    close: float = 100.5
    volume: float = 1000
    vwap: float = 100.2


class TestClusterLows:
    """Test the wick clustering algorithm."""

    def test_single_low(self):
        from options_owl.collectors.support_levels import _cluster_lows
        result = _cluster_lows([100.0])
        assert len(result) == 1
        assert result[0] == (100.0, 1)

    def test_cluster_nearby_lows(self):
        """Lows within 0.15% should cluster together."""
        from options_owl.collectors.support_levels import _cluster_lows
        # 100.00, 100.10, 100.05 are all within 0.15% of each other
        result = _cluster_lows([100.00, 100.10, 100.05])
        assert len(result) == 1
        assert result[0][1] == 3  # 3 touches

    def test_separate_clusters(self):
        """Lows far apart should form separate clusters."""
        from options_owl.collectors.support_levels import _cluster_lows
        # Two distinct levels: ~100 and ~102
        lows = [100.0, 100.05, 100.10, 102.0, 102.05, 102.10]
        result = _cluster_lows(lows)
        assert len(result) == 2
        # Both should have 3 touches
        assert result[0][1] == 3
        assert result[1][1] == 3

    def test_empty_lows(self):
        from options_owl.collectors.support_levels import _cluster_lows
        assert _cluster_lows([]) == []

    def test_strongest_cluster_first(self):
        """Clusters sorted by touch count descending."""
        from options_owl.collectors.support_levels import _cluster_lows
        lows = [100.0, 100.05, 100.10, 100.08, 102.0, 102.05]
        result = _cluster_lows(lows)
        assert result[0][1] >= result[-1][1]  # first has more touches


class TestFindSupportLevels:
    """Test the full support detection across timeframes."""

    def _make_bars(self, lows: list[float], base_price: float = 605.0) -> list[FakeBar]:
        """Create fake candle bars with specified lows."""
        bars = []
        for i, low in enumerate(lows):
            bars.append(FakeBar(
                timestamp=i * 300_000,
                open=base_price,
                high=base_price + 1.0,
                low=low,
                close=base_price - 0.5,
                volume=10000,
                vwap=base_price,
            ))
        return bars

    def test_single_timeframe_support(self):
        from options_owl.collectors.support_levels import find_support_levels

        # 5m bars with repeated lows near $604.80
        bars_5m = self._make_bars([604.80, 604.75, 604.82, 604.78, 605.50, 605.60])
        candle_data = {"5m": bars_5m, "15m": [], "1h": [], "4h": []}

        levels = find_support_levels(candle_data, current_price=606.0)
        assert len(levels) >= 1
        # Strongest support should be near $604.80
        assert abs(levels[0].price - 604.80) < 0.5
        assert levels[0].strength >= 3

    def test_multi_timeframe_confluence(self):
        from options_owl.collectors.support_levels import find_support_levels

        # Support at ~$604.80 on both 5m and 1h
        bars_5m = self._make_bars([604.80, 604.75, 604.82, 604.78, 605.50, 605.60])
        bars_1h = self._make_bars([604.70, 604.85, 604.90, 606.0, 606.5] * 2)

        candle_data = {"5m": bars_5m, "15m": [], "1h": bars_1h, "4h": []}

        levels = find_support_levels(candle_data, current_price=606.0)
        assert len(levels) >= 1
        # Should have confluence >= 2 (5m + 1h)
        best = levels[0]
        assert best.confluence >= 2
        assert "5m" in best.timeframes
        assert "1h" in best.timeframes

    def test_no_support_above_price(self):
        """Levels above current price are resistance, not support."""
        from options_owl.collectors.support_levels import find_support_levels

        bars_5m = self._make_bars([607.0, 607.05, 607.10, 607.08])
        candle_data = {"5m": bars_5m, "15m": [], "1h": [], "4h": []}

        levels = find_support_levels(candle_data, current_price=605.0)
        assert len(levels) == 0

    def test_no_data_returns_empty(self):
        from options_owl.collectors.support_levels import find_support_levels
        levels = find_support_levels({}, current_price=100.0)
        assert levels == []


class TestIsAtSupport:
    """Test the high-level at_support check."""

    def _make_bars(self, lows: list[float], base_price: float = 605.0) -> list[FakeBar]:
        bars = []
        for i, low in enumerate(lows):
            bars.append(FakeBar(
                timestamp=i * 300_000,
                open=base_price,
                high=base_price + 1.0,
                low=low,
                close=base_price - 0.5,
                volume=10000,
                vwap=base_price,
            ))
        return bars

    def test_at_support_true(self):
        from options_owl.collectors.support_levels import is_at_support

        # Strong support at ~604.80, price at 605.0 (0.03% away)
        bars_5m = self._make_bars([604.80, 604.75, 604.82, 604.78, 605.50, 605.60])
        bars_15m = self._make_bars([604.70, 604.85, 604.90, 605.00, 605.50])
        candle_data = {"5m": bars_5m, "15m": bars_15m, "1h": [], "4h": []}

        result, detail = is_at_support(candle_data, current_price=605.0)
        assert result is True
        assert "support=" in detail

    def test_not_at_support_too_far(self):
        from options_owl.collectors.support_levels import is_at_support

        # Support at ~600, but price is at 610 (1.6% away)
        bars_5m = self._make_bars([600.0, 600.05, 600.10, 600.08, 605.0, 605.0],
                                   base_price=610.0)
        candle_data = {"5m": bars_5m, "15m": [], "1h": [], "4h": []}

        result, detail = is_at_support(candle_data, current_price=610.0)
        assert result is False

    def test_no_data(self):
        from options_owl.collectors.support_levels import is_at_support
        result, detail = is_at_support({}, current_price=100.0)
        assert result is False
        assert "no support" in detail
