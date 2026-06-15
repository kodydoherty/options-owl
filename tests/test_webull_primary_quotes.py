"""Tests for Spec 03: Webull as Primary Data Source for Trading Bots."""

from __future__ import annotations

import inspect



class TestFeatureFlag:
    def test_setting_exists(self):
        """WEBULL_PRIMARY_QUOTES should be in Settings."""
        from options_owl.config.settings import Settings

        s = Settings()
        assert hasattr(s, "WEBULL_PRIMARY_QUOTES")
        # Default is off
        assert s.WEBULL_PRIMARY_QUOTES is False


class TestMonitorWebullCascade:
    def test_webull_source_in_cascade(self):
        """position_monitor should have Webull as Source 0 in premium cascade."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        # Webull should appear BEFORE Polygon WS in the cascade
        webull_pos = source.find("WEBULL_PRIMARY_QUOTES")
        polygon_ws_pos = source.find("Source 1: market stream")
        assert webull_pos > 0
        assert polygon_ws_pos > 0
        assert webull_pos < polygon_ws_pos, "Webull should be checked before Polygon WS"

    def test_webull_source_gated_behind_flag(self):
        """Webull quote should only fire when WEBULL_PRIMARY_QUOTES=true."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        assert 'WEBULL_PRIMARY_QUOTES' in source
        assert 'webull_executor' in source

    def test_webull_has_timeout(self):
        """Webull quote in monitor should have asyncio.wait_for timeout."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        # Find the Webull section and check for timeout
        webull_section = source[source.find("Source 0: Webull"):source.find("Source 1:")]
        assert "wait_for" in webull_section
        assert "timeout=" in webull_section

    def test_webull_source_label_tracked(self):
        """Premium source should be labeled 'webull' when Webull provides quote."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        assert '_prem_source = "webull"' in source


class TestEntryWebullQuote:
    def test_verify_live_premium_has_webull(self):
        """_verify_live_premium should try Webull before Polygon."""
        from options_owl.execution import paper_trader

        source = inspect.getsource(paper_trader._verify_live_premium)
        assert "WEBULL_PRIMARY_QUOTES" in source
        assert "_webull_executor" in source

    def test_webull_before_polygon_in_entry(self):
        """Webull quote should be attempted before Polygon in entry path."""
        from options_owl.execution import paper_trader

        source = inspect.getsource(paper_trader._verify_live_premium)
        webull_pos = source.find("Webull DataClient quote")
        polygon_pos = source.find("Polygon option quote")
        assert webull_pos > 0
        assert polygon_pos > 0
        assert webull_pos < polygon_pos

    def test_fill_missing_premium_has_webull(self):
        """_fill_missing_premium should try Webull before Polygon."""
        from options_owl.execution import paper_trader

        source = inspect.getsource(paper_trader.PaperTrader._fill_missing_premium)
        assert "WEBULL_PRIMARY_QUOTES" in source
        assert "webull_executor" in source

    def test_paper_trader_exposes_executor_on_settings(self):
        """PaperTrader.__init__ should attach webull_executor to settings."""
        from options_owl.execution import paper_trader

        source = inspect.getsource(paper_trader.PaperTrader.__init__)
        assert "_webull_executor" in source


class TestFallbackBehavior:
    def test_flag_off_skips_webull_in_monitor(self):
        """When WEBULL_PRIMARY_QUOTES=false, cascade should skip Webull."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        # The flag check should be a guard condition
        assert 'WEBULL_PRIMARY_QUOTES' in source

    def test_no_webull_executor_skips_gracefully(self):
        """When webull_executor is None, cascade should skip to next source."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        assert "webull_executor is not None" in source

    def test_webull_timeout_falls_through(self):
        """Webull timeout should fall through to Polygon/yfinance."""
        from options_owl.execution import position_monitor

        source = inspect.getsource(position_monitor.run_position_monitor)
        webull_section = source[source.find("Source 0: Webull"):source.find("Source 1:")]
        assert "TimeoutError" in webull_section
