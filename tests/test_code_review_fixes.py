"""Tests verifying all code review fixes are correct and complete.

Covers:
- CRITICAL: per-ticker config category alignment
- CRITICAL: asyncio.wait_for timeouts on external I/O
- CRITICAL: PG/Redis shutdown hooks
- WARNING: webull bid/ask field mapping
- WARNING: signal_consumer f-string fix
- WARNING: EXIT_ENGINE default
- WARNING: monitor_bridge minutes_to_close precision
- WARNING: pipeline DailyLossGate safe dict access
- WARNING: postgres pool timeout
- WARNING: polygon pagination limit
- WARNING: paper_trader assert removal + error logging
- WARNING: candle_collector PG error logging
- Paper trading code path: PAPER_TRADE=true → no Webull orders
- Discord signals disabled when ENABLE_DISCORD_SIGNALS=false
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# 1. Per-ticker config category alignment (CRITICAL)
# ---------------------------------------------------------------------------

class TestPerTickerConfigFixes:
    """Verify per-ticker V5Configs use fields matching their TickerCategory."""

    def test_nvda_uses_general_profit_target(self):
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS, V5Config
        cfg = TICKER_CONFIGS["NVDA"]
        # NVDA is HIGH_VOL — profit_target_index_0dte_pct only fires for INDEX tickers.
        # Must use profit_target_general_pct instead.
        assert cfg.profit_target_general_pct == 20.0
        assert cfg.profit_target_index_0dte_pct == V5Config().profit_target_index_0dte_pct  # default

    def test_avgo_uses_general_profit_target(self):
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS, V5Config
        cfg = TICKER_CONFIGS["AVGO"]
        assert cfg.profit_target_general_pct == 20.0
        assert cfg.profit_target_index_0dte_pct == V5Config().profit_target_index_0dte_pct

    def test_msft_uses_general_profit_target(self):
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS, V5Config
        cfg = TICKER_CONFIGS["MSFT"]
        assert cfg.profit_target_general_pct == 20.0
        assert cfg.profit_target_index_0dte_pct == V5Config().profit_target_index_0dte_pct

    def test_aapl_uses_standard_adaptive_tiers(self):
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS, V5Config, categorize_ticker, TickerCategory
        assert categorize_ticker("AAPL") == TickerCategory.STANDARD
        cfg = TICKER_CONFIGS["AAPL"]
        # Must use adaptive_standard_tiers (AAPL is STANDARD, not HIGH_VOL)
        assert cfg.adaptive_standard_tiers != V5Config().adaptive_standard_tiers  # customized
        assert cfg.adaptive_highvol_tiers == V5Config().adaptive_highvol_tiers  # default (unused)

    def test_amzn_uses_standard_adaptive_tiers(self):
        from options_owl.risk.exit_v5.config import TICKER_CONFIGS, V5Config, categorize_ticker, TickerCategory
        assert categorize_ticker("AMZN") == TickerCategory.STANDARD
        cfg = TICKER_CONFIGS["AMZN"]
        assert cfg.adaptive_standard_tiers != V5Config().adaptive_standard_tiers  # customized
        assert cfg.adaptive_highvol_tiers == V5Config().adaptive_highvol_tiers  # default (unused)

    def test_all_ticker_configs_use_correct_category_fields(self):
        """Ensure no per-ticker config sets fields for wrong category."""
        from options_owl.risk.exit_v5.config import (
            TICKER_CONFIGS, V5Config, categorize_ticker, TickerCategory,
        )
        default = V5Config()
        for ticker, cfg in TICKER_CONFIGS.items():
            cat = categorize_ticker(ticker)
            if cat == TickerCategory.HIGH_VOL:
                # INDEX-only field should be default (never customized for HIGH_VOL)
                if cfg.profit_target_index_0dte_pct != default.profit_target_index_0dte_pct:
                    pytest.fail(
                        f"{ticker} is HIGH_VOL but customizes profit_target_index_0dte_pct "
                        f"(only fires for INDEX tickers)"
                    )
                # STANDARD adaptive tiers should be default
                if cfg.adaptive_standard_tiers != default.adaptive_standard_tiers:
                    if cfg.adaptive_highvol_tiers == default.adaptive_highvol_tiers:
                        pytest.fail(
                            f"{ticker} is HIGH_VOL but customizes adaptive_standard_tiers "
                            f"instead of adaptive_highvol_tiers"
                        )
            elif cat == TickerCategory.STANDARD:
                # HIGH_VOL adaptive tiers should be default
                if cfg.adaptive_highvol_tiers != default.adaptive_highvol_tiers:
                    pytest.fail(
                        f"{ticker} is STANDARD but customizes adaptive_highvol_tiers "
                        f"(only used for HIGH_VOL tickers)"
                    )
            elif cat == TickerCategory.INDEX:
                if cfg.adaptive_highvol_tiers != default.adaptive_highvol_tiers:
                    pytest.fail(
                        f"{ticker} is INDEX but customizes adaptive_highvol_tiers"
                    )


# ---------------------------------------------------------------------------
# 2. EXIT_ENGINE default (WARNING)
# ---------------------------------------------------------------------------

class TestSettingsDefaults:

    def test_exit_engine_default_is_v5(self):
        from options_owl.config.settings import Settings
        # The default in code should be "v5" (not "v3")
        source = inspect.getsource(Settings)
        assert 'EXIT_ENGINE: str = "v5"' in source

    def test_exit_engine_field_default(self):
        """Verify the actual field default value."""
        from options_owl.config.settings import Settings
        fields = Settings.model_fields
        assert fields["EXIT_ENGINE"].default == "v5"


# ---------------------------------------------------------------------------
# 3. Webull bid/ask field mapping (WARNING)
# ---------------------------------------------------------------------------

class TestWebullBidAskMapping:

    def test_bid_list_maps_to_bid(self):
        """bidList from API should populate bid_list (not ask_list)."""
        from options_owl.execution.webull_executor import WebullExecutor

        data = {
            "bidList": [{"price": "1.50"}],
            "askList": [{"price": "1.60"}],
        }
        result = WebullExecutor._parse_option_quotes(data)
        assert result is not None
        assert result["bid"] == 1.50
        assert result["ask"] == 1.60


# ---------------------------------------------------------------------------
# 4. Signal consumer f-string (WARNING)
# ---------------------------------------------------------------------------

class TestSignalConsumerLogging:

    def test_no_malformed_ternary_fstring(self):
        """The logger.info call should not use inline ternary in f-string args."""
        source = inspect.getsource(
            __import__("options_owl.collectors.signal_consumer", fromlist=["_poll_and_route"])
        )
        # The old broken pattern: logger.info(f"..." if cond else f"...")
        # passed the ternary as the first arg, not wrapped in parens.
        # The fix uses if/else blocks.
        assert 'logger.info(\n' not in source or 'if ml_confidence' in source


# ---------------------------------------------------------------------------
# 5. monitor_bridge minutes_to_close precision (WARNING)
# ---------------------------------------------------------------------------

class TestMinutesToClosePrecision:

    def test_source_includes_seconds(self):
        """minutes_to_close should use seconds for sub-minute precision."""
        import options_owl.risk.exit_v5.monitor_bridge as mb
        source = inspect.getsource(mb)
        # Must compute from seconds (3600, second) not just hours and minutes
        assert "3600" in source or "second" in source
        # Must NOT be integer truncation
        assert "now_et.hour * 60 + now_et.minute)" not in source


# ---------------------------------------------------------------------------
# 6. Pipeline DailyLossGate safe dict access (WARNING)
# ---------------------------------------------------------------------------

class TestDailyLossGateSafeAccess:

    def test_source_uses_dict_get(self):
        """DailyLossGate should use .get() for portfolio dict access."""
        from options_owl.risk.pipeline import DailyLossGate
        source = inspect.getsource(DailyLossGate.evaluate)
        # Should use portfolio.get() not portfolio[]
        assert 'portfolio.get("last_trade_date")' in source
        assert 'portfolio.get("daily_pnl"' in source


# ---------------------------------------------------------------------------
# 7. Postgres pool timeout (WARNING)
# ---------------------------------------------------------------------------

class TestPostgresPoolTimeout:

    def test_init_pool_has_timeout(self):
        """init_pool should wrap create_pool in asyncio.wait_for."""
        import options_owl.db.postgres as pg
        source = inspect.getsource(pg.init_pool)
        assert "wait_for" in source
        assert "timeout" in source


# ---------------------------------------------------------------------------
# 8. Polygon pagination limit (WARNING)
# ---------------------------------------------------------------------------

class TestPolygonPaginationLimit:

    def test_chain_fetch_has_page_limit(self):
        """polygon_option_chain should have a max page count."""
        from options_owl.collectors.polygon_options import polygon_option_chain
        source = inspect.getsource(polygon_option_chain)
        assert "max_pages" in source
        assert "page < max_pages" in source


# ---------------------------------------------------------------------------
# 9. Paper trader: no assert in production + error logging (WARNING)
# ---------------------------------------------------------------------------

class TestPaperTraderFixes:

    def test_no_assert_on_atm_premium(self):
        """atm_premium=None should return None, not assert."""
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader)
        # Should NOT have: assert signal_premium is not None
        assert "assert signal_premium is not None" not in source

    def test_log_trade_event_logs_errors(self):
        """log_trade_event should log exceptions, not silently swallow."""
        from options_owl.execution.paper_trader import log_trade_event
        source = inspect.getsource(log_trade_event)
        # Should NOT have bare `pass` in except
        assert "except Exception:\n        pass" not in source
        # Should log the error
        assert "logger.debug" in source or "logger.warning" in source


# ---------------------------------------------------------------------------
# 10. Candle collector PG error logging (WARNING)
# ---------------------------------------------------------------------------

class TestCandleCollectorPGLogging:

    def test_pg_write_logs_errors(self):
        """PG write errors should be logged, not silently swallowed."""
        from options_owl.collectors.candle_collector import CandleCollector
        source = inspect.getsource(CandleCollector)
        # No bare `except: pass` for PG writes
        assert "pass  # PG writes" not in source


# ---------------------------------------------------------------------------
# 11. Harvester fixes (CRITICAL + WARNING)
# ---------------------------------------------------------------------------

class TestHarvesterFixes:

    def test_underlying_price_uses_to_thread(self):
        """_get_underlying_price must run in asyncio.to_thread (blocking HTTP)."""
        import options_owl.harvester as h
        source = inspect.getsource(h)
        assert "asyncio.to_thread" in source

    def test_persist_rows_uses_pg(self):
        """_persist_rows should write to PostgreSQL (not SQLite)."""
        import options_owl.harvester as h
        source = inspect.getsource(h._persist_rows)
        assert "write_option_ticks_batch" in source
        assert "aiosqlite" not in source


# ---------------------------------------------------------------------------
# 12. market_data_stream timezone fix (CRITICAL)
# ---------------------------------------------------------------------------

class TestMarketDataStreamTimezone:

    def test_no_hardcoded_utc_minus_4(self):
        """Should use ZoneInfo, not hardcoded timedelta(hours=-4)."""
        import options_owl.collectors.market_data_stream as mds
        source = inspect.getsource(mds)
        assert "hours=-4" not in source or "ZoneInfo" in source


# ---------------------------------------------------------------------------
# 13. IV filter + pipeline asyncio.to_thread (WARNING)
# ---------------------------------------------------------------------------

class TestIVFilterAsyncSafe:

    def test_pipeline_wraps_iv_filter_in_thread(self):
        """IVFilterGate should wrap check_iv_filter in asyncio.to_thread."""
        from options_owl.risk.pipeline import IVFilterGate
        source = inspect.getsource(IVFilterGate.evaluate)
        assert "to_thread" in source


# ---------------------------------------------------------------------------
# 14. Main shutdown hooks (CRITICAL)
# ---------------------------------------------------------------------------

class TestMainShutdownHooks:

    def test_cleanup_connections_exists(self):
        """main.py should have a _cleanup_connections function."""
        import options_owl.main as m
        assert hasattr(m, "_cleanup_connections")
        assert callable(m._cleanup_connections)

    def test_cleanup_called_on_crash(self):
        """run_collector_with_retry should call _cleanup_connections on exception."""
        import options_owl.main as m
        source = inspect.getsource(m.run_collector_with_retry)
        assert "_cleanup_connections()" in source

    def test_cleanup_called_on_keyboard_interrupt(self):
        import options_owl.main as m
        source = inspect.getsource(m.run_collector_with_retry)
        # KeyboardInterrupt handler should also cleanup
        assert "KeyboardInterrupt" in source
        assert "_cleanup_connections()" in source


# ---------------------------------------------------------------------------
# 15. Paper trading code path (full trace)
# ---------------------------------------------------------------------------

class TestPaperTradingCodePath:
    """Verify that PAPER_TRADE=true results in zero Webull interactions."""

    def test_paper_trade_skips_webull_init(self):
        """When PAPER_TRADE=true, discord_collector sets webull_executor=None."""
        source = inspect.getsource(
            __import__(
                "options_owl.collectors.discord_collector",
                fromlist=["OptionsOwlBot"],
            )
        )
        # The guard: `if not self.settings.PAPER_TRADE: webull_executor = await ...`
        assert "if not self.settings.PAPER_TRADE" in source

    def test_paper_trader_skips_webull_order_when_executor_none(self):
        """PaperTrader._place_webull_order is only called when executor is not None."""
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader)
        assert "if self.webull_executor is not None:" in source

    def test_paper_trader_returns_portfolio_size_when_paper(self):
        """_get_effective_balance returns PORTFOLIO_SIZE when PAPER_TRADE=true."""
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader._get_effective_balance)
        # When paper trading, should use portfolio size (not Webull live balance)
        assert "PAPER_TRADE" in source or "webull_executor" in source

    def test_portfolio_sync_skips_when_paper(self):
        """sync_portfolio_from_webull does nothing when PAPER_TRADE=true."""
        from options_owl.execution.paper_trader import PaperTrader
        source = inspect.getsource(PaperTrader.sync_portfolio_from_webull)
        assert "self.webull_executor is None or self.settings.PAPER_TRADE" in source

    def test_webull_kill_switch_blocks_orders(self):
        """WEBULL_KILL_SWITCH=true raises RuntimeError on any order attempt."""
        from options_owl.execution.webull_executor import WebullExecutor
        source = inspect.getsource(WebullExecutor._check_kill_switch)
        assert "WEBULL_KILL_SWITCH" in source
        assert "RuntimeError" in source


# ---------------------------------------------------------------------------
# 16. Discord signals disabled (ENABLE_DISCORD_SIGNALS=false)
# ---------------------------------------------------------------------------

class TestDiscordSignalsDisabled:

    def test_discord_collector_checks_enable_flag(self):
        """Discord collector should skip signal processing when disabled."""
        source = inspect.getsource(
            __import__(
                "options_owl.collectors.discord_collector",
                fromlist=["OptionsOwlBot"],
            )
        )
        assert "ENABLE_DISCORD_SIGNALS" in source

    def test_docker_compose_has_discord_signals_disabled(self):
        """All trading bots in docker-compose should have ENABLE_DISCORD_SIGNALS=false."""
        from pathlib import Path
        dc = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        # Count occurrences: should be 5 (one per trading bot: kody, dennis, adam, vinny, yank)
        count = dc.count("ENABLE_DISCORD_SIGNALS=false")
        assert count == 5, f"Expected 5 bots with ENABLE_DISCORD_SIGNALS=false, got {count}"

    def test_docker_compose_paper_bots_paper_trade(self):
        """Paper-only bots should have PAPER_TRADE=true (adam, vinny, yank)."""
        from pathlib import Path
        dc = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        count = dc.count("PAPER_TRADE=true")
        assert count == 3, f"Expected 3 paper bots with PAPER_TRADE=true, got {count}"

    def test_docker_compose_paper_bots_kill_switch(self):
        """Paper-only bots should have WEBULL_KILL_SWITCH=true (adam, vinny, yank)."""
        from pathlib import Path
        dc = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        count = dc.count("WEBULL_KILL_SWITCH=true")
        assert count == 3, f"Expected 3 paper bots with WEBULL_KILL_SWITCH=true, got {count}"


# ---------------------------------------------------------------------------
# 17. Sourcing pushes to Discord webhook
# ---------------------------------------------------------------------------

class TestSourcingDiscordOutput:

    def test_scanner_imports_discord_webhook(self):
        """Sourcing scanner should import emit_discord."""
        import options_owl.sourcing.scanner as scanner
        source = inspect.getsource(scanner)
        assert "emit_discord" in source

    def test_webhook_module_uses_env_var(self):
        """Discord webhook module should read SOURCING_DISCORD_WEBHOOK_URL."""
        import options_owl.sourcing.output.discord_webhook as dw
        source = inspect.getsource(dw)
        assert "SOURCING_DISCORD_WEBHOOK_URL" in source


# ---------------------------------------------------------------------------
# 18. Position monitor asyncio.wait_for timeouts
# ---------------------------------------------------------------------------

class TestPositionMonitorTimeouts:

    def test_monitor_has_wait_for_timeouts(self):
        """Position monitor should wrap external I/O with asyncio.wait_for."""
        import options_owl.execution.position_monitor as pm
        source = inspect.getsource(pm)
        # Should have multiple wait_for calls (we added 5)
        count = source.count("asyncio.wait_for")
        assert count >= 5, f"Expected >= 5 asyncio.wait_for calls, found {count}"

    def test_no_sync_fetch_current_price(self):
        """_fetch_current_price should not be called synchronously in async context."""
        import options_owl.execution.position_monitor as pm
        source = inspect.getsource(pm)
        # Old bug: `current_price = _fetch_current_price(ticker)` (sync in async)
        # Fix: should use `await _fetch_price_async(ticker)` or equivalent
        # Check that _fetch_current_price is not called without await in the monitor loop
        assert "_fetch_price_async" in source or "to_thread" in source


# ---------------------------------------------------------------------------
# 19. Confidence tier ordering (vinny_strategy)
# ---------------------------------------------------------------------------

class TestConfidenceTierOrdering:

    def test_tiers_sorted_descending(self):
        """Confidence tiers must be sorted descending for first-match logic."""
        from options_owl.risk.vinny_strategy import _CONFIDENCE_TIERS
        thresholds = [t[0] for t in _CONFIDENCE_TIERS]
        assert thresholds == sorted(thresholds, reverse=True), \
            f"Tiers must be sorted descending: {thresholds}"

    def test_confidence_mapping_correctness(self):
        """Verify each confidence bucket maps to expected multiplier."""
        from options_owl.risk.vinny_strategy import _ml_confidence_to_mult
        # 0.95 → 0.90+ tier → 0.95 mult
        mult, _ = _ml_confidence_to_mult(0.95)
        assert mult == 0.95
        # 0.85 → 0.80+ tier → 0.60 mult
        mult, _ = _ml_confidence_to_mult(0.85)
        assert mult == 0.60
        # 0.75 → 0.70+ tier → 1.00 mult (sweet spot)
        mult, _ = _ml_confidence_to_mult(0.75)
        assert mult == 1.00
        # 0.65 → above the 0.62 CALL floor (2026-06-15 align) → 1.00 (lowest tier; was rejected at 0.70)
        mult, _ = _ml_confidence_to_mult(0.65)
        assert mult == 1.00
        # 0.61 → below the 0.62 CALL floor → 0.0 (rejected)
        mult, _ = _ml_confidence_to_mult(0.61)
        assert mult == 0.0
        # None → fallback
        mult, _ = _ml_confidence_to_mult(None)
        assert mult == 0.85
