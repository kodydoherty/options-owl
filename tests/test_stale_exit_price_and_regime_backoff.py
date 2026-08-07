"""Regression tests for the 2026-08-07 stale-exit-price incident.

Two independent bugs surfaced the same day (kody -$1,063, MSTR #671 cut at a
cached $0.66 while the live bid was $0.47):

1. The exit monitor rejects a Redis snapshot older than EXIT_SNAPSHOT_MAX_AGE_SEC
   and then falls through to MarketDataStream.get_option_premium, whose own caches
   accepted 120s (Redis) / 30s (WS) — silently undoing the guard.
2. check_ticker_regime never cached a FAILED regime build, so every scan re-paid
   the full 15s timeout (8-11k timeouts/day/bot, ~256 per ticker).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from options_owl.collectors.market_data_stream import MarketDataStream
from options_owl.config.settings import Settings


def _make_settings() -> Settings:
    return Settings(
        POLYGON_API_KEY="test_key",
        DISCORD_TOKEN="x",
        _env_file=None,
    )


class TestExitPremiumFreshnessBound:
    """max_age_sec must bound BOTH in-process caches, not just the WS one."""

    @pytest.mark.asyncio
    async def test_ws_cache_within_default_window_is_served(self):
        """No max_age_sec => legacy 30s WS window, unchanged for other callers."""
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")
        stream._option_cache["O:SPY260414C00691000"] = (1.25, time.time() - 20)

        premium = await stream.get_option_premium("SPY", 691.0, "2026-04-14", "call")
        assert premium == 1.25, "20s-old WS quote is fine under the legacy window"

    @pytest.mark.asyncio
    async def test_exit_bound_rejects_ws_quote_the_legacy_window_would_serve(self):
        """THE BUG: a 20s-old quote passed the 30s WS window even though the exit
        path had just rejected a 20s-old snapshot. With max_age_sec=10 it must
        fall through to a live fetch instead."""
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")
        stream._option_cache["O:SPY260414C00691000"] = (1.25, time.time() - 20)
        stream._polygon_rest_option_premium = AsyncMock(return_value=0.90)

        premium = await stream.get_option_premium(
            "SPY", 691.0, "2026-04-14", "call", max_age_sec=10.0
        )
        assert premium == 0.90, "must use the LIVE price, not the 20s-old cache"
        stream._polygon_rest_option_premium.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exit_bound_still_serves_a_genuinely_fresh_quote(self):
        """The bound must not force a live fetch when the cache IS fresh —
        otherwise every 5s monitor tick would hammer Polygon REST."""
        stream = MarketDataStream(_make_settings())
        await stream.subscribe_option("SPY", 691.0, "2026-04-14", "call")
        stream._option_cache["O:SPY260414C00691000"] = (1.25, time.time() - 2)
        stream._polygon_rest_option_premium = AsyncMock(return_value=0.90)

        premium = await stream.get_option_premium(
            "SPY", 691.0, "2026-04-14", "call", max_age_sec=10.0
        )
        assert premium == 1.25
        stream._polygon_rest_option_premium.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exit_bound_rejects_stale_redis_cache(self):
        """The Redis branch (120s legacy window) is the one that actually bit on
        MSTR — it must honour the exit bound too."""
        import options_owl.db.redis_client as redis_client

        stream = MarketDataStream(_make_settings())
        orig_connected = redis_client.is_connected
        orig_get = redis_client.get_option_premium
        redis_client.is_connected = lambda: True
        # 60s old: inside the legacy 120s window, far outside a 10s exit bound
        redis_client.get_option_premium = AsyncMock(
            return_value={"mid": 0.66, "t": time.time() - 60}
        )
        stream._polygon_rest_option_premium = AsyncMock(return_value=0.47)
        try:
            legacy = await stream.get_option_premium(
                "MSTR", 104.0, "2026-08-07", "call"
            )
            assert legacy == 0.66, "legacy window still serves the 60s-old price"

            bounded = await stream.get_option_premium(
                "MSTR", 104.0, "2026-08-07", "call", max_age_sec=10.0
            )
            assert bounded == 0.47, "exit bound must reach the LIVE bid"
        finally:
            redis_client.is_connected = orig_connected
            redis_client.get_option_premium = orig_get


class TestExitPathPassesTheBound:
    """Source-level guard: the monitor must actually pass max_age_sec."""

    def test_position_monitor_passes_max_age_to_stream(self):
        import inspect

        from options_owl.execution import position_monitor

        src = inspect.getsource(position_monitor.run_position_monitor)
        idx = src.find("market_stream.get_option_premium(")
        assert idx != -1, "exit path no longer calls get_option_premium"
        call = src[idx : idx + 700]
        assert "max_age_sec" in call, (
            "exit path must bound the stream cache or the "
            "EXIT_SNAPSHOT_MAX_AGE_SEC guard is silently defeated"
        )
        assert "EXIT_SNAPSHOT_MAX_AGE_SEC" in call


class TestRegimeFailureBackoff:
    """A timed-out regime build must not be retried on every scan."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        from options_owl.sourcing import ml_pipeline

        ml_pipeline._ticker_regime_cache.clear()
        ml_pipeline._ticker_regime_fail_cache.clear()
        yield
        ml_pipeline._ticker_regime_cache.clear()
        ml_pipeline._ticker_regime_fail_cache.clear()

    @pytest.mark.asyncio
    async def test_failure_is_computed_once_then_backed_off(self, monkeypatch):
        from options_owl.sourcing import ml_pipeline

        calls = {"n": 0}

        async def _always_times_out(ticker, models, settings):
            calls["n"] += 1
            return None

        monkeypatch.setattr(
            ml_pipeline, "_compute_regime_score_for_ticker", _always_times_out
        )

        class _Models:
            regime_model = object()

        models, settings = _Models(), object()

        # 20 scans of the same ticker, as the live loop does all day
        for _ in range(20):
            allowed = await ml_pipeline.check_ticker_regime("SPY", models, settings)
            assert allowed is True, "must fail OPEN — never block on a timeout"

        assert calls["n"] == 1, (
            f"regime build ran {calls['n']}x across 20 scans — the retry storm "
            "is back (each miss costs 15s inside the scan loop)"
        )

    @pytest.mark.asyncio
    async def test_backoff_expires_so_regime_can_recover(self, monkeypatch):
        from options_owl.sourcing import ml_pipeline

        calls = {"n": 0}

        async def _times_out(ticker, models, settings):
            calls["n"] += 1
            return None

        monkeypatch.setattr(
            ml_pipeline, "_compute_regime_score_for_ticker", _times_out
        )

        class _Models:
            regime_model = object()

        models, settings = _Models(), object()

        await ml_pipeline.check_ticker_regime("SPY", models, settings)
        assert calls["n"] == 1

        # Age the recorded failure past the backoff window
        key = next(iter(ml_pipeline._ticker_regime_fail_cache))
        ml_pipeline._ticker_regime_fail_cache[key] = (
            time.monotonic() - ml_pipeline.REGIME_FAIL_BACKOFF_SEC - 1
        )

        await ml_pipeline.check_ticker_regime("SPY", models, settings)
        assert calls["n"] == 2, "backoff must expire so a recovered DB is picked up"

    @pytest.mark.asyncio
    async def test_success_still_caches_and_is_not_treated_as_failure(
        self, monkeypatch
    ):
        from options_owl.sourcing import ml_pipeline

        calls = {"n": 0}

        async def _ok(ticker, models, settings):
            calls["n"] += 1
            return 0.99

        monkeypatch.setattr(ml_pipeline, "_compute_regime_score_for_ticker", _ok)

        class _Models:
            regime_model = object()

        class _Settings:
            ML_REGIME_THRESHOLD = 0.20

        models, settings = _Models(), _Settings()

        for _ in range(5):
            assert await ml_pipeline.check_ticker_regime("SPY", models, settings)

        assert calls["n"] == 1, "successful score must still be cached for the day"
        assert not ml_pipeline._ticker_regime_fail_cache


class TestMLPremiumCapConfigurable:
    """The ML signal-level premium cap was a hardcoded 6.0 in bot_runner.

    Real Webull fills (2026-08-07) show >=$3 contracts are ~half of ml_sourcing's
    loss (31 trades, 16% WR, -$2,425) because they carry the lowest MFE (13.4%).
    Making the cap configurable is what lets us act on that without a code change.
    """

    def test_default_is_unchanged_behaviour(self):
        """Default MUST stay 6.0 — anything else silently changes every bot."""
        s = Settings(DISCORD_TOKEN="x", _env_file=None)
        assert s.ML_PREMIUM_CAP == 6.0

    def test_cap_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("ML_PREMIUM_CAP", "3.0")
        assert Settings(DISCORD_TOKEN="x", _env_file=None).ML_PREMIUM_CAP == 3.0

    def test_scan_loop_reads_the_setting_not_a_constant(self):
        """Guard against the cap being re-hardcoded. The scan loop must source it
        from settings, or lowering the cap in env would silently do nothing."""
        import inspect

        from options_owl import bot_runner

        src = inspect.getsource(bot_runner._run_ml_scan_loop)
        assert "ML_PREMIUM_CAP" in src, "scan loop no longer reads settings.ML_PREMIUM_CAP"
        assert "PREMIUM_CAP = 6.0" not in src, "cap was re-hardcoded"


class TestSizingAuditInitialisation:
    """_sizing_audit must be initialised OUTSIDE every conditional.

    The sizing layers run only under `use_vinny and use_score_sizing`, but the
    INSERT that persists them runs unconditionally. Initialising inside the branch
    raised UnboundLocalError on every non-vinny trade — caught by 36 test failures
    during development. Same class as the 2026-05-07 monitor freeze that stopped all
    exits. This test fails if anyone moves the init back inside a branch.
    """

    def test_init_dominates_every_use(self):
        import ast
        import inspect

        from options_owl.execution import paper_trader

        tree = ast.parse(inspect.getsource(paper_trader))
        target = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "_open_single_trade"
        )
        init = None
        loads = []
        for n in ast.walk(target):
            if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_sizing_audit":
                init = n.lineno
            if isinstance(n, ast.Name) and n.id == "_sizing_audit" and isinstance(n.ctx, ast.Load):
                loads.append(n.lineno)
        assert init is not None, "_sizing_audit is no longer initialised"
        assert loads, "_sizing_audit is never read — persistence was removed?"
        assert all(u > init for u in loads), (
            f"_sizing_audit read at {sorted(x for x in loads if x < init)} before its "
            f"init at {init} — conditional-only assignment, will UnboundLocalError"
        )

    def test_persisted_columns_are_declared(self):
        """The audit columns must be migrated, or the INSERT fails at runtime."""
        import inspect

        from options_owl.execution import paper_trader

        src = inspect.getsource(paper_trader)
        for col in ("p_runner", "size_conv_mult", "size_entry_delta"):
            assert f'"{col} REAL"' in src, f"{col} migration missing"
            assert col in src, f"{col} not persisted"
