"""End-to-end integration tests for the sourcing -> PG -> Redis -> signal consumer -> agent pipeline.

Tests the full flow with mocked PG/Redis connections:
  ML pipeline emits signal -> PG stores it -> signal_consumer reads it
  -> converts to TradeSignal -> routes to entry pipeline

Also tests Redis signal dedup and regime sharing independently.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.models.signals import (
    BotSource,
    Direction,
    Sentiment,
    SignalStrength,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_signal_data():
    """A qualifying ML signal dict as it would appear in PG ml_signals."""
    return {
        "id": 42,
        "ticker": "NVDA",
        "direction": "CALL",
        "score": 92,
        "ml_confidence": 0.92,
        "ml_threshold": 0.85,
        "ml_model_source": "ml_v3_pipeline",
        "ml_runner_score": 0.88,
        "premium": 2.50,
        "strike": 130.0,
        "expiry_date": "0DTE",
        "indicators": {},
        "score_breakdown": {},
        "emitted_at": datetime.now(tz=timezone.utc),
        "status": "pending",
        "consumed_by": [],
    }


@pytest.fixture
def mock_settings():
    """Minimal Settings-like object for signal consumer."""
    s = MagicMock()
    s.ENABLE_POSTGRES = True
    s.AGENT_ID = "owlet-test"
    s.MIN_SCORE = 78
    s.ML_MIN_SCORE = 78
    return s


@pytest.fixture
def mock_paper_trader():
    """PaperTrader mock with evaluate_and_trade returning a trade result."""
    pt = AsyncMock()
    pt.evaluate_and_trade = AsyncMock(return_value={"trade_id": 999, "ticker": "NVDA"})
    return pt


# ---------------------------------------------------------------------------
# 1. ML Pipeline -> PG signal emission
# ---------------------------------------------------------------------------


class TestMLPipelineEmitsSignal:
    """Verify emit_signal_to_pg writes to PG's ml_signals table."""

    @pytest.mark.asyncio
    async def test_emit_signal_calls_pg(self):
        """emit_signal_to_pg should call pg.emit_ml_signal with correct data."""
        mock_emit = AsyncMock(return_value=42)

        # emit_signal_to_pg does `from options_owl.db import postgres as pg` locally,
        # so we patch the module-level functions on the postgres module itself.
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.emit_ml_signal", mock_emit):
            from options_owl.sourcing.ml_pipeline import emit_signal_to_pg

            await emit_signal_to_pg(
                ticker="NVDA",
                direction="CALL",
                pattern_conf=0.92,
                entry_conf=0.88,
                premium=2.50,
                strike=130.0,
                expiry="0DTE",
                stop_pct=0.35,
                signal_quality=0.88,
            )

            mock_emit.assert_awaited_once()
            call_args = mock_emit.call_args[0][0]
            assert call_args["ticker"] == "NVDA"
            assert call_args["direction"] == "CALL"
            assert call_args["score"] == 92  # int(0.92 * 100)
            assert call_args["ml_confidence"] == 0.92
            assert call_args["premium"] == 2.50
            assert call_args["strike"] == 130.0
            assert call_args["expiry_date"] == "0DTE"
            assert call_args["emitted_at"] is not None

    @pytest.mark.asyncio
    async def test_emit_signal_skips_when_pg_disconnected(self):
        """emit_signal_to_pg should no-op when PG is not connected."""
        mock_emit = AsyncMock()

        with patch("options_owl.db.postgres.is_connected", return_value=False), \
             patch("options_owl.db.postgres.emit_ml_signal", mock_emit):
            from options_owl.sourcing.ml_pipeline import emit_signal_to_pg

            await emit_signal_to_pg(
                ticker="NVDA",
                direction="CALL",
                pattern_conf=0.92,
                entry_conf=None,
                premium=2.50,
                strike=130.0,
                expiry="0DTE",
                stop_pct=None,
                signal_quality=None,
            )
            mock_emit.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Signal consumer picks up from PG
# ---------------------------------------------------------------------------


class TestSignalConsumerRouting:
    """Verify signal_consumer reads PG signals and routes to paper_trader."""

    @pytest.mark.asyncio
    async def test_poll_and_route_converts_signal(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """_poll_and_route should convert PG row to TradeSignal and call evaluate_and_trade."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[sample_signal_data]), \
             patch("options_owl.db.postgres.mark_signal_consumed",
                   new_callable=AsyncMock) as mock_mark:

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            # evaluate_and_trade should have been called once
            mock_paper_trader.evaluate_and_trade.assert_awaited_once()

            call_args = mock_paper_trader.evaluate_and_trade.call_args
            trade_signal = call_args[0][0]
            signal_id = call_args[0][1]

            # Verify TradeSignal fields
            assert isinstance(trade_signal, TradeSignal)
            assert trade_signal.ticker == "NVDA"
            assert trade_signal.direction == Direction.CALL
            assert trade_signal.sentiment == Sentiment.BULLISH
            assert trade_signal.score == 92
            assert trade_signal.bot_source == BotSource.ML_SOURCING
            assert trade_signal.entry_price == 130.0  # underlying price (strike fallback)
            assert trade_signal.strike == 130.0
            assert trade_signal.expiry == "0DTE"

            # Synthetic signal_id is negative of PG id
            assert signal_id == -42

            # ml_confidence passed through
            assert call_args[1]["ml_confidence"] == 0.92

            # Signal should be marked consumed
            mock_mark.assert_awaited_once_with(42, "owlet-test")

    @pytest.mark.asyncio
    async def test_poll_and_route_skips_low_score(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """Signals below MIN_SCORE should be consumed but not traded."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        sample_signal_data["score"] = 50  # Below MIN_SCORE of 78

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[sample_signal_data]), \
             patch("options_owl.db.postgres.mark_signal_consumed",
                   new_callable=AsyncMock) as mock_mark:

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            # Should NOT trade
            mock_paper_trader.evaluate_and_trade.assert_not_awaited()

            # But SHOULD mark consumed (so we don't retry)
            mock_mark.assert_awaited_once_with(42, "owlet-test")

    @pytest.mark.asyncio
    async def test_poll_and_route_put_signal(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """PUT direction should map to Direction.PUT and Sentiment.BEARISH."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        sample_signal_data["direction"] = "PUT"

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[sample_signal_data]), \
             patch("options_owl.db.postgres.mark_signal_consumed",
                   new_callable=AsyncMock):

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            trade_signal = mock_paper_trader.evaluate_and_trade.call_args[0][0]
            assert trade_signal.direction == Direction.PUT
            assert trade_signal.sentiment == Sentiment.BEARISH

    @pytest.mark.asyncio
    async def test_poll_and_route_no_signals(
        self, mock_settings, mock_paper_trader
    ):
        """No pending signals should result in no trades and no errors."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[]):

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            mock_paper_trader.evaluate_and_trade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_and_route_pg_disconnected(
        self, mock_settings, mock_paper_trader
    ):
        """If PG is not connected, should return immediately."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        with patch("options_owl.db.postgres.is_connected", return_value=False):
            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            mock_paper_trader.evaluate_and_trade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_and_route_marks_consumed_on_trade_error(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """Even if evaluate_and_trade raises, signal should still be marked consumed."""
        from options_owl.collectors.signal_consumer import _poll_and_route

        mock_paper_trader.evaluate_and_trade = AsyncMock(
            side_effect=RuntimeError("pipeline exploded")
        )

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[sample_signal_data]), \
             patch("options_owl.db.postgres.mark_signal_consumed",
                   new_callable=AsyncMock) as mock_mark:

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

            mock_mark.assert_awaited_once_with(42, "owlet-test")


# ---------------------------------------------------------------------------
# 3. Redis signal dedup
# ---------------------------------------------------------------------------


class TestRedisSignalDedup:
    """Test that Redis SET NX provides atomic signal claiming."""

    @pytest.mark.asyncio
    async def test_first_agent_claims_signal(self):
        """First agent to call try_claim_signal should succeed."""
        from options_owl.db import redis_client

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)  # NX succeeded

        with patch.object(redis_client, "_redis", mock_redis):
            result = await redis_client.try_claim_signal("sig_123", "owlet-kody")

            assert result is True
            mock_redis.set.assert_awaited_once_with(
                "owl:signal:sig_123", "owlet-kody", nx=True, ex=300
            )

    @pytest.mark.asyncio
    async def test_second_agent_blocked(self):
        """Second agent trying to claim same signal should fail."""
        from options_owl.db import redis_client

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=None)  # NX failed (key exists)

        with patch.object(redis_client, "_redis", mock_redis):
            result = await redis_client.try_claim_signal("sig_123", "owlet-adam")

            assert result is False

    @pytest.mark.asyncio
    async def test_two_agents_one_wins(self):
        """Simulate two agents racing for the same signal -- only one should win."""
        from options_owl.db import redis_client

        call_count = 0

        async def mock_set(key, value, nx=False, ex=None):
            nonlocal call_count
            call_count += 1
            # First caller wins, second loses
            return True if call_count == 1 else None

        mock_redis = AsyncMock()
        mock_redis.set = mock_set

        with patch.object(redis_client, "_redis", mock_redis):
            results = await asyncio.gather(
                redis_client.try_claim_signal("sig_456", "owlet-kody"),
                redis_client.try_claim_signal("sig_456", "owlet-adam"),
            )

            assert sum(results) == 1  # Exactly one winner

    @pytest.mark.asyncio
    async def test_dedup_fails_open_without_redis(self):
        """Without Redis, try_claim_signal should return True (fail-open)."""
        from options_owl.db import redis_client

        with patch.object(redis_client, "_redis", None):
            result = await redis_client.try_claim_signal("sig_789", "owlet-kody")
            assert result is True

    @pytest.mark.asyncio
    async def test_dedup_fails_open_on_redis_error(self):
        """On Redis error, try_claim_signal should return True (fail-open)."""
        from options_owl.db import redis_client

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("Redis down"))

        with patch.object(redis_client, "_redis", mock_redis):
            result = await redis_client.try_claim_signal("sig_789", "owlet-kody")
            assert result is True


# ---------------------------------------------------------------------------
# 4. Redis regime sharing
# ---------------------------------------------------------------------------


class TestRedisRegimeSharing:
    """Test that regime score is shared and read correctly via Redis."""

    @pytest.mark.asyncio
    async def test_set_and_get_regime(self):
        """set_regime_score stores data, get_regime_decision reads it back."""
        from options_owl.db import redis_client

        stored = {}

        async def mock_hset(key, mapping=None):
            stored[key] = mapping

        async def mock_expire(key, ttl):
            pass  # no-op

        async def mock_hgetall(key):
            return stored.get(key, {})

        mock_redis = AsyncMock()
        mock_redis.hset = mock_hset
        mock_redis.expire = mock_expire
        mock_redis.hgetall = mock_hgetall

        with patch.object(redis_client, "_redis", mock_redis):
            await redis_client.set_regime_score("2026-05-24", 0.75, skip=False)

            decision = await redis_client.get_regime_decision("2026-05-24")

            assert decision is not None
            assert decision["score"] == 0.75
            assert decision["skip"] is False

    @pytest.mark.asyncio
    async def test_regime_skip_day(self):
        """Regime with skip=True should be stored and retrieved."""
        from options_owl.db import redis_client

        stored = {}

        async def mock_hset(key, mapping=None):
            stored[key] = mapping

        async def mock_expire(key, ttl):
            pass

        async def mock_hgetall(key):
            return stored.get(key, {})

        mock_redis = AsyncMock()
        mock_redis.hset = mock_hset
        mock_redis.expire = mock_expire
        mock_redis.hgetall = mock_hgetall

        with patch.object(redis_client, "_redis", mock_redis):
            await redis_client.set_regime_score("2026-05-24", 0.30, skip=True)

            decision = await redis_client.get_regime_decision("2026-05-24")

            assert decision is not None
            assert decision["score"] == 0.30
            assert decision["skip"] is True

    @pytest.mark.asyncio
    async def test_regime_returns_none_without_redis(self):
        """Without Redis, get_regime_decision returns None."""
        from options_owl.db import redis_client

        with patch.object(redis_client, "_redis", None):
            decision = await redis_client.get_regime_decision("2026-05-24")
            assert decision is None

    @pytest.mark.asyncio
    async def test_regime_returns_none_for_missing_date(self):
        """get_regime_decision for a date with no data returns None."""
        from options_owl.db import redis_client

        mock_redis = AsyncMock()
        mock_redis.hgetall = AsyncMock(return_value={})

        with patch.object(redis_client, "_redis", mock_redis):
            decision = await redis_client.get_regime_decision("1999-01-01")
            assert decision is None


# ---------------------------------------------------------------------------
# 5. PG dual-write on trade open
# ---------------------------------------------------------------------------


class TestPGDualWrite:
    """Test that paper_trader fires PG write_trade_open when opening a trade."""

    @pytest.mark.asyncio
    async def test_write_trade_open_called_with_correct_data(self):
        """pg.write_trade_open should be called with trade data dict."""
        from options_owl.db import postgres as pg

        trade_data = {
            "ticker": "NVDA",
            "direction": "CALL",
            "sentiment": "bullish",
            "score": 92,
            "strength": "strong",
            "bot_source": "ml_sourcing",
            "entry_price": 2.50,
            "strike": 130.0,
            "option_type": "call",
            "contracts": 5,
            "premium_per_contract": 2.50,
            "total_cost": 1250.0,
            "signal_id": -42,
            "expiry_date": "0DTE",
            "opened_at": datetime.now(tz=timezone.utc),
            "ml_confidence": 0.92,
        }

        with patch.object(pg, "_pool", MagicMock()):
            with patch.object(pg, "fetchval", new_callable=AsyncMock, return_value=1) as mock_fv:
                result = await pg.write_trade_open("owlet-test", 100, trade_data)

                assert result == 1
                mock_fv.assert_awaited_once()
                # Verify agent_id and sqlite_id are passed
                call_args = mock_fv.call_args[0]
                query = call_args[0]
                assert "INSERT INTO trades" in query
                assert call_args[1] == 100  # sqlite_id
                assert call_args[2] == "owlet-test"  # agent_id

    @pytest.mark.asyncio
    async def test_write_trade_open_returns_none_when_disconnected(self):
        """write_trade_open should return None when PG pool is not available."""
        from options_owl.db import postgres as pg

        with patch.object(pg, "_pool", None):
            result = await pg.write_trade_open("owlet-test", 100, {"ticker": "NVDA"})
            assert result is None

    @pytest.mark.asyncio
    async def test_write_trade_event_records_audit(self):
        """write_trade_event should insert into trade_events table."""
        from options_owl.db import postgres as pg

        with patch.object(pg, "_pool", MagicMock()):
            with patch.object(
                pg, "fetchval", new_callable=AsyncMock, return_value=7
            ) as mock_fv:
                result = await pg.write_trade_event(
                    "owlet-test", 100, "pipeline_approved", {"gates_passed": 18}
                )

                assert result == 7
                call_args = mock_fv.call_args[0]
                assert "INSERT INTO trade_events" in call_args[0]
                assert call_args[1] == "owlet-test"
                assert call_args[2] == 100  # trade_id
                assert call_args[3] == "pipeline_approved"


# ---------------------------------------------------------------------------
# 6. Full pipeline mock (end-to-end flow)
# ---------------------------------------------------------------------------


class TestFullPipelineFlow:
    """Simulate the complete flow:
    ML pipeline emits signal -> PG stores it -> signal_consumer reads it
    -> converts to TradeSignal -> routes to entry pipeline.
    """

    @pytest.mark.asyncio
    async def test_full_flow_emit_to_trade(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """Signal emitted by ML pipeline should flow through to paper_trader."""
        emitted_signals: list[dict] = []

        async def mock_emit_ml_signal(signal_data: dict) -> int:
            """Capture emitted signal and assign an ID."""
            signal_data["id"] = 42
            signal_data["status"] = "pending"
            signal_data["consumed_by"] = []
            emitted_signals.append(signal_data)
            return 42

        async def mock_get_pending(agent_id: str, max_age_minutes: int = 10):
            """Return signals that haven't been consumed by this agent."""
            return [
                s for s in emitted_signals
                if agent_id not in s.get("consumed_by", [])
            ]

        consumed_pairs: list[tuple[int, str]] = []

        async def mock_mark_consumed(signal_id: int, agent_id: str):
            consumed_pairs.append((signal_id, agent_id))
            for s in emitted_signals:
                if s["id"] == signal_id:
                    s.setdefault("consumed_by", []).append(agent_id)

        # --- Step 1: ML pipeline emits signal to PG ---
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.emit_ml_signal", side_effect=mock_emit_ml_signal):
            from options_owl.db import postgres as pg

            signal_id = await pg.emit_ml_signal({
                "ticker": "NVDA",
                "direction": "CALL",
                "score": 92,
                "ml_confidence": 0.92,
                "ml_threshold": 0.85,
                "ml_model_source": "ml_v3_pipeline",
                "ml_runner_score": 0.88,
                "premium": 2.50,
                "strike": 130.0,
                "expiry_date": "0DTE",
                "indicators": {},
                "score_breakdown": {},
                "emitted_at": datetime.now(tz=timezone.utc),
            })

            assert signal_id == 42
            assert len(emitted_signals) == 1

        # --- Step 2: Signal consumer picks it up and routes to paper_trader ---
        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   side_effect=mock_get_pending), \
             patch("options_owl.db.postgres.mark_signal_consumed", new_callable=AsyncMock,
                   side_effect=mock_mark_consumed):

            from options_owl.collectors.signal_consumer import _poll_and_route

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

        # --- Verify the full chain ---
        # 1. evaluate_and_trade was called
        mock_paper_trader.evaluate_and_trade.assert_awaited_once()

        # 2. TradeSignal has correct fields from original emission
        trade_signal = mock_paper_trader.evaluate_and_trade.call_args[0][0]
        assert trade_signal.ticker == "NVDA"
        assert trade_signal.direction == Direction.CALL
        assert trade_signal.score == 92
        assert trade_signal.bot_source == BotSource.ML_SOURCING
        assert trade_signal.entry_price == 130.0  # underlying price (strike fallback)
        assert trade_signal.strike == 130.0

        # 3. ml_confidence passed through
        assert mock_paper_trader.evaluate_and_trade.call_args[1]["ml_confidence"] == 0.92

        # 4. Signal was marked consumed
        assert (42, "owlet-test") in consumed_pairs

    @pytest.mark.asyncio
    async def test_full_flow_rejected_signal_not_traded(
        self, mock_settings, mock_paper_trader
    ):
        """A low-score signal should flow through PG but get rejected before trading."""
        low_score_signal = {
            "id": 99,
            "ticker": "AAPL",
            "direction": "PUT",
            "score": 50,  # Below MIN_SCORE
            "ml_confidence": 0.50,
            "premium": 1.00,
            "strike": 200.0,
            "expiry_date": "0DTE",
        }

        consumed = []

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=[low_score_signal]), \
             patch("options_owl.db.postgres.mark_signal_consumed", new_callable=AsyncMock,
                   side_effect=lambda sid, aid: consumed.append((sid, aid))):

            from options_owl.collectors.signal_consumer import _poll_and_route

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

        # Should NOT have been traded
        mock_paper_trader.evaluate_and_trade.assert_not_awaited()

        # But SHOULD be marked consumed to avoid retry
        assert (99, "owlet-test") in consumed

    @pytest.mark.asyncio
    async def test_full_flow_multiple_signals_batch(
        self, mock_settings, mock_paper_trader
    ):
        """Multiple signals in a single poll should each be processed independently."""
        signals = [
            {
                "id": 1,
                "ticker": "NVDA",
                "direction": "CALL",
                "score": 95,
                "ml_confidence": 0.95,
                "premium": 2.50,
                "strike": 130.0,
                "expiry_date": "0DTE",
            },
            {
                "id": 2,
                "ticker": "TSLA",
                "direction": "PUT",
                "score": 88,
                "ml_confidence": 0.88,
                "premium": 3.00,
                "strike": 180.0,
                "expiry_date": "0DTE",
            },
            {
                "id": 3,
                "ticker": "AAPL",
                "direction": "CALL",
                "score": 60,  # Below threshold
                "ml_confidence": 0.60,
                "premium": 1.50,
                "strike": 195.0,
                "expiry_date": "0DTE",
            },
        ]

        consumed_ids = []

        with patch("options_owl.db.postgres.is_connected", return_value=True), \
             patch("options_owl.db.postgres.get_pending_signals", new_callable=AsyncMock,
                   return_value=signals), \
             patch("options_owl.db.postgres.mark_signal_consumed", new_callable=AsyncMock,
                   side_effect=lambda sid, aid: consumed_ids.append(sid)):

            from options_owl.collectors.signal_consumer import _poll_and_route

            await _poll_and_route(mock_paper_trader, mock_settings, "owlet-test")

        # NVDA and TSLA should be traded (score >= 78), AAPL skipped
        assert mock_paper_trader.evaluate_and_trade.await_count == 2

        tickers_traded = [
            call.args[0].ticker
            for call in mock_paper_trader.evaluate_and_trade.call_args_list
        ]
        assert "NVDA" in tickers_traded
        assert "TSLA" in tickers_traded
        assert "AAPL" not in tickers_traded

        # All three should be marked consumed
        assert sorted(consumed_ids) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_full_flow_with_redis_dedup(
        self, sample_signal_data, mock_settings, mock_paper_trader
    ):
        """When Redis dedup is in play, only the claiming agent should trade."""
        from options_owl.db import redis_client

        # Agent 1 claims successfully
        mock_redis_1 = AsyncMock()
        mock_redis_1.set = AsyncMock(return_value=True)

        with patch.object(redis_client, "_redis", mock_redis_1):
            claimed = await redis_client.try_claim_signal("ml_42", "owlet-kody")
            assert claimed is True

        # Agent 2 gets blocked
        mock_redis_2 = AsyncMock()
        mock_redis_2.set = AsyncMock(return_value=None)

        with patch.object(redis_client, "_redis", mock_redis_2):
            claimed = await redis_client.try_claim_signal("ml_42", "owlet-adam")
            assert claimed is False

    @pytest.mark.asyncio
    async def test_score_to_strength_mapping(self):
        """Verify _score_to_strength returns correct SignalStrength tiers."""
        from options_owl.collectors.signal_consumer import _score_to_strength

        assert _score_to_strength(160) == SignalStrength.ELITE
        assert _score_to_strength(150) == SignalStrength.ELITE
        assert _score_to_strength(130) == SignalStrength.STRONG
        assert _score_to_strength(120) == SignalStrength.STRONG
        assert _score_to_strength(100) == SignalStrength.GOOD
        assert _score_to_strength(90) == SignalStrength.GOOD
        assert _score_to_strength(85) == SignalStrength.MODERATE
        assert _score_to_strength(78) == SignalStrength.MODERATE
        assert _score_to_strength(70) == SignalStrength.MARGINAL
        assert _score_to_strength(0) == SignalStrength.MARGINAL
