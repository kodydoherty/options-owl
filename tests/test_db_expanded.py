"""Expanded tests for the database layer — covers all new DB functions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from options_owl.journal import db


# ---------------------------------------------------------------------------
# Helper to insert a raw message (most DB functions require a message_id FK)
# ---------------------------------------------------------------------------


async def _insert_msg(path: str, author: str = "Captain Hook", content: str = "test") -> int:
    return await db.save_message(
        path,
        guild_id=1,
        channel_id=1,
        author_id=1,
        author_name=author,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _signal_dict(**overrides) -> dict:
    defaults = {
        "bot_source": "Captain Hook",
        "ticker": "NVDA",
        "sentiment": "bearish",
        "direction": "put",
        "score": 100,
        "strength": "strong",
        "entry_price": 170.0,
        "target_price": 167.0,
        "expected_move_pct": 1.8,
        "strike": 170.0,
        "expiry": "0DTE",
        "risk_reward": 1.5,
        "target_1": 168.0,
        "target_1_pct": 0.5,
        "target_2": 167.0,
        "target_2_pct": 0.9,
        "stop_price": 171.0,
        "stop_pct": -0.5,
        "exit_by": "10:40",
        "atm_strike": 170.0,
        "atm_premium": 1.70,
        "otm_strike": 167.5,
        "otm_premium": 0.46,
        "key_signals": ["BB 2σ Touch", "EMA Bounce"],
        "is_elite": True,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# save_trade_signal / get_trade_signal round-trip
# ---------------------------------------------------------------------------


class TestTradeSignalRoundTrip:
    @pytest.mark.asyncio
    async def test_all_fields_saved_and_retrieved(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        sig_dict = _signal_dict()

        sig_id = await db.save_trade_signal(tmp_db_path, message_id=msg_id, signal=sig_dict)
        assert sig_id is not None

        row = await db.get_trade_signal(tmp_db_path, sig_id)
        assert row is not None
        assert row["ticker"] == "NVDA"
        assert row["bot_source"] == "Captain Hook"
        assert row["sentiment"] == "bearish"
        assert row["direction"] == "put"
        assert row["score"] == 100
        assert row["strength"] == "strong"
        assert row["entry_price"] == 170.0
        assert row["target_price"] == 167.0
        assert row["expected_move_pct"] == 1.8
        assert row["strike"] == 170.0
        assert row["expiry"] == "0DTE"
        assert row["risk_reward"] == 1.5
        assert row["target_1"] == 168.0
        assert row["target_1_pct"] == 0.5
        assert row["target_2"] == 167.0
        assert row["target_2_pct"] == 0.9
        assert row["stop_price"] == 171.0
        assert row["stop_pct"] == -0.5
        assert row["exit_by"] == "10:40"
        assert row["atm_strike"] == 170.0
        assert row["atm_premium"] == 1.70
        assert row["otm_strike"] == 167.5
        assert row["otm_premium"] == 0.46
        assert row["key_signals"] == ["BB 2σ Touch", "EMA Bounce"]
        assert row["is_elite"] is True
        assert row["created_at"] is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_signal_returns_none(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        assert await db.get_trade_signal(tmp_db_path, 9999) is None


# ---------------------------------------------------------------------------
# get_unresolved_signals
# ---------------------------------------------------------------------------


class TestUnresolvedSignals:
    @pytest.mark.asyncio
    async def test_returns_only_unresolved(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        sig1_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(ticker="NVDA")
        )
        sig2_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(ticker="TSLA")
        )

        # Resolve sig1 only
        await db.save_signal_outcome(
            tmp_db_path,
            outcome={
                "signal_id": sig1_id,
                "outcome": "t1_hit",
                "pnl_underlying_pct": 1.0,
            },
        )

        unresolved = await db.get_unresolved_signals(tmp_db_path)
        assert len(unresolved) == 1
        assert unresolved[0]["ticker"] == "TSLA"
        assert unresolved[0]["id"] == sig2_id

    @pytest.mark.asyncio
    async def test_all_resolved_returns_empty(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        sig_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )
        await db.save_signal_outcome(
            tmp_db_path,
            outcome={"signal_id": sig_id, "outcome": "t2_hit", "pnl_underlying_pct": 2.0},
        )

        assert await db.get_unresolved_signals(tmp_db_path) == []


# ---------------------------------------------------------------------------
# get_signals_by_bot
# ---------------------------------------------------------------------------


class TestSignalsByBot:
    @pytest.mark.asyncio
    async def test_filters_by_bot_source(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(bot_source="Captain Hook")
        )
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(bot_source="Neverland Pan", ticker="TSLA")
        )
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(bot_source="Captain Hook", ticker="AAPL")
        )

        hook_sigs = await db.get_signals_by_bot(tmp_db_path, "Captain Hook")
        assert len(hook_sigs) == 2
        tickers = {s["ticker"] for s in hook_sigs}
        assert tickers == {"NVDA", "AAPL"}

        pan_sigs = await db.get_signals_by_bot(tmp_db_path, "Neverland Pan")
        assert len(pan_sigs) == 1
        assert pan_sigs[0]["ticker"] == "TSLA"

    @pytest.mark.asyncio
    async def test_no_matching_bot_returns_empty(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(bot_source="Captain Hook")
        )

        result = await db.get_signals_by_bot(tmp_db_path, "Tinker")
        assert result == []


# ---------------------------------------------------------------------------
# get_signals_by_date_range
# ---------------------------------------------------------------------------


class TestSignalsByDateRange:
    @pytest.mark.asyncio
    async def test_date_range_filtering(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        # Insert signals — created_at is set to datetime.now() inside save_trade_signal
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(ticker="NVDA")
        )
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict(ticker="TSLA")
        )

        # Wide range should catch everything
        signals = await db.get_signals_by_date_range(
            tmp_db_path, "2020-01-01", "2030-12-31"
        )
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_narrow_range_excludes(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )

        # Range in the past should return nothing
        signals = await db.get_signals_by_date_range(
            tmp_db_path, "2020-01-01", "2020-12-31"
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# save_price_snapshots / get_price_snapshots
# ---------------------------------------------------------------------------


class TestPriceSnapshots:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        sig_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )

        bars = [
            {
                "timestamp": "2026-03-27T09:30:00",
                "open": 170.0,
                "high": 170.8,
                "low": 169.5,
                "close": 170.5,
                "volume": 10000,
            },
            {
                "timestamp": "2026-03-27T09:31:00",
                "open": 170.5,
                "high": 171.0,
                "low": 170.0,
                "close": 170.2,
                "volume": 12000,
            },
        ]

        await db.save_price_snapshots(
            tmp_db_path, signal_id=sig_id, ticker="NVDA", bars=bars, interval="1m"
        )

        retrieved = await db.get_price_snapshots(tmp_db_path, sig_id)
        assert len(retrieved) == 2
        assert retrieved[0]["open"] == 170.0
        assert retrieved[0]["high"] == 170.8
        assert retrieved[0]["interval"] == "1m"
        assert retrieved[1]["close"] == 170.2
        assert retrieved[1]["volume"] == 12000

    @pytest.mark.asyncio
    async def test_no_snapshots_returns_empty(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        result = await db.get_price_snapshots(tmp_db_path, 9999)
        assert result == []

    @pytest.mark.asyncio
    async def test_default_volume_zero(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        sig_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )

        bars = [
            {
                "timestamp": "2026-03-27T09:30:00",
                "open": 170.0,
                "high": 170.8,
                "low": 169.5,
                "close": 170.5,
                # no volume key
            },
        ]
        await db.save_price_snapshots(
            tmp_db_path, signal_id=sig_id, ticker="NVDA", bars=bars
        )

        retrieved = await db.get_price_snapshots(tmp_db_path, sig_id)
        assert retrieved[0]["volume"] == 0


# ---------------------------------------------------------------------------
# save_signal_outcome / get_outcome
# ---------------------------------------------------------------------------


class TestSignalOutcome:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_outcome(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        sig_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )

        outcome_id = await db.save_signal_outcome(
            tmp_db_path,
            outcome={
                "signal_id": sig_id,
                "outcome": "t1_hit",
                "hit_price": 168.0,
                "hit_time": "2026-03-27T10:15:00",
                "pnl_underlying_pct": 1.18,
                "pnl_atm_est": 35.0,
                "pnl_otm_est": 65.0,
                "max_favorable_pct": 1.5,
                "max_adverse_pct": 0.3,
            },
        )
        assert outcome_id is not None

        row = await db.get_outcome(tmp_db_path, sig_id)
        assert row is not None
        assert row["outcome"] == "t1_hit"
        assert row["hit_price"] == 168.0
        assert row["hit_time"] == "2026-03-27T10:15:00"
        assert row["pnl_underlying_pct"] == 1.18
        assert row["pnl_atm_est"] == 35.0
        assert row["pnl_otm_est"] == 65.0
        assert row["max_favorable_pct"] == 1.5
        assert row["max_adverse_pct"] == 0.3
        assert row["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_get_outcome_nonexistent(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        assert await db.get_outcome(tmp_db_path, 9999) is None

    @pytest.mark.asyncio
    async def test_outcome_defaults(self, tmp_db_path):
        """Ensure default values are applied for optional fields."""
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)
        sig_id = await db.save_trade_signal(
            tmp_db_path, message_id=msg_id, signal=_signal_dict()
        )

        await db.save_signal_outcome(
            tmp_db_path,
            outcome={
                "signal_id": sig_id,
                "outcome": "expired",
                # no hit_price, hit_time, pnl_atm_est, pnl_otm_est
            },
        )

        row = await db.get_outcome(tmp_db_path, sig_id)
        assert row["hit_price"] is None
        assert row["hit_time"] is None
        assert row["pnl_underlying_pct"] == 0
        assert row["pnl_atm_est"] is None
        assert row["pnl_otm_est"] is None
        assert row["max_favorable_pct"] == 0
        assert row["max_adverse_pct"] == 0


# ---------------------------------------------------------------------------
# get_outcomes_by_bot (join)
# ---------------------------------------------------------------------------


class TestOutcomesByBot:
    @pytest.mark.asyncio
    async def test_joins_signal_and_outcome(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        sig1_id = await db.save_trade_signal(
            tmp_db_path,
            message_id=msg_id,
            signal=_signal_dict(bot_source="Captain Hook", ticker="NVDA"),
        )
        sig2_id = await db.save_trade_signal(
            tmp_db_path,
            message_id=msg_id,
            signal=_signal_dict(bot_source="Neverland Pan", ticker="TSLA"),
        )

        await db.save_signal_outcome(
            tmp_db_path,
            outcome={"signal_id": sig1_id, "outcome": "t2_hit", "pnl_underlying_pct": 1.76},
        )
        await db.save_signal_outcome(
            tmp_db_path,
            outcome={"signal_id": sig2_id, "outcome": "stop_hit", "pnl_underlying_pct": -0.4},
        )

        hook_outcomes = await db.get_outcomes_by_bot(tmp_db_path, "Captain Hook")
        assert len(hook_outcomes) == 1
        assert hook_outcomes[0]["ticker"] == "NVDA"
        assert hook_outcomes[0]["outcome"] == "t2_hit"
        assert hook_outcomes[0]["bot_source"] == "Captain Hook"
        assert hook_outcomes[0]["pnl_underlying_pct"] == 1.76

        pan_outcomes = await db.get_outcomes_by_bot(tmp_db_path, "Neverland Pan")
        assert len(pan_outcomes) == 1
        assert pan_outcomes[0]["ticker"] == "TSLA"
        assert pan_outcomes[0]["outcome"] == "stop_hit"

    @pytest.mark.asyncio
    async def test_no_outcomes_returns_empty(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        result = await db.get_outcomes_by_bot(tmp_db_path, "Captain Hook")
        assert result == []


# ---------------------------------------------------------------------------
# save_smee_performance / get_smee_performance
# ---------------------------------------------------------------------------


class TestSmeePerformance:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path, author="Smee")

        perf_id = await db.save_smee_performance(
            tmp_db_path,
            message_id=msg_id,
            date="2026-03-27",
            perf={
                "wins": 6,
                "losses": 1,
                "win_rate_pct": 86.0,
                "avg_pnl_pct": 0.89,
                "all_time_wins": 8,
                "all_time_total": 10,
                "trades": [
                    {"ticker": "QQQ", "pnl_pct": 0.99},
                    {"ticker": "AMD", "pnl_pct": -0.07},
                ],
            },
        )
        assert perf_id is not None

        rows = await db.get_smee_performance(tmp_db_path)
        assert len(rows) == 1
        assert rows[0]["wins"] == 6
        assert rows[0]["losses"] == 1
        assert rows[0]["win_rate_pct"] == 86.0
        assert rows[0]["avg_pnl_pct"] == 0.89
        assert rows[0]["all_time_wins"] == 8
        assert rows[0]["all_time_total"] == 10
        assert len(rows[0]["trades_json"]) == 2

    @pytest.mark.asyncio
    async def test_filter_by_date(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        await db.save_smee_performance(
            tmp_db_path,
            message_id=msg_id,
            date="2026-03-26",
            perf={"wins": 3, "losses": 2, "win_rate_pct": 60.0, "avg_pnl_pct": 0.30, "trades": []},
        )
        await db.save_smee_performance(
            tmp_db_path,
            message_id=msg_id,
            date="2026-03-27",
            perf={"wins": 6, "losses": 1, "win_rate_pct": 86.0, "avg_pnl_pct": 0.89, "trades": []},
        )

        rows = await db.get_smee_performance(tmp_db_path, date="2026-03-27")
        assert len(rows) == 1
        assert rows[0]["win_rate_pct"] == 86.0

    @pytest.mark.asyncio
    async def test_no_performance_returns_empty(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        rows = await db.get_smee_performance(tmp_db_path)
        assert rows == []

    @pytest.mark.asyncio
    async def test_all_dates_ordered_desc(self, tmp_db_path):
        await db.init_db(tmp_db_path)
        msg_id = await _insert_msg(tmp_db_path)

        await db.save_smee_performance(
            tmp_db_path,
            message_id=msg_id,
            date="2026-03-25",
            perf={"wins": 2, "losses": 3, "win_rate_pct": 40.0, "avg_pnl_pct": -0.10, "trades": []},
        )
        await db.save_smee_performance(
            tmp_db_path,
            message_id=msg_id,
            date="2026-03-27",
            perf={"wins": 6, "losses": 1, "win_rate_pct": 86.0, "avg_pnl_pct": 0.89, "trades": []},
        )

        rows = await db.get_smee_performance(tmp_db_path)
        assert len(rows) == 2
        # Should be ordered by date DESC
        assert rows[0]["date"] == "2026-03-27"
        assert rows[1]["date"] == "2026-03-25"
