from datetime import datetime, timezone

import pytest

from options_owl.journal import db
from options_owl.models.signals import BotSource
from options_owl.signals.performance_tracker import (
    compute_bot_performance,
    format_report,
)


@pytest.fixture
async def seeded_db(tmp_db_path):
    """Create a DB with signals and outcomes for testing."""
    await db.init_db(tmp_db_path)

    # Save two raw messages
    msg1 = await db.save_message(
        tmp_db_path,
        guild_id=1,
        channel_id=1,
        author_id=1,
        author_name="Captain Hook",
        content="signal 1",
        timestamp=datetime.now(timezone.utc),
    )
    msg2 = await db.save_message(
        tmp_db_path,
        guild_id=1,
        channel_id=1,
        author_id=1,
        author_name="Captain Hook",
        content="signal 2",
        timestamp=datetime.now(timezone.utc),
    )

    # Save two trade signals
    sig1 = await db.save_trade_signal(
        tmp_db_path,
        message_id=msg1,
        signal={
            "bot_source": "Captain Hook",
            "ticker": "NVDA",
            "sentiment": "bearish",
            "direction": "put",
            "score": 100,
            "strength": "strong",
            "entry_price": 170.0,
            "target_price": 167.0,
            "expected_move_pct": 0.9,
            "strike": 170.0,
            "expiry": "0DTE",
            "risk_reward": 1.5,
            "target_1": 168.0,
            "target_2": 167.0,
            "stop_price": 171.0,
            "is_elite": True,
            "key_signals": ["BB 2σ Touch", "EMA Bounce"],
        },
    )
    sig2 = await db.save_trade_signal(
        tmp_db_path,
        message_id=msg2,
        signal={
            "bot_source": "Captain Hook",
            "ticker": "AAPL",
            "sentiment": "bearish",
            "direction": "put",
            "score": 57,
            "strength": "marginal",
            "entry_price": 252.0,
            "target_price": 251.0,
            "expected_move_pct": 0.4,
            "strike": 252.5,
            "expiry": "0DTE",
            "risk_reward": 1.5,
            "target_1": 251.5,
            "target_2": 251.0,
            "stop_price": 253.0,
            "is_elite": False,
            "key_signals": ["MACD Bear Cross"],
        },
    )

    # Save outcomes: sig1 wins (T2), sig2 loses (stop)
    await db.save_signal_outcome(
        tmp_db_path,
        outcome={
            "signal_id": sig1,
            "outcome": "t2_hit",
            "hit_price": 167.0,
            "pnl_underlying_pct": 1.76,
            "pnl_atm_est": 52.0,
            "max_favorable_pct": 1.8,
            "max_adverse_pct": 0.1,
        },
    )
    await db.save_signal_outcome(
        tmp_db_path,
        outcome={
            "signal_id": sig2,
            "outcome": "stop_hit",
            "hit_price": 253.0,
            "pnl_underlying_pct": -0.4,
            "pnl_atm_est": -15.0,
            "max_favorable_pct": 0.2,
            "max_adverse_pct": 0.4,
        },
    )

    return tmp_db_path


class TestBotPerformance:
    @pytest.mark.asyncio
    async def test_compute_captain_hook(self, seeded_db):
        report = await compute_bot_performance(seeded_db, BotSource.CAPTAIN_HOOK)
        assert report.total_signals == 2
        assert report.resolved_signals == 2
        assert report.wins == 1
        assert report.losses == 1
        assert report.win_rate_pct == 50.0
        assert report.avg_pnl_pct != 0
        assert report.best_trade_pnl > 0
        assert report.worst_trade_pnl < 0

    @pytest.mark.asyncio
    async def test_empty_bot(self, seeded_db):
        report = await compute_bot_performance(seeded_db, BotSource.NEVERLAND_PAN)
        assert report.total_signals == 0
        assert report.resolved_signals == 0
        assert report.win_rate_pct == 0.0


class TestFormatReport:
    @pytest.mark.asyncio
    async def test_format_with_data(self, seeded_db):
        report = await compute_bot_performance(seeded_db, BotSource.CAPTAIN_HOOK)
        output = format_report([report])
        assert "Captain Hook" in output
        assert "1W / 1L" in output

    def test_format_empty(self):
        output = format_report([])
        assert "No data" in output
