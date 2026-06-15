"""Read-only PG queries for the dashboard. Never writes to trading tables."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncpg


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


async def get_open_trades(pool: asyncpg.Pool, agent_id: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM trades
               WHERE agent_id = $1 AND status = 'open'
               ORDER BY opened_at DESC""",
            agent_id,
        )
        return [dict(r) for r in rows]


async def get_closed_trades(
    pool: asyncpg.Pool,
    agent_id: str,
    days: int = 7,
    limit: int = 100,
) -> list[dict]:
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               ORDER BY closed_at DESC
               LIMIT $3""",
            agent_id, since, limit,
        )
        return [dict(r) for r in rows]


async def get_trade_by_id(
    pool: asyncpg.Pool, agent_id: str, trade_id: int
) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT * FROM trades
               WHERE agent_id = $1 AND (id = $2 OR sqlite_id = $2)""",
            agent_id, trade_id,
        )
        return dict(row) if row else None


async def get_trade_events(
    pool: asyncpg.Pool, agent_id: str, trade_id: int
) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM trade_events
               WHERE agent_id = $1 AND trade_id = $2
               ORDER BY created_at ASC""",
            agent_id, trade_id,
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Portfolio / Agent State
# ---------------------------------------------------------------------------


async def get_agent_state(pool: asyncpg.Pool, agent_id: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM agent_state WHERE agent_id = $1", agent_id
        )
        return dict(row) if row else None


async def get_portfolio_stats(pool: asyncpg.Pool, agent_id: str) -> dict:
    """Compute portfolio stats from trades table."""
    async with pool.acquire() as conn:
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        open_count = await conn.fetchval(
            "SELECT COUNT(*) FROM trades WHERE agent_id = $1 AND status = 'open'",
            agent_id,
        )

        today_stats = await conn.fetchrow(
            """SELECT
                 COALESCE(SUM(pnl_dollars), 0) as daily_pnl,
                 COUNT(*) as today_trades,
                 COUNT(*) FILTER (WHERE pnl_dollars > 0) as today_wins
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2""",
            agent_id, today,
        )

        all_stats = await conn.fetchrow(
            """SELECT
                 COUNT(*) as total_trades,
                 COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                 COUNT(*) FILTER (WHERE pnl_dollars <= 0) as losses,
                 COALESCE(SUM(pnl_dollars), 0) as total_pnl
               FROM trades
               WHERE agent_id = $1 AND status = 'closed'""",
            agent_id,
        )

        total = all_stats["total_trades"] if all_stats else 0
        wins = all_stats["wins"] if all_stats else 0
        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "open_count": open_count or 0,
            "daily_pnl": float(today_stats["daily_pnl"]) if today_stats else 0,
            "today_trades": today_stats["today_trades"] if today_stats else 0,
            "today_wins": today_stats["today_wins"] if today_stats else 0,
            "total_trades": total,
            "total_wins": wins,
            "total_losses": all_stats["losses"] if all_stats else 0,
            "total_pnl": float(all_stats["total_pnl"]) if all_stats else 0,
            "win_rate": round(win_rate, 1),
        }


# ---------------------------------------------------------------------------
# Premium ticks (for trade detail chart)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Analytics queries
# ---------------------------------------------------------------------------


async def get_pnl_curve(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> list[dict]:
    """Cumulative P&L over time, one point per closed trade."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT closed_at, pnl_dollars,
                      SUM(pnl_dollars) OVER (ORDER BY closed_at) as cumulative_pnl
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               ORDER BY closed_at ASC""",
            agent_id, since,
        )
        return [dict(r) for r in rows]


async def get_daily_pnl(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> list[dict]:
    """Daily P&L aggregated by date."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT DATE(closed_at) as trade_date,
                      COUNT(*) as trades,
                      COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                      COALESCE(SUM(pnl_dollars), 0) as daily_pnl
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               GROUP BY DATE(closed_at)
               ORDER BY trade_date ASC""",
            agent_id, since,
        )
        return [dict(r) for r in rows]


async def get_exit_distribution(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> list[dict]:
    """Count of trades per exit reason."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT COALESCE(exit_reason, 'unknown') as exit_reason,
                      COUNT(*) as count,
                      COALESCE(SUM(pnl_dollars), 0) as total_pnl,
                      COALESCE(AVG(pnl_dollars), 0) as avg_pnl
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               GROUP BY exit_reason
               ORDER BY count DESC""",
            agent_id, since,
        )
        return [dict(r) for r in rows]


async def get_ticker_performance(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> list[dict]:
    """P&L breakdown by ticker."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT ticker,
                      COUNT(*) as trades,
                      COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                      COALESCE(SUM(pnl_dollars), 0) as total_pnl,
                      COALESCE(AVG(pnl_dollars), 0) as avg_pnl,
                      COALESCE(MAX(pnl_dollars), 0) as best_trade,
                      COALESCE(MIN(pnl_dollars), 0) as worst_trade
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               GROUP BY ticker
               ORDER BY total_pnl DESC""",
            agent_id, since,
        )
        return [dict(r) for r in rows]


async def get_hourly_performance(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> list[dict]:
    """Win rate and P&L by hour of day (ET approximation via UTC-4)."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT EXTRACT(HOUR FROM opened_at AT TIME ZONE 'America/New_York') as hour,
                      COUNT(*) as trades,
                      COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                      COALESCE(SUM(pnl_dollars), 0) as total_pnl
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
               GROUP BY hour
               ORDER BY hour ASC""",
            agent_id, since,
        )
        return [dict(r) for r in rows]


async def get_trade_duration_stats(
    pool: asyncpg.Pool, agent_id: str, days: int = 30
) -> dict:
    """Average hold time for winners vs losers."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT
                 AVG(hold_minutes) FILTER (WHERE pnl_dollars > 0) as avg_win_hold,
                 AVG(hold_minutes) FILTER (WHERE pnl_dollars <= 0) as avg_loss_hold,
                 AVG(pnl_pct) FILTER (WHERE pnl_dollars > 0) as avg_win_pct,
                 AVG(pnl_pct) FILTER (WHERE pnl_dollars <= 0) as avg_loss_pct,
                 MAX(pnl_dollars) as best_trade,
                 MIN(pnl_dollars) as worst_trade,
                 AVG(pnl_dollars) as avg_trade
               FROM trades
               WHERE agent_id = $1 AND status = 'closed' AND closed_at >= $2
                 AND hold_minutes IS NOT NULL""",
            agent_id, since,
        )
        return dict(row) if row else {}


async def get_premium_ticks(
    pool: asyncpg.Pool, agent_id: str, trade_id: int
) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT captured_at, premium, bid, ask, underlying_price
               FROM trade_premium_ticks
               WHERE agent_id = $1 AND trade_id = $2
               ORDER BY captured_at ASC""",
            agent_id, trade_id,
        )
        return [dict(r) for r in rows]
