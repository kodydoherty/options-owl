"""Read-only PG queries for the dashboard. Never writes to trading tables."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import asyncpg

_ET = ZoneInfo("America/New_York")


def _closed_filter(
    agent_id: str,
    days: int | None = None,
    ticker: str | None = None,
    live_only: bool = True,
) -> tuple[str, list]:
    """Build a shared WHERE fragment (+ ordered params, agent_id as $1) for closed-trade
    queries so the range / ticker / live-only filters scope EVERY stat and chart uniformly.

    - live_only=True → only trades that actually executed on Webull (webull_order_id set);
      paper-only fills are excluded so the numbers match real money.
    - days == 1 → the ET *calendar* day (matches the fleet's "today"), not a rolling 24h.
    """
    conds = ["agent_id = $1", "status = 'closed'"]
    params: list = [agent_id]
    if live_only:
        conds.append("webull_order_id IS NOT NULL")
    if days is not None:
        if days == 1:
            since = (
                datetime.now(tz=_ET)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .astimezone(timezone.utc)
            )
        else:
            since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        params.append(since)
        conds.append(f"closed_at >= ${len(params)}")
    if ticker:
        params.append(ticker.upper())
        conds.append(f"ticker = ${len(params)}")
    return " AND ".join(conds), params


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
    ticker: str | None = None,
    live_only: bool = True,
) -> list[dict]:
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    params.append(limit)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT * FROM trades WHERE {where}
                ORDER BY closed_at DESC LIMIT ${len(params)}""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_distinct_tickers(pool: asyncpg.Pool, agent_id: str) -> list[str]:
    """Tickers this agent has ever traded — populates the dashboard ticker filter."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM trades WHERE agent_id = $1 ORDER BY ticker",
            agent_id,
        )
        return [r["ticker"] for r in rows]


async def get_candles(
    pool: asyncpg.Pool, ticker: str, timeframe: str = "5m", limit: int = 200
) -> list[dict]:
    """Underlying OHLC candles from the harvester's shared feed (market data — NOT agent
    scoped). Oldest-first for charting."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT bar_time, open, high, low, close, volume, vwap
               FROM stock_candles
               WHERE ticker = $1 AND timeframe = $2
               ORDER BY bar_time DESC
               LIMIT $3""",
            ticker.upper(), timeframe, limit,
        )
        return [dict(r) for r in reversed(rows)]


async def get_ticker_trades(
    pool: asyncpg.Pool, agent_id: str, ticker: str, days: int = 5
) -> list[dict]:
    """This agent's entries/exits on one ticker, for overlaying markers on the candle chart.
    Agent-scoped — you only ever see your own owlet's trades."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT sqlite_id, direction, status, opened_at, closed_at,
                      premium_per_contract, exit_premium, pnl_dollars, exit_reason, contracts
               FROM trades
               WHERE agent_id = $1 AND ticker = $2 AND opened_at >= $3
               ORDER BY opened_at ASC""",
            agent_id, ticker.upper(), since,
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


async def get_recent_events(
    pool: asyncpg.Pool, agent_id: str, limit: int = 100,
    event_type: str | None = None,
) -> list[dict]:
    """This agent's most recent trade-lifecycle events (all trades), newest first.

    Joined to ``trades`` for the ticker so the feed is readable. Scoped to agent_id — a
    user only ever sees their own owlet's events.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT e.trade_id, e.event_type, e.details, e.created_at,
                      t.ticker, t.direction, t.status
               FROM trade_events e
               LEFT JOIN trades t
                 ON t.agent_id = e.agent_id AND t.sqlite_id = e.trade_id
               WHERE e.agent_id = $1
                 AND ($3::text IS NULL OR e.event_type = $3)
               ORDER BY e.created_at DESC
               LIMIT $2""",
            agent_id, limit, event_type,
        )
        return [dict(r) for r in rows]


async def get_recent_signals(
    pool: asyncpg.Pool, agent_id: str, limit: int = 100
) -> list[dict]:
    """Recent ML signals from the shared fleet-wide pool, newest first.

    ml_signals is NOT per-account (the harvester emits, every bot consumes the same market
    observations) so there's nothing account-private to leak. ``consumed_by_me`` flags the ones
    this agent actually picked up, for relevance.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT ticker, direction, score, ml_confidence, ml_threshold,
                      ml_model_source, ml_runner_score, premium, strike, expiry_date,
                      emitted_at, status,
                      ($1 = ANY(consumed_by)) AS consumed_by_me
               FROM ml_signals
               ORDER BY emitted_at DESC
               LIMIT $2""",
            agent_id, limit,
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


async def get_fleet_overview(pool: asyncpg.Pool) -> list[dict]:
    """One row per agent for the ADMIN fleet view — portfolio size, today's P&L, open
    positions, all-time win rate, and heartbeat staleness. Not agent-scoped by design; the
    route gates this behind ``is_admin`` (kody only).

    LIVE trades only: every aggregate filters ``webull_order_id IS NOT NULL`` so the fleet shows
    real-money performance, and paper shadow bots (no Webull fills) drop out entirely.

    Today's P&L / count are recomputed from the trades table on the ET calendar day (more
    authoritative than agent_state.daily_pnl, which the bot writes opportunistically).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT
                 s.agent_id,
                 s.portfolio_size,
                 s.last_heartbeat,
                 COALESCE(o.open_count, 0)     AS open_count,
                 COALESCE(d.daily_pnl, 0)      AS daily_pnl,
                 COALESCE(d.today_trades, 0)   AS today_trades,
                 COALESCE(c.total_trades, 0)   AS total_trades,
                 COALESCE(c.wins, 0)           AS wins,
                 COALESCE(c.total_pnl, 0)      AS total_pnl
               FROM agent_state s
               LEFT JOIN (
                 SELECT agent_id, COUNT(*) AS open_count
                 FROM trades WHERE status = 'open' AND webull_order_id IS NOT NULL
                 GROUP BY agent_id
               ) o ON o.agent_id = s.agent_id
               LEFT JOIN (
                 SELECT agent_id,
                        SUM(pnl_dollars) AS daily_pnl,
                        COUNT(*) AS today_trades
                 FROM trades
                 WHERE status = 'closed' AND webull_order_id IS NOT NULL
                   AND (closed_at AT TIME ZONE 'America/New_York')::date
                       = (NOW() AT TIME ZONE 'America/New_York')::date
                 GROUP BY agent_id
               ) d ON d.agent_id = s.agent_id
               LEFT JOIN (
                 SELECT agent_id,
                        COUNT(*) AS total_trades,
                        COUNT(*) FILTER (WHERE pnl_dollars > 0) AS wins,
                        SUM(pnl_dollars) AS total_pnl
                 FROM trades WHERE status = 'closed' AND webull_order_id IS NOT NULL
                 GROUP BY agent_id
               ) c ON c.agent_id = s.agent_id
               ORDER BY s.agent_id""",
        )
        out = []
        for r in rows:
            d = dict(r)
            total = d["total_trades"] or 0
            # Live bots only — a bot with zero Webull-executed trades is a paper shadow.
            if total == 0:
                continue
            d["win_rate"] = round(d["wins"] / total * 100, 1) if total else 0.0
            out.append(d)
        return out


async def get_portfolio_stats(
    pool: asyncpg.Pool,
    agent_id: str,
    days: int | None = None,
    ticker: str | None = None,
    live_only: bool = True,
) -> dict:
    """Headline stats — scoped to the selected range + ticker + live-only so the WHOLE
    dashboard updates with the filter (the totals/win-rate/best cards, not just the trade list).
    The 'today' card stays the ET-calendar day regardless of the range."""
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    live_clause = "AND webull_order_id IS NOT NULL" if live_only else ""
    today = (
        datetime.now(tz=_ET)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(timezone.utc)
    )
    async with pool.acquire() as conn:
        open_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM trades WHERE agent_id = $1 AND status = 'open' {live_clause}",
            agent_id,
        )
        today_stats = await conn.fetchrow(
            f"""SELECT COALESCE(SUM(pnl_dollars), 0) as daily_pnl,
                       COUNT(*) as today_trades,
                       COUNT(*) FILTER (WHERE pnl_dollars > 0) as today_wins
                FROM trades
                WHERE agent_id = $1 AND status = 'closed' {live_clause} AND closed_at >= $2""",
            agent_id, today,
        )
        stats = await conn.fetchrow(
            f"""SELECT COUNT(*) as total_trades,
                       COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                       COUNT(*) FILTER (WHERE pnl_dollars <= 0) as losses,
                       COALESCE(SUM(pnl_dollars), 0) as total_pnl,
                       COALESCE(AVG(pnl_dollars), 0) as avg_trade,
                       COALESCE(MAX(pnl_dollars), 0) as best_trade,
                       COALESCE(MIN(pnl_dollars), 0) as worst_trade
                FROM trades WHERE {where}""",
            *params,
        )
        total = stats["total_trades"] if stats else 0
        wins = stats["wins"] if stats else 0
        win_rate = (wins / total * 100) if total > 0 else 0
        return {
            "open_count": open_count or 0,
            "daily_pnl": float(today_stats["daily_pnl"]) if today_stats else 0,
            "today_trades": today_stats["today_trades"] if today_stats else 0,
            "today_wins": today_stats["today_wins"] if today_stats else 0,
            "total_trades": total,
            "total_wins": wins,
            "total_losses": stats["losses"] if stats else 0,
            "total_pnl": float(stats["total_pnl"]) if stats else 0,
            "avg_trade": float(stats["avg_trade"]) if stats else 0,
            "best_trade": float(stats["best_trade"]) if stats else 0,
            "worst_trade": float(stats["worst_trade"]) if stats else 0,
            "win_rate": round(win_rate, 1),
        }


# ---------------------------------------------------------------------------
# Premium ticks (for trade detail chart)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Analytics queries
# ---------------------------------------------------------------------------


async def get_pnl_curve(
    pool: asyncpg.Pool, agent_id: str, days: int = 30,
    ticker: str | None = None, live_only: bool = True,
) -> list[dict]:
    """Cumulative P&L over time, one point per closed trade."""
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT closed_at, pnl_dollars,
                       SUM(pnl_dollars) OVER (ORDER BY closed_at) as cumulative_pnl
                FROM trades WHERE {where}
                ORDER BY closed_at ASC""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_daily_pnl(
    pool: asyncpg.Pool, agent_id: str, days: int = 30,
    ticker: str | None = None, live_only: bool = True,
) -> list[dict]:
    """Daily P&L aggregated by date."""
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT DATE(closed_at) as trade_date,
                       COUNT(*) as trades,
                       COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                       COALESCE(SUM(pnl_dollars), 0) as daily_pnl
                FROM trades WHERE {where}
                GROUP BY DATE(closed_at)
                ORDER BY trade_date ASC""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_exit_distribution(
    pool: asyncpg.Pool, agent_id: str, days: int = 30,
    ticker: str | None = None, live_only: bool = True,
) -> list[dict]:
    """Count of trades per exit reason."""
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT COALESCE(exit_reason, 'unknown') as exit_reason,
                       COUNT(*) as count,
                       COALESCE(SUM(pnl_dollars), 0) as total_pnl,
                       COALESCE(AVG(pnl_dollars), 0) as avg_pnl
                FROM trades WHERE {where}
                GROUP BY exit_reason
                ORDER BY count DESC""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_ticker_performance(
    pool: asyncpg.Pool, agent_id: str, days: int = 30, live_only: bool = True,
) -> list[dict]:
    """P&L breakdown by ticker."""
    where, params = _closed_filter(agent_id, days=days, live_only=live_only)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT ticker,
                       COUNT(*) as trades,
                       COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                       COALESCE(SUM(pnl_dollars), 0) as total_pnl,
                       COALESCE(AVG(pnl_dollars), 0) as avg_pnl,
                       COALESCE(MAX(pnl_dollars), 0) as best_trade,
                       COALESCE(MIN(pnl_dollars), 0) as worst_trade
                FROM trades WHERE {where}
                GROUP BY ticker
                ORDER BY total_pnl DESC""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_hourly_performance(
    pool: asyncpg.Pool, agent_id: str, days: int = 30, live_only: bool = True,
) -> list[dict]:
    """Win rate and P&L by hour of day (ET)."""
    where, params = _closed_filter(agent_id, days=days, live_only=live_only)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT EXTRACT(HOUR FROM opened_at AT TIME ZONE 'America/New_York') as hour,
                       COUNT(*) as trades,
                       COUNT(*) FILTER (WHERE pnl_dollars > 0) as wins,
                       COALESCE(SUM(pnl_dollars), 0) as total_pnl
                FROM trades WHERE {where}
                GROUP BY hour
                ORDER BY hour ASC""",
            *params,
        )
        return [dict(r) for r in rows]


async def get_trade_duration_stats(
    pool: asyncpg.Pool, agent_id: str, days: int = 30,
    ticker: str | None = None, live_only: bool = True,
) -> dict:
    """Average hold time for winners vs losers (+ avg/best/worst trade)."""
    where, params = _closed_filter(agent_id, days=days, ticker=ticker, live_only=live_only)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""SELECT
                  AVG(hold_minutes) FILTER (WHERE pnl_dollars > 0) as avg_win_hold,
                  AVG(hold_minutes) FILTER (WHERE pnl_dollars <= 0) as avg_loss_hold,
                  AVG(pnl_pct) FILTER (WHERE pnl_dollars > 0) as avg_win_pct,
                  AVG(pnl_pct) FILTER (WHERE pnl_dollars <= 0) as avg_loss_pct,
                  MAX(pnl_dollars) as best_trade,
                  MIN(pnl_dollars) as worst_trade,
                  AVG(pnl_dollars) as avg_trade
                FROM trades WHERE {where} AND hold_minutes IS NOT NULL""",
            *params,
        )
        return dict(row) if row else {}


async def get_premium_ticks(
    pool: asyncpg.Pool, agent_id: str, trade_id: int
) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT captured_at, premium, bid, ask, underlying_price,
                      fsm_state, gain_pct, peak_gain_pct, active_gate
               FROM trade_premium_ticks
               WHERE agent_id = $1 AND trade_id = $2
               ORDER BY captured_at ASC""",
            agent_id, trade_id,
        )
        return [dict(r) for r in rows]
