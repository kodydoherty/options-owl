"""Analyst / bot performance tracking and filtering."""

from __future__ import annotations

import aiosqlite

from options_owl.journal.db import connect as _connect_db


async def get_bot_stats(db_path: str, bot_source: str) -> dict:
    """Query historical performance of a bot from the paper_trades table.

    Returns a dict with keys:
        total_trades, wins, losses, win_rate, avg_pnl_pct,
        recent_win_rate (last 10 trades).
    """
    stats: dict = {
        "bot_source": bot_source,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "recent_win_rate": 0.0,
    }

    async with _connect_db(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        # All-time stats
        cursor = await conn.execute(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN pnl_pct >= 0 THEN 1 ELSE 0 END) AS wins, "
            "  SUM(CASE WHEN pnl_pct < 0 THEN 1 ELSE 0 END) AS losses, "
            "  AVG(pnl_pct) AS avg_pnl "
            "FROM paper_trades "
            "WHERE status = 'closed' AND bot_source = ?",
            (bot_source,),
        )
        row = await cursor.fetchone()
        if row is None:
            return stats

        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        avg_pnl = row["avg_pnl"] or 0.0

        stats["total_trades"] = total
        stats["wins"] = wins
        stats["losses"] = losses
        stats["win_rate"] = (wins / total * 100.0) if total > 0 else 0.0
        stats["avg_pnl_pct"] = round(avg_pnl, 2)

        # Recent trend: last 10 closed trades
        cursor = await conn.execute(
            "SELECT pnl_pct FROM paper_trades "
            "WHERE status = 'closed' AND bot_source = ? "
            "ORDER BY closed_at DESC LIMIT 10",
            (bot_source,),
        )
        recent_rows = await cursor.fetchall()
        if recent_rows:
            recent_wins = sum(1 for r in recent_rows if (r["pnl_pct"] or 0) >= 0)
            stats["recent_win_rate"] = round(
                recent_wins / len(recent_rows) * 100.0, 1
            )

    return stats


async def check_analyst_filter(
    db_path: str,
    bot_source: str,
    settings: object,
) -> tuple[bool, str, dict]:
    """Check whether a bot passes the analyst performance filter.

    Args:
        db_path: Path to the sqlite database.
        bot_source: The bot identifier string.
        settings: A ``Settings`` instance with analyst filter fields.

    Returns:
        (passes, reason, stats) — *passes* is True if the signal should be
        accepted, *reason* explains a rejection, *stats* is the raw stats dict.
    """
    if not getattr(settings, "ENABLE_ANALYST_FILTER", False):
        return True, "Analyst filter disabled", {}

    stats = await get_bot_stats(db_path, bot_source)

    min_trades: int = getattr(settings, "ANALYST_MIN_TRADES", 10)
    min_win_rate: float = getattr(settings, "ANALYST_MIN_WIN_RATE", 0.0)

    # Not enough data — let it through
    if stats["total_trades"] < min_trades:
        return True, (
            f"Not enough data ({stats['total_trades']}/{min_trades} trades)"
        ), stats

    # No filter threshold set
    if min_win_rate <= 0:
        return True, "No minimum win rate configured", stats

    # Check win rate
    if stats["win_rate"] < min_win_rate:
        return False, (
            f"Bot {bot_source} win rate {stats['win_rate']:.1f}% "
            f"< required {min_win_rate:.1f}% "
            f"({stats['total_trades']} trades)"
        ), stats

    return True, (
        f"Bot {bot_source} passes: {stats['win_rate']:.1f}% win rate "
        f"({stats['total_trades']} trades)"
    ), stats


def format_bot_stats(stats: dict) -> str:
    """Return a human-readable summary of bot performance stats."""
    if not stats or stats.get("total_trades", 0) == 0:
        return f"Bot {stats.get('bot_source', '?')}: no closed trades"

    return (
        f"Bot {stats['bot_source']}: "
        f"{stats['total_trades']} trades | "
        f"{stats['wins']}W/{stats['losses']}L | "
        f"Win rate: {stats['win_rate']:.1f}% | "
        f"Avg P&L: {stats['avg_pnl_pct']:+.2f}% | "
        f"Recent(10): {stats['recent_win_rate']:.1f}%"
    )
