"""Dynamic Kelly Criterion position sizing.

Computes optimal position sizes based on historical bot performance,
with conservative defaults and drawdown protection.
"""

from __future__ import annotations

from loguru import logger

from options_owl.journal.db import connect as _connect_db

from options_owl.models.signals import BotSource
from options_owl.signals.analyst_tracker import get_bot_stats


def compute_kelly_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """Return the raw Kelly fraction given win/loss statistics.

    Args:
        win_rate: Probability of winning (0.0 to 1.0).
        avg_win_pct: Average win size as a positive percentage (e.g. 30.0).
        avg_loss_pct: Average loss size as a positive percentage (e.g. 20.0).

    Returns:
        Raw Kelly fraction (can be negative if the edge is negative).
    """
    if avg_loss_pct <= 0 or avg_win_pct <= 0:
        return 0.0

    # Kelly formula: f* = (p * b - q) / b
    # where p = win_rate, q = 1 - p, b = avg_win / avg_loss
    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return kelly


async def compute_dynamic_position_pct(
    db_path: str,
    bot_source: str,
    settings: object,
) -> float:
    """Compute the dynamic position size percentage using Kelly Criterion.

    Steps:
        1. Fetch bot stats from analyst_tracker.
        2. If not enough data (< KELLY_MIN_TRADES), fall back to MAX_POSITION_PCT.
        3. Compute full Kelly, then apply KELLY_FRACTION (fractional Kelly).
        4. Clamp between KELLY_MIN_PCT and KELLY_MAX_PCT.
        5. Apply drawdown reduction if portfolio is down significantly.

    Returns:
        Position size as a percentage of the portfolio.
    """
    kelly_min_trades: int = getattr(settings, "KELLY_MIN_TRADES", 20)
    kelly_fraction: float = getattr(settings, "KELLY_FRACTION", 0.25)
    kelly_min_pct: float = getattr(settings, "KELLY_MIN_PCT", 5.0)
    kelly_max_pct: float = getattr(settings, "KELLY_MAX_PCT", 25.0)
    max_position_pct: float = getattr(settings, "MAX_POSITION_PCT", 5.0)
    drawdown_halve_pct: float = getattr(settings, "KELLY_DRAWDOWN_HALVE_PCT", 10.0)
    portfolio_size: float = getattr(settings, "PORTFOLIO_SIZE", 0.0)

    stats = await get_bot_stats(db_path, bot_source)

    if stats["total_trades"] < kelly_min_trades:
        logger.debug(
            f"Kelly: {bot_source} has {stats['total_trades']}/{kelly_min_trades} trades, "
            f"using fallback {max_position_pct}%"
        )
        return max_position_pct

    win_rate = stats["win_rate"] / 100.0  # convert from percentage to fraction

    # Separate avg win and avg loss from closed trades
    avg_win_pct, avg_loss_pct = await _get_avg_win_loss(db_path, bot_source)

    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        logger.debug(f"Kelly: {bot_source} missing win/loss data, using fallback")
        return max_position_pct

    raw_kelly = compute_kelly_fraction(win_rate, avg_win_pct, avg_loss_pct)

    if raw_kelly <= 0:
        logger.info(
            f"Kelly: {bot_source} has negative edge (kelly={raw_kelly:.4f}), "
            f"using minimum {kelly_min_pct}%"
        )
        return kelly_min_pct

    # Apply fractional Kelly (e.g. quarter Kelly)
    position_pct = raw_kelly * kelly_fraction * 100.0

    # Clamp to configured bounds
    position_pct = max(kelly_min_pct, min(kelly_max_pct, position_pct))

    # Drawdown reduction: halve sizing if portfolio is down more than threshold
    if portfolio_size > 0:
        current_balance = await _get_current_balance(db_path, portfolio_size)
        drawdown_pct = (portfolio_size - current_balance) / portfolio_size * 100.0
        if drawdown_pct > drawdown_halve_pct:
            position_pct *= 0.5
            position_pct = max(kelly_min_pct, position_pct)
            logger.info(
                f"Kelly: drawdown {drawdown_pct:.1f}% > {drawdown_halve_pct}%, "
                f"halving position to {position_pct:.1f}%"
            )

    logger.info(
        f"Kelly: {bot_source} — win_rate={win_rate:.2f} "
        f"avg_win={avg_win_pct:.1f}% avg_loss={avg_loss_pct:.1f}% "
        f"raw_kelly={raw_kelly:.4f} → position={position_pct:.1f}%"
    )
    return position_pct


async def get_kelly_summary(db_path: str, settings: object) -> str:
    """Return a formatted string showing Kelly sizing per bot."""
    if not getattr(settings, "ENABLE_KELLY_SIZING", False):
        return "Kelly sizing: DISABLED"

    lines = ["=== Kelly Position Sizing ==="]

    for bot in BotSource:
        if bot == BotSource.UNKNOWN:
            continue
        try:
            pct = await compute_dynamic_position_pct(db_path, bot.value, settings)
            stats = await get_bot_stats(db_path, bot.value)
            trades = stats["total_trades"]
            win_rate = stats["win_rate"]
            lines.append(
                f"  {bot.value}: {pct:.1f}% "
                f"({trades} trades, {win_rate:.1f}% win rate)"
            )
        except Exception as exc:
            lines.append(f"  {bot.value}: error — {exc}")

    lines.append("=" * 28)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _get_avg_win_loss(
    db_path: str, bot_source: str
) -> tuple[float, float]:
    """Return (avg_win_pct, avg_loss_pct) for a bot's closed trades.

    Both values are returned as positive numbers.
    """
    async with _connect_db(db_path) as conn:
        cursor = await conn.execute(
            "SELECT AVG(pnl_pct) FROM paper_trades "
            "WHERE status = 'closed' AND bot_source = ? AND pnl_pct >= 0",
            (bot_source,),
        )
        row = await cursor.fetchone()
        avg_win = float(row[0]) if row and row[0] is not None else 0.0  # type: ignore[index]

        cursor = await conn.execute(
            "SELECT AVG(pnl_pct) FROM paper_trades "
            "WHERE status = 'closed' AND bot_source = ? AND pnl_pct < 0",
            (bot_source,),
        )
        row = await cursor.fetchone()
        avg_loss = abs(float(row[0])) if row and row[0] is not None else 0.0  # type: ignore[index]

    return avg_win, avg_loss


async def _get_current_balance(db_path: str, default: float) -> float:
    """Read the current portfolio balance from the paper_portfolio table."""
    try:
        async with _connect_db(db_path) as conn:
            cursor = await conn.execute(
                "SELECT current_balance FROM paper_portfolio ORDER BY id DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            return float(row[0]) if row else default  # type: ignore[index]
    except Exception:
        return default
