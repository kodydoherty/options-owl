"""Emergency alert system — sends Discord DMs when critical trading events occur.

Alerts fire for:
- Exit failures (sell order rejected, premium lookup failed 3+ times)
- Expiry danger (open trade within N minutes of market close)
- Force-close events (trade closed at market price as last resort)
- Premium blackout (can't get exit premium for an open trade)
- Webull order errors (live trading only)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

import discord
from loguru import logger

if TYPE_CHECKING:
    from options_owl.config.settings import Settings

# Track per-trade alert state to avoid spamming
_alerted_trades: set[tuple[int, str]] = set()  # (trade_id, alert_type)


def _parse_alert_user_ids(settings: Settings) -> list[int]:
    raw = getattr(settings, "DISCORD_ALERT_USER_IDS", "") or ""
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


async def send_alert(
    client: discord.Client,
    settings: Settings,
    title: str,
    message: str,
    trade_id: int | None = None,
    alert_type: str = "generic",
    force: bool = False,
) -> None:
    """Send a DM alert to all configured alert users.

    Deduplicates by (trade_id, alert_type) to avoid spamming the same alert
    every poll cycle. Use force=True to bypass dedup.
    """
    if trade_id is not None and not force:
        key = (trade_id, alert_type)
        if key in _alerted_trades:
            return
        _alerted_trades.add(key)

    user_ids = _parse_alert_user_ids(settings)
    if not user_ids:
        # No alert users configured — just log it
        logger.warning(f"ALERT [{title}]: {message}")
        return

    embed = discord.Embed(
        title=f"🚨 {title}",
        description=message,
        color=discord.Color.red(),
        timestamp=datetime.utcnow(),
    )

    for uid in user_ids:
        try:
            user = await client.fetch_user(uid)
            await user.send(embed=embed)
            logger.info(f"Alert DM sent to user {uid}: {title}")
        except Exception as exc:
            logger.error(f"Failed to DM alert to user {uid}: {exc}")


async def alert_premium_blackout(
    client: discord.Client,
    settings: Settings,
    trade: dict,
    attempts: int,
) -> None:
    """Alert when we can't get an exit premium for an open trade."""
    await send_alert(
        client, settings,
        title="Premium Blackout",
        message=(
            f"**{trade['ticker']} {trade['strike']} {trade['option_type'].upper()}** (#{trade['id']})\n"
            f"Cannot get exit premium after {attempts} consecutive poll cycles.\n"
            f"Entry: ${trade['premium_per_contract']:.2f} × {trade['contracts']} contracts\n"
            f"Using delta estimate as fallback — exit timing may be inaccurate."
        ),
        trade_id=trade["id"],
        alert_type="premium_blackout",
    )


async def alert_expiry_danger(
    client: discord.Client,
    settings: Settings,
    trade: dict,
    minutes_left: int,
) -> None:
    """Alert when an open trade is close to expiry."""
    await send_alert(
        client, settings,
        title="Expiry Danger — Force Close Imminent",
        message=(
            f"**{trade['ticker']} {trade['strike']} {trade['option_type'].upper()}** (#{trade['id']})\n"
            f"**{minutes_left} minutes until market close** — trade will be force-closed.\n"
            f"Entry: ${trade['premium_per_contract']:.2f} × {trade['contracts']} contracts\n"
            f"Current P&L unknown (exit premium may be stale)."
        ),
        trade_id=trade["id"],
        alert_type="expiry_danger",
    )


async def alert_force_closed(
    client: discord.Client,
    settings: Settings,
    trade: dict,
    exit_premium: float,
    reason: str,
) -> None:
    """Alert when a trade is force-closed (expiry safety, error recovery)."""
    entry_prem = trade["premium_per_contract"]
    pnl_pct = ((exit_premium - entry_prem) / entry_prem * 100) if entry_prem > 0 else 0
    pnl_dollars = (exit_premium - entry_prem) * trade["contracts"] * 100

    await send_alert(
        client, settings,
        title="Trade Force-Closed",
        message=(
            f"**{trade['ticker']} {trade['strike']} {trade['option_type'].upper()}** (#{trade['id']})\n"
            f"Reason: {reason}\n"
            f"Entry: ${entry_prem:.2f} → Exit: ${exit_premium:.2f} ({pnl_pct:+.1f}%)\n"
            f"P&L: ${pnl_dollars:+.2f} on {trade['contracts']} contracts"
        ),
        trade_id=trade["id"],
        alert_type="force_closed",
        force=True,  # always send force-close alerts
    )


async def alert_exit_error(
    client: discord.Client,
    settings: Settings,
    trade: dict,
    error: str,
) -> None:
    """Alert when an exit order fails."""
    await send_alert(
        client, settings,
        title="Exit Order Failed",
        message=(
            f"**{trade['ticker']} {trade['strike']} {trade['option_type'].upper()}** (#{trade['id']})\n"
            f"Error: {error}\n"
            f"Entry: ${trade['premium_per_contract']:.2f} × {trade['contracts']} contracts\n"
            f"**Manual intervention may be required.**"
        ),
        trade_id=trade["id"],
        alert_type="exit_error",
    )


async def alert_position_mismatch(
    client: discord.Client,
    settings: Settings,
    message: str,
) -> None:
    """Alert when Webull positions don't match paper DB."""
    await send_alert(
        client, settings,
        title="Position Mismatch",
        message=message,
        trade_id=0,
        alert_type="position_mismatch",
    )


async def alert_critical(
    client: discord.Client,
    settings: Settings,
    message: str,
) -> None:
    """Alert for critical issues like stuck sell orders."""
    await send_alert(
        client, settings,
        title="CRITICAL",
        message=message,
        trade_id=0,
        alert_type="critical",
    )


def clear_alerts_for_trade(trade_id: int) -> None:
    """Clear alert dedup state when a trade is successfully closed."""
    to_remove = {k for k in _alerted_trades if k[0] == trade_id}
    _alerted_trades.difference_update(to_remove)
