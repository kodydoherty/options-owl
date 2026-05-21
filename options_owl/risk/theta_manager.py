"""Theta decay exit rules — close positions when time decay is working against us."""

from __future__ import annotations

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone, timedelta
    _ET = timezone(timedelta(hours=-5))


def _now_et() -> datetime:
    """Current time in Eastern Time (DST-aware)."""
    return datetime.now(tz=_ET)


def calc_time_to_expiry_days(expiry_date: str) -> float:
    """Calculate days to expiry from a YYYY-MM-DD expiry date string.

    Returns fractional days (e.g. 0.5 = 12 hours).  Returns 0.0 if the
    expiry date is today or in the past, or if parsing fails.
    """
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        now = _now_et().replace(tzinfo=None)
        delta = expiry_dt - now.replace(hour=0, minute=0, second=0, microsecond=0)
        return max(delta.days, 0.0)
    except (ValueError, TypeError):
        return 0.0


def should_theta_exit(
    trade: dict,
    current_premium: float,
    settings: object,
) -> tuple[bool, str]:
    """Determine whether a trade should be closed due to theta decay.

    Args:
        trade: A paper_trades row dict with keys like ``opened_at``,
            ``premium_per_contract``, ``expiry_date``, etc.
        current_premium: The estimated or live current premium per contract.
        settings: A ``Settings`` instance carrying theta-related fields.

    Returns:
        (should_exit, reason) — *reason* is a human-readable explanation when
        *should_exit* is True, or an empty string otherwise.
    """
    if not getattr(settings, "ENABLE_THETA_DECAY_EXIT", False):
        return False, ""

    dte_threshold: int = getattr(settings, "THETA_EXIT_DTE_THRESHOLD", 1)
    loss_pct_limit: float = getattr(settings, "THETA_EXIT_LOSS_PCT", 50.0)
    time_limit_minutes: int = getattr(settings, "THETA_EXIT_TIME_MINUTES", 60)

    # --- P&L calculation ---
    entry_premium = trade.get("premium_per_contract", 0.0)
    if entry_premium <= 0:
        return False, ""

    pnl_pct = ((current_premium - entry_premium) / entry_premium) * 100.0
    is_losing = pnl_pct < 0

    # --- Time held ---
    opened_at_str = trade.get("opened_at", "")
    if not opened_at_str:
        return False, ""

    try:
        opened_at = datetime.fromisoformat(opened_at_str)
    except (ValueError, TypeError):
        return False, ""

    now = _now_et().replace(tzinfo=None)
    minutes_held = (now - opened_at).total_seconds() / 60.0

    # --- DTE ---
    expiry_date = trade.get("expiry_date") or ""
    dte = calc_time_to_expiry_days(expiry_date) if expiry_date else 0.0

    # Rule 1: 0DTE — exit if held > time limit with no profit
    if dte == 0 and minutes_held > time_limit_minutes and is_losing:
        return True, (
            f"0DTE theta exit: held {minutes_held:.0f}min "
            f"(limit {time_limit_minutes}min) with P&L {pnl_pct:+.1f}%"
        )

    # Rule 2: non-0DTE but DTE <= threshold and losing badly
    if 0 < dte <= dte_threshold and is_losing and abs(pnl_pct) >= loss_pct_limit:
        return True, (
            f"Theta decay exit: DTE={dte:.0f} (<= {dte_threshold}), "
            f"P&L {pnl_pct:+.1f}% exceeds -{loss_pct_limit:.0f}% limit"
        )

    # Rule 3: 0DTE losing more than loss limit regardless of time
    if dte == 0 and is_losing and abs(pnl_pct) >= loss_pct_limit:
        return True, (
            f"0DTE loss exit: P&L {pnl_pct:+.1f}% exceeds "
            f"-{loss_pct_limit:.0f}% limit with accelerating theta"
        )

    return False, ""
