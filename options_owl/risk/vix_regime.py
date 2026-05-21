"""VIX regime detection for position sizing and trade gating.

Feature flag: ENABLE_VIX_FILTER
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import yfinance as yf
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from options_owl.config.settings import Settings

# Cache: (timestamp, vix_level)
_vix_cache: tuple[float, float] | None = None
_CACHE_TTL_SECONDS = 5 * 60  # 5 minutes


class VixRegime(BaseModel):
    """Current VIX regime assessment."""

    level: float
    regime: str  # "low", "normal", "high", "extreme"
    can_trade: bool
    position_size_multiplier: float
    reason: str


def fetch_vix_level() -> float | None:
    """Fetch the current VIX level via yfinance.

    Results are cached for 5 minutes.

    Returns:
        Current VIX level, or None on failure.
    """
    global _vix_cache

    if _vix_cache is not None:
        ts, level = _vix_cache
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return level

    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if hist.empty:
            logger.warning("VIX fetch returned empty data")
            return None
        level = float(hist["Close"].iloc[-1])
        _vix_cache = (time.time(), level)
        logger.debug(f"VIX level fetched: {level:.1f}")
        return level
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}")
        return None


def check_vix_regime(settings: Settings) -> VixRegime:
    """Assess the current VIX regime and determine trading parameters.

    Args:
        settings: Application settings with VIX thresholds.

    Returns:
        VixRegime object describing current conditions.
    """
    if not settings.ENABLE_VIX_FILTER:
        return VixRegime(
            level=0.0,
            regime="normal",
            can_trade=True,
            position_size_multiplier=1.0,
            reason="VIX filter is disabled",
        )

    vix_level = fetch_vix_level()
    if vix_level is not None:
        logger.debug(f"VIX regime assessment: level={vix_level:.1f}")
    if vix_level is None:
        return VixRegime(
            level=0.0,
            regime="normal",
            can_trade=True,
            position_size_multiplier=1.0,
            reason="Could not fetch VIX data, defaulting to normal regime",
        )

    # Determine regime
    if vix_level > settings.VIX_MAX:
        return VixRegime(
            level=vix_level,
            regime="extreme",
            can_trade=False,
            position_size_multiplier=0.0,
            reason=f"VIX at {vix_level:.1f} exceeds maximum {settings.VIX_MAX:.1f} — trading paused",
        )

    if vix_level > settings.VIX_HIGH_THRESHOLD:
        multiplier = 1.0 - (settings.VIX_POSITION_REDUCTION_PCT / 100.0)
        return VixRegime(
            level=vix_level,
            regime="high",
            can_trade=True,
            position_size_multiplier=multiplier,
            reason=(
                f"VIX at {vix_level:.1f} is elevated "
                f"(>{settings.VIX_HIGH_THRESHOLD:.1f}) — "
                f"reducing position size by {settings.VIX_POSITION_REDUCTION_PCT:.0f}%"
            ),
        )

    if vix_level < 15.0:
        return VixRegime(
            level=vix_level,
            regime="low",
            can_trade=True,
            position_size_multiplier=1.0,
            reason=f"VIX at {vix_level:.1f} — low volatility regime",
        )

    return VixRegime(
        level=vix_level,
        regime="normal",
        can_trade=True,
        position_size_multiplier=1.0,
        reason=f"VIX at {vix_level:.1f} — normal regime",
    )
