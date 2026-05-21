"""Main scan loop: 3-min interval, market hours guard, ticker iteration.

Entry point for the owlet-sourcing container:
    python -m options_owl.sourcing.scanner
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from loguru import logger
from zoneinfo import ZoneInfo

from options_owl.sourcing.config import SourcingSettings

ET = ZoneInfo("America/New_York")


def _is_market_open() -> bool:
    """Check if US equity market is currently open (9:33 AM - 3:57 PM ET)."""
    now = datetime.now(tz=ET)
    if now.weekday() >= 5:  # Saturday, Sunday
        return False
    market_open = now.replace(hour=9, minute=33, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=57, second=0, microsecond=0)
    return market_open <= now <= market_close


async def scan_once(settings: SourcingSettings) -> None:
    """Execute a single scan cycle across all tickers.

    TODO: Wire up data fetch, indicator engine, scoring, filters, output.
    """
    logger.info("SCAN: starting scan cycle")
    # Placeholder — will be wired in Phase 1-3
    logger.info("SCAN: complete (scaffold only)")


async def scan_loop() -> None:
    """Main loop: fire scan_once every SCAN_INTERVAL_SECONDS during market hours."""
    settings = SourcingSettings()
    logger.info(
        f"owlet-sourcing starting | interval={settings.SCAN_INTERVAL_SECONDS}s "
        f"| tickers={settings.SOURCING_TICKERS}"
    )

    while True:
        if not _is_market_open():
            await asyncio.sleep(60)
            continue

        scan_start = time.monotonic()
        try:
            await scan_once(settings)
        except Exception:
            logger.exception("SCAN: unhandled exception in scan cycle")

        elapsed = time.monotonic() - scan_start
        sleep_for = max(0, settings.SCAN_INTERVAL_SECONDS - elapsed)
        logger.debug(f"SCAN: elapsed={elapsed:.1f}s, sleeping {sleep_for:.1f}s")
        await asyncio.sleep(sleep_for)


def main() -> None:
    asyncio.run(scan_loop())


if __name__ == "__main__":
    main()
