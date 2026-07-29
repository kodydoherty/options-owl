"""Held-contract options-WS feed — real-time premiums for the monitor's exit logic.

The harvester runs this as one isolated task: it opens a Polygon **/options** WebSocket (via the tested
``MarketDataStream`` client) subscribed to ONLY the union of currently-held contracts across all bots
(a small, dynamic set — each bot registers its open contracts in Redis), and republishes the per-tick NBBO
to Redis ``owl:optquote:<key>``. The monitor reads that ahead of the 15s REST snapshot, so trail/stop/
profit-lock exit gates react in ~sub-second instead of being bounded by the harvester's 15s snapshot cadence.

SAFETY: strictly additive + isolated. Flag-gated (``ENABLE_HELD_OPTION_WS``). Every loop is wrapped so a
failure here can NEVER disturb the harvester's REST snapshot / candle / flow pipeline — on any error the
monitor simply falls back to the existing 15s snapshots. See
specs/todo/2026-07-22_held-contract-options-ws-feed.md.
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger

from options_owl.collectors.market_data_stream import MarketDataStream

# Only republish a contract's quote if the WS cache ticked within this window — a contract that stopped
# ticking must go stale (let its owl:optquote key expire) so the monitor falls back to the snapshot.
_FRESH_SEC = 12.0


def _parse_contract_key(key: str) -> tuple[str, str, float, str] | None:
    """'TICKER:type:strike:expiry' -> (ticker, option_type, strike, expiry) or None if malformed."""
    parts = key.split(":")
    if len(parts) != 4:
        return None
    ticker, otype, strike_s, expiry = parts
    try:
        strike = float(strike_s)
    except (TypeError, ValueError):
        return None
    if otype.lower() not in ("call", "put") or not ticker or not expiry:
        return None
    return ticker.upper(), otype.lower(), strike, expiry


async def run_held_option_ws(settings, stop_event: asyncio.Event | None = None) -> None:
    """Long-running harvester task. No-op unless ENABLE_HELD_OPTION_WS. Never raises."""
    if getattr(settings, "ENABLE_HELD_OPTION_WS", False) is not True:
        return
    if not getattr(settings, "POLYGON_API_KEY", ""):
        logger.warning("held-option-ws: no POLYGON_API_KEY — skipping real-time options feed")
        return

    from options_owl.db import redis_client

    # Force the options WS ON for THIS stream only (harvester's other paths are unaffected).
    try:
        ws_settings = settings.model_copy(update={"ENABLE_POLYGON_WS": True})
    except Exception:  # noqa: BLE001 - fall back to a shallow attr flip
        ws_settings = settings
        try:
            ws_settings.ENABLE_POLYGON_WS = True
        except Exception:  # noqa: BLE001
            pass

    stream = MarketDataStream(ws_settings)
    refresh = float(getattr(settings, "HELD_OPTION_WS_REFRESH_SEC", 3.0))
    subscribed: dict[str, tuple[str, str, float, str]] = {}  # contract_key -> parsed
    try:
        await stream.start()
        logger.info("held-option-ws: started (Polygon /options WS for held contracts)")
        while stop_event is None or not stop_event.is_set():
            try:
                held = await redis_client.get_all_held_contracts()
                parsed = {k: p for k in held if (p := _parse_contract_key(k)) is not None}

                # Subscribe newly-held, unsubscribe no-longer-held (diff — small set).
                for k, p in parsed.items():
                    if k not in subscribed:
                        await stream.subscribe_option(p[0], p[2], p[3], p[1])
                        subscribed[k] = p
                for k in [k for k in subscribed if k not in parsed]:
                    p = subscribed.pop(k)
                    try:
                        await stream.unsubscribe_option(p[0], p[2], p[3], p[1])
                    except Exception:  # noqa: BLE001
                        pass

                # Republish fresh per-tick quotes for every held contract.
                now = time.time()
                published = 0
                for k, p in subscribed.items():
                    sym = MarketDataStream.build_option_contract_ticker(p[0], p[2], p[3], p[1])
                    entry = stream._option_cache.get(sym)
                    if not entry:
                        continue
                    mid, ts = entry
                    if mid and mid > 0 and (now - ts) <= _FRESH_SEC:
                        # _option_cache holds mid only; publish mid as bid/ask too (v1 — the FSM uses mid).
                        await redis_client.publish_optquote(k, bid=mid, ask=mid, mid=mid)
                        published += 1
                if subscribed:
                    logger.debug(
                        f"held-option-ws: {len(subscribed)} subscribed, {published} fresh quotes published"
                    )
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the feed
                logger.warning(f"held-option-ws cycle error (non-fatal): {exc}")
            await asyncio.sleep(refresh)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the feed must never take down the harvester
        logger.error(f"held-option-ws fatal (feed disabled, monitor falls back to snapshots): {exc}")
    finally:
        try:
            await stream.stop()
        except Exception:  # noqa: BLE001
            pass
