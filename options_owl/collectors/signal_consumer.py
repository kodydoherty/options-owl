"""Signal consumer — polls PostgreSQL for ML-sourced signals and feeds them
to the existing entry pipeline (paper_trader.evaluate_and_trade).

This bridges the sourcing scanner output to the trading bots. Each bot runs
this consumer in its event loop alongside the Discord collector.

Flow:
  owlet-sourcing → ml_signals table (PG) → signal_consumer → TradeSignal
  → paper_trader.evaluate_and_trade → entry pipeline → Webull

The consumer runs every 30s, fetching signals not yet consumed by this agent.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from options_owl.config.settings import Settings
from options_owl.models.signals import BotSource, Direction, Sentiment, SignalStrength, TradeSignal

POLL_INTERVAL_SECONDS = 5


async def run_signal_consumer(paper_trader, settings: Settings) -> None:
    """Background loop that polls ml_signals and routes them to the entry pipeline.

    Args:
        paper_trader: PaperTrader instance (has evaluate_and_trade method)
        settings: Bot settings (needs AGENT_ID, ENABLE_POSTGRES)
    """
    if not getattr(settings, "ENABLE_POSTGRES", False):
        logger.debug("Signal consumer disabled (ENABLE_POSTGRES=false)")
        return

    agent_id = getattr(settings, "AGENT_ID", "") or "unknown"
    logger.info(f"Signal consumer starting for {agent_id} (poll every {POLL_INTERVAL_SECONDS}s)")

    # Wait for PG pool to be ready
    await asyncio.sleep(5)

    while True:
        try:
            await _poll_and_route(paper_trader, settings, agent_id)
        except Exception:
            logger.exception("Signal consumer: unhandled error in poll cycle")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _poll_and_route(paper_trader, settings: Settings, agent_id: str) -> None:
    """Single poll cycle: fetch pending signals, convert, route to entry pipeline."""
    from options_owl.db import postgres as pg

    if not pg.is_connected():
        return

    signals = await pg.get_pending_signals(agent_id, max_age_minutes=10)
    if not signals:
        return

    logger.info(f"Signal consumer: {len(signals)} pending ML signals")

    for sig in signals:
        ticker = sig["ticker"]
        direction_str = sig.get("direction", "CALL")
        score = sig.get("score", 0)
        ml_confidence = sig.get("ml_confidence")
        premium = sig.get("premium")
        strike = sig.get("strike")
        expiry = sig.get("expiry_date")

        # Skip if score below min
        if score < settings.MIN_SCORE:
            logger.debug(
                f"Signal consumer: skipping {ticker} score={score} < {settings.MIN_SCORE}"
            )
            await pg.mark_signal_consumed(sig["id"], agent_id)
            continue

        # Convert to TradeSignal (the format the entry pipeline expects)
        try:
            direction = Direction.CALL if direction_str == "CALL" else Direction.PUT
            trade_signal = TradeSignal(
                ticker=ticker,
                direction=direction,
                sentiment=Sentiment.BULLISH if direction == Direction.CALL else Sentiment.BEARISH,
                score=score,
                strength=_score_to_strength(score),
                bot_source=BotSource.ML_SOURCING,
                entry_price=premium or 0,
                target_price=0,
                expected_move_pct=0,
                strike=strike or 0,
                expiry=expiry or "0DTE",
                risk_reward=0,
                target_1=None,
                target_2=None,
                stop_price=None,
                exit_by=None,
                atm_strike=strike,
                atm_premium=premium,
            )
        except Exception as exc:
            logger.warning(f"Signal consumer: failed to build TradeSignal for {ticker}: {exc}")
            await pg.mark_signal_consumed(sig["id"], agent_id)
            continue

        # Route through the full entry pipeline
        if ml_confidence is not None:
            logger.info(
                f"Signal consumer: routing ML signal {ticker} {direction_str} "
                f"score={score} conf={ml_confidence:.2f}"
            )
        else:
            logger.info(
                f"Signal consumer: routing ML signal {ticker} {direction_str} score={score}"
            )

        try:
            # Create a synthetic signal_id (negative to distinguish from Discord signals)
            synthetic_signal_id = -sig["id"]
            result = await paper_trader.evaluate_and_trade(
                trade_signal, synthetic_signal_id, ml_confidence=ml_confidence,
            )
            if result:
                logger.info(
                    f"Signal consumer: TRADED {ticker} {direction_str} "
                    f"trade_id={result['trade_id']}"
                )
            else:
                logger.info(f"Signal consumer: {ticker} rejected by entry pipeline")
        except Exception:
            logger.exception(f"Signal consumer: error routing {ticker}")

        # Mark consumed regardless of outcome
        await pg.mark_signal_consumed(sig["id"], agent_id)


def _score_to_strength(score: int) -> SignalStrength:
    """Map score to strength tier."""
    if score >= 150:
        return SignalStrength.ELITE
    elif score >= 120:
        return SignalStrength.STRONG
    elif score >= 90:
        return SignalStrength.GOOD
    elif score >= 78:
        return SignalStrength.MODERATE
    return SignalStrength.MARGINAL
