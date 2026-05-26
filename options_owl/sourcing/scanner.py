"""Main scan loop: 3-min interval, market hours guard, ticker iteration.

Entry point for the owlet-sourcing container:
    python -m options_owl.sourcing.scanner
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from zoneinfo import ZoneInfo

from options_owl.sourcing.config import SourcingSettings
from options_owl.sourcing.data.candle_provider import fetch_candles
from options_owl.sourcing.data.indicator_engine import compute_indicators
from options_owl.sourcing.filters.cooldown_manager import (
    check_circuit_breaker,
    check_daily_cap,
    is_on_cooldown,
    record_signal_emitted,
)
from options_owl.sourcing.filters.penalty_veto import check_penalty_veto
from options_owl.sourcing.filters.quality_gate import check_quality_gate
from options_owl.sourcing.filters.veto_gates import run_veto_gates
from options_owl.sourcing.output.audit_logger import log_audit
from options_owl.sourcing.output.discord_webhook import emit_discord
from options_owl.sourcing.output.signal_db_writer import emit_signal_db
from options_owl.sourcing.scoring.engine import compute_score
from options_owl.sourcing.scoring.ml_gates.signal_model import (
    compute_option_features_from_live,
    predict_entry_confidence,
)
from options_owl.sourcing.scoring.types import Direction, SignalContext, SignalState

ET = ZoneInfo("America/New_York")


def _is_market_open() -> bool:
    """Check if US equity market is currently open (9:33 AM - 3:57 PM ET)."""
    now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=33, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=57, second=0, microsecond=0)
    return market_open <= now <= market_close


async def scan_ticker(ticker: str, settings: SourcingSettings) -> SignalContext | None:
    """Run the full pipeline for a single ticker.

    Returns SignalContext if signal was scored, None on data failure.
    """
    scan_time = datetime.now(tz=timezone.utc).isoformat()

    # --- Stage 1: Fetch candles ---
    try:
        candles_5m = await asyncio.wait_for(
            fetch_candles(ticker, "5min", bars=78, db_path=settings.SHARED_CANDLE_DB),
            timeout=15,
        )
    except asyncio.TimeoutError:
        logger.warning(f"SCAN {ticker}: candle fetch timed out (15s)")
        candles_5m = None

    if not candles_5m or len(candles_5m) < 10:
        logger.debug(f"SCAN {ticker}: insufficient candle data ({len(candles_5m) if candles_5m else 0} bars)")
        return None

    # Optionally fetch 15m candles for multi-TF
    candles_15m = None
    try:
        candles_15m = await asyncio.wait_for(
            fetch_candles(ticker, "15min", bars=26, db_path=settings.SHARED_CANDLE_DB),
            timeout=15,
        )
    except asyncio.TimeoutError:
        pass

    # --- Stage 2: Compute indicators ---
    indicators = compute_indicators(candles_5m)

    # --- Stage 2.5: Determine direction ---
    # Use EMA cross and MACD to determine CALL vs PUT
    direction = _infer_direction(indicators)

    # --- Stage 2.5b: PUT ticker exclusion ---
    if direction == Direction.PUT:
        excluded_str = getattr(settings, "PUT_EXCLUDED_TICKERS", "")
        excluded = {t.strip().upper() for t in excluded_str.split(",") if t.strip()}
        if ticker.upper() in excluded:
            logger.debug(f"SCAN {ticker}: PUT excluded (backtest loser)")
            return None

    # --- Stage 2.6: Fetch real option data from harvester for ML ---
    option_snapshot = None
    option_history = None
    if settings.ENABLE_ML_SIGNAL_MODEL:
        try:
            from options_owl.sourcing.data.harvester_options import (
                fetch_atm_option_snapshot,
                fetch_option_history,
            )
            direction_str = "CALL" if direction == Direction.CALL else "PUT"
            option_snapshot, option_history = await asyncio.gather(
                asyncio.wait_for(
                    fetch_atm_option_snapshot(ticker, direction_str, settings.SHARED_CANDLE_DB),
                    timeout=5,
                ),
                asyncio.wait_for(
                    fetch_option_history(ticker, direction_str, settings.SHARED_CANDLE_DB),
                    timeout=5,
                ),
                return_exceptions=True,
            )
            if isinstance(option_snapshot, Exception):
                logger.debug(f"SCAN {ticker}: harvester snapshot error: {option_snapshot}")
                option_snapshot = None
            if isinstance(option_history, Exception):
                option_history = None
        except Exception:
            pass  # harvester data is best-effort

    # --- Stage 3: Build context and score ---
    ctx = SignalContext(
        ticker=ticker,
        scan_time=scan_time,
        state=SignalState.INDICATED,
        direction=direction,
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        candle_source="harvester_db",
        indicators=indicators,
    )

    # Attach harvester option data for ML gate (avoids candle-as-option mismatch)
    ctx._option_snapshot = option_snapshot  # type: ignore[attr-defined]
    ctx._option_history = option_history  # type: ignore[attr-defined]

    scored = compute_score(ctx)
    ctx.score_total = scored.score
    ctx.state = SignalState.SCORED

    # --- Stage 4: Threshold filter ---
    if scored.rejected:
        ctx.state = SignalState.REJECTED
        ctx.filter_result = "rejected"
        ctx.filter_reason = scored.reject_reason or "scoring_rejected"
        logger.info(
            f"SCAN {ticker}: REJECTED score={scored.score} "
            f"reason={ctx.filter_reason}"
        )
        return ctx

    if scored.score < settings.SCORE_THRESHOLD:
        ctx.state = SignalState.REJECTED
        ctx.filter_result = "below_threshold"
        ctx.filter_reason = f"score {scored.score} < {settings.SCORE_THRESHOLD}"
        logger.debug(
            f"SCAN {ticker}: below threshold score={scored.score}/{settings.SCORE_THRESHOLD}"
        )
        return ctx

    # Enrich context with real option data when available
    if option_snapshot and option_snapshot.midpoint > 0:
        ctx.premium = option_snapshot.midpoint
        ctx.strike = option_snapshot.strike

    # --- Stage 4a: ML signal model (optional) ---
    if settings.ENABLE_ML_SIGNAL_MODEL:
        ml_result = _run_ml_gate(ctx)
        ctx.ml_confidence = ml_result.get("confidence")
        ctx.ml_threshold = ml_result.get("threshold")
        ctx.ml_is_signal = ml_result.get("is_signal")
        ctx.ml_runner_score = ml_result.get("runner_score")
        ctx.ml_model_source = ml_result.get("model_source", "")

        if ml_result["model_source"] != "none" and not ml_result["is_signal"]:
            ctx.state = SignalState.REJECTED
            ctx.filter_result = "ml_veto"
            ctx.filter_reason = (
                f"ML confidence {ml_result['confidence']:.2f} "
                f"< threshold {ml_result['threshold']:.2f}"
            )
            logger.info(f"SCAN {ticker}: ML VETO {ctx.filter_reason}")
            return ctx

    # --- Stage 4b: Quality gate (multi-tier contribution check) ---
    if not check_quality_gate(ctx, settings.SCORE_THRESHOLD):
        ctx.state = SignalState.REJECTED
        logger.info(f"SCAN {ticker}: quality gate failed: {ctx.filter_reason}")
        return ctx

    # --- Stage 4c: Penalty veto (dangerous combo check) ---
    if check_penalty_veto(ctx):
        ctx.state = SignalState.REJECTED
        logger.info(f"SCAN {ticker}: penalty veto: {ctx.filter_reason}")
        return ctx

    # --- Stage 4d: Simpsons-inspired hard veto gates ---
    veto_blocked, veto_reason = run_veto_gates(ctx)
    if veto_blocked:
        ctx.state = SignalState.REJECTED
        ctx.filter_result = "vetoed"
        ctx.filter_reason = veto_reason
        return ctx

    ctx.state = SignalState.FILTERED
    logger.info(
        f"SCAN {ticker}: SIGNAL {direction.value if direction else '?'} "
        f"score={scored.score} ema_cross={indicators.ema_cross_strength:.2f} "
        f"rsi={indicators.rsi9:.1f} vol_ratio={indicators.volume_ratio:.1f}"
    )
    return ctx


def _run_ml_gate(ctx: SignalContext) -> dict:
    """Run ML signal model using REAL option data from harvester DB.

    Reads actual option bid/ask/IV/delta/volume from the shared harvester
    snapshots, NOT stock candle proxies. This matches the ThetaData features
    the model was trained on.

    Falls back to candle-derived features if harvester data is unavailable.
    """
    is_call = ctx.direction == Direction.CALL
    direction_str = "CALL" if is_call else "PUT"

    # Try to get real option data from harvester (preferred — matches training data)
    option_snap = getattr(ctx, "_option_snapshot", None)
    option_hist = getattr(ctx, "_option_history", None)

    if option_snap and option_snap.midpoint > 0:
        # Use REAL option data — matches ThetaData training features
        premium = option_snap.midpoint
        bid = option_snap.bid
        ask = option_snap.ask
        iv = option_snap.iv if option_snap.iv > 0 else 0.3
        delta = abs(option_snap.delta) if option_snap.delta else 0.50
        theta = option_snap.theta if option_snap.theta else -0.05
        vega = option_snap.vega if option_snap.vega else 0.10
        volume = option_snap.volume
        underlying_price = option_snap.underlying_price

        # Build history from harvester snapshots
        if option_hist and option_hist.snapshots:
            premium_history = option_hist.premium_history
            volume_history = option_hist.volume_history
            underlying_history = option_hist.underlying_history
        else:
            premium_history = [premium]
            volume_history = [volume]
            underlying_history = [underlying_price]

        logger.debug(
            f"SCAN {ctx.ticker}: ML using REAL option data — "
            f"premium=${premium:.2f} bid=${bid:.2f} ask=${ask:.2f} "
            f"iv={iv:.3f} delta={delta:.3f} vol={volume}"
        )
    else:
        # Fallback to candle-derived features (less accurate but functional)
        indicators = ctx.indicators
        candles = ctx.candles_5m or []

        if not candles or not indicators:
            return {"confidence": 0.0, "threshold": 1.0, "is_signal": False,
                    "runner_score": 0.0, "model_source": "none"}

        last = candles[-1]
        premium = last.get("close", 0)
        bid = last.get("low", 0)
        ask = last.get("high", 0)
        volume = int(last.get("volume", 0))
        underlying_price = last.get("close", 0)

        premium_history = [c["close"] for c in candles[-15:]]
        volume_history = [int(c.get("volume", 0)) for c in candles[-15:]]
        underlying_history = [c["close"] for c in candles[-15:]]

        iv = getattr(indicators, "iv", 0.3) or 0.3
        delta = 0.50
        theta = -0.05
        vega = 0.10

        logger.debug(f"SCAN {ctx.ticker}: ML using CANDLE fallback (no harvester option data)")

    # Minutes since market open (9:30 ET)
    now_et = datetime.now(tz=ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = max(0, int((now_et - market_open).total_seconds() / 60))

    features = compute_option_features_from_live(
        ticker=ctx.ticker,
        premium=premium,
        bid=bid,
        ask=ask,
        iv=iv,
        delta=delta,
        theta=theta,
        vega=vega,
        volume=volume,
        underlying_price=underlying_price,
        minutes_since_open=minutes_since_open,
        is_call=is_call,
        premium_history=premium_history,
        volume_history=volume_history,
        underlying_history=underlying_history,
    )
    return predict_entry_confidence(ctx.ticker, features, direction=direction_str)


def _infer_direction(indicators) -> Direction:
    """Infer CALL/PUT from technical indicators.

    Uses EMA cross strength as primary, MACD as confirmation.
    """
    bullish_signals = 0
    bearish_signals = 0

    # EMA cross (primary)
    if indicators.ema_cross_strength > 0.05:
        bullish_signals += 2
    elif indicators.ema_cross_strength < -0.05:
        bearish_signals += 2

    # MACD
    if indicators.macd_line > 0:
        bullish_signals += 1
    elif indicators.macd_line < 0:
        bearish_signals += 1

    # VWAP
    if indicators.vwap > 0 and indicators.last_close > indicators.vwap:
        bullish_signals += 1
    elif indicators.vwap > 0 and indicators.last_close < indicators.vwap:
        bearish_signals += 1

    return Direction.CALL if bullish_signals >= bearish_signals else Direction.PUT


async def scan_once(settings: SourcingSettings) -> list[SignalContext]:
    """Execute a single scan cycle across all tickers.

    Returns list of all scored contexts (both passed and rejected).
    """
    tickers = settings.ticker_list
    logger.info(f"SCAN: starting cycle for {len(tickers)} tickers")

    # Pre-flight: circuit breaker
    try:
        if await check_circuit_breaker():
            logger.warning("SCAN: CIRCUIT BREAKER ACTIVE — skipping cycle")
            return []
    except Exception:
        pass  # DB not available, continue anyway

    # Pre-flight: daily cap
    try:
        if await check_daily_cap(max_daily=50):
            logger.warning("SCAN: daily cap reached — skipping cycle")
            return []
    except Exception:
        pass

    results: list[SignalContext] = []
    scan_start_ms = int(time.monotonic() * 1000)

    for ticker in tickers:
        try:
            ctx = await scan_ticker(ticker, settings)
            if ctx is None:
                continue

            results.append(ctx)

            # Audit log every evaluation (pass or fail)
            try:
                elapsed_ms = int(time.monotonic() * 1000) - scan_start_ms
                await log_audit(ctx, scan_duration_ms=elapsed_ms)
            except Exception:
                pass  # audit is best-effort

            # Emit signals that passed all filters
            if ctx.state == SignalState.FILTERED:
                direction_str = ctx.direction.value if ctx.direction else "CALL"

                # Cooldown check
                try:
                    if await is_on_cooldown(ticker, direction_str):
                        ctx.state = SignalState.REJECTED
                        ctx.filter_result = "cooldown"
                        ctx.filter_reason = f"{ticker} {direction_str} on cooldown"
                        logger.info(f"SCAN {ticker}: cooldown active, skipping emit")
                        continue
                except Exception:
                    pass  # DB not available, skip cooldown check

                # Emit to PostgreSQL signals table
                if settings.SOURCING_DB_OUTPUT:
                    try:
                        await emit_signal_db(ctx)
                    except Exception:
                        logger.exception(f"SCAN {ticker}: signal DB write failed")

                # Emit to ml_signals table (consumed by trading bots)
                try:
                    from options_owl.db import postgres as pg
                    if pg.is_connected():
                        await pg.emit_ml_signal({
                            "ticker": ctx.ticker,
                            "direction": ctx.direction.value if ctx.direction else "CALL",
                            "score": ctx.score_total,
                            "ml_confidence": ctx.ml_confidence,
                            "ml_threshold": ctx.ml_threshold,
                            "ml_model_source": ctx.ml_model_source,
                            "ml_runner_score": ctx.ml_runner_score,
                            "premium": ctx.premium,
                            "strike": ctx.strike,
                            "expiry_date": None,
                            "indicators": {},
                            "score_breakdown": {},
                            "emitted_at": datetime.now(tz=timezone.utc),
                        })
                except Exception:
                    logger.debug(f"SCAN {ticker}: ml_signals write failed (non-critical)")

                # Emit to Discord webhook
                if settings.SOURCING_DISCORD_OUTPUT:
                    try:
                        await emit_discord(ctx)
                    except Exception:
                        logger.exception(f"SCAN {ticker}: Discord webhook failed")

                # Record cooldown
                try:
                    await record_signal_emitted(ticker, direction_str)
                except Exception:
                    pass

                ctx.state = SignalState.EMITTED

        except Exception:
            logger.exception(f"SCAN {ticker}: unhandled error")

    passed = [r for r in results if r.state in (SignalState.FILTERED, SignalState.EMITTED)]
    rejected = [r for r in results if r.state == SignalState.REJECTED]
    logger.info(
        f"SCAN: complete | {len(passed)} signals emitted, "
        f"{len(rejected)} rejected, "
        f"{len(tickers) - len(results)} no data"
    )
    return results


async def scan_loop() -> None:
    """Main loop: fire scan_once every SCAN_INTERVAL_SECONDS during market hours."""
    settings = SourcingSettings()
    logger.info(
        f"owlet-sourcing starting | interval={settings.SCAN_INTERVAL_SECONDS}s "
        f"| tickers={settings.SOURCING_TICKERS}"
    )

    # Initialize PostgreSQL connection pool + schema
    try:
        from options_owl.sourcing import db
        await db.init_pool()
    except Exception:
        logger.exception("Failed to initialize PostgreSQL — running without DB output")

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
    from options_owl.main import configure_logging
    configure_logging()
    asyncio.run(scan_loop())


if __name__ == "__main__":
    main()
