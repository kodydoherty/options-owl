"""Standalone bot runner — no Discord dependency.

Each bot runs its own ML signal scanner + entry pipeline + position monitor.
Uses V3 ML models (pattern_entry, entry_timing, regime_classifier) matching
the gold standard backtest for signal generation.

Flow:
  1. Init Webull, PaperTrader, MarketDataStream, Redis, PG
  2. Load V3 ML models at startup
  3. Run regime filter at 9:45 AM ET — skip entire day if market bad
  4. Scan every minute 9:35-11:00 ET — fetch live option data, run ML models
  5. Route passing signals directly to paper_trader.evaluate_and_trade
  6. Position monitor runs in background for exit management
  7. Heartbeat for Docker healthcheck

Usage:
    python -m options_owl.bot_runner
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

from options_owl.config.settings import Settings
from options_owl.main import configure_logging, write_heartbeat

ET = ZoneInfo("America/New_York")

# Discord webhook for ML signal alerts (fire-and-forget)
_SIGNAL_WEBHOOK_URL = os.getenv("SOURCING_DISCORD_WEBHOOK_URL", "")


def _is_market_open() -> bool:
    """Check if US equity market is currently open (9:30 AM - 3:57 PM ET)."""
    now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=57, second=0, microsecond=0)
    return market_open <= now <= market_close


def _minutes_since_open() -> int:
    """Minutes since 9:30 AM ET. Negative if before open."""
    now = datetime.now(tz=ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return int((now - market_open).total_seconds() / 60)


def select_flow_strike(chain: list[dict], spot: float, is_put: bool,
                       otm_mode: bool, target: float) -> tuple[float | None, str | None]:
    """Pure strike selector for a flow signal. Returns (strike, mode) or (None, None).

    Default (otm_mode False): ATM — the strike closest to spot (gold-standard, avoids the
    whale's often deep-ITM/expensive strike). otm_mode True (validated combos only): the OTM-side
    strike (puts below spot, calls above) whose mid is closest to `target` $/share — backtested
    to lift PF on AMD/INTC/META/SPY calls + TSLA puts. Falls back to ATM if no OTM strike found.
    """
    if not chain or spot <= 0:
        return None, None
    if otm_mode:
        side = [c for c in chain
                if (c.get("strike", 0) < spot if is_put else c.get("strike", 0) > spot)
                and (c.get("mid") or c.get("last_price") or 0) > 0.05]
        if side:
            pick = min(side, key=lambda c: abs((c.get("mid") or c.get("last_price") or 0) - target))
            if pick.get("strike", 0) > 0:
                return pick["strike"], "OTM"
    best = min(chain, key=lambda c: abs(c.get("strike", 0) - spot))
    s = best.get("strike", 0)
    return (s, "ATM") if s > 0 else (None, None)


async def _init_webull(settings: Settings):
    """Initialize Webull executor with retries. Returns executor or None."""
    if settings.PAPER_TRADE:
        logger.info("PAPER_TRADE=true — skipping Webull init")
        return None

    from options_owl.execution.webull_executor import WebullExecutor

    for attempt in range(1, 4):
        executor = WebullExecutor(settings)
        try:
            account_id = await executor.init()
            info = await executor.get_account_info()
            logger.info(
                f"LIVE TRADING enabled — Webull account {account_id}, "
                f"buying power: ${info.buying_power:,.2f} (attempt {attempt})"
            )
            return executor
        except Exception as exc:
            logger.error(f"Webull init attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                await asyncio.sleep(5 * attempt)

    logger.error(
        "Webull init FAILED after all retries — falling back to paper trading only. "
        "TRADES WILL NOT REACH WEBULL until next restart."
    )
    return None


# Track posted signals to avoid duplicate Discord alerts.
# Key = "TICKER:DIRECTION:STRIKE" — only post once per ticker/direction/strike per day.
_posted_signals: set[str] = set()
_posted_signals_date: str = ""


async def _post_signal_to_discord(
    signal: dict, traded: bool, trade_id: int | None = None, agent_id: str = "",
) -> None:
    """Post ML signal to Discord webhook (fire-and-forget).

    Only owlet_kody publishes to avoid duplicate alerts from all 4 bots.
    Deduplicates: same ticker/direction/strike only posted once per day.
    """
    global _posted_signals, _posted_signals_date

    if not _SIGNAL_WEBHOOK_URL:
        return
    if agent_id and agent_id != "owlet_kody":
        return

    # Reset dedup set on new trading day
    today = datetime.now(tz=ET).strftime("%Y-%m-%d")
    if today != _posted_signals_date:
        _posted_signals = set()
        _posted_signals_date = today

    # Dedup key: ticker + direction + strike
    ticker = signal.get("ticker", "")
    direction = signal.get("direction", "")
    strike = signal.get("strike", 0)
    dedup_key = f"{ticker}:{direction}:{strike}"
    if dedup_key in _posted_signals:
        return
    _posted_signals.add(dedup_key)
    try:
        import httpx

        ticker = signal["ticker"]
        direction = signal["direction"]
        is_call = direction == "CALL"
        premium = signal.get("premium", 0)
        strike = signal.get("strike", 0)
        pattern_conf = signal.get("ml_confidence", 0)
        underlying = signal.get("underlying_price", 0)
        expiry = signal.get("expiry", "0DTE")

        color = 0x00FF00 if is_call else 0xFF0000
        status = f"TRADED (#{trade_id})" if traded else "SIGNAL (rejected by pipeline)"
        emoji = "🟢" if is_call else "🔴"

        now_et = datetime.now(tz=ET)

        embed = {
            "title": f"{emoji} {direction} {ticker} ${strike:.0f} — {expiry}",
            "color": color,
            "fields": [
                {"name": "Premium", "value": f"${premium:.2f}", "inline": True},
                {"name": "Underlying", "value": f"${underlying:.2f}", "inline": True},
                {"name": "ML Confidence", "value": f"{pattern_conf:.1%}", "inline": True},
                {"name": "Status", "value": status, "inline": False},
            ],
            "footer": {"text": f"OptionsOwl ML • {now_et.strftime('%I:%M %p ET')}"},
            "timestamp": now_et.isoformat(),
        }

        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                _SIGNAL_WEBHOOK_URL,
                json={"embeds": [embed]},
                headers={"Content-Type": "application/json"},
            )
    except Exception:
        pass  # fire-and-forget, never block trading


def _ml_signal_to_trade_signal(
    ticker: str,
    direction: str,
    score: int,
    premium: float,
    strike: float,
    expiry: str,
    ml_confidence: float | None = None,
    underlying_price: float = 0,
):
    """Convert ML pipeline signal to TradeSignal for the entry pipeline."""
    from options_owl.models.signals import (
        BotSource,
        Direction,
        Sentiment,
        SignalStrength,
        TradeSignal,
    )

    dir_enum = Direction.CALL if direction == "CALL" else Direction.PUT

    # Map score to strength tier
    if score >= 150:
        strength = SignalStrength.ELITE
    elif score >= 120:
        strength = SignalStrength.STRONG
    elif score >= 90:
        strength = SignalStrength.GOOD
    elif score >= 78:
        strength = SignalStrength.MODERATE
    else:
        strength = SignalStrength.MARGINAL

    # entry_price = underlying stock price (anti_chase gate compares this to current price)
    entry_price = underlying_price if underlying_price > 0 else strike

    # stop_price: 0.5% adverse from entry (matches Discord signal convention)
    if dir_enum == Direction.CALL:
        stop_price = round(entry_price * 0.995, 2)
    else:
        stop_price = round(entry_price * 1.005, 2)

    return TradeSignal(
        ticker=ticker,
        direction=dir_enum,
        sentiment=Sentiment.BULLISH if dir_enum == Direction.CALL else Sentiment.BEARISH,
        score=score,
        strength=strength,
        bot_source=BotSource.ML_SOURCING,
        entry_price=entry_price,
        target_price=0,
        expected_move_pct=0,
        strike=strike,
        expiry=expiry or "0DTE",
        risk_reward=0,
        target_1=None,
        target_2=None,
        stop_price=stop_price,
        exit_by=None,
        atm_strike=strike,
        atm_premium=premium,
    )


async def _fetch_snapshot_from_redis(
    ticker: str, strike: float, expiry: str, option_type: str = "call",
) -> dict | None:
    """Read full option snapshot from Redis (published by harvester)."""
    try:
        from options_owl.db import redis_client

        if not redis_client.is_connected():
            return None
        contract_key = f"{ticker}:{option_type}:{strike}:{expiry}"
        snap = await redis_client.get_option_snapshot(contract_key)
        if snap and snap.get("mid", 0) > 0:
            return snap
        return None
    except Exception:
        return None


async def _resolve_atm_from_redis(
    ticker: str, expiry: str, underlying_price: float, option_type: str = "call",
) -> dict | None:
    """Find ATM strike from Redis snapshots published by harvester."""
    try:
        from options_owl.db import redis_client

        if not redis_client.is_connected():
            return None
        snaps = await redis_client.get_option_snapshots_for_ticker(ticker, expiry)
        filtered = [s for s in snaps if option_type in s.get("contract_key", "").lower()]
        if not filtered:
            return None
        best = min(filtered, key=lambda s: abs(float(s.get("strike", 0)) - underlying_price))
        return best
    except Exception:
        return None


async def _run_ml_scan_loop(paper_trader, settings: Settings) -> None:
    """ML-powered scan loop — matches the gold standard backtest.

    Reads option data streamed by the harvester via Redis (full snapshots with
    greeks, IV, volume). Falls back to Polygon REST only if Redis has no data.
    Scans every minute 9:35-11:00 AM ET.
    """
    from options_owl.sourcing.ml_pipeline import (
        EXCLUDED_TICKERS,
        TICKERS,
        TickerScanState,
        check_ticker_regime,
        fetch_live_option_chain,
        fetch_live_underlying_price,
        fetch_option_snapshot_data,
        find_atm_strike,
        get_todays_expiry,
        load_models,
        load_settings_from_env,
        run_regime_filter,
    )

    # Load ML models
    models = load_models()

    # ML pipeline settings from environment
    ml_settings = load_settings_from_env()
    # Polygon key only needed for fallback
    polygon_key = getattr(settings, "POLYGON_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")
    ml_settings.POLYGON_API_KEY = polygon_key

    agent_id = getattr(settings, "AGENT_ID", "") or "unknown"

    # Priority-ordered tickers: best backtest P&L first.
    # High-priority tickers get their signals processed first when multiple
    # fire in the same scan cycle. All tickers still run concurrently for
    # data fetching, but signal routing is sequential in this order.
    PRIORITY_TICKERS = [
        "AAPL", "AMZN", "TSLA", "SPY", "MSTR", "PLTR", "QQQ",
        "NVDA", "IWM", "AMD", "META", "NFLX",
        # Lower priority (weaker backtest performance)
        "SMCI", "GOOGL", "JPM", "BA",
    ]
    active_tickers = [
        t for t in PRIORITY_TICKERS if t in TICKERS and t not in EXCLUDED_TICKERS
    ]
    # Append any tickers in TICKERS not in PRIORITY_TICKERS (future additions)
    for t in TICKERS:
        if t not in EXCLUDED_TICKERS and t not in active_tickers:
            active_tickers.append(t)

    logger.info(
        f"ML scan loop starting for {agent_id} | "
        f"pattern_t={ml_settings.ML_PATTERN_THRESHOLD} "
        f"entry_t={ml_settings.ML_ENTRY_THRESHOLD} "
        f"put_entry_t={ml_settings.PUT_ENTRY_TIMING_THRESHOLD} "
        f"regime_t={ml_settings.ML_REGIME_THRESHOLD} | "
        f"scan=calls {ml_settings.ML_SCAN_START_MIN}-{ml_settings.ML_SCAN_END_MIN}min / "
        f"puts {ml_settings.ML_SCAN_START_MIN}-360min | "
        f"tickers={','.join(active_tickers)} | "
        f"data_source=redis (fallback=polygon)"
    )

    # Import constants for entry gates
    PREMIUM_FLOOR = 0.20
    PREMIUM_CAP = 6.0
    SPREAD_GATE_PCT = 15.0

    # Scan interval: 15s — catches new WS data within seconds of arrival.
    # ML models accumulate one snapshot per minute (TickerScanState deduplicates),
    # but scanning at 15s means we process each new minute's data immediately
    # instead of waiting for the next clock-aligned minute.
    SCAN_INTERVAL_SECONDS = 15

    while True:
        write_heartbeat()

        # Wait for market open
        if not _is_market_open():
            await asyncio.sleep(30)
            continue

        today = get_todays_expiry()
        logger.info(f"ML_SCAN: Market open — starting scan day {today}")

        # Reset per-day state — dual chain: CALL + PUT states per ticker
        ticker_states: dict[str, TickerScanState] = {}      # CALL chains
        put_ticker_states: dict[str, TickerScanState] = {}   # PUT chains
        regime_checked = False
        regime_allowed = True
        redis_hits = 0
        polygon_fallbacks = 0
        last_log_minute = -1
        last_regime_check_minute = -1

        # Intraday regime detector (spec 06)
        regime_detector = None
        regime_enabled = getattr(settings, "ENABLE_REGIME_DETECTOR", False)
        extended_scan = getattr(settings, "ENABLE_EXTENDED_SCAN", False)

        if regime_enabled:
            from options_owl.risk.regime_detector import RegimeDetector
            regime_detector = RegimeDetector(
                hysteresis_checks=getattr(settings, "REGIME_HYSTERESIS_CHECKS", 2),
                min_hold_minutes=getattr(settings, "REGIME_MIN_HOLD_MIN", 15),
                hard_reversal_pct=getattr(settings, "REGIME_HARD_REVERSAL_PCT", 0.5),
                choppy_size_mult=getattr(settings, "REGIME_CHOPPY_SIZE_MULT", 0.6),
            )
            # Share with position_monitor for stop tightening (spec 08)
            try:
                from options_owl.execution.position_monitor import set_regime_detector
                set_regime_detector(regime_detector)
            except ImportError:
                pass

        try:
            while _is_market_open():
                write_heartbeat()
                minute = _minutes_since_open()

                # Regime filter at minute 15 (9:45 AM ET)
                if not regime_checked and minute >= 15:
                    regime_checked = True
                    regime_allowed = await run_regime_filter(models, ml_settings)
                    if not regime_allowed:
                        logger.info("ML_SCAN: Day skipped by regime filter. Waiting for EOD.")
                        while _is_market_open():
                            write_heartbeat()
                            await asyncio.sleep(60)
                        break

                # Update intraday regime detector every 5 minutes
                if (regime_detector and minute >= 15
                        and minute - last_regime_check_minute >= 5):
                    last_regime_check_minute = minute
                    try:
                        candle_cache_for_regime = getattr(paper_trader, '_candle_cache', None)
                        if candle_cache_for_regime:
                            await regime_detector.update(candle_cache_for_regime)
                    except Exception:
                        logger.debug("ML_SCAN: Regime detector update failed")

                # Only scan during the configured window
                if minute < ml_settings.ML_SCAN_START_MIN:
                    await asyncio.sleep(10)
                    continue

                # Scan window: CALLs stop at ML_SCAN_END_MIN (90),
                # PUTs scan all day until 30 min before close (360 min)
                PUT_SCAN_END = 360  # 3:30 PM ET (30 min before close)
                effective_call_end = 330 if extended_scan else ml_settings.ML_SCAN_END_MIN
                call_window_open = minute <= effective_call_end
                put_window_open = minute <= PUT_SCAN_END

                if not call_window_open and not put_window_open:
                    logger.info(
                        f"ML_SCAN: All scan windows closed (minute {minute}). "
                        f"Waiting for EOD."
                    )
                    while _is_market_open():
                        write_heartbeat()
                        await asyncio.sleep(60)
                    break

                # Scan all tickers CONCURRENTLY (not sequentially)
                # Each ticker's data fetch + ML runs in parallel, bounded by
                # a semaphore to limit concurrent Polygon API calls.
                scan_start = time.monotonic()
                signals_emitted = 0
                new_snapshots = 0
                expiry = today

                # Limit concurrent Polygon REST calls to avoid rate limits
                polygon_sem = asyncio.Semaphore(5)

                async def _scan_one_ticker(ticker):
                    """Fetch data + run ML for one ticker. Returns (signal, counts) or None."""
                    nonlocal redis_hits, polygon_fallbacks

                    # Initialize CALL + PUT states for this ticker
                    if ticker not in ticker_states:
                        ticker_states[ticker] = TickerScanState(expiry=expiry)
                    if ticker not in put_ticker_states:
                        put_ticker_states[ticker] = TickerScanState(expiry=expiry)

                    call_state = ticker_states[ticker]
                    put_state = put_ticker_states[ticker]

                    # Skip if both directions already emitted
                    if call_state.entry_emitted and put_state.entry_emitted:
                        return None, 0

                    # Per-ticker regime check (after minute 15)
                    if minute >= 15 and models.regime_model is not None:
                        ticker_ok = await check_ticker_regime(
                            ticker, models, ml_settings
                        )
                        if not ticker_ok:
                            return None, 0

                    # --- Get underlying price (Redis first, Polygon fallback) ---
                    underlying_price = None
                    try:
                        from options_owl.db import redis_client

                        if redis_client.is_connected():
                            result = await redis_client.get_price(ticker, max_age=90)
                            if result:
                                underlying_price = result[0]
                    except Exception:
                        pass

                    if not underlying_price or underlying_price <= 0:
                        async with polygon_sem:
                            underlying_price = await asyncio.wait_for(
                                fetch_live_underlying_price(polygon_key, ticker),
                                timeout=10,
                            )
                    if not underlying_price or underlying_price <= 0:
                        return None, 0

                    # --- Determine direction from underlying price action ---
                    direction = "CALL"
                    if call_state.underlyings and len(call_state.underlyings) >= 3:
                        open_u = call_state.underlyings[0]
                        curr_u = underlying_price
                        if open_u > 0 and curr_u > 0:
                            move_pct = (curr_u - open_u) / open_u * 100
                            if move_pct < -0.15:
                                direction = "PUT"

                    # --- Resolve CALL ATM strike (re-resolve if underlying drifts >1%) ---
                    _need_call_resolve = call_state.strike <= 0
                    if (
                        not _need_call_resolve
                        and call_state.strike_resolved_price > 0
                        and underlying_price > 0
                        and not call_state.entry_emitted
                    ):
                        _drift_pct = abs(underlying_price - call_state.strike_resolved_price) / call_state.strike_resolved_price * 100
                        if _drift_pct >= 1.0:
                            logger.info(
                                f"ML_SCAN: {ticker} CALL underlying drifted {_drift_pct:.1f}% "
                                f"(${call_state.strike_resolved_price:.2f}→${underlying_price:.2f}), "
                                f"re-resolving ATM strike (was ${call_state.strike:.0f})"
                            )
                            _need_call_resolve = True
                            # Reset accumulated data — it's for the old strike
                            call_state.strike = 0.0
                            call_state.closes.clear()
                            call_state.volumes.clear()
                            call_state.ivs.clear()
                            call_state.deltas.clear()
                            call_state.thetas.clear()
                            call_state.vegas.clear()
                            call_state.bids.clear()
                            call_state.asks.clear()
                            call_state.bid_sizes.clear()
                            call_state.ask_sizes.clear()
                            call_state.last_append_minute = -1
                            call_state.data_changed = False

                    if _need_call_resolve:
                        atm_snap = await _resolve_atm_from_redis(
                            ticker, expiry, underlying_price, option_type="call"
                        )
                        if not atm_snap:
                            tomorrow = (
                                datetime.now(tz=ET) + timedelta(days=1)
                            ).strftime("%Y-%m-%d")
                            atm_snap = await _resolve_atm_from_redis(
                                ticker, tomorrow, underlying_price, option_type="call"
                            )
                            if atm_snap:
                                call_state.expiry = tomorrow

                        if atm_snap:
                            call_state.strike = float(atm_snap.get("strike", 0))
                            if call_state.expiry == expiry:
                                call_state.expiry = atm_snap.get("expiry_date", expiry)
                            redis_hits += 1
                        else:
                            async with polygon_sem:
                                chain = await fetch_live_option_chain(
                                    polygon_key, ticker, expiry
                                )
                                if not chain:
                                    tomorrow = (
                                        datetime.now(tz=ET) + timedelta(days=1)
                                    ).strftime("%Y-%m-%d")
                                    chain = await fetch_live_option_chain(
                                        polygon_key, ticker, tomorrow
                                    )
                                    if chain:
                                        call_state.expiry = tomorrow
                                    else:
                                        return None, 0
                                else:
                                    call_state.expiry = expiry

                            atm = find_atm_strike(chain, underlying_price)
                            if not atm:
                                return None, 0
                            call_state.strike = atm.get("strike", 0)
                            polygon_fallbacks += 1

                        if call_state.strike <= 0:
                            return None, 0
                        call_state.strike_resolved_price = underlying_price
                        logger.info(
                            f"ML_SCAN: {ticker} CALL ATM strike=${call_state.strike:.0f} "
                            f"expiry={call_state.expiry} (underlying=${underlying_price:.2f})"
                        )

                    # --- Resolve PUT ATM strike (re-resolve if underlying drifts >1%) ---
                    _need_put_resolve = put_state.strike <= 0
                    if (
                        not _need_put_resolve
                        and put_state.strike_resolved_price > 0
                        and underlying_price > 0
                        and not put_state.entry_emitted
                    ):
                        _put_drift = abs(underlying_price - put_state.strike_resolved_price) / put_state.strike_resolved_price * 100
                        if _put_drift >= 1.0:
                            logger.info(
                                f"ML_SCAN: {ticker} PUT underlying drifted {_put_drift:.1f}% "
                                f"(${put_state.strike_resolved_price:.2f}→${underlying_price:.2f}), "
                                f"re-resolving ATM strike (was ${put_state.strike:.0f})"
                            )
                            _need_put_resolve = True
                            put_state.strike = 0.0
                            put_state.closes.clear()
                            put_state.volumes.clear()
                            put_state.ivs.clear()
                            put_state.deltas.clear()
                            put_state.thetas.clear()
                            put_state.vegas.clear()
                            put_state.bids.clear()
                            put_state.asks.clear()
                            put_state.bid_sizes.clear()
                            put_state.ask_sizes.clear()
                            put_state.last_append_minute = -1
                            put_state.data_changed = False

                    if _need_put_resolve:
                        put_atm = await _resolve_atm_from_redis(
                            ticker, call_state.expiry, underlying_price,
                            option_type="put",
                        )
                        if not put_atm:
                            tomorrow = (
                                datetime.now(tz=ET) + timedelta(days=1)
                            ).strftime("%Y-%m-%d")
                            put_atm = await _resolve_atm_from_redis(
                                ticker, tomorrow, underlying_price,
                                option_type="put",
                            )
                            if put_atm:
                                put_state.expiry = tomorrow
                        if put_atm:
                            put_state.strike = float(put_atm.get("strike", 0))
                            if put_state.expiry == expiry:
                                put_state.expiry = put_atm.get("expiry_date", expiry)
                            put_state.strike_resolved_price = underlying_price
                            redis_hits += 1
                            logger.info(
                                f"ML_SCAN: {ticker} PUT ATM strike=${put_state.strike:.0f} "
                                f"expiry={put_state.expiry} (underlying=${underlying_price:.2f})"
                            )

                    # --- Fetch CALL snapshot (always, for underlying tracking) ---
                    call_snap = await _fetch_snapshot_from_redis(
                        ticker, call_state.strike, call_state.expiry,
                        option_type="call",
                    )
                    if call_snap:
                        redis_hits += 1
                    else:
                        async with polygon_sem:
                            call_snap = await asyncio.wait_for(
                                fetch_option_snapshot_data(
                                    polygon_key, ticker, call_state.strike,
                                    call_state.expiry,
                                ),
                                timeout=10,
                            )
                        if call_snap:
                            polygon_fallbacks += 1

                    if not call_snap:
                        call_snap = {
                            "mid": 0, "bid": 0, "ask": 0,
                            "iv": 0, "delta": 0, "theta": 0, "vega": 0,
                            "volume": 0, "underlying_price": underlying_price,
                            "bid_size": 0, "ask_size": 0,
                        }
                    elif call_snap.get("underlying_price", 0) <= 0:
                        call_snap["underlying_price"] = underlying_price

                    prev_len = len(call_state.closes)
                    call_state.append_snapshot(call_snap, minute)
                    ticker_new_snaps = 1 if len(call_state.closes) > prev_len else 0

                    # --- Fetch PUT snapshot (when strike resolved) ---
                    if put_state.strike > 0:
                        put_snap = await _fetch_snapshot_from_redis(
                            ticker, put_state.strike, put_state.expiry,
                            option_type="put",
                        )
                        if put_snap:
                            redis_hits += 1
                        if not put_snap:
                            put_snap = {
                                "mid": 0, "bid": 0, "ask": 0,
                                "iv": 0, "delta": 0, "theta": 0, "vega": 0,
                                "volume": 0, "underlying_price": underlying_price,
                                "bid_size": 0, "ask_size": 0,
                            }
                        elif put_snap.get("underlying_price", 0) <= 0:
                            put_snap["underlying_price"] = underlying_price
                        put_state.append_snapshot(put_snap, minute)

                    # --- Run ML on the correct chain based on direction ---
                    signal = None

                    if direction == "CALL" and not call_state.entry_emitted and call_window_open:
                        if call_state.data_changed:
                            signal = await _run_ml_for_ticker(
                                ticker, minute, call_state, models, ml_settings,
                                PREMIUM_FLOOR, PREMIUM_CAP, SPREAD_GATE_PCT,
                                forced_direction="CALL",
                            )
                    elif direction == "PUT" and not put_state.entry_emitted and put_window_open:
                        if put_state.data_changed and put_state.strike > 0:
                            signal = await _run_ml_for_ticker(
                                ticker, minute, put_state, models, ml_settings,
                                PREMIUM_FLOOR, PREMIUM_CAP, SPREAD_GATE_PCT,
                                forced_direction="PUT",
                                call_state=call_state,
                            )

                    if signal:
                        # Regime direction gate (spec 06/07/10)
                        # When ENABLE_DYNAMIC_PUTS is on, skip hard direction blocking
                        # here — the DirectionSlotGate in the pipeline handles soft
                        # slot-based throttling instead (allows some counter-trend trades).
                        dynamic_puts = getattr(settings, "ENABLE_DYNAMIC_PUTS", False)
                        sig_direction = signal.get("direction", "CALL")
                        if regime_detector and not dynamic_puts:
                            from options_owl.risk.regime_detector import (
                                get_allowed_directions,
                            )
                            allowed = get_allowed_directions(
                                minute,
                                regime_detector.state,
                                extended_scan_enabled=extended_scan,
                            )
                            if sig_direction not in allowed:
                                logger.info(
                                    f"ML_SCAN REGIME_BLOCK: {ticker} "
                                    f"{sig_direction} blocked by "
                                    f"regime={regime_detector.state.value} "
                                    f"allowed={allowed}"
                                )
                                signal = None

                    return signal, ticker_new_snaps

                async def _scan_ticker_safe(ticker):
                    """Wrap _scan_one_ticker with per-ticker timeout and error handling."""
                    try:
                        # 25s outer: a Redis-miss ticker does up to 2 sequential Polygon
                        # snapshot fetches (call+put) at 10s each — the old 15s killed those
                        # mid-success (MSTR/PLTR/NFLX false timeouts). Inner 10s caps still
                        # prevent any single hung call from stalling the loop.
                        return await asyncio.wait_for(
                            _scan_one_ticker(ticker), timeout=25,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"ML_SCAN: {ticker} timed out at minute {minute}")
                        return None, 0
                    except Exception:
                        logger.exception(f"ML_SCAN: {ticker} error at minute {minute}")
                        return None, 0

                # Run ALL tickers concurrently
                results = await asyncio.gather(
                    *[_scan_ticker_safe(t) for t in active_tickers]
                )

                # Process signals sequentially (trade evaluation needs serialization)
                for ticker, (signal, ticker_snaps) in zip(active_tickers, results):
                    new_snapshots += ticker_snaps
                    if signal:
                        try:
                            trade_signal = _ml_signal_to_trade_signal(**signal)
                            signal_id = -int(time.time() * 1000) % 1_000_000
                            result = await paper_trader.evaluate_and_trade(
                                trade_signal,
                                signal_id,
                                ml_confidence=signal.get("ml_confidence"),
                            )
                            call_state = ticker_states.get(ticker)
                            put_state = put_ticker_states.get(ticker)
                            if result:
                                if signal["direction"] == "PUT" and put_state:
                                    put_state.entry_emitted = True
                                elif call_state:
                                    call_state.entry_emitted = True
                                signals_emitted += 1
                                logger.info(
                                    f"ML_SCAN TRADE: {ticker} {signal['direction']} "
                                    f"pattern={signal['ml_confidence']:.3f} "
                                    f"premium=${signal['premium']:.2f} "
                                    f"trade_id={result['trade_id']}"
                                )
                                asyncio.create_task(
                                    _post_signal_to_discord(
                                        signal, True, result["trade_id"], agent_id,
                                    )
                                )
                            else:
                                logger.info(
                                    f"ML_SCAN REJECT: {ticker} {signal['direction']} "
                                    f"pattern={signal['ml_confidence']:.3f} "
                                    f"— entry pipeline rejected"
                                )
                                asyncio.create_task(
                                    _post_signal_to_discord(
                                        signal, False, agent_id=agent_id,
                                    )
                                )
                        except Exception:
                            logger.exception(f"ML_SCAN: error routing {ticker}")

                scan_elapsed = time.monotonic() - scan_start

                # Log on new data or every 5 minutes
                if new_snapshots > 0 or (minute != last_log_minute and minute % 5 == 0):
                    if new_snapshots > 0 or signals_emitted > 0:
                        logger.info(
                            f"ML_SCAN: min={minute} new_snaps={new_snapshots} "
                            f"signals={signals_emitted} ({scan_elapsed:.1f}s) "
                            f"redis={redis_hits} polygon_fb={polygon_fallbacks}"
                        )
                    last_log_minute = minute

                # Sleep 15s — catches new WS data within seconds of arrival
                await asyncio.sleep(SCAN_INTERVAL_SECONDS)

        except Exception:
            logger.exception("ML_SCAN: unhandled exception in scan loop")

        # EOD summary
        call_signals = sum(1 for s in ticker_states.values() if s.entry_emitted)
        put_signals = sum(1 for s in put_ticker_states.values() if s.entry_emitted)
        total_signals = call_signals + put_signals
        total_scanned = sum(len(s.closes) for s in ticker_states.values())
        logger.info(
            f"ML_SCAN: EOD {today} — {total_signals} signals traded "
            f"({call_signals} CALL, {put_signals} PUT), "
            f"{total_scanned} total snapshots, "
            f"redis_hits={redis_hits} polygon_fallbacks={polygon_fallbacks}"
        )

        # Wait for next trading day
        while not _is_market_open():
            write_heartbeat()
            await asyncio.sleep(60)


async def _run_ml_for_ticker(
    ticker: str,
    minute: int,
    state,
    models,
    settings,
    premium_floor: float,
    premium_cap: float,
    spread_gate_pct: float,
    forced_direction: str = "CALL",
    call_state=None,
) -> dict | None:
    """Run V3 ML models for a single ticker at a single minute.

    Returns signal dict if entry should be taken, None otherwise.

    With dual-chain scanning, ``state`` already contains the correct option
    chain data (CALL state for CALL direction, PUT state for PUT direction).
    ``forced_direction`` tells us which direction we're evaluating — no more
    post-hoc direction detection or premium swapping.

    For PUT direction, if a dedicated PUT pattern model is loaded, it is used
    instead of the generic CALL pattern model. ``call_state`` provides CALL
    chain data for cross-chain features (iv_skew, put_call_volume_ratio).
    """
    import numpy as np

    from options_owl.sourcing.ml_pipeline import (
        compute_entry_timing_features,
        compute_pattern_features,
        compute_put_pattern_features,
    )

    arrays = state.to_numpy()
    idx = len(state.closes) - 1

    if idx < 5:
        return None

    # Step 1: Pattern model — use dedicated PUT model when available
    use_put_model = (
        forced_direction == "PUT"
        and models.put_pattern_model is not None
    )

    if use_put_model:
        # Build cross-chain arrays for PUT-specific features
        call_ivs = None
        call_volumes = None
        if call_state and len(call_state.ivs) > 0:
            call_ivs = np.array(call_state.ivs, dtype=np.float64)
            call_volumes = np.array(call_state.volumes, dtype=np.float64)

        feat = compute_put_pattern_features(
            arrays["closes"], arrays["volumes"], arrays["ivs"],
            arrays["deltas"], arrays["thetas"], arrays["underlyings"],
            arrays["bids"], arrays["asks"], idx, state.opening_price,
            vegas=arrays.get("vegas"),
            bid_sizes=arrays.get("bid_sizes"),
            ask_sizes=arrays.get("ask_sizes"),
            call_ivs=call_ivs, call_volumes=call_volumes,
        )
        if feat is None:
            return None

        X_pattern = np.array(
            [[feat.get(f, 0) for f in models.put_pattern_features]],
            dtype=np.float32,
        )
        pattern_conf = float(models.put_pattern_model.predict(X_pattern)[0])
        threshold = models.put_pattern_threshold
    else:
        feat = compute_pattern_features(
            arrays["closes"], arrays["volumes"], arrays["ivs"],
            arrays["deltas"], arrays["thetas"], arrays["underlyings"],
            arrays["bids"], arrays["asks"], idx, state.opening_price,
        )
        if feat is None:
            return None

        X_pattern = np.array(
            [[feat.get(f, 0) for f in models.pattern_features]],
            dtype=np.float32,
        )
        pattern_conf = float(models.pattern_model.predict(X_pattern)[0])
        threshold = settings.ML_PATTERN_THRESHOLD

    if pattern_conf < threshold:
        return None

    model_tag = "PUT_MODEL" if use_put_model else "PATTERN"
    logger.info(
        f"ML_SCAN: {ticker} min={minute} {model_tag} PASS {forced_direction} "
        f"conf={pattern_conf:.3f} >= {threshold}"
    )

    # Step 2: Entry timing model.
    # CALLs use the CALL entry timing model (entry_timing.txt).
    # PUTs use the dedicated PUT entry timing model (put_entry_timing.txt) if available,
    # otherwise PUTs skip entry timing entirely (matches backtest_afternoon.py behavior).
    entry_conf = None
    if forced_direction == "CALL" and models.entry_model and models.entry_features:
        et_feat = compute_entry_timing_features(
            arrays["closes"], arrays["volumes"],
            arrays["bids"], arrays["asks"],
            arrays["bid_sizes"], arrays["ask_sizes"],
            arrays["ivs"], arrays["deltas"], arrays["thetas"], arrays["vegas"],
            arrays["underlyings"],
            arrays["stock_closes"], arrays["stock_highs"], arrays["stock_lows"],
            idx, models.entry_features,
        )
        if et_feat is not None:
            X_entry = np.array(
                [[et_feat.get(f, 0) for f in models.entry_features]],
                dtype=np.float32,
            )
            entry_conf = float(models.entry_model.predict(X_entry)[0])
            if entry_conf < settings.ML_ENTRY_THRESHOLD:
                logger.info(
                    f"ML_SCAN: {ticker} min={minute} ENTRY BLOCKED "
                    f"conf={entry_conf:.3f} < {settings.ML_ENTRY_THRESHOLD}"
                )
                return None
    elif forced_direction == "PUT" and models.put_entry_model and models.put_entry_features:
        put_threshold = settings.PUT_ENTRY_TIMING_THRESHOLD
        if put_threshold > 0:
            et_feat = compute_entry_timing_features(
                arrays["closes"], arrays["volumes"],
                arrays["bids"], arrays["asks"],
                arrays["bid_sizes"], arrays["ask_sizes"],
                arrays["ivs"], arrays["deltas"], arrays["thetas"], arrays["vegas"],
                arrays["underlyings"],
                arrays["stock_closes"], arrays["stock_highs"], arrays["stock_lows"],
                idx, models.put_entry_features,
            )
            if et_feat is not None:
                X_entry = np.array(
                    [[et_feat.get(f, 0) for f in models.put_entry_features]],
                    dtype=np.float32,
                )
                entry_conf = float(models.put_entry_model.predict(X_entry)[0])
                if entry_conf < put_threshold:
                    logger.info(
                        f"ML_SCAN: {ticker} min={minute} PUT_ENTRY BLOCKED "
                        f"conf={entry_conf:.3f} < {put_threshold}"
                    )
                    return None

    # Step 3: Premium from the correct chain (state already has it)
    current_ask = state.asks[-1] if state.asks else 0
    current_bid = state.bids[-1] if state.bids else 0
    current_mid = state.closes[-1] if state.closes else 0
    premium = current_ask if current_ask > 0 else current_mid

    # Premium gates
    if premium <= 0:
        return None
    if premium < premium_floor:
        return None
    if premium > premium_cap:
        return None

    if current_bid > 0 and premium > 0:
        spread_pct = (premium - current_bid) / premium * 100
        if spread_pct > spread_gate_pct:
            return None

    # Map ML confidence to score.
    # CALL model: conf 0.74-1.0 → score 74-100 (threshold 0.74, so 74+ passes)
    # PUT model: conf 0.65-1.0 → scale to 78-100 so MIN_SCORE=78 works correctly.
    # Without scaling, PUT conf 0.70 → score 70 (rejected), even though it's well
    # above the PUT threshold of 0.65.
    if use_put_model:
        # Linear scale: threshold (0.65) → 78, conf 1.0 → 100
        score = int(78 + (pattern_conf - threshold) / (1.0 - threshold) * 22)
        score = max(0, min(100, score))
    else:
        score = int(pattern_conf * 100)

    logger.info(
        f"ML_SCAN: {ticker} min={minute} SIGNAL {forced_direction} "
        f"pattern={pattern_conf:.3f} entry={f'{entry_conf:.3f}' if entry_conf is not None else 'N/A'} "
        f"score={score} premium=${premium:.2f} strike=${state.strike:.0f}"
    )

    underlying_price = state.underlyings[-1] if state.underlyings else 0

    return {
        "ticker": ticker,
        "direction": forced_direction,
        "score": score,
        "premium": premium,
        "strike": state.strike,
        "expiry": state.expiry,
        "ml_confidence": pattern_conf,
        "underlying_price": underlying_price,
    }


async def _supervised_task(
    name: str, coro_factory, *args, max_restarts: int = 10, backoff_base: float = 5,
) -> None:
    """Run a coroutine with automatic restart on crash.

    If the task crashes, it's restarted with exponential backoff (capped at 60s).
    After max_restarts consecutive failures without 5 minutes of healthy run time,
    the task is considered permanently failed and the exception is re-raised.
    """
    consecutive_failures = 0
    while consecutive_failures < max_restarts:
        start_time = time.monotonic()
        try:
            logger.info(f"SUPERVISOR: starting {name}")
            await coro_factory(*args)
            # Clean exit — restart anyway (scan loop should run forever)
            logger.warning(f"SUPERVISOR: {name} exited cleanly — restarting")
            consecutive_failures = 0
        except asyncio.CancelledError:
            logger.info(f"SUPERVISOR: {name} cancelled")
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            if elapsed > 300:  # ran for 5+ minutes = healthy before crash
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            backoff = min(backoff_base * (2 ** consecutive_failures), 60)
            logger.error(
                f"SUPERVISOR: {name} crashed ({consecutive_failures}/{max_restarts}): "
                f"{type(exc).__name__}: {exc} — restarting in {backoff:.0f}s"
            )
            logger.debug(f"SUPERVISOR: {name} traceback:", exc_info=True)
            await asyncio.sleep(backoff)

    logger.critical(f"SUPERVISOR: {name} failed {max_restarts} times — giving up")
    raise RuntimeError(f"{name} exceeded max restarts ({max_restarts})")


async def run_bot(settings: Settings) -> None:
    """Main bot coroutine — initializes everything and runs forever."""
    agent_id = getattr(settings, "AGENT_ID", "") or "unknown"
    logger.info(f"OptionsOwl bot starting ({agent_id})")
    write_heartbeat()

    # --- Webull ---
    webull_executor = await _init_webull(settings)

    # --- PaperTrader ---
    from options_owl.execution.paper_trader import PaperTrader

    paper_trader = PaperTrader(settings, webull_executor=webull_executor)
    await paper_trader.init()
    status = await paper_trader.get_status()
    mode = "LIVE (Webull)" if webull_executor else "PAPER ONLY"
    logger.info(f"{mode} trading enabled (${settings.PORTFOLIO_SIZE:,.2f} portfolio)")
    logger.info(f"\n{status}")

    # --- Market data stream ---
    from options_owl.collectors.market_data_stream import MarketDataStream

    market_stream = MarketDataStream(settings)
    await market_stream.start()
    logger.info(f"Market data stream started (provider={market_stream.provider.value})")
    paper_trader.market_stream = market_stream

    # --- Redis ---
    if getattr(settings, "ENABLE_REDIS", False):
        try:
            from options_owl.db import redis_client

            await redis_client.init_redis(
                getattr(settings, "REDIS_URL", "redis://redis:6379/0")
            )
            logger.info("Redis cross-agent coordination enabled")
            if agent_id:
                paper = getattr(settings, "PAPER_TRADE", True)
                kill = getattr(settings, "WEBULL_KILL_SWITCH", False)
                # Always sync env vars to Redis on startup — stale Redis values
                # from a previous config (e.g. paper=true) must not override
                # docker-compose changes (e.g. paper=false after deploy).
                await redis_client.set_control(agent_id, "paper_mode", paper)
                await redis_client.set_control(agent_id, "kill_switch", kill)
                logger.info(f"Dashboard controls seeded: paper={paper} kill={kill}")
        except Exception as exc:
            logger.warning(f"Redis init failed (continuing without coordination): {exc}")

    # --- PostgreSQL ---
    if getattr(settings, "ENABLE_POSTGRES", False):
        try:
            from options_owl.db import postgres as pg

            await pg.init_pool(getattr(settings, "DATABASE_URL", None))
            logger.info("PostgreSQL shared DB connected")
        except Exception as exc:
            logger.warning(f"PostgreSQL init failed (continuing without): {exc}")

    # --- Supervised background tasks ---
    from options_owl.execution.position_monitor import run_position_monitor

    async def _monitor_factory():
        await run_position_monitor(paper_trader, market_stream, discord_client=None)

    async def _scan_factory():
        await _run_ml_scan_loop(paper_trader, settings)

    async def _heartbeat_factory():
        while True:
            write_heartbeat()
            await asyncio.sleep(30)

    monitor_task = asyncio.create_task(
        _supervised_task("position_monitor", _monitor_factory)
    )
    scan_task = asyncio.create_task(
        _supervised_task("ml_scan_loop", _scan_factory)
    )
    heartbeat_task = asyncio.create_task(_heartbeat_factory())

    # UW flow signal source (Track 4) — whale sweeps → entry pipeline (gate-bypassed) → V7 exits.
    # Gated behind ENABLE_UW_FLOW_SIGNAL (paper-first). Off = no task created.
    supervised = [monitor_task, scan_task, heartbeat_task]
    if getattr(settings, "ENABLE_UW_FLOW_SIGNAL", False) and getattr(settings, "UNUSUAL_WHALES_API_KEY", ""):
        from options_owl.collectors.polygon_options import polygon_option_chain
        from options_owl.collectors.uw_flow_collector import (
            flow_signal_to_trade_signal,
            run_uw_flow_collector,
        )
        from options_owl.models.signals import Direction as _FlowDir
        from options_owl.sourcing.ml_pipeline import fetch_live_underlying_price
        _flow_id = {"n": 0}
        _flow_pk = getattr(settings, "POLYGON_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")
        _otm_on = bool(getattr(settings, "ENABLE_FLOW_OTM_STRIKE", False))
        _otm_call = {t.strip().upper() for t in
                     getattr(settings, "FLOW_OTM_CALL_TICKERS", "").split(",") if t.strip()}
        _otm_put = {t.strip().upper() for t in
                    getattr(settings, "FLOW_OTM_PUT_TICKERS", "").split(",") if t.strip()}
        _otm_target = float(getattr(settings, "FLOW_OTM_TARGET_PREMIUM", 2.0) or 2.0)

        async def _resolve_flow_strike(fs):
            """Resolve the NEAREST-expiry strike for a flow signal.

            Matches the flow backtest (dte0 = nearest DTE) and the V7 exit engine (built for
            0DTE/short-DTE momentum). The whale's OWN expiry (fs.expiry) is IGNORED — it can be
            far-dated (e.g. a 45-DTE index-hedge put) which is UNTESTED (thetadata only has 0-4
            DTE) and the FSM does not fit. Walks today → next business days, first non-empty chain
            wins. Strike = ATM, or cheaper OTM for the validated combos (flag-gated). Returns
            (strike, expiry, mode, spot) or (None, None, None, spot) → caller SKIPS the trade
            (never falls back to the whale's far-dated contract).
            """
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            from zoneinfo import ZoneInfo

            from options_owl.db import redis_client
            spot = 0.0
            try:
                is_put = fs.direction == _FlowDir.PUT
                otype = "put" if is_put else "call"
                otm_wl = _otm_put if is_put else _otm_call
                use_otm = _otm_on and fs.ticker.upper() in otm_wl
                _rconn = redis_client.is_connected()
                # Underlying: harvester's Redis price first, Polygon only on miss.
                if _rconn:
                    r = await redis_client.get_price(fs.ticker, max_age=120)
                    if r and r[0] > 0:
                        spot = r[0]
                if spot <= 0:
                    spot = await asyncio.wait_for(
                        fetch_live_underlying_price(_flow_pk, fs.ticker), timeout=10) or 0.0
                if spot <= 0:
                    return None, None, None, 0.0
                # Nearest tradeable expiry: today, then the next business days.
                base = _dt.now(tz=ZoneInfo("America/New_York")).date()
                candidates, d = [], base
                while len(candidates) < 5:
                    if d.weekday() < 5:
                        candidates.append(d.strftime("%Y-%m-%d"))
                    d = d + _td(days=1)
                for exp in candidates:
                    chain = None
                    # Harvester chain from Redis (ATM±10%, covers ATM + OTM) — no Polygon hit.
                    if _rconn:
                        snaps = await redis_client.get_option_snapshots_for_ticker(fs.ticker, exp)
                        rc = [{"strike": float(s.get("strike", 0)), "mid": float(s.get("mid", 0) or 0)}
                              for s in (snaps or [])
                              if otype in str(s.get("contract_key", "")).lower()
                              and float(s.get("strike", 0)) > 0]
                        if rc:
                            chain = rc
                    # Polygon fallback only when Redis has nothing for this expiry.
                    if not chain:
                        try:
                            chain = await asyncio.wait_for(
                                polygon_option_chain(_flow_pk, fs.ticker, exp, option_type=otype),
                                timeout=10) or None
                        except asyncio.TimeoutError:
                            chain = None
                    if not chain:
                        continue
                    strike, mode = select_flow_strike(chain, spot, is_put, use_otm, _otm_target)
                    if strike:
                        return strike, exp, mode, spot
                return None, None, None, spot
            except Exception as exc:
                logger.warning(f"UW_FLOW: strike resolve failed for {fs.ticker}: {exc!r}")
                return None, None, None, spot

        async def _on_flow_signal(fs):
            _flow_id["n"] -= 1  # negative synthetic id (distinct from Discord/ML)
            strike, expiry, mode, spot = await _resolve_flow_strike(fs)
            if not strike or spot <= 0:
                # Could not resolve a near-dated contract — SKIP. Do NOT trade the whale's
                # far-dated expiry: it's untested and the V7 exits don't fit it (see #266).
                logger.info(
                    f"UW_FLOW: {fs.ticker} {fs.direction.value} SKIPPED — no near-dated contract "
                    f"resolved (whale exp={fs.expiry})")
                return
            ts = flow_signal_to_trade_signal(fs)
            is_put = fs.direction == _FlowDir.PUT
            # Flow carries no stop (exits via the V7 FSM); assign the same 0.5% underlying stop ML
            # uses so it clears the stop_price gate + paper_trader validation.
            ts = ts.model_copy(update={
                "strike": strike, "atm_strike": strike, "expiry": expiry,
                "entry_price": spot,
                "stop_price": round(spot * (1.005 if is_put else 0.995), 2),
            })
            logger.info(
                f"UW_FLOW: {fs.ticker} {mode} strike ${strike:g} exp={expiry} "
                f"(whale ${fs.strike:g}/{fs.expiry}) stop=${ts.stop_price:g}")
            logger.info(
                f"UW_FLOW: {fs.direction.value} {fs.ticker} ${fs.total_premium:,.0f} "
                f"ask={fs.ask_frac:.0%} voi={fs.volume_oi_ratio:.1f} -> entry pipeline"
            )
            await paper_trader.evaluate_and_trade(ts, _flow_id["n"])

        async def _flow_factory():
            await run_uw_flow_collector(settings, _on_flow_signal)

        supervised.append(asyncio.create_task(_supervised_task("uw_flow_collector", _flow_factory)))
        logger.info("UW_FLOW: collector task started (ENABLE_UW_FLOW_SIGNAL=true)")

    logger.info(
        f"OptionsOwl bot {agent_id} fully initialized — "
        f"ML scanning (redis→harvester), supervised tasks running"
    )

    # Wait for any task to fail permanently (after all retries exhausted)
    done, pending = await asyncio.wait(
        supervised,
        return_when=asyncio.FIRST_EXCEPTION,
    )
    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    for task in done:
        if task.exception():
            logger.error(f"Critical supervised task permanently failed: {task.exception()}")
            raise task.exception()


def main() -> None:
    """Entry point with retry loop (mirrors run_collector_with_retry)."""
    configure_logging()
    settings = Settings()

    # Polygon freshness check (fail-fast for live mode)
    from options_owl.main import check_polygon_realtime_entitlement

    check_polygon_realtime_entitlement(settings)

    backoff = 5
    attempt = 0
    max_retries = 50

    while attempt < max_retries:
        attempt += 1
        try:
            logger.info(f"Starting bot (attempt {attempt})")
            write_heartbeat()
            asyncio.run(run_bot(settings))
            logger.warning(f"Bot exited cleanly — restarting in {backoff}s")
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            return
        except Exception as exc:
            logger.error(f"Bot crashed: {type(exc).__name__}: {exc}")
            logger.debug("Full traceback:", exc_info=True)

        from options_owl.main import _cleanup_connections

        _cleanup_connections()
        logger.info(f"Retrying in {backoff}s")
        time.sleep(backoff)
        backoff = min(backoff * 2, 300)

    logger.critical(f"Bot failed after {max_retries} attempts — giving up")
    sys.exit(1)


if __name__ == "__main__":
    main()
