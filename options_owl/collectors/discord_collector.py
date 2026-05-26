from __future__ import annotations

import asyncio
import re
from datetime import datetime

import discord
from loguru import logger

from options_owl.config.settings import Settings
from options_owl.journal import db
from options_owl.models.signals import (
    BotSource,
    DailyPerformance,
    Direction,
    PerformanceEntry,
    Sentiment,
    SignalStrength,
    TradeSignal,
    WatchlistEntry,
)

# ---------------------------------------------------------------------------
# Bot author detection
# ---------------------------------------------------------------------------

BOT_NAME_MAP: dict[str, BotSource] = {
    "captain hook": BotSource.CAPTAIN_HOOK,
    "neverland pan": BotSource.NEVERLAND_PAN,
    "tinker": BotSource.TINKER,
    "smee": BotSource.SMEE,
    "rufio": BotSource.RUFIO,
}


def _detect_bot(author: str) -> BotSource:
    lower = author.lower()
    for name, source in BOT_NAME_MAP.items():
        if name in lower:
            return source
    return BotSource.UNKNOWN


# ---------------------------------------------------------------------------
# Trade signal parser (Captain Hook, Neverland Pan, Tinker)
# ---------------------------------------------------------------------------

# Header: 🐻 NVDA - Bearish (PUT) 💎  or  🐂 TSLA - Bullish (CALL)
HEADER_RE = re.compile(
    r"([🐻🐂]|💫)\s*([A-Z]{1,5})\s*-\s*(Bearish|Bullish|Elite Reversal)\s*\((PUT|CALL)\)\s*(💎)?",
    re.IGNORECASE,
)

# Score: 100/100 (Strong) 🟢  or  68/100 (Solid) 🟡  or  57/100 (Marginal) 🟠
# May include raw score: 100/100 (Strong) 🟢 (raw 164)
# Accept any word in parens so new tiers (Elite, Moderate, etc.) don't break parsing
SCORE_RE = re.compile(r"(\d{1,3})/100\s*\((\w+)\)")
RAW_SCORE_RE = re.compile(r"\(raw\s+(\d+)\)")

# Entry/target: $168.685 ➡ $167.09 (+0.9%)  or  **$168.685** ➡ **$167.09** (+0.9%)
ENTRY_TARGET_RE = re.compile(
    r"\*{0,2}\$(\d+(?:\.\d+)?)\*{0,2}\s*➡\s*\*{0,2}\$(\d+(?:\.\d+)?)\*{0,2}\s*\(\+?(-?\d+(?:\.\d+)?)%\)"
)

# Trade idea: Buy Puts | Strike: $170 Put | Expiry: 0DTE | R:R 1.50:1
TRADE_IDEA_RE = re.compile(
    r"Buy\s+(Puts?|Calls?)\s*\|\s*Strike:\s*\$(\d+(?:\.\d+)?)\s*(?:Put|Call)\s*\|\s*Expiry:\s*(\S+)\s*\|\s*R:R\s*(\d+(?:\.\d+)?):1",
    re.IGNORECASE,
)

# Exit targets (T1-T2 legacy): T1: $167.89 (+0.5%) | T2: $167.09 (+0.9%) | Stop: $169.43 (-0.5%)
EXIT_RE = re.compile(
    r"T1:\s*\$(\d+(?:\.\d+)?)\s*\(\+?(-?\d+(?:\.\d+)?)%\)\s*\|\s*"
    r"T2:\s*\$(\d+(?:\.\d+)?)\s*\(\+?(-?\d+(?:\.\d+)?)%\)\s*\|\s*"
    r"Stop:\s*\$(\d+(?:\.\d+)?)\s*\(\+?(-?\d+(?:\.\d+)?)%\)"
)

# Exit targets (T1-T5 new format): T1: $576.90 | T2: $577.50 | T3: $578.00 | T4: $578.31 | T5: $579.00 | Stop: $574.18
# Each Tn can optionally have a percentage like (+0.2%)
TARGET_RE = re.compile(r"T(\d):\s*\$(\d+(?:\.\d+)?)(?:\s*\(\+?(-?\d+(?:\.\d+)?)%\))?")
STOP_RE = re.compile(r"Stop:\s*\$(\d+(?:\.\d+)?)(?:\s*\(\+?(-?\d+(?:\.\d+)?)%\))?")

# Exit by: Exit by 10:40
EXIT_BY_RE = re.compile(r"Exit by\s+(\d{1,2}:\d{2})")

# ATM/OTM picks: $170 put @ ~$1.70 (~+-3893% est.)
OPTION_PICK_RE = re.compile(
    r"\$(\d+(?:\.\d+)?)\s+(?:put|call)\s+@\s+~\$(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Key signals line (after 🔑 Key Signals)
KEY_SIGNALS_RE = re.compile(r"🔑\s*Key Signals\n(.+)", re.MULTILINE)


def parse_trade_signal(
    text: str,
    *,
    message_id: int = 0,
    channel: str = "",
    author: str = "",
    timestamp: datetime | None = None,
) -> TradeSignal | None:
    """Parse a structured trade signal from Captain Hook / Neverland Pan / Tinker."""
    header = HEADER_RE.search(text)
    if not header:
        return None

    score_match = SCORE_RE.search(text)
    entry_match = ENTRY_TARGET_RE.search(text)
    trade_match = TRADE_IDEA_RE.search(text)

    if not score_match or not entry_match:
        return None

    # Require either a trade idea line OR targets/stop — reject truncated messages
    has_targets = bool(TARGET_RE.search(text))
    has_stop = bool(STOP_RE.search(text))
    if not trade_match and not has_targets and not has_stop:
        return None

    ticker = header.group(2)
    sentiment_raw = header.group(3)
    direction_raw = header.group(4)
    is_elite = header.group(5) is not None

    direction = Direction.PUT if direction_raw.upper() == "PUT" else Direction.CALL

    # "Elite Reversal" doesn't say bearish/bullish — derive from direction
    if "bear" in sentiment_raw.lower():
        sentiment = Sentiment.BEARISH
    elif "bull" in sentiment_raw.lower():
        sentiment = Sentiment.BULLISH
    else:
        sentiment = Sentiment.BEARISH if direction == Direction.PUT else Sentiment.BULLISH

    # Prefer raw score (uncapped) over display score (capped at 100)
    raw_match = RAW_SCORE_RE.search(text)
    score = int(raw_match.group(1)) if raw_match else int(score_match.group(1))
    strength_raw = score_match.group(2).lower()
    strength = {
        "elite": SignalStrength.ELITE,
        "strong": SignalStrength.STRONG,
        "good": SignalStrength.GOOD,
        "solid": SignalStrength.SOLID,
        "moderate": SignalStrength.MODERATE,
        "marginal": SignalStrength.MARGINAL,
    }.get(strength_raw, SignalStrength.MARGINAL)

    entry_price = float(entry_match.group(1))
    target_price = float(entry_match.group(2))
    expected_move_pct = float(entry_match.group(3))

    if trade_match:
        strike = float(trade_match.group(2))
        expiry = trade_match.group(3)
        risk_reward = float(trade_match.group(4))
    else:
        # Trade idea line has "Strike: Check chain" or other non-numeric strike.
        # Derive strike from ATM pick or round entry price to nearest option strike.
        strike = None
        expiry = "0DTE"
        risk_reward = 1.5
        # Try to extract expiry and R:R from the trade idea line even without strike
        trade_fallback = re.search(
            r"Expiry:\s*(\S+)\s*\|\s*R:R\s*(\d+(?:\.\d+)?):1", text, re.IGNORECASE,
        )
        if trade_fallback:
            expiry = trade_fallback.group(1)
            risk_reward = float(trade_fallback.group(2))

    # Exit targets — try new T1-T5 format first, fall back to legacy T1/T2/Stop
    targets: dict[int, tuple[float, float | None]] = {}  # {1: (price, pct|None), ...}
    stop = stop_pct = None

    target_matches = TARGET_RE.findall(text)
    stop_match = STOP_RE.search(text)

    if target_matches:
        for tn, price_str, pct_str in target_matches:
            pct = float(pct_str) if pct_str else None
            targets[int(tn)] = (float(price_str), pct)

    if stop_match:
        stop = float(stop_match.group(1))
        stop_pct = float(stop_match.group(2)) if stop_match.group(2) else None

    # If new-format parsing didn't find targets, try legacy EXIT_RE
    if not targets:
        exit_match = EXIT_RE.search(text)
        if exit_match:
            targets[1] = (float(exit_match.group(1)), float(exit_match.group(2)))
            targets[2] = (float(exit_match.group(3)), float(exit_match.group(4)))
            stop = float(exit_match.group(5))
            stop_pct = float(exit_match.group(6))

    t1 = targets.get(1, (None, None))[0]
    t1_pct = targets.get(1, (None, None))[1]
    t2 = targets.get(2, (None, None))[0]
    t2_pct = targets.get(2, (None, None))[1]
    t3 = targets.get(3, (None, None))[0]
    t3_pct = targets.get(3, (None, None))[1]
    t4 = targets.get(4, (None, None))[0]
    t4_pct = targets.get(4, (None, None))[1]
    t5 = targets.get(5, (None, None))[0]
    t5_pct = targets.get(5, (None, None))[1]

    exit_by = None
    exit_by_match = EXIT_BY_RE.search(text)
    if exit_by_match:
        exit_by = exit_by_match.group(1)

    # ATM/OTM picks — detect labels to assign correctly.
    # Discord format: "⚡ PRIMARY: OTM Pick\n$265 call @ ~$0.02"
    #                 "💰 Conservative: ATM\n$260 call @ ~$0.70"
    # OTM pick appears first in the message, ATM second.
    picks = OPTION_PICK_RE.findall(text)
    atm_strike = atm_prem = otm_strike = otm_prem = None
    if len(picks) >= 2:
        # Check labels to assign correctly (OTM first, ATM second in message)
        otm_label_pos = text.find("OTM")
        atm_label_pos = text.find("ATM")
        if otm_label_pos != -1 and atm_label_pos != -1 and otm_label_pos < atm_label_pos:
            # OTM comes first → picks[0] is OTM, picks[1] is ATM
            otm_strike, otm_prem = float(picks[0][0]), float(picks[0][1])
            atm_strike, atm_prem = float(picks[1][0]), float(picks[1][1])
        else:
            # ATM comes first (or no labels) → picks[0] is ATM, picks[1] is OTM
            atm_strike, atm_prem = float(picks[0][0]), float(picks[0][1])
            otm_strike, otm_prem = float(picks[1][0]), float(picks[1][1])
    elif len(picks) == 1:
        # Single pick — check if it's labeled OTM or ATM
        pick_pos = text.find(f"${picks[0][0]}")
        before = text[max(0, pick_pos - 80):pick_pos] if pick_pos > 0 else ""
        if "OTM" in before:
            otm_strike, otm_prem = float(picks[0][0]), float(picks[0][1])
        else:
            atm_strike, atm_prem = float(picks[0][0]), float(picks[0][1])

    # Fallback: derive strike from ATM pick or entry price when "Check chain"
    if strike is None:
        if atm_strike:
            strike = atm_strike
        else:
            # Round entry price to nearest 0.50 for most options chains
            # Use math.floor(x + 0.5) to avoid Python's banker's rounding
            import math
            strike = math.floor(entry_price * 2 + 0.5) / 2
        logger.debug(
            f"Strike derived from {'ATM pick' if atm_strike else 'entry price'}: "
            f"${strike} for {ticker}"
        )

    # Key signals
    key_signals: list[str] = []
    ks_match = KEY_SIGNALS_RE.search(text)
    if ks_match:
        key_signals = [s.strip() for s in ks_match.group(1).split("|") if s.strip()]

    bot_source = _detect_bot(author)

    return TradeSignal(
        ticker=ticker,
        sentiment=sentiment,
        direction=direction,
        score=score,
        strength=strength,
        entry_price=entry_price,
        target_price=target_price,
        expected_move_pct=expected_move_pct,
        strike=strike,
        expiry=expiry,
        risk_reward=risk_reward,
        target_1=t1,
        target_1_pct=t1_pct,
        target_2=t2,
        target_2_pct=t2_pct,
        target_3=t3,
        target_3_pct=t3_pct,
        target_4=t4,
        target_4_pct=t4_pct,
        target_5=t5,
        target_5_pct=t5_pct,
        stop_price=stop,
        stop_pct=stop_pct,
        exit_by=exit_by,
        atm_strike=atm_strike,
        atm_premium=atm_prem,
        otm_strike=otm_strike,
        otm_premium=otm_prem,
        key_signals=key_signals,
        bot_source=bot_source,
        is_elite=is_elite,
        source_message_id=message_id,
        source_channel=channel,
        author=author,
        timestamp=timestamp,
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Watchlist parser (Rufio)
# ---------------------------------------------------------------------------

# ENPH: Stage 1 (bearish, score 80)
WATCHLIST_ENTRY_RE = re.compile(
    r"([A-Z]{1,5}):\s*Stage\s+(\d+)\s*\((bearish|bullish),\s*score\s+(\d+)\)",
    re.IGNORECASE,
)


def parse_watchlist(text: str) -> list[WatchlistEntry]:
    """Parse Rufio's pre-market watchlist."""
    if "Active Watchlist" not in text and "Catalyst Sentinel" not in text:
        return []
    entries = []
    for m in WATCHLIST_ENTRY_RE.finditer(text):
        entries.append(
            WatchlistEntry(
                ticker=m.group(1),
                stage=int(m.group(2)),
                sentiment=Sentiment.BEARISH if m.group(3).lower() == "bearish" else Sentiment.BULLISH,
                score=int(m.group(4)),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Performance parser (Smee)
# ---------------------------------------------------------------------------

# 6W / 1L (86%) | Avg PnL: 0.89%
PERF_SUMMARY_RE = re.compile(
    r"(\d+)W\s*/\s*(\d+)L\s*\((\d+)%\)\s*\|\s*Avg PnL:\s*(-?\d+(?:\.\d+)?)%"
)

# ✅ QQQ bearish | Score: 87 | PnL: 0.99%
# ❌ AMD bearish | Score: 85 | PnL: -0.07%
PERF_TRADE_RE = re.compile(
    r"([✅❌])\s*([A-Z]{1,5})\s+(bearish|bullish)\s*\|\s*Score:\s*(\d+)\s*\|\s*PnL:\s*(-?\d+(?:\.\d+)?)%",
    re.IGNORECASE,
)

# All-time: 8/10 (80%) across 10 trades
ALL_TIME_RE = re.compile(r"(\d+)/(\d+)\s*\(\d+%\)\s*across\s+\d+\s+trades")


def parse_performance(text: str) -> DailyPerformance | None:
    """Parse Smee's daily performance summary."""
    if "DAILY PERFORMANCE SUMMARY" not in text:
        return None

    summary = PERF_SUMMARY_RE.search(text)
    if not summary:
        return None

    wins = int(summary.group(1))
    losses = int(summary.group(2))
    win_rate = float(summary.group(3))
    avg_pnl = float(summary.group(4))

    trades = []
    for m in PERF_TRADE_RE.finditer(text):
        trades.append(
            PerformanceEntry(
                ticker=m.group(2),
                sentiment=Sentiment.BEARISH if m.group(3).lower() == "bearish" else Sentiment.BULLISH,
                score=int(m.group(4)),
                pnl_pct=float(m.group(5)),
                won=m.group(1) == "✅",
            )
        )

    all_time = ALL_TIME_RE.search(text)
    at_wins = at_total = None
    if all_time:
        at_wins = int(all_time.group(1))
        at_total = int(all_time.group(2))

    return DailyPerformance(
        wins=wins,
        losses=losses,
        win_rate_pct=win_rate,
        avg_pnl_pct=avg_pnl,
        trades=trades,
        all_time_wins=at_wins,
        all_time_total=at_total,
    )


# ---------------------------------------------------------------------------
# Stand-down detector (Smee)
# ---------------------------------------------------------------------------


def is_stand_down(text: str) -> bool:
    """Check if Smee is in stand-down mode (no trades)."""
    return "STAND DOWN MODE" in text


# ---------------------------------------------------------------------------
# Unified message handler
# ---------------------------------------------------------------------------


def parse_message(
    text: str,
    *,
    message_id: int = 0,
    channel: str = "",
    author: str = "",
    timestamp: datetime | None = None,
) -> TradeSignal | list[WatchlistEntry] | DailyPerformance | None:
    """Try all parsers in priority order. Returns the first match or None."""
    # Trade signals (Captain Hook, Neverland Pan, Tinker)
    trade = parse_trade_signal(
        text, message_id=message_id, channel=channel, author=author, timestamp=timestamp
    )
    if trade:
        return trade

    # Performance summary (Smee)
    perf = parse_performance(text)
    if perf:
        return perf

    # Watchlist (Rufio)
    watchlist = parse_watchlist(text)
    if watchlist:
        return watchlist

    return None


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------


def extract_text_from_message(message: discord.Message) -> str:
    """Build a single text string from message content + embeds.

    The Neverland Pirates bots send signals as Discord embeds (not plain text),
    so we reconstruct the text from embed title + fields.
    """
    parts: list[str] = []
    if message.content:
        parts.append(message.content)

    for embed in message.embeds:
        if embed.title:
            parts.append(embed.title)
        if embed.description:
            parts.append(embed.description)
        for field in embed.fields:
            name = field.name.strip()
            value = field.value.strip()
            # Skip empty spacer fields (zero-width spaces)
            if name in ("​", "") and value in ("​", ""):
                continue
            # Score field: name="100/100 (Strong) 🟢" value="**$168.685** ➡ **$167.09** (+0.9%)"
            parts.append(f"{name}\n{value}")

    return "\n".join(parts)


class OptionsOwlBot(discord.Client):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)
        self.settings = settings
        self.paper_trader: object | None = None
        self._initialized = False  # guard against duplicate on_ready

    async def _init_webull_with_retry(self, max_retries: int = 3):
        """Initialize Webull executor with retries. Returns executor or None."""
        from options_owl.execution.webull_executor import WebullExecutor

        for attempt in range(1, max_retries + 1):
            executor = WebullExecutor(self.settings)
            try:
                account_id = await executor.init()
                info = await executor.get_account_info()
                logger.info(
                    f"LIVE TRADING enabled — Webull account {account_id}, "
                    f"buying power: ${info.buying_power:,.2f} (attempt {attempt})"
                )
                return executor
            except Exception as exc:
                logger.error(
                    f"Webull init attempt {attempt}/{max_retries} failed: {exc}"
                )
                if attempt < max_retries:
                    delay = 5 * attempt
                    logger.info(f"Retrying Webull init in {delay}s...")
                    await asyncio.sleep(delay)

        logger.error(
            "Webull init FAILED after all retries — falling back to paper trading only. "
            "TRADES WILL NOT REACH WEBULL until next restart."
        )
        return None

    async def on_ready(self) -> None:
        from options_owl.main import write_heartbeat

        write_heartbeat()

        # Guard: on_ready fires on EVERY Discord reconnect. Only do full init once.
        # Set flag IMMEDIATELY to prevent race condition when on_ready fires twice
        # rapidly (9s apart) during reconnect — second call must not re-init.
        if self._initialized:
            logger.info(
                f"Discord reconnected as {self.user} — reusing existing "
                f"PaperTrader/Webull/monitor (no re-init)"
            )
            return

        self._initialized = True
        logger.info(f"OptionsOwl connected as {self.user} (first on_ready)")

        # Initialize Webull executor for live trading (or None for paper-only)
        webull_executor = None
        if not self.settings.PAPER_TRADE:
            webull_executor = await self._init_webull_with_retry()

        # Always init PaperTrader �� it's our position tracker + risk pipeline.
        # When webull_executor is set, real orders are placed alongside paper records.
        from options_owl.execution.paper_trader import PaperTrader

        self.paper_trader = PaperTrader(
            self.settings,
            webull_executor=webull_executor,
        )
        await self.paper_trader.init()
        status = await self.paper_trader.get_status()
        mode = "LIVE (Webull)" if webull_executor else "PAPER ONLY (Webull failed)"
        logger.info(f"{mode} trading enabled (${self.settings.PORTFOLIO_SIZE:,.2f} portfolio)")
        logger.info(f"\n{status}")

        # Start market data stream
        from options_owl.collectors.market_data_stream import MarketDataStream

        self._market_stream = MarketDataStream(self.settings)
        await self._market_stream.start()
        logger.info(
            f"Market data stream started (provider={self._market_stream.provider.value})"
        )

        # Wire market stream to paper trader for dip-confirm entry
        self.paper_trader.market_stream = self._market_stream

        # Initialize Redis for cross-agent coordination
        if getattr(self.settings, "ENABLE_REDIS", False):
            try:
                from options_owl.db import redis_client
                await redis_client.init_redis(
                    getattr(self.settings, "REDIS_URL", "redis://redis:6379/0")
                )
                logger.info("Redis cross-agent coordination enabled")
            except Exception as exc:
                logger.warning(f"Redis init failed (continuing without coordination): {exc}")

        # Initialize shared PostgreSQL (Phase 1: dual-write)
        if getattr(self.settings, "ENABLE_POSTGRES", False):
            try:
                from options_owl.db import postgres as pg
                await pg.init_pool(
                    getattr(self.settings, "DATABASE_URL", None)
                )
                logger.info("PostgreSQL shared DB connected — dual-write active")
            except Exception as exc:
                logger.warning(f"PostgreSQL init failed (continuing with SQLite only): {exc}")

        # Launch background position monitor with the data stream
        from options_owl.execution.position_monitor import run_position_monitor

        self._position_monitor_task = asyncio.create_task(
            run_position_monitor(self.paper_trader, self._market_stream, discord_client=self)
        )
        logger.info("Position monitor background task launched")

        # Launch ML signal consumer (polls PG for sourcing signals)
        if getattr(self.settings, "ENABLE_POSTGRES", False):
            from options_owl.collectors.signal_consumer import run_signal_consumer
            self._signal_consumer_task = asyncio.create_task(
                run_signal_consumer(self.paper_trader, self.settings)
            )
            logger.info("ML signal consumer background task launched")

        # Start heartbeat loop for Docker healthcheck
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        for guild in self.guilds:
            logger.info(f"  Guild: {guild.name} (id={guild.id})")
            for channel in guild.text_channels:
                logger.info(f"    #{channel.name} (id={channel.id})")

    async def _heartbeat_loop(self) -> None:
        """Write heartbeat file every 30s so Docker healthcheck can verify liveness."""
        from options_owl.main import write_heartbeat

        while not self.is_closed():
            write_heartbeat()
            await asyncio.sleep(30)

    async def on_disconnect(self) -> None:
        logger.warning("Discord connection lost — discord.py will auto-reconnect")

    async def on_resumed(self) -> None:
        from options_owl.main import write_heartbeat

        write_heartbeat()
        logger.info("Discord connection resumed")

    async def on_error(self, event: str, *args, **kwargs) -> None:
        logger.exception(f"Unhandled error in event '{event}'")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        if not message.guild:
            return

        # Filter by guild
        if (
            self.settings.guild_ids
            and message.guild.id not in self.settings.guild_ids
        ):
            return

        # Filter by channel (empty = accept all in guild)
        if (
            self.settings.channel_ids
            and message.channel.id not in self.settings.channel_ids
        ):
            return

        # Extract full text (content + embeds)
        full_text = extract_text_from_message(message)

        # Save raw message (store full reconstructed text)
        msg_id = await db.save_message(
            self.settings.DB_PATH,
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            author_id=message.author.id,
            author_name=str(message.author),
            content=full_text or message.content,
            timestamp=message.created_at,
        )
        logger.debug(f"Saved message {msg_id} from {message.author} in #{message.channel}")

        if not full_text.strip():
            return

        # Parse message
        result = parse_message(
            full_text,
            message_id=msg_id,
            channel=str(message.channel),
            author=str(message.author),
            timestamp=message.created_at,
        )

        if isinstance(result, TradeSignal):
            sig_id = await db.save_trade_signal(
                self.settings.DB_PATH,
                message_id=msg_id,
                signal=result.model_dump(mode="json"),
            )
            await db.mark_parsed(self.settings.DB_PATH, msg_id)
            elite = " 💎" if result.is_elite else ""
            logger.info(
                f"TRADE{elite}: {result.bot_source.value} | {result.ticker} "
                f"{result.strike}{result.direction.value[0].upper()} "
                f"score={result.score} ({result.strength.value}) "
                f"entry=${result.entry_price} target=${result.target_price} "
                f"({result.expected_move_pct:+.1f}%)"
            )

            # Redis signal dedup: prevent 4 bots from entering the same signal
            if getattr(self.settings, "ENABLE_REDIS", False):
                try:
                    from options_owl.db import redis_client
                    agent_id = getattr(self.settings, "AGENT_ID", "unknown")
                    signal_key = f"{result.ticker}:{result.direction.value}:{result.strike}:{result.entry_price}"
                    claimed = await redis_client.try_claim_signal(signal_key, agent_id)
                    if not claimed:
                        logger.info(
                            f"SIGNAL DEDUP: {result.ticker} already claimed by another agent — skipping"
                        )
                        return
                except Exception as exc:
                    logger.debug(f"Redis dedup check failed (proceeding): {exc}")

            # Evaluate signal and trade (paper + Webull if live)
            # Skip if Discord signals disabled (sourcing-only mode)
            if not getattr(self.settings, "ENABLE_DISCORD_SIGNALS", True):
                logger.info(
                    f"DISCORD SIGNAL SKIPPED (sourcing-only mode): {result.ticker} "
                    f"{result.direction.value} score={result.score}"
                )
                return

            if self.paper_trader:
                try:
                    await self.paper_trader.evaluate_and_trade(result, sig_id)
                except Exception as exc:
                    logger.error(
                        f"TRADE EVAL FAILED for signal {sig_id} ({result.ticker} "
                        f"{result.strike}{result.direction.value[0].upper()}): "
                        f"{type(exc).__name__}: {exc}",
                        exc_info=True,
                    )

        elif isinstance(result, DailyPerformance):
            today = message.created_at.strftime("%Y-%m-%d")
            await db.save_smee_performance(
                self.settings.DB_PATH,
                message_id=msg_id,
                date=today,
                perf=result.model_dump(mode="json"),
            )
            await db.mark_parsed(self.settings.DB_PATH, msg_id)
            logger.info(
                f"PERF: {result.wins}W/{result.losses}L ({result.win_rate_pct}%) "
                f"avg={result.avg_pnl_pct:+.2f}%"
            )

        elif isinstance(result, list) and result:  # watchlist
            await db.mark_parsed(self.settings.DB_PATH, msg_id)
            tickers = ", ".join(f"{e.ticker}({e.sentiment.value[0].upper()}{e.score})" for e in result)
            logger.info(f"WATCHLIST: {len(result)} tickers — {tickers}")

        elif is_stand_down(full_text):
            logger.info("STAND DOWN: No high-edge setups")


async def run_collector(settings: Settings) -> None:
    await db.init_db(settings.DB_PATH)
    bot = OptionsOwlBot(settings)
    await bot.start(settings.DISCORD_TOKEN)
