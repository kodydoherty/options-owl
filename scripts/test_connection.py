"""Quick connection test: connect to Discord, list channels, read recent messages, parse signals."""

import asyncio
import sys

sys.path.insert(0, ".")

import discord
from loguru import logger

from options_owl.collectors.discord_collector import extract_text_from_message, parse_message
from options_owl.config.settings import Settings


async def test_connection():
    settings = Settings()
    if not settings.DISCORD_TOKEN:
        logger.error("No DISCORD_TOKEN set")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        logger.info(f"Connected as {client.user}")
        logger.info(f"Latency: {client.latency * 1000:.0f}ms")

        for guild in client.guilds:
            if guild.id not in settings.guild_ids:
                continue

            logger.info(f"\nGuild: {guild.name} (id={guild.id})")

            for ch in guild.text_channels:
                logger.info(f"\n--- #{ch.name} ---")
                try:
                    signals_found = 0
                    messages_read = 0

                    async for msg in ch.history(limit=15):
                        messages_read += 1
                        full_text = extract_text_from_message(msg)
                        if not full_text.strip():
                            continue

                        result = parse_message(
                            full_text,
                            message_id=msg.id,
                            channel=ch.name,
                            author=str(msg.author),
                            timestamp=msg.created_at,
                        )
                        if result is not None:
                            signals_found += 1
                            if hasattr(result, "ticker"):
                                logger.info(
                                    f"  SIGNAL: {result.ticker} {result.direction.value.upper()} "
                                    f"score={result.score} strike=${result.strike} "
                                    f"entry=${result.entry_price} → target=${result.target_price} "
                                    f"| {msg.author}"
                                )
                            elif isinstance(result, list):
                                tickers = [e.ticker for e in result]
                                logger.info(f"  WATCHLIST: {tickers} | {msg.author}")
                            else:
                                logger.info(f"  PERF: {result} | {msg.author}")
                        else:
                            logger.debug(f"  (no parse) {msg.author}: {full_text[:80]}")

                    logger.info(f"  {messages_read} msgs, {signals_found} signals parsed")

                except discord.Forbidden:
                    logger.warning(f"  No access to #{ch.name}")

        await client.close()

    await client.start(settings.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(test_connection())
