"""Dump raw Webull order history JSON structure for debugging."""
import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
for name in ("webull", "webull_trade_sdk", "urllib3", "httpx"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).handlers = []
os.environ.setdefault("WEBULL_LOG_LEVEL", "CRITICAL")

# Suppress loguru too
from loguru import logger
logger.remove()

from options_owl.execution.webull_executor import WebullExecutor
from options_owl.config.settings import Settings


async def main():
    s = Settings()
    w = WebullExecutor(s)
    await w.init()
    acct = w._account_id
    tc = w._trade_client

    resp = await asyncio.to_thread(
        tc.order_v2.get_order_history, acct,
        page_size=10, start_date="2026-05-18", end_date="2026-05-19"
    )
    data = resp.json() if hasattr(resp, "json") else resp

    # Dump first 3 entries as formatted JSON
    if isinstance(data, list):
        for i, entry in enumerate(data[:3]):
            print(json.dumps(entry, indent=2, default=str))
            print("---")
        print(f"Total entries: {len(data)}", file=sys.stderr)
    else:
        print(json.dumps(data, indent=2, default=str))


asyncio.run(main())
