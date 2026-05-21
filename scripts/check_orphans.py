#!/usr/bin/env python3
"""Check all bots for orphaned Webull positions not tracked in DB."""
import asyncio
import os
from options_owl.config.settings import Settings
from options_owl.execution.webull_executor import WebullExecutor
import aiosqlite


async def check_bot(name: str):
    print(f"\n=== {name} ===")
    s = Settings()
    w = WebullExecutor(s)
    try:
        await w.init()
    except Exception as e:
        print(f"  Webull init failed: {e}")
        return

    await asyncio.sleep(2)

    # Get Webull positions
    info = await w.get_account_info()
    webull_positions = []
    for p in info.positions:
        for leg in p.get("legs", []):
            webull_positions.append({
                "symbol": leg.get("symbol"),
                "option_type": leg.get("option_type"),
                "strike": float(leg.get("option_exercise_price", 0)),
                "expiry": leg.get("option_expire_date"),
                "qty": int(p.get("quantity", 0)),
                "last": float(leg.get("last_price", 0)),
                "pnl": float(p.get("unrealized_profit_loss", 0)),
            })

    # Get DB open trades
    db_path = f"/app/journal/owlet-{name}/raw_messages.db"
    if not os.path.exists(db_path):
        db_path = f"journal/owlet-{name}/raw_messages.db"
    db_trades = []
    try:
        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM paper_trades WHERE status='open'")
            db_trades = [dict(r) for r in await cursor.fetchall()]
    except Exception as e:
        print(f"  DB error: {e}")

    print(f"  Webull positions: {len(webull_positions)}")
    print(f"  DB open trades: {len(db_trades)}")
    print(f"  Cash: ${info.cash_balance:.2f}")

    # Find orphans
    for wp in webull_positions:
        matched = False
        for dt in db_trades:
            if (dt["ticker"] == wp["symbol"]
                and dt["option_type"].upper() == wp["option_type"]
                and abs(dt["strike"] - wp["strike"]) < 0.01):
                matched = True
                if dt["contracts"] != wp["qty"]:
                    print(f"  MISMATCH: {wp['symbol']} {wp['option_type']} ${wp['strike']} "
                          f"— Webull has {wp['qty']}, DB has {dt['contracts']}")
                break
        if not matched:
            print(f"  ORPHAN: {wp['symbol']} {wp['option_type']} ${wp['strike']} "
                  f"x{wp['qty']} exp={wp['expiry']} last=${wp['last']:.2f} pnl=${wp['pnl']:.2f}")

    for dt in db_trades:
        matched = any(
            wp["symbol"] == dt["ticker"]
            and wp["option_type"] == dt["option_type"].upper()
            and abs(wp["strike"] - dt["strike"]) < 0.01
            for wp in webull_positions
        )
        if not matched:
            print(f"  GHOST: DB trade #{dt['id']} {dt['ticker']} {dt['option_type']} "
                  f"${dt['strike']} x{dt['contracts']} — NOT on Webull")

    if not webull_positions and not db_trades:
        print("  Clean — no positions anywhere")
    elif len(webull_positions) == len(db_trades) and not any(
        not any(
            wp["symbol"] == dt["ticker"]
            and wp["option_type"] == dt["option_type"].upper()
            for wp in webull_positions
        ) for dt in db_trades
    ):
        print("  All positions matched")


asyncio.run(check_bot(os.environ.get("BOT_NAME", "kody")))
