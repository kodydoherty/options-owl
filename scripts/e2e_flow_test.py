"""E2E UW-flow replication test — run INSIDE a bot container against its REAL config + DB.

Replicates the full whale-flow entry path with a SYNTHETIC alert (no live market needed) and
asserts every stage behaves as we expect, then CLEANS UP any test rows it created:

  1. evaluate_flow_alert     — real filter (qualify / reject cases)
  2. flow_signal_to_trade_signal — real builder (source=uw_flow, direction, score=90)
  3. run_entry_pipeline      — REAL 28-gate pipeline: the 4 directional gates must SKIP
                               ("UW flow source — bypassed"); risk gates must still EVALUATE.
                               A non-flow (ML) signal is run as contrast (gates must NOT bypass).
  4. ExitFSM (V7)            — real exit engine fires on a synthetic losing state (backstop)
  5. DB round-trip + CLEANUP — insert a sentinel paper_trade, read it back, then DELETE it
                               (in finally — runs even on failure). Sentinel = signal_id -987654.

Usage (per bot, on the droplet):
  docker compose exec owlet-adam  python scripts/e2e_flow_test.py
  docker compose exec owlet-vinny python scripts/e2e_flow_test.py
  docker compose exec owlet-yank  python scripts/e2e_flow_test.py
Exit code 0 = all assertions passed.
"""

import asyncio
import os
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from options_owl.collectors.uw_flow_collector import (
    evaluate_flow_alert,
    flow_signal_to_trade_signal,
)
from options_owl.config.settings import Settings
from options_owl.models.signals import BotSource, Direction
from options_owl.risk.exit_v5.config import apply_v7_wide_trail_exits, get_ticker_config
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState
from options_owl.risk.pipeline import GateResult, run_entry_pipeline

ET = ZoneInfo("America/New_York")
SENTINEL = -987654  # unique signal_id marker for the test trade (cleaned up in finally)
BYPASS = {"put_ticker_exclusion", "put_market_direction", "put_bearish_confirm", "directional_regime"}
BYPASS_MSG = "bypassed"
DB_PATH = os.getenv("E2E_DB_PATH", "/app/journal/raw_messages.db")

_fail = []
def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _fail.append(msg)


def _alert(ticker, typ, ask_frac=0.75, prem=300_000.0, sweep=True):
    return {"ticker": ticker, "type": typ, "total_premium": prem,
            "total_ask_side_prem": prem * ask_frac, "has_sweep": sweep,
            "strike": 100.0, "expiry": "", "volume_oi_ratio": 3.0, "option_chain": ""}


def _ctx(settings, signal):
    """Best-effort pipeline ctx with data that lets the RISK gates run (not bypass)."""
    signal.atm_premium = 2.00
    signal.entry_price = 100.0
    signal.strike = 100.0
    return {
        "signal": signal, "settings": settings,
        "bid": 1.95, "ask": 2.05,           # 5% spread — passes spread gate
        "current_price": 100.0, "entry_delta": 0.45,
        "candle_data": {}, "now_et": datetime.now(ET),
        "portfolio": {"balance": 20000.0, "starting_balance": 20000.0},
        "open_count": 0, "open_positions": [], "open_tickers": set(),
        "open_calls": 0, "open_puts": 0,
        "premium_history": [], "underlying_price_history": [],
        "db_path": DB_PATH, "webull_executor": None,
    }


async def stage_filter(settings):
    print("\n── Stage 1: evaluate_flow_alert (real filter) ──")
    put_tk = settings.UW_FLOW_PUT_TICKERS.split(",")[0].strip()
    call_tk = settings.UW_FLOW_CALL_TICKERS.split(",")[0].strip()
    check(evaluate_flow_alert(_alert(put_tk, "put"), settings) is not None,
          f"qualifying PUT sweep on {put_tk} -> FlowSignal")
    check(evaluate_flow_alert(_alert(call_tk, "call"), settings) is not None,
          f"qualifying CALL sweep on {call_tk} -> FlowSignal")
    check(evaluate_flow_alert(_alert(put_tk, "put", ask_frac=0.30), settings) is None,
          "bid-side dominant -> rejected")
    check(evaluate_flow_alert(_alert(put_tk, "put", prem=50_000.0), settings) is None,
          "sub-threshold premium -> rejected")
    check(evaluate_flow_alert(_alert("ZZZZ", "put"), settings) is None,
          "non-whitelist ticker -> rejected")
    check(evaluate_flow_alert(_alert(put_tk, "put", sweep=False), settings) is None,
          "no sweep -> rejected")
    return put_tk, call_tk


def stage_builder(settings, put_tk):
    print("\n── Stage 2: flow_signal_to_trade_signal (real builder) ──")
    fs = evaluate_flow_alert(_alert(put_tk, "put"), settings)
    ts = flow_signal_to_trade_signal(fs, underlying=100.0)
    check(ts.bot_source == BotSource.UW_FLOW, f"bot_source = {ts.bot_source}")
    check(ts.direction == Direction.PUT, f"direction = {ts.direction}")
    check(ts.score == 90, f"score = {ts.score}")
    return ts


async def stage_gates(settings, put_tk):
    print("\n── Stage 3: run_entry_pipeline — bypass gates SKIP, risk gates EVALUATE ──")
    fs = evaluate_flow_alert(_alert(put_tk, "put"), settings)
    flow_sig = flow_signal_to_trade_signal(fs, underlying=100.0)
    res = await run_entry_pipeline(_ctx(settings, flow_sig))
    by_name = {o.gate_name: o for o in res.outcomes}
    print(f"  {'gate':<26}{'result':<7}reason")
    for o in res.outcomes:
        print(f"  {o.gate_name:<26}{o.result.value:<7}{o.reason[:60]}")
    print("  ── assertions ──")
    for g in BYPASS:
        o = by_name.get(g)
        check(o is not None and o.result == GateResult.SKIP and BYPASS_MSG in o.reason.lower(),
              f"{g} SKIPs as flow-bypassed")
    # risk gates must NOT be bypassed (they evaluate on their own logic)
    for g in ("v6_spread_gate", "score", "v6_premium_cap"):
        o = by_name.get(g)
        if o is not None:
            check(BYPASS_MSG not in o.reason.lower(), f"risk gate {g} is NOT flow-bypassed ({o.result.value})")

    # contrast: a non-flow (ML) signal must NOT bypass the 4 directional gates
    print("  ── contrast: non-flow ML signal (gates must NOT bypass) ──")
    ml_sig = flow_signal_to_trade_signal(fs, underlying=100.0)
    ml_sig.bot_source = BotSource.ML_SOURCING
    res2 = await run_entry_pipeline(_ctx(settings, ml_sig))
    by2 = {o.gate_name: o for o in res2.outcomes}
    not_bypassed = sum(1 for g in BYPASS if g in by2 and BYPASS_MSG not in by2[g].reason.lower())
    check(not_bypassed >= 1, f"non-flow signal: {not_bypassed}/4 directional gates evaluated (not bypassed)")


def stage_exit(settings):
    print("\n── Stage 4: V7 ExitFSM fires on a synthetic loss (backstop) ──")
    cfg = apply_v7_wide_trail_exits(get_ticker_config("META", use_per_ticker=True, option_type="put"), is_put=True)
    fsm = ExitFSM(cfg, settings=settings)
    st = TradeState(trade_id=1, ticker="META", option_type="put", entry_premium=2.0,
                    entry_time=datetime.now(ET), contracts=3, peak_premium=2.0,
                    entry_underlying_price=100.0, dte=0, expiry_date=datetime.now(ET).strftime("%Y-%m-%d"))
    # premium collapsed to 0.50 (-75%) -> backstop must fire
    act = fsm.evaluate(st, 0.50, 0.48, 0.50, datetime.now(ET),
                       current_underlying=101.0, minutes_to_close=120, candle_data={})
    check(act is not None and getattr(act, "should_exit", False),
          f"catastrophic loss triggers exit: {getattr(act, 'reason', '?')}")


def stage_db_cleanup():
    print("\n── Stage 5: DB round-trip + CLEANUP (sentinel signal_id) ──")
    con = sqlite3.connect(DB_PATH)
    try:
        now = datetime.now(ET).isoformat()
        con.execute(
            "INSERT INTO paper_trades (signal_id,ticker,direction,sentiment,score,strength,"
            "bot_source,entry_price,strike,option_type,contracts,premium_per_contract,total_cost,"
            "status,opened_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (SENTINEL, "E2ETEST", "put", "bearish", 90, "strong", "uw_flow",
             100.0, 100.0, "put", 1, 2.0, 200.0, "open", now))
        con.commit()
        n = con.execute("SELECT COUNT(*) FROM paper_trades WHERE signal_id=?", (SENTINEL,)).fetchone()[0]
        check(n == 1, "sentinel trade written + read back")
    finally:
        con.execute("DELETE FROM paper_trades WHERE signal_id=?", (SENTINEL,))
        con.execute("DELETE FROM trade_events WHERE trade_id IN "
                    "(SELECT id FROM paper_trades WHERE signal_id=?)", (SENTINEL,))
        con.commit()
        left = con.execute("SELECT COUNT(*) FROM paper_trades WHERE signal_id=?", (SENTINEL,)).fetchone()[0]
        con.close()
        check(left == 0, "sentinel trade CLEANED UP (0 rows remain)")


async def main():
    agent = os.getenv("AGENT_ID", "?")
    s = Settings()
    print(f"=== E2E FLOW TEST — agent={agent} db={DB_PATH} ===")
    print(f"ENABLE_UW_FLOW_SIGNAL={s.ENABLE_UW_FLOW_SIGNAL} PAPER_TRADE={s.PAPER_TRADE} "
          f"PUT_WL={s.UW_FLOW_PUT_TICKERS} CALL_WL={s.UW_FLOW_CALL_TICKERS}")
    put_tk, _ = await stage_filter(s)
    stage_builder(s, put_tk)
    await stage_gates(s, put_tk)
    stage_exit(s)
    stage_db_cleanup()
    print(f"\n=== {'ALL PASSED' if not _fail else f'{len(_fail)} FAILURES: ' + '; '.join(_fail)} ===")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
