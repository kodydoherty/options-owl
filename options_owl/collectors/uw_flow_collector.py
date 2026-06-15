"""Unusual Whales flow → trade-signal collector (Track 4, production).

Subscribes to the UW flow-alerts WebSocket, filters to whale ask-side option SWEEPS on the
validated whitelists (PUT: META/AMZN/AAPL/TSLA; CALL: TSLA/AAPL/AMD/AVGO/PLTR), and emits a
TradeSignal into the sourcing signals path → entry pipeline → V7 exits. Flow signals are a
high-conviction SOURCE (like Discord): they bypass the pattern/entry-timing ML gates and
the bearish-confirm/regime gates, but keep the risk gates (spread/delta/premium/EOD).

Gated behind ENABLE_UW_FLOW_SIGNAL (default off — paper-first). The filter/builder are pure
functions for unit testing; the WS loop is thin.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import websockets
from loguru import logger

from options_owl.models.signals import BotSource, Direction, Sentiment

WS_URI = "wss://api.unusualwhales.com/socket?token={token}"
FLOW_CHANNEL = "flow-alerts"


@dataclass
class FlowSignal:
    """A qualifying whale-sweep signal extracted from a UW flow alert."""
    ticker: str
    direction: Direction          # CALL or PUT
    strike: float
    expiry: str
    total_premium: float
    ask_frac: float
    volume_oi_ratio: float
    option_chain: str
    cluster_count: int = 1  # # qualifying same-ticker+dir sweeps in the rolling window (Stage D)


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def evaluate_flow_alert(alert: dict, settings) -> FlowSignal | None:
    """Pure filter+builder: return a FlowSignal if the alert qualifies, else None.

    Qualifies when: whale-size premium, ask-side dominant (bought = conviction), a sweep
    (if required), and the ticker is on the direction-appropriate whitelist (PUT names for
    puts, CALL names for calls). PUT sweep -> bearish/PUT; CALL sweep -> bullish/CALL.
    """
    # type/strike/expiry: REST flow-alerts has these as fields; the live WS does NOT — it encodes
    # them in the OCC `option_chain` symbol (e.g. MSTR260618C00115000). Derive from there when absent.
    opt_type = str(alert.get("type", "")).lower()
    occ_strike = _f(alert.get("strike"))
    occ_expiry = str(alert.get("expiry", ""))
    if opt_type not in ("put", "call"):
        from options_owl.collectors.flow_collector import parse_occ_ticker
        chain = str(alert.get("option_chain", ""))
        parsed = parse_occ_ticker(chain if chain.startswith("O:") else f"O:{chain}") if chain else None
        if not parsed:
            return None
        opt_type = parsed["type"]
        occ_strike = occ_strike or parsed["strike"]
        occ_expiry = occ_expiry or parsed["expiry"]
    if opt_type not in ("put", "call"):
        return None
    ticker = str(alert.get("ticker", "")).upper()
    if not ticker:
        return None

    put_wl = {t.strip().upper() for t in settings.UW_FLOW_PUT_TICKERS.split(",") if t.strip()}
    call_wl = {t.strip().upper() for t in settings.UW_FLOW_CALL_TICKERS.split(",") if t.strip()}
    whitelist = put_wl if opt_type == "put" else call_wl
    if ticker not in whitelist:
        return None

    total = _f(alert.get("total_premium"))
    if total < settings.UW_FLOW_MIN_PREMIUM:
        return None
    ask_frac = _f(alert.get("total_ask_side_prem")) / total if total > 0 else 0.0
    if ask_frac < settings.UW_FLOW_ASK_FRAC:
        return None
    if settings.UW_FLOW_REQUIRE_SWEEP and not bool(alert.get("has_sweep")):
        return None

    return FlowSignal(
        ticker=ticker,
        direction=Direction.PUT if opt_type == "put" else Direction.CALL,
        strike=occ_strike,
        expiry=occ_expiry,
        total_premium=total,
        ask_frac=round(ask_frac, 3),
        volume_oi_ratio=_f(alert.get("volume_oi_ratio")),
        option_chain=str(alert.get("option_chain", "")),
    )


def flow_signal_to_trade_signal(fs: FlowSignal, underlying: float = 0.0):
    """Build a TradeSignal (BotSource.UW_FLOW) for the entry pipeline. Score is set high —
    flow conviction is the signal; it bypasses the ML pattern gate by source, not score."""
    from options_owl.models.signals import SignalStrength, TradeSignal
    return TradeSignal(
        ticker=fs.ticker,
        direction=fs.direction,
        sentiment=Sentiment.BEARISH if fs.direction == Direction.PUT else Sentiment.BULLISH,
        score=90,
        strength=SignalStrength.STRONG,
        bot_source=BotSource.UW_FLOW,
        entry_price=underlying,
        target_price=0, expected_move_pct=0, risk_reward=0,
        strike=fs.strike or 0, expiry=fs.expiry or "0DTE",
        target_1=None, target_2=None, stop_price=None, exit_by=None,
        atm_strike=fs.strike or None, atm_premium=None,
        flow_cluster_count=fs.cluster_count,
        flow_total_premium=fs.total_premium,
        flow_ask_frac=fs.ask_frac,
    )


async def run_uw_flow_collector(settings, on_signal) -> None:
    """Connect to the UW flow WS and call on_signal(FlowSignal) for each qualifying alert.

    Reconnects with backoff. on_signal is the integration hook (emit → pipeline); in shadow
    mode it just logs. Guarded by ENABLE_UW_FLOW_SIGNAL upstream.
    """
    uri = WS_URI.format(token=settings.UNUSUAL_WHALES_API_KEY)
    attempt = 0
    # Real-time cluster detection (Stage D): rolling window of recent qualifying-sweep times
    # per (ticker, direction). cluster_count = sweeps in the last UW_FLOW_CLUSTER_WINDOW_MIN.
    win_s = getattr(settings, "UW_FLOW_CLUSTER_WINDOW_MIN", 30) * 60
    recent: dict[tuple, deque] = defaultdict(deque)

    def _cluster_count(fs) -> int:
        key = (fs.ticker, fs.direction)
        now = time.monotonic()
        dq = recent[key]
        dq.append(now)
        while dq and now - dq[0] > win_s:
            dq.popleft()
        return len(dq)

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                await ws.send(json.dumps({"channel": FLOW_CHANNEL, "msg_type": "join"}))
                logger.info(f"UW_FLOW: connected + joined {FLOW_CHANNEL}")
                attempt = 0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    payloads = msg if isinstance(msg, list) else [msg.get("payload", msg)]
                    for p in payloads:
                        if not isinstance(p, dict):
                            continue
                        fs = evaluate_flow_alert(p, settings)
                        if fs is not None:
                            fs.cluster_count = _cluster_count(fs)
                            try:
                                await on_signal(fs)
                            except Exception as exc:  # never let one signal kill the loop
                                logger.warning(f"UW_FLOW: on_signal failed for {fs.ticker}: {exc}")
        except Exception as exc:
            attempt += 1
            delay = min(5 * attempt, 60)
            logger.warning(f"UW_FLOW: WS dropped ({type(exc).__name__}: {exc}); reconnect in {delay}s")
            await asyncio.sleep(delay)
