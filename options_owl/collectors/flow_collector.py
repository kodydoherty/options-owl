"""Options flow collector — captures institutional option trades from Polygon WS.

Connects to ``wss://socket.polygon.io/options`` (separate endpoint from the
``/stocks`` WS used by CandleCollector) and subscribes to ``T.*`` and ``Q.*``
(all options trades + quotes), filtering client-side to the harvester universe.

Filters trades by size (>= MIN_FLOW_SIZE) and DTE (<= MAX_DTE) to capture
only meaningful institutional flow.  Aggregates into 5-minute bars for
efficient ML feature reads.

Usage in harvester::

    flow = FlowCollector(["SPY", "QQQ", "TSLA", ...])
    await flow.start_ws(api_key)

    # Periodically flush accumulated flow to PG
    await flow.flush()

    # On shutdown
    await flow.stop_ws()
"""

from __future__ import annotations

import asyncio
import json as _json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FLOW_SIZE = 10       # ignore trades smaller than this (retail noise)
LARGE_TRADE_SIZE = 50    # flag as "large" for ML features
MAX_DTE = 7              # only capture near-term flow (matches harvester)
SWEEP_CONDITION = 12     # Polygon condition code for intermarket sweep
WS_STALE_TIMEOUT = 120   # seconds — force reconnect if no WS messages during market hours

# OCC ticker regex: O:{TICKER}{YYMMDD}{C|P}{STRIKE*1000}
_OCC_RE = re.compile(
    r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FlowTrade:
    """Single option trade from Polygon WS."""

    ticker: str              # underlying (SPY, TSLA, etc.)
    contract: str            # full OCC ticker
    option_type: str         # 'call' or 'put'
    strike: float
    expiry_date: str         # YYYY-MM-DD
    trade_price: float
    trade_size: int
    trade_value: float       # price * size * 100
    conditions: list[int]
    is_sweep: bool
    is_above_ask: bool | None
    is_below_bid: bool | None
    exchange_id: int | None
    sip_timestamp: datetime


@dataclass
class FlowBar5m:
    """5-minute aggregated flow for one ticker."""

    ticker: str
    bar_time: datetime
    call_volume: int = 0
    call_value: float = 0.0
    call_sweeps: int = 0
    call_large_trades: int = 0
    call_buyer_count: int = 0
    call_total_count: int = 0
    put_volume: int = 0
    put_value: float = 0.0
    put_sweeps: int = 0
    put_large_trades: int = 0
    put_buyer_count: int = 0
    put_total_count: int = 0

    def to_dict(self) -> dict:
        total_trades = self.call_total_count + self.put_total_count
        total_sweeps = self.call_sweeps + self.put_sweeps
        return {
            "ticker": self.ticker,
            "bar_time": self.bar_time.isoformat() if isinstance(self.bar_time, datetime) else str(self.bar_time),
            "call_volume": self.call_volume,
            "call_value": self.call_value,
            "call_sweeps": self.call_sweeps,
            "call_large_trades": self.call_large_trades,
            "call_buyer_aggressor_pct": (
                self.call_buyer_count / self.call_total_count * 100
                if self.call_total_count > 0 else None
            ),
            "put_volume": self.put_volume,
            "put_value": self.put_value,
            "put_sweeps": self.put_sweeps,
            "put_large_trades": self.put_large_trades,
            "put_buyer_aggressor_pct": (
                self.put_buyer_count / self.put_total_count * 100
                if self.put_total_count > 0 else None
            ),
            "call_put_ratio": (
                self.call_volume / self.put_volume
                if self.put_volume > 0 else None
            ),
            "net_flow_dollars": self.call_value - self.put_value,
            "sweep_ratio": (
                total_sweeps / total_trades
                if total_trades > 0 else None
            ),
        }


# ---------------------------------------------------------------------------
# OCC ticker parsing
# ---------------------------------------------------------------------------


def parse_occ_ticker(contract: str) -> dict | None:
    """Parse a Polygon OCC option ticker into components.

    Example: O:SPY260526C00530000 ->
        {"ticker": "SPY", "expiry": "2026-05-26", "type": "call", "strike": 530.0}
    """
    m = _OCC_RE.match(contract)
    if not m:
        return None
    ticker, date_str, cp, strike_str = m.groups()
    try:
        expiry = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None
    return {
        "ticker": ticker,
        "expiry": expiry,
        "type": "call" if cp == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def detect_aggressor(
    trade_price: float, bid: float, ask: float,
) -> tuple[bool | None, bool | None]:
    """Detect trade aggressor from price vs NBBO.

    Returns (is_above_ask, is_below_bid).
    """
    is_above = None
    is_below = None
    if ask > 0:
        is_above = trade_price >= ask
    if bid > 0:
        is_below = trade_price <= bid
    return is_above, is_below


def _bar_start(ts: datetime) -> datetime:
    """Round a timestamp down to the nearest 5-minute boundary."""
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Local greeks computation (pure math, no I/O)
# ---------------------------------------------------------------------------

_RISK_FREE_RATE = 0.045


def _compute_greeks(
    mid: float, underlying: float, strike: float,
    expiry_str: str, option_type: str,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Compute IV + greeks from Black-Scholes. Returns (iv, delta, gamma, theta, vega)."""
    from options_owl.risk.greeks import (
        calc_delta,
        calc_gamma,
        calc_iv_from_premium,
        calc_theta,
        calc_vega,
    )

    expiry_dt = date.fromisoformat(expiry_str)
    today = date.today()
    days_to_expiry = (expiry_dt - today).days

    if days_to_expiry <= 0:
        from datetime import datetime as _dt
        now = _dt.now()
        hours_left = max(16 - now.hour - (now.minute / 60.0), 0.1)
        T = hours_left / (24.0 * 365.0)
    else:
        T = days_to_expiry / 365.0

    if T <= 0:
        return None, None, None, None, None

    iv = calc_iv_from_premium(mid, underlying, strike, T, _RISK_FREE_RATE, option_type)
    if not iv or iv <= 0.001 or iv >= 5.0:
        return None, None, None, None, None

    delta = round(calc_delta(underlying, strike, T, _RISK_FREE_RATE, iv, option_type), 4)
    gamma = round(calc_gamma(underlying, strike, T, _RISK_FREE_RATE, iv), 6)
    theta = round(calc_theta(underlying, strike, T, _RISK_FREE_RATE, iv, option_type), 4)
    vega = round(calc_vega(underlying, strike, T, _RISK_FREE_RATE, iv), 4)
    iv = round(iv, 4)

    return iv, delta, gamma, theta, vega


# ---------------------------------------------------------------------------
# FlowCollector
# ---------------------------------------------------------------------------


class FlowCollector:
    """Captures option trade flow from Polygon WS and stores to PostgreSQL.

    Also publishes full option snapshots (bid/ask/greeks) to Redis for the
    ML scan loop in each trading bot. Greeks are computed locally via BS model.
    """

    def __init__(
        self,
        tickers: list[str],
        price_getter: callable | None = None,
    ) -> None:
        self._tickers = frozenset(t.upper() for t in tickers if t.upper() != "VIX")
        self._trade_buffer: list[FlowTrade] = []
        self._flow_bars: dict[tuple[str, datetime], FlowBar5m] = {}

        # Underlying price getter — returns float or None.
        # Set by harvester to read from CandleCollector or yfinance cache.
        self._price_getter = price_getter

        # NBBO cache: contract -> (bid, ask, timestamp)
        self._nbbo_cache: dict[str, tuple[float, float, float]] = {}

        # Per-contract volume tracker (from trade events, reset daily)
        self._contract_volume: dict[str, int] = {}
        self._volume_date: str = ""

        # Redis quote buffer: contract_key -> full snapshot dict
        # Flushed to Redis every 1s to avoid per-event overhead
        self._redis_quote_buffer: dict[str, tuple[float, float, float]] = {}
        self._redis_snapshot_buffer: dict[str, dict] = {}
        self._redis_flush_task: asyncio.Task[None] | None = None

        # WS state
        self._ws_running = False
        self._ws_task: asyncio.Task[None] | None = None
        self._ws_connected = False
        self._last_msg_time: float = 0.0  # monotonic time of last WS message

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    @property
    def last_msg_age(self) -> float:
        """Seconds since last WS message (0 if never received)."""
        if self._last_msg_time == 0.0:
            return 0.0
        return time.monotonic() - self._last_msg_time

    async def start_ws(self, api_key: str) -> None:
        """Start the Options WS for real-time trade flow."""
        if self._ws_running:
            return
        self._ws_running = True
        self._ws_task = asyncio.create_task(self._ws_loop(api_key))
        self._redis_flush_task = asyncio.create_task(self._redis_quote_flush_loop())
        logger.info("FlowCollector: WS started")

    async def stop_ws(self) -> None:
        """Stop the WS connection."""
        self._ws_running = False
        if self._redis_flush_task and not self._redis_flush_task.done():
            self._redis_flush_task.cancel()
            try:
                await self._redis_flush_task
            except asyncio.CancelledError:
                pass
        self._redis_flush_task = None
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self._ws_task = None
        self._ws_connected = False
        # Final flush
        await self.flush()
        logger.info("FlowCollector: WS stopped")

    async def _redis_quote_flush_loop(self) -> None:
        """Flush buffered option quotes + full snapshots to Redis every 1s."""
        while self._ws_running:
            try:
                await asyncio.sleep(1)

                # Flush premium-only quotes (backward compat for position monitor)
                if self._redis_quote_buffer:
                    buf = self._redis_quote_buffer.copy()
                    self._redis_quote_buffer.clear()
                    try:
                        from options_owl.db import redis_client
                        if redis_client.is_connected():
                            for contract_key, (bid, ask, mid) in buf.items():
                                await redis_client.publish_option_premium(
                                    contract_key, bid, ask, mid,
                                )
                    except Exception:
                        pass

                # Flush full snapshots (greeks + premium for ML scan loop)
                if self._redis_snapshot_buffer:
                    snap_buf = self._redis_snapshot_buffer.copy()
                    self._redis_snapshot_buffer.clear()
                    try:
                        from options_owl.db import redis_client
                        if redis_client.is_connected():
                            for contract_key, snap in snap_buf.items():
                                await redis_client.publish_option_snapshot(
                                    contract_key, snap,
                                )
                    except Exception:
                        pass

            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def flush(self) -> int:
        """Write buffered trades and aggregated bars to PG. Returns trade count."""
        if not self._trade_buffer and not self._flow_bars:
            return 0

        trades = self._trade_buffer.copy()
        bars = [b.to_dict() for b in self._flow_bars.values()]
        self._trade_buffer.clear()
        self._flow_bars.clear()

        count = 0
        try:
            from options_owl.db import postgres as pg
            if not pg.is_connected():
                return 0

            if trades:
                await pg.write_option_flow_batch([
                    {
                        "ticker": t.ticker,
                        "contract": t.contract,
                        "option_type": t.option_type,
                        "strike": t.strike,
                        "expiry_date": t.expiry_date,
                        "trade_price": t.trade_price,
                        "trade_size": t.trade_size,
                        "trade_value": t.trade_value,
                        "conditions": t.conditions or None,
                        "is_sweep": t.is_sweep,
                        "is_above_ask": t.is_above_ask,
                        "is_below_bid": t.is_below_bid,
                        "exchange_id": t.exchange_id,
                        "sip_timestamp": t.sip_timestamp,
                    }
                    for t in trades
                ])
                count = len(trades)

            if bars:
                await pg.write_option_flow_5m(bars)

                # Publish to Redis for real-time delivery to all agents
                try:
                    from options_owl.db import redis_client
                    if redis_client.is_connected():
                        for bar in bars:
                            await redis_client.publish_flow_bar(bar)
                            await redis_client.set_latest_flow(bar["ticker"], bar)
                except Exception:
                    pass  # fire-and-forget

        except Exception as exc:
            logger.warning(f"FlowCollector flush failed: {exc}")

        return count

    # ------------------------------------------------------------------
    # WS connection
    # ------------------------------------------------------------------

    async def _ws_loop(self, api_key: str) -> None:
        """Connect to Polygon Options WS and collect O.T.* trade events."""
        try:
            import websockets
        except ImportError:
            logger.error("FlowCollector requires `websockets` package")
            return

        url = "wss://socket.polygon.io/options"
        reconnect_attempts = 0
        max_reconnect_delay = 300
        policy_violations = 0

        while self._ws_running:
            try:
                connected_at = 0.0
                async with websockets.connect(
                    url, ping_interval=30, ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    # Wait for initial "connected" message from Polygon
                    conn_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    conn_data = _json.loads(conn_msg)

                    # Check for max_connections
                    if isinstance(conn_data, list):
                        for item in conn_data:
                            if isinstance(item, dict) and item.get("status") == "max_connections":
                                logger.warning(
                                    "FlowCollector WS: max connections — "
                                    "waiting 30s for stale connections to clear"
                                )
                                await asyncio.sleep(30)
                                continue

                    # Authenticate
                    await ws.send(_json.dumps({"action": "auth", "params": api_key}))
                    auth_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    auth_data = _json.loads(auth_msg)

                    if not _ws_auth_ok(auth_data):
                        logger.error(f"FlowCollector WS auth failed: {auth_data}")
                        await asyncio.sleep(min(2 ** reconnect_attempts, max_reconnect_delay))
                        reconnect_attempts += 1
                        continue

                    logger.info("FlowCollector: authenticated on Polygon Options WS")

                    # Subscribe to T.* and Q.* (all options trades + quotes).
                    # Polygon Options WS does NOT support per-underlying prefix
                    # subscriptions (T.O:SPY silently matches nothing). We subscribe
                    # to ALL and filter client-side in _process_trade/_process_quote.
                    await ws.send(_json.dumps({
                        "action": "subscribe",
                        "params": "T.*,Q.*",
                    }))
                    logger.info(
                        f"FlowCollector: subscribed to T.*,Q.* "
                        f"(filtering client-side to {len(self._tickers)} tickers)"
                    )

                    self._ws_connected = True
                    self._last_msg_time = time.monotonic()
                    connected_at = time.time()

                    while self._ws_running:
                        try:
                            raw_msg = await asyncio.wait_for(
                                ws.recv(), timeout=WS_STALE_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            # No messages for WS_STALE_TIMEOUT seconds
                            from options_owl.execution.position_monitor import (
                                _is_market_hours,
                            )
                            if _is_market_hours():
                                age = self.last_msg_age
                                logger.warning(
                                    f"FlowCollector WS STALE: no messages for "
                                    f"{age:.0f}s during market hours — "
                                    f"forcing reconnect"
                                )
                                break  # exit inner loop → reconnect
                            continue  # outside market hours, silence is expected
                        try:
                            events = _json.loads(raw_msg)
                            if not isinstance(events, list):
                                events = [events]
                            for event in events:
                                ev_type = event.get("ev", "")
                                # Options WS uses T/Q (not OT/OQ)
                                if ev_type in ("T", "OT"):
                                    self._process_trade(event)
                                elif ev_type in ("Q", "OQ"):
                                    self._process_quote(event)
                            self._last_msg_time = time.monotonic()
                        except Exception as exc:
                            logger.debug(f"FlowCollector msg error: {exc}")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._ws_connected = False
                err_str = str(exc)

                # Detect policy violation (subscription tier doesn't include Options WS)
                if "1008" in err_str and "policy" in err_str:
                    policy_violations += 1
                    if policy_violations >= 3:
                        delay = min(300, 60 * policy_violations)
                        logger.error(
                            f"FlowCollector: {policy_violations} consecutive policy "
                            f"violations — Polygon plan may not include Options WS. "
                            f"Backing off {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                # Only reset reconnect counter if we were connected for > 60s
                # (avoids rapid reconnect loops when auth works but data feed rejects)
                if connected_at and (time.time() - connected_at) > 60:
                    reconnect_attempts = 0
                    policy_violations = 0

                delay = min(2 ** reconnect_attempts, max_reconnect_delay)
                logger.warning(
                    f"FlowCollector WS error: {exc} — reconnecting in {delay}s"
                )
                await asyncio.sleep(delay)
                reconnect_attempts += 1

        self._ws_connected = False

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_event(self, event: dict) -> None:
        """Process an event from an external WS (e.g. market_data_stream).

        Accepts both Options WS event types (OT/OQ) and standard types (T/Q)
        for option contracts (sym starts with "O:").
        """
        ev_type = event.get("ev", "")
        sym = event.get("sym", "")
        if ev_type in ("OT", "T") and sym.startswith("O:"):
            self._process_trade(event)
        elif ev_type in ("OQ", "Q") and sym.startswith("O:"):
            self._process_quote(event)

    def _process_quote(self, event: dict) -> None:
        """Update NBBO cache from O.Q.* quote event, compute greeks, publish to Redis."""
        contract = event.get("sym", "")
        bid = float(event.get("bp", 0) or 0)
        ask = float(event.get("ap", 0) or 0)
        if not contract or (bid <= 0 and ask <= 0):
            return

        self._nbbo_cache[contract] = (bid, ask, time.time())

        parsed = parse_occ_ticker(contract)
        if not parsed:
            return

        # Only process tickers in our universe (T.*/Q.* sends ALL options)
        if parsed["ticker"] not in self._tickers:
            return

        mid = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else bid or ask
        contract_key = (
            f"{parsed['ticker']}:{parsed['type']}:{parsed['strike']}:"
            f"{parsed['expiry']}"
        )

        # Legacy premium-only (for position monitor)
        self._redis_quote_buffer[contract_key] = (bid, ask, mid)

        # Full snapshot with greeks (for ML scan loop)
        bid_size = float(event.get("bs", 0) or 0)
        ask_size = float(event.get("as", 0) or 0)

        underlying_price = self._get_underlying_price(parsed["ticker"])
        iv = delta = gamma = theta = vega = None

        if underlying_price and mid > 0:
            try:
                iv, delta, gamma, theta, vega = _compute_greeks(
                    mid, underlying_price, parsed["strike"],
                    parsed["expiry"], parsed["type"],
                )
            except Exception:
                pass

        # Reset daily volume tracker
        today = date.today().isoformat()
        if self._volume_date != today:
            self._contract_volume.clear()
            self._volume_date = today

        self._redis_snapshot_buffer[contract_key] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "iv": iv,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "volume": self._contract_volume.get(contract, 0),
            "open_interest": 0,  # not available from WS, REST fallback fills this
            "underlying_price": underlying_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "strike": parsed["strike"],
            "expiry_date": parsed["expiry"],
            "option_type": parsed["type"],
        }

    def _get_underlying_price(self, ticker: str) -> float | None:
        """Get current underlying price from price_getter callback."""
        if self._price_getter:
            try:
                return self._price_getter(ticker)
            except Exception:
                pass
        return None

    def _process_trade(self, event: dict) -> None:
        """Process an O.T.* option trade event."""
        contract = event.get("sym", "")
        parsed = parse_occ_ticker(contract)
        if not parsed:
            return

        trade_size = int(event.get("s", 0) or 0)
        if trade_size < MIN_FLOW_SIZE:
            return

        # DTE filter
        try:
            expiry_dt = date.fromisoformat(parsed["expiry"])
            dte = (expiry_dt - date.today()).days
            if dte < 0 or dte > MAX_DTE:
                return
        except (ValueError, TypeError):
            return

        # Only process tickers in our universe
        if parsed["ticker"] not in self._tickers:
            return

        trade_price = float(event.get("p", 0) or 0)
        if trade_price <= 0:
            return

        conditions = event.get("c", []) or []
        is_sweep = SWEEP_CONDITION in conditions
        trade_value = trade_price * trade_size * 100

        # Aggressor detection from NBBO cache
        nbbo = self._nbbo_cache.get(contract)
        is_above_ask = None
        is_below_bid = None
        if nbbo:
            bid, ask, _ = nbbo
            is_above_ask, is_below_bid = detect_aggressor(trade_price, bid, ask)

        # Parse SIP timestamp
        # Polygon's options WS sends the trade timestamp 't' in MILLISECONDS.
        # Normalize defensively by magnitude (ns/us/ms/s) so we never produce
        # 1970-epoch bars (the prior code assumed ns and divided by 1e9, which
        # turned real ms timestamps into ~1970, breaking all option_flow_5m writes).
        raw_t = event.get("t", 0) or 0
        if raw_t > 1e16:
            _ts_s = raw_t / 1_000_000_000   # nanoseconds
        elif raw_t > 1e13:
            _ts_s = raw_t / 1_000_000       # microseconds
        elif raw_t > 1e10:
            _ts_s = raw_t / 1_000           # milliseconds (Polygon options WS)
        elif raw_t > 1e8:
            _ts_s = raw_t                   # seconds
        else:
            _ts_s = 0
        sip_dt = (
            datetime.fromtimestamp(_ts_s, tz=timezone.utc)
            if _ts_s else datetime.now(tz=timezone.utc)
        )

        flow_trade = FlowTrade(
            ticker=parsed["ticker"],
            contract=contract,
            option_type=parsed["type"],
            strike=parsed["strike"],
            expiry_date=parsed["expiry"],
            trade_price=trade_price,
            trade_size=trade_size,
            trade_value=trade_value,
            conditions=conditions,
            is_sweep=is_sweep,
            is_above_ask=is_above_ask,
            is_below_bid=is_below_bid,
            exchange_id=event.get("x"),
            sip_timestamp=sip_dt,
        )

        self._trade_buffer.append(flow_trade)

        # Track per-contract volume for snapshot enrichment
        self._contract_volume[contract] = (
            self._contract_volume.get(contract, 0) + trade_size
        )

        # Aggregate into 5-minute bar
        bar_key = (parsed["ticker"], _bar_start(sip_dt))
        bar = self._flow_bars.get(bar_key)
        if bar is None:
            bar = FlowBar5m(ticker=parsed["ticker"], bar_time=_bar_start(sip_dt))
            self._flow_bars[bar_key] = bar

        if parsed["type"] == "call":
            bar.call_volume += trade_size
            bar.call_value += trade_value
            bar.call_total_count += 1
            if is_sweep:
                bar.call_sweeps += 1
            if trade_size >= LARGE_TRADE_SIZE:
                bar.call_large_trades += 1
            if is_above_ask:
                bar.call_buyer_count += 1
        else:
            bar.put_volume += trade_size
            bar.put_value += trade_value
            bar.put_total_count += 1
            if is_sweep:
                bar.put_sweeps += 1
            if trade_size >= LARGE_TRADE_SIZE:
                bar.put_large_trades += 1
            if is_above_ask:
                bar.put_buyer_count += 1


def _ws_auth_ok(data: list | dict) -> bool:
    """Check if Polygon WS auth response indicates success."""
    if isinstance(data, list):
        return any(
            isinstance(m, dict) and m.get("status") == "auth_success"
            for m in data
        )
    if isinstance(data, dict):
        return data.get("status") == "auth_success"
    return False
