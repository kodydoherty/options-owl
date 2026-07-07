"""Webull executor — places real options orders via the official Webull OpenAPI.

Safety rails:
- PAPER_TRADE must be explicitly set to False to place real orders
- Max order size hard cap (MAX_ORDER_CONTRACTS)
- Every order is logged with full details before and after placement
- Kill switch: set WEBULL_KILL_SWITCH=true to halt all new orders instantly
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from loguru import logger

from options_owl.config.settings import Settings


# ---------------------------------------------------------------------------
# Hard safety limits — cannot be overridden by settings
# ---------------------------------------------------------------------------

MAX_ORDER_CONTRACTS = 100  # safety cap — sizing logic handles real limits
MAX_ORDER_VALUE = 5000.0  # absolute max dollar value per order


@dataclass
class OrderResult:
    """Result of placing an order."""

    success: bool
    order_id: str | None = None
    client_order_id: str | None = None
    error: str | None = None
    details: dict | None = None
    fill_status: str = "UNKNOWN"  # FILLED, PARTIAL, SUBMITTED, CANCELLED, FAILED
    filled_quantity: int | None = None  # How many contracts actually filled


@dataclass
class AccountInfo:
    """Webull account snapshot."""

    account_id: str
    total_asset: float
    cash_balance: float
    buying_power: float
    positions: list[dict]


def _round_option_price(price: float, side: str) -> float:
    """Round option limit price to Webull's required increments.

    Webull rules:
    - Premium >= $3.00 → must be in $0.05 increments
    - Premium < $3.00 → $0.01 increments (no rounding needed)

    For BUY orders, round UP to ensure fill (we're willing to pay slightly more).
    For SELL orders, round DOWN to ensure fill (we're willing to accept slightly less).
    """
    import math
    if price < 3.00:
        return round(price, 2)
    if side.upper() == "BUY":
        return round(math.ceil(price / 0.05) * 0.05, 2)
    else:
        return round(math.floor(price / 0.05) * 0.05, 2)


def _is_webull_rate_limit(exc: Exception) -> bool:
    """True if a Webull API exception is a 429 / rate-limit error."""
    s = str(exc).upper()
    return "429" in s or "TOO_MANY" in s or "RATE" in s and "LIMIT" in s


class WebullExecutor:
    """Manages the Webull OpenAPI connection and order execution."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._api_client = None
        self._trade_client = None
        self._data_client = None
        self._account_id: str | None = settings.WEBULL_ACCOUNT_ID or None
        # Cache: (ticker, strike, expiry, option_type) -> instrument_id
        self._instrument_cache: dict[tuple[str, float, str, str], str] = {}
        # Cache: instrument_id -> (bid, ask, mid, timestamp)
        self._quote_cache: dict[str, tuple[float, float, float, float]] = {}
        self._quote_cache_ttl = 3.0  # seconds
        # Balance cache: (value, timestamp) — avoid hitting API on every signal
        self._balance_cache: tuple[float, float] | None = None
        self._balance_cache_ttl = 60.0  # seconds
        # Order-status throttle (fix Webull 429 storm): per-order cache + min-interval + 429 cooldown
        # + a SHARED open-orders snapshot so the fill poll batches (one call covers all pending orders).
        self._os_cache: dict[str, tuple[dict | None, float]] = {}   # client_order_id -> (result, ts)
        self._os_cache_ttl = 2.5
        self._os_min_interval = 1.0      # min seconds between any order-status API calls
        self._os_cooldown = 6.0          # pause status calls this long after a 429
        self._os_last_call_ts = 0.0
        self._os_cooldown_until = 0.0
        self._oo_snapshot: dict[str, str] = {}   # client_order_id -> status (batch via get_open_orders)
        self._oo_snapshot_ts = 0.0
        self._oo_snapshot_ttl = 2.5
        self._os_lock: object = None     # lazily-created asyncio.Lock (event-loop bound)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _reconnect(self) -> None:
        """Force-reinitialize the Webull SDK clients (e.g., after stale connection)."""
        logger.warning("Webull reconnect: tearing down stale clients and reinitializing")
        self._api_client = None
        self._trade_client = None
        # The market-data client wraps _api_client; if we don't drop it too,
        # _ensure_data_client() (which early-returns when non-None) keeps serving a
        # client built on the torn-down api client — silently breaking the live-quote
        # entry-chase fallback after every reconnect.
        self._data_client = None
        self._ensure_clients()

    def _ensure_clients(self) -> None:
        """Lazy-init the Webull SDK clients."""
        if self._api_client is not None:
            return

        if not self.settings.WEBULL_APP_KEY or not self.settings.WEBULL_APP_SECRET:
            raise RuntimeError("WEBULL_APP_KEY and WEBULL_APP_SECRET must be set in .env")

        from webull.core.client import ApiClient
        from webull.trade.trade_client import TradeClient

        self._api_client = ApiClient(
            app_key=self.settings.WEBULL_APP_KEY,
            app_secret=self.settings.WEBULL_APP_SECRET,
            region_id="us",
        )

        try:
            self._trade_client = TradeClient(self._api_client)
        except Exception as exc:
            # Handle MANY_TOO_TOKEN: SDK tries to create auth tokens on init
            # and Webull limits to 10.  Bypass by disabling auto-token init.
            if "MANY_TOO_TOKEN" in str(exc) or "more than 10" in str(exc):
                logger.warning(
                    "Webull token limit hit — bypassing auto-token init. "
                    "Old tokens will expire in ~24h. Retrying without token check..."
                )
                # Monkey-patch to skip token init
                from webull.core.http.initializer.client_initializer import (
                    ClientInitializer,
                )
                _orig = ClientInitializer.init_token
                ClientInitializer.init_token = staticmethod(lambda *a, **kw: None)
                try:
                    self._trade_client = TradeClient(self._api_client)
                finally:
                    ClientInitializer.init_token = _orig
            else:
                raise

        logger.info("Webull API clients initialized")

    async def init(self) -> str:
        """Initialize and return the account ID.

        If WEBULL_ACCOUNT_ID is not set, auto-detects the account from the API.
        Selects MARGIN or CASH account based on MARGIN_ACCOUNT setting.
        """
        self._ensure_clients()
        acct_type = "MARGIN" if self.settings.MARGIN_ACCOUNT else "CASH"

        if not self._account_id:
            self._account_id = await self._detect_account_id()
            logger.info(f"Webull {acct_type} account detected: {self._account_id}")
        else:
            if not self.settings.MARGIN_ACCOUNT:
                # Only verify cash when not in margin mode
                await self._verify_cash_account(self._account_id)
            logger.info(f"Webull {acct_type} account configured: {self._account_id}")

        # Verify connectivity by fetching balance
        info = await self.get_account_info()
        logger.info(
            f"Webull connected — total: ${info.total_asset:,.2f}, "
            f"cash: ${info.cash_balance:,.2f}, "
            f"buying power: ${info.buying_power:,.2f}"
        )
        return self._account_id

    async def _detect_account_id(self) -> str:
        """Fetch account list and return the appropriate account ID.

        Selects MARGIN account when MARGIN_ACCOUNT=true, CASH otherwise.
        """
        response = await asyncio.to_thread(
            self._trade_client.account_v2.get_account_list
        )
        accounts = response.json() if hasattr(response, 'json') else response
        if isinstance(accounts, dict):
            account_list = accounts.get("accounts", accounts.get("data", []))
        elif isinstance(accounts, list):
            account_list = accounts
        else:
            raise RuntimeError(f"Unexpected account list response: {accounts}")

        if not account_list:
            raise RuntimeError("No Webull accounts found for this API key")

        want_margin = self.settings.MARGIN_ACCOUNT
        target_type = "MARGIN" if want_margin else "CASH"

        # Log all available accounts for visibility
        for acct in account_list:
            aid = acct.get("account_id", acct.get("accountId", ""))
            atype = acct.get("account_type", acct.get("accountType", ""))
            aclass = acct.get("account_class", acct.get("accountClass", ""))
            logger.info(f"Webull account found: id={aid} type={atype} class={aclass}")

        # Find the target account type
        for acct in account_list:
            acct_type = acct.get("account_type", acct.get("accountType", "")).upper()
            acct_class = acct.get("account_class", acct.get("accountClass", "")).upper()
            if target_type in acct_type and "INDIVIDUAL" in acct_class:
                account_id = str(acct.get("account_id", acct.get("accountId", "")))
                if account_id:
                    logger.info(
                        f"Selected {target_type} account: {account_id} "
                        f"(type={acct_type}, class={acct_class})"
                    )
                    return account_id

        raise RuntimeError(
            f"No Individual {target_type} account found. "
            f"MARGIN_ACCOUNT={want_margin}. Available accounts: "
            + ", ".join(
                f"{a.get('account_type')}({a.get('account_class')})"
                for a in account_list
            )
        )

    async def _verify_cash_account(self, account_id: str) -> None:
        """Verify that the configured account ID is a CASH account.

        Raises RuntimeError if the account is margin, futures, or crypto.
        """
        response = await asyncio.to_thread(
            self._trade_client.account_v2.get_account_list
        )
        accounts = response.json() if hasattr(response, 'json') else response
        if isinstance(accounts, dict):
            account_list = accounts.get("accounts", accounts.get("data", []))
        elif isinstance(accounts, list):
            account_list = accounts
        else:
            logger.warning("Could not verify account type — proceeding with caution")
            return

        for acct in account_list:
            aid = str(acct.get("account_id", acct.get("accountId", "")))
            if aid == account_id:
                acct_type = acct.get("account_type", acct.get("accountType", "")).upper()
                acct_class = acct.get("account_class", acct.get("accountClass", "")).upper()
                if acct_type != "CASH" or "INDIVIDUAL" not in acct_class:
                    raise RuntimeError(
                        f"WEBULL_ACCOUNT_ID {account_id} is a {acct_type} ({acct_class}) account. "
                        f"OptionsOwl only trades on Individual Cash accounts — never margin. "
                        f"Set WEBULL_ACCOUNT_ID to your cash account ID."
                    )
                logger.info(f"Verified account {account_id} is CASH ({acct_class})")
                return

        logger.warning(
            f"Account {account_id} not found in account list — "
            f"cannot verify type, proceeding with caution"
        )

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    async def get_account_info(self) -> AccountInfo:
        """Fetch current account balance and positions."""
        self._ensure_clients()

        try:
            balance_resp = await asyncio.to_thread(
                self._trade_client.account_v2.get_account_balance,
                self._account_id,
            )
        except (ValueError, ConnectionError, OSError) as exc:
            if "connection" in str(exc).lower():
                logger.warning(f"Webull stale connection in get_account_info ({exc}) — reconnecting")
                # _reconnect does blocking HTTP token init — run off the event
                # loop so a slow Webull (the exact reconnect trigger) can't freeze
                # the monitor's sell path for every other trade.
                await asyncio.to_thread(self._reconnect)
                balance_resp = await asyncio.to_thread(
                    self._trade_client.account_v2.get_account_balance,
                    self._account_id,
                )
            else:
                raise
        balance = balance_resp.json() if hasattr(balance_resp, 'json') else balance_resp
        if isinstance(balance, dict):
            bal = balance
        else:
            bal = {}

        # Extract fields (API response structure may vary)
        total_asset = float(bal.get("total_asset", bal.get("totalAsset", 0)))
        cash_balance = float(bal.get("total_cash_balance", bal.get("totalCashBalance", 0)))

        # Buying power from currency assets
        buying_power = cash_balance
        currency_assets = bal.get("account_currency_assets", bal.get("accountCurrencyAssets", []))
        if currency_assets:
            for ca in currency_assets:
                bp = ca.get("cash_power", ca.get("cashPower"))
                if bp:
                    buying_power = float(bp)
                    break

        # Positions
        pos_resp = await asyncio.to_thread(
            self._trade_client.account_v2.get_account_position,
            self._account_id,
        )
        pos_data = pos_resp.json() if hasattr(pos_resp, 'json') else pos_resp
        holdings = []
        if isinstance(pos_data, dict):
            for key in ("positions", "holdings", "data", "option_positions"):
                if key in pos_data:
                    holdings = pos_data[key]
                    break
        elif isinstance(pos_data, list):
            holdings = pos_data

        return AccountInfo(
            account_id=self._account_id,
            total_asset=total_asset,
            cash_balance=cash_balance,
            buying_power=buying_power,
            positions=holdings,
        )

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def _check_kill_switch(self) -> None:
        """Raise if kill switch is active (env OR Redis dashboard override)."""
        # Check env first
        if getattr(self.settings, "WEBULL_KILL_SWITCH", False):
            raise RuntimeError("WEBULL_KILL_SWITCH is active — all orders blocked")
        # Check Redis dashboard override
        try:
            from options_owl.db import redis_client
            agent_id = getattr(self.settings, "AGENT_ID", "")
            if agent_id and redis_client.is_connected():
                override = await redis_client.get_kill_switch(agent_id)
                if override is True:
                    raise RuntimeError("Kill switch activated via dashboard — all orders blocked")
        except RuntimeError:
            raise
        except Exception:
            pass  # Redis failure should never block trading

    def _check_safety_limits(
        self, contracts: int, premium: float, action: str,
    ) -> None:
        """Enforce hard safety caps.  Caps only apply to BUY (entry) orders —
        SELL (exit) orders must be able to close any position size."""
        if action.upper() != "SELL":
            if contracts > MAX_ORDER_CONTRACTS:
                raise ValueError(
                    f"Order size {contracts} exceeds hard cap {MAX_ORDER_CONTRACTS}"
                )
            order_value = contracts * premium * 100
            if order_value > MAX_ORDER_VALUE:
                raise ValueError(
                    f"Order value ${order_value:.2f} exceeds hard cap ${MAX_ORDER_VALUE:.2f}"
                )
        if self.settings.PAPER_TRADE:
            raise RuntimeError(
                "PAPER_TRADE=true — set PAPER_TRADE=false in .env to place real orders"
            )

    async def _find_position_id(
        self,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        retries: int = 3,
    ) -> str | None:
        """Look up the Webull position_id for an open option position.

        Retries up to 3 times with backoff — the API sometimes returns an empty
        list on the first call after a buy (timing) or after an auth refresh.
        """
        for attempt in range(retries):
            if attempt > 0:
                await asyncio.sleep(2 * attempt)  # 2s, 4s backoff

            try:
                response = await asyncio.to_thread(
                    self._trade_client.account_v2.get_account_position,
                    self._account_id,
                )
                # Handle various response shapes from the SDK
                if hasattr(response, "json"):
                    try:
                        positions = response.json()
                    except Exception:
                        positions = response
                else:
                    positions = response

                # Log raw response on first attempt to diagnose format issues
                if attempt == 0:
                    logger.debug(
                        f"_find_position_id: raw response type={type(positions).__name__}, "
                        f"keys={list(positions.keys()) if isinstance(positions, dict) else 'N/A'}, "
                        f"preview={str(positions)[:500]}"
                    )

                # Unwrap nested response — try multiple known wrapper keys
                if isinstance(positions, dict):
                    for key in ("positions", "holdings", "data", "option_positions"):
                        if key in positions:
                            positions = positions[key]
                            break
                    else:
                        # If dict has no known wrapper, it might be a single position
                        if "ticker" in positions or "symbol" in positions:
                            positions = [positions]
                        else:
                            positions = []
                if not isinstance(positions, list):
                    logger.warning(f"_find_position_id: unexpected response type {type(positions)}: {str(positions)[:300]}")
                    continue

                logger.debug(
                    f"_find_position_id: searching {len(positions)} positions "
                    f"for {ticker} ${strike} {option_type} exp={expiry_date}"
                    f" (attempt {attempt + 1}/{retries})"
                )

                # If API returned 0 positions and we have retries left, retry
                if len(positions) == 0 and attempt < retries - 1:
                    logger.warning(
                        f"_find_position_id: Webull returned 0 positions "
                        f"(attempt {attempt + 1}/{retries}), retrying..."
                    )
                    continue

                for pos in positions:
                    # Try top-level fields first (flat response), then nested legs
                    pos_ticker = pos.get("ticker", pos.get("symbol", "")).upper()
                    pos_type = pos.get("option_type", "").upper()
                    pos_expiry = pos.get("option_expire_date", pos.get("expiry_date", ""))
                    pos_strike = float(pos.get("option_exercise_price", pos.get("strike_price", pos.get("strike", 0))))
                    pid = pos.get("position_id", pos.get("id", ""))

                    if (
                        pos_ticker == ticker.upper()
                        and pos_type == option_type.upper()
                        and pos_expiry == expiry_date
                        and abs(pos_strike - strike) < 0.01
                        and pid
                    ):
                        return str(pid)

                    # Also check nested legs (multi-leg positions)
                    legs = pos.get("legs", [])
                    for leg in legs:
                        if (
                            leg.get("symbol", "").upper() == ticker.upper()
                            and leg.get("option_type", "").upper() == option_type.upper()
                            and leg.get("option_expire_date", "") == expiry_date
                            and abs(float(leg.get("option_exercise_price", 0)) - strike) < 0.01
                        ):
                            return str(pos.get("position_id", ""))

                # Positions returned but no match — don't retry, it's genuinely not there
                if len(positions) > 0:
                    logger.warning(
                        f"_find_position_id: {len(positions)} positions found but none match "
                        f"{ticker} ${strike} {option_type} exp={expiry_date}"
                    )
                    return None

            except Exception as exc:
                logger.warning(f"_find_position_id attempt {attempt + 1} failed: {exc}")

        logger.warning(
            f"_find_position_id: gave up after {retries} attempts for "
            f"{ticker} ${strike} {option_type} exp={expiry_date}"
        )
        return None

    async def place_option_order(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        side: str,
        contracts: int,
        limit_price: float,
        has_webull_order_id: bool = False,
    ) -> OrderResult:
        """Place a single-leg option order.

        Parameters
        ----------
        ticker : str
            Underlying symbol (e.g., "SPY")
        strike : float
            Option strike price
        expiry_date : str
            Expiry in YYYY-MM-DD format
        option_type : str
            "CALL" or "PUT"
        side : str
            "BUY" or "SELL"
        contracts : int
            Number of contracts
        limit_price : float
            Limit price per contract
        """
        self._ensure_clients()
        await self._check_kill_switch()
        logger.debug(
            f"WEBULL safety check: side={side} contracts={contracts} "
            f"limit=${limit_price:.2f} paper_trade={self.settings.PAPER_TRADE} "
            f"kill_switch={getattr(self.settings, 'WEBULL_KILL_SWITCH', False)}"
        )
        self._check_safety_limits(contracts, limit_price, side)

        # Webull price increment rules: >= $3.00 must use $0.05 steps, < $3.00 uses $0.01
        limit_price = _round_option_price(limit_price, side)

        client_order_id = uuid.uuid4().hex[:32]

        logger.info(
            f"WEBULL ORDER: {side} {contracts}x {ticker} "
            f"${strike} {option_type} exp={expiry_date} @ ${limit_price:.2f} "
            f"(value=${contracts * limit_price * 100:.2f}) "
            f"[client_id={client_order_id}]"
        )

        # close_contracts payload for SELL-to-close (None for BUY).  Computed
        # once up front so the BUY-side fill-escalation ladder can rebuild the
        # payload on each re-priced attempt without re-doing the lookup.
        close_contracts: list[dict] | None = None

        order_payload = self._build_order_payload(
            client_order_id=client_order_id,
            ticker=ticker,
            strike=strike,
            expiry_date=expiry_date,
            option_type=option_type,
            side=side,
            contracts=contracts,
            limit_price=limit_price,
            close_contracts=close_contracts,
        )

        # For SELL orders, look up the position_id and add close_contracts
        # so Webull knows this is sell-to-close (not sell-to-open / covered call)
        if side.upper() == "SELL":
            position_id = await self._find_position_id(
                ticker, strike, expiry_date, option_type,
            )
            if position_id:
                order_payload[0]["close_contracts"] = [{
                    "position_id": position_id,
                    "quantity": str(contracts),
                }]
                logger.info(f"WEBULL SELL_TO_CLOSE: position_id={position_id}")
            elif has_webull_order_id:
                # Position lookup failed but we bought this on Webull.
                # BLOCK the sell — without close_contracts, Webull may interpret
                # this as sell-to-open (naked short).  Log an alert so we can
                # investigate and manually close on Webull if needed.
                logger.error(
                    f"WEBULL SELL BLOCKED (no position_id): {ticker} ${strike} "
                    f"{option_type} exp={expiry_date} — trade has webull_order_id "
                    f"but position lookup returned nothing after retries. "
                    f"MANUAL CLOSE MAY BE REQUIRED on Webull."
                )
                return OrderResult(
                    success=False,
                    error=(
                        f"Position lookup failed for {ticker} ${strike} {option_type} "
                        f"— blocked sell to prevent accidental short. Manual close may be needed."
                    ),
                )
            else:
                # No matching position on Webull and no webull_order_id —
                # this was paper-only, skip.
                logger.warning(
                    f"WEBULL SELL SKIPPED: no position found for {ticker} ${strike} "
                    f"{option_type} exp={expiry_date} — no live position to close"
                )
                return OrderResult(
                    success=False,
                    error=f"No Webull position found for {ticker} ${strike} {option_type} — nothing to close",
                )

        logger.debug(f"WEBULL payload: {order_payload}")

        # BUY (entry) orders: chase the fill with a re-pricing escalation ladder.
        if side.upper() == "BUY":
            return await self._place_buy_with_escalation(
                ticker=ticker,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type,
                contracts=contracts,
                initial_limit=limit_price,
            )

        # SELL (exit) orders. When ENABLE_FAST_EXIT_CHASE is on, run the tight in-line
        # tiered chase (a few progressively-marketable prices, ~2.5s apart) so a fast-diving
        # 0DTE fills before the price craters — same double-fill safety as the entry chase.
        # When OFF (default), keep the legacy single-submit path below; exit-side escalation
        # is then driven only by paper_trader.close_webull_position across monitor cycles.
        if getattr(self.settings, "ENABLE_FAST_EXIT_CHASE", False):
            return await self._place_sell_with_escalation(
                ticker=ticker,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type,
                contracts=contracts,
                initial_limit=limit_price,
            )

        # ---- SELL: single submit + wait (legacy behavior, flag OFF) ----
        try:
            order_id, result, error = await self._submit_order_payload(order_payload)
            if not order_id:
                logger.error(f"WEBULL ORDER REJECTED: {error}")
                return OrderResult(
                    success=False,
                    client_order_id=client_order_id,
                    error=str(error),
                    details=result if isinstance(result, dict) else None,
                    fill_status="FAILED",
                )

            logger.info(
                f"WEBULL ORDER SUBMITTED: {side} {contracts}x {ticker} "
                f"${strike} {option_type} — order_id={order_id}, verifying fill..."
            )

            # SELL orders get a SHORT wait — 0DTE premiums crash fast, so we retry
            # quickly with a fresh (lower) price if the first attempt doesn't fill.
            # 10s/attempt let a tanking option run away across 4 retries (~50s) while
            # the limit sat above the falling market; matched to the fast-exit cadence.
            timeout = float(getattr(self.settings, "WEBULL_EXIT_PER_ATTEMPT_SEC", 2.5) or 2.5) * 2
            poll = float(getattr(self.settings, "WEBULL_EXIT_POLL_SEC", 1.0) or 1.0)
            fill_status = await self._wait_for_fill(
                client_order_id, timeout_seconds=timeout, poll_interval=poll,
            )

            if fill_status == "FILLED":
                logger.info(
                    f"WEBULL ORDER FILLED: {side} {contracts}x {ticker} "
                    f"${strike} {option_type} — order_id={order_id}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="FILLED",
                )
            elif fill_status in ("PARTIAL_FILLED", "PARTIAL"):
                filled_qty = await self._get_filled_quantity(client_order_id)
                logger.warning(
                    f"WEBULL ORDER PARTIAL FILL: {side} {filled_qty or '?'}/{contracts}x "
                    f"{ticker} ${strike} {option_type} — order_id={order_id}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="PARTIAL",
                    filled_quantity=filled_qty,
                )
            else:
                # Order submitted but not filled — cancel it to avoid stale orders
                logger.warning(
                    f"WEBULL ORDER NOT FILLED (status={fill_status}): {side} {contracts}x "
                    f"{ticker} ${strike} {option_type} — cancelling stale order"
                )
                await self.cancel_order(client_order_id)
                return OrderResult(
                    success=False,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    error=f"Order not filled after {timeout:.0f}s (status={fill_status}), cancelled",
                    fill_status=fill_status or "SUBMITTED",
                )

        except Exception as exc:
            logger.error(f"WEBULL ORDER ERROR: {type(exc).__name__}: {exc}")
            return OrderResult(
                success=False,
                client_order_id=client_order_id,
                error=str(exc),
                fill_status="FAILED",
            )

    @staticmethod
    def _build_order_payload(
        *,
        client_order_id: str,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        side: str,
        contracts: int,
        limit_price: float,
        close_contracts: list[dict] | None = None,
    ) -> list[dict]:
        """Build the single-leg option order payload Webull's API expects."""
        # Defensive chokepoint: EVERY order's limit price must sit on a legal
        # Webull increment (penny < $3, nickel >= $3) or the API rejects it with
        # OPTION_PRICE_STEP_GTE. Round here so no caller path can submit e.g. $3.07.
        limit_price = _round_option_price(limit_price, side)
        leg = {
            "side": side.upper(),
            "quantity": str(contracts),
            "symbol": ticker.upper(),
            "strike_price": str(strike),
            "option_expire_date": expiry_date,
            "instrument_type": "OPTION",
            "option_type": option_type.upper(),
            "market": "US",
        }
        order = {
            "client_order_id": client_order_id,
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "quantity": str(contracts),
            "limit_price": f"{limit_price:.2f}",
            "option_strategy": "SINGLE",
            "side": side.upper(),
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "legs": [leg],
        }
        if close_contracts:
            order["close_contracts"] = close_contracts
        return [order]

    async def _submit_order_payload(
        self, order_payload: list[dict],
    ) -> tuple[str | None, object, object]:
        """Submit an order payload (with stale-connection auto-reconnect).

        Returns ``(order_id, raw_result, error)``.  ``order_id`` is ``None`` when
        the order was rejected.
        """
        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.place_option,
                self._account_id,
                order_payload,
            )
        except (ValueError, ConnectionError, OSError) as conn_exc:
            if "no active connection" in str(conn_exc).lower() or "connection" in str(conn_exc).lower():
                logger.warning(
                    f"WEBULL stale connection ({conn_exc}) — reconnecting and retrying order"
                )
                # Run blocking reconnect off the event loop (see get_account_info).
                await asyncio.to_thread(self._reconnect)
                response = await asyncio.to_thread(
                    self._trade_client.order_v2.place_option,
                    self._account_id,
                    order_payload,
                )
            else:
                raise
        result = response.json() if hasattr(response, "json") else response
        logger.debug(f"WEBULL raw response: {result}")

        if isinstance(result, dict):
            order_id = result.get("order_id", result.get("orderId"))
            error = result.get("error", result.get("msg"))
        elif isinstance(result, list) and result:
            order_id = result[0].get("order_id", result[0].get("orderId"))
            error = result[0].get("error")
        else:
            order_id = None
            error = f"Unexpected response: {result}"
        return order_id, result, error

    async def _get_filled_quantity(self, client_order_id: str) -> int | None:
        """Best-effort lookup of how many contracts have filled for an order."""
        try:
            detail = await self.get_order_status(client_order_id)
            if detail and isinstance(detail, dict):
                return int(float(detail.get("filled_quantity", 0) or 0))
        except Exception:
            pass
        return None

    async def _confirm_cancelled(
        self, client_order_id: str, timeout_seconds: float = 6.0,
    ) -> str:
        """Cancel an order and confirm it is no longer working.

        CRITICAL double-fill guard: the BUY escalation ladder MUST NOT submit a
        re-priced order while a prior order is still live, or we could end up
        double-filled.  This cancels the prior order and then polls its status
        until it reports a terminal state (CANCELLED / REJECTED / EXPIRED) or,
        importantly, FILLED / PARTIAL_FILLED (the order filled in the race
        between our timeout and the cancel — the caller must honor that fill).

        Returns the terminal status observed (or the last status seen on
        timeout).  Callers treat anything other than a clean cancel as a fill
        that must be respected.
        """
        import time as _time

        await self.cancel_order(client_order_id)

        deadline = _time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"
        while _time.monotonic() < deadline:
            detail = await self.get_order_status(client_order_id)
            status = ""
            if isinstance(detail, dict):
                status = (detail.get("status") or "").upper()
                if not status:
                    orders = detail.get("orders") or []
                    if orders and isinstance(orders, list):
                        status = (orders[0].get("status") or "").upper()
                if not status:
                    filled = float(detail.get("filled_quantity", 0) or 0)
                    total = float(detail.get("total_quantity", 0) or 0)
                    if filled > 0:
                        status = "FILLED" if (total and filled >= total) else "PARTIAL_FILLED"
            if status:
                last_status = status
                if status in (
                    "CANCELLED", "REJECTED", "EXPIRED",
                    "FILLED", "PARTIAL_FILLED", "PARTIAL",
                ):
                    return status
            await asyncio.sleep(1.0)

        logger.warning(
            f"WEBULL CANCEL UNCONFIRMED: {client_order_id} still status={last_status} "
            f"after {timeout_seconds:.0f}s — NOT re-submitting to avoid double-fill"
        )
        return last_status

    async def _place_buy_with_escalation(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        contracts: int,
        initial_limit: float,
    ) -> OrderResult:
        """Place a BUY (entry) order, chasing the fill with a re-pricing ladder.

        0DTE premiums move fast — a single limit at ``ask × 1.05`` frequently
        misses (cost us a verified +100% NVDA put and a QQQ entry).  This mirrors
        the SELL-side fast escalation: short per-attempt waits, re-pricing UP
        toward/through the ask on each miss, capped so we never pay absurdly
        above the ask.

        Ladder (attempt N, 0-indexed):
            limit = ask × (1 + step) where step grows 5% / 10% / 15% / ...
            ceiling = ask × (1 + WEBULL_ENTRY_MAX_CHASE_PCT/100)  (default 15%)

        ``initial_limit`` is the caller-supplied entry price (already
        ``ask × (1 + WEBULL_ENTRY_AGGRESS_PCT/100)``); it anchors attempt 1 and,
        when no fresh quote is available, the implied ask for the ceiling.

        DOUBLE-FILL GUARD: before each re-priced attempt we cancel the prior
        order and CONFIRM it reached a terminal/non-working state.  If the prior
        order actually filled during the race, we return that fill instead of
        submitting again.  A fill on any attempt stops the ladder immediately.

        Tunable via settings (add to settings.py / .env to override the inline
        defaults):
            WEBULL_ENTRY_FILL_ATTEMPTS  (default 3)
            WEBULL_ENTRY_MAX_CHASE_PCT  (default 15.0)
        """
        max_attempts = int(getattr(self.settings, "WEBULL_ENTRY_FILL_ATTEMPTS", 4) or 4)
        max_attempts = max(1, max_attempts)
        max_chase_pct = float(getattr(self.settings, "WEBULL_ENTRY_MAX_CHASE_PCT", 15.0) or 15.0)
        # Index 0DTE (SPY/QQQ/...) crosses harder on rung 1 — penny-wide spreads make the bigger
        # cross cheap, and the fast index tape punishes a passive limit (the 2026-06-25 misses).
        try:
            from options_owl.risk.exit_v5.config import INDEX_TICKERS
            is_index = ticker.upper() in INDEX_TICKERS
        except Exception:
            is_index = False
        if is_index:
            aggress_pct = float(getattr(self.settings, "WEBULL_ENTRY_INDEX_AGGRESS_PCT", 10.0) or 10.0)
        else:
            aggress_pct = float(getattr(self.settings, "WEBULL_ENTRY_AGGRESS_PCT", 5.0) or 5.0)

        # Faster cadence than the old hardcoded 12s/3s: a fast 0DTE premium runs away before a
        # 12s rung times out, so we re-price every few seconds instead.
        per_attempt_timeout = float(getattr(self.settings, "WEBULL_ENTRY_PER_ATTEMPT_SEC", 4.0) or 4.0)
        poll_interval = float(getattr(self.settings, "WEBULL_ENTRY_POLL_SEC", 1.0) or 1.0)
        use_live_quote = bool(getattr(self.settings, "WEBULL_ENTRY_USE_LIVE_QUOTE", True))

        # Fallback reference ask derived from the caller's aggressive limit so the ceiling is
        # anchored even when no live quote is available.
        base_ask = initial_limit / (1 + aggress_pct / 100) if aggress_pct else initial_limit
        last_result: OrderResult | None = None
        last_client_id: str | None = None

        for attempt in range(max_attempts):
            # Price EVERY rung (including rung 0) off the freshest available ask — harvester live
            # Redis quote first, HTTP fallback — so the limit isn't stale by the time the order
            # rests (the root cause of the choppy-tape misses). Only when no live quote is
            # available do we fall back to the back-derived base_ask.
            ask = base_ask
            if use_live_quote or attempt > 0:
                fresh = await self._fetch_ask(ticker, strike, expiry_date, option_type)
                if fresh and fresh > 0:
                    ask = fresh
                    base_ask = fresh  # keep ceiling tied to the latest ask

            # Step ladder: aggress_pct over the ask on rung 1 (index starts higher), +5%/rung
            # thereafter, capped at the configured max chase percentage.
            step_pct = min(aggress_pct + attempt * 5.0, max_chase_pct)
            ceiling = round(base_ask * (1 + max_chase_pct / 100), 2)
            limit = round(ask * (1 + step_pct / 100), 2)
            if limit > ceiling:
                limit = ceiling
            limit = _round_option_price(limit, "BUY")
            # Never exceed the ceiling after rounding-up. The ceiling itself is a
            # plain round(.,2) and may NOT be a nickel multiple, so rounding it UP
            # can overshoot and `ceiling - 0.05` would land on an illegal step
            # (e.g. 3.12 -> 3.07). Round DOWN to the largest legal increment at or
            # below the ceiling instead (penny < $3, nickel >= $3).
            if limit > ceiling:
                limit = _round_option_price(ceiling, "SELL")

            # Hard value-cap rail, re-applied per rung. _check_safety_limits validates the
            # INITIAL limit once (pre-escalation); the chase reprices UPWARD off a fresh ask,
            # so the escalated limit can float the order value past MAX_ORDER_VALUE. Clamp the
            # limit (rounded DOWN to a legal step) so the rail holds on a fast run-up. No-op at
            # current account sizes; a real brake at scale. An un-marketable clamped limit
            # correctly means "don't overpay past the hard cap".
            if contracts > 0 and limit * contracts * 100 > MAX_ORDER_VALUE:
                capped = MAX_ORDER_VALUE / (contracts * 100)
                clamped = _round_option_price(capped, "SELL")
                if clamped < limit:
                    logger.warning(
                        f"WEBULL ENTRY CHASE: limit ${limit:.2f} → ${clamped:.2f} to hold "
                        f"${MAX_ORDER_VALUE:.0f} order-value rail ({contracts}x)"
                    )
                    limit = clamped

            client_order_id = uuid.uuid4().hex[:32]
            last_client_id = client_order_id

            logger.info(
                f"WEBULL ENTRY CHASE attempt {attempt + 1}/{max_attempts}: "
                f"BUY {contracts}x {ticker} ${strike} {option_type} "
                f"@ ${limit:.2f} (ask=${ask:.2f}, step={step_pct:.0f}%, "
                f"ceiling=${ceiling:.2f}) [client_id={client_order_id}]"
            )

            payload = self._build_order_payload(
                client_order_id=client_order_id,
                ticker=ticker,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type,
                side="BUY",
                contracts=contracts,
                limit_price=limit,
            )

            try:
                order_id, result, error = await self._submit_order_payload(payload)
            except Exception as exc:
                logger.error(f"WEBULL ORDER ERROR: {type(exc).__name__}: {exc}")
                last_result = OrderResult(
                    success=False,
                    client_order_id=client_order_id,
                    error=str(exc),
                    fill_status="FAILED",
                )
                # Submission raised — nothing was placed, safe to try next rung.
                continue

            if not order_id:
                logger.error(f"WEBULL ORDER REJECTED: {error}")
                last_result = OrderResult(
                    success=False,
                    client_order_id=client_order_id,
                    error=str(error),
                    details=result if isinstance(result, dict) else None,
                    fill_status="FAILED",
                )
                # No live order to cancel; advance the ladder.
                continue

            fill_status = await self._wait_for_fill(
                client_order_id,
                timeout_seconds=per_attempt_timeout,
                poll_interval=poll_interval,
            )

            if fill_status == "FILLED":
                logger.info(
                    f"WEBULL ENTRY FILLED (attempt {attempt + 1}): BUY {contracts}x "
                    f"{ticker} ${strike} {option_type} @ ${limit:.2f} — order_id={order_id}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="FILLED",
                )

            if fill_status in ("PARTIAL_FILLED", "PARTIAL"):
                # Partial fill: stop the ladder. Cancelling the remainder is
                # fine (we already own some contracts); re-pricing the rest
                # risks an over-large position. Return the partial as success.
                filled_qty = await self._get_filled_quantity(client_order_id)
                logger.warning(
                    f"WEBULL ENTRY PARTIAL FILL (attempt {attempt + 1}): "
                    f"{filled_qty or '?'}/{contracts}x {ticker} ${strike} "
                    f"{option_type} @ ${limit:.2f} — order_id={order_id}, "
                    f"cancelling unfilled remainder"
                )
                await self.cancel_order(client_order_id)
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="PARTIAL",
                    filled_quantity=filled_qty,
                )

            # Not filled. Record the miss outcome in case this is the last rung.
            last_result = OrderResult(
                success=False,
                order_id=str(order_id),
                client_order_id=client_order_id,
                error=(
                    f"Order not filled after {per_attempt_timeout:.0f}s "
                    f"(status={fill_status})"
                ),
                fill_status=fill_status or "SUBMITTED",
            )

            # DOUBLE-FILL GUARD: cancel-and-CONFIRM before any re-submit.
            confirm_status = await self._confirm_cancelled(client_order_id)
            if confirm_status in ("FILLED", "PARTIAL_FILLED", "PARTIAL"):
                # The order filled in the race with our cancel — honor it and
                # STOP. Never submit another rung on top of a live fill.
                filled_qty = (
                    None if confirm_status == "FILLED"
                    else await self._get_filled_quantity(client_order_id)
                )
                logger.warning(
                    f"WEBULL ENTRY FILLED DURING CANCEL (attempt {attempt + 1}): "
                    f"{ticker} ${strike} {option_type} status={confirm_status} "
                    f"order_id={order_id} — honoring fill, halting chase"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="FILLED" if confirm_status == "FILLED" else "PARTIAL",
                    filled_quantity=filled_qty,
                )

            if confirm_status not in ("CANCELLED", "REJECTED", "EXPIRED"):
                # Could not confirm the prior order is dead. Do NOT re-submit —
                # an uncancelled working order + a new one = double fill.
                logger.error(
                    f"WEBULL ENTRY CHASE ABORTED: could not confirm cancel of "
                    f"order_id={order_id} (status={confirm_status}) — refusing to "
                    f"re-submit to avoid double-fill"
                )
                return OrderResult(
                    success=False,
                    order_id=str(order_id),
                    client_order_id=client_order_id,
                    error=(
                        f"Entry not filled and prior order cancel unconfirmed "
                        f"(status={confirm_status}) — aborted chase to avoid double-fill"
                    ),
                    fill_status=fill_status or "SUBMITTED",
                )

            logger.warning(
                f"WEBULL ENTRY NOT FILLED (attempt {attempt + 1}/{max_attempts}): "
                f"{ticker} ${strike} {option_type} @ ${limit:.2f} — prior order "
                f"cancelled (confirmed), escalating"
            )

        # Ladder exhausted without a fill. Prior order already cancelled-confirmed.
        if last_result is None:
            last_result = OrderResult(
                success=False,
                client_order_id=last_client_id,
                error="No order attempts made",
                fill_status="FAILED",
            )
        logger.warning(
            f"WEBULL ENTRY MISS: BUY {contracts}x {ticker} ${strike} {option_type} "
            f"— not filled after {max_attempts} chase attempts"
        )
        return last_result

    async def _place_sell_with_escalation(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        contracts: int,
        initial_limit: float,
    ) -> OrderResult:
        """Fast, tiered EXIT chase — the sell-side mirror of _place_buy_with_escalation.

        A fast-diving 0DTE gives back profit while the slow cross-cycle retry advances one
        price tier per 5s monitor cycle. This crosses DOWN toward/below the FRESH bid in a
        few tight rungs (~2.5s apart): rung 1 barely marketable, each further rung more
        marketable, floored at WEBULL_EXIT_MAX_DISCOUNT_PCT below the bid — so if one price
        doesn't fill fast we cross harder immediately.

        SAFETY (identical invariant to the entry chase): cancel-AND-CONFIRM the working
        order before EVERY re-price. If a rung fills during the cancel race, honor it and
        STOP. If the cancel cannot be confirmed dead, ABORT — never leave two live sell
        orders (a stray second sell would oversell the position / go naked short).
        """
        max_attempts = int(getattr(self.settings, "WEBULL_EXIT_FILL_ATTEMPTS", 4) or 4)
        max_attempts = max(1, max_attempts)
        per_attempt_timeout = float(getattr(self.settings, "WEBULL_EXIT_PER_ATTEMPT_SEC", 2.5) or 2.5)
        poll_interval = float(getattr(self.settings, "WEBULL_EXIT_POLL_SEC", 1.0) or 1.0)
        aggress_pct = float(getattr(self.settings, "WEBULL_EXIT_AGGRESS_PCT", 2.0) or 2.0)
        step_pct = float(getattr(self.settings, "WEBULL_EXIT_STEP_PCT", 6.0) or 6.0)
        max_discount = float(getattr(self.settings, "WEBULL_EXIT_MAX_DISCOUNT_PCT", 25.0) or 25.0)
        use_live_quote = bool(getattr(self.settings, "WEBULL_ENTRY_USE_LIVE_QUOTE", True))

        # Reference bid: the caller's limit was priced off a bid at the start of the cycle;
        # use it as the fallback anchor when no fresh quote is available. The per-rung floor
        # is measured off this anchor so the discount can't run away on a bad fresh read.
        base_bid = initial_limit if initial_limit and initial_limit > 0 else 0.01
        last_result: OrderResult | None = None
        last_client_id: str | None = None

        for attempt in range(max_attempts):
            # Price each rung off the FRESHEST bid (harvester Redis first, HTTP fallback).
            bid = base_bid
            if use_live_quote or attempt > 0:
                fresh = await self._fetch_bid(ticker, strike, expiry_date, option_type)
                if fresh and fresh > 0:
                    bid = fresh
                    base_bid = fresh  # keep the floor anchored to the latest bid

            # Cross progressively further BELOW the bid; capped at max_discount below it.
            discount_pct = min(aggress_pct + attempt * step_pct, max_discount)
            floor_price = round(base_bid * (1 - max_discount / 100), 2)
            limit = round(bid * (1 - discount_pct / 100), 2)
            if limit < floor_price:
                limit = floor_price
            limit = max(0.01, limit)
            # Round to a legal step. Round DOWN (SELL) so we never round UP through the floor.
            limit = _round_option_price(limit, "SELL")
            if limit < 0.01:
                limit = 0.01

            client_order_id = uuid.uuid4().hex[:32]
            last_client_id = client_order_id

            logger.info(
                f"WEBULL EXIT CHASE attempt {attempt + 1}/{max_attempts}: "
                f"SELL {contracts}x {ticker} ${strike} {option_type} "
                f"@ ${limit:.2f} (bid=${bid:.2f}, cross={discount_pct:.0f}% below, "
                f"floor=${floor_price:.2f}) [client_id={client_order_id}]"
            )

            payload = self._build_order_payload(
                client_order_id=client_order_id,
                ticker=ticker,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type,
                side="SELL",
                contracts=contracts,
                limit_price=limit,
            )

            try:
                order_id, result, error = await self._submit_order_payload(payload)
            except Exception as exc:
                logger.error(f"WEBULL EXIT ORDER ERROR: {type(exc).__name__}: {exc}")
                last_result = OrderResult(
                    success=False, client_order_id=client_order_id,
                    error=str(exc), fill_status="FAILED",
                )
                continue  # submission raised — nothing placed, safe to try next rung

            if not order_id:
                logger.error(f"WEBULL EXIT ORDER REJECTED: {error}")
                last_result = OrderResult(
                    success=False, client_order_id=client_order_id, error=str(error),
                    details=result if isinstance(result, dict) else None, fill_status="FAILED",
                )
                continue  # no live order to cancel; advance the ladder

            fill_status = await self._wait_for_fill(
                client_order_id, timeout_seconds=per_attempt_timeout, poll_interval=poll_interval,
            )

            if fill_status == "FILLED":
                logger.info(
                    f"WEBULL EXIT FILLED (attempt {attempt + 1}): SELL {contracts}x "
                    f"{ticker} ${strike} {option_type} @ ${limit:.2f} — order_id={order_id}"
                )
                return OrderResult(
                    success=True, order_id=str(order_id), client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None, fill_status="FILLED",
                )

            if fill_status in ("PARTIAL_FILLED", "PARTIAL"):
                # Some contracts sold. STOP the ladder and return the partial — the monitor's
                # next cycle re-looks-up the (now smaller) position and sells the remainder.
                # Re-pricing the remainder here would need a fresh position_id and risks a
                # double-submit; the cross-cycle retry is the safe path for the tail.
                filled_qty = await self._get_filled_quantity(client_order_id)
                logger.warning(
                    f"WEBULL EXIT PARTIAL FILL (attempt {attempt + 1}): "
                    f"{filled_qty or '?'}/{contracts}x {ticker} ${strike} {option_type} "
                    f"@ ${limit:.2f} — cancelling remainder, monitor will finish next cycle"
                )
                await self.cancel_order(client_order_id)
                return OrderResult(
                    success=True, order_id=str(order_id), client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="PARTIAL", filled_quantity=filled_qty,
                )

            last_result = OrderResult(
                success=False, order_id=str(order_id), client_order_id=client_order_id,
                error=f"Exit not filled after {per_attempt_timeout:.0f}s (status={fill_status})",
                fill_status=fill_status or "SUBMITTED",
            )

            # DOUBLE-FILL GUARD (naked-short prevention): cancel-and-CONFIRM before any re-submit.
            confirm_status = await self._confirm_cancelled(client_order_id)
            if confirm_status in ("FILLED", "PARTIAL_FILLED", "PARTIAL"):
                # Filled in the race with our cancel — honor it and STOP. Never place another
                # sell on top of a live fill (that would oversell the position).
                filled_qty = (
                    None if confirm_status == "FILLED"
                    else await self._get_filled_quantity(client_order_id)
                )
                logger.warning(
                    f"WEBULL EXIT FILLED DURING CANCEL (attempt {attempt + 1}): "
                    f"{ticker} ${strike} {option_type} status={confirm_status} "
                    f"order_id={order_id} — honoring fill, halting chase"
                )
                return OrderResult(
                    success=True, order_id=str(order_id), client_order_id=client_order_id,
                    details=result if isinstance(result, dict) else None,
                    fill_status="FILLED" if confirm_status == "FILLED" else "PARTIAL",
                    filled_quantity=filled_qty,
                )

            if confirm_status not in ("CANCELLED", "REJECTED", "EXPIRED"):
                # Cannot confirm the prior sell is dead — refuse to re-submit (an uncancelled
                # working sell + a new one = oversell / naked short).
                logger.error(
                    f"WEBULL EXIT CHASE ABORTED: could not confirm cancel of "
                    f"order_id={order_id} (status={confirm_status}) — refusing to re-submit "
                    f"to avoid overselling"
                )
                return OrderResult(
                    success=False, order_id=str(order_id), client_order_id=client_order_id,
                    error=(
                        f"Exit not filled and prior order cancel unconfirmed "
                        f"(status={confirm_status}) — aborted chase to avoid overselling"
                    ),
                    fill_status=fill_status or "SUBMITTED",
                )

            logger.warning(
                f"WEBULL EXIT NOT FILLED (attempt {attempt + 1}/{max_attempts}): "
                f"{ticker} ${strike} {option_type} @ ${limit:.2f} — prior order "
                f"cancelled (confirmed), crossing harder"
            )

        if last_result is None:
            last_result = OrderResult(
                success=False, client_order_id=last_client_id,
                error="No exit attempts made", fill_status="FAILED",
            )
        logger.warning(
            f"WEBULL EXIT MISS: SELL {contracts}x {ticker} ${strike} {option_type} "
            f"— not filled after {max_attempts} chase attempts (monitor will retry next cycle)"
        )
        return last_result

    async def _fetch_ask(
        self, ticker: str, strike: float, expiry_date: str, option_type: str,
    ) -> float | None:
        """Freshest available ask for an option, used to chase the current ask.

        Prefers the harvester's live Redis snapshot (sub-second, off the API path) and
        falls back to the HTTP quote. Pricing the chase off a FRESH quote — rather than one
        that's stale by the time the order rests — is what keeps the limit marketable in a
        fast 0DTE tape (the 2026-06-25 not-filled fix). The Redis read is freshness-guarded
        so a stale snapshot can never make the limit worse than the HTTP path.
        """
        # 1) Harvester live Redis snapshot (preferred — fresh, no API round-trip)
        if getattr(self.settings, "WEBULL_ENTRY_USE_LIVE_QUOTE", True):
            try:
                import time

                from options_owl.db import redis_client

                max_age = float(getattr(self.settings, "WEBULL_ENTRY_QUOTE_MAX_AGE_SEC", 20.0) or 20.0)
                contract_key = f"{ticker.upper()}:{option_type.lower()}:{float(strike)}:{expiry_date}"
                snap = await redis_client.get_option_snapshot(contract_key)
                if snap:
                    ask = float(snap.get("ask") or 0)
                    ts = float(snap.get("t") or 0)
                    # Only trust the Redis ask when its age can be VERIFIED and is fresh.
                    # A missing/zero timestamp (ts <= 0) means we can't prove freshness, so
                    # fall through to the HTTP quote rather than pricing the chase off an
                    # ask of unknown age (the freshness guard must never be bypassable).
                    if ask > 0 and ts > 0 and (time.time() - ts) <= max_age:
                        return ask
            except Exception as exc:
                logger.debug(f"_fetch_ask redis miss for {ticker} ${strike} {option_type}: {exc}")

        # 2) HTTP quote fallback
        try:
            quote = await self.get_option_quote(ticker, strike, expiry_date, option_type)
        except Exception as exc:
            logger.debug(f"_fetch_ask failed for {ticker} ${strike} {option_type}: {exc}")
            return None
        if quote and quote.get("ask"):
            try:
                ask = float(quote["ask"])
                return ask if ask > 0 else None
            except (TypeError, ValueError):
                return None
        return None

    async def _fetch_bid(
        self, ticker: str, strike: float, expiry_date: str, option_type: str,
    ) -> float | None:
        """Freshest available BID for an option — the sell-side mirror of _fetch_ask.

        Sourced from Webull's OWN venue quote FIRST (the book we sell into — authoritative
        for what will actually fill), then the harvester's freshness-guarded Redis snapshot
        as a fast fallback. The venue-first ordering is deliberate: Polygon-fed quotes can be
        stale/wide for a thin contract (the MU #399 lesson — Polygon $12 vs Webull $5.25), so
        for EXITS we trust the venue, unlike the entry chase (_fetch_ask) which prefers Redis
        for speed. Prices each sell rung off the CURRENT bid so the limit stays marketable in
        a fast-diving 0DTE.
        """
        # 1) WEBULL venue quote — authoritative for what fills.
        try:
            quote = await self.get_option_quote(ticker, strike, expiry_date, option_type)
            if quote and quote.get("bid"):
                bid = float(quote["bid"])
                if bid > 0:
                    return bid
        except Exception as exc:
            logger.debug(f"_fetch_bid venue quote failed for {ticker} ${strike}: {exc}")

        # 2) Harvester live Redis snapshot (fast fallback, freshness-guarded).
        if getattr(self.settings, "WEBULL_ENTRY_USE_LIVE_QUOTE", True):
            try:
                import time

                from options_owl.db import redis_client

                max_age = float(getattr(self.settings, "WEBULL_ENTRY_QUOTE_MAX_AGE_SEC", 20.0) or 20.0)
                contract_key = f"{ticker.upper()}:{option_type.lower()}:{float(strike)}:{expiry_date}"
                snap = await redis_client.get_option_snapshot(contract_key)
                if snap:
                    bid = float(snap.get("bid") or 0)
                    ts = float(snap.get("t") or 0)
                    if bid > 0 and ts > 0 and (time.time() - ts) <= max_age:
                        return bid
            except Exception as exc:
                logger.debug(f"_fetch_bid redis miss for {ticker} ${strike} {option_type}: {exc}")
        return None

    async def _wait_for_fill(
        self,
        client_order_id: str,
        timeout_seconds: float = 15,
        poll_interval: float = 3.0,
    ) -> str:
        """Poll order status until filled, cancelled, or timeout.

        Returns the final status string: FILLED, PARTIAL_FILLED, SUBMITTED,
        CANCELLED, or UNKNOWN.
        """
        import time

        deadline = time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"

        while time.monotonic() < deadline:
            try:
                # BATCH first: the shared open-orders snapshot covers ALL pending orders in one call.
                snap = await self._open_order_status(client_order_id)
                if snap in ("PARTIAL_FILLED", "PARTIAL"):
                    return snap
                if snap and snap not in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
                    last_status = snap                  # still pending — no per-order call needed
                    await asyncio.sleep(poll_interval)
                    continue
                # Not in the open list (or terminal) -> confirm the final status with ONE detail call.
                detail = await self.get_order_status(client_order_id)
                if not detail:
                    await asyncio.sleep(poll_interval)
                    continue

                # Extract status from response — handle nested structures
                status = ""
                if isinstance(detail, dict):
                    # Try top-level status
                    status = detail.get("status", "")
                    # Try nested in orders list
                    if not status:
                        orders = detail.get("orders", [])
                        if orders and isinstance(orders, list):
                            status = orders[0].get("status", "")
                    # Check filled_quantity
                    filled = float(detail.get("filled_quantity", 0) or 0)
                    total = float(detail.get("total_quantity", 0) or 0)
                    if not status and filled > 0:
                        status = "FILLED" if filled >= total else "PARTIAL_FILLED"

                status = status.upper()
                last_status = status or last_status

                if status in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
                    return status
                if status in ("PARTIAL_FILLED", "PARTIAL"):
                    return status

                logger.debug(f"Order {client_order_id}: status={status}, waiting...")
            except Exception as exc:
                logger.debug(f"Fill poll error: {exc}")

            await asyncio.sleep(poll_interval)

        return last_status

    async def get_fill_price(self, client_order_id: str, retries: int = 3) -> float | None:
        """Get the avg filled price for a completed order.

        Returns the actual fill price from Webull, or None if unavailable.
        Retries with backoff to handle 429 rate limits after order placement.
        """
        for attempt in range(retries):
            if attempt > 0:
                await asyncio.sleep(2 * attempt)  # 2s, 4s backoff

            detail = await self.get_order_status(client_order_id)
            if not detail or not isinstance(detail, dict):
                logger.debug(
                    f"get_fill_price: attempt {attempt + 1}/{retries} — "
                    f"no detail for {client_order_id}: {detail}"
                )
                continue

            price = self._extract_fill_price(detail)
            if price is not None:
                # A1b (2026-07-02): surface the payload shape + any commission/fee field so we can
                # wire real-fee subtraction against the VERIFIED structure (never guess the field
                # name — a wrong guess would over-subtract and create a NEW inaccuracy). Pure
                # diagnostic logging; remove once the fee field is confirmed from a live close.
                try:
                    fee_like = {}
                    _scan = [detail, *detail.get("orders", []), *detail.get("legs", [])]
                    for _d in _scan:
                        if isinstance(_d, dict):
                            for _k, _v in _d.items():
                                if isinstance(_k, str) and ("fee" in _k.lower() or "comm" in _k.lower()):
                                    fee_like[_k] = _v
                    logger.info(
                        "get_fill_price: %s → $%.2f | keys=%s | fee-like=%s",
                        client_order_id, price, list(detail.keys())[:20], fee_like,
                    )
                except Exception:
                    logger.debug(f"get_fill_price: {client_order_id} → ${price:.2f}")
                return price

            # Log the full response so we can see what format Webull returns
            logger.warning(
                f"get_fill_price: could not extract price from response "
                f"(attempt {attempt + 1}/{retries}), keys={list(detail.keys())}, "
                f"response={str(detail)[:500]}"
            )

        logger.warning(f"get_fill_price: gave up after {retries} attempts for {client_order_id}")
        return None

    @staticmethod
    def _extract_fill_price(detail: dict) -> float | None:
        """Extract avg fill price from a Webull order detail response."""
        # Try top-level avg_filled_price (camelCase and snake_case)
        for key in ("avg_filled_price", "avgFilledPrice", "filled_price", "filledPrice"):
            price = detail.get(key)
            if price:
                return float(price)

        # Try nested in orders list
        orders = detail.get("orders", detail.get("order_list", []))
        if orders and isinstance(orders, list):
            for order in orders:
                for key in ("avg_filled_price", "avgFilledPrice", "filled_price"):
                    price = order.get(key)
                    if price:
                        return float(price)
                # Check legs
                for leg in order.get("legs", order.get("option_legs", [])):
                    for key in ("avg_filled_price", "avgFilledPrice", "filled_price"):
                        price = leg.get(key)
                        if price:
                            return float(price)

        # Try direct legs at top level
        for leg in detail.get("legs", detail.get("option_legs", [])):
            for key in ("avg_filled_price", "avgFilledPrice", "filled_price"):
                price = leg.get(key)
                if price:
                    return float(price)

        return None

    async def get_account_balance(self) -> float:
        """Get the current total account value (net liquidation value).

        Returns the real Webull account balance for position sizing.
        Uses a 60s TTL cache to avoid hammering the API on every signal.
        """
        import time

        # Return cached value if fresh
        if self._balance_cache is not None:
            cached_val, cached_ts = self._balance_cache
            if time.time() - cached_ts < self._balance_cache_ttl:
                return cached_val

        self._ensure_clients()

        for attempt in range(2):
            try:
                balance_resp = await asyncio.to_thread(
                    self._trade_client.account_v2.get_account_balance,
                    self._account_id,
                )
                balance = balance_resp.json() if hasattr(balance_resp, 'json') else balance_resp
                if isinstance(balance, dict):
                    # Try net liquidation value first (most accurate for sizing)
                    nlv = balance.get(
                        "total_net_liquidation_value",
                        balance.get("total_asset", balance.get("totalAsset", 0)),
                    )
                    result = float(nlv)
                    self._balance_cache = (result, time.time())
                    return result
            except (ValueError, ConnectionError, OSError) as exc:
                if attempt == 0 and "connection" in str(exc).lower():
                    logger.warning(f"Webull stale connection in get_account_balance ({exc}) — reconnecting")
                    # Run blocking reconnect off the event loop (see get_account_info).
                    await asyncio.to_thread(self._reconnect)
                    continue
                logger.warning(f"Failed to get account balance: {exc}")
            except Exception as exc:
                logger.warning(f"Failed to get account balance: {exc}")
                break

        return 0.0

    async def get_open_option_positions(self) -> list[dict]:
        """Return all open option positions from Webull.

        Each position dict has: ticker, strike, expiry_date, option_type, quantity.
        Used for reconciliation against the paper DB.
        """
        import time
        if time.monotonic() < self._os_cooldown_until:   # respect the shared 429 cooldown
            logger.debug("get_open_option_positions: in Webull 429 cooldown — skipping this cycle")
            return []
        self._ensure_clients()
        try:
            response = await asyncio.to_thread(
                self._trade_client.account_v2.get_account_position,
                self._account_id,
            )
            if hasattr(response, "json"):
                try:
                    positions = response.json()
                except Exception:
                    positions = response
            else:
                positions = response

            # Log raw response to diagnose format
            logger.debug(
                f"get_open_option_positions: raw type={type(positions).__name__}, "
                f"keys={list(positions.keys()) if isinstance(positions, dict) else 'N/A'}, "
                f"preview={str(positions)[:500]}"
            )

            # Unwrap nested response — try multiple known wrapper keys
            if isinstance(positions, dict):
                for key in ("positions", "holdings", "data", "option_positions"):
                    if key in positions:
                        positions = positions[key]
                        break
                else:
                    if "ticker" in positions or "symbol" in positions:
                        positions = [positions]
                    else:
                        positions = []
            if not isinstance(positions, list):
                logger.warning(f"get_open_option_positions: unexpected type {type(positions)}")
                return []

            results = []
            for pos in positions:
                qty = int(float(pos.get("quantity", pos.get("position", 0))))
                if qty <= 0:
                    continue

                # Webull returns option details in nested 'legs' array
                legs = pos.get("legs", [])
                leg = legs[0] if legs else {}

                ticker = (
                    pos.get("ticker")
                    or pos.get("symbol")
                    or leg.get("symbol")
                    or ""
                ).upper()
                option_type = (
                    pos.get("option_type")
                    or leg.get("option_type")
                    or ""
                ).upper()
                expiry = (
                    pos.get("option_expire_date")
                    or pos.get("expiry_date")
                    or leg.get("option_expire_date")
                    or ""
                )
                strike = float(
                    pos.get("option_exercise_price")
                    or pos.get("strike_price")
                    or pos.get("strike")
                    or leg.get("option_exercise_price")
                    or leg.get("strike_price")
                    or 0
                )

                if ticker and option_type and expiry:
                    results.append({
                        "ticker": ticker,
                        "strike": strike,
                        "expiry_date": expiry,
                        "option_type": option_type.lower(),
                        "quantity": qty,
                        "cost_price": float(pos.get("cost_price", 0)),
                        "last_price": float(pos.get("last_price", leg.get("last_price", 0))),
                        "unrealized_pnl": float(pos.get("unrealized_profit_loss", 0)),
                        "position_id": pos.get("position_id", ""),
                    })
                else:
                    logger.warning(
                        f"get_open_option_positions: skipping position with "
                        f"missing fields: ticker={ticker} type={option_type} "
                        f"expiry={expiry} raw_keys={list(pos.keys())}"
                    )
            return results
        except Exception as exc:
            if _is_webull_rate_limit(exc):
                import time
                self._os_cooldown_until = time.monotonic() + self._os_cooldown
                logger.warning(f"Webull 429 on positions — cooldown {self._os_cooldown:.0f}s")
            else:
                logger.warning(f"Failed to get open positions: {exc}")
            return []

    # ------------------------------------------------------------------
    # Market data — real-time option quotes from Webull
    # ------------------------------------------------------------------

    def _ensure_data_client(self) -> None:
        """Lazy-init the Webull DataClient for market data queries."""
        if self._data_client is not None:
            return

        self._ensure_clients()

        from webull.data.data_client import DataClient
        self._data_client = DataClient(self._api_client)
        logger.info("Webull DataClient initialized for market data")

    async def _lookup_instrument_id(
        self,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
    ) -> str | None:
        """Look up the Webull instrument_id for an option contract.

        Uses trade_instrument.get_trade_security_detail() and caches the result.
        """
        key = (ticker.upper(), strike, expiry_date, option_type.lower())
        if key in self._instrument_cache:
            return self._instrument_cache[key]

        self._ensure_clients()

        inst_type = "CALL_OPTION" if option_type.lower() == "call" else "PUT_OPTION"
        # Format strike cleanly: 675.0 → "675", 197.5 → "197.5"
        strike_str = f"{strike:g}"
        try:
            response = await asyncio.to_thread(
                self._trade_client.trade_instrument.get_trade_security_detail,
                ticker.upper(),
                "US",
                "OPTION",
                inst_type,
                strike_str,
                expiry_date,
            )
            result = response.json() if hasattr(response, "json") else response
            if isinstance(result, dict):
                inst_id = result.get("instrument_id", result.get("instrumentId", ""))
                if inst_id:
                    self._instrument_cache[key] = str(inst_id)
                    logger.debug(
                        f"Webull instrument_id for {ticker} ${strike} {option_type} "
                        f"exp={expiry_date}: {inst_id}"
                    )
                    return str(inst_id)
            logger.debug(
                f"Webull instrument lookup returned no ID for "
                f"{ticker} ${strike_str} {inst_type} exp={expiry_date}: {result}"
            )
        except Exception as exc:
            logger.debug(
                f"Webull instrument lookup failed for "
                f"{ticker} ${strike_str} {inst_type} exp={expiry_date}: {exc}"
            )

        return None

    async def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
    ) -> dict | None:
        """Fetch real-time bid/ask/mid for an option from Webull's market data.

        Returns dict with keys: bid, ask, mid, last, instrument_id
        or None if unavailable.

        This is the **same data source** as the execution venue, eliminating
        the Polygon/yfinance estimation gap that causes premature exits.
        """
        import time as _time
        from webull.data.common.category import Category

        inst_id = await self._lookup_instrument_id(
            ticker, strike, expiry_date, option_type,
        )
        if not inst_id:
            return None

        # Check quote cache
        cached = self._quote_cache.get(inst_id)
        if cached:
            bid, ask, mid, ts = cached
            if _time.time() - ts < self._quote_cache_ttl:
                return {"bid": bid, "ask": ask, "mid": mid, "instrument_id": inst_id}

        self._ensure_data_client()

        try:
            response = await asyncio.to_thread(
                self._data_client.market_data.get_snapshot,
                inst_id,
                Category.US_OPTION,
            )
            result = response.json() if hasattr(response, "json") else response
            logger.debug(f"Webull option snapshot for {inst_id}: {str(result)[:500]}")

            # Parse the snapshot response — extract bid/ask/last
            quote = self._parse_option_snapshot(result)
            if quote:
                self._quote_cache[inst_id] = (
                    quote["bid"], quote["ask"], quote["mid"], _time.time(),
                )
                quote["instrument_id"] = inst_id
                return quote

        except Exception as exc:
            logger.debug(f"Webull option snapshot failed for {inst_id}: {exc}")

        # Fallback: try get_quotes (depth quotes)
        try:
            response = await asyncio.to_thread(
                self._data_client.market_data.get_quotes,
                inst_id,
                Category.US_OPTION,
            )
            result = response.json() if hasattr(response, "json") else response
            logger.debug(f"Webull option quotes for {inst_id}: {str(result)[:500]}")

            quote = self._parse_option_quotes(result)
            if quote:
                self._quote_cache[inst_id] = (
                    quote["bid"], quote["ask"], quote["mid"], _time.time(),
                )
                quote["instrument_id"] = inst_id
                return quote

        except Exception as exc:
            logger.debug(f"Webull option quotes failed for {inst_id}: {exc}")

        return None

    @staticmethod
    def _parse_option_snapshot(data: dict | list) -> dict | None:
        """Extract bid/ask/mid from a Webull snapshot response."""
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None

        # Webull snapshot fields vary; try common patterns
        bid = float(data.get("bid", data.get("bidPrice", data.get("bid_price", 0))) or 0)
        ask = float(data.get("ask", data.get("askPrice", data.get("ask_price", 0))) or 0)
        last = float(data.get("last", data.get("lastPrice", data.get("last_price",
                     data.get("price", data.get("close", 0))))) or 0)

        # Try nested quote structure
        if bid <= 0 or ask <= 0:
            quote_data = data.get("quote", data.get("snapshot"))
            if isinstance(quote_data, dict):
                nested_bid = float(quote_data.get("bid", quote_data.get("bidPrice", 0)) or 0)
                nested_ask = float(quote_data.get("ask", quote_data.get("askPrice", 0)) or 0)
                nested_last = float(quote_data.get("last", quote_data.get("lastPrice",
                                    quote_data.get("close", 0))) or 0)
                if nested_bid > 0:
                    bid = nested_bid
                if nested_ask > 0:
                    ask = nested_ask
                if nested_last > 0:
                    last = nested_last

        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2.0, 2)
            return {"bid": round(bid, 2), "ask": round(ask, 2), "mid": mid, "last": round(last, 2)}

        if last > 0:
            return {"bid": 0.0, "ask": 0.0, "mid": round(last, 2), "last": round(last, 2)}

        return None

    @staticmethod
    def _parse_option_quotes(data: dict | list) -> dict | None:
        """Extract bid/ask from a Webull depth quotes response."""
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None

        # Depth quotes typically have askList/bidList arrays
        bid_list = data.get("bidList", data.get("bid_list", data.get("bids", [])))
        ask_list = data.get("askList", data.get("ask_list", data.get("asks", [])))

        bid = 0.0
        ask = 0.0

        if bid_list and isinstance(bid_list, list):
            bid = float(bid_list[0].get("price", 0) or 0)
        if ask_list and isinstance(ask_list, list):
            ask = float(ask_list[0].get("price", 0) or 0)

        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2.0, 2)
            return {"bid": round(bid, 2), "ask": round(ask, 2), "mid": mid}

        # Try flat structure
        bid = float(data.get("bid", data.get("bidPrice", 0)) or 0)
        ask = float(data.get("ask", data.get("askPrice", 0)) or 0)
        last = float(data.get("last", data.get("lastPrice", data.get("close", 0))) or 0)

        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2.0, 2)
            return {"bid": round(bid, 2), "ask": round(ask, 2), "mid": mid}

        if last > 0:
            return {"bid": 0.0, "ask": 0.0, "mid": round(last, 2)}

        return None

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an open order by client_order_id."""
        self._ensure_clients()

        logger.info(f"WEBULL CANCEL: client_id={client_order_id}")

        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.cancel_option,
                self._account_id,
                client_order_id,
            )
            result = response.json() if hasattr(response, 'json') else response
            logger.info(f"WEBULL CANCEL result: {result}")
            return True
        except Exception as exc:
            logger.error(f"WEBULL CANCEL ERROR: {exc}")
            return False

    async def get_order_status(self, client_order_id: str) -> dict | None:
        """Get the current status of an order — THROTTLED to avoid Webull 429s.

        Per-order cache (TTL), a shared min-interval between API calls, and a cooldown after a 429
        (serve the last cached value instead of hammering the throttled endpoint). Order placement
        still works; we just check status less aggressively.
        """
        import time
        now = time.monotonic()
        cached = self._os_cache.get(client_order_id)
        if cached and (now - cached[1]) < self._os_cache_ttl:
            return cached[0]
        if now < self._os_cooldown_until:           # 429 cooldown — serve stale, don't call
            return cached[0] if cached else None
        if self._os_lock is None:
            self._os_lock = asyncio.Lock()
        async with self._os_lock:
            now = time.monotonic()
            cached = self._os_cache.get(client_order_id)
            if cached and (now - cached[1]) < self._os_cache_ttl:
                return cached[0]
            if now < self._os_cooldown_until:
                return cached[0] if cached else None
            wait = self._os_min_interval - (now - self._os_last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._os_last_call_ts = time.monotonic()
            self._ensure_clients()
            try:
                response = await asyncio.to_thread(
                    self._trade_client.order_v2.get_order_detail,
                    self._account_id, client_order_id,
                )
                result = response.json() if hasattr(response, 'json') else response
                self._os_cache[client_order_id] = (result, time.monotonic())
                return result
            except Exception as exc:
                if _is_webull_rate_limit(exc):
                    self._os_cooldown_until = time.monotonic() + self._os_cooldown
                    logger.warning(f"Webull 429 on order status — cooldown {self._os_cooldown:.0f}s "
                                   f"(serving cached status)")
                else:
                    logger.warning(f"Failed to get order status for {client_order_id}: {exc}")
                return cached[0] if cached else None

    async def _open_order_status(self, client_order_id: str) -> str | None:
        """BATCH: status via a shared, briefly-cached get_open_orders() snapshot — ONE call covers all
        pending orders. Returns the status string if the order is still OPEN; None if it's not in the
        open list (completed/unknown — caller confirms with a single get_order_status). Fails open."""
        import time
        now = time.monotonic()
        if (now - self._oo_snapshot_ts) >= self._oo_snapshot_ttl and now >= self._os_cooldown_until:
            if self._os_lock is None:
                self._os_lock = asyncio.Lock()
            async with self._os_lock:
                if (time.monotonic() - self._oo_snapshot_ts) >= self._oo_snapshot_ttl:
                    wait = self._os_min_interval - (time.monotonic() - self._os_last_call_ts)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._os_last_call_ts = time.monotonic()
                    try:
                        orders = await self.get_open_orders()
                        snap: dict[str, str] = {}
                        for o in (orders or []):
                            if not isinstance(o, dict):
                                continue
                            coid = o.get("client_order_id") or o.get("clientOrderId")
                            st = (o.get("status") or "").upper()
                            if not st:
                                sub = o.get("orders")
                                if isinstance(sub, list) and sub:
                                    st = (sub[0].get("status") or "").upper()
                            if coid:
                                snap[coid] = st
                        self._oo_snapshot = snap
                        self._oo_snapshot_ts = time.monotonic()
                    except Exception as exc:
                        if _is_webull_rate_limit(exc):
                            self._os_cooldown_until = time.monotonic() + self._os_cooldown
        return self._oo_snapshot.get(client_order_id)

    async def get_open_orders(self) -> list[dict]:
        """Get all open/pending orders."""
        self._ensure_clients()

        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.get_order_open,
                self._account_id,
                100,  # page_size
            )
            result = response.json() if hasattr(response, 'json') else response
            if isinstance(result, dict):
                return result.get("orders", result.get("data", []))
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning(f"Failed to get open orders: {exc}")
            return []

    async def get_order_history(
        self, start_date: str, end_date: str, page_size: int = 100,
    ) -> list[dict]:
        """Get filled order history from Webull for a date range.

        Args:
            start_date: yyyy-MM-dd format
            end_date: yyyy-MM-dd format
            page_size: max orders per page (default 100)

        Returns list of order dicts with fill prices, quantities, etc.
        """
        self._ensure_clients()

        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.get_order_history,
                self._account_id,
                page_size,
                start_date,
                end_date,
            )
            result = response.json() if hasattr(response, 'json') else response
            if isinstance(result, dict):
                return result.get("orders", result.get("data", []))
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning(f"Failed to get order history: {exc}")
            return []

    # ------------------------------------------------------------------
    # Dry run: preview order without placing
    # ------------------------------------------------------------------

    async def preview_option_order(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        side: str,
        contracts: int,
        limit_price: float,
    ) -> dict | None:
        """Preview an order (cost + fees) without placing it. Works even with PAPER_TRADE=true."""
        self._ensure_clients()

        preview_payload = [{
            "client_order_id": f"preview_{uuid.uuid4().hex[:16]}",
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "quantity": str(contracts),
            "limit_price": f"{limit_price:.2f}",
            "option_strategy": "SINGLE",
            "side": side.upper(),
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "legs": [{
                "side": side.upper(),
                "quantity": str(contracts),
                "symbol": ticker.upper(),
                "strike_price": str(strike),
                "option_expire_date": expiry_date,
                "instrument_type": "OPTION",
                "option_type": option_type.upper(),
                "market": "US",
            }],
        }]

        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.preview_option,
                self._account_id,
                preview_payload,
            )
            result = response.json() if hasattr(response, 'json') else response
            logger.info(
                f"WEBULL PREVIEW: {side} {contracts}x {ticker} ${strike} {option_type} "
                f"@ ${limit_price:.2f} → {result}"
            )
            return result if isinstance(result, dict) else {"raw": result}
        except Exception as exc:
            logger.warning(f"Preview failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Convenience: buy/sell options
    # ------------------------------------------------------------------

    async def buy_option(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        contracts: int,
        limit_price: float,
    ) -> OrderResult:
        """Buy to open an option position."""
        return await self.place_option_order(
            ticker=ticker,
            strike=strike,
            expiry_date=expiry_date,
            option_type=option_type,
            side="BUY",
            contracts=contracts,
            limit_price=limit_price,
        )

    async def sell_option(
        self,
        *,
        ticker: str,
        strike: float,
        expiry_date: str,
        option_type: str,
        contracts: int,
        limit_price: float,
        has_webull_order_id: bool = False,
    ) -> OrderResult:
        """Sell to close an option position."""
        return await self.place_option_order(
            ticker=ticker,
            strike=strike,
            expiry_date=expiry_date,
            option_type=option_type,
            side="SELL",
            contracts=contracts,
            limit_price=limit_price,
            has_webull_order_id=has_webull_order_id,
        )
