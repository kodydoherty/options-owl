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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _reconnect(self) -> None:
        """Force-reinitialize the Webull SDK clients (e.g., after stale connection)."""
        logger.warning("Webull reconnect: tearing down stale clients and reinitializing")
        self._api_client = None
        self._trade_client = None
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
                self._reconnect()
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

    def _check_kill_switch(self) -> None:
        """Raise if kill switch is active."""
        if getattr(self.settings, "WEBULL_KILL_SWITCH", False):
            raise RuntimeError("WEBULL_KILL_SWITCH is active — all orders blocked")

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
        self._check_kill_switch()
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

        order_payload = [{
            "client_order_id": client_order_id,
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

        try:
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
                    self._reconnect()
                    response = await asyncio.to_thread(
                        self._trade_client.order_v2.place_option,
                        self._account_id,
                        order_payload,
                    )
                else:
                    raise
            result = response.json() if hasattr(response, 'json') else response
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

            # Poll for fill status — options orders usually fill within seconds.
            # SELL orders use a shorter timeout (10s) because 0DTE premiums
            # crash fast — waiting 45s causes massive slippage on retries.
            # BUY orders use 10s — if not filled, the caller retries with
            # a fresh ask price (smart entry retry in paper_trader).
            sell_timeout = 10.0
            fill_status = await self._wait_for_fill(
                client_order_id, timeout_seconds=sell_timeout, poll_interval=2.0,
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
                # Determine how many contracts filled
                filled_qty = None
                try:
                    detail = await self.get_order_status(client_order_id)
                    if detail and isinstance(detail, dict):
                        filled_qty = int(float(detail.get("filled_quantity", 0) or 0))
                except Exception:
                    pass
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
                    error=f"Order not filled after {sell_timeout:.0f}s (status={fill_status}), cancelled",
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

    async def _wait_for_fill(
        self,
        client_order_id: str,
        timeout_seconds: float = 15,
        poll_interval: float = 1.5,
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
                    self._reconnect()
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
        """Get the current status of an order."""
        self._ensure_clients()

        try:
            response = await asyncio.to_thread(
                self._trade_client.order_v2.get_order_detail,
                self._account_id,
                client_order_id,
            )
            result = response.json() if hasattr(response, 'json') else response
            logger.debug(
                f"get_order_status({client_order_id}): "
                f"type={type(result).__name__}, "
                f"keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}, "
                f"preview={str(result)[:400]}"
            )
            return result
        except Exception as exc:
            logger.warning(f"Failed to get order status for {client_order_id}: {exc}")
            return None

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
