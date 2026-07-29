"""Broker-side stop-loss manager (design A: resting stop = pure backstop; monitor owns all exits).

A resting Webull STOP_LOSS fires the microsecond premium hits the level — the give-back our 5s poll can't
catch (measured avg 19pt, tail +17%→-31%). This manager owns PLACEMENT + TRACKING + CLEANUP of those resting
stops; the double-fill guard (cancel-before-sell) lives at the sell chokepoint in
``paper_trader.close_webull_position``; orphan reconciliation on restart is layered on top.

ROBUSTNESS CONTRACT (the whole point — it sits over the live sell path):
  * **Retry, bounded.** Placement is retried across monitor cycles up to ``BROKER_STOP_MAX_ATTEMPTS``.
  * **Fall back to the old way.** ANY failure (disabled, no executor, timeout, API error, exhausted retries)
    leaves the trade with NO resting stop — the existing poll-based FSM fully protects it. The broker stop is
    strictly ADDITIVE; it can never block or replace the battle-tested exit path.
  * **Clean up after itself.** A closed / vanished trade's resting stop is cancelled (best-effort) and its
    tracking dropped, so nothing is orphaned at the venue or leaked in memory.
  * **Never raises into the monitor loop.** Every public coroutine swallows and logs its own errors.

See specs/active/2026-07-22_broker-side-stop-loss.md.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger


# ---------------------------------------------------------------------------
# Circuit breaker (2026-07-23): a resting stop reserves the position's holding quantity, which can BLOCK
# the FSM's own exit (OPTION_LONG_POSITION_MUST_BE_CLOSE_THAN_SELL_SHORT). On 2026-07-23 that produced a
# 5,000+ blocked-sell storm on live money. This process-wide fuse trips on the FIRST such blocked-sell:
# broker stops disable themselves everywhere (no new placements) so the FSM reverts to clean poll-only,
# capping the blast radius at 1 event instead of thousands. Reset only on process restart.
# ---------------------------------------------------------------------------
_KILLED = False


def kill_broker_stops(reason: str) -> None:
    """Trip the fuse: disable all broker-stop placement process-wide. Idempotent, safe to call repeatedly."""
    global _KILLED
    if not _KILLED:
        _KILLED = True
        logger.critical(
            f"BROKER STOP CIRCUIT BREAKER TRIPPED — disabling broker stops process-wide. Reason: {reason}. "
            f"FSM poll-only exits still fully protect all positions."
        )


def broker_stops_killed() -> bool:
    return _KILLED


def reset_broker_stops_kill() -> None:
    """Test-only: clear the fuse."""
    global _KILLED
    _KILLED = False


# ---------------------------------------------------------------------------
# Tracked-stop registry (2026-07-27): the sell path must cancel the resting stop BEFORE it sells, or the
# stop's reserved quantity blocks the FSM's own exit (MUST_BE_CLOSE_THAN_SELL_SHORT → fuse trip). The old
# sell-path guard re-derived the stop's leg fields from get_open_orders() and FUZZY-matched — which silently
# missed (strike/expiry/type format drift) on GOOG #633/#378, so both live bots blocked-then-fuse-tripped.
# This module-level registry maps trade_id → the resting stop's client_order_id, so the sell chokepoint in
# paper_trader can cancel-and-CONFIRM the EXACT order by id (no matching, no miss). Mirrors the manager's
# per-instance ``_state`` but is reachable from paper_trader (which holds no manager ref), like the fuse above.
# A stale entry is harmless: _confirm_cancelled on an already-terminal order just returns its status.
# ---------------------------------------------------------------------------
_ACTIVE_STOP_CLIENT_IDS: dict[int, str] = {}


def get_active_stop_client_id(trade_id: int) -> str | None:
    """The client_order_id of the resting stop tracked for this trade, if any (for the sell-path cancel)."""
    return _ACTIVE_STOP_CLIENT_IDS.get(trade_id)


def _register_stop(trade_id: int, client_order_id: str | None) -> None:
    if client_order_id:
        _ACTIVE_STOP_CLIENT_IDS[trade_id] = client_order_id


def _forget_stop(trade_id: int) -> None:
    _ACTIVE_STOP_CLIENT_IDS.pop(trade_id, None)


class StopStatus(str, Enum):
    NONE = "none"        # no attempt yet
    PLACED = "placed"    # resting at the venue (client_order_id set)
    FAILED = "failed"    # placement failed; retry until attempts exhausted → poll-only fallback


@dataclass
class StopState:
    status: StopStatus = StopStatus.NONE
    client_order_id: str | None = None
    order_id: str | None = None
    stop_price: float | None = None
    attempts: int = 0
    peak_premium: float | None = None   # highest premium seen (for the trailing ratchet)
    last_replace_ts: float = 0.0        # monotonic ts of the last place/replace (churn guard)


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_stop_price(entry_premium: float, settings) -> float | None:
    """The BASE resting-stop price for a trade: entry premium × ``BROKER_STOP_ENTRY_FRAC`` (0.75 = -25%).

    Returns ``None`` for a non-positive entry premium (nothing sensible to protect) so the caller skips.
    """
    entry = _f(entry_premium)
    if entry is None or entry <= 0:
        return None
    frac = _f(getattr(settings, "BROKER_STOP_ENTRY_FRAC", 0.75), 0.75)
    price = entry * frac
    return price if price > 0 else None


def compute_desired_stop(entry_premium: float, peak_premium: float | None, settings) -> float | None:
    """The stop price the trailing ratchet WANTS right now, given the entry + peak premium seen.

    - base = entry × BROKER_STOP_ENTRY_FRAC (the -25% floor).
    - once peak gain >= BROKER_STOP_RATCHET_ARM_PCT: ratchet up to max(base, breakeven, peak × KEEP_FRAC).
      Breakeven (= entry) is the floor once armed, so an armed runner can never round-trip to red.
    Monotonic non-decreasing in peak. Returns None on a bad entry.
    """
    base = compute_stop_price(entry_premium, settings)
    if base is None:
        return None
    entry = _f(entry_premium)
    peak = _f(peak_premium)
    if peak is None or entry is None or entry <= 0:
        return base
    arm_pct = _f(getattr(settings, "BROKER_STOP_RATCHET_ARM_PCT", 20.0), 20.0)
    peak_gain_pct = (peak / entry - 1.0) * 100.0
    if peak_gain_pct < arm_pct:
        return base
    keep = _f(getattr(settings, "BROKER_STOP_TRAIL_KEEP_FRAC", 0.75), 0.75)
    trail = peak * keep
    return max(base, entry, trail)  # never below breakeven once armed


class BrokerStopManager:
    """Places + tracks + cleans up resting broker stops. Holds no lock on the sell path — pure additive."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._state: dict[int, StopState] = {}

    @property
    def enabled(self) -> bool:
        # Respects the process-wide circuit breaker — once tripped, no more placements anywhere.
        return getattr(self.settings, "ENABLE_BROKER_STOP", False) is True and not broker_stops_killed()

    def _max_attempts(self) -> int:
        try:
            return int(getattr(self.settings, "BROKER_STOP_MAX_ATTEMPTS", 3))
        except (TypeError, ValueError):
            return 3

    def active_stops(self) -> dict[int, str]:
        """trade_id → client_order_id for every currently-PLACED stop (for reconcile / diagnostics)."""
        return {
            tid: st.client_order_id
            for tid, st in self._state.items()
            if st.status is StopStatus.PLACED and st.client_order_id
        }

    async def ensure_stop(self, trade: dict, executor, current_premium: float | None = None) -> None:
        """Ensure a resting stop exists for a live trade AND ratchet it up as the contract runs.

        Idempotent, bounded-retry, rate-limited, never raises. No-op when disabled / paper / no executor /
        the trade is paper-only (no ``webull_order_id``) / retries are exhausted. ``current_premium`` (the
        monitor's fresh exit premium) drives the trailing ratchet; omit it and only the base stop is placed.
        On any failure the trade simply has no/stale broker stop and the FSM protects it (fallback).
        """
        try:
            # Fuse tripped → cancel EVERY resting stop we still track (free all reserved quantity so the
            # FSM is never blocked), then stay off. Runs once; after cleanup there's nothing left to cancel.
            if broker_stops_killed():
                if executor is not None and self.active_stops():
                    await self.cancel_all(executor)
                return
            if not self.enabled or executor is None:
                return
            if getattr(self.settings, "PAPER_TRADE", False) is True:
                return
            if not trade.get("webull_order_id"):
                return  # paper-only leg — no live position to protect
            trade_id = trade["id"]
            st = self._state.setdefault(trade_id, StopState())
            if st.status is StopStatus.FAILED and st.attempts >= self._max_attempts():
                return  # gave up → poll-only fallback (logged once when it exhausted)

            entry = trade.get("premium_per_contract", 0.0)
            # Track the running peak (drives the trailing ratchet). Never let noise lower it.
            cur = _f(current_premium)
            if cur is not None and cur > 0:
                st.peak_premium = cur if st.peak_premium is None else max(st.peak_premium, cur)

            desired = compute_desired_stop(entry, st.peak_premium, self.settings)
            if desired is None:
                return  # no sensible entry premium — skip silently, FSM protects

            if st.status is StopStatus.PLACED:
                await self._maybe_replace(trade, executor, st, desired)
                return

            # First placement (or after a failed attempt): place at the desired stop.
            await self._place(trade, executor, st, desired)
        except Exception as exc:  # noqa: BLE001 - absolute backstop: the monitor loop must not die here
            logger.warning(f"BrokerStopManager.ensure_stop swallowed error: {exc}")

    async def _place(self, trade: dict, executor, st: StopState, stop_price: float) -> None:
        """Place a fresh resting stop and record it (or mark failed for bounded retry)."""
        trade_id = trade["id"]
        st.attempts += 1
        try:
            result = await asyncio.wait_for(
                executor.place_stop_loss(
                    ticker=trade["ticker"],
                    strike=trade["strike"],
                    expiry_date=trade.get("expiry_date") or "",
                    option_type=str(trade["option_type"]).upper(),
                    contracts=trade["contracts"],
                    stop_price=stop_price,
                ),
                timeout=15,
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 - must never raise into loop
            self._mark_failed(trade_id, st, f"exception: {exc}")
            return
        if getattr(result, "success", False) is True:
            st.status = StopStatus.PLACED
            st.client_order_id = getattr(result, "client_order_id", None)
            st.order_id = getattr(result, "order_id", None)
            st.stop_price = stop_price
            st.last_replace_ts = time.monotonic()
            _register_stop(trade_id, st.client_order_id)  # sell path cancels by this id (no fuzzy match)
            logger.info(
                f"BROKER STOP tracked: trade#{trade_id} {trade['ticker']} "
                f"stop=${stop_price:.2f} client_id={st.client_order_id}"
            )
        else:
            self._mark_failed(
                trade_id, st,
                f"{getattr(result, 'fill_status', '?')}: {getattr(result, 'error', '?')}",
            )

    async def _maybe_replace(self, trade: dict, executor, st: StopState, desired: float) -> None:
        """Ratchet the resting stop UP to ``desired`` by MODIFYING it in place (Webull ``replace_option``).

        This is the 2026-07-23 fix. The old cancel-then-place approach raced Webull's async holding-quantity
        release and, on failure, left orphan stops that reserved the position and BLOCKED the FSM's own exits
        (the 5,000-event storm). Modifying in place keeps the order's single quantity reservation — there is
        NEVER a window with two orders, so it cannot double-reserve or orphan. On ANY modify failure we KEEP
        the existing stop unchanged (the stop just doesn't trail higher this cycle) — we never cancel+replace.
        Rate-limited by min-step + min-interval so we don't churn the modify endpoint.
        """
        trade_id = trade["id"]
        cur_stop = st.stop_price or 0.0
        if desired <= cur_stop:
            return  # ratchet only moves UP
        entry = _f(trade.get("premium_per_contract", 0.0)) or 0.0
        min_step = (_f(getattr(self.settings, "BROKER_STOP_MIN_STEP_FRAC", 0.05), 0.05)) * entry
        if (desired - cur_stop) < min_step:
            return  # too small a move — don't churn orders
        min_interval = _f(getattr(self.settings, "BROKER_STOP_MIN_REPLACE_SEC", 30.0), 30.0)
        if (time.monotonic() - st.last_replace_ts) < min_interval:
            return  # replaced too recently — churn guard
        if not st.client_order_id:
            return  # nothing to modify (shouldn't happen for a PLACED stop)

        try:
            result = await asyncio.wait_for(
                executor.replace_stop_loss(
                    client_order_id=st.client_order_id,
                    ticker=trade["ticker"], strike=trade["strike"],
                    expiry_date=trade.get("expiry_date") or "",
                    option_type=str(trade["option_type"]).upper(),
                    contracts=trade["contracts"], stop_price=desired,
                ),
                timeout=15,
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 - keep existing stop, never raise
            logger.warning(
                f"BROKER STOP ratchet: modify failed for trade#{trade_id} ({exc}) — keeping stop at "
                f"${cur_stop:.2f}"
            )
            return
        if getattr(result, "success", False) is True:
            st.stop_price = desired
            st.last_replace_ts = time.monotonic()
            logger.info(
                f"BROKER STOP ratchet: trade#{trade_id} {trade['ticker']} "
                f"${cur_stop:.2f} → ${desired:.2f} (peak=${st.peak_premium:.2f}) [modify-in-place]"
            )
        else:
            # Modify failed/rejected → the OLD stop is still resting and valid. Just don't trail this cycle.
            logger.info(
                f"BROKER STOP ratchet: modify rejected for trade#{trade_id} "
                f"({getattr(result, 'error', '?')}) — keeping stop at ${cur_stop:.2f}"
            )
            st.last_replace_ts = time.monotonic()  # rate-limit retry of a failing modify

    def _mark_failed(self, trade_id: int, st: StopState, why: str) -> None:
        st.status = StopStatus.FAILED
        st.client_order_id = None
        _forget_stop(trade_id)
        # A placement rejected for MUST_BE_CLOSE_THAN_SELL_SHORT means a stop already reserves this
        # position's quantity (a leftover/orphan) — the exact condition that can block the FSM's exit.
        # Trip the circuit breaker so broker stops disable + clean up process-wide (belt-and-suspenders
        # with the sell-path fuse).
        if "MUST_BE_CLOSE_THAN_SELL_SHORT" in str(why):
            kill_broker_stops(f"stop placement blocked by reserved qty on trade#{trade_id}")
        if st.attempts >= self._max_attempts():
            logger.warning(
                f"BROKER STOP giving up: trade#{trade_id} after {st.attempts} attempt(s) "
                f"({why}) — falling back to poll-only exit (FSM still protects)"
            )
        else:
            logger.info(
                f"BROKER STOP place failed: trade#{trade_id} attempt {st.attempts} ({why}) — will retry"
            )

    async def release(self, trade_id: int, executor) -> None:
        """A trade closed — cancel its resting stop (best-effort) and drop tracking. Never raises.

        If the stop already filled (it WAS the exit) or was cancelled by the sell-path guard, ``cancel_order``
        simply returns False harmlessly. Either way the tracking is dropped so nothing leaks.
        """
        try:
            st = self._state.pop(trade_id, None)
            _forget_stop(trade_id)
            if st is None or st.status is not StopStatus.PLACED or not st.client_order_id:
                return
            if executor is None:
                return
            try:
                await asyncio.wait_for(executor.cancel_order(st.client_order_id), timeout=15)
                logger.info(f"BROKER STOP released: trade#{trade_id} client_id={st.client_order_id}")
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    f"BROKER STOP release cancel failed: trade#{trade_id} "
                    f"client_id={st.client_order_id} ({exc}) — reconcile sweep will catch any orphan"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"BrokerStopManager.release swallowed error: {exc}")

    async def release_and_confirm(self, trade_id: int, executor) -> bool:
        """Cancel this trade's resting stop by its TRACKED client_id AND CONFIRM it is terminal (the venue has
        provably freed the reserved quantity) BEFORE the FSM sell — the fix for the reserved-qty collision.

        Unlike ``release`` (fire-and-forget cancel), this polls the order to a terminal state via
        ``_confirm_cancelled`` so the subsequent sell-to-close cannot be blocked by MUST_BE_CLOSE_THAN_SELL_SHORT.
        Looks the stop up by tracked id — NO fuzzy leg-matching, which is what silently missed and let the fuse
        trip on GOOG #633/#378. Returns True if a tracked stop was cancelled+confirmed, False if none was tracked
        or the confirm failed (in which case the sell path's fuzzy sweep + the fuse still guard). Never raises.
        """
        try:
            st = self._state.pop(trade_id, None)
            _forget_stop(trade_id)
            if st is None or st.status is not StopStatus.PLACED or not st.client_order_id or executor is None:
                return False
            try:
                status = await asyncio.wait_for(
                    executor._confirm_cancelled(st.client_order_id, timeout_seconds=6.0), timeout=15,
                )
                logger.info(
                    f"BROKER STOP confirm-cancelled before sell: trade#{trade_id} "
                    f"client_id={st.client_order_id} status={status}"
                )
                return True
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    f"BROKER STOP release_and_confirm failed: trade#{trade_id} "
                    f"client_id={st.client_order_id} ({exc}) — sell-path fuzzy sweep + fuse still guard"
                )
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"BrokerStopManager.release_and_confirm swallowed error: {exc}")
            return False

    async def prune_closed(self, open_trade_ids: set[int], executor) -> None:
        """Release stops for any tracked trade no longer in the open set (self-cleanup each cycle)."""
        for tid in [t for t in self._state if t not in open_trade_ids]:
            await self.release(tid, executor)

    async def cancel_all(self, executor) -> None:
        """Cancel EVERY tracked resting stop (used when the circuit breaker trips). Best-effort, never raises."""
        for tid in list(self._state.keys()):
            await self.release(tid, executor)

    async def reconcile_orphans(self, open_trades: list[dict], executor) -> None:
        """Startup cleanup: cancel resting STOP_LOSS orders left at Webull by a previous run.

        On restart the in-memory tracking is empty, so any STOP_LOSS we placed before the restart is an
        untracked orphan resting at the venue. Query Webull's open orders, find STOP_LOSS orders on option
        contracts, and CANCEL every one whose contract is not a currently-open trade (a stale stop on a
        closed position). Matches to open trades are left resting and RE-ADOPTED into tracking so the monitor
        won't place a duplicate. Best-effort, never raises — a failure just leaves the old behavior (the sell
        path's cancel-before-sell guard still catches a stale stop at exit time).
        """
        if not self.enabled or executor is None:
            return
        if getattr(self.settings, "PAPER_TRADE", False) is True:
            return
        try:
            open_orders = await asyncio.wait_for(executor.get_open_orders(), timeout=15)
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.warning(f"BROKER STOP reconcile: get_open_orders failed ({exc}) — skipping")
            return

        # Index open trades by contract signature for matching.
        def _sig(ticker, strike, exp, ot):
            return (str(ticker).upper(), str(strike), str(exp or ""), str(ot).upper())

        open_by_sig: dict[tuple, dict] = {}
        for t in open_trades:
            open_by_sig[_sig(t["ticker"], t["strike"], t.get("expiry_date"), t["option_type"])] = t

        adopted = orphaned = 0
        for order in open_orders or []:
            if str(order.get("order_type", "")).upper() != "STOP_LOSS":
                continue
            coid = order.get("client_order_id")
            if not coid:
                continue
            matched = None
            for leg in order.get("legs") or []:
                sig = _sig(
                    leg.get("symbol"), leg.get("strike_price"),
                    leg.get("option_expire_date"), leg.get("option_type"),
                )
                if sig in open_by_sig:
                    matched = open_by_sig[sig]
                    break
            if matched is not None:
                # Re-adopt: a valid stop for a still-open trade — track it, don't re-place.
                st = self._state.setdefault(matched["id"], StopState())
                st.status = StopStatus.PLACED
                st.client_order_id = coid
                st.order_id = order.get("order_id")
                st.stop_price = _f(order.get("stop_price"))
                st.last_replace_ts = time.monotonic()
                adopted += 1
            else:
                # Orphan: a resting stop on a contract we no longer hold — cancel it.
                try:
                    await asyncio.wait_for(executor.cancel_order(coid), timeout=15)
                    orphaned += 1
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    logger.warning(f"BROKER STOP reconcile: cancel orphan {coid} failed ({exc})")
        if adopted or orphaned:
            logger.info(
                f"BROKER STOP reconcile: re-adopted {adopted} stop(s), cancelled {orphaned} orphan(s)"
            )
